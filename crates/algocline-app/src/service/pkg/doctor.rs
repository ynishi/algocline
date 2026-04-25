//! `pkg_doctor` — read-only diagnosis for package state (Wave 2 of local-first DX).
//!
//! The actuator counterpart is [`super::repair`] (`pkg_repair`). `pkg_doctor`
//! classifies packages into five buckets without touching the filesystem:
//!
//! | Bucket              | Source-of-truth                                         | Condition                                          |
//! |---------------------|---------------------------------------------------------|----------------------------------------------------|
//! | `healthy`           | `installed.json` + `~/.algocline/packages/{name}`       | dest directory exists (resolved through symlinks)  |
//! | `installed_missing` | `installed.json`                                        | dest missing (non-symlink), `pkg_install` can heal |
//! | `symlink_dangling`  | `installed.json` (manifest-pass) + filesystem scan      | dest is a symlink whose target is missing          |
//! | `path_missing`      | `alc.toml` / `alc.local.toml`                           | declared `path = ...` does not exist               |
//! | `incomplete_pkg`    | `installed.json` + `{pkg_dir}/{name}/init.lua`          | init.lua requires sibling sub (`pkg.sub`) but      |
//! |                     |                                                         | `sub.lua` / `sub/init.lua` is missing              |
//!
//! Contract:
//! - **No side effects.** No `fs::write`, `fs::remove_*`, `fs::create_*`,
//!   symlink operations, or `pkg_install`. Filesystem is read-only.
//! - Reuses [`super::repair`]'s `pub(super)` helpers to keep the classification
//!   logic authoritative in one place (symlink-dangling suggestion wording and
//!   the path-missing scan in particular).
//!
//! The JSON output schema always contains five top-level buckets:
//! `healthy`, `incomplete_pkg`, `installed_missing`, `path_missing`,
//! `symlink_dangling`. Key order within the serialized string follows
//! `serde_json`'s default (alphabetical when `preserve_order` is off, as it is
//! in this workspace) — the contract is "these five keys always present", not
//! textual ordering.
//!
//! ## `incomplete_pkg` detection
//!
//! Only **static string-literal `require`** calls of the form
//! `require("pkg_name.sub")` or `require('pkg_name.sub')` are scanned.
//! Dynamic require forms (`require(variable)`) and non-quoted forms
//! (`require "foo.bar"`) are **not** detected (MVP scope — false negatives
//! are acceptable; false positives are not). A future version may use mlua
//! to perform a real module resolution dry-run.

use std::path::Path;

use tracing::warn;

use super::super::manifest::{load_manifest, Manifest, ManifestEntry};
use super::super::project::resolve_project_root;
use super::super::resolve::packages_dir;
use super::super::source::PackageSource;
use super::super::AppService;
use super::repair::{
    collect_path_missing, collect_unattached_dangling_symlinks, symlink_dangling_suggestion,
    ProjectPathSource,
};

/// Classification of a single manifest-tracked package (read-only).
#[derive(Debug)]
enum DoctorOutcome {
    /// Destination exists and is reachable — no action required.
    Healthy,
    /// Destination is a symlink whose target is missing.
    SymlinkDangling { reason: String, suggestion: String },
    /// Destination is missing (non-symlink). Install can heal from `source`.
    InstalledMissing { reason: String, suggestion: String },
    /// Package directory exists but one or more submodule files required by
    /// `init.lua` are missing.
    IncompletePkg {
        missing_subs: Vec<String>,
        suggestion: String,
    },
}

/// Accumulator for the five JSON output buckets.
#[derive(Default)]
struct DoctorBuckets {
    healthy: Vec<serde_json::Value>,
    installed_missing: Vec<serde_json::Value>,
    symlink_dangling: Vec<serde_json::Value>,
    path_missing: Vec<serde_json::Value>,
    incomplete_pkg: Vec<serde_json::Value>,
}

impl DoctorBuckets {
    fn any_matched(&self) -> bool {
        !self.healthy.is_empty()
            || !self.installed_missing.is_empty()
            || !self.symlink_dangling.is_empty()
            || !self.path_missing.is_empty()
            || !self.incomplete_pkg.is_empty()
    }

    fn into_json(self) -> String {
        // All five buckets are always emitted (empty arrays when no entries).
        // `serde_json::json!` serializes keys alphabetically without the
        // `preserve_order` feature — consumers parse as a Map, not by order.
        serde_json::json!({
            "healthy": self.healthy,
            "incomplete_pkg": self.incomplete_pkg,
            "installed_missing": self.installed_missing,
            "symlink_dangling": self.symlink_dangling,
            "path_missing": self.path_missing,
        })
        .to_string()
    }
}

/// Parse static string-literal `require` calls from Lua source and return the
/// list of submodule names that belong to `pkg_name`.
///
/// Recognized pattern (parenthesised string literal, single or double quote):
/// ```text
/// require("pkg_name.sub")
/// require('pkg_name.sub')
/// ```
///
/// **Not** recognized (MVP scope — false negatives are acceptable):
/// - `require "foo.bar"` (no parentheses)
/// - `require(variable)` (dynamic)
/// - `require([[foo.bar]])` (long-string literal)
fn extract_required_subs(lua_src: &str, pkg_name: &str) -> Vec<String> {
    let mut subs = Vec::new();
    let prefix = format!("{pkg_name}.");
    let mut remaining = lua_src;

    while let Some(pos) = remaining.find("require") {
        remaining = &remaining[pos + "require".len()..];

        // Skip whitespace after `require`.
        let trimmed = remaining.trim_start_matches([' ', '\t']);

        // Must be followed by `(`.
        if !trimmed.starts_with('(') {
            continue;
        }
        let after_paren = &trimmed[1..];
        let after_paren = after_paren.trim_start_matches([' ', '\t']);

        // Must be followed by a string quote.
        let quote = match after_paren.chars().next() {
            Some(q @ '"') | Some(q @ '\'') => q,
            _ => continue,
        };
        let content = &after_paren[1..];
        let end = match content.find(quote) {
            Some(i) => i,
            None => continue,
        };
        let module = &content[..end];

        if let Some(sub) = module.strip_prefix(&prefix) {
            if !sub.is_empty() && !sub.contains('.') {
                // Only direct children: `pkg.sub`, not `pkg.sub.deeper`.
                subs.push(sub.to_string());
            }
        }
    }

    subs.sort();
    subs.dedup();
    subs
}

/// Build the suggestion string for an `incomplete_pkg` entry, branched by
/// whether the package came from a symlink (link path) or an installed copy.
fn incomplete_pkg_suggestion(name: &str, is_symlink: bool) -> String {
    if is_symlink {
        format!("Re-run alc_pkg_link <path> to re-link {name:?} with the complete source directory")
    } else {
        format!(
            "Run alc_pkg_install --force {name:?} to reinstall {name:?} with all submodule files"
        )
    }
}

/// Suggestion string for `installed_missing`, branched by source kind to
/// mirror `pkg_repair`'s routing:
///
/// - `Git` → `alc_pkg_install(<url>)`
/// - `Path` → `alc_pkg_install(<path>)` (local re-copy)
/// - `Bundled` → `alc_init` (bundled packages cannot be reinstalled via
///   `alc_pkg_install`; they ship inside the algocline binary)
/// - `Installed` → legacy marker with no re-fetch info (user must re-record
///   source via `alc_pkg_install`)
/// - `Unknown` → reindex + reinstall (pre-typed manifest with no source)
fn installed_missing_suggestion(name: &str, entry_source: &PackageSource) -> String {
    match entry_source {
        PackageSource::Bundled { .. } => {
            "alc_init (reinstalls bundled packages from the algocline binary)".to_string()
        }
        PackageSource::Path { path } => {
            format!("alc_pkg_install({path:?}) to reinstall {name:?} from local path")
        }
        PackageSource::Git { url, .. } => {
            format!("alc_pkg_install({url:?}) to reinstall {name:?} from Git")
        }
        PackageSource::Installed => {
            format!(
                "alc_pkg_install <path-or-url> to re-record source for {name:?} \
                 (legacy 'installed' marker carries no path)"
            )
        }
        PackageSource::Unknown => {
            format!(
                "alc_hub_reindex then alc_pkg_install <path-or-url> for {name:?} \
                 (source unknown — legacy entry)"
            )
        }
    }
}

/// Push a manifest-pass outcome into the appropriate bucket.
fn push_doctor_outcome(name: &str, outcome: DoctorOutcome, buckets: &mut DoctorBuckets) {
    match outcome {
        DoctorOutcome::Healthy => buckets.healthy.push(serde_json::json!({
            "name": name,
        })),
        DoctorOutcome::SymlinkDangling { reason, suggestion } => {
            buckets.symlink_dangling.push(serde_json::json!({
                "name": name,
                "kind": "symlink_dangling",
                "reason": reason,
                "suggestion": suggestion,
            }))
        }
        DoctorOutcome::InstalledMissing { reason, suggestion } => {
            buckets.installed_missing.push(serde_json::json!({
                "name": name,
                "kind": "installed_missing",
                "reason": reason,
                "suggestion": suggestion,
            }))
        }
        DoctorOutcome::IncompletePkg {
            missing_subs,
            suggestion,
        } => buckets.incomplete_pkg.push(serde_json::json!({
            "name": name,
            "kind": "incomplete_pkg",
            "missing_subs": missing_subs,
            "suggestion": suggestion,
        })),
    }
}

/// Check whether the package directory at `dest` is incomplete: read
/// `init.lua`, extract static `require("pkg.sub")` calls, and verify that
/// each referenced sub-file exists as `{dest}/{sub}.lua` or
/// `{dest}/{sub}/init.lua`.
///
/// Returns `Some(DoctorOutcome::IncompletePkg { .. })` when one or more
/// submodule files are missing, `None` when everything is present or when
/// `init.lua` cannot be read (IO errors are logged as warnings and treated as
/// "no incomplete evidence" rather than propagated — the directory-level
/// `Healthy` classification already passed and the init.lua read is
/// best-effort).
fn check_incomplete(name: &str, dest: &Path, is_symlink: bool) -> Option<DoctorOutcome> {
    let init_lua = dest.join("init.lua");
    let src = match std::fs::read_to_string(&init_lua) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                error = %e,
                path = %init_lua.display(),
                "could not read init.lua for incomplete check; skipping"
            );
            return None;
        }
    };

    let required_subs = extract_required_subs(&src, name);
    if required_subs.is_empty() {
        return None;
    }

    let missing: Vec<String> = required_subs
        .into_iter()
        .filter(|sub| {
            let as_file = dest.join(format!("{sub}.lua"));
            let as_dir = dest.join(sub).join("init.lua");
            !as_file.exists() && !as_dir.exists()
        })
        .collect();

    if missing.is_empty() {
        return None;
    }

    Some(DoctorOutcome::IncompletePkg {
        missing_subs: missing,
        suggestion: incomplete_pkg_suggestion(name, is_symlink),
    })
}

/// Classify a manifest entry by inspecting only the destination directory.
/// Mirrors the pre-install branch of [`super::repair::repair_installed`] but
/// never attempts an install.
///
/// After confirming the package directory is reachable, performs an additional
/// best-effort incomplete check: reads `init.lua` to detect missing sibling
/// submodule files. See [`check_incomplete`].
fn classify_installed(name: &str, entry: &ManifestEntry, pkg_dir: &Path) -> DoctorOutcome {
    let dest = pkg_dir.join(name);

    let is_symlink = dest
        .symlink_metadata()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if is_symlink {
        // `try_exists` follows the symlink — true iff target is alive.
        let target_alive = match dest.try_exists() {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, path = %dest.display(), "try_exists failed; treating symlink target as dead");
                false
            }
        };
        if target_alive {
            // Symlink alive — check for missing submodule files.
            if let Some(incomplete) = check_incomplete(name, &dest, true) {
                return incomplete;
            }
            return DoctorOutcome::Healthy;
        }
        let link_target = match dest.read_link() {
            Ok(t) => t.display().to_string(),
            Err(e) => {
                warn!(error = %e, path = %dest.display(), "read_link failed; using placeholder for dangling target");
                "<unknown>".to_string()
            }
        };
        return DoctorOutcome::SymlinkDangling {
            reason: format!("symlink target missing: {link_target}"),
            suggestion: symlink_dangling_suggestion(name),
        };
    }

    if dest.exists() {
        // Directory exists — check for missing submodule files.
        if let Some(incomplete) = check_incomplete(name, &dest, false) {
            return incomplete;
        }
        return DoctorOutcome::Healthy;
    }

    DoctorOutcome::InstalledMissing {
        reason: format!("installed directory missing: {}", dest.display()),
        suggestion: installed_missing_suggestion(name, &entry.source),
    }
}

/// Classify every manifest entry into the four buckets. When `target_filter`
/// is `Some(name)`, look the entry up directly (O(log N) on BTreeMap) instead
/// of scanning the full map.
fn run_manifest_pass(
    manifest: &Manifest,
    target_filter: Option<&str>,
    pkg_dir: &Path,
    buckets: &mut DoctorBuckets,
) {
    if let Some(target) = target_filter {
        if let Some(entry) = manifest.packages.get(target) {
            let outcome = classify_installed(target, entry, pkg_dir);
            push_doctor_outcome(target, outcome, buckets);
        }
        return;
    }
    for (pkg_name, entry) in &manifest.packages {
        let outcome = classify_installed(pkg_name, entry, pkg_dir);
        push_doctor_outcome(pkg_name, outcome, buckets);
    }
}

/// Drain the unattached-symlink scan results into the `symlink_dangling`
/// bucket. The shared helper writes tagged entries into a scratch vec so
/// its signature can stay aligned with `pkg_repair`'s unrepairable bucket.
fn run_unattached_symlink_pass(
    pkg_dir: &Path,
    target_filter: Option<&str>,
    manifest: &Manifest,
    buckets: &mut DoctorBuckets,
) {
    let mut scratch: Vec<serde_json::Value> = Vec::new();
    collect_unattached_dangling_symlinks(pkg_dir, target_filter, &manifest.packages, &mut scratch);
    buckets.symlink_dangling.extend(scratch);
}

/// Scan `alc.toml` + `alc.local.toml` for declared paths that no longer
/// resolve. `resolved_root = None` means no project context was located,
/// which mirrors `pkg_repair`'s skip-on-missing behavior.
fn run_path_missing_pass(
    resolved_root: Option<&Path>,
    target_filter: Option<&str>,
    buckets: &mut DoctorBuckets,
) {
    let Some(root) = resolved_root else {
        return;
    };
    let mut scratch: Vec<serde_json::Value> = Vec::new();
    collect_path_missing(
        root,
        target_filter,
        "project",
        &mut scratch,
        ProjectPathSource::Toml,
    );
    collect_path_missing(
        root,
        target_filter,
        "variant",
        &mut scratch,
        ProjectPathSource::Local,
    );
    buckets.path_missing.extend(scratch);
}

impl AppService {
    /// Diagnose package state without any side effects. Returns a JSON string
    /// with five arrays (`healthy`, `incomplete_pkg`, `installed_missing`,
    /// `symlink_dangling`, `path_missing`).
    ///
    /// `name` restricts the report to a single package; `None` inspects every
    /// known package. `project_root` is only consulted for the
    /// `alc.toml` / `alc.local.toml` pass. Falls back to ancestor walk from
    /// cwd when `None`.
    ///
    /// Error surface matches `pkg_repair`:
    /// - `load_manifest()` / `packages_dir()` failures propagate via `?`.
    /// - Per-entry `fs::read_dir` errors inside the unattached-symlink scan
    ///   are logged via `tracing::warn!` and skipped (helper's behavior).
    /// - `init.lua` read errors during the incomplete check are logged via
    ///   `tracing::warn!` and skipped (best-effort, no propagation).
    /// - When `name = Some(target)` and every bucket ends empty, returns
    ///   `Err` with the same wording used by `pkg_repair`.
    pub async fn pkg_doctor(
        &self,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        let app_dir = self.log_config.app_dir();
        let manifest = load_manifest(&app_dir)?;
        let pkg_dir = packages_dir(&app_dir);
        let resolved_root = resolve_project_root(project_root.as_deref());
        let target_filter = name.as_deref();

        let mut buckets = DoctorBuckets::default();
        run_manifest_pass(&manifest, target_filter, &pkg_dir, &mut buckets);
        run_unattached_symlink_pass(&pkg_dir, target_filter, &manifest, &mut buckets);
        run_path_missing_pass(resolved_root.as_deref(), target_filter, &mut buckets);

        if let Some(target) = target_filter {
            if !buckets.any_matched() {
                return Err(format!(
                    "Package '{target}' not found in installed.json, ~/.algocline/packages/, alc.toml, or alc.local.toml"
                ));
            }
        }

        Ok(buckets.into_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a minimal `ManifestEntry` with a `PackageSource::Path`.
    /// Takes a legacy path string so the existing tests keep reading
    /// naturally; the arg is wrapped into the typed `Path` variant.
    fn mk_entry(source: &str) -> ManifestEntry {
        ManifestEntry {
            version: None,
            source: PackageSource::Path {
                path: source.to_string(),
            },
            installed_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn classify_installed_healthy_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path();
        std::fs::create_dir(pkg_dir.join("p")).unwrap();

        let outcome = classify_installed("p", &mk_entry("/src/p"), pkg_dir);
        assert!(matches!(outcome, DoctorOutcome::Healthy));
    }

    #[test]
    fn classify_installed_missing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path();

        let outcome = classify_installed("p", &mk_entry("/src/p"), pkg_dir);
        match outcome {
            DoctorOutcome::InstalledMissing { reason, suggestion } => {
                assert!(
                    reason.contains("installed directory missing"),
                    "reason = {reason}"
                );
                assert!(
                    suggestion.contains("alc_pkg_install"),
                    "suggestion = {suggestion}"
                );
                assert!(
                    suggestion.contains("/src/p"),
                    "suggestion carries source: {suggestion}"
                );
            }
            _ => panic!("expected InstalledMissing"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn classify_installed_symlink_dangling() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path();
        let dangling_target = PathBuf::from("/nonexistent/path/for/doctor_test");
        symlink(&dangling_target, pkg_dir.join("p")).unwrap();

        let outcome = classify_installed("p", &mk_entry("/src/p"), pkg_dir);
        match outcome {
            DoctorOutcome::SymlinkDangling { reason, suggestion } => {
                assert!(reason.contains("symlink target missing"), "{reason}");
                assert!(suggestion.contains("alc_pkg_unlink"), "{suggestion}");
            }
            _ => panic!("expected SymlinkDangling"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn classify_installed_symlink_alive() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let real_target = tmp.path().join("real_target_dir");
        std::fs::create_dir(&real_target).unwrap();

        let pkg_dir = tmp.path().join("pkgs");
        std::fs::create_dir(&pkg_dir).unwrap();
        symlink(&real_target, pkg_dir.join("q")).unwrap();

        let outcome = classify_installed("q", &mk_entry("/src/q"), &pkg_dir);
        assert!(matches!(outcome, DoctorOutcome::Healthy));
    }

    #[test]
    fn buckets_into_json_emits_all_five_keys() {
        // NOTE: `serde_json` without the `preserve_order` feature emits JSON
        // object keys in alphabetical order, matching `pkg_repair`'s actual
        // behavior. The spec's "fixed order" requirement is satisfied by
        // always emitting these five top-level keys; consumers parse as a
        // Map rather than relying on textual key order.
        let mut b = DoctorBuckets::default();
        b.healthy.push(serde_json::json!({"name": "h"}));
        b.installed_missing
            .push(serde_json::json!({"name": "i", "kind": "installed_missing"}));
        b.symlink_dangling
            .push(serde_json::json!({"name": "s", "kind": "symlink_dangling"}));
        b.path_missing
            .push(serde_json::json!({"name": "p", "kind": "path_missing"}));
        b.incomplete_pkg
            .push(serde_json::json!({"name": "c", "kind": "incomplete_pkg"}));

        let out = b.into_json();
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let obj = parsed.as_object().expect("JSON object");
        assert!(obj.contains_key("healthy"));
        assert!(obj.contains_key("installed_missing"));
        assert!(obj.contains_key("symlink_dangling"));
        assert!(obj.contains_key("path_missing"));
        assert!(obj.contains_key("incomplete_pkg"));
        assert_eq!(obj.len(), 5, "exactly five top-level buckets: {out}");

        assert_eq!(obj["healthy"][0]["name"], "h");
        assert_eq!(obj["installed_missing"][0]["name"], "i");
        assert_eq!(obj["symlink_dangling"][0]["name"], "s");
        assert_eq!(obj["path_missing"][0]["name"], "p");
        assert_eq!(obj["incomplete_pkg"][0]["name"], "c");
    }

    #[test]
    fn any_matched_tracks_all_buckets() {
        let mut b = DoctorBuckets::default();
        assert!(!b.any_matched());
        b.healthy.push(serde_json::json!({"name": "h"}));
        assert!(b.any_matched());

        let mut b = DoctorBuckets::default();
        b.installed_missing.push(serde_json::json!({}));
        assert!(b.any_matched());

        let mut b = DoctorBuckets::default();
        b.symlink_dangling.push(serde_json::json!({}));
        assert!(b.any_matched());

        let mut b = DoctorBuckets::default();
        b.path_missing.push(serde_json::json!({}));
        assert!(b.any_matched());

        let mut b = DoctorBuckets::default();
        b.incomplete_pkg.push(serde_json::json!({}));
        assert!(b.any_matched());
    }

    #[test]
    fn installed_missing_suggestion_shape() {
        let git = PackageSource::Git {
            url: "github.com/foo/bar".to_string(),
            rev: None,
        };
        let s = installed_missing_suggestion("ucb", &git);
        assert!(s.contains("alc_pkg_install"), "{s}");
        assert!(s.contains("\"ucb\""), "{s}");
        assert!(s.contains("github.com/foo/bar"), "{s}");
    }

    /// A bundled-source entry must route the user to `alc_init`, NOT
    /// `alc_pkg_install("bundled")` (which would fail — bundled packages
    /// ship inside the algocline binary and are restored via `alc_init`).
    /// Mirrors `repair.rs` bundled arm.
    #[test]
    fn installed_missing_suggestion_routes_bundled_to_alc_init() {
        let bundled = PackageSource::Bundled { collection: None };
        let s = installed_missing_suggestion("ucb", &bundled);
        assert!(s.contains("alc_init"), "bundled must suggest alc_init: {s}");
        assert!(
            !s.contains("alc_pkg_install"),
            "bundled must NOT suggest alc_pkg_install: {s}"
        );
    }

    /// A `Path` source entry emits a suggestion pointing at
    /// `alc_pkg_install(<path>)` — matching repair's LocalPath installer
    /// route. (Under the typed migration, `alc_pkg_install` now records
    /// local installs as `Path { path }` rather than the legacy
    /// `Installed` coercion, so this is the canonical local-reinstall
    /// suggestion.)
    #[test]
    fn installed_missing_suggestion_routes_absolute_path_to_pkg_install() {
        let local = PackageSource::Path {
            path: "/abs/path/to/src".to_string(),
        };
        let s = installed_missing_suggestion("local_pkg", &local);
        assert!(s.contains("alc_pkg_install"), "{s}");
        assert!(s.contains("/abs/path/to/src"), "{s}");
    }

    /// `Unknown` source (legacy pre-typed entry with no recorded source)
    /// must route the user to `alc_hub_reindex` before attempting a
    /// reinstall — mirrors the `Unrepairable` routing in `repair.rs`.
    #[test]
    fn installed_missing_suggestion_routes_unknown_to_reindex() {
        let s = installed_missing_suggestion("legacy_pkg", &PackageSource::Unknown);
        assert!(
            s.contains("alc_hub_reindex"),
            "Unknown must suggest alc_hub_reindex: {s}"
        );
    }

    // ── extract_required_subs ────────────────────────────────────────────

    #[test]
    fn extract_subs_double_quote() {
        let src = r#"
local M = {}
local check = require("mypkg.check")
local t = require("mypkg.t")
return M
"#;
        let subs = extract_required_subs(src, "mypkg");
        assert_eq!(subs, vec!["check", "t"]);
    }

    #[test]
    fn extract_subs_single_quote() {
        let src = "local x = require('mypkg.sub')";
        let subs = extract_required_subs(src, "mypkg");
        assert_eq!(subs, vec!["sub"]);
    }

    #[test]
    fn extract_subs_ignores_other_packages() {
        let src = r#"
local x = require("other.sub")
local y = require("mypkg.mine")
"#;
        let subs = extract_required_subs(src, "mypkg");
        assert_eq!(subs, vec!["mine"]);
    }

    #[test]
    fn extract_subs_deduplicates() {
        let src = r#"
local a = require("mypkg.check")
local b = require("mypkg.check")
"#;
        let subs = extract_required_subs(src, "mypkg");
        assert_eq!(subs, vec!["check"]);
    }

    #[test]
    fn extract_subs_ignores_dynamic_require() {
        // Dynamic require (no parenthesised string literal) must not be detected.
        let src = r#"local x = require(mod_name)"#;
        let subs = extract_required_subs(src, "mypkg");
        assert!(subs.is_empty(), "dynamic require must be ignored: {subs:?}");
    }

    #[test]
    fn extract_subs_ignores_nested_dots() {
        // Only direct children: `pkg.sub`, not `pkg.sub.deeper`.
        let src = r#"local x = require("mypkg.sub.deeper")"#;
        let subs = extract_required_subs(src, "mypkg");
        assert!(
            subs.is_empty(),
            "nested dotted require must be ignored: {subs:?}"
        );
    }

    #[test]
    fn extract_subs_empty_for_no_require() {
        let src = r#"local M = {} return M"#;
        let subs = extract_required_subs(src, "mypkg");
        assert!(subs.is_empty());
    }

    // ── check_incomplete ─────────────────────────────────────────────────

    #[test]
    fn check_incomplete_returns_none_when_all_subs_present_as_lua() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"local c = require("mypkg.check") return {}"#,
        )
        .unwrap();
        std::fs::write(dest.join("check.lua"), "return {}").unwrap();

        assert!(check_incomplete("mypkg", &dest, false).is_none());
    }

    #[test]
    fn check_incomplete_returns_none_when_sub_is_dir_init() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"local c = require("mypkg.sub") return {}"#,
        )
        .unwrap();
        // sub/ directory with init.lua
        std::fs::create_dir(dest.join("sub")).unwrap();
        std::fs::write(dest.join("sub").join("init.lua"), "return {}").unwrap();

        assert!(check_incomplete("mypkg", &dest, false).is_none());
    }

    #[test]
    fn check_incomplete_detects_missing_sub() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"
local check = require("mypkg.check")
local t = require("mypkg.t")
return {}
"#,
        )
        .unwrap();
        // only `check.lua` present, `t.lua` missing
        std::fs::write(dest.join("check.lua"), "return {}").unwrap();

        let outcome = check_incomplete("mypkg", &dest, false).expect("should detect incomplete");
        match outcome {
            DoctorOutcome::IncompletePkg {
                missing_subs,
                suggestion,
            } => {
                assert_eq!(missing_subs, vec!["t"], "missing_subs: {missing_subs:?}");
                assert!(
                    suggestion.contains("alc_pkg_install"),
                    "non-symlink suggestion: {suggestion}"
                );
            }
            _ => panic!("expected IncompletePkg"),
        }
    }

    #[test]
    fn check_incomplete_suggestion_uses_link_for_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"local x = require("mypkg.missing") return {}"#,
        )
        .unwrap();
        // `missing.lua` absent

        let outcome = check_incomplete("mypkg", &dest, true).expect("should detect incomplete");
        match outcome {
            DoctorOutcome::IncompletePkg { suggestion, .. } => {
                assert!(
                    suggestion.contains("alc_pkg_link"),
                    "symlink suggestion: {suggestion}"
                );
            }
            _ => panic!("expected IncompletePkg"),
        }
    }

    #[test]
    fn check_incomplete_returns_none_when_no_init_lua() {
        // Package with no init.lua at all — best-effort skip, returns None.
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mypkg");
        std::fs::create_dir(&dest).unwrap();

        assert!(check_incomplete("mypkg", &dest, false).is_none());
    }

    #[test]
    fn classify_installed_incomplete_pkg() {
        // classify_installed should return IncompletePkg when sub.lua is missing.
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path();
        let dest = pkg_dir.join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"local x = require("mypkg.sub") return {}"#,
        )
        .unwrap();
        // sub.lua intentionally absent

        let outcome = classify_installed("mypkg", &mk_entry("/src/mypkg"), pkg_dir);
        match outcome {
            DoctorOutcome::IncompletePkg {
                missing_subs,
                suggestion,
            } => {
                assert_eq!(missing_subs, vec!["sub"]);
                assert!(suggestion.contains("alc_pkg_install"), "{suggestion}");
            }
            _ => panic!("expected IncompletePkg, got {outcome:?}"),
        }
    }

    #[test]
    fn classify_installed_healthy_when_all_subs_present() {
        // classify_installed should return Healthy when all required subs exist.
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path();
        let dest = pkg_dir.join("mypkg");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(
            dest.join("init.lua"),
            r#"local x = require("mypkg.sub") return {}"#,
        )
        .unwrap();
        std::fs::write(dest.join("sub.lua"), "return {}").unwrap();

        let outcome = classify_installed("mypkg", &mk_entry("/src/mypkg"), pkg_dir);
        assert!(
            matches!(outcome, DoctorOutcome::Healthy),
            "expected Healthy, got {outcome:?}"
        );
    }
}
