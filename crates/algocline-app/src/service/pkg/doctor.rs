//! `pkg_doctor` — read-only diagnosis for package state (Wave 2 of local-first DX).
//!
//! The actuator counterpart is [`super::repair`] (`pkg_repair`). `pkg_doctor`
//! classifies packages into four buckets without touching the filesystem:
//!
//! | Bucket              | Source-of-truth                                   | Condition                                          |
//! |---------------------|---------------------------------------------------|----------------------------------------------------|
//! | `healthy`           | `installed.json` + `~/.algocline/packages/{name}` | dest directory exists (resolved through symlinks)  |
//! | `installed_missing` | `installed.json`                                  | dest missing (non-symlink), `pkg_install` can heal |
//! | `symlink_dangling`  | filesystem scan                                   | dest is a symlink whose target is missing          |
//! | `path_missing`      | `alc.toml` / `alc.local.toml`                     | declared `path = ...` does not exist               |
//!
//! Contract:
//! - **No side effects.** No `fs::write`, `fs::remove_*`, `fs::create_*`,
//!   symlink operations, or `pkg_install`. Filesystem is read-only.
//! - Reuses [`super::repair`]'s `pub(super)` helpers to keep the classification
//!   logic authoritative in one place (symlink-dangling suggestion wording and
//!   the path-missing scan in particular).
//!
//! The JSON output schema always contains these four top-level buckets:
//! `healthy`, `installed_missing`, `symlink_dangling`, `path_missing`. Key
//! order within the serialized string follows `serde_json`'s default
//! (alphabetical when `preserve_order` is off, as it is in this workspace) —
//! the contract is "these four keys always present", not textual ordering.

use std::path::Path;

use tracing::warn;

use super::super::manifest::{load_manifest, Manifest, ManifestEntry};
use super::super::project::resolve_project_root;
use super::super::resolve::packages_dir;
use super::super::AppService;
use super::repair::{
    collect_path_missing, collect_unattached_dangling_symlinks, symlink_dangling_suggestion,
    ProjectPathSource,
};

/// Classification of a single manifest-tracked package (read-only).
enum DoctorOutcome {
    /// Destination exists and is reachable — no action required.
    Healthy,
    /// Destination is a symlink whose target is missing.
    SymlinkDangling { reason: String, suggestion: String },
    /// Destination is missing (non-symlink). Install can heal from `source`.
    InstalledMissing { reason: String, suggestion: String },
}

/// Accumulator for the four JSON output buckets.
#[derive(Default)]
struct DoctorBuckets {
    healthy: Vec<serde_json::Value>,
    installed_missing: Vec<serde_json::Value>,
    symlink_dangling: Vec<serde_json::Value>,
    path_missing: Vec<serde_json::Value>,
}

impl DoctorBuckets {
    fn any_matched(&self) -> bool {
        !self.healthy.is_empty()
            || !self.installed_missing.is_empty()
            || !self.symlink_dangling.is_empty()
            || !self.path_missing.is_empty()
    }

    fn into_json(self) -> String {
        // All four buckets are always emitted (empty arrays when no entries).
        // `serde_json::json!` serializes keys alphabetically without the
        // `preserve_order` feature — consumers parse as a Map, not by order.
        serde_json::json!({
            "healthy": self.healthy,
            "installed_missing": self.installed_missing,
            "symlink_dangling": self.symlink_dangling,
            "path_missing": self.path_missing,
        })
        .to_string()
    }
}

/// Suggestion string for `installed_missing` — routes the caller to
/// `alc_pkg_install`. Kept local to the doctor module (doctor never calls
/// `pkg_install` itself, so reusing the install helper in `repair.rs` would
/// be misleading).
fn installed_missing_suggestion(name: &str, source: &str) -> String {
    format!("alc_pkg_install({name:?}) to reinstall from source ({source})")
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
    }
}

/// Classify a manifest entry by inspecting only the destination directory.
/// Mirrors the pre-install branch of [`super::repair::repair_installed`] but
/// never attempts an install.
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
        return DoctorOutcome::Healthy;
    }

    DoctorOutcome::InstalledMissing {
        reason: format!("installed directory missing: {}", dest.display()),
        suggestion: installed_missing_suggestion(name, &entry.source),
    }
}

/// Classify every manifest entry into the four buckets.
fn run_manifest_pass(
    manifest: &Manifest,
    target_filter: Option<&str>,
    pkg_dir: &Path,
    buckets: &mut DoctorBuckets,
) {
    for (pkg_name, entry) in &manifest.packages {
        if let Some(target) = target_filter {
            if target != pkg_name.as_str() {
                continue;
            }
        }
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
    /// with four arrays (`healthy`, `installed_missing`, `symlink_dangling`,
    /// `path_missing`).
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
    /// - When `name = Some(target)` and every bucket ends empty, returns
    ///   `Err` with the same wording used by `pkg_repair`.
    pub async fn pkg_doctor(
        &self,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        let manifest = load_manifest()?;
        let pkg_dir = packages_dir()?;
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

    fn mk_entry(source: &str) -> ManifestEntry {
        ManifestEntry {
            version: None,
            source: source.to_string(),
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
    fn buckets_into_json_emits_all_four_keys() {
        // NOTE: `serde_json` without the `preserve_order` feature emits JSON
        // object keys in alphabetical order, matching `pkg_repair`'s actual
        // behavior. The spec's "fixed order" requirement is satisfied by
        // always emitting these four top-level keys; consumers parse as a
        // Map rather than relying on textual key order.
        let mut b = DoctorBuckets::default();
        b.healthy.push(serde_json::json!({"name": "h"}));
        b.installed_missing
            .push(serde_json::json!({"name": "i", "kind": "installed_missing"}));
        b.symlink_dangling
            .push(serde_json::json!({"name": "s", "kind": "symlink_dangling"}));
        b.path_missing
            .push(serde_json::json!({"name": "p", "kind": "path_missing"}));

        let out = b.into_json();
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let obj = parsed.as_object().expect("JSON object");
        assert!(obj.contains_key("healthy"));
        assert!(obj.contains_key("installed_missing"));
        assert!(obj.contains_key("symlink_dangling"));
        assert!(obj.contains_key("path_missing"));
        assert_eq!(obj.len(), 4, "exactly four top-level buckets: {out}");

        assert_eq!(obj["healthy"][0]["name"], "h");
        assert_eq!(obj["installed_missing"][0]["name"], "i");
        assert_eq!(obj["symlink_dangling"][0]["name"], "s");
        assert_eq!(obj["path_missing"][0]["name"], "p");
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
    }

    #[test]
    fn installed_missing_suggestion_shape() {
        let s = installed_missing_suggestion("ucb", "github.com/foo/bar");
        assert!(s.contains("alc_pkg_install"), "{s}");
        assert!(s.contains("\"ucb\""), "{s}");
        assert!(s.contains("github.com/foo/bar"), "{s}");
    }
}
