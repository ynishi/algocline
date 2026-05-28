//! `pkg_repair` — heal broken package state (Wave 2 of local-first DX).
//!
//! Scope (decisions.md Q3, issue.md `G2 stale link 修復`):
//!
//! | Broken kind | Source-of-truth | Repair? |
//! |---|---|---|
//! | (B) installed dir missing (manifest entry exists) | `installed.json.source` | ✓ via `pkg_install` |
//! | (A) global symlink dangling | none (`pkg_link` doesn't write manifest) | ✗ |
//! | (C) `alc.toml` `path = ...` missing | user-authored path | ✗ |
//! | (D) `alc.local.toml` `path = ...` missing | user-authored path | ✗ |
//!
//! `alc_pkg_repair` is an actuator (side-effecting). The sensor side
//! (`alc_pkg_list`) is intentionally read-only — see decisions.md Q3.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::super::alc_toml::{self, PackageDep};
use super::super::lockfile::load_lockfile;
use super::super::manifest::{load_manifest, ManifestEntry};
use super::super::resolve::{packages_dir, LUA_TYPE_AUTODETECT};
use super::super::source::PackageSource;
use super::super::AppService;
use super::install::InstallSource;

/// Outcome of repairing a single manifest-tracked package.
enum RepairOutcome {
    /// Successfully reinstalled from `source`.
    Repaired { source: String },
    /// Package is healthy — nothing to do.
    Skipped,
    /// Cannot repair automatically — user must intervene. `kind` is emitted
    /// verbatim into the JSON bucket entry, letting a single variant carry
    /// both the `installed_missing` sub-kinds (bundled / path) and the
    /// `symlink_dangling` case (dangling symlink at a manifest-tracked name).
    Unrepairable {
        kind: &'static str,
        reason: String,
        suggestion: String,
    },
    /// Repair was attempted but failed.
    Failed { reason: String },
}

/// Accumulator for the four JSON output buckets.
#[derive(Default)]
struct Buckets {
    repaired: Vec<serde_json::Value>,
    skipped: Vec<serde_json::Value>,
    unrepairable: Vec<serde_json::Value>,
    failed: Vec<serde_json::Value>,
}

impl Buckets {
    fn any_matched(&self) -> bool {
        !self.repaired.is_empty()
            || !self.skipped.is_empty()
            || !self.unrepairable.is_empty()
            || !self.failed.is_empty()
    }

    fn into_json(self) -> String {
        serde_json::json!({
            "repaired": self.repaired,
            "skipped": self.skipped,
            "unrepairable": self.unrepairable,
            "failed": self.failed,
        })
        .to_string()
    }
}

/// Suggestion string shared by the manifest-pass dangling-symlink case and
/// the (A) unattached-symlink pass.
pub(super) fn symlink_dangling_suggestion(name: &str) -> String {
    format!("alc_pkg_unlink({name:?}) then alc_pkg_link with the new path")
}

/// Routing bucket for alive-symlink entries detected by
/// `collect_alive_unregistered_symlinks`. The JSON shape is built by
/// `run_alive_unregistered_symlink_pass` in `doctor.rs`.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum AliveBucket {
    /// `type_source == "auto_detected_library"` from the Lua VM eval —
    /// no explicit `M.meta.type` in init.lua; routes to the `unmarked_library`
    /// doctor bucket.
    UnmarkedLibrary,
    /// All other entries (explicit type, auto-detected runnable, eval failure,
    /// or type_source absent) — routes to `unregistered_pkg`.
    Unregistered,
}

/// Push a manifest-pass outcome into the appropriate bucket. Non-Unrepairable
/// outcomes use `kind = "installed_missing"`; Unrepairable carries its own
/// kind so both `installed_missing` (bundled/path) and `symlink_dangling`
/// can flow through the same helper.
fn push_installed_outcome(name: &str, outcome: RepairOutcome, buckets: &mut Buckets) {
    match outcome {
        RepairOutcome::Repaired { source } => buckets.repaired.push(serde_json::json!({
            "name": name,
            "kind": "installed_missing",
            "action": "reinstall",
            "source": source,
        })),
        RepairOutcome::Skipped => buckets.skipped.push(serde_json::json!({
            "name": name,
            "reason": "healthy",
        })),
        RepairOutcome::Unrepairable {
            kind,
            reason,
            suggestion,
        } => buckets.unrepairable.push(serde_json::json!({
            "name": name,
            "kind": kind,
            "reason": reason,
            "suggestion": suggestion,
        })),
        RepairOutcome::Failed { reason } => buckets.failed.push(serde_json::json!({
            "name": name,
            "kind": "installed_missing",
            "reason": reason,
        })),
    }
}

impl AppService {
    /// Heal broken packages by re-installing from `installed.json` source.
    ///
    /// `name` — restrict to a single package; `None` repairs every broken pkg.
    /// `project_root` — used for project / variant pkg path checks. Falls back
    /// to ancestor walk from cwd.
    ///
    /// Returns JSON with `repaired`, `skipped`, `unrepairable`, `failed`
    /// arrays (each entry has `name` + per-bucket fields). Repair is
    /// best-effort: the per-pkg result is reported regardless of outcome.
    pub async fn pkg_repair(
        &self,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        let app_dir = self.log_config.app_dir();
        let manifest = load_manifest(&app_dir)?;
        let pkg_dir = packages_dir(&app_dir);
        let resolved_root = self.resolve_root(project_root.as_deref());

        let mut buckets = Buckets::default();
        let target_filter = name.as_deref();

        // ── (B) installed pkgs from manifest ──────────────────────
        for (pkg_name, entry) in &manifest.packages {
            if let Some(target) = target_filter {
                if target != pkg_name.as_str() {
                    continue;
                }
            }
            let outcome = self.repair_installed(pkg_name, entry, &pkg_dir).await;
            push_installed_outcome(pkg_name, outcome, &mut buckets);
        }

        // ── (A) unattached dangling symlinks (no manifest entry) ──
        collect_unattached_dangling_symlinks(
            &pkg_dir,
            target_filter,
            &manifest.packages,
            &mut buckets.unrepairable,
        );

        // ── (C) project `path = ...` missing ──────────────────────
        // ── (D) variant `path = ...` missing ──────────────────────
        if let Some(root) = resolved_root.as_ref() {
            collect_path_missing(
                root,
                target_filter,
                "project",
                &mut buckets.unrepairable,
                ProjectPathSource::Toml,
            );
            collect_path_missing(
                root,
                target_filter,
                "variant",
                &mut buckets.unrepairable,
                ProjectPathSource::Local,
            );
        }

        if let Some(target) = target_filter {
            if !buckets.any_matched() {
                return Err(format!(
                    "Package '{target}' not found in installed.json, ~/.algocline/packages/, alc.toml, or alc.local.toml"
                ));
            }
        }

        Ok(buckets.into_json())
    }

    /// Attempt to repair a single manifest-tracked package by re-running
    /// `pkg_install` with the recorded `source`. Returns `Skipped` when the
    /// package directory already exists (healthy), or Unrepairable with
    /// `kind = "symlink_dangling"` when dest is a dangling symlink — the
    /// (A) pass's "skip if in manifest" rule would otherwise drop this case.
    async fn repair_installed(
        &self,
        name: &str,
        entry: &ManifestEntry,
        pkg_dir: &Path,
    ) -> RepairOutcome {
        let dest = pkg_dir.join(name);

        let is_symlink = dest
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if is_symlink {
            // `try_exists` follows the symlink — true iff target is alive.
            let target_alive = dest.try_exists().unwrap_or(false);
            if target_alive {
                return RepairOutcome::Skipped;
            }
            let link_target = dest
                .read_link()
                .map(|t| t.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            return RepairOutcome::Unrepairable {
                kind: "symlink_dangling",
                reason: format!("symlink target missing: {link_target}"),
                suggestion: symlink_dangling_suggestion(name),
            };
        }

        if dest.exists() {
            return RepairOutcome::Skipped;
        }

        // Source classification: only `Path` (local copy) and `Git` can be
        // re-fetched. Bundled is conceptually re-installable via `alc_init`;
        // `Installed` is a legacy marker that carries no re-fetch info (the
        // typed successor is `Path { path }`). `Unknown` is the pre-typed
        // "source unrecorded" landing site and is structurally unrepairable.
        //
        // States detectable before attempting install belong in `unrepairable`,
        // not `failed`. `failed` is reserved for runtime errors during an
        // actual install attempt.
        let install_source = match &entry.source {
            PackageSource::Path { path } => InstallSource::LocalPath(PathBuf::from(path)),
            PackageSource::Git { url, .. } => InstallSource::GitUrl(normalize_git_url(url)),
            PackageSource::Bundled { .. } => {
                return RepairOutcome::Unrepairable {
                    kind: "installed_missing",
                    reason: "bundled package — restore via `alc_init` or reinstall algocline"
                        .to_string(),
                    suggestion: "alc_init (reinstalls bundled packages from the algocline binary)"
                        .to_string(),
                };
            }
            PackageSource::Installed => {
                // Legacy marker: pre-typed manifest that recorded a local install
                // as `source: "installed"` / absolute path (see
                // `infer_from_legacy_source_string`). The actual source path is
                // lost, so we cannot re-fetch automatically.
                return RepairOutcome::Unrepairable {
                    kind: "installed_missing",
                    reason: "legacy 'installed' marker carries no source path".to_string(),
                    suggestion: "alc_pkg_install <path-or-url> to re-record source, \
                                 then alc_pkg_repair"
                        .to_string(),
                };
            }
            PackageSource::Unknown => {
                // Pre-typed manifest entry with `source: ""` (never recorded).
                // Routed here per the Phase 3 spec: `Unknown` must land in
                // `Unrepairable`, not be silently coerced.
                return RepairOutcome::Unrepairable {
                    kind: "installed_missing",
                    reason: "source unknown (legacy entry; run alc_hub_reindex)".to_string(),
                    suggestion: "alc_hub_reindex to rebuild the index, or \
                                 alc_pkg_install <path-or-url> to re-record source"
                        .to_string(),
                };
            }
        };

        // Pre-check: a LocalPath is structurally unrepairable when
        // (a) the source directory no longer exists, or
        // (b) the source exists but the named package's subdirectory
        //     (`<source>/<name>/init.lua`) is absent.
        //
        // Since Single-package mode was removed in v0.36.0, all local installs
        // use collection layout: the recorded source is the collection root and
        // each package lives at `<source>/<name>/init.lua`.  We check for the
        // named package's own init.lua rather than a root-level one.
        //
        // Git sources are deliberately not pre-checked here: network/remote
        // availability is a runtime concern that belongs in the attempt path.
        if let InstallSource::LocalPath(ref p) = install_source {
            if !p.exists() {
                return RepairOutcome::Unrepairable {
                    kind: "installed_missing",
                    reason: format!("source directory missing: {}", p.display()),
                    suggestion: format!(
                        "alc_pkg_install from a valid source, or remove the '{name}' entry from ~/.algocline/installed.json"
                    ),
                };
            }
            if !p.join(name).join("init.lua").exists() {
                return RepairOutcome::Unrepairable {
                    kind: "installed_missing",
                    reason: format!(
                        "source directory has no init.lua at root: {}",
                        p.display()
                    ),
                    suggestion: format!(
                        "alc_pkg_install from a valid source, or remove the '{name}' entry from ~/.algocline/installed.json"
                    ),
                };
            }
        }

        // Re-install from the collection root.  Single-package mode was removed
        // in v0.36.0, so `name` is not passed; `install_from_local_path` will
        // scan `<source>/<name>/init.lua` and reinstall all packages found in
        // the collection.  For the common case of a 1-entry collection this is
        // equivalent to targeted reinstall.
        match self.pkg_install_typed(install_source, None, None).await {
            Ok(_) => RepairOutcome::Repaired {
                // Emit a human-readable source string (legacy schema). The
                // typed source is already persisted back into the manifest
                // by the install path — this field is just display.
                source: entry.source.display_string(),
            },
            Err(e) => RepairOutcome::Failed { reason: e },
        }
    }
}

/// Apply the same URL scheme normalization `classify_install_url` uses
/// without re-checking whether the string refers to a local directory.
/// Repair has already established the source is Git (typed
/// `PackageSource::Git`); re-classifying via the directory heuristic would
/// be both redundant and racy. Delegates to the shared
/// [`super::install::prefix_git_scheme_if_missing`] helper so that install
/// and repair stay in lockstep on scheme handling.
fn normalize_git_url(url: &str) -> String {
    super::install::prefix_git_scheme_if_missing(url)
}

/// Scan `pkg_dir` for dangling symlinks whose name is *not* present in the
/// manifest. Manifest-tracked names are handled by `repair_installed` so
/// they're skipped here to avoid double-counting.
pub(super) fn collect_unattached_dangling_symlinks(
    pkg_dir: &Path,
    target_filter: Option<&str>,
    manifest_names: &std::collections::BTreeMap<String, ManifestEntry>,
    unrepairable: &mut Vec<serde_json::Value>,
) {
    let read = match std::fs::read_dir(pkg_dir) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                "pkg: failed to read packages_dir at {}: {e}",
                pkg_dir.display()
            );
            return;
        }
    };

    for dir_entry_result in read {
        let dir_entry = match dir_entry_result {
            Ok(e) => e,
            Err(e) => {
                // Previously this scan used `read.flatten()` which dropped
                // per-entry I/O errors silently. Some names (permission
                // denials, transient FS errors) therefore slipped through
                // the dangling-symlink check without diagnosis. Log here
                // so at least the repair attempt leaves a trail.
                tracing::warn!(
                    "pkg: skipping unreadable entry in {}: {e}",
                    pkg_dir.display()
                );
                continue;
            }
        };
        let path = dir_entry.path();
        let pkg_name = dir_entry.file_name().to_string_lossy().to_string();

        if let Some(target) = target_filter {
            if target != pkg_name.as_str() {
                continue;
            }
        }
        if manifest_names.contains_key(&pkg_name) {
            continue;
        }

        let is_symlink = path
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if !is_symlink {
            continue;
        }
        let target_exists = path.try_exists().unwrap_or(false);
        if target_exists {
            continue;
        }

        let link_target = path
            .read_link()
            .map(|t| t.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());

        unrepairable.push(serde_json::json!({
            "name": pkg_name,
            "kind": "symlink_dangling",
            "reason": format!("symlink target missing: {link_target}"),
            "suggestion": symlink_dangling_suggestion(&pkg_name),
        }));
    }
}

/// Which TOML file is the source of truth for path entries.
#[derive(Debug, Clone, Copy)]
pub(super) enum ProjectPathSource {
    /// `alc.toml` `[packages.x] path = ...` (project scope).
    Toml,
    /// `alc.local.toml` `[packages.x] path = ...` (variant scope).
    Local,
}

/// Append `path_missing` unrepairable entries for either alc.toml or
/// alc.local.toml. Filtering by `target_filter` (Some(name)) restricts
/// to a single package.
pub(super) fn collect_path_missing(
    root: &Path,
    target_filter: Option<&str>,
    scope: &'static str,
    unrepairable: &mut Vec<serde_json::Value>,
    src: ProjectPathSource,
) {
    let loaded = match src {
        ProjectPathSource::Toml => alc_toml::load_alc_toml(root),
        ProjectPathSource::Local => alc_toml::load_alc_local_toml(root),
    };
    let Ok(Some(toml_data)) = loaded else {
        return;
    };

    // For project scope, the lockfile is the more accurate source for the
    // resolved path (it absorbs canonicalization done at install time). Fall
    // back to the alc.toml declaration when no lockfile exists.
    //
    // TODO(variant-canonicalization): variant scope reads the raw
    // alc.local.toml path verbatim. If `pkg_link --scope=variant` ever starts
    // writing relative paths (today it writes absolute), this block will
    // diverge from what `pkg_list` / `pkg_run` resolve — mirror the project
    // lockfile lookup for variants at that point.
    let lock_lookup = if matches!(src, ProjectPathSource::Toml) {
        load_lockfile(root).ok().flatten().map(|l| {
            l.packages
                .into_iter()
                .filter_map(|p| match p.source {
                    PackageSource::Path { path } => Some((p.name, path)),
                    _ => None,
                })
                .collect::<std::collections::HashMap<String, String>>()
        })
    } else {
        None
    };

    for (name, dep) in &toml_data.packages {
        if let Some(t) = target_filter {
            if t != name.as_str() {
                continue;
            }
        }

        let raw = match dep {
            PackageDep::Path { path, .. } => path,
            _ => continue,
        };

        let resolved_raw = lock_lookup
            .as_ref()
            .and_then(|m| m.get(name).cloned())
            .unwrap_or_else(|| raw.clone());

        let p = Path::new(&resolved_raw);
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            root.join(p)
        };

        if abs.exists() {
            continue;
        }

        let suggestion = match src {
            ProjectPathSource::Toml => {
                format!("update or remove [packages.{name}] in alc.toml")
            }
            ProjectPathSource::Local => {
                format!("alc_pkg_unlink({name:?}) or update [packages.{name}] in alc.local.toml")
            }
        };

        unrepairable.push(serde_json::json!({
            "name": name,
            "kind": "path_missing",
            "scope": scope,
            "reason": format!("declared path does not exist: {}", abs.display()),
            "suggestion": suggestion,
        }));
    }
}

/// Walk `pkg_dir` and collect physical directories that contain `init.lua` but
/// are not registered in any of the three authoritative sources:
/// `installed.json` (manifest), `alc.toml [packages]`, or
/// `alc.local.toml [packages]`.
///
/// # Arguments
///
/// * `pkg_dir` — `~/.algocline/packages/` (or the path under test)
/// * `registered` — set of package names known to any registration source
/// * `registered_paths` — canonicalized absolute paths declared in
///   `[packages.x] path = "..."` entries from alc.toml / alc.local.toml; used
///   to skip false positives where a path-dep points inside `pkg_dir`
/// * `target_filter` — when `Some(name)`, restrict output to that single name
///
/// # Returns
///
/// A `Vec<serde_json::Value>` of `unregistered_pkg` bucket entries on success.
/// Each entry carries `name`, `kind`, `source`, `reason`, and `suggestion`
/// (array of four strings, Clippy-style multi-line).
///
/// # Errors
///
/// Returns `Err(String)` if `pkg_dir` exists but cannot be read (any `io::Error`
/// other than `NotFound`). `NotFound` is treated as empty (no packages installed)
/// and returns `Ok(vec![])`.
pub(super) fn collect_unregistered_pkg_dirs(
    pkg_dir: &Path,
    registered: &HashSet<String>,
    registered_paths: &[PathBuf],
    target_filter: Option<&str>,
) -> Result<Vec<serde_json::Value>, String> {
    let read = match std::fs::read_dir(pkg_dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // packages_dir absent == empty, not an error (file absent === empty).
            return Ok(vec![]);
        }
        Err(e) => {
            return Err(format!(
                "pkg: failed to read packages_dir at {}: {e}",
                pkg_dir.display()
            ));
        }
    };

    let mut entries = Vec::new();

    for dir_entry_result in read {
        let dir_entry = match dir_entry_result {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    "pkg: skipping unreadable entry in {}: {e}",
                    pkg_dir.display()
                );
                continue;
            }
        };

        let path = dir_entry.path();
        let pkg_name = dir_entry.file_name().to_string_lossy().to_string();

        // When a specific target is requested, skip all others.
        if let Some(target) = target_filter {
            if target != pkg_name.as_str() {
                continue;
            }
        }

        // Skip if name is already in one of the three registration sources.
        if registered.contains(&pkg_name) {
            continue;
        }

        // Only physical directories with init.lua qualify.
        let meta = match path.symlink_metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("pkg: cannot stat {}: {e}", path.display());
                continue;
            }
        };
        if !meta.is_dir() {
            // Symlinks are handled by run_unattached_symlink_pass; skip here.
            continue;
        }
        if !path.join("init.lua").exists() {
            // Empty or non-package directory — skip (AC-2).
            continue;
        }

        // Canonical path comparison: skip if any alc.toml / alc.local.toml
        // path entry resolves to the same physical directory (AC-4).
        let canonical_pkg_path = match path.canonicalize() {
            Ok(c) => c,
            Err(e) => {
                return Err(format!(
                    "pkg: failed to canonicalize existing dir {}: {e}",
                    path.display()
                ));
            }
        };
        if registered_paths.contains(&canonical_pkg_path) {
            continue;
        }

        // Build Clippy-style multi-line suggestion (crux constraint: 4 elements,
        // suggestion field is array<string> for unregistered_pkg only).
        let abs_path = path.display().to_string();
        let suggestion = serde_json::json!([
            format!(
                "If this pkg was scaffolded outside `alc_pkg_scaffold` and you want it installed: \
                `alc_pkg_install --force {abs_path}` (re-copy + register in installed.json)"
            ),
            format!(
                "If you are actively iterating on this pkg in-tree: \
                `alc_pkg_link {abs_path}` (symlink-based, no copy)"
            ),
            format!("If this dir is stale/abandoned: `rm -rf {abs_path}` to clean it up"),
            "Note: source is unknown — git URL cannot be inferred from the bare directory. \
            Re-record via one of the above."
                .to_string(),
        ]);

        entries.push(serde_json::json!({
            "name": pkg_name,
            "kind": "unregistered_pkg",
            "source": "unknown",
            "reason": format!(
                "physical dir with init.lua exists but is not registered in \
                installed.json, alc.toml, or alc.local.toml: {}",
                path.display()
            ),
            "suggestion": suggestion,
        }));
    }

    Ok(entries)
}

/// Returns `true` iff `name` is safe to interpolate into a Lua `require()` call.
///
/// Accepts ASCII alphanumerics, `_` and `-`. Empty strings are rejected.
/// Mirrors the implementation in `list.rs` (which also allows `-` for hyphenated
/// package names such as `crdt-doc`).
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl AppService {
    /// Walk `pkg_dir` and collect **alive symlinks** that contain `init.lua` at the
    /// resolved target but are not registered in any of the three authoritative
    /// sources: `installed.json` (manifest), `alc.toml [packages]`, or
    /// `alc.local.toml [packages]`.
    ///
    /// Unlike `collect_unregistered_pkg_dirs` (physical dirs only) and
    /// `collect_unattached_dangling_symlinks` (dangling symlinks only), this
    /// helper exclusively handles alive symlinks — those where the link target
    /// exists and is reachable via `path.try_exists()`.
    ///
    /// Each qualifying entry is classified into an [`AliveBucket`] by executing
    /// the `LUA_TYPE_AUTODETECT` snippet via `eval_simple_with_paths` — the same
    /// runtime path used by `pkg_list`. The `type_source` field in the returned
    /// meta determines the bucket:
    ///
    /// - `"auto_detected_library"` → [`AliveBucket::UnmarkedLibrary`]
    /// - All other values (explicit, auto_detected_runnable, eval failure, or
    ///   absent) → [`AliveBucket::Unregistered`]
    ///
    /// JSON shape construction is deferred to `run_alive_unregistered_symlink_pass`
    /// in `doctor.rs`; this method is detection-only.
    ///
    /// # Arguments
    ///
    /// * `pkg_dir` — `~/.algocline/packages/` (or the path under test)
    /// * `registered` — set of package names known to any registration source
    /// * `registered_paths` — canonicalized absolute paths declared in
    ///   `[packages.x] path = "..."` entries from alc.toml / alc.local.toml; used
    ///   to skip false positives where a path-dep symlink resolves to a registered dir
    /// * `target_filter` — when `Some(name)`, restrict output to that single name
    ///
    /// # Returns
    ///
    /// A `Vec<(String, AliveBucket)>` of `(pkg_name, bucket)` pairs on success.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if `pkg_dir` exists but cannot be read (any `io::Error`
    /// other than `NotFound`). `NotFound` is treated as empty (no packages installed)
    /// and returns `Ok(vec![])`. Individual entry stat or eval failures emit a
    /// `tracing::warn!` and continue. `canonicalize` failure returns `Err`.
    pub(super) async fn collect_alive_unregistered_symlinks(
        &self,
        pkg_dir: &Path,
        registered: &HashSet<String>,
        registered_paths: &[PathBuf],
        target_filter: Option<&str>,
    ) -> Result<Vec<(String, AliveBucket)>, String> {
        let read = match std::fs::read_dir(pkg_dir) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // packages_dir absent == empty, not an error (file absent === empty).
                return Ok(vec![]);
            }
            Err(e) => {
                return Err(format!(
                    "pkg: failed to read packages_dir at {}: {e}",
                    pkg_dir.display()
                ));
            }
        };

        let mut entries = Vec::new();

        for dir_entry_result in read {
            let dir_entry = match dir_entry_result {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        "pkg: skipping unreadable entry in {}: {e}",
                        pkg_dir.display()
                    );
                    continue;
                }
            };

            let path = dir_entry.path();
            let pkg_name = dir_entry.file_name().to_string_lossy().to_string();

            // When a specific target is requested, skip all others.
            if let Some(target) = target_filter {
                if target != pkg_name.as_str() {
                    continue;
                }
            }

            // Skip if name is already in one of the three registration sources.
            if registered.contains(&pkg_name) {
                continue;
            }

            // Only symlinks qualify for this pass.
            let meta = match path.symlink_metadata() {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("pkg: cannot stat {}: {e}", path.display());
                    continue;
                }
            };
            if !meta.file_type().is_symlink() {
                // Physical dirs are handled by collect_unregistered_pkg_dirs; skip here.
                continue;
            }

            // Only alive symlinks — dangling ones belong to collect_unattached_dangling_symlinks.
            // `try_exists` follows the link: true iff target is reachable.
            let target_exists = path.try_exists().unwrap_or(false);
            if !target_exists {
                continue;
            }

            // Link target must have an init.lua (follow through symlink via std::fs::exists).
            if !path.join("init.lua").exists() {
                continue;
            }

            // Canonical path comparison: skip if any alc.toml / alc.local.toml
            // path entry resolves to the same physical directory (false-positive guard).
            // `canonicalize` follows the symlink and returns the link target's absolute path.
            let canonical_pkg_path = match path.canonicalize() {
                Ok(c) => c,
                Err(e) => {
                    return Err(format!(
                        "pkg: failed to canonicalize symlink target {}: {e}",
                        path.display()
                    ));
                }
            };
            if registered_paths.contains(&canonical_pkg_path) {
                continue;
            }

            // Classify via eval_simple + LUA_TYPE_AUTODETECT — same runtime path as
            // pkg_list. pkg_dir is passed as an extra lib path so require() resolves
            // the package in both production (~/.algocline/packages/) and tests.
            let bucket = if is_safe_pkg_name(&pkg_name) {
                let code = format!(
                    r#"package.loaded["{pkg_name}"] = nil
local pkg = require("{pkg_name}")
local meta = pkg.meta or {{ name = "{pkg_name}" }}
{LUA_TYPE_AUTODETECT}
return meta"#,
                    pkg_name = pkg_name,
                    LUA_TYPE_AUTODETECT = LUA_TYPE_AUTODETECT,
                );
                match self
                    .executor
                    .eval_simple_with_paths(code, vec![pkg_dir.to_path_buf()], vec![])
                    .await
                {
                    Ok(meta) => {
                        if meta.get("type_source").and_then(|v| v.as_str())
                            == Some("auto_detected_library")
                        {
                            AliveBucket::UnmarkedLibrary
                        } else {
                            AliveBucket::Unregistered
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "pkg: alive-symlink type_source eval failed for {pkg_name}: {e}"
                        );
                        AliveBucket::Unregistered
                    }
                }
            } else {
                // Unsafe name cannot be interpolated into require() — skip eval,
                // fall back to Unregistered.
                AliveBucket::Unregistered
            };

            entries.push((pkg_name, bucket));
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    mod alive_symlink_tests {
        use super::super::super::super::test_support::make_app_service_at;
        use super::*;
        use std::os::unix::fs::symlink as unix_symlink;

        /// Write a minimal init.lua with `M.meta.name` but no explicit type and no
        /// `M.run` function — `LUA_TYPE_AUTODETECT` will classify this as
        /// `type_source = "auto_detected_library"`.
        fn write_auto_library_init_lua(pkg_dir: &std::path::Path, pkg_name: &str) {
            let pkg = pkg_dir.join(pkg_name);
            std::fs::create_dir_all(&pkg).expect("create pkg dir");
            std::fs::write(
                pkg.join("init.lua"),
                format!("local M = {{}}\nM.meta = {{ name = \"{pkg_name}\" }}\nreturn M\n"),
            )
            .expect("write init.lua");
        }

        /// Write an init.lua with an explicit `M.meta.type = "library"` — this
        /// gives `type_source = "explicit"` via `LUA_TYPE_AUTODETECT`, so the
        /// entry must go to `Unregistered`.
        fn write_explicit_type_init_lua(pkg_dir: &std::path::Path, pkg_name: &str) {
            let pkg = pkg_dir.join(pkg_name);
            std::fs::create_dir_all(&pkg).expect("create pkg dir");
            std::fs::write(
                pkg.join("init.lua"),
                format!(
                    "local M = {{}}\nM.meta = {{ name = \"{pkg_name}\", type = \"library\" }}\nreturn M\n"
                ),
            )
            .expect("write init.lua");
        }

        /// (a) A dangling symlink (target directory absent) must be excluded.
        #[tokio::test]
        async fn dangling_symlink_excluded() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            // Point the symlink at a non-existent directory.
            let link = pkg_dir.join("ghost_pkg");
            unix_symlink(tmp.path().join("does_not_exist"), &link)
                .expect("create dangling symlink");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let registered = HashSet::new();
            let registered_paths: Vec<PathBuf> = vec![];
            let result = svc
                .collect_alive_unregistered_symlinks(&pkg_dir, &registered, &registered_paths, None)
                .await
                .expect("helper should not error");

            assert!(
                result.is_empty(),
                "dangling symlink must not appear in result"
            );
        }

        /// (b) An alive symlink + unregistered + init.lua with no explicit type and no
        /// run function → `LUA_TYPE_AUTODETECT` sets type_source = "auto_detected_library"
        /// → `AliveBucket::UnmarkedLibrary`.
        #[tokio::test]
        async fn alive_unregistered_auto_library_routes_to_unmarked_library() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let real_pkgs = tmp.path().join("real");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            // Real pkg directory (link target).
            write_auto_library_init_lua(&real_pkgs, "my_lib");

            // Alive symlink in packages/ pointing at real/my_lib.
            unix_symlink(real_pkgs.join("my_lib"), pkg_dir.join("my_lib"))
                .expect("create alive symlink");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let registered = HashSet::new();
            let registered_paths: Vec<PathBuf> = vec![];
            let result = svc
                .collect_alive_unregistered_symlinks(&pkg_dir, &registered, &registered_paths, None)
                .await
                .expect("helper should not error");

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].0, "my_lib");
            assert_eq!(result[0].1, AliveBucket::UnmarkedLibrary);
        }

        /// (c) An alive symlink + unregistered + init.lua with explicit
        /// `M.meta.type = "library"` → `type_source = "explicit"` via
        /// `LUA_TYPE_AUTODETECT` → `AliveBucket::Unregistered`.
        #[tokio::test]
        async fn alive_unregistered_explicit_type_routes_to_unregistered() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let real_pkgs = tmp.path().join("real");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            write_explicit_type_init_lua(&real_pkgs, "explicit_lib");

            unix_symlink(real_pkgs.join("explicit_lib"), pkg_dir.join("explicit_lib"))
                .expect("create alive symlink");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let registered = HashSet::new();
            let registered_paths: Vec<PathBuf> = vec![];
            let result = svc
                .collect_alive_unregistered_symlinks(&pkg_dir, &registered, &registered_paths, None)
                .await
                .expect("helper should not error");

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].0, "explicit_lib");
            assert_eq!(result[0].1, AliveBucket::Unregistered);
        }

        /// (d) An alive symlink whose name appears in `registered` must be skipped.
        #[tokio::test]
        async fn alive_registered_pkg_excluded() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let real_pkgs = tmp.path().join("real");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            write_auto_library_init_lua(&real_pkgs, "known_pkg");

            unix_symlink(real_pkgs.join("known_pkg"), pkg_dir.join("known_pkg"))
                .expect("create alive symlink");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let mut registered = HashSet::new();
            registered.insert("known_pkg".to_string());
            let registered_paths: Vec<PathBuf> = vec![];
            let result = svc
                .collect_alive_unregistered_symlinks(&pkg_dir, &registered, &registered_paths, None)
                .await
                .expect("helper should not error");

            assert!(
                result.is_empty(),
                "registered pkg must not appear in result"
            );
        }

        /// (e) target_filter restricts output to the named package only.
        #[tokio::test]
        async fn target_filter_restricts_output() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let real_pkgs = tmp.path().join("real");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            write_auto_library_init_lua(&real_pkgs, "lib_a");
            write_auto_library_init_lua(&real_pkgs, "lib_b");

            unix_symlink(real_pkgs.join("lib_a"), pkg_dir.join("lib_a"))
                .expect("create symlink lib_a");
            unix_symlink(real_pkgs.join("lib_b"), pkg_dir.join("lib_b"))
                .expect("create symlink lib_b");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let registered = HashSet::new();
            let registered_paths: Vec<PathBuf> = vec![];
            let result = svc
                .collect_alive_unregistered_symlinks(
                    &pkg_dir,
                    &registered,
                    &registered_paths,
                    Some("lib_a"),
                )
                .await
                .expect("helper should not error");

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].0, "lib_a");
        }

        /// (f) An entry whose canonicalized path appears in `registered_paths`
        /// must be skipped (path-dep false-positive guard).
        #[tokio::test]
        async fn registered_path_dep_excluded() {
            let tmp = tempfile::tempdir().expect("create tempdir");
            let real_pkgs = tmp.path().join("real");
            let pkg_dir = tmp.path().join("packages");
            std::fs::create_dir_all(&pkg_dir).expect("create packages dir");

            write_auto_library_init_lua(&real_pkgs, "path_dep_lib");

            let real_dir = real_pkgs.join("path_dep_lib");
            unix_symlink(&real_dir, pkg_dir.join("path_dep_lib")).expect("create alive symlink");

            // Canonicalize the real dir to simulate what registered_paths contains.
            let canonical = real_dir.canonicalize().expect("canonicalize real dir");

            let svc = make_app_service_at(tmp.path().to_path_buf()).await;
            let registered = HashSet::new();
            let registered_paths = vec![canonical];
            let result = svc
                .collect_alive_unregistered_symlinks(&pkg_dir, &registered, &registered_paths, None)
                .await
                .expect("helper should not error");

            assert!(
                result.is_empty(),
                "path-dep registered entry must not appear in result"
            );
        }
    }
}
