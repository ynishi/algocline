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

use std::path::{Path, PathBuf};

use super::super::alc_toml::{self, PackageDep};
use super::super::lockfile::load_lockfile;
use super::super::manifest::{load_manifest, ManifestEntry};
use super::super::project::resolve_project_root;
use super::super::resolve::packages_dir;
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
        let manifest = load_manifest()?;
        let pkg_dir = packages_dir()?;
        let resolved_root = resolve_project_root(project_root.as_deref());

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
        // (b) the source exists but has no `init.lua` at its root.
        // (b) matters because `install_from_local_path` routes a no-root-init
        // source into collection mode, which rejects the `name` argument that
        // repair must pass — the combination is unreachable with the current
        // install layer, so there are no bytes to copy for *this* named pkg.
        // Classify both up front rather than letting the install layer fail
        // at runtime; that would land in `failed`, mixing structural
        // impossibility with transient runtime failures.
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
            if !p.join("init.lua").exists() {
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

        match self
            .pkg_install_typed(install_source, Some(name.to_string()))
            .await
        {
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

/// Apply the same URL normalization `classify_install_url` uses (prefix
/// `https://` to bare domain-style URLs) without re-checking whether the
/// string refers to a local directory. Repair has already established the
/// source is Git; re-classifying via the directory heuristic would be both
/// redundant and racy.
fn normalize_git_url(url: &str) -> String {
    if url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("file://")
        || url.starts_with("git@")
    {
        url.to_string()
    } else {
        format!("https://{url}")
    }
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
