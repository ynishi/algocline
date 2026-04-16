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

use std::path::Path;

use super::super::alc_toml::{self, PackageDep};
use super::super::lockfile::load_lockfile;
use super::super::manifest::{load_manifest, ManifestEntry};
use super::super::project::resolve_project_root;
use super::super::resolve::packages_dir;
use super::super::source::PackageSource;
use super::super::AppService;

/// Outcome of repairing a single package.
enum RepairOutcome {
    /// Successfully reinstalled from `source`.
    Repaired { source: String },
    /// Package is healthy — nothing to do.
    Skipped,
    /// Cannot repair automatically — user must intervene.
    Unrepairable { reason: String, suggestion: String },
    /// Repair was attempted but failed.
    Failed { reason: String },
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

        let mut repaired: Vec<serde_json::Value> = Vec::new();
        let mut skipped: Vec<serde_json::Value> = Vec::new();
        let mut unrepairable: Vec<serde_json::Value> = Vec::new();
        let mut failed: Vec<serde_json::Value> = Vec::new();

        let target_filter = name.as_deref();

        // ── (B) installed pkgs from manifest ──────────────────────
        for (pkg_name, entry) in &manifest.packages {
            if let Some(target) = target_filter {
                if target != pkg_name.as_str() {
                    continue;
                }
            }

            match self.repair_installed(pkg_name, entry, &pkg_dir).await {
                RepairOutcome::Repaired { source } => repaired.push(serde_json::json!({
                    "name": pkg_name,
                    "kind": "installed_missing",
                    "action": "reinstall",
                    "source": source,
                })),
                RepairOutcome::Skipped => skipped.push(serde_json::json!({
                    "name": pkg_name,
                    "reason": "healthy",
                })),
                RepairOutcome::Unrepairable { reason, suggestion } => {
                    unrepairable.push(serde_json::json!({
                        "name": pkg_name,
                        "kind": "installed_missing",
                        "reason": reason,
                        "suggestion": suggestion,
                    }));
                }
                RepairOutcome::Failed { reason } => failed.push(serde_json::json!({
                    "name": pkg_name,
                    "kind": "installed_missing",
                    "reason": reason,
                })),
            }
        }

        // ── (A) global symlinks dangling — surface as unrepairable ──
        if let Ok(read) = std::fs::read_dir(&pkg_dir) {
            for dir_entry in read.flatten() {
                let path = dir_entry.path();
                let pkg_name = dir_entry.file_name().to_string_lossy().to_string();

                if let Some(target) = target_filter {
                    if target != pkg_name.as_str() {
                        continue;
                    }
                }
                // Already covered by manifest pass.
                if manifest.packages.contains_key(&pkg_name) {
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
                    "suggestion": format!(
                        "alc_pkg_unlink({pkg_name:?}) then alc_pkg_link with the new path"
                    ),
                }));
            }
        }

        // ── (C) project `path = ...` missing ──────────────────────
        // ── (D) variant `path = ...` missing ──────────────────────
        if let Some(root) = resolved_root.as_ref() {
            collect_path_missing(
                root,
                target_filter,
                "project",
                &mut unrepairable,
                ProjectPathSource::Toml,
            );
            collect_path_missing(
                root,
                target_filter,
                "variant",
                &mut unrepairable,
                ProjectPathSource::Local,
            );
        }

        // ── target_filter sanity: nothing matched at all ─────────
        if let Some(target) = target_filter {
            let any_matched = !repaired.is_empty()
                || !skipped.is_empty()
                || !unrepairable.is_empty()
                || !failed.is_empty();
            if !any_matched {
                return Err(format!(
                    "Package '{target}' not found in installed.json, ~/.algocline/packages/, alc.toml, or alc.local.toml"
                ));
            }
        }

        Ok(serde_json::json!({
            "repaired": repaired,
            "skipped": skipped,
            "unrepairable": unrepairable,
            "failed": failed,
        })
        .to_string())
    }

    /// Attempt to repair a single manifest-tracked package by re-running
    /// `pkg_install` with the recorded `source`. Returns `Skipped` when the
    /// package directory already exists (healthy).
    async fn repair_installed(
        &self,
        name: &str,
        entry: &ManifestEntry,
        pkg_dir: &Path,
    ) -> RepairOutcome {
        let dest = pkg_dir.join(name);

        // If dest is a symlink (live or dangling), it's a pkg_link case and
        // we don't touch it here — it will be handled by the (A) pass.
        let is_symlink = dest
            .symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if is_symlink {
            return RepairOutcome::Skipped;
        }

        if dest.exists() {
            return RepairOutcome::Skipped;
        }

        // Source classification: only `Installed` (local copy) and `Git` can be
        // re-fetched. Bundled is conceptually re-installable via `init` but
        // out of scope for `pkg_repair`. Path is not tracked in manifest.
        let inferred = super::super::source::infer_from_legacy_source_string(&entry.source);
        match inferred {
            PackageSource::Installed | PackageSource::Git { .. } => {
                match self
                    .pkg_install(entry.source.clone(), Some(name.to_string()))
                    .await
                {
                    Ok(_) => RepairOutcome::Repaired {
                        source: entry.source.clone(),
                    },
                    Err(e) => RepairOutcome::Failed { reason: e },
                }
            }
            PackageSource::Bundled { .. } => RepairOutcome::Unrepairable {
                reason: "bundled package — restore via `alc_init` or reinstall algocline"
                    .to_string(),
                suggestion: format!("alc_pkg_install({:?})", entry.source),
            },
            PackageSource::Path { path } => RepairOutcome::Unrepairable {
                reason: format!("path source ({path}) — not tracked in manifest for repair"),
                suggestion: "edit alc.toml or alc.local.toml directly".to_string(),
            },
        }
    }
}

/// Which TOML file is the source of truth for path entries.
#[derive(Debug, Clone, Copy)]
enum ProjectPathSource {
    /// `alc.toml` `[packages.x] path = ...` (project scope).
    Toml,
    /// `alc.local.toml` `[packages.x] path = ...` (variant scope).
    Local,
}

/// Append `path_missing` unrepairable entries for either alc.toml or
/// alc.local.toml. Filtering by `target_filter` (Some(name)) restricts
/// to a single package.
fn collect_path_missing(
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
            ProjectPathSource::Local => format!(
                "alc_pkg_unlink({name:?}) or update [packages.{name}] in alc.local.toml"
            ),
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

