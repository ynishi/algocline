//! `pkg_remove` — remove a package from project scope (alc.lock) or global scope (filesystem).

use super::super::lockfile::{load_lockfile, lockfile_path, save_lockfile};
use super::super::manifest;
use super::super::path::ContainedPath;
use super::super::project::resolve_project_root;
use super::super::resolve::packages_dir;
use super::super::AppService;

impl AppService {
    /// Remove an installed package.
    ///
    /// - `project_root` specified or `scope == "project"`: removes the entry
    ///   from `alc.lock` only. The physical files are **not** deleted.
    /// - `scope == "global"`: removes from `~/.algocline/packages/` (existing
    ///   behavior), even if the package exists in `alc.lock`.
    /// - Default (no scope, no project_root): tries project first, falls back
    ///   to global.
    pub async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        scope: Option<String>,
    ) -> Result<String, String> {
        let effective_scope =
            determine_remove_scope(name, project_root.as_deref(), scope.as_deref());

        match effective_scope {
            RemoveScope::Project(root) => {
                // Remove from alc.lock only; do NOT delete physical files.
                let mut lock = match load_lockfile(&root)? {
                    Some(l) => l,
                    None => {
                        return Err(format!(
                            "Package '{name}' not found in project (no alc.lock at {})",
                            root.display()
                        ));
                    }
                };

                let before = lock.packages.len();
                lock.packages.retain(|p| p.name != name);

                if lock.packages.len() == before {
                    return Err(format!(
                        "Package '{name}' not found in alc.lock at {}",
                        root.display()
                    ));
                }

                save_lockfile(&root, &lock)?;

                Ok(serde_json::json!({
                    "removed": name,
                    "scope": "project",
                    "lockfile": lockfile_path(&root).display().to_string(),
                })
                .to_string())
            }
            RemoveScope::Global => {
                // Original behavior: delete physical files.
                let pkg_dir = packages_dir()?;
                let dest = ContainedPath::child(&pkg_dir, name)?;

                if !dest.as_ref().exists() {
                    return Err(format!("Package '{name}' not found"));
                }

                std::fs::remove_dir_all(&dest)
                    .map_err(|e| format!("Failed to remove '{name}': {e}"))?;

                // Remove from manifest (best-effort)
                let _ = manifest::record_remove(name);

                Ok(serde_json::json!({ "removed": name, "scope": "global" }).to_string())
            }
        }
    }
}

// ─── Remove scope resolution ─────────────────────────────────────

enum RemoveScope {
    Project(std::path::PathBuf),
    Global,
}

/// Determine the effective remove scope.
///
/// Priority:
/// 1. `scope == "global"` → always Global (ignores project_root).
/// 2. `project_root` is `Some` → Project (use that root).
/// 3. `scope == "project"` → Project (auto-detect root, must find package).
/// 4. Auto-detect: resolve root; if alc.lock contains the package → Project,
///    otherwise → Global.
fn determine_remove_scope(
    name: &str,
    project_root: Option<&str>,
    scope: Option<&str>,
) -> RemoveScope {
    // Explicit global scope — skip all project logic.
    if scope == Some("global") {
        return RemoveScope::Global;
    }

    let root = match resolve_project_root(project_root) {
        Some(r) => r,
        None => return RemoveScope::Global,
    };

    // Explicit project_root provided → always project scope.
    if project_root.is_some() {
        return RemoveScope::Project(root);
    }

    // Explicit scope == "project" → project scope (auto-detected root).
    if scope == Some("project") {
        return RemoveScope::Project(root);
    }

    // Auto-detection: only use project scope when the package is actually
    // listed in alc.lock.
    match load_lockfile(&root) {
        Ok(Some(lock)) if lock.packages.iter().any(|p| p.name == name) => {
            RemoveScope::Project(root)
        }
        _ => RemoveScope::Global,
    }
}
