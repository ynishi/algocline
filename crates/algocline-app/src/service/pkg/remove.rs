//! `pkg_remove` — remove a package declaration from `alc.toml` and `alc.lock`.

use super::super::alc_toml::{load_alc_toml_document, remove_package_entry, save_alc_toml};
use super::super::lockfile::{load_lockfile, lockfile_path, save_lockfile};
use super::super::project::resolve_project_root;
use super::super::AppService;

impl AppService {
    /// Remove a package declaration from `alc.toml` and `alc.lock`.
    ///
    /// Requires an `alc.toml` to be found via `project_root` or ancestor walk.
    /// Physical files in `~/.algocline/packages/` are **not** deleted.
    ///
    /// - `name`: package name to remove.
    /// - `project_root`: optional explicit project root.
    /// - `version`: optional version constraint (when specified, only removes
    ///   the matching `alc.lock` entry; omit to remove any version).
    pub async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        version: Option<String>,
    ) -> Result<String, String> {
        // 1. Resolve project root via alc.toml ancestor walk.
        let root = resolve_project_root(project_root.as_deref()).ok_or_else(|| {
            format!(
                "alc.toml not found: cannot remove '{name}' without a project root. \
                 Provide project_root or run from a project directory."
            )
        })?;

        // 2. Remove from alc.toml (best-effort: entry may not exist if manually removed).
        match load_alc_toml_document(&root)? {
            Some(mut doc) => {
                remove_package_entry(&mut doc, name);
                save_alc_toml(&root, &doc)?;
            }
            None => {
                return Err(format!("alc.toml not found at {}", root.display()));
            }
        }

        // 3. Remove from alc.lock.
        let alc_lock_path = lockfile_path(&root);
        match load_lockfile(&root)? {
            Some(mut lock) => {
                let before = lock.packages.len();
                lock.packages.retain(|p| {
                    if p.name != name {
                        return true; // keep
                    }
                    // name matches: if version is specified, only remove matching version
                    match &version {
                        Some(v) => p.version.as_deref() != Some(v.as_str()),
                        None => false, // remove all with this name
                    }
                });

                if lock.packages.len() == before {
                    return Err(format!(
                        "Package '{name}' not found in alc.lock at {}",
                        alc_lock_path.display()
                    ));
                }

                save_lockfile(&root, &lock)?;
            }
            None => {
                return Err(format!(
                    "Package '{name}' not found in alc.lock at {}",
                    alc_lock_path.display()
                ));
            }
        }

        Ok(serde_json::json!({
            "removed": name,
            "alc_toml": root.join("alc.toml").display().to_string(),
            "alc_lock": alc_lock_path.display().to_string(),
        })
        .to_string())
    }
}
