//! `pkg_remove` — remove a package entry, scoped by `scope`:
//! `"project"` (default) removes from `alc.toml` + `alc.lock`;
//! `"global"` removes from `~/.algocline/installed.json`;
//! `"all"` removes from both.
//!
//! Physical files in `~/.algocline/packages/{name}/` are never deleted by any
//! scope. See `PkgRemoveScope` in the MCP layer for the enum definition and
//! CHANGELOG for the semantic difference from the historical 0.14.0 `scope`.

use super::super::alc_toml::{load_alc_toml_document, remove_package_entry, save_alc_toml};
use super::super::lockfile::{load_lockfile, lockfile_path, save_lockfile};
use super::super::manifest::{load_manifest, record_remove};
use super::super::project::resolve_project_root;
use super::super::AppService;

impl AppService {
    /// Remove a package entry scoped by `scope`. See module-level docs.
    ///
    /// Parameters:
    /// - `name`: package name to remove.
    /// - `project_root`: optional explicit project root. Required for
    ///   `"project"` / `"all"`; ignored for `"global"`.
    /// - `version`: optional version constraint (only affects `alc.lock`
    ///   removal in project scope; the global manifest is version-agnostic).
    /// - `scope`: `"project"` (default, back-compat), `"global"`, or `"all"`.
    ///   Any other value errors.
    pub async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        version: Option<String>,
        scope: Option<String>,
    ) -> Result<String, String> {
        let scope = scope.as_deref().unwrap_or("project");
        match scope {
            "project" => remove_from_project(name, project_root, version),
            "global" => remove_from_global(name),
            "all" => remove_from_all(name, project_root, version),
            other => Err(format!(
                "invalid scope '{other}': expected one of project, global, all"
            )),
        }
    }
}

/// Remove from `alc.toml` + `alc.lock`. Existing 0.15.0+ behavior.
fn remove_from_project(
    name: &str,
    project_root: Option<String>,
    version: Option<String>,
) -> Result<String, String> {
    let root = resolve_project_root(project_root.as_deref()).ok_or_else(|| {
        format!(
            "alc.toml not found: cannot remove '{name}' without a project root. \
             Provide project_root or run from a project directory."
        )
    })?;

    // alc.toml (best-effort: entry may already be gone).
    match load_alc_toml_document(&root)? {
        Some(mut doc) => {
            remove_package_entry(&mut doc, name);
            save_alc_toml(&root, &doc)?;
        }
        None => {
            return Err(format!("alc.toml not found at {}", root.display()));
        }
    }

    // alc.lock (authoritative: absence is an error so callers can't silently
    // no-op on a typo'd name).
    let alc_lock_path = lockfile_path(&root);
    match load_lockfile(&root)? {
        Some(mut lock) => {
            let before = lock.packages.len();
            lock.packages.retain(|p| {
                if p.name != name {
                    return true;
                }
                match &version {
                    Some(v) => p.version.as_deref() != Some(v.as_str()),
                    None => false,
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
        "scope": "project",
        "alc_toml": root.join("alc.toml").display().to_string(),
        "alc_lock": alc_lock_path.display().to_string(),
    })
    .to_string())
}

/// Remove from `~/.algocline/installed.json`. Physical `packages/{name}/` is
/// untouched — symmetric with the project scope's no-delete policy.
fn remove_from_global(name: &str) -> Result<String, String> {
    let manifest = load_manifest()?;
    if !manifest.packages.contains_key(name) {
        return Err(format!(
            "Package '{name}' not found in global manifest (~/.algocline/installed.json)"
        ));
    }

    record_remove(name)?;

    Ok(serde_json::json!({
        "removed": name,
        "scope": "global",
        "installed_json": manifest_path_display(),
    })
    .to_string())
}

/// Remove from both project and global. Lenient: success if either scope
/// had the entry; only errors when neither did.
fn remove_from_all(
    name: &str,
    project_root: Option<String>,
    version: Option<String>,
) -> Result<String, String> {
    let project_res = remove_from_project(name, project_root, version);
    let global_res = remove_from_global(name);

    let (project_ok, project_err) = match project_res {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e)),
    };
    let (global_ok, global_err) = match global_res {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e)),
    };

    if !project_ok && !global_ok {
        return Err(format!(
            "Package '{name}' not found in any scope:\n  project: {}\n  global: {}",
            project_err.unwrap_or_default(),
            global_err.unwrap_or_default()
        ));
    }

    Ok(serde_json::json!({
        "removed": name,
        "scope": "all",
        "project_removed": project_ok,
        "global_removed": global_ok,
        "project_note": project_err,
        "global_note": global_err,
    })
    .to_string())
}

/// Best-effort display string for `~/.algocline/installed.json`. Returns an
/// empty string if `$HOME` cannot be resolved — the caller surfaces this only
/// for informational JSON, never for correctness.
fn manifest_path_display() -> String {
    dirs::home_dir()
        .map(|h| {
            h.join(".algocline")
                .join("installed.json")
                .display()
                .to_string()
        })
        .unwrap_or_default()
}
