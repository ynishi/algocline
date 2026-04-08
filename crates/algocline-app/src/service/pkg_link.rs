//! `alc_pkg_link` — link a local directory as a project-local package.
//!
//! Unlike `pkg_install` (copy-based), `pkg_link` records the directory path in
//! `alc.lock` without copying any files. The linked directory is resolved at
//! `alc_run` time via `FsResolver` using `extra_lib_paths`.

use std::path::Path;

use super::lockfile::{load_lockfile, lockfile_path, save_lockfile, LockFile, LockPackage};
use super::project::resolve_project_root;
use super::source::PackageSource;
use super::AppService;

impl AppService {
    /// Link a local directory as a project-local package (symlink to cache).
    ///
    /// Phase 2 (subtask 5) will rewrite this with full symlink logic.
    /// This version preserves minimal backward-compat using the legacy `project_root`-based
    /// path recording (to be removed in Phase 2).
    ///
    /// `path`: source directory to link.
    /// `name`: optional package name override.
    /// `force`: optional flag to force overwrite of existing symlinks.
    pub async fn pkg_link(
        &self,
        path: String,
        _name: Option<String>,
        _force: Option<bool>,
    ) -> Result<String, String> {
        // Phase 2 will implement symlink-to-cache logic.
        // Interim: resolve project root from cwd/env ancestor walk.
        let root = resolve_project_root(None).ok_or_else(|| {
            "Cannot determine project root: alc.toml not found in current directory or any ancestor. \
             Set ALC_PROJECT_ROOT or run from a project directory."
                .to_string()
        })?;

        // 2. Resolve path (absolute: use as-is, relative: join with root).
        let raw_path = Path::new(&path);
        let canon_path = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            root.join(raw_path)
        };

        if !canon_path.is_dir() {
            return Err(format!("Path is not a directory: {}", canon_path.display()));
        }

        // Containment check: the linked directory must live under the project
        // root. Symlinks are resolved via `canonicalize` so an in-tree symlink
        // pointing outside the project is also rejected. This prevents an
        // `alc.lock` entry from exposing arbitrary filesystem paths (e.g.
        // `/etc`, `../../..`) as Lua package search locations.
        let canon_root = std::fs::canonicalize(&root)
            .map_err(|e| format!("Cannot canonicalize project_root {}: {e}", root.display()))?;
        let canon_path = std::fs::canonicalize(&canon_path)
            .map_err(|e| format!("Cannot canonicalize path {}: {e}", canon_path.display()))?;
        if !canon_path.starts_with(&canon_root) {
            return Err(format!(
                "Path must be inside project_root ({}): {}",
                canon_root.display(),
                canon_path.display()
            ));
        }

        // 3. Determine mode: single (init.lua at root) or collection (subdirs with init.lua).
        let mode = detect_mode(&canon_path)?;

        // 4. Load or create alc.lock.
        let mut lock = match load_lockfile(&root)? {
            Some(existing) => existing,
            None => LockFile {
                version: 1,
                packages: Vec::new(),
            },
        };

        // 5. Build entries and upsert into lock.
        let linked_names = match mode {
            PackageMode::Single => {
                let name = canon_path
                    .file_name()
                    .ok_or_else(|| {
                        format!(
                            "Cannot determine package name from path: {}",
                            canon_path.display()
                        )
                    })?
                    .to_string_lossy()
                    .to_string();

                let stored_path = relative_or_absolute_path(&canon_path, &canon_root);
                upsert_lock_entry(&mut lock, name.clone(), stored_path);
                vec![name]
            }
            PackageMode::Collection => {
                let entries = std::fs::read_dir(&canon_path).map_err(|e| {
                    format!("Failed to read directory {}: {e}", canon_path.display())
                })?;

                let mut names = Vec::new();
                for entry in entries {
                    let entry =
                        entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
                    let pkg_path = entry.path();
                    if !pkg_path.is_dir() {
                        continue;
                    }
                    if !pkg_path.join("init.lua").exists() {
                        continue;
                    }
                    let name = entry.file_name().to_string_lossy().to_string();
                    let stored_path = relative_or_absolute_path(&pkg_path, &canon_root);
                    upsert_lock_entry(&mut lock, name.clone(), stored_path);
                    names.push(name);
                }

                if names.is_empty() {
                    return Err(format!(
                        "No init.lua found in any subdirectory of: {}",
                        canon_path.display()
                    ));
                }

                names.sort();
                names
            }
        };

        // 6. Save alc.lock.
        save_lockfile(&root, &lock)?;

        // 7. Return result.
        let mode_str = match mode {
            PackageMode::Single => "single",
            PackageMode::Collection => "collection",
        };

        Ok(serde_json::json!({
            "linked": linked_names,
            "mode": mode_str,
            "lockfile": lockfile_path(&root).display().to_string(),
        })
        .to_string())
    }
}

// ─── Internal helpers ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum PackageMode {
    Single,
    Collection,
}

/// Determine whether `path` is a single package or a collection.
fn detect_mode(path: &Path) -> Result<PackageMode, String> {
    if path.join("init.lua").exists() {
        return Ok(PackageMode::Single);
    }

    // Check if any subdirectory has an init.lua.
    let entries = std::fs::read_dir(path).map_err(|e| format!("Failed to read directory: {e}"))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let sub = entry.path();
        if sub.is_dir() && sub.join("init.lua").exists() {
            return Ok(PackageMode::Collection);
        }
    }

    Err(format!(
        "No init.lua found in {} or any of its subdirectories",
        path.display()
    ))
}

/// Return `path` as a relative string from `base` if possible, otherwise absolute.
///
/// Uses `strip_prefix` to relativize. If the paths cannot be made relative
/// (e.g. different mount points, or canonicalization introduced symlink
/// resolution mismatch), falls back to the absolute string.
fn relative_or_absolute_path(path: &Path, base: &Path) -> String {
    match path.strip_prefix(base) {
        Ok(rel) => rel.to_string_lossy().to_string(),
        Err(_) => path.to_string_lossy().to_string(),
    }
}

/// Insert or update a `LockPackage` entry.
///
/// If an entry with the same `name` already exists, updates the `path` inside
/// `PackageSource::Path`. Otherwise appends a new entry.
fn upsert_lock_entry(lock: &mut LockFile, name: String, path: String) {
    if let Some(existing) = lock.packages.iter_mut().find(|p| p.name == name) {
        existing.source = PackageSource::Path { path };
    } else {
        lock.packages.push(LockPackage {
            name,
            version: None,
            source: PackageSource::Path { path },
        });
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::service::lockfile::load_lockfile;

    /// Build a minimal AppService for tests.
    async fn make_app_service() -> AppService {
        let executor = Arc::new(
            algocline_engine::Executor::new(vec![])
                .await
                .expect("executor"),
        );
        AppService {
            executor,
            registry: Arc::new(algocline_engine::SessionRegistry::new()),
            log_config: crate::service::config::AppConfig {
                log_dir: None,
                log_dir_source: crate::service::config::LogDirSource::None,
                log_enabled: false,
            },
            search_paths: vec![],
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    #[tokio::test]
    async fn pkg_link_single() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Phase 2 note: pkg_link resolves project root via alc.toml ancestor walk.
        // Create alc.toml so resolve_project_root finds this directory.
        std::fs::write(project_root.join("alc.toml"), "[packages]\n").unwrap();

        // Create a single-package dir inside the project root.
        let pkg_dir = project_root.join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        // ALC_PROJECT_ROOT makes resolve_project_root deterministic without chdir.
        std::env::set_var("ALC_PROJECT_ROOT", project_root.to_string_lossy().as_ref());
        let svc = make_app_service().await;
        let result = svc
            .pkg_link(pkg_dir.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();
        std::env::remove_var("ALC_PROJECT_ROOT");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["mode"], "single");
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));

        // Verify alc.lock was written.
        let lock = load_lockfile(project_root).unwrap().unwrap();
        assert_eq!(lock.packages.len(), 1);
        assert_eq!(lock.packages[0].name, "my_pkg");
        assert!(matches!(
            &lock.packages[0].source,
            PackageSource::Path { .. }
        ));
    }

    #[tokio::test]
    async fn pkg_link_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        std::fs::write(project_root.join("alc.toml"), "[packages]\n").unwrap();

        // Create a collection dir with two packages.
        let collection = project_root.join("collection");
        std::fs::create_dir_all(collection.join("pkg_a")).unwrap();
        std::fs::create_dir_all(collection.join("pkg_b")).unwrap();
        std::fs::write(collection.join("pkg_a").join("init.lua"), "return {}").unwrap();
        std::fs::write(collection.join("pkg_b").join("init.lua"), "return {}").unwrap();

        std::env::set_var("ALC_PROJECT_ROOT", project_root.to_string_lossy().as_ref());
        let svc = make_app_service().await;
        let result = svc
            .pkg_link(collection.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();
        std::env::remove_var("ALC_PROJECT_ROOT");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["mode"], "collection");

        let linked = json["linked"].as_array().unwrap();
        let mut names: Vec<&str> = linked.iter().map(|v| v.as_str().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["pkg_a", "pkg_b"]);

        // Verify alc.lock has both entries.
        let lock = load_lockfile(project_root).unwrap().unwrap();
        assert_eq!(lock.packages.len(), 2);
    }

    #[tokio::test]
    async fn pkg_link_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        std::fs::write(project_root.join("alc.toml"), "[packages]\n").unwrap();

        let pkg_dir = project_root.join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        std::env::set_var("ALC_PROJECT_ROOT", project_root.to_string_lossy().as_ref());
        let svc = make_app_service().await;

        // Link once.
        svc.pkg_link(pkg_dir.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let lock1 = load_lockfile(project_root).unwrap().unwrap();
        assert_eq!(lock1.packages.len(), 1);

        // Link again (same path).
        svc.pkg_link(pkg_dir.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let lock2 = load_lockfile(project_root).unwrap().unwrap();
        // Only one entry (no duplicate).
        assert_eq!(lock2.packages.len(), 1);
        std::env::remove_var("ALC_PROJECT_ROOT");
    }

    #[tokio::test]
    async fn pkg_link_path_not_found_returns_error() {
        // When project root is set but the given path does not exist, expect an error.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alc.toml"), "[packages]\n").unwrap();
        let non_dir = tmp.path().join("does_not_exist");

        std::env::set_var("ALC_PROJECT_ROOT", tmp.path().to_string_lossy().as_ref());
        let svc = make_app_service().await;
        let result = svc
            .pkg_link(non_dir.to_string_lossy().to_string(), None, None)
            .await;
        std::env::remove_var("ALC_PROJECT_ROOT");

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a directory"));
    }
}
