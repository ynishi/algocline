//! `alc_pkg_link` — link a local directory as a project-local package.
//!
//! Unlike `pkg_install` (copy-based), `pkg_link` records the directory path in
//! `alc.lock` without copying any files. The linked directory is resolved at
//! `alc_run` time via `FsResolver` using `extra_lib_paths`.

use std::path::Path;

use super::lockfile::{load_lockfile, lockfile_path, save_lockfile, LockFile, LockPackage};
use super::manifest::now_iso8601;
use super::project::resolve_project_root;
use super::source::PackageSource;
use super::AppService;

impl AppService {
    /// Link a local directory as a project-local package (no copy).
    ///
    /// `path`: source directory — single package (has `init.lua`) or collection
    /// (subdirectories have `init.lua`). May be absolute or relative to `project_root`.
    ///
    /// `project_root`: optional explicit project root (where `alc.lock` lives).
    /// Falls back to `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    pub async fn pkg_link(
        &self,
        path: String,
        project_root: Option<String>,
    ) -> Result<String, String> {
        // 1. Resolve project root.
        let root = resolve_project_root(project_root.as_deref()).ok_or_else(|| {
            "Cannot determine project root: provide project_root or set ALC_PROJECT_ROOT"
                .to_string()
        })?;

        // 2. Resolve path (absolute: use as-is, relative: join with project_root).
        let raw_path = Path::new(&path);
        let resolved_path = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            root.join(raw_path)
        };

        if !resolved_path.is_dir() {
            return Err(format!(
                "Path is not a directory: {}",
                resolved_path.display()
            ));
        }

        // 3. Determine mode: single (init.lua at root) or collection (subdirs with init.lua).
        let mode = detect_mode(&resolved_path)?;

        // 4. Load or create alc.lock.
        let mut lock = match load_lockfile(&root)? {
            Some(existing) => existing,
            None => LockFile {
                version: 1,
                packages: Vec::new(),
            },
        };

        // 5. Build entries and upsert into lock.
        let now = now_iso8601();
        let linked_names = match mode {
            PackageMode::Single => {
                let name = resolved_path
                    .file_name()
                    .ok_or_else(|| {
                        format!(
                            "Cannot determine package name from path: {}",
                            resolved_path.display()
                        )
                    })?
                    .to_string_lossy()
                    .to_string();

                let stored_path = relative_or_absolute_path(&resolved_path, &root);
                upsert_lock_entry(&mut lock, name.clone(), stored_path, now);
                vec![name]
            }
            PackageMode::Collection => {
                let entries = std::fs::read_dir(&resolved_path).map_err(|e| {
                    format!("Failed to read directory {}: {e}", resolved_path.display())
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
                    let stored_path = relative_or_absolute_path(&pkg_path, &root);
                    upsert_lock_entry(&mut lock, name.clone(), stored_path, now.clone());
                    names.push(name);
                }

                if names.is_empty() {
                    return Err(format!(
                        "No init.lua found in any subdirectory of: {}",
                        resolved_path.display()
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
/// If an entry with the same `name` already exists, updates `linked_at` and
/// the `path` inside `PackageSource::LocalDir`. Otherwise appends a new entry.
fn upsert_lock_entry(lock: &mut LockFile, name: String, path: String, linked_at: String) {
    if let Some(existing) = lock.packages.iter_mut().find(|p| p.name == name) {
        existing.source = PackageSource::LocalDir { path };
        existing.linked_at = linked_at;
    } else {
        lock.packages.push(LockPackage {
            name,
            source: PackageSource::LocalDir { path },
            linked_at,
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

        // Create a single-package dir.
        let pkg_dir = project_root.join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                pkg_dir.to_string_lossy().to_string(),
                Some(project_root.to_string_lossy().to_string()),
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["mode"], "single");
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));

        // Verify alc.lock was written.
        let lock = load_lockfile(project_root).unwrap().unwrap();
        assert_eq!(lock.packages.len(), 1);
        assert_eq!(lock.packages[0].name, "my_pkg");
        assert!(matches!(
            &lock.packages[0].source,
            PackageSource::LocalDir { .. }
        ));
    }

    #[tokio::test]
    async fn pkg_link_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create a collection dir with two packages.
        let collection = project_root.join("collection");
        std::fs::create_dir_all(collection.join("pkg_a")).unwrap();
        std::fs::create_dir_all(collection.join("pkg_b")).unwrap();
        std::fs::write(collection.join("pkg_a").join("init.lua"), "return {}").unwrap();
        std::fs::write(collection.join("pkg_b").join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                collection.to_string_lossy().to_string(),
                Some(project_root.to_string_lossy().to_string()),
            )
            .await
            .unwrap();

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

        let pkg_dir = project_root.join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;

        // Link once.
        svc.pkg_link(
            pkg_dir.to_string_lossy().to_string(),
            Some(project_root.to_string_lossy().to_string()),
        )
        .await
        .unwrap();

        let lock1 = load_lockfile(project_root).unwrap().unwrap();
        let first_linked_at = lock1.packages[0].linked_at.clone();

        // Small sleep to ensure timestamp can differ.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Link again (same path).
        svc.pkg_link(
            pkg_dir.to_string_lossy().to_string(),
            Some(project_root.to_string_lossy().to_string()),
        )
        .await
        .unwrap();

        let lock2 = load_lockfile(project_root).unwrap().unwrap();
        // Only one entry (no duplicate).
        assert_eq!(lock2.packages.len(), 1);
        // linked_at must be updated.
        // (In practice the timestamp has 1-second resolution; we just verify
        // the field exists and is non-empty. A precise comparison would be
        // flaky depending on system clock resolution.)
        assert!(!lock2.packages[0].linked_at.is_empty());
        // The field should be >= first_linked_at (monotonic).
        assert!(lock2.packages[0].linked_at >= first_linked_at);
    }

    #[tokio::test]
    async fn pkg_link_no_project_root_returns_error() {
        // When no project_root is given AND there is no ALC_PROJECT_ROOT env
        // AND cwd has no alc.lock ancestors, resolve_project_root may return
        // Some(cwd). We explicitly pass an invalid dir to ensure we hit Err.
        let tmp = tempfile::tempdir().unwrap();
        let non_dir = tmp.path().join("does_not_exist");

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                non_dir.to_string_lossy().to_string(),
                Some(tmp.path().to_string_lossy().to_string()),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a directory"));
    }
}
