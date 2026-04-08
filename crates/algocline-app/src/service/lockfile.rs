//! `alc.lock` — project-local package declarations.
//!
//! ## File location
//! `alc.lock` lives at the project root (the directory passed as `project_root`).
//!
//! ## Path resolution base
//! Relative paths in `PackageSource::LocalDir.path` and `PackageSource::LocalCopy.path`
//! are resolved relative to the `alc.lock` file location (= project root).
//! Absolute paths are used as-is.
//!
//! ## Version compatibility
//! `version` must equal 1. Any other value causes `load_lockfile` to return `Err`.
//!
//! ## Schema example
//! ```toml
//! version = 1
//!
//! [[package]]
//! name = "head_agent"
//! linked_at = "2026-04-08T12:00:00Z"
//!
//! [package.source]
//! type = "local_dir"
//! path = "packages/head_agent"
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::source::PackageSource;

// ─── Types ─────────────────────────────────────────────────────────────────

/// Top-level structure of `alc.lock`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LockFile {
    /// Schema version. Must be 1.
    pub version: u32,
    /// Declared packages. Serialised as `[[package]]` in TOML.
    #[serde(default, rename = "package")]
    pub packages: Vec<LockPackage>,
}

/// A single package entry in `alc.lock`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LockPackage {
    /// Package name (must match the Lua module name).
    pub name: String,
    /// How the package is sourced.
    pub source: PackageSource,
    /// ISO 8601 timestamp of when this entry was added/updated.
    pub linked_at: String,
}

// ─── Paths ──────────────────────────────────────────────────────────────────

/// Returns the canonical path to `alc.lock` for the given project root.
pub(crate) fn lockfile_path(project_root: &Path) -> PathBuf {
    project_root.join("alc.lock")
}

// ─── Read / Write ────────────────────────────────────────────────────────────

/// Load `alc.lock` from disk.
///
/// Returns `Ok(None)` if the file does not exist.
/// Returns `Err` if the file exists but cannot be parsed, or if `version != 1`.
pub(crate) fn load_lockfile(project_root: &Path) -> Result<Option<LockFile>, String> {
    let path = lockfile_path(project_root);
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read alc.lock at {}: {e}", path.display()))?;

    let lock: LockFile = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse alc.lock at {}: {e}", path.display()))?;

    if lock.version != 1 {
        return Err(format!(
            "unsupported alc.lock version {}: expected 1",
            lock.version
        ));
    }

    Ok(Some(lock))
}

/// Write `alc.lock` to disk (pretty-printed TOML).
///
/// Creates the parent directory if necessary.
pub(crate) fn save_lockfile(project_root: &Path, lock: &LockFile) -> Result<(), String> {
    let path = lockfile_path(project_root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory for alc.lock: {e}"))?;
    }

    let content =
        toml::to_string_pretty(lock).map_err(|e| format!("Failed to serialize alc.lock: {e}"))?;

    std::fs::write(&path, content)
        .map_err(|e| format!("Failed to write alc.lock at {}: {e}", path.display()))
}

// ─── Resolution ─────────────────────────────────────────────────────────────

/// Extract resolved absolute paths for all `LocalDir` entries in the lock file.
///
/// - Relative paths are resolved against `project_root`.
/// - Absolute paths are used as-is.
/// - Entries whose resolved path does not exist are skipped with a warning.
pub(crate) fn resolve_local_dir_paths(project_root: &Path, lock: &LockFile) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for pkg in &lock.packages {
        let PackageSource::LocalDir { path: ref raw } = pkg.source else {
            continue;
        };

        let resolved = {
            let p = Path::new(raw);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                project_root.join(p)
            }
        };

        if !resolved.exists() {
            eprintln!(
                "alc.lock: local_dir path for '{}' does not exist, skipping: {}",
                pkg.name,
                resolved.display()
            );
            continue;
        }

        paths.push(resolved);
    }

    paths
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::source::PackageSource;

    fn make_local_dir_lock(path: &str) -> LockFile {
        LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: "test_pkg".to_string(),
                source: PackageSource::LocalDir {
                    path: path.to_string(),
                },
                linked_at: "2026-04-08T12:00:00Z".to_string(),
            }],
        }
    }

    #[test]
    fn lockfile_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        let original = LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: "head_agent".to_string(),
                source: PackageSource::LocalDir {
                    path: "packages/head_agent".to_string(),
                },
                linked_at: "2026-04-08T12:00:00Z".to_string(),
            }],
        };

        save_lockfile(project_root, &original).unwrap();
        let loaded = load_lockfile(project_root).unwrap();

        assert_eq!(loaded, Some(original));
    }

    #[test]
    fn lockfile_version_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("alc.lock");

        std::fs::write(
            &path,
            r#"version = 2

[[package]]
name = "foo"
linked_at = "2026-04-08T00:00:00Z"

[package.source]
type = "local_dir"
path = "packages/foo"
"#,
        )
        .unwrap();

        let result = load_lockfile(tmp.path());
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("unsupported alc.lock version 2"), "{msg}");
    }

    #[test]
    fn lockfile_missing() {
        let tmp = tempfile::tempdir().unwrap();
        // No alc.lock created.
        let result = load_lockfile(tmp.path()).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_local_dir_relative() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create the subdirectory that the relative path points to.
        let pkg_dir = project_root.join("packages").join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();

        let lock = make_local_dir_lock("packages/my_pkg");
        let paths = resolve_local_dir_paths(project_root, &lock);

        assert_eq!(paths, vec![pkg_dir]);
    }

    #[test]
    fn resolve_local_dir_absolute() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create the target directory at an absolute path (in a different tempdir).
        let abs_tmp = tempfile::tempdir().unwrap();
        let abs_path = abs_tmp.path().to_path_buf();

        let lock = make_local_dir_lock(abs_path.to_str().unwrap());
        let paths = resolve_local_dir_paths(project_root, &lock);

        assert_eq!(paths, vec![abs_path]);
    }

    #[test]
    fn resolve_local_dir_skip_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Path does not exist — should be skipped silently.
        let lock = make_local_dir_lock("nonexistent/path");
        let paths = resolve_local_dir_paths(project_root, &lock);

        assert!(paths.is_empty());
    }
}
