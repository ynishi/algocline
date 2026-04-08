//! `alc.lock` — project-local package declarations.
//!
//! ## File location
//! `alc.lock` lives at the project root (the directory passed as `project_root`).
//!
//! ## Path resolution base
//! Relative paths in `PackageSource::Path.path` are resolved relative to the
//! `alc.lock` file location (= project root). Absolute paths are used as-is.
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
//! version = "0.3.0"
//!
//! [package.source]
//! type = "path"
//! path = "packages/head_agent"
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Process-wide guard serialising `save_lockfile` callers.
///
/// Cross-process races (an external editor writing while we rename) remain
/// possible — the rename itself is atomic, but concurrent read-modify-write
/// sequences across processes can still lose changes. Within the MCP daemon
/// this mutex guarantees no in-process concurrent writers trample each other.
static SAVE_GUARD: Mutex<()> = Mutex::new(());

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
    /// Package version (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// How the package is sourced.
    pub source: PackageSource,
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
/// Writes via a temp file in the same directory and `rename`s into place so
/// readers never observe a half-written file. Creates the parent directory
/// if necessary.
///
/// In-process concurrent writers are serialised through [`SAVE_GUARD`]. This
/// is **not** a cross-process lock: an external editor writing concurrently
/// can still cause lost updates. For the intended use-case (interactive
/// `alc_pkg_link` inside a single MCP daemon) the in-process guard suffices.
pub(crate) fn save_lockfile(project_root: &Path, lock: &LockFile) -> Result<(), String> {
    // Held for the duration of the write. `unwrap_or_else` on poison keeps
    // progress — the mutex protects only serialisation, not shared state.
    let _guard = SAVE_GUARD.lock().unwrap_or_else(|p| p.into_inner());

    let path = lockfile_path(project_root);
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Cannot determine parent directory for alc.lock at {}",
            path.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create directory for alc.lock: {e}"))?;

    let content =
        toml::to_string_pretty(lock).map_err(|e| format!("Failed to serialize alc.lock: {e}"))?;

    // Write to a sibling temp file, then atomically rename.
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|e| format!("Failed to create temp file for alc.lock: {e}"))?;
    {
        use std::io::Write;
        tmp.write_all(content.as_bytes())
            .map_err(|e| format!("Failed to write alc.lock staging: {e}"))?;
        tmp.as_file()
            .sync_all()
            .map_err(|e| format!("Failed to fsync alc.lock staging: {e}"))?;
    }
    tmp.persist(&path)
        .map_err(|e| format!("Failed to persist alc.lock at {}: {e}", path.display()))?;
    Ok(())
}

// ─── Resolution ─────────────────────────────────────────────────────────────

/// Extract resolved absolute paths for all `Path` entries in the lock file.
///
/// - Relative paths are resolved against `project_root`.
/// - Absolute paths are used as-is.
/// - Entries whose resolved path does not exist are skipped with a warning.
pub(crate) fn resolve_path_entries(project_root: &Path, lock: &LockFile) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for pkg in &lock.packages {
        let PackageSource::Path { path: ref raw } = pkg.source else {
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
            tracing::warn!(
                "alc.lock: path entry for '{}' does not exist, skipping: {}",
                pkg.name,
                resolved.display()
            );
            continue;
        }

        paths.push(resolved);
    }

    paths
}

/// Extract resolved absolute paths for all `Installed` entries in the lock file.
// Used in subtask 2+ (resolve_extra_lib_paths for Installed packages)
#[allow(dead_code)]
///
/// Derives the package path as `~/.algocline/packages/{name}` or
/// `~/.algocline/packages/{name}@{version}` when a version is present.
/// Entries whose resolved path does not exist are skipped with a warning.
pub(crate) fn resolve_installed_paths(lock: &LockFile) -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        tracing::warn!("alc.lock: cannot determine home directory for Installed path resolution");
        return vec![];
    };
    let packages_dir = home.join(".algocline").join("packages");

    let mut paths = Vec::new();
    for pkg in &lock.packages {
        if !matches!(pkg.source, PackageSource::Installed) {
            continue;
        }

        let dir_name = match &pkg.version {
            Some(v) => format!("{}@{}", pkg.name, v),
            None => pkg.name.clone(),
        };
        let resolved = packages_dir.join(&dir_name);

        if !resolved.exists() {
            tracing::warn!(
                "alc.lock: installed path for '{}' does not exist, skipping: {}",
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

    fn make_path_lock(path: &str) -> LockFile {
        LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: "test_pkg".to_string(),
                version: None,
                source: PackageSource::Path {
                    path: path.to_string(),
                },
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
                version: Some("0.3.0".to_string()),
                source: PackageSource::Path {
                    path: "packages/head_agent".to_string(),
                },
            }],
        };

        save_lockfile(project_root, &original).unwrap();
        let loaded = load_lockfile(project_root).unwrap();

        assert_eq!(loaded, Some(original));
    }

    #[test]
    fn lockfile_roundtrip_no_version() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        let original = LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: "my_pkg".to_string(),
                version: None,
                source: PackageSource::Installed,
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

[package.source]
type = "path"
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
    fn resolve_path_relative() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create the subdirectory that the relative path points to.
        let pkg_dir = project_root.join("packages").join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();

        let lock = make_path_lock("packages/my_pkg");
        let paths = resolve_path_entries(project_root, &lock);

        let expected = project_root.join("packages").join("my_pkg");
        assert_eq!(paths, vec![expected]);
    }

    #[test]
    fn resolve_path_absolute_inside_project() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        let pkg_dir = project_root.join("abs_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();

        let lock = make_path_lock(pkg_dir.to_str().unwrap());
        let paths = resolve_path_entries(project_root, &lock);

        assert_eq!(paths, vec![pkg_dir]);
    }

    #[test]
    fn resolve_path_absolute_outside_project_accepted() {
        // containment check なし — project外パスも許可
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        let abs_tmp = tempfile::tempdir().unwrap();
        let abs_path = abs_tmp.path().to_path_buf();

        let lock = make_path_lock(abs_path.to_str().unwrap());
        let paths = resolve_path_entries(project_root, &lock);

        // 存在するので受け入れる
        assert_eq!(paths, vec![abs_path]);
    }

    #[test]
    fn resolve_path_relative_traversal_accepted() {
        // containment check なし — traversal も許可
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().join("project");
        std::fs::create_dir_all(&project_root).unwrap();

        let sibling = tmp.path().join("sibling");
        std::fs::create_dir_all(&sibling).unwrap();

        let lock = make_path_lock("../sibling");
        let paths = resolve_path_entries(&project_root, &lock);

        let expected = project_root.join("../sibling");
        assert_eq!(paths, vec![expected]);
    }

    #[test]
    fn resolve_path_skip_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Path does not exist — should be skipped silently.
        let lock = make_path_lock("nonexistent/path");
        let paths = resolve_path_entries(project_root, &lock);

        assert!(paths.is_empty());
    }
}
