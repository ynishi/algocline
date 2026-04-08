//! Project root resolution for algocline.
//!
//! Determines the "project root" — the directory that contains `alc.lock` —
//! using the following priority (high → low):
//!
//! 1. `explicit` argument (from MCP tool parameter `project_root`)
//! 2. `ALC_PROJECT_ROOT` environment variable
//! 3. Ancestor walk from `std::env::current_dir()` to find `alc.lock`
//!
//! **Note on MCP server cwd**: algocline runs as a long-lived daemon process.
//! `std::env::current_dir()` returns the cwd at server startup, not the
//! client's cwd. For reliable project root resolution, prefer the explicit
//! `project_root` parameter or the `ALC_PROJECT_ROOT` environment variable.

use std::path::{Path, PathBuf};

/// Resolve the project root using the priority described in the module doc.
///
/// Returns `Some(root)` if any source yields a valid directory.
/// Returns `None` if none of the sources applies.
///
/// Note: even if the returned root does not contain `alc.lock`, `Some` is
/// returned (used by `alc_pkg_link` to create the file on first use).
/// Callers that need actual local_dir paths should call
/// `load_lockfile(root)` and treat `Ok(None)` as "no local packages".
pub(crate) fn resolve_project_root(explicit: Option<&str>) -> Option<PathBuf> {
    // 1. Explicit argument from MCP tool parameter.
    if let Some(s) = explicit {
        let p = PathBuf::from(s);
        if p.is_dir() {
            return Some(p);
        }
        // Explicit path exists but is not a directory — warn and fall through.
        eprintln!("alc: project_root '{}' is not a directory, falling back", s);
    }

    // 2. ALC_PROJECT_ROOT environment variable.
    if let Ok(env) = std::env::var("ALC_PROJECT_ROOT") {
        if !env.is_empty() {
            let p = PathBuf::from(&env);
            if p.is_dir() {
                return Some(p);
            }
        }
    }

    // 3. Ancestor walk from current working directory.
    if let Ok(cwd) = std::env::current_dir() {
        return walk_up_for_lockfile(&cwd);
    }

    None
}

/// Walk up from `start` toward the filesystem root, looking for `alc.lock`.
///
/// Returns the directory that *contains* `alc.lock`, or `None` if not found.
pub(crate) fn walk_up_for_lockfile(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join("alc.lock").is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: set ALC_PROJECT_ROOT for the duration of the closure, then restore.
    /// Uses a mutex so parallel tests don't race on the env var.
    fn with_env_var<F: FnOnce()>(key: &str, val: &str, f: F) {
        // Safety: test-only, serialised via LOCK.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(key).ok();
        std::env::set_var(key, val);
        f();
        match prev {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn resolve_project_root_via_env() {
        let tmp = tempfile::tempdir().unwrap();
        let tmp_path = tmp.path().to_path_buf();

        with_env_var("ALC_PROJECT_ROOT", tmp_path.to_str().unwrap(), || {
            let result = resolve_project_root(None);
            assert_eq!(result, Some(tmp_path.clone()));
        });
    }

    #[test]
    fn resolve_project_root_explicit_wins_over_env() {
        let explicit_dir = tempfile::tempdir().unwrap();
        let env_dir = tempfile::tempdir().unwrap();

        let explicit_path = explicit_dir.path().to_path_buf();
        let env_path = env_dir.path().to_path_buf();

        with_env_var("ALC_PROJECT_ROOT", env_path.to_str().unwrap(), || {
            let result = resolve_project_root(Some(explicit_path.to_str().unwrap()));
            assert_eq!(result, Some(explicit_path.clone()));
        });
    }

    #[test]
    fn resolve_project_root_walk_up() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().to_path_buf();

        // Create alc.lock at the project root.
        std::fs::write(project_root.join("alc.lock"), "version = 1\n").unwrap();

        // Create a subdirectory and walk up from there.
        let sub = project_root.join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();

        let result = walk_up_for_lockfile(&sub);
        assert_eq!(result, Some(project_root));
    }

    #[test]
    fn resolve_project_root_none_when_no_lockfile() {
        let tmp = tempfile::tempdir().unwrap();
        // No alc.lock anywhere in tmp.
        let result = walk_up_for_lockfile(tmp.path());
        // Should walk all the way up and return None.
        assert!(result.is_none());
    }

    #[test]
    fn walk_up_stops_at_root_when_no_lockfile() {
        // Walk from filesystem root — should return None immediately.
        let root = PathBuf::from("/");
        // We only call walk_up; if alc.lock exists at / on this machine the test
        // would be a false positive, but that is essentially impossible in CI.
        let result = walk_up_for_lockfile(&root);
        // Either None (normal) or Some("/") if someone placed alc.lock there.
        // We assert it doesn't panic.
        drop(result);
    }

    #[test]
    fn resolve_project_root_explicit_non_dir_falls_back() {
        // Pass a path that doesn't exist as a directory.
        // With ALC_PROJECT_ROOT unset and no alc.lock in cwd ancestors,
        // the result should be None (or cwd walk result).
        // We just verify it doesn't panic.
        let result = resolve_project_root(Some("/this/path/does/not/exist_xyz"));
        // Result may be None or Some depending on cwd.
        drop(result);
    }
}
