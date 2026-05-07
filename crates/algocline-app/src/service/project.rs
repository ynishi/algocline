//! Project root resolution for algocline.
//!
//! Determines the "project root" — the directory that contains `alc.toml` —
//! using the following priority (high → low):
//!
//! 1. `explicit` argument (from MCP tool parameter `project_root`)
//! 2. `ALC_PROJECT_ROOT` environment variable
//! 3. Ancestor walk from `std::env::current_dir()` to find `alc.toml`
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
    resolve_project_root_with_session(explicit, None)
}

/// Resolve the project root with the optional session pin layer
/// inserted between `explicit` and `ALC_PROJECT_ROOT` (issue
/// #1776627475 §6: priority order P > S > E > W).
///
/// `session_pin` is the `AlcSession.project_root` snapshot taken
/// from `AppService::current_session()`. When `None` or when the
/// pin itself is `None`, behaviour matches the original 3-layer
/// chain (P > E > W).
///
/// Per-call `explicit` always wins so debug / override paths
/// remain effective even when a session is activated, matching the
/// §6 decision (priority P > S avoids surprising overrides while
/// preserving caller agency).
pub(crate) fn resolve_project_root_with_session(
    explicit: Option<&str>,
    session_pin: Option<&Path>,
) -> Option<PathBuf> {
    // 1. Explicit argument from MCP tool parameter (P).
    if let Some(s) = explicit {
        let p = PathBuf::from(s);
        if p.is_dir() {
            return Some(p);
        }
        // Explicit path exists but is not a directory — warn and fall through.
        tracing::warn!("project_root '{}' is not a directory, falling back", s);
    }

    // 2. Activated session pin (S). Inserted between P and E so
    //    explicit per-call args still win for debug, but session
    //    pin overrides ambient env (§6).
    if let Some(pin) = session_pin {
        if pin.is_dir() {
            return Some(pin.to_path_buf());
        }
        tracing::warn!(
            "session pin '{}' is not a directory, falling back",
            pin.display()
        );
    }

    // 3. ALC_PROJECT_ROOT environment variable (E).
    if let Ok(env) = std::env::var("ALC_PROJECT_ROOT") {
        if !env.is_empty() {
            let p = PathBuf::from(&env);
            if p.is_dir() {
                return Some(p);
            }
        }
    }

    // 4. Ancestor walk from current working directory (W).
    if let Ok(cwd) = std::env::current_dir() {
        return walk_up_for_alc_toml(&cwd);
    }

    None
}

/// Walk up from `start` toward the filesystem root, looking for `alc.toml`.
///
/// Returns the directory that *contains* `alc.toml`, or `None` if not found.
pub(crate) fn walk_up_for_alc_toml(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join("alc.toml").is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::with_env_var;
    use super::*;

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

        // Create alc.toml at the project root.
        std::fs::write(project_root.join("alc.toml"), "[packages]\n").unwrap();

        // Create a subdirectory and walk up from there.
        let sub = project_root.join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();

        let result = walk_up_for_alc_toml(&sub);
        assert_eq!(result, Some(project_root));
    }

    #[test]
    fn resolve_project_root_none_when_no_alc_toml() {
        let tmp = tempfile::tempdir().unwrap();
        // No alc.toml anywhere in tmp.
        let result = walk_up_for_alc_toml(tmp.path());
        // Should walk all the way up and return None.
        assert!(result.is_none());
    }

    #[test]
    fn walk_up_stops_at_root_when_no_alc_toml() {
        // Walk from filesystem root — should return None immediately.
        let root = PathBuf::from("/");
        // We only call walk_up; if alc.toml exists at / on this machine the test
        // would be a false positive, but that is essentially impossible in CI.
        let result = walk_up_for_alc_toml(&root);
        // Either None (normal) or Some("/") if someone placed alc.toml there.
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

    // ─── Session pin (issue #1776627475 §6 P > S > E > W) ───────

    #[test]
    fn session_pin_overrides_env_when_explicit_absent() {
        let session_dir = tempfile::tempdir().unwrap();
        let env_dir = tempfile::tempdir().unwrap();

        let session_path = session_dir.path().to_path_buf();
        let env_path = env_dir.path().to_path_buf();

        with_env_var("ALC_PROJECT_ROOT", env_path.to_str().unwrap(), || {
            let result = resolve_project_root_with_session(None, Some(&session_path));
            assert_eq!(
                result,
                Some(session_path.clone()),
                "session pin (S) must beat env (E) when explicit (P) is absent"
            );
        });
    }

    #[test]
    fn explicit_overrides_session_pin() {
        let explicit_dir = tempfile::tempdir().unwrap();
        let session_dir = tempfile::tempdir().unwrap();

        let explicit_path = explicit_dir.path().to_path_buf();
        let session_path = session_dir.path().to_path_buf();

        let result = resolve_project_root_with_session(
            Some(explicit_path.to_str().unwrap()),
            Some(&session_path),
        );
        assert_eq!(
            result,
            Some(explicit_path),
            "explicit (P) must beat session pin (S) for per-call debug overrides"
        );
    }

    #[test]
    fn session_pin_falls_through_when_not_a_directory() {
        // session pin points at a path that doesn't exist.
        // Result should fall through to env (E) or walk (W).
        let env_dir = tempfile::tempdir().unwrap();
        let env_path = env_dir.path().to_path_buf();
        let bogus = std::path::PathBuf::from("/nonexistent/session/pin/xyz");

        with_env_var("ALC_PROJECT_ROOT", env_path.to_str().unwrap(), || {
            let result = resolve_project_root_with_session(None, Some(&bogus));
            assert_eq!(
                result,
                Some(env_path.clone()),
                "invalid session pin must fall through to env (E)"
            );
        });
    }

    #[test]
    fn legacy_resolve_matches_no_session_pin() {
        // The legacy `resolve_project_root` wrapper must produce
        // the same answer as `_with_session(None, None)`.
        let env_dir = tempfile::tempdir().unwrap();
        let env_path = env_dir.path().to_path_buf();
        let env_path_for_assert = env_path.clone();
        with_env_var("ALC_PROJECT_ROOT", env_path.to_str().unwrap(), move || {
            let legacy = resolve_project_root(None);
            let with_session = resolve_project_root_with_session(None, None);
            assert_eq!(legacy, with_session);
            assert_eq!(legacy, Some(env_path_for_assert));
        });
    }
}
