//! `alc_pkg_unlink` — remove a symlinked package from `~/.algocline/packages/`.
//!
//! Only operates on symlinks. For installed (copied) packages, use `pkg_remove`.

use super::alc_toml::validate_package_name;
use super::resolve::packages_dir;
use super::AppService;

impl AppService {
    /// Remove a symlinked package from the global cache.
    ///
    /// - If `~/.algocline/packages/{name}` is a symlink: removes it.
    /// - If it is a real directory: returns an error directing to `pkg_remove`.
    /// - If it does not exist: returns an error.
    pub async fn pkg_unlink(&self, name: String) -> Result<String, String> {
        validate_package_name(&name)?;

        let pkgs = packages_dir()?;
        let dest = pkgs.join(&name);

        // Use symlink_metadata so dangling symlinks are also detected.
        match dest.symlink_metadata() {
            Ok(m) => {
                if m.file_type().is_symlink() {
                    std::fs::remove_file(&dest)
                        .map_err(|e| format!("Failed to remove symlink {}: {e}", dest.display()))?;
                } else {
                    return Err(format!(
                        "Package '{}' is not a symlink. Use pkg_remove to remove installed packages.",
                        name
                    ));
                }
            }
            Err(_) => {
                return Err(format!("Package '{name}' is not installed"));
            }
        }

        Ok(serde_json::json!({ "unlinked": name }).to_string())
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::symlink;
    use std::sync::Arc;

    use super::*;

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

    fn with_fake_home<F: FnOnce(&std::path::Path)>(f: F) {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        f(tmp.path());
        std::env::remove_var("HOME");
    }

    #[tokio::test]
    async fn pkg_unlink_removes_symlink() {
        with_fake_home(|home| {
            let pkgs = home.join(".algocline").join("packages");
            std::fs::create_dir_all(&pkgs).unwrap();

            let target = home.join("my_pkg");
            std::fs::create_dir_all(&target).unwrap();
            let dest = pkgs.join("my_pkg");
            symlink(&target, &dest).unwrap();

            let svc = tokio::runtime::Handle::current().block_on(make_app_service());

            let result = tokio::runtime::Handle::current()
                .block_on(svc.pkg_unlink("my_pkg".to_string()))
                .unwrap();

            let json: serde_json::Value = serde_json::from_str(&result).unwrap();
            assert_eq!(json["unlinked"], "my_pkg");
            assert!(!dest.symlink_metadata().is_ok());
        });
    }

    #[tokio::test]
    async fn pkg_unlink_real_dir_returns_error() {
        with_fake_home(|home| {
            let pkgs = home.join(".algocline").join("packages");
            let dest = pkgs.join("my_pkg");
            std::fs::create_dir_all(&dest).unwrap();

            let svc = tokio::runtime::Handle::current().block_on(make_app_service());

            let err = tokio::runtime::Handle::current()
                .block_on(svc.pkg_unlink("my_pkg".to_string()))
                .unwrap_err();

            assert!(err.contains("not a symlink"), "got: {err}");
        });
    }

    #[tokio::test]
    async fn pkg_unlink_not_installed_returns_error() {
        with_fake_home(|home| {
            let pkgs = home.join(".algocline").join("packages");
            std::fs::create_dir_all(&pkgs).unwrap();

            let svc = tokio::runtime::Handle::current().block_on(make_app_service());

            let err = tokio::runtime::Handle::current()
                .block_on(svc.pkg_unlink("nonexistent".to_string()))
                .unwrap_err();

            assert!(err.contains("not installed"), "got: {err}");
        });
    }

    #[tokio::test]
    async fn pkg_unlink_dangling_symlink_removed() {
        with_fake_home(|home| {
            let pkgs = home.join(".algocline").join("packages");
            std::fs::create_dir_all(&pkgs).unwrap();

            let dest = pkgs.join("dangling_pkg");
            symlink(home.join("nowhere"), &dest).unwrap();
            assert!(!dest.exists()); // dangling

            let svc = tokio::runtime::Handle::current().block_on(make_app_service());

            let result = tokio::runtime::Handle::current()
                .block_on(svc.pkg_unlink("dangling_pkg".to_string()))
                .unwrap();

            let json: serde_json::Value = serde_json::from_str(&result).unwrap();
            assert_eq!(json["unlinked"], "dangling_pkg");
            assert!(!dest.symlink_metadata().is_ok());
        });
    }
}
