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

        let pkgs = packages_dir(&self.log_config.app_dir());
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

    use crate::service::test_support::make_app_service_at;

    #[tokio::test]
    async fn pkg_unlink_removes_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();

        let pkgs = home.join("packages");
        std::fs::create_dir_all(&pkgs).unwrap();

        let target = home.join("my_pkg");
        std::fs::create_dir_all(&target).unwrap();
        let dest = pkgs.join("my_pkg");
        symlink(&target, &dest).unwrap();

        let svc = make_app_service_at(home.to_path_buf()).await;
        let result = svc.pkg_unlink("my_pkg".to_string()).await.unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["unlinked"], "my_pkg");
        assert!(dest.symlink_metadata().is_err());
    }

    #[tokio::test]
    async fn pkg_unlink_real_dir_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();

        let pkgs = home.join("packages");
        let dest = pkgs.join("my_pkg");
        std::fs::create_dir_all(&dest).unwrap();

        let svc = make_app_service_at(home.to_path_buf()).await;
        let err = svc.pkg_unlink("my_pkg".to_string()).await.unwrap_err();

        assert!(err.contains("not a symlink"), "got: {err}");
    }

    #[tokio::test]
    async fn pkg_unlink_not_installed_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();

        let pkgs = home.join("packages");
        std::fs::create_dir_all(&pkgs).unwrap();

        let svc = make_app_service_at(home.to_path_buf()).await;
        let err = svc.pkg_unlink("nonexistent".to_string()).await.unwrap_err();

        assert!(err.contains("not installed"), "got: {err}");
    }

    #[tokio::test]
    async fn pkg_unlink_dangling_symlink_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();

        let pkgs = home.join("packages");
        std::fs::create_dir_all(&pkgs).unwrap();

        let dest = pkgs.join("dangling_pkg");
        symlink(home.join("nowhere"), &dest).unwrap();
        assert!(!dest.exists()); // dangling

        let svc = make_app_service_at(home.to_path_buf()).await;
        let result = svc.pkg_unlink("dangling_pkg".to_string()).await.unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["unlinked"], "dangling_pkg");
        assert!(dest.symlink_metadata().is_err());
    }
}
