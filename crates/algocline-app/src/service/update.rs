//! `alc_update` — re-resolve all `alc.toml` entries and rewrite `alc.lock`.

use super::alc_toml::{load_alc_toml, PackageDep};
use super::lockfile::{lockfile_path, save_lockfile, LockFile, LockPackage};
use super::path::copy_dir;
use super::project::resolve_project_root;
use super::resolve::packages_dir;
use super::source::PackageSource;
use super::AppService;

impl AppService {
    pub async fn update(&self, project_root: Option<String>) -> Result<String, String> {
        let root = resolve_project_root(project_root.as_deref())
            .ok_or_else(|| "No alc.toml found. Run alc_init first.".to_string())?;

        let toml = load_alc_toml(&root)?
            .ok_or_else(|| "alc.toml not found at resolved project root".to_string())?;

        let pkg_dir = packages_dir()?;
        let mut resolved: Vec<LockPackage> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        for (name, dep) in &toml.packages {
            match dep {
                PackageDep::Version(v) if v == "*" => {
                    let dir = pkg_dir.join(name);
                    if dir.is_dir() {
                        resolved.push(LockPackage {
                            name: name.clone(),
                            version: None,
                            source: PackageSource::Installed,
                        });
                    } else {
                        errors.push(format!(
                            "'{name}': not installed (not found in packages_dir)"
                        ));
                    }
                }
                PackageDep::Version(v) => {
                    let versioned = pkg_dir.join(format!("{name}@{v}"));
                    if versioned.is_dir() {
                        resolved.push(LockPackage {
                            name: name.clone(),
                            version: Some(v.clone()),
                            source: PackageSource::Installed,
                        });
                    } else {
                        let base = pkg_dir.join(name);
                        if base.is_dir() {
                            // lazy creation: copy base/ → {name}@{version}/
                            copy_dir(&base, &versioned)
                                .map_err(|e| format!("Failed to create {name}@{v}: {e}"))?;
                            resolved.push(LockPackage {
                                name: name.clone(),
                                version: Some(v.clone()),
                                source: PackageSource::Installed,
                            });
                        } else {
                            errors.push(format!(
                                "'{name}@{v}': not found in packages_dir (neither versioned nor base dir)"
                            ));
                        }
                    }
                }
                PackageDep::Path { path, version: ver } => {
                    // Phase 1: use version from alc.toml as-is
                    resolved.push(LockPackage {
                        name: name.clone(),
                        version: ver.clone(),
                        source: PackageSource::Path { path: path.clone() },
                    });
                }
                PackageDep::Git { .. } => {
                    errors.push(format!("'{name}': Git source not supported in Phase 1"));
                }
            }
        }

        let lock = LockFile {
            version: 1,
            packages: resolved.clone(),
        };
        save_lockfile(&root, &lock)?;

        let lock_path = lockfile_path(&root);
        let result = serde_json::json!({
            "resolved": resolved.len(),
            "errors": errors,
            "alc_lock": lock_path.display().to_string(),
        });
        Ok(result.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::super::AppService;
    use crate::service::config::{AppConfig, LogDirSource};
    use std::sync::Arc;

    async fn make_service() -> AppService {
        let executor = Arc::new(
            algocline_engine::Executor::new(vec![])
                .await
                .expect("executor"),
        );
        AppService {
            executor,
            registry: Arc::new(algocline_engine::SessionRegistry::new()),
            log_config: AppConfig {
                log_dir: None,
                log_dir_source: LogDirSource::None,
                log_enabled: false,
            },
            search_paths: vec![],
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    #[tokio::test]
    async fn update_fails_without_alc_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service().await;
        let err = svc
            .update(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap_err();
        // resolve_project_root returns None (no alc.toml)
        assert!(
            err.contains("No alc.toml found") || err.contains("alc.toml not found"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn update_with_path_dep_writes_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path().join("mypkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();

        std::fs::write(
            tmp.path().join("alc.toml"),
            format!("[packages.mypkg]\npath = \"{}\"\n", pkg_dir.display()),
        )
        .unwrap();

        let svc = make_service().await;
        let result = svc
            .update(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        assert!(result.contains("\"resolved\":1"), "{result}");
        assert!(result.contains("\"errors\":[]"), "{result}");
        assert!(tmp.path().join("alc.lock").exists());
    }

    #[tokio::test]
    async fn update_git_dep_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("alc.toml"),
            "[packages.mypkg]\ngit = \"https://github.com/user/pkg\"\n",
        )
        .unwrap();

        let svc = make_service().await;
        let result = svc
            .update(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        // errors list is non-empty
        assert!(result.contains("Git source not supported"), "{result}");
    }
}
