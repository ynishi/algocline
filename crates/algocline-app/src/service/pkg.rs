use std::path::Path;

use super::path::{copy_dir, ContainedPath};
use super::resolve::{
    install_scenarios_from_dir, is_system_package, packages_dir, scenarios_dir, DirEntryFailures,
    AUTO_INSTALL_SOURCES,
};
use super::AppService;

impl AppService {
    /// List installed packages with metadata.
    pub async fn pkg_list(&self) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        if !pkg_dir.is_dir() {
            return Ok(serde_json::json!({ "packages": [] }).to_string());
        }

        let mut packages = Vec::new();
        let entries =
            std::fs::read_dir(&pkg_dir).map_err(|e| format!("Failed to read packages dir: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let init_lua = path.join("init.lua");
            if !init_lua.exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip system packages (not user-facing strategies)
            if is_system_package(&name) {
                continue;
            }
            let code = format!(
                r#"local pkg = require("{name}")
return pkg.meta or {{ name = "{name}" }}"#
            );
            match self.executor.eval_simple(code).await {
                Ok(meta) => packages.push(meta),
                Err(_) => {
                    packages
                        .push(serde_json::json!({ "name": name, "error": "failed to load meta" }));
                }
            }
        }

        Ok(serde_json::json!({ "packages": packages }).to_string())
    }

    /// Install a package from a Git URL or local path.
    pub async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let _ = std::fs::create_dir_all(&pkg_dir);

        // Local path: copy directly (supports uncommitted/dirty working trees)
        let local_path = Path::new(&url);
        if local_path.is_absolute() && local_path.is_dir() {
            return self.install_from_local_path(local_path, &pkg_dir, name);
        }

        // Normalize URL: add https:// only for bare domain-style URLs
        let git_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
            || url.starts_with("git@")
        {
            url.clone()
        } else {
            format!("https://{url}")
        };

        // Clone to temp directory first to detect single vs collection
        let staging = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let output = tokio::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                &git_url,
                &staging.path().to_string_lossy(),
            ])
            .output()
            .await
            .map_err(|e| format!("Failed to run git: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {stderr}"));
        }

        // Remove .git dir from staging
        let _ = std::fs::remove_dir_all(staging.path().join(".git"));

        // Detect: single package (init.lua at root) vs collection (subdirs with init.lua)
        if staging.path().join("init.lua").exists() {
            // Single package mode
            let name = name.unwrap_or_else(|| {
                url.trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .trim_end_matches(".git")
                    .to_string()
            });

            let dest = ContainedPath::child(&pkg_dir, &name)?;
            if dest.as_ref().exists() {
                return Err(format!(
                    "Package '{name}' already exists at {}. Remove it first.",
                    dest.as_ref().display()
                ));
            }

            copy_dir(staging.path(), dest.as_ref())
                .map_err(|e| format!("Failed to copy package: {e}"))?;

            Ok(serde_json::json!({
                "installed": [name],
                "mode": "single",
            })
            .to_string())
        } else {
            // Collection mode: scan for subdirs containing init.lua
            if name.is_some() {
                // name parameter is only meaningful for single-package repos
                return Err(
                    "The 'name' parameter is only supported for single-package repos (init.lua at root). \
                     This repository is a collection (subdirs with init.lua)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut skipped = Vec::new();

            let entries = std::fs::read_dir(staging.path())
                .map_err(|e| format!("Failed to read staging dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                let dest = pkg_dir.join(&pkg_name);
                if dest.exists() {
                    skipped.push(pkg_name);
                    continue;
                }
                copy_dir(&path, &dest)
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                installed.push(pkg_name);
            }

            // Install bundled scenarios only when an explicit `scenarios/` subdir exists.
            // Unlike `scenario_install` (which falls back to root via `resolve_scenario_source`),
            // bundled scenarios are optional — we don't scan the package root for .lua files.
            let scenarios_subdir = staging.path().join("scenarios");
            let mut scenarios_installed: Vec<String> = Vec::new();
            let mut scenarios_failures: DirEntryFailures = Vec::new();
            if scenarios_subdir.is_dir() {
                if let Ok(sc_dir) = scenarios_dir() {
                    std::fs::create_dir_all(&sc_dir)
                        .map_err(|e| format!("Failed to create scenarios dir: {e}"))?;
                    if let Ok(result) = install_scenarios_from_dir(&scenarios_subdir, &sc_dir) {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                            if let Some(arr) = parsed.get("installed").and_then(|v| v.as_array()) {
                                scenarios_installed = arr
                                    .iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect();
                            }
                            if let Some(arr) = parsed.get("failures").and_then(|v| v.as_array()) {
                                scenarios_failures = arr
                                    .iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect();
                            }
                        }
                    }
                }
            }

            if installed.is_empty() && skipped.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            Ok(serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "scenarios_installed": scenarios_installed,
                "scenarios_failures": scenarios_failures,
                "mode": "collection",
            })
            .to_string())
        }
    }

    /// Install from a local directory path (supports dirty/uncommitted files).
    fn install_from_local_path(
        &self,
        source: &Path,
        pkg_dir: &Path,
        name: Option<String>,
    ) -> Result<String, String> {
        if source.join("init.lua").exists() {
            // Single package
            let name = name.unwrap_or_else(|| {
                source
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });

            let dest = ContainedPath::child(pkg_dir, &name)?;
            if dest.as_ref().exists() {
                // Overwrite for local installs (dev workflow)
                let _ = std::fs::remove_dir_all(&dest);
            }

            copy_dir(source, dest.as_ref()).map_err(|e| format!("Failed to copy package: {e}"))?;
            // Remove .git if copied
            let _ = std::fs::remove_dir_all(dest.as_ref().join(".git"));

            Ok(serde_json::json!({
                "installed": [name],
                "mode": "local_single",
            })
            .to_string())
        } else {
            // Collection mode
            if name.is_some() {
                return Err(
                    "The 'name' parameter is only supported for single-package dirs (init.lua at root)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut updated = Vec::new();

            let entries =
                std::fs::read_dir(source).map_err(|e| format!("Failed to read source dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() || !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                let dest = pkg_dir.join(&pkg_name);
                let existed = dest.exists();
                if existed {
                    let _ = std::fs::remove_dir_all(&dest);
                }
                copy_dir(&path, &dest)
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                let _ = std::fs::remove_dir_all(dest.join(".git"));
                if existed {
                    updated.push(pkg_name);
                } else {
                    installed.push(pkg_name);
                }
            }

            if installed.is_empty() && updated.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            Ok(serde_json::json!({
                "installed": installed,
                "updated": updated,
                "mode": "local_collection",
            })
            .to_string())
        }
    }

    /// Remove an installed package.
    pub async fn pkg_remove(&self, name: &str) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let dest = ContainedPath::child(&pkg_dir, name)?;

        if !dest.as_ref().exists() {
            return Err(format!("Package '{name}' not found"));
        }

        std::fs::remove_dir_all(&dest).map_err(|e| format!("Failed to remove '{name}': {e}"))?;

        Ok(serde_json::json!({ "removed": name }).to_string())
    }

    /// Install all bundled sources (collections + single packages).
    pub(super) async fn auto_install_bundled_packages(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();
        for url in AUTO_INSTALL_SOURCES {
            tracing::info!("auto-installing from {url}");
            if let Err(e) = self.pkg_install(url.to_string(), None).await {
                tracing::warn!("failed to auto-install from {url}: {e}");
                errors.push(format!("{url}: {e}"));
            }
        }
        // Fail only if ALL sources failed
        if errors.len() == AUTO_INSTALL_SOURCES.len() {
            return Err(format!(
                "Failed to auto-install bundled packages: {}",
                errors.join("; ")
            ));
        }
        Ok(())
    }
}
