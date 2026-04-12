//! `pkg_install` — install a package from a Git URL or local path.

use std::path::Path;

use super::super::alc_toml::{
    add_package_entry, load_alc_toml_document, save_alc_toml, PackageDep,
};
use super::super::hub;
use super::super::lockfile::{load_lockfile, save_lockfile, LockFile, LockPackage};
use super::super::manifest;
use super::super::path::{copy_dir, ContainedPath};
use super::super::project::resolve_project_root;
use super::super::resolve::{
    install_scenarios_from_dir, packages_dir, scenarios_dir, DirEntryFailures, AUTO_INSTALL_SOURCES,
};
use super::super::source::PackageSource;
use super::super::AppService;

impl AppService {
    /// Install a package from a Git URL or local path.
    pub async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let _ = std::fs::create_dir_all(&pkg_dir);

        // Local path: copy directly (supports uncommitted/dirty working trees)
        let local_path = Path::new(&url);
        if local_path.is_absolute() && local_path.is_dir() {
            return self
                .install_from_local_path(local_path, &pkg_dir, name)
                .await;
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

            // Record in manifest (best-effort; install itself already succeeded)
            let _ = manifest::record_install(&name, None, &url);
            hub::register_source(&url, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(std::slice::from_ref(&name))
                .await;

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "single",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
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

            // Import bundled cards from each package's cards/ subdirectory.
            let mut cards_installed: Vec<String> = Vec::new();
            for pkg_name in installed.iter().chain(skipped.iter()) {
                let cards_subdir = staging.path().join(pkg_name).join("cards");
                if cards_subdir.is_dir() {
                    let imported =
                        crate::AppService::import_pkg_bundled_cards(pkg_name, &cards_subdir);
                    cards_installed.extend(imported);
                }
            }

            // Install bundled scenarios only when an explicit `scenarios/` subdir exists.
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

            // Record in manifest (best-effort)
            let _ = manifest::record_install_batch(&installed, &url);
            hub::register_source(&url, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(&installed).await;

            let mut response = serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "cards_installed": cards_installed,
                "scenarios_installed": scenarios_installed,
                "scenarios_failures": scenarios_failures,
                "mode": "collection",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        }
    }

    /// Install from a local directory path (supports dirty/uncommitted files).
    async fn install_from_local_path(
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

            // Record in manifest (best-effort)
            let source_str_local = source.display().to_string();
            let _ = manifest::record_install(&name, None, &source_str_local);
            hub::register_source(&source_str_local, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(std::slice::from_ref(&name))
                .await;

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "local_single",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
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

            // Import bundled cards from each package's cards/ subdirectory.
            let mut cards_installed: Vec<String> = Vec::new();
            for pkg_name in installed.iter().chain(updated.iter()) {
                let cards_subdir = source.join(pkg_name).join("cards");
                if cards_subdir.is_dir() {
                    let imported =
                        crate::AppService::import_pkg_bundled_cards(pkg_name, &cards_subdir);
                    cards_installed.extend(imported);
                }
            }

            // Record in manifest (best-effort)
            let source_str = source.display().to_string();
            let all_names: Vec<String> = installed.iter().chain(updated.iter()).cloned().collect();
            let _ = manifest::record_install_batch(&all_names, &source_str);
            hub::register_source(&source_str, "pkg_install");

            // Update alc.toml + alc.lock for newly installed packages
            self.update_project_files_for_install(&installed).await;

            let mut response = serde_json::json!({
                "installed": installed,
                "updated": updated,
                "cards_installed": cards_installed,
                "mode": "local_collection",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        }
    }

    /// After a successful cache install, update `alc.toml` and `alc.lock` if a project
    /// root (containing `alc.toml`) is found.  Failures are logged but not propagated —
    /// the install itself already succeeded.
    async fn update_project_files_for_install(&self, names: &[String]) {
        let root = match resolve_project_root(None) {
            Some(r) => r,
            None => return, // No project root → skip (current-compat)
        };

        // Load alc.toml document (preserving comments/formatting).
        let mut doc = match load_alc_toml_document(&root) {
            Ok(Some(d)) => d,
            Ok(None) => return, // alc.toml not found → skip
            Err(e) => {
                tracing::warn!("pkg_install: failed to load alc.toml: {e}");
                return;
            }
        };

        // Load or create alc.lock.
        let mut lock = match load_lockfile(&root) {
            Ok(Some(l)) => l,
            Ok(None) => LockFile {
                version: 1,
                packages: Vec::new(),
            },
            Err(e) => {
                tracing::warn!("pkg_install: failed to load alc.lock: {e}");
                return;
            }
        };

        for name in names {
            // Add to alc.toml (no-op if already present).
            add_package_entry(&mut doc, name, &PackageDep::Version("*".to_string()));

            // Resolve version via eval_simple (best-effort).
            let version = self.fetch_pkg_version(name).await;

            // Upsert into alc.lock.
            upsert_lock_entry(&mut lock, name.clone(), version, PackageSource::Installed);
        }

        if let Err(e) = save_alc_toml(&root, &doc) {
            tracing::warn!("pkg_install: failed to save alc.toml: {e}");
        }
        if let Err(e) = save_lockfile(&root, &lock) {
            tracing::warn!("pkg_install: failed to save alc.lock: {e}");
        }
    }

    /// Fetch package version via `eval_simple` (best-effort; returns `None` on failure).
    async fn fetch_pkg_version(&self, name: &str) -> Option<String> {
        if !is_safe_pkg_name(name) {
            return None;
        }
        let code = format!(
            r#"package.loaded["{name}"] = nil
local pkg = require("{name}")
return (pkg.meta or {{}}).version"#
        );
        match self.executor.eval_simple(code).await {
            Ok(serde_json::Value::String(v)) if !v.is_empty() => Some(v),
            _ => None,
        }
    }

    /// Install all bundled sources (collections + single packages).
    pub(in crate::service) async fn auto_install_bundled_packages(&self) -> Result<(), String> {
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

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Returns `true` iff `name` is safe to interpolate into a Lua source string.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Insert or update a `LockPackage` entry in the lockfile.
fn upsert_lock_entry(
    lock: &mut LockFile,
    name: String,
    version: Option<String>,
    source: PackageSource,
) {
    if let Some(existing) = lock.packages.iter_mut().find(|p| p.name == name) {
        existing.version = version;
        existing.source = source;
    } else {
        lock.packages.push(LockPackage {
            name,
            version,
            source,
        });
    }
}
