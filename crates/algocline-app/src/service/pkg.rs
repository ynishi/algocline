use std::collections::HashMap;
use std::path::Path;

use super::lockfile::{load_lockfile, lockfile_path};
use super::manifest;
use super::path::{copy_dir, ContainedPath};
use super::project::resolve_project_root;
use super::resolve::{
    install_scenarios_from_dir, is_system_package, packages_dir, scenarios_dir, DirEntryFailures,
    AUTO_INSTALL_SOURCES,
};
use super::source::{infer_from_legacy_source_string, PackageSource};
use super::AppService;

// ─── Intermediate DTO for pkg_list ───────────────────────────────

#[derive(Debug)]
enum Scope {
    Project,
    Global,
}

/// Typed intermediate representation of a single package list entry.
/// Converted to `serde_json::Value` only at the final serialisation step.
/// Fields that are `None` are omitted from the output JSON.
#[derive(Debug)]
struct PackageListEntry {
    name: String,
    scope: Scope,
    /// Absent (`None`) when the package is not recorded in `installed.json`.
    source_type: Option<String>,
    /// Absolute path — project-local packages only.
    path: Option<String>,
    /// Search-path directory — global packages only.
    source: Option<String>,
    active: bool,
    linked_at: Option<String>,
    installed_at: Option<String>,
    updated_at: Option<String>,
    /// Legacy source string from `installed.json` (the raw URL/path).
    install_source: Option<String>,
    overrides: Option<Vec<String>>,
    meta: serde_json::Value,
    error: Option<String>,
}

impl PackageListEntry {
    fn into_json(self) -> serde_json::Value {
        let scope_str = match self.scope {
            Scope::Project => "project",
            Scope::Global => "global",
        };

        let mut map = serde_json::Map::new();
        map.insert("name".to_string(), serde_json::Value::String(self.name));
        map.insert(
            "scope".to_string(),
            serde_json::Value::String(scope_str.to_string()),
        );

        // source_type: only insert when resolved (no fallback to "global")
        if let Some(st) = self.source_type {
            map.insert("source_type".to_string(), serde_json::Value::String(st));
        }

        if let Some(p) = self.path {
            map.insert("path".to_string(), serde_json::Value::String(p));
        }
        if let Some(s) = self.source {
            map.insert("source".to_string(), serde_json::Value::String(s));
        }

        map.insert("active".to_string(), serde_json::Value::Bool(self.active));

        if let Some(la) = self.linked_at {
            map.insert("linked_at".to_string(), serde_json::Value::String(la));
        }
        if let Some(ia) = self.installed_at {
            map.insert("installed_at".to_string(), serde_json::Value::String(ia));
        }
        if let Some(ua) = self.updated_at {
            map.insert("updated_at".to_string(), serde_json::Value::String(ua));
        }
        if let Some(is) = self.install_source {
            map.insert("install_source".to_string(), serde_json::Value::String(is));
        }
        if let Some(ov) = self.overrides {
            map.insert("overrides".to_string(), serde_json::json!(ov));
        }

        // Merge meta fields (Lua pkg.meta) into the top-level object.
        if let serde_json::Value::Object(meta_map) = self.meta {
            for (k, v) in meta_map {
                // Never let meta overwrite the fields we have already set.
                map.entry(k).or_insert(v);
            }
        }

        if let Some(err) = self.error {
            map.insert("error".to_string(), serde_json::Value::String(err));
        }

        serde_json::Value::Object(map)
    }
}

impl AppService {
    /// List installed packages with metadata, showing the full override chain.
    ///
    /// When `project_root` is provided (or resolvable), project-local packages
    /// from `alc.lock` are prepended with `scope: "project"`. Global packages
    /// carry `scope: "global"`. If a project package and a global package share
    /// the same name, the project one is `active: true` and the global one
    /// `active: false`.
    pub async fn pkg_list(&self, project_root: Option<String>) -> Result<String, String> {
        // ── Load manifest once upfront ─────────────────────────────────────
        let manifest_data = manifest::load_manifest().unwrap_or_default();

        // ── Project-local packages (from alc.lock) ─────────────────────────
        let resolved_root = resolve_project_root(project_root.as_deref());

        let mut project_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut entries: Vec<PackageListEntry> = Vec::new();
        let mut project_root_str: Option<String> = None;
        let mut lockfile_path_str: Option<String> = None;

        if let Some(ref root) = resolved_root {
            project_root_str = Some(root.display().to_string());
            lockfile_path_str = Some(lockfile_path(root).display().to_string());

            match load_lockfile(root) {
                Ok(Some(lock)) => {
                    for pkg in &lock.packages {
                        let PackageSource::LocalDir { path: ref raw_path } = pkg.source else {
                            continue;
                        };
                        let abs_path = {
                            let p = Path::new(raw_path);
                            if p.is_absolute() {
                                p.to_path_buf()
                            } else {
                                root.join(p)
                            }
                        };

                        project_names.insert(pkg.name.clone());
                        entries.push(PackageListEntry {
                            name: pkg.name.clone(),
                            scope: Scope::Project,
                            source_type: Some("local_dir".to_string()),
                            path: Some(abs_path.display().to_string()),
                            source: None,
                            active: true,
                            linked_at: Some(pkg.linked_at.clone()),
                            installed_at: None,
                            updated_at: None,
                            install_source: None,
                            overrides: None,
                            meta: serde_json::Value::Object(serde_json::Map::new()),
                            error: None,
                        });
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("alc: failed to load alc.lock: {e}");
                }
            }
        }

        // ── Global packages (from search paths) ────────────────────────────
        // Key: package name → list of (search_path_index, source_display)
        let mut seen: HashMap<String, Vec<(usize, String)>> = HashMap::new();
        // Separate Vec so overrides pass can reference seen after collection.
        let global_start_idx = entries.len();

        for (idx, sp) in self.search_paths.iter().enumerate() {
            if !sp.path.is_dir() {
                continue;
            }
            let read_entries = match std::fs::read_dir(&sp.path) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for dir_entry in read_entries.flatten() {
                let path = dir_entry.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("init.lua").exists() {
                    continue;
                }
                let name = dir_entry.file_name().to_string_lossy().to_string();
                if is_system_package(&name) {
                    continue;
                }

                let source_display = sp.path.display().to_string();
                seen.entry(name.clone())
                    .or_default()
                    .push((idx, source_display.clone()));

                // active among globals: first occurrence wins; also shadowed
                // by project-local if same name
                let global_active = seen[&name].len() == 1 && !project_names.contains(&name);

                // Evaluate Lua meta (best-effort; error captured in entry).
                // Only interpolate the name into Lua source when it matches a
                // strict whitelist (alnum / `_` / `-`). Names outside this set
                // cannot be `require`d by Lua anyway; refusing them here also
                // forecloses any Lua string-injection via crafted directory
                // names under search paths.
                let (meta, eval_error) = if is_safe_pkg_name(&name) {
                    let code = format!(
                        r#"package.loaded["{name}"] = nil
local pkg = require("{name}")
return pkg.meta or {{ name = "{name}" }}"#
                    );
                    match self.executor.eval_simple(code).await {
                        Ok(v) => (v, None),
                        Err(_) => (
                            serde_json::Value::Object(serde_json::Map::new()),
                            Some("failed to load meta".to_string()),
                        ),
                    }
                } else {
                    (
                        serde_json::Value::Object(serde_json::Map::new()),
                        Some("invalid package name".to_string()),
                    )
                };

                // Look up manifest to determine source_type at collection time.
                let (source_type, installed_at, updated_at, install_source) =
                    if let Some(entry) = manifest_data.packages.get(&name) {
                        let st = match infer_from_legacy_source_string(&entry.source) {
                            PackageSource::Git { .. } => "git",
                            PackageSource::LocalCopy { .. } => "local_copy",
                            PackageSource::LocalDir { .. } => "local_dir",
                            PackageSource::Bundled { .. } => "bundled",
                        };
                        (
                            Some(st.to_string()),
                            Some(entry.installed_at.clone()),
                            Some(entry.updated_at.clone()),
                            Some(entry.source.clone()),
                        )
                    } else {
                        // Not registered in manifest → source_type absent
                        (None, None, None, None)
                    };

                entries.push(PackageListEntry {
                    name,
                    scope: Scope::Global,
                    source_type,
                    path: None,
                    source: Some(source_display),
                    active: global_active,
                    linked_at: None,
                    installed_at,
                    updated_at,
                    install_source,
                    overrides: None,
                    meta,
                    error: eval_error,
                });
            }
        }

        // ── Overrides pass (global packages only) ─────────────────────────
        // For each active global whose name appears in more than one search path,
        // record the lower-priority search-path paths as `overrides`.
        for entry in entries[global_start_idx..].iter_mut() {
            if !entry.active {
                continue;
            }
            if let Some(occurrences) = seen.get(&entry.name) {
                if occurrences.len() > 1 {
                    entry.overrides =
                        Some(occurrences.iter().skip(1).map(|(_, s)| s.clone()).collect());
                }
            }
        }

        // ── Serialise ─────────────────────────────────────────────────────
        let all_packages: Vec<serde_json::Value> =
            entries.into_iter().map(|e| e.into_json()).collect();

        let search_paths_json: Vec<serde_json::Value> = self
            .search_paths
            .iter()
            .map(|sp| {
                serde_json::json!({
                    "path": sp.path.display().to_string(),
                    "source": sp.source.to_string(),
                })
            })
            .collect();

        let mut result = serde_json::json!({
            "packages": all_packages,
            "search_paths": search_paths_json,
        });

        if let Some(root_str) = project_root_str {
            result["project_root"] = serde_json::Value::String(root_str);
        }
        if let Some(lp) = lockfile_path_str {
            result["lockfile_path"] = serde_json::Value::String(lp);
        }

        Ok(result.to_string())
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

            // Record in manifest (best-effort; install itself already succeeded)
            let _ = manifest::record_install(&name, None, &url);

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "single",
            });
            if let Some(tp) = super::resolve::types_stub_path() {
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

            // Record in manifest (best-effort)
            let _ = manifest::record_install_batch(&installed, &url);

            let mut response = serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "scenarios_installed": scenarios_installed,
                "scenarios_failures": scenarios_failures,
                "mode": "collection",
            });
            if let Some(tp) = super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
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

            // Record in manifest (best-effort)
            let _ = manifest::record_install(&name, None, &source.display().to_string());

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "local_single",
            });
            if let Some(tp) = super::resolve::types_stub_path() {
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

            // Record in manifest (best-effort)
            let source_str = source.display().to_string();
            let all_names: Vec<String> = installed.iter().chain(updated.iter()).cloned().collect();
            let _ = manifest::record_install_batch(&all_names, &source_str);

            let mut response = serde_json::json!({
                "installed": installed,
                "updated": updated,
                "mode": "local_collection",
            });
            if let Some(tp) = super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        }
    }

    /// Remove an installed package.
    ///
    /// - `project_root` specified or `scope == "project"`: removes the entry
    ///   from `alc.lock` only. The physical files are **not** deleted.
    /// - `scope == "global"`: removes from `~/.algocline/packages/` (existing
    ///   behavior), even if the package exists in `alc.lock`.
    /// - Default (no scope, no project_root): tries project first, falls back
    ///   to global.
    pub async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        scope: Option<String>,
    ) -> Result<String, String> {
        let effective_scope =
            determine_remove_scope(name, project_root.as_deref(), scope.as_deref());

        match effective_scope {
            RemoveScope::Project(root) => {
                // Remove from alc.lock only; do NOT delete physical files.
                let mut lock = match load_lockfile(&root)? {
                    Some(l) => l,
                    None => {
                        return Err(format!(
                            "Package '{name}' not found in project (no alc.lock at {})",
                            root.display()
                        ));
                    }
                };

                let before = lock.packages.len();
                lock.packages.retain(|p| p.name != name);

                if lock.packages.len() == before {
                    return Err(format!(
                        "Package '{name}' not found in alc.lock at {}",
                        root.display()
                    ));
                }

                super::lockfile::save_lockfile(&root, &lock)?;

                Ok(serde_json::json!({
                    "removed": name,
                    "scope": "project",
                    "lockfile": lockfile_path(&root).display().to_string(),
                })
                .to_string())
            }
            RemoveScope::Global => {
                // Original behavior: delete physical files.
                let pkg_dir = packages_dir()?;
                let dest = ContainedPath::child(&pkg_dir, name)?;

                if !dest.as_ref().exists() {
                    return Err(format!("Package '{name}' not found"));
                }

                std::fs::remove_dir_all(&dest)
                    .map_err(|e| format!("Failed to remove '{name}': {e}"))?;

                // Remove from manifest (best-effort)
                let _ = manifest::record_remove(name);

                Ok(serde_json::json!({ "removed": name, "scope": "global" }).to_string())
            }
        }
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

// ─── Name validation ─────────────────────────────────────────────

/// Returns `true` iff `name` is safe to interpolate into a Lua source string.
///
/// Accepts ASCII alphanumerics, `_` and `-`. Empty strings are rejected.
/// This matches the set of names that Lua's `require` can actually resolve
/// against `FsResolver`, so nothing legitimate is excluded.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

// ─── Remove scope resolution ─────────────────────────────────────

enum RemoveScope {
    Project(std::path::PathBuf),
    Global,
}

/// Determine the effective remove scope.
///
/// Priority:
/// 1. `scope == "global"` → always Global (ignores project_root).
/// 2. `project_root` is `Some` → Project (use that root).
/// 3. `scope == "project"` → Project (auto-detect root, must find package).
/// 4. Auto-detect: resolve root; if alc.lock contains the package → Project,
///    otherwise → Global.
fn determine_remove_scope(
    name: &str,
    project_root: Option<&str>,
    scope: Option<&str>,
) -> RemoveScope {
    // Explicit global scope — skip all project logic.
    if scope == Some("global") {
        return RemoveScope::Global;
    }

    let root = match resolve_project_root(project_root) {
        Some(r) => r,
        None => return RemoveScope::Global,
    };

    // Explicit project_root provided → always project scope.
    if project_root.is_some() {
        return RemoveScope::Project(root);
    }

    // Explicit scope == "project" → project scope (auto-detected root).
    if scope == Some("project") {
        return RemoveScope::Project(root);
    }

    // Auto-detection: only use project scope when the package is actually
    // listed in alc.lock.
    match load_lockfile(&root) {
        Ok(Some(lock)) if lock.packages.iter().any(|p| p.name == name) => {
            RemoveScope::Project(root)
        }
        _ => RemoveScope::Global,
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::lockfile::{load_lockfile, LockFile, LockPackage};
    use crate::service::source::PackageSource;

    fn make_lock_with_pkg(name: &str) -> LockFile {
        LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: name.to_string(),
                source: PackageSource::LocalDir {
                    path: format!("packages/{name}"),
                },
                linked_at: "2026-04-08T12:00:00Z".to_string(),
            }],
        }
    }

    async fn make_app_service() -> AppService {
        make_app_service_with_search_paths(vec![]).await
    }

    async fn make_app_service_with_search_paths(
        search_paths: Vec<crate::service::resolve::SearchPath>,
    ) -> AppService {
        use std::sync::Arc;

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
            search_paths,
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    // ── pkg_list tests ───────────────────────────────────────────

    #[tokio::test]
    async fn pkg_list_with_project() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create a project-local package.
        let pkg_dir = project_root.join("my_local_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        // Write alc.lock.
        let lock = make_lock_with_pkg("my_local_pkg");
        // Adjust path to be relative.
        let lock = LockFile {
            packages: vec![LockPackage {
                name: "my_local_pkg".to_string(),
                source: PackageSource::LocalDir {
                    path: "my_local_pkg".to_string(),
                },
                linked_at: "2026-04-08T12:00:00Z".to_string(),
            }],
            ..lock
        };
        super::super::lockfile::save_lockfile(project_root, &lock).unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_list(Some(project_root.to_string_lossy().to_string()))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let packages = json["packages"].as_array().unwrap();

        // Should have the project-local package.
        let project_pkg = packages
            .iter()
            .find(|p| p["name"] == "my_local_pkg")
            .expect("my_local_pkg not found in pkg_list output");

        assert_eq!(project_pkg["scope"], "project");
        assert_eq!(project_pkg["source_type"], "local_dir");
        assert_eq!(project_pkg["active"], true);

        // project_root and lockfile_path must be present.
        assert!(json["project_root"].is_string());
        assert!(json["lockfile_path"].is_string());
    }

    #[tokio::test]
    async fn pkg_list_no_project_root() {
        let svc = make_app_service().await;

        // Should succeed even without project_root (no crash).
        let result = svc.pkg_list(None).await.unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(json["packages"].is_array());
    }

    // ── pkg_remove tests ─────────────────────────────────────────

    #[tokio::test]
    async fn pkg_remove_project_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Create the physical directory (should remain after removal).
        let pkg_dir = project_root.join("my_local_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        // Write alc.lock with the package.
        let lock = LockFile {
            version: 1,
            packages: vec![LockPackage {
                name: "my_local_pkg".to_string(),
                source: PackageSource::LocalDir {
                    path: "my_local_pkg".to_string(),
                },
                linked_at: "2026-04-08T12:00:00Z".to_string(),
            }],
        };
        super::super::lockfile::save_lockfile(project_root, &lock).unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_remove(
                "my_local_pkg",
                Some(project_root.to_string_lossy().to_string()),
                None,
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["removed"], "my_local_pkg");
        assert_eq!(json["scope"], "project");

        // Physical directory must still exist.
        assert!(pkg_dir.exists(), "physical directory was deleted");

        // alc.lock must no longer contain the entry.
        let lock_after = load_lockfile(project_root).unwrap().unwrap();
        assert!(
            lock_after.packages.is_empty(),
            "alc.lock still contains the entry"
        );
    }

    #[tokio::test]
    async fn pkg_remove_project_scope_not_found_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path();

        // Write an alc.lock without the target package.
        let lock = make_lock_with_pkg("other_pkg");
        super::super::lockfile::save_lockfile(project_root, &lock).unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_remove(
                "nonexistent_pkg",
                Some(project_root.to_string_lossy().to_string()),
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found in alc.lock"));
    }

    /// A global package that exists on disk but is NOT registered in
    /// `installed.json` must NOT emit a `source_type` field.
    ///
    /// Previously the code wrote `source_type: "global"` (an invalid enum
    /// value) as a placeholder. After the typed DTO rewrite, absent manifest
    /// entries leave `source_type` out of the output entirely.
    #[tokio::test]
    async fn pkg_list_global_unregistered_has_no_source_type() {
        let tmp = tempfile::tempdir().unwrap();
        let search_dir = tmp.path().join("pkgs");
        std::fs::create_dir_all(&search_dir).unwrap();

        // Create a package directory with init.lua — but do NOT write
        // installed.json (simulating a hand-copied / ALC_PACKAGES_PATH package).
        let pkg_dir = search_dir.join("hand_copied_pkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(
            pkg_dir.join("init.lua"),
            "return { meta = { name = 'hand_copied_pkg' } }",
        )
        .unwrap();

        let search_path = crate::service::resolve::SearchPath {
            path: search_dir,
            source: crate::service::resolve::SearchPathSource::Env,
        };
        let svc = make_app_service_with_search_paths(vec![search_path]).await;
        let result = svc.pkg_list(None).await.unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let packages = json["packages"].as_array().unwrap();

        let pkg = packages
            .iter()
            .find(|p| p["name"] == "hand_copied_pkg")
            .expect("hand_copied_pkg not found in pkg_list output");

        // source_type must be absent (not "global" or any other invalid value).
        // serde_json::Value's Index impl returns Null for missing keys; we check
        // the underlying map directly to distinguish "absent" from "null".
        let pkg_map = pkg
            .as_object()
            .expect("package entry must be a JSON object");
        assert!(
            !pkg_map.contains_key("source_type"),
            "source_type should be absent for unregistered package, got: {:?}",
            pkg_map.get("source_type")
        );
        assert_eq!(pkg["scope"], "global");
        assert_eq!(pkg["active"], true);
    }
}
