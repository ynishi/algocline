//! `pkg_list` — enumerate installed packages (project-local + global).

use std::collections::HashMap;
use std::path::Path;

use super::super::lockfile::{load_lockfile, lockfile_path};
use super::super::manifest;
use super::super::project::resolve_project_root;
use super::super::resolve::is_system_package;
use super::super::source::{infer_from_legacy_source_string, PackageSource};
use super::super::AppService;

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
                    tracing::warn!("failed to load alc.lock: {e}");
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
