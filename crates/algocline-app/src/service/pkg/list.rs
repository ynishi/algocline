//! `pkg_list` — enumerate installed packages (project-local + global).

use std::collections::HashMap;
use std::path::Path;

use super::super::alc_toml::{self, load_alc_toml};
use super::super::lockfile::{load_lockfile, lockfile_path};
use super::super::manifest;
use super::super::project::resolve_project_root;
use super::super::resolve::{is_system_package, packages_dir};
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
    /// Package version from alc.lock or meta evaluation.
    version: Option<String>,
    installed_at: Option<String>,
    updated_at: Option<String>,
    /// Legacy source string from `installed.json` (the raw URL/path).
    install_source: Option<String>,
    overrides: Option<Vec<String>>,
    meta: serde_json::Value,
    error: Option<String>,
    /// `Some(true)` when this package directory is a symlink (linked package).
    linked: Option<bool>,
    /// Resolved symlink target path (only present when `linked` is `Some(true)`).
    link_target: Option<String>,
    /// `Some(true)` when the symlink target does not exist (dangling symlink).
    broken: Option<bool>,
    /// Canonical absolute path of the Lua source directory for this package.
    /// Absent for broken entries or when canonicalization fails.
    resolved_source_path: Option<String>,
    /// Origin of `resolved_source_path`: one of `"installed"`, `"linked"`,
    /// `"local_path"`, or `"bundled"`. Future values may appear.
    resolved_source_kind: Option<String>,
    /// Canonical absolute paths of same-name packages that are shadowed by
    /// this (active) entry. Only present when overrides exist.
    override_paths: Option<Vec<String>>,
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

        if let Some(v) = self.version {
            map.insert("version".to_string(), serde_json::Value::String(v));
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
        if let Some(rsp) = self.resolved_source_path {
            map.insert(
                "resolved_source_path".to_string(),
                serde_json::Value::String(rsp),
            );
        }
        if let Some(rsk) = self.resolved_source_kind {
            map.insert(
                "resolved_source_kind".to_string(),
                serde_json::Value::String(rsk),
            );
        }
        if let Some(op) = self.override_paths {
            map.insert("override_paths".to_string(), serde_json::json!(op));
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

        if let Some(linked) = self.linked {
            map.insert("linked".to_string(), serde_json::Value::Bool(linked));
        }
        if let Some(target) = self.link_target {
            map.insert("link_target".to_string(), serde_json::Value::String(target));
        }
        if let Some(broken) = self.broken {
            map.insert("broken".to_string(), serde_json::Value::Bool(broken));
        }

        serde_json::Value::Object(map)
    }
}

impl AppService {
    /// List installed packages with metadata, showing the full override chain.
    ///
    /// When `project_root` is provided (or resolvable), project-local packages
    /// from `alc.toml` are prepended with `scope: "project"`, merged with
    /// version/source info from `alc.lock`. Global packages carry `scope: "global"`.
    /// If a project package and a global package share the same name, the project
    /// one is `active: true` and the global one `active: false`.
    pub async fn pkg_list(&self, project_root: Option<String>) -> Result<String, String> {
        // ── Load manifest once upfront ─────────────────────────────────────
        let manifest_data = manifest::load_manifest().unwrap_or_default();

        // ── Project-local packages (from alc.toml + alc.lock) ─────────────
        let resolved_root = resolve_project_root(project_root.as_deref());

        let mut project_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut entries: Vec<PackageListEntry> = Vec::new();
        let mut project_root_str: Option<String> = None;
        let mut lockfile_path_str: Option<String> = None;

        if let Some(ref root) = resolved_root {
            project_root_str = Some(root.display().to_string());
            lockfile_path_str = Some(lockfile_path(root).display().to_string());

            // Load alc.lock for version/source lookup (may not exist yet).
            let lock_map: HashMap<String, (Option<String>, PackageSource)> =
                match load_lockfile(root) {
                    Ok(Some(lock)) => lock
                        .packages
                        .into_iter()
                        .map(|p| (p.name, (p.version, p.source)))
                        .collect(),
                    Ok(None) => HashMap::new(),
                    Err(e) => {
                        tracing::warn!("failed to load alc.lock: {e}");
                        HashMap::new()
                    }
                };

            // Enumerate project packages from alc.toml declarations.
            match load_alc_toml(root) {
                Ok(Some(alc_toml)) => {
                    for (name, dep) in &alc_toml.packages {
                        let (version, source_type, abs_path) =
                            resolve_project_pkg_info(name, dep, &lock_map, root);
                        project_names.insert(name.clone());

                        // Resolve canonical source path depending on source_type.
                        let (rsp, rsk, resolve_err) = match source_type.as_deref() {
                            Some("path") => {
                                // abs_path is already absolutized; canonicalize it.
                                let rsp = abs_path
                                    .as_ref()
                                    .and_then(|p| resolve_source_path(std::path::Path::new(p)));
                                (rsp, Some("local_path".to_string()), None)
                            }
                            Some("bundled") => {
                                let (rsp, err) = match packages_dir() {
                                    Ok(dir) => (resolve_source_path(&dir.join(name)), None),
                                    Err(e) => {
                                        (None, Some(format!("cannot resolve packages_dir: {e}")))
                                    }
                                };
                                (rsp, Some("bundled".to_string()), err)
                            }
                            Some(_) => {
                                // "installed" or "git" → packages_dir/{name}
                                let (rsp, err) = match packages_dir() {
                                    Ok(dir) => (resolve_source_path(&dir.join(name)), None),
                                    Err(e) => {
                                        (None, Some(format!("cannot resolve packages_dir: {e}")))
                                    }
                                };
                                (rsp, Some("installed".to_string()), err)
                            }
                            None => (None, None, None),
                        };

                        entries.push(make_project_entry(
                            name.clone(),
                            version,
                            source_type,
                            abs_path,
                            rsp,
                            rsk,
                            resolve_err,
                        ));
                    }
                }
                Ok(None) => {
                    // No alc.toml — fall back to alc.lock Path entries for backward compat.
                    collect_path_entries_from_lock(
                        &lock_map,
                        root,
                        &mut project_names,
                        &mut entries,
                    );
                }
                Err(e) => {
                    tracing::warn!("failed to load alc.toml: {e}");
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

                // Detect symlink status before is_dir() check so dangling symlinks
                // are also enumerated (dangling symlinks have is_dir() == false).
                let is_symlink = path
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false);

                let link_target = if is_symlink {
                    path.read_link().ok().map(|t| t.display().to_string())
                } else {
                    None
                };

                // broken = symlink exists but target does not.
                let broken = if is_symlink {
                    Some(!path.exists())
                } else {
                    None
                };

                // For dangling symlinks: init.lua check will fail, so we allow
                // them through (they show as broken: true without init.lua check).
                // For non-symlinks and live symlinks: require is_dir().
                if !is_symlink && !path.is_dir() {
                    continue;
                }

                // Skip if no init.lua (only for non-broken entries).
                if broken != Some(true) && !path.join("init.lua").exists() {
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
                        let inferred = infer_from_legacy_source_string(&entry.source);
                        let st = match &inferred {
                            PackageSource::Git { .. } => "git".to_string(),
                            PackageSource::Installed => {
                                // I-6: supplement with original path/URL from installed.json
                                format!("installed (from: {})", entry.source)
                            }
                            PackageSource::Path { .. } => "path".to_string(),
                            PackageSource::Bundled { .. } => "bundled".to_string(),
                        };
                        (
                            Some(st),
                            Some(entry.installed_at.clone()),
                            Some(entry.updated_at.clone()),
                            Some(entry.source.clone()),
                        )
                    } else {
                        // Not registered in manifest → source_type absent
                        (None, None, None, None)
                    };

                // Resolve canonical source path for this global entry.
                let (resolved_source_path, resolved_source_kind) = if is_symlink {
                    // linked package
                    let kind = Some("linked".to_string());
                    if broken == Some(true) {
                        // dangling symlink — omit path, keep kind
                        (None, kind)
                    } else {
                        // resolve symlink target; make absolute if relative
                        let candidate = path.read_link().ok().map(|target| {
                            if target.is_absolute() {
                                target
                            } else {
                                sp.path.join(target)
                            }
                        });
                        let rsp = candidate.as_deref().and_then(resolve_source_path);
                        (rsp, kind)
                    }
                } else {
                    // normal (non-symlink) entry
                    let candidate = sp.path.join(&name);
                    let rsp = resolve_source_path(&candidate);
                    let kind = match source_type.as_deref() {
                        Some(st) if st.starts_with("bundled") => Some("bundled".to_string()),
                        _ => Some("installed".to_string()),
                    };
                    (rsp, kind)
                };

                entries.push(PackageListEntry {
                    name,
                    scope: Scope::Global,
                    source_type,
                    path: None,
                    source: Some(source_display),
                    active: global_active,
                    version: None,
                    installed_at,
                    updated_at,
                    install_source,
                    overrides: None,
                    meta,
                    error: eval_error,
                    linked: if is_symlink { Some(true) } else { None },
                    link_target,
                    broken,
                    resolved_source_path,
                    resolved_source_kind,
                    override_paths: None,
                });
            }
        }

        // ── Overrides pass (global packages only) ─────────────────────────
        // For each active global whose name appears in more than one search path,
        // record the lower-priority search-path paths as `overrides` (existing
        // behaviour) and the canonicalized pkg directories as `override_paths`
        // (new, §3.2-a).
        for entry in entries[global_start_idx..].iter_mut() {
            if !entry.active {
                continue;
            }
            if let Some(occurrences) = seen.get(&entry.name) {
                if occurrences.len() > 1 {
                    entry.overrides =
                        Some(occurrences.iter().skip(1).map(|(_, s)| s.clone()).collect());

                    // §3.2-a: canonicalized pkg directories for shadowed global entries.
                    let override_ps: Vec<String> = occurrences
                        .iter()
                        .skip(1)
                        .filter_map(|(idx, _)| {
                            let candidate = self.search_paths[*idx].path.join(&entry.name);
                            resolve_source_path(&candidate)
                        })
                        .collect();
                    if !override_ps.is_empty() {
                        entry.override_paths = Some(override_ps);
                    }
                }
            }
        }

        // ── Project-shadows-global pass (§3.2-b) ──────────────────────────
        // For each active project entry whose name also appears in global seen map,
        // expose all global occurrences as override_paths on the project entry.
        //
        // A project `installed` / `git` / `bundled` entry's own `resolved_source_path`
        // typically resolves to `packages_dir()/{name}`, which is itself one of the
        // search paths. Filter those occurrences out so an entry never lists itself
        // as a shadow target — `override_paths` should only contain genuinely distinct
        // same-name packages.
        for entry in entries[..global_start_idx].iter_mut() {
            let self_path = entry.resolved_source_path.as_deref();
            if let Some(occurrences) = seen.get(&entry.name) {
                let ps: Vec<String> = occurrences
                    .iter()
                    .filter_map(|(idx, _)| {
                        let candidate = self.search_paths[*idx].path.join(&entry.name);
                        resolve_source_path(&candidate)
                    })
                    .filter(|p| Some(p.as_str()) != self_path)
                    .collect();
                if !ps.is_empty() {
                    entry.override_paths = Some(ps);
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

// ─── Project package helpers ─────────────────────────────────────

/// Resolve version, source_type, and absolute path for a project package entry
/// by merging `alc.toml` dep declaration with `alc.lock` data.
fn resolve_project_pkg_info(
    name: &str,
    dep: &alc_toml::PackageDep,
    lock_map: &HashMap<String, (Option<String>, PackageSource)>,
    root: &Path,
) -> (Option<String>, Option<String>, Option<String>) {
    if let Some((ver, source)) = lock_map.get(name) {
        match source {
            PackageSource::Path { path: raw_path } => {
                let p = Path::new(raw_path);
                let abs = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    root.join(p)
                };
                (
                    ver.clone(),
                    Some("path".to_string()),
                    Some(abs.display().to_string()),
                )
            }
            PackageSource::Installed => (ver.clone(), Some("installed".to_string()), None),
            PackageSource::Git { .. } => (ver.clone(), Some("git".to_string()), None),
            PackageSource::Bundled { .. } => (ver.clone(), Some("bundled".to_string()), None),
        }
    } else {
        let st = match dep {
            alc_toml::PackageDep::Version(_) => Some("installed".to_string()),
            alc_toml::PackageDep::Path { .. } => Some("path".to_string()),
            alc_toml::PackageDep::Git { .. } => Some("git".to_string()),
        };
        (None, st, None)
    }
}

/// Create a `PackageListEntry` for a project-scoped package.
fn make_project_entry(
    name: String,
    version: Option<String>,
    source_type: Option<String>,
    abs_path: Option<String>,
    resolved_source_path: Option<String>,
    resolved_source_kind: Option<String>,
    error: Option<String>,
) -> PackageListEntry {
    PackageListEntry {
        name,
        scope: Scope::Project,
        source_type,
        path: abs_path,
        source: None,
        active: true,
        version,
        installed_at: None,
        updated_at: None,
        install_source: None,
        overrides: None,
        meta: serde_json::Value::Object(serde_json::Map::new()),
        error,
        linked: None,
        link_target: None,
        broken: None,
        resolved_source_path,
        resolved_source_kind,
        override_paths: None,
    }
}

/// Backward-compat fallback: collect `Path` entries from `alc.lock` when no `alc.toml` exists.
fn collect_path_entries_from_lock(
    lock_map: &HashMap<String, (Option<String>, PackageSource)>,
    root: &Path,
    project_names: &mut std::collections::HashSet<String>,
    entries: &mut Vec<PackageListEntry>,
) {
    for (name, (version, source)) in lock_map {
        if let PackageSource::Path { path: raw_path } = source {
            let p = Path::new(raw_path);
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                root.join(p)
            };
            project_names.insert(name.clone());
            let rsp = resolve_source_path(&abs);
            entries.push(make_project_entry(
                name.clone(),
                version.clone(),
                Some("path".to_string()),
                Some(abs.display().to_string()),
                rsp,
                Some("local_path".to_string()),
                None,
            ));
        }
    }
}

// ─── Path resolution ─────────────────────────────────────────────

/// Canonicalize `candidate` and return the canonical absolute path string,
/// or `None` on failure (broken symlink, race condition, missing dir, etc.).
/// The `kind` decision is left to the caller; this helper focuses solely on
/// the canonicalize step.
fn resolve_source_path(candidate: &std::path::Path) -> Option<String> {
    std::fs::canonicalize(candidate)
        .ok()
        .map(|p| p.display().to_string())
}

// ─── Name validation ─────────────────────────────────────────────

/// Returns `true` iff `name` is safe to interpolate into a Lua source string.
///
/// Accepts ASCII alphanumerics, `_` and `-`. Empty strings are rejected.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
