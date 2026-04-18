//! `pkg_list` — enumerate installed packages (project-local + global).

use std::collections::HashMap;
use std::path::Path;

use super::super::alc_toml::{self, load_alc_toml};
use super::super::list_opts::{
    apply_sort_by_value, matches_filter, parse_sort, project_fields, resolve_fields, ListOpts,
    PKG_LIST_FULL, PKG_LIST_SUMMARY,
};
use super::super::lockfile::{load_lockfile, lockfile_path};
use super::super::manifest;
use super::super::project::resolve_project_root;
use super::super::resolve::{is_system_package, packages_dir};
use super::super::source::{infer_from_legacy_source_string, PackageSource};
use super::super::AppService;

// ─── Intermediate DTO for pkg_list ───────────────────────────────

#[derive(Debug)]
enum Scope {
    /// Worktree-scoped override from `alc.local.toml` (gitignored).
    /// Highest priority — shadows same-name Project and Global entries.
    Variant,
    Project,
    Global,
}

/// Origin of a package's `resolved_source_path`.
///
/// Stringified form is part of the MCP wire contract (`resolved_source_kind`
/// field of `alc_pkg_list` entries). Adding a new variant is a backward-
/// compatible extension; renaming an existing one is a breaking change.
#[derive(Debug, Clone, Copy)]
enum ResolvedSourceKind {
    /// Package materialised under `packages_dir()` via git clone / copy.
    Installed,
    /// Symlink under `packages_dir()` or a search path (dev workflow).
    Linked,
    /// Project vendor directory referenced by `path = "..."` in alc.toml.
    LocalPath,
    /// Package shipped with algocline via `BUNDLED_SOURCES`.
    Bundled,
    /// Worktree-scoped override declared in `alc.local.toml`.
    Variant,
}

impl ResolvedSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            ResolvedSourceKind::Installed => "installed",
            ResolvedSourceKind::Linked => "linked",
            ResolvedSourceKind::LocalPath => "local_path",
            ResolvedSourceKind::Bundled => "bundled",
            ResolvedSourceKind::Variant => "variant",
        }
    }
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
    /// Origin of `resolved_source_path`. Serialised as the variant string
    /// (`"installed"` / `"linked"` / `"local_path"` / `"bundled"`).
    resolved_source_kind: Option<ResolvedSourceKind>,
    /// Canonical absolute paths of same-name packages that are shadowed by
    /// this (active) entry. Only present when overrides exist.
    override_paths: Option<Vec<String>>,
}

impl PackageListEntry {
    fn into_json(self) -> serde_json::Value {
        let scope_str = match self.scope {
            Scope::Variant => "variant",
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
                serde_json::Value::String(rsk.as_str().to_string()),
            );
        }
        if let Some(op) = self.override_paths {
            map.insert("override_paths".to_string(), serde_json::json!(op));
        }

        // All host-authoritative fields must be inserted BEFORE the meta
        // merge so `map.entry().or_insert` skips them — otherwise Lua meta
        // can masquerade as host-authoritative state (e.g. meta.linked
        // silently overriding the real symlink status).
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

        // Merge meta fields (Lua pkg.meta) into the top-level object.
        if let serde_json::Value::Object(meta_map) = self.meta {
            for (k, v) in meta_map {
                // Never let meta overwrite the fields we have already set.
                map.entry(k).or_insert(v);
            }
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
    ///
    /// `opts` carries the list-tool knob set (`limit / sort / filter /
    /// fields / verbose`); see [`super::super::list_opts`] for the
    /// projection / sort / filter primitives. Top-level keys
    /// (`packages`, `search_paths`, `project_root`, `lockfile_path`)
    /// are never projected away — only the per-entry objects inside
    /// `packages` are subject to projection.
    pub(crate) async fn pkg_list(
        &self,
        project_root: Option<String>,
        opts: ListOpts,
    ) -> Result<String, String> {
        // ── Resolve list-tool knobs up-front ─────────────────────────────
        // Validate sort / verbose strings before doing any filesystem IO
        // so user-input errors short-circuit fast.
        //
        // Default sort is `"-active,-installed_at"` (both descending):
        // - `-active` (desc) puts `active=true` first, `active=false` last
        //   (bool DESC: true > false in apply_sort_by_value).
        // - `-installed_at` (desc) breaks ties with newest install first.
        // The plan.md §3.3 prose says "active=true 先頭"; using `-active`
        // (DESC) is the only way to satisfy that with the bool ordering
        // contract — see context-st2.md Pitfall #3.
        let sort_str = opts.sort.as_deref().unwrap_or("-active,-installed_at");
        let sort_keys = parse_sort(sort_str)?;
        let fields = resolve_fields(
            opts.verbose.as_deref(),
            opts.fields.as_deref(),
            PKG_LIST_SUMMARY,
            PKG_LIST_FULL,
        )?;

        // ── Load manifest once upfront ─────────────────────────────────────
        let manifest_data = manifest::load_manifest().unwrap_or_default();

        // ── Project-local packages (from alc.toml + alc.lock) ─────────────
        let resolved_root = resolve_project_root(project_root.as_deref());

        let mut project_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut variant_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut entries: Vec<PackageListEntry> = Vec::new();
        let mut project_root_str: Option<String> = None;
        let mut lockfile_path_str: Option<String> = None;

        if let Some(ref root) = resolved_root {
            project_root_str = Some(root.display().to_string());
            lockfile_path_str = Some(lockfile_path(root).display().to_string());

            // Variant pkgs from alc.local.toml (worktree-scoped, gitignored).
            // Highest priority — recorded first so they shadow same-name
            // project / global entries via `variant_names` set.
            collect_variant_entries(root, &mut variant_names, &mut entries);

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
                        // `path` → vendor dir from alc.toml; everything else
                        // (`installed` / `git` / `bundled`) resolves under
                        // `packages_dir()/{name}` and differs only in `kind`.
                        let (rsp, rsk, resolve_err): (
                            Option<String>,
                            Option<ResolvedSourceKind>,
                            Option<String>,
                        ) = match source_type.as_deref() {
                            Some("path") => {
                                let rsp = abs_path
                                    .as_ref()
                                    .and_then(|p| resolve_source_path(std::path::Path::new(p)));
                                (rsp, Some(ResolvedSourceKind::LocalPath), None)
                            }
                            Some(st) => {
                                let kind = if st == "bundled" {
                                    ResolvedSourceKind::Bundled
                                } else {
                                    ResolvedSourceKind::Installed
                                };
                                match packages_dir() {
                                    Ok(dir) => {
                                        (resolve_source_path(&dir.join(name)), Some(kind), None)
                                    }
                                    Err(e) => (
                                        None,
                                        Some(kind),
                                        Some(format!("cannot resolve packages_dir: {e}")),
                                    ),
                                }
                            }
                            None => (None, None, None),
                        };

                        let mut entry = make_project_entry(
                            name.clone(),
                            version,
                            source_type,
                            abs_path,
                            rsp,
                            rsk,
                            resolve_err,
                        );
                        if variant_names.contains(name) {
                            entry.active = false;
                        }
                        entries.push(entry);
                    }
                }
                Ok(None) => {
                    // No alc.toml — fall back to alc.lock Path entries for backward compat.
                    collect_path_entries_from_lock(
                        &lock_map,
                        root,
                        &variant_names,
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
                //
                // `try_exists()` distinguishes Err (IO / permission failure)
                // from Ok(false) (confirmed non-existent). On Err we cannot
                // prove the target is intact, so we conservatively report
                // `broken: true` — the user cannot use the target either
                // way, so the signal is more useful than silently hiding
                // the symlink. `path.exists()` collapsed these cases.
                let broken = if is_symlink {
                    Some(!path.try_exists().unwrap_or(false))
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
                // by project-local or variant if same name
                let global_active = seen[&name].len() == 1
                    && !project_names.contains(&name)
                    && !variant_names.contains(&name);

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
                let (resolved_source_path, resolved_source_kind): (
                    Option<String>,
                    Option<ResolvedSourceKind>,
                ) = if is_symlink {
                    let kind = Some(ResolvedSourceKind::Linked);
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
                        Some("bundled") => ResolvedSourceKind::Bundled,
                        _ => ResolvedSourceKind::Installed,
                    };
                    (rsp, Some(kind))
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
        let mut all_packages: Vec<serde_json::Value> =
            entries.into_iter().map(|e| e.into_json()).collect();

        // ── List-tool pipeline: filter → sort → truncate → project ──────
        // Applied to the per-entry `packages` array only. Top-level
        // shape (`search_paths`, `project_root`, `lockfile_path`) is
        // never projected — see context-st2.md.
        if let Some(ref filter_map) = opts.filter {
            if !filter_map.is_empty() {
                all_packages.retain(|v| matches_filter(v, filter_map));
            }
        }

        apply_sort_by_value(&mut all_packages, &sort_keys);

        let limit = opts.limit.unwrap_or(50);
        all_packages.truncate(limit);

        let projected: Vec<serde_json::Value> = all_packages
            .into_iter()
            .map(|v| project_fields(v, &fields))
            .collect();

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
            "packages": projected,
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

/// Enumerate variant pkgs from `alc.local.toml` and push them as
/// `Scope::Variant` entries.
///
/// Variant pkgs are worktree-scoped (gitignored) overrides resolved by
/// `algocline_engine::VariantPkg`. They have the highest priority — same-name
/// project / global entries are demoted to `active: false`.
///
/// Failures (missing / malformed `alc.local.toml`) are logged at `warn` and
/// degrade to no variant entries (consistent with `resolve_extra_lib_paths`).
fn collect_variant_entries(
    root: &Path,
    variant_names: &mut std::collections::HashSet<String>,
    entries: &mut Vec<PackageListEntry>,
) {
    let local = match alc_toml::load_alc_local_toml(root) {
        Ok(Some(l)) => l,
        Ok(None) => return,
        Err(e) => {
            tracing::warn!("failed to load alc.local.toml at {}: {e}", root.display());
            return;
        }
    };

    for vp in alc_toml::resolve_local_variant_pkgs(root, &local) {
        variant_names.insert(vp.name.clone());
        let abs_path = vp.pkg_dir.display().to_string();
        let rsp = resolve_source_path(&vp.pkg_dir);
        entries.push(PackageListEntry {
            name: vp.name,
            scope: Scope::Variant,
            source_type: Some("path".to_string()),
            path: Some(abs_path),
            source: None,
            active: true,
            version: None,
            installed_at: None,
            updated_at: None,
            install_source: None,
            overrides: None,
            meta: serde_json::Value::Object(serde_json::Map::new()),
            error: None,
            linked: None,
            link_target: None,
            broken: None,
            resolved_source_path: rsp,
            resolved_source_kind: Some(ResolvedSourceKind::Variant),
            override_paths: None,
        });
    }
}

/// Create a `PackageListEntry` for a project-scoped package.
fn make_project_entry(
    name: String,
    version: Option<String>,
    source_type: Option<String>,
    abs_path: Option<String>,
    resolved_source_path: Option<String>,
    resolved_source_kind: Option<ResolvedSourceKind>,
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
    variant_names: &std::collections::HashSet<String>,
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
            let mut entry = make_project_entry(
                name.clone(),
                version.clone(),
                Some("path".to_string()),
                Some(abs.display().to_string()),
                rsp,
                Some(ResolvedSourceKind::LocalPath),
                None,
            );
            if variant_names.contains(name) {
                entry.active = false;
            }
            entries.push(entry);
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
