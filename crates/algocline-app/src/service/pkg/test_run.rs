//! `pkg_test` — run mlua-lspec tests for a package, a file, or inline code.
//!
//! # Input routing
//!
//! Exactly one of `pkg`, `code_file`, or `code` must be provided:
//! - **`pkg`**: discovers `*_spec.lua` files under `<pkg_root>/<spec_dir>/`
//!   (default `"spec"`), runs each in its own mlua VM sequentially inside a
//!   single `spawn_blocking` task.
//! - **`code_file`**: reads the given absolute path and runs it as a single
//!   test file.
//! - **`code`**: runs inline Lua source as a single test (chunk name
//!   `"@inline.lua"`).
//!
//! # Two-tier error model
//!
//! - **Per-spec-file crashes** (mlua-lspec `Err`): absorbed — `failed += 1`,
//!   a synthetic error entry is appended to `tests`, and execution continues.
//! - **Setup failures** (VM init, pkg not found, zero spec files, I/O errors,
//!   `spawn_blocking` panic): propagated as typed `Err(String)` to MCP wire.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use algocline_engine::bridge as engine_bridge;
use mlua::{Lua, Value as LuaValue, Variadic};
use mlua_lspec::{doubles, framework};
use serde::Serialize;
use serde_json::{json, Value};
use tracing::warn;

use super::super::alc_toml::{load_alc_local_toml, load_alc_toml, PackageDep};
use super::super::AppService;

// ─── Auto search path types ───────────────────────────────────────────────────

/// Source of an auto-resolved search path entry.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AutoSearchPathSource {
    /// Path comes from the installed packages directory (`~/.algocline/packages/`).
    Installed,
    /// Path comes from a `[packages]` path entry in `alc.toml`.
    #[serde(rename = "alc.toml")]
    AlcToml,
    /// Path comes from a `[packages]` path entry in `alc.local.toml`.
    #[serde(rename = "alc.local.toml")]
    AlcLocalToml,
}

/// A single package-name → parent-dir mapping returned by auto-resolution.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResolvedSearchPath {
    /// Package name as declared in the registry.
    pub name: String,
    /// Canonicalized parent directory that should be added to `package.path`.
    pub search_dir: String,
    /// Which registry source this entry came from.
    pub source: AutoSearchPathSource,
}

impl AppService {
    /// Collect auto-resolved `package.path` directories from all three
    /// registry sources.
    ///
    /// # Arguments
    ///
    /// * `project_root` — optional project root string; if `None` (or if root
    ///   resolution fails), `alc.toml` / `alc.local.toml` sources are skipped
    ///   and only installed packages are returned.
    ///
    /// # Returns
    ///
    /// `(mapping, warnings)` where:
    /// - `mapping` — one row per package, preserving per-package detail even
    ///   when multiple packages share the same parent directory.
    /// - `warnings` — non-fatal errors encountered during resolution (parse
    ///   failures, canonicalize errors, etc.).
    ///
    /// # Errors
    ///
    /// Never returns `Err`; all errors are surfaced in the `warnings` vector.
    pub(crate) fn collect_auto_search_paths(
        &self,
        project_root: Option<&str>,
    ) -> (Vec<ResolvedSearchPath>, Vec<String>) {
        let mut results: Vec<ResolvedSearchPath> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        // Track parent dirs already added to avoid injecting the same dir
        // multiple times into `package.path` (dir-level dedupe).
        // Note: `results` still retains one row per package for diagnostic
        // purposes — the dedupe only applies to the actual dirs injected.
        let mut seen_dirs: HashSet<PathBuf> = HashSet::new();

        // ── Source 1: ~/.algocline/packages/ sub-dirs ─────────────────────────
        let app_dir = self.log_config.app_dir();
        let pkg_dir = app_dir.packages_dir();
        match std::fs::read_dir(&pkg_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    // Only directories (including symlinks to dirs).
                    let is_dir = path.metadata().map(|m| m.is_dir()).unwrap_or(false);
                    if !is_dir {
                        continue;
                    }
                    // Only include dirs that contain an `init.lua`.
                    if !path.join("init.lua").exists() {
                        continue;
                    }
                    let pkg_name = entry.file_name().to_string_lossy().into_owned();
                    // The search dir is the parent of the pkg dir, i.e. pkg_dir
                    // itself (so `require("pkg_name")` resolves to
                    // `pkg_dir/pkg_name/init.lua`).
                    let search_dir = pkg_dir.clone();
                    results.push(ResolvedSearchPath {
                        name: pkg_name,
                        search_dir: search_dir.to_string_lossy().into_owned(),
                        source: AutoSearchPathSource::Installed,
                    });
                    seen_dirs.insert(search_dir);
                }
            }
            Err(e) => {
                warnings.push(format!(
                    "failed to read packages dir {}: {e}",
                    pkg_dir.display()
                ));
            }
        }

        // ── Sources 2 + 3: alc.toml and alc.local.toml ───────────────────────
        // Only available when project root can be resolved.
        let resolved_root = self.resolve_root(project_root);
        if let Some(ref root) = resolved_root {
            // Source 2: alc.toml [packages] path entries
            match load_alc_toml(root) {
                Ok(Some(toml_data)) => {
                    for (name, dep) in &toml_data.packages {
                        let PackageDep::Path { path, .. } = dep else {
                            continue;
                        };
                        let raw = std::path::Path::new(path);
                        let abs = if raw.is_absolute() {
                            raw.to_path_buf()
                        } else {
                            root.join(raw)
                        };
                        match abs.canonicalize() {
                            Ok(canonical_pkg_dir) => {
                                // The pkg_dir is the package directory itself
                                // (e.g. `.../packages/swarm_frame`). The search
                                // dir for `require` is its parent
                                // (`.../packages/`).
                                let search_dir = canonical_pkg_dir
                                    .parent()
                                    .map(|p| p.to_path_buf())
                                    .unwrap_or_else(|| canonical_pkg_dir.clone());
                                results.push(ResolvedSearchPath {
                                    name: name.clone(),
                                    search_dir: search_dir.to_string_lossy().into_owned(),
                                    source: AutoSearchPathSource::AlcToml,
                                });
                                seen_dirs.insert(search_dir);
                            }
                            Err(e) => {
                                warnings.push(format!(
                                    "cannot canonicalize alc.toml path entry for '{}' ({}): {e}",
                                    name,
                                    abs.display()
                                ));
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warnings.push(format!(
                        "failed to load alc.toml at {}: {e}",
                        root.display()
                    ));
                }
            }

            // Source 3: alc.local.toml [packages] path entries
            match load_alc_local_toml(root) {
                Ok(Some(local_data)) => {
                    for (name, dep) in &local_data.packages {
                        let PackageDep::Path { path, .. } = dep else {
                            continue;
                        };
                        let raw = std::path::Path::new(path);
                        let abs = if raw.is_absolute() {
                            raw.to_path_buf()
                        } else {
                            root.join(raw)
                        };
                        match abs.canonicalize() {
                            Ok(canonical_pkg_dir) => {
                                let search_dir = canonical_pkg_dir
                                    .parent()
                                    .map(|p| p.to_path_buf())
                                    .unwrap_or_else(|| canonical_pkg_dir.clone());
                                results.push(ResolvedSearchPath {
                                    name: name.clone(),
                                    search_dir: search_dir.to_string_lossy().into_owned(),
                                    source: AutoSearchPathSource::AlcLocalToml,
                                });
                                seen_dirs.insert(search_dir);
                            }
                            Err(e) => {
                                warnings.push(format!(
                                    "cannot canonicalize alc.local.toml path entry for '{}' ({}): {e}",
                                    name,
                                    abs.display()
                                ));
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warnings.push(format!(
                        "failed to load alc.local.toml at {}: {e}",
                        root.display()
                    ));
                }
            }
        }
        // When resolved_root is None: alc.toml / alc.local.toml are skipped
        // gracefully (no warning added — Crux constraint 1 note).

        (results, warnings)
    }

    /// Run mlua-lspec tests for a package, a single file, or inline code.
    ///
    /// Exactly one of `pkg`, `code_file`, `code` must be provided.
    /// Zero or more than one returns a typed `Err`.
    ///
    /// # Arguments
    ///
    /// * `pkg` — installed package name. Spec files are discovered under
    ///   `<pkg_root>/<spec_dir>/*_spec.lua` (default `spec_dir = "spec"`).
    /// * `code_file` — absolute path to a single `.lua` test file.
    /// * `code` — inline Lua source code containing lspec tests.
    /// * `spec_dir` — subdirectory inside the pkg root for spec files
    ///   (default `"spec"`). Only used when `pkg` is provided.
    /// * `filter` — substring filter on spec file stems (only for `pkg`).
    /// * `search_paths` — additional dirs prepended to `package.path` inside
    ///   the Lua VM. These are appended *after* auto-resolved paths.
    /// * `project_root` — optional project root for variant-scope resolution
    ///   (`alc.local.toml`). Falls back to ancestor walk from cwd.
    /// * `auto_search_paths` — when `true` (default) or `None`, auto-prepends
    ///   parent dirs of all linked/installed packages (installed
    ///   `~/.algocline/packages/`, `alc.toml` path entries, `alc.local.toml`
    ///   path entries) to `package.path`. When `false`, no auto-resolve is
    ///   performed. Resolved mapping is returned in the JSON response
    ///   `resolved_search_paths` field.
    ///
    /// # Returns
    ///
    /// On success: JSON string with shape
    /// `{passed, failed, pending, total, duration_ms, spec_files: [{path,
    /// passed, failed, total, duration_ms, tests: [{suite, name, passed,
    /// pending, error}]}], resolved_search_paths: [{name, search_dir,
    /// source}], search_path_warnings?: [...]}`.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` for setup failures (VM init, pkg not found, zero
    /// spec files, I/O errors, `spawn_blocking` panic). Per-spec Lua crashes
    /// are absorbed, not propagated.
    // 9 parameters are justified by the MCP wire shape: 3 mutually exclusive
    // input sources (pkg / code_file / code) plus filtering/path/auto-resolve
    // options.
    #[allow(clippy::too_many_arguments)]
    pub async fn pkg_test(
        &self,
        pkg: Option<String>,
        code_file: Option<String>,
        code: Option<String>,
        spec_dir: Option<String>,
        filter: Option<String>,
        search_paths: Option<Vec<String>>,
        project_root: Option<String>,
        auto_search_paths: Option<bool>,
    ) -> Result<String, String> {
        // ── Crux constraint 1: exactly-one input exclusivity ──────────────────
        let input_count = pkg.is_some() as u8 + code_file.is_some() as u8 + code.is_some() as u8;
        if input_count != 1 {
            return Err("pkg_test: provide exactly one of pkg, code_file, code".to_string());
        }

        let caller_search_paths: Vec<String> = search_paths.unwrap_or_default();

        // ── Auto-resolve: collect parent dirs from all 3 registry sources ─────
        // Crux constraint 2: when auto_search_paths == Some(false), skip
        // entirely (zero I/O, zero injection).
        let (resolved_mapping, search_path_warnings) = if auto_search_paths == Some(false) {
            (Vec::new(), Vec::new())
        } else {
            self.collect_auto_search_paths(project_root.as_deref())
        };

        // Deduplicate auto-resolved parent dirs (dir-level, not pkg-level)
        // to avoid duplicate entries in package.path.
        let mut seen_auto_dirs: HashSet<&str> = HashSet::new();
        let auto_dirs: Vec<String> = resolved_mapping
            .iter()
            .filter_map(|r| {
                if seen_auto_dirs.insert(r.search_dir.as_str()) {
                    Some(r.search_dir.clone())
                } else {
                    None
                }
            })
            .collect();

        if let Some(inline_code) = code {
            // `code` path: single VM, inline source.
            // Order: [auto..., caller...]
            let mut search = auto_dirs;
            search.extend(caller_search_paths);
            let result_json = run_inline(inline_code, search).await?;
            Ok(attach_resolved_meta(
                result_json,
                &resolved_mapping,
                &search_path_warnings,
            ))
        } else if let Some(file_path) = code_file {
            // `code_file` path: read file then run.
            // Order: [file_parent, auto..., caller...]
            let abs_path = PathBuf::from(&file_path);
            let src = std::fs::read_to_string(&abs_path)
                .map_err(|e| format!("pkg_test: failed to read {file_path}: {e}"))?;
            let parent = abs_path
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let chunk_name = format!("@{file_path}");
            let mut paths = vec![parent];
            paths.extend(auto_dirs);
            paths.extend(caller_search_paths);
            let result_json = run_single_spec(src, chunk_name, paths).await?;
            Ok(attach_resolved_meta(
                result_json,
                &resolved_mapping,
                &search_path_warnings,
            ))
        } else {
            // `pkg` path: spec_dir scan.
            // input_count == 1 and neither `code` nor `code_file` is Some,
            // so `pkg` must be Some here by the exclusivity check above.
            let Some(pkg_name) = pkg else {
                unreachable!("pkg must be Some: input_count==1 and code/code_file are None")
            };
            let init_path = self
                .pkg_resolve_init_path(&pkg_name, project_root.as_deref())
                .map_err(|e| format!("pkg_test: {e}"))?
                .ok_or_else(|| {
                    format!(
                        "pkg_test: package '{pkg_name}' not found in <project_root>/<name>/, alc.local.toml, or ~/.algocline/packages/"
                    )
                })?;
            let pkg_root = init_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| init_path.clone());

            let spec_subdir = spec_dir.as_deref().unwrap_or("spec");
            let spec_dir_path = pkg_root.join(spec_subdir);

            // Collect *_spec.lua files deterministically.
            let spec_files = collect_spec_files(&spec_dir_path, filter.as_deref())?;

            // Order: [pkg_root, auto..., caller...]
            let pkg_root_str = pkg_root.to_string_lossy().into_owned();
            let mut search = vec![pkg_root_str];
            search.extend(auto_dirs);
            search.extend(caller_search_paths);

            let result_json = run_pkg_specs(spec_files, search).await?;
            Ok(attach_resolved_meta(
                result_json,
                &resolved_mapping,
                &search_path_warnings,
            ))
        }
    }
}

/// Attach `resolved_search_paths` (and optionally `search_path_warnings`) to a
/// JSON result string returned by the internal run helpers.
///
/// # Arguments
///
/// * `result_json` — the JSON string produced by `run_inline`,
///   `run_single_spec`, or `run_pkg_specs`.
/// * `resolved_mapping` — per-package mapping collected by
///   `collect_auto_search_paths`.
/// * `warnings` — non-fatal warnings from auto-resolution.
///
/// # Returns
///
/// A new JSON string with `resolved_search_paths` added. When `warnings` is
/// non-empty, `search_path_warnings` is also added.
fn attach_resolved_meta(
    result_json: String,
    resolved_mapping: &[ResolvedSearchPath],
    warnings: &[String],
) -> String {
    let mut obj: Value = match serde_json::from_str(&result_json) {
        Ok(v) => v,
        Err(e) => {
            warn!("attach_resolved_meta: failed to parse result JSON: {e}");
            return result_json;
        }
    };
    if let Some(map) = obj.as_object_mut() {
        // Crux constraint 3: resolved_search_paths must appear in the JSON
        // return value as a structured field (not only in logs).
        let rows: Vec<Value> = resolved_mapping
            .iter()
            .map(|r| {
                json!({
                    "name": r.name,
                    "search_dir": r.search_dir,
                    "source": serde_json::to_value(&r.source)
                        .unwrap_or(Value::String(String::new()))
                })
            })
            .collect();
        map.insert("resolved_search_paths".to_string(), Value::Array(rows));
        if !warnings.is_empty() {
            map.insert(
                "search_path_warnings".to_string(),
                Value::Array(warnings.iter().map(|w| Value::String(w.clone())).collect()),
            );
        }
    }
    obj.to_string()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Collect `*_spec.lua` entries from `spec_dir_path`, sorted deterministically.
///
/// Returns `Err` if the directory does not exist or zero files remain after
/// applying `filter`.
///
/// # Arguments
///
/// * `spec_dir_path` — absolute path to the spec directory.
/// * `filter` — optional substring matched against the file stem (e.g.
///   `"shape"` matches `shape_spec.lua`).
///
/// # Errors
///
/// Returns `Err(String)` when the directory is absent or no spec files match.
fn collect_spec_files(spec_dir_path: &Path, filter: Option<&str>) -> Result<Vec<PathBuf>, String> {
    if !spec_dir_path.exists() {
        return Err(format!(
            "pkg_test: no spec files found in {} (looked for *_spec.lua)",
            spec_dir_path.display()
        ));
    }

    let read_result = std::fs::read_dir(spec_dir_path).map_err(|e| {
        format!(
            "pkg_test: failed to read spec dir {}: {e}",
            spec_dir_path.display()
        )
    })?;

    let mut set = BTreeSet::new();
    for entry in read_result.flatten() {
        let fname = entry.file_name();
        let name_str = fname.to_string_lossy();
        if name_str.ends_with("_spec.lua") {
            if let Some(f) = filter {
                let stem = name_str.trim_end_matches("_spec.lua");
                if !stem.contains(f) {
                    continue;
                }
            }
            set.insert(entry.path());
        }
    }

    if set.is_empty() {
        return Err(format!(
            "pkg_test: no spec files found in {} (looked for *_spec.lua)",
            spec_dir_path.display()
        ));
    }

    Ok(set.into_iter().collect())
}

/// Run inline Lua code as a single lspec test.
///
/// `framework::run_tests` pre-loads the `lust` global — no separate
/// `framework::register` call is needed (and calling it would risk double
/// registration).
///
/// # Errors
///
/// Returns `Err` if the blocking task panics.
async fn run_inline(code: String, search_paths: Vec<String>) -> Result<String, String> {
    run_single_spec(code, "@inline.lua".to_string(), search_paths).await
}

/// Execute one spec source string inside a fresh mlua VM on a blocking thread.
///
/// Per-spec Lua crashes are represented as a failing test entry (crux
/// constraint 2 — per-spec absorption) rather than propagating the error.
///
/// # Arguments
///
/// * `code` — Lua source to execute.
/// * `chunk_name` — Lua chunk name convention: `"@inline.lua"` or
///   `"@<abs_path>"`.
/// * `search_paths` — directories prepended to `package.path` inside the VM.
///
/// # Errors
///
/// Run a single Lua spec inside an `alc_pkg_test` sandbox VM.
///
/// Equivalent to [`mlua_lspec::framework::run_tests`], but the VM is
/// pre-loaded with the full `alc.*` primitive surface via
/// [`algocline_engine::bridge::install_for_pkg_test`] plus the Pure-Lua
/// mock layer (`with_alc` / `alc_mock` / `alc.spy`).  This guarantees
/// `production primitive surface ⊆ test sandbox primitive surface`, so
/// specs can call `alc.json_encode` etc. without inline workarounds
/// (fix for issue 7dc77cc7 — alc_run / alc_pkg_test VM狭広逆転).
///
/// # Errors
///
/// Returns the same `String` error shape as `framework::run_tests` for
/// per-spec crashes (registration / search-path / load / collect-results).
fn run_pkg_test_in_sandbox(
    code: &str,
    chunk_name: &str,
    search_paths: &[&str],
) -> Result<mlua_lspec::TestSummary, String> {
    let lua = Lua::new();

    engine_bridge::install_for_pkg_test(&lua)
        .map_err(|e| format!("Failed to install alc.* sandbox: {e}"))?;
    framework::register(&lua).map_err(|e| format!("Failed to register test framework: {e}"))?;
    doubles::register(&lua).map_err(|e| format!("Failed to register test doubles: {e}"))?;

    // Prepend search paths (mirrors mlua_lspec::framework::prepend_search_paths,
    // which is private).
    if !search_paths.is_empty() {
        let package: mlua::Table = lua
            .globals()
            .get("package")
            .map_err(|e| format!("Failed to get package table: {e}"))?;
        let current: String = package
            .get("path")
            .map_err(|e| format!("Failed to get package.path: {e}"))?;
        let mut prefix = String::new();
        for dir in search_paths {
            let dir = dir.trim_end_matches('/');
            prefix.push_str(dir);
            prefix.push_str("/?.lua;");
            prefix.push_str(dir);
            prefix.push_str("/?/init.lua;");
        }
        prefix.push_str(&current);
        package
            .set("path", prefix)
            .map_err(|e| format!("Failed to set package.path: {e}"))?;
    }

    // Suppress lust's print() output so stdio MCP transports stay clean.
    lua.globals()
        .set(
            "print",
            lua.create_function(|_, _: Variadic<LuaValue>| Ok(()))
                .map_err(|e| format!("Failed to override print: {e}"))?,
        )
        .map_err(|e| format!("Failed to override print: {e}"))?;

    lua.load(code)
        .set_name(chunk_name)
        .exec()
        .map_err(|e| format!("Test execution error: {e}"))?;

    let summary =
        framework::collect_results(&lua).map_err(|e| format!("Failed to collect results: {e}"))?;

    Ok(summary)
}

/// Returns `Err` when the task panics.
async fn run_single_spec(
    code: String,
    chunk_name: String,
    search_paths: Vec<String>,
) -> Result<String, String> {
    let total_start = Instant::now();

    let (spec_file_entry, agg_passed, agg_failed) = tokio::task::spawn_blocking(move || {
        // mlua::Lua is constructed per-spec inside `run_pkg_test_in_sandbox`
        // (which installs `alc.*` + the mock layer).  No external VM here.

        let search_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();
        let spec_start = Instant::now();

        let (tests_json, passed, failed) =
            match run_pkg_test_in_sandbox(&code, &chunk_name, &search_refs) {
                Ok(summary) => {
                    let tests: Vec<Value> = summary
                        .tests
                        .iter()
                        .map(|t| {
                            json!({
                                "suite": t.suite,
                                "name": t.name,
                                "passed": t.passed,
                                "pending": false,
                                "error": t.error
                            })
                        })
                        .collect();
                    (tests, summary.passed, summary.failed)
                }
                Err(e) => {
                    // Per-spec crash: absorbed, contributes 1 failing entry.
                    let tests = vec![json!({
                        "suite": chunk_name,
                        "name": "<top-level>",
                        "passed": false,
                        "pending": false,
                        "error": e.to_string()
                    })];
                    (tests, 0usize, 1usize)
                }
            };

        let spec_duration_ms = spec_start.elapsed().as_millis() as u64;
        let total_tests = passed + failed;

        let spec_entry = json!({
            "path": chunk_name,
            "passed": passed,
            "failed": failed,
            "total": total_tests,
            "duration_ms": spec_duration_ms,
            "tests": tests_json
        });

        (spec_entry, passed, failed)
    })
    .await
    .map_err(|e| format!("pkg_test: blocking task panicked: {e}"))?;

    let duration_ms = total_start.elapsed().as_millis() as u64;

    let result = json!({
        "passed": agg_passed,
        "failed": agg_failed,
        "pending": 0,
        "total": agg_passed + agg_failed,
        "duration_ms": duration_ms,
        "spec_files": [spec_file_entry]
    });

    Ok(result.to_string())
}

/// Run multiple spec files sequentially inside a single `spawn_blocking` task.
///
/// Each spec file gets its own fresh mlua VM. Per-spec crashes are absorbed
/// (crux constraint 2). Aggregate counts and per-file entries are returned.
///
/// # Arguments
///
/// * `spec_files` — sorted list of absolute paths to `*_spec.lua` files.
/// * `search_paths` — directories prepended to `package.path` inside each VM.
///
/// # Errors
///
/// Returns `Err` if the task panics.
async fn run_pkg_specs(
    spec_files: Vec<PathBuf>,
    search_paths: Vec<String>,
) -> Result<String, String> {
    let total_start = Instant::now();

    let (spec_entries, agg_passed, agg_failed) = tokio::task::spawn_blocking(move || {
        let mut entries: Vec<Value> = Vec::new();
        let mut total_passed = 0usize;
        let mut total_failed = 0usize;

        for spec_path in &spec_files {
            let path_str = spec_path.to_string_lossy().to_string();
            let code = match std::fs::read_to_string(spec_path) {
                Ok(s) => s,
                Err(e) => {
                    // I/O failure for this spec file: absorbed as failing entry.
                    entries.push(json!({
                        "path": path_str,
                        "passed": 0,
                        "failed": 1,
                        "total": 1,
                        "duration_ms": 0,
                        "tests": [{
                            "suite": path_str,
                            "name": "<top-level>",
                            "passed": false,
                            "pending": false,
                            "error": format!("pkg_test: failed to read {path_str}: {e}")
                        }]
                    }));
                    total_failed += 1;
                    continue;
                }
            };

            let chunk_name = format!("@{path_str}");
            let search_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

            // mlua::Lua is constructed per-spec inside `run_pkg_test_in_sandbox`.
            let spec_start = Instant::now();

            let (tests_json, passed, failed) =
                match run_pkg_test_in_sandbox(&code, &chunk_name, &search_refs) {
                    Ok(summary) => {
                        let tests: Vec<Value> = summary
                            .tests
                            .iter()
                            .map(|t| {
                                json!({
                                    "suite": t.suite,
                                    "name": t.name,
                                    "passed": t.passed,
                                    "pending": false,
                                    "error": t.error
                                })
                            })
                            .collect();
                        (tests, summary.passed, summary.failed)
                    }
                    Err(e) => {
                        // Per-spec crash: absorbed per crux constraint 2.
                        let tests = vec![json!({
                            "suite": chunk_name,
                            "name": "<top-level>",
                            "passed": false,
                            "pending": false,
                            "error": e.to_string()
                        })];
                        (tests, 0usize, 1usize)
                    }
                };

            let spec_duration_ms = spec_start.elapsed().as_millis() as u64;
            let total_tests = passed + failed;

            entries.push(json!({
                "path": path_str,
                "passed": passed,
                "failed": failed,
                "total": total_tests,
                "duration_ms": spec_duration_ms,
                "tests": tests_json
            }));

            total_passed += passed;
            total_failed += failed;
        }

        (entries, total_passed, total_failed)
    })
    .await
    .map_err(|e| format!("pkg_test: blocking task panicked: {e}"))?;

    let duration_ms = total_start.elapsed().as_millis() as u64;

    let result = json!({
        "passed": agg_passed,
        "failed": agg_failed,
        "pending": 0,
        "total": agg_passed + agg_failed,
        "duration_ms": duration_ms,
        "spec_files": spec_entries
    });

    Ok(result.to_string())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;

    use super::super::super::test_support::make_app_service_at;

    // T1: happy path — inline code with a passing test.
    #[tokio::test]
    async fn inline_passing_test_returns_passed_one() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('suite', function()\n",
            "    it('passes', function() expect(1).to.equal(1) end)\n",
            "end)\n",
        )
        .to_string();

        let result = svc
            .pkg_test(None, None, Some(lua_code), None, None, None, None, None)
            .await
            .expect("pkg_test should succeed");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["passed"], 1, "expected 1 passed: {result}");
        assert_eq!(json["failed"], 0, "expected 0 failed: {result}");
        assert_eq!(json["pending"], 0, "expected 0 pending: {result}");
    }

    // T2: edge case — inline code with a failing assertion returns Ok with
    // failed=1 (per-spec crash absorption).
    #[tokio::test]
    async fn inline_failing_test_absorbed_returns_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('suite', function()\n",
            "    it('fails', function() expect(1).to.equal(2) end)\n",
            "end)\n",
        )
        .to_string();

        let result = svc
            .pkg_test(None, None, Some(lua_code), None, None, None, None, None)
            .await
            .expect("pkg_test returns Ok even for failing tests");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["failed"], 1, "expected 1 failed: {result}");
        assert_eq!(json["passed"], 0, "expected 0 passed: {result}");
    }

    // T3: error path — zero inputs triggers typed Err (crux constraint 1).
    #[tokio::test]
    async fn zero_inputs_returns_exclusivity_error() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc
            .pkg_test(None, None, None, None, None, None, None, None)
            .await
            .expect_err("should return Err for zero inputs");

        assert_eq!(err, "pkg_test: provide exactly one of pkg, code_file, code");
    }

    // T3: error path — multiple inputs triggers typed Err (crux constraint 1).
    #[tokio::test]
    async fn multiple_inputs_returns_exclusivity_error() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc
            .pkg_test(
                Some("mypkg".into()),
                None,
                Some("return 1".into()),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect_err("should return Err for multiple inputs");

        assert_eq!(err, "pkg_test: provide exactly one of pkg, code_file, code");
    }

    // T3: error path — pkg not found returns typed Err (crux constraint 2
    // propagated tier).
    #[tokio::test]
    async fn pkg_not_found_returns_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc
            .pkg_test(
                Some("nonexistent_pkg_xyz".into()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect_err("should return Err for missing pkg");

        assert!(
            err.contains("nonexistent_pkg_xyz"),
            "error must mention pkg name: {err}"
        );
        assert!(err.contains("not found"), "error must say not found: {err}");
    }

    // T2: edge — code_file with non-existent path returns typed Err.
    #[tokio::test]
    async fn code_file_not_found_returns_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc
            .pkg_test(
                None,
                Some("/nonexistent/path/missing_spec.lua".into()),
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect_err("should return Err for missing code_file");

        assert!(
            err.contains("failed to read"),
            "error must describe I/O failure: {err}"
        );
    }

    // T1 (new): opt-out — auto_search_paths=false returns resolved_search_paths: []
    // Crux constraint 2: zero auto-resolved paths when opt-out.
    #[tokio::test]
    async fn auto_search_paths_false_returns_empty_resolved_mapping() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('s', function()\n",
            "    it('ok', function() expect(1).to.equal(1) end)\n",
            "end)\n",
        )
        .to_string();

        let result = svc
            .pkg_test(
                None,
                None,
                Some(lua_code),
                None,
                None,
                None,
                None,
                Some(false),
            )
            .await
            .expect("pkg_test should succeed with auto_search_paths=false");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Crux 2: resolved_search_paths must be empty array when opt-out.
        assert!(
            json["resolved_search_paths"].is_array(),
            "resolved_search_paths must be present: {result}"
        );
        assert_eq!(
            json["resolved_search_paths"].as_array().unwrap().len(),
            0,
            "resolved_search_paths must be empty when auto_search_paths=false: {result}"
        );
        // Crux 3: key must be present even when empty.
        assert!(
            json.get("resolved_search_paths").is_some(),
            "resolved_search_paths key must be present: {result}"
        );
    }

    // T1 (new): installed source — a pkg with init.lua in packages/ is auto-resolved
    // and appears in resolved_search_paths with source="installed".
    // Crux constraint 1: installed source is included.
    #[tokio::test]
    async fn installed_pkg_appears_in_resolved_search_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let app_root = tmp.path().to_path_buf();

        // Create a dummy installed package: {app_root}/packages/mypkg/init.lua
        let pkg_dir = app_root.join("packages").join("mypkg");
        fs::create_dir_all(&pkg_dir).unwrap();
        fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service_at(app_root.clone()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('s', function()\n",
            "    it('ok', function() expect(1).to.equal(1) end)\n",
            "end)\n",
        )
        .to_string();

        let result = svc
            .pkg_test(None, None, Some(lua_code), None, None, None, None, None)
            .await
            .expect("pkg_test should succeed");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let rows = json["resolved_search_paths"]
            .as_array()
            .expect("resolved_search_paths must be array");

        let installed_row = rows
            .iter()
            .find(|r| r["source"] == "installed" && r["name"] == "mypkg");
        assert!(
            installed_row.is_some(),
            "mypkg with source=installed must appear in resolved_search_paths: {result}"
        );

        // The search_dir should be the packages/ directory (parent of mypkg/).
        let expected_dir = app_root.join("packages").to_string_lossy().into_owned();
        let actual_dir = installed_row.unwrap()["search_dir"].as_str().unwrap_or("");
        assert_eq!(
            actual_dir, expected_dir,
            "search_dir must be the packages/ parent dir: {result}"
        );
    }

    // T1 (new): alc.toml source — a path entry in alc.toml is resolved and
    // appears in resolved_search_paths with source="alc.toml".
    // Crux constraint 1: alc.toml source is included.
    #[tokio::test]
    async fn alc_toml_path_entry_appears_in_resolved_search_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().to_path_buf();

        // Create an external pkg directory structure:
        // {project_root}/ext_pkgs/ext_pkg/init.lua
        let ext_pkgs = project_root.join("ext_pkgs");
        let ext_pkg_dir = ext_pkgs.join("ext_pkg");
        fs::create_dir_all(&ext_pkg_dir).unwrap();
        fs::write(ext_pkg_dir.join("init.lua"), "return {}").unwrap();

        // Write alc.toml with a path entry pointing to ext_pkgs/ext_pkg.
        let alc_toml_content = "[packages.ext_pkg]\npath = \"ext_pkgs/ext_pkg\"\n";
        fs::write(project_root.join("alc.toml"), alc_toml_content).unwrap();

        // AppService rooted at project_root (so app_dir = project_root, and
        // project root resolution = project_root when passed explicitly).
        let svc = make_app_service_at(project_root.clone()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('s', function()\n",
            "    it('ok', function() expect(1).to.equal(1) end)\n",
            "end)\n",
        )
        .to_string();

        let project_root_str = project_root.to_string_lossy().into_owned();
        let result = svc
            .pkg_test(
                None,
                None,
                Some(lua_code),
                None,
                None,
                None,
                Some(project_root_str),
                None,
            )
            .await
            .expect("pkg_test should succeed");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let rows = json["resolved_search_paths"]
            .as_array()
            .expect("resolved_search_paths must be array");

        let toml_row = rows
            .iter()
            .find(|r| r["source"] == "alc.toml" && r["name"] == "ext_pkg");
        assert!(
            toml_row.is_some(),
            "ext_pkg with source=alc.toml must appear in resolved_search_paths: {result}"
        );

        // search_dir should be the parent of ext_pkg_dir (i.e. ext_pkgs/).
        let expected_parent = ext_pkgs
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let actual_dir = toml_row.unwrap()["search_dir"].as_str().unwrap_or("");
        assert_eq!(
            actual_dir, expected_parent,
            "search_dir must be the canonicalized parent of the pkg dir: {result}"
        );
    }

    // T1 (new): alc.local.toml source — a path entry in alc.local.toml appears
    // in resolved_search_paths with source="alc.local.toml".
    // Crux constraint 1: alc.local.toml source is included.
    #[tokio::test]
    async fn alc_local_toml_path_entry_appears_in_resolved_search_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().to_path_buf();

        // Create variant pkg: {project_root}/variant_pkgs/variant_pkg/init.lua
        let variant_pkgs = project_root.join("variant_pkgs");
        let variant_pkg_dir = variant_pkgs.join("variant_pkg");
        fs::create_dir_all(&variant_pkg_dir).unwrap();
        fs::write(variant_pkg_dir.join("init.lua"), "return {}").unwrap();

        // Write alc.local.toml with a path entry.
        // Use [packages.name] table syntax (not [[packages.name]] array) for PackageDep::Path.
        let alc_local_content = "[packages.variant_pkg]\npath = \"variant_pkgs/variant_pkg\"\n";
        fs::write(project_root.join("alc.local.toml"), alc_local_content).unwrap();

        let svc = make_app_service_at(project_root.clone()).await;

        let lua_code = concat!(
            "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
            "describe('s', function()\n",
            "    it('ok', function() expect(1).to.equal(1) end)\n",
            "end)\n",
        )
        .to_string();

        let project_root_str = project_root.to_string_lossy().into_owned();
        let result = svc
            .pkg_test(
                None,
                None,
                Some(lua_code),
                None,
                None,
                None,
                Some(project_root_str),
                None,
            )
            .await
            .expect("pkg_test should succeed");

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let rows = json["resolved_search_paths"]
            .as_array()
            .expect("resolved_search_paths must be array");

        let local_row = rows
            .iter()
            .find(|r| r["source"] == "alc.local.toml" && r["name"] == "variant_pkg");
        assert!(
            local_row.is_some(),
            "variant_pkg with source=alc.local.toml must appear in resolved_search_paths: {result}"
        );

        let expected_parent = variant_pkgs
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let actual_dir = local_row.unwrap()["search_dir"].as_str().unwrap_or("");
        assert_eq!(
            actual_dir, expected_parent,
            "search_dir must be the canonicalized parent: {result}"
        );
    }

    // T2 (new): prepend order — collect_auto_search_paths helper returns
    // installed entries before caller-supplied search_paths are appended.
    // This tests the order invariant: [auto..., caller...] for the inline path.
    // Crux constraint 1: auto is prepended before caller entries.
    #[tokio::test]
    async fn auto_paths_prepended_before_caller_search_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let app_root = tmp.path().to_path_buf();

        // Create a dummy installed pkg so auto-resolve returns at least one dir.
        let pkg_dir = app_root.join("packages").join("autopkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service_at(app_root.clone()).await;

        // Call collect_auto_search_paths directly to verify the mapping.
        let (mapping, warnings) = svc.collect_auto_search_paths(None);
        assert!(warnings.is_empty(), "no warnings expected: {warnings:?}");

        // The installed source must be present.
        let found = mapping.iter().any(|r| r.name == "autopkg");
        assert!(
            found,
            "autopkg must appear in auto-resolved mapping: {mapping:?}"
        );

        // The search_dir must be the packages/ parent (not the pkg dir itself).
        let expected_parent = app_root.join("packages").to_string_lossy().into_owned();
        let row = mapping.iter().find(|r| r.name == "autopkg").unwrap();
        assert_eq!(
            row.search_dir, expected_parent,
            "search_dir must be packages/ parent: {row:?}"
        );
    }
}
