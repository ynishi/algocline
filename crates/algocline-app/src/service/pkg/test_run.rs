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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlua::Lua;
use mlua_lspec::framework;
use serde_json::{json, Value};

use super::super::AppService;

impl AppService {
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
    ///   the Lua VM.
    /// * `project_root` — optional project root for variant-scope resolution
    ///   (`alc.local.toml`). Falls back to ancestor walk from cwd.
    ///
    /// # Returns
    ///
    /// On success: JSON string with shape
    /// `{passed, failed, pending, total, duration_ms, spec_files: [{path,
    /// passed, failed, total, duration_ms, tests: [{suite, name, passed,
    /// pending, error}]}]}`.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` for setup failures (VM init, pkg not found, zero
    /// spec files, I/O errors, `spawn_blocking` panic). Per-spec Lua crashes
    /// are absorbed, not propagated.
    // 8 parameters are justified by the MCP wire shape: 3 mutually exclusive
    // input sources (pkg / code_file / code) plus filtering/path options.
    #[allow(clippy::too_many_arguments)]
    pub async fn pkg_test(
        &self,
        pkg: Option<String>,
        code_file: Option<String>,
        code: Option<String>,
        spec_dir: Option<String>,
        filter: Option<String>,
        search_paths: Option<Vec<String>>,
        _project_root: Option<String>,
    ) -> Result<String, String> {
        // ── Crux constraint 1: exactly-one input exclusivity ──────────────────
        let input_count = pkg.is_some() as u8 + code_file.is_some() as u8 + code.is_some() as u8;
        if input_count != 1 {
            return Err("pkg_test: provide exactly one of pkg, code_file, code".to_string());
        }

        let extra_search_paths: Vec<String> = search_paths.unwrap_or_default();

        if let Some(inline_code) = code {
            // `code` path: single VM, inline source.
            run_inline(inline_code, extra_search_paths).await
        } else if let Some(file_path) = code_file {
            // `code_file` path: read file then run.
            let abs_path = PathBuf::from(&file_path);
            let src = std::fs::read_to_string(&abs_path)
                .map_err(|e| format!("pkg_test: failed to read {file_path}: {e}"))?;
            let parent = abs_path
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let chunk_name = format!("@{file_path}");
            let mut paths = vec![parent];
            paths.extend(extra_search_paths);
            run_single_spec(src, chunk_name, paths).await
        } else {
            // `pkg` path: spec_dir scan.
            // input_count == 1 and neither `code` nor `code_file` is Some,
            // so `pkg` must be Some here by the exclusivity check above.
            let Some(pkg_name) = pkg else {
                unreachable!("pkg must be Some: input_count==1 and code/code_file are None")
            };
            let init_path = self
                .pkg_resolve_init_path(&pkg_name)
                .map_err(|e| format!("pkg_test: {e}"))?
                .ok_or_else(|| {
                    format!(
                        "pkg_test: package '{pkg_name}' not found in alc.toml or ~/.algocline/packages/"
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

            let pkg_root_str = pkg_root.to_string_lossy().into_owned();
            let mut search = vec![pkg_root_str];
            search.extend(extra_search_paths);

            run_pkg_specs(spec_files, search).await
        }
    }
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
/// Returns `Err` when the task panics.
async fn run_single_spec(
    code: String,
    chunk_name: String,
    search_paths: Vec<String>,
) -> Result<String, String> {
    let total_start = Instant::now();

    let (spec_file_entry, agg_passed, agg_failed) = tokio::task::spawn_blocking(move || {
        // mlua::Lua is !Send — construct inside the blocking task.
        let lua = Lua::new();

        let search_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();
        let spec_start = Instant::now();

        let (tests_json, passed, failed) =
            match framework::run_tests(&code, &chunk_name, &search_refs) {
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

        // Keep lua alive until here to avoid premature Drop.
        drop(lua);

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

            // mlua::Lua is !Send — construct fresh per spec file.
            let lua = Lua::new();
            let spec_start = Instant::now();

            let (tests_json, passed, failed) =
                match framework::run_tests(&code, &chunk_name, &search_refs) {
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

            // Keep lua alive until end of this iteration, then drop.
            drop(lua);
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
            .pkg_test(None, None, Some(lua_code), None, None, None, None)
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
            .pkg_test(None, None, Some(lua_code), None, None, None, None)
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
            .pkg_test(None, None, None, None, None, None, None)
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
            )
            .await
            .expect_err("should return Err for missing code_file");

        assert!(
            err.contains("failed to read"),
            "error must describe I/O failure: {err}"
        );
    }
}
