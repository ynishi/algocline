use std::path::Path;
use std::sync::Arc;

use proptest::prelude::*;

use crate::service::config::{AppConfig, LogDirSource};
use crate::service::eval_store::{build_meta, extract_strategy_from_id, list_eval_history};
use crate::service::path::{copy_dir, ContainedPath};
use crate::service::resolve::{
    display_name, install_scenarios_from_dir, is_package_installed, make_require_code,
    resolve_code, resolve_scenario_code, resolve_scenario_source, scenarios_dir,
};
use crate::service::AppService;

proptest! {
    /// resolve_code never panics.
    #[test]
    fn resolve_code_never_panics(
        code in proptest::option::of("[a-z]{0,50}"),
        file in proptest::option::of("[a-z]{0,50}"),
    ) {
        let _ = resolve_code(code, file);
    }

    /// ContainedPath always rejects ".." components.
    #[test]
    fn contained_path_rejects_traversal(
        prefix in "[a-z]{0,5}",
        suffix in "[a-z]{0,5}",
    ) {
        let dir = tempfile::tempdir().unwrap();
        let name = format!("{prefix}/../{suffix}");
        let result = ContainedPath::child(dir.path(), &name);
        prop_assert!(result.is_err());
    }

    /// ContainedPath accepts simple alphanumeric names.
    #[test]
    fn contained_path_accepts_simple_names(name in "[a-z][a-z0-9_-]{0,20}\\.json") {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), &name);
        prop_assert!(result.is_ok());
    }

    /// make_require_code always contains the strategy name in a require call.
    #[test]
    fn make_require_code_contains_name(name in "[a-z_]{1,20}") {
        let code = make_require_code(&name);
        let expected = format!("require(\"{}\")", name);
        prop_assert!(code.contains(&expected));
        prop_assert!(code.contains("pkg.run(ctx)"));
    }

    /// copy_dir preserves file contents for arbitrary data.
    #[test]
    fn copy_dir_preserves_content(content in "[a-zA-Z0-9 ]{1,200}") {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        std::fs::write(src.path().join("test.txt"), &content).unwrap();
        let dst_path = dst.path().join("out");
        copy_dir(src.path(), &dst_path).unwrap();

        let read = std::fs::read_to_string(dst_path.join("test.txt")).unwrap();
        prop_assert_eq!(&read, &content);
    }
}

// ─── eval tests ───

#[test]
fn eval_rejects_no_scenario() {
    let result = resolve_scenario_code(None, None, None);
    assert!(result.is_err());
}

#[test]
fn resolve_scenario_code_inline() {
    let result = resolve_scenario_code(Some("return 1".into()), None, None);
    assert_eq!(result.unwrap(), "return 1");
}

#[test]
fn resolve_scenario_code_from_file() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut tmp, b"return 42").unwrap();
    let result = resolve_scenario_code(None, Some(tmp.path().to_string_lossy().into()), None);
    assert_eq!(result.unwrap(), "return 42");
}

#[test]
fn resolve_scenario_code_rejects_multiple() {
    let result = resolve_scenario_code(Some("code".into()), Some("file".into()), None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("only one"));

    let result2 = resolve_scenario_code(Some("code".into()), None, Some("name".into()));
    assert!(result2.is_err());
}

#[test]
fn resolve_scenario_code_by_name_not_found() {
    // scenario_name resolves from ~/.algocline/scenarios/ which won't have this
    let result = resolve_scenario_code(None, None, Some("nonexistent_test_xyz".into()));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ─── scenario management tests ───

#[test]
fn scenarios_dir_ends_with_expected_path() {
    let dir = scenarios_dir().unwrap();
    assert!(
        dir.ends_with(".algocline/scenarios"),
        "dir: {}",
        dir.display()
    );
}

#[test]
fn install_scenarios_from_dir_copies_lua_files() {
    let source = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();

    // Create test .lua files
    std::fs::write(source.path().join("math_basic.lua"), "return {}").unwrap();
    std::fs::write(source.path().join("safety.lua"), "return {}").unwrap();
    // Non-lua file should be skipped
    std::fs::write(source.path().join("README.md"), "# docs").unwrap();

    let result = install_scenarios_from_dir(source.path(), dest.path()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let installed = parsed["installed"].as_array().unwrap();
    assert_eq!(installed.len(), 2);
    assert!(dest.path().join("math_basic.lua").exists());
    assert!(dest.path().join("safety.lua").exists());
    assert!(!dest.path().join("README.md").exists());
    assert_eq!(parsed["failures"].as_array().unwrap().len(), 0);
}

#[test]
fn install_scenarios_from_dir_skips_existing() {
    let source = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();

    std::fs::write(source.path().join("existing.lua"), "return {new=true}").unwrap();
    std::fs::write(dest.path().join("existing.lua"), "return {old=true}").unwrap();

    let result = install_scenarios_from_dir(source.path(), dest.path()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["skipped"].as_array().unwrap().len(), 1);
    assert_eq!(parsed["installed"].as_array().unwrap().len(), 0);
    assert_eq!(parsed["failures"].as_array().unwrap().len(), 0);

    // Original file should be preserved
    let content = std::fs::read_to_string(dest.path().join("existing.lua")).unwrap();
    assert!(content.contains("old=true"));
}

#[test]
fn install_scenarios_from_dir_empty_source_errors() {
    let source = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();

    let result = install_scenarios_from_dir(source.path(), dest.path());
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No .lua"));
}

#[test]
fn install_scenarios_from_dir_collects_copy_failures() {
    let source = tempfile::tempdir().unwrap();
    // dest is a non-existent path inside a read-only dir to force copy failure
    let dest = tempfile::tempdir().unwrap();
    let bad_dest = dest.path().join("nonexistent_subdir");
    // Don't create bad_dest — copy will fail

    std::fs::write(source.path().join("ok.lua"), "return 1").unwrap();

    let result = install_scenarios_from_dir(source.path(), &bad_dest);
    // ContainedPath::child won't fail, but fs::copy to nonexistent dir will
    let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    let failures = parsed["failures"].as_array().unwrap();
    assert_eq!(failures.len(), 1, "expected 1 copy failure");
    assert_eq!(parsed["installed"].as_array().unwrap().len(), 0);
}

#[test]
fn display_name_prefers_stem() {
    let path = Path::new("/tmp/math_basic.lua");
    assert_eq!(display_name(path, "math_basic.lua"), "math_basic");
}

#[test]
fn display_name_falls_back_to_file_name() {
    // file_stem returns None only for paths like "" or "/"
    let path = Path::new("");
    assert_eq!(display_name(path, "fallback"), "fallback");
}

#[test]
fn resolve_scenario_source_prefers_subdir() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("scenarios")).unwrap();
    std::fs::write(root.path().join("scenarios").join("a.lua"), "").unwrap();
    std::fs::write(root.path().join("root.lua"), "").unwrap();

    let source = resolve_scenario_source(root.path());
    assert_eq!(source, root.path().join("scenarios"));
}

#[test]
fn resolve_scenario_source_falls_back_to_root() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.lua"), "").unwrap();

    let source = resolve_scenario_source(root.path());
    assert_eq!(source, root.path());
}

#[test]
fn eval_auto_installs_evalframe_on_missing() {
    // Serialize with FakeHome tests to prevent HOME env var races.
    let _home_lock = super::super::test_support::lock_home();

    // Skip if evalframe is already installed globally
    if is_package_installed("evalframe") {
        return;
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let fake_pkg_dir = tmp.path().join("empty_packages");
    std::fs::create_dir_all(&fake_pkg_dir).unwrap();

    let executor = Arc::new(rt.block_on(async {
        algocline_engine::Executor::new(vec![fake_pkg_dir])
            .await
            .unwrap()
    }));
    let config = AppConfig {
        log_dir: Some(tmp.path().join("logs")),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: false,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // AppService::new() calls spawn_gc_task() which requires a tokio runtime context.
    // Scope the enter guard so it is dropped before rt.block_on() below.
    let svc = {
        let _guard = rt.enter();
        AppService::new(executor, config, vec![])
    };

    let scenario = r#"return { cases = {} }"#;
    let result = rt.block_on(svc.eval(Some(scenario.into()), None, None, "cove", None, false));
    assert!(result.is_err());
    // Auto-install is attempted first; error is about bundled install failure
    // (git clone) or evalframe still missing after install
    let err = result.unwrap_err();
    assert!(
        err.contains("bundled") || err.contains("evalframe"),
        "unexpected error: {err}"
    );
}

// ─── comparison helper tests ───

#[test]
fn extract_strategy_from_id_splits_correctly() {
    assert_eq!(extract_strategy_from_id("cove_1710672000"), Some("cove"));
    assert_eq!(
        extract_strategy_from_id("my_strat_1710672000"),
        Some("my_strat")
    );
    assert_eq!(extract_strategy_from_id("nostamp"), None);
}

#[test]
fn save_compare_result_persists_file() {
    let tmp = tempfile::tempdir().unwrap();
    let evals = tmp.path().join(".algocline").join("evals");
    std::fs::create_dir_all(&evals).unwrap();

    // save_compare_result uses evals_dir() which reads HOME.
    // Test ContainedPath + write logic directly instead.
    let filename = "compare_a_1_vs_b_2.json";
    let path = ContainedPath::child(&evals, filename).unwrap();
    let data = r#"{"test": true}"#;
    std::fs::write(&*path, data).unwrap();

    let read = std::fs::read_to_string(&*path).unwrap();
    assert_eq!(read, data);
}

// ─── build_meta tests ───

#[test]
fn build_meta_extracts_aggregated_fields() {
    let result_json = serde_json::json!({
        "result": {
            "aggregated": {
                "pass_rate": 0.75,
                "scores": { "mean": 8.5 },
                "total": 4,
                "passed": 3
            },
            "summary": "3/4 passed"
        },
        "stats": {
            "auto": {
                "llm_calls": 12,
                "elapsed_ms": 5000
            }
        }
    });

    let meta = build_meta("cot_1700000000", "cot", 1700000000, &result_json);

    assert_eq!(meta["eval_id"], "cot_1700000000");
    assert_eq!(meta["strategy"], "cot");
    assert_eq!(meta["timestamp"], 1700000000_u64);
    assert_eq!(meta["pass_rate"], 0.75);
    assert_eq!(meta["mean_score"], 8.5);
    assert_eq!(meta["total_cases"], 4);
    assert_eq!(meta["passed"], 3);
    assert_eq!(meta["llm_calls"], 12);
    assert_eq!(meta["elapsed_ms"], 5000);
    assert_eq!(meta["summary"], "3/4 passed");
}

#[test]
fn build_meta_handles_missing_fields() {
    // Minimal result with no aggregated/stats
    let result_json = serde_json::json!({});
    let meta = build_meta("sc_1700000000", "sc", 1700000000, &result_json);

    assert_eq!(meta["eval_id"], "sc_1700000000");
    assert_eq!(meta["strategy"], "sc");
    assert!(meta["pass_rate"].is_null());
    assert!(meta["mean_score"].is_null());
    assert!(meta["llm_calls"].is_null());
}

// ─── list_eval_history tests ───

/// Helper: write a meta file + corresponding result file into a tmpdir.
fn write_eval_files(dir: &Path, eval_id: &str, strategy: &str, timestamp: u64) {
    let meta = serde_json::json!({
        "eval_id": eval_id,
        "strategy": strategy,
        "timestamp": timestamp,
        "pass_rate": 1.0,
    });
    let meta_path = dir.join(format!("{eval_id}.meta.json"));
    std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();

    // list_eval_history skips meta files and looks for *.json (non-meta) files
    // that have a corresponding .meta.json. So we need the result file too.
    let result_path = dir.join(format!("{eval_id}.json"));
    std::fs::write(&result_path, r#"{"result":{}}"#).unwrap();
}

#[test]
fn eval_history_empty_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let result = list_eval_history(tmp.path(), None, 10).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["evals"].as_array().unwrap().len(), 0);
}

#[test]
fn eval_history_nonexistent_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("nonexistent");
    let result = list_eval_history(&missing, None, 10).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["evals"].as_array().unwrap().len(), 0);
}

#[test]
fn eval_history_sorts_newest_first() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_eval_files(dir, "cot_1000", "cot", 1000);
    write_eval_files(dir, "cot_3000", "cot", 3000);
    write_eval_files(dir, "cot_2000", "cot", 2000);

    let result = list_eval_history(dir, None, 10).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let evals = parsed["evals"].as_array().unwrap();

    assert_eq!(evals.len(), 3);
    assert_eq!(evals[0]["timestamp"], 3000);
    assert_eq!(evals[1]["timestamp"], 2000);
    assert_eq!(evals[2]["timestamp"], 1000);
}

#[test]
fn eval_history_filters_by_strategy() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_eval_files(dir, "cot_1000", "cot", 1000);
    write_eval_files(dir, "sc_2000", "sc", 2000);
    write_eval_files(dir, "cot_3000", "cot", 3000);

    let result = list_eval_history(dir, Some("cot"), 10).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let evals = parsed["evals"].as_array().unwrap();

    assert_eq!(evals.len(), 2);
    assert!(evals.iter().all(|e| e["strategy"] == "cot"));
}

#[test]
fn eval_history_respects_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_eval_files(dir, "cot_1000", "cot", 1000);
    write_eval_files(dir, "cot_2000", "cot", 2000);
    write_eval_files(dir, "cot_3000", "cot", 3000);

    let result = list_eval_history(dir, None, 2).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let evals = parsed["evals"].as_array().unwrap();

    assert_eq!(evals.len(), 2);
    // Should be newest first, so 3000 and 2000
    assert_eq!(evals[0]["timestamp"], 3000);
    assert_eq!(evals[1]["timestamp"], 2000);
}

#[test]
fn eval_history_skips_entries_without_meta() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Only result file, no meta file
    std::fs::write(dir.join("orphan_1000.json"), r#"{"result":{}}"#).unwrap();

    // This one has both
    write_eval_files(dir, "cot_2000", "cot", 2000);

    let result = list_eval_history(dir, None, 10).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let evals = parsed["evals"].as_array().unwrap();

    assert_eq!(evals.len(), 1);
    assert_eq!(evals[0]["eval_id"], "cot_2000");
}
