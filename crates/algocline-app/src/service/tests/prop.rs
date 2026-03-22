use std::path::Path;
use std::sync::Arc;

use proptest::prelude::*;

use crate::service::config::{AppConfig, LogDirSource};
use crate::service::eval_store::extract_strategy_from_id;
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
    };
    let svc = AppService::new(executor, config);

    let scenario = r#"return { cases = {} }"#;
    let result = rt.block_on(svc.eval(Some(scenario.into()), None, None, "cove", None));
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
