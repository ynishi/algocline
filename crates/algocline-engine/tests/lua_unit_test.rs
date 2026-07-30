//! Cargo test harness for `tests/lua/*.lua` mlua-lspec specs.
//!
//! Activates the shipped Lua spec suite under `cargo test --workspace`.
//! Each `#[test]` runs one spec file through `mlua_lspec::framework::run_tests`
//! and bridges per-test results into Rust `panic!` so failures show up in the
//! standard cargo test output.
//!
//! Reference implementation: `crates/algocline-app/src/service/pkg/test_run.rs`
//! (the `alc_pkg_test` MCP tool uses the same framework call shape).

use std::fs;
use std::path::PathBuf;

use mlua_lspec::framework;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("lua")
}

fn prelude_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("prelude.lua")
}

/// Absolute path to the workspace-root `examples/gameai/` directory.
///
/// Resolved from `CARGO_MANIFEST_DIR` (which points at
/// `crates/algocline-engine/`) rather than the process CWD, which
/// differs between `cargo test` and IDE runners.
fn gameai_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine crate parent (crates/)")
        .parent()
        .expect("workspace root")
        .join("examples")
        .join("gameai")
}

/// Execute one spec file and translate per-test results into a single
/// Rust assertion. Per-spec setup errors (Err from `run_tests`) and per-test
/// failures both panic with the spec name and the offending test list.
fn run_spec(name: &str) {
    let path = spec_dir().join(name);
    let code = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let chunk_name = format!("@{}", path.display());

    let summary = framework::run_tests(&code, &chunk_name, &[])
        .unwrap_or_else(|e| panic!("{name}: spec execution error: {e}"));

    if summary.failed > 0 {
        let mut msg = format!(
            "{name}: {failed} failed / {passed} passed / {total} total",
            failed = summary.failed,
            passed = summary.passed,
            total = summary.total,
        );
        for t in summary.tests.iter().filter(|t| !t.passed) {
            msg.push_str(&format!(
                "\n  FAIL  [{suite}] {name} — {err}",
                suite = t.suite,
                name = t.name,
                err = t.error.as_deref().unwrap_or("<no error message>"),
            ));
        }
        panic!("{msg}");
    }
}

#[test]
fn prelude_spec() {
    run_spec("prelude_test.lua");
}

#[test]
fn bridge_spec() {
    // bridge_test.lua does `dofile(os.getenv("PRELUDE_PATH") or <relative>)`.
    // The relative default is cwd-dependent and breaks under cargo test, so
    // always inject an absolute path resolved from CARGO_MANIFEST_DIR.
    std::env::set_var("PRELUDE_PATH", prelude_path());
    run_spec("bridge_test.lua");
}

#[test]
fn card_duel_rules_spec() {
    // card_duel_rules_test.lua reads `ALC_TEST_GAMEAI_DIR` to put
    // `examples/gameai/?/init.lua` on `package.path`. The spec stubs
    // `alc.math` itself, so no bridge is needed and this runs on the
    // default feature set (the `nn` half is fenced by
    // `tests/gameai_smoke_test.rs`).
    std::env::set_var("ALC_TEST_GAMEAI_DIR", gameai_dir());
    run_spec("card_duel_rules_test.lua");
}

#[test]
fn eval_integration_spec() {
    // eval_integration_test.lua also dofiles `PRELUDE_PATH`; cargo test may
    // run this before bridge_spec so the env var must be set here too.
    std::env::set_var("PRELUDE_PATH", prelude_path());

    // The spec requires evalframe installed at
    // `$ALC_TEST_APP_DIR/packages/evalframe/`. Probe `~/.algocline` (the dev
    // setup populated by `alc init`) and skip with a visible note when absent
    // so a fresh checkout without `alc init` still keeps cargo test green.
    let Some(home) = dirs::home_dir() else {
        eprintln!("eval_integration_spec: skipped — HOME unresolved");
        return;
    };
    let app_dir = home.join(".algocline");
    let evalframe = app_dir.join("packages").join("evalframe");
    if !evalframe.exists() {
        eprintln!(
            "eval_integration_spec: skipped — evalframe not installed at {}.\n  Run `alc init` to install bundled packages.",
            evalframe.display()
        );
        return;
    }
    std::env::set_var("ALC_TEST_APP_DIR", &app_dir);
    run_spec("eval_integration_test.lua");
}
