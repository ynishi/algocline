//! Phase A re-spike for algocline issue 1779637712-85681.
//!
//! Goal: measure mlua-check (v0.2 library API) false / true positive ratio
//! on shipped Lua under the **library-direct** path. The earlier MCP-mediated
//! probe was bounded by `'Custom globals are not available'`; here we drive
//! `mlua_check::vm::register(&Lua)` directly, which auto-bridges live VM
//! globals into the SymbolTable.
//!
//! Setup mirrors the production bridge minimally:
//!   1. seed `alc` table with stub fields (state / json_* / log / llm / eval /
//!      time / _dirs) so prelude.lua loads,
//!   2. dofile prelude.lua so `alc.map` / `alc.reduce` / etc. become real
//!      table entries,
//!   3. call `mlua_check::vm::register(&lua)` to harvest the SymbolTable,
//!   4. lint each shipped Lua file and tally counts.
//!
//! Marked `#[ignore]` because the goal is evidence collection, not gating
//! `cargo test`. Run with:
//!   `cargo test -p algocline-engine --test lua_lint_spike -- --ignored --nocapture`

use std::fs;
use std::path::{Path, PathBuf};

use mlua::Lua;
use mlua_check::vm::register;

fn engine_crate() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    engine_crate()
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .expect("workspace root resolvable from engine crate")
}

/// Prepare a Lua VM with the production `alc` global so the SymbolTable
/// reflects what shipped Lua actually sees at runtime.
fn make_vm() -> Lua {
    let lua = Lua::new();

    let setup = r#"
        alc = {}
        -- Service-layer stubs (normally injected by the Rust bridge)
        alc.state = {
            get = function(_, default) return default end,
            set = function() end,
        }
        alc.json_encode = function() end
        alc.json_decode = function() end
        alc.log = function() end
        alc.llm = function() end
        alc.eval = function() end
        alc.time = function() return 0 end
        alc._dirs = { scenarios = "/dev/null" }
    "#;
    lua.load(setup)
        .set_name("@alc_stub_setup")
        .exec()
        .expect("alc stub setup must succeed");

    let prelude = engine_crate().join("src").join("prelude.lua");
    let code =
        fs::read_to_string(&prelude).unwrap_or_else(|e| panic!("read {}: {e}", prelude.display()));
    lua.load(&code)
        .set_name(format!("@{}", prelude.display()))
        .exec()
        .expect("prelude.lua must load on stubbed alc");

    lua
}

fn shipped_files() -> Vec<PathBuf> {
    let root = workspace_root();
    let shapes = root
        .join("crates")
        .join("algocline-app")
        .join("src")
        .join("service")
        .join("gendoc")
        .join("alc_shapes");
    let docs = root
        .join("crates")
        .join("algocline-app")
        .join("src")
        .join("service")
        .join("lua")
        .join("gendoc");
    let prelude = engine_crate().join("src").join("prelude.lua");

    let mut out: Vec<PathBuf> = Vec::new();
    for sub in [&shapes, &docs] {
        if !sub.exists() {
            continue;
        }
        collect_lua(sub, &mut out);
    }
    out.push(prelude);
    out.sort();
    out
}

fn collect_lua(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_lua(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("lua") {
            out.push(path);
        }
    }
}

#[test]
#[ignore]
fn spike_lint_shipped_lua() {
    let lua = make_vm();
    let engine = register(&lua).expect("register(&lua) must succeed");

    let files = shipped_files();
    assert!(!files.is_empty(), "shipped_files() yielded zero entries");

    println!("\n=== mlua-check library-direct spike ===");
    println!("VM: alc stubs + prelude.lua loaded");
    println!("Files: {}\n", files.len());

    let mut total_warnings = 0usize;
    let mut total_errors = 0usize;

    for file in &files {
        let code = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                println!("SKIP  {} — read error: {e}", file.display());
                continue;
            }
        };
        let chunk = format!("@{}", file.display());
        let result = engine.lint(&code, &chunk);
        total_warnings += result.warning_count;
        total_errors += result.error_count;

        let rel = file
            .strip_prefix(workspace_root())
            .unwrap_or(file)
            .display();
        println!(
            "{:>3} warn / {:>2} err  {}",
            result.warning_count, result.error_count, rel
        );

        for diag in result.diagnostics.iter().take(6) {
            println!(
                "    [{:?}] L{}:{}  {}  ({})",
                diag.severity, diag.line, diag.column, diag.message, diag.rule,
            );
        }
        if result.diagnostics.len() > 6 {
            println!(
                "    ... {} more diagnostics suppressed",
                result.diagnostics.len() - 6
            );
        }
    }

    println!(
        "\nTotal: {total_warnings} warnings / {total_errors} errors across {} files",
        files.len()
    );
}
