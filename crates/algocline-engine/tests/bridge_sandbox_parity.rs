//! `bridge::install_for_pkg_test` regression tests.
//!
//! Enforces the invariant from issue 7dc77cc7:
//!
//!     production primitive surface ⊆ test sandbox primitive surface
//!
//! Built specs (`alc_pkg_test`) must be able to call every `alc.*` helper
//! that production strategies (`alc_run`) can call.  This test fixes the
//! "spec uses alc.json_encode, production does, but sandbox doesn't" gap
//! at compile-time of the test suite.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::{Function, Lua, Table, Value};

/// Build a fully-configured production VM (the shape `alc_run` would build
/// when a session is dispatched), with `llm_tx = Some` so that
/// `alc.llm` / `alc.llm_batch` / `alc.fork` are all registered.
fn production_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("test tempdir");
    let root: PathBuf = tmp.path().to_path_buf();

    // A live channel is required to register `alc.llm` etc. The receiver
    // half is dropped immediately — the test never actually sends.
    let (llm_tx, _llm_rx) = tokio::sync::mpsc::channel(1);

    let config = BridgeConfig {
        llm_tx: Some(llm_tx),
        ns: "default".into(),
        custom_metrics: metrics.custom_metrics_handle(),
        stats: metrics.stats_handle(),
        budget: metrics.budget_handle(),
        progress: metrics.progress_handle(),
        lib_paths: vec![],
        variant_pkgs: vec![],
        state_store: Arc::new(JsonFileStore::new(root.join("state"))),
        card_store: Arc::new(FileCardStore::new(root.join("cards"))),
        card_run_enabled: true,
        scenarios_dir: root.join("scenarios"),
        nn_dir: root.join("nn"),
        log_sink: None,
    };

    let alc_table = lua.create_table().expect("create alc table");
    bridge::register(&lua, &alc_table, config).expect("production register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    lua.load(bridge::PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .expect("load prelude");
    (lua, tmp)
}

fn pkg_test_vm() -> Lua {
    let lua = Lua::new();
    bridge::install_for_pkg_test(&lua).expect("install_for_pkg_test");
    lua
}

fn alc_keys(lua: &Lua) -> BTreeSet<String> {
    let alc: Table = lua.globals().get("alc").expect("alc table missing");
    let mut keys = BTreeSet::new();
    for pair in alc.pairs::<Value, Value>() {
        let (k, _) = pair.expect("alc pair");
        if let Value::String(s) = k {
            keys.insert(s.to_str().expect("utf8").to_string());
        }
    }
    keys
}

#[test]
fn production_alc_keys_are_subset_of_pkg_test_alc_keys() {
    let (prod_lua, _tmp) = production_vm();
    let test_lua = pkg_test_vm();

    let prod_keys = alc_keys(&prod_lua);
    let test_keys = alc_keys(&test_lua);

    let missing: Vec<_> = prod_keys.difference(&test_keys).cloned().collect();
    assert!(
        missing.is_empty(),
        "pkg_test sandbox is missing alc.* keys present in production: {missing:?}\n\
         (invariant violation: production ⊄ pkg_test — see issue 7dc77cc7)"
    );
}

#[test]
fn pkg_test_sandbox_has_json_encode_and_decode() {
    let lua = pkg_test_vm();
    let out: String = lua
        .load(r#"return alc.json_encode({a = 1, b = "two"})"#)
        .eval()
        .expect("alc.json_encode callable in sandbox");
    // serde_json sorts neither — just check both keys appear with right shape.
    assert!(out.contains("\"a\":1"), "encoded JSON: {out}");
    assert!(out.contains("\"b\":\"two\""), "encoded JSON: {out}");

    let n: i64 = lua
        .load(r#"return alc.json_decode('{"n":42}').n"#)
        .eval()
        .expect("alc.json_decode callable");
    assert_eq!(n, 42);
}

#[test]
fn pkg_test_sandbox_stubs_external_io_with_clear_error() {
    let lua = pkg_test_vm();
    let err = lua
        .load(r#"return alc.llm("hello")"#)
        .exec()
        .expect_err("alc.llm stub must error without with_alc override");
    let msg = format!("{err}");
    assert!(
        msg.contains("mock required: alc.llm"),
        "stub error must guide spec author: {msg}"
    );
    assert!(
        msg.contains("with_alc"),
        "stub error must mention with_alc: {msg}"
    );
}

#[test]
fn with_alc_overrides_llm_then_restores() {
    let lua = pkg_test_vm();
    // Before: stub.  After body: stub again.
    let out: String = lua
        .load(
            r#"
            local before_ok = pcall(alc.llm, "x")
            assert(not before_ok, "expected stub to error before with_alc")

            local seen = with_alc({ llm = function(p) return "echo:" .. p end }, function()
                return alc.llm("hello")
            end)

            local after_ok = pcall(alc.llm, "x")
            assert(not after_ok, "expected stub to restore after with_alc")

            return seen
            "#,
        )
        .eval()
        .expect("with_alc must run body and restore");
    assert_eq!(out, "echo:hello");
}

#[test]
fn with_alc_restores_even_on_error_in_body() {
    let lua = pkg_test_vm();
    lua.load(
        r#"
        local ok, _err = pcall(function()
            with_alc({ json_encode = function(_) error("boom") end }, function()
                return alc.json_encode({})
            end)
        end)
        assert(not ok, "expected with_alc body to propagate error")

        -- After the error, alc.json_encode must be the real Rust impl again.
        local restored = alc.json_encode({a = 1})
        assert(restored:find('"a":1'), "real json_encode not restored: " .. restored)
        "#,
    )
    .exec()
    .expect("restore-on-error path must work");
}

#[test]
fn alc_spy_records_calls_and_returns_default() {
    let lua = pkg_test_vm();
    let (count, first_arg, ret): (i64, String, String) = lua
        .load(
            r#"
            local spy = alc.spy("llm", function(p) return "stub:" .. p end)
            local r1 = alc.llm("hello")
            local r2 = alc.llm("world")
            return spy.call_count, spy.calls[1].args[1], r2
            "#,
        )
        .eval()
        .expect("alc.spy must wrap and observe");
    assert_eq!(count, 2);
    assert_eq!(first_arg, "hello");
    assert_eq!(ret, "stub:world");
}

#[test]
fn alc_mock_install_and_restore_pop_one_frame() {
    let lua = pkg_test_vm();
    lua.load(
        r#"
        -- Initial: alc.llm is a stub
        assert(not pcall(alc.llm, "x"))

        alc_mock.install({ llm = function(p) return "A:" .. p end })
        assert(alc.llm("hi") == "A:hi")

        alc_mock.install({ llm = function(p) return "B:" .. p end })
        assert(alc.llm("hi") == "B:hi")

        alc_mock.restore()
        assert(alc.llm("hi") == "A:hi", "expected to fall back to A after one restore")

        alc_mock.restore()
        assert(not pcall(alc.llm, "x"), "expected stub after second restore")
        "#,
    )
    .exec()
    .expect("alc_mock.install / restore must form a stack");
}

#[test]
fn nested_with_alc_unwinds_in_lifo_order() {
    let lua = pkg_test_vm();
    lua.load(
        r#"
        local order = {}
        with_alc({ llm = function() return "outer" end }, function()
            table.insert(order, alc.llm())
            with_alc({ llm = function() return "inner" end }, function()
                table.insert(order, alc.llm())
            end)
            table.insert(order, alc.llm())
        end)
        assert(order[1] == "outer")
        assert(order[2] == "inner")
        assert(order[3] == "outer", "expected outer to be restored after inner with_alc")
        "#,
    )
    .exec()
    .expect("nested with_alc must unwind LIFO");
}

#[test]
fn pkg_test_sandbox_exposes_fingerprint_and_fuzzy() {
    let lua = pkg_test_vm();
    let fp_match: bool = lua
        .load(r#"return alc.fingerprint("Hello") == alc.fingerprint("hello")"#)
        .eval()
        .expect("alc.fingerprint usable");
    assert!(fp_match);

    let _: Function = lua
        .globals()
        .get::<Table>("alc")
        .expect("alc table")
        .get::<Function>("match_enum")
        .expect("alc.match_enum must be registered in sandbox");
}
