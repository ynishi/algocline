#![cfg(feature = "nn")]
//! Engine-level smoke test for the `alc.nn` bridge (feature `nn`).
//!
//! The `algocline-nn` crate tests exercise the candle wrapper in isolation. This
//! test closes the remaining gap: it confirms the full engine registration path
//! (`bridge::register` -> `register_nn`) actually places `alc.nn` on the `alc`
//! table of a real engine VM and that a tensor op round-trips through Lua.
//!
//! It installs via `install_for_pkg_test`, the same path the pkg-test sandbox
//! uses, so with `nn` enabled the `alc.nn` surface is also present in the sandbox
//! (keeping the `production ⊆ sandbox` parity invariant intact).
//!
//! Only compiled with `--features nn`; the default build has no `alc.nn` and no
//! candle link.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

fn nn_vm() -> Lua {
    let lua = Lua::new();
    bridge::install_for_pkg_test(&lua).expect("install_for_pkg_test");
    lua
}

/// Build a production-shaped VM (llm_tx = Some) so `alc.llm` is registered.
/// The mpsc receiver is dropped immediately; the whole point of the in-process
/// routing test is that `alc.llm` never actually sends when role="nn".
fn production_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("test tempdir");
    let root: PathBuf = tmp.path().to_path_buf();

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
        card_run_enabled: false,
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

#[test]
fn alc_nn_tensor_add_roundtrips_through_engine_bridge() {
    let lua = nn_vm();
    let out: Vec<f32> = lua
        .load(
            r#"
            local a = alc.nn.tensor({ 1, 2, 3 })
            local b = alc.nn.tensor({ 10, 20, 30 })
            return a:add(b):to_vec()
        "#,
        )
        .eval()
        .expect("alc.nn tensor add roundtrip");
    assert_eq!(out, vec![11.0, 22.0, 33.0]);
}

#[test]
fn alc_nn_tensor_dims_reachable_through_engine_bridge() {
    let lua = nn_vm();
    let dims: Vec<usize> = lua
        .load("return alc.nn.tensor({ 1, 2, 3, 4 }):dims()")
        .eval()
        .expect("alc.nn.tensor(...):dims()");
    assert_eq!(dims, vec![4]);
}

/// `alc.llm(prompt, {role="nn", model=name})` dispatches to the alc.nn model
/// registry in-process — no yield, no host round-trip. Register a trivial Lua
/// closure as "echo" and call `alc.llm` through the production bridge. The
/// mpsc receiver is deliberately dropped, so if the normal Host path were
/// taken this would fail with "send failed"; a success means the in-process
/// short-circuit worked. Requires an async runtime because `alc.llm` is an
/// async function.
#[test]
fn alc_llm_role_nn_routes_to_registered_model_in_process() {
    let (lua, _tmp) = production_vm();

    // Register a tiny "model": returns the prompt with an "echo:" prefix.
    lua.load(
        r#"
        alc.nn.register("echo", function(prompt)
            return "echo:" .. prompt
        end)
    "#,
    )
    .exec()
    .expect("register echo model");

    // Call alc.llm with role="nn" through a coroutine so the async function
    // can resolve. The routing branch returns synchronously (no yield), so the
    // coroutine finishes in one resume.
    let out: String = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(async {
            let f: mlua::Function = lua
                .load(
                    r#"
                    return function()
                        return alc.llm("hello", { role = "nn", model = "echo" })
                    end
                "#,
                )
                .eval()
                .expect("build caller");
            f.call_async::<String>(())
                .await
                .expect("alc.llm async call")
        });

    assert_eq!(out, "echo:hello");
}

/// role="nn" with an unregistered model name surfaces a clear Lua error rather
/// than falling through to the Host path. Distinguishes "no such model" from
/// "Host bridge send failed" so callers get an actionable message.
#[test]
fn alc_llm_role_nn_unknown_model_errors() {
    let (lua, _tmp) = production_vm();

    let err = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(async {
            let f: mlua::Function = lua
                .load(
                    r#"
                    return function()
                        return alc.llm("hi", { role = "nn", model = "does-not-exist" })
                    end
                "#,
                )
                .eval()
                .expect("build caller");
            f.call_async::<String>(())
                .await
                .expect_err("should error on unknown model")
                .to_string()
        });

    assert!(
        err.contains("no model registered"),
        "unexpected error: {err}"
    );
}
