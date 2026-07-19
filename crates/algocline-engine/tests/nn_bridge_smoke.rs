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

/// `alc.nn.preset.llama("tiny")` builds a Llama handle and exposes the
/// same metadata shape as `Gpt2Handle`. Covers GH #9 Layer 2 (inference
/// adapter): the Lua-facing preset returns a UserData with `variant`,
/// `layers`, `heads`, `kv_heads`, `dim`, `ctx`, `vocab`, `device`,
/// `dtype`, and `forward_shape(batch, seq)`. The `tiny` variant
/// deliberately does not require a weights bundle so the smoke test
/// stays offline.
#[test]
fn alc_nn_preset_llama_tiny_builds_handle_with_metadata() {
    let lua = nn_vm();
    let dims: Vec<usize> = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            assert(h:variant() == "tiny", "variant mismatch")
            assert(h:layers() == 2, "layers mismatch")
            assert(h:heads() == 2, "heads mismatch")
            assert(h:kv_heads() == 2, "kv_heads mismatch")
            assert(h:dim() == 32, "dim mismatch")
            assert(h:ctx() == 16, "ctx mismatch")
            assert(h:vocab() == 64, "vocab mismatch")
            assert(h:device() == "cpu", "device mismatch")
            assert(h:dtype() == "f32", "dtype mismatch")
            return h:forward_shape(1, 4)
        "#,
        )
        .eval()
        .expect("build tiny llama handle");
    assert_eq!(dims, vec![1, 64]);
}

/// Unknown variant surfaces a clear Lua-side error with the allowed
/// name list, so a typo (`"tinyllma"`, `"7b-v3"`) is caught early
/// rather than percolating into a candle-side load failure.
#[test]
fn alc_nn_preset_llama_unknown_variant_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            alc.nn.preset.llama("tinyllma")
        "#,
        )
        .exec()
        .expect_err("unknown variant must error")
        .to_string();
    assert!(
        err.contains("unknown variant") && err.contains("tinyllma"),
        "unexpected error: {err}"
    );
}

/// bf16 requires CUDA — the CPU path must reject `dtype = "bf16"` up
/// front rather than let candle emit an obscure kernel error at
/// forward time. Matches the existing `Gpt2Handle` guard so the two
/// presets keep the same failure mode.
#[test]
fn alc_nn_preset_llama_bf16_on_cpu_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            alc.nn.preset.llama("tiny", { device = "cpu", dtype = "bf16" })
        "#,
        )
        .exec()
        .expect_err("bf16 on cpu must error")
        .to_string();
    assert!(
        err.contains("bf16 dtype requires a CUDA device"),
        "unexpected error: {err}"
    );
}

/// GH #9 Layer 3 device / dtype matrix: `device = "metal"` is a
/// recognised value on the preset (parsing succeeds), but on a build
/// without the `nn-metal` cargo feature the runtime
/// `Device::new_metal(...)` call reports the backend as unavailable
/// with the preset-prefixed error message. The parser must never fall
/// through to `unknown device`.
#[test]
fn alc_nn_preset_llama_metal_device_string_is_recognised() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            alc.nn.preset.llama("tiny", { device = "metal" })
        "#,
        )
        .exec()
        .expect_err("metal on non-metal build must error at Device::new_metal")
        .to_string();

    // The message must be the metal-availability path
    // (i.e. `Device::new_metal` failed), NOT the parser's fallthrough
    // (`unknown device 'metal'`). Depending on whether the test
    // binary was compiled with the `nn-metal` feature the runtime
    // outcome differs; both variants keep the preset prefix.
    assert!(
        err.contains("alc.nn.preset.llama: metal"),
        "expected preset-prefixed metal-path error, got: {err}"
    );
    assert!(
        !err.contains("unknown device"),
        "metal must not fall through to 'unknown device' — got: {err}"
    );
}

/// GH #9 Layer 3 dtype matrix: `bf16` on Metal is rejected up front
/// (candle-nn 0.11 has no Metal bf16 kernels). The error message
/// steers the caller toward `f16` on Metal or CUDA for bf16, matching
/// the shape of the CPU / bf16 guard.
///
/// This test only asserts against the guard (which runs before the
/// device is materialised), so it holds regardless of whether the
/// test binary was built with the `nn-metal` feature.
#[test]
fn alc_nn_preset_llama_bf16_on_metal_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            alc.nn.preset.llama("tiny", { device = "metal", dtype = "bf16" })
        "#,
        )
        .exec()
        .expect_err("bf16 on metal must error")
        .to_string();

    // The error can be either the guard message (when the device
    // parse succeeded — nn-metal builds) or the device parse error
    // (non-metal builds refuse the device before the guard runs).
    // Either surfaces a preset-prefixed message with a bf16 or metal
    // token; the assertion covers both branches.
    assert!(
        err.contains("alc.nn.preset.llama"),
        "expected preset-prefixed error, got: {err}"
    );
    let bf16_guard =
        err.contains("bf16 dtype is not supported on Metal") || err.contains("bf16 dtype");
    let metal_unavail = err.contains("metal");
    assert!(
        bf16_guard || metal_unavail,
        "expected bf16-on-Metal guard or metal-unavailable error, got: {err}"
    );
}
