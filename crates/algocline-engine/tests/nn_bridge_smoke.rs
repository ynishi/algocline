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

// ─── alc.nn.preset.gpt2("custom", ...) — nn arch Phase 3 ─────────────
//
// Lua-facing regression fence for the Gpt2Custom expose (issue
// 9218e983). tests/e2e.rs cannot exercise this surface: the default
// `alc` binary is built without the `nn` feature, so the canonical
// "normal / representative-error / shape-mismatch" trio lives here,
// on the same engine bridge path `alc_run` executes when the feature
// is on (`just test-nn` runs this file in CI).

/// Normal path: a kitchen-sink custom spec (every Phase 1+2 axis off
/// the reference, Post-LN left out because it excludes Parallel)
/// builds a handle on the tiny base shape with a `vocab` override,
/// then completes a short `full_ft` run — the issue's definition of
/// done is "rewire from Lua AND train through the existing trainer
/// binding".
#[test]
fn alc_nn_preset_gpt2_custom_builds_and_trains() {
    let (lua, _tmp) = production_vm();
    let step: usize = lua
        .load(
            r#"
            local h = alc.nn.preset.gpt2("custom", {
                pretrained = false,
                device = "cpu",
                act = "swiglu",
                norm = "rmsnorm",
                residual = "parallel",
                mlp_ratio = 3,
                pos = "rope",
                kv_heads = 1,
                window = 4,
                untied_head = true,
                vocab = 96,
            })
            assert(h:variant() == "custom", "variant mismatch: " .. h:variant())
            assert(h:layers() == 2, "layers must stay at the tiny base (2)")
            assert(h:heads() == 2, "heads must stay at the tiny base (2)")
            assert(h:dim() == 32, "dim must stay at the tiny base (32)")
            assert(h:vocab() == 96, "vocab override must apply")
            assert(h:pretrained() == false, "custom is random-init only")

            local rows = {
                { 1, 5, 12, 20, 33, 44, 51, 60 },
                { 2, 8, 15, 22, 30, 40, 55, 63 },
            }
            local ds = alc.nn.data.synthetic(rows, {
                batch_size = 1,
                ctx_len = 8,
                shuffle = false,
                pad_id = 0,
            })
            local ckpt = alc.nn.trainer.full_ft(h, ds, {
                lr = 3e-4,
                batch_size = 1,
                steps = 2,
                warmup = 0,
                schedule = "cosine",
                weight_decay = 0.0,
                ckpt_every = 0,
                card_id = "smoke_custom_full_ft",
            })
            assert(type(ckpt.train_loss) == "number", "train_loss must be a number")
            return ckpt.step
        "#,
        )
        .eval()
        .expect("custom preset build + full_ft");
    assert_eq!(step, 2);
}

/// MoE co-placement: `moe = {...}` composes with the non-dense axes
/// (build succeeds), while the dense-MLP knobs (`act` / `mlp_ratio`)
/// combined with `moe` are rejected Rust-side with the message
/// steering back to the reference values.
#[test]
fn alc_nn_preset_gpt2_custom_moe_composes_and_rejects_dense_knobs() {
    let lua = nn_vm();

    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("custom", {
            pretrained = false,
            norm = "rmsnorm",
            moe = { n_experts = 2 },
        })
        assert(h:variant() == "custom", "variant mismatch")
    "#,
    )
    .exec()
    .expect("custom + moe (non-dense axes) must build");

    let err = lua
        .load(
            r#"alc.nn.preset.gpt2("custom", {
                pretrained = false, act = "swiglu", moe = { n_experts = 2 } })"#,
        )
        .exec()
        .expect_err("dense-MLP knob + moe must be rejected")
        .to_string();
    assert!(
        err.contains("do not apply to the MoE experts"),
        "unexpected error: {err}"
    );
}

/// Representative errors: Rust-side validation messages propagate to
/// Lua as actionable strings (Service-layer error discipline) — the
/// PostLN×Parallel exclusion, the GQA divisibility check, the
/// random-init-only pretrained guard, and the custom-keys-on-stock-
/// variant rejection.
#[test]
fn alc_nn_preset_gpt2_custom_validation_errors_propagate() {
    let lua = nn_vm();

    let err = lua
        .load(
            r#"alc.nn.preset.gpt2("custom", {
                pretrained = false, placement = "postln", residual = "parallel" })"#,
        )
        .exec()
        .expect_err("PostLN x Parallel must be rejected")
        .to_string();
    assert!(err.contains("Post-LN"), "unexpected error: {err}");

    let err = lua
        .load(
            r#"alc.nn.preset.gpt2("custom", {
                pretrained = false, kv_heads = 3 })"#, // tiny base: heads = 2
        )
        .exec()
        .expect_err("non-divisible kv_heads must be rejected")
        .to_string();
    assert!(err.contains("divisible"), "unexpected error: {err}");

    let err = lua
        .load(r#"alc.nn.preset.gpt2("custom", { act = "swiglu" })"#)
        .exec()
        .expect_err("default pretrained=true must be rejected for custom")
        .to_string();
    assert!(
        err.contains("random-init only") && err.contains("pretrained = false"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(r#"alc.nn.preset.gpt2("medium", { act = "swiglu", pretrained = false })"#)
        .exec()
        .expect_err("custom axis on a stock variant must be rejected, not ignored")
        .to_string();
    assert!(
        err.contains("only applies to the 'custom' variant"),
        "unexpected error: {err}"
    );
}

/// Shape mismatches: a present key of the wrong Lua type is a hard
/// error naming the key and the expected type — never a silently
/// ignored option (the legacy `.ok()`-swallow pattern is explicitly
/// not carried into the custom parse).
#[test]
fn alc_nn_preset_gpt2_custom_type_mismatches_error() {
    let lua = nn_vm();

    let err = lua
        .load(r#"alc.nn.preset.gpt2("custom", { pretrained = false, act = true })"#)
        .exec()
        .expect_err("boolean act must be a type error")
        .to_string();
    assert!(
        err.contains("'act' must be a string"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(r#"alc.nn.preset.gpt2("custom", { pretrained = false, mlp_ratio = "three" })"#)
        .exec()
        .expect_err("string mlp_ratio must be a type error")
        .to_string();
    assert!(
        err.contains("'mlp_ratio' must be an integer"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(r#"alc.nn.preset.gpt2("custom", { pretrained = false, moe = { top_k = 2 } })"#)
        .exec()
        .expect_err("moe without n_experts must error")
        .to_string();
    assert!(
        err.contains("moe.n_experts is required"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(r#"alc.nn.preset.gpt2("custom", { pretrained = false, act = "swiglue" })"#)
        .exec()
        .expect_err("typo'd enum value must list the valid set")
        .to_string();
    assert!(
        err.contains("unknown act 'swiglue'") && err.contains("'swiglu'"),
        "unexpected error: {err}"
    );
}

/// The arch-neutral `alc.nn.preset(arch, variant, opts)` handle and the
/// typed `alc.nn.preset.llama(variant, opts)` handle must answer every
/// shared accessor identically.
///
/// Both project through the bridge's `HandleMeta` now, instead of the
/// union carrying its own per-accessor `match` over the three arch
/// arms. A divergence here means one of the two paths stopped
/// delegating — exactly the drift the old duplicated dispatch was prone
/// to, since adding an accessor meant remembering to add it in both
/// places.
#[test]
fn neutral_and_typed_llama_handles_agree_on_shared_accessors() {
    let lua = nn_vm();
    let checked: usize = lua
        .load(
            r#"
            local opts    = { device = "cpu", dtype = "f32" }
            local typed   = alc.nn.preset.llama("tiny", opts)
            local neutral = alc.nn.preset("llama", "tiny", opts)

            local scalar_accessors = {
                "variant", "layers", "heads", "kv_heads", "dim",
                "ctx", "vocab", "device", "dtype", "pretrained",
            }
            for _, name in ipairs(scalar_accessors) do
                local a = typed[name](typed)
                local b = neutral[name](neutral)
                assert(a == b, name .. " mismatch: " .. tostring(a) .. " vs " .. tostring(b))
            end

            local a = typed:forward_shape(2, 5)
            local b = neutral:forward_shape(2, 5)
            assert(#a == #b, "forward_shape rank mismatch")
            for i = 1, #a do
                assert(a[i] == b[i], "forward_shape[" .. i .. "] mismatch")
            end
            -- Adapter arch slices the last-token logits, so seq drops out.
            assert(#a == 2 and a[1] == 2 and a[2] == 64, "adapter logits shape must be [batch, vocab]")

            -- `arch` is the one accessor unique to the neutral union.
            assert(neutral:arch() == "llama", "arch mismatch")
            assert(typed.arch == nil, "typed handle must not grow an arch accessor")

            return #scalar_accessors
        "#,
        )
        .eval()
        .expect("neutral and typed llama handles agree");
    assert_eq!(checked, 10);
}

/// `kv_heads` and `pretrained` are now part of the shared accessor
/// surface every handle exposes, so the typed handles answer them too.
///
/// Previously `Gpt2Handle` had no `kv_heads` and `LlamaHandle` had no
/// `pretrained` — both were reachable only after wrapping the handle in
/// the neutral union, which meant a Lua caller's available methods
/// depended on which entry point built the handle. GPT-2 is multi-head
/// attention, so its `kv_heads` mirrors `heads`; the Llama adapter is
/// inference-only, so its `pretrained` is `true`.
#[test]
fn typed_handles_expose_the_full_shared_accessor_surface() {
    let lua = nn_vm();
    lua.load(
        r#"
            local g = alc.nn.preset.gpt2("custom", { pretrained = false, device = "cpu" })
            assert(g:kv_heads() == g:heads(), "gpt2 is MHA: kv_heads must mirror heads")
            assert(g:pretrained() == false, "random-init handle must report pretrained=false")

            local l = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            assert(l:pretrained() == true, "adapter handles always report pretrained=true")
            assert(l:kv_heads() == 2, "llama tiny kv_heads mismatch")
        "#,
    )
    .exec()
    .expect("typed handles expose the shared accessor surface");
}
