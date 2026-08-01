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

/// Hand-authored `tokenizers` fixture (WordLevel + WhitespaceSplit),
/// mirroring the one `bridge/nn_card.rs` tests use. Seeded on disk under
/// the VM's `nn_dir` so `HfTokenizer::load_cached("gpt2", ..)` takes its
/// cache-hit branch and the test never reaches the HuggingFace hub.
const FIXTURE_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
 "added_tokens":[{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
 "normalizer":null,
 "pre_tokenizer":{"type":"WhitespaceSplit"},
 "post_processor":null,"decoder":null,
 "model":{"type":"WordLevel","vocab":{"[UNK]":0,"alpha":1,"beta":2,"gamma":3,"delta":4,"epsilon":5},"unk_token":"[UNK]"}}"#;

/// `tokenizer_config.json` for the fixture preset, seeded beside the
/// tokenizer under the name `HfTokenizer` reads (`<preset>-config.json`).
/// The template is the shape a small chat model ships — one wrapped
/// block per turn plus an `add_generation_prompt`-gated assistant
/// opening — so `alc.nn.chat_prompt` resolves without a hub fetch.
const FIXTURE_TOKENIZER_CONFIG: &str = r#"{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}{{ eos_token }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
}"#;

/// VM whose `nn_dir` already holds the fixture tokenizer, so
/// `alc.nn.tokenize` / `alc.nn.detokenize` / `alc.nn.chat_prompt`
/// resolve offline. The `install_for_pkg_test` path cannot be used here:
/// it owns its tempdir internally, leaving no place to seed the cache
/// before registration.
fn tokenizer_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("test tempdir");
    let root: PathBuf = tmp.path().to_path_buf();
    let nn_dir = root.join("nn");
    std::fs::create_dir_all(nn_dir.join("tokenizers")).expect("seed tokenizer dir");
    std::fs::write(nn_dir.join("tokenizers/gpt2.json"), FIXTURE_TOKENIZER)
        .expect("seed tokenizer fixture");
    std::fs::write(
        nn_dir.join("tokenizers/gpt2-config.json"),
        FIXTURE_TOKENIZER_CONFIG,
    )
    .expect("seed tokenizer config fixture");

    let config = BridgeConfig {
        llm_tx: None,
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
        nn_dir,
        log_sink: None,
    };

    let alc_table = lua.create_table().expect("create alc table");
    bridge::register(&lua, &alc_table, config).expect("register");
    lua.globals().set("alc", alc_table).expect("set alc global");
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
                ckpt_prefix = "smoke_custom_full_ft",
            })
            assert(type(ckpt.train_loss) == "number", "train_loss must be a number")
            return ckpt.step
        "#,
        )
        .eval()
        .expect("custom preset build + full_ft");
    assert_eq!(step, 2);
}

/// `alc.nn.trainer.full_ft`'s `on_ckpt` hook fires at each
/// `ckpt_every` boundary and can early-break the run by returning
/// `"break"`. The returned Checkpoint carries the `early_break = 1.0`
/// metric so downstream consumers can distinguish an early stop from
/// a full-run save.
///
/// End-to-end coverage of the Lua-side wiring added for the metric
/// infra iter (`fc070f15`): parse `on_ckpt` opts →
/// `extract_on_ckpt_hook` → `CkptHook` adapter → trainer fire path.
#[test]
fn alc_nn_trainer_full_ft_on_ckpt_break_stops_early() {
    let (lua, _tmp) = production_vm();
    let (step, early_break, fires) = lua
        .load(
            r#"
            local h = alc.nn.preset.gpt2("custom", {
                pretrained = false,
                device = "cpu",
                vocab = 32,
            })
            -- Enough rows for `steps=8` at batch_size=1 so the loop
            -- can reach the second `ckpt_every=2` boundary before
            -- exhausting the dataset (TokenizedDataset does not
            -- cycle when `shuffle=false`).
            local rows = {
                { 1, 5, 12, 20, 30, 3, 8, 15 },
                { 2, 8, 15, 22, 30, 4, 9, 16 },
                { 3, 7, 14, 21, 28, 5, 10, 17 },
                { 4, 6, 13, 19, 27, 6, 11, 18 },
                { 5, 9, 16, 23, 29, 7, 12, 19 },
                { 6, 10, 11, 24, 26, 8, 13, 20 },
                { 7, 11, 17, 25, 31, 9, 14, 21 },
                { 8, 12, 18, 20, 25, 10, 15, 22 },
            }
            local ds = alc.nn.data.synthetic(rows, {
                batch_size = 1,
                ctx_len = 8,
                shuffle = false,
                pad_id = 0,
            })

            local fires = 0
            local ckpt = alc.nn.trainer.full_ft(h, ds, {
                lr = 3e-4,
                batch_size = 1,
                steps = 8,
                warmup = 0,
                schedule = "constant",
                weight_decay = 0.0,
                ckpt_every = 2,
                ckpt_keep = 3,
                ckpt_prefix = "smoke_on_ckpt_break",
                on_ckpt = function(info)
                    -- Sanity: every documented field is present and
                    -- has a plausible type.
                    assert(type(info.step) == "number", "info.step must be number")
                    assert(type(info.ckpt_path) == "string", "info.ckpt_path must be string")
                    assert(type(info.train_loss) == "number", "info.train_loss must be number")
                    assert(type(info.lr) == "number", "info.lr must be number")
                    assert(type(info.grad_norm) == "number", "info.grad_norm must be number")
                    assert(type(info.elapsed_ms) == "number", "info.elapsed_ms must be number")
                    assert(type(info.min_train_loss) == "number",
                        "info.min_train_loss must be number")
                    fires = fires + 1
                    if fires >= 2 then
                        return "break"
                    end
                    return "continue"
                end,
            })

            return ckpt.step, ckpt.metrics.early_break or 0, fires
        "#,
        )
        .eval::<(i64, f64, i64)>()
        .expect("full_ft with on_ckpt early break");

    // ckpt_every = 2, break on 2nd fire → step 4.
    assert_eq!(step, 4, "trainer must stop at the break-triggering step");
    assert!(
        (early_break - 1.0).abs() < 1e-9,
        "metrics.early_break must be 1.0 on early-break exit, got {early_break}"
    );
    assert_eq!(fires, 2, "hook must have fired exactly twice before break");
}

/// A Lua-side error inside `on_ckpt` surfaces as a Lua error with the
/// `alc.nn.trainer` prefix (via `TrainError::Hook` → `train_err_to_lua`),
/// not as a silent PASS or a Rust panic.
#[test]
fn alc_nn_trainer_full_ft_on_ckpt_error_propagates() {
    let (lua, _tmp) = production_vm();
    let err = lua
        .load(
            r#"
            local h = alc.nn.preset.gpt2("custom", {
                pretrained = false,
                device = "cpu",
                vocab = 32,
            })
            local rows = {
                { 1, 5, 12, 20, 30, 3, 8, 15 },
                { 2, 8, 15, 22, 30, 4, 9, 16 },
            }
            local ds = alc.nn.data.synthetic(rows, {
                batch_size = 1,
                ctx_len = 8,
                shuffle = false,
                pad_id = 0,
            })
            alc.nn.trainer.full_ft(h, ds, {
                lr = 3e-4,
                batch_size = 1,
                steps = 4,
                warmup = 0,
                schedule = "constant",
                weight_decay = 0.0,
                ckpt_every = 2,
                ckpt_keep = 1,
                ckpt_prefix = "smoke_on_ckpt_err",
                on_ckpt = function(_info)
                    return "sideways"
                end,
            })
        "#,
        )
        .exec()
        .expect_err("invalid on_ckpt return must surface as Lua error");
    let msg = err.to_string();
    assert!(
        msg.contains("alc.nn.trainer") && msg.contains("on_ckpt"),
        "error must carry the trainer prefix and mention on_ckpt: {msg}"
    );
    assert!(
        msg.contains("sideways"),
        "error must name the offending return value: {msg}"
    );
}

/// Cross-module seam guard: the Card that `alc.nn.trainer.run_full_ft`
/// writes (bridge/nn_trainer.rs) must resolve through
/// `alc.nn.card.load_handle` (bridge/nn_card.rs) — bundle path,
/// `bundle_ref = "nn/<card_id>"`, and `architecture` family-prefix
/// dispatch all have to line up across the two modules. The per-module
/// unit tests each assert their own half; this is the only place the
/// whole chain runs from Lua. Motivated by the 2026-07-30 eval iter,
/// where the raw-Checkpoint surface's then-named `ckpt.card_id` (a
/// checkpoint filename prefix, not a Card id — renamed to
/// `ckpt.ckpt_prefix` in the Card-domain refactor) was mistaken for a
/// loadable Card id.
#[test]
fn alc_nn_run_full_ft_card_load_handle_roundtrip() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("tiny", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
        })

        local rows = {}
        for r = 1, 8 do
            local row = {}
            for j = 1, 8 do row[j] = ((r + j) % 60) + 1 end
            rows[r] = row
        end
        local ds = alc.nn.data.synthetic(rows, {
            batch_size = 1,
            ctx_len = 8,
            shuffle = false,
            pad_id = 0,
        })

        local card_id = alc.nn.trainer.run_full_ft(h, ds, {
            lr = 1e-3,
            batch = 1,
            steps = 3,
            warmup = 0,
            schedule = "Constant",
            name = "rt_smoke",
        })
        assert(type(card_id) == "string" and #card_id > 0,
            "run_full_ft must return a non-empty card_id")

        local reloaded = alc.nn.card.load_handle(card_id)
        assert(reloaded ~= nil, "load_handle returned nil")
        assert(reloaded:vocab() == h:vocab(),
            "vocab mismatch: " .. reloaded:vocab() .. " vs " .. h:vocab())
        assert(reloaded:ctx() == h:ctx(), "ctx mismatch")
        assert(reloaded:layers() == h:layers(), "layers mismatch")
        assert(reloaded:dim() == h:dim(), "dim mismatch")

        -- The reloaded NnHandle must also generate: this is the last
        -- leg of the Lua-only "train -> Card -> reload -> generate"
        -- loop (stateless session backend on the trainable arch).
        local session = reloaded:generate_session({ 1, 2, 3 })
        local logits = session:next_logits()
        assert(logits:vocab() == h:vocab(),
            "reloaded handle logits must span the model vocab")
        session:append(logits:argmax())
        assert(session:position() == 4, "session must advance after append")
    "#,
    )
    .exec()
    .expect("run_full_ft -> card.load_handle round-trip");
}

/// Same cross-module seam as the round-trip above, on a `custom`
/// architecture. This is the only place the whole save/load contract
/// for a customized shape runs end to end:
///
/// - the trainer records `metadata.nn.candle.custom` off the live
///   `Gpt2Config` (bridge/nn_trainer.rs), and
/// - `load_handle` rebuilds that exact config from the Card
///   (bridge/nn_card.rs) — `architecture = "gpt2-custom"` pins no
///   shape, so without the branch the load has nothing to go on.
///
/// Every axis below is off the reference *and* changes the bundle's
/// Var set or a weight shape (`swiglu` adds `mlp.c_gate`, `rmsnorm`
/// drops the LayerNorm biases, `rope` drops `wpe`, `untied_head` adds
/// `lm_head.weight`, `mlp_ratio = 3` resizes `mlp.c_fc`, `kv_heads =
/// 1` resizes the KV projections), so a spec field lost in the Card
/// round-trip surfaces as a load failure rather than a silently
/// different model.
#[test]
fn alc_nn_run_full_ft_custom_card_load_handle_roundtrip() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("custom", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
            act = "swiglu",
            norm = "rmsnorm",
            pos = "rope",
            mlp_ratio = 3,
            kv_heads = 1,
            untied_head = true,
            vocab = 96,
            ctx = 32,
        })

        local rows = {}
        for r = 1, 8 do
            local row = {}
            for j = 1, 8 do row[j] = ((r + j) % 90) + 1 end
            rows[r] = row
        end
        local ds = alc.nn.data.synthetic(rows, {
            batch_size = 1,
            ctx_len = 8,
            shuffle = false,
            pad_id = 0,
        })

        local card_id = alc.nn.trainer.run_full_ft(h, ds, {
            lr = 1e-3,
            batch = 1,
            steps = 2,
            warmup = 0,
            schedule = "Constant",
            name = "rt_custom_smoke",
        })
        assert(type(card_id) == "string" and #card_id > 0,
            "run_full_ft must return a non-empty card_id")

        -- Save side: the shape block must be on the Card, with the
        -- same vocabulary the caller wrote above.
        local card = alc.card.get(card_id)
        assert(card ~= nil, "card.get returned nil")
        local custom = card.metadata.nn.candle.custom
        assert(custom ~= nil, "metadata.nn.candle.custom must be recorded")
        assert(custom.vocab == 96, "custom.vocab: " .. tostring(custom.vocab))
        assert(custom.ctx == 32, "custom.ctx: " .. tostring(custom.ctx))
        assert(custom.layers == 2, "custom.layers: " .. tostring(custom.layers))
        assert(custom.heads == 2, "custom.heads: " .. tostring(custom.heads))
        assert(custom.dim == 32, "custom.dim: " .. tostring(custom.dim))
        assert(custom.moe == nil, "no moe was requested")
        assert(custom.spec.act == "swiglu", "spec.act: " .. tostring(custom.spec.act))
        assert(custom.spec.norm == "rmsnorm", "spec.norm: " .. tostring(custom.spec.norm))
        assert(custom.spec.pos == "rope", "spec.pos: " .. tostring(custom.spec.pos))
        assert(custom.spec.mlp_ratio == 3,
            "spec.mlp_ratio: " .. tostring(custom.spec.mlp_ratio))
        assert(custom.spec.kv_heads == 1,
            "spec.kv_heads: " .. tostring(custom.spec.kv_heads))
        assert(custom.spec.untied_head == true,
            "spec.untied_head: " .. tostring(custom.spec.untied_head))

        -- Load side: the rebuilt config must match the bundle's Var
        -- set (a mismatch errors inside from_safetensors_file) and
        -- report the trained shape back.
        local reloaded = alc.nn.card.load_handle(card_id)
        assert(reloaded ~= nil, "load_handle returned nil")
        assert(reloaded:vocab() == 96, "vocab mismatch: " .. reloaded:vocab())
        assert(reloaded:ctx() == 32, "ctx mismatch: " .. reloaded:ctx())
        assert(reloaded:layers() == 2, "layers mismatch: " .. reloaded:layers())
        assert(reloaded:dim() == 32, "dim mismatch: " .. reloaded:dim())

        -- ... and generate, closing the same "train -> Card -> reload
        -- -> generate" loop the named-variant test walks.
        local session = reloaded:generate_session({ 1, 2, 3 })
        for _ = 1, 3 do
            local logits = session:next_logits()
            assert(logits:vocab() == 96,
                "reloaded custom handle logits must span the custom vocab")
            session:append(logits:argmax())
        end
        assert(session:position() == 6, "session must advance once per append")
    "#,
    )
    .exec()
    .expect("custom run_full_ft -> card.load_handle round-trip");
}

/// Regression guard for the GPT-2 GQA `kv_heads` accessor.
///
/// `Gpt2Handle::meta` used to hard-mirror `self.heads` into the
/// `kv_heads` slot of the shared accessor, so a `custom` build that
/// opted into GQA (`kv_heads = 1` on a 2-head base) misreported
/// `h:kv_heads() == 2` to Lua callers even though the internal
/// `Block` builder used the correct `1`. The forward pass stayed
/// numerically sound, but any Lua caller that trusted `:kv_heads()`
/// for a KV-cache size estimate or a GQA branch selector silently
/// wired the wrong value.
///
/// The reload path (`alc.nn.card.load_handle`) is a fresh consumer of
/// the same handle field — the `custom_bundle_reloads_from_safetensors`
/// unit test in the `nn` crate proves the config round-trips through
/// the bundle, but the Lua-facing accessor is only exercisable
/// through the engine bridge. Both build and reload must pin
/// `:kv_heads() == 1` here.
#[test]
fn alc_nn_gpt2_custom_gqa_reports_configured_kv_heads_build_and_reload() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("custom", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
            kv_heads = 1,
            vocab = 96,
            ctx = 16,
        })
        assert(h:heads() == 2, "tiny base has 2 query heads")
        assert(h:kv_heads() == 1,
            "GQA build path must report the configured kv_heads (got " ..
            tostring(h:kv_heads()) .. "), not mirror :heads()")

        -- Round-trip the shape through train -> Card -> reload so the
        -- reload path (`gpt2_from_safetensors`) is also pinned. Two
        -- training steps are enough to close the save side; the
        -- accessor assertion below covers the load side.
        local rows = {}
        for r = 1, 8 do
            local row = {}
            for j = 1, 8 do row[j] = ((r + j) % 90) + 1 end
            rows[r] = row
        end
        local ds = alc.nn.data.synthetic(rows, {
            batch_size = 1,
            ctx_len = 8,
            shuffle = false,
            pad_id = 0,
        })
        local card_id = alc.nn.trainer.run_full_ft(h, ds, {
            lr = 1e-3,
            batch = 1,
            steps = 2,
            warmup = 0,
            schedule = "Constant",
            name = "kv_heads_regression",
        })
        assert(type(card_id) == "string" and #card_id > 0,
            "run_full_ft must return a non-empty card_id")

        local reloaded = alc.nn.card.load_handle(card_id)
        assert(reloaded ~= nil, "load_handle returned nil")
        assert(reloaded:heads() == 2,
            "reloaded heads mismatch: " .. tostring(reloaded:heads()))
        assert(reloaded:kv_heads() == 1,
            "GQA reload path must report the configured kv_heads (got " ..
            tostring(reloaded:kv_heads()) .. "), not mirror :heads()")
    "#,
    )
    .exec()
    .expect("gpt2 custom GQA kv_heads must survive build + reload");
}

/// The stateless session backend on a trainable GPT-2 handle: the
/// decode loop written for Llama sessions runs unchanged, including
/// the no-pending-tokens guard.
#[test]
fn alc_nn_gpt2_generate_session_stateless_loop() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("tiny", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
        })
        local s = h:generate_session({ 1, 2, 3 })
        local l1 = s:next_logits()
        assert(l1:vocab() == h:vocab(), "logits row must span the model vocab")
        s:append(l1:argmax())
        assert(s:position() == 4, "position after one append")
        local l2 = s:next_logits()
        assert(l2:vocab() == h:vocab(), "second step logits row")
        local ok, err = pcall(function() return s:next_logits() end)
        assert(not ok, "next_logits without append must error")
        assert(tostring(err):find("no pending tokens", 1, true),
            "unexpected error: " .. tostring(err))
    "#,
    )
    .exec()
    .expect("gpt2 stateless session loop");
}

/// TinyLlama arm of the stateless session backend (the second
/// trainable arch goes through its own `Module::forward`, so the
/// GPT-2 test alone would leave this dispatch arm unexercised).
#[test]
fn alc_nn_tinyllama_generate_session_stateless_loop() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.tinyllama("tiny", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
        })
        local s = h:generate_session({ 1, 2 })
        local l = s:next_logits()
        assert(l:vocab() == h:vocab(), "logits row must span the model vocab")
        s:append(l:argmax())
        assert(s:position() == 3, "position after one append")
    "#,
    )
    .exec()
    .expect("tinyllama stateless session loop");
}

/// A stateless session cannot generate past the model's context
/// window — the guard must fire as a session-level error naming ctx,
/// not as a positional-embedding failure from inside candle.
#[test]
fn alc_nn_gpt2_session_history_over_ctx_errors() {
    let (lua, _tmp) = production_vm();
    lua.load(
        r#"
        local h = alc.nn.preset.gpt2("tiny", {
            pretrained = false,
            device = "cpu",
            dtype = "f32",
        })
        assert(h:ctx() == 16, "tiny preset ctx expected to be 16")
        local prompt = {}
        for i = 1, 17 do prompt[i] = 1 end
        local s = h:generate_session(prompt)
        local ok, err = pcall(function() return s:next_logits() end)
        assert(not ok, "over-ctx history must error")
        assert(tostring(err):find("context window", 1, true),
            "unexpected error: " .. tostring(err))
    "#,
    )
    .exec()
    .expect("gpt2 over-ctx guard");
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

// ─── generation sessions (Sampler plan Step 4) ───────────────────────
//
// The Lua-facing decode surface: `handle:generate_session(prompt)` plus
// `next_logits` / `append` / `tokens` / `position`. Value-level claims
// about the logits row (shape, dtype, two-session isolation) live in
// `bridge/nn_gen.rs`'s in-crate tests, because `LogitsHandle` is opaque
// by design and this crate cannot read it.

/// The decode loop runs from Lua and the session's bookkeeping stays
/// consistent: `position` counts every token (prompt included),
/// `tokens` returns the full history in order, and `next_logits`
/// answers with a `[vocab]`-wide row after each committed token.
#[test]
fn alc_nn_generate_session_loop_runs_from_lua() {
    let lua = nn_vm();
    let position: usize = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            local s = h:generate_session({ 1, 2, 3 })
            assert(s:position() == 3, "prompt tokens must count toward position")

            local first = s:next_logits()
            assert(first:vocab() == 64, "logits row must span the vocabulary")

            s:append(7)
            assert(s:position() == 4, "append must advance position")
            local second = s:next_logits()
            assert(second:vocab() == 64, "incremental step must return a full row")

            s:append(9)
            local t = s:tokens()
            assert(#t == 5, "tokens() must return prompt + appended")
            assert(t[1] == 1 and t[2] == 2 and t[3] == 3, "prompt must be preserved in order")
            assert(t[4] == 7 and t[5] == 9, "appended tokens must be preserved in order")

            return s:position()
        "#,
        )
        .eval()
        .expect("generation session loop");
    assert_eq!(position, 5);
}

/// Two sessions built from the same handle keep separate histories. The
/// weights are shared through an `Arc`; the KV cache and the token list
/// are not, which is the whole reason the bare adapter `forward` is not
/// exposed to Lua.
#[test]
fn alc_nn_two_sessions_from_one_handle_keep_separate_state() {
    let lua = nn_vm();
    lua.load(
        r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            local a = h:generate_session({ 1, 2, 3 })
            local b = h:generate_session({ 10, 11 })

            a:next_logits()
            b:next_logits()
            a:append(20)
            b:append(30)
            a:next_logits()
            b:next_logits()

            assert(a:position() == 4, "session A position: " .. a:position())
            assert(b:position() == 3, "session B position: " .. b:position())
            local ta, tb = a:tokens(), b:tokens()
            assert(ta[1] == 1 and ta[4] == 20, "session A history leaked")
            assert(tb[1] == 10 and tb[3] == 30, "session B history leaked")
        "#,
    )
    .exec()
    .expect("two independent sessions");
}

/// Calling `next_logits` twice without an intervening `append` is a
/// loud error: there is nothing new to forward, and silently repeating
/// the previous row would let a decode loop that forgot to commit its
/// sampled token spin forever.
#[test]
fn alc_nn_next_logits_without_append_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            local s = h:generate_session({ 1, 2, 3 })
            s:next_logits()
            s:next_logits()
        "#,
        )
        .exec()
        .expect_err("second next_logits without append must error")
        .to_string();
    assert!(err.contains("no pending tokens"), "unexpected error: {err}");
}

/// An empty prompt has nothing to forward, so the session is refused at
/// construction rather than failing on the first `next_logits`.
#[test]
fn alc_nn_generate_session_empty_prompt_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            h:generate_session({})
        "#,
        )
        .exec()
        .expect_err("empty prompt must error")
        .to_string();
    assert!(
        err.contains("prompt_tokens is empty"),
        "unexpected error: {err}"
    );
}

/// Token ids are checked against the model's vocabulary at the bridge
/// boundary, so an out-of-range id names itself and the bound instead of
/// surfacing as a candle index failure inside the embedding lookup.
#[test]
fn alc_nn_append_out_of_range_token_errors() {
    let lua = nn_vm();
    let err = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            local s = h:generate_session({ 1, 2, 3 })
            s:append(64)   -- tiny variant vocab is 64, so 64 is one past the end
        "#,
        )
        .exec()
        .expect_err("out-of-range token must error")
        .to_string();
    assert!(
        err.contains("outside the model vocabulary"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(
            r#"
            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            h:generate_session({ 1, 999 })
        "#,
        )
        .exec()
        .expect_err("out-of-range prompt token must error")
        .to_string();
    assert!(
        err.contains("prompt_tokens[2]") && err.contains("outside the model vocabulary"),
        "unexpected error: {err}"
    );
}

/// `alc.nn.tokenize` / `alc.nn.detokenize` round-trip through the same
/// `<nn_dir>/tokenizers` cache the `alc.nn.data.*` producers use. The
/// fixture is seeded on disk first, so the tokenizer resolves from cache
/// and the test stays offline.
#[test]
fn alc_nn_tokenize_detokenize_roundtrip() {
    let (lua, _tmp) = tokenizer_vm();
    let text: String = lua
        .load(
            r#"
            local ids = alc.nn.tokenize("gpt2", "alpha beta gamma")
            assert(#ids == 3, "expected three ids, got " .. #ids)
            assert(ids[1] == 1 and ids[2] == 2 and ids[3] == 3, "unexpected ids")
            return alc.nn.detokenize("gpt2", ids)
        "#,
        )
        .eval()
        .expect("tokenize / detokenize roundtrip");
    assert_eq!(text, "alpha beta gamma");
}

/// An unknown preset is rejected by the tokenizer layer with the preset
/// named, rather than falling through to a hub download attempt.
#[test]
fn alc_nn_tokenize_unknown_preset_errors() {
    let (lua, _tmp) = tokenizer_vm();
    let err = lua
        .load(r#"alc.nn.tokenize("nonsense-preset-xyz", "alpha")"#)
        .exec()
        .expect_err("unknown preset must error")
        .to_string();
    assert!(
        err.contains("alc.nn.tokenize") && err.contains("nonsense-preset-xyz"),
        "unexpected error: {err}"
    );
}

// ─── chat templates (Sampler plan Step 6) ────────────────────────────
//
// `alc.nn.chat_prompt(preset, messages)` renders a conversation through
// the preset's own Jinja chat template. The fixture preset seeded by
// `tokenizer_vm` ships one, so these stay offline.

/// The rendered prompt is the preset's template applied verbatim: turn
/// markers, the special tokens from `tokenizer_config.json`, and the
/// assistant opening that makes it a prompt to continue
/// (`add_generation_prompt` is fixed to true on this entry point).
#[test]
fn alc_nn_chat_prompt_renders_a_conversation() {
    let (lua, _tmp) = tokenizer_vm();
    let prompt: String = lua
        .load(
            r#"
            return alc.nn.chat_prompt("gpt2", {
                { role = "system", content = "be brief" },
                { role = "user", content = "hi" },
            })
        "#,
        )
        .eval()
        .expect("chat_prompt renders");
    assert_eq!(
        prompt,
        "<s><|system|>\nbe brief</s>\n<|user|>\nhi</s>\n<|assistant|>\n"
    );
}

/// A turn missing `content` is a caller bug that names its own position:
/// defaulting it would render a turn the model never sees, which
/// surfaces as a bad answer rather than an error.
#[test]
fn alc_nn_chat_prompt_malformed_message_errors() {
    let (lua, _tmp) = tokenizer_vm();
    let err = lua
        .load(r#"alc.nn.chat_prompt("gpt2", { { role = "user" } })"#)
        .exec()
        .expect_err("a message without content must error")
        .to_string();
    assert!(
        err.contains("messages[1] is missing 'content'"),
        "unexpected error: {err}"
    );

    let err = lua
        .load(r#"alc.nn.chat_prompt("gpt2", { "just a string" })"#)
        .exec()
        .expect_err("a non-table message must error")
        .to_string();
    assert!(
        err.contains("messages[1] must be a table"),
        "unexpected error: {err}"
    );
}

/// An unknown role is refused instead of being passed to the template:
/// templates branch on the role string, so an unrecognised one usually
/// falls through every branch and drops the turn from the prompt.
#[test]
fn alc_nn_chat_prompt_unknown_role_errors() {
    let (lua, _tmp) = tokenizer_vm();
    let err = lua
        .load(r#"alc.nn.chat_prompt("gpt2", { { role = "random", content = "hi" } })"#)
        .exec()
        .expect_err("an unknown role must error")
        .to_string();
    assert!(
        err.contains("'random' is not a chat role") && err.contains("assistant"),
        "unexpected error: {err}"
    );
}

/// Preset resolution is the tokenizer layer's, so an unknown preset
/// fails the same way `alc.nn.tokenize` does rather than attempting a
/// hub download.
#[test]
fn alc_nn_chat_prompt_unknown_preset_errors() {
    let (lua, _tmp) = tokenizer_vm();
    let err = lua
        .load(r#"alc.nn.chat_prompt("nonsense-preset-xyz", { { role = "user", content = "hi" } })"#)
        .exec()
        .expect_err("unknown preset must error")
        .to_string();
    assert!(
        err.contains("alc.nn.chat_prompt") && err.contains("nonsense-preset-xyz"),
        "unexpected error: {err}"
    );
}

// ─── samplers and constraints (Sampler plan Step 5) ──────────────────
//
// The Lua-facing token-choosing surface: `alc.nn.sampler.*`,
// `alc.nn.constraint.*`, and the two ranking accessors on a logits
// handle. Every test drives a real `tiny` Llama session, so the claims
// hold on the same path a strategy takes. Value-level claims that need a
// `Tensor` (callback return validation, move semantics at the Rust
// boundary) live in `bridge/nn_sampler.rs`'s in-crate tests.

/// Lua preamble: a `tiny` Llama handle in `h`. The variant is
/// random-init and CPU-only, so every test below stays offline.
const TINY_HANDLE: &str = r#"
    local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
"#;

fn eval_with_handle<T: mlua::FromLuaMulti>(lua: &Lua, body: &str, what: &str) -> T {
    lua.load(format!("{TINY_HANDLE}\n{body}"))
        .eval()
        .unwrap_or_else(|e| panic!("{what}: {e}"))
}

/// Every factory produces a sampler a decode loop can drive: each draws
/// an in-vocabulary token, and an unconstrained sampler never reports
/// itself done (termination is the constraint layer's business).
#[test]
fn alc_nn_sampler_factories_drive_a_session_loop() {
    let lua = nn_vm();
    let positions: Vec<usize> = eval_with_handle(
        &lua,
        r#"
        local samplers = {
            alc.nn.sampler.greedy(),
            alc.nn.sampler.temperature(0.8, 42),
            alc.nn.sampler.top_k_top_p(8, 0.9, 1.0, 7),
            -- both truncations disabled: the sampler reduces to plain
            -- temperature sampling, which is the documented nil handling.
            alc.nn.sampler.top_k_top_p(nil, nil, 1.0, 7),
        }
        local out = {}
        for i, s in ipairs(samplers) do
            local session = h:generate_session({ 1, 2, 3 })
            for _ = 1, 3 do
                local id = s:sample(session:next_logits())
                assert(id >= 0 and id < 64, "sampler " .. i .. " returned " .. tostring(id))
                assert(not s:is_done(), "an unconstrained sampler is never done")
                session:append(id)
            end
            out[i] = session:position()
        end
        return out
    "#,
        "sampler factories drive a session loop",
    );
    assert_eq!(positions, vec![6, 6, 6, 6], "3 prompt + 3 sampled tokens");
}

/// The seed reproducibility guarantee survives the Lua boundary: two
/// temperature samplers built from one seed and fed the same logits
/// stream produce the same tokens.
///
/// That the stream is stochastic at all (different seeds diverge) is
/// asserted in `algocline_nn::sampling`'s unit tests, where the logits
/// are fixed and the claim is not left to a random-init model.
#[test]
fn alc_nn_temperature_sampler_is_reproducible_by_seed_through_lua() {
    let lua = nn_vm();
    let streams: Vec<Vec<u32>> = eval_with_handle(
        &lua,
        r#"
        local function stream(seed)
            local session = h:generate_session({ 1, 2, 3 })
            local s = alc.nn.sampler.temperature(1.0, seed)
            local out = {}
            for i = 1, 5 do
                out[i] = s:sample(session:next_logits())
                session:append(out[i])
            end
            return out
        end

        local a, b = stream(4242), stream(4242)
        for i = 1, 5 do
            assert(a[i] == b[i],
                "seeded streams diverged at " .. i .. ": " .. a[i] .. " vs " .. b[i])
        end
        return { a, b }
    "#,
        "seeded temperature streams",
    );
    assert_eq!(streams[0], streams[1]);
    assert_eq!(streams[0].len(), 5);
}

/// A stop-token constraint terminates the loop: the stop token is
/// sampled like any other (it is never masked away) and `is_done` flips
/// the moment it lands in the prefix.
///
/// The stop set is discovered by probing what greedy emits first, so the
/// test asserts the termination *mechanism* rather than a token id that
/// depends on random-init weights.
#[test]
fn alc_nn_constrained_stop_tokens_terminate_the_loop() {
    let lua = nn_vm();
    let steps: usize = eval_with_handle(
        &lua,
        r#"
        local probe = h:generate_session({ 1, 2, 3 })
        local first = alc.nn.sampler.greedy():sample(probe:next_logits())

        local s = alc.nn.sampler.constrained(
            alc.nn.sampler.greedy(),
            alc.nn.constraint.stop_tokens({ first })
        )
        assert(not s:is_done(), "an empty prefix is not terminal")

        local session = h:generate_session({ 1, 2, 3 })
        local steps = 0
        for _ = 1, 8 do
            local id = s:sample(session:next_logits())
            session:append(id)
            steps = steps + 1
            if s:is_done() then break end
        end
        assert(steps == 1, "the stop token was drawn first but the loop ran " .. steps .. " steps")
        return steps
    "#,
        "stop-token termination",
    );
    assert_eq!(steps, 1);
}

/// Composition moves the inner sampler out of its handle. Using the
/// spent handle afterwards is a loud error, not a second alias onto one
/// RNG — two handles driving one sampler would interleave draws and
/// quietly break the seed reproducibility both of them promise.
#[test]
fn alc_nn_moved_inner_sampler_cannot_be_reused() {
    let lua = nn_vm();
    let err: String = eval_with_handle(
        &lua,
        r#"
        local inner = alc.nn.sampler.greedy()
        alc.nn.sampler.constrained(inner, alc.nn.constraint.stop_tokens({ 0 }))

        local session = h:generate_session({ 1, 2, 3 })
        local logits = session:next_logits()
        local ok, err = pcall(function() return inner:sample(logits) end)
        assert(not ok, "a moved-from sampler must refuse to sample")
        return tostring(err)
    "#,
        "moved inner sampler",
    );
    assert!(err.contains("moved"), "unexpected error: {err}");
}

/// Same move semantics on the constraint side: a constraint carries the
/// prefix-matching state of one generation, so handing it to a second
/// `constrained` call is refused rather than silently shared.
#[test]
fn alc_nn_moved_constraint_cannot_be_reused() {
    let lua = nn_vm();
    let err: String = eval_with_handle(
        &lua,
        r#"
        local c = alc.nn.constraint.stop_tokens({ 0 })
        alc.nn.sampler.constrained(alc.nn.sampler.greedy(), c)

        local ok, err = pcall(alc.nn.sampler.constrained, alc.nn.sampler.greedy(), c)
        assert(not ok, "a moved-from constraint must be refused")
        return tostring(err)
    "#,
        "moved constraint",
    );
    assert!(err.contains("moved"), "unexpected error: {err}");
}

/// A Lua function is a sampler. The callback here reimplements greedy
/// through `logits:top(1)`, so its stream must match
/// `alc.nn.sampler.greedy()` token for token — which is what makes this
/// an assertion about the bridge rather than about the model.
#[test]
fn alc_nn_lua_callback_sampler_drives_a_loop() {
    let lua = nn_vm();
    let (custom, greedy): (Vec<u32>, Vec<u32>) = eval_with_handle(
        &lua,
        r#"
        local function stream(sampler)
            local session = h:generate_session({ 1, 2, 3 })
            local out = {}
            for i = 1, 4 do
                out[i] = sampler:sample(session:next_logits())
                session:append(out[i])
            end
            return out
        end

        local custom = alc.nn.sampler.lua(function(logits)
            assert(logits:vocab() == 64, "the callback sees the full row")
            return logits:top(1)[1].id
        end)
        return stream(custom), stream(alc.nn.sampler.greedy())
    "#,
        "lua callback sampler",
    );
    assert_eq!(custom.len(), 4);
    assert_eq!(
        custom, greedy,
        "a top(1) callback must reproduce the greedy stream"
    );
}

/// The callback's answer is checked, never absorbed: a fractional
/// number, an out-of-range id and a non-number are all caller bugs that
/// surface at the call site instead of as a stray token in the output.
#[test]
fn alc_nn_lua_callback_bad_return_values_error() {
    let lua = nn_vm();
    let errors: Vec<String> = eval_with_handle(
        &lua,
        r#"
        local session = h:generate_session({ 1, 2, 3 })
        local logits = session:next_logits()

        local function fails(fn)
            local s = alc.nn.sampler.lua(fn)
            local ok, err = pcall(function() return s:sample(logits) end)
            assert(not ok, "callback must be rejected")
            return tostring(err)
        end

        return {
            fails(function() return 1.5 end),
            fails(function() return 64 end),     -- tiny vocab is 64: one past the end
            fails(function() return -1 end),
            fails(function() return "seven" end),
            fails(function() return nil end),
        }
    "#,
        "lua callback validation",
    );
    assert_eq!(errors.len(), 5);
    assert!(
        errors[0].contains("integer token id"),
        "fractional: {}",
        errors[0]
    );
    assert!(
        errors[1].contains("outside the vocabulary"),
        "out of range: {}",
        errors[1]
    );
    assert!(
        errors[2].contains("outside the vocabulary"),
        "negative: {}",
        errors[2]
    );
    for err in &errors[3..] {
        assert!(err.contains("integer token id"), "non-number: {err}");
    }
}

/// `logits:top(n)` ranks descending and agrees with `logits:argmax()`,
/// which in turn agrees with `alc.nn.sampler.greedy()` — the three
/// answers are the same question asked from three places, and a
/// disagreement would make a hand-written Lua sampler subtly differ from
/// the Rust one it is meant to replace.
#[test]
fn alc_nn_logits_top_and_argmax_agree_with_greedy() {
    let lua = nn_vm();
    let checked: usize = eval_with_handle(
        &lua,
        r#"
        local session = h:generate_session({ 1, 2, 3 })
        local logits = session:next_logits()

        local top = logits:top(5)
        assert(#top == 5, "top(5) must return five entries, got " .. #top)
        for i = 2, #top do
            assert(top[i - 1].value >= top[i].value,
                "top() is not descending at " .. i)
        end
        assert(logits:argmax() == top[1].id, "argmax must be the top-ranked id")
        assert(alc.nn.sampler.greedy():sample(logits) == logits:argmax(),
            "greedy must pick the argmax")

        assert(#logits:top(64) == 64, "top(vocab) must return the whole ranking")

        -- Out-of-band n errors rather than clamping: a shortened list
        -- would hide the caller's off-by-one until it indexed past the end.
        assert(not pcall(function() return logits:top(0) end), "top(0) must error")
        assert(not pcall(function() return logits:top(65) end), "top(vocab + 1) must error")

        return #top
    "#,
        "logits ranking accessors",
    );
    assert_eq!(checked, 5);
}

/// A regex constraint composed into the loop keeps every drawn token on
/// the pattern. The vocabulary is supplied as a table (a hand-rolled
/// tokenizer, one entry per model token id) so the test needs no
/// tokenizer download, and a saturated pattern fails loudly rather than
/// emitting an off-pattern token.
#[test]
fn alc_nn_regex_constraint_shapes_the_generated_tokens() {
    let lua = nn_vm();
    let drawn: Vec<u32> = eval_with_handle(
        &lua,
        r#"
        -- ids 0-9 are the digits, id 10 is the separator, and every
        -- remaining id spells a letter no digit pattern can accept.
        -- The table must cover the model's whole vocabulary: the
        -- constraint reasons about the ids it was given, so a short
        -- table would leave the tail of the row unconstrained.
        local vocab = {}
        for id = 0, 9 do vocab[id + 1] = tostring(id) end
        vocab[11] = "-"
        for id = 11, 63 do vocab[id + 1] = "z" end

        local s = alc.nn.sampler.constrained(
            alc.nn.sampler.greedy(),
            alc.nn.constraint.regex([[\d{3}-\d{4}]], vocab)
        )

        local session = h:generate_session({ 1, 2, 3 })
        local out = {}
        for i = 1, 8 do
            out[i] = s:sample(session:next_logits())
            session:append(out[i])
        end
        for i = 1, 3 do assert(out[i] < 10, "position " .. i .. " must be a digit") end
        assert(out[4] == 10, "position 4 must be the separator, got " .. out[4])
        for i = 5, 8 do assert(out[i] < 10, "position " .. i .. " must be a digit") end
        assert(s:is_done(), "a complete match must be terminal")

        local ok = pcall(function() return s:sample(session:next_logits()) end)
        assert(not ok, "a saturated pattern must error rather than emit")

        return out
    "#,
        "regex-constrained generation",
    );
    assert_eq!(drawn.len(), 8);
    assert_eq!(drawn[3], 10);
    assert!(drawn[..3].iter().all(|id| *id < 10));
    assert!(drawn[4..].iter().all(|id| *id < 10));
}

/// `reset` returns a constrained sampler to its pre-generation state so
/// one sampler can drive several generations — the prefix is dropped and
/// the terminal flag with it.
#[test]
fn alc_nn_constrained_sampler_is_reusable_after_reset() {
    let lua = nn_vm();
    let (first, second): (u32, u32) = eval_with_handle(
        &lua,
        r#"
        local probe = h:generate_session({ 1, 2, 3 })
        local stop = alc.nn.sampler.greedy():sample(probe:next_logits())

        local s = alc.nn.sampler.constrained(
            alc.nn.sampler.greedy(),
            alc.nn.constraint.stop_tokens({ stop })
        )

        local a = h:generate_session({ 1, 2, 3 })
        local first = s:sample(a:next_logits())
        assert(s:is_done(), "the stop token must terminate the first generation")

        s:reset()
        assert(not s:is_done(), "reset must clear the terminal state")

        local b = h:generate_session({ 1, 2, 3 })
        local second = s:sample(b:next_logits())
        assert(s:is_done(), "the reused sampler must still detect the stop token")
        return first, second
    "#,
        "reset then reuse",
    );
    assert_eq!(
        first, second,
        "the same prompt through the same greedy sampler must repeat"
    );
}

/// The allow-list constraint is the one-line legality guarantee: a noisy
/// temperature sampler under `constraint.allow_list` draws only listed
/// ids, and the chain is rebuilt per decision (the composition consumes
/// both handles, so that is the intended shape rather than a
/// workaround). An empty list is refused at construction, where the
/// caller's legality computation is, instead of at the draw.
#[test]
fn alc_nn_allow_list_constraint_confines_the_generated_tokens() {
    let lua = nn_vm();
    let (drawn, err): (Vec<u32>, String) = eval_with_handle(
        &lua,
        r#"
        -- An "illegal" argmax: whatever greedy would draw first is left
        -- out of the legal set, so a mask that stopped binding shows up
        -- as an out-of-set token rather than as a coincidence.
        local probe = h:generate_session({ 1, 2, 3 })
        local greedy_pick = alc.nn.sampler.greedy():sample(probe:next_logits())
        local legal, seen = {}, {}
        for id = 0, 63, 7 do
            if id ~= greedy_pick then
                legal[#legal + 1] = id
                seen[id] = true
            end
        end

        local session = h:generate_session({ 1, 2, 3 })
        local out = {}
        for turn = 1, 6 do
            -- One fresh chain per decision, seed derived from the turn.
            local s = alc.nn.sampler.constrained(
                alc.nn.sampler.temperature(1.2, 1000 + turn),
                alc.nn.constraint.allow_list(legal)
            )
            assert(not s:is_done(), "an allow list never terminates on its own")
            local id = s:sample(session:next_logits())
            assert(seen[id], "turn " .. turn .. " drew illegal token " .. tostring(id))
            assert(not s:is_done(), "an allow list must stay non-terminal after a draw")
            session:append(id)
            out[turn] = id
        end

        local ok, err = pcall(alc.nn.constraint.allow_list, {})
        assert(not ok, "an empty allow list must be rejected at construction")
        return out, tostring(err)
    "#,
        "allow-list constrained generation",
    );
    assert_eq!(drawn.len(), 6);
    assert!(
        drawn.iter().all(|id| *id % 7 == 0 && *id < 64),
        "illegal token in the stream: {drawn:?}"
    );
    assert!(
        err.contains("allow_list") && err.contains("empty"),
        "unexpected error: {err}"
    );
}

/// The JSON-schema constraint is reachable from Lua and rejects a schema
/// it cannot translate at construction — before a single token is
/// generated, which is the whole point of compiling the pattern up
/// front.
#[test]
fn alc_nn_json_schema_constraint_builds_and_rejects_bad_schemas() {
    let lua = nn_vm();
    let err: String = eval_with_handle(
        &lua,
        r#"
        local vocab = { "{", "}", "\"", "a", ":", "1", "," }

        -- A translatable schema composes into a sampler like any other
        -- constraint.
        local c = alc.nn.constraint.json_schema(
            { type = "object", properties = { a = { type = "integer" } }, required = { "a" } },
            vocab
        )
        local s = alc.nn.sampler.constrained(alc.nn.sampler.greedy(), c)
        assert(not s:is_done(), "an empty prefix is not a complete document")

        local ok, err = pcall(alc.nn.constraint.json_schema, { type = "widget" }, vocab)
        assert(not ok, "an untranslatable schema must be rejected at construction")
        return tostring(err)
    "#,
        "json schema constraint",
    );
    assert!(
        err.contains("json_schema") && err.contains("widget"),
        "unexpected error: {err}"
    );
}

// ─── end-to-end chat generation (Sampler plan Step 7) ────────────────

/// The whole chat surface in one pass: `chat_prompt` renders the turns,
/// `tokenize` turns them into ids, `generate_session` decodes with a
/// constrained sampler driving the loop, and `detokenize` turns the
/// drawn ids back into text. Every earlier test covers one hop; this one
/// exists to catch a break *between* hops (a shape that only two
/// adjacent layers agree on, an entry point that never got registered on
/// the same table), which none of them can see.
///
/// Nothing is claimed about the text itself. The model is random-init,
/// so its output carries no meaning — the assertion is that the pipeline
/// runs to completion and yields a Lua string.
///
/// # Fixture pairing
///
/// The tokenizer fixture (a 6-entry WordLevel vocabulary) and the model
/// fixture (`tiny`, a random-init Llama with a 64-token vocabulary) are
/// unrelated, exactly as a real preset pair would not be. That is
/// tolerable in both directions here:
///
/// - prompt ids land in `0..=5`, well inside the model's vocabulary, so
///   the prompt needs no clamping to be a legal session input;
/// - generated ids mostly fall outside the *tokenizer's* vocabulary, and
///   `decode` drops ids it has no surface form for (documented on
///   `HfTokenizer::vocab_strings`), so those contribute nothing to the
///   decoded string rather than failing it.
///
/// Making the two fixtures agree would mean shipping a real tokenizer,
/// which is a download; wiring is what this test is for.
#[test]
fn alc_nn_chat_roundtrip_returns_assistant_string() {
    let (lua, _tmp) = tokenizer_vm();
    let (text, steps): (String, usize) = lua
        .load(
            r#"
            local preset = "gpt2"
            local prompt = alc.nn.chat_prompt(preset, {
                { role = "system", content = "You are terse." },
                { role = "user", content = "Hi." },
            })
            local tokens = alc.nn.tokenize(preset, prompt)
            assert(#tokens > 0, "the rendered prompt must tokenize to at least one id")

            local h = alc.nn.preset.llama("tiny", { device = "cpu", dtype = "f32" })
            for i, id in ipairs(tokens) do
                assert(id < h:vocab(),
                    "prompt id at " .. i .. " (" .. id .. ") is outside the model vocabulary")
            end

            local session = h:generate_session(tokens)
            local s = alc.nn.sampler.constrained(
                alc.nn.sampler.greedy(),
                -- Any in-vocabulary id serves as the stop token: greedy
                -- over random weights gives no guarantee of drawing it,
                -- so the step budget below is the load-bearing terminator
                -- and the constraint is here to prove it composes.
                alc.nn.constraint.stop_tokens({ 0 })
            )

            -- The tiny variant's context window is 16 positions and the
            -- prompt already occupies part of it, so the budget is what
            -- is left rather than a round number: overrunning it is a
            -- rope/cache error, not a generation that trails off.
            local budget = h:ctx() - #tokens
            assert(budget > 0, "the prompt must leave room to generate")

            local generated = {}
            while not s:is_done() and #generated < budget do
                local id = s:sample(session:next_logits())
                session:append(id)
                generated[#generated + 1] = id
            end
            assert(session:position() <= h:ctx(), "the loop ran past the context window")

            return alc.nn.detokenize(preset, generated), #generated
        "#,
        )
        .eval()
        .expect("chat prompt -> tokenize -> generate -> detokenize");

    // `text` being a String is the claim; its contents are the model's
    // business and the model is random.
    let _ = text;
    assert!(
        (1..=9).contains(&steps),
        "the loop must draw at least one token and stop inside the budget, got {steps}"
    );
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
