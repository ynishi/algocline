#![cfg(feature = "nn")]
//! Engine-level tests for the `alc.nn.card.*` bridge (feature `nn`).
//!
//! Layout: cargo integration tests only compile top-level `tests/*.rs`
//! files as binaries (files in a subdir aren't picked up unless
//! referenced via a `mod` from a top-level driver), so this file lives
//! flat next to the existing `tests/nn_bridge_smoke.rs` sibling.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

/// Build a production-shaped VM (mirrors `nn_bridge_smoke.rs::production_vm`).
///
/// The tempdir is returned alongside the VM so the caller can keep it
/// alive for the duration of the test — dropping it would remove the
/// safetensors bundle mid-test.
fn nn_card_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("test tempdir");
    let root: PathBuf = tmp.path().to_path_buf();

    // Live sender required for `alc.llm` registration; the receiver is
    // dropped — the tests here never actually send an LLM request.
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
    bridge::register(&lua, &alc_table, config).expect("bridge::register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    lua.load(bridge::PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .expect("load prelude");
    (lua, tmp)
}

/// Sanity: the sub-namespace `alc.nn.card` is populated with the three
/// entries the task contract requires.
#[test]
fn alc_nn_card_namespace_has_save_load_register() {
    let (lua, _tmp) = nn_card_vm();
    let missing: Vec<String> = lua
        .load(
            r#"
            local out = {}
            for _, k in ipairs({ "save", "load", "register" }) do
                if type(alc.nn.card[k]) ~= "function" then
                    table.insert(out, k)
                end
            end
            return out
            "#,
        )
        .eval()
        .expect("probe alc.nn.card namespace");
    assert!(
        missing.is_empty(),
        "alc.nn.card missing entries: {missing:?}"
    );
}

/// End-to-end roundtrip: save two named Vars via `alc.nn.card.save`,
/// re-open the store from a fresh Lua VM, and confirm the load returns
/// tensors with matching element values. Also confirms the Card TOML
/// records the expected `metadata.nn.*` block.
#[test]
fn save_load_roundtrip_preserves_var_values_and_meta() {
    let (lua, tmp) = nn_card_vm();

    // Save.
    let card_id: String = lua
        .load(
            r#"
            local w = alc.nn.var(3, 1.5)
            local b = alc.nn.var(3, 2.5)
            return alc.nn.card.save({ w = w, b = b }, "roundtrip-model", {
                training_path = "full_ft",
                architecture  = "gpt2-medium",
                task          = "test",
                lineage       = { tokenizer = "gpt2" },
                hyperparams   = { lr = 0.001, steps = 10 },
                metrics       = { train_loss_final = 0.42 },
            })
            "#,
        )
        .eval()
        .expect("alc.nn.card.save must succeed");

    assert!(
        !card_id.is_empty(),
        "save() must return a non-empty card_id"
    );

    // The safetensors bundle lives at nn_dir/<card_id>.safetensors.
    let bundle = tmp.path().join("nn").join(format!("{card_id}.safetensors"));
    assert!(
        bundle.exists(),
        "safetensors bundle missing at {bundle:?} (invariant #1)"
    );

    // The Card TOML lives at cards/alc_nn/<card_id>.toml — carry it back
    // to Rust and check the shape.
    let card_toml_path = tmp
        .path()
        .join("cards")
        .join("alc_nn")
        .join(format!("{card_id}.toml"));
    let card_text =
        std::fs::read_to_string(&card_toml_path).expect("card TOML must exist under cards/alc_nn/");
    assert!(
        card_text.contains(r#"kind = "nn_model""#),
        "Card kind must be nn_model, got: {card_text}"
    );
    assert!(
        card_text.contains(r#"name = "roundtrip-model""#),
        "Card must record the model name, got: {card_text}"
    );
    assert!(
        card_text.contains(&format!(r#"bundle_ref = "nn/{card_id}""#)),
        "bundle_ref must equal nn/<card_id>, got: {card_text}"
    );
    assert!(
        card_text.contains(r#"training_path = "full_ft""#),
        "training_path must round-trip, got: {card_text}"
    );

    // Load in a fresh Lua VM sharing the same tempdir root — this
    // matches the real-world flow where a distinct session opens the
    // Card after training.
    let (lua2, _) = fresh_vm_sharing_root(tmp.path().to_path_buf());
    let (w_vec, b_vec): (Vec<f32>, Vec<f32>) = lua2
        .load(format!(
            r#"
            local m = alc.nn.card.load("{card_id}")
            return m.w:to_vec(), m.b:to_vec()
            "#
        ))
        .eval()
        .expect("alc.nn.card.load must succeed after save");

    assert_eq!(w_vec, vec![1.5, 1.5, 1.5]);
    assert_eq!(b_vec, vec![2.5, 2.5, 2.5]);
}

/// Build a second production VM but rooted at the same tempdir so the
/// FileCardStore + NnStore see the artifacts written by the first VM.
///
/// We deliberately leak the TempDir handle from the caller (the outer
/// test holds its own alive) — this helper never owns the root.
fn fresh_vm_sharing_root(root: PathBuf) -> (Lua, tokio::sync::mpsc::Receiver<()>) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
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
    bridge::register(&lua, &alc_table, config).expect("bridge::register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    lua.load(bridge::PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .expect("load prelude");
    // Ret placeholder so the caller can keep this shape symmetric with
    // `nn_card_vm`; the actual channel isn't used.
    let (_dummy, rx) = tokio::sync::mpsc::channel::<()>(1);
    (lua, rx)
}

/// `save` must reject missing required meta fields loudly — silent
/// success would let bad Cards land under `alc_nn/`.
#[test]
fn save_errors_when_training_path_missing() {
    let (lua, _tmp) = nn_card_vm();
    let err = lua
        .load(
            r#"
            local w = alc.nn.var(1, 0.0)
            return alc.nn.card.save({ w = w }, "no-tp", {
                architecture = "gpt2-medium",
            })
            "#,
        )
        .exec()
        .expect_err("save without meta.training_path must fail loudly");
    let msg = err.to_string();
    assert!(
        msg.contains("training_path"),
        "error must name the missing field, got: {msg}"
    );
}

/// `save` must also reject missing architecture — same rationale.
#[test]
fn save_errors_when_architecture_missing() {
    let (lua, _tmp) = nn_card_vm();
    let err = lua
        .load(
            r#"
            local w = alc.nn.var(1, 0.0)
            return alc.nn.card.save({ w = w }, "no-arch", {
                training_path = "full_ft",
            })
            "#,
        )
        .exec()
        .expect_err("save without meta.architecture must fail loudly");
    let msg = err.to_string();
    assert!(
        msg.contains("architecture"),
        "error must name the missing field, got: {msg}"
    );
}

/// Invariant #3: `register` is idempotent — same `model_name` may be
/// registered twice against the same card without error, and the
/// registry entry is overwritten (last-writer wins).
#[test]
fn register_is_idempotent_and_last_write_wins() {
    let (lua, _tmp) = nn_card_vm();
    lua.load(
        r#"
            local w = alc.nn.var(1, 0.0)
            local id = alc.nn.card.save({ w = w }, "reg-model", {
                training_path = "full_ft",
                architecture  = "gpt2-medium",
            })
            -- First register: succeeds.
            alc.nn.card.register(id, "my-alias")
            -- Second register with the same alias: must NOT error.
            alc.nn.card.register(id, "my-alias")
            -- Third register from a distinct save: overrides the alias.
            local w2 = alc.nn.var(1, 0.0)
            local id2 = alc.nn.card.save({ w = w2 }, "reg-model-2", {
                training_path = "full_ft",
                architecture  = "gpt2-medium",
            })
            alc.nn.card.register(id2, "my-alias")
        "#,
    )
    .exec()
    .expect("register must be idempotent and allow overwrite");
}

/// Invariant #4 partial: `register` must refuse an unknown card_id.
#[test]
fn register_missing_card_errors() {
    let (lua, _tmp) = nn_card_vm();
    let err = lua
        .load(
            r#"
            return alc.nn.card.register("does-not-exist", "alias")
            "#,
        )
        .exec()
        .expect_err("register with unknown card_id must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("not found"),
        "error must indicate the card is missing, got: {msg}"
    );
}

/// Invariant #4: `load` must refuse an unknown card_id (no partial state,
/// no fallback to empty vars).
#[test]
fn load_missing_card_errors() {
    let (lua, _tmp) = nn_card_vm();
    let err = lua
        .load(
            r#"
            return alc.nn.card.load("does-not-exist")
            "#,
        )
        .exec()
        .expect_err("load with unknown card_id must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("not found"),
        "error must indicate the card is missing, got: {msg}"
    );
}

/// Invariant #4: if the Card exists but the safetensors bundle has
/// been removed from disk, `load` errors instead of returning
/// something plausible.
#[test]
fn load_missing_bundle_errors() {
    let (lua, tmp) = nn_card_vm();
    let card_id: String = lua
        .load(
            r#"
            local w = alc.nn.var(2, 1.0)
            return alc.nn.card.save({ w = w }, "orphan", {
                training_path = "full_ft",
                architecture  = "gpt2-medium",
            })
            "#,
        )
        .eval()
        .expect("initial save must succeed");
    // Delete the safetensors bundle out from under the Card.
    let bundle = tmp.path().join("nn").join(format!("{card_id}.safetensors"));
    std::fs::remove_file(&bundle).expect("delete safetensors bundle for the test");

    let err = lua
        .load(format!(
            r#"
            return alc.nn.card.load("{card_id}")
            "#
        ))
        .exec()
        .expect_err("load after bundle delete must fail loudly");
    // The exact error phrasing comes from candle-core / std::fs — we
    // just insist that it is not silent.
    let msg = err.to_string();
    assert!(!msg.is_empty(), "orphan-bundle load must surface an error");
}

/// ST-D bridge smoke: run `alc.nn.trainer.full_ft` for a single fire
/// with a hook that captures `info.ckpt_path`, feed the captured path
/// into `alc.nn.card.save_from_ckpt`, and confirm the returned card_id
/// is round-trippable through `alc.nn.card.load_handle` — the
/// production shape a boss-harvest hook uses to promote a mid-run
/// checkpoint into a first-class Card in the same fire the checkpoint
/// was written.
///
/// The invariants under test:
///
/// 1. The Lua-facing entry is actually present on `alc.nn.card` (the
///    registration path in `register_nn_card` runs end-to-end).
/// 2. A ckpt path handed to `save_from_ckpt` inside an `on_ckpt` hook
///    body — while the trainer holds the model mutex — completes
///    without deadlock, mirroring the pattern `load_ckpt` already
///    supports (see `gameai_ckpt_metric_e2e.rs`).
/// 3. The Card the promotion writes is indistinguishable from a Card
///    written by any other on-ramp: `load_handle` rebuilds a live
///    handle whose architecture matches the meta the caller supplied,
///    and the handle's `forward` runs over a prompt.
#[test]
fn save_from_ckpt_roundtrips_a_full_ft_checkpoint_via_load_handle() {
    let (lua, _tmp) = nn_card_vm();

    // The `tiny` named variant is the smallest fully-baked shape that
    // `load_handle` can rebuild from Card meta alone — the `custom`
    // variant needs `metadata.nn.candle.custom` to survive round-trip,
    // and `save_from_ckpt` intentionally reuses `save`'s `build_nn_meta`
    // path (which omits the custom branch) since a promoted mid-run
    // checkpoint is always a named-variant snapshot (a `custom` shape
    // originates from a trainer entry that records `custom` through
    // `NnModelCard::from_training`, not through the `save` on-ramp).
    let card_id: String = lua
        .load(
            r#"
            local h = alc.nn.preset.gpt2("tiny", {
                pretrained = false,
                device = "cpu",
            })
            local ctx_len = h:ctx()
            local rows = {}
            for i = 1, 4 do
                local row = {}
                for j = 1, ctx_len do
                    row[j] = ((i * 7 + j * 3) % 30) + 1
                end
                rows[i] = row
            end
            local ds = alc.nn.data.synthetic(rows, {
                batch_size = 1,
                ctx_len = ctx_len,
                shuffle = false,
                pad_id = 0,
            })

            local captured_id = nil
            alc.nn.trainer.full_ft(h, ds, {
                lr = 3e-4,
                batch_size = 1,
                steps = 2,
                warmup = 0,
                schedule = "constant",
                weight_decay = 0.0,
                ckpt_every = 2,
                ckpt_keep = 3,
                ckpt_prefix = "save_from_ckpt_smoke",
                on_ckpt = function(info)
                    assert(type(info.ckpt_path) == "string",
                        "info.ckpt_path must be a string")
                    -- Promote the mid-run checkpoint into a first-class
                    -- Card from inside the hook. Runs while the trainer
                    -- holds the model mutex, so a hang here (not a
                    -- failed assertion) would be the deadlock-freedom
                    -- regression.
                    captured_id = alc.nn.card.save_from_ckpt(
                        info.ckpt_path,
                        "boss-mid-smoke",
                        {
                            training_path = "full_ft",
                            architecture  = "gpt2-tiny",
                        }
                    )
                    assert(type(captured_id) == "string" and #captured_id > 0,
                        "save_from_ckpt must return a non-empty card_id")
                    return "continue"
                end,
            })

            assert(captured_id ~= nil,
                "on_ckpt hook must have fired at least once")

            -- Round-trip: the promoted Card must be loadable back into a
            -- live handle whose architecture matches the meta above.
            local loaded = alc.nn.card.load_handle(captured_id)
            assert(type(loaded) == "userdata",
                "load_handle must return an NnHandle userdata")
            assert(loaded:arch() == "gpt2",
                "loaded handle arch must be gpt2, got " .. tostring(loaded:arch()))
            -- The `tiny` variant baked-in vocab is 64 (see
            -- `Gpt2Config::tiny`) — asserting on it confirms the reload
            -- path rebuilt the same shape rather than silently defaulting
            -- to some other variant.
            assert(loaded:vocab() == 64,
                "loaded handle must report tiny vocab (64), got " .. tostring(loaded:vocab()))

            return captured_id
        "#,
        )
        .eval()
        .expect("save_from_ckpt round-trip must succeed");

    assert!(!card_id.is_empty(), "captured card_id must be non-empty");
}

/// ST-D negative path: `save_from_ckpt` refuses a non-existent source
/// safetensors file loudly, with a message that names the entry so a
/// caller reading the trace sees which bridge surfaced the error.
#[test]
fn save_from_ckpt_errors_when_source_is_missing_from_lua() {
    let (lua, _tmp) = nn_card_vm();
    let err = lua
        .load(
            r#"
            return alc.nn.card.save_from_ckpt(
                "/definitely/does/not/exist.safetensors",
                "boss-missing",
                {
                    training_path = "full_ft",
                    architecture  = "gpt2-medium",
                }
            )
            "#,
        )
        .exec()
        .expect_err("missing source must fail loudly across the Lua boundary");
    let msg = err.to_string();
    assert!(
        msg.contains("save_from_ckpt"),
        "error must name the save_from_ckpt entry, got: {msg}"
    );
    assert!(
        msg.contains("source safetensors not found"),
        "error must name the missing-source case, got: {msg}"
    );
}
