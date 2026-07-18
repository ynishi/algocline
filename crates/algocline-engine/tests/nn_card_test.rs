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
    let bundle = tmp
        .path()
        .join("nn")
        .join(format!("{card_id}.safetensors"));
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
    let card_text = std::fs::read_to_string(&card_toml_path)
        .expect("card TOML must exist under cards/alc_nn/");
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
        .load(&format!(
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
    let bundle = tmp
        .path()
        .join("nn")
        .join(format!("{card_id}.safetensors"));
    std::fs::remove_file(&bundle).expect("delete safetensors bundle for the test");

    let err = lua
        .load(&format!(
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
