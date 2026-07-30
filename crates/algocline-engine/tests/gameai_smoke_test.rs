#![cfg(feature = "nn")]
//! End-to-end smoke for the GameAI card duel demo
//! (`examples/gameai/`), driven the same way `nn_smoke_test.rs` drives
//! the `nn_*_smoke.lua` examples: a production-shaped Lua VM over a
//! tempdir, one script evaluated against it, assertions on the returned
//! table.
//!
//! What it fences, in one training run:
//!
//! 1. the training loss descends below the uniform-model baseline
//!    (`ln(vocab)`), so gradients actually flowed;
//! 2. the decode gate returns a legal action for every probe state;
//! 3. two independent decode sessions over the same state agree.
//!
//! What it deliberately does not fence is the style compliance rate.
//! At 40 steps the model has barely moved, and asserting a compliance
//! threshold here would make the test a function of the training budget
//! rather than of the code under test — `examples/gameai/card_duel_scenario.lua`
//! is where compliance is measured.
//!
//! Every Card, safetensors bundle and alias written by the run lands in
//! the per-test tempdir, so the developer's `~/.algocline` is untouched.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

/// Path to the workspace-root `examples/gameai/` directory, resolved
/// via `CARGO_MANIFEST_DIR` so the test does not depend on the process
/// CWD.
fn gameai_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine crate parent (crates/)")
        .parent()
        .expect("workspace root")
        .join("examples")
        .join("gameai")
}

/// Build a production-shaped VM mirroring `nn_smoke_test.rs::smoke_vm`,
/// with `examples/gameai/` on `package.path` so the script can
/// `require("card_duel")` / `require("card_duel_npc")`.
///
/// The tempdir is returned alongside the VM: dropping it mid-test would
/// delete the safetensors bundle and Card TOML that
/// `alc.nn.trainer.run_full_ft` writes and that
/// `alc.nn.card.load_handle` reads back.
fn gameai_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("gameai smoke tempdir");
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
    bridge::register(&lua, &alc_table, config).expect("bridge::register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    lua.load(bridge::PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .expect("load prelude");

    // `lib_paths` only reaches forked child VMs, so the parent's
    // `package.path` is extended here instead.
    let path_prefix = format!("{}/?/init.lua;", gameai_dir().display());
    lua.load("local prefix = ... package.path = prefix .. package.path")
        .set_name("@gameai_package_path")
        .call::<()>(path_prefix)
        .expect("extend package.path");

    (lua, tmp)
}

/// Fields the training script returns that this test asserts on.
struct SmokeOut {
    ok: bool,
    card_id: String,
    rows: i64,
    ctx_len: i64,
    train_loss: f64,
    baseline_loss: f64,
    loss_descended: bool,
    decide_legal: bool,
    deterministic: bool,
    decisions: String,
}

/// Run `train_card_duel_npc.lua` with a smoke-sized budget and extract
/// the returned table into owned Rust values.
///
/// The extraction happens here rather than in the test body because the
/// returned `mlua::Table` borrows into the VM built above; handing it
/// back would leave the caller reading through a dropped VM.
fn run_gameai_smoke() -> SmokeOut {
    let (lua, _tmp) = gameai_vm();

    // Smoke budget: 40 steps at batch 16. The script grows the corpus to
    // cover `steps * batch` rows on its own, so `games` is only a floor.
    // Large enough for the loss to leave the uniform baseline on a
    // 2-layer / dim-32 model, small enough to stay well inside a CPU
    // test run.
    let ctx = lua.create_table().expect("create ctx table");
    ctx.set("games", 12).expect("set games");
    ctx.set("steps", 40).expect("set steps");
    ctx.set("batch", 16).expect("set batch");
    ctx.set("lr", 3e-3).expect("set lr");
    ctx.set("seed", 20260731).expect("set seed");
    ctx.set("name", "card-duel-npc-smoke").expect("set name");
    lua.globals().set("ctx", ctx).expect("set ctx global");

    let path = gameai_dir().join("train_card_duel_npc.lua");
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let out: mlua::Table = lua
        .load(src.as_str())
        .set_name("@train_card_duel_npc.lua")
        .eval()
        .unwrap_or_else(|e| panic!("execute {}: {e}", path.display()));

    SmokeOut {
        ok: out.get("ok").expect("ok"),
        card_id: out.get("card_id").expect("card_id"),
        rows: out.get("rows").expect("rows"),
        ctx_len: out.get("ctx_len").expect("ctx_len"),
        train_loss: out.get("train_loss").expect("train_loss"),
        baseline_loss: out.get("baseline_loss").expect("baseline_loss"),
        loss_descended: out.get("loss_descended").expect("loss_descended"),
        decide_legal: out.get("decide_legal").expect("decide_legal"),
        deterministic: out.get("deterministic").expect("deterministic"),
        decisions: out.get("decisions").expect("decisions"),
    }
}

#[test]
fn card_duel_npc_train_decode_gate_smoke() {
    let out = run_gameai_smoke();

    assert!(
        !out.card_id.is_empty(),
        "run_full_ft must register a Card; decisions = {}",
        out.decisions
    );
    assert!(
        out.rows >= 40 * 16,
        "the corpus must cover steps x batch rows, got {}",
        out.rows
    );
    assert_eq!(
        out.rows % 10,
        0,
        "every playout contributes 5 rounds x 2 seats, so rows must be a multiple of 10; got {}",
        out.rows
    );
    assert_eq!(
        out.ctx_len, 16,
        "the gpt2 tiny preset context window is the row width the encoding is sized against"
    );

    // (a) learning happened at all.
    assert!(
        out.loss_descended,
        "train_loss {} must fall below the uniform baseline {}",
        out.train_loss, out.baseline_loss
    );

    // (b) the decode gate never emits an illegal action.
    assert!(
        out.decide_legal,
        "every gated decode must return a legal action; decisions = {}",
        out.decisions
    );
    assert!(
        out.decisions.contains("legal=true"),
        "the decide summary must carry the legality flag; got {}",
        out.decisions
    );

    // (c) greedy decoding is reproducible across independent sessions.
    assert!(
        out.deterministic,
        "two independent decode sessions must agree; decisions = {}",
        out.decisions
    );

    assert!(
        out.ok,
        "training script reported failure; decisions = {}",
        out.decisions
    );
}
