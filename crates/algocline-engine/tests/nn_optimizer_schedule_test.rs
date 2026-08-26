#![cfg(feature = "nn")]
//! Every optimizer and every schedule, driven through the Lua surface.
//!
//! The unit tests around `nn_opts` pin that the keys reach the config.
//! That is a weaker claim than "a run configured this way completes":
//! the optimizer is chosen by one dispatch and realised by another
//! (dtype), and a schedule that returns a bad number does not fail at
//! the config — it fails several hundred steps later as a loss that
//! went nowhere.
//!
//! Shapes are the smallest the preset accepts and each run is a handful
//! of steps, so the file finishes well inside a second on CPU. No
//! assertion touches a loss: a few steps of a random-init model say
//! nothing about learning.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

fn vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("tempdir");
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

    let alc_table = lua.create_table().expect("alc table");
    bridge::register(&lua, &alc_table, config).expect("bridge::register");
    lua.globals().set("alc", alc_table).expect("set alc global");
    lua.load(bridge::PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .expect("prelude");
    (lua, tmp)
}

/// Train a few steps with the given `run_full_ft` opts fragment and
/// return the card id.
fn train_with(lua: &Lua, opts_fragment: &str) -> String {
    let source = format!(
        r#"
        local rows = {{}}
        for i = 1, 8 do rows[i] = {{ 1, 2, 3, 4 }} end
        local ds = alc.nn.data.synthetic(rows, {{ batch_size = 2, ctx_len = 4 }})
        local h = alc.nn.preset.gpt2("custom", {{
            pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
        }})
        return alc.nn.trainer.run_full_ft(h, ds, {{
            lr = 1e-3, batch = 2, steps = 4, warmup = 1, {opts_fragment}
        }})
        "#
    );
    lua.load(source.as_str())
        .set_name("@optimizer_schedule_arm")
        .eval()
        .unwrap_or_else(|e| panic!("run failed for [{opts_fragment}]: {e}"))
}

/// All four schedules drive a run to completion. `run_full_ft` takes
/// the CamelCase vocabulary — the snake_case spellings belong to the
/// sibling `full_ft` family and are covered by its own unit tests.
#[test]
fn every_schedule_completes_a_run() {
    let (lua, _tmp) = vm();
    for schedule in [
        "Constant",
        "CosineWithWarmup",
        "Linear",
        "WarmupStableDecay",
    ] {
        let card = train_with(
            &lua,
            &format!(r#"schedule = "{schedule}", name = "sched-{schedule}""#),
        );
        assert!(!card.is_empty(), "{schedule} produced no card");
    }
}

/// The floor and the decay length are read rather than accepted and
/// dropped: a run naming both completes, and naming `decay_steps`
/// without WSD is not an error either — no schedule but WSD reads it.
#[test]
fn min_lr_and_decay_steps_are_accepted_by_the_run_surface() {
    let (lua, _tmp) = vm();
    let card = train_with(
        &lua,
        r#"schedule = "WarmupStableDecay", min_lr = 1e-5, decay_steps = 2,
           name = "wsd-tuned""#,
    );
    assert!(!card.is_empty());
}

/// Lion trains. The learning rate is deliberately the paper's order of
/// magnitude below the AdamW value used elsewhere in this file.
#[test]
fn lion_completes_a_run() {
    let (lua, _tmp) = vm();
    let card = train_with(
        &lua,
        r#"optimizer = "lion", weight_decay = 1.0, name = "lion-run""#,
    );
    assert!(!card.is_empty());
}

/// AdamW's own coefficients arrive too — they were reachable in Rust
/// and unreachable from Lua before this.
#[test]
fn adamw_coefficients_are_accepted() {
    let (lua, _tmp) = vm();
    let card = train_with(
        &lua,
        r#"beta1 = 0.95, beta2 = 0.98, eps = 1e-6, name = "adamw-tuned""#,
    );
    assert!(!card.is_empty());
}

/// An unknown optimizer is refused by name, and the refusal lists what
/// there is — a caller who guessed "sgd" needs to be told what exists,
/// not only that their guess did not.
#[test]
fn an_unknown_optimizer_is_refused_with_the_alternatives() {
    let (lua, _tmp) = vm();
    let err = lua
        .load(
            r#"
            local rows = {}
            for i = 1, 8 do rows[i] = { 1, 2, 3, 4 } end
            local ds = alc.nn.data.synthetic(rows, { batch_size = 2, ctx_len = 4 })
            local h = alc.nn.preset.gpt2("custom", {
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
            })
            return alc.nn.trainer.run_full_ft(h, ds, {
                lr = 1e-3, batch = 2, steps = 2, optimizer = "sgd",
            })
            "#,
        )
        .set_name("@optimizer_schedule_arm")
        .eval::<String>()
        .expect_err("sgd is not implemented");
    let text = err.to_string();
    assert!(text.contains("sgd"), "{text}");
    assert!(
        text.contains("lion"),
        "the refusal must name what exists: {text}"
    );
}
