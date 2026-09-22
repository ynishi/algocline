#![cfg(feature = "nn")]
//! End-to-end smoke for the checkpoint → metric loop a level-sweep
//! learner is built on, driven the same way `nn_gate_smoke.rs` drives
//! the gate: a production-shaped Lua VM over a tempdir, one embedded
//! script evaluated against it, assertions on the returned table.
//!
//! The domain is the `pick` fixture under `tests/lua/fixtures/`.
//!
//! What it fences, in one training run:
//!
//! 1. `alc.nn.trainer.run_full_ft` fires the `opts.on_ckpt` hook at
//!    every `ckpt_every` boundary (a hook that never fires would leave
//!    every downstream assertion vacuously true, so the fire count is
//!    asserted first);
//! 2. `alc.nn.card.load_ckpt` turns the mid-run `info.ckpt_path` into a
//!    live handle from inside that hook — while the trainer still holds
//!    the model mutex and the dataset lock;
//! 3. a Lua metric (`pick.legal_entropy`) consumes that handle and
//!    returns a number.
//!
//! Step 3 is where the three pieces meet: the handle is `NnHandle`
//! **userdata**, and `pick.require_handle` accepts it by capability
//! (`generate_session` is callable) rather than by Lua type. A bare
//! spec cannot build userdata, so this is the only place the userdata
//! leg of that guard is exercised — a regression there surfaces here as
//! a loud `pick.require_handle: userdata has no generate_session …`
//! rather than a silent skip.
//!
//! What it deliberately does not fence is the *value* of the metric
//! beyond its mathematical range. Four training steps do not move a
//! from-scratch model into any particular entropy, and asserting a
//! threshold would make the test a function of the training budget
//! rather than of the code under test.
//!
//! Every Card, safetensors bundle and rotating checkpoint the run
//! writes lands in the per-test tempdir, so the developer's
//! `~/.algocline` is untouched.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

/// Path to `tests/lua/fixtures/`, resolved via `CARGO_MANIFEST_DIR` so
/// the test does not depend on the process CWD.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("lua")
        .join("fixtures")
}

/// Build a production-shaped VM with the fixtures directory on
/// `package.path`, mirroring `nn_gate_smoke.rs::fixture_vm`.
///
/// The tempdir is returned alongside the VM: dropping it mid-test would
/// delete the checkpoints the hook loads back.
fn fixture_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("nn ckpt-hook tempdir");
    let root: PathBuf = tmp.path().to_path_buf();

    // Live sender required for `alc.llm` registration; the receiver is
    // dropped — this test never sends an LLM request.
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
    let path_prefix = format!("{}/?/init.lua;", fixtures_dir().display());
    lua.load("local prefix = ... package.path = prefix .. package.path")
        .set_name("@fixture_package_path")
        .call::<()>(path_prefix)
        .expect("extend package.path");

    (lua, tmp)
}

/// Embedded Lua driver. Reads the `SMOKE` config table (set from Rust
/// below), trains a from-scratch gpt2-tiny on a fixture corpus, and
/// evaluates `pick.legal_entropy` from inside the checkpoint hook.
const SCRIPT: &str = r#"
local pick = require("pick")
local VOCAB = pick.vocab()

local handle = alc.nn.preset.gpt2("tiny", {
    device = "cpu",
    dtype = "f32",
    pretrained = false,
})
local ctx_len = handle:ctx()
local model_vocab = handle:vocab()
assert(
    VOCAB.size <= model_vocab,
    string.format("alphabet of %d chars exceeds model vocab %d", VOCAB.size, model_vocab)
)

local base_rows = pick.build_corpus(pick.teacher, {
    ctx_len = ctx_len,
    episodes = SMOKE.episodes,
    seed = SMOKE.seed,
    pad_id = VOCAB.pad_id,
})

-- The trainer asks the dataset for `steps * batch` samples and the
-- dataset does not wrap, so the corpus is repeated until it can answer.
local need = SMOKE.steps * SMOKE.batch
local rows = {}
while #rows < need do
    for _, row in ipairs(base_rows) do
        rows[#rows + 1] = row
    end
end

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = SMOKE.batch,
    ctx_len = ctx_len,
    shuffle = false,
    pad_id = VOCAB.pad_id,
})

-- Prompt set for the metric: the fixed probe states, which cover the
-- turn and score-gap branches and have 5 / 4 / 3 legal values.
local prompt_set = {}
local states = pick.check_states()
for i = 1, math.min(SMOKE.prompts, #states) do
    prompt_set[i] = states[i]
end
assert(#prompt_set > 0, "prompt_set is empty")

-- ─── The loop under test ────────────────────────────────────────────

local fires = 0
local steps_seen = {}
local values = {}

local card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
    lr = SMOKE.lr,
    batch = SMOKE.batch,
    steps = SMOKE.steps,
    warmup = 0,
    schedule = "Constant",
    ckpt_every = SMOKE.ckpt_every,
    -- Keep every rotating checkpoint; the hook loads the file it is
    -- handed, and rotation during the run would race that read.
    ckpt_keep = SMOKE.steps + 1,
    name = "nn-ckpt-hook-e2e",
    on_ckpt = function(info)
        fires = fires + 1
        steps_seen[#steps_seen + 1] = info.step

        -- Cardless load of the mid-run checkpoint. Runs while the
        -- trainer holds the model mutex and the dataset lock, so it must
        -- not touch the training handle (see load_ckpt docs).
        local ckpt = alc.nn.card.load_ckpt(info.ckpt_path, { arch = "gpt2-tiny" })
        assert(
            type(ckpt) == "userdata",
            "load_ckpt must return an NnHandle userdata, got " .. type(ckpt)
        )

        local sum = 0
        for _, state in ipairs(prompt_set) do
            local h = pick.legal_entropy(ckpt, state, 1.0)
            assert(type(h) == "number", "legal_entropy must return a number, got " .. type(h))
            sum = sum + h
        end
        values[#values + 1] = sum / #prompt_set
        return "continue"
    end,
})

return {
    card_id = card_id,
    fires = fires,
    steps_seen = table.concat(steps_seen, ","),
    values = values,
    prompts = #prompt_set,
    rows = #rows,
    ctx_len = ctx_len,
}
"#;

/// Fields the driver returns that this test asserts on.
struct SmokeOut {
    card_id: String,
    fires: i64,
    steps_seen: String,
    values: Vec<f64>,
    prompts: i64,
    rows: i64,
    ctx_len: i64,
}

/// Run the driver with a smoke-sized budget and extract the returned
/// table into owned Rust values.
///
/// The extraction happens here rather than in the test body because the
/// returned `mlua::Table` borrows into the VM built above; handing it
/// back would leave the caller reading through a dropped VM.
fn run_ckpt_hook_smoke() -> SmokeOut {
    let (lua, _tmp) = fixture_vm();

    // Smoke budget: 4 steps at batch 2, checkpointing every 2 → exactly
    // two hook fires. Small enough that the two extra model builds the
    // hook performs (one per fire) stay negligible on CPU.
    let cfg = lua.create_table().expect("create SMOKE table");
    cfg.set("episodes", 4).expect("set episodes");
    cfg.set("steps", 4).expect("set steps");
    cfg.set("batch", 2).expect("set batch");
    cfg.set("ckpt_every", 2).expect("set ckpt_every");
    cfg.set("lr", 3e-3).expect("set lr");
    cfg.set("seed", 20260801).expect("set seed");
    cfg.set("prompts", 3).expect("set prompts");
    lua.globals().set("SMOKE", cfg).expect("set SMOKE global");

    let out: mlua::Table = lua
        .load(SCRIPT)
        .set_name("@nn_ckpt_hook_e2e")
        .eval()
        .expect("nn ckpt-hook e2e script");

    SmokeOut {
        card_id: out.get("card_id").expect("card_id"),
        fires: out.get("fires").expect("fires"),
        steps_seen: out.get("steps_seen").expect("steps_seen"),
        values: out.get("values").expect("values"),
        prompts: out.get("prompts").expect("prompts"),
        rows: out.get("rows").expect("rows"),
        ctx_len: out.get("ctx_len").expect("ctx_len"),
    }
}

#[test]
fn on_ckpt_load_ckpt_metric_e2e() {
    let out = run_ckpt_hook_smoke();

    eprintln!(
        "[nn-ckpt-hook] card_id={} fires={} steps=[{}] values={:?} rows={} ctx_len={}",
        out.card_id, out.fires, out.steps_seen, out.values, out.rows, out.ctx_len,
    );

    // (a) the hook actually ran. Asserted before anything else: a
    // silent no-op hook would make every later assertion vacuous.
    assert!(
        out.fires > 0,
        "on_ckpt must fire at least once (steps=4, ckpt_every=2); a zero count means \
         the hook was never wired through run_full_ft"
    );
    assert_eq!(
        out.fires, 2,
        "on_ckpt must fire once per ckpt_every boundary (steps=4, ckpt_every=2); \
         fired at steps [{}]",
        out.steps_seen
    );
    assert_eq!(
        out.steps_seen, "2,4",
        "the hook must see the checkpoint boundaries in order"
    );

    // (b) every fire produced a metric reading through the userdata
    // handle `load_ckpt` returned.
    assert_eq!(
        out.values.len() as i64,
        out.fires,
        "each fire must contribute exactly one metric reading"
    );
    assert_eq!(
        out.prompts, 3,
        "the prompt set must reach the metric intact"
    );
    for (i, v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "metric reading {i} must be finite, got {v}");
        // Normalised entropy lives in [0, 1]. The bounds are the metric's
        // definition, not a training-budget expectation — four steps do
        // not pin the value any further.
        assert!(
            *v >= 0.0 && *v <= 1.0 + 1e-9,
            "metric reading {i} must sit in [0, 1], got {v}"
        );
    }

    // (c) the run still finished and persisted its Card.
    assert!(
        !out.card_id.is_empty(),
        "run_full_ft must register a Card after a non-break hook run"
    );
    assert_eq!(
        out.ctx_len, 16,
        "the gpt2 tiny preset context window is the row width the encoding is sized against"
    );
    assert!(
        out.rows >= 4 * 2,
        "the corpus must cover steps x batch rows, got {}",
        out.rows
    );
}
