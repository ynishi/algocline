#![cfg(feature = "nn")]
//! End-to-end smoke for the train → Card → gated decode loop, driven the
//! same way `nn_smoke_test.rs` drives the `nn_*_smoke.lua` examples: a
//! production-shaped Lua VM over a tempdir, one embedded script
//! evaluated against it, assertions on the returned table.
//!
//! The domain is the `pick` fixture under `tests/lua/fixtures/` — a
//! sequential-choice toy with a per-state legal set, a deterministic
//! teacher and a fixed-width encoding. It stands in for any caller that
//! clones a policy into a small model and decodes through a legal gate.
//!
//! What it fences, in one training run:
//!
//! 1. the training loss descends below the uniform-model baseline
//!    (`ln(vocab)`), so gradients actually flowed;
//! 2. the decode gate returns a legal value for every probe state, and
//!    the gated path reports whether it had to move off the argmax;
//! 3. two independent decode sessions over the same state agree.
//!
//! What it deliberately does not fence is how often the model agrees
//! with the teacher. At 40 steps the model has barely moved, and
//! asserting an agreement threshold would make the test a function of
//! the training budget rather than of the code under test.
//!
//! Every Card and safetensors bundle written by the run lands in the
//! per-test tempdir, so the developer's `~/.algocline` is untouched.

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

/// Build a production-shaped VM mirroring `nn_smoke_test.rs::smoke_vm`,
/// with the fixtures directory on `package.path` so the script can
/// `require("pick")`.
///
/// The tempdir is returned alongside the VM: dropping it mid-test would
/// delete the safetensors bundle and Card TOML that
/// `alc.nn.trainer.run_full_ft` writes and `alc.nn.card.load_handle`
/// reads back.
fn fixture_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("nn gate smoke tempdir");
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
    let path_prefix = format!("{}/?/init.lua;", fixtures_dir().display());
    lua.load("local prefix = ... package.path = prefix .. package.path")
        .set_name("@fixture_package_path")
        .call::<()>(path_prefix)
        .expect("extend package.path");

    (lua, tmp)
}

/// Embedded Lua driver. Reads the `SMOKE` config table (set from Rust
/// below), clones the fixture teacher into a from-scratch gpt2-tiny,
/// registers the Card, and decodes three probe states through the gate.
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

-- A synthetic dataset walks its rows once, so the episode count is
-- raised until the corpus covers `steps * batch` rows.
local episodes = math.max(SMOKE.episodes, math.ceil(SMOKE.steps * SMOKE.batch / pick.ROWS_PER_EPISODE))
local rows = pick.build_corpus(pick.teacher, {
    ctx_len = ctx_len,
    episodes = episodes,
    seed = SMOKE.seed,
    pad_id = VOCAB.pad_id,
})

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = SMOKE.batch,
    ctx_len = ctx_len,
    shuffle = true,
    pad_id = VOCAB.pad_id,
})

local card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
    lr = SMOKE.lr,
    batch = SMOKE.batch,
    steps = SMOKE.steps,
    warmup = 0,
    schedule = "Constant",
    name = SMOKE.name,
})
assert(type(card_id) == "string" and #card_id > 0, "run_full_ft returned no card_id")

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
assert(type(train_loss) == "number", "metadata.nn.metrics.train_loss missing from the Card")

-- Decode through a handle loaded back from the Card, not the training
-- handle: that is the path a consumer takes.
local loaded = alc.nn.card.load_handle(card_id)
assert(type(loaded) == "userdata", "load_handle must return an NnHandle userdata")

local decide_legal = true
local teacher_hits = 0
local reports = {}
local states = pick.check_states()
for i, state in ipairs(states) do
    local d = pick.decide(loaded, state)
    local legal = false
    for _, v in ipairs(pick.legal(state)) do
        if v == d.value then
            legal = true
        end
    end
    if not legal then
        decide_legal = false
    end
    if d.value == pick.teacher(state) then
        teacher_hits = teacher_hits + 1
    end
    reports[i] = string.format(
        "%s -> value=%d legal=%s raw_legal=%s gated=%s",
        pick.encode(state), d.value, tostring(legal), tostring(d.raw_legal), tostring(d.gated)
    )
end

local first = pick.decide(loaded, states[1])
local second = pick.decide(loaded, states[1])

return {
    ok = decide_legal and train_loss < baseline_loss,
    card_id = card_id,
    rows = #rows,
    ctx_len = ctx_len,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    decide_legal = decide_legal,
    teacher_hits = teacher_hits,
    deterministic = first.value == second.value,
    decisions = table.concat(reports, " | "),
}
"#;

/// Fields the driver returns that this test asserts on.
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

/// Run the driver with a smoke-sized budget and extract the returned
/// table into owned Rust values.
///
/// The extraction happens here rather than in the test body because the
/// returned `mlua::Table` borrows into the VM built above; handing it
/// back would leave the caller reading through a dropped VM.
fn run_gate_smoke() -> SmokeOut {
    let (lua, _tmp) = fixture_vm();

    // Smoke budget: 40 steps at batch 16. Large enough for the loss to
    // leave the uniform baseline on a 2-layer / dim-32 model, small
    // enough to stay well inside a CPU test run.
    let cfg = lua.create_table().expect("create SMOKE table");
    cfg.set("episodes", 12).expect("set episodes");
    cfg.set("steps", 40).expect("set steps");
    cfg.set("batch", 16).expect("set batch");
    cfg.set("lr", 3e-3).expect("set lr");
    cfg.set("seed", 20260731).expect("set seed");
    cfg.set("name", "nn-gate-smoke").expect("set name");
    lua.globals().set("SMOKE", cfg).expect("set SMOKE global");

    let out: mlua::Table = lua
        .load(SCRIPT)
        .set_name("@nn_gate_smoke")
        .eval()
        .expect("nn gate smoke script");

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
fn train_card_decode_gate_smoke() {
    let out = run_gate_smoke();

    eprintln!(
        "[nn-gate-smoke] card_id={} rows={} ctx_len={} train_loss={:.4} baseline={:.4} \
         decisions={}",
        out.card_id, out.rows, out.ctx_len, out.train_loss, out.baseline_loss, out.decisions
    );

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
        "every episode contributes 5 turns x 2 seats, so rows must be a multiple of 10; got {}",
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

    // (b) the decode gate never emits an illegal value.
    assert!(
        out.decide_legal,
        "every gated decode must return a legal value; decisions = {}",
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
        "driver reported failure; decisions = {}",
        out.decisions
    );
}
