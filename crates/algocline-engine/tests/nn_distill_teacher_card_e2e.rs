#![cfg(feature = "nn")]
//! Opt-in end-to-end test for the teacher-log distillation supply loop
//! (feature `nn`).
//!
//! What this proves: the whole real data path works together —
//! `alc.card.create` mints a Tier 1 Card declaring
//! `[metadata] loss_mask = "response"`, `alc.card.write_samples` writes
//! the teacher log to the Card samples sidecar, `alc.nn.data.from_card`
//! reads it back with a **real** gpt2 tokenizer and returns a
//! mask-carrying dataset, and `alc.nn.trainer.run_distill` trains a
//! from-scratch gpt2 student on it and mints a
//! `training_path = "distillation"` Card whose recorded `train_loss`
//! is below the untrained expectation.
//!
//! Network: on first run this downloads the gpt2 tokenizer from the
//! HuggingFace hub (`HfTokenizer::load_cached`). That is why the test
//! is gated: with `NN_SMOKE_DISTILL_CARD` unset it returns immediately
//! with a visible skip message and never touches the hub, so the
//! default `cargo test --workspace` stays network-free. Set
//! `NN_SMOKE_TOKENIZER_DIR` to a directory holding a previously
//! downloaded `gpt2.json` to make repeated local runs offline.
//!
//! Run it locally with:
//!
//! ```bash
//! NN_SMOKE_DISTILL_CARD=1 cargo test -p algocline-engine --features nn \
//!   --test nn_distill_teacher_card_e2e -- --nocapture
//! ```
//!
//! Expected cost: `gpt2 medium` (24 layers / 1024 dim, ~355M
//! parameters) is the only from-scratch gpt2 preset whose vocabulary
//! (50257) can hold real gpt2 token ids, so a CPU run with AdamW
//! optimizer state needs several GB of RSS and minutes of wall clock
//! at the default step count. Tune with `NN_SMOKE_STEPS` /
//! `NN_SMOKE_CTX` / `NN_SMOKE_BATCH` / `NN_SMOKE_LR` /
//! `NN_SMOKE_ROWS`.
//!
//! Scope limit (stated honestly): this test does **not** inspect the
//! mask contents. The Lua batch table exposes only `input_ids` /
//! `is_last`, so exact mask values are verified by the network-free
//! bridge tests in
//! `crates/algocline-engine/src/bridge/nn_card.rs`
//! (`from_card_with_loss_mask_declaration_returns_masked_batch` and
//! its paired legacy test), and the "masked loss actually descends"
//! property is covered by
//! `crates/algocline-nn/tests/distill_synthetic.rs`. What this file
//! adds is the real-tokenizer, real-sidecar wiring evidence.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

/// Build a production-shaped VM (mirrors `nn_card_test.rs::nn_card_vm`).
///
/// The tempdir is returned alongside the VM so the caller can keep it
/// alive for the duration of the test — dropping it would remove the
/// safetensors bundle mid-test.
fn e2e_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("test tempdir");
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
    (lua, tmp)
}

/// Embedded Lua driver. Reads the `SMOKE` config table (set from Rust
/// below) and walks the supply loop, returning the distillation card id.
const SCRIPT: &str = r#"
local rows = {}
for _ = 1, SMOKE.rows do
    table.insert(rows, {
        prompt = "Q: what is the capital of France?",
        response = "A: Paris.",
    })
end

local created = alc.card.create({
    pkg = { name = "alc_nn" },
    metadata = { kind = "teacher_log", loss_mask = "response" },
})
assert(
    type(created) == "table" and type(created.card_id) == "string",
    "alc.card.create must return a table carrying card_id"
)
local teacher_card_id = created.card_id

alc.card.write_samples(teacher_card_id, rows)

local ds = alc.nn.data.from_card(teacher_card_id, {
    tokenizer = "gpt2",
    batch_size = SMOKE.batch,
    ctx_len = SMOKE.ctx,
})

local student = alc.nn.preset.gpt2("medium", { pretrained = false, device = "cpu" })

local card_id = alc.nn.trainer.run_distill(student, ds, {
    lr = SMOKE.lr,
    batch = SMOKE.batch,
    steps = SMOKE.steps,
    warmup = 0,
    schedule = "CosineWithWarmup",
    loss_kind = "ce",
    name = "teacher-card-e2e",
})
assert(
    type(card_id) == "string" and #card_id > 0,
    "run_distill must return a non-empty card id string"
)

local card = alc.card.get(card_id)
assert(type(card) == "table", "alc.card.get must return the distillation card")
local nn = card.metadata and card.metadata.nn
assert(type(nn) == "table", "distillation card must carry metadata.nn")
assert(
    nn.training_path == "distillation",
    "training_path must be 'distillation', got " .. tostring(nn.training_path)
)
assert(
    type(nn.hyperparams) == "table" and nn.hyperparams.loss_kind == "ce",
    "hyperparams.loss_kind must be 'ce', got " .. tostring(nn.hyperparams and nn.hyperparams.loss_kind)
)

local loss = nn.metrics and nn.metrics.train_loss
assert(type(loss) == "number", "metrics.train_loss must be a number, got " .. type(loss))
assert(loss == loss and loss ~= math.huge and loss ~= -math.huge,
    "metrics.train_loss must be finite, got " .. tostring(loss))
assert(
    loss < SMOKE.max_loss,
    string.format(
        "masked distill run did not descend: train_loss=%.4f must be < %.4f (= ln(50257), the "
            .. "cross-entropy of a uniform softmax over the gpt2 vocabulary at random init)",
        loss,
        SMOKE.max_loss
    )
)

print(string.format(
    "[distill-card-e2e] card_id=%s train_loss=%.4f (< %.4f)",
    card_id, loss, SMOKE.max_loss
))
return card_id
"#;

/// End-to-end: teacher-log Card sidecar → masked dataset → distillation.
///
/// Skipped (visibly) unless `NN_SMOKE_DISTILL_CARD=1`; see the module
/// doc comment for the run command and the resource expectation.
#[test]
fn distill_from_teacher_card_with_real_tokenizer() {
    if std::env::var("NN_SMOKE_DISTILL_CARD").ok().as_deref() != Some("1") {
        eprintln!(
            "skipped: set NN_SMOKE_DISTILL_CARD=1 to run the real-tokenizer teacher-card distill E2E"
        );
        return;
    }

    let steps = env_usize("NN_SMOKE_STEPS", 8);
    let ctx = env_usize("NN_SMOKE_CTX", 64);
    let batch = env_usize("NN_SMOKE_BATCH", 1);
    let lr = env_f64("NN_SMOKE_LR", 3e-4);
    let rows = env_usize("NN_SMOKE_ROWS", 24);
    // Cross-entropy of a uniform softmax over the gpt2 vocabulary
    // (50257 classes) is ln(50257) ~= 10.8248 — the loss an untrained
    // student is expected to sit at.
    let max_loss = 10.82_f64;

    eprintln!(
        "[distill-card-e2e] config: steps={steps} ctx={ctx} batch={batch} lr={lr} rows={rows} \
         max_loss={max_loss}"
    );

    let (lua, tmp) = e2e_vm();
    let nn_dir = tmp.path().join("nn");

    // Optional offline path for repeated local runs: reuse a
    // previously downloaded tokenizer instead of hitting the hub.
    let tokenizer_dir = env_string("NN_SMOKE_TOKENIZER_DIR", "");
    if !tokenizer_dir.is_empty() {
        let src = PathBuf::from(&tokenizer_dir).join("gpt2.json");
        std::fs::create_dir_all(nn_dir.join("tokenizers"))
            .expect("create <nn_dir>/tokenizers for the pre-seeded tokenizer");
        std::fs::copy(&src, nn_dir.join("tokenizers/gpt2.json")).unwrap_or_else(|e| {
            panic!(
                "NN_SMOKE_TOKENIZER_DIR is set but copying {} failed: {e}",
                src.display()
            )
        });
        eprintln!(
            "[distill-card-e2e] pre-seeded tokenizer from {}",
            src.display()
        );
    }

    let cfg = lua.create_table().expect("create SMOKE table");
    cfg.set("steps", steps).expect("set steps");
    cfg.set("ctx", ctx).expect("set ctx");
    cfg.set("batch", batch).expect("set batch");
    cfg.set("lr", lr).expect("set lr");
    cfg.set("rows", rows).expect("set rows");
    cfg.set("max_loss", max_loss).expect("set max_loss");
    lua.globals().set("SMOKE", cfg).expect("set SMOKE global");

    let card_id: String = lua
        .load(SCRIPT)
        .set_name("@nn_distill_teacher_card_e2e")
        .eval()
        .expect("teacher-card distill E2E script");

    let bundle = nn_dir.join(format!("{card_id}.safetensors"));
    assert!(
        bundle.exists(),
        "distillation bundle must exist at {}",
        bundle.display()
    );
}
