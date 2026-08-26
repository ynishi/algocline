#![cfg(feature = "nn")]
//! The four training arms a corpus can drive, run end to end through
//! the Lua surface.
//!
//! The unit tests around `alc.nn.data.corpus` pin that the allowed-id
//! sets reach the dataset. That is not the same claim as "a run using
//! them completes": the sets are read by the loss in one arm and by the
//! forward pass in another, and which arm runs follows the shape the
//! handle was built with rather than anything the caller states. So the
//! four combinations are exercised here rather than reasoned about —
//! plain, conditioned, masked loss, and allowed-as-input.
//!
//! Shapes are the smallest the preset accepts (1 layer / 2 heads /
//! dim 8 / ctx 4 / vocab 16) and each run is two optimizer steps, which
//! keeps the whole file inside a couple of seconds on CPU. Nothing here
//! asserts on a loss: two steps of a random-init model say nothing about
//! learning, and an assertion that pretended otherwise would fail on the
//! seed rather than on a defect.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

/// Build a production-shaped VM. Mirrors `nn_smoke_test.rs::smoke_vm`;
/// the tempdir comes back with it because the trainer writes
/// safetensors and Card TOML under it for the life of the test.
fn corpus_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("corpus arm tempdir");
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
    (lua, tmp)
}

/// A corpus of two rows, without allowed-id sets.
fn plain_corpus(dir: &tempfile::TempDir, name: &str, rows: &str) -> String {
    write_corpus(
        dir,
        name,
        &format!(r#"{{ "meta": {{ "ctx_len": 4, "vocab_size": 16 }}, "rows": {rows} }}"#),
    )
}

/// A corpus of two rows carrying allowed-id sets.
///
/// Every set holds the token its own position carries — a set that did
/// not would be refused at the dataset, which is a different test.
fn constrained_corpus(dir: &tempfile::TempDir, name: &str) -> String {
    write_corpus(
        dir,
        name,
        r#"{ "meta": { "ctx_len": 4, "vocab_size": 16,
                       "requires": ["per_row_allowed"] },
              "rows": [[1, 2, 3, 4], [5, 6, 7, 8]],
              "allowed": [{ "2": [2, 9], "3": [3, 10], "4": [4, 11] },
                          { "2": [6, 9], "3": [7, 10], "4": [8, 11] }] }"#,
    )
}

fn write_corpus(dir: &tempfile::TempDir, name: &str, body: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, body).expect("write corpus fixture");
    // Embedded into Lua source below, so the separators have to survive
    // the quoting rather than be read as escapes.
    path.to_string_lossy().replace('\\', "\\\\")
}

/// Run one arm and return the card id it produced.
fn run_arm(lua: &Lua, source: &str) -> String {
    lua.load(source)
        .set_name("@corpus_arm")
        .eval()
        .unwrap_or_else(|e| panic!("arm failed: {e}"))
}

/// Two steps over a corpus with no side channel at all: the arm every
/// other one is a deviation from.
#[test]
fn a_plain_corpus_trains() {
    let (lua, tmp) = corpus_vm();
    let path = plain_corpus(&tmp, "plain.json", "[[1, 2, 3, 4], [5, 6, 7, 8]]");

    let card = run_arm(
        &lua,
        &format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{path}" }}, {{ batch_size = 2, epochs = 4 }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                name = "corpus-plain",
            }})
            "#
        ),
    );
    assert!(!card.is_empty(), "the plain arm must return a card id");
}

/// Two corpora, one conditioning-table row each. `run_full_ft` routes
/// to the conditioned pass off the handle's `cond_slots`, so what this
/// pins is that the corpus's per-source labels survive the merge and
/// the repeat into a run that completes.
#[test]
fn a_conditioned_corpus_trains() {
    let (lua, tmp) = corpus_vm();
    let a = plain_corpus(&tmp, "a.json", "[[1, 2, 3, 4], [5, 6, 7, 8]]");
    let b = plain_corpus(&tmp, "b.json", "[[9, 10, 11, 12], [13, 14, 15, 1]]");

    let card = run_arm(
        &lua,
        &format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{a}", "{b}" }}, {{
                batch_size = 2, epochs = 4, cond = {{ 0, 1 }}, cond_slots = 2,
            }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
                cond_slots = 2,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                name = "corpus-cond",
            }})
            "#
        ),
    );
    assert!(
        !card.is_empty(),
        "the conditioned arm must return a card id"
    );
}

/// The sets read as a mask on the loss. The model is shaped exactly as
/// the plain arm's — nothing about the handle says the run is
/// constrained — so this arm exists to show the sets travel from the
/// corpus to the loss without the model carrying a table for them.
#[test]
fn a_corpus_with_sets_trains_with_the_loss_masked() {
    let (lua, tmp) = corpus_vm();
    let path = constrained_corpus(&tmp, "masked.json");

    let card = run_arm(
        &lua,
        &format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{path}" }}, {{ batch_size = 2, epochs = 4 }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                mask_disallowed_logits = true, name = "corpus-masked",
            }})
            "#
        ),
    );
    assert!(!card.is_empty(), "the masked arm must return a card id");
}

/// The sets read as an input to the model. Here the handle does carry a
/// table, which is what routes `run_full_ft` to the allowed pass, and
/// the corpus has to supply a set at every position that pass reads.
#[test]
fn a_corpus_with_sets_trains_with_the_sets_as_input() {
    let (lua, tmp) = corpus_vm();
    let path = constrained_corpus(&tmp, "input.json");

    let card = run_arm(
        &lua,
        &format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{path}" }}, {{ batch_size = 2, epochs = 4 }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
                allowed_input = true,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                name = "corpus-allowed-input",
            }})
            "#
        ),
    );
    assert!(
        !card.is_empty(),
        "the allowed-as-input arm must return a card id"
    );
}

/// Both readers of the sets at once: the model is told what is
/// available and the loss scores the target among the same ids. The
/// two are independent switches, and nothing else in this file runs
/// them together.
#[test]
fn the_two_readers_of_the_sets_run_together() {
    let (lua, tmp) = corpus_vm();
    let path = constrained_corpus(&tmp, "both.json");

    let card = run_arm(
        &lua,
        &format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{path}" }}, {{ batch_size = 2, epochs = 4 }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
                allowed_input = true,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                mask_disallowed_logits = true, name = "corpus-both",
            }})
            "#
        ),
    );
    assert!(
        !card.is_empty(),
        "both readers together must return a card id"
    );
}

/// Masking asks the loss for the ids a position allowed, so a corpus
/// that never recorded any leaves it with nothing to score against.
/// Refused rather than run unmasked: the run would otherwise report the
/// numbers of a masked one.
#[test]
fn masking_a_corpus_that_carries_no_sets_is_refused() {
    let (lua, tmp) = corpus_vm();
    let path = plain_corpus(&tmp, "plain.json", "[[1, 2, 3, 4], [5, 6, 7, 8]]");

    let err = lua
        .load(format!(
            r#"
            local ds = alc.nn.data.corpus({{ "{path}" }}, {{ batch_size = 2, epochs = 4 }})
            local h = alc.nn.preset.gpt2("custom", {{
                pretrained = false, layers = 1, heads = 2, dim = 8, ctx = 4, vocab = 16,
            }})
            return alc.nn.trainer.run_full_ft(h, ds, {{
                lr = 1e-3, batch = 2, steps = 2, warmup = 0, schedule = "Constant",
                mask_disallowed_logits = true, name = "corpus-masked-without-sets",
            }})
            "#
        ))
        .set_name("@corpus_arm")
        .eval::<String>()
        .expect_err("masking without sets must be refused");
    let text = err.to_string();
    assert!(
        text.contains("allowed"),
        "the refusal must name the missing sets: {text}"
    );
}
