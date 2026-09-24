//! End-to-end coverage for the `corpus_bake` example.
//!
//! The example is a binary whose whole interface is the environment, so
//! it is exercised the way a pod would: fixtures are written into a temp
//! directory, the binary is spawned with the variables set, and the
//! summary line it prints on stdout is parsed back.
//!
//! Shapes are kept tiny (1 layer / 2 heads / dim 16 / 2 steps) so each
//! run is a CPU matter of milliseconds — this covers the driver's
//! plumbing, not its training behaviour.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{json, Value};
use tempfile::TempDir;

/// Path to the compiled example.
///
/// `cargo test` builds examples alongside the test targets, so this
/// resolves from the test binary's own location: `…/<profile>/deps/<test>`
/// sits next to `…/<profile>/examples/corpus_bake`.
fn bake_bin() -> PathBuf {
    let mut dir = std::env::current_exe().expect("test binary path");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    let bin = dir
        .join("examples")
        .join(format!("corpus_bake{}", std::env::consts::EXE_SUFFIX));
    assert!(
        bin.exists(),
        "example binary not found at {bin:?} — run these tests through `cargo test -p algocline-nn`, \
         which builds the crate's examples"
    );
    bin
}

/// Serialize `doc` into the fixture directory and return its path.
fn write_json(dir: &TempDir, name: &str, doc: &Value) -> String {
    let path = dir.path().join(name);
    std::fs::write(
        &path,
        serde_json::to_string(doc).expect("serialize fixture"),
    )
    .expect("write fixture");
    path.to_string_lossy().into_owned()
}

/// The rows every fixture is built from: `rows` sequences of `ctx_len`
/// ids drawn from `vocab_size`, offset by `seed` so two fixtures differ.
fn fixture_rows(rows: usize, ctx_len: usize, vocab_size: u32, seed: u32) -> Vec<Vec<u32>> {
    (0..rows)
        .map(|r| {
            (0..ctx_len)
                .map(|p| (seed + r as u32 + p as u32) % vocab_size)
                .collect()
        })
        .collect()
}

/// Write a corpus JSON in the plain format.
fn write_corpus(
    dir: &TempDir,
    name: &str,
    rows: usize,
    ctx_len: usize,
    vocab_size: u32,
    seed: u32,
) -> String {
    let body = fixture_rows(rows, ctx_len, vocab_size, seed);
    let doc = json!({
        "meta": { "ctx_len": ctx_len, "vocab_size": vocab_size, "note": "fixture" },
        "rows": body,
    });
    write_json(dir, name, &doc)
}

/// Write a corpus carrying per-row allowed-id sets.
///
/// The sets are sparse and keyed by 1-based position, so position `q`
/// governs row index `q - 1` and has to hold that index's token; a
/// second id is added beside it so the set constrains something.
/// Position 3 is left out of every row, which is how the format says
/// "unconstrained".
fn write_allowed_corpus(
    dir: &TempDir,
    name: &str,
    rows: usize,
    ctx_len: usize,
    vocab_size: u32,
    seed: u32,
) -> String {
    let body = fixture_rows(rows, ctx_len, vocab_size, seed);
    let sets: Vec<Value> = body
        .iter()
        .map(|row| {
            let mut map = serde_json::Map::new();
            for q in 2..=ctx_len {
                if q == 3 {
                    continue;
                }
                let token = row[q - 1];
                map.insert(q.to_string(), json!([token, (token + 1) % vocab_size]));
            }
            Value::Object(map)
        })
        .collect();

    let doc = json!({
        "meta": {
            "ctx_len": ctx_len,
            "vocab_size": vocab_size,
            "note": "fixture",
            "requires": ["per_row_allowed"],
        },
        "rows": body,
        "allowed": sets,
    });
    write_json(dir, name, &doc)
}

/// Run the example with the given extra variables on top of a small
/// CPU-sized baseline.
fn run_bake(extra: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(bake_bin());
    cmd.env("NN_BAKE_STEPS", "2")
        .env("NN_BAKE_BATCH", "2")
        .env("NN_BAKE_LAYERS", "1")
        .env("NN_BAKE_HEADS", "2")
        .env("NN_BAKE_DIM", "16")
        .env("NN_BAKE_LR", "1e-3");
    // The pod's environment is not this test's: anything a caller could
    // have exported that the baseline does not set is cleared, so a
    // stray variable cannot change which run is being tested.
    for (key, _) in std::env::vars() {
        if key.starts_with("NN_BAKE_")
            && !matches!(
                key.as_str(),
                "NN_BAKE_STEPS"
                    | "NN_BAKE_BATCH"
                    | "NN_BAKE_LAYERS"
                    | "NN_BAKE_HEADS"
                    | "NN_BAKE_DIM"
                    | "NN_BAKE_LR"
            )
        {
            cmd.env_remove(key);
        }
    }
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn corpus_bake")
}

fn summary_of(out: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().unwrap_or_default();
    serde_json::from_str(line).unwrap_or_else(|e| {
        panic!(
            "stdout is not a JSON summary line ({e}); stdout={stdout:?} stderr={:?}",
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// One corpus, no conditions: the plain trainer runs, the bundle lands
/// where `NN_BAKE_OUT` named it, and the summary says so.
#[test]
fn an_unconditioned_run_writes_its_bundle_and_summary() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("plain.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    assert!(out.exists(), "no bundle at {out:?}");

    let summary = summary_of(&result);
    // Echoed so a `--nocapture` run shows the line a pod operator reads.
    println!("summary: {summary}");
    assert_eq!(summary["steps"], json!(2));
    assert_eq!(summary["cond_slots"], Value::Null);
    assert_eq!(summary["ctx_len"], json!(8));
    assert_eq!(summary["vocab_size"], json!(32));
    assert_eq!(summary["corpus_rows"], json!(6));
    assert_eq!(summary["out"], json!(out.to_string_lossy()));
    assert!(
        summary["final_loss"].as_f64().expect("final_loss") > 0.0,
        "{summary}"
    );
    // The dataset is cycled to cover steps × batch plus a batch of margin.
    assert_eq!(summary["rows"], json!(6));
}

/// Two corpora with a condition each: the conditioned trainer runs, the
/// table is sized from the slots named, and the bundle lands.
#[test]
fn a_conditioned_run_takes_one_slot_per_corpus() {
    let dir = TempDir::new().unwrap();
    let a = write_corpus(&dir, "a.json", 5, 8, 32, 1);
    let b = write_corpus(&dir, "b.json", 5, 8, 32, 9);
    let out = dir.path().join("cond.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_COND", "0,1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    assert!(out.exists(), "no bundle at {out:?}");

    let summary = summary_of(&result);
    assert_eq!(summary["steps"], json!(2));
    assert_eq!(summary["cond_slots"], json!(2));
    assert_eq!(summary["corpus_rows"], json!(10));
    assert_eq!(summary["device"], json!("cpu"));
}

/// A condition list of a different length than the corpus list would
/// train some corpus under another corpus's condition, and every shape
/// would still agree — so it is refused by name.
#[test]
fn a_condition_list_of_the_wrong_arity_is_refused() {
    let dir = TempDir::new().unwrap();
    let a = write_corpus(&dir, "a.json", 4, 8, 32, 1);
    let b = write_corpus(&dir, "b.json", 4, 8, 32, 9);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_COND", "0"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_COND"), "{err}");
    assert!(err.contains("NN_BAKE_CORPUS"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// A corpus carrying allowed-id sets, run with the loss scoring each
/// target among them: the run completes, the bundle lands, and the
/// summary records which switch was on and how wide the sets were.
#[test]
fn a_masked_run_writes_its_bundle_and_records_the_switch() {
    let dir = TempDir::new().unwrap();
    let corpus = write_allowed_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("masked.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_MASK_LOGITS", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    assert!(out.exists(), "no bundle at {out:?}");

    let summary = summary_of(&result);
    println!("summary: {summary}");
    assert_eq!(summary["steps"], json!(2));
    assert_eq!(summary["mask_disallowed_logits"], json!(true));
    assert_eq!(summary["allowed_input"], json!(false));
    // The fixture lists positions up to the context length.
    assert_eq!(summary["allowed_positions"], json!(8));
    assert_eq!(summary["cond_slots"], Value::Null);
}

/// The same sets handed to the model as an input instead: the model is
/// built with the allowed-id table and trained through the entry point
/// that carries it.
#[test]
fn an_allowed_input_run_writes_its_bundle_and_records_the_switch() {
    let dir = TempDir::new().unwrap();
    let corpus = write_allowed_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("channel.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_ALLOWED_INPUT", "true"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    assert!(out.exists(), "no bundle at {out:?}");

    let summary = summary_of(&result);
    println!("summary: {summary}");
    assert_eq!(summary["allowed_input"], json!(true));
    assert_eq!(summary["mask_disallowed_logits"], json!(false));
    assert_eq!(summary["allowed_positions"], json!(8));
}

/// The corpus format's own refusals belong to `train::corpus` and are
/// pinned there. This one stands for all of them: the loader's error
/// reaches the caller through the driver, naming the variable the path
/// came from and the requirement it does not implement.
#[test]
fn a_corpus_the_loader_refuses_is_refused_by_the_driver_too() {
    let dir = TempDir::new().unwrap();
    let doc = json!({
        "meta": {
            "ctx_len": 8,
            "vocab_size": 32,
            "requires": ["per_row_weights"],
        },
        "rows": fixture_rows(4, 8, 32, 1),
    });
    let corpus = write_json(&dir, "a.json", &doc);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_CORPUS"), "{err}");
    assert!(err.contains("per_row_weights"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// A switch asking for sets no corpus carries would otherwise train a
/// plain run under the name of a constrained one.
#[test]
fn a_switch_without_the_sets_it_consumes_is_refused() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 4, 8, 32, 1);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_MASK_LOGITS", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_MASK_LOGITS"), "{err}");
    assert!(err.contains("per_row_allowed"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// Sets that nothing consumes are the same failure seen from the corpus
/// side: they would be read and then used for nothing.
#[test]
fn sets_no_switch_consumes_are_refused() {
    let dir = TempDir::new().unwrap();
    let corpus = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_MASK_LOGITS"), "{err}");
    assert!(err.contains("NN_BAKE_ALLOWED_INPUT"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// The allowed-id input and conditioning have no combined forward pass,
/// so asking for both is refused at the driver rather than a corpus read
/// later at the model builder.
#[test]
fn the_allowed_input_cannot_be_combined_with_conditioning() {
    let dir = TempDir::new().unwrap();
    let a = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1);
    let b = write_allowed_corpus(&dir, "b.json", 4, 8, 32, 9);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_COND", "0,1"),
        ("NN_BAKE_ALLOWED_INPUT", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_ALLOWED_INPUT"), "{err}");
    assert!(err.contains("NN_BAKE_COND"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// Two corpora carrying sets, interleaved: each row keeps its own sets
/// through the merge, which the dataset checks by refusing any set that
/// excludes the token its own position holds — so a run that completes
/// is a run whose pairing survived.
#[test]
fn interleaved_corpora_keep_each_row_with_its_own_sets() {
    let dir = TempDir::new().unwrap();
    let a = write_allowed_corpus(&dir, "a.json", 5, 8, 32, 1);
    let b = write_allowed_corpus(&dir, "b.json", 5, 8, 32, 9);
    let out = dir.path().join("merged.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_MASK_LOGITS", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let summary = summary_of(&result);
    assert_eq!(summary["corpus_rows"], json!(10));
    assert_eq!(summary["allowed_positions"], json!(8));
}

/// Refusal helper: the run exits non-zero, writes no bundle, and every
/// fragment appears in what it printed.
fn assert_refused(result: &Output, out: &std::path::Path, fragments: &[&str]) {
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(result);
    for fragment in fragments {
        assert!(err.contains(fragment), "missing {fragment:?} in: {err}");
    }
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// Split a safetensors file into its parsed header and its data section.
fn header_and_data(path: &std::path::Path) -> (Value, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read bundle");
    let len = u64::from_le_bytes(bytes[..8].try_into().expect("length prefix")) as usize;
    let header = serde_json::from_slice(&bytes[8..8 + len]).expect("header is JSON");
    (header, bytes[8 + len..].to_vec())
}

/// Two runs under one seed start from the same weights, see the same
/// rows in the same order, and so write the same tensors.
///
/// Compared as tensors plus a parsed header rather than as whole files:
/// the checkpoint writer serialises `__metadata__` from a hash map, so
/// the header's key order differs between two saves of the same values.
#[test]
fn a_seeded_run_is_repeatable() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let mut bundles = Vec::new();
    for name in ["first", "second"] {
        let out = dir.path().join(format!("{name}.safetensors"));
        let result = run_bake(&[
            ("NN_BAKE_CORPUS", corpus.as_str()),
            ("NN_BAKE_SEED", "7"),
            ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
        ]);
        assert!(
            result.status.success(),
            "run failed: {}",
            stderr_of(&result)
        );
        let summary = summary_of(&result);
        assert_eq!(summary["seed"], json!(7));
        let (header, data) = header_and_data(&out);
        bundles.push((summary["final_loss"].clone(), header, data));
    }
    assert_eq!(bundles[0].0, bundles[1].0, "final losses differ");
    assert_eq!(bundles[0].1, bundles[1].1, "the two headers differ");
    assert!(
        bundles[0].2 == bundles[1].2,
        "the two runs wrote different tensors"
    );
}

/// A held-out corpus scored every step: the summary carries the
/// held-out loss, and the held-out rows are read once rather than
/// cycled like the training rows.
#[test]
fn a_run_with_a_held_out_corpus_reports_its_loss() {
    let dir = TempDir::new().unwrap();
    let train = write_corpus(&dir, "train.json", 6, 8, 32, 1);
    let val = write_corpus(&dir, "val.json", 4, 8, 32, 17);
    let out = dir.path().join("val.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", train.as_str()),
        ("NN_BAKE_VAL_CORPUS", val.as_str()),
        ("NN_BAKE_EVAL_EVERY", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let summary = summary_of(&result);
    println!("summary: {summary}");
    assert!(
        summary["val_loss"].as_f64().expect("val_loss") > 0.0,
        "{summary}"
    );
    assert!(summary["min_val_loss"].as_f64().is_some(), "{summary}");
    assert_eq!(summary["val_rows"], json!(4));
    assert_eq!(summary["stopped_early"], json!(false));
}

/// The trainer's rule that a held-out set and `eval_every` go together
/// reaches the caller unchanged: a held-out corpus nothing scores is
/// refused, not paid for and ignored.
#[test]
fn a_held_out_corpus_without_eval_every_is_refused() {
    let dir = TempDir::new().unwrap();
    let train = write_corpus(&dir, "train.json", 6, 8, 32, 1);
    let val = write_corpus(&dir, "val.json", 4, 8, 32, 17);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", train.as_str()),
        ("NN_BAKE_VAL_CORPUS", val.as_str()),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&result, &out, &["eval_every"]);
}

/// A conditioned model cannot be scored on a row that names no
/// condition, so a conditioned run's held-out corpus needs its own.
#[test]
fn a_conditioned_run_needs_conditions_for_its_held_out_corpus() {
    let dir = TempDir::new().unwrap();
    let a = write_corpus(&dir, "a.json", 4, 8, 32, 1);
    let b = write_corpus(&dir, "b.json", 4, 8, 32, 9);
    let val = write_corpus(&dir, "val.json", 4, 8, 32, 17);
    let out = dir.path().join("never.safetensors");
    let refused = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_COND", "0,1"),
        ("NN_BAKE_VAL_CORPUS", val.as_str()),
        ("NN_BAKE_EVAL_EVERY", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&refused, &out, &["NN_BAKE_VAL_COND"]);

    let out = dir.path().join("cond-val.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_COND", "0,1"),
        ("NN_BAKE_VAL_CORPUS", val.as_str()),
        ("NN_BAKE_VAL_COND", "1"),
        ("NN_BAKE_EVAL_EVERY", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let summary = summary_of(&result);
    assert!(summary["val_loss"].as_f64().is_some(), "{summary}");
    assert_eq!(summary["cond_slots"], json!(2));
}

/// Early stopping has no default on either half, so one half alone is
/// refused by the driver, naming the one that is missing.
#[test]
fn early_stop_needs_both_halves() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_EARLY_STOP_PATIENCE", "2"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&result, &out, &["NN_BAKE_EARLY_STOP_MIN_DELTA"]);
}

/// A run asked for a curve leaves one line per point beside its bundle,
/// and the summary points at the file.
#[test]
fn metrics_every_leaves_a_curve_beside_the_bundle() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("curve.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_METRICS_EVERY", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let summary = summary_of(&result);
    let curve = dir.path().join("curve-metrics.jsonl");
    assert_eq!(summary["metrics"], json!(curve.to_string_lossy()));
    let text = std::fs::read_to_string(&curve).expect("read curve");
    let points: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("curve line is JSON"))
        .collect();
    assert_eq!(points.len(), 2, "{text}");
    assert_eq!(points[1]["step"], json!(2));
}

/// Schedule and optimizer names are the trainer's wire names; one it
/// does not know is refused with the alternatives, not replaced by the
/// default.
#[test]
fn an_unknown_schedule_or_optimizer_is_refused_with_the_alternatives() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_SCHEDULE", "cosine-annealing"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&result, &out, &["NN_BAKE_SCHEDULE", "wsd"]);

    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_OPTIMIZER", "sgd"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&result, &out, &["NN_BAKE_OPTIMIZER", "lion"]);

    let ok = dir.path().join("lion.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_SCHEDULE", "cosine"),
        ("NN_BAKE_WARMUP", "1"),
        ("NN_BAKE_OPTIMIZER", "lion"),
        ("NN_BAKE_CLIP_GRAD_NORM", "1.0"),
        ("NN_BAKE_OUT", ok.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let summary = summary_of(&result);
    assert_eq!(summary["schedule"], json!("cosine"));
    assert_eq!(summary["optimizer"], json!("lion"));
}

/// The pad id fills every row short of `ctx_len`, so one outside the
/// corpus's id space would teach the model an id the corpus says does
/// not exist.
#[test]
fn a_pad_id_outside_the_vocabulary_is_refused() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_PAD_ID", "32"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert_refused(&result, &out, &["NN_BAKE_PAD_ID", "vocab_size"]);
}

/// A checkpoint written with its optimizer state can be handed back as
/// `NN_BAKE_INIT_FROM`, and the run picks up from it.
#[test]
fn a_run_resumes_from_a_checkpoint_it_wrote() {
    let dir = TempDir::new().unwrap();
    let corpus = write_corpus(&dir, "a.json", 6, 8, 32, 1);
    let first = dir.path().join("first.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_SEED", "3"),
        ("NN_BAKE_CKPT_EVERY", "1"),
        ("NN_BAKE_SAVE_OPTIMIZER_STATE", "1"),
        ("NN_BAKE_OUT", first.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    let step1 = dir.path().join("first-step1.safetensors");
    assert!(step1.exists(), "no rotating checkpoint at {step1:?}");

    let second = dir.path().join("second.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_INIT_FROM", step1.to_string_lossy().as_ref()),
        ("NN_BAKE_OUT", second.to_string_lossy().as_ref()),
    ]);
    assert!(
        result.status.success(),
        "run failed: {}",
        stderr_of(&result)
    );
    assert!(second.exists(), "no bundle at {second:?}");
}
