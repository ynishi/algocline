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

/// Which halves of the announcement/field pairing a fixture writes.
#[derive(Clone, Copy)]
enum Allowed {
    /// `meta.requires` names the extension and `allowed` is written.
    Announced,
    /// The requirement without the field it names.
    RequirementOnly,
    /// The field without the requirement announcing it.
    FieldOnly,
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
    pairing: Allowed,
) -> String {
    let announce = matches!(pairing, Allowed::Announced | Allowed::RequirementOnly);
    let carry = matches!(pairing, Allowed::Announced | Allowed::FieldOnly);
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

    let mut meta = json!({ "ctx_len": ctx_len, "vocab_size": vocab_size, "note": "fixture" });
    if announce {
        meta["requires"] = json!(["per_row_allowed"]);
    }
    let mut doc = json!({ "meta": meta, "rows": body });
    if carry {
        doc["allowed"] = Value::Array(sets);
    }
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
        .env("NN_BAKE_LR", "1e-3")
        .env_remove("NN_BAKE_COND")
        .env_remove("NN_BAKE_COND_SLOTS")
        .env_remove("NN_BAKE_MASK_LOGITS")
        .env_remove("NN_BAKE_ALLOWED_INPUT")
        .env_remove("NN_BAKE_PAD_ID");
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
    let corpus = write_allowed_corpus(&dir, "a.json", 6, 8, 32, 1, Allowed::Announced);
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
    let corpus = write_allowed_corpus(&dir, "a.json", 6, 8, 32, 1, Allowed::Announced);
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

/// `meta.requires` naming the extension while the field it points at is
/// absent is a producer that meant to write sets and did not.
#[test]
fn a_requirement_without_the_field_it_names_is_refused() {
    let dir = TempDir::new().unwrap();
    let corpus = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1, Allowed::RequirementOnly);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_MASK_LOGITS", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("per_row_allowed"), "{err}");
    assert!(err.contains("allowed"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// The field without the requirement announcing it is the other
/// direction: a reader that does not implement it would train on the
/// same rows unconstrained and report the same numbers.
#[test]
fn the_field_without_its_requirement_is_refused() {
    let dir = TempDir::new().unwrap();
    let corpus = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1, Allowed::FieldOnly);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", corpus.as_str()),
        ("NN_BAKE_MASK_LOGITS", "1"),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("meta.requires"), "{err}");
    assert!(err.contains("allowed"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}

/// A requirement this driver does not implement is refused by name
/// rather than ignored as one more unread `meta` field.
#[test]
fn a_requirement_this_driver_does_not_implement_is_refused_by_name() {
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
    let corpus = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1, Allowed::Announced);
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
    let a = write_allowed_corpus(&dir, "a.json", 4, 8, 32, 1, Allowed::Announced);
    let b = write_allowed_corpus(&dir, "b.json", 4, 8, 32, 9, Allowed::Announced);
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
    let a = write_allowed_corpus(&dir, "a.json", 5, 8, 32, 1, Allowed::Announced);
    let b = write_allowed_corpus(&dir, "b.json", 5, 8, 32, 9, Allowed::Announced);
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

/// Corpora that declare different shapes cannot be trained into one
/// model; the disagreement is reported with both paths.
#[test]
fn corpora_that_disagree_on_meta_are_refused() {
    let dir = TempDir::new().unwrap();
    let a = write_corpus(&dir, "a.json", 4, 8, 32, 1);
    let b = write_corpus(&dir, "b.json", 4, 8, 24, 3);
    let out = dir.path().join("never.safetensors");
    let result = run_bake(&[
        ("NN_BAKE_CORPUS", format!("{a},{b}").as_str()),
        ("NN_BAKE_OUT", out.to_string_lossy().as_ref()),
    ]);
    assert!(!result.status.success(), "the run should have refused");
    let err = stderr_of(&result);
    assert!(err.contains("NN_BAKE_CORPUS"), "{err}");
    assert!(err.contains("disagree on shape"), "{err}");
    assert!(err.contains("vocab_size"), "{err}");
    assert!(!out.exists(), "a refused run wrote {out:?}");
}
