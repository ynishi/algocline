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

/// Write a corpus JSON holding `rows` sequences of `ctx_len` ids drawn
/// from `vocab_size`, offset by `seed` so two fixtures differ.
fn write_corpus(
    dir: &TempDir,
    name: &str,
    rows: usize,
    ctx_len: usize,
    vocab_size: u32,
    seed: u32,
) -> String {
    let body: Vec<Vec<u32>> = (0..rows)
        .map(|r| {
            (0..ctx_len)
                .map(|p| (seed + r as u32 + p as u32) % vocab_size)
                .collect()
        })
        .collect();
    let path = dir.path().join(name);
    let doc = json!({
        "meta": { "ctx_len": ctx_len, "vocab_size": vocab_size, "note": "fixture" },
        "rows": body,
    });
    std::fs::write(
        &path,
        serde_json::to_string(&doc).expect("serialize fixture"),
    )
    .expect("write fixture");
    path.to_string_lossy().into_owned()
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
