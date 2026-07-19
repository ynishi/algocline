#![cfg(feature = "nn")]
//! Engine-level integration tests for the `examples/nn_*_smoke.lua`
//! scripts (ST-e). Each test constructs a production-shaped Lua VM
//! (same helper pattern as `nn_card_test.rs::nn_card_vm`) and
//! executes one smoke example against it, asserting the returned
//! table shape.
//!
//! Purpose: keep the shipped Lua examples runnable + regression-fenced
//! in CI. The tiny preset (`Gpt2Config::tiny`) + synthetic dataset
//! path (`alc.nn.data.synthetic`) added in the ST-e-infra commit
//! (`fe3406b`) let each example complete in a couple of seconds on
//! CPU without downloading pretrained bundles.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{self, BridgeConfig};
use algocline_engine::card::FileCardStore;
use algocline_engine::state::JsonFileStore;
use mlua::Lua;

/// Path to the workspace-root `examples/` directory. Resolved via
/// `CARGO_MANIFEST_DIR` (which points at `crates/algocline-engine/`)
/// and walking up two levels — avoids depending on the process CWD
/// which varies between `cargo test` and IDE runners.
fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine crate parent (crates/)")
        .parent()
        .expect("workspace root")
        .join("examples")
}

/// Build a production-shaped VM mirroring `nn_card_test.rs`. The
/// tempdir is returned alongside the VM so the caller keeps it alive
/// for the duration of the test (dropping mid-test would delete
/// per-run safetensors + Card TOML written by `alc.nn.trainer.*`
/// / `alc.nn.card.save`).
fn smoke_vm() -> (Lua, tempfile::TempDir) {
    let lua = Lua::new();
    let metrics = ExecutionMetrics::new();
    let tmp = tempfile::tempdir().expect("smoke tempdir");
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

/// Load a smoke example file and evaluate it against a fresh VM,
/// then extract every returned field into an owned Rust struct.
///
/// The extraction must happen INSIDE this helper because the
/// returned Lua table borrows into the local VM — returning the
/// `mlua::Table` directly would leave the caller reading through a
/// dropped VM (mlua panics with "Lua instance is destroyed").
fn run_smoke(script_name: &str) -> std::collections::HashMap<String, SmokeValue> {
    let (lua, _tmp) = smoke_vm();
    let path = examples_dir().join(script_name);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read example {}: {e}", path.display()));
    let chunk_name = format!("@{}", script_name);
    let out: mlua::Table = lua
        .load(src.as_str())
        .set_name(&chunk_name)
        .eval()
        .unwrap_or_else(|e| panic!("execute {}: {e}", path.display()));

    let mut extracted = std::collections::HashMap::new();
    for pair in out.pairs::<String, mlua::Value>() {
        let (k, v) = pair.unwrap_or_else(|e| panic!("read pair from {script_name}: {e}"));
        let owned = match v {
            mlua::Value::Boolean(b) => SmokeValue::Bool(b),
            mlua::Value::Integer(i) => SmokeValue::Int(i),
            mlua::Value::Number(n) => SmokeValue::Num(n),
            mlua::Value::String(s) => SmokeValue::Str(
                s.to_str()
                    .unwrap_or_else(|e| panic!("string decode from {script_name}.{k}: {e}"))
                    .to_string(),
            ),
            other => panic!("{script_name}.{k} has unsupported type: {other:?}"),
        };
        extracted.insert(k, owned);
    }
    extracted
}

/// Owned wire form of the values the smoke scripts return. The
/// harness only asserts on booleans, integers, and strings; the
/// `Num` variant exists so `run_smoke`'s pair extraction can accept
/// a float-valued `train_loss` field without panicking on
/// unsupported type (assertions on the float itself would be flaky
/// under smoke-scale training and are intentionally omitted).
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum SmokeValue {
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
}

impl SmokeValue {
    fn as_bool(&self) -> bool {
        match self {
            SmokeValue::Bool(b) => *b,
            other => panic!("expected bool, got {other:?}"),
        }
    }
    fn as_int(&self) -> i64 {
        match self {
            SmokeValue::Int(i) => *i,
            other => panic!("expected int, got {other:?}"),
        }
    }
    fn as_str(&self) -> &str {
        match self {
            SmokeValue::Str(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        }
    }
}

fn field<'a>(out: &'a std::collections::HashMap<String, SmokeValue>, name: &str) -> &'a SmokeValue {
    out.get(name).unwrap_or_else(|| {
        panic!(
            "field '{name}' missing from smoke output; got keys {:?}",
            out.keys().collect::<Vec<_>>()
        )
    })
}

#[test]
fn full_ft_smoke_runs() {
    let out = run_smoke("nn_full_ft_smoke.lua");
    assert!(
        field(&out, "ok").as_bool(),
        "full_ft smoke must report ok=true"
    );
    assert_eq!(field(&out, "variant").as_str(), "full_ft");
    assert_eq!(field(&out, "step").as_int(), 3);
    let bundle_ref = field(&out, "bundle_ref").as_str();
    assert!(
        bundle_ref.ends_with(".safetensors"),
        "bundle_ref must be a safetensors filename, got: {bundle_ref}"
    );
}

#[test]
fn lora_smoke_runs() {
    let out = run_smoke("nn_lora_smoke.lua");
    assert!(
        field(&out, "ok").as_bool(),
        "lora smoke must report ok=true"
    );
    assert_eq!(field(&out, "variant").as_str(), "lora");
    assert_eq!(field(&out, "step").as_int(), 3);
    assert!(
        !field(&out, "card_id").as_str().is_empty(),
        "card_id must be non-empty"
    );
    assert_eq!(field(&out, "lora_rank").as_int(), 4);
    assert_eq!(
        field(&out, "lora_targets").as_int(),
        6,
        "default target_modules must have 6 canonical entries"
    );
}

#[test]
fn distill_smoke_runs() {
    let out = run_smoke("nn_distill_smoke.lua");
    assert!(
        field(&out, "ok").as_bool(),
        "distill smoke must report ok=true"
    );
    assert_eq!(field(&out, "variant").as_str(), "distill");
    assert_eq!(field(&out, "step").as_int(), 3);
    let bundle_ref = field(&out, "bundle_ref").as_str();
    assert!(
        bundle_ref.ends_with(".safetensors"),
        "bundle_ref must be a safetensors filename, got: {bundle_ref}"
    );
}
