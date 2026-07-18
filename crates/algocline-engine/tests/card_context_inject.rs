//! Phase 3-F integration tests: `alc.llm(prompt, {card_context = …})`
//! injects a `<past_cards>` prefix onto the outgoing `QueryRequest.system`.
//!
//! Each test wires the production `bridge::register` path with a real
//! [`FileCardStore`] backed by a `tempdir`, plumbs a `tokio::sync::mpsc`
//! channel to capture the resulting [`LlmRequest`], and drives an
//! `alc.llm(...)` call via `lua.load(...).exec_async()`.  A concurrent
//! async block drains the mpsc receiver and unblocks the coroutine's
//! oneshot response so the `alc.llm` call returns.
//!
//! `tokio::time::timeout(Duration::from_secs(5), ...)` guards each test
//! against deadlock (e.g. a wiring regression that drops the receiver
//! before responding).

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use algocline_core::ExecutionMetrics;
use algocline_engine::bridge::{register, BridgeConfig};
use algocline_engine::{FileCardStore, JsonFileStore, LlmRequest};
use mlua::Lua;

/// Write a Card TOML directly onto disk (bypasses `create_with_store`
/// so tests exercise the on-disk shape without auto-inject side effects).
fn write_card_toml(root: &Path, pkg: &str, card_id: &str, toml_text: &str) {
    let pkg_dir = root.join(pkg);
    fs::create_dir_all(&pkg_dir).expect("test setup: create pkg dir");
    let path = pkg_dir.join(format!("{card_id}.toml"));
    fs::write(&path, toml_text).expect("test setup: write card toml");
}

/// Build a minimum-valid Card TOML with optional `[stats].pass_rate` and
/// `[run].status`.
fn fixture_toml(
    pkg: &str,
    card_id: &str,
    created_at: &str,
    pass_rate: Option<f64>,
    run_status: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str("schema_version = \"card/v0\"\n");
    out.push_str(&format!("card_id = \"{card_id}\"\n"));
    out.push_str(&format!("created_at = \"{created_at}\"\n"));
    out.push_str("\n[pkg]\n");
    out.push_str(&format!("name = \"{pkg}\"\n"));
    if let Some(rate) = pass_rate {
        out.push_str("\n[stats]\n");
        out.push_str(&format!("pass_rate = {rate}\n"));
    }
    if let Some(status) = run_status {
        out.push_str("\n[run]\n");
        out.push_str(&format!("status = \"{status}\"\n"));
    }
    out
}

/// Fixture bundle: root tempdir + Lua VM wired to a real card store +
/// LlmRequest receiver.
struct Fixture {
    _tmp: tempfile::TempDir,
    lua: Lua,
    rx: tokio::sync::mpsc::Receiver<LlmRequest>,
    root: std::path::PathBuf,
}

fn setup_fixture() -> Fixture {
    let tmp = tempfile::tempdir().expect("test setup: tempdir");
    let root = tmp.path().to_path_buf();
    let metrics = ExecutionMetrics::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<LlmRequest>(8);

    let config = BridgeConfig {
        llm_tx: Some(tx),
        ns: "default".into(),
        custom_metrics: metrics.custom_metrics_handle(),
        stats: metrics.stats_handle(),
        budget: metrics.budget_handle(),
        progress: metrics.progress_handle(),
        lib_paths: vec![],
        variant_pkgs: vec![],
        state_store: Arc::new(JsonFileStore::new(root.join("state"))),
        card_store: Arc::new(FileCardStore::new(root.join("cards"))),
        card_run_enabled: true,
        scenarios_dir: root.join("scenarios"),
        log_sink: None,
    };

    let lua = Lua::new();
    let alc_table = lua.create_table().expect("test setup: create alc table");
    register(&lua, &alc_table, config).expect("test setup: register bridge");
    lua.globals()
        .set("alc", alc_table)
        .expect("test setup: install alc global");

    Fixture {
        _tmp: tmp,
        lua,
        rx,
        root,
    }
}

/// Drive one `alc.llm(...)` invocation to completion, returning the
/// captured [`LlmRequest`].  Uses `tokio::join!` (single-task, avoids
/// mlua Send/lifetime friction from `tokio::spawn`) and a 5-second
/// timeout to guard against deadlock.
async fn drive_llm_call(
    lua: &Lua,
    rx: &mut tokio::sync::mpsc::Receiver<LlmRequest>,
    script: &str,
) -> LlmRequest {
    let capture = async {
        let mut req = rx
            .recv()
            .await
            .expect("mpsc: receiver must receive one LlmRequest");
        // Drain the oneshot channels so the yielded coroutine resumes.
        // Replace each resp_tx with a fresh one so the returned
        // `LlmRequest` remains structurally valid for assertions.
        for q in req.queries.iter_mut() {
            let (fresh_tx, _fresh_rx) = tokio::sync::oneshot::channel();
            let old_tx = std::mem::replace(&mut q.resp_tx, fresh_tx);
            let _ = old_tx.send(Ok("ok".to_string()));
        }
        req
    };
    let lua_fut = lua.load(script).exec_async();

    let outcome = tokio::time::timeout(Duration::from_secs(5), async {
        let (req, lua_result) = tokio::join!(capture, lua_fut);
        (req, lua_result)
    })
    .await
    .expect("test must not deadlock within 5s");

    outcome.1.expect("alc.llm script must succeed");
    outcome.0
}

#[tokio::test]
async fn single_card_id_injects_prefix() {
    let mut fx = setup_fixture();
    let cards_root = fx.root.join("cards");
    let toml_text = fixture_toml(
        "cot",
        "cot_test_1",
        "2026-07-18T00:00:00Z",
        Some(4.5),
        Some("succeeded"),
    );
    write_card_toml(&cards_root, "cot", "cot_test_1", &toml_text);

    let script = r#"alc.llm("hi", { card_context = "cot_test_1", system = "S" })"#;
    let req = drive_llm_call(&fx.lua, &mut fx.rx, script).await;

    assert_eq!(req.queries.len(), 1, "single query expected");
    let system = req.queries[0]
        .system
        .as_deref()
        .expect("system must be set (prefix + original)");
    assert!(
        system.starts_with("<past_cards>\n"),
        "prefix must open with tag: {system}"
    );
    assert!(
        system.contains("card_id=cot_test_1"),
        "must reference the injected card_id: {system}"
    );
    assert!(
        system.contains("[run.status=succeeded]"),
        "must include run.status: {system}"
    );
    assert!(
        system.contains("Rating 4.5"),
        "must include rating: {system}"
    );
    assert!(
        system.ends_with("\n</past_cards>\n\nS"),
        "must close with tag and preserve original system: {system}"
    );
}

#[tokio::test]
async fn query_form_injects_recent() {
    let mut fx = setup_fixture();
    let cards_root = fx.root.join("cards");
    for i in 0..5 {
        let card_id = format!("cot_multi_{i}");
        let ts = format!("2026-07-{:02}T00:00:00Z", 10 + i);
        let toml_text = fixture_toml("cot", &card_id, &ts, Some(4.0), Some("succeeded"));
        write_card_toml(&cards_root, "cot", &card_id, &toml_text);
    }

    let script = r#"alc.llm("hi", { card_context = { pkg = "cot", limit = 3 } })"#;
    let req = drive_llm_call(&fx.lua, &mut fx.rx, script).await;

    let system = req.queries[0]
        .system
        .as_deref()
        .expect("system must be set to the past_cards block");
    assert!(
        system.starts_with("<past_cards>\n"),
        "must open with tag: {system}"
    );
    // Newest first: cot_multi_4, cot_multi_3, cot_multi_2; oldest two
    // (cot_multi_0, cot_multi_1) must not appear.
    assert!(
        system.contains("card_id=cot_multi_4"),
        "newest card must be present: {system}"
    );
    assert!(
        system.contains("card_id=cot_multi_3"),
        "second newest card must be present: {system}"
    );
    assert!(
        system.contains("card_id=cot_multi_2"),
        "third newest card must be present: {system}"
    );
    assert!(
        !system.contains("card_id=cot_multi_0"),
        "oldest card must be excluded by limit=3: {system}"
    );
    assert!(
        !system.contains("card_id=cot_multi_1"),
        "next-oldest card must be excluded by limit=3: {system}"
    );
    let pos_4 = system
        .find("card_id=cot_multi_4")
        .expect("cot_multi_4 must be present");
    let pos_2 = system
        .find("card_id=cot_multi_2")
        .expect("cot_multi_2 must be present");
    assert!(
        pos_4 < pos_2,
        "created_at desc: newest card must appear first (pos_4={pos_4}, pos_2={pos_2})"
    );
}

#[tokio::test]
async fn unset_baseline_unchanged() {
    let mut fx = setup_fixture();
    let script = r#"alc.llm("hi", { system = "S" })"#;
    let req = drive_llm_call(&fx.lua, &mut fx.rx, script).await;

    assert_eq!(
        req.queries[0].system.as_deref(),
        Some("S"),
        "unset card_context must leave system verbatim"
    );
}

#[tokio::test]
async fn not_found_silent_no_op() {
    let mut fx = setup_fixture();
    // Store deliberately left empty.
    let script = r#"alc.llm("hi", { card_context = "nonexistent", system = "S" })"#;
    let req = drive_llm_call(&fx.lua, &mut fx.rx, script).await;

    assert_eq!(
        req.queries[0].system.as_deref(),
        Some("S"),
        "not-found must silently fall through (no LuaError, no prefix)"
    );
}
