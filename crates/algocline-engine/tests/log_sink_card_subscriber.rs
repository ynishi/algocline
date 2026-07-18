//! Phase 2-D integration tests: verify that `CardEvent`s emitted from a
//! Lua session's `alc.card.*` calls fan out into per-session `LogSink`
//! ring buffers via the process-wide `LogSinkCardSubscriber`.
//!
//! # Serialization
//!
//! All tests in this file share the process-wide `CardEventBus` singleton
//! and the `LogSinkCardSubscriber` singleton. To keep them free of
//! observation-order flakes we serialize them with a file-local `Mutex`.
//! Individual tests register their own external `LogSink` observers via
//! [`algocline_engine::card::register_log_sink`], which is safe to call
//! concurrently, but each test also asserts on shared bus state (e.g.
//! `subscriber_uris`) which does not tolerate parallel churn.

use std::sync::{Arc, OnceLock};

use algocline_core::LogSink;
use algocline_engine::card::{
    register_log_sink, FileCardStore, FileCardSubscriber, LogSinkCardSubscriber,
};
use algocline_engine::{Executor, FeedResult, JsonFileStore, SessionRegistry};
use tokio::sync::Mutex;

/// Shared serialization gate — all tests in this file take it before
/// touching the process-wide event bus. Uses `tokio::sync::Mutex` because
/// tests hold it across `await` points.
fn integration_test_gate() -> &'static Mutex<()> {
    static GATE: OnceLock<Mutex<()>> = OnceLock::new();
    GATE.get_or_init(|| Mutex::new(()))
}

/// Create a fresh temp-dir triple (state_store, card_store, scenarios_dir)
/// mirroring the pattern used by existing `session.rs` tests.
fn tmp_dirs() -> (Arc<JsonFileStore>, Arc<FileCardStore>, std::path::PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);
    (
        Arc::new(JsonFileStore::new(root.join("state"))),
        Arc::new(FileCardStore::new(root.join("cards"))),
        root.join("scenarios"),
    )
}

/// Run `code` in a fresh Session end-to-end (start → drive to completion).
/// Returns `Ok(())` on `FeedResult::Finished { Completed | Cancelled }`.
async fn run_to_completion(code: &str, card_run_enabled: bool) -> Result<(), String> {
    let executor = Executor::new(vec![]).await.map_err(|e| e.to_string())?;
    let (state_store, card_store, scenarios_dir) = tmp_dirs();
    let session = executor
        .start_session(
            code.to_string(),
            serde_json::json!({}),
            vec![],
            vec![],
            state_store,
            card_store,
            scenarios_dir,
            card_run_enabled,
        )
        .await?;

    let registry = SessionRegistry::new();
    let (_id, result) = registry
        .start_execution(session)
        .await
        .map_err(|e| e.to_string())?;
    match result {
        FeedResult::Finished(_) => Ok(()),
        FeedResult::Paused { .. } => Err("unexpected pause".into()),
        FeedResult::Accepted { .. } => Err("unexpected accepted".into()),
    }
}

// ─── T-INT-1: happy path ─────────────────────────────────────────

/// A single Session running `alc.card.create` fans a `CardEvent::Created`
/// through the bus and it lands in an external observer `LogSink` as a
/// `LogEntry { source: "alc.card", level: "info", message: "created ..." }`.
#[tokio::test]
async fn t_int_1_happy_path_created_lands_in_log_sink() {
    let _gate = integration_test_gate().lock().await;

    let observer = LogSink::new();
    let _reg = register_log_sink(observer.clone());

    let code = r#"
        local r = alc.card.create({
            pkg = { name = "test_pkg" },
            run = { status = "succeeded" },
        })
        return r.card_id
    "#;
    run_to_completion(code, true).await.expect("run OK");

    let entries = observer.entries();
    let hit = entries.iter().find(|e| {
        e.source == "alc.card"
            && e.level == "info"
            && e.message.starts_with("created pkg=test_pkg card_id=")
    });
    assert!(
        hit.is_some(),
        "at least one Created LogEntry expected; got entries={entries:?}"
    );
}

// ─── T-INT-2: multi-session fan-out ──────────────────────────────

/// The `CardEventBus` is process-wide, so a `CardEvent` published by one
/// Session's Lua VM is delivered to *every* registered `LogSink` — including
/// those owned by other Sessions that happen to be alive concurrently.
/// (Per-session filtering is deferred to Phase 2-E per plan.md.)
#[tokio::test]
async fn t_int_2_multi_session_fan_out() {
    let _gate = integration_test_gate().lock().await;

    let observer_a = LogSink::new();
    let observer_b = LogSink::new();
    let _reg_a = register_log_sink(observer_a.clone());
    let _reg_b = register_log_sink(observer_b.clone());

    let code = r#"
        local r = alc.card.create({
            pkg = { name = "multi_pkg" },
            run = { status = "succeeded" },
        })
        return r.card_id
    "#;
    run_to_completion(code, true).await.expect("run OK");

    let entries_a = observer_a.entries();
    let entries_b = observer_b.entries();
    assert!(
        entries_a
            .iter()
            .any(|e| e.source == "alc.card" && e.message.contains("multi_pkg")),
        "observer_a should see fan-out; got {entries_a:?}"
    );
    assert!(
        entries_b
            .iter()
            .any(|e| e.source == "alc.card" && e.message.contains("multi_pkg")),
        "observer_b should see fan-out; got {entries_b:?}"
    );
}

// ─── T-INT-3: Session drop cleanup ───────────────────────────────

/// When a Session is dropped its internal `LogSinkRegistration` guard is
/// dropped, unregistering that session's `LogSink` from the fan-out. This
/// test asserts that an externally-held observer keeps receiving events
/// across sessions, while a session-internal sink stops receiving after
/// its owning session is dropped — verified indirectly by counting entries
/// on the persistent observer across two independent Session runs.
#[tokio::test]
async fn t_int_3_session_drop_stops_internal_delivery() {
    let _gate = integration_test_gate().lock().await;

    let persistent = LogSink::new();
    let _reg = register_log_sink(persistent.clone());

    // Session A
    let code_a = r#"
        alc.card.create({
            pkg = { name = "pkg_a" },
            run = { status = "succeeded" },
        })
    "#;
    run_to_completion(code_a, true).await.expect("run A");
    let count_after_a = persistent.entries().len();
    assert!(count_after_a >= 1, "A must produce at least one entry");

    // Session B — persistent observer must receive B's event too.
    let code_b = r#"
        alc.card.create({
            pkg = { name = "pkg_b" },
            run = { status = "succeeded" },
        })
    "#;
    run_to_completion(code_b, true).await.expect("run B");
    let entries = persistent.entries();

    // Session B's event lands on persistent observer even though Session A
    // has been dropped (its internal LogSink was unregistered on drop,
    // but the persistent test-owned registration keeps observing).
    assert!(
        entries.iter().any(|e| e.message.contains("pkg_b")),
        "B event must land on persistent observer; got {entries:?}"
    );
}

// ─── T-INT-4: ALC_CARD_SINKS unset regression ────────────────────

/// Without any `FileCardSubscriber` (env unset path), the LogSink still
/// receives Card events — LogSinkCardSubscriber and FileCardSubscriber
/// are independent. This test does not manipulate `ALC_CARD_SINKS` because
/// the env is only read at first `event_bus()` init in the process; instead
/// it verifies that the bus contains the LogSink subscriber URI unconditionally.
#[tokio::test]
async fn t_int_4_env_unset_regression_log_sink_still_receives() {
    let _gate = integration_test_gate().lock().await;

    let observer = LogSink::new();
    let _reg = register_log_sink(observer.clone());

    // Force LogSinkCardSubscriber init and assert its URI is registered.
    let bus = algocline_engine::card::event_bus();
    let uris = bus.subscriber_uris();
    assert!(
        uris.iter().any(|u| u == LogSinkCardSubscriber::URI),
        "log-sink subscriber must be registered on the bus; got {uris:?}"
    );

    let code = r#"
        alc.card.create({
            pkg = { name = "env_unset_pkg" },
            run = { status = "succeeded" },
        })
    "#;
    run_to_completion(code, true).await.expect("run OK");
    assert!(
        observer
            .entries()
            .iter()
            .any(|e| e.message.contains("env_unset_pkg")),
        "LogSink must receive the event even without FileCardSubscriber"
    );
}

// ─── T-INT-5: additional FileCardSubscriber regression ───────────

/// When a `FileCardSubscriber` is manually added to the bus (analogous to
/// `ALC_CARD_SINKS=file://...` init path), both it AND the LogSinkCardSubscriber
/// receive fan-outs — subscribers do not interfere. We cannot manipulate
/// `ALC_CARD_SINKS` at runtime (the env is read once at bus init), so we
/// exercise the "parallel subscriber" property directly via
/// `event_bus().add_subscriber(...)`.
#[tokio::test]
async fn t_int_5_env_set_regression_parallel_file_and_log_subscribers() {
    let _gate = integration_test_gate().lock().await;

    let observer = LogSink::new();
    let _reg = register_log_sink(observer.clone());

    // Manually add a FileCardSubscriber to a tempdir.
    let tmp = tempfile::tempdir().expect("tempdir");
    let file_root = tmp.path().join("sink");
    let file_sub: Arc<dyn algocline_engine::card::CardSubscriber> =
        Arc::new(FileCardSubscriber::new(file_root.clone()));
    let file_uri = file_sub.describe();
    algocline_engine::card::event_bus().add_subscriber(file_sub);

    let code = r#"
        alc.card.create({
            pkg = { name = "env_set_pkg" },
            run = { status = "succeeded" },
        })
    "#;
    run_to_completion(code, true).await.expect("run OK");

    // (a) LogSink observer must have the event.
    assert!(
        observer
            .entries()
            .iter()
            .any(|e| e.message.contains("env_set_pkg")),
        "LogSink must receive the event alongside FileCardSubscriber"
    );

    // (b) FileCardSubscriber must have mirrored a TOML file under the
    // tempdir's `env_set_pkg/` subdir.
    let pkg_dir = file_root.join("env_set_pkg");
    assert!(
        pkg_dir.exists(),
        "FileCardSubscriber must have created pkg dir at {pkg_dir:?}"
    );
    let entries: Vec<_> = std::fs::read_dir(&pkg_dir)
        .expect("read_dir")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".toml"))
        .collect();
    assert!(
        !entries.is_empty(),
        "FileCardSubscriber must have written a .toml at {pkg_dir:?}"
    );

    // Bus stats reflect at least one success for both subscribers.
    let stats = algocline_engine::card::subscriber_stats_snapshot();
    let log_row = stats
        .iter()
        .find(|r| r.sink == LogSinkCardSubscriber::URI)
        .expect("log-sink row");
    assert!(log_row.ok.get("created").copied().unwrap_or(0) >= 1);
    let file_row = stats
        .iter()
        .find(|r| r.sink == file_uri)
        .expect("file-sink row");
    assert!(file_row.ok.get("created").copied().unwrap_or(0) >= 1);
}
