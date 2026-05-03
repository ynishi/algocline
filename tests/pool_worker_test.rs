//! Integration tests for the pool worker subprocess entrypoint.
//!
//! Each test spawns the `alc` binary with the `pool-worker` subcommand,
//! communicates over a Unix-domain socket using `PoolClient`, and verifies
//! the round-trip behaviour.
//!
//! The `alc` binary path is provided by Cargo via `CARGO_BIN_EXE_alc`.
//!
//! # Process lifecycle
//!
//! Every test MUST call `child.kill().await` before returning to avoid
//! zombie processes, even when the test panics.  The guard pattern below
//! ensures this.

use std::path::PathBuf;
use std::time::Duration;

use algocline_app::pool::{PoolClient, PoolRequest, PoolResponseData};
use tempfile::TempDir;
use tokio::process::Command;
use tokio::time::sleep;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Returns a temporary directory and an unused socket path inside it.
fn temp_sock() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let sock = dir.path().join("worker.sock");
    (dir, sock)
}

/// Path to the compiled `alc` binary under test.
fn alc_bin() -> String {
    // CARGO_BIN_EXE_alc is set by Cargo in the integration-test context.
    env!("CARGO_BIN_EXE_alc").to_string()
}

/// Spawn the `alc pool-worker` subprocess.
///
/// Returns a `tokio::process::Child`. The caller is responsible for calling
/// `child.kill().await` when done.
async fn spawn_worker(sid: &str, sock: &std::path::Path) -> tokio::process::Child {
    Command::new(alc_bin())
        .args(["pool-worker", "--sid", sid, "--sock"])
        .arg(sock)
        // Redirect stdout/stderr so test output stays clean.
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn pool-worker subprocess")
}

/// Poll `PoolClient::connect` until the worker binds or the timeout elapses.
///
/// The worker needs a brief moment to bind the socket after `spawn_worker`
/// returns. This helper retries with 10 ms intervals up to `max_wait`.
async fn connect_with_retry(sock: &std::path::Path, max_wait: Duration) -> PoolClient {
    let deadline = tokio::time::Instant::now() + max_wait;
    loop {
        match PoolClient::connect(sock).await {
            Ok(client) => return client,
            Err(_) if tokio::time::Instant::now() < deadline => {
                sleep(Duration::from_millis(10)).await;
            }
            Err(e) => panic!("could not connect to worker within {max_wait:?}: {e}"),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Full lifecycle: handshake (implicit in connect) → run (returns Paused)
/// → continue → shutdown.
///
/// Uses `alc.llm('test prompt')` as the Lua code so the worker pauses on
/// the LLM bridge call, then we resume it with `PoolRequest::Continue`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_subprocess_handshake_run_pause_continue_shutdown() {
    let (_dir, sock_path) = temp_sock();
    let mut child = spawn_worker("test-sid", &sock_path).await;

    // Connect — handshake is performed inside PoolClient::connect.
    let mut client = connect_with_retry(&sock_path, Duration::from_secs(5)).await;

    // Send Run with Lua that will pause on alc.llm().
    let run_resp = client
        .send_request(PoolRequest::Run {
            code: "return alc.llm('test prompt')".to_string(),
            ctx: None,
            lib_paths: vec![],
        })
        .await
        .expect("Run request");

    // The worker should respond with a Feed that reflects a paused session.
    assert!(
        run_resp.ok,
        "Run response should be ok: {:?}",
        run_resp.error
    );
    let session_id = match run_resp.data {
        Some(PoolResponseData::Feed { session_id, .. }) => {
            assert!(
                !session_id.is_empty(),
                "session_id must be non-empty after Run"
            );
            session_id
        }
        other => panic!("expected Feed response from Run, got: {other:?}"),
    };

    // Send Continue to resume the paused session.
    let continue_resp = client
        .send_request(PoolRequest::Continue {
            sid: session_id.clone(),
            response: "LLM answer".to_string(),
            query_id: None,
            usage: None,
        })
        .await
        .expect("Continue request");

    assert!(
        continue_resp.ok,
        "Continue response should be ok: {:?}",
        continue_resp.error
    );
    assert!(
        matches!(continue_resp.data, Some(PoolResponseData::Feed { .. })),
        "expected Feed response from Continue"
    );

    // Send Shutdown — worker should exit gracefully.
    let shutdown_resp = client
        .send_request(PoolRequest::Shutdown)
        .await
        .expect("Shutdown request");

    assert!(
        shutdown_resp.ok,
        "Shutdown response should be ok: {:?}",
        shutdown_resp.error
    );
    assert!(
        matches!(shutdown_resp.data, Some(PoolResponseData::Shutdown)),
        "expected Shutdown response data"
    );

    // Always kill the child to prevent zombie processes.
    // The worker may have already exited after Shutdown; kill() is idempotent.
    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// Status before any Run → idle.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_status_idle_before_run() {
    let (_dir, sock_path) = temp_sock();
    let mut child = spawn_worker("idle-test-sid", &sock_path).await;

    let mut client = connect_with_retry(&sock_path, Duration::from_secs(5)).await;

    let status_resp = client
        .send_request(PoolRequest::Status)
        .await
        .expect("Status request");

    assert!(status_resp.ok, "Status should be ok");
    match status_resp.data {
        Some(PoolResponseData::Status {
            has_session,
            session_id,
        }) => {
            assert!(!has_session, "worker should have no session before Run");
            assert!(session_id.is_none(), "session_id should be None before Run");
        }
        other => panic!("expected Status response, got: {other:?}"),
    }

    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// Duplicate Run is rejected after the first Run is in-flight.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_duplicate_run() {
    let (_dir, sock_path) = temp_sock();
    let mut child = spawn_worker("dup-run-sid", &sock_path).await;

    let mut client = connect_with_retry(&sock_path, Duration::from_secs(5)).await;

    // First Run — pauses on alc.llm().
    let run1 = client
        .send_request(PoolRequest::Run {
            code: "return alc.llm('q')".to_string(),
            ctx: None,
            lib_paths: vec![],
        })
        .await
        .expect("first Run");

    assert!(run1.ok, "first Run should succeed");

    // Second Run — must be rejected.
    let run2 = client
        .send_request(PoolRequest::Run {
            code: "return 1".to_string(),
            ctx: None,
            lib_paths: vec![],
        })
        .await
        .expect("second Run request");

    assert!(!run2.ok, "second Run must be rejected");
    assert!(
        run2.error.as_deref().is_some_and(|e| e.contains("already")),
        "error should mention 'already': {:?}",
        run2.error
    );

    let _ = child.kill().await;
    let _ = child.wait().await;
}
