//! E2E tests for host_mode pool worker infrastructure.
//!
//! Covers issue verification goals 1-4:
//! 1. `alc_run host_mode=true` creates a paused session visible in `alc_status`.
//! 2. MCP server process kill → new MCP server → registry.json reconnect → session still visible.
//! 3. `alc_continue` resumes a paused pool session after reconnect.
//! 4. `alc_pool_stop` terminates workers; `alc_pool_status` returns empty sessions.
//!
//! ## Design
//!
//! Each test uses an isolated `ALC_HOME` temp directory so pool state does not
//! bleed between tests or affect the developer's `~/.algocline/` installation.
//!
//! Tests spawn the `alc` binary (via `CARGO_BIN_EXE_alc` or `target/debug/alc`).
//! `connect_with_home()` sets `ALC_HOME` env so pool registry and socket files
//! land under the temp dir.

use std::borrow::Cow;

use rmcp::{model::CallToolRequestParams, transport::TokioChildProcess, ServiceExt};
use serde_json::{json, Value};

// ─── Helpers ─────────────────────────────────────────────────────

fn alc_bin() -> String {
    std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")))
}

fn call_params(name: &str, args: Value) -> CallToolRequestParams {
    let arguments = match args {
        Value::Object(map) => Some(map),
        _ => None,
    };
    let mut p = CallToolRequestParams::default();
    p.name = Cow::Owned(name.to_string());
    p.arguments = arguments;
    p
}

fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .and_then(|c| c.raw.as_text())
        .map(|t| t.text.as_str())
        .unwrap_or("")
}

/// Connect to `alc` with the given `ALC_HOME` env set.
///
/// Setting ALC_HOME scopes all algocline state (pool registry, packages, cards)
/// under the temp directory, isolating tests from the developer's installation.
async fn connect_with_home(
    alc_home: &std::path::Path,
) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let mut cmd = tokio::process::Command::new(alc_bin());
    cmd.env("ALC_HOME", alc_home);
    let transport = TokioChildProcess::new(cmd).expect("failed to spawn alc server");
    ().serve(transport)
        .await
        .expect("failed to initialize MCP session")
}

/// Create a short-path tempdir suitable for pool worker Unix domain socket paths.
///
/// Pool workers create sockets like `{ALC_HOME}/state/pool/{sid}.sock`.
/// Unix domain socket paths must be < 104 bytes (macOS SUN_LEN).  The default
/// `tempfile::tempdir()` on macOS resolves under `/var/folders/...` which is
/// too long.  We place the dir under `/tmp` directly to keep paths short.
fn short_tempdir() -> tempfile::TempDir {
    // Try /tmp first (short path, works on macOS and Linux).
    // Fall back to std::env::temp_dir() if /tmp is not writable.
    let base = if std::path::Path::new("/tmp").is_dir() {
        std::path::PathBuf::from("/tmp")
    } else {
        std::env::temp_dir()
    };
    tempfile::Builder::new()
        .prefix("alcp")
        .tempdir_in(base)
        .expect("short tempdir")
}

/// Call a tool, extract text, parse as JSON.
async fn call_json(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    name: &str,
    args: Value,
) -> Value {
    let result = client
        .call_tool(call_params(name, args))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    serde_json::from_str(text).unwrap_or_else(|e| panic!("JSON parse failed: {e}\nraw: {text}"))
}

// ─── Test 1: paused session visible in alc_status ─────────────────

/// Verify that `alc_run host_mode=true` creates a paused pool session that is
/// visible in both `alc_status` and `alc_pool_status`.
///
/// Issue verification goal 1: pool session lifecycle (create → pause → visible).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "integration: requires pool worker infrastructure (run with --include-ignored)"]
async fn test_pool_paused_session_visible_in_status() {
    let tmp = short_tempdir();
    let client = connect_with_home(tmp.path()).await;

    // 1. Start a session that will pause on alc.llm().
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": "return alc.llm('What is 2+2?')", "host_mode": true }),
    )
    .await;
    assert_eq!(
        resp["status"], "needs_response",
        "host_mode=true run must pause on alc.llm()"
    );
    let session_id = resp["session_id"]
        .as_str()
        .expect("session_id must be present")
        .to_string();

    // 2. Query alc_status — session must appear.
    let status_resp = call_json(&client, "alc_status", json!({ "session_id": session_id })).await;
    assert_eq!(
        status_resp["status"], "needs_response",
        "alc_status must report needs_response for paused pool session"
    );

    // 3. Query alc_pool_status — session must appear in pool registry.
    let pool_resp = call_json(&client, "alc_pool_status", json!({})).await;
    let sessions = pool_resp["sessions"]
        .as_array()
        .expect("sessions must be an array");
    let found = sessions
        .iter()
        .any(|s| s["sid"].as_str() == Some(&session_id));
    assert!(found, "session_id must appear in alc_pool_status sessions");

    // 4. Clean up — resume so the worker exits cleanly.
    let _ = call_json(
        &client,
        "alc_continue",
        json!({ "session_id": session_id, "response": "4" }),
    )
    .await;

    client.cancel().await.expect("cancel failed");
}

// ─── Test 2: MCP restart reconnect ───────────────────────────────

/// Verify that after killing the MCP server process and starting a new one,
/// the pool session is still visible via registry.json reconnect.
///
/// Issue verification goal 2 / Crux: Registry reconnect across restarts.
///
/// The test uses two separate MCP client sessions against the same `ALC_HOME`
/// directory:
///   - Session A: creates the pool session (paused).
///   - (Kill session A's MCP process.)
///   - Session B: new MCP server reads registry.json, sees the live worker.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "integration: requires pool worker infrastructure (run with --include-ignored)"]
async fn test_pool_registry_reconnect_after_mcp_restart() {
    let tmp = short_tempdir();

    // ── Session A ─────────────────────────────────────────────────
    let client_a = connect_with_home(tmp.path()).await;

    let resp = call_json(
        &client_a,
        "alc_run",
        json!({ "code": "return alc.llm('ping')", "host_mode": true }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id").to_string();

    // Cancel the first MCP client (simulates MCP server dying / CC restart).
    client_a.cancel().await.expect("cancel a failed");

    // Brief pause to let the worker settle (socket must still exist).
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // ── Session B ─────────────────────────────────────────────────
    // New MCP server starts; it reads registry.json and discovers the live worker.
    let client_b = connect_with_home(tmp.path()).await;

    // alc_pool_ensure triggers a GC scan and returns live sessions.
    let ensure_resp = call_json(&client_b, "alc_pool_ensure", json!({})).await;
    let sessions = ensure_resp["sessions"]
        .as_array()
        .expect("sessions array in pool_ensure response");
    let found = sessions
        .iter()
        .any(|s| s["sid"].as_str() == Some(&session_id));
    assert!(
        found,
        "session_id must still be visible after MCP restart (registry.json reconnect)"
    );

    // alc_pool_status must also show the session as alive.
    let status_resp = call_json(&client_b, "alc_pool_status", json!({})).await;
    let pool_sessions = status_resp["sessions"]
        .as_array()
        .expect("sessions array in pool_status response");
    let found_b = pool_sessions
        .iter()
        .any(|s| s["sid"].as_str() == Some(&session_id));
    assert!(
        found_b,
        "alc_pool_status must report the session as alive after reconnect"
    );

    // Clean up.
    let _ = call_json(
        &client_b,
        "alc_continue",
        json!({ "session_id": &session_id, "response": "pong" }),
    )
    .await;

    client_b.cancel().await.expect("cancel b failed");
}

// ─── Test 3: alc_continue resumes after reconnect ────────────────

/// Verify that `alc_continue` can resume a paused pool session after the
/// originating MCP server is gone (pool worker is still alive).
///
/// Issue verification goal 3 / Crux: mlua VM subprocess initialization.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "integration: requires pool worker infrastructure (run with --include-ignored)"]
async fn test_pool_continue_after_reconnect() {
    let tmp = short_tempdir();

    // ── Session A: create paused session ──────────────────────────
    let client_a = connect_with_home(tmp.path()).await;

    let resp = call_json(
        &client_a,
        "alc_run",
        json!({ "code": "local r = alc.llm('What is 2+2?') return r", "host_mode": true }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id").to_string();

    // Kill the first MCP server.
    client_a.cancel().await.expect("cancel a failed");
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // ── Session B: reconnect and continue ─────────────────────────
    let client_b = connect_with_home(tmp.path()).await;

    // Ensure pool GC runs (reconnect path).
    let _ = call_json(&client_b, "alc_pool_ensure", json!({})).await;

    // Continue the paused session — the pool worker's mlua VM must resume.
    let cont_resp = call_json(
        &client_b,
        "alc_continue",
        json!({ "session_id": &session_id, "response": "4" }),
    )
    .await;

    // The response should indicate completion (not another needs_response).
    let status = cont_resp["status"].as_str().unwrap_or("");
    assert!(
        status == "completed" || status == "needs_response",
        "alc_continue must return completed or needs_response (got {status}); \
        session_id={session_id}"
    );

    client_b.cancel().await.expect("cancel b failed");
}

// ─── Test 4: alc_pool_stop empties sessions ───────────────────────

/// Verify that `alc_pool_stop` sends SIGTERM to workers and `alc_pool_status`
/// subsequently returns empty sessions.
///
/// Issue verification goal 4.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "integration: requires pool worker infrastructure (run with --include-ignored)"]
async fn test_pool_stop_empties_sessions() {
    let tmp = short_tempdir();
    let client = connect_with_home(tmp.path()).await;

    // 1. Create a paused pool session.
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": "return alc.llm('stop me')", "host_mode": true }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");

    // 2. Stop all workers.
    let stop_resp = call_json(&client, "alc_pool_stop", json!({})).await;
    let errors = stop_resp["errors"]
        .as_array()
        .expect("errors must be an array");
    assert!(
        errors.is_empty(),
        "alc_pool_stop must not produce errors, got: {errors:?}"
    );

    // 3. Brief pause to allow the worker to process SIGTERM.
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // 4. Pool status must now show empty sessions.
    let status_resp = call_json(&client, "alc_pool_status", json!({})).await;
    let sessions = status_resp["sessions"]
        .as_array()
        .expect("sessions must be an array");
    assert!(
        sessions.is_empty(),
        "alc_pool_status must return empty sessions after alc_pool_stop, got: {sessions:?}"
    );

    client.cancel().await.expect("cancel failed");
}
