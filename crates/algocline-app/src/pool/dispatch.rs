//! Host-mode dispatch helpers.
//!
//! Provides [`run_via_pool`] and [`continue_via_pool`] which orchestrate
//! worker subprocess spawning, UDS connection, registry persistence, and
//! response forwarding.
//!
//! ## Crux invariants enforced here
//!
//! - (MCP thin proxy IPC boundary): all host_mode=true calls reach a worker
//!   subprocess via [`PoolClient`] over a Unix domain socket.  The MCP-internal
//!   [`SessionRegistry`] is never touched on this path.
//! - (Registry reconnect across restarts): every successful [`run_via_pool`]
//!   call persists the session entry to `registry.json` via
//!   [`with_registry_lock`] before returning.  A restarted MCP process can
//!   rediscover live workers from that file.

use std::path::{Path, PathBuf};

use tokio::time::{sleep, timeout, Duration};

use crate::pool::{
    registry::with_registry_lock, PoolClient, PoolError, PoolRegistry, PoolRequest, PoolResponse,
    PoolResponseData, PoolSessionEntry,
};

// ─── Worker spawn ─────────────────────────────────────────────────────────────

/// Spawn a pool worker subprocess for the given session.
///
/// # Arguments
///
/// * `pool_dir` — directory for UDS sockets and the registry lock file.
/// * `reg_path` — path to `registry.json`.
/// * `lock_path` — path to the advisory lock sentinel file.
/// * `sid` — session ID string (UUID).
///
/// # Returns
///
/// `Ok((pid, sock_path))` once the worker socket file is visible.
///
/// # Errors
///
/// - `PoolError::Spawn` — `current_exe()` failed or `Command::spawn()` failed.
/// - `PoolError::Handshake` — the socket file did not appear within 5 seconds.
fn spawn_worker(pool_dir: &Path, sid: &str) -> Result<(u32, PathBuf), PoolError> {
    let sock = pool_dir.join(format!("{sid}.sock"));

    let exe = std::env::current_exe()
        .map_err(|e| PoolError::Spawn(format!("current_exe failed: {e}")))?;

    let mut cmd = std::process::Command::new(&exe);
    // "pool-worker" is a clap subcommand (kebab-case of PoolWorker enum variant).
    // The worker binary must be invoked as `alc pool-worker --sid <sid> --sock <path>`.
    cmd.args([
        "pool-worker",
        "--sid",
        sid,
        "--sock",
        &sock.to_string_lossy(),
    ]);

    // SAFETY: pre_exec runs in the child process after fork, before exec.
    // `libc::setsid()` is an async-signal-safe syscall.  No allocator,
    // mutex, or tokio runtime is used inside the closure.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                // Detach the child from the parent's session so that
                // parent (MCP server) death does not deliver SIGHUP to
                // the worker.
                libc::setsid();
                Ok(())
            });
        }
    }

    let child = cmd
        .spawn()
        .map_err(|e| PoolError::Spawn(format!("worker spawn failed: {e}")))?;

    // Record the PID immediately, before any wait() can reap the child.
    let pid = child.id();
    // Drop the Child handle — the worker continues running independently
    // (std::process::Child drop does not send any signal; the process
    // continues as a detached session leader after setsid()).
    drop(child);

    Ok((pid, sock))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Generate a non-deterministic session ID for pool workers.
///
/// Uses timestamp + random bytes, matching the approach in
/// `algocline-engine::session::gen_session_id`.
fn gen_pool_sid() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Timestamp prefix for rough ordering.  `unwrap_or_default` is
    // justified: UNIX_EPOCH underflow only happens on misconfigured clocks
    // where the timestamp suffix is 0.  The 16-hex-char random suffix alone
    // guarantees uniqueness.
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let random: u64 = {
        let s = RandomState::new();
        let mut h = s.build_hasher();
        h.write_u128(ts);
        h.finish()
    };
    format!("p-{ts:x}-{random:016x}")
}

/// Spawn a worker subprocess, connect via UDS, proxy a `Run` request,
/// and persist the session entry to `registry.json`.
///
/// On success, returns `(session_id, json_response, Option<pool_save_error>)`.
/// The optional `pool_save_error` is `Some(msg)` when the registry write
/// failed — callers must surface this as an additive field on the MCP wire
/// response (CLAUDE.md §Service 層の Error 伝播規律: write failures must not
/// be silently dropped).
///
/// # Arguments
///
/// * `pool_dir` — directory containing the socket files and registry files.
/// * `reg_path` — path to `registry.json`.
/// * `lock_path` — path to the advisory lock sentinel.
/// * `extra_lib_paths` — additional Lua library search paths.
/// * `code` — Lua source code to run.
/// * `ctx` — JSON context passed as `alc.ctx`.
///
/// # Errors
///
/// Returns `Err(PoolError)` if worker spawn, socket wait, or UDS run
/// round-trip fails.  Registry persistence failure is returned in the
/// `pool_save_error` field, not as an `Err`.
///
/// # Concurrency
///
/// Acquires `RwLock<PoolRegistry>` (write) + advisory `fs4` file lock for
/// the registry add/save.  These locks are released before returning.
pub async fn run_via_pool(
    pool_dir: &Path,
    reg_path: &Path,
    lock_path: &Path,
    extra_lib_paths: Vec<PathBuf>,
    code: String,
    ctx: serde_json::Value,
) -> Result<(String, String, Option<String>), PoolError> {
    // 1. Generate session ID and socket path.
    let sid = gen_pool_sid();

    // 2. Spawn worker subprocess (synchronous — Command::spawn is not async).
    let (pid, sock) = spawn_worker(pool_dir, &sid)?;

    // 3. Poll for socket file appearance (timeout 5s).
    {
        let sock_clone = sock.clone();
        timeout(Duration::from_secs(5), async {
            loop {
                if sock_clone.exists() {
                    break;
                }
                sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .map_err(|_| {
            PoolError::Handshake(format!(
                "timeout waiting for worker socket at {}",
                sock.display()
            ))
        })?;
    }

    // 4. Connect (includes handshake) and send Run request.
    // Function-local PoolClient: connect → send → drop.
    // Cancel safety: if this future is cancelled, the PoolClient is dropped
    // and the connection is closed.  The next alc_continue call reconnects
    // via registry.json.
    let mut client = PoolClient::connect(&sock).await?;
    let resp = client
        .send_request(PoolRequest::Run {
            code,
            ctx: Some(ctx),
            lib_paths: extra_lib_paths,
        })
        .await?;

    let (worker_sid, feed_result_json) = extract_feed_response(resp)?;

    // Convert the serde-format FeedResult to MCP wire JSON.
    // The worker serializes FeedResult with standard serde ({"Paused": {...}})
    // but callers expect the MCP format ({"status": "needs_response", ...}).
    let mcp_json = feed_result_to_mcp_json(&worker_sid, &feed_result_json);

    // 5. Persist the session entry to registry.json.
    // Write failures are non-fatal: the session was already started.
    // Callers surface the error as `pool_save_error` on the wire response.
    let pool_save_error = persist_entry(
        reg_path,
        lock_path,
        PoolSessionEntry::new(&worker_sid, pid, sock, env!("CARGO_PKG_VERSION")),
    );

    Ok((worker_sid, mcp_json.to_string(), pool_save_error))
}

/// Reconnect to an existing pool worker via its registry entry and forward a
/// `Continue` request.
///
/// # Arguments
///
/// * `entry` — the registry entry for the target worker (sock path, etc.).
/// * `sid` — session ID to resume.
/// * `response` — LLM response text.
/// * `query_id` — optional query ID being answered.
/// * `usage` — optional token usage.
///
/// # Returns
///
/// `Ok(json_response)` — the stringified `feed_result` JSON from the worker.
///
/// # Errors
///
/// Returns `Err(PoolError)` on UDS connect failure or invalid response.
///
/// # Cancel safety
///
/// If this future is cancelled mid-await, the `PoolClient` is dropped and
/// the socket is closed.  `read_line` is not cancel-safe; a partial line
/// would corrupt the buffer.  Dropping the client is the correct recovery —
/// the next `continue_via_pool` call creates a fresh connection.
pub async fn continue_via_pool(
    entry: &PoolSessionEntry,
    sid: &str,
    response: String,
    query_id: Option<String>,
    usage: Option<algocline_core::TokenUsage>,
) -> Result<String, PoolError> {
    // Function-local PoolClient: connect → send → drop.
    let mut client = PoolClient::connect(&entry.sock).await?;
    let resp = client
        .send_request(PoolRequest::Continue {
            sid: sid.to_string(),
            response,
            query_id,
            usage,
        })
        .await?;

    let (session_id, feed_result_json) = extract_feed_response(resp)?;
    // Convert serde-format FeedResult to MCP wire JSON (same translation as run_via_pool).
    let mcp_json = feed_result_to_mcp_json(&session_id, &feed_result_json);
    Ok(mcp_json.to_string())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Convert a worker `FeedResult` serde-JSON value to the MCP wire format.
///
/// The worker serializes `FeedResult` with standard serde:
/// - `FeedResult::Paused`   → `{"Paused": {"queries": [...]}}`
/// - `FeedResult::Finished` → `{"Finished": {"state": ..., "metrics": ...}}`
/// - `FeedResult::Accepted` → `{"Accepted": {"remaining": N}}`
///
/// This mirrors the logic of `algocline_engine::FeedResult::to_json` without
/// requiring `FeedResult` to implement `Deserialize`.  The worker's serde
/// output is the authoritative source.
///
/// # Arguments
///
/// * `session_id` — the pool worker's session ID (embedded in `needs_response`).
/// * `feed_result` — the raw serde value from `PoolResponseData::Feed.feed_result`.
///
/// # Returns
///
/// MCP wire JSON.  On unrecognised shapes, falls back to
/// `{"status": "error", "error": "..."}`.
fn feed_result_to_mcp_json(session_id: &str, feed_result: &serde_json::Value) -> serde_json::Value {
    use serde_json::json;

    if let Some(paused) = feed_result.get("Paused") {
        let queries = paused.get("queries").and_then(|q| q.as_array());
        match queries {
            Some(qs) if qs.len() == 1 => {
                let q = &qs[0];
                let mut obj = json!({
                    "status": "needs_response",
                    "session_id": session_id,
                    "query_id": q.get("id").and_then(|v| v.as_str()).unwrap_or("q-0"),
                    "prompt": q.get("prompt").cloned().unwrap_or(serde_json::Value::Null),
                    "system": q.get("system").cloned().unwrap_or(serde_json::Value::Null),
                    "max_tokens": q.get("max_tokens").cloned().unwrap_or(json!(1024)),
                });
                if q.get("grounded").and_then(|v| v.as_bool()).unwrap_or(false) {
                    obj["grounded"] = json!(true);
                }
                if q.get("underspecified")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    obj["underspecified"] = json!(true);
                }
                obj
            }
            Some(qs) => {
                let mapped: Vec<serde_json::Value> = qs
                    .iter()
                    .map(|q| {
                        let mut obj = json!({
                            "id": q.get("id").cloned().unwrap_or(json!("q-0")),
                            "prompt": q.get("prompt").cloned().unwrap_or(serde_json::Value::Null),
                            "system": q.get("system").cloned().unwrap_or(serde_json::Value::Null),
                            "max_tokens": q.get("max_tokens").cloned().unwrap_or(json!(1024)),
                        });
                        if q.get("grounded").and_then(|v| v.as_bool()).unwrap_or(false) {
                            obj["grounded"] = json!(true);
                        }
                        if q.get("underspecified")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            obj["underspecified"] = json!(true);
                        }
                        obj
                    })
                    .collect();
                json!({
                    "status": "needs_response",
                    "session_id": session_id,
                    "queries": mapped,
                })
            }
            None => json!({
                "status": "needs_response",
                "session_id": session_id,
            }),
        }
    } else if let Some(finished) = feed_result.get("Finished") {
        let state = finished.get("state");
        let metrics = finished.get("metrics");
        if let Some(completed) = state.and_then(|s| s.get("Completed")) {
            json!({
                "status": "completed",
                "result": completed.get("result").cloned().unwrap_or(serde_json::Value::Null),
                "stats": metrics.cloned().unwrap_or(serde_json::Value::Null),
            })
        } else if let Some(failed) = state.and_then(|s| s.get("Failed")) {
            json!({
                "status": "error",
                "error": failed.get("error").and_then(|v| v.as_str()).unwrap_or("execution failed"),
            })
        } else {
            json!({
                "status": "cancelled",
                "stats": metrics.cloned().unwrap_or(serde_json::Value::Null),
            })
        }
    } else if let Some(accepted) = feed_result.get("Accepted") {
        json!({
            "status": "accepted",
            "remaining": accepted.get("remaining").cloned().unwrap_or(json!(0)),
        })
    } else {
        // Unrecognised shape — surface as error so callers can observe it.
        json!({
            "status": "error",
            "error": format!("unrecognised FeedResult shape from worker: {feed_result}"),
        })
    }
}

/// Extract `(session_id, feed_result)` from a `PoolResponse`.
///
/// # Errors
///
/// Returns `PoolError::ResponseParse` if the response is not a `Feed` variant.
fn extract_feed_response(resp: PoolResponse) -> Result<(String, serde_json::Value), PoolError> {
    match resp.data {
        Some(PoolResponseData::Feed {
            session_id,
            feed_result,
        }) => Ok((session_id, feed_result)),
        Some(other) => Err(PoolError::ResponseParse(format!(
            "expected Feed response, got {other:?}"
        ))),
        None => {
            let err = resp.error.unwrap_or_else(|| "unknown error".to_string());
            Err(PoolError::ResponseParse(format!(
                "worker returned error: {err}"
            )))
        }
    }
}

/// Persist a session entry to `registry.json` under the advisory lock.
///
/// Returns `Some(error_message)` if the persist failed, `None` on success.
/// This never returns `Err` — registry write failures are surfaced as
/// additive `pool_save_error` fields on the MCP response, not as hard errors,
/// because the session was already started successfully.
fn persist_entry(reg_path: &Path, lock_path: &Path, entry: PoolSessionEntry) -> Option<String> {
    // Registry I/O is synchronous; called from async context but we do NOT
    // use spawn_blocking here because:
    // (a) this is a brief lock + read + write cycle (~1ms on SSD),
    // (b) the lock is advisory so no OS-level blocking occurs unless another
    //     MCP process is simultaneously writing, which is rare,
    // (c) adding spawn_blocking would require cloning paths and returning a
    //     JoinHandle, complicating the caller without meaningful latency benefit.
    match with_registry_lock(lock_path, || {
        let mut reg = PoolRegistry::load_or_default(reg_path)?;
        reg.add(entry);
        reg.save(reg_path)
    }) {
        Ok(()) => None,
        Err(e) => Some(e.to_string()),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::pool::{protocol::PoolResponseData, PoolResponse};

    // ── T1: happy path ────────────────────────────────────────────────────────

    /// T1 — extract_feed_response returns Ok for a valid Feed response.
    #[test]
    fn extract_feed_response_ok_on_feed() {
        let resp = PoolResponse::success(PoolResponseData::Feed {
            session_id: "test-sid".to_string(),
            feed_result: serde_json::json!({"status": "needs_response"}),
        });
        let (sid, json) = extract_feed_response(resp).expect("should extract feed");
        assert_eq!(sid, "test-sid");
        assert_eq!(json["status"], "needs_response");
    }

    // ── T2: boundary / edge ───────────────────────────────────────────────────

    /// T2 — gen_pool_sid returns distinct IDs on successive calls.
    #[test]
    fn gen_pool_sid_is_unique() {
        let ids: Vec<_> = (0..20).map(|_| gen_pool_sid()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "all generated session IDs must be distinct"
        );
    }

    /// T2b — gen_pool_sid has expected prefix.
    #[test]
    fn gen_pool_sid_has_prefix() {
        let sid = gen_pool_sid();
        assert!(sid.starts_with("p-"), "sid must start with 'p-', got {sid}");
    }

    // ── T3: error path ────────────────────────────────────────────────────────

    /// T3 — extract_feed_response returns ResponseParse error for a non-Feed variant.
    #[test]
    fn extract_feed_response_error_on_non_feed() {
        let resp = PoolResponse::success(PoolResponseData::Shutdown);
        let err = extract_feed_response(resp).expect_err("should fail on Shutdown response");
        assert!(
            matches!(err, PoolError::ResponseParse(_)),
            "expected ResponseParse, got {err:?}"
        );
    }

    /// T3b — extract_feed_response propagates the worker error message when ok=false.
    #[test]
    fn extract_feed_response_error_on_failure_response() {
        let resp = PoolResponse::failure("something went wrong");
        let err = extract_feed_response(resp).expect_err("should fail on error response");
        match err {
            PoolError::ResponseParse(msg) => {
                assert!(
                    msg.contains("something went wrong"),
                    "error must include worker message, got: {msg}"
                );
            }
            other => panic!("expected ResponseParse, got {other:?}"),
        }
    }

    /// T3c — persist_entry returns Some(error) when the lock path parent
    ///        directory cannot be created (permission denied simulation via
    ///        using a path under a file, not a directory).
    #[test]
    fn persist_entry_returns_some_on_io_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Place a regular file where the parent directory is expected.
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"not a dir").expect("write blocker");
        let reg_path = blocker.join("registry.json"); // parent is a file
        let lock_path = blocker.join("registry.lock");

        let entry = PoolSessionEntry::new(
            "test-sid",
            std::process::id(),
            PathBuf::from("/tmp/test.sock"),
            "0.30.0",
        );

        let result = persist_entry(&reg_path, &lock_path, entry);
        assert!(
            result.is_some(),
            "persist_entry must return Some(error) on I/O failure"
        );
    }

    // ── T1: feed_result_to_mcp_json happy paths ───────────────────────────────

    /// T1 — Paused with one query maps to needs_response with session_id.
    #[test]
    fn feed_result_to_mcp_json_paused_single_query() {
        let feed = serde_json::json!({
            "Paused": {
                "queries": [{
                    "id": "q-0",
                    "prompt": "What is 1+1?",
                    "system": null,
                    "max_tokens": 1024,
                    "grounded": false,
                    "underspecified": false
                }]
            }
        });
        let mcp = feed_result_to_mcp_json("sid-abc", &feed);
        assert_eq!(mcp["status"], "needs_response");
        assert_eq!(mcp["session_id"], "sid-abc");
        assert_eq!(mcp["query_id"], "q-0");
        assert_eq!(mcp["prompt"], "What is 1+1?");
    }

    /// T1b — Finished(Completed) maps to completed with result.
    #[test]
    fn feed_result_to_mcp_json_finished_completed() {
        let feed = serde_json::json!({
            "Finished": {
                "state": {
                    "Completed": { "result": {"answer": 42} }
                },
                "metrics": {}
            }
        });
        let mcp = feed_result_to_mcp_json("sid-xyz", &feed);
        assert_eq!(mcp["status"], "completed");
        assert_eq!(mcp["result"]["answer"], 42);
    }

    /// T2 — Paused with multiple queries maps to queries array.
    #[test]
    fn feed_result_to_mcp_json_paused_multi_query() {
        let feed = serde_json::json!({
            "Paused": {
                "queries": [
                    {"id": "q-0", "prompt": "P1", "system": null, "max_tokens": 512},
                    {"id": "q-1", "prompt": "P2", "system": null, "max_tokens": 512}
                ]
            }
        });
        let mcp = feed_result_to_mcp_json("sid-multi", &feed);
        assert_eq!(mcp["status"], "needs_response");
        assert_eq!(mcp["session_id"], "sid-multi");
        let qs = mcp["queries"].as_array().expect("queries array");
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0]["id"], "q-0");
    }

    /// T3 — Unrecognised FeedResult shape maps to error status.
    #[test]
    fn feed_result_to_mcp_json_unknown_shape_is_error() {
        let feed = serde_json::json!({"Unknown": {}});
        let mcp = feed_result_to_mcp_json("sid-bad", &feed);
        assert_eq!(mcp["status"], "error");
        assert!(
            mcp["error"].as_str().unwrap_or("").contains("unrecognised"),
            "error message must mention 'unrecognised', got: {}",
            mcp["error"]
        );
    }
}
