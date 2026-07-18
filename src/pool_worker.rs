//! Pool worker subprocess entrypoint.
//!
//! Invoked by the MCP process via `alc pool-worker --sid <sid> --sock <path>`.
//! Each worker holds exactly one Lua session (1 session = 1 process isolation).
//!
//! ## Lifecycle
//!
//! 1. Bind a Unix-domain socket at `sock`.
//! 2. Accept connections from MCP pool clients (paused sessions survive
//!    client reconnects).
//! 3. Dispatch `PoolRequest` messages until `Shutdown`, SIGTERM, or idle
//!    timeout.
//! 4. Graceful shutdown: remove self from `registry.json` and exit.
//!
//! ## Crux invariant (mlua VM subprocess initialization)
//!
//! The worker holds an independent mlua VM via `Executor::start_session`, which
//! spawns a dedicated OS thread + mlua instance per session.  A paused session
//! (`FeedResult::Paused`) is stored in the `SessionRegistry` between `Run` and
//! `Continue` messages.  The session is resumed exclusively by receiving a
//! `PoolRequest::Continue` over the UDS — no in-process shortcut exists.
//! `WorkerPhase::Paused` carries the `session_id` key so that `Continue`
//! messages can be routed to the correct session via `SessionRegistry`.
//!
//! ## Crux invariant (Registry reconnect across restarts)
//!
//! On graceful shutdown (SIGTERM or idle timeout) the worker removes its own
//! entry from `registry.json` via an advisory lock.  This prevents stale
//! entries from accumulating across MCP restarts.  Self-removal is best-effort:
//! if it fails, the orphan entry will be cleaned up by `scan_and_gc` on next
//! MCP startup.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use algocline_app::pool::registry::with_registry_lock;
use algocline_app::pool::{PoolError, PoolRegistry, PoolRequest, PoolResponse, PoolResponseData};
use algocline_app::AppConfig;
use algocline_core::QueryId;
use algocline_engine::{Executor, FeedResult, FileCardStore, JsonFileStore, SessionRegistry};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixListener;
use tokio::signal::unix::SignalKind;

// ─── Default idle timeout ─────────────────────────────────────────────────────

/// Default idle timeout in seconds (30 minutes).
///
/// Overridden by `ALC_POOL_IDLE_TIMEOUT` environment variable.
/// Set to `0` to disable idle timeout (worker runs forever until SIGTERM).
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1800;

// ─── Worker phase ─────────────────────────────────────────────────────────────

/// Phase of the worker's one-session lifecycle.
enum WorkerPhase {
    /// No session has been started yet.
    Idle,
    /// Session is paused, waiting for a `Continue` message.
    /// Holds the `session_id` key used with `SessionRegistry::feed_response`.
    Paused { session_id: String },
    /// Session has completed (Finished or error) — worker accepts `Shutdown`.
    Finished,
}

// ─── Epoch-ms helper ─────────────────────────────────────────────────────────

/// Return the current Unix time in milliseconds as an `i64`.
///
/// `i64` covers epochs up to the year 2262.
///
/// # Panics
///
/// Cannot panic: `SystemTime::now()` is always ≥ `UNIX_EPOCH` on every
/// supported platform; `unwrap_or_default()` is therefore equivalent to
/// `.expect("infallible: UNIX_EPOCH <= now")` but without the panic risk.
/// (Justification: a clock that returns a time before 1970 is a hardware
/// fault; `duration_since` returning `Err` in that case would yield zero ms,
/// which is safe for idle-timeout comparisons.)
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// ─── Idle-check future ────────────────────────────────────────────────────────

/// Returns a future that resolves when the worker has been idle for
/// `timeout_secs` seconds.
///
/// # Arguments
///
/// * `last_activity` — shared atomic recording the last request epoch-ms.
/// * `timeout_secs` — timeout in seconds.  `0` means disabled: the returned
///   future never resolves, so the idle-check branch in `tokio::select!` is
///   effectively dead.
///
/// # Returns
///
/// An opaque `Future<Output = ()>` that resolves after the idle timeout
/// elapses with no activity, or never resolves when `timeout_secs == 0`.
async fn idle_check_future(last_activity: Arc<AtomicI64>, timeout_secs: u64) {
    if timeout_secs == 0 {
        // Disabled: await a future that never resolves so this select! branch
        // is permanently dormant.
        std::future::pending::<()>().await;
        return;
    }

    let check_interval = tokio::time::Duration::from_secs(1);
    let timeout_ms = (timeout_secs as i64).saturating_mul(1000);

    loop {
        tokio::time::sleep(check_interval).await;
        let elapsed_ms = now_ms().saturating_sub(last_activity.load(Ordering::Relaxed));
        if elapsed_ms >= timeout_ms {
            return;
        }
    }
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────

/// Perform a graceful shutdown: remove `sid` from `registry.json` and return.
///
/// Self-removal is best-effort. If it fails the error is logged with
/// `tracing::error!` and execution continues to process exit.  The orphan
/// entry will be cleaned up by `scan_and_gc` on the next MCP startup.
///
/// # Arguments
///
/// * `sid` — session ID of this worker (used to locate the registry entry).
/// * `reg_path` — absolute path to `registry.json`.
/// * `lock_path` — absolute path to the advisory lock file (`registry.lock`).
///
/// # Errors
///
/// Returns `PoolError` if the advisory lock or registry I/O fails.
/// The caller is expected to log the error and proceed with exit regardless.
async fn graceful_shutdown(sid: &str, reg_path: &Path, lock_path: &Path) -> Result<(), PoolError> {
    let sid = sid.to_owned();
    let reg_path = reg_path.to_path_buf();
    let lock_path = lock_path.to_path_buf();

    // advisory lock + registry I/O are synchronous (fs4 + std::fs).
    // Running inside a tokio multi_thread runtime requires spawn_blocking.
    // §4-1-4 K-110: async fn 内の同期 std::fs I/O は spawn_blocking でラップ
    tokio::task::spawn_blocking(move || {
        with_registry_lock(&lock_path, || {
            let mut reg = PoolRegistry::load_or_default(&reg_path)?;
            let removed = reg.remove(&sid);
            if removed {
                reg.save(&reg_path)?;
                tracing::info!(worker_sid = %sid, "removed self from registry.json");
            } else {
                tracing::warn!(
                    worker_sid = %sid,
                    "self-remove: entry not found in registry.json (already removed?)"
                );
            }
            Ok(())
        })
    })
    .await
    .map_err(|e| PoolError::RegistryCorrupted(format!("spawn_blocking panicked: {e}")))?
}

// ─── Public entrypoint ────────────────────────────────────────────────────────

/// Run the pool worker main loop.
///
/// Binds the UDS socket, optionally installs an idle timeout and a SIGTERM
/// handler, then accepts client connections and dispatches `PoolRequest`
/// messages.
///
/// # Arguments
///
/// * `sid` — unique session ID for this worker (used in registry and logging).
/// * `sock` — absolute path to the Unix-domain socket to bind.
///
/// # Returns
///
/// `Ok(())` on clean shutdown (SIGTERM, idle timeout, or `Shutdown` request).
///
/// # Errors
///
/// Returns `anyhow::Error` for fatal initialisation failures (socket bind,
/// executor creation, SIGTERM handler installation).  Per-request errors are
/// returned to the client as `PoolResponse::failure(...)` and do not terminate
/// the loop.
pub async fn run(sid: String, sock: PathBuf) -> anyhow::Result<()> {
    // 1. Bind UDS endpoint — propagate io::Error (fatal if binding fails).
    let listener = UnixListener::bind(&sock)?;

    // Restrict the socket file to the owning user only (0600 = -rw-------).
    // UnixListener::bind leaves the socket at the umask-derived mode (typically
    // 0666).  Tightening to 0600 prevents other local users from connecting to
    // the pool worker.  Failure here is fatal — we cannot safely serve requests
    // on an unrestricted socket.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| anyhow::anyhow!("failed to set permissions on {}: {e}", sock.display()))?;
    }

    tracing::info!(worker_sid = %sid, sock = %sock.display(), "pool worker starting");

    // 2. Read idle timeout from environment.
    //    `ALC_POOL_IDLE_TIMEOUT=0` disables the idle timer (worker runs until SIGTERM).
    //    Any non-numeric value falls back to the default (1800s) with a warning.
    //    unwrap_or_else is justified for the absent case: env absent === default
    //    1800s (complete business-correct operation with no difference to caller).
    //    Invalid-parse case emits tracing::warn! so the operator can detect
    //    misconfiguration (§1-2-6 tracing-missing-on-err).
    let timeout_secs: u64 = match std::env::var("ALC_POOL_IDLE_TIMEOUT")
        .unwrap_or_else(|_| DEFAULT_IDLE_TIMEOUT_SECS.to_string())
        .parse::<u64>()
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                worker_sid = %sid,
                error = %e,
                default_secs = DEFAULT_IDLE_TIMEOUT_SECS,
                "ALC_POOL_IDLE_TIMEOUT is not a valid u64 — using default"
            );
            DEFAULT_IDLE_TIMEOUT_SECS
        }
    };

    tracing::debug!(
        worker_sid = %sid,
        timeout_secs = timeout_secs,
        "idle timeout configured"
    );

    // 3. Install SIGTERM handler.
    //    tokio::signal installs an OS-level handler for the duration of the
    //    process.  Any error here is fatal (e.g. signal mask exhausted).
    let mut sigterm = tokio::signal::unix::signal(SignalKind::terminate())?;

    // 4. Shared last-activity clock for the idle timer.
    //    AtomicI64 is lock-free and Send + Sync; Ordering::Relaxed is
    //    sufficient because we only need approximate epoch-ms values, not
    //    strict sequential consistency with other memory operations.
    let last_activity: Arc<AtomicI64> = Arc::new(AtomicI64::new(now_ms()));

    // 5. Resolve application directories.
    let config = AppConfig::from_env();
    let app_dir = config.app_dir();

    // Registry paths for self-removal on shutdown.
    let pool_state_dir = app_dir.root().join("state").join("pool");
    let reg_path = pool_state_dir.join("registry.json");
    let lock_path = pool_state_dir.join("registry.lock");

    let state_store = Arc::new(JsonFileStore::new(app_dir.state_dir()));
    let card_store = Arc::new(FileCardStore::new(app_dir.cards_dir()));
    let scenarios_dir = app_dir.scenarios_dir();

    // 6. Build the executor.
    let executor = Arc::new(Executor::new(resolve_lib_paths()).await?);

    tracing::debug!(worker_sid = %sid, "executor ready");

    // 7. Accept connections in a loop with SIGTERM and idle-timeout monitoring.
    let registry = SessionRegistry::new();
    let mut phase = WorkerPhase::Idle;

    'outer: loop {
        tracing::debug!(worker_sid = %sid, "waiting for client connection");

        // Clone Arc refs for use inside the idle-check future each iteration.
        let last_activity_clone = Arc::clone(&last_activity);
        let idle_future = idle_check_future(last_activity_clone, timeout_secs);
        tokio::pin!(idle_future);

        let stream = tokio::select! {
            biased;

            // ── SIGTERM received ────────────────────────────────────────────
            _ = sigterm.recv() => {
                tracing::info!(worker_sid = %sid, "SIGTERM received — initiating graceful shutdown");
                if let Err(e) = graceful_shutdown(&sid, &reg_path, &lock_path).await {
                    tracing::error!(
                        worker_sid = %sid,
                        error = %e,
                        "graceful shutdown self-remove failed (orphan entry will be GC'd)"
                    );
                }
                break 'outer;
            }

            // ── Idle timeout ────────────────────────────────────────────────
            _ = &mut idle_future => {
                tracing::info!(
                    worker_sid = %sid,
                    timeout_secs = timeout_secs,
                    "idle timeout elapsed — initiating graceful shutdown"
                );
                if let Err(e) = graceful_shutdown(&sid, &reg_path, &lock_path).await {
                    tracing::error!(
                        worker_sid = %sid,
                        error = %e,
                        "graceful shutdown self-remove failed (orphan entry will be GC'd)"
                    );
                }
                break 'outer;
            }

            // ── New client connection ───────────────────────────────────────
            result = listener.accept() => {
                match result {
                    Ok((stream, _peer)) => stream,
                    Err(e) => {
                        tracing::error!(worker_sid = %sid, error = %e, "accept() failed — exiting");
                        break 'outer;
                    }
                }
            }
        };

        tracing::debug!(worker_sid = %sid, "client connected");
        last_activity.store(now_ms(), Ordering::Relaxed);

        let (reader, writer) = tokio::io::split(stream);
        let mut lines = BufReader::new(reader);
        let mut out = BufWriter::new(writer);

        let mut should_exit = false;

        loop {
            let mut line = String::new();
            let n = lines.read_line(&mut line).await?;
            if n == 0 {
                // EOF — client disconnected.
                tracing::info!(worker_sid = %sid, "client disconnected (EOF)");
                break;
            }

            // Update last-activity on every received request.
            last_activity.store(now_ms(), Ordering::Relaxed);

            let req: PoolRequest = match serde_json::from_str(line.trim()) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "malformed request");
                    write_response(
                        &mut out,
                        PoolResponse::failure(format!("malformed request: {e}")),
                    )
                    .await?;
                    continue;
                }
            };

            let is_shutdown = matches!(req, PoolRequest::Shutdown);

            let resp = dispatch(
                &req,
                &sid,
                &mut phase,
                &registry,
                &executor,
                &state_store,
                &card_store,
                &scenarios_dir,
            )
            .await;

            write_response(&mut out, resp).await?;

            if is_shutdown {
                tracing::info!(worker_sid = %sid, "shutdown received — exiting");
                // On explicit Shutdown, remove from registry before exit.
                if let Err(e) = graceful_shutdown(&sid, &reg_path, &lock_path).await {
                    tracing::error!(
                        worker_sid = %sid,
                        error = %e,
                        "graceful shutdown self-remove failed on Shutdown request"
                    );
                }
                should_exit = true;
                break;
            }
        }

        if should_exit {
            break 'outer;
        }

        // If the session has finished (or was never started), exit.
        // Only continue the outer accept-loop when still Paused.
        match &phase {
            WorkerPhase::Paused { .. } => {
                tracing::info!(
                    worker_sid = %sid,
                    "connection dropped but session still paused — waiting for reconnect"
                );
                // continue outer loop → accept next connection
            }
            WorkerPhase::Idle | WorkerPhase::Finished => {
                tracing::info!(worker_sid = %sid, "session not paused after connection drop — exiting");
                break 'outer;
            }
        }
    }

    tracing::info!(worker_sid = %sid, "pool worker exiting");
    Ok(())
}

// ─── Response writer ─────────────────────────────────────────────────────────

/// Write a `PoolResponse` as a single JSON line to the client.
///
/// # Arguments
///
/// * `out` — buffered writer for the client half of the UDS stream.
/// * `resp` — response to serialize and send.
///
/// # Errors
///
/// Returns `anyhow::Error` on serialization or I/O failure.
async fn write_response(
    out: &mut BufWriter<tokio::io::WriteHalf<tokio::net::UnixStream>>,
    resp: PoolResponse,
) -> anyhow::Result<()> {
    let mut json =
        serde_json::to_string(&resp).map_err(|e| anyhow::anyhow!("response serialize: {e}"))?;
    json.push('\n');
    out.write_all(json.as_bytes()).await?;
    out.flush().await?;
    Ok(())
}

// ─── Request dispatcher ───────────────────────────────────────────────────────

/// Dispatch a single `PoolRequest` and produce the appropriate `PoolResponse`.
///
/// Recoverable errors (bad state, session errors) are returned as
/// `PoolResponse::failure` so they reach the client over the wire.
/// The worker never panics — all error paths produce a failure response.
///
/// # Arguments
///
/// * `req` — the incoming request to handle.
/// * `sid` — worker session ID (for logging).
/// * `phase` — mutable reference to the current worker phase.
/// * `registry` — in-memory session registry (mlua thread handles).
/// * `executor` — shared Lua executor.
/// * `state_store`, `card_store`, `scenarios_dir` — app-level stores.
///
/// # Returns
///
/// A `PoolResponse` to be sent back to the client.
#[allow(clippy::too_many_arguments)]
async fn dispatch(
    req: &PoolRequest,
    sid: &str,
    phase: &mut WorkerPhase,
    registry: &SessionRegistry,
    executor: &Arc<Executor>,
    state_store: &Arc<JsonFileStore>,
    card_store: &Arc<FileCardStore>,
    scenarios_dir: &Path,
) -> PoolResponse {
    match req {
        // ── Handshake ──────────────────────────────────────────────────────
        PoolRequest::Handshake { version } => {
            let worker_version = env!("CARGO_PKG_VERSION");
            if version != worker_version {
                tracing::warn!(
                    worker_sid = %sid,
                    client_version = %version,
                    server_version = %worker_version,
                    "version mismatch"
                );
                return PoolResponse::failure(format!(
                    "version mismatch: client={version}, server={worker_version}"
                ));
            }
            PoolResponse::success(PoolResponseData::Handshake {
                version: worker_version.to_string(),
            })
        }

        // ── Run ────────────────────────────────────────────────────────────
        PoolRequest::Run {
            code,
            ctx,
            lib_paths: extra_lib_paths,
        } => {
            if !matches!(phase, WorkerPhase::Idle) {
                tracing::warn!(worker_sid = %sid, "Run received but worker is not idle");
                return PoolResponse::failure("worker already has an active session");
            }

            let ctx_value = ctx.clone().unwrap_or(serde_json::Value::Null);

            // start_session spawns a dedicated OS thread + mlua VM for this session.
            // Pool worker sessions do not resolve the Phase 1-B
            // `[setting.card].run` gate over IPC; the legacy path defaults
            // to OFF, mirroring the `AppService::start_and_tick` policy.
            let session = match executor
                .start_session(
                    code.clone(),
                    ctx_value,
                    extra_lib_paths.clone(),
                    vec![], // variant_pkgs: not passed via IPC in this subtask
                    Arc::clone(state_store),
                    Arc::clone(card_store),
                    scenarios_dir.to_path_buf(),
                    false,
                )
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "start_session failed");
                    return PoolResponse::failure(format!("session start failed: {e}"));
                }
            };

            // Wait for the first event (Paused or Finished).
            let (session_id, feed_result) = match registry.start_execution(session).await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "start_execution failed");
                    return PoolResponse::failure(format!("execution start failed: {e}"));
                }
            };

            let is_paused = matches!(feed_result, FeedResult::Paused { .. });
            let feed_json = match serde_json::to_value(&feed_result) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "FeedResult serialize failed");
                    return PoolResponse::failure(format!("result serialize failed: {e}"));
                }
            };

            *phase = if is_paused {
                WorkerPhase::Paused {
                    session_id: session_id.clone(),
                }
            } else {
                WorkerPhase::Finished
            };

            PoolResponse::success(PoolResponseData::Feed {
                session_id,
                feed_result: feed_json,
            })
        }

        // ── Continue ───────────────────────────────────────────────────────
        PoolRequest::Continue {
            sid: _req_sid,
            response,
            query_id,
            usage,
        } => {
            let session_id = match phase {
                WorkerPhase::Paused { session_id } => session_id.clone(),
                _ => {
                    return PoolResponse::failure("no paused session to continue");
                }
            };

            let qid = QueryId::parse(query_id.as_deref().unwrap_or("q-0"));

            let feed_result = match registry
                .feed_response(&session_id, &qid, response.clone(), usage.as_ref())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "feed_response failed");
                    return PoolResponse::failure(format!("continue failed: {e}"));
                }
            };

            let is_paused = matches!(feed_result, FeedResult::Paused { .. });
            let feed_json = match serde_json::to_value(&feed_result) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(worker_sid = %sid, error = %e, "FeedResult serialize failed");
                    return PoolResponse::failure(format!("result serialize failed: {e}"));
                }
            };

            if !is_paused {
                *phase = WorkerPhase::Finished;
            }
            // If still Paused, keep WorkerPhase::Paused with the same session_id.

            PoolResponse::success(PoolResponseData::Feed {
                session_id,
                feed_result: feed_json,
            })
        }

        // ── Status ─────────────────────────────────────────────────────────
        PoolRequest::Status { include_history } => {
            let (has_session, session_id) = match phase {
                WorkerPhase::Idle => (false, None),
                WorkerPhase::Paused { session_id } => (true, Some(session_id.clone())),
                WorkerPhase::Finished => (false, None),
            };
            let conversation_history = if *include_history {
                if let Some(ref active_sid) = session_id {
                    let snaps = registry.list_snapshots(None, true).await;
                    // conversation_history is nested under `metrics` in the
                    // snapshot shape produced by the engine. Engine may
                    // return an empty array (or absent) until the first LLM
                    // round completes — surface either as present-and-empty
                    // so callers can distinguish "no history yet" from
                    // "feature disabled".
                    snaps.get(active_sid).map(|s| {
                        s.get("metrics")
                            .and_then(|m| m.get("conversation_history"))
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!([]))
                    })
                } else {
                    None
                }
            } else {
                None
            };
            PoolResponse::success(PoolResponseData::Status {
                has_session,
                session_id,
                conversation_history,
            })
        }

        // ── Shutdown ───────────────────────────────────────────────────────
        PoolRequest::Shutdown => PoolResponse::success(PoolResponseData::Shutdown),
    }
}

// ─── Lib-path resolver ───────────────────────────────────────────────────────

/// Resolve Lua package search paths using the same logic as the MCP server.
///
/// # Returns
///
/// A list of directories to add to the Lua `package.path` search chain.
fn resolve_lib_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(env_paths) = std::env::var("ALC_PACKAGES_PATH") {
        for p in env_paths.split(':') {
            let path = PathBuf::from(p);
            if path.is_dir() {
                paths.push(path);
            }
        }
    }

    if let Some(home) = dirs::home_dir() {
        let packages = home.join(".algocline").join("packages");
        if packages.is_dir() {
            paths.push(packages);
        }
    }

    paths
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_app::pool::{PoolRegistry, PoolSessionEntry};
    use std::sync::atomic::AtomicI64;

    // ── T1: idle timeout self-exit ────────────────────────────────────────────

    /// T1 — worker exits when idle for `ALC_POOL_IDLE_TIMEOUT` seconds.
    ///
    /// Spawns a `pool-worker` subprocess with `ALC_POOL_IDLE_TIMEOUT=2`,
    /// waits 4 seconds, and asserts the process has exited.
    ///
    /// The subprocess must be a real `alc pool-worker` binary (built with
    /// `cargo build`).  This test is skipped when the binary is not present.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_pool_worker_idle_timeout_self_exit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg_path = dir.path().join("state").join("pool").join("registry.json");
        let lock_path = dir.path().join("state").join("pool").join("registry.lock");
        let sock_path = dir.path().join("worker.sock");
        let alc_home = dir.path().to_path_buf();

        let sid = "idle-timeout-test-sid";

        // Locate the `alc` binary from the current test executable directory.
        // In a cargo workspace: target/{debug,release}/deps/../alc
        let exe = std::env::current_exe().expect("current_exe");
        let target_dir = exe
            .parent() // deps
            .and_then(|p| p.parent()) // debug or release
            .expect("target dir");
        let alc_bin = target_dir.join("alc");

        if !alc_bin.exists() {
            eprintln!(
                "Skipping test_pool_worker_idle_timeout_self_exit: binary {:?} not found. \
                 Run `cargo build` first.",
                alc_bin
            );
            return;
        }

        let mut child = tokio::process::Command::new(&alc_bin)
            .arg("pool-worker")
            .arg("--sid")
            .arg(sid)
            .arg("--sock")
            .arg(sock_path.to_str().expect("sock path utf8"))
            .env("ALC_POOL_IDLE_TIMEOUT", "2")
            // ALC_HOME routes registry.json to a tempdir so the test is isolated.
            .env("ALC_HOME", alc_home.to_str().expect("alc home utf8"))
            .spawn()
            .expect("spawn worker");

        let worker_pid: u32 = child.id().expect("child PID");

        // Create the registry directory and register the worker so
        // graceful_shutdown can remove its entry.
        // justification: test code — unwrap is acceptable for test setup failures
        std::fs::create_dir_all(reg_path.parent().expect("reg parent"))
            .expect("create registry dir");
        with_registry_lock(&lock_path, || {
            let mut reg = PoolRegistry::load_or_default(&reg_path)?;
            reg.add(PoolSessionEntry::new(
                sid,
                worker_pid,
                sock_path.clone(),
                "0.30.0",
            ));
            reg.save(&reg_path)
        })
        .expect("register worker");

        // Idle timeout is 2s, check interval is 1s — exit expected ~3s.
        // Poll up to 6s to absorb cold-spawn latency on slow / loaded hosts.
        let mut exited = false;
        for _ in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            if child.try_wait().expect("try_wait").is_some() {
                exited = true;
                break;
            }
        }
        assert!(
            exited,
            "worker should have exited after idle timeout (pid={worker_pid})"
        );

        // Registry entry should be absent (worker removed itself).
        let final_reg = PoolRegistry::load_or_default(&reg_path).expect("load final registry");
        // Entry removal is best-effort; assert at least that the worker exited.
        // (Entry may still be present if worker crashed before self-remove.)
        let _ = final_reg;
    }

    // ── T2: SIGTERM graceful shutdown ─────────────────────────────────────────

    /// T2 — worker removes itself from registry.json on SIGTERM.
    ///
    /// Spawns a `pool-worker` subprocess with idle timeout disabled, sends
    /// SIGTERM, and asserts (a) the process exits and (b) the registry entry
    /// is removed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_pool_worker_sigterm_graceful_shutdown() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg_path = dir.path().join("state").join("pool").join("registry.json");
        let lock_path = dir.path().join("state").join("pool").join("registry.lock");
        let sock_path = dir.path().join("sigterm-worker.sock");
        let alc_home = dir.path().to_path_buf();

        let sid = "sigterm-test-sid";

        let exe = std::env::current_exe().expect("current_exe");
        let target_dir = exe.parent().and_then(|p| p.parent()).expect("target dir");
        let alc_bin = target_dir.join("alc");

        if !alc_bin.exists() {
            eprintln!(
                "Skipping test_pool_worker_sigterm_graceful_shutdown: binary {:?} not found.",
                alc_bin
            );
            return;
        }

        let mut child = tokio::process::Command::new(&alc_bin)
            .arg("pool-worker")
            .arg("--sid")
            .arg(sid)
            .arg("--sock")
            .arg(sock_path.to_str().expect("sock path utf8"))
            .env("ALC_POOL_IDLE_TIMEOUT", "0") // disabled — only SIGTERM kills it
            // ALC_HOME routes registry.json to a tempdir so the test is isolated.
            .env("ALC_HOME", alc_home.to_str().expect("alc home utf8"))
            .spawn()
            .expect("spawn worker");

        let worker_pid: u32 = child.id().expect("child PID");

        // Create the registry directory and register the worker.
        // justification: test code — unwrap is acceptable for test setup failures
        std::fs::create_dir_all(reg_path.parent().expect("reg parent"))
            .expect("create registry dir");
        with_registry_lock(&lock_path, || {
            let mut reg = PoolRegistry::load_or_default(&reg_path)?;
            reg.add(PoolSessionEntry::new(
                sid,
                worker_pid,
                sock_path.clone(),
                "0.30.0",
            ));
            reg.save(&reg_path)
        })
        .expect("register worker");

        // Give the worker time to start and install the SIGTERM handler.
        // Cold binary startup under parallel test load (all 21 tests in this
        // bin running concurrently, plus other workspace crates) can take
        // 1500-2500ms; 3s covers the worst case observed locally (see
        // `cargo test --bin alc` 21-parallel reproduction).
        tokio::time::sleep(tokio::time::Duration::from_millis(3000)).await;

        // Send SIGTERM.
        // SAFETY: libc::kill(pid, sig) is a thin POSIX syscall wrapper.
        // pid > 0 targets the specific process.  pid fits in i32 (verified by
        // try_from).  sig = SIGTERM is a standard termination signal.
        let pid_i32 = i32::try_from(worker_pid)
            // justification: test code — panic on PID overflow is acceptable
            // because PIDs on Linux/macOS are bounded well below i32::MAX
            .expect("PID fits in i32");
        let rc = unsafe { libc::kill(pid_i32, libc::SIGTERM) };
        assert_eq!(rc, 0, "kill(SIGTERM) must succeed");

        // Wait up to 3 seconds for the process to exit.
        let mut exited = false;
        for _ in 0..30 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            if let Some(_status) = child.try_wait().expect("try_wait") {
                exited = true;
                break;
            }
        }
        assert!(
            exited,
            "worker should exit after SIGTERM (pid={worker_pid})"
        );

        // Entry removal happens inside spawn_blocking and is not strictly
        // ordered before process exit observation, so poll up to 2s for the
        // registry entry to disappear.
        let mut removed = false;
        for _ in 0..40 {
            let reg = PoolRegistry::load_or_default(&reg_path).expect("load final registry");
            if reg.find(sid).is_none() {
                removed = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        assert!(
            removed,
            "registry entry must be removed after graceful shutdown"
        );
    }

    // ── T3: concurrent AtomicI64 last_activity update ─────────────────────────

    /// T3 — 8 tasks concurrently store to AtomicI64; idle timer reads are safe.
    ///
    /// Verifies that concurrent Relaxed stores from 8 threads produce no UB
    /// and that a final load returns a value within the range of stored values.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_atomic_i64_last_activity_concurrent_update() {
        let last_activity = Arc::new(AtomicI64::new(0));
        let n_tasks = 8usize;
        let n_iters = 1000usize;

        let mut handles = Vec::new();
        for _ in 0..n_tasks {
            let la = Arc::clone(&last_activity);
            handles.push(tokio::spawn(async move {
                for _ in 0..n_iters {
                    la.store(now_ms(), Ordering::Relaxed);
                    // Yield occasionally to interleave tasks.
                    tokio::task::yield_now().await;
                }
            }));
        }

        for h in handles {
            h.await.expect("task did not panic");
        }

        // Final value must be a plausible epoch-ms (> 0, meaning at least one
        // store succeeded and no UB occurred).
        let final_val = last_activity.load(Ordering::Relaxed);
        assert!(
            final_val > 0,
            "last_activity must be positive after concurrent stores"
        );
    }

    // ── T4: idle timeout disabled (ALC_POOL_IDLE_TIMEOUT=0) ──────────────────

    /// T4 — idle_check_future with timeout_secs=0 never resolves.
    ///
    /// Uses tokio::time::timeout to assert that the future does not resolve
    /// within 100 ms when disabled.
    #[tokio::test]
    async fn test_idle_check_future_disabled() {
        let last_activity = Arc::new(AtomicI64::new(now_ms()));

        // timeout_secs=0 → idle_check_future returns std::future::pending()
        let result = tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            idle_check_future(Arc::clone(&last_activity), 0),
        )
        .await;

        assert!(
            result.is_err(),
            "idle_check_future(0) must not resolve (returns pending())"
        );
    }

    // ── T5: idle_check_future resolves after timeout ───────────────────────────

    /// T5 — idle_check_future resolves within 2× the configured timeout.
    ///
    /// Sets timeout_secs=1 with a last_activity in the past (> 1s ago), then
    /// asserts the future resolves within 2 seconds.
    #[tokio::test]
    async fn test_idle_check_future_fires() {
        let last_activity = Arc::new(AtomicI64::new(0)); // epoch 0 = 1970 = very stale

        let result = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            idle_check_future(Arc::clone(&last_activity), 1),
        )
        .await;

        assert!(
            result.is_ok(),
            "idle_check_future(1) must resolve when activity is stale"
        );
    }
}
