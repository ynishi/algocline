//! Pool worker subprocess entrypoint.
//!
//! Invoked by the MCP process via `alc pool-worker --sid <sid> --sock <path>`.
//! Each worker holds exactly one Lua session (1 session = 1 process isolation).
//!
//! ## Lifecycle
//!
//! 1. Bind a Unix-domain socket at `sock`.
//! 2. Accept a single connection from the MCP pool client.
//! 3. Dispatch `PoolRequest` messages until `Shutdown` or EOF.
//! 4. Return from `run()` — tokio runtime cleans up naturally.
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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use algocline_app::pool::{PoolRequest, PoolResponse, PoolResponseData};
use algocline_app::AppConfig;
use algocline_core::QueryId;
use algocline_engine::{Executor, FeedResult, FileCardStore, JsonFileStore, SessionRegistry};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixListener;

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

/// Run the pool worker main loop.
///
/// Binds the UDS socket, accepts one client connection, and dispatches
/// `PoolRequest` messages until `Shutdown` or EOF.
///
/// # Errors
///
/// Returns `anyhow::Error` for fatal initialisation failures (socket bind,
/// executor creation).  Per-request errors are returned to the client as
/// `PoolResponse::failure(...)` and do not terminate the loop.
pub async fn run(sid: String, sock: PathBuf) -> anyhow::Result<()> {
    // 1. Bind UDS endpoint — propagate io::Error (fatal if binding fails).
    let listener = UnixListener::bind(&sock)?;

    tracing::info!(worker_sid = %sid, sock = %sock.display(), "pool worker starting");

    // 2. Resolve application directories using the same path as the MCP server.
    let config = AppConfig::from_env();
    let app_dir = config.app_dir();

    let state_store = Arc::new(JsonFileStore::new(app_dir.state_dir()));
    let card_store = Arc::new(FileCardStore::new(app_dir.cards_dir()));
    let scenarios_dir = app_dir.scenarios_dir();

    // 3. Build the executor with resolved global package paths.
    //    Executor::new spawns a shared VM for eval_simple (lightweight) and
    //    stores lib_paths for per-session VM spawns via start_session.
    //    Extra project-local lib_paths arrive in the `Run` request and are
    //    passed through to start_session as extra_lib_paths.
    let executor = Arc::new(Executor::new(resolve_lib_paths()).await?);

    tracing::debug!(worker_sid = %sid, "executor ready");

    // 4. Accept exactly one connection.
    //    The worker never reads stdin — if the parent MCP process dies and
    //    stdin closes, the UDS accept loop continues uninterrupted.
    let (stream, _peer) = listener.accept().await?;

    tracing::debug!(worker_sid = %sid, "client connected");

    let (reader, writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader);
    let mut out = BufWriter::new(writer);

    // 5. Dispatch loop.
    let registry = SessionRegistry::new();
    let mut phase = WorkerPhase::Idle;

    loop {
        let mut line = String::new();
        let n = lines.read_line(&mut line).await?;
        if n == 0 {
            // EOF — client disconnected. Exit cleanly.
            tracing::info!(worker_sid = %sid, "client disconnected (EOF)");
            break;
        }

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
            break;
        }
    }

    tracing::info!(worker_sid = %sid, "pool worker exiting");
    Ok(())
}

/// Write a `PoolResponse` as a single JSON line to the client.
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

/// Dispatch a single `PoolRequest` and produce the appropriate `PoolResponse`.
///
/// Recoverable errors (bad state, session errors) are returned as
/// `PoolResponse::failure` so they reach the client over the wire.
/// The worker never panics — all error paths produce a failure response.
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
            let session = match executor
                .start_session(
                    code.clone(),
                    ctx_value,
                    extra_lib_paths.clone(),
                    vec![], // variant_pkgs: not passed via IPC in this subtask
                    Arc::clone(state_store),
                    Arc::clone(card_store),
                    scenarios_dir.to_path_buf(),
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
        PoolRequest::Status => {
            let (has_session, session_id) = match phase {
                WorkerPhase::Idle => (false, None),
                WorkerPhase::Paused { session_id } => (true, Some(session_id.clone())),
                WorkerPhase::Finished => (false, None),
            };
            PoolResponse::success(PoolResponseData::Status {
                has_session,
                session_id,
            })
        }

        // ── Shutdown ───────────────────────────────────────────────────────
        PoolRequest::Shutdown => PoolResponse::success(PoolResponseData::Shutdown),
    }
}

/// Resolve Lua package search paths using the same logic as the MCP server.
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
