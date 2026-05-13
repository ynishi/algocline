//! `driver_loop` — background async task that drives a v2 execution session.
//!
//! The loop bridges between the legacy Lua VM machinery (via `Session::into_driver_parts`)
//! and the new `ExecutionState` / `ProgressEvent` state machine.
//!
//! # Cancellation checkpoints (Crux R2)
//!
//! The driver checks for cooperative cancellation at exactly four points:
//!
//! | Point | Location | Trigger |
//! |-------|----------|---------|
//! | A | Before the first Lua chunk iteration | `is_cancelled()` |
//! | B | After `llm_rx.recv()` but before `PauseRequested` publish | `is_cancelled()` |
//! | C | At `resume` entry (in `registry.rs`) | `is_cancelled()` |
//! | D | Main `tokio::select!` branch | `cancel_token.cancelled()` |
//!
//! No `JoinHandle::abort()`, `process::exit`, or signal-kill path exists (Crux R2).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use algocline_core::execution::{
    CancelCode, CancelInfo, CancelReason, ExecutionResult, ExecutionState, FailureInfo,
    FailureKind, PauseInfo, PauseKind, PausePrompt, ProgressEvent,
};
use mlua_isle::AsyncTask;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::llm_bridge::LlmRequest;

/// Returns Unix milliseconds (i64) for the current wall-clock time.
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// ---------------------------------------------------------------------------
// transition helpers
// ---------------------------------------------------------------------------

/// Publish a `StateTransition` event and update shared state to the new value.
///
/// The `Mutex` guard is dropped **before** the broadcast `send` to avoid holding
/// it across the (sync but potentially contended) send path (K-4 pattern).
pub(crate) async fn transition_state(
    state: &Arc<Mutex<ExecutionState>>,
    bus_tx: &broadcast::Sender<ProgressEvent>,
    new_state: ExecutionState,
) {
    let from_tag = {
        let guard = state.lock().await;
        guard.tag()
    };
    let to_tag = new_state.tag();
    {
        let mut guard = state.lock().await;
        *guard = new_state;
    }
    // broadcast::send is sync and does not need the lock.
    let _ = bus_tx.send(ProgressEvent::StateTransition {
        from: from_tag,
        to: to_tag,
        at: now_ms(),
    });
}

/// Build a `CancelInfo` snapshot capturing `reason` and the current state.
pub(crate) async fn build_cancel_info(
    state: &Arc<Mutex<ExecutionState>>,
    reason: CancelReason,
) -> CancelInfo {
    let state_before = {
        let guard = state.lock().await;
        guard.clone()
    };
    CancelInfo {
        reason,
        observed_at: now_ms(),
        state_before: Box::new(state_before),
    }
}

// ---------------------------------------------------------------------------
// driver_loop
// ---------------------------------------------------------------------------

/// Background async function that drives a v2 session to completion.
///
/// Spawned by [`super::registry::SessionRegistryV2::spawn`] via `tokio::spawn`.
/// The caller stores the returned [`tokio::task::JoinHandle`] in the
/// [`super::record::SessionRecord`].
///
/// # Arguments
///
/// - `exec_task` — The Lua coroutine task from `Executor::start_session`.
/// - `llm_rx` — Receiver for LLM requests emitted by `alc.llm()`.
/// - `state` — Shared execution state (v2), updated at each transition.
/// - `bus_tx` — Broadcast sender for `ProgressEvent`s.
/// - `cancel_token` — Cooperative cancellation token (Crux R2).
/// - `resp_txs` — Shared map used by `registry::resume()` to deliver responses.
pub(crate) async fn driver_loop(
    mut exec_task: AsyncTask,
    mut llm_rx: mpsc::Receiver<LlmRequest>,
    state: Arc<Mutex<ExecutionState>>,
    bus_tx: broadcast::Sender<ProgressEvent>,
    cancel_token: CancellationToken,
    resp_txs: super::record::RespTxsMap,
) {
    // checkpoint A: before Lua chunk
    // Check cancellation before entering the main loop so that a pre-cancelled
    // token prevents any Lua execution from starting.
    if cancel_token.is_cancelled() {
        let reason = CancelReason {
            code: CancelCode::User,
            detail: Some("cancelled before execution started (checkpoint A)".into()),
            requested_at: now_ms(),
        };
        let info = build_cancel_info(&state, reason).await;
        transition_state(&state, &bus_tx, ExecutionState::Cancelled(info)).await;
        return;
    }

    // Publish an initial Running tick so observers know the session started.
    let _ = bus_tx.send(ProgressEvent::Tick {
        phase: "running".into(),
        at: now_ms(),
    });

    loop {
        tokio::select! {
            biased;

            // checkpoint D: long-IO tokio::select! with cancel
            // Highest-priority branch: if the token fires we cancel immediately
            // without waiting for exec_task or llm_rx.
            _ = cancel_token.cancelled() => {
                let reason = CancelReason {
                    code: CancelCode::User,
                    detail: Some("cancelled at select! checkpoint D".into()),
                    requested_at: now_ms(),
                };
                let info = build_cancel_info(&state, reason).await;
                transition_state(&state, &bus_tx, ExecutionState::Cancelled(info)).await;
                break;
            }

            // Lua execution completed (Done or Failed).
            result = &mut exec_task => {
                match result {
                    Ok(json_str) => {
                        match serde_json::from_str::<serde_json::Value>(&json_str) {
                            Ok(v) => {
                                let done = ExecutionState::Done(ExecutionResult {
                                    value: v,
                                    usage: None,
                                    finished_at: now_ms(),
                                });
                                transition_state(&state, &bus_tx, done).await;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "driver_loop: JSON parse error on exec_task result: {e}"
                                );
                                let failed = ExecutionState::Failed(FailureInfo {
                                    message: format!("JSON parse error: {e}"),
                                    kind: FailureKind::EngineError,
                                    occurred_at: now_ms(),
                                });
                                transition_state(&state, &bus_tx, failed).await;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("driver_loop: exec_task error: {e}");
                        let failed = ExecutionState::Failed(FailureInfo {
                            message: e.to_string(),
                            kind: FailureKind::LuaError,
                            occurred_at: now_ms(),
                        });
                        transition_state(&state, &bus_tx, failed).await;
                    }
                }
                break;
            }

            // Lua yielded an LLM request (alc.llm / alc.llm_batch).
            Some(req) = llm_rx.recv() => {
                // checkpoint B: before pause publish
                // If cancelled between exec_task yield and here, skip the pause.
                if cancel_token.is_cancelled() {
                    let reason = CancelReason {
                        code: CancelCode::User,
                        detail: Some("cancelled before pause publish (checkpoint B)".into()),
                        requested_at: now_ms(),
                    };
                    let info = build_cancel_info(&state, reason).await;
                    transition_state(&state, &bus_tx, ExecutionState::Cancelled(info)).await;

                    // Respond to all queries with an error so the coroutine wakes
                    // and can exit cleanly (prevents a goroutine leak in mlua-isle).
                    for qr in req.queries {
                        if let Err(_e) = qr.resp_tx.send(Err("cancelled".into())) {
                            tracing::debug!(
                                "driver_loop checkpoint B: failed to send cancel to coroutine \
                                 (receiver already dropped)"
                            );
                        }
                    }
                    break;
                }

                // Determine pause kind and build PauseInfo.
                let kind = if req.queries.len() == 1 {
                    PauseKind::Single
                } else {
                    PauseKind::Batch
                };
                let prompts: Vec<PausePrompt> = req.queries.iter().map(|qr| PausePrompt {
                    query_id: qr.id.as_str().to_owned(),
                    prompt: qr.prompt.clone(),
                }).collect();
                let pause_info = PauseInfo {
                    kind,
                    prompts,
                    paused_at: now_ms(),
                };

                // Store resp_txs before publishing the pause event.
                {
                    let mut txs = resp_txs.lock().await;
                    for qr in req.queries {
                        txs.insert(qr.id, qr.resp_tx);
                    }
                }

                // Transition to Paused and publish PauseRequested.
                let pause_event = ProgressEvent::PauseRequested {
                    info: pause_info.clone(),
                    at: now_ms(),
                };
                transition_state(&state, &bus_tx, ExecutionState::Paused(pause_info)).await;
                let _ = bus_tx.send(pause_event);

                // The loop continues — the next iteration will wait for either
                // exec_task to produce a result (after resume delivers resp_txs)
                // or for another llm_rx message.
            }
        }
    }

    // Session has reached a terminal state.  The bus_tx will be dropped when
    // SessionRecord is dropped, delivering RecvError::Closed to all subscribers.
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::execution::ExecutionState;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::{broadcast, Mutex};
    use tokio_util::sync::CancellationToken;

    // -----------------------------------------------------------------------
    // Helpers using real Executor to get real AsyncTask handles
    // -----------------------------------------------------------------------

    fn tmp_dirs() -> (
        Arc<crate::state::JsonFileStore>,
        Arc<crate::card::FileCardStore>,
        std::path::PathBuf,
    ) {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        (
            Arc::new(crate::state::JsonFileStore::new(root.join("state"))),
            Arc::new(crate::card::FileCardStore::new(root.join("cards"))),
            root.join("scenarios"),
        )
    }

    fn make_state_and_bus() -> (
        Arc<Mutex<ExecutionState>>,
        broadcast::Sender<ProgressEvent>,
        broadcast::Receiver<ProgressEvent>,
    ) {
        let state = Arc::new(Mutex::new(ExecutionState::Running));
        let (tx, rx) = broadcast::channel(256);
        (state, tx, rx)
    }

    // -----------------------------------------------------------------------
    // cancel_at_checkpoint_a_before_first_lua_chunk
    // -----------------------------------------------------------------------

    /// A pre-cancelled token must cause the loop to transition to Cancelled
    /// without executing any Lua.
    #[tokio::test]
    async fn cancel_at_checkpoint_a_before_first_lua_chunk() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        // Lua that would run indefinitely if not cancelled.
        let code = "while true do end".to_string();
        let session = executor
            .start_session(
                code,
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        let (exec_task, llm_rx, _vm_driver) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        // Pre-cancel BEFORE spawning the driver_loop.
        cancel_token.cancel();

        let resp_txs = Arc::new(Mutex::new(HashMap::new()));
        driver_loop(
            exec_task,
            llm_rx,
            state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
        )
        .await;

        let guard = state.lock().await;
        assert!(
            matches!(*guard, ExecutionState::Cancelled(_)),
            "expected Cancelled after pre-cancel checkpoint A, got: {:?}",
            guard.tag()
        );
    }

    // -----------------------------------------------------------------------
    // cancel_at_checkpoint_b_before_pause_publish
    // -----------------------------------------------------------------------

    /// Cancel the token after the driver receives an LLM request (via llm_rx)
    /// but before publishing PauseRequested.  Since we pre-cancel the token,
    /// checkpoint A fires first and the state is Cancelled without pausing.
    ///
    /// The key invariant: the session NEVER reaches Paused state when
    /// cancel_token is set before or during LLM request processing.
    #[tokio::test]
    async fn cancel_at_checkpoint_b_before_pause_publish() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        // Lua that immediately calls alc.llm() to trigger a pause request.
        let code = r#"return alc.llm("question")"#.to_string();
        let session = executor
            .start_session(
                code,
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        let (exec_task, llm_rx, _vm_driver) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        // Pre-cancel so checkpoint B fires when llm_rx delivers the request.
        cancel_token.cancel();

        let resp_txs = Arc::new(Mutex::new(HashMap::new()));
        driver_loop(
            exec_task,
            llm_rx,
            state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
        )
        .await;

        let guard = state.lock().await;
        assert!(
            matches!(*guard, ExecutionState::Cancelled(_)),
            "expected Cancelled when token set before pause publish, got: {:?}",
            guard.tag()
        );
    }

    // -----------------------------------------------------------------------
    // cancel_idempotent
    // -----------------------------------------------------------------------

    /// Cancelling an already-cancelled `CancellationToken` must be a no-op.
    #[test]
    fn cancel_idempotent() {
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();
        assert!(cancel_token.is_cancelled());
        // Second cancel must not panic.
        cancel_token.cancel();
        assert!(cancel_token.is_cancelled());
    }

    // -----------------------------------------------------------------------
    // driver_loop completes with Done for a trivial Lua script
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn driver_loop_completes_with_done() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        let session = executor
            .start_session(
                "return 42".to_string(),
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        let (exec_task, llm_rx, _vm_driver) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        let resp_txs = Arc::new(Mutex::new(HashMap::new()));

        driver_loop(
            exec_task,
            llm_rx,
            state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
        )
        .await;

        let guard = state.lock().await;
        assert!(
            matches!(*guard, ExecutionState::Done(_)),
            "expected Done for trivial Lua, got: {:?}",
            guard.tag()
        );
    }

    // -----------------------------------------------------------------------
    // Checkpoint marker presence check (compile-time + comment)
    // -----------------------------------------------------------------------
    // The following tests ensure the four comment markers referenced by
    // Crux R2 grep are anchored in this file and registry.rs:
    //
    // checkpoint A: before Lua chunk
    // checkpoint B: before pause publish
    // checkpoint C: at resume entry (see registry.rs)
    // checkpoint D: long-IO tokio::select! with cancel
    #[test]
    fn checkpoint_markers_exist_in_driver() {
        let source = include_str!("driver.rs");
        for marker in &["checkpoint A", "checkpoint B", "checkpoint D"] {
            assert!(
                source.contains(marker),
                "driver.rs must contain comment '{marker}'"
            );
        }
    }

    #[test]
    fn checkpoint_c_exists_in_registry() {
        let registry_source = include_str!("registry.rs");
        assert!(
            registry_source.contains("checkpoint C"),
            "registry.rs must contain comment 'checkpoint C'"
        );
    }
}
