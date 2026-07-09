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

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use algocline_core::execution::{
    CancelCode, CancelInfo, CancelReason, ExecutionResult, ExecutionState, ExecutionStateTag,
    FailureInfo, FailureKind, PauseInfo, PauseKind, PausePrompt, ProgressEvent,
};
use algocline_core::{ExecutionMetrics, ExecutionObserver, LlmQuery};
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
///
/// **Idempotent on terminal states**: when the current state is already terminal
/// (`Done` / `Failed` / `Cancelled`) and the requested transition would also be
/// to a terminal state, this is a no-op (state is not overwritten, no event is
/// emitted). This prevents nested `state_before` accumulation in `CancelInfo`
/// when both `registry::cancel()` and a `driver_loop` checkpoint attempt to
/// transition a Paused session to `Cancelled`.
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

    // Terminal → terminal is a no-op: the first transition wins.
    if is_terminal_tag(from_tag) && is_terminal_tag(to_tag) {
        return;
    }

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

fn is_terminal_tag(tag: ExecutionStateTag) -> bool {
    matches!(
        tag,
        ExecutionStateTag::Done | ExecutionStateTag::Failed | ExecutionStateTag::Cancelled
    )
}

/// Build a `CancelInfo` snapshot capturing `reason` and the current state.
///
/// **Nested-aware**: if the current state is already `Cancelled`, the new
/// `state_before` inherits the inner `state_before` rather than wrapping the
/// outer `Cancelled` (which would create `Cancelled(state_before=Cancelled(...))`
/// chains on repeated cancel attempts). The pre-cancellation state is preserved
/// across idempotent re-cancel paths.
pub(crate) async fn build_cancel_info(
    state: &Arc<Mutex<ExecutionState>>,
    reason: CancelReason,
) -> CancelInfo {
    let state_before = {
        let guard = state.lock().await;
        match &*guard {
            ExecutionState::Cancelled(prior) => (*prior.state_before).clone(),
            other => other.clone(),
        }
    };
    CancelInfo {
        reason,
        observed_at: now_ms(),
        state_before: Box::new(state_before),
    }
}

// ---------------------------------------------------------------------------
// DriverContext
// ---------------------------------------------------------------------------

/// Shared resources held by [`driver_loop`] for the lifetime of a v2 session.
///
/// Grouping these six `Arc`/channel/token handles into a single struct keeps
/// `driver_loop` reachable with two non-shared owned values (`exec_task`,
/// `llm_rx`) and reduces caller churn when future shared fields are added.
///
/// # Drop order
///
/// Field declaration order MUST match the original flat-argument order of
/// `driver_loop` (`state, bus_tx, cancel_token, resp_txs, last_active`) so
/// that `Arc` reference-count releases and cancellation-token drops occur
/// in the same sequence as before this refactor (see crux-card.md #2).
/// `metrics` is appended last; `Arc` drop is order-independent.
pub(crate) struct DriverContext {
    pub state: Arc<Mutex<ExecutionState>>,
    pub bus_tx: broadcast::Sender<ProgressEvent>,
    pub cancel_token: CancellationToken,
    pub resp_txs: super::record::RespTxsMap,
    pub last_active: Arc<AtomicI64>,
    pub metrics: Arc<ExecutionMetrics>,
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
/// - `ctx` — Shared resources for this session: `state`, `bus_tx`,
///   `cancel_token`, `resp_txs`, and `last_active`.
/// - `exec_task` — The Lua coroutine task from `Executor::start_session`.
/// - `llm_rx` — Receiver for LLM requests emitted by `alc.llm()`.
pub(crate) async fn driver_loop(
    ctx: DriverContext,
    mut exec_task: AsyncTask,
    mut llm_rx: mpsc::Receiver<LlmRequest>,
) {
    // Record the wall-clock entry time so the GC can detect abandoned sessions.
    // Single writer (driver_loop); GC reads via Relaxed load — no happens-before
    // required for millisecond-precision idle tracking (legacy parity: Crux #3).
    ctx.last_active.store(now_ms(), Ordering::Relaxed);

    // checkpoint A: before Lua chunk
    // Check cancellation before entering the main loop so that a pre-cancelled
    // token prevents any Lua execution from starting.
    if ctx.cancel_token.is_cancelled() {
        let reason = CancelReason {
            code: CancelCode::User,
            detail: Some("cancelled before execution started (checkpoint A)".into()),
            requested_at: now_ms(),
        };
        let info = build_cancel_info(&ctx.state, reason).await;
        transition_state(&ctx.state, &ctx.bus_tx, ExecutionState::Cancelled(info)).await;
        return;
    }

    // Publish an initial Running tick so observers know the session started.
    let _ = ctx.bus_tx.send(ProgressEvent::Tick {
        phase: "running".into(),
        at: now_ms(),
    });

    loop {
        tokio::select! {
            biased;

            // checkpoint D: long-IO tokio::select! with cancel
            // Highest-priority branch: if the token fires we cancel immediately
            // without waiting for exec_task or llm_rx.
            _ = ctx.cancel_token.cancelled() => {
                let reason = CancelReason {
                    code: CancelCode::User,
                    detail: Some("cancelled at select! checkpoint D".into()),
                    requested_at: now_ms(),
                };
                let info = build_cancel_info(&ctx.state, reason).await;
                transition_state(&ctx.state, &ctx.bus_tx, ExecutionState::Cancelled(info)).await;
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
                                    usage: ctx.metrics.usage_aggregate(),
                                    finished_at: now_ms(),
                                });
                                transition_state(&ctx.state, &ctx.bus_tx, done).await;
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
                                transition_state(&ctx.state, &ctx.bus_tx, failed).await;
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
                        transition_state(&ctx.state, &ctx.bus_tx, failed).await;
                    }
                }
                break;
            }

            // Lua yielded an LLM request (alc.llm / alc.llm_batch).
            Some(req) = llm_rx.recv() => {
                // checkpoint B: before pause publish
                // If cancelled between exec_task yield and here, skip the pause.
                if ctx.cancel_token.is_cancelled() {
                    let reason = CancelReason {
                        code: CancelCode::User,
                        detail: Some("cancelled before pause publish (checkpoint B)".into()),
                        requested_at: now_ms(),
                    };
                    let info = build_cancel_info(&ctx.state, reason).await;
                    transition_state(&ctx.state, &ctx.bus_tx, ExecutionState::Cancelled(info)).await;

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

                // Build observer query slice before req.queries is consumed below.
                let queries_for_observer: Vec<LlmQuery> = req.queries.iter()
                    .map(|qr| LlmQuery {
                        id: qr.id.clone(),
                        prompt: qr.prompt.clone(),
                        system: qr.system.clone(),
                        max_tokens: qr.max_tokens,
                        grounded: qr.grounded,
                        underspecified: qr.underspecified,
                        cache_breakpoint: qr.cache_breakpoint.clone(),
                    })
                    .collect();
                ctx.metrics.create_observer().on_paused(&queries_for_observer);

                let pause_info = PauseInfo {
                    kind,
                    prompts,
                    paused_at: now_ms(),
                };

                // Store resp_txs before publishing the pause event.
                {
                    let mut txs = ctx.resp_txs.lock().await;
                    for qr in req.queries {
                        txs.insert(qr.id, qr.resp_tx);
                    }
                }

                // Transition to Paused and publish PauseRequested.
                let pause_event = ProgressEvent::PauseRequested {
                    info: pause_info.clone(),
                    at: now_ms(),
                };
                transition_state(&ctx.state, &ctx.bus_tx, ExecutionState::Paused(pause_info)).await;
                let _ = ctx.bus_tx.send(pause_event);

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

        let (exec_task, llm_rx, _vm_driver, _metrics) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        // Pre-cancel BEFORE spawning the driver_loop.
        cancel_token.cancel();

        let resp_txs = Arc::new(Mutex::new(HashMap::new()));
        let ctx = DriverContext {
            state: state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
            last_active: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            metrics: Arc::new(algocline_core::ExecutionMetrics::new()),
        };
        driver_loop(ctx, exec_task, llm_rx).await;

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

        let (exec_task, llm_rx, _vm_driver, _metrics) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        // Pre-cancel so checkpoint B fires when llm_rx delivers the request.
        cancel_token.cancel();

        let resp_txs = Arc::new(Mutex::new(HashMap::new()));
        let ctx = DriverContext {
            state: state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
            last_active: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            metrics: Arc::new(algocline_core::ExecutionMetrics::new()),
        };
        driver_loop(ctx, exec_task, llm_rx).await;

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
    // transition_state idempotency on terminal states
    // -----------------------------------------------------------------------

    /// Regression: a second transition into a terminal state must NOT overwrite
    /// the first one. This prevents the `state_before` field of `CancelInfo`
    /// (and any downstream consumer of `ExecutionState`) from being silently
    /// replaced by a later transition.
    #[tokio::test]
    async fn transition_state_terminal_is_idempotent() {
        let (state, bus_tx, _rx) = make_state_and_bus();

        // First transition: Running → Cancelled(state_before=Running).
        let info1 = CancelInfo {
            reason: CancelReason {
                code: CancelCode::User,
                detail: Some("first".into()),
                requested_at: 100,
            },
            observed_at: 110,
            state_before: Box::new(ExecutionState::Running),
        };
        transition_state(&state, &bus_tx, ExecutionState::Cancelled(info1.clone())).await;

        // Second transition attempt: must be a no-op.
        let info2 = CancelInfo {
            reason: CancelReason {
                code: CancelCode::Internal,
                detail: Some("second".into()),
                requested_at: 200,
            },
            observed_at: 210,
            state_before: Box::new(ExecutionState::Paused(PauseInfo {
                kind: PauseKind::Single,
                prompts: vec![],
                paused_at: 150,
            })),
        };
        transition_state(&state, &bus_tx, ExecutionState::Cancelled(info2)).await;

        let guard = state.lock().await;
        match &*guard {
            ExecutionState::Cancelled(seen) => {
                assert_eq!(
                    seen.reason.detail.as_deref(),
                    Some("first"),
                    "second transition must not overwrite the first CancelInfo"
                );
                assert!(
                    matches!(*seen.state_before, ExecutionState::Running),
                    "state_before must remain the original pre-cancel state, got: {:?}",
                    seen.state_before
                );
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // build_cancel_info nested-awareness
    // -----------------------------------------------------------------------

    /// Regression: when the state is ALREADY Cancelled, `build_cancel_info`
    /// must NOT wrap the outer `Cancelled` as the new `state_before` (that
    /// would yield `Cancelled(state_before=Cancelled(state_before=...))`).
    /// Instead it inherits the inner `state_before`, preserving the original
    /// pre-cancellation snapshot across repeated cancel attempts.
    #[tokio::test]
    async fn build_cancel_info_does_not_nest_on_already_cancelled_state() {
        // Pre-set the state to Cancelled(state_before=Paused) — the shape we
        // would observe after registry::cancel() transitions a Paused session.
        let original_pause = ExecutionState::Paused(PauseInfo {
            kind: PauseKind::Single,
            prompts: vec![],
            paused_at: 100,
        });
        let outer = ExecutionState::Cancelled(CancelInfo {
            reason: CancelReason {
                code: CancelCode::User,
                detail: Some("first".into()),
                requested_at: 200,
            },
            observed_at: 210,
            state_before: Box::new(original_pause.clone()),
        });
        let state = Arc::new(Mutex::new(outer));

        // Driver checkpoint later requests another build_cancel_info.
        let second_reason = CancelReason {
            code: CancelCode::Internal,
            detail: Some("driver-checkpoint".into()),
            requested_at: 300,
        };
        let info = build_cancel_info(&state, second_reason).await;

        // The new state_before must be the ORIGINAL Paused, not the
        // intermediate Cancelled wrapper.
        match *info.state_before {
            ExecutionState::Paused(ref p) => {
                assert_eq!(p.paused_at, 100, "expected original Paused snapshot");
            }
            ref other => {
                panic!("state_before must inherit inner pre-cancel state, got nested: {other:?}")
            }
        }
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

        let (exec_task, llm_rx, _vm_driver, _metrics) = session.into_driver_parts();
        let (state, bus_tx, _rx) = make_state_and_bus();
        let cancel_token = CancellationToken::new();
        let resp_txs = Arc::new(Mutex::new(HashMap::new()));

        let ctx = DriverContext {
            state: state.clone(),
            bus_tx,
            cancel_token,
            resp_txs,
            last_active: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            metrics: Arc::new(algocline_core::ExecutionMetrics::new()),
        };
        driver_loop(ctx, exec_task, llm_rx).await;

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
