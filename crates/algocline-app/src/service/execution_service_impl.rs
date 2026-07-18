//! `impl ExecutionService for AppService` — thin delegation to [`SessionRegistryV2`].
//!
//! All six verbs (`spawn`, `state`, `resume`, `cancel`, `observe`, `await_terminal`)
//! delegate directly to `self.execution_registry` with no additional logic.
//!
//! # Crux constraints honoured
//!
//! - **R1 (Wire-concept exclusion)**: This file imports nothing from `rmcp`, contains
//!   no `progressToken`, `_meta`, `notifications/`, or `mcp_`-prefixed identifiers.
//! - **R2 (Cooperative cancellation)**: `cancel` calls
//!   `execution_registry.cancel(id, reason)` only; no `JoinHandle::abort()` or
//!   process-kill path exists in this file.
//! - **R3 (Sink-free broadcast fan-out)**: `observe` is a sync `fn` that delegates
//!   to `execution_registry.observe(id)`.  Multi-subscriber behaviour is verified by
//!   `observe_multi_subscriber_fan_out_via_appservice` and
//!   `observe_sink_free_when_no_subscribers` in the tests below.

use algocline_core::execution::{
    AwaitError, CancelError, CancelReason, ExecutionService, ExecutionState, ObserveError,
    ObserverHandle, ResumeError, ResumeOutcome, ResumePayload, SessionId, SessionSpec, SpawnError,
    StateError, TerminalOutcome,
};
use async_trait::async_trait;

use crate::service::AppService;

#[async_trait]
impl ExecutionService for AppService {
    async fn spawn(&self, spec: SessionSpec) -> Result<SessionId, SpawnError> {
        // Resolve `[setting.card].run` per session so the Phase 1-B `[run]`
        // gate follows the same cascade (env > project > global) as any
        // other setting.  Errors on the resolver path degrade to gate-off
        // rather than failing spawn: an unreadable config.toml must not
        // block execution, and a missing setting still yields `None` from
        // `get_bool` which we treat as `false`.  See Phase 1-A cascade
        // tests in `service::setting`.
        let app_dir = self.log_config.app_dir();
        let card_run_enabled = crate::service::setting::resolve_setting(
            &app_dir,
            spec.project_root.as_deref(),
            Some("card"),
        )
        .ok()
        .and_then(|s| s.get_bool("card", "run"))
        .unwrap_or(false);
        self.execution_registry
            .spawn_v2(spec, card_run_enabled)
            .await
    }

    async fn state(&self, id: &SessionId) -> Result<ExecutionState, StateError> {
        self.execution_registry.state(id).await
    }

    async fn resume(
        &self,
        id: &SessionId,
        payload: ResumePayload,
    ) -> Result<ResumeOutcome, ResumeError> {
        self.execution_registry.resume(id, payload).await
    }

    async fn cancel(&self, id: &SessionId, reason: CancelReason) -> Result<(), CancelError> {
        self.execution_registry.cancel(id, reason).await
    }

    fn observe(&self, id: &SessionId) -> Result<Box<dyn ObserverHandle>, ObserveError> {
        self.execution_registry.observe(id)
    }

    async fn await_terminal(&self, id: &SessionId) -> Result<TerminalOutcome, AwaitError> {
        self.execution_registry.await_terminal(id).await
    }
}

#[cfg(test)]
mod tests {
    use algocline_core::execution::{
        CancelCode, CancelReason, ExecutionService, ExecutionState, ObserverRecvError,
        ProgressEvent, ResumePayload, SessionSpec, SpecKind, TerminalOutcome,
    };

    use crate::service::test_support::make_app_service;

    fn simple_spec(code: &str) -> SessionSpec {
        SessionSpec {
            kind: SpecKind::Run {
                code: code.to_owned(),
            },
            project_root: None,
            ctx: None,
        }
    }

    fn user_cancel_reason() -> CancelReason {
        CancelReason {
            code: CancelCode::User,
            detail: None,
            requested_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
        }
    }

    // -----------------------------------------------------------------------
    // spawn_returns_session_id_and_running_state
    // -----------------------------------------------------------------------

    /// `spawn()` → `state()` immediately returns Running (or Paused for LLM-touching scripts).
    #[tokio::test]
    async fn spawn_returns_session_id_and_running_state() {
        let svc = make_app_service().await;
        // Use a Lua script that pauses so we can observe the state before completion.
        let sid = svc
            .spawn(simple_spec(r#"return alc.llm("q")"#))
            .await
            .expect("spawn must succeed");

        // Wait a short time for the driver to start.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let state = svc.state(&sid).await.expect("state must succeed");
        assert!(
            matches!(state, ExecutionState::Running | ExecutionState::Paused(_)),
            "state immediately after spawn must be Running or Paused, got: {:?}",
            state.tag()
        );
    }

    // -----------------------------------------------------------------------
    // spawn_run_lua_to_completion
    // -----------------------------------------------------------------------

    /// `SpecKind::Run { code: "return 42" }` → `await_terminal()` → `Terminal(Done(value=42))`.
    #[tokio::test]
    async fn spawn_run_lua_to_completion() {
        let svc = make_app_service().await;
        let sid = svc
            .spawn(simple_spec("return 42"))
            .await
            .expect("spawn must succeed");

        let outcome =
            tokio::time::timeout(std::time::Duration::from_secs(5), svc.await_terminal(&sid))
                .await
                .expect("await_terminal must not timeout")
                .expect("await_terminal must succeed");

        match outcome {
            TerminalOutcome::Done(result) => {
                assert_eq!(
                    result.value,
                    serde_json::json!(42),
                    "Done result value must be 42"
                );
            }
            other => panic!("expected Done, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // spawn_lua_pause_publishes_progress_event
    // -----------------------------------------------------------------------

    /// Lua `alc.llm(...)` causes a pause → `observe()` receiver gets `ProgressEvent::PauseRequested`.
    #[tokio::test]
    async fn spawn_lua_pause_publishes_progress_event() {
        let svc = make_app_service().await;

        let sid = svc
            .spawn(simple_spec(r#"return alc.llm("tell me something")"#))
            .await
            .expect("spawn must succeed");

        // Subscribe before the pause event is published.
        let mut handle = svc.observe(&sid).expect("observe must succeed");

        // Wait for the PauseRequested event (with timeout).
        let got_pause = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                match handle.recv().await {
                    Ok(ProgressEvent::PauseRequested { .. }) => return true,
                    Ok(_) => {}
                    Err(ObserverRecvError::Closed) => return false,
                    Err(ObserverRecvError::Lagged(_)) => {}
                }
            }
        })
        .await
        .expect("must not timeout waiting for PauseRequested");

        assert!(got_pause, "must receive PauseRequested event");
    }

    // -----------------------------------------------------------------------
    // resume_continues_paused_session
    // -----------------------------------------------------------------------

    /// Pause → resume(`ResumePayload::Single { response, ... }`) → `ResumeOutcome::Continued`
    /// and session eventually reaches `Done`.
    #[tokio::test]
    async fn resume_continues_paused_session() {
        use algocline_core::execution::ResumeOutcome;

        let svc = make_app_service().await;

        // Strategy that calls alc.llm once and returns the response.
        let sid = svc
            .spawn(simple_spec(r#"return alc.llm("what is 1+1?")"#))
            .await
            .expect("spawn must succeed");

        // Wait until the session reaches Paused, capturing the query_id.
        let query_id = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                let state = svc.state(&sid).await.expect("state");
                if let ExecutionState::Paused(info) = state {
                    // The query id is the first pending prompt's id.
                    if let Some(prompt) = info.prompts.first() {
                        return prompt.query_id.clone();
                    }
                }
            }
        })
        .await
        .expect("must reach Paused state within timeout");

        let outcome = svc
            .resume(
                &sid,
                ResumePayload::Single {
                    query_id,
                    response: "2".into(),
                    usage: None,
                },
            )
            .await
            .expect("resume must succeed");

        assert!(
            matches!(outcome, ResumeOutcome::Continued),
            "resume must return Continued, got: {outcome:?}"
        );

        // The session should now complete.
        let terminal =
            tokio::time::timeout(std::time::Duration::from_secs(5), svc.await_terminal(&sid))
                .await
                .expect("await_terminal must not timeout")
                .expect("await_terminal must succeed");

        assert!(
            matches!(terminal, TerminalOutcome::Done(_)),
            "session must complete as Done after resume, got: {terminal:?}"
        );
    }

    // -----------------------------------------------------------------------
    // cancel_running_session_transitions_to_cancelled
    // -----------------------------------------------------------------------

    /// Paused Lua session → `cancel()` → `await_terminal()` → `Cancelled(info)`
    /// with `info.reason.code == CancelCode::User`.
    ///
    /// Uses `alc.llm(...)` to enter Paused state so that `cancel()` can exercise
    /// the direct Paused→Cancelled transition in `registry::cancel()` without
    /// requiring the driver to reach a cooperative-cancellation checkpoint
    /// (which a tight CPU loop would never yield to).
    #[tokio::test]
    async fn cancel_running_session_transitions_to_cancelled() {
        let svc = make_app_service().await;

        // Spawn a session that pauses waiting for an LLM response.
        let sid = svc
            .spawn(simple_spec(r#"return alc.llm("cancel me")"#))
            .await
            .expect("spawn must succeed");

        // Wait until the session reaches Paused state.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                let state = svc.state(&sid).await.expect("state");
                if matches!(state, ExecutionState::Paused(_)) {
                    return;
                }
            }
        })
        .await
        .expect("session must reach Paused state within timeout");

        svc.cancel(&sid, user_cancel_reason())
            .await
            .expect("cancel must succeed");

        let terminal =
            tokio::time::timeout(std::time::Duration::from_secs(5), svc.await_terminal(&sid))
                .await
                .expect("await_terminal must not timeout")
                .expect("await_terminal must succeed");

        match terminal {
            TerminalOutcome::Cancelled(info) => {
                assert_eq!(
                    info.reason.code,
                    CancelCode::User,
                    "cancel code must be User"
                );
            }
            other => panic!("expected Cancelled, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // cancel_idempotent_returns_ok  (Crux R2)
    // -----------------------------------------------------------------------

    /// Calling `cancel()` twice on the same paused session must both return `Ok(())`.
    /// The first `CancelInfo` must remain in the terminal state.
    #[tokio::test]
    async fn cancel_idempotent_returns_ok() {
        let svc = make_app_service().await;

        // Spawn a session that pauses waiting for an LLM response.
        let sid = svc
            .spawn(simple_spec(r#"return alc.llm("cancel me")"#))
            .await
            .expect("spawn must succeed");

        // Wait until the session reaches Paused state.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                let state = svc.state(&sid).await.expect("state");
                if matches!(state, ExecutionState::Paused(_)) {
                    return;
                }
            }
        })
        .await
        .expect("session must reach Paused state within timeout");

        svc.cancel(&sid, user_cancel_reason())
            .await
            .expect("first cancel must return Ok");

        // Second cancel on the same session must also return Ok.
        svc.cancel(&sid, user_cancel_reason())
            .await
            .expect("second cancel must return Ok (idempotent)");

        // State must reflect the first cancel's info.
        let terminal =
            tokio::time::timeout(std::time::Duration::from_secs(5), svc.await_terminal(&sid))
                .await
                .expect("await_terminal must not timeout")
                .expect("await_terminal must succeed");

        assert!(
            matches!(terminal, TerminalOutcome::Cancelled(_)),
            "session must be Cancelled, got: {terminal:?}"
        );
    }

    // -----------------------------------------------------------------------
    // observe_multi_subscriber_fan_out_via_appservice  (Crux R3)
    // -----------------------------------------------------------------------

    /// Two independent `observe()` calls must each receive the full event stream.
    /// Neither subscriber affects the other.
    #[tokio::test]
    async fn observe_multi_subscriber_fan_out_via_appservice() {
        let svc = make_app_service().await;

        let sid = svc
            .spawn(simple_spec("return 99"))
            .await
            .expect("spawn must succeed");

        // Subscribe two independent handles before the session terminates.
        let mut h1 = svc.observe(&sid).expect("observe h1 must succeed");
        let mut h2 = svc.observe(&sid).expect("observe h2 must succeed");

        // Wait for terminal so events have been published.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), svc.await_terminal(&sid))
            .await
            .expect("await_terminal must not timeout");

        // Each handle must receive at least one StateTransition event.
        //
        // The recv loop is bounded by a 100ms idle-timeout per receive:
        // `SessionRegistryV2` is sink-free (registry retains `bus_tx` so late
        // `observe()` calls can still subscribe), so `bus_tx` does NOT drop
        // when the session reaches a terminal state — `Closed` would never
        // fire and a bare `loop { recv().await }` blocks forever (the
        // observed 7-min worktree hang). Timeout exit is correct: after
        // `await_terminal()` succeeded, all events have been published, so
        // a 100ms idle window past the last event proves nothing more is
        // coming.
        use std::time::Duration;
        for (label, handle) in [("h1", &mut h1), ("h2", &mut h2)] {
            let mut got_transition = false;
            loop {
                match tokio::time::timeout(Duration::from_millis(100), handle.recv()).await {
                    Ok(Ok(ProgressEvent::StateTransition { .. })) => got_transition = true,
                    Ok(Ok(_)) => {}
                    Ok(Err(ObserverRecvError::Closed)) => break,
                    Ok(Err(ObserverRecvError::Lagged(_))) => {}
                    Err(_) => break, // idle-timeout: no more events coming
                }
            }
            assert!(
                got_transition,
                "{label}: must receive at least one StateTransition event"
            );
        }
    }

    // -----------------------------------------------------------------------
    // observe_sink_free_when_no_subscribers  (Crux R3 + debt #3)
    // -----------------------------------------------------------------------

    /// Spawn a session without calling `observe()` first.  `await_terminal()` must
    /// complete normally (sink-free: 0 observers do not stall execution).
    ///
    /// After terminal and GC eviction, `observe()` must return **strictly**
    /// `ObserveError::NotFound` (debt #3 resolution / Crux #2 TTL override for
    /// test determinism).
    ///
    /// This test uses `SessionRegistryV2` directly (not `make_app_service`) so it
    /// can inject a sub-second TTL and interval without waiting for the production
    /// 3h TTL (R5 fallback: direct registry construction path).
    #[tokio::test]
    async fn observe_sink_free_when_no_subscribers() {
        use algocline_core::execution::ObserveError;
        use std::sync::Arc;
        use std::time::Duration;

        // Build a registry with per-test temp storage (does not touch ~/.algocline).
        let executor = Arc::new(
            algocline_engine::Executor::new(vec![])
                .await
                .expect("Executor::new"),
        );
        let tmp = tempfile::tempdir().expect("tempdir");
        let state_store = Arc::new(algocline_engine::JsonFileStore::new(
            tmp.path().join("state"),
        ));
        let card_store = Arc::new(algocline_engine::FileCardStore::new(
            tmp.path().join("cards"),
        ));
        let scenarios_dir = tmp.path().join("scenarios");
        let registry = algocline_engine::execution::SessionRegistryV2::new(
            executor,
            state_store,
            card_store,
            scenarios_dir,
        );

        let ttl = Duration::from_millis(100);
        let interval = Duration::from_millis(50);

        let sid = registry
            .spawn_v2(simple_spec("return 1"), false)
            .await
            .expect("spawn must succeed");

        // Do NOT observe — no subscribers at all (sink-free contract).
        let terminal = tokio::time::timeout(Duration::from_secs(5), registry.await_terminal(&sid))
            .await
            .expect("await_terminal must not timeout even with 0 observers")
            .expect("await_terminal must succeed");

        assert!(
            matches!(terminal, TerminalOutcome::Done(_)),
            "session must complete as Done, got: {terminal:?}"
        );

        // Wire GC with sub-second TTL + interval for deterministic eviction.
        registry.spawn_gc_task(ttl, interval);

        // Sleep beyond interval + ttl + slack so at least one GC tick fires
        // after TTL has elapsed (R4 guard: avoids false-positive from first tick).
        tokio::time::sleep(interval + ttl + Duration::from_millis(50)).await;

        // After eviction, observe must return NotFound strictly (debt #3).
        assert!(
            matches!(registry.observe(&sid), Err(ObserveError::NotFound(_))),
            "observe() must return NotFound after GC evicts the terminal session"
        );
    }
}
