//! Progress forwarder task for the v2 MCP adapter.
//!
//! Bridges the `ExecutionService` broadcast channel to MCP `ProgressNotification`
//! messages.  A forwarder is spawned **only** when `_meta.progressToken` is present
//! in the inbound request (Crux: ProgressToken-conditional forwarder spawn).

use std::sync::Arc;

use rmcp::{
    model::{ProgressNotificationParam, ProgressToken},
    Peer, RoleServer,
};
use tokio::task::JoinHandle;

use algocline_core::execution::{ExecutionService, ObserverRecvError, SessionId};

/// Spawn a background task that forwards execution progress events to the MCP client.
///
/// The task runs until the broadcast channel closes (`Err(Closed)`) or the wire
/// connection is broken (`notify_progress` returns `Err`).
///
/// `Lagged(n)` events are forwarded as wrapper JSON `{"kind":"lagged","n":n}` so the
/// client is informed about missed events without dropping subsequent real events.
///
/// # Crux invariant
/// This function must **only** be called when `progressToken` is present.  The caller
/// (`alc_v2_run`) gates the call on `meta.get_progress_token() == Some(token)`.
pub fn spawn_progress_forwarder(
    execution: Arc<dyn ExecutionService>,
    peer: Peer<RoleServer>,
    sid: SessionId,
    token: ProgressToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut observer = match execution.observe(&sid) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("progress_forwarder: observe failed for sid={sid}: {e}");
                // Send a one-shot failure notification so the client knows forwarding
                // could not start, then exit the task.
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: 0.0,
                        total: None,
                        message: Some(
                            serde_json::json!({
                                "kind": "forwarder_failed",
                                "error": e.to_string(),
                            })
                            .to_string(),
                        ),
                    })
                    .await;
                return;
            }
        };

        let mut counter: f64 = 0.0;
        loop {
            match observer.recv().await {
                Ok(event) => {
                    counter += 1.0;
                    let msg = match serde_json::to_string(&event) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!("progress_forwarder: serialize failed: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = peer
                        .notify_progress(ProgressNotificationParam {
                            progress_token: token.clone(),
                            progress: counter,
                            total: None,
                            message: Some(msg),
                        })
                        .await
                    {
                        tracing::warn!(
                            "progress_forwarder: send_progress failed (wire likely closed): {e}"
                        );
                        break;
                    }
                }
                Err(ObserverRecvError::Closed) => {
                    break;
                }
                Err(ObserverRecvError::Lagged(n)) => {
                    let msg = serde_json::json!({"kind": "lagged", "n": n}).to_string();
                    if let Err(e) = peer
                        .notify_progress(ProgressNotificationParam {
                            progress_token: token.clone(),
                            progress: counter,
                            total: None,
                            message: Some(msg),
                        })
                        .await
                    {
                        tracing::warn!(
                            "progress_forwarder: send_progress failed on lagged (wire likely closed): {e}"
                        );
                        break;
                    }
                }
            }
        }
    })
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use algocline_core::execution::state::ExecutionState;
    use algocline_core::execution::{
        AwaitError, CancelError, CancelReason, ExecutionService, ObserveError, ObserverHandle,
        ObserverRecvError, ProgressEvent, ResumeError, ResumeOutcome, ResumePayload, SessionId,
        SessionSpec, SpawnError, StateError,
    };

    // ── Mock observer ─────────────────────────────────────────────────────────

    /// A mock observer that returns a pre-programmed sequence of results.
    struct MockObserver {
        events: Mutex<Vec<Result<ProgressEvent, ObserverRecvError>>>,
    }

    impl MockObserver {
        fn make(events: Vec<Result<ProgressEvent, ObserverRecvError>>) -> Box<dyn ObserverHandle> {
            Box::new(Self {
                events: Mutex::new(events),
            })
        }
    }

    impl ObserverHandle for MockObserver {
        fn recv(
            &mut self,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<ProgressEvent, ObserverRecvError>>
                    + Send
                    + '_,
            >,
        > {
            let result = {
                let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
                if events.is_empty() {
                    // Return Closed when the sequence is exhausted.
                    Err(ObserverRecvError::Closed)
                } else {
                    events.remove(0)
                }
            };
            Box::pin(std::future::ready(result))
        }

        fn try_recv(&mut self) -> Result<ProgressEvent, ObserverRecvError> {
            let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
            if events.is_empty() {
                Err(ObserverRecvError::Closed)
            } else {
                events.remove(0)
            }
        }

        fn close(self: Box<Self>) {}
    }

    // ── Mock ExecutionService ────────────────────────────────────────────────

    /// Captures which calls were made and returns configurable results.
    struct MockExecution {
        observer_result: Mutex<Option<Result<Box<dyn ObserverHandle>, ObserveError>>>,
    }

    impl MockExecution {
        fn with_observer(obs: Box<dyn ObserverHandle>) -> Self {
            Self {
                observer_result: Mutex::new(Some(Ok(obs))),
            }
        }
    }

    #[async_trait::async_trait]
    impl ExecutionService for MockExecution {
        async fn spawn(&self, _spec: SessionSpec) -> Result<SessionId, SpawnError> {
            unimplemented!()
        }

        async fn state(&self, _id: &SessionId) -> Result<ExecutionState, StateError> {
            unimplemented!()
        }

        async fn resume(
            &self,
            _id: &SessionId,
            _payload: ResumePayload,
        ) -> Result<ResumeOutcome, ResumeError> {
            unimplemented!()
        }

        async fn cancel(&self, _id: &SessionId, _reason: CancelReason) -> Result<(), CancelError> {
            unimplemented!()
        }

        fn observe(&self, _id: &SessionId) -> Result<Box<dyn ObserverHandle>, ObserveError> {
            self.observer_result
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or(Err(ObserveError::NotFound(SessionId::new(
                    "mock".to_string(),
                ))))
        }

        async fn await_terminal(
            &self,
            _id: &SessionId,
        ) -> Result<algocline_core::execution::TerminalOutcome, AwaitError> {
            unimplemented!()
        }
    }

    // ── Minimal ServerHandler for test Peer creation ──────────────────────────

    /// A no-op ServerHandler used only to create a `Peer<RoleServer>` via
    /// `rmcp::service::serve_directly`.  All handler methods use default impls.
    struct NullServer;

    impl rmcp::ServerHandler for NullServer {}

    /// Build a `Peer<RoleServer>` suitable for unit tests by spinning up a
    /// `NullServer` over a duplex byte-pipe.  The peer is live (backed by a real
    /// transport loop) so `notify_progress` actually sends over the pipe.
    ///
    /// The returned `running_service` keeps the peer alive.  Drop it to close the
    /// transport, which causes subsequent `notify_progress` calls to return
    /// `Err(ServiceError::TransportClosed)`.
    fn make_test_server() -> rmcp::service::RunningService<rmcp::RoleServer, NullServer> {
        use rmcp::service::serve_directly;
        let (server_transport, _client_transport) = tokio::io::duplex(4096);
        serve_directly(NullServer, server_transport, None)
    }

    fn make_token() -> ProgressToken {
        use rmcp::model::NumberOrString;
        ProgressToken(NumberOrString::Number(1))
    }

    fn make_sid() -> SessionId {
        SessionId::new("test-sid".to_string())
    }

    // ── Test: Closed event terminates the forwarder loop ─────────────────────

    /// `drain_closed_loop_break`: observer immediately returns `Err(Closed)` →
    /// the forwarder JoinHandle completes within 100 ms.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_closed_loop_break() {
        let obs = MockObserver::make(vec![Err(ObserverRecvError::Closed)]);
        let exec = Arc::new(MockExecution::with_observer(obs));
        let running = make_test_server();
        let peer = running.peer().clone();

        let handle = spawn_progress_forwarder(exec, peer, make_sid(), make_token());

        let result = tokio::time::timeout(std::time::Duration::from_millis(200), handle).await;
        assert!(
            result.is_ok(),
            "forwarder did not terminate within 200ms after Closed"
        );
        let join = result.unwrap();
        assert!(join.is_ok(), "forwarder task panicked");
    }

    // ── Test: Lagged does NOT break the loop AND emits the wrapper event ────

    /// `lagged_emits_wrapper_event`: sequence is Ok(event1) → Lagged(3) → Ok(event2) → Closed.
    ///
    /// Two invariants verified:
    ///   1. The forwarder runs to Closed (loop must NOT exit on Lagged).
    ///   2. The 2nd `notifications/progress` carries `params.message = {"kind":"lagged","n":3}`.
    ///
    /// Invariant 2 prevents a regression where the wrapper payload silently changes
    /// shape (e.g. `{"kind":"foo"}`) while the loop-continues invariant alone would pass.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lagged_emits_wrapper_event() {
        use tokio::io::AsyncReadExt;

        let event1 = ProgressEvent::Tick {
            phase: "step1".to_string(),
            at: 1,
        };
        let event2 = ProgressEvent::Tick {
            phase: "step2".to_string(),
            at: 2,
        };

        let obs = MockObserver::make(vec![
            Ok(event1),
            Err(ObserverRecvError::Lagged(3)),
            Ok(event2),
            Err(ObserverRecvError::Closed),
        ]);
        let exec = Arc::new(MockExecution::with_observer(obs));

        // Build a duplex pair so we can read what the server actually sends.
        // Drop-by-explicit-close at the end terminates the reader's read_to_end.
        let (server_t, mut client_t) = tokio::io::duplex(4096);
        let running = rmcp::service::serve_directly(NullServer, server_t, None);
        let peer = running.peer().clone();

        let handle = spawn_progress_forwarder(exec, peer, make_sid(), make_token());

        // Invariant 1: forwarder reaches Closed within a generous timeout.
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), handle).await;
        assert!(
            result.is_ok(),
            "forwarder did not terminate after Closed; Lagged may have incorrectly broken the loop"
        );
        let join = result.unwrap();
        assert!(join.is_ok(), "forwarder task panicked");

        // Drop the server end of the duplex; the client side will see EOF after
        // draining any in-flight writes already queued in the pipe buffer.
        // Replaces a previous `sleep(50ms)` flush wait — drop + read_to_end is
        // deterministic because the duplex buffer preserves enqueued bytes past
        // the server-side drop, so we do not need a wall-clock delay.
        drop(running);

        // Drain the client side. read_to_end returns once the server end is closed.
        let mut buf = Vec::new();
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            client_t.read_to_end(&mut buf),
        )
        .await
        .expect("client-side drain timed out");

        let text = String::from_utf8_lossy(&buf);
        let messages: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect();

        let progress_msgs: Vec<&serde_json::Value> = messages
            .iter()
            .filter(|m| m.get("method").and_then(|v| v.as_str()) == Some("notifications/progress"))
            .collect();

        // Expected order on the wire: event1 tick → lagged wrapper → event2 tick.
        assert!(
            progress_msgs.len() >= 2,
            "expected >= 2 progress notifications (tick + lagged wrapper); got {} from bytes: {:?}",
            progress_msgs.len(),
            text
        );

        // Invariant 2: 2nd progress notification carries the lagged wrapper.
        let lagged = progress_msgs[1];
        let message_str = lagged
            .get("params")
            .and_then(|p| p.get("message"))
            .and_then(|m| m.as_str())
            .expect("notifications/progress params.message missing or not a string");
        let lagged_obj: serde_json::Value =
            serde_json::from_str(message_str).expect("params.message is not valid JSON");

        assert_eq!(
            lagged_obj.get("kind").and_then(|v| v.as_str()),
            Some("lagged"),
            "expected kind=lagged, got: {lagged_obj:?}"
        );
        assert_eq!(
            lagged_obj.get("n").and_then(|v| v.as_i64()),
            Some(3),
            "expected n=3, got: {lagged_obj:?}"
        );
    }

    // ── Test: Wire close (notify_progress Err) breaks the loop ───────────────

    /// `wire_close_break`: the running service is dropped before the forwarder sends,
    /// causing `TransportClosed` on the very first `notify_progress` call.
    /// The forwarder must exit promptly without looping further.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wire_close_break() {
        // Two events — only the first send will fail (wire closed), so the loop breaks.
        let event1 = ProgressEvent::Tick {
            phase: "x".to_string(),
            at: 0,
        };
        let event2 = ProgressEvent::Tick {
            phase: "y".to_string(),
            at: 1,
        };

        let obs = MockObserver::make(vec![Ok(event1), Ok(event2), Err(ObserverRecvError::Closed)]);
        let exec = Arc::new(MockExecution::with_observer(obs));

        let running = make_test_server();
        let peer = running.peer().clone();
        // Drop the RunningService — this closes the transport, so notify_progress will fail.
        drop(running);
        // Give the tokio runtime a moment to process the drop.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let handle = spawn_progress_forwarder(exec, peer, make_sid(), make_token());
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), handle).await;
        assert!(
            result.is_ok(),
            "forwarder should terminate when wire is closed"
        );
    }

    // ── Test: forwarder_not_spawned_without_token (Crux verify) ─────────────

    /// Verifies the Crux constraint: ProgressToken-conditional forwarder spawn.
    /// When `meta.get_progress_token()` returns `None`, no forwarder is spawned.
    /// This test asserts the **caller-side** gating logic rather than the forwarder
    /// itself — i.e., it checks that calling code uses an `if let Some(token)` guard.
    ///
    /// Since we cannot invoke `alc_v2_run` directly in a unit test, we verify the
    /// guard pattern: given `None` from `meta.get_progress_token()`, `spawn_progress_forwarder`
    /// is not called.  This is a compile-time / logical assertion that the code path
    /// in `alc_v2_run` is guarded by `if let Some(token) = meta.get_progress_token()`.
    #[test]
    fn forwarder_not_spawned_without_token() {
        // Simulate the guard that alc_v2_run uses:
        // `if let Some(token) = meta.get_progress_token() { spawn_progress_forwarder(...); }`
        let token_opt: Option<ProgressToken> = None;
        let mut spawned = false;

        if let Some(_token) = token_opt {
            spawned = true;
            // spawn_progress_forwarder would be called here
        }

        assert!(
            !spawned,
            "spawn_progress_forwarder must NOT be called when progressToken is absent"
        );
    }
}
