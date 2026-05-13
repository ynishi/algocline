//! `BroadcastObserverHandle` — concrete [`ObserverHandle`] implementation backed
//! by a `tokio::sync::broadcast::Receiver<ProgressEvent>`.
//!
//! Multiple handles may exist concurrently; each receives the full event stream
//! independently (sink-free fan-out, Crux R3).

use algocline_core::execution::{ObserverHandle, ObserverRecvError, ProgressEvent};
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// BroadcastObserverHandle
// ---------------------------------------------------------------------------

/// Wraps a `broadcast::Receiver<ProgressEvent>` and implements the
/// [`ObserverHandle`] trait defined in `algocline-core`.
///
/// Obtaining a handle is synchronous (`broadcast::Sender::subscribe()` does not
/// perform I/O) and the handle is immediately valid — no pre-registered sink is
/// required (Crux R3 / design-v1.md §5.4).
pub struct BroadcastObserverHandle {
    rx: broadcast::Receiver<ProgressEvent>,
}

impl BroadcastObserverHandle {
    /// Create a new handle by subscribing to the given broadcast sender.
    pub(crate) fn new(tx: &broadcast::Sender<ProgressEvent>) -> Self {
        Self { rx: tx.subscribe() }
    }
}

impl ObserverHandle for BroadcastObserverHandle {
    /// Await the next [`ProgressEvent`] from the broadcast channel.
    ///
    /// - Returns `Ok(event)` on success.
    /// - Returns `Err(ObserverRecvError::Lagged(n))` when the receiver fell behind
    ///   and `n` events were skipped; the next call resumes from the latest available
    ///   event (no silent drop — the gap is reported).
    /// - Returns `Err(ObserverRecvError::Closed)` when the sender is dropped
    ///   (session terminated).
    fn recv(
        &mut self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ProgressEvent, ObserverRecvError>> + Send + '_>,
    > {
        Box::pin(async move {
            match self.rx.recv().await {
                Ok(event) => Ok(event),
                Err(broadcast::error::RecvError::Lagged(n)) => Err(ObserverRecvError::Lagged(n)),
                Err(broadcast::error::RecvError::Closed) => Err(ObserverRecvError::Closed),
            }
        })
    }

    /// Non-blocking receive.
    ///
    /// Returns an event immediately if one is available, or an appropriate
    /// `Err` variant otherwise.
    fn try_recv(&mut self) -> Result<ProgressEvent, ObserverRecvError> {
        match self.rx.try_recv() {
            Ok(event) => Ok(event),
            Err(broadcast::error::TryRecvError::Empty) => {
                // No event available right now — map to Closed so callers
                // treat it as "nothing to drain", consistent with the `Empty`
                // variant not existing on `ObserverRecvError`.
                // Callers that need non-empty semantics should use `recv()`.
                Err(ObserverRecvError::Closed)
            }
            Err(broadcast::error::TryRecvError::Lagged(n)) => Err(ObserverRecvError::Lagged(n)),
            Err(broadcast::error::TryRecvError::Closed) => Err(ObserverRecvError::Closed),
        }
    }

    fn close(self: Box<Self>) {
        // Drop self — the receiver is automatically unsubscribed when dropped.
        // No explicit close call is needed on `broadcast::Receiver`.
        drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::execution::{ExecutionStateTag, ProgressEvent};
    use tokio::sync::broadcast;

    fn make_state_transition(from: ExecutionStateTag, to: ExecutionStateTag) -> ProgressEvent {
        ProgressEvent::StateTransition { from, to, at: 0 }
    }

    // -----------------------------------------------------------------------
    // observe_sink_free (Crux R3)
    // -----------------------------------------------------------------------

    /// Sink-free property (Crux R3): a new observer can subscribe at any point
    /// in the session lifetime without any pre-registration requirement.
    /// Events published before subscription are NOT delivered (broadcast does not
    /// buffer past events for new subscribers), but all events published after
    /// subscription are received correctly.
    ///
    /// Note: `tokio::sync::broadcast::Sender::send` returns `Err` when there are
    /// zero active receivers — this is expected tokio behavior.  The driver_loop
    /// uses `let _ = bus_tx.send(...)` to tolerate zero-receiver sends gracefully.
    /// Sink-free means "no pre-registration is required to subscribe", not that
    /// send must succeed with 0 receivers.
    #[tokio::test]
    async fn observe_sink_free() {
        let (tx, _initial_rx) = broadcast::channel::<ProgressEvent>(256);
        // Drop the initial receiver → 0 subscribers.
        drop(_initial_rx);

        // Send with 0 receivers: tokio broadcast returns Err here, which is
        // expected.  The driver tolerates this with `let _ = bus_tx.send(...)`.
        let event = make_state_transition(ExecutionStateTag::Running, ExecutionStateTag::Done);
        let _ = tx.send(event); // result intentionally ignored

        // Subscribe after the above send; the historical event is NOT delivered
        // (broadcast does not buffer past events for new subscribers).
        let mut handle = BroadcastObserverHandle::new(&tx);

        // Publish a new event — the fresh subscriber must receive it.
        let new_event =
            make_state_transition(ExecutionStateTag::Running, ExecutionStateTag::Paused);
        tx.send(new_event).expect("send after subscribe");

        let received = handle.recv().await.expect("recv after subscribe");
        assert!(matches!(
            received,
            ProgressEvent::StateTransition {
                to: ExecutionStateTag::Paused,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // observe_multi_subscriber_fan_out (Crux R3)
    // -----------------------------------------------------------------------

    /// N=3 independent receivers each get the same event without interfering.
    #[tokio::test]
    async fn observe_multi_subscriber_fan_out() {
        let (tx, _initial_rx) = broadcast::channel::<ProgressEvent>(256);
        drop(_initial_rx);

        let mut h1 = BroadcastObserverHandle::new(&tx);
        let mut h2 = BroadcastObserverHandle::new(&tx);
        let mut h3 = BroadcastObserverHandle::new(&tx);

        let events = vec![
            make_state_transition(ExecutionStateTag::Running, ExecutionStateTag::Paused),
            make_state_transition(ExecutionStateTag::Paused, ExecutionStateTag::Running),
            make_state_transition(ExecutionStateTag::Running, ExecutionStateTag::Done),
        ];

        for e in &events {
            tx.send(e.clone()).expect("send");
        }

        // Drop tx so receivers eventually get Closed after draining.
        drop(tx);

        for handle in [&mut h1, &mut h2, &mut h3] {
            let mut received = Vec::new();
            loop {
                match handle.recv().await {
                    Ok(e) => received.push(e),
                    Err(ObserverRecvError::Closed) => break,
                    Err(ObserverRecvError::Lagged(n)) => {
                        panic!("unexpected lag: {n}")
                    }
                }
            }
            assert_eq!(
                received.len(),
                3,
                "each subscriber must receive all 3 events"
            );
        }
    }

    // -----------------------------------------------------------------------
    // observe_lagged_observable
    // -----------------------------------------------------------------------

    /// A slow observer that falls behind capacity gets `RecvError::Lagged(n)`.
    #[tokio::test]
    async fn observe_lagged_observable() {
        // Small capacity so overflow is easy to trigger.
        let (tx, _initial_rx) = broadcast::channel::<ProgressEvent>(4);
        drop(_initial_rx);

        let mut slow_handle = BroadcastObserverHandle::new(&tx);

        // Publish more events than the channel capacity (4+1=5 events).
        for _ in 0..5 {
            tx.send(make_state_transition(
                ExecutionStateTag::Running,
                ExecutionStateTag::Running,
            ))
            .ok(); // ok() — we intentionally overflow
        }

        // The slow receiver must see Lagged, not silently drop events.
        let result = slow_handle.recv().await;
        assert!(
            matches!(result, Err(ObserverRecvError::Lagged(_))),
            "slow observer must receive Lagged error, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // terminal_event_closes_receiver
    // -----------------------------------------------------------------------

    /// After the sender is dropped (session terminated), receivers observe Closed.
    #[tokio::test]
    async fn terminal_event_closes_receiver() {
        let (tx, _initial_rx) = broadcast::channel::<ProgressEvent>(256);
        drop(_initial_rx);

        let mut handle = BroadcastObserverHandle::new(&tx);

        // Publish a terminal transition and then drop the sender.
        tx.send(make_state_transition(
            ExecutionStateTag::Running,
            ExecutionStateTag::Done,
        ))
        .expect("send terminal");
        drop(tx);

        // Drain the one event.
        let first = handle.recv().await.expect("terminal event");
        assert!(matches!(
            first,
            ProgressEvent::StateTransition {
                to: ExecutionStateTag::Done,
                ..
            }
        ));

        // Next call must return Closed.
        let result = handle.recv().await;
        assert!(
            matches!(result, Err(ObserverRecvError::Closed)),
            "after sender drop, recv must return Closed, got: {result:?}"
        );
    }
}
