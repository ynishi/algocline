//! `SessionRecord` — per-session ownership bundle for the v2 execution path.
//!
//! Holds the shared state, the progress broadcast sender, the cancellation token,
//! the background task handle, and the per-query oneshot senders used to resume
//! a paused Lua coroutine.
//!
//! `SessionRecord` is intentionally free of any MCP / rmcp concepts (Crux R1).
//! All cancellation uses [`tokio_util::sync::CancellationToken`]; no
//! `JoinHandle::abort()` path exists (Crux R2).

use std::collections::HashMap;
use std::sync::Arc;

use algocline_core::execution::{CancelInfo, ExecutionState};
use algocline_core::QueryId;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use algocline_core::execution::ProgressEvent;

// ---------------------------------------------------------------------------
// SessionRecord
// ---------------------------------------------------------------------------

/// Type alias for the shared resp_txs map.
///
/// `Arc<Mutex<...>>` so the driver_loop and the registry can share the same map
/// without a circular ownership relationship.
pub(crate) type RespTxsMap =
    Arc<Mutex<HashMap<QueryId, tokio::sync::oneshot::Sender<Result<String, String>>>>>;

/// Ownership bundle for a single v2 execution session.
///
/// Created by [`super::registry::SessionRegistryV2::spawn_v2`] and kept alive
/// inside the registry's `HashMap` until the session reaches a terminal state
/// (or is explicitly removed by GC).
///
/// Drop order:
/// 1. `join_handle` — task has already completed (normal path) or panicked.
/// 2. `bus_tx` — triggers `RecvError::Closed` on every open receiver.
/// 3. `cancel_token` — releasing any child tokens.
pub struct SessionRecord {
    /// Shared execution state (v2), protected by a per-session `Mutex`.
    ///
    /// The driver loop is the only writer; readers use `clone-then-release`
    /// to avoid holding the lock across `.await` points (Crux K-4).
    pub(crate) state: Arc<Mutex<ExecutionState>>,

    /// Broadcast sender for [`ProgressEvent`]s.
    ///
    /// Capacity 256 (design-v1.md §5.1).  When this field is dropped, every
    /// open `broadcast::Receiver` observes `RecvError::Closed` — signalling
    /// session termination (Crux R3 / design-v1.md §5.6).
    pub(crate) bus_tx: broadcast::Sender<ProgressEvent>,

    /// Sentinel receiver that keeps `bus_tx` alive even with 0 user observers.
    ///
    /// `tokio::sync::broadcast::Sender::send` returns `Err` when there are no
    /// active receivers.  Holding this sentinel ensures `bus_tx.send()` always
    /// returns `Ok(n)` (n ≥ 1) regardless of how many user subscribers exist,
    /// satisfying the sink-free requirement (Crux R3 / design-v1.md §5.4).
    ///
    /// Events accumulate in the sentinel's buffer; it is never read, so they
    /// are silently discarded when the record is dropped.
    #[allow(dead_code)]
    pub(crate) _sentinel_rx: broadcast::Receiver<ProgressEvent>,

    /// Cooperative cancellation token.
    ///
    /// Calling `.cancel()` sets the internal flag; the driver loop observes it
    /// at exactly four checkpoints (A/B/C/D) and transitions to `Cancelled`.
    /// `.abort()` on the join handle is never called (Crux R2).
    pub(crate) cancel_token: CancellationToken,

    /// Handle to the background driver task.
    ///
    /// Held so that `await_terminal` can join without polling state repeatedly.
    /// Never `.abort()`-ed — cancellation uses `cancel_token` only (Crux R2).
    #[allow(dead_code)]
    pub(crate) join_handle: JoinHandle<()>,

    /// Per-query oneshot senders to wake the paused Lua coroutine.
    ///
    /// Shared via `Arc<Mutex<...>>` between this record and the driver_loop task
    /// so that `resume()` can deliver responses into the same map the driver reads.
    ///
    /// Populated when the driver publishes `PauseRequested`; consumed by
    /// `resume()`.
    pub(crate) resp_txs: RespTxsMap,

    /// Stores the first `CancelInfo` observed for idempotent cancel (Crux R2).
    ///
    /// `Some` once `cancel()` has been called; subsequent calls return `Ok(())`
    /// without overwriting this entry.
    pub(crate) first_cancel_info: Mutex<Option<CancelInfo>>,
}

impl SessionRecord {
    /// Create a new `SessionRecord`.
    ///
    /// `bus_capacity` is the broadcast channel buffer size (typically 256).
    #[cfg(test)]
    pub(crate) fn new(
        state: Arc<Mutex<ExecutionState>>,
        bus_capacity: usize,
        cancel_token: CancellationToken,
        join_handle: JoinHandle<()>,
        resp_txs: RespTxsMap,
    ) -> Self {
        let (bus_tx, sentinel_rx) = broadcast::channel(bus_capacity);
        Self {
            state,
            bus_tx,
            _sentinel_rx: sentinel_rx,
            cancel_token,
            join_handle,
            resp_txs,
            first_cancel_info: Mutex::new(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::execution::ExecutionState;
    use tokio::task;

    #[tokio::test]
    async fn record_created_with_running_state() {
        let state = Arc::new(Mutex::new(ExecutionState::Running));
        let cancel_token = CancellationToken::new();
        let handle = task::spawn(async {});
        let resp_txs: RespTxsMap = Arc::new(Mutex::new(HashMap::new()));
        let record = SessionRecord::new(state.clone(), 256, cancel_token, handle, resp_txs);

        let guard = record.state.lock().await;
        assert!(matches!(*guard, ExecutionState::Running));
    }

    #[tokio::test]
    async fn bus_tx_send_succeeds_with_zero_observers() {
        // Crux R3: sink-free — send succeeds even with no subscribers.
        use algocline_core::execution::{ExecutionStateTag, ProgressEvent};

        let state = Arc::new(Mutex::new(ExecutionState::Running));
        let cancel_token = CancellationToken::new();
        let handle = task::spawn(async {});
        let resp_txs: RespTxsMap = Arc::new(Mutex::new(HashMap::new()));
        let record = SessionRecord::new(state, 256, cancel_token, handle, resp_txs);

        let event = ProgressEvent::StateTransition {
            from: ExecutionStateTag::Running,
            to: ExecutionStateTag::Done,
            at: 0,
        };
        // `send` returns Ok(receiver_count); 0 receivers is NOT an error.
        let result = record.bus_tx.send(event);
        assert!(
            result.is_ok(),
            "send with 0 receivers must succeed (sink-free), got: {result:?}"
        );
    }
}
