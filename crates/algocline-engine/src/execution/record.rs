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
    ///
    /// **Crux R3 (sink-free)**: `send()` is called with `let _ = ...` at every
    /// site in `driver_loop` (`driver.rs`); when 0 receivers are subscribed,
    /// `send` returns `Err(SendError)` and the event is silently dropped
    /// without affecting the caller's control flow.  No sentinel receiver is
    /// held — the contract is "caller is not crashed by 0 observers", not
    /// "`send` always returns `Ok`".  See `bus_tx_does_not_crash_caller_with_zero_observers`.
    pub(crate) bus_tx: broadcast::Sender<ProgressEvent>,

    /// Cooperative cancellation token.
    ///
    /// Calling `.cancel()` sets the internal flag; the driver loop observes it
    /// at exactly four checkpoints (A/B/C/D) and transitions to `Cancelled`.
    /// `.abort()` on the join handle is never called (Crux R2).
    pub(crate) cancel_token: CancellationToken,

    /// Handle to the background driver task.
    ///
    /// Wrapped in `Mutex<Option<...>>` so `await_terminal` can take ownership
    /// and `.await` on the handle directly (single-awaiter semantics) without
    /// busy-polling the `state` mutex.  Subsequent awaiters observe `None` and
    /// fall through to a direct state read (the driver_loop has either already
    /// completed or is about to complete — `await_terminal` returns the
    /// resulting terminal state, or `AwaitError::Joined` in the rare concurrent
    /// race case).
    ///
    /// Never `.abort()`-ed — cancellation uses `cancel_token` only (Crux R2).
    pub(crate) join_handle: Mutex<Option<JoinHandle<()>>>,

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
        let (bus_tx, _) = broadcast::channel(bus_capacity);
        Self {
            state,
            bus_tx,
            cancel_token,
            join_handle: Mutex::new(Some(join_handle)),
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
    async fn bus_tx_does_not_crash_caller_with_zero_observers() {
        // Crux R3 (sink-free): the contract is "caller is not crashed by 0
        // observers".  After removing the sentinel receiver, `bus_tx.send`
        // returns `Err(SendError)` when no receivers are subscribed — that
        // error must be ignorable (the event is dropped, but the caller's
        // control flow is intact).  The production sites in `driver_loop`
        // all use `let _ = bus_tx.send(...)` to enact this invariant.
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
        // 0 receivers → Err(SendError).  Production code drops the result via
        // `let _ = ...`; assert that the call itself does not panic.
        let _ = record.bus_tx.send(event);
    }
}
