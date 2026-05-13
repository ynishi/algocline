//! `ExecutionService` trait — the primary verb surface of the new service layer.

use async_trait::async_trait;

use super::cancel::CancelReason;
use super::error::{AwaitError, CancelError, ObserveError, ResumeError, SpawnError, StateError};
use super::progress::ObserverHandle;
use super::resume::{ResumeOutcome, TerminalOutcome};
use super::session_id::SessionId;
use super::spec::SessionSpec;
use super::state::ExecutionState;

/// Pure service trait for managing execution sessions.
///
/// Implementors must ensure the following invariants hold:
///
/// 1. **Wire-concept exclusion**: no MCP/rmcp types, `progressToken`, `_meta` fields,
///    `notifications/*` paths, or `mcp_`-prefixed identifiers are referenced by this
///    trait or its supporting types.
/// 2. **`SessionId` is the sole handle**: all verbs after `spawn` accept only a
///    `&SessionId`; no session-internal handles leak through the API.
/// 3. **Pure value types**: all inputs and outputs are value types (no callbacks,
///    no `Arc`-leaked internals).
/// 4. **Cooperative cancellation only**: `cancel` signals the session via a
///    `CancellationToken`; no `JoinHandle::abort()` or process kill may be invoked.
/// 5. **Sink-free progress fan-out**: `observe` returns a `broadcast::Receiver`
///    wrapper that is valid without any pre-registered observer.  Multiple concurrent
///    subscribers each receive the full event stream independently.
/// 6. **Immediate `SessionId` return**: `spawn` returns the `SessionId` before
///    execution completes.
/// 7. **Single trait, all verbs**: CLI, Server, and programmatic callers all use
///    this single trait.
///
/// # Async vs sync verbs
/// All verbs are `async fn` except `observe`, which is a synchronous `fn`.
/// `observe` is sync because `broadcast::Sender::subscribe()` is itself synchronous
/// and does not perform I/O — making it `async` would force callers to `.await`
/// without reason and obscure the sink-free subscription semantics.
#[async_trait]
pub trait ExecutionService: Send + Sync {
    /// Spawn a new execution session from the given specification.
    ///
    /// Returns the `SessionId` immediately; execution proceeds in the background.
    ///
    /// # Errors
    /// - [`SpawnError::Engine`] — the engine failed to initialize the session.
    /// - [`SpawnError::InvalidSpec`] — the provided `SessionSpec` is malformed.
    async fn spawn(&self, spec: SessionSpec) -> Result<SessionId, SpawnError>;

    /// Query the current state of a session.
    ///
    /// # Errors
    /// - [`StateError::NotFound`] — no session with the given id exists.
    async fn state(&self, id: &SessionId) -> Result<ExecutionState, StateError>;

    /// Resume a paused session by providing LLM responses.
    ///
    /// The `payload` kind must match the pause kind of the session (single vs batch).
    ///
    /// # Errors
    /// - [`ResumeError::NotFound`] — no session with the given id exists.
    /// - [`ResumeError::NotPaused`] — the session is not in the `Paused` state.
    /// - [`ResumeError::AlreadyCancelled`] — the session is already cancelled.
    /// - [`ResumeError::FeedError`] — the underlying feed operation failed.
    async fn resume(
        &self,
        id: &SessionId,
        payload: super::resume::ResumePayload,
    ) -> Result<ResumeOutcome, ResumeError>;

    /// Request cooperative cancellation of a session.
    ///
    /// `cancel` is idempotent: calling it on a session already in a terminal state
    /// (`Done`, `Failed`, `Cancelled`) returns `Ok(())`.  The only error case is when
    /// no session with the given id exists.
    ///
    /// Cancellation is delivered via a `CancellationToken`; the engine checks at
    /// exactly four checkpoints (A/B/C/D) and transitions cooperatively to `Cancelled`.
    /// No `JoinHandle::abort()` or process kill path is introduced.
    ///
    /// # Errors
    /// - [`CancelError::NotFound`] — no session with the given id exists.
    async fn cancel(&self, id: &SessionId, reason: CancelReason) -> Result<(), CancelError>;

    /// Subscribe to the progress event stream for a session.
    ///
    /// Returns a `Box<dyn ObserverHandle>` that delivers [`super::progress::ProgressEvent`]
    /// events from the session's broadcast channel.  Multiple concurrent subscribers
    /// each receive the full event stream independently (sink-free fan-out).
    ///
    /// This is a **synchronous** `fn` (not `async`) because
    /// `broadcast::Sender::subscribe()` is synchronous and performs no I/O.
    ///
    /// # Errors
    /// - [`ObserveError::NotFound`] — no session with the given id exists.
    fn observe(&self, id: &SessionId) -> Result<Box<dyn ObserverHandle>, ObserveError>;

    /// Await the terminal state of a session.
    ///
    /// Blocks (asynchronously) until the session transitions to `Done`, `Cancelled`,
    /// or `Failed`.  Returns the terminal outcome.
    ///
    /// # Errors
    /// - [`AwaitError::NotFound`] — no session with the given id exists.
    /// - [`AwaitError::Joined`] — the background task failed unexpectedly.
    async fn await_terminal(&self, id: &SessionId) -> Result<TerminalOutcome, AwaitError>;
}
