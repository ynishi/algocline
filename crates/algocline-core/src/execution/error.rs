//! Dedicated error enums for each `ExecutionService` verb.
//!
//! All error types are **closed enums** (no `#[non_exhaustive]`): wire consumers
//! must handle every variant, and the adapter layer converts to a string before
//! crossing the MCP boundary (design-v1.md §2).
//!
//! All error types derive `serde::Serialize + serde::Deserialize` so they can be
//! embedded in structured JSON responses if needed by callers.

use serde::{Deserialize, Serialize};

use super::session_id::SessionId;
use super::state::ExecutionStateTag;
use crate::FeedError;

// ---------------------------------------------------------------------------
// SpawnError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::spawn`].
///
/// # Note on `#[from] EngineError`
/// In this subtask, the engine-level error is represented as a `String` wrapper.
/// Subtask 2 will replace `Engine(String)` with `Engine(#[from] SessionError)` once
/// the engine's `SessionError` type is available, using `#[from]` for automatic
/// conversion.
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum SpawnError {
    /// The engine failed to start the session.
    #[error("engine error: {0}")]
    Engine(String),
    /// The provided `SessionSpec` was invalid.
    #[error("invalid spec: {0}")]
    InvalidSpec(String),
}

// ---------------------------------------------------------------------------
// StateError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::state`].
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum StateError {
    /// No session with the given id exists in the registry.
    #[error("session not found: {0}")]
    NotFound(SessionId),
}

// ---------------------------------------------------------------------------
// ResumeError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::resume`].
///
/// # Note on `#[from]` and serde compatibility
/// `FeedError` (from `crates/algocline-core/src/state.rs`) does not derive
/// `serde::Serialize + serde::Deserialize`, so we cannot use `#[from] FeedError`
/// directly on a serde-enabled enum.  Instead, the `FeedError` variant stores the
/// display string, and a manual `From<FeedError>` impl converts it.
///
/// Subtask 2 may add `From<SessionError>` to this enum; if a second `#[from]`
/// conversion causes a trait impl conflict (K-79), the later conversion must use
/// `map_err` instead.
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum ResumeError {
    /// No session with the given id exists in the registry.
    #[error("session not found: {0}")]
    NotFound(SessionId),
    /// The session exists but is not in the `Paused` state.
    #[error("session is not paused; actual state: {actual_tag:?}")]
    NotPaused {
        /// The state the session is actually in.
        actual_tag: ExecutionStateTag,
    },
    /// The session is already cancelled; resuming is a no-op that returns this error
    /// so callers are explicitly informed.
    #[error("session is already cancelled")]
    AlreadyCancelled,
    /// The feed operation on the underlying session failed.
    ///
    /// The original `FeedError` message is preserved as a `String` because `FeedError`
    /// does not implement `serde::Serialize + serde::Deserialize`.
    #[error("feed error: {0}")]
    FeedError(String),
}

impl From<FeedError> for ResumeError {
    /// Converts a [`FeedError`] into a [`ResumeError::FeedError`] by formatting
    /// the error message.  This preserves the error chain for callers while
    /// remaining serde-compatible.
    fn from(e: FeedError) -> Self {
        Self::FeedError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// CancelError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::cancel`].
///
/// `cancel` is idempotent for sessions already in a terminal state: calling
/// `cancel` on a `Done`, `Failed`, or already-`Cancelled` session returns `Ok(())`.
/// The only error is when the session does not exist at all.
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum CancelError {
    /// No session with the given id exists in the registry.
    #[error("session not found: {0}")]
    NotFound(SessionId),
}

// ---------------------------------------------------------------------------
// ObserveError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::observe`].
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum ObserveError {
    /// No session with the given id exists in the registry.
    #[error("session not found: {0}")]
    NotFound(SessionId),
}

// ---------------------------------------------------------------------------
// AwaitError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ExecutionService::await_terminal`].
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum AwaitError {
    /// No session with the given id exists in the registry.
    #[error("session not found: {0}")]
    NotFound(SessionId),
    /// The background `JoinHandle` failed (e.g., the task panicked).  The string
    /// carries the error message; `JoinHandle::abort()` is never called (invariant 4).
    #[error("join error: {0}")]
    Joined(String),
}

// ---------------------------------------------------------------------------
// ObserverRecvError
// ---------------------------------------------------------------------------

/// Error returned by [`crate::execution::ObserverHandle::recv`] and
/// [`crate::execution::ObserverHandle::try_recv`].
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum ObserverRecvError {
    /// The observer fell behind and `n` events were dropped.  Subsequent calls
    /// will continue from the most recent available event (matches
    /// `tokio::sync::broadcast::error::RecvError::Lagged` semantics).
    #[error("observer lagged by {0} events")]
    Lagged(u64),
    /// The broadcast sender was dropped (session terminated).  No further events
    /// will be delivered.
    #[error("broadcast channel closed")]
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryId;

    #[test]
    fn resume_error_serde_roundtrip() {
        // Test standard variants
        let not_found = ResumeError::NotFound(SessionId::new("sess-1".into()));
        let json = serde_json::to_string(&not_found).expect("serialize NotFound");
        let _: ResumeError = serde_json::from_str(&json).expect("deserialize NotFound");

        let not_paused = ResumeError::NotPaused {
            actual_tag: ExecutionStateTag::Running,
        };
        let json = serde_json::to_string(&not_paused).expect("serialize NotPaused");
        let _: ResumeError = serde_json::from_str(&json).expect("deserialize NotPaused");

        let already_cancelled = ResumeError::AlreadyCancelled;
        let json = serde_json::to_string(&already_cancelled).expect("serialize AlreadyCancelled");
        let _: ResumeError = serde_json::from_str(&json).expect("deserialize AlreadyCancelled");

        // Test FeedError #[from] conversion path
        let feed_err = FeedError::UnknownQuery(QueryId::parse("q1"));
        let resume_err: ResumeError = feed_err.into();
        let json = serde_json::to_string(&resume_err).expect("serialize FeedError variant");
        let _: ResumeError = serde_json::from_str(&json).expect("deserialize FeedError variant");
    }

    #[test]
    fn observer_recv_error_serde_roundtrip() {
        let lagged = ObserverRecvError::Lagged(42);
        let json = serde_json::to_string(&lagged).expect("serialize Lagged");
        let _: ObserverRecvError = serde_json::from_str(&json).expect("deserialize Lagged");

        let closed = ObserverRecvError::Closed;
        let json = serde_json::to_string(&closed).expect("serialize Closed");
        let _: ObserverRecvError = serde_json::from_str(&json).expect("deserialize Closed");
    }

    #[test]
    fn all_error_enums_display() {
        // Verify thiserror display works for all enums
        let e = SpawnError::InvalidSpec("bad field".into());
        assert!(e.to_string().contains("bad field"));

        let e = StateError::NotFound(SessionId::new("s1".into()));
        assert!(e.to_string().contains("s1"));

        let e = CancelError::NotFound(SessionId::new("s2".into()));
        assert!(e.to_string().contains("s2"));

        let e = ObserveError::NotFound(SessionId::new("s3".into()));
        assert!(e.to_string().contains("s3"));

        let e = AwaitError::Joined("panic message".into());
        assert!(e.to_string().contains("panic message"));
    }
}
