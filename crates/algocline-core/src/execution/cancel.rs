//! Cancellation and failure types for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};

use super::state::ExecutionState;

/// Reason supplied to [`crate::execution::ExecutionService::cancel`].
///
/// `requested_at` is a Unix timestamp in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelReason {
    /// Categorized cause of the cancellation.
    pub code: CancelCode,
    /// Optional human-readable detail string.
    pub detail: Option<String>,
    /// Unix timestamp (milliseconds) when the cancellation was requested.
    pub requested_at: i64,
}

/// Categorized cause of a cancellation.
///
/// This is a **closed** enum — `#[non_exhaustive]` is intentionally absent so that
/// exhaustive `match` is enforced on consumers (design-v1.md §2).
///
/// No forceful kill variants (e.g., `Aborted`, `Killed`, `ForcedExit`) are included;
/// all cancellation in this design is cooperative (crux: Cooperative cancellation
/// 4-checkpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelCode {
    /// Requested explicitly by a human or calling client.
    User,
    /// A deadline or time budget was exceeded.
    Timeout,
    /// The parent context (e.g., the service instance) was dropped.
    ParentDropped,
    /// The process is shutting down gracefully.
    SystemShutdown,
    /// Internal engine policy triggered cancellation.
    Internal,
}

/// Information recorded when a session transitions to `Cancelled`.
///
/// `state_before` is boxed to avoid a recursive type without indirection.
/// `observed_at` is a Unix timestamp in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelInfo {
    /// The reason that triggered cancellation.
    pub reason: CancelReason,
    /// Unix timestamp (milliseconds) when the cancellation was observed by the engine.
    pub observed_at: i64,
    /// A snapshot of the [`ExecutionState`] immediately before cancellation was applied.
    #[serde(rename = "state_before")]
    pub state_before: Box<ExecutionState>,
}

/// Information recorded when a session transitions to `Failed`.
///
/// `occurred_at` is a Unix timestamp in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    /// Human-readable error message.
    pub message: String,
    /// Categorized kind of failure.
    pub kind: FailureKind,
    /// Unix timestamp (milliseconds) when the failure was recorded.
    pub occurred_at: i64,
}

/// Categorized kind of failure.
///
/// Closed enum — `#[non_exhaustive]` intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// A Lua runtime error occurred.
    LuaError,
    /// An internal engine error occurred.
    EngineError,
    /// A timeout was exceeded during execution.
    Timeout,
    /// Any other failure not covered by the above variants.
    Other,
}
