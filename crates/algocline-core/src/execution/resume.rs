//! Resume-related types for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};

use super::cancel::{CancelInfo, FailureInfo};
use super::state::ExecutionResult;
use crate::TokenUsage;

/// Payload supplied to [`crate::execution::ExecutionService::resume`].
///
/// Must match the pause kind of the session being resumed: a `Single`-paused session
/// requires `Single`, a `Batch`-paused session requires `Batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "payload_kind", rename_all = "snake_case")]
pub enum ResumePayload {
    /// Resume a session that is waiting for a single LLM response.
    Single {
        /// The LLM response text.
        response: String,
        /// Token usage metadata for this response, if available.
        usage: Option<TokenUsage>,
        /// Identifies the query this response corresponds to.
        query_id: String,
    },
    /// Resume a session that is waiting for multiple LLM responses.
    Batch(Vec<QueryResponse>),
}

/// A single LLM response in a [`ResumePayload::Batch`].
///
/// Note: this type is distinct from the legacy `algocline_core::QueryResponse`
/// (re-exported via `engine_api.rs`) to avoid naming conflicts.  Access this type
/// via `algocline_core::execution::QueryResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Identifies the query this response is for.
    pub query_id: String,
    /// The LLM response text.
    pub response: String,
    /// Token usage metadata for this response, if available.
    pub usage: Option<TokenUsage>,
}

/// Outcome returned by [`crate::execution::ExecutionService::resume`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ResumeOutcome {
    /// The session accepted the response and is continuing execution.
    Continued,
    /// The session reached a terminal state after processing the response.
    Terminal(TerminalOutcome),
}

/// Terminal outcome for a session — carried by [`ResumeOutcome::Terminal`] and
/// by [`crate::execution::ExecutionService::await_terminal`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "terminal", rename_all = "snake_case")]
pub enum TerminalOutcome {
    /// The session completed successfully.
    Done(ExecutionResult),
    /// The session was cancelled cooperatively.
    Cancelled(CancelInfo),
    /// The session ended with an error.
    Failed(FailureInfo),
}
