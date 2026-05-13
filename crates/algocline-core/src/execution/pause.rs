//! Pause-related types for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};

/// Information about a paused execution session.
///
/// Carried by [`crate::execution::ExecutionState::Paused`] and emitted in
/// [`crate::execution::ProgressEvent::PauseRequested`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseInfo {
    /// Whether the pause expects a single response or a batch.
    pub kind: PauseKind,
    /// The pending LLM prompts that need responses.
    pub prompts: Vec<PausePrompt>,
    /// Unix timestamp (milliseconds) when the pause was recorded.
    pub paused_at: i64,
}

/// Discriminant indicating whether a pause expects a single or batch response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseKind {
    /// Exactly one LLM response is expected.
    Single,
    /// Multiple LLM responses are expected (batch mode).
    Batch,
}

/// A single pending LLM prompt within a paused session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PausePrompt {
    /// Stable identifier for this query within the session.
    pub query_id: String,
    /// The prompt text to send to the LLM.
    pub prompt: String,
}
