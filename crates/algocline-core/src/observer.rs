use crate::{ExecutionSpec, LlmQuery, QueryId, TokenUsage};

/// Observer for execution state transitions.
///
/// Hooks cross-cutting concerns (stats, logging) without
/// polluting the Execution core.
pub trait ExecutionObserver: Send + Sync {
    fn on_started(&self, _spec: &ExecutionSpec) {}
    /// LLM request issued (transition to Paused).
    fn on_paused(&self, _queries: &[LlmQuery]) {}
    /// Partial response arrived (not yet complete).
    fn on_partial_feed(&self, _query_id: &QueryId, _remaining: usize) {}
    /// A single LLM response has been fed back.
    /// `usage` contains host-provided token counts when available.
    fn on_response_fed(&self, _query_id: &QueryId, _response: &str, _usage: Option<&TokenUsage>) {}
    /// All responses arrived, Lua resuming (transition to Running).
    fn on_resumed(&self) {}
    fn on_completed(&self, _result: &serde_json::Value) {}
    fn on_failed(&self, _error: &str) {}
    /// Host-initiated cancellation.
    fn on_cancelled(&self) {}
}
