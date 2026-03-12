//! Channel-based bridge between Lua coroutines and async MCP Sampling.
//!
//! Lua calls `alc.llm(prompt, opts)` or `alc.llm_batch(items)` →
//! coroutine yields (non-blocking) → request is sent through a tokio mpsc channel →
//! async handler processes the queries → responses flow back through
//! tokio oneshot channels (one per query).

use algocline_core::QueryId;

/// A batch of LLM queries from a single Lua call.
///
/// For `alc.llm()`: contains exactly one QueryRequest.
/// For `alc.llm_batch()`: contains N QueryRequests.
pub struct LlmRequest {
    pub queries: Vec<QueryRequest>,
}

/// A single query within an LlmRequest batch.
pub struct QueryRequest {
    pub id: QueryId,
    pub prompt: String,
    pub system: Option<String>,
    pub max_tokens: u32,
    /// Channel to send the response back to the yielded Lua coroutine.
    pub resp_tx: tokio::sync::oneshot::Sender<Result<String, String>>,
}
