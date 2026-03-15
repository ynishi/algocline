use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::query::{LlmQuery, QueryId};

#[derive(Debug, thiserror::Error)]
#[error("invalid state transition: expected {expected}, got {actual}")]
pub struct TransitionError {
    pub expected: &'static str,
    pub actual: &'static str,
}

#[derive(Debug, thiserror::Error)]
pub enum FeedError {
    #[error("unknown query_id: {0}")]
    UnknownQuery(QueryId),
    #[error("already responded to query_id: {0}")]
    AlreadyResponded(QueryId),
    #[error(transparent)]
    InvalidState(#[from] TransitionError),
}

/// Join barrier that collects N LLM responses.
///
/// Responses can be fed in any order and concurrency.
/// Becomes complete when all queries have been responded to.
#[derive(Debug, Serialize, Deserialize)]
pub struct PendingQueries {
    /// Issued queries (insertion order preserved via IndexMap).
    queries: IndexMap<QueryId, LlmQuery>,
    responses: HashMap<QueryId, String>,
}

impl PendingQueries {
    pub fn new(queries: Vec<LlmQuery>) -> Self {
        let map = queries
            .into_iter()
            .map(|q| (q.id.clone(), q))
            .collect::<IndexMap<_, _>>();
        Self {
            queries: map,
            responses: HashMap::new(),
        }
    }

    /// Feed one response. Returns `true` if all queries are now complete.
    pub fn feed(&mut self, id: &QueryId, response: String) -> Result<bool, FeedError> {
        if !self.queries.contains_key(id) {
            return Err(FeedError::UnknownQuery(id.clone()));
        }
        if self.responses.contains_key(id) {
            return Err(FeedError::AlreadyResponded(id.clone()));
        }
        self.responses.insert(id.clone(), response);
        Ok(self.is_complete())
    }

    pub fn pending_queries(&self) -> Vec<&LlmQuery> {
        self.queries
            .values()
            .filter(|q| !self.responses.contains_key(&q.id))
            .collect()
    }

    pub fn remaining(&self) -> usize {
        self.queries.len() - self.responses.len()
    }

    pub fn is_complete(&self) -> bool {
        self.responses.len() == self.queries.len()
    }

    /// Consume and return responses in query insertion order.
    /// Corresponds to the Paused → Running transition.
    pub fn into_ordered_responses(self) -> Vec<String> {
        self.queries
            .keys()
            .map(|id| {
                // is_complete() guarantees queries and responses share the same key set,
                // but fall back to empty string if called without checking is_complete()
                self.responses.get(id).cloned().unwrap_or_default()
            })
            .collect()
    }
}

pub enum ExecutionState {
    Running,
    /// Awaiting 1..N LLM responses.
    Paused(PendingQueries),
    Completed {
        result: serde_json::Value,
    },
    Failed {
        error: String,
    },
    /// Explicit cancellation by the host.
    Cancelled,
}

impl ExecutionState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }

    /// Number of pending queries. Returns 0 for non-Paused states.
    pub fn remaining(&self) -> usize {
        match self {
            Self::Paused(pending) => pending.remaining(),
            _ => 0,
        }
    }

    /// Returns the state name (for error messages).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Paused(_) => "Paused",
            Self::Completed { .. } => "Completed",
            Self::Failed { .. } => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    /// Feed a response. Only valid in Paused state.
    /// Returns `Ok(true)` when all queries are complete, `Ok(false)` otherwise.
    pub fn feed(&mut self, id: &QueryId, response: String) -> Result<bool, FeedError> {
        match self {
            Self::Paused(pending) => pending.feed(id, response),
            other => Err(TransitionError {
                expected: "Paused",
                actual: other.name(),
            }
            .into()),
        }
    }

    /// Extract responses from a complete Paused state.
    /// Transitions self to Running (preparing for Lua resumption).
    pub fn take_responses(&mut self) -> Result<Vec<String>, TransitionError> {
        match std::mem::replace(self, Self::Running) {
            Self::Paused(pending) if pending.is_complete() => Ok(pending.into_ordered_responses()),
            prev => {
                let actual = prev.name();
                *self = prev;
                Err(TransitionError {
                    expected: "Paused(complete)",
                    actual,
                })
            }
        }
    }

    /// Running → Completed.
    pub fn complete(&mut self, result: serde_json::Value) -> Result<(), TransitionError> {
        match self {
            Self::Running => {
                *self = Self::Completed { result };
                Ok(())
            }
            other => Err(TransitionError {
                expected: "Running",
                actual: other.name(),
            }),
        }
    }

    /// Running → Failed.
    pub fn fail(&mut self, error: String) -> Result<(), TransitionError> {
        match self {
            Self::Running => {
                *self = Self::Failed { error };
                Ok(())
            }
            other => Err(TransitionError {
                expected: "Running",
                actual: other.name(),
            }),
        }
    }

    /// Running → Paused (triggered by alc.llm() / alc.llm_batch()).
    pub fn pause(&mut self, queries: Vec<LlmQuery>) -> Result<(), TransitionError> {
        match self {
            Self::Running => {
                *self = Self::Paused(PendingQueries::new(queries));
                Ok(())
            }
            other => Err(TransitionError {
                expected: "Running",
                actual: other.name(),
            }),
        }
    }

    /// Running | Paused → Cancelled (explicit host cancellation).
    pub fn cancel(&mut self) -> Result<(), TransitionError> {
        match self {
            Self::Running | Self::Paused(_) => {
                *self = Self::Cancelled;
                Ok(())
            }
            other => Err(TransitionError {
                expected: "Running or Paused",
                actual: other.name(),
            }),
        }
    }
}

/// Return type of Session.resume(). Never returns Running.
pub enum ResumeOutcome {
    /// Lua resumed and paused again at alc.llm().
    Paused {
        queries: Vec<LlmQuery>,
    },
    Completed {
        result: serde_json::Value,
    },
    Failed {
        error: String,
    },
    /// Cancelled during resume.
    Cancelled,
}

/// Terminal execution state. Only Completed, Failed, or Cancelled.
#[derive(Debug)]
pub enum TerminalState {
    Completed { result: serde_json::Value },
    Failed { error: String },
    Cancelled,
}

impl TryFrom<ExecutionState> for TerminalState {
    type Error = TransitionError;

    fn try_from(state: ExecutionState) -> Result<Self, TransitionError> {
        match state {
            ExecutionState::Completed { result } => Ok(Self::Completed { result }),
            ExecutionState::Failed { error } => Ok(Self::Failed { error }),
            ExecutionState::Cancelled => Ok(Self::Cancelled),
            other => Err(TransitionError {
                expected: "Completed, Failed, or Cancelled",
                actual: other.name(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{LlmQuery, QueryId};
    use serde_json::json;

    fn make_query(index: usize) -> LlmQuery {
        LlmQuery {
            id: QueryId::batch(index),
            prompt: format!("prompt-{index}"),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }
    }

    // ─── PendingQueries tests ───

    #[test]
    fn pending_queries_single_feed() {
        let mut pq = PendingQueries::new(vec![make_query(0)]);
        assert_eq!(pq.remaining(), 1);
        assert!(!pq.is_complete());

        let complete = pq.feed(&QueryId::batch(0), "resp".into()).unwrap();
        assert!(complete);
        assert_eq!(pq.remaining(), 0);
    }

    #[test]
    fn pending_queries_multi_feed_ordering() {
        let mut pq = PendingQueries::new(vec![make_query(0), make_query(1), make_query(2)]);

        // feed in reverse order
        assert!(!pq.feed(&QueryId::batch(2), "resp-2".into()).unwrap());
        assert!(!pq.feed(&QueryId::batch(0), "resp-0".into()).unwrap());
        assert!(pq.feed(&QueryId::batch(1), "resp-1".into()).unwrap());

        // into_ordered_responses returns in insertion order
        let responses = pq.into_ordered_responses();
        assert_eq!(responses, vec!["resp-0", "resp-1", "resp-2"]);
    }

    #[test]
    fn pending_queries_unknown_query_error() {
        let mut pq = PendingQueries::new(vec![make_query(0)]);
        let err = pq.feed(&QueryId::batch(99), "resp".into()).unwrap_err();
        assert!(matches!(err, FeedError::UnknownQuery(_)));
    }

    #[test]
    fn pending_queries_double_feed_error() {
        let mut pq = PendingQueries::new(vec![make_query(0)]);
        pq.feed(&QueryId::batch(0), "resp".into()).unwrap();
        let err = pq.feed(&QueryId::batch(0), "resp2".into()).unwrap_err();
        assert!(matches!(err, FeedError::AlreadyResponded(_)));
    }

    #[test]
    fn pending_queries_pending_list() {
        let mut pq = PendingQueries::new(vec![make_query(0), make_query(1)]);
        assert_eq!(pq.pending_queries().len(), 2);

        pq.feed(&QueryId::batch(0), "resp".into()).unwrap();
        let pending = pq.pending_queries();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, QueryId::batch(1));
    }

    #[test]
    fn pending_queries_roundtrip_json() {
        let mut pq = PendingQueries::new(vec![make_query(0), make_query(1)]);
        pq.feed(&QueryId::batch(0), "resp-0".into()).unwrap();

        let json = serde_json::to_value(&pq).unwrap();
        let restored: PendingQueries = serde_json::from_value(json).unwrap();
        assert_eq!(restored.remaining(), 1);
        assert_eq!(restored.queries.len(), 2);
    }

    // ─── ExecutionState transition tests ───

    #[test]
    fn running_to_paused() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0)]).unwrap();
        assert_eq!(state.name(), "Paused");
    }

    #[test]
    fn paused_feed_and_take() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0), make_query(1)]).unwrap();

        assert!(!state.feed(&QueryId::batch(0), "r0".into()).unwrap());
        assert!(state.feed(&QueryId::batch(1), "r1".into()).unwrap());

        let responses = state.take_responses().unwrap();
        assert_eq!(responses, vec!["r0", "r1"]);
        assert_eq!(state.name(), "Running");
    }

    #[test]
    fn take_responses_incomplete_fails() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0), make_query(1)]).unwrap();
        state.feed(&QueryId::batch(0), "r0".into()).unwrap();

        let err = state.take_responses().unwrap_err();
        assert_eq!(err.actual, "Paused");
        // state should remain Paused
        assert_eq!(state.name(), "Paused");
    }

    #[test]
    fn running_to_completed() {
        let mut state = ExecutionState::Running;
        state.complete(json!({"answer": 42})).unwrap();
        assert!(state.is_terminal());
        assert_eq!(state.name(), "Completed");
    }

    #[test]
    fn running_to_failed() {
        let mut state = ExecutionState::Running;
        state.fail("boom".into()).unwrap();
        assert!(state.is_terminal());
        assert_eq!(state.name(), "Failed");
    }

    #[test]
    fn cancel_from_running() {
        let mut state = ExecutionState::Running;
        state.cancel().unwrap();
        assert!(state.is_terminal());
        assert_eq!(state.name(), "Cancelled");
    }

    #[test]
    fn cancel_from_paused() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0)]).unwrap();
        state.cancel().unwrap();
        assert_eq!(state.name(), "Cancelled");
    }

    // ─── remaining() tests ───

    #[test]
    fn remaining_running_is_zero() {
        let state = ExecutionState::Running;
        assert_eq!(state.remaining(), 0);
    }

    #[test]
    fn remaining_tracks_feeds() {
        let mut state = ExecutionState::Running;
        state
            .pause(vec![make_query(0), make_query(1), make_query(2)])
            .unwrap();
        assert_eq!(state.remaining(), 3);

        state.feed(&QueryId::batch(0), "r".into()).unwrap();
        assert_eq!(state.remaining(), 2);

        state.feed(&QueryId::batch(1), "r".into()).unwrap();
        assert_eq!(state.remaining(), 1);
    }

    #[test]
    fn remaining_terminal_is_zero() {
        let state = ExecutionState::Completed {
            result: json!(null),
        };
        assert_eq!(state.remaining(), 0);
    }

    // ─── Invalid transition tests ───

    #[test]
    fn feed_on_running_fails() {
        let mut state = ExecutionState::Running;
        let err = state.feed(&QueryId::single(), "r".into()).unwrap_err();
        assert!(matches!(err, FeedError::InvalidState(_)));
    }

    #[test]
    fn pause_on_paused_fails() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0)]).unwrap();
        let err = state.pause(vec![make_query(1)]).unwrap_err();
        assert_eq!(err.expected, "Running");
    }

    #[test]
    fn complete_on_paused_fails() {
        let mut state = ExecutionState::Running;
        state.pause(vec![make_query(0)]).unwrap();
        let err = state.complete(json!(null)).unwrap_err();
        assert_eq!(err.expected, "Running");
    }

    #[test]
    fn cancel_on_completed_fails() {
        let mut state = ExecutionState::Running;
        state.complete(json!(null)).unwrap();
        let err = state.cancel().unwrap_err();
        assert_eq!(err.expected, "Running or Paused");
    }

    #[test]
    fn cancel_on_failed_fails() {
        let mut state = ExecutionState::Running;
        state.fail("e".into()).unwrap();
        let err = state.cancel().unwrap_err();
        assert_eq!(err.expected, "Running or Paused");
    }

    #[test]
    fn terminal_state_rejects_non_terminal() {
        let state = ExecutionState::Running;
        let err = TerminalState::try_from(state).unwrap_err();
        assert_eq!(err.actual, "Running");
    }

    #[test]
    fn terminal_state_from_completed() {
        let state = ExecutionState::Completed { result: json!(42) };
        let terminal = TerminalState::try_from(state).unwrap();
        assert!(matches!(terminal, TerminalState::Completed { .. }));
    }

    #[test]
    fn terminal_state_from_cancelled() {
        let state = ExecutionState::Cancelled;
        let terminal = TerminalState::try_from(state).unwrap();
        assert!(matches!(terminal, TerminalState::Cancelled));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::query::{LlmQuery, QueryId};
    use proptest::prelude::*;

    fn make_query(index: usize) -> LlmQuery {
        LlmQuery {
            id: QueryId::batch(index),
            prompt: format!("prompt-{index}"),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }
    }

    proptest! {
        /// into_ordered_responses returns insertion order regardless of feed order.
        #[test]
        fn feed_order_independent(size in 1usize..8) {
            let queries: Vec<LlmQuery> = (0..size).map(make_query).collect();
            let mut pq = PendingQueries::new(queries);

            // feed in reverse order
            for i in (0..size).rev() {
                let _ = pq.feed(&QueryId::batch(i), format!("resp-{i}"));
            }

            let responses = pq.into_ordered_responses();
            // must return in insertion order (0, 1, 2, ...)
            for (i, resp) in responses.iter().enumerate() {
                prop_assert_eq!(resp, &format!("resp-{i}"));
            }
        }

        /// Feeding the same query twice returns AlreadyResponded error.
        #[test]
        fn double_feed_always_errors(size in 1usize..8, target in 0usize..8) {
            let target = target % size; // clamp to valid range
            let queries: Vec<LlmQuery> = (0..size).map(make_query).collect();
            let mut pq = PendingQueries::new(queries);

            pq.feed(&QueryId::batch(target), "first".into()).unwrap();
            let err = pq.feed(&QueryId::batch(target), "second".into()).unwrap_err();
            prop_assert!(matches!(err, FeedError::AlreadyResponded(_)));
        }

        /// Feeding a non-existent query_id returns UnknownQuery error.
        #[test]
        fn unknown_query_always_errors(size in 1usize..8, bad_id in 100usize..200) {
            let queries: Vec<LlmQuery> = (0..size).map(make_query).collect();
            let mut pq = PendingQueries::new(queries);

            let err = pq.feed(&QueryId::batch(bad_id), "resp".into()).unwrap_err();
            prop_assert!(matches!(err, FeedError::UnknownQuery(_)));
        }

        /// remaining() decreases by 1 with each feed.
        #[test]
        fn remaining_decreases_monotonically(size in 1usize..10) {
            let queries: Vec<LlmQuery> = (0..size).map(make_query).collect();
            let mut pq = PendingQueries::new(queries);

            for i in 0..size {
                prop_assert_eq!(pq.remaining(), size - i);
                let _ = pq.feed(&QueryId::batch(i), format!("r-{i}"));
            }
            prop_assert_eq!(pq.remaining(), 0);
            prop_assert!(pq.is_complete());
        }
    }
}
