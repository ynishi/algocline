//! Session-based Lua execution with pause/resume on alc.llm() calls.
//!
//! Runtime layer: ties Domain (ExecutionState) and Metrics (ExecutionMetrics)
//! together with channel-based Lua pause/resume machinery.

use std::collections::HashMap;
use std::sync::Arc;

use algocline_core::{
    ExecutionMetrics, ExecutionObserver, ExecutionState, LlmQuery, MetricsObserver, QueryId,
    TerminalState,
};
use mlua_isle::{AsyncIsleDriver, AsyncTask};
use serde_json::json;
use tokio::sync::Mutex;

use crate::llm_bridge::LlmRequest;

// ─── Error types (Runtime layer) ─────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session '{0}' not found")]
    NotFound(String),
    #[error(transparent)]
    Feed(#[from] algocline_core::FeedError),
    #[error("invalid transition: {0}")]
    InvalidTransition(String),
}

// ─── Result types (Runtime layer) ────────────────────────────

/// Session completion data: terminal state + metrics.
pub struct ExecutionResult {
    pub state: TerminalState,
    pub metrics: ExecutionMetrics,
}

/// Result of a session interaction (start or feed).
pub enum FeedResult {
    /// Partial feed accepted, still waiting for more responses.
    Accepted { remaining: usize },
    /// All queries answered, Lua re-paused with new queries.
    Paused { queries: Vec<LlmQuery> },
    /// Execution completed (success, failure, or cancellation).
    Finished(ExecutionResult),
}

impl FeedResult {
    /// Convert to JSON for MCP tool response.
    pub fn to_json(&self, session_id: &str) -> serde_json::Value {
        match self {
            Self::Accepted { remaining } => json!({
                "status": "accepted",
                "remaining": remaining,
            }),
            Self::Paused { queries } => {
                if queries.len() == 1 {
                    let q = &queries[0];
                    let mut obj = json!({
                        "status": "needs_response",
                        "session_id": session_id,
                        "query_id": q.id.as_str(),
                        "prompt": q.prompt,
                        "system": q.system,
                        "max_tokens": q.max_tokens,
                    });
                    if q.grounded {
                        obj["grounded"] = json!(true);
                    }
                    if q.underspecified {
                        obj["underspecified"] = json!(true);
                    }
                    obj
                } else {
                    let qs: Vec<_> = queries
                        .iter()
                        .map(|q| {
                            let mut obj = json!({
                                "id": q.id.as_str(),
                                "prompt": q.prompt,
                                "system": q.system,
                                "max_tokens": q.max_tokens,
                            });
                            if q.grounded {
                                obj["grounded"] = json!(true);
                            }
                            if q.underspecified {
                                obj["underspecified"] = json!(true);
                            }
                            obj
                        })
                        .collect();
                    json!({
                        "status": "needs_response",
                        "session_id": session_id,
                        "queries": qs,
                    })
                }
            }
            Self::Finished(result) => match &result.state {
                TerminalState::Completed { result: val } => json!({
                    "status": "completed",
                    "result": val,
                    "stats": result.metrics.to_json(),
                }),
                TerminalState::Failed { error } => json!({
                    "status": "error",
                    "error": error,
                }),
                TerminalState::Cancelled => json!({
                    "status": "cancelled",
                    "stats": result.metrics.to_json(),
                }),
            },
        }
    }
}

// ─── Session ─────────────────────────────────────────────────

/// A Lua execution session with domain state tracking.
///
/// Each session owns a dedicated Lua VM via `_vm_driver`. The VM's OS thread
/// stays alive as long as the driver is held, and exits cleanly when the
/// session is dropped (channel closes → Lua thread drains and exits).
pub struct Session {
    state: ExecutionState,
    metrics: ExecutionMetrics,
    observer: MetricsObserver,
    llm_rx: tokio::sync::mpsc::Receiver<LlmRequest>,
    exec_task: AsyncTask,
    /// QueryId → resp_tx. Populated on Paused, cleared on resume.
    resp_txs: HashMap<QueryId, tokio::sync::oneshot::Sender<Result<String, String>>>,
    /// Per-session VM lifecycle driver. Keeps the Lua thread alive.
    /// Dropped when the session completes or is abandoned.
    _vm_driver: AsyncIsleDriver,
}

impl Session {
    pub fn new(
        llm_rx: tokio::sync::mpsc::Receiver<LlmRequest>,
        exec_task: AsyncTask,
        metrics: ExecutionMetrics,
        vm_driver: AsyncIsleDriver,
    ) -> Self {
        let observer = metrics.create_observer();
        Self {
            state: ExecutionState::Running,
            metrics,
            observer,
            llm_rx,
            exec_task,
            resp_txs: HashMap::new(),
            _vm_driver: vm_driver,
        }
    }

    /// Wait for the next event from Lua execution.
    ///
    /// Called after initial start or after feeding all responses.
    /// State must be Running when called.
    async fn wait_event(&mut self) -> Result<FeedResult, SessionError> {
        tokio::select! {
            result = &mut self.exec_task => {
                match result {
                    Ok(json_str) => match serde_json::from_str::<serde_json::Value>(&json_str) {
                        Ok(v) => {
                            self.state.complete(v.clone()).map_err(|e| {
                                SessionError::InvalidTransition(e.to_string())
                            })?;
                            self.observer.on_completed(&v);
                            Ok(FeedResult::Finished(ExecutionResult {
                                state: TerminalState::Completed { result: v },
                                metrics: self.take_metrics(),
                            }))
                        }
                        Err(e) => self.fail_with(format!("JSON parse: {e}")),
                    },
                    Err(e) => self.fail_with(e.to_string()),
                }
            }
            Some(req) = self.llm_rx.recv() => {
                let queries: Vec<LlmQuery> = req.queries.iter().map(|qr| LlmQuery {
                    id: qr.id.clone(),
                    prompt: qr.prompt.clone(),
                    system: qr.system.clone(),
                    max_tokens: qr.max_tokens,
                    grounded: qr.grounded,
                    underspecified: qr.underspecified,
                }).collect();

                for qr in req.queries {
                    self.resp_txs.insert(qr.id, qr.resp_tx);
                }

                self.state.pause(queries.clone()).map_err(|e| {
                    SessionError::InvalidTransition(e.to_string())
                })?;
                self.observer.on_paused(&queries);
                Ok(FeedResult::Paused { queries })
            }
        }
    }

    /// Feed one response by query_id.
    ///
    /// Returns Ok(true) if all queries are now complete, Ok(false) if still waiting.
    fn feed_one(&mut self, query_id: &QueryId, response: String) -> Result<bool, SessionError> {
        // Track response before ownership transfer.
        self.observer.on_response_fed(query_id, &response);

        // Runtime: send response to Lua thread (unblocks resp_rx.recv())
        if let Some(tx) = self.resp_txs.remove(query_id) {
            let _ = tx.send(Ok(response.clone()));
        }

        // Domain: record in state machine
        let complete = self
            .state
            .feed(query_id, response)
            .map_err(SessionError::Feed)?;

        if complete {
            // Domain: transition Paused(complete) → Running
            self.state
                .take_responses()
                .map_err(|e| SessionError::InvalidTransition(e.to_string()))?;
            self.observer.on_resumed();
        } else {
            self.observer
                .on_partial_feed(query_id, self.state.remaining());
        }

        Ok(complete)
    }

    fn fail_with(&mut self, msg: String) -> Result<FeedResult, SessionError> {
        self.state
            .fail(msg.clone())
            .map_err(|e| SessionError::InvalidTransition(e.to_string()))?;
        self.observer.on_failed(&msg);
        Ok(FeedResult::Finished(ExecutionResult {
            state: TerminalState::Failed { error: msg },
            metrics: self.take_metrics(),
        }))
    }

    fn take_metrics(&mut self) -> ExecutionMetrics {
        std::mem::take(&mut self.metrics)
    }

    /// Lightweight snapshot for external observation (alc_status).
    ///
    /// Returns session state label and running metrics without consuming
    /// or modifying the session.
    pub fn snapshot(&self) -> serde_json::Value {
        let state_label = match &self.state {
            ExecutionState::Running => "running",
            ExecutionState::Paused(_) => "paused",
            _ => "terminal",
        };

        let mut json = serde_json::json!({
            "state": state_label,
        });

        let metrics = self.metrics.snapshot();
        if !metrics.is_null() {
            json["metrics"] = metrics;
        }

        // Include pending query count when paused
        if let ExecutionState::Paused(_) = &self.state {
            json["pending_queries"] = self.state.remaining().into();
        }

        json
    }
}

// ─── Registry ────────────────────────────────────────────────

/// Manages active sessions.
///
/// # Locking design (lock **C**)
///
/// Uses `tokio::sync::Mutex` because `feed_response` holds the lock
/// while calling `Session::feed_one()` (which itself acquires the
/// per-session `std::sync::Mutex<SessionStatus>`, lock **A**). The lock
/// ordering invariant is always **C → A** — no code path acquires A
/// then C, so deadlock is structurally impossible.
///
/// `tokio::sync::Mutex` is chosen here (rather than `std::sync::Mutex`)
/// because `feed_response` must take the session out of the map for
/// the async `wait_event()` call. The two-phase pattern (lock → remove
/// → unlock → await → lock → reinsert) requires an async-aware mutex
/// to avoid holding the lock across the `wait_event().await`.
///
/// ## Contention
///
/// `list_snapshots()` (from `alc_status`) holds lock C while iterating
/// all sessions. During this time, `feed_response` for any session is
/// blocked. Given that snapshot iteration is O(n) with n = active
/// sessions (typically 1–3) and each snapshot takes microseconds, this
/// is acceptable. If session count grows significantly, consider
/// switching to a concurrent map or per-session locks.
///
/// ## Interaction with lock A
///
/// `Session::snapshot()` (called under lock C in `list_snapshots`)
/// acquires lock A via `ExecutionMetrics::snapshot()`. This is safe:
/// - Lock order: C → A (consistent with `feed_response`)
/// - Lock A hold time: microseconds (JSON field reads)
/// - Lock A is per-session (no cross-session contention)
pub struct SessionRegistry {
    sessions: Arc<Mutex<HashMap<String, Session>>>,
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start execution and wait for first event (pause or completion).
    pub async fn start_execution(
        &self,
        mut session: Session,
    ) -> Result<(String, FeedResult), SessionError> {
        let session_id = gen_session_id();
        let result = session.wait_event().await?;

        if matches!(result, FeedResult::Paused { .. }) {
            self.sessions
                .lock()
                .await
                .insert(session_id.clone(), session);
        }

        Ok((session_id, result))
    }

    /// Feed one response to a paused session by query_id.
    ///
    /// If this completes all pending queries, the session resumes and
    /// returns the next event (Paused or Finished).
    /// If queries remain, returns Accepted { remaining }.
    pub async fn feed_response(
        &self,
        session_id: &str,
        query_id: &QueryId,
        response: String,
    ) -> Result<FeedResult, SessionError> {
        // 1. Feed under lock
        let complete = {
            let mut map = self.sessions.lock().await;
            let session = map
                .get_mut(session_id)
                .ok_or_else(|| SessionError::NotFound(session_id.into()))?;

            let complete = session.feed_one(query_id, response)?;

            if !complete {
                return Ok(FeedResult::Accepted {
                    remaining: session.state.remaining(),
                });
            }

            complete
        };

        // 2. All complete → take session out for async resume
        debug_assert!(complete);
        let mut session = {
            let mut map = self.sessions.lock().await;
            map.remove(session_id)
                .ok_or_else(|| SessionError::NotFound(session_id.into()))?
        };

        let result = session.wait_event().await?;

        if matches!(result, FeedResult::Paused { .. }) {
            self.sessions
                .lock()
                .await
                .insert(session_id.into(), session);
        }

        Ok(result)
    }

    /// Resolve the sole pending query ID for a session.
    ///
    /// When `alc_continue` is called without an explicit `query_id`, this
    /// method checks if exactly one query is pending and returns its ID.
    /// Returns an error if zero or multiple queries are pending.
    pub async fn resolve_sole_pending_id(&self, session_id: &str) -> Result<QueryId, SessionError> {
        let map = self.sessions.lock().await;
        let session = map
            .get(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.into()))?;
        let keys: Vec<QueryId> = session.resp_txs.keys().cloned().collect();
        match keys.len() {
            0 => Err(SessionError::InvalidTransition("no pending queries".into())),
            1 => keys
                .into_iter()
                .next()
                .ok_or_else(|| SessionError::InvalidTransition("unexpected empty keys".into())),
            n => Err(SessionError::InvalidTransition(format!(
                "{n} queries pending; specify query_id explicitly"
            ))),
        }
    }

    /// Snapshot all active sessions for external observation (alc_status).
    ///
    /// Returns a map of session_id → snapshot JSON. Only includes sessions
    /// currently held in the registry (i.e. paused, awaiting responses).
    /// Sessions that have completed are already removed from the registry.
    pub async fn list_snapshots(&self) -> HashMap<String, serde_json::Value> {
        let map = self.sessions.lock().await;
        map.iter()
            .map(|(id, session)| (id.clone(), session.snapshot()))
            .collect()
    }
}

/// Generate a non-deterministic session ID.
///
/// MCP spec requires "secure, non-deterministic session IDs" to prevent
/// session hijacking. Uses timestamp + random bytes for uniqueness and
/// unpredictability.
fn gen_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // 8 random bytes → 16 hex chars of entropy
    let random: u64 = {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let s = RandomState::new();
        let mut h = s.build_hasher();
        h.write_u128(ts);
        h.finish()
    };
    format!("s-{ts:x}-{random:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::{ExecutionMetrics, LlmQuery, QueryId};
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

    // ─── FeedResult::to_json tests ───

    #[test]
    fn to_json_accepted() {
        let result = FeedResult::Accepted { remaining: 3 };
        let json = result.to_json("s-123");
        assert_eq!(json["status"], "accepted");
        assert_eq!(json["remaining"], 3);
    }

    #[test]
    fn to_json_paused_single_query() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "What is 2+2?".into(),
            system: Some("You are a calculator.".into()),
            max_tokens: 50,
            grounded: false,
            underspecified: false,
        };
        let result = FeedResult::Paused {
            queries: vec![query],
        };
        let json = result.to_json("s-abc");

        assert_eq!(json["status"], "needs_response");
        assert_eq!(json["session_id"], "s-abc");
        assert_eq!(json["prompt"], "What is 2+2?");
        assert_eq!(json["system"], "You are a calculator.");
        assert_eq!(json["max_tokens"], 50);
        // single query mode: no "queries" array
        assert!(json.get("queries").is_none());
        // grounded=false must be absent
        assert!(
            json.get("grounded").is_none(),
            "grounded key must be absent when false"
        );
        // underspecified=false must be absent
        assert!(
            json.get("underspecified").is_none(),
            "underspecified key must be absent when false"
        );
    }

    #[test]
    fn to_json_paused_single_query_grounded() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "verify this claim".into(),
            system: None,
            max_tokens: 200,
            grounded: true,
            underspecified: false,
        };
        let result = FeedResult::Paused {
            queries: vec![query],
        };
        let json = result.to_json("s-grounded");

        assert_eq!(json["status"], "needs_response");
        assert_eq!(
            json["grounded"], true,
            "grounded must appear in single-query MCP JSON"
        );
    }

    #[test]
    fn to_json_paused_single_query_underspecified() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "what output format do you need?".into(),
            system: None,
            max_tokens: 200,
            grounded: false,
            underspecified: true,
        };
        let result = FeedResult::Paused {
            queries: vec![query],
        };
        let json = result.to_json("s-underspec");

        assert_eq!(json["status"], "needs_response");
        assert_eq!(
            json["underspecified"], true,
            "underspecified must appear in single-query MCP JSON"
        );
        assert!(
            json.get("grounded").is_none(),
            "grounded must be absent when false"
        );
    }

    #[test]
    fn to_json_paused_multiple_queries_mixed_grounded() {
        let grounded_query = LlmQuery {
            id: QueryId::batch(0),
            prompt: "verify".into(),
            system: None,
            max_tokens: 100,
            grounded: true,
            underspecified: false,
        };
        let normal_query = LlmQuery {
            id: QueryId::batch(1),
            prompt: "generate".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        };
        let result = FeedResult::Paused {
            queries: vec![grounded_query, normal_query],
        };
        let json = result.to_json("s-batch");

        let qs = json["queries"].as_array().expect("queries should be array");
        assert_eq!(
            qs[0]["grounded"], true,
            "grounded query must have grounded=true"
        );
        assert!(
            qs[1].get("grounded").is_none(),
            "non-grounded query must omit grounded key"
        );
    }

    #[test]
    fn to_json_paused_multiple_queries_mixed_underspecified() {
        let underspec_query = LlmQuery {
            id: QueryId::batch(0),
            prompt: "clarify intent".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: true,
        };
        let normal_query = LlmQuery {
            id: QueryId::batch(1),
            prompt: "generate".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        };
        let result = FeedResult::Paused {
            queries: vec![underspec_query, normal_query],
        };
        let json = result.to_json("s-batch-us");

        let qs = json["queries"].as_array().expect("queries should be array");
        assert_eq!(
            qs[0]["underspecified"], true,
            "underspecified query must have underspecified=true"
        );
        assert!(
            qs[1].get("underspecified").is_none(),
            "non-underspecified query must omit underspecified key"
        );
    }

    #[test]
    fn to_json_paused_single_query_no_system() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "hello".into(),
            system: None,
            max_tokens: 1024,
            grounded: false,
            underspecified: false,
        };
        let result = FeedResult::Paused {
            queries: vec![query],
        };
        let json = result.to_json("s-x");

        assert_eq!(json["status"], "needs_response");
        assert!(json["system"].is_null());
    }

    #[test]
    fn to_json_paused_multiple_queries() {
        let queries = vec![make_query(0), make_query(1), make_query(2)];
        let result = FeedResult::Paused { queries };
        let json = result.to_json("s-multi");

        assert_eq!(json["status"], "needs_response");
        assert_eq!(json["session_id"], "s-multi");

        let qs = json["queries"].as_array().expect("queries should be array");
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0]["id"], "q-0");
        assert_eq!(qs[0]["prompt"], "prompt-0");
        assert_eq!(qs[1]["id"], "q-1");
        assert_eq!(qs[2]["id"], "q-2");
    }

    #[test]
    fn to_json_finished_completed() {
        let result = FeedResult::Finished(ExecutionResult {
            state: TerminalState::Completed {
                result: json!({"answer": 42}),
            },
            metrics: ExecutionMetrics::new(),
        });
        let json = result.to_json("s-done");

        assert_eq!(json["status"], "completed");
        assert_eq!(json["result"]["answer"], 42);
        assert!(json.get("stats").is_some());
    }

    #[test]
    fn to_json_finished_failed() {
        let result = FeedResult::Finished(ExecutionResult {
            state: TerminalState::Failed {
                error: "lua error: bad argument".into(),
            },
            metrics: ExecutionMetrics::new(),
        });
        let json = result.to_json("s-err");

        assert_eq!(json["status"], "error");
        assert_eq!(json["error"], "lua error: bad argument");
    }

    #[test]
    fn to_json_finished_cancelled() {
        let result = FeedResult::Finished(ExecutionResult {
            state: TerminalState::Cancelled,
            metrics: ExecutionMetrics::new(),
        });
        let json = result.to_json("s-cancel");

        assert_eq!(json["status"], "cancelled");
        assert!(json.get("stats").is_some());
    }

    // ─── gen_session_id tests ───

    #[test]
    fn session_id_starts_with_prefix() {
        let id = gen_session_id();
        assert!(id.starts_with("s-"), "id should start with 's-': {id}");
    }

    #[test]
    fn session_id_uniqueness() {
        let ids: Vec<String> = (0..10).map(|_| gen_session_id()).collect();
        let set: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(set.len(), 10, "10 IDs should all be unique");
    }
}
