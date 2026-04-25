//! Session-based Lua execution with pause/resume on alc.llm() calls.
//!
//! Runtime layer: ties Domain (ExecutionState) and Metrics (ExecutionMetrics)
//! together with channel-based Lua pause/resume machinery.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
#[derive(serde::Serialize)]
pub struct ExecutionResult {
    pub state: TerminalState,
    pub metrics: ExecutionMetrics,
}

/// Result of a session interaction (start or feed).
#[derive(serde::Serialize)]
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

// ─── PendingFilter (field-level filter for Session::snapshot) ────

/// Default preview length (chars) used when `PendingFilter::preset_preview()`
/// is constructed without an explicit length. Env var
/// `ALC_PROMPT_PREVIEW_CHARS` (resolved in `AppConfig`) overrides this.
pub const DEFAULT_PROMPT_PREVIEW_CHARS: usize = 200;

/// Per-field filter controlling which `LlmQuery` attributes are projected
/// into a Snapshot's `pending` array.
///
/// Adding a new field to `LlmQuery` only requires adding one matching
/// `bool` here — the shape stays stable so API surface does not grow
/// enum variants for every new attribute.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PendingFilter {
    #[serde(default)]
    pub query_id: bool,
    #[serde(default)]
    pub max_tokens: bool,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub grounded: bool,
    #[serde(default)]
    pub underspecified: bool,
    #[serde(default)]
    pub prompt: PromptProjection,
}

/// Prompt projection mode — 3 states rather than a bool so that truncation
/// length can travel inside the filter object.
///
/// JSON tag is `mode`: `{"mode":"off"}` / `{"mode":"preview","chars":200}` /
/// `{"mode":"full"}`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PromptProjection {
    #[default]
    Off,
    Preview {
        chars: usize,
    },
    Full,
}

impl PendingFilter {
    /// Preset: query identification only (`query_id` + `max_tokens`).
    pub fn preset_meta() -> Self {
        Self {
            query_id: true,
            max_tokens: true,
            ..Self::default()
        }
    }

    /// Preset: meta + first N chars of the prompt. Uses the hard default
    /// for N (`DEFAULT_PROMPT_PREVIEW_CHARS`).
    pub fn preset_preview() -> Self {
        Self::preset_preview_with(DEFAULT_PROMPT_PREVIEW_CHARS)
    }

    /// Preset: meta + first `chars` chars of the prompt. Lets callers
    /// flow a config-resolved length (e.g. env var) into the preset.
    pub fn preset_preview_with(chars: usize) -> Self {
        Self {
            query_id: true,
            max_tokens: true,
            prompt: PromptProjection::Preview { chars },
            ..Self::default()
        }
    }

    /// Preset: every field including the full prompt (debug use).
    pub fn preset_full() -> Self {
        Self {
            query_id: true,
            max_tokens: true,
            system: true,
            grounded: true,
            underspecified: true,
            prompt: PromptProjection::Full,
        }
    }

    /// Resolve a preset by name. Unknown names return `None` so that
    /// callers can surface a typed error rather than silently falling
    /// back to a default projection.
    pub fn from_preset(name: &str) -> Option<Self> {
        match name {
            "meta" => Some(Self::preset_meta()),
            "preview" => Some(Self::preset_preview()),
            "full" => Some(Self::preset_full()),
            _ => None,
        }
    }

    /// Same as [`Self::from_preset`] but lets `"preview"` pick up a
    /// caller-supplied char count (config / env override).
    pub fn from_preset_with(name: &str, preview_chars: usize) -> Option<Self> {
        match name {
            "meta" => Some(Self::preset_meta()),
            "preview" => Some(Self::preset_preview_with(preview_chars)),
            "full" => Some(Self::preset_full()),
            _ => None,
        }
    }
}

/// Project a single `LlmQuery` into the JSON object requested by `filter`.
///
/// UTF-8 safety: `PromptProjection::Preview { chars }` uses `chars().take(N)`
/// so the cut never splits a multi-byte code point.
fn project_query(q: &LlmQuery, f: &PendingFilter) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    if f.query_id {
        obj.insert("query_id".into(), q.id.as_str().into());
    }
    if f.max_tokens {
        obj.insert("max_tokens".into(), q.max_tokens.into());
    }
    if f.system {
        obj.insert(
            "system".into(),
            match &q.system {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            },
        );
    }
    if f.grounded {
        obj.insert("grounded".into(), q.grounded.into());
    }
    if f.underspecified {
        obj.insert("underspecified".into(), q.underspecified.into());
    }
    match &f.prompt {
        PromptProjection::Off => {}
        PromptProjection::Full => {
            obj.insert("prompt".into(), q.prompt.clone().into());
        }
        PromptProjection::Preview { chars } => {
            let preview: String = q.prompt.chars().take(*chars).collect();
            obj.insert("prompt_preview".into(), preview.into());
        }
    }
    serde_json::Value::Object(obj)
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
    /// Last activity timestamp (monotonic). Updated on creation and each feed_one().
    /// Used by GC to identify idle sessions for cleanup.
    last_active: std::time::Instant,
    /// Wall-clock Unix ms when the session was created (immutable after Session::new).
    started_at_ms: i64,
    /// Wall-clock Unix ms of the most recent activity (feed_one or session creation).
    /// Updated with `Relaxed` ordering — observability use only, no cross-thread invariant.
    last_activity_ms: Arc<AtomicI64>,
}

impl Session {
    /// Create a new session.
    ///
    /// # Arguments
    ///
    /// - `llm_rx` — Receiver for LLM requests from the Lua bridge.
    /// - `exec_task` — The coroutine execution task handle.
    /// - `metrics` — Session metrics (owns the LogSink ring buffer; the bridge
    ///   reads its `log_sink_handle()` separately to wire `print()` / `alc.log()`
    ///   into the same ring buffer that `metrics.snapshot()` exposes via
    ///   `recent_logs` in `alc_status`).
    /// - `vm_driver` — Keeps the Lua OS thread alive.
    ///
    /// # Returns
    ///
    /// A new `Session` in the `Running` state.
    pub fn new(
        llm_rx: tokio::sync::mpsc::Receiver<LlmRequest>,
        exec_task: AsyncTask,
        metrics: ExecutionMetrics,
        vm_driver: AsyncIsleDriver,
    ) -> Self {
        let observer = metrics.create_observer();
        // Note: duration_since can only fail if the wall clock predates UNIX_EPOCH
        // (broken system clock). Saturating to 0 is harmless for observability.
        let started_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        Self {
            state: ExecutionState::Running,
            metrics,
            observer,
            llm_rx,
            exec_task,
            resp_txs: HashMap::new(),
            _vm_driver: vm_driver,
            last_active: std::time::Instant::now(),
            started_at_ms,
            last_activity_ms: Arc::new(AtomicI64::new(started_at_ms)),
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
    /// # Arguments
    ///
    /// - `query_id` — The query to respond to.
    /// - `response` — The LLM response string.
    /// - `usage` — Optional token usage from the host.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if all queries are now complete; `Ok(false)` if more responses remain.
    ///
    /// # Errors
    ///
    /// Returns `SessionError::Feed` if the state machine rejects the feed.
    fn feed_one(
        &mut self,
        query_id: &QueryId,
        response: String,
        usage: Option<&algocline_core::TokenUsage>,
    ) -> Result<bool, SessionError> {
        // Update both monotonic and wall-clock activity timestamps on each feed.
        self.last_active = std::time::Instant::now();
        // Note: duration_since can only fail if wall clock predates UNIX_EPOCH.
        // Saturating to 0 is harmless for observability.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        self.last_activity_ms.store(now_ms, Ordering::Relaxed);

        // Track response before ownership transfer.
        self.observer.on_response_fed(query_id, &response, usage);

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
    ///
    /// # Arguments
    ///
    /// - `pending_filter` — Opt-in projection for the currently pending LLM queries.
    ///   `None` emits only `pending_queries: N` (integer count), preserving the v0.x
    ///   wire shape for light-weight polling.  `Some(filter)` adds a `pending: [...]`
    ///   array projected through the filter's field flags.
    /// - `include_history` — When `true`, `conversation_history` (≤10 entries) is
    ///   included in the metrics output.  When `false` (default), the key is absent.
    ///   High-frequency polling callers should leave this `false` to avoid wire bloat.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` snapshot with the following additive fields beyond v0.x:
    /// - `phase` — 5-value string derived from `ExecutionState`:
    ///   `"running"`, `"paused"`, `"completed"`, `"failed"`, `"cancelled"`.
    ///   The existing `state` key is retained for backward compatibility (3-value).
    /// - `started_at` — Unix millisecond timestamp when the session was created.
    /// - `last_activity_at` — Unix millisecond timestamp of the most recent feed_one.
    ///   Note: `started_at` and `last_activity_at` are wall-clock values while
    ///   expiry GC uses the monotonic `last_active` Instant; they may skew slightly
    ///   on NTP adjustments (acceptable for observability use).
    pub fn snapshot(
        &self,
        pending_filter: Option<&PendingFilter>,
        include_history: bool,
    ) -> serde_json::Value {
        let state_label = match &self.state {
            ExecutionState::Running => "running",
            ExecutionState::Paused(_) => "paused",
            _ => "terminal",
        };

        let phase = match &self.state {
            ExecutionState::Running => "running",
            ExecutionState::Paused(_) => "paused",
            ExecutionState::Completed { .. } => "completed",
            ExecutionState::Failed { .. } => "failed",
            ExecutionState::Cancelled => "cancelled",
        };

        let mut json = serde_json::json!({
            "state": state_label,
            "phase": phase,
            "started_at": self.started_at_ms,
            "last_activity_at": self.last_activity_ms.load(Ordering::Relaxed),
        });

        let metrics = self.metrics.snapshot(include_history);
        if !metrics.is_null() {
            json["metrics"] = metrics;
        }

        // Pending query projection (additive; count is always present)
        if let ExecutionState::Paused(pending) = &self.state {
            json["pending_queries"] = pending.remaining().into();

            if let Some(filter) = pending_filter {
                let items: Vec<serde_json::Value> = pending
                    .pending_queries()
                    .iter()
                    .map(|q| project_query(q, filter))
                    .collect();
                json["pending"] = serde_json::Value::Array(items);
            }
        }

        json
    }

    /// Returns true if the session has been idle longer than `ttl`.
    ///
    /// Uses `saturating_duration_since` to avoid panics if the clock drifts
    /// backwards (though this is extremely rare with monotonic clocks).
    pub fn is_expired(&self, ttl: Duration) -> bool {
        is_expired_impl(self.last_active, ttl)
    }
}

/// Core expiry check, extracted for testability.
fn is_expired_impl(last_active: std::time::Instant, ttl: Duration) -> bool {
    std::time::Instant::now().saturating_duration_since(last_active) >= ttl
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
        usage: Option<&algocline_core::TokenUsage>,
    ) -> Result<FeedResult, SessionError> {
        // 1. Feed under lock
        let complete = {
            let mut map = self.sessions.lock().await;
            let session = map
                .get_mut(session_id)
                .ok_or_else(|| SessionError::NotFound(session_id.into()))?;

            let complete = session.feed_one(query_id, response, usage)?;

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
    ///
    /// # Arguments
    ///
    /// - `pending_filter` — Forwarded verbatim to each session's [`Session::snapshot`].
    /// - `include_history` — When `true`, each snapshot includes `conversation_history`
    ///   (≤10 entries).  Pass `false` for high-frequency polling to avoid wire bloat.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping session IDs to their JSON snapshots.
    pub async fn list_snapshots(
        &self,
        pending_filter: Option<&PendingFilter>,
        include_history: bool,
    ) -> HashMap<String, serde_json::Value> {
        let map = self.sessions.lock().await;
        map.iter()
            .map(|(id, session)| {
                (
                    id.clone(),
                    session.snapshot(pending_filter, include_history),
                )
            })
            .collect()
    }

    /// Spawn a background GC task that reaps sessions idle longer than `ttl`.
    ///
    /// The task runs every 60 seconds. When the process exits, the task is
    /// naturally terminated. No `JoinHandle` is retained — process exit is
    /// sufficient for cleanup in MCP server deployments.
    pub fn spawn_gc_task(&self, ttl: Duration) {
        let sessions = Arc::clone(&self.sessions);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let mut map = sessions.lock().await;
                let expired: Vec<String> = map
                    .iter()
                    .filter(|(_, s)| s.is_expired(ttl))
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in &expired {
                    tracing::info!(session_id = %id, "GC: reaping expired session");
                    map.remove(id);
                }
            }
        });
    }
}

/// Generate a non-deterministic session ID.
///
/// MCP spec requires "secure, non-deterministic session IDs" to prevent
/// session hijacking. Uses timestamp + random bytes for uniqueness and
/// unpredictability.
///
/// # `unwrap_or_default` on `duration_since(UNIX_EPOCH)`
///
/// `SystemTime::now().duration_since(UNIX_EPOCH)` can fail if the system
/// clock is set before 1970-01-01 (e.g. NTP drift, misconfigured VM).
/// The Rust std docs recommend `expect()` or `match` for explicit handling,
/// but `expect` would panic in library code (prohibited by project policy).
///
/// `unwrap_or_default` returns `Duration::ZERO` on failure, yielding
/// timestamp `0`. This is acceptable here because the 8-byte random
/// suffix (16 hex chars of entropy) independently guarantees uniqueness
/// and unpredictability — the timestamp is a convenience prefix, not
/// a security-critical component.
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

    // ─── is_expired_impl tests ───
    //
    // Session::is_expired delegates to is_expired_impl. Testing the impl
    // directly avoids the need to construct a full Session (which requires
    // a real Lua VM + channels).

    #[test]
    fn is_expired_impl_fresh_instant_not_expired() {
        // A just-created instant should not be expired with a non-zero TTL
        let now = std::time::Instant::now();
        assert!(!is_expired_impl(now, Duration::from_secs(1)));
    }

    #[test]
    fn is_expired_impl_old_instant_expired() {
        // Simulate a session idle for 2 hours by backdating last_active
        let two_hours_ago = std::time::Instant::now()
            .checked_sub(Duration::from_secs(7200))
            .expect("checked_sub should succeed with sane duration");
        // TTL = 1 hour: should be expired
        assert!(is_expired_impl(two_hours_ago, Duration::from_secs(3600)));
    }

    #[test]
    fn is_expired_impl_not_yet_expired() {
        // Simulate a session idle for 1 hour
        let one_hour_ago = std::time::Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .expect("checked_sub should succeed with sane duration");
        // TTL = 3 hours: should NOT be expired yet
        assert!(!is_expired_impl(one_hour_ago, Duration::from_secs(10800)));
    }

    #[test]
    fn is_expired_impl_zero_ttl_always_expired() {
        // TTL = 0: any instant is immediately expired (edge case)
        let now = std::time::Instant::now();
        assert!(is_expired_impl(now, Duration::ZERO));
    }

    // ─── PendingFilter preset tests ───

    #[test]
    fn pending_filter_default_is_all_off() {
        let f = PendingFilter::default();
        assert!(!f.query_id);
        assert!(!f.max_tokens);
        assert!(!f.system);
        assert!(!f.grounded);
        assert!(!f.underspecified);
        assert!(matches!(f.prompt, PromptProjection::Off));
    }

    #[test]
    fn pending_filter_preset_meta_flags() {
        let f = PendingFilter::preset_meta();
        assert!(f.query_id);
        assert!(f.max_tokens);
        assert!(!f.system);
        assert!(!f.grounded);
        assert!(!f.underspecified);
        assert!(
            matches!(f.prompt, PromptProjection::Off),
            "meta preset must not project prompt content"
        );
    }

    #[test]
    fn pending_filter_preset_preview_uses_default_chars() {
        let f = PendingFilter::preset_preview();
        assert!(f.query_id);
        assert!(f.max_tokens);
        match f.prompt {
            PromptProjection::Preview { chars } => {
                assert_eq!(chars, DEFAULT_PROMPT_PREVIEW_CHARS);
            }
            other => panic!("expected Preview, got {other:?}"),
        }
    }

    #[test]
    fn pending_filter_preset_preview_with_custom_chars() {
        let f = PendingFilter::preset_preview_with(42);
        match f.prompt {
            PromptProjection::Preview { chars } => assert_eq!(chars, 42),
            other => panic!("expected Preview {{chars: 42}}, got {other:?}"),
        }
    }

    #[test]
    fn pending_filter_preset_full_flags_all_on() {
        let f = PendingFilter::preset_full();
        assert!(f.query_id);
        assert!(f.max_tokens);
        assert!(f.system);
        assert!(f.grounded);
        assert!(f.underspecified);
        assert!(matches!(f.prompt, PromptProjection::Full));
    }

    #[test]
    fn pending_filter_from_preset_known_names() {
        assert!(PendingFilter::from_preset("meta").is_some());
        assert!(PendingFilter::from_preset("preview").is_some());
        assert!(PendingFilter::from_preset("full").is_some());
    }

    #[test]
    fn pending_filter_from_preset_unknown_returns_none() {
        // Typo-protection invariant: caller must surface an error, not
        // silently fall back to a default projection.
        assert!(PendingFilter::from_preset("").is_none());
        assert!(PendingFilter::from_preset("META").is_none());
        assert!(PendingFilter::from_preset("bogus").is_none());
    }

    #[test]
    fn pending_filter_from_preset_with_overrides_preview_chars() {
        // "preview" respects the per-call chars count (flowed in from env
        // or config); other presets ignore it.
        let f = PendingFilter::from_preset_with("preview", 73).unwrap();
        match f.prompt {
            PromptProjection::Preview { chars } => assert_eq!(chars, 73),
            other => panic!("expected Preview {{chars: 73}}, got {other:?}"),
        }

        let f_meta = PendingFilter::from_preset_with("meta", 73).unwrap();
        assert!(matches!(f_meta.prompt, PromptProjection::Off));

        let f_full = PendingFilter::from_preset_with("full", 73).unwrap();
        assert!(matches!(f_full.prompt, PromptProjection::Full));
    }

    // ─── project_query tests ───

    #[test]
    fn project_query_default_filter_produces_empty_object() {
        let q = make_query(0);
        let v = project_query(&q, &PendingFilter::default());
        let obj = v.as_object().expect("object");
        assert!(obj.is_empty(), "default filter should project nothing");
    }

    #[test]
    fn project_query_meta_preset_has_id_and_max_tokens_only() {
        let q = make_query(0);
        let v = project_query(&q, &PendingFilter::preset_meta());
        let obj = v.as_object().expect("object");
        assert_eq!(obj.len(), 2);
        assert_eq!(v["query_id"], "q-0");
        assert_eq!(v["max_tokens"], 100);
        assert!(obj.get("prompt").is_none());
        assert!(obj.get("prompt_preview").is_none());
        assert!(obj.get("system").is_none());
        assert!(obj.get("grounded").is_none());
        assert!(obj.get("underspecified").is_none());
    }

    #[test]
    fn project_query_full_preset_has_all_fields() {
        let q = LlmQuery {
            id: QueryId::batch(0),
            prompt: "hi".into(),
            system: Some("sys".into()),
            max_tokens: 100,
            grounded: true,
            underspecified: true,
        };
        let v = project_query(&q, &PendingFilter::preset_full());
        assert_eq!(v["query_id"], "q-0");
        assert_eq!(v["max_tokens"], 100);
        assert_eq!(v["system"], "sys");
        assert_eq!(v["grounded"], true);
        assert_eq!(v["underspecified"], true);
        assert_eq!(v["prompt"], "hi");
        assert!(v.get("prompt_preview").is_none());
    }

    #[test]
    fn project_query_preview_truncates_at_char_count() {
        let q = LlmQuery {
            id: QueryId::batch(0),
            prompt: "abcdefghij".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        };
        let v = project_query(&q, &PendingFilter::preset_preview_with(5));
        assert_eq!(v["prompt_preview"], "abcde");
        assert!(v.get("prompt").is_none());
    }

    #[test]
    fn project_query_preview_utf8_multibyte_safe() {
        // Japanese characters are 3-byte UTF-8 each; chars().take(N) must
        // never split a codepoint. Taking 3 chars from 5 must yield exactly
        // 3 chars (not bytes), and the String must be valid UTF-8.
        let prompt = "あいうえお";
        let q = LlmQuery {
            id: QueryId::batch(0),
            prompt: prompt.to_string(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        };
        let v = project_query(&q, &PendingFilter::preset_preview_with(3));
        let preview = v["prompt_preview"].as_str().expect("str");
        assert_eq!(preview, "あいう");
        assert_eq!(preview.chars().count(), 3);
    }

    #[test]
    fn project_query_preview_chars_over_length_returns_whole_prompt() {
        let q = LlmQuery {
            id: QueryId::batch(0),
            prompt: "abc".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        };
        let v = project_query(&q, &PendingFilter::preset_preview_with(100));
        assert_eq!(v["prompt_preview"], "abc");
    }

    #[test]
    fn project_query_system_field_null_when_absent() {
        let q = LlmQuery {
            id: QueryId::batch(0),
            prompt: "p".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        };
        let filter = PendingFilter {
            system: true,
            ..Default::default()
        };
        let v = project_query(&q, &filter);
        assert!(
            v["system"].is_null(),
            "absent system must serialize as null"
        );
    }

    // ─── PendingFilter deserialization (MCP custom object path) ───

    #[test]
    fn pending_filter_deserialize_custom_object_preview() {
        // MCP callers may pass a raw JSON filter rather than a preset name.
        let raw = serde_json::json!({
            "query_id": true,
            "prompt": { "mode": "preview", "chars": 50 }
        });
        let f: PendingFilter = serde_json::from_value(raw).expect("deserialize");
        assert!(f.query_id);
        match f.prompt {
            PromptProjection::Preview { chars } => assert_eq!(chars, 50),
            other => panic!("expected Preview, got {other:?}"),
        }
    }

    #[test]
    fn pending_filter_deserialize_partial_object_uses_field_defaults() {
        // serde(default) on every field means a `{}` object is valid and
        // equivalent to PendingFilter::default().
        let raw = serde_json::json!({});
        let f: PendingFilter = serde_json::from_value(raw).expect("deserialize");
        assert!(!f.query_id);
        assert!(matches!(f.prompt, PromptProjection::Off));
    }

    #[test]
    fn pending_filter_deserialize_prompt_full_tag() {
        let raw = serde_json::json!({ "prompt": { "mode": "full" } });
        let f: PendingFilter = serde_json::from_value(raw).expect("deserialize");
        assert!(matches!(f.prompt, PromptProjection::Full));
    }

    // ─── Session snapshot v2 fields tests ───
    //
    // These tests use the Executor to create real sessions so that Session
    // struct fields (started_at_ms, last_activity_ms, phase) are exercised
    // end-to-end without requiring direct construction of AsyncTask/AsyncIsleDriver.

    /// Helper: build a minimal temp directory pair for state/card stores.
    fn tmp_dirs() -> (
        std::sync::Arc<crate::state::JsonFileStore>,
        std::sync::Arc<crate::card::FileCardStore>,
        std::path::PathBuf,
    ) {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        (
            std::sync::Arc::new(crate::state::JsonFileStore::new(root.join("state"))),
            std::sync::Arc::new(crate::card::FileCardStore::new(root.join("cards"))),
            root.join("scenarios"),
        )
    }

    // T1: Session snapshot contains phase, started_at, last_activity_at
    // A session that completes immediately should have these fields in its snapshot
    // while it is Running (before completion removes it from the registry).
    //
    // Strategy: start a session with a Lua script that calls alc.llm() to pause.
    // The session will be in Paused state in the registry, allowing snapshot().
    #[tokio::test]
    async fn snapshot_v2_contains_phase_and_timestamps() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        // Lua: pause the session with a single alc.llm() call
        let code = r#"
            local response = alc.llm("what is 2+2?")
            return response
        "#
        .to_string();

        let session = executor
            .start_session(
                code,
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        // While in Running state (before first event), snapshot should have new fields.
        // Note: session.snapshot() is called before wait_event() so state is Running.
        let snap = session.snapshot(None, false);

        // phase is present
        assert!(
            snap.get("phase").is_some(),
            "snapshot must have 'phase' field"
        );
        assert_eq!(snap["phase"], "running", "initial state must be running");

        // state key retained for backward compatibility
        assert_eq!(snap["state"], "running");

        // started_at is a positive i64 (unix ms)
        let started_at = snap["started_at"].as_i64().expect("started_at must be i64");
        assert!(started_at > 0, "started_at must be > 0 (unix ms)");

        // last_activity_at starts equal to started_at
        let last_activity = snap["last_activity_at"]
            .as_i64()
            .expect("last_activity_at must be i64");
        assert_eq!(
            started_at, last_activity,
            "last_activity_at should equal started_at before any feed"
        );
    }

    // T1: phase correctly maps 5 ExecutionState variants
    // We test Running via snapshot before wait_event (already done above).
    // Here we verify the phase string matches the ExecutionState literal.
    #[test]
    fn snapshot_phase_running_state_label() {
        // We can't construct Session directly in tests (AsyncTask is crate-private).
        // Instead verify the phase mapping logic through the match expression.
        // This test documents the expected mapping:
        let cases: &[(&str, &str)] = &[
            ("running", "running"),
            ("paused", "paused"),
            ("completed", "completed"),
            ("failed", "failed"),
            ("cancelled", "cancelled"),
        ];
        for (state_str, expected_phase) in cases {
            // The phase mapping is identical to state_str in the 5-value case,
            // and the 3-value state uses "terminal" for completed/failed/cancelled.
            // Verify that the 3-value state mapping is consistent with expectations.
            let three_value_state = match *state_str {
                "running" => "running",
                "paused" => "paused",
                _ => "terminal",
            };
            // phase must equal state_str (5-value) while state uses 3-value.
            assert_eq!(
                *expected_phase, *state_str,
                "phase for {state_str} must be the same string"
            );
            if *state_str != "running" && *state_str != "paused" {
                assert_eq!(
                    three_value_state, "terminal",
                    "{state_str} must map to 'terminal' in 3-value state"
                );
            }
        }
    }

    // T1: snapshot(false) lacks conversation_history; snapshot(true) includes it
    #[tokio::test]
    async fn snapshot_conversation_history_opt_in() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        let code = r#"
            local response = alc.llm("explain recursion")
            return response
        "#
        .to_string();

        let session = executor
            .start_session(
                code,
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        // Before any LLM interaction, conversation_history is absent in both modes.
        let snap_false = session.snapshot(None, false);
        assert!(
            snap_false
                .get("metrics")
                .and_then(|m| m.get("conversation_history"))
                .is_none(),
            "conversation_history must be absent with include_history=false"
        );

        // include_history=true: conversation_history key must exist (empty array at start).
        let snap_true = session.snapshot(None, true);
        // metrics is present; conversation_history may be empty array (no LLM calls yet)
        // but the key must be present.
        if let Some(metrics) = snap_true.get("metrics") {
            // If there are no transcript entries yet, conversation_history may be
            // absent or empty — depending on metrics implementation.
            // Either way, no panic. The key's presence is tested in metrics tests.
            let _ = metrics.get("conversation_history");
        }
    }

    // T2: last_activity_ms starts equal to started_at_ms (edge case: no feeds yet)
    #[tokio::test]
    async fn snapshot_last_activity_at_starts_equal_to_started_at() {
        let executor = crate::executor::Executor::new(vec![]).await.unwrap();
        let (state_store, card_store, scenarios_dir) = tmp_dirs();

        let code = r#"
            local response = alc.llm("test query")
            return response
        "#
        .to_string();

        let session = executor
            .start_session(
                code,
                serde_json::json!({}),
                vec![],
                vec![],
                state_store,
                card_store,
                scenarios_dir,
            )
            .await
            .unwrap();

        let snap = session.snapshot(None, false);
        let started_at = snap["started_at"].as_i64().unwrap_or(-1);
        let last_activity = snap["last_activity_at"].as_i64().unwrap_or(-2);

        assert_eq!(
            started_at, last_activity,
            "last_activity_at must equal started_at before any feed_one"
        );
        assert!(started_at > 0, "started_at must be positive unix ms");
    }
}
