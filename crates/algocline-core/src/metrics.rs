use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::budget::Budget;
use crate::observer::ExecutionObserver;
use crate::progress::ProgressInfo;
use crate::recent_log::{LogEntry, LogSink};
use crate::tokens::{estimate_tokens, TokenCount, TokenSource};
use crate::{BudgetHandle, CustomMetrics, CustomMetricsHandle, LlmQuery, ProgressHandle, QueryId};

// ─── Transcript ─────────────────────────────────────────────

/// A single prompt/response exchange in the transcript.
///
/// Each entry is the authoritative token record for one LLM call.
/// Token counts start as character-based estimates (`on_paused`) and
/// are upgraded to host-provided values when available (`on_response_fed`).
/// Session-level totals are computed by summing across all entries.
struct TranscriptEntry {
    query_id: String,
    prompt: String,
    system: Option<String>,
    response: Option<String>,
    /// Prompt token count for this query (Estimated or Provided).
    prompt_tokens: u64,
    prompt_source: TokenSource,
    /// Response token count for this query (Estimated or Provided).
    /// Zero until `on_response_fed` is called.
    response_tokens: u64,
    response_source: TokenSource,
    /// Unix millisecond timestamp when the LLM request was issued (on_paused).
    started_at_ms: i64,
    /// Unix millisecond timestamp when the LLM response was received (on_response_fed).
    /// None until the response arrives.
    completed_at_ms: Option<i64>,
}

impl TranscriptEntry {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "query_id": self.query_id,
            "prompt": self.prompt,
            "system": self.system,
            "response": self.response,
        })
    }

    /// Project this entry into the `conversation_history` JSON shape for
    /// `alc_status` output (opt-in via `include_history=true`).
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` with fields:
    /// `query_id`, `prompt`, `response`, `prompt_tokens`, `response_tokens`,
    /// `started_at` (unix ms), `completed_at` (unix ms or null).
    fn to_history_json(&self) -> serde_json::Value {
        serde_json::json!({
            "query_id": self.query_id,
            "prompt": self.prompt,
            "response": self.response,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "started_at": self.started_at_ms,
            "completed_at": self.completed_at_ms,
        })
    }
}

/// Metrics automatically derived from the execution lifecycle.
///
/// # Locking design
///
/// `SessionStatus` is wrapped in `Arc<std::sync::Mutex>` and shared across:
///
/// | Consumer | Thread | Access | Via |
/// |---|---|---|---|
/// | `MetricsObserver` | tokio async task | write (on_paused, on_response_fed, etc.) | `Arc<Mutex<SessionStatus>>` |
/// | `BudgetHandle` | Lua OS thread | read (check, remaining) | `Arc<Mutex<SessionStatus>>` |
/// | `ProgressHandle` | Lua OS thread | write (set) | `Arc<Mutex<SessionStatus>>` |
/// | `ExecutionMetrics` | tokio async task | read (to_json, snapshot, transcript_to_json) | `Arc<Mutex<SessionStatus>>` |
///
/// ## Why `std::sync::Mutex` (not `tokio::sync::Mutex`)
///
/// All lock holders complete within microseconds (field reads, arithmetic,
/// small JSON construction) and **never hold the lock across `.await` points**.
/// Per tokio guidance, `std::sync::Mutex` is preferred when the critical
/// section is short and synchronous.
///
/// ## Lock ordering
///
/// When nested with `SessionRegistry`'s `tokio::sync::Mutex` (lock **C**),
/// the invariant is always **C → A** (registry lock acquired first).
/// No code path acquires A then C, so deadlock is structurally impossible.
///
/// ## Contention analysis
///
/// Each session creates its own `ExecutionMetrics` instance (see
/// `Executor::start_session`), so the `SessionStatus` mutex is **not shared
/// across sessions**. Within a single session, the Lua thread and the
/// tokio async task alternate via mpsc channel handoff:
///
/// 1. Lua calls `alc.llm()` → `BudgetHandle::check()` locks A (Lua thread)
/// 2. Lock released, then `tx.send(LlmRequest)` (mpsc)
/// 3. `Session::wait_event()` receives request → `on_paused()` locks A (async task)
///
/// Steps 1 and 3 are sequenced by the mpsc channel, so they never contend.
/// The only true contention is `snapshot()` (from `alc_status`) vs. observer
/// methods, which is harmless given microsecond hold times.
///
/// ## Poison policy
///
/// Poison can only occur if a thread panics while holding this lock.
/// The only panic-capable code under the lock is `Vec::push` and
/// `serde_json::json!` (both panic only on OOM). On OOM the process
/// is unrecoverable, so poison handling is academic.
///
/// Policy: `BudgetHandle::check()` propagates poison as `Err` (because
/// it gates Lua control flow). All other consumers silently skip on
/// poison (observation/recording — degraded but non-fatal).
/// If you encounter a poison error in production, it indicates either
/// OOM or a bug in code executed under the lock.
pub(crate) struct SessionStatus {
    started_at: Instant,
    ended_at: Option<Instant>,
    pub(crate) llm_calls: u64,
    pauses: u64,
    rounds: u64,
    total_prompt_chars: u64,
    total_response_chars: u64,
    transcript: Vec<TranscriptEntry>,
    pub(crate) budget: Option<Budget>,
    pub(crate) progress: Option<ProgressInfo>,
}

impl SessionStatus {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            ended_at: None,
            llm_calls: 0,
            pauses: 0,
            rounds: 0,
            total_prompt_chars: 0,
            total_response_chars: 0,
            transcript: Vec::new(),
            budget: None,
            progress: None,
        }
    }

    /// Aggregate prompt tokens from all transcript entries.
    fn prompt_token_count(&self) -> TokenCount {
        let mut tc = TokenCount::new(TokenSource::Definite);
        for e in &self.transcript {
            tc.accumulate(e.prompt_tokens, e.prompt_source);
        }
        tc
    }

    /// Aggregate response tokens from all transcript entries.
    fn response_token_count(&self) -> TokenCount {
        let mut tc = TokenCount::new(TokenSource::Definite);
        for e in &self.transcript {
            tc.accumulate(e.response_tokens, e.response_source);
        }
        tc
    }

    /// Total tokens (prompt + response) across all transcript entries.
    fn total_tokens(&self) -> u64 {
        self.transcript
            .iter()
            .map(|e| e.prompt_tokens + e.response_tokens)
            .sum()
    }

    /// Wall-clock elapsed milliseconds since session start.
    fn elapsed_ms(&self) -> u64 {
        self.ended_at
            .map(|end| end.duration_since(self.started_at).as_millis() as u64)
            .unwrap_or_else(|| self.started_at.elapsed().as_millis() as u64)
    }

    fn to_json(&self) -> serde_json::Value {
        let prompt_tc = self.prompt_token_count();
        let response_tc = self.response_token_count();
        let total_tc = TokenCount {
            tokens: prompt_tc.tokens + response_tc.tokens,
            source: prompt_tc.source.weaker(response_tc.source),
        };
        let mut json = serde_json::json!({
            "elapsed_ms": self.elapsed_ms(),
            "llm_calls": self.llm_calls,
            "pauses": self.pauses,
            "rounds": self.rounds,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars,
            "prompt_tokens": prompt_tc.to_json(),
            "response_tokens": response_tc.to_json(),
            "total_tokens": total_tc.to_json(),
        });
        if let Some(ref b) = self.budget {
            json["budget"] = b.to_json();
        }
        json
    }

    pub(crate) fn check_budget(&self) -> Result<(), String> {
        match self.budget {
            Some(ref b) => b.check(self.llm_calls, self.elapsed_ms(), self.total_tokens()),
            None => Ok(()),
        }
    }

    /// Lightweight snapshot for external observation (alc_status).
    ///
    /// Returns running metrics with additive v2 fields:
    /// - `tokens` — cumulative prompt/response/total counts plus `current_query`
    ///   for the in-flight request (if any).  Always included.
    /// - `recent_logs` — capped ring buffer (≤20) of recent log entries.
    ///   Always included.
    /// - `conversation_history` — last ≤10 transcript entries.
    ///   Included **only** when `include_history=true` to protect high-frequency
    ///   polling callers from wire-size inflation (see design: wf-sim
    ///   restructure_shape verdict and metrics.rs doc "without transcript which
    ///   can be large").
    ///
    /// # Arguments
    ///
    /// - `include_history` — When `true`, `conversation_history` (≤10 entries)
    ///   is appended to the output JSON.  When `false`, the key is absent.
    /// - `log_sink` — The session's [`LogSink`] from which `recent_logs` is
    ///   populated (held at `ExecutionMetrics` level to allow lock-free cloning).
    fn snapshot(&self, include_history: bool, log_sink: &LogSink) -> serde_json::Value {
        // Build token aggregates.
        let prompt_tc = self.prompt_token_count();
        let response_tc = self.response_token_count();
        let total_tokens = prompt_tc.tokens + response_tc.tokens;

        // Determine in-flight query (last transcript entry without a response,
        // only meaningful while paused).
        let current_query = self.transcript.last().and_then(|e| {
            if e.response.is_none() {
                Some(serde_json::json!({
                    "query_id": e.query_id,
                    "prompt_tokens": e.prompt_tokens,
                    "started_waiting_at": e.started_at_ms,
                }))
            } else {
                None
            }
        });

        let mut json = serde_json::json!({
            "elapsed_ms": self.elapsed_ms(),
            "llm_calls": self.llm_calls,
            "rounds": self.rounds,
            "tokens": {
                "prompt_total": prompt_tc.tokens,
                "response_total": response_tc.tokens,
                "total": total_tokens,
                "current_query": current_query,
            },
            "recent_logs": log_sink.to_json(),
        });

        if let Some(ref p) = self.progress {
            json["progress"] = serde_json::json!({
                "step": p.step,
                "total": p.total,
                "message": p.message,
            });
        }

        if let Some(ref b) = self.budget {
            json["budget_remaining"] =
                b.remaining_json(self.llm_calls, self.elapsed_ms(), self.total_tokens());
        }

        if include_history {
            // Emit the last ≤10 transcript entries in chronological order.
            let history: Vec<serde_json::Value> = self
                .transcript
                .iter()
                .rev()
                .take(10)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|e| e.to_history_json())
                .collect();
            json["conversation_history"] = serde_json::Value::Array(history);
        }

        json
    }

    pub(crate) fn budget_remaining(&self) -> serde_json::Value {
        match self.budget {
            None => serde_json::Value::Null,
            Some(ref b) => b.remaining_json(self.llm_calls, self.elapsed_ms(), self.total_tokens()),
        }
    }
}

/// Measurement data for a single execution.
///
/// Created per-session in `Executor::start_session()`. The `auto` and
/// `custom` mutexes are **not shared across sessions** — each session
/// gets independent instances. Handles (`BudgetHandle`, `ProgressHandle`,
/// `MetricsObserver`) are cloned from the same `Arc` and handed to the
/// Lua bridge and observer respectively.
///
/// The `log_sink` is a separate `Arc`-backed ring buffer for per-session log
/// capture. It is shared with the Lua bridge (via `log_sink_handle()`) so that
/// `print()` and `alc.log()` output is routed directly without acquiring the
/// `SessionStatus` mutex.
pub struct ExecutionMetrics {
    auto: Arc<Mutex<SessionStatus>>,
    custom: Arc<Mutex<CustomMetrics>>,
    log_sink: LogSink,
}

impl ExecutionMetrics {
    pub fn new() -> Self {
        Self {
            auto: Arc::new(Mutex::new(SessionStatus::new())),
            custom: Arc::new(Mutex::new(CustomMetrics::new())),
            log_sink: LogSink::new(),
        }
    }

    /// JSON snapshot combining auto and custom metrics.
    pub fn to_json(&self) -> serde_json::Value {
        let auto_json = self
            .auto
            .lock()
            .map(|m| m.to_json())
            .unwrap_or(serde_json::Value::Null);

        let custom_json = self
            .custom
            .lock()
            .map(|m| m.to_json())
            .unwrap_or(serde_json::Value::Null);

        serde_json::json!({
            "auto": auto_json,
            "custom": custom_json,
        })
    }

    /// Transcript entries as JSON array.
    pub fn transcript_to_json(&self) -> Vec<serde_json::Value> {
        self.auto
            .lock()
            .map(|m| m.transcript.iter().map(|e| e.to_json()).collect())
            .unwrap_or_default()
    }

    /// Handle for custom metrics, passed to the Lua bridge.
    pub fn custom_metrics_handle(&self) -> CustomMetricsHandle {
        CustomMetricsHandle::new(Arc::clone(&self.custom))
    }

    /// Set session budget limits.
    pub fn set_budget(&self, budget: Budget) {
        if let Ok(mut m) = self.auto.lock() {
            m.budget = Some(budget);
        }
    }

    /// Create a budget handle for the Lua bridge to check limits.
    pub fn budget_handle(&self) -> BudgetHandle {
        BudgetHandle::new(Arc::clone(&self.auto))
    }

    /// Create a progress handle for the Lua bridge to report progress.
    pub fn progress_handle(&self) -> ProgressHandle {
        ProgressHandle::new(Arc::clone(&self.auto))
    }

    /// Lightweight snapshot for external observation (alc_status).
    ///
    /// Returns metrics without transcript by default; pass `include_history=true`
    /// to additionally include the last ≤10 conversation exchanges.
    ///
    /// # Arguments
    ///
    /// - `include_history` — When `true`, `conversation_history` (≤10 entries)
    ///   is included in the JSON output.  When `false` (default), the key is absent.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` snapshot, or `Value::Null` if the internal mutex
    /// is poisoned (only possible on OOM-induced panic — degraded but non-fatal).
    pub fn snapshot(&self, include_history: bool) -> serde_json::Value {
        self.auto
            .lock()
            .map(|m| m.snapshot(include_history, &self.log_sink))
            .unwrap_or(serde_json::Value::Null)
    }

    pub fn create_observer(&self) -> MetricsObserver {
        MetricsObserver::new(Arc::clone(&self.auto), self.log_sink.clone())
    }

    /// Return a cloned handle to the session's log-capture ring buffer.
    ///
    /// The returned [`LogSink`] shares the same underlying `Arc<Mutex<VecDeque>>`
    /// as the observer.  Pass this to the Lua bridge so that `print()` /
    /// `alc.log()` output is routed into the per-session ring buffer.
    ///
    /// # Returns
    ///
    /// A cloned [`LogSink`] that shares state with the observer's sink.
    pub fn log_sink_handle(&self) -> LogSink {
        self.log_sink.clone()
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl serde::Serialize for ExecutionMetrics {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(serializer)
    }
}

/// Updates SessionStatus via the ExecutionObserver trait.
pub struct MetricsObserver {
    auto: Arc<Mutex<SessionStatus>>,
    log_sink: LogSink,
}

impl MetricsObserver {
    pub(crate) fn new(auto: Arc<Mutex<SessionStatus>>, log_sink: LogSink) -> Self {
        Self { auto, log_sink }
    }
}

impl ExecutionObserver for MetricsObserver {
    fn on_paused(&self, queries: &[LlmQuery]) {
        // Safety: duration_since fails only if wall clock is before UNIX_EPOCH
        // (broken system clock). Saturating to zero is harmless for timestamps.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        if let Ok(mut m) = self.auto.lock() {
            m.pauses += 1;
            m.llm_calls += queries.len() as u64;
            for q in queries {
                m.total_prompt_chars += q.prompt.len() as u64;
                let mut est = estimate_tokens(&q.prompt);
                if let Some(ref sys) = q.system {
                    m.total_prompt_chars += sys.len() as u64;
                    est += estimate_tokens(sys);
                }
                m.transcript.push(TranscriptEntry {
                    query_id: q.id.as_str().to_string(),
                    prompt: q.prompt.clone(),
                    system: q.system.clone(),
                    response: None,
                    prompt_tokens: est,
                    prompt_source: TokenSource::Estimated,
                    response_tokens: 0,
                    response_source: TokenSource::Estimated,
                    started_at_ms: now_ms,
                    completed_at_ms: None,
                });
            }
        }
    }

    fn on_response_fed(
        &self,
        query_id: &QueryId,
        response: &str,
        usage: Option<&crate::TokenUsage>,
    ) {
        // Safety: duration_since fails only if wall clock is before UNIX_EPOCH
        // (broken system clock). Saturating to zero is harmless for timestamps.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        if let Ok(mut m) = self.auto.lock() {
            m.total_response_chars += response.len() as u64;

            if let Some(entry) = m
                .transcript
                .iter_mut()
                .rev()
                .find(|e| e.query_id == query_id.as_str())
            {
                entry.response = Some(response.to_string());
                entry.completed_at_ms = Some(now_ms);

                // Prompt tokens: upgrade to Provided if host reported them.
                if let Some(pt) = usage.and_then(|u| u.prompt_tokens) {
                    entry.prompt_tokens = pt;
                    entry.prompt_source = TokenSource::Provided;
                }

                // Response tokens: Provided if available, else Estimated.
                match usage.and_then(|u| u.completion_tokens) {
                    Some(ct) => {
                        entry.response_tokens = ct;
                        entry.response_source = TokenSource::Provided;
                    }
                    None => {
                        entry.response_tokens = estimate_tokens(response);
                        entry.response_source = TokenSource::Estimated;
                    }
                }
            }
        }
    }

    fn on_log(&self, entry: &LogEntry) {
        self.log_sink.push(entry.clone());
    }

    fn on_resumed(&self) {
        if let Ok(mut m) = self.auto.lock() {
            m.rounds += 1;
        }
    }

    fn on_completed(&self, _result: &serde_json::Value) {
        if let Ok(mut m) = self.auto.lock() {
            m.ended_at = Some(Instant::now());
        }
    }

    fn on_failed(&self, _error: &str) {
        if let Ok(mut m) = self.auto.lock() {
            m.ended_at = Some(Instant::now());
        }
    }

    fn on_cancelled(&self) {
        if let Ok(mut m) = self.auto.lock() {
            m.ended_at = Some(Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmQuery, QueryId};

    #[test]
    fn metrics_to_json_has_auto_and_custom() {
        let metrics = ExecutionMetrics::new();
        let json = metrics.to_json();
        assert!(json.get("auto").is_some());
        assert!(json.get("custom").is_some());
    }

    #[test]
    fn custom_handle_shares_state() {
        let metrics = ExecutionMetrics::new();
        let handle = metrics.custom_metrics_handle();

        handle.record("key".into(), serde_json::json!("value"));

        let json = metrics.to_json();
        let custom = json.get("custom").unwrap();
        assert_eq!(custom.get("key").unwrap(), "value");
    }

    #[test]
    fn observer_updates_auto_metrics() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let queries = vec![LlmQuery {
            id: QueryId::batch(0),
            prompt: "test".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }];

        observer.on_paused(&queries);
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let auto = json.get("auto").unwrap();
        assert_eq!(auto.get("llm_calls").unwrap(), 1);
        assert_eq!(auto.get("pauses").unwrap(), 1);
        assert_eq!(auto.get("rounds").unwrap(), 0);
        assert_eq!(auto.get("total_prompt_chars").unwrap(), 4); // "test" = 4 chars
        assert_eq!(auto.get("total_response_chars").unwrap(), 0);
    }

    #[test]
    fn observer_tracks_prompt_and_response_chars() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let queries = vec![
            LlmQuery {
                id: QueryId::batch(0),
                prompt: "hello".into(),     // 5 chars
                system: Some("sys".into()), // 3 chars
                max_tokens: 100,
                grounded: false,
                underspecified: false,
            },
            LlmQuery {
                id: QueryId::batch(1),
                prompt: "world".into(), // 5 chars
                system: None,
                max_tokens: 100,
                grounded: false,
                underspecified: false,
            },
        ];

        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::batch(0), &"x".repeat(42), None);
        observer.on_response_fed(&QueryId::batch(1), &"y".repeat(58), None);
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let auto = json.get("auto").unwrap();
        assert_eq!(auto.get("total_prompt_chars").unwrap(), 13); // 5+3+5
        assert_eq!(auto.get("total_response_chars").unwrap(), 100); // 42+58
        assert_eq!(auto.get("rounds").unwrap(), 1);
    }

    #[test]
    fn observer_tracks_multiple_rounds() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "p".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }];

        // Round 1
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), &"x".repeat(10), None);
        observer.on_resumed();
        // Round 2
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), &"y".repeat(20), None);
        observer.on_resumed();
        // Round 3
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), &"z".repeat(30), None);
        observer.on_resumed();

        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let auto = json.get("auto").unwrap();
        assert_eq!(auto.get("rounds").unwrap(), 3);
        assert_eq!(auto.get("pauses").unwrap(), 3);
        assert_eq!(auto.get("llm_calls").unwrap(), 3);
        assert_eq!(auto.get("total_prompt_chars").unwrap(), 3); // "p" x 3
        assert_eq!(auto.get("total_response_chars").unwrap(), 60); // 10+20+30
    }

    #[test]
    fn transcript_records_prompt_response_pairs() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let queries = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "What is 2+2?".into(),
            system: Some("You are a calculator.".into()),
            max_tokens: 50,
            grounded: false,
            underspecified: false,
        }];

        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::single(), "4", None);
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let transcript = metrics.transcript_to_json();
        assert_eq!(transcript.len(), 1);
        assert_eq!(transcript[0]["query_id"], "q-0");
        assert_eq!(transcript[0]["prompt"], "What is 2+2?");
        assert_eq!(transcript[0]["system"], "You are a calculator.");
        assert_eq!(transcript[0]["response"], "4");
    }

    #[test]
    fn transcript_not_in_stats() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "p".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), "r", None);
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        assert!(json["auto"].get("transcript").is_none());
    }

    #[test]
    fn transcript_multi_round() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        // Round 1
        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "step1".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), "answer1", None);
        observer.on_resumed();

        // Round 2
        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "step2".into(),
            system: Some("expert".into()),
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), "answer2", None);
        observer.on_resumed();

        observer.on_completed(&serde_json::json!(null));

        let transcript = metrics.transcript_to_json();
        assert_eq!(transcript.len(), 2);

        assert_eq!(transcript[0]["prompt"], "step1");
        assert!(transcript[0]["system"].is_null());
        assert_eq!(transcript[0]["response"], "answer1");

        assert_eq!(transcript[1]["prompt"], "step2");
        assert_eq!(transcript[1]["system"], "expert");
        assert_eq!(transcript[1]["response"], "answer2");
    }

    #[test]
    fn transcript_batch_queries() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let queries = vec![
            LlmQuery {
                id: QueryId::batch(0),
                prompt: "q0".into(),
                system: None,
                max_tokens: 50,
                grounded: false,
                underspecified: false,
            },
            LlmQuery {
                id: QueryId::batch(1),
                prompt: "q1".into(),
                system: None,
                max_tokens: 50,
                grounded: false,
                underspecified: false,
            },
        ];

        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::batch(0), "r0", None);
        observer.on_response_fed(&QueryId::batch(1), "r1", None);
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let transcript = metrics.transcript_to_json();
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[0]["query_id"], "q-0");
        assert_eq!(transcript[0]["response"], "r0");
        assert_eq!(transcript[1]["query_id"], "q-1");
        assert_eq!(transcript[1]["response"], "r1");
    }

    // ── v2 tests ────────────────────────────────────────────────

    // T1: on_log routes entries into the LogSink shared with metrics
    #[test]
    fn on_log_routes_to_log_sink() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        observer.on_log(&crate::LogEntry::new("info", "engine", "hello"));
        observer.on_log(&crate::LogEntry::new("warn", "alc.log", "world"));

        let sink = metrics.log_sink_handle();
        let entries = sink.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].level, "info");
        assert_eq!(entries[0].source, "engine");
        assert_eq!(entries[0].message, "hello");
        assert_eq!(entries[1].level, "warn");
        assert_eq!(entries[1].message, "world");
    }

    // T2: boundary — on_log cap=20 enforcement via observer
    #[test]
    fn on_log_cap_enforcement_via_observer() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        for i in 0..=20u32 {
            observer.on_log(&crate::LogEntry::new("info", "engine", format!("msg-{i}")));
        }

        let sink = metrics.log_sink_handle();
        let entries = sink.entries();
        assert_eq!(entries.len(), crate::recent_log::LOG_SINK_CAP);
        assert_eq!(entries[0].message, "msg-1");
        assert_eq!(
            entries[crate::recent_log::LOG_SINK_CAP - 1].message,
            "msg-20"
        );
    }

    // T1: on_paused records started_at_ms; on_response_fed sets completed_at_ms
    // Verified via snapshot(true) which projects TranscriptEntry timestamps into JSON.
    #[test]
    fn transcript_timestamps_recorded() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "ts-test".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);

        observer.on_response_fed(&QueryId::single(), "response", None);

        let after_fed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // Use snapshot(true) to expose timestamps from transcript projection.
        let snap = metrics.snapshot(true);
        let history = snap["conversation_history"]
            .as_array()
            .expect("conversation_history must be array");
        assert_eq!(history.len(), 1);

        let started_at = history[0]["started_at"]
            .as_i64()
            .expect("started_at must be i64");
        let completed_at = history[0]["completed_at"]
            .as_i64()
            .expect("completed_at must be i64 (not null)");

        assert!(
            started_at >= before,
            "started_at ({started_at}) should be >= before ({before})"
        );
        assert!(
            completed_at >= started_at,
            "completed_at ({completed_at}) should be >= started_at ({started_at})"
        );
        assert!(
            completed_at <= after_fed,
            "completed_at ({completed_at}) should be <= after_fed ({after_fed})"
        );
    }

    // T1: paused state shows current_query in snapshot (include_history=false)
    #[test]
    fn snapshot_current_query_while_paused() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "in-flight".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);

        // Snapshot without completing the response — last entry has response=None
        let snap = metrics.snapshot(false);

        let tokens = snap.get("tokens").expect("tokens field must be present");
        let current_query = tokens
            .get("current_query")
            .expect("current_query must be present");
        assert!(
            !current_query.is_null(),
            "current_query should be non-null while paused"
        );
        assert_eq!(current_query["query_id"], "q-0");
        // conversation_history must be absent with include_history=false
        assert!(
            snap.get("conversation_history").is_none(),
            "conversation_history must be absent when include_history=false"
        );
    }

    // T2: after response is fed, current_query becomes null
    #[test]
    fn snapshot_current_query_null_after_response() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "done".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), "answer", None);

        let snap = metrics.snapshot(false);
        let tokens = snap.get("tokens").expect("tokens must be present");
        let current_query = &tokens["current_query"];
        assert!(
            current_query.is_null(),
            "current_query should be null after response is fed"
        );
    }

    // T1/T3: conversation_history only when include_history=true
    #[test]
    fn snapshot_conversation_history_opt_in() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "hello".into(),
            system: None,
            max_tokens: 50,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), "world", None);
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        // false: conversation_history key must be absent
        let snap_false = metrics.snapshot(false);
        assert!(
            snap_false.get("conversation_history").is_none(),
            "conversation_history must be absent with include_history=false"
        );

        // true: conversation_history key must be present
        let snap_true = metrics.snapshot(true);
        let history = snap_true
            .get("conversation_history")
            .expect("conversation_history must be present with include_history=true");
        let arr = history
            .as_array()
            .expect("conversation_history must be an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["query_id"], "q-0");
        assert_eq!(arr[0]["prompt"], "hello");
        assert_eq!(arr[0]["response"], "world");
        // started_at and completed_at must be present
        assert!(arr[0].get("started_at").is_some());
        assert!(arr[0].get("completed_at").is_some());
    }

    // T2: conversation_history capped at 10 entries
    #[test]
    fn snapshot_conversation_history_capped_at_10() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        for i in 0..15u32 {
            observer.on_paused(&[LlmQuery {
                id: QueryId::single(),
                prompt: format!("prompt-{i}"),
                system: None,
                max_tokens: 10,
                grounded: false,
                underspecified: false,
            }]);
            observer.on_response_fed(&QueryId::single(), &format!("resp-{i}"), None);
            observer.on_resumed();
        }

        let snap = metrics.snapshot(true);
        let history = snap["conversation_history"]
            .as_array()
            .expect("must be array");
        assert_eq!(history.len(), 10, "capped at 10 entries");
        // Should be the last 10: prompt-5 through prompt-14
        assert_eq!(history[0]["prompt"], "prompt-5");
        assert_eq!(history[9]["prompt"], "prompt-14");
    }

    // T1: recent_logs appears in snapshot output
    #[test]
    fn snapshot_includes_recent_logs() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_log(&crate::LogEntry::new("info", "engine", "test-log"));

        let snap = metrics.snapshot(false);
        let logs = snap
            .get("recent_logs")
            .expect("recent_logs must be in snapshot");
        let arr = logs.as_array().expect("recent_logs must be array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["message"], "test-log");
    }

    // T1: tokens aggregate is correct in snapshot
    #[test]
    fn snapshot_tokens_aggregate() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "x".repeat(100),
            system: None,
            max_tokens: 50,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&QueryId::single(), &"y".repeat(50), None);
        observer.on_resumed();

        let snap = metrics.snapshot(false);
        let tokens = snap.get("tokens").expect("tokens must be in snapshot");
        let prompt_total = tokens["prompt_total"]
            .as_u64()
            .expect("prompt_total must be u64");
        let response_total = tokens["response_total"]
            .as_u64()
            .expect("response_total must be u64");
        let total = tokens["total"].as_u64().expect("total must be u64");
        // Estimates: 100 chars / 4 ≈ 25, 50 chars / 4 ≈ 12 (estimate_tokens rounding)
        assert!(prompt_total > 0, "prompt_total must be positive");
        assert!(response_total > 0, "response_total must be positive");
        assert_eq!(total, prompt_total + response_total);
    }
}
