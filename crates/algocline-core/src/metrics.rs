use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::budget::Budget;
use crate::observer::ExecutionObserver;
use crate::progress::ProgressInfo;
use crate::tokens::{estimate_tokens, TokenCount, TokenSource};
use crate::{BudgetHandle, CustomMetrics, CustomMetricsHandle, LlmQuery, ProgressHandle, QueryId};

// ─── Transcript ─────────────────────────────────────────────

/// A single prompt/response exchange in the transcript.
struct TranscriptEntry {
    query_id: String,
    prompt: String,
    system: Option<String>,
    response: Option<String>,
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
    prompt_tokens: TokenCount,
    response_tokens: TokenCount,
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
            prompt_tokens: TokenCount::new(TokenSource::Estimated),
            response_tokens: TokenCount::new(TokenSource::Estimated),
            transcript: Vec::new(),
            budget: None,
            progress: None,
        }
    }

    /// Wall-clock elapsed milliseconds since session start.
    fn elapsed_ms(&self) -> u64 {
        self.ended_at
            .map(|end| end.duration_since(self.started_at).as_millis() as u64)
            .unwrap_or_else(|| self.started_at.elapsed().as_millis() as u64)
    }

    fn to_json(&self) -> serde_json::Value {
        let total_tokens = TokenCount {
            tokens: self.prompt_tokens.tokens + self.response_tokens.tokens,
            source: self
                .prompt_tokens
                .source
                .weaker(self.response_tokens.source),
        };
        let mut json = serde_json::json!({
            "elapsed_ms": self.elapsed_ms(),
            "llm_calls": self.llm_calls,
            "pauses": self.pauses,
            "rounds": self.rounds,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars,
            "prompt_tokens": self.prompt_tokens.to_json(),
            "response_tokens": self.response_tokens.to_json(),
            "total_tokens": total_tokens.to_json(),
        });
        if let Some(ref b) = self.budget {
            json["budget"] = b.to_json();
        }
        json
    }

    pub(crate) fn check_budget(&self) -> Result<(), String> {
        match self.budget {
            Some(ref b) => b.check(self.llm_calls, self.elapsed_ms()),
            None => Ok(()),
        }
    }

    /// Lightweight snapshot for external observation (alc_status).
    ///
    /// Returns running metrics without transcript (which can be large).
    fn snapshot(&self) -> serde_json::Value {
        let mut json = serde_json::json!({
            "elapsed_ms": self.elapsed_ms(),
            "llm_calls": self.llm_calls,
            "rounds": self.rounds,
        });

        if let Some(ref p) = self.progress {
            json["progress"] = serde_json::json!({
                "step": p.step,
                "total": p.total,
                "message": p.message,
            });
        }

        if let Some(ref b) = self.budget {
            json["budget_remaining"] = b.remaining_json(self.llm_calls, self.elapsed_ms());
        }

        json
    }

    pub(crate) fn budget_remaining(&self) -> serde_json::Value {
        match self.budget {
            None => serde_json::Value::Null,
            Some(ref b) => b.remaining_json(self.llm_calls, self.elapsed_ms()),
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
pub struct ExecutionMetrics {
    auto: Arc<Mutex<SessionStatus>>,
    custom: Arc<Mutex<CustomMetrics>>,
}

impl ExecutionMetrics {
    pub fn new() -> Self {
        Self {
            auto: Arc::new(Mutex::new(SessionStatus::new())),
            custom: Arc::new(Mutex::new(CustomMetrics::new())),
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
    /// Returns metrics without transcript.
    pub fn snapshot(&self) -> serde_json::Value {
        self.auto
            .lock()
            .map(|m| m.snapshot())
            .unwrap_or(serde_json::Value::Null)
    }

    pub fn create_observer(&self) -> MetricsObserver {
        MetricsObserver::new(Arc::clone(&self.auto))
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
}

impl MetricsObserver {
    pub(crate) fn new(auto: Arc<Mutex<SessionStatus>>) -> Self {
        Self { auto }
    }
}

impl ExecutionObserver for MetricsObserver {
    fn on_paused(&self, queries: &[LlmQuery]) {
        if let Ok(mut m) = self.auto.lock() {
            m.pauses += 1;
            m.llm_calls += queries.len() as u64;
            for q in queries {
                m.total_prompt_chars += q.prompt.len() as u64;
                m.prompt_tokens
                    .accumulate(estimate_tokens(&q.prompt), TokenSource::Estimated);
                if let Some(ref sys) = q.system {
                    m.total_prompt_chars += sys.len() as u64;
                    m.prompt_tokens
                        .accumulate(estimate_tokens(sys), TokenSource::Estimated);
                }
                m.transcript.push(TranscriptEntry {
                    query_id: q.id.as_str().to_string(),
                    prompt: q.prompt.clone(),
                    system: q.system.clone(),
                    response: None,
                });
            }
        }
    }

    fn on_response_fed(&self, query_id: &QueryId, response: &str) {
        if let Ok(mut m) = self.auto.lock() {
            m.total_response_chars += response.len() as u64;
            m.response_tokens
                .accumulate(estimate_tokens(response), TokenSource::Estimated);
            // Fill response into matching transcript entry (last match for this query_id).
            if let Some(entry) = m
                .transcript
                .iter_mut()
                .rev()
                .find(|e| e.query_id == query_id.as_str())
            {
                entry.response = Some(response.to_string());
            }
        }
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
        observer.on_response_fed(&QueryId::batch(0), &"x".repeat(42));
        observer.on_response_fed(&QueryId::batch(1), &"y".repeat(58));
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
        observer.on_response_fed(&QueryId::single(), &"x".repeat(10));
        observer.on_resumed();
        // Round 2
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), &"y".repeat(20));
        observer.on_resumed();
        // Round 3
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), &"z".repeat(30));
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
        observer.on_response_fed(&QueryId::single(), "4");
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
        observer.on_response_fed(&QueryId::single(), "r");
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
        observer.on_response_fed(&QueryId::single(), "answer1");
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
        observer.on_response_fed(&QueryId::single(), "answer2");
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
        observer.on_response_fed(&QueryId::batch(0), "r0");
        observer.on_response_fed(&QueryId::batch(1), "r1");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let transcript = metrics.transcript_to_json();
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[0]["query_id"], "q-0");
        assert_eq!(transcript[0]["response"], "r0");
        assert_eq!(transcript[1]["query_id"], "q-1");
        assert_eq!(transcript[1]["response"], "r1");
    }
}
