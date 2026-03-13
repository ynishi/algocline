use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::observer::ExecutionObserver;
use crate::{CustomMetrics, LlmQuery, QueryId};

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
pub(crate) struct AutoMetrics {
    started_at: Instant,
    ended_at: Option<Instant>,
    llm_calls: u64,
    pauses: u64,
    rounds: u64,
    total_prompt_chars: u64,
    total_response_chars: u64,
    transcript: Vec<TranscriptEntry>,
}

impl AutoMetrics {
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
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let elapsed_ms = self
            .ended_at
            .map(|end| end.duration_since(self.started_at).as_millis() as u64)
            .unwrap_or_else(|| self.started_at.elapsed().as_millis() as u64);

        let transcript: Vec<serde_json::Value> =
            self.transcript.iter().map(|e| e.to_json()).collect();

        serde_json::json!({
            "elapsed_ms": elapsed_ms,
            "llm_calls": self.llm_calls,
            "pauses": self.pauses,
            "rounds": self.rounds,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars,
            "transcript": transcript,
        })
    }
}

/// Measurement data for a single execution.
pub struct ExecutionMetrics {
    auto: Arc<Mutex<AutoMetrics>>,
    custom: Arc<Mutex<CustomMetrics>>,
}

impl ExecutionMetrics {
    pub fn new() -> Self {
        Self {
            auto: Arc::new(Mutex::new(AutoMetrics::new())),
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

    /// Handle for custom metrics, passed to the Lua bridge.
    pub fn custom_handle(&self) -> Arc<Mutex<CustomMetrics>> {
        Arc::clone(&self.custom)
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

/// Updates AutoMetrics via the ExecutionObserver trait.
pub struct MetricsObserver {
    auto: Arc<Mutex<AutoMetrics>>,
}

impl MetricsObserver {
    pub(crate) fn new(auto: Arc<Mutex<AutoMetrics>>) -> Self {
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
                if let Some(ref sys) = q.system {
                    m.total_prompt_chars += sys.len() as u64;
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
        let handle = metrics.custom_handle();

        handle
            .lock()
            .unwrap()
            .record("key".into(), serde_json::json!("value"));

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
            },
            LlmQuery {
                id: QueryId::batch(1),
                prompt: "world".into(), // 5 chars
                system: None,
                max_tokens: 100,
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
        }];

        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::single(), "4");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let transcript = json["auto"]["transcript"].as_array().unwrap();
        assert_eq!(transcript.len(), 1);
        assert_eq!(transcript[0]["query_id"], "q-0");
        assert_eq!(transcript[0]["prompt"], "What is 2+2?");
        assert_eq!(transcript[0]["system"], "You are a calculator.");
        assert_eq!(transcript[0]["response"], "4");
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
        }]);
        observer.on_response_fed(&QueryId::single(), "answer1");
        observer.on_resumed();

        // Round 2
        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "step2".into(),
            system: Some("expert".into()),
            max_tokens: 100,
        }]);
        observer.on_response_fed(&QueryId::single(), "answer2");
        observer.on_resumed();

        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let transcript = json["auto"]["transcript"].as_array().unwrap();
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
            },
            LlmQuery {
                id: QueryId::batch(1),
                prompt: "q1".into(),
                system: None,
                max_tokens: 50,
            },
        ];

        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::batch(0), "r0");
        observer.on_response_fed(&QueryId::batch(1), "r1");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let transcript = json["auto"]["transcript"].as_array().unwrap();
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[0]["query_id"], "q-0");
        assert_eq!(transcript[0]["response"], "r0");
        assert_eq!(transcript[1]["query_id"], "q-1");
        assert_eq!(transcript[1]["response"], "r1");
    }
}
