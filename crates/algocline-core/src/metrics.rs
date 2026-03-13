use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::observer::ExecutionObserver;
use crate::{CustomMetrics, LlmQuery};

/// Metrics automatically derived from the execution lifecycle.
pub(crate) struct AutoMetrics {
    started_at: Instant,
    ended_at: Option<Instant>,
    llm_calls: u64,
    pauses: u64,
    rounds: u64,
    total_prompt_chars: u64,
    total_response_chars: u64,
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
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let elapsed_ms = self
            .ended_at
            .map(|end| end.duration_since(self.started_at).as_millis() as u64)
            .unwrap_or_else(|| self.started_at.elapsed().as_millis() as u64);

        serde_json::json!({
            "elapsed_ms": elapsed_ms,
            "llm_calls": self.llm_calls,
            "pauses": self.pauses,
            "rounds": self.rounds,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars,
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
            }
        }
    }

    fn on_response_fed(&self, response_chars: u64) {
        if let Ok(mut m) = self.auto.lock() {
            m.total_response_chars += response_chars;
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
        observer.on_response_fed(42);
        observer.on_response_fed(58);
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
        observer.on_response_fed(10);
        observer.on_resumed();
        // Round 2
        observer.on_paused(&q);
        observer.on_response_fed(20);
        observer.on_resumed();
        // Round 3
        observer.on_paused(&q);
        observer.on_response_fed(30);
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
}
