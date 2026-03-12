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
}

impl AutoMetrics {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            ended_at: None,
            llm_calls: 0,
            pauses: 0,
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
    }
}
