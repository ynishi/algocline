use std::sync::{Arc, Mutex};

use crate::metrics::SessionStatus;

// ─── Budget ──────────────────────────────────────────────────

/// Session-level resource limits.
///
/// Extracted from `ctx.budget` at session start. When a limit is reached,
/// `alc.llm()` raises a catchable Lua error (`"budget_exceeded: ..."`)
/// **before** sending the request to the host. The check happens at
/// call-site, not after the LLM response arrives.
///
/// Budget is shared across the entire session — if `alc.pipe()` chains
/// multiple strategies, they all draw from the same budget.
#[derive(Debug, Clone, Default)]
pub struct Budget {
    /// Maximum number of LLM calls allowed in this session.
    /// Checked against `SessionStatus::llm_calls` (incremented in `on_paused`).
    pub max_llm_calls: Option<u64>,
    /// Maximum wall-clock time (ms) allowed for this session.
    /// Measured from session start (`Instant::now()` at construction).
    /// Note: this is wall-clock, not CPU time. Includes time spent
    /// waiting for host LLM responses.
    pub max_elapsed_ms: Option<u64>,
    /// Maximum total tokens (prompt + response) allowed in this session.
    /// Checked against accumulated `prompt_tokens + response_tokens`.
    /// Token counts may be estimated (±30%) or host-provided depending
    /// on `TokenSource`. Budget check uses whatever is available.
    pub max_tokens: Option<u64>,
}

impl Budget {
    /// Extract budget from ctx JSON. Returns None if no budget field present.
    pub fn from_ctx(ctx: &serde_json::Value) -> Option<Self> {
        let obj = ctx.as_object()?.get("budget")?.as_object()?;
        let max_llm_calls = obj.get("max_llm_calls").and_then(|v| v.as_u64());
        let max_elapsed_ms = obj.get("max_elapsed_ms").and_then(|v| v.as_u64());
        let max_tokens = obj.get("max_tokens").and_then(|v| v.as_u64());
        if max_llm_calls.is_none() && max_elapsed_ms.is_none() && max_tokens.is_none() {
            return None;
        }
        Some(Self {
            max_llm_calls,
            max_elapsed_ms,
            max_tokens,
        })
    }

    /// Check if the session is within budget given current counters.
    /// Returns Err with a structured message if any limit is exceeded.
    pub fn check(&self, llm_calls: u64, elapsed_ms: u64, total_tokens: u64) -> Result<(), String> {
        if let Some(max) = self.max_llm_calls {
            if llm_calls >= max {
                return Err(format!(
                    "budget_exceeded: max_llm_calls ({max}) reached ({llm_calls} used)"
                ));
            }
        }
        if let Some(max_ms) = self.max_elapsed_ms {
            if elapsed_ms >= max_ms {
                return Err(format!(
                    "budget_exceeded: max_elapsed_ms ({max_ms}ms) reached ({elapsed_ms}ms elapsed)"
                ));
            }
        }
        if let Some(max) = self.max_tokens {
            if total_tokens >= max {
                return Err(format!(
                    "budget_exceeded: max_tokens ({max}) reached ({total_tokens} used)"
                ));
            }
        }
        Ok(())
    }

    /// Remaining budget as JSON given current counters.
    /// Returns `{ llm_calls: N|null, elapsed_ms: N|null, tokens: N|null }`.
    pub fn remaining_json(
        &self,
        llm_calls: u64,
        elapsed_ms: u64,
        total_tokens: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "llm_calls": self.max_llm_calls.map(|max| max.saturating_sub(llm_calls)),
            "elapsed_ms": self.max_elapsed_ms.map(|max| max.saturating_sub(elapsed_ms)),
            "tokens": self.max_tokens.map(|max| max.saturating_sub(total_tokens)),
        })
    }

    /// Serialize budget limits to JSON (for stats output).
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if let Some(max) = self.max_llm_calls {
            map.insert("max_llm_calls".into(), max.into());
        }
        if let Some(max) = self.max_elapsed_ms {
            map.insert("max_elapsed_ms".into(), max.into());
        }
        if let Some(max) = self.max_tokens {
            map.insert("max_tokens".into(), max.into());
        }
        serde_json::Value::Object(map)
    }
}

/// Cheap, cloneable handle for budget checking from the Lua bridge.
///
/// Wraps the shared `SessionStatus` to expose only budget-related queries.
/// Passed to `bridge::register_llm()` where it gates every `alc.llm()`
/// and `alc.llm_batch()` call.
///
/// # Call site and threading
///
/// Both `check()` and `remaining()` are called exclusively from Lua
/// closures registered in `bridge.rs`, which run on the Lua OS thread.
/// They acquire `std::sync::Mutex<SessionStatus>` for a few microseconds
/// (read-only field comparisons). See `SessionStatus` doc for full
/// locking design.
///
/// # TOCTOU safety of Lua-side `alc.budget_check()`
///
/// The prelude's `alc.budget_check()` calls `remaining()` then the
/// caller decides whether to call `alc.llm()` (which calls `check()`).
/// Between `remaining()` release and `check()` acquire, `llm_calls`
/// could theoretically change — but within a single session this is
/// structurally impossible: the Lua thread is the only writer path
/// (via observer callbacks), and observer callbacks only fire after
/// the Lua thread yields control through the mpsc channel. Lua is
/// single-threaded and does not yield between `budget_check()` and
/// `alc.llm()`.
///
/// # Poison policy
///
/// `check()` propagates poison as `Err` — this surfaces as a Lua error,
/// which is the correct behavior since a poisoned mutex indicates an
/// unrecoverable state (OOM panic under lock). `remaining()` returns
/// `Null` on poison — it is observational and non-fatal.
#[derive(Clone)]
pub struct BudgetHandle {
    auto: Arc<Mutex<SessionStatus>>,
}

impl BudgetHandle {
    pub(crate) fn new(auto: Arc<Mutex<SessionStatus>>) -> Self {
        Self { auto }
    }

    /// Check if the session is within budget. Returns Err with a message if exceeded.
    ///
    /// Propagates mutex poison as `Err` (see BudgetHandle doc for rationale).
    pub fn check(&self) -> Result<(), String> {
        let m = self
            .auto
            .lock()
            .map_err(|_| "budget check: mutex poisoned".to_string())?;
        m.check_budget()
    }

    /// Remaining budget as JSON: `{ llm_calls: N|null, elapsed_ms: N|null }`.
    /// Returns `serde_json::Value::Null` if no budget is set.
    ///
    /// Returns `Null` on mutex poison (observational, non-fatal).
    pub fn remaining(&self) -> serde_json::Value {
        let m = match self.auto.lock() {
            Ok(m) => m,
            Err(_) => return serde_json::Value::Null,
        };
        m.budget_remaining()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionMetrics, ExecutionObserver, LlmQuery, QueryId};

    #[test]
    fn budget_from_ctx_none_when_missing() {
        let ctx = serde_json::json!({"task": "test"});
        assert!(Budget::from_ctx(&ctx).is_none());
    }

    #[test]
    fn budget_from_ctx_none_when_empty() {
        let ctx = serde_json::json!({"budget": {}});
        assert!(Budget::from_ctx(&ctx).is_none());
    }

    #[test]
    fn budget_from_ctx_extracts_llm_calls() {
        let ctx = serde_json::json!({"budget": {"max_llm_calls": 10}});
        let budget = Budget::from_ctx(&ctx).expect("should parse");
        assert_eq!(budget.max_llm_calls, Some(10));
        assert_eq!(budget.max_elapsed_ms, None);
    }

    #[test]
    fn budget_from_ctx_extracts_elapsed_ms() {
        let ctx = serde_json::json!({"budget": {"max_elapsed_ms": 5000}});
        let budget = Budget::from_ctx(&ctx).expect("should parse");
        assert_eq!(budget.max_llm_calls, None);
        assert_eq!(budget.max_elapsed_ms, Some(5000));
    }

    #[test]
    fn budget_from_ctx_extracts_both() {
        let ctx = serde_json::json!({"budget": {"max_llm_calls": 5, "max_elapsed_ms": 30000}});
        let budget = Budget::from_ctx(&ctx).expect("should parse");
        assert_eq!(budget.max_llm_calls, Some(5));
        assert_eq!(budget.max_elapsed_ms, Some(30000));
    }

    #[test]
    fn budget_check_passes_when_within_limits() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: Some(5),
            max_elapsed_ms: None,
            max_tokens: None,
        });
        let handle = metrics.budget_handle();
        assert!(handle.check().is_ok());
    }

    #[test]
    fn budget_check_fails_when_llm_calls_exceeded() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: Some(2),
            max_elapsed_ms: None,
            max_tokens: None,
        });
        let observer = metrics.create_observer();
        let handle = metrics.budget_handle();

        // Simulate 2 LLM calls
        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "p".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), "r", None);
        observer.on_resumed();
        observer.on_paused(&q);

        // Now at 2 calls, budget is max_llm_calls=2
        let result = handle.check();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("budget_exceeded"));
    }

    #[test]
    fn budget_check_fails_when_tokens_exceeded() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: None,
            max_elapsed_ms: None,
            max_tokens: Some(10),
        });
        let observer = metrics.create_observer();
        let handle = metrics.budget_handle();

        // Simulate an LLM call with a prompt that estimates to ≥10 tokens.
        // "abcdefghijklmnopqrstuvwxyz0123456789abcd" = 40 ASCII chars → ceil(40/4) = 10 tokens
        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "abcdefghijklmnopqrstuvwxyz0123456789abcd".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), "r", None);
        observer.on_resumed();

        let result = handle.check();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max_tokens"));
    }

    #[test]
    fn budget_check_passes_when_tokens_within_limit() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: None,
            max_elapsed_ms: None,
            max_tokens: Some(1000),
        });
        let observer = metrics.create_observer();
        let handle = metrics.budget_handle();

        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "short".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), "reply", None);
        observer.on_resumed();

        assert!(handle.check().is_ok());
    }

    #[test]
    fn budget_remaining_tracks_tokens() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: None,
            max_elapsed_ms: None,
            max_tokens: Some(100),
        });
        let observer = metrics.create_observer();
        let handle = metrics.budget_handle();

        // "test" = 4 chars → ceil(4/4) = 1 token prompt, "r" → 1 token response
        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "test".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&q);
        observer.on_response_fed(&QueryId::single(), "r", None);
        observer.on_resumed();

        let remaining = handle.remaining();
        // 100 - (1 prompt + 1 response) = 98
        assert_eq!(remaining["tokens"], 98);
    }

    #[test]
    fn budget_from_ctx_extracts_max_tokens() {
        let ctx = serde_json::json!({"budget": {"max_tokens": 5000}});
        let budget = Budget::from_ctx(&ctx).expect("should parse");
        assert_eq!(budget.max_llm_calls, None);
        assert_eq!(budget.max_elapsed_ms, None);
        assert_eq!(budget.max_tokens, Some(5000));
    }

    #[test]
    fn budget_remaining_null_when_no_budget() {
        let metrics = ExecutionMetrics::new();
        let handle = metrics.budget_handle();
        assert!(handle.remaining().is_null());
    }

    #[test]
    fn budget_remaining_tracks_llm_calls() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: Some(5),
            max_elapsed_ms: None,
            max_tokens: None,
        });
        let observer = metrics.create_observer();
        let handle = metrics.budget_handle();

        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "p".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&q);

        let remaining = handle.remaining();
        assert_eq!(remaining["llm_calls"], 4); // 5 - 1
    }

    #[test]
    fn budget_in_stats_json() {
        let metrics = ExecutionMetrics::new();
        metrics.set_budget(Budget {
            max_llm_calls: Some(10),
            max_elapsed_ms: Some(60000),
            max_tokens: None,
        });
        let observer = metrics.create_observer();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let budget = &json["auto"]["budget"];
        assert_eq!(budget["max_llm_calls"], 10);
        assert_eq!(budget["max_elapsed_ms"], 60000);
    }

    #[test]
    fn no_budget_in_stats_json_when_not_set() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        assert!(json["auto"].get("budget").is_none());
    }
}
