use serde::{Deserialize, Serialize};
use std::fmt;

/// Query identifier within a batch.
///
/// Use `single()` for alc.llm(), `batch(index)` for alc.llm_batch().
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(String);

impl QueryId {
    /// For single alc.llm() calls.
    pub fn single() -> Self {
        Self("q-0".into())
    }

    /// For alc.llm_batch() with the given index.
    pub fn batch(index: usize) -> Self {
        Self(format!("q-{index}"))
    }

    /// Construct from an arbitrary string (e.g. MCP parameters).
    pub fn parse(s: &str) -> Self {
        Self(s.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// LLM request emitted during execution.
/// Transport-agnostic (no channel, HTTP, or MCP Sampling details).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmQuery {
    pub id: QueryId,
    pub prompt: String,
    pub system: Option<String>,
    pub max_tokens: u32,
    /// When true, the host should ground the response in external evidence
    /// (web search, code reading, documentation, etc.) rather than relying
    /// solely on LLM internal knowledge. The host decides the means.
    #[serde(default, skip_serializing_if = "is_false")]
    pub grounded: bool,
}

fn is_false(v: &bool) -> bool {
    !v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_query_id() {
        let id = QueryId::single();
        assert_eq!(id.as_str(), "q-0");
        assert_eq!(id.to_string(), "q-0");
    }

    #[test]
    fn batch_query_ids_are_unique() {
        let ids: Vec<QueryId> = (0..5).map(QueryId::batch).collect();
        let set: std::collections::HashSet<&QueryId> = ids.iter().collect();
        assert_eq!(set.len(), 5);
        assert_eq!(ids[0].as_str(), "q-0");
        assert_eq!(ids[3].as_str(), "q-3");
    }

    #[test]
    fn single_equals_batch_zero() {
        assert_eq!(QueryId::single(), QueryId::batch(0));
    }

    #[test]
    fn parse_roundtrip() {
        let id = QueryId::parse("q-42");
        assert_eq!(id.as_str(), "q-42");
        assert_eq!(id, QueryId::batch(42));
    }

    #[test]
    fn parse_arbitrary() {
        let id = QueryId::parse("custom-id");
        assert_eq!(id.as_str(), "custom-id");
    }

    #[test]
    fn query_id_roundtrip_json() {
        let id = QueryId::batch(42);
        let json = serde_json::to_string(&id).unwrap();
        let restored: QueryId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn llm_query_roundtrip_json() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "test prompt".into(),
            system: Some("system".into()),
            max_tokens: 1024,
            grounded: false,
        };
        let json = serde_json::to_value(&query).unwrap();
        assert!(
            json.get("grounded").is_none(),
            "grounded key must be absent when false (skip_serializing_if)"
        );
        let restored: LlmQuery = serde_json::from_value(json).unwrap();
        assert_eq!(restored.id, query.id);
        assert_eq!(restored.prompt, query.prompt);
        assert_eq!(restored.system, query.system);
        assert_eq!(restored.max_tokens, query.max_tokens);
        assert!(!restored.grounded);
    }

    #[test]
    fn llm_query_grounded_serde() {
        let query = LlmQuery {
            id: QueryId::single(),
            prompt: "verify this".into(),
            system: None,
            max_tokens: 200,
            grounded: true,
        };
        let json = serde_json::to_value(&query).unwrap();
        assert_eq!(
            json["grounded"], true,
            "grounded key must be present when true"
        );
        let restored: LlmQuery = serde_json::from_value(json).unwrap();
        assert!(restored.grounded);
    }

    #[test]
    fn llm_query_grounded_default_on_missing_key() {
        let json = serde_json::json!({
            "id": "q-single",
            "prompt": "test",
            "system": null,
            "max_tokens": 100
        });
        let query: LlmQuery = serde_json::from_value(json).unwrap();
        assert!(
            !query.grounded,
            "grounded must default to false when key absent"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn parse_roundtrip_arbitrary(s in "\\PC{1,100}") {
            let id = QueryId::parse(&s);
            prop_assert_eq!(id.as_str(), s.as_str());
        }

        #[test]
        fn batch_roundtrip(index in 0usize..10_000) {
            let id = QueryId::batch(index);
            let expected = format!("q-{index}");
            prop_assert_eq!(id.as_str(), expected.as_str());
        }

        #[test]
        fn display_matches_as_str(s in "\\PC{1,100}") {
            let id = QueryId::parse(&s);
            prop_assert_eq!(id.to_string(), id.as_str().to_string());
        }

        #[test]
        fn serde_roundtrip_arbitrary(s in "\\PC{1,100}") {
            let id = QueryId::parse(&s);
            let json = serde_json::to_string(&id).unwrap();
            let restored: QueryId = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(id, restored);
        }
    }
}
