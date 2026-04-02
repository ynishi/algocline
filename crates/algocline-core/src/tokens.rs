// ─── Token usage (host-provided) ─────────────────────────────

/// Token usage reported by the host LLM alongside its response.
///
/// When the host includes usage data in `alc_continue`, these counts
/// replace the character-based estimates for that specific response,
/// upgrading `TokenSource` from `Estimated` to `Provided`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens consumed by this LLM call, as reported by the host.
    pub prompt_tokens: Option<u64>,
    /// Completion (response) tokens produced by this LLM call.
    pub completion_tokens: Option<u64>,
}

// ─── Token tracking ─────────────────────────────────────────

/// How a token count was obtained.
///
/// When a session mixes sources (e.g. some calls estimated, some provided),
/// the aggregate source degrades to the weakest (least precise) variant
/// via [`TokenSource::weaker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    /// Character-based heuristic (ASCII ~4c/t, CJK ~1.5c/t). ±30% accuracy.
    Estimated,
    /// Reported by the host (e.g. MCP Sampling `usage` metadata).
    /// Accuracy depends on the host's tokenizer.
    Provided,
    /// Exact count from a known tokenizer (e.g. local BPE).
    Definite,
}

impl TokenSource {
    /// Return the weaker (less precise) of two sources.
    ///
    /// Used when accumulating across multiple LLM calls in a session.
    /// If any call is `Estimated`, the aggregate is `Estimated`.
    pub fn weaker(self, other: Self) -> Self {
        match (self, other) {
            (Self::Estimated, _) | (_, Self::Estimated) => Self::Estimated,
            (Self::Provided, _) | (_, Self::Provided) => Self::Provided,
            _ => Self::Definite,
        }
    }
}

/// Accumulated token count with provenance.
///
/// Tracks both the total token count and the weakest [`TokenSource`]
/// across all accumulated calls. This lets consumers (e.g. `alc_eval_compare`)
/// know whether a comparison is between precise or estimated values.
#[derive(Debug, Clone)]
pub struct TokenCount {
    pub tokens: u64,
    pub source: TokenSource,
}

impl TokenCount {
    /// New zero-count with the given source.
    pub(crate) fn new(source: TokenSource) -> Self {
        Self { tokens: 0, source }
    }

    /// Add tokens, degrading source to the weaker of the two.
    pub(crate) fn accumulate(&mut self, tokens: u64, source: TokenSource) {
        self.tokens += tokens;
        self.source = self.source.weaker(source);
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "tokens": self.tokens,
            "source": self.source,
        })
    }
}

/// Estimate token count from a string using a character-based heuristic.
///
/// For mixed-language text (English + CJK), we use a blended approach:
/// - ASCII characters: ~4 chars per token (GPT/Claude typical)
/// - Non-ASCII characters (CJK, etc.): ~1.5 chars per token
///
/// **Accuracy**: This is an order-of-magnitude estimate. Actual token counts
/// depend on the model's tokenizer (BPE). Expect ±30% deviation for typical
/// English text, potentially more for code or heavily structured text.
/// Intended for cost trend analysis (eval comparison), not billing.
pub(crate) fn estimate_tokens(text: &str) -> u64 {
    let mut ascii_chars: u64 = 0;
    let mut non_ascii_chars: u64 = 0;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_chars += 1;
        } else {
            non_ascii_chars += 1;
        }
    }
    // ASCII: ~4 chars/token, Non-ASCII: ~1.5 chars/token
    let ascii_tokens = ascii_chars.div_ceil(4);
    let non_ascii_tokens = (non_ascii_chars * 2).div_ceil(3);
    ascii_tokens + non_ascii_tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionMetrics, ExecutionObserver, LlmQuery, QueryId};

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_ascii() {
        // "hello world" = 11 ASCII chars → ceil(11/4) = 3
        assert_eq!(estimate_tokens("hello world"), 3);
    }

    #[test]
    fn token_source_weaker_estimated_wins() {
        assert_eq!(
            TokenSource::Estimated.weaker(TokenSource::Definite),
            TokenSource::Estimated
        );
        assert_eq!(
            TokenSource::Definite.weaker(TokenSource::Estimated),
            TokenSource::Estimated
        );
    }

    #[test]
    fn token_source_weaker_provided_over_definite() {
        assert_eq!(
            TokenSource::Provided.weaker(TokenSource::Definite),
            TokenSource::Provided
        );
    }

    #[test]
    fn token_source_weaker_same_returns_same() {
        assert_eq!(
            TokenSource::Definite.weaker(TokenSource::Definite),
            TokenSource::Definite
        );
        assert_eq!(
            TokenSource::Estimated.weaker(TokenSource::Estimated),
            TokenSource::Estimated
        );
    }

    #[test]
    fn token_count_accumulate_degrades_source() {
        let mut tc = TokenCount::new(TokenSource::Definite);
        tc.accumulate(10, TokenSource::Definite);
        assert_eq!(tc.source, TokenSource::Definite);

        tc.accumulate(5, TokenSource::Provided);
        assert_eq!(tc.tokens, 15);
        assert_eq!(tc.source, TokenSource::Provided);

        tc.accumulate(3, TokenSource::Estimated);
        assert_eq!(tc.tokens, 18);
        assert_eq!(tc.source, TokenSource::Estimated);
    }

    #[test]
    fn token_count_to_json_format() {
        let tc = TokenCount {
            tokens: 42,
            source: TokenSource::Provided,
        };
        let json = tc.to_json();
        assert_eq!(json["tokens"], 42);
        assert_eq!(json["source"], "provided");
    }

    #[test]
    fn token_source_serde_roundtrip() {
        let source = TokenSource::Estimated;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, r#""estimated""#);
        let restored: TokenSource = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, source);
    }

    #[test]
    fn estimate_tokens_cjk() {
        // "あいう" = 3 non-ASCII chars → ceil(3/1.5) = ceil(6/3) = 2
        assert_eq!(estimate_tokens("あいう"), 2);
    }

    #[test]
    fn estimate_tokens_mixed() {
        // "hello あ" = 6 ASCII + 1 non-ASCII
        // ASCII: ceil(6/4) = 2, CJK: ceil(1/1.5) = ceil(2/3) = 1
        assert_eq!(estimate_tokens("hello あ"), 3);
    }

    #[test]
    fn token_estimation_in_stats() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let queries = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "What is 2+2?".into(), // 12 ASCII → ceil(12/4) = 3
            system: Some("Expert".into()), // 6 ASCII → ceil(6/4) = 2
            max_tokens: 50,
            grounded: false,
            underspecified: false,
        }];
        observer.on_paused(&queries);
        observer.on_response_fed(&QueryId::single(), "4", None); // 1 ASCII → ceil(1/4) = 1
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let auto = &json["auto"];
        assert_eq!(auto["prompt_tokens"]["tokens"], 5); // 3 + 2
        assert_eq!(auto["prompt_tokens"]["source"], "estimated");
        assert_eq!(auto["response_tokens"]["tokens"], 1);
        assert_eq!(auto["response_tokens"]["source"], "estimated");
        assert_eq!(auto["total_tokens"]["tokens"], 6);
        assert_eq!(auto["total_tokens"]["source"], "estimated");
    }

    #[test]
    fn token_estimation_accumulates_across_rounds() {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let q = vec![LlmQuery {
            id: QueryId::single(),
            prompt: "test".into(), // 4 ASCII → ceil(4/4) = 1
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }];

        // 3 rounds
        for _ in 0..3 {
            observer.on_paused(&q);
            observer.on_response_fed(&QueryId::single(), "reply here", None); // 10 → ceil(10/4) = 3
            observer.on_resumed();
        }
        observer.on_completed(&serde_json::json!(null));

        let json = metrics.to_json();
        let auto = &json["auto"];
        assert_eq!(auto["prompt_tokens"]["tokens"], 3); // 1 * 3
        assert_eq!(auto["prompt_tokens"]["source"], "estimated");
        assert_eq!(auto["response_tokens"]["tokens"], 9); // 3 * 3
        assert_eq!(auto["response_tokens"]["source"], "estimated");
    }
}
