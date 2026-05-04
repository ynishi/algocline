use std::path::PathBuf;

use algocline_core::TokenUsage;
use serde::{Deserialize, Serialize};

// ─── Request ─────────────────────────────────────────────────────────────────

/// A message sent from the MCP process (pool client) to a pool worker.
///
/// Serialised as a single JSON line with an `"op"` discriminant field.
///
/// ```json
/// {"op":"handshake","version":"0.31.0"}
/// {"op":"run","code":"return 1","ctx":null,"lib_paths":[]}
/// {"op":"continue","sid":"abc","response":"yes","query_id":"q1","usage":null}
/// {"op":"status"}
/// {"op":"shutdown"}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PoolRequest {
    /// Version negotiation; must be the first message sent.
    Handshake {
        /// Crate version of the client (e.g. `env!("CARGO_PKG_VERSION")`).
        version: String,
    },

    /// Start a new Lua execution session.
    Run {
        /// Lua source code to execute.
        code: String,
        /// Optional JSON context passed as `alc.ctx`.
        ctx: Option<serde_json::Value>,
        /// Additional Lua library search paths.
        lib_paths: Vec<PathBuf>,
    },

    /// Resume a paused session with the LLM response.
    Continue {
        /// Session ID originally returned in the `Run` response.
        sid: String,
        /// LLM response text.
        response: String,
        /// The specific query being answered (if multi-query).
        query_id: Option<String>,
        /// Token usage reported by the host alongside this response.
        usage: Option<TokenUsage>,
    },

    /// Query the worker's current health/state (read-only).
    Status,

    /// Ask the worker to finish gracefully and exit.
    Shutdown,
}

// ─── Response ────────────────────────────────────────────────────────────────

/// Data payload carried by a successful [`PoolResponse`].
///
/// `FeedResult`-equivalent payloads are represented as `serde_json::Value` so
/// that the `pool` protocol module does not introduce a hard dependency on the
/// `algocline-engine` crate's concrete type.  Subtask 5 (AppService dispatch)
/// will deserialize the value into the appropriate engine type at the call site.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PoolResponseData {
    /// Reply to a `Handshake` request.
    Handshake {
        /// Crate version of the worker process.
        version: String,
    },

    /// Reply to a `Run` or `Continue` request.
    Feed {
        /// The session ID assigned or continued by the worker.
        session_id: String,
        /// JSON-serialised `FeedResult` (Accepted / Paused / Finished).
        feed_result: serde_json::Value,
    },

    /// Reply to a `Status` request.
    Status {
        /// Whether the worker currently has an active session.
        has_session: bool,
        /// Session ID of the active session, if any.
        session_id: Option<String>,
    },

    /// Reply to a `Shutdown` request (worker will exit after sending this).
    Shutdown,
}

/// A message sent from a pool worker back to the pool client.
///
/// Serialised as a single JSON line:
///
/// ```json
/// {"ok":true,"data":{"kind":"handshake","version":"0.31.0"}}
/// {"ok":false,"error":"worker handshake failed: version mismatch"}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolResponse {
    /// `true` if the operation succeeded.
    pub ok: bool,
    /// Present when `ok` is `true`.
    pub data: Option<PoolResponseData>,
    /// Present when `ok` is `false`.
    pub error: Option<String>,
}

impl PoolResponse {
    /// Construct a successful response.
    pub fn success(data: PoolResponseData) -> Self {
        Self {
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    /// Construct an error response.
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(error.into()),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a `Handshake` request round-trips through JSON without loss.
    ///
    /// The JSON line format must preserve the `"op"` discriminant so the worker
    /// can dispatch on it without additional framing.
    #[test]
    fn handshake_request_roundtrip() {
        let req = PoolRequest::Handshake {
            version: "0.31.0".to_string(),
        };

        let json = serde_json::to_string(&req).expect("serialize");
        // Must be a single line (no embedded newlines).
        assert!(!json.contains('\n'), "JSON line must not contain newlines");
        // Must carry the op discriminant.
        assert!(json.contains("\"op\":\"handshake\""), "op field missing");
        // Must carry the version field.
        assert!(
            json.contains("\"version\":\"0.31.0\""),
            "version field missing"
        );

        let decoded: PoolRequest = serde_json::from_str(&json).expect("deserialize");
        match decoded {
            PoolRequest::Handshake { version } => {
                assert_eq!(version, "0.31.0");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// Verify that a successful `PoolResponse` round-trips through JSON.
    #[test]
    fn response_success_roundtrip() {
        let resp = PoolResponse::success(PoolResponseData::Handshake {
            version: "0.31.0".to_string(),
        });

        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"ok\":true"), "ok flag missing");

        let decoded: PoolResponse = serde_json::from_str(&json).expect("deserialize");
        assert!(decoded.ok);
        assert!(decoded.error.is_none());
        match decoded.data {
            Some(PoolResponseData::Handshake { version }) => {
                assert_eq!(version, "0.31.0");
            }
            other => panic!("unexpected data: {other:?}"),
        }
    }

    /// Verify that an error `PoolResponse` round-trips through JSON.
    #[test]
    fn response_failure_roundtrip() {
        let resp = PoolResponse::failure("version mismatch");

        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"ok\":false"), "ok flag missing");

        let decoded: PoolResponse = serde_json::from_str(&json).expect("deserialize");
        assert!(!decoded.ok);
        assert!(decoded.data.is_none());
        assert_eq!(decoded.error.as_deref(), Some("version mismatch"));
    }
}
