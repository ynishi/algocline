//! Session identifier type for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};

/// Opaque handle that uniquely identifies an execution session across all verb calls.
///
/// `SessionId` is a newtype over [`String`] and is the sole cross-boundary handle
/// needed to address a session.  Generation is the responsibility of the engine layer;
/// the core value layer only provides construction and inspection primitives.
///
/// # Serde
/// Serializes as a plain JSON string (transparent newtype) so that wire consumers do
/// not need to unwrap a wrapper object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Wraps an existing string as a `SessionId`.
    pub fn new(s: String) -> Self {
        Self(s)
    }

    /// Returns the inner string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_serde_string_form() {
        // SessionId must serialize as a bare JSON string, not as {"0": "…"}.
        let id = SessionId::new("01HX1234567890ABCDEFGHJKMP".to_owned());
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, r#""01HX1234567890ABCDEFGHJKMP""#);

        let roundtripped: SessionId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtripped, id);
    }

    #[test]
    fn session_id_display() {
        let id = SessionId::new("abc123".to_owned());
        assert_eq!(id.to_string(), "abc123");
    }
}
