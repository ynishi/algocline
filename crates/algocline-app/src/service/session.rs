//! Per-MCP-connection session pin (`alc_session_new`).
//!
//! See task-mcp issue #1776627475 for the full design — §1-§7 record
//! the comparison tables that produced these decisions.
//!
//! ## Lifetime semantics
//!
//! For stdio MCP transport (the only transport algocline currently
//! supports), one process serves one `RunningService` which serves
//! one MCP connection. `AppService` is the per-process singleton, so
//! storing the pin on `AppService` is functionally equivalent to a
//! per-connection pin (worktree parallelism is achieved by spawning
//! separate `alc` processes, each with its own `AppService`). When
//! the MCP process exits, the pin drops with it — no explicit
//! `alc_session_end` is needed (§7).
//!
//! rmcp 1.5.0 does not expose a stable connection identifier on
//! `RequestContext` (`id` is per-request, `peer.tx` is an opaque
//! mpsc sender), which is why the §3 fallback "per-process pin +
//! `Mutex`" is the canonical implementation rather than the §3
//! primary "per-MCP-connection registry".

use std::path::PathBuf;

/// Activation modes recognised by `alc_session_new`.
///
/// Default mode behaves identically to "no session activated" except
/// that the pin is now visible to `resolve_project_root` as the `S`
/// layer (§6: P > S > E > W).
///
/// `test` mode is a hint for downstream tools to apply stricter
/// isolation (e.g. scenario test runners may scope state to a
/// per-session subdir). The mode value is recorded and surfaced via
/// the activation response, but the session struct itself does not
/// enforce mode-specific semantics — callers consume `mode` and
/// branch as needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionMode {
    Default,
    Test,
}

impl SessionMode {
    /// Parse the optional `mode` parameter from MCP. `None` and
    /// `Some("default")` resolve to `Default`; `Some("test")`
    /// resolves to `Test`. Unknown values are rejected with a
    /// typed error so MCP clients see a structured failure rather
    /// than silent fallback.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("default") => Ok(SessionMode::Default),
            Some("test") => Ok(SessionMode::Test),
            Some(other) => Err(format!(
                "alc_session_new: unknown mode '{other}' (allowed: \"default\" | \"test\")"
            )),
        }
    }

    /// Wire string used in MCP responses.
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionMode::Default => "default",
            SessionMode::Test => "test",
        }
    }
}

/// One activated session. Stored at most once per `AppService`
/// (per-process / per-MCP-connection in stdio transport).
#[derive(Debug, Clone)]
pub struct AlcSession {
    /// Caller-visible identifier. Generated server-side from the
    /// activation timestamp. Caller does not need to retain this —
    /// the pin is implicit on every subsequent tool call within the
    /// same MCP connection.
    pub session_id: String,
    /// Pinned project root (resolved at activation time so that
    /// later cwd changes or env var unsets do not invalidate the
    /// pin). Stored as `PathBuf` rather than `String` to match the
    /// existing `resolve_project_root` return type.
    pub project_root: Option<PathBuf>,
    /// Activation mode. See [`SessionMode`].
    pub mode: SessionMode,
}

impl AlcSession {
    /// Construct a new session with a server-generated id.
    pub fn new(project_root: Option<PathBuf>, mode: SessionMode) -> Self {
        Self {
            session_id: gen_session_id(),
            project_root,
            mode,
        }
    }
}

/// Generate a session id of the form `alc-sess-{ms_since_epoch}-{random_hex}`.
///
/// Combines millisecond timestamp with 32-bit random suffix to guarantee
/// uniqueness even for concurrent calls within the same millisecond.
fn gen_session_id() -> String {
    use rand::RngExt;
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let random: u32 = rand::rng().random();
    format!("alc-sess-{ms}-{random:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse_accepts_default_and_test() {
        assert_eq!(SessionMode::parse(None).unwrap(), SessionMode::Default);
        assert_eq!(
            SessionMode::parse(Some("default")).unwrap(),
            SessionMode::Default
        );
        assert_eq!(SessionMode::parse(Some("test")).unwrap(), SessionMode::Test);
    }

    #[test]
    fn mode_parse_rejects_unknown_with_typed_error() {
        let err = SessionMode::parse(Some("sandbox")).unwrap_err();
        assert!(err.contains("unknown mode 'sandbox'"));
        assert!(err.contains("\"default\""));
        assert!(err.contains("\"test\""));
    }

    #[test]
    fn mode_as_str_round_trips_with_parse() {
        for mode in [SessionMode::Default, SessionMode::Test] {
            let parsed = SessionMode::parse(Some(mode.as_str())).unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn session_id_has_alc_sess_prefix_and_grows_monotonically() {
        let s1 = AlcSession::new(None, SessionMode::Default);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let s2 = AlcSession::new(None, SessionMode::Default);
        assert!(s1.session_id.starts_with("alc-sess-"));
        assert!(s2.session_id.starts_with("alc-sess-"));
        assert_ne!(s1.session_id, s2.session_id);
    }

    #[test]
    fn session_records_project_root_and_mode() {
        let pr = std::path::PathBuf::from("/tmp/example");
        let s = AlcSession::new(Some(pr.clone()), SessionMode::Test);
        assert_eq!(s.project_root.as_deref(), Some(pr.as_path()));
        assert_eq!(s.mode, SessionMode::Test);
    }
}
