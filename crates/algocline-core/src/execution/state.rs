//! Execution state types (v2) for the `ExecutionService` layer.
//!
//! This module defines [`ExecutionState`] v2, [`ExecutionStateTag`], and [`ExecutionResult`].
//! These types are **distinct** from the legacy `state::ExecutionState` in
//! `crates/algocline-core/src/state.rs`; both coexist while the migration to the new
//! `ExecutionService` API is in progress.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::cancel::{CancelInfo, FailureInfo};
use super::pause::PauseInfo;
use crate::TokenUsage;

/// Rich execution state used by [`crate::execution::ExecutionService`].
///
/// This is the v2 variant; the legacy `crate::ExecutionState` from `state.rs` remains
/// untouched for backward compatibility with existing engine consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExecutionState {
    /// The session is actively executing.
    Running,
    /// The session is paused, waiting for one or more LLM responses.
    Paused(PauseInfo),
    /// The session completed successfully.
    Done(ExecutionResult),
    /// The session was cancelled cooperatively.
    Cancelled(CancelInfo),
    /// The session ended with an error.
    Failed(FailureInfo),
}

impl ExecutionState {
    /// Returns the lightweight tag for this state variant.
    pub fn tag(&self) -> ExecutionStateTag {
        ExecutionStateTag::from(self)
    }

    /// Returns `true` if this state is a terminal (non-resumable) state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done(_) | Self::Cancelled(_) | Self::Failed(_))
    }
}

/// Lightweight discriminant for [`ExecutionState`].
///
/// Used in [`crate::execution::ProgressEvent::StateTransition`] to avoid carrying
/// the full state payload in every event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStateTag {
    /// Corresponds to [`ExecutionState::Running`].
    Running,
    /// Corresponds to [`ExecutionState::Paused`].
    Paused,
    /// Corresponds to [`ExecutionState::Done`].
    Done,
    /// Corresponds to [`ExecutionState::Cancelled`].
    Cancelled,
    /// Corresponds to [`ExecutionState::Failed`].
    Failed,
}

impl From<&ExecutionState> for ExecutionStateTag {
    /// Extracts the discriminant tag from a state reference without cloning the payload.
    fn from(state: &ExecutionState) -> Self {
        match state {
            ExecutionState::Running => Self::Running,
            ExecutionState::Paused(_) => Self::Paused,
            ExecutionState::Done(_) => Self::Done,
            ExecutionState::Cancelled(_) => Self::Cancelled,
            ExecutionState::Failed(_) => Self::Failed,
        }
    }
}

/// Payload carried by [`ExecutionState::Done`].
///
/// `finished_at` is stored as a Unix timestamp in milliseconds to ensure
/// lossless serde without additional crate dependencies (the standard library
/// `SystemTime` has no built-in serde support).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The JSON value returned by the Lua execution.
    pub value: JsonValue,
    /// Aggregate token usage for the session, if available.
    pub usage: Option<TokenUsage>,
    /// Unix timestamp (milliseconds) when the session finished.
    pub finished_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::cancel::{CancelCode, CancelInfo, CancelReason};
    use crate::execution::pause::{PauseInfo, PauseKind};

    fn make_pause_info() -> PauseInfo {
        PauseInfo {
            kind: PauseKind::Single,
            prompts: vec![],
            paused_at: 0,
        }
    }

    fn make_cancel_info() -> CancelInfo {
        CancelInfo {
            reason: CancelReason {
                code: CancelCode::User,
                detail: None,
                requested_at: 0,
            },
            observed_at: 0,
            state_before: Box::new(ExecutionState::Running),
        }
    }

    fn make_failure_info() -> FailureInfo {
        crate::execution::cancel::FailureInfo {
            message: "test".into(),
            kind: crate::execution::cancel::FailureKind::Other,
            occurred_at: 0,
        }
    }

    #[test]
    fn execution_state_tag_from_state() {
        assert_eq!(
            ExecutionStateTag::from(&ExecutionState::Running),
            ExecutionStateTag::Running
        );
        assert_eq!(
            ExecutionStateTag::from(&ExecutionState::Paused(make_pause_info())),
            ExecutionStateTag::Paused
        );
        assert_eq!(
            ExecutionStateTag::from(&ExecutionState::Done(ExecutionResult {
                value: serde_json::Value::Null,
                usage: None,
                finished_at: 0,
            })),
            ExecutionStateTag::Done
        );
        assert_eq!(
            ExecutionStateTag::from(&ExecutionState::Cancelled(make_cancel_info())),
            ExecutionStateTag::Cancelled
        );
        assert_eq!(
            ExecutionStateTag::from(&ExecutionState::Failed(make_failure_info())),
            ExecutionStateTag::Failed
        );
    }

    #[test]
    fn execution_state_serde_roundtrip() {
        let states: Vec<ExecutionState> = vec![
            ExecutionState::Running,
            ExecutionState::Paused(make_pause_info()),
            ExecutionState::Done(ExecutionResult {
                value: serde_json::json!({"ok": true}),
                usage: None,
                finished_at: 1_700_000_000_000,
            }),
            ExecutionState::Cancelled(make_cancel_info()),
            ExecutionState::Failed(make_failure_info()),
        ];

        for state in states {
            let json = serde_json::to_string(&state).expect("serialize");
            let _: ExecutionState = serde_json::from_str(&json).expect("deserialize");
        }
    }
}
