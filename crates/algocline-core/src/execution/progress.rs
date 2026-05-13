//! Progress observation types for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};

use super::error::ObserverRecvError;
use super::pause::{PauseInfo, PauseKind};
use super::state::ExecutionStateTag;
use crate::TokenUsage;

/// An event emitted on the per-session broadcast channel.
///
/// All variants carry an `at` field (Unix timestamp in milliseconds).
/// The channel is emitted from [`crate::execution::ExecutionService`] implementations
/// inside the engine layer; multiple independent observers may subscribe via
/// [`crate::execution::ExecutionService::observe`] without affecting one another
/// (crux: Sink-free broadcast Progress fan-out).
///
/// # Serde
/// Uses `#[serde(tag = "kind", rename_all = "snake_case")]` internally tagged
/// representation, so JSON consumers see `{"kind": "state_transition", ...}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// The session transitioned from one state to another.
    StateTransition {
        /// The previous state tag.
        from: ExecutionStateTag,
        /// The new state tag.
        to: ExecutionStateTag,
        /// Unix timestamp (milliseconds) of the transition.
        at: i64,
    },
    /// The session entered a paused state and is waiting for LLM responses.
    PauseRequested {
        /// Details of the pause (pending prompts, kind, timestamp).
        info: PauseInfo,
        /// Unix timestamp (milliseconds) when the pause was requested.
        at: i64,
    },
    /// A `resume` call was accepted and execution is continuing.
    ResumeAccepted {
        /// The pause kind (single or batch) that was resolved.
        payload_kind: PauseKind,
        /// Unix timestamp (milliseconds) when the resume was accepted.
        at: i64,
    },
    /// A free-form diagnostic note from the engine.
    Note {
        /// Optional short title for the note.
        title: Option<String>,
        /// Note body.
        content: String,
        /// Unix timestamp (milliseconds) of the note.
        at: i64,
    },
    /// An LLM call was dispatched for the given query.
    LlmCallBegin {
        /// The query identifier.
        query_id: String,
        /// Unix timestamp (milliseconds) when the call began.
        at: i64,
    },
    /// An LLM call completed and a response was received.
    LlmCallEnd {
        /// The query identifier.
        query_id: String,
        /// Token usage for this call, if reported.
        usage: Option<TokenUsage>,
        /// Unix timestamp (milliseconds) when the call ended.
        at: i64,
    },
    /// A periodic heartbeat tick from the execution loop.
    Tick {
        /// A short label describing the current execution phase.
        phase: String,
        /// Unix timestamp (milliseconds) of the tick.
        at: i64,
    },
}

/// Handle returned by [`crate::execution::ExecutionService::observe`].
///
/// Wraps a `tokio::sync::broadcast::Receiver<ProgressEvent>` (provided by the engine
/// layer, Subtask 2).  Multiple handles may exist concurrently; each receives the full
/// event stream independently.  When the session's broadcast sender is dropped (session
/// terminates), `recv()` returns `Err(ObserverRecvError::Closed)`.
///
/// This is a **trait** to allow the engine layer (Subtask 2) to provide a concrete
/// `BroadcastObserverHandle` struct without a circular dependency on core.
pub trait ObserverHandle: Send {
    /// Receive the next event, waiting asynchronously.
    ///
    /// Returns `Err(ObserverRecvError::Closed)` when the broadcast sender is dropped
    /// (session terminated).  Returns `Err(ObserverRecvError::Lagged(n))` when the
    /// receiver fell behind and `n` events were skipped; subsequent calls continue
    /// from the latest available event.
    fn recv(
        &mut self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ProgressEvent, ObserverRecvError>> + Send + '_>,
    >;

    /// Non-blocking receive.  Returns an event if one is immediately available,
    /// `Err(ObserverRecvError::Closed)` if the sender is gone, or
    /// `Err(ObserverRecvError::Lagged(n))` on a lag event.
    ///
    /// Callers should treat `Err` variants consistently with [`Self::recv`].
    fn try_recv(&mut self) -> Result<ProgressEvent, ObserverRecvError>;

    /// Consume and close this handle.  After calling `close` the handle is dropped
    /// and no further events can be received.
    fn close(self: Box<Self>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_event_serde_tagged_kind() {
        // StateTransition must serialize as {"kind": "state_transition", ...}
        let event = ProgressEvent::StateTransition {
            from: ExecutionStateTag::Running,
            to: ExecutionStateTag::Paused,
            at: 1_700_000_000_000,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(
            json.contains(r#""kind":"state_transition""#),
            "expected tagged kind in JSON, got: {json}"
        );
        let roundtripped: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        // Verify roundtrip by re-serializing and comparing
        let json2 = serde_json::to_string(&roundtripped).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn all_progress_event_variants_serde() {
        use crate::execution::pause::{PauseInfo, PauseKind};

        let events: Vec<ProgressEvent> = vec![
            ProgressEvent::StateTransition {
                from: ExecutionStateTag::Running,
                to: ExecutionStateTag::Done,
                at: 0,
            },
            ProgressEvent::PauseRequested {
                info: PauseInfo {
                    kind: PauseKind::Single,
                    prompts: vec![],
                    paused_at: 0,
                },
                at: 0,
            },
            ProgressEvent::ResumeAccepted {
                payload_kind: PauseKind::Batch,
                at: 0,
            },
            ProgressEvent::Note {
                title: Some("test".into()),
                content: "hello".into(),
                at: 0,
            },
            ProgressEvent::LlmCallBegin {
                query_id: "q1".into(),
                at: 0,
            },
            ProgressEvent::LlmCallEnd {
                query_id: "q1".into(),
                usage: None,
                at: 0,
            },
            ProgressEvent::Tick {
                phase: "running".into(),
                at: 0,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).expect("serialize");
            let _: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        }
    }
}
