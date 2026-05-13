//! Pure execution service layer for `algocline-core`.
//!
//! This module provides the [`ExecutionService`] trait and all associated value types
//! for the new service layer design.  It coexists with the legacy types in
//! `crate::state` and `crate::engine_api` without modifying them (parallel evolution).
//!
//! ## Module structure
//!
//! | Module | Key types |
//! |--------|-----------|
//! | [`session_id`] | [`SessionId`] |
//! | [`spec`] | [`SessionSpec`], [`SpecKind`], [`ScenarioRef`] |
//! | [`state`] | [`ExecutionState`] (v2), [`ExecutionStateTag`], [`ExecutionResult`] |
//! | [`pause`] | [`PauseInfo`], [`PauseKind`], [`PausePrompt`] |
//! | [`resume`] | [`ResumePayload`], [`QueryResponse`], [`ResumeOutcome`], [`TerminalOutcome`] |
//! | [`cancel`] | [`CancelReason`], [`CancelCode`], [`CancelInfo`], [`FailureInfo`], [`FailureKind`] |
//! | [`progress`] | [`ProgressEvent`], [`ObserverHandle`] |
//! | [`error`] | [`SpawnError`], [`StateError`], [`ResumeError`], [`CancelError`], [`ObserveError`], [`AwaitError`], [`ObserverRecvError`] |
//! | [`service`] | [`ExecutionService`] |
//!
//! ## Access path
//!
//! Types in this module are accessed as `algocline_core::execution::Foo`.
//! They are **not** re-exported at the top level of `algocline_core` to avoid
//! naming conflicts with legacy top-level types (e.g., `QueryResponse`).

pub mod cancel;
pub mod error;
pub mod pause;
pub mod progress;
pub mod resume;
pub mod service;
pub mod session_id;
pub mod spec;
pub mod state;

// Re-export all public items for convenient access via `algocline_core::execution::*`.
pub use cancel::{CancelCode, CancelInfo, CancelReason, FailureInfo, FailureKind};
pub use error::{
    AwaitError, CancelError, ObserveError, ObserverRecvError, ResumeError, SpawnError, StateError,
};
pub use pause::{PauseInfo, PauseKind, PausePrompt};
pub use progress::{ObserverHandle, ProgressEvent};
pub use resume::{QueryResponse, ResumeOutcome, ResumePayload, TerminalOutcome};
pub use service::ExecutionService;
pub use session_id::SessionId;
pub use spec::{ScenarioRef, SessionSpec, SpecKind};
pub use state::{ExecutionResult, ExecutionState, ExecutionStateTag};

// Re-export TokenUsage into this namespace so engine-layer code can use
// `algocline_core::execution::TokenUsage` without a separate import.
pub use crate::TokenUsage;
