//! Re-exports for domain types.
//!
//! This module exists to provide a clean `algocline_core::domain::*` import path.
//! All types are also re-exported at the crate root.

pub use crate::query::{LlmQuery, QueryId};
pub use crate::spec::ExecutionSpec;
pub use crate::state::{
    ExecutionState, FeedError, PendingQueries, ResumeOutcome, TerminalState, TransitionError,
};
