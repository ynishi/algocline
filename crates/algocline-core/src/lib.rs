//! Core domain types and primitives shared across the algocline workspace.
//!
//! Provides the shared vocabulary consumed by `algocline-engine`,
//! `algocline-app`, and `algocline-mcp`: [`AppDir`] path handling,
//! [`Budget`] / token accounting, the [`EngineApi`] trait, execution
//! and query domain types, metrics collection, package (`pkg`) metadata,
//! progress reporting ([`ProgressInfo`]), the recent-log ring buffer,
//! and shared state primitives.

mod app_dir;
mod budget;
mod custom;
pub mod domain;
mod engine_api;
pub mod execution;
pub mod metrics;
mod observer;
pub mod pkg;
mod progress;
mod query;
pub mod recent_log;
mod spec;
mod state;
mod tokens;

pub use app_dir::AppDir;
pub use budget::{Budget, BudgetHandle};
pub use custom::{CustomMetrics, CustomMetricsHandle};
pub use engine_api::{EngineApi, QueryResponse};
pub use metrics::{ExecutionMetrics, MetricsObserver, StatsHandle};
pub use observer::ExecutionObserver;
pub use pkg::{PkgEntity, PkgType, TypeSource};
pub use progress::{ProgressHandle, ProgressInfo};
pub use query::{LlmQuery, QueryId};
pub use recent_log::{LogEntry, LogSink};
pub use spec::ExecutionSpec;
pub use state::{
    ExecutionState, FeedError, PendingQueries, ResumeOutcome, TerminalState, TransitionError,
};
pub use tokens::{TokenCount, TokenSource, TokenUsage};
