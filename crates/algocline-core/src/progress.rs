use std::sync::{Arc, Mutex};

use crate::metrics::SessionStatus;

// ─── Progress ───────────────────────────────────────────────

/// Structured progress information reported by strategies via `alc.progress()`.
///
/// Stored in `SessionStatus` and readable via `alc_status` MCP tool.
/// Not all strategies report progress — this is opt-in for strategies
/// that benefit from structured step tracking (e.g. multi-round pipelines).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressInfo {
    /// Current step (1-based).
    pub step: u64,
    /// Total expected steps (0 = unknown/indeterminate).
    pub total: u64,
    /// Optional human-readable message for the current step.
    pub message: Option<String>,
}

/// Cheap, cloneable handle for writing progress from the Lua bridge.
///
/// Wraps the shared `SessionStatus` to expose only progress-related writes.
/// Passed to `bridge::register_progress()`.
///
/// # Call site and threading
///
/// Called exclusively from the Lua OS thread via `alc.progress()`.
/// Acquires `std::sync::Mutex<SessionStatus>` for a few microseconds
/// (single field assignment). See `SessionStatus` doc for full locking design.
///
/// # Poison policy
///
/// Silently skips on poison. Progress is observational (consumed by
/// `alc_status`) — a missed update degrades monitoring but does not
/// affect execution correctness. If you observe stale progress in
/// `alc_status` while the session is active, mutex poison from an
/// earlier OOM panic is a possible cause.
#[derive(Clone)]
pub struct ProgressHandle {
    auto: Arc<Mutex<SessionStatus>>,
}

impl ProgressHandle {
    pub(crate) fn new(auto: Arc<Mutex<SessionStatus>>) -> Self {
        Self { auto }
    }

    /// Set the current progress. Called from `alc.progress(step, total, msg?)`.
    ///
    /// Silently skips on mutex poison (see ProgressHandle doc for rationale).
    pub fn set(&self, step: u64, total: u64, message: Option<String>) {
        if let Ok(mut m) = self.auto.lock() {
            m.progress = Some(ProgressInfo {
                step,
                total,
                message,
            });
        }
    }
}
