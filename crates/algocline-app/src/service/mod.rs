mod config;
mod eval;
mod eval_store;
mod logging;
pub(crate) mod path;
mod pkg;
pub mod resolve;
mod run;
mod scenario;
mod transcript;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use algocline_engine::{Executor, SessionRegistry};

pub use config::{AppConfig, LogDirSource};
pub use resolve::{QueryResponse, SearchPath};

// ─── Application Service ────────────────────────────────────────

/// Tracks which sessions are eval sessions and their strategy name.
type EvalSessions = std::sync::Mutex<std::collections::HashMap<String, String>>;

/// Tracks session_id → strategy name for all strategy-based sessions (advice, eval).
type SessionStrategies = std::sync::Mutex<std::collections::HashMap<String, String>>;

#[derive(Clone)]
pub struct AppService {
    executor: Arc<Executor>,
    registry: Arc<SessionRegistry>,
    log_config: AppConfig,
    /// Package search paths in priority order (first = highest).
    search_paths: Vec<resolve::SearchPath>,
    /// session_id → strategy name for eval sessions (cleared on completion).
    eval_sessions: Arc<EvalSessions>,
    /// session_id → strategy name for log/stats tracking (cleared on session completion).
    session_strategies: Arc<SessionStrategies>,
}

impl AppService {
    pub fn new(
        executor: Arc<Executor>,
        log_config: AppConfig,
        search_paths: Vec<resolve::SearchPath>,
    ) -> Self {
        Self {
            executor,
            registry: Arc::new(SessionRegistry::new()),
            log_config,
            search_paths,
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Returns the log directory, or an error if file logging is unavailable.
    fn require_log_dir(&self) -> Result<&Path, String> {
        self.log_config
            .log_dir
            .as_deref()
            .ok_or_else(|| "File logging is not available (no writable log directory)".to_string())
    }
}
