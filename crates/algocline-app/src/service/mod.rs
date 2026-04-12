pub(crate) mod alc_toml;
mod card;
mod config;
mod engine_api_impl;
mod eval;
mod eval_store;
mod init;
pub(crate) mod lockfile;
mod logging;
pub(crate) mod manifest;
mod migrate;
pub(crate) mod path;
mod pkg;
mod pkg_link;
mod pkg_unlink;
pub(crate) mod project;
pub mod resolve;
mod run;
mod scenario;
pub(crate) mod source;
mod status;
mod transcript;
mod update;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use algocline_engine::{Executor, SessionRegistry};

pub use algocline_core::{EngineApi, TokenUsage};
pub use config::{AppConfig, LogDirSource};
pub use resolve::{QueryResponse, SearchPath};

// ─── Application Service ────────────────────────────────────────

/// Tracks in-flight eval sessions: session_id → strategy name.
///
/// Kept between `alc_eval` invocation and eventual completion (which may
/// arrive via `alc_continue` after LLM round-trips). Used by
/// `run.rs::maybe_save_eval` to persist the result to `~/.algocline/evals/`.
/// Card emission is handled by `alc.eval()` Lua-side — no Rust tracking needed.
///
/// `std::sync::Mutex` is used (not tokio) because all operations are
/// single HashMap insert/remove/get completing in microseconds, and no
/// `.await` is held across the lock. Poison is silently skipped.
type EvalSessions = std::sync::Mutex<std::collections::HashMap<String, String>>;

/// Tracks session_id → strategy name for all strategy-based sessions (advice, eval).
///
/// Same locking rationale as `EvalSessions`. Used by `alc_status` and
/// transcript logging. Poison is silently skipped — strategy name is
/// non-critical metadata for observability.
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
        let registry = Arc::new(SessionRegistry::new());
        // TTL = 3 hours. Complex strategies may run 30–60 min; 3h covers
        // legitimate paused sessions while eventually reclaiming abandoned ones.
        registry.spawn_gc_task(std::time::Duration::from_secs(10800));
        Self {
            executor,
            registry,
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

    /// Resolve extra lib paths for a request.
    ///
    /// 1. Determines the project root from `project_root` (explicit) or
    ///    `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    /// 2. Reads `alc.lock` from that root.
    /// 3. Returns the resolved absolute paths of all `local_dir` entries.
    ///
    /// Returns an empty `Vec` if no project root is found, if `alc.lock`
    /// does not exist, or if no `local_dir` entries are present.
    pub(crate) fn resolve_extra_lib_paths(
        &self,
        project_root: Option<&str>,
    ) -> Vec<std::path::PathBuf> {
        let Some(root) = project::resolve_project_root(project_root) else {
            return vec![];
        };

        match lockfile::load_lockfile(&root) {
            Ok(Some(lock)) => {
                self.warn_toml_lock_mismatch(&root, &lock);
                lockfile::resolve_path_entries(&root, &lock)
            }
            Ok(None) => vec![],
            Err(e) => {
                tracing::warn!("failed to load alc.lock at {}: {e}", root.display());
                vec![]
            }
        }
    }

    fn warn_toml_lock_mismatch(&self, root: &Path, lock: &lockfile::LockFile) {
        let toml = match alc_toml::load_alc_toml(root) {
            Ok(Some(t)) => t,
            _ => return,
        };

        use std::collections::BTreeSet;
        let toml_names: BTreeSet<&str> = toml.packages.keys().map(|s| s.as_str()).collect();
        let lock_names: BTreeSet<&str> = lock.packages.iter().map(|p| p.name.as_str()).collect();

        for name in toml_names.difference(&lock_names) {
            eprintln!(
                "warning: '{name}' is declared in alc.toml but missing from alc.lock. Run `alc_update` to sync."
            );
        }
        for name in lock_names.difference(&toml_names) {
            eprintln!("warning: '{name}' is in alc.lock but not declared in alc.toml.");
        }
    }
}
