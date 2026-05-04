pub(crate) mod alc_toml;
mod error;
pub(crate) use error::{
    HubRegistriesError, PkgListError, ProjectFilesError, ServiceError, TranscriptError,
};
mod card;
mod config;
mod dist;
mod engine_api_impl;
mod eval;
mod eval_store;
pub(crate) mod gendoc;
mod hub;
pub mod hub_dist_preset;
mod init;
pub(crate) mod list_opts;
pub(crate) mod lock;
pub(crate) mod lockfile;
mod logging;
pub(crate) mod manifest;
mod migrate;
pub(crate) mod path;
mod pkg;
mod pkg_link;
mod pkg_scaffold;
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

use crate::pool::{registry::with_registry_lock, PoolError, PoolRegistry};
use algocline_engine::{Executor, FileCardStore, JsonFileStore, SessionRegistry, VariantPkg};

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
    /// Persistent KV store backing `alc.state.*`.
    ///
    /// Rooted at `log_config.app_dir().state_dir()` and resolved once at
    /// construction; `Arc`-wrapped so per-session clones are cheap.
    state_store: Arc<JsonFileStore>,
    /// Card store backing `alc.card.*`.
    ///
    /// Rooted at `log_config.app_dir().cards_dir()`, same `Arc` pattern.
    card_store: Arc<FileCardStore>,
    /// session_id → strategy name for eval sessions (cleared on completion).
    eval_sessions: Arc<EvalSessions>,
    /// session_id → strategy name for log/stats tracking (cleared on session completion).
    session_strategies: Arc<SessionStrategies>,
    /// Pool worker registry (persistent, backed by registry.json).
    ///
    /// `RwLock` because multiple concurrent callers may check for pool sessions
    /// while a single writer spawns a new worker.  The lock is never held across
    /// an `.await` boundary (K-4: clone-then-release pattern).
    pub(crate) pool_registry: Arc<tokio::sync::RwLock<PoolRegistry>>,
    /// Filesystem paths for pool registry management.
    ///
    /// Stored here so `run.rs` / `engine_api_impl.rs` can reach them without
    /// re-computing from `AppConfig` on every call.
    pub(crate) pool_reg_path: std::path::PathBuf,
    pub(crate) pool_lock_path: std::path::PathBuf,
    pub(crate) pool_dir: std::path::PathBuf,
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
        let app_dir = log_config.app_dir();
        let state_store = Arc::new(JsonFileStore::new(app_dir.state_dir()));
        let card_store = Arc::new(FileCardStore::new(app_dir.cards_dir()));

        // ─── Pool registry setup ───────────────────────────────────────────────
        // Paths: ~/.algocline/state/pool/{registry.json, registry.lock}
        // No pool_dir() helper in AppDir (not in scope); derive manually.
        let pool_dir = app_dir.state_dir().join("pool");
        let pool_reg_path = pool_dir.join("registry.json");
        let pool_lock_path = pool_dir.join("registry.lock");

        // Startup GC: remove dead worker entries accumulated from previous
        // MCP sessions.  Runs synchronously in new() (called before any
        // tokio tasks spawn) so spawn_blocking is not needed (K-110).
        //
        // If GC fails (corrupt registry, lock I/O error), we start with an
        // empty registry and emit a tracing::warn.  This is justified for
        // startup housekeeping only: the worker processes themselves remain
        // alive — they accumulate as orphans until the next restart, which
        // is acceptable for a POC.  A corrupt registry.json on startup is
        // surfaced via the tracing warn so operators can investigate.
        //
        // NOTE: this is the ONLY place in AppService that uses
        // tracing::warn without propagating to MCP wire.  It is defensible
        // because new() has no return type capable of carrying the error, and
        // GC failure has no correctness impact on the current session.
        let pool_registry = match with_registry_lock(&pool_lock_path, || {
            let mut reg = PoolRegistry::load_or_default(&pool_reg_path)?;
            let _ = reg.scan_and_gc()?;
            reg.save(&pool_reg_path)?;
            Ok::<_, PoolError>(reg)
        }) {
            Ok(reg) => reg,
            Err(e) => {
                tracing::warn!("pool registry startup GC failed (workers may accumulate): {e}");
                PoolRegistry::default()
            }
        };

        Self {
            executor,
            registry,
            log_config,
            search_paths,
            state_store,
            card_store,
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            pool_registry: Arc::new(tokio::sync::RwLock::new(pool_registry)),
            pool_reg_path,
            pool_lock_path,
            pool_dir,
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
    /// Merges two layers in priority order (first = highest = prepended
    /// by the Executor to `package.path`):
    ///
    /// 1. `alc.local.toml` path entries — worktree-scoped override
    ///    (git-ignored, not persisted to alc.lock, loaded every call).
    /// 2. `alc.lock` path entries — alc.toml-derived, git-managed.
    ///
    /// Returns `(paths, warnings)`. An empty `paths` is returned when no project
    /// root is found. Corruption errors (parse failures) are returned as warning
    /// strings in the second element rather than dropped silently — callers are
    /// responsible for surfacing them on the MCP wire response. File-absent is
    /// `Ok(None)` in the underlying loaders and produces no warning.
    pub(crate) fn resolve_extra_lib_paths(
        &self,
        project_root: Option<&str>,
    ) -> (Vec<std::path::PathBuf>, Vec<String>) {
        let Some(root) = project::resolve_project_root(project_root) else {
            return (vec![], vec![]);
        };

        let mut warnings: Vec<String> = Vec::new();

        // Local override layer (highest priority) — merged every call,
        // never persisted to alc.lock (decisions.md FsResolver priority).
        let local_paths: Vec<std::path::PathBuf> = match alc_toml::load_alc_local_toml(&root) {
            Ok(Some(local)) => alc_toml::resolve_local_path_entries(&root, &local),
            Ok(None) => Vec::new(),
            Err(e) => {
                warnings.push(format!(
                    "failed to load alc.local.toml at {}: {e}",
                    root.display()
                ));
                Vec::new()
            }
        };

        // Existing alc.lock layer.
        let lock_paths: Vec<std::path::PathBuf> = match lockfile::load_lockfile(&root) {
            Ok(Some(lock)) => {
                self.warn_toml_lock_mismatch(&root, &lock);
                let (paths, path_warnings) = lockfile::resolve_path_entries(&root, &lock);
                warnings.extend(path_warnings);
                paths
            }
            Ok(None) => Vec::new(),
            Err(e) => {
                warnings.push(format!(
                    "failed to load alc.lock at {}: {e}",
                    root.display()
                ));
                Vec::new()
            }
        };

        let mut merged = local_paths;
        merged.extend(lock_paths);
        (merged, warnings)
    }

    /// Resolve variant pkg overrides for a request.
    ///
    /// Reads `alc.local.toml` (worktree-scoped, gitignored) and emits one
    /// [`VariantPkg`] per `[packages.{name}] path = "..."` entry, preserving
    /// the explicit `(name, pkg_dir)` mapping. Returns `(pkgs, warnings)` —
    /// an empty `pkgs` when no project root is found or `alc.local.toml` is
    /// absent. Corruption (parse failures) is returned as a warning string so
    /// callers can surface it on the MCP wire response.
    pub(crate) fn resolve_variant_pkgs(
        &self,
        project_root: Option<&str>,
    ) -> (Vec<VariantPkg>, Vec<String>) {
        let Some(root) = project::resolve_project_root(project_root) else {
            return (vec![], vec![]);
        };

        match alc_toml::load_alc_local_toml(&root) {
            Ok(Some(local)) => (alc_toml::resolve_local_variant_pkgs(&root, &local), vec![]),
            Ok(None) => (Vec::new(), vec![]),
            Err(e) => {
                let msg = format!("failed to load alc.local.toml at {}: {e}", root.display());
                (Vec::new(), vec![msg])
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
