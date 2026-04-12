use async_trait::async_trait;

// ─── Parameter types (transport-independent) ─────────────────────

/// A single query response in a batch feed.
#[derive(Debug)]
pub struct QueryResponse {
    /// Query ID (e.g. "q-0", "q-1").
    pub query_id: String,
    /// The host LLM's response for this query.
    pub response: String,
    /// Token usage reported by the host for this query.
    pub usage: Option<crate::TokenUsage>,
}

// ─── Engine API trait ────────────────────────────────────────────

/// Transport-independent API for the algocline engine.
///
/// Abstracts the full public surface of AppService so that callers
/// (MCP handler, future daemon client, etc.) can operate through
/// `Arc<dyn EngineApi>` without depending on the concrete implementation.
///
/// All methods are async to support both local (in-process) and remote
/// (socket/HTTP) implementations uniformly.
#[async_trait]
pub trait EngineApi: Send + Sync {
    // ─── Core execution ──────────────────────────────────────

    /// Execute Lua code with optional JSON context.
    async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String>;

    /// Apply an installed strategy package. Task is optional.
    async fn advice(
        &self,
        strategy: &str,
        task: Option<String>,
        opts: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String>;

    /// Continue a paused execution — single response (with optional query_id).
    async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
        usage: Option<crate::TokenUsage>,
    ) -> Result<String, String>;

    /// Continue a paused execution — batch feed.
    async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String>;

    // ─── Session status ──────────────────────────────────────

    /// Query active session status.
    async fn status(&self, session_id: Option<&str>) -> Result<String, String>;

    // ─── Evaluation ──────────────────────────────────────────

    /// Run an evalframe evaluation suite.
    ///
    /// `auto_card`: when true, emit an immutable Card
    /// (`~/.algocline/cards/{strategy}/{card_id}.toml`) summarizing the run.
    async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        scenario_name: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
        auto_card: bool,
    ) -> Result<String, String>;

    /// List eval history, optionally filtered by strategy.
    async fn eval_history(&self, strategy: Option<&str>, limit: usize) -> Result<String, String>;

    /// View a specific eval result by ID.
    async fn eval_detail(&self, eval_id: &str) -> Result<String, String>;

    /// Compare two eval results with statistical significance testing.
    async fn eval_compare(&self, eval_id_a: &str, eval_id_b: &str) -> Result<String, String>;

    // ─── Scenarios ───────────────────────────────────────────

    /// List available scenarios.
    async fn scenario_list(&self) -> Result<String, String>;

    /// Show the content of a named scenario.
    async fn scenario_show(&self, name: &str) -> Result<String, String>;

    /// Install scenarios from a Git URL or local path.
    async fn scenario_install(&self, url: String) -> Result<String, String>;

    // ─── Packages ────────────────────────────────────────────

    /// Link a local directory as a project-local package (symlink to cache).
    ///
    /// Phase 2 (subtask 5) will implement the full symlink logic.
    async fn pkg_link(
        &self,
        path: String,
        name: Option<String>,
        force: Option<bool>,
    ) -> Result<String, String>;

    /// List installed packages with metadata.
    ///
    /// When `project_root` is provided, project-local packages from `alc.toml`/`alc.lock`
    /// are included with `scope: "project"`. Global packages carry `scope: "global"`.
    async fn pkg_list(&self, project_root: Option<String>) -> Result<String, String>;

    /// Install a package from a Git URL or local path.
    async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String>;

    /// Remove a symlinked package from `~/.algocline/packages/`.
    ///
    /// Only removes symlinks; for installed (copied) packages, use `pkg_remove`.
    async fn pkg_unlink(&self, name: String) -> Result<String, String>;

    /// Remove a package declaration from `alc.toml` and `alc.lock`.
    ///
    /// Requires an `alc.toml` to be found (via `project_root` or ancestor walk).
    /// Does NOT delete physical files from `~/.algocline/packages/`.
    async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        version: Option<String>,
    ) -> Result<String, String>;

    // ─── Logging ─────────────────────────────────────────────

    /// Append a note to a session's log file.
    async fn add_note(
        &self,
        session_id: &str,
        content: &str,
        title: Option<&str>,
    ) -> Result<String, String>;

    /// View session logs.
    async fn log_view(
        &self,
        session_id: Option<&str>,
        limit: Option<usize>,
        max_chars: Option<usize>,
    ) -> Result<String, String>;

    /// Aggregate stats across all logged sessions.
    async fn stats(
        &self,
        strategy_filter: Option<&str>,
        days: Option<u64>,
    ) -> Result<String, String>;

    // ─── Project lifecycle ────────────────────────────────────

    /// Initialize `alc.toml` in the given project root.
    ///
    /// Creates a minimal `alc.toml` (`[packages]` section only).
    /// Fails if `alc.toml` already exists (no overwrite).
    async fn init(&self, project_root: Option<String>) -> Result<String, String>;

    /// Re-resolve all `alc.toml` entries and rewrite `alc.lock`.
    ///
    /// Requires an `alc.toml` to be present. Returns resolved count and errors.
    async fn update(&self, project_root: Option<String>) -> Result<String, String>;

    /// Migrate a legacy `alc.lock` to `alc.toml` + new `alc.lock` format.
    ///
    /// Detects legacy format via `linked_at` / `local_dir` fields.
    /// Backs up the old lock file as `alc.lock.bak`.
    async fn migrate(&self, project_root: Option<String>) -> Result<String, String>;

    // ─── Cards ───────────────────────────────────────────────

    /// List Card summaries, optionally filtered by pkg.
    async fn card_list(&self, pkg: Option<String>) -> Result<String, String>;

    /// Fetch a full Card by id.
    async fn card_get(&self, card_id: &str) -> Result<String, String>;

    /// Filter/sort Cards with optional pkg / scenario / model / sort / limit / min_pass_rate.
    async fn card_find(
        &self,
        pkg: Option<String>,
        scenario: Option<String>,
        model: Option<String>,
        sort: Option<String>,
        limit: Option<usize>,
        min_pass_rate: Option<f64>,
    ) -> Result<String, String>;

    /// List aliases, optionally filtered by pkg.
    async fn card_alias_list(&self, pkg: Option<String>) -> Result<String, String>;

    /// Resolve an alias name to its bound Card and return the full Card JSON.
    async fn card_get_by_alias(&self, name: &str) -> Result<String, String>;

    /// Bind (or rebind) an alias to a Card.
    async fn card_alias_set(
        &self,
        name: &str,
        card_id: &str,
        pkg: Option<String>,
        note: Option<String>,
    ) -> Result<String, String>;

    /// Append new top-level fields to an existing Card (additive-only).
    async fn card_append(&self, card_id: &str, fields: serde_json::Value)
        -> Result<String, String>;

    /// Install Cards from a Card Collection repo (Git URL or local path).
    async fn card_install(&self, url: String) -> Result<String, String>;

    /// Read per-case samples from a Card's sidecar JSONL file.
    async fn card_samples(
        &self,
        card_id: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, String>;

    // ─── Hub ─────────────────────────────────────────────────

    /// Rebuild hub index from a packages directory.
    ///
    /// When `source_dir` is provided, scans that directory directly
    /// (pure metadata, no manifest).  When omitted, scans `~/.algocline/packages/`.
    async fn hub_reindex(
        &self,
        output_path: Option<String>,
        source_dir: Option<String>,
    ) -> Result<String, String>;

    /// Show detailed information for a single package.
    async fn hub_info(&self, pkg: String) -> Result<String, String>;

    /// Search packages across remote index + local install state.
    async fn hub_search(
        &self,
        query: Option<String>,
        category: Option<String>,
        installed_only: Option<bool>,
        limit: Option<usize>,
    ) -> Result<String, String>;

    // ─── Diagnostics ─────────────────────────────────────────

    /// Show server configuration and diagnostic info.
    async fn info(&self) -> String;
}
