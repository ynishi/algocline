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
    async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        scenario_name: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
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

    /// List installed packages with metadata.
    async fn pkg_list(&self) -> Result<String, String>;

    /// Install a package from a Git URL or local path.
    async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String>;

    /// Remove an installed package.
    async fn pkg_remove(&self, name: &str) -> Result<String, String>;

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

    // ─── Diagnostics ─────────────────────────────────────────

    /// Show server configuration and diagnostic info.
    async fn info(&self) -> String;
}
