use std::sync::Arc;

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use serde::Deserialize;

use algocline_app::{AppService, TranscriptConfig};
use algocline_engine::Executor;

// ─── MCP Parameter types (schemars-annotated) ───────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunParams {
    /// Lua source code (inline). Provide either `code` or `code_file`, not both.
    pub code: Option<String>,
    /// Path to a Lua source file. Provide either `code` or `code_file`, not both.
    pub code_file: Option<String>,
    /// Context passed to Lua as the `ctx` global (JSON object).
    pub ctx: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContinueParams {
    /// Session ID returned by alc_run.
    pub session_id: String,
    /// Single response (backward-compatible). Used when query_id is absent
    /// or when there is exactly one pending query.
    pub response: Option<String>,
    /// Query ID for partial feed. Required when multiple queries are pending.
    pub query_id: Option<String>,
    /// Batch responses. Feed multiple query responses at once.
    pub responses: Option<Vec<McpQueryResponse>>,
}

/// A single query response in a batch feed (MCP schema).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct McpQueryResponse {
    /// Query ID (e.g. "q-0", "q-1").
    pub query_id: String,
    /// The host LLM's response for this query.
    pub response: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgInstallParams {
    /// Git URL or local path of a package or collection
    /// (e.g. "github.com/user/my-pkg", "file:///path/to/local/pkg").
    /// Single package: repo has init.lua at root → installed as one package.
    /// Collection: repo has subdirs with init.lua → each subdir installed as a package.
    pub url: String,
    /// Optional package name override (single package mode only).
    /// Defaults to the last segment of the URL. Ignored for collections.
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgRemoveParams {
    /// Name of the package to remove.
    pub name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NoteParams {
    /// Session ID of the execution to annotate.
    pub session_id: String,
    /// Note content (free text).
    pub content: String,
    /// Short label for what this note refers to (e.g. "Step 2", "overall").
    pub title: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LogViewParams {
    /// Session ID to view in detail. Omit to list all sessions.
    pub session_id: Option<String>,
    /// Max sessions to return in list mode (default: 50). Ignored when session_id is provided.
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AdviceParams {
    /// Package name: "ucb" (UCB1 hypothesis exploration), "panel" (multi-perspective),
    /// "cot" (chain-of-thought), "sc" (self-consistency), "cove" (chain-of-verification),
    /// or any installed package. Loaded via require("{name}").
    pub strategy: String,
    /// The task or question to process.
    pub task: String,
    /// Additional strategy-specific options (merged into ctx).
    pub opts: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvalParams {
    /// Scenario definition as inline Lua code. Returns a table with bindings and cases.
    /// Provide either `scenario` or `scenario_file`, not both.
    ///
    /// Example:
    /// ```lua
    /// local ef = require("evalframe")
    /// return {
    ///   ef.bind { ef.graders.contains },
    ///   cases = {
    ///     ef.case { input = "What is 2+2?", expected = "4" },
    ///   },
    /// }
    /// ```
    pub scenario: Option<String>,
    /// Path to a scenario Lua file. Provide either `scenario` or `scenario_file`, not both.
    pub scenario_file: Option<String>,
    /// Strategy package name to evaluate (e.g. "cove", "reflect", "ucb").
    /// Loaded via `ef.providers.algocline { strategy = "..." }`.
    pub strategy: String,
    /// Additional strategy-specific options (merged into provider opts).
    pub strategy_opts: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvalHistoryParams {
    /// Filter by strategy name (e.g. "cove", "reflect"). Omit to list all.
    pub strategy: Option<String>,
    /// Max results to return (default: 20).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvalDetailParams {
    /// Eval ID (e.g. "cove_1710672000"). Returned in eval history listing.
    pub eval_id: String,
}

// ─── MCP Handler ────────────────────────────────────────────────

#[derive(Clone)]
pub struct AlcService {
    tool_router: ToolRouter<Self>,
    app: AppService,
}

#[tool_router]
impl AlcService {
    pub fn new(executor: Arc<Executor>, log_config: TranscriptConfig) -> Self {
        Self {
            tool_router: Self::tool_router(),
            app: AppService::new(executor, log_config),
        }
    }

    /// Execute Lua code with optional JSON context.
    /// Returns the Lua return value as JSON.
    /// Lua code can call `alc.llm(prompt, opts)` to invoke the Host LLM
    /// via MCP Sampling.
    #[tool(name = "alc_run", annotations(open_world_hint = false))]
    async fn run(&self, Parameters(params): Parameters<RunParams>) -> Result<String, String> {
        self.app
            .run(params.code, params.code_file, params.ctx)
            .await
    }

    /// Apply a built-in strategy to a task.
    ///
    /// Applies any installed package by name. Official packages include:
    /// "ucb", "panel", "cot", "sc", "cove", "calibrate", "cod", "decompose",
    /// "distill", "factscore", "maieutic", "rank", "reflect", "review", "sot", "triad".
    /// Uses require("{name}") to load the package.
    #[tool(name = "alc_advice", annotations(open_world_hint = false))]
    async fn advice(&self, Parameters(params): Parameters<AdviceParams>) -> Result<String, String> {
        self.app
            .advice(&params.strategy, params.task, params.opts)
            .await
    }

    /// Continue a paused Lua execution by providing the host LLM's response.
    ///
    /// When `alc_run` or `alc_advice` returns `{"status": "needs_response", ...}`,
    /// the host processes the prompt and calls this tool with the response to resume.
    ///
    /// Supports three modes:
    /// - Single response: `{ "session_id": "...", "response": "..." }`
    /// - Partial feed: `{ "session_id": "...", "query_id": "q-0", "response": "..." }`
    /// - Batch feed: `{ "session_id": "...", "responses": [{ "query_id": "q-0", "response": "..." }, ...] }`
    #[tool(name = "alc_continue", annotations(open_world_hint = false))]
    async fn cont(&self, Parameters(params): Parameters<ContinueParams>) -> Result<String, String> {
        let sid = &params.session_id;

        // Mode 1: Batch feed
        if let Some(responses) = params.responses {
            let app_responses = responses
                .into_iter()
                .map(|r| algocline_app::QueryResponse {
                    query_id: r.query_id,
                    response: r.response,
                })
                .collect();
            return self.app.continue_batch(sid, app_responses).await;
        }

        // Mode 2/3: Single response (with or without query_id)
        let response = params
            .response
            .ok_or("Either 'response' or 'responses' must be provided")?;

        self.app
            .continue_single(sid, response, params.query_id.as_deref())
            .await
    }

    // ─── Evaluation ────────────────────────────────────────────

    /// Run an evalframe evaluation suite.
    ///
    /// Evaluates a strategy against a scenario (cases + graders).
    /// The evalframe package must be installed (`alc_pkg_install`).
    /// The strategy is automatically wired as the provider via
    /// `ef.providers.algocline { strategy = "..." }`.
    ///
    /// Returns the suite report (summary, scores, failures).
    #[tool(name = "alc_eval", annotations(open_world_hint = false))]
    async fn eval(&self, Parameters(params): Parameters<EvalParams>) -> Result<String, String> {
        self.app
            .eval(
                params.scenario,
                params.scenario_file,
                &params.strategy,
                params.strategy_opts,
            )
            .await
    }

    /// List past eval results. Filter by strategy, sorted newest-first.
    /// Results are persisted in ~/.algocline/evals/.
    #[tool(
        name = "alc_eval_history",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn eval_history(
        &self,
        Parameters(params): Parameters<EvalHistoryParams>,
    ) -> Result<String, String> {
        self.app
            .eval_history(params.strategy.as_deref(), params.limit.unwrap_or(20))
    }

    /// View a specific eval result in full detail.
    #[tool(
        name = "alc_eval_detail",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn eval_detail(
        &self,
        Parameters(params): Parameters<EvalDetailParams>,
    ) -> Result<String, String> {
        self.app.eval_detail(&params.eval_id)
    }

    // ─── Package Management ─────────────────────────────────────

    /// List installed packages with metadata.
    /// Returns name, version, description, and category for each package.
    #[tool(
        name = "alc_pkg_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn pkg_list(&self) -> Result<String, String> {
        self.app.pkg_list().await
    }

    /// Install a package from a Git URL or local path.
    /// Clones the repository into ~/.algocline/packages/{name}/.
    /// Supports: `github.com/user/pkg`, `https://...`, `git@...`,
    /// `file:///absolute/path`, or bare `/absolute/path`.
    /// The package must have an init.lua at its root.
    #[tool(
        name = "alc_pkg_install",
        annotations(destructive_hint = true, open_world_hint = true)
    )]
    async fn pkg_install(
        &self,
        Parameters(params): Parameters<PkgInstallParams>,
    ) -> Result<String, String> {
        self.app.pkg_install(params.url, params.name).await
    }

    /// Remove an installed package.
    #[tool(
        name = "alc_pkg_remove",
        annotations(destructive_hint = true, open_world_hint = false)
    )]
    async fn pkg_remove(
        &self,
        Parameters(params): Parameters<PkgRemoveParams>,
    ) -> Result<String, String> {
        self.app.pkg_remove(&params.name).await
    }

    // ─── Logging ─────────────────────────────────────────────

    /// Add a note to a completed session's log.
    ///
    /// Appends free-text feedback or observations to the transcript log file.
    /// The session must have completed and have logging enabled.
    #[tool(name = "alc_note", annotations(open_world_hint = false))]
    async fn note(&self, Parameters(params): Parameters<NoteParams>) -> Result<String, String> {
        self.app
            .add_note(&params.session_id, &params.content, params.title.as_deref())
            .await
    }

    /// View session logs.
    ///
    /// Without session_id: returns a summary list of all logged sessions.
    /// With session_id: returns the full log (stats, transcript, notes).
    #[tool(
        name = "alc_log_view",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn log_view(
        &self,
        Parameters(params): Parameters<LogViewParams>,
    ) -> Result<String, String> {
        self.app
            .log_view(params.session_id.as_deref(), params.limit)
            .await
    }
}

#[tool_handler]
impl ServerHandler for AlcService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "algocline — LLM amplification engine. Execute Lua strategies that structurally \
                 enhance LLM reasoning via alc.run(). Strategies are Pure Lua modules with \
                 access to alc.* StdLib (json, log, state, llm).\n\n\
                 Tools:\n\
                 - alc_run: Execute Lua code with optional JSON context. Returns result as JSON.\n\
                 - alc_continue: Continue a paused execution by providing the LLM response.\n\
                 - alc_advice: Apply an installed package (ucb, panel, cot, sc, cove, reflect, etc.) to a task.\n\n\
                 When Lua calls alc.llm(prompt), execution pauses and returns the prompt.\n\
                 The host processes it and calls alc_continue with the response to resume.\n\n\
                 Evaluation:\n\
                 - alc_eval: Evaluate a strategy against a scenario. Pass scenario (cases + graders) and strategy name.\n\
                 - alc_eval_history: List past eval results. Filter by strategy, sorted newest-first.\n\
                 - alc_eval_detail: View a specific eval result in full detail.\n\n\
                 Package Management:\n\
                 - alc_pkg_list: List installed packages with metadata.\n\
                 - alc_pkg_install: Install a package or collection from a Git URL (e.g. github.com/user/my-pkg).\n\
                 - alc_pkg_remove: Remove an installed package.\n\n\
                 Logging:\n\
                 - alc_note: Add a note to a completed session's log (feedback, observations).\n\
                 - alc_log_view: View session logs. Omit session_id for summary list, provide it for full detail."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
