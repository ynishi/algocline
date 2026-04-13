use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use serde::Deserialize;

use std::sync::Arc;

use algocline_app::{EngineApi, QueryResponse};

// ─── MCP Parameter types (schemars-annotated) ───────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunParams {
    /// Lua source code (inline). Provide either `code` or `code_file`, not both.
    pub code: Option<String>,
    /// Path to a Lua source file. Provide either `code` or `code_file`, not both.
    pub code_file: Option<String>,
    /// Context passed to Lua as the `ctx` global (JSON object).
    pub ctx: Option<serde_json::Value>,
    /// Optional absolute path to the project root containing `alc.lock`.
    /// Falls back to `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    pub project_root: Option<String>,
}

/// Host-reported token usage for an LLM call (MCP schema).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct McpTokenUsage {
    /// Prompt tokens consumed by this LLM call.
    pub prompt_tokens: Option<u64>,
    /// Completion (response) tokens produced by this LLM call.
    pub completion_tokens: Option<u64>,
}

impl From<McpTokenUsage> for algocline_app::TokenUsage {
    fn from(u: McpTokenUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        }
    }
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
    /// Token usage reported by the host for this response.
    /// Provides accurate token counts instead of character-based estimates.
    pub usage: Option<McpTokenUsage>,
}

/// A single query response in a batch feed (MCP schema).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct McpQueryResponse {
    /// Query ID (e.g. "q-0", "q-1").
    pub query_id: String,
    /// The host LLM's response for this query.
    pub response: String,
    /// Token usage reported by the host for this query.
    pub usage: Option<McpTokenUsage>,
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
pub struct PkgLinkParams {
    /// Absolute or relative path to the directory to link as a package.
    /// If the directory contains init.lua: single package mode.
    /// If subdirectories contain init.lua: collection mode (each subdir becomes a package).
    pub path: String,
    /// Optional name override for the package (single package mode only).
    /// Defaults to the directory name.
    pub name: Option<String>,
    /// Force overwrite of existing symlinks. Default: false.
    pub force: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgUnlinkParams {
    /// Name of the linked package to remove from `~/.algocline/packages/`.
    pub name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgListParams {
    /// Optional absolute path to project root.
    /// When provided, project-local packages from alc.lock are included
    /// alongside global packages. Each package carries a `scope` field
    /// ("project" or "global") and an `active` boolean.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgRemoveParams {
    /// Name of the package to remove.
    pub name: String,
    /// Optional absolute path to project root containing alc.toml.
    /// Falls back to ALC_PROJECT_ROOT env or ancestor walk from cwd.
    /// An alc.toml must be found — if not, the operation fails.
    pub project_root: Option<String>,
    /// Optional version constraint. When specified, only the alc.lock entry
    /// matching this version is removed. Omit to remove any version.
    pub version: Option<String>,
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
    /// Max response size in characters for detail mode (default: 100000).
    /// When exceeded, transcript is truncated from oldest rounds.
    /// Set to 0 for unlimited.
    pub max_chars: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatsParams {
    /// Filter by strategy name (e.g. "ucb", "cove"). Omit to see all strategies.
    pub strategy: Option<String>,
    /// Show only sessions from the last N days. Omit for all time.
    pub days: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AdviceParams {
    /// Package name: "ucb" (UCB1 hypothesis exploration), "panel" (multi-perspective),
    /// "cot" (chain-of-thought), "sc" (self-consistency), "cove" (chain-of-verification),
    /// or any installed package. Loaded via require("{name}").
    pub strategy: String,
    /// The task or question to process (optional).
    pub task: Option<String>,
    /// Additional strategy-specific options (merged into ctx).
    pub opts: Option<serde_json::Value>,
    /// Optional absolute path to the project root containing `alc.lock`.
    /// Falls back to `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvalParams {
    /// Scenario definition as inline Lua code. Returns a table with bindings and cases.
    /// Provide exactly one of: `scenario`, `scenario_file`, or `scenario_name`.
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
    /// Path to a scenario Lua file. Provide exactly one of: `scenario`, `scenario_file`, or `scenario_name`.
    pub scenario_file: Option<String>,
    /// Name of an installed scenario (e.g. "math_basic").
    /// Resolved from `~/.algocline/scenarios/{name}.lua`.
    /// Provide exactly one of: `scenario`, `scenario_file`, or `scenario_name`.
    pub scenario_name: Option<String>,
    /// Strategy package name to evaluate (e.g. "cove", "reflect", "ucb").
    /// Loaded via `ef.providers.algocline { strategy = "..." }`.
    pub strategy: String,
    /// Additional strategy-specific options (merged into provider opts).
    pub strategy_opts: Option<serde_json::Value>,
    /// If true, also emit an immutable Card (`~/.algocline/cards/{strategy}/{card_id}.toml`)
    /// summarizing this eval run. Default: false. Schema: `card/v0`.
    pub auto_card: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ScenarioShowParams {
    /// Scenario name (e.g. "math_basic"). Resolved from `~/.algocline/scenarios/{name}.lua`.
    pub name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ScenarioInstallParams {
    /// Git URL or local absolute path containing scenario `.lua` files.
    /// If the source contains a `scenarios/` subdirectory, files are read from there.
    pub url: String,
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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvalCompareParams {
    /// First eval ID to compare.
    pub eval_id_a: String,
    /// Second eval ID to compare.
    pub eval_id_b: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatusParams {
    /// Session ID to inspect. Omit to list all active sessions.
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InitParams {
    /// Optional absolute path to the project root. Falls back to ALC_PROJECT_ROOT or cwd.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UpdateParams {
    /// Optional absolute path to the project root. Falls back to ALC_PROJECT_ROOT or ancestor walk.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MigrateParams {
    /// Optional absolute path to the project root. Falls back to ALC_PROJECT_ROOT or cwd.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardListParams {
    /// Optional pkg filter — restrict listing to `~/.algocline/cards/{pkg}/`.
    pub pkg: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardGetParams {
    /// Card ID (e.g. "prompt_ab_demo_opus46_20260412T120000_abc123").
    pub card_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardFindParams {
    /// Optional pkg filter.  Restricts the filesystem scan to a single
    /// pkg subdir — use it when you know the target package for speed.
    pub pkg: Option<String>,
    /// Prisma-style `where` predicate.  Nested objects are interpreted
    /// as section paths; keys whose value is an object whose every key
    /// is a reserved operator name (`eq ne lt lte gt gte in nin exists
    /// contains starts_with`) become leaf comparisons.  Logical ops:
    /// `_and` / `_or` / `_not`.  Example:
    /// `{ "stats": { "pass_rate": { "gte": 0.8 } }, "model": { "id": "claude-opus-4-6" } }`
    pub r#where: Option<serde_json::Value>,
    /// Sort keys.  Accepts a single dotted-path string (`"stats.pass_rate"`,
    /// `"-stats.pass_rate"` for desc) or an array of such strings.
    /// Defaults to `created_at` descending.
    pub order_by: Option<serde_json::Value>,
    /// Max rows returned.
    pub limit: Option<usize>,
    /// Skip this many rows before `limit` applies.
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardAliasListParams {
    /// Optional pkg filter.
    pub pkg: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardGetByAliasParams {
    /// Alias name (e.g. "best_prompt_ab").
    pub name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardAliasSetParams {
    /// Alias name (unique; rebinding overwrites).
    pub name: String,
    /// Card ID to bind. Must exist on disk.
    pub card_id: String,
    /// Optional pkg tag stored on the alias row.
    pub pkg: Option<String>,
    /// Optional free-form note.
    pub note: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardSamplesParams {
    /// Card ID whose sidecar samples to read.
    pub card_id: String,
    /// Skip this many rows from the start of the JSONL file. Default 0.
    pub offset: Option<usize>,
    /// Max rows returned. Omit to return everything from `offset`.
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardAppendParams {
    /// Card ID to append fields to.
    pub card_id: String,
    /// Top-level fields to merge. Existing keys are rejected (Cards are
    /// immutable for already-present data).
    pub fields: serde_json::Value,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardInstallParams {
    /// Git URL or local absolute path to a Card Collection.
    /// A Card Collection is a repo with `alc_cards.toml` at root and
    /// subdirectories named after packages, each containing Card TOML files.
    /// Example: `github.com/user/my-alcards` or `/path/to/local/cards`.
    pub url: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HubInfoParams {
    /// Package name to get detailed information for.
    pub pkg: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HubReindexParams {
    /// File path to write the generated index JSON to (e.g. for CI publishing).
    pub output_path: Option<String>,
    /// Directory to scan for packages (e.g. a repo checkout).
    /// When omitted, scans `~/.algocline/packages/` (local install state).
    /// When provided, generates a pure index from that directory only
    /// — no manifest sources or card counts are mixed in.
    pub source_dir: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HubSearchParams {
    /// Search query (matched against package name, description, category).
    /// Omit to list all available packages.
    pub query: Option<String>,
    /// Filter by category (e.g. "reasoning", "aggregation", "synthesis").
    pub category: Option<String>,
    /// When true, only show locally installed packages.
    pub installed_only: Option<bool>,
    /// Maximum number of results (default: 50).
    pub limit: Option<usize>,
}

// ─── MCP Handler ────────────────────────────────────────────────

#[derive(Clone)]
pub struct AlcService {
    tool_router: ToolRouter<Self>,
    app: Arc<dyn EngineApi>,
}

#[tool_router]
impl AlcService {
    pub fn new(app: Arc<dyn EngineApi>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            app,
        }
    }

    /// Execute Lua code with optional JSON context.
    /// Returns the Lua return value as JSON.
    /// Lua code can call `alc.llm(prompt, opts)` to invoke the Host LLM
    /// via MCP Sampling.
    #[tool(name = "alc_run", annotations(open_world_hint = false))]
    async fn run(&self, Parameters(params): Parameters<RunParams>) -> Result<String, String> {
        self.app
            .run(
                params.code,
                params.code_file,
                params.ctx,
                params.project_root,
            )
            .await
    }

    /// Apply a built-in strategy to a task (task is optional).
    ///
    /// Applies any installed package by name. Official packages include:
    /// "ucb", "panel", "cot", "sc", "cove", "calibrate", "cod", "decompose",
    /// "distill", "factscore", "maieutic", "rank", "reflect", "review", "sot", "triad".
    /// Uses require("{name}") to load the package.
    #[tool(name = "alc_advice", annotations(open_world_hint = false))]
    async fn advice(&self, Parameters(params): Parameters<AdviceParams>) -> Result<String, String> {
        self.app
            .advice(
                &params.strategy,
                params.task,
                params.opts,
                params.project_root,
            )
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
                .map(|r| QueryResponse {
                    query_id: r.query_id,
                    response: r.response,
                    usage: r.usage.map(Into::into),
                })
                .collect();
            return self.app.continue_batch(sid, app_responses).await;
        }

        // Mode 2/3: Single response (with or without query_id)
        let response = params
            .response
            .ok_or("Either 'response' or 'responses' must be provided")?;

        self.app
            .continue_single(
                sid,
                response,
                params.query_id.as_deref(),
                params.usage.map(Into::into),
            )
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
                params.scenario_name,
                &params.strategy,
                params.strategy_opts,
                params.auto_card.unwrap_or(false),
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
            .await
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
        self.app.eval_detail(&params.eval_id).await
    }

    /// Compare two eval results with Welch's t-test for statistical significance.
    ///
    /// Returns per-strategy descriptive statistics (mean, std_dev, median),
    /// score delta, Welch's t-test result (t-stat, df, significant),
    /// winner determination, and a human-readable summary.
    #[tool(
        name = "alc_eval_compare",
        annotations(read_only_hint = false, open_world_hint = false)
    )]
    async fn eval_compare(
        &self,
        Parameters(params): Parameters<EvalCompareParams>,
    ) -> Result<String, String> {
        self.app
            .eval_compare(&params.eval_id_a, &params.eval_id_b)
            .await
    }

    // ─── Scenario Management ───────────────────────────────────

    /// List available scenarios in ~/.algocline/scenarios/.
    #[tool(
        name = "alc_scenario_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn scenario_list(&self) -> Result<String, String> {
        self.app.scenario_list().await
    }

    /// Show the content of an installed scenario by name.
    #[tool(
        name = "alc_scenario_show",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn scenario_show(
        &self,
        Parameters(params): Parameters<ScenarioShowParams>,
    ) -> Result<String, String> {
        self.app.scenario_show(&params.name).await
    }

    /// Install scenarios from a Git URL or local path into ~/.algocline/scenarios/.
    /// Expects the source to contain `.lua` files at root or in a `scenarios/` subdirectory.
    #[tool(name = "alc_scenario_install", annotations(open_world_hint = false))]
    async fn scenario_install(
        &self,
        Parameters(params): Parameters<ScenarioInstallParams>,
    ) -> Result<String, String> {
        self.app.scenario_install(params.url).await
    }

    // ─── Package Management ─────────────────────────────────────

    /// Link a local directory as a package (symlink to cache).
    ///
    /// Creates a symlink from `~/.algocline/packages/{name}` to the given path.
    /// Changes to files in the source directory are reflected immediately on the next `alc_run`.
    ///
    /// Single mode: directory has `init.lua` at root → one package (name = dirname or `name` param).
    /// Collection mode: subdirectories have `init.lua` → each subdir is a package.
    #[tool(
        name = "alc_pkg_link",
        annotations(
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn pkg_link(
        &self,
        Parameters(params): Parameters<PkgLinkParams>,
    ) -> Result<String, String> {
        self.app
            .pkg_link(params.path, params.name, params.force)
            .await
    }

    /// List installed packages with metadata.
    ///
    /// When `project_root` is provided, project-local packages from `alc.lock`
    /// are included alongside global packages. Each package entry includes
    /// `scope` ("project" or "global") and `active` (effective vs shadowed).
    #[tool(
        name = "alc_pkg_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn pkg_list(
        &self,
        Parameters(params): Parameters<PkgListParams>,
    ) -> Result<String, String> {
        self.app.pkg_list(params.project_root).await
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

    /// Remove a package declaration from `alc.toml` and `alc.lock`.
    ///
    /// Physical files in `~/.algocline/packages/` are NOT deleted.
    /// Requires an `alc.toml` to be found (via `project_root` or ancestor walk from cwd).
    /// Pass `version` to remove only a specific version from `alc.lock`.
    #[tool(
        name = "alc_pkg_remove",
        annotations(destructive_hint = true, open_world_hint = false)
    )]
    async fn pkg_remove(
        &self,
        Parameters(params): Parameters<PkgRemoveParams>,
    ) -> Result<String, String> {
        self.app
            .pkg_remove(&params.name, params.project_root, params.version)
            .await
    }

    /// Remove a symlinked package from `~/.algocline/packages/`.
    ///
    /// Only removes symlinks created by `alc_pkg_link`. For installed (copied)
    /// packages, use `alc_pkg_remove` instead.
    #[tool(
        name = "alc_pkg_unlink",
        annotations(destructive_hint = true, open_world_hint = false)
    )]
    async fn pkg_unlink(
        &self,
        Parameters(params): Parameters<PkgUnlinkParams>,
    ) -> Result<String, String> {
        self.app.pkg_unlink(params.name).await
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
            .log_view(params.session_id.as_deref(), params.limit, params.max_chars)
            .await
    }

    /// Aggregate usage stats across all logged sessions.
    ///
    /// Returns per-strategy counts, averages, and totals.
    /// Filter by strategy name or time window (last N days).
    #[tool(
        name = "alc_stats",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn stats(&self, Parameters(params): Parameters<StatsParams>) -> Result<String, String> {
        self.app
            .stats(params.strategy.as_deref(), params.days)
            .await
    }

    // ─── Session Status ─────────────────────────────────────────

    /// Query active session status for external observation.
    ///
    /// Without session_id: lists all active (paused) sessions with state,
    /// metrics snapshot, progress, and strategy name.
    /// With session_id: returns detailed status for one session.
    ///
    /// Only shows sessions currently held in the registry (paused, awaiting
    /// host LLM responses). Completed sessions are not listed — use
    /// `alc_log_view` for historical data.
    #[tool(
        name = "alc_status",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn status(&self, Parameters(params): Parameters<StatusParams>) -> Result<String, String> {
        self.app.status(params.session_id.as_deref()).await
    }

    // ─── Project lifecycle ──────────────────────────────────────

    /// Initialize `alc.toml` in the project root.
    ///
    /// Creates a minimal `alc.toml` with an empty `[packages]` section.
    /// Fails if `alc.toml` already exists (no overwrite).
    #[tool(
        name = "alc_init",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn init(&self, Parameters(params): Parameters<InitParams>) -> Result<String, String> {
        self.app.init(params.project_root).await
    }

    /// Re-resolve all `alc.toml` entries and rewrite `alc.lock`.
    ///
    /// Reads `alc.toml`, resolves each package against the installed cache,
    /// and writes a new `alc.lock`. Requires `alc.toml` to exist.
    /// Returns `{ "resolved": N, "errors": [...], "alc_lock": path }`.
    #[tool(name = "alc_update", annotations(open_world_hint = false))]
    async fn update(&self, Parameters(params): Parameters<UpdateParams>) -> Result<String, String> {
        self.app.update(params.project_root).await
    }

    /// Migrate a legacy `alc.lock` to `alc.toml` + new `alc.lock` format.
    ///
    /// Detects legacy format via `linked_at` or `local_dir` fields.
    /// Creates `alc.toml` from `local_dir` entries and renames `alc.lock` → `alc.lock.bak`.
    /// Run `alc_update` afterwards to generate the new `alc.lock`.
    #[tool(name = "alc_migrate", annotations(open_world_hint = false))]
    async fn migrate(
        &self,
        Parameters(params): Parameters<MigrateParams>,
    ) -> Result<String, String> {
        self.app.migrate(params.project_root).await
    }

    // ─── Cards ──────────────────────────────────────────────────

    /// List Card summaries from `~/.algocline/cards/`. Newest-first.
    /// Each row: card_id, pkg, created_at, model, scenario, pass_rate.
    #[tool(
        name = "alc_card_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_list(
        &self,
        Parameters(params): Parameters<CardListParams>,
    ) -> Result<String, String> {
        self.app.card_list(params.pkg).await
    }

    /// Fetch a full Card (all fields) by card_id.
    #[tool(
        name = "alc_card_get",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_get(
        &self,
        Parameters(params): Parameters<CardGetParams>,
    ) -> Result<String, String> {
        self.app.card_get(&params.card_id).await
    }

    /// Filter/sort Cards using the Prisma-style `where` DSL.
    ///
    /// Supports nested-object predicates, reserved operator objects,
    /// `_and`/`_or`/`_not` logical ops, and `order_by` with dotted
    /// paths + optional `-` prefix for descending.  See
    /// `CardFindParams` for the exact shape.
    #[tool(
        name = "alc_card_find",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_find(
        &self,
        Parameters(params): Parameters<CardFindParams>,
    ) -> Result<String, String> {
        self.app
            .card_find(
                params.pkg,
                params.r#where,
                params.order_by,
                params.limit,
                params.offset,
            )
            .await
    }

    /// List aliases from `~/.algocline/cards/_aliases.toml`.
    #[tool(
        name = "alc_card_alias_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_alias_list(
        &self,
        Parameters(params): Parameters<CardAliasListParams>,
    ) -> Result<String, String> {
        self.app.card_alias_list(params.pkg).await
    }

    /// Resolve an alias name to its bound Card and return the full Card JSON.
    /// Shortcut for `alc_card_alias_list` → filter → `alc_card_get`.
    #[tool(
        name = "alc_card_get_by_alias",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_get_by_alias(
        &self,
        Parameters(params): Parameters<CardGetByAliasParams>,
    ) -> Result<String, String> {
        self.app.card_get_by_alias(&params.name).await
    }

    /// Bind (or rebind) an alias to a Card. Aliases are mutable even
    /// though Cards are not.
    #[tool(
        name = "alc_card_alias_set",
        annotations(
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn card_alias_set(
        &self,
        Parameters(params): Parameters<CardAliasSetParams>,
    ) -> Result<String, String> {
        self.app
            .card_alias_set(&params.name, &params.card_id, params.pkg, params.note)
            .await
    }

    /// Append new top-level fields to an existing Card.
    /// Additive only — attempting to overwrite an existing key fails.
    #[tool(
        name = "alc_card_append",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn card_append(
        &self,
        Parameters(params): Parameters<CardAppendParams>,
    ) -> Result<String, String> {
        self.app.card_append(&params.card_id, params.fields).await
    }

    /// Read per-case samples from a Card's sidecar JSONL file.
    /// Returns `[]` when the Card has no samples sidecar.
    /// Use `offset` + `limit` to page through large suites.
    #[tool(
        name = "alc_card_samples",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_samples(
        &self,
        Parameters(params): Parameters<CardSamplesParams>,
    ) -> Result<String, String> {
        self.app
            .card_samples(&params.card_id, params.offset, params.limit)
            .await
    }

    /// Install Cards from a Card Collection (Git repo or local directory).
    ///
    /// A Card Collection has `alc_cards.toml` at root and subdirectories
    /// named after packages, each containing `*.toml` Card files and optional
    /// `*.samples.jsonl` sidecars. Cards are imported into `~/.algocline/cards/{pkg}/`.
    /// Existing Cards with the same id are skipped (immutable, first-writer wins).
    #[tool(
        name = "alc_card_install",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn card_install(
        &self,
        Parameters(params): Parameters<CardInstallParams>,
    ) -> Result<String, String> {
        self.app.card_install(params.url).await
    }

    // ─── Hub ────────────────────────────────────────────────────

    /// Show detailed information for a single package.
    ///
    /// Returns package metadata, all Cards (newest first), aliases,
    /// and aggregated stats (card count, eval count, best pass rate).
    /// Looks up the package in remote indices and local install state.
    #[tool(
        name = "alc_hub_info",
        annotations(read_only_hint = true, open_world_hint = true)
    )]
    async fn hub_info(
        &self,
        Parameters(params): Parameters<HubInfoParams>,
    ) -> Result<String, String> {
        self.app.hub_info(params.pkg).await
    }

    /// Generate a Hub index from a packages directory.
    ///
    /// When `source_dir` is provided, scans that directory directly
    /// (e.g. a repo checkout) for pure metadata extraction — no manifest
    /// or card data mixed in.  When omitted, scans `~/.algocline/packages/`.
    /// Writes the index to `output_path` for CI publishing. Does NOT
    /// touch the remote search cache used by `alc_hub_search`.
    #[tool(
        name = "alc_hub_reindex",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn hub_reindex(
        &self,
        Parameters(params): Parameters<HubReindexParams>,
    ) -> Result<String, String> {
        self.app
            .hub_reindex(params.output_path, params.source_dir)
            .await
    }

    /// Search packages across remote Hub indices and local install state.
    ///
    /// Discovers index URLs from installed package sources and bundled
    /// seeds, fetches each (cached per-source for 1 hour), merges with
    /// locally installed packages/cards, and returns results with
    /// `installed: true/false` for each entry. Use this to discover
    /// available strategies — uninstalled packages can be installed via
    /// `alc_pkg_install` using the `source` URL from the result.
    #[tool(
        name = "alc_hub_search",
        annotations(read_only_hint = true, open_world_hint = true)
    )]
    async fn hub_search(
        &self,
        Parameters(params): Parameters<HubSearchParams>,
    ) -> Result<String, String> {
        self.app
            .hub_search(
                params.query,
                params.category,
                params.installed_only,
                params.limit,
            )
            .await
    }

    // ─── Diagnostics ────────────────────────────────────────────

    /// Show algocline server configuration and diagnostic info.
    ///
    /// Returns resolved log directory (with source), tracing mode,
    /// packages directory, and version. Similar to `mise doctor`.
    #[tool(
        name = "alc_info",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn info(&self) -> Result<String, String> {
        Ok(self.app.info().await)
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
                 - alc_advice: Apply an installed package (ucb, panel, cot, sc, cove, reflect, etc.) to a task. Task is optional — if omitted, opts alone are passed as context.\n\n\
                 When Lua calls alc.llm(prompt), execution pauses and returns the prompt.\n\
                 The host processes it and calls alc_continue with the response to resume.\n\n\
                 Evaluation:\n\
                 - alc_eval: Evaluate a strategy against a scenario. Pass scenario (cases + graders) and strategy name.\n\
                 - alc_eval_history: List past eval results. Filter by strategy, sorted newest-first.\n\
                 - alc_eval_detail: View a specific eval result in full detail.\n\
                 - alc_eval_compare: Compare two eval results with Welch's t-test for statistical significance.\n\n\
                 Scenario Management:\n\
                 - alc_scenario_list: List available scenarios in ~/.algocline/scenarios/.\n\
                 - alc_scenario_show: Show the content of an installed scenario by name.\n\
                 - alc_scenario_install: Install scenarios from a Git URL or local path.\n\n\
                 Package Management:\n\
                 - alc_pkg_link: Link a local directory as a package (symlink to cache). Changes reflect immediately on next alc_run.\n\
                 - alc_pkg_list: List installed packages with metadata. Pass project_root to include project-local packages.\n\
                 - alc_pkg_install: Install a package or collection from a Git URL (e.g. github.com/user/my-pkg).\n\
                 - alc_pkg_remove: Remove a package from alc.toml and alc.lock. Physical files are NOT deleted.\n\
                 - alc_pkg_unlink: Remove a symlinked package from ~/.algocline/packages/. Use pkg_remove for installed packages.\n\
                 - alc_init: Initialize alc.toml in the project root.\n\
                 - alc_update: Re-resolve all alc.toml entries and rewrite alc.lock.\n\
                 - alc_migrate: Migrate legacy alc.lock to alc.toml + new alc.lock format.\n\n\
                 Logging:\n\
                 - alc_note: Add a note to a completed session's log (feedback, observations).\n\
                 - alc_log_view: View session logs. Omit session_id for summary list, provide it for full detail.\n\n\
                 Session Status:\n\
                 - alc_status: Query active session status. Omit session_id to list all, provide it for detail.\n\n\
                 Cards (immutable run snapshots in ~/.algocline/cards/):\n\
                 - alc_card_list: List Card summaries (newest-first). Filter by pkg.\n\
                 - alc_card_get: Fetch a full Card by card_id.\n\
                 - alc_card_find: Filter/sort Cards with a Prisma-style `where` DSL (nested eq/lt/gte/in/_and/_or/_not) and dotted-path `order_by`.\n\
                 - alc_card_alias_list: List aliases from _aliases.toml.\n\
                 - alc_card_get_by_alias: Resolve an alias name to the full Card JSON (shortcut for alias_list → filter → get).\n\
                 - alc_card_alias_set: Bind (or rebind) an alias to a Card.\n\
                 - alc_card_append: Append new top-level fields to a Card (additive-only).\n\
                 - alc_card_samples: Read per-case detail from a Card's {card_id}.samples.jsonl sidecar (auto-emitted by alc_eval auto_card=true).\n\
                 - alc_card_install: Install Cards from a Card Collection repo (Git URL or local path with alc_cards.toml).\n\n\
                 Hub:\n\
                 - alc_hub_search: Search packages across remote Hub indices (auto-discovered from installed sources + collection URL) + local state. Shows installed/uninstalled packages with descriptions and categories. Use source URL with alc_pkg_install to install.\n\
                 - alc_hub_info: Show detailed information for a single package — metadata, all Cards, aliases, and stats (card count, eval count, best pass rate).\n\
                 - alc_hub_reindex: Rebuild the Hub index from locally installed packages. Extracts M.meta from init.lua without Lua VM. Writes to a file for CI publishing.\n\n\
                 Diagnostics:\n\
                 - alc_info: Show server configuration and diagnostic info (log dir, tracing mode, version)."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
