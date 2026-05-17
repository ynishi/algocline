use rmcp::{
    handler::server::{
        router::tool::ToolRouter, tool::RequestId as RmcpRequestId, wrapper::Parameters,
    },
    model::{
        CancelledNotificationParam, CompleteRequestParams, CompleteResult, CompletionInfo,
        GetPromptRequestParams, GetPromptResult, ListPromptsResult, ListResourceTemplatesResult,
        ListResourcesResult, Meta, PaginatedRequestParams, ReadResourceRequestParams, Reference,
        ServerCapabilities, ServerInfo,
    },
    schemars,
    service::RoleServer,
    service::{NotificationContext, Peer, RequestContext},
    tool, tool_handler, tool_router, ServerHandler,
};
use serde::Deserialize;

use std::sync::Arc;

use algocline_app::{EngineApi, QueryResponse};
use algocline_core::{
    execution::{
        CancelCode, CancelReason, ExecutionService, ResumePayload, SessionId, SessionSpec, SpecKind,
    },
    AppDir,
};
use tokio::task::JoinHandle;

use crate::progress_forwarder::spawn_progress_forwarder;
use crate::req_registry::ReqIdRegistry;

use crate::prompts::PromptCatalog;
use crate::resources::{build_list_resources_result, build_list_templates_result, ResourceCatalog};

// ─── MCP Parameter types (schemars-annotated) ───────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunParams {
    /// Lua source code (inline). Provide either `code` or `code_file`, not both.
    pub code: Option<String>,
    /// Path to a Lua source file. Provide either `code` or `code_file`, not both.
    pub code_file: Option<String>,
    /// Context passed to Lua as the `ctx` global (JSON object).
    #[schemars(with = "Option<serde_json::Map<String, serde_json::Value>>")]
    pub ctx: Option<serde_json::Value>,
    /// Optional absolute path to the project root containing `alc.lock`.
    /// Falls back to `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    pub project_root: Option<String>,
    /// Whether to route this execution through the persistent host-mode pool.
    /// `None` (default) and `false` use the existing in-process path unchanged.
    #[serde(default)]
    pub host_mode: Option<bool>,
}

/// Parameters for `alc_pool_status`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PoolStatusParams {
    /// Optional session ID to restrict status to a single worker.
    /// Omit to return status for all registered workers.
    pub sid: Option<String>,
}

/// Parameters for `alc_pool_stop`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PoolStopParams {
    /// Optional session ID. When provided, sends SIGTERM to that single worker.
    /// Omit to stop all registered workers.
    pub sid: Option<String>,
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
    /// Git URL or local path of a package collection.
    /// (e.g. "github.com/user/my-pkg", "file:///path/to/local/pkg").
    /// The repository must use collection layout: each subdirectory with
    /// an `init.lua` is installed as a separate package.
    pub url: String,
    /// Optional package name hint. Reserved for future use; currently ignored
    /// for collection-mode installs. Defaults to the last segment of the URL.
    pub name: Option<String>,
    /// Overwrite existing packages at dest (default `false`).
    #[serde(default)]
    pub force: Option<bool>,
}

/// Scope selector for `alc_pkg_link`.
///
/// - `global` (default): creates a symlink in `~/.algocline/packages/{name}`.
///   Visible to all projects that share the host's `~/.algocline/` cache.
///   Unix-only (requires `symlink(2)`).
/// - `variant`: records the path in `alc.local.toml` at the project root
///   instead of creating a symlink. Worktree-scoped override (git-ignored,
///   loaded every `alc_run`). Works on all platforms.
///
/// The logical scope name (`variant`) is intentionally decoupled from the
/// physical filename (`alc.local.toml`). Rationale: the filename follows
/// the dotenv `.env.local` convention (machine-specific, gitignored),
/// while the scope name describes its semantic layer (Sub-unit of Repo).
#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PkgLinkScope {
    /// Symlink into `~/.algocline/packages/` (all projects).
    Global,
    /// Append to `alc.local.toml` (worktree-scoped, gitignored).
    Variant,
}

impl PkgLinkScope {
    /// String form passed to the App-layer `EngineApi::pkg_link(scope: Option<String>)`.
    ///
    /// The EngineApi boundary takes `String` rather than this enum so that
    /// `algocline-core` does not depend on `schemars`. This keeps the
    /// schemars-dependent types pinned to the MCP crate.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Variant => "variant",
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgLinkParams {
    /// Absolute or relative path to the directory to link as a package.
    /// Must follow collection layout: subdirectories containing `init.lua`
    /// (each subdirectory becomes one linked package). Single-package mode
    /// (`init.lua` at the directory root) was removed in v0.36.0.
    pub path: String,
    /// Reserved for future use; currently ignored. Package names are derived
    /// from subdirectory names under `path`.
    pub name: Option<String>,
    /// Force overwrite of existing symlinks. Default: false.
    /// Only meaningful when `scope = "global"` — ignored in `"variant"` scope.
    pub force: Option<bool>,
    /// Scope of the link. Default: `"global"`.
    ///
    /// - `"global"`: symlink into `~/.algocline/packages/{name}` (existing behavior).
    /// - `"variant"`: append to `{project_root}/alc.local.toml`
    ///   (worktree-scoped override, gitignored).
    pub scope: Option<PkgLinkScope>,
    /// Optional absolute path to the project root. Used only in `scope = "variant"`.
    /// Falls back to `ALC_PROJECT_ROOT` env or ancestor walk from cwd.
    pub project_root: Option<String>,
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
    /// Maximum number of `packages` entries to return (default: 50).
    ///
    /// - `null` (omit) → default cap 50
    /// - `0`           → **no limit** (return all entries — empty-means-all idiom)
    /// - negative      → clamped to 0, i.e. also "no limit"
    ///
    /// Truncation happens **after** filter/sort, so the highest-priority
    /// entries survive.
    pub limit: Option<i32>,
    /// Sort order — MongoDB-style comma-separated keys with optional
    /// `-` prefix for descending. Examples: `"name"`, `"-installed_at"`,
    /// `"-active,-installed_at"` (the default). Unknown or empty keys
    /// are rejected with an error. Default: `"-active,-installed_at"`
    /// — active packages first (DESC on bool puts `true` before
    /// `false`), then ties broken by newest install.
    pub sort: Option<String>,
    /// Key-value filter applied to each projected entry (after the
    /// project/global merge). Exact equality on each key. Unknown keys
    /// miss (i.e. no entry matches).
    pub filter: Option<serde_json::Value>,
    /// Sparse fieldsets — explicit list of per-entry keys to include
    /// in the output. Unknown keys are silently skipped (JSON:API
    /// convention). **`fields` wins over `verbose` when both are
    /// supplied**, so `fields` is the way to get an exact projection.
    pub fields: Option<Vec<String>>,
    /// Output shape preset. Accepted values: `"summary"` (default) or
    /// `"full"`. Any other value returns an error — there is no silent
    /// fallback. When both `fields` and `verbose` are supplied,
    /// **`fields` wins** and `verbose` is ignored.
    ///
    /// Preset field sets (verbatim — any change here is a semver event,
    /// see "Preset drift policy" below):
    ///
    /// - `"summary"` = `["name", "scope", "version", "active",
    ///   "resolved_source_path", "resolved_source_kind"]`
    /// - `"full"` = summary plus `["install_source", "installed_at",
    ///   "updated_at", "override_paths", "overrides", "linked",
    ///   "link_target", "broken", "path", "source", "source_type",
    ///   "meta", "error"]`
    ///
    /// ### Preset drift policy
    ///
    /// Changing these preset arrays is a semver-relevant action. Adding
    /// a field is a **minor** version bump (existing callers can
    /// safely ignore the new key). Removing a field is a **major**
    /// version bump (output parsers can break). Every such change
    /// MUST be recorded verbatim in `CHANGELOG.md` — listing both the
    /// pre-change and post-change preset contents (see plan.md §3.3.2
    /// and ST3 deliverable).
    ///
    /// ### Projection scope
    ///
    /// `verbose` / `fields` only affect the per-entry objects inside
    /// the top-level `packages` array. Top-level keys (`search_paths`,
    /// `project_root`, `lockfile_path`) are always returned regardless
    /// of the chosen preset or field list.
    pub verbose: Option<String>,
}

/// Scope selector for `alc_pkg_remove`.
///
/// Mirrors `PkgLinkScope`'s snake_case enum pattern so that `scope` in
/// `pkg_*` tools stays consistent at the schema level.
///
/// - `project` (default): remove the package declaration from
///   `{project_root}/alc.toml` + `alc.lock`. Global manifest and cached
///   files are untouched.
/// - `global`: remove the package's entry from the global manifest
///   `~/.algocline/installed.json`. The cached directory under
///   `~/.algocline/packages/{name}/` is **not** deleted (symmetric with
///   the `project` scope's "physical files preserved" policy).
/// - `all`: apply both. Lenient: succeeds if either scope had an entry
///   to remove. Errors only when neither scope has the package.
///
/// History: a `scope` parameter existed in 0.14.0 with different
/// semantics (it deleted the physical cache dir) and was removed in
/// 0.15.0 for safety. The re-introduced parameter is manifest-only —
/// no filesystem destruction — and so reuses the name without the
/// earlier danger.
#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PkgRemoveScope {
    /// Remove from `alc.toml` + `alc.lock` (existing behavior).
    Project,
    /// Remove from `~/.algocline/installed.json` only.
    Global,
    /// Remove from both project and global manifests.
    All,
}

impl PkgRemoveScope {
    /// String form passed to the App-layer `EngineApi::pkg_remove(scope: Option<String>)`.
    ///
    /// The EngineApi boundary takes `String` rather than this enum so that
    /// `algocline-core` does not depend on `schemars`. Same split as
    /// `PkgLinkScope`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Global => "global",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgRemoveParams {
    /// Name of the package to remove.
    pub name: String,
    /// Optional absolute path to project root containing alc.toml.
    /// Falls back to ALC_PROJECT_ROOT env or ancestor walk from cwd.
    /// Required when `scope = "project"` or `"all"`; ignored when
    /// `scope = "global"`.
    pub project_root: Option<String>,
    /// Optional version constraint. When specified, only the alc.lock entry
    /// matching this version is removed. Omit to remove any version.
    /// Has no effect on the global manifest entry (which is version-agnostic).
    pub version: Option<String>,
    /// Scope of the removal. Default: `"project"` (backward-compatible).
    ///
    /// - `"project"`: remove from `alc.toml` + `alc.lock` (existing behavior).
    /// - `"global"`: remove from `~/.algocline/installed.json` only.
    /// - `"all"`: remove from both.
    ///
    /// Physical files in `~/.algocline/packages/{name}/` are never deleted
    /// by any scope.
    pub scope: Option<PkgRemoveScope>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgRepairParams {
    /// Optional name. When omitted, every broken package is inspected.
    pub name: Option<String>,
    /// Optional absolute path to project root for `alc.toml` /
    /// `alc.local.toml` checks. Falls back to ancestor walk from cwd.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgDoctorParams {
    /// Optional name. When omitted, every known package is inspected.
    pub name: Option<String>,
    /// Optional absolute path to project root for `alc.toml` /
    /// `alc.local.toml` checks. Falls back to ancestor walk from cwd.
    pub project_root: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PkgTestParams {
    /// Package name to test. Discovers `*_spec.lua` files under
    /// `<pkg_root>/<spec_dir>/` (default `spec_dir = "spec"`).
    /// Mutually exclusive with `code_file` and `code` — exactly one required.
    pub pkg: Option<String>,
    /// Absolute path to a single `.lua` spec file to run.
    /// Mutually exclusive with `pkg` and `code` — exactly one required.
    pub code_file: Option<String>,
    /// Inline Lua source code containing lspec tests to run.
    /// Mutually exclusive with `pkg` and `code_file` — exactly one required.
    pub code: Option<String>,
    /// Subdirectory within the package root that holds spec files.
    /// Defaults to `"spec"`. Only used when `pkg` is provided.
    pub spec_dir: Option<String>,
    /// Substring filter applied to spec file stems when `pkg` is provided.
    /// For example `"shape"` matches `shape_spec.lua` but not `other_spec.lua`.
    pub filter: Option<String>,
    /// Additional directories prepended to `package.path` inside the Lua VM.
    pub search_paths: Option<Vec<String>>,
    /// Absolute path to the project root for variant-scope package resolution
    /// (`alc.local.toml`). Falls back to ancestor walk from cwd when omitted.
    pub project_root: Option<String>,
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
    #[schemars(with = "Option<serde_json::Map<String, serde_json::Value>>")]
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
    #[schemars(with = "Option<serde_json::Map<String, serde_json::Value>>")]
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
    /// Pending query projection. Accepts either a preset name (string)
    /// or a custom field filter (object).
    ///
    /// Preset names:
    /// - `"meta"`    — `query_id` + `max_tokens`
    /// - `"preview"` — meta + first N chars of prompt (N from env
    ///   `ALC_PROMPT_PREVIEW_CHARS`, default 200)
    /// - `"full"`    — every field including the full prompt (debug)
    ///
    /// Custom object example:
    /// `{ "query_id": true, "prompt": { "mode": "preview", "chars": 500 } }`
    ///
    /// Unknown preset names return an error (no silent fallback). Omit
    /// this field to retain the legacy count-only snapshot.
    #[serde(default)]
    pub pending_filter: Option<serde_json::Value>,
    /// If true, include `conversation_history` (cap=10) in each session
    /// snapshot. Default false to keep the response lightweight for
    /// high-frequency polling (preserves the existing "snapshot without
    /// transcript" contract from `metrics.rs:189`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "If true, include `conversation_history` (cap=10) in each session snapshot. Default false to keep response lightweight for high-frequency polling (preserves the existing 'snapshot without transcript' contract)."
    )]
    pub include_history: Option<bool>,
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
pub struct SessionNewParams {
    /// Optional absolute path to the project root to pin for this MCP
    /// connection. Resolved at activation time using the existing
    /// `project_root` fallback chain (P > E > W). When omitted, the
    /// pin is recorded with `None` and tool calls fall back to the
    /// usual chain.
    pub project_root: Option<String>,
    /// Activation mode. Accepts `"default"` (or omitted) and `"test"`.
    /// `"test"` is a hint for downstream tools to apply stricter
    /// isolation (scenario test runners may scope state under a
    /// per-session subdir). Unknown values return a typed error.
    pub mode: Option<String>,
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
pub struct CardAnalyzeParams {
    /// Card ID to analyze. Host loads `~/.algocline/cards/{pkg}/{id}.toml`
    /// (Tier 1 body) and `{id}.samples.jsonl` (Tier 2 sidecar) and
    /// passes both to the analyzer pkg as the Lua ctx.
    pub card_id: String,
    /// Analyzer package name. Defaults to `"card_analysis"` when
    /// omitted. The package must expose `M.run(ctx) -> ctx` with
    /// `ctx.result` populated. Any installed pkg (bundled, project
    /// variant, or user-installed) of this name will satisfy the
    /// dispatch — the host does not hard-depend on any specific pkg.
    pub pkg: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardSamplesParams {
    /// Card ID whose sidecar samples to read.
    pub card_id: String,
    /// Skip this many **matched** rows. Default 0.
    /// When `where` is set, offset applies to the post-filter stream.
    pub offset: Option<usize>,
    /// Max rows returned. Omit to return everything from `offset`.
    pub limit: Option<usize>,
    /// Prisma-style `where` predicate applied to each sample row.
    /// Same DSL as `alc_card_find.where`, but the row object is the
    /// top-level scope (samples have no section wrapping).
    /// Example: `{ "score": { "lt": 0.5 } }`.
    pub r#where: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CardLineageParams {
    /// Card ID to start the walk from.
    pub card_id: String,
    /// Walk direction: `"up"` (ancestors, default), `"down"` (descendants),
    /// or `"both"`.
    pub direction: Option<String>,
    /// Max traversal depth. Default 10.
    pub depth: Option<usize>,
    /// Include each node's `[stats]` section.  Default true.
    pub include_stats: Option<bool>,
    /// Optional list of accepted `metadata.prior_relation` values.
    /// When set, edges whose relation is not in the list are not followed.
    pub relation_filter: Option<Vec<String>>,
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
pub struct CardSinkBackfillParams {
    /// Subscriber URI (e.g. `file:///path/to/mirror`). Must be registered
    /// via `ALC_CARD_SINKS` at startup.
    pub sink: String,
    /// When true, report what would be pushed without writing to the sink.
    #[serde(default)]
    pub dry_run: bool,
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
pub struct PkgScaffoldParams {
    /// Package name (snake_case recommended).
    ///
    /// Must start with a lowercase ASCII letter followed by zero or more
    /// lowercase ASCII letters, digits, or underscores.  Length ≤ 64.
    pub name: String,
    /// Directory under which `<name>/init.lua` is created.
    /// Defaults to `"."` (the algocline server's current working directory).
    pub target_dir: Option<String>,
    /// Optional category written into `M.meta.category` (uncommented).
    /// When omitted, a commented-out placeholder line is emitted instead.
    pub category: Option<String>,
    /// Optional one-line description written into `M.meta.description` (uncommented).
    /// When omitted, a commented-out placeholder line is emitted instead.
    pub description: Option<String>,
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
pub struct HubGendocParams {
    /// Directory containing the repository (`hub_index.json` plus
    /// the per-package directories referenced by it).
    pub source_dir: String,
    /// Output directory for generated documentation. Defaults to
    /// `{source_dir}/docs`.
    pub out_dir: Option<String>,
    /// Projections to emit. Any subset of
    /// `["hub", "context7", "devin", "lint", "lint_only", "luacats", "narrative", "llms"]`.
    /// Unknown values are rejected with a typed `gendoc:` error.
    /// When omitted, only `narrative/{pkg}.md` + `llms.txt` +
    /// `llms-full.txt` are produced.
    pub projections: Option<Vec<String>>,
    /// Path to a TOML config file (optional). When omitted, the project
    /// root's `alc.toml` is auto-explored and `[hub.context7]` / `[hub.devin]`
    /// sections are used. Core defaults are applied when neither a
    /// `config_path` nor `alc.toml` provides projection config. Passing a
    /// `.lua` file is a typed error (retired in this version).
    pub config_path: Option<String>,
    /// Treat lint errors as a hard failure (equivalent to `--strict`).
    pub lint_strict: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HubDistParams {
    /// Directory containing the repository (package directories).
    pub source_dir: String,
    /// Path to write the generated `hub_index.json` (reindex step).
    /// Callers typically pass `{source_dir}/hub_index.json` so the
    /// subsequent gendoc step can read it back.
    pub output_path: Option<String>,
    /// Output directory for generated docs (see `alc_hub_gendoc`).
    /// Defaults to `{source_dir}/docs`.
    pub out_dir: Option<String>,
    /// Optional preset name expanded by `alc_hub_dist` into primitive
    /// `alc_hub_gendoc` arguments (currently supports `"publish"`).
    pub preset: Option<String>,
    /// Optional project root containing `alc.toml` used to resolve
    /// `[hub.dist.presets.<preset>]` overrides.
    pub project_root: Option<String>,
    /// Projections to emit (see `alc_hub_gendoc`).
    /// Unknown values are rejected with a typed `gendoc:` error.
    pub projections: Option<Vec<String>>,
    /// Path to a TOML config file (optional). When omitted, the project
    /// root's `alc.toml` is auto-explored and `[hub.context7]` / `[hub.devin]`
    /// sections are used. Core defaults are applied when neither a
    /// `config_path` nor `alc.toml` provides projection config. Passing a
    /// `.lua` file is a typed error (retired in this version).
    pub config_path: Option<String>,
    /// Treat lint errors as a hard failure (see `alc_hub_gendoc`).
    pub lint_strict: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HubSearchParams {
    /// Search query (matched against package name, description, category,
    /// and docstring). Omit to list all available packages. When the
    /// query matches **only** the docstring (i.e. not name/description/
    /// category), the corresponding result carries `docstring_matched:
    /// true` so the caller can tell that the hit came from the
    /// full-text channel rather than the primary metadata.
    pub query: Option<String>,
    /// Filter by category (e.g. "reasoning", "aggregation", "synthesis").
    /// Prefer the generic `filter` parameter for new callers; `category`
    /// is kept for backward compatibility and is folded into `filter`
    /// on the server side. If `filter` already contains a `"category"`
    /// key, the explicit `filter` entry wins.
    pub category: Option<String>,
    /// When true, only show locally installed packages.
    /// Prefer the generic `filter` parameter (`filter = {"installed":
    /// true}`) for new callers; `installed_only` is kept for
    /// backward compatibility. If `filter` already contains an
    /// `"installed"` key, the explicit `filter` entry wins.
    pub installed_only: Option<bool>,
    /// Maximum number of results (default: 50).
    ///
    /// - `null` (omit) → default cap 50
    /// - `0`           → **no limit** (return all entries — empty-means-all idiom)
    /// - negative      → clamped to 0, i.e. also "no limit"
    pub limit: Option<i32>,
    /// Sort order — MongoDB-style comma-separated keys with optional
    /// `-` prefix for descending. Examples: `"name"`, `"-installed"`,
    /// `"-installed,name"` (the default). Unknown or empty keys are
    /// rejected with an error. Default: `"-installed,name"` — installed
    /// packages first, then ascending by name.
    pub sort: Option<String>,
    /// Key-value filter applied to each projected result (after
    /// query/category/installed_only). Exact equality on each key.
    /// Unknown keys miss (i.e. no entry matches). When both `filter`
    /// and legacy `category` / `installed_only` are supplied, the
    /// explicit `filter` entry wins on key conflict.
    pub filter: Option<serde_json::Value>,
    /// Sparse fieldsets — explicit list of per-entry keys to include
    /// in the output. Unknown keys are silently skipped (JSON:API
    /// convention). **`fields` wins over `verbose` when both are
    /// supplied**, so `fields` is the way to get an exact projection.
    pub fields: Option<Vec<String>>,
    /// Output shape preset. Accepted values: `"summary"` (default) or
    /// `"full"`. Any other value returns an error — there is no silent
    /// fallback. When both `fields` and `verbose` are supplied,
    /// **`fields` wins** and `verbose` is ignored.
    ///
    /// Preset field sets (verbatim — any change here is a semver event,
    /// see "Preset drift policy" below):
    ///
    /// - `"summary"` = `["name", "version", "description", "category",
    ///   "installed", "docstring_matched"]`
    /// - `"full"` = summary plus `["source", "card_count", "best_card",
    ///   "docstring"]`
    ///
    /// `docstring_matched` is only present in individual result entries
    /// when the query matched docstring and not the primary metadata;
    /// otherwise the key is omitted from that entry.
    ///
    /// ### Preset drift policy
    ///
    /// Changing these preset arrays is a semver-relevant action. Adding
    /// a field is a **minor** version bump (existing callers can
    /// safely ignore the new key). Removing a field is a **major**
    /// version bump (output parsers can break). Every such change
    /// MUST be recorded verbatim in `CHANGELOG.md` — listing both the
    /// pre-change and post-change preset contents (see plan.md §3.3.2
    /// and ST3 deliverable).
    ///
    /// ### Projection scope
    ///
    /// `verbose` / `fields` only affect the per-entry objects inside
    /// the top-level `results` array. Top-level keys (`total`,
    /// `sources`, `warnings`) are always returned regardless of the
    /// chosen preset or field list.
    pub verbose: Option<String>,
    /// Local `hub_index.json` file paths to merge into search results.
    /// Useful for pre-push verification or air-gapped use. Each path
    /// is read as a single `HubIndex` JSON file and its packages are
    /// appended to the remote-fetched results (same dedup as Collection
    /// sources via `seen_names`).
    pub local_indices: Option<Vec<String>>,
}

// ─── V2 MCP Parameter types ──────────────────────────────────────

/// Parameters for `alc_v2_run`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct V2RunParams {
    /// Lua source code to execute.
    pub code: String,
    /// Optional JSON context forwarded to the Lua environment as `ctx`.
    #[schemars(with = "Option<serde_json::Value>")]
    pub ctx: Option<serde_json::Value>,
    /// Optional absolute path to the project root containing `alc.lock`.
    pub project_root: Option<String>,
}

/// Parameters for `alc_v2_state`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct V2StateParams {
    /// Session ID returned by `alc_v2_run`.
    pub session_id: String,
}

/// Parameters for `alc_v2_resume`.
///
/// `payload` is typed as raw `serde_json::Value` because some clients ship
/// the field as a stringified JSON object even though the published schema
/// declares `type: object`. The handler reparses a `Value::String` into a
/// `Value::Object` via [`normalize_stringified_json_object`] before
/// deserializing into [`ResumePayload`]. Conforming clients pay no extra
/// cost — their object passes through the normalize helper unchanged.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct V2ResumeParams {
    /// Session ID returned by `alc_v2_run`.
    pub session_id: String,
    /// Resume payload — must match the pause kind of the session.
    pub payload: serde_json::Value,
}

/// Parameters for `alc_v2_cancel`.
///
/// `reason` mirrors the `payload` decoding strategy of [`V2ResumeParams`]
/// to tolerate clients that stringify the object form.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct V2CancelParams {
    /// Session ID returned by `alc_v2_run`.
    pub session_id: String,
    /// Optional cancellation reason. Defaults to `CancelCode::User` with no detail.
    pub reason: Option<serde_json::Value>,
}

/// Reparse a `Value::String` whose body is itself a JSON object or array
/// into the corresponding `Value::Object` / `Value::Array`. All other shapes
/// (primitive scalars, valid objects/arrays, non-JSON strings) pass through
/// untouched.
///
/// Mirrors `algocline_app::service::run::normalize_stringified_json_object`
/// which is `pub(crate)` to that crate; duplicating the 8-line helper here
/// avoids cross-crate API widening for a wire-layer concern. Same rationale
/// as commit 0154010 (`fix(mcp): auto-decode stringified ctx/opts`).
fn normalize_stringified_json_object(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(ref s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(parsed @ serde_json::Value::Object(_)) => parsed,
            Ok(parsed @ serde_json::Value::Array(_)) => parsed,
            _ => v,
        },
        other => other,
    }
}

// ─── V2 adapter helpers ───────────────────────────────────────────

/// Returns the current time as a Unix timestamp in milliseconds.
///
/// `unwrap_or(0)` is used as a defensive default: the only reachable failure is a
/// system clock set before the Unix epoch, which is not a realistic production
/// scenario.  Returning 0 is preferable to panicking (panic-free invariant).
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Spawn a background task that awaits the terminal state of a session and then
/// removes its entry from the registry (entry-deletion path (a)).
///
/// The `await_terminal` result is intentionally absorbed with `let _ = ...` so that
/// a `NotFound` or `Joined` error does not prevent the registry cleanup.  The
/// `remove_by_session` call is unconditional — this satisfies the
/// `test_terminal_cleanup_task_unconditional_remove` invariant.
fn spawn_terminal_cleanup(
    exec: Arc<dyn ExecutionService>,
    registry: Arc<ReqIdRegistry>,
    sid: SessionId,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(e) = exec.await_terminal(&sid).await {
            tracing::warn!("alc_v2_run terminal-cleanup: await_terminal error for sid={sid}: {e}");
        }
        // Unconditional: always remove even when await_terminal returns Err.
        registry.remove_by_session(&sid).await;
    })
}

// ─── MCP Handler ────────────────────────────────────────────────

#[derive(Clone)]
pub struct AlcService {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    app: Arc<dyn EngineApi>,
    resource_catalog: Arc<ResourceCatalog>,
    prompt_catalog: Arc<PromptCatalog>,
    /// Adapter-exclusive mapping from MCP `RequestId` to [`SessionId`].
    /// Service-layer crates must never reference this field or any wire type it contains.
    execution: Arc<dyn ExecutionService>,
    /// Owned by the adapter; service layer is never exposed to wire identifiers.
    req_registry: Arc<ReqIdRegistry>,
}

#[tool_router]
impl AlcService {
    pub fn new(
        app: Arc<dyn EngineApi>,
        execution: Arc<dyn ExecutionService>,
        app_dir: Arc<AppDir>,
    ) -> Self {
        let resource_catalog = Arc::new(ResourceCatalog::new(app.clone(), app_dir.clone()));
        let prompt_catalog = Arc::new(PromptCatalog::new(app.clone(), app_dir));
        Self {
            tool_router: Self::tool_router(),
            app,
            resource_catalog,
            prompt_catalog,
            execution,
            req_registry: Arc::new(ReqIdRegistry::default()),
        }
    }

    /// Execute Lua code with optional JSON context.
    /// Returns the Lua return value as JSON.
    /// Lua code can call `alc.llm(prompt, opts)` to invoke the Host LLM
    /// via MCP Sampling.
    /// Pass `host_mode: true` to route through the persistent worker pool
    /// (subprocess-isolated, survives MCP restart).
    #[tool(name = "alc_run", annotations(open_world_hint = false))]
    async fn run(&self, Parameters(params): Parameters<RunParams>) -> Result<String, String> {
        self.app
            .run(
                params.code,
                params.code_file,
                params.ctx,
                params.project_root,
                params.host_mode,
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

    /// Link a local directory as a package.
    ///
    /// Two scopes:
    /// - `scope = "global"` (default): creates a symlink from
    ///   `~/.algocline/packages/{name}` to the given path. Changes to files
    ///   in the source directory are reflected immediately on the next `alc_run`.
    ///   Visible to all projects (all worktrees share the same cache).
    /// - `scope = "variant"`: appends a `[packages.{name}]` entry to
    ///   `{project_root}/alc.local.toml`. Worktree-scoped override
    ///   (git-ignored, loaded every `alc_run`). No symlink is created, so
    ///   sibling worktrees are unaffected.
    ///
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
        let scope = params.scope.map(|s| s.as_str().to_string());
        let result = self
            .app
            .pkg_link(
                params.path,
                params.name,
                params.force,
                scope,
                params.project_root,
            )
            .await;
        result
    }

    /// List installed packages with metadata.
    ///
    /// When `project_root` is provided, project-local packages from `alc.lock`
    /// are included alongside global packages. Each package entry includes
    /// `scope` ("project" or "global") and `active` (effective vs shadowed).
    /// Each entry also includes `resolved_source_path` (canonical absolute directory),
    /// `resolved_source_kind` (installed / linked / local_path / bundled; future values may appear),
    /// and `override_paths` (shadowed same-name packages).
    ///
    /// ### List-tool params
    /// - `verbose="summary"` (default) / `"full"` — preset field selector.
    ///   - summary: `name, scope, version, active, resolved_source_path, resolved_source_kind`
    ///   - full: summary + `install_source, installed_at, updated_at, override_paths, overrides, linked, link_target, broken, path, source, source_type, meta, error`
    /// - `fields=["name","scope"]` — explicit field list (wins over `verbose` when both set).
    /// - `sort="-active,-installed_at"` (default) — comma-separated keys, `-` prefix for descending.
    /// - `filter={"scope":"global","active":true}` — key-value exact match.
    /// - `limit=50` (default).
    #[tool(
        name = "alc_pkg_list",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn pkg_list(
        &self,
        Parameters(params): Parameters<PkgListParams>,
    ) -> Result<String, String> {
        self.app
            .pkg_list(
                params.project_root,
                params.limit,
                params.sort,
                params.filter,
                params.fields,
                params.verbose,
            )
            .await
    }

    /// Install a package collection from a Git URL or local path.
    /// Clones the repository into ~/.algocline/packages/{name}/.
    /// Supports: `github.com/user/pkg`, `https://...`, `git@...`,
    /// `file:///absolute/path`, or bare `/absolute/path`.
    /// The repository must use collection layout (`<name>/init.lua` nested under a subdir).
    /// `force` (optional, default false): overwrite existing packages at dest.
    #[tool(
        name = "alc_pkg_install",
        annotations(destructive_hint = true, open_world_hint = true)
    )]
    async fn pkg_install(
        &self,
        Parameters(params): Parameters<PkgInstallParams>,
    ) -> Result<String, String> {
        let result = self
            .app
            .pkg_install(params.url, params.name, params.force)
            .await;
        result
    }

    /// Remove a package entry, scoped by `scope`:
    ///
    /// - `scope = "project"` (default): remove from `alc.toml` + `alc.lock`.
    ///   Requires an `alc.toml` found via `project_root` or ancestor walk.
    /// - `scope = "global"`: remove the entry from `~/.algocline/installed.json`.
    ///   `project_root` is ignored in this scope.
    /// - `scope = "all"`: remove from both. Succeeds if either scope had
    ///   the entry; errors only when neither did.
    ///
    /// Physical files in `~/.algocline/packages/{name}/` are **never** deleted
    /// by any scope. Pass `version` to remove only a specific version from
    /// `alc.lock` (project scope only).
    #[tool(
        name = "alc_pkg_remove",
        annotations(destructive_hint = true, open_world_hint = false)
    )]
    async fn pkg_remove(
        &self,
        Parameters(params): Parameters<PkgRemoveParams>,
    ) -> Result<String, String> {
        let scope = params.scope.map(|s| s.as_str().to_string());
        let result = self
            .app
            .pkg_remove(&params.name, params.project_root, params.version, scope)
            .await;
        result
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
        let result = self.app.pkg_unlink(params.name).await;
        result
    }

    /// Heal broken package state by re-running `pkg_install` for entries whose
    /// installed directory is missing.
    ///
    /// Detects four classes of breakage:
    ///   - installed dir missing (manifest entry exists) — repaired via reinstall
    ///   - global symlink dangling — surfaced as unrepairable (no source-of-truth)
    ///   - `alc.toml` `path = ...` missing — unrepairable, suggests user edit
    ///   - `alc.local.toml` `path = ...` missing — unrepairable, suggests `pkg_unlink`
    ///
    /// Returns JSON with `repaired`, `skipped`, `unrepairable`, `failed`
    /// arrays. Repair is best-effort: per-package outcome is reported
    /// regardless of overall success.
    #[tool(
        name = "alc_pkg_repair",
        annotations(destructive_hint = true, open_world_hint = true)
    )]
    async fn pkg_repair(
        &self,
        Parameters(params): Parameters<PkgRepairParams>,
    ) -> Result<String, String> {
        let result = self.app.pkg_repair(params.name, params.project_root).await;
        result
    }

    /// Diagnose package state without side effects (read-only counterpart
    /// of `alc_pkg_repair`).
    ///
    /// Returns JSON with nine arrays:
    ///   - `healthy` — package directory exists and is reachable
    ///   - `incomplete_pkg` — package dir exists but `init.lua` requires
    ///     sibling submodule files (`pkg.sub`) that are missing; use
    ///     `alc_pkg_install --force` or `alc_pkg_link` to restore
    ///   - `installed_missing` — manifest entry exists but install dir is gone;
    ///     use `alc_pkg_install` to reinstall
    ///   - `symlink_dangling` — symlink target missing; use `alc_pkg_unlink`
    ///   - `path_missing` — `alc.toml` / `alc.local.toml` `path = ...` points
    ///     to a non-existent directory
    ///   - `missing_meta` — installed pkg dir has init.lua but M.meta.name is
    ///     absent/empty; fix by editing init.lua to declare
    ///     `M.meta = { name = "...", version = "..." }`
    ///   - `missing_hub_index` — project_root has 2+ pkg dirs but
    ///     hub_index.json is missing; generate it with
    ///     `alc_hub_reindex --source_dir <project_root>`
    ///   - `spec_missing` — installed pkg has `spec/` dir but contains zero
    ///     `*_spec.lua` files; add at least one or remove `spec/` to opt out
    ///   - `stale_cache` — hub cache file (`~/.algocline/hub_cache/{hash}.json`)
    ///     older than 3600s (CACHE_TTL_SECS); refresh via `alc_hub_search`
    ///
    /// No `pkg_install` calls, no filesystem writes. Safe to invoke freely.
    #[tool(
        name = "alc_pkg_doctor",
        annotations(read_only_hint = true, idempotent_hint = true, open_world_hint = false)
    )]
    async fn pkg_doctor(
        &self,
        Parameters(params): Parameters<PkgDoctorParams>,
    ) -> Result<String, String> {
        self.app.pkg_doctor(params.name, params.project_root).await
    }

    /// Run mlua-lspec tests for a package, a single file, or inline Lua code.
    ///
    /// Exactly one of `pkg`, `code_file`, or `code` must be provided:
    ///
    /// - `pkg` — inspects `<pkg_root>/<spec_dir>/*_spec.lua` (default
    ///   `spec_dir = "spec"`). Requires the package to be installed.
    /// - `code_file` — runs a single `.lua` file at the given absolute path.
    /// - `code` — runs inline Lua source; useful for quick ad-hoc tests.
    ///
    /// Returns JSON `{passed, failed, pending, total, duration_ms,
    /// spec_files: [{path, passed, failed, total, duration_ms,
    /// tests: [{suite, name, passed, pending, error}]}]}`.
    ///
    /// Per-spec-file Lua crashes are absorbed (failed count incremented,
    /// execution continues). Setup failures (VM init, package not found,
    /// zero spec files) are returned as a typed error.
    ///
    /// Typed errors:
    /// - Zero or multiple input sources → `"pkg_test: provide exactly one of
    ///   pkg, code_file, code"`.
    /// - `pkg` not installed → `"pkg_test: package '<name>' not found …"`.
    /// - No spec files found → `"pkg_test: no spec files found in <path> …"`.
    #[tool(
        name = "alc_pkg_test",
        annotations(
            read_only_hint = true,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn pkg_test(
        &self,
        Parameters(params): Parameters<PkgTestParams>,
    ) -> Result<String, String> {
        self.app
            .pkg_test(
                params.pkg,
                params.code_file,
                params.code,
                params.spec_dir,
                params.filter,
                params.search_paths,
                params.project_root,
            )
            .await
    }

    /// Generate a minimal package skeleton at `<target_dir>/<name>/init.lua`.
    ///
    /// Creates `<target_dir>/<name>/` (via `fs::create_dir_all`) and writes a
    /// single `init.lua` containing:
    /// - `M.meta` with `name`, `version = "0.1.0"`, and a pre-filled
    ///   `alc_shapes_compat` range derived from the embedded alc_shapes
    ///   version (e.g. embedded `0.25.1` → `">=0.25.0, <0.26"`).
    ///   Optional `category` / `description` are written as uncommented fields
    ///   when provided; otherwise commented-out placeholder lines are emitted.
    /// - `M.spec.entries.run` stub with commented-out `T.shape` declarations.
    /// - `M.run(ctx)` stub using `alc.llm(prompt)`.
    ///
    /// Typed errors are surfaced via the MCP wire response:
    /// - `NameInvalid` — name empty / too long / wrong character set.
    /// - `AlreadyExists` — `<target_dir>/<name>/init.lua` already present.
    /// - `IoError` — filesystem operation failed.
    #[tool(
        name = "alc_pkg_scaffold",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn pkg_scaffold(
        &self,
        Parameters(params): Parameters<PkgScaffoldParams>,
    ) -> Result<String, String> {
        let result = self
            .app
            .pkg_scaffold(
                params.name,
                params.target_dir,
                params.category,
                params.description,
            )
            .await;
        result
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
        let include_history = params.include_history.unwrap_or(false);
        self.app
            .status(
                params.session_id.as_deref(),
                params.pending_filter,
                include_history,
            )
            .await
    }

    // ─── Project lifecycle ──────────────────────────────────────

    /// Initialize `alc.toml` in the project root.
    ///
    /// Creates a minimal `alc.toml` with an empty `[packages]` section and
    /// ensures `alc.local.toml` is listed in `.gitignore` (creating the
    /// file if absent, appending the entry otherwise). Fails if `alc.toml`
    /// already exists (no overwrite).
    ///
    /// Returns `{ "created", "gitignore_path", "gitignore_updated" }`.
    /// `gitignore_updated=false` means the entry was already present.
    #[tool(
        name = "alc_init",
        annotations(
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
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

    /// Pin a project_root + mode for the current MCP connection (issue #1776627475).
    ///
    /// Optional activation. Without calling this, every tool falls
    /// back to the existing `project_root` chain (P > E > W). After
    /// activation, the chain becomes P > **S** > E > W where S is
    /// the pinned root from this call — solving the
    /// "AI forgets project_root and pollutes global manifest" class
    /// of accident in multi-worktree workflows.
    ///
    /// Mode `"default"` matches legacy resolution. Mode `"test"`
    /// signals that downstream scenario tools should apply stricter
    /// isolation (specific behaviour deferred to those tools).
    ///
    /// Lifetime: pinned for the duration of the MCP connection. No
    /// explicit `alc_session_end` — closing the MCP connection (or
    /// activating again) drops the previous pin.
    #[tool(name = "alc_session_new", annotations(open_world_hint = false))]
    async fn session_new(
        &self,
        Parameters(params): Parameters<SessionNewParams>,
    ) -> Result<String, String> {
        self.app.session_new(params.project_root, params.mode).await
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
    /// Accepts a Prisma-style `where` predicate (same nested-object DSL
    /// as `alc_card_find`, evaluated against each row); `offset` + `limit`
    /// page through the post-filter stream.
    #[tool(
        name = "alc_card_samples",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_samples(
        &self,
        Parameters(params): Parameters<CardSamplesParams>,
    ) -> Result<String, String> {
        self.app
            .card_samples(&params.card_id, params.offset, params.limit, params.r#where)
            .await
    }

    /// Run a Card analyzer package over a single Card.
    ///
    /// The host loads the Card body + samples sidecar and dispatches them
    /// to a Lua analyzer pkg via `require(pkg).run(ctx)`. The pkg builds a
    /// prompt from the failure samples, calls `alc.llm`, and returns
    /// improvement hints. Default pkg name is `"card_analysis"` (overridable
    /// via the `pkg` arg) — an IF promise, not a hard dependency on
    /// bundled-packages.
    ///
    /// ctx (host → pkg, passed to `M.run(ctx)`):
    /// ```jsonc
    /// {
    ///   "card_id": "<string id>",
    ///   "card":    <Card body, same shape as alc_card_get>,
    ///   "samples": [<sidecar rows, same shape as alc_card_samples>]
    /// }
    /// ```
    ///
    /// Result (pkg → host, validated host-side as `CardAnalyzeResult`):
    /// ```jsonc
    /// {
    ///   "pattern":          "<one-line failure pattern summary>",
    ///   "suggested_change": "<concrete prompt or Lua-level change>",
    ///   "confidence":       0.0..=1.0,
    ///   "failure_count":    <int, optional>,
    ///   "sample_count":     <int, optional>
    /// }
    /// ```
    /// Output that fails to deserialize as the typed contract returns a
    /// typed error rather than passing freeform JSON to the caller.
    ///
    /// Sister tool to `alc_advice`: `alc_advice` runs a generic strategy
    /// over a free-form task; `alc_card_analyze` runs an analyzer over a
    /// Card. The Card schema is owned by the host so the pkg only deals
    /// with prompt + `alc.llm` + hint formatting.
    #[tool(name = "alc_card_analyze", annotations(open_world_hint = false))]
    async fn card_analyze(
        &self,
        Parameters(params): Parameters<CardAnalyzeParams>,
    ) -> Result<String, String> {
        self.app.card_analyze(&params.card_id, params.pkg).await
    }

    /// Walk a Card's lineage tree via `metadata.prior_card_id`.
    ///
    /// Follows the `prior_card_id` parent pointer (`direction="up"`, default),
    /// collects descendants (`direction="down"`), or both. Returns nodes,
    /// edges, and a `truncated` flag indicating whether the walk hit the
    /// depth limit.
    #[tool(
        name = "alc_card_lineage",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn card_lineage(
        &self,
        Parameters(params): Parameters<CardLineageParams>,
    ) -> Result<String, String> {
        self.app
            .card_lineage(
                &params.card_id,
                params.direction,
                params.depth,
                params.include_stats,
                params.relation_filter,
            )
            .await
    }

    /// Backfill one subscriber (`sink` URI) with all cards from the
    /// primary store.
    ///
    /// Drift-safe: cards already present on the subscriber are skipped
    /// (never overwritten). The primary store is not touched — only the
    /// subscriber receives writes. Returns a `SinkBackfillReport` JSON
    /// with `pushed`, `skipped`, `failed`, and `pushed_samples` lists.
    #[tool(
        name = "alc_card_sink_backfill",
        annotations(
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false,
        )
    )]
    async fn card_sink_backfill(
        &self,
        Parameters(params): Parameters<CardSinkBackfillParams>,
    ) -> Result<String, String> {
        self.app
            .card_sink_backfill(params.sink, params.dry_run)
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

    /// Generate human-readable documentation artifacts from a hub
    /// index.
    ///
    /// Runs the embedded `gen_docs` pipeline (the same pipeline
    /// previously shipped under `algocline-bundled-packages/tools/`)
    /// against the repository at `source_dir`, which must contain a
    /// fresh `hub_index.json`. Emits `narrative/{pkg}.md`,
    /// `llms.txt`, `llms-full.txt` under `out_dir` (defaulting to
    /// `{source_dir}/docs`), plus optional projections
    /// (`hub` / `context7` / `devin`) and optional lint output.
    ///
    /// Returns a JSON string containing `source_dir`, `out_dir`,
    /// and the captured Lua-side `stdout` / `stderr` for
    /// observability.
    #[tool(
        name = "alc_hub_gendoc",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn hub_gendoc(
        &self,
        Parameters(params): Parameters<HubGendocParams>,
    ) -> Result<String, String> {
        self.app
            .hub_gendoc(
                params.source_dir,
                params.out_dir,
                params.projections,
                params.config_path,
                params.lint_strict,
            )
            .await
    }

    /// Run `alc_hub_reindex` and `alc_hub_gendoc` in sequence.
    ///
    /// This is a facade for hub maintainers who always want the index
    /// and the public docs to move together. The response is a JSON
    /// object `{ "reindex": ..., "gendoc": ..., "preset_catalog_version": ..., "preset?": ... }`
    /// embedding the two underlying tool responses verbatim plus preset
    /// catalog metadata (and an optional `preset` object when `preset`
    /// was requested).
    ///
    /// Error semantics (caller-visible via MCP wire `Err`):
    ///
    /// - If the reindex step fails, `gendoc` is not invoked and the
    ///   error text starts with `dist: reindex failed:`.
    /// - If the gendoc step fails after a successful reindex, the
    ///   error text is `dist: gendoc failed: {inner}\nreindex result
    ///   (succeeded): {json}` so callers see both outcomes. The
    ///   reindex-side side effect (the updated `hub_index.json`) is
    ///   not rolled back.
    #[tool(
        name = "alc_hub_dist",
        annotations(destructive_hint = false, open_world_hint = false)
    )]
    async fn hub_dist(
        &self,
        Parameters(params): Parameters<HubDistParams>,
    ) -> Result<String, String> {
        self.app
            .hub_dist(
                params.source_dir,
                params.output_path,
                params.out_dir,
                params.preset,
                params.project_root,
                params.projections,
                params.config_path,
                params.lint_strict,
            )
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
    ///
    /// ### List-tool params
    /// - `verbose="summary"` (default) / `"full"` — preset field selector.
    ///   - summary: `name, version, description, category, installed, docstring_matched`
    ///   - full: summary + `source, card_count, best_card, docstring`
    /// - `fields=["name","version"]` — explicit field list (wins over `verbose` when both set).
    /// - `sort="-installed,name"` (default) — comma-separated keys, `-` prefix for descending.
    /// - `filter={"category":"reasoning","installed":true}` — key-value exact match.
    ///   Legacy `category` / `installed_only` keep working; when both given, `filter` wins.
    /// - `limit=50` (default).
    ///
    /// `docstring_matched=true` appears only when `query` hits the docstring
    /// alone (missed `name` / `description` / `category`); otherwise the
    /// field is omitted. The internal `docstring` is still searched — to
    /// surface it in output use `verbose="full"` or `fields=["docstring"]`.
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
                params.sort,
                params.filter,
                params.fields,
                params.verbose,
                params.local_indices,
            )
            .await
    }

    // ─── Pool management ────────────────────────────────────────

    /// Scan the pool registry, GC dead workers, and return live sessions.
    ///
    /// Idempotent: calling twice produces the same result when no workers change
    /// state between calls. Does NOT spawn new workers — new workers are spawned
    /// on demand by `alc_run` with `host_mode=true`.
    ///
    /// Returns `{"sessions": [...], "pool_version": "..."}`.
    #[tool(
        name = "alc_pool_ensure",
        annotations(idempotent_hint = true, open_world_hint = false)
    )]
    async fn pool_ensure(&self) -> Result<String, String> {
        self.app.pool_ensure().await
    }

    /// Return pool worker status (registry.json + liveness).
    ///
    /// Uses kill -0 to check each registered worker. When `sid` is provided,
    /// restricts output to that single worker.
    ///
    /// Returns `{"sessions": [{"sid","pid","sock","version","created_at","status"},...], "pool_version":"..."}`.
    #[tool(
        name = "alc_pool_status",
        annotations(
            read_only_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn pool_status(
        &self,
        Parameters(params): Parameters<PoolStatusParams>,
    ) -> Result<String, String> {
        self.app.pool_status(params.sid).await
    }

    /// Send SIGTERM to all pool workers (`sid` omitted) or a single worker.
    ///
    /// Removes stopped entries from registry.json. SIGTERM failures are
    /// returned in the `errors` array rather than dropped silently.
    ///
    /// Returns `{"stopped": [...], "errors": [...]}`.
    #[tool(
        name = "alc_pool_stop",
        annotations(
            destructive_hint = true,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn pool_stop(
        &self,
        Parameters(params): Parameters<PoolStopParams>,
    ) -> Result<String, String> {
        self.app.pool_stop(params.sid).await
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

    // ─── V2 execution tools ───────────────────────────────────────

    /// Spawn a new execution session from Lua code (v2 API).
    ///
    /// Returns a JSON object `{"session_id": "<id>"}`.  The session runs in the
    /// background; use `alc_v2_state` to poll status, `alc_v2_resume` to feed LLM
    /// responses when paused, and `alc_v2_cancel` to cancel.
    ///
    /// When `_meta.progressToken` is present, a progress forwarder task is spawned
    /// and execution events are forwarded as `ProgressNotification` messages.
    #[tool(name = "alc_v2_run", annotations(open_world_hint = false))]
    async fn v2_run(
        &self,
        Parameters(params): Parameters<V2RunParams>,
        RmcpRequestId(req_id): RmcpRequestId,
        meta: Meta,
        peer: Peer<RoleServer>,
    ) -> Result<String, String> {
        let project_root = params.project_root.map(std::path::PathBuf::from);
        let spec = SessionSpec {
            kind: SpecKind::Run { code: params.code },
            project_root,
            ctx: params.ctx,
        };
        let sid = self
            .execution
            .spawn(spec)
            .await
            .map_err(|e| e.to_string())?;

        // Register request_id → session_id mapping for on_cancelled reverse-lookup.
        self.req_registry.insert(req_id, sid.clone()).await;

        // Spawn progress forwarder only when progressToken is present (Crux).
        if let Some(token) = meta.get_progress_token() {
            spawn_progress_forwarder(self.execution.clone(), peer, sid.clone(), token);
        }

        // Spawn terminal-cleanup task: removes registry entry when session finishes.
        spawn_terminal_cleanup(
            self.execution.clone(),
            self.req_registry.clone(),
            sid.clone(),
        );

        Ok(serde_json::json!({"session_id": sid.as_str()}).to_string())
    }

    /// Query the current state of a v2 execution session.
    ///
    /// Returns the `ExecutionState` serialized as JSON.
    #[tool(
        name = "alc_v2_state",
        annotations(read_only_hint = true, open_world_hint = false, idempotent_hint = true)
    )]
    async fn v2_state(
        &self,
        Parameters(params): Parameters<V2StateParams>,
    ) -> Result<String, String> {
        let sid = SessionId::from(params.session_id.as_str());
        let state = self
            .execution
            .state(&sid)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&state).map_err(|e| e.to_string())
    }

    /// Resume a paused v2 execution session by supplying LLM responses.
    ///
    /// The `payload` must match the pause kind of the session (`single` or `batch`).
    /// Returns a `ResumeOutcome` serialized as JSON.
    #[tool(name = "alc_v2_resume", annotations(open_world_hint = false))]
    async fn v2_resume(
        &self,
        Parameters(params): Parameters<V2ResumeParams>,
    ) -> Result<String, String> {
        let sid = SessionId::from(params.session_id.as_str());
        // Tolerate stringified JSON payloads (see V2ResumeParams doc).
        let payload_value = normalize_stringified_json_object(params.payload);
        let payload: ResumePayload =
            serde_json::from_value(payload_value).map_err(|e| format!("invalid payload: {e}"))?;
        let outcome = self
            .execution
            .resume(&sid, payload)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&outcome).map_err(|e| e.to_string())
    }

    /// Request cooperative cancellation of a v2 execution session.
    ///
    /// Idempotent: cancelling a session already in a terminal state returns `{}`.
    /// Returns `{}` on success.
    #[tool(
        name = "alc_v2_cancel",
        annotations(open_world_hint = false, idempotent_hint = true)
    )]
    async fn v2_cancel(
        &self,
        Parameters(params): Parameters<V2CancelParams>,
    ) -> Result<String, String> {
        let sid = SessionId::from(params.session_id.as_str());
        // Tolerate stringified JSON reason (see V2CancelParams doc).
        let reason = match params.reason {
            Some(raw) => {
                let normalized = normalize_stringified_json_object(raw);
                serde_json::from_value::<CancelReason>(normalized)
                    .map_err(|e| format!("invalid reason: {e}"))?
            }
            None => CancelReason {
                code: CancelCode::User,
                detail: None,
                requested_at: now_ms(),
            },
        };
        self.execution
            .cancel(&sid, reason)
            .await
            .map_err(|e| e.to_string())?;
        // Entry-deletion path (b): remove after successful cancel.
        self.req_registry.remove_by_session(&sid).await;
        Ok("{}".to_string())
    }
}

#[tool_handler]
impl ServerHandler for AlcService {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_resources()
            .enable_completions()
            .enable_prompts()
            .build();
        info.instructions = Some(
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
                 - alc_pkg_link: Link a local directory as a package. scope=\"global\" (default) symlinks into ~/.algocline/packages/ (all worktrees); scope=\"variant\" appends to {project_root}/alc.local.toml (worktree-scoped, gitignored).\n\
                 - alc_pkg_list: List installed packages with metadata. Pass project_root to include project-local packages.\n\
                 - alc_pkg_install: Install a package or collection from a Git URL (e.g. github.com/user/my-pkg).\n\
                 - alc_pkg_remove: Remove a package from alc.toml and alc.lock. Physical files are NOT deleted.\n\
                 - alc_pkg_unlink: Remove a symlinked package from ~/.algocline/packages/. Use pkg_remove for installed packages.\n\
                 - alc_pkg_doctor: Diagnose package state (read-only). Returns JSON with healthy/incomplete_pkg/installed_missing/missing_meta/missing_hub_index/path_missing/spec_missing/stale_cache/symlink_dangling buckets. incomplete_pkg fires when init.lua requires sibling submodules that are missing. missing_meta fires when installed pkg dir has init.lua but M.meta.name is absent/empty. missing_hub_index fires when project_root has 2+ pkg dirs but hub_index.json is missing. spec_missing fires when installed pkg has spec/ directory but contains zero *_spec.lua files. stale_cache fires when ~/.algocline/hub_cache/{hash}.json mtime exceeds 3600s. Use pkg_unlink to remove dangling symlinks.\n\
                 - alc_pkg_test: Run mlua-lspec tests against a package's spec directory (pkg), a single file (code_file), or inline code. Returns JSON {passed, failed, pending, total, duration_ms, spec_files[]}. Exactly one of pkg/code_file/code must be provided.\n\
                 - alc_pkg_repair: Heal broken packages — reinstalls entries whose installed dir is missing; surfaces dangling symlinks and missing path = ... declarations as unrepairable with suggestions.\n\
                 - alc_init: Initialize alc.toml in the project root and ensure alc.local.toml is listed in .gitignore.\n\
                 - alc_update: Re-resolve all alc.toml entries and rewrite alc.lock.\n\
                 - alc_migrate: Migrate legacy alc.lock to alc.toml + new alc.lock format.\n\n\
                 Accessing a pkg's Lua source:\n\
                 1. If `resolved_source_path` is set AND your client can access the local filesystem, read files under that absolute directory path directly (e.g., via the Read tool or a filesystem MCP server).\n\
                 2. If `resolved_source_path` is absent OR your client cannot reach the local filesystem, and `remote_download_url` is set (future extension), fetch via that URL.\n\
                 3. `resolved_source_kind` indicates the origin of the path (installed / linked / local_path / bundled; future values may appear — treat unknown kinds as \"unknown, attempt filesystem access\").\n\n\
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
                 - alc_card_samples: Read per-case detail from a Card's {card_id}.samples.jsonl sidecar (auto-emitted by alc_eval auto_card=true). Supports the same `where` DSL as alc_card_find.\n\
                 - alc_card_lineage: Walk a Card's ancestry/descendant tree via metadata.prior_card_id. Direction up/down/both, optional depth + relation_filter.\n\
                 - alc_card_install: Install Cards from a Card Collection repo (Git URL or local path with alc_cards.toml).\n\
                 - alc_card_sink_backfill: Backfill one subscriber (ALC_CARD_SINKS URI) with every Card already in the primary store. Drift-safe: existing Cards on the sink are skipped, never overwritten. Supports dry_run.\n\
                 - alc_card_analyze: Run a Card analyzer pkg over a single Card. Host loads the Card body + samples sidecar and dispatches them to `require(pkg).run(ctx)` (default pkg=`card_analysis`). Sister tool to `alc_advice`: advice runs a generic strategy over a free-form task; card_analyze runs an analyzer over a Card. Returns the pkg's `ctx.result` shape (typically `{ pattern, suggested_change, confidence }`).\n\n\
                 Hub:\n\
                 - alc_hub_search: Search packages across remote Hub indices (auto-discovered from installed sources + collection URL) + local state. Shows installed/uninstalled packages with descriptions and categories. Use source URL with alc_pkg_install to install.\n\
                 - alc_hub_info: Show detailed information for a single package — metadata, all Cards, aliases, and stats (card count, eval count, best pass rate).\n\
                 - alc_hub_reindex: Rebuild the Hub index from locally installed packages. Extracts M.meta from init.lua without Lua VM. Writes to a file for CI publishing.\n\
                 - alc_hub_gendoc: Generate human-readable documentation artifacts (narrative/{pkg}.md, llms.txt, llms-full.txt, optional hub/context7/devin projections) from a hub_index.json. Runs the embedded gen_docs Lua pipeline.\n\
                 - alc_hub_dist: Facade that runs alc_hub_reindex followed by alc_hub_gendoc and returns a composed `{ reindex, gendoc, preset_catalog_version, preset? }` response (optional `preset` expands into primitive gendoc args; `preset_catalog_version` is always included for observability). Fails fast on reindex error; surfaces reindex result in the error text on gendoc failure.\n\n\
                 Diagnostics:\n\
                 - alc_info: Show server configuration and diagnostic info (log dir, tracing mode, version)."
                .into(),
        );
        info
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, rmcp::ErrorData> {
        Ok(build_list_resources_result(&self.resource_catalog))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, rmcp::ErrorData> {
        Ok(build_list_templates_result(&self.resource_catalog))
    }

    async fn read_resource(
        &self,
        req: ReadResourceRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ReadResourceResult, rmcp::ErrorData> {
        self.resource_catalog.read(&req.uri).await
    }

    /// Handle `completion/complete` for `ref/resource` template arguments.
    ///
    /// Dispatches to `ResourceCatalog::complete_resource_arg` when the reference
    /// is `ref/resource`. For `ref/prompt` (algocline has no Prompts capability)
    /// and any future reference types, an empty completion result is returned —
    /// empty completion is a valid MCP response and never causes a client error.
    ///
    /// # Arguments
    ///
    /// * `req` — contains the resource URI template reference and the argument
    ///   name + partial value being completed.
    ///
    /// # Returns
    ///
    /// `Ok(CompleteResult)` always. Engine errors during candidate lookup are
    /// absorbed and result in an empty candidate list rather than an MCP error.
    async fn complete(
        &self,
        req: CompleteRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> Result<CompleteResult, rmcp::ErrorData> {
        let (uri_template, arg_name, prefix) = match &req.r#ref {
            Reference::Resource(resource_ref) => (
                resource_ref.uri.as_str(),
                req.argument.name.as_str(),
                req.argument.value.as_str(),
            ),
            // ref/prompt: Phase 1 declares capability only. Full `ref/prompt`
            // completion is deferred to Phase 1.x — empty response maintained.
            Reference::Prompt(_) => {
                return Ok(CompleteResult::new(CompletionInfo::default()));
            }
        };

        let candidates = self
            .resource_catalog
            .complete_resource_arg(uri_template, arg_name, prefix)
            .await;

        // CompletionInfo::new validates the 100-item cap; from_all already enforced
        // it, so we construct directly. The `total` / `has_more` fields are always
        // set for observability.
        Ok(CompleteResult::new(CompletionInfo {
            values: candidates.values,
            total: Some(candidates.total),
            has_more: Some(candidates.has_more),
        }))
    }

    /// Return all installed packages as MCP prompts.
    ///
    /// Enumerates packages by reading `alc.toml` + `~/.algocline/packages/` on
    /// every request via `EngineApi::pkg_list`. No static or startup-time list
    /// is used (Crux #1 constraint).
    ///
    /// # Arguments
    ///
    /// * `_req` — optional pagination cursor (ignored; all prompts are returned)
    /// * `_cx` — MCP request context (unused)
    ///
    /// # Returns
    ///
    /// `Ok(ListPromptsResult)` with one `Prompt` per installed package, or an
    /// MCP error when the underlying enumeration fails.
    async fn list_prompts(
        &self,
        _req: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, rmcp::ErrorData> {
        let prompts = self.prompt_catalog.list_prompts().await?;
        Ok(ListPromptsResult::with_all_items(prompts))
    }

    /// Return the prompt messages for the named package with `task` substituted.
    ///
    /// Verifies that `name` matches an installed package (enumerating on every
    /// call — Crux #1) and substitutes the `task` argument into the message
    /// template at runtime (Crux #2).
    ///
    /// # Arguments
    ///
    /// * `req` — prompt name and optional `arguments` map (expects `task` key)
    /// * `_cx` — MCP request context (unused)
    ///
    /// # Returns
    ///
    /// `Ok(GetPromptResult)` with one `User`-role text message, or:
    /// * `-32602` (invalid params) when the prompt name is unknown
    /// * `-32603` (internal error) on enumeration failure
    async fn get_prompt(
        &self,
        req: GetPromptRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, rmcp::ErrorData> {
        self.prompt_catalog
            .get_prompt(&req.name, req.arguments.as_ref())
            .await
    }

    /// Handle `notifications/cancelled` from the MCP client.
    ///
    /// Resolves the `request_id` → `SessionId` via the registry and delegates
    /// cancellation to `ExecutionService::cancel` exclusively (Crux:
    /// `on_cancelled reverse-lookup via registry`).  No `JoinHandle::abort()` or
    /// direct channel-close path is used.
    async fn on_cancelled(
        &self,
        notification: CancelledNotificationParam,
        _context: NotificationContext<RoleServer>,
    ) {
        let req_id = notification.request_id;
        let sid = match self.req_registry.lookup(&req_id).await {
            Some(s) => s,
            None => {
                tracing::debug!("on_cancelled: no mapping for request_id {req_id:?}");
                return;
            }
        };
        let reason = CancelReason {
            code: CancelCode::User,
            detail: notification.reason,
            requested_at: now_ms(),
        };
        if let Err(e) = self.execution.cancel(&sid, reason).await {
            tracing::warn!("on_cancelled: cancel failed for sid={sid}: {e}");
        }
        self.req_registry.remove_by_request(&req_id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkg_link_scope_deserializes_snake_case_global() {
        let params: PkgLinkParams = serde_json::from_value(serde_json::json!({
            "path": "/tmp/x",
            "scope": "global",
        }))
        .unwrap();
        assert_eq!(params.scope, Some(PkgLinkScope::Global));
    }

    #[test]
    fn pkg_link_scope_deserializes_snake_case_variant() {
        let params: PkgLinkParams = serde_json::from_value(serde_json::json!({
            "path": "/tmp/x",
            "scope": "variant",
        }))
        .unwrap();
        assert_eq!(params.scope, Some(PkgLinkScope::Variant));
    }

    #[test]
    fn pkg_link_scope_unknown_value_is_rejected_by_schema() {
        // `local` used to be the candidate before scope-matrix.md → variant
        // rename. It must now be a schema error rather than silently accepted.
        let err: Result<PkgLinkParams, _> = serde_json::from_value(serde_json::json!({
            "path": "/tmp/x",
            "scope": "local",
        }));
        assert!(err.is_err(), "expected schema error for unknown scope");
    }

    #[test]
    fn pkg_link_scope_as_str_round_trip() {
        assert_eq!(PkgLinkScope::Global.as_str(), "global");
        assert_eq!(PkgLinkScope::Variant.as_str(), "variant");
    }

    #[test]
    fn pkg_link_params_accepts_project_root_and_optional_scope() {
        // scope omitted → None (App layer will treat as default "global").
        let params: PkgLinkParams = serde_json::from_value(serde_json::json!({
            "path": "/tmp/x",
            "project_root": "/tmp/proj",
        }))
        .unwrap();
        assert!(params.scope.is_none());
        assert_eq!(params.project_root.as_deref(), Some("/tmp/proj"));
    }

    #[test]
    fn pkg_link_scope_enum_exposes_both_variants_in_json_schema() {
        use schemars::schema_for;
        let schema = schema_for!(PkgLinkScope);
        let json = serde_json::to_value(&schema).unwrap();
        let s = json.to_string();
        // snake_case serialization — both variants must appear.
        assert!(s.contains("\"global\""), "schema missing global: {s}");
        assert!(s.contains("\"variant\""), "schema missing variant: {s}");
        // The pre-decision value must not leak into the schema.
        assert!(
            !s.contains("\"local\""),
            "schema unexpectedly contains legacy 'local': {s}"
        );
    }

    // ── PkgRemoveScope / PkgRemoveParams — mirror PkgLinkScope coverage ──

    #[test]
    fn pkg_remove_scope_deserializes_snake_case_project() {
        let params: PkgRemoveParams = serde_json::from_value(serde_json::json!({
            "name": "x",
            "scope": "project",
        }))
        .unwrap();
        assert_eq!(params.scope, Some(PkgRemoveScope::Project));
    }

    #[test]
    fn pkg_remove_scope_deserializes_snake_case_global() {
        let params: PkgRemoveParams = serde_json::from_value(serde_json::json!({
            "name": "x",
            "scope": "global",
        }))
        .unwrap();
        assert_eq!(params.scope, Some(PkgRemoveScope::Global));
    }

    #[test]
    fn pkg_remove_scope_deserializes_snake_case_all() {
        let params: PkgRemoveParams = serde_json::from_value(serde_json::json!({
            "name": "x",
            "scope": "all",
        }))
        .unwrap();
        assert_eq!(params.scope, Some(PkgRemoveScope::All));
    }

    #[test]
    fn pkg_remove_scope_unknown_value_is_rejected_by_schema() {
        let err: Result<PkgRemoveParams, _> = serde_json::from_value(serde_json::json!({
            "name": "x",
            "scope": "everywhere",
        }));
        assert!(err.is_err(), "expected schema error for unknown scope");
    }

    #[test]
    fn pkg_remove_scope_as_str_round_trip() {
        assert_eq!(PkgRemoveScope::Project.as_str(), "project");
        assert_eq!(PkgRemoveScope::Global.as_str(), "global");
        assert_eq!(PkgRemoveScope::All.as_str(), "all");
    }

    #[test]
    fn pkg_remove_params_accepts_name_only_for_backcompat() {
        // scope omitted → None (App layer treats as default "project").
        // Back-compat guard: callers written against the pre-scope schema
        // must keep deserializing unchanged.
        let params: PkgRemoveParams =
            serde_json::from_value(serde_json::json!({ "name": "x" })).unwrap();
        assert!(params.scope.is_none());
        assert!(params.project_root.is_none());
        assert!(params.version.is_none());
    }

    #[test]
    fn pkg_remove_scope_enum_exposes_all_variants_in_json_schema() {
        use schemars::schema_for;
        let schema = schema_for!(PkgRemoveScope);
        let json = serde_json::to_value(&schema).unwrap();
        let s = json.to_string();
        assert!(s.contains("\"project\""), "schema missing project: {s}");
        assert!(s.contains("\"global\""), "schema missing global: {s}");
        assert!(s.contains("\"all\""), "schema missing all: {s}");
    }

    // ── V2 adapter tests ──────────────────────────────────────────

    /// Verifies that the terminal-cleanup background task unconditionally removes the
    /// registry entry even when `await_terminal` returns `Err(AwaitError::NotFound)`.
    ///
    /// This is the direct invariant gate for the registry-entry-leak risk
    /// (plan.md §Risks: "registry entry of the leak").
    ///
    /// Uses `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]` as required by
    /// concurrency-analysis.md §2 `test_terminal_cleanup_task_unconditional_remove`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn terminal_cleanup_task_unconditional_remove() {
        use algocline_core::execution::state::ExecutionState;
        use algocline_core::execution::{
            AwaitError, CancelError, CancelReason, ExecutionService, ObserveError, ObserverHandle,
            ResumeError, ResumeOutcome, ResumePayload, SessionId, SessionSpec, SpawnError,
            StateError, TerminalOutcome,
        };
        use std::sync::Arc;

        // Mock ExecutionService whose await_terminal always returns Err(NotFound).
        struct MockExecErrTerminal;

        #[async_trait::async_trait]
        impl ExecutionService for MockExecErrTerminal {
            async fn spawn(&self, _spec: SessionSpec) -> Result<SessionId, SpawnError> {
                unimplemented!()
            }

            async fn state(&self, _id: &SessionId) -> Result<ExecutionState, StateError> {
                unimplemented!()
            }

            async fn resume(
                &self,
                _id: &SessionId,
                _payload: ResumePayload,
            ) -> Result<ResumeOutcome, ResumeError> {
                unimplemented!()
            }

            async fn cancel(
                &self,
                _id: &SessionId,
                _reason: CancelReason,
            ) -> Result<(), CancelError> {
                unimplemented!()
            }

            fn observe(&self, _id: &SessionId) -> Result<Box<dyn ObserverHandle>, ObserveError> {
                unimplemented!()
            }

            async fn await_terminal(&self, id: &SessionId) -> Result<TerminalOutcome, AwaitError> {
                Err(AwaitError::NotFound(id.clone()))
            }
        }

        let exec: Arc<dyn ExecutionService> = Arc::new(MockExecErrTerminal);
        let registry = Arc::new(ReqIdRegistry::default());
        let sid = SessionId::new("cleanup-test-sid".to_string());
        let req_id = crate::req_registry::RequestId::Number(99);

        // Pre-populate the registry.
        registry.insert(req_id.clone(), sid.clone()).await;
        assert_eq!(
            registry.lookup(&req_id).await,
            Some(sid.clone()),
            "entry must be present before cleanup"
        );

        // Run the cleanup task — await_terminal will Err immediately.
        let handle = spawn_terminal_cleanup(exec, registry.clone(), sid.clone());
        handle.await.expect("cleanup task panicked");

        // Registry entry must be removed despite the await_terminal error.
        assert_eq!(
            registry.lookup(&req_id).await,
            None,
            "registry entry must be removed even when await_terminal returns Err"
        );
    }
}
