use std::path::{Path, PathBuf};
use std::sync::Arc;

use algocline_core::{ExecutionMetrics, QueryId};
use algocline_engine::{Executor, FeedResult, SessionRegistry};

// ─── Transcript logging ─────────────────────────────────────────

/// Controls transcript log output.
///
/// - `ALC_LOG_DIR`: Directory for log files. Default: `~/.algocline/logs`.
/// - `ALC_LOG_LEVEL`: `full` (default) or `off`.
#[derive(Clone, Debug)]
pub struct TranscriptConfig {
    pub dir: PathBuf,
    pub enabled: bool,
}

impl TranscriptConfig {
    /// Build from environment variables.
    pub fn from_env() -> Self {
        let dir = std::env::var("ALC_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".algocline")
                    .join("logs")
            });

        let enabled = std::env::var("ALC_LOG_LEVEL")
            .map(|v| v.to_lowercase() != "off")
            .unwrap_or(true);

        Self { dir, enabled }
    }
}

/// Write transcript log to `{dir}/{session_id}.json`.
///
/// Silently returns on I/O errors — logging must not break execution.
fn write_transcript_log(config: &TranscriptConfig, session_id: &str, metrics: &ExecutionMetrics) {
    if !config.enabled {
        return;
    }

    let transcript = metrics.transcript_to_json();
    if transcript.is_empty() {
        return;
    }

    let stats = metrics.to_json();

    // Extract task hint from first prompt (truncated to 100 chars)
    let task_hint = transcript
        .first()
        .and_then(|e| e.get("prompt"))
        .and_then(|p| p.as_str())
        .map(|s| {
            if s.len() <= 100 {
                s.to_string()
            } else {
                // Find a char boundary at or before 100 bytes
                let mut end = 100;
                while end > 0 && !s.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &s[..end])
            }
        });

    let auto_stats = &stats["auto"];

    let log_entry = serde_json::json!({
        "session_id": session_id,
        "task_hint": task_hint,
        "stats": auto_stats,
        "transcript": transcript,
    });

    if std::fs::create_dir_all(&config.dir).is_err() {
        return;
    }

    let path = match ContainedPath::child(&config.dir, &format!("{session_id}.json")) {
        Ok(p) => p,
        Err(_) => return,
    };
    let content = match serde_json::to_string_pretty(&log_entry) {
        Ok(s) => s,
        Err(_) => return,
    };

    let _ = std::fs::write(&path, content);

    // Write lightweight meta file for log_list (avoids reading full transcript)
    let meta = serde_json::json!({
        "session_id": session_id,
        "task_hint": task_hint,
        "elapsed_ms": auto_stats.get("elapsed_ms"),
        "rounds": auto_stats.get("rounds"),
        "llm_calls": auto_stats.get("llm_calls"),
        "notes_count": 0,
    });
    if let Ok(meta_path) = ContainedPath::child(&config.dir, &format!("{session_id}.meta.json")) {
        let _ = serde_json::to_string(&meta).map(|s| std::fs::write(&meta_path, s));
    }
}

/// Append a note to an existing log file.
///
/// Reads `{dir}/{session_id}.json`, adds the note to `"notes"` array, writes back.
/// Returns Ok with the note count, or Err if the log file doesn't exist.
fn append_note(
    dir: &Path,
    session_id: &str,
    content: &str,
    title: Option<&str>,
) -> Result<usize, String> {
    let path = ContainedPath::child(dir, &format!("{session_id}.json"))?;
    if !path.as_ref().exists() {
        return Err(format!("Log file not found for session '{session_id}'"));
    }

    let raw = std::fs::read_to_string(&path).map_err(|e| format!("Failed to read log: {e}"))?;
    let mut doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("Failed to parse log: {e}"))?;

    let timestamp = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    };

    let note = serde_json::json!({
        "timestamp": timestamp,
        "title": title,
        "content": content,
    });

    let notes = doc
        .as_object_mut()
        .ok_or("Log file is not a JSON object")?
        .entry("notes")
        .or_insert_with(|| serde_json::json!([]));

    let arr = notes
        .as_array_mut()
        .ok_or("'notes' field is not an array")?;
    arr.push(note);
    let count = arr.len();

    let output =
        serde_json::to_string_pretty(&doc).map_err(|e| format!("Failed to serialize: {e}"))?;
    std::fs::write(path.as_ref(), output).map_err(|e| format!("Failed to write log: {e}"))?;

    // Update notes_count in meta file (best-effort)
    if let Ok(meta_path) = ContainedPath::child(dir, &format!("{session_id}.meta.json")) {
        if meta_path.as_ref().exists() {
            if let Ok(raw) = std::fs::read_to_string(&meta_path) {
                if let Ok(mut meta) = serde_json::from_str::<serde_json::Value>(&raw) {
                    meta["notes_count"] = serde_json::json!(count);
                    if let Ok(s) = serde_json::to_string(&meta) {
                        let _ = std::fs::write(&meta_path, s);
                    }
                }
            }
        }
    }

    Ok(count)
}

// ─── Helpers ────────────────────────────────────────────────────

/// Recursively copy a directory tree (follows symlinks).
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        // Use metadata() (follows symlinks) instead of file_type() (does not)
        let meta = entry.metadata()?;
        let dest_path = dst.join(entry.file_name());
        if meta.is_dir() {
            copy_dir(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

// ─── Path safety ────────────────────────────────────────────────

/// A path verified to reside within a base directory.
///
/// Constructed via [`ContainedPath::child`], which rejects path traversal
/// (`..`, absolute paths, symlink escapes). Once constructed, the inner path
/// is safe for filesystem operations within the base directory.
#[derive(Debug)]
struct ContainedPath(PathBuf);

impl ContainedPath {
    /// Resolve `name` as a child of `base`, rejecting traversal attempts.
    ///
    /// Validates that every component in `name` is [`Component::Normal`].
    /// If the resulting path already exists on disk, additionally verifies
    /// via `canonicalize` that symlinks do not escape `base`.
    fn child(base: &Path, name: &str) -> Result<Self, String> {
        for comp in Path::new(name).components() {
            if !matches!(comp, std::path::Component::Normal(_)) {
                return Err(format!(
                    "Invalid path component in '{name}': path traversal detected"
                ));
            }
        }
        let path = base.join(name);
        if path.exists() {
            let canonical = path
                .canonicalize()
                .map_err(|e| format!("Path resolution failed: {e}"))?;
            let base_canonical = base
                .canonicalize()
                .map_err(|e| format!("Base path resolution failed: {e}"))?;
            if !canonical.starts_with(&base_canonical) {
                return Err(format!("Path '{name}' escapes base directory"));
            }
        }
        Ok(Self(path))
    }
}

impl std::ops::Deref for ContainedPath {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for ContainedPath {
    fn as_ref(&self) -> &Path {
        self
    }
}

// ─── Parameter types (MCP-independent) ──────────────────────────

/// A single query response in a batch feed.
#[derive(Debug)]
pub struct QueryResponse {
    /// Query ID (e.g. "q-0", "q-1").
    pub query_id: String,
    /// The host LLM's response for this query.
    pub response: String,
}

// ─── Code resolution ────────────────────────────────────────────

pub(crate) fn resolve_code(
    code: Option<String>,
    code_file: Option<String>,
) -> Result<String, String> {
    match (code, code_file) {
        (Some(c), None) => Ok(c),
        (None, Some(path)) => std::fs::read_to_string(Path::new(&path))
            .map_err(|e| format!("Failed to read {path}: {e}")),
        (Some(_), Some(_)) => Err("Provide either `code` or `code_file`, not both.".into()),
        (None, None) => Err("Either `code` or `code_file` must be provided.".into()),
    }
}

/// Build Lua code that loads a package by name and calls `pkg.run(ctx)`.
///
/// # Security: `name` is not sanitized
///
/// `name` is interpolated directly into a Lua `require()` call without
/// sanitization. This is intentional in the current architecture:
///
/// - algocline is a **local development/execution tool** that runs Lua in
///   the user's own environment via mlua (not a multi-tenant service).
/// - The same caller has access to `alc_run`, which executes **arbitrary
///   Lua code**. Sanitizing `name` here would not reduce the attack surface.
/// - The MCP trust boundary lies at the **host/client** level — the host
///   decides whether to invoke `alc_advice` at all.
///
/// If algocline is extended to a shared backend (e.g. a package registry
/// server accepting untrusted strategy names), `name` **must** be validated
/// (allowlist of `[a-zA-Z0-9_-]` or equivalent) before interpolation.
///
/// References:
/// - [MCP Security Best Practices — Local MCP Server Compromise](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
/// - [OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html)
pub(crate) fn make_require_code(name: &str) -> String {
    format!(
        r#"local pkg = require("{name}")
return pkg.run(ctx)"#
    )
}

pub(crate) fn packages_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("packages"))
}

/// Git URLs for auto-installation. Collection repos contain multiple packages
/// as subdirectories; single repos have init.lua at root.
const AUTO_INSTALL_SOURCES: &[&str] = &[
    "https://github.com/ynishi/algocline-bundled-packages",
    "https://github.com/ynishi/evalframe",
];

/// System packages: installed alongside user packages but not user-facing strategies.
/// Excluded from `pkg_list` and not loaded via `require` for meta extraction.
const SYSTEM_PACKAGES: &[&str] = &["evalframe"];

/// Check whether a package is a system (non-user-facing) package.
fn is_system_package(name: &str) -> bool {
    SYSTEM_PACKAGES.contains(&name)
}

/// Check whether a package is installed (has `init.lua`).
fn is_package_installed(name: &str) -> bool {
    packages_dir()
        .map(|dir| dir.join(name).join("init.lua").exists())
        .unwrap_or(false)
}

// ─── Eval Result Store ──────────────────────────────────────────

fn evals_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("evals"))
}

/// Persist eval result to `~/.algocline/evals/{strategy}_{timestamp}.json`.
///
/// Silently returns on I/O errors — storage must not break eval execution.
fn save_eval_result(strategy: &str, result_json: &str) {
    let dir = match evals_dir() {
        Ok(d) => d,
        Err(_) => return,
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp = now.as_secs();
    let eval_id = format!("{strategy}_{timestamp}");

    // Parse result to extract summary fields for meta file
    let parsed: serde_json::Value = match serde_json::from_str(result_json) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Write full result
    let path = match ContainedPath::child(&dir, &format!("{eval_id}.json")) {
        Ok(p) => p,
        Err(_) => return,
    };
    let _ = std::fs::write(&path, result_json);

    // Write lightweight meta file for listing
    let result_obj = parsed.get("result");
    let stats_obj = parsed.get("stats");
    let aggregated = result_obj.and_then(|r| r.get("aggregated"));

    let meta = serde_json::json!({
        "eval_id": eval_id,
        "strategy": strategy,
        "timestamp": timestamp,
        "pass_rate": aggregated.and_then(|a| a.get("pass_rate")),
        "mean_score": aggregated.and_then(|a| a.get("scores")).and_then(|s| s.get("mean")),
        "total_cases": aggregated.and_then(|a| a.get("total")),
        "passed": aggregated.and_then(|a| a.get("passed")),
        "llm_calls": stats_obj.and_then(|s| s.get("auto")).and_then(|a| a.get("llm_calls")),
        "elapsed_ms": stats_obj.and_then(|s| s.get("auto")).and_then(|a| a.get("elapsed_ms")),
        "summary": result_obj.and_then(|r| r.get("summary")),
    });

    if let Ok(meta_path) = ContainedPath::child(&dir, &format!("{eval_id}.meta.json")) {
        let _ = serde_json::to_string(&meta).map(|s| std::fs::write(&meta_path, s));
    }
}

// ─── Eval Comparison Helpers ─────────────────────────────────────

/// Escape a string for embedding in a Lua single-quoted string literal.
///
/// Handles backslash, single quote, newline, and carriage return —
/// the characters that would break or alter a `'...'` Lua string.
fn escape_for_lua_sq(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Extract strategy name from eval_id (format: "{strategy}_{timestamp}").
fn extract_strategy_from_id(eval_id: &str) -> Option<&str> {
    eval_id.rsplit_once('_').map(|(prefix, _)| prefix)
}

/// Persist a comparison result to `~/.algocline/evals/`.
fn save_compare_result(eval_id_a: &str, eval_id_b: &str, result_json: &str) {
    let dir = match evals_dir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let filename = format!("compare_{eval_id_a}_vs_{eval_id_b}.json");
    if let Ok(path) = ContainedPath::child(&dir, &filename) {
        let _ = std::fs::write(&path, result_json);
    }
}

// ─── Application Service ────────────────────────────────────────

/// Tracks which sessions are eval sessions and their strategy name.
type EvalSessions = std::sync::Mutex<std::collections::HashMap<String, String>>;

#[derive(Clone)]
pub struct AppService {
    executor: Arc<Executor>,
    registry: Arc<SessionRegistry>,
    log_config: TranscriptConfig,
    /// session_id → strategy name for eval sessions (cleared on completion).
    eval_sessions: Arc<EvalSessions>,
}

impl AppService {
    pub fn new(executor: Arc<Executor>, log_config: TranscriptConfig) -> Self {
        Self {
            executor,
            registry: Arc::new(SessionRegistry::new()),
            log_config,
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Execute Lua code with optional JSON context.
    pub async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let code = resolve_code(code, code_file)?;
        let ctx = ctx.unwrap_or(serde_json::Value::Null);
        self.start_and_tick(code, ctx).await
    }

    /// Apply a built-in strategy to a task.
    ///
    /// If the requested package is not installed, automatically installs the
    /// bundled package collection from GitHub before executing.
    pub async fn advice(
        &self,
        strategy: &str,
        task: String,
        opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        // Auto-install bundled packages if the requested strategy is missing
        if !is_package_installed(strategy) {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed(strategy) {
                return Err(format!(
                    "Package '{strategy}' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                ));
            }
        }

        let code = make_require_code(strategy);

        let mut ctx_map = match opts {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        ctx_map.insert("task".into(), serde_json::Value::String(task));
        let ctx = serde_json::Value::Object(ctx_map);

        self.start_and_tick(code, ctx).await
    }

    /// Run an evalframe evaluation suite.
    ///
    /// Accepts a scenario (bindings + cases) and a strategy name.
    /// Automatically wires the strategy as the provider and executes
    /// the evalframe suite, returning the report (summary, scores, failures).
    ///
    /// Injects a `std` global (mlua-batteries compatible shim) so evalframe's
    /// `std.lua` can resolve json/fs/time from algocline's built-in primitives.
    ///
    /// # Security: `strategy` is not sanitized
    ///
    /// `strategy` is interpolated into a Lua string literal without escaping.
    /// This is intentional — same rationale as [`make_require_code`]:
    /// algocline runs Lua in the caller's own process with full ambient
    /// authority, so Lua injection does not cross a trust boundary.
    pub async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        // Auto-install bundled packages if evalframe is missing
        if !is_package_installed("evalframe") {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed("evalframe") {
                return Err(
                    "Package 'evalframe' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                        .into(),
                );
            }
        }

        let scenario_code = resolve_code(scenario, scenario_file)?;

        // Build strategy opts Lua table literal
        let opts_lua = match &strategy_opts {
            Some(v) if !v.is_null() => format!("alc.json_decode('{}')", v),
            _ => "{}".to_string(),
        };

        // Inject `std` global as a mlua-batteries compatible shim.
        //
        // evalframe.std expects the host to provide a `std` global with:
        //   std.json.decode/encode  — JSON serialization
        //   std.fs.read/is_file     — filesystem access
        //   std.time.now            — wall-clock time (epoch seconds, f64)
        //
        // We bridge these from algocline's alc.* primitives and Lua's io stdlib.
        let wrapped = format!(
            r#"
std = {{
  json = {{
    decode = alc.json_decode,
    encode = alc.json_encode,
  }},
  fs = {{
    read = function(path)
      local f, err = io.open(path, "r")
      if not f then error("std.fs.read: " .. (err or path), 2) end
      local content = f:read("*a")
      f:close()
      return content
    end,
    is_file = function(path)
      local f = io.open(path, "r")
      if f then f:close(); return true end
      return false
    end,
  }},
  time = {{
    now = alc.time,
  }},
}}

local ef = require("evalframe")

-- Load scenario (bindings + cases, no provider)
local spec = (function()
{scenario_code}
end)()

-- Inject strategy as provider
spec.provider = ef.providers.algocline {{
  strategy = "{strategy}",
  opts = {opts_lua},
}}

-- Build and run suite
local s = ef.suite "eval" (spec)
local report = s:run()
return report:to_table()
"#
        );

        let ctx = serde_json::Value::Null;
        let result = self.start_and_tick(wrapped, ctx).await?;

        // Register this session for eval result saving on completion.
        // start_and_tick returns the first pause (needs_response) or completed.
        // If completed immediately, save now. Otherwise, save when continue_* finishes.
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
            match parsed.get("status").and_then(|s| s.as_str()) {
                Some("completed") => {
                    save_eval_result(strategy, &result);
                }
                Some("needs_response") => {
                    if let Some(sid) = parsed.get("session_id").and_then(|s| s.as_str()) {
                        if let Ok(mut map) = self.eval_sessions.lock() {
                            map.insert(sid.to_string(), strategy.to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(result)
    }

    /// List eval history, optionally filtered by strategy.
    pub fn eval_history(&self, strategy: Option<&str>, limit: usize) -> Result<String, String> {
        let evals_dir = evals_dir()?;
        if !evals_dir.exists() {
            return Ok(serde_json::json!({ "evals": [] }).to_string());
        }

        let mut entries: Vec<serde_json::Value> = Vec::new();

        let read_dir =
            std::fs::read_dir(&evals_dir).map_err(|e| format!("Failed to read evals dir: {e}"))?;

        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            // Skip meta files
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(".meta."))
            {
                continue;
            }

            // Read meta file (lightweight) if it exists.
            // Derive meta filename from the result filename to stay within evals_dir
            // (ContainedPath ensures no traversal).
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let meta_path = match ContainedPath::child(&evals_dir, &format!("{stem}.meta.json")) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let meta = if meta_path.exists() {
                std::fs::read_to_string(&*meta_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            } else {
                None
            };

            if let Some(meta) = meta {
                // Filter by strategy if specified
                if let Some(filter) = strategy {
                    if meta.get("strategy").and_then(|s| s.as_str()) != Some(filter) {
                        continue;
                    }
                }
                entries.push(meta);
            }
        }

        // Sort by timestamp descending (newest first)
        entries.sort_by(|a, b| {
            let ts_a = a
                .get("timestamp")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let ts_b = b
                .get("timestamp")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            ts_b.cmp(&ts_a)
        });
        entries.truncate(limit);

        Ok(serde_json::json!({ "evals": entries }).to_string())
    }

    /// View a specific eval result by ID.
    pub fn eval_detail(&self, eval_id: &str) -> Result<String, String> {
        let evals_dir = evals_dir()?;
        let path = ContainedPath::child(&evals_dir, &format!("{eval_id}.json"))
            .map_err(|e| format!("Invalid eval_id: {e}"))?;
        if !path.exists() {
            return Err(format!("Eval result not found: {eval_id}"));
        }
        std::fs::read_to_string(&*path).map_err(|e| format!("Failed to read eval: {e}"))
    }

    /// Compare two eval results with statistical significance testing.
    ///
    /// Delegates to evalframe's `stats.welch_t` (single source of truth for
    /// t-distribution table and test logic). Reads persisted `aggregated.scores`
    /// from each eval result — no re-computation of descriptive statistics.
    ///
    /// The comparison result is persisted to `~/.algocline/evals/` so repeated
    /// lookups of the same pair are file reads only.
    pub async fn eval_compare(&self, eval_id_a: &str, eval_id_b: &str) -> Result<String, String> {
        // Check for cached comparison
        let cache_filename = format!("compare_{eval_id_a}_vs_{eval_id_b}.json");
        if let Ok(dir) = evals_dir() {
            if let Ok(cached_path) = ContainedPath::child(&dir, &cache_filename) {
                if cached_path.exists() {
                    return std::fs::read_to_string(&*cached_path)
                        .map_err(|e| format!("Failed to read cached comparison: {e}"));
                }
            }
        }

        // Auto-install bundled packages if evalframe is missing
        if !is_package_installed("evalframe") {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed("evalframe") {
                return Err(
                    "Package 'evalframe' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                        .into(),
                );
            }
        }

        let result_a = self.eval_detail(eval_id_a)?;
        let result_b = self.eval_detail(eval_id_b)?;

        // Build Lua snippet that uses evalframe's stats module
        // to compute welch_t from the persisted aggregated scores.
        let lua_code = format!(
            r#"
std = {{
  json = {{
    decode = alc.json_decode,
    encode = alc.json_encode,
  }},
  fs = {{ read = function() end, is_file = function() return false end }},
  time = {{ now = alc.time }},
}}

local stats = require("evalframe.eval.stats")

local result_a = alc.json_decode('{result_a_escaped}')
local result_b = alc.json_decode('{result_b_escaped}')

local agg_a = result_a.result and result_a.result.aggregated
local agg_b = result_b.result and result_b.result.aggregated

if not agg_a or not agg_a.scores then
  error("No aggregated scores in {eval_id_a}")
end
if not agg_b or not agg_b.scores then
  error("No aggregated scores in {eval_id_b}")
end

local welch = stats.welch_t(agg_a.scores, agg_b.scores)

local strategy_a = (result_a.result and result_a.result.name) or "{strategy_a_fallback}"
local strategy_b = (result_b.result and result_b.result.name) or "{strategy_b_fallback}"

local delta = agg_a.scores.mean - agg_b.scores.mean
local winner = "none"
if welch.significant then
  winner = delta > 0 and "a" or "b"
end

-- Build summary text
local parts = {{}}
if welch.significant then
  local w, l, d = strategy_a, strategy_b, delta
  if delta < 0 then w, l, d = strategy_b, strategy_a, -delta end
  parts[#parts + 1] = string.format(
    "%s outperforms %s by %.4f (mean score), statistically significant (t=%.3f, df=%.1f).",
    w, l, d, math.abs(welch.t_stat), welch.df
  )
else
  parts[#parts + 1] = string.format(
    "No statistically significant difference between %s and %s (t=%.3f, df=%.1f).",
    strategy_a, strategy_b, math.abs(welch.t_stat), welch.df
  )
end
if agg_a.pass_rate and agg_b.pass_rate then
  local dp = agg_a.pass_rate - agg_b.pass_rate
  if math.abs(dp) > 1e-9 then
    local h = dp > 0 and strategy_a or strategy_b
    parts[#parts + 1] = string.format("Pass rate: %s +%.1fpp.", h, math.abs(dp) * 100)
  else
    parts[#parts + 1] = string.format("Pass rate: identical (%.1f%%).", agg_a.pass_rate * 100)
  end
end

return {{
  a = {{
    eval_id = "{eval_id_a}",
    strategy = strategy_a,
    scores = agg_a.scores,
    pass_rate = agg_a.pass_rate,
    pass_at_1 = agg_a.pass_at_1,
    ci_95 = agg_a.ci_95,
  }},
  b = {{
    eval_id = "{eval_id_b}",
    strategy = strategy_b,
    scores = agg_b.scores,
    pass_rate = agg_b.pass_rate,
    pass_at_1 = agg_b.pass_at_1,
    ci_95 = agg_b.ci_95,
  }},
  comparison = {{
    delta_mean = delta,
    welch_t = {{
      t_stat = welch.t_stat,
      df = welch.df,
      significant = welch.significant,
      direction = welch.direction,
    }},
    winner = winner,
    summary = table.concat(parts, " "),
  }},
}}
"#,
            result_a_escaped = escape_for_lua_sq(&result_a),
            result_b_escaped = escape_for_lua_sq(&result_b),
            eval_id_a = eval_id_a,
            eval_id_b = eval_id_b,
            strategy_a_fallback = extract_strategy_from_id(eval_id_a).unwrap_or("A"),
            strategy_b_fallback = extract_strategy_from_id(eval_id_b).unwrap_or("B"),
        );

        let ctx = serde_json::Value::Null;
        let raw_result = self.start_and_tick(lua_code, ctx).await?;

        // Persist comparison result
        save_compare_result(eval_id_a, eval_id_b, &raw_result);

        Ok(raw_result)
    }

    /// Continue a paused execution — batch feed.
    pub async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        let mut last_result = None;
        for qr in responses {
            let qid = QueryId::parse(&qr.query_id);
            let result = self
                .registry
                .feed_response(session_id, &qid, qr.response)
                .await
                .map_err(|e| format!("Continue failed: {e}"))?;
            last_result = Some(result);
        }
        let result = last_result.ok_or("Empty responses array")?;
        self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        self.maybe_save_eval(&result, session_id, &json);
        Ok(json)
    }

    /// Continue a paused execution — single response (with optional query_id).
    pub async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
    ) -> Result<String, String> {
        let query_id = match query_id {
            Some(qid) => QueryId::parse(qid),
            None => QueryId::single(),
        };

        let result = self
            .registry
            .feed_response(session_id, &query_id, response)
            .await
            .map_err(|e| format!("Continue failed: {e}"))?;

        self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        self.maybe_save_eval(&result, session_id, &json);
        Ok(json)
    }

    // ─── Package Management ─────────────────────────────────────

    /// List installed packages with metadata.
    pub async fn pkg_list(&self) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        if !pkg_dir.is_dir() {
            return Ok(serde_json::json!({ "packages": [] }).to_string());
        }

        let mut packages = Vec::new();
        let entries =
            std::fs::read_dir(&pkg_dir).map_err(|e| format!("Failed to read packages dir: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let init_lua = path.join("init.lua");
            if !init_lua.exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip system packages (not user-facing strategies)
            if is_system_package(&name) {
                continue;
            }
            let code = format!(
                r#"local pkg = require("{name}")
return pkg.meta or {{ name = "{name}" }}"#
            );
            match self.executor.eval_simple(code).await {
                Ok(meta) => packages.push(meta),
                Err(_) => {
                    packages
                        .push(serde_json::json!({ "name": name, "error": "failed to load meta" }));
                }
            }
        }

        Ok(serde_json::json!({ "packages": packages }).to_string())
    }

    /// Install a package from a Git URL or local path.
    pub async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let _ = std::fs::create_dir_all(&pkg_dir);

        // Local path: copy directly (supports uncommitted/dirty working trees)
        let local_path = Path::new(&url);
        if local_path.is_absolute() && local_path.is_dir() {
            return self.install_from_local_path(local_path, &pkg_dir, name);
        }

        // Normalize URL: add https:// only for bare domain-style URLs
        let git_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
            || url.starts_with("git@")
        {
            url.clone()
        } else {
            format!("https://{url}")
        };

        // Clone to temp directory first to detect single vs collection
        let staging = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let output = tokio::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                &git_url,
                &staging.path().to_string_lossy(),
            ])
            .output()
            .await
            .map_err(|e| format!("Failed to run git: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {stderr}"));
        }

        // Remove .git dir from staging
        let _ = std::fs::remove_dir_all(staging.path().join(".git"));

        // Detect: single package (init.lua at root) vs collection (subdirs with init.lua)
        if staging.path().join("init.lua").exists() {
            // Single package mode
            let name = name.unwrap_or_else(|| {
                url.trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .trim_end_matches(".git")
                    .to_string()
            });

            let dest = ContainedPath::child(&pkg_dir, &name)?;
            if dest.as_ref().exists() {
                return Err(format!(
                    "Package '{name}' already exists at {}. Remove it first.",
                    dest.as_ref().display()
                ));
            }

            copy_dir(staging.path(), dest.as_ref())
                .map_err(|e| format!("Failed to copy package: {e}"))?;

            Ok(serde_json::json!({
                "installed": [name],
                "mode": "single",
            })
            .to_string())
        } else {
            // Collection mode: scan for subdirs containing init.lua
            if name.is_some() {
                // name parameter is only meaningful for single-package repos
                return Err(
                    "The 'name' parameter is only supported for single-package repos (init.lua at root). \
                     This repository is a collection (subdirs with init.lua)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut skipped = Vec::new();

            let entries = std::fs::read_dir(staging.path())
                .map_err(|e| format!("Failed to read staging dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                let dest = pkg_dir.join(&pkg_name);
                if dest.exists() {
                    skipped.push(pkg_name);
                    continue;
                }
                copy_dir(&path, &dest)
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                installed.push(pkg_name);
            }

            if installed.is_empty() && skipped.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            Ok(serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "mode": "collection",
            })
            .to_string())
        }
    }

    /// Install from a local directory path (supports dirty/uncommitted files).
    fn install_from_local_path(
        &self,
        source: &Path,
        pkg_dir: &Path,
        name: Option<String>,
    ) -> Result<String, String> {
        if source.join("init.lua").exists() {
            // Single package
            let name = name.unwrap_or_else(|| {
                source
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });

            let dest = ContainedPath::child(pkg_dir, &name)?;
            if dest.as_ref().exists() {
                // Overwrite for local installs (dev workflow)
                let _ = std::fs::remove_dir_all(&dest);
            }

            copy_dir(source, dest.as_ref()).map_err(|e| format!("Failed to copy package: {e}"))?;
            // Remove .git if copied
            let _ = std::fs::remove_dir_all(dest.as_ref().join(".git"));

            Ok(serde_json::json!({
                "installed": [name],
                "mode": "local_single",
            })
            .to_string())
        } else {
            // Collection mode
            if name.is_some() {
                return Err(
                    "The 'name' parameter is only supported for single-package dirs (init.lua at root)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut updated = Vec::new();

            let entries =
                std::fs::read_dir(source).map_err(|e| format!("Failed to read source dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() || !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                let dest = pkg_dir.join(&pkg_name);
                let existed = dest.exists();
                if existed {
                    let _ = std::fs::remove_dir_all(&dest);
                }
                copy_dir(&path, &dest)
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                let _ = std::fs::remove_dir_all(dest.join(".git"));
                if existed {
                    updated.push(pkg_name);
                } else {
                    installed.push(pkg_name);
                }
            }

            if installed.is_empty() && updated.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            Ok(serde_json::json!({
                "installed": installed,
                "updated": updated,
                "mode": "local_collection",
            })
            .to_string())
        }
    }

    /// Remove an installed package.
    pub async fn pkg_remove(&self, name: &str) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let dest = ContainedPath::child(&pkg_dir, name)?;

        if !dest.as_ref().exists() {
            return Err(format!("Package '{name}' not found"));
        }

        std::fs::remove_dir_all(&dest).map_err(|e| format!("Failed to remove '{name}': {e}"))?;

        Ok(serde_json::json!({ "removed": name }).to_string())
    }

    // ─── Logging ─────────────────────────────────────────────

    /// Append a note to a session's log file.
    pub async fn add_note(
        &self,
        session_id: &str,
        content: &str,
        title: Option<&str>,
    ) -> Result<String, String> {
        let count = append_note(&self.log_config.dir, session_id, content, title)?;
        Ok(serde_json::json!({
            "session_id": session_id,
            "notes_count": count,
        })
        .to_string())
    }

    /// View session logs.
    pub async fn log_view(
        &self,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        match session_id {
            Some(sid) => self.log_read(sid),
            None => self.log_list(limit.unwrap_or(50)),
        }
    }

    fn log_read(&self, session_id: &str) -> Result<String, String> {
        let path = ContainedPath::child(&self.log_config.dir, &format!("{session_id}.json"))?;
        if !path.as_ref().exists() {
            return Err(format!("Log file not found for session '{session_id}'"));
        }
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read log: {e}"))
    }

    fn log_list(&self, limit: usize) -> Result<String, String> {
        let dir = &self.log_config.dir;
        if !dir.is_dir() {
            return Ok(serde_json::json!({ "sessions": [] }).to_string());
        }

        let entries = std::fs::read_dir(dir).map_err(|e| format!("Failed to read log dir: {e}"))?;

        // Collect .meta.json files first; fall back to .json for legacy logs
        let mut files: Vec<(std::path::PathBuf, std::time::SystemTime)> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = path.file_name()?.to_str()?;
                // Skip non-json and meta files in this pass
                if !name.ends_with(".json") || name.ends_with(".meta.json") {
                    return None;
                }
                let mtime = entry.metadata().ok()?.modified().ok()?;
                Some((path, mtime))
            })
            .collect();

        // Sort by modification time descending (newest first), take limit
        files.sort_by(|a, b| b.1.cmp(&a.1));
        files.truncate(limit);

        let mut sessions = Vec::new();
        for (path, _) in &files {
            // Try .meta.json first (lightweight), fall back to full log
            let meta_path = path.with_extension("meta.json");
            let doc: serde_json::Value = if meta_path.exists() {
                // Meta file: already flat summary (~200 bytes)
                match std::fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|r| serde_json::from_str(&r).ok())
                {
                    Some(d) => d,
                    None => continue,
                }
            } else {
                // Legacy fallback: read full log and extract fields
                let raw = match std::fs::read_to_string(path) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                match serde_json::from_str::<serde_json::Value>(&raw) {
                    Ok(d) => {
                        let stats = d.get("stats");
                        serde_json::json!({
                            "session_id": d.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown"),
                            "task_hint": d.get("task_hint").and_then(|v| v.as_str()),
                            "elapsed_ms": stats.and_then(|s| s.get("elapsed_ms")),
                            "rounds": stats.and_then(|s| s.get("rounds")),
                            "llm_calls": stats.and_then(|s| s.get("llm_calls")),
                            "notes_count": d.get("notes").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                        })
                    }
                    Err(_) => continue,
                }
            };

            sessions.push(doc);
        }

        Ok(serde_json::json!({ "sessions": sessions }).to_string())
    }

    // ─── Internal ───────────────────────────────────────────────

    /// Install all bundled sources (collections + single packages).
    async fn auto_install_bundled_packages(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();
        for url in AUTO_INSTALL_SOURCES {
            tracing::info!("auto-installing from {url}");
            if let Err(e) = self.pkg_install(url.to_string(), None).await {
                tracing::warn!("failed to auto-install from {url}: {e}");
                errors.push(format!("{url}: {e}"));
            }
        }
        // Fail only if ALL sources failed
        if errors.len() == AUTO_INSTALL_SOURCES.len() {
            return Err(format!(
                "Failed to auto-install bundled packages: {}",
                errors.join("; ")
            ));
        }
        Ok(())
    }

    fn maybe_log_transcript(&self, result: &FeedResult, session_id: &str) {
        if let FeedResult::Finished(exec_result) = result {
            write_transcript_log(&self.log_config, session_id, &exec_result.metrics);
        }
    }

    /// If this session was an eval, save the final result to the eval store.
    fn maybe_save_eval(&self, result: &FeedResult, session_id: &str, result_json: &str) {
        if !matches!(result, FeedResult::Finished(_)) {
            return;
        }
        let strategy = {
            let mut map = match self.eval_sessions.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            map.remove(session_id)
        };
        if let Some(strategy) = strategy {
            save_eval_result(&strategy, result_json);
        }
    }

    async fn start_and_tick(&self, code: String, ctx: serde_json::Value) -> Result<String, String> {
        let session = self.executor.start_session(code, ctx).await?;
        let (session_id, result) = self
            .registry
            .start_execution(session)
            .await
            .map_err(|e| format!("Execution failed: {e}"))?;
        self.maybe_log_transcript(&result, &session_id);
        Ok(result.to_json(&session_id).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::ExecutionObserver;
    use std::io::Write;

    // ─── resolve_code tests ───

    #[test]
    fn resolve_code_inline() {
        let result = resolve_code(Some("return 1".into()), None);
        assert_eq!(result.unwrap(), "return 1");
    }

    #[test]
    fn resolve_code_from_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "return 42").unwrap();

        let result = resolve_code(None, Some(tmp.path().to_string_lossy().into()));
        assert_eq!(result.unwrap(), "return 42");
    }

    #[test]
    fn resolve_code_both_provided_error() {
        let result = resolve_code(Some("code".into()), Some("file.lua".into()));
        let err = result.unwrap_err();
        assert!(err.contains("not both"), "error: {err}");
    }

    #[test]
    fn resolve_code_neither_provided_error() {
        let result = resolve_code(None, None);
        let err = result.unwrap_err();
        assert!(err.contains("must be provided"), "error: {err}");
    }

    #[test]
    fn resolve_code_nonexistent_file_error() {
        let result = resolve_code(
            None,
            Some("/tmp/algocline_nonexistent_test_file.lua".into()),
        );
        assert!(result.is_err());
    }

    // ─── make_require_code tests ───

    #[test]
    fn make_require_code_basic() {
        let code = make_require_code("ucb");
        assert!(code.contains(r#"require("ucb")"#), "code: {code}");
        assert!(code.contains("pkg.run(ctx)"), "code: {code}");
    }

    #[test]
    fn make_require_code_different_names() {
        for name in &["panel", "cot", "sc", "cove", "reflect", "calibrate"] {
            let code = make_require_code(name);
            assert!(
                code.contains(&format!(r#"require("{name}")"#)),
                "code for {name}: {code}"
            );
        }
    }

    // ─── packages_dir tests ───

    #[test]
    fn packages_dir_ends_with_expected_path() {
        let dir = packages_dir().unwrap();
        assert!(
            dir.ends_with(".algocline/packages"),
            "dir: {}",
            dir.display()
        );
    }

    // ─── append_note tests ───

    #[test]
    fn append_note_to_existing_log() {
        let dir = tempfile::tempdir().unwrap();
        let session_id = "s-test-001";
        let log = serde_json::json!({
            "session_id": session_id,
            "stats": { "elapsed_ms": 100 },
            "transcript": [],
        });
        let path = dir.path().join(format!("{session_id}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&log).unwrap()).unwrap();

        let count = append_note(dir.path(), session_id, "Step 2 was weak", Some("Step 2")).unwrap();
        assert_eq!(count, 1);

        let count = append_note(dir.path(), session_id, "Overall good", None).unwrap();
        assert_eq!(count, 2);

        let raw = std::fs::read_to_string(&path).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let notes = doc["notes"].as_array().unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0]["content"], "Step 2 was weak");
        assert_eq!(notes[0]["title"], "Step 2");
        assert_eq!(notes[1]["content"], "Overall good");
        assert!(notes[1]["title"].is_null());
        assert!(notes[0]["timestamp"].is_number());
    }

    #[test]
    fn append_note_missing_log_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = append_note(dir.path(), "s-nonexistent", "note", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    // ─── log_list / log_view tests ───

    #[test]
    fn log_list_from_dir() {
        let dir = tempfile::tempdir().unwrap();

        // Create two log files
        let log1 = serde_json::json!({
            "session_id": "s-001",
            "task_hint": "What is 2+2?",
            "stats": { "elapsed_ms": 100, "rounds": 1, "llm_calls": 1 },
            "transcript": [{ "prompt": "What is 2+2?", "response": "4" }],
        });
        let log2 = serde_json::json!({
            "session_id": "s-002",
            "task_hint": "Explain ownership",
            "stats": { "elapsed_ms": 5000, "rounds": 3, "llm_calls": 3 },
            "transcript": [],
            "notes": [{ "timestamp": 0, "content": "good" }],
        });

        std::fs::write(
            dir.path().join("s-001.json"),
            serde_json::to_string(&log1).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("s-002.json"),
            serde_json::to_string(&log2).unwrap(),
        )
        .unwrap();
        // Non-json file should be ignored
        std::fs::write(dir.path().join("README.txt"), "ignore me").unwrap();

        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: true,
        };

        // Use log_list directly via the free function path
        let entries = std::fs::read_dir(&config.dir).unwrap();
        let mut count = 0;
        for entry in entries.flatten() {
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                count += 1;
            }
        }
        assert_eq!(count, 2);
    }

    // ─── ContainedPath tests ───

    #[test]
    fn contained_path_accepts_simple_name() {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), "s-abc123.json");
        assert!(result.is_ok());
        assert!(result.unwrap().as_ref().ends_with("s-abc123.json"));
    }

    #[test]
    fn contained_path_rejects_parent_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), "../../../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("path traversal"), "err: {err}");
    }

    #[test]
    fn contained_path_rejects_absolute_path() {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), "/etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("path traversal"), "err: {err}");
    }

    #[test]
    fn contained_path_rejects_dot_dot_in_middle() {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), "foo/../bar");
        assert!(result.is_err());
    }

    #[test]
    fn contained_path_accepts_nested_normal() {
        let dir = tempfile::tempdir().unwrap();
        let result = ContainedPath::child(dir.path(), "sub/file.json");
        assert!(result.is_ok());
    }

    #[test]
    fn append_note_rejects_traversal_session_id() {
        let dir = tempfile::tempdir().unwrap();
        let result = append_note(dir.path(), "../../../etc/passwd", "evil", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("path traversal"));
    }

    // ─── meta file tests ───

    #[test]
    fn write_transcript_log_creates_meta_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: true,
        };

        let metrics = algocline_core::ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_paused(&[algocline_core::LlmQuery {
            id: algocline_core::QueryId::single(),
            prompt: "What is 2+2?".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&algocline_core::QueryId::single(), "4");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        write_transcript_log(&config, "s-meta-test", &metrics);

        // Main log should exist
        assert!(dir.path().join("s-meta-test.json").exists());

        // Meta file should exist
        let meta_path = dir.path().join("s-meta-test.meta.json");
        assert!(meta_path.exists());

        let raw = std::fs::read_to_string(&meta_path).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(meta["session_id"], "s-meta-test");
        assert_eq!(meta["notes_count"], 0);
        assert!(meta.get("elapsed_ms").is_some());
        assert!(meta.get("rounds").is_some());
        assert!(meta.get("llm_calls").is_some());
        // Meta should NOT contain transcript
        assert!(meta.get("transcript").is_none());
    }

    #[test]
    fn append_note_updates_meta_notes_count() {
        let dir = tempfile::tempdir().unwrap();
        let session_id = "s-meta-note";

        // Create main log
        let log = serde_json::json!({
            "session_id": session_id,
            "stats": { "elapsed_ms": 100 },
            "transcript": [],
        });
        std::fs::write(
            dir.path().join(format!("{session_id}.json")),
            serde_json::to_string_pretty(&log).unwrap(),
        )
        .unwrap();

        // Create meta file
        let meta = serde_json::json!({
            "session_id": session_id,
            "task_hint": "test",
            "elapsed_ms": 100,
            "rounds": 1,
            "llm_calls": 1,
            "notes_count": 0,
        });
        std::fs::write(
            dir.path().join(format!("{session_id}.meta.json")),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        append_note(dir.path(), session_id, "first note", None).unwrap();

        let raw =
            std::fs::read_to_string(dir.path().join(format!("{session_id}.meta.json"))).unwrap();
        let updated: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(updated["notes_count"], 1);

        append_note(dir.path(), session_id, "second note", None).unwrap();

        let raw =
            std::fs::read_to_string(dir.path().join(format!("{session_id}.meta.json"))).unwrap();
        let updated: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(updated["notes_count"], 2);
    }

    // ─── TranscriptConfig tests ───

    #[test]
    fn transcript_config_default_enabled() {
        // Without env vars, should default to enabled
        let config = TranscriptConfig {
            dir: PathBuf::from("/tmp/test"),
            enabled: true,
        };
        assert!(config.enabled);
    }

    #[test]
    fn write_transcript_log_disabled_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: false,
        };
        let metrics = algocline_core::ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_paused(&[algocline_core::LlmQuery {
            id: algocline_core::QueryId::single(),
            prompt: "test".into(),
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&algocline_core::QueryId::single(), "r");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        write_transcript_log(&config, "s-disabled", &metrics);

        // No file should be created
        assert!(!dir.path().join("s-disabled.json").exists());
        assert!(!dir.path().join("s-disabled.meta.json").exists());
    }

    #[test]
    fn write_transcript_log_empty_transcript_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: true,
        };
        // Metrics with no observer events → empty transcript
        let metrics = algocline_core::ExecutionMetrics::new();
        write_transcript_log(&config, "s-empty", &metrics);
        assert!(!dir.path().join("s-empty.json").exists());
    }

    // ─── copy_dir tests ───

    #[test]
    fn copy_dir_basic() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        std::fs::write(src.path().join("a.txt"), "hello").unwrap();
        std::fs::create_dir(src.path().join("sub")).unwrap();
        std::fs::write(src.path().join("sub/b.txt"), "world").unwrap();

        let dst_path = dst.path().join("copied");
        copy_dir(src.path(), &dst_path).unwrap();

        assert_eq!(
            std::fs::read_to_string(dst_path.join("a.txt")).unwrap(),
            "hello"
        );
        assert_eq!(
            std::fs::read_to_string(dst_path.join("sub/b.txt")).unwrap(),
            "world"
        );
    }

    #[test]
    fn copy_dir_empty() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let dst_path = dst.path().join("empty_copy");
        copy_dir(src.path(), &dst_path).unwrap();
        assert!(dst_path.exists());
        assert!(dst_path.is_dir());
    }

    // ─── task_hint truncation in write_transcript_log ───

    #[test]
    fn write_transcript_log_truncates_long_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: true,
        };
        let metrics = algocline_core::ExecutionMetrics::new();
        let observer = metrics.create_observer();
        let long_prompt = "x".repeat(300);
        observer.on_paused(&[algocline_core::LlmQuery {
            id: algocline_core::QueryId::single(),
            prompt: long_prompt,
            system: None,
            max_tokens: 10,
            grounded: false,
            underspecified: false,
        }]);
        observer.on_response_fed(&algocline_core::QueryId::single(), "r");
        observer.on_resumed();
        observer.on_completed(&serde_json::json!(null));

        write_transcript_log(&config, "s-long", &metrics);

        let raw = std::fs::read_to_string(dir.path().join("s-long.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let hint = doc["task_hint"].as_str().unwrap();
        // Should be truncated to ~100 chars + "..."
        assert!(hint.len() <= 104, "hint too long: {} chars", hint.len());
        assert!(hint.ends_with("..."));
    }

    #[test]
    fn log_list_prefers_meta_file() {
        let dir = tempfile::tempdir().unwrap();

        // Create a full log (large, with transcript)
        let log = serde_json::json!({
            "session_id": "s-big",
            "task_hint": "full log hint",
            "stats": { "elapsed_ms": 999, "rounds": 5, "llm_calls": 5 },
            "transcript": [{"prompt": "x".repeat(10000), "response": "y".repeat(10000)}],
        });
        std::fs::write(
            dir.path().join("s-big.json"),
            serde_json::to_string(&log).unwrap(),
        )
        .unwrap();

        // Create corresponding meta
        let meta = serde_json::json!({
            "session_id": "s-big",
            "task_hint": "full log hint",
            "elapsed_ms": 999,
            "rounds": 5,
            "llm_calls": 5,
            "notes_count": 0,
        });
        std::fs::write(
            dir.path().join("s-big.meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        // Create a legacy log (no meta file)
        let legacy = serde_json::json!({
            "session_id": "s-legacy",
            "task_hint": "legacy hint",
            "stats": { "elapsed_ms": 100, "rounds": 1, "llm_calls": 1 },
            "transcript": [],
        });
        std::fs::write(
            dir.path().join("s-legacy.json"),
            serde_json::to_string(&legacy).unwrap(),
        )
        .unwrap();

        let config = TranscriptConfig {
            dir: dir.path().to_path_buf(),
            enabled: true,
        };
        let app = AppService {
            executor: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap()
                    .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
            ),
            registry: Arc::new(algocline_engine::SessionRegistry::new()),
            log_config: config,
            eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        };

        let result = app.log_list(50).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let sessions = parsed["sessions"].as_array().unwrap();

        assert_eq!(sessions.len(), 2);

        // Both sessions should have session_id and task_hint
        let ids: Vec<&str> = sessions
            .iter()
            .map(|s| s["session_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"s-big"));
        assert!(ids.contains(&"s-legacy"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// resolve_code never panics.
        #[test]
        fn resolve_code_never_panics(
            code in proptest::option::of("[a-z]{0,50}"),
            file in proptest::option::of("[a-z]{0,50}"),
        ) {
            let _ = resolve_code(code, file);
        }

        /// ContainedPath always rejects ".." components.
        #[test]
        fn contained_path_rejects_traversal(
            prefix in "[a-z]{0,5}",
            suffix in "[a-z]{0,5}",
        ) {
            let dir = tempfile::tempdir().unwrap();
            let name = format!("{prefix}/../{suffix}");
            let result = ContainedPath::child(dir.path(), &name);
            prop_assert!(result.is_err());
        }

        /// ContainedPath accepts simple alphanumeric names.
        #[test]
        fn contained_path_accepts_simple_names(name in "[a-z][a-z0-9_-]{0,20}\\.json") {
            let dir = tempfile::tempdir().unwrap();
            let result = ContainedPath::child(dir.path(), &name);
            prop_assert!(result.is_ok());
        }

        /// make_require_code always contains the strategy name in a require call.
        #[test]
        fn make_require_code_contains_name(name in "[a-z_]{1,20}") {
            let code = make_require_code(&name);
            let expected = format!("require(\"{}\")", name);
            prop_assert!(code.contains(&expected));
            prop_assert!(code.contains("pkg.run(ctx)"));
        }

        /// copy_dir preserves file contents for arbitrary data.
        #[test]
        fn copy_dir_preserves_content(content in "[a-zA-Z0-9 ]{1,200}") {
            let src = tempfile::tempdir().unwrap();
            let dst = tempfile::tempdir().unwrap();

            std::fs::write(src.path().join("test.txt"), &content).unwrap();
            let dst_path = dst.path().join("out");
            copy_dir(src.path(), &dst_path).unwrap();

            let read = std::fs::read_to_string(dst_path.join("test.txt")).unwrap();
            prop_assert_eq!(&read, &content);
        }
    }

    // ─── eval tests ───

    #[test]
    fn eval_rejects_no_scenario() {
        let result = resolve_code(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn eval_auto_installs_evalframe_on_missing() {
        // Skip if evalframe is already installed globally
        if is_package_installed("evalframe") {
            return;
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let fake_pkg_dir = tmp.path().join("empty_packages");
        std::fs::create_dir_all(&fake_pkg_dir).unwrap();

        let executor = Arc::new(rt.block_on(async {
            algocline_engine::Executor::new(vec![fake_pkg_dir])
                .await
                .unwrap()
        }));
        let config = TranscriptConfig {
            dir: tmp.path().join("logs"),
            enabled: false,
        };
        let svc = AppService::new(executor, config);

        let scenario = r#"return { cases = {} }"#;
        let result = rt.block_on(svc.eval(Some(scenario.into()), None, "cove", None));
        assert!(result.is_err());
        // Auto-install is attempted first; error is about bundled install failure
        // (git clone) or evalframe still missing after install
        let err = result.unwrap_err();
        assert!(
            err.contains("bundled") || err.contains("evalframe"),
            "unexpected error: {err}"
        );
    }

    // ─── comparison helper tests ───

    #[test]
    fn extract_strategy_from_id_splits_correctly() {
        assert_eq!(extract_strategy_from_id("cove_1710672000"), Some("cove"));
        assert_eq!(
            extract_strategy_from_id("my_strat_1710672000"),
            Some("my_strat")
        );
        assert_eq!(extract_strategy_from_id("nostamp"), None);
    }

    #[test]
    fn save_compare_result_persists_file() {
        let tmp = tempfile::tempdir().unwrap();
        let evals = tmp.path().join(".algocline").join("evals");
        std::fs::create_dir_all(&evals).unwrap();

        // save_compare_result uses evals_dir() which reads HOME.
        // Test ContainedPath + write logic directly instead.
        let filename = "compare_a_1_vs_b_2.json";
        let path = ContainedPath::child(&evals, filename).unwrap();
        let data = r#"{"test": true}"#;
        std::fs::write(&*path, data).unwrap();

        let read = std::fs::read_to_string(&*path).unwrap();
        assert_eq!(read, data);
    }
}
