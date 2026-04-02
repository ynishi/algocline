//! Layer 0: Runtime Primitives
//!
//! Registers Rust-backed functions into the `alc.*` Lua namespace.
//! These provide capabilities that cannot be expressed in Pure Lua:
//! I/O (state), serialization (json), host communication (llm),
//! and text processing (chunk).
//!
//! All functions registered here are available in every Lua session
//! without explicit `require()`.

use std::path::PathBuf;

use algocline_core::{BudgetHandle, CustomMetricsHandle, ProgressHandle, QueryId};
use mlua::prelude::*;
use mlua::LuaSerdeExt;

mod fork;

use crate::llm_bridge::{LlmRequest, QueryRequest};
use crate::state;

/// Layer 1 prelude (also used by fork to setup child VMs).
const PRELUDE: &str = include_str!("../prelude.lua");

/// All handles needed by Layer 0 runtime primitives.
///
/// Collects the various per-session handles into a single config,
/// avoiding a growing parameter list on `register()`.
pub struct BridgeConfig {
    /// Channel for LLM requests (None for eval_simple sessions).
    pub llm_tx: Option<tokio::sync::mpsc::Sender<LlmRequest>>,
    /// Namespace for alc.state (from ctx._ns or "default").
    pub ns: String,
    /// Custom metrics handle for alc.stats.record/get.
    pub custom_metrics: CustomMetricsHandle,
    /// Budget checker for LLM call limits.
    pub budget: BudgetHandle,
    /// Progress reporter for alc.progress().
    pub progress: ProgressHandle,
    /// Package search paths (needed by alc.fork to setup child VMs).
    pub lib_paths: Vec<PathBuf>,
}

/// Register all Layer 0 runtime primitives onto the given table.
pub fn register(lua: &Lua, alc_table: &LuaTable, config: BridgeConfig) -> LuaResult<()> {
    register_json(lua, alc_table)?;
    register_fuzzy(lua, alc_table)?;
    register_log(lua, alc_table)?;
    register_state(lua, alc_table, config.ns)?;
    register_chunk(lua, alc_table)?;
    register_stats(lua, alc_table, config.custom_metrics)?;
    register_time(lua, alc_table)?;
    register_budget_remaining(lua, alc_table, config.budget.clone())?;
    register_progress(lua, alc_table, config.progress)?;
    if let Some(tx) = config.llm_tx {
        register_llm(lua, alc_table, tx.clone(), config.budget.clone())?;
        register_llm_batch(lua, alc_table, tx.clone(), config.budget.clone())?;
        fork::register_fork(lua, alc_table, tx, config.budget, config.lib_paths)?;
    }
    Ok(())
}

/// Register `alc.time()` — wall-clock time in fractional seconds.
///
/// Lua usage:
///   local start = alc.time()
///   -- ... work ...
///   local elapsed_secs = alc.time() - start
///
/// Returns: f64 seconds since Unix epoch (sub-millisecond precision).
fn register_time(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let time_fn = lua.create_function(|_, ()| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(mlua::Error::external)?;
        Ok(now.as_secs_f64())
    })?;
    alc_table.set("time", time_fn)?;
    Ok(())
}

/// Register `alc.chunk(text, opts?)` — split text into chunks.
///
/// Lua usage:
///   local chunks = alc.chunk(text, { mode = "lines", size = 50 })
///   local chunks = alc.chunk(text, { mode = "lines", size = 50, overlap = 10 })
///   local chunks = alc.chunk(text, { mode = "chars", size = 2000 })
///
/// Returns: array of strings.
fn register_chunk(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let chunk_fn = _lua.create_function(|lua, (text, opts): (String, Option<LuaTable>)| {
        let mode = opts
            .as_ref()
            .and_then(|o| o.get::<String>("mode").ok())
            .unwrap_or_else(|| "lines".into());
        let size = opts
            .as_ref()
            .and_then(|o| o.get::<usize>("size").ok())
            .unwrap_or(50);
        let overlap = opts
            .as_ref()
            .and_then(|o| o.get::<usize>("overlap").ok())
            .unwrap_or(0);

        let chunks: Vec<String> = match mode.as_str() {
            "chars" => chunk_by_chars(&text, size, overlap),
            _ => chunk_by_lines(&text, size, overlap),
        };

        lua.to_value(&chunks)
    })?;

    alc_table.set("chunk", chunk_fn)?;
    Ok(())
}

fn chunk_by_lines(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || size == 0 {
        return vec![];
    }
    let step = if overlap < size { size - overlap } else { 1 };
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let end = (i + size).min(lines.len());
        chunks.push(lines[i..end].join("\n"));
        i += step;
        if end == lines.len() {
            break;
        }
    }
    chunks
}

fn chunk_by_chars(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || size == 0 {
        return vec![];
    }
    let step = if overlap < size { size - overlap } else { 1 };
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + size).min(chars.len());
        chunks.push(chars[i..end].iter().collect());
        i += step;
        if end == chars.len() {
            break;
        }
    }
    chunks
}

fn register_json(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let encode = lua.create_function(|lua, value: LuaValue| {
        let json: serde_json::Value = lua.from_value(value)?;
        serde_json::to_string(&json).map_err(LuaError::external)
    })?;

    let decode = lua.create_function(|lua, s: String| {
        let value: serde_json::Value = serde_json::from_str(&s).map_err(LuaError::external)?;
        lua.to_value(&value)
    })?;

    alc_table.set("json_encode", encode)?;
    alc_table.set("json_decode", decode)?;
    Ok(())
}

fn register_log(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let log = _lua.create_function(|_, (level, msg): (String, String)| {
        match level.as_str() {
            "error" => tracing::error!("{}", msg),
            "warn" => tracing::warn!("{}", msg),
            "info" => tracing::info!("{}", msg),
            "debug" => tracing::debug!("{}", msg),
            _ => tracing::info!("{}", msg),
        }
        Ok(())
    })?;

    alc_table.set("log", log)?;
    Ok(())
}

/// Register `alc.state` table with get/set/keys/delete.
///
/// Lua usage:
///   alc.state.set("score", 42)
///   local v = alc.state.get("score")       -- 42
///   local v = alc.state.get("missing", 0)  -- 0 (default)
///   local k = alc.state.keys()             -- {"score"}
///   alc.state.delete("score")
fn register_state(lua: &Lua, alc_table: &LuaTable, ns: String) -> LuaResult<()> {
    let state_table = lua.create_table()?;

    // alc.state.get(key, default?)
    let ns_get = ns.clone();
    let get =
        lua.create_function(
            move |lua, (key, default): (String, Option<LuaValue>)| match state::get(&ns_get, &key) {
                Ok(Some(v)) => lua.to_value(&v),
                Ok(None) => Ok(default.unwrap_or(LuaValue::Nil)),
                Err(e) => Err(LuaError::external(e)),
            },
        )?;

    // alc.state.set(key, value)
    let ns_set = ns.clone();
    let set = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        state::set(&ns_set, &key, json).map_err(LuaError::external)
    })?;

    // alc.state.keys()
    let ns_keys = ns.clone();
    let keys = lua.create_function(move |lua, ()| {
        let k = state::keys(&ns_keys).map_err(LuaError::external)?;
        lua.to_value(&k)
    })?;

    // alc.state.delete(key)
    let ns_del = ns.clone();
    let delete = lua.create_function(move |_, key: String| {
        state::delete(&ns_del, &key).map_err(LuaError::external)
    })?;

    state_table.set("get", get)?;
    state_table.set("set", set)?;
    state_table.set("keys", keys)?;
    state_table.set("delete", delete)?;

    alc_table.set("state", state_table)?;
    Ok(())
}

/// Register `alc.stats` table with record/get.
///
/// Lua usage:
///   alc.stats.record("accuracy", 0.95)
///   local v = alc.stats.get("accuracy")  -- 0.95
fn register_stats(
    lua: &Lua,
    alc_table: &LuaTable,
    custom_metrics: CustomMetricsHandle,
) -> LuaResult<()> {
    let stats_table = lua.create_table()?;

    // alc.stats.record(key, value)
    let cm_record = custom_metrics.clone();
    let record = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        cm_record.record(key, json);
        Ok(())
    })?;

    // alc.stats.get(key)
    let cm_get = custom_metrics;
    let get = lua.create_function(move |lua, key: String| match cm_get.get(&key) {
        Some(v) => lua.to_value(&v),
        None => Ok(LuaValue::Nil),
    })?;

    stats_table.set("record", record)?;
    stats_table.set("get", get)?;

    alc_table.set("stats", stats_table)?;
    Ok(())
}

/// Register `alc.llm(prompt, opts?)` — calls Host LLM via coroutine yield.
///
/// Registered as an async function so the Lua coroutine yields while
/// waiting for the LLM response, allowing other coroutines to progress.
///
/// Lua usage:
///   local response = alc.llm("What is 2+2?")
///   local response = alc.llm("Explain X", { system = "You are an expert.", max_tokens = 500 })
fn register_llm(
    lua: &Lua,
    alc_table: &LuaTable,
    llm_tx: tokio::sync::mpsc::Sender<LlmRequest>,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let llm = lua.create_async_function(move |_, (prompt, opts): (String, Option<LuaTable>)| {
        let tx = llm_tx.clone();
        let bh = budget.clone();
        async move {
            bh.check().map_err(LuaError::external)?;
            let system = opts.as_ref().and_then(|o| o.get::<String>("system").ok());
            let max_tokens = opts
                .as_ref()
                .and_then(|o| o.get::<u32>("max_tokens").ok())
                .unwrap_or(1024);
            let grounded = opts
                .as_ref()
                .and_then(|o| o.get::<bool>("grounded").ok())
                .unwrap_or(false);
            let underspecified = opts
                .as_ref()
                .and_then(|o| o.get::<bool>("underspecified").ok())
                .unwrap_or(false);

            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();

            tx.send(LlmRequest {
                queries: vec![QueryRequest {
                    id: QueryId::single(),
                    prompt,
                    system,
                    max_tokens,
                    grounded,
                    underspecified,
                    resp_tx,
                }],
            })
            .await
            .map_err(|e| LuaError::external(format!("LLM bridge send failed: {e}")))?;

            resp_rx
                .await
                .map_err(|e| LuaError::external(format!("LLM bridge recv failed: {e}")))?
                .map_err(LuaError::external)
        }
    })?;

    alc_table.set("llm", llm)?;
    Ok(())
}

/// Register `alc.llm_batch(items)` — parallel LLM calls via coroutine yield.
///
/// All queries are sent as a single batch, then the coroutine yields
/// while awaiting all responses concurrently.
///
/// Lua usage:
///   local responses = alc.llm_batch({
///       { prompt = "Analyze A" },
///       { prompt = "Analyze B", system = "expert", max_tokens = 500 },
///   })
///   -- responses[1], responses[2] in same order as input
/// Register `alc.budget_remaining()` — query remaining budget.
///
/// Lua return type:
/// - `nil` if no budget was set (ctx.budget absent)
/// - `{ llm_calls = N|nil, elapsed_ms = N|nil }` where each field is
///   present only if the corresponding limit was set. Values are
///   remaining capacity (saturating at 0).
fn register_budget_remaining(
    lua: &Lua,
    alc_table: &LuaTable,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let budget_fn = lua.create_function(move |lua, ()| {
        let remaining = budget.remaining();
        lua.to_value(&remaining)
    })?;
    alc_table.set("budget_remaining", budget_fn)?;
    Ok(())
}

/// Register `alc.progress(step, total, msg?)` — report structured progress.
///
/// Writes progress info into SessionStatus, readable via `alc_status` MCP tool.
/// Not all strategies need to call this — it is opt-in for strategies that
/// benefit from structured step tracking.
///
/// Lua usage:
///   alc.progress(1, 5, "Analyzing chunk 1")
///   alc.progress(2, 5)  -- message is optional
fn register_progress(lua: &Lua, alc_table: &LuaTable, progress: ProgressHandle) -> LuaResult<()> {
    let progress_fn =
        lua.create_function(move |_, (step, total, msg): (u64, u64, Option<String>)| {
            progress.set(step, total, msg);
            Ok(())
        })?;
    alc_table.set("progress", progress_fn)?;
    Ok(())
}

fn register_llm_batch(
    lua: &Lua,
    alc_table: &LuaTable,
    llm_tx: tokio::sync::mpsc::Sender<LlmRequest>,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let llm_batch = lua.create_async_function(move |_, items: LuaTable| {
        let tx = llm_tx.clone();
        let bh = budget.clone();
        async move {
            bh.check().map_err(LuaError::external)?;
            let len = items.len()? as usize;
            if len == 0 {
                return Err(LuaError::external("alc.llm_batch: empty items array"));
            }

            let mut query_requests = Vec::with_capacity(len);
            let mut resp_rxs = Vec::with_capacity(len);

            for i in 1..=len {
                let item: LuaTable = items.get(i)?;
                let prompt: String = item.get("prompt")?;
                let system: Option<String> = item.get::<LuaValue>("system").ok().and_then(|v| {
                    if let LuaValue::String(s) = v {
                        Some(s.to_str().ok()?.to_string())
                    } else {
                        None
                    }
                });
                let max_tokens: u32 = item.get::<u32>("max_tokens").unwrap_or(1024);
                let grounded: bool = item.get::<bool>("grounded").unwrap_or(false);
                let underspecified: bool = item.get::<bool>("underspecified").unwrap_or(false);

                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                resp_rxs.push(resp_rx);

                query_requests.push(QueryRequest {
                    id: QueryId::batch(i - 1), // 0-indexed
                    prompt,
                    system,
                    max_tokens,
                    grounded,
                    underspecified,
                    resp_tx,
                });
            }

            // Send all queries as a single batch
            tx.send(LlmRequest {
                queries: query_requests,
            })
            .await
            .map_err(|e| LuaError::external(format!("LLM bridge send failed: {e}")))?;

            // Await all responses concurrently (order matches input)
            let mut responses = Vec::with_capacity(len);
            for (i, rx) in resp_rxs.into_iter().enumerate() {
                let resp = rx
                    .await
                    .map_err(|e| {
                        LuaError::external(format!("LLM bridge recv failed for q-{i}: {e}"))
                    })?
                    .map_err(LuaError::external)?;
                responses.push(resp);
            }

            Ok(responses)
        }
    })?;

    alc_table.set("llm_batch", llm_batch)?;
    Ok(())
}

/// Register `alc.match_enum(text, candidates, opts?)` — fuzzy enum matcher for LLM output.
///
/// Finds which candidate string appears in `text` (case-insensitive substring match).
/// If multiple candidates match, returns the one whose last occurrence is latest
/// (LLMs tend to state conclusions last).
/// If no substring match, falls back to fuzzy matching via `fuzzy_parser::distance`.
///
/// Lua usage:
///   local verdict = alc.match_enum(response, {"PASS", "BLOCKED"})
///   -- returns "PASS", "BLOCKED", or nil
///
/// opts (optional table):
///   threshold: minimum similarity for fuzzy fallback (default 0.7)
fn register_fuzzy(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let match_enum = _lua.create_function(
        |_, (text, candidates, opts): (String, Vec<String>, Option<LuaTable>)| {
            let threshold = opts
                .as_ref()
                .and_then(|t| t.get::<f64>("threshold").ok())
                .unwrap_or(0.7);

            let text_lower = text.to_lowercase();

            // Phase 1: case-insensitive substring match.
            // If multiple candidates match, pick the one whose last occurrence is latest.
            let mut best: Option<(usize, &str)> = None; // (last_pos, candidate)
            for c in &candidates {
                let c_lower = c.to_lowercase();
                if let Some(pos) = text_lower.rfind(&c_lower) {
                    match best {
                        Some((prev_pos, _)) if pos > prev_pos => best = Some((pos, c)),
                        None => best = Some((pos, c)),
                        _ => {}
                    }
                }
            }
            if let Some((_, matched)) = best {
                return Ok(Some(matched.to_string()));
            }

            // Phase 2: fuzzy fallback — split text into words, compare each
            // word against candidates. Jaro-Winkler is designed for short strings,
            // so per-word comparison is more effective than whole-text comparison.
            let candidates_lower: Vec<String> =
                candidates.iter().map(|c| c.to_lowercase()).collect();
            let mut best_match: Option<(f64, usize)> = None; // (similarity, candidate_index)
            for token in text_lower.split_whitespace() {
                // Strip surrounding punctuation from the token for cleaner matching.
                let token = token.trim_matches(|c: char| !c.is_alphanumeric());
                if token.is_empty() {
                    continue;
                }
                for (i, cl) in candidates_lower.iter().enumerate() {
                    let sim = fuzzy_parser::distance::similarity(
                        token,
                        cl,
                        fuzzy_parser::distance::Algorithm::JaroWinkler,
                    );
                    if sim >= threshold {
                        match best_match {
                            Some((prev_sim, _)) if sim > prev_sim => {
                                best_match = Some((sim, i));
                            }
                            None => best_match = Some((sim, i)),
                            _ => {}
                        }
                    }
                }
            }
            if let Some((_, idx)) = best_match {
                return Ok(Some(candidates[idx].clone()));
            }

            Ok(None)
        },
    )?;

    alc_table.set("match_enum", match_enum)?;

    // alc.match_bool(text) -> true | false | nil
    //
    // Normalizes yes/no-style LLM responses.
    // Scans for affirmative/negative keywords (case-insensitive substring).
    // Returns the polarity of the last-occurring keyword, or nil if ambiguous/absent.
    //
    // Lua usage:
    //   local ok = alc.match_bool("Approved. The plan looks good.")  -- true
    //   local ok = alc.match_bool("rejected: missing tests")         -- false
    //   local ok = alc.match_bool("I need more information")         -- nil
    let match_bool = _lua.create_function(|_, text: String| {
        const TRUE_WORDS: &[&str] = &[
            "approved", "yes", "ok", "accept", "pass", "confirm", "agree", "true", "lgtm",
        ];
        const FALSE_WORDS: &[&str] = &[
            "rejected", "no", "deny", "block", "fail", "refuse", "disagree", "false",
        ];

        let text_lower = text.to_lowercase();
        let bytes = text_lower.as_bytes();

        // Check that the character at the given byte position is not alphanumeric (ASCII).
        // Returns true if pos is out of bounds or the character is a word boundary.
        let is_boundary =
            |pos: usize| -> bool { pos >= bytes.len() || !bytes[pos].is_ascii_alphanumeric() };

        // Find the last whole-word occurrence of any keyword from either group.
        let mut last_pos: Option<(usize, bool)> = None; // (pos, is_true)
        for word in TRUE_WORDS {
            // Scan all occurrences (rfind only gives the last, but we need boundary check)
            let w = word.as_bytes();
            let mut start = 0;
            while let Some(rel) = text_lower[start..].find(word) {
                let pos = start + rel;
                let before_ok = pos == 0 || is_boundary(pos - 1);
                let after_ok = is_boundary(pos + w.len());
                if before_ok && after_ok {
                    match last_pos {
                        Some((prev, _)) if pos > prev => last_pos = Some((pos, true)),
                        None => last_pos = Some((pos, true)),
                        _ => {}
                    }
                }
                start = pos + 1;
            }
        }
        for word in FALSE_WORDS {
            let w = word.as_bytes();
            let mut start = 0;
            while let Some(rel) = text_lower[start..].find(word) {
                let pos = start + rel;
                let before_ok = pos == 0 || is_boundary(pos - 1);
                let after_ok = is_boundary(pos + w.len());
                if before_ok && after_ok {
                    match last_pos {
                        Some((prev, _)) if pos > prev => last_pos = Some((pos, false)),
                        None => last_pos = Some((pos, false)),
                        _ => {}
                    }
                }
                start = pos + 1;
            }
        }

        Ok(last_pos.map(|(_, v)| v))
    })?;

    alc_table.set("match_bool", match_bool)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use algocline_core::ExecutionMetrics;

    fn test_config() -> BridgeConfig {
        let metrics = ExecutionMetrics::new();
        BridgeConfig {
            llm_tx: None,
            ns: "default".into(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
        }
    }

    fn test_config_with_ns(ns: &str) -> BridgeConfig {
        let metrics = ExecutionMetrics::new();
        BridgeConfig {
            llm_tx: None,
            ns: ns.into(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
        }
    }

    #[test]
    fn json_roundtrip() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.json_encode({hello = "world", n = 42})"#)
            .eval()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["hello"], "world");
        assert_eq!(parsed["n"], 42);
    }

    #[test]
    fn json_decode_encode() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(
                r#"
                local val = alc.json_decode('{"a":1,"b":"two"}')
                val.c = true
                return alc.json_encode(val)
            "#,
            )
            .eval()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["a"], 1);
        assert_eq!(parsed["b"], "two");
        assert_eq!(parsed["c"], true);
    }

    #[test]
    fn state_get_set() {
        let ns = "_test_bridge_state";
        // Clean up
        let _ = crate::state::delete(ns, "x");

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config_with_ns(ns)).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Set and get
        lua.load(r#"alc.state.set("x", 99)"#).exec().unwrap();
        let result: i64 = lua.load(r#"return alc.state.get("x")"#).eval().unwrap();
        assert_eq!(result, 99);

        // Default value
        let result: i64 = lua
            .load(r#"return alc.state.get("missing", 0)"#)
            .eval()
            .unwrap();
        assert_eq!(result, 0);

        // Nil for missing without default
        let result: LuaValue = lua
            .load(r#"return alc.state.get("missing")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());

        // Clean up
        let _ = crate::state::delete(ns, "x");
    }

    #[test]
    fn stats_record_get() {
        let metrics = ExecutionMetrics::new();
        let custom_handle = metrics.custom_metrics_handle();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(
            &lua,
            &t,
            BridgeConfig {
                llm_tx: None,
                ns: "default".into(),
                custom_metrics: custom_handle.clone(),
                budget: metrics.budget_handle(),
                progress: metrics.progress_handle(),
                lib_paths: vec![],
            },
        )
        .unwrap();
        lua.globals().set("alc", t).unwrap();

        // Record from Lua
        lua.load(r#"alc.stats.record("score", 42)"#).exec().unwrap();
        let result: i64 = lua.load(r#"return alc.stats.get("score")"#).eval().unwrap();
        assert_eq!(result, 42);

        // Verify via Handle
        assert_eq!(custom_handle.get("score"), Some(serde_json::json!(42)));

        // Missing key returns nil
        let result: LuaValue = lua
            .load(r#"return alc.stats.get("missing")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    // ─── chunk_by_lines tests ───

    #[test]
    fn chunk_lines_empty_text() {
        assert_eq!(chunk_by_lines("", 5, 0), Vec::<String>::new());
    }

    #[test]
    fn chunk_lines_single_line_exact_size() {
        let result = chunk_by_lines("hello", 1, 0);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn chunk_lines_single_line_size_larger() {
        let result = chunk_by_lines("hello", 10, 0);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn chunk_lines_exact_division() {
        let text = "a\nb\nc\nd";
        let result = chunk_by_lines(text, 2, 0);
        assert_eq!(result, vec!["a\nb", "c\nd"]);
    }

    #[test]
    fn chunk_lines_remainder() {
        let text = "a\nb\nc\nd\ne";
        let result = chunk_by_lines(text, 2, 0);
        assert_eq!(result, vec!["a\nb", "c\nd", "e"]);
    }

    #[test]
    fn chunk_lines_size_larger_than_total() {
        let text = "a\nb\nc";
        let result = chunk_by_lines(text, 100, 0);
        assert_eq!(result, vec!["a\nb\nc"]);
    }

    #[test]
    fn chunk_lines_with_overlap() {
        let text = "a\nb\nc\nd\ne";
        // size=3, overlap=1 → step=2
        let result = chunk_by_lines(text, 3, 1);
        assert_eq!(result, vec!["a\nb\nc", "c\nd\ne"]);
    }

    #[test]
    fn chunk_lines_overlap_equals_size_minus_one() {
        let text = "a\nb\nc\nd";
        // size=2, overlap=1 → step=1 (sliding window)
        let result = chunk_by_lines(text, 2, 1);
        assert_eq!(result, vec!["a\nb", "b\nc", "c\nd"]);
    }

    #[test]
    fn chunk_lines_overlap_ge_size_step_is_one() {
        let text = "a\nb\nc";
        // overlap >= size → step=1
        let result = chunk_by_lines(text, 2, 5);
        assert_eq!(result, vec!["a\nb", "b\nc"]);
    }

    #[test]
    fn chunk_lines_size_zero_returns_empty() {
        // size=0 should not produce infinite chunks
        let result = chunk_by_lines("a\nb\nc", 0, 0);
        assert_eq!(result, Vec::<String>::new());
    }

    // ─── chunk_by_chars tests ───

    #[test]
    fn chunk_chars_empty_text() {
        assert_eq!(chunk_by_chars("", 5, 0), Vec::<String>::new());
    }

    #[test]
    fn chunk_chars_exact_division() {
        let result = chunk_by_chars("abcdef", 3, 0);
        assert_eq!(result, vec!["abc", "def"]);
    }

    #[test]
    fn chunk_chars_remainder() {
        let result = chunk_by_chars("abcde", 3, 0);
        assert_eq!(result, vec!["abc", "de"]);
    }

    #[test]
    fn chunk_chars_size_larger_than_text() {
        let result = chunk_by_chars("abc", 100, 0);
        assert_eq!(result, vec!["abc"]);
    }

    #[test]
    fn chunk_chars_with_overlap() {
        // size=4, overlap=2 → step=2
        let result = chunk_by_chars("abcdef", 4, 2);
        assert_eq!(result, vec!["abcd", "cdef"]);
    }

    #[test]
    fn chunk_chars_overlap_ge_size_step_is_one() {
        // overlap >= size → step=1
        let result = chunk_by_chars("abc", 2, 3);
        assert_eq!(result, vec!["ab", "bc"]);
    }

    #[test]
    fn chunk_chars_multibyte() {
        // multibyte chars (3 bytes each in UTF-8, but split by char boundary)
        let result = chunk_by_chars("あいうえお", 2, 0);
        assert_eq!(result, vec!["あい", "うえ", "お"]);
    }

    #[test]
    fn chunk_chars_size_one() {
        let result = chunk_by_chars("abc", 1, 0);
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn chunk_chars_size_zero_returns_empty() {
        // size=0 should not produce infinite chunks
        let result = chunk_by_chars("abc", 0, 0);
        assert_eq!(result, Vec::<String>::new());
    }

    // ─── Prelude helpers ───

    /// Setup Lua VM with Layer 0 bridge + Layer 1 prelude loaded.
    fn setup_with_prelude() -> Lua {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();
        lua.load(PRELUDE).exec().unwrap();
        lua
    }

    // ─── alc.cache tests (non-LLM parts) ───

    #[test]
    fn cache_info_initial_state() {
        let lua = setup_with_prelude();
        let result: LuaValue = lua.load("return alc.cache_info()").eval().unwrap();
        let tbl = result.as_table().unwrap();
        assert_eq!(tbl.get::<i64>("entries").unwrap(), 0);
        assert_eq!(tbl.get::<i64>("hits").unwrap(), 0);
        assert_eq!(tbl.get::<i64>("misses").unwrap(), 0);
    }

    #[test]
    fn cache_clear_resets_state() {
        let lua = setup_with_prelude();
        lua.load(
            r#"
            -- Simulate cache state by calling cache_info before/after clear
            local info1 = alc.cache_info()
            alc.cache_clear()
            local info2 = alc.cache_info()
            assert(info2.entries == 0)
            assert(info2.hits == 0)
            assert(info2.misses == 0)
            "#,
        )
        .exec()
        .unwrap();
    }

    // ─── alc.parallel tests (validation) ───

    #[test]
    fn parallel_rejects_empty_items() {
        let lua = setup_with_prelude();
        let result: Result<LuaValue, _> = lua
            .load(r#"return alc.parallel({}, function(x) return x end)"#)
            .eval();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-empty array"),
            "expected non-empty array error, got: {err}"
        );
    }

    #[test]
    fn parallel_rejects_non_function_prompt_fn() {
        let lua = setup_with_prelude();
        let result: Result<LuaValue, _> = lua
            .load(r#"return alc.parallel({"a", "b"}, "not a function")"#)
            .eval();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("prompt_fn must be a function"),
            "expected function error, got: {err}"
        );
    }

    #[test]
    fn parallel_rejects_invalid_prompt_fn_return() {
        let lua = setup_with_prelude();
        let result: Result<LuaValue, _> = lua
            .load(r#"return alc.parallel({"a"}, function(x) return 42 end)"#)
            .eval();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("must return string or table"),
            "expected type error, got: {err}"
        );
    }

    #[test]
    fn parallel_rejects_table_without_prompt() {
        let lua = setup_with_prelude();
        let result: Result<LuaValue, _> = lua
            .load(r#"return alc.parallel({"a"}, function(x) return { system = "hi" } end)"#)
            .eval();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("without .prompt"),
            "expected prompt field error, got: {err}"
        );
    }

    // ─── alc.fingerprint tests (used by cache) ───

    #[test]
    fn fingerprint_deterministic() {
        let lua = setup_with_prelude();
        let result: bool = lua
            .load(r#"return alc.fingerprint("hello") == alc.fingerprint("hello")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn fingerprint_normalized() {
        let lua = setup_with_prelude();
        let result: bool = lua
            .load(r#"return alc.fingerprint("  Hello  World  ") == alc.fingerprint("hello world")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    // ─── alc.match_enum tests ───

    #[test]
    fn match_enum_exact_substring() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.match_enum("Verdict: BLOCKED. Fix the issues.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_case_insensitive() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.match_enum("verdict: pass. all good.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "PASS");
    }

    #[test]
    fn match_enum_last_wins() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Both appear, but PASS is last → PASS wins
        let result: String = lua
            .load(r#"return alc.match_enum("Initially BLOCKED, but after review: PASS", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "PASS");
    }

    #[test]
    fn match_enum_no_match_returns_nil() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: LuaValue = lua
            .load(r#"return alc.match_enum("something unrelated", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_enum_fuzzy_typo_in_short_response() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "BLOKED" is a typo for "BLOCKED" — fuzzy should catch it
        let result: String = lua
            .load(r#"return alc.match_enum("BLOKED", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_fuzzy_works_in_long_text() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Long sentence with a typo "BLCKED" buried in it — per-word fuzzy should find it
        let result: String = lua
            .load(r#"return alc.match_enum("After careful review of all the evidence and considering multiple factors, the final verdict is BLCKED due to missing tests.", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert_eq!(result, "BLOCKED");
    }

    #[test]
    fn match_enum_fuzzy_nil_when_no_close_word() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // No word is close enough to any candidate
        let result: LuaValue = lua
            .load(r#"return alc.match_enum("The weather is nice today", {"PASS", "BLOCKED"})"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    // ─── alc.match_bool tests ───

    #[test]
    fn match_bool_approved() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: bool = lua
            .load(r#"return alc.match_bool("Approved. The plan looks good.")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_rejected() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: bool = lua
            .load(r#"return alc.match_bool("rejected: missing test coverage")"#)
            .eval()
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn match_bool_nil_on_ambiguous() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: LuaValue = lua
            .load(r#"return alc.match_bool("I need more information about the design")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_last_keyword_wins() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "no" appears, then "approved" later → true
        let result: bool = lua
            .load(r#"return alc.match_bool("No issues found. Approved.")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_rejects_partial_word_ok_in_bypass() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "ok" should NOT match inside "token" or "okay" without boundary
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("This is a broken token")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_rejects_pass_in_bypass() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "pass" should NOT match inside "bypass"
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("We need to bypass the filter")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_rejects_no_in_innovation() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "no" should NOT match inside "innovation"
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("Great innovation in technology")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn match_bool_word_boundary_with_punctuation() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "yes" followed by punctuation should match
        let result: bool = lua
            .load(r#"return alc.match_bool("yes, that works")"#)
            .eval()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn match_bool_fail_in_failed_matches() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // "fail" at word boundary within "failed" — "fail" + "ed" where 'e' is alphanumeric
        // should NOT match
        let result: LuaValue = lua
            .load(r#"return alc.match_bool("The process failed gracefully")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    // ─── alc.parse_number tests ───

    #[test]
    fn parse_number_basic() {
        let lua = setup_with_prelude();
        let result: f64 = lua
            .load(r#"return alc.parse_number("Found 3 subtasks to implement")"#)
            .eval()
            .unwrap();
        assert!((result - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_number_decimal() {
        let lua = setup_with_prelude();
        let result: f64 = lua
            .load(r#"return alc.parse_number("Score: 7.5/10")"#)
            .eval()
            .unwrap();
        assert!((result - 7.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_number_with_pattern() {
        let lua = setup_with_prelude();
        let result: f64 = lua
            .load(r#"return alc.parse_number("Created 3 subtasks for implementation", "(%d+)%s+subtask")"#)
            .eval()
            .unwrap();
        assert!((result - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_number_nil_on_no_match() {
        let lua = setup_with_prelude();
        let result: LuaValue = lua
            .load(r#"return alc.parse_number("no numbers here")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    #[test]
    fn parse_number_negative() {
        let lua = setup_with_prelude();
        let result: f64 = lua
            .load(r#"return alc.parse_number("Temperature: -5 degrees")"#)
            .eval()
            .unwrap();
        assert!((result - (-5.0)).abs() < f64::EPSILON);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// chunk_by_lines never panics regardless of input.
        #[test]
        fn chunk_lines_never_panics(text in "\\PC{0,500}", size in 0usize..50, overlap in 0usize..50) {
            let _ = chunk_by_lines(&text, size, overlap);
        }

        /// chunk_by_chars never panics regardless of input.
        #[test]
        fn chunk_chars_never_panics(text in "\\PC{0,500}", size in 0usize..50, overlap in 0usize..50) {
            let _ = chunk_by_chars(&text, size, overlap);
        }

        /// All chars from the original text appear in at least one chunk (no data loss).
        #[test]
        fn chunk_chars_covers_all_input(text in "[a-z]{1,100}", size in 1usize..20) {
            let chunks = chunk_by_chars(&text, size, 0);
            let reconstructed: String = if chunks.len() <= 1 {
                chunks.into_iter().collect()
            } else {
                // Without overlap, concatenation should reproduce the original
                chunks.join("")
            };
            prop_assert_eq!(&reconstructed, &text);
        }

        /// All lines from the original text appear in at least one chunk (no data loss).
        #[test]
        fn chunk_lines_covers_all_input(
            lines in proptest::collection::vec("[a-z]{1,20}", 1..20),
            size in 1usize..10,
        ) {
            let text = lines.join("\n");
            let chunks = chunk_by_lines(&text, size, 0);
            let reconstructed = chunks.join("\n");
            prop_assert_eq!(&reconstructed, &text);
        }

        /// Each chunk has at most `size` characters.
        #[test]
        fn chunk_chars_respects_size(text in "[a-z]{1,200}", size in 1usize..50) {
            let chunks = chunk_by_chars(&text, size, 0);
            for chunk in &chunks {
                prop_assert!(chunk.chars().count() <= size,
                    "chunk length {} exceeds size {}", chunk.chars().count(), size);
            }
        }

        /// Each chunk has at most `size` lines.
        #[test]
        fn chunk_lines_respects_size(
            lines in proptest::collection::vec("[a-z]{1,10}", 1..30),
            size in 1usize..10,
        ) {
            let text = lines.join("\n");
            let chunks = chunk_by_lines(&text, size, 0);
            for chunk in &chunks {
                let line_count = chunk.lines().count();
                prop_assert!(line_count <= size,
                    "chunk has {} lines, exceeds size {}", line_count, size);
            }
        }

        /// With overlap, adjacent chunks share `overlap` characters.
        #[test]
        fn chunk_chars_overlap_shared(
            text in "[a-z]{10,100}",
            size in 3usize..15,
            overlap in 1usize..3,
        ) {
            prop_assume!(overlap < size);
            let chunks = chunk_by_chars(&text, size, overlap);
            if chunks.len() >= 2 {
                for i in 0..chunks.len() - 1 {
                    let suffix: String = chunks[i].chars().rev().take(overlap).collect::<Vec<_>>().into_iter().rev().collect();
                    let prefix: String = chunks[i + 1].chars().take(overlap).collect();
                    prop_assert_eq!(&suffix, &prefix,
                        "chunk[{}] suffix != chunk[{}] prefix", i, i + 1);
                }
            }
        }
    }
}
