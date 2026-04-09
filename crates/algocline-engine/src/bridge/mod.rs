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

use algocline_core::{BudgetHandle, CustomMetricsHandle, ProgressHandle};
use mlua::prelude::*;

mod data;
mod fork;
mod fuzzy;
mod llm;
mod text;

use crate::llm_bridge::LlmRequest;

/// Layer 1 prelude (also used by fork to setup child VMs).
pub(crate) const PRELUDE: &str = include_str!("../prelude.lua");

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
    data::register_json(lua, alc_table)?;
    fuzzy::register_fuzzy(lua, alc_table)?;
    data::register_log(lua, alc_table)?;
    data::register_state(lua, alc_table, config.ns)?;
    text::register_chunk(lua, alc_table)?;
    data::register_stats(lua, alc_table, config.custom_metrics)?;
    register_time(lua, alc_table)?;
    register_math(lua, alc_table)?;
    llm::register_budget_remaining(lua, alc_table, config.budget.clone())?;
    llm::register_progress(lua, alc_table, config.progress)?;
    if let Some(tx) = config.llm_tx {
        llm::register_llm(lua, alc_table, tx.clone(), config.budget.clone())?;
        llm::register_llm_batch(lua, alc_table, tx.clone(), config.budget.clone())?;
        fork::register_fork(lua, alc_table, tx, config.budget, config.lib_paths)?;
    }
    Ok(())
}

/// Register `alc.math` — mlua-mathlib v0.3 (RNG, distributions, statistics, hypothesis testing, ranking, information theory, time series).
fn register_math(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let math_table = mlua_mathlib::module(lua)?;
    alc_table.set("math", math_table)?;
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
