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
use std::sync::Arc;

use algocline_core::{
    BudgetHandle, CustomMetricsHandle, ExecutionMetrics, LogSink, ProgressHandle, StatsHandle,
};
use mlua::prelude::*;
use tempfile::TempDir;

mod data;
mod evalframe;
mod fork;
mod fuzzy;
mod llm;
#[cfg(feature = "nn")]
mod nn_card;
mod text;

use crate::card::FileCardStore;
use crate::llm_bridge::LlmRequest;
use crate::state::JsonFileStore;
use crate::variant_pkg::VariantPkg;

/// Layer 1 prelude (also used by fork to setup child VMs).
pub const PRELUDE: &str = include_str!("../prelude.lua");

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
    /// Stats handle for `alc.stats.llm_calls()` (auto-counted session metrics).
    pub stats: StatsHandle,
    /// Budget checker for LLM call limits.
    pub budget: BudgetHandle,
    /// Progress reporter for alc.progress().
    pub progress: ProgressHandle,
    /// Package search paths (needed by alc.fork to setup child VMs).
    pub lib_paths: Vec<PathBuf>,
    /// Variant pkg overrides (`alc.local.toml`) — propagated to fork children.
    pub variant_pkgs: Vec<VariantPkg>,
    /// State store for `alc.state.*` (service layer resolves the root).
    pub state_store: Arc<JsonFileStore>,
    /// Card store for `alc.card.*` (service layer resolves the root).
    pub card_store: Arc<FileCardStore>,
    /// Cached `[setting.card].run` value resolved at session start.
    ///
    /// When `false` (default), `alc.card.create` / `alc.card.append` calls
    /// that carry a top-level `run` field become no-op: the closures return
    /// Lua `nil` without touching the card store or publishing a
    /// `CardEvent`.  Calls without a `run` field are unaffected (Phase 1-B
    /// gate is additive — existing pkgs that never populate `run` see no
    /// behavioural change).
    pub card_run_enabled: bool,
    /// Scenarios directory exposed to Lua via `alc._dirs.scenarios`.
    pub scenarios_dir: PathBuf,
    /// Store root for `alc.nn` model bundles.
    ///
    /// Consumed by `register_nn` under the `nn` feature to construct the
    /// filesystem-backed [`algocline_nn::FsStore`] wired via
    /// [`algocline_nn::install_store`]. Resolved by the service layer via
    /// [`algocline_core::AppDir::nn_dir`] so the engine crate never touches
    /// `$HOME` / `$ALC_HOME` directly.
    ///
    /// An empty [`PathBuf`] means "do not install a store" — `alc.nn.save` /
    /// `alc.nn.load` then error with `"no NN store registered"`. Test sites
    /// that never exercise save/load can pass [`PathBuf::new`]. Ignored when
    /// the `nn` feature is disabled.
    pub nn_dir: PathBuf,
    /// Per-session log-capture ring buffer.
    ///
    /// Obtained from `ExecutionMetrics::log_sink_handle()`.  Passed to
    /// `alc.log` and `print()` overrides so log output is routed into the
    /// ring buffer for `alc_status` recent_logs.
    ///
    /// `None` for `eval_simple` / fork child sessions where observability
    /// is not needed; in that case log entries are emitted to tracing only.
    pub log_sink: Option<LogSink>,
}

pub use data::register_env;
pub use evalframe::register_evalframe;

/// Register all Layer 0 runtime primitives onto the given table.
pub fn register(lua: &Lua, alc_table: &LuaTable, config: BridgeConfig) -> LuaResult<()> {
    data::register_json(lua, alc_table)?;
    fuzzy::register_fuzzy(lua, alc_table)?;
    // Register alc.log — pass LogSink when available so entries reach the ring buffer.
    if let Some(sink) = config.log_sink.clone() {
        data::register_log(lua, alc_table, sink.clone())?;
        // Override global print() to also push to the ring buffer.
        data::register_print(lua, sink)?;
    } else {
        // Fallback: tracing-only path for eval_simple / fork children.
        data::register_log(lua, alc_table, algocline_core::LogSink::new())?;
    }
    data::register_state(lua, alc_table, config.ns, Arc::clone(&config.state_store))?;
    data::register_card(
        lua,
        alc_table,
        Arc::clone(&config.card_store),
        config.card_run_enabled,
    )?;
    data::register_dirs(
        lua,
        alc_table,
        config.state_store.root(),
        config.card_store.root(),
        &config.scenarios_dir,
    )?;
    text::register_chunk(lua, alc_table)?;
    data::register_stats(lua, alc_table, config.custom_metrics, config.stats)?;
    register_time(lua, alc_table)?;
    register_math(lua, alc_table)?;
    #[cfg(feature = "nn")]
    register_nn(lua, alc_table, config.nn_dir.clone())?;
    #[cfg(feature = "nn")]
    nn_card::register_nn_card(
        lua,
        alc_table,
        Arc::clone(&config.card_store),
        config.nn_dir.clone(),
    )?;
    llm::register_budget_remaining(lua, alc_table, config.budget.clone())?;
    llm::register_progress(lua, alc_table, config.progress)?;
    if let Some(tx) = config.llm_tx {
        llm::register_llm(
            lua,
            alc_table,
            tx.clone(),
            config.budget.clone(),
            Arc::clone(&config.card_store),
        )?;
        llm::register_llm_batch(lua, alc_table, tx.clone(), config.budget.clone())?;
        fork::register_fork(
            lua,
            alc_table,
            tx,
            config.budget,
            config.lib_paths,
            config.variant_pkgs,
            config.state_store,
            config.card_store,
            config.card_run_enabled,
            config.scenarios_dir,
            config.nn_dir,
        )?;
    }
    Ok(())
}

/// Register `alc.math` — mlua-mathlib v0.3 (RNG, distributions, statistics, hypothesis testing, ranking, information theory, time series).
fn register_math(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let math_table = mlua_mathlib::module(lua)?;
    alc_table.set("math", math_table)?;
    Ok(())
}

/// Register `alc.nn` — thin candle wrapper (feature `nn`, default off).
///
/// Only compiled when the `nn` feature is enabled, so the default build never
/// links candle. Mirrors [`register_math`]: delegate table construction to the
/// dedicated bridge crate and set it as `alc.nn`.
///
/// When `nn_dir` is non-empty, an [`algocline_nn::FsStore`] rooted there is
/// installed on the VM so `alc.nn.save` / `alc.nn.load` resolve through it.
/// An empty `nn_dir` skips the install; save/load then error with
/// `"no NN store registered"` — suitable for test VMs that never exercise
/// persistence. The path itself is resolved by the service layer via
/// [`algocline_core::AppDir::nn_dir`]; the engine crate stays free of any
/// `$HOME` / `$ALC_HOME` reads.
#[cfg(feature = "nn")]
fn register_nn(lua: &Lua, alc_table: &LuaTable, nn_dir: PathBuf) -> LuaResult<()> {
    let nn_table = algocline_nn::module(lua)?;
    alc_table.set("nn", nn_table)?;
    if !nn_dir.as_os_str().is_empty() {
        algocline_nn::install_store(lua, Arc::new(algocline_nn::FsStore::new(nn_dir)))?;
    }
    Ok(())
}

/// Embedded mock layer (`with_alc` / `alc_mock` / `alc.spy`) installed
/// on top of the standard `alc.*` surface in `install_for_pkg_test`.
pub(crate) const MOCK_LAYER: &str = include_str!("mock.lua");

/// Install the production `alc.*` primitive surface plus the mock layer
/// on `lua` for use by the `alc_pkg_test` sandbox.
///
/// Spec authors get:
/// * the full `alc.*` surface that `alc_run` exposes (stateless helpers
///   like `alc.json_encode`, `alc.fingerprint`, `alc.parse_number`,
///   `alc.fuzzy.*`, plus stateful helpers backed by in-memory
///   per-VM tempdirs for `alc.state.*` and `alc.card.*`);
/// * `alc.llm` / `alc.llm_batch` / `alc.fork` as stubs that error out
///   when called without a `with_alc({ llm = … }, …)` override;
/// * a Pure-Lua mock layer (`with_alc(overrides, fn)`,
///   `alc_mock.install/restore`, `alc.spy(name, default_fn?)`).
///
/// **Invariant** (enforced by `tests/bridge_sandbox_parity.rs`):
/// `production primitive surface ⊆ test sandbox primitive surface`.
/// Every key reachable on `_G.alc` after a successful production
/// [`register`] call is also reachable after `install_for_pkg_test`.
///
/// The per-VM tempdir backing `state_store` / `card_store` is held on
/// the Lua VM via `set_app_data` and dropped together with the VM.
pub fn install_for_pkg_test(lua: &Lua) -> LuaResult<()> {
    let metrics = ExecutionMetrics::new();
    let tmp = TempDir::new()
        .map_err(|e| LuaError::external(format!("install_for_pkg_test: tempdir: {e}")))?;
    let root = tmp.path().to_path_buf();
    // Tie tempdir lifetime to the Lua VM so it is cleaned up when the VM is dropped.
    lua.set_app_data::<TempDir>(tmp);

    let config = BridgeConfig {
        llm_tx: None,
        ns: "default".into(),
        custom_metrics: metrics.custom_metrics_handle(),
        stats: metrics.stats_handle(),
        budget: metrics.budget_handle(),
        progress: metrics.progress_handle(),
        lib_paths: vec![],
        variant_pkgs: vec![],
        state_store: std::sync::Arc::new(crate::state::JsonFileStore::new(root.join("state"))),
        card_store: std::sync::Arc::new(crate::card::FileCardStore::new(root.join("cards"))),
        // Enable the `[run]` gate by default in pkg-test sandboxes so
        // spec authors can exercise the new field without extra plumbing.
        // Production sessions receive the resolved `[setting.card].run`
        // value from the service layer.
        card_run_enabled: true,
        scenarios_dir: root.join("scenarios"),
        nn_dir: root.join("nn"),
        log_sink: None,
    };

    let alc_table = lua.create_table()?;
    register(lua, &alc_table, config)?;

    // Stateful / external I/O entries that production-only registers when
    // `llm_tx` is `Some`.  Install stubs so spec authors must mock them
    // explicitly via `with_alc({ llm = ... }, fn)` — calling the unmocked
    // entry surfaces a clear error instead of `attempt to call a nil value`.
    install_external_io_stub(lua, &alc_table, "llm")?;
    install_external_io_stub(lua, &alc_table, "llm_batch")?;
    install_external_io_stub(lua, &alc_table, "fork")?;

    lua.globals().set("alc", alc_table)?;
    evalframe::register_evalframe(lua).map_err(|e| {
        LuaError::external(format!("install_for_pkg_test: register_evalframe: {e}"))
    })?;
    lua.load(PRELUDE)
        .set_name("@alc_prelude")
        .exec()
        .map_err(|e| LuaError::external(format!("install_for_pkg_test: prelude: {e}")))?;
    lua.load(MOCK_LAYER)
        .set_name("@bridge_mock")
        .exec()
        .map_err(|e| LuaError::external(format!("install_for_pkg_test: mock layer: {e}")))?;
    Ok(())
}

fn install_external_io_stub(lua: &Lua, alc_table: &LuaTable, name: &'static str) -> LuaResult<()> {
    let stub = lua.create_function(
        move |_, _: mlua::Variadic<LuaValue>| -> LuaResult<LuaValue> {
            Err(LuaError::external(format!(
                "mock required: alc.{name} — wrap the call in `with_alc({{ {name} = fn }}, fn)` \
             inside your spec (alc_pkg_test sandbox stubs external I/O by design)"
            )))
        },
    )?;
    alc_table.set(name, stub)?;
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
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        BridgeConfig {
            llm_tx: None,
            ns: "default".into(),
            custom_metrics: metrics.custom_metrics_handle(),
            stats: metrics.stats_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
            variant_pkgs: vec![],
            state_store: Arc::new(JsonFileStore::new(root.join("state"))),
            card_store: Arc::new(FileCardStore::new(root.join("cards"))),
            card_run_enabled: false,
            scenarios_dir: root.join("scenarios"),
            nn_dir: root.join("nn"),
            log_sink: None,
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
