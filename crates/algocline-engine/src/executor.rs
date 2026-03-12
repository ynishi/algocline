//! Lua execution engine.
//!
//! Manages the mlua-isle VM and orchestrates StdLib injection
//! for each session:
//!
//! 1. **Layer 0** — [`bridge::register`] injects Rust-backed `alc.*` primitives
//! 2. **Layer 1** — [`PRELUDE`] adds Lua-based combinators (`alc.map`, etc.)
//! 3. **Layer 2** — [`mlua_pkg::Registry`] makes `require("explore")` etc.
//!    resolve from `~/.algocline/packages/`
//!
//! ## Execution models
//!
//! - **`eval_simple`** — sync eval (no LLM bridge). For lightweight ops
//!   like reading package metadata.
//! - **`start_session`** — coroutine-based execution. `alc.llm()` yields
//!   the coroutine instead of blocking the Lua thread, allowing other
//!   requests to proceed while waiting for LLM responses.

use std::path::PathBuf;

use algocline_core::{ExecutionMetrics, ExecutionSpec};
use mlua::LuaSerdeExt;
use mlua_isle::{AsyncIsle, AsyncIsleDriver, IsleError};
use mlua_pkg::{resolvers::FsResolver, Registry};

use crate::bridge;
use crate::llm_bridge::LlmRequest;
use crate::session::Session;

/// Layer 1: Prelude combinators (map, reduce, vote, filter).
/// Embedded at compile time and loaded into every session.
const PRELUDE: &str = include_str!("prelude.lua");

/// Lua execution engine.
///
/// Wraps mlua-isle AsyncIsle (Handle/Driver pattern) to provide
/// non-blocking, cancellable Lua execution with alc StdLib injected.
pub struct Executor {
    isle: AsyncIsle,
    _driver: AsyncIsleDriver,
}

impl Executor {
    pub async fn new(lib_paths: Vec<PathBuf>) -> anyhow::Result<Self> {
        let (isle, driver) = AsyncIsle::spawn(move |lua| {
            // Install mlua-pkg Registry once during VM initialization.
            // This survives across sessions since mlua-isle reuses the VM.
            let mut reg = Registry::new();
            for path in &lib_paths {
                if let Ok(resolver) = FsResolver::new(path) {
                    reg.add(resolver);
                }
            }
            reg.install(lua)?;
            Ok(())
        })
        .await?;

        Ok(Self {
            isle,
            _driver: driver,
        })
    }

    /// Evaluate Lua code without LLM bridge. For lightweight operations
    /// like reading package metadata.
    pub async fn eval_simple(&self, code: String) -> Result<serde_json::Value, String> {
        let task = self.isle.spawn_exec(move |lua| {
            let result: mlua::Value = lua
                .load(&code)
                .eval()
                .map_err(|e| IsleError::Lua(e.to_string()))?;
            let json: serde_json::Value = lua
                .from_value(result)
                .map_err(|e| IsleError::Lua(e.to_string()))?;
            serde_json::to_string(&json).map_err(|e| IsleError::Lua(format!("JSON serialize: {e}")))
        });

        let json_str = task.await.map_err(|e| e.to_string())?;
        serde_json::from_str(&json_str).map_err(|e| format!("JSON parse: {e}"))
    }

    /// Start a new Lua execution session.
    ///
    /// Phase 1 (sync exec): registers alc.* StdLib, ctx, prelude.
    /// Phase 2 (coroutine): executes user Lua code. When `alc.llm()`
    /// is called, the coroutine yields instead of blocking the Lua
    /// thread, allowing other requests to proceed.
    pub async fn start_session(
        &self,
        code: String,
        ctx: serde_json::Value,
    ) -> Result<Session, String> {
        let spec = ExecutionSpec::new(code, ctx);
        let metrics = ExecutionMetrics::new();
        let custom_handle = metrics.custom_handle();

        let (llm_tx, llm_rx) = tokio::sync::mpsc::channel::<LlmRequest>(16);

        let ns = spec.namespace.clone();
        let lua_ctx = spec.ctx.clone();
        let lua_code = spec.code.clone();

        // Phase 1: Setup (sync exec on Lua thread)
        // Registers alc.* with async LLM bridge, sets ctx, loads prelude,
        // clears package.loaded cache.
        self.isle
            .exec(move |lua| {
                // 1. Create alc StdLib table with async LLM bridge + state + stats
                let alc_table = lua.create_table()?;
                bridge::register(lua, &alc_table, Some(llm_tx), ns, custom_handle)?;
                lua.globals().set("alc", alc_table)?;

                // 2. Set ctx global
                let ctx_value = lua.to_value(&lua_ctx)?;
                lua.globals().set("ctx", ctx_value)?;

                // 3. Load prelude (alc.map, alc.reduce, alc.vote, alc.filter)
                lua.load(PRELUDE)
                    .exec()
                    .map_err(|e| IsleError::Lua(format!("Prelude load failed: {e}")))?;

                // 4. Clear package.loaded cache so each session gets fresh modules
                let loaded: mlua::Table =
                    lua.globals().get::<mlua::Table>("package")?.get("loaded")?;
                let keys: Vec<String> = loaded
                    .pairs::<String, mlua::Value>()
                    .filter_map(|r| r.ok().map(|(k, _)| k))
                    .collect();
                for key in keys {
                    loaded.set(key, mlua::Value::Nil)?;
                }

                Ok("ok".to_string())
            })
            .await
            .map_err(|e| format!("Session setup failed: {e}"))?;

        // Phase 2: Execute user code as a coroutine.
        // alc.llm() is an async function — when called, the coroutine
        // yields and other coroutines/requests can make progress.
        //
        // The user code is wrapped so its return value is JSON-serialized
        // via alc.json_encode. This matches the old spawn_exec behavior
        // where the closure did serde_json::to_string. coroutine_eval's
        // lua_value_to_string only does tostring(), which loses structure
        // for tables.
        let wrapped_code = format!("return alc.json_encode((function()\n{lua_code}\nend)())");
        let exec_task = self.isle.spawn_coroutine_eval(&wrapped_code);

        Ok(Session::new(llm_rx, exec_task, metrics))
    }
}
