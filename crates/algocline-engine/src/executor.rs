//! Lua execution engine.
//!
//! Orchestrates StdLib injection and Lua execution for each session:
//!
//! 1. **Layer 0** — [`bridge::register`] injects Rust-backed `alc.*` primitives
//! 2. **Layer 1** — [`PRELUDE`] adds Lua-based combinators (`alc.map`, etc.)
//! 3. **Layer 2** — [`mlua_pkg::Registry`] makes `require("ucb")` etc.
//!    resolve from `~/.algocline/packages/`
//!
//! ## Execution models
//!
//! - **`eval_simple`** — sync eval on a shared VM (no LLM bridge).
//!   For lightweight ops like reading package metadata.
//! - **`start_session`** — spawns a **dedicated VM per session**.
//!   Each session gets an isolated Lua VM so concurrent sessions
//!   cannot interfere with each other's globals (`alc`, `ctx`).
//!   `alc.llm()` yields the coroutine, and the VM is cleaned up
//!   when the session completes or is abandoned.

use std::path::PathBuf;

use algocline_core::{Budget, ExecutionMetrics, ExecutionSpec};
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
/// Holds a **shared VM** for lightweight stateless operations (`eval_simple`)
/// and spawns **per-session VMs** for coroutine-based execution (`start_session`).
///
/// Per-session VMs eliminate global namespace pollution between concurrent
/// sessions — each session's `alc`, `ctx`, and `package.loaded` are fully
/// isolated.
pub struct Executor {
    /// Shared VM for eval_simple (stateless, no session globals).
    isle: AsyncIsle,
    _driver: AsyncIsleDriver,
    /// Package resolver paths, cloned into each per-session VM.
    lib_paths: Vec<PathBuf>,
}

impl Executor {
    pub async fn new(lib_paths: Vec<PathBuf>) -> anyhow::Result<Self> {
        let paths_for_shared = lib_paths.clone();
        let (isle, driver) = AsyncIsle::spawn(move |lua| {
            let mut reg = Registry::new();
            for path in &paths_for_shared {
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
            lib_paths,
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

    /// Start a new Lua execution session on a **dedicated VM**.
    ///
    /// Each session gets its own Lua VM (OS thread + mlua instance) so
    /// concurrent sessions cannot interfere with each other's globals.
    /// The VM is cleaned up automatically when the session completes or
    /// is abandoned (all senders drop → channel closes → thread exits).
    pub async fn start_session(
        &self,
        code: String,
        ctx: serde_json::Value,
    ) -> Result<Session, String> {
        let spec = ExecutionSpec::new(code, ctx);
        let metrics = ExecutionMetrics::new();

        // Extract and apply budget from ctx.budget
        if let Some(budget) = Budget::from_ctx(&spec.ctx) {
            metrics.set_budget(budget);
        }

        let (llm_tx, llm_rx) = tokio::sync::mpsc::channel::<LlmRequest>(16);

        let bridge_config = bridge::BridgeConfig {
            llm_tx: Some(llm_tx),
            ns: spec.namespace.clone(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
        };
        let lua_ctx = spec.ctx.clone();
        let lua_code = spec.code.clone();

        // 1. Spawn a dedicated VM for this session.
        let lib_paths = self.lib_paths.clone();
        let (session_isle, session_driver) = AsyncIsle::spawn(move |lua| {
            let mut reg = Registry::new();
            for path in &lib_paths {
                if let Ok(resolver) = FsResolver::new(path) {
                    reg.add(resolver);
                }
            }
            reg.install(lua)?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Session VM spawn failed: {e}"))?;

        // 2. Setup: register alc.* StdLib, set ctx, load prelude.
        //    Safe to set globals — this VM is exclusively ours.
        session_isle
            .exec(move |lua| {
                let alc_table = lua.create_table()?;
                bridge::register(lua, &alc_table, bridge_config)?;
                lua.globals().set("alc", alc_table)?;

                let ctx_value = lua.to_value(&lua_ctx)?;
                lua.globals().set("ctx", ctx_value)?;

                lua.load(PRELUDE)
                    .exec()
                    .map_err(|e| IsleError::Lua(format!("Prelude load failed: {e}")))?;

                // No need to clear package.loaded — fresh VM.

                Ok("ok".to_string())
            })
            .await
            .map_err(|e| format!("Session setup failed: {e}"))?;

        // 3. Execute user code as a coroutine on the session VM.
        let wrapped_code = format!("return alc.json_encode((function()\n{lua_code}\nend)())");
        let exec_task = session_isle.spawn_coroutine_eval(&wrapped_code);

        // Handle no longer needed — all requests have been sent.
        // The driver keeps the channel alive until the session completes.
        drop(session_isle);

        Ok(Session::new(llm_rx, exec_task, metrics, session_driver))
    }
}
