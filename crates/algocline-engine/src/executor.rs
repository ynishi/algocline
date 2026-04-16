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
use mlua_pkg::{resolvers::FsResolver, sandbox::SymlinkAwareSandbox, Registry};

use crate::bridge;
use crate::llm_bridge::LlmRequest;
use crate::session::Session;
use crate::variant_pkg::{register_variant_pkgs, VariantPkg};

/// Build an `FsResolver` for the given path.
///
/// By default uses `SymlinkAwareSandbox` so that `alc_pkg_link` symlinks
/// are followed. Set `ALC_PKG_STRICT=1` to use the strict `FsSandbox`
/// (rejects all symlinks pointing outside the root).
fn make_resolver(path: &std::path::Path) -> Option<FsResolver> {
    let strict = std::env::var("ALC_PKG_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if strict {
        FsResolver::new(path).ok()
    } else {
        SymlinkAwareSandbox::new(path)
            .ok()
            .map(FsResolver::with_sandbox)
    }
}

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
                if let Some(resolver) = make_resolver(path) {
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
    ///
    /// Uses the shared VM. `extra_lib_paths` must be empty — use
    /// [`eval_simple_with_paths`] when project-local paths are needed.
    pub async fn eval_simple(&self, code: String) -> Result<serde_json::Value, String> {
        self.eval_simple_with_paths(code, vec![], vec![]).await
    }

    /// Evaluate Lua code without LLM bridge, with optional extra package paths
    /// and variant pkgs.
    ///
    /// When both `extra_lib_paths` and `variant_pkgs` are empty, reuses the
    /// shared VM (cheap). When either is non-empty, spawns a dedicated VM so
    /// the extra resolvers are active (slightly more expensive, but `pkg_list`
    /// is the only caller and it is low-frequency).
    pub async fn eval_simple_with_paths(
        &self,
        code: String,
        extra_lib_paths: Vec<PathBuf>,
        variant_pkgs: Vec<VariantPkg>,
    ) -> Result<serde_json::Value, String> {
        if extra_lib_paths.is_empty() && variant_pkgs.is_empty() {
            // Fast path: reuse the long-lived shared VM.
            let task = self.isle.spawn_exec(move |lua| {
                let result: mlua::Value = lua
                    .load(&code)
                    .eval()
                    .map_err(|e| IsleError::Lua(e.to_string()))?;
                let json: serde_json::Value = lua
                    .from_value(result)
                    .map_err(|e| IsleError::Lua(e.to_string()))?;
                serde_json::to_string(&json)
                    .map_err(|e| IsleError::Lua(format!("JSON serialize: {e}")))
            });
            let json_str = task.await.map_err(|e| e.to_string())?;
            return serde_json::from_str(&json_str).map_err(|e| format!("JSON parse: {e}"));
        }

        // Slow path: spawn a dedicated VM with extra resolvers prepended.
        let mut effective = extra_lib_paths;
        effective.extend(self.lib_paths.iter().cloned());

        let (tmp_isle, _tmp_driver) = AsyncIsle::spawn(move |lua| {
            let mut reg = Registry::new();
            // Variant pkgs first so alc.local.toml overrides win over global.
            register_variant_pkgs(&mut reg, &variant_pkgs);
            for path in &effective {
                if let Some(resolver) = make_resolver(path) {
                    reg.add(resolver);
                }
            }
            reg.install(lua)?;
            Ok(())
        })
        .await
        .map_err(|e| format!("eval_simple VM spawn failed: {e}"))?;

        let task = tmp_isle.spawn_exec(move |lua| {
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
    ///
    /// `extra_lib_paths` are prepended to `self.lib_paths` so project-local
    /// packages take precedence over the global package directory.
    /// `variant_pkgs` come from `alc.local.toml` and override both layers
    /// (registered at the highest priority).
    pub async fn start_session(
        &self,
        code: String,
        ctx: serde_json::Value,
        extra_lib_paths: Vec<PathBuf>,
        variant_pkgs: Vec<VariantPkg>,
    ) -> Result<Session, String> {
        let spec = ExecutionSpec::new(code, ctx);
        let metrics = ExecutionMetrics::new();

        // Extract and apply budget from ctx.budget
        if let Some(budget) = Budget::from_ctx(&spec.ctx) {
            metrics.set_budget(budget);
        }

        let (llm_tx, llm_rx) = tokio::sync::mpsc::channel::<LlmRequest>(16);

        // Build effective lib_paths: extra (project-local) first, then defaults.
        // Priority: variant_pkgs > extra_lib_paths > self.lib_paths
        // (ALC_PACKAGES_PATH + global default).
        let mut effective = extra_lib_paths;
        effective.extend(self.lib_paths.iter().cloned());

        let bridge_config = bridge::BridgeConfig {
            llm_tx: Some(llm_tx),
            ns: spec.namespace.clone(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: effective.clone(), // fork child VMs inherit project paths
            variant_pkgs: variant_pkgs.clone(), // fork child VMs inherit variant overrides
        };
        let lua_ctx = spec.ctx.clone();
        let lua_code = spec.code.clone();

        // 1. Spawn a dedicated VM for this session.
        let (session_isle, session_driver) = AsyncIsle::spawn(move |lua| {
            let mut reg = Registry::new();
            // Variant pkgs first so alc.local.toml overrides win over global.
            register_variant_pkgs(&mut reg, &variant_pkgs);
            for path in &effective {
                if let Some(resolver) = make_resolver(path) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temporary package directory with the given name and `init.lua` content.
    fn make_pkg_dir(parent: &std::path::Path, pkg_name: &str, init_lua: &str) -> PathBuf {
        let pkg_dir = parent.join(pkg_name);
        fs::create_dir_all(&pkg_dir).unwrap();
        fs::write(pkg_dir.join("init.lua"), init_lua).unwrap();
        parent.to_path_buf()
    }

    /// `extra_lib_paths=vec![]` — eval_simple must work as before.
    #[tokio::test]
    async fn no_extra_lib_paths_eval_simple() {
        let executor = Executor::new(vec![]).await.unwrap();
        let result = executor.eval_simple("return 42".to_string()).await.unwrap();
        assert_eq!(result, serde_json::json!(42));
    }

    /// `eval_simple_with_paths` with a project-local package.
    ///
    /// Creates a temp dir with `test_pkg/init.lua` returning `{value = 99}`,
    /// then verifies `require("test_pkg").value` == 99 via the extra resolver.
    #[tokio::test]
    async fn extra_lib_paths_reachable_via_eval_simple_with_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_root = make_pkg_dir(tmp.path(), "test_pkg", "return { value = 99 }");

        let executor = Executor::new(vec![]).await.unwrap();
        let code = r#"
            local pkg = require("test_pkg")
            return pkg.value
        "#
        .to_string();

        let result = executor
            .eval_simple_with_paths(code, vec![pkg_root], vec![])
            .await
            .unwrap();

        assert_eq!(result, serde_json::json!(99));
    }

    /// Variant pkg with a non-matching directory name resolves via
    /// `VariantRootResolver` + `PrefixResolver`.
    #[tokio::test]
    async fn variant_pkg_resolves_root_and_submodule() {
        let tmp = tempfile::tempdir().unwrap();
        // pkg dir name (`physical-dir`) intentionally differs from the
        // require name (`logical_name`) — variant scope must support this.
        let pkg_dir = tmp.path().join("physical-dir");
        fs::create_dir_all(&pkg_dir).unwrap();
        fs::write(
            pkg_dir.join("init.lua"),
            "return { greet = function(n) return 'hi-' .. n end, sub = require('logical_name.sub') }",
        )
        .unwrap();
        fs::write(pkg_dir.join("sub.lua"), "return { value = 7 }").unwrap();

        let executor = Executor::new(vec![]).await.unwrap();
        let code = r#"
            local pkg = require("logical_name")
            return { msg = pkg.greet("there"), sub_value = pkg.sub.value }
        "#
        .to_string();

        let result = executor
            .eval_simple_with_paths(code, vec![], vec![VariantPkg::new("logical_name", pkg_dir)])
            .await
            .unwrap();

        assert_eq!(result["msg"], serde_json::json!("hi-there"));
        assert_eq!(result["sub_value"], serde_json::json!(7));
    }

    /// Variant pkg overrides a same-name global pkg (priority: variant > global).
    #[tokio::test]
    async fn variant_pkg_overrides_global_same_name() {
        let global_tmp = tempfile::tempdir().unwrap();
        let variant_tmp = tempfile::tempdir().unwrap();

        // Global: my_pkg returns 1
        make_pkg_dir(global_tmp.path(), "my_pkg", "return { value = 1 }");
        // Variant: my_pkg returns 2 — must win
        let variant_dir = variant_tmp.path().join("my_pkg");
        fs::create_dir_all(&variant_dir).unwrap();
        fs::write(variant_dir.join("init.lua"), "return { value = 2 }").unwrap();

        let executor = Executor::new(vec![global_tmp.path().to_path_buf()])
            .await
            .unwrap();

        let code = r#"
            local pkg = require("my_pkg")
            return pkg.value
        "#
        .to_string();

        let result = executor
            .eval_simple_with_paths(code, vec![], vec![VariantPkg::new("my_pkg", variant_dir)])
            .await
            .unwrap();

        assert_eq!(result, serde_json::json!(2));
    }

    /// When `extra_lib_paths` has a pkg with the same name as one in global paths,
    /// the extra one takes priority (it is prepended).
    #[tokio::test]
    async fn extra_lib_paths_priority_over_default() {
        let global_tmp = tempfile::tempdir().unwrap();
        let extra_tmp = tempfile::tempdir().unwrap();

        // Global: test_pkg returns 1
        make_pkg_dir(global_tmp.path(), "test_pkg", "return { value = 1 }");
        // Extra (project-local): test_pkg returns 2
        let extra_root = make_pkg_dir(extra_tmp.path(), "test_pkg", "return { value = 2 }");

        // Executor has global as its lib_paths.
        let executor = Executor::new(vec![global_tmp.path().to_path_buf()])
            .await
            .unwrap();

        let code = r#"
            local pkg = require("test_pkg")
            return pkg.value
        "#
        .to_string();

        let result = executor
            .eval_simple_with_paths(code, vec![extra_root], vec![])
            .await
            .unwrap();

        // extra (2) must win over global (1)
        assert_eq!(result, serde_json::json!(2));
    }
}
