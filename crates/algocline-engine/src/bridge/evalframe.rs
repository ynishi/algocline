//! Vendored evalframe registration.
//!
//! Registers the vendored evalframe Lua modules (~3660 LOC, 26 files) on
//! `package.preload` and injects the `std` global that evalframe.std
//! requires. Called at session VM init (executor.rs) before PRELUDE
//! load, so `require("evalframe")` works from both MCP `alc_eval` and
//! Lua-side `alc.eval` paths.
//!
//! # Vendored source
//!
//! Files live at `crates/algocline-engine/src/vendor/evalframe/**/*.lua`
//! and are `include_str!`'d verbatim. Upstream sync is manual (see per-file
//! origin markers at the top of each vendored file).

use mlua::prelude::*;

/// Lua shim that bridges algocline's `alc.*` primitives to the `std` global
/// expected by `evalframe.std` (json / fs.read / fs.is_file / time.now).
///
/// Injected once at session VM init; provides the same contract as the
/// legacy inline shim previously prepended by
/// `algocline-app/src/service/eval.rs`.
pub const EVALFRAME_STD_SHIM: &str = r#"
std = {
  json = {
    decode = alc.json_decode,
    encode = alc.json_encode,
  },
  fs = {
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
  },
  time = {
    now = alc.time,
  },
}
"#;

// ── Vendored evalframe source (26 files, include_str! from crates/algocline-engine/src/vendor/evalframe/) ──

const EF_INIT: &str = include_str!("../vendor/evalframe/init.lua");
const EF_STD: &str = include_str!("../vendor/evalframe/std.lua");
const EF_VARIANTS: &str = include_str!("../vendor/evalframe/variants.lua");

const EF_EVAL_INIT: &str = include_str!("../vendor/evalframe/eval/init.lua");
const EF_EVAL_LOADER: &str = include_str!("../vendor/evalframe/eval/loader.lua");
const EF_EVAL_REPORT: &str = include_str!("../vendor/evalframe/eval/report.lua");
const EF_EVAL_RUNNER: &str = include_str!("../vendor/evalframe/eval/runner.lua");
const EF_EVAL_STATS: &str = include_str!("../vendor/evalframe/eval/stats.lua");

const EF_MODEL_BINDING: &str = include_str!("../vendor/evalframe/model/binding.lua");
const EF_MODEL_CASE: &str = include_str!("../vendor/evalframe/model/case.lua");
const EF_MODEL_GRADER: &str = include_str!("../vendor/evalframe/model/grader.lua");
const EF_MODEL_SCORER: &str = include_str!("../vendor/evalframe/model/scorer.lua");

const EF_PRESETS_GRADERS: &str = include_str!("../vendor/evalframe/presets/graders.lua");
const EF_PRESETS_LLM_GRADERS: &str = include_str!("../vendor/evalframe/presets/llm_graders.lua");
const EF_PRESETS_SCORERS: &str = include_str!("../vendor/evalframe/presets/scorers.lua");

const EF_PROVIDERS_ALGOCLINE: &str = include_str!("../vendor/evalframe/providers/algocline.lua");
const EF_PROVIDERS_CLAUDE_CLI: &str = include_str!("../vendor/evalframe/providers/claude_cli.lua");
const EF_PROVIDERS_MOCK: &str = include_str!("../vendor/evalframe/providers/mock.lua");

const EF_SWARM_ACTIONS: &str = include_str!("../vendor/evalframe/swarm/actions.lua");
const EF_SWARM_ANALYSIS: &str = include_str!("../vendor/evalframe/swarm/analysis.lua");
const EF_SWARM_CONFIG: &str = include_str!("../vendor/evalframe/swarm/config.lua");
const EF_SWARM_ENV: &str = include_str!("../vendor/evalframe/swarm/env.lua");
const EF_SWARM_GRADERS: &str = include_str!("../vendor/evalframe/swarm/graders.lua");
const EF_SWARM_INIT: &str = include_str!("../vendor/evalframe/swarm/init.lua");
const EF_SWARM_PROVIDER: &str = include_str!("../vendor/evalframe/swarm/provider.lua");
const EF_SWARM_TRACE: &str = include_str!("../vendor/evalframe/swarm/trace.lua");

/// Vendored evalframe module registration table.
///
/// Order matters only for the top-level `evalframe` module which
/// requires all sub-modules — sub-modules are registered before the
/// top-level so `require` resolution follows the natural chain.
const EVALFRAME_PRELOADS: &[(&str, &str)] = &[
    // Leaf modules (no evalframe.* deps)
    ("evalframe.std", EF_STD),
    ("evalframe.variants", EF_VARIANTS),
    // eval/
    ("evalframe.eval.stats", EF_EVAL_STATS),
    ("evalframe.eval.loader", EF_EVAL_LOADER),
    ("evalframe.eval.report", EF_EVAL_REPORT),
    ("evalframe.eval.runner", EF_EVAL_RUNNER),
    ("evalframe.eval", EF_EVAL_INIT),
    // model/
    ("evalframe.model.case", EF_MODEL_CASE),
    ("evalframe.model.grader", EF_MODEL_GRADER),
    ("evalframe.model.scorer", EF_MODEL_SCORER),
    ("evalframe.model.binding", EF_MODEL_BINDING),
    // presets/
    ("evalframe.presets.graders", EF_PRESETS_GRADERS),
    ("evalframe.presets.scorers", EF_PRESETS_SCORERS),
    ("evalframe.presets.llm_graders", EF_PRESETS_LLM_GRADERS),
    // providers/
    ("evalframe.providers.algocline", EF_PROVIDERS_ALGOCLINE),
    ("evalframe.providers.claude_cli", EF_PROVIDERS_CLAUDE_CLI),
    ("evalframe.providers.mock", EF_PROVIDERS_MOCK),
    // swarm/
    ("evalframe.swarm.config", EF_SWARM_CONFIG),
    ("evalframe.swarm.env", EF_SWARM_ENV),
    ("evalframe.swarm.trace", EF_SWARM_TRACE),
    ("evalframe.swarm.provider", EF_SWARM_PROVIDER),
    ("evalframe.swarm.graders", EF_SWARM_GRADERS),
    ("evalframe.swarm.analysis", EF_SWARM_ANALYSIS),
    ("evalframe.swarm.actions", EF_SWARM_ACTIONS),
    ("evalframe.swarm", EF_SWARM_INIT),
    // Top-level (requires all above)
    ("evalframe", EF_INIT),
];

/// Register vendored evalframe on `package.preload` and inject the `std` global.
///
/// Call at session VM init (both `start_session` / `start_session_with_env`
/// / `install_for_pkg_test` / fork child VM) **before** the PRELUDE load,
/// so `alc.eval` (`prelude.lua`) can `pcall(require, "evalframe")` and
/// `evalframe.std` finds the batteries-compatible `std` global.
pub fn register_evalframe(lua: &Lua) -> LuaResult<()> {
    // 1. Inject std global (depends on `alc.*` being present).
    lua.load(EVALFRAME_STD_SHIM)
        .set_name("@evalframe_std_shim")
        .exec()?;

    // 2. Register preloads.
    let package: LuaTable = lua.globals().get("package")?;
    let preload: LuaTable = package.get("preload")?;
    for (mod_name, src) in EVALFRAME_PRELOADS.iter().copied() {
        let chunk_name = format!("@vendored:evalframe/{mod_name}");
        let loader = lua.create_function(move |lua, ()| {
            lua.load(src)
                .set_name(chunk_name.clone())
                .eval::<LuaValue>()
        })?;
        preload.set(mod_name, loader)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_alc_stub(lua: &Lua) {
        let alc: LuaTable = lua.create_table().unwrap();
        alc.set(
            "json_encode",
            lua.create_function(|_, _: LuaValue| Ok("{}".to_string()))
                .unwrap(),
        )
        .unwrap();
        alc.set(
            "json_decode",
            lua.create_function(|_, _: String| Ok(mlua::Value::Nil))
                .unwrap(),
        )
        .unwrap();
        alc.set(
            "time",
            lua.create_function(|_, ()| Ok(0.0_f64)).unwrap(),
        )
        .unwrap();
        alc.set(
            "llm",
            lua.create_function(|_, _: LuaValue| Ok("stub".to_string()))
                .unwrap(),
        )
        .unwrap();
        lua.globals().set("alc", alc).unwrap();
    }

    #[test]
    fn register_evalframe_injects_std_global() {
        let lua = Lua::new();
        install_alc_stub(&lua);

        register_evalframe(&lua).unwrap();
        let std_table: LuaTable = lua.globals().get("std").unwrap();
        let json: LuaTable = std_table.get("json").unwrap();
        let _decode: LuaFunction = json.get("decode").unwrap();
        let fs: LuaTable = std_table.get("fs").unwrap();
        let _read: LuaFunction = fs.get("read").unwrap();
        let _is_file: LuaFunction = fs.get("is_file").unwrap();
        let time: LuaTable = std_table.get("time").unwrap();
        let _now: LuaFunction = time.get("now").unwrap();
    }

    #[test]
    fn evalframe_top_level_require_succeeds() {
        let lua = Lua::new();
        install_alc_stub(&lua);

        register_evalframe(&lua).unwrap();
        // require("evalframe") should succeed and return a table with expected fields.
        let ef: LuaTable = lua
            .load(r#"return require("evalframe")"#)
            .eval()
            .unwrap();
        let _suite: LuaValue = ef.get("suite").unwrap();
        let _providers: LuaTable = ef.get("providers").unwrap();
        let _llm_graders: LuaTable = ef.get("llm_graders").unwrap();
    }
}
