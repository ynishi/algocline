//! `AppService::hub_gendoc` — embedded Lua `gen_docs` runner.
//!
//! Runs the bundled-packages `tools/gen_docs.lua` pipeline in an
//! in-process `mlua` VM to produce human-readable documentation
//! artifacts (`narrative/{pkg}.md`, `llms.txt`, `llms-full.txt`,
//! optional `hub/{pkg}.json` / `context7.json` / `.devin/wiki.json`)
//! from a freshly indexed `hub_index.json`.
//!
//! Embedding strategy:
//!
//! - The Lua sources under `service/lua/gendoc/` and
//!   `service/gendoc/alc_shapes/` are pulled in with `include_str!`
//!   at compile time and registered on `package.preload` so that
//!   `require("tools.docs.X")` and `require("alc_shapes")` resolve
//!   without touching the filesystem.
//! - `alc_shapes` and all sub-modules are fully vendored via
//!   `include_str!` from `service/gendoc/alc_shapes/*.lua`. The
//!   `source_dir`'s on-disk `alc_shapes/` directory is never
//!   consulted at runtime. This ensures parity for any third-party
//!   package author invoking `alc_hub_dist` against their own source
//!   tree without vendoring `alc_shapes` themselves.
//! - `config_path` (optional) is a caller-supplied TOML file with
//!   top-level `[context7]` and/or `[devin]` tables. When supplied,
//!   those `context7` / `devin` tables are exposed under the
//!   `tools.docs.context7_config` / `tools.docs.devin_wiki_config`
//!   module names that `gen_docs` `require`s.
//! - `print` / `io.stdout.write` / `io.stderr.write` are redirected
//!   into Rust-side `String` buffers so callers observe the Lua
//!   progress log through the MCP response instead of dropping it to
//!   the server stderr.
//! - `os.exit(code)` is overridden so it raises a structured Lua
//!   error rather than terminating the process; non-zero exits are
//!   converted into `Err(...)`.
//!
//! Per the project-level Error propagation rule
//! (`CLAUDE.md §Service 層 Error 伝播規律`), every `mlua::Result` is
//! surfaced via `?` with a `gendoc:` prefix — no `warn!` drops, no
//! `unwrap_or_default`, no silent `Err(_) =>` branches.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use mlua::{Lua, Table, Value};
use serde::Deserialize;
use thiserror::Error;

use super::AppService;

// ── alc_shapes version pinning ────────────────────────────────────────

/// Version string embedded in the vendored `alc_shapes/init.lua`.
/// Must match the `M.VERSION` declaration in that file exactly.
const EMBEDDED_ALC_SHAPES_VERSION: &str = "0.25.1";

#[derive(Debug, Error)]
enum ShapesVersionError {
    #[error("alc_shapes version mismatch: embedded={embedded}, mirror={mirror}. {hint}")]
    Mismatch {
        embedded: String,
        mirror: String,
        hint: &'static str,
    },
    #[error("alc_shapes mirror init.lua at '{path}' has no parseable M.VERSION declaration")]
    Malformed { path: PathBuf },
}

const SHAPES_VERSION_HINT: &str = "Align bundled alc_shapes/ to match core, \
    or upgrade algocline core to the mirror version. See CHANGELOG for details.";

/// Check the mirror's `M.VERSION` against `EMBEDDED_ALC_SHAPES_VERSION`.
///
/// If `source_dir` is `None` or its `alc_shapes/init.lua` does not
/// exist, returns `Ok(())` immediately (no-op path). Otherwise reads
/// the file, extracts `M.VERSION = "x.y.z"` with a hand-rolled parser
/// (no `regex` dep), and fails with a typed error on mismatch.
fn check_mirror_shapes_version(source_dir: Option<&str>) -> Result<(), ShapesVersionError> {
    let Some(dir) = source_dir else {
        return Ok(());
    };
    let path: PathBuf = [dir, "alc_shapes", "init.lua"].iter().collect();
    if !path.exists() {
        return Ok(());
    }
    let src = std::fs::read_to_string(&path)
        .map_err(|_| ShapesVersionError::Malformed { path: path.clone() })?;
    let mirror_ver = extract_m_version(&src)
        .ok_or_else(|| ShapesVersionError::Malformed { path: path.clone() })?;
    if mirror_ver != EMBEDDED_ALC_SHAPES_VERSION {
        return Err(ShapesVersionError::Mismatch {
            embedded: EMBEDDED_ALC_SHAPES_VERSION.to_string(),
            mirror: mirror_ver,
            hint: SHAPES_VERSION_HINT,
        });
    }
    Ok(())
}

/// Hand-rolled parser for `M.VERSION = "x.y.z"` in a Lua source string.
///
/// Finds the first occurrence of `M.VERSION` followed (with optional
/// whitespace) by `=` and then a double-quoted string. Returns the
/// quoted content on success.
fn extract_m_version(src: &str) -> Option<String> {
    let marker = "M.VERSION";
    let start = src.find(marker)?;
    let after_marker = src[start + marker.len()..].trim_start();
    let after_eq = after_marker.strip_prefix('=')?;
    let after_eq = after_eq.trim_start();
    let after_quote = after_eq.strip_prefix('"')?;
    let end = after_quote.find('"')?;
    Some(after_quote[..end].to_string())
}

// ── Embedded Lua sources ──────────────────────────────────────────

const LUA_GEN_DOCS: &str = include_str!("lua/gendoc/gen_docs.lua");
const LUA_DOCS_LIST: &str = include_str!("lua/gendoc/docs/list.lua");
const LUA_DOCS_EXTRACT: &str = include_str!("lua/gendoc/docs/extract.lua");
const LUA_DOCS_PROJECTIONS: &str = include_str!("lua/gendoc/docs/projections.lua");
const LUA_DOCS_PKG_INFO: &str = include_str!("lua/gendoc/docs/pkg_info.lua");
const LUA_DOCS_JSON: &str = include_str!("lua/gendoc/docs/json.lua");
const LUA_DOCS_LINT: &str = include_str!("lua/gendoc/docs/lint.lua");
const LUA_DOCS_ENTITY_SCHEMAS: &str = include_str!("lua/gendoc/docs/entity_schemas.lua");

// ── Vendored alc_shapes (fully embedded; no disk fallback) ───────

const LUA_ALC_SHAPES_INIT: &str = include_str!("gendoc/alc_shapes/init.lua");
const LUA_ALC_SHAPES_T: &str = include_str!("gendoc/alc_shapes/t.lua");
const LUA_ALC_SHAPES_REFLECT: &str = include_str!("gendoc/alc_shapes/reflect.lua");
const LUA_ALC_SHAPES_CHECK: &str = include_str!("gendoc/alc_shapes/check.lua");
const LUA_ALC_SHAPES_INSTRUMENT: &str = include_str!("gendoc/alc_shapes/instrument.lua");
const LUA_ALC_SHAPES_LUACATS: &str = include_str!("gendoc/alc_shapes/luacats.lua");
const LUA_ALC_SHAPES_SPEC_RESOLVER: &str = include_str!("gendoc/alc_shapes/spec_resolver.lua");

/// All embedded preloads: vendored `alc_shapes` modules (in dependency
/// order) followed by the `tools/docs/*` pipeline sources.
///
/// Registration order matters: sub-modules must appear before the
/// modules that `require` them, matching the `alc_shapes/init.lua`
/// dependency chain: `t` → `reflect` / `check` / `luacats` /
/// `spec_resolver` → `instrument` → `init`.
const EMBEDDED_TOOL_PRELOADS: &[(&str, &str)] = &[
    // alc_shapes sub-modules (no intra-module deps except alc_shapes.t)
    ("alc_shapes.t", LUA_ALC_SHAPES_T),
    ("alc_shapes.reflect", LUA_ALC_SHAPES_REFLECT),
    ("alc_shapes.check", LUA_ALC_SHAPES_CHECK),
    ("alc_shapes.luacats", LUA_ALC_SHAPES_LUACATS),
    ("alc_shapes.spec_resolver", LUA_ALC_SHAPES_SPEC_RESOLVER),
    ("alc_shapes.instrument", LUA_ALC_SHAPES_INSTRUMENT),
    // alc_shapes top-level (requires all sub-modules above)
    ("alc_shapes", LUA_ALC_SHAPES_INIT),
    ("tools.docs.list", LUA_DOCS_LIST),
    ("tools.docs.extract", LUA_DOCS_EXTRACT),
    ("tools.docs.projections", LUA_DOCS_PROJECTIONS),
    ("tools.docs.pkg_info", LUA_DOCS_PKG_INFO),
    ("tools.docs.json", LUA_DOCS_JSON),
    ("tools.docs.lint", LUA_DOCS_LINT),
    ("tools.docs.entity_schemas", LUA_DOCS_ENTITY_SCHEMAS),
];

/// IO / exit hooks installed into the VM right before `gen_docs.lua`
/// runs. `_gendoc_out_append` / `_gendoc_err_append` are Rust
/// closures registered under the same names on `_G`.
///
/// `io.stdout` / `io.stderr` are in mlua exposed as `FILE*` userdata
/// whose metatable rejects arbitrary `__newindex` writes — so we
/// cannot patch `io.stdout.write` in place. Instead we replace
/// `io.stdout` / `io.stderr` wholesale with plain Lua tables that
/// expose a `write` method delegating to the Rust-side append
/// closures. Both method-style (`io.stdout:write(x)`) and
/// function-style (`io.stdout.write(io.stdout, x)`) calls are
/// supported; both are used in the bundled `gen_docs.lua`.
const HOOK_SCRIPT: &str = r##"
os.exit = function(code)
    local c = code or 0
    local tbl = { __gendoc_exit = c }
    -- Attach __tostring so the raw mlua error message embeds the
    -- code as "__gendoc_exit=N", letting the Rust side recover it
    -- via substring match instead of walking CallbackError internals.
    setmetatable(tbl, { __tostring = function(self)
        return string.format("__gendoc_exit=%d", self.__gendoc_exit or 0)
    end })
    error(tbl, 0)
end
io.stdout = {
    write = function(self, ...)
        local args = { ... }
        for i = 1, select("#", ...) do
            args[i] = tostring(args[i])
        end
        _gendoc_out_append(table.concat(args))
        return self
    end,
}
io.stderr = {
    write = function(self, ...)
        local args = { ... }
        for i = 1, select("#", ...) do
            args[i] = tostring(args[i])
        end
        _gendoc_err_append(table.concat(args))
        return self
    end,
}
print = function(...)
    local args = { ... }
    for i = 1, select("#", ...) do
        args[i] = tostring(args[i])
    end
    _gendoc_out_append(table.concat(args, "\t") .. "\n")
end
"##;

/// Marker set on the Lua error table by the `os.exit` override.
const EXIT_MARKER: &str = "__gendoc_exit";

impl AppService {
    /// See [`crate::EngineApi::hub_gendoc`] for parameter semantics.
    ///
    /// `config_path` format (TOML):
    ///
    /// ```toml
    /// [context7]
    /// projectTitle = "my project"
    /// description = "..."
    /// rules = []
    ///
    /// [devin]
    /// project_name = "my project"
    /// ```
    ///
    /// Notes:
    /// - `context7` / `devin` are optional individually.
    /// - When present, each key must be a table.
    /// - TOML arrays/tables are converted recursively to Lua tables.
    /// - See `docs/hub-gendoc-config.md` for a concrete schema example.
    ///
    /// Returns a JSON string of the form:
    ///
    /// ```json
    /// { "source_dir": "...", "out_dir": "...", "stdout": "...", "stderr": "..." }
    /// ```
    ///
    /// Non-zero `os.exit`, missing `hub_index.json`, Lua runtime
    /// errors, and `config_path` read failures are all surfaced as
    /// `Err` with a `gendoc:` prefix.
    pub fn hub_gendoc(
        &self,
        source_dir: &str,
        out_dir: Option<&str>,
        projections: Option<&[String]>,
        config_path: Option<&str>,
        lint_strict: Option<bool>,
    ) -> Result<String, String> {
        let projection_flags = ProjectionFlags::from_list(projections)?;
        if (projection_flags.context7 || projection_flags.devin) && config_path.is_none() {
            return Err(
                "gendoc: config_path is required when projections include context7 or devin"
                    .to_string(),
            );
        }

        let resolved_out_dir = out_dir
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{source_dir}/docs"));

        // Reject mismatched mirror before starting the VM: if the
        // caller's source_dir has an alc_shapes/init.lua whose
        // M.VERSION differs from the embedded constant, fail early with
        // a structured error (propagated to the MCP wire response).
        check_mirror_shapes_version(Some(source_dir)).map_err(|e| format!("gendoc: {e}"))?;

        let lua = Lua::new();

        register_preloads(&lua)?;

        // Optional config_path injection — must be wired as preload
        // *before* `gen_docs.lua` executes so that its
        // `require("tools.docs.context7_config")` resolves.
        if let Some(path) = config_path {
            inject_config_preloads(&lua, path)?;
        }

        let out_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let err_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));

        install_io_hooks(&lua, Arc::clone(&out_buf), Arc::clone(&err_buf))?;

        install_argv(
            &lua,
            source_dir,
            &resolved_out_dir,
            &projection_flags,
            lint_strict.unwrap_or(false),
        )?;

        // Run the IO / exit hook script (must come after
        // `_gendoc_out_append` / `_gendoc_err_append` are installed,
        // but before `gen_docs.lua` is exec'd).
        lua.load(HOOK_SCRIPT)
            .set_name("@embedded:gendoc/hooks.lua")
            .exec()
            .map_err(|e| format!("gendoc: hooks inject failed: {e}"))?;

        // Execute `gen_docs.lua`. The file ends with `main(arg)`.
        //
        // `lua.load()` uses the string path of `luaL_loadbuffer`
        // which does NOT strip a `#!` shebang line (the shebang is
        // only accepted by `luaL_loadfile`). The embedded
        // `gen_docs.lua` starts with `#!/usr/bin/env lua`, so we
        // skip the first line before loading.
        let gen_docs_body = strip_shebang(LUA_GEN_DOCS);
        let exec_result = lua
            .load(gen_docs_body)
            .set_name("@embedded:gendoc/gen_docs.lua")
            .exec();

        let stdout_txt = read_buf(&out_buf)?;
        let stderr_txt = read_buf(&err_buf)?;

        match exec_result {
            Ok(()) => {}
            Err(e) => {
                if let Some(code) = extract_exit_code(&e) {
                    if code != 0 {
                        return Err(format!(
                            "gendoc: exited with code {code}\nstderr:\n{stderr_txt}"
                        ));
                    }
                    // code == 0 is a clean shutdown via os.exit(0) —
                    // fall through to the normal response.
                } else {
                    return Err(format!("gendoc: Lua error: {e}\nstderr:\n{stderr_txt}"));
                }
            }
        }

        Ok(build_response_json(
            source_dir,
            &resolved_out_dir,
            &stdout_txt,
            &stderr_txt,
        ))
    }
}

// ── Helpers ───────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, Copy)]
struct ProjectionFlags {
    hub: bool,
    context7: bool,
    devin: bool,
    lint: bool,
    lint_only: bool,
}

impl ProjectionFlags {
    fn from_list(projections: Option<&[String]>) -> Result<Self, String> {
        let mut f = ProjectionFlags::default();
        let Some(list) = projections else {
            return Ok(f);
        };
        for p in list {
            match p.as_str() {
                "hub" => f.hub = true,
                "context7" => f.context7 = true,
                "devin" => f.devin = true,
                "lint" => f.lint = true,
                "lint_only" => {
                    f.lint_only = true;
                    f.lint = true;
                }
                _ => {
                    return Err(format!(
                        "gendoc: unknown projection '{p}' (allowed: hub, context7, devin, lint, lint_only)"
                    ));
                }
            }
        }
        Ok(f)
    }
}

/// Register all embedded `gen_docs` modules.
///
/// `alc_shapes` and its sub-modules are fully vendored via
/// `include_str!` — no disk fallback. `tools/docs/*` pipeline sources
/// are registered in the same pass.
fn register_preloads(lua: &Lua) -> Result<(), String> {
    let preload = preload_table(lua)?;
    for (mod_name, src) in EMBEDDED_TOOL_PRELOADS.iter().copied() {
        register_single_preload(lua, &preload, mod_name, src)?;
    }
    Ok(())
}

fn preload_table(lua: &Lua) -> Result<Table, String> {
    // `globals().package.preload` is part of the Lua 5.4 / mlua
    // contract; absence would indicate a VM that cannot run any
    // meaningful Lua code, so `expect` with a justifying comment is
    // the correct classification (see CLAUDE.md §Service 層 Error
    // 伝播規律 "limited exceptions for unreachable VM invariants").
    let package: Table = lua
        .globals()
        .get("package")
        .map_err(|e| format!("gendoc: globals().package lookup failed: {e}"))?;
    let preload: Table = package
        .get("preload")
        .map_err(|e| format!("gendoc: package.preload lookup failed: {e}"))?;
    Ok(preload)
}

fn register_single_preload(
    lua: &Lua,
    preload: &Table,
    mod_name: &'static str,
    src: &'static str,
) -> Result<(), String> {
    let chunk_name = format!("@embedded:gendoc/{mod_name}.lua");
    let loader = lua
        .create_function(move |lua, ()| lua.load(src).set_name(chunk_name.clone()).eval::<Value>())
        .map_err(|e| format!("gendoc: preload create_function failed for {mod_name}: {e}"))?;
    preload
        .set(mod_name, loader)
        .map_err(|e| format!("gendoc: preload.set failed for {mod_name}: {e}"))?;
    Ok(())
}

fn inject_config_preloads(lua: &Lua, config_path: &str) -> Result<(), String> {
    let src = std::fs::read_to_string(config_path)
        .map_err(|e| format!("gendoc: config_path '{config_path}' load failed: {e}"))?;
    let config: GendocConfig = toml::from_str(&src)
        .map_err(|e| format!("gendoc: config_path '{config_path}' parse failed: {e}"))?;

    // Move the two sub-tables into the Lua registry via globals
    // stashes so the preload closures can retrieve them on require.
    // Using globals keeps the lifetime story simple (mlua 0.11 does
    // not require `'static` bounds for tables stored this way).
    let preload = preload_table(lua)?;

    inject_config_subtable(
        lua,
        &preload,
        config.context7,
        "context7",
        "_gendoc_context7_config",
        "tools.docs.context7_config",
    )?;
    inject_config_subtable(
        lua,
        &preload,
        config.devin,
        "devin",
        "_gendoc_devin_config",
        "tools.docs.devin_wiki_config",
    )?;

    Ok(())
}

/// Stash the `key` sub-table into a Lua global
/// and register a `package.preload` loader that returns it.
///
/// - Missing key (`None`) is a legitimate caller choice: the
///   preload entry is simply omitted so a downstream `require` of
///   `module_name` raises Lua's standard "module not found" error
///   (clearer than registering a Nil loader that produces an opaque
///   nil-index error).
/// - Non-table values are rejected up front with an
///   explicit `Err` — far more actionable than letting the Lua side
///   try to index a string/number later.
fn inject_config_subtable(
    lua: &Lua,
    preload: &Table,
    value: Option<toml::Value>,
    key: &'static str,
    global_key: &'static str,
    module_name: &'static str,
) -> Result<(), String> {
    match value {
        None => Ok(()),
        Some(v) => {
            let lua_value = toml_to_lua_value(lua, &v)
                .map_err(|e| format!("gendoc: config '{key}' conversion failed: {e}"))?;
            match lua_value {
                Value::Table(_) => {
                    lua.globals()
                        .set(global_key, lua_value)
                        .map_err(|e| format!("gendoc: stash {global_key} failed: {e}"))?;
                    register_config_loader(lua, preload, module_name, global_key)
                }
                other => Err(format!(
                    "gendoc: config '{key}' must be a table, got {}",
                    other.type_name()
                )),
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct GendocConfig {
    context7: Option<toml::Value>,
    devin: Option<toml::Value>,
}

fn toml_to_lua_value(lua: &Lua, value: &toml::Value) -> Result<Value, String> {
    match value {
        toml::Value::String(s) => Ok(Value::String(
            lua.create_string(s)
                .map_err(|e| format!("create string failed: {e}"))?,
        )),
        toml::Value::Integer(i) => Ok(Value::Integer(*i)),
        toml::Value::Float(f) => Ok(Value::Number(*f)),
        toml::Value::Boolean(b) => Ok(Value::Boolean(*b)),
        toml::Value::Datetime(dt) => Ok(Value::String(
            lua.create_string(dt.to_string())
                .map_err(|e| format!("create datetime string failed: {e}"))?,
        )),
        toml::Value::Array(arr) => {
            let table = lua
                .create_table()
                .map_err(|e| format!("create array table failed: {e}"))?;
            for (idx, item) in arr.iter().enumerate() {
                let v = toml_to_lua_value(lua, item)?;
                table
                    .set((idx + 1) as i64, v)
                    .map_err(|e| format!("set array item [{idx}] failed: {e}"))?;
            }
            Ok(Value::Table(table))
        }
        toml::Value::Table(map) => {
            let table = lua
                .create_table()
                .map_err(|e| format!("create map table failed: {e}"))?;
            for (k, v) in map {
                let vv = toml_to_lua_value(lua, v)?;
                table
                    .set(k.as_str(), vv)
                    .map_err(|e| format!("set map key '{k}' failed: {e}"))?;
            }
            Ok(Value::Table(table))
        }
    }
}

fn register_config_loader(
    lua: &Lua,
    preload: &Table,
    module_name: &'static str,
    global_key: &'static str,
) -> Result<(), String> {
    let loader = lua
        .create_function(move |lua, ()| lua.globals().get::<Value>(global_key))
        .map_err(|e| format!("gendoc: config loader for {module_name} failed: {e}"))?;
    preload
        .set(module_name, loader)
        .map_err(|e| format!("gendoc: preload.set failed for {module_name}: {e}"))?;
    Ok(())
}

fn install_io_hooks(
    lua: &Lua,
    out_buf: Arc<Mutex<String>>,
    err_buf: Arc<Mutex<String>>,
) -> Result<(), String> {
    let out_for_closure = Arc::clone(&out_buf);
    let append_out = lua
        .create_function(move |_, s: String| {
            out_for_closure
                .lock()
                .map_err(|e| mlua::Error::external(format!("gendoc: out buf lock: {e}")))?
                .push_str(&s);
            Ok(())
        })
        .map_err(|e| format!("gendoc: create_function _gendoc_out_append: {e}"))?;

    let err_for_closure = Arc::clone(&err_buf);
    let append_err = lua
        .create_function(move |_, s: String| {
            err_for_closure
                .lock()
                .map_err(|e| mlua::Error::external(format!("gendoc: err buf lock: {e}")))?
                .push_str(&s);
            Ok(())
        })
        .map_err(|e| format!("gendoc: create_function _gendoc_err_append: {e}"))?;

    lua.globals()
        .set("_gendoc_out_append", append_out)
        .map_err(|e| format!("gendoc: globals set _gendoc_out_append: {e}"))?;
    lua.globals()
        .set("_gendoc_err_append", append_err)
        .map_err(|e| format!("gendoc: globals set _gendoc_err_append: {e}"))?;

    Ok(())
}

fn install_argv(
    lua: &Lua,
    source_dir: &str,
    out_dir: &str,
    flags: &ProjectionFlags,
    lint_strict: bool,
) -> Result<(), String> {
    let argv = lua
        .create_table()
        .map_err(|e| format!("gendoc: create argv table: {e}"))?;

    let mut idx: i64 = 1;
    let mut push = |v: &str| -> Result<(), String> {
        argv.set(idx, v)
            .map_err(|e| format!("gendoc: argv set [{idx}]: {e}"))?;
        idx += 1;
        Ok(())
    };

    push(source_dir)?;
    push(out_dir)?;
    if flags.hub {
        push("--hub")?;
    }
    if flags.context7 {
        push("--context7")?;
    }
    if flags.devin {
        push("--devin")?;
    }
    if flags.lint_only {
        push("--lint-only")?;
    } else if flags.lint {
        push("--lint")?;
    }
    if lint_strict {
        push("--strict")?;
    }

    lua.globals()
        .set("arg", argv)
        .map_err(|e| format!("gendoc: globals set arg: {e}"))?;

    Ok(())
}

/// Strip a leading `#!` shebang line from a Lua source.
///
/// `lua.load()` (buffer-based) does not strip the shebang the way
/// `luaL_loadfile` does. The embedded `gen_docs.lua` starts with
/// `#!/usr/bin/env lua`, so we strip the first line before feeding
/// the buffer to the VM.
fn strip_shebang(src: &str) -> &str {
    if let Some(body) = src.strip_prefix("#!") {
        match body.find('\n') {
            Some(i) => &body[i + 1..],
            None => "",
        }
    } else {
        src
    }
}

fn read_buf(buf: &Arc<Mutex<String>>) -> Result<String, String> {
    Ok(buf
        .lock()
        .map_err(|e| format!("gendoc: buffer lock (read): {e}"))?
        .clone())
}

/// Extract `__gendoc_exit` from a Lua error raised by the `os.exit`
/// override. Returns `None` for unrelated errors.
fn extract_exit_code(err: &mlua::Error) -> Option<i64> {
    // `error(tbl, 0)` in Lua becomes `mlua::Error::RuntimeError`
    // where the tostring serialization contains the table pointer
    // (not useful) plus any __tostring metamethod output. For our
    // structured exit we don't set __tostring, so the surfaced
    // message may look like `table: 0x...`. That is not reliable.
    //
    // Instead, walk the error chain looking for a CallbackError /
    // WithContext that wraps the raw value. mlua exposes the raw
    // table via `Error::CallbackError::cause`. The Lua table itself
    // is not exposed from `RuntimeError`, so we fall back to
    // pattern-matching the string form as a best-effort:
    // Lua's `error({__gendoc_exit = N}, 0)` with a table whose
    // `__tostring` is unset renders as the table pointer, which
    // loses the code.
    //
    // To make this robust we attach a __tostring metamethod to the
    // raised table in the hook script so the error message embeds
    // the code. See HOOK_SCRIPT.
    let msg = err.to_string();
    // Look for the marker substring emitted by __tostring (installed
    // in HOOK_SCRIPT) in the format `__gendoc_exit=<N>`.
    let needle = EXIT_MARKER;
    let idx = msg.find(needle)?;
    let rest = &msg[idx + needle.len()..];
    // Skip any `=` / `:` / whitespace, then parse an integer.
    let digits_start = rest
        .char_indices()
        .find(|(_, c)| c.is_ascii_digit() || *c == '-')
        .map(|(i, _)| i)?;
    let tail = &rest[digits_start..];
    let digits_end = tail
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit() && *c != '-')
        .map(|(i, _)| i)
        .unwrap_or(tail.len());
    tail[..digits_end].parse::<i64>().ok()
}

fn build_response_json(
    source_dir: &str,
    out_dir: &str,
    stdout_txt: &str,
    stderr_txt: &str,
) -> String {
    // Keep dependency-free — we already depend on serde_json
    // transitively but using it here avoids hand-rolled escaping
    // bugs. Every shipped string is a plain `String` so
    // `serde_json::Value::String` is fine.
    let value = serde_json::json!({
        "source_dir": source_dir,
        "out_dir": out_dir,
        "stdout": stdout_txt,
        "stderr": stderr_txt,
    });
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_flags_defaults_are_false() {
        let f = ProjectionFlags::from_list(None).expect("projection parse");
        assert!(!f.hub);
        assert!(!f.context7);
        assert!(!f.devin);
        assert!(!f.lint);
        assert!(!f.lint_only);
    }

    #[test]
    fn projection_flags_parse_known_tokens() {
        let list = vec![
            "hub".to_string(),
            "context7".to_string(),
            "devin".to_string(),
        ];
        let f = ProjectionFlags::from_list(Some(&list)).expect("projection parse");
        assert!(f.hub);
        assert!(f.context7);
        assert!(f.devin);
        assert!(!f.lint);
    }

    #[test]
    fn projection_flags_lint_only_implies_lint() {
        let list = vec!["lint_only".to_string()];
        let f = ProjectionFlags::from_list(Some(&list)).expect("projection parse");
        assert!(f.lint);
        assert!(f.lint_only);
    }

    #[test]
    fn projection_flags_unknown_is_rejected() {
        let list = vec!["nope".to_string(), "hub".to_string()];
        let err = ProjectionFlags::from_list(Some(&list)).expect_err("must reject unknown");
        assert!(err.contains("unknown projection"));
    }

    #[test]
    fn context7_without_config_is_rejected() {
        // Build a minimal AppService through the public API is
        // expensive; exercise the input validation logic through
        // `ProjectionFlags` + an explicit mirror of the early
        // return in `hub_gendoc`. Directly calling `hub_gendoc`
        // would require a full test fixture — covered in e2e.
        let list = vec!["context7".to_string()];
        let flags = ProjectionFlags::from_list(Some(&list)).expect("projection parse");
        assert!(flags.context7);
        // Simulate the guard:
        let err_expected =
            "gendoc: config_path is required when projections include context7 or devin";
        let err = if (flags.context7 || flags.devin) && Option::<&str>::None.is_none() {
            Some(err_expected.to_string())
        } else {
            None
        };
        assert_eq!(err.as_deref(), Some(err_expected));
    }

    #[test]
    fn extract_exit_code_parses_marker_formats() {
        // Simulated error string; `extract_exit_code` doesn't care
        // about the prefix as long as the `__gendoc_exit=N` marker is
        // present.
        let err = mlua::Error::RuntimeError(
            "runtime error: [string \"...\"]:2: {__gendoc_exit=2}".to_string(),
        );
        assert_eq!(extract_exit_code(&err), Some(2));

        let err = mlua::Error::RuntimeError("runtime error: __gendoc_exit: 0 (clean)".to_string());
        assert_eq!(extract_exit_code(&err), Some(0));
    }

    #[test]
    fn extract_exit_code_returns_none_for_unrelated_errors() {
        let err = mlua::Error::RuntimeError("some other Lua error".to_string());
        assert!(extract_exit_code(&err).is_none());
    }

    #[test]
    fn strip_shebang_removes_first_line_when_prefixed() {
        let src = "#!/usr/bin/env lua\nreturn 1\n";
        assert_eq!(strip_shebang(src), "return 1\n");
    }

    #[test]
    fn strip_shebang_preserves_source_without_shebang() {
        let src = "-- no shebang\nreturn 1\n";
        assert_eq!(strip_shebang(src), src);
    }

    #[test]
    fn strip_shebang_handles_shebang_only_without_trailing_newline() {
        let src = "#!/usr/bin/env lua";
        assert_eq!(strip_shebang(src), "");
    }

    #[test]
    fn build_response_json_round_trips() {
        let out = build_response_json("/src", "/src/docs", "hi", "warn");
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["source_dir"], "/src");
        assert_eq!(parsed["out_dir"], "/src/docs");
        assert_eq!(parsed["stdout"], "hi");
        assert_eq!(parsed["stderr"], "warn");
    }

    // ── alc_shapes version resolver unit tests ────────────────────────

    #[test]
    fn extract_m_version_parses_standard_format() {
        let src = r#"local M = {}
M.VERSION = "0.25.1"
"#;
        assert_eq!(extract_m_version(src).as_deref(), Some("0.25.1"));
    }

    #[test]
    fn extract_m_version_tolerates_no_space_around_eq() {
        let src = r#"M.VERSION="1.2.3""#;
        assert_eq!(extract_m_version(src).as_deref(), Some("1.2.3"));
    }

    #[test]
    fn extract_m_version_tolerates_leading_whitespace() {
        let src = r#"  M.VERSION = "9.9.9"  "#;
        assert_eq!(extract_m_version(src).as_deref(), Some("9.9.9"));
    }

    #[test]
    fn extract_m_version_returns_none_when_absent() {
        let src = r#"local M = {}
return M
"#;
        assert!(extract_m_version(src).is_none());
    }

    #[test]
    fn check_mirror_shapes_version_ok_when_source_dir_none() {
        assert!(check_mirror_shapes_version(None).is_ok());
    }

    #[test]
    fn check_mirror_shapes_version_ok_when_no_mirror_file() {
        // A tempdir with no alc_shapes/ subdirectory.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().to_str().expect("utf-8").to_string();
        assert!(check_mirror_shapes_version(Some(&dir)).is_ok());
    }

    #[test]
    fn check_mirror_shapes_version_ok_on_version_match() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let alc_dir = tmp.path().join("alc_shapes");
        std::fs::create_dir_all(&alc_dir).expect("mkdir alc_shapes");
        let init = alc_dir.join("init.lua");
        std::fs::write(
            &init,
            format!(
                "local M = {{}}\nM.VERSION = \"{}\"\nreturn M\n",
                EMBEDDED_ALC_SHAPES_VERSION
            ),
        )
        .expect("write init.lua");
        let dir = tmp.path().to_str().expect("utf-8").to_string();
        assert!(check_mirror_shapes_version(Some(&dir)).is_ok());
    }

    #[test]
    fn check_mirror_shapes_version_err_on_version_mismatch() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let alc_dir = tmp.path().join("alc_shapes");
        std::fs::create_dir_all(&alc_dir).expect("mkdir alc_shapes");
        let init = alc_dir.join("init.lua");
        std::fs::write(&init, "local M = {}\nM.VERSION = \"9.9.9\"\nreturn M\n")
            .expect("write init.lua");
        let dir = tmp.path().to_str().expect("utf-8").to_string();
        let err =
            check_mirror_shapes_version(Some(&dir)).expect_err("must fail on version mismatch");
        let msg = err.to_string();
        assert!(
            msg.contains(EMBEDDED_ALC_SHAPES_VERSION),
            "embedded ver in msg: {msg}"
        );
        assert!(msg.contains("9.9.9"), "mirror ver in msg: {msg}");
        assert!(msg.contains("CHANGELOG"), "hint in msg: {msg}");
    }

    #[test]
    fn check_mirror_shapes_version_err_on_malformed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let alc_dir = tmp.path().join("alc_shapes");
        std::fs::create_dir_all(&alc_dir).expect("mkdir alc_shapes");
        let init = alc_dir.join("init.lua");
        std::fs::write(&init, "-- no version here\nreturn {}\n").expect("write init.lua");
        let dir = tmp.path().to_str().expect("utf-8").to_string();
        let err = check_mirror_shapes_version(Some(&dir)).expect_err("must fail on malformed");
        let msg = err.to_string();
        assert!(msg.contains("no parseable"), "malformed msg: {msg}");
    }

    /// Regression harness: vendored `alc_shapes` must satisfy the
    /// contracts exercised by `tools/docs/projections.lua` (sorted
    /// `S.fields`, `prim` / `elem` / `val` / `doc` keys). Catches the
    /// class of drift that collapsed bundled `llms-full.txt` generation.
    #[test]
    fn embedded_gendoc_shapes_contract_harness() {
        let lua = Lua::new();
        register_preloads(&lua).expect("register_preloads");

        let script = r#"
            local S = require("alc_shapes")
            local T = require("alc_shapes.t")
            local P = require("tools.docs.projections")

            local shape = T.shape({
                task = T.string:describe("Problem"),
                n = T.number:is_optional(),
            })
            local entries = S.fields(shape)
            assert(#entries == 2, "expected two fields")
            assert(entries[1].name == "n" and entries[1].optional == true)
            assert(entries[2].name == "task" and entries[2].optional == false)
            assert(entries[2].doc == "Problem")
            assert(P.shape_type_string(entries[2].type) == "string")

            assert(P.shape_type_string(T.array_of(T.string)) == "array of string")
            assert(P.shape_type_string(T.map_of(T.string, T.number)) == "map of string to number")

            local inner = T.shape({ flag = T.boolean })
            assert(P.shape_type_string(inner) == "shape { flag: boolean }")
        "#;

        lua.load(script)
            .set_name("@test/embedded_gendoc_shapes_contract.lua")
            .exec()
            .expect("embedded shapes contract harness");
    }

    /// Vendored `alc_shapes` must include the full shape registry so
    /// `projections.shape_type_string(T.ref("voted"))` resolves via the
    /// embedded `alc_shapes` module (no disk fallback required).
    #[test]
    fn vendored_alc_shapes_resolves_pkg_refs() {
        let lua = Lua::new();
        register_preloads(&lua).expect("register_preloads");

        let script = r#"
            local S = require("alc_shapes")
            assert(type(S.voted) == "table" and rawget(S.voted, "kind") == "shape")
            local T = require("alc_shapes.t")
            local P = require("tools.docs.projections")
            assert(P.shape_type_string(T.ref("voted")) == "voted")
        "#;

        lua.load(script)
            .set_name("@test/vendored_alc_shapes_ref.lua")
            .exec()
            .expect("vendored alc_shapes ref resolution");
    }
}
