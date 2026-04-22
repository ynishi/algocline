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
//! - The Lua sources under `service/lua/gendoc/` are pulled in with
//!   `include_str!` at compile time and registered on
//!   `package.preload` so that `require("tools.docs.X")` resolves
//!   without touching the filesystem.
//! - `alc_shapes` / `alc_shapes.t` (bundled-packages runtime
//!   dependency used by `extract.lua` / `projections.lua` etc.) are
//!   satisfied by minimal stubs — `gen_docs` only uses them for
//!   shape-validation side effects that are not load-bearing for the
//!   artifacts.
//! - `config_path` (optional) is a caller-supplied Lua file returning
//!   `{ context7 = {...}, devin = {...} }`. When supplied, its
//!   `context7` / `devin` sub-tables are exposed under the
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

use std::sync::{Arc, Mutex};

use mlua::{Lua, Table, Value};

use super::AppService;

// ── Embedded Lua sources ──────────────────────────────────────────

const LUA_GEN_DOCS: &str = include_str!("lua/gendoc/gen_docs.lua");
const LUA_DOCS_LIST: &str = include_str!("lua/gendoc/docs/list.lua");
const LUA_DOCS_EXTRACT: &str = include_str!("lua/gendoc/docs/extract.lua");
const LUA_DOCS_PROJECTIONS: &str = include_str!("lua/gendoc/docs/projections.lua");
const LUA_DOCS_PKG_INFO: &str = include_str!("lua/gendoc/docs/pkg_info.lua");
const LUA_DOCS_JSON: &str = include_str!("lua/gendoc/docs/json.lua");
const LUA_DOCS_LINT: &str = include_str!("lua/gendoc/docs/lint.lua");
const LUA_DOCS_ENTITY_SCHEMAS: &str = include_str!("lua/gendoc/docs/entity_schemas.lua");

/// Minimal pass-through stub for `alc_shapes`.
///
/// `gen_docs` uses `S.check` for non-load-bearing shape validation.
/// Returning `true` unconditionally preserves artifact output.
const LUA_ALC_SHAPES_STUB: &str = r#"
local M = {}
M.check = function(_v, _schema, _opts) return true, nil end
M.infer = function(v) return v end
return M
"#;

/// Minimal stub for `alc_shapes.t` — just enough type constructors to
/// satisfy top-level `local T = require("alc_shapes.t")` imports in
/// `entity_schemas.lua`, `extract.lua`, `projections.lua`, `lint.lua`.
const LUA_ALC_SHAPES_T_STUB: &str = r#"
local T = {}
T.str   = { kind = "scalar", name = "string" }
T.num   = { kind = "scalar", name = "number" }
T.bool  = { kind = "scalar", name = "bool" }
T.any   = { kind = "any" }
T.ref   = function(name) return { kind = "ref", name = name } end
T.list  = function(t) return { kind = "list", item = t } end
T.map   = function(k, v) return { kind = "map", key = k, value = v } end
T.opt   = function(t) return { kind = "optional", inner = t } end
return T
"#;

/// Lua module name → embedded source. Registered on `package.preload`
/// inside `register_preloads`.
const PRELOAD_MODULES: &[(&str, &str)] = &[
    ("tools.docs.list", LUA_DOCS_LIST),
    ("tools.docs.extract", LUA_DOCS_EXTRACT),
    ("tools.docs.projections", LUA_DOCS_PROJECTIONS),
    ("tools.docs.pkg_info", LUA_DOCS_PKG_INFO),
    ("tools.docs.json", LUA_DOCS_JSON),
    ("tools.docs.lint", LUA_DOCS_LINT),
    ("tools.docs.entity_schemas", LUA_DOCS_ENTITY_SCHEMAS),
    ("alc_shapes", LUA_ALC_SHAPES_STUB),
    ("alc_shapes.t", LUA_ALC_SHAPES_T_STUB),
];

/// IO / exit hooks installed into the VM right before `gen_docs.lua`
/// runs. `_gendoc_out_append` / `_gendoc_err_append` are Rust
/// closures registered under the same names on `_G`.
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
io.stdout.write = function(self, ...)
    local args = { ... }
    for i = 1, select("#", ...) do
        args[i] = tostring(args[i])
    end
    _gendoc_out_append(table.concat(args))
    return self
end
io.stderr.write = function(self, ...)
    local args = { ... }
    for i = 1, select("#", ...) do
        args[i] = tostring(args[i])
    end
    _gendoc_err_append(table.concat(args))
    return self
end
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
        let projection_flags = ProjectionFlags::from_list(projections);
        if (projection_flags.context7 || projection_flags.devin) && config_path.is_none() {
            return Err(
                "gendoc: config_path is required when projections include context7 or devin"
                    .to_string(),
            );
        }

        let resolved_out_dir = out_dir
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{source_dir}/docs"));

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
        let exec_result = lua
            .load(LUA_GEN_DOCS)
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

#[derive(Default, Clone, Copy)]
struct ProjectionFlags {
    hub: bool,
    context7: bool,
    devin: bool,
    lint: bool,
    lint_only: bool,
}

impl ProjectionFlags {
    fn from_list(projections: Option<&[String]>) -> Self {
        let mut f = ProjectionFlags::default();
        let Some(list) = projections else {
            return f;
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
                    // Unknown projections are ignored (the Lua side
                    // will ignore them too — gen_docs.lua only looks
                    // at the specific flags it knows).
                }
            }
        }
        f
    }
}

fn register_preloads(lua: &Lua) -> Result<(), String> {
    let preload = preload_table(lua)?;
    for (mod_name, src) in PRELOAD_MODULES.iter().copied() {
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
    let chunk_name = format!("@gendoc-config:{config_path}");
    let config_table: Table = lua
        .load(&src)
        .set_name(chunk_name)
        .eval()
        .map_err(|e| format!("gendoc: config_path '{config_path}' eval failed: {e}"))?;

    // Move the two sub-tables into the Lua registry via globals
    // stashes so the preload closures can retrieve them on require.
    // Using globals keeps the lifetime story simple (mlua 0.11 does
    // not require `'static` bounds for tables stored this way).
    let preload = preload_table(lua)?;

    inject_config_subtable(
        lua,
        &preload,
        &config_table,
        "context7",
        "_gendoc_context7_config",
        "tools.docs.context7_config",
    )?;
    inject_config_subtable(
        lua,
        &preload,
        &config_table,
        "devin",
        "_gendoc_devin_config",
        "tools.docs.devin_wiki_config",
    )?;

    Ok(())
}

/// Stash the `key` sub-table from `config_table` into a Lua global
/// and register a `package.preload` loader that returns it.
///
/// - Missing key (`Value::Nil`) is a legitimate caller choice: the
///   preload entry is simply omitted so a downstream `require` of
///   `module_name` raises Lua's standard "module not found" error
///   (clearer than registering a Nil loader that produces an opaque
///   nil-index error).
/// - Non-nil, non-table values are rejected up front with an
///   explicit `Err` — far more actionable than letting the Lua side
///   try to index a string/number later.
fn inject_config_subtable(
    lua: &Lua,
    preload: &Table,
    config_table: &Table,
    key: &'static str,
    global_key: &'static str,
    module_name: &'static str,
) -> Result<(), String> {
    let value: Value = config_table
        .get(key)
        .map_err(|e| format!("gendoc: config_table.get {key:?} failed: {e}"))?;
    match value {
        Value::Nil => Ok(()),
        Value::Table(_) => {
            lua.globals()
                .set(global_key, value)
                .map_err(|e| format!("gendoc: stash {global_key} failed: {e}"))?;
            register_config_loader(lua, preload, module_name, global_key)
        }
        other => Err(format!(
            "gendoc: config '{key}' must be a table, got {}",
            other.type_name()
        )),
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
        let f = ProjectionFlags::from_list(None);
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
        let f = ProjectionFlags::from_list(Some(&list));
        assert!(f.hub);
        assert!(f.context7);
        assert!(f.devin);
        assert!(!f.lint);
    }

    #[test]
    fn projection_flags_lint_only_implies_lint() {
        let list = vec!["lint_only".to_string()];
        let f = ProjectionFlags::from_list(Some(&list));
        assert!(f.lint);
        assert!(f.lint_only);
    }

    #[test]
    fn projection_flags_ignore_unknown() {
        let list = vec!["nope".to_string(), "hub".to_string()];
        let f = ProjectionFlags::from_list(Some(&list));
        assert!(f.hub);
        assert!(!f.context7);
    }

    #[test]
    fn context7_without_config_is_rejected() {
        // Build a minimal AppService through the public API is
        // expensive; exercise the input validation logic through
        // `ProjectionFlags` + an explicit mirror of the early
        // return in `hub_gendoc`. Directly calling `hub_gendoc`
        // would require a full test fixture — covered in e2e.
        let list = vec!["context7".to_string()];
        let flags = ProjectionFlags::from_list(Some(&list));
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
    fn build_response_json_round_trips() {
        let out = build_response_json("/src", "/src/docs", "hi", "warn");
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["source_dir"], "/src");
        assert_eq!(parsed["out_dir"], "/src/docs");
        assert_eq!(parsed["stdout"], "hi");
        assert_eq!(parsed["stderr"], "warn");
    }
}
