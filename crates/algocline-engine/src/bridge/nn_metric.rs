//! `alc.nn.metric.*` — distribution distance / entropy primitives plus a
//! per-VM Lua-side `registry` for name → fn lookup (feature `nn`).
//!
//! # Layer boundary
//!
//! The four primitives (`kl`, `js`, `tvd`, `entropy`) are thin bridges over
//! [`algocline_nn::metric`]: Lua tables come in, `Vec<f32>` cross the FFI,
//! typed [`algocline_nn::metric::MetricError`] variants surface as
//! [`LuaError::external`] with the primitive name preserved in the display
//! string. The primitives themselves live in Rust so the (a) validation
//! contract stays authoritative and (b) domain-specific composition
//! (`gameai_metrics`) reuses them without touching FFI details.
//!
//! # Why the registry is Pure Lua
//!
//! `alc.nn.metric.registry` is installed as a plain Lua table on VM boot.
//! Holding registered callbacks Rust-side would require crossing the mlua
//! ownership boundary (`RegistryKey`, `Send + 'static` on
//! `Fn(&Lua, LuaValue)`), while a Lua table gives the same name-resolved
//! lookup with no cross-boundary lifetimes. Scope is naturally per-VM: a
//! fresh VM starts with an empty registry, and `gameai_metrics` (or any
//! other pkg) `register`s into it after `require`.
//!
//! The trainer `on_ckpt` hook reaches the registry via
//! `alc.nn.metric.registry.evaluate(name, ctx)` — a single Lua entry point
//! that centralises the "look up + call" pair so callers never handle a
//! missing metric silently.

use mlua::prelude::*;

use algocline_nn::metric;

/// Register `alc.nn.metric.*` onto the pre-existing `alc.nn` table.
///
/// Called from [`super::register`] after [`super::register_nn`] has
/// populated `alc.nn`. Installs the four primitives (`kl` / `js` / `tvd`
/// / `entropy`) followed by the pure-Lua `registry` sub-table.
///
/// The registry is initialised empty on every VM: consumers
/// (`gameai_metrics` pkg, spec fixtures) call `register(name, fn)` to
/// populate it, and the trainer hook (`on_ckpt`) calls
/// `evaluate(name, ctx)` to dispatch by name.
pub(super) fn register_nn_metric(lua: &Lua, nn_table: &LuaTable) -> LuaResult<()> {
    let metric_ns = lua.create_table()?;

    // ── Primitives ──────────────────────────────────────────────────
    // Each closure pulls the pairwise / single-distribution arguments
    // as `Vec<f32>` (mlua's built-in `FromLuaMulti` handles Lua array
    // tables), delegates to the Rust primitive, and surfaces
    // `MetricError` via `LuaError::external`. The Display impl of
    // `MetricError` already prefixes `metric:` so the error string is
    // self-describing on the Lua side.

    let kl = lua.create_function(|_, (p, q): (Vec<f32>, Vec<f32>)| {
        metric::kl(&p, &q).map_err(LuaError::external)
    })?;
    metric_ns.set("kl", kl)?;

    let js = lua.create_function(|_, (p, q): (Vec<f32>, Vec<f32>)| {
        metric::js(&p, &q).map_err(LuaError::external)
    })?;
    metric_ns.set("js", js)?;

    let tvd = lua.create_function(|_, (p, q): (Vec<f32>, Vec<f32>)| {
        metric::tvd(&p, &q).map_err(LuaError::external)
    })?;
    metric_ns.set("tvd", tvd)?;

    let entropy =
        lua.create_function(|_, p: Vec<f32>| metric::entropy(&p).map_err(LuaError::external))?;
    metric_ns.set("entropy", entropy)?;

    // ── Registry (Pure Lua, per-VM) ─────────────────────────────────
    // Kept in Lua for two reasons:
    //   1. `Fn(&Lua, LuaValue)` callbacks would require `Send + 'static`
    //      and a `RegistryKey` per entry — the Lua-side lookup gives the
    //      same name→fn dispatch with zero FFI ceremony.
    //   2. Scope is naturally per-VM: a fresh VM boots with an empty
    //      table, and `gameai_metrics` re-registers on every `require`.
    //
    // `register` / `evaluate` reject invalid input loudly (`error(..)`
    // with `level = 2` so the caller frame is blamed). `get` returns
    // `nil` on miss (query surface — the caller inspects). `list`
    // returns a sorted name array so downstream consumers get a
    // deterministic view of what is registered.
    //
    // The chunk receives `metric_ns` as its sole vararg and attaches
    // `registry` to it directly. This avoids depending on the `alc`
    // global existing yet: `register()` in `bridge/mod.rs` installs
    // everything onto a fresh `alc_table` before the caller publishes
    // it as `_G.alc`, so a snippet that read `alc.nn.metric.registry`
    // would fail with a nil-index error here.
    let installer: LuaFunction = lua
        .load(REGISTRY_INSTALL)
        .set_name("@alc.nn.metric.registry")
        .into_function()?;
    installer.call::<()>(metric_ns.clone())?;

    // Publish after the registry is attached so the surface is complete
    // on first read (no partial `alc.nn.metric` visible from Lua).
    nn_table.set("metric", metric_ns)?;

    Ok(())
}

/// Pure-Lua installer for `alc.nn.metric.registry`.
///
/// Executed once per VM in [`register_nn_metric`] after the primitives
/// are attached. The chunk takes the metric namespace as its sole
/// vararg (`local metric = ...`) so it does not depend on the `alc`
/// global being published yet.
///
/// Kept as a `const &str` (not `include_str!`) because the snippet is
/// short, self-contained, and belongs conceptually with the Rust install
/// path — a separate `.lua` file would fragment the SoT across two
/// files with no additional benefit.
const REGISTRY_INSTALL: &str = r#"
local metric = ...
local registry = { _registry = {} }

registry.register = function(name, fn)
    if type(name) ~= "string" or name == "" then
        error("alc.nn.metric.registry.register: name must be non-empty string", 2)
    end
    if type(fn) ~= "function" then
        error("alc.nn.metric.registry.register: fn must be function", 2)
    end
    registry._registry[name] = fn
end

registry.get = function(name)
    return registry._registry[name]
end

registry.evaluate = function(name, ctx)
    local fn = registry._registry[name]
    if not fn then
        error("alc.nn.metric.registry.evaluate: no metric registered as '" .. tostring(name) .. "'", 2)
    end
    return fn(ctx or {})
end

registry.list = function()
    local names = {}
    for k, _ in pairs(registry._registry) do
        table.insert(names, k)
    end
    table.sort(names)
    return names
end

metric.registry = registry
"#;
