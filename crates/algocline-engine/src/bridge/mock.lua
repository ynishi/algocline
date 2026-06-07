--- bridge/mock.lua — pkg_test sandbox mock API
---
--- Installed by `bridge::install_for_pkg_test` after the standard
--- `register()` flow.  Provides spec-author-facing mock primitives that
--- swap `alc.*` entries via table mutation (no Rust support required).
---
--- Public surface (Lua globals + `alc.*` field):
---   with_alc(overrides, fn)          — scoped mock, auto-restore on return/error
---   alc_mock.install(overrides)      — persistent override for before_each setup
---   alc_mock.restore()               — restore all entries saved by install/with_alc
---   alc.spy(name, default_fn?)       — install observable wrapper; returns spy handle
---
--- Invariants:
--- * Nested `with_alc` is supported via a stack; inner exit restores 1 level.
--- * `pcall` wraps the body so panic still restores before re-throwing.
--- * Spy handle: { call_count, calls = { { args = {...} }, ... }, reset() }.

local M = {}

-- ─── Restore stack ──────────────────────────────────────────────────────────
-- Each frame is a list of { key = string, prev = any (nil sentinel preserved) }
-- entries.  We track presence explicitly because `nil` is a legal prior value.
local _stack = {}

local function _snapshot(keys)
    local frame = {}
    for _, k in ipairs(keys) do
        frame[#frame + 1] = { key = k, prev = alc[k], had = (alc[k] ~= nil) }
    end
    return frame
end

local function _apply(overrides)
    local keys = {}
    for k, _ in pairs(overrides) do
        keys[#keys + 1] = k
    end
    local frame = _snapshot(keys)
    for k, v in pairs(overrides) do
        alc[k] = v
    end
    return frame
end

local function _restore_frame(frame)
    for _, entry in ipairs(frame) do
        if entry.had then
            alc[entry.key] = entry.prev
        else
            alc[entry.key] = nil
        end
    end
end

-- ─── with_alc(overrides, fn) ────────────────────────────────────────────────
--- Scoped mock: replace alc.* entries for the duration of fn(), then restore.
---
--- @param overrides table  Map of `name -> function` to install on `alc`.
--- @param fn function       Body executed under the override.
--- @return any              Whatever fn() returns.
function with_alc(overrides, fn)
    if type(overrides) ~= "table" then
        error("with_alc: overrides must be a table", 2)
    end
    if type(fn) ~= "function" then
        error("with_alc: fn must be a function", 2)
    end
    local frame = _apply(overrides)
    local ok, result = pcall(fn)
    _restore_frame(frame)
    if not ok then
        error(result, 0)
    end
    return result
end

-- ─── alc_mock.install / restore ─────────────────────────────────────────────
--- Persistent override for `before_each`-style setup.  Each `install` pushes
--- a frame; `restore()` pops the most-recent frame.
function M.install(overrides)
    if type(overrides) ~= "table" then
        error("alc_mock.install: overrides must be a table", 2)
    end
    local frame = _apply(overrides)
    _stack[#_stack + 1] = frame
end

function M.restore()
    local n = #_stack
    if n == 0 then
        return
    end
    local frame = _stack[n]
    _stack[n] = nil
    _restore_frame(frame)
end

--- Drop every pushed frame (test teardown safety net).
function M.restore_all()
    while #_stack > 0 do
        M.restore()
    end
end

-- ─── alc.spy(name, default_fn?) ─────────────────────────────────────────────
--- Wrap `alc[name]` with an observable proxy.  Returns a handle exposing
--- `call_count`, `calls = { { args = {...} }, ... }`, and `reset()`.
--- The previous `alc[name]` is preserved on the same restore stack as
--- `with_alc` / `alc_mock.install`, so spies installed inside `with_alc`
--- are torn down on body exit.
function alc.spy(name, default_fn)
    if type(name) ~= "string" then
        error("alc.spy: name must be a string", 2)
    end
    local impl = default_fn
    if impl == nil then
        impl = alc[name]
    end
    local handle = {
        call_count = 0,
        calls = {},
        reset = function(self)
            self.call_count = 0
            self.calls = {}
        end,
    }
    local wrapper = function(...)
        handle.call_count = handle.call_count + 1
        handle.calls[#handle.calls + 1] = { args = { ... } }
        if type(impl) == "function" then
            return impl(...)
        end
        return nil
    end
    local frame = _apply({ [name] = wrapper })
    _stack[#_stack + 1] = frame
    return handle
end

-- ─── Export alc_mock to global ──────────────────────────────────────────────
alc_mock = M

return M
