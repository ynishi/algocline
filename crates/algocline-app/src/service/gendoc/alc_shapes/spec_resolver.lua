--- alc_shapes.spec_resolver — normalize pkg I/O contract for routing/recipe layers.
---
--- Consumers (routing / recipe / Workflow composer) call into pkgs without
--- caring whether the pkg is a typed bundled pkg (declares `M.spec`) or an
--- opaque external pkg (omits it). Both forms resolve to the same
--- `ResolvedSpec` shape; opaque pkgs simply have empty `entries`.
---
--- Public API:
---   resolve(pkg)                   -> ResolvedSpec
---   run(pkg, ctx, entry_name?)     -> result  (auto-assert_dev when typed)
---   is_passthrough(pkg, shape_name) -> boolean
---
--- ResolvedSpec shape (plain data; Schema-as-Data doctrine):
---   {
---     kind    = "typed" | "opaque",
---     origin  = "spec"  | "none",
---     entries = {
---       [name] = {
---         input  = <schema|nil>,                 -- ctx-threading mode
---         result = <schema|nil>,
---         args   = <array<schema|nil>|nil>,      -- direct-args mode (Pure)
---       }, ...
---     },
---     compose = <spec.compose> | nil,
---     exports = <spec.exports> | nil,
---   }
---
--- Short-form string `spec.entries.*.input|result` is coerced to
--- `T.ref(name)` so downstream consumers always see a kind-tagged schema
--- (matches `tools.docs.extract.build_pkg_info` behaviour).
---
--- ## Direct-args mode (`spec.entries.{e}.args`)
---
--- Library-style pkgs (pure functions `fn(a, b) -> scalar`, e.g. bft /
--- kemeny / scoring_rule) declare positional shapes via `args` instead
--- of `input`. Each element is a shape (string or schema). `args` and
--- `input` are mutually exclusive per entry — the resolver raises on
--- both being set. `instrument` inspects `entry.args` to decide between
--- ctx-threading wrapping and per-argument wrapping.

local T     = require("alc_shapes.t")
local check = require("alc_shapes.check")

local M = {}

local function is_schema(v)
    return type(v) == "table" and rawget(v, "kind") ~= nil
end

local function coerce_shape_ref(v)
    if v == nil then return nil end
    if type(v) == "string" then
        return T.ref(v)
    end
    if is_schema(v) then
        return v
    end
    error(
        "alc_shapes.spec_resolver: shape field must be string or schema, got "
            .. type(v), 2)
end

-- Coerce a positional-args shape list. Each slot is either a shape
-- (string or schema) or `nil` (skip validation at that position).
-- Returns `nil` when the whole list is absent.
local function coerce_args_list(v, entry_name)
    if v == nil then return nil end
    if type(v) ~= "table" then
        error(string.format(
            "alc_shapes.spec_resolver: spec.entries.%s.args must be an array "
                .. "of shapes (got %s)",
            tostring(entry_name), type(v)), 2)
    end
    local n = #v
    local out = {}
    for i = 1, n do
        local slot = v[i]
        if slot == nil then
            out[i] = nil
        elseif type(slot) == "string" then
            out[i] = T.ref(slot)
        elseif is_schema(slot) then
            out[i] = slot
        else
            error(string.format(
                "alc_shapes.spec_resolver: spec.entries.%s.args[%d] must be "
                    .. "string / schema / nil (got %s)",
                tostring(entry_name), i, type(slot)), 2)
        end
    end
    return out
end

function M.resolve(pkg)
    if type(pkg) ~= "table" then
        error("alc_shapes.spec_resolver.resolve: pkg must be a table", 2)
    end

    local spec = rawget(pkg, "spec")
    if type(spec) == "table" and type(spec.entries) == "table" then
        local entries = {}
        for name, entry in pairs(spec.entries) do
            if type(entry) ~= "table" then
                error(string.format(
                    "alc_shapes.spec_resolver: spec.entries.%s must be a table",
                    tostring(name)), 2)
            end
            if entry.input ~= nil and entry.args ~= nil then
                error(string.format(
                    "alc_shapes.spec_resolver: spec.entries.%s declares both "
                        .. "`input` (ctx-threading) and `args` (direct-args); "
                        .. "these modes are mutually exclusive",
                    tostring(name)), 2)
            end
            entries[name] = {
                input  = coerce_shape_ref(entry.input),
                result = coerce_shape_ref(entry.result),
                args   = coerce_args_list(entry.args, name),
            }
        end
        return {
            kind    = "typed",
            origin  = "spec",
            entries = entries,
            compose = spec.compose,
            exports = spec.exports,
        }
    end

    return {
        kind    = "opaque",
        origin  = "none",
        entries = {},
        compose = nil,
        exports = nil,
    }
end

function M.run(pkg, ctx, entry_name)
    entry_name = entry_name or "run"
    local fn = rawget(pkg, entry_name)
    if type(fn) ~= "function" then
        error(string.format(
            "alc_shapes.spec_resolver.run: pkg has no function '%s'",
            entry_name), 2)
    end

    local resolved = M.resolve(pkg)
    local pkg_name = (type(pkg.meta) == "table" and pkg.meta.name) or "<anon>"
    local ctx_hint = pkg_name .. "." .. entry_name

    local entry = resolved.entries[entry_name]

    if entry and entry.input then
        check.assert_dev(ctx, entry.input, ctx_hint .. ":input")
    end

    local returned = fn(ctx)

    -- AlcCtx 規約: pkg は ctx を返し、実際の result 形は ctx.result に入る。
    -- 最小限の柔軟性として、returned が table 以外、または .result が nil
    -- なら returned そのものを検査対象にする (shape を直接返す external pkg
    -- を許容するフォールバック)。bundled 9 pkg はすべて前者。
    if entry and entry.result then
        local actual
        if type(returned) == "table" and returned.result ~= nil then
            actual = returned.result
        else
            actual = returned
        end
        check.assert_dev(actual, entry.result, ctx_hint .. ":result")
    end

    return returned
end

function M.is_passthrough(pkg, shape_name)
    local resolved = M.resolve(pkg)
    if resolved.kind ~= "typed" then return false end
    if not resolved.compose then return false end
    local pt = resolved.compose.passthrough
    if pt == nil then return false end
    if type(pt) == "string" then
        return pt == shape_name
    end
    if type(pt) == "table" then
        for _, n in ipairs(pt) do
            if n == shape_name then return true end
        end
    end
    return false
end

return M
