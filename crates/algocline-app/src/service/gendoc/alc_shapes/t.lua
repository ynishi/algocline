--- alc_shapes.t — DSL combinators and schema internal structure.
---
--- Schema-as-Data contract (after Malli TypeSchemaAsData):
---   * Every schema is a plain Lua table whose state is held in
---     `rawget`-readable fields: `kind`, `prim`, `inner`, `elem`,
---     `values`, `fields`, `open`, `doc`, `name`, `key`, `val`,
---     `variants`, `tag`.
---   * Metatables carry combinator sugar only (`:is_optional()`,
---     `:describe(doc)`). Stripping the metatable must not change
---     validation behaviour — this is what makes schemas persistable.
---   * Combinators return new tables; schemas are never mutated
---     in place.
---
--- See alc_shapes/README.md §Core concept.

local M = {}

local combinators = {}
local schema_mt = { __index = combinators }

local function is_schema(v)
    return type(v) == "table" and rawget(v, "kind") ~= nil
end

function combinators:is_optional()
    return setmetatable({ kind = "optional", inner = self }, schema_mt)
end

function combinators:describe(doc)
    if type(doc) ~= "string" then
        error("alc_shapes.t: describe expects string doc", 2)
    end
    return setmetatable({ kind = "described", inner = self, doc = doc }, schema_mt)
end

M.string  = setmetatable({ kind = "prim", prim = "string" },  schema_mt)
M.number  = setmetatable({ kind = "prim", prim = "number" },  schema_mt)
M.boolean = setmetatable({ kind = "prim", prim = "boolean" }, schema_mt)
M.table   = setmetatable({ kind = "prim", prim = "table" },   schema_mt)
M.any     = setmetatable({ kind = "any" },                    schema_mt)

--- T.shape(fields, opts) — named key set.
---
--- Layer contract:
---   Runtime shapes (ctx.result etc.):  open = true (default)
---   Entity shapes (boundary contract): open = false (explicit)
---
--- The default is deliberately open to preserve the pass-through
--- culture of ctx.result: packages chain outputs through ctx and may
--- attach trace / metrics / debug keys without invalidating downstream
--- consumers. Entity layer (see tools/docs/entity_schemas.lua) opts
--- into strict mode because its fields are the boundary contract
--- itself, and extra keys signal drift.
function M.shape(fields, opts)
    if type(fields) ~= "table" then
        error("alc_shapes.t: shape expects fields table as first argument", 2)
    end
    -- C3: shallow-copy the fields table so a caller who later mutates
    -- the passed table does not silently mutate this schema. Schema-as-Data
    -- doctrine treats schemas as immutable plain data; capturing the caller's
    -- live reference contradicts that invariant. `one_of.values` already
    -- does this (see M.one_of below); shape/discriminated did not.
    local copy = {}
    for name, sub in pairs(fields) do
        if type(name) ~= "string" then
            error("alc_shapes.t: shape field name must be string, got " .. type(name), 2)
        end
        if not is_schema(sub) then
            error(string.format(
                "alc_shapes.t: shape field '%s' must be a schema (table with kind)", name), 2)
        end
        copy[name] = sub
    end
    local open
    if opts == nil then
        open = true
    else
        if type(opts) ~= "table" then
            error("alc_shapes.t: shape expects opts table as second argument", 2)
        end
        if opts.open == nil then
            open = true
        else
            open = opts.open and true or false
        end
    end
    return setmetatable({ kind = "shape", fields = copy, open = open }, schema_mt)
end

function M.array_of(elem)
    if not is_schema(elem) then
        error("alc_shapes.t: array_of expects a schema as argument", 2)
    end
    -- C1 guard: Lua's `#` operator is unspecified on arrays with holes,
    -- so a runtime validator cannot reliably distinguish `{1, nil, 2}`
    -- from `{1}` when iterating `1..#value`. Admitting `nil` at the
    -- element position would make `check` silently under-validate such
    -- arrays while LuaCATS generates `(T|nil)[]` — a doc/runtime gap.
    -- `described` wrappers are transparent (doc-only), so peel them
    -- before the check: `array_of(T:describe(...):is_optional())` and
    -- `array_of(T:is_optional():describe(...))` are both rejected.
    local probe = elem
    while rawget(probe, "kind") == "described" do
        probe = rawget(probe, "inner")
    end
    if rawget(probe, "kind") == "optional" then
        error(
            "alc_shapes.t: array_of(optional(T)) is not allowed — " ..
            "Lua's `#` cannot reliably validate arrays with nil holes. " ..
            "Use array_of(T) (require dense) or model the nil-admission " ..
            "at the enclosing field (e.g. T.array_of(T):is_optional()).",
            2)
    end
    return setmetatable({ kind = "array_of", elem = elem }, schema_mt)
end

function M.one_of(values)
    if type(values) ~= "table" then
        error("alc_shapes.t: one_of expects a values table as argument", 2)
    end
    local n = 0
    for _ in pairs(values) do n = n + 1 end
    if n == 0 then
        error("alc_shapes.t: one_of expects at least one value", 2)
    end
    for i = 1, n do
        local v = values[i]
        if v == nil then
            error("alc_shapes.t: one_of expects a 1-based dense array of values", 2)
        end
        local t = type(v)
        if t ~= "string" and t ~= "number" and t ~= "boolean" then
            error(string.format(
                "alc_shapes.t: one_of values must be string/number/boolean, got %s at index %d",
                t, i), 2)
        end
    end
    -- C5: reject duplicate literals. `T.one_of({"a", "a"})` is almost
    -- certainly a typo (copy-paste / merge glitch), and duplicate values
    -- produce a redundant expected-list in error messages. Lua has no
    -- native set; use a small hash keyed by `type..value` so string "1"
    -- and number 1 are distinguished.
    local seen = {}
    local copy = {}
    for i = 1, n do
        local v = values[i]
        local key = type(v) .. ":" .. tostring(v)
        if seen[key] then
            error(string.format(
                "alc_shapes.t: one_of has duplicate value %s at index %d",
                (type(v) == "string") and string.format("%q", v) or tostring(v),
                i), 2)
        end
        seen[key] = true
        copy[i] = v
    end
    return setmetatable({ kind = "one_of", values = copy }, schema_mt)
end

function M.discriminated(tag, variants)
    if type(tag) ~= "string" or tag == "" then
        error("alc_shapes.t: discriminated expects non-empty string tag", 2)
    end
    if type(variants) ~= "table" then
        error("alc_shapes.t: discriminated expects variants table", 2)
    end
    -- C3: shallow-copy for the same reason as M.shape (immutability of
    -- constructed schemas).
    -- C4: enforce that each variant declares the discriminant tag as
    -- one of its own fields. The validator (handlers.discriminated)
    -- dispatches by variant key but then re-validates the variant shape
    -- itself, which only catches the tag value mismatch if the variant
    -- shape constrains it. In practice every production variant uses
    -- `name = T.one_of({"X"})` as belt-and-suspenders; DSL-formalize that
    -- convention so typos ("forgot to add the tag field") fail loud at
    -- construction time rather than silently pass through.
    local copy = {}
    local count = 0
    for k, v in pairs(variants) do
        if type(k) ~= "string" then
            error("alc_shapes.t: discriminated variant key must be string, got " .. type(k), 2)
        end
        if not is_schema(v) or rawget(v, "kind") ~= "shape" then
            error(string.format(
                "alc_shapes.t: discriminated variant '%s' must be a shape schema", k), 2)
        end
        if rawget(rawget(v, "fields"), tag) == nil then
            error(string.format(
                "alc_shapes.t: discriminated variant '%s' must declare the tag field '%s'",
                k, tag), 2)
        end
        copy[k] = v
        count = count + 1
    end
    if count == 0 then
        error("alc_shapes.t: discriminated expects at least one variant", 2)
    end
    return setmetatable({ kind = "discriminated", tag = tag, variants = copy }, schema_mt)
end

function M.ref(name)
    if type(name) ~= "string" or name == "" then
        error("alc_shapes.t: ref expects non-empty string name", 2)
    end
    return setmetatable({ kind = "ref", name = name }, schema_mt)
end

function M.map_of(key, val)
    if not is_schema(key) then
        error("alc_shapes.t: map_of expects a schema as key argument", 2)
    end
    if not is_schema(val) then
        error("alc_shapes.t: map_of expects a schema as val argument", 2)
    end
    return setmetatable({ kind = "map_of", key = key, val = val }, schema_mt)
end

M._internal = {
    schema_mt   = schema_mt,
    combinators = combinators,
    is_schema   = is_schema,
}

return M
