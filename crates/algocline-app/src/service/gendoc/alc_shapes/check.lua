--- alc_shapes.check — validator (check / assert / assert_dev / is_dev_mode).
---
--- API:
---   check(value, schema, opts?)                -> ok:boolean, reason:string?
---   assert(value, schema_or_name, hint, opts?) -> value (or throws)
---   assert_dev(value, ..., hint, opts?)        -> value (no-op pass if dev off)
---   is_dev_mode()                              -> boolean
---
--- opts (optional, plain table; closures NG — Schema-as-Data 主義):
---   { registry = { name = schema, ... } }
---     `T.ref(name)` and string-name `assert` look schemas up in this
---     plain `{name → schema}` table. When omitted, the alc_shapes
---     module itself is used as the default registry (it is the
---     plain-data dictionary built during Schema-as-Data unification).
---     Multiple registries are merged at the call site (data merge),
---     not by passing closures here.
---
--- Paths use JSONPath-ish form with 1-based array indices:
---   $.field, $[1], $.stages[2].name
---
--- Error message:
---   shape violation at <path>: <detail> (ctx: <hint>)

local M = {}

-- Default registry: the alc_shapes module itself. Resolved lazily to
-- avoid circular load on init. Returned as plain data; we never invoke
-- a closure to look up a name.
local function default_registry()
    return require("alc_shapes")
end

-- Primitive type test (separate from schema kind).
local function lua_type_of(v) return type(v) end

-- Forward decls.
local check_node

local handlers = {}

handlers.any = function(_value, _schema, _path)
    return true
end

handlers.prim = function(value, schema, path)
    local expected = schema.prim
    local got = lua_type_of(value)
    if got ~= expected then
        return false, string.format(
            "shape violation at %s: expected %s, got %s",
            path, expected, got)
    end
    return true
end

handlers.optional = function(value, schema, path, ctx)
    if value == nil then return true end
    return check_node(value, schema.inner, path, ctx)
end

handlers.described = function(value, schema, path, ctx)
    return check_node(value, schema.inner, path, ctx)
end

handlers.array_of = function(value, schema, path, ctx)
    if type(value) ~= "table" then
        return false, string.format(
            "shape violation at %s: expected table (array), got %s",
            path, type(value))
    end
    -- iterate 1-based dense indices
    for i = 1, #value do
        local item = value[i]
        local sub_path = path .. "[" .. i .. "]"
        local ok, reason = check_node(item, schema.elem, sub_path, ctx)
        if not ok then return false, reason end
    end
    return true
end

handlers.shape = function(value, schema, path, ctx)
    if type(value) ~= "table" then
        return false, string.format(
            "shape violation at %s: expected table, got %s",
            path, type(value))
    end
    local fields = schema.fields
    -- Determinism: sort field names so first-fail reports the same
    -- violating field across runs. Lua `pairs` order is unspecified;
    -- tableshape / Zod / Joi all leave this to implementation but we
    -- require reproducibility for CI + conformance tests.
    local names = {}
    for name in pairs(fields) do names[#names + 1] = name end
    table.sort(names)
    for i = 1, #names do
        local name = names[i]
        local sub_schema = fields[name]
        local sub_path = (path == "$") and ("$." .. name) or (path .. "." .. name)
        local sub_val = value[name]
        local ok, reason = check_node(sub_val, sub_schema, sub_path, ctx)
        if not ok then return false, reason end
    end
    -- strict mode: reject extra keys when open=false. Also sorted
    -- for deterministic error reporting (Q1).
    if schema.open == false then
        local extra = {}
        for name in pairs(value) do
            if type(name) == "string" and fields[name] == nil then
                extra[#extra + 1] = name
            end
        end
        table.sort(extra)
        if extra[1] ~= nil then
            local name = extra[1]
            local sub_path = (path == "$") and ("$." .. name) or (path .. "." .. name)
            return false, string.format(
                "shape violation at %s: unexpected field", sub_path)
        end
    end
    return true
end

handlers.discriminated = function(value, schema, path, ctx)
    if type(value) ~= "table" then
        return false, string.format(
            "shape violation at %s: expected table, got %s",
            path, type(value))
    end
    local tag = schema.tag
    local tag_val = value[tag]
    if tag_val == nil then
        return false, string.format(
            "shape violation at %s: missing discriminant field '%s'",
            path, tag)
    end
    local variant = schema.variants[tag_val]
    if variant == nil then
        local keys = {}
        for k in pairs(schema.variants) do keys[#keys + 1] = k end
        table.sort(keys)
        local parts = {}
        for i = 1, #keys do parts[i] = string.format("%q", keys[i]) end
        return false, string.format(
            "shape violation at %s: discriminant '%s' = %q not in [%s]",
            path, tag, tostring(tag_val), table.concat(parts, ", "))
    end
    return handlers.shape(value, variant, path, ctx)
end

handlers.map_of = function(value, schema, path, ctx)
    if type(value) ~= "table" then
        return false, string.format(
            "shape violation at %s: expected table (map), got %s",
            path, type(value))
    end
    for k, v in pairs(value) do
        local key_path = path .. "[key=" .. tostring(k) .. "]"
        local ok, reason = check_node(k, schema.key, key_path, ctx)
        if not ok then return false, reason end
        local val_path = path .. "[" .. tostring(k) .. "]"
        ok, reason = check_node(v, schema.val, val_path, ctx)
        if not ok then return false, reason end
    end
    return true
end

-- ref handler: registry lookup is `rawget(registry_data, name)`. The
-- registry is a plain `{name → schema}` table passed via ctx. No
-- closure resolver — Schema-as-Data 主義に従う。
handlers.ref = function(value, schema, path, ctx)
    local name = schema.name
    local registry = ctx.registry
    local resolved = rawget(registry, name)
    if resolved == nil or type(resolved) ~= "table"
            or rawget(resolved, "kind") == nil then
        return false, string.format(
            "shape violation at %s: unresolved ref '%s'", path, name)
    end
    return check_node(value, resolved, path, ctx)
end

handlers.one_of = function(value, schema, path, _ctx)
    local vs = schema.values
    for i = 1, #vs do
        if value == vs[i] then return true end
    end
    local parts = {}
    for i = 1, #vs do
        local v = vs[i]
        if type(v) == "string" then
            parts[i] = string.format("%q", v)
        else
            parts[i] = tostring(v)
        end
    end
    return false, string.format(
        "shape violation at %s: expected one of [%s], got %s",
        path, table.concat(parts, ", "), tostring(value))
end

-- EE8 cycle guard: a recursive schema (e.g. linked-list `next = T.ref("listnode")`)
-- combined with a self-referencing value (`node.next = node`) would otherwise
-- recurse forever. A pure schema self-loop (`A = T.ref("A")`) in the registry
-- would loop even on primitive values via the ref handler. A depth cap catches
-- both paths at a single site (all recursing kinds — shape/array_of/ref/
-- optional/described/map_of/discriminated — flow through check_node here).
-- 256 is far above any realistic schema depth in this codebase (SoT Entity
-- shapes top out at 3 levels) while still catching pathological cycles.
local MAX_CHECK_DEPTH = 256

check_node = function(value, schema, path, ctx)
    if schema == nil then return true end
    ctx.depth = (ctx.depth or 0) + 1
    if ctx.depth > MAX_CHECK_DEPTH then
        error(string.format(
            "alc_shapes.check: recursion depth exceeded at %s " ..
            "(> %d; cycle in schema or value?)",
            path, MAX_CHECK_DEPTH), 2)
    end
    local kind = rawget(schema, "kind")
    if kind == nil then
        error("alc_shapes.check: schema missing 'kind' field", 2)
    end
    local h = handlers[kind]
    if h == nil then
        error("alc_shapes.check: unknown kind '" .. tostring(kind) .. "'", 2)
    end
    local ok, reason = h(value, schema, path, ctx)
    ctx.depth = ctx.depth - 1
    return ok, reason
end

-- Build the validation ctx from caller-supplied opts. ctx.registry is
-- always a plain `{name → schema}` table; we never store a closure here.
local function build_ctx(opts)
    if opts == nil then
        return { registry = default_registry() }
    end
    if type(opts) ~= "table" then
        error("alc_shapes.check: opts must be a table or nil (got "
            .. type(opts) .. ")", 3)
    end
    local registry = opts.registry
    if registry == nil then
        registry = default_registry()
    elseif type(registry) ~= "table" then
        error("alc_shapes.check: opts.registry must be a plain table "
            .. "of {name = schema} (closures NG); got " .. type(registry), 3)
    end
    return { registry = registry }
end

--- Return (ok, reason). Never throws for normal schema violations.
--- opts.registry: plain {name → schema} table (default: alc_shapes module).
function M.check(value, schema, opts)
    if schema == nil then return true end
    local ctx = build_ctx(opts)
    return check_node(value, schema, "$", ctx)
end

local function compose_msg(reason, ctx_hint)
    if ctx_hint == nil or ctx_hint == "" then return reason end
    return reason .. " (ctx: " .. tostring(ctx_hint) .. ")"
end

--- Assert schema; returns value on pass, throws on fail.
--- Overloads on schema_or_name:
---   nil          -> loud-fail (intent violation; use S.check for silent pass)
---   "any"        -> no-op pass
---   other string -> lookup in registry (default: alc_shapes module).
---                   Loud-fail if unknown.
---   table        -> direct schema
--- opts (optional, last position):
---   { registry = { name = schema, ... } }
---
--- EE7: `assert` means "I intended to validate". Passing `nil` is a
--- typo signal (e.g. `S.assert(r, S.votd, ...)` where `S.votd` resolves
--- to nil) and must loud-fail. Use `S.check(v, nil)` if silent pass is
--- intentional (projection-side guard pattern).
function M.assert(value, schema_or_name, ctx_hint, opts)
    local schema
    if schema_or_name == nil then
        error("alc_shapes.assert: schema_or_name must not be nil. "
            .. "Use S.check(v, nil) for silent pass, or pass a schema / "
            .. "name / \"any\" explicitly.", 2)
    elseif type(schema_or_name) == "string" then
        if schema_or_name == "any" then return value end
        local registry = (opts and opts.registry) or default_registry()
        if type(registry) ~= "table" then
            error("alc_shapes.assert: opts.registry must be a plain table "
                .. "of {name = schema} (closures NG); got " .. type(registry), 2)
        end
        schema = rawget(registry, schema_or_name)
        if schema == nil or type(schema) ~= "table" or rawget(schema, "kind") == nil then
            error("alc_shapes.assert: unknown shape name '" .. schema_or_name .. "'", 2)
        end
    elseif type(schema_or_name) == "table" then
        schema = schema_or_name
    else
        error("alc_shapes.assert: schema_or_name must be nil, string, or table (got "
            .. type(schema_or_name) .. ")", 2)
    end
    local ok, reason = M.check(value, schema, opts)
    if not ok then
        error(compose_msg(reason, ctx_hint), 2)
    end
    return value
end

--- Dev-mode-only assert: no-op pass when ALC_SHAPE_CHECK != "1".
function M.assert_dev(value, schema_or_name, ctx_hint, opts)
    if not M.is_dev_mode() then return value end
    return M.assert(value, schema_or_name, ctx_hint, opts)
end

function M.is_dev_mode()
    return os.getenv("ALC_SHAPE_CHECK") == "1"
end

M._internal = {
    handlers          = handlers,
    check_node        = check_node,
    compose_msg       = compose_msg,
    build_ctx         = build_ctx,
    default_registry  = default_registry,
}

return M
