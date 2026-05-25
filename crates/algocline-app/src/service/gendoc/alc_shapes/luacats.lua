--- alc_shapes.luacats — LuaCATS codegen for shape registries.
---
--- class_for(class_name, shape_schema)       -> LuaCATS class text
--- gen(shapes_table, class_prefix?)          -> full d.lua contents

local reflect = require("alc_shapes.reflect")

local M = {}

local function pascal_case(name)
    -- "voted" -> "Voted"; "safe_panel" -> "SafePanel".
    local out = {}
    local start = 1
    while true do
        local us = name:find("[_-]", start)
        local token
        if us then
            token = name:sub(start, us - 1)
            start = us + 1
        else
            token = name:sub(start)
        end
        if #token > 0 then
            out[#out + 1] = token:sub(1, 1):upper() .. token:sub(2)
        end
        if not us then
            break
        end
    end
    return table.concat(out)
end

--- Forward declaration: inline_shape_type and type_of are mutually recursive
--- (array_of(shape(...)) threads array -> shape -> inline expansion -> type_of
--- on each field).
---@type fun(node: any, class_prefix: string?): string
local type_of

--- Detect a top-level `|` in a rendered LuaCATS type string, ignoring
--- pipes that sit inside `{ ... }` or `( ... )` groups.
--- Used to decide whether `array_of(union)` needs parens (`(A|B)[]`)
--- vs. a non-union single type (`T[]`). `T.one_of({"a"})` → `"a"` needs
--- no parens, but `T.one_of({"a","b"})` → `"a"|"b"` does.
local function has_top_level_pipe(s)
    local depth = 0
    for i = 1, #s do
        local c = s:sub(i, i)
        if c == "{" or c == "(" then
            depth = depth + 1
        elseif c == "}" or c == ")" then
            depth = depth - 1
        elseif c == "|" and depth == 0 then
            return true
        end
    end
    return false
end

--- Render a nested shape as an inline LuaLS table type literal, e.g.
---     { answer?: string, reasoning: string }
--- Field order is alphabetical (via reflect.fields). Empty shapes render
--- as bare `table` since `{ }` has no meaningful LuaLS interpretation.
---
--- Top-level shapes are NOT rendered this way — `class_for`/`gen` emit
--- them as `---@class` blocks. This helper is invoked only from the
--- `kind == "shape"` branch of `type_of`, which fires only for nested
--- shapes reached via `array_of` / shape-field / map_of value etc.
local function inline_shape_type(node, class_prefix)
    local entries = reflect.fields(node)
    if #entries == 0 then
        return "table"
    end
    local parts = {}
    for i = 1, #entries do
        local e = entries[i]
        local suffix = e.optional and "?" or ""
        parts[i] = string.format("%s%s: %s", e.name, suffix, type_of(e.type, class_prefix))
    end
    return "{ " .. table.concat(parts, ", ") .. " }"
end

--- Map a schema node to its LuaCATS type string.
--- `class_prefix` is threaded through so `ref(name)` resolves to the
--- correct class identifier. Nested shapes are inline-expanded as
--- `{ field: type, ... }` via `inline_shape_type`; `discriminated` is
--- still collapsed to `table` (union codegen deferred).
type_of = function(node, class_prefix)
    class_prefix = class_prefix or "AlcResult"
    local kind = rawget(node, "kind")
    if kind == "prim" then
        return node.prim
    elseif kind == "any" then
        return "any"
    elseif kind == "optional" then
        return type_of(rawget(node, "inner"), class_prefix)
    elseif kind == "described" then
        return type_of(rawget(node, "inner"), class_prefix)
    elseif kind == "array_of" then
        -- `described` wrappers are transparent (doc-only); peel them.
        --
        -- NOTE: `array_of(optional(T))` is rejected at DSL construction
        -- (see t.lua M.array_of C1 guard). The `(T|nil)[]` branch below
        -- is therefore unreachable via the DSL, but we keep it for
        -- Schema-as-Data persistence round-trips: a plain-data schema
        -- reloaded from JSON bypasses the DSL constructor, and the
        -- codegen must still render something reasonable instead of
        -- crashing. Production shapes in this codebase do not use it.
        local elem = rawget(node, "elem")
        local had_optional = false
        while true do
            local k = rawget(elem, "kind")
            if k == "optional" then
                had_optional = true
                elem = rawget(elem, "inner")
            elseif k == "described" then
                elem = rawget(elem, "inner")
            else
                break
            end
        end
        local inner_type = type_of(elem, class_prefix)
        if had_optional then
            return "(" .. inner_type .. "|nil)[]"
        end
        -- C2: when the element resolves to a top-level union (e.g.
        -- `discriminated` → `{...}|{...}`, or `one_of` with >1 value →
        -- `"a"|"b"`), LuaLS parses `A|B[]` as `A|(B[])`. Parenthesize
        -- to preserve the intended `(A|B)[]` precedence.
        if has_top_level_pipe(inner_type) then
            return "(" .. inner_type .. ")[]"
        end
        return inner_type .. "[]"
    elseif kind == "discriminated" then
        -- C2: render as a union of inline shape literals, one per variant,
        -- sorted alphabetically by variant key for diff stability. Each
        -- variant's `tag` field (enforced present at construction by C4)
        -- typically carries a `T.one_of({"<key>"})` and inline_shape_type
        -- renders it as a string literal, preserving the discriminant
        -- information in the emitted type. Callers then get per-branch
        -- field autocomplete based on the current value of `name`.
        local variants = rawget(node, "variants")
        local keys = {}
        for k in pairs(variants) do
            keys[#keys + 1] = k
        end
        table.sort(keys)
        local parts = {}
        for i = 1, #keys do
            parts[i] = inline_shape_type(variants[keys[i]], class_prefix)
        end
        return table.concat(parts, "|")
    elseif kind == "map_of" then
        local k = type_of(rawget(node, "key"), class_prefix)
        local v = type_of(rawget(node, "val"), class_prefix)
        return "table<" .. k .. ", " .. v .. ">"
    elseif kind == "shape" then
        -- Nested shape: inline-expand to a LuaLS table type literal so
        -- IDE completion / typo detection survive for walkable fields
        -- like `voted.paths[i].answer`.
        return inline_shape_type(node, class_prefix)
    elseif kind == "ref" then
        -- ref renders as the target named class (PascalCase of the ref name).
        return class_prefix .. pascal_case(rawget(node, "name"))
    elseif kind == "one_of" then
        local vs = rawget(node, "values")
        local parts = {}
        for i = 1, #vs do
            local v = vs[i]
            if type(v) == "string" then
                parts[i] = string.format("%q", v)
            else
                parts[i] = tostring(v)
            end
        end
        return table.concat(parts, "|")
    else
        error("alc_shapes.luacats: unknown kind '" .. tostring(kind) .. "'", 2)
    end
end

--- Render a single class block for a shape schema.
--- `class_prefix` is used to resolve `ref(name)` occurrences inside the
--- schema; defaults to "AlcResult" to match `gen()`.
function M.class_for(class_name, schema, class_prefix)
    if type(class_name) ~= "string" or class_name == "" then
        error("alc_shapes.luacats.class_for: class_name must be non-empty string", 2)
    end
    if type(schema) ~= "table" or rawget(schema, "kind") ~= "shape" then
        error("alc_shapes.luacats.class_for: schema must be kind='shape'", 2)
    end
    class_prefix = class_prefix or "AlcResult"
    local lines = { "---@class " .. class_name }
    local entries = reflect.fields(schema)
    for i = 1, #entries do
        local e = entries[i]
        local suffix = e.optional and "?" or ""
        local ty = type_of(e.type, class_prefix)
        local line = string.format("---@field %s%s %s", e.name, suffix, ty)
        if e.doc then
            line = line .. " @" .. e.doc
        end
        lines[#lines + 1] = line
    end
    return table.concat(lines, "\n") .. "\n"
end

--- Render an `---@alias` line for a single ref schema.
---
--- Used by `gen_pkgs` to project per-pkg `T.ref(name)` shapes into a
--- stable `AlcPkgInput_<pkg>` / `AlcPkgResult_<pkg>` alias surface so
--- pkg consumers see consistent class names regardless of whether the
--- pkg's run.input/result is an inline shape or a ref into the
--- alc_shapes registry.
---
--- `class_prefix` controls the resolution of the ref target name and
--- defaults to "AlcResult" to match `gen()` / `class_for()`.
function M.alias_for(class_name, ref_schema, class_prefix)
    if type(class_name) ~= "string" or class_name == "" then
        error("alc_shapes.luacats.alias_for: class_name must be non-empty string", 2)
    end
    if type(ref_schema) ~= "table" or rawget(ref_schema, "kind") ~= "ref" then
        error("alc_shapes.luacats.alias_for: schema must be kind='ref'", 2)
    end
    class_prefix = class_prefix or "AlcResult"
    local target = class_prefix .. pascal_case(rawget(ref_schema, "name"))
    return "---@alias " .. class_name .. " " .. target .. "\n"
end

--- Generate the full contents of types/alc_pkgs.d.lua from a list of
--- per-pkg shape specs.
---
--- `pkg_specs` must be an array of `{ name = <pkg_name>, input = <schema?>, result = <schema?> }`
--- entries (matching the `PkgInfo.shape` format produced by
--- `tools.docs.extract.build_pkg_info`).
---
--- For each spec, produces (in pkg name asc order):
---   * `AlcPkgInput_<pkg>`  ←  spec.input  (omitted when nil)
---   * `AlcPkgResult_<pkg>` ←  spec.result (omitted when nil)
---
--- Inline shape schemas (kind="shape") become `---@class` blocks via
--- `class_for`. Ref schemas (kind="ref") become `---@alias` lines via
--- `alias_for`. Other top-level kinds are silently skipped (would
--- otherwise require an artificial wrapper class — out of scope here).
---
--- Output always ends with a newline. Returns just `"---@meta\n"` when
--- no spec carries any shape (caller can still write the file as a
--- placeholder so downstream `workspace.library` references resolve).
function M.gen_pkgs(pkg_specs)
    if pkg_specs ~= nil and type(pkg_specs) ~= "table" then
        error("alc_shapes.luacats.gen_pkgs: pkg_specs must be table or nil", 2)
    end

    -- Sort by pkg name for deterministic output (drift-check friendly).
    local sorted = {}
    if pkg_specs ~= nil then
        for i = 1, #pkg_specs do
            sorted[i] = pkg_specs[i]
        end
        table.sort(sorted, function(a, b)
            return tostring(a.name) < tostring(b.name)
        end)
    end

    local out = { "---@meta" }

    local function emit(class_name, schema)
        if type(schema) ~= "table" then
            return
        end
        local kind = rawget(schema, "kind")
        if kind == "shape" then
            out[#out + 1] = ""
            out[#out + 1] = M.class_for(class_name, schema, "AlcResult"):gsub("\n$", "")
        elseif kind == "ref" then
            out[#out + 1] = ""
            out[#out + 1] = M.alias_for(class_name, schema, "AlcResult"):gsub("\n$", "")
        end
        -- Other kinds (string/number/list/etc. as a top-level pkg shape) are
        -- structurally unusual and skipped silently — pkgs should wrap them
        -- in a shape if they want a typed surface.
    end

    for _, spec in ipairs(sorted) do
        local pkg_name = spec.name
        if type(pkg_name) == "string" and pkg_name ~= "" then
            emit("AlcPkgInput_" .. pkg_name, spec.input)
            emit("AlcPkgResult_" .. pkg_name, spec.result)
        end
    end

    return table.concat(out, "\n") .. "\n"
end

--- Generate the full contents of types/alc_shapes.d.lua from a
--- table of `{ [name] = shape_schema, ... }`. Output always ends
--- with a newline (drift check compatibility).
function M.gen(shapes_table, class_prefix)
    if shapes_table ~= nil and type(shapes_table) ~= "table" then
        error("alc_shapes.luacats.gen: shapes_table must be table or nil", 2)
    end
    class_prefix = class_prefix or "AlcResult"

    local names = {}
    if shapes_table ~= nil then
        for name, schema in pairs(shapes_table) do
            if type(schema) == "table" and rawget(schema, "kind") == "shape" then
                names[#names + 1] = name
            end
        end
    end
    table.sort(names)

    local out = { "---@meta" }
    if shapes_table ~= nil then
        for _, name in ipairs(names) do
            local class_name = class_prefix .. pascal_case(name)
            out[#out + 1] = ""
            out[#out + 1] = M.class_for(class_name, shapes_table[name], class_prefix):gsub(
                "\n$",
                ""
            )
        end
    end

    return table.concat(out, "\n") .. "\n"
end

M._internal = {
    pascal_case = pascal_case,
    type_of = type_of,
    inline_shape_type = inline_shape_type,
}

return M
