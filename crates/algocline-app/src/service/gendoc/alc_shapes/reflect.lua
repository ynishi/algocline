--- alc_shapes.reflect — reflection over shape schemas.
---
--- fields(schema) -> { { name, type, optional, doc? }, ... } sorted by name.
--- walk(schema, visitor) -> DFS visit of all nested schemas.
---
--- Uses rawget exclusively so it works on plain tables even without the
--- combinator metatable attached.

local M = {}

local function unwrap(schema)
    local optional = false
    local doc = nil
    while true do
        local kind = rawget(schema, "kind")
        if kind == "optional" then
            optional = true
            schema = rawget(schema, "inner")
        elseif kind == "described" then
            if doc == nil then doc = rawget(schema, "doc") end
            schema = rawget(schema, "inner")
        else
            break
        end
    end
    return schema, optional, doc
end

--- Enumerate direct fields of a shape schema.
--- @param schema table shape-kind schema
--- @return table[] list of { name, type, optional, doc? } records, sorted by name
function M.fields(schema)
    if type(schema) ~= "table" then
        error("alc_shapes.reflect.fields: expected table, got " .. type(schema), 2)
    end
    if rawget(schema, "kind") ~= "shape" then
        error("alc_shapes.reflect.fields: schema must be kind='shape'", 2)
    end
    local fields_tbl = rawget(schema, "fields")
    if type(fields_tbl) ~= "table" then
        error("alc_shapes.reflect.fields: schema.fields is not a table", 2)
    end

    local names = {}
    for name in pairs(fields_tbl) do
        names[#names + 1] = name
    end
    table.sort(names)

    local out = {}
    for i = 1, #names do
        local name = names[i]
        local inner, optional, doc = unwrap(fields_tbl[name])
        local entry = { name = name, type = inner, optional = optional }
        if doc ~= nil then entry.doc = doc end
        out[i] = entry
    end
    return out
end

--- DFS-walk every schema node in the tree, calling visitor(node).
--- Visits the root first, then descends into children.
function M.walk(schema, visitor)
    if type(schema) ~= "table" then
        error("alc_shapes.reflect.walk: expected table schema, got " .. type(schema), 2)
    end
    if type(visitor) ~= "function" then
        error("alc_shapes.reflect.walk: visitor must be a function", 2)
    end
    local function visit(node)
        visitor(node)
        local kind = rawget(node, "kind")
        if kind == "shape" then
            local fields_tbl = rawget(node, "fields")
            local names = {}
            for name in pairs(fields_tbl) do names[#names + 1] = name end
            table.sort(names)
            for i = 1, #names do visit(fields_tbl[names[i]]) end
        elseif kind == "array_of" then
            visit(rawget(node, "elem"))
        elseif kind == "discriminated" then
            local variants = rawget(node, "variants")
            local keys = {}
            for k in pairs(variants) do keys[#keys + 1] = k end
            table.sort(keys)
            for i = 1, #keys do visit(variants[keys[i]]) end
        elseif kind == "map_of" then
            visit(rawget(node, "key"))
            visit(rawget(node, "val"))
        elseif kind == "optional" or kind == "described" then
            visit(rawget(node, "inner"))
        end
        -- `ref` / `prim` / `any` / `one_of` are leaves: no descent.
    end
    visit(schema)
end

M._internal = { unwrap = unwrap }

return M
