--- tools.docs.projections — PkgInfo → derived artifacts.
---
--- Pure functions. Each projection takes a PkgInfo (or a collection)
--- and returns the rendered artifact as a string. No I/O.
---
--- Single-AST doctrine (see alc_shapes/README.md §Core concept):
--- `PkgInfo.shape.input` / `PkgInfo.shape.result` are alc_shapes
--- schemas directly. This module reads them via `rawget` (schemas are
--- persistable plain tables — see alc_shapes/README.md §Persistable)
--- and projects into the target artifact. Field iteration goes
--- through `alc_shapes.fields()` which already returns sorted
--- `{name, type, optional, doc?}` entries with optional/described
--- wrappers peeled.

local S = require("alc_shapes")

local M = {}

-- ── helpers ────────────────────────────────────────────────────────────

local function escape_md_table_cell(s)
    -- Escape pipe so it doesn't break the table row.
    return (s:gsub("|", "\\|"))
end

local function escape_yaml_string(s)
    -- Minimal: escape double-quote and backslash for a "..."-wrapped YAML value.
    local out = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    return out
end

--- Peel optional / described wrappers off a schema.
---
--- Returns: inner_schema, optional, doc
local function peel(schema)
    local optional = false
    local doc = ""
    while true do
        local k = rawget(schema, "kind")
        if k == "optional" then
            optional = true
            schema = rawget(schema, "inner")
        elseif k == "described" then
            if doc == "" then doc = rawget(schema, "doc") or "" end
            schema = rawget(schema, "inner")
        else
            break
        end
    end
    return schema, optional, doc
end

-- ── alc_shapes schema → string projection ─────────────────────────────

--- Pretty-print an alc_shapes schema in the default human-readable form.
---
--- Examples:
---   T.string                        → "string"
---   T.array_of(T.string)            → "array of string"
---   T.map_of(T.string, T.number)    → "map of string to number"
---   T.shape({a=T.string})           → "shape { a: string }"
---   T.one_of({"a","b"})             → 'one_of("a", "b")'
---   T.discriminated("name", {...})  → 'discriminated by "name"'
---   T.ref("paneled")                → "paneled"
---   T.any                           → "any"
---
--- Nested `shape` is expanded inline per pipeline-spec §7.1. Fields
--- come from `S.fields()` (alphabetically sorted, wrappers peeled);
--- optional fields carry a trailing `?` on the field name.
---
--- `optional` / `described` wrappers at the entry are peeled
--- transparently; optional-ness at the entry is NOT expressed here
--- (it's carried by the enclosing field's `?` suffix instead).
function M.shape_type_string(schema)
    local peeled = peel(schema)
    local k = rawget(peeled, "kind")
    if k == "prim" then
        return rawget(peeled, "prim")
    elseif k == "any" then
        return "any"
    elseif k == "array_of" then
        return "array of " .. M.shape_type_string(rawget(peeled, "elem"))
    elseif k == "map_of" then
        return string.format("map of %s to %s",
            M.shape_type_string(rawget(peeled, "key")),
            M.shape_type_string(rawget(peeled, "val")))
    elseif k == "one_of" then
        local values = rawget(peeled, "values")
        local parts = {}
        for i = 1, #values do
            local v = values[i]
            if type(v) == "string" then
                parts[i] = string.format("%q", v)
            else
                parts[i] = tostring(v)
            end
        end
        return "one_of(" .. table.concat(parts, ", ") .. ")"
    elseif k == "shape" then
        local entries = S.fields(peeled)
        if #entries == 0 then return "shape { }" end
        local parts = {}
        for i = 1, #entries do
            local e = entries[i]
            local mark = e.optional and "?" or ""
            parts[i] = string.format(
                "%s%s: %s", e.name, mark, M.shape_type_string(e.type))
        end
        return "shape { " .. table.concat(parts, ", ") .. " }"
    elseif k == "discriminated" then
        return string.format('discriminated by "%s"', rawget(peeled, "tag"))
    elseif k == "ref" then
        return rawget(peeled, "name")
    else
        return "?"
    end
end

--- YAML flow-plain scalars forbid a leading indicator (`-`, `?`, `:`,
--- `,`, `[`, `]`, `{`, `}`, `#`, `&`, `*`, `!`, `|`, `>`, `'`, `"`, `%`,
--- `@`, `` ` ``) and forbid `: ` / ` #` anywhere. Emit a quoted scalar
--- whenever any of these would break the plain form.
local function yaml_scalar(s)
    if s == "" then return '""' end
    local first = s:sub(1, 1)
    local needs_quote =
        first:match("[%-%?:,%[%]{}#&%*!|>'\"%%@`]") ~= nil
        or s:find(": ", 1, true) ~= nil
        or s:find(" #", 1, true) ~= nil
        or s:find("\n", 1, true) ~= nil
    if needs_quote then
        return '"' .. escape_yaml_string(s) .. '"'
    end
    return s
end

-- ── per-pkg narrative.md ───────────────────────────────────────────────

local function build_frontmatter(pkg_info)
    local id = pkg_info.identity
    local shape = pkg_info.shape
    local lines = { "---" }
    lines[#lines + 1] = "name: " .. yaml_scalar(id.name)
    if id.version and id.version ~= "" then
        lines[#lines + 1] = "version: " .. yaml_scalar(id.version)
    end
    if id.category and id.category ~= "" then
        lines[#lines + 1] = "category: " .. yaml_scalar(id.category)
    end
    if shape.result ~= nil then
        local result_str = M.shape_type_string(shape.result)
        if result_str ~= "" then
            lines[#lines + 1] = "result_shape: " .. yaml_scalar(result_str)
        end
    end
    if id.description and id.description ~= "" then
        -- description is always quoted: it's human prose and likely to
        -- contain `:` (em dash phrasing, package references, etc.)
        lines[#lines + 1] = string.format('description: "%s"',
                                          escape_yaml_string(id.description))
    end
    lines[#lines + 1] = "source: " .. yaml_scalar(id.source_path)
    lines[#lines + 1] = "generated: gen_docs (V0)"
    lines[#lines + 1] = "---"
    return table.concat(lines, "\n")
end

local function render_parameters_table(input_schema)
    local lines = {
        "## Parameters {#parameters}",
        "",
        "| key | type | required | description |",
        "|---|---|---|---|",
    }
    local entries = S.fields(input_schema)
    for i = 1, #entries do
        local e = entries[i]
        local req = e.optional and "optional" or "**required**"
        local type_str = M.shape_type_string(e.type)
        local doc = escape_md_table_cell(e.doc or "")
        lines[#lines + 1] = string.format(
            "| `ctx.%s` | %s | %s | %s |",
            e.name, type_str, req, doc)
    end
    return table.concat(lines, "\n")
end

--- Resolve a result schema to its underlying shape for rendering.
---
--- Returns: (shape_schema, ref_name?) or (nil, nil) when the result is
--- not a renderable shape. `ref_name` is present when the input was
--- T.ref(name), so the caller can surface the label in the header.
---
--- `ref` lookup goes through the alc_shapes module table (M.voted /
--- M.paneled / ...) — the same SSoT used by spec_resolver at runtime.
--- An unregistered ref returns (nil, nil) so narrative generation
--- degrades gracefully; strict validation is the responsibility of
--- spec_resolver at runtime and lint at commit time. This mirrors the
--- `type_to_json` tolerance on line 391-392 (emits label without the
--- registry being complete).
local function resolve_shape(schema)
    local peeled = peel(schema)
    local k = rawget(peeled, "kind")
    if k == "ref" then
        local name = rawget(peeled, "name")
        local target = rawget(S, name)
        if target == nil then
            return nil, nil
        end
        peeled = peel(target)
        if rawget(peeled, "kind") ~= "shape" then
            return nil, nil
        end
        return peeled, name
    elseif k == "shape" then
        return peeled, nil
    else
        return nil, nil
    end
end

local function render_result_section(result_schema)
    local shape_peeled, ref_name = resolve_shape(result_schema)
    if shape_peeled == nil then
        return ""
    end
    local header
    if ref_name then
        header = string.format("Returns `%s` shape:", ref_name)
    else
        header = "Returns:"
    end
    local lines = {
        "## Result {#result}",
        "",
        header,
        "",
        "| key | type | optional | description |",
        "|---|---|---|---|",
    }
    local entries = S.fields(shape_peeled)
    for i = 1, #entries do
        local e = entries[i]
        local opt = e.optional and "optional" or "—"
        local type_str = M.shape_type_string(e.type)
        local doc = escape_md_table_cell(e.doc or "")
        lines[#lines + 1] = string.format(
            "| `%s` | %s | %s | %s |",
            e.name, type_str, opt, doc)
    end
    return table.concat(lines, "\n")
end

local function has_renderable_result(result_schema)
    if result_schema == nil then return false end
    local shape_peeled = resolve_shape(result_schema)
    return shape_peeled ~= nil
end

local function render_toc(sections, has_parameters, has_result)
    if #sections == 0 and not has_parameters and not has_result then
        return ""
    end
    local lines = { "## Contents", "" }
    for i = 1, #sections do
        local s = sections[i]
        local indent = string.rep("  ", s.level - 2)
        lines[#lines + 1] = string.format("%s- [%s](#%s)", indent, s.heading, s.anchor)
    end
    if has_parameters then
        lines[#lines + 1] = "- [Parameters](#parameters)"
    end
    if has_result then
        lines[#lines + 1] = "- [Result](#result)"
    end
    return table.concat(lines, "\n")
end

function M.narrative_md(pkg_info)
    local nar = pkg_info.narrative
    local shape = pkg_info.shape
    local has_params = shape.input ~= nil
    local has_result = has_renderable_result(shape.result)

    local parts = {}
    parts[#parts + 1] = build_frontmatter(pkg_info)
    parts[#parts + 1] = ""
    parts[#parts + 1] = "# " .. nar.title

    if nar.summary and nar.summary ~= "" then
        parts[#parts + 1] = ""
        parts[#parts + 1] = "> " .. nar.summary
    end

    local toc = render_toc(nar.sections, has_params, has_result)
    if toc ~= "" then
        parts[#parts + 1] = ""
        parts[#parts + 1] = toc
    end

    for i = 1, #nar.sections do
        local s = nar.sections[i]
        local hashes = string.rep("#", s.level)
        parts[#parts + 1] = ""
        parts[#parts + 1] = string.format("%s %s {#%s}", hashes, s.heading, s.anchor)
        if s.body_md and s.body_md ~= "" then
            parts[#parts + 1] = ""
            parts[#parts + 1] = s.body_md
        end
    end

    if #nar.sections == 0 and nar.summary == "" then
        parts[#parts + 1] = ""
        parts[#parts + 1] = "_(no additional narrative found)_"
    end

    if has_params then
        parts[#parts + 1] = ""
        parts[#parts + 1] = render_parameters_table(shape.input)
    end

    if has_result then
        parts[#parts + 1] = ""
        parts[#parts + 1] = render_result_section(shape.result)
    end

    return table.concat(parts, "\n") .. "\n"
end

-- ── JSON encoder (minimal, pure Lua) ──────────────────────────────────
--
-- Only types produced by our PkgInfo entity are supported:
--   string / number / boolean / nil / table (as array or object).
-- An empty table serialises as `{}` (object) per spec §7.4 — PkgInfo
-- never produces empty arrays at the hub_entry boundary.
--
-- Output is deterministic: object keys are sorted alphabetically so
-- the hub_entry bytes are stable across runs (necessary for hub_index
-- caching / content-addressed storage).

local JSON_ESCAPES = {
    ['"']  = '\\"',  ['\\'] = '\\\\', ['\b'] = '\\b', ['\f'] = '\\f',
    ['\n'] = '\\n',  ['\r'] = '\\r',  ['\t'] = '\\t',
}

local function json_escape_string(s)
    local out = s:gsub('[%z\1-\31\\"]', function(c)
        return JSON_ESCAPES[c] or string.format('\\u%04x', c:byte())
    end)
    return '"' .. out .. '"'
end

local function is_array(t)
    local n = 0
    for k, _ in pairs(t) do
        if type(k) ~= "number" then return false end
        n = n + 1
    end
    for i = 1, n do
        if t[i] == nil then return false end
    end
    return n > 0, n
end

local json_encode_value  -- forward

local function json_encode_array(t, n)
    local parts = {}
    for i = 1, n do
        parts[i] = json_encode_value(t[i])
    end
    return "[" .. table.concat(parts, ",") .. "]"
end

local function json_encode_object(t)
    local keys = {}
    for k, _ in pairs(t) do
        if type(k) ~= "string" then
            error("json_encode: object key must be a string, got " .. type(k))
        end
        keys[#keys + 1] = k
    end
    table.sort(keys)
    local parts = {}
    for i = 1, #keys do
        local k = keys[i]
        parts[i] = json_escape_string(k) .. ":" .. json_encode_value(t[k])
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

function json_encode_value(v)
    local ty = type(v)
    if v == nil then
        return "null"
    elseif ty == "string" then
        return json_escape_string(v)
    elseif ty == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            error("json_encode: non-finite number is not JSON-representable")
        end
        return tostring(v)
    elseif ty == "boolean" then
        return v and "true" or "false"
    elseif ty == "table" then
        local arr, n = is_array(v)
        if arr then
            return json_encode_array(v, n)
        end
        return json_encode_object(v)
    else
        error("json_encode: unsupported type '" .. ty .. "'")
    end
end

local function json_encode(v)
    return json_encode_value(v)
end

-- ── Schema → JSON form (hub_entry projection) ─────────────────────────
--
-- The hub_entry schema exposes field types structurally so consumers
-- can walk the type tree without re-parsing the human string form.
-- This is a PROJECTION — not a parallel AST — so we use stable JSON
-- tag names (`primitive` / `array_of` / `shape` / `label`) that keep
-- byte-identical output across refactors of the underlying alc_shapes
-- DSL. Mapping from alc_shapes kinds:
--   prim          → { kind = "primitive", name = <prim> }
--   any           → { kind = "primitive", name = "any" }
--   array_of      → { kind = "array_of", of = <sub> }
--   map_of        → { kind = "map_of", key, val }
--   one_of        → { kind = "one_of", values }
--   shape         → { kind = "shape", shape = <shape_to_json> }
--   discriminated → { kind = "discriminated", tag, variants }
--   ref           → { kind = "label", name }
-- `optional` / `described` wrappers are peeled (optional-ness is
-- expressed on the enclosing Field, not on the type).

local function type_to_json(schema)
    local peeled = peel(schema)
    local k = rawget(peeled, "kind")
    if k == "prim" then
        return { kind = "primitive", name = rawget(peeled, "prim") }
    elseif k == "any" then
        return { kind = "primitive", name = "any" }
    elseif k == "array_of" then
        return { kind = "array_of", of = type_to_json(rawget(peeled, "elem")) }
    elseif k == "map_of" then
        return {
            kind = "map_of",
            key  = type_to_json(rawget(peeled, "key")),
            val  = type_to_json(rawget(peeled, "val")),
        }
    elseif k == "one_of" then
        local src = rawget(peeled, "values")
        local values = {}
        for i = 1, #src do values[i] = src[i] end
        return { kind = "one_of", values = values }
    elseif k == "shape" then
        return { kind = "shape", shape = M.shape_to_json(peeled) }
    elseif k == "discriminated" then
        local src_variants = rawget(peeled, "variants")
        local variants = {}
        for name, variant_schema in pairs(src_variants) do
            variants[name] = M.shape_to_json(variant_schema)
        end
        return {
            kind = "discriminated",
            tag = rawget(peeled, "tag"),
            variants = variants,
        }
    elseif k == "ref" then
        return { kind = "label", name = rawget(peeled, "name") }
    else
        error("type_to_json: unknown kind '" .. tostring(k) .. "'")
    end
end

--- Convert an alc_shapes `shape` schema to a JSON-ready Lua table.
---
--- Expected input: a schema with kind="shape" (optionally wrapped in
--- optional/described — those wrappers are peeled).
function M.shape_to_json(schema)
    local peeled = peel(schema)
    if rawget(peeled, "kind") ~= "shape" then
        error("shape_to_json: expected kind='shape', got '"
              .. tostring(rawget(peeled, "kind")) .. "'", 2)
    end
    local entries = S.fields(peeled)
    local fields = {}
    for i = 1, #entries do
        local e = entries[i]
        fields[i] = {
            name     = e.name,
            type     = type_to_json(e.type),
            optional = e.optional and true or false,
            doc      = e.doc or "",
        }
    end
    return {
        fields = fields,
        open   = rawget(peeled, "open") and true or false,
    }
end

-- ── hub_entry JSON projection (pipeline-spec §7.4) ────────────────────

--- Build the hub_entry JSON for one PkgInfo.
---
--- Schema (per pipeline-spec.md §7.4):
---   {
---     "name": str, "version": str, "category": str, "description": str,
---     "narrative_md": str,         -- full narrative.md (frontmatter included)
---     "input_shape":  Shape|null,  -- shape_to_json form (fields + open)
---     "result_shape": TypeJSON|null -- type_to_json form (kind-tagged)
---   }
---
--- `TypeJSON` kinds: "primitive" / "array_of" / "map_of" / "one_of" /
--- "shape" / "discriminated" / "label". Consumers dispatch on `kind`:
---   label → registry lookup by `name`
---   shape → structural walk via `shape.fields`
---   other → kind-specific handling
---
--- Role split (hub=machine, narrative=human): hub JSON is the machine
--- contract consumed by alc_hub_info / LLM tooling / IDE / cross-pkg
--- type lint. Human-readable shape form lives in docs/narrative/*.md
--- YAML frontmatter (shape_type_string).
---
--- The bytes are deterministic (keys sorted) so downstream hub_index
--- caching can content-address the JSON.
function M.hub_entry(pkg_info)
    local id  = pkg_info.identity
    local shp = pkg_info.shape
    local entry = {
        name         = id.name,
        version      = id.version or "",
        category     = id.category or "",
        description  = id.description or "",
        narrative_md = M.narrative_md(pkg_info),
    }
    if shp.input ~= nil then
        entry.input_shape = M.shape_to_json(shp.input)
    end
    if shp.result ~= nil then
        entry.result_shape = type_to_json(shp.result)
    end
    return json_encode(entry)
end

-- ── context7.json (pipeline-spec §7.6) ────────────────────────────────
--
-- Public schema: https://context7.com/schema/context7.json
-- All fields are optional, so we emit only what the human-curated
-- `context7_config` module provides plus the two deterministic fields
-- fixed by our pipeline:
--   * $schema : the canonical Context7 schema URL.
--   * folders : ["docs/narrative"] — see pipeline-spec §7.6 ("content
--               is narrative.md and only the narrative/ subfolder is
--               the Context7 source of truth"). Hardcoded here so that
--               the config module cannot drift from the pipeline's
--               output layout.
--
-- Output bytes are deterministic (json_encode sorts object keys).

local CONTEXT7_SCHEMA_URL = "https://context7.com/schema/context7.json"
local CONTEXT7_FOLDERS    = { "docs/narrative" }

--- Copy a list of strings (defensive — the caller's table is not mutated).
local function copy_str_list(src)
    local out = {}
    for i = 1, #src do out[i] = src[i] end
    return out
end

--- Copy a list of objects with a single known key (tag or branch).
--- Only copies the recognised keys so that arbitrary user fields cannot
--- bleed into context7.json (Context7 validates against the public
--- schema and rejects unknown keys).
local function copy_version_list(src, key)
    local out = {}
    for i = 1, #src do
        local v = src[i]
        if type(v) ~= "table" or type(v[key]) ~= "string" or v[key] == "" then
            error(string.format(
                "context7_config: %s[%d] must be a table with a non-empty '%s' field",
                key == "tag" and "previousVersions" or "branchVersions", i, key), 2)
        end
        out[i] = { [key] = v[key] }
    end
    return out
end

--- Build the context7.json body from a human-curated config table.
---
--- Config shape (all keys optional):
---   {
---     projectTitle     = "string",
---     description      = "string",
---     branch           = "string",
---     excludeFolders   = { "string", ... },
---     excludeFiles     = { "string", ... },
---     rules            = { "string", ... },
---     previousVersions = { { tag = "v1.2.1" }, ... },
---     branchVersions   = { { branch = "legacy" }, ... },
---   }
---
--- Returns a deterministic JSON string.
function M.context7_config(config)
    if type(config) ~= "table" then
        error("context7_config: expected a table", 2)
    end
    local entry = {
        ["$schema"] = CONTEXT7_SCHEMA_URL,
        folders     = copy_str_list(CONTEXT7_FOLDERS),
    }
    if type(config.projectTitle) == "string" and config.projectTitle ~= "" then
        entry.projectTitle = config.projectTitle
    end
    if type(config.description) == "string" and config.description ~= "" then
        entry.description = config.description
    end
    if type(config.branch) == "string" and config.branch ~= "" then
        entry.branch = config.branch
    end
    if type(config.excludeFolders) == "table" and #config.excludeFolders > 0 then
        entry.excludeFolders = copy_str_list(config.excludeFolders)
    end
    if type(config.excludeFiles) == "table" and #config.excludeFiles > 0 then
        entry.excludeFiles = copy_str_list(config.excludeFiles)
    end
    if type(config.rules) == "table" and #config.rules > 0 then
        entry.rules = copy_str_list(config.rules)
    end
    if type(config.previousVersions) == "table" and #config.previousVersions > 0 then
        entry.previousVersions = copy_version_list(config.previousVersions, "tag")
    end
    if type(config.branchVersions) == "table" and #config.branchVersions > 0 then
        entry.branchVersions = copy_version_list(config.branchVersions, "branch")
    end
    return json_encode(entry)
end

-- ── .devin/wiki.json (pipeline-spec §7.6) ─────────────────────────────
--
-- Public schema (docs.devin.ai/work-with-devin/deepwiki, 2026-04-17):
--   {
--     repo_notes: [{content: str <=10000, author?: str}],
--     pages:      [{title: str, purpose: str, parent?: str,
--                   page_notes?: [{content: str, author?: str}]}]
--   }
--   Limits: <=30 pages (enterprise 80), <=100 total notes.
--
-- DeepWiki does NOT support a `folders` field — it auto-crawls the
-- repository. pipeline-spec §7.6's "folder 明示" is inapplicable here,
-- so the projection leans on `repo_notes` to tell DeepWiki that
-- `docs/narrative/` is the authoritative source of truth.
--
-- Output bytes are deterministic (json_encode sorts object keys).

local DEVIN_MAX_NOTE_CHARS = 10000
local DEVIN_MAX_PAGES      = 30
local DEVIN_MAX_NOTES      = 100

local function validate_note(note, where, i)
    if type(note) ~= "table" or type(note.content) ~= "string"
        or note.content == "" then
        error(string.format(
            "devin_wiki: %s[%d] must be a table with a non-empty 'content' string",
            where, i), 2)
    end
    if #note.content > DEVIN_MAX_NOTE_CHARS then
        error(string.format(
            "devin_wiki: %s[%d].content exceeds %d chars (%d)",
            where, i, DEVIN_MAX_NOTE_CHARS, #note.content), 2)
    end
    if note.author ~= nil and type(note.author) ~= "string" then
        error(string.format(
            "devin_wiki: %s[%d].author must be a string when present",
            where, i), 2)
    end
end

local function copy_note(note)
    local out = { content = note.content }
    if type(note.author) == "string" and note.author ~= "" then
        out.author = note.author
    end
    return out
end

local function copy_page(page, i, total_notes_ref)
    if type(page) ~= "table" or type(page.title) ~= "string"
        or page.title == "" or type(page.purpose) ~= "string"
        or page.purpose == "" then
        error(string.format(
            "devin_wiki: pages[%d] must be a table with non-empty " ..
            "'title' and 'purpose' strings", i), 2)
    end
    local out = { title = page.title, purpose = page.purpose }
    if type(page.parent) == "string" and page.parent ~= "" then
        out.parent = page.parent
    end
    if page.page_notes ~= nil then
        if type(page.page_notes) ~= "table" then
            error(string.format(
                "devin_wiki: pages[%d].page_notes must be an array", i), 2)
        end
        local notes = {}
        for j = 1, #page.page_notes do
            validate_note(page.page_notes[j],
                "pages[" .. i .. "].page_notes", j)
            notes[j] = copy_note(page.page_notes[j])
            total_notes_ref.n = total_notes_ref.n + 1
        end
        if #notes > 0 then
            out.page_notes = notes
        end
    end
    return out
end

--- Build the .devin/wiki.json body from a human-curated config table.
---
--- Config shape (all keys optional):
---   {
---     repo_notes = { { content = "...", author? = "..." }, ... },
---     pages      = { { title = "...", purpose = "...",
---                      parent? = "...", page_notes? = { ... } }, ... },
---   }
---
--- Returns a deterministic JSON string.
function M.devin_wiki(config)
    if type(config) ~= "table" then
        error("devin_wiki: expected a table", 2)
    end
    local entry = {}
    local total_notes = { n = 0 }

    if config.repo_notes ~= nil then
        if type(config.repo_notes) ~= "table" then
            error("devin_wiki: repo_notes must be an array", 2)
        end
        local notes = {}
        for i = 1, #config.repo_notes do
            validate_note(config.repo_notes[i], "repo_notes", i)
            notes[i] = copy_note(config.repo_notes[i])
            total_notes.n = total_notes.n + 1
        end
        if #notes > 0 then
            entry.repo_notes = notes
        end
    end

    if config.pages ~= nil then
        if type(config.pages) ~= "table" then
            error("devin_wiki: pages must be an array", 2)
        end
        if #config.pages > DEVIN_MAX_PAGES then
            error(string.format(
                "devin_wiki: pages exceeds max %d (%d provided)",
                DEVIN_MAX_PAGES, #config.pages), 2)
        end
        local pages = {}
        local seen_titles = {}
        for i = 1, #config.pages do
            local p = copy_page(config.pages[i], i, total_notes)
            if seen_titles[p.title] then
                error(string.format(
                    "devin_wiki: pages[%d].title '%s' is duplicated " ..
                    "(must be unique)", i, p.title), 2)
            end
            seen_titles[p.title] = true
            pages[i] = p
        end
        if #pages > 0 then
            entry.pages = pages
        end
    end

    if total_notes.n > DEVIN_MAX_NOTES then
        error(string.format(
            "devin_wiki: combined note count %d exceeds max %d",
            total_notes.n, DEVIN_MAX_NOTES), 2)
    end

    return json_encode(entry)
end

-- ── llms.txt / llms-full.txt ──────────────────────────────────────────
--
-- Two-layer API per pipeline-spec §3.2:
--   * `llms_index_line(p)`  — one pkg → one bullet line (per-pkg primitive)
--   * `llms_full_chunk(p, narrative_md?)` — one pkg → one concat chunk
--   * `llms_index(pkg_infos)`  — aggregate the above into llms.txt
--   * `llms_full(pkg_infos)`   — aggregate the above into llms-full.txt
--
-- The aggregators compose the per-pkg primitives, so an external
-- consumer can either take the whole file or drop individual lines /
-- chunks into a custom layout without re-implementing the formatting.

local LLMS_INDEX_DESC_MAX = 200

--- Build one bullet line for `llms.txt` from a single PkgInfo.
---
--- Shape: `- [{name}](narrative/{name}.md): {description}`
---
--- Description is truncated at {LLMS_INDEX_DESC_MAX} chars (trailing
--- "...") so a pathological pkg cannot bloat the index. The path prefix
--- can be overridden via `opts.href_prefix` (default `"narrative/"`) for
--- consumers that nest the narratives in a different folder.
function M.llms_index_line(pkg_info, opts)
    opts = opts or {}
    local prefix = opts.href_prefix or "narrative/"
    local name = pkg_info.identity.name
    local desc = pkg_info.identity.description or ""
    if #desc > LLMS_INDEX_DESC_MAX then
        desc = desc:sub(1, LLMS_INDEX_DESC_MAX - 3) .. "..."
    end
    return string.format("- [%s](%s%s.md): %s", name, prefix, name, desc)
end

--- Strip the YAML frontmatter block from a narrative.md body.
---
--- Safe no-op when the body has no frontmatter.
local function strip_frontmatter(narrative_md)
    if narrative_md:sub(1, 4) ~= "---\n" then
        return narrative_md
    end
    local _, close_idx = narrative_md:find("\n---\n", 5, true)
    if not close_idx then
        return narrative_md
    end
    return narrative_md:sub(close_idx + 1)
end

--- Build the per-pkg concat chunk for `llms-full.txt` from a PkgInfo.
---
--- Shape:
---   <!-- ── {name}.md ── -->
---
---   {narrative.md body, frontmatter stripped, trimmed}
---
---   ---
---
--- `narrative_md` is accepted as an optional second argument so the
--- caller can pass a pre-computed Markdown body (avoiding a re-render
--- when iterating many pkgs). When omitted, `narrative_md(pkg_info)`
--- is invoked. Both paths are pure — no I/O.
function M.llms_full_chunk(pkg_info, narrative_md)
    local body = narrative_md or M.narrative_md(pkg_info)
    body = strip_frontmatter(body)
    body = body:gsub("%s+$", "")
    return string.format("<!-- ── %s.md ── -->\n\n%s\n\n---",
        pkg_info.identity.name, body)
end

--- Build the full llms.txt index from a list of PkgInfo.
---
--- Composes `llms_index_line` per pkg, grouping by category (sorted
--- alphabetically; missing category → "uncategorized"). Pkgs within a
--- category are emitted in name order.
function M.llms_index(pkg_infos, opts)
    opts = opts or {}
    local header = opts.header or "# algocline"
    local tagline = opts.tagline or
        ("LLM amplification engine. Pure Lua strategies executed via " ..
         "`alc.run(ctx)`; this index lists every installed strategy " ..
         "package with a one-liner and a link to its full narrative.")

    local lines = { header, "", "> " .. tagline, "" }

    local by_category = {}
    local categories  = {}
    for i = 1, #pkg_infos do
        local p   = pkg_infos[i]
        local cat = p.identity.category
        if cat == nil or cat == "" then cat = "uncategorized" end
        if not by_category[cat] then
            by_category[cat] = {}
            categories[#categories + 1] = cat
        end
        local bucket = by_category[cat]
        bucket[#bucket + 1] = p
    end
    table.sort(categories)

    local line_opts = { href_prefix = opts.href_prefix }
    for _, cat in ipairs(categories) do
        lines[#lines + 1] = "## " .. cat
        lines[#lines + 1] = ""
        local bucket = by_category[cat]
        table.sort(bucket, function(a, b)
            return a.identity.name < b.identity.name
        end)
        for _, p in ipairs(bucket) do
            lines[#lines + 1] = M.llms_index_line(p, line_opts)
        end
        lines[#lines + 1] = ""
    end

    while #lines > 0 and lines[#lines] == "" do
        lines[#lines] = nil
    end
    return table.concat(lines, "\n") .. "\n"
end

--- Concatenate all narrative.md bodies into the llms-full.txt file.
---
--- `entries` is a list of either:
---   (a) PkgInfo, or
---   (b) { name = "...", narrative_md = "...", pkg_info = PkgInfo? }
---
--- Form (b) is backward-compatible with callers that pre-render
--- narratives (e.g. `gen_docs.lua` avoids a second render pass by
--- passing the cached Markdown body). Form (a) lets a fresh caller
--- pass PkgInfo directly; the aggregator renders as needed.
function M.llms_full(entries, opts)
    opts = opts or {}
    local header = opts.header or "# algocline — full narrative index"
    local tagline = opts.tagline or
        ("Concatenation of every package narrative. Intended for bulk AI " ..
         "context injection; for selective access see `llms.txt`.")

    local lines = { header, "", "> " .. tagline, "" }

    local normalized = {}
    for i = 1, #entries do
        local e = entries[i]
        if e.identity ~= nil then
            -- PkgInfo form
            normalized[i] = { name = e.identity.name, pkg_info = e }
        else
            normalized[i] = {
                name         = e.name,
                narrative_md = e.narrative_md,
                pkg_info     = e.pkg_info,
            }
        end
    end
    table.sort(normalized, function(a, b) return a.name < b.name end)

    for i = 1, #normalized do
        local e = normalized[i]
        local info = e.pkg_info
        if info == nil then
            -- Legacy form: narrative_md-only entry. Synthesise a minimal
            -- PkgInfo with just the identity the chunk renderer needs.
            info = { identity = { name = e.name } }
        end
        lines[#lines + 1] = M.llms_full_chunk(info, e.narrative_md)
        lines[#lines + 1] = ""
    end

    while #lines > 0 and lines[#lines] == "" do
        lines[#lines] = nil
    end
    return table.concat(lines, "\n") .. "\n"
end

M._internal = {
    peel                     = peel,
    build_frontmatter        = build_frontmatter,
    render_parameters_table  = render_parameters_table,
    render_toc               = render_toc,
    escape_yaml_string       = escape_yaml_string,
    escape_md_table_cell     = escape_md_table_cell,
    yaml_scalar              = yaml_scalar,
    json_encode              = json_encode,
    type_to_json             = type_to_json,
    strip_frontmatter        = strip_frontmatter,
    LLMS_INDEX_DESC_MAX      = LLMS_INDEX_DESC_MAX,
    CONTEXT7_SCHEMA_URL      = CONTEXT7_SCHEMA_URL,
    CONTEXT7_FOLDERS         = CONTEXT7_FOLDERS,
    DEVIN_MAX_NOTE_CHARS     = DEVIN_MAX_NOTE_CHARS,
    DEVIN_MAX_PAGES          = DEVIN_MAX_PAGES,
    DEVIN_MAX_NOTES          = DEVIN_MAX_NOTES,
}

return M
