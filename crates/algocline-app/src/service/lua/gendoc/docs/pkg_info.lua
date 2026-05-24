--- tools.docs.pkg_info — Core Entity schema.
---
--- Data-only module. Defines the PkgInfo entity and its pure
--- constructors. All downstream modules (extract / projections / lint)
--- produce or consume this schema.
---
--- Single-AST doctrine (see alc_shapes/README.md §Core concept):
--- `shape.input` / `shape.result` are alc_shapes schemas directly —
--- no parallel TypeExpr AST. Projections (`tools.docs.projections`)
--- walk the schema via `rawget` and `alc_shapes.fields()`.
---
--- PkgInfo layout
---
---   {
---     identity  = { name, version, category, description, source_path },
---     narrative = {
---       title   = <string>,       -- docstring 1st line
---       summary = <string>,       -- abstract paragraph (joined)
---       sections = { Section, ... },
---     },
---     shape = {
---       input  = <alc_shapes schema>|nil,   -- typically T.shape(...)
---       result = <alc_shapes schema>|nil,   -- T.shape / T.ref / ...
---     },
---     docs = {
---       schema_version = <number>|nil,   -- M.docs field-set version, default 1
---     },
---     -- Note: M.docs.narrative removed in #1778221491-39903.
---     -- Narrative content is now served on-the-fly from the init.lua
---     -- docstring H2 sections via the gendoc pipeline.
---   }
---
---   Section = { level = 2|3, heading = <string>,
---               anchor = <string>, body_md = <string> }

local M = {}

function M.make_section(level, heading, anchor, body_md)
    return {
        level = level,
        heading = heading,
        anchor = anchor,
        body_md = body_md,
    }
end

function M.make_pkg_info(identity, narrative, shape, docs)
    return {
        identity = identity,
        narrative = narrative,
        shape = shape,
        docs = docs or { schema_version = nil },
    }
end

return M
