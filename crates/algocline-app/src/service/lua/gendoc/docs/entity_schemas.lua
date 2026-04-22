--- tools.docs.entity_schemas — Entity registry for the docs pipeline.
---
--- Plain `{name → schema}` data. No closures, no opaque state.
--- Pass via `S.check(v, EntitySchemas.PkgInfo, { registry = EntitySchemas })`.
---
--- Entity 境界は strict (open=false)。未知 field は契約違反として
--- loud-fail する。このポリシーは Runtime result shape (alc_shapes の
--- pass-through culture) とは意図的に反対。
---
--- Single source of truth: pipeline-spec.md §3 (Core Entity: PkgInfo)
---
--- Design notes:
---   * Schema-as-Data 主義に従い、このモジュールは closure を保持しない。
---   * T.ref("Identity") 等の内部参照は `check` 側で registry (このモジュール
---     自身) を辿ることで解決される。
---   * shape.input / shape.result は alc_shapes schema data を受ける。
---     完全 meta-schema は記述せず、"kind field を持つ table" という
---     structural invariant のみを要求する (`AlcSchema` を open=true で定義)。

local T = require("alc_shapes.t")

local M = {}

-- Leaf: alc_shapes schema data.
--
-- C6: kind は既知 kind 集合に閉じる (whitelist)。Entity 境界は strict で
-- あるはずなのに、従来は kind = T.string だったため `{kind="garbage"}`
-- のような malformed schema が entity 検証を通過し、後段の check_node
-- で "unknown kind 'garbage'" として遅延失敗していた。whitelist 化に
-- より "Entity strict = 境界で落とす" ポリシーと整合する。
--
-- 新しい kind を t.lua に追加するときは ALC_SCHEMA_KINDS も更新する
-- (年単位で稀な保守コスト)。open=true により残りの structural body
-- (prim/elem/fields/variants 等) は alc_shapes 側の invariant に委ねる。
local ALC_SCHEMA_KINDS = {
    "any", "array_of", "described", "discriminated", "map_of",
    "one_of", "optional", "prim", "ref", "shape",
}

local AlcSchema = T.shape({
    kind = T.one_of(ALC_SCHEMA_KINDS),
}, { open = true })

M.Identity = T.shape({
    name        = T.string,
    version     = T.string,
    category    = T.string,
    description = T.string,
    source_path = T.string,
}, { open = false })

M.Section = T.shape({
    level   = T.one_of({ 2, 3 }),
    heading = T.string,
    anchor  = T.string,
    body_md = T.string,
}, { open = false })

M.Narrative = T.shape({
    title    = T.string,
    summary  = T.string,
    sections = T.array_of(T.ref("Section")),
}, { open = false })

M.Shape = T.shape({
    input  = AlcSchema:is_optional(),
    result = AlcSchema:is_optional(),
}, { open = false })

M.PkgInfo = T.shape({
    identity  = T.ref("Identity"),
    narrative = T.ref("Narrative"),
    shape     = T.ref("Shape"),
}, { open = false })

return M
