--- gameai_metrics.tier_manifest — read a boss tier manifest and resolve
--- one tier into the decode condition it ships under.
---
--- A tier manifest is the file a harvest run's selection ends at: one
--- entry per operational tier (`weak` / `mid` / `strong`), each naming
--- the Card that tier ships **and the decode condition it was measured
--- at**. The pair is the unit — a Card read at another temperature is
--- not the tier that was promoted, it is a different measurement of the
--- same weights — so this reader hands both back together and never
--- invents either half.
---
--- ## Why this lives in gameai_metrics (not next to the NPC)
---
--- `card_id` / `alias` / `tier` / band bookkeeping are boss-harvest
--- vocabulary, the same argument `harvest_collection.lua` makes for
--- itself: the manifest is written by the measurement side and read by a
--- caller wiring the measured condition into a run. The NPC packages
--- stay a decode surface and learn nothing about manifests.
---
--- ## Contract
---
--- `load(path) -> manifest`
---
--- - `path` is required. There is no default: a reader that guessed a
---   workspace path would answer a caller who moved the file with the
---   numbers of the one they left behind.
--- - The file is read through `io.open` and decoded with
---   `alc.json_decode`.
--- - `kind` and `schema_version` are checked **first**, before a single
---   entry is read. The sibling manifests in this package (a harvest
---   collection, a fight matrix) are also `schema_version = 1` JSON
---   objects with an `entries` array, so pointing at one by mistake has
---   to fail on what the file *is* rather than on a field it happens to
---   be missing three checks later.
--- - The routing fields are validated strictly (`style`, and per entry
---   `tier` / `alias` / `card_id` / `decode`), because every one of them
---   silently changes what gets decoded: a missing `style` would leave
---   the NPC on its own default basis, a duplicated `tier` would make
---   the routing depend on iteration order, and a `decode` that is
---   neither `"greedy"` nor a positive temperature is not a decode
---   condition at all.
--- - Every other field is passed through untouched. The measurement
---   bookkeeping (`runtime_level_p`, `runtime_band`, `pool_margins_T`,
---   `evidence`, ...) is what makes the manifest readable as provenance,
---   and a reader that dropped or policed it would either lose it or go
---   stale on the next field the measurement side adds.
--- - `style` is checked as a non-empty string, not against
---   `guardian_duel.STYLES`. The whitelist belongs to the decode
---   surface (`guardian_duel_npc` validates the basis it is handed) and
---   a persona basis is a legitimate value here.
---
--- `resolve(manifest, tier) -> {tier, card_id, alias, style, decode}`
---
--- - `style` is lifted from the manifest top level, where it is
---   declared once for the whole file. Carrying it in the resolved row
---   is the point: a caller that read only the entry would decode a
---   non-`guardian` manifest against whatever basis its decode surface
---   defaults to, and still return a number.
--- - `decode` is returned exactly as written — `"greedy"` stays the
---   string. It is not normalised to `0` or `nil`, because the caller
---   has two different code paths (a scan and a draw) and a number that
---   happened to mean "no draw" would be one substitution away from
---   seeding a sampler with it.
--- - An unknown tier is a loud error listing the tiers the manifest
---   carries, rather than a `nil` a caller could read as "greedy".
---
--- ## Usage
---
--- ```lua
--- local tm = require("gameai_metrics.tier_manifest")
--- local manifest = tm.load("workspace/gameai-harvest/tier_v1.json")
--- local row = tm.resolve(manifest, "strong")
--- -- row = { tier = "strong", alias = "guardian_duel_npc_strong",
--- --         card_id = "guardian_duel_npc_strong_...",
--- --         style = "guardian", decode = 0.5 }
--- local task = row.decode == "greedy"
---         and { mode = "decide", state = state }
---     or { mode = "decide_noisy", state = state, temperature = row.decode, seed = seed }
--- ```
---
--- The `card_id` rides along because the alias is a rebindable pin: the
--- Card a tier was measured as is the `card_id`, so a caller seating the
--- alias should assert the two still agree before it decodes.

local M = {}

---@type AlcMeta
M.meta = {
    name = "gameai_metrics.tier_manifest",
    version = "0.1.0",
    description = "Read a boss tier manifest and resolve a tier into its (Card, decode condition) pair.",
    category = "game",
}

--- The only manifest kind this reader answers for.
local KIND = "tier_manifest"

--- The only schema version this reader answers for.
local SCHEMA_VERSION = 1

--- Decode condition spelling that means the legal-gated greedy scan.
local GREEDY = "greedy"

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "tier_manifest: alc.json_decode is required to read the manifest (host bridge missing)",
            0
        )
    end
    return alc.json_decode
end

--- Read and decode the file at `path`.
local function read_json(path)
    local decode = require_json_decoder()
    local f, open_err = io.open(path, "r")
    if f == nil then
        error(string.format("tier_manifest.load: cannot open %q: %s", path, tostring(open_err)), 3)
    end
    local body = f:read("a")
    f:close()
    local ok, parsed = pcall(decode, body)
    if not ok then
        error(
            string.format("tier_manifest.load: failed to decode %q: %s", path, tostring(parsed)),
            3
        )
    end
    if type(parsed) ~= "table" then
        error(
            string.format(
                "tier_manifest.load: %q must decode to an object, got %s",
                path,
                type(parsed)
            ),
            3
        )
    end
    return parsed
end

--- Assert the file is a tier manifest of a version this reader knows.
---
--- Runs before anything reads `entries`, so a harvest collection or a
--- fight report handed over by mistake is named as the wrong kind of
--- file rather than as a tier manifest with odd entries.
local function require_kind(parsed, path)
    if parsed.kind ~= KIND then
        error(
            string.format(
                "tier_manifest.load: %q is not a tier manifest (kind = %s, expected %q); "
                    .. "the harvest collection and the fight reports are different files",
                path,
                parsed.kind == nil and "absent" or string.format("%q", tostring(parsed.kind)),
                KIND
            ),
            3
        )
    end
    if parsed.schema_version ~= SCHEMA_VERSION then
        error(
            string.format(
                "tier_manifest.load: %q has schema_version %s, expected %d",
                path,
                tostring(parsed.schema_version),
                SCHEMA_VERSION
            ),
            3
        )
    end
end

--- Assert a decode condition is one this reader can hand to a caller:
--- the greedy spelling, or the positive finite temperature a draw takes.
local function require_decode(raw, path, index)
    if raw == GREEDY then
        return raw
    end
    if type(raw) == "number" and raw == raw and raw ~= math.huge and raw > 0 then
        return raw
    end
    error(
        string.format(
            "tier_manifest.load: %q entries[%d].decode must be %q or a finite positive "
                .. "temperature, got %s",
            path,
            index,
            GREEDY,
            tostring(raw)
        ),
        4
    )
end

local function require_string_field(entry, field, path, index)
    local value = entry[field]
    if type(value) ~= "string" or value == "" then
        error(
            string.format(
                "tier_manifest.load: %q entries[%d].%s must be a non-empty string, got %s",
                path,
                index,
                field,
                type(value)
            ),
            4
        )
    end
    return value
end

--- Validate the routing fields of every entry, leaving the rest alone.
local function require_entries(parsed, path)
    local entries = parsed.entries
    if type(entries) ~= "table" then
        error(
            string.format(
                "tier_manifest.load: %q has no entries array (got %s)",
                path,
                type(entries)
            ),
            3
        )
    end
    if #entries == 0 then
        error(
            string.format("tier_manifest.load: %q carries zero entries; nothing to route", path),
            3
        )
    end
    local seen = {}
    for index, entry in ipairs(entries) do
        if type(entry) ~= "table" then
            error(
                string.format(
                    "tier_manifest.load: %q entries[%d] must be an object, got %s",
                    path,
                    index,
                    type(entry)
                ),
                3
            )
        end
        local tier = require_string_field(entry, "tier", path, index)
        if seen[tier] then
            error(string.format("tier_manifest.load: %q lists tier %q twice", path, tier), 3)
        end
        seen[tier] = true
        require_string_field(entry, "alias", path, index)
        require_string_field(entry, "card_id", path, index)
        require_decode(entry.decode, path, index)
    end
end

--- Read a tier manifest off disk.
---
--- Returns the decoded manifest as-is (measurement fields included), so
--- a caller reading provenance and a caller routing a decode read the
--- same object.
---@param path string Manifest path; required, no default
---@return table manifest
function M.load(path)
    if type(path) ~= "string" or path == "" then
        error("tier_manifest.load: path must be a non-empty string", 2)
    end
    local parsed = read_json(path)
    require_kind(parsed, path)
    if type(parsed.style) ~= "string" or parsed.style == "" then
        error(
            string.format(
                "tier_manifest.load: %q has no top-level style string; it is the basis every "
                    .. "entry is decoded against and there is no default worth guessing",
                path
            ),
            2
        )
    end
    require_entries(parsed, path)
    return parsed
end

--- The tiers a manifest carries, in file order, for error messages.
local function tier_names(manifest)
    local names = {}
    for _, entry in ipairs(manifest.entries) do
        names[#names + 1] = tostring(entry.tier)
    end
    return table.concat(names, ", ")
end

--- Resolve one tier into the (Card, decode condition) pair it ships as.
---@param manifest table Manifest returned by `load`
---@param tier string Tier name, e.g. `"strong"`
---@return table row `{ tier, card_id, alias, style, decode }`
function M.resolve(manifest, tier)
    if type(manifest) ~= "table" or type(manifest.entries) ~= "table" then
        error(
            "tier_manifest.resolve: manifest must be the table tier_manifest.load returned, got "
                .. type(manifest),
            2
        )
    end
    if type(tier) ~= "string" or tier == "" then
        error("tier_manifest.resolve: tier must be a non-empty string", 2)
    end
    for _, entry in ipairs(manifest.entries) do
        if entry.tier == tier then
            return {
                tier = entry.tier,
                card_id = entry.card_id,
                alias = entry.alias,
                -- Declared once at the top level, carried per row: a
                -- caller that read the entry alone would decode against
                -- whatever basis its decode surface defaults to.
                style = manifest.style,
                decode = entry.decode,
            }
        end
    end
    error(
        string.format(
            "tier_manifest.resolve: unknown tier %q (this manifest carries: %s)",
            tier,
            tier_names(manifest)
        ),
        2
    )
end

M.KIND = KIND
M.SCHEMA_VERSION = SCHEMA_VERSION
M.GREEDY = GREEDY

return M
