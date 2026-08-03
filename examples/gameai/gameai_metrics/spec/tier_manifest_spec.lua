-- gameai_metrics/spec/tier_manifest_spec.lua
--
-- Package-level spec for `tier_manifest`. The reader touches two host
-- surfaces: `io.open` (to read the file) and `alc.json_decode` (to parse
-- it). Both are exercised for real — the manifests below are written to
-- temp files as JSON text and read back through the module — because the
-- failures this reader exists to produce (a sibling manifest handed over
-- by mistake, a body that is not JSON at all) only happen on that path.
--
-- The decoder installed on `alc` is the same subset the other specs in
-- this package carry: numbers, strings, booleans, null, arrays and
-- objects. That is everything a tier manifest holds.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- The pkg_test VM installs `alc` before the spec runs; the fallback
-- keeps the file runnable under a bare mlua VM too.
alc = alc or {}

-- ─── JSON decoder ───────────────────────────────────────────────────

local function json_decode(text)
    local pos = 1
    local parse_value
    local function skip_ws()
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == " " or ch == "\n" or ch == "\r" or ch == "\t" then
                pos = pos + 1
            else
                return
            end
        end
    end
    local function parse_string()
        assert(text:sub(pos, pos) == '"', "spec json_decode: expected string at " .. pos)
        pos = pos + 1
        local parts, start = {}, pos
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == '"' then
                parts[#parts + 1] = text:sub(start, pos - 1)
                pos = pos + 1
                return table.concat(parts)
            elseif ch == "\\" then
                parts[#parts + 1] = text:sub(start, pos - 1)
                local esc = text:sub(pos + 1, pos + 1)
                if esc == '"' or esc == "\\" or esc == "/" then
                    parts[#parts + 1] = esc
                elseif esc == "n" then
                    parts[#parts + 1] = "\n"
                elseif esc == "t" then
                    parts[#parts + 1] = "\t"
                elseif esc == "r" then
                    parts[#parts + 1] = "\r"
                else
                    error("spec json_decode: unsupported escape \\" .. esc, 0)
                end
                pos = pos + 2
                start = pos
            else
                pos = pos + 1
            end
        end
        error("spec json_decode: unterminated string", 0)
    end
    local function parse_number()
        local start = pos
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch:match("[%-%+%d%.eE]") then
                pos = pos + 1
            else
                break
            end
        end
        local value = tonumber(text:sub(start, pos - 1))
        if value == nil then
            error("spec json_decode: not a value at " .. start, 0)
        end
        return value
    end
    local function parse_array()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "]" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            out[#out + 1] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "]" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or ] at " .. pos, 0)
            end
        end
    end
    local function parse_object()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "}" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            local key = parse_string()
            skip_ws()
            assert(text:sub(pos, pos) == ":", "spec json_decode: expected : at " .. pos)
            pos = pos + 1
            skip_ws()
            out[key] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "}" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or } at " .. pos, 0)
            end
        end
    end
    parse_value = function()
        skip_ws()
        local ch = text:sub(pos, pos)
        if ch == "{" then
            return parse_object()
        end
        if ch == "[" then
            return parse_array()
        end
        if ch == '"' then
            return parse_string()
        end
        if ch == "t" and text:sub(pos, pos + 3) == "true" then
            pos = pos + 4
            return true
        end
        if ch == "f" and text:sub(pos, pos + 4) == "false" then
            pos = pos + 5
            return false
        end
        if ch == "n" and text:sub(pos, pos + 3) == "null" then
            pos = pos + 4
            return nil
        end
        return parse_number()
    end
    skip_ws()
    return parse_value()
end

alc.json_decode = json_decode

local tm = require("gameai_metrics.tier_manifest")

-- ─── Temp file helper ───────────────────────────────────────────────

local _tmp_counter = 0
local function write_manifest(body)
    _tmp_counter = _tmp_counter + 1
    local path = os.tmpname() .. "-tier-" .. tostring(_tmp_counter)
    local f = assert(io.open(path, "w"))
    f:write(body)
    f:close()
    return path
end

--- A well-formed manifest, shaped after `workspace/gameai-harvest/
--- tier_v1.json`: the three tiers, one greedy and two temperatures, with
--- the measurement bookkeeping the selection recorded next to them.
local FULL = [[
{
    "schema_version": 1,
    "kind": "tier_manifest",
    "style": "guardian",
    "issue": "c8ff4ca3",
    "entries": [
        {
            "tier": "weak",
            "alias": "guardian_duel_npc_weak",
            "card_id": "guardian_duel_npc_weak_1785614346278270",
            "step": 60,
            "decode": "greedy",
            "runtime_level_p": 0.0681,
            "runtime_band": [0.03, 0.2],
            "evidence": ["audit_run8_greedy_seedB.json"]
        },
        {
            "tier": "mid",
            "alias": "guardian_duel_npc_mid",
            "card_id": "guardian_duel_npc_mid_v2_1785767264397937",
            "provenance_alias": "guardian_duel_npc_mid_v2",
            "decode": 1.0,
            "runtime_level_p": 0.8756,
            "runtime_band": [0.6, 0.9]
        },
        {
            "tier": "strong",
            "alias": "guardian_duel_npc_strong",
            "card_id": "guardian_duel_npc_strong_1785614347088735",
            "decode": 0.5,
            "runtime_level_p": 0.93,
            "runtime_band": [0.851, 0.98]
        }
    ]
}
]]

--- Marker for "leave this field out".
---
--- A `nil` cannot say it: a table value of `nil` reads as "not
--- overridden", which would quietly hand the case the default it was
--- written to remove.
local ABSENT = {}

--- One entry, with the two override tables splicing over the defaults,
--- as a manifest body. Written as text rather than encoded so a case can
--- plant a value no encoder would produce (a bare `true` decode, a field
--- that is not there at all).
local function manifest_with(top, entry)
    local defaults = {
        schema_version = "1",
        kind = '"tier_manifest"',
        style = '"guardian"',
    }
    for key, value in pairs(top or {}) do
        defaults[key] = value
    end
    local entry_defaults = {
        tier = '"mid"',
        alias = '"guardian_duel_npc_mid"',
        card_id = '"guardian_duel_npc_mid_v2_1785767264397937"',
        decode = "1.0",
    }
    for key, value in pairs(entry or {}) do
        entry_defaults[key] = value
    end
    local head, body = {}, {}
    for key, value in pairs(defaults) do
        if value ~= ABSENT then
            head[#head + 1] = string.format('"%s": %s', key, value)
        end
    end
    for key, value in pairs(entry_defaults) do
        if value ~= ABSENT then
            body[#body + 1] = string.format('"%s": %s', key, value)
        end
    end
    return "{" .. table.concat(head, ",") .. ', "entries": [{' .. table.concat(body, ",") .. "}]}"
end

describe("gameai_metrics.tier_manifest load", function()
    it("reads a well-formed manifest", function()
        local manifest = tm.load(write_manifest(FULL))
        expect(manifest.kind).to.equal("tier_manifest")
        expect(manifest.schema_version).to.equal(1)
        expect(manifest.style).to.equal("guardian")
        expect(#manifest.entries).to.equal(3)
        expect(manifest.entries[1].tier).to.equal("weak")
        expect(manifest.entries[1].decode).to.equal("greedy")
        expect(manifest.entries[3].decode).to.equal(0.5)
    end)

    it("passes the measurement fields through untouched", function()
        -- The manifest is provenance as much as routing: a reader that
        -- kept only the fields it validates would drop the numbers that
        -- say what the tier was promoted on.
        local manifest = tm.load(write_manifest(FULL))
        local weak = manifest.entries[1]
        expect(weak.runtime_level_p).to.equal(0.0681)
        expect(weak.runtime_band[1]).to.equal(0.03)
        expect(weak.runtime_band[2]).to.equal(0.2)
        expect(weak.evidence[1]).to.equal("audit_run8_greedy_seedB.json")
        expect(weak.step).to.equal(60)
        expect(manifest.entries[2].provenance_alias).to.equal("guardian_duel_npc_mid_v2")
        expect(manifest.issue).to.equal("c8ff4ca3")
    end)

    it("requires a path", function()
        -- No default: a reader that guessed a workspace path would
        -- answer a caller who moved the file with stale numbers.
        expect(function()
            tm.load(nil)
        end).to.fail()
        expect(function()
            tm.load("")
        end).to.fail()
    end)

    it("names a file it cannot open", function()
        local ok, err = pcall(tm.load, os.tmpname() .. "-tier-missing")
        expect(ok).to.equal(false)
        expect(err:find("cannot open") ~= nil).to.equal(true)
    end)

    it("names a body that is not JSON", function()
        local ok, err = pcall(tm.load, write_manifest("not json at all"))
        expect(ok).to.equal(false)
        expect(err:find("decode") ~= nil).to.equal(true)
    end)

    it("refuses a body that is not an object", function()
        expect(function()
            tm.load(write_manifest("[1, 2, 3]"))
        end).to.fail()
    end)

    it("refuses a sibling manifest before it reads a single entry", function()
        -- A harvest collection is also a schema_version 1 object with an
        -- entries array, so the mistake has to fail on what the file is.
        local collection = [[
{
    "schema_version": 1,
    "run_id": "run8",
    "style": "guardian",
    "policy": "first_writer_wins",
    "entries": [{"label": "mid", "step": 120, "card_id": "x", "alias": "y"}]
}
]]
        local ok, err = pcall(tm.load, write_manifest(collection))
        expect(ok).to.equal(false)
        expect(err:find("kind") ~= nil).to.equal(true)
        expect(err:find("absent") ~= nil).to.equal(true)
    end)

    it("refuses a manifest of another kind", function()
        local ok, err = pcall(tm.load, write_manifest(manifest_with({ kind = '"fight_matrix"' })))
        expect(ok).to.equal(false)
        expect(err:find("tier manifest") ~= nil).to.equal(true)
    end)

    it("refuses a schema version it does not know", function()
        local ok, err = pcall(tm.load, write_manifest(manifest_with({ schema_version = "2" })))
        expect(ok).to.equal(false)
        expect(err:find("schema_version") ~= nil).to.equal(true)
    end)

    it("refuses a manifest with no top-level style", function()
        -- The basis is declared once for the file; without it the NPC
        -- would decode against whatever it defaults to and still answer.
        local ok, err = pcall(tm.load, write_manifest(manifest_with({ style = ABSENT })))
        expect(ok).to.equal(false)
        expect(err:find("style") ~= nil).to.equal(true)
    end)

    it("refuses a manifest with no entries", function()
        expect(function()
            tm.load(write_manifest('{"schema_version": 1, "kind": "tier_manifest", "style": "g"}'))
        end).to.fail()
        expect(function()
            tm.load(
                write_manifest(
                    '{"schema_version": 1, "kind": "tier_manifest", "style": "g", "entries": []}'
                )
            )
        end).to.fail()
    end)

    it("refuses an entry that is not an object", function()
        expect(function()
            tm.load(
                write_manifest(
                    '{"schema_version": 1, "kind": "tier_manifest", "style": "g", '
                        .. '"entries": ["mid"]}'
                )
            )
        end).to.fail()
    end)

    it("refuses a tier listed twice", function()
        -- Two rows for one tier makes the routing depend on iteration
        -- order, which is exactly the silent half of a wrong decode.
        local body = '{"schema_version": 1, "kind": "tier_manifest", "style": "guardian", '
            .. '"entries": ['
            .. '{"tier": "mid", "alias": "a", "card_id": "c1", "decode": 1.0},'
            .. '{"tier": "mid", "alias": "b", "card_id": "c2", "decode": 0.5}]}'
        local ok, err = pcall(tm.load, write_manifest(body))
        expect(ok).to.equal(false)
        expect(err:find("twice") ~= nil).to.equal(true)
    end)

    it("refuses an entry without a tier", function()
        expect(function()
            tm.load(write_manifest(manifest_with(nil, { tier = ABSENT })))
        end).to.fail()
        expect(function()
            tm.load(write_manifest(manifest_with(nil, { tier = '""' })))
        end).to.fail()
    end)

    it("refuses an entry without an alias", function()
        expect(function()
            tm.load(write_manifest(manifest_with(nil, { alias = ABSENT })))
        end).to.fail()
    end)

    it("refuses an entry without a card_id", function()
        -- The alias is a rebindable pin; the card_id is what the tier
        -- was measured as, so an entry missing it cannot be checked.
        local ok, err = pcall(tm.load, write_manifest(manifest_with(nil, { card_id = ABSENT })))
        expect(ok).to.equal(false)
        expect(err:find("card_id") ~= nil).to.equal(true)
        expect(function()
            tm.load(write_manifest(manifest_with(nil, { card_id = '""' })))
        end).to.fail()
    end)

    it("accepts both spellings of a decode condition", function()
        local greedy = tm.load(write_manifest(manifest_with(nil, { decode = '"greedy"' })))
        expect(greedy.entries[1].decode).to.equal("greedy")
        local warm = tm.load(write_manifest(manifest_with(nil, { decode = "0.5" })))
        expect(warm.entries[1].decode).to.equal(0.5)
    end)

    it("refuses a decode that is neither greedy nor a positive temperature", function()
        for _, spelling in ipairs({ "0", "-0.5", '"greeedy"', "true", '"0.5"' }) do
            local ok, err =
                pcall(tm.load, write_manifest(manifest_with(nil, { decode = spelling })))
            expect(ok).to.equal(false)
            expect(err:find("decode") ~= nil).to.equal(true)
        end
        expect(function()
            tm.load(write_manifest(manifest_with(nil, { decode = ABSENT })))
        end).to.fail()
    end)
end)

describe("gameai_metrics.tier_manifest resolve", function()
    local manifest = tm.load(write_manifest(FULL))

    it("returns the Card and the decode condition as one pair", function()
        expect(tm.resolve(manifest, "strong")).to.equal({
            tier = "strong",
            alias = "guardian_duel_npc_strong",
            card_id = "guardian_duel_npc_strong_1785614347088735",
            style = "guardian",
            decode = 0.5,
        })
    end)

    it("carries the manifest style into every row", function()
        -- The basis lives at the top level and the entry never repeats
        -- it, so a resolver that did not lift it would hand the caller a
        -- row that silently decodes against a default.
        for _, tier in ipairs({ "weak", "mid", "strong" }) do
            expect(tm.resolve(manifest, tier).style).to.equal("guardian")
        end
    end)

    it("keeps greedy as the literal it was written as", function()
        -- Not normalised to 0 or nil: the caller branches on it, and a
        -- number meaning "no draw" is one substitution away from being
        -- handed to a sampler.
        local weak = tm.resolve(manifest, "weak")
        expect(weak.decode).to.equal("greedy")
        expect(tm.resolve(manifest, "mid").decode).to.equal(1.0)
    end)

    it("refuses an unknown tier and lists the ones it carries", function()
        local ok, err = pcall(tm.resolve, manifest, "elite")
        expect(ok).to.equal(false)
        expect(err:find("elite") ~= nil).to.equal(true)
        expect(err:find("weak, mid, strong") ~= nil).to.equal(true)
    end)

    it("refuses a tier that is not a non-empty string", function()
        expect(function()
            tm.resolve(manifest, nil)
        end).to.fail()
        expect(function()
            tm.resolve(manifest, "")
        end).to.fail()
    end)

    it("refuses something that is not a loaded manifest", function()
        expect(function()
            tm.resolve({ style = "guardian" }, "mid")
        end).to.fail()
        expect(function()
            tm.resolve("workspace/gameai-harvest/tier_v1.json", "mid")
        end).to.fail()
    end)
end)
