-- gameai_metrics/spec/harvest_collection_spec.lua
--
-- Package-level spec for `harvest_collection`. The helper only touches
-- two host surfaces: `alc.json_encode` (to serialise the manifest) and
-- `io.open` (to write it). Both are exercised with a real temp file so
-- the round-trip through JSON is checked end-to-end rather than
-- inspecting the intermediate encoded string.
--
-- The spec never speaks to any real Card or metric registry — the
-- helper does not either. Every `records` table the spec hands over is
-- a hand-built array whose shape matches what `anymetric.observe`
-- would have produced on a real fire (the same `{step, view_id, metric,
-- values}` / `{step, view_id, error}` records).

local describe, it, expect = lust.describe, lust.it, lust.expect

-- The pkg_test VM installs `alc` before the spec runs; the fallback
-- keeps the file runnable under a bare mlua VM too.
alc = alc or {}

-- ─── JSON stub ──────────────────────────────────────────────────────
--
-- The helper only calls `alc.json_encode`. The paired decoder below is
-- used by the spec itself so the round-trip test can read back what
-- the helper wrote and assert on the parsed structure rather than on
-- fragile substring matches over the encoded string.

local function encode(value)
    local kind = type(value)
    if value == nil then
        return "null"
    end
    if kind == "boolean" or kind == "number" then
        return tostring(value)
    end
    if kind == "string" then
        return string.format("%q", value)
    end
    if kind ~= "table" then
        error("spec json_encode: unsupported type " .. kind, 0)
    end
    -- Detect array shape by checking that keys are exactly 1..n. Falls
    -- back to object encoding when either the length is zero and the
    -- table still holds keys or the numeric keys are sparse.
    local n = #value
    if n > 0 then
        local total = 0
        for _ in pairs(value) do
            total = total + 1
        end
        if total == n then
            local items = {}
            for index = 1, n do
                items[index] = encode(value[index])
            end
            return "[" .. table.concat(items, ",") .. "]"
        end
    end
    local keys = {}
    for key in pairs(value) do
        keys[#keys + 1] = key
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    local fields = {}
    for _, key in ipairs(keys) do
        fields[#fields + 1] = string.format("%q", tostring(key)) .. ":" .. encode(value[key])
    end
    return "{" .. table.concat(fields, ",") .. "}"
end

alc.json_encode = encode

--- Minimal recursive-descent JSON decoder used by the round-trip test.
--- Only understands the subset the encoder above emits (numbers,
--- booleans, null, quoted strings without escapes beyond `\"` and
--- `\\`, arrays, objects); that is exactly what the manifest holds.
local function json_decode(text)
    local pos = 1
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
    local parse_value
    local function parse_string()
        assert(text:sub(pos, pos) == '"', "spec json_decode: expected string at " .. pos)
        pos = pos + 1
        local start = pos
        local parts = {}
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
        return tonumber(text:sub(start, pos - 1))
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

-- ─── Temp file helper ───────────────────────────────────────────────
--
-- os.tmpname on macOS returns a path under /var/folders/... that this
-- process can write. Append a small tag so parallel specs in the same
-- suite do not collide.

local _tmp_counter = 0
local function tmp_path()
    _tmp_counter = _tmp_counter + 1
    return os.tmpname() .. "-harvest-" .. tostring(_tmp_counter)
end

-- ─── Records builders ───────────────────────────────────────────────
--
-- Hand-built records with the exact shape `anymetric.observe` would
-- emit. The helper only reads `view_id` / `values.*` / `error`, so
-- these skeletons are enough.

local function level_record(step, win_rate, ci_lower)
    return {
        step = step,
        view_id = "level",
        metric = "level",
        values = { win_rate = win_rate, ci_lower = ci_lower, ci_upper = 1.0 },
    }
end

local function sd_record(step, value)
    return {
        step = step,
        view_id = "sd_teacher",
        metric = "style_distance",
        values = { value = value },
    }
end

local function trickiness_record(step, value)
    return {
        step = step,
        view_id = "trickiness",
        metric = "trickiness",
        values = { value = value, raw_mean = value * 1.2 },
    }
end

local function harvest_decision(label, step, values)
    return {
        action = "harvest",
        reason = string.format("hit %s at step %d", label, step),
        meta = { label = label, step = step, values = values },
    }
end

local hc = require("gameai_metrics.harvest_collection")

-- ─── Specs ──────────────────────────────────────────────────────────

describe("gameai_metrics.harvest_collection.new", function()
    it("stores path / bands / policy default (first_writer_wins)", function()
        local coll = hc.new({
            path = "/tmp/does-not-matter-yet.json",
            meta = { style = "guardian", steps = 300 },
            bands = {
                { lo = 0.10, hi = 0.30, label = "weak" },
                { lo = 0.55, hi = 0.85, label = "mid" },
            },
        })
        expect(coll:path()).to.equal("/tmp/does-not-matter-yet.json")
        expect(coll:policy()).to.equal("first_writer_wins")
        expect(#coll:entries()).to.equal(0)
    end)

    it("accepts an explicit last_writer_wins policy", function()
        local coll = hc.new({
            path = "/tmp/x.json",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
            policy = "last_writer_wins",
        })
        expect(coll:policy()).to.equal("last_writer_wins")
    end)

    it("refuses an unknown policy", function()
        local ok, err = pcall(hc.new, {
            path = "/tmp/x.json",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
            policy = "average_writer_wins",
        })
        expect(ok).to.equal(false)
        expect(err:find("policy") ~= nil).to.equal(true)
    end)

    it("refuses a missing path", function()
        local ok, err = pcall(hc.new, {
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        expect(ok).to.equal(false)
        expect(err:find("path") ~= nil).to.equal(true)
    end)

    it("refuses a missing bands", function()
        local ok, err = pcall(hc.new, { path = "/tmp/x.json" })
        expect(ok).to.equal(false)
        expect(err:find("bands") ~= nil).to.equal(true)
    end)

    it("refuses a non-table band entry", function()
        local ok, err = pcall(hc.new, {
            path = "/tmp/x.json",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" }, "not a band" },
        })
        expect(ok).to.equal(false)
        expect(err:find("bands%[2%]") ~= nil).to.equal(true)
    end)

    it("keeps a band's require clause in the recorded bands", function()
        local coll = hc.new({
            path = "/tmp/x.json",
            bands = {
                { lo = 0.03, hi = 0.20, label = "weak_v2" },
                {
                    lo = 0.60,
                    hi = 0.85,
                    label = "mid_v2",
                    require = { view_id = "trickiness", field = "value", min = 0.57 },
                },
            },
        })
        local bands = coll:_manifest().bands
        expect(bands[1].require).to.equal(nil)
        expect(type(bands[2].require)).to.equal("table")
        expect(bands[2].require.view_id).to.equal("trickiness")
        expect(bands[2].require.field).to.equal("value")
        expect(bands[2].require.min).to.equal(0.57)
    end)

    it("records only the three require fields, not whatever else the caller attached", function()
        local coll = hc.new({
            path = "/tmp/x.json",
            bands = {
                {
                    lo = 0.60,
                    hi = 0.85,
                    label = "mid_v2",
                    require = {
                        view_id = "trickiness",
                        field = "value",
                        min = 0.57,
                        note = "not part of the contract",
                    },
                },
            },
        })
        local required = coll:_manifest().bands[1].require
        expect(required.note).to.equal(nil)
        expect(required.min).to.equal(0.57)
    end)

    it("refuses a non-table require", function()
        local ok, err = pcall(hc.new, {
            path = "/tmp/x.json",
            bands = { { lo = 0.60, hi = 0.85, label = "mid_v2", require = "trickiness" } },
        })
        expect(ok).to.equal(false)
        expect(err:find("require") ~= nil).to.equal(true)
    end)

    it("leaves the recorded band shape unchanged when no band requires anything", function()
        local coll = hc.new({
            path = "/tmp/x.json",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        local band = coll:_manifest().bands[1]
        local keys = 0
        for _ in pairs(band) do
            keys = keys + 1
        end
        expect(keys).to.equal(3)
        expect(band.require).to.equal(nil)
    end)

    it("copies meta so a later caller mutation does not bleed into the manifest", function()
        local meta = { style = "guardian", steps = 300 }
        local coll = hc.new({
            path = "/tmp/x.json",
            meta = meta,
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        meta.style = "invader"
        local m = coll:_manifest()
        expect(m.style).to.equal("guardian")
    end)
end)

describe("gameai_metrics.harvest_collection:append", function()
    local function fresh(policy)
        return hc.new({
            path = "/tmp/append-test.json",
            meta = { style = "guardian", steps = 300 },
            bands = {
                { lo = 0.10, hi = 0.30, label = "weak" },
                { lo = 0.55, hi = 0.85, label = "mid" },
                { lo = 0.85, hi = 0.98, label = "strong" },
            },
            policy = policy,
        })
    end

    it("stores label / step / ckpt_path from dec + info", function()
        local coll = fresh()
        local dec = harvest_decision("mid", 180, { win_rate = 0.93 })
        local info = { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" }
        local stored = coll:append(dec, info, {
            level_record(180, 0.93, 0.825),
        })
        expect(stored).to.equal(true)
        local entries = coll:entries()
        expect(#entries).to.equal(1)
        expect(entries[1].label).to.equal("mid")
        expect(entries[1].step).to.equal(180)
        expect(entries[1].ckpt_path).to.equal("/nn/ckpt-000180.safetensors")
    end)

    it(
        "extracts level_win_rate / level_ci_lower / sd_teacher / trickiness_norm by view_id",
        function()
            local coll = fresh()
            local records = {
                level_record(180, 0.93, 0.825),
                sd_record(180, 0.089),
                trickiness_record(180, 0.417),
            }
            coll:append(
                harvest_decision("mid", 180, {}),
                { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
                records
            )
            local e = coll:entries()[1]
            expect(e.level_win_rate).to.equal(0.93)
            expect(e.level_ci_lower).to.equal(0.825)
            expect(e.sd_teacher).to.equal(0.089)
            expect(e.trickiness_norm).to.equal(0.417)
        end
    )

    it("carries extra fields (card_id, alias) verbatim", function()
        local coll = fresh()
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            {},
            { card_id = "guardian_duel_npc_mid_abc123", alias = "guardian_duel_npc_mid" }
        )
        local e = coll:entries()[1]
        expect(e.card_id).to.equal("guardian_duel_npc_mid_abc123")
        expect(e.alias).to.equal("guardian_duel_npc_mid")
    end)

    it("leaves fields absent when the record is missing or errored", function()
        local coll = fresh()
        -- Only trickiness present; no level record; sd_teacher errored.
        local records = {
            { step = 180, view_id = "sd_teacher", error = "metric blew up" },
            trickiness_record(180, 0.417),
        }
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            records
        )
        local e = coll:entries()[1]
        expect(e.level_win_rate).to.equal(nil)
        expect(e.level_ci_lower).to.equal(nil)
        expect(e.sd_teacher).to.equal(nil)
        expect(e.trickiness_norm).to.equal(0.417)
    end)

    it("first_writer_wins: a second harvest of the same label is skipped", function()
        local coll = fresh()
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            { level_record(180, 0.93, 0.825) }
        )
        local second = coll:append(
            harvest_decision("mid", 240, {}),
            { step = 240, ckpt_path = "/nn/ckpt-000240.safetensors" },
            { level_record(240, 0.90, 0.786) }
        )
        expect(second).to.equal(false)
        local entries = coll:entries()
        expect(#entries).to.equal(1)
        expect(entries[1].step).to.equal(180)
        expect(entries[1].ckpt_path).to.equal("/nn/ckpt-000180.safetensors")
        expect(entries[1].level_win_rate).to.equal(0.93)
    end)

    it("last_writer_wins: a second harvest of the same label overwrites in place", function()
        local coll = fresh("last_writer_wins")
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            { level_record(180, 0.93, 0.825) }
        )
        local second = coll:append(
            harvest_decision("mid", 240, {}),
            { step = 240, ckpt_path = "/nn/ckpt-000240.safetensors" },
            { level_record(240, 0.90, 0.786) }
        )
        expect(second).to.equal(true)
        local entries = coll:entries()
        expect(#entries).to.equal(1)
        expect(entries[1].step).to.equal(240)
        expect(entries[1].ckpt_path).to.equal("/nn/ckpt-000240.safetensors")
        expect(entries[1].level_win_rate).to.equal(0.90)
    end)

    it("preserves insertion order when several labels are stored", function()
        local coll = fresh()
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "mid.st" },
            { level_record(180, 0.7, 0.6) }
        )
        coll:append(
            harvest_decision("weak", 240, {}),
            { step = 240, ckpt_path = "weak.st" },
            { level_record(240, 0.2, 0.1) }
        )
        coll:append(
            harvest_decision("strong", 300, {}),
            { step = 300, ckpt_path = "strong.st" },
            { level_record(300, 0.9, 0.87) }
        )
        local entries = coll:entries()
        expect(#entries).to.equal(3)
        expect(entries[1].label).to.equal("mid")
        expect(entries[2].label).to.equal("weak")
        expect(entries[3].label).to.equal("strong")
    end)

    it("refuses a non-harvest decision", function()
        local coll = fresh()
        local ok, err = pcall(
            coll.append,
            coll,
            { action = "continue", reason = "still climbing" },
            { step = 60, ckpt_path = "/x.st" },
            {}
        )
        expect(ok).to.equal(false)
        expect(err:find("harvest") ~= nil).to.equal(true)
    end)

    it("refuses a harvest without meta.label", function()
        local coll = fresh()
        local ok, err = pcall(coll.append, coll, {
            action = "harvest",
            reason = "no label",
            meta = { step = 60 },
        }, { step = 60, ckpt_path = "/x.st" }, {})
        expect(ok).to.equal(false)
        expect(err:find("label") ~= nil).to.equal(true)
    end)

    it("refuses an info without a numeric step", function()
        local coll = fresh()
        local ok, err = pcall(coll.append, coll, harvest_decision("mid", 180, {}), {
            step = "one-eighty",
            ckpt_path = "/x.st",
        }, {})
        expect(ok).to.equal(false)
        expect(err:find("step") ~= nil).to.equal(true)
    end)

    it("refuses a non-table extra", function()
        local coll = fresh()
        local ok, err = pcall(
            coll.append,
            coll,
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/x.st" },
            {},
            "card=abc"
        )
        expect(ok).to.equal(false)
        expect(err:find("extra") ~= nil).to.equal(true)
    end)
end)

describe("gameai_metrics.harvest_collection:save", function()
    it("writes schema_version = 1 into the JSON manifest", function()
        local path = tmp_path()
        local coll = hc.new({
            path = path,
            meta = { style = "guardian", steps = 300 },
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        coll:save()
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(parsed.schema_version).to.equal(1)
    end)

    it("persists the run metadata / bands / policy / partial entries end-to-end", function()
        local path = tmp_path()
        local coll = hc.new({
            path = path,
            meta = {
                run_id = "run-42",
                style = "guardian",
                steps = 300,
                ckpt_every = 60,
                gate_games = 50,
                seed = 0,
            },
            bands = {
                { lo = 0.10, hi = 0.30, label = "weak" },
                { lo = 0.55, hi = 0.85, label = "mid" },
                { lo = 0.85, hi = 0.98, label = "strong" },
            },
        })
        -- Only the mid band actually fires this run, matching the
        -- partial-collection case from the design's §1 observations.
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            {
                level_record(180, 0.93, 0.825),
                sd_record(180, 0.089),
                trickiness_record(180, 0.417),
            },
            { card_id = "card_mid", alias = "guardian_duel_npc_mid" }
        )
        coll:save()
        local f = io.open(path, "r")
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(parsed.style).to.equal("guardian")
        expect(parsed.steps).to.equal(300)
        expect(parsed.ckpt_every).to.equal(60)
        expect(parsed.gate_games).to.equal(50)
        expect(parsed.seed).to.equal(0)
        expect(parsed.run_id).to.equal("run-42")
        expect(parsed.policy).to.equal("first_writer_wins")
        expect(#parsed.bands).to.equal(3)
        expect(parsed.bands[1].label).to.equal("weak")
        expect(parsed.bands[3].label).to.equal("strong")
        -- Partial: only one entry, but the bands list still names all three.
        expect(#parsed.entries).to.equal(1)
        local entry = parsed.entries[1]
        expect(entry.label).to.equal("mid")
        expect(entry.step).to.equal(180)
        expect(entry.ckpt_path).to.equal("/nn/ckpt-000180.safetensors")
        expect(entry.card_id).to.equal("card_mid")
        expect(entry.alias).to.equal("guardian_duel_npc_mid")
        expect(entry.level_win_rate).to.equal(0.93)
        expect(entry.level_ci_lower).to.equal(0.825)
        expect(entry.sd_teacher).to.equal(0.089)
        expect(entry.trickiness_norm).to.equal(0.417)
    end)

    it("round-trips a band require clause through the written JSON", function()
        -- The selection rule a harvest ran under has to survive into the
        -- file: a manifest that records `mid_v2` as a bare interval
        -- cannot tell a reader that the entry also had to clear a floor
        -- on a second view.
        local path = tmp_path()
        local coll = hc.new({
            path = path,
            meta = { style = "guardian" },
            bands = {
                { lo = 0.03, hi = 0.20, label = "weak_v2" },
                {
                    lo = 0.60,
                    hi = 0.85,
                    label = "mid_v2",
                    require = { view_id = "trickiness", field = "value", min = 0.57 },
                },
            },
        })
        coll:append(
            harvest_decision("mid_v2", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            {
                level_record(180, 0.93, 0.825),
                trickiness_record(180, 0.795),
            }
        )
        coll:save()
        local f = io.open(path, "r")
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(#parsed.bands).to.equal(2)
        expect(parsed.bands[1].require).to.equal(nil)
        expect(parsed.bands[2].require.view_id).to.equal("trickiness")
        expect(parsed.bands[2].require.field).to.equal("value")
        expect(parsed.bands[2].require.min).to.equal(0.57)
        expect(parsed.entries[1].label).to.equal("mid_v2")
        expect(parsed.entries[1].trickiness_norm).to.equal(0.795)
    end)

    it("writes the same band shape as before when no band requires anything", function()
        local path = tmp_path()
        local coll = hc.new({
            path = path,
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        coll:save()
        local f = io.open(path, "r")
        local body = f:read("a")
        f:close()
        os.remove(path)
        local band = json_decode(body).bands[1]
        expect(band.lo).to.equal(0.55)
        expect(band.hi).to.equal(0.85)
        expect(band.label).to.equal("mid")
        expect(band.require).to.equal(nil)
    end)

    it("can be called repeatedly and reflects the current entries every time", function()
        local path = tmp_path()
        local coll = hc.new({
            path = path,
            bands = {
                { lo = 0.10, hi = 0.30, label = "weak" },
                { lo = 0.55, hi = 0.85, label = "mid" },
            },
        })
        coll:append(
            harvest_decision("mid", 120, {}),
            { step = 120, ckpt_path = "mid.st" },
            { level_record(120, 0.7, 0.6) }
        )
        coll:save()
        coll:append(
            harvest_decision("weak", 60, {}),
            { step = 60, ckpt_path = "weak.st" },
            { level_record(60, 0.2, 0.1) }
        )
        coll:save()
        local f = io.open(path, "r")
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(#parsed.entries).to.equal(2)
    end)

    it("raises loudly when the output path is not writable", function()
        -- The path is an existing directory: `ensure_parent_dir` sees
        -- the parent already exists (mkdir -p is a no-op) and then
        -- io.open(path, "w") fails because you cannot open a directory
        -- for writing. Keeps the io.open error path exercised even
        -- after the auto-mkdir was inserted at the top of :save().
        local coll = hc.new({
            path = "/tmp",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        local ok, err = pcall(coll.save, coll)
        expect(ok).to.equal(false)
        expect(err:find("cannot open") ~= nil).to.equal(true)
    end)

    it("auto-creates a missing parent directory before writing", function()
        -- Build a path whose parent directory does not yet exist so
        -- the ensure_parent_dir call at the top of :save() has to
        -- actually run mkdir -p. os.tmpname() reserves a fresh file
        -- path under a writable tmp root; we append a nested subdir
        -- so the parent is guaranteed absent when :save() starts.
        local base = os.tmpname()
        os.remove(base) -- reclaim the reserved file, we only wanted a unique prefix
        local dir = base .. "-parent"
        local path = dir .. "/nested/harvest.json"
        local coll = hc.new({
            path = path,
            meta = { style = "guardian" },
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/nn/ckpt-000180.safetensors" },
            { level_record(180, 0.93, 0.825) }
        )
        coll:save()
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        os.remove(dir .. "/nested")
        os.remove(dir)
        local parsed = json_decode(body)
        expect(parsed.schema_version).to.equal(1)
        expect(#parsed.entries).to.equal(1)
        expect(parsed.entries[1].label).to.equal("mid")
    end)

    it("is a no-op when the parent directory already exists", function()
        -- Second save() into the same parent directory must succeed
        -- without complaint (mkdir -p is idempotent). Byte-equality of
        -- the two writes anchors the "existing parent is safe" contract.
        local base = os.tmpname()
        os.remove(base)
        local dir = base .. "-existing"
        assert(os.execute("mkdir -p '" .. dir .. "'"))
        local path = dir .. "/harvest.json"
        local coll = hc.new({
            path = path,
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/x.st" },
            { level_record(180, 0.7, 0.6) }
        )
        coll:save()
        local f1 = io.open(path, "r")
        local body1 = f1:read("a")
        f1:close()
        coll:save()
        local f2 = io.open(path, "r")
        local body2 = f2:read("a")
        f2:close()
        os.remove(path)
        os.remove(dir)
        expect(body1).to.equal(body2)
    end)
end)

describe("gameai_metrics.harvest_collection:entries", function()
    it("returns a fresh copy so a caller cannot shorten the collection", function()
        local coll = hc.new({
            path = "/tmp/x.json",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        coll:append(
            harvest_decision("mid", 180, {}),
            { step = 180, ckpt_path = "/x.st" },
            { level_record(180, 0.7, 0.6) }
        )
        local view = coll:entries()
        table.remove(view, 1)
        expect(#view).to.equal(0)
        -- Original collection is untouched.
        expect(#coll:entries()).to.equal(1)
    end)
end)
