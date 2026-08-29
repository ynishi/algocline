-- gameai_metrics/spec/audit_matrix_spec.lua
--
-- Package-level spec for `audit_matrix`. Stubs `alc.card`,
-- `alc.nn.card.load_handle`, the `gameai_metrics` ctx adapters and
-- `alc.json_encode` / `alc.json_decode` so the runner logic (option
-- validation, view assembly, per-Card observation extraction,
-- symmetric SD matrix build, JSON round-trip through
-- `_fs.ensure_parent_dir`) is exercised without a real Card or model.
--
-- The stub `evaluate` returns canned numbers keyed on the marker
-- table `load_handle` produces per alias, so an audit run produces a
-- deterministic report the spec can assert on field-by-field.

local describe, it, expect = lust.describe, lust.it, lust.expect

alc = alc or {}

-- ─── Minimal JSON encoder / decoder ─────────────────────────────────
--
-- Same subset as harvest_collection_spec.lua: numbers, strings (with
-- `\"` / `\\` escapes), booleans, arrays (dense 1..n) and objects with
-- keys sorted for determinism. Enough to round-trip the audit report
-- and let the parent-dir-absent test read the manifest back.

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

alc.json_encode = encode
alc.json_decode = json_decode

-- ─── Card / metric stubs ────────────────────────────────────────────
--
-- `make_handle(alias)` returns a marker table so `evaluate` can look up
-- per-Card canned numbers by identity. `load_handle` is keyed on
-- `card_id`, so the spec seeds `alc.card.get_by_alias` and the
-- collection-path branch with `"card-" .. alias` ids.

local HANDLES_BY_ALIAS = {}
local HANDLES_BY_CARD_ID = {}

local function make_handle(alias)
    local handle = { alias = alias }
    HANDLES_BY_ALIAS[alias] = handle
    HANDLES_BY_CARD_ID["card-" .. alias] = handle
    return handle
end

alc.card = {
    get_by_alias = function(alias)
        if HANDLES_BY_ALIAS[alias] == nil then
            return nil
        end
        return { card_id = "card-" .. alias }
    end,
}

alc.nn = alc.nn or {}
alc.nn.card = {
    load_handle = function(card_id)
        return HANDLES_BY_CARD_ID[card_id]
    end,
}

-- Canned metric evaluators. The runner calls the adapters with
-- `{card = handle, ...}` for level / trickiness and
-- `{card_a = ha, card_b = hb, ...}` for pair-wise style_distance;
-- the stubs pick the return value off `handle.alias`.

local LEVEL_BY_ALIAS = {}
local TRICKY_BY_ALIAS = {}
local SD_TEACHER_BY_ALIAS = {}
local SD_PAIR = {}

local function pair_key(a, b)
    if a <= b then
        return a .. "|" .. b
    end
    return b .. "|" .. a
end

local function set_sd_pair(a, b, value)
    SD_PAIR[pair_key(a, b)] = value
end

local function set_level(alias, win_rate, ci_lower, ci_upper)
    LEVEL_BY_ALIAS[alias] = {
        win_rate = win_rate,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        wins = win_rate,
        n_games = 1,
        win_rate_min = win_rate,
        per_opponent = {},
    }
end

--- Every ctx a stub adapter was called with, in fire order (the
--- `EVAL_CALLS` harness `spec/train_guardian_npc_spec.lua` uses for the
--- same job). A view config is not readable from the outside, so the
--- view-assembly specs read it where it actually lands: the ctx the
--- metric was handed. `anymetric.observe` merges the view config with
--- the shared ctx (`{card, step}` only here), so a key absent from
--- both is absent on this ctx — which is exactly what "this view
--- carries no temperature" means.
local EVAL_CALLS = {}

--- Every ctx the named metric was evaluated with, in fire order.
local function calls_to(name)
    local out = {}
    for _, call in ipairs(EVAL_CALLS) do
        if call.name == name then
            out[#out + 1] = call.ctx
        end
    end
    return out
end

alc.nn.metric = alc.nn.metric or {}

--- The one dispatcher the three stub adapters share, so the canned
--- value tables stay keyed in a single place.
local function evaluate(name, ctx)
    EVAL_CALLS[#EVAL_CALLS + 1] = { name = name, ctx = ctx }
    if name == "level" then
        local alias = ctx.card and ctx.card.alias
        local canned = LEVEL_BY_ALIAS[alias]
        if canned == nil then
            error("spec: level has no canned value for alias " .. tostring(alias), 0)
        end
        return canned
    end
    if name == "trickiness" then
        local alias = ctx.card and ctx.card.alias
        local canned = TRICKY_BY_ALIAS[alias]
        if canned == nil then
            error("spec: trickiness has no canned value for alias " .. tostring(alias), 0)
        end
        return { value = canned, raw_mean = canned * 1.2 }
    end
    if name == "style_distance" then
        -- Distinguish the per-Card sd_teacher fire (card_b is the
        -- teacher alias string) from the pair-wise matrix fire
        -- (card_b is a Card handle table). The per-Card path
        -- carries the Card under measurement in `ctx.card` (from
        -- shared_ctx), while the SD-matrix path carries it in
        -- `ctx.card_a`; the real init.lua compose falls back the
        -- same way (`ctx.card_a or ctx.card`).
        if type(ctx.card_b) == "string" then
            local carrier = ctx.card_a or ctx.card
            local alias = carrier and carrier.alias
            local canned = SD_TEACHER_BY_ALIAS[alias]
            if canned == nil then
                error("spec: sd_teacher has no canned value for alias " .. tostring(alias), 0)
            end
            return canned
        end
        local a = ctx.card_a and ctx.card_a.alias
        local b = ctx.card_b and ctx.card_b.alias
        local canned = SD_PAIR[pair_key(a, b)]
        if canned == nil then
            error(
                "spec: SD_PAIR has no canned value for " .. tostring(a) .. " / " .. tostring(b),
                0
            )
        end
        return canned
    end
    error("spec: unexpected metric name " .. tostring(name), 0)
end

-- `audit_matrix` reaches its metrics through the package table
-- (`gm.metrics.*`), resolved per call, so the fake package must be in
-- place before the runner is required — and a case that wants to count
-- fires can swap one entry of this table afterwards.
package.loaded["gameai_metrics"] = {
    metrics = {
        level = function(ctx)
            return evaluate("level", ctx)
        end,
        trickiness = function(ctx)
            return evaluate("trickiness", ctx)
        end,
        style_distance = function(ctx)
            return evaluate("style_distance", ctx)
        end,
    },
}

-- ─── Prompt-set / seed anchor ───────────────────────────────────────
--
-- The runner samples its prompt set from real `guardian_duel` self-play
-- (`new_game` / `apply` / the two random policies), so a real
-- `guardian_duel` require is fine and the rules are never stubbed —
-- the states under test have to be states the engine can actually
-- produce. The specs never pass their own prompt_set (letting the
-- default path exercise itself), except where the override branch or a
-- smaller hand-built matrix is the subject.

local duel = require("guardian_duel")

--- Deterministic stand-in for the host RNG bridge both seats of a
--- sampling rollout draw from. Same LCG the sibling specs use
--- (`gameai_metrics/spec/level_spec.lua:164-183`,
--- `spec/guardian_duel_noisy_equivalence_spec.lua:370-378`), which
--- mirrors the host signature the engine-level harness pins
--- (`crates/algocline-engine/tests/lua/card_duel_rules_test.lua:31-35`):
--- `rng_int(rng, min, max)` is inclusive at both ends.
---
--- Only reproducibility matters here. The sampler's contract is "the
--- same (seed, size, style) draws the same set", not "the draws are
--- uniform", so a spec-grade generator pins everything the spec can
--- legitimately assert.
alc.math = alc.math or {}
alc.math.rng_create = function(seed)
    local state = math.floor(seed or 0)
    if state == 0 then
        state = 0x9E3779B9
    end
    return { _state = state % 2147483648 }
end
alc.math.rng_int = function(rng, min, max)
    local s = (rng._state * 1103515245 + 12345) % 2147483648
    rng._state = s
    return min + (s // 65536) % (max - min + 1)
end

local function boss_state(seed)
    return duel.new_game(seed).boss
end

local audit_matrix = require("gameai_metrics.audit_matrix")

local BASE_STYLE = "guardian"

-- Convenience: pre-seed a set of canned metric values for a list of
-- aliases so each `it` block can call `set_scenario({"weak","mid","strong"})`
-- once and get a self-contained scenario.
local function reset_all()
    HANDLES_BY_ALIAS = {}
    HANDLES_BY_CARD_ID = {}
    LEVEL_BY_ALIAS = {}
    TRICKY_BY_ALIAS = {}
    SD_TEACHER_BY_ALIAS = {}
    SD_PAIR = {}
    -- Fire log is per-case: a leaked ctx from an earlier case would
    -- make a "the level view carries a temperature" assertion pass off
    -- another case's fire.
    EVAL_CALLS = {}
    -- Re-install so the load_handle closure sees the fresh tables.
    alc.nn.card.load_handle = function(card_id)
        return HANDLES_BY_CARD_ID[card_id]
    end
    alc.card.get_by_alias = function(alias)
        if HANDLES_BY_ALIAS[alias] == nil then
            return nil
        end
        return { card_id = "card-" .. alias }
    end
end

local function seed_alias(alias, level_tuple, tricky, sd_teacher)
    make_handle(alias)
    set_level(alias, level_tuple[1], level_tuple[2], level_tuple[3])
    TRICKY_BY_ALIAS[alias] = tricky
    if sd_teacher ~= nil then
        SD_TEACHER_BY_ALIAS[alias] = sd_teacher
    end
end

local _tmp_counter = 0
local function tmp_path()
    _tmp_counter = _tmp_counter + 1
    return os.tmpname() .. "-audit-" .. tostring(_tmp_counter)
end

-- ─── Specs: option validation ───────────────────────────────────────

describe("gameai_metrics.audit_matrix.new — options", function()
    it("refuses opts that are not a table", function()
        local ok, err = pcall(audit_matrix.new, "nope")
        expect(ok).to.equal(false)
        expect(err:find("opts must be a table") ~= nil).to.equal(true)
    end)

    it("refuses both collection_path and aliases at once", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            collection_path = "/tmp/x.json",
            aliases = { "a", "b" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("not both") ~= nil).to.equal(true)
    end)

    it("refuses neither collection_path nor aliases", function()
        local ok, err = pcall(audit_matrix.new, { style = BASE_STYLE })
        expect(ok).to.equal(false)
        expect(err:find("required") ~= nil).to.equal(true)
    end)

    it("refuses a missing style", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, { aliases = { "a", "b" } })
        expect(ok).to.equal(false)
        expect(err:find("style") ~= nil).to.equal(true)
    end)

    it("refuses fewer than two Cards", function()
        reset_all()
        make_handle("solo")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "solo" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("pair%-wise") ~= nil).to.equal(true)
    end)

    it("defaults n_games=200 / prompt_set_size=16 / seed=0", function()
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.3)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.3)
        set_sd_pair("a", "b", 0.1)
        local audit = audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
        })
        local report = audit:run()
        expect(report.meta.n_games).to.equal(200)
        expect(report.meta.prompt_set_size).to.equal(16)
        expect(report.meta.seed).to.equal(0)
        expect(report.meta.style).to.equal(BASE_STYLE)
        expect(report.meta.teacher_alias).to.equal(nil)
    end)

    it("refuses an unknown alias (get_by_alias returns nil)", function()
        reset_all()
        make_handle("a")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "ghost" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("ghost") ~= nil).to.equal(true)
    end)

    it("refuses a duplicated alias", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b", "a" },
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("twice") ~= nil).to.equal(true)
    end)

    it("refuses an empty teacher_alias", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            teacher_alias = "",
        })
        expect(ok).to.equal(false)
        expect(err:find("teacher_alias") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: run() happy path ────────────────────────────────────────

describe("gameai_metrics.audit_matrix:run — 3-alias per_card / sd_matrix", function()
    local function three_alias_audit(teacher_alias)
        reset_all()
        seed_alias("weak", { 0.20, 0.15, 0.30 }, 0.10, teacher_alias and 0.72 or nil)
        seed_alias("mid", { 0.70, 0.60, 0.80 }, 0.35, teacher_alias and 0.09 or nil)
        seed_alias("strong", { 0.92, 0.86, 0.98 }, 0.42, teacher_alias and 0.24 or nil)
        set_sd_pair("weak", "mid", 0.55)
        set_sd_pair("weak", "strong", 0.60)
        set_sd_pair("mid", "strong", 0.11)
        if teacher_alias ~= nil then
            make_handle(teacher_alias)
        end
        return audit_matrix.new({
            aliases = { "weak", "mid", "strong" },
            style = BASE_STYLE,
            teacher_alias = teacher_alias,
            n_games = 50,
            prompt_set_size = 4,
            seed = 7,
        })
    end

    it("populates per_card with win_rate / ci / trickiness / sd_teacher", function()
        local audit = three_alias_audit("guardian_duel_npc")
        local report = audit:run()
        expect(report.per_card.weak.win_rate).to.equal(0.20)
        expect(report.per_card.weak.ci_lower).to.equal(0.15)
        expect(report.per_card.weak.ci_upper).to.equal(0.30)
        expect(report.per_card.weak.trickiness_norm).to.equal(0.10)
        expect(report.per_card.weak.sd_teacher).to.equal(0.72)
        expect(report.per_card.mid.sd_teacher).to.equal(0.09)
        expect(report.per_card.strong.sd_teacher).to.equal(0.24)
    end)

    it("builds a symmetric SD matrix with diagonal zero", function()
        local audit = three_alias_audit(nil)
        local report = audit:run()
        for _, alias in ipairs({ "weak", "mid", "strong" }) do
            expect(report.sd_matrix[alias][alias]).to.equal(0.0)
        end
        expect(report.sd_matrix.weak.mid).to.equal(report.sd_matrix.mid.weak)
        expect(report.sd_matrix.weak.strong).to.equal(report.sd_matrix.strong.weak)
        expect(report.sd_matrix.mid.strong).to.equal(report.sd_matrix.strong.mid)
        expect(report.sd_matrix.weak.mid).to.equal(0.55)
        expect(report.sd_matrix.weak.strong).to.equal(0.60)
        expect(report.sd_matrix.mid.strong).to.equal(0.11)
    end)

    it("skips the sd_teacher view when teacher_alias is nil", function()
        local audit = three_alias_audit(nil)
        local report = audit:run()
        expect(report.per_card.weak.sd_teacher).to.equal(nil)
        expect(report.per_card.mid.sd_teacher).to.equal(nil)
        expect(report.per_card.strong.sd_teacher).to.equal(nil)
        expect(report.meta.teacher_alias).to.equal(nil)
    end)

    it("carries meta.teacher_alias when it was set", function()
        local audit = three_alias_audit("guardian_duel_npc")
        local report = audit:run()
        expect(report.meta.teacher_alias).to.equal("guardian_duel_npc")
    end)

    it("returns the same report from :report() as from :run()", function()
        local audit = three_alias_audit(nil)
        local report = audit:run()
        expect(audit:report()).to.equal(report)
    end)

    it("raises when :report() is called before :run()", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        set_level("a", 0.5, 0.4, 0.6)
        set_level("b", 0.5, 0.4, 0.6)
        TRICKY_BY_ALIAS.a = 0.1
        TRICKY_BY_ALIAS.b = 0.1
        set_sd_pair("a", "b", 0.2)
        local audit = audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
        })
        local ok, err = pcall(audit.report, audit)
        expect(ok).to.equal(false)
        expect(err:find("no report yet") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: load_handle failure surface ─────────────────────────────

describe("gameai_metrics.audit_matrix:run — Card handle failures", function()
    it("raises loudly when load_handle returns nil for a card_id", function()
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.1)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.1)
        set_sd_pair("a", "b", 0.1)
        local audit = audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
        })
        -- Simulate the card_id being removed from the Card store
        -- between new() and run() — the classic "rotation deleted the
        -- checkpoint" case the runner has to name loudly.
        HANDLES_BY_CARD_ID["card-b"] = nil
        local ok, err = pcall(audit.run, audit)
        expect(ok).to.equal(false)
        expect(err:find('alias "b"') ~= nil).to.equal(true)
    end)

    it("propagates load_handle raises", function()
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.1)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.1)
        set_sd_pair("a", "b", 0.1)
        local audit = audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
        })
        alc.nn.card.load_handle = function(_)
            error("engine: candle backend fell over", 0)
        end
        local ok, err = pcall(audit.run, audit)
        expect(ok).to.equal(false)
        expect(err:find("load_handle") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: pair count for larger collections ───────────────────────

describe("gameai_metrics.audit_matrix:run — pair-wise SD count", function()
    it("evaluates n*(n-1)/2 unique pairs for a 5-alias collection", function()
        reset_all()
        local aliases = { "a1", "a2", "a3", "a4", "a5" }
        for _, alias in ipairs(aliases) do
            seed_alias(alias, { 0.5, 0.4, 0.6 }, 0.2)
        end
        -- Seed every unordered pair once; the matrix must fill both
        -- halves symmetrically, so the runner is only allowed to
        -- evaluate each key once (pair_key normalises to a<=b).
        local calls = 0
        for i = 1, #aliases - 1 do
            for j = i + 1, #aliases do
                set_sd_pair(aliases[i], aliases[j], 0.1 * (i * 10 + j))
            end
        end
        local metrics = package.loaded["gameai_metrics"].metrics
        local base_sd = metrics.style_distance
        metrics.style_distance = function(ctx)
            if type(ctx.card_b) == "table" then
                calls = calls + 1
            end
            return base_sd(ctx)
        end
        local audit = audit_matrix.new({
            aliases = aliases,
            style = BASE_STYLE,
        })
        local report = audit:run()
        -- 5 * 4 / 2 = 10 unique pairs.
        expect(calls).to.equal(10)
        -- And every off-diagonal cell of the reported matrix must be
        -- reachable from either side.
        for i = 1, #aliases do
            for j = 1, #aliases do
                local sd = report.sd_matrix[aliases[i]][aliases[j]]
                expect(type(sd)).to.equal("number")
            end
        end
    end)
end)

-- ─── Specs: collection_path branch ──────────────────────────────────

describe("gameai_metrics.audit_matrix:run — collection_path", function()
    local function write_collection(path, entries)
        local body = alc.json_encode({
            schema_version = 1,
            style = BASE_STYLE,
            bands = {
                { lo = 0.10, hi = 0.30, label = "weak" },
                { lo = 0.55, hi = 0.85, label = "mid" },
                { lo = 0.85, hi = 0.98, label = "strong" },
            },
            policy = "first_writer_wins",
            entries = entries,
        })
        local f = io.open(path, "w")
        f:write(body)
        f:close()
    end

    it("reads entries[].alias / card_id / step and produces the same shape", function()
        reset_all()
        seed_alias("guardian_duel_npc_weak", { 0.20, 0.15, 0.30 }, 0.10, 0.72)
        seed_alias("guardian_duel_npc_mid", { 0.70, 0.60, 0.80 }, 0.35, 0.09)
        seed_alias("guardian_duel_npc_strong", { 0.92, 0.86, 0.98 }, 0.42, 0.24)
        set_sd_pair("guardian_duel_npc_weak", "guardian_duel_npc_mid", 0.55)
        set_sd_pair("guardian_duel_npc_weak", "guardian_duel_npc_strong", 0.60)
        set_sd_pair("guardian_duel_npc_mid", "guardian_duel_npc_strong", 0.11)
        make_handle("guardian_duel_npc")
        local path = tmp_path()
        write_collection(path, {
            {
                label = "weak",
                step = 60,
                ckpt_path = "/weak.st",
                alias = "guardian_duel_npc_weak",
                card_id = "card-guardian_duel_npc_weak",
            },
            {
                label = "mid",
                step = 180,
                ckpt_path = "/mid.st",
                alias = "guardian_duel_npc_mid",
                card_id = "card-guardian_duel_npc_mid",
            },
            {
                label = "strong",
                step = 300,
                ckpt_path = "/strong.st",
                alias = "guardian_duel_npc_strong",
                card_id = "card-guardian_duel_npc_strong",
            },
        })
        local audit = audit_matrix.new({
            collection_path = path,
            style = BASE_STYLE,
            teacher_alias = "guardian_duel_npc",
            n_games = 200,
            prompt_set_size = 4,
        })
        local report = audit:run()
        os.remove(path)
        expect(report.per_card.guardian_duel_npc_weak.step).to.equal(60)
        expect(report.per_card.guardian_duel_npc_mid.step).to.equal(180)
        expect(report.per_card.guardian_duel_npc_strong.step).to.equal(300)
        expect(report.per_card.guardian_duel_npc_mid.win_rate).to.equal(0.70)
        expect(report.sd_matrix.guardian_duel_npc_weak.guardian_duel_npc_mid).to.equal(0.55)
        expect(report.meta.collection_path).to.equal(path)
    end)

    it("raises loudly when an entry has no card_id", function()
        reset_all()
        seed_alias("weak", { 0.20, 0.15, 0.30 }, 0.10)
        seed_alias("mid", { 0.70, 0.60, 0.80 }, 0.35)
        local path = tmp_path()
        write_collection(path, {
            { label = "weak", step = 60, alias = "weak", card_id = "card-weak" },
            { label = "mid", step = 180, alias = "mid" }, -- card_id missing
        })
        local ok, err = pcall(audit_matrix.new, {
            collection_path = path,
            style = BASE_STYLE,
        })
        os.remove(path)
        expect(ok).to.equal(false)
        expect(err:find("card_id") ~= nil).to.equal(true)
    end)

    it("raises loudly when the manifest has zero entries", function()
        reset_all()
        local path = tmp_path()
        write_collection(path, {})
        local ok, err = pcall(audit_matrix.new, {
            collection_path = path,
            style = BASE_STYLE,
        })
        os.remove(path)
        expect(ok).to.equal(false)
        expect(err:find("zero entries") ~= nil).to.equal(true)
    end)

    it("raises loudly when the file cannot be opened", function()
        reset_all()
        local ok, err = pcall(audit_matrix.new, {
            collection_path = "/no/such/dir/audit-missing.json",
            style = BASE_STYLE,
        })
        expect(ok).to.equal(false)
        expect(err:find("cannot open") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: save() and _fs.ensure_parent_dir integration ────────────

describe("gameai_metrics.audit_matrix:save", function()
    local function tiny_audit()
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.1)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.1)
        set_sd_pair("a", "b", 0.2)
        return audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
            n_games = 10,
            prompt_set_size = 2,
        })
    end

    it("refuses save() before run()", function()
        local audit = tiny_audit()
        local ok, err = pcall(audit.save, audit, "/tmp/never.json")
        expect(ok).to.equal(false)
        expect(err:find("no report") ~= nil).to.equal(true)
    end)

    it("writes the report as JSON that round-trips through json_decode", function()
        local audit = tiny_audit()
        audit:run()
        local path = tmp_path()
        audit:save(path)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        local parsed = json_decode(body)
        expect(parsed.meta.style).to.equal(BASE_STYLE)
        expect(parsed.meta.n_games).to.equal(10)
        expect(parsed.per_card.a.win_rate).to.equal(0.5)
        expect(parsed.sd_matrix.a.b).to.equal(0.2)
        expect(parsed.sd_matrix.b.a).to.equal(0.2)
        expect(parsed.sd_matrix.a.a).to.equal(0.0)
    end)

    it("creates missing parent directories via _fs.ensure_parent_dir", function()
        local audit = tiny_audit()
        audit:run()
        -- Reserve a unique tmp path, remove the file, and use its
        -- basename as the root of a nested directory that does not
        -- exist yet — mirrors the harvest_collection_spec pattern.
        local base = os.tmpname()
        os.remove(base)
        local dir = base .. "-audit-parent"
        local path = dir .. "/nested/audit.json"
        audit:save(path)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        os.remove(dir .. "/nested")
        os.remove(dir)
        local parsed = json_decode(body)
        expect(parsed.per_card.a.win_rate).to.equal(0.5)
    end)

    it("raises loudly when path is not a string", function()
        local audit = tiny_audit()
        audit:run()
        local ok, err = pcall(audit.save, audit, 42)
        expect(ok).to.equal(false)
        expect(err:find("non%-empty string") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: temperature / view assembly ─────────────────────────────
--
-- Three constraints shape how these are written:
--
-- 1. The observable surface is the *merged* ctx, not the view config.
--    `shared_ctx` here is `{card, step}`, so `ctx.temperature == nil`
--    on a fire is equivalent to "the view config carried no
--    temperature key".
-- 2. `anymetric.observe` wraps every fire in `pcall` and turns a raise
--    into an error record, so a wrong assertion inside a stub would be
--    swallowed. Every case therefore captures positively: it asserts
--    the fire count first and then reads the recorded ctx.
-- 3. `style_distance` fires through two paths — the `sd_teacher` view
--    (`ctx.card_b` is the teacher alias *string*) and the pair-wise
--    matrix (`ctx.card_b` is a handle *table*). The existing stub
--    already splits on that, and so does `sd_teacher_calls` below.

--- The `sd_teacher` view fires only, i.e. the `style_distance` ctxs
--- whose `card_b` is a teacher alias string rather than a handle.
local function sd_teacher_calls()
    local out = {}
    for _, sd_ctx in ipairs(calls_to("style_distance")) do
        if type(sd_ctx.card_b) == "string" then
            out[#out + 1] = sd_ctx
        end
    end
    return out
end

describe("gameai_metrics.audit_matrix — temperature view assembly", function()
    --- Two audited Cards plus a pinned teacher, so all three view
    --- kinds fire: level, trickiness and sd_teacher (per Card) and one
    --- pair-wise style_distance for the (a, b) pair.
    local function temp_audit(temperature)
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.3, 0.21)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.3, 0.22)
        set_sd_pair("a", "b", 0.1)
        make_handle("teacher")
        return audit_matrix.new({
            aliases = { "a", "b" },
            style = BASE_STYLE,
            teacher_alias = "teacher",
            n_games = 50,
            prompt_set_size = 2,
            seed = 7,
            temperature = temperature,
        })
    end

    it("pushes the temperature into every level fire", function()
        local audit = temp_audit(0.7)
        audit:run()
        local level_ctxs = calls_to("level")
        -- One fire per audited Card; the count is asserted first so a
        -- swallowed raise cannot pass as "nothing to check".
        expect(#level_ctxs).to.equal(2)
        for _, level_ctx in ipairs(level_ctxs) do
            expect(level_ctx.temperature).to.equal(0.7)
            -- The rest of the level view config is untouched.
            expect(level_ctx.seat).to.equal("boss")
            expect(level_ctx.style).to.equal(BASE_STYLE)
            expect(level_ctx.n_games).to.equal(50)
            expect(level_ctx.seed).to.equal(7)
            expect(level_ctx.opponents[1]).to.equal("random")
        end
    end)

    it("leaves the trickiness fires without a temperature", function()
        local audit = temp_audit(0.7)
        audit:run()
        local tricky_ctxs = calls_to("trickiness")
        expect(#tricky_ctxs).to.equal(2)
        for _, tricky_ctx in ipairs(tricky_ctxs) do
            -- The trickiness adapter reads `ctx.temperature or 1.0`, so an
            -- absent key here keeps the entropy axis on the same scale
            -- every earlier audit measured it on.
            expect(tricky_ctx.temperature).to.equal(nil)
            expect(tricky_ctx.seat).to.equal("boss")
        end
    end)

    it("leaves the sd_teacher fires without a temperature", function()
        local audit = temp_audit(0.7)
        audit:run()
        local sd_ctxs = sd_teacher_calls()
        expect(#sd_ctxs).to.equal(2)
        for _, sd_ctx in ipairs(sd_ctxs) do
            expect(sd_ctx.temperature).to.equal(nil)
            expect(sd_ctx.card_b).to.equal("teacher")
        end
    end)

    it("leaves the pair-wise style_distance fires without a temperature", function()
        local audit = temp_audit(0.7)
        audit:run()
        local pair_ctxs = {}
        for _, sd_ctx in ipairs(calls_to("style_distance")) do
            if type(sd_ctx.card_b) == "table" then
                pair_ctxs[#pair_ctxs + 1] = sd_ctx
            end
        end
        -- 2 * 1 / 2 = 1 unordered pair.
        expect(#pair_ctxs).to.equal(1)
        expect(pair_ctxs[1].temperature).to.equal(nil)
    end)

    it("records meta.temperature when one was supplied", function()
        local audit = temp_audit(1.0)
        local report = audit:run()
        expect(report.meta.temperature).to.equal(1.0)
    end)

    it("omits meta.temperature and the level view key when none was supplied", function()
        local audit = temp_audit(nil)
        local report = audit:run()
        expect(report.meta.temperature).to.equal(nil)
        local level_ctxs = calls_to("level")
        expect(#level_ctxs).to.equal(2)
        for _, level_ctx in ipairs(level_ctxs) do
            -- Greedy is the absence of the key, not a `1.0` default:
            -- this is what keeps the pre-temperature report shape.
            expect(level_ctx.temperature).to.equal(nil)
        end
    end)

    it("refuses a zero temperature", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            temperature = 0,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
        expect(err:find("finite positive number") ~= nil).to.equal(true)
    end)

    it("refuses a negative temperature", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            temperature = -0.5,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)

    it("refuses a non-finite temperature", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            temperature = math.huge,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)

    it("refuses a NaN temperature", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            temperature = 0 / 0,
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)

    it("refuses a non-numeric temperature", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            temperature = "hot",
        })
        expect(ok).to.equal(false)
        expect(err:find("temperature") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: the rollout prompt set ──────────────────────────────────
--
-- The prompt set is the measurement condition of `trickiness` and
-- `style_distance`: both average over it, so what the set contains
-- decides what the numbers mean. The builder these specs pin replaced
-- one that returned `guardian_duel.new_game(seed + i).boss` `size`
-- times — sixteen separate tables that all encoded to the same single
-- prompt, because a fight opening does not vary with its seed.
--
-- **The equivalence every distinctness assertion below is written on is
-- the decode prompt `guardian_duel.encode(state, style)` under the
-- style being measured — never table identity.** Table identity is
-- exactly the basis on which the collapsed builder looks healthy (it
-- allocated sixteen tables), which is what the adversarial arm at the
-- end of this section pins.

local boss_seat = require("gameai_metrics.boss_seat")

--- The prompt every state in `prompt_set` will be read as, in order.
local function encodings_of(prompt_set, style)
    local out = {}
    for i, state in ipairs(prompt_set) do
        out[i] = duel.encode(state, style)
    end
    return out
end

--- How many *prompts* a set carries — the quantity the average has
--- terms for, as opposed to how many tables it holds.
local function distinct_encodings(prompt_set, style)
    local seen, n = {}, 0
    for _, prompt in ipairs(encodings_of(prompt_set, style)) do
        if seen[prompt] == nil then
            seen[prompt] = true
            n = n + 1
        end
    end
    return n
end

--- How many distinct *tables* a set holds. Only used by the adversarial
--- arm, to show what a spec written on the wrong equivalence would have
--- measured.
local function distinct_tables(prompt_set)
    local seen, n = {}, 0
    for _, state in ipairs(prompt_set) do
        if seen[state] == nil then
            seen[state] = true
            n = n + 1
        end
    end
    return n
end

local function mode_counts(prompt_set)
    local counts = { [0] = 0, [1] = 0 }
    for _, state in ipairs(prompt_set) do
        counts[state.mode] = counts[state.mode] + 1
    end
    return counts[0], counts[1]
end

local function as_set(list)
    local out = {}
    for _, value in ipairs(list) do
        out[value] = true
    end
    return out
end

--- The builder this section replaced, restated here so the adversarial
--- arm can run the same assertion against it. Kept literal on purpose:
--- an arm that only describes the old behaviour proves nothing.
local function legacy_build_prompt_set(seed, size)
    local out = {}
    for i = 1, size do
        out[i] = duel.new_game(seed + i).boss
    end
    return out
end

describe("gameai_metrics.audit_matrix — rollout prompt set", function()
    it("draws `size` states that are distinct decode prompts under the measured style", function()
        local set, composition = audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE)
        expect(#set).to.equal(16)
        expect(distinct_encodings(set, BASE_STYLE)).to.equal(16)
        expect(composition.distinct).to.equal(16)
    end)

    it("keys the set on the style being measured, not on a fixed basis", function()
        -- Every style has to produce a set that is distinct *under that
        -- style*: the encoded distance field is measured against the
        -- style's own shift threshold, so the prompt a state is read as
        -- depends on the basis.
        for _, style in ipairs(duel.STYLES) do
            local set = audit_matrix._build_prompt_set(20260731, 12, style)
            expect(distinct_encodings(set, style)).to.equal(12)
        end
        -- And the key really does move with the basis, so a sampler
        -- that de-duplicated on a hard-coded "guardian" would be
        -- keying on prompts the measurement never reads.
        local turtle_set = audit_matrix._build_prompt_set(20260731, 12, "turtle")
        local basis_differs = 0
        for _, state in ipairs(turtle_set) do
            if duel.encode(state, "turtle") ~= duel.encode(state, BASE_STYLE) then
                basis_differs = basis_differs + 1
            end
        end
        expect(basis_differs > 0).to.equal(true)
    end)

    it("splits the set half mode 0 / half mode 1 whatever the seed", function()
        -- The composition is what makes two runs of this audit
        -- comparable to each other: a mix that drifted with the seed
        -- would make every run a different quantity.
        for _, seed in ipairs({ 0, 7, 20260731, 20260804, 20260811 }) do
            local set, composition = audit_matrix._build_prompt_set(seed, 16, BASE_STYLE)
            local mode0, mode1 = mode_counts(set)
            expect(mode0).to.equal(8)
            expect(mode1).to.equal(8)
            expect(composition.mode0_count).to.equal(8)
            expect(composition.mode1_count).to.equal(8)
            expect(composition.distinct).to.equal(16)
        end
    end)

    it("gives an odd size's extra slot to mode 0", function()
        local set = audit_matrix._build_prompt_set(20260731, 7, BASE_STYLE)
        local mode0, mode1 = mode_counts(set)
        expect(mode0).to.equal(4)
        expect(mode1).to.equal(3)
    end)

    it("draws the same set twice for the same seed / size / style", function()
        local first =
            encodings_of(audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE), BASE_STYLE)
        local second =
            encodings_of(audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE), BASE_STYLE)
        expect(#first).to.equal(16)
        for i = 1, #first do
            expect(second[i]).to.equal(first[i])
        end
    end)

    it("draws a different set for a different seed, apart from the shared opening", function()
        local a = encodings_of(audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE), BASE_STYLE)
        local b = encodings_of(audit_matrix._build_prompt_set(20260804, 16, BASE_STYLE), BASE_STYLE)
        local in_a, in_b = as_set(a), as_set(b)
        -- Every fight opens on the same board, so this one prompt is in
        -- every set the sampler can build; "the sets differ" is a claim
        -- about the rest.
        local opening = duel.encode(duel.new_game(1).boss, BASE_STYLE)
        expect(in_a[opening]).to.equal(true)
        expect(in_b[opening]).to.equal(true)
        local only_in_a = 0
        for _, prompt in ipairs(a) do
            if prompt ~= opening and in_b[prompt] == nil then
                only_in_a = only_in_a + 1
            end
        end
        expect(only_in_a > 0).to.equal(true)
    end)

    it("takes at most four states from one rollout", function()
        local _, composition = audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE)
        -- Sixteen states at four per fight cannot come from fewer than
        -- four fights. Without the cap the first two rollouts would
        -- cover the set (a fight runs up to TURN_LIMIT turns), and the
        -- sixteen states would be two correlated trajectories.
        expect(composition.games_consumed >= 4).to.equal(true)
        expect(composition.games_consumed <= audit_matrix.ROLLOUT_GAME_CAP).to.equal(true)
    end)

    it("raises naming the short stratum when the rollout budget runs out", function()
        local ok, err = pcall(audit_matrix._build_prompt_set, 20260731, 100000, BASE_STYLE)
        expect(ok).to.equal(false)
        expect(err:find("rollout budget", 1, true) ~= nil).to.equal(true)
        expect(err:find("mode 0: ", 1, true) ~= nil).to.equal(true)
        expect(err:find("mode 1: ", 1, true) ~= nil).to.equal(true)
        expect(err:find(BASE_STYLE, 1, true) ~= nil).to.equal(true)
    end)

    it("returns engine-produced states only, never hand-built tables", function()
        local set = audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE)
        local _, mode1 = mode_counts(set)
        -- A mode-1 state cannot be opened into: it exists only after a
        -- `duel.apply` played the roll-up, so its presence is the proof
        -- the states came off a real fight rather than a literal.
        expect(mode1 > 0).to.equal(true)
        for i, state in ipairs(set) do
            boss_seat.require_state(state, string.format("spec: prompt_set[%d]", i))
            -- `block` / `thorns` are engine bookkeeping that `encode`
            -- never reads, so a hand-written state table in a spec
            -- routinely omits them; `new_game` and `apply` never do.
            expect(type(state.block)).to.equal("number")
            expect(type(state.thorns)).to.equal("number")
            expect(state.turn >= 1).to.equal(true)
            expect(state.turn <= duel.TURN_LIMIT).to.equal(true)
        end
    end)

    it("adversarial: the replaced builder fails the same distinctness assertion", function()
        local legacy = legacy_build_prompt_set(20260731, 16)
        expect(#legacy).to.equal(16)
        -- Sixteen separate tables: an assertion written on table
        -- identity would have called this set healthy and shipped the
        -- collapse. This line is here to pin that the identity basis is
        -- the wrong one, not to endorse it.
        expect(distinct_tables(legacy)).to.equal(16)
        -- On the basis the measurement actually reads — the decode
        -- prompt under the audited style — the same set is one state
        -- written down sixteen times, so the assertion arm (a) uses
        -- goes red on it.
        expect(distinct_encodings(legacy, BASE_STYLE)).to.equal(1)
        -- The same assertion, same helper, on the current builder.
        local sampled = audit_matrix._build_prompt_set(20260731, 16, BASE_STYLE)
        expect(distinct_encodings(sampled, BASE_STYLE)).to.equal(16)
    end)
end)

-- ─── Specs: prompt-set provenance on the report ─────────────────────

describe("gameai_metrics.audit_matrix — prompt set provenance", function()
    local function provenance_audit(opts)
        reset_all()
        seed_alias("a", { 0.5, 0.4, 0.6 }, 0.3)
        seed_alias("b", { 0.5, 0.4, 0.6 }, 0.3)
        set_sd_pair("a", "b", 0.1)
        local new_opts = {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            n_games = 10,
            seed = 20260731,
        }
        for key, value in pairs(opts or {}) do
            new_opts[key] = value
        end
        return audit_matrix.new(new_opts)
    end

    it("records a sampled set as rollout_v1 with its measured composition", function()
        local audit = provenance_audit({ prompt_set_size = 8 })
        local report = audit:run()
        expect(report.meta.prompt_set_source).to.equal("rollout_v1")
        expect(report.meta.prompt_set_size).to.equal(8)
        local composition = report.meta.prompt_set_composition
        expect(composition.mode0_count).to.equal(4)
        expect(composition.mode1_count).to.equal(4)
        expect(composition.distinct).to.equal(8)
        expect(composition.games_consumed >= 2).to.equal(true)
    end)

    it("hands the sampled set to the trickiness / style_distance views", function()
        local audit = provenance_audit({ prompt_set_size = 8 })
        audit:run()
        local tricky_ctxs = calls_to("trickiness")
        expect(#tricky_ctxs).to.equal(2)
        for _, tricky_ctx in ipairs(tricky_ctxs) do
            expect(#tricky_ctx.prompt_set).to.equal(8)
            expect(distinct_encodings(tricky_ctx.prompt_set, BASE_STYLE)).to.equal(8)
        end
    end)

    it("de-duplicates against the audited style, not against guardian", function()
        local audit = provenance_audit({ prompt_set_size = 8, style = "turtle" })
        audit:run()
        local tricky_ctxs = calls_to("trickiness")
        expect(#tricky_ctxs).to.equal(2)
        for _, tricky_ctx in ipairs(tricky_ctxs) do
            expect(tricky_ctx.style).to.equal("turtle")
            expect(distinct_encodings(tricky_ctx.prompt_set, "turtle")).to.equal(8)
        end
    end)

    it("uses a caller-supplied prompt_set verbatim and reports it as caller", function()
        local supplied = { boss_state(1), boss_state(2) }
        local audit = provenance_audit({ prompt_set = supplied, prompt_set_size = 8 })
        local report = audit:run()
        -- The override wins over prompt_set_size, exactly as before.
        expect(report.meta.prompt_set_size).to.equal(2)
        expect(report.meta.prompt_set_source).to.equal("caller")
        -- No composition: the runner did not draw this set and has
        -- nothing measured to report about it.
        expect(report.meta.prompt_set_composition).to.equal(nil)
        local tricky_ctxs = calls_to("trickiness")
        expect(#tricky_ctxs).to.equal(2)
        for _, tricky_ctx in ipairs(tricky_ctxs) do
            -- Verbatim means the very table, not an equal one: a caller
            -- passing `check_states` means those states in that order.
            expect(rawequal(tricky_ctx.prompt_set, supplied)).to.equal(true)
        end
    end)

    it("refuses an empty caller-supplied prompt_set", function()
        reset_all()
        make_handle("a")
        make_handle("b")
        local ok, err = pcall(audit_matrix.new, {
            aliases = { "a", "b" },
            style = BASE_STYLE,
            prompt_set = {},
        })
        expect(ok).to.equal(false)
        expect(err:find("prompt_set") ~= nil).to.equal(true)
    end)
end)
