-- spec/guardian_duel_noisy_equivalence_spec.lua
--
-- The noisy boss decode exists twice on purpose: once in
-- `guardian_duel_npc` (the shipping surface, where `decide_noisy` is a
-- mode of the strategy) and once inside `gameai_metrics.level` (the
-- measurement surface, which composes `boss_seat.legal` / `.encode` with
-- the same sampler chain). The duplication is the convention this repo
-- documents in `gameai_metrics/boss_seat.lua:12-33`: the NPC's public
-- API is `run` / `reset_cache`, so the measurement side cannot require
-- the decode, and widening the NPC contract for a measurement shortcut
-- is refused.
--
-- The cost of that convention is drift, and drift here is not a style
-- problem: the shipped tiers are (Card, decode condition) pairs whose
-- numbers were measured through `level`. If the two decodes stopped
-- agreeing, the boss that ships would not be the boss that was measured,
-- and nothing in either package would say so.
--
-- This spec is that guard. One fight is played through the real
-- `gameai_metrics.level` on the boss seat at a temperature, its
-- per-move transcript is replayed position by position through
-- `guardian_duel_npc.decide_noisy`, and the two are compared on the four
-- things a decode is:
--
--   (a) the prompt the model is asked (encode + separator, one basis),
--   (b) the allow list the draw is masked to (the state's legal ids),
--   (c) the chain the draw goes through (temperature, per-decision seed),
--   (d) the id that came out.
--
-- Nothing here loads a model: the handle is a stub whose logits are
-- planted, and the sampler namespaces are the chain-capturing stubs the
-- two package specs already use, so a chain reused across decisions is a
-- loud failure exactly as it is against the real bridge.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/guardian_duel_noisy_equivalence_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })

local describe, it, expect = lust.describe, lust.it, lust.expect

alc = alc or {}

local duel = require("guardian_duel")

local VOCAB = duel.vocab()

-- ─── JSON (the NPC entry takes its task as a string) ────────────────

local function encode(value)
    local kind = type(value)
    if value == nil then
        return "null"
    end
    if kind == "boolean" then
        return tostring(value)
    end
    if kind == "number" then
        if value == math.floor(value) and math.abs(value) < 1e15 then
            return string.format("%d", value)
        end
        return string.format("%.17g", value)
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

-- ─── Host stubs ─────────────────────────────────────────────────────

--- Logit ranking: the twin slam first, so the raw argmax is illegal on
--- every mode-0 state. It changes nothing about the draw (the mask does
--- that) and everything about `raw_legal`, which is what makes the stub
--- a model that is still worth gating.
local ORDER = {}
do
    local seen = {}
    for _, action in ipairs({ "t", "d", "c", "f", "v", "w" }) do
        local id = VOCAB.to_id[action]
        ORDER[#ORDER + 1] = { id = id }
        seen[id] = true
    end
    for id = VOCAB.size - 1, 0, -1 do
        if not seen[id] then
            ORDER[#ORDER + 1] = { id = id }
        end
    end
end

local LOGITS = {
    argmax = function()
        return ORDER[1].id
    end,
    vocab = function()
        return VOCAB.size
    end,
    top = function(_, n)
        local out = {}
        for i = 1, math.min(n, #ORDER) do
            out[i] = ORDER[i]
        end
        return out
    end,
}

--- Prompts of every decode since the last reset, oldest first.
local PROMPTS = {}

local HANDLE = {
    generate_session = function(_, ids)
        PROMPTS[#PROMPTS + 1] = ids
        return {
            next_logits = function()
                return LOGITS
            end,
        }
    end,
}

alc.card = {
    get_by_alias = function(alias)
        return { card_id = "stub-card-" .. alias }
    end,
}

-- ─── Sampler stubs (the two package specs' harness) ─────────────────

local CHAINS = {}

local function draw(seed, allow)
    local mixed = (seed * 2654435761 + 1013904223) % 4294967296
    return allow[math.floor(mixed / 65536) % #allow + 1]
end

local function spendable(kind)
    return { kind = kind, spent = false }
end

local function spend(handle, kind)
    if handle.spent then
        error(
            "spec: this "
                .. kind
                .. " was moved into a constrained sampler; build a fresh one instead"
        )
    end
    handle.spent = true
end

alc.nn = {
    card = {
        load_handle = function()
            return HANDLE
        end,
    },
    sampler = {
        temperature = function(temperature, seed)
            local handle = spendable("sampler")
            handle.temperature = temperature
            handle.seed = seed
            return handle
        end,
        constrained = function(inner, constraint)
            spend(inner, "sampler")
            spend(constraint, "constraint")
            local chain = {
                temperature = inner.temperature,
                seed = inner.seed,
                allow = constraint.ids,
                drawn = nil,
            }
            CHAINS[#CHAINS + 1] = chain
            return {
                sample = function(_, logits)
                    if logits ~= LOGITS then
                        error("spec: a sampler must be handed the session's own logits row")
                    end
                    chain.drawn = draw(chain.seed, chain.allow)
                    return chain.drawn
                end,
            }
        end,
    },
    constraint = {
        allow_list = function(ids)
            if type(ids) ~= "table" or #ids == 0 then
                error("spec: an empty allow list must be rejected at construction")
            end
            local handle = spendable("constraint")
            local copy = {}
            for i, id in ipairs(ids) do
                copy[i] = id
            end
            handle.ids = copy
            return handle
        end,
    },
}

--- Deterministic stand-in for the host RNG bridge, which the `"random"`
--- player opponent of the measured fight draws its moves from. Only its
--- reproducibility matters here: the fight has to be the same fight
--- twice, once measured and once replayed.
alc.math = {
    rng_create = function(seed)
        return { state = math.floor(seed or 0) % 2147483647 }
    end,
    rng_int = function(rng, lo, hi)
        rng.state = (rng.state * 1103515245 + 12345) % 2147483648
        return lo + math.floor(rng.state / 65536) % (hi - lo + 1)
    end,
}

local npc = require("guardian_duel_npc")
local level = require("gameai_metrics.level")
local boss_seat = require("gameai_metrics.boss_seat")

-- ─── The measured fight ─────────────────────────────────────────────

local SEED = 5
local TEMPERATURE = 0.8
local STYLE = "guardian"

local function reset()
    PROMPTS, CHAINS = {}, {}
    npc.reset_cache()
end

--- Play one fight through the real `level` on the boss seat and hand
--- back its transcript together with everything the decode stubs saw.
---
--- `per_move` is what makes the replay possible: the transcript carries
--- the two moves of every turn, so the positions the boss chose from can
--- be rebuilt with `guardian_duel.apply` rather than guessed at.
local function measured_fight()
    reset()
    local report = level(HANDLE, "random", 1, SEED, {
        seat = "boss",
        style = STYLE,
        temperature = TEMPERATURE,
        per_game = true,
        per_move = true,
    })
    return {
        moves = report.per_opponent.random.games[1].moves,
        prompts = PROMPTS,
        chains = CHAINS,
    }
end

--- Replay the same fight through the NPC entry, one `decide_noisy` per
--- turn, deriving the seed the way `level` derives it: a run-local
--- counter over the base seed, incremented once per draw. On the boss
--- seat against the scripted player, the boss decision is the only draw
--- of a turn, so turn `i` draws under `SEED + i - 1`.
local function replayed_fight(moves)
    reset()
    local state = duel.new_game(SEED + 1)
    local actions = {}
    for i, move in ipairs(moves) do
        local text = npc.run({
            task = encode({
                mode = "decide_noisy",
                state = state.boss,
                seed = SEED + i - 1,
                temperature = TEMPERATURE,
            }),
            style = STYLE,
        }).result
        actions[i] = text:match("action=(%a)")
        state = duel.apply(state, move.player_action, move.boss_action)
    end
    return { actions = actions, prompts = PROMPTS, chains = CHAINS }
end

local MEASURED = measured_fight()
local REPLAYED = replayed_fight(MEASURED.moves)

describe("guardian_duel_npc.decide_noisy vs gameai_metrics.level boss seat", function()
    it("decides once per turn on both sides", function()
        -- The comparison below is index by index, so a side that decided
        -- a different number of times would compare the wrong pairs.
        local turns = #MEASURED.moves
        expect(turns > 1).to.equal(true)
        expect(#MEASURED.chains).to.equal(turns)
        expect(#MEASURED.prompts).to.equal(turns)
        expect(#REPLAYED.chains).to.equal(turns)
        expect(#REPLAYED.prompts).to.equal(turns)
    end)

    it("asks the model the same prompt", function()
        -- Same encoding, same basis, same separator — as token ids,
        -- since that is what the model is actually handed.
        for i, measured in ipairs(MEASURED.prompts) do
            local replayed = REPLAYED.prompts[i]
            expect(#replayed).to.equal(#measured)
            for k, id in ipairs(measured) do
                expect(replayed[k]).to.equal(id)
            end
        end
    end)

    it("masks the draw to the same legal ids", function()
        -- The boss legal set moves with the state, so this is the
        -- assertion that would break first if one side hoisted the mask
        -- out of the loop or ordered it differently.
        for i, measured in ipairs(MEASURED.chains) do
            local replayed = REPLAYED.chains[i]
            expect(#replayed.allow).to.equal(#measured.allow)
            for k, id in ipairs(measured.allow) do
                expect(replayed.allow[k]).to.equal(id)
            end
        end
    end)

    it("builds the same chain for the same decision", function()
        for i, measured in ipairs(MEASURED.chains) do
            local replayed = REPLAYED.chains[i]
            expect(replayed.temperature).to.equal(measured.temperature)
            expect(replayed.seed).to.equal(measured.seed)
        end
    end)

    it("draws the same token, and it is the move that was played", function()
        for i, measured in ipairs(MEASURED.chains) do
            expect(REPLAYED.chains[i].drawn).to.equal(measured.drawn)
            expect(REPLAYED.actions[i]).to.equal(MEASURED.moves[i].boss_action)
        end
    end)

    it("agrees with boss_seat on the legal set of every position", function()
        -- The third account of the same rule: the mask both sides built
        -- is the one `boss_seat.legal` computes for the position, in its
        -- order. Rebuilt from the transcript, so it is the fight's own
        -- positions rather than a hand-written state.
        local state = duel.new_game(SEED + 1)
        for i, move in ipairs(MEASURED.moves) do
            local legal = boss_seat.legal(state.boss)
            local allow = MEASURED.chains[i].allow
            expect(#allow).to.equal(#legal.ids)
            for k, id in ipairs(legal.ids) do
                expect(allow[k]).to.equal(id)
            end
            expect(legal.by_id[MEASURED.chains[i].drawn]).to.equal(move.boss_action)
            state = duel.apply(state, move.player_action, move.boss_action)
        end
    end)

    it("covers a position the mask actually changes on", function()
        -- Five moves while the boss is unrolled, six once the slam is
        -- available. A fight that never left mode 0 would pass every
        -- assertion above without ever testing the state-dependent half
        -- of the mask, so the fight itself is checked for the mode it
        -- reached.
        local sizes = {}
        for _, chain in ipairs(MEASURED.chains) do
            sizes[#chain.allow] = true
        end
        expect(sizes[5]).to.equal(true)
        expect(sizes[6]).to.equal(true)
    end)
end)
