-- guardian_duel_npc/spec/guardian_duel_npc_spec.lua
--
-- Package-level spec for the boss NPC strategy. Run with
-- `alc_pkg_test pkg="guardian_duel_npc"` after `alc_pkg_link` has
-- registered `guardian_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No Card and no model are touched: `alc.card` and `alc.nn` are
-- replaced by stubs that hand out a fake handle whose logits rank the
-- twin slam first and the defensive move second. The decode gate
-- therefore answers `t` while the boss is rolled up and `d` everywhere
-- else, which makes the stub model an exact copy of `STUB_SOURCE` and
-- lets the self-play compliance numbers be asserted instead of merely
-- parsed. It also exercises the gate itself on every mode-0 state,
-- since the raw argmax is illegal there.
--
-- What is left under test is the request surface: the distance basis
-- the states are encoded against, the alias resolution, the teacher
-- resolution (`style` versus the synthesised `policy_source`) and the
-- rejection of a field no mode reads.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("guardian_duel")

-- ─── Host stubs ─────────────────────────────────────────────────────

local VOCAB = duel.vocab()

--- Token ids: the twin slam first, the defensive move second, the rest
--- of the boss moves after them, then everything else.
local function ranked_ids()
    local order, seen = {}, {}
    for _, action in ipairs({ "t", "d", "c", "f", "v", "w" }) do
        local id = VOCAB.to_id[action]
        order[#order + 1] = { id = id }
        seen[id] = true
    end
    for id = VOCAB.size - 1, 0, -1 do
        if not seen[id] then
            order[#order + 1] = { id = id }
        end
    end
    return order
end

local ORDER = ranked_ids()

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

local SESSION = {
    next_logits = function()
        return LOGITS
    end,
}

--- Prompt of the most recent decode, as token ids. The distance basis
--- is only visible here: the stub answers the same way whatever it is
--- handed, so the prompt is what proves the right one was encoded.
local last_prompt = nil

local HANDLE = {
    generate_session = function(_, ids)
        last_prompt = ids
        return SESSION
    end,
}

--- Alias of the most recent Card lookup.
local last_alias = nil

alc.card = {
    get_by_alias = function(alias)
        last_alias = alias
        return { card_id = "stub-card-" .. alias }
    end,
}

alc.nn = {
    card = {
        load_handle = function()
            return HANDLE
        end,
    },
}

local npc = require("guardian_duel_npc")

--- The policy the stub model plays, as a persona bake would write it.
local STUB_SOURCE = [[
return function(state)
    if state.mode == 1 then
        return "t"
    end
    return "d"
end
]]

local BROKEN_SOURCE = "return function(state"
local ILLEGAL_SOURCE = "return function(state) return 'x' end"
local SLAM_SOURCE = "return function(state) return 't' end"

--- A boss state with every field present, overridden field by field.
local function boss_state(fields)
    local state = {
        cycle = 0,
        mode = 0,
        hp = duel.BOSS_MAX_HP,
        damage_since_shift = 0,
        last_player = 0,
        turn = 1,
        shifts = 0,
    }
    for key, value in pairs(fields or {}) do
        state[key] = value
    end
    return state
end

local STATE = boss_state()

--- A rolled-up boss, taken from a real fight rather than assembled.
---
--- Turn six of `A A A b b b` against the teacher: the counter is what
--- rolled the boss up in the first place, and the health it has left is
--- the damage it took to get there. Mode alone decides what the gate
--- may answer, so an invented state would test the same branch — but a
--- rolled-up boss with an empty counter is a shape no fight produces,
--- and fixtures get copied.
local ROLLED = boss_state({
    cycle = 2,
    mode = 1,
    hp = 23,
    damage_since_shift = 22,
    last_player = 3,
    turn = 6,
})

local function ask(payload, extra)
    local request = { task = alc.json_encode(payload) }
    for k, v in pairs(extra or {}) do
        request[k] = v
    end
    npc.reset_cache()
    last_prompt, last_alias = nil, nil
    return npc.run(request).result
end

--- The last prompt read back as the line the model saw.
local function prompt_text()
    local chars = {}
    for _, id in ipairs(last_prompt or {}) do
        chars[#chars + 1] = VOCAB.to_char[id]
    end
    return table.concat(chars)
end

-- ─── decide / determinism ───────────────────────────────────────────

describe("guardian_duel_npc decide", function()
    it("gates the illegal argmax down to the first legal move", function()
        expect(ask({ mode = "decide", state = STATE })).to.equal(
            "action=d legal=true raw_legal=false gated=true"
        )
    end)

    it("keeps the argmax once the state allows it", function()
        expect(ask({ mode = "decide", state = ROLLED })).to.equal(
            "action=t legal=true raw_legal=true gated=false"
        )
    end)

    it("prompts with the encoded state and the separator", function()
        ask({ mode = "decide", state = ROLLED })
        expect(prompt_text()).to.equal(duel.encode(ROLLED, "guardian") .. ">")
    end)

    it("rejects a state that is not an object", function()
        expect(function()
            ask({ mode = "decide", state = "C0M0H9D3L0T1" })
        end).to.fail()
    end)

    it("rejects a state missing a field the encoding reads", function()
        -- Defaulting `shifts` would move the distance the model reads,
        -- so the rules module names the field instead.
        local state = boss_state()
        state.shifts = nil
        expect(function()
            ask({ mode = "decide", state = state })
        end).to.fail()
    end)
end)

describe("guardian_duel_npc determinism", function()
    it("agrees across two independent sessions", function()
        expect(ask({ mode = "determinism", state = STATE })).to.equal("deterministic=true action=d")
    end)
end)

-- ─── distance basis ─────────────────────────────────────────────────

describe("guardian_duel_npc style basis", function()
    it("encodes against the style the ctx names", function()
        ask({ mode = "decide", state = STATE }, { style = "turtle" })
        expect(prompt_text()).to.equal(duel.encode(STATE, "turtle") .. ">")
    end)

    it("defaults to the teacher basis", function()
        ask({ mode = "decide", state = STATE })
        expect(prompt_text()).to.equal(duel.encode(STATE, "guardian") .. ">")
        -- The two bases really do differ on this state, so the check
        -- above is not passing on a coincidence.
        expect(duel.encode(STATE, "turtle") ~= duel.encode(STATE, "guardian")).to.equal(true)
    end)

    it("rejects a basis outside guardian_duel.STYLES", function()
        expect(function()
            ask({ mode = "decide", state = STATE }, { style = "berserker" })
        end).to.fail()
    end)

    it("rejects a basis that is not a string", function()
        expect(function()
            ask({ mode = "decide", state = STATE }, { style = 3 })
        end).to.fail()
    end)
end)

-- ─── alias resolution ───────────────────────────────────────────────

describe("guardian_duel_npc card alias", function()
    it("falls back to the bare alias", function()
        ask({ mode = "decide", state = STATE })
        expect(last_alias).to.equal("guardian_duel_npc")
    end)

    it("reads the alias from the task JSON", function()
        ask({ mode = "decide", state = STATE, card_alias = "guardian_duel_npc_turtle" })
        expect(last_alias).to.equal("guardian_duel_npc_turtle")
    end)

    it("prefers the ctx alias over the task one", function()
        ask({ mode = "decide", state = STATE, card_alias = "guardian_duel_npc_turtle" }, {
            card_alias = "guardian_duel_npc_rusher",
        })
        expect(last_alias).to.equal("guardian_duel_npc_rusher")
    end)

    it("rejects an alias that is not a string", function()
        expect(function()
            ask({ mode = "decide", state = STATE, card_alias = 7 })
        end).to.fail()
    end)

    it("rejects an empty alias", function()
        expect(function()
            ask({ mode = "decide", state = STATE, card_alias = "" })
        end).to.fail()
    end)
end)

-- ─── selfplay: style path ───────────────────────────────────────────

describe("guardian_duel_npc selfplay style", function()
    it("reports the summary fields for any known style", function()
        local text = ask({ mode = "selfplay", games = 2, seed = 5, style = "guardian" })
        local pattern = "^winrate=%d+%.%d%d illegal=%d+ style_match=%d+%.%d%d style_hits=%d+/%d+$"
        expect(text:match(pattern) ~= nil).to.equal(true)
    end)

    it("counts the gated moves as raw illegal answers", function()
        -- Every mode-0 answer of the stub is a gated one, so a run that
        -- reports zero would mean the telemetry stopped watching.
        local text = ask({ mode = "selfplay", games = 2, seed = 5, style = "guardian" })
        expect(tonumber(text:match("illegal=(%d+)")) > 0).to.equal(true)
    end)

    it("defaults the teacher to the style the ctx names", function()
        local defaulted = ask({ mode = "selfplay", games = 2, seed = 5 }, { style = "turtle" })
        local named = ask({ mode = "selfplay", games = 2, seed = 5, style = "turtle" }, {
            style = "turtle",
        })
        expect(defaulted).to.equal(named)
    end)

    it("rejects a style outside guardian_duel.STYLES", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, style = "berserker" })
        end).to.fail()
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            ask({ mode = "selfplay", games = 0, seed = 5, style = "guardian" })
        end).to.fail()
    end)
end)

-- ─── selfplay: synthesised teacher override ─────────────────────────

describe("guardian_duel_npc selfplay policy_source", function()
    it("scores a full match against the chunk the stub plays", function()
        local text = ask({ mode = "selfplay", games = 2, seed = 5, policy_source = STUB_SOURCE })
        expect(text:match("style_match=1%.00") ~= nil).to.equal(true)
    end)

    it("bypasses the style whitelist", function()
        -- A persona Card has no entry in guardian_duel.STYLES, so a
        -- request carrying both fields must follow the chunk rather than
        -- fail on the name.
        local text = ask({
            mode = "selfplay",
            games = 2,
            seed = 5,
            style = "berserker",
            policy_source = STUB_SOURCE,
        })
        expect(text:match("style_match=1%.00") ~= nil).to.equal(true)
    end)

    it("accepts the chunk on the strategy ctx", function()
        local on_ctx = ask({ mode = "selfplay", games = 2, seed = 5 }, {
            policy_source = STUB_SOURCE,
        })
        local on_task = ask({ mode = "selfplay", games = 2, seed = 5, policy_source = STUB_SOURCE })
        expect(on_ctx).to.equal(on_task)
    end)

    it("rejects a chunk that does not compile", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = BROKEN_SOURCE })
        end).to.fail()
    end)

    it("rejects a chunk that answers outside the boss moves", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = ILLEGAL_SOURCE })
        end).to.fail()
    end)

    it("rejects a chunk that slams without spikes", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = SLAM_SOURCE })
        end).to.fail()
    end)

    it("rejects a chunk reaching for a global outside the sandbox", function()
        expect(function()
            ask({
                mode = "selfplay",
                games = 2,
                seed = 5,
                policy_source = "return function(state) return os.time() end",
            })
        end).to.fail()
    end)

    it("rejects a policy_source that is not a string", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = 7 })
        end).to.fail()
    end)
end)

-- ─── entry guards ───────────────────────────────────────────────────

describe("guardian_duel_npc run", function()
    it("rejects an unknown mode", function()
        expect(function()
            ask({ mode = "rampage", state = STATE })
        end).to.fail()
    end)

    it("rejects a task that is not a JSON string", function()
        expect(function()
            npc.run({ task = { mode = "decide" } })
        end).to.fail()
    end)

    it("rejects a misspelled task field", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, stlye = "guardian" })
        end).to.fail()
    end)

    it("rejects a field another mode reads", function()
        -- `games` is a self-play field; honouring it silently in decide
        -- mode would answer a request nobody made.
        expect(function()
            ask({ mode = "decide", state = STATE, games = 2 })
        end).to.fail()
    end)
end)
