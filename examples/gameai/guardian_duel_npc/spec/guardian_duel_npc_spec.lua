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

-- ─── Sampler stubs ──────────────────────────────────────────────────
--
-- Ported from `guardian_player_npc/spec`, with the one difference that
-- matters on this seat: the mask is not a module constant, so the specs
-- read the allow list back per state rather than against a fixed set of
-- four ids.
--
-- There is no RNG of the real kind, but the *contract* of the bridge is
-- kept: a composition consumes both of its arguments and a spent handle
-- is a loud error, so a chain reused across decisions fails here exactly
-- as it would against `alc.nn.sampler.constrained`.

--- Token ids of the moves a state allows, in `legal_actions` order —
--- the shape `alc.nn.constraint.allow_list` takes and the order the
--- module builds it in.
local function legal_ids(state)
    local ids = {}
    for _, action in ipairs(duel.legal_actions(state)) do
        ids[#ids + 1] = VOCAB.to_id[action]
    end
    return ids
end

--- Every chain built since the last reset, oldest first. Each entry is
--- `{ temperature, seed, allow, draws }`.
local CHAINS = {}

--- The stand-in for a temperature draw.
---
--- Not a model of the real sampler — there is no row of logits behind
--- it — but a function of the seed and the allow list alone, which is
--- the property the specs are about: a replay that derives the same
--- seed draws the same move, and neighbouring seeds do not all land on
--- one id.
local function draw(seed, allow)
    local mixed = (seed * 2654435761 + 1013904223) % 4294967296
    return allow[math.floor(mixed / 65536) % #allow + 1]
end

--- A handle that can be moved out of, like the bridge's.
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
                draws = 0,
            }
            CHAINS[#CHAINS + 1] = chain
            return {
                sample = function(self, logits)
                    if logits ~= LOGITS then
                        error("spec: a sampler must be handed the session's own logits row")
                    end
                    self.chain.draws = self.chain.draws + 1
                    return draw(chain.seed, chain.allow)
                end,
                chain = chain,
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

--- The sampler namespaces, kept aside so a case can take them away and
--- put them back (a build with `alc.nn.card` and no draw surface).
local SAMPLER_NS, CONSTRAINT_NS = alc.nn.sampler, alc.nn.constraint

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
    CHAINS = {}
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

-- ─── decide_noisy ───────────────────────────────────────────────────

--- The move the stub draws for `seed` at `state`, as the summary spells
--- it. The mask is read off the state rather than off a constant: that
--- is the whole difference between this seat and the player one.
local function drawn(seed, state)
    return VOCAB.to_char[draw(seed, legal_ids(state))]
end

--- Run one case with `duel.legal_actions` answering `actions`.
---
--- The rules module is shared with the package under test, so replacing
--- the function here is what a state with an unusual legal set looks
--- like from the inside. Restored on the way out, failure or not.
local function with_legal_actions(actions, body)
    local real = duel.legal_actions
    duel.legal_actions = function()
        local out = {}
        for i, action in ipairs(actions) do
            out[i] = action
        end
        return out
    end
    local ok, err = pcall(body)
    duel.legal_actions = real
    if not ok then
        error(err, 0)
    end
end

describe("guardian_duel_npc decide_noisy", function()
    it("draws a legal move and reports the draw", function()
        -- The whole line, not a match: the shape is what a caller
        -- parses, and `noisy=true` is what tells a sweep the number in
        -- front of it came from a draw rather than from the scan.
        expect(ask({ mode = "decide_noisy", state = ROLLED, seed = 7 })).to.equal(
            string.format(
                "action=%s legal=true raw_legal=true noisy=true temperature=1 seed=7",
                drawn(7, ROLLED)
            )
        )
    end)

    it("masks the moves the state allows and nothing else", function()
        -- Legality is a property of the mask, not of a check after the
        -- draw. On this seat the mask moves with the state: five moves
        -- while the boss is unrolled, six once the slam is available.
        ask({ mode = "decide_noisy", state = STATE, seed = 7 })
        expect(#CHAINS).to.equal(1)
        local mode0 = legal_ids(STATE)
        expect(#CHAINS[1].allow).to.equal(5)
        for i, id in ipairs(mode0) do
            expect(CHAINS[1].allow[i]).to.equal(id)
        end

        ask({ mode = "decide_noisy", state = ROLLED, seed = 7 })
        local mode1 = legal_ids(ROLLED)
        expect(#CHAINS[1].allow).to.equal(6)
        for i, id in ipairs(mode1) do
            expect(CHAINS[1].allow[i]).to.equal(id)
        end
    end)

    it("keeps the twin slam out of the mask while it is illegal", function()
        ask({ mode = "decide_noisy", state = STATE, seed = 7 })
        local slam = VOCAB.to_id["t"]
        for _, id in ipairs(CHAINS[1].allow) do
            expect(id ~= slam).to.equal(true)
        end
    end)

    it("reports the raw argmax legality off the ungated row", function()
        -- The draw is masked, so it cannot say whether the model was
        -- still answering the question. The argmax still can — and on
        -- this seat it also catches a model reaching for the slam on a
        -- state that forbids it, which is what the stub ranking does.
        expect(ask({ mode = "decide_noisy", state = STATE, seed = 7 })).to.equal(
            string.format(
                "action=%s legal=true raw_legal=false noisy=true temperature=1 seed=7",
                drawn(7, STATE)
            )
        )
    end)

    it("draws the same move from the same seed", function()
        local first = ask({ mode = "decide_noisy", state = ROLLED, seed = 11 })
        local second = ask({ mode = "decide_noisy", state = ROLLED, seed = 11 })
        expect(first).to.equal(second)
    end)

    it("does not answer every seed with one move", function()
        -- The point of the mode. A sampler that collapsed onto a single
        -- id would still be legal and still be useless.
        local seen, distinct = {}, 0
        for seed = 1, 12 do
            local action =
                ask({ mode = "decide_noisy", state = ROLLED, seed = seed }):match("action=(%a)")
            if not seen[action] then
                seen[action] = true
                distinct = distinct + 1
            end
        end
        expect(distinct > 1).to.equal(true)
    end)

    it("prompts with the encoded state and the separator", function()
        -- The basis reaches the draw exactly as it reaches the scan: a
        -- noisy decode under another threshold would be a different
        -- question asked of the same model.
        ask({ mode = "decide_noisy", state = ROLLED, seed = 7 }, { style = "turtle" })
        expect(prompt_text()).to.equal(duel.encode(ROLLED, "turtle") .. ">")
    end)

    it("carries the requested temperature into the sampler", function()
        local text = ask({ mode = "decide_noisy", state = ROLLED, seed = 7, temperature = 0.75 })
        expect(CHAINS[1].temperature).to.equal(0.75)
        expect(text:match("temperature=([%d%.]+)")).to.equal("0.75")
    end)

    it("defaults the temperature to 1.0", function()
        ask({ mode = "decide_noisy", state = ROLLED, seed = 7 })
        expect(CHAINS[1].temperature).to.equal(1.0)
    end)

    it("passes the seed the caller derived", function()
        ask({ mode = "decide_noisy", state = ROLLED, seed = 42 })
        expect(CHAINS[1].seed).to.equal(42)
    end)

    it("floors a fractional seed and echoes the one it drew under", function()
        -- The echoed seed is what a replay is derived from, so it has to
        -- be the integer the sampler saw rather than the number the
        -- caller happened to send.
        local text = ask({ mode = "decide_noisy", state = ROLLED, seed = 7.9 })
        expect(CHAINS[1].seed).to.equal(7)
        expect(text).to.equal(
            string.format(
                "action=%s legal=true raw_legal=true noisy=true temperature=1 seed=7",
                drawn(7, ROLLED)
            )
        )
    end)

    it("draws the only move a one-move state allows", function()
        -- A mask of one is still a mask: the draw has nowhere else to
        -- land, and the summary still reports it as a draw.
        with_legal_actions({ "d" }, function()
            local text = ask({ mode = "decide_noisy", state = STATE, seed = 7 })
            expect(#CHAINS[1].allow).to.equal(1)
            expect(CHAINS[1].allow[1]).to.equal(VOCAB.to_id["d"])
            expect(text).to.equal(
                "action=d legal=true raw_legal=false noisy=true temperature=1 seed=7"
            )
        end)
    end)

    it("refuses a state with no legal move before it builds a chain", function()
        -- Order matters: an empty allow list is a loud failure at the
        -- constraint, and reaching it would report a sampler problem for
        -- what is a rules one. The legal set is read first.
        with_legal_actions({}, function()
            expect(function()
                ask({ mode = "decide_noisy", state = STATE, seed = 7 })
            end).to.fail()
            expect(#CHAINS).to.equal(0)
        end)
    end)

    it("refuses the draw on a build with no sampler surface", function()
        -- A build with `alc.nn.card` but no `allow_list` still answers
        -- every greedy mode, so the greedy caller must not be turned
        -- away by a surface only the draw needs.
        alc.nn.sampler, alc.nn.constraint = nil, nil
        local ok = pcall(ask, { mode = "decide_noisy", state = ROLLED, seed = 7 })
        local greedy = ask({ mode = "decide", state = ROLLED })
        alc.nn.sampler, alc.nn.constraint = SAMPLER_NS, CONSTRAINT_NS
        expect(ok).to.equal(false)
        expect(greedy).to.equal("action=t legal=true raw_legal=true gated=false")
    end)

    it("requires a seed", function()
        -- A default seed would make the draw depend on nothing the
        -- caller can write down, which is the reproducibility the
        -- sampler carries its own RNG for.
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED })
        end).to.fail()
    end)

    it("rejects a seed that is not a non-negative number", function()
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED, seed = "7" })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED, seed = -1 })
        end).to.fail()
    end)

    it("rejects a temperature that is not a finite positive number", function()
        -- Zero is a caller who means greedy, and greedy has a mode.
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED, seed = 7, temperature = 0 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED, seed = 7, temperature = -0.5 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide_noisy", state = ROLLED, seed = 7, temperature = "hot" })
        end).to.fail()
    end)

    it("rejects a state that is not an object", function()
        expect(function()
            ask({ mode = "decide_noisy", state = "C0M0H9D3L0T1", seed = 7 })
        end).to.fail()
    end)

    it("leaves the greedy modes alone", function()
        -- The two paths share the prompt and the argmax, nothing else:
        -- a greedy request must not reach the sampler at all, or the
        -- determinism the scenarios fence would depend on a seed.
        expect(ask({ mode = "decide", state = STATE })).to.equal(
            "action=d legal=true raw_legal=false gated=true"
        )
        expect(#CHAINS).to.equal(0)
        expect(ask({ mode = "determinism", state = STATE })).to.equal("deterministic=true action=d")
        expect(#CHAINS).to.equal(0)
        local text = ask({ mode = "selfplay", games = 2, seed = 5, style = "guardian" })
        expect(text:match("^winrate=") ~= nil).to.equal(true)
        expect(#CHAINS).to.equal(0)
    end)

    it("rejects the noisy fields on the greedy decode", function()
        expect(function()
            ask({ mode = "decide", state = STATE, seed = 7 })
        end).to.fail()
        expect(function()
            ask({ mode = "decide", state = STATE, temperature = 0.8 })
        end).to.fail()
    end)

    it("rejects a field another mode reads", function()
        expect(function()
            ask({ mode = "decide_noisy", state = STATE, seed = 7, games = 2 })
        end).to.fail()
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
