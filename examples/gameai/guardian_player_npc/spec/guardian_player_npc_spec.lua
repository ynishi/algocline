-- guardian_player_npc/spec/guardian_player_npc_spec.lua
--
-- Package-level spec for the player NPC strategy. Run with
-- `alc_pkg_test pkg="guardian_player_npc"` after `alc_pkg_link` has
-- registered `guardian_duel`, `guardian_duel_npc` and this package. The
-- `lust` globals are pre-loaded by the runner.
--
-- No Card and no model are touched: `alc.card` and `alc.nn` are
-- replaced by stubs that hand out a fake handle whose logit ranking the
-- tests set. A ranking that opens with the block makes the model an
-- exact copy of "block every turn", so the autoplay numbers can be
-- asserted instead of merely parsed; a ranking that opens with a field
-- letter makes the raw argmax a token that is not a move at all, which
-- is the case the gate exists for.
--
-- What is left under test is the request surface: the view the model is
-- prompted with, the alias resolution, the boss the autoplay is seated
-- against, and the rejection of a field no mode reads.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("guardian_duel")

-- ─── Host stubs ─────────────────────────────────────────────────────

local VOCAB = duel.player_vocab()

--- The current logit ranking, as `top` hands it out.
local ORDER = {}

--- Rank `chars` first and the rest of the alphabet after them.
local function rank(chars)
    local order, seen = {}, {}
    for _, ch in ipairs(chars) do
        local id = VOCAB.to_id[ch]
        if id == nil then
            error("spec: char " .. tostring(ch) .. " is outside the player alphabet")
        end
        order[#order + 1] = { id = id }
        seen[id] = true
    end
    for id = VOCAB.size - 1, 0, -1 do
        if not seen[id] then
            order[#order + 1] = { id = id }
        end
    end
    ORDER = order
end

--- The block first, so the stub model plays `b` on every position.
local function rank_default()
    rank({ "b", "a", "p", "A" })
end

rank_default()

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

--- Prompt of the most recent decode, as token ids.
local last_prompt = nil

local HANDLE = {
    generate_session = function(_, ids)
        last_prompt = ids
        return SESSION
    end,
}

--- Alias of the most recent Card lookup.
local last_alias = nil

--- Persona the stub hands out with a Card, keyed by alias. A persona
--- bake writes `persona.basis_style` onto its Card; the canonical
--- teacher Cards carry none, which is the empty entry here.
local PERSONA_BY_ALIAS = {}

alc.card = {
    get_by_alias = function(alias)
        last_alias = alias
        return { card_id = "stub-card-" .. alias, persona = PERSONA_BY_ALIAS[alias] }
    end,
}

alc.nn = {
    card = {
        load_handle = function()
            return HANDLE
        end,
    },
}

--- The boss Card seat, stubbed with the teacher so a fight against an
--- alias is the same fight as one against `policy_guardian`.
local last_boss_alias = nil
local last_boss_style = nil

package.preload["guardian_duel_npc"] = function()
    return {
        run = function(ctx)
            local req = alc.json_decode(ctx.task)
            last_boss_alias = ctx.card_alias
            last_boss_style = ctx.style
            local action = duel.policy_guardian(req.state)
            return {
                result = string.format("action=%s legal=true raw_legal=true gated=false", action),
            }
        end,
    }
end

local npc = require("guardian_player_npc")

--- A player view with every field present, overridden field by field.
local function player_view(fields)
    local view = {
        turn = 1,
        mode = 0,
        boss_hp = duel.BOSS_MAX_HP,
        shift_distance = duel.threshold_damage("guardian", 0),
        hp = duel.PLAYER_MAX_HP,
        weakened = false,
        exposed = false,
        spikes = false,
        intent = duel.NO_INTENT,
    }
    for key, value in pairs(fields or {}) do
        view[key] = value
    end
    return view
end

local VIEW = player_view()

local function ask(payload, extra)
    local request = { task = alc.json_encode(payload) }
    for k, v in pairs(extra or {}) do
        request[k] = v
    end
    npc.reset_cache()
    rank_default()
    last_prompt, last_alias = nil, nil
    last_boss_alias, last_boss_style = nil, nil
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

describe("guardian_player_npc decide", function()
    it("answers the top-ranked move", function()
        expect(ask({ mode = "decide", view = VIEW })).to.equal(
            "action=b legal=true raw_legal=true gated=false"
        )
    end)

    it("gates an argmax that is not a move at all", function()
        -- All four moves are always legal, so the gate is not there to
        -- remove an option: it is there for the turn the model answers
        -- with a field letter.
        npc.reset_cache()
        rank({ "M", "p", "b", "a", "A" })
        local text = npc.run({ task = alc.json_encode({ mode = "decide", view = VIEW }) }).result
        expect(text).to.equal("action=p legal=true raw_legal=false gated=true")
    end)

    it("prompts with the encoded view and the separator", function()
        ask({ mode = "decide", view = VIEW })
        expect(prompt_text()).to.equal(duel.player_encode(VIEW) .. ">")
    end)

    it("prompts over the player alphabet", function()
        -- The two alphabets are different id spaces, so a prompt built
        -- with the boss one would decode to a different line entirely.
        ask({ mode = "decide", view = VIEW })
        local expected = duel.player_to_ids(duel.player_encode(VIEW) .. ">")
        expect(#last_prompt).to.equal(duel.PLAYER_ENCODED_LEN + 1)
        for i, id in ipairs(expected) do
            expect(last_prompt[i]).to.equal(id)
        end
    end)

    it("rejects a view that is not an object", function()
        expect(function()
            ask({ mode = "decide", view = "M0H9D3Y9T1S0-" })
        end).to.fail()
    end)

    it("rejects a view missing a field the encoding reads", function()
        -- Defaulting a flag would flip a bit the caller never set.
        local view = player_view()
        view.spikes = nil
        expect(function()
            ask({ mode = "decide", view = view })
        end).to.fail()
    end)

    it("rejects a view that carries no intent", function()
        -- A substituted placeholder would tell the model the board
        -- showed nothing on a turn the player had bought a look at.
        local view = player_view()
        view.intent = nil
        expect(function()
            ask({ mode = "decide", view = view })
        end).to.fail()
    end)

    it("prompts with the answer a poke revealed", function()
        ask({ mode = "decide", view = player_view({ mode = 1, intent = "t" }) })
        expect(prompt_text()).to.equal("M1H9D3Y9T1S0t>")
    end)
end)

describe("guardian_player_npc determinism", function()
    it("agrees across two independent sessions", function()
        expect(ask({ mode = "determinism", view = VIEW })).to.equal("deterministic=true action=b")
    end)
end)

-- ─── alias resolution ───────────────────────────────────────────────

describe("guardian_player_npc card alias", function()
    it("falls back to the bare alias", function()
        ask({ mode = "decide", view = VIEW })
        expect(last_alias).to.equal("guardian_player_npc")
    end)

    it("reads the alias from the task JSON", function()
        ask({ mode = "decide", view = VIEW, card_alias = "guardian_player_npc_ytk" })
        expect(last_alias).to.equal("guardian_player_npc_ytk")
    end)

    it("prefers the ctx alias over the task one", function()
        ask({ mode = "decide", view = VIEW, card_alias = "guardian_player_npc_ytk" }, {
            card_alias = "guardian_player_npc_other",
        })
        expect(last_alias).to.equal("guardian_player_npc_other")
    end)

    it("rejects an alias that is not a string", function()
        expect(function()
            ask({ mode = "decide", view = VIEW, card_alias = 7 })
        end).to.fail()
    end)

    it("rejects an empty alias", function()
        expect(function()
            ask({ mode = "decide", view = VIEW, card_alias = "" })
        end).to.fail()
    end)
end)

-- ─── autoplay ───────────────────────────────────────────────────────

describe("guardian_player_npc autoplay", function()
    it("plays the whole fight and reports how it went", function()
        -- Nine turns of block against the teacher: the boss takes
        -- nothing, so it never staggers, and its two fierce swings get
        -- through for three each. The player ends a health bucket down
        -- and loses on the comparison.
        expect(ask({ mode = "autoplay", games = 1, seed = 5, boss_style = "guardian" })).to.equal(
            "winrate=0.00 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0"
        )
    end)

    it("plays one fight when the caller asks for no count", function()
        -- Both seats decode greedily and the opening is fixed, so a
        -- batch is a repeat rather than a sample.
        expect(ask({ mode = "autoplay", seed = 5, boss_style = "guardian" })).to.equal(
            "winrate=0.00 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0"
        )
    end)

    it("repeats the same fight for a larger batch", function()
        expect(ask({ mode = "autoplay", games = 3, seed = 5, boss_style = "guardian" })).to.equal(
            "winrate=0.00 raw_legal=1.00 moves=27 a=0 A=0 b=27 p=0"
        )
    end)

    it("seats the boss the ctx names", function()
        -- The defensive variant lands nothing at all through the block,
        -- so the same nine turns end level.
        expect(ask({ mode = "autoplay", games = 1, seed = 5 }, { boss_style = "turtle" })).to.equal(
            "winrate=0.50 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0"
        )
    end)

    it("lets the task override the boss on the ctx", function()
        local overridden = ask({
            mode = "autoplay",
            games = 1,
            seed = 5,
            boss_style = "turtle",
        }, { boss_style = "guardian" })
        expect(overridden).to.equal("winrate=0.50 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0")
    end)

    it("shows the model the answer its own poke bought", function()
        -- A model that pokes every turn buys a look at every turn but
        -- the first, so the view it is asked about has to carry the
        -- boss answer: autoplaying it on the placeholder would be
        -- asking a question the log it was baked from never contained.
        npc.reset_cache()
        rank({ "p", "b", "a", "A" })
        local text = npc.run({
            task = alc.json_encode({ mode = "autoplay", games = 1, seed = 5 }),
        }).result
        expect(text).to.equal("winrate=0.00 raw_legal=1.00 moves=9 a=0 A=0 b=0 p=9")
        -- The last prompt is the ninth turn, bought by the eighth poke.
        -- Nothing the pokes did was enough to stagger the teacher, so it
        -- is still walking its four-move cycle and turn nine is back at
        -- the head of it: the charge.
        local intent = prompt_text():sub(duel.PLAYER_ENCODED_LEN, duel.PLAYER_ENCODED_LEN)
        expect(intent).to.equal("c")
    end)

    it("counts a gated answer against the raw legality rate", function()
        npc.reset_cache()
        rank({ "M", "b", "a", "p", "A" })
        local text = npc.run({
            task = alc.json_encode({ mode = "autoplay", games = 1, boss_style = "guardian" }),
        }).result
        expect(text).to.equal("winrate=0.00 raw_legal=0.00 moves=9 a=0 A=0 b=9 p=0")
    end)

    it("seats a boss Card when the task names one", function()
        local text = ask({
            mode = "autoplay",
            games = 1,
            seed = 5,
            boss_card_alias = "guardian_duel_npc_stalker",
            boss_style = "guardian",
        })
        expect(text).to.equal("winrate=0.00 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0")
        expect(last_boss_alias).to.equal("guardian_duel_npc_stalker")
        -- The boss Card is decoded under the same basis the player views
        -- are built on, or the two seats read different distances.
        expect(last_boss_style).to.equal("guardian")
    end)

    it("reads the basis off a seated boss Card", function()
        -- A persona bake records the basis its corpus was encoded
        -- against on the Card, so a seated boss knows its own threshold
        -- and the caller does not have to repeat it. The fight is the
        -- same one as above — the stub boss answers with the teacher
        -- either way — and what the basis moves is the `D` field of
        -- every view the model is asked about.
        PERSONA_BY_ALIAS["guardian_duel_npc_stalker"] = { basis_style = "turtle" }
        local text = ask({
            mode = "autoplay",
            games = 1,
            seed = 5,
            boss_card_alias = "guardian_duel_npc_stalker",
        })
        expect(text).to.equal("winrate=0.00 raw_legal=1.00 moves=9 a=0 A=0 b=9 p=0")
        expect(last_boss_style).to.equal("turtle")
        PERSONA_BY_ALIAS["guardian_duel_npc_stalker"] = nil
    end)

    it("lets boss_style override the basis on the Card", function()
        -- The Card names a basis only when nobody else did: an explicit
        -- `boss_style` is the caller measuring the fights against a
        -- threshold of their own choosing.
        PERSONA_BY_ALIAS["guardian_duel_npc_stalker"] = { basis_style = "turtle" }
        ask({
            mode = "autoplay",
            games = 1,
            seed = 5,
            boss_card_alias = "guardian_duel_npc_stalker",
            boss_style = "guardian",
        })
        expect(last_boss_style).to.equal("guardian")
        PERSONA_BY_ALIAS["guardian_duel_npc_stalker"] = nil
    end)

    it("rejects a boss Card that records no basis when no boss_style is named", function()
        -- The canonical teacher Cards carry no persona basis. Falling
        -- back to the teacher default would measure every view against
        -- a boss that is not seated, so the ask is loud instead.
        expect(function()
            ask({
                mode = "autoplay",
                games = 1,
                seed = 5,
                boss_card_alias = "guardian_duel_npc_plain",
            })
        end).to.fail()
    end)

    it("rejects a Card basis outside guardian_duel.STYLES", function()
        -- A basis read off a Card is checked like one read off the
        -- request: a threshold that does not exist cannot be encoded.
        PERSONA_BY_ALIAS["guardian_duel_npc_bogus"] = { basis_style = "berserker" }
        expect(function()
            ask({
                mode = "autoplay",
                games = 1,
                seed = 5,
                boss_card_alias = "guardian_duel_npc_bogus",
            })
        end).to.fail()
        PERSONA_BY_ALIAS["guardian_duel_npc_bogus"] = nil
    end)

    it("rejects a boss Card alias that is not a string", function()
        expect(function()
            ask({ mode = "autoplay", games = 1, boss_card_alias = 7 })
        end).to.fail()
    end)

    it("rejects a boss outside guardian_duel.STYLES", function()
        expect(function()
            ask({ mode = "autoplay", games = 1, boss_style = "berserker" })
        end).to.fail()
        expect(function()
            ask({ mode = "autoplay", games = 1 }, { boss_style = "berserker" })
        end).to.fail()
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            ask({ mode = "autoplay", games = 0, boss_style = "guardian" })
        end).to.fail()
    end)
end)

-- ─── entry guards ───────────────────────────────────────────────────

describe("guardian_player_npc run", function()
    it("rejects an unknown mode", function()
        expect(function()
            ask({ mode = "rampage", view = VIEW })
        end).to.fail()
    end)

    it("rejects a task that is not a JSON string", function()
        expect(function()
            npc.run({ task = { mode = "decide" } })
        end).to.fail()
    end)

    it("rejects a misspelled task field", function()
        expect(function()
            ask({ mode = "autoplay", games = 1, boss_stlye = "guardian" })
        end).to.fail()
    end)

    it("rejects a field another mode reads", function()
        -- `games` is an autoplay field; honouring it silently in decide
        -- mode would answer a request nobody made.
        expect(function()
            ask({ mode = "decide", view = VIEW, games = 2 })
        end).to.fail()
        expect(function()
            ask({ mode = "autoplay", games = 1, view = VIEW })
        end).to.fail()
    end)
end)
