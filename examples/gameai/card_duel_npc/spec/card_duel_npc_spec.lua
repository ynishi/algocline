-- card_duel_npc/spec/card_duel_npc_spec.lua
--
-- Package-level spec for the NPC strategy. Run with
-- `alc_pkg_test pkg="card_duel_npc"` after `alc_pkg_link` has
-- registered `card_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No Card and no model are touched: `alc.card` and `alc.nn` are
-- replaced by stubs that hand out a fake handle whose logits rank the
-- rank tokens from nine down to one. The decode gate therefore always
-- lands on the highest legal rank, which makes the stub model an exact
-- `card_duel.policy_bold` player and lets the self-play compliance
-- numbers be asserted instead of merely parsed.
--
-- What is left under test is the request surface: the teacher
-- resolution (`style` versus the synthesised `policy_source`), the
-- guards around a bad `policy_source`, and the modes themselves.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("card_duel")

-- ─── Host stubs ─────────────────────────────────────────────────────

local VOCAB = duel.vocab()

--- Token ids, rank nine first and rank one last, then everything else.
---
--- The NPC scans this ranking and takes the first token that spells a
--- legal rank, so the stub plays the highest card in hand.
local function ranked_ids()
    local order, seen = {}, {}
    for rank = duel.MAX_RANK, duel.MIN_RANK, -1 do
        local id = VOCAB.to_id[tostring(rank)]
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

local HANDLE = {
    generate_session = function()
        return SESSION
    end,
}

alc.card = {
    get_by_alias = function(alias)
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

local npc = require("card_duel_npc")

--- Policy chunks a persona bake would produce, in their accepted and
--- rejected forms.
local BOLD_SOURCE = "return function(state) return state.my_hand[#state.my_hand] end"
local TIMID_SOURCE = "return function(state) return state.my_hand[1] end"
local BROKEN_SOURCE = "return function(state"
local ILLEGAL_SOURCE = "return function(state) return 42 end"

local STATE = {
    round = 1,
    my_hand = { 1, 3, 5, 7, 9 },
    my_points = 0,
    opp_points = 0,
    opp_played = {},
}

local function ask(payload, extra)
    local request = { task = alc.json_encode(payload) }
    for k, v in pairs(extra or {}) do
        request[k] = v
    end
    npc.reset_cache()
    return npc.run(request).result
end

-- ─── decide / determinism ───────────────────────────────────────────

describe("card_duel_npc decide", function()
    it("returns the highest legal rank of the gated ranking", function()
        expect(ask({ mode = "decide", state = STATE })).to.equal(
            "action=9 legal=true raw_legal=true gated=false"
        )
    end)

    it("stays inside the hand when the top token is not legal", function()
        local state = { round = 2, my_hand = { 2, 4 }, my_points = 0, opp_points = 1 }
        expect(ask({ mode = "decide", state = state })).to.equal(
            "action=4 legal=true raw_legal=false gated=true"
        )
    end)

    it("rejects a state without a hand", function()
        expect(function()
            ask({ mode = "decide", state = { round = 1 } })
        end).to.fail()
    end)
end)

describe("card_duel_npc determinism", function()
    it("agrees across two independent sessions", function()
        expect(ask({ mode = "determinism", state = STATE })).to.equal("deterministic=true action=9")
    end)
end)

-- ─── selfplay: style path (regression) ──────────────────────────────

describe("card_duel_npc selfplay style", function()
    it("scores a full match against the style the stub plays", function()
        local text = ask({ mode = "selfplay", games = 2, seed = 5, style = "bold" })
        expect(text:match("style_match=1%.00") ~= nil).to.equal(true)
    end)

    it("reports the summary fields for any known style", function()
        local text = ask({ mode = "selfplay", games = 2, seed = 5, style = "timid" })
        local pattern = "^winrate=%d+%.%d%d illegal=%d+ style_match=%d+%.%d%d style_hits=%d+/%d+$"
        expect(text:match(pattern) ~= nil).to.equal(true)
    end)

    it("defaults to the aggressive teacher", function()
        local defaulted = ask({ mode = "selfplay", games = 2, seed = 5 })
        local named = ask({ mode = "selfplay", games = 2, seed = 5, style = "aggressive" })
        expect(defaulted).to.equal(named)
    end)

    it("rejects a style outside card_duel.STYLES", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, style = "reckless" })
        end).to.fail()
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            ask({ mode = "selfplay", games = 0, seed = 5, style = "bold" })
        end).to.fail()
    end)
end)

-- ─── selfplay: synthesised teacher override ─────────────────────────

describe("card_duel_npc selfplay policy_source", function()
    it("scores the model against the synthesised policy", function()
        local text = ask({
            mode = "selfplay",
            games = 2,
            seed = 5,
            policy_source = BOLD_SOURCE,
        })
        expect(text:match("style_match=1%.00") ~= nil).to.equal(true)
    end)

    it("matches the equivalent style exactly", function()
        local sourced = ask({
            mode = "selfplay",
            games = 3,
            seed = 9,
            policy_source = TIMID_SOURCE,
        })
        local styled = ask({ mode = "selfplay", games = 3, seed = 9, style = "timid" })
        expect(sourced).to.equal(styled)
    end)

    it("bypasses the style whitelist", function()
        -- A persona Card has no entry in card_duel.STYLES, so a request
        -- carrying both fields must follow the chunk rather than fail on
        -- the name.
        local text = ask({
            mode = "selfplay",
            games = 2,
            seed = 5,
            style = "reckless",
            policy_source = BOLD_SOURCE,
        })
        expect(text:match("style_match=1%.00") ~= nil).to.equal(true)
    end)

    it("accepts the chunk on the strategy ctx", function()
        local on_ctx = ask({ mode = "selfplay", games = 2, seed = 5 }, {
            policy_source = BOLD_SOURCE,
        })
        local on_task = ask({
            mode = "selfplay",
            games = 2,
            seed = 5,
            policy_source = BOLD_SOURCE,
        })
        expect(on_ctx).to.equal(on_task)
    end)

    it("rejects a chunk that does not compile", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = BROKEN_SOURCE })
        end).to.fail()
    end)

    it("rejects a chunk that answers outside the legal actions", function()
        expect(function()
            ask({ mode = "selfplay", games = 2, seed = 5, policy_source = ILLEGAL_SOURCE })
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

describe("card_duel_npc run", function()
    it("rejects an unknown mode", function()
        expect(function()
            ask({ mode = "gamble", state = STATE })
        end).to.fail()
    end)

    it("rejects a task that is not a JSON string", function()
        expect(function()
            npc.run({ task = { mode = "decide" } })
        end).to.fail()
    end)
end)
