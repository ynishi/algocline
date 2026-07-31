-- card_duel/spec/card_duel_spec.lua
--
-- Package-level spec for the card duel rules. Run with
-- `alc_pkg_test pkg="card_duel"` after `alc_pkg_link` has registered
-- the package. The `lust` globals are pre-loaded by the runner.
--
-- The CI-side counterpart is
-- `crates/algocline-engine/tests/lua/card_duel_rules_test.lua`, which
-- covers the same invariants inside the plain Lua VM used by
-- `cargo test`. This file stays focused on the behaviour an author
-- iterating on the package cares about.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("card_duel")

local function play_teacher_vs_random(seed)
    local g = duel.new_game(seed)
    local rng = alc.math.rng_create(seed + 1)
    while not duel.is_over(g) do
        g = duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_random(g.p2, rng))
    end
    return g
end

describe("card_duel.new_game", function()
    it("deals HAND_SIZE cards to both players", function()
        local g = duel.new_game(7)
        expect(#g.p1.my_hand).to.equal(duel.HAND_SIZE)
        expect(#g.p2.my_hand).to.equal(duel.HAND_SIZE)
    end)

    it("draws ranks inside the configured range", function()
        local g = duel.new_game(11)
        for _, rank in ipairs(g.p1.my_hand) do
            expect(rank >= duel.MIN_RANK and rank <= duel.MAX_RANK).to.equal(true)
        end
    end)

    it("is reproducible from the seed", function()
        local a = duel.new_game(99)
        local b = duel.new_game(99)
        expect(duel.encode(a.p1)).to.equal(duel.encode(b.p1))
        expect(duel.encode(a.p2)).to.equal(duel.encode(b.p2))
    end)
end)

describe("card_duel.legal_actions", function()
    it("returns distinct ranks in ascending order", function()
        local state = { round = 1, my_hand = { 4, 1, 4, 9, 1 }, my_points = 0, opp_points = 0 }
        local legal = duel.legal_actions(state)
        expect(#legal).to.equal(3)
        expect(legal[1]).to.equal(1)
        expect(legal[2]).to.equal(4)
        expect(legal[3]).to.equal(9)
    end)
end)

describe("card_duel.apply", function()
    it("awards one point to the higher rank", function()
        local g = duel.new_game(3)
        g.p1.my_hand = { 1, 2, 3, 4, 9 }
        g.p2.my_hand = { 1, 2, 3, 4, 5 }
        local next_g = duel.apply(g, 9, 5)
        expect(next_g.p1.my_points).to.equal(1)
        expect(next_g.p2.my_points).to.equal(0)
    end)

    it("awards nothing on a tie", function()
        local g = duel.new_game(3)
        g.p1.my_hand = { 1, 2, 3, 4, 6 }
        g.p2.my_hand = { 1, 2, 3, 4, 6 }
        local next_g = duel.apply(g, 6, 6)
        expect(next_g.p1.my_points).to.equal(0)
        expect(next_g.p2.my_points).to.equal(0)
    end)

    it("rejects a rank that is not in hand", function()
        local g = duel.new_game(3)
        g.p1.my_hand = { 1, 2, 3, 4, 5 }
        g.p2.my_hand = { 1, 2, 3, 4, 5 }
        expect(function()
            duel.apply(g, 9, 5)
        end).to.fail()
    end)
end)

describe("card_duel.encode", function()
    it("is stable for the same state", function()
        local state = {
            round = 2,
            my_hand = { 5, 1, 9, 3 },
            my_points = 1,
            opp_points = 0,
            opp_played = { 4 },
        }
        expect(duel.encode(state)).to.equal(duel.encode(state))
    end)

    it("sorts the hand so deal order does not leak", function()
        local a = { round = 1, my_hand = { 9, 1, 5, 3, 7 }, my_points = 0, opp_points = 0 }
        local b = { round = 1, my_hand = { 1, 3, 5, 7, 9 }, my_points = 0, opp_points = 0 }
        expect(duel.encode(a)).to.equal(duel.encode(b))
    end)

    it("keeps prompt plus action inside the tiny preset context", function()
        local g = duel.new_game(21)
        local rng = alc.math.rng_create(5)
        while not duel.is_over(g) do
            local prompt = duel.encode(g.p1) .. ">"
            expect(#duel.to_ids(prompt) + 1 <= 16).to.equal(true)
            g = duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_random(g.p2, rng))
        end
    end)
end)

describe("card_duel.policy_aggressive", function()
    it("plays the highest card when not ahead", function()
        local state = { round = 1, my_hand = { 2, 8, 5 }, my_points = 0, opp_points = 1 }
        expect(duel.policy_aggressive(state)).to.equal(8)
    end)

    it("plays the lowest card when ahead", function()
        local state = { round = 1, my_hand = { 2, 8, 5 }, my_points = 2, opp_points = 1 }
        expect(duel.policy_aggressive(state)).to.equal(2)
    end)

    it("is deterministic", function()
        local state = { round = 3, my_hand = { 6, 6, 1 }, my_points = 1, opp_points = 1 }
        expect(duel.policy_aggressive(state)).to.equal(duel.policy_aggressive(state))
    end)
end)

describe("card_duel.STYLES", function()
    it("exports a policy for every canonical style", function()
        expect(#duel.STYLES).to.equal(6)
        for _, style in ipairs(duel.STYLES) do
            expect(type(duel["policy_" .. style])).to.equal("function")
        end
    end)
end)

describe("card_duel style zoo", function()
    it("plays the lowest card as timid", function()
        local state = { round = 1, my_hand = { 2, 8, 5 }, my_points = 3, opp_points = 0 }
        expect(duel.policy_timid(state)).to.equal(2)
    end)

    it("plays the highest card as bold", function()
        local state = { round = 1, my_hand = { 2, 8, 5 }, my_points = 3, opp_points = 0 }
        expect(duel.policy_bold(state)).to.equal(8)
    end)

    it("holds back while behind as defensive", function()
        local state = { round = 2, my_hand = { 2, 8, 5 }, my_points = 0, opp_points = 1 }
        expect(duel.policy_defensive(state)).to.equal(2)
    end)

    it("presses from round three as late bloomer", function()
        local early = { round = 2, my_hand = { 2, 8, 5 }, my_points = 0, opp_points = 0 }
        local late = { round = 3, my_hand = { 2, 8, 5 }, my_points = 0, opp_points = 0 }
        expect(duel.policy_late_bloomer(early)).to.equal(2)
        expect(duel.policy_late_bloomer(late)).to.equal(8)
    end)

    it("answers the last opponent card as mimic", function()
        local state = {
            round = 3,
            my_hand = { 2, 8, 5 },
            my_points = 1,
            opp_points = 1,
            opp_played = { 9, 6 },
        }
        expect(duel.policy_mimic(state)).to.equal(5)
    end)
end)

describe("card_duel game loop", function()
    it("terminates and names a winner", function()
        local g = play_teacher_vs_random(31)
        expect(duel.is_over(g)).to.equal(true)
        local w = duel.winner(g)
        expect(w == "p1" or w == "p2" or w == "draw").to.equal(true)
    end)

    it("has no winner before the last round", function()
        local g = duel.new_game(31)
        expect(duel.winner(g)).to.equal(nil)
    end)
end)

-- ─── Corpus ─────────────────────────────────────────────────────────

local CTX_LEN = 16

--- Decode a padded token row back to the training line it carries.
local function row_text(row)
    local v = duel.vocab()
    local chars = {}
    for _, id in ipairs(row) do
        if id == v.pad_id then
            break
        end
        chars[#chars + 1] = v.to_char[id]
    end
    return table.concat(chars)
end

describe("card_duel.build_corpus", function()
    it("emits one row per seat per round", function()
        local rows = duel.build_corpus(duel.policy_aggressive, {
            ctx_len = CTX_LEN,
            games = 3,
            seed = 11,
        })
        expect(#rows).to.equal(3 * duel.ROWS_PER_GAME)
    end)

    it("writes state, separator and action on every line", function()
        local rows = duel.build_corpus(duel.policy_timid, {
            ctx_len = CTX_LEN,
            games = 2,
            seed = 13,
        })
        for _, row in ipairs(rows) do
            local text = row_text(row)
            expect(text:match("^R%d+H%d*P%d%dO%d*>%d\n$") ~= nil).to.equal(true)
        end
    end)

    it("pads every row to the context window", function()
        local rows = duel.build_corpus(duel.policy_bold, {
            ctx_len = CTX_LEN,
            games = 2,
            seed = 17,
        })
        local pad_id = duel.vocab().pad_id
        for _, row in ipairs(rows) do
            expect(#row).to.equal(CTX_LEN)
            expect(row[#row]).to.equal(pad_id)
        end
    end)

    it("honours a pad id override", function()
        local rows = duel.build_corpus(duel.policy_bold, {
            ctx_len = CTX_LEN,
            games = 1,
            seed = 19,
            pad_id = 2,
        })
        expect(rows[1][#rows[1]]).to.equal(2)
    end)

    it("is reproducible from the seed", function()
        local opts = { ctx_len = CTX_LEN, games = 2, seed = 5 }
        local a = duel.build_corpus(duel.policy_aggressive, opts)
        local b = duel.build_corpus(duel.policy_aggressive, opts)
        expect(row_text(a[1])).to.equal(row_text(b[1]))
        expect(row_text(a[#a])).to.equal(row_text(b[#b]))
    end)

    it("refuses a context window the line does not fit in", function()
        -- Truncating instead would teach the model a state it can never
        -- be asked about at decode time.
        expect(function()
            duel.build_corpus(duel.policy_timid, { ctx_len = 4, games = 1, seed = 23 })
        end).to.fail()
    end)

    it("rejects a policy that is not a function", function()
        expect(function()
            duel.build_corpus("timid", { ctx_len = CTX_LEN, games = 1, seed = 23 })
        end).to.fail()
    end)

    it("rejects a missing context window or game count", function()
        expect(function()
            duel.build_corpus(duel.policy_timid, { games = 1 })
        end).to.fail()
        expect(function()
            duel.build_corpus(duel.policy_timid, { ctx_len = CTX_LEN, games = 0 })
        end).to.fail()
    end)
end)

describe("card_duel.sample_states", function()
    it("collects both seats of every round", function()
        expect(#duel.sample_states({ games = 4, seed = 3 })).to.equal(4 * duel.ROWS_PER_GAME)
    end)

    it("is reproducible from the seed", function()
        local a = duel.sample_states({ games = 2, seed = 29 })
        local b = duel.sample_states({ games = 2, seed = 29 })
        for i, state in ipairs(a) do
            expect(duel.encode(state)).to.equal(duel.encode(b[i]))
        end
    end)

    it("rejects a non-positive game count", function()
        expect(function()
            duel.sample_states({ games = 0 })
        end).to.fail()
    end)
end)

-- ─── Synthesised policies ───────────────────────────────────────────

describe("card_duel.compile_policy", function()
    local states = { { round = 1, my_hand = { 1, 3, 5, 7, 9 }, my_points = 0, opp_points = 0 } }

    it("accepts a chunk that answers legally and deterministically", function()
        local policy = duel.compile_policy("return function(state) return state.my_hand[1] end", {
            games = 2,
            seed = 31,
        })
        expect(policy(states[1])).to.equal(1)
    end)

    it("rejects a chunk that does not compile", function()
        expect(function()
            duel.compile_policy("return function(state", { states = states })
        end).to.fail()
    end)

    it("rejects a chunk that does not return a function", function()
        expect(function()
            duel.compile_policy("return 7", { states = states })
        end).to.fail()
    end)

    it("rejects an answer outside the legal actions", function()
        expect(function()
            duel.compile_policy("return function(state) return 42 end", { states = states })
        end).to.fail()
    end)

    it("rejects an answer that is not an integer rank", function()
        expect(function()
            duel.compile_policy("return function(state) return nil end", { states = states })
        end).to.fail()
    end)

    it("rejects a policy that answers the same state twice differently", function()
        local flip = [[
local n = 0
return function(state)
    n = n + 1
    if n % 2 == 0 then
        return state.my_hand[1]
    end
    return state.my_hand[#state.my_hand]
end
]]
        expect(function()
            duel.compile_policy(flip, { states = states })
        end).to.fail()
    end)

    it("rejects a chunk reaching for a global outside the sandbox", function()
        -- `os` / `io` / `load` / `setmetatable` are all absent, so the
        -- call raises instead of touching the host.
        expect(function()
            duel.compile_policy("return function(state) return os.time() end", { states = states })
        end).to.fail()
        expect(function()
            duel.compile_policy("return function(state) return load('return 1')() end", {
                states = states,
            })
        end).to.fail()
    end)

    it("rejects a source that is not a string", function()
        expect(function()
            duel.compile_policy(nil, { states = states })
        end).to.fail()
    end)

    it("rejects an empty validation batch", function()
        expect(function()
            duel.compile_policy("return function(state) return state.my_hand[1] end", {
                states = {},
            })
        end).to.fail()
    end)

    it("keeps a mutating chunk away from the live state", function()
        local mutating = [[
return function(state)
    table.remove(state.my_hand)
    return state.my_hand[1]
end
]]
        local state = { round = 1, my_hand = { 1, 3, 5, 7, 9 }, my_points = 0, opp_points = 0 }
        local policy = duel.compile_policy(mutating, { states = { state } })
        expect(#state.my_hand).to.equal(5)
        expect(policy(state)).to.equal(1)
        expect(#state.my_hand).to.equal(5)
    end)

    it("labels a corpus like any other policy", function()
        local policy = duel.compile_policy(
            "return function(state) return state.my_hand[#state.my_hand] end",
            { games = 2, seed = 37 }
        )
        local rows = duel.build_corpus(policy, { ctx_len = CTX_LEN, games = 2, seed = 37 })
        expect(#rows).to.equal(2 * duel.ROWS_PER_GAME)
        for _, row in ipairs(rows) do
            expect(row_text(row):match("^R%d+H%d*P%d%dO%d*>%d\n$") ~= nil).to.equal(true)
        end
    end)
end)

describe("card_duel.vocab", function()
    it("fits the tiny preset vocabulary", function()
        expect(duel.vocab().size <= 64).to.equal(true)
    end)

    it("round-trips every alphabet char", function()
        local v = duel.vocab()
        local text = "R1H12345P00O>"
        local ids = duel.to_ids(text)
        local back = {}
        for _, id in ipairs(ids) do
            back[#back + 1] = v.to_char[id]
        end
        expect(table.concat(back)).to.equal(text)
    end)
end)
