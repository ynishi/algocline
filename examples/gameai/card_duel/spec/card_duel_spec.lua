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
