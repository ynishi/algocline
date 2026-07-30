--- card_duel rules tests (mlua-lspec)
---
--- Fences the pure-Lua half of the GameAI demo
--- (`examples/gameai/card_duel/init.lua`) on the default feature set,
--- so a rules regression fails `cargo test` without needing the `nn`
--- feature or a trained Card.
---
--- The module only touches the host through `alc.math.rng_create` /
--- `alc.math.rng_int`, which are stubbed below with a plain LCG. The
--- properties under test (subset, point transitions, encoding
--- stability, teacher determinism, seeded replay) hold for any RNG, so
--- the stub does not weaken them; what it buys is a spec that runs in
--- the bare Lua VM the harness provides.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ── 1. Package path: make card_duel requireable ──────────────
-- ALC_TEST_GAMEAI_DIR is set by the Rust harness to the absolute
-- `examples/gameai` directory. Lua tests MUST NOT guess a relative
-- path off the process CWD, which differs between `cargo test` and
-- IDE runners.
local gameai_dir = os.getenv("ALC_TEST_GAMEAI_DIR") or ""
package.path = gameai_dir .. "/?/init.lua;" .. package.path

-- ── 2. Stub alc.math (linear congruential generator) ─────────
alc = {}
alc.math = {
    rng_create = function(seed)
        return { state = math.floor(seed) % 2147483647 }
    end,
    rng_int = function(rng, min, max)
        rng.state = (rng.state * 1103515245 + 12345) % 2147483648
        local span = max - min + 1
        return min + (rng.state // 65536) % span
    end,
}

local duel = require("card_duel")

--- Play a full game with the teacher on one side and random on the
--- other, returning the finished game plus the per-round trace.
local function playout(seed)
    local g = duel.new_game(seed)
    local rng = alc.math.rng_create(seed + 1)
    local trace = {}
    while not duel.is_over(g) do
        local a1 = duel.policy_aggressive(g.p1)
        local a2 = duel.policy_random(g.p2, rng)
        trace[#trace + 1] = string.format("%s>%d/%d", duel.encode(g.p1), a1, a2)
        g = duel.apply(g, a1, a2)
    end
    return g, trace
end

-- ─── legal_actions ───

describe("card_duel.legal_actions", function()
    it("returns a subset of the hand", function()
        local g = duel.new_game(5)
        local in_hand = {}
        for _, rank in ipairs(g.p1.my_hand) do
            in_hand[rank] = true
        end
        local legal = duel.legal_actions(g.p1)
        expect(#legal > 0).to.equal(true)
        for _, rank in ipairs(legal) do
            expect(in_hand[rank]).to.equal(true)
        end
    end)

    it("is deduplicated and ascending", function()
        local state = { round = 1, my_hand = { 7, 2, 7, 2, 5 }, my_points = 0, opp_points = 0 }
        local legal = duel.legal_actions(state)
        expect(#legal).to.equal(3)
        expect(legal[1]).to.equal(2)
        expect(legal[2]).to.equal(5)
        expect(legal[3]).to.equal(7)
    end)

    it("loses exactly one card from the hand per round", function()
        local g = duel.new_game(13)
        local before = #g.p1.my_hand
        g = duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_aggressive(g.p2))
        expect(#g.p1.my_hand).to.equal(before - 1)
    end)
end)

-- ─── apply ───

describe("card_duel.apply", function()
    it("gives the point to the higher rank", function()
        local g = duel.new_game(2)
        g.p1.my_hand = { 1, 2, 3, 4, 8 }
        g.p2.my_hand = { 1, 2, 3, 4, 6 }
        local next_g = duel.apply(g, 8, 6)
        expect(next_g.p1.my_points).to.equal(1)
        expect(next_g.p2.my_points).to.equal(0)
        expect(next_g.p1.opp_points).to.equal(0)
        expect(next_g.p2.opp_points).to.equal(1)
    end)

    it("gives no point on a tie", function()
        local g = duel.new_game(2)
        g.p1.my_hand = { 1, 2, 3, 4, 5 }
        g.p2.my_hand = { 1, 2, 3, 4, 5 }
        local next_g = duel.apply(g, 5, 5)
        expect(next_g.p1.my_points).to.equal(0)
        expect(next_g.p2.my_points).to.equal(0)
    end)

    it("records the opponent move in the other seat's history", function()
        local g = duel.new_game(2)
        g.p1.my_hand = { 1, 2, 3, 4, 9 }
        g.p2.my_hand = { 1, 2, 3, 4, 7 }
        local next_g = duel.apply(g, 9, 7)
        expect(next_g.p1.opp_played[1]).to.equal(7)
        expect(next_g.p2.opp_played[1]).to.equal(9)
    end)

    it("advances the round on both views", function()
        local g = duel.new_game(2)
        local next_g = duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_aggressive(g.p2))
        expect(next_g.round).to.equal(2)
        expect(next_g.p1.round).to.equal(2)
        expect(next_g.p2.round).to.equal(2)
    end)

    it("does not mutate the previous state", function()
        local g = duel.new_game(2)
        local before = duel.encode(g.p1)
        duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_aggressive(g.p2))
        expect(duel.encode(g.p1)).to.equal(before)
    end)

    it("rejects a rank that is not in hand", function()
        local g = duel.new_game(2)
        g.p1.my_hand = { 1, 1, 1, 1, 1 }
        g.p2.my_hand = { 2, 2, 2, 2, 2 }
        expect(function()
            duel.apply(g, 9, 2)
        end).to.fail()
    end)
end)

-- ─── is_over / winner ───

describe("card_duel.is_over", function()
    it("is false on a fresh deal", function()
        expect(duel.is_over(duel.new_game(17))).to.equal(false)
    end)

    it("is true after HAND_SIZE rounds", function()
        local g = playout(17)
        expect(duel.is_over(g)).to.equal(true)
        expect(g.round).to.equal(duel.HAND_SIZE + 1)
        expect(#g.p1.my_hand).to.equal(0)
    end)

    it("refuses to apply past the end", function()
        local g = playout(17)
        expect(function()
            duel.apply(g, 1, 1)
        end).to.fail()
    end)
end)

describe("card_duel.winner", function()
    it("is nil while the game runs", function()
        expect(duel.winner(duel.new_game(17))).to.equal(nil)
    end)

    it("names the higher score", function()
        local g = playout(23)
        local w = duel.winner(g)
        if g.p1.my_points > g.p2.my_points then
            expect(w).to.equal("p1")
        elseif g.p2.my_points > g.p1.my_points then
            expect(w).to.equal("p2")
        else
            expect(w).to.equal("draw")
        end
    end)

    it("keeps the total points within the round budget", function()
        local g = playout(23)
        expect(g.p1.my_points + g.p2.my_points <= duel.HAND_SIZE).to.equal(true)
    end)
end)

-- ─── encode ───

describe("card_duel.encode", function()
    it("is stable across calls for the same state", function()
        local g = duel.new_game(29)
        expect(duel.encode(g.p1)).to.equal(duel.encode(g.p1))
    end)

    it("ignores hand order", function()
        local a = { round = 2, my_hand = { 8, 2, 5, 1 }, my_points = 0, opp_points = 1, opp_played = { 3 } }
        local b = { round = 2, my_hand = { 1, 2, 5, 8 }, my_points = 0, opp_points = 1, opp_played = { 3 } }
        expect(duel.encode(a)).to.equal(duel.encode(b))
    end)

    it("distinguishes the score gap", function()
        local behind = { round = 3, my_hand = { 4, 6, 9 }, my_points = 0, opp_points = 2, opp_played = { 1, 2 } }
        local ahead = { round = 3, my_hand = { 4, 6, 9 }, my_points = 2, opp_points = 0, opp_played = { 1, 2 } }
        expect(duel.encode(behind) ~= duel.encode(ahead)).to.equal(true)
    end)

    it("keeps prompt plus action inside the tiny preset context", function()
        local g = duel.new_game(37)
        local rng = alc.math.rng_create(1)
        while not duel.is_over(g) do
            expect(#duel.to_ids(duel.encode(g.p1) .. ">") + 1 <= 16).to.equal(true)
            expect(#duel.to_ids(duel.encode(g.p2) .. ">") + 1 <= 16).to.equal(true)
            g = duel.apply(g, duel.policy_aggressive(g.p1), duel.policy_random(g.p2, rng))
        end
    end)
end)

describe("card_duel.to_ids", function()
    it("maps every alphabet char inside the tiny vocabulary", function()
        local v = duel.vocab()
        for _, id in ipairs(duel.to_ids("R1H13579P42O86>")) do
            expect(id >= 0 and id < 64).to.equal(true)
        end
        expect(v.pad_id).to.equal(0)
    end)

    it("rejects a char outside the alphabet", function()
        expect(function()
            duel.to_ids("R1X")
        end).to.fail()
    end)
end)

-- ─── policies ───

describe("card_duel.policy_aggressive", function()
    it("plays the highest legal rank when level", function()
        local state = { round = 1, my_hand = { 3, 9, 6 }, my_points = 1, opp_points = 1 }
        expect(duel.policy_aggressive(state)).to.equal(9)
    end)

    it("plays the highest legal rank when behind", function()
        local state = { round = 1, my_hand = { 3, 9, 6 }, my_points = 0, opp_points = 2 }
        expect(duel.policy_aggressive(state)).to.equal(9)
    end)

    it("plays the lowest legal rank when ahead", function()
        local state = { round = 1, my_hand = { 3, 9, 6 }, my_points = 2, opp_points = 0 }
        expect(duel.policy_aggressive(state)).to.equal(3)
    end)

    it("is deterministic over a whole playout", function()
        local g = duel.new_game(41)
        while not duel.is_over(g) do
            local first = duel.policy_aggressive(g.p1)
            expect(duel.policy_aggressive(g.p1)).to.equal(first)
            g = duel.apply(g, first, duel.policy_aggressive(g.p2))
        end
    end)

    it("only ever returns a legal action", function()
        local g = duel.new_game(43)
        while not duel.is_over(g) do
            local a1 = duel.policy_aggressive(g.p1)
            local found = false
            for _, rank in ipairs(duel.legal_actions(g.p1)) do
                if rank == a1 then
                    found = true
                end
            end
            expect(found).to.equal(true)
            g = duel.apply(g, a1, duel.policy_aggressive(g.p2))
        end
    end)
end)

describe("card_duel.policy_random", function()
    it("only ever returns a legal action", function()
        local g = duel.new_game(47)
        local rng = alc.math.rng_create(3)
        while not duel.is_over(g) do
            local a2 = duel.policy_random(g.p2, rng)
            local found = false
            for _, rank in ipairs(duel.legal_actions(g.p2)) do
                if rank == a2 then
                    found = true
                end
            end
            expect(found).to.equal(true)
            g = duel.apply(g, duel.policy_aggressive(g.p1), a2)
        end
    end)
end)

-- ─── seeded replay ───

describe("card_duel seeded replay", function()
    it("reproduces the deal for the same seed", function()
        local a = duel.new_game(53)
        local b = duel.new_game(53)
        expect(duel.encode(a.p1)).to.equal(duel.encode(b.p1))
        expect(duel.encode(a.p2)).to.equal(duel.encode(b.p2))
    end)

    it("reproduces the whole playout for the same seed", function()
        local _, first = playout(59)
        local _, second = playout(59)
        expect(#first).to.equal(#second)
        expect(table.concat(first, ";")).to.equal(table.concat(second, ";"))
    end)

    it("separates different seeds", function()
        local _, a = playout(61)
        local _, b = playout(67)
        expect(table.concat(a, ";") ~= table.concat(b, ";")).to.equal(true)
    end)
end)

-- ─── strategy entry ───

describe("card_duel.run", function()
    it("returns the encoded opening state", function()
        local out = duel.run({ seed = 71 })
        expect(out.result).to.equal(duel.encode(duel.new_game(71).p1))
    end)

    it("defaults the seed", function()
        expect(duel.run({}).result).to.equal(duel.run().result)
    end)
end)
