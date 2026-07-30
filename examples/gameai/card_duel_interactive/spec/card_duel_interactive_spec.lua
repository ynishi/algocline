-- card_duel_interactive/spec/card_duel_interactive_spec.lua
--
-- Package-level spec for the interactive session. Run with
-- `alc_pkg_test pkg="card_duel_interactive"` after `alc_pkg_link` has
-- registered `card_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No model and no state file are touched: `card_duel_npc` is replaced
-- through `package.preload` by a stub that answers from the rules, and
-- `alc.state` is swapped for an in-memory table the assertions can read
-- back. What is left under test is the session lifecycle, the move
-- validation and the seat bookkeeping.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("card_duel")

-- ─── Host stubs ─────────────────────────────────────────────────────

--- In-memory replacement for the file-backed `alc.state` namespace.
local store = {}

alc.state = {
    get = function(key, default)
        local value = store[key]
        if value == nil then
            return default
        end
        return value
    end,
    set = function(key, value)
        store[key] = value
    end,
    delete = function(key)
        store[key] = nil
    end,
}

--- Last state the NPC was asked about, so the seat wiring is checkable.
local last_npc_state = nil

package.preload["card_duel_npc"] = function()
    return {
        run = function(ctx)
            local req = alc.json_decode(ctx.task)
            last_npc_state = req.state
            -- Timid: always the lowest legal rank.
            local rank = duel.legal_actions(req.state)[1]
            return {
                result = string.format("action=%d legal=true raw_legal=true gated=false", rank),
            }
        end,
    }
end

local game = require("card_duel_interactive")

local KEY = "gameai_interactive:default"

local function reset()
    for key in pairs(store) do
        store[key] = nil
    end
    last_npc_state = nil
end

--- Deal a game and pin both hands, so the seat assertions do not depend
--- on what the seed happens to deal.
local function new_with_hands(opts, p1_hand, p2_hand)
    local view = game.run(opts)
    local session = store[KEY]
    session.g.p1.my_hand = p1_hand
    session.g.p2.my_hand = p2_hand
    store[KEY] = session
    return view
end

-- ─── new ────────────────────────────────────────────────────────────

describe("card_duel_interactive new", function()
    it("opens a board on round one", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 7 })
        expect(view.round).to.equal(1)
        expect(view.status).to.equal("your_turn")
        expect(#view.your_hand).to.equal(duel.HAND_SIZE)
        expect(view.your_points).to.equal(0)
        expect(view.npc_points).to.equal(0)
        expect(#view.npc_played).to.equal(0)
        expect(#view.legal_actions > 0).to.equal(true)
        expect(type(view.text)).to.equal("string")
        expect(view.result).to.equal(view.text)
    end)

    it("stores the session under the game id", function()
        reset()
        game.run({ action = "new", seed = 7, game_id = "demo" })
        expect(store["gameai_interactive:demo"] ~= nil).to.equal(true)
        expect(store[KEY]).to.equal(nil)
    end)

    it("rejects a style outside card_duel.STYLES", function()
        reset()
        expect(function()
            game.run({ action = "new", style = "reckless", seed = 7 })
        end).to.fail()
    end)

    it("rejects a seat other than one or two", function()
        reset()
        expect(function()
            game.run({ action = "new", seed = 7, user_seat = 3 })
        end).to.fail()
    end)

    it("rejects a seat that is not a number", function()
        -- Falling back to seat one here would deal the human the other
        -- hand and name the wrong side as the winner.
        reset()
        expect(function()
            game.run({ action = "new", seed = 7, user_seat = "second" })
        end).to.fail()
    end)

    it("rejects a seed that is not a number", function()
        reset()
        expect(function()
            game.run({ action = "new", seed = "later" })
        end).to.fail()
    end)

    it("rejects an unknown action", function()
        reset()
        expect(function()
            game.run({ action = "resign" })
        end).to.fail()
    end)
end)

-- ─── play ───────────────────────────────────────────────────────────

describe("card_duel_interactive play", function()
    it("reaches a finished game after five rounds", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for round = 1, duel.HAND_SIZE do
            expect(view.status).to.equal("your_turn")
            expect(view.round).to.equal(round)
            view = game.run({ action = "play", rank = view.legal_actions[#view.legal_actions] })
        end
        expect(view.status).to.equal("finished")
        expect(view.winner == "you" or view.winner == "npc" or view.winner == "draw").to.equal(true)
        expect(view.your_points + view.npc_points <= duel.HAND_SIZE).to.equal(true)
        expect(#view.legal_actions).to.equal(0)
    end)

    it("reports the round that was just played", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        local view = game.run({ action = "play", rank = 1 })
        expect(view.last_round.round).to.equal(1)
        expect(view.last_round.you).to.equal(1)
        expect(view.last_round.npc).to.equal(9)
        expect(view.last_round.outcome).to.equal("the npc scores")
        expect(view.your_points).to.equal(0)
        expect(view.npc_points).to.equal(1)
        expect(view.npc_played[1]).to.equal(9)
    end)

    it("asks the NPC about its own seat", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        game.run({ action = "play", rank = 1 })
        expect(last_npc_state.my_hand[1]).to.equal(9)
    end)

    it("rejects a rank that is not in hand", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        expect(function()
            game.run({ action = "play", rank = 4 })
        end).to.fail()
    end)

    it("rejects a move without an active game", function()
        reset()
        expect(function()
            game.run({ action = "play", rank = 1 })
        end).to.fail()
    end)

    it("rejects a move after the game is over", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for _ = 1, duel.HAND_SIZE do
            view = game.run({ action = "play", rank = view.legal_actions[1] })
        end
        expect(function()
            game.run({ action = "play", rank = 1 })
        end).to.fail()
    end)
end)

-- ─── Seat two ───────────────────────────────────────────────────────

describe("card_duel_interactive seat two", function()
    it("applies the human move to the second seat", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21, user_seat = 2 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        local view = game.run({ action = "play", rank = 9 })
        -- The human plays 9 from seat two against the stub's 1: swapping
        -- the pair passed to `apply` would credit the point to the NPC.
        expect(view.last_round.you).to.equal(9)
        expect(view.last_round.npc).to.equal(1)
        expect(view.your_points).to.equal(1)
        expect(view.npc_points).to.equal(0)
        expect(view.npc_played[1]).to.equal(1)
    end)

    it("shows the second seat hand as the human hand", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21, user_seat = 2 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        local view = game.run({ action = "show" })
        expect(view.your_hand[1]).to.equal(9)
        expect(view.legal_actions[1]).to.equal(9)
    end)
end)

-- ─── show / end ─────────────────────────────────────────────────────

describe("card_duel_interactive show", function()
    it("returns the current board without moving", function()
        reset()
        local opened = game.run({ action = "new", style = "timid", seed = 21 })
        local shown = game.run({ action = "show" })
        expect(shown.round).to.equal(opened.round)
        expect(shown.text).to.equal(opened.text)
    end)

    it("rejects a show without an active game", function()
        reset()
        expect(function()
            game.run({ action = "show" })
        end).to.fail()
    end)
end)

describe("card_duel_interactive end", function()
    it("drops the session and reports the final board", function()
        reset()
        game.run({ action = "new", style = "timid", seed = 21 })
        local view = game.run({ action = "end" })
        expect(view.ended).to.equal(true)
        expect(store[KEY]).to.equal(nil)
    end)

    it("leaves other sessions alone", function()
        reset()
        game.run({ action = "new", style = "timid", seed = 21 })
        game.run({ action = "new", style = "timid", seed = 21, game_id = "other" })
        game.run({ action = "end" })
        expect(store[KEY]).to.equal(nil)
        expect(store["gameai_interactive:other"] ~= nil).to.equal(true)
    end)

    it("carries the winner once the game is over", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for _ = 1, duel.HAND_SIZE do
            view = game.run({ action = "play", rank = view.legal_actions[#view.legal_actions] })
        end
        local ended = game.run({ action = "end" })
        expect(ended.status).to.equal("finished")
        expect(ended.winner).to.equal(view.winner)
    end)

    it("rejects an end without an active game", function()
        reset()
        expect(function()
            game.run({ action = "end" })
        end).to.fail()
    end)
end)
