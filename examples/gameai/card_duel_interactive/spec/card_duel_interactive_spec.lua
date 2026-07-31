-- card_duel_interactive/spec/card_duel_interactive_spec.lua
--
-- Package-level spec for the interactive session. Run with
-- `alc_pkg_test pkg="card_duel_interactive"` after `alc_pkg_link` has
-- registered `card_duel` and this package. The `lust` globals are
-- pre-loaded by the runner.
--
-- No model, no Card and no state file are touched: `card_duel_npc` is
-- replaced through `package.preload` by a stub that answers from the
-- rules, `alc.card` hands out the aliases the test registers and
-- `alc.state` is swapped for an in-memory table the assertions can read
-- back. What is left under test is the session lifecycle, the move
-- validation, the seat bookkeeping and the style resolution — a
-- canonical style comes from `card_duel`, a persona style only from a
-- pinned Card.

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

--- Cards a persona bake would have pinned, keyed by alias.
---
--- Empty by default, so a style outside `card_duel.STYLES` is a typo
--- unless a test says otherwise.
local persona_cards = {}

--- Alias lookups made so far, so the canonical path can be shown to
--- stay off the Card layer.
local alias_lookups = 0

alc.card = {
    get_by_alias = function(alias)
        alias_lookups = alias_lookups + 1
        local card_id = persona_cards[alias]
        if card_id == nil then
            return nil
        end
        return { card_id = card_id }
    end,
}

--- Last state the NPC was asked about, so the seat wiring is checkable.
local last_npc_state = nil

--- Last alias the NPC was asked to decode with, which is the only thing
--- that makes a persona game differ from a canonical one.
local last_npc_alias = nil

package.preload["card_duel_npc"] = function()
    return {
        run = function(ctx)
            local req = alc.json_decode(ctx.task)
            last_npc_state = req.state
            last_npc_alias = ctx.card_alias
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
    for alias in pairs(persona_cards) do
        persona_cards[alias] = nil
    end
    last_npc_state = nil
    last_npc_alias = nil
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

-- ─── Persona styles ─────────────────────────────────────────────────

describe("card_duel_interactive persona styles", function()
    it("accepts a style that exists only as a pinned Card", function()
        reset()
        persona_cards["card_duel_npc_gambler"] = "stub-card-gambler"
        local view = game.run({ action = "new", style = "gambler", seed = 7 })
        expect(view.style).to.equal("gambler")
        expect(view.style_kind).to.equal("persona")
        expect(view.status).to.equal("your_turn")
    end)

    it("decodes the npc seat through the persona alias", function()
        reset()
        persona_cards["card_duel_npc_gambler"] = "stub-card-gambler"
        new_with_hands(
            { action = "new", style = "gambler", seed = 21 },
            { 9, 9, 9, 9, 9 },
            { 1, 1, 1, 1, 1 }
        )
        local view = game.run({ action = "play", rank = 9 })
        -- Play is a decode through the alias and nothing else, so a
        -- persona game reaches a scored round like a canonical one.
        expect(last_npc_alias).to.equal("card_duel_npc_gambler")
        expect(view.last_round.you).to.equal(9)
        expect(view.last_round.npc).to.equal(1)
        expect(view.your_points).to.equal(1)
    end)

    it("keeps the persona kind across the session actions", function()
        reset()
        persona_cards["card_duel_npc_gambler"] = "stub-card-gambler"
        game.run({ action = "new", style = "gambler", seed = 7 })
        expect(game.run({ action = "show" }).style_kind).to.equal("persona")
        expect(game.run({ action = "end" }).style_kind).to.equal("persona")
    end)

    it("refuses a style with neither a policy nor a Card", function()
        reset()
        local ok, err = pcall(game.run, { action = "new", style = "gambler", seed = 7 })
        expect(ok).to.equal(false)
        -- The message has to name both places that were searched, or a
        -- caller cannot tell a misspelt style from an unbaked persona.
        expect(err:match("canonical:") ~= nil).to.equal(true)
        expect(err:match("card_duel_npc_gambler") ~= nil).to.equal(true)
    end)

    it("refuses a persona style when the Card layer cannot answer", function()
        reset()
        local card_ns = alc.card
        alc.card = nil
        local ok, err = pcall(game.run, { action = "new", style = "gambler", seed = 7 })
        alc.card = card_ns
        -- Falling back to the default style here would seat an NPC the
        -- caller never asked for, so the missing surface is reported.
        expect(ok).to.equal(false)
        expect(err:match("get_by_alias is unavailable") ~= nil).to.equal(true)
    end)
end)

-- ─── Canonical styles (regression) ──────────────────────────────────

describe("card_duel_interactive canonical styles", function()
    it("reports a shipped style as canonical", function()
        reset()
        local view = game.run({ action = "new", style = "bold", seed = 7 })
        expect(view.style).to.equal("bold")
        expect(view.style_kind).to.equal("canonical")
    end)

    it("defaults to the aggressive style", function()
        reset()
        local view = game.run({ action = "new", seed = 7 })
        expect(view.style).to.equal("aggressive")
        expect(view.style_kind).to.equal("canonical")
    end)

    it("never asks the Card layer about a shipped style", function()
        reset()
        local before = alias_lookups
        for _, style in ipairs(duel.STYLES) do
            game.run({ action = "new", style = style, seed = 7 })
        end
        -- A persona Card must not be able to shadow a shipped style, so
        -- the alias lookup stays behind the policy check.
        expect(alias_lookups).to.equal(before)
    end)

    it("still refuses a style that is not a string", function()
        reset()
        expect(function()
            game.run({ action = "new", style = 7, seed = 7 })
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

-- ─── Play log ───────────────────────────────────────────────────────

describe("card_duel_interactive move log", function()
    it("records the position the human answered", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        game.run({ action = "play", rank = 1 })
        local log = game.run({ action = "end" }).move_log
        expect(#log).to.equal(1)
        expect(log[1].round).to.equal(1)
        -- The hand still holds five cards and the history is empty: the
        -- entry is the position before the round was applied, which is
        -- the only one the human was ever asked about.
        expect(#log[1].my_hand).to.equal(duel.HAND_SIZE)
        expect(log[1].my_points).to.equal(0)
        expect(log[1].opp_points).to.equal(0)
        expect(#log[1].opp_played).to.equal(0)
        expect(log[1].action).to.equal(1)
    end)

    it("appends one entry per move, in order", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for _ = 1, duel.HAND_SIZE do
            view = game.run({ action = "play", rank = view.legal_actions[1] })
        end
        local log = game.run({ action = "end" }).move_log
        expect(#log).to.equal(duel.HAND_SIZE)
        for round, move in ipairs(log) do
            expect(move.round).to.equal(round)
            expect(#move.my_hand).to.equal(duel.HAND_SIZE - round + 1)
            expect(#move.opp_played).to.equal(round - 1)
        end
    end)

    it("carries the running score of the logged position", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 9, 9, 9, 9, 9 },
            { 1, 1, 1, 1, 1 }
        )
        game.run({ action = "play", rank = 9 })
        game.run({ action = "play", rank = 9 })
        local log = game.run({ action = "end" }).move_log
        expect(log[1].my_points).to.equal(0)
        expect(log[2].my_points).to.equal(1)
        expect(log[2].opp_played[1]).to.equal(1)
    end)

    it("logs the human seat when the human sits second", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21, user_seat = 2 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        game.run({ action = "play", rank = 9 })
        local log = game.run({ action = "end" }).move_log
        -- Seat two holds the nines here, so a log built from seat one
        -- would carry a hand the human never held.
        expect(log[1].my_hand[1]).to.equal(9)
        expect(log[1].action).to.equal(9)
    end)

    it("keeps a snapshot rather than a reference into the position", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        game.run({ action = "play", rank = 1 })
        -- Rewrite the live hand between the two moves, which is what
        -- `new_with_hands` does at the start of every seat test.
        local session = store[KEY]
        session.g.p1.my_hand = { 2, 2, 2, 2 }
        store[KEY] = session
        game.run({ action = "play", rank = 2 })
        local log = game.run({ action = "end" }).move_log
        expect(log[1].my_hand[1]).to.equal(1)
        expect(#log[1].my_hand).to.equal(duel.HAND_SIZE)
        expect(log[2].my_hand[1]).to.equal(2)
    end)

    it("hands out a copy the caller cannot write back into", function()
        reset()
        new_with_hands(
            { action = "new", style = "timid", seed = 21 },
            { 1, 1, 1, 1, 1 },
            { 9, 9, 9, 9, 9 }
        )
        game.run({ action = "play", rank = 1 })
        local stored = store[KEY].move_log
        local view = game.run({ action = "end" })
        view.move_log[1].my_hand[1] = 7
        view.move_log[1].opp_played[1] = 7
        expect(stored[1].my_hand[1]).to.equal(1)
        expect(#stored[1].opp_played).to.equal(0)
    end)

    it("returns an empty log for a session that took no move", function()
        reset()
        game.run({ action = "new", style = "timid", seed = 21 })
        local log = game.run({ action = "end" }).move_log
        expect(type(log)).to.equal("table")
        expect(#log).to.equal(0)
    end)

    it("is only handed out by end", function()
        reset()
        local opened = game.run({ action = "new", style = "timid", seed = 21 })
        expect(opened.move_log).to.equal(nil)
        expect(game.run({ action = "play", rank = opened.legal_actions[1] }).move_log).to.equal(nil)
        expect(game.run({ action = "show" }).move_log).to.equal(nil)
        -- A log is a whole session, so an unfinished one would hand the
        -- bake path rows the next move is about to relabel.
        expect(#game.run({ action = "end" }).move_log).to.equal(1)
    end)

    it("leaves the rest of the end view untouched", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for _ = 1, duel.HAND_SIZE do
            view = game.run({ action = "play", rank = view.legal_actions[1] })
        end
        local ended = game.run({ action = "end" })
        expect(ended.ended).to.equal(true)
        expect(ended.status).to.equal("finished")
        expect(ended.winner).to.equal(view.winner)
        expect(ended.result).to.equal(ended.text)
        expect(store[KEY]).to.equal(nil)
    end)

    it("produces a log the bake path accepts as is", function()
        reset()
        local view = game.run({ action = "new", style = "timid", seed = 21 })
        for _ = 1, duel.HAND_SIZE do
            view = game.run({ action = "play", rank = view.legal_actions[1] })
        end
        local log = game.run({ action = "end" }).move_log
        -- `examples/gameai/bake_card_duel_from_log.lua` feeds the log
        -- straight to this call, so the two shapes have to agree.
        local rows, plays = duel.rows_from_moves(log, { ctx_len = 16 })
        expect(#rows).to.equal(duel.HAND_SIZE)
        expect(#plays).to.equal(duel.HAND_SIZE)
        expect(plays[1].action).to.equal(log[1].action)
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
