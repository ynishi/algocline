-- guardian_duel_interactive/spec/guardian_duel_interactive_spec.lua
--
-- Package-level spec for the interactive boss fight. Run with
-- `alc_pkg_test pkg="guardian_duel_interactive"` after `alc_pkg_link`
-- has registered `guardian_duel` and this package. The `lust` globals
-- are pre-loaded by the runner.
--
-- No model, no Card and no state file are touched: `guardian_duel_npc`
-- is replaced through `package.preload` by a stub that answers with
-- `guardian_duel.policy_guardian`, `alc.card` hands out the persona
-- Cards the test registers and `alc.state` is swapped for an in-memory
-- table the assertions can read back. Because the stub is the teacher,
-- every board below is exactly reproducible, so the answers, the health
-- totals and the revealed move are asserted rather than merely parsed.
--
-- What is left under test is the session lifecycle, the move
-- validation, the style and distance basis resolution, the transcript
-- and the poke: a revealed answer has to be the move the next turn
-- actually plays, and it has to be replayed rather than decoded twice.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("guardian_duel")

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
--- Empty by default, so a style outside `guardian_duel.STYLES` is a typo
--- unless a test says otherwise.
local persona_cards = {}

--- Alias lookups made so far, so the canonical path can be shown to
--- stay off the Card layer.
local alias_lookups = 0

alc.card = {
    get_by_alias = function(alias)
        alias_lookups = alias_lookups + 1
        return persona_cards[alias]
    end,
}

--- The last request the NPC package received, and how many it has had.
--- The decode count is what separates a replayed reveal from a second
--- decode of the same position.
local last_npc_state = nil
local last_npc_alias = nil
local last_npc_style = nil
local npc_calls = 0

package.preload["guardian_duel_npc"] = function()
    return {
        run = function(ctx)
            local req = alc.json_decode(ctx.task)
            npc_calls = npc_calls + 1
            last_npc_state = req.state
            last_npc_alias = ctx.card_alias
            last_npc_style = ctx.style
            -- The teacher itself, so the fight is reproducible: the
            -- state validation inside it also fails the test if the
            -- session ever sends an incomplete boss state.
            local action = duel.policy_guardian(req.state)
            return {
                result = string.format("action=%s legal=true raw_legal=true gated=false", action),
            }
        end,
    }
end

local fight = require("guardian_duel_interactive")

local KEY = "gameai_guardian:default"

local function reset()
    for key in pairs(store) do
        store[key] = nil
    end
    for alias in pairs(persona_cards) do
        persona_cards[alias] = nil
    end
    last_npc_state, last_npc_alias, last_npc_style = nil, nil, nil
    npc_calls = 0
end

--- Pin a persona Card the way `bake_guardian_persona.lua` leaves one.
local function pin_persona(name, basis)
    persona_cards["guardian_duel_npc_" .. name] = {
        card_id = "stub-card-" .. name,
        persona = basis and { basis_style = basis, prompt = "a stub boss" } or nil,
    }
end

-- ─── new ────────────────────────────────────────────────────────────

describe("guardian_duel_interactive new", function()
    it("opens a board on turn one", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(view.turn).to.equal(1)
        expect(view.status).to.equal("your_turn")
        expect(view.your_hp).to.equal(duel.PLAYER_MAX_HP)
        expect(view.boss_hp).to.equal(duel.BOSS_MAX_HP)
        expect(view.boss_mode).to.equal(0)
        expect(view.boss_shifts).to.equal(0)
        expect(#view.legal_actions).to.equal(4)
        expect(view.intent).to.equal(nil)
        expect(type(view.text)).to.equal("string")
        expect(view.result).to.equal(view.text)
    end)

    it("opens at the full distance to the first mode shift", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(view.shift_distance).to.equal(duel.threshold_damage("guardian", 0))
    end)

    it("explains every move it offers", function()
        reset()
        local view = fight.run({ action = "new", seed = 7 })
        for _, move in ipairs(view.legal_actions) do
            expect(type(view.move_help[move])).to.equal("string")
        end
    end)

    it("stores the session under the game id", function()
        reset()
        fight.run({ action = "new", seed = 7, game_id = "demo" })
        expect(store["gameai_guardian:demo"] ~= nil).to.equal(true)
        expect(store[KEY]).to.equal(nil)
    end)

    it("rejects a style that is neither canonical nor pinned", function()
        reset()
        expect(function()
            fight.run({ action = "new", style = "berserker", seed = 7 })
        end).to.fail()
    end)

    it("rejects a style that is not a string", function()
        reset()
        expect(function()
            fight.run({ action = "new", style = 7, seed = 7 })
        end).to.fail()
    end)

    it("rejects a seed that is not a number", function()
        reset()
        expect(function()
            fight.run({ action = "new", seed = "later" })
        end).to.fail()
    end)

    it("rejects an unknown action", function()
        reset()
        expect(function()
            fight.run({ action = "flee" })
        end).to.fail()
    end)
end)

-- ─── Styles and the distance basis ──────────────────────────────────

describe("guardian_duel_interactive canonical styles", function()
    it("defaults to the guardian style and its own basis", function()
        reset()
        local view = fight.run({ action = "new", seed = 7 })
        expect(view.style).to.equal("guardian")
        expect(view.style_kind).to.equal("canonical")
        expect(view.basis_style).to.equal("guardian")
    end)

    it("takes its distance basis from the style itself", function()
        reset()
        local view = fight.run({ action = "new", style = "turtle", seed = 7 })
        expect(view.basis_style).to.equal("turtle")
        expect(view.shift_distance).to.equal(duel.threshold_damage("turtle", 0))
    end)

    it("never asks the Card layer about a shipped style", function()
        reset()
        local before = alias_lookups
        for _, style in ipairs(duel.STYLES) do
            fight.run({ action = "new", style = style, seed = 7 })
        end
        -- A persona Card must not be able to shadow a shipped style, so
        -- the alias lookup stays behind the policy check.
        expect(alias_lookups).to.equal(before)
    end)

    it("refuses a basis that would contradict the style", function()
        reset()
        -- A canonical style is its own basis, so honouring the field
        -- would encode states against a threshold the teacher does not
        -- use while still reporting the style that was asked for.
        expect(function()
            fight.run({ action = "new", style = "guardian", basis_style = "turtle", seed = 7 })
        end).to.fail()
    end)
end)

describe("guardian_duel_interactive persona styles", function()
    it("accepts a style that exists only as a pinned Card", function()
        reset()
        pin_persona("stalker", "turtle")
        local view = fight.run({ action = "new", style = "stalker", seed = 7 })
        expect(view.style).to.equal("stalker")
        expect(view.style_kind).to.equal("persona")
        expect(view.status).to.equal("your_turn")
    end)

    it("borrows the distance basis recorded on the Card", function()
        reset()
        pin_persona("stalker", "turtle")
        local view = fight.run({ action = "new", style = "stalker", seed = 7 })
        expect(view.basis_style).to.equal("turtle")
        expect(view.shift_distance).to.equal(duel.threshold_damage("turtle", 0))
    end)

    it("decodes the boss seat through the persona alias and basis", function()
        reset()
        pin_persona("stalker", "turtle")
        fight.run({ action = "new", style = "stalker", seed = 7 })
        fight.run({ action = "play", move = "a" })
        expect(last_npc_alias).to.equal("guardian_duel_npc_stalker")
        expect(last_npc_style).to.equal("turtle")
    end)

    it("accepts an explicit basis over the one on the Card", function()
        reset()
        pin_persona("stalker", "turtle")
        local view = fight.run({
            action = "new",
            style = "stalker",
            basis_style = "rusher",
            seed = 7,
        })
        expect(view.basis_style).to.equal("rusher")
    end)

    it("rejects an explicit basis outside guardian_duel.STYLES", function()
        reset()
        pin_persona("stalker", "turtle")
        expect(function()
            fight.run({ action = "new", style = "stalker", basis_style = "berserker", seed = 7 })
        end).to.fail()
    end)

    it("refuses a persona Card that records no basis", function()
        reset()
        pin_persona("stalker", nil)
        -- Guessing a basis here would feed the model a distance its
        -- corpus never carried while every answer stayed legal.
        local ok, err = pcall(fight.run, { action = "new", style = "stalker", seed = 7 })
        expect(ok).to.equal(false)
        expect(err:match("basis_style") ~= nil).to.equal(true)
    end)

    it("refuses a style with neither a policy nor a Card", function()
        reset()
        local ok, err = pcall(fight.run, { action = "new", style = "stalker", seed = 7 })
        expect(ok).to.equal(false)
        -- The message has to name both places that were searched, or a
        -- caller cannot tell a misspelt style from an unbaked persona.
        expect(err:match("canonical:") ~= nil).to.equal(true)
        expect(err:match("guardian_duel_npc_stalker") ~= nil).to.equal(true)
    end)

    it("refuses a persona style when the Card layer cannot answer", function()
        reset()
        local card_ns = alc.card
        alc.card = nil
        local ok, err = pcall(fight.run, { action = "new", style = "stalker", seed = 7 })
        alc.card = card_ns
        -- Falling back to the default style here would seat a boss the
        -- caller never asked for, so the missing surface is reported.
        expect(ok).to.equal(false)
        expect(err:match("get_by_alias is unavailable") ~= nil).to.equal(true)
    end)
end)

-- ─── play ───────────────────────────────────────────────────────────

describe("guardian_duel_interactive play", function()
    it("applies both moves and reports the turn", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local view = fight.run({ action = "play", move = "a" })
        -- The teacher opens on the head of its cycle, which is the
        -- charge: no damage to the player, five block against the next
        -- attack.
        expect(view.last_turn.turn).to.equal(1)
        expect(view.last_turn.you).to.equal("a")
        expect(view.last_turn.boss).to.equal("c")
        expect(view.last_turn.revealed).to.equal(false)
        expect(view.boss_hp).to.equal(duel.BOSS_MAX_HP - 4)
        expect(view.your_hp).to.equal(duel.PLAYER_MAX_HP)
        expect(view.turn).to.equal(2)
    end)

    it("asks the boss about the position at the head of the turn", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "a" })
        expect(last_npc_state.cycle).to.equal(0)
        expect(last_npc_state.turn).to.equal(1)
        expect(last_npc_state.hp).to.equal(duel.BOSS_MAX_HP)
        -- The block and spike counters are engine bookkeeping and never
        -- reach the model, so they are left out of the request.
        expect(last_npc_state.block).to.equal(nil)
        expect(last_npc_state.thorns).to.equal(nil)
    end)

    it("closes the fight on the turn limit", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        for turn = 1, duel.TURN_LIMIT do
            expect(view.status).to.equal("your_turn")
            expect(view.turn).to.equal(turn)
            view = fight.run({ action = "play", move = "b" })
        end
        -- Nine turns of block deal the boss nothing, so it walks its
        -- cycle untouched and wins on the health buckets.
        expect(view.status).to.equal("finished")
        expect(view.winner).to.equal("boss")
        expect(view.boss_hp).to.equal(duel.BOSS_MAX_HP)
        expect(#view.legal_actions).to.equal(0)
        expect(view.shift_distance).to.equal(nil)
    end)

    it("shifts the boss once it has taken enough damage", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local view
        for _ = 1, 4 do
            view = fight.run({ action = "play", move = "A" })
        end
        -- Four heavy attacks are past the fifteen damage the teacher
        -- tolerates, so it drops the cycle and rolls up.
        expect(view.last_turn.boss).to.equal("d")
        expect(view.boss_mode).to.equal(1)
    end)

    it("rejects a move outside the four the rules allow", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(function()
            fight.run({ action = "play", move = "x" })
        end).to.fail()
    end)

    it("rejects a move that is not a string", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(function()
            fight.run({ action = "play", move = 1 })
        end).to.fail()
    end)

    it("rejects a move without an active fight", function()
        reset()
        expect(function()
            fight.run({ action = "play", move = "a" })
        end).to.fail()
    end)

    it("rejects a move after the fight is over", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        for _ = 1, duel.TURN_LIMIT do
            view = fight.run({ action = "play", move = "b" })
        end
        expect(view.status).to.equal("finished")
        expect(function()
            fight.run({ action = "play", move = "b" })
        end).to.fail()
    end)
end)

-- ─── The poke ───────────────────────────────────────────────────────

describe("guardian_duel_interactive poke", function()
    it("reveals the answer of the following turn", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local view = fight.run({ action = "play", move = "p" })
        -- The poke leaves the teacher two damage short of nothing, so
        -- the second entry of the cycle is what comes next.
        expect(view.intent).to.equal("f")
        expect(view.text:match("it will answer f") ~= nil).to.equal(true)
    end)

    it("leaves no reveal behind on any other move", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(fight.run({ action = "play", move = "a" }).intent).to.equal(nil)
    end)

    it("shows the reveal again without decoding a second time", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "p" })
        local before = npc_calls
        local view = fight.run({ action = "show" })
        expect(view.intent).to.equal("f")
        expect(npc_calls).to.equal(before)
    end)

    it("plays the move it revealed", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local revealed = fight.run({ action = "play", move = "p" }).intent
        local before = npc_calls
        local view = fight.run({ action = "play", move = "a" })
        -- A revealed answer is replayed rather than decoded again, so
        -- the board cannot promise one move and the fight play another.
        expect(view.last_turn.boss).to.equal(revealed)
        expect(view.last_turn.revealed).to.equal(true)
        expect(npc_calls).to.equal(before)
        expect(view.intent).to.equal(nil)
    end)

    it("keeps revealing while the player keeps poking", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "p" })
        local view = fight.run({ action = "play", move = "p" })
        expect(view.last_turn.revealed).to.equal(true)
        expect(view.intent).to.equal("v")
    end)

    it("refuses a reveal that belongs to another turn", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "p" })
        local session = store[KEY]
        session.intent.turn = 7
        store[KEY] = session
        -- Showing nothing here would drop a look the player spent a
        -- turn buying, so the mismatch is reported instead.
        expect(function()
            fight.run({ action = "show" })
        end).to.fail()
    end)
end)

-- ─── Transcript ─────────────────────────────────────────────────────

describe("guardian_duel_interactive move log", function()
    it("records the position the boss answered", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "a" })
        local log = fight.run({ action = "end" }).move_log
        expect(#log).to.equal(1)
        expect(log[1].turn).to.equal(1)
        -- The entry is the position before the turn was applied, which
        -- is the only one the boss was ever asked about.
        expect(log[1].boss.cycle).to.equal(0)
        expect(log[1].boss.hp).to.equal(duel.BOSS_MAX_HP)
        expect(log[1].boss.damage_since_shift).to.equal(0)
        expect(log[1].boss_action).to.equal("c")
        expect(log[1].player_action).to.equal("a")
        expect(log[1].revealed).to.equal(false)
    end)

    it("appends one entry per turn, in order", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        for _ = 1, duel.TURN_LIMIT do
            view = fight.run({ action = "play", move = "b" })
        end
        expect(view.status).to.equal("finished")
        local log = fight.run({ action = "end" }).move_log
        expect(#log).to.equal(duel.TURN_LIMIT)
        for turn, entry in ipairs(log) do
            expect(entry.turn).to.equal(turn)
            expect(entry.boss.turn).to.equal(turn)
        end
    end)

    it("marks the turns the player had already seen", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "p" })
        fight.run({ action = "play", move = "a" })
        local log = fight.run({ action = "end" }).move_log
        expect(log[1].revealed).to.equal(false)
        expect(log[2].revealed).to.equal(true)
    end)

    it("hands out a copy the caller cannot write back into", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "play", move = "a" })
        local stored = store[KEY].move_log
        local view = fight.run({ action = "end" })
        view.move_log[1].boss.hp = 1
        view.move_log[1].boss_action = "t"
        expect(stored[1].boss.hp).to.equal(duel.BOSS_MAX_HP)
        expect(stored[1].boss_action).to.equal("c")
    end)

    it("returns an empty log for a fight that took no turn", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local log = fight.run({ action = "end" }).move_log
        expect(type(log)).to.equal("table")
        expect(#log).to.equal(0)
    end)

    it("is only handed out by end", function()
        reset()
        local opened = fight.run({ action = "new", style = "guardian", seed = 7 })
        expect(opened.move_log).to.equal(nil)
        expect(fight.run({ action = "play", move = "a" }).move_log).to.equal(nil)
        expect(fight.run({ action = "show" }).move_log).to.equal(nil)
        expect(#fight.run({ action = "end" }).move_log).to.equal(1)
    end)
end)

-- ─── show / end ─────────────────────────────────────────────────────

describe("guardian_duel_interactive show", function()
    it("returns the current board without moving", function()
        reset()
        local opened = fight.run({ action = "new", style = "guardian", seed = 7 })
        local shown = fight.run({ action = "show" })
        expect(shown.turn).to.equal(opened.turn)
        expect(shown.text).to.equal(opened.text)
    end)

    it("rejects a show without an active fight", function()
        reset()
        expect(function()
            fight.run({ action = "show" })
        end).to.fail()
    end)
end)

describe("guardian_duel_interactive end", function()
    it("drops the session and reports the final board", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        local view = fight.run({ action = "end" })
        expect(view.ended).to.equal(true)
        expect(view.result).to.equal(view.text)
        expect(store[KEY]).to.equal(nil)
    end)

    it("leaves other sessions alone", function()
        reset()
        fight.run({ action = "new", style = "guardian", seed = 7 })
        fight.run({ action = "new", style = "guardian", seed = 7, game_id = "other" })
        fight.run({ action = "end" })
        expect(store[KEY]).to.equal(nil)
        expect(store["gameai_guardian:other"] ~= nil).to.equal(true)
    end)

    it("carries the winner once the fight is over", function()
        reset()
        local view = fight.run({ action = "new", style = "guardian", seed = 7 })
        for _ = 1, duel.TURN_LIMIT do
            view = fight.run({ action = "play", move = "b" })
        end
        local ended = fight.run({ action = "end" })
        expect(ended.status).to.equal("finished")
        expect(ended.winner).to.equal(view.winner)
    end)

    it("rejects an end without an active fight", function()
        reset()
        expect(function()
            fight.run({ action = "end" })
        end).to.fail()
    end)
end)
