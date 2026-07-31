-- guardian_duel/spec/guardian_duel_player_spec.lua
--
-- Package-level spec for the player side of the guardian duel rules —
-- the view, the encoding, the alphabet and the play-log corpus builder.
-- Run with `alc_pkg_test pkg="guardian_duel"` after `alc_pkg_link` has
-- registered the package. The `lust` globals are pre-loaded by the
-- runner.
--
-- The boss side is pinned by `guardian_duel_spec.lua`; what is pinned
-- here is everything the boss half must not be able to reach: a layout
-- of its own, an alphabet of its own, and a corpus builder that labels
-- logged moves instead of sampled states.

local describe, it, expect = lust.describe, lust.it, lust.expect

local duel = require("guardian_duel")

local CTX_LEN = 16

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

--- A transcript entry as the interactive session writes one.
---
--- The entry-level `revealed` flag follows the intent of the view, the
--- way the session writes the two: the flag describes the turn and the
--- field is what the model reads.
local function log_entry(view, action)
    return {
        turn = view.turn,
        boss = { note = "the boss half, which this path must not read" },
        player = view,
        boss_action = "c",
        player_action = action,
        revealed = view.intent ~= duel.NO_INTENT,
    }
end

--- Decode a padded token row back to the training line it carries.
local function row_text(row)
    local v = duel.player_vocab()
    local chars = {}
    for _, id in ipairs(row) do
        if id == v.pad_id then
            break
        end
        chars[#chars + 1] = v.to_char[id]
    end
    return table.concat(chars)
end

-- ─── Encoding ───────────────────────────────────────────────────────

describe("guardian_duel.player_encode", function()
    it("writes the six fields and the intent in layout order", function()
        -- An untouched board: the boss at full health three buckets from
        -- its first stagger, the player at full health on turn one with
        -- nothing on either of them and no reveal bought.
        expect(duel.player_encode(player_view())).to.equal("M0H9D3Y9T1S0-")
    end)

    it("is exactly thirteen chars", function()
        local encoded = duel.player_encode(player_view())
        expect(#encoded).to.equal(duel.PLAYER_ENCODED_LEN)
        expect(#encoded).to.equal(13)
    end)

    it("keeps view plus move inside the tiny preset context", function()
        -- Thirteen chars, the separator, the move and the newline: the
        -- sixteen tokens the tiny preset has room for, exactly. This is
        -- why the intent has no letter of its own — a seventh label
        -- would put the line outside the window.
        local line = duel.player_encode(player_view()) .. ">a\n"
        expect(#duel.player_to_ids(line)).to.equal(16)
        expect(#duel.player_to_ids(line)).to.equal(duel.CTX_BUDGET)
    end)

    it("writes the boss answer a poke revealed", function()
        -- The whole point of the field: the same board, once with the
        -- slam spelled out and once with it left to inference.
        expect(duel.player_encode(player_view({ mode = 1, intent = "t" }))).to.equal(
            "M1H9D3Y9T1S0t"
        )
        expect(duel.player_encode(player_view({ mode = 1 }))).to.equal("M1H9D3Y9T1S0-")
    end)

    it("spells every boss answer the reveal can carry", function()
        for _, move in ipairs({ "c", "f", "v", "w", "d", "t" }) do
            local encoded = duel.player_encode(player_view({ intent = move }))
            expect(#encoded).to.equal(duel.PLAYER_ENCODED_LEN)
            expect(encoded:sub(-1)).to.equal(move)
        end
    end)

    it("rejects an intent that is not a boss move", function()
        -- A player letter here would be the human's own move recorded as
        -- the answer they were shown.
        expect(function()
            duel.player_encode(player_view({ intent = "a" }))
        end).to.fail()
        expect(function()
            duel.player_encode(player_view({ intent = "cf" }))
        end).to.fail()
        expect(function()
            duel.player_encode(player_view({ intent = 1 }))
        end).to.fail()
    end)

    it("names the missing field when the view predates the intent", function()
        -- A view without the field is a log recorded by an older
        -- session, and the fix is to record a new one rather than to
        -- guess what the board showed, so the message says so.
        local view = player_view()
        view.intent = nil
        local ok, err = pcall(duel.player_encode, view)
        expect(ok).to.equal(false)
        expect(err:match("view%.intent") ~= nil).to.equal(true)
    end)

    it("keeps bucket zero for an exhausted side", function()
        expect(duel.player_encode(player_view({ hp = 0 })):match("Y(%d)")).to.equal("0")
        expect(duel.player_encode(player_view({ hp = 1 })):match("Y(%d)")).to.equal("1")
        expect(duel.player_encode(player_view({ boss_hp = 0 })):match("H(%d)")).to.equal("0")
        expect(duel.player_encode(player_view({ boss_hp = 1 })):match("H(%d)")).to.equal("1")
    end)

    it("keeps distance zero for a stagger that is due", function()
        -- `D0` is the shift condition itself, so a remaining sliver has
        -- to round up rather than read as "it rolls up now".
        expect(duel.player_encode(player_view({ shift_distance = 0 })):match("D(%d)")).to.equal("0")
        expect(duel.player_encode(player_view({ shift_distance = 1 })):match("D(%d)")).to.equal("1")
    end)

    it("folds the three status flags into one digit", function()
        expect(duel.player_encode(player_view({ weakened = true }))).to.equal("M0H9D3Y9T1S1-")
        expect(duel.player_encode(player_view({ exposed = true }))).to.equal("M0H9D3Y9T1S2-")
        expect(duel.player_encode(player_view({ spikes = true }))).to.equal("M0H9D3Y9T1S4-")
        local all = player_view({ weakened = true, exposed = true, spikes = true })
        expect(duel.player_encode(all)).to.equal("M0H9D3Y9T1S7-")
    end)

    it("keeps every flag combination inside one char", function()
        for _, weakened in ipairs({ false, true }) do
            for _, exposed in ipairs({ false, true }) do
                for _, spikes in ipairs({ false, true }) do
                    local encoded = duel.player_encode(player_view({
                        weakened = weakened,
                        exposed = exposed,
                        spikes = spikes,
                    }))
                    expect(#encoded).to.equal(duel.PLAYER_ENCODED_LEN)
                end
            end
        end
    end)

    it("writes the rolled-up mode", function()
        expect(duel.player_encode(player_view({ mode = 1 })):match("M(%d)")).to.equal("1")
    end)

    it("writes the last turn of the fight", function()
        expect(duel.player_encode(player_view({ turn = duel.TURN_LIMIT }))).to.equal(
            "M0H9D3Y9T9S0-"
        )
    end)

    it("rejects a turn outside the fight", function()
        -- Turn ten would need two chars and blow the thirteen-char
        -- layout.
        expect(function()
            duel.player_encode(player_view({ turn = duel.TURN_LIMIT + 1 }))
        end).to.fail()
        expect(function()
            duel.player_encode(player_view({ turn = 0 }))
        end).to.fail()
    end)

    it("rejects a mode the rules cannot produce", function()
        expect(function()
            duel.player_encode(player_view({ mode = 2 }))
        end).to.fail()
    end)

    it("rejects health outside either side's range", function()
        expect(function()
            duel.player_encode(player_view({ hp = duel.PLAYER_MAX_HP + 1 }))
        end).to.fail()
        expect(function()
            duel.player_encode(player_view({ boss_hp = -1 }))
        end).to.fail()
    end)

    it("rejects a negative distance", function()
        expect(function()
            duel.player_encode(player_view({ shift_distance = -1 }))
        end).to.fail()
    end)

    it("rejects a flag that is not a boolean", function()
        -- `weakened = "no"` would read as true and flip a flag the log
        -- says was down.
        expect(function()
            duel.player_encode(player_view({ weakened = "no" }))
        end).to.fail()
        expect(function()
            duel.player_encode(player_view({ spikes = 1 }))
        end).to.fail()
    end)

    it("rejects a view missing a field", function()
        local view = player_view()
        view.exposed = nil
        expect(function()
            duel.player_encode(view)
        end).to.fail()
    end)

    it("rejects a view that is not a table", function()
        expect(function()
            duel.player_encode("M0H9D3Y9T1S0-")
        end).to.fail()
    end)

    it("carries no field the player cannot see", function()
        -- The boss cycle index is the field that was left out: the board
        -- never shows it, and the poke exists so that look can be bought
        -- for a turn. Two boards that differ only in the boss script
        -- position therefore encode alike, which is the intended shape
        -- rather than a collision.
        local g = duel.new_game(1)
        local head = duel.player_view(g, "guardian")
        local moved = duel.apply(g, "b", "c")
        local later = duel.player_view(moved, "guardian")
        expect(moved.boss.cycle).to.equal(1)
        expect(g.boss.cycle).to.equal(0)
        -- Only the turn separates them, so the cycle really is absent.
        expect(duel.player_encode(head)).to.equal("M0H9D3Y9T1S0-")
        expect(duel.player_encode(later)).to.equal("M0H9D3Y9T2S0-")
    end)
end)

-- ─── The view ───────────────────────────────────────────────────────

describe("guardian_duel.player_view", function()
    it("reads the board an opening fight shows", function()
        local view = duel.player_view(duel.new_game(3), "guardian")
        expect(view.turn).to.equal(1)
        expect(view.mode).to.equal(0)
        expect(view.boss_hp).to.equal(duel.BOSS_MAX_HP)
        expect(view.hp).to.equal(duel.PLAYER_MAX_HP)
        expect(view.shift_distance).to.equal(duel.threshold_damage("guardian", 0))
        expect(view.weakened).to.equal(false)
        expect(view.exposed).to.equal(false)
        expect(view.spikes).to.equal(false)
        -- Nothing has been bought on the opening turn, so the field says
        -- the board showed no answer rather than carrying one.
        expect(view.intent).to.equal(duel.NO_INTENT)
    end)

    it("carries the answer the poke of the previous turn bought", function()
        local g = duel.apply(duel.new_game(3), "p", "c")
        expect(g.revealed).to.equal(true)
        local answer = duel.policy_guardian(g.boss)
        expect(duel.player_view(g, "guardian", answer).intent).to.equal(answer)
    end)

    it("refuses a revealed turn built without the answer", function()
        -- The board showed the human a move here. A view that says
        -- nothing was shown would train the model on a question the
        -- player never faced.
        local g = duel.apply(duel.new_game(3), "p", "c")
        local ok, err = pcall(duel.player_view, g, "guardian")
        expect(ok).to.equal(false)
        expect(err:match("poke") ~= nil).to.equal(true)
        expect(function()
            duel.player_view(g, "guardian", duel.NO_INTENT)
        end).to.fail()
    end)

    it("refuses an answer on a turn no poke bought a look at", function()
        -- The other direction of the same rule: the encoding may not
        -- carry a value the board did not show.
        expect(function()
            duel.player_view(duel.new_game(3), "guardian", "c")
        end).to.fail()
        local g = duel.apply(duel.new_game(3), "a", "c")
        expect(g.revealed).to.equal(false)
        expect(function()
            duel.player_view(g, "guardian", "f")
        end).to.fail()
    end)

    it("refuses an answer the boss could not play from here", function()
        -- The reveal is the answer to this very turn, so it has to be
        -- one of the moves the position allows: the slam needs the boss
        -- rolled up, and one promised from a cycle turn came from a
        -- position other than the one being viewed.
        local g = duel.apply(duel.new_game(3), "p", "c")
        expect(g.boss.mode).to.equal(0)
        expect(function()
            duel.player_view(g, "guardian", "t")
        end).to.fail()
    end)

    it("refuses an intent that is not a boss move at all", function()
        local g = duel.apply(duel.new_game(3), "p", "c")
        expect(function()
            duel.player_view(g, "guardian", "a")
        end).to.fail()
        expect(function()
            duel.player_view(g, "guardian", 3)
        end).to.fail()
    end)

    it("measures the distance against the style it is given", function()
        local g = duel.new_game(3)
        expect(duel.player_view(g, "turtle").shift_distance).to.equal(
            duel.threshold_damage("turtle", 0)
        )
        expect(duel.player_view(g, "rusher").shift_distance).to.equal(
            duel.threshold_damage("rusher", 0)
        )
    end)

    it("raises the exposure flag after a heavy attack", function()
        local g = duel.apply(duel.new_game(3), "A", "c")
        local view = duel.player_view(g, "guardian")
        expect(view.exposed).to.equal(true)
        expect(view.boss_hp).to.equal(duel.BOSS_MAX_HP - 9)
    end)

    it("drops the exposure flag on any other move", function()
        local g = duel.apply(duel.apply(duel.new_game(3), "A", "c"), "b", "f")
        expect(duel.player_view(g, "guardian").exposed).to.equal(false)
    end)

    it("raises the weakened flag after the vent", function()
        local g = duel.apply(duel.new_game(3), "b", "v")
        expect(duel.player_view(g, "guardian").weakened).to.equal(true)
    end)

    it("raises the spikes flag with the mode shift", function()
        local g = duel.apply(duel.new_game(3), "b", "d")
        local view = duel.player_view(g, "guardian")
        expect(view.spikes).to.equal(true)
        expect(view.mode).to.equal(1)
    end)

    it("shows the shrinking distance as the boss takes damage", function()
        local g = duel.apply(duel.new_game(3), "a", "f")
        expect(duel.player_view(g, "guardian").shift_distance).to.equal(
            duel.threshold_damage("guardian", 0) - 4
        )
    end)

    it("rejects a position past the turn limit", function()
        local g = duel.new_game(3)
        for _ = 1, duel.TURN_LIMIT do
            g = duel.apply(g, "b", duel.policy_guardian(g.boss))
        end
        -- The fight is over, so there is no board to play from; the
        -- fields have left the range the encoding covers.
        expect(function()
            duel.player_view(g, "guardian")
        end).to.fail()
    end)

    it("rejects an unknown or missing style", function()
        expect(function()
            duel.player_view(duel.new_game(3), "berserker")
        end).to.fail()
        expect(function()
            duel.player_view(duel.new_game(3))
        end).to.fail()
    end)

    it("rejects a game that carries no seats", function()
        expect(function()
            duel.player_view({ turn = 1 }, "guardian")
        end).to.fail()
    end)
end)

-- ─── Vocabulary ─────────────────────────────────────────────────────

describe("guardian_duel.player_vocab", function()
    it("fits the tiny preset vocabulary", function()
        local v = duel.player_vocab()
        expect(v.size).to.equal(30)
        expect(v.size <= 64).to.equal(true)
        expect(v.pad_id).to.equal(0)
    end)

    it("maps every char back to itself", function()
        local v = duel.player_vocab()
        for ch, id in pairs(v.to_id) do
            expect(v.to_char[id]).to.equal(ch)
        end
    end)

    it("carries the four player moves", function()
        local v = duel.player_vocab()
        for _, move in ipairs(duel.player_legal_actions()) do
            expect(type(v.to_id[move])).to.equal("number")
        end
    end)

    it("carries the boss moves the intent field spells", function()
        -- They are in the alphabet for the intent field alone. Spelling
        -- a revealed answer as anything other than the move's own letter
        -- would give the model a second name for a move it already has
        -- one for.
        local v = duel.player_vocab()
        for _, move in ipairs({ "c", "f", "v", "w", "d", "t" }) do
            expect(type(v.to_id[move])).to.equal("number")
        end
        expect(type(v.to_id[duel.NO_INTENT])).to.equal("number")
    end)

    it("stays separate from the boss alphabet in both directions", function()
        -- The two are still different id spaces: the boss alphabet has
        -- no player move and no player field letter, and the player one
        -- has neither of the boss field letters, so a line from either
        -- corpus fails loudly against the other tokeniser.
        local boss = duel.vocab()
        for _, move in ipairs(duel.player_legal_actions()) do
            expect(boss.to_id[move]).to.equal(nil)
        end
        expect(boss.to_id["Y"]).to.equal(nil)
        expect(boss.to_id["S"]).to.equal(nil)
        expect(boss.to_id[duel.NO_INTENT]).to.equal(nil)
        local player = duel.player_vocab()
        expect(player.to_id["C"]).to.equal(nil)
        expect(player.to_id["L"]).to.equal(nil)
        expect(function()
            duel.to_ids("a")
        end).to.fail()
        expect(function()
            duel.player_to_ids("C0M0H9D3L0T1>c\n")
        end).to.fail()
        expect(function()
            duel.to_ids("M0H9D3Y9T1S0->b\n")
        end).to.fail()
    end)

    it("hands out fresh copies", function()
        local v = duel.player_vocab()
        local id = v.to_id["a"]
        v.to_id["a"] = 99
        expect(duel.player_vocab().to_id["a"]).to.equal(id)
    end)

    it("rejects a char outside the alphabet", function()
        expect(function()
            duel.player_to_ids("M0H9D3Y9T1S0>x\n")
        end).to.fail()
    end)

    it("rejects text that is not a string", function()
        expect(function()
            duel.player_to_ids(42)
        end).to.fail()
    end)
end)

-- ─── Corpus from a play log ─────────────────────────────────────────

describe("guardian_duel.rows_from_player_moves", function()
    -- Three positions a fight really reaches: the opening, a rolled-up
    -- boss whose slam the player poked for on the turn before, and a
    -- stagger that is due on the turn after a heavy attack. A fixture
    -- that could not occur would be copied into the next spec, so the
    -- numbers here hang together.
    local moves = {
        log_entry(player_view(), "a"),
        log_entry(
            player_view({
                turn = 5,
                mode = 1,
                spikes = true,
                boss_hp = 16,
                shift_distance = 0,
                hp = 30,
                intent = "t",
            }),
            "b"
        ),
        log_entry(
            player_view({
                turn = 4,
                boss_hp = 30,
                shift_distance = 0,
                exposed = true,
                hp = 33,
            }),
            "A"
        ),
    }

    it("writes one row per logged move", function()
        local rows, plays = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN })
        expect(#rows).to.equal(3)
        expect(#plays).to.equal(3)
    end)

    it("fills the context window, and pads a wider one", function()
        local v = duel.player_vocab()
        local rows = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN })
        for _, row in ipairs(rows) do
            -- With the intent field the line is exactly the window, so
            -- there is nothing left to pad and the row ends on the
            -- newline that closes it.
            expect(#row).to.equal(CTX_LEN)
            expect(row[#row]).to.equal(v.to_id["\n"])
        end
        local wider = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN + 2 })
        for _, row in ipairs(wider) do
            expect(#row).to.equal(CTX_LEN + 2)
            expect(row[#row]).to.equal(v.pad_id)
        end
    end)

    it("writes the view, the separator and the move that was played", function()
        local rows = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN })
        expect(row_text(rows[1])).to.equal("M0H9D3Y9T1S0->a\n")
        -- The second entry is the reveal doing its work: the slam was
        -- bought on the turn before and the block is the answer to
        -- having seen it.
        expect(row_text(rows[2])).to.equal("M1H4D0Y6T5S4t>b\n")
        expect(row_text(rows[3])).to.equal("M0H6D0Y7T4S2->A\n")
    end)

    it("returns the pairs behind the rows", function()
        local _, plays = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN })
        expect(plays[2].action).to.equal("b")
        expect(plays[2].view.turn).to.equal(5)
        expect(plays[2].view.spikes).to.equal(true)
        -- The copy carries the reveal too, so a caller replaying the log
        -- through a Card asks the same question the row trained.
        expect(plays[2].view.intent).to.equal("t")
        expect(plays[1].view.intent).to.equal(duel.NO_INTENT)
    end)

    it("hands out views the caller cannot write back into", function()
        local _, plays = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN })
        plays[1].view.hp = 1
        expect(moves[1].player.hp).to.equal(duel.PLAYER_MAX_HP)
    end)

    it("takes the pad id from the player alphabet", function()
        local rows = duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN + 1, pad_id = 7 })
        expect(rows[1][#rows[1]]).to.equal(7)
    end)

    it("rejects a log recorded before the view carried an intent", function()
        -- The gap cannot be filled in: nothing here knows whether the
        -- board showed an answer on that turn, and the placeholder would
        -- teach the model that a reveal the human paid for said nothing.
        local view = player_view()
        view.intent = nil
        local broken = { log_entry(player_view(), "a"), log_entry(view, "b") }
        local ok, err = pcall(duel.rows_from_player_moves, broken, { ctx_len = CTX_LEN })
        expect(ok).to.equal(false)
        expect(err:match("move 2") ~= nil).to.equal(true)
        expect(err:match("view%.intent") ~= nil).to.equal(true)
    end)

    it("rejects a logged intent that is not a boss move", function()
        expect(function()
            duel.rows_from_player_moves({ log_entry(player_view({ intent = "a" }), "b") }, {
                ctx_len = CTX_LEN,
            })
        end).to.fail()
    end)

    it("rejects an entry that carries no player view", function()
        -- A transcript from a session that never recorded the player's
        -- side is a gap rather than a turn the player sat out.
        expect(function()
            duel.rows_from_player_moves({ { player_action = "a" } }, { ctx_len = CTX_LEN })
        end).to.fail()
    end)

    it("names the entry it rejected", function()
        local broken = {
            log_entry(player_view(), "a"),
            log_entry(player_view({ mode = 3 }), "a"),
        }
        local ok, err = pcall(duel.rows_from_player_moves, broken, { ctx_len = CTX_LEN })
        expect(ok).to.equal(false)
        expect(err:match("move 2") ~= nil).to.equal(true)
        expect(err:match("view%.mode") ~= nil).to.equal(true)
    end)

    it("rejects a move that is not one of the four", function()
        -- A boss letter here would be a transcript read from the wrong
        -- half of the entry.
        expect(function()
            duel.rows_from_player_moves({ log_entry(player_view(), "d") }, { ctx_len = CTX_LEN })
        end).to.fail()
        expect(function()
            duel.rows_from_player_moves({ log_entry(player_view(), nil) }, { ctx_len = CTX_LEN })
        end).to.fail()
    end)

    it("rejects an entry that is not a table", function()
        expect(function()
            duel.rows_from_player_moves({ "a" }, { ctx_len = CTX_LEN })
        end).to.fail()
    end)

    it("rejects an empty log", function()
        expect(function()
            duel.rows_from_player_moves({}, { ctx_len = CTX_LEN })
        end).to.fail()
    end)

    it("rejects a log that is not a table", function()
        expect(function()
            duel.rows_from_player_moves("a", { ctx_len = CTX_LEN })
        end).to.fail()
    end)

    it("rejects a context too small for the line", function()
        -- A truncated line teaches the model a state it can never be
        -- asked about at decode time.
        expect(function()
            duel.rows_from_player_moves(moves, { ctx_len = 10 })
        end).to.fail()
    end)

    it("rejects missing or malformed opts", function()
        expect(function()
            duel.rows_from_player_moves(moves)
        end).to.fail()
        expect(function()
            duel.rows_from_player_moves(moves, {})
        end).to.fail()
        expect(function()
            duel.rows_from_player_moves(moves, { ctx_len = CTX_LEN, pad_id = "zero" })
        end).to.fail()
    end)

    it("bakes a whole fight without leaving the context", function()
        local g = duel.new_game(11)
        local log = {}
        local script = { "a", "A", "b", "p", "a", "b", "A", "p", "b" }
        for i = 1, duel.TURN_LIMIT do
            local move = script[i]
            -- The teacher answers from the position at the head of the
            -- turn, so on the turn after a poke its answer is exactly
            -- what the board showed the player.
            local answer = duel.policy_guardian(g.boss)
            log[#log + 1] = {
                player = duel.player_view(g, "guardian", g.revealed and answer or nil),
                player_action = move,
            }
            g = duel.apply(g, move, answer)
        end
        local rows, plays = duel.rows_from_player_moves(log, { ctx_len = CTX_LEN })
        expect(#rows).to.equal(duel.TURN_LIMIT)
        expect(#plays).to.equal(duel.TURN_LIMIT)
        for _, row in ipairs(rows) do
            expect(#row).to.equal(CTX_LEN)
        end
        -- The script pokes on turns four and eight, so the two turns
        -- after them are the ones that carry an answer.
        expect(plays[1].view.intent).to.equal(duel.NO_INTENT)
        expect(plays[4].view.intent).to.equal(duel.NO_INTENT)
        expect(plays[5].view.intent ~= duel.NO_INTENT).to.equal(true)
        expect(plays[9].view.intent ~= duel.NO_INTENT).to.equal(true)
    end)
end)
