-- othello6/spec/othello6_spec.lua
--
-- Package-level spec for the 6x6 Othello rules. Run with
-- `alc_pkg_test pkg="othello6"` after `alc_pkg_link` has registered the
-- package, or through the `mlua-probe` test runner with
-- `examples/gameai` on the search path. The `lust` globals are
-- pre-loaded by both runners.
--
-- Everything the later phases stand on is pinned here: the opening, the
-- eight bracket directions, the pass, the two endings, the winner, the
-- move sequence a position carries and the corpus shape. A rule that
-- quietly changed would invalidate every measurement taken on top of
-- it, so the rules are fixed before the teacher is written.
--
-- The encoding changed after the first round of training: a position is
-- now written as the moves that reached it rather than as a board. The
-- rule cases below are untouched by that — a rule is a rule whichever
-- way the position is spelled — while the encoding and corpus cases are
-- written against the sequence.
--
-- Every line opens with `othello6.BOS`. The cases below fix that it is
-- always there, that it is a token of its own rather than the padding,
-- and that the opening therefore encodes to something a decode session
-- can be opened over. Without it the opening is the empty line, which
-- the bridge refuses, and no row holds the position the first move is
-- played from.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- The module reaches for the host RNG. The LCG below matches the shape
-- the engine-level harness installs (see
-- `crates/algocline-engine/tests/lua/card_duel_rules_test.lua`); the
-- properties under test hold for any stream, so the stub does not weaken
-- them. A real `alc.math` is left alone when one is present.
_G.alc = _G.alc or {}
alc.math = alc.math or {}
alc.math.rng_create = alc.math.rng_create
    or function(seed)
        return { state = math.floor(seed) % 2147483647 }
    end
alc.math.rng_int = alc.math.rng_int
    or function(rng, min, max)
        rng.state = (rng.state * 1103515245 + 12345) % 2147483648
        return min + (rng.state // 65536) % (max - min + 1)
    end

local othello = require("othello6")

local CTX_LEN = othello.CTX_BUDGET
local EMPTY_ROWS = { "......", "......", "......", "......", "......", "......" }

--- Absolute rendering of a position, for assertions that read like a
--- board rather than like an encoding.
local function board_rows(state)
    local rows = {}
    for r = 0, othello.BOARD_SIZE - 1 do
        local chars = {}
        for c = 0, othello.BOARD_SIZE - 1 do
            local cell = state.board[r * othello.BOARD_SIZE + c + 1]
            if cell == othello.EMPTY then
                chars[c + 1] = "."
            elseif cell == othello.BLACK then
                chars[c + 1] = "B"
            else
                chars[c + 1] = "W"
            end
        end
        rows[r + 1] = table.concat(chars)
    end
    return rows
end

local function board_text(state)
    return table.concat(board_rows(state), "/")
end

--- A board of `.` with the listed squares overwritten. Each entry is
--- `{row, col, char}`.
local function rows_with(cells)
    local rows = {}
    for i, line in ipairs(EMPTY_ROWS) do
        rows[i] = line
    end
    for _, cell in ipairs(cells) do
        local r, c, ch = cell[1], cell[2], cell[3]
        local line = rows[r + 1]
        rows[r + 1] = line:sub(1, c) .. ch .. line:sub(c + 2)
    end
    return rows
end

local function joined(list)
    return table.concat(list, ",")
end

local function contains(list, wanted)
    for _, item in ipairs(list) do
        if item == wanted then
            return true
        end
    end
    return false
end

--- Decode a padded token row back to the line it carries: the opening
--- marker and the move sequence after it.
local function row_text(row)
    local v = othello.vocab()
    local chars = {}
    for _, id in ipairs(row) do
        if id == v.pad_id then
            break
        end
        chars[#chars + 1] = v.to_char[id]
    end
    return table.concat(chars)
end

--- Play an encoded line out from the opening.
---
--- The leading `BOS` is asserted rather than skipped past: a line that
--- lost it is a line no training row has the shape of, and every caller
--- of this helper hands in something that came out of `encode` or out of
--- a corpus row.
---
--- Every move after it is checked against the legal set of the position
--- it is played into, so a line that replays at all carries legal moves
--- only — which is what a corpus row has to be for the model trained on
--- it to be learning Othello.
local function replay(sequence)
    expect(sequence:sub(1, 1)).to.equal(othello.BOS)
    local state = othello.new_game(1)
    for i = 2, #sequence do
        local move = sequence:sub(i, i)
        expect(contains(othello.legal_actions(state), move)).to.equal(true)
        state = othello.apply(state, move)
    end
    return state
end

--- Play one game out with a uniform random legal player.
local function random_game(seed)
    local rng = alc.math.rng_create(seed * 7919 + 13)
    local random_move = othello.policy_random(rng)
    local state = othello.new_game(seed)
    while not othello.is_over(state) do
        state = othello.apply(state, random_move(state))
    end
    return state
end

--- Deterministic policy: the legal move with the lowest square index.
--- Used where the corpus has to be reproducible and the choice itself
--- does not matter.
local function first_legal(state)
    return othello.legal_actions(state)[1]
end

-- ─── Opening ────────────────────────────────────────────────────────

describe("othello6.new_game", function()
    it("opens with the four centre squares filled", function()
        local g = othello.new_game(1)
        expect(board_text(g)).to.equal("....../....../..WB../..BW../....../......")
    end)

    it("opens with black to move and nothing passed yet", function()
        local g = othello.new_game(1)
        expect(othello.side_to_move(g)).to.equal("black")
        expect(g.passes).to.equal(0)
        expect(g.ply).to.equal(0)
        expect(g.seed).to.equal(1)
    end)

    it("opens with an empty move history", function()
        local g = othello.new_game(1)
        expect(#g.moves).to.equal(0)
    end)

    it("offers black exactly the four opening moves", function()
        local g = othello.new_game(1)
        -- (1,2) (2,1) (3,4) (4,3), in square order.
        expect(joined(othello.legal_actions(g))).to.equal("i,n,y,E")
    end)

    it("rejects a seed that is not a number", function()
        expect(function()
            othello.new_game("1")
        end).to.fail()
    end)
end)

-- ─── Flipping ───────────────────────────────────────────────────────

describe("othello6.apply", function()
    it("flips along every one of the eight directions", function()
        local directions = {
            { -1, -1 },
            { -1, 0 },
            { -1, 1 },
            { 0, -1 },
            { 0, 1 },
            { 1, -1 },
            { 1, 0 },
            { 1, 1 },
        }
        for _, dir in ipairs(directions) do
            local dr, dc = dir[1], dir[2]
            -- Black plays (2,2), brackets one white disc at (2+d) and
            -- anchors on its own disc at (2+2d).
            local rows = rows_with({
                { 2 + dr, 2 + dc, "W" },
                { 2 + 2 * dr, 2 + 2 * dc, "B" },
            })
            local state = othello.state_from_rows(rows, "black")
            local move = othello.action_of_index(2 * othello.BOARD_SIZE + 2)
            expect(contains(othello.legal_actions(state), move)).to.equal(true)
            local after = othello.apply(state, move)
            local board = board_rows(after)
            expect(board[3]:sub(3, 3)).to.equal("B")
            expect(board[2 + dr + 1]:sub(2 + dc + 1, 2 + dc + 1)).to.equal("B")
            expect(board[2 + 2 * dr + 1]:sub(2 + 2 * dc + 1, 2 + 2 * dc + 1)).to.equal("B")
            expect(othello.side_to_move(after)).to.equal("white")
        end
    end)

    it("flips two runs at once when a move brackets in two directions", function()
        -- Black plays (2,2): a run to the right and a run downwards.
        local rows = rows_with({
            { 2, 3, "W" },
            { 2, 4, "B" },
            { 3, 2, "W" },
            { 4, 2, "B" },
        })
        local state = othello.state_from_rows(rows, "black")
        local after = othello.apply(state, othello.action_of_index(14))
        expect(board_text(after)).to.equal("....../....../..BBB./..B.../..B.../......")
    end)

    it("leaves the position it was handed untouched", function()
        local g = othello.new_game(1)
        local before = board_text(g)
        othello.apply(g, "i")
        expect(board_text(g)).to.equal(before)
        expect(othello.side_to_move(g)).to.equal("black")
        -- Including its history: the move went onto the new position,
        -- not onto the one the caller still holds.
        expect(#g.moves).to.equal(0)
    end)

    it("records the move it played", function()
        local g = othello.new_game(1)
        local after = othello.apply(g, "i")
        expect(joined(after.moves)).to.equal("i")
        local later = othello.apply(after, othello.legal_actions(after)[1])
        expect(#later.moves).to.equal(2)
        expect(later.moves[1]).to.equal("i")
        expect(later.moves[2]).to.equal(othello.legal_actions(after)[1])
    end)

    it("rejects a square that brackets nothing", function()
        local g = othello.new_game(1)
        expect(function()
            othello.apply(g, "a")
        end).to.fail()
    end)

    it("rejects a character that is not a move", function()
        local g = othello.new_game(1)
        expect(function()
            othello.apply(g, "Z")
        end).to.fail()
    end)
end)

describe("othello6.legal_actions", function()
    it("leaves out empty squares that bracket nothing", function()
        local g = othello.new_game(1)
        local legal = othello.legal_actions(g)
        -- (0,0) and (1,1) are empty; neither brackets a white disc.
        expect(contains(legal, othello.action_of_index(0))).to.equal(false)
        expect(contains(legal, othello.action_of_index(7))).to.equal(false)
        expect(#legal).to.equal(4)
    end)

    it("leaves out occupied squares", function()
        local g = othello.new_game(1)
        local legal = othello.legal_actions(g)
        expect(contains(legal, othello.action_of_index(14))).to.equal(false)
        expect(contains(legal, othello.action_of_index(15))).to.equal(false)
    end)

    it("answers with the pass, never with an empty list", function()
        -- No white disc on the board, so black brackets nothing.
        local rows = rows_with({ { 0, 0, "B" }, { 0, 1, "B" } })
        local black = othello.state_from_rows(rows, "black")
        local white = othello.state_from_rows(rows, "white")
        expect(joined(othello.legal_actions(black))).to.equal(othello.PASS)
        expect(joined(othello.legal_actions(white))).to.equal(othello.PASS)
    end)
end)

-- ─── Pass and endings ───────────────────────────────────────────────

describe("othello6 pass", function()
    it("hands the turn over and is refused while a placement is legal", function()
        local rows = rows_with({ { 0, 0, "B" }, { 0, 1, "B" } })
        local state = othello.state_from_rows(rows, "black")
        local after = othello.apply(state, othello.PASS)
        expect(othello.side_to_move(after)).to.equal("white")
        expect(after.passes).to.equal(1)
        expect(othello.is_over(after)).to.equal(false)

        expect(function()
            othello.apply(othello.new_game(1), othello.PASS)
        end).to.fail()
    end)

    it("ends the game once both sides have passed in a row", function()
        local rows = rows_with({ { 0, 0, "B" }, { 0, 1, "B" } })
        local state = othello.state_from_rows(rows, "black")
        state = othello.apply(state, othello.PASS)
        state = othello.apply(state, othello.PASS)
        expect(state.passes).to.equal(2)
        expect(othello.is_over(state)).to.equal(true)
        expect(othello.winner(state)).to.equal("black")
    end)

    it("clears the pass count once a placement follows it", function()
        -- One side passed; the other has a placement, so the game is not
        -- two passes in a row and the count starts over.
        local passed = othello.new_game(1)
        passed.passes = 1
        local after = othello.apply(passed, "i")
        expect(after.passes).to.equal(0)
        expect(othello.is_over(after)).to.equal(false)
    end)
end)

describe("othello6.is_over", function()
    it("ends the game on a full board", function()
        local rows = { "BWBWBW", "WBWBWB", "BWBWBW", "WBWBWB", "BWBWBW", "WBWBWB" }
        local state = othello.state_from_rows(rows, "black")
        expect(othello.is_over(state)).to.equal(true)
    end)

    it("keeps the opening running", function()
        expect(othello.is_over(othello.new_game(1))).to.equal(false)
    end)
end)

describe("othello6.winner", function()
    it("gives the game to the side holding more discs", function()
        local black_rows = { "BBBBBB", "BBBBBB", "BBBBBB", "WWWWWW", "WWWWWW", "WWWWWB" }
        local white_rows = { "WWWWWW", "WWWWWW", "WWWWWW", "BBBBBB", "BBBBBB", "BBBBBW" }
        expect(othello.winner(othello.state_from_rows(black_rows, "black"))).to.equal("black")
        expect(othello.winner(othello.state_from_rows(white_rows, "black"))).to.equal("white")
    end)

    it("calls an equal count a draw", function()
        local rows = { "BWBWBW", "WBWBWB", "BWBWBW", "WBWBWB", "BWBWBW", "WBWBWB" }
        expect(othello.winner(othello.state_from_rows(rows, "black"))).to.equal("draw")
    end)

    it("has no answer while the game is running", function()
        expect(othello.winner(othello.new_game(1))).to.equal(nil)
    end)
end)

-- ─── Encoding ───────────────────────────────────────────────────────

describe("othello6.encode", function()
    it("writes the opening marker and nothing else at the opening", function()
        -- Not the empty string. An empty prompt is one the bridge
        -- refuses to open a session over ("alc.nn generate_session:
        -- prompt_tokens is empty"), so a Card could not be asked to open
        -- a game at all, and the corpus would carry no row holding the
        -- position the first move is played from.
        local opening = othello.encode(othello.new_game(1))
        expect(opening).to.equal(othello.BOS)
        expect(#opening).to.equal(1)
    end)

    it("opens every line with the marker", function()
        local rng = alc.math.rng_create(29)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(7)
        expect(othello.encode(state):sub(1, 1)).to.equal(othello.BOS)
        while not othello.is_over(state) do
            state = othello.apply(state, random_move(state))
            expect(othello.encode(state):sub(1, 1)).to.equal(othello.BOS)
        end
    end)

    it("writes the marker once and never again", function()
        -- It marks where a game begins, so a second one inside a line
        -- would mark a beginning that is not one. No move maps to it —
        -- the alphabet is disjoint — which is what makes the count fixed
        -- rather than merely usual.
        local state = random_game(3)
        local text = othello.encode(state)
        local count = 0
        for i = 1, #text do
            if text:sub(i, i) == othello.BOS then
                count = count + 1
            end
        end
        expect(count).to.equal(1)
    end)

    it("grows by one character per move", function()
        local rng = alc.math.rng_create(11)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(3)
        local played = 0
        while not othello.is_over(state) do
            local move = random_move(state)
            state = othello.apply(state, move)
            played = played + 1
            local text = othello.encode(state)
            expect(#text).to.equal(played + othello.BOS_LEN)
            expect(text:sub(-1)).to.equal(move)
        end
        expect(played > 0).to.equal(true)
    end)

    it("writes a pass like any other move", function()
        -- Black holds two discs and there is no white disc to bracket,
        -- so the pass is the only move.
        local rows = rows_with({ { 0, 0, "B" }, { 0, 1, "B" } })
        local state = othello.state_from_rows(rows, "black")
        expect(joined(othello.legal_actions(state))).to.equal(othello.PASS)
        local after = othello.apply(state, othello.PASS)
        expect(othello.encode(after)).to.equal(othello.BOS .. othello.PASS)
        expect(othello.encode(othello.apply(after, othello.PASS))).to.equal(
            othello.BOS .. othello.PASS .. othello.PASS
        )
    end)

    it("replays to the position it was taken from", function()
        for seed = 1, 8 do
            local state = random_game(seed)
            local replayed = replay(othello.encode(state))
            expect(board_text(replayed)).to.equal(board_text(state))
            expect(othello.side_to_move(replayed)).to.equal(othello.side_to_move(state))
            expect(replayed.passes).to.equal(state.passes)
            expect(othello.winner(replayed)).to.equal(othello.winner(state))
        end
    end)

    it("replays a game that contained a pass", function()
        -- A pass is part of Othello, so a sequence that dropped it could
        -- not be replayed: the reader would lose track of whose move
        -- each character is.
        local found = nil
        for seed = 1, 80 do
            local state = random_game(seed)
            if othello.encode(state):find(othello.PASS, 1, true) then
                found = state
                break
            end
        end
        expect(found ~= nil).to.equal(true)
        local sequence = othello.encode(found)
        local replayed = replay(sequence)
        expect(board_text(replayed)).to.equal(board_text(found))
        expect(othello.winner(replayed)).to.equal(othello.winner(found))
    end)

    it("replays every prefix of a game to the position it stood at", function()
        local rng = alc.math.rng_create(23)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(5)
        while not othello.is_over(state) do
            expect(board_text(replay(othello.encode(state)))).to.equal(board_text(state))
            state = othello.apply(state, random_move(state))
        end
    end)

    it("stays inside the move budget the context was sized for", function()
        for seed = 1, 40 do
            local state = random_game(seed)
            expect(#othello.encode(state) <= othello.ROW_LEN).to.equal(true)
            expect(#state.moves <= othello.MAX_MOVES).to.equal(true)
        end
    end)

    it("keeps a whole game inside the marker plus the longest game seen", function()
        -- The line is one token longer than the game it writes, so the
        -- budget the preset has to hold is `BOS` plus the longest game
        -- 300k playouts produced: 1 + 40 = 41, which the context of 48
        -- covers with room over. A future edit that made the marker
        -- longer, or the encoding wider, has to move this number.
        expect(othello.BOS_LEN + othello.LONGEST_OBSERVED_GAME).to.equal(41)
        expect(othello.BOS_LEN + othello.LONGEST_OBSERVED_GAME <= othello.CTX_BUDGET).to.equal(true)
        for seed = 1, 40 do
            local text = othello.encode(random_game(seed))
            expect(#text <= othello.BOS_LEN + othello.LONGEST_OBSERVED_GAME).to.equal(true)
        end
    end)

    it("gives the opening a prompt the bridge can open a session over", function()
        -- The bridge refuses `prompt_tokens` with nothing in it, so the
        -- number that matters is one rather than zero. This is the case
        -- the marker was added for.
        local opening = othello.to_ids(othello.encode(othello.new_game(1)))
        expect(#opening >= 1).to.equal(true)
        expect(#opening).to.equal(1)
        expect(opening[1]).to.equal(othello.vocab().to_id[othello.BOS])
    end)

    it("has only the marker to write for a position that was not played", function()
        local rows = { "......", "......", "..WB..", "..BW..", "......", "......" }
        expect(othello.encode(othello.state_from_rows(rows, "black"))).to.equal(othello.BOS)
    end)
end)

-- ─── Vocabulary ─────────────────────────────────────────────────────

describe("othello6.vocab", function()
    it("holds the forty-six characters the corpus is written in", function()
        -- Forty-five, and then the opening marker.
        local v = othello.vocab()
        expect(v.size).to.equal(46)
        expect(v.size).to.equal(othello.VOCAB_SIZE)
        expect(v.pad_id).to.equal(0)
        expect(v.to_char[v.pad_id]).to.equal("\0")
    end)

    it("gives the marker a token of its own, apart from the padding", function()
        -- Sharing the padding's id would put "open a game" and "the row
        -- ended here" on one target: the model would be trained to answer
        -- the first move from every position a row runs out at.
        local v = othello.vocab()
        local bos_id = v.to_id[othello.BOS]
        expect(bos_id).to_not.equal(nil)
        expect(bos_id).to_not.equal(v.pad_id)
        expect(v.to_char[bos_id]).to.equal(othello.BOS)
        expect(othello.BOS).to_not.equal("\0")
        expect(othello.BOS_LEN).to.equal(1)
        expect(#othello.BOS).to.equal(1)
    end)

    it("keeps the move alphabet disjoint from the board fields and the marker", function()
        local v = othello.vocab()
        local seen = {}
        for index = 0, othello.CELLS - 1 do
            local ch = othello.action_of_index(index)
            expect(seen[ch]).to.equal(nil)
            seen[ch] = true
            expect(othello.index_of_action(ch)).to.equal(index)
            expect(v.to_id[ch]).to_not.equal(nil)
        end
        for _, ch in ipairs({ ".", "x", "o", "B", "W", ">", "\n", othello.BOS }) do
            expect(seen[ch]).to.equal(nil)
        end
        expect(othello.index_of_action(othello.PASS)).to.equal(nil)
        expect(othello.BOS).to_not.equal(othello.PASS)
        -- The marker is not a move, so asking for its square is a
        -- question with no answer rather than one answered with a guess.
        expect(function()
            othello.index_of_action(othello.BOS)
        end).to.fail()
    end)

    it("maps a whole game to one id per character", function()
        local sequence = othello.encode(random_game(1))
        expect(#sequence > 0).to.equal(true)
        expect(#othello.to_ids(sequence)).to.equal(#sequence)
        -- The longest line the preset has to hold is the marker and one
        -- whole game.
        expect(othello.ROW_LEN).to.equal(othello.BOS_LEN + othello.MAX_MOVES)
        expect(othello.ROW_LEN).to.equal(othello.CTX_BUDGET)
        expect(othello.CTX_BUDGET).to.equal(48)
    end)

    it("keeps the budget above the longest game that has been observed", function()
        -- 300k random playouts topped out at 40 moves, which is a
        -- measurement and not a bound: a 41-move game is not ruled out by
        -- them, and a budget sitting exactly on the observation would
        -- turn one into a corpus run that dies partway through. The seven
        -- moves of headroom left after the marker took a token are the
        -- point of the number, so a future edit that spends them has to
        -- say so here.
        expect(othello.LONGEST_OBSERVED_GAME).to.equal(40)
        expect(othello.MAX_MOVES - othello.LONGEST_OBSERVED_GAME).to.equal(7)
        expect(othello.MAX_MOVES + othello.BOS_LEN).to.equal(othello.CTX_BUDGET)
    end)

    it("refuses a character outside the vocabulary", function()
        expect(function()
            othello.to_ids("Z")
        end).to.fail()
        expect(function()
            othello.to_ids("?")
        end).to.fail()
    end)
end)

-- ─── Corpus ─────────────────────────────────────────────────────────

describe("othello6.build_corpus", function()
    it("writes one row per game", function()
        for _, games in ipairs({ 1, 3, 7 }) do
            local rows =
                othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = games, seed = 5 })
            expect(#rows).to.equal(games)
        end
        expect(othello.ROWS_PER_GAME_ESTIMATE).to.equal(1)
    end)

    it("pads every row to the context window", function()
        local rows = othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 3, seed = 5 })
        for _, row in ipairs(rows) do
            expect(#row).to.equal(CTX_LEN)
        end
    end)

    it("opens every row with the marker", function()
        -- The position the first move is played from exists in the
        -- corpus only because the row starts one token before the game
        -- does.
        local v = othello.vocab()
        local rows = othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 6, seed = 5 })
        for _, row in ipairs(rows) do
            expect(row[1]).to.equal(v.to_id[othello.BOS])
            expect(row_text(row):sub(1, 1)).to.equal(othello.BOS)
        end
    end)

    it("keeps every row inside the marker plus the longest game seen", function()
        local rows = othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 12, seed = 5 })
        for _, row in ipairs(rows) do
            expect(#row_text(row) <= othello.BOS_LEN + othello.LONGEST_OBSERVED_GAME).to.equal(true)
        end
    end)

    it("writes the marker, the move sequence and then padding, and nothing else", function()
        local v = othello.vocab()
        local rows = othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 3, seed = 5 })
        for _, row in ipairs(rows) do
            local text = row_text(row)
            expect(#text > 1).to.equal(true)
            expect(text:sub(1, 1)).to.equal(othello.BOS)
            -- No separator and no line ending: every position in the row
            -- is a move the model is asked to predict.
            expect(text:find(">", 1, true)).to.equal(nil)
            expect(text:find("\n", 1, true)).to.equal(nil)
            -- And nothing after the marker is the marker again.
            for i = 2, #text do
                local move = text:sub(i, i)
                local is_move = move == othello.PASS or othello.index_of_action(move) ~= nil
                expect(is_move).to.equal(true)
            end
            -- The padding runs to the end once it starts.
            for i = #text + 1, #row do
                expect(row[i]).to.equal(v.pad_id)
            end
        end
    end)

    it("writes a row whose every token is legal in the position before it", function()
        local rows = othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 6, seed = 9 })
        for _, row in ipairs(rows) do
            -- `replay` checks each move against the legal set of the
            -- position it is played into and fails on the first one that
            -- is not there.
            local state = replay(row_text(row))
            -- And the row is a whole game, not a fragment of one.
            expect(othello.is_over(state)).to.equal(true)
        end
    end)

    it("does not repeat one game across seeds", function()
        local seen = {}
        local distinct = 0
        for seed = 1, 12 do
            local rows =
                othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 1, seed = seed })
            expect(#rows).to.equal(1)
            -- From the second character: the first is the marker, which
            -- every row shares.
            local opening = row_text(rows[1]):sub(2, 4)
            if not seen[opening] then
                seen[opening] = true
                distinct = distinct + 1
            end
        end
        -- The randomised prefix is what keeps these apart; without it
        -- every game would be the same deterministic playout and this
        -- count would be 1.
        expect(distinct >= 4).to.equal(true)
    end)

    it("builds a different corpus for a different seed", function()
        local function digest(seed)
            local rows =
                othello.build_corpus(first_legal, { ctx_len = CTX_LEN, games = 4, seed = seed })
            local parts = {}
            for _, row in ipairs(rows) do
                parts[#parts + 1] = row_text(row)
            end
            return table.concat(parts, "|")
        end
        expect(digest(1)).to_not.equal(digest(2))
    end)

    it("honours a zero-length random opening", function()
        local rows = othello.build_corpus(first_legal, {
            ctx_len = CTX_LEN,
            games = 2,
            seed = 4,
            random_opening_max = 0,
        })
        -- With no random prefix the policy plays both seats from the one
        -- opening, so the two games are the same game.
        expect(row_text(rows[1])).to.equal(row_text(rows[2]))
        expect(row_text(rows[1]):sub(1, 1)).to.equal(othello.BOS)
        expect(row_text(rows[1]):sub(2, 2)).to.equal(othello.legal_actions(othello.new_game(1))[1])
    end)

    it("rejects options it cannot size a corpus from", function()
        expect(function()
            othello.build_corpus(first_legal, { games = 1 })
        end).to.fail()
        expect(function()
            othello.build_corpus(first_legal, { ctx_len = CTX_LEN })
        end).to.fail()
        expect(function()
            othello.build_corpus("first_legal", { ctx_len = CTX_LEN, games = 1 })
        end).to.fail()
        expect(function()
            othello.build_corpus(first_legal, {
                ctx_len = CTX_LEN,
                games = 1,
                random_opening_max = -1,
            })
        end).to.fail()
    end)

    it("cannot fit a game into a context the preset would not offer", function()
        expect(function()
            othello.build_corpus(first_legal, { ctx_len = 8, games = 1 })
        end).to.fail()
    end)
end)

describe("othello6.policy_random", function()
    it("answers with a legal move", function()
        local rng = alc.math.rng_create(21)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(2)
        local plies = 0
        while not othello.is_over(state) do
            local move = random_move(state)
            expect(contains(othello.legal_actions(state), move)).to.equal(true)
            state = othello.apply(state, move)
            plies = plies + 1
            expect(plies < 200).to.equal(true)
        end
        expect(othello.winner(state)).to_not.equal(nil)
    end)

    it("names the styles the teacher package will implement", function()
        expect(joined(othello.STYLES)).to.equal("corner,mobility,greedy")
    end)
end)
