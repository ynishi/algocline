--- othello6 — 6x6 Othello, the board game used as the SLM training testbed
---
--- Standard Othello on a 6x6 board. No house rules are added: the point
--- of the game here is that it is settled, so a disagreement between a
--- Card and a teacher is a fact about the model rather than a fact about
--- a rule this package invented.
---
--- What the package owns is the rules, the projection of a position to
--- one line of text, and the corpus builder that turns a labelling
--- policy into training rows. The teacher (`othello6_teacher`) and the
--- decode surface (`othello6_npc`) live in their own packages and reach
--- for this one.
---
--- ## Usage
---
--- ```lua
--- local othello = require("othello6")
--- local g = othello.new_game(1)
--- local random = othello.policy_random(alc.math.rng_create(7))
--- while not othello.is_over(g) do
---     g = othello.apply(g, random(g))
--- end
--- print(othello.winner(g), othello.encode(g)) -- result, and the game as one line
--- ```
---
--- ## Rules
---
--- 1. `new_game(seed)` opens with the four centre squares filled —
---    `(2,2)` and `(3,3)` white, `(2,3)` and `(3,2)` black — and black
---    to move.
--- 2. A move is legal on an empty square that brackets at least one
---    opposing disc along one of the eight directions. Every bracketed
---    run flips.
--- 3. A side with no legal placement passes (`PASS`). Two passes in a
---    row end the game, as does a full board.
--- 4. The side holding more discs at the end wins; an equal count is a
---    draw.
--- 5. A position carries the moves that reached it in `state.moves`,
---    appended by `apply` and never rewritten in place.
---
--- ## Encoding
---
--- `encode` writes an opening marker and then the moves played so far,
--- one character each, and nothing else. The board is not handed to the
--- model: it has to build the position internally from the sequence,
--- which is the form every published Othello transformer is trained in
--- (Li et al., ICLR 2023, arXiv:2210.13382) and the form the linear
--- probes that found a board representation inside those models were run
--- against.
---
--- The marker (`BOS`) is what makes the opening a position like any
--- other. A line that began at the first move would leave the model
--- with no prompt to answer the opening from and no training position
--- that asked for a first move at all; the published models carry the
--- same reserved token for the same reason. It is a different character
--- from the padding, so "open a game" and "the row ended here" are two
--- targets rather than one.
---
--- The first version of this package wrote the 36 squares and the side
--- to move instead. Trained on that, the model reached 9.1% legal moves
--- [measured: two bakes], and the gap was located rather than guessed:
--- it had learnt where in a line a move belongs but not how to read
--- legality off a board. The board characters (`.` / `x` / `o`) and the
--- side-to-move characters (`B` / `W`) are therefore unused in this
--- version. They stay in the vocabulary because the move letters were
--- assigned around them, and moving the letters would rewrite the
--- teacher's spec to buy nothing.
---
--- A pass is a move like any other and is written into the sequence.
--- Leaving it out would make the line ambiguous — a reader could no
--- longer tell which side each move belongs to, and the position could
--- not be replayed from it.
---
--- ## Move alphabet (deviation from the design note)
---
--- A move is one character, because the decode surface constrains a
--- single sampled token against the legal set. Square `(r, c)` is index
--- `r * 6 + c`, and the index is written as the `index + 1`-th entry of
--- `ACTION_CHARS`; a pass is `PASS`.
---
--- The design note assigned `a`..`z` and `A`..`J` to the 36 indices and
--- counted a 45-character vocabulary. Those two statements contradict
--- each other: `x` and `o` are already the board characters and `B` is
--- already the black side-to-move marker, so that assignment collides on
--- three characters and leaves 42 distinct tokens, not 45. Rather than
--- let one token id mean both "my disc here" and "play square 23", the
--- move letters skip the four characters the other fields own (`o`, `x`,
--- `B`, `W`) and continue into the upper case letters. The mapping stays
--- monotone in the index, the vocabulary is the 45 the design asked for
--- plus the opening marker, and no caller hard-codes the letters —
--- `action_of_index` and `index_of_action` are the only places that know
--- them.
---
--- ## Host calls
---
--- `alc.math.rng_create` / `alc.math.rng_int` are the only host calls
--- this module makes, and every one of them goes through a caller-owned
--- handle. It never touches `math.random`, so two runs of the same seed
--- build the same corpus no matter what else the process did.

-- `alc_shapes` is optional: the rules module has to stay loadable in a
-- bare Lua VM with no package registry (the spec runner, the engine
-- level test harness). When the shapes package is present the typed
-- entry is declared, otherwise the entry is declared without it rather
-- than failing the load.
local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "othello6",
    version = "0.1.0",
    description = "6x6 Othello rules, turn-relative encoding and corpus builder for SLM experiments",
    category = "game",
}

local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            seed = T.number
                :is_optional()
                :describe("Seed carried by the opening position (default: 1)"),
        }),
        result = T.string:describe("Encoded opening position"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

-- ─── Constants ──────────────────────────────────────────────────────

--- Squares along one edge of the board.
local BOARD_SIZE = 6

--- Squares on the board, which is also the number of placements.
local CELLS = BOARD_SIZE * BOARD_SIZE

--- Contents of a square.
local EMPTY = 0
local BLACK = 1
local WHITE = 2

--- The move a side with no legal placement plays.
---
--- It is a move rather than the absence of one because the decode
--- surface constrains the sampled token against the legal set, and an
--- empty legal set is rejected when the constraint is built. A position
--- with no placement therefore answers `{ PASS }`, never `{}`.
local PASS = "-"

--- The token every sequence opens with.
---
--- It is not a move and not the padding: it is the position from which
--- the first move is predicted. Without it the opening encodes to
--- nothing, and two holes open at once. A decode from the opening
--- cannot be asked for at all — the bridge refuses a session with no
--- token to forward ("alc.nn generate_session: prompt_tokens is empty",
--- pinned by `alc_nn_generate_session_empty_prompt_errors` in
--- `tests/nn_bridge_smoke.rs`) — and no training row ever holds the
--- position "no moves yet, what is played first", because a row would
--- begin at the first move. The published Othello models carry the same
--- token for the same reason: `d_vocab = 61` is 60 squares and one
--- reserved id.
---
--- It is a separate character from the padding rather than the same one.
--- Sharing would put "open a game" and "the row ended here" on one
--- target, and the model would be trained to answer both from whichever
--- context happened to precede the token.
---
--- The caret is not a move letter (those run `a`..`M`), not a board or
--- side-to-move character (`.` / `x` / `o` / `B` / `W`), not the unused
--- separator (`>`) and not the pass (`-`).
local BOS = "^"

--- Tokens the leading marker costs a line.
local BOS_LEN = 1

--- Longest game seen, in moves and passes.
---
--- A game holds at most `CELLS - 4` = 32 placements, and a pass may fall
--- between two of them (two in a row end the game), so the length is not
--- fixed and the placement count alone does not bound it. Over 300k
--- uniformly random playouts the longest line ran 40 moves and 40
--- occurred three times [measured: 300k playouts, tail 36:1657 37:380
--- 38:92 39:13 40:3], so the design note's estimate of 36 is short of
--- what the game actually produces.
---
--- It is a measurement rather than a proof: 300k playouts are 300k
--- playouts, and a 41-move game is not ruled out by them. That is why
--- the budget below sits above this number rather than on it.
local LONGEST_OBSERVED_GAME = 40

--- Context window of the preset the corpus targets.
---
--- Eight tokens above the longest game observed. A budget set on the
--- observation would have no room for a game the sample missed, and a
--- line that does not fit is a loud error in `build_corpus` rather than
--- a silent truncation, so the cost of being exactly right would be a
--- corpus run that dies partway through.
local CTX_BUDGET = 48

--- Moves a game is allowed to hold, passes included.
---
--- One row is one game and the row is the context window, minus the
--- opening marker the line starts with. A game past this length does not
--- fit the preset and `build_corpus` says so rather than trimming it.
local MAX_MOVES = CTX_BUDGET - BOS_LEN

-- A budget at or below the longest game already seen would reject a
-- game the rules produce, so it is checked here rather than discovered
-- by a corpus run that stops on row 200k.
if MAX_MOVES <= LONGEST_OBSERVED_GAME then
    error(
        string.format(
            "othello6: the move budget is %d but a game of %d moves has been observed",
            MAX_MOVES,
            LONGEST_OBSERVED_GAME
        )
    )
end

--- Tokens the longest training line costs.
---
--- One row is one game, one move is one token and the line opens with
--- `BOS`: no separator, no newline. Every position in it is a position
--- the model is asked to answer — the marker included, which is where
--- the opening move is learnt — and that is what the board encoding
--- could not offer: there, 47 of the 48 tokens were the board and the
--- padding, and only one was the move being learnt.
local ROW_LEN = BOS_LEN + MAX_MOVES

-- A line that does not fit the preset cannot be trained on at all, so
-- the budget is checked once at load time rather than per row.
if ROW_LEN > CTX_BUDGET then
    error(
        string.format(
            "othello6: a training line needs %d tokens but the preset context is %d",
            ROW_LEN,
            CTX_BUDGET
        )
    )
end

--- Rows one playout contributes, for sizing a corpus.
---
--- One game is one row. It is exact rather than an estimate — the name
--- is kept because the trainer sizes its corpus through it — and it is
--- what makes the corpus arithmetic simple: a trainer that wants
--- `steps * batch` rows wants that many games.
local ROWS_PER_GAME_ESTIMATE = 1

--- Plies of random play a corpus game opens with, at most.
---
--- The opening of Othello is one position, so a corpus built from it
--- would be one position repeated. Every game therefore starts with a
--- uniform draw from `0..RANDOM_OPENING_MAX` random plies before the
--- labelled side takes its seat.
local RANDOM_OPENING_MAX = 6

--- Stride between the corpus seed and the per-game RNG seed, so two
--- games of the same batch never share a stream.
local RNG_STRIDE = 7919

--- The eight directions a bracket may run along, as `{dr, dc}`.
local DIRECTIONS = {
    { -1, -1 },
    { -1, 0 },
    { -1, 1 },
    { 0, -1 },
    { 0, 1 },
    { 1, -1 },
    { 1, 0 },
    { 1, 1 },
}

--- Vocabulary size the preset must cover.
local VOCAB_BUDGET = 64

M.BOARD_SIZE = BOARD_SIZE
M.CELLS = CELLS
M.EMPTY = EMPTY
M.BLACK = BLACK
M.WHITE = WHITE
M.PASS = PASS
M.BOS = BOS
M.BOS_LEN = BOS_LEN
M.MAX_MOVES = MAX_MOVES
M.CTX_BUDGET = CTX_BUDGET
M.LONGEST_OBSERVED_GAME = LONGEST_OBSERVED_GAME
M.ROW_LEN = ROW_LEN
M.ROWS_PER_GAME_ESTIMATE = ROWS_PER_GAME_ESTIMATE
M.RANDOM_OPENING_MAX = RANDOM_OPENING_MAX

--- Style names the teacher package implements, in the order the trainer
--- and the eval scenario iterate them.
---
--- The evaluation functions themselves live in `othello6_teacher`; this
--- list is here because the corpus builder, the trainer and the metrics
--- all need the same spelling of the names and none of them should own
--- the copy.
M.STYLES = { "corner", "mobility", "greedy" }

-- ─── Alphabet ───────────────────────────────────────────────────────

--- Board, side-to-move and separator characters, which the move letters
--- must not reuse (see the module header).
local RESERVED = { ["."] = true, ["x"] = true, ["o"] = true, ["B"] = true, ["W"] = true }

--- Move characters, ordered by square index: `ACTION_CHARS[index + 1]`
--- is the answer for square `index`.
---
--- Lower case first and upper case after, skipping the reserved
--- characters, so the letters stay monotone in the index and the
--- alphabet stays disjoint from the position fields.
local ACTION_CHARS = {}
do
    local letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i = 1, #letters do
        if #ACTION_CHARS == CELLS then
            break
        end
        local ch = letters:sub(i, i)
        if not RESERVED[ch] then
            ACTION_CHARS[#ACTION_CHARS + 1] = ch
        end
    end
    if #ACTION_CHARS ~= CELLS then
        error(
            string.format(
                "othello6: the move alphabet holds %d letters but the board has %d squares",
                #ACTION_CHARS,
                CELLS
            )
        )
    end
end

--- Square index of a move character, zero-based.
local ACTION_INDEX = {}
for i, ch in ipairs(ACTION_CHARS) do
    ACTION_INDEX[ch] = i - 1
end

--- Characters the corpus is written in, in token id order.
---
--- Index 1 holds token id 0 (the padding token), so `id = index - 1`.
---
--- `BOS` is appended last rather than filed with the other non-move
--- characters at the head. The ids of the moves are what a baked model
--- and every spec written against one are expressed in, and an insertion
--- in the middle would renumber all thirty-seven of them to buy nothing
--- but a tidier table.
local CHARS = { "\0", "\n", ">", ".", "x", "o", "B", "W" }
for _, ch in ipairs(ACTION_CHARS) do
    CHARS[#CHARS + 1] = ch
end
CHARS[#CHARS + 1] = PASS
CHARS[#CHARS + 1] = BOS

local TO_ID = {}
local TO_CHAR = {}
for index, ch in ipairs(CHARS) do
    local id = index - 1
    if TO_ID[ch] ~= nil then
        error(string.format("othello6: character %q is listed twice in the alphabet", ch))
    end
    TO_ID[ch] = id
    TO_CHAR[id] = ch
end

-- The duplicate check above already refuses a `BOS` that repeats any
-- other character, but it reports it as a table listing one character
-- twice. The one collision that matters is with the padding, and it is
-- named here: the marker and the padding sharing an id would train "open
-- a game" and "the row ended here" onto one target.
if TO_ID[BOS] == TO_ID["\0"] then
    error("othello6: the opening marker and the padding must not be the same token")
end

if #CHARS > VOCAB_BUDGET then
    error(
        string.format(
            "othello6: the alphabet holds %d characters but the preset vocabulary is %d",
            #CHARS,
            VOCAB_BUDGET
        )
    )
end

M.VOCAB_SIZE = #CHARS

--- Char-to-token-id map shared by the trainer and the NPC.
---
--- Returned tables are fresh copies so a caller cannot corrupt the
--- module-level maps every other entry point reads.
---@return table vocab `{ size, pad_id, to_id, to_char }`
function M.vocab()
    local to_id, to_char = {}, {}
    for ch, id in pairs(TO_ID) do
        to_id[ch] = id
    end
    for id, ch in pairs(TO_CHAR) do
        to_char[id] = ch
    end
    return {
        size = #CHARS,
        pad_id = TO_ID["\0"],
        to_id = to_id,
        to_char = to_char,
    }
end

--- Map a string over the module alphabet to token ids.
---
--- Errors on an unknown character instead of substituting a filler: a
--- silently replaced char would train the model on a position it can
--- never be asked about at decode time.
---@param text string
---@return integer[] ids
function M.to_ids(text)
    if type(text) ~= "string" then
        error("othello6.to_ids: text must be a string, got " .. type(text))
    end
    local ids = {}
    for i = 1, #text do
        local ch = text:sub(i, i)
        local id = TO_ID[ch]
        if id == nil then
            error(string.format("othello6.to_ids: char %q at %d is outside the vocabulary", ch, i))
        end
        ids[#ids + 1] = id
    end
    return ids
end

--- Move character of a zero-based square index.
---@param index integer `0..CELLS - 1`
---@return string action
function M.action_of_index(index)
    if type(index) ~= "number" or index ~= math.floor(index) or index < 0 or index >= CELLS then
        error(
            string.format(
                "othello6.action_of_index: index must be an integer in 0..%d, got %s",
                CELLS - 1,
                tostring(index)
            )
        )
    end
    return ACTION_CHARS[index + 1]
end

--- Zero-based square index of a move character.
---
--- Returns `nil` for the pass, which is a move but not a square.
---@param action string
---@return integer|nil index
function M.index_of_action(action)
    if type(action) ~= "string" then
        error("othello6.index_of_action: action must be a string, got " .. type(action))
    end
    if action == PASS then
        return nil
    end
    local index = ACTION_INDEX[action]
    if index == nil then
        error(string.format("othello6.index_of_action: %q is not a move character", action))
    end
    return index
end

-- ─── Internal helpers ───────────────────────────────────────────────

local function require_rng()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("othello6: alc.math is required (alc.math.rng_create / alc.math.rng_int)")
    end
    return alc.math
end

--- One-based board slot of a zero-based row and column.
local function slot(row, col)
    return row * BOARD_SIZE + col + 1
end

--- Validate every field the rules and the encoder read.
---
--- The check runs before any of them touches a field, so a position
--- built by hand (a probe, a logged game) fails on the field it got
--- wrong rather than on the character it would have produced.
local function require_state(fn, state)
    if type(state) ~= "table" then
        error(string.format("othello6.%s: state must be a table, got %s", fn, type(state)))
    end
    if type(state.board) ~= "table" then
        error(
            string.format("othello6.%s: state.board must be a table, got %s", fn, type(state.board))
        )
    end
    for i = 1, CELLS do
        local cell = state.board[i]
        if cell ~= EMPTY and cell ~= BLACK and cell ~= WHITE then
            error(
                string.format(
                    "othello6.%s: state.board[%d] must be 0, 1 or 2, got %s",
                    fn,
                    i,
                    tostring(cell)
                )
            )
        end
    end
    if state.turn ~= BLACK and state.turn ~= WHITE then
        error(
            string.format(
                "othello6.%s: state.turn must be %d or %d, got %s",
                fn,
                BLACK,
                WHITE,
                tostring(state.turn)
            )
        )
    end
    if
        type(state.passes) ~= "number"
        or state.passes < 0
        or state.passes ~= math.floor(state.passes)
    then
        error(
            string.format(
                "othello6.%s: state.passes must be a non-negative integer, got %s",
                fn,
                tostring(state.passes)
            )
        )
    end
    -- The history is what `encode` reads, so a position carrying the
    -- wrong type here would fail inside the encoder rather than here.
    -- A missing history is read as an empty one: a position built from
    -- rows is a board without a game behind it, which is a legitimate
    -- thing for a probe or a spec to hand in.
    if state.moves ~= nil and type(state.moves) ~= "table" then
        error(
            string.format("othello6.%s: state.moves must be a table, got %s", fn, type(state.moves))
        )
    end
    return state
end

--- Copy of a position, so nothing a caller-supplied policy does to the
--- table it was handed reaches the live game.
---
--- The history is copied rather than shared for the same reason the
--- board is: `apply` appends to the copy, and a shared array would grow
--- the position the caller still holds.
local function copy_state(state)
    local board = {}
    for i = 1, CELLS do
        board[i] = state.board[i]
    end
    local moves = {}
    local played = state.moves
    if played ~= nil then
        for i = 1, #played do
            moves[i] = played[i]
        end
    end
    return {
        board = board,
        turn = state.turn,
        passes = state.passes,
        ply = state.ply or 0,
        seed = state.seed,
        moves = moves,
    }
end

--- Discs the mover flips by playing a square, or `nil` when the square
--- is not a legal placement.
---
--- The empty check and the bracket check are the whole legality rule, so
--- `legal_actions` and `apply` share this function rather than each
--- carrying their own reading of it.
local function flips_at(board, index, color)
    if board[index + 1] ~= EMPTY then
        return nil
    end
    local opponent = 3 - color
    local row = index // BOARD_SIZE
    local col = index % BOARD_SIZE
    local out = nil
    for _, dir in ipairs(DIRECTIONS) do
        local dr, dc = dir[1], dir[2]
        local r, c = row + dr, col + dc
        local run = {}
        while r >= 0 and r < BOARD_SIZE and c >= 0 and c < BOARD_SIZE do
            local cell = board[slot(r, c)]
            if cell == opponent then
                run[#run + 1] = slot(r, c)
            elseif cell == color then
                if #run > 0 then
                    out = out or {}
                    for _, s in ipairs(run) do
                        out[#out + 1] = s
                    end
                end
                break
            else
                break
            end
            r, c = r + dr, c + dc
        end
    end
    return out
end

--- Discs each side holds, plus the empty squares.
local function count_discs(board)
    local black, white, empty = 0, 0, 0
    for i = 1, CELLS do
        local cell = board[i]
        if cell == BLACK then
            black = black + 1
        elseif cell == WHITE then
            white = white + 1
        else
            empty = empty + 1
        end
    end
    return black, white, empty
end

-- ─── Game progression ───────────────────────────────────────────────

--- Open a game.
---
--- Every seed opens on the same position — Othello has one opening — so
--- the seed is carried for tracing a corpus row back to its playout
--- rather than to vary the board. The variety in a corpus comes from
--- `build_corpus` playing a random prefix, not from the opening.
---@param seed integer
---@return table state `{ board, turn, passes, ply, seed, moves }`
function M.new_game(seed)
    if type(seed) ~= "number" then
        error("othello6.new_game: seed must be a number, got " .. type(seed))
    end
    local board = {}
    for i = 1, CELLS do
        board[i] = EMPTY
    end
    board[slot(2, 2)] = WHITE
    board[slot(2, 3)] = BLACK
    board[slot(3, 2)] = BLACK
    board[slot(3, 3)] = WHITE
    return {
        board = board,
        turn = BLACK,
        passes = 0,
        ply = 0,
        seed = seed,
        moves = {},
    }
end

--- Build a position from six row strings of `.` / `B` / `W`.
---
--- Rows are written in absolute colours because that is how a position
--- is read off a board. Probes and spec cases need a way in that does
--- not go through a move sequence.
---
--- The position that comes back has an empty history, because there is
--- no game behind it — the rules read the board, so every rule answers
--- normally, but `encode` has nothing to write. A position meant to be
--- encoded has to be reached with `apply`.
---@param rows string[] Six strings of six characters
---@param turn string `"black"` or `"white"`
---@return table state
function M.state_from_rows(rows, turn)
    if type(rows) ~= "table" or #rows ~= BOARD_SIZE then
        error(
            string.format(
                "othello6.state_from_rows: rows must be %d strings, got %s",
                BOARD_SIZE,
                type(rows) == "table" and tostring(#rows) or type(rows)
            )
        )
    end
    if turn ~= "black" and turn ~= "white" then
        error('othello6.state_from_rows: turn must be "black" or "white", got ' .. tostring(turn))
    end
    local board = {}
    for r = 1, BOARD_SIZE do
        local line = rows[r]
        if type(line) ~= "string" or #line ~= BOARD_SIZE then
            error(
                string.format(
                    "othello6.state_from_rows: row %d must be %d characters, got %s",
                    r,
                    BOARD_SIZE,
                    tostring(line)
                )
            )
        end
        for c = 1, BOARD_SIZE do
            local ch = line:sub(c, c)
            local cell
            if ch == "." then
                cell = EMPTY
            elseif ch == "B" then
                cell = BLACK
            elseif ch == "W" then
                cell = WHITE
            else
                error(
                    string.format(
                        "othello6.state_from_rows: row %d column %d is %q, expected . B or W",
                        r,
                        c,
                        ch
                    )
                )
            end
            board[slot(r - 1, c - 1)] = cell
        end
    end
    return {
        board = board,
        turn = turn == "black" and BLACK or WHITE,
        passes = 0,
        ply = 0,
        seed = 0,
        moves = {},
    }
end

--- Name of the side to move.
---@param state table Position
---@return string side `"black"` or `"white"`
function M.side_to_move(state)
    require_state("side_to_move", state)
    return state.turn == BLACK and "black" or "white"
end

--- Moves the side to move may play, as move characters in square order.
---
--- A position with no placement answers `{ PASS }` rather than `{}`:
--- the decode surface builds its constraint from this list and an empty
--- constraint is rejected when it is built, so an empty answer would
--- turn a normal Othello position into a crash. `PASS` appears only in
--- that answer — a side with a placement is not allowed to pass.
---@param state table Position
---@return string[] actions
function M.legal_actions(state)
    require_state("legal_actions", state)
    local out = {}
    for index = 0, CELLS - 1 do
        if flips_at(state.board, index, state.turn) ~= nil then
            out[#out + 1] = ACTION_CHARS[index + 1]
        end
    end
    if #out == 0 then
        out[1] = PASS
    end
    return out
end

--- Play one move.
---
--- Returns a new position; the one it was handed is left untouched, so a
--- caller can keep a history without copying defensively.
---
--- The move is appended to `state.moves`, a pass along with the rest.
--- That list is the whole input the model is trained on, so a move that
--- did not reach it would be a move the model never learns to expect.
---
--- An illegal move is a loud error rather than a skipped turn: a policy
--- that answers off the legal set is a bug in the policy, and a corpus
--- that quietly dropped those answers would teach the model the bug.
---@param state table Position
---@param action string Move character, or `PASS`
---@return table state Position after the move
function M.apply(state, action)
    require_state("apply", state)
    if type(action) ~= "string" then
        error("othello6.apply: action must be a string, got " .. type(action))
    end
    local next_state = copy_state(state)
    if action == PASS then
        for index = 0, CELLS - 1 do
            if flips_at(state.board, index, state.turn) ~= nil then
                error(
                    string.format(
                        "othello6.apply: %s passed but %s is legal",
                        M.side_to_move(state),
                        ACTION_CHARS[index + 1]
                    )
                )
            end
        end
        next_state.passes = state.passes + 1
    else
        local index = ACTION_INDEX[action]
        if index == nil then
            error(string.format("othello6.apply: %q is not a move character", action))
        end
        local flips = flips_at(state.board, index, state.turn)
        if flips == nil then
            error(
                string.format(
                    "othello6.apply: %s is not legal for %s at (%d,%d)",
                    action,
                    M.side_to_move(state),
                    index // BOARD_SIZE,
                    index % BOARD_SIZE
                )
            )
        end
        next_state.board[index + 1] = state.turn
        for _, s in ipairs(flips) do
            next_state.board[s] = state.turn
        end
        next_state.passes = 0
    end
    next_state.turn = 3 - state.turn
    next_state.ply = (state.ply or 0) + 1
    next_state.moves[#next_state.moves + 1] = action
    return next_state
end

--- Whether the game is finished.
---
--- Two passes in a row is the standard ending; a full board is the same
--- ending reached one ply earlier, and it is checked separately so a
--- loop does not have to walk two forced passes to notice.
---@param state table Position
---@return boolean over
function M.is_over(state)
    require_state("is_over", state)
    if state.passes >= 2 then
        return true
    end
    local _, _, empty = count_discs(state.board)
    return empty == 0
end

--- Winner of a finished game.
---
--- Returns `nil` while the game is still running: the caller asked a
--- question that has no answer yet, and inventing `"draw"` would hide a
--- loop that stopped one move early.
---@param state table Position
---@return string|nil winner `"black"` / `"white"` / `"draw"`, or nil when unfinished
function M.winner(state)
    if not M.is_over(state) then
        return nil
    end
    local black, white = count_discs(state.board)
    if black > white then
        return "black"
    elseif white > black then
        return "white"
    end
    return "draw"
end

-- ─── Encoding ───────────────────────────────────────────────────────

--- Project a position to the line the model reads: `BOS`, then the moves
--- that reached it, one character each, passes included.
---
--- The length is one plus the number of moves played, so it grows
--- through a game and the opening — no moves yet — encodes to `BOS`
--- alone. That is the correct prompt for the first move: the model is
--- asked what to play from a game it has been told nothing about, which
--- is the one position Othello always starts from. It is also the reason
--- the marker exists at all; see `BOS` for the two holes an empty
--- opening line opens.
---
--- The board is not written. A model reading this line has to hold the
--- position itself, and the point of the experiment is whether it does.
---
--- A position built by `state_from_rows` encodes to `BOS` alone whatever
--- is on its board, because no moves led to it. Encoding is for
--- positions that were played.
---@param state table Position
---@return string encoded `BOS` and one character per move played
function M.encode(state)
    require_state("encode", state)
    local moves = state.moves
    if moves == nil then
        return BOS
    end
    return BOS .. table.concat(moves)
end

-- ─── Policies ───────────────────────────────────────────────────────

--- A policy that answers with a uniform draw from the legal moves.
---
--- Takes the RNG and returns the policy rather than the move, because a
--- legal set depends on the position while a stream depends on the
--- caller: the returned function has the `policy(state)` shape
--- `build_corpus` labels with, and two policies built from two handles
--- never share a stream.
---@param rng userdata `alc.math.rng_create` handle
---@return fun(state: table): string policy
function M.policy_random(rng)
    if rng == nil then
        error("othello6.policy_random: rng must be an alc.math.rng_create handle")
    end
    local math_ns = require_rng()
    return function(state)
        local legal = M.legal_actions(state)
        return legal[math_ns.rng_int(rng, 1, #legal)]
    end
end

-- ─── Corpus ─────────────────────────────────────────────────────────

--- Encode one finished game and pad it to the model context window.
---
--- The line is `BOS` and the move sequence, and nothing else. A game
--- that does not fit is a loud error rather than a truncation: a trimmed
--- game ends on a move the position never called for, and the model
--- would learn that ending as a real one. Games that long are rare but
--- they exist — the longest of 300k random playouts ran
--- `LONGEST_OBSERVED_GAME` moves, which is why the budget sits above
--- that rather than on it — so the caller is told which game it was
--- rather than handed a quietly shortened corpus.
local function make_row(state, ctx_len, pad_id)
    local ids = M.to_ids(M.encode(state))
    if #ids > ctx_len then
        error(
            string.format(
                "othello6.build_corpus: a game of %d moves needs %d tokens and does not fit "
                    .. "a context of %d",
                #ids - BOS_LEN,
                #ids,
                ctx_len
            )
        )
    end
    for _ = #ids + 1, ctx_len do
        ids[#ids + 1] = pad_id
    end
    return ids
end

local function require_positive_int(fn, field, value)
    local number = tonumber(value)
    if number == nil or number < 1 then
        error(string.format("othello6.%s: opts.%s must be a positive number", fn, field))
    end
    return math.floor(number)
end

--- Build the supervised corpus that teaches `policy` to a model.
---
--- One row is one game: the opening marker, the whole move sequence,
--- then padding. Every position in that row is a position the model is
--- asked to answer — the marker included, which is the only place the
--- opening move can be learnt — so the loss lands on move prediction
--- everywhere along the line and no mask is needed to keep it there.
---
--- `policy` plays **both seats**. One row cannot carry one side's moves
--- and not the other's — telling them apart at training time would take
--- a per-token mask, and the trainer has no way to pass one — so a game
--- half played by a random opponent would teach the model that half as
--- if the policy had chosen it. Self-play keeps every labelled token
--- the policy's own.
---
--- Each game opens with a uniform draw of `0..random_opening_max`
--- random plies, so the corpus is not one opening repeated. Those plies
--- are learnt too, which costs nothing worth avoiding: the corpora that
--- taught the published models legal play were uniformly random
--- throughout (arXiv:2210.13382), and the playing style the corpus is
--- built for lives in the moves after the prefix.
---
--- `alc.nn.data.synthetic` walks its rows once, so a trainer asking for
--- `steps * batch` rows needs `games >= steps * batch` — one row per
--- game, `ROWS_PER_GAME_ESTIMATE` being 1.
---@param policy fun(state: table): string Move policy, played by both sides
---@param opts table `{ ctx_len, games, seed?, pad_id?, random_opening_max? }`
---@return integer[][] rows Token id rows, one per game, each `ctx_len` long
function M.build_corpus(policy, opts)
    if type(policy) ~= "function" then
        error("othello6.build_corpus: policy must be a function, got " .. type(policy))
    end
    if type(opts) ~= "table" then
        error("othello6.build_corpus: opts must be a table, got " .. type(opts))
    end
    local ctx_len = require_positive_int("build_corpus", "ctx_len", opts.ctx_len)
    local games = require_positive_int("build_corpus", "games", opts.games)
    local seed = math.floor(tonumber(opts.seed) or 1)
    local opening_max = opts.random_opening_max
    if opening_max == nil then
        opening_max = RANDOM_OPENING_MAX
    end
    opening_max = tonumber(opening_max)
    if opening_max == nil or opening_max < 0 then
        error("othello6.build_corpus: opts.random_opening_max must be a non-negative number")
    end
    opening_max = math.floor(opening_max)
    local pad_id = opts.pad_id
    if pad_id == nil then
        pad_id = TO_ID["\0"]
    end
    if type(pad_id) ~= "number" then
        error("othello6.build_corpus: opts.pad_id must be a number, got " .. type(pad_id))
    end

    local math_ns = require_rng()
    local rows = {}
    for i = 1, games do
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        local random_move = M.policy_random(rng)
        local state = M.new_game(seed + i)
        local opening = math_ns.rng_int(rng, 0, opening_max)
        for _ = 1, opening do
            if M.is_over(state) then
                break
            end
            state = M.apply(state, random_move(state))
        end
        while not M.is_over(state) do
            state = M.apply(state, policy(state))
        end
        rows[#rows + 1] = make_row(state, ctx_len, pad_id)
    end
    return rows
end

return M
