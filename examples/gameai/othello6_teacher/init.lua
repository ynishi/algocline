--- othello6_teacher — depth-parameterised negamax teachers for 6x6 Othello
---
--- The teacher labels the corpus the Cards are baked from. Its strength
--- is the search depth `d` and its playing style is the evaluation
--- function `s`, so both axes of the experiment are dials the caller
--- sets rather than quantities anyone has to estimate from a win rate.
--- A Card trained on depth 4 is compared against the depth 4 teacher's
--- own answers; nothing here measures strength by playing games.
---
--- ## Usage
---
--- ```lua
--- local othello = require("othello6")
--- local teacher = require("othello6_teacher")
---
--- local state = othello.new_game(1)
--- local action, value = teacher.search(state, 4, "corner")
---
--- -- The same thing in the shape `build_corpus` labels with:
--- local rows = othello.build_corpus(teacher.policy(4, "corner"), {
---     ctx_len = othello.CTX_BUDGET, games = 200, seed = 1,
--- })
--- ```
---
--- ## Entry contract
---
--- - `evaluate(state, style)` — static value of a position, from the
---   side to move's point of view
--- - `search(state, depth, style)` — `action, value` under negamax with
---   alpha-beta
--- - `search_naive(state, depth, style)` — the same answer without the
---   pruning, for the spec to check the pruning against
--- - `policy(depth, style)` — `fun(state): string`, the shape
---   `othello6.build_corpus` and `guardian_duel.compile_policy` label
---   with
--- - `STYLES` / `DEPTHS` — the style names and the depths the
---   experiment sweeps
---
--- ## Evaluation functions
---
--- Every evaluator answers **from the side to move's point of view**,
--- which is what negamax needs: the value of a position for one side is
--- the negation of its value for the other.
---
--- | style | what it reads |
--- |---|---|
--- | `corner` | square weights only — corner `+30`, the squares that give a corner away `-12`, the rest of an edge `+5`, everything else `+1` |
--- | `mobility` | placements available to each side, `x10`, plus the disc difference `x1` |
--- | `greedy` | the disc difference and nothing else |
---
--- `corner` deliberately never counts discs on its own account: holding
--- more discs in the middlegame is the classic Othello mistake, and a
--- style that both weights squares and counts discs would be a blend of
--- two styles rather than one of them. The weights are starting values.
--- They exist to make the three styles answer differently often enough
--- to measure, so they are tuning knobs of the testbed rather than of a
--- game anyone plays.
---
--- ## Terminal positions
---
--- A finished game is scored by its result, `+WIN_SCORE + margin` for a
--- won position and `-WIN_SCORE + margin` for a lost one, so any win
--- outranks every unfinished position under every style and the margin
--- only orders wins among themselves. A position that is over is
--- returned at once even when depth is left, which is also what keeps
--- the two-pass ending from being searched past.
---
--- ## Passes
---
--- `othello6.legal_actions` answers `{ PASS }` when a side has no
--- placement, so a pass is searched like any other move and costs one
--- ply of depth. Two passes in a row end the game and the recursion
--- stops there, so a blocked position cannot loop.
---
--- ## Determinism
---
--- Ties go to the lowest square index. `legal_actions` lists placements
--- in index order and the search only replaces its best move on a
--- strictly greater value, so the first of several equal moves wins.
--- The search reads nothing but the position: no clock, no RNG, no
--- carried state, so `policy(d, s)` answers the same move to the same
--- position however many times it is asked. That is the property
--- `compile_policy` checks for and the reason a Card's agreement with
--- the teacher is a well defined number at all.
---
--- ## Pruning
---
--- `search` prunes and `search_naive` does not, and the two are required
--- to return the same move and the same value. The pruning is the one
--- piece of this package that can be wrong without looking wrong — a
--- teacher that quietly answers a slightly worse move still produces a
--- corpus, and every measurement taken afterwards would be measuring the
--- bug. The naive twin is kept in the shipped module rather than in the
--- spec so that the check runs against the same code a caller loads.
---
--- This module makes no host calls at all: no `alc.llm`, no `alc.math`.
--- It runs in a bare Lua VM with `othello6` on the path.

local othello = require("othello6")

local M = {}

---@type AlcMeta
M.meta = {
    name = "othello6_teacher",
    version = "0.1.0",
    description = "Negamax teachers for 6x6 Othello: depth is the strength, evaluation weights are the style",
    category = "game",
}

-- ─── Constants ──────────────────────────────────────────────────────

local BOARD_SIZE = othello.BOARD_SIZE
local CELLS = othello.CELLS
local EMPTY = othello.EMPTY
local BLACK = othello.BLACK
local WHITE = othello.WHITE
local PASS = othello.PASS

--- Value a decided game carries on top of its margin.
---
--- Large enough that no positional score can reach it: the widest
--- `corner` board is well under a thousand and `mobility` under a
--- hundred, so a won position always outranks an unfinished one.
local WIN_SCORE = 10000

M.WIN_SCORE = WIN_SCORE

--- `corner` square weights.
local CORNER_WEIGHT = 30
local CORNER_NEIGHBOUR_WEIGHT = -12
local EDGE_WEIGHT = 5
local INNER_WEIGHT = 1

--- `mobility` term weights.
local MOBILITY_WEIGHT = 10
local MOBILITY_DISC_WEIGHT = 1

--- Depths the experiment sweeps, in the order it reports them.
M.DEPTHS = { 1, 2, 4, 6 }

--- Weight of every square for `corner`, indexed by board slot.
---
--- A corner cannot be flipped once taken, the three squares around it
--- are the squares that hand it over, and the rest of an edge is worth
--- holding because an edge run is hard to attack from behind. On a 6x6
--- board that is four corners, twelve corner neighbours (the four X
--- squares and the eight C squares carry the same penalty because they
--- give the same corner away), eight remaining edge squares and twelve
--- interior ones.
local SQUARE_WEIGHTS = {}
do
    local function on_edge(coord)
        return coord == 0 or coord == BOARD_SIZE - 1
    end
    local function beside_edge(coord)
        return coord == 1 or coord == BOARD_SIZE - 2
    end
    for row = 0, BOARD_SIZE - 1 do
        for col = 0, BOARD_SIZE - 1 do
            local weight
            if on_edge(row) and on_edge(col) then
                weight = CORNER_WEIGHT
            elseif
                (on_edge(row) and beside_edge(col))
                or (beside_edge(row) and on_edge(col))
                or (beside_edge(row) and beside_edge(col))
            then
                weight = CORNER_NEIGHBOUR_WEIGHT
            elseif on_edge(row) or on_edge(col) then
                weight = EDGE_WEIGHT
            else
                weight = INNER_WEIGHT
            end
            SQUARE_WEIGHTS[row * BOARD_SIZE + col + 1] = weight
        end
    end
end

-- ─── Internal helpers ───────────────────────────────────────────────

--- Fail on a position that is not even a table.
---
--- The field-level check belongs to `othello6`, which runs it on every
--- entry point this module calls; naming the teacher on the shallow
--- check keeps a caller that passed `nil` from reading an error about a
--- module it never called.
local function require_state(fn, state)
    if type(state) ~= "table" then
        error(string.format("othello6_teacher.%s: state must be a table, got %s", fn, type(state)))
    end
    return state
end

--- Discs the side to move holds, less the discs the other side holds.
local function disc_diff(state)
    local mover = state.turn
    local diff = 0
    for i = 1, CELLS do
        local cell = state.board[i]
        if cell == mover then
            diff = diff + 1
        elseif cell ~= EMPTY then
            diff = diff - 1
        end
    end
    return diff
end

--- The same position with the other side to move.
---
--- Built so `mobility` can ask `othello6.legal_actions` what the
--- opponent could play, which is a question the rules module only
--- answers about the side to move. The board table is shared rather
--- than copied because `legal_actions` reads it and never writes to it,
--- and the copy would be paid for at every leaf of the search.
local function opponent_view(state)
    return {
        board = state.board,
        turn = state.turn == BLACK and WHITE or BLACK,
        passes = state.passes,
        ply = state.ply,
        seed = state.seed,
    }
end

--- Placements the side to move has, counting a pass as none.
---
--- `legal_actions` answers `{ PASS }` rather than `{}` for a blocked
--- side, so the pass has to be discounted here: a side with no
--- placement has no mobility, and counting its pass as one move would
--- make being blocked look as good as having a move.
local function placement_count(state)
    local actions = othello.legal_actions(state)
    if #actions == 1 and actions[1] == PASS then
        return 0
    end
    return #actions
end

-- ─── Evaluation ─────────────────────────────────────────────────────

local EVALUATORS = {}

--- Square weights, signed by owner. Discs are not counted.
function EVALUATORS.corner(state)
    local mover = state.turn
    local score = 0
    for i = 1, CELLS do
        local cell = state.board[i]
        if cell == mover then
            score = score + SQUARE_WEIGHTS[i]
        elseif cell ~= EMPTY then
            score = score - SQUARE_WEIGHTS[i]
        end
    end
    return score
end

--- Difference in available placements, with the disc count as a tiebreak.
function EVALUATORS.mobility(state)
    local mine = placement_count(state)
    local theirs = placement_count(opponent_view(state))
    return (mine - theirs) * MOBILITY_WEIGHT + disc_diff(state) * MOBILITY_DISC_WEIGHT
end

--- Disc difference, which is the quantity the game is scored on and the
--- wrong quantity to maximise before it is over.
function EVALUATORS.greedy(state)
    return disc_diff(state)
end

--- Style names this package implements, in the order the trainer and the
--- eval scenario iterate them.
---
--- Taken from `othello6.STYLES` so the spelling has one owner, and
--- checked against the evaluators in both directions: a name without an
--- evaluator would fail at the first corpus row, and an evaluator no
--- name reaches would be a style the experiment silently never ran.
M.STYLES = {}
do
    local named = {}
    for _, style in ipairs(othello.STYLES) do
        if EVALUATORS[style] == nil then
            error(
                string.format(
                    "othello6_teacher: othello6 declares style %q but no evaluator implements it",
                    style
                )
            )
        end
        named[style] = true
        M.STYLES[#M.STYLES + 1] = style
    end
    for style in pairs(EVALUATORS) do
        if not named[style] then
            error(
                string.format(
                    "othello6_teacher: evaluator %q is not one of the styles othello6 declares",
                    style
                )
            )
        end
    end
end

local function require_style(fn, style)
    local evaluator = EVALUATORS[type(style) == "string" and style or ""]
    if evaluator == nil then
        error(
            string.format(
                "othello6_teacher.%s: style must be one of %s, got %s",
                fn,
                table.concat(M.STYLES, " / "),
                tostring(style)
            )
        )
    end
    return evaluator
end

local function require_depth(fn, depth)
    if type(depth) ~= "number" or depth ~= math.floor(depth) or depth < 1 then
        error(
            string.format(
                "othello6_teacher.%s: depth must be a positive integer, got %s",
                fn,
                tostring(depth)
            )
        )
    end
    return depth
end

--- Static value of a position for the side to move.
---
--- A finished game answers by its result whatever the style, because
--- what a won position is worth is a rule of the game rather than a
--- matter of taste; the margin rides along so that a bigger win reads as
--- better than a narrower one.
---@param state table Position
---@param style string One of `STYLES`
---@return number value Positive when the side to move stands better
function M.evaluate(state, style)
    local evaluator = require_style("evaluate", style)
    require_state("evaluate", state)
    -- `is_over` validates every field the evaluators go on to read.
    if othello.is_over(state) then
        local margin = disc_diff(state)
        if margin > 0 then
            return WIN_SCORE + margin
        elseif margin < 0 then
            return -WIN_SCORE + margin
        end
        return 0
    end
    return evaluator(state)
end

-- ─── Search ─────────────────────────────────────────────────────────

--- Negamax with alpha-beta.
---
--- Returns the value of the position for the side to move and the move
--- that realises it. Ties keep the earlier move because the comparison
--- is strict and `legal_actions` is ordered by square index.
---
--- The window is fail-soft: a cut off subtree returns a bound rather
--- than its exact value. That bound can never displace the best move at
--- the root, because a cut off child is one whose value cannot exceed
--- the alpha the best move already set, and the root itself is searched
--- with an open beta so it never cuts off. `search_naive` and the spec
--- are what hold that reasoning to account.
local function negamax(state, depth, style, alpha, beta)
    if depth <= 0 or othello.is_over(state) then
        return M.evaluate(state, style), nil
    end
    local best_value, best_action = -math.huge, nil
    for _, action in ipairs(othello.legal_actions(state)) do
        local value = -negamax(othello.apply(state, action), depth - 1, style, -beta, -alpha)
        if value > best_value then
            best_value, best_action = value, action
        end
        if best_value > alpha then
            alpha = best_value
        end
        if alpha >= beta then
            break
        end
    end
    return best_value, best_action
end

--- Negamax with no pruning, kept as the reference `search` is checked
--- against. Same answer, more nodes.
local function negamax_naive(state, depth, style)
    if depth <= 0 or othello.is_over(state) then
        return M.evaluate(state, style), nil
    end
    local best_value, best_action = -math.huge, nil
    for _, action in ipairs(othello.legal_actions(state)) do
        local value = -negamax_naive(othello.apply(state, action), depth - 1, style)
        if value > best_value then
            best_value, best_action = value, action
        end
    end
    return best_value, best_action
end

--- Best move of the depth `depth` teacher playing style `style`.
---
--- Answers `nil` for a finished game, along with the value of the
--- result: the caller asked for a move in a position that has none, and
--- inventing a pass would let a loop that ran one ply too long look like
--- it had worked.
---@param state table Position
---@param depth integer Plies to search, at least 1
---@param style string One of `STYLES`
---@return string|nil action Move character, `PASS`, or nil when the game is over
---@return number value Value of the position for the side to move
function M.search(state, depth, style)
    require_state("search", state)
    require_depth("search", depth)
    require_style("search", style)
    if othello.is_over(state) then
        return nil, M.evaluate(state, style)
    end
    local value, action = negamax(state, depth, style, -math.huge, math.huge)
    return action, value
end

--- `search` without the pruning, for checking `search` against.
---
--- Same signature and the same answer; exponentially slower, so it is a
--- verification tool rather than a second teacher.
---@param state table Position
---@param depth integer Plies to search, at least 1
---@param style string One of `STYLES`
---@return string|nil action
---@return number value
function M.search_naive(state, depth, style)
    require_state("search_naive", state)
    require_depth("search_naive", depth)
    require_style("search_naive", style)
    if othello.is_over(state) then
        return nil, M.evaluate(state, style)
    end
    local value, action = negamax_naive(state, depth, style)
    return action, value
end

--- The depth `depth`, style `style` teacher as a labelling policy.
---
--- Takes the two dials and returns the policy rather than the move, so
--- that the result has the `policy(state)` shape `othello6.build_corpus`
--- labels a corpus with and `compile_policy` validates. The returned
--- function holds no state, so two policies built from the same pair
--- answer alike and either one may be reused across games.
---@param depth integer Plies to search, at least 1
---@param style string One of `STYLES`
---@return fun(state: table): string policy
function M.policy(depth, style)
    require_depth("policy", depth)
    require_style("policy", style)
    return function(state)
        local action = M.search(state, depth, style)
        if action == nil then
            error("othello6_teacher.policy: the game is over, so there is no move to label")
        end
        return action
    end
end

return M
