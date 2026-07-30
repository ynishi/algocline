--- card_duel — five-round rank duel used as the GameAI SLM NPC playground
---
--- Pure-Lua rules for a two-player card game small enough that a
--- from-scratch tiny SLM can learn a play style from it, yet rich
--- enough that the style is state-dependent. The module owns the game
--- state, the legal-action enumeration, the char-level encoding shared
--- by the trainer and the NPC, and the two reference policies (the
--- deterministic teacher style and the random opponent).
---
--- ## Usage
---
--- ```lua
--- local duel = require("card_duel")
--- local g = duel.new_game(42)
--- while not duel.is_over(g) do
---     local a1 = duel.policy_aggressive(g.p1)
---     local a2 = duel.policy_random(g.p2, alc.math.rng_create(7))
---     g = duel.apply(g, a1, a2)
--- end
--- print(duel.winner(g))
--- ```
---
--- ## Algorithm
---
--- 1. `new_game(seed)` deals each player five ranks drawn uniformly
---    from 1..9 with replacement, using a seeded RNG.
--- 2. Each of the five rounds both players reveal one card from hand
---    simultaneously; the higher rank scores one point, a tie scores
---    nothing, and both cards leave the hands.
--- 3. After five rounds the higher score wins; equal scores draw.
---
--- The teacher style (`policy_aggressive`) plays the highest card while
--- it is not ahead and the lowest card once it leads. That rule is
--- deterministic and depends on the score gap, so a model can only
--- reproduce it by reading the encoded state rather than memorising a
--- constant.
---
--- ## Entry contract
---
--- - `new_game` / `apply` / `is_over` / `winner` — game progression
--- - `legal_actions` / `encode` / `vocab` / `to_ids` — NPC-facing view
--- - `policy_aggressive` / `policy_random` — reference policies
--- - `run` — Strategy entry; returns the encoded opening state
---
--- ## Caveats
---
--- The encoding is sized against the `gpt2 tiny` preset context window
--- (16 tokens). Every state encodes to exactly 12 characters because
--- the hand shrinks by one card per round while the opponent history
--- grows by one, so `encode(state) .. ">" .. action` always fits in 14
--- tokens regardless of the round. Widening the state (adding a second
--- opponent history, more rounds, ranks above 9) breaks that budget and
--- requires a larger preset.
---
--- `alc.math.rng_create` / `alc.math.rng_int` are the only host calls
--- this module makes. It never calls `alc.llm`, so it can run inside a
--- plain Lua VM that stubs `alc.math`.

-- `alc_shapes` is optional: the rules module has to stay loadable in
-- the bare mlua-lspec VM used by `crates/algocline-engine/tests/lua`,
-- which has no package registry. When the shapes package is present
-- (normal MCP session) the full typed spec is declared; otherwise the
-- entry is declared without shapes rather than failing the load.
local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "card_duel",
    version = "0.1.0",
    description = "Five-round rank duel rules, encoding and reference policies for SLM NPC demos",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, so the module still loads in a VM
-- without a package registry.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            seed = T.number:is_optional():describe("Deal seed for the opening state (default: 1)"),
        }),
        result = T.string:describe("Encoded opening state of player one"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

-- ─── Constants ──────────────────────────────────────────────────────

--- Ranks a dealt card may take.
local MIN_RANK, MAX_RANK = 1, 9

--- Cards per hand, which is also the number of rounds.
local HAND_SIZE = 5

M.HAND_SIZE = HAND_SIZE
M.MIN_RANK = MIN_RANK
M.MAX_RANK = MAX_RANK

--- Char alphabet, indexed by model token id.
---
--- Index 1 holds token id 0 (the padding token), so `id = index - 1`.
--- Seventeen entries keep the whole alphabet inside the `gpt2 tiny`
--- vocabulary of 64.
local CHARS = {
    "\0",
    "\n",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "R",
    "H",
    "P",
    "O",
    ">",
}

local TO_ID = {}
local TO_CHAR = {}
for index, ch in ipairs(CHARS) do
    local id = index - 1
    TO_ID[ch] = id
    TO_CHAR[id] = ch
end

--- Char-to-token-id map shared by the trainer and the NPC.
---
--- Returned tables are fresh copies so a caller cannot corrupt the
--- module-level maps that every other entry point reads.
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
--- Errors on an unknown character instead of substituting a filler:
--- a silently replaced char would train the model on a state it can
--- never be asked about at decode time.
---@param text string
---@return integer[] ids
function M.to_ids(text)
    if type(text) ~= "string" then
        error("card_duel.to_ids: text must be a string, got " .. type(text))
    end
    local ids = {}
    for i = 1, #text do
        local ch = text:sub(i, i)
        local id = TO_ID[ch]
        if id == nil then
            error(string.format("card_duel.to_ids: char %q at %d is outside the vocabulary", ch, i))
        end
        ids[#ids + 1] = id
    end
    return ids
end

-- ─── Internal helpers ───────────────────────────────────────────────

local function copy_list(list)
    local out = {}
    for i, v in ipairs(list) do
        out[i] = v
    end
    return out
end

local function sorted_copy(list)
    local out = copy_list(list)
    table.sort(out)
    return out
end

local function require_rng()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("card_duel: alc.math is required (alc.math.rng_create / alc.math.rng_int)")
    end
    return alc.math
end

--- Build the per-player view a policy / the encoder consumes.
local function make_state(round, hand, my_points, opp_points, opp_played)
    return {
        round = round,
        my_hand = hand,
        my_points = my_points,
        opp_points = opp_points,
        opp_played = opp_played,
    }
end

--- Remove one instance of `rank` from `hand`, returning a new list.
local function without_one(hand, rank)
    local out = {}
    local dropped = false
    for _, v in ipairs(hand) do
        if v == rank and not dropped then
            dropped = true
        else
            out[#out + 1] = v
        end
    end
    if not dropped then
        error(string.format("card_duel.apply: rank %s is not in hand", tostring(rank)))
    end
    return out
end

local function appended(list, value)
    local out = copy_list(list)
    out[#out + 1] = value
    return out
end

-- ─── Game progression ───────────────────────────────────────────────

--- Deal a fresh game.
---
--- Both hands are drawn from the same seeded RNG, so one seed fully
--- determines the deal and replays reproduce it exactly.
---@param seed integer
---@return table game `{ round, p1, p2, seed }` where `p1` / `p2` are per-player states
function M.new_game(seed)
    if type(seed) ~= "number" then
        error("card_duel.new_game: seed must be a number, got " .. type(seed))
    end
    local math_ns = require_rng()
    local rng = math_ns.rng_create(seed)
    local h1, h2 = {}, {}
    for _ = 1, HAND_SIZE do
        h1[#h1 + 1] = math_ns.rng_int(rng, MIN_RANK, MAX_RANK)
    end
    for _ = 1, HAND_SIZE do
        h2[#h2 + 1] = math_ns.rng_int(rng, MIN_RANK, MAX_RANK)
    end
    return {
        round = 1,
        seed = seed,
        p1 = make_state(1, sorted_copy(h1), 0, 0, {}),
        p2 = make_state(1, sorted_copy(h2), 0, 0, {}),
    }
end

--- Distinct ranks in hand, ascending.
---@param state table Per-player state
---@return integer[] actions
function M.legal_actions(state)
    if type(state) ~= "table" or type(state.my_hand) ~= "table" then
        error("card_duel.legal_actions: state.my_hand must be a table")
    end
    local seen, out = {}, {}
    for _, rank in ipairs(state.my_hand) do
        if not seen[rank] then
            seen[rank] = true
            out[#out + 1] = rank
        end
    end
    table.sort(out)
    return out
end

--- Play one round.
---
--- Both actions are validated against the corresponding hand before any
--- state is built, so an illegal move is a loud error rather than a
--- half-applied round.
---@param g table Game
---@param a_p1 integer
---@param a_p2 integer
---@return table game Next game state
function M.apply(g, a_p1, a_p2)
    if M.is_over(g) then
        error("card_duel.apply: game is already over")
    end
    local h1 = without_one(g.p1.my_hand, a_p1)
    local h2 = without_one(g.p2.my_hand, a_p2)

    local p1_points, p2_points = g.p1.my_points, g.p2.my_points
    if a_p1 > a_p2 then
        p1_points = p1_points + 1
    elseif a_p2 > a_p1 then
        p2_points = p2_points + 1
    end

    local next_round = g.round + 1
    return {
        round = next_round,
        seed = g.seed,
        p1 = make_state(
            next_round,
            sorted_copy(h1),
            p1_points,
            p2_points,
            appended(g.p1.opp_played, a_p2)
        ),
        p2 = make_state(
            next_round,
            sorted_copy(h2),
            p2_points,
            p1_points,
            appended(g.p2.opp_played, a_p1)
        ),
    }
end

---@param g table Game
---@return boolean over
function M.is_over(g)
    if type(g) ~= "table" or type(g.round) ~= "number" then
        error("card_duel.is_over: game.round must be a number")
    end
    return g.round > HAND_SIZE
end

--- Winner of a finished game.
---
--- Returns `nil` while the game is still running: the caller asked a
--- question that has no answer yet, and inventing `"draw"` would hide
--- a loop that terminated one round early.
---@param g table Game
---@return string|nil winner `"p1"` / `"p2"` / `"draw"`, or nil when unfinished
function M.winner(g)
    if not M.is_over(g) then
        return nil
    end
    if g.p1.my_points > g.p2.my_points then
        return "p1"
    elseif g.p2.my_points > g.p1.my_points then
        return "p2"
    end
    return "draw"
end

-- ─── Encoding ───────────────────────────────────────────────────────

--- Encode a per-player state as one line over the module alphabet.
---
--- Layout: `R<round>H<hand>P<mine><theirs>O<opponent history>`. The
--- hand is emitted sorted so two states that differ only in deal order
--- encode identically.
---@param state table Per-player state
---@return string encoded
function M.encode(state)
    if type(state) ~= "table" then
        error("card_duel.encode: state must be a table, got " .. type(state))
    end
    local hand = sorted_copy(state.my_hand)
    local parts = { "R", tostring(state.round), "H" }
    for _, rank in ipairs(hand) do
        parts[#parts + 1] = tostring(rank)
    end
    parts[#parts + 1] = "P"
    parts[#parts + 1] = tostring(state.my_points)
    parts[#parts + 1] = tostring(state.opp_points)
    parts[#parts + 1] = "O"
    for _, rank in ipairs(state.opp_played or {}) do
        parts[#parts + 1] = tostring(rank)
    end
    return table.concat(parts)
end

-- ─── Policies ───────────────────────────────────────────────────────

--- Teacher style: press while behind or level, conserve while ahead.
---@param state table Per-player state
---@return integer action
function M.policy_aggressive(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_aggressive: no legal action (empty hand)")
    end
    if state.my_points <= state.opp_points then
        return legal[#legal]
    end
    return legal[1]
end

--- Uniform choice over the legal actions, driven by a caller-owned RNG.
---
--- The RNG is a parameter rather than module state so a self-play loop
--- stays reproducible from its own seed.
---@param state table Per-player state
---@param rng userdata `alc.math.rng_create` handle
---@return integer action
function M.policy_random(state, rng)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_random: no legal action (empty hand)")
    end
    local math_ns = require_rng()
    return legal[math_ns.rng_int(rng, 1, #legal)]
end

-- ─── Strategy entry ─────────────────────────────────────────────────

--- Return the encoded opening state for a seed.
---@param ctx table `{ seed? }`
---@return table result `{ result = <encoded state> }`
function M.run(ctx)
    ctx = ctx or {}
    local seed = tonumber(ctx.seed) or 1
    local g = M.new_game(seed)
    return { result = M.encode(g.p1) }
end

return M
