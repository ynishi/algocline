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
--- The same rules carry a whole zoo of such styles (`M.STYLES`): the
--- other five branch on the round, on the opponent history or on nothing
--- at all, which lets one set of rules train and compare several NPCs.
---
--- ## Entry contract
---
--- - `new_game` / `apply` / `is_over` / `winner` — game progression
--- - `legal_actions` / `encode` / `vocab` / `to_ids` — NPC-facing view
--- - `policy_<style>` for every name in `STYLES` — deterministic styles
--- - `policy_random` — the random opponent used by self-play
--- - `build_corpus` — supervised training lines for any policy
--- - `sample_states` / `compile_policy` — sandbox and validation for a
---   synthesised (LLM-written) policy chunk
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

--- Training lines one playout contributes: both seats speak once per
--- round, and a game runs `HAND_SIZE` rounds.
local ROWS_PER_GAME = 2 * HAND_SIZE

--- Stride between the deal seed and the opponent RNG seed of a playout,
--- so two playouts of the same batch never share an RNG stream.
local RNG_STRIDE = 7919

M.HAND_SIZE = HAND_SIZE
M.MIN_RANK = MIN_RANK
M.MAX_RANK = MAX_RANK
M.ROWS_PER_GAME = ROWS_PER_GAME

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

--- Timid style: always play the lowest legal rank.
---@param state table Per-player state
---@return integer action
function M.policy_timid(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_timid: no legal action (empty hand)")
    end
    return legal[1]
end

--- Bold style: always play the highest legal rank.
---@param state table Per-player state
---@return integer action
function M.policy_bold(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_bold: no legal action (empty hand)")
    end
    return legal[#legal]
end

--- Defensive style: the mirror image of the teacher.
---
--- Holds the high cards back while behind or level and spends them once
--- ahead, so a model that learned `policy_aggressive` scores zero on
--- this style rather than half its states by accident.
---@param state table Per-player state
---@return integer action
function M.policy_defensive(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_defensive: no legal action (empty hand)")
    end
    if state.my_points <= state.opp_points then
        return legal[1]
    end
    return legal[#legal]
end

--- Late bloomer style: coast through the opening, press at the end.
---
--- The branch is on the round rather than the score, so the style is
--- only reproducible from the round field of the encoded state.
---@param state table Per-player state
---@return integer action
function M.policy_late_bloomer(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_late_bloomer: no legal action (empty hand)")
    end
    if state.round <= 2 then
        return legal[1]
    end
    return legal[#legal]
end

--- Mimic style: answer the opponent's last card with the nearest rank.
---
--- Ties in distance resolve to the lower rank so the style stays
--- deterministic, and an empty history (round one) falls back to the
--- middle of the legal actions rather than erroring: the state is legal,
--- there is simply nothing to mirror yet.
---@param state table Per-player state
---@return integer action
function M.policy_mimic(state)
    local legal = M.legal_actions(state)
    if #legal == 0 then
        error("card_duel.policy_mimic: no legal action (empty hand)")
    end
    local history = state.opp_played or {}
    local target = history[#history]
    if target == nil then
        return legal[math.ceil(#legal / 2)]
    end
    local best, best_gap = legal[1], math.abs(legal[1] - target)
    for i = 2, #legal do
        local gap = math.abs(legal[i] - target)
        if gap < best_gap then
            best, best_gap = legal[i], gap
        end
    end
    return best
end

--- Canonical style names, in the order the trainer and the tournament
--- iterate them.
---
--- Every entry `s` has a matching `M["policy_" .. s]`; callers that take
--- a style name validate against this list instead of hard-coding their
--- own copy.
M.STYLES = { "timid", "bold", "aggressive", "defensive", "late_bloomer", "mimic" }

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

-- ─── Corpus ─────────────────────────────────────────────────────────

--- Encode one training line and pad it to the model context window.
---
--- The line is `<encoded state>><action>\n`. A line that does not fit
--- is a loud error rather than a truncation: a truncated line teaches
--- the model a state it can never be asked about at decode time.
local function make_row(state, action, ctx_len, pad_id)
    local ids = M.to_ids(M.encode(state) .. ">" .. tostring(action) .. "\n")
    if #ids > ctx_len then
        error(
            string.format(
                "card_duel.build_corpus: encoded line needs %d tokens but the context is %d",
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

--- Build the supervised corpus that teaches `policy` to a model.
---
--- Both seats contribute a line every round: player one actually plays
--- the `policy` move, player two plays a random move but its state is
--- still labelled with the `policy` action, which widens the state
--- coverage without changing the target function. One playout is
--- therefore `ROWS_PER_GAME` rows.
---
--- `alc.nn.data.synthetic` walks its rows once, so a trainer asking for
--- `steps * batch` rows needs `games >= steps * batch / ROWS_PER_GAME`;
--- computing that floor is left to the caller, which is the only side
--- that knows the training budget.
---@param policy fun(state: table): integer Labelling policy
---@param opts table `{ ctx_len, games, seed?, pad_id? }`
---@return integer[][] rows Token id rows, each `ctx_len` long
function M.build_corpus(policy, opts)
    if type(policy) ~= "function" then
        error("card_duel.build_corpus: policy must be a function, got " .. type(policy))
    end
    if type(opts) ~= "table" then
        error("card_duel.build_corpus: opts must be a table, got " .. type(opts))
    end
    local ctx_len = tonumber(opts.ctx_len)
    if ctx_len == nil or ctx_len < 1 then
        error("card_duel.build_corpus: opts.ctx_len must be a positive number")
    end
    ctx_len = math.floor(ctx_len)
    local games = tonumber(opts.games)
    if games == nil or games < 1 then
        error("card_duel.build_corpus: opts.games must be a positive number")
    end
    games = math.floor(games)
    local seed = math.floor(tonumber(opts.seed) or 1)
    local pad_id = opts.pad_id
    if pad_id == nil then
        pad_id = TO_ID["\0"]
    end
    if type(pad_id) ~= "number" then
        error("card_duel.build_corpus: opts.pad_id must be a number, got " .. type(pad_id))
    end

    local math_ns = require_rng()
    local rows = {}
    for i = 1, games do
        local g = M.new_game(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not M.is_over(g) do
            local a1 = policy(g.p1)
            rows[#rows + 1] = make_row(g.p1, a1, ctx_len, pad_id)
            rows[#rows + 1] = make_row(g.p2, policy(g.p2), ctx_len, pad_id)
            g = M.apply(g, a1, M.policy_random(g.p2, rng))
        end
    end
    return rows
end

-- ─── Synthesised policies ───────────────────────────────────────────
--
-- A persona NPC starts as a Lua chunk written by an LLM
-- (`examples/gameai/bake_card_duel_persona.lua`). Such a chunk is
-- never loaded raw: it is compiled into a restricted environment and
-- then has to answer a batch of sampled states legally and
-- deterministically before it is allowed to label a single training
-- line.

--- Collect per-player states from random self-play.
---
--- The states are the ones a policy is actually asked about during a
--- game rather than a hand-picked list, and the walk is fully
--- determined by `seed`, so a validation verdict is reproducible.
---@param opts table|nil `{ games?, seed? }`
---@return table[] states `games * ROWS_PER_GAME` per-player states
function M.sample_states(opts)
    opts = opts or {}
    local games = math.floor(tonumber(opts.games) or 20)
    if games < 1 then
        error("card_duel.sample_states: opts.games must be a positive number")
    end
    local seed = math.floor(tonumber(opts.seed) or 1)
    local math_ns = require_rng()
    local states = {}
    for i = 1, games do
        local g = M.new_game(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not M.is_over(g) do
            states[#states + 1] = g.p1
            states[#states + 1] = g.p2
            g = M.apply(g, M.policy_random(g.p1, rng), M.policy_random(g.p2, rng))
        end
    end
    return states
end

--- Environment a synthesised chunk is compiled into.
---
--- Only the pure parts of the standard library are reachable, and
--- `math` / `table` are shallow copies so a chunk cannot reach the host
--- tables through them. There is no `load`, no `setmetatable`, no
--- `os` / `io` / `require`, so a chunk can compute but cannot observe
--- or change anything outside its own argument. `math.random` is
--- present but useless: the determinism check below rejects any policy
--- that answers the same state twice with two different ranks.
local function sandbox_env()
    local math_copy, table_copy = {}, {}
    for k, v in pairs(math) do
        math_copy[k] = v
    end
    for k, v in pairs(table) do
        table_copy[k] = v
    end
    return {
        math = math_copy,
        table = table_copy,
        ipairs = ipairs,
        pairs = pairs,
    }
end

--- Copy of a per-player state, so a chunk cannot mutate the live game.
local function copy_state(state)
    return make_state(
        state.round,
        copy_list(state.my_hand),
        state.my_points,
        state.opp_points,
        copy_list(state.opp_played or {})
    )
end

--- Whether `action` is one of the ranks `state` may still play.
local function is_legal_action(state, action)
    if type(action) ~= "number" or action ~= math.floor(action) then
        return false
    end
    for _, rank in ipairs(M.legal_actions(state)) do
        if rank == action then
            return true
        end
    end
    return false
end

--- Compile and validate a synthesised policy chunk.
---
--- `source` is expected to be `return function(state) ... end`. It is
--- loaded in text mode only (never bytecode) into `sandbox_env`, and
--- the returned function is then asked about every state in
--- `opts.states` (sampled from random self-play when the caller passes
--- none). A candidate is accepted only when, for every state, the call
--- succeeds, the answer is a legal rank, and a second call with the
--- same state returns the same rank.
---
--- Every rejection is a loud error naming the state and the reason, so
--- a caller driving an LLM can feed the message straight back into the
--- next synthesis attempt.
---
--- The accepted policy is wrapped so it receives a copy of the state: a
--- chunk that sorts or empties `my_hand` would otherwise corrupt the
--- game it is labelling, which is a silent data fault rather than a
--- visible one.
---@param source string Lua chunk returning a policy function
---@param opts table|nil `{ states?, games?, seed?, chunk_name? }`
---@return fun(state: table): integer policy
function M.compile_policy(source, opts)
    if type(source) ~= "string" then
        error("card_duel.compile_policy: source must be a string, got " .. type(source))
    end
    opts = opts or {}
    local chunk_name = opts.chunk_name or "synthesised_policy"
    local chunk, load_err = load(source, "=" .. chunk_name, "t", sandbox_env())
    if chunk == nil then
        error("card_duel.compile_policy: source does not compile: " .. tostring(load_err))
    end
    local ran, policy = pcall(chunk)
    if not ran then
        error(
            "card_duel.compile_policy: chunk raised while returning the policy: "
                .. tostring(policy)
        )
    end
    if type(policy) ~= "function" then
        error("card_duel.compile_policy: chunk must return a function, got " .. type(policy))
    end

    local states = opts.states
    if states == nil then
        states = M.sample_states({ games = opts.games, seed = opts.seed })
    end
    if type(states) ~= "table" or #states == 0 then
        error("card_duel.compile_policy: opts.states must hold at least one state")
    end

    local guarded = function(state)
        return policy(copy_state(state))
    end

    for i, state in ipairs(states) do
        local ok, action = pcall(guarded, state)
        if not ok then
            error(
                string.format(
                    "card_duel.compile_policy: policy raised on state %d (%s): %s",
                    i,
                    M.encode(state),
                    tostring(action)
                )
            )
        end
        if not is_legal_action(state, action) then
            error(
                string.format(
                    "card_duel.compile_policy: policy answered %s on state %d (%s), "
                        .. "which is not one of the legal ranks %s",
                    tostring(action),
                    i,
                    M.encode(state),
                    table.concat(M.legal_actions(state), ", ")
                )
            )
        end
        local ok_again, again = pcall(guarded, state)
        if not ok_again or again ~= action then
            error(
                string.format(
                    "card_duel.compile_policy: policy is not deterministic on state %d (%s): "
                        .. "%s then %s",
                    i,
                    M.encode(state),
                    tostring(action),
                    tostring(again)
                )
            )
        end
    end

    return guarded
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
