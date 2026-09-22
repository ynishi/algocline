--- pick — a sequential-choice toy that the engine's `alc.nn` tests drive.
---
--- This is a test fixture, not a product and not a game. It is the
--- smallest domain with the shape the gate and hook tests need to fence:
--- a per-state legal set that shrinks every turn, a deterministic
--- teacher whose move depends on the state (so cloning it means reading
--- the state, not memorising a constant), a fixed-width text encoding
--- that fits the `gpt2 tiny` context, and a decode gate that maps model
--- logits back onto the legal set.
---
--- Two seats each hold `TURNS` values drawn from `MIN_VALUE..MAX_VALUE`
--- with replacement. Every turn both seats commit one value from their
--- pool; the higher value scores a point, a tie scores nothing, both
--- values leave their pools. After `TURNS` turns the higher score wins.
---
--- Host access is limited to `alc.math.rng_create` / `alc.math.rng_int`
--- (episode generation) and, on the decode side, `alc.math.softmax` /
--- `alc.math.entropy` plus the `generate_session` method of an nn handle.
--- A spec that runs without the bridge stubs `alc.math` itself.

local M = {}

local TURNS = 5
local MIN_VALUE = 1
local MAX_VALUE = 9
--- Both seats contribute one row per turn.
local ROWS_PER_EPISODE = TURNS * 2
local RNG_STRIDE = 1000003

M.TURNS = TURNS
M.MIN_VALUE = MIN_VALUE
M.MAX_VALUE = MAX_VALUE
M.ROWS_PER_EPISODE = ROWS_PER_EPISODE

-- ─── Alphabet ───────────────────────────────────────────────────────

--- Char alphabet, indexed by model token id (`id = index - 1`; index 1
--- is the padding token). Seventeen entries keep the whole alphabet
--- inside the `gpt2 tiny` vocabulary of 64.
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
    "T",
    "L",
    "S",
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

--- Char-to-token-id map shared by the corpus builder and the gate.
--- Fresh copies, so a caller cannot corrupt the module-level maps.
---@return table vocab `{ size, pad_id, to_id, to_char }`
function M.vocab()
    local to_id, to_char = {}, {}
    for ch, id in pairs(TO_ID) do
        to_id[ch] = id
    end
    for id, ch in pairs(TO_CHAR) do
        to_char[id] = ch
    end
    return { size = #CHARS, pad_id = TO_ID["\0"], to_id = to_id, to_char = to_char }
end

--- Map a string over the alphabet to token ids. Errors on an unknown
--- character instead of substituting a filler.
---@param text string
---@return integer[] ids
function M.to_ids(text)
    if type(text) ~= "string" then
        error("pick.to_ids: text must be a string, got " .. type(text))
    end
    local ids = {}
    for i = 1, #text do
        local ch = text:sub(i, i)
        local id = TO_ID[ch]
        if id == nil then
            error(string.format("pick.to_ids: char %q at %d is outside the alphabet", ch, i))
        end
        ids[#ids + 1] = id
    end
    return ids
end

-- ─── Helpers ────────────────────────────────────────────────────────

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
    if
        type(alc) ~= "table"
        or type(alc.math) ~= "table"
        or type(alc.math.rng_create) ~= "function"
        or type(alc.math.rng_int) ~= "function"
    then
        error("pick: alc.math.rng_create / rng_int are unavailable")
    end
    return alc.math
end

local function make_seat(turn, pool, my_score, opp_score, seen)
    return { turn = turn, pool = pool, my_score = my_score, opp_score = opp_score, seen = seen }
end

--- Remove one instance of `value` from `pool`, returning a new list.
local function without_one(pool, value)
    local out = {}
    local dropped = false
    for _, v in ipairs(pool) do
        if v == value and not dropped then
            dropped = true
        else
            out[#out + 1] = v
        end
    end
    if not dropped then
        error(string.format("pick.apply: value %s is not in the pool", tostring(value)))
    end
    return out
end

local function appended(list, value)
    local out = copy_list(list)
    out[#out + 1] = value
    return out
end

-- ─── Episode progression ────────────────────────────────────────────

--- Deal a fresh episode. One seed fully determines both pools.
---@param seed integer
---@return table episode `{ turn, seed, a, b }` where `a` / `b` are seat states
function M.new_episode(seed)
    if type(seed) ~= "number" then
        error("pick.new_episode: seed must be a number, got " .. type(seed))
    end
    local math_ns = require_rng()
    local rng = math_ns.rng_create(seed)
    local pa, pb = {}, {}
    for _ = 1, TURNS do
        pa[#pa + 1] = math_ns.rng_int(rng, MIN_VALUE, MAX_VALUE)
    end
    for _ = 1, TURNS do
        pb[#pb + 1] = math_ns.rng_int(rng, MIN_VALUE, MAX_VALUE)
    end
    return {
        turn = 1,
        seed = seed,
        a = make_seat(1, sorted_copy(pa), 0, 0, {}),
        b = make_seat(1, sorted_copy(pb), 0, 0, {}),
    }
end

--- Distinct values still in the pool, ascending.
---@param state table Seat state
---@return integer[] values
function M.legal(state)
    if type(state) ~= "table" or type(state.pool) ~= "table" then
        error("pick.legal: state.pool must be a table")
    end
    local seen, out = {}, {}
    for _, v in ipairs(state.pool) do
        if not seen[v] then
            seen[v] = true
            out[#out + 1] = v
        end
    end
    table.sort(out)
    return out
end

--- Resolve one turn. Returns a new episode; the input is not mutated.
---@param ep table Episode
---@param va integer Seat a's value
---@param vb integer Seat b's value
---@return table episode
function M.apply(ep, va, vb)
    if M.is_over(ep) then
        error("pick.apply: the episode is over")
    end
    local a, b = ep.a, ep.b
    local pa = without_one(a.pool, va)
    local pb = without_one(b.pool, vb)
    local sa, sb = a.my_score, b.my_score
    if va > vb then
        sa = sa + 1
    elseif vb > va then
        sb = sb + 1
    end
    local turn = ep.turn + 1
    return {
        turn = turn,
        seed = ep.seed,
        a = make_seat(turn, pa, sa, sb, appended(a.seen, vb)),
        b = make_seat(turn, pb, sb, sa, appended(b.seen, va)),
    }
end

---@param ep table Episode
---@return boolean
function M.is_over(ep)
    return ep.turn > TURNS
end

--- Fixed-width text view of one seat: turn, pool, both scores, the
--- values the other seat has shown so far. The pool loses one char per
--- turn while `seen` gains one, so every state encodes to 12 chars and
--- prompt plus action plus newline is 15 tokens.
---@param state table Seat state
---@return string
function M.encode(state)
    if type(state) ~= "table" then
        error("pick.encode: state must be a table, got " .. type(state))
    end
    local parts = { "T", tostring(state.turn), "L" }
    for _, v in ipairs(sorted_copy(state.pool)) do
        parts[#parts + 1] = tostring(v)
    end
    parts[#parts + 1] = "S"
    parts[#parts + 1] = tostring(state.my_score)
    parts[#parts + 1] = tostring(state.opp_score)
    parts[#parts + 1] = "O"
    for _, v in ipairs(state.seen or {}) do
        parts[#parts + 1] = tostring(v)
    end
    return table.concat(parts)
end

-- ─── Policies ───────────────────────────────────────────────────────

--- The teacher: commit the highest legal value while level or behind,
--- the lowest once ahead. Deterministic, and a function of the score
--- gap, so a model that clones it has to read the state.
---@param state table Seat state
---@return integer value
function M.teacher(state)
    local legal = M.legal(state)
    if #legal == 0 then
        error("pick.teacher: no legal value (empty pool)")
    end
    if state.my_score <= state.opp_score then
        return legal[#legal]
    end
    return legal[1]
end

--- Uniform choice over the legal set from a caller-owned RNG.
---@param state table Seat state
---@param rng table RNG from `alc.math.rng_create`
---@return integer value
function M.random_policy(state, rng)
    local legal = M.legal(state)
    if #legal == 0 then
        error("pick.random_policy: no legal value (empty pool)")
    end
    return legal[require_rng().rng_int(rng, 1, #legal)]
end

-- ─── Corpus ─────────────────────────────────────────────────────────

local function make_row(state, value, ctx_len, pad_id)
    local ids = M.to_ids(M.encode(state) .. ">" .. tostring(value) .. "\n")
    if #ids > ctx_len then
        error(
            string.format(
                "pick.build_corpus: encoded line needs %d tokens but the context is %d",
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

--- Supervised corpus that teaches `policy`. Seat a plays the policy;
--- seat b plays at random but its states are still labelled with the
--- policy's answer, which widens state coverage without changing the
--- target function. One episode is `ROWS_PER_EPISODE` rows.
---@param policy fun(state: table): integer
---@param opts table `{ ctx_len, episodes, seed?, pad_id? }`
---@return integer[][] rows Token id rows, each `ctx_len` long
function M.build_corpus(policy, opts)
    if type(policy) ~= "function" then
        error("pick.build_corpus: policy must be a function, got " .. type(policy))
    end
    if type(opts) ~= "table" then
        error("pick.build_corpus: opts must be a table, got " .. type(opts))
    end
    local ctx_len = tonumber(opts.ctx_len)
    if ctx_len == nil or ctx_len < 1 then
        error("pick.build_corpus: opts.ctx_len must be a positive number")
    end
    ctx_len = math.floor(ctx_len)
    local episodes = tonumber(opts.episodes)
    if episodes == nil or episodes < 1 then
        error("pick.build_corpus: opts.episodes must be a positive number")
    end
    episodes = math.floor(episodes)
    local seed = math.floor(tonumber(opts.seed) or 1)
    local pad_id = opts.pad_id
    if pad_id == nil then
        pad_id = TO_ID["\0"]
    end
    if type(pad_id) ~= "number" then
        error("pick.build_corpus: opts.pad_id must be a number, got " .. type(pad_id))
    end

    local math_ns = require_rng()
    local rows = {}
    for i = 1, episodes do
        local ep = M.new_episode(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not M.is_over(ep) do
            local va = policy(ep.a)
            rows[#rows + 1] = make_row(ep.a, va, ctx_len, pad_id)
            rows[#rows + 1] = make_row(ep.b, policy(ep.b), ctx_len, pad_id)
            ep = M.apply(ep, va, M.random_policy(ep.b, rng))
        end
    end
    return rows
end

--- Three fixed states covering the turn and score-gap branches the
--- teacher reads. Fresh tables on every call.
---@return table[] states
function M.check_states()
    return {
        { turn = 1, pool = { 9, 7, 5, 3, 1 }, my_score = 0, opp_score = 0, seen = {} },
        { turn = 2, pool = { 1, 3, 5, 7 }, my_score = 1, opp_score = 0, seen = { 4 } },
        { turn = 3, pool = { 3, 5, 8 }, my_score = 1, opp_score = 1, seen = { 2, 6 } },
    }
end

-- ─── Decode gate ────────────────────────────────────────────────────

--- Token ids that spell a legal value for this state, as `id -> value`,
--- plus the ascending value list.
---@param state table Seat state
---@return table by_id, integer[] values
function M.legal_ids(state)
    local by_id, values = {}, {}
    for _, v in ipairs(M.legal(state)) do
        local id = TO_ID[tostring(v)]
        if id == nil then
            error(string.format("pick.legal_ids: value %s has no token id", tostring(v)))
        end
        by_id[id] = v
        values[#values + 1] = v
    end
    if #values == 0 then
        error("pick.legal_ids: state has no legal value")
    end
    return by_id, values
end

--- Accept an nn handle by capability rather than by Lua type.
---
--- `alc.nn.card.load_handle` / `load_ckpt` return **userdata**; a spec
--- without the bridge hands in a table that mimics it. Both are fine as
--- long as `generate_session` is callable, and the read goes through
--- `pcall` because indexing a userdata without that field can raise.
---@param handle any
---@return any handle The same value, once checked
function M.require_handle(handle)
    local kind = type(handle)
    if kind ~= "table" and kind ~= "userdata" then
        error("pick.require_handle: handle must be a table or userdata, got " .. kind)
    end
    local ok, method = pcall(function()
        return handle.generate_session
    end)
    if not ok or type(method) ~= "function" then
        error(
            string.format(
                "pick.require_handle: %s has no generate_session method; "
                    .. "expected a handle from alc.nn.card.load_handle or load_ckpt",
                kind
            )
        )
    end
    return handle
end

--- One gated greedy decision: the highest-ranked token whose id spells a
--- legal value. Also reports whether the ungated argmax was already
--- legal and whether the gate had to move away from it.
---@param handle any nn handle (see `require_handle`)
---@param state table Seat state
---@return table decision `{ value, raw_legal, gated }`
function M.decide(handle, state)
    M.require_handle(handle)
    local by_id = M.legal_ids(state)
    local session = handle:generate_session(M.to_ids(M.encode(state) .. ">"))
    local logits = session:next_logits()

    local raw = logits:argmax()
    local raw_legal = by_id[raw] ~= nil

    for _, entry in ipairs(logits:top(logits:vocab())) do
        local v = by_id[entry.id]
        if v ~= nil then
            return { value = v, raw_legal = raw_legal, gated = entry.id ~= raw }
        end
    end
    -- `top(vocab)` enumerates the whole vocabulary and the legal set is
    -- non-empty, so this is a loud failure for a future ranking change
    -- that starts truncating, not a fallback.
    error("pick.decide: no legal token found in the full logit ranking")
end

--- Normalised Shannon entropy of the model's next-token distribution
--- restricted to the legal values of `state`, at `temperature`.
---
--- In `[0, 1]`: `0` when the model is certain (or only one value is
--- legal), `1` when it is uniform over the legal set. The value is a
--- metric that consumes a handle, which is the shape the checkpoint
--- hook test needs; its magnitude after a handful of steps is not
--- something any test should pin.
---@param handle any nn handle (see `require_handle`)
---@param state table Seat state
---@param temperature number|nil Finite positive; defaults to 1
---@return number entropy
function M.legal_entropy(handle, state, temperature)
    M.require_handle(handle)
    if not (alc and alc.math and type(alc.math.softmax) == "function") then
        error("pick.legal_entropy: alc.math.softmax is unavailable")
    end
    if type(alc.math.entropy) ~= "function" then
        error("pick.legal_entropy: alc.math.entropy is unavailable")
    end
    local t = temperature
    if t == nil then
        t = 1.0
    end
    if type(t) ~= "number" or t ~= t or t == math.huge or t <= 0 then
        error("pick.legal_entropy: temperature must be a finite positive number")
    end

    local by_id, values = M.legal_ids(state)
    if #values == 1 then
        return 0.0
    end
    local session = handle:generate_session(M.to_ids(M.encode(state) .. ">"))
    local logits = session:next_logits()

    local raw = {}
    local seen = 0
    for _, entry in ipairs(logits:top(logits:vocab())) do
        local v = by_id[entry.id]
        if v ~= nil and raw[v] == nil then
            raw[v] = entry.value / t
            seen = seen + 1
            if seen == #values then
                break
            end
        end
    end
    if seen ~= #values then
        error("pick.legal_entropy: a legal value is missing from the logits ranking")
    end
    local scaled = {}
    for i, v in ipairs(values) do
        scaled[i] = raw[v]
    end
    return alc.math.entropy(alc.math.softmax(scaled)) / math.log(#values)
end

return M
