--- card_duel_npc — SLM card duel NPC with a legal-action decode gate
---
--- Wraps a tuned tiny SLM (a Card produced by
--- `examples/gameai/train_card_duel_npc.lua`) into an algocline
--- strategy that picks one card per round. The model never touches the
--- game state: `card_duel.legal_actions` enumerates the choices and the
--- gate walks the logit ranking until it hits one of them, so an
--- illegal move is structurally impossible while the raw argmax is
--- still reported as telemetry.
---
--- ## Usage
---
--- ```lua
--- local npc = require("card_duel_npc")
--- return npc.run({
---     task = alc.json_encode({ mode = "decide", state = state }),
---     card_alias = "card_duel_npc",
--- })
--- ```
---
--- ## Algorithm
---
--- 1. Resolve the Card behind `card_alias` and load an `NnHandle`
---    (cached per VM, since a decode session is cheap but a model load
---    is not).
--- 2. Encode the state with `card_duel.encode` and append `">"`, the
---    separator the training lines use.
--- 3. Open a generation session over those token ids and read one
---    logits row.
--- 4. Record whether the raw argmax is a legal action (`raw_legal`),
---    then scan `logits:top(vocab)` in descending order and take the
---    first token that maps to a legal rank. That is greedy decoding
---    restricted to the legal subset.
---
--- ## Entry contract
---
--- `ctx.task` is a JSON object with a `mode` field:
---
--- - `decide` — `{ state }`; returns `action=<rank> legal=true raw_legal=<bool> gated=<bool>`
--- - `determinism` — `{ state }`; decodes twice through independent
---   sessions and returns `deterministic=<bool> action=<rank>`
--- - `selfplay` — `{ games, seed }`; plays the NPC against
---   `card_duel.policy_random` and returns `winrate=<x.xx> illegal=<n>`
---
--- ## Caveats
---
--- `alc.llm` is never called, so an eval over this strategy runs to
--- completion without a host round trip. The cost is that the package
--- only works in a build with the `nn` feature: without `alc.nn` the
--- entry fails loudly at Card load rather than degrading to a
--- hand-written policy, which would silently score the wrong thing.
---
--- The self-play win rate counts a draw as half a win, matching the
--- usual convention for symmetric two-player games. With five rounds
--- and a nine-rank deck draws are common enough that scoring them as
--- losses would depress the number for a reason unrelated to play.

local duel = require("card_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "card_duel_npc",
    version = "0.1.0",
    description = "Card duel NPC driven by a tuned tiny SLM with a legal-action decode gate",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `card_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            task = T.string:describe(
                "JSON object with a mode field: decide / determinism / selfplay"
            ),
            card_alias = T.string
                :is_optional()
                :describe("Card alias holding the tuned model (default: card_duel_npc)"),
        }),
        result = T.string:describe("Flat key=value summary of the requested mode"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

--- Default Card alias written by the training script.
local DEFAULT_ALIAS = "card_duel_npc"

--- Loaded handles keyed by alias, per VM.
---
--- A generation session is cheap; reloading the safetensors bundle for
--- every decision is not, and a self-play run makes one decision per
--- round per game.
local handle_cache = {}

local VOCAB = duel.vocab()

-- ─── Host surface guards ────────────────────────────────────────────

local function require_nn()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("card_duel_npc: alc.nn.card is unavailable; build algocline with --features nn")
    end
    return alc.nn
end

--- Resolve the alias to a loaded model handle.
---
--- The Card layer has no alias entry of its own, so the generic
--- `alc.card.get_by_alias` resolves the pin that the training script
--- writes with `alc.card.alias_set`.
local function resolve_handle(alias)
    local cached = handle_cache[alias]
    if cached then
        return cached
    end
    local nn = require_nn()
    if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
        error("card_duel_npc: alc.card.get_by_alias is unavailable")
    end
    local card = alc.card.get_by_alias(alias)
    if not card then
        error(
            string.format(
                "card_duel_npc: no Card bound to alias %q; run examples/gameai/train_card_duel_npc.lua first",
                alias
            )
        )
    end
    local card_id = card.card_id
    if type(card_id) ~= "string" or #card_id == 0 then
        error(string.format("card_duel_npc: alias %q resolved to a Card without card_id", alias))
    end
    local handle = nn.card.load_handle(card_id)
    handle_cache[alias] = handle
    return handle
end

-- ─── Decode ─────────────────────────────────────────────────────────

--- Token ids that spell a legal action for this state.
local function legal_token_ids(state)
    local ids, ranks = {}, {}
    for _, rank in ipairs(duel.legal_actions(state)) do
        local id = VOCAB.to_id[tostring(rank)]
        if id == nil then
            error(string.format("card_duel_npc: rank %s has no token id", tostring(rank)))
        end
        ids[id] = rank
        ranks[#ranks + 1] = rank
    end
    if #ranks == 0 then
        error("card_duel_npc: state has no legal action")
    end
    return ids, ranks
end

--- One gated greedy decision.
---
--- Returns the chosen rank plus the two telemetry flags the eval
--- scenario fences on: whether the ungated argmax was already legal,
--- and whether the gate had to move away from it.
---@param handle userdata NnHandle
---@param state table Per-player state
---@return table decision `{ rank, raw_legal, gated }`
local function decide(handle, state)
    local legal_ids = legal_token_ids(state)
    local prompt = duel.encode(state) .. ">"
    local session = handle:generate_session(duel.to_ids(prompt))
    local logits = session:next_logits()

    local raw = logits:argmax()
    local raw_legal = legal_ids[raw] ~= nil

    local ranked = logits:top(logits:vocab())
    for _, entry in ipairs(ranked) do
        local rank = legal_ids[entry.id]
        if rank ~= nil then
            return { rank = rank, raw_legal = raw_legal, gated = entry.id ~= raw }
        end
    end
    -- Unreachable: `top(vocab)` enumerates the whole vocabulary and the
    -- legal set is non-empty, so a legal id is always present. Kept as a
    -- loud failure rather than a silent fallback in case a future
    -- ranking change starts truncating.
    error("card_duel_npc: no legal token found in the full logit ranking")
end

-- ─── Modes ──────────────────────────────────────────────────────────

local function decode_state(req)
    local state = req.state
    if type(state) ~= "table" then
        error("card_duel_npc: task.state must be an object")
    end
    if type(state.my_hand) ~= "table" then
        error("card_duel_npc: task.state.my_hand must be an array")
    end
    state.my_points = tonumber(state.my_points) or 0
    state.opp_points = tonumber(state.opp_points) or 0
    state.round = tonumber(state.round) or 1
    state.opp_played = state.opp_played or {}
    return state
end

local function mode_decide(handle, req)
    local state = decode_state(req)
    local d = decide(handle, state)
    return string.format(
        "action=%d legal=true raw_legal=%s gated=%s",
        d.rank,
        tostring(d.raw_legal),
        tostring(d.gated)
    )
end

local function mode_determinism(handle, req)
    local state = decode_state(req)
    local first = decide(handle, state)
    local second = decide(handle, state)
    local same = first.rank == second.rank
    return string.format("deterministic=%s action=%d", tostring(same), first.rank)
end

local function mode_selfplay(handle, req)
    local games = math.floor(tonumber(req.games) or 20)
    local seed = math.floor(tonumber(req.seed) or 1)
    if games <= 0 then
        error("card_duel_npc: task.games must be a positive integer")
    end

    local score, illegal = 0.0, 0
    for i = 1, games do
        local g = duel.new_game(seed + i)
        local rng = alc.math.rng_create(seed * 1000 + i)
        while not duel.is_over(g) do
            local d = decide(handle, g.p1)
            if not d.raw_legal then
                illegal = illegal + 1
            end
            g = duel.apply(g, d.rank, duel.policy_random(g.p2, rng))
        end
        local w = duel.winner(g)
        if w == "p1" then
            score = score + 1.0
        elseif w == "draw" then
            score = score + 0.5
        end
    end
    return string.format("winrate=%.2f illegal=%d", score / games, illegal)
end

-- ─── Strategy entry ─────────────────────────────────────────────────

---@param ctx table `{ task, card_alias? }`
---@return table result `{ result = <flat key=value summary> }`
function M.run(ctx)
    ctx = ctx or {}
    local task = ctx.task
    if type(task) ~= "string" then
        error("card_duel_npc: ctx.task must be a JSON string, got " .. type(task))
    end
    local req = alc.json_decode(task)
    if type(req) ~= "table" or type(req.mode) ~= "string" then
        error("card_duel_npc: ctx.task must decode to an object with a mode field")
    end

    local handle = resolve_handle(ctx.card_alias or DEFAULT_ALIAS)

    local out
    if req.mode == "decide" then
        out = mode_decide(handle, req)
    elseif req.mode == "determinism" then
        out = mode_determinism(handle, req)
    elseif req.mode == "selfplay" then
        out = mode_selfplay(handle, req)
    else
        error(
            string.format(
                "card_duel_npc: unknown mode %q (expected decide / determinism / selfplay)",
                req.mode
            )
        )
    end

    return { result = out }
end

--- Drop cached handles. Exposed for the training script, which binds a
--- new Card to the alias inside a VM that may already hold the old one.
function M.reset_cache()
    handle_cache = {}
end

return M
