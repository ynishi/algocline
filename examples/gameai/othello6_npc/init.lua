--- othello6_npc — 6x6 Othello NPC with a legal-move decode gate
---
--- Wraps a tuned tiny SLM (a Card produced by the 6x6 Othello training
--- driver) into an algocline strategy that answers one move per turn.
--- The model never touches the rules: `othello6.legal_actions`
--- enumerates the moves the position allows and the gate walks the logit
--- ranking until it hits one of them, so an illegal placement is
--- structurally impossible while the raw argmax is still reported as
--- telemetry.
---
--- ## Usage
---
--- ```lua
--- local npc = require("othello6_npc")
--- return npc.run({
---     task = alc.json_encode({ mode = "decide", state = position }),
---     card_alias = "othello6_npc_d2_corner",
---     depth = 2,
---     style = "corner",
--- })
--- ```
---
--- ## Algorithm
---
--- 1. Resolve the Card behind the alias and load an `NnHandle` (cached
---    per VM, since a decode session is cheap but a model load is not).
--- 2. Encode the position with `othello6.encode`. The line is the
---    opening marker and the moves that reached the position, and
---    nothing else: no separator, because a training row has none
---    either — a row is one game written move by move behind the same
---    marker, and every position along it is a move the model was asked
---    to predict. A prompt is therefore a row cut off at the point the
---    model is asked to continue it, and which side is to move falls out
---    of the sequence rather than being handed in beside it.
--- 3. Open a generation session over those token ids and read one logits
---    row.
--- 4. Record whether the raw argmax is a legal move (`raw_legal`), then
---    scan `logits:top(vocab)` in descending order and take the first
---    token that spells a legal move. That is greedy decoding restricted
---    to the legal subset.
---
--- ## The pass
---
--- Othello produces positions with no placement, and they are normal
--- rather than exceptional. `othello6.legal_actions` answers `{ PASS }`
--- for such a position instead of an empty list, because
--- `alc.nn.constraint.allow_list` rejects an empty list when it is
--- built — a mask with nothing in it is a request to draw from nothing.
--- That contract is asserted here as well as promised there: an empty
--- answer fails loudly in `legal_token_ids` naming the rules module,
--- rather than arriving at the constraint as an unexplained construction
--- error one layer down.
---
--- ## The noisy decode
---
--- Greedy decoding answers one position one way forever, which is what
--- the determinism check fences. The noisy path answers the same
--- position by *drawing* from the model, and it stays legal by
--- construction rather than by a scan: a temperature sampler is wrapped
--- in `alc.nn.constraint.allow_list` over the token ids of the moves
--- this position allows, so every other logit is `-inf` before the draw
--- happens.
---
--- The allow list is built per decision, because the legal set of an
--- Othello position is a function of the position: a mask hoisted out of
--- the loop would offer a square that is empty now and occupied two
--- plies later.
---
--- The chain is rebuilt for every decision too. That is the intended
--- shape rather than a cost: `alc.nn.sampler.constrained` consumes both
--- of its arguments (a sampler owns its RNG, and two handles onto one
--- RNG would interleave draws from generations that each believe they
--- are reproducible from their seed), so a cached chain is a spent one.
--- The seed is therefore required rather than defaulted — the caller
--- derives it, and a replay that derives it the same way draws the same
--- move.
---
--- ## Entry contract
---
--- `ctx.task` is a JSON object with a `mode` field:
---
--- - `decide` — `{ state }`; returns
---   `action=<move> legal=true raw_legal=<bool> gated=<bool>`
--- - `decide_noisy` — `{ state, seed, temperature? }`; draws one move
---   under the legal mask of that position and returns
---   `action=<move> legal=true raw_legal=<bool> noisy=true
---   temperature=<t> seed=<n>`. `seed` is required and `temperature`
---   defaults to 1.0. There is no `gated` field: a draw that lands away
---   from the argmax is the sampler doing its job, not a gate stepping
---   in.
--- - `determinism` — `{ state }`; decodes twice through independent
---   sessions and returns `deterministic=<bool> action=<move>`
--- - `selfplay` — `{ games, seed, depth?, style? }`; plays the Card at
---   both seats and scores every move it makes against an
---   `othello6_teacher` policy, returning
---   `winrate=<x.xx> illegal=<n> style_match=<x.xx> style_hits=<n>/<n>`.
---   `depth` and `style` name the teacher every model move is compared
---   against; they default to the pair the Card was trained as, so
---   naming a different pair is how a cross-depth or cross-style
---   comparison is asked for.
---
--- `ctx.depth` and `ctx.style` are the pair the Card was trained as.
--- They do not reach the encoding — an Othello position encodes the same
--- way whatever teacher labelled it — so they exist only as the
--- self-play default. They are validated on every mode all the same: a
--- misspelled style that only failed on the self-play path would let a
--- decode sweep run under a request that was never honoured.
---
--- Every mode also reads an optional `card_alias` from the task JSON;
--- `ctx.card_alias` wins when both carry one. A task field outside the
--- set its mode reads is a loud error rather than a silently dropped
--- one: a misspelled `depth` would otherwise score the model against the
--- default teacher and report the number as though it had been asked
--- for.
---
--- ## What self-play measures, and what it does not
---
--- The Card plays **both seats**, and every move it makes is compared
--- against the move the teacher would have played in that same position.
--- That is the shape of the corpus rather than a choice made here:
--- `othello6.build_corpus` runs one policy at both seats and writes the
--- whole game as one row, so a Card trained on it was asked to imitate
--- the teacher on black and on white alike. A run that let the teacher
--- answer half the positions would score the Card on half of what it was
--- taught, and would score it over games only half of which it steered.
---
--- `style_match` is the measurement: the share of model moves that equal
--- the teacher move for the same position, over positions the model
--- reaches by playing rather than over a fixed list. It is the direct
--- reading of "did this Card reproduce this teacher", and it is the only
--- number in the summary that answers a question about the model.
---
--- `winrate` is the black seat's share, with a draw counted as half a
--- win, and it is **not** a measurement of anything. Both seats are the
--- same Card here, so the number says which colour that Card happens to
--- favour against itself; it is in the summary because the surrounding
--- eval scenarios parse this shape. Strength in this experiment is the
--- teacher's search depth, which is a parameter rather than something to
--- be inferred from a win rate at all.
---
--- Every game opens with a uniform draw of `0..RANDOM_OPENING_MAX`
--- random plies before the model takes over, which is the same opening
--- randomisation `othello6.build_corpus` labels under. Othello has one
--- opening and the greedy decode is deterministic, so without the random
--- prefix a batch of games would be one game repeated.
---
--- ## Caveats
---
--- `alc.llm` is never called, so an eval over this strategy runs to
--- completion without a host round trip. The cost is that the package
--- only works in a build with the `nn` feature: without `alc.nn` the
--- entry fails loudly at Card load rather than degrading to a
--- hand-written policy, which would silently score the wrong thing.
---
--- The opening decodes like any other position. It has no move behind
--- it, so the line `othello6.encode` writes for it is `othello6.BOS`
--- alone — one token, which is what `generate_session` needs to open a
--- session at all: it refuses a prompt with nothing to forward with
--- `alc.nn generate_session: prompt_tokens is empty; provide at least
--- one token id to forward` [measured: `bridge/nn_gen.rs`
--- `GenSession::validate_prompt`, pinned by
--- `alc_nn_generate_session_empty_prompt_errors` in
--- `tests/nn_bridge_smoke.rs`]. The corpus opens its rows with the same
--- marker, so the position a first move is played from is one the model
--- was trained on rather than one it meets for the first time here.
---
--- A decoded position is handed to `othello6` as it arrives. Nothing is
--- defaulted, because every field of a position moves the answer — a
--- substituted `turn` would flip which discs the encoding calls `x`, and
--- a substituted `passes` would move the ending — so the rules module
--- rejects the position naming the field instead.

local othello = require("othello6")
local teacher = require("othello6_teacher")

-- `alc_shapes` is optional, mirroring `othello6`: the package has to
-- stay loadable in a bare Lua VM with no package registry.
local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "othello6_npc",
    version = "0.1.0",
    description = "6x6 Othello NPC driven by a tuned tiny SLM with a legal-move decode gate",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `othello6`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            task = T.string:describe(
                "JSON object with a mode field: decide / decide_noisy / determinism / "
                    .. "selfplay (decide_noisy takes the seed it draws under, selfplay an "
                    .. "optional depth and style naming the teacher it is scored against)"
            ),
            card_alias = T.string:is_optional():describe(
                "Card alias holding the tuned model, also readable from the task JSON "
                    .. "(default: othello6_npc)"
            ),
            depth = T.number:is_optional():describe(
                "Search depth the Card was trained against, the default self-play teacher "
                    .. "depth (default: 2)"
            ),
            style = T.string:is_optional():describe(
                "Evaluation style the Card was trained against, the default self-play "
                    .. "teacher style (default: corner)"
            ),
        }),
        result = T.string:describe("Flat key=value summary of the requested mode"),
    }
end

---@type AlcSpec
M.spec = { entries = { run = run_entry } }

M.docs = {
    schema_version = 1,
}

--- Default Card alias written by the training script for the teacher.
local DEFAULT_ALIAS = "othello6_npc"

--- Teacher pair the NPC is scored against when the caller names none.
--- It matches the pair behind the bare `DEFAULT_ALIAS`, which is the
--- depth and style the first bake of this experiment runs.
local DEFAULT_DEPTH = 2
local DEFAULT_STYLE = "corner"

--- Games a self-play run plays when the caller names no count.
local DEFAULT_GAMES = 20

--- Stride between the self-play seed and the per-game RNG seed, so two
--- games of the same batch never share a stream. The value is
--- `othello6.build_corpus`'s, so a self-play run and a corpus built from
--- the same seed walk the same openings.
local RNG_STRIDE = 7919

--- Loaded handles keyed by alias, per VM.
---
--- A generation session is cheap; reloading the safetensors bundle for
--- every decision is not, and a self-play run makes one decision per
--- turn per game.
local handle_cache = {}

local VOCAB = othello.vocab()

-- ─── Host surface guards ────────────────────────────────────────────

local function require_nn()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("othello6_npc: alc.nn.card is unavailable; build algocline with --features nn")
    end
    return alc.nn
end

--- The two namespaces the noisy path draws through.
---
--- Checked separately from `require_nn` rather than folded into it: a
--- build old enough to have `alc.nn.card` but not
--- `alc.nn.constraint.allow_list` can still answer every greedy mode, so
--- the greedy caller must not be turned away by a surface only the draw
--- needs.
local function require_sampler()
    local nn = require_nn()
    if
        type(nn.sampler) ~= "table"
        or type(nn.sampler.temperature) ~= "function"
        or type(nn.sampler.constrained) ~= "function"
        or type(nn.constraint) ~= "table"
        or type(nn.constraint.allow_list) ~= "function"
    then
        error(
            "othello6_npc: alc.nn.sampler.temperature / .constrained and "
                .. "alc.nn.constraint.allow_list are required for a noisy decode"
        )
    end
    return nn
end

local function require_math()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("othello6_npc: alc.math is required for self-play (alc.math.rng_create)")
    end
    return alc.math
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
        error("othello6_npc: alc.card.get_by_alias is unavailable")
    end
    local card = alc.card.get_by_alias(alias)
    if not card then
        error(
            string.format(
                "othello6_npc: no Card bound to alias %q; bake one with the 6x6 Othello "
                    .. "training driver first",
                alias
            )
        )
    end
    local card_id = card.card_id
    if type(card_id) ~= "string" or #card_id == 0 then
        error(string.format("othello6_npc: alias %q resolved to a Card without card_id", alias))
    end
    local handle = nn.card.load_handle(card_id)
    handle_cache[alias] = handle
    return handle
end

-- ─── Teacher pair ───────────────────────────────────────────────────

--- Check a style name against `othello6.STYLES`.
---
--- The name is checked against the canonical list rather than against
--- the presence of an evaluator, so a typo is rejected with the valid
--- names spelled out instead of resolving to nothing.
---@param raw any Requested style name
---@param field string Name of the field it came from, for the message
---@return string style
local function require_style(raw, field)
    if type(raw) ~= "string" then
        error(string.format("othello6_npc: %s must be a string, got %s", field, type(raw)))
    end
    for _, name in ipairs(othello.STYLES) do
        if name == raw then
            return raw
        end
    end
    error(
        string.format(
            "othello6_npc: unknown %s %q (valid: %s)",
            field,
            raw,
            table.concat(othello.STYLES, ", ")
        )
    )
end

--- Check a requested search depth.
---
--- The range beyond "a positive integer" belongs to
--- `othello6_teacher.search`, which is what actually walks the tree; the
--- check here is the one that lets a bad field name the request it came
--- from rather than the module it reached.
---@param raw any Requested depth
---@param field string Name of the field it came from, for the message
---@return integer depth
local function require_depth(raw, field)
    if type(raw) ~= "number" or raw ~= math.floor(raw) or raw < 1 then
        error(
            string.format(
                "othello6_npc: %s must be a positive integer, got %s",
                field,
                tostring(raw)
            )
        )
    end
    return raw
end

-- ─── Decode ─────────────────────────────────────────────────────────

--- Token ids that spell a legal move for this position.
---
--- The empty check is the assertion of the pass contract described in
--- the module header: `othello6.legal_actions` answers `{ PASS }` for a
--- position with no placement, and a future change that started
--- answering `{}` has to fail here — naming the rules module — rather
--- than inside `alc.nn.constraint.allow_list`, which would report an
--- empty mask for what is a rules problem.
local function legal_token_ids(state)
    local ids, moves = {}, {}
    for _, action in ipairs(othello.legal_actions(state)) do
        local id = VOCAB.to_id[action]
        if id == nil then
            error(string.format("othello6_npc: move %s has no token id", tostring(action)))
        end
        ids[id] = action
        moves[#moves + 1] = action
    end
    if #moves == 0 then
        error(
            "othello6_npc: othello6.legal_actions answered an empty set; a position with no "
                .. "placement must answer the pass so the allow list is never empty"
        )
    end
    return ids, moves
end

--- One gated greedy decision.
---
--- Returns the chosen move plus the two telemetry flags the eval
--- scenario fences on: whether the ungated argmax was already legal, and
--- whether the gate had to move away from it.
---@param handle userdata NnHandle
---@param state table Position
---@return table decision `{ action, raw_legal, gated }`
local function decide(handle, state)
    local legal_ids = legal_token_ids(state)
    local prompt = othello.encode(state)
    local session = handle:generate_session(othello.to_ids(prompt))
    local logits = session:next_logits()

    local raw = logits:argmax()
    local raw_legal = legal_ids[raw] ~= nil

    local ranked = logits:top(logits:vocab())
    for _, entry in ipairs(ranked) do
        local action = legal_ids[entry.id]
        if action ~= nil then
            return { action = action, raw_legal = raw_legal, gated = entry.id ~= raw }
        end
    end
    -- Unreachable: `top(vocab)` enumerates the whole vocabulary and the
    -- legal set is non-empty, so a legal id is always present. Kept as a
    -- loud failure rather than a silent fallback in case a future
    -- ranking change starts truncating.
    error("othello6_npc: no legal token found in the full logit ranking")
end

--- The legal token ids of one position as a list, in
--- `othello6.legal_actions` order, which is the shape
--- `alc.nn.constraint.allow_list` takes.
---
--- A second reading of the same set rather than a change to
--- `legal_token_ids`: the greedy gate wants the id-keyed map (it walks a
--- ranking and asks "is this one legal"), the mask wants the sequence,
--- and the two callers are answered without either paying for the
--- other's shape. The list is built off the `moves` the map function
--- already returned, so both come from one `legal_actions` call and
--- cannot drift; the empty-position and unknown-token failures are that
--- function's, and they fire before any sampler surface is touched.
---@param state table Position
---@return integer[] ids Legal token ids, in legal-move order
---@return table by_id Legal token id -> move, the greedy gate's map
local function legal_token_id_list(state)
    local by_id, moves = legal_token_ids(state)
    local ids = {}
    for _, action in ipairs(moves) do
        ids[#ids + 1] = VOCAB.to_id[action]
    end
    return ids, by_id
end

--- Temperature a noisy decode draws at when the caller names none.
local DEFAULT_TEMPERATURE = 1.0

--- Check a requested temperature.
---
--- Zero is rejected rather than folded into greedy decoding: a caller
--- who means greedy has a mode for it, and a division by zero inside the
--- sampler is not the way to find out that they did not.
local function decode_temperature(raw)
    if raw == nil then
        return DEFAULT_TEMPERATURE
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error(
            string.format(
                "othello6_npc: task.temperature must be a finite positive number, got %s",
                tostring(raw)
            )
        )
    end
    return raw
end

--- Check a requested sampler seed.
---
--- Required, never defaulted. A default would make the draw of a given
--- position depend on nothing the caller can write down, which is
--- exactly the reproducibility the sampler carries its own RNG for: the
--- caller derives the seed (a turn number, a run seed plus an index) and
--- a replay that derives it the same way draws the same move.
local function require_seed(raw, field)
    if raw == nil then
        error(
            string.format(
                "othello6_npc: %s is required for a noisy decode; derive it from the "
                    .. "caller's own counter so the draw can be replayed",
                field
            )
        )
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw < 0 then
        error(
            string.format(
                "othello6_npc: %s must be a non-negative finite number, got %s",
                field,
                tostring(raw)
            )
        )
    end
    return math.floor(raw)
end

--- One noisy decision, legal by construction.
---
--- The mask is applied inside the sampler rather than after it: every
--- logit outside the position's legal ids is `-inf` before the draw, so
--- an illegal move is not rejected and redrawn, it is not representable.
--- The chain is built here, per decision, because
--- `alc.nn.sampler.constrained` moves both of its arguments — a chain
--- held across decisions would be a spent handle on the second one.
---@param handle userdata NnHandle
---@param state table Position
---@param temperature number Draw temperature, already validated
---@param seed integer Sampler seed, already validated
---@return table decision `{ action, raw_legal }`
local function decide_noisy(handle, state, temperature, seed)
    local allow_ids, legal_ids = legal_token_id_list(state)
    local nn = require_sampler()
    local prompt = othello.encode(state)
    local session = handle:generate_session(othello.to_ids(prompt))
    local logits = session:next_logits()

    local raw_legal = legal_ids[logits:argmax()] ~= nil

    local sampler = nn.sampler.constrained(
        nn.sampler.temperature(temperature, seed),
        nn.constraint.allow_list(allow_ids)
    )
    local id = sampler:sample(logits)
    local action = legal_ids[id]
    if action == nil then
        -- Unreachable while the mask holds: the allow list is the legal
        -- set of this very position. Kept loud so a mask that stopped
        -- binding surfaces here rather than as an illegal move reaching
        -- `othello6.apply`.
        error(
            string.format(
                "othello6_npc: the constrained sampler drew token %s, which is not a legal move",
                tostring(id)
            )
        )
    end
    return { action = action, raw_legal = raw_legal }
end

-- ─── Modes ──────────────────────────────────────────────────────────

--- The position a decode request carries.
---
--- Nothing is defaulted here on purpose: `othello6` validates every
--- field the rules and the encoding read and names the one that is
--- missing, whereas a default would answer a question the caller did not
--- ask — a substituted `turn` flips which discs the encoding calls `x`,
--- and a substituted `passes` moves the ending.
local function decode_state(req)
    local state = req.state
    if type(state) ~= "table" then
        error("othello6_npc: task.state must be an object, got " .. type(state))
    end
    return state
end

local function mode_decide(handle, req)
    local d = decide(handle, decode_state(req))
    return string.format(
        "action=%s legal=true raw_legal=%s gated=%s",
        d.action,
        tostring(d.raw_legal),
        tostring(d.gated)
    )
end

--- Format a temperature for the summary.
---
--- `%g` rather than a fixed number of decimals: the field is an echo of
--- what the caller asked for, and rounding an echo makes a sweep report
--- a temperature nobody ran.
local function format_temperature(t)
    return string.format("%g", t)
end

local function mode_decide_noisy(handle, req)
    local state = decode_state(req)
    local temperature = decode_temperature(req.temperature)
    local seed = require_seed(req.seed, "task.seed")
    local d = decide_noisy(handle, state, temperature, seed)
    return string.format(
        "action=%s legal=true raw_legal=%s noisy=true temperature=%s seed=%d",
        d.action,
        tostring(d.raw_legal),
        format_temperature(temperature),
        seed
    )
end

local function mode_determinism(handle, req)
    local state = decode_state(req)
    local first = decide(handle, state)
    local second = decide(handle, state)
    local same = first.action == second.action
    return string.format("deterministic=%s action=%s", tostring(same), first.action)
end

--- Resolve the teacher one self-play run is scored against.
---
--- The request overrides the pair the Card was trained as, one field at
--- a time, because that is what a cross-depth or cross-style comparison
--- is: the same Card asked against a teacher it was not labelled by.
---@param req table Decoded task
---@param basis table `{ depth, style }` the Card was trained as
---@return fun(state: table): string policy
---@return integer depth
---@return string style
local function resolve_teacher(req, basis)
    local depth = basis.depth
    if req.depth ~= nil then
        depth = require_depth(req.depth, "task.depth")
    end
    local style = basis.style
    if req.style ~= nil then
        style = require_style(req.style, "task.style")
    end
    return teacher.policy(depth, style), depth, style
end

--- Play `games` games of the Card against itself, scoring every move
--- against the teacher.
---
--- The loop is `othello6.build_corpus`'s with the Card in the teacher's
--- place: a uniform random opening prefix, then one policy at both seats
--- to the end of the game. Every decision the Card makes is compared
--- against the move the teacher would have played in that same position,
--- which is what `style_match` counts; see the module header for why the
--- win rate that rides along measures nothing.
local function mode_selfplay(handle, req, basis)
    local games = math.floor(tonumber(req.games) or DEFAULT_GAMES)
    local seed = math.floor(tonumber(req.seed) or 1)
    if games <= 0 then
        error("othello6_npc: task.games must be a positive integer")
    end
    local policy = resolve_teacher(req, basis)
    local math_ns = require_math()

    local score, illegal = 0.0, 0
    local moves, hits = 0, 0
    for i = 1, games do
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(seed + i)
        local opening = math_ns.rng_int(rng, 0, othello.RANDOM_OPENING_MAX)
        for _ = 1, opening do
            if othello.is_over(state) then
                break
            end
            state = othello.apply(state, random_move(state))
        end
        while not othello.is_over(state) do
            local d = decide(handle, state)
            if not d.raw_legal then
                illegal = illegal + 1
            end
            moves = moves + 1
            if d.action == policy(state) then
                hits = hits + 1
            end
            state = othello.apply(state, d.action)
        end
        local winner = othello.winner(state)
        if winner == "black" then
            score = score + 1.0
        elseif winner == "draw" then
            score = score + 0.5
        end
    end
    if moves == 0 then
        -- Unreachable while a fresh game leaves at least one turn to
        -- play. Kept loud so a rules change that empties the loop cannot
        -- report style_match as a division by zero or as a silent 0.00.
        error("othello6_npc: self-play made no move")
    end
    return string.format(
        "winrate=%.2f illegal=%d style_match=%.2f style_hits=%d/%d",
        score / games,
        illegal,
        hits / moves,
        hits,
        moves
    )
end

-- ─── Strategy entry ─────────────────────────────────────────────────

--- Task fields every mode reads, whatever it does with them.
local COMMON_FIELDS = { mode = true, card_alias = true }

--- The modes, each with the fields it reads beyond the common ones.
---
--- The field sets are what makes an unknown field detectable: they are
--- declared next to the handler that reads them, so a mode cannot grow a
--- field without saying so here.
local MODES = {
    decide = { fields = { state = true }, run = mode_decide },
    decide_noisy = {
        fields = { state = true, temperature = true, seed = true },
        run = mode_decide_noisy,
    },
    determinism = { fields = { state = true }, run = mode_determinism },
    selfplay = {
        fields = { games = true, seed = true, depth = true, style = true },
        run = mode_selfplay,
    },
}

--- Sorted names, for the diagnostics below.
local function sorted_keys(t)
    local names = {}
    for name in pairs(t) do
        names[#names + 1] = tostring(name)
    end
    table.sort(names)
    return names
end

local MODE_NAMES = table.concat(sorted_keys(MODES), " / ")

--- Reject a task field the requested mode does not read.
---
--- A typo in a request field is otherwise invisible: the run reports a
--- number for the default teacher, the default alias or the default game
--- count, and nothing says the request was not the one honoured.
local function require_known_fields(req, mode, fields)
    local unknown = {}
    for key in pairs(req) do
        if not COMMON_FIELDS[key] and not fields[key] then
            unknown[#unknown + 1] = tostring(key)
        end
    end
    if #unknown == 0 then
        return
    end
    table.sort(unknown)
    local known = {}
    for _, name in ipairs(sorted_keys(COMMON_FIELDS)) do
        known[#known + 1] = name
    end
    for _, name in ipairs(sorted_keys(fields)) do
        known[#known + 1] = name
    end
    error(
        string.format(
            "othello6_npc: task field(s) %s are not read in %s mode (known: %s)",
            table.concat(unknown, ", "),
            mode,
            table.concat(known, ", ")
        )
    )
end

--- Alias the decision is decoded with.
---
--- `ctx.card_alias` wins over the task field because the ctx is where
--- the eval runner merges `strategy_opts` and where an embedding caller
--- binds the Card it means. The task field is read as the fallback
--- rather than dropped: a caller whose only channel is the request JSON
--- would otherwise be answered by the default Card without a word.
local function resolve_alias(ctx, req)
    local alias = ctx.card_alias
    if alias == nil then
        alias = req.card_alias
    end
    if alias == nil then
        return DEFAULT_ALIAS
    end
    if type(alias) ~= "string" then
        error("othello6_npc: card_alias must be a string, got " .. type(alias))
    end
    if #alias == 0 then
        error("othello6_npc: card_alias must not be empty")
    end
    return alias
end

--- The teacher pair the Card was trained as.
---
--- Validated for every mode rather than only for self-play: the pair is
--- a property of the Card, so a misspelled one is a misdescribed Card
--- whichever question is being asked of it, and a decode sweep that only
--- failed once it reached self-play would have already reported numbers
--- under a request nobody made.
local function resolve_basis(ctx)
    local basis = { depth = DEFAULT_DEPTH, style = DEFAULT_STYLE }
    if ctx.depth ~= nil then
        basis.depth = require_depth(ctx.depth, "ctx.depth")
    end
    if ctx.style ~= nil then
        basis.style = require_style(ctx.style, "ctx.style")
    end
    return basis
end

---@param ctx table `{ task, card_alias?, depth?, style? }`
---@return table result `{ result = <flat key=value summary> }`
function M.run(ctx)
    ctx = ctx or {}
    local task = ctx.task
    if type(task) ~= "string" then
        error("othello6_npc: ctx.task must be a JSON string, got " .. type(task))
    end
    local req = alc.json_decode(task)
    if type(req) ~= "table" or type(req.mode) ~= "string" then
        error("othello6_npc: ctx.task must decode to an object with a mode field")
    end
    local mode = MODES[req.mode]
    if mode == nil then
        error(string.format("othello6_npc: unknown mode %q (expected %s)", req.mode, MODE_NAMES))
    end
    require_known_fields(req, req.mode, mode.fields)

    local basis = resolve_basis(ctx)
    local handle = resolve_handle(resolve_alias(ctx, req))

    return { result = mode.run(handle, req, basis) }
end

--- Drop cached handles. Exposed for the training script, which binds a
--- new Card to the alias inside a VM that may already hold the old one.
function M.reset_cache()
    handle_cache = {}
end

return M
