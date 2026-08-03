--- guardian_duel_npc — SLM boss NPC with a legal-action decode gate
---
--- Wraps a tuned tiny SLM (a Card produced by
--- `examples/gameai/train_guardian_npc.lua`) into an algocline strategy
--- that answers one boss move per turn of a `guardian_duel` fight. The
--- model never touches the fight: `guardian_duel.legal_actions`
--- enumerates the moves the state allows and the gate walks the logit
--- ranking until it hits one of them, so a twin slam without spikes is
--- structurally impossible while the raw argmax is still reported as
--- telemetry.
---
--- ## Usage
---
--- ```lua
--- local npc = require("guardian_duel_npc")
--- return npc.run({
---     task = alc.json_encode({ mode = "decide", state = boss_state }),
---     card_alias = "guardian_duel_npc_guardian",
---     style = "guardian",
--- })
--- ```
---
--- ## Algorithm
---
--- 1. Resolve the Card behind the alias and load an `NnHandle` (cached
---    per VM, since a decode session is cheap but a model load is not).
--- 2. Encode the boss state with `guardian_duel.encode` against the
---    style this NPC answers as, and append `">"`, the separator the
---    training lines use.
--- 3. Open a generation session over those token ids and read one
---    logits row.
--- 4. Record whether the raw argmax is a legal move (`raw_legal`), then
---    scan `logits:top(vocab)` in descending order and take the first
---    token that spells a legal move. That is greedy decoding
---    restricted to the legal subset.
---
--- ## The noisy decode
---
--- Greedy decoding answers one position one way forever, which is what
--- the determinism check fences and what makes a batch of fights a
--- batch of copies. The noisy path answers the same position by
--- *drawing* from the model, and it stays legal by construction rather
--- than by a scan: a temperature sampler is wrapped in
--- `alc.nn.constraint.allow_list` over the token ids of the moves this
--- state allows, so every other logit is `-inf` before the draw
--- happens.
---
--- The allow list is built per decision here, unlike the player seat
--- where the four moves are a module constant: the boss legal set is a
--- function of the state (the twin slam needs mode 1), so a mask hoisted
--- out of the loop would offer the slam on a state that forbids it.
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
--- `raw_legal` is computed on the noisy path exactly as on the greedy
--- one, off the ungated argmax, and on this seat it carries more than
--- the player one does: the legal set moves with the state, so an
--- argmax that was legal on a rolled-up boss is illegal on the same
--- model's next position. A `raw_legal` that sags is therefore either a
--- model that stopped answering the question or one that keeps reaching
--- for the slam it may not play — both are the gate doing the work the
--- draw cannot report on its own.
---
--- The seed is the caller's to derive. Three conventions live in this
--- repo and they are all per-decision seeds handed to one chain, which
--- is the semantics that has to agree; how the number is reached is the
--- caller's layer:
---
--- - `gameai_metrics.level` — `base + k` for a run-local counter `k`
---   incremented once per draw, so a whole pool replays from one seed.
--- - `guardian_player_npc` autoplay — `seed + game * (TURN_LIMIT + 1)
---   + turn`, a stride one wider than the longest fight, so a single
---   turn of a single game replays without walking the fights before it.
--- - `guardian_duel_npc` self-play — `seed * RNG_STRIDE + i` for the
---   *player* RNG of fight `i` (`alc.math.rng_create`), the same
---   one-stream-per-fight rule applied to a scripted opponent.
---
--- ## Entry contract
---
--- `ctx.task` is a JSON object with a `mode` field:
---
--- - `decide` — `{ state }`; returns
---   `action=<move> legal=true raw_legal=<bool> gated=<bool>`
--- - `decide_noisy` — `{ state, seed, temperature? }`; draws one move
---   under the legal mask of that state and returns
---   `action=<move> legal=true raw_legal=<bool> noisy=true
---   temperature=<t> seed=<n>`. `seed` is required and `temperature`
---   defaults to 1.0. There is no `gated` field: a draw that lands away
---   from the argmax is the sampler doing its job, not a gate stepping
---   in.
--- - `determinism` — `{ state }`; decodes twice through independent
---   sessions and returns `deterministic=<bool> action=<move>`
--- - `selfplay` — `{ games, seed, style?, policy_source? }`; plays the
---   NPC boss against `guardian_duel.policy_player_random` and returns
---   `winrate=<x.xx> illegal=<n> style_match=<x.xx> style_hits=<n>/<n>`.
---   `style` names the teacher policy every model move is compared
---   against (default: the style the NPC answers as); `policy_source`
---   overrides it with a synthesised policy chunk, which is what a
---   persona Card — a Card with no entry in `guardian_duel.STYLES` — is
---   scored against. `policy_source` may also be passed on the strategy
---   ctx, which is where the eval runner merges `strategy_opts`.
---
--- `ctx.style` is the style the Card was trained as. It decides what
--- the `D` field of every encoded state measures, because
--- `guardian_duel.encode` writes the distance left to *that* style's
--- mode shift; handing the model a distance its corpus was not labelled
--- against would ask it about states it never saw. It is also the
--- teacher self-play falls back to. A persona Card is trained under a
--- declared basis style (`bake_guardian_persona.lua`), so it is
--- answered for with that basis and scored against its own chunk.
---
--- Every mode also reads an optional `card_alias` from the task JSON;
--- `ctx.card_alias` wins when both carry one. A task field outside the
--- set its mode reads is a loud error rather than a silently dropped
--- one: a misspelled `style` would otherwise score the model against
--- the default teacher and report the number as though it had been
--- asked for.
---
--- ## Caveats
---
--- `alc.llm` is never called, so an eval over this strategy runs to
--- completion without a host round trip. The cost is that the package
--- only works in a build with the `nn` feature: without `alc.nn` the
--- entry fails loudly at Card load rather than degrading to a
--- hand-written policy, which would silently score the wrong thing.
---
--- A decoded state is handed to `guardian_duel` as it arrives. Nothing
--- is defaulted, because every field of a boss state moves the answer —
--- a missing `shifts` would silently move the distance basis, and a
--- missing `mode` would silently offer the twin slam — so the rules
--- module rejects the state naming the field instead.
---
--- The self-play win rate is reported from the boss seat, with a draw
--- counted as half a win, but it is not what self-play is for: it
--- depends on the random player's swings as much as on the model.
--- `style_match` is the direct measurement — the share of model moves
--- that equal the teacher move for the same state, over states the
--- model reaches by playing rather than over a fixed list. A model that
--- learned its style scores near 1.00 whatever the win rate says.

local duel = require("guardian_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "guardian_duel_npc",
    version = "0.1.0",
    description = "Guardian duel boss NPC driven by a tuned tiny SLM with a legal-action decode gate",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `guardian_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            task = T.string:describe(
                "JSON object with a mode field: decide / decide_noisy / determinism / "
                    .. "selfplay (decide_noisy takes the seed it draws under, selfplay "
                    .. "an optional style naming the teacher policy)"
            ),
            card_alias = T.string:is_optional():describe(
                "Card alias holding the tuned model, also readable from the task JSON "
                    .. "(default: guardian_duel_npc)"
            ),
            style = T.string:is_optional():describe(
                "Boss style the Card was trained as: the distance basis every state is "
                    .. "encoded against, and the default self-play teacher (default: guardian)"
            ),
            policy_source = T.string:is_optional():describe(
                "Synthesised policy chunk self-play scores the model against, "
                    .. "used for persona Cards that have no entry in guardian_duel.STYLES"
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
local DEFAULT_ALIAS = "guardian_duel_npc"

--- Style the NPC answers as when the caller names none. It matches the
--- style behind the bare `DEFAULT_ALIAS`.
local DEFAULT_STYLE = "guardian"

--- Stride between the self-play seed and the player RNG seed, so two
--- fights of the same batch never share a stream.
local RNG_STRIDE = 1000

--- Loaded handles keyed by alias, per VM.
---
--- A generation session is cheap; reloading the safetensors bundle for
--- every decision is not, and a self-play run makes one decision per
--- turn per fight.
local handle_cache = {}

local VOCAB = duel.vocab()

-- ─── Host surface guards ────────────────────────────────────────────

local function require_nn()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("guardian_duel_npc: alc.nn.card is unavailable; build algocline with --features nn")
    end
    return alc.nn
end

--- The two namespaces the noisy path draws through.
---
--- Checked separately from `require_nn` rather than folded into it: a
--- build old enough to have `alc.nn.card` but not
--- `alc.nn.constraint.allow_list` can still answer every greedy mode,
--- so the greedy caller must not be turned away by a surface only the
--- draw needs.
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
            "guardian_duel_npc: alc.nn.sampler.temperature / .constrained and "
                .. "alc.nn.constraint.allow_list are required for a noisy decode"
        )
    end
    return nn
end

local function require_math()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("guardian_duel_npc: alc.math is required for self-play (alc.math.rng_create)")
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
        error("guardian_duel_npc: alc.card.get_by_alias is unavailable")
    end
    local card = alc.card.get_by_alias(alias)
    if not card then
        error(
            string.format(
                "guardian_duel_npc: no Card bound to alias %q; run examples/gameai/train_guardian_npc.lua first",
                alias
            )
        )
    end
    local card_id = card.card_id
    if type(card_id) ~= "string" or #card_id == 0 then
        error(
            string.format("guardian_duel_npc: alias %q resolved to a Card without card_id", alias)
        )
    end
    local handle = nn.card.load_handle(card_id)
    handle_cache[alias] = handle
    return handle
end

-- ─── Styles ─────────────────────────────────────────────────────────

--- Check a style name against `guardian_duel.STYLES`.
---
--- The name is checked against the canonical list rather than against
--- the presence of a `policy_<name>` field, so a typo is rejected with
--- the valid names spelled out instead of resolving to nothing.
---@param raw any Requested style name
---@param field string Name of the field it came from, for the message
---@return string style
local function require_style(raw, field)
    if type(raw) ~= "string" then
        error(string.format("guardian_duel_npc: %s must be a string, got %s", field, type(raw)))
    end
    for _, name in ipairs(duel.STYLES) do
        if name == raw then
            return raw
        end
    end
    error(
        string.format(
            "guardian_duel_npc: unknown %s %q (valid: %s)",
            field,
            raw,
            table.concat(duel.STYLES, ", ")
        )
    )
end

-- ─── Decode ─────────────────────────────────────────────────────────

--- Token ids that spell a legal move for this state.
local function legal_token_ids(state)
    local ids, moves = {}, {}
    for _, action in ipairs(duel.legal_actions(state)) do
        local id = VOCAB.to_id[action]
        if id == nil then
            error(string.format("guardian_duel_npc: move %s has no token id", tostring(action)))
        end
        ids[id] = action
        moves[#moves + 1] = action
    end
    if #moves == 0 then
        error("guardian_duel_npc: state has no legal move")
    end
    return ids, moves
end

--- One gated greedy decision.
---
--- Returns the chosen move plus the two telemetry flags the eval
--- scenario fences on: whether the ungated argmax was already legal,
--- and whether the gate had to move away from it.
---@param handle userdata NnHandle
---@param state table Boss state
---@param style string Distance basis the state is encoded against
---@return table decision `{ action, raw_legal, gated }`
local function decide(handle, state, style)
    local legal_ids = legal_token_ids(state)
    local prompt = duel.encode(state, style) .. ">"
    local session = handle:generate_session(duel.to_ids(prompt))
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
    error("guardian_duel_npc: no legal token found in the full logit ranking")
end

--- The legal token ids of one state as a list, in
--- `guardian_duel.legal_actions` order, which is the shape
--- `alc.nn.constraint.allow_list` takes.
---
--- A second reading of the same set rather than a change to
--- `legal_token_ids`: the greedy gate wants the id-keyed map (it walks a
--- ranking and asks "is this one legal"), the mask wants the sequence,
--- and the two callers are answered without either paying for the
--- other's shape. The list is built off the `moves` the map function
--- already returned, so both come from one `legal_actions` call and
--- cannot drift; the empty-state and unknown-token failures are that
--- function's, and they fire before any sampler surface is touched.
---@param state table Boss state
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
--- who means greedy has a mode for it, and a division by zero inside
--- the sampler is not the way to find out that they did not.
local function decode_temperature(raw)
    if raw == nil then
        return DEFAULT_TEMPERATURE
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error(
            string.format(
                "guardian_duel_npc: task.temperature must be a finite positive number, got %s",
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
--- caller derives the seed (a turn number, a run seed plus an index)
--- and a replay that derives it the same way draws the same move.
local function require_seed(raw, field)
    if raw == nil then
        error(
            string.format(
                "guardian_duel_npc: %s is required for a noisy decode; derive it from the "
                    .. "caller's own counter so the draw can be replayed",
                field
            )
        )
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw < 0 then
        error(
            string.format(
                "guardian_duel_npc: %s must be a non-negative finite number, got %s",
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
--- logit outside the state's legal ids is `-inf` before the draw, so an
--- illegal move is not rejected and redrawn, it is not representable.
--- The chain is built here, per decision, because
--- `alc.nn.sampler.constrained` moves both of its arguments — a chain
--- held across decisions would be a spent handle on the second one.
---
--- This is the same chain `gameai_metrics.level` builds for the boss
--- seat (`boss_seat.legal` ids, `boss_seat.encode` prompt, one
--- `constrained(temperature(t, seed), allow_list(ids))` per decision).
--- The two are separate implementations on purpose — the NPC's public
--- API is `run` / `reset_cache`, so the measurement side cannot require
--- this one (see `gameai_metrics/boss_seat.lua`) — and
--- `spec/guardian_duel_noisy_equivalence_spec.lua` pins them together.
---@param handle userdata NnHandle
---@param state table Boss state
---@param style string Distance basis the state is encoded against
---@param temperature number Draw temperature, already validated
---@param seed integer Sampler seed, already validated
---@return table decision `{ action, raw_legal }`
local function decide_noisy(handle, state, style, temperature, seed)
    local allow_ids, legal_ids = legal_token_id_list(state)
    local nn = require_sampler()
    local prompt = duel.encode(state, style) .. ">"
    local session = handle:generate_session(duel.to_ids(prompt))
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
        -- set of this very state. Kept loud so a mask that stopped
        -- binding surfaces here rather than as an illegal move reaching
        -- `guardian_duel.apply`.
        error(
            string.format(
                "guardian_duel_npc: the constrained sampler drew token %s, "
                    .. "which is not a legal boss move",
                tostring(id)
            )
        )
    end
    return { action = action, raw_legal = raw_legal }
end

-- ─── Modes ──────────────────────────────────────────────────────────

--- The boss state a decode request carries.
---
--- Nothing is defaulted here on purpose: `guardian_duel` validates every
--- field the encoding reads and names the one that is missing, whereas a
--- default would answer a question the caller did not ask — a
--- substituted `shifts` moves the distance the model reads, and a
--- substituted `mode` changes which moves are legal.
local function decode_state(req)
    local state = req.state
    if type(state) ~= "table" then
        error("guardian_duel_npc: task.state must be an object, got " .. type(state))
    end
    return state
end

local function mode_decide(handle, req, style)
    local d = decide(handle, decode_state(req), style)
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

local function mode_decide_noisy(handle, req, style)
    local state = decode_state(req)
    local temperature = decode_temperature(req.temperature)
    local seed = require_seed(req.seed, "task.seed")
    local d = decide_noisy(handle, state, style, temperature, seed)
    return string.format(
        "action=%s legal=true raw_legal=%s noisy=true temperature=%s seed=%d",
        d.action,
        tostring(d.raw_legal),
        format_temperature(temperature),
        seed
    )
end

local function mode_determinism(handle, req, style)
    local state = decode_state(req)
    local first = decide(handle, state, style)
    local second = decide(handle, state, style)
    local same = first.action == second.action
    return string.format("deterministic=%s action=%s", tostring(same), first.action)
end

--- Resolve the teacher for one self-play run.
---
--- `policy_source` takes precedence over `style`: a persona Card is
--- trained on an LLM-written policy that has no entry in
--- `guardian_duel.STYLES`, so the style whitelist would reject it by
--- construction. The chunk is never loaded raw — `compile_policy` runs
--- it in the restricted environment and makes it answer sampled states
--- legally and deterministically before a single move is scored against
--- it, which is the same gate the bake script applies.
---@param req table Decoded task
---@param style string Style the NPC answers as, the teacher default
---@param seed integer Self-play seed, reused for the validation states
---@return fun(state: table): string policy
local function resolve_teacher(req, style, seed)
    local source = req.policy_source
    if source == nil then
        local name = style
        if req.style ~= nil then
            name = require_style(req.style, "task.style")
        end
        return duel["policy_" .. name]
    end
    if type(source) ~= "string" then
        error("guardian_duel_npc: task.policy_source must be a string, got " .. type(source))
    end
    return duel.compile_policy(source, { seed = seed, chunk_name = "persona_policy" })
end

local function mode_selfplay(handle, req, style)
    local games = math.floor(tonumber(req.games) or 20)
    local seed = math.floor(tonumber(req.seed) or 1)
    if games <= 0 then
        error("guardian_duel_npc: task.games must be a positive integer")
    end
    local policy = resolve_teacher(req, style, seed)
    local math_ns = require_math()

    local score, illegal = 0.0, 0
    local moves, hits = 0, 0
    for i = 1, games do
        local g = duel.new_game(seed + i)
        local rng = math_ns.rng_create(seed * RNG_STRIDE + i)
        while not duel.is_over(g) do
            local d = decide(handle, g.boss, style)
            if not d.raw_legal then
                illegal = illegal + 1
            end
            moves = moves + 1
            if d.action == policy(g.boss) then
                hits = hits + 1
            end
            g = duel.apply(g, duel.policy_player_random(rng), d.action)
        end
        local w = duel.winner(g)
        if w == "boss" then
            score = score + 1.0
        elseif w == "draw" then
            score = score + 0.5
        end
    end
    if moves == 0 then
        -- Unreachable while a fresh fight runs to the turn limit. Kept
        -- loud so a rules change that empties the loop cannot report
        -- style_match as a division by zero or as a silent 0.00.
        error("guardian_duel_npc: self-play made no move")
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
--- declared next to the handler that reads them, so a mode cannot grow
--- a field without saying so here.
local MODES = {
    decide = { fields = { state = true }, run = mode_decide },
    decide_noisy = {
        fields = { state = true, temperature = true, seed = true },
        run = mode_decide_noisy,
    },
    determinism = { fields = { state = true }, run = mode_determinism },
    selfplay = {
        fields = { games = true, seed = true, style = true, policy_source = true },
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
--- number for the default teacher, the default alias or the default
--- game count, and nothing says the request was not the one honoured.
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
            "guardian_duel_npc: task field(s) %s are not read in %s mode (known: %s)",
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
        error("guardian_duel_npc: card_alias must be a string, got " .. type(alias))
    end
    if #alias == 0 then
        error("guardian_duel_npc: card_alias must not be empty")
    end
    return alias
end

---@param ctx table `{ task, card_alias?, style?, policy_source? }`
---@return table result `{ result = <flat key=value summary> }`
function M.run(ctx)
    ctx = ctx or {}
    local task = ctx.task
    if type(task) ~= "string" then
        error("guardian_duel_npc: ctx.task must be a JSON string, got " .. type(task))
    end
    local req = alc.json_decode(task)
    if type(req) ~= "table" or type(req.mode) ~= "string" then
        error("guardian_duel_npc: ctx.task must decode to an object with a mode field")
    end
    local mode = MODES[req.mode]
    if mode == nil then
        error(
            string.format("guardian_duel_npc: unknown mode %q (expected %s)", req.mode, MODE_NAMES)
        )
    end
    require_known_fields(req, req.mode, mode.fields)

    -- A synthesised teacher can arrive on the ctx instead of inside the
    -- task JSON, because that is where the eval runner merges
    -- `strategy_opts`. The task form wins when both carry one, and the
    -- merge runs after the field check so a ctx-wide chunk does not turn
    -- the scenario's decide cases into unknown-field errors.
    if req.policy_source == nil and ctx.policy_source ~= nil then
        req.policy_source = ctx.policy_source
    end

    -- The style the NPC answers as is a property of the Card rather than
    -- of the request, so it is read from the ctx and validated once for
    -- every mode: an unknown one would encode a distance against a
    -- threshold that does not exist.
    local style = DEFAULT_STYLE
    if ctx.style ~= nil then
        style = require_style(ctx.style, "ctx.style")
    end

    local handle = resolve_handle(resolve_alias(ctx, req))

    return { result = mode.run(handle, req, style) }
end

--- Drop cached handles. Exposed for the training script, which binds a
--- new Card to the alias inside a VM that may already hold the old one.
function M.reset_cache()
    handle_cache = {}
end

return M
