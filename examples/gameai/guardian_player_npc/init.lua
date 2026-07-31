--- guardian_player_npc — SLM player NPC for the guardian duel
---
--- The other seat of `guardian_duel_npc`. Where that package wraps a
--- Card that learned a boss script, this one wraps a Card that learned
--- how somebody played *against* one — baked from a transcript by
--- `examples/gameai/bake_guardian_player_from_log.lua` — and answers one
--- player move per turn.
---
--- ## Usage
---
--- ```lua
--- local npc = require("guardian_player_npc")
--- return npc.run({
---     task = alc.json_encode({ mode = "decide", view = player_view }),
---     card_alias = "guardian_player_npc_ytk",
--- })
--- ```
---
--- ## Algorithm
---
--- 1. Resolve the Card behind the alias and load an `NnHandle` (cached
---    per VM, since a decode session is cheap but a model load is not).
--- 2. Encode the player view with `guardian_duel.player_encode` and
---    append `">"`, the separator the training lines use.
--- 3. Open a generation session over those token ids and read one
---    logits row.
--- 4. Record whether the raw argmax is a player move (`raw_legal`), then
---    scan `logits:top(vocab)` in descending order and take the first
---    token that spells one. That is greedy decoding restricted to the
---    legal subset.
---
--- The player has all four moves on every turn of every fight, so the
--- gate never removes an option the rules would have allowed. It is
--- still here, and `raw_legal` is still reported, because the model can
--- answer with a digit, a field letter or the padding token — a token
--- that is not a move at all — and a run whose `raw_legal` rate sags is
--- a model that has stopped answering the question rather than one
--- playing badly.
---
--- ## Entry contract
---
--- `ctx.task` is a JSON object with a `mode` field:
---
--- - `decide` — `{ view }`; returns
---   `action=<move> legal=true raw_legal=<bool> gated=<bool>`
--- - `determinism` — `{ view }`; decodes twice through independent
---   sessions and returns `deterministic=<bool> action=<move>`
--- - `autoplay` — `{ games, seed, boss_style?, boss_card_alias? }`;
---   plays the model against a boss and returns
---   `winrate=<x.xx> raw_legal=<x.xx> moves=<n> a=<n> A=<n> b=<n>
---   p=<n>`. The win rate is read from the player seat with a draw
---   counted as half a win, and the four counts are the model's own
---   move distribution.
---
--- Both seats decode greedily and `guardian_duel.new_game` opens the
--- same board for every seed, so a fight is a function of the two
--- Cards alone: `games` runs of one pairing are `games` copies of one
--- fight, and the win rate can only come out as `0.00`, `0.50` or
--- `1.00`. The default is therefore one game, and the field is honoured
--- as asked rather than clamped — the request shape is the boss NPC's
--- and a sampled decode would need it — but a batch is a repeat, not a
--- sample, and `seed` changes nothing on this path.
---
--- `boss_style` is the boss the fights are played against and, with it,
--- the distance basis every generated view is measured against. Those
--- are one field rather than two because they have to agree: the `D`
--- field of a player view is the damage *that* boss still tolerates, so
--- a model logged against the teacher and autoplayed against the
--- impatient variant would be reading a number that means something
--- else. `boss_card_alias` seats a boss Card instead of the teacher
--- policy, decoded through `guardian_duel_npc` under the same basis.
---
--- Every mode also reads an optional `card_alias` from the task JSON;
--- `ctx.card_alias` wins when both carry one. A task field outside the
--- set its mode reads is a loud error rather than a silently dropped
--- one: a misspelled `boss_style` would otherwise play the default boss
--- and report the number as though it had been asked for.
---
--- ## Caveats
---
--- `alc.llm` is never called, so an eval over this strategy runs to
--- completion without a host round trip. The cost is that the package
--- only works in a build with the `nn` feature: without `alc.nn` the
--- entry fails loudly at Card load rather than degrading to a
--- hand-written policy, which would silently score the wrong thing.
---
--- The player alphabet is not the boss alphabet, and the two are
--- different token id spaces (`guardian_duel.player_vocab` versus
--- `guardian_duel.vocab`). A boss Card decoded through this package
--- would answer legal moves out of noise, which is why the aliases live
--- in their own `guardian_player_npc_*` namespace and are never
--- defaulted across.
---
--- There is no `style_match` here and there cannot be one. A boss NPC
--- is measured against the teacher policy that labelled its corpus; a
--- player Card is baked from a log, which has no policy behind it, so
--- the only fit that can be computed is against the log itself and the
--- bake script reports it as `log_match`. What autoplay adds is the
--- other half: how the model does in positions the log never contained.

local duel = require("guardian_duel")

local shapes_ok, S = pcall(require, "alc_shapes")
local T = shapes_ok and S.T or nil

local M = {}

---@type AlcMeta
M.meta = {
    name = "guardian_player_npc",
    version = "0.1.0",
    description = "Guardian duel player NPC driven by a tiny SLM baked from a play log",
    category = "game",
}

-- Runtime contract for `run`. Declared with the shapes DSL when it is
-- available and left empty otherwise, mirroring `guardian_duel`.
local run_entry = {}
if T then
    run_entry = {
        input = T.shape({
            task = T.string:describe(
                "JSON object with a mode field: decide / determinism / autoplay "
                    .. "(autoplay also takes the boss it plays against)"
            ),
            card_alias = T.string:is_optional():describe(
                "Card alias holding the tuned model, also readable from the task JSON "
                    .. "(default: guardian_player_npc)"
            ),
            boss_style = T.string:is_optional():describe(
                "Boss the autoplay fights are played against, and the distance basis "
                    .. "every generated view is measured against (default: guardian)"
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

--- Default Card alias, the one a bake with no name would land on.
local DEFAULT_ALIAS = "guardian_player_npc"

--- Boss the model is autoplayed against when the caller names none, and
--- the distance basis its views are encoded against.
local DEFAULT_BOSS_STYLE = "guardian"

--- Loaded handles keyed by alias, per VM.
local handle_cache = {}

local VOCAB = duel.player_vocab()

--- The four moves, in the order the summary reports them.
local MOVES = duel.player_legal_actions()

-- ─── Host surface guards ────────────────────────────────────────────

local function require_nn()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("guardian_player_npc: alc.nn.card is unavailable; build algocline with --features nn")
    end
    return alc.nn
end

local function require_json()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error("guardian_player_npc: alc.json_encode is required to seat a boss Card")
    end
end

--- Resolve the alias to a loaded model handle.
local function resolve_handle(alias)
    local cached = handle_cache[alias]
    if cached then
        return cached
    end
    local nn = require_nn()
    if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
        error("guardian_player_npc: alc.card.get_by_alias is unavailable")
    end
    local card = alc.card.get_by_alias(alias)
    if not card then
        error(
            string.format(
                "guardian_player_npc: no Card bound to alias %q; "
                    .. "run examples/gameai/bake_guardian_player_from_log.lua first",
                alias
            )
        )
    end
    local card_id = card.card_id
    if type(card_id) ~= "string" or #card_id == 0 then
        error(
            string.format("guardian_player_npc: alias %q resolved to a Card without card_id", alias)
        )
    end
    local handle = nn.card.load_handle(card_id)
    handle_cache[alias] = handle
    return handle
end

-- ─── Styles ─────────────────────────────────────────────────────────

--- Check a boss style against `guardian_duel.STYLES`.
local function require_style(raw, field)
    if type(raw) ~= "string" then
        error(string.format("guardian_player_npc: %s must be a string, got %s", field, type(raw)))
    end
    for _, name in ipairs(duel.STYLES) do
        if name == raw then
            return raw
        end
    end
    error(
        string.format(
            "guardian_player_npc: unknown %s %q (valid: %s)",
            field,
            raw,
            table.concat(duel.STYLES, ", ")
        )
    )
end

-- ─── Decode ─────────────────────────────────────────────────────────

--- Token ids that spell a player move, built once: the four are legal
--- on every turn of every fight, so the set does not depend on a state.
local LEGAL_IDS = {}
for _, action in ipairs(MOVES) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error(string.format("guardian_player_npc: move %s has no token id", tostring(action)))
    end
    LEGAL_IDS[id] = action
end

--- One gated greedy decision.
---
--- Returns the chosen move plus the two telemetry flags: whether the
--- ungated argmax already spelled a move, and whether the gate had to
--- move away from it.
---@param handle userdata NnHandle
---@param view table Player view
---@return table decision `{ action, raw_legal, gated }`
local function decide(handle, view)
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()

    local raw = logits:argmax()
    local raw_legal = LEGAL_IDS[raw] ~= nil

    local ranked = logits:top(logits:vocab())
    for _, entry in ipairs(ranked) do
        local action = LEGAL_IDS[entry.id]
        if action ~= nil then
            return { action = action, raw_legal = raw_legal, gated = entry.id ~= raw }
        end
    end
    -- Unreachable: `top(vocab)` enumerates the whole vocabulary and the
    -- four moves are always in it. Kept as a loud failure rather than a
    -- silent fallback in case a future ranking change starts truncating.
    error("guardian_player_npc: no player move found in the full logit ranking")
end

-- ─── Modes ──────────────────────────────────────────────────────────

--- The player view a decode request carries.
---
--- Nothing is defaulted: `guardian_duel` validates every field the
--- encoding reads and names the one that is missing, whereas a default
--- would answer a question the caller did not ask — a substituted
--- `shift_distance` moves the field the model reads hardest.
local function decode_view(req)
    local view = req.view
    if type(view) ~= "table" then
        error("guardian_player_npc: task.view must be an object, got " .. type(view))
    end
    return view
end

local function mode_decide(handle, req)
    local d = decide(handle, decode_view(req))
    return string.format(
        "action=%s legal=true raw_legal=%s gated=%s",
        d.action,
        tostring(d.raw_legal),
        tostring(d.gated)
    )
end

local function mode_determinism(handle, req)
    local view = decode_view(req)
    local first = decide(handle, view)
    local second = decide(handle, view)
    local same = first.action == second.action
    return string.format("deterministic=%s action=%s", tostring(same), first.action)
end

--- The seven fields of a boss state the boss NPC reads.
---
--- The live state also carries the block and spike counters, which are
--- engine bookkeeping: they never reach that model either, so they are
--- left out of the request rather than shipped across the JSON boundary
--- as though they mattered to the answer.
local function boss_payload(boss)
    return {
        cycle = boss.cycle,
        mode = boss.mode,
        hp = boss.hp,
        damage_since_shift = boss.damage_since_shift,
        last_player = boss.last_player,
        turn = boss.turn,
        shifts = boss.shifts,
    }
end

--- Seat the opponent for one autoplay run.
---
--- With no `boss_card_alias` the boss is the teacher policy of
--- `boss_style`, which makes a run reproducible from its seed alone.
--- With one, the boss is another Card, decoded through
--- `guardian_duel_npc` under the same basis — the shape a demo uses to
--- put a baked player against a baked boss.
---@param req table Decoded task
---@param style string Boss style, the teacher default and the basis
---@return fun(state: table): string boss
local function resolve_boss(req, style)
    local alias = req.boss_card_alias
    if alias == nil then
        return duel["policy_" .. style]
    end
    if type(alias) ~= "string" or #alias == 0 then
        error("guardian_player_npc: task.boss_card_alias must be a non-empty string")
    end
    require_json()
    local boss_npc = require("guardian_duel_npc")
    return function(state)
        local out = boss_npc.run({
            task = alc.json_encode({ mode = "decide", state = boss_payload(state) }),
            card_alias = alias,
            style = style,
        })
        local text = type(out) == "table" and out.result or out
        if type(text) ~= "string" then
            error("guardian_player_npc: boss answer must be a string, got " .. type(text))
        end
        local action = text:match("action=(%a)")
        if action == nil then
            error(string.format("guardian_player_npc: boss answer %q carries no action", text))
        end
        return action
    end
end

--- Play the model against a boss and report how it went.
---
--- The win rate is the model's own, with a draw counted as half a win,
--- and it is the number a player Card is actually judged on: unlike a
--- boss Card there is no teacher move to compare against, so how the
--- fights end is the measurement rather than a side effect of it. The
--- move distribution sits next to it because a model that answers `b`
--- to everything can win the health-bucket comparison while having
--- learned nothing, and the two numbers together say so.
---
--- One fight is the whole measurement for a given pairing: both seats
--- are greedy and the opening is fixed, so a larger `games` repeats it.
--- The loop still runs what it was asked for rather than collapsing the
--- batch silently, because the caller may be timing decodes.
local function mode_autoplay(handle, req, style)
    local games = math.floor(tonumber(req.games) or 1)
    local seed = math.floor(tonumber(req.seed) or 1)
    if games <= 0 then
        error("guardian_player_npc: task.games must be a positive integer")
    end
    if req.boss_style ~= nil then
        style = require_style(req.boss_style, "task.boss_style")
    end
    local boss = resolve_boss(req, style)

    local score, moves, raw_legal = 0.0, 0, 0
    local counts = {}
    for _, action in ipairs(MOVES) do
        counts[action] = 0
    end
    for i = 1, games do
        local g = duel.new_game(seed + i)
        while not duel.is_over(g) do
            local d = decide(handle, duel.player_view(g, style))
            moves = moves + 1
            if d.raw_legal then
                raw_legal = raw_legal + 1
            end
            counts[d.action] = counts[d.action] + 1
            g = duel.apply(g, d.action, boss(g.boss))
        end
        local w = duel.winner(g)
        if w == "player" then
            score = score + 1.0
        elseif w == "draw" then
            score = score + 0.5
        end
    end
    if moves == 0 then
        -- Unreachable while a fresh fight runs to the turn limit. Kept
        -- loud so a rules change that empties the loop cannot report the
        -- rates as a division by zero or as a silent 0.00.
        error("guardian_player_npc: autoplay made no move")
    end
    return string.format(
        "winrate=%.2f raw_legal=%.2f moves=%d a=%d A=%d b=%d p=%d",
        score / games,
        raw_legal / moves,
        moves,
        counts.a,
        counts.A,
        counts.b,
        counts.p
    )
end

-- ─── Strategy entry ─────────────────────────────────────────────────

--- Task fields every mode reads, whatever it does with them.
local COMMON_FIELDS = { mode = true, card_alias = true }

--- The modes, each with the fields it reads beyond the common ones.
local MODE_SPECS = {
    decide = { fields = { view = true }, run = mode_decide },
    determinism = { fields = { view = true }, run = mode_determinism },
    autoplay = {
        fields = { games = true, seed = true, boss_style = true, boss_card_alias = true },
        run = mode_autoplay,
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

local MODE_NAMES = table.concat(sorted_keys(MODE_SPECS), " / ")

--- Reject a task field the requested mode does not read.
---
--- A typo in a request field is otherwise invisible: the run reports a
--- number for the default boss, the default alias or the default game
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
            "guardian_player_npc: task field(s) %s are not read in %s mode (known: %s)",
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
        error("guardian_player_npc: card_alias must be a string, got " .. type(alias))
    end
    if #alias == 0 then
        error("guardian_player_npc: card_alias must not be empty")
    end
    return alias
end

---@param ctx table `{ task, card_alias?, boss_style? }`
---@return table result `{ result = <flat key=value summary> }`
function M.run(ctx)
    ctx = ctx or {}
    local task = ctx.task
    if type(task) ~= "string" then
        error("guardian_player_npc: ctx.task must be a JSON string, got " .. type(task))
    end
    local req = alc.json_decode(task)
    if type(req) ~= "table" or type(req.mode) ~= "string" then
        error("guardian_player_npc: ctx.task must decode to an object with a mode field")
    end
    local spec = MODE_SPECS[req.mode]
    if spec == nil then
        error(
            string.format(
                "guardian_player_npc: unknown mode %q (expected %s)",
                req.mode,
                MODE_NAMES
            )
        )
    end
    require_known_fields(req, req.mode, spec.fields)

    -- The boss a Card was logged against is a property of the Card
    -- rather than of the request, so it is read from the ctx and
    -- validated once for every mode: it decides the distance basis the
    -- views are built on, and an unknown one would measure against a
    -- threshold that does not exist.
    local style = DEFAULT_BOSS_STYLE
    if ctx.boss_style ~= nil then
        style = require_style(ctx.boss_style, "ctx.boss_style")
    end

    local handle = resolve_handle(resolve_alias(ctx, req))

    return { result = spec.run(handle, req, style) }
end

--- Drop cached handles. Exposed for the bake script, which binds a new
--- Card to the alias inside a VM that may already hold the old one.
function M.reset_cache()
    handle_cache = {}
end

return M
