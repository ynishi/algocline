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
--- answer with a digit, a field letter, a boss letter (the intent field
--- of the view puts the six of them in the alphabet) or the padding
--- token — a token that is not a player move at all — and a run whose
--- `raw_legal` rate sags is a model that has stopped answering the
--- question rather than one playing badly.
---
--- ## The noisy decode
---
--- Greedy decoding answers one position one way forever, which is what
--- the determinism check fences and what makes a batch of fights a
--- batch of copies. The noisy path answers the same position by
--- *drawing* from the model, and it stays legal by construction rather
--- than by a scan: a temperature sampler is wrapped in
--- `alc.nn.constraint.allow_list` over the four move ids, so every
--- other logit is `-inf` before the draw happens.
---
--- The chain is rebuilt for every decision. That is the intended shape
--- rather than a cost: `alc.nn.sampler.constrained` consumes both of
--- its arguments (a sampler owns its RNG, and two handles onto one RNG
--- would interleave draws from generations that each believe they are
--- reproducible from their seed), so a cached chain is a spent one. The
--- seed is therefore required rather than defaulted — the caller
--- derives it, and a replay that derives it the same way draws the same
--- move.
---
--- `raw_legal` is computed on the noisy path exactly as on the greedy
--- one, off the ungated argmax. It is the same health signal there: the
--- draw says what the model played, the argmax says whether it was
--- still answering the question.
---
--- ## Entry contract
---
--- `ctx.task` is a JSON object with a `mode` field:
---
--- - `decide` — `{ view }`; returns
---   `action=<move> legal=true raw_legal=<bool> gated=<bool>`
--- - `decide_noisy` — `{ view, seed, temperature? }`; draws one move
---   under the legal mask and returns
---   `action=<move> legal=true raw_legal=<bool> noisy=true
---   temperature=<t> seed=<n>`. `seed` is required and `temperature`
---   defaults to 1.0. There is no `gated` field: a draw that lands
---   away from the argmax is the sampler doing its job, not a gate
---   stepping in.
--- - `determinism` — `{ view }`; decodes twice through independent
---   sessions and returns `deterministic=<bool> action=<move>`
--- - `autoplay` — `{ games, seed, temperature?, boss_style?,
---   boss_card_alias? }`;
---   plays the model against a boss and returns
---   `winrate=<x.xx> raw_legal=<x.xx> moves=<n> a=<n> A=<n> b=<n>
---   p=<n>`. The win rate is read from the player seat with a draw
---   counted as half a win, and the four counts are the model's own
---   move distribution. A one-game run appends
---   `player_seq=<moves> boss_seq=<moves>`, the two seats' answers turn
---   by turn in the order they were played, so a caller replaying the
---   fight can check its own transcript against the run that produced
---   it without replaying the loop to find out what happened. A batch
---   is a repeat of one fight, so the pair is left out above one game
---   rather than printed as many identical copies of itself. The views
---   the moves were chosen from are not reported either: the fight is a
---   function of the seed and the two sequences, so `guardian_duel`
---   rebuilds any of them from what is already here.
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
--- `temperature` is what turns it into a sample. With it present every
--- player decision of the run goes through the noisy path, and `games`
--- becomes a real sample size: a win rate over ten noisy fights is ten
--- fights rather than one counted ten times, which is what a
--- temperature sweep reads. The summary then carries `noisy=true
--- temperature=<t>` so three runs at three temperatures are three
--- self-describing lines. The sampler seed of one decision is
--- `seed + game * (TURN_LIMIT + 1) + turn`, with `game` the 1-based
--- index of the fight, `turn` the turn it is being played on and a
--- stride one wider than the longest possible fight — so no two
--- decisions of a run share a draw, and the whole run is still a
--- function of `seed`. `seed` is checked on this path exactly as the
--- `decide_noisy` one is — required, a non-negative finite number,
--- never defaulted — since the derived seeds are what the sampler is
--- built from. Greedy, where the seed only picks a board, it stays the
--- lenient optional field it has always been.
---
--- `boss_style` is the boss the fights are played against and, with it,
--- the distance basis every generated view is measured against. Those
--- are one field rather than two because they have to agree: the `D`
--- field of a player view is the damage *that* boss still tolerates, so
--- a model logged against the teacher and autoplayed against the
--- impatient variant would be reading a number that means something
--- else. `boss_card_alias` seats a boss Card instead of the teacher
--- policy, decoded through `guardian_duel_npc` under the same basis.
--- With a Card seated and no `boss_style`, the basis is read from the
--- Card itself — `persona.basis_style`, the field the persona bake
--- records — because defaulting to the teacher basis here would
--- measure every view against a boss that is not seated. A seated
--- Card that records no basis (the canonical teacher Cards do not) is
--- a loud error asking for `boss_style` rather than a guess.
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
                "JSON object with a mode field: decide / decide_noisy / determinism / "
                    .. "autoplay (decide_noisy takes the seed it draws under, autoplay "
                    .. "the boss it plays against)"
            ),
            card_alias = T.string:is_optional():describe(
                "Card alias holding the tuned model, also readable from the task JSON "
                    .. "(default: guardian_player_npc)"
            ),
            boss_style = T.string:is_optional():describe(
                "Boss the autoplay fights are played against, and the distance basis "
                    .. "every generated view is measured against (default: guardian, "
                    .. "or the basis recorded on a seated boss Card)"
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
            "guardian_player_npc: alc.nn.sampler.temperature / .constrained and "
                .. "alc.nn.constraint.allow_list are required for a noisy decode"
        )
    end
    return nn
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

--- The same four ids as a list, in `MOVES` order, which is the shape
--- `alc.nn.constraint.allow_list` takes.
---
--- Resolved here rather than per handle, let alone per decision: the
--- alphabet a player Card is decoded under is `guardian_duel`'s own
--- (`player_vocab`, the same table `player_to_ids` builds the prompt
--- with), so the ids are a property of the rules module and not of a
--- loaded model. A Card whose tokenizer disagreed with it would already
--- be mis-prompted, which is the louder failure and the one that
--- happens first. A move with no id at all fails at load, before any
--- caller can be answered with a mask that quietly lost an option.
local LEGAL_ID_LIST = {}

for _, action in ipairs(MOVES) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error(string.format("guardian_player_npc: move %s has no token id", tostring(action)))
    end
    LEGAL_IDS[id] = action
    LEGAL_ID_LIST[#LEGAL_ID_LIST + 1] = id
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
                "guardian_player_npc: task.temperature must be a finite positive number, got %s",
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
                "guardian_player_npc: %s is required for a noisy decode; derive it from the "
                    .. "caller's own counter so the draw can be replayed",
                field
            )
        )
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw < 0 then
        error(
            string.format(
                "guardian_player_npc: %s must be a non-negative finite number, got %s",
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
--- logit outside `LEGAL_ID_LIST` is `-inf` before the draw, so an
--- illegal move is not rejected and redrawn, it is not representable.
--- The chain is built here, per decision, because
--- `alc.nn.sampler.constrained` moves both of its arguments — a chain
--- held across decisions would be a spent handle on the second one.
---@param handle userdata NnHandle
---@param view table Player view
---@param temperature number Draw temperature, already validated
---@param seed integer Sampler seed, already validated
---@return table decision `{ action, raw_legal }`
local function decide_noisy(handle, view, temperature, seed)
    local nn = require_sampler()
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()

    local raw_legal = LEGAL_IDS[logits:argmax()] ~= nil

    local sampler = nn.sampler.constrained(
        nn.sampler.temperature(temperature, seed),
        nn.constraint.allow_list(LEGAL_ID_LIST)
    )
    local id = sampler:sample(logits)
    local action = LEGAL_IDS[id]
    if action == nil then
        -- Unreachable while the mask holds: the allow list is the four
        -- moves. Kept loud so a mask that stopped binding surfaces here
        -- rather than as an illegal move reaching `guardian_duel.apply`.
        error(
            string.format(
                "guardian_player_npc: the constrained sampler drew token %s, "
                    .. "which is not a player move",
                tostring(id)
            )
        )
    end
    return { action = action, raw_legal = raw_legal }
end

-- ─── Modes ──────────────────────────────────────────────────────────

--- The player view a decode request carries.
---
--- Nothing is defaulted: `guardian_duel` validates every field the
--- encoding reads and names the one that is missing, whereas a default
--- would answer a question the caller did not ask — a substituted
--- `shift_distance` moves the field the model reads hardest, and a
--- substituted `intent` would tell the model the board showed nothing
--- on a turn the player had bought a look at.
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

--- Format a temperature for the summary.
---
--- `%g` rather than a fixed number of decimals: the field is an echo of
--- what the caller asked for, and rounding an echo makes a sweep report
--- a temperature nobody ran.
local function format_temperature(t)
    return string.format("%g", t)
end

local function mode_decide_noisy(handle, req)
    local view = decode_view(req)
    local temperature = decode_temperature(req.temperature)
    local seed = require_seed(req.seed, "task.seed")
    local d = decide_noisy(handle, view, temperature, seed)
    return string.format(
        "action=%s legal=true raw_legal=%s noisy=true temperature=%s seed=%d",
        d.action,
        tostring(d.raw_legal),
        format_temperature(temperature),
        seed
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

--- Distance basis recorded on a seated boss Card.
---
--- A persona bake writes the basis its corpus was encoded against onto
--- the Card (`persona.basis_style`), which is where
--- `guardian_duel_interactive` reads it from too. A Card that carries
--- none — the canonical teacher Cards do not — cannot name its own
--- threshold, so the caller has to: guessing the teacher default here
--- is exactly the footgun this lookup replaces, a `D` field measured
--- against a boss that is not seated.
---@param alias string Boss Card alias, already validated
---@return string basis
local function card_basis(alias)
    if
        type(alc) ~= "table"
        or type(alc.card) ~= "table"
        or type(alc.card.get_by_alias) ~= "function"
    then
        error("guardian_player_npc: alc.card.get_by_alias is unavailable")
    end
    local card = alc.card.get_by_alias(alias)
    if type(card) ~= "table" then
        error(string.format("guardian_player_npc: no Card bound to boss alias %q", alias))
    end
    local persona = card.persona
    local basis = type(persona) == "table" and persona.basis_style or nil
    if basis == nil then
        error(
            string.format(
                "guardian_player_npc: boss Card %q carries no persona.basis_style; "
                    .. "pass boss_style to name the distance basis its fights are measured against",
                alias
            )
        )
    end
    return require_style(basis, "persona.basis_style of " .. alias)
end

--- Seat the opponent for one autoplay run.
---
--- With no boss Card the boss is the teacher policy of `style`, which
--- makes a run reproducible from its seed alone. With one, the boss is
--- another Card, decoded through `guardian_duel_npc` under the same
--- basis — the shape a demo uses to put a baked player against a baked
--- boss.
---@param alias string|nil Boss Card alias, already validated
---@param style string Boss style, the teacher default and the basis
---@return fun(state: table): string boss
local function resolve_boss(alias, style)
    if alias == nil then
        return duel["policy_" .. style]
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

--- Sampler seed of one decision of a noisy autoplay.
---
--- `seed + game * (TURN_LIMIT + 1) + turn`. The stride is one wider
--- than the longest fight the rules allow, so `(game, turn)` maps to a
--- distinct seed and no two decisions of a run draw the same way by
--- accident. Deriving it rather than letting one sampler run down the
--- whole batch is what keeps a fight replayable on its own: a caller
--- holding the run seed can rebuild the draw of turn 4 of game 2
--- without replaying the three fights before it.
---@param seed integer Run seed, already validated as non-negative
---@param game integer 1-based index of the fight
---@param turn integer Turn the decision is played on
---@return integer seed
local function noisy_turn_seed(seed, game, turn)
    if turn > duel.TURN_LIMIT then
        -- Unreachable while `guardian_duel` ends a fight at its own
        -- limit. Kept loud because a rules change that lengthened the
        -- fight would otherwise silently collide two turns' seeds.
        error(
            string.format(
                "guardian_player_npc: turn %d is past the %d-turn limit the seed stride assumes",
                turn,
                duel.TURN_LIMIT
            )
        )
    end
    return seed + game * (duel.TURN_LIMIT + 1) + turn
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
---
--- That is also why the turn-by-turn sequences are reported for a
--- single game and only then. They are what a replay is checked
--- against — a ghost that walks the same fight has to answer the same
--- moves in the same order, and a rate cannot say whether it did — but
--- above one game every copy is the same string, so a batch would pay
--- for the summary N times to learn what the first game already said.
---
--- A `temperature` moves every one of those statements: the decisions
--- are drawn instead of scanned, so the games of a batch differ, the
--- win rate is a rate over samples, and the sequences of a lone fight
--- are the transcript of one draw rather than of the only fight this
--- pairing has. They are still reported for a lone fight and still
--- replayable, because the seed of every decision is derived from the
--- run seed alone (see `noisy_turn_seed`).
local function mode_autoplay(handle, req, style)
    local games = math.floor(tonumber(req.games) or 1)
    if games <= 0 then
        error("guardian_player_npc: task.games must be a positive integer")
    end
    -- `nil` is the greedy path, byte for byte what it was before the
    -- noisy one existed; a number switches every decision of the run.
    local temperature = req.temperature ~= nil and decode_temperature(req.temperature) or nil
    local seed
    if temperature == nil then
        -- Greedy, the seed only picks a board and a fight is the same
        -- fight whichever number lands here, so a loose one is taken as
        -- it always was.
        seed = math.floor(tonumber(req.seed) or 1)
    else
        -- Noisy, it is the base every per-decision sampler seed is
        -- derived from, so it is checked exactly as `decide_noisy`
        -- checks its own: a malformed seed quietly becoming 1 would
        -- make the whole run reproducible from a number the caller
        -- never wrote down.
        seed = require_seed(req.seed, "task.seed")
    end
    local alias = req.boss_card_alias
    if alias ~= nil and (type(alias) ~= "string" or #alias == 0) then
        error("guardian_player_npc: task.boss_card_alias must be a non-empty string")
    end
    if req.boss_style ~= nil then
        style = require_style(req.boss_style, "task.boss_style")
    end
    if style == nil then
        -- Nobody named a basis. With a boss Card seated the Card names
        -- it; with the teacher seated the teacher is its own basis.
        style = alias ~= nil and card_basis(alias) or DEFAULT_BOSS_STYLE
    end
    local boss = resolve_boss(alias, style)

    local score, moves, raw_legal = 0.0, 0, 0
    -- Collected only for a lone fight, which is also the only run whose
    -- sequences are reported: keeping them for a batch would build a
    -- list per game that nothing ever reads.
    local player_seq = games == 1 and {} or nil
    local boss_seq = games == 1 and {} or nil
    local counts = {}
    for _, action in ipairs(MOVES) do
        counts[action] = 0
    end
    for i = 1, games do
        local g = duel.new_game(seed + i)
        while not duel.is_over(g) do
            -- The boss answers from the position at the head of the
            -- turn, which the player's own move never touches, so
            -- asking it first changes nothing about the fight and is
            -- what makes the reveal available: on a turn the model
            -- poked for, this is the move the board would have shown
            -- it, and the view has to carry it or the model is
            -- autoplayed on a question it was never trained on.
            local boss_action = boss(g.boss)
            local view = duel.player_view(g, style, g.revealed and boss_action or nil)
            local d
            if temperature == nil then
                d = decide(handle, view)
            else
                d = decide_noisy(handle, view, temperature, noisy_turn_seed(seed, i, g.turn))
            end
            moves = moves + 1
            if d.raw_legal then
                raw_legal = raw_legal + 1
            end
            counts[d.action] = counts[d.action] + 1
            if player_seq then
                player_seq[#player_seq + 1] = d.action
                boss_seq[#boss_seq + 1] = boss_action
            end
            g = duel.apply(g, d.action, boss_action)
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
    local summary = string.format(
        "winrate=%.2f raw_legal=%.2f moves=%d a=%d A=%d b=%d p=%d",
        score / games,
        raw_legal / moves,
        moves,
        counts.a,
        counts.A,
        counts.b,
        counts.p
    )
    if temperature ~= nil then
        -- A rate over draws and a rate over one repeated fight are not
        -- the same measurement, so the line says which one it is.
        summary =
            string.format("%s noisy=true temperature=%s", summary, format_temperature(temperature))
    end
    if player_seq then
        summary = string.format(
            "%s player_seq=%s boss_seq=%s",
            summary,
            table.concat(player_seq),
            table.concat(boss_seq)
        )
    end
    return summary
end

-- ─── Strategy entry ─────────────────────────────────────────────────

--- Task fields every mode reads, whatever it does with them.
local COMMON_FIELDS = { mode = true, card_alias = true }

--- The modes, each with the fields it reads beyond the common ones.
local MODE_SPECS = {
    decide = { fields = { view = true }, run = mode_decide },
    decide_noisy = {
        fields = { view = true, temperature = true, seed = true },
        run = mode_decide_noisy,
    },
    determinism = { fields = { view = true }, run = mode_determinism },
    autoplay = {
        fields = {
            games = true,
            seed = true,
            temperature = true,
            boss_style = true,
            boss_card_alias = true,
        },
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

    -- A ctx basis is validated once for every mode: it decides the
    -- distance basis the views are built on, and an unknown one would
    -- measure against a threshold that does not exist. It is left nil
    -- rather than defaulted when the ctx names none, because the right
    -- default depends on the boss seat — a seated boss Card names its
    -- own basis — and only autoplay knows what is seated.
    local style = nil
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
