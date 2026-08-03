--- level — win rate + Wilson 95% CI of one Card over N autoplay games,
--- measured from either seat against a pool of opponents.
---
--- ## Contract
---
--- `level(card, opponent, n_games?, seed?, opts?) -> table`
---
--- - `card` — string alias, or a live handle (table or userdata) with a
---   `generate_session(ids)` method, same shape as `style_distance`
---   (see `style_distance.lua`, including the `on_ckpt` +
---   `alc.nn.card.load_ckpt` usage). It occupies the seat `opts.seat`
---   names.
---
--- - `opponent` — the policy on the *other* seat, as a single value. The
---   pool form (`opts.opponents`) supersedes it; passing both is a loud
---   error rather than a silent precedence rule.
---
--- - `n_games` — optional positive integer, default 32. It is the count
---   **per opponent**: a pool of two runs `2 * n_games` fights, and the
---   returned `n_games` is that pool total.
---
--- - `seed` — optional non-negative integer, default 0. Fights use
---   `guardian_duel.new_game(seed + game_index)`, every stochastic
---   policy is seeded from `seed`, and every temperature draw takes
---   `seed + k` for a run-local counter `k`, so the whole pool replays
---   from it. It does *not* make two opponents a paired comparison:
---   `new_game` opens the same position for every seed (see Temperature
---   below), so there is no per-game opening for two runs to share. An
---   earlier revision of this header claimed otherwise; the claim is
---   withdrawn.
---
--- - `opts` — optional table, every field additive (omitting `opts`
---   entirely reproduces the pre-seat behaviour byte for byte):
---   - `seat` — `"player"` (default) or `"boss"`, the seat `card` plays.
---   - `style` — one of `guardian_duel.STYLES`. Required when
---     `seat = "boss"`: the boss prompt measures its distance field
---     against the style's mode-shift threshold, so a Card decoded under
---     a basis other than the one it was baked under reads a distance its
---     labels never followed. No default is invented.
---   - `opponents` — non-empty array of opponent names, one measurement
---     per entry. Duplicated names are a loud error.
---   - `opponent_style` — one of `guardian_duel.STYLES`. Required when
---     the opponent is a boss Card (`seat = "player"`), refused
---     everywhere else. It is the basis that Card's prompt encodes
---     distances against, for the reason `style` is required on the boss
---     seat, and it is also the basis the player view is built under so
---     both sides read the same board.
---   - `temperature` — positive finite number, or `nil` (the default) for
---     greedy decoding. See Temperature below.
---
--- ## Seats
---
--- - `seat = "player"` — `card` plays the player seat and the opponent
---   names the boss policy: `"greedy"` (default,
---   `guardian_duel.policy_guardian`), `"random"`
---   (`guardian_duel.policy_boss_random`, seeded from `seed`), or a boss
---   Card — a string alias resolved through `alc.card.get_by_alias`, or a
---   live handle passed directly. A boss Card is decoded through
---   `boss_seat` against `opts.opponent_style`, legal-gated to the moves
---   of the state. A win is a player win.
---
--- - `seat = "boss"` — `card` plays the boss seat through `boss_seat`
---   (encode against `style`, gated to the legal moves of the state) and
---   the opponent names the *player* policy: `"random"` (default,
---   `guardian_duel.policy_player_random`, seeded from `seed`), or a
---   player Card (alias or handle) decoded against the player view of the
---   same `opts.style` the boss reads — the fight runs on one basis, and
---   `opts.opponent_style` is refused here rather than silently ignored.
---   `"greedy"` stays a loud error: the repo carries no deterministic
---   player policy, so the name would promise something that does not
---   exist. A win is a boss win.
---
--- ## What a Card opponent does not cover
---
--- - `opponent_style` is one scalar for the whole pool, so a pool cannot
---   mix boss Cards baked under different bases. Per-opponent bases are
---   additive if a caller ever needs them.
--- - The reserved names shadow aliases: a Card bound to the alias
---   `"greedy"` or `"random"` is unreachable through this argument on the
---   seat that reserves the name. Pass its handle instead — the
---   `per_opponent` key then reads `"<table>"` / `"<userdata>"`, since a
---   handle carries no name.
--- - `opts.opponents` is an array of *names*, so the pool form reaches
---   Cards by alias only; a handle is a single-opponent call.
---
--- ## Temperature
---
--- `opts.temperature` is the only source of variance in a fight.
--- `guardian_duel` carries no RNG and `new_game` opens the same position
--- for every seed, so two greedy Cards replay one identical game N times
--- and a rate over that batch is a rate over a single sample. Naming a
--- temperature switches **every** Card decode of the run — the measured
--- Card and a Card opponent, on either seat — to a legal-masked draw
--- (`alc.nn.sampler.constrained(alc.nn.sampler.temperature(t, seed),
--- alc.nn.constraint.allow_list(ids))`, the same chain
--- `guardian_player_npc.decide_noisy` builds). The mask is applied inside
--- the sampler, so an illegal move is not rejected and redrawn — it is
--- not representable.
---
--- Each draw takes `seed + k` for a run-local counter `k` incremented
--- once per draw, so a run replays from its seed alone. The scripted
--- opponents (`"greedy"` / `"random"`) are unaffected: a temperature over
--- a policy that carries no logits means nothing.
---
--- ## Output
---
--- ```lua
--- {
---     win_rate     = 0.42,   -- over the whole pool
---     ci_lower     = 0.30,   -- pooled Wilson 95%
---     ci_upper     = 0.55,
---     win_rate_min = 0.38,   -- weakest single opponent
---     wins         = 21.0,   -- draws count as 0.5, so wins may be fractional
---     n_games      = 50,     -- pool total (n_games * #opponents)
---     game_length_mean     = 6.4,   -- turns played before the fight ended
---     final_hp_margin_mean = 3.1,   -- boss hp minus player hp, at the end
---     per_opponent = {
---         random = {
---             win_rate = 0.42, ci_lower = 0.30, ci_upper = 0.55, n_games = 50,
---             game_length_mean = 6.4, final_hp_margin_mean = 3.1,
---         },
---     },
--- }
--- ```
---
--- `win_rate` is the pooled rate (`wins / n_games`); since every opponent
--- plays the same `n_games`, that is also the mean of the per-opponent
--- rates. `win_rate_min` reports the pool's weakest matchup separately,
--- because a mean hides a single opponent the Card cannot beat.
---
--- `game_length_mean` counts the turns actually played (`state.turn - 1`
--- at the end, so a fight that ran to the turn limit reports the limit).
--- `final_hp_margin_mean` is always `boss.hp - player.hp` — a
--- boss-perspective quantity on **both** seats, so the sign does not
--- change meaning when the seat does. The two answer a question a rate
--- cannot: whether a matchup is a close brawl or a rout, which is the
--- same distinction a win rate near 0.5 leaves open.
---
--- The CI is a 95% Wilson score interval (`z = 1.96`) computed on the
--- wins-plus-half-draws count. Wilson is preferred over the
--- normal-approximation interval because it stays inside `[0, 1]` at
--- extreme `p̂` values. Two known approximations ride along and belong in
--- any write-up that quotes the number: draws are folded in as half a win
--- (so the count is not a Bernoulli one), and with a fixed `seed` the
--- interval describes the uncertainty over a fixed set of openings rather
--- than generalisation to unseen ones.
---
--- ## What this metric is and is not
---
--- It is a measurement of game-optimality from one seat: a rate and its
--- interval, nothing else. Reading it — comparing it against a target,
--- deciding whether a run should stop — belongs to the judgment layer
--- that consumes the record, not to this file (measurement and judgment
--- are separate layers by design).
---
--- ## Why compose here rather than call guardian_player_npc.run
---
--- `guardian_player_npc` reports one summary string per batch, so a
--- caller wanting a CI would have to run one game at a time and parse
--- N summaries. The inner loops below share the mask/decode contract
--- with `guardian_player_npc.decide` (player seat) and
--- `guardian_duel_npc.decide` (boss seat, via `boss_seat.lua`) but keep
--- the per-game outcome as a native Lua value the CI computation reads
--- directly.

local duel = require("guardian_duel")
local boss_seat = require("gameai_metrics.boss_seat")

local M

local VOCAB = duel.player_vocab()
local LEGAL_ACTIONS = duel.player_legal_actions()
local LEGAL_IDS_MAP = {}

--- The same ids as a list, in `player_legal_actions` order, which is the
--- shape `alc.nn.constraint.allow_list` takes. The four are legal on
--- every turn of every fight, so unlike the boss set this one is a
--- module constant rather than a per-state read.
local LEGAL_ID_LIST = {}
for _, action in ipairs(LEGAL_ACTIONS) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error("level: legal action " .. tostring(action) .. " is outside player_vocab")
    end
    LEGAL_IDS_MAP[id] = action
    LEGAL_ID_LIST[#LEGAL_ID_LIST + 1] = id
end

--- 95% Wilson score interval half-width factor.
local Z_95 = 1.959964

--- Opponent assumed when the caller names none, per seat.
local DEFAULT_OPPONENT = { player = "greedy", boss = "random" }

--- Opponent names that spell a scripted policy rather than a Card alias,
--- per seat. Anything else a caller names is read as an alias, which is
--- what makes these two names shadow any Card bound under them (see the
--- header).
local RESERVED_OPPONENTS = {
    player = { greedy = true, random = true },
    boss = { random = true, greedy = true },
}

local function require_nn_card()
    if type(alc) ~= "table" or type(alc.nn) ~= "table" or type(alc.nn.card) ~= "table" then
        error("level: alc.nn.card is unavailable; build algocline with --features nn")
    end
end

local function require_math_rng()
    if
        type(alc) ~= "table"
        or type(alc.math) ~= "table"
        or type(alc.math.rng_create) ~= "function"
    then
        error("level: alc.math.rng_create is required for the random opponent")
    end
end

--- The two namespaces a temperature draw goes through.
---
--- Checked apart from `require_nn_card` for the reason
--- `guardian_player_npc.require_sampler` is: a build with `alc.nn.card`
--- but no `alc.nn.constraint.allow_list` still answers every greedy
--- measurement, so a greedy caller must not be turned away by a surface
--- only the draw needs.
local function require_sampler()
    local nn = type(alc) == "table" and alc.nn or nil
    if
        type(nn) ~= "table"
        or type(nn.sampler) ~= "table"
        or type(nn.sampler.temperature) ~= "function"
        or type(nn.sampler.constrained) ~= "function"
        or type(nn.constraint) ~= "table"
        or type(nn.constraint.allow_list) ~= "function"
    then
        error(
            "level: alc.nn.sampler.temperature / .constrained and alc.nn.constraint.allow_list "
                .. "are required by opts.temperature; build algocline with --features nn"
        )
    end
    return nn
end

--- Match `style_distance.has_generate_session`; see that file for why
--- the method rather than the Lua type is what the guard reads, and why
--- the read is wrapped in `pcall`.
local function has_generate_session(card)
    local ok, method = pcall(function()
        return card.generate_session
    end)
    return ok and type(method) == "function"
end

local function resolve_card_handle(card, which)
    local kind = type(card)
    if kind == "table" or kind == "userdata" then
        if not has_generate_session(card) then
            error(
                string.format(
                    "level: %s is a %s but has no generate_session method; "
                        .. "expected a handle returned by alc.nn.card.load_handle "
                        .. "or alc.nn.card.load_ckpt",
                    which,
                    kind
                )
            )
        end
        return card
    end
    if type(card) == "string" then
        if #card == 0 then
            error(string.format("level: %s must be a non-empty string alias", which))
        end
        require_nn_card()
        if type(alc.card) ~= "table" or type(alc.card.get_by_alias) ~= "function" then
            error("level: alc.card.get_by_alias is unavailable")
        end
        local entry = alc.card.get_by_alias(card)
        if not entry then
            error(string.format("level: %s alias %q is not bound to any Card", which, card))
        end
        local card_id = entry.card_id
        if type(card_id) ~= "string" or #card_id == 0 then
            error(
                string.format("level: %s alias %q resolved to a Card without card_id", which, card)
            )
        end
        return alc.nn.card.load_handle(card_id)
    end
    error(
        string.format(
            "level: %s must be a string alias or a handle (table or userdata), got %s",
            which,
            type(card)
        )
    )
end

--- One greedy player decision, mask-gated to the four legal moves.
---
--- Byte-for-byte the same rule `guardian_player_npc.decide` applies:
--- the argmax may be a field letter, so the scan takes the first legal
--- id in the full ranking rather than the top one.
local function decide_greedy(handle, view)
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()
    local ranked = logits:top(logits:vocab())
    for _, entry in ipairs(ranked) do
        local action = LEGAL_IDS_MAP[entry.id]
        if action ~= nil then
            return action
        end
    end
    error("level: no player move found in the full logit ranking")
end

--- Check a requested temperature.
---
--- `nil` means greedy: every measurement in this package predates the
--- draw, so an omitted opt has to reproduce the old numbers exactly.
--- Zero is refused rather than folded into greedy — a caller who means
--- greedy already has a spelling for it, and a division by zero inside
--- the sampler is not the way to find out they did not.
local function decode_temperature(raw)
    if raw == nil then
        return nil
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw <= 0 then
        error(
            string.format(
                "level: opts.temperature must be a finite positive number, got %s",
                tostring(raw)
            )
        )
    end
    return raw
end

--- Seed source shared by every draw of one run.
---
--- The sampler takes a seed per decision rather than carrying one across
--- them, so the caller derives it: `base + k` for a counter incremented
--- once per draw. That is what makes a whole run — both seats, every
--- game — replay from the one `seed` argument.
local function new_draws(base_seed)
    return { base = base_seed, count = 0 }
end

local function next_seed(draws)
    local seed = draws.base + draws.count
    draws.count = draws.count + 1
    return seed
end

--- Draw one id under `temperature`, masked to `allow_ids`.
---
--- The chain is rebuilt per decision because `alc.nn.sampler.constrained`
--- moves both of its arguments — one held across decisions would be a
--- spent handle on the second.
local function sample_id(logits, allow_ids, temperature, seed)
    local nn = require_sampler()
    local sampler = nn.sampler.constrained(
        nn.sampler.temperature(temperature, seed),
        nn.constraint.allow_list(allow_ids)
    )
    return sampler:sample(logits)
end

--- One player decision: greedy when `temperature` is nil, otherwise a
--- legal-masked draw taking the next seed of the run.
local function decide_player(handle, view, temperature, draws)
    if temperature == nil then
        return decide_greedy(handle, view)
    end
    local prompt = duel.player_encode(view) .. ">"
    local session = handle:generate_session(duel.player_to_ids(prompt))
    local logits = session:next_logits()
    local id = sample_id(logits, LEGAL_ID_LIST, temperature, next_seed(draws))
    local action = LEGAL_IDS_MAP[id]
    if action == nil then
        -- Unreachable while the mask holds: the allow list is the four
        -- moves. Kept loud so a mask that stopped binding surfaces here
        -- rather than as an illegal move reaching `guardian_duel.apply`.
        error(
            string.format(
                "level: the constrained sampler drew token %s, which is not a player move",
                tostring(id)
            )
        )
    end
    return action
end

--- One boss decision, the same two modes over the state-dependent legal
--- set `boss_seat` computes.
local function decide_boss(handle, state, style, temperature, draws)
    if temperature == nil then
        return boss_seat.decide(handle, state, style)
    end
    local legal = boss_seat.legal(state)
    local session = handle:generate_session(duel.to_ids(boss_seat.encode(state, style)))
    local logits = session:next_logits()
    local id = sample_id(logits, legal.ids, temperature, next_seed(draws))
    local action = legal.by_id[id]
    if action == nil then
        error(
            string.format(
                "level: the constrained sampler drew token %s, which is not a legal boss move",
                tostring(id)
            )
        )
    end
    return action
end

--- Whether an opponent spec names a Card rather than a scripted policy.
---
--- The reserved names are checked before the alias lookup, which is what
--- makes them shadow a Card bound under the same name (header).
local function is_card_opponent(spec, seat)
    if spec == nil then
        return false
    end
    if type(spec) == "string" then
        return not RESERVED_OPPONENTS[seat][spec]
    end
    return type(spec) == "table" or type(spec) == "userdata"
end

--- Boss chooser factory, used when the Card plays the player seat.
---
--- Returns a `fn(boss_state) -> boss_action` closure plus the style the
--- player view on the other side of the same fight is built under. The
--- closure carries its own RNG for `"random"` so the run is reproducible
--- from `seed` alone.
---
--- The scripted policies keep the pre-Card view basis (`"guardian"`,
--- which is what `policy_guardian` follows) so an unchanged call reads
--- the same numbers byte for byte. A boss Card moves the basis to
--- `opponent_style`: the two sides of a fight have to read one board,
--- and the Card's own prompt is encoded against that basis.
local function boss_from(opponent, seed, opponent_style, temperature, draws)
    if opponent == nil or opponent == "greedy" then
        return function(boss)
            return duel.policy_guardian(boss)
        end, "guardian"
    end
    if opponent == "random" then
        require_math_rng()
        local rng = alc.math.rng_create(seed)
        return function(boss)
            return duel.policy_boss_random(boss, rng)
        end,
            "guardian"
    end
    local handle = resolve_card_handle(opponent, "opponent")
    return function(boss)
        return decide_boss(handle, boss, opponent_style, temperature, draws)
    end,
        opponent_style
end

--- Player chooser factory, used when the Card plays the boss seat.
---
--- Returns a `fn(state, boss_action) -> player_action` closure. The two
--- arguments are what a player Card needs to build its view; `"random"`
--- ignores them, so the scripted path is unchanged.
---
--- `"greedy"` is refused rather than resolved as an alias: the repo
--- carries no deterministic *player* policy (the player side is a Card,
--- not a scripted style), and letting the name fall through to the alias
--- lookup would answer a caller who meant the policy with a message
--- about a missing Card.
local function player_from(opponent, seed, style, temperature, draws)
    if opponent == "random" then
        require_math_rng()
        local rng = alc.math.rng_create(seed)
        return function()
            return duel.policy_player_random(rng)
        end
    end
    if opponent == "greedy" then
        error(
            'level: seat="boss" opponent "greedy" is not implemented; the only scripted player '
                .. 'policy this iter carries is "random" (any other name is read as a player '
                .. "Card alias)"
        )
    end
    local handle = resolve_card_handle(opponent, "opponent")
    return function(state, boss_action)
        local view = duel.player_view(state, style, state.revealed and boss_action or nil)
        return decide_player(handle, view, temperature, draws)
    end
end

local function decode_int(raw, default, field, must_be_positive)
    if raw == nil then
        return default
    end
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge then
        error(string.format("level: %s must be a finite integer, got %s", field, tostring(raw)))
    end
    local i = math.floor(raw)
    if must_be_positive and i <= 0 then
        error(string.format("level: %s must be a positive integer, got %s", field, tostring(raw)))
    end
    if not must_be_positive and i < 0 then
        error(
            string.format("level: %s must be a non-negative integer, got %s", field, tostring(raw))
        )
    end
    return i
end

--- Wilson 95% score interval around `p̂ = wins / n`. Never leaves
--- `[0, 1]`, which is why it is used here rather than the normal
--- approximation.
local function wilson_ci(wins, n)
    if n <= 0 then
        error("level: wilson_ci requires n > 0")
    end
    local p = wins / n
    local z = Z_95
    local z2 = z * z
    local denom = 1 + z2 / n
    local center = (p + z2 / (2 * n)) / denom
    local half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    local lo = center - half
    local hi = center + half
    if lo < 0 then
        lo = 0
    end
    if hi > 1 then
        hi = 1
    end
    -- In exact arithmetic the interval always contains `p̂`, and at the
    -- endpoints it touches it: at `p̂ = 1` the two halves of the numerator
    -- sum to the denominator, so `hi` is exactly 1. In floating point the
    -- residue lands a few ulp short, which would leave a reader of
    -- `ci_lower` / `ci_upper` with a bound that excludes the estimate it
    -- brackets. The clamp repairs only that residue.
    if lo > p then
        lo = p
    end
    if hi < p then
        hi = p
    end
    return lo, hi
end

--- Normalise the two opponent forms into one list of names.
---
--- The single `opponent` argument and the `opts.opponents` pool are
--- mutually exclusive: honouring one and dropping the other silently
--- would answer a question the caller did not ask.
local function decode_pool(opponent, opponents, seat)
    if opponents == nil then
        local spec = opponent
        if spec == nil then
            spec = DEFAULT_OPPONENT[seat]
        end
        local name = type(spec) == "string" and spec or ("<" .. type(spec) .. ">")
        return { { name = name, spec = spec } }
    end
    if opponent ~= nil then
        error("level: pass either opponent or opts.opponents, not both")
    end
    if type(opponents) ~= "table" then
        error("level: opts.opponents must be an array of names, got " .. type(opponents))
    end
    local n = #opponents
    if n == 0 then
        error("level: opts.opponents is empty; nothing to measure against")
    end
    local pool, seen = {}, {}
    for i = 1, n do
        local name = opponents[i]
        if type(name) ~= "string" or #name == 0 then
            error(
                string.format(
                    "level: opts.opponents[%d] must be a non-empty opponent name, got %s",
                    i,
                    type(name)
                )
            )
        end
        if seen[name] then
            error(string.format("level: opts.opponents lists %q twice", name))
        end
        seen[name] = true
        pool[#pool + 1] = { name = name, spec = name }
    end
    return pool
end

--- The pool as one comma-separated list, for error messages.
local function pool_names(pool)
    local names = {}
    for i, entry in ipairs(pool) do
        names[i] = entry.name
    end
    return table.concat(names, ", ")
end

--- Decode `opts.opponent_style` against the pool it is meant for.
---
--- Required as soon as one pool entry names a boss Card, refused when
--- none does. An unused opt accepted in silence would read as "the basis
--- I asked for is in effect", which is the misreading this metric can
--- least afford: a Card decoded under the wrong basis answers from a
--- distance field its labels never followed, and still returns a number.
local function decode_opponent_style(raw, pool, seat)
    local named
    for _, entry in ipairs(pool) do
        if is_card_opponent(entry.spec, seat) then
            named = named or entry.name
        end
    end
    if seat ~= "player" or named == nil then
        if raw ~= nil then
            error(
                string.format(
                    "level: opts.opponent_style names the basis a boss Card opponent is decoded "
                        .. "under, but this call has none (seat=%q, opponents: %s); drop the opt "
                        .. "rather than leave it reading as though it applied",
                    seat,
                    pool_names(pool)
                )
            )
        end
        return nil
    end
    if raw == nil then
        error(
            string.format(
                "level: opponent %q is read as a boss Card, which needs opts.opponent_style — "
                    .. "the basis its prompt encodes distances against (one of %s). There is no "
                    .. 'default. The scripted policies are "greedy" and "random".',
                named,
                table.concat(duel.STYLES, ", ")
            )
        )
    end
    return boss_seat.require_style(raw, "level: opts.opponent_style")
end

--- Score one finished fight from `seat`'s point of view.
local function outcome(winner, seat)
    if winner == seat then
        return 1.0
    end
    if winner == "draw" then
        return 0.5
    end
    return 0.0
end

--- Turns actually played by a finished fight.
---
--- `guardian_duel` numbers the turn about to be played, so a fight that
--- ended on turn `n` carries `n + 1`. A fight that ran out the clock
--- reports `TURN_LIMIT`.
local function game_length(state)
    return state.turn - 1
end

--- Fights where the Card holds the player seat.
---
--- Returns the win total plus the two per-fight aggregates, all summed
--- over `games` (the caller divides, so a pooled mean stays exact).
local function play_player_seat(handle, spec, games, base_seed, cfg)
    local boss, view_style =
        boss_from(spec, base_seed, cfg.opponent_style, cfg.temperature, cfg.draws)
    local wins = 0.0
    local turns, margin = 0.0, 0.0
    for g = 1, games do
        local state = duel.new_game(base_seed + g)
        while not duel.is_over(state) do
            local boss_action = boss(state.boss)
            local view = duel.player_view(state, view_style, state.revealed and boss_action or nil)
            local player_action = decide_player(handle, view, cfg.temperature, cfg.draws)
            state = duel.apply(state, player_action, boss_action)
        end
        wins = wins + outcome(duel.winner(state), "player")
        turns = turns + game_length(state)
        margin = margin + (state.boss.hp - state.player.hp)
    end
    return wins, turns, margin
end

--- Fights where the Card holds the boss seat.
local function play_boss_seat(handle, spec, games, base_seed, style, cfg)
    local player = player_from(spec, base_seed, style, cfg.temperature, cfg.draws)
    local wins = 0.0
    local turns, margin = 0.0, 0.0
    for g = 1, games do
        local state = duel.new_game(base_seed + g)
        while not duel.is_over(state) do
            local boss_action = decide_boss(handle, state.boss, style, cfg.temperature, cfg.draws)
            local player_action = player(state, boss_action)
            state = duel.apply(state, player_action, boss_action)
        end
        wins = wins + outcome(duel.winner(state), "boss")
        turns = turns + game_length(state)
        margin = margin + (state.boss.hp - state.player.hp)
    end
    return wins, turns, margin
end

M = function(card, opponent, n_games, seed, opts)
    if opts ~= nil and type(opts) ~= "table" then
        error("level: opts must be a table, got " .. type(opts))
    end
    opts = opts or {}

    local games = decode_int(n_games, 32, "n_games", true)
    local base_seed = decode_int(seed, 0, "seed", false)
    local seat = boss_seat.require_seat(opts.seat, "level")
    local style = nil
    if seat == "boss" then
        style = boss_seat.require_style(opts.style, "level")
    end
    local pool = decode_pool(opponent, opts.opponents, seat)
    local opponent_style = decode_opponent_style(opts.opponent_style, pool, seat)
    local temperature = decode_temperature(opts.temperature)
    if temperature ~= nil then
        -- Checked before any fight so a build without the sampler fails
        -- on the argument rather than after N games of setup.
        require_sampler()
    end
    local cfg = {
        opponent_style = opponent_style,
        temperature = temperature,
        draws = new_draws(base_seed),
    }

    local handle = resolve_card_handle(card, "card")

    local per_opponent = {}
    local total_wins, total_games = 0.0, 0
    local total_turns, total_margin = 0.0, 0.0
    local win_rate_min
    for _, entry in ipairs(pool) do
        local wins, turns, margin
        if seat == "boss" then
            wins, turns, margin = play_boss_seat(handle, entry.spec, games, base_seed, style, cfg)
        else
            wins, turns, margin = play_player_seat(handle, entry.spec, games, base_seed, cfg)
        end
        local rate = wins / games
        local lo, hi = wilson_ci(wins, games)
        per_opponent[entry.name] = {
            win_rate = rate,
            ci_lower = lo,
            ci_upper = hi,
            n_games = games,
            game_length_mean = turns / games,
            final_hp_margin_mean = margin / games,
        }
        total_wins = total_wins + wins
        total_games = total_games + games
        total_turns = total_turns + turns
        total_margin = total_margin + margin
        if win_rate_min == nil or rate < win_rate_min then
            win_rate_min = rate
        end
    end

    local lo, hi = wilson_ci(total_wins, total_games)
    return {
        win_rate = total_wins / total_games,
        ci_lower = lo,
        ci_upper = hi,
        wins = total_wins,
        n_games = total_games,
        win_rate_min = win_rate_min,
        game_length_mean = total_turns / total_games,
        final_hp_margin_mean = total_margin / total_games,
        per_opponent = per_opponent,
    }
end

return M
