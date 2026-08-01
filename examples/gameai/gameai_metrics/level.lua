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
---   `guardian_duel.new_game(seed + game_index)` and every stochastic
---   policy is seeded from `seed`, so the whole pool replays from it.
---   Each opponent replays the same game seeds, which makes the
---   comparison between two opponents a paired one.
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
---
--- ## Seats
---
--- - `seat = "player"` — `card` plays the player seat and the opponent
---   names the boss policy: `"greedy"` (default,
---   `guardian_duel.policy_guardian`), `"random"`
---   (`guardian_duel.policy_boss_random`, seeded from `seed`). A win is a
---   player win. Boss Card handles and other aliases stay unwired in this
---   iter and are refused loudly.
---
--- - `seat = "boss"` — `card` plays the boss seat through
---   `boss_seat.decide` (encode against `style`, greedy gated to the legal
---   moves of the state) and the opponent names the *player* policy:
---   `"random"` (default, `guardian_duel.policy_player_random`, seeded
---   from `seed`) is the only one this iter implements, because the repo
---   carries no deterministic player policy to fight against. Any other
---   name is a loud error naming what is available. A win is a boss win.
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
---     per_opponent = {
---         random = { win_rate = 0.42, ci_lower = 0.30, ci_upper = 0.55, n_games = 50 },
---     },
--- }
--- ```
---
--- `win_rate` is the pooled rate (`wins / n_games`); since every opponent
--- plays the same `n_games`, that is also the mean of the per-opponent
--- rates. `win_rate_min` reports the pool's weakest matchup separately,
--- because a mean hides a single opponent the Card cannot beat.
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
for _, action in ipairs(LEGAL_ACTIONS) do
    local id = VOCAB.to_id[action]
    if id == nil then
        error("level: legal action " .. tostring(action) .. " is outside player_vocab")
    end
    LEGAL_IDS_MAP[id] = action
end

--- 95% Wilson score interval half-width factor.
local Z_95 = 1.959964

--- Opponent assumed when the caller names none, per seat.
local DEFAULT_OPPONENT = { player = "greedy", boss = "random" }

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

--- Boss chooser factory, used when the Card plays the player seat.
---
--- Returns a `fn(boss_state) -> boss_action` closure. The closure carries
--- its own RNG for `"random"` so the run is reproducible from `seed`
--- alone.
local function boss_from(opponent, seed)
    if opponent == nil or opponent == "greedy" then
        return function(boss)
            return duel.policy_guardian(boss)
        end
    end
    if opponent == "random" then
        require_math_rng()
        local rng = alc.math.rng_create(seed)
        return function(boss)
            return duel.policy_boss_random(boss, rng)
        end
    end
    if type(opponent) == "table" or type(opponent) == "userdata" then
        if not has_generate_session(opponent) then
            error(
                string.format(
                    "level: opponent is a %s but has no generate_session method; "
                        .. "expected a boss handle or a policy fn",
                    type(opponent)
                )
            )
        end
        -- Treat as a boss Card handle. Decoding a boss NPC goes through
        -- `guardian_duel_npc`, which is a whole separate strategy — out
        -- of scope for this iter.
        error(
            "level: boss Card handle opponent is not wired in this iter; "
                .. 'pass "greedy" or "random", or call guardian_duel_npc directly'
        )
    end
    if type(opponent) == "string" then
        error(
            string.format(
                "level: opponent alias %q is not wired in this iter; "
                    .. 'pass "greedy" or "random" (Follow-up: boss Card seat via guardian_duel_npc)',
                opponent
            )
        )
    end
    error('level: opponent must be "greedy" / "random" / string alias, got ' .. type(opponent))
end

--- Player chooser factory, used when the Card plays the boss seat.
---
--- Returns a `fn() -> player_action` closure; the player moves are legal
--- on every turn, so the state is not an argument.
---
--- Only `"random"` is implemented: the repo carries no deterministic
--- player policy (the player side is a Card, not a scripted style), so a
--- name that promised one would measure something that does not exist.
--- Widening this pool (teacher player Cards, style-named policies) is the
--- next iter's work.
local function player_from(opponent, seed)
    if opponent == "random" then
        require_math_rng()
        local rng = alc.math.rng_create(seed)
        return function()
            return duel.policy_player_random(rng)
        end
    end
    error(
        string.format(
            'level: seat="boss" opponent %s is not implemented; the only player policy this '
                .. 'iter carries is "random"',
            type(opponent) == "string" and string.format("%q", opponent) or type(opponent)
        )
    )
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

--- Fights where the Card holds the player seat (pre-seat behaviour).
local function play_player_seat(handle, spec, games, base_seed)
    local boss = boss_from(spec, base_seed)
    local wins = 0.0
    for g = 1, games do
        local state = duel.new_game(base_seed + g)
        while not duel.is_over(state) do
            local boss_action = boss(state.boss)
            local view = duel.player_view(state, "guardian", state.revealed and boss_action or nil)
            local player_action = decide_greedy(handle, view)
            state = duel.apply(state, player_action, boss_action)
        end
        wins = wins + outcome(duel.winner(state), "player")
    end
    return wins
end

--- Fights where the Card holds the boss seat.
local function play_boss_seat(handle, spec, games, base_seed, style)
    local player = player_from(spec, base_seed)
    local wins = 0.0
    for g = 1, games do
        local state = duel.new_game(base_seed + g)
        while not duel.is_over(state) do
            local boss_action = boss_seat.decide(handle, state.boss, style)
            local player_action = player()
            state = duel.apply(state, player_action, boss_action)
        end
        wins = wins + outcome(duel.winner(state), "boss")
    end
    return wins
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

    local handle = resolve_card_handle(card, "card")

    local per_opponent = {}
    local total_wins, total_games = 0.0, 0
    local win_rate_min
    for _, entry in ipairs(pool) do
        local wins
        if seat == "boss" then
            wins = play_boss_seat(handle, entry.spec, games, base_seed, style)
        else
            wins = play_player_seat(handle, entry.spec, games, base_seed)
        end
        local rate = wins / games
        local lo, hi = wilson_ci(wins, games)
        per_opponent[entry.name] = {
            win_rate = rate,
            ci_lower = lo,
            ci_upper = hi,
            n_games = games,
        }
        total_wins = total_wins + wins
        total_games = total_games + games
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
        per_opponent = per_opponent,
    }
end

return M
