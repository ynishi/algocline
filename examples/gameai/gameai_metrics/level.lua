--- level — Card win rate + Wilson 95% CI against a baseline boss opponent
--- over N autoplay games.
---
--- ## Contract
---
--- `level(card, opponent, n_games?, seed?) -> { win_rate, ci_lower, ci_upper }`
---
--- - `card` — string alias, or a live handle (table or userdata) with a
---   `generate_session(ids)` method, same shape as `style_distance`
---   (see `style_distance.lua`, including the `on_ckpt` +
---   `alc.nn.card.load_ckpt` usage). Plays the player seat.
---
--- - `opponent` — describes the boss seat:
---   - `"greedy"` (default alias) — the teacher policy
---     `guardian_duel.policy_guardian`, deterministic. With the Card
---     also deterministic (greedy decode) every fight of a batch would
---     be a copy of one fight, so `level` prints a matching CI when the
---     draw variance is zero.
---   - `"random"` — `guardian_duel.policy_boss_random`, seeded from
---     `seed` so a batch is reproducible.
---   - any other string — treated as a boss Card alias resolved through
---     `guardian_duel_npc` (behind an `alc.card.get_by_alias` /
---     `alc.nn.card.load_handle` pair). This path is the one Level
---     Sweep will use to fight a bake against a prior generation.
---
--- - `n_games` — optional positive integer, default 32.
---
--- - `seed` — optional non-negative integer, default 0. Fights use
---   `guardian_duel.new_game(seed + game_index)` and the random boss RNG
---   is seeded from `seed`.
---
--- ## Output
---
--- ```lua
--- {
---     win_rate = 0.42,
---     ci_lower = 0.30,
---     ci_upper = 0.55,
---     wins     = 13.5,   -- draws count as 0.5, so wins may be fractional
---     n_games  = 32,
--- }
--- ```
---
--- The CI is a 95% Wilson score interval (`z = 1.96`) computed on the
--- integer wins-plus-half-draws count. Wilson is preferred over the
--- normal-approximation interval because it stays inside `[0, 1]` at
--- extreme `p̂` values, which is exactly where a Level Sweep gate would
--- read it.
---
--- ## Why compose here rather than call guardian_player_npc.run
---
--- `guardian_player_npc` reports one summary string per batch, so a
--- caller wanting a CI would have to run one game at a time and parse
--- N summaries. The inner loop below shares the mask/decode contract
--- with `guardian_player_npc.decide` but keeps the per-game outcome as
--- a native Lua value the CI computation reads directly.

local duel = require("guardian_duel")

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

--- Boss chooser factory. Called once per `level` call.
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
        -- of scope for this iter. See ST4 report Follow-up.
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
    return lo, hi
end

M = function(card, opponent, n_games, seed)
    local games = decode_int(n_games, 32, "n_games", true)
    local base_seed = decode_int(seed, 0, "seed", false)

    local handle = resolve_card_handle(card, "card")
    local boss = boss_from(opponent, base_seed)

    local wins = 0.0
    for g = 1, games do
        local state = duel.new_game(base_seed + g)
        while not duel.is_over(state) do
            local boss_action = boss(state.boss)
            local view = duel.player_view(state, "guardian", state.revealed and boss_action or nil)
            local player_action = decide_greedy(handle, view)
            state = duel.apply(state, player_action, boss_action)
        end
        local winner = duel.winner(state)
        if winner == "player" then
            wins = wins + 1.0
        elseif winner == "draw" then
            wins = wins + 0.5
        end
    end

    local lo, hi = wilson_ci(wins, games)
    return {
        win_rate = wins / games,
        ci_lower = lo,
        ci_upper = hi,
        wins = wins,
        n_games = games,
    }
end

return M
