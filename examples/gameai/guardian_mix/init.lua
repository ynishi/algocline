-- Mixed-style boss policy: two teacher styles, one draw per decision.
--
-- `guardian_duel` ships three hand-written boss styles, and a Card baked
-- from any one of them answers that style and nothing else. This module
-- is the labelling function for the boss *between* two of them: on every
-- decision it draws once and answers with one parent or the other, so a
-- corpus labelled by it carries each parent's answer in the proportion
-- the draw was weighted with. What a model fitted on that corpus learns
-- is the mixture itself — the Bayes-optimal answer to a line that
-- appeared with two different labels is the label distribution of that
-- line — rather than either parent.
--
-- Two properties of the rules decide what this can and cannot express:
--
--   * In mode 1 the boss walks `SHIFT_SEQUENCE`, which is a module
--     constant rather than a style field, so *every* style answers the
--     same move there. The mixture is therefore invisible on mode-1
--     states: the draw still happens, and both branches return the same
--     letter. Mixing only bites on the states where the parents
--     disagree, which for `rusher` vs `turtle` is the whole of mode 0
--     bar the shared stagger answer.
--   * The parents are pure functions of the state, so the mixed policy
--     is a function of (state, draw) and the draw is the only thing this
--     module adds.
--
-- Intentional deviation, stated once here so it is not rediscovered at
-- the call site: **the mixed policy is not run through
-- `guardian_duel.compile_policy`**. That validator exists for
-- LLM-written chunks and makes a candidate answer the same state the
-- same way twice, which a mixture by construction does not
-- ("policy is not deterministic"). `guardian_duel.build_corpus` asks for
-- a function and nothing else, so the corpus path is open to it while
-- the persona path is not. The consequence travels with the Card: a
-- mixed boss cannot be handed to `guardian_duel_npc` as a
-- `task.policy_source` teacher, because that field is compiled through
-- the very gate this policy fails. Self-play against a mixed Card is
-- scored against one *parent* at a time (`task.style`) instead.
--
-- Determinism lives one level up: the policy is not deterministic per
-- state, but it is fully determined by `(styles, beta, seed)` plus the
-- order the decisions are asked in. Two runs that ask the same states in
-- the same order under the same seed produce the same labels, which is
-- what makes a corpus reproducible.

local duel = require("guardian_duel")

local M = {}

--- Stride between a caller seed and the stream it opens, the same
--- constant every other seeded stream in this demo uses
--- (`guardian_duel.build_corpus`, `guardian_duel_npc` self-play,
--- `gameai_metrics.audit_matrix`).
local RNG_STRIDE = 7919

--- Offset that keeps the mix stream clear of the per-playout streams a
--- corpus build opens.
---
--- A corpus round opens one stream per fight at `seed * RNG_STRIDE + i`
--- with `i` counting fights, so the mix stream is placed at an `i` no
--- fight batch of this demo reaches. It is opened once per policy and
--- advanced by exactly one draw per decision, so decision *k* reads the
--- *k*-th number of that one stream and no fight's player moves are
--- replayed as mixing draws.
local MIX_SEED_OFFSET = 104729

--- Granularity of the mixing draw.
---
--- The host bridge hands out integers, so a weight is expressed as a
--- cut point on `1..PRECISION`: a thousand steps, i.e. whole tenths of a
--- percent. Finer weights are refused rather than silently rounded, so a
--- caller cannot ask for a mixture it did not get.
local PRECISION = 1000

--- The host RNG bridge, or a loud error naming what is missing.
local function require_rng()
    if type(alc) ~= "table" or type(alc.math) ~= "table" then
        error("guardian_mix: alc.math is required (alc.math.rng_create / alc.math.rng_int)")
    end
    return alc.math
end

--- Validate the parent pair and hand back the two names.
---
--- Exactly two, both known, and different: one style is not a mixture,
--- three parents are a different design (the weight is a single scalar),
--- and the same style twice is a mixture of one thing whose corpus is
--- indistinguishable from the parent's own.
---@param styles table Two `guardian_duel.STYLES` names
---@return string first, string second
local function require_styles(styles)
    if type(styles) ~= "table" then
        error("guardian_mix.mixed_policy: styles must be a table, got " .. type(styles))
    end
    if #styles ~= 2 then
        error(
            string.format(
                "guardian_mix.mixed_policy: styles must name exactly two parents, got %d",
                #styles
            )
        )
    end
    for index = 1, 2 do
        local name = styles[index]
        if type(name) ~= "string" then
            error(
                string.format(
                    "guardian_mix.mixed_policy: styles[%d] must be a string, got %s",
                    index,
                    type(name)
                )
            )
        end
        local known = false
        for _, style in ipairs(duel.STYLES) do
            known = known or style == name
        end
        if not known then
            error(
                string.format(
                    "guardian_mix.mixed_policy: styles[%d] must be one of %s, got %s",
                    index,
                    table.concat(duel.STYLES, ", "),
                    name
                )
            )
        end
    end
    if styles[1] == styles[2] then
        error(
            string.format(
                "guardian_mix.mixed_policy: styles must name two different parents, got %s twice",
                styles[1]
            )
        )
    end
    return styles[1], styles[2]
end

--- Validate the mixing weight and hand back the cut point it names.
---
--- The interval is open on both ends: `0` and `1` are the parents
--- themselves, and asking for a parent through a mixture would bake a
--- Card whose alias claims a mixture that never happened. Weights below
--- the draw's granularity are refused for the same reason.
---@param beta number Weight of the first parent, in `(0, 1)`
---@return integer cut Draws at or below this pick the first parent
local function require_beta(beta)
    if type(beta) ~= "number" or beta ~= beta then
        error(
            "guardian_mix.mixed_policy: beta must be a number in the open interval (0, 1), got "
                .. tostring(beta)
        )
    end
    if beta <= 0 or beta >= 1 then
        error(
            string.format(
                "guardian_mix.mixed_policy: beta must sit in the open interval (0, 1), got %s; "
                    .. "0 and 1 are the parents themselves, which are trained with "
                    .. "train_guardian_npc",
                tostring(beta)
            )
        )
    end
    local cut = math.floor(beta * PRECISION + 0.5)
    if cut < 1 or cut >= PRECISION then
        error(
            string.format(
                "guardian_mix.mixed_policy: beta %s is finer than the draw's granularity of 1/%d "
                    .. "and would round onto a parent",
                tostring(beta),
                PRECISION
            )
        )
    end
    return cut
end

--- Validate the stream seed and hand it back as an integer.
---@param seed number Base seed of the mixing stream
---@return integer seed
local function require_seed(seed)
    if
        type(seed) ~= "number"
        or seed ~= seed
        or seed == math.huge
        or seed == -math.huge
        or seed ~= math.floor(seed)
    then
        error("guardian_mix.mixed_policy: seed must be an integer, got " .. tostring(seed))
    end
    return math.floor(seed)
end

--- Build the labelling policy of a mixed boss.
---
--- The returned function has the shape `guardian_duel.build_corpus`
--- takes — one boss state in, one move letter out — and answers with
--- `styles[1]`'s move on a share `beta` of its decisions and with
--- `styles[2]`'s move on the rest. Both parents are asked the state they
--- were given, so the answer is always a move that parent would have
--- played from that state and is legal by construction.
---
--- The stream is opened here rather than per call, so the sequence of
--- draws belongs to the policy: one `mixed_policy` handle is one
--- reproducible labelling run. A caller that wants the same labels twice
--- builds two handles with the same arguments and asks them the same
--- states in the same order.
---@param styles table Two different `guardian_duel.STYLES` names
---@param beta number Share of decisions answered by `styles[1]`, in `(0, 1)`
---@param seed number Base seed of the mixing stream
---@return fun(state: table): string policy
function M.mixed_policy(styles, beta, seed)
    local first_name, second_name = require_styles(styles)
    local cut = require_beta(beta)
    local base = require_seed(seed)

    local math_ns = require_rng()
    local first = duel["policy_" .. first_name]
    local second = duel["policy_" .. second_name]
    local rng = math_ns.rng_create(base * RNG_STRIDE + MIX_SEED_OFFSET)

    return function(state)
        if math_ns.rng_int(rng, 1, PRECISION) <= cut then
            return first(state)
        end
        return second(state)
    end
end

return M
