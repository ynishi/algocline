--- othello6_mix — style-mixed 6x6 Othello teacher, one draw per decision.
---
--- `othello6_teacher` ships three evaluation functions, and a Card baked
--- from any one of them answers that style and nothing else. This module
--- is the labelling function for the teacher *between* two of them: at
--- the same search depth, on every decision it draws once and answers
--- with one parent or the other, so a corpus labelled by it carries each
--- parent's move in the proportion the draw was weighted with. What a
--- model fitted on that corpus learns is the mixture itself — the
--- Bayes-optimal answer to a line that appeared with two different
--- labels is the label distribution of that line — rather than either
--- parent.
---
--- ## What the axis is
---
--- Depth is held fixed and style is mixed. The two dials of
--- `othello6_teacher.policy(depth, style)` are not interchangeable here:
--- depth is a strength axis, and mixing two depths would produce a
--- teacher that is sometimes stronger and sometimes weaker rather than
--- one that plays somewhere between two tastes. `depth` is therefore a
--- required argument that both parents share, and the only thing the
--- draw picks is which evaluation function answers.
---
--- ## Where the mixture is visible
---
--- Only on positions where the two parents disagree. Both parents search
--- the same rules from the same position, so they answer the same move
--- whenever the position is forced — a single legal placement, or a pass
--- — and on every position their evaluations happen to order alike. The
--- draw still happens there; both branches simply return the same move.
--- A measurement of the mixing ratio has to be taken on the
--- disagreement set, which is what `spec/othello6_mix_spec.lua` does.
---
--- ## Legality
---
--- Both parents are asked the position they were given and each answers
--- from `othello6.legal_actions` of that position, so whichever branch
--- the draw takes the move is legal by construction. Nothing here has to
--- re-check the rules.
---
--- Intentional deviation, stated once here so it is not rediscovered at
--- the call site: **a mixed policy is not deterministic per position**.
--- It cannot be handed to a validator that asks a candidate to answer
--- the same position the same way twice, which a mixture by construction
--- does not. `othello6.build_corpus` asks for a function and nothing
--- else, so the corpus path is open to it. Agreement with a teacher is
--- scored against one *parent* at a time instead.
---
--- Determinism lives one level up: the policy is not deterministic per
--- position, but it is fully determined by `(styles, beta, seed, depth)`
--- plus the order the decisions are asked in. Two runs that ask the same
--- positions in the same order under the same seed produce the same
--- labels, which is what makes a corpus reproducible.

local othello = require("othello6")
local teacher = require("othello6_teacher")

local M = {}

---@type AlcMeta
M.meta = {
    name = "othello6_mix",
    version = "0.1.0",
    description = "Style-mixed 6x6 Othello teacher: two evaluation functions at one depth, one draw per decision",
    category = "game",
}

--- Stride between a caller seed and the stream it opens, the same
--- constant every other seeded stream in this demo uses
--- (`othello6.build_corpus`, `othello6_npc` self-play).
local RNG_STRIDE = 7919

--- Offset that keeps the mix stream clear of the per-game streams a
--- corpus build opens.
---
--- A corpus round opens one stream per game at `seed * RNG_STRIDE + i`
--- with `i` counting games, so the mix stream is placed at an `i` no
--- corpus of this demo reaches: the runs this module was written for
--- build a few thousand games, four orders of magnitude below this
--- offset. It is opened once per policy and advanced by exactly one draw
--- per decision, so decision *k* reads the *k*-th number of that one
--- stream and no game's random opening is replayed as a mixing draw.
---
--- It is also one prime clear of `guardian_mix.MIX_SEED_OFFSET`
--- (104729), so the two mixers of this demo never share a stream even
--- when they are given the same seed.
local MIX_SEED_OFFSET = 104743

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
        error("othello6_mix: alc.math is required (alc.math.rng_create / alc.math.rng_int)")
    end
    return alc.math
end

--- Validate the parent pair and hand back the two names.
---
--- Exactly two, both known, and different: one style is not a mixture,
--- three parents are a different design (the weight is a single scalar),
--- and the same style twice is a mixture of one thing whose corpus is
--- indistinguishable from the parent's own.
---@param styles table Two `othello6.STYLES` names
---@return string first, string second
local function require_styles(styles)
    if type(styles) ~= "table" then
        error("othello6_mix.mixed_policy: styles must be a table, got " .. type(styles))
    end
    if #styles ~= 2 then
        error(
            string.format(
                "othello6_mix.mixed_policy: styles must name exactly two parents, got %d",
                #styles
            )
        )
    end
    for index = 1, 2 do
        local name = styles[index]
        if type(name) ~= "string" then
            error(
                string.format(
                    "othello6_mix.mixed_policy: styles[%d] must be a string, got %s",
                    index,
                    type(name)
                )
            )
        end
        local known = false
        for _, style in ipairs(othello.STYLES) do
            known = known or style == name
        end
        if not known then
            error(
                string.format(
                    "othello6_mix.mixed_policy: styles[%d] must be one of %s, got %s",
                    index,
                    table.concat(othello.STYLES, ", "),
                    name
                )
            )
        end
    end
    if styles[1] == styles[2] then
        error(
            string.format(
                "othello6_mix.mixed_policy: styles must name two different parents, got %s twice",
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
            "othello6_mix.mixed_policy: beta must be a number in the open interval (0, 1), got "
                .. tostring(beta)
        )
    end
    if beta <= 0 or beta >= 1 then
        error(
            string.format(
                "othello6_mix.mixed_policy: beta must sit in the open interval (0, 1), got %s; "
                    .. "0 and 1 are the parents themselves, which are labelled with "
                    .. "othello6_teacher.policy",
                tostring(beta)
            )
        )
    end
    local cut = math.floor(beta * PRECISION + 0.5)
    if cut < 1 or cut >= PRECISION then
        error(
            string.format(
                "othello6_mix.mixed_policy: beta %s is finer than the draw's granularity of 1/%d "
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
        error("othello6_mix.mixed_policy: seed must be an integer, got " .. tostring(seed))
    end
    return math.floor(seed)
end

--- Validate the shared search depth against the swept set.
---
--- `othello6_teacher.policy` accepts any positive integer, but the
--- experiment this module feeds sweeps `othello6_teacher.DEPTHS` and
--- names its Cards after a member of it. A depth outside that set would
--- produce a mixture no single-style Card can be compared against, so it
--- is refused here rather than discovered at comparison time.
---@param depth number Plies both parents search
---@return integer depth
local function require_depth(depth)
    if type(depth) == "number" then
        for _, known in ipairs(teacher.DEPTHS) do
            if depth == known then
                return known
            end
        end
    end
    local names = {}
    for index, known in ipairs(teacher.DEPTHS) do
        names[index] = tostring(known)
    end
    error(
        string.format(
            "othello6_mix.mixed_policy: depth must be one of %s, got %s",
            table.concat(names, ", "),
            tostring(depth)
        )
    )
end

--- Build the labelling policy of a style-mixed teacher.
---
--- The returned function has the shape `othello6.build_corpus` takes —
--- one position in, one move character out — and answers with
--- `styles[1]`'s move on a share `beta` of its decisions and with
--- `styles[2]`'s move on the rest. Both parents search the position they
--- were given at the same `depth`, so the answer is always a move that
--- parent would have played from that position and is legal by
--- construction.
---
--- The stream is opened here rather than per call, so the sequence of
--- draws belongs to the policy: one `mixed_policy` handle is one
--- reproducible labelling run. A caller that wants the same labels twice
--- builds two handles with the same arguments and asks them the same
--- positions in the same order.
---@param styles table Two different `othello6.STYLES` names
---@param beta number Share of decisions answered by `styles[1]`, in `(0, 1)`
---@param seed number Base seed of the mixing stream
---@param depth number Plies both parents search, one of `othello6_teacher.DEPTHS`
---@return fun(state: table): string policy
function M.mixed_policy(styles, beta, seed, depth)
    local first_name, second_name = require_styles(styles)
    local cut = require_beta(beta)
    local base = require_seed(seed)
    local plies = require_depth(depth)

    local math_ns = require_rng()
    local first = teacher.policy(plies, first_name)
    local second = teacher.policy(plies, second_name)
    local rng = math_ns.rng_create(base * RNG_STRIDE + MIX_SEED_OFFSET)

    return function(state)
        if math_ns.rng_int(rng, 1, PRECISION) <= cut then
            return first(state)
        end
        return second(state)
    end
end

return M
