-- othello6_mix/spec/othello6_mix_spec.lua
--
-- Package-level spec for the style-mixed Othello teacher. Run with
-- `alc_pkg_test pkg="othello6_mix"` after `alc_pkg_link` has registered
-- `othello6`, `othello6_teacher` and this package, or through the
-- `mlua-probe` runner with `examples/gameai` on the search path:
--
--     test_launch(code_file = "examples/gameai/othello6_mix/spec/othello6_mix_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })
--
-- The module has one entry point and four things worth pinning: that a
-- mixture is reproducible from its arguments, that the share of
-- decisions each parent answers is the share that was asked for, that
-- every answer is a legal move of the position it was asked about, and
-- that everything outside the open interval (0, 1), outside
-- `othello6.STYLES` and outside `othello6_teacher.DEPTHS` is refused
-- before a corpus is built rather than after.
--
-- Equivalence to a parent is asserted literally: an action is "the
-- corner answer" when it equals `othello6_teacher.policy(2, "corner")`'s
-- answer for that same position, never when it merely looks like a move
-- a corner-weighted teacher might play. The two parents search the same
-- rules, so they agree on every forced position and on every position
-- their evaluations order alike; the mixing ratio is therefore measured
-- on the disagreement set and consensus is checked on the agreement set.
--
-- Both parents are asked once per sampled position and the answers are
-- cached, so the measurements below cost one negamax search per mixed
-- decision rather than three.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Seeds `rng_create` was handed, oldest first.
---
--- Recorded rather than ignored because the seed a mixture opens its
--- stream at is part of the contract: it has to stay clear of the
--- per-game streams a corpus build opens (`seed * 7919 + i`).
local RNG_SEEDS = {}

--- Deterministic stand-in for the host RNG bridge.
---
--- The real `alc.math` is ChaCha12; this is a 64-bit LCG read from its
--- high bits, which is enough for the two things the specs below ask of
--- it — reproducibility from a seed, and a flat enough distribution over
--- a thousand buckets for a mixing ratio to be measurable.
alc.math = {
    rng_create = function(seed)
        RNG_SEEDS[#RNG_SEEDS + 1] = seed
        local state = math.floor(seed) & 0x7FFFFFFFFFFFFFFF
        if state == 0 then
            state = 0x9E3779B97F4A7C15
        end
        return { state = state }
    end,
    rng_int = function(rng, min, max)
        rng.state = (rng.state * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFFFFFFFFFF
        return min + (rng.state >> 33) % (max - min + 1)
    end,
}

local othello = require("othello6")
local teacher = require("othello6_teacher")
local mix = require("othello6_mix")

-- ─── Fixtures ───────────────────────────────────────────────────────

--- The two parents every case below mixes. `corner` reads square
--- weights only and `greedy` reads the disc difference only, so the two
--- order the same position differently often enough to measure.
local PAIR = { "corner", "greedy" }

--- The depth both parents search. A member of `othello6_teacher.DEPTHS`,
--- and the shallowest one that still makes the parents look ahead.
local DEPTH = 2

local FIRST = teacher.policy(DEPTH, PAIR[1])
local SECOND = teacher.policy(DEPTH, PAIR[2])

--- Positions from random self-play, terminal ones left out.
---
--- Random play is what reaches the middlegame and endgame shapes a
--- scripted opening never produces, so both the agreement and the
--- disagreement halves come out non-empty without any board being
--- written by hand.
---@param games integer Playouts to walk
---@param seed integer Base seed of the playout streams
---@return table[] states
local function sample_states(games, seed)
    local out = {}
    for game = 1, games do
        local rng = alc.math.rng_create(seed * 7919 + game)
        local random_move = othello.policy_random(rng)
        local state = othello.new_game(seed + game)
        while not othello.is_over(state) do
            out[#out + 1] = state
            state = othello.apply(state, random_move(state))
        end
    end
    return out
end

--- Both parents' answers for one position, asked once and kept.
local ANSWERS = {}

--- Positions split by whether the two parents answer them alike.
local DISAGREE, AGREE = {}, {}
for _, state in ipairs(sample_states(8, 11)) do
    local first, second = FIRST(state), SECOND(state)
    ANSWERS[state] = { first = first, second = second }
    if first == second then
        AGREE[#AGREE + 1] = state
    else
        DISAGREE[#DISAGREE + 1] = state
    end
end

--- Ask one policy for `count` decisions, cycling the position list.
---@param policy fun(state: table): string
---@param states table[] Positions asked in order, repeated
---@param count integer Decisions to take
---@return table answers `{ {state, action}, ... }`
local function decisions(policy, states, count)
    local out = {}
    for index = 1, count do
        local state = states[(index - 1) % #states + 1]
        out[index] = { state = state, action = policy(state) }
    end
    return out
end

--- Share of `answers` that carry each parent's own move.
---
--- The comparison is literal: the action equals what
--- `othello6_teacher.policy(DEPTH, style)` returns for that very
--- position. Both shares are returned so a third answer (a move neither
--- parent would have played) shows up as the two failing to add to one.
---@param answers table `decisions` output
---@return number first_share, number second_share
local function shares(answers)
    local first, second = 0, 0
    for _, answer in ipairs(answers) do
        local expected = ANSWERS[answer.state]
        if answer.action == expected.first then
            first = first + 1
        end
        if answer.action == expected.second then
            second = second + 1
        end
    end
    return first / #answers, second / #answers
end

-- Decisions behind a measured ratio. At a thousand draws the standard
-- error of a share is under 0.016, so the 0.05 band the cases below
-- assert is three standard errors wide and the measurement is about the
-- weight rather than about the draw.
local SAMPLES = 1000
local RATIO_BAND = 0.05

-- Decisions behind a case that only has to see two streams diverge or
-- agree, which no number of samples makes more true.
local SHORT = 200

local BETAS = { 0.25, 0.5, 0.75 }

--- One mixture per weight, measured once and read by several cases:
--- a negamax search per decision is the expensive part of this file.
local MEASURED = {}
for _, beta in ipairs(BETAS) do
    local answers = decisions(mix.mixed_policy(PAIR, beta, 3, DEPTH), DISAGREE, SAMPLES)
    local first, second = shares(answers)
    MEASURED[beta] = { answers = answers, first = first, second = second }
end

describe("othello6_mix fixtures", function()
    it("reaches positions where the parents disagree and positions where they agree", function()
        -- Both halves are load-bearing: the ratio cases read the first
        -- and the consensus cases read the second.
        expect(#DISAGREE > 50).to.equal(true)
        expect(#AGREE > 10).to.equal(true)
    end)

    it("mixes a depth the experiment sweeps", function()
        local known = false
        for _, depth in ipairs(teacher.DEPTHS) do
            known = known or depth == DEPTH
        end
        expect(known).to.equal(true)
    end)
end)

describe("othello6_mix.mixed_policy determinism", function()
    it("answers the same positions the same way under the same seed", function()
        local a = decisions(mix.mixed_policy(PAIR, 0.5, 7, DEPTH), DISAGREE, SHORT)
        local b = decisions(mix.mixed_policy(PAIR, 0.5, 7, DEPTH), DISAGREE, SHORT)
        for index, answer in ipairs(a) do
            expect(b[index].action).to.equal(answer.action)
        end
    end)

    it("answers differently under a different seed", function()
        -- Not a statistical claim: over two hundred decisions two
        -- streams that never diverge would mean the seed is not reaching
        -- the draw at all.
        local a = decisions(mix.mixed_policy(PAIR, 0.5, 7, DEPTH), DISAGREE, SHORT)
        local b = decisions(mix.mixed_policy(PAIR, 0.5, 8, DEPTH), DISAGREE, SHORT)
        local diverged = false
        for index, answer in ipairs(a) do
            diverged = diverged or b[index].action ~= answer.action
        end
        expect(diverged).to.equal(true)
    end)

    it("opens its stream clear of the per-game streams of a corpus", function()
        -- `othello6.build_corpus` opens one stream per game at
        -- `seed * 7919 + i`; the mixture opens one stream at an `i` no
        -- corpus of this demo reaches, and opens it once.
        RNG_SEEDS = {}
        local policy = mix.mixed_policy(PAIR, 0.5, 20260804, DEPTH)
        expect(#RNG_SEEDS).to.equal(1)
        expect(RNG_SEEDS[1]).to.equal(20260804 * 7919 + 104743)
        decisions(policy, DISAGREE, 10)
        expect(#RNG_SEEDS).to.equal(1)
    end)
end)

describe("othello6_mix.mixed_policy mixing ratio", function()
    it("answers with the first parent on the share it was given", function()
        for _, beta in ipairs(BETAS) do
            local measured = MEASURED[beta]
            expect(math.abs(measured.first - beta) < RATIO_BAND).to.equal(true)
            expect(math.abs(measured.second - (1 - beta)) < RATIO_BAND).to.equal(true)
            -- On a disagreement every decision belongs to exactly one
            -- parent, so a third move would break this sum.
            expect(math.abs(measured.first + measured.second - 1.0) < 1e-9).to.equal(true)
        end
    end)

    it("moves the share with the weight rather than with the positions", function()
        -- The same positions, the same seed, three weights: the ordering
        -- is the weight's alone.
        expect(MEASURED[0.25].first < MEASURED[0.5].first).to.equal(true)
        expect(MEASURED[0.5].first < MEASURED[0.75].first).to.equal(true)
    end)

    it("reads the weight as the first named parent's share", function()
        -- Swapping the pair swaps whose share `beta` names, which is the
        -- one thing a caller could get backwards without any error.
        local swapped =
            decisions(mix.mixed_policy({ PAIR[2], PAIR[1] }, 0.25, 5, DEPTH), DISAGREE, SAMPLES)
        local first, second = shares(swapped)
        -- `shares` always names `PAIR[1]` first, so the swapped mixture
        -- gives the *second* returned share to the weight.
        expect(math.abs(second - 0.25) < RATIO_BAND).to.equal(true)
        expect(math.abs(first - 0.75) < RATIO_BAND).to.equal(true)
    end)
end)

describe("othello6_mix.mixed_policy legality", function()
    it("answers a move that is legal in the position it was asked about", function()
        -- Both branches route through a parent that searches the legal
        -- set, so this is a property of the wiring rather than of the
        -- draw; it is asserted because a mixture that reached past its
        -- parents would produce a corpus `othello6.apply` rejects.
        for _, answer in ipairs(MEASURED[0.5].answers) do
            local legal = false
            for _, action in ipairs(othello.legal_actions(answer.state)) do
                legal = legal or action == answer.action
            end
            expect(legal).to.equal(true)
        end
    end)

    it("answers moves a corpus build can apply", function()
        local policy = mix.mixed_policy(PAIR, 0.5, 29, DEPTH)
        local state = othello.new_game(29)
        local plies = 0
        while not othello.is_over(state) and plies < 20 do
            state = othello.apply(state, policy(state))
            plies = plies + 1
        end
        expect(plies > 0).to.equal(true)
        expect(#state.moves).to.equal(plies)
    end)
end)

describe("othello6_mix.mixed_policy consensus", function()
    it("answers the shared move on a position both parents agree on", function()
        -- The draw still happens on these positions; both branches
        -- simply return the same move. A mixture that answered anything
        -- else here would be inventing a move rather than mixing two.
        local policy = mix.mixed_policy(PAIR, 0.5, 13, DEPTH)
        for _, answer in ipairs(decisions(policy, AGREE, SHORT)) do
            expect(answer.action).to.equal(ANSWERS[answer.state].first)
            expect(answer.action).to.equal(ANSWERS[answer.state].second)
        end
    end)

    it("agrees on an agreement whatever the weight is", function()
        local state = AGREE[1]
        local expected = ANSWERS[state].first
        for _, beta in ipairs({ 0.001, 0.5, 0.999 }) do
            local policy = mix.mixed_policy(PAIR, beta, 17, DEPTH)
            for _ = 1, 20 do
                expect(policy(state)).to.equal(expected)
            end
        end
    end)
end)

describe("othello6_mix.mixed_policy rejections", function()
    it("refuses a weight outside the open interval", function()
        for _, beta in ipairs({ 0, 1, -0.25, 1.5, 2 }) do
            expect(function()
                mix.mixed_policy(PAIR, beta, 1, DEPTH)
            end).to.fail()
        end
    end)

    it("refuses a weight that is not a number", function()
        expect(function()
            mix.mixed_policy(PAIR, "0.5", 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, nil, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0 / 0, 1, DEPTH)
        end).to.fail()
    end)

    it("refuses a weight finer than the draw", function()
        -- A thousandth is the granularity; anything under half of one
        -- would round onto a parent and bake a Card whose alias claims a
        -- mixture that never happened.
        expect(function()
            mix.mixed_policy(PAIR, 0.0001, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.9999, 1, DEPTH)
        end).to.fail()
    end)

    it("refuses a parent list that is not a pair", function()
        expect(function()
            mix.mixed_policy({ "corner" }, 0.5, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy({ "corner", "greedy", "mobility" }, 0.5, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy({}, 0.5, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy("corner", 0.5, 1, DEPTH)
        end).to.fail()
    end)

    it("refuses an unknown parent", function()
        expect(function()
            mix.mixed_policy({ "corner", "aggressive" }, 0.5, 1, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy({ "corner", 3 }, 0.5, 1, DEPTH)
        end).to.fail()
    end)

    it("refuses the same parent twice", function()
        -- A mixture of one thing is that thing, and its alias would
        -- claim otherwise.
        expect(function()
            mix.mixed_policy({ "greedy", "greedy" }, 0.5, 1, DEPTH)
        end).to.fail()
    end)

    it("refuses a seed that is not an integer", function()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, nil, DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, "20260804", DEPTH)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, 1.5, DEPTH)
        end).to.fail()
    end)

    it("refuses a depth the experiment does not sweep", function()
        -- `othello6_teacher.policy` takes any positive integer; the
        -- mixture is narrower on purpose, because a mixed Card is only
        -- comparable against single-style Cards baked at a swept depth.
        for _, depth in ipairs({ 3, 5, 0, -2, 2.5 }) do
            expect(function()
                mix.mixed_policy(PAIR, 0.5, 1, depth)
            end).to.fail()
        end
        expect(function()
            mix.mixed_policy(PAIR, 0.5, 1, nil)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, 1, "2")
        end).to.fail()
    end)

    it("accepts every depth the experiment does sweep", function()
        for _, depth in ipairs(teacher.DEPTHS) do
            expect(type(mix.mixed_policy(PAIR, 0.5, 1, depth))).to.equal("function")
        end
    end)
end)
