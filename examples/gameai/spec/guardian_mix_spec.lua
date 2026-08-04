-- spec/guardian_mix_spec.lua
--
-- Two arms, one file.
--
-- The first arm is `guardian_mix.mixed_policy`, the labelling function
-- of a boss between two teacher styles. It is a small module with one
-- entry point and three things worth pinning: that a mixture is
-- reproducible from its arguments, that the share of decisions each
-- parent answers is the share that was asked for, and that everything
-- outside the open interval (0, 1) and outside `guardian_duel.STYLES` is
-- refused before a corpus is built rather than after.
--
-- Equivalence to a parent is asserted literally throughout: an action is
-- "the rusher answer" when it equals `guardian_duel.policy_rusher(state)`
-- for that same state, never when it merely looks like a move a rusher
-- might play. The two parents disagree on most of mode 0 and agree on
-- the whole of mode 1 (the defensive sequence is a module constant, not
-- a style field), so the mixing ratio is measured on states where they
-- disagree and consensus is checked on states where they do not.
--
-- The second arm is the `train_guardian_mix.lua` driver, exercised the
-- way `spec/train_guardian_npc_spec.lua` exercises its own script: the
-- whole file is loaded once per case against stubs, and the assertions
-- read the ctx handling — the required weight, the default pair, the
-- alias the mixture is pinned under and the two self-play calls that
-- replace the single-teacher compliance line. Everything expensive is a
-- stub, so no `nn` feature and no training budget is involved.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/guardian_mix_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Seeds `rng_create` was handed, oldest first.
---
--- Recorded rather than ignored because the seed a mixture opens its
--- stream at is part of the contract: it has to stay clear of the
--- per-fight streams a corpus build opens (`seed * 7919 + i`).
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

local duel = require("guardian_duel")
local mix = require("guardian_mix")

-- ─── Fixtures ───────────────────────────────────────────────────────

--- The two parents every case below mixes, and the pair the driver
--- defaults to: their cycles differ at every index, so mode 0 is almost
--- entirely disagreement.
local PAIR = { "rusher", "turtle" }

--- Boss states from random self-play, split by whether the two parents
--- answer them the same way.
---
--- Random play is what reaches the mode / cycle / damage combinations a
--- scripted fight never produces, so both halves come out non-empty
--- without any state being written by hand.
local function split_states(games, seed)
    local disagree, agree = {}, {}
    for _, state in ipairs(duel.sample_states({ games = games, seed = seed })) do
        if duel.policy_rusher(state) == duel.policy_turtle(state) then
            agree[#agree + 1] = state
        else
            disagree[#disagree + 1] = state
        end
    end
    return disagree, agree
end

local DISAGREE, AGREE = split_states(40, 11)

--- Ask one policy for `count` decisions, cycling the state list.
---@param policy fun(state: table): string
---@param states table[] States asked in order, repeated
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

--- Share of `answers` that carry the first parent's own move.
---
--- The comparison is literal: the action equals what
--- `guardian_duel.policy_<style>` returns for that very state. Both
--- shares are returned so a third answer (a move neither parent would
--- have played) shows up as the two failing to add to one.
---@param answers table `decisions` output
---@param styles table Parent pair
---@return number first_share, number second_share
local function shares(answers, styles)
    local first, second = 0, 0
    for _, answer in ipairs(answers) do
        if answer.action == duel["policy_" .. styles[1]](answer.state) then
            first = first + 1
        end
        if answer.action == duel["policy_" .. styles[2]](answer.state) then
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

--- Share of one mixture's decisions that carry the first parent's move,
--- measured over `SAMPLES` decisions on the disagreement states.
---@param styles table Parent pair
---@param beta number Weight of the first parent
---@param seed number Mixing seed
---@return number share
local function first_share(styles, beta, seed)
    local policy = mix.mixed_policy(styles, beta, seed)
    local first = shares(decisions(policy, DISAGREE, SAMPLES), styles)
    return first
end

describe("guardian_mix.mixed_policy fixtures", function()
    it("reaches states where the parents disagree and states where they agree", function()
        -- Both halves are load-bearing: the ratio cases read the first
        -- and the consensus case reads the second.
        expect(#DISAGREE > 100).to.equal(true)
        expect(#AGREE > 10).to.equal(true)
    end)

    it("finds every agreement inside the shared defensive sequence", function()
        -- Mode 1 walks `SHIFT_SEQUENCE`, a module constant rather than a
        -- style field, so no style can differ there. In mode 0 the
        -- parents share only the stagger answer, which their thresholds
        -- put at different moments.
        local mode1 = 0
        for _, state in ipairs(AGREE) do
            if state.mode == 1 then
                mode1 = mode1 + 1
            end
        end
        expect(mode1 > 0).to.equal(true)
    end)
end)

describe("guardian_mix.mixed_policy determinism", function()
    it("answers the same states the same way under the same seed", function()
        local a = decisions(mix.mixed_policy(PAIR, 0.5, 7), DISAGREE, SAMPLES)
        local b = decisions(mix.mixed_policy(PAIR, 0.5, 7), DISAGREE, SAMPLES)
        for index, answer in ipairs(a) do
            expect(b[index].action).to.equal(answer.action)
        end
    end)

    it("answers differently under a different seed", function()
        -- Not a statistical claim: over a thousand decisions two streams
        -- that never diverge would mean the seed is not reaching the
        -- draw at all.
        local a = decisions(mix.mixed_policy(PAIR, 0.5, 7), DISAGREE, SAMPLES)
        local b = decisions(mix.mixed_policy(PAIR, 0.5, 8), DISAGREE, SAMPLES)
        local diverged = false
        for index, answer in ipairs(a) do
            diverged = diverged or b[index].action ~= answer.action
        end
        expect(diverged).to.equal(true)
    end)

    it("opens its stream clear of the per-fight streams of a corpus", function()
        -- `guardian_duel.build_corpus` opens one stream per fight at
        -- `seed * 7919 + i`; the mixture opens one stream at an `i` no
        -- fight batch of this demo reaches, and opens it once.
        RNG_SEEDS = {}
        local policy = mix.mixed_policy(PAIR, 0.5, 20260731)
        expect(#RNG_SEEDS).to.equal(1)
        expect(RNG_SEEDS[1]).to.equal(20260731 * 7919 + 104729)
        decisions(policy, DISAGREE, 10)
        expect(#RNG_SEEDS).to.equal(1)
    end)
end)

describe("guardian_mix.mixed_policy mixing ratio", function()
    it("answers with the first parent on the share it was given", function()
        for _, beta in ipairs({ 0.25, 0.5, 0.75 }) do
            local answers = decisions(mix.mixed_policy(PAIR, beta, 3), DISAGREE, SAMPLES)
            local first, second = shares(answers, PAIR)
            expect(math.abs(first - beta) < RATIO_BAND).to.equal(true)
            expect(math.abs(second - (1 - beta)) < RATIO_BAND).to.equal(true)
            -- On a disagreement every decision belongs to exactly one
            -- parent, so a third move would break this sum.
            expect(math.abs(first + second - 1.0) < 1e-9).to.equal(true)
        end
    end)

    it("moves the share with the weight rather than with the states", function()
        -- The same states, the same seed, three weights: the ordering is
        -- the weight's alone.
        local low = first_share(PAIR, 0.25, 3)
        local mid = first_share(PAIR, 0.5, 3)
        local high = first_share(PAIR, 0.75, 3)
        expect(low < mid).to.equal(true)
        expect(mid < high).to.equal(true)
    end)

    it("reads the weight as the first named parent's share", function()
        -- Swapping the pair swaps whose share `beta` names, which is the
        -- one thing a caller could get backwards without any error.
        local forward = first_share(PAIR, 0.25, 5)
        local swapped = first_share({ PAIR[2], PAIR[1] }, 0.25, 5)
        expect(math.abs(forward - 0.25) < RATIO_BAND).to.equal(true)
        expect(math.abs(swapped - 0.25) < RATIO_BAND).to.equal(true)
    end)
end)

describe("guardian_mix.mixed_policy consensus", function()
    it("answers the shared move on a state both parents agree on", function()
        -- The draw still happens on these states; both branches simply
        -- return the same letter. A mixture that answered anything else
        -- here would be inventing a move rather than mixing two.
        local policy = mix.mixed_policy(PAIR, 0.5, 13)
        local answers = decisions(policy, AGREE, SAMPLES)
        for _, answer in ipairs(answers) do
            expect(answer.action).to.equal(duel.policy_rusher(answer.state))
            expect(answer.action).to.equal(duel.policy_turtle(answer.state))
        end
    end)

    it("agrees on an agreement whatever the weight is", function()
        local state = AGREE[1]
        local expected = duel.policy_rusher(state)
        for _, beta in ipairs({ 0.001, 0.5, 0.999 }) do
            local policy = mix.mixed_policy(PAIR, beta, 17)
            for _ = 1, 50 do
                expect(policy(state)).to.equal(expected)
            end
        end
    end)
end)

describe("guardian_mix.mixed_policy rejections", function()
    it("refuses a weight outside the open interval", function()
        for _, beta in ipairs({ 0, 1, -0.25, 1.5, 2 }) do
            expect(function()
                mix.mixed_policy(PAIR, beta, 1)
            end).to.fail()
        end
    end)

    it("refuses a weight that is not a number", function()
        expect(function()
            mix.mixed_policy(PAIR, "0.5", 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, nil, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0 / 0, 1)
        end).to.fail()
    end)

    it("refuses a weight finer than the draw", function()
        -- A thousandth is the granularity; anything under half of one
        -- would round onto a parent and bake a Card whose alias claims a
        -- mixture that never happened.
        expect(function()
            mix.mixed_policy(PAIR, 0.0001, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.9999, 1)
        end).to.fail()
    end)

    it("refuses a parent list that is not a pair", function()
        expect(function()
            mix.mixed_policy({ "rusher" }, 0.5, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy({ "rusher", "turtle", "guardian" }, 0.5, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy({}, 0.5, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy("rusher", 0.5, 1)
        end).to.fail()
    end)

    it("refuses an unknown parent", function()
        expect(function()
            mix.mixed_policy({ "rusher", "berserker" }, 0.5, 1)
        end).to.fail()
        expect(function()
            mix.mixed_policy({ "rusher", 3 }, 0.5, 1)
        end).to.fail()
    end)

    it("refuses the same parent twice", function()
        -- A mixture of one thing is that thing, and its alias would
        -- claim otherwise.
        expect(function()
            mix.mixed_policy({ "turtle", "turtle" }, 0.5, 1)
        end).to.fail()
    end)

    it("refuses a seed that is not an integer", function()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, nil)
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, "20260731")
        end).to.fail()
        expect(function()
            mix.mixed_policy(PAIR, 0.5, 1.5)
        end).to.fail()
    end)
end)

-- ─── Driver arm: train_guardian_mix.lua ─────────────────────────────

--- Minimal JSON encoder standing in for the host `alc.json_encode`.
---
--- Object keys are emitted in sorted order so a case can match a fixed
--- substring against the task the driver handed the NPC package.
local function json_encode(value)
    local kind = type(value)
    if value == nil then
        return "null"
    end
    if kind == "number" or kind == "boolean" then
        return tostring(value)
    end
    if kind == "string" then
        return string.format("%q", value)
    end
    if kind ~= "table" then
        error("spec json_encode: unsupported type " .. kind, 0)
    end
    if #value > 0 then
        local items = {}
        for index = 1, #value do
            items[index] = json_encode(value[index])
        end
        return "[" .. table.concat(items, ",") .. "]"
    end
    local keys = {}
    for key in pairs(value) do
        keys[#keys + 1] = key
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)
    local fields = {}
    for _, key in ipairs(keys) do
        fields[#fields + 1] = string.format("%q", tostring(key)) .. ":" .. json_encode(value[key])
    end
    return "{" .. table.concat(fields, ",") .. "}"
end

alc.json_encode = json_encode
alc.log = function() end

--- alias_set invocations of the last drive, in fire order.
local ALIAS_SET_CALLS = {}

alc.card = {
    alias_set = function(alias, card_id, opts)
        ALIAS_SET_CALLS[#ALIAS_SET_CALLS + 1] = { alias = alias, card_id = card_id, opts = opts }
    end,
    get = function()
        -- Below `math.log(model_vocab)`, so the driver's "gradients
        -- flowed" line reports on the wiring rather than on the stub.
        return { metadata = { nn = { metrics = { train_loss = 0.1 } } } }
    end,
}

--- Opts the driver handed to `run_full_ft` on the last drive.
local TRAIN_OPTS = nil

alc.nn = {
    preset = {
        gpt2 = function()
            return {
                ctx = function()
                    return duel.CTX_BUDGET
                end,
                vocab = function()
                    return 64
                end,
            }
        end,
    },
    data = {
        synthetic = function(rows)
            return { rows = #rows }
        end,
    },
    trainer = {
        run_full_ft = function(_, _, opts)
            TRAIN_OPTS = opts
            return "card-stub-0001"
        end,
    },
}

--- Self-play requests of the last drive, in fire order. Each entry is
--- the whole opts table the driver handed `npc.run`, so a case can read
--- both the task (which teacher the moves are scored against) and the
--- decode basis the Card is played under.
local NPC_CALLS = {}

package.preload["guardian_duel_npc"] = function()
    return {
        reset_cache = function() end,
        run = function(opts)
            NPC_CALLS[#NPC_CALLS + 1] = opts
            local task = tostring(opts.task)
            -- Two different numbers, so a case that reads them back
            -- cannot pass on a driver that reported one twice.
            local match = task:find('"style":"turtle"', 1, true) and "0.60" or "0.30"
            return {
                result = string.format(
                    "winrate=0.50 illegal=0 style_match=%s style_hits=3/6",
                    match
                ),
            }
        end,
    }
end

--- Sentinel that removes a ctx field instead of overriding it.
local NONE = {}

--- Load the driver once against the current stubs.
---
--- `ctx` is a global in the `alc_run` contract, so it is planted as one
--- here. The defaults are the smallest run that still builds a corpus
--- and reaches the trainer: the training itself is a stub.
---@param overrides table ctx fields for this case
---@return table result the driver's return value
local function drive(overrides)
    ALIAS_SET_CALLS = {}
    NPC_CALLS = {}
    TRAIN_OPTS = nil
    local script_ctx = {
        beta = 0.25,
        games = 1,
        steps = 2,
        batch = 1,
        check_games = 1,
    }
    for key, value in pairs(overrides or {}) do
        if value == NONE then
            script_ctx[key] = nil
        else
            script_ctx[key] = value
        end
    end
    ctx = script_ctx
    package.loaded["train_guardian_mix"] = nil
    return require("train_guardian_mix")
end

describe("train_guardian_mix ctx", function()
    it("requires a mixing weight", function()
        expect(function()
            drive({ beta = NONE })
        end).to.fail()
    end)

    it("refuses a weight the mixture itself would refuse", function()
        -- The interval check lives in `guardian_mix`; what this pins is
        -- that the driver runs it before spending a training budget.
        expect(function()
            drive({ beta = 0 })
        end).to.fail()
        expect(function()
            drive({ beta = 1 })
        end).to.fail()
    end)

    it("defaults to the rusher/turtle pair", function()
        local out = drive({ beta = 0.25 })
        expect(out.style_a).to.equal("rusher")
        expect(out.style_b).to.equal("turtle")
        expect(out.beta).to.equal(0.25)
        expect(out.basis_style).to.equal("guardian")
    end)

    it("names the alias after the pair and the weight", function()
        local out = drive({ beta = 0.25 })
        expect(out.alias).to.equal("guardian_duel_npc_mix_rt_b25")
        expect(#ALIAS_SET_CALLS).to.equal(1)
        expect(ALIAS_SET_CALLS[1].alias).to.equal("guardian_duel_npc_mix_rt_b25")
        expect(ALIAS_SET_CALLS[1].card_id).to.equal("card-stub-0001")

        local other = drive({ beta = 0.5, styles = { "guardian", "turtle" } })
        expect(other.alias).to.equal("guardian_duel_npc_mix_gt_b50")
    end)

    it("refuses a weight that would round two mixtures onto one alias", function()
        expect(function()
            drive({ beta = 0.125 })
        end).to.fail()
        -- An explicit alias is the way to ask for such a weight.
        local out = drive({ beta = 0.125, alias = "guardian_duel_npc_mix_rt_b125" })
        expect(out.alias).to.equal("guardian_duel_npc_mix_rt_b125")
    end)

    it("keeps the distance basis out of the ctx", function()
        expect(function()
            drive({ beta = 0.25, basis_style = "turtle" })
        end).to.fail()
        expect(function()
            drive({ beta = 0.25, basis = "turtle" })
        end).to.fail()
    end)

    it("scores the Card against each parent from the one decode basis", function()
        local out = drive({ beta = 0.25 })
        expect(#NPC_CALLS).to.equal(2)
        -- Same Card, same decode basis, one teacher each: only
        -- `task.style` moves between the two runs.
        for _, call in ipairs(NPC_CALLS) do
            expect(call.card_alias).to.equal("guardian_duel_npc_mix_rt_b25")
            expect(call.style).to.equal("guardian")
        end
        expect(NPC_CALLS[1].task:find('"style":"rusher"', 1, true) ~= nil).to.equal(true)
        expect(NPC_CALLS[2].task:find('"style":"turtle"', 1, true) ~= nil).to.equal(true)
        expect(out.match_a).to.equal(0.30)
        expect(out.match_b).to.equal(0.60)
        -- Summed in floating point, so the sum is compared as one.
        expect(math.abs(out.match_sum - 0.90) < 1e-9).to.equal(true)
    end)

    it("hands the training budget through unchanged", function()
        local out = drive({ beta = 0.75, steps = 3, batch = 2, lr = 1e-3 })
        expect(TRAIN_OPTS.steps).to.equal(3)
        expect(TRAIN_OPTS.batch).to.equal(2)
        expect(TRAIN_OPTS.lr).to.equal(1e-3)
        expect(TRAIN_OPTS.schedule).to.equal("Constant")
        expect(out.card_id).to.equal("card-stub-0001")
        expect(out.rows_target).to.equal(3 * 2 + 2)
        expect(out.rows >= out.rows_target).to.equal(true)
        expect(out.loss_descended).to.equal(true)
        expect(out.ok).to.equal(true)
    end)
end)
