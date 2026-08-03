-- gameai_metrics/spec/registry_adapter_spec.lua
--
-- Spec for the registry adapter in `gameai_metrics/init.lua`: the
-- `self_register()` block that wraps each metric module into an
-- `fn(ctx)` and lifts the shared seat options out of the ctx
-- (`seat_opts`).
--
-- Every other spec in this directory measures a metric module or a
-- runner. The adapter itself sits between them and had no coverage:
-- a ctx key that `seat_opts` forgets to carry does not raise anywhere,
-- it simply arrives as `nil` at the metric, which then takes a
-- perfectly valid default path. `temperature` is the sharp case — a
-- dropped key turns a temperature-scaled measurement into a greedy one
-- while the caller's own record still says the temperature was asked
-- for, so a decode-effect comparison reads a real difference as zero.
--
-- The three metric modules are replaced with recording fakes *before*
-- `gameai_metrics` is required, the way `fight_matrix_spec.lua` swaps
-- `gameai_metrics.level`: `init.lua` binds each module at require time,
-- so a fake installed afterwards would never be seen. The registry is
-- a stub that keeps the registered functions instead of dispatching
-- them, which is what lets a case call one directly.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/gameai_metrics/spec/registry_adapter_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Registry stub ──────────────────────────────────────────────────

alc = alc or {}
alc.nn = alc.nn or {}
alc.nn.metric = alc.nn.metric or {}

--- name -> fn(ctx), captured from `register` at require time.
local REGISTERED = {}

alc.nn.metric.registry = {
    register = function(name, fn)
        REGISTERED[name] = fn
    end,
    evaluate = function(name)
        -- The adapter never evaluates through the registry; the specs
        -- call the registered fn directly so a failing assertion is not
        -- routed through a dispatch path that could swallow it.
        error("registry_adapter_spec: unexpected evaluate('" .. tostring(name) .. "')", 0)
    end,
}

-- ─── Metric module fakes ────────────────────────────────────────────

local LEVEL_CALLS = {}
local TRICKY_CALLS = {}
local SD_CALLS = {}

--- Same positional contract as `gameai_metrics.level`:
--- `(card, opponent, n_games, seed, opts)`.
package.loaded["gameai_metrics.level"] = function(card, opponent, n_games, seed, opts)
    LEVEL_CALLS[#LEVEL_CALLS + 1] = {
        card = card,
        opponent = opponent,
        n_games = n_games,
        seed = seed,
        opts = opts or {},
    }
    return { win_rate = 0.5, ci_lower = 0.4, ci_upper = 0.6 }
end

--- Same positional contract as `gameai_metrics.trickiness`:
--- `(card, prompt_set, temperature, opts)`.
package.loaded["gameai_metrics.trickiness"] = function(card, prompt_set, temperature, opts)
    TRICKY_CALLS[#TRICKY_CALLS + 1] = {
        card = card,
        prompt_set = prompt_set,
        temperature = temperature,
        opts = opts or {},
    }
    return 0.25
end

--- Same positional contract as `gameai_metrics.style_distance`:
--- `(card_a, card_b, prompt_set, opts)`.
package.loaded["gameai_metrics.style_distance"] = function(card_a, card_b, prompt_set, opts)
    SD_CALLS[#SD_CALLS + 1] = {
        card_a = card_a,
        card_b = card_b,
        prompt_set = prompt_set,
        opts = opts or {},
    }
    return 0.1
end

-- Force `init.lua` to run against the fakes above even when an earlier
-- require in the same VM already loaded the real package.
package.loaded["gameai_metrics"] = nil
require("gameai_metrics")

local function reset_calls()
    LEVEL_CALLS = {}
    TRICKY_CALLS = {}
    SD_CALLS = {}
end

local CARD = { _card = "under-measurement" }
local PROMPT_SET = { { cycle = 0, mode = 0 }, { cycle = 1, mode = 0 } }

-- ─── Specs: registration ────────────────────────────────────────────

describe("gameai_metrics registry adapter — registration", function()
    it("registers level / trickiness / style_distance as functions", function()
        expect(type(REGISTERED.level)).to.equal("function")
        expect(type(REGISTERED.trickiness)).to.equal("function")
        expect(type(REGISTERED.style_distance)).to.equal("function")
    end)
end)

-- ─── Specs: level ctx -> opts transport ─────────────────────────────

describe("gameai_metrics registry adapter — level", function()
    it("carries ctx.temperature through seat_opts into the level opts", function()
        reset_calls()
        REGISTERED.level({
            card = CARD,
            seat = "boss",
            style = "guardian",
            opponents = { "random" },
            temperature = 0.7,
            n_games = 1,
            seed = 0,
        })
        expect(#LEVEL_CALLS).to.equal(1)
        local call = LEVEL_CALLS[1]
        -- The transport under test: without this line in `seat_opts`
        -- the audit's level view would silently decode greedily.
        expect(call.opts.temperature).to.equal(0.7)
        expect(call.opts.seat).to.equal("boss")
        expect(call.opts.style).to.equal("guardian")
        expect(#call.opts.opponents).to.equal(1)
        expect(call.opts.opponents[1]).to.equal("random")
        expect(call.card).to.equal(CARD)
        expect(call.n_games).to.equal(1)
        expect(call.seed).to.equal(0)
    end)

    it("leaves opts.temperature absent when the ctx carries none", function()
        reset_calls()
        REGISTERED.level({
            card = CARD,
            seat = "boss",
            style = "guardian",
            opponents = { "random" },
            n_games = 4,
            seed = 3,
        })
        expect(#LEVEL_CALLS).to.equal(1)
        -- Greedy is the absence of the key all the way down; the
        -- adapter must not invent a default on the way.
        expect(LEVEL_CALLS[1].opts.temperature).to.equal(nil)
    end)

    it("carries ctx.opponent_style and ctx.opponent through", function()
        reset_calls()
        REGISTERED.level({
            card = CARD,
            seat = "boss",
            style = "guardian",
            opponent_style = "sentinel",
            opponent = "random",
            opponents = { "random" },
        })
        expect(#LEVEL_CALLS).to.equal(1)
        expect(LEVEL_CALLS[1].opts.opponent_style).to.equal("sentinel")
        expect(LEVEL_CALLS[1].opponent).to.equal("random")
    end)

    it("defaults n_games to 32 and seed to 0 when the ctx omits them", function()
        reset_calls()
        REGISTERED.level({ card = CARD, seat = "boss", style = "guardian" })
        expect(#LEVEL_CALLS).to.equal(1)
        expect(LEVEL_CALLS[1].n_games).to.equal(32)
        expect(LEVEL_CALLS[1].seed).to.equal(0)
    end)
end)

-- ─── Specs: trickiness ctx -> temperature argument ──────────────────

describe("gameai_metrics registry adapter — trickiness", function()
    it("defaults the temperature argument to 1.0 when the ctx omits it", function()
        reset_calls()
        REGISTERED.trickiness({
            card = CARD,
            prompt_set = PROMPT_SET,
            seat = "boss",
            style = "guardian",
        })
        expect(#TRICKY_CALLS).to.equal(1)
        -- This is why the audit keeps `temperature` off the trickiness
        -- view: the adapter already pins the entropy scale at 1.0, the
        -- same scale a fight decodes on.
        expect(TRICKY_CALLS[1].temperature).to.equal(1.0)
        expect(TRICKY_CALLS[1].opts.seat).to.equal("boss")
    end)

    it("uses ctx.temperature for the entropy scale when one is present", function()
        reset_calls()
        REGISTERED.trickiness({
            card = CARD,
            prompt_set = PROMPT_SET,
            seat = "boss",
            style = "guardian",
            temperature = 2.0,
        })
        expect(#TRICKY_CALLS).to.equal(1)
        expect(TRICKY_CALLS[1].temperature).to.equal(2.0)
    end)
end)

-- ─── Specs: style_distance card_a fallback ──────────────────────────

describe("gameai_metrics registry adapter — style_distance", function()
    it("falls back from ctx.card_a to ctx.card for the measured Card", function()
        reset_calls()
        REGISTERED.style_distance({
            card = CARD,
            card_b = "teacher-alias",
            prompt_set = PROMPT_SET,
            seat = "boss",
            style = "guardian",
        })
        expect(#SD_CALLS).to.equal(1)
        expect(SD_CALLS[1].card_a).to.equal(CARD)
        expect(SD_CALLS[1].card_b).to.equal("teacher-alias")
        expect(SD_CALLS[1].opts.seat).to.equal("boss")
    end)

    it("prefers an explicit ctx.card_a over ctx.card", function()
        reset_calls()
        local explicit = { _card = "explicit-a" }
        REGISTERED.style_distance({
            card = CARD,
            card_a = explicit,
            card_b = { _card = "b" },
            prompt_set = PROMPT_SET,
            seat = "boss",
            style = "guardian",
        })
        expect(#SD_CALLS).to.equal(1)
        expect(SD_CALLS[1].card_a).to.equal(explicit)
    end)
end)
