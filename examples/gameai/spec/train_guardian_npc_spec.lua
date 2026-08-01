-- spec/train_guardian_npc_spec.lua
--
-- Spec for the mid-run observation wiring of `train_guardian_npc.lua`.
--
-- The script is a self-contained `alc_run` driver rather than a package,
-- so it is exercised the way it actually runs: the whole file is loaded
-- once per case against a stubbed host, and the assertions read what it
-- handed to `alc.nn.trainer.run_full_ft` plus what came back out of the
-- hook. Everything expensive is a stub — the model, the dataset, the
-- trainer, the checkpoint loader and the three metrics — so no `nn`
-- feature and no training budget is involved. `guardian_duel` and
-- `anymetric` are the real modules: the states the prompt set is built
-- from and the record / decision / adapter contract under test are
-- exactly what a real run would use.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/train_guardian_npc_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })
--
-- so `require("train_guardian_npc")` / `require("guardian_duel")` /
-- `require("anymetric")` all resolve out of that one directory.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Minimal JSON encoder standing in for the host `alc.json_encode`.
---
--- Object keys are emitted in sorted order so a spec can match a fixed
--- substring against the result. The script feeds it two shapes: the
--- decision payloads it hands to the NPC package, and the observation
--- records it renders into the return value.
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

--- Log lines the script emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- `guardian_duel` is the real module and reaches for the host RNG. The
-- LCG below matches the shape the engine-level harness installs
-- (see gameai_metrics/spec/level_spec.lua); the corpus only has to be
-- reproducible, not statistically sound.
alc.math = alc.math or {}
alc.math.rng_create = function(seed)
    local state = seed
    if state == 0 then
        state = 0x9E3779B9
    end
    return { _state = state }
end
alc.math.rng_int = function(rng, min, max)
    local s = (rng._state * 1103515245 + 12345) % 2147483648
    rng._state = s
    return min + (s // 65536) % (max - min + 1)
end

alc.card = alc.card or {}
alc.card.alias_set = function() end
alc.card.get = function()
    -- Below `math.log(model_vocab)`, so the script's "gradients flowed"
    -- assertion passes and `ok` reports on the wiring instead.
    return { metadata = { nn = { metrics = { train_loss = 0.1 } } } }
end

alc.nn = alc.nn or {}

--- ctx tables the registry was asked to evaluate, in fire order.
local EVAL_CALLS = {}
--- metric name -> fn(ctx), swapped per case by `configure`.
local METRICS = {}

alc.nn.metric = alc.nn.metric or {}
alc.nn.metric.registry = {
    register = function() end,
    evaluate = function(name, ctx)
        EVAL_CALLS[#EVAL_CALLS + 1] = { name = name, ctx = ctx }
        local fn = METRICS[name]
        if fn == nil then
            error("spec: no stub metric named '" .. tostring(name) .. "'", 0)
        end
        return fn(ctx)
    end,
}

local duel = require("guardian_duel")

--- Stub model handle. Only the two accessors the script reads exist.
local function make_handle()
    return {
        ctx = function()
            return duel.CTX_BUDGET
        end,
        vocab = function()
            return 64
        end,
    }
end

alc.nn.preset = { gpt2 = make_handle }
alc.nn.data = {
    synthetic = function()
        return { _dataset = true }
    end,
}

--- Handles `load_ckpt` returned, in fire order.
local LOADED = {}
--- When set, `load_ckpt` raises this message instead of returning.
local LOAD_CKPT_ERROR = nil

alc.nn.card = {
    load_ckpt = function(path, opts)
        if LOAD_CKPT_ERROR ~= nil then
            error(LOAD_CKPT_ERROR, 0)
        end
        local handle = { _ckpt_path = path, _arch = opts and opts.arch }
        LOADED[#LOADED + 1] = handle
        return handle
    end,
}

-- ─── Trainer stub ───────────────────────────────────────────────────

--- Checkpoint fires the trainer stub performs when a hook is wired.
local PLANNED_FIRES = 3
--- Opts the script handed to `run_full_ft` on the last drive.
local TRAIN_OPTS = nil
--- What the hook returned on each fire, in order.
local HOOK_ACTIONS = {}

alc.nn.trainer = {
    run_full_ft = function(_, _, opts)
        TRAIN_OPTS = opts
        if type(opts.on_ckpt) == "function" then
            for fire = 1, PLANNED_FIRES do
                local action = opts.on_ckpt({
                    step = fire * opts.ckpt_every,
                    ckpt_path = "ckpt-" .. fire .. ".safetensors",
                    train_loss = 1.0 / fire,
                    lr = 3e-3,
                    grad_norm = 0.5,
                    elapsed_ms = fire * 10,
                    min_train_loss = 1.0 / fire,
                })
                HOOK_ACTIONS[#HOOK_ACTIONS + 1] = action
                -- The trainer stops the run on "break", the same way
                -- the real loop reads the hook's answer.
                if action == "break" then
                    break
                end
            end
        end
        return "card-stub-0001"
    end,
}

-- ─── Package stubs ──────────────────────────────────────────────────

--- The NPC package is replaced wholesale: the decode checks it answers
--- are the pre-existing part of the script, not what is under test.
--- The payload arrives as the JSON the script encoded, so the mode is
--- read back out of that string.
package.preload["guardian_duel_npc"] = function()
    return {
        reset_cache = function() end,
        run = function(opts)
            local task = tostring(opts.task)
            if task:find('"mode":"decide"', 1, true) then
                return { result = "action=a legal=true" }
            end
            if task:find('"mode":"determinism"', 1, true) then
                return { result = "deterministic=true" }
            end
            if task:find('"mode":"selfplay"', 1, true) then
                return { result = "style_match=0.500 compliance=1.000" }
            end
            error("spec: unexpected npc.run task " .. task, 0)
        end,
    }
end

--- The real pkg self-registers into the registry; the stub registry
--- above owns that mapping instead, so the require only has to succeed.
package.preload["gameai_metrics"] = function()
    return {}
end

-- ─── Driver ─────────────────────────────────────────────────────────

--- Values the stub `level` metric reports, overridden per case.
local LEVEL_VALUES = {}
--- When set, the `level` metric raises this message.
local LEVEL_ERROR = nil

local function configure()
    EVAL_CALLS = {}
    LOADED = {}
    LOG_LINES = {}
    HOOK_ACTIONS = {}
    TRAIN_OPTS = nil
    LOAD_CKPT_ERROR = nil
    LEVEL_ERROR = nil
    LEVEL_VALUES = {
        win_rate = 0.40,
        ci_lower = 0.28,
        ci_upper = 0.53,
        wins = 20.0,
        n_games = 50,
        win_rate_min = 0.40,
    }
    METRICS = {
        level = function()
            if LEVEL_ERROR ~= nil then
                error(LEVEL_ERROR, 0)
            end
            return LEVEL_VALUES
        end,
        -- Scalar return, exercising the values-lifting rule on the way
        -- into the record.
        style_distance = function()
            return 0.1875
        end,
        trickiness = function()
            return { value = 0.62, raw_mean = 1.11 }
        end,
    }
end

--- Load the script once against the current stubs.
---
--- `ctx` is a global in the `alc_run` contract, so it is planted as one
--- here. `package.loaded` is cleared first: every case needs the
--- top-level ctx decoding to run again.
---@param overrides table ctx fields for this case
---@return table result the script's return value
local function drive(overrides)
    local script_ctx = {
        -- Smallest run that still builds a corpus and reaches the
        -- trainer: the training itself is a stub.
        games = 1,
        steps = 2,
        batch = 1,
        check_games = 1,
        style = "guardian",
        ckpt_every = 2,
        ckpt_keep = 2,
        gate_games = 6,
    }
    for key, value in pairs(overrides or {}) do
        script_ctx[key] = value
    end
    ctx = script_ctx
    package.loaded["train_guardian_npc"] = nil
    return require("train_guardian_npc")
end

--- Every ctx the named metric was evaluated with.
local function calls_to(name)
    local out = {}
    for _, call in ipairs(EVAL_CALLS) do
        if call.name == name then
            out[#out + 1] = call.ctx
        end
    end
    return out
end

local function contains(haystack, needle)
    return tostring(haystack):find(needle, 1, true) ~= nil
end

-- ─── Cases ──────────────────────────────────────────────────────────

describe("train_guardian_npc checkpoint observation", function()
    it("wires the hook and the checkpoint keys together", function()
        configure()
        local out = drive({})

        expect(type(TRAIN_OPTS.on_ckpt)).to.equal("function")
        expect(TRAIN_OPTS.ckpt_every).to.equal(2)
        expect(TRAIN_OPTS.ckpt_keep).to.equal(2)
        -- The pre-existing training config is untouched.
        expect(TRAIN_OPTS.steps).to.equal(2)
        expect(TRAIN_OPTS.batch).to.equal(1)
        expect(TRAIN_OPTS.schedule).to.equal("Constant")
        expect(out.card_id).to.equal("card-stub-0001")
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
    end)

    it("observes every view once per fire", function()
        configure()
        local out = drive({})

        expect(#calls_to("level")).to.equal(PLANNED_FIRES)
        expect(#calls_to("style_distance")).to.equal(PLANNED_FIRES)
        expect(#calls_to("trickiness")).to.equal(PLANNED_FIRES)
        expect(#LOADED).to.equal(PLANNED_FIRES)
        -- Three records per fire land in the log and come back out as
        -- the JSON trail the observation note is written from.
        expect(contains(out.observations, '"view_id":"level"')).to.equal(true)
        expect(contains(out.observations, '"view_id":"sd_teacher"')).to.equal(true)
        expect(contains(out.observations, '"view_id":"trickiness"')).to.equal(true)
        expect(contains(out.observations, '"step":2')).to.equal(true)
        expect(contains(out.observations, '"step":6')).to.equal(true)
    end)

    it("hands each metric the seat, the basis and its own config", function()
        configure()
        drive({ style = "guardian", teacher_alias = "teacher-card", gate_games = 6 })

        local level_ctx = calls_to("level")[1]
        expect(level_ctx.seat).to.equal("boss")
        expect(level_ctx.style).to.equal("guardian")
        expect(level_ctx.opponents[1]).to.equal("random")
        expect(#level_ctx.opponents).to.equal(1)
        expect(level_ctx.n_games).to.equal(6)
        expect(level_ctx.step).to.equal(2)
        -- The per-fire checkpoint handle, not a config value.
        expect(level_ctx.card).to.equal(LOADED[1])
        expect(level_ctx.card._arch).to.equal("gpt2-tiny")

        local sd_ctx = calls_to("style_distance")[1]
        expect(sd_ctx.seat).to.equal("boss")
        expect(sd_ctx.card_b).to.equal("teacher-card")
        expect(sd_ctx.card).to.equal(LOADED[1])

        local trick_ctx = calls_to("trickiness")[1]
        expect(trick_ctx.seat).to.equal("boss")
        expect(trick_ctx.style).to.equal("guardian")

        -- The prompt set is the four reachable branch states, which on
        -- the boss seat are boss states rather than player views.
        expect(#sd_ctx.prompt_set).to.equal(4)
        expect(#trick_ctx.prompt_set).to.equal(4)
        expect(type(sd_ctx.prompt_set[1].cycle)).to.equal("number")
        expect(type(sd_ctx.prompt_set[1].mode)).to.equal("number")
    end)

    it("keeps the seed fixed across fires so two fires are paired", function()
        configure()
        drive({ seed = 4242 })

        local seeds = calls_to("level")
        expect(#seeds).to.equal(PLANNED_FIRES)
        for _, level_ctx in ipairs(seeds) do
            expect(level_ctx.seed).to.equal(4242)
        end
    end)
end)

describe("train_guardian_npc judgment", function()
    it("never stops the run while the gate is disabled", function()
        configure()
        -- Well past the target: with the gate off it still must not
        -- end the run.
        LEVEL_VALUES.ci_lower = 0.99
        local out = drive({ enable_gate = false, target_win_rate_lo = 0.55 })

        expect(out.gate_enabled).to.equal(false)
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
    end)

    it("stops at the first fire that reaches the lower bound", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.61
        local out = drive({ enable_gate = true, target_win_rate_lo = 0.55 })

        expect(out.gate_enabled).to.equal(true)
        expect(out.gate_target_lo).to.equal(0.55)
        expect(HOOK_ACTIONS[1]).to.equal("break")
        expect(#HOOK_ACTIONS).to.equal(1)
        expect(out.ckpt_fires).to.equal(1)
        -- The run still lands its Card: the trainer saves after a
        -- hook-requested stop.
        expect(out.card_id).to.equal("card-stub-0001")
    end)

    it("keeps going while the lower bound is short of the target", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.31
        local out = drive({ enable_gate = true, target_win_rate_lo = 0.55 })

        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
    end)

    it("reads the strength view only", function()
        configure()
        -- A distance of zero and an entropy of zero are as extreme as
        -- the personality axes get; neither may move the gate.
        METRICS.style_distance = function()
            return 0.0
        end
        METRICS.trickiness = function()
            return { value = 0.0, raw_mean = 0.0 }
        end
        LEVEL_VALUES.ci_lower = 0.10
        local out = drive({ enable_gate = true, target_win_rate_lo = 0.55 })

        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
    end)
end)

describe("train_guardian_npc measurement failure", function()
    it("records a failing metric and lets the run finish", function()
        configure()
        LEVEL_ERROR = "level exploded on purpose"
        local out = drive({})

        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
        expect(out.card_id).to.equal("card-stub-0001")
        expect(contains(out.observations, "level exploded on purpose")).to.equal(true)
        -- The other two views were still measured on the same fire.
        expect(#calls_to("style_distance")).to.equal(PLANNED_FIRES)
        expect(contains(out.observations, '"view_id":"trickiness"')).to.equal(true)
    end)

    it("does not let a measurement gap satisfy an enabled gate", function()
        configure()
        LEVEL_ERROR = "level exploded on purpose"
        local out = drive({ enable_gate = true, target_win_rate_lo = 0.55 })

        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
    end)

    it("records a failing checkpoint load and lets the run finish", function()
        configure()
        LOAD_CKPT_ERROR = "load_ckpt: no such file"
        local out = drive({})

        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
        expect(out.card_id).to.equal("card-stub-0001")
        expect(contains(out.observations, '"view_id":"ckpt_load"')).to.equal(true)
        -- No handle reached a metric, so none of them ran.
        expect(#EVAL_CALLS).to.equal(0)
    end)
end)

describe("train_guardian_npc with observation disabled", function()
    it("hands the trainer no hook and no checkpoint keys", function()
        configure()
        local out = drive({ ckpt_every = 0 })

        expect(TRAIN_OPTS.on_ckpt).to.equal(nil)
        expect(TRAIN_OPTS.ckpt_every).to.equal(nil)
        expect(TRAIN_OPTS.ckpt_keep).to.equal(nil)
        expect(#EVAL_CALLS).to.equal(0)
        expect(#LOADED).to.equal(0)
        expect(out.ckpt_fires).to.equal(0)
        expect(out.observations).to.equal("[]")
        -- The rest of the run is untouched.
        expect(out.card_id).to.equal("card-stub-0001")
        expect(out.style_total).to.equal(4)
        expect(out.deterministic).to.equal(true)
    end)
end)

describe("train_guardian_npc observation config", function()
    it("refuses a checkpoint budget that cannot be observed", function()
        configure()
        local ok, err = pcall(drive, { ckpt_every = 4, ckpt_keep = 0 })
        expect(ok).to.equal(false)
        expect(contains(err, "ckpt_keep")).to.equal(true)
    end)

    it("refuses a gate target outside [0, 1]", function()
        configure()
        local ok, err = pcall(drive, { ckpt_every = 4, target_win_rate_lo = 1.5 })
        expect(ok).to.equal(false)
        expect(contains(err, "target_win_rate_lo")).to.equal(true)
    end)

    it("refuses a non-positive gate game count", function()
        configure()
        local ok, err = pcall(drive, { ckpt_every = 4, gate_games = 0 })
        expect(ok).to.equal(false)
        expect(contains(err, "gate_games")).to.equal(true)
    end)
end)
