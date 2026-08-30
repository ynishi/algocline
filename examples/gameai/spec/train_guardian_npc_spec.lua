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

--- alias_set invocations recorded in fire order. Each entry is
--- `{alias, card_id, opts}`; specs assert on both the alias name (built
--- from `<prefix>_<label>`) and the count so a double-bake would show
--- up as an over-count rather than silently as a duplicate row.
local ALIAS_SET_CALLS = {}
alc.card.alias_set = function(alias, card_id, opts)
    ALIAS_SET_CALLS[#ALIAS_SET_CALLS + 1] = { alias = alias, card_id = card_id, opts = opts }
end
alc.card.get = function()
    -- Below `math.log(model_vocab)`, so the script's "gradients flowed"
    -- assertion passes and `ok` reports on the wiring instead.
    return { metadata = { nn = { metrics = { train_loss = 0.1 } } } }
end

alc.nn = alc.nn or {}

--- ctx tables a metric was called with, in fire order.
local EVAL_CALLS = {}
--- metric name -> fn(ctx), swapped per case by `configure`.
local METRICS = {}

--- Build one stub ctx adapter. The name is resolved against `METRICS`
--- at call time, so `configure` can swap a case's metrics long after
--- the script bound the adapter into its views.
local function stub_metric(name)
    return function(ctx)
        EVAL_CALLS[#EVAL_CALLS + 1] = { name = name, ctx = ctx }
        local fn = METRICS[name]
        if fn == nil then
            error("spec: no stub metric named '" .. tostring(name) .. "'", 0)
        end
        return fn(ctx)
    end
end

-- The script reaches its metrics through the package table, so the fake
-- package has to be in place before the script runs.
package.loaded["gameai_metrics"] = {
    metrics = {
        level = stub_metric("level"),
        trickiness = stub_metric("trickiness"),
        style_distance = stub_metric("style_distance"),
    },
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

--- save_from_ckpt invocations recorded in fire order. Each entry is
--- `{ckpt_path, name, meta}` so a spec can check that the copy source
--- is the very ckpt path the hook was handed (rather than a cached
--- previous path) and that the meta carries `training_path = "full_ft"`.
local SAVE_FROM_CKPT_CALLS = {}
--- Card IDs the save_from_ckpt stub hands back, one per call.
local NEXT_CARD_ID = 0

local function next_card_id()
    NEXT_CARD_ID = NEXT_CARD_ID + 1
    return string.format("card-harvest-%04d", NEXT_CARD_ID)
end

alc.nn.card = {
    load_ckpt = function(path, opts)
        if LOAD_CKPT_ERROR ~= nil then
            error(LOAD_CKPT_ERROR, 0)
        end
        local handle = { _ckpt_path = path, _arch = opts and opts.arch }
        LOADED[#LOADED + 1] = handle
        return handle
    end,
    save_from_ckpt = function(path, name, meta)
        local card_id = next_card_id()
        SAVE_FROM_CKPT_CALLS[#SAVE_FROM_CKPT_CALLS + 1] = {
            ckpt_path = path,
            name = name,
            meta = meta,
            card_id = card_id,
        }
        return card_id
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

--- Records handed to the harvest collection helper, in append order.
--- Each entry mirrors the tuple the train script feeds `:append`.
local COLL_APPEND_CALLS = {}
--- Number of `:save()` calls the collection saw during the drive.
local COLL_SAVE_CALLS = 0
--- Opts the script handed to `hc.new` on the last drive (nil when the
--- staged path was off).
local COLL_NEW_OPTS = nil

--- Stub collection helper. Mirrors the real one's contract without
--- touching the filesystem: `:append` respects first-writer-wins per
--- label, `:save` is a counter, and the two invocation logs above let
--- a spec assert on both the tuple shape and the call ordering. The
--- real helper is spec'd separately under
--- `gameai_metrics/spec/harvest_collection_spec.lua`.
package.preload["gameai_metrics.harvest_collection"] = function()
    return {
        new = function(opts)
            COLL_NEW_OPTS = opts
            local coll = {
                _entries = {},
                _by_label = {},
            }
            function coll:append(dec, info, records, extra)
                COLL_APPEND_CALLS[#COLL_APPEND_CALLS + 1] = {
                    dec = dec,
                    info = info,
                    records = records,
                    extra = extra,
                }
                local label = dec and dec.meta and dec.meta.label
                if type(label) ~= "string" or label == "" then
                    return false
                end
                if self._by_label[label] then
                    return false
                end
                self._by_label[label] = true
                self._entries[#self._entries + 1] = { label = label, extra = extra }
                return true
            end
            function coll:save()
                COLL_SAVE_CALLS = COLL_SAVE_CALLS + 1
            end
            function coll:entries()
                local out = {}
                for index, entry in ipairs(self._entries) do
                    out[index] = entry
                end
                return out
            end
            return coll
        end,
    }
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
    ALIAS_SET_CALLS = {}
    SAVE_FROM_CKPT_CALLS = {}
    NEXT_CARD_ID = 0
    COLL_APPEND_CALLS = {}
    COLL_SAVE_CALLS = 0
    COLL_NEW_OPTS = nil
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

--- Filter `ALIAS_SET_CALLS` down to the ones matching `alias`. Kept
--- separate from a raw length check because `train_style` also pins
--- the training aliases (`guardian_duel_npc_<style>` and the bare
--- `guardian_duel_npc`) on every run, which have nothing to do with
--- the staged harvest path and would otherwise dominate the count.
local function alias_set_calls_for(alias)
    local out = {}
    for _, call in ipairs(ALIAS_SET_CALLS) do
        if call.alias == alias then
            out[#out + 1] = call
        end
    end
    return out
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

describe("train_guardian_npc staged harvest", function()
    it("does not touch the harvest path when enable_stages is off", function()
        configure()
        -- ci_lower sits inside the default weak band; the staged path
        -- would harvest here if it were live. With the switch off it
        -- must not fire the bake, the alias, or the collection.
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({})

        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        -- The training aliases still get pinned per style (that path
        -- is pre-existing); only the harvest-specific aliases stay
        -- absent when the staged switch is off.
        expect(#alias_set_calls_for("guardian_duel_npc_weak")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_mid")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_strong")).to.equal(0)
        expect(#COLL_APPEND_CALLS).to.equal(0)
        expect(COLL_SAVE_CALLS).to.equal(0)
        expect(COLL_NEW_OPTS).to.equal(nil)
        expect(out.stages_enabled).to.equal(false)
        expect(#out.stages_harvested).to.equal(0)
        expect(out.collection_path).to.equal(nil)
        -- Existing hook path stays byte-invariant.
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
    end)

    it("bakes a Card, pins the alias and records one entry on harvest", function()
        configure()
        -- Sits in the default weak band [0.10, 0.30].
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({ enable_stages = true })

        expect(type(COLL_NEW_OPTS)).to.equal("table")
        expect(COLL_NEW_OPTS.path).to.equal("workspace/gameai-harvest/collection.json")
        expect(#COLL_NEW_OPTS.bands).to.equal(3)
        expect(COLL_NEW_OPTS.meta.style).to.equal("guardian")

        -- Only the first fire in the band pays the bake; the other two
        -- hit first-writer-wins and skip the Card copy entirely.
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(1)
        expect(SAVE_FROM_CKPT_CALLS[1].ckpt_path).to.equal("ckpt-1.safetensors")
        expect(SAVE_FROM_CKPT_CALLS[1].name).to.equal("guardian_duel_npc_weak")
        expect(SAVE_FROM_CKPT_CALLS[1].meta.architecture).to.equal("gpt2-tiny")
        expect(SAVE_FROM_CKPT_CALLS[1].meta.training_path).to.equal("full_ft")

        local weak_aliases = alias_set_calls_for("guardian_duel_npc_weak")
        expect(#weak_aliases).to.equal(1)
        expect(weak_aliases[1].card_id).to.equal(SAVE_FROM_CKPT_CALLS[1].card_id)
        expect(contains(weak_aliases[1].opts.note, "label=weak")).to.equal(true)

        expect(#COLL_APPEND_CALLS).to.equal(1)
        expect(COLL_APPEND_CALLS[1].extra.card_id).to.equal(SAVE_FROM_CKPT_CALLS[1].card_id)
        expect(COLL_APPEND_CALLS[1].extra.alias).to.equal("guardian_duel_npc_weak")
        -- Write-through: save() rides along with the append that
        -- actually stored an entry.
        expect(COLL_SAVE_CALLS).to.equal(1)

        expect(out.stages_enabled).to.equal(true)
        expect(#out.stages_harvested).to.equal(1)
        expect(out.stages_harvested[1]).to.equal("weak")
        expect(out.collection_path).to.equal("workspace/gameai-harvest/collection.json")
        -- A harvesting fire answers the trainer with the keep table
        -- rather than a bare "continue": the run goes on *and* the
        -- checkpoint the manifest names is held out of the rotation,
        -- so `ckpt_path` in the manifest still resolves after the run.
        for _, action in ipairs(HOOK_ACTIONS) do
            if type(action) == "table" then
                expect(action.action).to.equal("continue")
                expect(action.keep).to.equal("weak")
            else
                expect(action).to.equal("continue")
            end
        end
    end)

    it("stops the run when the staged judgment breaks above the top band", function()
        configure()
        -- Above default strong.hi = 0.98: the staged judgment answers
        -- "break", and no Card is baked (harvest and break are
        -- exclusive per fire).
        LEVEL_VALUES.ci_lower = 0.99
        local out = drive({ enable_stages = true })

        expect(HOOK_ACTIONS[1]).to.equal("break")
        expect(#HOOK_ACTIONS).to.equal(1)
        expect(out.ckpt_fires).to.equal(1)
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_strong")).to.equal(0)
        expect(#COLL_APPEND_CALLS).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        -- The Card is still minted by the trainer stub on a
        -- hook-requested stop.
        expect(out.card_id).to.equal("card-stub-0001")
    end)

    it("harvests then continues when only the staged path fires under a coexisting gate", function()
        configure()
        -- Below the gate target, inside the weak band. Staged harvests
        -- weak, gate continues, hook keeps going.
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({
            enable_stages = true,
            enable_gate = true,
            target_win_rate_lo = 0.55,
        })

        expect(#SAVE_FROM_CKPT_CALLS).to.equal(1)
        expect(SAVE_FROM_CKPT_CALLS[1].name).to.equal("guardian_duel_npc_weak")
        expect(out.stages_harvested[1]).to.equal("weak")
        expect(out.gate_enabled).to.equal(true)
        -- Same as above: the harvesting fire keeps, the others just
        -- continue. Neither stops the run under a coexisting gate.
        for _, action in ipairs(HOOK_ACTIONS) do
            if type(action) == "table" then
                expect(action.action).to.equal("continue")
                expect(action.keep).to.equal("weak")
            else
                expect(action).to.equal("continue")
            end
        end
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
    end)

    it("breaks on the gate when the staged path is only harvesting under coexistence", function()
        configure()
        -- Sits in the default mid band [0.55, 0.85] and above the
        -- gate target: staged harvests mid, gate breaks. Harvest
        -- side-effects still land on the fire the break happens on,
        -- so the manifest keeps the mid entry.
        LEVEL_VALUES.ci_lower = 0.75
        local out = drive({
            enable_stages = true,
            enable_gate = true,
            target_win_rate_lo = 0.55,
        })

        expect(HOOK_ACTIONS[1]).to.equal("break")
        expect(#HOOK_ACTIONS).to.equal(1)
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(1)
        expect(SAVE_FROM_CKPT_CALLS[1].name).to.equal("guardian_duel_npc_mid")
        expect(out.stages_harvested[1]).to.equal("mid")
    end)

    it("prefers the staged break when the staged and gate breaks fire together", function()
        configure()
        -- Above default strong.hi and above the gate target: staged
        -- says break, gate says break. Either way the hook stops the
        -- run; the assertion is that no harvest side-effect leaks in
        -- (a staged break is exclusive with harvest for the same fire).
        LEVEL_VALUES.ci_lower = 0.99
        local out = drive({
            enable_stages = true,
            enable_gate = true,
            target_win_rate_lo = 0.55,
        })

        expect(HOOK_ACTIONS[1]).to.equal("break")
        expect(#HOOK_ACTIONS).to.equal(1)
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_strong")).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
    end)

    it("skips the harvest when the strength view records an error", function()
        configure()
        LEVEL_ERROR = "level exploded on purpose"
        local out = drive({ enable_stages = true })

        -- Staged reads the error record as a miss and continues; the
        -- run keeps going without paying any bake or alias.
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_weak")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_mid")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_strong")).to.equal(0)
        expect(#COLL_APPEND_CALLS).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
    end)

    it("skips the harvest when the checkpoint fails to load", function()
        configure()
        LOAD_CKPT_ERROR = "load_ckpt: no such file"
        local out = drive({ enable_stages = true })

        -- No handle reached the staged judgment; nothing to harvest.
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_weak")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_mid")).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_strong")).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
    end)

    it("refuses ckpt_keep < 2 when stages are enabled", function()
        configure()
        local ok, err = pcall(drive, { enable_stages = true, ckpt_keep = 1 })
        expect(ok).to.equal(false)
        expect(contains(err, "ckpt_keep")).to.equal(true)
        expect(contains(err, "enable_stages")).to.equal(true)
    end)

    it("threads a custom alias prefix through both the bake and the alias", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({
            enable_stages = true,
            stage_alias_prefix = "custom_boss",
        })

        expect(SAVE_FROM_CKPT_CALLS[1].name).to.equal("custom_boss_weak")
        expect(#alias_set_calls_for("custom_boss_weak")).to.equal(1)
        expect(out.stages_harvested[1]).to.equal("weak")
    end)

    it("respects a caller-supplied collection path in the manifest options", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({
            enable_stages = true,
            collection_path = "/tmp/spec-collection.json",
        })

        expect(COLL_NEW_OPTS.path).to.equal("/tmp/spec-collection.json")
        expect(out.collection_path).to.equal("/tmp/spec-collection.json")
    end)
end)

describe("train_guardian_npc stage band require", function()
    --- A one-band schedule whose only band is conditional on the
    --- trickiness view. `min` is moved per case; the stub trickiness
    --- metric reports 0.62.
    local function require_bands(min, view_id)
        return {
            {
                lo = 0.10,
                hi = 0.30,
                label = "mid_v2",
                require = { view_id = view_id or "trickiness", field = "value", min = min },
            },
        }
    end

    it("refuses a require pointed at a view this run does not observe", function()
        configure()
        local ok, err = pcall(drive, {
            enable_stages = true,
            stage_bands = require_bands(0.57, "counter_resistance"),
        })
        expect(ok).to.equal(false)
        expect(contains(err, "counter_resistance")).to.equal(true)
        expect(contains(err, "mid_v2")).to.equal(true)
        -- The message lists what the run does observe, so the fix is
        -- readable off the error itself.
        expect(contains(err, "trickiness")).to.equal(true)
        -- Refused before any training budget is spent.
        expect(TRAIN_OPTS).to.equal(nil)
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
    end)

    it("accepts a require naming one of the three observed views", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({ enable_stages = true, stage_bands = require_bands(0.57) })

        -- trickiness reports 0.62, which clears the 0.57 floor.
        expect(#SAVE_FROM_CKPT_CALLS).to.equal(1)
        expect(SAVE_FROM_CKPT_CALLS[1].name).to.equal("guardian_duel_npc_mid_v2")
        expect(out.stages_harvested[1]).to.equal("mid_v2")
        -- The requirement rides into the manifest options as well.
        expect(COLL_NEW_OPTS.bands[1].require.view_id).to.equal("trickiness")
        expect(COLL_NEW_OPTS.bands[1].require.min).to.equal(0.57)
        -- The harvest tuple carries the satisfied requirement.
        expect(COLL_APPEND_CALLS[1].dec.meta.require.value).to.equal(0.62)
        expect(#out.stages_require_gaps).to.equal(0)
    end)

    it("harvests nothing while the required view sits below the floor", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({ enable_stages = true, stage_bands = require_bands(0.90) })

        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_mid_v2")).to.equal(0)
        expect(#COLL_APPEND_CALLS).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        expect(out.ckpt_fires).to.equal(PLANNED_FIRES)
        for _, action in ipairs(HOOK_ACTIONS) do
            expect(action).to.equal("continue")
        end
        -- The view was read on every fire, so this is a refusal rather
        -- than a measurement gap.
        expect(#out.stages_require_gaps).to.equal(0)
    end)

    it("ships the trickiness floor on the default mid band", function()
        configure()
        -- Mid-band strength, but the policy has collapsed under the
        -- default floor of 0.57: the shipped default refuses the
        -- harvest instead of baking a counter-farmable mid.
        LEVEL_VALUES.ci_lower = 0.75
        METRICS.trickiness = function()
            return { value = 0.40, raw_mean = 1.11 }
        end
        local out = drive({ enable_stages = true })

        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#alias_set_calls_for("guardian_duel_npc_mid")).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        -- The view read numbers on every fire: a refusal, not a gap.
        expect(#out.stages_require_gaps).to.equal(0)
    end)

    it("reports a require whose view never produced a numeric reading", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        METRICS.trickiness = function()
            error("trickiness exploded on purpose", 0)
        end
        local out = drive({ enable_stages = true, stage_bands = require_bands(0.57) })

        expect(#SAVE_FROM_CKPT_CALLS).to.equal(0)
        expect(#out.stages_harvested).to.equal(0)
        -- Without this the run is indistinguishable from one where the
        -- model simply never qualified.
        expect(#out.stages_require_gaps).to.equal(1)
        expect(contains(out.stages_require_gaps[1], "trickiness.value")).to.equal(true)
        expect(contains(out.stages_require_gaps[1], "mid_v2")).to.equal(true)
        local logged = false
        for _, line in ipairs(LOG_LINES) do
            if contains(line, "REQUIRE GAP") then
                logged = true
            end
        end
        expect(logged).to.equal(true)
    end)

    it("reports no gap and no field when the staged path is off", function()
        configure()
        local out = drive({})
        expect(out.stages_require_gaps).to.equal(nil)
    end)

    it("carries the trickiness floor on the default mid band only", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({ enable_stages = true })

        -- The default schedule gates mid on decode entropy and nothing
        -- else: weak and strong harvest unconditionally.
        expect(COLL_NEW_OPTS.bands[1].require).to.equal(nil)
        expect(COLL_NEW_OPTS.bands[2].require.view_id).to.equal("trickiness")
        expect(COLL_NEW_OPTS.bands[2].require.field).to.equal("value")
        expect(COLL_NEW_OPTS.bands[2].require.min).to.equal(0.57)
        expect(COLL_NEW_OPTS.bands[3].require).to.equal(nil)
        expect(out.stages_harvested[1]).to.equal("weak")
        expect(#out.stages_require_gaps).to.equal(0)
    end)
end)

describe("train_guardian_npc bare alias pin", function()
    it("pins the bare alias for the guardian style by default", function()
        configure()
        local out = drive({})
        expect(out.pin_bare_alias).to.equal(true)
        expect(#alias_set_calls_for("guardian_duel_npc")).to.equal(1)
        expect(alias_set_calls_for("guardian_duel_npc")[1].card_id).to.equal("card-stub-0001")
    end)

    it("leaves the bare alias alone when pin_bare_alias is false", function()
        configure()
        local out = drive({ pin_bare_alias = false })

        expect(out.pin_bare_alias).to.equal(false)
        expect(#alias_set_calls_for("guardian_duel_npc")).to.equal(0)
        -- The style-specific alias is still pinned: only the shared
        -- fallback the teacher lives on is left where it was.
        expect(#alias_set_calls_for("guardian_duel_npc_guardian")).to.equal(1)
        expect(out.card_id).to.equal("card-stub-0001")
        local skipped = false
        for _, line in ipairs(LOG_LINES) do
            if contains(line, "left untouched") then
                skipped = true
            end
        end
        expect(skipped).to.equal(true)
    end)

    it("still skips the harvest aliases' own pinning rules", function()
        configure()
        LEVEL_VALUES.ci_lower = 0.20
        local out = drive({ enable_stages = true, pin_bare_alias = false })

        -- Per-band aliases are unaffected by the bare-alias switch.
        expect(#alias_set_calls_for("guardian_duel_npc_weak")).to.equal(1)
        expect(#alias_set_calls_for("guardian_duel_npc")).to.equal(0)
        expect(out.stages_harvested[1]).to.equal("weak")
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
