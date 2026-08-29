-- anymetric/spec/anymetric_spec.lua
--
-- Package-level spec for the AnyMetric domain module. The module touches
-- no host surface at all — a view carries the metric function itself —
-- so the suite hands it plain closures and records every ctx they were
-- called with. No model, no nn feature, no Card, no `alc`.
--
-- The stub metrics stand in for the gameai instances: `scalar_metric`
-- returns a bare number (exercising the values lifting rule),
-- `table_metric` returns a table, `boom_metric` raises.

local describe, it, expect = lust.describe, lust.it, lust.expect

--- Every metric call the module made since the last reset.
local EVAL_CALLS = {}
--- name -> fn(ctx) stub metrics the current test wants to expose.
local METRICS = {}

--- Build the metric function a view holds. The name is resolved against
--- `METRICS` at call time, not at build time, so `reset()` can swap the
--- stub behind a view that was already constructed — and an entry the
--- test never defined raises from inside the view, which is where a
--- caller's own broken metric would raise too.
local function metric(name)
    return function(ctx)
        EVAL_CALLS[#EVAL_CALLS + 1] = { name = name, ctx = ctx }
        local fn = METRICS[name]
        if fn == nil then
            error("unknown metric '" .. tostring(name) .. "'", 0)
        end
        return fn(ctx)
    end
end

local am = require("anymetric")

local function reset()
    EVAL_CALLS = {}
    METRICS = {
        scalar_metric = function()
            return 0.75
        end,
        table_metric = function()
            return { win_rate = 0.62, ci_lower = 0.48, ci_upper = 0.74 }
        end,
        boom_metric = function()
            error("metric exploded on purpose", 0)
        end,
        nil_metric = function()
            return nil
        end,
        echo_metric = function(ctx)
            return { seen = 1, ctx = ctx }
        end,
    }
end

--- Last ctx the named metric was evaluated with.
local function last_ctx(name)
    for index = #EVAL_CALLS, 1, -1 do
        if EVAL_CALLS[index].name == name then
            return EVAL_CALLS[index].ctx
        end
    end
    return nil
end

--- A record whose every non-raw field access raises. Reading anything off
--- it proves a judgment strayed outside the view it was bound to.
local function booby_record(view_id)
    return setmetatable({ view_id = view_id }, {
        __index = function(_, key)
            error("judgment touched foreign record field '" .. tostring(key) .. "'", 0)
        end,
    })
end

describe("anymetric.view", function()
    it("binds view_id / metric / config", function()
        reset()
        local fn = metric("level_metric")
        local v = am.view("level", fn, { opponents = { "random" }, n_games = 50 })
        expect(v.view_id).to.equal("level")
        expect(v.metric).to.equal(fn)
        expect(v.config.n_games).to.equal(50)
    end)

    it("copies the config so later caller mutation cannot reach the view", function()
        reset()
        local config = { n_games = 50 }
        local v = am.view("level", metric("level_metric"), config)
        config.n_games = 1
        expect(v.config.n_games).to.equal(50)
    end)

    it("defaults config to an empty table", function()
        reset()
        local v = am.view("t", metric("trickiness"), nil)
        expect(type(v.config)).to.equal("table")
        expect(next(v.config)).to.equal(nil)
    end)

    it("rejects an empty or non-string view_id", function()
        reset()
        local fn = metric("m")
        expect(pcall(am.view, "", fn, {})).to.equal(false)
        expect(pcall(am.view, nil, fn, {})).to.equal(false)
        expect(pcall(am.view, 7, fn, {})).to.equal(false)
    end)

    it("rejects a metric that is not a function, naming the view", function()
        reset()
        -- A name is the shape this argument used to take, so it is the
        -- one a stale caller is most likely to still pass.
        local ok, err = pcall(am.view, "level", "level_metric", {})
        expect(ok).to.equal(false)
        expect(err:find("level") ~= nil).to.equal(true)
        expect(pcall(am.view, "level", nil, {})).to.equal(false)
        expect(pcall(am.view, "level", {}, {})).to.equal(false)
    end)

    it("keeps a declared required key that the config supplies", function()
        reset()
        local v = am.view(
            "level",
            metric("level_metric"),
            { opponents = { "random" }, required = { "opponents" } }
        )
        expect(v.config.opponents[1]).to.equal("random")
    end)

    it("strips `required` from the bound config so it never reaches the metric", function()
        reset()
        local v = am.view(
            "level",
            metric("level_metric"),
            { opponents = { "random" }, required = { "opponents" } }
        )
        expect(v.config.required).to.equal(nil)
    end)

    it("raises loudly when a required key is missing, naming view and key", function()
        reset()
        local ok, err = pcall(
            am.view,
            "sd_teacher",
            metric("style_distance"),
            { required = { "card_b", "prompt_set" } }
        )
        expect(ok).to.equal(false)
        expect(err:find("sd_teacher") ~= nil).to.equal(true)
        expect(err:find("card_b") ~= nil).to.equal(true)
    end)

    it("rejects a config that sets a reserved key", function()
        reset()
        local ok, err = pcall(am.view, "level", metric("level_metric"), { card = "handle" })
        expect(ok).to.equal(false)
        expect(err:find("card") ~= nil).to.equal(true)
        expect(err:find("level") ~= nil).to.equal(true)
    end)

    it("rejects a reserved key declared as required", function()
        reset()
        local ok, err = pcall(am.view, "level", metric("level_metric"), { required = { "card" } })
        expect(ok).to.equal(false)
        expect(err:find("card") ~= nil).to.equal(true)
    end)

    it("rejects a non-table config and a non-array required", function()
        reset()
        expect(pcall(am.view, "level", metric("m"), "nope")).to.equal(false)
        expect(pcall(am.view, "level", metric("m"), { required = "card_b" })).to.equal(false)
    end)
end)

describe("anymetric.observe", function()
    it("returns one uniform record per view, in view order", function()
        reset()
        local records = am.observe({
            am.view("a", metric("table_metric"), {}),
            am.view("b", metric("scalar_metric"), {}),
        }, { card = "handle", step = 120 })
        expect(#records).to.equal(2)
        expect(records[1].view_id).to.equal("a")
        expect(records[1].step).to.equal(120)
        expect(records[2].view_id).to.equal("b")
        expect(records[2].step).to.equal(120)
    end)

    it("passes a table result straight through as values", function()
        reset()
        local records = am.observe(
            { am.view("a", metric("table_metric"), {}) },
            { card = "h", step = 1 }
        )
        expect(records[1].values.win_rate).to.equal(0.62)
        expect(records[1].values.ci_lower).to.equal(0.48)
    end)

    it("lifts a scalar result into { value = x }", function()
        reset()
        local records = am.observe(
            { am.view("b", metric("scalar_metric"), {}) },
            { card = "h", step = 1 }
        )
        expect(records[1].values.value).to.equal(0.75)
    end)

    it("merges config into the ctx and keeps config-only keys", function()
        reset()
        am.observe({
            am.view("e", metric("echo_metric"), { opponents = { "random" }, n_games = 50 }),
        }, { card = "handle", step = 60 })
        local ctx = last_ctx("echo_metric")
        expect(ctx.n_games).to.equal(50)
        expect(ctx.opponents[1]).to.equal("random")
        expect(ctx.card).to.equal("handle")
        expect(ctx.step).to.equal(60)
    end)

    it("lets shared_ctx win over a colliding config key", function()
        reset()
        -- A hand-built view can carry keys `am.view` would refuse, which
        -- is exactly the collision the merge rule has to resolve.
        am.observe({
            {
                view_id = "e",
                metric = metric("echo_metric"),
                config = { card = "stale", seed = 7 },
            },
        }, { card = "fresh", step = 60 })
        local ctx = last_ctx("echo_metric")
        expect(ctx.card).to.equal("fresh")
        expect(ctx.step).to.equal(60)
        expect(ctx.seed).to.equal(7)
    end)

    it("does not leak the merged ctx back into the view config", function()
        reset()
        local v = am.view("e", metric("echo_metric"), { n_games = 50 })
        am.observe({ v }, { card = "handle", step = 60 })
        expect(v.config.card).to.equal(nil)
        expect(v.config.step).to.equal(nil)
    end)

    it("turns a raising metric into an ErrorRecord carrying view_id and message", function()
        reset()
        local records = am.observe(
            { am.view("boom", metric("boom_metric"), {}) },
            { card = "h", step = 5 }
        )
        expect(#records).to.equal(1)
        expect(records[1].view_id).to.equal("boom")
        expect(records[1].step).to.equal(5)
        expect(records[1].values).to.equal(nil)
        expect(type(records[1].error)).to.equal("string")
        expect(records[1].error:find("exploded") ~= nil).to.equal(true)
    end)

    it("keeps evaluating the remaining views after one fails", function()
        reset()
        local records = am.observe({
            am.view("a", metric("table_metric"), {}),
            am.view("boom", metric("boom_metric"), {}),
            am.view("b", metric("scalar_metric"), {}),
        }, { card = "h", step = 5 })
        expect(#records).to.equal(3)
        expect(records[1].values.win_rate).to.equal(0.62)
        expect(records[2].error ~= nil).to.equal(true)
        expect(records[3].values.value).to.equal(0.75)
    end)

    it("turns a metric that cannot resolve its work into an ErrorRecord, not a raise", function()
        reset()
        local records = am.observe(
            { am.view("x", metric("no_such_metric"), {}) },
            { card = "h", step = 5 }
        )
        expect(records[1].error:find("no_such_metric") ~= nil).to.equal(true)
    end)

    it("turns a nil metric result into an ErrorRecord naming the view", function()
        reset()
        local records = am.observe(
            { am.view("n", metric("nil_metric"), {}) },
            { card = "h", step = 5 }
        )
        expect(records[1].error:find("nil") ~= nil).to.equal(true)
        expect(records[1].error:find("'n'") ~= nil).to.equal(true)
    end)

    it("raises loudly on a duplicate view_id, before any metric runs", function()
        reset()
        local ok, err = pcall(am.observe, {
            am.view("level", metric("table_metric"), {}),
            am.view("level", metric("scalar_metric"), {}),
        }, { card = "h", step = 1 })
        expect(ok).to.equal(false)
        expect(err:find("duplicate") ~= nil).to.equal(true)
        expect(err:find("level") ~= nil).to.equal(true)
        expect(#EVAL_CALLS).to.equal(0)
    end)

    it("raises when shared_ctx has no numeric step", function()
        reset()
        local ok, err = pcall(
            am.observe,
            { am.view("a", metric("table_metric"), {}) },
            { card = "h" }
        )
        expect(ok).to.equal(false)
        expect(err:find("step") ~= nil).to.equal(true)
    end)

    it("raises on a malformed view entry", function()
        reset()
        expect(pcall(am.observe, { "not a view" }, { step = 1 })).to.equal(false)
        expect(pcall(am.observe, { { metric = "table_metric" } }, { step = 1 })).to.equal(false)
        expect(pcall(am.observe, { { view_id = "a" } }, { step = 1 })).to.equal(false)
    end)

    it("returns an empty array for an empty view list", function()
        reset()
        local records = am.observe({}, { card = "h", step = 1 })
        expect(#records).to.equal(0)
    end)
end)

describe("anymetric.run_log", function()
    it("accumulates records across fires in append order", function()
        reset()
        local log = am.run_log.new()
        log:append(
            am.observe({ am.view("a", metric("scalar_metric"), {}) }, { card = "h", step = 60 })
        )
        log:append(
            am.observe({ am.view("a", metric("scalar_metric"), {}) }, { card = "h", step = 120 })
        )
        local all = log:all()
        expect(#all).to.equal(2)
        expect(all[1].step).to.equal(60)
        expect(all[2].step).to.equal(120)
    end)

    it("hands out a copy so a caller cannot shorten the log", function()
        reset()
        local log = am.run_log.new()
        log:append({ { view_id = "a", step = 1 } })
        local all = log:all()
        table.remove(all)
        expect(#all).to.equal(0)
        expect(#log:all()).to.equal(1)
    end)

    it("appending an empty batch is a no-op", function()
        reset()
        local log = am.run_log.new()
        log:append({})
        expect(#log:all()).to.equal(0)
    end)

    it("rejects a non-table batch or a non-table record", function()
        reset()
        local log = am.run_log.new()
        expect(pcall(log.append, log, "nope")).to.equal(false)
        expect(pcall(log.append, log, { "nope" })).to.equal(false)
    end)
end)

describe("anymetric.judgment.threshold", function()
    local function level_records(ci_lower)
        return {
            {
                step = 60,
                view_id = "level",
                values = { win_rate = 0.6, ci_lower = ci_lower },
            },
        }
    end

    it("breaks once the field reaches the threshold", function()
        reset()
        local judge = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })
        local d = judge(level_records(0.61))
        expect(d.action).to.equal("break")
        expect(d.reason:find("ci_lower") ~= nil).to.equal(true)
    end)

    it("continues while the field is below the threshold", function()
        reset()
        local judge = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })
        local d = judge(level_records(0.41))
        expect(d.action).to.equal("continue")
        expect(d.reason:find("0.4100") ~= nil).to.equal(true)
    end)

    it("dispatches on the op literal", function()
        reset()
        local recs = level_records(0.50)
        expect(
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.50 })(
                recs
            ).action
        ).to.equal("break")
        expect(
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">", value = 0.50 })(
                recs
            ).action
        ).to.equal("continue")
        expect(
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = "<=", value = 0.50 })(
                recs
            ).action
        ).to.equal("break")
        expect(
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = "<", value = 0.50 })(
                recs
            ).action
        ).to.equal("continue")
    end)

    it("continues when the target view produced an ErrorRecord", function()
        reset()
        local judge =
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.1 })
        local d = judge({ { step = 60, view_id = "level", error = "metric exploded on purpose" } })
        expect(d.action).to.equal("continue")
        expect(d.reason:find("errored") ~= nil).to.equal(true)
        expect(d.reason:find("exploded") ~= nil).to.equal(true)
    end)

    it("continues when the target view produced no record at all", function()
        reset()
        local judge =
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.1 })
        local d = judge({ { step = 60, view_id = "trickiness", values = { value = 0.9 } } })
        expect(d.action).to.equal("continue")
        expect(d.reason:find("no record") ~= nil).to.equal(true)
    end)

    it("continues when the field is absent or not a number", function()
        reset()
        local judge =
            am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.1 })
        local absent = judge({ { step = 1, view_id = "level", values = { win_rate = 0.9 } } })
        expect(absent.action).to.equal("continue")
        local wrong = judge({ { step = 1, view_id = "level", values = { ci_lower = "high" } } })
        expect(wrong.action).to.equal("continue")
    end)

    it("reads only the view it was bound to, even when foreign records are hostile", function()
        reset()
        -- Both foreign records raise on any non-raw field read, so the
        -- judgment completing at all proves it never looked at them —
        -- the strength gate stays decoupled from the personality axis.
        local records = {
            booby_record("sd_teacher"),
            { step = 60, view_id = "level", values = { ci_lower = 0.61 } },
            booby_record("trickiness"),
        }
        local judge = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })
        local d = judge(records)
        expect(d.action).to.equal("break")
        expect(d.reason:find("sd_teacher") == nil).to.equal(true)
        expect(d.reason:find("trickiness") == nil).to.equal(true)
    end)

    it("does not mutate the records it judges", function()
        reset()
        local records = level_records(0.61)
        am.judgment.threshold({ view_id = "level", field = "ci_lower", op = ">=", value = 0.55 })(
            records
        )
        expect(#records).to.equal(1)
        expect(records[1].values.ci_lower).to.equal(0.61)
        expect(records[1].action).to.equal(nil)
    end)

    it("judges the most recent record when a whole run log is passed", function()
        reset()
        local judge = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })
        local d = judge({
            { step = 60, view_id = "level", values = { ci_lower = 0.10 } },
            { step = 120, view_id = "level", values = { ci_lower = 0.80 } },
        })
        expect(d.action).to.equal("break")
    end)

    it("rejects an unknown op and a malformed spec at construction time", function()
        reset()
        local ok, err = pcall(
            am.judgment.threshold,
            { view_id = "level", field = "ci_lower", op = "~=", value = 0.5 }
        )
        expect(ok).to.equal(false)
        expect(err:find("op") ~= nil).to.equal(true)
        expect(pcall(am.judgment.threshold, { field = "ci_lower", op = ">=", value = 0.5 })).to.equal(
            false
        )
        expect(pcall(am.judgment.threshold, { view_id = "level", op = ">=", value = 0.5 })).to.equal(
            false
        )
        expect(pcall(am.judgment.threshold, { view_id = "level", field = "ci_lower", op = ">=" })).to.equal(
            false
        )
        expect(pcall(am.judgment.threshold, "nope")).to.equal(false)
    end)
end)

describe("anymetric.judgment.threshold Decision.meta absence", function()
    -- The threshold judgment must not carry `meta` on either arm,
    -- because downstream callers (the trainer hook adapter, log lines,
    -- collection helpers) read the same {action, reason} shape they
    -- always did. `meta` is opt-in for band / staged only.
    it("leaves Decision.meta absent on break", function()
        reset()
        local d = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })({ { step = 60, view_id = "level", values = { ci_lower = 0.61 } } })
        expect(d.action).to.equal("break")
        expect(d.meta).to.equal(nil)
    end)

    it("leaves Decision.meta absent on continue", function()
        reset()
        local d = am.judgment.threshold({
            view_id = "level",
            field = "ci_lower",
            op = ">=",
            value = 0.55,
        })({ { step = 60, view_id = "level", values = { ci_lower = 0.41 } } })
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
    end)
end)

describe("anymetric.judgment.band", function()
    local function level_records(ci_lower)
        return {
            {
                step = 180,
                view_id = "level",
                values = { ci_lower = ci_lower, win_rate = 0.9 },
            },
        }
    end

    it("harvests when the field falls inside [lo, hi] (label carried on meta)", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        local d = judge(level_records(0.70))
        expect(d.action).to.equal("harvest")
        expect(d.reason:find("mid") ~= nil).to.equal(true)
        expect(type(d.meta)).to.equal("table")
        expect(d.meta.label).to.equal("mid")
        expect(d.meta.step).to.equal(180)
        expect(d.meta.values.ci_lower).to.equal(0.70)
        expect(d.meta.values.win_rate).to.equal(0.9)
    end)

    it("harvests on both endpoints (both inclusive)", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        expect(judge(level_records(0.55)).action).to.equal("harvest")
        expect(judge(level_records(0.85)).action).to.equal("harvest")
    end)

    it("continues (no meta) when the field is below lo", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        local d = judge(level_records(0.40))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("below") ~= nil).to.equal(true)
    end)

    it("breaks (no meta) when the field is above hi", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        local d = judge(level_records(0.99))
        expect(d.action).to.equal("break")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("above") ~= nil).to.equal(true)
    end)

    it("continues (no meta) when the target view produced no record", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
        })
        local d = judge({ { step = 60, view_id = "trickiness", values = { value = 0.9 } } })
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("no record") ~= nil).to.equal(true)
    end)

    it("continues (no meta) on an ErrorRecord", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.10,
            hi = 0.30,
        })
        local d = judge({ { step = 60, view_id = "level", error = "boom" } })
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("errored") ~= nil).to.equal(true)
    end)

    it("continues when the field is absent or non-numeric", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.10,
            hi = 0.30,
        })
        expect(judge({ { step = 1, view_id = "level", values = { win_rate = 0.9 } } }).action).to.equal(
            "continue"
        )
        expect(judge({ { step = 1, view_id = "level", values = { ci_lower = "hi" } } }).action).to.equal(
            "continue"
        )
    end)

    it("omits meta.label when label is not supplied", function()
        reset()
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
        })
        local d = judge(level_records(0.70))
        expect(d.action).to.equal("harvest")
        expect(type(d.meta)).to.equal("table")
        expect(d.meta.label).to.equal(nil)
        expect(d.meta.step).to.equal(180)
    end)

    it("rejects lo > hi at construction time", function()
        reset()
        local ok, err = pcall(am.judgment.band, {
            view_id = "level",
            field = "ci_lower",
            lo = 0.85,
            hi = 0.55,
        })
        expect(ok).to.equal(false)
        expect(err:find("lo") ~= nil and err:find("hi") ~= nil).to.equal(true)
    end)

    it("rejects malformed opts at construction time", function()
        reset()
        expect(pcall(am.judgment.band, "nope")).to.equal(false)
        expect(pcall(am.judgment.band, { field = "ci_lower", lo = 0.1, hi = 0.2 })).to.equal(false)
        expect(pcall(am.judgment.band, { view_id = "level", lo = 0.1, hi = 0.2 })).to.equal(false)
        expect(pcall(am.judgment.band, { view_id = "level", field = "x", hi = 0.2 })).to.equal(
            false
        )
        expect(pcall(am.judgment.band, { view_id = "level", field = "x", lo = 0.1 })).to.equal(
            false
        )
        expect(pcall(am.judgment.band, {
            view_id = "level",
            field = "x",
            lo = 0.1,
            hi = 0.2,
            label = 7,
        })).to.equal(false)
    end)

    it("reads only the view it was bound to, even when foreign records are hostile", function()
        reset()
        local records = {
            booby_record("sd_teacher"),
            { step = 180, view_id = "level", values = { ci_lower = 0.70 } },
            booby_record("trickiness"),
        }
        local judge = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        local d = judge(records)
        expect(d.action).to.equal("harvest")
        expect(d.meta.label).to.equal("mid")
        expect(d.reason:find("sd_teacher") == nil).to.equal(true)
        expect(d.reason:find("trickiness") == nil).to.equal(true)
    end)
end)

describe("anymetric.judgment.staged", function()
    local function level_records(ci_lower)
        return {
            {
                step = 240,
                view_id = "level",
                values = { ci_lower = ci_lower, win_rate = 0.9 },
            },
        }
    end

    local function default_bands()
        return {
            { lo = 0.10, hi = 0.30, label = "weak" },
            { lo = 0.55, hi = 0.85, label = "mid" },
            { lo = 0.85 + 1e-9, hi = 0.98, label = "strong" },
        }
    end

    it("harvests the matching band and copies its label into meta", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        local weak = judge(level_records(0.20))
        expect(weak.action).to.equal("harvest")
        expect(weak.meta.label).to.equal("weak")
        expect(weak.meta.step).to.equal(240)
        local mid = judge(level_records(0.70))
        expect(mid.action).to.equal("harvest")
        expect(mid.meta.label).to.equal("mid")
        local strong = judge(level_records(0.90))
        expect(strong.action).to.equal("harvest")
        expect(strong.meta.label).to.equal("strong")
    end)

    it("continues below the lowest band's lo", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        local d = judge(level_records(0.05))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
    end)

    it("continues between two bands (in the gap)", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        local d = judge(level_records(0.45))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
    end)

    it("breaks once the value rises above the topmost band's hi", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        local d = judge(level_records(0.999))
        expect(d.action).to.equal("break")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("above") ~= nil).to.equal(true)
    end)

    it("continues (no meta) on missing record / ErrorRecord / non-numeric field", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        expect(judge({ { step = 1, view_id = "other", values = { value = 0.9 } } }).action).to.equal(
            "continue"
        )
        expect(judge({ { step = 1, view_id = "level", error = "boom" } }).action).to.equal(
            "continue"
        )
        expect(judge({ { step = 1, view_id = "level", values = { ci_lower = "hi" } } }).action).to.equal(
            "continue"
        )
    end)

    it("rejects overlapping / non-disjoint bands loudly", function()
        reset()
        local ok, err = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {
                { lo = 0.10, hi = 0.50, label = "a" },
                { lo = 0.40, hi = 0.80, label = "b" },
            },
        })
        expect(ok).to.equal(false)
        expect(err:find("disjoint") ~= nil).to.equal(true)
    end)

    it(
        "rejects touching bands (equal boundary) loudly — label would be non-deterministic",
        function()
            reset()
            local ok = pcall(am.judgment.staged, {
                view_id = "level",
                field = "ci_lower",
                bands = {
                    { lo = 0.10, hi = 0.50, label = "a" },
                    { lo = 0.50, hi = 0.80, label = "b" },
                },
            })
            expect(ok).to.equal(false)
        end
    )

    it("rejects descending bands loudly", function()
        reset()
        local ok = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {
                { lo = 0.60, hi = 0.90, label = "hi" },
                { lo = 0.10, hi = 0.30, label = "lo" },
            },
        })
        expect(ok).to.equal(false)
    end)

    it("rejects an empty bands list loudly", function()
        reset()
        local ok, err = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {},
        })
        expect(ok).to.equal(false)
        expect(err:find("empty") ~= nil).to.equal(true)
    end)

    it("rejects a band with lo > hi", function()
        reset()
        local ok = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.85, hi = 0.55, label = "bad" } },
        })
        expect(ok).to.equal(false)
    end)

    it("rejects malformed opts / missing bands / non-table band", function()
        reset()
        expect(pcall(am.judgment.staged, "nope")).to.equal(false)
        expect(pcall(am.judgment.staged, { field = "x", bands = { { lo = 0, hi = 1 } } })).to.equal(
            false
        )
        expect(pcall(am.judgment.staged, { view_id = "v", bands = { { lo = 0, hi = 1 } } })).to.equal(
            false
        )
        expect(pcall(am.judgment.staged, { view_id = "v", field = "x" })).to.equal(false)
        expect(pcall(am.judgment.staged, { view_id = "v", field = "x", bands = { "not a band" } })).to.equal(
            false
        )
    end)

    it("a one-band staged judgment is equivalent to a plain band judgment", function()
        reset()
        local staged = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.55, hi = 0.85, label = "mid" } },
        })
        local band = am.judgment.band({
            view_id = "level",
            field = "ci_lower",
            lo = 0.55,
            hi = 0.85,
            label = "mid",
        })
        for _, v in ipairs({ 0.10, 0.55, 0.70, 0.85, 0.99 }) do
            local s = staged(level_records(v))
            local b = band(level_records(v))
            expect(s.action).to.equal(b.action)
            expect(s.meta and s.meta.label or nil).to.equal(b.meta and b.meta.label or nil)
        end
    end)

    it("is stateless: two harvest calls on the same band both return harvest", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        expect(judge(level_records(0.70)).action).to.equal("harvest")
        expect(judge(level_records(0.72)).action).to.equal("harvest")
    end)

    it("reads only the view it was bound to, even when foreign records are hostile", function()
        reset()
        local records = {
            booby_record("sd_teacher"),
            { step = 240, view_id = "level", values = { ci_lower = 0.90 } },
            booby_record("trickiness"),
        }
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = default_bands(),
        })
        local d = judge(records)
        expect(d.action).to.equal("harvest")
        expect(d.meta.label).to.equal("strong")
        expect(d.reason:find("sd_teacher") == nil).to.equal(true)
        expect(d.reason:find("trickiness") == nil).to.equal(true)
    end)
end)

describe("anymetric.judgment.staged band require", function()
    -- A band may make its harvest conditional on a second view of the
    -- same fire. The cases below fix the whole contract: the floor is
    -- inclusive, a reading under it continues, and a requirement that
    -- could not be read at all continues too — a measurement gap is not
    -- a pass.

    --- One fire's records. `trickiness` is the second view; pass nil to
    --- leave it out, "error" to make it an ErrorRecord, or a string to
    --- put a non-numeric value where the number should be.
    local function fire(ci_lower, trickiness)
        local records = {
            {
                step = 120,
                view_id = "level",
                values = { ci_lower = ci_lower, win_rate = 0.93 },
            },
        }
        if trickiness == nil then
            return records
        end
        if trickiness == "error" then
            records[#records + 1] = { step = 120, view_id = "trickiness", error = "boom" }
            return records
        end
        records[#records + 1] = {
            step = 120,
            view_id = "trickiness",
            values = { value = trickiness, raw_mean = 1.2 },
        }
        return records
    end

    --- The design's three-band schedule: only the middle band carries a
    --- requirement, so the same judgment covers the mixed case.
    local function mixed_bands()
        return {
            { lo = 0.03, hi = 0.20, label = "weak_v2" },
            {
                lo = 0.60,
                hi = 0.85,
                label = "mid_v2",
                require = { view_id = "trickiness", field = "value", min = 0.57 },
            },
            { lo = 0.851, hi = 0.98, label = "strong_v2" },
        }
    end

    local function mixed_judge()
        return am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = mixed_bands(),
        })
    end

    it("harvests when the required view clears the floor", function()
        reset()
        local d = mixed_judge()(fire(0.70, 0.795))
        expect(d.action).to.equal("harvest")
        expect(d.meta.label).to.equal("mid_v2")
        expect(d.meta.step).to.equal(120)
        expect(d.meta.values.ci_lower).to.equal(0.70)
    end)

    it("carries the satisfied requirement in meta.require", function()
        reset()
        local d = mixed_judge()(fire(0.70, 0.795))
        expect(type(d.meta.require)).to.equal("table")
        expect(d.meta.require.view_id).to.equal("trickiness")
        expect(d.meta.require.field).to.equal("value")
        expect(d.meta.require.value).to.equal(0.795)
        -- The reason names the second condition too, so a run log line
        -- shows why this band was allowed through.
        expect(d.reason:find("trickiness") ~= nil).to.equal(true)
        expect(d.reason:find("0.5700") ~= nil).to.equal(true)
    end)

    it("treats the floor as inclusive (value == min harvests)", function()
        reset()
        local d = mixed_judge()(fire(0.70, 0.57))
        expect(d.action).to.equal("harvest")
        expect(d.meta.require.value).to.equal(0.57)
    end)

    it("continues when the required view is below the floor, naming both numbers", function()
        reset()
        local d = mixed_judge()(fire(0.70, 0.354))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("band hit %(mid_v2%)") ~= nil).to.equal(true)
        expect(d.reason:find("below required min=0.5700") ~= nil).to.equal(true)
        expect(d.reason:find("0.3540") ~= nil).to.equal(true)
    end)

    it("continues when the required view produced no record at all", function()
        reset()
        local d = mixed_judge()(fire(0.70, nil))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("could not be read") ~= nil).to.equal(true)
        expect(d.reason:find("no record") ~= nil).to.equal(true)
    end)

    it("continues when the required view errored", function()
        reset()
        local d = mixed_judge()(fire(0.70, "error"))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("could not be read") ~= nil).to.equal(true)
        expect(d.reason:find("errored") ~= nil).to.equal(true)
    end)

    it("continues when the required field is present but not a number", function()
        reset()
        local d = mixed_judge()(fire(0.70, "very tricky"))
        expect(d.action).to.equal("continue")
        expect(d.meta).to.equal(nil)
        expect(d.reason:find("could not be read") ~= nil).to.equal(true)
        expect(d.reason:find("no numeric field") ~= nil).to.equal(true)
    end)

    it("continues when the required field name is absent from the values table", function()
        reset()
        local judge = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = {
                {
                    lo = 0.60,
                    hi = 0.85,
                    label = "mid_v2",
                    require = { view_id = "trickiness", field = "raw_value", min = 0.57 },
                },
            },
        })
        local d = judge(fire(0.70, 0.795))
        expect(d.action).to.equal("continue")
        expect(d.reason:find("raw_value") ~= nil).to.equal(true)
    end)

    it("leaves bands without a require untouched in the same judgment", function()
        reset()
        -- The weak band has no requirement, so it harvests even on a
        -- fire where the second view is missing entirely.
        local weak = mixed_judge()(fire(0.10, nil))
        expect(weak.action).to.equal("harvest")
        expect(weak.meta.label).to.equal("weak_v2")
        expect(weak.meta.require).to.equal(nil)
        -- ... and on a fire whose second view would have failed the
        -- mid band's floor.
        local weak_again = mixed_judge()(fire(0.10, 0.01))
        expect(weak_again.action).to.equal("harvest")
        expect(weak_again.meta.require).to.equal(nil)
    end)

    it("still breaks above the top band regardless of the requirement", function()
        reset()
        local d = mixed_judge()(fire(0.999, 0.01))
        expect(d.action).to.equal("break")
        expect(d.meta).to.equal(nil)
    end)

    it("reads only the primary view and the one the band named", function()
        reset()
        -- sd_teacher raises on any field read: the judgment completing
        -- proves the opt-in requirement widened the read by exactly one
        -- named view and no further.
        local records = fire(0.70, 0.795)
        table.insert(records, 1, booby_record("sd_teacher"))
        local d = mixed_judge()(records)
        expect(d.action).to.equal("harvest")
        expect(d.meta.require.value).to.equal(0.795)
        expect(d.reason:find("sd_teacher") == nil).to.equal(true)
    end)

    it("rejects a non-table require at construction time", function()
        reset()
        local ok, err = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.6, hi = 0.85, label = "mid_v2", require = "trickiness" } },
        })
        expect(ok).to.equal(false)
        expect(err:find("require") ~= nil).to.equal(true)
        expect(pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.6, hi = 0.85, require = 7 } },
        })).to.equal(false)
    end)

    it("rejects a require without a non-empty string view_id", function()
        reset()
        local ok, err = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.6, hi = 0.85, require = { field = "value", min = 0.57 } } },
        })
        expect(ok).to.equal(false)
        expect(err:find("view_id") ~= nil).to.equal(true)
        expect(pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {
                { lo = 0.6, hi = 0.85, require = { view_id = "", field = "value", min = 0.5 } },
            },
        })).to.equal(false)
        expect(pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {
                { lo = 0.6, hi = 0.85, require = { view_id = 7, field = "value", min = 0.5 } },
            },
        })).to.equal(false)
    end)

    it("rejects a require without a non-empty string field", function()
        reset()
        local ok, err = pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = { { lo = 0.6, hi = 0.85, require = { view_id = "trickiness", min = 0.57 } } },
        })
        expect(ok).to.equal(false)
        expect(err:find("field") ~= nil).to.equal(true)
        expect(pcall(am.judgment.staged, {
            view_id = "level",
            field = "ci_lower",
            bands = {
                {
                    lo = 0.6,
                    hi = 0.85,
                    require = { view_id = "trickiness", field = "", min = 0.5 },
                },
            },
        })).to.equal(false)
    end)

    it("rejects a non-finite or non-numeric require min", function()
        reset()
        local function with_min(min)
            return pcall(am.judgment.staged, {
                view_id = "level",
                field = "ci_lower",
                bands = {
                    {
                        lo = 0.6,
                        hi = 0.85,
                        require = { view_id = "trickiness", field = "value", min = min },
                    },
                },
            })
        end
        local ok, err = with_min("0.57")
        expect(ok).to.equal(false)
        expect(err:find("min") ~= nil).to.equal(true)
        expect(with_min(nil)).to.equal(false)
        expect(with_min(math.huge)).to.equal(false)
        expect(with_min(-math.huge)).to.equal(false)
        expect(with_min(0 / 0)).to.equal(false)
    end)

    it("refuses a require on the single-view band() judgment", function()
        reset()
        local ok, err = pcall(am.judgment.band, {
            view_id = "level",
            field = "ci_lower",
            lo = 0.60,
            hi = 0.85,
            label = "mid_v2",
            require = { view_id = "trickiness", field = "value", min = 0.57 },
        })
        expect(ok).to.equal(false)
        -- Loud rather than ignored, and the message points at the path
        -- that can honour it.
        expect(err:find("require") ~= nil).to.equal(true)
        expect(err:find("staged") ~= nil).to.equal(true)
    end)

    it("leaves a require-free staged judgment byte-identical in behaviour", function()
        reset()
        local plain = am.judgment.staged({
            view_id = "level",
            field = "ci_lower",
            bands = {
                { lo = 0.03, hi = 0.20, label = "weak_v2" },
                { lo = 0.60, hi = 0.85, label = "mid_v2" },
                { lo = 0.851, hi = 0.98, label = "strong_v2" },
            },
        })
        -- The second view is present and would have failed any floor;
        -- with no require declared it is not read and cannot matter.
        local d = plain(fire(0.70, 0.01))
        expect(d.action).to.equal("harvest")
        expect(d.meta.label).to.equal("mid_v2")
        expect(d.meta.require).to.equal(nil)
        expect(d.reason:find("required") == nil).to.equal(true)
    end)
end)

describe("anymetric.judgment.never_break", function()
    it("continues even on records a threshold would break on", function()
        reset()
        local d = am.judgment.never_break()({
            { step = 60, view_id = "level", values = { ci_lower = 0.99 } },
        })
        expect(d.action).to.equal("continue")
        expect(type(d.reason)).to.equal("string")
    end)

    it("continues on an empty record list", function()
        reset()
        expect(am.judgment.never_break()({}).action).to.equal("continue")
    end)
end)

describe("anymetric.to_hook_action", function()
    it("projects break and continue onto the hook ABI verbatim", function()
        reset()
        expect(am.to_hook_action({ action = "break", reason = "done" })).to.equal("break")
        expect(am.to_hook_action({ action = "continue", reason = "keep going" })).to.equal(
            "continue"
        )
    end)

    it("projects harvest onto continue and appends a marker record", function()
        reset()
        local log = am.run_log.new()
        local action = am.to_hook_action({ action = "harvest", reason = "band 0.45-0.55 hit" }, log)
        expect(action).to.equal("continue")
        local all = log:all()
        expect(#all).to.equal(1)
        expect(all[1].harvest).to.equal(true)
        expect(all[1].reason).to.equal("band 0.45-0.55 hit")
    end)

    it("keeps earlier records when a harvest marker is appended", function()
        reset()
        local log = am.run_log.new()
        log:append(
            am.observe({ am.view("a", metric("scalar_metric"), {}) }, { card = "h", step = 60 })
        )
        am.to_hook_action({ action = "harvest", reason = "keep this bake" }, log)
        local all = log:all()
        expect(#all).to.equal(2)
        expect(all[1].view_id).to.equal("a")
        expect(all[2].harvest).to.equal(true)
    end)

    it("raises when a harvest decision has no run log to write to", function()
        reset()
        local ok, err = pcall(am.to_hook_action, { action = "harvest", reason = "x" })
        expect(ok).to.equal(false)
        expect(err:find("run_log") ~= nil).to.equal(true)
    end)

    it("raises on an unknown action or a malformed decision", function()
        reset()
        local ok, err = pcall(am.to_hook_action, { action = "halt", reason = "x" })
        expect(ok).to.equal(false)
        expect(err:find("halt") ~= nil).to.equal(true)
        expect(pcall(am.to_hook_action, "break")).to.equal(false)
        expect(pcall(am.to_hook_action, {})).to.equal(false)
    end)

    it("merges Decision.meta into the harvest marker record", function()
        reset()
        local log = am.run_log.new()
        local meta = { label = "mid", step = 180, values = { ci_lower = 0.70 } }
        local action = am.to_hook_action(
            { action = "harvest", reason = "band [0.55, 0.85] hit (mid)", meta = meta },
            log
        )
        expect(action).to.equal("continue")
        local all = log:all()
        expect(#all).to.equal(1)
        expect(all[1].harvest).to.equal(true)
        expect(all[1].reason).to.equal("band [0.55, 0.85] hit (mid)")
        expect(all[1].label).to.equal("mid")
        expect(all[1].step).to.equal(180)
        expect(all[1].values.ci_lower).to.equal(0.70)
    end)

    it("meta is copy-only: mutating the marker does not leak into the source table", function()
        reset()
        local log = am.run_log.new()
        local meta = { label = "mid", step = 180 }
        am.to_hook_action({ action = "harvest", reason = "x", meta = meta }, log)
        -- The marker got its own label copy — reading it back must not
        -- reflect a later change to `meta`.
        meta.label = "mutated"
        expect(log:all()[1].label).to.equal("mid")
    end)

    it("harvest without meta keeps the previous {harvest, reason} shape byte-for-byte", function()
        reset()
        local log = am.run_log.new()
        am.to_hook_action({ action = "harvest", reason = "no meta here" }, log)
        local marker = log:all()[1]
        expect(marker.harvest).to.equal(true)
        expect(marker.reason).to.equal("no meta here")
        expect(marker.label).to.equal(nil)
        expect(marker.step).to.equal(nil)
        expect(marker.values).to.equal(nil)
        -- Count keys: exactly {harvest, reason}, nothing else.
        local count = 0
        for _ in pairs(marker) do
            count = count + 1
        end
        expect(count).to.equal(2)
    end)

    it("refuses to fold in a non-table meta", function()
        reset()
        local log = am.run_log.new()
        local ok, err = pcall(am.to_hook_action, {
            action = "harvest",
            reason = "x",
            meta = "not a table",
        }, log)
        expect(ok).to.equal(false)
        expect(err:find("meta") ~= nil).to.equal(true)
    end)

    it("built-in fields (harvest, reason) always win over meta keys of the same name", function()
        reset()
        local log = am.run_log.new()
        am.to_hook_action({
            action = "harvest",
            reason = "authoritative",
            meta = { reason = "meta wins?", harvest = false, label = "mid" },
        }, log)
        local marker = log:all()[1]
        expect(marker.harvest).to.equal(true)
        expect(marker.reason).to.equal("authoritative")
        expect(marker.label).to.equal("mid")
    end)
end)

describe("anymetric.log_line", function()
    it("renders step, every view id and its values on one line", function()
        reset()
        local records = am.observe({
            am.view("level", metric("table_metric"), {}),
            am.view("trickiness", metric("scalar_metric"), {}),
        }, { card = "h", step = 120 })
        local line = am.log_line(records)
        expect(type(line)).to.equal("string")
        expect(line:find("step=120") ~= nil).to.equal(true)
        expect(line:find("level") ~= nil).to.equal(true)
        expect(line:find("win_rate=0.6200") ~= nil).to.equal(true)
        expect(line:find("trickiness") ~= nil).to.equal(true)
        expect(line:find("value=0.7500") ~= nil).to.equal(true)
        expect(line:find("\n") == nil).to.equal(true)
    end)

    it("orders the fields of a view deterministically", function()
        reset()
        local records = am.observe(
            { am.view("level", metric("table_metric"), {}) },
            { card = "h", step = 1 }
        )
        local line = am.log_line(records)
        expect(line:find("ci_lower") < line:find("ci_upper")).to.equal(true)
        expect(line:find("ci_upper") < line:find("win_rate")).to.equal(true)
    end)

    it("marks an ErrorRecord instead of pretending it has values", function()
        reset()
        local records = am.observe({
            am.view("boom", metric("boom_metric"), {}),
            am.view("ok", metric("scalar_metric"), {}),
        }, { card = "h", step = 7 })
        local line = am.log_line(records)
        expect(line:find("ERROR") ~= nil).to.equal(true)
        expect(line:find("exploded") ~= nil).to.equal(true)
        expect(line:find("value=0.7500") ~= nil).to.equal(true)
    end)

    it("handles an empty record list", function()
        reset()
        expect(am.log_line({}):find("no records") ~= nil).to.equal(true)
    end)

    it("rejects a non-table argument", function()
        reset()
        expect(pcall(am.log_line, nil)).to.equal(false)
    end)
end)
