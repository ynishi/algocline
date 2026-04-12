--- alc.eval integration tests (mlua-lspec)
---
--- Tests alc.eval() with real evalframe loaded. Exercises the full path:
--- scenario resolution → provider wiring → evalframe suite → report.
--- Requires evalframe installed at ~/.algocline/packages/evalframe/.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ── 1. Package path: make evalframe requireable ──────────────
local home = os.getenv("HOME") or os.getenv("USERPROFILE") or ""
local pkg_root = home .. "/.algocline/packages"
package.path = pkg_root .. "/?.lua;"
    .. pkg_root .. "/?/init.lua;"
    .. package.path

-- ── 2. Mock std global (evalframe.std requires mlua-batteries) ──
_G.std = {
    json = {
        decode = function(str)
            -- Minimal: load as Lua expression (sufficient for test data)
            local fn = load("return " .. str)
            if fn then
                local ok, result = pcall(fn)
                if ok then return result end
            end
            error("json decode error")
        end,
        encode = function(val)
            if val == nil then return "null" end
            if type(val) == "string" then return '"' .. val .. '"' end
            if type(val) == "number" or type(val) == "boolean" then
                return tostring(val)
            end
            -- Shallow table encoding (sufficient for tests)
            if type(val) == "table" then return "{}" end
            return tostring(val)
        end,
    },
    fs = {
        read = function() return "" end,
        is_file = function() return false end,
    },
    time = {
        now = function() return os.clock() end,
    },
}

-- ── 3. Mock alc global ───────────────────────────────────────
alc = {}

alc.log = function() end

alc.json_encode = function(val)
    if type(val) == "table" then return "{}" end
    return tostring(val)
end

alc.json_decode = function(str)
    local fn = load("return " .. str)
    if fn then
        local ok, result = pcall(fn)
        if ok then return result end
    end
    error("json decode error")
end

alc.llm = function(prompt)
    return "mock llm response"
end

-- Card mock with tracking
local cards = {}
local card_samples = {}
local card_counter = 0

alc.card = {
    create = function(data)
        card_counter = card_counter + 1
        local card_id = "test_card_" .. card_counter
        cards[card_id] = data
        return { card_id = card_id }
    end,
    write_samples = function(card_id, samples)
        card_samples[card_id] = samples
    end,
}

-- State stub
local state_store = {}
alc.state = {
    get = function(key, default)
        local val = state_store[key]
        if val == nil then return default end
        return val
    end,
    set = function(key, value)
        state_store[key] = value
    end,
}

-- ── 4. Load prelude ──────────────────────────────────────────
local prelude_path = os.getenv("PRELUDE_PATH")
    or "crates/algocline-engine/src/prelude.lua"
dofile(prelude_path)

-- ── 5. Register mock strategy ────────────────────────────────
-- The algocline provider does require("mock_strategy").run(ctx)
-- ctx.task = case.input, result is passed to extract_text()
package.loaded["mock_strategy"] = {
    run = function(ctx)
        local answers = {
            ["2+2?"]               = "4",
            ["Capital of France?"] = "Paris",
            ["Color of sky?"]      = "blue",
            ["Fail me"]            = "wrong",
        }
        return { result = answers[ctx.task] or "unknown" }
    end,
}

-- ── 6. Verify evalframe is loadable ──────────────────────────
local ef_ok, ef = pcall(require, "evalframe")
if not ef_ok then
    error("evalframe not found at " .. pkg_root .. ". Install it first.\n" .. tostring(ef))
end

-- ═══════════════════════════════════════════════════════════════
-- Tests
-- ═══════════════════════════════════════════════════════════════

describe("alc.eval integration", function()

    -- ── Simple form: all pass ────────────────────────────────
    describe("simple form", function()
        it("evaluates exact_match — all pass", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                    { input = "Capital of France?", expected = "Paris" },
                },
                graders = { "exact_match" },
            }, "mock_strategy")

            expect(report.aggregated).to.exist()
            expect(report.aggregated.total).to.equal(2)
            expect(report.aggregated.passed).to.equal(2)
            expect(report.aggregated.pass_rate).to.equal(1.0)
            expect(report.summary).to.exist()
        end)

        it("evaluates contains — partial pass", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                    { input = "Fail me", expected = "correct" },
                },
                graders = { "contains" },
            }, "mock_strategy")

            expect(report.aggregated.total).to.equal(2)
            expect(report.aggregated.passed).to.equal(1)
            expect(report.aggregated.pass_rate).to.equal(0.5)
            expect(#report.failures).to.equal(1)
        end)

        it("defaults graders to exact_match when omitted", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                },
            }, "mock_strategy")

            expect(report.aggregated.total).to.equal(1)
            expect(report.aggregated.passed).to.equal(1)
        end)

        it("supports multiple graders", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                },
                graders = { "exact_match", "not_empty" },
            }, "mock_strategy")

            expect(report.aggregated.total).to.equal(1)
            expect(report.aggregated.passed).to.equal(1)
            -- Score should be 1.0 (both graders pass, equal weight)
            expect(report.aggregated.scores.mean).to.equal(1.0)
        end)
    end)

    -- ── Full evalframe form ──────────────────────────────────
    describe("full evalframe form", function()
        it("accepts ef.bind / ef.case in scenario", function()
            local report = alc.eval({
                ef.bind { ef.graders.exact_match },
                cases = {
                    ef.case { input = "2+2?", expected = "4" },
                    ef.case { input = "Color of sky?", expected = "blue" },
                },
            }, "mock_strategy")

            expect(report.aggregated.total).to.equal(2)
            expect(report.aggregated.passed).to.equal(2)
        end)

        it("supports tags in full form", function()
            local report = alc.eval({
                ef.bind { ef.graders.exact_match },
                cases = {
                    ef.case { input = "2+2?", expected = "4", tags = { "math" } },
                    ef.case { input = "Capital of France?", expected = "Paris", tags = { "geo" } },
                },
            }, "mock_strategy")

            expect(report.aggregated.total).to.equal(2)
            expect(report.aggregated.by_tag).to.exist()
            expect(report.aggregated.by_tag.math).to.exist()
            expect(report.aggregated.by_tag.geo).to.exist()
        end)
    end)

    -- ── Report structure ─────────────────────────────────────
    describe("report structure", function()
        it("includes suite name as strategy:scenario", function()
            local report = alc.eval({
                name = "my_test",
                cases = { { input = "2+2?", expected = "4" } },
                graders = { "exact_match" },
            }, "mock_strategy")

            expect(report.name).to.equal("mock_strategy:my_test")
        end)

        it("uses 'inline' when scenario has no name", function()
            local report = alc.eval({
                cases = { { input = "2+2?", expected = "4" } },
            }, "mock_strategy")

            expect(report.name).to.equal("mock_strategy:inline")
        end)

        it("has scores statistics", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                    { input = "Fail me", expected = "correct" },
                },
                graders = { "exact_match" },
            }, "mock_strategy")

            local scores = report.aggregated.scores
            expect(scores.n).to.equal(2)
            expect(scores.mean).to.equal(0.5)
            expect(scores.min).to.equal(0)
            expect(scores.max).to.equal(1)
        end)

        it("has 95% CI", function()
            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                    { input = "Capital of France?", expected = "Paris" },
                    { input = "Color of sky?", expected = "blue" },
                },
                graders = { "exact_match" },
            }, "mock_strategy")

            local ci = report.aggregated.ci_95
            expect(ci).to.exist()
            expect(ci.lower).to.exist()
            expect(ci.upper).to.exist()
            -- All pass → mean=1.0, CI upper clamped to 1.0
            expect(ci.upper).to.equal(1.0)
        end)

        it("results contain per-case detail", function()
            local report = alc.eval({
                cases = { { input = "2+2?", expected = "4" } },
                graders = { "exact_match" },
            }, "mock_strategy")

            expect(#report.results).to.equal(1)
            local r = report.results[1]
            expect(r.score).to.exist()
            expect(r.passed).to.exist()
        end)
    end)

    -- ── Auto-card emission ───────────────────────────────────
    describe("auto_card", function()
        it("emits Card when auto_card=true", function()
            -- Reset card state
            cards = {}
            card_samples = {}
            card_counter = 0

            local report = alc.eval({
                cases = {
                    { input = "2+2?", expected = "4" },
                    { input = "Capital of France?", expected = "Paris" },
                },
                graders = { "exact_match" },
            }, "mock_strategy", { auto_card = true })

            expect(report.card_id).to.exist()
            expect(report.card_id).to.equal("test_card_1")

            -- Verify card was created with correct fields
            local card = cards["test_card_1"]
            expect(card).to.exist()
            expect(card.pkg.name).to.equal("mock_strategy")
            expect(card.stats.pass_rate).to.equal(1.0)
            expect(card.stats.n).to.equal(2)

            -- Verify samples were written (Tier 2)
            local samples = card_samples["test_card_1"]
            expect(samples).to.exist()
            expect(#samples).to.equal(2)
        end)

        it("uses card_pkg override", function()
            cards = {}
            card_samples = {}
            card_counter = 0

            local report = alc.eval({
                cases = { { input = "2+2?", expected = "4" } },
            }, "mock_strategy", { auto_card = true, card_pkg = "custom_pkg" })

            local card = cards["test_card_1"]
            expect(card.pkg.name).to.equal("custom_pkg")
        end)

        it("does not emit Card when auto_card is absent", function()
            cards = {}
            card_counter = 0

            local report = alc.eval({
                cases = { { input = "2+2?", expected = "4" } },
            }, "mock_strategy")

            expect(report.card_id).to.equal(nil)
            expect(card_counter).to.equal(0)
        end)
    end)

    -- ── Named scenario resolution ─────────────────────────────
    describe("named scenario resolution", function()
        it("resolves from require (package.loaded)", function()
            -- Register a scenario module
            package.loaded["test_inline_scenario"] = {
                cases = {
                    { input = "2+2?", expected = "4" },
                },
                graders = { "exact_match" },
            }

            local report = alc.eval("test_inline_scenario", "mock_strategy")

            expect(report.aggregated.total).to.equal(1)
            expect(report.aggregated.passed).to.equal(1)
            expect(report.name).to.equal("mock_strategy:test_inline_scenario")

            package.loaded["test_inline_scenario"] = nil
        end)

        it("resolves from ~/.algocline/scenarios/{name}.lua", function()
            -- math_basic.lua should exist in ~/.algocline/scenarios/
            -- It uses full evalframe form (ef.bind with weighted graders)
            local report = alc.eval("math_basic", "mock_strategy")

            expect(report.aggregated).to.exist()
            expect(report.aggregated.total > 0).to.equal(true)
            expect(report.name).to.equal("mock_strategy:math_basic")
        end)

        it("resolves from direct file path", function()
            local scenario_path = home .. "/.algocline/scenarios/math_basic.lua"
            local report = alc.eval(scenario_path, "mock_strategy")

            expect(report.aggregated).to.exist()
            expect(report.aggregated.total > 0).to.equal(true)
        end)

        it("errors on nonexistent scenario string", function()
            local ok, err = pcall(alc.eval, "nonexistent_scenario_xyz", "mock_strategy")
            expect(ok).to.equal(false)
            local msg = tostring(err)
            expect(msg:find("could not resolve scenario") ~= nil).to.equal(true)
        end)

        it("errors when named scenario resolves to non-table", function()
            package.loaded["returns_number"] = 42

            local ok, err = pcall(alc.eval, "returns_number", "mock_strategy")
            expect(ok).to.equal(false)
            local msg = tostring(err)
            expect(msg:find("resolved to number") ~= nil).to.equal(true)

            package.loaded["returns_number"] = nil
        end)
    end)

    -- ── Error cases ──────────────────────────────────────────
    describe("error handling", function()
        it("errors on unknown grader name", function()
            local ok, err = pcall(alc.eval, {
                cases = { { input = "x", expected = "y" } },
                graders = { "nonexistent_grader" },
            }, "mock_strategy")

            expect(ok).to.equal(false)
            local msg = tostring(err)
            expect(msg:find("unknown grader") ~= nil).to.equal(true)
        end)

        it("errors when scenario resolves to non-table", function()
            -- Register a module that returns a string
            package.loaded["bad_scenario"] = "not a table"

            local ok, err = pcall(alc.eval, "bad_scenario", "mock_strategy")
            expect(ok).to.equal(false)
            local msg = tostring(err)
            expect(msg:find("resolved to string") ~= nil).to.equal(true)

            package.loaded["bad_scenario"] = nil
        end)

        it("strategy error is captured in response, not thrown", function()
            -- algocline provider catches errors via pcall
            package.loaded["broken_strategy"] = { meta = {} }

            local report = alc.eval({
                cases = { { input = "x", expected = "y" } },
                graders = { "exact_match" },
            }, "broken_strategy")

            -- Provider returns empty text on error → grader fails → 0 passed
            expect(report.aggregated.total).to.equal(1)
            expect(report.aggregated.passed).to.equal(0)

            package.loaded["broken_strategy"] = nil
        end)
    end)
end)
