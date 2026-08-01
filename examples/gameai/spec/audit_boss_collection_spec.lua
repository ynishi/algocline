-- spec/audit_boss_collection_spec.lua
--
-- Spec for the `audit_boss_collection.lua` driver. Same pattern the
-- `train_guardian_npc_spec.lua` uses: the whole driver file is loaded
-- once per case against a stubbed host, and the assertions read what it
-- handed to `gameai_metrics.audit_matrix` plus the log lines it emitted.
--
-- Only `gameai_metrics.audit_matrix` is stubbed here — a stand-in
-- module registered under `package.loaded` before the driver requires
-- it. This lets the spec drive the option decoding / default fallback /
-- log-line assembly in isolation without booting the real audit runner
-- (which would try to reach for `alc.nn.card.load_handle` and the
-- metric registry).
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/audit_boss_collection_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

alc.json_decode = alc.json_decode
    or function(text)
        -- The driver does not itself call `alc.json_decode`; the stub is
        -- present so a bystander require inside the driver's dependency
        -- graph does not trip on its absence.
        return text
    end

alc.json_encode = alc.json_encode or function(value)
    return tostring(value)
end

--- Log lines the driver emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- ─── audit_matrix stub ──────────────────────────────────────────────
--
-- The stub records the last `new()` opts, the number of `:run()` and
-- `:save()` calls, and the `path` handed to `:save()`. The report the
-- runner would produce is canned so the driver's summary lines are
-- deterministic.

local NEW_OPTS = nil
local RUN_CALLS = 0
local SAVE_CALLS = 0
local SAVE_PATH = nil
local NEXT_REPORT = nil

local audit_matrix_stub = {}
audit_matrix_stub.DEFAULT_N_GAMES = 200
audit_matrix_stub.DEFAULT_PROMPT_SET_SIZE = 16
audit_matrix_stub.DEFAULT_SEED = 20260731

local Audit = {}
Audit.__index = Audit

function Audit:run()
    RUN_CALLS = RUN_CALLS + 1
    self._report = NEXT_REPORT
    return self._report
end

function Audit:save(path)
    SAVE_CALLS = SAVE_CALLS + 1
    SAVE_PATH = path
end

function audit_matrix_stub.new(opts)
    NEW_OPTS = opts
    return setmetatable({ _report = nil }, Audit)
end

--- Reset every observable stub between cases so a leaked variable
--- from an earlier case cannot mask a fresh assertion.
local function configure()
    LOG_LINES = {}
    NEW_OPTS = nil
    RUN_CALLS = 0
    SAVE_CALLS = 0
    SAVE_PATH = nil
    NEXT_REPORT = {
        per_card = {
            weak = { win_rate = 0.20, ci_lower = 0.15, ci_upper = 0.30, trickiness_norm = 0.10 },
            mid = { win_rate = 0.70, ci_lower = 0.60, ci_upper = 0.80, trickiness_norm = 0.35 },
            strong = { win_rate = 0.92, ci_lower = 0.86, ci_upper = 0.98, trickiness_norm = 0.42 },
        },
        sd_matrix = {
            weak = { weak = 0.0, mid = 0.55, strong = 0.60 },
            mid = { weak = 0.55, mid = 0.0, strong = 0.11 },
            strong = { weak = 0.60, mid = 0.11, strong = 0.0 },
        },
        meta = {
            n_games = 200,
            prompt_set_size = 16,
            seed = 20260731,
            style = "guardian",
        },
    }
end

-- ─── Driver loader ──────────────────────────────────────────────────

--- Drive the script once against the current stubs. `ctx` is a global
--- under the `alc_run` contract, so it is planted as one. Both the
--- driver and its `audit_matrix` require are cleared from
--- `package.loaded` first so the top-level ctx decoding runs again
--- for every case; the audit_matrix stub is re-installed after the
--- clear.
---@param overrides table ctx fields for this case
---@return table|nil result the script's return value on success, nil
---                     when the driver raised
---@return string|nil  error message when the driver raised
local function drive(overrides)
    local script_ctx = {}
    for key, value in pairs(overrides or {}) do
        script_ctx[key] = value
    end
    ctx = script_ctx
    package.loaded["audit_boss_collection"] = nil
    package.loaded["gameai_metrics.audit_matrix"] = audit_matrix_stub
    local ok, result = pcall(require, "audit_boss_collection")
    if ok then
        return result, nil
    end
    return nil, result
end

-- Some Lua stdlib packages installed by the pkg_test VM eagerly hold
-- onto the real `gameai_metrics.audit_matrix`; the stub reinstall on
-- every drive() keeps the driver looking at the spec's fake instead.

-- ─── Specs: required-field enforcement ──────────────────────────────

describe("audit_boss_collection — required fields", function()
    it("refuses a missing collection_path", function()
        configure()
        local _, err = drive({ output = "workspace/out.json" })
        expect(err ~= nil).to.equal(true)
        expect(err:find("collection_path") ~= nil).to.equal(true)
    end)

    it("refuses an empty collection_path", function()
        configure()
        local _, err = drive({ collection_path = "", output = "workspace/out.json" })
        expect(err ~= nil).to.equal(true)
        expect(err:find("collection_path") ~= nil).to.equal(true)
    end)

    it("refuses a non-string collection_path", function()
        configure()
        local _, err = drive({ collection_path = 42, output = "workspace/out.json" })
        expect(err ~= nil).to.equal(true)
        expect(err:find("collection_path") ~= nil).to.equal(true)
    end)

    it("refuses a missing output", function()
        configure()
        local _, err = drive({ collection_path = "workspace/in.json" })
        expect(err ~= nil).to.equal(true)
        expect(err:find("output") ~= nil).to.equal(true)
    end)

    it("refuses an empty output", function()
        configure()
        local _, err = drive({ collection_path = "workspace/in.json", output = "" })
        expect(err ~= nil).to.equal(true)
        expect(err:find("output") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: default fallbacks ───────────────────────────────────────

describe("audit_boss_collection — optional defaults", function()
    it(
        "falls back to audit_matrix.DEFAULT_* when ctx omits n_games / prompt_set_size / seed",
        function()
            configure()
            local result, err = drive({
                collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
                output = "workspace/gameai-harvest/audit_run2.json",
            })
            expect(err).to.equal(nil)
            expect(result).to.equal(NEXT_REPORT)
            expect(NEW_OPTS.n_games).to.equal(audit_matrix_stub.DEFAULT_N_GAMES)
            expect(NEW_OPTS.prompt_set_size).to.equal(audit_matrix_stub.DEFAULT_PROMPT_SET_SIZE)
            expect(NEW_OPTS.seed).to.equal(audit_matrix_stub.DEFAULT_SEED)
        end
    )

    it("defaults style to 'guardian' when ctx omits style", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
        })
        expect(err).to.equal(nil)
        expect(NEW_OPTS.style).to.equal("guardian")
    end)

    it("passes teacher_alias through as nil when ctx omits it", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
        })
        expect(err).to.equal(nil)
        expect(NEW_OPTS.teacher_alias).to.equal(nil)
    end)

    it("forwards every ctx field when the caller supplies them", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
            n_games = 400,
            prompt_set_size = 8,
            seed = 12345,
            style = "sentinel",
            teacher_alias = "guardian_duel_npc",
        })
        expect(err).to.equal(nil)
        expect(NEW_OPTS.collection_path).to.equal("workspace/in.json")
        expect(NEW_OPTS.n_games).to.equal(400)
        expect(NEW_OPTS.prompt_set_size).to.equal(8)
        expect(NEW_OPTS.seed).to.equal(12345)
        expect(NEW_OPTS.style).to.equal("sentinel")
        expect(NEW_OPTS.teacher_alias).to.equal("guardian_duel_npc")
    end)

    it("refuses a non-numeric n_games", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
            n_games = "lots",
        })
        expect(err ~= nil).to.equal(true)
        expect(err:find("n_games") ~= nil).to.equal(true)
    end)

    it("refuses an empty teacher_alias", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
            teacher_alias = "",
        })
        expect(err ~= nil).to.equal(true)
        expect(err:find("teacher_alias") ~= nil).to.equal(true)
    end)
end)

-- ─── Specs: runner wiring ───────────────────────────────────────────

describe("audit_boss_collection — runner wiring", function()
    it("calls new -> run -> save exactly once each and passes output through to save", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/gameai-harvest/audit_run2.json",
        })
        expect(err).to.equal(nil)
        expect(RUN_CALLS).to.equal(1)
        expect(SAVE_CALLS).to.equal(1)
        expect(SAVE_PATH).to.equal("workspace/gameai-harvest/audit_run2.json")
    end)

    it("returns the runner's report table verbatim", function()
        configure()
        local result, err = drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
        })
        expect(err).to.equal(nil)
        expect(result).to.equal(NEXT_REPORT)
        expect(result.per_card.weak.win_rate).to.equal(0.20)
        expect(result.sd_matrix.mid.strong).to.equal(0.11)
        expect(result.meta.style).to.equal("guardian")
    end)
end)

-- ─── Specs: log emission ────────────────────────────────────────────

--- Case-insensitive plain substring predicate; the log lines the
--- driver assembles are literal, so a plain match is enough.
local function has_line(needle)
    for _, line in ipairs(LOG_LINES) do
        if line:find(needle, 1, true) ~= nil then
            return true
        end
    end
    return false
end

describe("audit_boss_collection — log summary", function()
    it("emits the header line with style / n_games / prompt_set / aliases / output", function()
        configure()
        drive({
            collection_path = "workspace/in.json",
            output = "workspace/gameai-harvest/audit_run2.json",
        })
        expect(has_line("[gameai-audit] audit: style=guardian n_games=200 prompt_set=16 aliases=3")).to.equal(
            true
        )
        expect(has_line("-> workspace/gameai-harvest/audit_run2.json")).to.equal(true)
    end)

    it("emits one per_card line per alias with win_rate / sd_teacher / trickiness", function()
        configure()
        NEXT_REPORT.per_card.weak.sd_teacher = 0.72
        NEXT_REPORT.per_card.mid.sd_teacher = 0.09
        NEXT_REPORT.per_card.strong.sd_teacher = 0.24
        drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
            teacher_alias = "guardian_duel_npc",
        })
        expect(has_line("per_card\tweak\twin_rate=0.200\tsd_teacher=0.720\ttrickiness=0.100")).to.equal(
            true
        )
        expect(has_line("per_card\tmid\twin_rate=0.700\tsd_teacher=0.090\ttrickiness=0.350")).to.equal(
            true
        )
        expect(has_line("per_card\tstrong\twin_rate=0.920\tsd_teacher=0.240\ttrickiness=0.420")).to.equal(
            true
        )
    end)

    it("prints '-' for sd_teacher when the runner omitted it (no teacher_alias)", function()
        configure()
        drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
        })
        expect(has_line("per_card\tweak\twin_rate=0.200\tsd_teacher=-\ttrickiness=0.100")).to.equal(
            true
        )
    end)

    it("emits one SD pair line per unordered pair, sorted alphabetically", function()
        configure()
        drive({
            collection_path = "workspace/in.json",
            output = "workspace/out.json",
        })
        -- Sorted aliases: mid, strong, weak → pairs (mid,strong), (mid,weak), (strong,weak).
        expect(has_line("SD(mid,strong)=0.110")).to.equal(true)
        expect(has_line("SD(mid,weak)=0.550")).to.equal(true)
        expect(has_line("SD(strong,weak)=0.600")).to.equal(true)
        -- And exactly three pair lines for the 3-alias report (n*(n-1)/2).
        local pair_lines = 0
        for _, line in ipairs(LOG_LINES) do
            if line:find("SD(", 1, true) ~= nil then
                pair_lines = pair_lines + 1
            end
        end
        expect(pair_lines).to.equal(3)
    end)
end)
