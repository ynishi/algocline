-- spec/fight_boss_sweep_spec.lua
--
-- Spec for the `fight_boss_sweep.lua` driver. The harness is the union
-- of the two the repo already uses: the whole driver file is loaded once
-- per case against stubbed hosts (`audit_boss_collection_spec.lua`'s
-- `drive()`), and the stubbed runner records its calls into arrays
-- rather than a single slot (`gameai_metrics/spec/fight_matrix_spec.lua`)
-- because a sweep calls the runner once per grid point.
--
-- Only `gameai_metrics.fight_matrix` is stubbed — a stand-in module
-- registered under `package.loaded` before the driver requires it. The
-- stub answers `:run()` with a canned report whose numbers depend on the
-- call index, which is what makes "grid point i carries the matrix of
-- run i" assertable. It deliberately has **no** `:save()`: the sweep
-- driver owns its own save (one file for the whole grid), so a driver
-- that regressed to per-point `fight:save()` would fail here loudly.
--
-- `gameai_metrics._fs` is *not* stubbed. The save cases exercise the
-- real `ensure_parent_dir`, which is the whole point of routing the
-- write through it.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/fight_boss_sweep_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })

local describe, it, expect = lust.describe, lust.it, lust.expect

alc = alc or {}

-- ─── Minimal JSON encoder / decoder ─────────────────────────────────
--
-- Same subset as `fight_matrix_spec.lua`: numbers, strings (with `\"` /
-- `\\` escapes), booleans, arrays (dense 1..n) and objects with keys
-- sorted for determinism. Enough to write the sweep report and read it
-- back, which is all the round-trip case needs.

local function encode(value)
    local kind = type(value)
    if value == nil then
        return "null"
    end
    if kind == "boolean" or kind == "number" then
        return tostring(value)
    end
    if kind == "string" then
        return string.format("%q", value)
    end
    if kind ~= "table" then
        error("spec json_encode: unsupported type " .. kind, 0)
    end
    local n = #value
    if n > 0 then
        local total = 0
        for _ in pairs(value) do
            total = total + 1
        end
        if total == n then
            local items = {}
            for index = 1, n do
                items[index] = encode(value[index])
            end
            return "[" .. table.concat(items, ",") .. "]"
        end
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
        fields[#fields + 1] = string.format("%q", tostring(key)) .. ":" .. encode(value[key])
    end
    return "{" .. table.concat(fields, ",") .. "}"
end

local function json_decode(text)
    local pos = 1
    local function skip_ws()
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == " " or ch == "\n" or ch == "\r" or ch == "\t" then
                pos = pos + 1
            else
                return
            end
        end
    end
    local parse_value
    local function parse_string()
        assert(text:sub(pos, pos) == '"', "spec json_decode: expected string at " .. pos)
        pos = pos + 1
        local start = pos
        local parts = {}
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch == '"' then
                parts[#parts + 1] = text:sub(start, pos - 1)
                pos = pos + 1
                return table.concat(parts)
            elseif ch == "\\" then
                parts[#parts + 1] = text:sub(start, pos - 1)
                local esc = text:sub(pos + 1, pos + 1)
                if esc == '"' or esc == "\\" or esc == "/" then
                    parts[#parts + 1] = esc
                elseif esc == "n" then
                    parts[#parts + 1] = "\n"
                elseif esc == "t" then
                    parts[#parts + 1] = "\t"
                elseif esc == "r" then
                    parts[#parts + 1] = "\r"
                else
                    error("spec json_decode: unsupported escape \\" .. esc, 0)
                end
                pos = pos + 2
                start = pos
            else
                pos = pos + 1
            end
        end
        error("spec json_decode: unterminated string", 0)
    end
    local function parse_number()
        local start = pos
        while pos <= #text do
            local ch = text:sub(pos, pos)
            if ch:match("[%-%+%d%.eE]") then
                pos = pos + 1
            else
                break
            end
        end
        return tonumber(text:sub(start, pos - 1))
    end
    local function parse_array()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "]" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            out[#out + 1] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "]" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or ] at " .. pos, 0)
            end
        end
    end
    local function parse_object()
        pos = pos + 1
        skip_ws()
        local out = {}
        if text:sub(pos, pos) == "}" then
            pos = pos + 1
            return out
        end
        while true do
            skip_ws()
            local key = parse_string()
            skip_ws()
            assert(text:sub(pos, pos) == ":", "spec json_decode: expected : at " .. pos)
            pos = pos + 1
            skip_ws()
            out[key] = parse_value()
            skip_ws()
            local ch = text:sub(pos, pos)
            if ch == "," then
                pos = pos + 1
            elseif ch == "}" then
                pos = pos + 1
                return out
            else
                error("spec json_decode: expected , or } at " .. pos, 0)
            end
        end
    end
    parse_value = function()
        skip_ws()
        local ch = text:sub(pos, pos)
        if ch == "{" then
            return parse_object()
        end
        if ch == "[" then
            return parse_array()
        end
        if ch == '"' then
            return parse_string()
        end
        if ch == "t" and text:sub(pos, pos + 3) == "true" then
            pos = pos + 4
            return true
        end
        if ch == "f" and text:sub(pos, pos + 4) == "false" then
            pos = pos + 5
            return false
        end
        if ch == "n" and text:sub(pos, pos + 3) == "null" then
            pos = pos + 4
            return nil
        end
        return parse_number()
    end
    skip_ws()
    return parse_value()
end

alc.json_encode = encode
alc.json_decode = json_decode

--- Log lines the driver emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- ─── fight_matrix stub ──────────────────────────────────────────────
--
-- `new()` appends its opts to `NEW_OPTS` and hands back a runner bound
-- to that call index; `:run()` produces a report through `REPORT_HOOK`
-- (the canned generator by default) and remembers it in `REPORTS`, so a
-- case can assert that grid point i carries exactly the matrix run i
-- returned — identity, not a value that happens to match.

local BOSSES = { "boss_weak", "boss_mid" }

local NEW_OPTS = {}
local RUN_CALLS = 0
local REPORTS = {}
local REPORT_HOOK = nil

local function copy_array(source)
    local out = {}
    for i = 1, #source do
        out[i] = source[i]
    end
    return out
end

--- One canned report for call `index`. Every number carries the index
--- so a mixed-up grid point is visible in the assertion message rather
--- than hidden behind two identical matrices. `meta` mirrors what the
--- real runner reports: the boss axis comes from the collection (fixed
--- here), the player axis from the opts, the rest echoes the call.
local function canned_report(index, opts)
    local matrix = {}
    for _, boss in ipairs(BOSSES) do
        local row = {}
        for _, player in ipairs(opts.players) do
            local win_rate = 0.1 * index
            row[player] = {
                win_rate = win_rate,
                ci_lower = win_rate - 0.05,
                ci_upper = win_rate + 0.05,
                n_games = opts.n_games,
                game_length_mean = 6.0 + index,
                final_hp_margin_mean = 2.0 * index,
            }
        end
        matrix[boss] = row
    end
    return {
        matrix = matrix,
        meta = {
            n_games = opts.n_games,
            seed = opts.seed,
            style = opts.style,
            temperature = opts.temperature,
            per_game = opts.per_game,
            per_move = opts.per_move,
            -- Fresh arrays every call: the driver's cross-point check
            -- has to compare element by element, not by table identity.
            bosses = copy_array(BOSSES),
            players = copy_array(opts.players),
        },
    }
end

local fight_matrix_stub = {}
fight_matrix_stub.DEFAULT_N_GAMES = 200
fight_matrix_stub.DEFAULT_SEED = 20260731
fight_matrix_stub.DEFAULT_TEMPERATURE = 1.0

local Fight = {}
Fight.__index = Fight

function Fight:run()
    RUN_CALLS = RUN_CALLS + 1
    local produce = REPORT_HOOK or canned_report
    local report = produce(self._index, self._opts)
    REPORTS[self._index] = report
    return report
end

function fight_matrix_stub.new(opts)
    NEW_OPTS[#NEW_OPTS + 1] = opts
    return setmetatable({ _index = #NEW_OPTS, _opts = opts }, Fight)
end

-- ─── Temp output paths ──────────────────────────────────────────────
--
-- Every successful drive writes its report, so each case needs its own
-- `output`. Paths are derived from `os.tmpname()` (never the reserved
-- name itself, which already exists) and removed at the start of the
-- next case.

local CREATED_PATHS = {}
local _tmp_counter = 0

local function tmp_path()
    _tmp_counter = _tmp_counter + 1
    local path = os.tmpname() .. "-sweep-" .. tostring(_tmp_counter)
    CREATED_PATHS[#CREATED_PATHS + 1] = path
    return path
end

--- Reset every observable stub between cases so a leaked variable from
--- an earlier case cannot mask a fresh assertion, and drop the files
--- earlier cases wrote.
local function configure()
    LOG_LINES = {}
    NEW_OPTS = {}
    RUN_CALLS = 0
    REPORTS = {}
    REPORT_HOOK = nil
    for _, path in ipairs(CREATED_PATHS) do
        os.remove(path)
    end
    CREATED_PATHS = {}
end

-- ─── Driver loader ──────────────────────────────────────────────────

--- Drive the script once against the current stubs. `ctx` is a global
--- under the `alc_run` contract, so it is planted as one. Both the
--- driver and its `fight_matrix` require are cleared from
--- `package.loaded` first so the top-level ctx decoding runs again for
--- every case; the stub is re-installed after the clear.
---@param script_ctx table the whole ctx for this case (no defaults are
---                        filled in here — a case that omits a required
---                        field means to omit it)
---@return table|nil result the script's return value on success, nil
---                     when the driver raised
---@return string|nil  error message when the driver raised
local function drive(script_ctx)
    ctx = script_ctx
    package.loaded["fight_boss_sweep"] = nil
    package.loaded["gameai_metrics.fight_matrix"] = fight_matrix_stub
    local ok, result = pcall(require, "fight_boss_sweep")
    if ok then
        return result, nil
    end
    return nil, result
end

--- A complete ctx with every required field present, `overrides`
--- applied on top. Cases that test a *missing* field build their ctx
--- literally instead — a nil override cannot delete a key.
local function full_ctx(overrides)
    local script_ctx = {
        collection_path = "workspace/gameai-harvest/run2_measured_bands.json",
        players = { "player_sentinel" },
        output = tmp_path(),
        temperatures = { 0.25, 0.5, 1.0 },
    }
    for key, value in pairs(overrides or {}) do
        script_ctx[key] = value
    end
    return script_ctx
end

--- Plain (non-pattern) substring predicate over the emitted log lines.
--- Plain matching matters here: the temperatures printed in the lines
--- contain `.`, which a pattern match would treat as a wildcard.
local function has_line(needle)
    for _, line in ipairs(LOG_LINES) do
        if line:find(needle, 1, true) ~= nil then
            return true
        end
    end
    return false
end

--- Plain substring predicate for error messages, for the same reason.
local function says(err, needle)
    return err ~= nil and err:find(needle, 1, true) ~= nil
end

-- ─── Specs: required-field enforcement ──────────────────────────────

describe("fight_boss_sweep — required fields", function()
    it("refuses a missing collection_path", function()
        configure()
        local _, err = drive({
            players = { "p1" },
            output = tmp_path(),
            temperatures = { 1.0 },
        })
        expect(err ~= nil).to.equal(true)
        expect(says(err, "collection_path")).to.equal(true)
        expect(says(err, "fight_boss_sweep")).to.equal(true)
    end)

    it("refuses a non-string collection_path", function()
        configure()
        local _, err = drive(full_ctx({ collection_path = 42 }))
        expect(says(err, "collection_path")).to.equal(true)
    end)

    it("refuses a missing players array", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            output = tmp_path(),
            temperatures = { 1.0 },
        })
        expect(says(err, "players")).to.equal(true)
    end)

    it("refuses an empty players array", function()
        configure()
        local _, err = drive(full_ctx({ players = {} }))
        expect(says(err, "players")).to.equal(true)
    end)

    it("refuses a missing output", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            players = { "p1" },
            temperatures = { 1.0 },
        })
        expect(says(err, "output")).to.equal(true)
    end)

    it("refuses an empty output", function()
        configure()
        local _, err = drive(full_ctx({ output = "" }))
        expect(says(err, "output")).to.equal(true)
    end)

    it("runs nothing when a required field is missing", function()
        configure()
        local _, err = drive({ players = { "p1" } })
        expect(err ~= nil).to.equal(true)
        expect(#NEW_OPTS).to.equal(0)
        expect(RUN_CALLS).to.equal(0)
    end)

    it("refuses a singular ctx.temperature rather than letting the grid outvote it", function()
        configure()
        local _, err = drive(full_ctx({ temperature = 1.0 }))
        expect(err ~= nil).to.equal(true)
        expect(says(err, "ctx.temperature (singular)")).to.equal(true)
        expect(says(err, "ctx.temperatures")).to.equal(true)
        expect(#NEW_OPTS).to.equal(0)
    end)
end)

-- ─── Specs: the temperature grid ────────────────────────────────────

describe("fight_boss_sweep — temperatures grid", function()
    it("refuses a missing temperatures array", function()
        configure()
        local _, err = drive({
            collection_path = "workspace/in.json",
            players = { "p1" },
            output = tmp_path(),
        })
        expect(says(err, "temperatures")).to.equal(true)
        expect(says(err, "fight_boss_sweep")).to.equal(true)
    end)

    it("refuses a non-array temperatures", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = 1.0 }))
        expect(says(err, "temperatures")).to.equal(true)
    end)

    it("refuses an empty temperatures array", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = {} }))
        expect(says(err, "temperatures")).to.equal(true)
    end)

    it("refuses a non-number grid element and names its index", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { 0.5, "hot", 1.5 } }))
        expect(says(err, "temperatures[2]")).to.equal(true)
        expect(says(err, "string")).to.equal(true)
    end)

    it("refuses a zero grid element (there is no greedy spelling)", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { 0.5, 0 } }))
        expect(says(err, "temperatures[2]")).to.equal(true)
        expect(says(err, "greedy")).to.equal(true)
    end)

    it("refuses a negative grid element", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { -1.0 } }))
        expect(says(err, "temperatures[1]")).to.equal(true)
    end)

    it("refuses an infinite grid element", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { 1.0, math.huge } }))
        expect(says(err, "temperatures[2]")).to.equal(true)
    end)

    it("refuses a NaN grid element", function()
        configure()
        local nan = 0 / 0
        local _, err = drive(full_ctx({ temperatures = { nan } }))
        expect(says(err, "temperatures[1]")).to.equal(true)
    end)

    it("refuses a repeated grid point and names both indices", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { 0.5, 1.0, 0.5 } }))
        -- Plain match: the message quotes the value as `0.5`, whose `.`
        -- would be a pattern wildcard.
        expect(says(err, "lists 0.5 twice")).to.equal(true)
        expect(says(err, "index 1 and 3")).to.equal(true)
    end)

    it("validates the whole grid before running the first point", function()
        configure()
        -- The bad point is last: a driver that validated lazily would
        -- have already measured two grid points by the time it raised.
        local _, err = drive(full_ctx({ temperatures = { 0.5, 1.0, -2.0 } }))
        expect(err ~= nil).to.equal(true)
        expect(#NEW_OPTS).to.equal(0)
        expect(RUN_CALLS).to.equal(0)
    end)
end)

-- ─── Specs: what the driver asks the runner for ─────────────────────

describe("fight_boss_sweep — runner invocation", function()
    it("builds one runner per grid point, each with its own temperature", function()
        configure()
        local _, err = drive(full_ctx({ temperatures = { 0.25, 0.5, 1.0, 2.0 } }))
        expect(err).to.equal(nil)
        expect(#NEW_OPTS).to.equal(4)
        expect(RUN_CALLS).to.equal(4)
        expect(NEW_OPTS[1].temperature).to.equal(0.25)
        expect(NEW_OPTS[2].temperature).to.equal(0.5)
        expect(NEW_OPTS[3].temperature).to.equal(1.0)
        expect(NEW_OPTS[4].temperature).to.equal(2.0)
    end)

    it("holds every other runner opt fixed across the grid", function()
        configure()
        local players = { "p1", "p2" }
        local _, err = drive(full_ctx({
            collection_path = "workspace/gameai-harvest/collection.json",
            players = players,
            temperatures = { 0.25, 1.0, 2.0 },
            n_games = 400,
            seed = 12345,
            style = "sentinel",
            per_game = true,
        }))
        expect(err).to.equal(nil)
        for index = 1, 3 do
            local opts = NEW_OPTS[index]
            expect(opts.collection_path).to.equal("workspace/gameai-harvest/collection.json")
            -- Identity, not equality: the driver forwards the caller's
            -- array rather than rebuilding an axis per point.
            expect(opts.players).to.equal(players)
            expect(opts.n_games).to.equal(400)
            expect(opts.seed).to.equal(12345)
            expect(opts.style).to.equal("sentinel")
            expect(opts.per_game).to.equal(true)
        end
    end)

    it("forwards per_move to every grid point", function()
        configure()
        local _, err = drive(full_ctx({
            temperatures = { 0.25, 1.0 },
            per_game = true,
            per_move = true,
        }))
        expect(err).to.equal(nil)
        for index = 1, 2 do
            -- Dropping the flag here would cost nothing visible: the
            -- sweep would run, save and summarise, minus the transcripts
            -- the caller asked for.
            expect(NEW_OPTS[index].per_move).to.equal(true)
        end
    end)

    it("defaults per_move to false", function()
        configure()
        local _, err = drive(full_ctx({}))
        expect(err).to.equal(nil)
        expect(NEW_OPTS[1].per_move).to.equal(false)
    end)

    it("refuses a non-boolean per_move", function()
        configure()
        local _, err = drive(full_ctx({ per_move = "true" }))
        expect(says(err, "per_move")).to.equal(true)
        expect(says(err, "fight_boss_sweep")).to.equal(true)
    end)

    it("falls back to fight_matrix.DEFAULT_* for n_games / seed", function()
        configure()
        local _, err = drive(full_ctx({}))
        expect(err).to.equal(nil)
        expect(NEW_OPTS[1].n_games).to.equal(fight_matrix_stub.DEFAULT_N_GAMES)
        expect(NEW_OPTS[1].seed).to.equal(fight_matrix_stub.DEFAULT_SEED)
    end)

    it("defaults style to 'guardian' and per_game to false", function()
        configure()
        local _, err = drive(full_ctx({}))
        expect(err).to.equal(nil)
        expect(NEW_OPTS[1].style).to.equal("guardian")
        expect(NEW_OPTS[1].per_game).to.equal(false)
    end)

    it("refuses a non-numeric n_games", function()
        configure()
        local _, err = drive(full_ctx({ n_games = "lots" }))
        expect(says(err, "n_games")).to.equal(true)
        expect(says(err, "fight_boss_sweep")).to.equal(true)
    end)

    it("refuses a non-boolean per_game", function()
        configure()
        local _, err = drive(full_ctx({ per_game = "false" }))
        expect(says(err, "per_game")).to.equal(true)
        expect(says(err, "fight_boss_sweep")).to.equal(true)
    end)

    it("refuses an empty style", function()
        configure()
        local _, err = drive(full_ctx({ style = "" }))
        expect(says(err, "style")).to.equal(true)
    end)
end)

-- ─── Specs: the sweep array ─────────────────────────────────────────

describe("fight_boss_sweep — sweep array", function()
    it("keeps the grid order the caller gave, unsorted", function()
        configure()
        local grid = { 1.0, 0.25, 2.0, 0.5 }
        local result, err = drive(full_ctx({ temperatures = grid }))
        expect(err).to.equal(nil)
        expect(#result.sweep).to.equal(4)
        for index = 1, #grid do
            expect(result.sweep[index].temperature).to.equal(grid[index])
        end
    end)

    it("pairs every grid point with the matrix its own run returned", function()
        configure()
        local result, err = drive(full_ctx({ temperatures = { 0.25, 0.5, 1.0 } }))
        expect(err).to.equal(nil)
        for index = 1, 3 do
            expect(result.sweep[index].matrix).to.equal(REPORTS[index].matrix)
            -- The canned numbers carry the call index, so a swapped
            -- pair would show up as a wrong win_rate here too.
            expect(result.sweep[index].matrix.boss_weak.player_sentinel.win_rate).to.equal(
                0.1 * index
            )
        end
    end)

    it("carries a per_game cell's games array through untouched", function()
        configure()
        REPORT_HOOK = function(index, opts)
            local report = canned_report(index, opts)
            report.matrix.boss_weak.player_sentinel.games = {
                { outcome = 1.0, game_length = 7, final_hp_margin = 3 },
                { outcome = 0.0, game_length = 9, final_hp_margin = -2 },
            }
            return report
        end
        local result, err = drive(full_ctx({ temperatures = { 0.5, 1.0 }, per_game = true }))
        expect(err).to.equal(nil)
        local games = result.sweep[2].matrix.boss_weak.player_sentinel.games
        expect(type(games)).to.equal("table")
        expect(#games).to.equal(2)
        expect(games[1].outcome).to.equal(1.0)
        expect(games[2].final_hp_margin).to.equal(-2)
    end)
end)

-- ─── Specs: meta aggregation ────────────────────────────────────────

describe("fight_boss_sweep — meta", function()
    it("reports the axes of the first grid point", function()
        configure()
        local result, err = drive(full_ctx({ players = { "p1", "p2" } }))
        expect(err).to.equal(nil)
        expect(table.concat(result.meta.bosses, ",")).to.equal(table.concat(BOSSES, ","))
        expect(table.concat(result.meta.players, ",")).to.equal("p1,p2")
    end)

    it("reports the run parameters the runner measured under", function()
        configure()
        local result, err = drive(full_ctx({
            collection_path = "workspace/gameai-harvest/collection.json",
            n_games = 50,
            seed = 7,
            style = "sentinel",
            per_game = true,
        }))
        expect(err).to.equal(nil)
        expect(result.meta.n_games).to.equal(50)
        expect(result.meta.seed).to.equal(7)
        expect(result.meta.style).to.equal("sentinel")
        expect(result.meta.per_game).to.equal(true)
        expect(result.meta.collection_path).to.equal("workspace/gameai-harvest/collection.json")
    end)

    it("reads per_move back off the reference grid point", function()
        configure()
        local off, off_err = drive(full_ctx({}))
        expect(off_err).to.equal(nil)
        expect(off.meta.per_move).to.equal(false)
        configure()
        local on, on_err = drive(full_ctx({ per_game = true, per_move = true }))
        expect(on_err).to.equal(nil)
        -- Read back off the runner's own meta rather than echoed from
        -- the ctx, the same path `per_game` takes: what the report
        -- describes is what was measured.
        expect(on.meta.per_move).to.equal(true)
    end)

    it("records the grid in order, as a copy of the ctx array", function()
        configure()
        local grid = { 1.5, 0.25, 1.0 }
        local result, err = drive(full_ctx({ temperatures = grid }))
        expect(err).to.equal(nil)
        expect(table.concat(result.meta.temperatures, ",")).to.equal("1.5,0.25,1.0")
        -- A copy, so a caller mutating its own array afterwards cannot
        -- desynchronise the report from what was measured.
        expect(result.meta.temperatures ~= grid).to.equal(true)
    end)
end)

-- ─── Specs: the cross-point invariant ───────────────────────────────
--
-- Every case here makes the *second* grid point disagree with the first
-- on one field. Under the real runner none of these can happen, which is
-- why the check is worth having: it is the tripwire for the day one of
-- them starts being derived per call.

describe("fight_boss_sweep — cross-point invariant", function()
    local function drift(mutate)
        configure()
        REPORT_HOOK = function(index, opts)
            local report = canned_report(index, opts)
            if index == 2 then
                mutate(report)
            end
            return report
        end
        return drive(full_ctx({ temperatures = { 0.5, 1.0, 2.0 } }))
    end

    it("refuses a boss axis that changed between points", function()
        local _, err = drift(function(report)
            report.meta.bosses[2] = "boss_other"
        end)
        expect(says(err, "bosses[2]")).to.equal(true)
        expect(says(err, "grid point 2")).to.equal(true)
        expect(says(err, "temperature 1")).to.equal(true)
    end)

    it("refuses a boss axis that lost an entry", function()
        local _, err = drift(function(report)
            report.meta.bosses[2] = nil
        end)
        expect(says(err, "bosses")).to.equal(true)
        expect(says(err, "entries")).to.equal(true)
    end)

    it("refuses a player axis that changed between points", function()
        local _, err = drift(function(report)
            report.meta.players[1] = "player_other"
        end)
        expect(says(err, "players[1]")).to.equal(true)
    end)

    it("refuses an n_games that changed between points", function()
        local _, err = drift(function(report)
            report.meta.n_games = 1
        end)
        expect(says(err, "n_games")).to.equal(true)
        expect(says(err, "grid point 2")).to.equal(true)
    end)

    it("refuses a seed that changed between points", function()
        local _, err = drift(function(report)
            report.meta.seed = 99
        end)
        expect(says(err, "seed")).to.equal(true)
    end)

    it("refuses a style that changed between points", function()
        local _, err = drift(function(report)
            report.meta.style = "sentinel"
        end)
        expect(says(err, "style")).to.equal(true)
    end)

    it("stops at the offending point instead of finishing the grid", function()
        local _, err = drift(function(report)
            report.meta.seed = 99
        end)
        expect(err ~= nil).to.equal(true)
        expect(RUN_CALLS).to.equal(2)
    end)

    it("refuses a report without a matrix / meta table", function()
        configure()
        REPORT_HOOK = function()
            return "not a report"
        end
        local _, err = drive(full_ctx({}))
        expect(says(err, "no matrix / meta")).to.equal(true)
    end)

    it("refuses a first point that reports no boss axis", function()
        configure()
        REPORT_HOOK = function(index, opts)
            local report = canned_report(index, opts)
            report.meta.bosses = {}
            return report
        end
        local _, err = drive(full_ctx({}))
        expect(says(err, "no bosses axis")).to.equal(true)
    end)
end)

-- ─── Specs: save ────────────────────────────────────────────────────

describe("fight_boss_sweep — save", function()
    it("writes JSON that round-trips through json_decode", function()
        configure()
        local path = tmp_path()
        local result, err = drive(full_ctx({
            output = path,
            temperatures = { 0.25, 0.5, 1.0 },
            n_games = 50,
            seed = 7,
        }))
        expect(err).to.equal(nil)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        local parsed = json_decode(body)
        expect(#parsed.sweep).to.equal(3)
        expect(parsed.sweep[2].temperature).to.equal(0.5)
        expect(parsed.sweep[2].matrix.boss_mid.player_sentinel.win_rate).to.equal(0.2)
        expect(table.concat(parsed.meta.temperatures, ",")).to.equal("0.25,0.5,1.0")
        expect(parsed.meta.n_games).to.equal(50)
        expect(parsed.meta.seed).to.equal(7)
        expect(parsed.meta.style).to.equal("guardian")
        expect(parsed.meta.per_game).to.equal(false)
        expect(table.concat(parsed.meta.bosses, ",")).to.equal(table.concat(BOSSES, ","))
        -- The returned table and the written file describe the same run.
        expect(result.meta.n_games).to.equal(parsed.meta.n_games)
    end)

    it("creates missing parent directories via _fs.ensure_parent_dir", function()
        configure()
        local base = os.tmpname()
        os.remove(base)
        local dir = base .. "-sweep-parent"
        local path = dir .. "/nested/sweep.json"
        local _, err = drive(full_ctx({ output = path, temperatures = { 1.0 } }))
        expect(err).to.equal(nil)
        local f = io.open(path, "r")
        expect(f ~= nil).to.equal(true)
        local body = f:read("a")
        f:close()
        os.remove(path)
        os.remove(dir .. "/nested")
        os.remove(dir)
        local parsed = json_decode(body)
        expect(#parsed.sweep).to.equal(1)
    end)

    it("writes nothing when a later grid point fails", function()
        configure()
        local path = tmp_path()
        REPORT_HOOK = function(index, opts)
            if index == 3 then
                error("spec: runner blew up on grid point 3", 0)
            end
            return canned_report(index, opts)
        end
        local _, err = drive(full_ctx({ output = path, temperatures = { 0.25, 0.5, 1.0 } }))
        expect(says(err, "grid point 3")).to.equal(true)
        -- No half-written JSON for a reader to mistake for a full sweep.
        local f = io.open(path, "r")
        expect(f).to.equal(nil)
    end)
end)

-- ─── Specs: log summary ─────────────────────────────────────────────

describe("fight_boss_sweep — log summary", function()
    it("emits the sweep header with style / n_games / per_game / grid size / axes", function()
        configure()
        local path = tmp_path()
        drive(full_ctx({ output = path, temperatures = { 0.25, 0.5, 1.0 } }))
        expect(
            has_line(
                "[gameai-sweep] sweep: style=guardian n_games=200 per_game=false "
                    .. "temperatures=3 bosses=2 players=1 cells=6"
            )
        ).to.equal(true)
        expect(has_line("-> " .. path)).to.equal(true)
    end)

    it("emits one header line per grid point", function()
        configure()
        drive(full_ctx({ temperatures = { 0.25, 0.5, 1.0 } }))
        expect(has_line("sweep\tT=0.25\tpoint 1/3\tcells=2")).to.equal(true)
        expect(has_line("sweep\tT=0.5\tpoint 2/3\tcells=2")).to.equal(true)
        expect(has_line("sweep\tT=1\tpoint 3/3\tcells=2")).to.equal(true)
    end)

    it("emits one cell line per (grid point, boss, player) triple", function()
        configure()
        drive(full_ctx({ temperatures = { 0.25, 0.5 } }))
        expect(
            has_line(
                "sweep\tT=0.25\tboss_weak\tvs\tplayer_sentinel\twin_rate=0.100\t"
                    .. "ci=[0.050,0.150]\tlen=7.000\tmargin=2.000"
            )
        ).to.equal(true)
        expect(
            has_line(
                "sweep\tT=0.5\tboss_mid\tvs\tplayer_sentinel\twin_rate=0.200\t"
                    .. "ci=[0.150,0.250]\tlen=8.000\tmargin=4.000"
            )
        ).to.equal(true)
        local cell_lines = 0
        for _, line in ipairs(LOG_LINES) do
            if line:find("\tvs\t", 1, true) ~= nil then
                cell_lines = cell_lines + 1
            end
        end
        -- 2 grid points x 2 bosses x 1 player.
        expect(cell_lines).to.equal(4)
    end)

    it("prints '-' for a cell field the runner omitted", function()
        configure()
        REPORT_HOOK = function(index, opts)
            local report = canned_report(index, opts)
            report.matrix.boss_weak.player_sentinel.game_length_mean = nil
            return report
        end
        drive(full_ctx({ temperatures = { 1.0 } }))
        expect(has_line("win_rate=0.100\tci=[0.050,0.150]\tlen=-\tmargin=2.000")).to.equal(true)
    end)
end)
