-- spec/gen_othello6_mix_corpus_spec.lua
--
-- Spec for the ctx contract, the mixture validation and the file
-- contract of `gen_othello6_mix_corpus.lua`.
--
-- The script is a self-contained `alc_run` driver rather than a package,
-- so it is exercised the way it actually runs: the whole file is loaded
-- once per case against a stubbed host, and the assertions read what it
-- asked the generator for and what it handed the file.
--
-- Nothing costly runs. A mixed playout costs about twice a single-style
-- one -- both parents search every position before the draw picks
-- between them -- so at the measured rate the default of eight thousand
-- games would be three and a half minutes per case. `build_corpus` is
-- therefore a spy that answers a fixed yield per requested game, and the
-- default is read off the request rather than played. One case delegates
-- to the real generator on three games, which is what pins the rows in
-- the file to the width the encoding produces and exercises the real
-- mixed policy end to end.
--
-- The filesystem is stubbed too: `io.open` is replaced with an
-- in-memory store and `gameai_metrics._fs` with a recorder, so a spec
-- run writes nothing and never shells out to `mkdir -p`. What is
-- asserted is the bytes the script handed the file, decoded back through
-- the matching decoder.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/gen_othello6_mix_corpus_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })
--
-- so `require("gen_othello6_mix_corpus")` / `require("othello6")` /
-- `require("othello6_mix")` all resolve out of that one directory.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Minimal JSON encoder standing in for the host `alc.json_encode`.
---
--- Object keys are emitted in sorted order so a decoded round trip is
--- deterministic. The script feeds it the whole corpus object.
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

--- Matching decoder, so the file contract is read back out of the bytes
--- the script wrote rather than off the table it returned.
local function json_decode(text)
    local pos = 1
    local parse_value

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

    local function parse_string()
        assert(text:sub(pos, pos) == '"', "spec json_decode: expected a string at " .. pos)
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
            if text:sub(pos, pos):match("[%-%+%d%.eE]") then
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

alc.json_encode = json_encode
alc.json_decode = json_decode

--- Log lines the script emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- `othello6` and `othello6_mix` are the real modules and both reach for
-- the host RNG. The LCG below matches the shape the engine-level harness
-- installs; the playouts and the mixing draws only have to be
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

-- ─── In-memory filesystem ───────────────────────────────────────────

--- Path -> file contents.
local FILES = {}
--- `{path, mode}` of every open, in call order.
local OPEN_CALLS = {}
--- Paths handed to the stubbed `ensure_parent_dir`, in call order.
local PARENT_DIRS = {}

io.open = function(path, mode)
    mode = mode or "r"
    OPEN_CALLS[#OPEN_CALLS + 1] = { path = path, mode = mode }
    if mode:find("r", 1, true) then
        local body = FILES[path]
        if body == nil then
            return nil, "spec io.open: no such file"
        end
        return {
            read = function()
                return body
            end,
            close = function() end,
        }
    end
    if mode:find("w", 1, true) then
        local chunks = {}
        return {
            write = function(_, chunk)
                chunks[#chunks + 1] = tostring(chunk)
            end,
            close = function()
                FILES[path] = table.concat(chunks)
            end,
        }
    end
    error("spec io.open: unsupported mode " .. tostring(mode), 0)
end

-- The real helper shells out to `mkdir -p`. The script has to call it
-- before the write -- that is what keeps a first save into a fresh
-- workspace sub-tree from failing -- so it is recorded rather than
-- removed.
package.preload["gameai_metrics._fs"] = function()
    return {
        ensure_parent_dir = function(path)
            PARENT_DIRS[#PARENT_DIRS + 1] = path
        end,
    }
end

-- ─── Corpus spy ─────────────────────────────────────────────────────

local othello = require("othello6")

local real_build_corpus = othello.build_corpus

--- Opts every `build_corpus` call was made with, in call order.
local CORPUS_CALLS = {}

--- Delegate to the real generator instead of answering stand-ins.
local CORPUS_REAL = false

--- Rows the spy answers per requested game. One row per game is what
--- the encoding promises; another value drives the case that checks a
--- broken promise is reported.
local CORPUS_ROWS_PER_GAME = 1

othello.build_corpus = function(policy, opts)
    CORPUS_CALLS[#CORPUS_CALLS + 1] = opts
    if CORPUS_REAL then
        return real_build_corpus(policy, opts)
    end
    -- Two tokens rather than `ctx_len`: the script does not read the row
    -- width, and the default case asks for eight thousand of them.
    local rows = {}
    for _ = 1, math.floor(opts.games * CORPUS_ROWS_PER_GAME) do
        rows[#rows + 1] = { opts.pad_id, opts.pad_id }
    end
    return rows
end

-- ─── Driver ─────────────────────────────────────────────────────────

--- Path the cases write to.
local CORPUS_FILE = "workspace/spec/othello6-mix-corpus.json"

local function configure()
    LOG_LINES = {}
    FILES = {}
    OPEN_CALLS = {}
    PARENT_DIRS = {}
    CORPUS_CALLS = {}
    CORPUS_REAL = false
    CORPUS_ROWS_PER_GAME = 1
end

--- Load the script once against the current stubs.
---
--- `ctx` is a global in the `alc_run` contract, so it is planted as one
--- here. `package.loaded` is cleared first: every case needs the
--- top-level ctx decoding to run again.
---@param overrides table ctx fields for this case
---@return table result the script's return value
local function drive(overrides)
    ctx = overrides or {}
    package.loaded["gen_othello6_mix_corpus"] = nil
    return require("gen_othello6_mix_corpus")
end

--- Drive a ctx that must be refused, and hand back the message.
local function drive_error(overrides)
    local ok, err = pcall(drive, overrides)
    expect(ok).to.equal(false)
    return tostring(err)
end

local function contains(haystack, needle)
    return tostring(haystack):find(needle, 1, true) ~= nil
end

--- A run small enough to read, with the file it writes to and the two
--- fields that have no defaults.
local function small(overrides)
    local out = { path = CORPUS_FILE, beta = 0.5, depth = 1, games = 3 }
    for key, value in pairs(overrides or {}) do
        out[key] = value
    end
    return out
end

--- Sorted key list of a table, as a comma-joined string.
local function key_list(t)
    local keys = {}
    for key in pairs(t) do
        keys[#keys + 1] = tostring(key)
    end
    table.sort(keys)
    return table.concat(keys, ",")
end

--- The corpus object as it landed in the file.
local function written_corpus()
    local body = FILES[CORPUS_FILE]
    expect(type(body)).to.equal("string")
    return json_decode(body)
end

-- ─── Cases ──────────────────────────────────────────────────────────

describe("gen_othello6_mix_corpus ctx contract", function()
    it("refuses a run without a mixing weight", function()
        configure()
        local err = drive_error({ path = CORPUS_FILE, depth = 1, games = 3 })

        -- The weight is the condition the file exists to fix, so there
        -- is no house default to fall back on: a corpus built at an
        -- unrequested beta is a mixture nobody asked for.
        expect(contains(err, "ctx.beta is required")).to.equal(true)
        expect(contains(err, "open interval (0, 1)")).to.equal(true)
        -- Refused before a game is played.
        expect(#CORPUS_CALLS).to.equal(0)
        expect(#PARENT_DIRS).to.equal(0)
    end)

    it("refuses a run that writes nowhere", function()
        configure()
        local err = drive_error({ beta = 0.5, depth = 1, games = 3 })

        expect(contains(err, "ctx.path is required")).to.equal(true)
        expect(#CORPUS_CALLS).to.equal(0)
        expect(#PARENT_DIRS).to.equal(0)
    end)

    it("refuses a path or a beta of the wrong type", function()
        configure()
        expect(contains(drive_error({ path = 7, beta = 0.5 }), "ctx.path must be a string")).to.equal(
            true
        )

        configure()
        expect(contains(drive_error({ path = "", beta = 0.5 }), "ctx.path must not be empty")).to.equal(
            true
        )

        configure()
        expect(
            contains(
                drive_error({ path = CORPUS_FILE, beta = "0.5" }),
                "ctx.beta must be a finite number"
            )
        ).to.equal(true)
    end)

    it("applies the documented defaults", function()
        configure()
        local out = drive({ path = CORPUS_FILE, beta = 0.5 })

        -- Eight thousand rows is 249 steps of batch 32 with a batch in
        -- hand, the budget the single-style sweep saturated at.
        expect(out.games).to.equal(8000)
        expect(out.styles[1]).to.equal("corner")
        expect(out.styles[2]).to.equal("greedy")
        expect(out.depth).to.equal(2)
        expect(out.beta).to.equal(0.5)
        expect(out.kind).to.equal("mix")
        expect(out.seed).to.equal(20260804)
        expect(out.ctx_len).to.equal(48)
        expect(out.random_opening_max).to.equal(othello.RANDOM_OPENING_MAX)
        expect(out.vocab_size).to.equal(othello.VOCAB_SIZE)
        expect(out.path).to.equal(CORPUS_FILE)
        expect(out.ok).to.equal(true)

        -- Asked for, not played: the spy answers the count.
        expect(#CORPUS_CALLS).to.equal(1)
        expect(CORPUS_CALLS[1].games).to.equal(8000)
        expect(CORPUS_CALLS[1].ctx_len).to.equal(48)
        expect(CORPUS_CALLS[1].seed).to.equal(20260804)
        expect(CORPUS_CALLS[1].pad_id).to.equal(othello.vocab().pad_id)
        expect(CORPUS_CALLS[1].random_opening_max).to.equal(othello.RANDOM_OPENING_MAX)
    end)

    it("takes the parents and the playout dials from ctx", function()
        configure()
        local out = drive(small({
            styles = { "mobility", "corner" },
            beta = 0.25,
            depth = 4,
            seed = 99,
            random_opening_max = 2,
        }))

        expect(out.styles[1]).to.equal("mobility")
        expect(out.styles[2]).to.equal("corner")
        expect(out.beta).to.equal(0.25)
        expect(out.depth).to.equal(4)
        expect(out.seed).to.equal(99)
        expect(out.random_opening_max).to.equal(2)
        expect(CORPUS_CALLS[1].seed).to.equal(99)
        expect(CORPUS_CALLS[1].random_opening_max).to.equal(2)
    end)

    it("leaves the mixture rules to the policy factory", function()
        -- One account of what a legal mixture is, rather than a copy
        -- here that can drift from it.
        configure()
        expect(contains(drive_error(small({ beta = 0 })), "open interval (0, 1)")).to.equal(true)
        expect(#CORPUS_CALLS).to.equal(0)

        configure()
        expect(contains(drive_error(small({ beta = 1 })), "open interval (0, 1)")).to.equal(true)

        configure()
        expect(
            contains(drive_error(small({ styles = { "corner" } })), "must name exactly two parents")
        ).to.equal(true)

        configure()
        expect(
            contains(
                drive_error(small({ styles = { "corner", "corner" } })),
                "two different parents"
            )
        ).to.equal(true)

        configure()
        expect(
            contains(
                drive_error(small({ styles = { "corner", "aggressive" } })),
                "must be one of corner, mobility, greedy"
            )
        ).to.equal(true)
    end)

    it("refuses a styles field that is not a table", function()
        configure()
        expect(contains(drive_error(small({ styles = "corner" })), "ctx.styles must be a table")).to.equal(
            true
        )
    end)

    it("refuses a depth the teacher does not sweep", function()
        configure()
        expect(
            contains(drive_error(small({ depth = 3 })), "ctx.depth = 3 is not one of 1, 2, 4, 6")
        ).to.equal(true)
        expect(#CORPUS_CALLS).to.equal(0)
    end)

    it("refuses an unknown ctx field, and names the single-style script", function()
        configure()
        local err = drive_error(small({ steps = 100 }))
        expect(contains(err, "unknown ctx field(s) steps")).to.equal(true)
        expect(contains(err, "known: beta, ctx_len, depth, games, path")).to.equal(true)
        expect(#CORPUS_CALLS).to.equal(0)

        configure()
        -- `style` singular is the single-style generator's field, and a
        -- caller reaching for it here wants a different script.
        local singular = drive_error(small({ style = "corner" }))
        expect(contains(singular, "unknown ctx field(s) style")).to.equal(true)
        expect(contains(singular, "gen_othello6_corpus.lua")).to.equal(true)
    end)

    it("refuses a window too narrow for the longest line", function()
        configure()
        local err = drive_error(small({ ctx_len = 40 }))

        expect(contains(err, "ctx.ctx_len = 40 is narrower than the 48 tokens a line needs")).to.equal(
            true
        )
        -- A property of the two numbers alone, so it is answered before
        -- a game is played.
        expect(#CORPUS_CALLS).to.equal(0)
    end)
end)

describe("gen_othello6_mix_corpus file contract", function()
    it("writes the meta the mixed bake driver reads back", function()
        configure()
        local out = drive(small({
            styles = { "corner", "greedy" },
            beta = 0.75,
            depth = 4,
            games = 3,
            seed = 99,
        }))

        local corpus = written_corpus()
        expect(corpus.meta.game).to.equal("othello6")
        -- `kind` is what tells this file apart from a single-style
        -- corpus; without it the two readers cannot refuse each other's
        -- file and a mixture is reported as one parent's result.
        expect(corpus.meta.kind).to.equal("mix")
        expect(corpus.meta.styles[1]).to.equal("corner")
        expect(corpus.meta.styles[2]).to.equal("greedy")
        expect(corpus.meta.beta).to.equal(0.75)
        expect(corpus.meta.depth).to.equal(4)
        expect(corpus.meta.games).to.equal(3)
        expect(corpus.meta.ctx_len).to.equal(48)
        expect(corpus.meta.seed).to.equal(99)
        expect(corpus.meta.random_opening_max).to.equal(othello.RANDOM_OPENING_MAX)
        expect(corpus.meta.vocab_size).to.equal(othello.VOCAB_SIZE)
        expect(corpus.meta.bos).to.equal(othello.BOS)
        expect(corpus.meta.rows_per_game_estimate).to.equal(1)
        expect(#corpus.rows).to.equal(3)

        -- The key set is the contract, so it is pinned whole: a field
        -- added here without being added to the reader is a corpus the
        -- reader cannot check itself against. Note `style` singular is
        -- absent -- that is the single-style file's key.
        expect(key_list(corpus.meta)).to.equal(
            "beta,bos,ctx_len,depth,game,games,kind,random_opening_max,rows_per_game_estimate,"
                .. "seed,styles,vocab_size"
        )

        -- The returned meta is the written one, so a harness can assert
        -- on the file contract without reopening the file.
        expect(key_list(out.meta)).to.equal(key_list(corpus.meta))
        expect(out.meta.kind).to.equal("mix")
    end)

    it("makes the parent directory before it opens the file", function()
        configure()
        drive(small({}))

        -- A first save into a fresh workspace sub-tree would otherwise
        -- fail on the open with a bare "No such file or directory".
        expect(#PARENT_DIRS).to.equal(1)
        expect(PARENT_DIRS[1]).to.equal(CORPUS_FILE)
        expect(OPEN_CALLS[#OPEN_CALLS].path).to.equal(CORPUS_FILE)
        expect(OPEN_CALLS[#OPEN_CALLS].mode).to.equal("w")
    end)

    it("writes rows of the width the encoding produces", function()
        configure()
        -- The real generator and the real mixed policy for this one, on
        -- three games: what lands in the file has to be rows of the
        -- width the bake driver will hold them to.
        CORPUS_REAL = true
        local out = drive(small({ depth = 1, games = 3 }))

        local corpus = written_corpus()
        expect(#corpus.rows).to.equal(3)
        expect(#corpus.rows[1]).to.equal(48)
        expect(#corpus.rows[3]).to.equal(48)
        expect(out.rows).to.equal(3)
    end)

    it("says so when it is replacing a corpus", function()
        configure()
        FILES[CORPUS_FILE] = "{}"
        drive(small({}))

        -- Overwriting retires the comparison the old file was the fixed
        -- condition of, and a mixed corpus costs minutes to replay, so
        -- the replacement appears in the log of the run that did it.
        local said = false
        for _, line in ipairs(LOG_LINES) do
            if contains(line, "already exists and is being replaced") then
                said = true
            end
        end
        expect(said).to.equal(true)
        expect(#written_corpus().rows).to.equal(3)
    end)

    it("says so when the generator answers a different count", function()
        configure()
        -- One game is one row is the encoding's promise, and the meta is
        -- written from the games asked for. A file whose meta and rows
        -- disagree is measured under a size it never had.
        CORPUS_ROWS_PER_GAME = 2
        local err = drive_error(small({ games = 3 }))

        expect(contains(err, "produced 6 rows")).to.equal(true)
        expect(contains(err, "one game is one row")).to.equal(true)
        expect(FILES[CORPUS_FILE]).to.equal(nil)
    end)
end)

describe("gen_othello6_mix_corpus cost reporting", function()
    it("reports what the generation cost", function()
        configure()
        local out = drive(small({ games = 3 }))

        -- The per-playout cost is what a caller sizing the next corpus
        -- multiplies, and it is the number that says whether the "twice
        -- a single-style playout" estimate in the header held.
        expect(type(out.elapsed_seconds)).to.equal("number")
        expect(out.elapsed_seconds >= 0).to.equal(true)
        expect(type(out.seconds_per_game)).to.equal("number")
        expect(out.seconds_per_game).to.equal(out.elapsed_seconds / 3)
    end)

    it("logs the mixture it is generating and what it wrote", function()
        configure()
        drive(small({ games = 3, beta = 0.25 }))

        local announced, wrote = false, false
        for _, line in ipairs(LOG_LINES) do
            if contains(line, "corner/greedy beta=0.25") then
                announced = true
            end
            if contains(line, "wrote 3 rows x 48 tokens to " .. CORPUS_FILE) then
                wrote = true
            end
        end
        expect(announced).to.equal(true)
        expect(wrote).to.equal(true)
    end)
end)
