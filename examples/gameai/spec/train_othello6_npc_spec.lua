-- spec/train_othello6_npc_spec.lua
--
-- Spec for the ctx contract, the pre-flight order, the model shape and
-- the corpus sizing of `train_othello6_npc.lua`.
--
-- The script is a self-contained `alc_run` driver rather than a
-- package, so it is exercised the way it actually runs: the whole file
-- is loaded once per case against a stubbed host, and the assertions
-- read what it handed to `alc.nn.*` on the way through. Everything
-- expensive is a stub -- the model, the dataset and the trainer -- so
-- no `nn` feature and no training budget is involved, and the corpus
-- generator is a spy that either counts the calls and delegates or
-- answers a fixed yield per game, so the sizing can be read off the
-- games it was asked for without playing them.
--
-- `othello6` and `othello6_teacher` are the real modules: the rows one
-- case checks the shape of are the rows a real run would train on.
-- `othello6_npc` is replaced wholesale, because what it answers is a
-- decode of a model that does not exist here.
--
-- The corpus file is stubbed the same way. `io.open` is replaced with an
-- in-memory store and `gameai_metrics._fs` with a recorder, so the
-- reuse cases read and write nothing on disk: a spec that wrote real
-- files would leave a corpus behind on every run and would exercise
-- `mkdir -p` through `os.execute` for no gain. What is asserted is the
-- bytes the driver handed the file and the rows it read back out of
-- them, which is the contract the file carries.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/train_othello6_npc_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })
--
-- so `require("train_othello6_npc")` / `require("othello6")` /
-- `require("othello6_teacher")` all resolve out of that one directory.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Minimal JSON encoder standing in for the host `alc.json_encode`.
---
--- Object keys are emitted in sorted order so a spec can match a fixed
--- substring against the result. The script feeds it the decision
--- payloads it hands to the NPC package and the checkpoint spec it
--- reports.
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

--- Matching decoder, so a corpus the script wrote is read back through
--- the same pair a host would use rather than handed straight back as a
--- Lua table. It covers the subset `json_encode` above emits: objects,
--- arrays, numbers, strings, booleans and null.
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

alc.json_decode = json_decode

-- ─── In-memory filesystem ───────────────────────────────────────────
--
-- `io.open` is replaced for the whole file. Nothing else in this spec
-- touches the disk, and the corpus cases are the only reason a driver
-- run would: a read answers whatever `FILES` holds for the path, and a
-- write lands there on close, so a case can plant a corpus and read
-- back the bytes the driver produced without either ever existing.

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

-- The real helper shells out to `mkdir -p`. The driver has to call it
-- before every write -- that is what keeps a first save into a fresh
-- workspace sub-tree from failing -- so it is recorded rather than
-- removed.
package.preload["gameai_metrics._fs"] = function()
    return {
        ensure_parent_dir = function(path)
            PARENT_DIRS[#PARENT_DIRS + 1] = path
        end,
    }
end

--- Log lines the script emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- `othello6` is the real module and reaches for the host RNG. The LCG
-- below matches the shape the engine-level harness installs; the
-- playouts only have to be reproducible, not statistically sound.
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

--- Host surfaces the script touched, in call order. The pre-flight
--- cases read this as "was any budget spent": an empty log after a
--- rejected ctx means the run stopped before the model was built and
--- before a single game was played.
local CALLS = {}

alc.card = alc.card or {}

--- alias_set invocations in fire order, each `{alias, card_id, opts}`.
local ALIAS_SET_CALLS = {}
alc.card.alias_set = function(alias, card_id, opts)
    ALIAS_SET_CALLS[#ALIAS_SET_CALLS + 1] = { alias = alias, card_id = card_id, opts = opts }
end

--- Final training loss the stub Card reports. Below `math.log(64)`, so
--- the script's "gradients flowed" assertion passes by default and the
--- cases that care about it can move it.
local TRAIN_LOSS = 0.1
alc.card.get = function()
    return { metadata = { nn = { metrics = { train_loss = TRAIN_LOSS } } } }
end

alc.nn = alc.nn or {}

--- `{variant, opts}` of the last `alc.nn.preset.gpt2` call.
local PRESET_CALL = nil

alc.nn.preset = {
    gpt2 = function(variant, opts)
        CALLS[#CALLS + 1] = "preset"
        PRESET_CALL = { variant = variant, opts = opts }
        return {
            ctx = function()
                return opts and opts.ctx or 0
            end,
            vocab = function()
                return opts and opts.vocab or 0
            end,
        }
    end,
}

--- Rows and opts the dataset builder was handed on the last drive.
local DATASET_ROWS = nil
local DATASET_OPTS = nil

alc.nn.data = {
    synthetic = function(rows, opts)
        CALLS[#CALLS + 1] = "synthetic"
        DATASET_ROWS = rows
        DATASET_OPTS = opts
        return { _dataset = true }
    end,
}

--- Opts the script handed to `run_full_ft` on the last drive.
local TRAIN_OPTS = nil

alc.nn.trainer = {
    run_full_ft = function(_, _, opts)
        CALLS[#CALLS + 1] = "train"
        TRAIN_OPTS = opts
        return "card-stub-0001"
    end,
}

-- ─── Package stubs ──────────────────────────────────────────────────

--- Self-play line the NPC stub answers with, swapped per case.
local SELFPLAY_RESULT = "winrate=0.50 illegal=0 style_match=0.99 style_hits=6/8"
--- Determinism line the NPC stub answers with.
local DETERMINISM_RESULT = "deterministic=true action=i"
--- Every `run` opts table the script handed the NPC package.
local NPC_CALLS = {}

--- The NPC package is replaced wholesale: it decodes a model, and the
--- model here is a stub with no weights. The payload arrives as the
--- JSON the script encoded, so the mode is read back out of it.
package.preload["othello6_npc"] = function()
    return {
        reset_cache = function() end,
        run = function(opts)
            NPC_CALLS[#NPC_CALLS + 1] = opts
            local task = tostring(opts.task)
            if task:find('"mode":"determinism"', 1, true) then
                return { result = DETERMINISM_RESULT }
            end
            if task:find('"mode":"selfplay"', 1, true) then
                return { result = SELFPLAY_RESULT }
            end
            error("spec: unexpected npc.run task " .. task, 0)
        end,
    }
end

-- ─── Corpus spy ─────────────────────────────────────────────────────

local othello = require("othello6")

local real_build_corpus = othello.build_corpus

--- Opts every `build_corpus` round was called with, in round order.
local CORPUS_CALLS = {}

--- Rows the spy answers per requested game. `nil` delegates to the real
--- generator, which is what one case wants (the rows a real run trains
--- on have to fit the model context). A number makes the yield a dial:
--- one row per game is what the encoding promises, and another value
--- drives the case that checks a broken promise is reported.
local CORPUS_ROWS_PER_GAME = nil

othello.build_corpus = function(policy, opts)
    CALLS[#CALLS + 1] = "corpus"
    CORPUS_CALLS[#CORPUS_CALLS + 1] = opts
    if CORPUS_ROWS_PER_GAME == nil then
        return real_build_corpus(policy, opts)
    end
    local rows = {}
    for _ = 1, opts.games * CORPUS_ROWS_PER_GAME do
        rows[#rows + 1] = { opts.pad_id }
    end
    return rows
end

-- ─── Driver ─────────────────────────────────────────────────────────

local function configure()
    CALLS = {}
    LOG_LINES = {}
    ALIAS_SET_CALLS = {}
    CORPUS_CALLS = {}
    NPC_CALLS = {}
    FILES = {}
    OPEN_CALLS = {}
    PARENT_DIRS = {}
    PRESET_CALL = nil
    DATASET_ROWS = nil
    DATASET_OPTS = nil
    TRAIN_OPTS = nil
    TRAIN_LOSS = 0.1
    SELFPLAY_RESULT = "winrate=0.50 illegal=0 style_match=0.99 style_hits=6/8"
    DETERMINISM_RESULT = "deterministic=true action=i"
    -- What the encoding promises: one game is one row. The sizing code
    -- is built on that number, so the default spy answers it.
    CORPUS_ROWS_PER_GAME = othello.ROWS_PER_GAME_ESTIMATE
end

--- Load the script once against the current stubs.
---
--- `ctx` is a global in the `alc_run` contract, so it is planted as
--- one here. `package.loaded` is cleared first: every case needs the
--- top-level ctx decoding to run again.
---@param overrides table ctx fields for this case
---@return table result the script's return value
local function drive(overrides)
    ctx = overrides or {}
    package.loaded["train_othello6_npc"] = nil
    return require("train_othello6_npc")
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

--- Whether the script reached a given host surface on the last drive.
local function called(name)
    for _, entry in ipairs(CALLS) do
        if entry == name then
            return true
        end
    end
    return false
end

--- The smallest healthy run: two steps of one row each, one self-play
--- game behind the readings, and a corpus floor far under the design
--- note's 20000 so a case that is not about the sizing does not carry
--- twenty thousand playouts. The floor is the default in the cases that
--- read it.
local function small(overrides)
    local out = { depth = 1, steps = 2, batch = 1, check_games = 1, games = 4 }
    for key, value in pairs(overrides or {}) do
        out[key] = value
    end
    return out
end

-- ─── Corpus file fixtures ───────────────────────────────────────────

--- Path the reuse cases read and write.
local CORPUS_FILE = "workspace/spec/othello6-corpus.json"

--- `small` without `ctx.games`.
---
--- A corpus that is already built holds the games it holds, so the
--- driver refuses a size request against one. Every case that loads a
--- file therefore has to arrive without the floor `small` carries.
local function reuse(overrides)
    local out = small({})
    out.games = nil
    for key, value in pairs(overrides or {}) do
        out[key] = value
    end
    return out
end

--- Meta of a corpus built by the pair `small` runs under.
local function corpus_meta(overrides)
    local out = {
        game = "othello6",
        depth = 1,
        style = "corner",
        games = 4,
        ctx_len = 48,
        seed = 20260804,
        random_opening_max = othello.RANDOM_OPENING_MAX,
        vocab_size = othello.VOCAB_SIZE,
        bos = othello.BOS,
        rows_per_game_estimate = othello.ROWS_PER_GAME_ESTIMATE,
    }
    for key, value in pairs(overrides or {}) do
        out[key] = value
    end
    return out
end

--- `count` rows of `width` pad tokens.
local function corpus_rows(count, width)
    local pad = othello.vocab().pad_id
    local rows = {}
    for index = 1, count do
        local row = {}
        for token = 1, width do
            row[token] = pad
        end
        rows[index] = row
    end
    return rows
end

--- Plant a saved corpus at `CORPUS_FILE`.
---
--- Written through the same encoder the driver saves with, so a case
--- reads it back over the real decode path rather than being handed a
--- Lua table the driver never parsed.
---@param meta table
---@param rows table|nil Defaults to four full-width rows
local function plant_corpus(meta, rows)
    FILES[CORPUS_FILE] = json_encode({ meta = meta, rows = rows or corpus_rows(4, 48) })
end

-- ─── Cases ──────────────────────────────────────────────────────────

describe("train_othello6_npc ctx contract", function()
    it("applies the documented defaults", function()
        configure()
        local out = drive({})

        expect(out.depth).to.equal(2)
        expect(out.style).to.equal("corner")
        expect(out.alias).to.equal("othello6_npc_d2_corner")
        expect(out.name).to.equal("othello6-npc")
        expect(out.seed).to.equal(20260804)
        expect(out.steps).to.equal(800)
        expect(out.batch).to.equal(32)
        expect(out.lr).to.equal(3e-3)
        expect(out.check_games).to.equal(20)
        expect(out.games_floor).to.equal(20000)
        expect(out.random_opening_max).to.equal(othello.RANDOM_OPENING_MAX)
        expect(out.ckpt_every).to.equal(0)
    end)

    it("names the Card alias after the teacher pair", function()
        configure()
        expect(drive(small({ depth = 4, style = "mobility" })).alias).to.equal(
            "othello6_npc_d4_mobility"
        )

        configure()
        expect(drive(small({ alias = "othello6_probe" })).alias).to.equal("othello6_probe")
    end)

    it("refuses a depth the teacher does not carry", function()
        configure()
        local err = drive_error(small({ depth = 3 }))

        expect(contains(err, "ctx.depth = 3 is not one of 1, 2, 4, 6")).to.equal(true)
        expect(#CALLS).to.equal(0)
    end)

    it("refuses an unknown style", function()
        configure()
        local err = drive_error(small({ style = "aggressive" }))

        expect(contains(err, "ctx.style = aggressive is not one of corner, mobility, greedy")).to.equal(
            true
        )
        expect(#CALLS).to.equal(0)
    end)

    it("refuses a field of the wrong type instead of coercing it", function()
        configure()
        expect(contains(drive_error(small({ steps = "800" })), "ctx.steps must be a finite number")).to.equal(
            true
        )

        configure()
        expect(contains(drive_error(small({ batch = 2.5 })), "ctx.batch must be an integer")).to.equal(
            true
        )

        configure()
        expect(contains(drive_error(small({ lr = 0 })), "ctx.lr must be positive")).to.equal(true)

        configure()
        expect(
            contains(drive_error(small({ check_games = 0 })), "ctx.check_games must be at least 1")
        ).to.equal(true)
    end)

    it("refuses an unknown ctx field", function()
        configure()
        local err = drive_error(small({ corpus_games = 4 }))

        expect(contains(err, "unknown ctx field(s) corpus_games")).to.equal(true)
        expect(contains(err, "known: alias, batch, check_games")).to.equal(true)
        expect(#CALLS).to.equal(0)
    end)
end)

describe("train_othello6_npc non-goals", function()
    -- The design note forbids reading a win rate as a level, so the
    -- fields that exist to do it are refused by name rather than
    -- ignored: a dropped `enable_gate` would let a caller believe a run
    -- was gated when nothing read the field.
    local REFUSED = {
        { field = "enable_stages", value = true },
        { field = "stage_bands", value = { { lo = 0.1, hi = 0.3, label = "weak" } } },
        { field = "stage_alias_prefix", value = "othello6_npc" },
        { field = "collection_path", value = "workspace/harvest.json" },
        { field = "enable_gate", value = true },
        { field = "gate_games", value = 50 },
        { field = "target_win_rate_lo", value = 0.55 },
        { field = "teacher_alias", value = "othello6_npc" },
        { field = "tier", value = "mid" },
        { field = "pin_bare_alias", value = false },
    }

    it("refuses every win-rate and staged-harvest field", function()
        for _, case in ipairs(REFUSED) do
            configure()
            local err = drive_error(small({ [case.field] = case.value }))

            expect(contains(err, case.field)).to.equal(true)
            expect(contains(err, "which this experiment does not carry")).to.equal(true)
            -- Refused before anything was built, so a forbidden request
            -- never costs a playout.
            expect(#CALLS).to.equal(0)
        end
    end)

    it("names the reason rather than reporting an unknown field", function()
        configure()
        local err = drive_error(small({ enable_gate = true, gate_games = 10 }))

        expect(contains(err, "enable_gate (win-rate gate)")).to.equal(true)
        expect(contains(err, "gate_games (win-rate gate)")).to.equal(true)
        expect(contains(err, "strength here is the teacher's search depth")).to.equal(true)
        expect(contains(err, "unknown ctx field")).to.equal(false)
    end)

    it("hands the trainer no checkpoint hook", function()
        configure()
        drive(small({ ckpt_every = 2, ckpt_keep = 3 }))

        expect(TRAIN_OPTS.ckpt_every).to.equal(2)
        expect(TRAIN_OPTS.ckpt_keep).to.equal(3)
        -- Checkpoints are written for a later look; nothing in this run
        -- reads a half-trained model, and a hook is what a run that
        -- judged one would need.
        expect(TRAIN_OPTS.on_ckpt).to.equal(nil)
    end)

    it("reports the shape a written checkpoint has to be rebuilt under", function()
        configure()
        local out = drive(small({ ckpt_every = 2 }))

        expect(out.ckpt_arch).to.equal("gpt2-custom")
        expect(contains(out.ckpt_spec, '"ctx":48')).to.equal(true)
        expect(contains(out.ckpt_spec, '"vocab":64')).to.equal(true)
        expect(contains(out.ckpt_spec, '"layers":4')).to.equal(true)

        configure()
        local plain = drive(small({}))
        expect(plain.ckpt_arch).to.equal(nil)
        expect(plain.ckpt_spec).to.equal(nil)
        expect(TRAIN_OPTS.ckpt_every).to.equal(nil)
        expect(TRAIN_OPTS.ckpt_keep).to.equal(nil)
    end)
end)

describe("train_othello6_npc model preset", function()
    it("builds the custom variant with the design shape", function()
        configure()
        drive(small({}))

        expect(PRESET_CALL.variant).to.equal("custom")
        expect(PRESET_CALL.opts.ctx).to.equal(48)
        expect(PRESET_CALL.opts.vocab).to.equal(64)
        expect(PRESET_CALL.opts.layers).to.equal(4)
        expect(PRESET_CALL.opts.heads).to.equal(4)
        expect(PRESET_CALL.opts.dim).to.equal(128)
        expect(PRESET_CALL.opts.device).to.equal("cpu")
        expect(PRESET_CALL.opts.dtype).to.equal("f32")
    end)

    it("asks for a random-init model", function()
        configure()
        local out = drive(small({}))

        -- Full FT of a customized GPT-2 is random-init only, and the
        -- bridge refuses a custom spec with `pretrained = true`.
        expect(PRESET_CALL.opts.pretrained).to.equal(false)
        expect(out.ctx_len).to.equal(48)
        expect(out.model_vocab).to.equal(64)
        expect(out.vocab_size).to.equal(othello.VOCAB_SIZE)
    end)

    it("defaults the shape dials to the design bid", function()
        configure()
        local out = drive(small({}))

        -- Four layers at 128 dimensions: the published 6x6 result sits
        -- at that width, and the attempt that went narrow and deep
        -- instead (six layers at 96) read legal play at 9.1%.
        expect(out.layers).to.equal(4)
        expect(out.heads).to.equal(4)
        expect(out.dim).to.equal(128)
    end)

    it("takes the depth and the width from ctx", function()
        configure()
        local out = drive(small({ layers = 6, heads = 8, dim = 256 }))

        expect(PRESET_CALL.opts.layers).to.equal(6)
        expect(PRESET_CALL.opts.heads).to.equal(8)
        expect(PRESET_CALL.opts.dim).to.equal(256)
        -- Reported back, so a sweep reads the shape a number came out of
        -- off the result rather than off its own request.
        expect(out.layers).to.equal(6)
        expect(out.heads).to.equal(8)
        expect(out.dim).to.equal(256)
        -- The window and the alphabet do not move with them.
        expect(PRESET_CALL.opts.ctx).to.equal(48)
        expect(PRESET_CALL.opts.vocab).to.equal(64)

        configure()
        -- A raw checkpoint carries weights and no shape, so the reported
        -- spec has to be the shape this run asked for.
        local ckpt = drive(small({ layers = 6, dim = 256, ckpt_every = 2 }))
        expect(contains(ckpt.ckpt_spec, '"layers":6')).to.equal(true)
        expect(contains(ckpt.ckpt_spec, '"dim":256')).to.equal(true)
    end)

    it("refuses a width the heads cannot split", function()
        configure()
        local err = drive_error(small({ heads = 5 }))

        expect(contains(err, "ctx.dim = 128 is not a multiple of ctx.heads = 5")).to.equal(true)
        -- A property of the two numbers alone, so it is answered before
        -- the model is built and before a game is played.
        expect(#CALLS).to.equal(0)
    end)

    it("refuses a shape dial that is not a positive integer", function()
        configure()
        expect(contains(drive_error(small({ layers = 0 })), "ctx.layers must be at least 1")).to.equal(
            true
        )

        configure()
        expect(contains(drive_error(small({ dim = 64.5 })), "ctx.dim must be an integer")).to.equal(
            true
        )
    end)

    it("refuses the context window and the vocabulary", function()
        configure()
        -- Both are properties of the encoding: a model built at another
        -- window would not fit the rows it is then trained on, so they
        -- are refused by name rather than reported as unknown.
        local err = drive_error(small({ ctx = 96 }))

        expect(contains(err, "ctx (the corpus row length)")).to.equal(true)
        expect(contains(err, "fixed by the corpus encoding")).to.equal(true)
        expect(contains(err, "unknown ctx field")).to.equal(false)
        expect(#CALLS).to.equal(0)

        configure()
        local vocab_err = drive_error(small({ vocab = 128 }))

        expect(contains(vocab_err, "vocab (the alphabet size)")).to.equal(true)
        expect(contains(vocab_err, "unknown ctx field")).to.equal(false)
        expect(#CALLS).to.equal(0)
    end)

    it("builds the model before it spends the corpus budget", function()
        configure()
        drive(small({}))

        expect(CALLS[1]).to.equal("preset")
        expect(CALLS[2]).to.equal("corpus")
        expect(CALLS[3]).to.equal("synthetic")
        expect(CALLS[4]).to.equal("train")
    end)
end)

describe("train_othello6_npc corpus sizing", function()
    it("covers steps * batch rows with a batch to spare", function()
        configure()
        local out = drive(small({ steps = 5, batch = 4 }))

        expect(out.rows_target).to.equal(5 * 4 + 4)
        expect(out.rows >= 5 * 4).to.equal(true)
        expect(out.rows >= out.rows_target).to.equal(true)
        expect(#DATASET_ROWS).to.equal(out.rows)
        expect(DATASET_OPTS.batch_size).to.equal(4)
        expect(DATASET_OPTS.ctx_len).to.equal(48)
    end)

    it("plays one game per row the run consumes", function()
        configure()
        -- One game is one row under the move-sequence encoding, so the
        -- count is arithmetic: a run consuming 101 rows asks for 101
        -- playouts, in one call, and gets exactly that many rows back.
        local out = drive(small({ steps = 100, batch = 1 }))

        expect(out.rows_target).to.equal(101)
        expect(#CORPUS_CALLS).to.equal(1)
        expect(CORPUS_CALLS[1].games).to.equal(101)
        expect(CORPUS_CALLS[1].seed).to.equal(20260804)
        expect(out.games).to.equal(101)
        expect(out.rows).to.equal(101)
        -- The playout count and the row count are the same number, and
        -- the result says so rather than leaving it to be inferred.
        expect(out.rows).to.equal(out.games)
    end)

    it("opens at the design note's twenty thousand games", function()
        configure()
        -- The published readings are per corpus size -- 1k games read
        -- legal play at 20.8%, 20k at 79.7% -- and 20k is where this
        -- experiment starts, so a run that consumes fewer rows than
        -- that still plays them.
        local out = drive({ depth = 1, steps = 2, batch = 1, check_games = 1 })

        expect(out.games_floor).to.equal(20000)
        expect(CORPUS_CALLS[1].games).to.equal(20000)
        expect(out.games).to.equal(20000)
        expect(out.rows).to.equal(20000)
        expect(out.rows_target).to.equal(3)
    end)

    it("plays past the floor when the training budget needs more rows", function()
        configure()
        -- Two floors on one number: `ctx.games` and the rows the
        -- trainer consumes. The larger one decides, because a corpus
        -- short of the budget is a trainer that runs out of data.
        local out = drive(small({ steps = 40, batch = 1, games = 10 }))

        expect(out.rows_target).to.equal(41)
        expect(CORPUS_CALLS[1].games).to.equal(41)
        expect(out.games).to.equal(41)
        expect(out.games_floor).to.equal(10)
    end)

    it("says so when the generator answers fewer rows than games", function()
        configure()
        -- One game is one row is the encoding's promise, and the sizing
        -- is arithmetic on it. A generator that breaks it is reported
        -- rather than topped up, because a corpus that quietly came up
        -- short is a training run that stops mid-way through.
        CORPUS_ROWS_PER_GAME = 0
        local err = drive_error(small({ steps = 4, batch = 1 }))

        expect(contains(err, "one game is one row")).to.equal(true)
        expect(contains(err, "fewer rows than it was asked for games")).to.equal(true)
    end)

    it("honours ctx.games as a floor on the playouts", function()
        configure()
        local out = drive(small({ steps = 2, batch = 1, games = 40 }))

        expect(CORPUS_CALLS[1].games).to.equal(40)
        expect(out.games).to.equal(40)
        expect(out.rows).to.equal(40)
        expect(out.games_floor).to.equal(40)
    end)

    it("labels the rows through the requested teacher and encoding", function()
        configure()
        drive(small({ depth = 4, style = "greedy", random_opening_max = 2 }))

        local opts = CORPUS_CALLS[1]
        expect(opts.ctx_len).to.equal(48)
        expect(opts.pad_id).to.equal(othello.vocab().pad_id)
        expect(opts.random_opening_max).to.equal(2)
    end)

    it("builds real rows that fit the model context", function()
        configure()
        -- The real generator for one case, on the smallest floor that
        -- still plays games: the rows a run actually trains on have to
        -- be the width the model was built at, and the real generator
        -- has to answer the one row per game the sizing counts on.
        CORPUS_ROWS_PER_GAME = nil
        local out = drive(small({ depth = 1, games = 4 }))

        expect(out.games).to.equal(4)
        expect(out.rows).to.equal(4)
        expect(out.rows >= out.rows_target).to.equal(true)
        expect(#DATASET_ROWS[1]).to.equal(48)
        expect(#DATASET_ROWS[out.rows]).to.equal(48)
    end)
end)

describe("train_othello6_npc corpus reuse", function()
    it("generates and keeps nothing when no path is named", function()
        configure()
        local out = drive(small({}))

        -- The behaviour every run had before the field existed: play the
        -- games, train, and let them go.
        expect(out.corpus_source).to.equal("generated")
        expect(out.corpus_path).to.equal(nil)
        expect(#CORPUS_CALLS).to.equal(1)
        expect(#OPEN_CALLS).to.equal(0)
        expect(#PARENT_DIRS).to.equal(0)
        expect(next(FILES)).to.equal(nil)
    end)

    it("saves the corpus it generated when the file is not there yet", function()
        configure()
        -- The real generator for this one: what lands in the file has to
        -- be rows of the width the reader will later hold them to, and
        -- the spy's stand-in rows are one token wide.
        CORPUS_ROWS_PER_GAME = nil
        local out = drive(small({ corpus_path = CORPUS_FILE }))

        expect(out.corpus_source).to.equal("generated_and_saved")
        expect(out.corpus_path).to.equal(CORPUS_FILE)
        expect(#CORPUS_CALLS).to.equal(1)
        -- The parent directory is made first: a save into a fresh
        -- workspace sub-tree would otherwise fail on the open.
        expect(#PARENT_DIRS).to.equal(1)
        expect(PARENT_DIRS[1]).to.equal(CORPUS_FILE)

        local written = json_decode(FILES[CORPUS_FILE])
        expect(#written.rows).to.equal(out.rows)
        expect(#written.rows[1]).to.equal(48)
        expect(written.meta.game).to.equal("othello6")
        expect(written.meta.depth).to.equal(1)
        expect(written.meta.style).to.equal("corner")
        expect(written.meta.ctx_len).to.equal(48)
        expect(written.meta.vocab_size).to.equal(othello.VOCAB_SIZE)
        expect(written.meta.games).to.equal(out.games)
        expect(written.meta.seed).to.equal(20260804)
        expect(written.meta.bos).to.equal(othello.BOS)
        expect(written.meta.rows_per_game_estimate).to.equal(1)
    end)

    it("reads an existing corpus instead of playing the games again", function()
        configure()
        plant_corpus(corpus_meta())
        local out = drive(reuse({ corpus_path = CORPUS_FILE }))

        -- The whole point of the field: not one playout is spent.
        expect(#CORPUS_CALLS).to.equal(0)
        expect(CALLS[1]).to.equal("preset")
        expect(CALLS[2]).to.equal("synthetic")
        expect(out.corpus_source).to.equal("loaded")
        expect(out.corpus_path).to.equal(CORPUS_FILE)
        -- The rows the trainer saw are the rows in the file.
        expect(out.rows).to.equal(4)
        expect(#DATASET_ROWS).to.equal(4)
        expect(#DATASET_ROWS[1]).to.equal(48)
        -- The conditions come off the file, and the floor never applied.
        expect(out.games).to.equal(4)
        expect(out.games_floor).to.equal(nil)
        expect(out.corpus_seed).to.equal(20260804)
        -- Read once, written never.
        expect(#OPEN_CALLS).to.equal(1)
        expect(OPEN_CALLS[1].mode).to.equal("r")
        expect(#PARENT_DIRS).to.equal(0)
    end)

    it("reports the file's own seed and opening draw", function()
        configure()
        plant_corpus(corpus_meta({ seed = 777, random_opening_max = 3 }))
        local out = drive(reuse({ corpus_path = CORPUS_FILE, seed = 20260804 }))

        -- A sweep reuses one corpus under whatever seed built it, so the
        -- difference is reported rather than refused -- and the result
        -- carries the corpus's values, not the ones ctx asked for.
        expect(out.corpus_source).to.equal("loaded")
        expect(out.corpus_seed).to.equal(777)
        expect(out.random_opening_max).to.equal(3)
        expect(out.seed).to.equal(20260804)
    end)

    it("refuses a corpus built under another condition", function()
        local MISMATCHES = {
            {
                meta = { game = "othello8" },
                needle = "meta.game = othello8",
                why = "the rows encode another game entirely",
            },
            {
                meta = { ctx_len = 40 },
                needle = "meta.ctx_len = 40",
                why = "rows of another width do not fit the model this run builds",
            },
            {
                meta = { vocab_size = 45 },
                needle = "meta.vocab_size = 45",
                why = "the same token id stands for another move under another alphabet",
            },
            {
                meta = { depth = 2 },
                needle = "meta.depth = 2",
                why = "a Card is then measured against a teacher that never labelled it",
            },
            {
                meta = { style = "greedy" },
                needle = "meta.style = greedy",
                why = "a Card is then measured against a teacher that never labelled it",
            },
        }

        for _, case in ipairs(MISMATCHES) do
            configure()
            -- `ctx_len` also decides the row width the reader checks, so
            -- the rows are planted at the width the meta claims: the case
            -- has to fail on the meta rather than on a ragged row.
            local meta = corpus_meta(case.meta)
            plant_corpus(meta, corpus_rows(4, meta.ctx_len))
            local err = drive_error(reuse({ corpus_path = CORPUS_FILE }))

            expect(contains(err, case.needle)).to.equal(true)
            expect(contains(err, case.why)).to.equal(true)
            -- Refused rather than trained around: nothing reached the
            -- trainer under a condition it was not measured for.
            expect(called("train")).to.equal(false)
        end
    end)

    it("refuses a row that is not the width the meta claims", function()
        configure()
        local rows = corpus_rows(4, 48)
        rows[3] = corpus_rows(1, 47)[1]
        plant_corpus(corpus_meta(), rows)
        local err = drive_error(reuse({ corpus_path = CORPUS_FILE }))

        -- Every row is measured, not just the first: a file that is
        -- right at the head and ragged further in trains fine and
        -- decodes to nothing.
        expect(contains(err, "row 3 of corpus_path")).to.equal(true)
        expect(contains(err, "47 tokens wide but meta.ctx_len is 48")).to.equal(true)
    end)

    it("refuses a corpus that holds no rows", function()
        configure()
        plant_corpus(corpus_meta(), {})
        local err = drive_error(reuse({ corpus_path = CORPUS_FILE }))

        expect(contains(err, "holds no rows")).to.equal(true)
    end)

    it("refuses a corpus shorter than the training budget", function()
        configure()
        plant_corpus(corpus_meta({ games = 4 }))
        -- Nine steps of batch one consume ten rows with the spare batch,
        -- and the file holds four. The rows are fixed by now, so the
        -- budget is what has to give -- and it is said before the
        -- trainer walks off the end of the data.
        local err = drive_error(reuse({ corpus_path = CORPUS_FILE, steps = 9, batch = 1 }))

        expect(contains(err, "holds 4 rows but 9 steps of batch 1 consume 10")).to.equal(true)
        expect(contains(err, "lower steps or batch, or generate a larger corpus")).to.equal(true)
    end)

    it("refuses a size request against a corpus that is already built", function()
        configure()
        plant_corpus(corpus_meta())
        local err = drive_error(reuse({ corpus_path = CORPUS_FILE, games = 20000 }))

        -- Honouring it is impossible, so it is refused rather than
        -- dropped: a silently ignored floor reports the numbers of a
        -- corpus the caller did not ask for.
        expect(contains(err, "ctx.games asks for a corpus size")).to.equal(true)
        expect(contains(err, "property of the file")).to.equal(true)
    end)

    it("refuses a corpus_path that is not a non-empty string", function()
        configure()
        expect(
            contains(drive_error(small({ corpus_path = 7 })), "ctx.corpus_path must be a string")
        ).to.equal(true)

        configure()
        expect(
            contains(drive_error(small({ corpus_path = "" })), "ctx.corpus_path must not be empty")
        ).to.equal(true)
    end)
end)

describe("train_othello6_npc greedy readings", function()
    it("returns the two measurements and the loss floor", function()
        configure()
        SELFPLAY_RESULT = "winrate=0.50 illegal=0 style_match=0.99 style_hits=6/8"
        local out = drive(small({}))

        -- Recomputed from the counts rather than parsed off the
        -- formatted field, which is rounded to two decimals.
        expect(out.style_match).to.equal(6 / 8)
        expect(out.style_hits).to.equal(6)
        expect(out.moves).to.equal(8)
        expect(out.legal_rate).to.equal(1.0)
        expect(out.illegal).to.equal(0)
        expect(out.train_loss).to.equal(0.1)
        expect(out.baseline_loss).to.equal(math.log(64))
        expect(out.loss_descended).to.equal(true)
        expect(out.deterministic).to.equal(true)
        expect(out.ok).to.equal(true)
    end)

    it("counts a raw-illegal decision against the legal rate", function()
        configure()
        SELFPLAY_RESULT = "winrate=0.50 illegal=2 style_match=0.75 style_hits=6/8"
        local out = drive(small({}))

        expect(out.legal_rate).to.equal(0.75)
        expect(out.illegal).to.equal(2)
        -- Every raw decision has to be legal for the run to pass.
        expect(out.ok).to.equal(false)
    end)

    it("fails the run when the loss did not go under the baseline", function()
        configure()
        TRAIN_LOSS = 5.0
        local out = drive(small({}))

        expect(out.loss_descended).to.equal(false)
        expect(out.ok).to.equal(false)
    end)

    it("reports the win rate without reading it", function()
        configure()
        SELFPLAY_RESULT = "winrate=0.05 illegal=0 style_match=0.99 style_hits=8/8"
        local out = drive(small({}))

        -- A win rate of 0.05 is not a verdict here: strength is the
        -- teacher's search depth, so nothing branches on this number.
        expect(out.winrate).to.equal(0.05)
        expect(out.ok).to.equal(true)
    end)

    it("refuses a self-play answer it cannot read", function()
        configure()
        SELFPLAY_RESULT = "style_match=0.75"
        local err = drive_error(small({}))

        expect(contains(err, "the self-play answer is not in the expected shape")).to.equal(true)
    end)

    it("asks the NPC package under the Card and the teacher pair", function()
        configure()
        drive(small({ depth = 4, style = "mobility", check_games = 3 }))

        expect(#NPC_CALLS).to.equal(2)
        for _, call in ipairs(NPC_CALLS) do
            expect(call.card_alias).to.equal("othello6_npc_d4_mobility")
            expect(call.depth).to.equal(4)
            expect(call.style).to.equal("mobility")
        end
        expect(contains(NPC_CALLS[1].task, '"mode":"determinism"')).to.equal(true)
        expect(contains(NPC_CALLS[2].task, '"mode":"selfplay"')).to.equal(true)
        expect(contains(NPC_CALLS[2].task, '"games":3')).to.equal(true)
    end)
end)

describe("train_othello6_npc alias pinning", function()
    it("pins the trained alias", function()
        configure()
        drive(small({ depth = 1, style = "greedy" }))

        expect(#ALIAS_SET_CALLS).to.equal(1)
        expect(ALIAS_SET_CALLS[1].alias).to.equal("othello6_npc_d1_greedy")
        expect(ALIAS_SET_CALLS[1].card_id).to.equal("card-stub-0001")
        expect(ALIAS_SET_CALLS[1].opts.pkg).to.equal("othello6_npc")
    end)

    it("claims the bare alias only for the pair the NPC defaults to", function()
        configure()
        local out = drive(small({ depth = 2, style = "corner" }))

        -- `othello6_npc` reads the bare alias with depth 2 / corner as
        -- its basis, so only that pair may claim it: a Card baked under
        -- another teacher would be scored against one that never
        -- labelled it.
        expect(#ALIAS_SET_CALLS).to.equal(2)
        expect(ALIAS_SET_CALLS[2].alias).to.equal("othello6_npc")
        expect(out.pinned_bare_alias).to.equal(true)

        configure()
        local other = drive(small({ depth = 2, style = "greedy" }))
        expect(#ALIAS_SET_CALLS).to.equal(1)
        expect(other.pinned_bare_alias).to.equal(false)
    end)
end)
