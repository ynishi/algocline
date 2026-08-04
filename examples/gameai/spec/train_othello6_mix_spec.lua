-- spec/train_othello6_mix_spec.lua
--
-- Spec for the ctx contract, the mixed-corpus meta check, the model
-- shape and the two-parent compliance reading of
-- `train_othello6_mix.lua`.
--
-- The script is a self-contained `alc_run` driver rather than a package,
-- so it is exercised the way it actually runs: the whole file is loaded
-- once per case against a stubbed host, and the assertions read what it
-- handed to `alc.nn.*` and to the NPC package on the way through.
--
-- Everything expensive is a stub. The model, the dataset and the trainer
-- answer without touching the `nn` feature, so no training budget is
-- involved; `othello6_npc` is replaced wholesale, because what it
-- answers is a decode of a model that does not exist here; and no corpus
-- is ever generated, which this driver could not do anyway -- it reads a
-- file or refuses. The corpus files are planted in an in-memory
-- filesystem through the same encoder the generator writes with, so the
-- meta checks run over the real decode path rather than over a Lua table
-- the driver never parsed.
--
-- Run it with `examples/gameai` on the search path, e.g.
--
--     test_launch(code_file = "examples/gameai/spec/train_othello6_mix_spec.lua",
--                 search_paths = { "<repo>/examples/gameai" })
--
-- so `require("train_othello6_mix")` / `require("othello6")` all resolve
-- out of that one directory.

local describe, it, expect = lust.describe, lust.it, lust.expect

-- ─── Host stubs ─────────────────────────────────────────────────────

alc = alc or {}

--- Minimal JSON encoder standing in for the host `alc.json_encode`.
---
--- Object keys are emitted in sorted order so a spec can match a fixed
--- substring against the payloads the driver hands the NPC package.
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

--- Matching decoder, so a planted corpus is read back through the same
--- pair a host would use. It covers the subset `json_encode` emits:
--- objects, arrays, numbers, strings, booleans and null.
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

--- Log lines the script emitted during the last drive.
local LOG_LINES = {}
alc.log = function(_, message)
    LOG_LINES[#LOG_LINES + 1] = tostring(message)
end

-- ─── In-memory filesystem ───────────────────────────────────────────
--
-- The driver only ever reads, so a case plants a corpus in `FILES` and
-- an absent key is the "not generated yet" branch the driver refuses.

--- Path -> file contents.
local FILES = {}

io.open = function(path, mode)
    mode = mode or "r"
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
    error("spec io.open: this driver does not write (" .. tostring(mode) .. ")", 0)
end

-- `othello6` is the real module and reaches for the host RNG. The LCG
-- below matches the shape the engine-level harness installs.
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

--- Host surfaces the driver touched, in call order. The pre-flight cases
--- read this as "was any budget spent": an empty log after a rejected
--- ctx means the run stopped before the trainer was reached.
local CALLS = {}

alc.card = alc.card or {}

--- alias_set invocations in fire order, each `{alias, card_id, opts}`.
local ALIAS_SET_CALLS = {}
alc.card.alias_set = function(alias, card_id, opts)
    ALIAS_SET_CALLS[#ALIAS_SET_CALLS + 1] = { alias = alias, card_id = card_id, opts = opts }
end

--- Final training loss the stub Card reports. Below `math.log(64)`, so
--- the driver's "gradients flowed" assertion passes by default.
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

--- Opts the driver handed to `run_full_ft` on the last drive.
local TRAIN_OPTS = nil

alc.nn.trainer = {
    run_full_ft = function(_, _, opts)
        CALLS[#CALLS + 1] = "train"
        TRAIN_OPTS = opts
        return "card-stub-0001"
    end,
}

-- ─── Package stubs ──────────────────────────────────────────────────

--- Self-play line the NPC stub answers, keyed by the parent the task
--- named. Both carry the same `illegal` and the same move count, which
--- is what a single Card playing the same games under the same seed
--- produces; only the hit count moves with the teacher.
local SELFPLAY_BY_STYLE = {}
--- Every `run` opts table the driver handed the NPC package.
local NPC_CALLS = {}

--- The NPC package is replaced wholesale: it decodes a model, and the
--- model here is a stub with no weights. The payload arrives as the
--- JSON the driver encoded, so the style is read back out of it.
package.preload["othello6_npc"] = function()
    return {
        reset_cache = function() end,
        run = function(opts)
            NPC_CALLS[#NPC_CALLS + 1] = opts
            local task = tostring(opts.task)
            if not task:find('"mode":"selfplay"', 1, true) then
                error("spec: unexpected npc.run task " .. task, 0)
            end
            local style = task:match('"style":"([%a_]+)"')
            local line = SELFPLAY_BY_STYLE[style]
            if line == nil then
                error("spec: no self-play line planted for style " .. tostring(style), 0)
            end
            return { result = line }
        end,
    }
end

local othello = require("othello6")

-- ─── Corpus fixtures ────────────────────────────────────────────────

--- Path every case reads.
local CORPUS_FILE = "workspace/spec/othello6-mix-corpus.json"

--- Meta of a mixed corpus as `gen_othello6_mix_corpus.lua` writes it.
local function mix_meta(overrides)
    local out = {
        game = "othello6",
        kind = "mix",
        styles = { "corner", "greedy" },
        beta = 0.5,
        depth = 2,
        games = 8,
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

--- Meta of a single-style corpus, i.e. the file a caller is most likely
--- to have pointed at this driver by mistake.
local function single_meta()
    return {
        game = "othello6",
        depth = 2,
        style = "corner",
        games = 8,
        ctx_len = 48,
        seed = 20260804,
        random_opening_max = othello.RANDOM_OPENING_MAX,
        vocab_size = othello.VOCAB_SIZE,
        bos = othello.BOS,
        rows_per_game_estimate = othello.ROWS_PER_GAME_ESTIMATE,
    }
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

--- Plant a corpus at `CORPUS_FILE`, written through the encoder the
--- generator saves with.
---@param meta table
---@param rows table|nil Defaults to eight full-width rows
local function plant_corpus(meta, rows)
    FILES[CORPUS_FILE] = json_encode({ meta = meta, rows = rows or corpus_rows(8, 48) })
end

-- ─── Driver ─────────────────────────────────────────────────────────

local function configure()
    CALLS = {}
    LOG_LINES = {}
    ALIAS_SET_CALLS = {}
    NPC_CALLS = {}
    FILES = {}
    PRESET_CALL = nil
    DATASET_ROWS = nil
    DATASET_OPTS = nil
    TRAIN_OPTS = nil
    TRAIN_LOSS = 0.1
    -- Same games, same raw-illegal count, different agreement: one Card
    -- scored against two teachers.
    SELFPLAY_BY_STYLE = {
        corner = "winrate=0.50 illegal=2 style_match=0.60 style_hits=6/10",
        greedy = "winrate=0.50 illegal=2 style_match=0.40 style_hits=4/10",
        mobility = "winrate=0.50 illegal=2 style_match=0.30 style_hits=3/10",
    }
    plant_corpus(mix_meta())
end

--- Load the driver once against the current stubs.
---@param overrides table ctx fields for this case
---@return table result the driver's return value
local function drive(overrides)
    ctx = overrides or {}
    package.loaded["train_othello6_mix"] = nil
    return require("train_othello6_mix")
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

--- Whether the driver reached a given host surface on the last drive.
local function called(name)
    for _, entry in ipairs(CALLS) do
        if entry == name then
            return true
        end
    end
    return false
end

--- The smallest healthy run: two steps of one row each against the
--- planted eight-row corpus, and one self-play game behind each reading.
local function small(overrides)
    local out = { corpus_path = CORPUS_FILE, steps = 2, batch = 1, check_games = 1 }
    for key, value in pairs(overrides or {}) do
        out[key] = value
    end
    return out
end

-- ─── Cases ──────────────────────────────────────────────────────────

describe("train_othello6_mix ctx contract", function()
    it("requires a corpus and never generates one", function()
        configure()
        local err = drive_error({ steps = 2, batch = 1 })

        -- A mixed playout costs about twice a single-style one, so a
        -- driver that could quietly generate turns a forty-second bake
        -- into a four-minute one whenever a path is mistyped.
        expect(contains(err, "ctx.corpus_path is required")).to.equal(true)
        expect(contains(err, "gen_othello6_mix_corpus.lua")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)

    it("refuses a corpus_path that is not there rather than playing one", function()
        configure()
        FILES[CORPUS_FILE] = nil
        local err = drive_error(small({}))

        expect(contains(err, "does not exist")).to.equal(true)
        expect(contains(err, "never generates one")).to.equal(true)
        expect(contains(err, "gen_othello6_mix_corpus.lua")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)

    it("refuses the fields that belong to the other two scripts", function()
        configure()
        -- A mixed corpus holds the games it holds; sizing is the
        -- generator's job.
        expect(contains(drive_error(small({ games = 4000 })), "games (")).to.equal(true)

        configure()
        -- `style` singular asks for a single-teacher bake.
        local singular = drive_error(small({ style = "corner" }))
        expect(contains(singular, "the field is `styles`")).to.equal(true)
        expect(contains(singular, "train_othello6_npc.lua")).to.equal(true)

        configure()
        expect(contains(drive_error(small({ ctx = 64 })), "fixed by the corpus encoding")).to.equal(
            true
        )

        configure()
        local unknown = drive_error(small({ enable_gate = true }))
        expect(contains(unknown, "unknown ctx field(s) enable_gate")).to.equal(true)
        expect(called("preset")).to.equal(false)
    end)

    it("builds the shape the measurements chose", function()
        configure()
        drive(small({}))

        -- Two layers, not four: the depth axis was measured over a 20k
        -- corpus and is monotone the wrong way.
        expect(PRESET_CALL.variant).to.equal("custom")
        expect(PRESET_CALL.opts.layers).to.equal(2)
        expect(PRESET_CALL.opts.dim).to.equal(128)
        expect(PRESET_CALL.opts.heads).to.equal(4)
        expect(PRESET_CALL.opts.ctx).to.equal(48)
        expect(PRESET_CALL.opts.vocab).to.equal(64)
        expect(PRESET_CALL.opts.pretrained).to.equal(false)
        expect(PRESET_CALL.opts.device).to.equal("cpu")
    end)

    it("takes the training budget and the width from ctx", function()
        configure()
        local out = drive(small({ steps = 3, batch = 2, lr = 1e-3, layers = 4, dim = 64 }))

        expect(PRESET_CALL.opts.layers).to.equal(4)
        expect(PRESET_CALL.opts.dim).to.equal(64)
        expect(TRAIN_OPTS.steps).to.equal(3)
        expect(TRAIN_OPTS.batch).to.equal(2)
        expect(TRAIN_OPTS.lr).to.equal(1e-3)
        expect(DATASET_OPTS.batch_size).to.equal(2)
        expect(DATASET_OPTS.pad_id).to.equal(othello.vocab().pad_id)
        expect(out.steps).to.equal(3)
        expect(out.layers).to.equal(4)
    end)

    it("refuses a width the heads cannot split", function()
        configure()
        local err = drive_error(small({ dim = 100, heads = 8 }))

        expect(contains(err, "is not a multiple of ctx.heads")).to.equal(true)
        -- A property of the two numbers alone, so the corpus is never
        -- opened.
        expect(called("preset")).to.equal(false)
    end)
end)

describe("train_othello6_mix corpus meta", function()
    it("refuses a single-style corpus by name", function()
        configure()
        plant_corpus(single_meta())
        local err = drive_error(small({}))

        -- The one file that passes every other check: same game, same
        -- width, same alphabet. Without `kind` the run would report one
        -- parent's teacher as a mixture.
        expect(contains(err, "meta.kind = nothing")).to.equal(true)
        expect(contains(err, 'reads "mix" corpora only')).to.equal(true)
        expect(contains(err, "train_othello6_npc.lua")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)

    it("refuses a corpus of another kind", function()
        configure()
        plant_corpus(mix_meta({ kind = "board" }))
        expect(contains(drive_error(small({})), 'meta.kind = "board"')).to.equal(true)
    end)

    it("refuses a game, a width or an alphabet this run cannot read", function()
        configure()
        plant_corpus(mix_meta({ game = "guardian_duel" }))
        local game_err = drive_error(small({}))
        expect(contains(game_err, "meta.game = guardian_duel")).to.equal(true)
        expect(contains(game_err, "another game entirely")).to.equal(true)

        configure()
        plant_corpus(mix_meta({ ctx_len = 32 }), corpus_rows(8, 32))
        local width_err = drive_error(small({}))
        expect(contains(width_err, "meta.ctx_len = 32")).to.equal(true)
        expect(contains(width_err, "do not fit the model this run builds")).to.equal(true)

        configure()
        plant_corpus(mix_meta({ vocab_size = 12 }))
        local vocab_err = drive_error(small({}))
        expect(contains(vocab_err, "meta.vocab_size = 12")).to.equal(true)
        expect(contains(vocab_err, "another alphabet")).to.equal(true)
    end)

    it("refuses a row of the wrong width", function()
        configure()
        local rows = corpus_rows(8, 48)
        rows[5] = { othello.vocab().pad_id }
        plant_corpus(mix_meta(), rows)

        -- Every row is measured, not just the first: a file that is
        -- right at the head and ragged further in trains fine and
        -- decodes to nothing.
        expect(contains(drive_error(small({})), "row 5 of corpus_path")).to.equal(true)
    end)

    it("refuses a corpus that cannot carry the training budget", function()
        configure()
        -- Eight rows against 8 * 1 + 1 = 9 consumed. `alc.nn.data
        -- .synthetic` walks its rows once, so a short corpus is a
        -- trainer that stops part of the way through.
        local err = drive_error(small({ steps = 8, batch = 1 }))

        expect(contains(err, "holds 8 rows but 8 steps of batch 1 consume 9")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)

    it("refuses a meta whose mixture is not one", function()
        configure()
        plant_corpus(mix_meta({ styles = { "corner", "corner" } }))
        expect(contains(drive_error(small({})), "as both parents")).to.equal(true)

        configure()
        -- Deleted rather than passed as an override: `pairs` skips a
        -- nil value, so the key would survive the merge.
        local without_styles = mix_meta()
        without_styles.styles = nil
        plant_corpus(without_styles)
        expect(contains(drive_error(small({})), "no meta.styles pair")).to.equal(true)

        configure()
        plant_corpus(mix_meta({ beta = 1 }))
        expect(contains(drive_error(small({})), "meta.beta = 1")).to.equal(true)

        configure()
        plant_corpus(mix_meta({ depth = 3 }))
        expect(contains(drive_error(small({})), "meta.depth = 3 is not one of 1, 2, 4, 6")).to.equal(
            true
        )
    end)
end)

describe("train_othello6_mix mixture resolution", function()
    it("takes the mixture from the corpus when ctx is silent", function()
        configure()
        plant_corpus(mix_meta({ styles = { "mobility", "greedy" }, beta = 0.25, depth = 4 }))
        local out = drive(small({}))

        -- The rows were labelled under one weight and no request here
        -- can change that, so the file is the source and the result
        -- carries its values rather than leaving a reader to infer them.
        expect(out.styles[1]).to.equal("mobility")
        expect(out.styles[2]).to.equal("greedy")
        expect(out.style_a).to.equal("mobility")
        expect(out.style_b).to.equal("greedy")
        expect(out.beta).to.equal(0.25)
        expect(out.depth).to.equal(4)
        expect(out.kind).to.equal("mix")
        expect(out.corpus_source).to.equal("loaded")
        expect(out.corpus_path).to.equal(CORPUS_FILE)
        expect(out.corpus_seed).to.equal(20260804)
        expect(out.corpus_games).to.equal(8)
    end)

    it("accepts a ctx that agrees with the corpus", function()
        configure()
        local out = drive(small({ styles = { "corner", "greedy" }, beta = 0.5, depth = 2 }))

        expect(out.beta).to.equal(0.5)
        expect(out.depth).to.equal(2)
        expect(out.styles[2]).to.equal("greedy")
        expect(called("train")).to.equal(true)
    end)

    it("refuses a ctx beta the corpus was not labelled at", function()
        configure()
        local err = drive_error(small({ beta = 0.75 }))

        expect(contains(err, "ctx.beta asks for 0.75")).to.equal(true)
        expect(contains(err, "cannot be changed by asking")).to.equal(true)
        expect(contains(err, "gen_othello6_mix_corpus.lua")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)

    it("refuses a ctx parent pair the corpus was not labelled by", function()
        configure()
        local swapped = drive_error(small({ styles = { "greedy", "corner" } }))
        expect(contains(swapped, "ctx.styles asks for greedy/corner")).to.equal(true)
        expect(contains(swapped, "labelled by corner/greedy")).to.equal(true)

        configure()
        expect(
            contains(drive_error(small({ styles = { "corner" } })), "ctx.styles asks for corner")
        ).to.equal(true)
    end)

    it("refuses a ctx depth the parents did not search", function()
        configure()
        local err = drive_error(small({ depth = 4 }))

        expect(contains(err, "ctx.depth asks for 4")).to.equal(true)
        expect(contains(err, "teachers that never labelled it")).to.equal(true)
        expect(called("train")).to.equal(false)
    end)
end)

describe("train_othello6_mix alias", function()
    it("names the parents and the weight it actually baked", function()
        configure()
        local out = drive(small({}))

        -- Those three things cannot be read back off the weights.
        expect(out.alias).to.equal("othello6_npc_mix_cg_b50")
        expect(#ALIAS_SET_CALLS).to.equal(1)
        expect(ALIAS_SET_CALLS[1].alias).to.equal("othello6_npc_mix_cg_b50")
        expect(ALIAS_SET_CALLS[1].card_id).to.equal("card-stub-0001")
        expect(ALIAS_SET_CALLS[1].opts.pkg).to.equal("othello6_npc")

        configure()
        plant_corpus(mix_meta({ styles = { "mobility", "greedy" }, beta = 0.25 }))
        expect(drive(small({})).alias).to.equal("othello6_npc_mix_mg_b25")
    end)

    it("refuses to round a weight onto another mixture's name", function()
        configure()
        plant_corpus(mix_meta({ beta = 0.505 }))
        local err = drive_error(small({}))

        -- Two betas a fraction of a percent apart would share an alias,
        -- and the second bake would silently re-point the first Card.
        expect(contains(err, "does not land on a whole percent")).to.equal(true)
        expect(contains(err, "pass ctx.alias")).to.equal(true)
    end)

    it("takes an explicit alias instead", function()
        configure()
        plant_corpus(mix_meta({ beta = 0.505 }))
        local out = drive(small({ alias = "othello6_npc_mix_probe" }))

        expect(out.alias).to.equal("othello6_npc_mix_probe")
        expect(ALIAS_SET_CALLS[1].alias).to.equal("othello6_npc_mix_probe")
    end)
end)

describe("train_othello6_mix compliance", function()
    it("scores the Card against each parent in turn", function()
        configure()
        local out = drive(small({ check_games = 5 }))

        -- A mixture has no single teacher, so the pair of readings is
        -- the measurement.
        expect(out.style_match_a).to.equal(0.6)
        expect(out.style_match_b).to.equal(0.4)
        expect(out.style_match_sum).to.equal(1.0)
        expect(contains(out.selfplay_a, "style_hits=6/10")).to.equal(true)
        expect(contains(out.selfplay_b, "style_hits=4/10")).to.equal(true)

        -- Two runs, one per parent, and the parent travels on the task
        -- rather than on the basis so neither inherits a default.
        expect(#NPC_CALLS).to.equal(2)
        expect(contains(NPC_CALLS[1].task, '"style":"corner"')).to.equal(true)
        expect(contains(NPC_CALLS[2].task, '"style":"greedy"')).to.equal(true)
        expect(contains(NPC_CALLS[1].task, '"games":5')).to.equal(true)
        expect(contains(NPC_CALLS[1].task, '"depth":2')).to.equal(true)
        expect(NPC_CALLS[1].card_alias).to.equal("othello6_npc_mix_cg_b50")
    end)

    it("reads the model's own legal rate off the shared run", function()
        configure()
        local out = drive(small({}))

        -- The move that is played is legal by construction, so the raw
        -- reading is the one that says something about the model.
        expect(out.moves).to.equal(10)
        expect(out.illegal).to.equal(2)
        expect(out.legal_rate).to.equal(0.8)
    end)

    it("refuses a pair of runs that did not play the same games", function()
        configure()
        -- Same Card, same seed, same game count: only the teacher scored
        -- against may move. If the move counts differ, the two numbers
        -- are not a split of one behaviour, which is the only reading
        -- they are reported for.
        SELFPLAY_BY_STYLE.greedy = "winrate=0.50 illegal=2 style_match=0.40 style_hits=4/12"
        local err = drive_error(small({}))
        expect(contains(err, "played differently")).to.equal(true)

        configure()
        SELFPLAY_BY_STYLE.greedy = "winrate=0.50 illegal=3 style_match=0.40 style_hits=4/10"
        expect(contains(drive_error(small({})), "played differently")).to.equal(true)
    end)

    it("refuses a self-play answer it cannot read", function()
        configure()
        SELFPLAY_BY_STYLE.corner = "winrate=0.50"
        expect(contains(drive_error(small({})), "not in the expected shape")).to.equal(true)
    end)
end)

describe("train_othello6_mix result", function()
    it("gates on the loss floor alone", function()
        configure()
        local out = drive(small({}))

        -- `illegal == 0` is not required: the measured legal rate at
        -- this scale is 0.78, so a zero would be a criterion no run of
        -- this size meets. The compliance numbers are data, not a gate.
        expect(out.illegal).to.equal(2)
        expect(out.ok).to.equal(true)
        expect(out.loss_descended).to.equal(true)
        expect(out.baseline_loss).to.equal(math.log(64))

        configure()
        TRAIN_LOSS = 9.0
        local failed = drive(small({}))
        expect(failed.ok).to.equal(false)
        expect(failed.loss_descended).to.equal(false)
        -- Still measured and still reported; only `ok` moved.
        expect(failed.style_match_a).to.equal(0.6)
    end)

    it("reports the corpus, the shape and the mixture together", function()
        configure()
        local out = drive(small({}))

        expect(out.card_id).to.equal("card-stub-0001")
        expect(out.name).to.equal("othello6-npc-mix")
        expect(out.rows).to.equal(8)
        expect(out.rows_target).to.equal(3)
        expect(out.ctx_len).to.equal(48)
        expect(out.model_vocab).to.equal(64)
        expect(out.vocab_size).to.equal(othello.VOCAB_SIZE)
        expect(out.random_opening_max).to.equal(othello.RANDOM_OPENING_MAX)
        expect(type(out.corpus_seconds)).to.equal("number")
        expect(out.check_games).to.equal(1)
        expect(#DATASET_ROWS).to.equal(8)
    end)

    it("logs the mixture it loaded and what it measured", function()
        configure()
        drive(small({}))

        local loaded, measured = false, false
        for _, line in ipairs(LOG_LINES) do
            if contains(line, "corner/greedy beta=0.5") then
                loaded = true
            end
            if contains(line, "style_match_a=0.6000 (corner), style_match_b=0.4000 (greedy)") then
                measured = true
            end
        end
        expect(loaded).to.equal(true)
        expect(measured).to.equal(true)
    end)
end)
