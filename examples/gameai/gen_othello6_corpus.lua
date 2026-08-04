-- Generate one 6x6 Othello teacher corpus and save it to a file.
--
-- Self-contained script for `alc_run` (`code_file` form). It plays
-- `games` self-play playouts under one `othello6_teacher` policy -- a
-- search depth and an evaluation style -- encodes them as one row per
-- game, and writes the rows together with the conditions they were
-- played under to `ctx.path`.
--
-- Nothing is trained here and no model is built. The output is the
-- input of `train_othello6_npc.lua`, which reads it back through its
-- `corpus_path` field.
--
-- ## Why this is a separate script
--
-- Corpus generation dominated the wall clock of a bake and was redone on
-- every one of them: measured on 2026-08-04, 400 s of an 11 min 30 s run
-- (58%) was spent playing games that the previous run had already
-- played. Splitting it out makes the corpus a fixed condition that a
-- sweep over `steps`, `layers`, `heads` or `dim` is compared under,
-- instead of a fresh roll of the data underneath each comparison.
--
-- The intended shape of a session is one generation and many bakes:
--
--     alc_run gen_othello6_corpus.lua  { path = "...", games = 2000 }
--     alc_run train_othello6_npc.lua   { corpus_path = "...", steps = 200 }
--     alc_run train_othello6_npc.lua   { corpus_path = "...", steps = 400 }
--
-- ## Cost -- READ THIS BEFORE RAISING `games`
--
-- One iteration of this experiment must finish in tens of seconds, which
-- is a hard constraint on the machine this runs on rather than a
-- preference; the incident that produced that rule is written up in
-- `workspace/EXPERIMENT-RULES.md`.
--
-- Measured on 2026-08-04 (local CPU, depth 2 / corner): a playout costs
-- about **0.0156 s**, read off the 25,632-game corpus that took 400 s.
-- The default of 2000 games is therefore roughly **31 s**, and the
-- arithmetic scales linearly:
--
--     games      1000     2000     5000    20000    25632
--     seconds      16       31       78      312      400
--
-- Anything past a few thousand games leaves the tens-of-seconds envelope
-- and belongs on a GPU pod (`workspace/tasks/alc-nn-gpu-smoke/runbook.md`)
-- rather than on this machine. The MCP client aborts a tool call after
-- 1800 s of silence and the abort does not stop the run, so a generation
-- projected past ~25 min cannot complete in one call and will hold the
-- machine after the caller has stopped waiting for it.
--
-- ## What is written
--
-- A JSON object of `meta` and `rows`. The meta is not decoration: rows
-- alone load under any run, so a file carrying only rows would let a
-- bake train on one teacher's games and report the result as another's.
-- `train_othello6_npc.lua` holds `game`, `ctx_len`, `vocab_size`,
-- `depth` and `style` to what it is asking for and refuses a mismatch,
-- which is what keeps the two scripts in step.
--
--     {
--       "meta": {
--         "game": "othello6", "depth": 2, "style": "corner",
--         "games": 2000, "ctx_len": 48, "seed": 20260804,
--         "random_opening_max": 6, "vocab_size": 46, "bos": "^",
--         "rows_per_game_estimate": 1
--       },
--       "rows": [[...], [...]]
--     }
--
-- ctx:
--   path        -- file the corpus is written to (required)
--   depth       -- teacher search depth (default 2); one of
--                  `othello6_teacher.DEPTHS`
--   style       -- teacher evaluation style (default "corner"); one of
--                  `othello6.STYLES`
--   games       -- playouts to play, one row each (default 2000, which
--                  is the tens-of-seconds budget at the measured cost
--                  above)
--   seed        -- base seed for the playouts (default 20260804)
--   ctx_len     -- width every row is padded to (default 48, the window
--                  the model preset is built at); has to hold the
--                  longest game the encoding allows
--   random_opening_max  -- upper bound of the uniform draw of random
--                  opening plies every game starts with (default
--                  `othello6.RANDOM_OPENING_MAX`)
--
-- Returns a flat table so a smoke harness can assert on it directly,
-- with the written meta along for the record.

local othello = require("othello6")
local teacher = require("othello6_teacher")
local fs = require("gameai_metrics._fs")

-- ─── ctx ────────────────────────────────────────────────────────────

--- The ctx object `alc_run` injects as a global.
---
--- A run that passes no ctx leaves the global a non-table userdata that
--- cannot be indexed, so the table is resolved once here and every read
--- afterwards goes through this copy.
local CTX = type(ctx) == "table" and ctx or {}

--- Fields this script reads.
local KNOWN_FIELDS = {
    ctx_len = true,
    depth = true,
    games = true,
    path = true,
    random_opening_max = true,
    seed = true,
    style = true,
}

local function sorted_keys(t)
    local names = {}
    for key in pairs(t) do
        names[#names + 1] = tostring(key)
    end
    table.sort(names)
    return names
end

--- Reject a ctx field this script does not read.
---
--- Runs before a single game is played: a run configured by a field
--- nobody reads writes a corpus built at the defaults and says nothing
--- about the request not being honoured, which is exactly the kind of
--- file that is later measured under conditions it was never built for.
local function check_ctx_fields()
    local unknown = {}
    for key in pairs(CTX) do
        local name = tostring(key)
        if not KNOWN_FIELDS[name] then
            unknown[#unknown + 1] = name
        end
    end
    table.sort(unknown)
    if #unknown > 0 then
        error(
            string.format(
                "gen_othello6_corpus: unknown ctx field(s) %s (known: %s)",
                table.concat(unknown, ", "),
                table.concat(sorted_keys(KNOWN_FIELDS), ", ")
            )
        )
    end
end

check_ctx_fields()

local function require_finite(name, raw)
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw == -math.huge then
        error(
            string.format(
                "gen_othello6_corpus: ctx.%s must be a finite number, got %s",
                name,
                tostring(raw)
            )
        )
    end
    return raw
end

--- Read an optional integer ctx field.
---
--- A present field of the wrong type fails rather than being coerced:
--- `games = "2000"` read as 2000 would write a corpus under a request
--- the caller has no way to tell apart from a typo that landed
--- somewhere else.
---@param name string ctx key
---@param default integer Value used when the field is absent
---@param min integer|nil Lowest value the field accepts
---@return integer
local function int_field(name, default, min)
    local raw = CTX[name]
    if raw == nil then
        return default
    end
    require_finite(name, raw)
    if raw ~= math.floor(raw) then
        error(
            string.format(
                "gen_othello6_corpus: ctx.%s must be an integer, got %s",
                name,
                tostring(raw)
            )
        )
    end
    raw = math.floor(raw)
    if min ~= nil and raw < min then
        error(
            string.format("gen_othello6_corpus: ctx.%s must be at least %d, got %d", name, min, raw)
        )
    end
    return raw
end

---@param name string ctx key
---@param default string|nil Value used when the field is absent, or
---       `nil` to make the field required
---@return string
local function string_field(name, default)
    local raw = CTX[name]
    if raw == nil then
        if default == nil then
            error(
                string.format(
                    "gen_othello6_corpus: ctx.%s is required; a corpus that is not written "
                        .. "anywhere is a run that costs playouts and leaves nothing behind",
                    name
                )
            )
        end
        return default
    end
    if type(raw) ~= "string" then
        error(
            string.format("gen_othello6_corpus: ctx.%s must be a string, got %s", name, type(raw))
        )
    end
    if #raw == 0 then
        error(string.format("gen_othello6_corpus: ctx.%s must not be empty", name))
    end
    return raw
end

--- Hold a field to the canonical list of values it names.
local function require_member(name, value, list)
    for _, entry in ipairs(list) do
        if entry == value then
            return value
        end
    end
    local names = {}
    for index, entry in ipairs(list) do
        names[index] = tostring(entry)
    end
    error(
        string.format(
            "gen_othello6_corpus: ctx.%s = %s is not one of %s",
            name,
            tostring(value),
            table.concat(names, ", ")
        )
    )
end

local PATH = string_field("path", nil)
local DEPTH = require_member("depth", int_field("depth", 2, 1), teacher.DEPTHS)
local STYLE = require_member("style", string_field("style", "corner"), othello.STYLES)

--- Playouts a generation costs tens of seconds at.
---
--- At the measured 0.0156 s per playout this is about 31 s, which is the
--- envelope one iteration of this experiment has to stay inside. The
--- design note opens at 20000 games; that is 5 minutes here and belongs
--- on a pod, so the default is the local one.
local DEFAULT_GAMES = 2000

local GAMES = int_field("games", DEFAULT_GAMES, 1)
local SEED = int_field("seed", 20260804)
local CTX_LEN = int_field("ctx_len", othello.CTX_BUDGET, 1)
local RANDOM_OPENING_MAX = int_field("random_opening_max", othello.RANDOM_OPENING_MAX, 0)

-- A row has to hold the longest game the encoding allows, marker
-- included. Checked before any budget is spent, because it is a
-- property of the two numbers alone and a narrow window would otherwise
-- surface as a failed playout somewhere in the middle of the run.
if CTX_LEN < othello.ROW_LEN then
    error(
        string.format(
            "gen_othello6_corpus: ctx.ctx_len = %d is narrower than the %d tokens a line "
                .. "needs, so a long game would not fit the row it is written to",
            CTX_LEN,
            othello.ROW_LEN
        )
    )
end

-- The sizing below is arithmetic on "one playout is one row". The game
-- module publishes that number, so it is read rather than assumed, and
-- an encoding that moved it says so here instead of writing a corpus
-- whose meta lies about its own shape.
if othello.ROWS_PER_GAME_ESTIMATE ~= 1 then
    error(
        string.format(
            "gen_othello6_corpus: the corpus is sized one playout per row but othello6 yields "
                .. "%s rows per game",
            tostring(othello.ROWS_PER_GAME_ESTIMATE)
        )
    )
end

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "gen_othello6_corpus: alc.json_encode is required to write a corpus (host bridge "
                .. "missing)"
        )
    end
    return alc.json_encode
end

local function log(msg)
    alc.log("info", "[othello6-corpus] " .. msg)
end

-- ─── Generate ───────────────────────────────────────────────────────

local VOCAB = othello.vocab()
local policy = teacher.policy(DEPTH, STYLE)

log(
    string.format(
        "generating %d playouts (depth %d, %s, seed %d, ctx_len %d) -> %s",
        GAMES,
        DEPTH,
        STYLE,
        SEED,
        CTX_LEN,
        PATH
    )
)

local started = os.clock()
local rows = othello.build_corpus(policy, {
    ctx_len = CTX_LEN,
    games = GAMES,
    seed = SEED,
    pad_id = VOCAB.pad_id,
    random_opening_max = RANDOM_OPENING_MAX,
})

-- One game is one row, so a shortfall is the generator breaking that
-- contract rather than a yield to measure. It is said here instead of
-- being written to the file, because the meta records the games asked
-- for and a corpus whose meta and rows disagree is measured under a
-- size it never had.
if #rows ~= GAMES then
    error(
        string.format(
            "gen_othello6_corpus: %d playouts of the depth %d %s teacher produced %d rows; "
                .. "one game is one row, so the generator answered a different count than it "
                .. "was asked for games",
            GAMES,
            DEPTH,
            STYLE,
            #rows
        )
    )
end

--- Conditions the rows were played under, written beside them.
---
--- The same shape `train_othello6_npc.lua` reads back and checks against
--- the run that loads it; a field added here has to be added there too,
--- and the reader is what reports a drift between the two.
local meta = {
    game = "othello6",
    depth = DEPTH,
    style = STYLE,
    games = GAMES,
    ctx_len = CTX_LEN,
    seed = SEED,
    random_opening_max = RANDOM_OPENING_MAX,
    vocab_size = VOCAB.size,
    bos = othello.BOS,
    rows_per_game_estimate = othello.ROWS_PER_GAME_ESTIMATE,
}

-- ─── Save ───────────────────────────────────────────────────────────

--- Replacing a file is allowed and said out loud.
---
--- A corpus is the fixed condition a set of bakes was compared under, so
--- overwriting one retires that comparison. It is cheap to rebuild --
--- tens of seconds -- and a stale file would be caught by the reader's
--- meta check anyway, so the write goes ahead; the line is here so the
--- replacement appears in the log of the run that did it.
local existing = io.open(PATH, "r")
if existing ~= nil then
    existing:close()
    log(string.format("%s already exists and is being replaced", PATH))
end

local encode = require_json_encoder()
fs.ensure_parent_dir(PATH)

local encoded_ok, body = pcall(encode, { meta = meta, rows = rows })
if not encoded_ok then
    error(
        string.format(
            "gen_othello6_corpus: failed to encode the corpus for %q: %s",
            PATH,
            tostring(body)
        )
    )
end

local handle, open_err = io.open(PATH, "w")
if handle == nil then
    error(
        string.format(
            "gen_othello6_corpus: cannot open ctx.path %q for writing: %s",
            PATH,
            tostring(open_err)
        )
    )
end
local wrote, write_err = pcall(function()
    handle:write(body)
end)
handle:close()
if not wrote then
    error(
        string.format(
            "gen_othello6_corpus: write to ctx.path %q failed: %s",
            PATH,
            tostring(write_err)
        )
    )
end

local elapsed_seconds = os.clock() - started
log(
    string.format(
        "wrote %d rows x %d tokens to %s in %.2fs (%.4fs per playout)",
        #rows,
        CTX_LEN,
        PATH,
        elapsed_seconds,
        elapsed_seconds / GAMES
    )
)

return {
    ok = true,
    path = PATH,
    rows = #rows,
    games = GAMES,
    depth = DEPTH,
    style = STYLE,
    seed = SEED,
    ctx_len = CTX_LEN,
    random_opening_max = RANDOM_OPENING_MAX,
    vocab_size = VOCAB.size,
    -- Wall clock of the generation and the write, and the per-playout
    -- cost it implies. The second number is what a caller sizing the
    -- next corpus should multiply, rather than the figure in the header,
    -- which was measured on one machine on one day.
    elapsed_seconds = elapsed_seconds,
    seconds_per_game = elapsed_seconds / GAMES,
    -- The meta exactly as written, so a harness can assert on the file
    -- contract without reopening the file.
    meta = meta,
}
