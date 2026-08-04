-- Generate one 6x6 Othello corpus labelled by a *mixture* of two
-- teacher styles, and save it to a file.
--
-- Self-contained script for `alc_run` (`code_file` form). It is the
-- style-mixed sibling of `gen_othello6_corpus.lua`: same playouts, same
-- encoding, same file shape, with one thing moved. Where that script
-- labels every decision with a single `othello6_teacher` policy, this
-- one labels each decision with `othello6_mix.mixed_policy`, which draws
-- once per decision and answers with `styles[1]` on a share `beta` of
-- them and with `styles[2]` on the rest.
--
-- Nothing is trained here and no model is built. The output is the input
-- of `train_othello6_mix.lua`, which reads it back through its
-- `corpus_path` field.
--
-- ## What the axis is
--
-- Style, at one fixed depth. Both parents search the same number of
-- plies, so the mixture is between two tastes rather than between two
-- strengths; `othello6_mix` refuses a depth per parent for that reason,
-- and `depth` is a single field here. What a model fitted on the rows
-- learns is the label distribution of a line, which is the mixture
-- itself rather than either parent.
--
-- ## Cost -- READ THIS BEFORE RAISING `games`
--
-- **A mixed decision costs about twice a single-style one.** The draw
-- itself is one integer, but both parents search the position before the
-- draw picks between them, so a decision pays two searches instead of
-- one.
--
-- Measured on 2026-08-04 (local CPU, depth 2 / corner): a single-style
-- playout costs **0.0129 s**. The mixed cost is therefore about
-- **0.026 s** per playout, and the default of 8000 games is roughly
-- **3 min 30 s**:
--
--     games       1000     2000     4000     8000    16000
--     seconds        26       52      104      206      413
--
-- That is past the tens-of-seconds envelope one *iteration* of this
-- experiment has to stay inside (`workspace/EXPERIMENT-RULES.md`), and
-- it is spent here on purpose: a corpus is generated once and read back
-- in a fraction of a second on every bake afterwards -- 20k rows loaded
-- in 0.058 s against 258 s to play them -- so the cost is an investment
-- amortised over a sweep rather than a per-iteration charge. The three
-- beta points this experiment sweeps are three separate calls of about
-- 3 min 30 s each, roughly **10 min 30 s** in total.
--
-- Two limits bound how far that can be pushed. The MCP client aborts a
-- tool call after 1800 s of silence and the abort does not stop the run,
-- so a single generation projected past ~25 min cannot complete in one
-- call and will hold the machine after the caller has stopped waiting
-- for it. And the machine this runs on belongs to someone working on it.
-- Anything larger belongs on a GPU pod
-- (`workspace/tasks/alc-nn-gpu-smoke/runbook.md`).
--
-- ## What is written
--
-- A JSON object of `meta` and `rows`, the same two keys
-- `gen_othello6_corpus.lua` writes -- with a different meta, on purpose.
--
-- A single-style corpus carries `style` (one name) and a mixed one
-- carries `styles` (two) plus `beta`, and the two are told apart by
-- `kind`. The separation is not cosmetic: `train_othello6_npc.lua` holds
-- `meta.style` to the single teacher it is measuring against, and a
-- mixed corpus that reused that key would either fail that check with a
-- confusing message or, worse, pass it and have a mixture reported as
-- one parent's result. `kind = "mix"` is what makes the two files
-- refuse each other's reader by name.
--
--     {
--       "meta": {
--         "game": "othello6", "kind": "mix",
--         "styles": ["corner", "greedy"], "beta": 0.5, "depth": 2,
--         "games": 8000, "ctx_len": 48, "seed": 20260804,
--         "random_opening_max": 6, "vocab_size": 46, "bos": "^",
--         "rows_per_game_estimate": 1
--       },
--       "rows": [[...], [...]]
--     }
--
-- ctx:
--   path        -- file the corpus is written to (required)
--   beta        -- share of the decisions `styles[1]` answers, in the
--                  open interval (0, 1) (required). There is no default:
--                  a mixing weight is the whole point of the file and a
--                  corpus built at an unrequested one is a measurement
--                  of a mixture nobody asked for
--   styles      -- the two parents, in order (default
--                  `{"corner", "greedy"}`); two different
--                  `othello6.STYLES` names
--   depth       -- plies *both* parents search (default 2); one of
--                  `othello6_teacher.DEPTHS`
--   games       -- playouts to play, one row each (default 8000)
--   seed        -- base seed for the playouts and for the mixing draw
--                  (default 20260804)
--   ctx_len     -- width every row is padded to (default 48, the window
--                  the model preset is built at)
--   random_opening_max  -- upper bound of the uniform draw of random
--                  opening plies every game starts with (default
--                  `othello6.RANDOM_OPENING_MAX`)
--
-- Returns a flat table so a smoke harness can assert on it directly,
-- with the written meta along for the record.

local othello = require("othello6")
local teacher = require("othello6_teacher")
local mix = require("othello6_mix")
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
    beta = true,
    ctx_len = true,
    depth = true,
    games = true,
    path = true,
    random_opening_max = true,
    seed = true,
    styles = true,
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
--- `style` singular is called out by name, because it is the field of
--- the single-style generator and a caller reaching for it here is
--- asking for a corpus this script does not produce.
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
        local hint = ""
        if CTX.style ~= nil then
            hint = "; a mixture has two parents, so the field is `styles` (a pair) and the "
                .. "single-parent corpus is generated by gen_othello6_corpus.lua"
        end
        error(
            string.format(
                "gen_othello6_mix_corpus: unknown ctx field(s) %s (known: %s)%s",
                table.concat(unknown, ", "),
                table.concat(sorted_keys(KNOWN_FIELDS), ", "),
                hint
            )
        )
    end
end

check_ctx_fields()

local function require_finite(name, raw)
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw == -math.huge then
        error(
            string.format(
                "gen_othello6_mix_corpus: ctx.%s must be a finite number, got %s",
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
--- `games = "8000"` read as 8000 would write a corpus under a request
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
                "gen_othello6_mix_corpus: ctx.%s must be an integer, got %s",
                name,
                tostring(raw)
            )
        )
    end
    raw = math.floor(raw)
    if min ~= nil and raw < min then
        error(
            string.format(
                "gen_othello6_mix_corpus: ctx.%s must be at least %d, got %d",
                name,
                min,
                raw
            )
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
                    "gen_othello6_mix_corpus: ctx.%s is required; a corpus that is not written "
                        .. "anywhere is a run that costs playouts and leaves nothing behind",
                    name
                )
            )
        end
        return default
    end
    if type(raw) ~= "string" then
        error(
            string.format(
                "gen_othello6_mix_corpus: ctx.%s must be a string, got %s",
                name,
                type(raw)
            )
        )
    end
    if #raw == 0 then
        error(string.format("gen_othello6_mix_corpus: ctx.%s must not be empty", name))
    end
    return raw
end

--- The parents this experiment mixes when ctx names none.
---
--- `corner` and `greedy` are the pair the style axis of the design note
--- opens on: they disagree often enough for a mixing ratio to be
--- readable off a corpus, which a pair that mostly agrees would not
--- give.
local DEFAULT_STYLES = { "corner", "greedy" }

--- Read the parent pair as a fresh two-element list.
---
--- Only the container shape is checked here. Which names are legal, that
--- there are exactly two of them and that they differ is decided by
--- `othello6_mix.mixed_policy`, so there is one account of what a legal
--- mixture is rather than a copy of it that can drift.
---
--- The list is copied rather than aliased: the same table goes into the
--- policy and into the meta, and a caller holding a reference to the ctx
--- field could otherwise move what the file claims it was built from.
---@return table styles
local function styles_field()
    local raw = CTX.styles
    if raw == nil then
        return { DEFAULT_STYLES[1], DEFAULT_STYLES[2] }
    end
    if type(raw) ~= "table" then
        error(
            string.format(
                "gen_othello6_mix_corpus: ctx.styles must be a table naming two parents, got %s",
                type(raw)
            )
        )
    end
    local out = {}
    for index = 1, #raw do
        out[index] = raw[index]
    end
    return out
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
            "gen_othello6_mix_corpus: ctx.%s = %s is not one of %s",
            name,
            tostring(value),
            table.concat(names, ", ")
        )
    )
end

local PATH = string_field("path", nil)

-- The weight has no default. Every other field of this script names a
-- condition that has a reasonable one; the mixing ratio is the thing the
-- file exists to hold, and a corpus silently built at some house value
-- would be measured as a mixture the caller never asked for.
if CTX.beta == nil then
    error(
        "gen_othello6_mix_corpus: ctx.beta is required (the share of the decisions styles[1] "
            .. "answers, in the open interval (0, 1)); it is the condition this corpus exists "
            .. "to fix, so there is no default to fall back on"
    )
end
local BETA = require_finite("beta", CTX.beta)

local STYLES = styles_field()
local DEPTH = require_member("depth", int_field("depth", 2, 1), teacher.DEPTHS)

--- Playouts a mixed generation defaults to.
---
--- Eight thousand rows is 249 steps of batch 32 with a batch in hand,
--- which is the training budget the single-style sweep saturated at
--- (steps 624 over 20k rows, i.e. one pass). It costs about 3 min 30 s
--- at the measured mixed rate; see the header before raising it.
local DEFAULT_GAMES = 8000

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
            "gen_othello6_mix_corpus: ctx.ctx_len = %d is narrower than the %d tokens a line "
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
            "gen_othello6_mix_corpus: the corpus is sized one playout per row but othello6 "
                .. "yields %s rows per game",
            tostring(othello.ROWS_PER_GAME_ESTIMATE)
        )
    )
end

-- ─── Host bridges ───────────────────────────────────────────────────

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "gen_othello6_mix_corpus: alc.json_encode is required to write a corpus (host "
                .. "bridge missing)"
        )
    end
    return alc.json_encode
end

local function log(msg)
    alc.log("info", "[othello6-mix-corpus] " .. msg)
end

-- ─── Generate ───────────────────────────────────────────────────────

local VOCAB = othello.vocab()

-- Built once, before the first playout, and handed to the generator as
-- a single function. Two things ride on that: the pair, the weight, the
-- seed and the depth are validated here rather than after the budget has
-- been spent, and the mixing stream belongs to this one handle, so the
-- sequence of draws -- and therefore the labels -- is reproducible from
-- `(styles, beta, seed, depth)` alone.
local policy = mix.mixed_policy(STYLES, BETA, SEED, DEPTH)

log(
    string.format(
        "generating %d playouts (depth %d, %s/%s beta=%g, seed %d, ctx_len %d) -> %s",
        GAMES,
        DEPTH,
        STYLES[1],
        STYLES[2],
        BETA,
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
            "gen_othello6_mix_corpus: %d playouts of the depth %d %s/%s beta=%g mixture "
                .. "produced %d rows; one game is one row, so the generator answered a "
                .. "different count than it was asked for games",
            GAMES,
            DEPTH,
            STYLES[1],
            STYLES[2],
            BETA,
            #rows
        )
    )
end

--- Conditions the rows were played under, written beside them.
---
--- The same shape `train_othello6_mix.lua` reads back and checks against
--- the run that loads it; a field added here has to be added there too,
--- and the reader is what reports a drift between the two.
---
--- `kind` is what tells this file apart from a single-style corpus. Both
--- carry `game = "othello6"` and rows of the same width, so without it
--- the two readers would have to infer which file they were handed from
--- the presence of a key, and the failure of that inference is a
--- mixture measured as one parent.
local meta = {
    game = "othello6",
    kind = "mix",
    styles = { STYLES[1], STYLES[2] },
    beta = BETA,
    depth = DEPTH,
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
--- overwriting one retires that comparison. A stale file would be caught
--- by the reader's meta check anyway, so the write goes ahead; the line
--- is here so the replacement appears in the log of the run that did it,
--- which for a mixed corpus is minutes of playouts rather than seconds.
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
            "gen_othello6_mix_corpus: failed to encode the corpus for %q: %s",
            PATH,
            tostring(body)
        )
    )
end

local handle, open_err = io.open(PATH, "w")
if handle == nil then
    error(
        string.format(
            "gen_othello6_mix_corpus: cannot open ctx.path %q for writing: %s",
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
            "gen_othello6_mix_corpus: write to ctx.path %q failed: %s",
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
    kind = "mix",
    styles = { STYLES[1], STYLES[2] },
    beta = BETA,
    depth = DEPTH,
    seed = SEED,
    ctx_len = CTX_LEN,
    random_opening_max = RANDOM_OPENING_MAX,
    vocab_size = VOCAB.size,
    -- Wall clock of the generation and the write, and the per-playout
    -- cost it implies. The second number is what a caller sizing the
    -- next corpus should multiply, rather than the figure in the header,
    -- which was measured on one machine on one day -- and it is the one
    -- that says whether the "twice a single-style playout" estimate the
    -- header is built on held.
    elapsed_seconds = elapsed_seconds,
    seconds_per_game = elapsed_seconds / GAMES,
    -- The meta exactly as written, so a harness can assert on the file
    -- contract without reopening the file.
    meta = meta,
}
