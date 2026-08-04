-- Train the 6x6 Othello NPC: teacher corpus -> Full FT -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). It labels a
-- corpus with one `othello6_teacher` policy -- a search depth and an
-- evaluation style -- tunes a from-scratch custom GPT-2 on it,
-- registers the resulting Card, pins an alias, and reads the model back
-- through `othello6_npc`.
--
-- No `alc.llm` call happens anywhere in this path, so the run never
-- pauses for a host response.
--
-- ## What is measured
--
-- Two numbers, both read under greedy decoding (`decide`), which is the
-- condition every measurement in this experiment is fixed to:
--
--   * `style_match` -- the share of model moves that equal the move the
--     labelling teacher would have played in the same position, over
--     positions the model reaches by playing. It is the direct reading
--     of "did this Card reproduce this teacher".
--   * `legal_rate` -- the share of decisions whose *raw* argmax was
--     already legal. The move that ends up played is legal by
--     construction, because `othello6_npc` walks the logit ranking
--     against the legal set, so the raw reading is the one that says
--     something about the model rather than about the gate.
--
-- `train_loss` against `ln(vocab)` rides along as the floor test: a
-- final loss under the uniform-random baseline is the evidence that
-- gradients flowed at all.
--
-- ## Cost and scale -- READ THIS BEFORE RAISING `steps` OR `games`
--
-- One iteration of this experiment must finish in tens of seconds. That
-- is a hard constraint on the machine this runs on, not a preference.
--
-- This rule exists because it was broken. On 2026-08-04 this script was
-- run five times back to back on a laptop CPU, holding the machine for
-- roughly 64 minutes (2:20 + 14:47 + 5:26 + 11:30 + 30:28) and driving
-- its 1-minute load average to 17.32. The owner of that machine had to
-- say "kill it" three times before the runs stopped, and the last half
-- hour produced nothing at all -- no Card, no return value, because the
-- MCP call had already been aborted while the process kept burning CPU.
-- The machine this runs on belongs to someone who is trying to work on
-- it. It is not compute budget.
--
-- Measured on 2026-08-04 (local CPU, ctx 48 / layers 4 / heads 4 / dim 128):
--
--   * steps 800, batch 32, 25,632 games -> 11 min 30 s total, of which
--     400 s (58%) was corpus generation alone
--   * steps 2400, batch 32, 76,832 games -> aborted at the 1800 s MCP
--     idle timeout, and the process kept burning 143% CPU afterwards
--
-- Consequences that are fixed, not negotiable:
--
--   1. The MCP client aborts a tool call after 1800 s of silence, and the
--      abort does not stop the run. Anything projected past ~25 min
--      cannot complete in one call and will hold the machine hostage.
--   2. Work that does not fit in tens of seconds does not run here. Send
--      it to a GPU pod -- the runbook with measured timings lives at
--      `workspace/tasks/alc-nn-gpu-smoke/runbook.md` -- or shrink it.
--   3. Corpus generation dominates the wall clock, so it is not redone
--      on every bake: pass `corpus_path` and the rows are written once
--      and read back on every later run. See below.
--
-- The full incident that produced these rules: `workspace/EXPERIMENT-RULES.md`.
--
-- ## Reusing a corpus -- how a bake is kept to tens of seconds
--
-- `corpus_path` names a JSON file holding the rows and the meta they
-- were built under. What the field does depends on whether that file is
-- already there:
--
--   * present   -- the rows are read, the meta is checked against this
--                  run, and no playout is generated (`corpus_source`
--                  comes back as `"loaded"`)
--   * absent    -- the rows are generated, saved to that path, and then
--                  trained on (`"generated_and_saved"`)
--   * no field  -- the rows are generated and thrown away afterwards,
--                  which is what every run did before (`"generated"`)
--
-- Generating once with `gen_othello6_corpus.lua` and pointing every bake
-- at the file it wrote is the shape a sweep over `steps`, `layers`,
-- `heads` or `dim` is meant to take: the corpus is the fixed condition
-- those runs are compared under, and re-rolling it per bake both costs
-- the majority of the wall clock and moves the data underneath the
-- comparison.
--
-- The meta check is deliberately unforgiving, because a reused corpus
-- fails silently by nature: a file built by another teacher, at another
-- width or over another alphabet still loads and still trains, and the
-- numbers that come out are then reported under a condition that never
-- produced them. That is the exact failure the previous iteration of
-- this experiment died of -- it measured under one condition and shipped
-- another -- so `game`, `ctx_len`, `vocab_size`, `depth` and `style` all
-- have to match this run, and a row of the wrong width is refused too.
--
-- The meta shape written here is the one `gen_othello6_corpus.lua`
-- writes. The two are kept in step by this reader: a generator that
-- drifted is answered by a loud mismatch on the next load.
--
-- ## What is deliberately absent
--
-- The guardian driver this script is shaped after carries a staged
-- harvest (`enable_stages` / `stage_bands`), a win-rate gate
-- (`enable_gate` / `target_win_rate_lo` / `gate_games`) and the tier
-- vocabulary the two write into a collection manifest. None of it is
-- here, and those ctx fields are refused by name rather than ignored.
--
-- Strength in this experiment is the teacher's search depth: it is a
-- dial of the run, not a quantity to be inferred, and a win rate
-- against one opponent says as much about that opponent as about the
-- model. A win rate is still reported so a reader can look at it, and
-- nothing in this script branches on it.
--
-- Mid-run checkpoints are still written when `ckpt_every` is set,
-- because a later experiment may want to look at half-trained models.
-- Nothing here reads them and nothing selects among them; a raw
-- checkpoint carries weights and no shape, so the architecture and the
-- shape keys one has to be rebuilt under are reported as `ckpt_arch`
-- and `ckpt_spec` for whoever loads one later.
--
-- ctx (all optional, JSON object passed to alc_run):
--   depth       -- teacher search depth (default 2); one of
--                  `othello6_teacher.DEPTHS`
--   style       -- teacher evaluation style (default "corner"); one of
--                  `othello6.STYLES`
--   games       -- floor on the playouts the corpus is built from
--                  (default 20000, the size the design note opens the
--                  experiment at). One playout is one row, so a run
--                  whose training budget needs more rows than that
--                  plays exactly as many games as it needs rows.
--                  Refused when `corpus_path` names a file that already
--                  exists: the size of a built corpus is a property of
--                  the file, not a request this run can honour
--   corpus_path -- JSON file the corpus is read from, or written to when
--                  it is not there yet (default none, which generates a
--                  corpus and discards it). See the section above
--   steps       -- Full FT steps (default 800)
--   lr          -- learning rate (default 3e-3)
--   batch       -- batch size (default 32)
--   layers      -- transformer blocks (default 4)
--   heads       -- attention heads per block (default 4); `dim` has to
--                  be a multiple of it
--   dim         -- model width (default 128); has to be a multiple of
--                  `heads`
--   seed        -- base seed for the playouts (default 20260804)
--   alias       -- Card alias to pin (default
--                  "othello6_npc_d<depth>_<style>")
--   name        -- Card name (default "othello6-npc")
--   check_games -- self-play games behind the two measurements
--                  (default 20)
--   ckpt_every  -- steps between mid-run checkpoints (default 0, which
--                  writes none)
--   ckpt_keep   -- rotating checkpoints kept on disk (default 6)
--   random_opening_max  -- upper bound of the uniform draw of random
--                  opening plies every corpus game starts with
--                  (default `othello6.RANDOM_OPENING_MAX`)
--
-- The context window and the vocabulary are not among them: both are
-- fixed by the encoding the corpus is written in, so `ctx` and `vocab`
-- are refused by name the way the win-rate fields are.
--
-- Returns a flat table of scalars so a smoke harness can assert on it
-- directly.

local othello = require("othello6")
local teacher = require("othello6_teacher")
local npc = require("othello6_npc")
local fs = require("gameai_metrics._fs")

-- ─── ctx ────────────────────────────────────────────────────────────

--- The ctx object `alc_run` injects as a global.
---
--- A run that passes no ctx leaves the global a non-table userdata that
--- cannot be indexed, so the table is resolved once here and every read
--- afterwards goes through this copy.
local CTX = type(ctx) == "table" and ctx or {}

--- Fields this driver reads.
local KNOWN_FIELDS = {
    alias = true,
    batch = true,
    check_games = true,
    ckpt_every = true,
    ckpt_keep = true,
    corpus_path = true,
    depth = true,
    dim = true,
    games = true,
    heads = true,
    layers = true,
    lr = true,
    name = true,
    random_opening_max = true,
    seed = true,
    steps = true,
    style = true,
}

--- Fields of the guardian driver this experiment refuses, and the path
--- each of them belongs to.
---
--- Every one of them exists to turn a win rate into a level: a band
--- schedule, a floor a checkpoint has to clear, the games those two are
--- measured over, and the manifest the harvested tiers are written to.
--- The design note forbids that reading outright, so the fields fail
--- loudly instead of being dropped: a silently ignored `enable_gate`
--- would let a caller believe a run was gated when nothing read the
--- field, which is exactly how a forbidden criterion comes back.
local REFUSED_FIELDS = {
    collection_path = "staged harvest",
    enable_gate = "win-rate gate",
    enable_stages = "staged harvest",
    gate_games = "win-rate gate",
    pin_bare_alias = "staged harvest",
    stage_alias_prefix = "staged harvest",
    stage_bands = "staged harvest",
    target_win_rate_lo = "win-rate gate",
    teacher_alias = "tier comparison",
    tier = "tier selection",
}

--- Shape fields the encoding fixes, and what fixes each of them.
---
--- `ctx` is the width every corpus row is padded to and `vocab` is the
--- size of the alphabet those rows are drawn from, so neither is a dial
--- of the experiment: a run that moved one would build a model the rows
--- it then trains on no longer fit. They are refused rather than
--- clamped, because a request that is silently overruled reports the
--- numbers of a shape the caller did not ask for. The depth and the
--- width -- `layers`, `heads`, `dim` -- are the dials.
local FIXED_FIELDS = {
    ctx = "the corpus row length",
    vocab = "the alphabet size",
}

local function sorted_keys(t)
    local names = {}
    for key in pairs(t) do
        names[#names + 1] = tostring(key)
    end
    table.sort(names)
    return names
end

--- Reject a ctx field this driver does not read.
---
--- Runs before anything else so a misspelled request never reaches the
--- corpus: a run configured by a field nobody reads reports numbers for
--- the defaults and says nothing about the request not being honoured.
local function check_ctx_fields()
    local refused, fixed, unknown = {}, {}, {}
    for key in pairs(CTX) do
        local name = tostring(key)
        if REFUSED_FIELDS[name] ~= nil then
            refused[#refused + 1] = string.format("%s (%s)", name, REFUSED_FIELDS[name])
        elseif FIXED_FIELDS[name] ~= nil then
            fixed[#fixed + 1] = string.format("%s (%s)", name, FIXED_FIELDS[name])
        elseif not KNOWN_FIELDS[name] then
            unknown[#unknown + 1] = name
        end
    end
    table.sort(refused)
    table.sort(fixed)
    table.sort(unknown)
    if #refused > 0 then
        error(
            string.format(
                "train_othello6_npc: ctx field(s) %s belong to the guardian driver's "
                    .. "win-rate paths, which this experiment does not carry: strength here is "
                    .. "the teacher's search depth, so no checkpoint is selected by a win rate",
                table.concat(refused, ", ")
            )
        )
    end
    if #fixed > 0 then
        error(
            string.format(
                "train_othello6_npc: ctx field(s) %s are fixed by the corpus encoding rather "
                    .. "than dials of the run: a model built at another one would not fit the "
                    .. "rows it is then trained on. The shape dials are layers, heads and dim",
                table.concat(fixed, ", ")
            )
        )
    end
    if #unknown > 0 then
        error(
            string.format(
                "train_othello6_npc: unknown ctx field(s) %s (known: %s)",
                table.concat(unknown, ", "),
                table.concat(sorted_keys(KNOWN_FIELDS), ", ")
            )
        )
    end
end

check_ctx_fields()

--- Reject a non-number and the two numbers that are not quantities.
local function require_finite(name, raw)
    if type(raw) ~= "number" or raw ~= raw or raw == math.huge or raw == -math.huge then
        error(
            string.format(
                "train_othello6_npc: ctx.%s must be a finite number, got %s",
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
--- `steps = "800"` read as 800 would train under a request the caller
--- has no way to tell apart from a typo that landed somewhere else.
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
                "train_othello6_npc: ctx.%s must be an integer, got %s",
                name,
                tostring(raw)
            )
        )
    end
    raw = math.floor(raw)
    if min ~= nil and raw < min then
        error(
            string.format("train_othello6_npc: ctx.%s must be at least %d, got %d", name, min, raw)
        )
    end
    return raw
end

---@param name string ctx key
---@param default number Value used when the field is absent
---@return number
local function positive_number_field(name, default)
    local raw = CTX[name]
    if raw == nil then
        return default
    end
    require_finite(name, raw)
    if raw <= 0 then
        error(
            string.format(
                "train_othello6_npc: ctx.%s must be positive, got %s",
                name,
                tostring(raw)
            )
        )
    end
    return raw
end

---@param name string ctx key
---@param default string Value used when the field is absent
---@return string
local function string_field(name, default)
    local raw = CTX[name]
    if raw == nil then
        return default
    end
    if type(raw) ~= "string" then
        error(string.format("train_othello6_npc: ctx.%s must be a string, got %s", name, type(raw)))
    end
    if #raw == 0 then
        error(string.format("train_othello6_npc: ctx.%s must not be empty", name))
    end
    return raw
end

--- Hold a field to the canonical list of values it names.
---
--- The list is the one the game and the teacher publish, so a typo is
--- answered with the valid spellings instead of resolving to nothing
--- several layers down.
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
            "train_othello6_npc: ctx.%s = %s is not one of %s",
            name,
            tostring(value),
            table.concat(names, ", ")
        )
    )
end

local DEPTH = require_member("depth", int_field("depth", 2, 1), teacher.DEPTHS)
local STYLE = require_member("style", string_field("style", "corner"), othello.STYLES)
--- Playouts the design note opens the experiment at.
---
--- The published readings are per corpus size -- 1k games read legal
--- play at 20.8%, 20k at 79.7% -- and 20k is where this experiment
--- starts, to be raised once the legal rate at that size is known. It
--- is a floor rather than the size: one game is one row, so a training
--- budget that consumes more rows than this decides the count instead.
local DEFAULT_GAMES = 20000

local GAMES_FLOOR = int_field("games", DEFAULT_GAMES, 0)
local STEPS = int_field("steps", 800, 1)
local BATCH = int_field("batch", 32, 1)
local SEED = int_field("seed", 20260804)
local LR = positive_number_field("lr", 3e-3)
local NAME = string_field("name", "othello6-npc")
local CHECK_GAMES = int_field("check_games", 20, 1)
local CKPT_EVERY = int_field("ckpt_every", 0, 0)
local CKPT_KEEP = int_field("ckpt_keep", 6, 1)
local RANDOM_OPENING_MAX = int_field("random_opening_max", othello.RANDOM_OPENING_MAX, 0)
local ALIAS = string_field("alias", string.format("othello6_npc_d%d_%s", DEPTH, STYLE))
--- File the corpus is read from or written to, or `nil` for neither.
---
--- `string_field` still does the type check, so a non-string or an empty
--- path is refused here rather than surfacing as an `io.open` error
--- after the model has been built.
local CORPUS_PATH = nil
if CTX.corpus_path ~= nil then
    CORPUS_PATH = string_field("corpus_path", "")
end

--- Whether the caller named a corpus size, as opposed to inheriting the
--- default. A built corpus holds the rows it holds, so the request has
--- to be told apart from the default to be refused against one.
local GAMES_REQUESTED = CTX.games ~= nil

--- Alias `othello6_npc` falls back to, and the pair it scores that
--- fallback against.
---
--- Only a run of that pair may claim the bare alias: the package reads
--- it with `depth = 2, style = "corner"` as its default basis, so a
--- Card baked under another teacher pinned there would be measured
--- against a teacher that never labelled it.
local BARE_ALIAS = "othello6_npc"
local BARE_DEPTH = 2
local BARE_STYLE = "corner"

-- ─── Model ──────────────────────────────────────────────────────────

--- Architecture a mid-run checkpoint has to be rebuilt under.
---
--- `load_ckpt` reads the shape off the spec it is handed rather than
--- off the file, and a customized GPT-2 is not a named variant, so both
--- the arch string and the shape keys below travel together.
local CKPT_ARCH = "gpt2-custom"

--- Shape of the model this experiment trains.
---
--- `ctx` leaves room over the longest move sequence a game produces --
--- 40 moves, passes included, over 300k playouts -- and `vocab` covers
--- the 46-character alphabet; both are properties of the encoding, so
--- neither is readable from ctx.
---
--- The width is where the published 6x6 result sits: three layers at
--- 128 dimensions read legal play at 99.3% top-1, and the 8x8 ablations
--- move with the width rather than the depth. A first attempt that went
--- the other way -- six layers at 96 -- read 9.1%, so the default is
--- four layers at 128 and the depth and the width stay ctx fields: they
--- are dials of the experiment, not of the difficulty. Raising them is
--- how "can a model this size reproduce the teacher at all" gets
--- answered, never how a model is made to win more.
local MODEL = {
    ctx = 48,
    vocab = 64,
    layers = int_field("layers", 4, 1),
    heads = int_field("heads", 4, 1),
    dim = int_field("dim", 128, 1),
}

-- Multi-head attention splits the width across the heads, so a width
-- that does not divide is not a slow model but an impossible one. It is
-- checked here, before the teacher is built and before a single game is
-- played, because it is a property of the two numbers alone.
if MODEL.dim % MODEL.heads ~= 0 then
    error(
        string.format(
            "train_othello6_npc: ctx.dim = %d is not a multiple of ctx.heads = %d, so the "
                .. "width cannot be split across the heads",
            MODEL.dim,
            MODEL.heads
        )
    )
end

--- The shape keys, as `alc.nn.preset.gpt2("custom", ...)` and
--- `alc.nn.card.load_ckpt` both take them.
---
--- One function rather than two literals: the preset that builds the
--- weights and the spec that rebuilds them from a checkpoint have to
--- name the same shape, and a second copy is a second thing to forget.
local function model_spec()
    return {
        ctx = MODEL.ctx,
        vocab = MODEL.vocab,
        layers = MODEL.layers,
        heads = MODEL.heads,
        dim = MODEL.dim,
    }
end

local VOCAB = othello.vocab()

-- Fit checks, before any budget is spent. Both are properties of the
-- alphabet and the encoding alone, so they are answerable without a
-- model, and answering them here means a shape that cannot carry the
-- corpus fails in the first millisecond of the run rather than after
-- the playouts have been generated.
if VOCAB.size > MODEL.vocab then
    error(
        string.format(
            "train_othello6_npc: the alphabet holds %d characters but the model vocabulary "
                .. "is %d",
            VOCAB.size,
            MODEL.vocab
        )
    )
end
if othello.ROW_LEN > MODEL.ctx then
    error(
        string.format(
            "train_othello6_npc: a training line needs %d tokens but the model context is %d",
            othello.ROW_LEN,
            MODEL.ctx
        )
    )
end

-- The teacher is built before the model too. It is the cheapest place
-- the depth/style pair is checked against the module that actually
-- searches with it, and building it costs nothing: the policy holds no
-- state.
local policy = teacher.policy(DEPTH, STYLE)

local function log(msg)
    alc.log("info", "[othello6-train] " .. msg)
end

-- ─── Corpus ─────────────────────────────────────────────────────────
--
-- One playout is one row: the whole move sequence of one game, padded
-- to the model context. The count is therefore arithmetic rather than a
-- measurement -- a run that consumes N rows plays N games -- which is
-- what the board encoding could not offer, where the rows a game
-- yielded depended on how long it ran and had to be counted after the
-- fact.
--
-- The corpus is built in one call for that reason. What the generator
-- answered is still checked against what was asked for, because
-- `alc.nn.data.synthetic` walks its rows once: a corpus short of the
-- training budget is a trainer that runs out of data mid-run.

--- Extra rows beyond `steps * batch`, so a corpus that lands exactly on
--- the boundary still has a batch in hand.
local CORPUS_SLACK_BATCHES = 1

-- The arithmetic below is the encoding's, not this script's: it holds
-- exactly as long as one playout is one row. The game module publishes
-- that number, so it is read rather than assumed, and an encoding that
-- moved it says so here instead of undersizing every corpus built after
-- the change.
if othello.ROWS_PER_GAME_ESTIMATE ~= 1 then
    error(
        string.format(
            "train_othello6_npc: the corpus is sized one playout per row but othello6 yields "
                .. "%s rows per game",
            tostring(othello.ROWS_PER_GAME_ESTIMATE)
        )
    )
end

--- Build a corpus of at least `target` rows, and report what it cost.
---
--- The playout count is the row count, so `ctx.games` and the training
--- budget are read as two floors on the same number: a caller asking
--- for 20000 playouts gets at least 20000, and a run consuming more
--- rows than that plays as many games as it consumes rows.
---@param ctx_len integer Model context window
---@param target integer Rows the training run consumes
---@return table rows, integer played
local function build_full_corpus(ctx_len, target)
    local played = math.max(GAMES_FLOOR, target)
    local rows = othello.build_corpus(policy, {
        ctx_len = ctx_len,
        games = played,
        seed = SEED,
        pad_id = VOCAB.pad_id,
        random_opening_max = RANDOM_OPENING_MAX,
    })
    -- One game is one row, so a shortfall is the generator breaking that
    -- contract rather than a yield to measure and top up. It is said
    -- here instead of being padded over, because a corpus that quietly
    -- came up short is a training run that stops mid-way through.
    if #rows < target then
        error(
            string.format(
                "train_othello6_npc: %d playouts of the depth %d %s teacher produced %d rows "
                    .. "but the run needs %d; one game is one row, so the generator answered "
                    .. "fewer rows than it was asked for games",
                played,
                DEPTH,
                STYLE,
                #rows,
                target
            )
        )
    end
    return rows, played
end

-- ─── Corpus file ────────────────────────────────────────────────────
--
-- A saved corpus is rows plus the conditions they were played under.
-- The conditions are the whole point: rows alone load under any run, so
-- a file carrying only rows would let a bake train on another teacher's
-- games and report the result as this teacher's. The reader below
-- refuses that rather than reading around it.
--
-- The same shape is written by `gen_othello6_corpus.lua`, which is the
-- script a corpus is normally generated with.

--- `meta.game` every corpus this driver reads has to carry.
local CORPUS_GAME = "othello6"

--- Why each meta field has to match, said in the error that reports it.
---
--- A mismatch is never a small one: the first three make the rows a
--- different kind of data than the model was built for, and the last
--- two make the measurement a reading of a teacher that never labelled
--- the Card. Naming the consequence is what keeps a caller from
--- "fixing" the report by relabelling the file.
local META_MISMATCH_REASONS = {
    game = "the rows encode another game entirely",
    ctx_len = "rows of another width do not fit the model this run builds",
    vocab_size = "the same token id stands for another move under another alphabet",
    depth = "a Card is then measured against a teacher that never labelled it",
    style = "a Card is then measured against a teacher that never labelled it",
}

local function require_json_encoder()
    if type(alc) ~= "table" or type(alc.json_encode) ~= "function" then
        error(
            "train_othello6_npc: alc.json_encode is required to save a corpus (host bridge "
                .. "missing)"
        )
    end
    return alc.json_encode
end

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "train_othello6_npc: alc.json_decode is required to read a corpus (host bridge "
                .. "missing)"
        )
    end
    return alc.json_decode
end

--- Conditions a corpus generated by this run was played under.
---@param ctx_len integer Row width
---@param played integer Playouts behind the rows
---@return table meta
local function corpus_meta(ctx_len, played)
    return {
        game = CORPUS_GAME,
        depth = DEPTH,
        style = STYLE,
        games = played,
        ctx_len = ctx_len,
        seed = SEED,
        random_opening_max = RANDOM_OPENING_MAX,
        vocab_size = VOCAB.size,
        bos = othello.BOS,
        rows_per_game_estimate = othello.ROWS_PER_GAME_ESTIMATE,
    }
end

--- Read `path`, or answer `nil` when it is not there.
---
--- Absence is the branch that generates and saves, so it is the one
--- thing a failed open is allowed to mean. Everything else -- an
--- unreadable file, a directory, a permission error -- would be
--- indistinguishable from "not generated yet" if this returned `nil` on
--- any failure, so the read is confirmed rather than assumed.
---@param path string
---@return string|nil body
local function read_corpus_body(path)
    local handle = io.open(path, "r")
    if handle == nil then
        return nil
    end
    local body, read_err = handle:read("a")
    handle:close()
    if type(body) ~= "string" then
        error(
            string.format(
                "train_othello6_npc: corpus_path %q opened but could not be read: %s",
                path,
                tostring(read_err)
            )
        )
    end
    return body
end

--- Write rows and the conditions behind them to `path`.
---@param path string
---@param meta table
---@param rows table
local function save_corpus(path, meta, rows)
    local encode = require_json_encoder()
    fs.ensure_parent_dir(path)
    local ok, body = pcall(encode, { meta = meta, rows = rows })
    if not ok then
        error(
            string.format(
                "train_othello6_npc: failed to encode the corpus for %q: %s",
                path,
                tostring(body)
            )
        )
    end
    local handle, open_err = io.open(path, "w")
    if handle == nil then
        error(
            string.format(
                "train_othello6_npc: cannot open corpus_path %q for writing: %s",
                path,
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
                "train_othello6_npc: write to corpus_path %q failed: %s",
                path,
                tostring(write_err)
            )
        )
    end
end

--- Hold one meta field to what this run needs, naming the consequence.
local function require_meta_match(path, meta, key, expected)
    local got = meta[key]
    if got ~= expected then
        error(
            string.format(
                "train_othello6_npc: the corpus at %q carries meta.%s = %s but this run needs "
                    .. "%s, and %s. Point corpus_path at a file built for this run, or at one "
                    .. "that does not exist yet to have it generated",
                path,
                key,
                tostring(got),
                tostring(expected),
                META_MISMATCH_REASONS[key]
            )
        )
    end
end

--- Decode a saved corpus and hold it to this run.
---@param path string
---@param body string File contents
---@param ctx_len integer Width the model was built at
---@return table rows, table meta
local function load_corpus(path, body, ctx_len)
    local decode = require_json_decoder()
    local ok, payload = pcall(decode, body)
    if not ok then
        error(
            string.format(
                "train_othello6_npc: corpus_path %q is not readable JSON: %s",
                path,
                tostring(payload)
            )
        )
    end
    if type(payload) ~= "table" or type(payload.meta) ~= "table" then
        error(
            string.format(
                "train_othello6_npc: corpus_path %q must hold an object with a meta table and "
                    .. "a rows array; a corpus without its conditions cannot be checked against "
                    .. "the run that reads it",
                path
            )
        )
    end
    local meta = payload.meta
    local rows = payload.rows
    if type(rows) ~= "table" or #rows == 0 then
        error(string.format("train_othello6_npc: corpus_path %q holds no rows", path))
    end

    require_meta_match(path, meta, "game", CORPUS_GAME)
    require_meta_match(path, meta, "ctx_len", ctx_len)
    require_meta_match(path, meta, "vocab_size", VOCAB.size)
    require_meta_match(path, meta, "depth", DEPTH)
    require_meta_match(path, meta, "style", STYLE)

    -- Every row is measured, not just the first: a file that is right at
    -- the head and ragged further in trains fine and decodes to nothing.
    for index = 1, #rows do
        local row = rows[index]
        if type(row) ~= "table" or #row ~= meta.ctx_len then
            error(
                string.format(
                    "train_othello6_npc: row %d of corpus_path %q is %s tokens wide but "
                        .. "meta.ctx_len is %s",
                    index,
                    path,
                    type(row) == "table" and tostring(#row) or ("a " .. type(row)),
                    tostring(meta.ctx_len)
                )
            )
        end
    end
    return rows, meta
end

-- ─── Probe position ─────────────────────────────────────────────────

--- A reachable position the trained model is asked to answer twice.
---
--- Played out rather than written by hand, and played without the
--- teacher, so it exists before the model does and stays a position the
--- corpus distribution actually contains: a hand-built board can encode
--- a line the model is never asked about, and agreeing with itself on
--- one of those measures nothing.
local function probe_state()
    local rng = alc.math.rng_create(SEED)
    local random_move = othello.policy_random(rng)
    local state = othello.new_game(SEED)
    for _ = 1, RANDOM_OPENING_MAX do
        if othello.is_over(state) then
            break
        end
        state = othello.apply(state, random_move(state))
    end
    return state
end

local probe = probe_state()

-- ─── Train ──────────────────────────────────────────────────────────

local preset_opts = model_spec()
preset_opts.device = "cpu"
preset_opts.dtype = "f32"
-- Full FT of a customized GPT-2 is random-init only; there is no
-- pretrained bundle for a shape that does not exist upstream, and the
-- bridge refuses `pretrained = true` for a custom spec outright.
preset_opts.pretrained = false

local handle = alc.nn.preset.gpt2("custom", preset_opts)

local ctx_len = handle:ctx()
local model_vocab = handle:vocab()
-- The handle is asked for its own shape rather than trusted to have
-- taken the one it was given: a preset that silently fell back to a
-- named variant would train a model the corpus rows do not fit.
if ctx_len ~= MODEL.ctx or model_vocab ~= MODEL.vocab then
    error(
        string.format(
            "train_othello6_npc: the preset built a ctx %d / vocab %d model but the run asked "
                .. "for ctx %d / vocab %d",
            ctx_len,
            model_vocab,
            MODEL.ctx,
            MODEL.vocab
        )
    )
end

local target = STEPS * BATCH + CORPUS_SLACK_BATCHES * BATCH
local corpus_started = os.clock()

local rows, playouts, corpus_seed, corpus_opening_max
--- Where the rows this run trains on came from: `"generated"`,
--- `"loaded"` or `"generated_and_saved"`.
local corpus_source
--- Floor that decided the playout count, or `nil` when the rows were
--- read rather than played.
local games_floor_applied

local saved_body = CORPUS_PATH ~= nil and read_corpus_body(CORPUS_PATH) or nil
if saved_body ~= nil then
    -- A built corpus holds the games it holds. `ctx.games` asks for a
    -- size, and honouring it here is impossible, so it is refused rather
    -- than dropped: a silently ignored floor reports the numbers of a
    -- corpus the caller did not ask for.
    if GAMES_REQUESTED then
        error(
            string.format(
                "train_othello6_npc: ctx.games asks for a corpus size but corpus_path %q is "
                    .. "already built, and the size of a saved corpus is a property of the "
                    .. "file. Drop ctx.games, or point corpus_path at a file that does not "
                    .. "exist yet to have one generated at that size",
                CORPUS_PATH
            )
        )
    end
    local meta
    rows, meta = load_corpus(CORPUS_PATH, saved_body, ctx_len)
    playouts = meta.games
    corpus_seed = meta.seed
    corpus_opening_max = meta.random_opening_max
    corpus_source = "loaded"
    -- The rows are fixed now, so the training budget is what has to give.
    -- Said here rather than mid-run, because `alc.nn.data.synthetic`
    -- walks its rows once and a short corpus is a trainer that stops
    -- part of the way through.
    if #rows < target then
        error(
            string.format(
                "train_othello6_npc: corpus_path %q holds %d rows but %d steps of batch %d "
                    .. "consume %d; lower steps or batch, or generate a larger corpus",
                CORPUS_PATH,
                #rows,
                STEPS,
                BATCH,
                target
            )
        )
    end
else
    rows, playouts = build_full_corpus(ctx_len, target)
    corpus_seed = SEED
    corpus_opening_max = RANDOM_OPENING_MAX
    games_floor_applied = GAMES_FLOOR
    if CORPUS_PATH ~= nil then
        save_corpus(CORPUS_PATH, corpus_meta(ctx_len, playouts), rows)
        corpus_source = "generated_and_saved"
    else
        corpus_source = "generated"
    end
end

local corpus_seconds = os.clock() - corpus_started
log(
    string.format(
        "corpus: %s %d rows x %d tokens from %d playouts, target %d rows, floor %s games, "
            .. "%.2fs (depth %d, %s%s)",
        corpus_source,
        #rows,
        ctx_len,
        playouts,
        target,
        games_floor_applied ~= nil and tostring(games_floor_applied) or "n/a",
        corpus_seconds,
        DEPTH,
        STYLE,
        CORPUS_PATH ~= nil and (", " .. CORPUS_PATH) or ""
    )
)
-- The seed and the opening draw of a loaded corpus are the file's, not
-- this run's. They differ legitimately -- a sweep over `steps` reuses
-- one corpus under whatever seed built it -- so the difference is
-- reported rather than refused, and the result carries the corpus's own
-- values so a reader is never left inferring them from ctx.
if corpus_source == "loaded" then
    if corpus_seed ~= SEED then
        log(
            string.format(
                "corpus seed %s differs from ctx.seed %d; the probe and the self-play still "
                    .. "run under ctx.seed",
                tostring(corpus_seed),
                SEED
            )
        )
    end
    if corpus_opening_max ~= RANDOM_OPENING_MAX then
        log(
            string.format(
                "corpus random_opening_max %s differs from this run's %d",
                tostring(corpus_opening_max),
                RANDOM_OPENING_MAX
            )
        )
    end
end

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = BATCH,
    ctx_len = ctx_len,
    shuffle = true,
    pad_id = VOCAB.pad_id,
})

log(string.format("full_ft: %d steps, lr=%g, batch=%d", STEPS, LR, BATCH))
local train_opts = {
    lr = LR,
    batch = BATCH,
    steps = STEPS,
    warmup = 0,
    schedule = "Constant",
    name = NAME,
}
-- Checkpoints are written for a later look, without a hook: nothing in
-- this run reads a half-trained model, and a hook is what a run that
-- judged one would need.
if CKPT_EVERY > 0 then
    train_opts.ckpt_every = CKPT_EVERY
    train_opts.ckpt_keep = CKPT_KEEP
    log(
        string.format(
            "checkpoints: every %d steps, keeping %d, unread by this run",
            CKPT_EVERY,
            CKPT_KEEP
        )
    )
end

local card_id = alc.nn.trainer.run_full_ft(handle, dataset, train_opts)
if type(card_id) ~= "string" or #card_id == 0 then
    error("train_othello6_npc: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "othello6_npc",
    note = string.format("6x6 Othello NPC, depth %d, %s style", DEPTH, STYLE),
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

local pinned_bare = false
if DEPTH == BARE_DEPTH and STYLE == BARE_STYLE and ALIAS ~= BARE_ALIAS then
    alc.card.alias_set(BARE_ALIAS, card_id, {
        pkg = "othello6_npc",
        note = "6x6 Othello NPC, default teacher pair",
    })
    pinned_bare = true
    log(string.format("bare alias %q pinned to the same card", BARE_ALIAS))
end

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("train_othello6_npc: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Read the model back ────────────────────────────────────────────

npc.reset_cache()

--- Ask the NPC package, naming the Card and the teacher pair it was
--- labelled by.
local function ask(payload)
    local out = npc.run({
        task = alc.json_encode(payload),
        card_alias = ALIAS,
        depth = DEPTH,
        style = STYLE,
    })
    return out.result
end

local determinism_text = ask({ mode = "determinism", state = probe })
log(string.format("determinism -> %s", determinism_text))

-- Both measurements come off one self-play batch, played greedily
-- against the teacher that labelled the corpus.
local selfplay = ask({ mode = "selfplay", games = CHECK_GAMES, seed = SEED })
log(string.format("selfplay -> %s", selfplay))

local illegal = tonumber(selfplay:match("illegal=(%d+)"))
local hits, moves = selfplay:match("style_hits=(%d+)/(%d+)")
hits, moves = tonumber(hits), tonumber(moves)
-- Reported for the record rather than read: see the header on why a win
-- rate is not a measurement of strength here.
local winrate = tonumber(selfplay:match("winrate=([%d%.]+)"))
if illegal == nil or hits == nil or moves == nil or winrate == nil then
    error("train_othello6_npc: the self-play answer is not in the expected shape: " .. selfplay)
end
if moves == 0 then
    error("train_othello6_npc: self-play made no move, so nothing was measured")
end

-- `style_match` is recomputed from the counts rather than parsed off
-- the formatted field, which is rounded to two decimals: a sweep that
-- compares depths reads the difference between two of these.
local style_match = hits / moves
local legal_rate = (moves - illegal) / moves
log(
    string.format(
        "measured: style_match=%.4f (%d/%d), legal_rate=%.4f (%d raw-illegal), "
            .. "train_loss=%.4f vs baseline %.4f",
        style_match,
        hits,
        moves,
        legal_rate,
        illegal,
        train_loss,
        baseline_loss
    )
)

return {
    -- The two conditions the phase is done under: every raw decision
    -- was legal, and the loss went under the uniform baseline. The win
    -- rate is not among them.
    ok = illegal == 0 and train_loss < baseline_loss,
    card_id = card_id,
    alias = ALIAS,
    pinned_bare_alias = pinned_bare,
    name = NAME,
    depth = DEPTH,
    style = STYLE,
    seed = SEED,
    -- Corpus, as it was actually built. The playout count and the row
    -- count are the same number by construction; both are reported so a
    -- reader does not have to know that to read the result.
    games = playouts,
    games_floor = games_floor_applied,
    rows = #rows,
    rows_target = target,
    corpus_seconds = corpus_seconds,
    -- Where the rows came from, and the conditions they carry. A run
    -- that reused a file reports the file's seed and opening draw, not
    -- the ones ctx asked for, so a reader of the result never has to
    -- guess which of the two the rows were played under.
    corpus_source = corpus_source,
    corpus_path = CORPUS_PATH,
    corpus_seed = corpus_seed,
    random_opening_max = corpus_opening_max,
    -- Model and training budget.
    ctx_len = ctx_len,
    model_vocab = model_vocab,
    vocab_size = VOCAB.size,
    layers = MODEL.layers,
    heads = MODEL.heads,
    dim = MODEL.dim,
    steps = STEPS,
    batch = BATCH,
    lr = LR,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    -- Greedy readings.
    check_games = CHECK_GAMES,
    moves = moves,
    illegal = illegal,
    legal_rate = legal_rate,
    style_match = style_match,
    style_hits = hits,
    deterministic = determinism_text:find("deterministic=true", 1, true) ~= nil,
    winrate = winrate,
    selfplay = selfplay,
    -- Checkpoints, and what it takes to rebuild one.
    ckpt_every = CKPT_EVERY,
    ckpt_keep = CKPT_EVERY > 0 and CKPT_KEEP or nil,
    ckpt_arch = CKPT_EVERY > 0 and CKPT_ARCH or nil,
    ckpt_spec = CKPT_EVERY > 0 and alc.json_encode(model_spec()) or nil,
}
