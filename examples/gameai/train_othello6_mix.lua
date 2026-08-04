-- Bake a 6x6 Othello NPC on a *style-mixed* corpus: saved rows -> Full
-- FT -> Card, then one compliance reading per parent.
--
-- Self-contained script for `alc_run` (`code_file` form). It is the
-- mixed sibling of `train_othello6_npc.lua`, and it differs in exactly
-- two places: the corpus it reads was labelled by
-- `othello6_mix.mixed_policy` rather than by a single teacher, and the
-- Card it produces is therefore measured against *both* parents instead
-- of against the one teacher that labelled it.
--
-- No `alc.llm` call happens anywhere in this path, so the run never
-- pauses for a host response.
--
-- ## Why the corpus is required rather than generated
--
-- `corpus_path` is a required field here, and it has to name a file that
-- already exists. The single-style driver generates when the file is
-- absent; this one refuses, for two reasons that both come out of
-- `workspace/EXPERIMENT-RULES.md`.
--
-- A mixed playout costs about twice a single-style one -- both parents
-- search every position before the draw picks between them -- so the
-- 8000-game corpus this experiment sweeps takes about 3 min 30 s to
-- play and 0.06 s to read back. A driver that could quietly generate is
-- a driver that turns a 40-second bake into a four-minute one whenever a
-- path is mistyped, which is precisely the failure the rules file was
-- written after.
--
-- The second reason is the comparison itself. The point of the sweep is
-- three bakes over three fixed corpora at beta 0.25 / 0.5 / 0.75, and a
-- corpus re-rolled underneath a bake moves the data the comparison is
-- being made on. Generating is `gen_othello6_mix_corpus.lua`'s job and
-- nothing else's.
--
-- ## What is measured
--
-- A mixture has no single teacher, so `style_match` against "the"
-- teacher does not exist. Two self-play runs are played instead, one
-- scored against each parent, and both numbers are reported:
--
--   * `style_match_a` -- share of model moves equal to `styles[1]`'s
--   * `style_match_b` -- share of model moves equal to `styles[2]`'s
--
-- Both runs decode the same Card at the same seed over the same number
-- of games, so the model plays the *same* games in both: only the
-- teacher the moves are compared against moves. That is what makes the
-- pair readable as a split of one behaviour rather than as two
-- measurements, and the driver holds the two runs to the same move count
-- and the same raw-illegal count to keep it true.
--
-- Neither number is gated on. What a fitted mixture should score against
-- each parent has not been measured yet -- that reading is the whole
-- point of this phase -- so a threshold here would be a guess dressed as
-- a criterion. `ok` reads the training result alone: `train_loss` under
-- `ln(vocab)`, the uniform-random baseline, which is the evidence that
-- gradients flowed at all. `illegal == 0` is *not* required either, for
-- the same reason it is not in the single-style driver at this scale:
-- the measured legal rate at 20k rows is 0.78, so a zero would be a
-- criterion no run of this size meets.
--
-- `legal_rate` rides along as the direct reading of the model: the share
-- of decisions whose *raw* argmax was already legal. The move that ends
-- up played is legal by construction, because `othello6_npc` walks the
-- logit ranking against the legal set.
--
-- ## The shape
--
-- `layers = 2` is the default, against the single-style driver's 4.
-- Measured on 2026-08-04 over a 20k corpus, the depth axis is monotone
-- the wrong way -- 2 layers reads legal 0.7584 in 95 s, 4 reads 0.5853
-- in 177 s at the same steps -- so the deeper default is the one that
-- was wrong. The full table is in
-- `workspace/tasks/f3bd6c0b-othello-slm/results.md`.
--
-- `steps = 249` is the budget an 8000-row corpus allows at batch 32:
-- `249 * 32 + 32 = 8000` exactly, one pass with a batch in hand. The
-- single-style sweep saturated at one pass, so it is also the budget
-- worth spending.
--
-- ctx:
--   corpus_path -- JSON corpus written by `gen_othello6_mix_corpus.lua`
--                  (required; it has to exist)
--   styles      -- the two parents, in order. Checked against the file
--                  when given; taken from the file when not
--   beta        -- share of the decisions `styles[1]` answered. Checked
--                  against the file when given; taken from it when not
--   depth       -- plies both parents searched. Checked against the file
--                  when given; taken from it when not
--   steps       -- training steps (default 249)
--   batch       -- batch size (default 32)
--   lr          -- learning rate (default 3e-3)
--   layers      -- transformer blocks (default 2)
--   heads       -- attention heads (default 4)
--   dim         -- model width (default 128)
--   check_games -- self-play games behind each compliance reading
--                  (default 20)
--   seed        -- seed of the two self-play runs (default 20260804)
--   name        -- Card name (default "othello6-npc-mix")
--   alias       -- alias the Card is pinned to (default
--                  `othello6_npc_mix_<initials>_b<percent>`, e.g.
--                  `othello6_npc_mix_cg_b50`)
--
-- Unlike the single-style driver, the mixture fields are *checked* when
-- present rather than required to be: beta is baked into the corpus by
-- the run that generated it and cannot be changed by asking, so the
-- file is the source and ctx is an assertion against it. Passing one
-- that disagrees is a caller who believes they are baking a different
-- mixture than they are, which is exactly what has to fail loudly.

local othello = require("othello6")
local teacher = require("othello6_teacher")
local npc = require("othello6_npc")

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
    beta = true,
    check_games = true,
    corpus_path = true,
    depth = true,
    dim = true,
    heads = true,
    layers = true,
    lr = true,
    name = true,
    seed = true,
    steps = true,
    styles = true,
}

--- Fields this driver refuses by name, and why each one cannot be
--- honoured here.
---
--- All three would be honoured somewhere else in the pipeline, which is
--- what makes silently dropping them dangerous: a caller passing
--- `games` is asking for a corpus size and would otherwise be handed the
--- size of whatever file `corpus_path` names, and a caller passing
--- `style` is asking for a single-teacher bake and would be handed a
--- mixture.
local REFUSED_FIELDS = {
    games = "a mixed corpus holds the games it holds; it is sized by "
        .. "gen_othello6_mix_corpus.lua, which this driver never calls",
    random_opening_max = "the opening draw is a property of the corpus that was already played, "
        .. "and is read back off its meta",
    style = "a mixture has two parents, so the field is `styles`; a single-teacher bake is "
        .. "train_othello6_npc.lua",
}

--- Shape fields the encoding fixes, and what fixes each of them.
---
--- `ctx` is the width every corpus row is padded to and `vocab` is the
--- size of the alphabet those rows are drawn from, so neither is a dial
--- of the experiment: a run that moved one would build a model the rows
--- it then trains on no longer fit. The dials are `layers`, `heads` and
--- `dim`.
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
--- Runs before anything else, so a misspelled request never reaches the
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
                "train_othello6_mix: ctx field(s) refused: %s",
                table.concat(refused, "; ")
            )
        )
    end
    if #fixed > 0 then
        error(
            string.format(
                "train_othello6_mix: ctx field(s) %s are fixed by the corpus encoding rather "
                    .. "than dials of the run: a model built at another one would not fit the "
                    .. "rows it is then trained on. The shape dials are layers, heads and dim",
                table.concat(fixed, ", ")
            )
        )
    end
    if #unknown > 0 then
        error(
            string.format(
                "train_othello6_mix: unknown ctx field(s) %s (known: %s)",
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
                "train_othello6_mix: ctx.%s must be a finite number, got %s",
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
--- `steps = "249"` read as 249 would train under a request the caller
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
                "train_othello6_mix: ctx.%s must be an integer, got %s",
                name,
                tostring(raw)
            )
        )
    end
    raw = math.floor(raw)
    if min ~= nil and raw < min then
        error(
            string.format("train_othello6_mix: ctx.%s must be at least %d, got %d", name, min, raw)
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
                "train_othello6_mix: ctx.%s must be positive, got %s",
                name,
                tostring(raw)
            )
        )
    end
    return raw
end

---@param name string ctx key
---@param default string|nil Value used when absent, or `nil` to make the
---       field required
---@return string
local function string_field(name, default)
    local raw = CTX[name]
    if raw == nil then
        if default == nil then
            error(
                string.format(
                    "train_othello6_mix: ctx.%s is required; a mixed corpus costs minutes to "
                        .. "play and is generated once by gen_othello6_mix_corpus.lua, so this "
                        .. "driver reads one rather than rolling its own",
                    name
                )
            )
        end
        return default
    end
    if type(raw) ~= "string" then
        error(string.format("train_othello6_mix: ctx.%s must be a string, got %s", name, type(raw)))
    end
    if #raw == 0 then
        error(string.format("train_othello6_mix: ctx.%s must not be empty", name))
    end
    return raw
end

local CORPUS_PATH = string_field("corpus_path", nil)
local STEPS = int_field("steps", 249, 1)
local BATCH = int_field("batch", 32, 1)
local SEED = int_field("seed", 20260804)
local LR = positive_number_field("lr", 3e-3)
local NAME = string_field("name", "othello6-npc-mix")
local CHECK_GAMES = int_field("check_games", 20, 1)

--- Requested mixture, or `nil` where ctx said nothing.
---
--- These are assertions against the corpus rather than dials: the file
--- was labelled under one mixture and no request here can change it, so
--- a value given is checked and a value absent is read off the file.
local WANT_BETA = nil
if CTX.beta ~= nil then
    WANT_BETA = require_finite("beta", CTX.beta)
end
local WANT_DEPTH = nil
if CTX.depth ~= nil then
    WANT_DEPTH = int_field("depth", 0, 1)
end
local WANT_STYLES = nil
if CTX.styles ~= nil then
    if type(CTX.styles) ~= "table" then
        error(
            string.format(
                "train_othello6_mix: ctx.styles must be a table naming two parents, got %s",
                type(CTX.styles)
            )
        )
    end
    WANT_STYLES = {}
    for index = 1, #CTX.styles do
        WANT_STYLES[index] = CTX.styles[index]
    end
end

--- `ctx.alias`, checked for type now and defaulted after the mixture is
--- known: the default name carries the parents and the weight, and both
--- of those come off the corpus.
local ALIAS_OVERRIDE = nil
if CTX.alias ~= nil then
    ALIAS_OVERRIDE = string_field("alias", "")
end

-- ─── Model ──────────────────────────────────────────────────────────

--- Shape of the model this experiment trains.
---
--- `ctx` and `vocab` are properties of the encoding. `layers` defaults
--- to 2 rather than the single-style driver's 4 because the depth axis
--- was measured and is monotone the wrong way; see the header.
local MODEL = {
    ctx = 48,
    vocab = 64,
    layers = int_field("layers", 2, 1),
    heads = int_field("heads", 4, 1),
    dim = int_field("dim", 128, 1),
}

-- Multi-head attention splits the width across the heads, so a width
-- that does not divide is not a slow model but an impossible one. It is
-- checked here, before the corpus is opened, because it is a property of
-- the two numbers alone.
if MODEL.dim % MODEL.heads ~= 0 then
    error(
        string.format(
            "train_othello6_mix: ctx.dim = %d is not a multiple of ctx.heads = %d, so the width "
                .. "cannot be split across the heads",
            MODEL.dim,
            MODEL.heads
        )
    )
end

--- The shape keys, as `alc.nn.preset.gpt2("custom", ...)` takes them.
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
-- model or a corpus.
if VOCAB.size > MODEL.vocab then
    error(
        string.format(
            "train_othello6_mix: the alphabet holds %d characters but the model vocabulary is %d",
            VOCAB.size,
            MODEL.vocab
        )
    )
end
if othello.ROW_LEN > MODEL.ctx then
    error(
        string.format(
            "train_othello6_mix: a training line needs %d tokens but the model context is %d",
            othello.ROW_LEN,
            MODEL.ctx
        )
    )
end

local function log(msg)
    alc.log("info", "[othello6-mix-train] " .. msg)
end

-- ─── Corpus file ────────────────────────────────────────────────────
--
-- A saved corpus is rows plus the conditions they were played under.
-- The conditions are the whole point: rows alone load under any run, so
-- a file carrying only rows would let a bake train on one mixture and
-- report the result as another's.
--
-- The shape is the one `gen_othello6_mix_corpus.lua` writes. The two are
-- kept in step by this reader: a generator that drifted is answered by a
-- loud mismatch on the next load.

--- `meta.game` every corpus this driver reads has to carry.
local CORPUS_GAME = "othello6"

--- `meta.kind` that marks a corpus as style-mixed.
---
--- The one field that tells this file apart from a single-style corpus:
--- both carry the same game, the same row width and the same alphabet,
--- so without it a mixture and a single teacher's games are
--- indistinguishable to a reader, and the failure of that distinction is
--- a mixture reported as one parent's result.
local CORPUS_KIND = "mix"

--- Why each meta field has to match, said in the error that reports it.
local META_MISMATCH_REASONS = {
    game = "the rows encode another game entirely",
    ctx_len = "rows of another width do not fit the model this run builds",
    vocab_size = "the same token id stands for another move under another alphabet",
}

local function require_json_decoder()
    if type(alc) ~= "table" or type(alc.json_decode) ~= "function" then
        error(
            "train_othello6_mix: alc.json_decode is required to read a corpus (host bridge "
                .. "missing)"
        )
    end
    return alc.json_decode
end

--- Read `path`, or answer `nil` when it is not there.
---
--- Absence is refused by the caller rather than handled here, so this
--- only has to tell "not there" apart from "there and unreadable": a
--- permission error returned as `nil` would be reported as a missing
--- corpus and send the caller off to regenerate a file that exists.
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
                "train_othello6_mix: corpus_path %q opened but could not be read: %s",
                path,
                tostring(read_err)
            )
        )
    end
    return body
end

--- Hold one meta field to what this run needs, naming the consequence.
local function require_meta_match(path, meta, key, expected)
    local got = meta[key]
    if got ~= expected then
        error(
            string.format(
                "train_othello6_mix: the corpus at %q carries meta.%s = %s but this run needs "
                    .. "%s, and %s. Point corpus_path at a file built for this run",
                path,
                key,
                tostring(got),
                tostring(expected),
                META_MISMATCH_REASONS[key]
            )
        )
    end
end

--- Hold a value to the canonical list it has to be a member of.
local function require_member(what, value, list)
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
            "train_othello6_mix: %s = %s is not one of %s",
            what,
            tostring(value),
            table.concat(names, ", ")
        )
    )
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
                "train_othello6_mix: corpus_path %q is not readable JSON: %s",
                path,
                tostring(payload)
            )
        )
    end
    if type(payload) ~= "table" or type(payload.meta) ~= "table" then
        error(
            string.format(
                "train_othello6_mix: corpus_path %q must hold an object with a meta table and a "
                    .. "rows array; a corpus without its conditions cannot be checked against "
                    .. "the run that reads it",
                path
            )
        )
    end
    local meta = payload.meta
    local rows = payload.rows
    if type(rows) ~= "table" or #rows == 0 then
        error(string.format("train_othello6_mix: corpus_path %q holds no rows", path))
    end

    require_meta_match(path, meta, "game", CORPUS_GAME)
    -- Reported before the shape checks and with its own message: a
    -- single-style corpus is the file a caller is most likely to have
    -- pointed here, it passes every other check, and the run that
    -- followed would report one parent's teacher as a mixture.
    if meta.kind ~= CORPUS_KIND then
        error(
            string.format(
                "train_othello6_mix: the corpus at %q carries meta.kind = %s but this driver "
                    .. "reads %q corpora only. A corpus written by gen_othello6_corpus.lua is "
                    .. "labelled by one teacher and is baked by train_othello6_npc.lua; a "
                    .. "mixture is written by gen_othello6_mix_corpus.lua",
                path,
                meta.kind == nil and "nothing" or string.format("%q", tostring(meta.kind)),
                CORPUS_KIND
            )
        )
    end
    require_meta_match(path, meta, "ctx_len", ctx_len)
    require_meta_match(path, meta, "vocab_size", VOCAB.size)

    -- Every row is measured, not just the first: a file that is right at
    -- the head and ragged further in trains fine and decodes to nothing.
    for index = 1, #rows do
        local row = rows[index]
        if type(row) ~= "table" or #row ~= meta.ctx_len then
            error(
                string.format(
                    "train_othello6_mix: row %d of corpus_path %q is %s tokens wide but "
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

--- Read the mixture off the corpus, checking it against ctx where ctx
--- spoke.
---
--- The file is the source: the labels were drawn under one weight, one
--- pair and one depth, and no request made here can change what is
--- already in the rows. A ctx field is therefore an assertion, and a
--- disagreement is a caller who believes they are baking a different
--- mixture than they are.
---@param path string
---@param meta table
---@return table styles, number beta, integer depth
local function resolve_mixture(path, meta)
    local styles = meta.styles
    if type(styles) ~= "table" or #styles ~= 2 then
        error(
            string.format(
                "train_othello6_mix: the corpus at %q carries no meta.styles pair, so the two "
                    .. "parents the Card has to be measured against are unknown",
                path
            )
        )
    end
    require_member("the corpus meta.styles[1]", styles[1], othello.STYLES)
    require_member("the corpus meta.styles[2]", styles[2], othello.STYLES)
    if styles[1] == styles[2] then
        error(
            string.format(
                "train_othello6_mix: the corpus at %q names %s as both parents, which is not a "
                    .. "mixture",
                path,
                tostring(styles[1])
            )
        )
    end

    local beta = meta.beta
    if type(beta) ~= "number" or beta ~= beta or beta <= 0 or beta >= 1 then
        error(
            string.format(
                "train_othello6_mix: the corpus at %q carries meta.beta = %s, which is not a "
                    .. "weight in the open interval (0, 1); 0 and 1 are the parents themselves",
                path,
                tostring(beta)
            )
        )
    end

    local depth = require_member("the corpus meta.depth", meta.depth, teacher.DEPTHS)

    if WANT_STYLES ~= nil then
        local mismatch = #WANT_STYLES ~= 2
        for index = 1, 2 do
            mismatch = mismatch or WANT_STYLES[index] ~= styles[index]
        end
        if mismatch then
            error(
                string.format(
                    "train_othello6_mix: ctx.styles asks for %s but the corpus at %q was "
                        .. "labelled by %s/%s; the parents are baked into the rows and cannot "
                        .. "be changed by asking",
                    table.concat(WANT_STYLES, "/"),
                    path,
                    styles[1],
                    styles[2]
                )
            )
        end
    end
    if WANT_BETA ~= nil and WANT_BETA ~= beta then
        error(
            string.format(
                "train_othello6_mix: ctx.beta asks for %s but the corpus at %q was labelled at "
                    .. "%s; the weight is baked into the rows and cannot be changed by asking. "
                    .. "Generate a corpus at that weight with gen_othello6_mix_corpus.lua",
                tostring(WANT_BETA),
                path,
                tostring(beta)
            )
        )
    end
    if WANT_DEPTH ~= nil and WANT_DEPTH ~= depth then
        error(
            string.format(
                "train_othello6_mix: ctx.depth asks for %d but the corpus at %q was labelled by "
                    .. "parents searching %s plies; a Card would then be measured against "
                    .. "teachers that never labelled it",
                WANT_DEPTH,
                path,
                tostring(depth)
            )
        )
    end

    return { styles[1], styles[2] }, beta, depth
end

--- Alias a mixture pins unless `ctx.alias` overrides it.
---
--- The name carries the two parents and the weight, because those are
--- the three things that cannot be read back off the weights. The weight
--- is written as whole percent: a beta between two percent points would
--- put two different mixtures on the same alias and the second bake
--- would silently re-point the first one's Card, so it is refused rather
--- than rounded.
---@param styles table Parent pair
---@param beta number Weight of the first parent
---@return string alias
local function default_alias(styles, beta)
    local percent = beta * 100
    local whole = math.floor(percent + 0.5)
    if math.abs(percent - whole) > 1e-9 then
        error(
            string.format(
                "train_othello6_mix: the corpus beta %s does not land on a whole percent, so "
                    .. "the default alias would name it the same as %g; pass ctx.alias "
                    .. "explicitly",
                tostring(beta),
                whole / 100
            )
        )
    end
    return string.format(
        "othello6_npc_mix_%s%s_b%d",
        styles[1]:sub(1, 1),
        styles[2]:sub(1, 1),
        whole
    )
end

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
            "train_othello6_mix: the preset built a ctx %d / vocab %d model but the run asked "
                .. "for ctx %d / vocab %d",
            ctx_len,
            model_vocab,
            MODEL.ctx,
            MODEL.vocab
        )
    )
end

local corpus_started = os.clock()
local saved_body = read_corpus_body(CORPUS_PATH)
if saved_body == nil then
    error(
        string.format(
            "train_othello6_mix: corpus_path %q does not exist. This driver reads a mixed "
                .. "corpus and never generates one -- a mixed playout costs about twice a "
                .. "single-style one, so a bake that quietly generated would turn a "
                .. "forty-second run into a four-minute one. Write the file first with "
                .. "gen_othello6_mix_corpus.lua",
            CORPUS_PATH
        )
    )
end

local rows, meta = load_corpus(CORPUS_PATH, saved_body, ctx_len)
local STYLES, BETA, DEPTH = resolve_mixture(CORPUS_PATH, meta)
local corpus_seconds = os.clock() - corpus_started

local ALIAS = ALIAS_OVERRIDE or default_alias(STYLES, BETA)

-- The rows are fixed, so the training budget is what has to give. Said
-- here rather than mid-run, because `alc.nn.data.synthetic` walks its
-- rows once and a short corpus is a trainer that stops part of the way
-- through.
local target = STEPS * BATCH + BATCH
if #rows < target then
    error(
        string.format(
            "train_othello6_mix: corpus_path %q holds %d rows but %d steps of batch %d consume "
                .. "%d (one batch of slack included); lower steps or batch, or generate a "
                .. "larger corpus",
            CORPUS_PATH,
            #rows,
            STEPS,
            BATCH,
            target
        )
    )
end

log(
    string.format(
        "corpus: loaded %d rows x %d tokens from %s in %.2fs (%s/%s beta=%g, depth %d, %s "
            .. "playouts, seed %s)",
        #rows,
        ctx_len,
        CORPUS_PATH,
        corpus_seconds,
        STYLES[1],
        STYLES[2],
        BETA,
        DEPTH,
        tostring(meta.games),
        tostring(meta.seed)
    )
)

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = BATCH,
    ctx_len = ctx_len,
    shuffle = true,
    pad_id = VOCAB.pad_id,
})

log(string.format("full_ft: %d steps, lr=%g, batch=%d, layers=%d", STEPS, LR, BATCH, MODEL.layers))
local card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
    lr = LR,
    batch = BATCH,
    steps = STEPS,
    warmup = 0,
    schedule = "Constant",
    name = NAME,
})
if type(card_id) ~= "string" or #card_id == 0 then
    error("train_othello6_mix: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "othello6_npc",
    note = string.format(
        "6x6 Othello NPC, %s/%s mixture at beta=%g, depth %d",
        STYLES[1],
        STYLES[2],
        BETA,
        DEPTH
    ),
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all, and it is the only
-- condition `ok` reads.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("train_othello6_mix: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Compliance: one self-play run per parent ───────────────────────
--
-- A mixture has no single teacher to be scored against, so it is scored
-- against each parent in turn. Both runs decode the same Card over the
-- same games at the same seed; `task.style` moves the teacher the moves
-- are compared against, not the moves themselves.

npc.reset_cache()

---@param parent string Parent the moves are scored against
---@return number style_match, integer illegal, integer moves, string line
local function selfplay_against(parent)
    local line = npc.run({
        task = alc.json_encode({
            mode = "selfplay",
            games = CHECK_GAMES,
            seed = SEED,
            depth = DEPTH,
            style = parent,
        }),
        card_alias = ALIAS,
        -- The basis pair is named on the task rather than here, so both
        -- runs override it explicitly and neither inherits the package
        -- default.
        depth = DEPTH,
    }).result
    log(string.format("selfplay vs %s -> %s", parent, line))

    local illegal = tonumber(line:match("illegal=(%d+)"))
    local hits, moves = line:match("style_hits=(%d+)/(%d+)")
    hits, moves = tonumber(hits), tonumber(moves)
    if illegal == nil or hits == nil or moves == nil then
        error(
            string.format(
                "train_othello6_mix: self-play against %s is not in the expected shape: %s",
                parent,
                line
            )
        )
    end
    if moves == 0 then
        error(
            string.format(
                "train_othello6_mix: self-play against %s made no move, so nothing was measured",
                parent
            )
        )
    end
    -- Recomputed from the counts rather than parsed off the formatted
    -- field, which is rounded to two decimals: the reading this phase
    -- wants is the difference between two of these.
    return hits / moves, illegal, moves, line
end

local match_a, illegal_a, moves_a, selfplay_a = selfplay_against(STYLES[1])
local match_b, illegal_b, moves_b, selfplay_b = selfplay_against(STYLES[2])

-- The same Card plays the same games under the same seed in both runs,
-- so only the teacher the moves are scored against differs. If the move
-- counts or the raw-illegal counts moved, something other than the
-- teacher did too, and the two numbers are then not a split of one
-- behaviour -- which is the only reading they are reported for.
if moves_a ~= moves_b or illegal_a ~= illegal_b then
    error(
        string.format(
            "train_othello6_mix: the two compliance runs played differently (%d moves / %d "
                .. "illegal against %s, %d / %d against %s), so the pair is not one behaviour "
                .. "scored twice",
            moves_a,
            illegal_a,
            STYLES[1],
            moves_b,
            illegal_b,
            STYLES[2]
        )
    )
end

local legal_rate = (moves_a - illegal_a) / moves_a
log(
    string.format(
        "measured: style_match_a=%.4f (%s), style_match_b=%.4f (%s), legal_rate=%.4f "
            .. "(%d raw-illegal of %d), train_loss=%.4f vs baseline %.4f",
        match_a,
        STYLES[1],
        match_b,
        STYLES[2],
        legal_rate,
        illegal_a,
        moves_a,
        train_loss,
        baseline_loss
    )
)

return {
    -- `ok` reads the training result alone. The two compliance numbers
    -- are the data this phase exists to collect, and no threshold on
    -- them has been measured yet; `illegal == 0` is not required either,
    -- because the measured legal rate at this scale is 0.78.
    ok = train_loss < baseline_loss,
    card_id = card_id,
    alias = ALIAS,
    name = NAME,
    -- The mixture, as the corpus carries it.
    kind = CORPUS_KIND,
    styles = { STYLES[1], STYLES[2] },
    style_a = STYLES[1],
    style_b = STYLES[2],
    beta = BETA,
    depth = DEPTH,
    -- Corpus. `corpus_source` is always `"loaded"` here -- the field is
    -- kept so a reader can compare a mixed result against a single-style
    -- one without special-casing the shape.
    corpus_source = "loaded",
    corpus_path = CORPUS_PATH,
    corpus_seed = meta.seed,
    corpus_games = meta.games,
    corpus_seconds = corpus_seconds,
    random_opening_max = meta.random_opening_max,
    rows = #rows,
    rows_target = target,
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
    seed = SEED,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    -- Greedy readings, one per parent plus the model's own legal rate.
    check_games = CHECK_GAMES,
    moves = moves_a,
    illegal = illegal_a,
    legal_rate = legal_rate,
    style_match_a = match_a,
    style_match_b = match_b,
    style_match_sum = match_a + match_b,
    selfplay_a = selfplay_a,
    selfplay_b = selfplay_b,
}
