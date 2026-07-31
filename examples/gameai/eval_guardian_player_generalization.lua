-- Prove that a guardian-duel play log trains a player Card that
-- generalises — that the bake learns the habits in the log rather than
-- memorising its positions.
--
-- Self-contained script for `alc_run` (`code_file` form). It is the
-- verification half of the pair shipped with the sample data:
-- `gen_guardian_sample_playlog.lua` produces the data, this script
-- consumes it. In: a training log and a held-out log. Out: the baked
-- Card plus match rates on positions the training corpus never
-- contained, checked against explicit fences. Rerunning it will not
-- reproduce the numbers bit for bit — training shuffles — which is why
-- the verdict is a set of rate fences rather than a golden value.
--
--   alc_run(
--     code_file = "<repo>/examples/gameai/eval_guardian_player_generalization.lua",
--     ctx = {
--       train_file   = "<repo>/examples/gameai/data/guardian_sample_playlog_train.json",
--       holdout_file = "<repo>/examples/gameai/data/guardian_sample_playlog_holdout.json",
--     }
--   )
--
-- The file form is the canonical one: the script reads and decodes the
-- JSON itself, so a few hundred logged turns never ride through a
-- caller's tool-call payload (where a single mistyped character is a
-- corrupted corpus — the strict per-entry validation catches enum
-- slips, but a plausible-but-wrong number would pass silently). Inline
-- arrays are still accepted as `train_moves` / `holdout_moves` for
-- callers that hold a fresh `move_log` and no file, e.g. straight out
-- of an interactive session.
--
-- The bake step mirrors `bake_guardian_player_from_log.lua` parameter
-- for parameter (gpt2 tiny, full FT, steps=800, lr=3e-3, batch=32).
-- What this script adds is the measurement that script deliberately
-- leaves out: `log_match` there is a train fit by construction, while
-- the held-out log here contains games the corpus never saw.
--
-- The sample data was played by a fixed conditional style ("sentinel",
-- documented in the generator). Its rules R1-R5 are functions of the
-- player view alone, so on every held-out position where a rule applies
-- the ground truth is computable — and a position can be classified as
-- seen or unseen by its 13-char encoding against the training set. The
-- style check is what turns "the model matched the log" into "the model
-- applies the habit on positions it never saw". Pass `style_check =
-- false` when feeding logs that were not played by the sentinel style;
-- the style fences are then skipped and only the style-free ones apply.
--
-- Fences (pass = all true):
--   core_holdout   -- style match on held-out rule positions >= 0.90
--                     (skipped when style_check = false)
--   raw_legal      -- raw decode legality on held-out positions >= 0.95
--   loss_descended -- final train loss below the uniform baseline
--
-- Measured on the bundled sample (2026-08-01): core rules 58/58 on
-- held-out positions including every unseen one, overall held-out match
-- ~0.67 with the residual concentrated in the free slots — positions
-- where the view genuinely underdetermines the move, which no amount of
-- data can (or should) close.
--
-- `alc.llm` is never called, so the run never pauses.

local duel = require("guardian_duel")
local npc = require("guardian_player_npc")

local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

--- Resolve one input set from either a file path or an inline array.
--- Exactly one of `ctx.<prefix>_file` / `ctx.<prefix>_moves` must be
--- given: both is ambiguous and refused rather than silently ranked.
local function load_moves(prefix)
    local inline = ctx_field(prefix .. "_moves")
    local path = ctx_field(prefix .. "_file")
    if inline ~= nil and path ~= nil then
        error(
            string.format(
                "eval_guardian_player_generalization: pass either ctx.%s_file or ctx.%s_moves, not both",
                prefix,
                prefix
            )
        )
    end
    if path ~= nil then
        if type(path) ~= "string" then
            error(
                string.format(
                    "eval_guardian_player_generalization: ctx.%s_file must be a string path",
                    prefix
                )
            )
        end
        local f, err = io.open(path, "r")
        if f == nil then
            error(
                string.format(
                    "eval_guardian_player_generalization: cannot open ctx.%s_file %q: %s",
                    prefix,
                    path,
                    tostring(err)
                )
            )
        end
        local body = f:read("a")
        f:close()
        local ok, decoded = pcall(alc.json_decode, body)
        if not ok then
            error(
                string.format(
                    "eval_guardian_player_generalization: ctx.%s_file %q is not valid JSON: %s",
                    prefix,
                    path,
                    tostring(decoded)
                )
            )
        end
        inline = decoded
    end
    if type(inline) ~= "table" or #inline == 0 then
        error(
            string.format(
                "eval_guardian_player_generalization: ctx.%s_file or ctx.%s_moves is required "
                    .. "(non-empty array of logged turns)",
                prefix,
                prefix
            )
        )
    end
    return inline
end

local TRAIN_MOVES = load_moves("train")
local HOLDOUT_MOVES = load_moves("holdout")
local NAME = ctx_field("name") or "sample"
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local STYLE_CHECK = ctx_field("style_check") ~= false

local CORE_FENCE = 0.90
local RAW_LEGAL_FENCE = 0.95

if type(NAME) ~= "string" or not NAME:match("^[a-z0-9_]+$") then
    error("eval_guardian_player_generalization: ctx.name must match ^[a-z0-9_]+$")
end

local ALIAS = "guardian_player_npc_" .. NAME
local CARD_NAME = "guardian-player-playlog-" .. NAME

local function log(msg)
    alc.log("info", "[guardian-player-generalization] " .. msg)
end

-- The sentinel style the sample data was played by. Returns the ruled
-- move and the rule id, or nil for a free-slot position.
local function core_rule(mode, intent)
    if intent == duel.NO_INTENT then
        intent = nil
    end
    if intent == "t" then
        return "b", "R1"
    end
    if intent == "f" or intent == "w" then
        return "b", "R2"
    end
    if intent == "c" or intent == "d" then
        return "A", "R3"
    end
    if intent == "v" then
        return "a", "R4"
    end
    if mode == 1 then
        return "p", "R5"
    end
    return nil, "R6"
end

-- ─── Phase 1: corpus (mirrors bake_guardian_player_from_log.lua) ────

local VOCAB = duel.player_vocab()

local handle = alc.nn.preset.gpt2("tiny", {
    device = "cpu",
    dtype = "f32",
    pretrained = false,
})

local ctx_len = handle:ctx()
local model_vocab = handle:vocab()
if VOCAB.size > model_vocab then
    error(
        string.format(
            "alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            model_vocab
        )
    )
end

local base_rows, train_plays = duel.rows_from_player_moves(TRAIN_MOVES, {
    ctx_len = ctx_len,
    pad_id = VOCAB.pad_id,
})

local replicate_factor = math.max(1, math.ceil(STEPS * BATCH / #base_rows))
local rows = {}
for _ = 1, replicate_factor do
    for _, row in ipairs(base_rows) do
        rows[#rows + 1] = row
    end
end
log(string.format("corpus: %d rows (%d logged moves x %d)", #rows, #base_rows, replicate_factor))

-- ─── Phase 2: training and alias ────────────────────────────────────

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = BATCH,
    ctx_len = ctx_len,
    shuffle = true,
    pad_id = VOCAB.pad_id,
})

log(string.format("full_ft: %d steps, lr=%g, batch=%d", STEPS, LR, BATCH))
local card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
    lr = LR,
    batch = BATCH,
    steps = STEPS,
    warmup = 0,
    schedule = "Constant",
    name = CARD_NAME,
})
if type(card_id) ~= "string" or #card_id == 0 then
    error("eval_guardian_player_generalization: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "guardian_player_npc",
    note = string.format("generalization eval bake (%d moves)", #train_plays),
})

local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("eval_guardian_player_generalization: train_loss missing from the Card")
end

alc.card.append(card_id, {
    persona = {
        source = "play_log",
        moves_count = #train_plays,
    },
})

-- ─── Phase 3: held-out measurement ──────────────────────────────────

local _, holdout_plays = duel.rows_from_player_moves(HOLDOUT_MOVES, {
    ctx_len = ctx_len,
    pad_id = VOCAB.pad_id,
})

local seen = {}
for _, play in ipairs(train_plays) do
    seen[duel.player_encode(play.view)] = true
end

npc.reset_cache()

local function decide(view)
    local answer = npc.run({
        task = alc.json_encode({ mode = "decide", view = view }),
        card_alias = ALIAS,
    }).result
    local action = answer:match("action=(%a)")
    if action == nil then
        error(
            "eval_guardian_player_generalization: NPC answer carried no action: "
                .. tostring(answer)
        )
    end
    return action, answer:match("raw_legal=(%a+)") == "true", answer:match("gated=(%a+)") == "true"
end

local rules = {}
for _, id in ipairs({ "R1", "R2", "R3", "R4", "R5", "R6" }) do
    rules[id] = { n = 0, hit = 0, unseen_n = 0, unseen_hit = 0 }
end
local n, hit = 0, 0
local unseen_n, unseen_hit = 0, 0
local core_n, core_hit = 0, 0
local core_unseen_n, core_unseen_hit = 0, 0
local raw_legal_n, gated_n = 0, 0

for _, play in ipairs(holdout_plays) do
    local ruled, rule_id = core_rule(play.view.mode, play.view.intent)
    local truth = ruled or play.action
    local action, raw_legal, gated = decide(play.view)
    local is_unseen = not seen[duel.player_encode(play.view)]
    local ok = action == truth

    n = n + 1
    if ok then
        hit = hit + 1
    end
    if raw_legal then
        raw_legal_n = raw_legal_n + 1
    end
    if gated then
        gated_n = gated_n + 1
    end
    if is_unseen then
        unseen_n = unseen_n + 1
        if ok then
            unseen_hit = unseen_hit + 1
        end
    end
    if ruled ~= nil then
        core_n = core_n + 1
        if ok then
            core_hit = core_hit + 1
        end
        if is_unseen then
            core_unseen_n = core_unseen_n + 1
            if ok then
                core_unseen_hit = core_unseen_hit + 1
            end
        end
    end
    local r = rules[rule_id]
    r.n = r.n + 1
    if ok then
        r.hit = r.hit + 1
    end
    if is_unseen then
        r.unseen_n = r.unseen_n + 1
        if ok then
            r.unseen_hit = r.unseen_hit + 1
        end
    end
end

local function rate(h, total)
    if total == 0 then
        return -1
    end
    return h / total
end

local per_rule = {}
for _, id in ipairs({ "R1", "R2", "R3", "R4", "R5", "R6" }) do
    local r = rules[id]
    per_rule[id] = string.format("all %d/%d, unseen %d/%d", r.hit, r.n, r.unseen_hit, r.unseen_n)
end

-- ─── Phase 4: fences ────────────────────────────────────────────────

local failures = {}
local core_rate = rate(core_hit, core_n)
if STYLE_CHECK then
    if core_n == 0 then
        failures[#failures + 1] = "core_holdout: no rule positions in the held-out log"
    elseif core_rate < CORE_FENCE then
        failures[#failures + 1] = string.format(
            "core_holdout: %.2f < %.2f (%d/%d)",
            core_rate,
            CORE_FENCE,
            core_hit,
            core_n
        )
    end
end
local raw_legal_rate = rate(raw_legal_n, n)
if raw_legal_rate < RAW_LEGAL_FENCE then
    failures[#failures + 1] = string.format(
        "raw_legal: %.2f < %.2f (%d/%d)",
        raw_legal_rate,
        RAW_LEGAL_FENCE,
        raw_legal_n,
        n
    )
end
if train_loss >= baseline_loss then
    failures[#failures + 1] = string.format(
        "loss_descended: train_loss %.3f did not fall below the %.3f baseline",
        train_loss,
        baseline_loss
    )
end

local pass = #failures == 0
log(
    string.format(
        "verdict: %s | core_holdout=%.2f raw_legal=%.2f holdout=%.2f unseen=%.2f",
        pass and "PASS" or "FAIL",
        core_rate,
        raw_legal_rate,
        rate(hit, n),
        rate(unseen_hit, unseen_n)
    )
)

return {
    pass = pass,
    failures = failures,
    card_id = card_id,
    alias = ALIAS,
    name = NAME,
    style_check = STYLE_CHECK,
    train_moves = #train_plays,
    holdout_moves = n,
    replicate_factor = replicate_factor,
    steps = STEPS,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    holdout_match = rate(hit, n),
    holdout_unseen = unseen_n,
    holdout_unseen_match = rate(unseen_hit, unseen_n),
    core_holdout_match = core_rate,
    core_holdout_n = core_n,
    core_unseen_match = rate(core_unseen_hit, core_unseen_n),
    core_unseen_n = core_unseen_n,
    raw_legal_rate = raw_legal_rate,
    gated_rate = rate(gated_n, n),
    per_rule = per_rule,
}
