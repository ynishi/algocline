-- Bake a guardian duel player NPC out of a play log.
--
-- Self-contained script for `alc_run` (`code_file` form). It is the
-- fourth bake input of the demo and the first that seats the human
-- side: `train_guardian_npc.lua` learns a hand-written boss style,
-- `bake_guardian_persona.lua` learns a boss style an LLM wrote from one
-- line of prose, `bake_card_duel_from_log.lua` learns a card duel log,
-- and this script learns how somebody played *against* the boss. The
-- log is the `move_log` that `guardian_duel_interactive` returns from
-- its `end` action, so a human can sit down, fight the boss a few times
-- and hand the session straight to the trainer.
--
-- Only the player half of each entry is read: the `player` view the
-- board showed and the `player_action` that was chosen from it. The
-- boss half of the same entry is left alone — a boss baked from a
-- transcript would be a copy of the model that produced it.
--
-- There is no teacher function anywhere in this path. A style has one —
-- the corpus can be regrown for any state, and self-play can score the
-- model on states it reaches by playing. A log has none: it is a finite
-- list of positions, and the only thing that can be measured against it
-- is the log itself. That is what `log_match` reports, and it is a
-- training fit rather than a generalisation number: it says how much of
-- the log the model reproduces, not how it plays a position the log
-- never contained. Measuring the latter needs a held-out log or a human
-- verdict, and neither is in this script; the nearest thing the demo
-- has is the `autoplay` mode of `guardian_player_npc`, which puts the
-- baked Card in front of a boss and reports how the fight goes.
--
-- The `D` field of a logged view is the damage the boss the human was
-- fighting still tolerated, so the corpus is measured against that
-- boss's threshold. A Card baked here is therefore only being asked the
-- question it was trained on while it is decoded under the same basis:
-- `guardian_player_npc` takes that basis as `boss_style`, and pointing
-- it at a different boss reads the same digit as something else.
--
-- `alc.nn.data.synthetic` walks its rows once, so a log of a few dozen
-- moves cannot feed `steps * batch` samples on its own. The rows are
-- replicated `ceil(steps * batch / #rows)` times, the same shape
-- `bake_card_duel_from_log.lua` uses. Replication buys steps, not
-- coverage: a short log trains a model that has seen very few distinct
-- positions, however low the loss goes.
--
-- ctx:
--   moves  -- required; array of logged turns, each carrying a `player`
--             view (`{ turn, mode, boss_hp, shift_distance, hp,
--             weakened, exposed, spikes }`) and a `player_action`. This
--             is the `move_log` array of a finished interactive
--             session, unchanged
--   name   -- required; slug matching ^[a-z0-9_]+$. The Card is pinned
--             to the alias `guardian_player_npc_<name>`
--   steps  -- Full FT steps (default 800)
--   lr     -- learning rate (default 3e-3)
--   batch  -- batch size (default 32)
--   seed   -- recorded on the result for bookkeeping (default
--             20260731). Nothing on this path samples: the corpus is
--             the log itself, so the value does not change what is
--             trained. It is accepted so a log bake can be logged next
--             to a style bake with the same run parameters
--
-- `alc.llm` is never called, so the run never pauses: one `alc_run`
-- carries it from the log to the pinned Card.
--
-- Returns a flat table of scalars: `card_id`, `alias`, `log_match`,
-- `log_hits`, `moves_count`, `replicate_factor`, `train_loss` and
-- `loss_descended`, plus the corpus shape. The fit is reported, not
-- asserted: what an acceptable one is depends on the log and on the
-- training budget, and that judgement belongs to the demo runbook.

local duel = require("guardian_duel")
local npc = require("guardian_player_npc")

-- `ctx` is a global injected by alc_run. When the caller passes no ctx
-- the global is a non-table userdata that cannot be indexed, so every
-- read goes through pcall.
local function ctx_field(k)
    local ok, v = pcall(function()
        return ctx and ctx[k]
    end)
    if ok then
        return v
    end
    return nil
end

local MOVES = ctx_field("moves")
local NAME = ctx_field("name")
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)

local VOCAB = duel.player_vocab()

local function log(msg)
    alc.log("info", "[guardian-player-bake-log] " .. msg)
end

-- ─── ctx validation ─────────────────────────────────────────────────
--
-- Only the shape of the request is checked here; every field of every
-- entry is checked by `guardian_duel.rows_from_player_moves`, which
-- names the entry and the reason so a malformed log can be fixed
-- without guessing.

if type(MOVES) ~= "table" or #MOVES == 0 then
    error(
        "bake_guardian_player_from_log: ctx.moves is required and must be a non-empty array "
            .. "of logged turns"
    )
end
if type(NAME) ~= "string" or not NAME:match("^[a-z0-9_]+$") then
    error(
        string.format(
            "bake_guardian_player_from_log: ctx.name must match ^[a-z0-9_]+$, got %s",
            tostring(NAME)
        )
    )
end
if STEPS <= 0 or BATCH <= 0 then
    error("bake_guardian_player_from_log: ctx.steps and ctx.batch must be positive integers")
end

local ALIAS = "guardian_player_npc_" .. NAME
local CARD_NAME = "guardian-player-playlog-" .. NAME

-- ─── Phase 1: corpus from the log ───────────────────────────────────

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
            "bake_guardian_player_from_log: alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            model_vocab
        )
    )
end

local base_rows, plays = duel.rows_from_player_moves(MOVES, {
    ctx_len = ctx_len,
    pad_id = VOCAB.pad_id,
})

-- The trainer asks the dataset for `steps * batch` samples and the
-- dataset does not wrap, so the log is repeated until it can answer.
local replicate_factor = math.max(1, math.ceil(STEPS * BATCH / #base_rows))
local rows = {}
for _ = 1, replicate_factor do
    for _, row in ipairs(base_rows) do
        rows[#rows + 1] = row
    end
end
log(
    string.format(
        "corpus: %d rows x %d tokens (%d logged moves x %d)",
        #rows,
        ctx_len,
        #base_rows,
        replicate_factor
    )
)

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
    error("bake_guardian_player_from_log: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "guardian_player_npc",
    note = string.format("baked from play log (%d moves)", #plays),
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("bake_guardian_player_from_log: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Phase 3: provenance record ─────────────────────────────────────
--
-- A persona Card records the prompt and the policy chunk it was grown
-- from. A log Card has neither, and the log itself is the caller's
-- document, so what is kept is where the Card came from and how much of
-- a log stands behind it — the two things that cannot be read back out
-- of the weights.

local appended = alc.card.append(card_id, {
    persona = {
        source = "play_log",
        moves_count = #plays,
    },
})
if type(appended) ~= "table" then
    error("bake_guardian_player_from_log: alc.card.append did not return the merged Card")
end

-- ─── Phase 4: log replay ────────────────────────────────────────────
--
-- Every logged view is decoded through the freshly pinned Card and
-- compared with the move the human played there. These are the states
-- the model was trained on, so this is a fit, not a compliance rate;
-- the field is named `log_match` rather than `style_match` because
-- there is no style function it could be compared against.

npc.reset_cache()

local hits = 0
for i, play in ipairs(plays) do
    local answer = npc.run({
        task = alc.json_encode({ mode = "decide", view = play.view }),
        card_alias = ALIAS,
    }).result
    local action = answer:match("action=(%a)")
    if action == nil then
        error(
            string.format(
                "bake_guardian_player_from_log: NPC answer for move %d carried no action: %s",
                i,
                tostring(answer)
            )
        )
    end
    if action == play.action then
        hits = hits + 1
    end
end

local log_match = hits / #plays
log(string.format("log_match=%.2f (%d/%d)", log_match, hits, #plays))

return {
    card_id = card_id,
    alias = ALIAS,
    name = NAME,
    moves_count = #plays,
    replicate_factor = replicate_factor,
    rows = #rows,
    ctx_len = ctx_len,
    steps = STEPS,
    seed = SEED,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    log_match = log_match,
    log_hits = hits,
}
