-- Train the card duel NPC: teacher self-play corpus -> Full FT -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). It generates
-- its own supervised corpus from `card_duel.policy_aggressive` (the
-- teacher style), tunes a from-scratch `gpt2 tiny` on it, registers the
-- resulting Card, pins the alias `card_duel_npc` reads, and finishes
-- with a decode check through the NPC package itself.
--
-- No `alc.llm` call happens anywhere in this path, so the run never
-- pauses for a host response.
--
-- ctx (all optional, JSON object passed to alc_run):
--   games      -- floor on the teacher playouts to generate (default 300).
--                 The corpus is always grown to at least `steps * batch`
--                 rows, because `alc.nn.data.synthetic` walks its rows
--                 once and errors out when the trainer asks for more.
--   steps      -- Full FT steps (default 800)
--   lr         -- learning rate (default 3e-3)
--   batch      -- batch size (default 32)
--   seed       -- base seed for the deals (default 20260731)
--   alias      -- Card alias to pin (default "card_duel_npc")
--   name       -- Card name (default "card-duel-npc")
--
-- Returns a flat table of scalars so a Rust smoke harness can assert on
-- it directly; see `crates/algocline-engine/tests/gameai_smoke_test.rs`.

local duel = require("card_duel")

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

local GAMES = math.floor(tonumber(ctx_field("games")) or 300)
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)
local ALIAS = ctx_field("alias") or "card_duel_npc"
local NAME = ctx_field("name") or "card-duel-npc"

local VOCAB = duel.vocab()

local function log(msg)
    alc.log("info", "[card-duel-train] " .. msg)
end

-- ─── Corpus ─────────────────────────────────────────────────────────
--
-- One training line is `<encoded state>><teacher action>\n`, padded to
-- the model context window. Both seats contribute a line every round:
-- player one actually plays the teacher move, player two plays a random
-- move but its state is still labelled with the teacher action, which
-- widens the state coverage without changing the target function.
--
-- A synthetic dataset walks its rows once, so the playout count is
-- raised until the corpus covers `steps * batch` rows; `ctx.games` acts
-- as a floor rather than the exact count.

local ROWS_PER_GAME = 2 * duel.HAND_SIZE

local function make_row(state, action, ctx_len)
    local ids = duel.to_ids(duel.encode(state) .. ">" .. tostring(action) .. "\n")
    if #ids > ctx_len then
        error(
            string.format(
                "train_card_duel_npc: encoded line needs %d tokens but the model context is %d",
                #ids,
                ctx_len
            )
        )
    end
    for _ = #ids + 1, ctx_len do
        ids[#ids + 1] = VOCAB.pad_id
    end
    return ids
end

local function build_corpus(ctx_len, playouts)
    local rows = {}
    for i = 1, playouts do
        local g = duel.new_game(SEED + i)
        local rng = alc.math.rng_create(SEED * 7919 + i)
        while not duel.is_over(g) do
            local a1 = duel.policy_aggressive(g.p1)
            rows[#rows + 1] = make_row(g.p1, a1, ctx_len)
            rows[#rows + 1] = make_row(g.p2, duel.policy_aggressive(g.p2), ctx_len)
            g = duel.apply(g, a1, duel.policy_random(g.p2, rng))
        end
    end
    return rows
end

-- ─── Train ──────────────────────────────────────────────────────────

local handle = alc.nn.preset.gpt2("tiny", {
    device = "cpu",
    dtype = "f32",
    pretrained = false,
})

local CTX_LEN = handle:ctx()
local MODEL_VOCAB = handle:vocab()
if VOCAB.size > MODEL_VOCAB then
    error(
        string.format(
            "train_card_duel_npc: alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            MODEL_VOCAB
        )
    )
end

local PLAYOUTS = math.max(GAMES, math.ceil(STEPS * BATCH / ROWS_PER_GAME))
local rows = build_corpus(CTX_LEN, PLAYOUTS)
log(string.format("corpus: %d rows x %d tokens from %d playouts", #rows, CTX_LEN, PLAYOUTS))

local dataset = alc.nn.data.synthetic(rows, {
    batch_size = BATCH,
    ctx_len = CTX_LEN,
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
    name = NAME,
})
if type(card_id) ~= "string" or #card_id == 0 then
    error("train_card_duel_npc: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "card_duel_npc",
    note = "card duel teacher-style NPC",
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(MODEL_VOCAB)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("train_card_duel_npc: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Quick check through the NPC package ────────────────────────────
--
-- Three fixed states covering the two branches of the teacher style,
-- decoded through the same entry point the eval scenario drives.

local CHECK_STATES = {
    { round = 1, my_hand = { 9, 7, 5, 3, 1 }, my_points = 0, opp_points = 0, opp_played = {} },
    { round = 2, my_hand = { 1, 3, 5, 7 }, my_points = 1, opp_points = 0, opp_played = { 4 } },
    { round = 3, my_hand = { 3, 5, 8 }, my_points = 1, opp_points = 1, opp_played = { 2, 6 } },
}

local npc = require("card_duel_npc")
npc.reset_cache()

local function ask(payload)
    local out = npc.run({ task = alc.json_encode(payload), card_alias = ALIAS })
    return out.result
end

local decide_legal = true
local style_hits = 0
local reports = {}
for i, state in ipairs(CHECK_STATES) do
    local text = ask({ mode = "decide", state = state })
    reports[#reports + 1] = text
    log(string.format("decide[%d] %s -> %s", i, duel.encode(state), text))

    local action = tonumber(text:match("action=(%d)"))
    local legal = false
    for _, rank in ipairs(duel.legal_actions(state)) do
        if rank == action then
            legal = true
        end
    end
    if not legal then
        decide_legal = false
    end
    if action == duel.policy_aggressive(state) then
        style_hits = style_hits + 1
    end
end

local determinism_text = ask({ mode = "determinism", state = CHECK_STATES[1] })
log("determinism -> " .. determinism_text)

return {
    ok = decide_legal and train_loss < baseline_loss,
    card_id = card_id,
    alias = ALIAS,
    games = PLAYOUTS,
    steps = STEPS,
    rows = #rows,
    ctx_len = CTX_LEN,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    decide_legal = decide_legal,
    style_hits = style_hits,
    style_total = #CHECK_STATES,
    deterministic = determinism_text:find("deterministic=true", 1, true) ~= nil,
    decisions = table.concat(reports, " | "),
}
