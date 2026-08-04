-- Bake a boss between two teacher styles: mixed corpus -> Full FT -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). Where
-- `train_guardian_npc.lua` learns one of the hand-written styles in
-- `guardian_duel.STYLES` and `bake_guardian_persona.lua` learns a boss
-- an LLM wrote, this script learns the boss *between* two of the shipped
-- styles: the corpus is labelled by `guardian_mix.mixed_policy`, which
-- answers with the first parent on a share `beta` of its decisions and
-- with the second on the rest. Everything downstream of the labelling
-- function — the corpus builder, the Full FT call, the Card, the alias —
-- is the same path the single-style trainer walks.
--
-- No `alc.llm` call happens anywhere in this path, so the run never
-- pauses for a host response.
--
-- ctx:
--   beta        -- required; share of the decisions answered by
--                  `styles[1]`, in the open interval (0, 1). `0` and `1`
--                  are the parents themselves and are trained with
--                  `train_guardian_npc.lua`
--   styles      -- two different `guardian_duel.STYLES` names (default
--                  `{"rusher", "turtle"}`, the pair whose cycles differ
--                  at every index)
--   games       -- floor on the teacher playouts (default 300). More
--                  fights are played on top of that floor until the
--                  corpus actually covers the training run, because
--                  `alc.nn.data.synthetic` walks its rows once and
--                  errors out when the trainer asks for more; how many
--                  fights that takes is measured rather than predicted
--   steps       -- Full FT steps (default 800, the budget the parents
--                  were accepted under)
--   lr          -- learning rate (default 3e-3)
--   batch       -- batch size (default 32)
--   seed        -- base seed for the fights and for the mixing stream
--                  (default 20260731)
--   alias       -- Card alias to pin (default
--                  `guardian_duel_npc_mix_<initials>_b<beta*100>`, e.g.
--                  `guardian_duel_npc_mix_rt_b25`)
--   name        -- Card name (default "guardian-duel-npc-mix")
--   check_games -- self-play fights behind each compliance line
--                  (default 20)
--
-- The distance basis is fixed to "guardian" and is deliberately not a
-- ctx field. `guardian_duel.encode` measures its `D` character against
-- one style's mode-shift threshold, so a mixed corpus has to borrow one
-- of them; on the `guardian` and `rusher` thresholds the fibre of that
-- projection splits the mixture cleanly, while on the `turtle`
-- threshold a small share of the lines pairs two different distances
-- onto one character and the mixture measured off such a Card would be
-- reading the basis rather than the boss. Opening the basis would
-- therefore offer a setting whose answer is known to be wrong for one of
-- its values; a run that wants another basis is a rules question, not a
-- ctx one, so passing `ctx.basis_style` is refused loudly below.
--
-- Compliance is reported through a pair of self-play runs rather than
-- one. A mixed Card has no single teacher to be scored against — the
-- policy that labelled its corpus is not deterministic and so cannot be
-- handed to `guardian_duel_npc` as a `task.policy_source` (see
-- `guardian_mix`) — so the Card is played twice from the same decode
-- basis, once with `task.style` naming each parent. The two style-match
-- numbers say how much of each parent the Card answers, and their sum
-- says how much of its play either parent accounts for. Both are
-- reported, neither is asserted: the fidelity of the mixture is measured
-- off the decode distribution in the acceptance driver, and a greedy
-- self-play run cannot see a weight it argmaxes away.
--
-- Returns a flat table of scalars so a smoke harness can assert on it
-- directly.

local duel = require("guardian_duel")
local npc = require("guardian_duel_npc")
local mix = require("guardian_mix")

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

local RAW_BETA = ctx_field("beta")
local STYLES = ctx_field("styles") or { "rusher", "turtle" }
local GAMES = math.floor(tonumber(ctx_field("games")) or 300)
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)
local ALIAS_OVERRIDE = ctx_field("alias")
local NAME = ctx_field("name") or "guardian-duel-npc-mix"
local CHECK_GAMES = math.floor(tonumber(ctx_field("check_games")) or 20)

--- Distance basis every line of this corpus is encoded against, and the
--- basis the Card is decoded under for the rest of its life. Fixed on
--- purpose; see the header.
local BASIS_STYLE = "guardian"

local VOCAB = duel.vocab()

local function log(msg)
    alc.log("info", "[guardian-mix] " .. msg)
end

-- ─── ctx validation ─────────────────────────────────────────────────
--
-- Everything that can be decided from the ctx and the rules alone is
-- decided here, before a single playout or training step is paid for.

if RAW_BETA == nil then
    error(
        "train_guardian_mix: ctx.beta is required (the share of the decisions the first style "
            .. "answers, in the open interval (0, 1))"
    )
end
local BETA = tonumber(RAW_BETA)
if BETA == nil then
    error("train_guardian_mix: ctx.beta must be a number, got " .. tostring(RAW_BETA))
end

if ctx_field("basis_style") ~= nil or ctx_field("basis") ~= nil then
    error(
        "train_guardian_mix: the distance basis is fixed to "
            .. string.format("%q", BASIS_STYLE)
            .. " and is not a ctx field; the mixture is only cleanly defined on a basis whose "
            .. "encoding does not fold two distances onto one character"
    )
end

if CHECK_GAMES <= 0 then
    error("train_guardian_mix: ctx.check_games must be a positive integer")
end

-- The parent pair, the weight and the seed are validated by the policy
-- factory rather than re-checked here, so there is one account of what a
-- legal mixture is. Building it first also means a bad ctx fails before
-- the first fight.
local policy = mix.mixed_policy(STYLES, BETA, SEED)

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
                "train_guardian_mix: ctx.beta %s does not land on a whole percent, so the default "
                    .. "alias would name it the same as %g; pass ctx.alias explicitly",
                tostring(beta),
                whole / 100
            )
        )
    end
    return string.format(
        "guardian_duel_npc_mix_%s%s_b%d",
        styles[1]:sub(1, 1),
        styles[2]:sub(1, 1),
        whole
    )
end

local ALIAS = ALIAS_OVERRIDE or default_alias(STYLES, BETA)
if type(ALIAS) ~= "string" or #ALIAS == 0 then
    error("train_guardian_mix: ctx.alias must be a non-empty string")
end

-- ─── Corpus ─────────────────────────────────────────────────────────
--
-- A fight lasts at most `MAX_ROWS_PER_GAME` turns but ends the moment
-- either side drops, so the rows one playout contributes are a property
-- of the labelling policy rather than of the rules. A mixture has no
-- fixed yield at all — it sits between two styles that end their fights
-- at different speeds — so the budget is measured instead of assumed:
-- each round is sized from the rows the previous rounds actually
-- produced, and the loop stops once the corpus covers the run.
--
-- `build_corpus` deals its fights from `seed + 1` upwards, so a round
-- opens where the previous one stopped and no two rounds replay a fight.
-- The mixing policy is built once, above, so its draws run as one stream
-- across every round: the whole corpus stays a function of `ctx.seed`,
-- `ctx.beta` and the parent pair.

--- Extra rows beyond `steps * batch`, so a corpus that lands exactly on
--- the boundary still has a batch in hand.
local CORPUS_SLACK_BATCHES = 1

--- Fights a round adds on top of the measured requirement, so the last
--- rounds do not shrink to one fight at a time.
local ROUND_PAD_GAMES = 16

--- Ceiling on the playouts one corpus may cost, as a multiple of the
--- most optimistic estimate. Reaching it means the fights are ending far
--- sooner than the rules allow, which is a rules or policy fault rather
--- than a budget to raise silently.
local CORPUS_GAMES_CAP_FACTOR = 3

---@param labelling fun(state: table): string Mixed labelling policy
---@param ctx_len integer Model context window
---@param target integer Rows the training run consumes
---@return table rows, integer played, integer rounds
local function build_full_corpus(labelling, ctx_len, target)
    local optimistic = math.ceil(target / duel.MAX_ROWS_PER_GAME)
    local cap = math.max(GAMES, optimistic) * CORPUS_GAMES_CAP_FACTOR
    local rows, played, rounds = {}, 0, 0
    while #rows < target do
        if played >= cap then
            error(
                string.format(
                    "train_guardian_mix: %d playouts of the %s/%s mixture produced %d rows but "
                        .. "the run needs %d; the fights are ending far earlier than the turn "
                        .. "limit allows",
                    played,
                    STYLES[1],
                    STYLES[2],
                    #rows,
                    target
                )
            )
        end
        local want
        if played == 0 then
            want = math.max(GAMES, optimistic)
        else
            -- Rows per fight measured on this mixture, on this seed.
            want = math.ceil((target - #rows) * played / #rows) + ROUND_PAD_GAMES
        end
        if want > cap - played then
            want = cap - played
        end
        local chunk = duel.build_corpus(labelling, {
            ctx_len = ctx_len,
            games = want,
            style = BASIS_STYLE,
            seed = SEED + played,
            pad_id = VOCAB.pad_id,
        })
        for _, row in ipairs(chunk) do
            rows[#rows + 1] = row
        end
        played = played + want
        rounds = rounds + 1
    end
    return rows, played, rounds
end

-- ─── Train ──────────────────────────────────────────────────────────

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
            "train_guardian_mix: alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            model_vocab
        )
    )
end

local corpus_target = STEPS * BATCH + CORPUS_SLACK_BATCHES * BATCH
local rows, playouts, corpus_rounds = build_full_corpus(policy, ctx_len, corpus_target)
log(
    string.format(
        "corpus: %d rows x %d tokens from %d playouts in %d round(s), target %d, mix %s/%s beta=%g "
            .. "on the %s basis",
        #rows,
        ctx_len,
        playouts,
        corpus_rounds,
        corpus_target,
        STYLES[1],
        STYLES[2],
        BETA,
        BASIS_STYLE
    )
)

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
    name = NAME,
})
if type(card_id) ~= "string" or #card_id == 0 then
    error("train_guardian_mix: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "guardian_duel_npc",
    note = string.format(
        "guardian duel %s/%s mixture, beta=%g, %s basis",
        STYLES[1],
        STYLES[2],
        BETA,
        BASIS_STYLE
    ),
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("train_guardian_mix: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Compliance: one self-play run per parent ───────────────────────
--
-- Both runs decode the Card under `BASIS_STYLE`, the basis its corpus
-- was encoded on; only `task.style` moves, and it moves the teacher the
-- moves are scored against rather than the encoding they are chosen
-- from. Scoring a mixture against one parent at a time is the closest a
-- greedy self-play run gets to the mixture, and it is reported rather
-- than gated for exactly that reason.

npc.reset_cache()

---@param teacher string Parent the moves are scored against
---@return number style_match, string line
local function selfplay_against(teacher)
    local line = npc.run({
        task = alc.json_encode({
            mode = "selfplay",
            games = CHECK_GAMES,
            seed = SEED,
            style = teacher,
        }),
        card_alias = ALIAS,
        style = BASIS_STYLE,
    }).result
    log(string.format("selfplay vs %s -> %s", teacher, line))
    local style_match = tonumber(line:match("style_match=([%d%.]+)"))
    if style_match == nil then
        error(
            string.format(
                "train_guardian_mix: self-play against %s carried no style_match: %s",
                teacher,
                line
            )
        )
    end
    return style_match, line
end

local match_a, selfplay_a = selfplay_against(STYLES[1])
local match_b, selfplay_b = selfplay_against(STYLES[2])

return {
    -- `ok` reads the training result alone: the two compliance numbers
    -- are data about the mixture, and no threshold on them has been
    -- measured yet.
    ok = train_loss < baseline_loss,
    card_id = card_id,
    alias = ALIAS,
    beta = BETA,
    style_a = STYLES[1],
    style_b = STYLES[2],
    basis_style = BASIS_STYLE,
    -- The fights actually played and the rows they actually produced,
    -- not the estimate that opened the loop.
    games = playouts,
    steps = STEPS,
    rows = #rows,
    rows_target = corpus_target,
    corpus_rounds = corpus_rounds,
    ctx_len = ctx_len,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    check_games = CHECK_GAMES,
    match_a = match_a,
    match_b = match_b,
    match_sum = match_a + match_b,
    selfplay_a = selfplay_a,
    selfplay_b = selfplay_b,
}
