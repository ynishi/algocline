-- Bake a guardian duel persona: NL prompt -> synthesised boss -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). Where
-- `train_guardian_npc.lua` learns one of the hand-written styles in
-- `guardian_duel.STYLES`, this script takes a one-line description of a
-- boss in plain language, asks the host LLM to write the matching Lua
-- policy, validates that policy, and then reuses exactly the same
-- corpus / Full FT / Card path. The teacher is synthesised; everything
-- downstream of it is unchanged.
--
-- The LLM answer is never loaded raw. `guardian_duel.compile_policy`
-- compiles it in a restricted environment (`math` / `table` / `ipairs`
-- / `pairs`, no `load`, no `os`, no `io`) and makes it answer a batch
-- of sampled states legally and deterministically first. A rejected
-- candidate is re-synthesised with the rejection message attached, up
-- to `MAX_RETRIES` times, after which the run fails loudly.
--
-- ctx:
--   prompt      -- required; one line of plain language describing the
--                  boss to bake
--   name        -- required; slug matching ^[a-z0-9_]+$. The Card is
--                  pinned to the alias `guardian_duel_npc_<name>`
--   basis_style -- distance basis the states are encoded against
--                  (default "guardian"); one of `guardian_duel.STYLES`
--   games       -- floor on the teacher playouts (default 300). More
--                  fights are played on top of that floor until the
--                  corpus actually covers the training run, because
--                  `alc.nn.data.synthetic` walks its rows once and
--                  errors out when the trainer asks for more. How many
--                  fights that takes is measured rather than predicted;
--                  see `build_full_corpus`
--   steps       -- Full FT steps (default 800)
--   lr          -- learning rate (default 3e-3)
--   batch       -- batch size (default 32)
--   seed        -- base seed for the fights (default 20260731)
--   check_games -- self-play fights used for the compliance report
--                  (default 20)
--
-- `basis_style` is the one field this script has that the card duel
-- version does not. The `D` character of an encoded state is the
-- distance left to *a* style's mode shift, so a corpus cannot be built
-- without naming whose threshold it is measured against. A persona
-- boss has no threshold of its own in the rules, so it borrows one and
-- is decoded under the same borrowed basis for the rest of its life;
-- the basis is written onto the Card for that reason. A persona that
-- staggers on a rule of its own needs a basis of its own, which is a
-- rules change rather than a prompt.
--
-- The run pauses once, at Phase 1, on the `alc.llm` call carrying the
-- synthesis prompt; the host answers and resumes via `alc_continue`.
-- Nothing after that needs a host round trip.
--
-- Returns a flat table of scalars: `card_id`, `alias`, `basis_style`,
-- `style_match`, `style_hits`, `train_loss` and `retries`, plus the
-- corpus shape. The compliance number is reported, not asserted: the
-- fence lives in the eval scenario and in the demo runbook, so a low
-- score here is data rather than a failed run.

local duel = require("guardian_duel")
local npc = require("guardian_duel_npc")

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

local PROMPT = ctx_field("prompt")
local NAME = ctx_field("name")
local BASIS_STYLE = ctx_field("basis_style") or "guardian"
local GAMES = math.floor(tonumber(ctx_field("games")) or 300)
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)
local CHECK_GAMES = math.floor(tonumber(ctx_field("check_games")) or 20)

--- Synthesis attempts beyond the first one.
local MAX_RETRIES = 3

--- Playouts behind the states a candidate policy is validated against.
--- Up to nine states per fight, so twenty fights is a batch of well
--- over a hundred.
local VALIDATION_GAMES = 20

--- Longest prompt excerpt kept on the alias note.
local NOTE_LEN = 80

local VOCAB = duel.vocab()

local function log(msg)
    alc.log("info", "[guardian-bake] " .. msg)
end

-- ─── ctx validation ─────────────────────────────────────────────────

if type(PROMPT) ~= "string" or #PROMPT == 0 then
    error("bake_guardian_persona: ctx.prompt is required and must be a non-empty string")
end
if type(NAME) ~= "string" or not NAME:match("^[a-z0-9_]+$") then
    error(
        string.format(
            "bake_guardian_persona: ctx.name must match ^[a-z0-9_]+$, got %s",
            tostring(NAME)
        )
    )
end
if CHECK_GAMES <= 0 then
    error("bake_guardian_persona: ctx.check_games must be a positive integer")
end

-- An unknown basis would be caught by `build_corpus` several phases
-- later, after the host has already been asked for a policy.
do
    local known = false
    for _, name in ipairs(duel.STYLES) do
        known = known or name == BASIS_STYLE
    end
    if not known then
        error(
            string.format(
                "bake_guardian_persona: ctx.basis_style must be one of %s, got %s",
                table.concat(duel.STYLES, ", "),
                tostring(BASIS_STYLE)
            )
        )
    end
end

local ALIAS = "guardian_duel_npc_" .. NAME
local CARD_NAME = "guardian-duel-persona-" .. NAME

-- ─── Phase 1 / 2: policy synthesis and validation ───────────────────
--
-- The contract below is the whole interface the LLM is given: the
-- rules, the state fields, the legality requirement, the purity
-- requirement, the exact set of globals its chunk will be able to
-- reach, and — unique to this game — what the trained model will
-- actually be able to see of a state. The rules half is written out
-- literally so the prompt stays readable in the run log; the numbers of
-- the distance basis are interpolated, because they are the one part
-- that changes with `ctx.basis_style`.

local CONTRACT = string.format(
    [[
You are writing one Lua function that plays the boss of a nine-turn duel.

Rules of the fight:
- One boss against one player, at most nine turns. Both sides start at 45
  health. The player acts first each turn, then the boss answers.
- The player can attack lightly (4 damage), attack heavily (9 damage, but
  the boss hits three harder on the following turn), block (absorbs 6 of
  the boss answer that same turn) or poke (2 damage).
- Your six moves are:
    "c" charge      -- no damage; absorbs 5 of the player's next attack
    "f" fierce      -- 9 damage
    "v" vent        -- 5 damage, and weakens the player's next attack by 2
    "w" whirlwind   -- 6 damage
    "d" defensive   -- no damage; raises spikes that deal 3 back on every
                       player attack, and rolls the boss up (mode becomes 1)
    "t" twin slam   -- 10 damage; spends the spikes, clears the damage
                       counter and rolls the boss back down (mode becomes 0)
- "t" is legal only while state.mode is 1. The other five are always legal.
  Answering "t" in mode 0 is rejected.
- The fight ends when a side reaches zero health or after nine turns, in
  which case the higher health wins.

The function takes one argument, a table describing the boss:
- state.cycle              integer 0 to 3, how far along its own sequence
                           the boss stands
- state.mode               0 while it walks that sequence, 1 while it is
                           rolled up behind its spikes
- state.hp                 integer 0 to 45, the boss health
- state.damage_since_shift integer, damage taken since the last time the
                           boss rolled back down
- state.last_player        0 before the first player move, then 1 light,
                           2 heavy, 3 block, 4 poke
- state.turn               integer 1 to 9
- state.shifts             integer, how many times the boss has already
                           rolled up and slammed back down

What the model will see of that state:

The function you write is the teacher. It labels a training corpus, and
every line of that corpus shows the model twelve characters:

    C<cycle> M<mode> H<health bucket> D<distance bucket> L<last player> T<turn>

with

    threshold = %d + %d * state.shifts
    distance  = threshold - state.damage_since_shift   (never below zero)
    D         = 0 when distance is 0, otherwise ceil(distance / %d), capped at 9
    H         = 0 when hp is 0, otherwise ceil(hp / %d)

Two states with the same twelve characters must therefore carry the same
answer, or the model is being taught both sides of a coin flip. Decide from
the six projected quantities above — cycle, mode, H, D, last_player, turn —
rather than from raw hp, raw damage_since_shift or shifts on their own.
D is the distance to a stagger: D0 means "the damage has piled up, roll up
now" and a larger D means "there is room left".

Hard requirements:
- Answer with the chunk only: `return function(state) ... end`. No prose, no
  markdown fence, no comments outside the chunk.
- The function must return exactly one of the strings "c", "f", "v", "w",
  "d", "t", and only "t" while state.mode is 1.
- The function must be pure and deterministic. The same state must always
  produce the same move. Never call math.random.
- Never modify the state table.
- Only these globals exist inside the chunk: math, table, ipairs, pairs.
  There is no string, no os, no io, no print, no require, no load, no
  setmetatable, no select, no type, no tostring.
- Read the state to decide. A function that ignores the state and returns a
  constant cannot express a boss.
]],
    duel.threshold_damage(BASIS_STYLE, 0),
    duel.DAMAGE_BUCKET_SIZE,
    duel.DAMAGE_BUCKET_SIZE,
    duel.HP_BUCKET_SIZE
)

--- Build the synthesis prompt, optionally carrying a rejection.
---
--- The rejection message from `guardian_duel.compile_policy` names the
--- state and the reason, which is the only feedback the next attempt
--- gets; paraphrasing it would drop the state that failed.
local function synthesis_prompt(rejection)
    local parts = {
        CONTRACT,
        "",
        "Boss to implement, in the author's words:",
        PROMPT,
        "",
        "Write the chunk now.",
    }
    if rejection ~= nil then
        parts[#parts + 1] = ""
        parts[#parts + 1] = "The previous attempt was rejected by the validator:"
        parts[#parts + 1] = rejection
        parts[#parts + 1] = "Fix that and answer with the corrected chunk only."
    end
    return table.concat(parts, "\n")
end

--- Pull the Lua chunk out of an LLM answer.
---
--- A fenced block is unwrapped, and prose in front of the chunk is
--- dropped by cutting at the first `return function`. Anything left
--- that does not compile is caught by `compile_policy` and fed back
--- into the next attempt, so this step never has to guess.
local function extract_chunk(response)
    local text = tostring(response or "")
    local fenced = text:match("```lua%s*(.-)```") or text:match("```%s*(.-)```")
    if fenced ~= nil then
        text = fenced
    end
    text = text:gsub("^%s+", ""):gsub("%s+$", "")
    if not text:match("^return%s") then
        local cut = text:match("(return%s+function.*)")
        if cut ~= nil then
            text = cut
        end
    end
    return text
end

local validation_states = duel.sample_states({ games = VALIDATION_GAMES, seed = SEED })
log(
    string.format(
        "synthesising a boss for %q on the %s distance basis against %d validation states",
        NAME,
        BASIS_STYLE,
        #validation_states
    )
)

local policy, policy_source, retries
local rejection = nil
for attempt = 1, MAX_RETRIES + 1 do
    local answer = alc.llm(synthesis_prompt(rejection), {
        system = "You write small, pure Lua functions and answer with code only.",
        max_tokens = 800,
    })
    local candidate = extract_chunk(answer)
    if #candidate == 0 then
        rejection = "the answer carried no Lua chunk"
    else
        local ok, result = pcall(duel.compile_policy, candidate, {
            states = validation_states,
            chunk_name = "persona_policy",
        })
        if ok then
            policy = result
            policy_source = candidate
            retries = attempt - 1
            log(string.format("policy accepted on attempt %d", attempt))
            break
        end
        rejection = tostring(result)
    end
    log(string.format("attempt %d rejected: %s", attempt, rejection))
end

if policy == nil then
    error(
        string.format(
            "bake_guardian_persona: no valid policy after %d attempts; last rejection: %s",
            MAX_RETRIES + 1,
            tostring(rejection)
        )
    )
end

-- ─── Phase 3: corpus, training, alias ───────────────────────────────
--
-- Identical to `train_guardian_npc.lua` from here on: the synthesised
-- policy is just another labelling function, so it goes through the
-- shared `guardian_duel.build_corpus` and the same Full FT call. The
-- only difference is that the style handed to `build_corpus` is the
-- borrowed distance basis rather than the policy's own name.
--
-- A fight lasts at most `MAX_ROWS_PER_GAME` turns but ends the moment
-- either side drops, so the rows one playout contributes depend on how
-- the policy plays: the teacher styles measure between 6.4 and 9.0 rows
-- per fight, and a synthesised boss is under no obligation to stay in
-- that range. Sizing the batch up front from the turn limit therefore
-- under-counts, and the trainer runs out of data mid-run ("dataset
-- exhausted after 797/1000 steps") — here after the host round trip of
-- Phase 1 has already been spent, which is the expensive place to fail.
--
-- The playouts are generated in rounds instead, each round sized from
-- the rows the previous rounds actually produced, until the corpus
-- covers the run. Each round carries its own seed, so the corpus stays a
-- function of `ctx.seed` and the accepted chunk alone.

--- Extra rows beyond `steps * batch`, so a corpus that lands exactly on
--- the boundary still has a batch in hand.
local CORPUS_SLACK_BATCHES = 1

--- Fights a round adds on top of the measured requirement, so the last
--- rounds do not shrink to one fight at a time.
local ROUND_PAD_GAMES = 16

--- Ceiling on the playouts one corpus may cost, as a multiple of the
--- most optimistic estimate. A synthesised boss that ends every fight in
--- two turns hits it, and that is a fact about the persona worth
--- failing on rather than a budget to raise silently.
local CORPUS_GAMES_CAP_FACTOR = 3

--- Build a corpus that covers `target` rows, and report what it cost.
---
--- `ctx.games` is honoured as a floor on the first round, which keeps
--- its original meaning: a caller asking for 300 fights still gets at
--- least 300.
---
--- `build_corpus` deals its fights from `seed + 1` upwards, so a round
--- opens where the previous one stopped. No two rounds replay a fight,
--- whatever size they end up being, and the whole corpus is still a
--- function of `ctx.seed` and the accepted chunk.
---@param labelling fun(state: table): string Accepted persona policy
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
                    "bake_guardian_persona: %d playouts of the synthesised boss produced %d rows "
                        .. "but the run needs %d; its fights end far earlier than the turn limit "
                        .. "allows",
                    played,
                    #rows,
                    target
                )
            )
        end
        local want
        if played == 0 then
            want = math.max(GAMES, optimistic)
        else
            -- Rows per fight measured on this policy, on this seed.
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
            "bake_guardian_persona: alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            model_vocab
        )
    )
end

local corpus_target = STEPS * BATCH + CORPUS_SLACK_BATCHES * BATCH
local rows, playouts, corpus_rounds = build_full_corpus(policy, ctx_len, corpus_target)
log(
    string.format(
        "corpus: %d rows x %d tokens from %d playouts in %d round(s), target %d",
        #rows,
        ctx_len,
        playouts,
        corpus_rounds,
        corpus_target
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
    name = CARD_NAME,
})
if type(card_id) ~= "string" or #card_id == 0 then
    error("bake_guardian_persona: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "guardian_duel_npc",
    note = PROMPT:sub(1, NOTE_LEN),
})
log(string.format("card %s pinned to alias %q", card_id, ALIAS))

-- Uniform-random baseline over the model vocabulary. A final loss below
-- it is the evidence that gradients flowed at all.
local baseline_loss = math.log(model_vocab)
local card = alc.card.get(card_id)
local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
local train_loss = metrics and metrics.train_loss
if type(train_loss) ~= "number" then
    error("bake_guardian_persona: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Phase 4: persona record ────────────────────────────────────────
--
-- The prompt, the chunk it produced and the distance basis it was
-- trained under are the three things that cannot be recovered from the
-- weights. Decoding this Card under another basis would feed it a `D`
-- its corpus never carried, so the basis travels with the Card. The
-- append surface is additive only, which is why `persona` is a fresh
-- top-level key rather than an edit of `metadata`.

local appended = alc.card.append(card_id, {
    persona = {
        prompt = PROMPT,
        policy_source = policy_source,
        basis_style = BASIS_STYLE,
    },
})
if type(appended) ~= "table" then
    error("bake_guardian_persona: alc.card.append did not return the merged Card")
end

-- ─── Phase 5: inline compliance report ──────────────────────────────
--
-- Self-play through the NPC package itself, scored against the very
-- policy the corpus was labelled with and decoded under the basis it
-- was labelled on. The number is reported, not fenced: what an
-- acceptable compliance rate is depends on the training budget, and that
-- judgement belongs to the scenario.

npc.reset_cache()

local selfplay = npc.run({
    task = alc.json_encode({
        mode = "selfplay",
        games = CHECK_GAMES,
        seed = SEED,
        policy_source = policy_source,
    }),
    card_alias = ALIAS,
    style = BASIS_STYLE,
}).result
log(string.format("selfplay -> %s", selfplay))

local style_match = tonumber(selfplay:match("style_match=([%d%.]+)"))
local style_hits = selfplay:match("style_hits=(%d+/%d+)")
if style_match == nil or style_hits == nil then
    error("bake_guardian_persona: self-play answer carried no style_match: " .. selfplay)
end

return {
    card_id = card_id,
    alias = ALIAS,
    name = NAME,
    prompt = PROMPT,
    basis_style = BASIS_STYLE,
    retries = retries,
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
    style_match = style_match,
    style_hits = style_hits,
    selfplay = selfplay,
}
