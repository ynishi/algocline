-- Bake a card duel persona: NL prompt -> synthesised policy -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). Where
-- `train_card_duel_npc.lua` learns one of the hand-written styles in
-- `card_duel.STYLES`, this script takes a one-line description of a
-- play style in plain language, asks the host LLM to write the matching
-- Lua policy, validates that policy, and then reuses exactly the same
-- corpus / Full FT / Card path. The teacher is synthesised; everything
-- downstream of it is unchanged.
--
-- The LLM answer is never loaded raw. `card_duel.compile_policy`
-- compiles it in a restricted environment (`math` / `table` / `ipairs`
-- / `pairs`, no `load`, no `os`, no `io`) and makes it answer a batch
-- of sampled states legally and deterministically first. A rejected
-- candidate is re-synthesised with the rejection message attached, up
-- to `MAX_RETRIES` times, after which the run fails loudly.
--
-- ctx:
--   prompt      -- required; one line of plain language describing the
--                  play style to bake
--   name        -- required; slug matching ^[a-z0-9_]+$. The Card is
--                  pinned to the alias `card_duel_npc_<name>`
--   games       -- floor on the teacher playouts (default 300). The
--                  corpus is always grown to at least `steps * batch`
--                  rows, because `alc.nn.data.synthetic` walks its rows
--                  once and errors out when the trainer asks for more
--   steps       -- Full FT steps (default 800)
--   lr          -- learning rate (default 3e-3)
--   batch       -- batch size (default 32)
--   seed        -- base seed for the deals (default 20260731)
--   check_games -- self-play games used for the compliance report
--                  (default 20)
--
-- The run pauses once, at Phase 1, on the `alc.llm` call carrying the
-- synthesis prompt; the host answers and resumes via `alc_continue`.
-- Nothing after that needs a host round trip.
--
-- Returns a flat table of scalars: `card_id`, `alias`, `style_match`,
-- `style_hits`, `train_loss` and `retries`, plus the corpus shape. The
-- compliance number is reported, not asserted: the fence lives in the
-- eval scenario and in the demo runbook, so a low score here is data
-- rather than a failed run.

local duel = require("card_duel")
local npc = require("card_duel_npc")

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
local GAMES = math.floor(tonumber(ctx_field("games")) or 300)
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)
local CHECK_GAMES = math.floor(tonumber(ctx_field("check_games")) or 20)

--- Synthesis attempts beyond the first one.
local MAX_RETRIES = 3

--- Playouts behind the states a candidate policy is validated against.
--- Ten states per game, so twenty games is a two hundred state batch.
local VALIDATION_GAMES = 20

--- Longest prompt excerpt kept on the alias note.
local NOTE_LEN = 80

local VOCAB = duel.vocab()

local function log(msg)
    alc.log("info", "[card-duel-bake] " .. msg)
end

-- ─── ctx validation ─────────────────────────────────────────────────

if type(PROMPT) ~= "string" or #PROMPT == 0 then
    error("bake_card_duel_persona: ctx.prompt is required and must be a non-empty string")
end
if type(NAME) ~= "string" or not NAME:match("^[a-z0-9_]+$") then
    error(
        string.format(
            "bake_card_duel_persona: ctx.name must match ^[a-z0-9_]+$, got %s",
            tostring(NAME)
        )
    )
end
if CHECK_GAMES <= 0 then
    error("bake_card_duel_persona: ctx.check_games must be a positive integer")
end

local ALIAS = "card_duel_npc_" .. NAME
local CARD_NAME = "card-duel-persona-" .. NAME

-- ─── Phase 1 / 2: policy synthesis and validation ───────────────────
--
-- The contract below is the whole interface the LLM is given: the
-- rules, the state fields, the legality requirement, the purity
-- requirement and the exact set of globals its chunk will be able to
-- reach. It is written out literally rather than derived from the
-- module so that the prompt stays readable in the run log.

local CONTRACT = [[
You are writing one Lua function that plays a five-round card duel.

Rules of the game:
- Two players, five rounds. Each player is dealt five ranks drawn uniformly
  from 1 to 9 with replacement.
- Every round both players reveal one card from hand at the same time. The
  higher rank scores one point, a tie scores nothing, and both cards are
  discarded.
- The higher score after five rounds wins.

The function takes one argument, a table describing your own seat:
- state.round        integer from 1 to 5, the round about to be played
- state.my_hand      array of integers, the ranks still in your hand, sorted
                     ascending
- state.my_points    integer, your score so far
- state.opp_points   integer, the opponent score so far
- state.opp_played   array of integers, the ranks the opponent has played,
                     oldest first (empty in round one)

The function must return one integer rank that is present in state.my_hand.
Duplicates collapse: the legal answers are the distinct values of
state.my_hand.

Hard requirements:
- Answer with the chunk only: `return function(state) ... end`. No prose, no
  markdown fence, no comments outside the chunk.
- The function must be pure and deterministic. The same state must always
  produce the same rank. Never call math.random.
- Never modify the state table or the tables inside it.
- Only these globals exist inside the chunk: math, table, ipairs, pairs.
  There is no string, no os, no io, no print, no require, no load, no
  setmetatable, no select, no type, no tostring.
- Read the state to decide. A function that ignores the state and returns a
  constant cannot express a play style.
]]

--- Build the synthesis prompt, optionally carrying a rejection.
---
--- The rejection message from `card_duel.compile_policy` names the
--- state and the reason, which is the only feedback the next attempt
--- gets; paraphrasing it would drop the state that failed.
local function synthesis_prompt(rejection)
    local parts = {
        CONTRACT,
        "",
        "Play style to implement, in the author's words:",
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
        "synthesising a policy for %q against %d validation states",
        NAME,
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
            "bake_card_duel_persona: no valid policy after %d attempts; last rejection: %s",
            MAX_RETRIES + 1,
            tostring(rejection)
        )
    )
end

-- ─── Phase 3: corpus, training, alias ───────────────────────────────
--
-- Identical to `train_card_duel_npc.lua` from here on: the synthesised
-- policy is just another labelling function, so it goes through the
-- shared `card_duel.build_corpus` and the same Full FT call.

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
            "bake_card_duel_persona: alphabet of %d chars exceeds the model vocabulary of %d",
            VOCAB.size,
            model_vocab
        )
    )
end

local playouts = math.max(GAMES, math.ceil(STEPS * BATCH / duel.ROWS_PER_GAME))
local rows = duel.build_corpus(policy, {
    ctx_len = ctx_len,
    games = playouts,
    seed = SEED,
    pad_id = VOCAB.pad_id,
})
log(string.format("corpus: %d rows x %d tokens from %d playouts", #rows, ctx_len, playouts))

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
    error("bake_card_duel_persona: run_full_ft returned no card_id")
end

alc.card.alias_set(ALIAS, card_id, {
    pkg = "card_duel_npc",
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
    error("bake_card_duel_persona: metadata.nn.metrics.train_loss missing from the Card")
end

-- ─── Phase 4: persona record ────────────────────────────────────────
--
-- The prompt and the chunk it produced are the two things that cannot
-- be recovered from the weights, so they are appended to the Card. The
-- append surface is additive only, which is why `persona` is a fresh
-- top-level key rather than an edit of `metadata`.

local appended = alc.card.append(card_id, {
    persona = {
        prompt = PROMPT,
        policy_source = policy_source,
    },
})
if type(appended) ~= "table" then
    error("bake_card_duel_persona: alc.card.append did not return the merged Card")
end

-- ─── Phase 5: inline compliance report ──────────────────────────────
--
-- Self-play through the NPC package itself, scored against the very
-- policy the corpus was labelled with. The number is reported, not
-- fenced: what an acceptable compliance rate is depends on the training
-- budget, and that judgement belongs to the scenario.

npc.reset_cache()

local selfplay = npc.run({
    task = alc.json_encode({
        mode = "selfplay",
        games = CHECK_GAMES,
        seed = SEED,
        policy_source = policy_source,
    }),
    card_alias = ALIAS,
}).result
log(string.format("selfplay -> %s", selfplay))

local style_match = tonumber(selfplay:match("style_match=([%d%.]+)"))
local style_hits = selfplay:match("style_hits=(%d+/%d+)")
if style_match == nil or style_hits == nil then
    error("bake_card_duel_persona: self-play answer carried no style_match: " .. selfplay)
end

return {
    card_id = card_id,
    alias = ALIAS,
    name = NAME,
    prompt = PROMPT,
    retries = retries,
    games = playouts,
    steps = STEPS,
    rows = #rows,
    ctx_len = ctx_len,
    train_loss = train_loss,
    baseline_loss = baseline_loss,
    loss_descended = train_loss < baseline_loss,
    check_games = CHECK_GAMES,
    style_match = style_match,
    style_hits = style_hits,
    selfplay = selfplay,
}
