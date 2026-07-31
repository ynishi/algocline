-- Train the guardian duel boss NPC: teacher corpus -> Full FT -> Card.
--
-- Self-contained script for `alc_run` (`code_file` form). It generates
-- its own supervised corpus from one of the deterministic styles in
-- `guardian_duel.STYLES`, tunes a from-scratch `gpt2 tiny` on it,
-- registers the resulting Card, pins the alias `guardian_duel_npc`
-- reads, and finishes with a decode check plus a self-play compliance
-- line through the NPC package itself.
--
-- No `alc.llm` call happens anywhere in this path, so the run never
-- pauses for a host response.
--
-- ctx (all optional, JSON object passed to alc_run):
--   games       -- floor on the teacher playouts to generate (default
--                  300). More fights are played on top of that floor
--                  until the corpus actually covers the training run,
--                  because `alc.nn.data.synthetic` walks its rows once
--                  and errors out when the trainer asks for more. How
--                  many fights that takes is measured rather than
--                  predicted; see `build_full_corpus`.
--   steps       -- Full FT steps (default 800)
--   lr          -- learning rate (default 3e-3)
--   batch       -- batch size (default 32)
--   seed        -- base seed for the fights (default 20260731)
--   style       -- teacher style to learn (default "guardian"); one of
--                  `guardian_duel.STYLES`, or "all" to train every style
--                  in turn
--   alias       -- Card alias to pin (default
--                  "guardian_duel_npc_<style>"; ignored when
--                  style = "all")
--   name        -- Card name (default "guardian-duel-npc")
--   check_games -- self-play fights behind the compliance line
--                  (default 20)
--
-- Aliases: the style-specific alias is always pinned, and the
-- `guardian` style additionally pins the bare `guardian_duel_npc` alias
-- to the same Card, which is the one the NPC package falls back to.
--
-- The style is not only the teacher: `guardian_duel.encode` measures its
-- `D` field against the style's mode-shift threshold, so the corpus, the
-- decode checks and the self-play run all name the same style. A model
-- trained under one basis and decoded under another reads a distance its
-- labels never followed.
--
-- Returns a flat table of scalars for a single style so a Rust smoke
-- harness can assert on it directly. With `style = "all"` the return is
-- `{ ok, styles = { [style] = <flat table> }, trained }` instead.

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

local GAMES = math.floor(tonumber(ctx_field("games")) or 300)
local STEPS = math.floor(tonumber(ctx_field("steps")) or 800)
local LR = tonumber(ctx_field("lr")) or 3e-3
local BATCH = math.floor(tonumber(ctx_field("batch")) or 32)
local SEED = math.floor(tonumber(ctx_field("seed")) or 20260731)
local STYLE = ctx_field("style") or "guardian"
local ALIAS_OVERRIDE = ctx_field("alias")
local NAME = ctx_field("name") or "guardian-duel-npc"
local CHECK_GAMES = math.floor(tonumber(ctx_field("check_games")) or 20)

--- Alias the NPC package falls back to, kept for the teacher style.
local BARE_ALIAS = "guardian_duel_npc"

local VOCAB = duel.vocab()

local function log(msg)
    alc.log("info", "[guardian-train] " .. msg)
end

--- Resolve a style name to its policy, or fail loudly.
---
--- An unknown style would otherwise train a model on `nil` labels far
--- from the call site, so the valid names are listed in the message.
local function policy_for(style)
    local fn = duel["policy_" .. tostring(style)]
    if type(fn) ~= "function" then
        error(
            "train_guardian_npc: unknown style '"
                .. tostring(style)
                .. "' (valid: "
                .. table.concat(duel.STYLES, ", ")
                .. ")"
        )
    end
    return fn
end

--- Default alias for a style, used unless `ctx.alias` overrides it.
local function default_alias(style)
    return "guardian_duel_npc_" .. style
end

-- ─── Corpus ─────────────────────────────────────────────────────────
--
-- A fight lasts at most `MAX_ROWS_PER_GAME` turns but ends the moment
-- either side drops, so the rows one playout contributes are a property
-- of the style rather than of the rules: the teacher styles measure
-- between 6.4 and 9.0 rows per fight. Sizing the batch up front from the
-- turn limit therefore under-counts every style, and the trainer runs
-- out of data mid-run ("dataset exhausted after 797/1000 steps") after
-- the whole corpus has already been built.
--
-- The playouts are generated in rounds instead. Each round is sized from
-- the rows the previous rounds actually produced, so the yield is
-- measured rather than assumed, and the loop stops once the corpus
-- covers the run. Each round carries its own seed, so the corpus stays a
-- function of `ctx.seed` alone.

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

--- Build a corpus that covers `target` rows, and report what it cost.
---
--- `ctx.games` is honoured as a floor on the first round, which keeps
--- its original meaning: a caller asking for 300 fights still gets at
--- least 300.
---
--- `build_corpus` deals its fights from `seed + 1` upwards, so a round
--- opens where the previous one stopped. No two rounds replay a fight,
--- whatever size they end up being, and the whole corpus is still a
--- function of `ctx.seed`.
---@param policy fun(state: table): string Labelling boss policy
---@param style string Distance basis the rows are encoded against
---@param ctx_len integer Model context window
---@param target integer Rows the training run consumes
---@return table rows, integer played, integer rounds
local function build_full_corpus(policy, style, ctx_len, target)
    local optimistic = math.ceil(target / duel.MAX_ROWS_PER_GAME)
    local cap = math.max(GAMES, optimistic) * CORPUS_GAMES_CAP_FACTOR
    local rows, played, rounds = {}, 0, 0
    while #rows < target do
        if played >= cap then
            error(
                string.format(
                    "train_guardian_npc: %d playouts of style %q produced %d rows but the run "
                        .. "needs %d; the fights are ending far earlier than the turn limit allows",
                    played,
                    style,
                    #rows,
                    target
                )
            )
        end
        local want
        if played == 0 then
            want = math.max(GAMES, optimistic)
        else
            -- Rows per fight measured on this style, on this seed.
            want = math.ceil((target - #rows) * played / #rows) + ROUND_PAD_GAMES
        end
        if want > cap - played then
            want = cap - played
        end
        local chunk = duel.build_corpus(policy, {
            ctx_len = ctx_len,
            games = want,
            style = style,
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

if STYLE ~= "all" then
    policy_for(STYLE)
end
if CHECK_GAMES <= 0 then
    error("train_guardian_npc: ctx.check_games must be a positive integer")
end

-- ─── Quick check through the NPC package ────────────────────────────
--
-- One state per branch of the decision, taken from fights that were
-- actually played rather than written out by hand.
--
-- The hand-written version put three of its four states outside the set
-- a fight can reach: a rolled-up boss whose damage counter was empty
-- (in a real fight the roll-up is what the full counter causes), and
-- counters paired with health the same fight could not have produced.
-- Health and damage are not independent — every point the boss loses is
-- a point on its counter, and a completed shift costs it a whole
-- threshold on top — so those states encode to lines the model is never
-- asked about, and agreeing or disagreeing with the teacher on them
-- measures nothing.
--
-- Playing three scripted fights with the style's own policy is the
-- cheapest way to be sure of the opposite. The states come out reachable
-- by construction, for every style and every threshold, and the move
-- expected of each one is the teacher's own answer at that point.

--- Player scripts the check fights are played against.
---
--- Heavy attacks push the boss over its threshold quickly; the blocks in
--- the third script cost the boss nothing, which buys the turns its
--- defensive sub-sequence needs to finish and hands the last turns back
--- with the threshold raised.
local CHECK_SCRIPTS = {
    { "A", "A", "A", "A", "A", "A", "A", "A", "A" },
    { "a", "a", "a", "a", "a", "a", "a", "a", "a" },
    { "A", "A", "A", "b", "b", "b", "A", "A", "A" },
}

--- Branches of a style's decision, in the order the check reports them.
local CHECK_BRANCHES = { "cycle", "stagger_due", "rolled_up", "cycle_after_shift" }

--- The branch a state sends the decision down.
---
--- The names follow `guardian_duel`'s own decision: mode 1 walks the
--- defensive sequence, a distance of zero is the stagger condition, and
--- the rest is the cycle — before its first shift or after one, which
--- are the two sides of the raised threshold the `D` field exists for.
local function branch_of(style, state)
    if state.mode == 1 then
        return "rolled_up"
    end
    if duel.shift_distance(style, state) == 0 then
        return "stagger_due"
    end
    if state.shifts > 0 then
        return "cycle_after_shift"
    end
    return "cycle"
end

--- First reachable state of every branch, with the move it expects.
---
--- A branch the scripts never reach is a loud error rather than a
--- shorter check: silently dropping the rolled-up case would leave the
--- defensive sequence untested while still reporting a full score.
---@param style string One of `guardian_duel.STYLES`
---@return table[] checks `{ branch, state, expected }`
local function check_states(style)
    local policy = policy_for(style)
    local found = {}
    for _, script in ipairs(CHECK_SCRIPTS) do
        local g = duel.new_game(SEED)
        for _, player_action in ipairs(script) do
            if duel.is_over(g) then
                break
            end
            local action = policy(g.boss)
            local branch = branch_of(style, g.boss)
            if found[branch] == nil then
                found[branch] = { branch = branch, state = g.boss, expected = action }
            end
            g = duel.apply(g, player_action, action)
        end
    end
    local checks = {}
    for _, branch in ipairs(CHECK_BRANCHES) do
        local hit = found[branch]
        if hit == nil then
            error(
                string.format(
                    "train_guardian_npc: the check fights never reach the %s branch of style %q; "
                        .. "extend CHECK_SCRIPTS until they do",
                    branch,
                    style
                )
            )
        end
        checks[#checks + 1] = hit
    end
    return checks
end

-- Coverage is settled before anything is trained. The states are a
-- function of the rules and the style alone, so a fixture that misses a
-- branch can be caught in milliseconds instead of after a Full FT run
-- has already been paid for.
if STYLE == "all" then
    for _, style in ipairs(duel.STYLES) do
        check_states(style)
    end
else
    check_states(STYLE)
end

-- ─── Train one style ────────────────────────────────────────────────

--- Run the whole pipeline for a single style and return the flat result
--- table a smoke harness reads.
---
--- Each call builds its own model handle, so `run_full_ft` takes and
--- releases the training lease once per style and the styles can be
--- looped sequentially.
local function train_style(style, alias)
    local policy = policy_for(style)
    -- Collected before the model exists: the check fights are played by
    -- the teacher, not by the Card.
    local checks = check_states(style)

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
                "train_guardian_npc: alphabet of %d chars exceeds the model vocabulary of %d",
                VOCAB.size,
                model_vocab
            )
        )
    end

    local target = STEPS * BATCH + CORPUS_SLACK_BATCHES * BATCH
    local rows, playouts, rounds = build_full_corpus(policy, style, ctx_len, target)
    log(
        string.format(
            "[%s] corpus: %d rows x %d tokens from %d playouts in %d round(s), target %d",
            style,
            #rows,
            ctx_len,
            playouts,
            rounds,
            target
        )
    )

    local dataset = alc.nn.data.synthetic(rows, {
        batch_size = BATCH,
        ctx_len = ctx_len,
        shuffle = true,
        pad_id = VOCAB.pad_id,
    })

    log(string.format("[%s] full_ft: %d steps, lr=%g, batch=%d", style, STEPS, LR, BATCH))
    local card_id = alc.nn.trainer.run_full_ft(handle, dataset, {
        lr = LR,
        batch = BATCH,
        steps = STEPS,
        warmup = 0,
        schedule = "Constant",
        name = NAME,
    })
    if type(card_id) ~= "string" or #card_id == 0 then
        error("train_guardian_npc: run_full_ft returned no card_id")
    end

    alc.card.alias_set(alias, card_id, {
        pkg = "guardian_duel_npc",
        note = "guardian duel " .. style .. "-style boss NPC",
    })
    log(string.format("[%s] card %s pinned to alias %q", style, card_id, alias))

    -- The NPC package defaults to the bare alias, so the teacher style
    -- keeps it pointing at its own Card.
    if style == "guardian" and alias ~= BARE_ALIAS then
        alc.card.alias_set(BARE_ALIAS, card_id, {
            pkg = "guardian_duel_npc",
            note = "guardian duel teacher-style boss NPC",
        })
        log(string.format("[%s] bare alias %q pinned to the same card", style, BARE_ALIAS))
    end

    -- Uniform-random baseline over the model vocabulary. A final loss
    -- below it is the evidence that gradients flowed at all.
    local baseline_loss = math.log(model_vocab)
    local card = alc.card.get(card_id)
    local metrics = card and card.metadata and card.metadata.nn and card.metadata.nn.metrics
    local train_loss = metrics and metrics.train_loss
    if type(train_loss) ~= "number" then
        error("train_guardian_npc: metadata.nn.metrics.train_loss missing from the Card")
    end

    npc.reset_cache()

    --- Ask the NPC package, naming both the Card and the style its
    --- states are encoded against.
    local function ask(payload)
        local out = npc.run({
            task = alc.json_encode(payload),
            card_alias = alias,
            style = style,
        })
        return out.result
    end

    local decide_legal = true
    local style_hits = 0
    local reports = {}
    for i, check in ipairs(checks) do
        local text = ask({ mode = "decide", state = check.state })
        reports[#reports + 1] = text
        log(
            string.format(
                "[%s] decide[%d] %s %s expected=%s -> %s",
                style,
                i,
                check.branch,
                duel.encode(check.state, style),
                check.expected,
                text
            )
        )

        local action = text:match("action=(%a)")
        local legal = false
        for _, move in ipairs(duel.legal_actions(check.state)) do
            if move == action then
                legal = true
            end
        end
        if not legal then
            decide_legal = false
        end
        -- `expected` is the teacher's own answer at that point of the
        -- fight, read off when the state was collected.
        if action == check.expected then
            style_hits = style_hits + 1
        end
    end

    local determinism_text = ask({ mode = "determinism", state = checks[1].state })
    log(string.format("[%s] determinism -> %s", style, determinism_text))

    -- Compliance over the states the model walks into by playing, which
    -- is the number the eval scenario fences on. The teacher defaults to
    -- the style the ctx names, so the request carries no style of its
    -- own.
    local selfplay = ask({ mode = "selfplay", games = CHECK_GAMES, seed = SEED })
    log(string.format("[%s] selfplay -> %s", style, selfplay))

    local style_match = tonumber(selfplay:match("style_match=([%d%.]+)"))
    if style_match == nil then
        error("train_guardian_npc: self-play answer carried no style_match: " .. selfplay)
    end

    return {
        ok = decide_legal and train_loss < baseline_loss,
        card_id = card_id,
        alias = alias,
        style = style,
        -- The fights actually played and the rows they actually
        -- produced, not the estimate that opened the loop.
        games = playouts,
        steps = STEPS,
        rows = #rows,
        rows_target = target,
        corpus_rounds = rounds,
        ctx_len = ctx_len,
        train_loss = train_loss,
        baseline_loss = baseline_loss,
        loss_descended = train_loss < baseline_loss,
        decide_legal = decide_legal,
        style_hits = style_hits,
        style_total = #checks,
        deterministic = determinism_text:find("deterministic=true", 1, true) ~= nil,
        check_games = CHECK_GAMES,
        style_match = style_match,
        selfplay = selfplay,
        decisions = table.concat(reports, " | "),
    }
end

-- ─── Entry ──────────────────────────────────────────────────────────

if STYLE ~= "all" then
    return train_style(STYLE, ALIAS_OVERRIDE or default_alias(STYLE))
end

-- `style = "all"`: every canonical style in turn, each with its own
-- default alias. `ctx.alias` is ignored here because one alias cannot
-- name three Cards.
local styles = {}
local all_ok = true
local trained = 0
for _, style in ipairs(duel.STYLES) do
    local out = train_style(style, default_alias(style))
    styles[style] = out
    trained = trained + 1
    if not out.ok then
        all_ok = false
    end
end

return {
    ok = all_ok,
    styles = styles,
    trained = trained,
}
