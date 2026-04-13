-- sweep_replay_demo.lua
--
-- Replay sweep: load existing Cards, derive new ones with different
-- alpha parameters, and chain via prior_card_id. Demonstrates the
-- two-tier content policy — Tier 1 (Card body) holds aggregate scalars,
-- Tier 2 (samples.jsonl) holds per-case raw data.
--
-- Self-contained — no LLM calls. Synthetic scores stand in for real
-- grader output so the derivation + samples flow can be exercised
-- end-to-end.
--
-- Run:
--   alc_run code_file=examples/cards/sweep_replay_demo.lua
--
-- Inspect:
--   ls ~/.algocline/cards/sweep_replay_demo/

local PKG = "sweep_replay_demo"
local SCENARIO = "ema_alpha_sweep"

local rng = alc.math.rng_create(42)

-- ─── Phase 1: Create seed Cards (simulating prior eval runs) ─────

local seed_ids = {}
for i = 1, 3 do
    local per_case = {}
    local total = 0
    for c = 1, 5 do
        local score = alc.math.rng_float(rng)
        per_case[c] = {
            case = "q" .. c,
            passed = score >= 0.5,
            score = score,
        }
        total = total + score
    end
    local pass_count = 0
    for _, row in ipairs(per_case) do
        if row.passed then pass_count = pass_count + 1 end
    end

    local result = alc.card.create({
        pkg = { name = PKG },
        scenario = { name = SCENARIO },
        model = { id = "claude-opus-4-6" },
        stats = {
            ev_raw = total / #per_case,
            pass_rate = pass_count / #per_case,
        },
    })
    seed_ids[i] = result.card_id

    alc.card.write_samples(result.card_id, per_case)
end

-- ─── Phase 2: Sweep alpha values, derive new Cards ──────────────

local alphas = { 0.3, 0.5, 0.7, 0.9 }
local derived = {}

for _, alpha in ipairs(alphas) do
    for _, seed_id in ipairs(seed_ids) do
        local seed = alc.card.get(seed_id)
        local raw_ev = seed.stats.ev_raw

        local prior_ev = 0.5
        local new_ev = alpha * raw_ev + (1 - alpha) * prior_ev

        local result = alc.card.create({
            pkg = { name = PKG },
            scenario = { name = SCENARIO },
            model = { id = "claude-opus-4-6" },
            params = { alpha = alpha },
            stats = {
                ev = new_ev,
                ev_raw = raw_ev,
            },
            metadata = {
                prior_card_id = seed_id,
                derived_from = "replay_sweep",
            },
        })
        derived[#derived + 1] = {
            card_id = result.card_id,
            alpha = alpha,
            ev = new_ev,
            prior_card_id = seed_id,
        }
    end
end

-- ─── Phase 3: Find the best derived Card ────────────────────────

local all = alc.card.find({
    pkg = PKG,
    where = { scenario = { name = SCENARIO } },
    limit = 100,
})

local best_alias = "best_" .. PKG
if #derived > 0 then
    local best = derived[1]
    for _, d in ipairs(derived) do
        if d.ev > best.ev then best = d end
    end
    alc.card.alias_set(best_alias, best.card_id, { pkg = PKG })
end

return {
    seeds = seed_ids,
    derived_count = #derived,
    derived = derived,
    total_cards = #all,
    alias = best_alias,
}
