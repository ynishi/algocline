-- prompt_ab_demo.lua
--
-- Generic LLM-world pattern: A/B-test prompt variants against a fixed
-- scenario, record each trial as an immutable Card, then query for the
-- best using alc.card.find and pin it with an alias.
--
-- This demo is self-contained — no LLM is actually called. Scores are
-- synthetic so the whole alc.card.* flow (create / get / list / append /
-- find / alias_set / alias_list) can be exercised end-to-end in one
-- alc_run invocation. Replace the synthetic scores with real eval
-- results (e.g. evalframe grader output) to turn this into a real
-- prompt-tuning workflow.
--
-- Run:
--   alc_run code_file=examples/cards/prompt_ab_demo.lua
--
-- Inspect:
--   ls ~/.algocline/cards/prompt_ab_demo/
--   cat ~/.algocline/cards/_aliases.toml

local PKG = "prompt_ab_demo"

-- ─── Trial matrix ────────────────────────────────────────────────
-- Three prompt variants crossed with two temperatures = 6 Cards.
local prompts = {
    { name = "terse",    system = "Answer in one sentence." },
    { name = "cot",      system = "Think step by step, then answer." },
    { name = "persona",  system = "You are a careful expert. Answer precisely." },
}
local temperatures = { 0.0, 0.7 }

-- Synthetic "accuracy" function — stands in for a real grader.
local function synthetic_score(prompt_name, temperature)
    local base = ({ terse = 0.62, cot = 0.81, persona = 0.74 })[prompt_name]
    local temp_penalty = temperature * 0.08  -- higher temp hurts exact-match
    return math.max(0, math.min(1, base - temp_penalty))
end

-- ─── Run trials + emit one Card per cell ────────────────────────
local results = {}
for _, p in ipairs(prompts) do
    for _, t in ipairs(temperatures) do
        local score = synthetic_score(p.name, t)
        local card = alc.card.create({
            pkg = { name = PKG },
            model = { id = "claude-opus-4-6" },
            scenario = { name = "factual_qa_sample50", case_count = 50 },
            params = {
                prompt_variant = p.name,
                temperature = t,
            },
            stats = {
                pass_rate = score,
                n = 50,
            },
            metadata = {
                system_prompt = p.system,
                experiment_tag = "prompt_ab_v1",
            },
        })
        results[#results + 1] = {
            card_id = card.card_id,
            variant = p.name,
            temperature = t,
            score = score,
        }
    end
end

-- ─── Query for the winner ───────────────────────────────────────
-- find() sorts by pass_rate desc and limits to the top row.
local best = alc.card.find({
    pkg = PKG,
    sort = "pass_rate",
    limit = 1,
})[1]

-- ─── Pin the winner with an alias ───────────────────────────────
alc.card.alias_set("best_prompt_ab", best.card_id, {
    pkg = PKG,
    note = "automated via prompt_ab_demo.lua",
})

-- ─── Post-hoc annotation via append ─────────────────────────────
-- Cards are immutable on existing keys, but new top-level keys can
-- be attached after the fact. Here we record a "reviewed" note.
alc.card.append(best.card_id, {
    review = {
        reviewer = "prompt_ab_demo",
        reviewed_at = os.date("!%Y-%m-%dT%H:%M:%SZ"),
        verdict = "accepted",
    },
})

-- ─── Return summary ─────────────────────────────────────────────
return {
    trials = results,
    best = {
        card_id = best.card_id,
        pass_rate = best.pass_rate,
        scenario = best.scenario,
    },
    aliases = alc.card.alias_list({ pkg = PKG }),
    total_cards_for_pkg = #alc.card.list({ pkg = PKG }),
}
