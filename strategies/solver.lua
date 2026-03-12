--- UCB1 Hypothesis Swarm
--- Generates multiple hypotheses, scores them with UCB1, refines the best.
---
--- ctx.task (required): The problem to solve
--- ctx.rounds: Number of rounds (default: 2)
--- ctx.n: Number of hypotheses (default: 3)

local task = ctx.task or error("ctx.task is required")
local rounds = ctx.rounds or 2
local n_hyp = ctx.n or 3

local hypotheses = {}
local scores = {}
local total_pulls = 0

-- UCB1 score: exploitation + exploration
local function ucb1(idx)
    if scores[idx].n == 0 then return math.huge end
    local avg = scores[idx].total / scores[idx].n
    local explore = math.sqrt(2 * math.log(total_pulls + 1) / scores[idx].n)
    return avg + explore
end

-- Generate initial hypotheses
for i = 1, n_hyp do
    local h = alc.llm(
        string.format(
            "Task: %s\n\nPropose hypothesis #%d. A distinct, specific approach. 2-3 sentences.",
            task, i
        ),
        { system = "You are a creative problem solver. Each hypothesis must differ from others.", max_tokens = 200 }
    )
    hypotheses[i] = h
    scores[i] = { total = 0, n = 0 }
end

-- Evaluate & refine rounds
for round = 1, rounds do
    -- Build context of all hypotheses
    local listing = ""
    for i, h in ipairs(hypotheses) do
        listing = listing .. string.format("#%d: %s\n", i, h)
    end

    -- Score each hypothesis
    for i = 1, #hypotheses do
        local score_str = alc.llm(
            string.format(
                "Task: %s\n\nHypotheses:\n%s\nRate hypothesis #%d (1-10) for quality, feasibility, and originality. Reply with ONLY the number.",
                task, listing, i
            ),
            { system = "You are a critical evaluator. Just the number.", max_tokens = 10 }
        )
        local score = tonumber(score_str:match("%d+")) or 5
        scores[i].total = scores[i].total + score
        scores[i].n = scores[i].n + 1
        total_pulls = total_pulls + 1
    end

    -- Select best by UCB1 and refine
    local best_idx = 1
    local best_ucb = -1
    for i = 1, #hypotheses do
        local u = ucb1(i)
        if u > best_ucb then
            best_ucb = u
            best_idx = i
        end
    end

    hypotheses[best_idx] = alc.llm(
        string.format(
            "Task: %s\n\nCurrent best hypothesis:\n%s\n\nRefine and strengthen it. Address weaknesses. 2-3 sentences.",
            task, hypotheses[best_idx]
        ),
        { system = "You are an expert refiner. Make it sharper and more actionable.", max_tokens = 200 }
    )
end

-- Final ranking
local ranked = {}
for i, h in ipairs(hypotheses) do
    local avg = scores[i].n > 0 and (scores[i].total / scores[i].n) or 0
    ranked[#ranked + 1] = { rank = 0, hypothesis = h, avg_score = avg, pulls = scores[i].n }
end
table.sort(ranked, function(a, b) return a.avg_score > b.avg_score end)
for i, r in ipairs(ranked) do r.rank = i end

return alc.json_encode({
    task = task,
    rounds = rounds,
    best = ranked[1].hypothesis,
    ranking = ranked,
})
