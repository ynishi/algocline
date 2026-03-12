--- Ensemble — independent sampling with majority vote
--- Samples multiple reasoning paths for the same problem,
--- then selects the most consistent answer by majority voting.
---
--- Based on: Wang et al., "Self-Consistency Improves Chain of Thought
--- Reasoning in Language Models" (2022, arXiv:2203.11171)
---
--- Usage:
---   local ensemble = require("ensemble")
---   return ensemble.run(ctx)
---
--- ctx.task (required): The problem to solve
--- ctx.n: Number of reasoning paths to sample (default: 5)
--- ctx.temperature_hint: Hint text for diversity (default: varies per sample)

local M = {}

M.meta = {
    name = "ensemble",
    version = "0.1.0",
    description = "Independent multi-path sampling with majority vote aggregation",
    category = "aggregation",
}

--- Extract a concise final answer from a reasoning chain.
local function extract_answer(reasoning, task)
    return alc.llm(
        string.format(
            "Original question: %s\n\nReasoning:\n%s\n\nExtract ONLY the final answer in one short sentence. No explanation.",
            task, reasoning
        ),
        { system = "Extract the final answer concisely. One sentence max.", max_tokens = 100 }
    )
end

function M.run(ctx)
    local task = ctx.task or error("ctx.task is required")
    local n = ctx.n or 5

    -- Diversity hints to encourage different reasoning paths
    local diversity_hints = {
        "Think step by step carefully.",
        "Approach this from first principles.",
        "Consider an alternative perspective.",
        "Work backwards from the expected outcome.",
        "Break this into smaller sub-problems.",
        "Use an analogy to reason about this.",
        "Consider edge cases and exceptions first.",
    }

    -- Sample N independent reasoning paths
    local paths = {}
    for i = 1, n do
        local hint = diversity_hints[((i - 1) % #diversity_hints) + 1]
        local reasoning = alc.llm(
            string.format(
                "Question: %s\n\n%s Show your reasoning, then give a clear final answer.",
                task, hint
            ),
            {
                system = "You are a careful reasoner. Think through the problem thoroughly before answering.",
                max_tokens = 400,
            }
        )
        local answer = extract_answer(reasoning, task)
        paths[i] = { reasoning = reasoning, answer = answer }
    end

    -- Ask LLM to cluster answers and pick the majority
    local answers_list = ""
    for i, p in ipairs(paths) do
        answers_list = answers_list .. string.format("Path %d answer: %s\n", i, p.answer)
    end

    local consensus = alc.llm(
        string.format(
            "Question: %s\n\nMultiple reasoning paths produced these answers:\n%s\n"
                .. "Group similar answers together. Which answer appears most frequently? "
                .. "State the majority answer and the vote count (e.g., '3 out of 5 paths agreed').",
            task, answers_list
        ),
        {
            system = "You are a precise vote counter. Identify the majority answer. Be exact.",
            max_tokens = 300,
        }
    )

    ctx.result = {
        consensus = consensus,
        paths = paths,
        n_sampled = n,
    }
    return ctx
end

return M
