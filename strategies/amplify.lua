--- Chain-of-Thought Amplification
--- Builds a reasoning chain step by step, then synthesizes.
---
--- ctx.task (required): The question/task
--- ctx.depth: Number of reasoning steps (default: 3)

local task = ctx.task or error("ctx.task is required")
local depth = ctx.depth or 3

local chain = {}

for i = 1, depth do
    local prompt
    if i == 1 then
        prompt = string.format(
            "Task: %s\n\nThink step by step. What is the first key insight? 1-2 sentences.",
            task
        )
    else
        local prev = ""
        for j, t in ipairs(chain) do
            prev = prev .. string.format("Step %d: %s\n", j, t)
        end
        prompt = string.format(
            "Task: %s\n\nReasoning so far:\n%s\n\nWhat is the next key insight? Build on previous reasoning. 1-2 sentences.",
            task, prev
        )
    end

    chain[#chain + 1] = alc.llm(prompt, {
        system = "You are a deep thinker. Each step should reveal a new insight.",
        max_tokens = 200,
    })
end

-- Final synthesis
local all = ""
for i, t in ipairs(chain) do
    all = all .. string.format("Step %d: %s\n", i, t)
end

local conclusion = alc.llm(
    string.format(
        "Task: %s\n\nReasoning chain:\n%s\n\nSynthesize into a clear, comprehensive answer.",
        task, all
    ),
    { system = "You are an expert synthesizer. Be thorough but concise.", max_tokens = 500 }
)

return alc.json_encode({
    task = task,
    chain = chain,
    conclusion = conclusion,
})
