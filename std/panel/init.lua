--- Panel — multi-perspective deliberation
--- Multiple roles present positions responding to prior arguments, then a moderator synthesizes.
---
--- Usage:
---   local panel = require("panel")
---   return panel.run(ctx)
---
--- ctx.task (required): The topic/question
--- ctx.roles: List of role names (default: {"advocate", "critic", "pragmatist"})

local M = {}

M.meta = {
    name = "panel",
    version = "0.1.0",
    description = "Multi-perspective deliberation — distinct roles engage, moderator synthesizes",
    category = "synthesis",
}

function M.run(ctx)
    local task = ctx.task or error("ctx.task is required")
    local roles = ctx.roles or { "advocate", "critic", "pragmatist" }

    local arguments = {}

    -- Each role presents their position, responding to prior arguments
    for _, role in ipairs(roles) do
        local context = ""
        for _, arg in ipairs(arguments) do
            context = context .. string.format("[%s]: %s\n", arg.role, arg.text)
        end

        local prompt
        if #arguments == 0 then
            prompt = string.format(
                "Topic: %s\n\nAs a %s, present your initial position. 2-3 sentences.",
                task, role
            )
        else
            prompt = string.format(
                "Topic: %s\n\nPrevious arguments:\n%s\n\nAs a %s, respond to the discussion. Engage with specific points. 2-3 sentences.",
                task, context, role
            )
        end

        local text = alc.llm(prompt, {
            system = string.format("You are a %s. Stay in character. Be specific.", role),
            max_tokens = 250,
        })

        arguments[#arguments + 1] = { role = role, text = text }
    end

    -- Synthesis
    local all_args = ""
    for _, arg in ipairs(arguments) do
        all_args = all_args .. string.format("[%s]: %s\n", arg.role, arg.text)
    end

    local synthesis = alc.llm(
        string.format(
            "Topic: %s\n\nDebate:\n%s\n\nSynthesize: identify agreements, key disagreements, and a balanced actionable conclusion. 3-4 sentences.",
            task, all_args
        ),
        { system = "You are a wise moderator. Be balanced and decisive.", max_tokens = 400 }
    )

    ctx.result = {
        arguments = arguments,
        synthesis = synthesis,
    }
    return ctx
end

return M
