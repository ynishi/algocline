--- Verify — draft-verify-revise cycle
--- Reduces hallucination by: draft → generate verification questions →
--- answer them independently → produce verified final response.
---
--- Based on: Dhuliawala et al., "Chain-of-Verification Reduces
--- Hallucination in Large Language Models" (2023, arXiv:2309.11495)
---
--- Usage:
---   local verify = require("verify")
---   return verify.run(ctx)
---
--- ctx.task (required): The question/task to answer
--- ctx.n_questions: Number of verification questions (default: 3)

local M = {}

M.meta = {
    name = "verify",
    version = "0.1.0",
    description = "Draft-verify-revise — reduces hallucination via independent fact-checking",
    category = "validation",
}

function M.run(ctx)
    local task = ctx.task or error("ctx.task is required")
    local n_questions = ctx.n_questions or 3

    -- Step 1: Generate baseline draft response
    local draft = alc.llm(
        string.format("Answer the following thoroughly:\n\n%s", task),
        {
            system = "You are a knowledgeable assistant. Provide a detailed, informative answer.",
            max_tokens = 500,
        }
    )

    -- Step 2: Plan verification questions
    local vq_raw = alc.llm(
        string.format(
            "Original question: %s\n\nDraft answer:\n%s\n\n"
                .. "Generate exactly %d verification questions that would help fact-check "
                .. "the specific claims in the draft answer. Each question should target "
                .. "a different factual claim. Format: one question per line, numbered.",
            task, draft, n_questions
        ),
        {
            system = "You are a fact-checker. Generate precise, targeted verification questions.",
            max_tokens = 300,
        }
    )

    -- Step 3: Answer each verification question INDEPENDENTLY
    -- (without seeing the draft, to avoid bias)
    local verifications = {}
    -- Parse numbered questions
    local questions = {}
    for line in vq_raw:gmatch("[^\n]+") do
        local q = line:match("^%d+[%.%)%s]+(.+)") or line:match("^[%-*]%s*(.+)")
        if q and #q > 10 then
            questions[#questions + 1] = q
        end
    end

    -- Limit to n_questions
    for i = 1, math.min(#questions, n_questions) do
        local answer = alc.llm(
            string.format("Answer this question accurately and concisely:\n\n%s", questions[i]),
            {
                system = "You are a fact-checker. Answer based on what you know. "
                    .. "If uncertain, say so. Do not fabricate.",
                max_tokens = 200,
            }
        )
        verifications[i] = { question = questions[i], answer = answer }
    end

    -- Step 4: Generate final verified response
    local verification_summary = ""
    for i, v in ipairs(verifications) do
        verification_summary = verification_summary
            .. string.format("Q%d: %s\nA%d: %s\n\n", i, v.question, i, v.answer)
    end

    local final_response = alc.llm(
        string.format(
            "Original question: %s\n\nDraft answer:\n%s\n\n"
                .. "Verification results:\n%s\n"
                .. "Revise the draft answer based on the verification results. "
                .. "Correct any inaccuracies found. If the draft was accurate, keep it. "
                .. "Mark any claims you could not verify with [unverified].",
            task, draft, verification_summary
        ),
        {
            system = "You are a meticulous editor. Correct factual errors found by verification. "
                .. "Preserve accurate information. Be transparent about uncertainty.",
            max_tokens = 600,
        }
    )

    ctx.result = {
        draft = draft,
        verifications = verifications,
        final_response = final_response,
    }
    return ctx
end

return M
