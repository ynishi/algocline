-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/presets/llm_graders.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  presets/llm_graders.lua — LLM-as-Judge graders (Tier 2)

  These call an LLM to grade another LLM's response.
  Use deterministic graders (presets/graders.lua) first.
  LLM judges supplement where deterministic checks can't reach.

  A provider is REQUIRED — there is no default.
  Pass any function conforming to provider(input) → response.

  Usage:
    local llm_graders = require("evalframe.presets.llm_graders")

    local judge = llm_graders.rubric("Rate accuracy 1-5", {
      provider = my_provider,
    })

    bind { judge, scorers.linear_1_5 }
]]

local grader = require("evalframe.model.grader")

local M = {}

-- ============================================================
-- Internal: build call function from provider
-- ============================================================

--- Build a call function from opts.provider.
--- Handles both string and table returns per Provider contract.
---@param opts table
---@return function(prompt) → text, err
local function make_call_fn(opts)
  local provider = opts and opts.provider
  if not provider then
    error("llm_graders: 'provider' is required (no default LLM provider)", 3)
  end
  if type(provider) ~= "function" then
    error(string.format("llm_graders: 'provider' must be function, got %s", type(provider)), 3)
  end

  return function(prompt)
    local resp = provider(prompt)
    if type(resp) == "string" then return resp end
    if type(resp) ~= "table" then
      return nil, string.format("provider returned %s (expected string or table)", type(resp))
    end
    if resp.error then return nil, resp.error end
    return resp.text or ""
  end
end

--- Simple string hash (djb2) for name uniqueness.
---@param s string
---@return number
local function str_hash(s)
  local h = 5381
  for i = 1, #s do
    h = (h * 33 + s:byte(i)) % 0x7FFFFFFF
  end
  return h
end

--- Generate a unique grader name from a base and a distinguishing label.
--- Appends a hash suffix to prevent collisions from truncation.
---@param base string
---@param label string
---@return string
local function judge_name(base, label)
  local short = label:sub(1, 32):gsub("[^%w_]", "_"):lower()
  return string.format("%s:%s_%07x", base, short, str_hash(label))
end

-- ============================================================
-- Rubric-based judge
-- ============================================================

--- Create an LLM-as-Judge grader with a rubric.
--- The judge sees the input, response, and rubric, returns a numeric rating.
---
---@param rubric string  Grading criteria
---@param opts table  { provider (required), scale_min?, scale_max? }
---@return table GraderDef
function M.rubric(rubric, opts)
  opts = opts or {}
  local scale_min = opts.scale_min or 1
  local scale_max = opts.scale_max or 5
  local call_fn = make_call_fn(opts)

  return grader(judge_name("llm_judge", rubric)) {
    check = function(resp, case)
      local prompt = string.format(
        [[You are an evaluation judge. Grade the following response.

<input>
%s
</input>

<response>
%s
</response>

RUBRIC: %s

Rate the response on a scale of %d to %d.
Reply with ONLY a single number, nothing else.]],
        case.input,
        resp.text or "",
        rubric,
        scale_min,
        scale_max
      )

      local text, err = call_fn(prompt)
      if not text then return nil, err end

      local rating = tonumber(text:match("(%d+%.?%d*)"))
      if not rating then
        return nil, string.format("judge did not return a number: %s", text:sub(1, 100))
      end

      return math.max(scale_min, math.min(scale_max, rating))
    end,

  }
end

-- ============================================================
-- Yes/No judge
-- ============================================================

--- Create a yes/no LLM judge.
---@param question string  Question to ask about the response
---@param opts table  { provider (required) }
---@return table GraderDef
function M.yes_no(question, opts)
  opts = opts or {}
  local call_fn = make_call_fn(opts)

  return grader(judge_name("llm_yes_no", question)) {
    check = function(resp, case)
      local prompt = string.format(
        [[You are an evaluation judge.

<input>
%s
</input>

<response>
%s
</response>

QUESTION: %s

Reply with ONLY "Yes" or "No".]],
        case.input,
        resp.text or "",
        question
      )

      local text, err = call_fn(prompt)
      if not text then return nil, err end

      local lower = text:lower():match("%a+")
      return lower == "yes"
    end,

  }
end

-- ============================================================
-- Factuality judge
-- ============================================================

--- Create a factuality judge that checks response against expected.
---@param opts table  { provider (required) }
---@return table GraderDef
function M.factuality(opts)
  opts = opts or {}
  local call_fn = make_call_fn(opts)

  return grader "llm_factuality" {
    check = function(resp, case)
      local expected_text = ""
      if case.expected then
        expected_text = table.concat(case.expected, " OR ")
      end

      local prompt = string.format(
        [[You are a factuality judge.

<question>
%s
</question>

<expected>
%s
</expected>

<response>
%s
</response>

Does the actual response convey the same factual content as the expected answer?
Minor wording differences are acceptable. Factual accuracy is what matters.

Rate from 1 to 5:
1 = Completely wrong
2 = Mostly wrong, some correct elements
3 = Partially correct
4 = Mostly correct, minor inaccuracies
5 = Factually equivalent

Reply with ONLY a single number.]],
        case.input,
        expected_text,
        resp.text or ""
      )

      local text, err = call_fn(prompt)
      if not text then return nil, err end

      local rating = tonumber(text:match("(%d+%.?%d*)"))
      if not rating then
        return nil, string.format("factuality judge error: %s", text:sub(1, 100))
      end

      return math.max(1, math.min(5, rating))
    end,

  }
end

return M
