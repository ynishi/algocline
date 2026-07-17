-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/eval/runner.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  eval/runner.lua — Eval pipeline executor

  Backend Contract: (bindings, cases, provider) → RawResults

  Pipeline:
    Step 1: Call provider for each case → response
    Step 2: Grade each response via bindings
    Step 3: Score each grade
    Step 4: Collect results

  This module handles single-run execution.
  pass@k is computed by stats from the single-run results.
]]

local Binding = require("evalframe.model.binding")
local std     = require("evalframe.std")

local M = {}

-- ============================================================
-- Provider Response Contract
--
-- Providers must conform to:
--   provider(input: string) → string | table
--
-- String return: shorthand for { text = string }
-- Table return fields:
--   text       : string  (required) — LLM response text
--   latency_ms : number  (optional) — provider-measured latency in ms
--   error      : string  (optional) — error description if call failed
--   model      : string  (optional) — model identifier
--   raw        : table   (optional) — provider-specific raw response (debug)
-- ============================================================

--- Normalize provider return value to canonical response shape.
--- Always returns a new table (never mutates the provider's return value).
local function normalize_response(raw, elapsed_ms)
  if type(raw) == "string" then
    return { text = raw, latency_ms = elapsed_ms }
  end

  if type(raw) ~= "table" then
    return {
      text       = "",
      latency_ms = elapsed_ms,
      error      = string.format("provider returned %s (expected string or table)", type(raw)),
    }
  end

  -- Preserve all provider fields (enables SwarmTrace passthrough),
  -- while ensuring required response contract fields are present.
  local resp = {}
  for k, v in pairs(raw) do
    resp[k] = v
  end
  resp.text       = type(raw.text) == "string" and raw.text or ""
  resp.latency_ms = raw.latency_ms or elapsed_ms
  return resp
end

-- ============================================================
-- Step 1: Call provider
-- ============================================================

local function call_provider(provider, case)
  local start = std.time()
  local ok, raw = pcall(provider, case.input)
  local elapsed = (std.time() - start) * 1000

  if not ok then
    return {
      text       = "",
      latency_ms = elapsed,
      error      = tostring(raw),
    }
  end

  return normalize_response(raw, elapsed)
end

-- ============================================================
-- Step 2-3: Grade + Score
-- ============================================================

local function evaluate_binding(binding, response, case)
  local raw_grade, err = binding.grader.check(response, case)

  if err then
    return {
      grader  = binding.grader.name,
      score   = 0.0,
      grade   = nil,
      error   = err,
      weight  = binding.weight,
    }
  end

  local score = binding.scorer.score(raw_grade)

  -- Detect silent type coercion: scorer returned 0.0 for a non-nil raw_grade
  -- that isn't false. This typically means the grader returned a type
  -- the scorer doesn't understand (e.g., string to a linear scorer).
  local warning
  if raw_grade ~= nil and raw_grade ~= false and score == 0.0 then
    local gt = type(raw_grade)
    if gt == "string" or (gt == "table") then
      warning = string.format(
        "grader '%s' returned %s but scorer produced 0.0 (type mismatch?)",
        binding.grader.name, gt
      )
    end
  end

  return {
    grader  = binding.grader.name,
    score   = score,
    grade   = raw_grade,
    weight  = binding.weight,
    warning = warning,
  }
end

-- ============================================================
-- Single case evaluation
-- ============================================================

local function eval_case(bindings, case, provider)
  local response = call_provider(provider, case)

  local grades = {}
  local total_score = 0.0
  local total_weight = 0.0

  for _, b in ipairs(bindings) do
    local result = evaluate_binding(b, response, case)
    grades[#grades + 1] = result
    total_score  = total_score + result.score * result.weight
    total_weight = total_weight + result.weight
  end

  local weighted_score = total_weight > 0 and (total_score / total_weight) or 0.0

  return {
    case     = case,
    response = response,
    grades   = grades,
    score    = weighted_score,
    passed   = weighted_score >= 1.0 - 1e-6,
  }
end

-- ============================================================
-- Run: evaluate all cases
-- ============================================================

---@param bindings table[]  Binding[]
---@param cases table[]     Case[]
---@param provider function(input: string) → response
---@return table[] CaseResult[]
function M.run(bindings, cases, provider)
  if type(provider) ~= "function" then
    error("runner.run: provider must be a function", 2)
  end

  local results = {}
  for _, case in ipairs(cases) do
    results[#results + 1] = eval_case(bindings, case, provider)
  end
  return results
end

return M
