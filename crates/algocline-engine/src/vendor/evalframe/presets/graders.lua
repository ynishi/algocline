-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/presets/graders.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  presets/graders.lua — Built-in grader catalog

  Deterministic graders (Tier 1 — always prefer over LLM-as-Judge).

  Usage:
    local graders = require("evalframe.presets.graders")
    bind { graders.exact_match, scorer_def }
]]

local grader = require("evalframe.model.grader")
local std    = require("evalframe.std")

local M = {}

-- ============================================================
-- Exact match: response.text == any expected value
-- ============================================================

M.exact_match = grader "exact_match" {
  check = function(resp, case)
    if not case.expected then return false end
    local text = resp.text or ""
    for _, exp in ipairs(case.expected) do
      if text == exp then return true end
    end
    return false
  end,

}

-- ============================================================
-- Contains: response.text contains any expected value
-- ============================================================

M.contains = grader "contains" {
  check = function(resp, case)
    if not case.expected then return false end
    local text = resp.text or ""
    for _, exp in ipairs(case.expected) do
      if text:find(exp, 1, true) then return true end
    end
    return false
  end,

}

-- ============================================================
-- Starts with: response.text starts with expected
-- ============================================================

M.starts_with = grader "starts_with" {
  check = function(resp, case)
    if not case.expected then return false end
    local text = resp.text or ""
    for _, exp in ipairs(case.expected) do
      if text:sub(1, #exp) == exp then return true end
    end
    return false
  end,

}

-- ============================================================
-- Regex: response.text matches Lua pattern
-- Expects case.context.pattern or case.expected[1] as pattern
-- ============================================================

M.regex = grader "regex" {
  check = function(resp, case)
    local text = resp.text or ""
    local pattern = (case.context and case.context.pattern)
                    or (case.expected and case.expected[1])
    if not pattern then return false end
    return text:match(pattern) ~= nil
  end,

}

-- ============================================================
-- JSON valid: response.text is valid JSON
-- ============================================================

M.json_valid = grader "json_valid" {
  check = function(resp, _case)
    local text = resp.text or ""
    local ok = pcall(std.json.decode, text)
    return ok
  end,

}

-- ============================================================
-- Length: returns text length as raw grade (pair with linear scorer)
-- ============================================================

M.length = grader "length" {
  check = function(resp, _case)
    return #(resp.text or "")
  end,

}

-- ============================================================
-- Latency: returns response latency in ms
-- ============================================================

M.latency = grader "latency" {
  check = function(resp, _case)
    return resp.latency_ms  -- nil when missing (scorer handles nil → 0)
  end,

}

-- ============================================================
-- Not empty: response.text is non-empty
-- ============================================================

M.not_empty = grader "not_empty" {
  check = function(resp, _case)
    local text = resp.text or ""
    return #text > 0
  end,

}

-- ============================================================
-- Composed graders (combinators)
-- ============================================================

--- Non-empty AND valid JSON. Common precondition for structured output.
M.valid_json_response = grader.all(M.not_empty, M.json_valid)

--- Exact match OR contains. Lenient matching that accepts both.
M.flexible_match = grader.any(M.exact_match, M.contains)

return M
