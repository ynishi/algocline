-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/model/grader.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  model/grader.lua — Grader definition

  Evaluates an LLM response against a case.
  Uniform signature: check(response, case) → raw_grade

  raw_grade is intentionally untyped:
    - bool:   pass/fail (exact_match, contains, regex)
    - number: rating scale (llm_judge 1-5, similarity 0-1)
    - string: extracted text (for downstream analysis)

  The Scorer (paired via Binding) normalizes raw_grade to [0,1].
  Graders that return bool get default scorer: true→1.0, false→0.0.

  Usage:
    local grader = require("evalframe.model.grader")

    grader "exact_match" {
      check = function(resp, case)
        return resp.text == case.expected[1]
      end
    }

    grader "rating" {
      check = function(resp, case)
        return tonumber(resp.text:match("%d")) or 0
      end
    }
]]

local M = {}

local GRADER_TAG = {}

-- ============================================================
-- Internal: safe check wrapper
-- ============================================================

local function safe_check(fn)
  return function(response, case)
    local ok, val, err = pcall(fn, response, case)
    if ok then return val, err end
    return nil, tostring(val)
  end
end

-- ============================================================
-- Internal builder
-- ============================================================

local function build(name, spec)
  if type(spec.check) ~= "function" then
    error(string.format("grader '%s': 'check' must be function, got %s", name, type(spec.check)), 3)
  end

  return {
    _tag  = GRADER_TAG,
    name  = name,
    check = safe_check(spec.check),
  }
end

-- ============================================================
-- DSL entry: grader "name" { check = fn }
-- ============================================================

setmetatable(M, {
  __call = function(_, name)
    if type(name) ~= "string" then
      error(string.format("grader: name must be string, got %s", type(name)), 2)
    end
    return function(spec)
      if type(spec) ~= "table" then
        error(string.format("grader '%s': spec must be a table", name), 2)
      end
      return build(name, spec)
    end
  end,
})

-- ============================================================
-- Combinators: compose graders
-- ============================================================

--- All graders must pass (AND).
--- Returns true only if every grader returns truthy. Otherwise false.
--- For numeric graders, returns the minimum value.
---@param ... table GraderDef[]
---@return table GraderDef
function M.all(...)
  local graders = { ... }
  local names = {}
  for _, g in ipairs(graders) do
    if not M.is_grader(g) then
      error("Grader.all: all arguments must be GraderDef", 2)
    end
    names[#names + 1] = g.name
  end

  return {
    _tag  = GRADER_TAG,
    name  = "all(" .. table.concat(names, ",") .. ")",
    check = safe_check(function(response, case)
      local min_val = nil
      local all_bool = true
      for _, g in ipairs(graders) do
        local val, err = g.check(response, case)
        if err then return nil, err end
        if not val then return false end
        if type(val) == "number" then
          all_bool = false
          if min_val == nil or val < min_val then min_val = val end
        elseif type(val) ~= "boolean" then
          all_bool = false
        end
      end
      if all_bool then return true end
      return min_val ~= nil and min_val or true
    end),
  }
end

--- Any grader passes (OR).
--- Returns the first truthy result. For numeric graders, returns the maximum.
---@param ... table GraderDef[]
---@return table GraderDef
function M.any(...)
  local graders = { ... }
  local names = {}
  for _, g in ipairs(graders) do
    if not M.is_grader(g) then
      error("Grader.any: all arguments must be GraderDef", 2)
    end
    names[#names + 1] = g.name
  end

  return {
    _tag  = GRADER_TAG,
    name  = "any(" .. table.concat(names, ",") .. ")",
    check = safe_check(function(response, case)
      local last_err
      for _, g in ipairs(graders) do
        local val, err = g.check(response, case)
        if not err and val then return val end
        last_err = err
      end
      return false, last_err
    end),
  }
end

-- ============================================================
-- Introspection
-- ============================================================

---@param v any
---@return boolean
function M.is_grader(v)
  return type(v) == "table" and v._tag == GRADER_TAG
end

return M
