-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/model/binding.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  model/binding.lua — Binding: Grader x Scorer pair (type-dispatched)

  Pairs "what to check" (Grader) x "how to score" (Scorer).
  Positional args are type-dispatched (order-independent).
  grader.name serves as the natural key.

  Usage:
    local bind   = require("evalframe.model.binding")
    local grader = require("evalframe.model.grader")
    local scorer = require("evalframe.model.scorer")

    bind { grader_def, scorer_def }
    bind { scorer_def, grader_def }  -- order-independent
    bind "label" { grader_def, scorer_def }

    -- Grader-only (uses default bool scorer)
    bind { grader_def }

    -- With weight override
    bind { grader_def, scorer_def, weight = 1.0 }
]]

local Grader = require("evalframe.model.grader")
local Scorer = require("evalframe.model.scorer")

local M = {}

local BINDING_TAG = {}

-- ============================================================
-- Internal builder
-- ============================================================

local function build(spec, label)
  label = label or "bind"

  if type(spec) ~= "table" then
    error(string.format("%s: spec must be a table", label), 3)
  end

  local grd, scr
  for _, v in ipairs(spec) do
    if Grader.is_grader(v) then
      if grd then error(string.format("%s: multiple GraderDef provided", label), 3) end
      grd = v
    elseif Scorer.is_scorer(v) then
      if scr then error(string.format("%s: multiple ScorerDef provided", label), 3) end
      scr = v
    end
  end

  if not grd then
    error(string.format("%s: GraderDef required", label), 3)
  end

  -- Default to bool scorer if none provided
  scr = scr or Scorer.default_bool

  -- Optional weight (default 1.0)
  local weight = spec.weight or 1.0
  if type(weight) ~= "number" or weight < 0 then
    error(string.format("%s: weight must be non-negative number, got %s", label, tostring(weight)), 3)
  end

  return {
    _tag    = BINDING_TAG,
    grader  = grd,
    scorer  = scr,
    weight  = weight,
  }
end

-- ============================================================
-- DSL entry:
--   bind { GraderDef, ScorerDef }
--   bind "label" { GraderDef, ScorerDef }
-- ============================================================

setmetatable(M, {
  __call = function(_, first)
    if type(first) == "table" then
      return build(first, "bind")
    elseif type(first) == "string" then
      return function(spec)
        return build(spec, string.format("bind '%s'", first))
      end
    else
      error(string.format("bind: expected table or string, got %s", type(first)), 2)
    end
  end,
})

-- ============================================================
-- Introspection
-- ============================================================

--- Natural key: grader.name
---@param b table Binding
---@return string
function M.key(b)
  return b.grader.name
end

---@param v any
---@return boolean
function M.is_binding(v)
  return type(v) == "table" and v._tag == BINDING_TAG
end

return M
