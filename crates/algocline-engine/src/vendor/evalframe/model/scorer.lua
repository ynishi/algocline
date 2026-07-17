-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/model/scorer.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  model/scorer.lua — Scorer definition

  Normalizes a Grader's raw grade to [0,1].
  Uniform signature: score(raw_grade) → number [0,1]

  Maps raw_grade → normalized score.

  Usage:
    local scorer = require("evalframe.model.scorer")

    scorer "bool"      { score = function(v) return v and 1.0 or 0.0 end }
    scorer "linear"    { min = 1, max = 5 }
    scorer "threshold" { pass = 0.8 }
]]

local M = {}

local SCORER_TAG = {}

-- ============================================================
-- Internal builder
-- ============================================================

local function build(name, spec)
  local score_fn

  if spec.score then
    -- Custom score function
    if type(spec.score) ~= "function" then
      error(string.format("scorer '%s': 'score' must be function, got %s", name, type(spec.score)), 3)
    end
    score_fn = spec.score

  elseif spec.min ~= nil and spec.max ~= nil then
    -- Linear: maps [min, max] → [0, 1]
    local lo, hi = spec.min, spec.max
    if type(lo) ~= "number" or type(hi) ~= "number" then
      error(string.format("scorer '%s': min/max must be numbers", name), 3)
    end
    if lo == hi then
      error(string.format("scorer '%s': min and max must differ", name), 3)
    end
    local range = hi - lo
    score_fn = function(v)
      if type(v) ~= "number" then return 0.0 end
      local normalized = (v - lo) / range
      return math.max(0.0, math.min(1.0, normalized))
    end

  elseif spec.pass ~= nil then
    -- Threshold: >= pass → 1.0, else 0.0
    local threshold = spec.pass
    if type(threshold) ~= "number" then
      error(string.format("scorer '%s': pass must be number, got %s", name, type(threshold)), 3)
    end
    score_fn = function(v)
      if type(v) == "boolean" then v = v and 1.0 or 0.0 end
      if type(v) ~= "number" then return 0.0 end
      return v >= threshold and 1.0 or 0.0
    end

  elseif spec.steps then
    -- Step-based: list of {threshold, score} pairs
    -- e.g. steps = { {0, 0.0}, {3, 0.5}, {5, 1.0} }
    -- input >= 5 → 1.0, input >= 3 → 0.5, else → 0.0
    local steps = spec.steps
    if type(steps) ~= "table" or #steps < 2 then
      error(string.format("scorer '%s': steps must be table with >= 2 entries", name), 3)
    end
    for i, step in ipairs(steps) do
      if type(step) ~= "table" or type(step[1]) ~= "number" or type(step[2]) ~= "number" then
        error(string.format("scorer '%s': steps[%d] must be {threshold, score}", name, i), 3)
      end
    end
    -- Copy and sort by threshold ascending (avoid mutating caller's table)
    local sorted = {}
    for i, s in ipairs(steps) do sorted[i] = s end
    table.sort(sorted, function(a, b) return a[1] < b[1] end)
    steps = sorted
    score_fn = function(v)
      if type(v) ~= "number" then return 0.0 end
      local result = 0.0
      for _, step in ipairs(steps) do
        if v >= step[1] then result = step[2] end
      end
      return math.max(0.0, math.min(1.0, result))
    end

  else
    error(string.format("scorer '%s': requires 'score', 'min'+'max', 'pass', or 'steps'", name), 3)
  end

  return {
    _tag  = SCORER_TAG,
    name  = name,
    score = score_fn,
  }
end

-- ============================================================
-- Default bool scorer (shared by Binding when no Scorer specified)
-- ============================================================

M.default_bool = build("_bool", {
  score = function(v)
    if type(v) == "boolean" then return v and 1.0 or 0.0 end
    if type(v) == "number" then return math.max(0.0, math.min(1.0, v)) end
    return 0.0
  end,
})

-- ============================================================
-- DSL entry: scorer "name" { ... }
-- ============================================================

setmetatable(M, {
  __call = function(_, name)
    if type(name) ~= "string" then
      error(string.format("scorer: name must be string, got %s", type(name)), 2)
    end
    return function(spec)
      if type(spec) ~= "table" then
        error(string.format("scorer '%s': spec must be a table", name), 2)
      end
      return build(name, spec)
    end
  end,
})

-- ============================================================
-- Introspection
-- ============================================================

---@param v any
---@return boolean
function M.is_scorer(v)
  return type(v) == "table" and v._tag == SCORER_TAG
end

return M
