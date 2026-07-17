-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/eval/init.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  eval/init.lua — Suite: eval orchestrator

  Composes Cases + Bindings + Provider into a runnable eval suite.

  Usage:
    local suite = require("evalframe.eval")

    local s = suite "math_eval" {
      provider = my_provider,
      bind { graders.exact_match },
      bind { graders.contains },
      cases = { case { input = "2+2?", expected = "4" } },
    }

    local report = s:run()
    print(report:summary())
]]

local Binding = require("evalframe.model.binding")
local Case    = require("evalframe.model.case")
local runner  = require("evalframe.eval.runner")
local stats   = require("evalframe.eval.stats")
local report  = require("evalframe.eval.report")

local M = {}

local Suite = {}
Suite.__index = Suite

-- ============================================================
-- Suite constructor
-- ============================================================

local function build_suite(name, spec)
  if type(spec) ~= "table" then
    error(string.format("suite '%s': spec must be a table", name), 3)
  end

  local self = setmetatable({}, Suite)
  self.name = name

  -- Extract provider
  self.provider = spec.provider
  if type(self.provider) ~= "function" then
    error(string.format("suite '%s': provider must be a function", name), 3)
  end

  -- Extract bindings (positional Binding entries in spec)
  self.bindings = {}
  for i, v in ipairs(spec) do
    if Binding.is_binding(v) then
      self.bindings[#self.bindings + 1] = v
    else
      error(string.format(
        "suite '%s': positional arg [%d] must be a Binding (use ef.bind { ... }), got %s",
        name, i, type(v)
      ), 3)
    end
  end
  if #self.bindings == 0 then
    error(string.format("suite '%s': at least one binding required", name), 3)
  end

  -- Extract cases
  self.cases = {}
  if spec.cases then
    if type(spec.cases) ~= "table" then
      error(string.format("suite '%s': cases must be a table", name), 3)
    end
    for i, c in ipairs(spec.cases) do
      if Case.is_case(c) then
        self.cases[#self.cases + 1] = c
      elseif type(c) == "table" and c.input then
        -- Auto-wrap raw tables as Cases
        self.cases[#self.cases + 1] = Case.new(c)
      else
        error(string.format(
          "suite '%s': cases[%d] is not a Case or case spec (got %s)",
          name, i, type(c)
        ), 3)
      end
    end
  end

  return self
end

-- ============================================================
-- DSL entry: suite "name" { ... }
-- ============================================================

setmetatable(M, {
  __call = function(_, name)
    if type(name) ~= "string" then
      error(string.format("suite: name must be string, got %s", type(name)), 2)
    end
    return function(spec)
      return build_suite(name, spec)
    end
  end,
})

-- ============================================================
-- Run
-- ============================================================

---@return table { results, aggregated, summary() }
function Suite:run()
  if #self.cases == 0 then
    error(string.format("suite '%s': no cases to evaluate", self.name), 2)
  end

  local results = runner.run(self.bindings, self.cases, self.provider)
  local agg = stats.aggregate(results)
  local suite_name = self.name  -- capture for closures below

  local r = {
    name       = suite_name,
    results    = results,
    aggregated = agg,
    failures   = report.failures(results),

    summary = function()
      return report.summary(agg, { name = suite_name })
    end,

    format_result = report.format_result,
  }

  --- Return a serialization-safe copy (no functions, no cycles).
  --- Includes summary text and strips response.raw to avoid bloat.
  function r:to_table()
    local function strip(t, seen)
      if type(t) ~= "table" then return t end
      seen = seen or {}
      if seen[t] then return nil end
      seen[t] = true
      local out = {}
      for k, v in pairs(t) do
        if type(v) ~= "function" then
          out[k] = strip(v, seen)
        end
      end
      return out
    end

    return {
      name       = self.name,
      aggregated = strip(self.aggregated),
      failures   = strip(self.failures),
      results    = strip(self.results),
      summary    = self:summary(),
    }
  end

  return r
end

return M
