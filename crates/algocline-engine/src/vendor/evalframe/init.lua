-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/init.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  evalframe — LLM Evaluation DSL

  Usage:
    local ef = require("evalframe")

    -- With Claude Code CLI provider
    local s = ef.suite "my_eval" {
      provider = ef.providers.claude_cli(),
      ef.bind { ef.graders.exact_match },
      cases = { ef.case { input = "2+2?", expected = "4" } },
    }
    local report = s:run()
    print(report:summary())

    -- With LLM-as-Judge (provider must be specified)
    local provider = ef.providers.claude_cli { model = "sonnet" }
    local s = ef.suite "quality_eval" {
      provider = provider,
      ef.bind { ef.llm_graders.rubric("Rate accuracy 1-5", { provider = provider }), ef.scorers.linear_1_5 },
      cases = { ... },
    }
]]

local M = {}

-- Hub metadata (consumed by alc_hub_reindex / alc_hub_dist to populate
-- hub_index.json so that `alc_hub_search` can discover this package).
M.meta = {
  name = "evalframe",
  version = "0.4.0",
  description = "Lua DSL for evaluating LLM outputs — declarative grading pipeline (suite / case / grader / scorer / provider) usable from any host with mlua.",
  category = "eval",
}

-- Pipeline
M.suite      = require("evalframe.eval")
M.report     = require("evalframe.eval.report")
M.stats      = require("evalframe.eval.stats")
M.load_cases = require("evalframe.eval.loader").load_file

-- DSL constructors
M.case    = require("evalframe.model.case")
M.grader  = require("evalframe.model.grader")
M.scorer  = require("evalframe.model.scorer")
M.bind    = require("evalframe.model.binding")

-- Parametric variations
local _variants = require("evalframe.variants")
M.variants = _variants.generate
M.vary     = _variants.vary

-- Preset catalogs
M.graders     = require("evalframe.presets.graders")
M.scorers     = require("evalframe.presets.scorers")
M.llm_graders = require("evalframe.presets.llm_graders")

-- Providers
M.providers = {
  claude_cli = require("evalframe.providers.claude_cli"),
  mock       = require("evalframe.providers.mock"),
}

-- Conditional: algocline provider (only when running inside algocline VM)
if type(alc) == "table" and type(alc.llm) == "function" then
  local ok, mod = pcall(require, "evalframe.providers.algocline")
  if ok then M.providers.algocline = mod end
end

-- Stdlib (json, fs, time)
M.std = require("evalframe.std")

return M
