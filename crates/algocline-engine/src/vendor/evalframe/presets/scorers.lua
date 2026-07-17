-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/presets/scorers.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  presets/scorers.lua — Built-in scorer catalog

  Usage:
    local scorers = require("evalframe.presets.scorers")
    bind { grader_def, scorers.linear_1_5 }
]]

local scorer = require("evalframe.model.scorer")

local M = {}

-- ============================================================
-- Bool: true → 1.0, false → 0.0
-- Re-exports scorer.default_bool to avoid duplicate logic.
-- ============================================================

M.bool = scorer.default_bool

-- ============================================================
-- Linear scales (common LLM judge ratings)
-- ============================================================

M.linear_1_5 = scorer "linear_1_5" { min = 1, max = 5 }
M.linear_1_10 = scorer "linear_1_10" { min = 1, max = 10 }
M.linear_0_100 = scorer "linear_0_100" { min = 0, max = 100 }

-- ============================================================
-- Thresholds
-- ============================================================

M.pass_50 = scorer "pass_50" { pass = 0.5 }
M.pass_80 = scorer "pass_80" { pass = 0.8 }

-- ============================================================
-- Inverse: lower is better (e.g., latency, token count)
-- ============================================================

M.inverse_linear = scorer "inverse_linear" {
  score = function(v)
    if type(v) ~= "number" or v <= 0 then return 1.0 end
    return 1.0 / (1.0 + v / 1000.0)
  end,
}

-- ============================================================
-- Step-based: non-linear band scoring for LLM judge ratings
-- ============================================================

--- Band scoring for 1-5 scale: 1-2→0.0, 3→0.5, 4-5→1.0
M.band_1_5 = scorer "band_1_5" {
  steps = { {1, 0.0}, {3, 0.5}, {4, 1.0} },
}

return M
