-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/providers/algocline.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  providers/algocline.lua — algocline providers

  Runs algocline strategy packages or direct alc.llm() as evalframe providers.
  Only works inside algocline's mlua VM (requires `alc` global).

  Usage:
    local ef = require("evalframe")

    -- Run "reflect" strategy on each case input
    local provider = ef.providers.algocline { strategy = "reflect" }

    -- With strategy-specific options
    local provider = ef.providers.algocline {
      strategy = "ucb",
      opts = { rounds = 5 },
    }

    -- Direct LLM access (for LLM-as-Judge graders)
    local judge = ef.providers.algocline.llm()

    -- Use in suite with LLM-as-Judge
    local s = ef.suite "eval_reflect" {
      provider = ef.providers.algocline { strategy = "reflect" },
      ef.bind {
        ef.llm_graders.rubric("Rate accuracy 1-5", { provider = judge }),
        ef.scorers.linear_1_5,
      },
      cases = { ef.case { input = "Review this code...", expected = { "..." } } },
    }
]]

local std = require("evalframe.std")

local M = {}

--- Extract text from a strategy result.
---@param result any  Strategy return value (typically ctx table with .result)
---@return string
local function extract_text(result)
  if type(result) == "string" then
    return result
  end
  if type(result) ~= "table" then
    return tostring(result)
  end
  -- ctx.result is the conventional output field
  local r = result.result
  if r == nil then
    -- Fallback: try encoding the whole result
    if alc and alc.json_encode then
      return alc.json_encode(result)
    end
    return tostring(result)
  end
  if type(r) == "string" then
    return r
  end
  if type(r) == "table" then
    -- Nested result: try .answer, .summary, .output, .text
    for _, key in ipairs({ "answer", "summary", "output", "text" }) do
      if type(r[key]) == "string" then
        return r[key]
      end
    end
    if alc and alc.json_encode then
      return alc.json_encode(r)
    end
  end
  return tostring(r)
end

-- ============================================================
-- Direct LLM provider (for LLM-as-Judge graders)
-- ============================================================

--- Create a direct alc.llm() provider.
--- Calls alc.llm() with the input prompt and returns the response.
--- Use this as the provider for llm_graders (rubric, yes_no, factuality).
---@param opts? table  Reserved for future options
---@return function provider(input) → response
function M.llm(opts)
  opts = opts or {}

  if type(alc) ~= "table" or type(alc.llm) ~= "function" then
    error("algocline.llm provider: requires algocline VM (alc global not found)", 2)
  end

  return function(input)
    local start = std.time()
    -- Call alc.llm() directly (no pcall wrapper).
    -- Reason: alc.llm() yields via coroutine. grader.lua's safe_check
    -- already wraps the grader in pcall, providing error protection.
    -- Adding a second pcall here is redundant and may interfere with
    -- mlua-isle's yield propagation through nested pcall boundaries.
    local text = alc.llm(input)
    local elapsed = (std.time() - start) * 1000

    return {
      text       = type(text) == "string" and text or tostring(text),
      model      = "alc_llm",
      latency_ms = elapsed,
    }
  end
end

-- ============================================================
-- Strategy provider
-- ============================================================

--- Create an algocline strategy provider.
---@param opts table  { strategy: string, opts?: table }
---@return function provider(input) → response
setmetatable(M, {
  __call = function(_, opts)
    opts = opts or {}

    local strategy_name = opts.strategy
    if not strategy_name or type(strategy_name) ~= "string" then
      error("algocline provider: 'strategy' must be a string", 2)
    end

    -- Verify alc global is available (running inside algocline VM)
    if type(alc) ~= "table" or type(alc.llm) ~= "function" then
      error("algocline provider: requires algocline VM (alc global not found)", 2)
    end

    local strategy_opts = opts.opts or {}

    return function(input)
      local strategy = require(strategy_name)
      if type(strategy.run) ~= "function" then
        error(string.format(
          "algocline provider: package '%s' has no run() function", strategy_name
        ), 2)
      end

      -- Build ctx: task = input, merge strategy opts
      local run_ctx = { task = input }
      for k, v in pairs(strategy_opts) do
        run_ctx[k] = v
      end

      local start = std.time()
      local ok, result = pcall(strategy.run, run_ctx)
      local elapsed = (std.time() - start) * 1000

      if not ok then
        return {
          text       = "",
          model      = "algocline:" .. strategy_name,
          error      = tostring(result),
          latency_ms = elapsed,
        }
      end

      local text = extract_text(result)
      return {
        text       = text,
        model      = "algocline:" .. strategy_name,
        latency_ms = elapsed,
        raw        = result,
      }
    end
  end,
})

return M
