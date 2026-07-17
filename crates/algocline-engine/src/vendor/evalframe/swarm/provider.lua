-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/provider.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  swarm/provider.lua — Runner → evalframe Provider adapter

  Wraps a user-provided runner function into the evalframe provider contract.
  The runner receives a config table and returns a SwarmTrace-conformant table.

  Usage:
    local sw = require("evalframe.swarm")

    local provider = sw.provider(runner, {
      env     = env,
      actions = actions,
      swarm   = swarm_cfg,
    })

    -- provider(input) → SwarmTrace (extends response contract)
]]

local env_mod     = require("evalframe.swarm.env")
local actions_mod = require("evalframe.swarm.actions")
local config_mod  = require("evalframe.swarm.config")
local trace_mod   = require("evalframe.swarm.trace")
local std         = require("evalframe.std")

local M = {}

---@param runner function(config: table) → table  (returns raw SwarmTrace)
---@param opts table  { env, actions, swarm }
---@return function provider(input: string) → SwarmTrace
function M.build(runner, opts)
  if type(runner) ~= "function" then
    error("sw.provider: runner must be a function", 2)
  end

  if type(opts) ~= "table" then
    error("sw.provider: opts must be a table", 2)
  end

  if not env_mod.is_env(opts.env) then
    error("sw.provider: env is required (use sw.env)", 2)
  end
  if not actions_mod.is_action_space(opts.actions) then
    error("sw.provider: actions is required (use sw.actions)", 2)
  end
  if not config_mod.is_swarm_config(opts.swarm) then
    error("sw.provider: swarm is required (use sw.swarm)", 2)
  end

  return function(input)
    local config = {
      input   = input,
      env     = opts.env,
      actions = opts.actions,
      swarm   = opts.swarm,
    }

    local start = std.time()
    local ok, raw = pcall(runner, config)
    local elapsed = (std.time() - start) * 1000

    if not ok then
      return {
        text       = "",
        latency_ms = elapsed,
        error      = tostring(raw),
      }
    end

    -- Set latency_ms fallback without mutating runner's return value
    local trace_input = raw
    if raw.latency_ms == nil then
      trace_input = {}
      for k, v in pairs(raw) do trace_input[k] = v end
      trace_input.latency_ms = elapsed
    end

    -- Validate and wrap as SwarmTrace
    return trace_mod.build(trace_input)
  end
end

return M
