-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/config.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  swarm/config.lua — Swarm configuration declaration

  Fixed schema: workers, max_ticks, managers, strategy.
  Unknown fields are rejected at construction time.

  Usage:
    local sw = require("evalframe.swarm")

    local cfg = sw.swarm {
      workers   = 3,
      managers  = 1,
      max_ticks = 20,
      strategy  = "ucb1",
    }
]]

local M = {}

local SWARM_CONFIG_TAG = {}

local KNOWN_FIELDS = {
  workers   = true,
  max_ticks = true,
  managers  = true,
  strategy  = true,
}

function M.build(spec)
  if type(spec) ~= "table" then
    error("sw.swarm: spec must be a table", 2)
  end

  -- Reject unknown fields
  for k in pairs(spec) do
    if not KNOWN_FIELDS[k] then
      error(string.format("sw.swarm: unknown field '%s'", k), 2)
    end
  end

  -- workers (required, positive integer)
  if spec.workers == nil then
    error("sw.swarm: workers is required", 2)
  end
  if type(spec.workers) ~= "number" or spec.workers < 1 or spec.workers ~= math.floor(spec.workers) then
    error("sw.swarm: workers must be a positive integer", 2)
  end

  -- max_ticks (required, positive integer)
  if spec.max_ticks == nil then
    error("sw.swarm: max_ticks is required", 2)
  end
  if type(spec.max_ticks) ~= "number" or spec.max_ticks < 1 or spec.max_ticks ~= math.floor(spec.max_ticks) then
    error("sw.swarm: max_ticks must be a positive integer", 2)
  end

  -- managers (optional, positive integer, default 1)
  local managers = spec.managers or 1
  if type(managers) ~= "number" or managers < 1 or managers ~= math.floor(managers) then
    error("sw.swarm: managers must be a positive integer", 2)
  end

  -- strategy (optional, string)
  if spec.strategy ~= nil and type(spec.strategy) ~= "string" then
    error(string.format("sw.swarm: strategy must be string, got %s", type(spec.strategy)), 2)
  end

  return {
    _tag      = SWARM_CONFIG_TAG,
    workers   = spec.workers,
    max_ticks = spec.max_ticks,
    managers  = managers,
    strategy  = spec.strategy,
  }
end

function M.is_swarm_config(v)
  return type(v) == "table" and v._tag == SWARM_CONFIG_TAG
end

return M
