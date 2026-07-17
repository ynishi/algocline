-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/trace.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  swarm/trace.lua — SwarmTrace construction and validation

  SwarmTrace extends the evalframe response contract with Swarm-specific
  fields (actions, metrics, termination, etc.).

  Graders and analysis functions access trace fields directly:
    resp.success, resp.ticks, resp.actions, resp.metrics, resp.termination

  The trace structure is validated at construction time by sw.trace(),
  so downstream consumers can rely on field presence and types.

  Usage:
    local sw = require("evalframe.swarm")

    local t = sw.trace {
      text = "done", success = true, ticks = 10,
      termination = "success",
      actions = { { tick = 1, worker = "w-0", action = "Act", result = "ok" } },
      metrics = { task_completion = 1.0 },
    }

    -- Direct field access (validated by build):
    t.success      -- boolean
    t.ticks        -- number
    t.actions      -- action records in tick order
    t.metrics      -- { [name] = value }
    t.termination  -- "success" | "failure" | "timeout"
]]

local M = {}

local TRACE_TAG = {}

local VALID_TERMINATIONS = {
  success = true,
  failure = true,
  timeout = true,
}

-- ============================================================
-- Construction
-- ============================================================

--- Shallow-copy an array table.
local function copy_list(t)
  local out = {}
  for i, v in ipairs(t) do out[i] = v end
  return out
end

--- Shallow-copy a hash table.
local function copy_table(t)
  local out = {}
  for k, v in pairs(t) do out[k] = v end
  return out
end

--- Validate a single action record.
local function validate_action_record(a, idx)
  if type(a) ~= "table" then
    error(string.format("sw.trace: actions[%d] must be a table", idx), 3)
  end
  if type(a.tick) ~= "number" then
    error(string.format("sw.trace: actions[%d].tick must be number, got %s", idx, type(a.tick)), 3)
  end
  if type(a.worker) ~= "string" then
    error(string.format("sw.trace: actions[%d].worker must be string, got %s", idx, type(a.worker)), 3)
  end
  if type(a.action) ~= "string" then
    error(string.format("sw.trace: actions[%d].action must be string, got %s", idx, type(a.action)), 3)
  end
end

function M.build(raw)
  if type(raw) ~= "table" then
    error("sw.trace: must be a table", 2)
  end

  -- actions (required)
  if raw.actions == nil then
    error("sw.trace: actions is required", 2)
  end
  if type(raw.actions) ~= "table" then
    error("sw.trace: actions must be a table", 2)
  end
  for i, a in ipairs(raw.actions) do
    validate_action_record(a, i)
  end

  -- termination (required, enum)
  if raw.termination == nil then
    error("sw.trace: termination is required", 2)
  end
  if not VALID_TERMINATIONS[raw.termination] then
    error(string.format(
      "sw.trace: termination must be 'success', 'failure', or 'timeout', got '%s'",
      tostring(raw.termination)
    ), 2)
  end

  -- Defensive copy to prevent caller mutation.
  local copied_actions = copy_list(raw.actions)
  local copied_metrics = type(raw.metrics) == "table" and copy_table(raw.metrics) or {}

  return {
    _tag        = TRACE_TAG,
    text        = type(raw.text) == "string" and raw.text or "",
    success     = raw.success == true,
    ticks       = type(raw.ticks) == "number" and raw.ticks or 0,
    termination = raw.termination,
    actions     = copied_actions,
    metrics     = copied_metrics,
    latency_ms  = type(raw.latency_ms) == "number" and raw.latency_ms or nil,
  }
end

function M.is_trace(v)
  return type(v) == "table" and v._tag == TRACE_TAG
end

return M
