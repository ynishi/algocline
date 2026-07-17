-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/graders.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  swarm/graders.lua — Swarm-specific grader catalog

  All graders follow the evalframe grader contract:
    check(response, case) → raw_grade

  Where `response` is a SwarmTrace (validated by sw.trace).
  Graders access trace fields directly: resp.success, resp.ticks,
  resp.actions, resp.metrics.

  Usage:
    local sw = require("evalframe.swarm")

    ef.bind { sw.graders.completed }
    ef.bind { sw.graders.efficiency { max_ticks = 20, optimal_ticks = 5 } }
    ef.bind { sw.graders.action_taken("ReadLogs") }
    ef.bind { sw.graders.action_count("Restart", { max = 2 }) }
    ef.bind { sw.graders.all_workers_active { workers = 3 } }
    ef.bind { sw.graders.at_tick(5, function(snap) return snap.action_count >= 2 end) }
]]

local grader = require("evalframe.model.grader")

local M = {}

-- ============================================================
-- Internal: snapshot at a given tick
-- ============================================================

--- Return actions and counts up to (inclusive) the given tick.
---@param resp table  SwarmTrace
---@param tick number  Tick number
---@return table { actions, action_count, action_counts }
local function snapshot_at_tick(resp, tick)
  local filtered = {}
  local counts = {}

  for _, a in ipairs(resp.actions) do
    if a.tick <= tick then
      filtered[#filtered + 1] = a
      counts[a.action] = (counts[a.action] or 0) + 1
    end
  end

  return {
    actions       = filtered,
    action_count  = #filtered,
    action_counts = counts,
  }
end

-- ============================================================
-- completed: success == true
-- ============================================================

M.completed = grader "sw.completed" {
  check = function(resp, _case)
    return resp.success == true
  end,
}

-- ============================================================
-- efficiency: tick-based linear scoring
-- ============================================================

---@param opts table { max_ticks: number, optimal_ticks?: number }
---@return table GraderDef
function M.efficiency(opts)
  if type(opts) ~= "table" then
    error("sw.graders.efficiency: spec must be a table", 2)
  end
  if opts.max_ticks == nil then
    error("sw.graders.efficiency: max_ticks is required", 2)
  end

  local max_t = opts.max_ticks
  local opt_t = opts.optimal_ticks or 0
  if opt_t >= max_t then
    error("sw.graders.efficiency: optimal_ticks must be less than max_ticks", 2)
  end

  return grader(string.format("sw.efficiency(max=%d,opt=%d)", max_t, opt_t)) {
    check = function(resp, _case)
      local ticks = resp.ticks
      if ticks <= opt_t then return 1.0 end
      if ticks >= max_t then return 0.0 end
      return 1.0 - (ticks - opt_t) / (max_t - opt_t)
    end,
  }
end

-- ============================================================
-- action_taken: check if action was ever executed
-- ============================================================

---@param action_name string
---@return table GraderDef
function M.action_taken(action_name)
  return grader("sw.action_taken:" .. action_name) {
    check = function(resp, _case)
      for _, a in ipairs(resp.actions) do
        if a.action == action_name then return true end
      end
      return false
    end,
  }
end

-- ============================================================
-- action_sequence: check if actions appeared in order
-- ============================================================

---@param sequence string[]
---@return table GraderDef
function M.action_sequence(sequence)
  if type(sequence) ~= "table" or #sequence < 1 then
    error("sw.graders.action_sequence: at least 1 action required", 2)
  end

  local name = "sw.action_sequence:" .. table.concat(sequence, ",")

  return grader(name) {
    check = function(resp, _case)
      local seq_idx = 1
      for _, a in ipairs(resp.actions) do
        if a.action == sequence[seq_idx] then
          seq_idx = seq_idx + 1
          if seq_idx > #sequence then return true end
        end
      end
      return false
    end,
  }
end

-- ============================================================
-- action_count: threshold-based action count check
-- ============================================================

---@param action_name string
---@param opts table { min?: number, max?: number }
---@return table GraderDef
function M.action_count(action_name, opts)
  if type(opts) ~= "table" then
    error("sw.graders.action_count: opts must be a table", 2)
  end
  if opts.min == nil and opts.max == nil then
    error("sw.graders.action_count: min or max is required", 2)
  end

  return grader("sw.action_count:" .. action_name) {
    check = function(resp, _case)
      local count = 0
      for _, a in ipairs(resp.actions) do
        if a.action == action_name then count = count + 1 end
      end
      if opts.min ~= nil and count < opts.min then return false end
      if opts.max ~= nil and count > opts.max then return false end
      return true
    end,
  }
end

-- ============================================================
-- all_workers_active: verify minimum distinct worker participation
--
-- opts.workers (number|nil):
--   When given, requires at least that many distinct workers.
--   When omitted, defaults to >= 2 (multi-agent collaboration).
-- ============================================================

---@param opts table|nil  { workers?: number }
---@return table GraderDef
function M.all_workers_active(opts)
  local min_workers = (opts and opts.workers) or 2

  return grader(string.format("sw.all_workers_active(min=%d)", min_workers)) {
    check = function(resp, _case)
      if #resp.actions == 0 then return false end

      local workers = {}
      for _, a in ipairs(resp.actions) do
        workers[a.worker] = true
      end

      local count = 0
      for _ in pairs(workers) do count = count + 1 end
      return count >= min_workers
    end,
  }
end

-- ============================================================
-- metric: threshold-based metric check
-- ============================================================

---@param metric_name string
---@param opts table { min?: number, max?: number }
---@return table GraderDef
function M.metric(metric_name, opts)
  if type(opts) ~= "table" then
    error("sw.graders.metric: opts must be a table", 2)
  end
  if opts.min == nil and opts.max == nil then
    error("sw.graders.metric: min or max is required", 2)
  end

  return grader("sw.metric:" .. metric_name) {
    check = function(resp, _case)
      local val = resp.metrics and resp.metrics[metric_name]
      if val == nil then return false end
      if opts.min ~= nil and val < opts.min then return false end
      if opts.max ~= nil and val > opts.max then return false end
      return true
    end,
  }
end

-- ============================================================
-- at_tick: post-hoc checkpoint at specific tick
-- ============================================================

---@param tick number
---@param check_fn function(snapshot) -> boolean
---@return table GraderDef
function M.at_tick(tick, check_fn)
  return grader(string.format("sw.at_tick(%d)", tick)) {
    check = function(resp, _case)
      local snap = snapshot_at_tick(resp, tick)
      return check_fn(snap)
    end,
  }
end

-- ============================================================
-- after_action: check state after first occurrence of action
-- ============================================================

---@param action_name string
---@param check_fn function(snapshot) -> boolean
---@return table GraderDef
function M.after_action(action_name, check_fn)
  return grader("sw.after_action:" .. action_name) {
    check = function(resp, _case)
      local target_tick
      for _, a in ipairs(resp.actions) do
        if a.action == action_name then
          target_tick = a.tick
          break
        end
      end
      if target_tick == nil then return false end

      local snap = snapshot_at_tick(resp, target_tick)
      return check_fn(snap)
    end,
  }
end

return M
