-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/analysis.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  swarm/analysis.lua — Multi-trace aggregate analysis

  Post-hoc analysis functions that operate on collections of SwarmTraces.
  These are NOT graders (which score a single trace at eval time).
  Analysis functions compute aggregate statistics across multiple runs.

  All functions access trace fields directly (resp.actions, resp.success,
  resp.ticks) — the trace structure is validated at construction time.

  Usage:
    local sw       = require("evalframe.swarm")
    local analysis = sw.analysis

    local freq  = analysis.action_sequences(traces, 3)
    local conv  = analysis.convergence(traces)
    local eff   = analysis.exploration_efficiency(trace)
    local coord = analysis.worker_coordination(trace)
    local qual  = analysis.action_validity(trace, function(a)
      return a.result ~= "parse_error"
    end)
]]

local stats = require("evalframe.eval.stats")

local M = {}

-- ============================================================
-- Action sequence frequency analysis
--
-- Extract n-gram action sequences from traces, count frequency,
-- and compute success rate per pattern.
--
-- Counting is per-trace (deduplicated): each n-gram is counted
-- at most once per trace to avoid bias from traces with repeated
-- action patterns. The success rate therefore represents "fraction
-- of traces containing this n-gram that succeeded".
--
-- Returns: { [sequence_key] = { count, success, rate } }
-- ============================================================

---@param traces table[]  Array of SwarmTrace
---@param ngram_size number  Length of action n-grams (default 3)
---@return table  { [key] = { count, success, rate } }
function M.action_sequences(traces, ngram_size)
  ngram_size = ngram_size or 3
  if ngram_size < 1 then
    error("analysis.action_sequences: ngram_size must be >= 1", 2)
  end
  local sequences = {}

  for _, trace in ipairs(traces) do
    local actions = trace.actions
    local succeeded = trace.success == true

    -- Collect unique n-grams within this trace
    local seen_in_trace = {}
    for i = 1, #actions - ngram_size + 1 do
      local parts = {}
      for j = 0, ngram_size - 1 do
        parts[#parts + 1] = actions[i + j].action
      end
      local key = table.concat(parts, ",")

      if not seen_in_trace[key] then
        seen_in_trace[key] = true
        if not sequences[key] then
          sequences[key] = { count = 0, success = 0 }
        end
        sequences[key].count = sequences[key].count + 1
        if succeeded then
          sequences[key].success = sequences[key].success + 1
        end
      end
    end
  end

  for _, data in pairs(sequences) do
    data.rate = data.count > 0 and data.success / data.count or 0
  end

  return sequences
end

-- ============================================================
-- Convergence analysis
--
-- Distribution of tick counts to resolution, or tick of first
-- occurrence of a target action across traces.
--
-- Returns: stats.describe_with_ci result (unbounded)
-- ============================================================

---@param traces table[]  Array of SwarmTrace
---@param target_action string|nil  If given, measure tick of first occurrence
---@return table  { n, mean, std_dev, min, max, median, ci_lower, ci_upper }
function M.convergence(traces, target_action)
  local values = {}

  for _, trace in ipairs(traces) do
    if target_action then
      for _, a in ipairs(trace.actions) do
        if a.action == target_action then
          values[#values + 1] = a.tick
          break
        end
      end
    else
      values[#values + 1] = trace.ticks
    end
  end

  return stats.describe_with_ci(values, { unbounded = true })
end

-- ============================================================
-- Exploration efficiency
--
-- Per-trace metrics: unique action ratio, duplicate rate.
-- ============================================================

---@param trace table  SwarmTrace
---@return table  { total, unique, unique_ratio, duplicate_rate }
function M.exploration_efficiency(trace)
  local actions = trace.actions
  local total = #actions

  if total == 0 then
    return { total = 0, unique = 0, unique_ratio = 0, duplicate_rate = 0 }
  end

  local seen = {}
  local unique = 0
  for _, a in ipairs(actions) do
    if not seen[a.action] then
      seen[a.action] = true
      unique = unique + 1
    end
  end

  return {
    total          = total,
    unique         = unique,
    unique_ratio   = unique / total,
    duplicate_rate = 1 - (unique / total),
  }
end

-- ============================================================
-- Multi-worker coordination analysis
--
-- Per-worker action counts and overlap metrics.
-- Overlap = action types performed by >1 worker / total action types.
-- ============================================================

---@param trace table  SwarmTrace
---@return table  { workers, overlap_rate, worker_count }
function M.worker_coordination(trace)
  local actions = trace.actions
  local workers = {}
  local action_workers = {}   -- action_name -> set of worker IDs

  for _, a in ipairs(actions) do
    if not workers[a.worker] then
      workers[a.worker] = { count = 0, action_set = {} }
    end
    workers[a.worker].count = workers[a.worker].count + 1
    workers[a.worker].action_set[a.action] = true

    if not action_workers[a.action] then
      action_workers[a.action] = {}
    end
    action_workers[a.action][a.worker] = true
  end

  local total_action_types = 0
  local overlapping = 0
  for _, worker_set in pairs(action_workers) do
    total_action_types = total_action_types + 1
    local count = 0
    for _ in pairs(worker_set) do count = count + 1 end
    if count > 1 then overlapping = overlapping + 1 end
  end

  local worker_count = 0
  for _ in pairs(workers) do worker_count = worker_count + 1 end

  return {
    workers        = workers,
    overlap_rate   = total_action_types > 0 and overlapping / total_action_types or 0,
    worker_count   = worker_count,
  }
end

-- ============================================================
-- Action validity
--
-- Computes rate of "valid" actions based on user-provided predicate.
-- ============================================================

---@param trace table  SwarmTrace
---@param is_valid_fn function(action_record) -> boolean
---@return table  { total, valid, rate }
function M.action_validity(trace, is_valid_fn)
  local actions = trace.actions
  local total = #actions
  local valid = 0

  for _, a in ipairs(actions) do
    if is_valid_fn(a) then valid = valid + 1 end
  end

  return {
    total = total,
    valid = valid,
    rate  = total > 0 and valid / total or 0,
  }
end

return M
