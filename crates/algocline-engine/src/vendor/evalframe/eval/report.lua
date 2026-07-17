-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/eval/report.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  eval/report.lua — Report formatting

  Produces human-readable and machine-readable output from eval results.
]]

local M = {}

-- ============================================================
-- Summary text
-- ============================================================

---@param agg table  Aggregated results from stats.aggregate
---@param opts table|nil  { name?: string }
---@return string
function M.summary(agg, opts)
  opts = opts or {}
  local lines = {}

  if opts.name then
    lines[#lines + 1] = string.format("Suite: %s", opts.name)
  end

  lines[#lines + 1] = string.format(
    "Cases: %d  Pass: %d  Fail: %d",
    agg.total, agg.passed, agg.total - agg.passed
  )

  -- pass@k
  local pass_parts = {}
  pass_parts[#pass_parts + 1] = string.format("pass@1: %.2f", agg.pass_at_1)
  if agg.pass_at_5 then
    pass_parts[#pass_parts + 1] = string.format("pass@5: %.2f", agg.pass_at_5)
  end
  if agg.pass_at_10 then
    pass_parts[#pass_parts + 1] = string.format("pass@10: %.2f", agg.pass_at_10)
  end
  lines[#lines + 1] = table.concat(pass_parts, "  ")

  -- Score stats
  if agg.scores.n > 0 then
    lines[#lines + 1] = string.format(
      "Mean: %.3f  StdDev: %.3f  95%% CI: [%.3f, %.3f]",
      agg.scores.mean, agg.scores.std_dev,
      agg.ci_95.lower, agg.ci_95.upper
    )
  end

  -- By tag
  if next(agg.by_tag) then
    lines[#lines + 1] = ""
    lines[#lines + 1] = "By tag:"
    -- Sort tags alphabetically
    local tags = {}
    for tag in pairs(agg.by_tag) do tags[#tags + 1] = tag end
    table.sort(tags)

    for _, tag in ipairs(tags) do
      local data = agg.by_tag[tag]
      lines[#lines + 1] = string.format(
        "  %-20s %d/%d (%.2f)",
        tag, data.pass, data.pass + data.fail, data.rate
      )
    end
  end

  return table.concat(lines, "\n")
end

-- ============================================================
-- Failures list
-- ============================================================

---@param results table[]  CaseResult[]
---@return table[] failed CaseResult[]
function M.failures(results)
  local failed = {}
  for _, r in ipairs(results) do
    if not r.passed then
      failed[#failed + 1] = r
    end
  end
  return failed
end

-- ============================================================
-- Detailed case format
-- ============================================================

---@param result table  CaseResult
---@return string
function M.format_result(result)
  local parts = {}
  -- Truncate at UTF-8 character boundary (avoid splitting multi-byte chars)
  local label
  if result.case.name ~= "" then
    label = result.case.name
  else
    local text = result.case.input
    if #text > 40 then
      local pos = 40
      -- Walk back past continuation bytes (0x80-0xBF) to find lead byte
      while pos > 0 and text:byte(pos) >= 0x80 and text:byte(pos) < 0xC0 do
        pos = pos - 1
      end
      -- If pos is a multi-byte lead byte, check if the full char fits
      if pos > 0 and text:byte(pos) >= 0xC0 then
        local b = text:byte(pos)
        local width = (b < 0xE0 and 2) or (b < 0xF0 and 3) or 4
        if pos + width - 1 > 40 then
          pos = pos - 1  -- exclude incomplete char
        end
      end
      label = text:sub(1, pos) .. "..."
    else
      label = text
    end
  end
  parts[#parts + 1] = string.format(
    "[%s] %s  score=%.3f",
    result.passed and "PASS" or "FAIL",
    label,
    result.score
  )

  for _, g in ipairs(result.grades) do
    local grade_str = tostring(g.grade)
    if #grade_str > 50 then grade_str = grade_str:sub(1, 47) .. "..." end
    local suffix = ""
    if g.error then suffix = string.format(" err=%s", g.error) end
    if g.warning then suffix = suffix .. string.format(" WARN=%s", g.warning) end
    parts[#parts + 1] = string.format(
      "  %s: %.3f (raw=%s%s)",
      g.grader, g.score, grade_str, suffix
    )
  end

  return table.concat(parts, "\n")
end

return M
