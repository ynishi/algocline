-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/providers/mock.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  providers/mock.lua — Mock provider for testing

  Usage:
    local mock = require("evalframe.providers.mock")

    -- Static response
    local provider = mock.static("42")

    -- Mapping-based
    local provider = mock.map({
      ["What is 2+2?"] = "4",
      ["Hello"] = "Hi there",
    })

    -- Function-based
    local provider = mock.fn(function(input)
      return "Echo: " .. input
    end)
]]

local M = {}

--- Build canonical mock response.
---@param text string
---@return table response
local function make_response(text)
  return { text = text, model = "mock", latency_ms = 0 }
end

--- Provider that always returns the same text.
---@param text string
---@return function provider
function M.static(text)
  return function(_input)
    return make_response(text)
  end
end

--- Provider that maps inputs to outputs.
--- Falls back to partial match (longest key first), then empty string.
--- Note: partial match iterates keys sorted by length (longest first)
--- to ensure deterministic results when multiple keys match.
---@param mapping table<string, string>
---@return function provider
function M.map(mapping)
  -- Pre-sort keys by length (longest first) for deterministic partial match
  local sorted_keys = {}
  for k in pairs(mapping) do sorted_keys[#sorted_keys + 1] = k end
  table.sort(sorted_keys, function(a, b) return #a > #b end)

  return function(input)
    -- Exact match
    local text = mapping[input]
    if text then
      return make_response(text)
    end

    -- Partial match (input contains key, longest key wins)
    for _, pattern in ipairs(sorted_keys) do
      if input:find(pattern, 1, true) then
        return make_response(mapping[pattern])
      end
    end

    return make_response("")
  end
end

--- Provider from custom function.
---@param fn function(input: string) → string
---@return function provider
function M.fn(fn)
  return function(input)
    return make_response(tostring(fn(input)))
  end
end

--- Provider that records all calls for inspection.
---@param responses string[]  Responses to return in order (cycles, must be non-empty)
---@return function provider, table log  { calls: string[] }
function M.recording(responses)
  if type(responses) ~= "table" or #responses == 0 then
    error("mock.recording: responses must be a non-empty list", 2)
  end
  local log = { calls = {} }
  local idx = 0

  local provider = function(input)
    log.calls[#log.calls + 1] = input
    idx = (idx % #responses) + 1
    return make_response(responses[idx])
  end

  return provider, log
end

return M
