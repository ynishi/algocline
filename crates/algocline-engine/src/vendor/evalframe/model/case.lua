-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/model/case.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  model/case.lua — Case: schema-validated eval data

  Immutable after construction. Enforced via __newindex guard.
  Backend Contract: maps 1:1 to Rust struct.

  Usage:
    local case = require("evalframe.model.case")
    case.new { input = "What is 2+2?", expected = "4" }
    case.new { name = "capital", input = "Capital of Japan?", expected = {"Tokyo", "東京"} }
]]

local M = {}

local CASE_TAG = {}

--- Create a frozen proxy: reads pass through, writes error.
--- Compatible with Lua 5.1+ / LuaJIT.
--- Note: __pairs/__ipairs are Lua 5.2+ only. On Lua 5.1,
--- pairs(case) will not iterate fields. Use direct field access instead.
local function freeze(data)
  local mt = {
    __index = data,
    __newindex = function(_, k, _)
      error(string.format("Case is immutable: cannot set '%s'", k), 2)
    end,
    __metatable = false,
  }

  -- Lua 5.2+: enable pairs/ipairs/# on frozen proxy.
  -- LuaJIT implements Lua 5.1 semantics and does NOT support __pairs/__ipairs.
  -- Detection: _VERSION based (reliable across custom builds) + jit guard.
  local major, minor = (_VERSION or ""):match("Lua (%d+)%.(%d+)")
  major, minor = tonumber(major) or 0, tonumber(minor) or 0
  local is_lua52_plus = (major > 5 or (major == 5 and minor >= 2))
                        and rawget(_G, "jit") == nil
  if is_lua52_plus then
    mt.__pairs  = function() return pairs(data) end
    mt.__ipairs = function() return ipairs(data) end
    mt.__len    = function() return #data end
  end

  return setmetatable({}, mt)
end

-- ============================================================
-- Constructor
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

--- Internal builder (error level 3 for call via __call).
local function build(raw)
  if type(raw) ~= "table" then
    error("Case.new: argument must be a table", 3)
  end

  local c = { _tag = CASE_TAG }
  local ctx = tostring(raw.name or raw.input or "?")

  -- input (required)
  if raw.input == nil then
    error(string.format("Case: 'input' is required (%s)", ctx), 3)
  end
  if type(raw.input) ~= "string" then
    error(string.format("Case: 'input' must be string, got %s (%s)", type(raw.input), ctx), 3)
  end
  c.input = raw.input

  -- name
  c.name = raw.name or ""
  if type(c.name) ~= "string" then
    error(string.format("Case: 'name' must be string, got %s (%s)", type(c.name), ctx), 3)
  end

  -- expected (polymorphic: string, string[], or nil)
  -- Always defensive-copy to prevent caller mutation.
  local exp = raw.expected
  if exp ~= nil then
    if type(exp) == "string" then
      c.expected = { exp }  -- normalize to list
    elseif type(exp) == "table" then
      for i, v in ipairs(exp) do
        if type(v) ~= "string" then
          error(string.format("Case: expected[%d] must be string, got %s (%s)", i, type(v), ctx), 3)
        end
      end
      c.expected = copy_list(exp)
    else
      error(string.format("Case: 'expected' must be string or string[], got %s (%s)", type(exp), ctx), 3)
    end
  end

  -- context (defensive copy)
  local raw_ctx = raw.context or {}
  if type(raw_ctx) ~= "table" then
    error(string.format("Case: 'context' must be table, got %s (%s)", type(raw_ctx), ctx), 3)
  end
  c.context = copy_table(raw_ctx)

  -- tags (defensive copy)
  local raw_tags = raw.tags or {}
  if type(raw_tags) ~= "table" then
    error(string.format("Case: 'tags' must be table, got %s (%s)", type(raw_tags), ctx), 3)
  end
  c.tags = copy_list(raw_tags)

  return freeze(c)
end

---@param raw table  Raw field values
---@return table Case
function M.new(raw)
  return build(raw)
end

-- ============================================================
-- DSL shorthand: case { input = "..." }
-- ============================================================

setmetatable(M, {
  __call = function(_, first)
    if type(first) == "table" then
      return M.new(first)
    elseif type(first) == "string" then
      -- case "name" { input = "..." }
      return function(spec)
        local copy = {}
        for k, v in pairs(spec) do copy[k] = v end
        copy.name = first
        return M.new(copy)
      end
    else
      error(string.format("case: expected table or string, got %s", type(first)), 2)
    end
  end,
})

-- ============================================================
-- Introspection
-- ============================================================

---@param v any
---@return boolean
function M.is_case(v)
  return type(v) == "table" and v._tag == CASE_TAG
end

--- Check if case has a specific tag
---@param c table Case
---@param tag string
---@return boolean
function M.has_tag(c, tag)
  for _, t in ipairs(c.tags) do
    if t == tag then return true end
  end
  return false
end

return M
