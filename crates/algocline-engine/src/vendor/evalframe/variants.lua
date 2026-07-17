-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/variants.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  variants.lua — Parametric variation generator

  Generates the cartesian product (cross) or 1:1 pairing (zip)
  of multiple dimensions, each merged onto a shared base config.

  Usage:
    local variants = require("evalframe.variants")

    local result = variants.generate {
      base = { temperature = 0.7 },

      variants.vary "model" {
        { model = "gpt-4",  name = "gpt4" },
        { model = "claude", name = "claude" },
      },

      variants.vary "temp" {
        { temperature = 0.0, name = "cold" },
        { temperature = 1.0, name = "hot" },
      },

      mode = "cross",  -- "cross" (default) or "zip"
    }
    -- → 4 variants: gpt4_cold, gpt4_hot, claude_cold, claude_hot
]]

local M = {}

local VARY_TAG = {}

-- ============================================================
-- Shallow copy (1-level deep, no mutation of source)
-- ============================================================

local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- ============================================================
-- vary: dimension declaration
-- ============================================================

--- Declare a variation dimension.
---@param dim_name string  Dimension name (e.g. "model", "temperature")
---@return function(entries: table[]) → VaryDef
function M.vary(dim_name)
  if type(dim_name) ~= "string" then
    error(string.format("vary: name must be string, got %s", type(dim_name)), 2)
  end

  return function(entries)
    if type(entries) ~= "table" or #entries < 1 then
      error(string.format("vary '%s': at least 1 entry required", dim_name), 2)
    end

    -- Copy entries with auto-generated names (never mutate caller's tables)
    local copied = {}
    for i, entry in ipairs(entries) do
      local c = shallow_copy(entry)
      if not c.name then
        c.name = string.format("%s_%d", dim_name, i)
      end
      copied[#copied + 1] = c
    end

    return {
      _tag      = VARY_TAG,
      dimension = dim_name,
      entries   = copied,
    }
  end
end

--- Check if value is a VaryDef.
local function is_vary(v)
  return type(v) == "table" and v._tag == VARY_TAG
end

-- ============================================================
-- Merge: base ← entry (entry wins, excluding 'name')
-- ============================================================

local function merge(base, entry)
  local out = shallow_copy(base)
  for k, v in pairs(entry) do
    if k ~= "name" then
      out[k] = v
    end
  end
  return out
end

-- ============================================================
-- Cross product of dimensions
-- ============================================================

local function cross_product(dimensions)
  -- Start with a single empty combination
  local combos = { { name_parts = {}, merged = {} } }

  for _, dim in ipairs(dimensions) do
    local next_combos = {}
    for _, combo in ipairs(combos) do
      for _, entry in ipairs(dim.entries) do
        local new_parts = shallow_copy(combo.name_parts)
        new_parts[#new_parts + 1] = entry.name

        local new_merged = merge(combo.merged, entry)

        next_combos[#next_combos + 1] = {
          name_parts = new_parts,
          merged     = new_merged,
        }
      end
    end
    combos = next_combos
  end

  return combos
end

-- ============================================================
-- Zip pairing of dimensions
-- ============================================================

local function zip_product(dimensions)
  -- Find shortest dimension
  local min_len = math.huge
  for _, dim in ipairs(dimensions) do
    if #dim.entries < min_len then
      min_len = #dim.entries
    end
  end

  local combos = {}
  for i = 1, min_len do
    local name_parts = {}
    local merged = {}
    for _, dim in ipairs(dimensions) do
      local entry = dim.entries[i]
      name_parts[#name_parts + 1] = entry.name
      merged = merge(merged, entry)
    end
    combos[#combos + 1] = {
      name_parts = name_parts,
      merged     = merged,
    }
  end

  return combos
end

-- ============================================================
-- generate: main entry point
-- ============================================================

---@param spec table  { base, vary(...)..., mode? }
---@return table[]  Array of variant configs
function M.generate(spec)
  if type(spec) ~= "table" then
    error("variants.generate: spec must be a table", 2)
  end

  local base = spec.base or {}
  local mode = spec.mode or "cross"

  if mode ~= "cross" and mode ~= "zip" then
    error(string.format("variants.generate: mode must be 'cross' or 'zip', got '%s'", mode), 2)
  end

  -- Extract vary dimensions from positional entries
  local dimensions = {}
  for _, v in ipairs(spec) do
    if is_vary(v) then
      dimensions[#dimensions + 1] = v
    end
  end

  if #dimensions == 0 then
    error("variants.generate: at least one vary dimension required", 2)
  end

  -- Generate combinations
  local combos
  if mode == "cross" then
    combos = cross_product(dimensions)
  else
    combos = zip_product(dimensions)
  end

  -- Build final variant list: base ← merged, with generated name
  local results = {}
  for _, combo in ipairs(combos) do
    local variant = merge(base, combo.merged)
    variant.name = table.concat(combo.name_parts, "_")
    results[#results + 1] = variant
  end

  return results
end

return M
