-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/std.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  std.lua — stdlib adapter

  Host (Rust) injects mlua-batteries as the `std` global table.
  This module adapts batteries' API to evalframe's internal contract.

  Contract (required):
    std.json.decode(str) → table
    std.json.encode(tbl) → string
    std.fs.read_file(path) → string
    std.fs.file_exists(path) → boolean
    std.time() → number (epoch seconds)

  Contract (optional):
    std.http — passthrough from batteries

  Boundary:
    Security (path traversal, sandboxing) is enforced by batteries' Rust
    implementation via PathPolicy. No Lua-side guards needed.
]]

local batteries = rawget(_G, "std")

if not batteries then
  error(
    "evalframe.std: 'std' global not found. "
    .. "evalframe requires a Rust host that injects mlua-batteries.",
    2
  )
end

local M = {}

-- ============================================================
-- json — direct passthrough (required)
-- ============================================================

if not batteries.json then
  error("evalframe.std: batteries.json is required but missing", 2)
end
M.json = batteries.json

-- ============================================================
-- fs — adapt batteries API names to evalframe contract (required)
--
-- batteries: read(path), is_file(path)
-- evalframe: read_file(path), file_exists(path)
-- ============================================================

if not batteries.fs then
  error("evalframe.std: batteries.fs is required but missing", 2)
end
M.fs = {
  read_file = batteries.fs.read,
  file_exists = batteries.fs.is_file,
}

-- ============================================================
-- time — adapt from table to callable (required)
--
-- batteries: std.time.now() → epoch seconds (f64)
-- evalframe: std.time()     → epoch seconds
-- ============================================================

if not (batteries.time and batteries.time.now) then
  error("evalframe.std: batteries.time.now is required but missing", 2)
end
M.time = batteries.time.now

-- ============================================================
-- http — passthrough (optional)
-- ============================================================

if batteries.http then
  M.http = batteries.http
end

return M
