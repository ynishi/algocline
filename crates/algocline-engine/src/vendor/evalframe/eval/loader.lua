-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/eval/loader.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  eval/loader.lua — Case batch loader

  Loads case specs from a Lua file via std.fs (no direct I/O in model layer).
  Path validation is std.fs.check_path's responsibility.
  Code execution is restricted via sandbox.

  Prototype: setfenv (5.1/LuaJIT) / _ENV (5.2+) restricts globals.
             os, io, require, load, dofile etc. are unavailable.
  Production: Rust builds a sandboxed environment with _ENV restrictions.

  Usage:
    local loader = require("evalframe.eval.loader")
    local cases = loader.load_file("cases/math.lua")

  File format: Lua file returning a list of case specs.
    return {
      { input = "2+2?", expected = "4" },
      { input = "3*3?", expected = "9" },
    }
]]

local Case = require("evalframe.model.case")
local std  = require("evalframe.std")

local M = {}

-- ============================================================
-- Sandbox: restrict globals available to loaded case files.
-- Only pure data construction is allowed.
-- ============================================================

local function make_sandbox()
  return {
    pairs    = pairs,
    ipairs   = ipairs,
    type     = type,
    tostring = tostring,
    tonumber = tonumber,
    select   = select,
    unpack   = unpack or table.unpack,
    string   = string,
    table    = table,
    math     = math,
    -- Explicitly excluded: os, io, require, load, loadstring,
    -- loadfile, dofile, rawset, rawget, debug, package
  }
end

--- Compile content in a sandboxed environment.
--- Lua 5.1/LuaJIT: loadstring + setfenv
--- Lua 5.2+: load with env parameter
---@param content string
---@param chunkname string
---@return function|nil chunk, string|nil err
local function load_sandboxed(content, chunkname)
  local sandbox = make_sandbox()
  if rawget(_G, "setfenv") then
    -- Lua 5.1 / LuaJIT
    local chunk, err = loadstring(content, chunkname)
    if not chunk then return nil, err end
    setfenv(chunk, sandbox)
    return chunk
  else
    -- Lua 5.2+
    return load(content, chunkname, "t", sandbox)
  end
end

--- Load cases from a Lua file that returns a list of case specs.
--- I/O goes through std.fs (injectable via mlua in production).
--- Code execution is sandboxed (no os/io/require access).
---@param path string
---@return table[] Case[]
function M.load_file(path)
  local content = std.fs.read_file(path)

  local chunk, err = load_sandboxed(content, "@" .. path)
  if not chunk then
    error(string.format("load_cases: compile error in %s: %s", path, err), 2)
  end

  local ok_exec, specs = pcall(chunk)
  if not ok_exec then
    error(string.format("load_cases: runtime error in %s: %s", path, tostring(specs)), 2)
  end

  if type(specs) ~= "table" then
    error(string.format("load_cases: file must return a table (%s)", path), 2)
  end

  local cases = {}
  for i, spec in ipairs(specs) do
    local ok, c = pcall(Case.new, spec)
    if not ok then
      error(string.format("load_cases: case[%d] in %s: %s", i, path, c), 2)
    end
    cases[#cases + 1] = c
  end
  return cases
end

return M
