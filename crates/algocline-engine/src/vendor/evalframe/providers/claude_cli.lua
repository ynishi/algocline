-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/providers/claude_cli.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  providers/claude_cli.lua — Claude Code CLI provider

  Calls Claude via `claude -p` CLI command.
  Conforms to evalframe provider signature: provider(input) → response

  Usage:
    local claude = require("evalframe.providers.claude_cli")

    -- Default (uses whatever model claude defaults to)
    local provider = claude()

    -- With model override
    local provider = claude { model = "sonnet" }

    -- With system prompt
    local provider = claude {
      model  = "sonnet",
      system = "You are a math tutor. Answer concisely.",
    }

    -- Use as provider in suite
    local s = suite "eval" {
      provider = provider,
      ...
    }
]]

local std = require("evalframe.std")

local M = {}

-- ============================================================
-- Shell utilities
-- ============================================================

--- Shell-safe escaping: single-quote with internal quote handling.
local function shell_escape(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

--- Build env prefix to strip nested session guard vars.
--- CRITICAL: When running inside Claude Code, these env vars
--- must be stripped to avoid nested session guard.
local function env_strip_prefix()
  local vars = { "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT" }
  local parts = {}
  for _, v in ipairs(vars) do
    parts[#parts + 1] = string.format("unset %s;", v)
  end
  return table.concat(parts, " ") .. " "
end

-- ============================================================
-- Internal: call Claude CLI via io.popen
--
-- Security (stderr handling):
--   Prototype: stderr discarded via 2>/dev/null. Diagnostic info lost on CLI failure.
--   Production: replaced by Rust provider using reqwest. No io.popen.
--   Error details are returned in the result.error field.
-- ============================================================

local function cli_call(prompt, opts)
  opts = opts or {}

  local parts = { env_strip_prefix() }
  parts[#parts + 1] = "claude"
  parts[#parts + 1] = "-p"

  if opts.model then
    parts[#parts + 1] = "--model"
    parts[#parts + 1] = shell_escape(opts.model)
  end

  if opts.max_tokens then
    parts[#parts + 1] = "--max-tokens"
    parts[#parts + 1] = tostring(opts.max_tokens)
  end

  if opts.cwd then
    parts[#parts + 1] = "--cwd"
    parts[#parts + 1] = shell_escape(opts.cwd)
  end

  parts[#parts + 1] = shell_escape(prompt)
  parts[#parts + 1] = "2>/dev/null"

  local cmd = table.concat(parts, " ")

  local handle = io.popen(cmd, "r")
  if not handle then
    return { ok = false, content = nil, error = "failed to spawn claude CLI" }
  end

  local content = handle:read("*a")

  -- Lua 5.1: close() returns no useful value (nil)
  -- LuaJIT:  close() returns true/false
  -- Lua 5.2+: close() returns true/nil, "exit"/"signal", code
  local close_ok, _, close_code = handle:close()

  local is_success
  if close_ok == nil then
    is_success = content ~= nil and #content > 0
  else
    is_success = close_ok == true
  end

  if is_success then
    if content and content:sub(-1) == "\n" then
      content = content:sub(1, -2)
    end
    return { ok = true, content = content, error = nil }
  else
    return {
      ok = false,
      content = nil,
      error = string.format("claude CLI exited with code %s", tostring(close_code or "?")),
    }
  end
end

-- ============================================================
-- Provider factory
-- ============================================================

--- Create a Claude Code CLI provider.
---@param opts table|nil  { model?, system?, max_tokens?, cwd? }
---@return function provider(input) → response
setmetatable(M, {
  __call = function(_, opts)
    opts = opts or {}

    if opts.model ~= nil and type(opts.model) ~= "string" then
      error(string.format("claude_cli: 'model' must be string, got %s", type(opts.model)), 2)
    end
    if opts.system ~= nil and type(opts.system) ~= "string" then
      error(string.format("claude_cli: 'system' must be string, got %s", type(opts.system)), 2)
    end
    if opts.max_tokens ~= nil and type(opts.max_tokens) ~= "number" then
      error(string.format("claude_cli: 'max_tokens' must be number, got %s", type(opts.max_tokens)), 2)
    end
    if opts.cwd ~= nil and type(opts.cwd) ~= "string" then
      error(string.format("claude_cli: 'cwd' must be string, got %s", type(opts.cwd)), 2)
    end

    return function(input)
      local prompt = input
      if opts.system then
        prompt = string.format(
          "<system>\n%s\n</system>\n\n%s",
          opts.system, input
        )
      end

      local start = std.time()
      local result = cli_call(prompt, {
        model      = opts.model,
        max_tokens = opts.max_tokens,
        cwd        = opts.cwd,
      })
      local elapsed = (std.time() - start) * 1000

      if result.ok then
        return {
          text       = result.content or "",
          latency_ms = elapsed,
          model      = opts.model or "claude",
          raw        = result,
        }
      else
        return {
          text       = "",
          latency_ms = elapsed,
          model      = opts.model or "claude",
          error      = result.error,
          raw        = result,
        }
      end
    end
  end,
})

return M
