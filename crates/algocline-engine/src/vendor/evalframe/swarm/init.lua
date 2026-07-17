-- vendored from evalframe v0.4.0 (~/.algocline/packages/evalframe/swarm/init.lua)
-- DO NOT EDIT here — upstream sync via manual copy
--[[
  evalframe.swarm — Swarm evaluation DSL

  Provides declarative primitives for Swarm-based evaluation.
  The actual tick loop (runner) is user-provided.
  Implement a runner function that receives config and returns a SwarmTrace.

  Usage:
    local sw = require("evalframe.swarm")

    local env = sw.env "troubleshooting" { scenario = "memory_leak" }

    local actions = sw.actions {
      sw.action "CheckStatus" { description = "Check service health" },
      sw.action "ReadLogs"    { description = "Read logs", target = "service" },
    }

    local swarm_cfg = sw.swarm { workers = 3, max_ticks = 20 }

    local provider = sw.provider(my_runner, {
      env = env, actions = actions, swarm = swarm_cfg,
    })
]]

local env_mod      = require("evalframe.swarm.env")
local actions_mod  = require("evalframe.swarm.actions")
local config_mod   = require("evalframe.swarm.config")
local trace_mod    = require("evalframe.swarm.trace")
local provider_mod = require("evalframe.swarm.provider")

local M = {}

-- Declarations
M.env     = env_mod.build
M.action  = actions_mod.build_action
M.actions = actions_mod.build_action_space
M.swarm   = config_mod.build

-- Trace construction
M.trace = trace_mod.build

-- Provider adapter
M.provider = provider_mod.build

-- Graders (DSL layer — the primary API for trace evaluation)
M.graders = require("evalframe.swarm.graders")

-- Analysis (multi-trace aggregate)
M.analysis = require("evalframe.swarm.analysis")

-- Type checks
M.is_env          = env_mod.is_env
M.is_action       = actions_mod.is_action
M.is_action_space = actions_mod.is_action_space
M.is_swarm_config = config_mod.is_swarm_config
M.is_trace        = trace_mod.is_trace

return M
