--- pkg_beta — minimal fixture package for alc_hub_dist E2E test.
---
--- Signal token: BETA_SIGNAL_INSTRUMENT_DESCRIBE
--- Exercises S.instrument and :describe on schema fields.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_beta",
    version     = "0.1.0",
    category    = "test",
    description = "BETA_SIGNAL_INSTRUMENT_DESCRIBE: instrument and describe fixture",
}

M.spec = {
    entries = {
        run = {
            input = T.shape({
                task = T.string:describe("Task description for BETA_SIGNAL_INSTRUMENT_DESCRIBE"),
            }, { open = true }),
            result = T.shape({
                answer = T.string:describe("Computed answer"),
            }, { open = true }),
        },
    },
}

function M.run(ctx)
    ctx.result = { answer = "beta" }
    return ctx
end

M.run = S.instrument(M, "run")

return M
