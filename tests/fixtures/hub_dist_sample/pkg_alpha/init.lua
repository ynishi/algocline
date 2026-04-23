--- pkg_alpha — minimal fixture package for alc_hub_dist E2E test.
---
--- Signal token: ALPHA_SIGNAL_BOOLEAN_TABLE
--- Exercises T.boolean and T.table type constructors via the run entry.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_alpha",
    version     = "0.1.0",
    category    = "test",
    description = "ALPHA_SIGNAL_BOOLEAN_TABLE: boolean and table parameter fixture",
}

M.spec = {
    entries = {
        run = {
            input = T.shape({
                flag   = T.boolean:describe("Toggle flag (ALPHA_SIGNAL_BOOLEAN_TABLE)"),
                params = T.table:describe("Arbitrary params table"),
            }, { open = true }),
            result = T.shape({
                answer = T.string,
            }, { open = true }),
        },
    },
}

function M.run(ctx)
    ctx.result = { answer = "alpha" }
    return ctx
end

return M
