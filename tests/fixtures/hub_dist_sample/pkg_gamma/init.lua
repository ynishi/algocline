--- pkg_gamma — minimal fixture package for alc_hub_dist E2E test.
---
--- Signal token: GAMMA_SIGNAL_NESTED_SHAPE
--- Exercises nested T.shape and T.array_of constructors.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_gamma",
    version     = "0.1.0",
    category    = "test",
    description = "GAMMA_SIGNAL_NESTED_SHAPE: nested shape and array_of fixture",
}

M.spec = {
    entries = {
        run = {
            input = T.shape({
                items = T.array_of(T.shape({
                    id    = T.string:describe("Item identifier (GAMMA_SIGNAL_NESTED_SHAPE)"),
                    score = T.number:describe("Item score"),
                })):describe("List of scored items"),
            }, { open = true }),
            result = T.shape({
                best = T.string:describe("Best item id"),
            }, { open = true }),
        },
    },
}

function M.run(ctx)
    ctx.result = { best = "gamma" }
    return ctx
end

return M
