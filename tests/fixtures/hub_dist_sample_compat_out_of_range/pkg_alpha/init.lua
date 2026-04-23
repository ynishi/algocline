--- pkg_alpha — minimal fixture package for compat_out_of_range E2E test.
---
--- Signal token: COMPAT_OOR_SIGNAL_ALPHA
--- Declares alc_shapes_compat = ">=0.26.0, <0.27" which does NOT include 0.25.1.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name             = "pkg_alpha",
    version          = "0.1.0",
    category         = "test",
    description      = "COMPAT_OOR_SIGNAL_ALPHA: compat declared out-of-range fixture",
    alc_shapes_compat = ">=0.26.0, <0.27",
}

M.spec = {
    entries = {
        run = {
            input = T.shape({
                flag = T.boolean:describe("Toggle flag"),
            }, { open = true }),
            result = T.shape({
                answer = T.string,
            }, { open = true }),
        },
    },
}

function M.run(ctx)
    ctx.result = { answer = "compat_out_of_range" }
    return ctx
end

return M
