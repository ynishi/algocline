--- pkg_alpha — minimal fixture package for compat_declared E2E test.
---
--- Signal token: COMPAT_DECL_SIGNAL_ALPHA
--- Declares alc_shapes_compat = ">=0.25.0, <0.26" which includes 0.25.1.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name             = "pkg_alpha",
    version          = "0.1.0",
    category         = "test",
    description      = "COMPAT_DECL_SIGNAL_ALPHA: compat declared in-range fixture",
    alc_shapes_compat = ">=0.25.0, <0.26",
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
    ctx.result = { answer = "compat_declared" }
    return ctx
end

return M
