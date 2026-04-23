--- pkg_alpha — minimal fixture package for compat_undeclared E2E test.
---
--- Signal token: COMPAT_UNDECL_SIGNAL_ALPHA
--- Has no alc_shapes_compat field — should warn and continue loading.

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_alpha",
    version     = "0.1.0",
    category    = "test",
    description = "COMPAT_UNDECL_SIGNAL_ALPHA: compat undeclared fixture",
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
    ctx.result = { answer = "compat_undeclared" }
    return ctx
end

return M
