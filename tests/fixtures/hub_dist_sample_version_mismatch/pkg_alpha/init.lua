--- pkg_alpha — minimal fixture package for version_mismatch E2E test.
---
--- Signal token: VMISMATCH_SIGNAL_ALPHA

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_alpha",
    version     = "0.1.0",
    category    = "test",
    description = "VMISMATCH_SIGNAL_ALPHA: version mismatch fixture (should error before gendoc)",
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
    ctx.result = { answer = "mismatch" }
    return ctx
end

return M
