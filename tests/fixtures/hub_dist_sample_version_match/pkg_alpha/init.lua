--- pkg_alpha — minimal fixture package for version_match E2E test.
---
--- Signal token: VMATCH_SIGNAL_ALPHA

local S = require("alc_shapes")
local T = S.T

local M = {}

M.meta = {
    name        = "pkg_alpha",
    version     = "0.1.0",
    category    = "test",
    description = "VMATCH_SIGNAL_ALPHA: version match fixture",
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
    ctx.result = { answer = "match" }
    return ctx
end

return M
