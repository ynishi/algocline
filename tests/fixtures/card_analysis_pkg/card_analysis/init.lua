--- card_analysis — e2e fixture for `alc_card_analyze`.
---
--- The host resolves this name (`DEFAULT_CARD_ANALYZE_PKG`), hands it
--- `ctx.card_id` / `ctx.card` / `ctx.samples`, and promotes `ctx.result`
--- to the typed `CardAnalyzeResult`. This fixture keeps the real package's
--- contract and nothing else, so the e2e test exercises the host side
--- without depending on which `card_analysis` a developer has installed:
---
--- * no samples  -> completes at once with a result
--- * any samples -> one `alc.llm` call, then a result from its JSON

local M = {}

M.meta = {
    name = "card_analysis",
    version = "0.0.0",
    description = "e2e fixture: the card_analysis contract, without the analysis",
    category = "debugging",
}

function M.run(ctx)
    local samples = ctx.samples or {}
    if #samples == 0 then
        ctx.result = {
            pattern = "no samples",
            suggested_change = "fixture: nothing to analyze",
            confidence = 1.0,
            failure_count = 0,
            sample_count = 0,
        }
        return ctx
    end

    local raw = alc.llm("fixture: analyze " .. #samples .. " samples; answer in JSON")
    local parsed = alc.json_extract(raw) or {}
    ctx.result = {
        pattern = parsed.pattern or "<missing>",
        suggested_change = parsed.suggested_change or "<missing>",
        confidence = tonumber(parsed.confidence) or 0.0,
        sample_count = #samples,
    }
    return ctx
end

return M
