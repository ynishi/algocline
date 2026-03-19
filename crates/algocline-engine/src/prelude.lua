--- Layer 1: Prelude Combinators
---
--- Higher-order functions that compose Layer 0 primitives.
--- Loaded automatically into every session (embedded via include_str!).
--- These extend the alc.* namespace alongside Rust-backed Layer 0 functions.

--- alc.map(items, fn) -> results
--- Apply fn(item, index) to each item, return array of results.
--- fn receives (item, index) and should return a value.
---
--- Usage:
---   local results = alc.map(chunks, function(chunk, i)
---       return alc.llm("Summarize:\n" .. chunk, { max_tokens = 200 })
---   end)
function alc.map(items, fn)
    local results = {}
    for i, item in ipairs(items) do
        results[i] = fn(item, i)
    end
    return results
end

--- alc.reduce(items, fn, init?) -> value
--- Reduce array to single value. fn(acc, item, index) -> new_acc.
--- If init is nil, uses items[1] as initial value.
---
--- Usage:
---   local summary = alc.reduce(summaries, function(acc, s, i)
---       return alc.llm(
---           "Combine these summaries:\n1: " .. acc .. "\n2: " .. s,
---           { max_tokens = 300 }
---       )
---   end)
function alc.reduce(items, fn, init)
    local acc = init
    local start = 1
    if acc == nil then
        acc = items[1]
        start = 2
    end
    for i = start, #items do
        acc = fn(acc, items[i], i)
    end
    return acc
end

--- alc.vote(answers) -> { winner, count, total }
--- Majority vote over an array of string answers.
--- Groups similar answers (exact match) and returns the most frequent.
---
--- Usage:
---   local result = alc.vote({"yes", "yes", "no", "yes"})
---   -- result.winner == "yes", result.count == 3, result.total == 4
function alc.vote(answers)
    local counts = {}
    local order = {}
    for _, a in ipairs(answers) do
        local key = tostring(a):gsub("^%s+", ""):gsub("%s+$", "")
        if counts[key] == nil then
            counts[key] = 0
            order[#order + 1] = key
        end
        counts[key] = counts[key] + 1
    end
    local winner, max_count = nil, 0
    for _, key in ipairs(order) do
        if counts[key] > max_count then
            winner = key
            max_count = counts[key]
        end
    end
    return { winner = winner, count = max_count, total = #answers }
end

--- alc.filter(items, fn) -> filtered
--- Keep items where fn(item, index) returns truthy.
---
--- Usage:
---   local critical = alc.filter(findings, function(f, i)
---       local verdict = alc.llm(
---           "Is this a critical issue? Answer YES or NO:\n" .. f,
---           { max_tokens = 10 }
---       )
---       return verdict:match("[Yy][Ee][Ss]")
---   end)
function alc.filter(items, fn)
    local result = {}
    for i, item in ipairs(items) do
        if fn(item, i) then
            result[#result + 1] = item
        end
    end
    return result
end

--- alc.ground(claim, opts?) -> string
--- Convenience wrapper: calls alc.llm with grounded = true.
--- The host should ground the response in external evidence
--- (web search, code reading, documentation, etc.).
---
--- Usage:
---   local verified = alc.ground("rmcp is tokio-only")
---   local verified = alc.ground("claim", { system = "expert" })
function alc.ground(claim, opts)
    local merged = {}
    for k, v in pairs(opts or {}) do merged[k] = v end
    merged.grounded = true
    return alc.llm(claim, merged)
end

--- alc.specify(prompt, opts?) -> string
--- Convenience wrapper: calls alc.llm with underspecified = true.
--- Signals that the prompt's preconditions depend on intent/goal
--- definitions outside the current context. The host decides the
--- resolution means (user query, RAG, DB lookup, delegated agent, etc.).
---
--- Usage:
---   local answer = alc.specify("What output format do you need?")
---   local answer = alc.specify("Which module?", { system = "concise" })
function alc.specify(prompt, opts)
    local merged = {}
    for k, v in pairs(opts or {}) do merged[k] = v end
    merged.underspecified = true
    return alc.llm(prompt, merged)
end

--- alc.parse_score(str, default?) -> number
--- Extract the first integer from a string. Returns default (or 5) on failure.
--- Clamps result to 1-10 range.
---
--- Usage:
---   local score = alc.parse_score(llm_response)       -- default 5
---   local score = alc.parse_score(llm_response, 3)    -- default 3
function alc.parse_score(str, default)
    default = default or 5
    local n = tonumber(tostring(str):match("%d+"))
    if n == nil then return default end
    if n < 1 then return 1 end
    if n > 10 then return 10 end
    return n
end

--- alc.json_extract(raw) -> table | nil
--- Extract JSON object or array from LLM output.
--- Handles raw JSON, markdown fences (```json ... ```), and
--- embedded JSON within surrounding text.
--- Returns nil if no valid JSON found.
---
--- Usage:
---   local data = alc.json_extract(llm_response)
---   if data then process(data) end
function alc.json_extract(raw)
    if type(raw) ~= "string" then return nil end
    -- Direct parse
    local ok, result = pcall(alc.json_decode, raw)
    if ok and type(result) == "table" then return result end
    -- Markdown fences
    local stripped = raw:match("```json%s*(.-)%s*```")
        or raw:match("```%s*(.-)%s*```")
    if stripped then
        ok, result = pcall(alc.json_decode, stripped)
        if ok and type(result) == "table" then return result end
    end
    -- Balanced brace/bracket extraction (try all matches)
    for json_str in raw:gmatch("%b{}") do
        ok, result = pcall(alc.json_decode, json_str)
        if ok and type(result) == "table" then return result end
    end
    for json_str in raw:gmatch("%b[]") do
        ok, result = pcall(alc.json_decode, json_str)
        if ok and type(result) == "table" then return result end
    end
    return nil
end

--- alc.state.update(key, fn, default?) -> updated_value
--- Read current value, apply fn, write back. Single-operation read-modify-write.
--- If key doesn't exist, uses default (or nil) as initial value.
--- fn receives current value and must return new value.
---
--- Usage:
---   alc.state.update("counter", function(n) return n + 1 end, 0)
---
---   alc.state.update("portfolio", function(p)
---       p.updated_at = alc.time()
---       table.insert(p.arms, new_arm)
---       return p
---   end, { arms = {}, history = {} })
function alc.state.update(key, fn, default)
    local current = alc.state.get(key, default)
    local updated = fn(current)
    alc.state.set(key, updated)
    return updated
end

--- alc.llm_safe(prompt, opts, default) -> string
--- Call alc.llm, returning default on failure instead of raising.
--- Logs the error at warn level. Use for optional LLM enrichment
--- where failure should not abort the pipeline.
---
--- Usage:
---   local summary = alc.llm_safe(
---       "Summarize: " .. text,
---       { max_tokens = 200 },
---       "(summary unavailable)"
---   )
function alc.llm_safe(prompt, opts, default)
    local ok, result = pcall(alc.llm, prompt, opts)
    if ok then return result end
    alc.log("warn", "alc.llm_safe: " .. tostring(result))
    return default
end

--- alc.fingerprint(str) -> string
--- Normalize text (lowercase, collapse whitespace, trim) and
--- return 8-char hex hash (DJB2). For deduplication, not cryptography.
---
--- Usage:
---   local fp = alc.fingerprint("  Fix the Login Bug  ")
---   -- fp == alc.fingerprint("fix the login bug")  -- true
function alc.fingerprint(str)
    local s = tostring(str):lower():gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
    local hash = 5381
    for i = 1, #s do
        hash = ((hash * 33) + s:byte(i)) % 0x100000000
    end
    return string.format("%08x", hash)
end

--- alc.tuning(defaults, ctx, opts?) -> table
--- Merge tuning defaults with ctx overrides. Deep-merges dict-like
--- nested tables; shallow-replaces arrays and scalars.
--- Strips _schema key (reserved for Layer 2 parameter metadata).
---
--- Override priority: ctx values > tuning.lua defaults
---
--- opts.prefix: namespace key in ctx (e.g. "biz_kernel" reads
---   ctx.biz_kernel.kill_threshold instead of ctx.kill_threshold)
---
--- Usage:
---   local cfg = alc.tuning(require("my_pkg.tuning"), ctx)
---   -- cfg.kill_threshold uses ctx.kill_threshold if present
---
---   -- With prefix (namespaced):
---   local cfg = alc.tuning(require("my_pkg.tuning"), ctx, { prefix = "my_pkg" })
---   -- reads from ctx.my_pkg.kill_threshold
---
---   -- Deep merge example:
---   -- defaults: { exponents = { alpha = 1.0, beta = 1.0 } }
---   -- ctx:      { exponents = { alpha = 2.0 } }
---   -- result:   { exponents = { alpha = 2.0, beta = 1.0 } }
function alc.tuning(defaults, ctx, opts)
    if type(defaults) ~= "table" then return defaults end
    opts = opts or {}
    local source = ctx or {}
    if opts.prefix then
        local ns = source[opts.prefix]
        if type(ns) == "table" then
            source = ns
        elseif ns ~= nil then
            alc.log("warn", "alc.tuning: prefix '" .. opts.prefix
                .. "' exists but is not a table, ignoring")
            source = {}
        end
    end
    local result = {}
    for k, v in pairs(defaults) do
        if k == "_schema" then
            -- reserved for parameter metadata, skip
        elseif source[k] ~= nil then
            if type(v) == "table" and type(source[k]) == "table" and v[1] == nil then
                -- deep merge dict-like tables (no integer key 1)
                result[k] = alc.tuning(v, source[k])
            else
                -- shallow replace: scalars, arrays, type changes
                result[k] = source[k]
            end
        else
            result[k] = v
        end
    end
    return result
end
