--- Layer 1: Prelude Combinators
---
--- Higher-order functions that compose Layer 0 primitives.
--- Loaded automatically into every session (embedded via include_str!).
--- These extend the alc.* namespace alongside Rust-backed Layer 0 functions.

--- alc.cache(prompt, opts?) -> string
--- Memoized LLM call. Returns cached response if the same prompt+opts
--- combination was seen before in this session. Drop-in replacement
--- for alc.llm() when repeated identical calls are expected.
---
--- Cache is session-scoped (in-memory table, cleared on session end).
--- Key is computed via alc.fingerprint(prompt + system + max_tokens).
---
--- opts: same as alc.llm() + cache control:
---   opts.cache_key:  explicit cache key (overrides auto-fingerprint)
---   opts.cache_skip: if true, bypass cache and always call LLM
---
--- Usage:
---   local resp = alc.cache("Summarize: " .. text)  -- first call: LLM
---   local resp = alc.cache("Summarize: " .. text)  -- second call: instant
---
---   local resp = alc.cache("Analyze", { system = "expert", cache_skip = true })
do
    local _cache = {}      -- key -> value
    local _order = {}      -- insertion order (array of keys)
    local _hits = 0
    local _misses = 0
    local _max_entries = 256

    function alc.cache(prompt, opts)
        opts = opts or {}
        if opts.cache_skip then
            return alc.llm(prompt, opts)
        end

        local key = opts.cache_key
        if not key then
            local sig = prompt
            if opts.system then sig = sig .. "\0" .. opts.system end
            if opts.max_tokens then sig = sig .. "\0" .. tostring(opts.max_tokens) end
            key = alc.fingerprint(sig)
        end

        if _cache[key] ~= nil then
            _hits = _hits + 1
            alc.log("debug", "alc.cache: hit " .. key)
            return _cache[key]
        end

        local resp = alc.llm(prompt, opts)
        _cache[key] = resp
        _order[#_order + 1] = key
        _misses = _misses + 1
        alc.log("debug", "alc.cache: miss " .. key)

        -- Evict oldest entries when over capacity
        while #_order > _max_entries do
            local evict_key = table.remove(_order, 1)
            _cache[evict_key] = nil
        end

        return resp
    end

    --- alc.cache_info() -> { entries, hits, misses, max_entries }
    --- Return cache statistics for the current session.
    function alc.cache_info()
        return { entries = #_order, hits = _hits, misses = _misses, max_entries = _max_entries }
    end

    --- alc.cache_clear()
    --- Clear all cached responses and reset counters.
    function alc.cache_clear()
        _cache = {}
        _hits = 0
        _misses = 0
    end
end

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

--- alc.parse_number(text, pattern?) -> number | nil
--- Extract a number from LLM output.
--- If pattern is given, uses it as a Lua pattern with a capture group.
--- Otherwise extracts the first number (integer or decimal, optionally negative).
---
--- Usage:
---   alc.parse_number("Found 3 subtasks")              -- 3
---   alc.parse_number("Score: 7.5/10")                  -- 7.5
---   alc.parse_number(response, "(%d+)%s+subtask")      -- 3
---   alc.parse_number("no numbers here")                -- nil
function alc.parse_number(text, pattern)
    if type(text) ~= "string" then return nil end
    if pattern then
        local m = text:match(pattern)
        return tonumber(m)
    end
    return tonumber(text:match("%-?%d+%.?%d*"))
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

--- alc.budget_check() -> boolean
--- Returns true if budget has remaining capacity (safe to continue).
--- Returns true if no budget is set (no limits).
--- Checks elapsed_ms at call time (wall-clock snapshot).
--- Use before optional LLM calls to skip gracefully when budget is low.
---
--- Note: even if budget_check() returns true, a subsequent alc.llm()
--- may still fail with "budget_exceeded" if another call consumed the
--- last remaining budget between the check and the call.
---
--- Usage:
---   if alc.budget_check() then
---       local extra = alc.llm("Optional enrichment: " .. data)
---   end
function alc.budget_check()
    local r = alc.budget_remaining()
    if r == nil then return true end
    -- Use type() check: JSON null from serde becomes userdata in mlua,
    -- not Lua nil. Comparing userdata with number would error.
    if type(r.llm_calls) == "number" and r.llm_calls <= 0 then return false end
    if type(r.elapsed_ms) == "number" and r.elapsed_ms <= 0 then return false end
    return true
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

--- alc.parallel(items, prompt_fn, opts?) -> results
--- Batch-parallel LLM calls over an array of items. Each item is
--- transformed into a prompt by prompt_fn, then all prompts are sent
--- as a single alc.llm_batch() call (one round-trip instead of N).
---
--- prompt_fn(item, index) must return:
---   - string: used as prompt (opts.system/max_tokens applied)
---   - table:  used as-is for llm_batch (must have .prompt field)
---
--- opts:
---   opts.system:     shared system prompt for all items
---   opts.max_tokens: shared max_tokens for all items
---   opts.post_fn:    post_fn(response, item, index) -> value
---
--- Usage:
---   -- Before (sequential: N round-trips)
---   local out = alc.map(chunks, function(c)
---       return alc.llm("Summarize:\n" .. c)
---   end)
---
---   -- After (parallel: 1 round-trip)
---   local out = alc.parallel(chunks, function(c)
---       return "Summarize:\n" .. c
---   end)
---
---   -- With post-processing
---   local scores = alc.parallel(candidates, function(c)
---       return "Rate 1-10:\n" .. c
---   end, {
---       post_fn = function(resp) return alc.parse_score(resp) end,
---   })
function alc.parallel(items, prompt_fn, opts)
    if type(items) ~= "table" or #items == 0 then
        error("alc.parallel: items must be a non-empty array", 2)
    end
    if type(prompt_fn) ~= "function" then
        error("alc.parallel: prompt_fn must be a function", 2)
    end
    opts = opts or {}

    -- Phase 1: build batch from prompt_fn (no LLM calls)
    local batch = {}
    for i, item in ipairs(items) do
        local p = prompt_fn(item, i)
        if type(p) == "string" then
            local entry = { prompt = p }
            if opts.system then entry.system = opts.system end
            if opts.max_tokens then entry.max_tokens = opts.max_tokens end
            batch[i] = entry
        elseif type(p) == "table" then
            if type(p.prompt) ~= "string" then
                error("alc.parallel: prompt_fn returned table without .prompt at index " .. i, 2)
            end
            batch[i] = p
        else
            error("alc.parallel: prompt_fn must return string or table, got "
                .. type(p) .. " at index " .. i, 2)
        end
    end

    -- Phase 2: single batch LLM call
    local responses = alc.llm_batch(batch)

    -- Phase 3: optional post-processing
    if opts.post_fn then
        local results = {}
        for i, resp in ipairs(responses) do
            results[i] = opts.post_fn(resp, items[i], i)
        end
        return results
    end

    return responses
end

--- alc.pipe(strategies, ctx, opts?) -> ctx
--- Sequential pipeline: run multiple strategies in order, passing
--- each stage's result as the next stage's task.
---
--- Each strategy is loaded via require() and must have M.run(ctx).
--- The pipeline shallow-copies ctx, then for each strategy:
---   1. Sets ctx.task to the previous stage's result (stringified)
---   2. Calls strategy.run(ctx)
---   3. Extracts ctx.result for the next stage
---
--- opts.on_stage(i, name, ctx): optional callback after each stage.
--- ctx.pipe_history: array of { strategy, result } for debugging.
---
--- Inter-stage data flow:
--- Between stages, ctx.result is converted to ctx.task as a **string**:
--- - table results: serialized via alc.json_encode() (JSON string)
--- - all other types: converted via tostring()
--- This means the next stage always receives ctx.task as a string.
--- Type information is intentionally discarded — each stage treats
--- ctx.task as raw text (prompt material), not structured data.
--- If a stage needs structured input, it should json_decode(ctx.task).
---
--- Limitations:
--- - Strategies must be pre-installed (require() is used, not alc_advice's
---   auto-install). Use alc_pkg_install or alc init beforehand.
--- - Budget (ctx.budget) is shared across all pipeline stages. A 3-stage
---   pipeline with max_llm_calls=10 gives ~3 calls per stage, not 10 each.
--- - Shallow copy: nested tables in ctx are shared by reference.
---   Stages that mutate nested ctx fields affect subsequent stages.
---
--- Usage:
---   local result = alc.pipe({"cot", "cove", "reflect"}, ctx)
---   -- result.pipe_history has intermediate results
---
---   -- With inline functions:
---   local result = alc.pipe({
---       "cot",
---       function(c) c.result = alc.llm("Verify: " .. c.task); return c end,
---       "reflect",
---   }, ctx)
function alc.pipe(strategies, ctx, opts)
    if type(strategies) ~= "table" or #strategies == 0 then
        error("alc.pipe: strategies must be a non-empty array", 2)
    end
    if type(ctx) ~= "table" then
        error("alc.pipe: ctx must be a table", 2)
    end
    opts = opts or {}
    local on_error = opts.on_error or "abort"

    -- Shallow-copy ctx to avoid mutating the original
    local pipe_ctx = {}
    for k, v in pairs(ctx) do pipe_ctx[k] = v end
    pipe_ctx.pipe_history = {}

    for i, entry in ipairs(strategies) do
        local name, run_fn

        if type(entry) == "string" then
            name = entry
            local ok, pkg = pcall(require, entry)
            if not ok then
                error("alc.pipe: failed to load strategy '" .. entry .. "': " .. tostring(pkg), 2)
            end
            if type(pkg) ~= "table" or type(pkg.run) ~= "function" then
                error("alc.pipe: strategy '" .. entry .. "' must export run(ctx)", 2)
            end
            run_fn = pkg.run
        elseif type(entry) == "function" then
            name = "(inline-" .. i .. ")"
            run_fn = entry
        else
            error("alc.pipe: strategy[" .. i .. "] must be a string or function", 2)
        end

        if on_error == "abort" then
            -- Default: propagate error with full stack trace (backward compatible)
            pipe_ctx = run_fn(pipe_ctx)

            if type(pipe_ctx) ~= "table" then
                error("alc.pipe: strategy '" .. name .. "' must return a table (ctx)", 2)
            end
        else
            local ok, result = pcall(run_fn, pipe_ctx)
            if not ok then
                -- Record error in history; pipe_ctx remains unchanged (previous value)
                alc.log("warn", "alc.pipe: stage '" .. name .. "' failed: " .. tostring(result))
                pipe_ctx.pipe_history[#pipe_ctx.pipe_history + 1] = {
                    strategy = name,
                    error = tostring(result),
                }
                -- "skip": ctx.task unchanged, advance to next stage
                -- "continue": pipe_ctx (including task) unchanged, advance to next stage
                goto next_stage
            end

            pipe_ctx = result
            if type(pipe_ctx) ~= "table" then
                error("alc.pipe: strategy '" .. name .. "' must return a table (ctx)", 2)
            end
        end

        -- Record history (success path only)
        local result_snapshot = pipe_ctx.result
        pipe_ctx.pipe_history[#pipe_ctx.pipe_history + 1] = {
            strategy = name,
            result = result_snapshot,
        }

        -- Transfer result → task for next stage
        if pipe_ctx.result ~= nil and i < #strategies then
            if type(pipe_ctx.result) == "table" then
                pipe_ctx.task = alc.json_encode(pipe_ctx.result)
            else
                pipe_ctx.task = tostring(pipe_ctx.result)
            end
        end

        -- Optional callback (success path only)
        if opts.on_stage then
            opts.on_stage(i, name, pipe_ctx)
        end

        ::next_stage::
    end

    return pipe_ctx
end
