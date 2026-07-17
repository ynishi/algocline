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
    local _cache = {} -- key -> value
    local _order = {} -- insertion order (array of keys)
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
            if opts.system then
                sig = sig .. "\0" .. opts.system
            end
            if opts.max_tokens then
                sig = sig .. "\0" .. tostring(opts.max_tokens)
            end
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
    for k, v in pairs(opts or {}) do
        merged[k] = v
    end
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
    for k, v in pairs(opts or {}) do
        merged[k] = v
    end
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
    if n == nil then
        return default
    end
    if n < 1 then
        return 1
    end
    if n > 10 then
        return 10
    end
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
    if type(text) ~= "string" then
        return nil
    end
    if pattern then
        local m = text:match(pattern)
        return tonumber(m)
    end
    return tonumber(text:match("%-?%d+%.?%d*"))
end

--- alc.fmt(fmt, ...) -> string
--- Safe string.format drop-in.
--- - Integer specs (%d %i %u %o %x %X %c) with non-integer float args use
---   half-away-from-zero rounding (1.5 -> 2, -1.5 -> -2).
--- - NaN / +Inf / -Inf args to integer specs rewrite the spec to %s and
---   substitute "NaN" / "Inf" / "-Inf".
--- - String args to integer specs are re-coerced via tonumber.
--- - %s + nil falls back to "<nil>".
--- - All other specs (%s %f %.Nf %q %g %e %g etc) are byte-for-byte identical
---   to string.format.
---
--- Edge cases: nil fmt is treated as "". Underflow (more specs than args)
--- propagates the native string.format error (we do not swallow).
---
--- Usage:
---   alc.fmt("%d", 1.5)              -- "2"
---   alc.fmt("revenue=$%d", 1234.5)  -- "revenue=$1235"
---   alc.fmt("%s", nil)              -- "<nil>"
---   alc.fmt("%d", 0/0)              -- "NaN"
function alc.fmt(fmt, ...)
    local args = { ... }
    local n = select("#", ...)
    local i = 0
    local out = (fmt or ""):gsub("%%[%-%+ #0]*%d*%.?%d*[diouxXcsqfeEgGpP%%]", function(spec)
        if spec == "%%" then
            return "%%"
        end
        i = i + 1
        if i > n then
            return spec
        end
        local v = args[i]
        local conv = spec:sub(-1)
        if
            conv == "d"
            or conv == "i"
            or conv == "u"
            or conv == "o"
            or conv == "x"
            or conv == "X"
            or conv == "c"
        then
            if type(v) == "string" then
                v = tonumber(v) or v
            end
            if type(v) == "number" then
                if v ~= v then
                    args[i] = "NaN"
                    return spec:sub(1, -2) .. "s"
                elseif v == math.huge then
                    args[i] = "Inf"
                    return spec:sub(1, -2) .. "s"
                elseif v == -math.huge then
                    args[i] = "-Inf"
                    return spec:sub(1, -2) .. "s"
                end
                if v % 1 ~= 0 then
                    args[i] = v >= 0 and math.floor(v + 0.5) or math.ceil(v - 0.5)
                else
                    args[i] = v
                end
            end
        elseif conv == "s" then
            if v == nil then
                args[i] = "<nil>"
            end
        end
        return spec
    end)
    return string.format(out, table.unpack(args, 1, n))
end

--- alc.log_fmt(level, fmt, ...) -> nil
--- Thin wrapper: equivalent to alc.log(level, alc.fmt(fmt, ...)).
---
--- Usage:
---   alc.log_fmt("info", "score=%d", 7.5)   -- logs "score=8"
function alc.log_fmt(level, fmt, ...)
    return alc.log(level, alc.fmt(fmt, ...))
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
    if type(raw) ~= "string" then
        return nil
    end
    -- Direct parse
    local ok, result = pcall(alc.json_decode, raw)
    if ok and type(result) == "table" then
        return result
    end
    -- Markdown fences
    local stripped = raw:match("```json%s*(.-)%s*```") or raw:match("```%s*(.-)%s*```")
    if stripped then
        ok, result = pcall(alc.json_decode, stripped)
        if ok and type(result) == "table" then
            return result
        end
    end
    -- Balanced brace/bracket extraction (try all matches)
    for json_str in raw:gmatch("%b{}") do
        ok, result = pcall(alc.json_decode, json_str)
        if ok and type(result) == "table" then
            return result
        end
    end
    for json_str in raw:gmatch("%b[]") do
        ok, result = pcall(alc.json_decode, json_str)
        if ok and type(result) == "table" then
            return result
        end
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
    if ok then
        return result
    end
    alc.log("warn", "alc.llm_safe: " .. tostring(result))
    return default
end

--- alc.llm_json(prompt, opts?) -> table|nil, string
--- Call alc.llm and parse the response as JSON. On parse failure,
--- retries once with a repair prompt that includes the previous output.
--- Uses alc.json_extract (3-stage fallback) for parsing.
---
--- Returns (parsed_table, raw_string) on success,
--- or (nil, raw_string) if JSON extraction fails after retry.
---
--- Usage:
---   local data, raw = alc.llm_json("Return a JSON object with fields: name, age")
---   if data then
---       print(data.name)
---   else
---       alc.log("error", "Failed to get JSON: " .. raw)
---   end
function alc.llm_json(prompt, opts)
    opts = opts or {}
    local raw = alc.llm(prompt, opts)
    local parsed = alc.json_extract(raw)
    if parsed then
        return parsed, raw
    end

    alc.log("warn", "alc.llm_json: JSON parse failed, retrying")
    local retry_opts = {}
    for k, v in pairs(opts) do
        retry_opts[k] = v
    end
    retry_opts.system = "Output ONLY valid JSON. No markdown fences, no explanation."

    raw = alc.llm(
        "The previous response was not valid JSON.\n\n"
            .. "Previous output:\n"
            .. raw
            .. "\n\n"
            .. "Fix the JSON and return ONLY valid JSON.\n\n"
            .. "Original request:\n"
            .. prompt,
        retry_opts
    )
    parsed = alc.json_extract(raw)
    if not parsed then
        alc.log("warn", "alc.llm_json: JSON parse failed after retry")
    end
    return parsed, raw
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
    if r == nil then
        return true
    end
    -- Use type() check: JSON null from serde becomes userdata in mlua,
    -- not Lua nil. Comparing userdata with number would error.
    if type(r.llm_calls) == "number" and r.llm_calls <= 0 then
        return false
    end
    if type(r.elapsed_ms) == "number" and r.elapsed_ms <= 0 then
        return false
    end
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
    if type(defaults) ~= "table" then
        return defaults
    end
    opts = opts or {}
    local source = ctx or {}
    if opts.prefix then
        local ns = source[opts.prefix]
        if type(ns) == "table" then
            source = ns
        elseif ns ~= nil then
            alc.log(
                "warn",
                "alc.tuning: prefix '" .. opts.prefix .. "' exists but is not a table, ignoring"
            )
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
            if opts.system then
                entry.system = opts.system
            end
            if opts.max_tokens then
                entry.max_tokens = opts.max_tokens
            end
            batch[i] = entry
        elseif type(p) == "table" then
            if type(p.prompt) ~= "string" then
                error("alc.parallel: prompt_fn returned table without .prompt at index " .. i, 2)
            end
            batch[i] = p
        else
            error(
                "alc.parallel: prompt_fn must return string or table, got "
                    .. type(p)
                    .. " at index "
                    .. i,
                2
            )
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
    for k, v in pairs(ctx) do
        pipe_ctx[k] = v
    end
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

--- alc.eval(scenario, strategy, opts?) -> report
--- Evaluate a strategy against a scenario. Thin facade over evalframe
--- that handles scenario resolution, provider wiring, and Card emission.
---
--- scenario: string (name in ~/.algocline/scenarios/) or table:
---   Simple form:  { cases = { {input=..., expected=...}, ... }, graders = {"exact_match"} }
---   Full form:    evalframe-compatible spec with ef.bind / ef.case
---
--- strategy: string (package name, e.g. "cot", "reflect")
---
--- opts:
---   strategy_opts  table   Extra opts passed to strategy.run()
---   auto_card      bool    Emit Card on completion (default: false)
---   card_pkg       string  Card pkg.name override
---
--- Returns: evalframe report table (aggregated, failures, results, summary)
do
    -- Default rubric for the `llm_rubric` judge (Boost Bench v3 axes).
    -- Criteria are specific and behavioral (no vague catch-alls); the judge
    -- reads them and returns a single overall 1-5 rating.
    local DEFAULT_RUBRIC = [[Grade the response against these three criteria, then give ONE overall rating:
1. Factual accuracy — every claim is correct and free of fabrication; no unsupported, outdated, or misleading statements.
2. Frontier reach — the answer reflects the current state of the art of the topic rather than introductory or obsolete content.
3. Depth of reasoning — the response explains why, works through trade-offs and edge cases, instead of only stating conclusions.]]

    -- Default question for the `llm_yes_no` judge (specific, behavioral).
    local DEFAULT_YES_NO_QUESTION =
        "Does the response correctly and completely answer the question posed in the input?"

    -- LLM-as-Judge provider. Calls alc.llm directly with role="grader".
    --
    -- NOT wrapped in pcall on purpose: the yield must propagate through the
    -- coroutine boundary so the host observes the judge pause as a distinct
    -- needs_response (same rationale as evalframe providers/algocline.lua).
    -- The role field lets the host route the grading call to a different
    -- model than the strategy under test (judge != subject cardinal rule).
    local function make_judge_provider()
        return function(p)
            return alc.llm(p, { role = "grader" })
        end
    end

    -- Build a per-case llm_rubric grader that reads `case.context._alc_rubric`
    -- (stashed by resolve_cases from `case.rubric`) and falls back to
    -- DEFAULT_RUBRIC when the case does not override it. Mirrors the
    -- prompt template and rating extraction from vendored
    -- evalframe/presets/llm_graders.lua M.rubric so behavior is symmetric
    -- with the upstream factory. Kept in prelude so vendored evalframe
    -- stays verbatim.
    local function build_llm_rubric_grader(ef, judge_provider)
        local scale_min, scale_max = 1, 5
        return ef.grader("llm_rubric")({
            check = function(resp, case)
                local rubric = (case.context and case.context._alc_rubric)
                    or DEFAULT_RUBRIC
                local prompt = string.format(
                    [[You are an evaluation judge. Grade the following response.

<input>
%s
</input>

<response>
%s
</response>

RUBRIC: %s

Rate the response on a scale of %d to %d.
Reply with ONLY a single number, nothing else.]],
                    case.input,
                    resp.text or "",
                    rubric,
                    scale_min,
                    scale_max
                )
                -- Direct provider call (no pcall) so the coroutine yield
                -- from alc.llm propagates through the grader boundary,
                -- matching evalframe.providers.algocline discipline.
                local text = judge_provider(prompt)
                if type(text) == "table" then
                    if text.error then
                        return nil, text.error
                    end
                    text = text.text or ""
                end
                if type(text) ~= "string" then
                    return nil, "judge returned non-string"
                end
                local rating = tonumber(text:match("(%d+%.?%d*)"))
                if not rating then
                    return nil,
                        string.format(
                            "judge did not return a number: %s",
                            text:sub(1, 100)
                        )
                end
                return math.max(scale_min, math.min(scale_max, rating))
            end,
        })
    end

    -- Resolve a grader shorthand to an evalframe Binding (grader + scorer).
    --
    -- Deterministic grader names ("exact_match") and raw functions bind with
    -- the default bool scorer. LLM-as-Judge names (`llm_rubric` /
    -- `llm_yes_no` / `llm_factuality`) auto-wire the matching scorer and a
    -- judge provider so they are usable directly from alc.eval simple form.
    local function resolve_grader(ef, g, judge_provider)
        if type(g) == "function" then
            return ef.bind({ g })
        end
        if type(g) == "string" then
            if g == "llm_rubric" then
                -- rubric judge returns a 1-5 rating → linear_1_5 → [0,1].
                -- Uses a per-case-aware grader so `case.rubric` overrides
                -- the default; the vendored evalframe factory stays
                -- available for callers that want a fixed rubric via
                -- the full `ef.bind` form.
                return ef.bind({
                    build_llm_rubric_grader(ef, judge_provider),
                    ef.scorers.linear_1_5,
                })
            elseif g == "llm_yes_no" then
                -- yes/no judge returns a boolean → bool scorer → {0,1}.
                return ef.bind({
                    ef.llm_graders.yes_no(DEFAULT_YES_NO_QUESTION, { provider = judge_provider }),
                    ef.scorers.bool,
                })
            elseif g == "llm_factuality" then
                -- factuality judge returns a 1-5 rating → linear_1_5 → [0,1].
                return ef.bind({
                    ef.llm_graders.factuality({ provider = judge_provider }),
                    ef.scorers.linear_1_5,
                })
            end
            local grader_fn = ef.graders[g]
            if not grader_fn then
                error("alc.eval: unknown grader '" .. g .. "'")
            end
            return ef.bind({ grader_fn })
        end
        error("alc.eval: grader must be a string name or function, got " .. type(g))
    end

    -- Wrap simple {input, expected} tables as ef.case if needed.
    --
    -- Case-level extension fields (`rubric`) that evalframe's `ef.case`
    -- constructor would drop are stashed into `context._alc_*` keys so
    -- they survive the wrap and remain reachable from grader `check`
    -- callbacks via the standard `case.context` field. Vendored
    -- evalframe source is not modified — the extension is fully local
    -- to alc.eval.
    local function resolve_cases(ef, raw_cases)
        local cases = {}
        for i, c in ipairs(raw_cases) do
            if type(c) == "table" and ef.case.is_case(c) then
                cases[i] = c
            elseif type(c) == "table" and c.input then
                local wrapped = c
                if c.rubric ~= nil then
                    -- Defensive shallow copy so the caller's table is not mutated,
                    -- then migrate `rubric` into `context._alc_rubric`.
                    wrapped = {}
                    for k, v in pairs(c) do
                        wrapped[k] = v
                    end
                    local ctx = {}
                    if type(c.context) == "table" then
                        for k, v in pairs(c.context) do
                            ctx[k] = v
                        end
                    end
                    ctx._alc_rubric = c.rubric
                    wrapped.context = ctx
                    wrapped.rubric = nil
                end
                cases[i] = ef.case(wrapped)
            else
                error("alc.eval: case #" .. i .. " must have an 'input' field")
            end
        end
        return cases
    end

    -- Build evalframe suite spec from scenario table.
    local function build_suite_spec(ef, spec, provider)
        -- Full form: spec already contains ef.bind entries as indexed elements
        local has_bindings = false
        for i = 1, #spec do
            if type(spec[i]) == "table" and ef.bind.is_binding(spec[i]) then
                has_bindings = true
                break
            end
        end

        -- scenario-side `provider` (if any) takes precedence over the auto-wired
        -- algocline provider. This allows replay / mock providers (e.g.
        -- ef.providers.recorded, mock.recording) to be used in alc_eval without
        -- changing the MCP wire shape. Falls back to the auto-wired provider
        -- when scenario does not specify one.
        local resolved_provider = spec.provider or provider

        if has_bindings then
            -- Full evalframe-compatible spec: copy indexed bindings + cases
            local suite_spec = { provider = resolved_provider }
            for i = 1, #spec do
                suite_spec[i] = spec[i]
            end
            suite_spec.cases = spec.cases
            return suite_spec
        end

        -- Simple form: resolve graders → bindings, cases → ef.case.
        -- resolve_grader returns a full Binding (grader + scorer) so LLM
        -- judges can attach the correct scorer; deterministic graders bind
        -- with the default bool scorer.
        local grader_names = spec.graders or { "exact_match" }
        local judge_provider = make_judge_provider()
        local suite_spec = { provider = resolved_provider }
        for i, g in ipairs(grader_names) do
            suite_spec[i] = resolve_grader(ef, g, judge_provider)
        end
        suite_spec.cases = resolve_cases(ef, spec.cases or {})
        return suite_spec
    end

    -- Emit Card from eval report (Two-Tier Content Policy).
    local function emit_eval_card(strategy, scenario_name, report, opts)
        local pkg_name = opts.card_pkg or strategy
        local agg = report.aggregated or {}
        local scores = agg.scores or {}

        local card = alc.card.create({
            pkg = { name = pkg_name },
            scenario = { name = scenario_name or "inline" },
            stats = {
                pass_rate = agg.pass_rate,
                mean_score = scores.mean,
                n = agg.total,
                passed = agg.passed,
            },
        })

        -- Tier 2: per-case results as samples sidecar
        if report.results and #report.results > 0 then
            alc.card.write_samples(card.card_id, report.results)
        end

        return card.card_id
    end

    function alc.eval(scenario, strategy, opts)
        if not scenario then
            error("alc.eval: scenario is required")
        end
        if type(scenario) ~= "string" and type(scenario) ~= "table" then
            error("alc.eval: scenario must be a string or table")
        end
        if not strategy or type(strategy) ~= "string" then
            error("alc.eval: strategy must be a string package name")
        end
        opts = opts or {}

        -- 1. Load evalframe
        local ok, ef = pcall(require, "evalframe")
        if not ok then
            error("alc.eval: evalframe not installed. Run alc_pkg_install to add it.")
        end

        -- 2. Resolve scenario
        local spec
        local scenario_name
        if type(scenario) == "string" then
            scenario_name = scenario
            local load_ok, loaded

            -- 2a. Try require (packages on package.path)
            load_ok, loaded = pcall(require, scenario)

            -- 2b. Try {alc._dirs.scenarios}/{name}.lua (service layer injects
            --     the absolute path so Lua never reads HOME directly).
            if not load_ok then
                local scenarios_dir = (alc and alc._dirs and alc._dirs.scenarios) or ""
                local path = scenarios_dir .. "/" .. scenario .. ".lua"
                local f = io.open(path, "r")
                if f then
                    local code = f:read("*a")
                    f:close()
                    local chunk, err = load(code, "@" .. path)
                    if not chunk then
                        error("alc.eval: failed to load scenario '" .. scenario .. "': " .. err)
                    end
                    loaded = chunk()
                    load_ok = true
                end
            end

            -- 2c. Try as a direct file path (absolute or relative)
            if not load_ok then
                local f = io.open(scenario, "r")
                if f then
                    local code = f:read("*a")
                    f:close()
                    local chunk, err = load(code, "@" .. scenario)
                    if not chunk then
                        error("alc.eval: failed to load scenario: " .. err)
                    end
                    loaded = chunk()
                    load_ok = true
                end
            end

            if not load_ok then
                error("alc.eval: could not resolve scenario '" .. scenario .. "'")
            end
            spec = loaded
        else -- type(scenario) == "table" (guaranteed by early validation)
            scenario_name = scenario.name
            spec = scenario
        end

        -- Validate resolved spec
        if type(spec) ~= "table" then
            error("alc.eval: scenario resolved to " .. type(spec) .. ", expected table")
        end

        -- 3. Build provider
        local provider = ef.providers.algocline({
            strategy = strategy,
            opts = opts.strategy_opts,
        })

        -- 4. Build and run suite
        local suite_spec = build_suite_spec(ef, spec, provider)
        local suite_name = strategy .. ":" .. (scenario_name or "inline")
        local suite = ef.suite(suite_name)(suite_spec)
        local report = suite:run():to_table()

        -- 5. Auto-card
        if opts.auto_card then
            local card_id = emit_eval_card(strategy, scenario_name, report, opts)
            report.card_id = card_id
            alc.log("info", "alc.eval: card emitted — " .. card_id)
        end

        return report
    end
end
