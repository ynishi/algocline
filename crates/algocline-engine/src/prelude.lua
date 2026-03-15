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
