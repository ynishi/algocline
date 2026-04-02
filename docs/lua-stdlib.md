# algocline Lua StdLib Reference

API reference for the `alc.*` namespace available in every Lua session.

## Architecture

```
Layer 0: Runtime Primitives (host-provided)
  Built into the runtime. Capabilities that require host interaction
  or cannot be expressed in Pure Lua (LLM calls, I/O, serialization).

Layer 1: Prelude Combinators (Pure Lua)
  Higher-order functions that compose Layer 0 primitives.
  Auto-loaded into every session.

Layer 2: Packages (require() from ~/.algocline/packages/)
  Not part of StdLib. Loaded explicitly via require().
```

Layer 0 and Layer 1 are always available without `require()`.

---

## Layer 0: Runtime Primitives

### LLM

#### `alc.llm(prompt, opts?) -> string`

Call the Host LLM. The Lua coroutine yields until the host responds.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `prompt` | string | yes | The prompt to send |
| `opts.system` | string | no | System prompt |
| `opts.max_tokens` | integer | no | Max tokens (default: 1024) |
| `opts.grounded` | boolean | no | Request grounded response (default: false) |
| `opts.underspecified` | boolean | no | Signal underspecified prompt (default: false) |

**Returns:** string (LLM response)

```lua
local response = alc.llm("What is 2+2?")
local response = alc.llm("Explain X", {
    system = "You are an expert.",
    max_tokens = 500,
})
```

#### `alc.llm_batch(items) -> string[]`

Send multiple LLM calls as a single batch. All queries are dispatched concurrently.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `items` | table[] | yes | Array of query tables |
| `items[i].prompt` | string | yes | The prompt |
| `items[i].system` | string | no | System prompt |
| `items[i].max_tokens` | integer | no | Max tokens (default: 1024) |
| `items[i].grounded` | boolean | no | Request grounded response |
| `items[i].underspecified` | boolean | no | Signal underspecified prompt |

**Returns:** string[] (responses in same order as input)

```lua
local responses = alc.llm_batch({
    { prompt = "Analyze A" },
    { prompt = "Analyze B", system = "expert", max_tokens = 500 },
})
-- responses[1], responses[2]
```

#### `alc.fork(strategies, ctx, opts?) -> table[]`

Spawn N independent Lua VMs, each running one strategy with the same ctx. LLM requests from all children are batched, achieving true LLM parallelism.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `strategies` | string[] | yes | Array of package names |
| `ctx` | table | yes | Context passed to each strategy's `run(ctx)` |
| `opts.on_error` | string | no | `"skip"` (default) or `"fail"` |

**Returns:** array of `{ strategy = name, result = ... }` or `{ strategy = name, error = ... }`

```lua
local results = alc.fork({"cot", "reflect", "cove"}, ctx)
local results = alc.fork({"cot", "reflect"}, ctx, { on_error = "skip" })
```

### JSON

#### `alc.json_encode(value) -> string`

Serialize a Lua value to a JSON string.

```lua
local s = alc.json_encode({ hello = "world", n = 42 })
-- '{"hello":"world","n":42}'
```

#### `alc.json_decode(str) -> any`

Deserialize a JSON string to a Lua value.

```lua
local data = alc.json_decode('{"a":1,"b":"two"}')
-- data.a == 1, data.b == "two"
```

### Fuzzy Matching

#### `alc.match_enum(text, candidates, opts?) -> string | nil`

Find which candidate string appears in LLM output (case-insensitive substring match).
If multiple candidates match, returns the one whose last occurrence is latest
(LLMs tend to state conclusions last). Falls back to fuzzy matching for typos.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | yes | LLM response text to search |
| `candidates` | string[] | yes | List of valid values |
| `opts` | table | no | `{ threshold = 0.7 }` — min similarity for fuzzy fallback |

```lua
local verdict = alc.match_enum(response, {"PASS", "BLOCKED"})
local decision = alc.match_enum(response, {"SCAFFOLD", "KILL", "DEFER"})
```

#### `alc.match_bool(text) -> boolean | nil`

Normalize yes/no-style LLM responses. Scans for affirmative/negative keywords
(case-insensitive) and returns the polarity of the last-occurring keyword.

Affirmative: `approved`, `yes`, `ok`, `accept`, `pass`, `confirm`, `agree`, `true`, `lgtm`
Negative: `rejected`, `no`, `deny`, `block`, `fail`, `refuse`, `disagree`, `false`

```lua
alc.match_bool("Approved. The plan looks good.")    -- true
alc.match_bool("rejected: missing test coverage")   -- false
alc.match_bool("I need more information")            -- nil
```

### Logging

#### `alc.log(level, msg)`

Emit a structured log message.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `level` | string | yes | `"error"`, `"warn"`, `"info"`, or `"debug"` |
| `msg` | string | yes | Log message |

```lua
alc.log("info", "Processing chunk 3 of 10")
alc.log("debug", "Score: " .. tostring(score))
```

### State

Persistent key-value store. Namespace-scoped (via `ctx._ns`, default: `"default"`). Values are serialized as JSON.

#### `alc.state.get(key, default?) -> any`

Read a value. Returns `default` (or nil) if key does not exist.

```lua
local v = alc.state.get("score")         -- nil if not set
local v = alc.state.get("score", 0)      -- 0 if not set
```

#### `alc.state.set(key, value)`

Write a value. Any JSON-serializable Lua value is accepted.

```lua
alc.state.set("score", 42)
alc.state.set("data", { items = {1, 2, 3} })
```

#### `alc.state.keys() -> string[]`

List all keys in the current namespace.

```lua
local k = alc.state.keys()  -- {"score", "data"}
```

#### `alc.state.delete(key)`

Remove a key from the store.

```lua
alc.state.delete("score")
```

### Text

#### `alc.chunk(text, opts?) -> string[]`

Split text into chunks by lines or characters with optional overlap.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | yes | Text to split |
| `opts.mode` | string | no | `"lines"` (default) or `"chars"` |
| `opts.size` | integer | no | Chunk size (default: 50) |
| `opts.overlap` | integer | no | Overlap between chunks (default: 0) |

**Returns:** string[]

```lua
local chunks = alc.chunk(text, { mode = "lines", size = 50 })
local chunks = alc.chunk(text, { mode = "lines", size = 50, overlap = 10 })
local chunks = alc.chunk(text, { mode = "chars", size = 2000 })
```

### Metrics

#### `alc.stats.record(key, value)`

Record a custom metric. Any JSON-serializable value.

```lua
alc.stats.record("accuracy", 0.95)
alc.stats.record("labels", {"positive", "negative"})
```

#### `alc.stats.get(key) -> any`

Retrieve a recorded metric. Returns nil if not recorded.

```lua
local v = alc.stats.get("accuracy")  -- 0.95
```

### Time

#### `alc.time() -> number`

Wall-clock time in fractional seconds since Unix epoch (sub-millisecond precision).

```lua
local start = alc.time()
-- ... work ...
local elapsed_secs = alc.time() - start
```

### Budget

#### `alc.budget_remaining() -> table | nil`

Query raw remaining budget. Returns nil if no budget was set.

**Returns:** `{ llm_calls = N | nil, elapsed_ms = N | nil }` where each field is present only if the corresponding limit was set. Values are remaining capacity (saturating at 0).

```lua
local r = alc.budget_remaining()
if r and r.llm_calls then
    alc.log("info", "Remaining LLM calls: " .. r.llm_calls)
end
```

### Progress

#### `alc.progress(step, total, msg?)`

Report structured progress. Readable via `alc_status` MCP tool. Opt-in for strategies that benefit from step tracking.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `step` | integer | yes | Current step number |
| `total` | integer | yes | Total number of steps |
| `msg` | string | no | Optional progress message |

```lua
alc.progress(1, 5, "Analyzing chunk 1")
alc.progress(2, 5)  -- message is optional
```

---

## Layer 1: Prelude Combinators

### LLM Wrappers

#### `alc.cache(prompt, opts?) -> string`

Memoized LLM call. Returns cached response if the same prompt+opts combination was seen before in this session. Drop-in replacement for `alc.llm()`.

Cache is session-scoped (in-memory, max 256 entries, oldest-first eviction).

**Parameters:** Same as `alc.llm()`, plus:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `opts.cache_key` | string | no | Explicit cache key (overrides auto-fingerprint) |
| `opts.cache_skip` | boolean | no | Bypass cache, always call LLM |

```lua
local resp = alc.cache("Summarize: " .. text)  -- first call: LLM
local resp = alc.cache("Summarize: " .. text)  -- second call: instant
```

#### `alc.cache_info() -> table`

Return cache statistics: `{ entries, hits, misses, max_entries }`.

#### `alc.cache_clear()`

Clear all cached responses and reset counters.

#### `alc.llm_safe(prompt, opts, default) -> string`

Call `alc.llm()`, returning `default` on failure instead of raising. Logs the error at warn level.

```lua
local summary = alc.llm_safe(
    "Summarize: " .. text,
    { max_tokens = 200 },
    "(summary unavailable)"
)
```

#### `alc.ground(claim, opts?) -> string`

Convenience wrapper: calls `alc.llm()` with `grounded = true`. The host should ground the response in external evidence.

```lua
local verified = alc.ground("Lua 5.4 supports integers natively")
```

#### `alc.specify(prompt, opts?) -> string`

Convenience wrapper: calls `alc.llm()` with `underspecified = true`. Signals that preconditions depend on intent/goal definitions outside current context.

```lua
local answer = alc.specify("What output format do you need?")
```

### Collection

#### `alc.map(items, fn) -> any[]`

Apply `fn(item, index)` to each item. Returns array of results.

```lua
local results = alc.map(chunks, function(chunk, i)
    return alc.llm("Summarize:\n" .. chunk, { max_tokens = 200 })
end)
```

#### `alc.reduce(items, fn, init?) -> any`

Fold array to single value. `fn(acc, item, index) -> new_acc`. If `init` is nil, uses `items[1]` as initial value.

```lua
local summary = alc.reduce(summaries, function(acc, s, i)
    return alc.llm("Combine:\n1: " .. acc .. "\n2: " .. s, { max_tokens = 300 })
end)
```

#### `alc.filter(items, fn) -> any[]`

Keep items where `fn(item, index)` returns truthy.

```lua
local critical = alc.filter(findings, function(f, i)
    local verdict = alc.llm("Is this critical? YES or NO:\n" .. f, { max_tokens = 10 })
    return verdict:match("[Yy][Ee][Ss]")
end)
```

#### `alc.parallel(items, prompt_fn, opts?) -> string[]`

Batch-parallel LLM calls over an array. Each item is transformed into a prompt by `prompt_fn`, then all prompts are sent as a single `alc.llm_batch()` call (one round-trip instead of N).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `items` | any[] | yes | Array of items |
| `prompt_fn` | function | yes | `fn(item, i) -> string or table` |
| `opts.system` | string | no | Shared system prompt |
| `opts.max_tokens` | integer | no | Shared max_tokens |
| `opts.post_fn` | function | no | `fn(response, item, i) -> value` |

`prompt_fn` return types:
- **string**: used as prompt (opts.system/max_tokens applied)
- **table**: used as-is for llm_batch (must have `.prompt` field)

```lua
-- Before (sequential: N round-trips)
local out = alc.map(chunks, function(c)
    return alc.llm("Summarize:\n" .. c)
end)

-- After (parallel: 1 round-trip)
local out = alc.parallel(chunks, function(c)
    return "Summarize:\n" .. c
end)

-- With post-processing
local scores = alc.parallel(candidates, function(c)
    return "Rate 1-10:\n" .. c
end, {
    post_fn = function(resp) return alc.parse_score(resp) end,
})
```

### Aggregation

#### `alc.vote(answers) -> table`

Majority vote over an array of string answers. Groups by exact match (trimmed).

**Returns:** `{ winner = string, count = integer, total = integer }`

```lua
local result = alc.vote({"yes", "yes", "no", "yes"})
-- result.winner == "yes", result.count == 3, result.total == 4
```

#### `alc.parse_score(str, default?) -> integer`

Extract the first integer from a string. Clamps to 1-10 range. Returns `default` (or 5) on failure.

```lua
local score = alc.parse_score(llm_response)       -- default 5
local score = alc.parse_score(llm_response, 3)    -- default 3
```

#### `alc.parse_number(text, pattern?) -> number | nil`

Extract a number from LLM output. If `pattern` is given, uses it as a Lua pattern
with a capture group. Otherwise extracts the first number (integer or decimal, optionally negative).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | yes | Text to extract from |
| `pattern` | string | no | Lua pattern with capture group |

```lua
alc.parse_number("Found 3 subtasks")              -- 3
alc.parse_number("Score: 7.5/10")                  -- 7.5
alc.parse_number("Temperature: -5 degrees")        -- -5
alc.parse_number(response, "(%d+)%s+subtask")      -- 3
alc.parse_number("no numbers here")                -- nil
```

### JSON

#### `alc.json_extract(raw) -> table | nil`

Extract JSON object or array from LLM output. 3-stage fallback:
1. Direct `json_decode`
2. Markdown fence removal (```` ```json ... ``` ````)
3. Balanced brace/bracket extraction (`%b{}` / `%b[]`)

Returns nil if no valid JSON found.

```lua
local data = alc.json_extract(llm_response)
if data then process(data) end
```

### State

#### `alc.state.update(key, fn, default?) -> any`

Read-modify-write. Reads current value, applies `fn`, writes back.

```lua
alc.state.update("counter", function(n) return n + 1 end, 0)

alc.state.update("portfolio", function(p)
    p.updated_at = alc.time()
    table.insert(p.arms, new_arm)
    return p
end, { arms = {}, history = {} })
```

### Pipeline

#### `alc.pipe(strategies, ctx, opts?) -> table`

Sequential pipeline: run multiple strategies in order, passing each stage's result as the next stage's task.

Each strategy is loaded via `require()` and must export `M.run(ctx)`. Inline functions are also accepted.

**Inter-stage data flow:** `ctx.result` is converted to `ctx.task` as a string between stages (tables are JSON-encoded). Each stage treats `ctx.task` as raw text.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `strategies` | (string\|function)[] | yes | Array of package names or inline functions |
| `ctx` | table | yes | Initial context |
| `opts.on_stage` | function | no | `fn(i, name, ctx)` callback after each stage |

**Returns:** ctx with `.result` and `.pipe_history` (array of `{ strategy, result }`)

**Limitations:**
- Strategies must be pre-installed (`require()` is used)
- Budget is shared across all pipeline stages
- Shallow copy: nested tables in ctx are shared by reference

```lua
local result = alc.pipe({"cot", "cove", "reflect"}, ctx)

-- With inline functions
local result = alc.pipe({
    "cot",
    function(c) c.result = alc.llm("Verify: " .. c.task); return c end,
    "reflect",
}, ctx)
```

### Tuning

#### `alc.tuning(defaults, ctx, opts?) -> table`

Merge tuning defaults with ctx overrides. Deep-merges dict-like nested tables; shallow-replaces arrays and scalars. Strips `_schema` key (reserved for parameter metadata).

Override priority: ctx values > tuning defaults.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `defaults` | table | yes | Default parameter table (typically from `tuning.lua`) |
| `ctx` | table | yes | Context with potential overrides |
| `opts.prefix` | string | no | Namespace key in ctx (reads `ctx[prefix].*` instead of `ctx.*`) |

```lua
local cfg = alc.tuning(require("my_pkg.tuning"), ctx)

-- With prefix (namespaced):
local cfg = alc.tuning(require("my_pkg.tuning"), ctx, { prefix = "my_pkg" })

-- Deep merge:
-- defaults: { exponents = { alpha = 1.0, beta = 1.0 } }
-- ctx:      { exponents = { alpha = 2.0 } }
-- result:   { exponents = { alpha = 2.0, beta = 1.0 } }
```

### Utility

#### `alc.fingerprint(str) -> string`

Normalize text (lowercase, collapse whitespace, trim) and return 8-char hex hash (DJB2). For deduplication, not cryptography.

```lua
local fp = alc.fingerprint("  Fix the Login Bug  ")
-- fp == alc.fingerprint("fix the login bug")  -- true
```

#### `alc.budget_check() -> boolean`

Returns true if budget has remaining capacity (safe to continue). Returns true if no budget is set.

**Note:** Even if `budget_check()` returns true, a subsequent `alc.llm()` may still fail with `"budget_exceeded"` if another call consumed the last remaining budget between the check and the call.

```lua
if alc.budget_check() then
    local extra = alc.llm("Optional enrichment: " .. data)
end
```
