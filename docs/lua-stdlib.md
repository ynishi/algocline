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

## Type Support

algocline distributes `types/alc.d.lua` — a [LuaCats](https://luals.github.io/wiki/annotations/) type definition file covering all `alc.*` functions. This enables editor completion and static analysis via [lua-language-server](https://github.com/LuaLS/lua-language-server).

### Setup

The type stub is installed to `~/.algocline/types/alc.d.lua` automatically on `alc init` and on every MCP server startup.

Add the types directory to your `.luarc.json`:

```json
{
  "workspace": { "library": ["~/.algocline/types"] },
  "diagnostics": { "globals": ["alc"] },
  "runtime": { "version": "Lua 5.4" }
}
```

### CI Integration

```bash
lua-language-server --check src/ --configpath .luarc.json --checklevel=Warning
```

Non-zero exit on diagnostics. Detects undefined `alc.*` calls not covered by `alc.d.lua`.

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

#### `alc.llm_json(prompt, opts?) -> table|nil, string`

Call `alc.llm()` and parse the response as JSON via `alc.json_extract()`. On parse failure, retries once with a repair prompt that includes the previous (broken) output, allowing the model to fix rather than regenerate.

Returns `(parsed_table, raw_string)` on success, or `(nil, raw_string)` if extraction fails after retry.

```lua
local data, raw = alc.llm_json("Return a JSON object with fields: name, age")
if data then
    print(data.name)
else
    alc.log("error", "Failed to get JSON: " .. raw)
end
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

### Evaluation

#### `alc.eval(scenario, strategy, opts?) -> report`

Evaluate a strategy against a scenario. Thin facade over
[evalframe](https://github.com/yutakanishimura/evalframe) that handles
scenario resolution, provider wiring, and optional Card emission.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scenario` | string or table | yes | Named scenario or inline spec |
| `strategy` | string | yes | Package name (e.g. `"cot"`, `"reflect"`) |
| `opts.strategy_opts` | table | no | Extra opts passed to strategy `run()` |
| `opts.auto_card` | boolean | no | Emit Card on completion (default: false) |
| `opts.card_pkg` | string | no | Card `pkg.name` override |

**Scenario formats:**

```lua
-- Simple form: cases + grader names
{
    cases = {
        { input = "2+2?", expected = "4" },
        { input = "sqrt(16)?", expected = "4" },
    },
    graders = { "exact_match" },
}

-- Full evalframe-compatible form
local ef = require("evalframe")
{
    ef.bind { ef.graders.exact_match },
    ef.bind { ef.graders.contains, weight = 0.5 },
    cases = {
        ef.case { input = "2+2?", expected = "4", tags = { "math" } },
    },
}

-- Named scenario (loads from ~/.algocline/scenarios/)
"gsm8k_100"
```

**Returns:** report table

```lua
report.aggregated.pass_rate    -- 0.8
report.aggregated.passed       -- 8
report.aggregated.total        -- 10
report.aggregated.scores.mean  -- 0.75
report.aggregated.scores.std_dev
report.aggregated.ci_95        -- { lower = 0.62, upper = 0.88 }
report.aggregated.by_tag       -- per-tag breakdown
report.failures                -- failed case details
report.results                 -- all case results
report.summary                 -- human-readable text
report.card_id                 -- set when auto_card = true
```

**Available graders** (string shorthand):

| Name | Returns | Behavior |
|------|---------|----------|
| `"exact_match"` | bool | Exact string match against expected |
| `"contains"` | bool | Expected substring found in response |
| `"starts_with"` | bool | Response starts with expected |
| `"regex"` | bool | Lua pattern match (via `context.pattern` or `expected[1]`) |
| `"json_valid"` | bool | Response is valid JSON |
| `"not_empty"` | bool | Non-empty response |

```lua
-- Basic eval
local report = alc.eval({
    cases = {
        { input = "2+2?", expected = "4" },
        { input = "Capital of France?", expected = "Paris" },
    },
    graders = { "contains" },
}, "cot")

-- With Card emission
local report = alc.eval("gsm8k_100", "reflect", {
    auto_card = true,
})
alc.log("info", "pass_rate: " .. report.aggregated.pass_rate)
```

---

## alc.card — Immutable Run-Result Snapshots

Persistent storage for evaluation / experiment results. Each Card is a
write-once TOML file under `~/.algocline/cards/{pkg}/{card_id}.toml`.

### Two-Tier Content Policy

Card storage follows a two-tier architecture aligned with industry
practice (MLflow, W&B, OpenAI Evals, LangSmith, etc.):

| Tier | Storage | Content | Size guidance |
|------|---------|---------|---------------|
| **Tier 1** — Card body (TOML) | `{card_id}.toml` | Aggregate scalars, decision values, identity/lineage, params fingerprint, single summary text | A few KB |
| **Tier 2** — Samples sidecar (JSONL) | `{card_id}.samples.jsonl` | Per-case raw data, per-sample I/O, per-persona scores, large transcripts | Unbounded |

Rule of thumb: if a value is **per-case** or **large**, it belongs in
Tier 2. Everything else goes in Tier 1.

### Schema Conventions

Cards are schemaless TOML: any section / field you write is preserved
and queryable via `where`. The following conventions are **recognized**
— not enforced, but tools and docs assume this layout when it exists.

**`[strategy_params]`** — parameters the strategy treats as tunable
(sweep knobs, optimizer targets). Kept as a first-class section so
sweep / optimize tooling can pick them up without pattern-matching
`[params]`. Example: `strategy_params = { alpha = 0.7, depth = 3 }`.

**`[metadata]` lineage fields:**

| Field | Meaning |
|-------|---------|
| `prior_card_id` | The parent Card's `card_id`, for derived runs (sweeps, reflections, re-scorings). |
| `prior_relation` | Short tag describing the relation type. Suggested values: `"sweep_variant"`, `"reflection_of"`, `"derived_from"`, `"rescored_from"`. |

Writing these lets future lineage tooling (`alc.card.lineage`, Step 4)
traverse Card ancestries without guessing field names.

```lua
alc.card.create({
    pkg = { name = "my_sweep" },
    strategy_params = { alpha = 0.7 },
    stats = { ev = 0.62 },
    metadata = {
        prior_card_id = seed_card_id,
        prior_relation = "sweep_variant",
    },
})
```

### Write API

#### `alc.card.create(table) -> { card_id, path }`

Write a new Card. Immutable — calling `create` with the same `card_id`
errors.

**Required fields:** `pkg.name`

Auto-injected: `schema_version`, `card_id`, `created_at`, `created_by`,
`param_fingerprint` (when `params` is present).

```lua
local result = alc.card.create({
    pkg = { name = "my_eval" },
    scenario = { name = "gsm8k_100" },
    model = { id = "claude-opus-4-6" },
    params = { temperature = 0.0, depth = 3 },
    stats = { pass_rate = 0.82, ev = 4.2 },
})
-- result.card_id, result.path
```

#### `alc.card.append(card_id, fields)`

Additive-only annotation. New top-level keys only — overwriting existing
keys is rejected.

```lua
alc.card.append(card_id, {
    caveats = { notes = "rescored after grader fix" },
    metadata = { reviewer = "yn" },
})
```

#### `alc.card.write_samples(card_id, samples)`

Write per-case data to the JSONL sidecar (Tier 2). Write-once per Card.
Column schema is package-defined — the engine does not interpret content.

```lua
alc.card.write_samples(card_id, {
    { case = "c0", passed = true, score = 1.0, response = "..." },
    { case = "c1", passed = false, score = 0.0, response = "..." },
})
```

#### `alc.card.alias_set(name, card_id, opts?)`

Pin a mutable alias to a Card. Aliases are global
(`~/.algocline/cards/_aliases.toml`). Re-binding overwrites the previous
target.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Alias name |
| `card_id` | string | yes | Target Card |
| `opts.pkg` | string | no | Package hint (metadata only) |
| `opts.note` | string | no | Free-text annotation |

```lua
alc.card.alias_set("best_gsm8k", card_id, { pkg = "my_eval" })
```

### Read API

#### `alc.card.get(card_id) -> table | nil`

Fetch full Card body by id.

#### `alc.card.get_by_alias(name) -> table | nil`

Resolve alias then fetch the Card.

#### `alc.card.list(filter?) -> summary[]`

List Cards as summaries (newest first).

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filter.pkg` | string | no | Filter by package |

#### `alc.card.find(query?) -> summary[]`

Query Cards with a Prisma-style `where` DSL plus dotted-path `order_by`.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query.pkg` | string | no | Restrict scan to a single pkg subdir (I/O hint) |
| `query.where` | table | no | Nested-object predicate (see below) |
| `query.order_by` | string \| string[] | no | Sort keys; `-` prefix = desc |
| `query.limit` | integer | no | Max results |
| `query.offset` | integer | no | Skip first N rows before `limit` |

**`where` DSL**

- Nested objects are interpreted as path extensions: `where.stats.pass_rate` targets Card root `[stats] pass_rate`.
- A value whose every key is a reserved operator name becomes a leaf comparison.
- Scalar values become implicit `eq`.
- Multiple keys in the same object combine with AND. Use `_and` / `_or` / `_not` for explicit logical ops.

**Reserved operators**: `eq ne lt lte gt gte in nin exists contains starts_with`. Card schemas must not use these names as field names anywhere.

**Missing-field semantics**: `eq/lt/lte/gt/gte/in/contains/starts_with` return false on missing fields; `ne/nin` return true; `exists` is explicit.

```lua
-- Best-scoring cot Card on gsm8k
local best = alc.card.find({
    pkg = "cot",
    where = {
        scenario = { name = "gsm8k_100" },
        stats = { pass_rate = { gte = 0.7 }, n = { gte = 30 } },
    },
    order_by = "-stats.pass_rate",
    limit = 1,
})

-- Cards where strategy temperature is >= 0.7 OR equilibrium is "dead"
local mixed = alc.card.find({
    where = {
        _or = {
            { strategy_params = { temperature = { gte = 0.7 } } },
            { stats = { equilibrium_position = "dead" } },
        },
    },
    order_by = { "-stats.pass_rate", "created_at" },
})

-- Cards that have no prior_card_id (roots)
local roots = alc.card.find({
    where = { prior_card_id = { exists = false } },
})
```

#### `alc.card.alias_list(filter?) -> alias[]`

List aliases, optionally filtered by `filter.pkg`.

#### `alc.card.read_samples(card_id, opts?) -> table[]`

Read per-case sidecar rows with optional filtering and paging.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `opts.offset` | integer | no | Skip first N matched rows (default: 0) |
| `opts.limit` | integer | no | Max rows to return |
| `opts.where` | table | no | Prisma-style predicate applied to each row |

`opts.where` uses the same nested-object DSL as
[`alc.card.find`](#alccardfindquery---summary), evaluated against each
sample row directly — samples are flat per-case objects, so no section
prefix is used. `offset` is applied **after** filtering (Prisma / SQL
convention).

```lua
local rows = alc.card.read_samples(card_id, {
  where  = { passed = true, score = { gte = 0.5 } },
  offset = 0,
  limit  = 50,
})
```

#### `alc.card.lineage(query) -> { root, nodes, edges, truncated } | nil`

Walk a Card's lineage tree via the `metadata.prior_card_id` convention.
Follows the parent pointer (`direction = "up"`, default), collects
descendants (`direction = "down"`), or both. Returns `nil` when the
starting Card does not exist.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query.card_id` | string | yes | Starting Card id. |
| `query.direction` | string | no | `"up"` (default), `"down"`, or `"both"`. |
| `query.depth` | integer | no | Max traversal depth (default 10). |
| `query.include_stats` | boolean | no | Include each node's `[stats]` section (default `true`). |
| `query.relation_filter` | string[] | no | If set, only edges whose `prior_relation` is in this list are followed. |

Return shape:

- `root` — the starting `card_id`.
- `nodes` — list of `{ card_id, pkg, depth, prior_card_id?, prior_relation?, stats? }`. `depth` is signed: `0` for the root, negative for ancestors, positive for descendants.
- `edges` — list of `{ from, to, relation? }` (child → parent).
- `truncated` — `true` when the walk hit the depth cap while more unwalked edges existed.

```lua
local tree = alc.card.lineage({
    card_id = current_id,
    direction = "up",
    depth = 5,
    relation_filter = { "sweep_variant", "rerun_of" },
})
if tree then
    for _, node in ipairs(tree.nodes) do
        alc.log("info", string.format("%+d  %s", node.depth, node.card_id))
    end
end
```

Cycle detection uses `card_id` visited-set; `card_id` embeds a UTC
timestamp so cycles cannot form naturally, but the guard is present.

---

## alc.math — Numeric Computing

Re-exported from [mlua-mathlib](https://crates.io/crates/mlua-mathlib) v0.3. Provides RNG, distribution sampling, descriptive statistics, CDF/PPF, special functions, hypothesis testing, ranking/IR metrics, information theory, and time series analysis backed by Rust (`rand`, `statrs`, `nalgebra`). Available as `alc.math.*` without `require()`.

### RNG

All sampling functions require a `LuaRng` object created via `rng_create`. RNG state is independent per instance (ChaCha12, passes TestU01).

#### `alc.math.rng_create(seed) -> LuaRng`

Create a new seeded RNG instance.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `seed` | integer | yes | 64-bit seed value |

```lua
local rng = alc.math.rng_create(42)
```

#### `alc.math.rng_float(rng) -> number`

Sample a uniform float in [0, 1).

```lua
local f = alc.math.rng_float(rng)  -- e.g. 0.5427
```

#### `alc.math.rng_int(rng, min, max) -> integer`

Sample a uniform integer in [min, max].

```lua
local n = alc.math.rng_int(rng, 1, 100)  -- e.g. 53
```

### Distribution Sampling

All sampling functions take `rng` as the first argument.

#### Continuous

| Function | Parameters | Description |
|----------|-----------|-------------|
| `normal_sample(rng, mean, stddev)` | mean: number, stddev: number | Normal (Gaussian) distribution |
| `beta_sample(rng, alpha, beta)` | alpha: number, beta: number | Beta distribution |
| `gamma_sample(rng, shape, scale)` | shape: number, scale: number | Gamma distribution |
| `exp_sample(rng, lambda)` | lambda: number | Exponential distribution |
| `uniform_sample(rng, low, high)` | low: number, high: number | Continuous uniform [low, high) |
| `lognormal_sample(rng, mu, sigma)` | mu: number, sigma: number | Log-normal distribution |
| `student_t_sample(rng, df)` | df: number | Student's t-distribution |
| `chi_squared_sample(rng, df)` | df: number | Chi-squared distribution |

```lua
local rng = alc.math.rng_create(42)
local x = alc.math.normal_sample(rng, 0, 1)
local b = alc.math.beta_sample(rng, 2, 5)
```

#### Discrete

| Function | Parameters | Description |
|----------|-----------|-------------|
| `poisson_sample(rng, lambda)` | lambda: number | Poisson distribution (returns integer) |
| `binomial_sample(rng, n, p)` | n: integer, p: number | Binomial distribution (returns integer) |

#### Multivariate

| Function | Parameters | Description |
|----------|-----------|-------------|
| `dirichlet_sample(rng, alphas)` | alphas: number[] (≥2 elements) | Dirichlet distribution (returns number[]) |
| `categorical_sample(rng, weights)` | weights: number[] (≥1 element) | Weighted categorical (returns 1-based index) |

```lua
local probs = alc.math.dirichlet_sample(rng, {1, 1, 1})  -- e.g. {0.33, 0.45, 0.22}
local idx = alc.math.categorical_sample(rng, {0.7, 0.2, 0.1})  -- e.g. 1
```

### Descriptive Statistics

All functions take a non-empty `number[]` array. NaN/Infinity values are rejected.

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `mean(data)` | data: number[] | number | Arithmetic mean |
| `variance(data)` | data: number[] | number | Sample variance (Welford's algorithm) |
| `stddev(data)` | data: number[] | number | Sample standard deviation |
| `median(data)` | data: number[] | number | Median (linear interpolation) |
| `percentile(data, p)` | data: number[], p: 0-100 | number | p-th percentile |
| `iqr(data)` | data: number[] | number | Interquartile range (Q3 - Q1) |

```lua
local avg = alc.math.mean({10, 20, 30, 40, 50})      -- 30.0
local sd = alc.math.stddev({10, 20, 30, 40, 50})      -- 15.81...
local p90 = alc.math.percentile({1,2,3,4,5,6,7,8,9,10}, 90)
```

### Bivariate Statistics

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `covariance(xs, ys)` | xs: number[], ys: number[] | number | Sample covariance (equal-length, ≥2) |
| `correlation(xs, ys)` | xs: number[], ys: number[] | number | Pearson correlation coefficient |

```lua
local r = alc.math.correlation({1,2,3,4,5}, {2,4,6,8,10})  -- 1.0
```

### Transforms & Utilities

#### `alc.math.softmax(data) -> number[]`

Numerically stable softmax (subtracts max before exp).

```lua
local probs = alc.math.softmax({1, 2, 3})  -- {0.090, 0.245, 0.665}
```

#### `alc.math.log_normalize(data) -> number[]`

Log-normalize positive values to [0, 100] scale. All values must be > 0.

```lua
local normed = alc.math.log_normalize({1, 10, 100, 1000})
```

#### `alc.math.histogram(data, bins) -> table`

Compute histogram bin counts and edges.

**Returns:** `{ counts = integer[], edges = number[] }` where `#edges == bins + 1`.

```lua
local h = alc.math.histogram({1,2,2,3,3,3,4,4,5}, 5)
-- h.counts = {1, 2, 3, 2, 1}, h.edges = {1.0, 1.8, 2.6, 3.4, 4.2, 5.0}
```

#### `alc.math.wilson_ci(successes, total, confidence) -> table`

Wilson score confidence interval for binomial proportions.

**Returns:** `{ lower = number, upper = number, center = number }`

```lua
local ci = alc.math.wilson_ci(50, 100, 0.95)
-- ci.center ≈ 0.5, ci.lower ≈ 0.404, ci.upper ≈ 0.596
```

### CDF & PPF (Inverse CDF)

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `normal_cdf(x, mu, sigma)` | x, mu, sigma: number | number | Normal CDF |
| `beta_cdf(x, alpha, beta)` | x, alpha, beta: number | number | Beta CDF |
| `gamma_cdf(x, shape, scale)` | x, shape, scale: number | number | Gamma CDF (scale param, not rate) |
| `poisson_cdf(k, lambda)` | k: integer, lambda: number | number | Poisson CDF |
| `normal_inverse_cdf(p, mu, sigma)` | p, mu, sigma: number | number | Normal PPF (p ∈ [0,1]) |
| `normal_ppf(p)` | p: number | number | Standard normal PPF (N(0,1), p ∈ [0,1]) |
| `beta_ppf(p, alpha, beta)` | p, alpha, beta: number | number | Beta PPF (p ∈ [0,1]) |

```lua
local p = alc.math.normal_cdf(0, 0, 1)            -- 0.5
local z = alc.math.normal_ppf(0.975)              -- ≈ 1.96
local x = alc.math.normal_inverse_cdf(0.975, 0, 1) -- ≈ 1.96
```

### Distribution Utilities

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `beta_mean(alpha, beta)` | alpha, beta: number (> 0) | number | Mean of Beta distribution |
| `beta_variance(alpha, beta)` | alpha, beta: number (> 0) | number | Variance of Beta distribution |

### Special Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `erf(x)` | x: number | number | Error function |
| `erfc(x)` | x: number | number | Complementary error function |
| `lgamma(x)` | x: number | number | Log-gamma (ln Γ(x)) |
| `beta(a, b)` | a, b: number | number | Beta function B(a,b) |
| `ln_beta(a, b)` | a, b: number | number | Log-beta function |
| `regularized_incomplete_beta(x, a, b)` | x, a, b: number | number | Regularized incomplete beta I_x(a,b) |
| `regularized_incomplete_gamma(a, x)` | a, x: number | number | Regularized lower incomplete gamma P(a,x) |
| `digamma(x)` | x: number | number | Digamma function ψ(x) |
| `factorial(n)` | n: integer (0-170) | number | n! (overflows f64 for n > 170) |
| `ln_factorial(n)` | n: integer | number | ln(n!) |
| `logsumexp(values)` | values: number[] | number | Log-sum-exp (numerically stable) |
| `logit(p)` | p: number (0,1) | number | Logit: log(p / (1-p)) |
| `expit(x)` | x: number | number | Expit (sigmoid): 1 / (1 + exp(-x)) |

```lua
local e = alc.math.erf(1.0)          -- ≈ 0.8427
local f = alc.math.factorial(10)      -- 3628800
local lf = alc.math.ln_factorial(100) -- ≈ 363.74
local lse = alc.math.logsumexp({1, 2, 3})  -- ≈ 3.408
local sig = alc.math.expit(0)              -- 0.5
```

### Hypothesis Testing

#### `alc.math.welch_t_test(xs, ys) -> table`

Welch's t-test for two independent samples with unequal variances.

**Returns:** `{ t_stat = number, df = number, p_value = number }`

```lua
local r = alc.math.welch_t_test({1,2,3,4,5}, {2,4,6,8,10})
-- r.t_stat, r.df, r.p_value
```

#### `alc.math.mann_whitney_u(xs, ys [, opts]) -> table`

Mann-Whitney U test (non-parametric). Optional `opts.continuity_correction` (default `true`).

**Returns:** `{ u_stat = number, z_score = number, p_value = number }`

```lua
local r = alc.math.mann_whitney_u({1,2,3}, {4,5,6})
```

#### `alc.math.chi_squared_test(observed, expected) -> table`

Chi-squared goodness-of-fit test.

**Returns:** `{ chi2_stat = number, df = number, p_value = number }`

```lua
local r = alc.math.chi_squared_test({10, 20, 30}, {20, 20, 20})
```

#### `alc.math.ks_test(xs, ys) -> table`

Kolmogorov-Smirnov two-sample test.

**Returns:** `{ d_stat = number, p_value = number }`

```lua
local r = alc.math.ks_test({1,2,3,4,5}, {1,3,5,7,9})
```

### Ranking & IR Metrics

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `rank(data)` | data: number[] | number[] | Average rank (ties averaged) |
| `spearman_correlation(xs, ys)` | xs, ys: number[] | number | Spearman rank correlation ρ |
| `kendall_tau(xs, ys)` | xs, ys: number[] | number | Kendall's τ-b rank correlation |
| `ndcg(relevance, k)` | relevance: number[], k: integer | number | Normalized DCG@k |
| `mrr(rankings)` | rankings: integer[] | number | Mean Reciprocal Rank |

```lua
local ranks = alc.math.rank({30, 10, 20})   -- {3, 1, 2}
local rho = alc.math.spearman_correlation({1,2,3}, {1,2,3})  -- 1.0
local score = alc.math.ndcg({3, 2, 1, 0}, 4)
local m = alc.math.mrr({1, 3, 2})  -- (1/1 + 1/3 + 1/2) / 3
```

### Information Theory

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `entropy(probs)` | probs: number[] | number | Shannon entropy (nats, base e) |
| `kl_divergence(p, q)` | p, q: number[] | number | KL divergence D_KL(P \|\| Q) |
| `js_divergence(p, q)` | p, q: number[] | number | Jensen-Shannon divergence |
| `cross_entropy(p, q)` | p, q: number[] | number | Cross entropy H(P, Q) |

```lua
local h = alc.math.entropy({0.5, 0.5})          -- ln(2) ≈ 0.693
local kl = alc.math.kl_divergence({0.5, 0.5}, {0.9, 0.1})
local js = alc.math.js_divergence({0.5, 0.5}, {0.9, 0.1})
```

### Time Series

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `moving_average(data, window)` | data: number[], window: integer | number[] | Simple moving average |
| `ewma(data, alpha)` | data: number[], alpha: number (0,1] | number[] | Exponentially weighted moving average |
| `autocorrelation(data, lag)` | data: number[], lag: integer | number | Autocorrelation at given lag |

```lua
local ma = alc.math.moving_average({1,2,3,4,5}, 3)  -- {2, 3, 4}
local ew = alc.math.ewma({1,2,3,4,5}, 0.3)
local acf = alc.math.autocorrelation({1,2,3,4,5,4,3,2,1}, 1)
```

### Combinatorics

#### `alc.math.permutations(n) -> table[]`

Generate all permutations of `{1, ..., n}`. Returns `n!` arrays. Recommended `n ≤ 10`.

```lua
local perms = alc.math.permutations(3)
-- {{1,2,3}, {1,3,2}, {2,1,3}, {2,3,1}, {3,1,2}, {3,2,1}}
```

### RNG Extensions

#### `alc.math.shuffle(rng, tbl) -> table`

Fisher-Yates shuffle (in-place). Returns the same table.

```lua
local rng = alc.math.rng_create(42)
local t = alc.math.shuffle(rng, {1, 2, 3, 4, 5})
```

#### `alc.math.sample_with_replacement(rng, tbl, n) -> table`

Sample `n` elements with replacement from `tbl`.

```lua
local samples = alc.math.sample_with_replacement(rng, {"a","b","c"}, 5)
```
