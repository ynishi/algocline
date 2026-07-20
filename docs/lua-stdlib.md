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
| `opts.cache_breakpoint` | string | no | Opaque prompt-cache hint forwarded to the host (e.g. `"context"` / `"prompt"`). Host is responsible for mapping to provider-specific cache API (Anthropic `cache_control` etc.). Hosts without prompt-cache support ignore the field. |
| `opts.card_context` | string \| table | no | Inject prior Cards into the system prompt as a fixed `<past_cards>` XML-like block. String form resolves a single Card by id; table form `{pkg=..., limit=N}` fetches the N most recent Cards for that pkg (default 5, capped at 100). See `alc.card` §Card context injection for the template. |

**Returns:** string (LLM response)

```lua
local response = alc.llm("What is 2+2?")
local response = alc.llm("Explain X", {
    system = "You are an expert.",
    max_tokens = 500,
})

-- Prompt cache hint: repeated calls with the same system prompt benefit
-- from Anthropic prompt cache (5-min TTL). The engine is opaque about the
-- value; the host does the mapping.
for i = 1, 5 do
    alc.llm(prompts[i], {
        system = shared_system_prompt,
        cache_breakpoint = "context",
    })
end

-- Card context injection: prepend prior success/failure history to the
-- system prompt as few-shot context. `card_context` fails silently if
-- the id/pkg is not found — an alc.llm call with a bad card_context
-- behaves identically to one without it.
alc.llm("Given past runs, what should I try next?", {
    system = "You are a helpful assistant.",
    card_context = { pkg = "cot", limit = 5 },
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
| `items[i].cache_breakpoint` | string | no | Opaque prompt-cache hint (see `alc.llm`) |

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

### Formatting

#### `alc.fmt(fmt, ...) -> string`

Safe `string.format` drop-in. Defensive on LLM-derived numeric arguments.

- Integer specs (`%d %i %u %o %x %X %c`) with non-integer float args use
  half-away-from-zero rounding (`1.5 -> 2`, `-1.5 -> -2`).
- `NaN`, `+Inf`, `-Inf` args to integer specs rewrite the spec to `%s` and
  substitute the literal strings `"NaN"`, `"Inf"`, `"-Inf"`.
- String args to integer specs are re-coerced via `tonumber`.
- `%s` + `nil` falls back to `"<nil>"`.
- All other specs (`%s %f %.Nf %q %g %e` ...) are byte-for-byte identical to
  `string.format`.

```lua
alc.fmt("%d", 1.5)                          -- "2"
alc.fmt("revenue=$%d users=%d", 1234.5, 10) -- "revenue=$1235 users=10"
alc.fmt("%d", 0 / 0)                        -- "NaN"
alc.fmt("%s", nil)                          -- "<nil>"
alc.fmt("%.2f", 3.14159)                    -- "3.14"
```

#### `alc.log_fmt(level, fmt, ...)`

Thin wrapper. Equivalent to `alc.log(level, alc.fmt(fmt, ...))`.

```lua
alc.log_fmt("info", "score=%d users=%d", 7.5, 10)
-- logs "score=8 users=10" at level info
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

#### `alc.state.has(key) -> boolean`

Check whether a key exists without reading the value.

```lua
if alc.state.has("score") then
  -- key exists
end
```

#### `alc.state.set_nx(key, value) -> boolean`

Set a value only if the key does **not** already exist. Returns `true` if the value was written, `false` if the key was already present.

```lua
local ok = alc.state.set_nx("lock", true)   -- true (first call)
local ok = alc.state.set_nx("lock", true)   -- false (already set)
```

#### `alc.state.incr(key, delta?, default?) -> number`

Counter increment (single-process atomic). Adds `delta` (default 1) to the current numeric value. If the key is missing, initialises from `default` (default 0) before adding. Returns the new value. Errors if the existing value is not a number. Integer-valued deltas are exact; fractional deltas may accumulate floating-point rounding over many calls.

```lua
alc.state.incr("counter")           -- 1   (0 + 1)
alc.state.incr("counter", 5)        -- 6   (1 + 5)
alc.state.incr("counter", -2)       -- 4   (6 - 2)
alc.state.incr("score", 10, 100)    -- 110 (init 100, + 10)
```

#### `alc.state.list(namespace) -> string[]`

List keys (file basenames without `.json`) under the dispatched layout `{state_root}/{namespace}/*.json`. Namespace must be a safe segment. Keys are returned sorted lexicographically. `.bak` / `.tmp` files are excluded.

```lua
local keys = alc.state.list("my_app")  -- {"alpha", "beta"}
```

> Note: `alc.state.list(ns)` reads the dispatched layout (`{state_root}/{ns}/*.json`). This is distinct from `alc.state.keys()` which reads the legacy single-file layout (`{state_root}/{ns}.json`). They are separate data and not migrated automatically.

#### `alc.state.show(namespace, key) -> table`

Read the full JSON content of `{state_root}/{namespace}/{key}.json`. Both arguments must be safe segments. Raises a typed error if the key does not exist (error message contains `not found`).

```lua
local state = alc.state.show("my_app", "task_42")
print(state.data.completed_steps[1])
```

#### `alc.state.reset(namespace, key, opts?) -> table`

Atomically mutate the JSON file at `{state_root}/{namespace}/{key}.json`:

1. Copy current file to `{key}.json.bak`
2. Remove items listed in `opts.steps` from the `data.completed_steps` array
3. Delete keys listed in `opts.fields` from the `data` table
4. Write the result via tempfile + rename

The file shape is expected to be `{ "data": { "completed_steps": [...], ... } }`. Other shapes raise `state: shape invalid`. The `.bak` snapshot lets callers recover the prior state.

```lua
local r = alc.state.reset("my_app", "task_42", {
    steps  = { "1b_REPO_READINESS" },
    fields = { "repo_readiness", "repo_readiness_report" },
})
-- r.ok = true
-- r.backup_path = "~/.algocline/state/orch/my_app/task_42.json.bak"
-- r.steps_removed = 1
-- r.fields_removed = 2
```

> Note: `namespace` is caller-specified. The engine itself does not know about specific application namespaces like `orch` / `incubator` — the examples here are illustrative only.

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

#### `alc.stats.llm_calls() -> integer`

Total LLM calls observed in the current session so far. Auto-counted by
the engine on every paused-cycle complete (one call per query in
`alc.llm_batch()` / one per `alc.llm()` invocation). Recipes and
ingredients can compute scoped deltas without tracking calls manually:

```lua
local before = alc.stats.llm_calls()
-- ... do work that may call alc.llm(...) on multiple branches ...
local count = alc.stats.llm_calls() - before
```

This avoids the `total_llm_calls = total_llm_calls + 1` pattern required
when only the per-session count was readable via the `alc_status` MCP
tool — Lua scripts can now read it directly without an external
round-trip.

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

### Neural Networks

`alc.nn` is a thin candle wrapper. The Host (Rust) owns tensors, the autograd
graph, parameters, gradients, and optimizer state; Lua composes the model and
the training loop. Optional — only present in builds enabled with the `nn`
feature (`cargo install --path . --features nn`); absent in the default MCP
build so it does not link candle.

Layer boundary: Rust exposes only the primitives below. Loops, batching, and
learning-rate schedules are written in Lua.

#### `alc.nn.tensor(data)`

Build a 1-D CPU `f32` tensor from a Lua array of numbers.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `data` | array of number | yes | 1-D input values (converted to `f32`) |

**Returns:** an `AlcTensor` handle (intermediate activation).

```lua
local x = alc.nn.tensor({ 1, 2, 3 })
```

#### `alc.nn.var(size, init)`

Create a trainable 1-D parameter registered in the Host-owned VarMap. Gradients
flow back to it and the optimizer updates it in place. Lua receives an `AlcVar`
handle usable as an operand.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `size` | integer | yes | Length of the parameter vector |
| `init` | number | yes | Constant initial value for every element |

**Returns:** an `AlcVar` handle.

```lua
local w = alc.nn.var(1, 0.0)
local b = alc.nn.var(1, 0.0)
```

#### `alc.nn.adamw(vars, lr)`

Create an AdamW optimizer over a list of `AlcVar`s. Optimizer state (momentum
etc.) stays in Rust.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `vars` | array of `AlcVar` | yes | Parameters to optimize |
| `lr` | number | yes | Learning rate |

**Returns:** an `AlcOptimizer` handle.

```lua
local opt = alc.nn.adamw({ w, b }, 0.1)
```

#### Tensor / Var op methods

Shared by `AlcTensor` and `AlcVar` — a `Var` can be used directly as an
operand and the autograd graph still tracks it. Every op returns a new
`AlcTensor`.

| Method | Description |
|--------|-------------|
| `t:add(other)` | Element-wise addition. `other` is an `AlcTensor` or `AlcVar` |
| `t:sub(other)` | Element-wise subtraction |
| `t:mul(other)` | Element-wise multiplication |
| `t:matmul(other)` | Matrix multiplication |
| `t:sqr()` | Element-wise square |
| `t:mean()` | Reduce to a scalar mean |
| `t:dims()` | Shape as a Lua array of integers |
| `t:to_vec()` | Read-back: flatten to a Lua array of numbers (`f32`) |

```lua
local out = w:mul(x):add(b)
local loss = out:sub(target):sqr():mean()
```

#### `opt:backward_step(loss)`

Fused path: candle runs `loss.backward()` into a `GradStore` and applies one
update step to the owned `Var`s in a single call. There is no `zero_grad` —
gradients are not accumulated on the parameters between steps.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `loss` | `AlcTensor` | yes | Scalar loss to backpropagate through |

**Returns:** nil.

```lua
for i = 1, steps do
    local loss = model.forward(batch_x, batch_y)
    opt:backward_step(loss)
end
```

#### `alc.nn.register(name, forward)`

Register a Lua closure as an in-VM responder for `alc.llm(prompt, {role="nn",
model=name})`. When the LLM bridge sees `role="nn"`, it looks up `model` in
this registry and dispatches to the closure synchronously — no coroutine
yield, no Host round-trip. The registry lives on the Lua VM as app data, so
it is scoped to the current session.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Key to address the model via `alc.llm(..., { model = name })` |
| `forward` | function | yes | `(prompt: string) -> string` — the model's forward function |

**Returns:** nil. Re-registering the same name overwrites the previous entry.

Errors are surfaced as Lua errors: an unknown `model` name returns
`alc.llm role="nn": no model registered as "<name>"`.

```lua
-- Train a tiny linear model for y = 2x + 1
local w = alc.nn.var(1, 0.0)
local b = alc.nn.var(1, 0.0)
local opt = alc.nn.adamw({ w, b }, 0.1)
local xs = { 0, 1, 2, 3, 4 }
local ys = { 1, 3, 5, 7, 9 }
for _ = 1, 400 do
    for i = 1, #xs do
        local x = alc.nn.tensor({ xs[i] })
        local target = alc.nn.tensor({ ys[i] })
        local loss = w:mul(x):add(b):sub(target):sqr():mean()
        opt:backward_step(loss)
    end
end

-- Expose the trained model as an alc.llm responder
alc.nn.register("tiny-linear", function(prompt)
    local x = tonumber(prompt) or 0
    local out = w:mul(alc.nn.tensor({ x })):add(b):to_vec()
    return tostring(out[1])
end)

-- Call it through the standard alc.llm API (in-process, no yield)
local reply = alc.llm("5", { role = "nn", model = "tiny-linear" })
-- reply == "10.98..."
```

#### `alc.nn.save(vars, name)`

Persist a set of trained parameters as a safetensors bundle. `vars` is a table
keyed by string; each key becomes an entry name inside the bundle. Storage is
resolved by an `NnStore` the Host injected at VM setup — the nn crate owns
tensor serialization but never chooses the location, so a session, a test, or
an alternate Host can each aim `save`/`load` at different roots without
changing this API.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `vars` | table<string, `AlcVar`> | yes | Named parameters to persist |
| `name` | string | yes | Bundle name (validated by the store; `FsStore` allows `[A-Za-z0-9_.-]` and rejects `..`) |

**Returns:** nil. Errors as `alc.nn.save: no NN store registered` if the Host
did not install one.

```lua
alc.nn.save({ w = w, b = b }, "tiny-linear")
```

#### `alc.nn.load(name)`

Restore a bundle previously written with `alc.nn.save`. Each entry becomes a
fresh `AlcVar` handle owned by the current VM; the returned table mirrors the
keys used at save time. Combine with `alc.nn.register` to expose a persisted
model as an `alc.llm` responder without re-training.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Bundle name previously passed to `save` |

**Returns:** table<string, `AlcVar`> — restored parameters. Errors as
`alc.nn.load: no NN store registered` when the Host did not install one, and
propagates the underlying loader error (e.g. missing file) otherwise.

```lua
local m = alc.nn.load("tiny-linear")
alc.nn.register("tiny-linear", function(prompt)
    local x = tonumber(prompt) or 0
    return tostring(m.w:mul(alc.nn.tensor({ x })):add(m.b):to_vec()[1])
end)
```

Notes:

- The registry is per-VM. A separate `alc_run` session starts empty; either
  retrain and re-register, or `alc.nn.load` a previously saved bundle and
  re-register from it.
- `role="nn"` calls do not count toward `stats.pauses` or `stats.llm_calls` —
  they never leave the VM.
- Any other `role` value (or absence) falls through to the normal Host LLM
  path unchanged.
- `save`/`load` require the Host to have installed an `NnStore` on the VM.
  The default engine wiring maps names to a Host-chosen directory (typically
  under the algocline app dir); tests and alternate Hosts can inject their
  own store without changing the Lua-visible API.

#### `alc.nn.preset(arch, variant, opts?)`

Arch-neutral trainable/inference preset entry point. Dispatches to
the arch's per-arch builder and hands back an `NnHandle` UserData
that behaves the same regardless of arch (see method table below).
New arches added to `algocline_nn::card::SUPPORTED_ARCHITECTURE_FAMILIES`
get their bridge preset registration through the same fn (no new
Lua-visible entry per arch).

**Parameters:**

| name    | type   | required | notes                                         |
|---------|--------|----------|-----------------------------------------------|
| arch    | string | yes      | family name (`"gpt2"` / `"tinyllama"` / `"llama"`) |
| variant | string | yes      | arch-specific variant (e.g. `"medium"` for gpt2, `"tinyllama-1.1b"` for tinyllama) |
| opts    | table  | no       | same shape as the typed alias's `opts`        |

**Returns:** `NnHandle` UserData. Methods (dispatched by arch):
`:arch()` — family name; `:variant()`, `:layers()`, `:heads()`,
`:kv_heads()`, `:dim()`, `:ctx()`, `:vocab()`, `:device()`,
`:dtype()`, `:pretrained()`, `:forward_shape(batch, seq) -> {…}`.
`:kv_heads()` returns `heads` for GPT-2 (MHA collapse) and the real
value for the two GQA archs. `:forward_shape` returns
`{batch, seq, vocab}` for trainable arches; the Llama adapter
returns `{batch, vocab}` (slices last-token logits).

**Errors:**

- `alc.nn.preset: arch 'X' not registered (expected one of gpt2 / tinyllama / llama)` — arch is declared in `SUPPORTED_ARCHITECTURE_FAMILIES` but has no bridge preset entry yet (qwen2 / phi / gemma).
- Any per-arch error from the underlying builder (variant unknown, device unsupported, etc.).

```lua
-- Arch-neutral form:
local h = alc.nn.preset("tinyllama", "tinyllama-tiny", { pretrained = false })
print(h:arch(), h:layers(), h:kv_heads())  -- "tinyllama"  2  1
local h2 = alc.nn.preset("gpt2", "medium")
print(h2:arch(), h2:layers())  -- "gpt2"  24

-- Typed aliases (backward compat):
local h3 = alc.nn.preset.gpt2("medium")        -- returns Gpt2Handle
local h4 = alc.nn.preset.tinyllama("tinyllama-1.1b")  -- returns TinyLlamaHandle
```

The typed aliases (`alc.nn.preset.gpt2` / `.tinyllama` / `.llama`)
remain callable and return their typed handles directly — use them
when arch-pinning at call time reads more naturally. The neutral
entry is preferred when writing arch-agnostic code (e.g. `local h
= alc.nn.preset(cfg.arch, cfg.variant)`).

#### `alc.nn.preset.tinyllama(variant, opts?)`

Build a trainable TinyLlama handle. Mirrors `alc.nn.preset.gpt2`:
same `opts` (device / dtype / pretrained), same VarMap `Some` /
`None` semantics. `pretrained=true` requires an HF-mapped variant
(`tinyllama-1.1b`); the `tinyllama-tiny` smoke variant only
supports `pretrained=false` (no HF bundle at that size).

**Parameters:**

| name    | type   | required | notes                                        |
|---------|--------|----------|----------------------------------------------|
| variant | string | yes      | `"tinyllama-1.1b"` / `"1.1b"` / `"tinyllama-tiny"` / `"tiny"` |
| opts    | table  | no       | `device` / `dtype` / `pretrained` — same shape as gpt2 preset |

**Returns:** `TinyLlamaHandle` UserData (adds `:kv_heads()` on top
of the Gpt2Handle method set).

```lua
local h = alc.nn.preset.tinyllama("tinyllama-1.1b", {
    device = "cuda:0",
    dtype = "bf16",
})
```

#### `alc.nn.preset.llama(variant, opts?)`

Build an inference-only Llama-family handle by wrapping
`candle_transformers::models::llama::Llama` behind the same UserData shape
`alc.nn.preset.gpt2` returns.

**Parameters:**

| name    | type   | required | notes                                         |
|---------|--------|----------|-----------------------------------------------|
| variant | string | yes      | `"tiny"` / `"7b-v1"` / `"7b-v2"` (or the `"llama-*"` aliases) |
| opts    | table  | no       | `device` / `dtype` / `weights` / `use_kv_cache` / `flash_attn` |

**opts fields (all optional):**

- `device` — `"cpu"` / `"cuda"` / `"cuda:N"` / `"metal"` / `"metal:N"`.
  Default: `"cpu"`. `"cuda"` requires a `--features nn-cuda` build;
  `"metal"` requires a `--features nn-metal` build (available on
  macOS). Unrecognised strings error with the full accepted list.
- `dtype` — `"f32"` / `"bf16"` / `"f16"`. Default: `"bf16"` on CUDA,
  `"f16"` on Metal, `"f32"` elsewhere. The device / dtype matrix is
  `f32` on any device, `bf16` on CUDA only, `f16` on CUDA / Metal /
  CPU (but CPU `f16` is slow). `bf16` on CPU or Metal errors up
  front with a message that points at the working combination.
- `weights` — a single safetensors path or an array of paths for a
  sharded bundle. Absent → random VarMap-backed handle (only useful
  for the `"tiny"` variant's shape assertions; production callers must
  supply weights).
- `use_kv_cache` — `bool`, default `true`. `false` disables the
  streaming KV cache for one-shot benchmarks.
- `flash_attn` — `bool`, default `false`. Enables fused attention on
  CUDA builds compiled with the `flash-attn` cargo feature.

**Returns:** `LlamaHandle` UserData. Methods: `:variant()`,
`:layers()`, `:heads()`, `:kv_heads()`, `:dim()`, `:ctx()`, `:vocab()`,
`:device()`, `:dtype()`, `:forward_shape(batch, seq) -> {batch, vocab}`.

**Errors:**

- `alc.nn.preset.llama: unknown variant '...'` — variant is not one of
  the accepted names.
- `alc.nn.preset.llama: bf16 dtype requires a CUDA device (use dtype='f32' on CPU)`.
- `alc.nn.preset.llama: opts.weights must be a string or an array of strings`.

```lua
-- Shape check on the tiny (offline) variant:
local h = alc.nn.preset.llama("tiny")
assert(h:forward_shape(1, 4)[2] == h:vocab())

-- Real load from local safetensors shards:
local h = alc.nn.preset.llama("7b-v2", {
    device = "cuda:0",
    dtype = "bf16",
    weights = { "/models/llama-7b/model-00001-of-00002.safetensors",
                "/models/llama-7b/model-00002-of-00002.safetensors" },
})
```

Notes:

- **Inference-only.** The handle does not carry a `VarMap`, so
  `alc.nn.trainer.full_ft` / `.lora` / `.distill` refuse it. Training
  on Llama is scoped to a sibling issue.
- **Adapter over candle-transformers.** The Rust-side wrapper lives in
  `algocline_nn::arch::adapter::llama`. Adding a new architecture
  (Qwen2 / Phi / Gemma) is a sibling module on the same shape.
- Wrap the returned handle in a Lua closure and pass it to
  `alc.nn.register(name, forward)` to expose the model through
  `alc.llm(prompt, { role = "nn", model = name })`.

#### `alc.nn.card.load_handle(card_id) -> NnHandle`

Arch-neutral loader for **self-contained** cards (`training_path`
= `full_ft` / `merged` / `distillation`). Reads the Card, resolves
the arch via `card.metadata.nn.architecture`, dispatches through
the bridge's arch registry, mmaps the safetensors bundle at
`<nn_dir>/<card_id>.safetensors`, and returns an `NnHandle`
UserData whose variant matches the card's arch.

**Errors:**

- `alc.nn.card.load_handle: card '...' not found`.
- `alc.nn.card.load_handle: card '...' has training_path="lora"; ... call `alc.nn.card.load_wrap(card_id, base)` instead` — LoRA cards need a base handle; directed to `load_wrap`.
- `alc.nn.card.load_handle: card '...' has unknown training_path "..."` — training_path is not one of the accepted values.
- `alc.nn.card.load_handle: card '...' architecture '...' has no bridge dispatch` — arch family is not registered on the bridge.
- `alc.nn.card.load_handle: card '...' architecture '...' does not support self-contained card load` — arch is registered but its `build_from_safetensors` slot is `None` (adapter-style archs like the current Llama adapter).
- `alc.nn.card.load_handle: bundle missing at ...` — bundle file has been cleaned or the Card's `bundle_ref` diverges from `nn/<card_id>`.

```lua
-- Load a merged (or full_ft / distillation) card as a fresh handle:
local h = alc.nn.card.load_handle("cards/domain-merged-042")
print(h:arch(), h:layers(), h:vocab())
```

#### `alc.nn.card.load_wrap(card_id, base) -> NnHandle`

Arch-neutral loader for **LoRA** cards (`training_path == "lora"`).
Takes a base handle (matching the card's arch), wraps the base
model in-place with a fresh LoRA layout that reproduces the
training-time `LoraConfig`, loads the recorded delta safetensors
into the wrap's fresh `VarMap`, and returns a new `NnHandle`
sharing the mutated base's Arc.

`base` accepts either an `NnHandle` (from `alc.nn.preset(arch,
…)`) or a typed handle (from `alc.nn.preset.<arch>(…)`) for
backward compat. Arch must match the card's arch.

**Errors:**

- `alc.nn.card.load_wrap: card '...' has training_path="..."; self-contained cards do not need a base handle — call `alc.nn.card.load_handle(card_id)` instead` — non-LoRA cards; directed to `load_handle`.
- `alc.nn.card.load_wrap: <arch> card requires a <arch> base handle; got '<other>'` — arch mismatch between card and base.
- Same schema-shape errors as `load_gpt2` (missing candle / lora / delta_path / delta file).

```lua
-- Load a LoRA card on top of a fresh base:
local base = alc.nn.preset("gpt2", "medium")
local h = alc.nn.card.load_wrap("cards/domain-lora-042", base)
```

#### `alc.nn.card.load_gpt2(card_id, base) -> Gpt2Handle` *(deprecated)*

Arch-pinned typed shortcut for GPT-2 LoRA card load. Delegates to
the shared `wrap_gpt2_lora_from_meta` core; identical observable
behaviour to `alc.nn.card.load_wrap(card_id, base)` when `base` is
a `Gpt2Handle`, but the returned handle is typed as `Gpt2Handle`
(not `NnHandle`) so callers can pass it into
`alc.nn.trainer.full_ft` / `.distill` etc. that still borrow the
typed handle directly.

**Migration:** call `alc.nn.card.load_wrap(card_id, base)` instead.
The trainer bindings that require a typed `Gpt2Handle` are
scheduled to accept `NnHandle` in a follow-up release; until they
do, `load_gpt2` stays as a working shim.

#### `alc.nn.card.load_vars(card_id) -> table`

Legacy raw-vars loader — returns a Lua table of `alc.nn.var`
tensors keyed by the safetensors names. Historically named
`alc.nn.card.load`; renamed to `load_vars` to free the `load`
slot for the arch-neutral handle-returning entry (currently exposed
as `load_handle` during the deprecation window). The old `load`
name continues to work as an alias for `load_vars` until the
deprecation cycle closes; new callers should use `load_vars`
explicitly.

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

**`[run]` section — Run-path outcome (opt-in, gated by `[setting.card].run`)**

Strategies driving production Runs (not just eval sweeps) can attach
per-run outcome data to a Card without leaving the primary
`alc.card.create` API. The section carries three fields:

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `status` | string enum | yes (when `run` is present) | One of `"succeeded"`, `"failed"`, `"skipped"`. Unrecognized values raise a Lua error before the write. |
| `reason` | string | no | Free-text explanation. Passed through to the LLM prompt when this Card is later injected via `card_context` (see `alc.llm`), so newlines are stripped there for template safety. |
| `action` | string | no | Free-form tag for the action tried (e.g. `"write"`, `"read"`, `"refine"`). |

The `[run]` section is **gated** by `[setting.card].run` (see
`README.md` §Global settings). The default is **off**, so existing
strategies that don't populate `run` see zero behavior change.

- When the gate is **off** and a caller passes `run`, `alc.card.create`
  / `alc.card.append` become a no-op and return `nil`: no file is
  written, no CardEvent is published, and the CardStore is untouched.
- When the gate is **on**, the section is written verbatim into
  Tier 1 alongside `pkg` / `stats` / `metadata` etc. and appears in
  `alc.card.get`'s returned table.

Enable per-project via `alc.local.toml`, per-user via `config.toml`, or
per-session via `ALC_SETTING_CARD_RUN=true`.

```lua
alc.card.create({
    pkg = { name = "cot" },
    stats = { pass_rate = 0.75 },
    run = {
        status = "failed",
        reason = "grader returned rating < 3",
        action = "write",
    },
})
-- With [setting.card].run = true: writes a Card whose TOML carries
--   [run]
--   status = "failed"
--   reason = "grader returned rating < 3"
--   action = "write"
-- With [setting.card].run absent or false: returns nil, no write.
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

### Card context injection

`alc.llm(prompt, {card_context = ...})` prepends resolved Cards to the
system prompt as a fixed XML-like block, giving the model a lightweight
few-shot view of prior Run outcomes without the strategy having to
hand-format anything. The block sits ahead of `opts.system`; when
`opts.system` is absent, the block becomes the whole system prompt.

**Spec forms:**

| `card_context` value | Resolves to |
|----------------------|-------------|
| string | The single Card with that `card_id` (empty result on miss). |
| `{pkg = "<name>", limit = N}` | The N most recent Cards for that pkg, ordered by `created_at` desc. `limit` defaults to 5; values above 100 are silently capped to keep the N+1 fetch bounded. |

Unknown forms (list of ids, alias, `where` filter, etc.) are silently
ignored — resolution failure is a no-op, not a Lua error, so `alc.llm`
behavior is unchanged when the id/pkg is missing.

**Fixed template** (each Card on one line, `created_at` desc):

```
<past_cards>
Card MM/DD pkg=<pkg> card_id=<id> [run.status=<status>] Rating <val> reason=<reason>
Card MM/DD pkg=<pkg> card_id=<id> [run.status=<status>]
...
</past_cards>
```

- `[run.status=...]` / `Rating <val>` / `reason=<r>` are each emitted
  only when the Card carries the corresponding field (`[run]` section
  from Phase 1 `alc.card.create({run=...})`, `stats.pass_rate` for
  Rating).
- `reason` / `pkg` / `card_id` / `run.status` are line-sanitized:
  newlines in stored values become spaces so a Card cannot break out
  of the `<past_cards>` wrapper.

```lua
-- Example: strategy consults its own last 5 Cards before proposing a
-- new prompt variant.
local response = alc.llm("Draft the next variant.", {
    system = "You are a prompt optimizer.",
    card_context = { pkg = "cot_variant", limit = 5 },
})

-- Emitted system prompt:
-- <past_cards>
-- Card 07/18 pkg=cot_variant card_id=cot_variant_... [run.status=succeeded] Rating 4.5
-- Card 07/17 pkg=cot_variant card_id=cot_variant_... [run.status=failed] reason=grader<3
-- ...
-- </past_cards>
--
-- You are a prompt optimizer.
```

**Prerequisites for populating Cards for injection**: the `[run]`
section only lands on disk when `[setting.card].run` is enabled (see
§Schema Conventions above); strategies that never write `run` still
resolve into the block using `pkg` / `card_id` / `stats.pass_rate`,
just without the `[run.status=...]` and `reason=...` suffixes.

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

---

## Test Sandbox (`alc_pkg_test`) — Mock API

When a package spec is executed via the `alc_pkg_test` MCP tool, the VM is
preloaded with the full `alc.*` primitive surface (so specs can call
`alc.json_encode`, `alc.fingerprint`, `alc.fuzzy.*`, etc. directly) and a
Pure-Lua mock layer that lets spec authors swap external-I/O entries
(`alc.llm`, `alc.llm_batch`, `alc.fork`) without monkey-patching.

The invariant enforced at test time is
`production primitive surface ⊆ test sandbox primitive surface`: every
`alc.*` key reachable inside a live `alc_run` session is also reachable
inside an `alc_pkg_test` spec. The regression guard lives in
`crates/algocline-engine/tests/bridge_sandbox_parity.rs`.

### `with_alc(overrides, fn) -> any`

Scoped override. Replaces the listed `alc.*` entries for the duration of
`fn()`, then restores the previous values — even when `fn()` raises.

```lua
it("classifies pass when llm returns true", function()
  with_alc({
    llm = function(prompt) return "true" end,
    cache_get = function(_) return nil end,
  }, function()
    expect(my_pkg.classify("ok")).to.equal(true)
  end)
end)
```

Nested `with_alc` calls form a LIFO stack: an inner restore returns to
the immediately enclosing override, not to the original value.

### `alc_mock.install(overrides)` / `alc_mock.restore()`

Persistent override for `before_each` / `after_each` setup. Each
`install` pushes one frame onto the same restore stack as `with_alc`;
`restore()` pops the most-recent frame.

```lua
describe("my_pkg", function()
  before_each(function()
    alc_mock.install({ llm = function() return "stub" end })
  end)
  after_each(alc_mock.restore)
  -- ... tests that call my_pkg, which itself calls alc.llm ...
end)
```

`alc_mock.restore_all()` drops every pushed frame (teardown safety net).

### `alc.spy(name, default_fn?) -> handle`

Wrap an `alc.*` entry with an observable proxy. The proxy still calls
`default_fn` (or the previous entry if `default_fn` is `nil`) on
invocation, while recording each call on the returned handle.

```lua
local llm_spy = alc.spy("llm", function(p) return "ok" end)
my_pkg.run("task")
expect(llm_spy.call_count).to.equal(3)
expect(llm_spy.calls[1].args[1]).to.match("task")
llm_spy:reset()  -- clears call_count and calls
```

The spy is pushed onto the same restore stack as `with_alc` /
`alc_mock.install`, so spies installed inside a `with_alc` body are
automatically torn down when the body exits.

### External-I/O Stubs

`alc.llm`, `alc.llm_batch`, and `alc.fork` are present in the sandbox
but, by default, raise a "mock required" error when called:

```
mock required: alc.llm — wrap the call in `with_alc({ llm = fn }, fn)`
inside your spec (alc_pkg_test sandbox stubs external I/O by design)
```

Specs that exercise these code paths must wrap their assertions in
`with_alc({ llm = ... }, fn)` (or `alc_mock.install({ llm = ... })`).
This is enforced by design — `alc_pkg_test` runs in an offline sandbox
with no LLM channel or fork executor wired up.
