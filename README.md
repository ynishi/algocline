# algocline

**Algorithm + LLM + Human.** State-of-the-art reasoning algorithms — UCB1 exploration, multi-agent debate, ensemble voting — available to anyone, instantly, as an MCP server.

## What it does

Research papers describe powerful techniques for improving LLM reasoning: Monte Carlo Tree Search, multi-perspective deliberation, iterative refinement. But using them today means Python environments, framework lock-in, and API key management.

algocline makes these techniques **immediately usable**. Each algorithm is a Pure Lua file (100-350 lines) that runs inside your existing MCP host. No infrastructure. No setup beyond `alc init`.

```
Human ──→ LLM (MCP host) ──→ algocline ──→ Algorithm (Lua) ──→ alc.llm() ──→ back to LLM
                                                                    ↑              │
                                                                    └──────────────┘
                                                                   (loop until done)
```

**Use** existing algorithms with one call. **Build** your own by writing Lua. **Share** them via Git.

## Why

| Approach | Limitation |
|---|---|
| Prompt engineering | Single-shot. No iteration, no scoring, no selection |
| DSPy / LangGraph | Python infra. Pip install. API keys. Framework lock-in |
| **algocline** | Zero infra. Pure Lua. Runs inside your existing MCP host. Use research-grade algorithms today, or write your own |

## Quick start

### 1. Install

```bash
cargo install algocline
```

### 2. Add to your MCP config

```json
{
  "mcpServers": {
    "algocline": {
      "command": "alc",
      "env": {}
    }
  }
}
```

### 3. Install official packages

```bash
alc init
```

### 4. Use

One-liner with an installed package:

```
alc_advice({ strategy: "ucb", task: "Design a rate limiter for a REST API" })
```

Or write your own Lua:

```lua
-- alc_run({ code: "..." })
local draft = alc.llm("Draft a solution for: " .. ctx.task)
local critique = alc.llm("Find flaws in:\n" .. draft)
local final = alc.llm("Revise based on critique:\n" .. draft .. "\n\nCritique:\n" .. critique)
return { result = final }
```

## Architecture

### Three-Layer StdLib

```
Layer 0: Runtime Primitives (Rust → alc.*)
│  alc.llm(prompt, opts?)        — Host LLM call via MCP Sampling
│  alc.json_encode/json_decode   — serde_json bridge
│  alc.log(level, msg)           — tracing bridge
│  alc.state.get/set/keys/delete — persistent key-value store
│
Layer 1: Prelude Combinators (Lua → alc.*)
│  alc.map(items, fn)            — transform each element
│  alc.reduce(items, fn, init)   — fold to single value
│  alc.vote(answers)             — majority aggregation
│  alc.filter(items, fn)         — conditional selection
│
Layer 2: Packages (require() from ~/.algocline/packages/)
   cot        — chain-of-thought                    [reasoning]
   maieutic   — maieutic prompting                  [reasoning]
   reflect    — self-reflection                     [reasoning]
   calibrate  — confidence calibration              [reasoning]
   sc         — self-consistency (majority vote)     [selection]
   rank       — pairwise ranking                    [selection]
   triad      — triad comparison                    [selection]
   ucb        — UCB1 hypothesis exploration          [selection]
   sot        — skeleton-of-thought                 [generation]
   decompose  — task decomposition                  [generation]
   distill    — knowledge distillation              [extraction]
   cod        — chain-of-density                    [extraction]
   cove       — chain-of-verification               [validation]
   factscore  — factual precision scoring           [validation]
   review     — structured code/text review         [validation]
   panel      — multi-perspective deliberation      [synthesis]
```

Layer 0/1 are always available. Layer 2 packages are installed separately from [algocline-packages](https://github.com/ynishi/algocline-packages) and loaded via `require()`.

### Crate structure

```
algocline (bin: alc)
├── algocline-core      — Domain types (SessionId, QueryId, TickResult)
├── algocline-engine    — Lua VM executor, session registry, bridge
└── algocline-mcp       — MCP tool handlers (alc_run, alc_advice, etc.)
```

### Execution model

`alc.llm()` is a **cooperative yield**. When Lua calls it, the VM pauses and returns the prompt to the MCP host. The host processes the prompt with its own LLM, then calls `alc_continue` with the response to resume execution.

```
alc_run(code)
  → Lua executes → alc.llm("prompt") → VM pauses
  → returns { status: "needs_response", prompt: "...", session_id: "..." }

alc_continue({ session_id, response })
  → Lua resumes → ... → alc.llm("next prompt") → VM pauses again
  → ...repeat until Lua returns a final value
```

## MCP Tools

| Tool | Description |
|---|---|
| `alc_run` | Execute Lua code with optional JSON context |
| `alc_continue` | Resume a paused execution with the host LLM's response |
| `alc_advice` | Apply an installed package by name |
| `alc_pkg_list` | List installed packages |
| `alc_pkg_install` | Install a package or collection from Git URL |
| `alc_pkg_remove` | Remove an installed package |

## Writing strategies

A strategy is a Lua file with an `init.lua` entry point:

```lua
-- my-strategy/init.lua
local M = {}

M.meta = {
    name = "my-strategy",
    version = "0.1.0",
    description = "What it does",
}

function M.run(ctx)
    local task = ctx.task or error("ctx.task is required")

    -- Your reasoning algorithm here
    local step1 = alc.llm("Analyze: " .. task)
    local step2 = alc.llm("Given analysis:\n" .. step1 .. "\n\nSynthesize a solution.")

    ctx.result = { answer = step2 }
    return ctx
end

return M
```

Install it:

```
alc_pkg_install({ url: "github.com/you/my-strategy" })
```

Use it:

```
alc_advice({ strategy: "my-strategy", task: "..." })
```

## Package management

### Official packages

Official packages are maintained in [algocline-packages](https://github.com/ynishi/algocline-packages).

```bash
alc init                  # Install official packages (from GitHub Releases)
alc init --force          # Overwrite existing packages
alc init --dev            # Install from local algocline-packages/ (development)
```

### Installing packages

Via MCP tool:

```
alc_pkg_install({ url: "github.com/user/my-strategy" })
```

`alc_pkg_install` automatically detects the repository layout:

| Layout | Detection | Behavior |
|---|---|---|
| **Single package** | `init.lua` at repo root | Installed as one package |
| **Collection** | Subdirs containing `init.lua` | Each subdir installed as a separate package |

Supported URL formats:

| Format | Example |
|---|---|
| GitHub shorthand | `github.com/user/my-strategy` |
| HTTPS | `https://github.com/user/my-strategy.git` |
| SSH | `git@github.com:user/my-strategy.git` |
| Local path (file://) | `file:///path/to/my-strategy` |
| Local path (absolute) | `/path/to/my-strategy` |

Optional name override (single package mode only):

```
alc_pkg_install({ url: "/path/to/repo", name: "custom-name" })
```

### Managing packages

```
alc_pkg_list()                          # List installed packages with metadata
alc_pkg_remove({ name: "my-strategy" }) # Remove a package
```

Packages live in `~/.algocline/packages/`. Each package is a directory with an `init.lua`.

## Strategy development

Strategies are Pure Lua files. The host LLM can read, write, and execute them via `alc_run` — that's enough to iterate.

For debugging and testing, there are dedicated tools that run on the same mlua VM:

- [mlua-probe](https://github.com/ynishi/mlua-probe) — Lua debugger (MCP server). Breakpoints, stepping, variable inspection, expression evaluation
- [mlua-lspec](https://github.com/ynishi/mlua-lspec) — BDD test framework (`describe`/`it`/`expect`). Structured test results for CI/LLM consumption

## Contributing

Bug reports and feature requests are welcome — please [open an issue](https://github.com/ynishi/algocline/issues).

Pull requests are also appreciated. For larger changes, consider opening an issue first to discuss the approach.

### Share your strategies

Writing a strategy package is straightforward: create `init.lua`, define `M.meta` and `M.run(ctx)`, and you're done. If you build something useful, publish it as a Git repo and others can install it with:

```
alc_pkg_install({ url: "github.com/you/your-strategy" })
```

See [Writing strategies](#writing-strategies) and [Package management](#package-management) for details.

## License

MIT OR Apache-2.0
