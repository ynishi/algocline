# algocline

**Research-grade reasoning for your LLM — as Pure Lua you can read, edit, and ship.**

UCB1 exploration, self-reflection, multi-agent debate, chain-of-verification — each strategy is a single Lua file. Install with `cargo install`, customize anything, no framework lock-in.

## What it does

LLMs are single-shot by default. Ask a question, get one answer, hope it's right. Research papers describe techniques that do better — iterating, scoring, selecting — but using them today means Python, framework lock-in, and API key management.

algocline makes these techniques **immediately usable**. Each algorithm is a Pure Lua file (100-350 lines) that runs inside your existing MCP host. No infrastructure. No setup beyond `alc init`.

```
Human ──→ LLM (MCP host) ──→ algocline ──→ Algorithm (Lua) ──→ alc.llm() ──→ back to LLM
                                                                    ↑              │
                                                                    └──────────────┘
                                                                   (loop until done)
```

**Use** existing algorithms with one call. **Build** your own by writing Lua. **Share** them via Git.

## When to use

| What you want to do | What happens | Strategy |
|---|---|---|
| Catch what your code review missed | Draft → self-critique → revise. Found that `std::fs::write` is non-atomic (crash = data loss) after the first draft said "no issues" | `reflect` |
| Make design decisions structurally, not by gut | Advocate, critic, and pragmatist debate. Resolved "do we need log rotation?" → "no, but add a display limit" in 4 rounds | `panel` |
| Fact-check LLM claims before trusting them | Auto-generates verification questions → checks each one. Escalated a path traversal from medium to high severity after confirming `PathBuf::join()` does zero validation | `cove` |
| Break down a complex problem step by step | Builds reasoning chain incrementally. Structured Vec vs HashMap trade-offs into cache locality → memory overhead → API ergonomics | `cot` |
| Pick the best option from multiple candidates | Scores each candidate on your criteria, pairwise comparison | `rank` |
| Get a reliable answer, not a lucky one | Generates N answers → majority vote picks the most consistent one | `sc` |

## Why algocline

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

### 2. Initialize packages

```bash
alc init
```

This downloads the bundled package collection (UCB1, chain-of-thought, self-consistency, etc.) into `~/.algocline/packages/`. Run `alc init --force` to overwrite existing packages.

It also distributes `alc.d.lua` — LuaCats type definitions for all `alc.*` StdLib functions — to `~/.algocline/types/alc.d.lua`. This enables editor completion in any editor backed by [lua-language-server](https://github.com/LuaLS/lua-language-server). If `.luarc.json` is not present in the current directory, a setup tip is printed.

> **Note**: If you skip this step, packages are auto-installed on first `alc_advice` call via MCP. But running `alc init` upfront is recommended for faster first use and offline availability.

### 3. Add to your MCP config

Add algocline as an MCP server in your host's configuration (e.g. Claude Code's `~/.claude.json`, Cursor's MCP settings, etc.):

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

After adding the config, restart your MCP host session so it picks up the new server.

#### Environment variables

| Variable | Description | Default |
|---|---|---|
| `ALC_LOG_DIR` | Directory for session transcript logs | `~/.algocline/logs` |
| `ALC_LOG_LEVEL` | `full` (enable logging) or `off` (disable) | `full` |
| `ALC_PACKAGES_PATH` | Additional package search paths (colon-separated). Takes priority over `~/.algocline/packages/` | (none) |
| `ALC_PROJECT_ROOT` | Project root directory for project-local package resolution. When omitted, auto-detected by walking up from cwd to find `alc.toml` | (auto-detect) |

Example: writing logs to a custom directory:

```json
{
  "mcpServers": {
    "algocline": {
      "command": "alc",
      "env": {
        "ALC_LOG_DIR": "/path/to/custom/logs"
      }
    }
  }
}
```

### 4. Use

Call algocline tools from your MCP host. The host LLM calls these tools on your behalf:

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
│  alc.llm_batch(items)          — parallel LLM calls (single round-trip)
│  alc.fork(strategies, ctx, o?) — parallel multi-VM strategy execution
│  alc.json_encode/json_decode   — serde_json bridge
│  alc.log(level, msg)           — tracing bridge
│  alc.state.get/set/keys/delete — persistent key-value store
│  alc.match_enum(text, cs, o?)  — fuzzy enum match from LLM output
│  alc.match_bool(text)          — yes/no normalizer for LLM output
│  alc.budget_remaining()        — remaining budget (calls/time)
│  alc.progress(step, total, m?) — structured progress reporting
│
Layer 1: Prelude Combinators (Lua → alc.*)
│  alc.cache(prompt, opts?)      — memoized LLM call (session-scoped)
│  alc.parallel(items, fn, o?)   — batch-parallel LLM over array
│  alc.map(items, fn)            — transform each element
│  alc.reduce(items, fn, init)   — fold to single value
│  alc.vote(answers)             — majority aggregation
│  alc.filter(items, fn)         — conditional selection
│  alc.json_extract(raw)         — extract JSON from LLM output
│  alc.parse_number(text, pat?)  — extract number from LLM output
│  alc.state.update(key, fn)     — read-modify-write for state
│  alc.llm_safe(prompt, opts, d) — non-throwing LLM wrapper
│  alc.llm_json(prompt, opts?)   — LLM call with JSON parse + retry
│  alc.fingerprint(str)          — normalize + DJB2 hash (dedup)
│  alc.tuning(defaults, ctx)     — config merge with deep merge
│  alc.budget_check()            — boolean budget guard
│  alc.pipe(strategies, ctx)     — sequential strategy pipeline
│  alc.eval(scenario, strategy)  — evalframe facade (eval + Card)
│
Layer 2: Bundled Packages (require() from ~/.algocline/packages/)
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
   panel      — multi-perspective deliberation      [synthesis]
```

Layer 0/1 are always available. Layer 2 packages are installed via `alc init` or `alc_pkg_install` from [algocline-bundled-packages](https://github.com/ynishi/algocline-bundled-packages) and loaded via `require()`.

### Crate structure

```
algocline (bin: alc)
├── algocline-core      — Domain types, EngineApi trait (transport-independent API surface)
├── algocline-engine    — Lua VM executor, session registry, bridge
└── algocline-mcp       — MCP tool handlers (alc_run, alc_advice, etc.)
```

### Execution model

Each `alc_run` / `alc_advice` call spawns a **dedicated Lua VM** (OS thread + mlua instance). Concurrent sessions are fully isolated — each session's `alc`, `ctx`, and `package.loaded` live in their own VM, so parallel executions cannot interfere with each other.

`alc.llm()` is a **cooperative yield**. When Lua calls it, the VM pauses and returns the prompt to the MCP host. The host processes the prompt with its own LLM, then calls `alc_continue` with the response to resume execution.

```
alc_run(code)
  → Spawn dedicated VM → Lua executes → alc.llm("prompt") → VM pauses
  → returns { status: "needs_response", prompt: "...", session_id: "..." }

alc_continue({ session_id, response })
  → Lua resumes → ... → alc.llm("next prompt") → VM pauses again
  → ...repeat until Lua returns a final value → VM cleaned up
```

## MCP Tools

| Tool | Description |
|---|---|
| `alc_run` | Execute Lua code with optional JSON context |
| `alc_continue` | Resume a paused execution with the host LLM's response |
| `alc_advice` | Apply an installed package by name |
| `alc_pkg_link` | Link a local directory as a project-local package via symlink. Records path in `alc.lock` |
| `alc_pkg_unlink` | Remove a symlink created by `alc_pkg_link` (rejects real directories) |
| `alc_pkg_list` | List installed packages with metadata. Pass `project_root` to include project-local packages |
| `alc_pkg_install` | Install a package or collection from Git URL or local path. Response includes `types_path` (absolute path to `alc.d.lua`) |
| `alc_pkg_remove` | Remove an installed package from `alc.toml` + `alc.lock`. Pass `project_root` to target project scope |
| `alc_init` | Initialize a project — creates `alc.toml` in the project root if absent |
| `alc_update` | Update packages declared in `alc.toml` by re-installing from their recorded sources |
| `alc_migrate` | Migrate a legacy `alc.lock` to the new `alc.toml` + `alc.lock` schema |
| `alc_eval` | Evaluate a strategy against a scenario (cases + graders) |
| `alc_eval_history` | List past eval results, filter by strategy |
| `alc_eval_detail` | View a specific eval result in full detail |
| `alc_eval_compare` | Compare two eval results with Welch's t-test |
| `alc_note` | Add a note to a completed session's log |
| `alc_log_view` | View session logs (list or detail) |
| `alc_stats` | Aggregate usage stats across sessions (per-strategy) |
| `alc_status` | Query active session status, progress, and metrics |
| `alc_info` | Show server configuration and diagnostics |
| `alc_hub_search` | Search packages across remote Hub indices + local state |
| `alc_hub_info` | Show detailed info for a single package (metadata, Cards, aliases, stats) |
| `alc_hub_reindex` | Rebuild Hub index from locally installed packages |
| `alc_hub_dist` | Generate and distribute documentation for a package hub (`hub`, `context7`, `devin`, `lint`, `lint_only`, `luacats`, `narrative`, `llms` projections) |
| `alc_hub_gendoc` | Generate documentation for a single package (`hub`, `context7`, `devin`, `lint`, `lint_only`, `luacats`, `narrative`, `llms` projections) |
| `alc_pkg_scaffold` | Generate a minimal package skeleton with `M.meta` / `M.run` template and pre-filled `alc_shapes_compat` range |
| `alc_scenario_list` | List installed eval scenarios |
| `alc_scenario_show` | Show an installed scenario's content |
| `alc_scenario_install` | Install scenarios from Git URL or local path |

## Host integration patterns

algocline's `alc.llm()` is a cooperative yield — it pauses the Lua VM and returns a prompt to the host. How the host handles this determines performance and quality.

### Pattern 1: Manual loop (baseline)

The host LLM reads each prompt, generates a response, and calls `alc_continue`. Simple but requires one round-trip per `alc.llm()` call.

```
Host LLM → alc_advice → needs_response → Host reads prompt → Host generates response → alc_continue → repeat
```

**Best for**: Interactive exploration where you want to inspect each step.

### Pattern 2: Autonomous agent delegation (recommended)

Delegate the entire strategy execution to a single agent that has MCP tool access. The agent calls `alc_advice`, handles every `needs_response` internally, and returns only the final result.

```
Host LLM → Agent(MCP-capable) → [alc_advice → needs_response → self-respond → alc_continue → ...] → final result
```

**Best for**: Production use. Zero host intervention. Fastest execution.

Example (Claude Code):

```
Agent(general-purpose) with prompt:
  1. Call alc_advice(strategy="explore", task="...")
  2. For each needs_response: generate a response following the system/prompt instructions
  3. Call alc_continue with your response
  4. Repeat until status="completed"
  5. Return the final result
```

> **Known limitation: MCP tool permissions in subagents**
>
> Claude Code subagents may not inherit MCP tool permissions from the parent session's `settings.json`. If `alc_run`/`alc_continue` calls fail with permission errors inside a subagent, add the following to your `~/.claude/settings.json` under `permissions.allow`:
>
> ```json
> "mcp__alc__*"
> ```
>
> If the issue persists, run the `alc_run`/`alc_continue` loop directly from the main agent instead of delegating to a subagent.

### Pattern 3: MCP Sampling (future)

When the MCP host supports server-initiated sampling, `alc.llm()` will resolve automatically without pausing. No agent delegation needed — the host responds inline.

### Performance comparison

Benchmarked on the same task (UCB1 explore, 11 LLM calls):

| Pattern | Time | Host interventions | Notes |
|---|---|---|---|
| Manual (Opus) | 152s | 9 | High quality, manual effort |
| SubAgent relay (Haiku) | 224s | 9 | Agent startup overhead per call |
| SubAgent relay (Opus) | 442s | 9 | Same overhead, better quality |
| **Autonomous agent** | **69s** | **0** | Single agent, full MCP access, no relay overhead |

The autonomous agent pattern eliminates relay overhead entirely. The agent handles the full `alc_advice → alc_continue` loop in-process, resulting in ~2x faster execution than even manual operation.

### Strategy selection guide

| Strategy | Structure | Best for |
|---|---|---|
| `explore` / `ucb` | Generate hypotheses → UCB1 score → refine best | Open-ended questions, design decisions |
| `triad` | 3-role debate (proponent/opponent/judge) | Comparative analysis, pro/con evaluation |
| `panel` | Multi-perspective deliberation + moderator | Complex problems needing diverse viewpoints |
| `verify` / `cove` | Draft → verify → revise | Factual accuracy, reducing hallucination |
| `reflect` | Generate → critique → revise loop | Iterative improvement |
| `ensemble` / `sc` | Multiple answers → majority vote | When consistency matters more than novelty |
| `rank` | Pairwise tournament ranking | Selecting best among candidates |
| `factscore` | Atomic claim decomposition + verification | Fact-checking, claim validation |
| `cod` | Iterative information densification | Summarization, compression |

## Evaluating strategies

algocline includes a built-in evaluation framework powered by [evalframe](https://github.com/ynishi/evalframe). Define scenarios with test cases and graders, then run them against any strategy to measure quality.

### Define a scenario

A scenario is a Lua table with bindings (graders) and cases (input/expected pairs):

```lua
-- scenario.lua
local ef = require("evalframe")
return {
  ef.bind { ef.graders.contains },
  cases = {
    ef.case { input = "What is 2+2?", expected = "4" },
    ef.case { input = "Capital of France?", expected = "Paris" },
  },
}
```

### Run an eval

```
alc_eval({ scenario: "...", strategy: "cove" })
```

Or point to a file:

```
alc_eval({ scenario_file: "/path/to/scenario.lua", strategy: "cove" })
```

The strategy is automatically wired as the provider — no boilerplate needed. Results include per-case scores, pass/fail status, and aggregate metrics.

### Track and compare results

Eval results are persisted to `~/.algocline/evals/` automatically.

```
alc_eval_history({ strategy: "cove", limit: 10 })   # List past results
alc_eval_detail({ eval_id: "cove_1710672000" })      # Full result detail
alc_eval_compare({ eval_id_a: "...", eval_id_b: "..." })  # Welch's t-test
```

`alc_eval_compare` performs a Welch's t-test on the score distributions, reporting whether the difference between two runs is statistically significant.

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

### Bundled packages

Bundled packages are maintained in [algocline-bundled-packages](https://github.com/ynishi/algocline-bundled-packages). Install them via CLI:

```bash
alc init            # Download and install all bundled packages
alc init --force    # Overwrite existing packages
```

If you call `alc_advice` with a package that isn't installed, algocline **automatically downloads the bundled collection** from GitHub. But `alc init` upfront is recommended.

To install or update via MCP:

```
alc_pkg_install({ url: "github.com/ynishi/algocline-bundled-packages" })
```

For local development (installs from a local checkout, supports uncommitted changes):

```
alc_pkg_install({ url: "/path/to/algocline-bundled-packages" })
```

### Installing third-party packages

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

### Project-local packages

Project-local packages are managed via two files at the project root:

- **`alc.toml`** — Package declarations (source of truth). Created by `alc_init` or automatically on first `alc_pkg_install`.
- **`alc.lock`** — Resolved lockfile written by install/link operations.

Initialize a project:

```
alc_init({ project_root: "/path/to/project" })
```

Link a local directory as a project-scoped package via symlink (no copy):

```
alc_pkg_link({ path: "/path/to/my-strategy" })
```

This creates a symlink in `~/.algocline/packages/` and records the entry in `alc.toml` + `alc.lock`:

```toml
# alc.toml
[packages.my-strategy]
source = "path"
path = "/path/to/my-strategy"
```

Remove the symlink:

```
alc_pkg_unlink({ name: "my-strategy" })
```

Project root is auto-detected by walking up the directory tree to find `alc.toml`. You can also pass `project_root` explicitly.

Resolution order (highest priority first):

1. `alc.lock` `path` entries — symlinked local directories
2. `ALC_PACKAGES_PATH` (environment)
3. `~/.algocline/packages/` (global default)

Migrate an existing project using the old `alc.lock` schema:

```
alc_migrate({ project_root: "/path/to/project" })
```

Use `project_root` to activate project scope in other tools:

```
alc_pkg_list({ project_root: "/path/to/project" })    # Lists both project and global packages
alc_pkg_remove({ name: "my-strategy", project_root: "/path/to/project" })  # Removes from alc.toml + alc.lock
alc_update({ project_root: "/path/to/project" })       # Re-installs all alc.toml packages
```

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
