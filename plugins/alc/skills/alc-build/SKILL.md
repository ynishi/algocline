---
name: alc-build
description: Delegates algocline package implementation to @alc-coder. Invoke (either by the User typing `/alc-build "<design_para>"` directly, or by the AI invoking it on the User's behalf) only once design_para is mature — package name, alc.llm usage intent, and pass conditions are all decided.
---

# /alc-build — algocline Package Implementation Kick

Hands a `design_para` finalized through design dialogue to `@alc-coder` and
lets it implement in an isolated context. Either the User invokes
`/alc-build "<design_para>"` literally, or the AI invokes it on the User's
behalf once the design conversation has matured (see "Maturity Self-check"
below).

A Skill to run only the moment dialogue on the main thread (User <-> AI) has
matured and the design has been condensed into a single paragraph. Never
trigger it in an intermediate state (when the design is vague, the package
name undecided, or the pass conditions missing).

### Maturity Self-check (AI-invoked path)

When the AI is about to invoke `/alc-build` on behalf of the User (rather
than the User typing it literally), the AI MUST verify all four of the
following from the conversation context before assembling the kick prompt.
If any item is missing, **do not invoke** — return one short question to
the User to fill the gap, then resume.

1. **Package name** is decided (snake_case, no collision concern raised).
2. **`alc.llm` usage intent** is stated (which prompt / loop drives the
   LLM, or an explicit "no `alc.llm` needed" declaration).
3. **Pass conditions** are stated (what `alc_pkg_test` should confirm).
4. **`--location` resolves unambiguously** for the current cwd (auto →
   either git root with `alc.toml` or `~/.algocline/packages`; collection →
   git root with `alc.toml` present; global → no check needed).

Skipping this self-check and dispatching with a half-baked `design_para`
wastes coder tokens and produces an unusable pkg skeleton. The check costs
one self-review pass; the cost of skipping is a full coder retry loop.

## Arguments

`/alc-build [--location=<auto|collection|global>] "<design_para, <= 200 chars>"`

`--location` (optional, default `auto`) decides where the package is written:

- `auto` (default) — if the current `cwd`'s git root contains `alc.toml`,
  treat that git root as a collection root and write under it;
  otherwise fall back to `~/.algocline/packages/` (global). This lets the
  user develop packages **inside their own arbitrary project directory** —
  any repo that carries `alc.toml` at its root is treated as a personal
  collection.
- `collection` — force-write under `<cwd's git root>/<pkg>/`. Errors out if the
  git root does not contain `alc.toml` (refuses to scatter pkg files into a
  non-collection repo).
- `global` — force-write under `~/.algocline/packages/<pkg>/` even when
  invoked from inside a collection repo.

> **Reserved name**: `--location=bundled` is **reserved** for a future mode
> targeting the official `algocline-bundled-packages` set (the canonical
> packages shipped/bundled with algocline core itself). It is **not**
> implemented now; do not use this enum value. User-owned collection repos
> — even one that happens to be a fork of `algocline-bundled-packages` —
> belong to `collection`, not `bundled`.

Required elements of `design_para`:

- **Package name** (snake_case; checking for collisions via `alc_pkg_list` at
  design time is recommended).
- **New-package origin** (for new packages, generate the skeleton via
  `alc_pkg_scaffold`; for changes to existing packages, state the base path
  explicitly).
- **`alc.llm` usage intent** (which prompt and loop drive the LLM).
- **Complete `M.meta`**: in addition to `name` / `version` / `description` /
  `category`, orchestrator and composite packages also need `docs` (usage +
  main helpers) and `narrative` (orchestrator story prose). The
  `alc_pkg_scaffold` default emits only the first four; the latter two must
  be filled in manually.
- **Pass conditions** (what `alc_pkg_test` should confirm to declare done —
  the coder's completion is bounded by this).
- **Card emission (swarm_frame orchestration packages only)**: wire
  `plugin_run_card.create({pkg_name="..."})` into
  `make_dispatcher({plugins=...})` (the default fixture for the nine major
  orchestrators; see wake -> Swarm framework). Single-shot strategy
  packages are not in scope, but orchestration packages should include it
  **unless there is a specific reason not to**.

**Pre-impl plan check recommended (optional, for important / complex
packages)**: before finalizing `design_para`, throw a plan-review query at
`@alc-adviser` so it can flag gaps, best-practice violations, divergences
from the major orchestrators, and required fixtures in a single turn. See
the "Pre-impl plan check" path at the bottom of wake. Light packages (<= 200
lines of Lua, single-shot strategies) can be skipped — it would be
over-engineering for trivial cases. Mid-sized packages and above, domain-
specific work, and reuse of major orchestrator patterns benefit from
running it (catches missing Cards / plugins). The adviser does not decide;
the main thread chooses whether to adopt the plan (client-decide pattern).

Example:

```
/alc-build "packages/quick_vote: use alc.llm to cast one vote per
candidate and aggregate the multi-arm voting result. spec/quick_vote_spec.lua
expects three-arm voting to return, with the vote sum matching the number of
arms as the pass condition."
```

## What It Does

1. **Receive `design_para` in full** — no truncation or summarization.
2. **Resolve `pkg_root` from `--location`** — decide where the package will
   be written (see "Package root resolution" below). The caller (this Skill)
   owns this decision; the Agent does not re-resolve.
3. **Resolve the journal configuration** — call
   `mcp__algocline__alc_setting_resolve(target="journal")` once and let ALC
   core resolve env / toml / default to return `{path, pkg}` (the caller
   does not own env / toml parsing logic).
4. **Assemble the kick prompt** — `design_para` + resolved `pkg_root` +
   journal configuration (`path` / `pkg`) + implementation-loop instructions
   (Write `init.lua` + spec -> `alc_pkg_test` -> Edit on failure -> <= 3
   retries) + return contract (three sections: `### Result` / `### Artifacts`
   / `### Key Observations`).
5. **Spawn `@alc-coder` via the `Task` tool** — no other Agents (external
   pipeline / implementation agents from other plugins) may be called.
6. Return the `result_summary` to the main thread and hand control back to
   the rendezvous where the User confirms, re-kicks, or injects a Delta.

### Scope

The responsibility of this Skill is **complete at the coder dispatch**.
Following the ALC core separation of concerns where
`alc_pkg_scaffold` only generates skeletons and `alc_pkg_install` registers
packages in the registry, **registry registration (via `alc_pkg_install`) is
out of scope for this Skill** — after the coder finishes, **the main AI
(main thread) must perform the install step explicitly**.

The standard post-impl install procedure run by the main AI is consolidated
in the footer of `/alc-wake` -> Post-impl install (not repeated here).

### Package Root Resolution

The `pkg_root` (the parent directory that will contain `<pkg>/init.lua`) is
decided by `--location` before the kick prompt is assembled:

| `--location` | Resolution |
|---|---|
| `global` | `pkg_root = ~/.algocline/packages` (no further checks). |
| `collection` | `git_root = $(git rev-parse --show-toplevel)`; require `<git_root>/alc.toml` to exist, otherwise abort with an error (refusing to scatter pkg files into a non-collection repo). `pkg_root = <git_root>`. |
| `auto` (default) | Try `git_root = $(git rev-parse --show-toplevel)`; if that succeeds AND `<git_root>/alc.toml` exists, use collection (`pkg_root = <git_root>`). Otherwise fall back to global (`pkg_root = ~/.algocline/packages`). |

Rationale: the primary use case is **the user developing packages inside
their own project directory**. Any repo with `alc.toml` + `[packages]` at
the root is, by convention, a personal/team collection — its mere presence
is a reliable signal. Repositories without `alc.toml` (including the
algocline source repo itself) correctly fall through to global, so running
`/alc-build` from an unrelated directory never scatters pkg files.

Note that the **official `algocline-bundled-packages` set** — the
canonical packages shipped with algocline core — is a separate concept
reserved for `--location=bundled` (not implemented yet; see the
"Reserved name" note above).

The resolved `pkg_root` is embedded literally in the kick prompt as
`Package root: <pkg_root>`. The coder MUST treat that line as the single
source of truth for write paths — do not let the Agent re-resolve via cwd
heuristics (Bash is not in its tool list anyway).

### `[setting.journal]` Config Schema

```toml
# alc.toml or alc.local.toml
[setting.journal]
path = "<shared journal path>"   # optional. Cross-package shared journal.
pkg  = true                       # optional. Also append to <pkg_root>/journal.md (default: false).
```

Resolution is a single `mcp__algocline__alc_setting_resolve(target="journal")`
call (a generic ALC core backbone). Internal precedence:

1. `${ALC_SETTING_JOURNAL_PATH}` env override (shared path only).
2. `[setting.journal] path` / `[setting.journal] pkg` in `alc.toml` /
   `alc.local.toml`.
3. Default — `${XDG_STATE_HOME:-~/.local/state}/algocline/journal.md`
   (shared), `pkg=false`.

The caller (Skill) receives `{path, pkg}` from one MCP tool call; it owns
no env / toml parsing logic. Shared and per-package are additive rather
than mutually exclusive. When both are enabled, the coder appends the same
section to both paths.

## Do

- Include `design_para` verbatim in the kick prompt.
- Resolve `pkg_root` from `--location` (default `auto`) before assembling
  the kick prompt; embed it as `Package root: <pkg_root>`.
- For `--location=auto` and `--location=collection`, run
  `git rev-parse --show-toplevel` and (for collection) `test -f <git_root>/alc.toml`
  via Bash from the Skill body.
- Immediately on startup, call `alc_setting_resolve(target="journal")` once
  to obtain `{path, pkg}` and embed it in the kick prompt.
- Spawn `@alc-coder` via the `Task` tool (exactly one).
- Run the Maturity Self-check before invoking on the User's behalf
  (`Package name / alc.llm intent / Pass conditions / --location` — all
  four must be settled). If anything is missing, ask one question to the
  User and wait, do not dispatch.
- Expect the result to come back as three sections:
  `### Result` / `### Artifacts` / `### Key Observations`.

## DoNot

- **Do not run design dialogue inside this Skill** (mixing it in breaks the
  Skill's scope).
- **Do not spawn any Agent other than `@alc-coder`** (no external pipeline
  agents from other plugins).
- Do not summarize or truncate `design_para` (it causes requirements to
  drop on the Agent side).
- **Do not invoke without the Maturity Self-check passing** (AI-invoked
  path only — half-baked design_para wastes coder tokens and produces an
  unusable skeleton).
- Do not mix upstream setup scope (such as `alc init` deployment).
- **Do not offload journal configuration resolution to the Agent** (it is
  the caller's responsibility — `alc.toml` parsing finishes inside this
  Skill and the Agent simply receives and appends).
- **Do not hard-code `~/.algocline/packages/` in the kick prompt** — the
  write target must be passed through the resolved `Package root:` line so
  collection dogfooding works. Locking the path defeats the whole
  `--location` mechanism.
- **Do not pass `--location=collection` without verifying `<git_root>/alc.toml`
  exists** — silently scattering pkg files into a non-collection repo is
  an accident (refuse with an error instead).

## Anti-patterns

- `prompt-body-bloat` — mixing design-dialogue logic into the Skill body.
- `maturity-check-skip` — invoking on the User's behalf without confirming
  package name / `alc.llm` intent / pass conditions / `--location` are all
  settled (dispatches a half-baked design_para and wastes coder tokens).
- `design-para-truncation` — failing to include the full `design_para` in
  the kick prompt.
- `wrong-agent-spawn` — spawning anything other than `@alc-coder`.
- `scope-creep` — mixing in upstream setup scope (such as `alc init` /
  deployment).
- `journal-resolve-skip` — skipping the one
  `alc_setting_resolve(target="journal")` call or offloading it to the
  Agent (embedding the resolved values in the kick prompt is mandatory).
- `journal-resolve-self-parse` — parsing env / toml in the caller instead of
  using `alc_setting_resolve` (breaks unification on the generic ALC core
  backbone).
- `location-hardcoded` — leaving `~/.algocline/packages/<pkg>/` literally in
  the kick prompt instead of passing the resolved `Package root:` line
  (breaks `--location=collection` and `auto` detection).
- `pkg-root-resolve-skip` — assembling the kick prompt without ever
  resolving `pkg_root` from `--location` (forces the Agent to guess).
- `collection-without-alc-toml` — running `--location=collection` from a git repo
  whose root has no `alc.toml` (must abort, not scatter files).

## Driver Loop

1. Entry: either the User invokes
   `/alc-build [--location=<auto|collection|global>] "<design_para>"`
   literally, or the AI invokes on the User's behalf after the Maturity
   Self-check (above) passes.
2. Extract `--location` (default `auto`) and `design_para`.
3. **Resolve `pkg_root`** per the table in "Package Root Resolution":
   - `global` → `pkg_root = ~/.algocline/packages`.
   - `collection` → `git_root = $(git rev-parse --show-toplevel)`; if
     `<git_root>/alc.toml` is missing, abort with an error message and stop.
     `pkg_root = <git_root>`.
   - `auto` → try `git rev-parse --show-toplevel`; if it succeeds AND
     `<git_root>/alc.toml` exists, `pkg_root = <git_root>`. Otherwise
     `pkg_root = ~/.algocline/packages`.
4. **Call `alc_setting_resolve(target="journal")` once** to fix
   `{path, pkg}` (env / toml / default resolution is completed inside ALC
   core).
5. Assemble the kick prompt (template below; embed both `Package root:` and
   the journal configuration).
6. `Task(subagent_type="alc-coder", description="impl pkg from design_para", prompt=<kick prompt>)`
7. Receive the `### Result / ### Artifacts / ### Key Observations`
   `result_summary` from the Agent.
8. Present it to the main thread as-is and wait for the User's rendezvous
   decision (confirm / re-kick / inject Delta).

### Kick Prompt Template

```
Design para:
<full design_para>

Package root: <resolved pkg_root, absolute path>
# Write target is <Package root>/<pkg>/init.lua and <Package root>/<pkg>/spec/<pkg>_spec.lua.
# Do NOT re-resolve via cwd or any other heuristic; this line is the single source of truth.

Journal config:
- shared path: <resolved journal_path, absolute path>
- per-pkg: <true|false>   # when true, also additively append to <Package root>/<pkg>/journal.md

Impl loop:
1. Extract package name / implementation requirements / pass conditions from design_para.
2. Write <Package root>/<pkg>/init.lua and <Package root>/<pkg>/spec/<pkg>_spec.lua.
3. Run mcp__algocline__alc_pkg_test.
4. On failure, Edit and re-test up to three retries.
5. On pass, return the result_summary under the contract below.
6. Append `## [YYYY-MM-DD] coder — <test_summary>` per the journal config
   (shared path is required; when per-pkg=true, also append the same section
   to <Package root>/<pkg>/journal.md).

Return contract (three required sections):
### Result
- pass / fail + retry count
### Artifacts
- init.lua path / spec path / related files
### Key Observations
- Design decisions / remaining work / next Delta candidates
```
