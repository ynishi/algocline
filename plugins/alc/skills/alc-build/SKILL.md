---
name: alc-build
description: The single manual-trigger Skill that delegates algocline package implementation to @alc-coder. Run manually once the User has finalized design_para.
disable-model-invocation: true
---

# /alc-build — algocline Package Implementation Kick

Hands a `design_para` finalized through design dialogue to `@alc-coder` and
lets it implement in an isolated context. **Started only via an explicit
manual trigger from the User** (`disable-model-invocation: true`).

A Skill to run only the moment dialogue on the main thread (User <-> AI) has
matured and the design has been condensed into a single paragraph. Never
trigger it in an intermediate state (when the design is vague, the package
name undecided, or the pass conditions missing).

## Arguments

`/alc-build "<design_para, <= 200 chars>"`

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
- **Card emission (swarm_frame orch packages only)**: wire
  `plugin_run_card.create({pkg_name="..."})` into
  `make_dispatcher({plugins=...})` (the default fixture for the nine major
  orchestrators; see wake -> Swarm framework). Single-shot strategy
  packages are not in scope, but orch packages should include it **unless
  there is a specific reason not to**.

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
2. **Resolve the journal configuration** — call
   `mcp__algocline__alc_setting_resolve(target="journal")` once and let ALC
   core resolve env / toml / default to return `{path, pkg}` (the caller
   does not own env / toml parsing logic).
3. **Assemble the kick prompt** — `design_para` + journal configuration
   (`path` / `pkg`) + implementation-loop instructions (Write `init.lua` +
   spec -> `alc_pkg_test` -> Edit on failure -> <= 3 retries) + return
   contract (three sections: `### Result` / `### Artifacts` /
   `### Key Observations`).
4. **Spawn `@alc-coder` via the `Task` tool** — no other Agents
   (`impl-lead` / orch agents, etc.) may be called.
5. Return the `result_summary` to the main thread and hand control back to
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
- Immediately on startup, call `alc_setting_resolve(target="journal")` once
  to obtain `{path, pkg}` and embed it in the kick prompt.
- Spawn `@alc-coder` via the `Task` tool (exactly one).
- Keep `disable-model-invocation: true` in the frontmatter.
- Expect the result to come back as three sections:
  `### Result` / `### Artifacts` / `### Key Observations`.

## DoNot

- **Do not run design dialogue inside this Skill** (mixing it in breaks the
  Skill's scope).
- **Do not spawn any Agent other than `@alc-coder`** (no orch agents,
  `impl-lead`, etc.).
- Do not summarize or truncate `design_para` (it causes requirements to
  drop on the Agent side).
- Do not remove `disable-model-invocation: true` (no turning it into an
  auto-trigger).
- Do not mix upstream setup scope (such as `alc init` deployment).
- **Do not offload journal configuration resolution to the Agent** (it is
  the caller's responsibility — `alc.toml` parsing finishes inside this
  Skill and the Agent simply receives and appends).

## Anti-patterns

- `prompt-body-bloat` — mixing design-dialogue logic into the Skill body.
- `disable-model-invocation-missing` — dropping the auto-trigger guard flag
  from the frontmatter.
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

## Driver Loop

1. The User invokes `/alc-build "<design_para>"` literally.
2. Extract `design_para`.
3. **Call `alc_setting_resolve(target="journal")` once** to fix
   `{path, pkg}` (env / toml / default resolution is completed inside ALC
   core).
4. Assemble the kick prompt (template below; embed the journal
   configuration).
5. `Task(subagent_type="alc-coder", description="impl pkg from design_para", prompt=<kick prompt>)`
6. Receive the `### Result / ### Artifacts / ### Key Observations`
   `result_summary` from the Agent.
7. Present it to the main thread as-is and wait for the User's rendezvous
   decision (confirm / re-kick / inject Delta).

### Kick Prompt Template

```
Design para:
<full design_para>

Journal config:
- shared path: <resolved journal_path, absolute path>
- per-pkg: <true|false>   # when true, also additively append to <pkg_root>/journal.md

Impl loop:
1. Extract package name / implementation requirements / pass conditions from design_para.
2. Write ~/.algocline/packages/<pkg>/init.lua and spec/<pkg>_spec.lua.
3. Run mcp__algocline__alc_pkg_test.
4. On failure, Edit and re-test up to three retries.
5. On pass, return the result_summary under the contract below.
6. Append `## [YYYY-MM-DD] coder — <test_summary>` per the journal config
   (shared path is required; when per-pkg=true, also append the same section
   to <pkg_root>/journal.md).

Return contract (three required sections):
### Result
- pass / fail + retry count
### Artifacts
- init.lua path / spec path / related files
### Key Observations
- Design decisions / remaining work / next Delta candidates
```
