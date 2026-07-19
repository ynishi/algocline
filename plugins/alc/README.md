# plugins/alc — algocline Package Development DX Set

A Set that supports package development and refinement on algocline (Lua DSL + Hub +
Card + MCP runtime). The design keeps the main thread (User <-> AI) focused on
dialogue while delegating heavy lookups, implementation, and improvement observation
to Agent workers via query / kick. The ALC conventions established in production
are injected into the main thread as load-only Skill prose at all times.

## Components

| Layer | Name | Role |
|---|---|---|
| Skill (load only) | `/alc-wake` | Injects algocline fundamentals, the installed package list, a journal tail, worker query paths, and ALC conventions into the main thread |
| Skill (manual trigger) | `/alc-build "<design_para>"` | Spawns one `@alc-coder` on an explicit user trigger (`disable-model-invocation: true`) |
| Agent (adviser) | `@alc-adviser` | Single-turn lookup, pre-impl plan check, and advice. Planning and decisions stay on the main thread (client-decide pattern) |
| Agent (coder) | `@alc-coder` | Writes `init.lua` and specs into `~/.algocline/packages/<name>/` and iterates up to three retries until `alc_pkg_test` passes. Appends a journal entry on completion |
| Agent (refiner) | `@alc-refiner` | Reads a journal excerpt and returns a single-turn improvement proposal for the target, then appends an entry to the journal |
| Agent (eval) | `@alc-eval` | Paused-session runner for measurement / AnyModel routing. Answers each pending `alc.llm` query as the LLM itself (the caller-picked `model` at dispatch is the model under test), loops via `alc_continue` until completion, and appends one journal entry (skippable with `journal: off`) |
| Agent (runner) | `@alc-runner` | Thin `alc_run` / `alc_continue` loop driver for arbitrary Lua strategies (pre_mortem, ucb, panel, cove, reflect, etc.). Answers each `alc.llm()` prompt from its own knowledge and returns the final result to the caller |

## Usage

1. Run `/alc-wake` at the start of the session to inject the fundamentals, the
   package map, and the ALC conventions into context.
2. Have the design dialogue on the main thread, then condense it into a single
   `design_para` paragraph (<= 200 characters).
3. (For important packages) run a pre-impl plan check via `@alc-adviser`.
4. `/alc-build "<design_para>"` -> dispatches `@alc-coder` -> returns a
   `result_summary`.
5. The main AI registers the package via `alc_pkg_install` (post-impl install).
6. (When LLM-path verification is needed) run a post-impl e2e smoke via
   `alc_eval` with a single-case scenario.
7. (When distributing to other environments) run `alc_hub_gendoc` and ship the
   collection (post-install distribution).

The standard procedure for each step is collected in the footer of `/alc-wake`.

## ALC Conventions

Eight conventions established in production are documented: `ctx.task` fallback
default, the `DONE path=X` verdict format, the `alc.json_decode` flat naming,
`%%` escaping, complete `M.meta`, explicit post-impl install by the main AI,
the `alc_eval` post-impl e2e smoke path, and the recorded mode of `alc_eval`
(scenario-file `provider =` plus dummy strategy borrow). They are injected
verbatim under `/alc-wake` fundamentals -> ALC Conventions.

## Journal Backbone

Configure the journal path and per-package toggle through the `[setting.journal]`
table in `alc.toml` / `alc.local.toml`:

```toml
[setting.journal]
path = "<shared journal path>"   # optional
pkg  = true                       # optional. Also append to <pkg_root>/journal.md
```

Resolution is a single `mcp__algocline__alc_setting_resolve(target="journal")`
call (the generic ALC core backbone `[setting.<x>]`). The internal precedence is
`${ALC_SETTING_JOURNAL_PATH}` env -> `[setting.journal] path` -> default
(`${XDG_STATE_HOME:-~/.local/state}/algocline/journal.md`). The write owners are
only the adviser, coder, refiner, and eval agents; `/alc-wake` and the main
thread are read-only. See `/alc-wake` -> Journal Backbone for details.

## Install

Distributed via the Claude Code Plugin Marketplace.

```
# Register the marketplace (one-time)
/plugin marketplace add <marketplace_source>

# Install the plugin
/plugin install alc@<marketplace_name>
```

The bundled `mcp.json` registers the `algocline` and `git-reader` MCP servers
automatically when the plugin is enabled.

The marketplace source and name depend on how this plugin is distributed; see the
algocline repository for the canonical marketplace location once published.

## Requirements

- **algocline core MCP server** (`alc_run` / `alc_continue` / `alc_pkg_*` /
  `alc_card_*` / `alc_hub_*` / `alc_scenario_*` / `alc_eval` tools)
- **swarm_frame / swarm_frame_algocline / flow** (required when building
  orchestration-style packages; deployed as linked packages via `alc_pkg_link`)
- **evalframe** (for post-impl e2e smokes; installed via `alc_pkg_install`)
- (optional) git-reader MCP (for the adviser to investigate the upstream repo)

## Status

The current backbone (journal write-owner serialization, the independent Refiner
Agent), the seven ALC conventions, and the post-impl install / dist / smoke
paths are all established. The following upstream proposals to algocline core and
evalframe were tracked here (implementation status is listed):

- ~~`[journal]` table schema (`alc.toml` parse + resolve API)~~ — **landed**
  (`[setting.journal]` + `alc_setting_resolve`, ALC core `[Unreleased]`)
- ~~Physical directory walk fallback for `alc_pkg_doctor` (`unregistered_pkg`
  bucket)~~ — **landed** (`unregistered_pkg` bucket + multi-line suggestion
  array, ALC core)
- ~~Automatic resolution of `search_paths` for `alc_pkg_test` (auto-prepend of
  the linked path)~~ — **landed** (auto-prepend of the linked package parent
  directory, opt-out supported, ALC core)
- ~~evalframe `ef.providers.stub` / `recorded` (zero-cost path traversal and
  smoke cost reduction)~~ — **landed (recorded only)** (`ef.providers.recorded`
  for replaying algocline session logs, evalframe `[Unreleased]`). `stub` is
  not added because the existing `mock.recording` / `mock.map` already cover
  the same surface. The MCP path (via the `alc_eval` MCP tool) was confirmed
  on 2026-05-21: a two-line patch to `prelude.lua` (algocline `9bccc23`)
  ensures the scenario-file provider reliably overrides the
  `ef.providers.algocline` auto-wire (measured: `response.model="recorded"`,
  `llm_calls=0`, `elapsed_ms=9`, `pass_rate=1.0`).

## License

MIT OR Apache-2.0
