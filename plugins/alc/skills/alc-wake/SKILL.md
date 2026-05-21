---
name: alc-wake
description: Load-only Skill that injects algocline fundamentals and the installed package list into the main thread. Run once at the start of the session. Does not spawn any Agent.
---

# /alc-wake — algocline DX Wake

A context primer to run before starting development on algocline (Lua DSL +
Hub + Card + MCP runtime). **Does not spawn any Agent.** It only injects the
fundamentals, the existing package map, and the worker query paths into the
main thread.

## What It Does

1. **Load algocline fundamentals into context** (expands the Fundamentals
   section of this Skill body into the main thread).
2. **Call `mcp__algocline__alc_pkg_list` once** -> present the installed
   package list in a formatted table.
3. **Inject a journal tail excerpt** (read the last N sections of the
   configured journal target into the main thread as read-only context).
4. **Guide worker query paths** (`@alc-adviser` query examples,
   `@alc-refiner` kick examples, `/alc-build` kick examples, plus the
   standard Post-impl install / Post-install dist procedures).

The User starts design dialogue on the main thread with that context in
mind. For heavy lookups or investigations, throw a single-turn query at
`@alc-adviser`; once the design is finalized, kick implementation with
`/alc-build "<design>"`.

## Fundamentals (prose injected verbatim into the main thread)

### algocline core primitives

- **`alc.run(strategy)`**: execute a Lua strategy. Pure Lua module with
  access to the `alc.*` StdLib (json, log, state, llm).
- **`alc.llm(prompt, opts)`**: call the LLM from Lua. Uses the
  **pause/resume pattern** — execution stops until the host returns a
  response, then `alc_continue(session_id, response)` resumes it.
- **`alc.state`**: state persisted inside a strategy. Externally CRUDable
  via the `alc_state_list / show / reset` MCP tools.
- **`alc.stats`**: token and LLM-call metrics, e.g. `alc.stats.llm_calls()`.
- **Card**: evaluation record emitted by each strategy execution. Listed,
  fetched, and analyzed via `alc_card_*` tools.
- **Hub**: package index sourced from bundled-packages plus external
  sources. Explore via `alc_hub_search` / `alc_hub_info`.
- **scenario**: input set (cases + graders) used by `alc_eval` to evaluate
  a strategy.

### Package model

- 1 package = `~/.algocline/packages/<name>/init.lua` + `spec/*.lua`
  (tests).
- `M.meta = { name=..., description=..., docs=..., narrative=... }` is the
  single source of truth for package metadata.
- `alc_pkg_test` runs spec-based tests; `alc_pkg_doctor` diagnoses
  structural health.

### Swarm framework (when building orch-style packages)

Multi-step, multi-role orchestrator packages should use **swarm_frame**.
Simple strategies do not need it; bring it in when you need "a state machine
plus a role call per step".

- **Three required deps**: `require("flow")` + `require("swarm_frame")` +
  `require("swarm_frame_algocline")`.
- **Four exports (orch convention)**: `M.parse_verdict(text)` /
  `M.build_prompt(step, state)` / `M.phases` / `M.run(ctx)`, plus
  `M.meta`.
- **init**: the default is `frame.init({ check_mode = "format" })` (orch
  code does not call `alc.llm` directly; the `alc_continue` round trip is
  handled automatically via `flow.llm_bound`).
- **role = path**: no separate role concept; the `/pkg/step_id/agent` path
  acts as the role (e.g. `/lcpo/scan/scanner`).
- **Step registration**: `frame.register(path, spec_builder_fn)` —
  `spec_builder(state)` returns
  `{ agent, task_dir, prompt, inputs, outputs, output_format, gate, report_back }`.
- **Two state layers**: `frame.State` (in-memory; `state:set/get`) +
  `flow.State` (flat `st.data` table; `flow.state_save(st)` persists to the
  `alc.state` KV for resume support).
- **Execution**: `frame.run_linear(PATHS, ctx)` runs steps sequentially
  with automatic `step_done` checks and `state:commit`.
- **Pure-Lua steps** (LLM-less steps) can use the inline handler pattern
  (`frame.register(path, {}, handler_fn)`; `entry.handler` skips the
  dispatcher).
- **Plugin hooks**: pass `plugins={...}` array to the dispatcher; four
  hooks (`before_dispatch` / `around_dispatch` / `after_dispatch` /
  `finalize`). Examples: `plugin_variant_ab` (Blue/Green slot rewrite),
  `plugin_run_card` (Card writes), `plugin_3step_gate`.
- **`plugin_run_card` is the default fixture for orch packages** (measured:
  all nine major orchestrators use it — coding_orch / olympian_weekly /
  modeling_orch / distillation_orch / saas_{scan,profile,digest,technique}
  / flow_implement / flow_refine_orch). Standard call site:

  ```lua
  local plugin_run_card = require("plugin_run_card")
  ctx.dispatcher = adapter.make_dispatcher({
      builder = build_instruction,
      state   = st,
      llm_opts = { ... },
      plugins = {
          plugin_run_card.create({ pkg_name = "<my_pkg>" }),
      },
  })
  ```

  Frame writes one `alc.card` per group at finalize. With Cards in place,
  `alc_card_list` / `alc_card_get` / `alc_card_analyze` can read execution
  traces. **Always include `plugin_run_card` when building an orch package
  unless there is a specific reason not to** (omitting it leaves zero Card
  emissions and no execution trace).

**Reference implementation**: `bundled_base_curator_orch` v0.3.0 (a complete
six-step / six-role implementation; the source pattern for
`SPEC_BUILDERS` / `PATHS` / `STEP_TO_PATH` / `register_steps` / `M.run`).
For new orch packages, read it verbatim first.

### ALC Conventions (rules established in production, applied on every coder dispatch)

Confirmed conventions for building and refining ALC packages. The cycle is:
surface candidates in production -> the Set owner judges "this is a
convention" -> promote it here as static prose. The coder Driver Loop's
static check and Key Observations stages, plus the main AI's post-impl
install / e2e smoke, all assume these. New candidates are surfaced in the
coder's Key Observations as one or two lines; the Set owner judges them
into this section as static promotions. (The journal backbone is for
chronological traces; this section is for confirmed static rules — a
separate axis.)

1. **Treat `ctx.task` as the default acceptance field** —
   `ef.providers.algocline` hard-codes `case.input` into `ctx.task`
   (`<evalframe_repo_root>/evalframe/providers/algocline.lua:139`). When
   package semantics prefer field names such as `ctx.text` or `ctx.idea`,
   add `ctx.task or ctx.text or ...` fallback at the top of `M.run` and
   normalize to the target field, so the same package can be driven from
   both `alc_eval` scenarios and coder dispatches.
2. **The verdict return format is `"DONE path=<verdict>"`** — the only
   forms recognized by `swarm_frame.parse_verdict` are `DONE path=X` /
   `DONE: X` / `DONE = X` (`init.lua:383-386`). Package-specific phrasing
   such as `"DONE verdict=X"` drops to `status: UNKNOWN` (caught and
   fixed in `biz_re_invest_validator` on 2026-05-19).
3. **`alc.json_decode` is the actual API name** — `alc.json.decode` (with
   a dot) is a typo trap that does not exist. The ALC StdLib uses the flat
   naming `alc.json_decode` / `alc.json_encode` / `alc.json_extract`.
4. **In `string.format`, escape literal `%` as `%%`** — prompts such as
   `8% -> 0.08` produce an unknown-conversion-spec runtime error in
   `string.format` because `% ->` is treated as a conversion. Escape it to
   `8%% -> 0.08` to render literally (caught and fixed in
   `biz_re_invest_validator` on 2026-05-19).
5. **Minimum requirements for complete `M.meta`**: the four scaffold
   default fields (name / version / description / category) plus, for
   orch / composite packages, the manually filled `docs` (usage + main
   helpers) and `narrative` (5-10 lines of orchestrator story prose).
   `alc_pkg_scaffold` only generates the first four; the coder is
   responsible for filling in the remaining two by hand.
6. **Post-impl install is the main AI's explicit responsibility** — the
   coder is responsible up to implementation and spec pass; registry
   registration (`alc_pkg_install`) is the next step for the main AI.
   The standard procedure is in this Skill's `Post-impl install` (collection
   layout + `git init` + `alc_pkg_install --force`). Missed installs are
   detected early by `alc_pkg_doctor`'s `unregistered_pkg` bucket (landed
   on the ALC core side); the bucket's suggestion array offers three
   recovery options (`alc_pkg_install --force` / `alc_pkg_link` / `rm`).
7. **Post-impl e2e smoke uses `alc_eval` + a single-case scenario** —
   `alc_pkg_test` only exercises the stub LLM path, letting LLM-path bugs
   (format escapes, `alc.*` typos, verdict parsing, etc.) slip through.
   After install, run `alc_eval scenario_name=<pkg>_post_impl
   strategy=<pkg>` to traverse the real LLM path once (~5-10K tokens) and
   catch them. See `Post-impl e2e-smoke` in this Skill.
8. **`alc_eval` recorded mode uses a scenario-file `provider =` plus a
   borrowed dummy strategy** — for zero-cost regression replay. In the
   scenario file, bind
   `provider = ef.providers.recorded { log_path = "~/.algocline/logs/s-<id>.json" }`,
   and pass a borrowed dummy package name as the `strategy` argument when
   calling `alc_eval` (no real LLM call; the log is replayed). The
   scenario-file provider overrides the `ef.providers.algocline { strategy }`
   auto-wire inside `alc_eval` (landed via algocline `9bccc23`; from
   v0.37.0+ the `prelude.lua` patch reliably gives the scenario provider
   priority). Use cases: zero-cost regression smoke, replay verification
   of old traces after a fix, lightweight e2e for CI. Measured:
   `response.model="recorded"` / `llm_calls=0` / `elapsed_ms=9` /
   `pass_rate=1.0` (sets/alc E2E Phase 4.2 follow-up on 2026-05-20).

### Journal backbone

Activity from the adviser / coder / refiner is recorded in an append-only
journal at section granularity. Configuration lives in the
`[setting.journal]` table of `alc.toml` / `alc.local.toml`:

```toml
[setting.journal]
path = "<shared journal path>"  # optional. Shared journal (cross-package).
pkg  = true                      # optional. ALSO write to <pkg_root>/journal.md (per-package journal; default: false).
```

Resolution is **completed by a single
`mcp__algocline__alc_setting_resolve(target="journal")` call** (the
generic ALC core backbone via `[setting.<x>]` + `alc_setting_resolve`).
The internal precedence is `${ALC_SETTING_JOURNAL_PATH}` env ->
`[setting.journal] path` -> default
(`${XDG_STATE_HOME:-~/.local/state}/algocline/journal.md`); the caller
receives `{path, pkg}` from one MCP tool call. When `pkg = true`, the same
section is also additively appended to the per-package journal (not
mutually exclusive with shared).

Write owners are `@alc-adviser` / `@alc-coder` / `@alc-refiner` only.
Wake is read-only (it only injects tail excerpts and never appends). Each
section follows the `## [YYYY-MM-DD] <role> — <summary>` format.

### Responsibility split between Skill / Agent / `alc` CLI / MCP

| Layer | Role |
|---|---|
| **Skill (`/alc-wake` / `/alc-build`)** | User-triggered context loads and Agent kicks. Runs directly on the main thread. |
| **Agent worker (`@alc-adviser` / `@alc-coder` / `@alc-refiner`)** | Handles heavy lookups, implementation, and refinement observation in an isolated context. Returns results in one turn. |
| **`alc` CLI** | Operational commands such as publish / install. Not invoked by AI. |
| **algocline MCP** | Runtime operations (`alc_run` / `alc_continue` / pkg / card / hub / scenario tools). |

## Driver Loop (steps the main thread runs)

1. Start this Skill -> the Fundamentals prose loads into the main thread
   (this body's Fundamentals section is read as-is).
2. Call `mcp__algocline__alc_pkg_list` once.
3. Format the output as a table of package name / version / description
   (<= 30 rows; collapse the rest into a category summary).
4. **Inject the journal tail excerpt**: read the configured journal
   target (precedence per Journal backbone) and extract the last **N=5**
   H2 sections (each `## [...]` line to the next H2 or EOF) into the main
   thread context. Do not add narrative (excerpt text only). When the
   file is missing, log "journal not found — first run" on one line and
   continue (do not abort). The `tail_n` argument can be specified;
   <= 0 falls back to the default of 5.
5. **Always emit a worker query path guidance block at the end** (state
   each Agent's argument literals explicitly):

   ```
   Worker query (heavy lookup / investigation):
     @alc-adviser { query: "<one sentence with concrete source path / pkg name>" }
     -> Returns one `### Query Result` section (<= 80 lines) to the main
        thread in a single turn, and appends
        `## [date] adviser — <summary>` to the configured journal.

   Pre-impl plan check (optional, design review before dispatching important
   or complex packages):
     @alc-adviser { query: "Review the package design for design_para `<draft>`:
                            flag gaps, BP violations, divergences from
                            major orchestrators, and required fixtures
                            (such as plugin_run_card) in a single turn.",
                    target_pkg?: "<new pkg name>" }
     -> Returns one `### Query Result` (bullet list of plan-gap pointers)
        in a single turn and appends
        `## [date] adviser — pre-impl plan check: <pkg>` to the journal.
     -> The main thread decides whether to adopt the pointers -> revises
        design_para (or adopts as-is) -> /alc-build.
     Note: light packages (<= 200 lines of Lua, single-shot strategies)
     can be skipped — over-engineering for trivial cases.
     Note: the adviser does not decide (client-decide pattern); plan
     adoption belongs to the main thread.

   Refine observation (best-practice improvement proposals against the
   target):
     @alc-refiner { target: "<pkg name or path or identifier, required>",
                    journal_excerpt?: "<tail-N-section markdown, optional>",
                    refine_trigger?: "<improvement focus, optional>" }
     -> The target can be anything (algocline pkg / app / Set / individual
        file). Self-reference is fine.
     -> Returns one `### Refiner Proposal (target: <target>)` section to
        the main thread in a single turn and appends
        `## [date] refiner — <target>: <summary>` to the journal.

   Impl kick (User-only manual trigger, `disable-model-invocation: true`):
     /alc-build "<design_para, <= 200 chars, condensing pkg name +
                  alc.llm usage intent + pass conditions>"
     -> User literal trigger -> spawns @alc-coder -> returns a three-section
        result_summary and appends `## [date] coder — <test_summary>` to
        the journal.

   Post-impl install (explicit, by the main AI, after coder completion):
     ALC core separation: scaffold = skeleton generation / install =
     registry registration (a separate step).
     The coder is responsible up to implementation + spec pass; registry
     registration is an explicit step the main AI takes.

     Standard procedure (collection layout + git init + alc_pkg_install):
       1. mkdir -p /tmp/alc-install-<rand>/
       2. cp -r ~/.algocline/packages/<NEW> /tmp/alc-install-<rand>/
       3. cd /tmp/alc-install-<rand>/ && git init -q && git add . && \
          git -c user.email=local -c user.name=local commit -q -m "install <NEW>"
       4. mcp__algocline__alc_pkg_install url=file:///tmp/alc-install-<rand> force=true
       5. mcp__algocline__alc_pkg_doctor name=<NEW>   -> confirm healthy
       6. (optional) mcp__algocline__alc_hub_info pkg=<NEW>   -> confirm
          installed:true / source disclosed

     Early detection of missed installs: when `alc_pkg_doctor` (full scan
     or with `name`) returns an `unregistered_pkg` bucket, that is the
     signal of a missed install. The same bucket's `suggestion` array
     (string[]) lists three recovery paths in a Clippy-like style:
       - `alc_pkg_install --force <path>` — standard install (re-copy +
         registry registration).
       - `alc_pkg_link <path>` — symlink-based, in-tree dev.
       - `rm -rf <path>` — stale / abandoned cleanup.
     The main AI reads the doctor output and chooses one path to execute.

   Post-impl e2e-smoke (explicit, by the main AI, to detect LLM-path bugs
   after install):
     The spec (alc_pkg_test) only exercises the stub LLM path, allowing
     LLM-path bugs (format escapes / alc.* API typos / verdict parsing) to
     slip through. Run alc_eval with a single-case scenario to traverse
     the real LLM path once and catch them.

     Standard procedure:
       1. Write ~/.algocline/scenarios/<NEW>_post_impl.lua
          (1 case + 1 grader = ef.graders.contains; `expected` is the
           `raw_verdict` form or a package-specific judgement keyword).
          Example (biz_re_invest_validator):
            local ef = require("evalframe")
            return {
                ef.bind { ef.graders.contains },
                cases = {
                    ef.case {
                        name  = "smoke_<NEW>_baseline",
                        input = "<minimal real-property text or input example>",
                        expected = "verdict",
                        context = { task_id = "<NEW>-smoke-<date>" },
                    },
                },
            }
       2. mcp__algocline__alc_eval scenario_name=<NEW>_post_impl strategy=<NEW>
          -> Traverses PARSE -> VALIDATE -> VERDICT once over the real LLM
             path.
          -> The existing scenario `biz_kernel_regression` is a reference
             example of `ef.providers.algocline { strategy }` auto-wire.
       3. If failures appear, Edit to fix the cause (format escapes /
          alc.* typos / verdict-parse mismatches) -> fix the strategy ->
          re-run alc_eval.

     Cost guideline: ~5-10K tokens per package per case (measured:
     biz_re_invest_validator at ~6K tokens immediately surfaced two bugs
     and a verdict-bug candidate).
     Input field mapping: how the provider's `case.input` arrives in `ctx`
     depends on the strategy-side ctx field name (biz_kernel uses
     ctx.task; biz_re_invest_validator uses ctx.text). Verify on the first
     `alc_eval` run; absorb mismatches via `strategy_opts` to override
     `input_field` or via `case.context` merging if needed.

   Post-install dist (standard procedure for distributing to other
   environments / other users; explicit user trigger required):
     ALC core separation:
       - alc_hub_gendoc   = generate docs/narrative/{pkg}.md (Hub
                            distribution documentation).
       - alc_hub_dist     = register Hub manifest (bundled-packages path).
       - alc_hub_reindex  = refresh the Hub cache.

     Standard procedure (assuming GitHub distribution from the collection
     layout):
       1. mcp__algocline__alc_hub_gendoc pkg=<NEW>
          -> Generates docs/narrative/<NEW>.md (consistent with alc.toml
             [hub]).
       2. (optional) Place the pkg dir as a standalone git repo or under a
          collection repo.
       3. The User performs the remote push (outside the AI's scope;
          ytk manual).
       4. Other users run mcp__algocline__alc_pkg_install
          url="github.com/<user>/<repo>" for collection install.
       5. (optional) mcp__algocline__alc_hub_reindex to refresh the Hub
          cache.

     Remote pushes are in the AI's forbidden category (same as
     `cargo publish` / `git push` / `gh release`); ytk does them manually.
     The AI is limited to gendoc, structural setup, and presenting the
     standard commands.
   ```

## Arguments

`/alc-wake` — no arguments. Once invoked, execute the four steps above and
finish.

## Do

- Inject into the main thread context only.
- Format the `alc_pkg_list` results concisely as a table.
- Always place the worker query guidance at the end.

## DoNot

- **Do not spawn `@alc-adviser` / `@alc-coder` / `@alc-refiner` or any
  other Agent** (load-only contract).
- **Do not append to the journal target** (wake is read-only; journal
  appends are the role of adviser / coder / refiner).
- **Do not inject the entire journal into the main thread** (tail-N
  excerpt only).
- Do not set `disable-model-invocation: true` (auto-load is the intended
  design).
- Do not rewrite or regenerate the Fundamentals prose at runtime (the
  Skill body is expected to be injected as-is).
- Do not paste the `alc_pkg_list` output as a one-line JSON (the User
  cannot read it).
- Do not start design dialogue inside this Skill (that is the
  main-thread dialogue's responsibility after the Skill runs).

## Anti-patterns

- `agent-spawn-in-wake` — spawning an Agent from wake.
- `journal-write-in-load` — wake appending to the journal target.
- `full-journal-inject` — embedding the entire journal into the main
  thread context (injection beyond the tail excerpt).
- `dynamic-prose-overwrite` — rewriting the static prose at runtime.
- `prompt-body-bloat` — embedding conditional branches / Agent calls /
  other logic in the Skill body.
- `missing-pkg-table` — omitting the package list because
  `alc_pkg_list` was not called.
- `disable-model-invocation-added` — attaching an auto-load blocker flag
  to a load-only Skill.
- `narrative-inject` — adding explanatory text on top of
  `journal_excerpt` (excerpt text only).
