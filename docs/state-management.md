# State Management

This document specifies how algocline exposes persistent state as a first-class
primitive, and how callers (orchestrators, swarm packages, AI sessions) interact
with that primitive.

## Scope

Two layers, with a strict boundary:

| Layer | Scope | Owns |
|---|---|---|
| **Alc MCP** (`algocline-mcp` + supporting crates) | namespace-generic CRUD over the on-disk state store | `alc_state_list` / `alc_state_show` / `alc_state_reset` (+ optional `set` / `delete`) — wire shape carries `namespace` as a string argument; no application terminology in any type signature |
| **Swarm package** (downstream packages under `algocline-swarm-frame` / bundled collections / consumer repos) | domain-aware update methods that compose the primitive into business actions | `M.run(opts)` entry points (e.g. `swarm_state_method.run(opts={action="…"})`) — domain verbs live here |

The boundary is enforced by a single rule: **shared crates
(`algocline-core` / `-engine` / `-app` / `-mcp`) must not contain any
application terminology in type names, module paths, or wire literals.** Caller
namespaces (`orch`, `incubator`, `dispatch`, `coding`, …) appear only as
opaque `String` arguments at the wire boundary, and as Lua-side identifiers
inside swarm packages.

## Background — why this boundary exists

algocline 0.37.0 deliberately rolled back an earlier proposal to add
`alc_state_list` / `alc_state_show` / `alc_state_reset` as MCP tools, because
that earlier shape introduced an `OrchState` type and leaked the `orch`
application term into shared crates. The 0.37.0 entry of `CHANGELOG.md`
documents the constraint:

> The new design keeps all application terms in caller packages.

The Lua method surface (`alc.state.list/show/reset`) was provided as the
post-rollback path, callable via a `mcp__algocline__alc_run` + Lua snippet.

That surface remains correct as a low-level escape hatch, but it forces every
caller to write a Lua snippet on the spot. For routine operations (especially
recovery from a stuck `/orch` step), structured MCP tools and structured swarm
methods are needed.

This document specifies how to re-introduce structured access **without**
re-triggering the 0.37.0 rollback constraints.

## Architecture

```
Caller (AI session / Skill / SOP / external client)
  │
  ├─→ Alc MCP layer (namespace-generic primitive, this repo)
  │     alc_state_list   { namespace }
  │     alc_state_show   { namespace, key }
  │     alc_state_reset  { namespace, key, steps?, fields? }
  │     alc_state_set?   { namespace, key, value }      ← Phase B (optional)
  │     alc_state_delete?{ namespace, key }             ← Phase B (optional)
  │       │
  │       ▼  inherent method (no EngineApi trait method)
  │     algocline-engine: JsonFileStore
  │       list_dispatched / show_dispatched /
  │       reset_dispatched_with_backup
  │       (+ set_dispatched / delete_dispatched on Phase B)
  │       Lua bridge: alc.state.* (existing surface, preserved)
  │
  └─→ alc_advice MCP layer (already shipped)
        strategy=<pkg_name> opts={action=..., ...}
          │
          ▼  M.run(opts) with opts.alc injection seam
        Swarm package layer (separate repos / collections)
          packages/<name>/init.lua
            M.meta = { category = "state_method", ... }
            domain action verbs (update_dispatch_record, rollback_step, …)
            calls opts.alc.state.* or the Alc MCP layer above
```

The on-disk layout is unchanged: a single store rooted at
`~/.algocline/state/` (resolved via `app_dir().state_dir()`), with files
dispatched under `<root>/<namespace>/<key>.json` plus sidecar `.bak` /
`.tmp` files for atomic mutation.

## Alc MCP layer — tool specification

All tools take a `namespace: String` argument and an optional `key: String`
(where required by the operation). Return values are JSON-serialised strings;
errors propagate as typed MCP errors via `Result<String, String>` on the wire,
following the project-wide error-propagation rule (no `warn!`-and-drop in the
service layer).

### Phase A — read / reset (required)

| Tool | Param | Return | Annotations |
|---|---|---|---|
| `alc_state_list` | `{ namespace: String }` | `{ keys: [string] }` (sorted, excludes `.bak` / `.tmp`) | `read_only_hint=true`, `idempotent_hint=true`, `open_world_hint=false` |
| `alc_state_show` | `{ namespace: String, key: String }` | parsed state JSON (`{ data: {…}, identity?: {…}, _token_value?: … }` — opaque object preserved as-is) | `read_only_hint=true`, `idempotent_hint=true`; typed `NOT_FOUND` for unknown `key` |
| `alc_state_reset` | `{ namespace: String, key: String, steps?: [string], fields?: [string] }` | `{ ok: bool, backup_path: string, steps_removed: usize, steps_input: [string], fields_removed: usize, fields_input: [string] }` | `read_only_hint=false`, `idempotent_hint=true` (atomic rename + set-difference semantics) |

`alc_state_reset` semantics:

1. Write `<file>.bak` (overwrites any previous `.bak`) before mutating.
2. Remove the listed entries from `data.completed_steps` (no error on missing
   entries — set-difference semantics).
3. Delete the listed top-level fields from `data` (true JSON deletion, not
   zero-set — consumers verify `field in data` is `false` post-reset).
4. Atomically rename `<file>.tmp` to `<file>`.
5. No-op when both `steps` and `fields` are empty (returns current state).

Engine-layer implementation already exists as inherent methods on
`JsonFileStore`:

- `list_dispatched` (`crates/algocline-engine/src/state.rs:391`)
- `show_dispatched` (`crates/algocline-engine/src/state.rs:442`)
- `reset_dispatched_with_backup` (`crates/algocline-engine/src/state.rs:513`)

These are already namespace-generic — Phase A is therefore a thin MCP wire +
service-layer wrapper on top of the existing engine surface, plus E2E tests.
No new trait methods, no new types in `algocline-core`.

### Phase B — write / delete (optional, on demand)

| Tool | Param | Return | Annotations |
|---|---|---|---|
| `alc_state_set` | `{ namespace: String, key: String, value: JsonValue }` | `{ ok: bool }` | `read_only_hint=false`, `idempotent_hint=true` (full replacement) |
| `alc_state_delete` | `{ namespace: String, key: String }` | `{ ok: bool, existed: bool }` | `read_only_hint=false`, `idempotent_hint=true` |

These require adding two inherent methods to `JsonFileStore`:

- `set_dispatched(ns: &str, key: &str, value: &Value) -> Result<(), StateError>`
- `delete_dispatched(ns: &str, key: &str) -> Result<bool, StateError>`

Both already exist as internal building blocks inside the private
`set` / `delete` methods (`crates/algocline-engine/src/state.rs:666` /
`684`); Phase B promotes them to inherent methods, exposes matching Lua
bridge entries (`alc.state.set_dispatched` / `alc.state.delete_dispatched`),
and adds MCP wire bindings.

Phase B is deferred until a concrete swarm-package use case requires
cross-namespace write access. Until then, the existing single-namespace
`alc.state.set` / `alc.state.delete` Lua methods remain the write path.

## Constraints inherited from the 0.37.0 rollback

These are hard constraints; violating any of them re-triggers the
rollback rationale.

1. **No application terminology in shared crate types or modules.** The wire
   carries `namespace: String`; no `OrchState`, no `CodingState`, no
   `DispatchKey` types are added to `algocline-core` / `-engine` / `-app` /
   `-mcp`.
2. **`algocline-core` may add `EngineApi` trait methods if their signatures are
   namespace-generic (`String` / `Option<Vec<String>>` / `serde_json::Value`-shaped)
   — application-specific types are forbidden.** The implementation lives as inherent
   methods on `JsonFileStore`; trait method additions follow the same
   namespace-generic rule and are permitted under the same precedent as the
   0.23.0 / 0.24.0 BREAKING additions.
3. **The existing `alc.state.*` Lua method surface is preserved unchanged.**
   The MCP tools added here are an additive wire surface, not a replacement.
4. **No silent state mutation.** Every write goes through the existing atomic
   `<file>.tmp` → `rename` path, with `.bak` snapshot on `reset` (Phase A) and
   on `set` / `delete` when those land (Phase B).
5. **Errors propagate.** Service-layer code uses `?` to propagate
   `StateError` (and its `From` conversions) to the MCP boundary. No
   `tracing::warn!`-and-drop in any new path.

(2026-06-13 update: constraint #2 narrowed from "no new trait methods" to "no application-term leak in trait method signatures". The 0.37.0 rollback rationale targeted type-name leak, not trait surface expansion.)

## Swarm package layer — pattern

Swarm packages express **domain actions**: high-level verbs that compose
primitive CRUD into a single business operation. They live outside this repo
(typically in `algocline-swarm-frame/packages/<name>/` or bundled collections)
and are dispatched via the existing `alc_advice` MCP tool.

### Layout

```
algocline-swarm-frame/packages/<pkg_name>/
  ├── init.lua       # M.meta + M.spec + M.run(opts)
  ├── alc.toml       # name = "<pkg_name>", version = "0.1.0"
  └── spec/          # mlua-lspec specs
```

### Entry-point shape

```lua
local S = require("alc_shapes")
local T = S.T
local M = {}

M.meta = {
    name = "<pkg_name>",
    version = "0.1.0",
    description = "Domain-aware state update verbs for <domain>.",
    category = "state_method",
    alc_shapes_compat = "^0.25",
}

M.spec = {
    entries = {
        run = {
            input = T.shape({
                action = T.string:describe("domain verb"),
                -- per-action params follow
            }),
            result = T.any(),
        },
    },
}

function M.run(opts)
    local alc = opts.alc or _G.alc
    local action = opts.action or error("opts.action required", 2)

    if action == "<verb>" then
        -- Compose primitives:
        --   alc.state.list / show / reset / set / delete
        -- or the Alc MCP layer above.
        return { ok = true, … }
    end

    error("unknown action: " .. tostring(action))
end

return M
```

### Caller invocation (no Lua coding required)

```
mcp__algocline__alc_advice
  strategy=<pkg_name>
  opts={"action":"<verb>", "<param>": "<value>", …}
```

The `opts.alc` injection seam (already convention in
`algocline-swarm-frame/packages/combinator_demo/init.lua:57`) lets the
package be tested with a mocked `alc` table and run in production with the
real one — no DI rewiring needed.

## Phase order and acceptance criteria

### Phase A — Alc MCP read / reset tools

Acceptance:

- [ ] `alc_state_list` lists keys for any `namespace`; sorted; excludes
      `.bak` / `.tmp`.
- [ ] `alc_state_show` returns full parsed JSON; typed `NOT_FOUND` on
      missing `key`.
- [ ] `alc_state_reset` deletes listed `completed_steps` entries and listed
      top-level `data` fields; writes `.bak` before mutating; atomic rename.
- [ ] All three tools appear in `tests/e2e.rs` with at least one happy-path
      and one typed-error case each.
- [ ] Full `cargo test --workspace` passes; no regression in
      `tests/e2e.rs`.
- [ ] No new types or modules in `algocline-core`; no new `EngineApi` trait
      method.
- [ ] `CHANGELOG.md` `[Unreleased]` entry under `### Added`, with a 1–2 line
      `### Notes` paragraph referencing the 0.37.0 rollback and explaining the
      constraints that make this iteration safe.

### Phase B — Alc MCP set / delete tools (optional)

Triggered by a concrete use case (typically the first swarm package that
needs cross-namespace write). Acceptance mirrors Phase A.

### Phase C — Swarm package (`swarm_state_method` or specific verbs)

Acceptance:

- [ ] One swarm package landed in `algocline-swarm-frame/packages/<name>/`
      with `M.meta` / `M.spec` / `M.run` + spec/.
- [ ] `alc_advice strategy=<name> opts={...}` produces the expected mutation
      visible via `alc_state_show`.
- [ ] `algocline-swarm-frame` version bumped; `BUNDLED_SOURCES` tag in
      `src/init.rs` updated on the next algocline release.

### Documentation

- This file is the canonical specification (linked from `AGENT_INDEX.md`).
- External consumer documentation (state.json editing runbooks in
  downstream orchestration projects) will be updated to point at the new
  MCP tools instead of the Lua snippet path, after Phase A lands.

## File reference

| Subject | Path |
|---|---|
| Engine-layer primitive | `crates/algocline-engine/src/state.rs:391` (`list_dispatched`), `:442` (`show_dispatched`), `:513` (`reset_dispatched_with_backup`) |
| Path resolution | `crates/algocline-core/src/app_dir.rs:45` (`state_dir`) |
| Path safety | `crates/algocline-engine/src/state.rs:146` (`is_safe_segment`) |
| Lua bridge | `crates/algocline-engine/src/bridge/data.rs:133-265` (`alc.state.*` register) |
| Existing wire-shape templates | `crates/algocline-mcp/src/service.rs:71` / `:2152` (`alc_setting_resolve`), `:352` / `:1402` (`alc_pkg_doctor`) |
| `alc_advice` dispatch | `crates/algocline-mcp/src/service.rs:1051` (tool), `crates/algocline-app/src/service/run.rs:312` (service) |
| Service-layer state store wiring | `crates/algocline-app/src/service/mod.rs:141` (`state_store: Arc<JsonFileStore>`) |
| Rollback record | `CHANGELOG.md` 0.37.0 entry (`### Notes` paragraph) |
