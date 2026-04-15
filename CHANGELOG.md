# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `alc init`: bundled `algocline-bundled-packages` tag bumped `v0.11.2` → `v0.12.0` (adds 12 Swarm foundation packages: `shapley`, `mwu`, `kemeny`, `scoring_rule`, F1-F5, N1-N5; plus recategorisation of 10 existing packages). `evalframe` stays at `v0.3.0`. Runtime behaviour unchanged — only the default set fetched by `alc init`.

## [0.20.1] - 2026-04-16

### Added

- `alc_pkg_list`: entries now include `resolved_source_path` (canonical absolute dir), `resolved_source_kind` (installed/linked/local_path/bundled), and `override_paths` (shadowed same-name pkg paths) for LLM agent source access.

### Fixed

- `alc_pkg_list`: project `installed` / `git` / `bundled` entries no longer list their own backing directory (`packages_dir()/{name}`) as a `override_paths` self-shadow. Only genuinely distinct same-name packages (e.g. a project `path` vendor dir overriding a global install) now appear in `override_paths`.

### Changed

- `alc_pkg_list` (internal): meta merge ordering tightened so every host-authoritative field (`error`, `linked`, `link_target`, `broken`, …) is uniformly protected from Lua `meta.*` clobbering. Output JSON shape is unchanged for conforming packages; packages whose `meta` illicitly shadowed these names now correctly return the host value.
- `alc_pkg_list` (internal): `resolved_source_kind` is now a typed enum internally (`Installed`/`Linked`/`LocalPath`/`Bundled`); wire format is identical (snake_case strings).

## [0.20.0] - 2026-04-13

### Added — `alc_card_samples` / `alc.card.read_samples` gains `where`

The per-case sidecar reader now accepts the same nested-object `where`
DSL as `alc_card_find`, evaluated against each JSONL row. `offset` is
applied after filtering (Prisma/SQL convention), so paging the matched
subset is predictable.

```lua
local failures = alc.card.read_samples(card_id, {
  where  = { passed = false, score = { lt = 0.5 } },
  offset = 0,
  limit  = 20,
})
```

Pure addition — calls without `where` keep previous semantics.

### Added — `alc_card_lineage` / `alc.card.lineage`

New lineage walker that traverses Card ancestry/descendants via the
`metadata.prior_card_id` convention. Directions: `"up"` (ancestors,
default), `"down"` (descendants), `"both"`. Optional `depth` cap,
`include_stats`, and `relation_filter` for following only edges with
specific `prior_relation` values. Returns `{ root, nodes, edges,
truncated }` where `nodes[*].depth` is signed (0 root, negative
ancestor, positive descendant).

Also documents `[strategy_params]` and `metadata.prior_card_id` /
`metadata.prior_relation` as recognized Card schema conventions.

### Changed — **BREAKING**: `alc_card_find` / `alc.card.find` DSL

Two breaking changes, plus one additive field. 0.x allows breaks —
migrate callers before upgrading.

**1. Filter fields → `where` object (Prisma-style)**

All ad-hoc filter fields (`scenario`, `model`, `min_pass_rate`) are
removed. Use a nested `where` object that walks Card sections, with
implicit equality on scalars and reserved operators on leaf objects.

```lua
-- Before
alc.card.find({ pkg = "foo", scenario = "bar", min_pass_rate = 0.5 })

-- After
alc.card.find({
    pkg = "foo",
    where = {
        scenario = { name = "bar" },
        stats = { pass_rate = { gte = 0.5 } },
    },
})
```

Operators: `eq ne lt lte gt gte in nin exists contains starts_with`,
plus logical `_and` / `_or` / `_not` at any level.

**2. `sort` → `order_by` (dotted path, descending prefix, multi-key)**

`sort = "pass_rate"` is replaced with `order_by` — a dotted-path string
(`"stats.pass_rate"`), a `-` prefix for descending, or an array for
multi-key sort with tiebreakers.

```lua
-- Before
sort = "pass_rate"

-- After
order_by = "-stats.pass_rate"
-- or
order_by = { "-stats.pass_rate", "created_at" }
```

**Added**: `offset` for pagination (pure addition, non-breaking).

Missing-field semantics: `eq`/`lt`/etc. evaluate false on missing
fields; `ne`/`nin` evaluate true; `exists: false` matches only when
the field is absent.

See `docs/lua-stdlib.md#alccardfind` for the full reference.

## [0.19.0] - 2026-04-13

### Added

- **`alc_hub_info`**: Show detailed information for a single package — metadata, all Cards, aliases, and aggregated stats (card count, eval count, best pass rate). Looks up remote indices first, falls back to local `init.lua` parse.
- **`collection_url` support**: New `[hub].collection_url` in `~/.algocline/config.toml` adds a Tier 0 aggregated index URL, fetched before per-source registries.

### Fixed

- **Path traversal guard** in `hub_info`: reject package names containing `..`, `/`, or `\`.
- **Duplicate `card::list` call** in `hub_info`: reuse a single call for both JSON output and stats.
- **`count_evals_for_pkg` ordering**: two-pass collection eliminates `read_dir` iteration-order dependency.

### Changed

- Enriched module-level RustDoc for `card.rs` (Card schema, design principles, storage layout) and `hub.rs` (staged design, index schema, 4-tier discovery, caching).

## [0.18.0] - 2026-04-12

### Added — Hub: Package Discovery & Search

Registry-based remote index discovery with per-source caching.

- **`alc_hub_search`**: Search packages across remote Hub indices + local install state. Index URLs are auto-discovered from hub registries (populated by `pkg_install` / `card_install`), the installed-packages manifest, and bundled seeds. Results include `installed: true/false`, descriptions, categories, and source URLs.
- **`alc_hub_reindex`**: Generate a hub index from a packages directory. New `source_dir` parameter enables pure metadata extraction from a repo checkout (no manifest or card data mixed in) for CI publishing.
- **Hub registries** (`~/.algocline/hub_registries.json`): Persistent registry of source URLs, auto-populated on `pkg_install` and `card_install`. Atomic writes via tempfile + rename.
- **Per-source cache** (`~/.algocline/hub_cache/{hash}.json`): Each remote index cached independently with 1-hour TTL using FNV-1a URL hashing.

### Changed

- Bump `algocline-bundled-packages` to v0.11.2 (adds `hub_index.json`)

## [0.17.1] - 2026-04-12

### Changed

- Bump `algocline-bundled-packages` from v0.11.0 to v0.11.1
  (Optimizer Card support)

## [0.17.0] - 2026-04-12

### Added — `alc.eval()` Lua function

Evalframe facade exposed as a first-class Lua function in prelude.
Accepts string scenario names or inline tables, wires the algocline
provider automatically, and optionally emits a Card on completion.

### Changed — `alc_eval` MCP tool delegates to `alc.eval()`

The MCP `alc_eval` tool now delegates to the prelude `alc.eval()`
function instead of hand-building evalframe Lua code. Card emission
is handled Lua-side, removing Rust-side `maybe_save_card`.
`eval_compare` shares the `STD_SHIM` constant with `eval`.

### Added — Card schema v0 (frozen)

Immutable run-result snapshots stored as TOML under
`~/.algocline/cards/{pkg}/{card_id}.toml`. The full v0 surface is now
considered frozen — future additions land behind a `v1` schema bump.

**v0 schema**:
- REQUIRED: `schema_version`, `card_id`, `created_at`, `pkg.name`
- Everything else is OPTIONAL and auto-injected when derivable
- `card_id` format: `{pkg}_{model_short}_{YYYYMMDDTHHMMSS}_{hash6}`
- Low-hex `hash6` (DJB2 last 6 chars) to avoid top-bit collisions
- `param_fingerprint` auto-computed from `[params]` when present

**Lua API (`alc.card.*`)**:
- `create(table)` — write a new Card (immutable)
- `get(card_id)` / `get_by_alias(name)` — fetch full Card
- `list(filter?)` / `find(query?)` — summaries with sort / filter
- `append(card_id, fields)` — additive-only annotation
- `alias_set(name, card_id, opts?)` / `alias_list(filter?)` — mutable aliases
- `write_samples(card_id, samples)` / `read_samples(card_id, opts?)` —
  write-once per-case JSONL sidecar

**MCP tools (host-side read surface)**:
- `alc_card_list` / `alc_card_get` / `alc_card_find`
- `alc_card_alias_list` / `alc_card_alias_set` / `alc_card_get_by_alias`
- `alc_card_append`
- `alc_card_samples` (per-case sidecar read with `offset` / `limit` paging)

**`alc_eval` integration**: Opt-in `auto_card=true` emits a Card from
the eval result on completion, and when per-case rows are present
dumps them to a `{card_id}.samples.jsonl` sidecar.

**Examples**: `examples/cards/prompt_ab_demo.lua` — a self-contained
6-trial prompt sweep exercising create / find / alias_set / append
end-to-end with no LLM calls.

## [0.15.1] - 2026-04-09

### Added

- **mlua-mathlib v0.3.0**: Upgraded from v0.2. Adds 22 new `alc.math` functions:
  - Hypothesis testing: `welch_t_test`, `mann_whitney_u`, `chi_squared_test`, `ks_test`
  - Ranking & IR metrics: `rank`, `spearman_correlation`, `kendall_tau`, `ndcg`, `mrr`
  - Information theory: `entropy`, `kl_divergence`, `js_divergence`, `cross_entropy`
  - Special functions: `logsumexp`, `logit`, `expit`
  - Time series: `moving_average`, `ewma`, `autocorrelation`
  - Combinatorics: `permutations`
  - RNG: `shuffle`, `sample_with_replacement`

## [0.15.0] - 2026-04-09

### Added

- **`alc_init` MCP tool**: Initialize project — creates `alc.toml` in the project root if absent. Equivalent to `alc init` for project-scoped setup via MCP
- **`alc_update` MCP tool**: Update installed packages declared in `alc.toml` — re-installs each entry from its recorded source URL and updates `alc.lock`
- **`alc_migrate` MCP tool**: Migrate legacy `alc.lock` (v1 `local_dir` entries) to the new `alc.toml` + `alc.lock` schema. Generates `alc.toml` from existing lock entries and rewrites `alc.lock` to the new format
- **`alc_pkg_unlink` MCP tool**: Remove a symlink created by `alc_pkg_link`. Rejects real directories (only symlinks are removed) to prevent accidental deletion of installed packages
- **`alc.toml`**: New project-level package declaration file. Declares packages with `name`, `source`, and optional `version`. Used as the source of truth for project-local package management
- **`alc.toml`-based project discovery**: Project root is now detected by walking up the directory tree to find `alc.toml` (previously `alc.lock`). `alc.lock` remains the resolved lockfile written by install/link operations
- **Lock mismatch warning**: Detects drift between `alc.toml` declarations and `alc.lock` resolved entries. Warns when packages declared in `alc.toml` are absent from `alc.lock` or vice versa
- **`PackageSource::Installed` / `PackageSource::Path`**: Renamed variants replacing `LocalCopy` and `LocalDir` respectively. `Installed` = package installed to cache from a URL; `Path` = symlinked local directory
- **`alc.toml` auto-append on install**: `alc_pkg_install` automatically appends the installed package to `alc.toml` when a project root is detected
- **Symlink-based `alc_pkg_link`**: Rewrites `pkg_link` to create a symlink inside `~/.algocline/packages/` pointing to the local directory. Removes the containment check entirely. `pkg_list` reports `linked`, `link_target`, and `broken` fields for symlink entries
- **Source provenance in `alc_pkg_list`**: Each entry now shows a `from` field indicating the install source (URL, path, or bundled)

### Changed

- **`alc_pkg_remove`**: Unified to remove from `alc.toml` + `alc.lock` only — cache directory is never deleted. The `scope` parameter is removed; removal always targets the project-local declaration
- **`alc_pkg_list`**: Project scope now reads from `alc.toml` (declarations) merged with `alc.lock` (resolved version/source), instead of reading `local_dir` entries directly from `alc.lock`
- **`PkgRemoveParams`**: `scope` field replaced by `version` (optional, for disambiguation)
- **`PkgLinkParams`**: `project_root` field removed; project root is auto-detected via `alc.toml` walk
- **`EngineApi` trait**: Removed `scope` from `pkg_remove`; added `alc_init`, `alc_update`, `alc_migrate`, `pkg_unlink` methods
- **`lockfile.rs`**: `LockPackage` loses `linked_at` field; gains `version: Option<String>`. `resolve_local_dir_paths` renamed to `resolve_path_entries` with containment check removed
- **`project.rs`**: `walk_up_for_lockfile` renamed to `walk_up_for_alc_toml`
- **`detect_legacy_format`**: Migrated from string-contains to TOML structural parsing to prevent false positives on package names containing `linked_at` or `local_dir`
- **Test helper consolidation**: Extracted duplicated `make_app_service` / `with_fake_home` into shared `test_support` module

### Fixed

- **`pkg_link` / `pkg_unlink` tests**: Replaced `Handle::block_on()` inside `#[tokio::test]` (runtime nesting panic) with `FakeHome` RAII guard pattern that allows direct `.await`. All 10 previously broken tests now pass
- **`eval_auto_installs_evalframe_on_missing` test**: Added `rt.enter()` guard for `AppService::new()` which calls `spawn_gc_task` requiring a runtime context; added `HOME_MUTEX` serialization to prevent env var races with `FakeHome` tests
- **Dead code cleanup**: Removed unused `resolve_installed_paths`, `resolve_abs`, and `#[allow(dead_code)]` annotations

## [0.14.0] - 2026-04-09

### Added

- **`alc_pkg_link`**: Link a local directory as a project-local package without copying. Records the path in `alc.lock`. Supports single package and collection layouts. Idempotent — re-linking updates the existing entry
- **`alc.lock`**: Project-local lockfile schema (version=1) for managing project-scoped package references. Stores `local_dir` entries pointing to on-disk paths
- **Project-local package resolution**: `alc.lock` `local_dir` entries are resolved as high-priority `FsResolver`s, taking precedence over `ALC_PACKAGES_PATH` and global `~/.algocline/packages/`. Enables per-project package overrides without modifying global state
- **`project_root` parameter**: `alc_run`, `alc_advice`, `alc_pkg_list`, `alc_pkg_remove` accept optional `project_root` to activate project-local package resolution. Auto-detected via `ALC_PROJECT_ROOT` env or `alc.lock` ancestor walk when omitted
- **`scope` parameter**: `alc_pkg_list` and `alc_pkg_remove` accept `scope` (`"project"` / `"global"`) for explicit scope targeting
- **`PackageSource` enum**: Type-safe representation of package origins (Git / LocalCopy / LocalDir / Bundled) with legacy string inference for backward compatibility

### Changed

- **`BUNDLED_VERSION`**: Updated bundled-packages from `v0.9.0` to `v0.11.0`
- **`EngineApi` trait**: `run` and `advice` gain `project_root: Option<String>` parameter; `pkg_list` gains `project_root`; `pkg_remove` gains `project_root` and `scope` (breaking for trait implementors)
- **`pkg.rs` → `pkg/` module**: Split monolithic `pkg.rs` into `pkg/install.rs`, `pkg/list.rs`, `pkg/remove.rs`, `pkg/tests.rs` submodules

### Fixed

- **Lua injection prevention**: Package names are whitelist-validated before interpolation into Lua source in `pkg_list` meta evaluation
- **Path containment**: `pkg_link` canonicalizes and containment-checks `LocalDir` paths so `alc.lock` cannot reference paths outside `project_root`
- **Atomic lockfile writes**: `save_lockfile` uses `NamedTempFile` + `persist` to prevent readers from observing half-written `alc.lock`
- **`eval_simple` require cache**: Clears `package.loaded[name]` before meta evaluation to avoid stale cached modules across calls

## [0.13.0] - 2026-04-04

### Added

- **`alc.llm_json(prompt, opts?)`**: LLM call with automatic JSON parsing and 1-retry repair. Uses `alc.json_extract` for 3-stage fallback parsing; on failure, retries with previous output included so the model can fix rather than regenerate
- **`alc.math`**: Numeric computing namespace (44 functions) via mlua-mathlib v0.2.0 — RNG, distribution sampling (Normal, Beta, Gamma, Poisson, Binomial, etc.), descriptive statistics, CDF/PPF, special functions (erf, gamma, beta, digamma, factorial), transforms (softmax, histogram, Wilson CI)
- **`docs/lua-stdlib.md`**: `alc.math` section with full API reference
- **`types/alc.d.lua`**: LuaCats type definitions for all `alc.math.*` functions

### Changed

- **`BUNDLED_VERSION`**: Updated bundled-packages from `v0.7.0` to `v0.9.0`, evalframe from `v0.1.0` to `v0.3.0`
- **Dependencies**: mlua-mathlib `0.1` → `0.2`

## [0.12.1] - 2026-04-02

### Fixed

- **`alc.match_bool`**: Add word boundary check to prevent false positives (e.g. `"ok"` in `"token"`, `"pass"` in `"bypass"`, `"no"` in `"innovation"`)
- **`alc.match_enum`**: Fuzzy fallback now splits text into words and compares per-word instead of whole-text, enabling typo detection in long LLM responses

### Added

- **`docs/lua-stdlib.md`**: Type Support section — LuaCats setup and `lua-language-server --check` CI integration guide

## [0.12.0] - 2026-04-02

### Added

- **`alc.match_enum(text, candidates, opts?)`**: Fuzzy enum matcher for LLM output. Case-insensitive substring match with Jaro-Winkler fuzzy fallback (Layer 0, powered by `fuzzy-parser` crate)
- **`alc.match_bool(text)`**: Yes/no normalizer for LLM responses. Returns `true`, `false`, or `nil` based on last-occurring affirmative/negative keyword (Layer 0)
- **`alc.parse_number(text, pattern?)`**: Extract numbers from LLM output with optional Lua pattern (Layer 1 Prelude)
- **Host token tracking**: `alc_continue` accepts optional `usage` field with `prompt_tokens` / `completion_tokens`. Tracked as `TokenSource::Host` in `ExecutionMetrics`, providing accurate token counts instead of character-based estimates
- **`max_tokens` budget**: Host can set `max_tokens` in `alc_run` context (`ctx._max_tokens`). When budget is exhausted, subsequent `alc.llm()` calls fail with a budget error
- **`alc init` / `alc update`**: Distributes `alc.d.lua` LuaCats type stub to `~/.algocline/types/alc.d.lua` on every run. Enables editor completion (Lua Language Server, `lua_ls`) for all `alc.*` StdLib functions. If `.luarc.json` is absent from the current directory, a setup tip is printed to stderr
- **MCP server startup**: Automatically distributes `alc.d.lua` on each server start, so the type stub is always up-to-date after `cargo install`
- **`alc_pkg_install` response**: Added `types_path` field — absolute path to the installed `alc.d.lua` stub — so MCP clients can surface the location without an extra tool call

### Changed

- **`alc_advice` `task` parameter**: Now optional (`Option<String>`). Packages that don't use `ctx.task` (e.g. `factscore`, `optimize`, `lineage`) can be called with `opts` alone
- **`EngineApi::advice` trait**: `task` parameter changed from `String` to `Option<String>` (breaking for trait implementors)
- **`EngineApi::continue_single` trait**: Added `usage: Option<TokenUsage>` parameter (breaking for trait implementors)

## [0.11.1] - 2026-04-01

### Changed

- **`alc_log_view`**: Added `max_chars` parameter for detail mode (default: 100KB). Truncates transcript from oldest rounds when response exceeds limit. Set `max_chars=0` for unlimited

## [0.11.0] - 2026-03-30

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.6.0` to `v0.7.0`

### Fixed

- **Clippy warnings**: Removed redundant closure in `spec.rs`, replaced `assert_eq!(…, true)` with `assert!()` in unit tests

## [0.10.0] - 2026-03-25

### Added

- **`alc.fork(strategies, ctx, opts?)`**: Parallel multi-VM strategy execution (Layer 0). Spawns N independent Lua VMs, each running one strategy with the same context. LLM requests from all children are batched through the parent's channel for true LLM parallelism. Strategy names validated (alphanumeric + underscore only)
- **`alc.cache(prompt, opts?)`**: Session-scoped memoized LLM call (Layer 1). Returns cached response for repeated identical prompts. FIFO eviction at 256 entries. Supports `cache_key` override and `cache_skip` bypass. `alc.cache_info()` / `alc.cache_clear()` for introspection
- **`alc.parallel(items, prompt_fn, opts?)`**: Batch-parallel LLM calls over an array (Layer 1). Transforms each item into a prompt via `prompt_fn`, sends all as a single `alc.llm_batch()` call. Optional `post_fn` for response post-processing
- **`QueryId::fork(vm_index, seq)`**: Fork-specific query ID format (`f-{vm}-{seq}`) for child VM LLM request tracking
- **`query_id` auto-resolve**: `alc_continue` without explicit `query_id` now auto-resolves when exactly one query is pending. Returns error for zero or multiple pending queries
- **`query_id` in response**: Single-query `needs_response` now includes `query_id` field for explicit identification

### Changed

- **`EngineApi` trait**: Extracted transport-independent API trait from `AppService` into `algocline-core`. MCP handler now operates through `Arc<dyn EngineApi>`, enabling future remote (socket/HTTP) implementations without depending on the concrete `AppService`
- **`FeedResult`, `ExecutionResult`, `TerminalState`**: Added `Serialize` derive for future transport serialization (HTTP/gRPC)
- **`BridgeConfig`**: Added `lib_paths` field for package search paths (needed by `alc.fork` to setup child VMs)
- **`bridge` module split**: Extracted `ForkEvent`, `ForkQuery`, `register_fork` into `bridge/fork.rs` submodule (bridge.rs 1249 → mod.rs 934 + fork.rs 345)

## [0.9.0] - 2026-03-24

### Added

- **Budget control**: `ctx.budget` with `max_llm_calls` and `max_elapsed_ms` limits. `alc.budget_remaining()` (Layer 0) returns remaining capacity, `alc.budget_check()` (Layer 1) provides boolean guard for optional LLM calls. Budget is enforced at `alc.llm()` / `alc.llm_batch()` call time
- **Token estimation**: `TokenCount` and `TokenSource` types for prompt/response token tracking in `ExecutionMetrics`
- **Progress reporting**: `alc.progress(step, total, msg?)` for structured step tracking, readable via `alc_status`
- **`alc_status`**: MCP tool to query active session status — state, metrics snapshot, progress, and strategy name. Omit `session_id` to list all active sessions
- **`alc.pipe(strategies, ctx, opts?)`**: Sequential pipeline combinator. Chains multiple strategies, passing each stage's result as the next stage's `ctx.task`. Supports both `require()`-based strategies and inline functions. Records `pipe_history` for debugging

### Changed

- **`BridgeConfig` struct**: Replaced growing parameter list in `bridge::register()` with a single config struct holding `llm_tx`, `ns`, `custom_metrics`, `budget`, and `progress` handles
- **Handle-based metrics access**: `CustomMetrics`, `Budget`, `Progress` now accessed via cloneable Handle types instead of `Arc<Mutex<T>>` directly

## [0.8.0] - 2026-03-24

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.4.0` to `v0.5.0` (9 new packages: 5 orchestration — orch_fixpipe, orch_gatephase, orch_adaptive, orch_nver, orch_escalate; 3 routing — router_daao, router_semantic, router_capability; 1 optimization — optimize)

## [0.7.1] - 2026-03-22

### Fixed

- **Per-session VM isolation**: Each `alc_run` / `alc_advice` call now spawns a dedicated Lua VM. Previously, all sessions shared a single VM, causing global namespace pollution (`alc`, `ctx`, `package.loaded`) between concurrent sessions. This eliminates coroutine cross-contamination when running multiple strategies in parallel

### Changed

- **`package.loaded` clearing removed**: No longer needed since each session starts with a fresh VM

## [0.7.0] - 2026-03-22

### Added

- **`alc_stats`**: Aggregate usage stats across all logged sessions. Per-strategy counts, averages (elapsed_ms, llm_calls, rounds), and totals. Optional `strategy` filter and `days` time window
- **`alc_info`**: Diagnostic tool showing server configuration — resolved log directory (with source), tracing mode, packages directory, and version. Similar to `mise doctor`
- **Strategy tracking**: Session logs (`.json` and `.meta.json`) now record `strategy` name for all advice/eval sessions, enabling per-strategy analytics

### Changed

- **`AppConfig`**: Replaced `TranscriptConfig` with centralized `AppConfig` resolved from environment variables. Single resolution point for all configuration
- **Log directory fallback chain**: `ALC_LOG_DIR` env → `~/.algocline/logs` → `$XDG_STATE_HOME/algocline/logs` → `<cwd>/algocline-logs` → None (stderr-only). Sandbox/container environments now preserve file logging via cwd fallback
- **Tracing**: Unified `setup_tracing` into single function accepting `Option<&Path>`. File + stderr when log dir available, stderr-only otherwise
- **Crate dependencies**: Removed `algocline-engine` dependency from `algocline-mcp` — accepts `AppService` directly

### Refactored

- **`algocline-app::service`**: Split 3099-line monolithic `service.rs` into domain-based module directory (`service/config.rs`, `path.rs`, `resolve.rs`, `transcript.rs`, `eval_store.rs`, `run.rs`, `eval.rs`, `pkg.rs`, `logging.rs`, `scenario.rs`, `tests/`). No API changes

## [0.6.0] - 2026-03-20

### Added

- **`alc.json_extract(raw)`**: Extract JSON object/array from LLM output. Handles raw JSON, markdown fences (` ```json ``` `), and embedded JSON within surrounding text via balanced brace/bracket iteration
- **`alc.state.update(key, fn, default?)`**: Single-operation read-modify-write for state. Reads current value, applies transform function, writes back
- **`alc.llm_safe(prompt, opts, default)`**: Non-throwing LLM wrapper. Returns default on failure instead of raising, logs warning. For optional enrichment where failure should not abort the pipeline
- **`alc.fingerprint(str)`**: Text normalization + DJB2 hash (8-char hex). For deduplication, not cryptography
- **`alc.tuning(defaults, ctx, opts?)`**: Config merge with deep-merge support for dict-like nested tables, shallow-replace for arrays/scalars. Supports `opts.prefix` for namespaced overrides, strips `_schema` key (reserved for Layer 2 parameter metadata)

### Changed

- **`BUNDLED_VERSION`**: Updated from `v0.3.0` to `v0.4.0` (6 new strategy packages: s2a, plan_solve, rstar, faithful, moa, bot)

### Fixed

- **`alc.json_extract`**: Iterate all balanced brace/bracket pairs via `gmatch` instead of first-match-only. Fixes false-negative when non-JSON balanced text precedes valid JSON
- **`alc.fingerprint`**: DJB2 modulo corrected from `0xFFFFFFFF` (2^32-1) to `0x100000000` (2^32) per standard specification
- **`alc.tuning`**: Warn and fall back to defaults when `opts.prefix` value exists but is not a table, preventing silent unintended overrides from top-level ctx keys

## [0.5.0] - 2026-03-18

### Added

- **Scenario management**: `alc_scenario_list`, `alc_scenario_show`, `alc_scenario_install` tools for managing reusable eval scenarios in `~/.algocline/scenarios/`
- **`scenario_name` parameter**: `alc_eval` now accepts `scenario_name` to load installed scenarios by name (e.g. `"math_basic"`), in addition to existing `scenario` (inline) and `scenario_file` (path)
- **Bundled scenarios**: `alc init` / `alc_pkg_install` automatically installs scenarios from `scenarios/` subdirectory in package collections
- **Resilience pattern**: `DirEntryFailures` type alias for batch I/O operations that collect per-entry failures instead of aborting. JSON responses include `"failures"` field for diagnostics

### Changed

- **`BUNDLED_VERSION`**: Updated from `0.2.0` to `v0.3.0` (includes 9 new strategy packages, robust_qa, and 3 bundled eval scenarios)

## [0.4.0] - 2026-03-17

### Added

- **`alc_eval`**: Evaluate a strategy against a scenario with test cases and graders. Accepts inline Lua (`scenario`) or file path (`scenario_file`) with a strategy name. Strategy is auto-wired as provider via `ef.providers.algocline`
- **`alc_eval_history`**: List past eval results with optional strategy filter, sorted newest-first
- **`alc_eval_detail`**: View a specific eval result by ID in full detail
- **`alc_eval_compare`**: Compare two eval results with Welch's t-test for statistical significance via evalframe's `stats.welch_t`
- **Eval persistence**: Results automatically saved to `~/.algocline/evals/` with full JSON result + lightweight meta files for fast listing
- **`alc.time()`**: Wall-clock primitive for evalframe latency tracking
- **evalframe**: Bundled as a system dependency, auto-installed on first `alc_eval` / `alc_eval_compare` use
- **Multi-source bundled installation**: `alc init` now supports multiple source repositories (Collection and Single kinds) instead of a single URL. `--dev` mode searches local sibling directories

### Changed

- **`alc_pkg_list`**: System packages (evalframe) excluded from listing to avoid require errors and declutter output
- **Lua string escaping**: Fixed escaping for newlines/carriage returns in bridge layer

## [0.3.0] - 2026-03-15

### Added

- **`underspecified` flag**: New domain primitive on `LlmQuery` for marking prompts whose preconditions depend on intent/goal definitions outside the current context. Same serde pattern as `grounded` flag
- **`alc.specify()`**: Layer 1 prelude convenience wrapper that sets `underspecified = true`, pairing with `alc.ground()` / `grounded` pattern
- **Bundled packages v0.2.0**: 15 new packages including intent understanding (ambig, prism, intent_discovery, intent_belief), reasoning strategies (ucb, panel, cot, sc, reflect, calibrate, contrastive, meta_prompt, factscore, cove), and combinators (deliberate, pre_mortem)

### Changed

- **`BUNDLED_VERSION`**: Updated from `0.1.0` to `0.2.0`

## [0.2.1] - 2026-03-15

### Changed

- **`alc init` versioning**: Decoupled bundled packages version from algocline's own `CARGO_PKG_VERSION`. Introduced `BUNDLED_VERSION` constant (`0.1.0`) so the two can evolve independently
- **`alc init` transport**: Replaced GitHub Releases tarball download with `git clone --branch v{BUNDLED_VERSION}`, eliminating the need for release asset management

### Removed

- **`review` package**: Removed from bundled package list (poor output quality)

## [0.2.0] - 2026-03-15

### Added

- **Transcript logging**: Full prompt/response transcript saved to `~/.algocline/logs/{session_id}.json` with lightweight `.meta.json` summaries for fast listing
- **Session notes**: `alc_note` tool to annotate completed sessions with feedback/observations; notes persisted in log files with `notes_count` tracked in meta
- **Log viewer**: `alc_log_view` tool to list sessions (from meta files) or view full transcript detail
- **Auto stats**: `rounds`, `total_prompt_chars`, `total_response_chars` tracked automatically via `MetricsObserver`
- **Transcript in stats**: `transcript_to_json()` on `ExecutionMetrics` for structured prompt/response history (excluded from `to_json()` stats output)
- **Local package install**: `alc_pkg_install` accepts absolute local paths, copying directly without git clone; supports both single packages and collections with overwrite semantics for dev workflow
- **Collection install**: Package repositories with `*/init.lua` subdirs are detected as collections and each subdir installed as a separate package
- **Test suite**: 151 tests across all crates — unit tests, property-based tests (proptest), path traversal rejection, chunk function invariants, state machine transitions

### Changed

- **Package architecture**: Standard packages extracted to separate `algocline-bundled-packages` repository; `alc_advice` auto-installs from GitHub if requested package is missing
- **MSRV**: Updated from 1.77 to 1.88

## [0.1.0] - 2026-03-01

### Added

- Initial release
- MCP server with `alc_run`, `alc_continue`, `alc_advice` tools
- Three-layer Lua StdLib: Layer 0 (Rust primitives), Layer 1 (Lua prelude), Layer 2 (packages via `require()`)
- `alc.llm()` / `alc.llm_batch()` — coroutine-based async LLM calls
- `alc.json_encode` / `alc.json_decode` — serde_json bridge
- `alc.log()` — tracing bridge
- `alc.state` — persistent key-value store (`~/.algocline/state/`)
- `alc.chunk()` — text segmentation (lines/chars with overlap)
- `alc.stats` — custom metrics recording
- Prelude combinators: `alc.map`, `alc.reduce`, `alc.vote`, `alc.filter`
- Package management: `alc_pkg_list`, `alc_pkg_install`, `alc_pkg_remove`
- `alc init` — bundled package installer (GitHub Releases + local fallback)
- Domain model: `ExecutionState` state machine with `PendingQueries` join barrier
- `ExecutionObserver` trait for cross-cutting concerns
- `SessionRegistry` for concurrent session management
- `ContainedPath` for path traversal prevention
- Coroutine-based execution via `mlua-isle` (non-blocking `alc.llm()`)
