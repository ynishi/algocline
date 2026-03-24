# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
