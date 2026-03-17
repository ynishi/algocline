# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
