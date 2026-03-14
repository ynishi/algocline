# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
