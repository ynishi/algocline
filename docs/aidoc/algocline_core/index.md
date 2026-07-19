# algocline-core 0.45.0

Core domain types and primitives shared across the algocline workspace.

Provides the shared vocabulary consumed by `algocline-engine`,
`algocline-app`, and `algocline-mcp`: [`AppDir`] path handling,
[`Budget`] / token accounting, the [`EngineApi`] trait, execution
and query domain types, metrics collection, package (`pkg`) metadata,
progress reporting ([`ProgressInfo`]), the recent-log ring buffer,
and shared state primitives.

## Modules

- [`domain`](domain.md): Re-exports for domain types.
- [`execution`](execution.md): Pure execution service layer for `algocline-core`.
- [`execution::cancel`](execution__cancel.md): Cancellation and failure types for the `ExecutionService` layer.
- [`execution::error`](execution__error.md): Dedicated error enums for each `ExecutionService` verb.
- [`execution::pause`](execution__pause.md): Pause-related types for the `ExecutionService` layer.
- [`execution::progress`](execution__progress.md): Progress observation types for the `ExecutionService` layer.
- [`execution::resume`](execution__resume.md): Resume-related types for the `ExecutionService` layer.
- [`execution::service`](execution__service.md): `ExecutionService` trait — the primary verb surface of the new service layer.
- [`execution::session_id`](execution__session_id.md): Session identifier type for the `ExecutionService` layer.
- [`execution::spec`](execution__spec.md): Session specification types for the `ExecutionService` layer.
- [`execution::state`](execution__state.md): Execution state types (v2) for the `ExecutionService` layer.
- [`metrics`](metrics.md): Metrics collection primitives.
- [`pkg`](pkg.md): Canonical projection of a Lua package's `M.meta` block.
- [`recent_log`](recent_log.md): Per-session recent-log ring buffer.

