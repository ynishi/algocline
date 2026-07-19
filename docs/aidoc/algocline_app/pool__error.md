# algocline-app::pool::error

Error types produced by the process-pool IPC layer.

All variants propagate to callers via `Result<T, PoolError>` and
are converted to `String` at the `AppService` boundary (Subtask
5/6) to satisfy the existing `EngineApi` wire contract.

## Types

- `PoolError` — Errors produced by the process-pool IPC layer.

