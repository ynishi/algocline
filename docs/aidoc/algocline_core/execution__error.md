# algocline-core::execution::error

Dedicated error enums for each `ExecutionService` verb.

All error types are **closed enums** (no `#[non_exhaustive]`): wire consumers
must handle every variant, and the adapter layer converts to a string before
crossing the MCP boundary (design-v1.md §2).

All error types derive `serde::Serialize + serde::Deserialize` so they can be
embedded in structured JSON responses if needed by callers.

## Types

- `AwaitError` — Error returned by [`crate::execution::ExecutionService::await_terminal`].
- `CancelError` — Error returned by [`crate::execution::ExecutionService::cancel`].
- `ObserveError` — Error returned by [`crate::execution::ExecutionService::observe`].
- `ObserverRecvError` — Error returned by [`crate::execution::ObserverHandle::recv`] and
- `ResumeError` — Error returned by [`crate::execution::ExecutionService::resume`].
- `SpawnError` — Error returned by [`crate::execution::ExecutionService::spawn`].
- `StateError` — Error returned by [`crate::execution::ExecutionService::state`].

