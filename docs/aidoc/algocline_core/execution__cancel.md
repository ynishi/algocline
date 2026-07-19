# algocline-core::execution::cancel

Cancellation and failure types for the `ExecutionService` layer.

## Types

- `CancelCode` — Categorized cause of a cancellation.
- `CancelInfo` — Information recorded when a session transitions to `Cancelled`.
- `CancelReason` — Reason supplied to [`crate::execution::ExecutionService::cancel`].
- `FailureInfo` — Information recorded when a session transitions to `Failed`.
- `FailureKind` — Categorized kind of failure.

