# algocline-core::execution::state

Execution state types (v2) for the `ExecutionService` layer.

This module defines [`ExecutionState`] v2, [`ExecutionStateTag`], and [`ExecutionResult`].
These types are **distinct** from the legacy `state::ExecutionState` in
`crates/algocline-core/src/state.rs`; both coexist while the migration to the new
`ExecutionService` API is in progress.

## Types

- `ExecutionResult` — Payload carried by [`ExecutionState::Done`].
- `ExecutionState` — Rich execution state used by [`crate::execution::ExecutionService`].
- `ExecutionStateTag` — Lightweight discriminant for [`ExecutionState`].

