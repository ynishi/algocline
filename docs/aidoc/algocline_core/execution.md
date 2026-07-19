# algocline-core::execution

Pure execution service layer for `algocline-core`.

This module provides the [`ExecutionService`] trait and all associated value types
for the new service layer design.  It coexists with the legacy types in
`crate::state` and `crate::engine_api` without modifying them (parallel evolution).

## Module structure

| Module | Key types |
|--------|-----------|
| [`session_id`] | [`SessionId`] |
| [`spec`] | [`SessionSpec`], [`SpecKind`], [`ScenarioRef`] |
| [`state`] | [`ExecutionState`] (v2), [`ExecutionStateTag`], [`ExecutionResult`] |
| [`pause`] | [`PauseInfo`], [`PauseKind`], [`PausePrompt`] |
| [`resume`] | [`ResumePayload`], [`QueryResponse`], [`ResumeOutcome`], [`TerminalOutcome`] |
| [`cancel`] | [`CancelReason`], [`CancelCode`], [`CancelInfo`], [`FailureInfo`], [`FailureKind`] |
| [`progress`] | [`ProgressEvent`], [`ObserverHandle`] |
| [`error`] | [`SpawnError`], [`StateError`], [`ResumeError`], [`CancelError`], [`ObserveError`], [`AwaitError`], [`ObserverRecvError`] |
| [`service`] | [`ExecutionService`] |

## Access path

Types in this module are accessed as `algocline_core::execution::Foo`.
They are **not** re-exported at the top level of `algocline_core` to avoid
naming conflicts with legacy top-level types (e.g., `QueryResponse`).

