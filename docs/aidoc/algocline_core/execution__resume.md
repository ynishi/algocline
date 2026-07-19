# algocline-core::execution::resume

Resume-related types for the `ExecutionService` layer.

## Types

- `QueryResponse` — A single LLM response in a [`ResumePayload::Batch`].
- `ResumeOutcome` — Outcome returned by [`crate::execution::ExecutionService::resume`].
- `ResumePayload` — Payload supplied to [`crate::execution::ExecutionService::resume`].
- `TerminalOutcome` — Terminal outcome for a session — carried by [`ResumeOutcome::Terminal`] and

