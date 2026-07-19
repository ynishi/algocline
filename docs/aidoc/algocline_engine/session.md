# algocline-engine::session

Session-based Lua execution with pause/resume on alc.llm() calls.

Runtime layer: ties Domain (ExecutionState) and Metrics (ExecutionMetrics)
together with channel-based Lua pause/resume machinery.

## Types

- `ExecutionResult` — Session completion data: terminal state + metrics.
- `FeedResult` — Result of a session interaction (start or feed).
- `PendingFilter` — Per-field filter controlling which `LlmQuery` attributes are projected
- `PromptProjection` — Prompt projection mode — 3 states rather than a bool so that truncation
- `Session` — A Lua execution session with domain state tracking.
- `SessionError` — (no documentation)
- `SessionRegistry` — Manages active sessions.

## Constants

- `DEFAULT_PROMPT_PREVIEW_CHARS` — Default preview length (chars) used when `PendingFilter::preset_preview()`

