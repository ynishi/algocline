# algocline-core::metrics

Metrics collection primitives.

Aggregates per-query token usage, budget consumption, custom metric
handles ([`CustomMetrics`]), and [`ProgressInfo`], and forwards
observation events through registered
[`ExecutionObserver`](crate::observer::ExecutionObserver) instances
and the recent-log sink.

## Types

- `ExecutionMetrics` — Measurement data for a single execution.
- `MetricsObserver` — Updates SessionStatus via the ExecutionObserver trait.
- `StatsHandle` — Read-only handle exposing auto-counted [`SessionStatus`] metrics to

