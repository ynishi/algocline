# algocline-core::recent_log

Per-session recent-log ring buffer.

[`LogEntry`] captures events from Lua `print()`, `alc.log()`, and
engine-internal callsites. [`LogSink`] accumulates entries with a
fixed cap (=20) for retrieval via `alc_log_view` and MCP resource
endpoints, providing bounded per-session observability without
unbounded memory growth.

## Types

- `LogEntry` — A single log entry captured from a running session.
- `LogSink` — A shared, bounded ring-buffer sink for [`LogEntry`] items.

## Constants

- `LOG_SINK_CAP` — Maximum number of entries retained in a [`LogSink`].

