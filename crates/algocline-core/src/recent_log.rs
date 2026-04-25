use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── LogEntry ────────────────────────────────────────────────

/// A single log entry captured from a running session.
///
/// Entries are produced by Lua `print()`, `alc.log()`, and engine-internal
/// events, then accumulated in a per-session ring buffer (cap=20) for
/// lightweight observability via `alc_status`.
///
/// # Fields
///
/// - `ts` — Unix milliseconds (i64) when the entry was recorded.
/// - `level` — Severity string: `"info"`, `"warn"`, `"error"`, `"debug"`, etc.
/// - `source` — Originator: `"alc.lua.print"`, `"alc.log"`, `"engine"`, etc.
/// - `message` — Human-readable log text.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEntry {
    /// Unix milliseconds timestamp of when the entry was recorded.
    pub ts: i64,
    /// Severity level string (e.g. "info", "warn", "error", "debug").
    pub level: String,
    /// Originator identifier (e.g. "alc.lua.print", "alc.log", "engine").
    pub source: String,
    /// Human-readable log message.
    pub message: String,
}

impl LogEntry {
    /// Create a new `LogEntry` with the current wall-clock timestamp.
    ///
    /// # Arguments
    ///
    /// - `level` — Severity label string.
    /// - `source` — Originator identifier string.
    /// - `message` — Log message text.
    ///
    /// # Returns
    ///
    /// A new `LogEntry` with `ts` set to the current Unix millisecond
    /// timestamp.  If `SystemTime` is before the Unix epoch (broken wall
    /// clock), `ts` is saturated to `0`.
    pub fn new(
        level: impl Into<String>,
        source: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        // Safety: duration_since can only fail if the wall clock predates
        // UNIX_EPOCH (1970-01-01), which indicates a broken system clock.
        // Saturating to zero is harmless for observability purposes.
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        Self {
            ts,
            level: level.into(),
            source: source.into(),
            message: message.into(),
        }
    }
}

// ─── LogSink ─────────────────────────────────────────────────

/// A shared, bounded ring-buffer sink for [`LogEntry`] items.
///
/// Wraps `Arc<Mutex<VecDeque<LogEntry>>>` and enforces a maximum capacity
/// of 20 entries.  Oldest entries are evicted when the cap is exceeded.
///
/// `LogSink` can be cloned cheaply (clones the `Arc`, not the buffer).
/// It is intended to be passed to the Lua bridge so that log output from
/// both `print()` and `alc.log()` is routed into the session's ring buffer.
#[derive(Clone, Debug)]
pub struct LogSink(Arc<Mutex<VecDeque<LogEntry>>>);

/// Maximum number of entries retained in a [`LogSink`].
pub const LOG_SINK_CAP: usize = 20;

impl LogSink {
    /// Create a new, empty `LogSink`.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::with_capacity(
            LOG_SINK_CAP + 1,
        ))))
    }

    /// Push a new entry into the ring buffer, evicting the oldest if necessary.
    ///
    /// # Arguments
    ///
    /// - `entry` — The [`LogEntry`] to append.
    ///
    /// # Errors
    ///
    /// If the internal mutex is poisoned (only possible on OOM-induced panic),
    /// the entry is silently dropped.  This is the approved "observation/recording"
    /// policy — log capture failure must not interrupt execution.
    pub fn push(&self, entry: LogEntry) {
        if let Ok(mut buf) = self.0.lock() {
            buf.push_back(entry);
            if buf.len() > LOG_SINK_CAP {
                buf.pop_front();
            }
        }
    }

    /// Snapshot the current ring-buffer contents as a JSON array.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value::Array` of serialized [`LogEntry`] objects,
    /// in chronological order (oldest first).  Returns an empty array if
    /// the mutex is poisoned.
    pub fn to_json(&self) -> serde_json::Value {
        if let Ok(buf) = self.0.lock() {
            let entries: Vec<serde_json::Value> = buf
                .iter()
                .filter_map(|e| serde_json::to_value(e).ok())
                .collect();
            serde_json::Value::Array(entries)
        } else {
            serde_json::Value::Array(vec![])
        }
    }

    /// Snapshot the current ring-buffer contents as a `Vec<LogEntry>`.
    ///
    /// Useful for programmatic access (e.g. in `SessionStatus::snapshot`).
    /// Returns an empty vec if the mutex is poisoned.
    pub fn entries(&self) -> Vec<LogEntry> {
        if let Ok(buf) = self.0.lock() {
            buf.iter().cloned().collect()
        } else {
            vec![]
        }
    }
}

impl Default for LogSink {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // T1: happy path — push entries and read them back
    #[test]
    fn log_sink_push_and_read() {
        let sink = LogSink::new();
        sink.push(LogEntry::new("info", "engine", "hello"));
        sink.push(LogEntry::new("warn", "alc.log", "world"));

        let entries = sink.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].level, "info");
        assert_eq!(entries[0].source, "engine");
        assert_eq!(entries[0].message, "hello");
        assert_eq!(entries[1].level, "warn");
        assert_eq!(entries[1].message, "world");
    }

    // T2: boundary — cap=20 enforcement: 21st entry evicts the oldest
    #[test]
    fn log_sink_cap_evicts_oldest() {
        let sink = LogSink::new();
        for i in 0..=20u32 {
            sink.push(LogEntry::new("info", "engine", format!("msg-{i}")));
        }

        let entries = sink.entries();
        assert_eq!(entries.len(), LOG_SINK_CAP);
        // The first entry should be msg-1 (msg-0 was evicted)
        assert_eq!(entries[0].message, "msg-1");
        // The last entry should be msg-20
        assert_eq!(entries[LOG_SINK_CAP - 1].message, "msg-20");
    }

    // T2: boundary — empty sink
    #[test]
    fn log_sink_empty() {
        let sink = LogSink::new();
        assert!(sink.entries().is_empty());
        let json = sink.to_json();
        assert_eq!(json, serde_json::Value::Array(vec![]));
    }

    // T1: to_json serializes correctly
    #[test]
    fn log_sink_to_json_shape() {
        let sink = LogSink::new();
        sink.push(LogEntry::new("debug", "alc.lua.print", "test-msg"));

        let json = sink.to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["level"], "debug");
        assert_eq!(arr[0]["source"], "alc.lua.print");
        assert_eq!(arr[0]["message"], "test-msg");
        assert!(arr[0].get("ts").is_some());
    }

    // T1: clone shares the same underlying buffer
    #[test]
    fn log_sink_clone_shares_buffer() {
        let sink = LogSink::new();
        let sink2 = sink.clone();
        sink.push(LogEntry::new("info", "engine", "shared"));
        assert_eq!(sink2.entries().len(), 1);
    }

    // T3: exactly at cap boundary (20 entries) — no eviction yet
    #[test]
    fn log_sink_exactly_at_cap() {
        let sink = LogSink::new();
        for i in 0..20u32 {
            sink.push(LogEntry::new("info", "engine", format!("msg-{i}")));
        }
        let entries = sink.entries();
        assert_eq!(entries.len(), 20);
        assert_eq!(entries[0].message, "msg-0");
        assert_eq!(entries[19].message, "msg-19");
    }
}
