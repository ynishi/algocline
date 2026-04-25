use std::path::Path;

use algocline_core::ExecutionMetrics;

use super::config::AppConfig;
use super::path::ContainedPath;

/// Write transcript log to `{dir}/{session_id}.json`.
///
/// Returns `Err` on main log write failure; the meta file
/// (`{session_id}.meta.json`) is best-effort — its failure is silent
/// because the main log holds the authoritative data and meta is only
/// a lightweight projection used by log_list to avoid reading full
/// transcripts.
pub(super) fn write_transcript_log(
    config: &AppConfig,
    session_id: &str,
    metrics: &ExecutionMetrics,
    strategy: Option<&str>,
) -> Result<(), String> {
    let log_dir = match (&config.log_dir, config.log_enabled) {
        (Some(dir), true) => dir,
        _ => return Ok(()),
    };

    let transcript = metrics.transcript_to_json();
    if transcript.is_empty() {
        return Ok(());
    }

    let stats = metrics.to_json();

    // Extract task hint from first prompt (truncated to 100 chars)
    let task_hint = transcript
        .first()
        .and_then(|e| e.get("prompt"))
        .and_then(|p| p.as_str())
        .map(|s| {
            if s.len() <= 100 {
                s.to_string()
            } else {
                // Find a char boundary at or before 100 bytes
                let mut end = 100;
                while end > 0 && !s.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &s[..end])
            }
        });

    let auto_stats = &stats["auto"];

    let log_entry = serde_json::json!({
        "session_id": session_id,
        "strategy": strategy,
        "task_hint": task_hint,
        "stats": auto_stats,
        "transcript": transcript,
    });

    std::fs::create_dir_all(log_dir)
        .map_err(|e| format!("transcript log: failed to create log dir: {e}"))?;

    let path = ContainedPath::child(log_dir, &format!("{session_id}.json"))
        .map_err(|e| format!("transcript log: invalid path: {e}"))?;
    let content = serde_json::to_string_pretty(&log_entry)
        .map_err(|e| format!("transcript log: failed to serialize: {e}"))?;

    std::fs::write(&path, content).map_err(|e| format!("transcript log: failed to write: {e}"))?;

    // Write lightweight meta file for log_list (avoids reading full transcript)
    let meta = serde_json::json!({
        "session_id": session_id,
        "strategy": strategy,
        "task_hint": task_hint,
        "elapsed_ms": auto_stats.get("elapsed_ms"),
        "rounds": auto_stats.get("rounds"),
        "llm_calls": auto_stats.get("llm_calls"),
        "total_prompt_chars": auto_stats.get("total_prompt_chars"),
        "total_response_chars": auto_stats.get("total_response_chars"),
        "prompt_tokens": auto_stats.get("prompt_tokens"),
        "response_tokens": auto_stats.get("response_tokens"),
        "total_tokens": auto_stats.get("total_tokens"),
        "notes_count": 0,
    });
    if let Ok(meta_path) = ContainedPath::child(log_dir, &format!("{session_id}.meta.json")) {
        let _ = serde_json::to_string(&meta).map(|s| std::fs::write(&meta_path, s));
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use algocline_core::{AppDir, ExecutionMetrics, ExecutionObserver, LlmQuery, QueryId};

    use super::super::config::{AppConfig, LogDirSource};
    use super::write_transcript_log;

    fn make_metrics_with_transcript() -> ExecutionMetrics {
        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();
        observer.on_paused(&[LlmQuery {
            id: QueryId::single(),
            prompt: "test prompt".into(),
            system: None,
            max_tokens: 100,
            grounded: false,
            underspecified: false,
        }]);
        metrics
    }

    fn config_with_log_dir(log_dir: PathBuf) -> AppConfig {
        AppConfig {
            log_dir: Some(log_dir),
            log_dir_source: LogDirSource::EnvVar,
            log_enabled: true,
            prompt_preview_chars: 200,
            app_dir: Arc::new(AppDir::new(PathBuf::from(".algocline"))),
        }
    }

    #[test]
    fn write_transcript_log_ok_on_valid_dir() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let config = config_with_log_dir(tmp.path().to_path_buf());
        let metrics = make_metrics_with_transcript();
        let result = write_transcript_log(&config, "test-session-ok", &metrics, None);
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }

    #[test]
    fn write_transcript_log_returns_ok_when_log_disabled() {
        let config = AppConfig::default(); // log_enabled = false
        let metrics = make_metrics_with_transcript();
        let result = write_transcript_log(&config, "test-session-nolog", &metrics, None);
        assert!(result.is_ok(), "disabled log should return Ok(())");
    }

    #[test]
    fn write_transcript_log_returns_ok_when_transcript_empty() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let config = config_with_log_dir(tmp.path().to_path_buf());
        let metrics = ExecutionMetrics::new(); // no transcript entries
        let result = write_transcript_log(&config, "test-session-empty", &metrics, None);
        assert!(result.is_ok(), "empty transcript should return Ok(())");
    }

    #[test]
    fn write_transcript_log_returns_err_on_write_failure() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let log_dir = tmp.path().to_path_buf();
        // Block the write target by creating a *directory* at the session file path.
        // create_dir_all(log_dir) succeeds, ContainedPath::child succeeds,
        // but std::fs::write fails with "Is a directory".
        std::fs::create_dir_all(log_dir.join("blocked-session.json"))
            .expect("pre-create dir to block write");
        let config = config_with_log_dir(log_dir);
        let metrics = make_metrics_with_transcript();
        let result = write_transcript_log(&config, "blocked-session", &metrics, None);
        assert!(
            result.is_err(),
            "expected Err when write target is a directory"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("transcript log"),
            "error message should mention 'transcript log', got: {msg}"
        );
    }
}

/// Append a note to an existing log file.
///
/// Reads `{dir}/{session_id}.json`, adds the note to `"notes"` array, writes back.
/// Returns Ok with the note count, or Err if the log file doesn't exist.
pub(super) fn append_note(
    dir: &Path,
    session_id: &str,
    content: &str,
    title: Option<&str>,
) -> Result<usize, String> {
    let path = ContainedPath::child(dir, &format!("{session_id}.json"))?;
    if !path.as_ref().exists() {
        return Err(format!("Log file not found for session '{session_id}'"));
    }

    let raw = std::fs::read_to_string(&path).map_err(|e| format!("Failed to read log: {e}"))?;
    let mut doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("Failed to parse log: {e}"))?;

    let timestamp = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    };

    let note = serde_json::json!({
        "timestamp": timestamp,
        "title": title,
        "content": content,
    });

    let notes = doc
        .as_object_mut()
        .ok_or("Log file is not a JSON object")?
        .entry("notes")
        .or_insert_with(|| serde_json::json!([]));

    let arr = notes
        .as_array_mut()
        .ok_or("'notes' field is not an array")?;
    arr.push(note);
    let count = arr.len();

    let output =
        serde_json::to_string_pretty(&doc).map_err(|e| format!("Failed to serialize: {e}"))?;
    std::fs::write(path.as_ref(), output).map_err(|e| format!("Failed to write log: {e}"))?;

    // Update notes_count in meta file (best-effort)
    if let Ok(meta_path) = ContainedPath::child(dir, &format!("{session_id}.meta.json")) {
        if meta_path.as_ref().exists() {
            if let Ok(raw) = std::fs::read_to_string(&meta_path) {
                if let Ok(mut meta) = serde_json::from_str::<serde_json::Value>(&raw) {
                    meta["notes_count"] = serde_json::json!(count);
                    if let Ok(s) = serde_json::to_string(&meta) {
                        let _ = std::fs::write(&meta_path, s);
                    }
                }
            }
        }
    }

    Ok(count)
}
