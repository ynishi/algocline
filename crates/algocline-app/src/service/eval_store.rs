use std::path::PathBuf;

use super::path::ContainedPath;

// ─── Eval Result Store ──────────────────────────────────────────

pub(super) fn evals_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("evals"))
}

/// Persist eval result to `~/.algocline/evals/{strategy}_{timestamp}.json`.
///
/// Silently returns on I/O errors — storage must not break eval execution.
pub(super) fn save_eval_result(strategy: &str, result_json: &str) {
    let dir = match evals_dir() {
        Ok(d) => d,
        Err(_) => return,
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp = now.as_secs();
    let eval_id = format!("{strategy}_{timestamp}");

    // Parse result to extract summary fields for meta file
    let parsed: serde_json::Value = match serde_json::from_str(result_json) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Write full result
    let path = match ContainedPath::child(&dir, &format!("{eval_id}.json")) {
        Ok(p) => p,
        Err(_) => return,
    };
    let _ = std::fs::write(&path, result_json);

    // Write lightweight meta file for listing
    let meta = build_meta(&eval_id, strategy, timestamp, &parsed);

    if let Ok(meta_path) = ContainedPath::child(&dir, &format!("{eval_id}.meta.json")) {
        let _ = serde_json::to_string(&meta).map(|s| std::fs::write(&meta_path, s));
    }
}

/// Build the lightweight meta JSON from a full eval result.
///
/// Pure function — no I/O, no side effects. Testable independently.
pub(super) fn build_meta(
    eval_id: &str,
    strategy: &str,
    timestamp: u64,
    parsed: &serde_json::Value,
) -> serde_json::Value {
    let result_obj = parsed.get("result");
    let stats_obj = parsed.get("stats");
    let aggregated = result_obj.and_then(|r| r.get("aggregated"));

    serde_json::json!({
        "eval_id": eval_id,
        "strategy": strategy,
        "timestamp": timestamp,
        "pass_rate": aggregated.and_then(|a| a.get("pass_rate")),
        "mean_score": aggregated.and_then(|a| a.get("scores")).and_then(|s| s.get("mean")),
        "total_cases": aggregated.and_then(|a| a.get("total")),
        "passed": aggregated.and_then(|a| a.get("passed")),
        "llm_calls": stats_obj.and_then(|s| s.get("auto")).and_then(|a| a.get("llm_calls")),
        "elapsed_ms": stats_obj.and_then(|s| s.get("auto")).and_then(|a| a.get("elapsed_ms")),
        "summary": result_obj.and_then(|r| r.get("summary")),
    })
}

/// List eval history from a given directory, optionally filtered by strategy.
///
/// Pure directory-scanning function — testable with tmpdir.
pub(super) fn list_eval_history(
    dir: &std::path::Path,
    strategy: Option<&str>,
    limit: usize,
) -> Result<String, String> {
    if !dir.exists() {
        return Ok(serde_json::json!({ "evals": [] }).to_string());
    }

    let mut entries: Vec<serde_json::Value> = Vec::new();

    let read_dir = std::fs::read_dir(dir).map_err(|e| format!("Failed to read evals dir: {e}"))?;

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        // Skip meta files
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".meta."))
        {
            continue;
        }

        // Read meta file (lightweight) if it exists.
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        let meta_path = match ContainedPath::child(dir, &format!("{stem}.meta.json")) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let meta = if meta_path.exists() {
            std::fs::read_to_string(&*meta_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        } else {
            None
        };

        if let Some(meta) = meta {
            if let Some(filter) = strategy {
                if meta.get("strategy").and_then(|s| s.as_str()) != Some(filter) {
                    continue;
                }
            }
            entries.push(meta);
        }
    }

    // Sort by timestamp descending (newest first)
    entries.sort_by(|a, b| {
        let ts_a = a
            .get("timestamp")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let ts_b = b
            .get("timestamp")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        ts_b.cmp(&ts_a)
    });
    entries.truncate(limit);

    Ok(serde_json::json!({ "evals": entries }).to_string())
}

// ─── Eval Comparison Helpers ─────────────────────────────────────

/// Escape a string for embedding in a Lua single-quoted string literal.
///
/// Handles backslash, single quote, newline, and carriage return —
/// the characters that would break or alter a `'...'` Lua string.
pub(super) fn escape_for_lua_sq(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Extract strategy name from eval_id (format: "{strategy}_{timestamp}").
pub(super) fn extract_strategy_from_id(eval_id: &str) -> Option<&str> {
    eval_id.rsplit_once('_').map(|(prefix, _)| prefix)
}

/// Persist a comparison result to `~/.algocline/evals/`.
pub(super) fn save_compare_result(eval_id_a: &str, eval_id_b: &str, result_json: &str) {
    let dir = match evals_dir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let filename = format!("compare_{eval_id_a}_vs_{eval_id_b}.json");
    if let Ok(path) = ContainedPath::child(&dir, &filename) {
        let _ = std::fs::write(&path, result_json);
    }
}
