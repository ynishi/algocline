use std::path::PathBuf;

use algocline_core::AppDir;

use super::path::ContainedPath;

// ─── Eval Result Store ──────────────────────────────────────────

pub(super) fn evals_dir(app_dir: &AppDir) -> PathBuf {
    app_dir.evals_dir()
}

/// Insert a top-level string field into a JSON object response so that
/// storage failures (eval persistence, comparison persistence) surface
/// on the MCP wire response. If `json_str` is not a JSON object the
/// helper returns it unchanged — callers should never pass a
/// non-object payload here, but staying defensive avoids a crash on
/// malformed strategy output.
pub(super) fn splice_response_string(json_str: &str, key: &str, value: &str) -> String {
    if let Ok(serde_json::Value::Object(mut map)) = serde_json::from_str(json_str) {
        map.insert(
            key.to_string(),
            serde_json::Value::String(value.to_string()),
        );
        return serde_json::Value::Object(map).to_string();
    }
    json_str.to_string()
}

/// Insert a top-level string-array field into a JSON object response.
///
/// When `values` is empty the helper returns `json_str` unchanged — callers
/// should only invoke this when there are actual warnings to surface. If
/// `json_str` is not a JSON object it is returned unchanged.
pub(super) fn splice_response_warnings(json_str: &str, key: &str, values: &[String]) -> String {
    if values.is_empty() {
        return json_str.to_string();
    }
    if let Ok(serde_json::Value::Object(mut map)) = serde_json::from_str(json_str) {
        map.insert(key.to_string(), serde_json::json!(values));
        return serde_json::Value::Object(map).to_string();
    }
    json_str.to_string()
}

/// Persist eval result to `{app_dir}/evals/{strategy}_{timestamp}.json`.
///
/// Returns `Ok(())` on success, `Err(String)` when the on-disk write
/// fails. Storage failures must not break eval execution itself, so the
/// caller surfaces the error as an additive warning on the response
/// rather than aborting the eval — but the caller MUST surface it,
/// otherwise the operator silently loses eval history.
pub(super) fn save_eval_result(
    app_dir: &AppDir,
    strategy: &str,
    result_json: &str,
) -> Result<(), String> {
    let dir = evals_dir(app_dir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create evals dir {}: {e}", dir.display()))?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp = now.as_secs();
    let eval_id = format!("{strategy}_{timestamp}");

    // Parse result to extract summary fields for meta file. A parse
    // failure here means the caller passed a malformed JSON blob —
    // return it as Err rather than silently dropping the persistence.
    let parsed: serde_json::Value = serde_json::from_str(result_json)
        .map_err(|e| format!("failed to parse eval result JSON: {e}"))?;

    // Write full result
    let path = ContainedPath::child(&dir, &format!("{eval_id}.json"))
        .map_err(|e| format!("invalid eval_id {eval_id}: {e}"))?;
    std::fs::write(&path, result_json)
        .map_err(|e| format!("failed to write eval result {}: {e}", path.display()))?;

    // Write lightweight meta file for listing
    let meta = build_meta(&eval_id, strategy, timestamp, &parsed);
    let meta_path = ContainedPath::child(&dir, &format!("{eval_id}.meta.json"))
        .map_err(|e| format!("invalid eval_id meta {eval_id}: {e}"))?;
    let meta_str =
        serde_json::to_string(&meta).map_err(|e| format!("failed to serialize eval meta: {e}"))?;
    std::fs::write(&meta_path, meta_str)
        .map_err(|e| format!("failed to write eval meta {}: {e}", meta_path.display()))
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
///
/// Per-file meta.json parse failures (corruption) are collected into a
/// `"warnings"` array in the returned JSON so the caller (MCP wire layer)
/// can surface them to the UI. Meta file absent is the legitimate
/// pre-v0.x case and stays silent.
pub(super) fn list_eval_history(
    dir: &std::path::Path,
    strategy: Option<&str>,
    limit: usize,
) -> Result<String, String> {
    if !dir.exists() {
        return Ok(serde_json::json!({ "evals": [] }).to_string());
    }

    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

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
        // Distinguish absent (legitimate) from corrupted (surface as warning).
        let meta: Option<serde_json::Value> = if meta_path.exists() {
            match std::fs::read_to_string(&*meta_path) {
                Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        warnings.push(format!("eval meta parse {}: {e}", meta_path.display()));
                        None
                    }
                },
                Err(e) => {
                    warnings.push(format!("eval meta read {}: {e}", meta_path.display()));
                    None
                }
            }
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

    let mut response = serde_json::json!({ "evals": entries });
    if !warnings.is_empty() {
        response["warnings"] = serde_json::json!(warnings);
    }
    Ok(response.to_string())
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

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod splice_warnings_tests {
    use super::splice_response_warnings;

    #[test]
    fn empty_warnings_returns_original_unchanged() {
        let json = r#"{"status":"ok"}"#;
        assert_eq!(splice_response_warnings(json, "warnings", &[]), json);
    }

    #[test]
    fn non_empty_warnings_added_as_array() {
        let json = r#"{"status":"ok"}"#;
        let warnings = vec!["alc.lock parse error".to_string()];
        let out = splice_response_warnings(json, "warnings", &warnings);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let arr = v["warnings"].as_array().expect("warnings must be array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].as_str(), Some("alc.lock parse error"));
    }

    #[test]
    fn non_object_json_returned_unchanged() {
        let json = r#""just a string""#;
        let warnings = vec!["something".to_string()];
        assert_eq!(splice_response_warnings(json, "warnings", &warnings), json);
    }
}

/// Persist a comparison result to `{app_dir}/evals/`.
///
/// Returns `Ok(())` on success or `Err(String)` when the on-disk write
/// fails. The caller is expected to surface failures as an additive
/// warning on the response so the operator notices that the cache layer
/// is degraded.
pub(super) fn save_compare_result(
    app_dir: &AppDir,
    eval_id_a: &str,
    eval_id_b: &str,
    result_json: &str,
) -> Result<(), String> {
    let dir = evals_dir(app_dir);
    let filename = format!("compare_{eval_id_a}_vs_{eval_id_b}.json");
    let path = ContainedPath::child(&dir, &filename)
        .map_err(|e| format!("invalid compare filename {filename}: {e}"))?;
    std::fs::write(&path, result_json)
        .map_err(|e| format!("failed to write compare result {}: {e}", path.display()))
}

// ─── Phase 3 MED batch: list_eval_history error-propagation tests ─

#[cfg(test)]
mod list_eval_history_tests {
    use super::list_eval_history;

    /// Helper: parse the returned JSON and extract optional `warnings` array.
    fn warnings_from_json(json: &str) -> Vec<String> {
        let v: serde_json::Value = serde_json::from_str(json).expect("must be valid JSON");
        v.get("warnings")
            .and_then(|w| w.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn absent_dir_returns_empty_no_warnings() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("evals");
        // dir does not exist
        let result = list_eval_history(&dir, None, 50).unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        let evals = v["evals"].as_array().unwrap();
        assert!(evals.is_empty(), "absent dir must return empty evals");
        let warns = warnings_from_json(&result);
        assert!(
            warns.is_empty(),
            "absent dir must produce no warnings, got {warns:?}"
        );
    }

    #[test]
    fn meta_absent_is_silent_no_warning() {
        // A .json eval file exists but no .meta.json — that is the
        // pre-meta legacy case; it should produce no warning.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("evals");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("cot_1.json"), b"{}").unwrap();
        // No cot_1.meta.json written.

        let result = list_eval_history(&dir, None, 50).unwrap();
        let warns = warnings_from_json(&result);
        assert!(
            warns.is_empty(),
            "absent meta file must produce no warnings, got {warns:?}"
        );
    }

    #[test]
    fn corrupt_meta_surfaces_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("evals");
        std::fs::create_dir_all(&dir).unwrap();
        // Write the eval result file and a corrupt meta file.
        std::fs::write(dir.join("cot_1.json"), b"{}").unwrap();
        std::fs::write(dir.join("cot_1.meta.json"), b"not json {{{{").unwrap();

        let result = list_eval_history(&dir, None, 50).unwrap();
        let warns = warnings_from_json(&result);
        assert!(
            !warns.is_empty(),
            "corrupt meta.json must produce at least one warning, got {warns:?}"
        );
        assert!(
            warns[0].contains("parse"),
            "warning must mention parse: {}",
            warns[0]
        );
    }

    #[test]
    fn valid_meta_included_no_warnings() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("evals");
        std::fs::create_dir_all(&dir).unwrap();
        let meta = r#"{"eval_id":"cot_1","strategy":"cot","timestamp":1}"#;
        std::fs::write(dir.join("cot_1.json"), b"{}").unwrap();
        std::fs::write(dir.join("cot_1.meta.json"), meta).unwrap();

        let result = list_eval_history(&dir, None, 50).unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        let evals = v["evals"].as_array().unwrap();
        assert_eq!(evals.len(), 1, "valid meta must appear in evals");
        let warns = warnings_from_json(&result);
        assert!(
            warns.is_empty(),
            "valid meta must produce no warnings, got {warns:?}"
        );
    }
}
