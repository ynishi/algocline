use super::hub_dist_preset::PRESET_CATALOG_VERSION;
use super::path::ContainedPath;
use super::transcript::append_note;
use super::AppService;

impl AppService {
    /// Append a note to a session's log file.
    pub async fn add_note(
        &self,
        session_id: &str,
        content: &str,
        title: Option<&str>,
    ) -> Result<String, String> {
        let count = append_note(self.require_log_dir()?, session_id, content, title)?;
        Ok(serde_json::json!({
            "session_id": session_id,
            "notes_count": count,
        })
        .to_string())
    }

    /// Default max response size for detail mode (100 KB).
    const DEFAULT_MAX_CHARS: usize = 100_000;

    /// View session logs.
    pub async fn log_view(
        &self,
        session_id: Option<&str>,
        limit: Option<usize>,
        max_chars: Option<usize>,
    ) -> Result<String, String> {
        match session_id {
            Some(sid) => self.log_read(sid, max_chars.unwrap_or(Self::DEFAULT_MAX_CHARS)),
            None => self.log_list(limit.unwrap_or(50)),
        }
    }

    fn log_read(&self, session_id: &str, max_chars: usize) -> Result<String, String> {
        let log_dir = self.require_log_dir()?;
        let path = ContainedPath::child(log_dir, &format!("{session_id}.json"))?;
        if !path.as_ref().exists() {
            return Err(format!("Log file not found for session '{session_id}'"));
        }
        let raw = std::fs::read_to_string(&path).map_err(|e| format!("Failed to read log: {e}"))?;

        // 0 means unlimited
        if max_chars == 0 || raw.len() <= max_chars {
            return Ok(raw);
        }

        // Parse and truncate transcript (oldest rounds first) to fit within max_chars
        let mut doc: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("Failed to parse log: {e}"))?;

        let original_rounds = doc
            .get("transcript")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        if original_rounds == 0 {
            // No transcript to truncate; return as-is
            return Ok(raw);
        }

        // Binary-search: keep the maximum number of newest rounds that fit
        let transcript = doc
            .get("transcript")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let mut kept = original_rounds;
        loop {
            if kept == 0 {
                // Even with empty transcript it might still be too large (unlikely)
                doc["transcript"] = serde_json::json!([]);
                break;
            }
            // Keep the newest `kept` rounds
            let slice = &transcript[original_rounds - kept..];
            doc["transcript"] = serde_json::Value::Array(slice.to_vec());
            let serialized =
                serde_json::to_string(&doc).map_err(|e| format!("Failed to serialize: {e}"))?;
            if serialized.len() <= max_chars {
                break;
            }
            // Halve for speed, then linear scan
            if kept > 8 {
                kept /= 2;
            } else {
                kept -= 1;
            }
        }

        let returned_rounds = doc
            .get("transcript")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        doc["truncated"] = serde_json::json!(true);
        doc["original_rounds"] = serde_json::json!(original_rounds);
        doc["returned_rounds"] = serde_json::json!(returned_rounds);

        serde_json::to_string_pretty(&doc).map_err(|e| format!("Failed to serialize: {e}"))
    }

    pub(super) fn log_list(&self, limit: usize) -> Result<String, String> {
        let dir = match self.log_config.log_dir.as_deref() {
            Some(d) if d.is_dir() => d,
            _ => return Ok(serde_json::json!({ "sessions": [] }).to_string()),
        };

        let entries = std::fs::read_dir(dir).map_err(|e| format!("Failed to read log dir: {e}"))?;

        // Collect .meta.json files first; fall back to .json for legacy logs
        let mut files: Vec<(std::path::PathBuf, std::time::SystemTime)> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = path.file_name()?.to_str()?;
                // Skip non-json and meta files in this pass
                if !name.ends_with(".json") || name.ends_with(".meta.json") {
                    return None;
                }
                let mtime = entry.metadata().ok()?.modified().ok()?;
                Some((path, mtime))
            })
            .collect();

        // Sort by modification time descending (newest first), take limit
        files.sort_by(|a, b| b.1.cmp(&a.1));
        files.truncate(limit);

        let mut sessions = Vec::new();
        for (path, _) in &files {
            // Try .meta.json first (lightweight), fall back to full log
            let meta_path = path.with_extension("meta.json");
            let doc: serde_json::Value = if meta_path.exists() {
                // Meta file: already flat summary (~200 bytes)
                match std::fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|r| serde_json::from_str(&r).ok())
                {
                    Some(d) => d,
                    None => continue,
                }
            } else {
                // Legacy fallback: read full log and extract fields
                let raw = match std::fs::read_to_string(path) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                match serde_json::from_str::<serde_json::Value>(&raw) {
                    Ok(d) => {
                        let stats = d.get("stats");
                        serde_json::json!({
                            "session_id": d.get("session_id").and_then(|v| v.as_str()).unwrap_or("unknown"),
                            "task_hint": d.get("task_hint").and_then(|v| v.as_str()),
                            "elapsed_ms": stats.and_then(|s| s.get("elapsed_ms")),
                            "rounds": stats.and_then(|s| s.get("rounds")),
                            "llm_calls": stats.and_then(|s| s.get("llm_calls")),
                            "notes_count": d.get("notes").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                        })
                    }
                    Err(_) => continue,
                }
            };

            sessions.push(doc);
        }

        Ok(serde_json::json!({ "sessions": sessions }).to_string())
    }

    // ─── Stats ──────────────────────────────────────────────────

    /// Return diagnostic info about the current configuration (mise doctor style).
    pub fn info(&self) -> String {
        let mut info = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "preset_catalog_version": PRESET_CATALOG_VERSION,
            "log_dir": {
                "resolved": self.log_config.log_dir.as_ref().map(|p| p.display().to_string()),
                "source": self.log_config.log_dir_source.to_string(),
            },
            "log_enabled": self.log_config.log_enabled,
            "tracing": if self.log_config.log_dir.is_some() { "file + stderr" } else { "stderr only" },
        });

        // search paths (package resolution chain, priority order)
        let search_paths_json: Vec<serde_json::Value> = self
            .search_paths
            .iter()
            .map(|sp| {
                serde_json::json!({
                    "path": sp.path.display().to_string(),
                    "source": sp.source.to_string(),
                })
            })
            .collect();
        info["search_paths"] = serde_json::json!(search_paths_json);

        // packages dir (kept for backward compatibility)
        let packages = self.log_config.app_dir().packages_dir();
        if packages.is_dir() {
            info["packages_dir"] = serde_json::json!(packages.display().to_string());
        }

        serde_json::to_string_pretty(&info).unwrap_or_else(|_| "{}".to_string())
    }

    /// Aggregate stats across all logged sessions.
    ///
    /// Scans `.meta.json` files (with `.json` fallback for legacy logs).
    /// Optional filters: `strategy` (exact match), `days` (last N days).
    ///
    /// # Legacy log compatibility
    ///
    /// Token fields (`prompt_tokens`, `response_tokens`) were introduced in v0.12.
    /// Logs written by earlier versions lack these fields entirely. When absent,
    /// the aggregation treats them as **0** (via `unwrap_or(0)`) — the same
    /// pattern used for other numeric fields (`elapsed_ms`, `total_prompt_chars`,
    /// etc.). This means per-strategy `total_tokens` may under-report if the
    /// dataset includes pre-v0.12 sessions.
    pub fn stats(
        &self,
        strategy_filter: Option<&str>,
        days: Option<u64>,
    ) -> Result<String, String> {
        let dir = match self.log_config.log_dir.as_deref() {
            Some(d) if d.is_dir() => d,
            _ => {
                let card_sinks = algocline_engine::card::subscriber_stats_snapshot();
                return Ok(serde_json::json!({
                    "total_sessions": 0,
                    "strategies": {},
                    "card_sinks": card_sinks,
                })
                .to_string());
            }
        };

        let cutoff = days.map(|d| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
                - d * 86_400_000
        });

        let entries = std::fs::read_dir(dir).map_err(|e| format!("Failed to read log dir: {e}"))?;

        #[derive(Default)]
        struct StrategyAcc {
            count: u64,
            sum_elapsed_ms: u64,
            sum_llm_calls: u64,
            sum_rounds: u64,
            sum_prompt_chars: u64,
            sum_response_chars: u64,
            sum_prompt_tokens: u64,
            sum_response_tokens: u64,
        }

        let mut acc: std::collections::HashMap<String, StrategyAcc> =
            std::collections::HashMap::new();
        let mut total: u64 = 0;

        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Read meta from .meta.json or fall back to .json
            let doc: serde_json::Value = if name.ends_with(".meta.json") {
                match std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|r| serde_json::from_str(&r).ok())
                {
                    Some(d) => d,
                    None => continue,
                }
            } else if name.ends_with(".json") && !name.ends_with(".meta.json") {
                // Skip full logs if meta exists
                let meta_name =
                    format!("{}.meta.json", name.strip_suffix(".json").unwrap_or(&name));
                let meta_path = dir.join(meta_name);
                if meta_path.exists() {
                    continue;
                }
                // Legacy fallback
                match std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
                {
                    Some(d) => {
                        let stats = d.get("stats");
                        serde_json::json!({
                            "strategy": d.get("strategy").and_then(|v| v.as_str()),
                            "elapsed_ms": stats.and_then(|s| s.get("elapsed_ms")),
                            "llm_calls": stats.and_then(|s| s.get("llm_calls")),
                            "rounds": stats.and_then(|s| s.get("rounds")),
                            "total_prompt_chars": stats.and_then(|s| s.get("total_prompt_chars")),
                            "total_response_chars": stats.and_then(|s| s.get("total_response_chars")),
                        })
                    }
                    None => continue,
                }
            } else {
                continue;
            };

            // Apply time filter via elapsed_ms proxy (file mtime would be better but
            // meta files don't store timestamps; use mtime as approximation)
            if let Some(cutoff_ms) = cutoff {
                let mtime = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                if mtime < cutoff_ms {
                    continue;
                }
            }

            let strat = doc
                .get("strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            // Apply strategy filter
            if let Some(filter) = strategy_filter {
                if strat != filter {
                    continue;
                }
            }

            let elapsed = doc.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            let llm = doc.get("llm_calls").and_then(|v| v.as_u64()).unwrap_or(0);
            let rounds = doc.get("rounds").and_then(|v| v.as_u64()).unwrap_or(0);
            let prompt_chars = doc
                .get("total_prompt_chars")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let response_chars = doc
                .get("total_response_chars")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            // Token counts: nested {"tokens": N, "source": "..."} or legacy absent
            let prompt_tokens = doc
                .get("prompt_tokens")
                .and_then(|v| v.get("tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let response_tokens = doc
                .get("response_tokens")
                .and_then(|v| v.get("tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            let a = acc.entry(strat).or_default();
            a.count += 1;
            a.sum_elapsed_ms += elapsed;
            a.sum_llm_calls += llm;
            a.sum_rounds += rounds;
            a.sum_prompt_chars += prompt_chars;
            a.sum_response_chars += response_chars;
            a.sum_prompt_tokens += prompt_tokens;
            a.sum_response_tokens += response_tokens;
            total += 1;
        }

        // Build response
        let mut strategies = serde_json::Map::new();
        for (strat, a) in &acc {
            let c = a.count.max(1); // avoid division by zero
            strategies.insert(
                strat.clone(),
                serde_json::json!({
                    "count": a.count,
                    "avg_elapsed_ms": (a.sum_elapsed_ms + c / 2) / c,
                    "avg_llm_calls": (a.sum_llm_calls + c / 2) / c,
                    "avg_rounds": (a.sum_rounds + c / 2) / c,
                    "total_prompt_chars": a.sum_prompt_chars,
                    "total_response_chars": a.sum_response_chars,
                    "total_prompt_tokens": a.sum_prompt_tokens,
                    "total_response_tokens": a.sum_response_tokens,
                    "total_tokens": a.sum_prompt_tokens + a.sum_response_tokens,
                }),
            );
        }

        let card_sinks = algocline_engine::card::subscriber_stats_snapshot();
        Ok(serde_json::json!({
            "total_sessions": total,
            "strategies": strategies,
            "card_sinks": card_sinks,
        })
        .to_string())
    }
}
