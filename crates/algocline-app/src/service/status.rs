use super::AppService;

impl AppService {
    /// Snapshot of all active sessions for external observation.
    ///
    /// Returns JSON with session status, metrics, progress, and strategy name.
    /// Only includes sessions currently held in the registry (paused, awaiting
    /// host LLM responses). Completed sessions are not listed here — use
    /// `alc_log_view` for historical data.
    pub async fn status(&self, session_id: Option<&str>) -> Result<String, String> {
        let snapshots = self.registry.list_snapshots().await;

        // If a specific session requested, return just that one
        if let Some(sid) = session_id {
            return match snapshots.get(sid) {
                Some(snapshot) => {
                    let mut result = snapshot.clone();
                    // Enrich with strategy name
                    if let Ok(strategies) = self.session_strategies.lock() {
                        if let Some(name) = strategies.get(sid) {
                            result["strategy"] = serde_json::json!(name);
                        }
                    }
                    result["session_id"] = serde_json::json!(sid);
                    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
                }
                None => Err(format!("session '{sid}' not found (may have completed)")),
            };
        }

        // List all active sessions
        if snapshots.is_empty() {
            return Ok(serde_json::json!({
                "active_sessions": 0,
                "sessions": [],
            })
            .to_string());
        }

        let strategies = self.session_strategies.lock().ok();
        let sessions: Vec<serde_json::Value> = snapshots
            .into_iter()
            .map(|(id, mut snapshot)| {
                if let Some(ref strats) = strategies {
                    if let Some(name) = strats.get(&id) {
                        snapshot["strategy"] = serde_json::json!(name);
                    }
                }
                snapshot["session_id"] = serde_json::json!(id);
                snapshot
            })
            .collect();

        let result = serde_json::json!({
            "active_sessions": sessions.len(),
            "sessions": sessions,
        });

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }
}
