use algocline_engine::PendingFilter;

use super::AppService;

impl AppService {
    /// Snapshot of all active sessions (or one by ID) for external observation.
    ///
    /// # Arguments
    ///
    /// * `session_id` - When `Some`, returns detail for one session; when `None`, lists all.
    /// * `pending_filter` - Optional preset name or custom field-filter for pending query projection.
    /// * `include_history` - When `true`, each snapshot includes `conversation_history` (cap=10).
    ///   Pass `false` (the default) for lightweight high-frequency polling snapshots.
    ///
    /// # Returns
    ///
    /// JSON string with either a single session object or `{active_sessions, sessions}` list.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `pending_filter` is an unknown preset name or an invalid shape.
    pub async fn status(
        &self,
        session_id: Option<&str>,
        pending_filter: Option<serde_json::Value>,
        include_history: bool,
    ) -> Result<String, String> {
        let filter = self.resolve_pending_filter(pending_filter)?;
        let snapshots = self
            .registry
            .list_snapshots(filter.as_ref(), include_history)
            .await;

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

    /// Decode the incoming `pending_filter` JSON value into an optional
    /// `PendingFilter`. Preset strings read the per-request char count
    /// from this service's `AppConfig`; custom objects use the values
    /// declared by the caller.
    fn resolve_pending_filter(
        &self,
        raw: Option<serde_json::Value>,
    ) -> Result<Option<PendingFilter>, String> {
        let Some(value) = raw else {
            return Ok(None);
        };
        match value {
            serde_json::Value::String(name) => PendingFilter::from_preset_with(
                &name,
                self.log_config.prompt_preview_chars,
            )
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "unknown pending_filter preset '{name}' (valid: \"meta\" | \"preview\" | \"full\")"
                )
            }),
            serde_json::Value::Object(_) => serde_json::from_value::<PendingFilter>(value)
                .map(Some)
                .map_err(|e| format!("invalid pending_filter object: {e}")),
            other => Err(format!(
                "pending_filter must be a preset name (string) or filter object, got {}",
                match other {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Array(_) => "array",
                    _ => "unknown",
                }
            )),
        }
    }
}
