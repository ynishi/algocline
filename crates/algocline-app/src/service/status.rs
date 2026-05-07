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
            if let Some(snapshot) = snapshots.get(sid) {
                let mut result = snapshot.clone();
                // Enrich with strategy name
                if let Ok(strategies) = self.session_strategies.lock() {
                    if let Some(name) = strategies.get(sid) {
                        result["strategy"] = serde_json::json!(name);
                    }
                }
                result["session_id"] = serde_json::json!(sid);
                return serde_json::to_string_pretty(&result).map_err(|e| e.to_string());
            }
            // Pool fallback: host_mode=true sessions live in pool_registry,
            // not SessionRegistry. Surface them as needs_response with a
            // `pool: true` marker so callers can distinguish backends.
            // include_history is ignored on this path — pool worker history
            // requires a separate IPC round-trip and is out of scope here.
            let pool_reg = self.pool_registry.read().await;
            if let Some(entry) = pool_reg.find(sid) {
                let result = serde_json::json!({
                    "status": "needs_response",
                    "session_id": sid,
                    "pool": true,
                    "pid": entry.pid,
                    "sock": entry.sock.to_string_lossy(),
                    "version": entry.version,
                    "created_at": entry.created_at,
                });
                return serde_json::to_string_pretty(&result).map_err(|e| e.to_string());
            }
            return Err(format!("session '{sid}' not found (may have completed)"));
        }

        // List all active sessions — merge SessionRegistry snapshots with
        // pool_registry live entries. Pool entries are surfaced in the same
        // shape used by the single-session fallback (see above), with a
        // `pool: true` marker. SessionRegistry takes precedence on sid
        // collision (defensive — host_mode design avoids collisions).
        // include_history is ignored on the pool path; per-session
        // conversation_history fetch over IPC is out of scope here.
        let pool_reg = self.pool_registry.read().await;
        if snapshots.is_empty() && pool_reg.sessions.is_empty() {
            return Ok(serde_json::json!({
                "active_sessions": 0,
                "sessions": [],
            })
            .to_string());
        }

        let mut sessions: Vec<serde_json::Value> = {
            let strategies = self.session_strategies.lock().ok();
            snapshots
                .iter()
                .map(|(id, snapshot)| {
                    let mut snap = snapshot.clone();
                    if let Some(ref strats) = strategies {
                        if let Some(name) = strats.get(id) {
                            snap["strategy"] = serde_json::json!(name);
                        }
                    }
                    snap["session_id"] = serde_json::json!(id);
                    snap
                })
                .collect()
        };

        for entry in pool_reg.sessions.iter() {
            if snapshots.contains_key(&entry.sid) {
                continue;
            }
            sessions.push(serde_json::json!({
                "status": "needs_response",
                "session_id": entry.sid,
                "pool": true,
                "pid": entry.pid,
                "sock": entry.sock.to_string_lossy(),
                "version": entry.version,
                "created_at": entry.created_at,
            }));
        }
        drop(pool_reg);

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
