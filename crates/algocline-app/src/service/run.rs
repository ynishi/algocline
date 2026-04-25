use std::sync::Arc;

use algocline_core::QueryId;
use algocline_engine::{FeedResult, VariantPkg};

use super::eval_store::splice_response_string;
use super::resolve::{is_package_installed, make_require_code, resolve_code, QueryResponse};
use super::transcript::write_transcript_log;
use super::AppService;

/// Splice `save_warning` into the JSON `result` when the optional
/// warning is `Some(_)`. Returns the original string unchanged when
/// there is no warning.
fn splice_save_warning(result_json: &str, warning: Option<String>) -> String {
    match warning {
        Some(msg) => splice_response_string(result_json, "save_warning", &msg),
        None => result_json.to_string(),
    }
}

/// Splice `transcript_warning` into the JSON `result` when the optional
/// warning is `Some(_)`. Returns the original string unchanged when
/// there is no warning.
fn splice_transcript_warning(result_json: &str, warning: Option<String>) -> String {
    match warning {
        Some(msg) => splice_response_string(result_json, "transcript_warning", &msg),
        None => result_json.to_string(),
    }
}

impl AppService {
    /// Execute Lua code with optional JSON context.
    ///
    /// `project_root` — optional absolute path to the project root containing
    /// `alc.lock`. Falls back to `ALC_PROJECT_ROOT` env or ancestor walk.
    pub async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        let code = resolve_code(code, code_file)?;
        let ctx = ctx.unwrap_or(serde_json::Value::Null);
        let extra = self.resolve_extra_lib_paths(project_root.as_deref());
        let variants = self.resolve_variant_pkgs(project_root.as_deref());
        self.start_and_tick(code, ctx, None, extra, variants).await
    }

    /// Apply a built-in strategy to a task.
    ///
    /// If the requested package is not installed, automatically installs the
    /// bundled package collection from GitHub before executing.
    ///
    /// `project_root` — optional absolute path to the project root containing
    /// `alc.lock`. Falls back to `ALC_PROJECT_ROOT` env or ancestor walk.
    pub async fn advice(
        &self,
        strategy: &str,
        task: Option<String>,
        opts: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        // Auto-install bundled packages if the requested strategy is missing
        let app_dir = self.log_config.app_dir();
        if !is_package_installed(&app_dir, strategy) {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed(&app_dir, strategy) {
                return Err(format!(
                    "Package '{strategy}' not found after installing bundled collection. \
                     Use alc_pkg_install to install it manually."
                ));
            }
        }

        let code = make_require_code(strategy);

        let mut ctx_map = match opts {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        if let Some(task) = task {
            ctx_map.insert("task".into(), serde_json::Value::String(task));
        }
        let ctx = serde_json::Value::Object(ctx_map);

        let extra = self.resolve_extra_lib_paths(project_root.as_deref());
        let variants = self.resolve_variant_pkgs(project_root.as_deref());
        self.start_and_tick(code, ctx, Some(strategy), extra, variants)
            .await
    }

    /// Continue a paused execution — batch feed.
    pub async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        let mut last_result = None;
        for qr in responses {
            let qid = QueryId::parse(&qr.query_id);
            let result = self
                .registry
                .feed_response(session_id, &qid, qr.response, qr.usage.as_ref())
                .await
                .map_err(|e| format!("Continue failed: {e}"))?;
            last_result = Some(result);
        }
        let result = last_result.ok_or("Empty responses array")?;
        let transcript_warning = self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        let json = splice_transcript_warning(&json, transcript_warning);
        let save_warning = self.maybe_save_eval(&result, session_id, &json);
        Ok(splice_save_warning(&json, save_warning))
    }

    /// Continue a paused execution — single response (with optional query_id).
    pub async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
        usage: Option<algocline_core::TokenUsage>,
    ) -> Result<String, String> {
        let query_id = match query_id {
            Some(qid) => QueryId::parse(qid),
            None => self
                .registry
                .resolve_sole_pending_id(session_id)
                .await
                .map_err(|e| format!("Continue failed: {e}"))?,
        };

        let result = self
            .registry
            .feed_response(session_id, &query_id, response, usage.as_ref())
            .await
            .map_err(|e| format!("Continue failed: {e}"))?;

        let transcript_warning = self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        let json = splice_transcript_warning(&json, transcript_warning);
        let save_warning = self.maybe_save_eval(&result, session_id, &json);
        Ok(splice_save_warning(&json, save_warning))
    }

    // ─── Internal ───────────────────────────────────────────────

    pub(super) fn maybe_log_transcript(
        &self,
        result: &FeedResult,
        session_id: &str,
    ) -> Option<String> {
        if let FeedResult::Finished(exec_result) = result {
            let strategy = self
                .session_strategies
                .lock()
                .ok()
                .and_then(|mut map| map.remove(session_id));
            write_transcript_log(
                &self.log_config,
                session_id,
                &exec_result.metrics,
                strategy.as_deref(),
            )
            .err()
            .map(|e| e.to_string())
        } else {
            None
        }
    }

    /// Persist eval result for a finished session, returning any storage
    /// failure as `Some(msg)` so the caller can surface it on the wire
    /// response. `None` covers both "not an eval session" and
    /// "successfully saved" — they are indistinguishable to the caller
    /// because both produce the same wire shape.
    pub(super) fn maybe_save_eval(
        &self,
        result: &FeedResult,
        session_id: &str,
        result_json: &str,
    ) -> Option<String> {
        if !matches!(result, FeedResult::Finished(_)) {
            return None;
        }
        let strategy = {
            let mut map = self.eval_sessions.lock().unwrap_or_else(|e| e.into_inner());
            map.remove(session_id)
        };
        strategy.and_then(|s| {
            super::eval_store::save_eval_result(&self.log_config.app_dir(), &s, result_json).err()
        })
    }

    pub(super) async fn start_and_tick(
        &self,
        code: String,
        ctx: serde_json::Value,
        strategy: Option<&str>,
        extra_lib_paths: Vec<std::path::PathBuf>,
        variant_pkgs: Vec<VariantPkg>,
    ) -> Result<String, String> {
        let scenarios_dir = self.log_config.app_dir().scenarios_dir();
        let session = self
            .executor
            .start_session(
                code,
                ctx,
                extra_lib_paths,
                variant_pkgs,
                Arc::clone(&self.state_store),
                Arc::clone(&self.card_store),
                scenarios_dir,
            )
            .await?;
        let (session_id, result) = self
            .registry
            .start_execution(session)
            .await
            .map_err(|e| format!("Execution failed: {e}"))?;
        if let Some(s) = strategy {
            if let Ok(mut map) = self.session_strategies.lock() {
                map.insert(session_id.clone(), s.to_string());
            }
        }
        let transcript_warning = self.maybe_log_transcript(&result, &session_id);
        let json = result.to_json(&session_id).to_string();
        Ok(splice_transcript_warning(&json, transcript_warning))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use algocline_core::{
        AppDir, ExecutionMetrics, ExecutionObserver, LlmQuery, QueryId, TerminalState,
    };
    use algocline_engine::{ExecutionResult, FeedResult};

    use super::super::config::{AppConfig, LogDirSource};
    use super::{splice_transcript_warning, AppService};

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

    fn make_finished_result(metrics: ExecutionMetrics) -> FeedResult {
        FeedResult::Finished(ExecutionResult {
            state: TerminalState::Completed {
                result: serde_json::json!({"ok": true}),
            },
            metrics,
        })
    }

    /// Build a minimal AppService with log_enabled and a custom log_dir.
    async fn make_app_service_with_log_dir(log_dir: PathBuf) -> AppService {
        let executor = Arc::new(
            algocline_engine::Executor::new(vec![])
                .await
                .expect("executor"),
        );
        let tmp_app = tempfile::tempdir().expect("test tempdir");
        let log_config = AppConfig {
            log_dir: Some(log_dir),
            log_dir_source: LogDirSource::EnvVar,
            log_enabled: true,
            prompt_preview_chars: 200,
            app_dir: Arc::new(AppDir::new(tmp_app.path().to_path_buf())),
        };
        std::mem::forget(tmp_app);
        AppService::new(executor, log_config, vec![])
    }

    // ── (b) maybe_log_transcript returns Some when write fails ──────────

    #[tokio::test]
    async fn maybe_log_transcript_returns_some_on_write_failure() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let log_dir = tmp.path().to_path_buf();
        // Block write by creating a directory at the session file path.
        std::fs::create_dir_all(log_dir.join("fail-session.json"))
            .expect("pre-create dir to block write");
        let svc = make_app_service_with_log_dir(log_dir).await;
        let metrics = make_metrics_with_transcript();
        let result = make_finished_result(metrics);
        let warning = svc.maybe_log_transcript(&result, "fail-session");
        assert!(warning.is_some(), "expected Some warning on write failure");
        let msg = warning.unwrap();
        assert!(
            msg.contains("transcript"),
            "warning should mention 'transcript', got: {msg}"
        );
    }

    #[tokio::test]
    async fn maybe_log_transcript_returns_none_on_non_finished() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;
        let result = FeedResult::Accepted { remaining: 1 };
        let warning = svc.maybe_log_transcript(&result, "any-session");
        assert!(warning.is_none(), "Accepted result should return None");
    }

    // ── (c) splice_transcript_warning inserts field into JSON ───────────

    #[test]
    fn splice_transcript_warning_injects_field_when_some() {
        let json = r#"{"status":"finished","result":{}}"#;
        let out = splice_transcript_warning(json, Some("write failed".to_string()));
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert_eq!(
            v["transcript_warning"].as_str(),
            Some("write failed"),
            "transcript_warning field should be present"
        );
        // Original fields are preserved.
        assert_eq!(v["status"].as_str(), Some("finished"));
    }

    #[test]
    fn splice_transcript_warning_passthrough_when_none() {
        let json = r#"{"status":"finished"}"#;
        let out = splice_transcript_warning(json, None);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(
            v.get("transcript_warning").is_none(),
            "transcript_warning must be absent when warning is None"
        );
    }
}
