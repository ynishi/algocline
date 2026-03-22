use algocline_core::QueryId;
use algocline_engine::FeedResult;

use super::resolve::{is_package_installed, make_require_code, resolve_code, QueryResponse};
use super::transcript::write_transcript_log;
use super::AppService;

impl AppService {
    /// Execute Lua code with optional JSON context.
    pub async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let code = resolve_code(code, code_file)?;
        let ctx = ctx.unwrap_or(serde_json::Value::Null);
        self.start_and_tick(code, ctx, None).await
    }

    /// Apply a built-in strategy to a task.
    ///
    /// If the requested package is not installed, automatically installs the
    /// bundled package collection from GitHub before executing.
    pub async fn advice(
        &self,
        strategy: &str,
        task: String,
        opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        // Auto-install bundled packages if the requested strategy is missing
        if !is_package_installed(strategy) {
            self.auto_install_bundled_packages().await?;
            if !is_package_installed(strategy) {
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
        ctx_map.insert("task".into(), serde_json::Value::String(task));
        let ctx = serde_json::Value::Object(ctx_map);

        self.start_and_tick(code, ctx, Some(strategy)).await
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
                .feed_response(session_id, &qid, qr.response)
                .await
                .map_err(|e| format!("Continue failed: {e}"))?;
            last_result = Some(result);
        }
        let result = last_result.ok_or("Empty responses array")?;
        self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        self.maybe_save_eval(&result, session_id, &json);
        Ok(json)
    }

    /// Continue a paused execution — single response (with optional query_id).
    pub async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
    ) -> Result<String, String> {
        let query_id = match query_id {
            Some(qid) => QueryId::parse(qid),
            None => QueryId::single(),
        };

        let result = self
            .registry
            .feed_response(session_id, &query_id, response)
            .await
            .map_err(|e| format!("Continue failed: {e}"))?;

        self.maybe_log_transcript(&result, session_id);
        let json = result.to_json(session_id).to_string();
        self.maybe_save_eval(&result, session_id, &json);
        Ok(json)
    }

    // ─── Internal ───────────────────────────────────────────────

    pub(super) fn maybe_log_transcript(&self, result: &FeedResult, session_id: &str) {
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
            );
        }
    }

    pub(super) fn maybe_save_eval(&self, result: &FeedResult, session_id: &str, result_json: &str) {
        if !matches!(result, FeedResult::Finished(_)) {
            return;
        }
        let strategy = {
            let mut map = match self.eval_sessions.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            map.remove(session_id)
        };
        if let Some(strategy) = strategy {
            super::eval_store::save_eval_result(&strategy, result_json);
        }
    }

    pub(super) async fn start_and_tick(
        &self,
        code: String,
        ctx: serde_json::Value,
        strategy: Option<&str>,
    ) -> Result<String, String> {
        let session = self.executor.start_session(code, ctx).await?;
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
        self.maybe_log_transcript(&result, &session_id);
        Ok(result.to_json(&session_id).to_string())
    }
}
