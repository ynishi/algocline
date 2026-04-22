use std::sync::Arc;

use algocline_core::QueryId;
use algocline_engine::{FeedResult, VariantPkg};

use super::resolve::{is_package_installed, make_require_code, resolve_code, QueryResponse};
use super::transcript::write_transcript_log;
use super::AppService;

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
        let info = {
            let mut map = match self.eval_sessions.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            map.remove(session_id)
        };
        if let Some(strategy) = info {
            super::eval_store::save_eval_result(&self.log_config.app_dir(), &strategy, result_json);
        }
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
        self.maybe_log_transcript(&result, &session_id);
        Ok(result.to_json(&session_id).to_string())
    }
}
