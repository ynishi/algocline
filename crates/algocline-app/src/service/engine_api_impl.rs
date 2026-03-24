use algocline_core::{EngineApi, QueryResponse};
use async_trait::async_trait;

use super::AppService;

#[async_trait]
impl EngineApi for AppService {
    // ─── Core execution ──────────────────────────────────────

    async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
    ) -> Result<String, String> {
        self.run(code, code_file, ctx).await
    }

    async fn advice(
        &self,
        strategy: &str,
        task: String,
        opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        self.advice(strategy, task, opts).await
    }

    async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
    ) -> Result<String, String> {
        self.continue_single(session_id, response, query_id).await
    }

    async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        self.continue_batch(session_id, responses).await
    }

    // ─── Session status ──────────────────────────────────────

    async fn status(&self, session_id: Option<&str>) -> Result<String, String> {
        self.status(session_id).await
    }

    // ─── Evaluation ──────────────────────────────────────────

    async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        scenario_name: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        self.eval(
            scenario,
            scenario_file,
            scenario_name,
            strategy,
            strategy_opts,
        )
        .await
    }

    async fn eval_history(&self, strategy: Option<&str>, limit: usize) -> Result<String, String> {
        self.eval_history(strategy, limit)
    }

    async fn eval_detail(&self, eval_id: &str) -> Result<String, String> {
        self.eval_detail(eval_id)
    }

    async fn eval_compare(&self, eval_id_a: &str, eval_id_b: &str) -> Result<String, String> {
        self.eval_compare(eval_id_a, eval_id_b).await
    }

    // ─── Scenarios ───────────────────────────────────────────

    async fn scenario_list(&self) -> Result<String, String> {
        self.scenario_list()
    }

    async fn scenario_show(&self, name: &str) -> Result<String, String> {
        self.scenario_show(name)
    }

    async fn scenario_install(&self, url: String) -> Result<String, String> {
        self.scenario_install(url).await
    }

    // ─── Packages ────────────────────────────────────────────

    async fn pkg_list(&self) -> Result<String, String> {
        self.pkg_list().await
    }

    async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        self.pkg_install(url, name).await
    }

    async fn pkg_remove(&self, name: &str) -> Result<String, String> {
        self.pkg_remove(name).await
    }

    // ─── Logging ─────────────────────────────────────────────

    async fn add_note(
        &self,
        session_id: &str,
        content: &str,
        title: Option<&str>,
    ) -> Result<String, String> {
        self.add_note(session_id, content, title).await
    }

    async fn log_view(
        &self,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        self.log_view(session_id, limit).await
    }

    async fn stats(
        &self,
        strategy_filter: Option<&str>,
        days: Option<u64>,
    ) -> Result<String, String> {
        self.stats(strategy_filter, days)
    }

    // ─── Diagnostics ─────────────────────────────────────────

    async fn info(&self) -> String {
        self.info()
    }
}
