use algocline_core::{EngineApi, QueryResponse};
use async_trait::async_trait;

use super::AppService;

/// Delegates each [`EngineApi`] method to the corresponding `AppService`
/// inherent method via fully-qualified syntax (`AppService::method(self, …)`).
///
/// This avoids ambiguity between the trait method and the inherent method
/// of the same name, preventing accidental infinite recursion if the
/// inherent method is ever removed or renamed.
#[async_trait]
impl EngineApi for AppService {
    // ─── Core execution ──────────────────────────────────────

    async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        AppService::run(self, code, code_file, ctx, project_root).await
    }

    async fn advice(
        &self,
        strategy: &str,
        task: Option<String>,
        opts: Option<serde_json::Value>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        AppService::advice(self, strategy, task, opts, project_root).await
    }

    async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
        usage: Option<algocline_core::TokenUsage>,
    ) -> Result<String, String> {
        AppService::continue_single(self, session_id, response, query_id, usage).await
    }

    async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        AppService::continue_batch(self, session_id, responses).await
    }

    // ─── Session status ──────────────────────────────────────

    async fn status(&self, session_id: Option<&str>) -> Result<String, String> {
        AppService::status(self, session_id).await
    }

    // ─── Evaluation ──────────────────────────────────────────

    async fn eval(
        &self,
        scenario: Option<String>,
        scenario_file: Option<String>,
        scenario_name: Option<String>,
        strategy: &str,
        strategy_opts: Option<serde_json::Value>,
        auto_card: bool,
    ) -> Result<String, String> {
        AppService::eval(
            self,
            scenario,
            scenario_file,
            scenario_name,
            strategy,
            strategy_opts,
            auto_card,
        )
        .await
    }

    async fn eval_history(&self, strategy: Option<&str>, limit: usize) -> Result<String, String> {
        AppService::eval_history(self, strategy, limit)
    }

    async fn eval_detail(&self, eval_id: &str) -> Result<String, String> {
        AppService::eval_detail(self, eval_id)
    }

    async fn eval_compare(&self, eval_id_a: &str, eval_id_b: &str) -> Result<String, String> {
        AppService::eval_compare(self, eval_id_a, eval_id_b).await
    }

    // ─── Scenarios ───────────────────────────────────────────

    async fn scenario_list(&self) -> Result<String, String> {
        AppService::scenario_list(self)
    }

    async fn scenario_show(&self, name: &str) -> Result<String, String> {
        AppService::scenario_show(self, name)
    }

    async fn scenario_install(&self, url: String) -> Result<String, String> {
        AppService::scenario_install(self, url).await
    }

    // ─── Packages ────────────────────────────────────────────

    async fn pkg_link(
        &self,
        path: String,
        name: Option<String>,
        force: Option<bool>,
    ) -> Result<String, String> {
        AppService::pkg_link(self, path, name, force).await
    }

    async fn pkg_unlink(&self, name: String) -> Result<String, String> {
        AppService::pkg_unlink(self, name).await
    }

    async fn pkg_list(&self, project_root: Option<String>) -> Result<String, String> {
        AppService::pkg_list(self, project_root).await
    }

    async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        AppService::pkg_install(self, url, name).await
    }

    async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        version: Option<String>,
    ) -> Result<String, String> {
        AppService::pkg_remove(self, name, project_root, version).await
    }

    // ─── Logging ─────────────────────────────────────────────

    async fn add_note(
        &self,
        session_id: &str,
        content: &str,
        title: Option<&str>,
    ) -> Result<String, String> {
        AppService::add_note(self, session_id, content, title).await
    }

    async fn log_view(
        &self,
        session_id: Option<&str>,
        limit: Option<usize>,
        max_chars: Option<usize>,
    ) -> Result<String, String> {
        AppService::log_view(self, session_id, limit, max_chars).await
    }

    async fn stats(
        &self,
        strategy_filter: Option<&str>,
        days: Option<u64>,
    ) -> Result<String, String> {
        AppService::stats(self, strategy_filter, days)
    }

    // ─── Project lifecycle ────────────────────────────────────

    async fn init(&self, project_root: Option<String>) -> Result<String, String> {
        AppService::init(self, project_root).await
    }

    async fn update(&self, project_root: Option<String>) -> Result<String, String> {
        AppService::update(self, project_root).await
    }

    async fn migrate(&self, project_root: Option<String>) -> Result<String, String> {
        AppService::migrate(self, project_root).await
    }

    // ─── Cards ───────────────────────────────────────────────

    async fn card_list(&self, pkg: Option<String>) -> Result<String, String> {
        AppService::card_list(self, pkg.as_deref())
    }

    async fn card_get(&self, card_id: &str) -> Result<String, String> {
        AppService::card_get(self, card_id)
    }

    async fn card_find(
        &self,
        pkg: Option<String>,
        scenario: Option<String>,
        model: Option<String>,
        sort: Option<String>,
        limit: Option<usize>,
        min_pass_rate: Option<f64>,
    ) -> Result<String, String> {
        AppService::card_find(self, pkg, scenario, model, sort, limit, min_pass_rate)
    }

    async fn card_alias_list(&self, pkg: Option<String>) -> Result<String, String> {
        AppService::card_alias_list(self, pkg.as_deref())
    }

    async fn card_get_by_alias(&self, name: &str) -> Result<String, String> {
        AppService::card_get_by_alias(self, name)
    }

    async fn card_alias_set(
        &self,
        name: &str,
        card_id: &str,
        pkg: Option<String>,
        note: Option<String>,
    ) -> Result<String, String> {
        AppService::card_alias_set(self, name, card_id, pkg.as_deref(), note.as_deref())
    }

    async fn card_append(
        &self,
        card_id: &str,
        fields: serde_json::Value,
    ) -> Result<String, String> {
        AppService::card_append(self, card_id, fields)
    }

    async fn card_install(&self, url: String) -> Result<String, String> {
        AppService::card_install(self, url).await
    }

    async fn card_samples(
        &self,
        card_id: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, String> {
        AppService::card_samples(self, card_id, offset.unwrap_or(0), limit)
    }

    // ─── Diagnostics ─────────────────────────────────────────

    async fn info(&self) -> String {
        AppService::info(self)
    }
}
