use std::collections::HashMap;

use algocline_core::{EngineApi, QueryResponse};
use async_trait::async_trait;

use crate::pool::{registry::with_registry_lock, PoolError, PoolRegistry};

use super::list_opts::ListOpts;
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
        host_mode: Option<bool>,
    ) -> Result<String, String> {
        AppService::run(self, code, code_file, ctx, project_root, host_mode).await
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

    async fn status(
        &self,
        session_id: Option<&str>,
        pending_filter: Option<serde_json::Value>,
        include_history: bool,
    ) -> Result<String, String> {
        AppService::status(self, session_id, pending_filter, include_history).await
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
        scope: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        AppService::pkg_link(self, path, name, force, scope, project_root).await
    }

    async fn pkg_unlink(&self, name: String) -> Result<String, String> {
        AppService::pkg_unlink(self, name).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn pkg_list(
        &self,
        project_root: Option<String>,
        limit: Option<i32>,
        sort: Option<String>,
        filter: Option<serde_json::Value>,
        fields: Option<Vec<String>>,
        verbose: Option<String>,
    ) -> Result<String, String> {
        // `filter` is a free-form JSON Value at the MCP boundary (so the
        // trait stays core-crate-pure). If the caller sends something
        // that is not a JSON object we treat it as "no filter" and log
        // the drop so operators can diagnose unexpected filter shapes
        // in production.
        let filter_map = match filter {
            None => None,
            Some(v) => match serde_json::from_value::<HashMap<String, serde_json::Value>>(v) {
                Ok(map) => Some(map),
                Err(e) => {
                    tracing::warn!(error = %e, "pkg_list: filter value is not a JSON object — treating as no filter");
                    None
                }
            },
        };

        // Negative limit values from MCP callers are clamped to 0 rather
        // than wrapping to a huge usize (unchecked-user-bound-input pattern).
        // Downstream semantics: `Some(0)` means "no limit" (return all) —
        // the truncate path in `AppService::pkg_list` short-circuits on 0.
        let opts = ListOpts {
            limit: limit.map(|n| n.max(0) as usize),
            sort,
            filter: filter_map,
            fields,
            verbose,
        };

        AppService::pkg_list(self, project_root, opts)
            .await
            .map_err(|e| e.to_string())
    }

    async fn pkg_install(
        &self,
        url: String,
        name: Option<String>,
        force: Option<bool>,
    ) -> Result<String, String> {
        AppService::pkg_install(self, url, name, force).await
    }

    async fn pkg_remove(
        &self,
        name: &str,
        project_root: Option<String>,
        version: Option<String>,
        scope: Option<String>,
    ) -> Result<String, String> {
        AppService::pkg_remove(self, name, project_root, version, scope).await
    }

    async fn pkg_repair(
        &self,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        AppService::pkg_repair(self, name, project_root).await
    }

    async fn pkg_doctor(
        &self,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        AppService::pkg_doctor(self, name, project_root).await
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

    // ─── Session activation (issue #1776627475) ──────────────

    async fn session_new(
        &self,
        project_root: Option<String>,
        mode: Option<String>,
    ) -> Result<String, String> {
        let session = self.activate_session(project_root.as_deref(), mode.as_deref())?;
        let result = serde_json::json!({
            "session_id": session.session_id,
            "project_root": session
                .project_root
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            "mode": session.mode.as_str(),
        });
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
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
        where_: Option<serde_json::Value>,
        order_by: Option<serde_json::Value>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<String, String> {
        AppService::card_find(self, pkg, where_, order_by, limit, offset)
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
        where_: Option<serde_json::Value>,
    ) -> Result<String, String> {
        AppService::card_samples(self, card_id, offset.unwrap_or(0), limit, where_)
    }

    async fn card_lineage(
        &self,
        card_id: &str,
        direction: Option<String>,
        depth: Option<usize>,
        include_stats: Option<bool>,
        relation_filter: Option<Vec<String>>,
    ) -> Result<String, String> {
        AppService::card_lineage(
            self,
            card_id,
            direction.as_deref(),
            depth,
            include_stats,
            relation_filter,
        )
    }

    async fn card_sink_backfill(&self, sink: String, dry_run: bool) -> Result<String, String> {
        AppService::card_sink_backfill(self, super::card::SinkBackfillParams { sink, dry_run })
    }

    // ─── Hub ─────────────────────────────────────────────────

    async fn hub_reindex(
        &self,
        output_path: Option<String>,
        source_dir: Option<String>,
    ) -> Result<String, String> {
        let svc = self.clone();
        tokio::task::spawn_blocking(move || {
            AppService::hub_reindex(&svc, output_path.as_deref(), source_dir.as_deref())
        })
        .await
        .map_err(|e| format!("hub_reindex task panicked: {e}"))?
    }

    async fn hub_gendoc(
        &self,
        source_dir: String,
        out_dir: Option<String>,
        projections: Option<Vec<String>>,
        config_path: Option<String>,
        lint_strict: Option<bool>,
    ) -> Result<String, String> {
        let svc = self.clone();
        tokio::task::spawn_blocking(move || {
            crate::AppService::hub_gendoc(
                &svc,
                &source_dir,
                out_dir.as_deref(),
                projections.as_deref(),
                config_path.as_deref(),
                lint_strict,
            )
        })
        .await
        .map_err(|e| format!("hub_gendoc task panicked: {e}"))?
    }

    async fn hub_dist(
        &self,
        source_dir: String,
        output_path: Option<String>,
        out_dir: Option<String>,
        preset: Option<String>,
        project_root: Option<String>,
        projections: Option<Vec<String>>,
        config_path: Option<String>,
        lint_strict: Option<bool>,
    ) -> Result<String, String> {
        let svc = self.clone();
        tokio::task::spawn_blocking(move || {
            AppService::hub_dist(
                &svc,
                &source_dir,
                output_path.as_deref(),
                out_dir.as_deref(),
                preset.as_deref(),
                project_root.as_deref(),
                projections.as_deref(),
                config_path.as_deref(),
                lint_strict,
            )
        })
        .await
        .map_err(|e| format!("hub_dist task panicked: {e}"))?
    }

    async fn hub_info(&self, pkg: String) -> Result<String, String> {
        let svc = self.clone();
        tokio::task::spawn_blocking(move || AppService::hub_info(&svc, &pkg))
            .await
            .map_err(|e| format!("hub_info task panicked: {e}"))?
    }

    #[allow(clippy::too_many_arguments)]
    async fn hub_search(
        &self,
        query: Option<String>,
        category: Option<String>,
        installed_only: Option<bool>,
        limit: Option<i32>,
        sort: Option<String>,
        filter: Option<serde_json::Value>,
        fields: Option<Vec<String>>,
        verbose: Option<String>,
    ) -> Result<String, String> {
        let svc = self.clone();

        // `filter` is a free-form JSON Value at the MCP boundary (so the
        // trait stays core-crate-pure). If the caller sends something
        // that is not a JSON object we treat it as "no filter" — the
        // explicit category/installed_only params still cover the common
        // cases. The MCP `JsonSchema` layer will have already flagged
        // hard type errors. We log the drop so operators can diagnose
        // unexpected filter shapes in production.
        let filter_map = match filter {
            None => None,
            Some(v) => match serde_json::from_value::<HashMap<String, serde_json::Value>>(v) {
                Ok(map) => Some(map),
                Err(e) => {
                    tracing::warn!(error = %e, "hub_search: filter value is not a JSON object — treating as no filter");
                    None
                }
            },
        };

        // Negative limit values from MCP callers are clamped to 0 rather
        // than wrapping to a huge usize (unchecked-user-bound-input pattern).
        // Downstream semantics: `Some(0)` means "no limit" (return all) —
        // the truncate path in `AppService::hub_search` short-circuits on 0.
        let opts = ListOpts {
            limit: limit.map(|n| n.max(0) as usize),
            sort,
            filter: filter_map,
            fields,
            verbose,
        };

        tokio::task::spawn_blocking(move || {
            AppService::hub_search(
                &svc,
                query.as_deref(),
                category.as_deref(),
                installed_only,
                opts,
            )
        })
        .await
        .map_err(|e| format!("hub_search task panicked: {e}"))?
    }

    // ─── Package read ─────────────────────────────────────────

    async fn pkg_read_init_lua(&self, name: &str) -> Result<String, String> {
        AppService::pkg_read_init_lua(self, name, None)
    }

    async fn pkg_get_narrative_md(&self, name: &str) -> Result<Option<String>, String> {
        AppService::pkg_get_narrative_md(self, name).await
    }

    async fn pkg_meta(&self, name: &str) -> Result<String, String> {
        let filter = serde_json::json!({ "name": name });
        let json_str = EngineApi::pkg_list(
            self,
            None,
            None,
            None,
            Some(filter),
            None,
            Some("full".to_string()),
        )
        .await?;
        let val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| format!("pkg_meta: failed to parse pkg_list response: {e}"))?;
        let pkgs = val
            .get("packages")
            .and_then(|p| p.as_array())
            .ok_or_else(|| "pkg_meta: pkg_list response missing 'packages' field".to_string())?;
        if pkgs.is_empty() {
            return Err(format!("pkg not found: {name}"));
        }
        serde_json::to_string(&pkgs[0]).map_err(|e| format!("pkg_meta: serialize entry: {e}"))
    }

    // ─── Package scaffold ─────────────────────────────────────

    async fn pkg_scaffold(
        &self,
        name: String,
        target_dir: Option<String>,
        category: Option<String>,
        description: Option<String>,
    ) -> Result<String, String> {
        let svc = self.clone();
        tokio::task::spawn_blocking(move || {
            AppService::pkg_scaffold(
                &svc,
                &name,
                target_dir.as_deref(),
                category.as_deref(),
                description.as_deref(),
            )
        })
        .await
        .map_err(|e| format!("pkg_scaffold task panicked: {e}"))?
    }

    // ─── Hub resources ───────────────────────────────────────

    /// Aggregate hub index across all registered cache sources.
    ///
    /// Delegates to `AppService::aggregate_index`, then serializes the
    /// result to a JSON string. Individual source failures and registry-load
    /// failures are embedded in the response JSON under a `"warnings"` field
    /// so the MCP caller can observe partial failures without losing the
    /// aggregate result.
    async fn hub_index_aggregate(&self) -> Result<String, String> {
        let svc = self.clone();
        let (index, warnings) = tokio::task::spawn_blocking(move || {
            AppService::aggregate_index(&svc).map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| format!("hub_index_aggregate task panicked: {e}"))??;

        let mut json = serde_json::to_value(&index)
            .map_err(|e| format!("hub_index_aggregate: serialize index: {e}"))?;
        if !warnings.is_empty() {
            if let Some(obj) = json.as_object_mut() {
                obj.insert("warnings".to_string(), serde_json::json!(warnings));
            }
        }
        serde_json::to_string(&json)
            .map_err(|e| format!("hub_index_aggregate: serialize final: {e}"))
    }

    // ─── Diagnostics ─────────────────────────────────────────

    async fn info(&self) -> String {
        AppService::info(self)
    }

    // ─── Pool management ─────────────────────────────────────

    async fn pool_ensure(&self) -> Result<String, String> {
        AppService::pool_ensure_impl(self).await
    }

    async fn pool_status(&self, sid: Option<String>) -> Result<String, String> {
        AppService::pool_status_impl(self, sid).await
    }

    async fn pool_stop(&self, sid: Option<String>) -> Result<String, String> {
        AppService::pool_stop_impl(self, sid).await
    }
}

// ─── Pool inherent helpers ────────────────────────────────────────────────────

impl AppService {
    /// Scan registry.json, GC dead workers, and return live sessions.
    ///
    /// Idempotent: calling twice produces the same result when no workers change
    /// state between calls.  Does NOT spawn new workers.
    pub(crate) async fn pool_ensure_impl(&self) -> Result<String, String> {
        let reg_path = self.pool_reg_path.clone();
        let lock_path = self.pool_lock_path.clone();

        let sessions =
            tokio::task::spawn_blocking(move || -> Result<Vec<serde_json::Value>, PoolError> {
                with_registry_lock(&lock_path, || {
                    let mut reg = PoolRegistry::load_or_default(&reg_path)?;
                    let survivors = reg.scan_and_gc()?;
                    // Persist GC result back to disk.
                    reg.save(&reg_path)?;
                    let entries = survivors
                        .iter()
                        .map(|e| {
                            serde_json::json!({
                                "sid": e.sid,
                                "pid": e.pid,
                                "sock": e.sock.to_string_lossy(),
                                "version": e.version,
                            })
                        })
                        .collect::<Vec<_>>();
                    Ok(entries)
                })
            })
            .await
            .map_err(|e| format!("pool_ensure: task panicked: {e}"))?
            .map_err(|e| e.to_string())?;

        let pool_version = env!("CARGO_PKG_VERSION");
        serde_json::to_string(&serde_json::json!({
            "sessions": sessions,
            "pool_version": pool_version,
        }))
        .map_err(|e| format!("pool_ensure: serialize: {e}"))
    }

    /// Return pool worker status from registry.json.
    ///
    /// When `sid` is `Some`, restricts output to that single worker.
    /// Uses kill -0 liveness check for each returned entry.
    pub(crate) async fn pool_status_impl(&self, sid: Option<String>) -> Result<String, String> {
        let reg_path = self.pool_reg_path.clone();
        let lock_path = self.pool_lock_path.clone();

        let sessions =
            tokio::task::spawn_blocking(move || -> Result<Vec<serde_json::Value>, PoolError> {
                with_registry_lock(&lock_path, || {
                    let mut reg = PoolRegistry::load_or_default(&reg_path)?;
                    // GC dead entries in-place so status reflects reality.
                    let _ = reg.scan_and_gc()?;
                    reg.save(&reg_path)?;

                    let entries: Vec<serde_json::Value> = reg
                        .sessions
                        .iter()
                        .filter(|e| sid.as_deref().map(|s| e.sid == s).unwrap_or(true))
                        .map(|e| {
                            serde_json::json!({
                                "sid": e.sid,
                                "pid": e.pid,
                                "sock": e.sock.to_string_lossy(),
                                "version": e.version,
                                "created_at": e.created_at,
                                // Status is "running" for all live entries (UDS ping not required in POC).
                                "status": "running",
                            })
                        })
                        .collect();
                    Ok(entries)
                })
            })
            .await
            .map_err(|e| format!("pool_status: task panicked: {e}"))?
            .map_err(|e| e.to_string())?;

        let pool_version = env!("CARGO_PKG_VERSION");
        serde_json::to_string(&serde_json::json!({
            "sessions": sessions,
            "pool_version": pool_version,
        }))
        .map_err(|e| format!("pool_status: serialize: {e}"))
    }

    /// Send SIGTERM to all workers or a single worker identified by `sid`.
    ///
    /// After SIGTERM, removes the entry from registry.json.
    /// Returns `{"stopped": [...], "errors": [...]}`.
    /// SIGTERM send failures are surfaced in the `errors` array (not dropped silently).
    pub(crate) async fn pool_stop_impl(&self, sid: Option<String>) -> Result<String, String> {
        let reg_path = self.pool_reg_path.clone();
        let lock_path = self.pool_lock_path.clone();

        let result = tokio::task::spawn_blocking(
            move || -> Result<(Vec<String>, Vec<String>), PoolError> {
                with_registry_lock(&lock_path, || {
                    let mut reg = PoolRegistry::load_or_default(&reg_path)?;

                    // Determine targets.
                    let targets: Vec<_> = reg
                        .sessions
                        .iter()
                        .filter(|e| sid.as_deref().map(|s| e.sid == s).unwrap_or(true))
                        .cloned()
                        .collect();

                    let mut stopped: Vec<String> = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for entry in &targets {
                        #[cfg(unix)]
                        {
                            // K-52: guard u32 → i32 (pid_t) overflow; also reject pid == 0
                            // (POSIX kill(2): pid=0 signals every process in the calling process
                            // group, pid<0 signals a process group — both are unsafe here).
                            let pid_t = match i32::try_from(entry.pid) {
                                Ok(p) if p > 0 => p,
                                Ok(_) => {
                                    errors.push(format!(
                                        "sid={}: pid={} is not a valid POSIX target pid (must be > 0); skipping SIGTERM",
                                        entry.sid, entry.pid
                                    ));
                                    reg.remove(&entry.sid);
                                    continue;
                                }
                                Err(_) => {
                                    errors.push(format!(
                                        "sid={}: pid={} exceeds i32::MAX, cannot send SIGTERM (K-52)",
                                        entry.sid, entry.pid
                                    ));
                                    // Remove the entry anyway (PID is invalid, worker is unreachable).
                                    reg.remove(&entry.sid);
                                    continue;
                                }
                            };

                            // SAFETY: libc::kill(pid, SIGTERM) is a thin syscall wrapper.
                            // pid_t > 0, verified by the match arm above.
                            // pid fits in i32 (verified above).
                            let ret = unsafe { libc::kill(pid_t, libc::SIGTERM) };
                            if ret == 0 {
                                stopped.push(entry.sid.clone());
                            } else {
                                let os_err = std::io::Error::last_os_error();
                                if os_err.raw_os_error() == Some(libc::ESRCH) {
                                    // Process already dead — treat as stopped (idempotent).
                                    stopped.push(entry.sid.clone());
                                } else {
                                    errors.push(format!(
                                        "sid={}: SIGTERM failed: {}",
                                        entry.sid, os_err
                                    ));
                                }
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            // Non-Unix: cannot send SIGTERM; report as unsupported.
                            errors.push(format!(
                                "sid={}: SIGTERM not supported on this platform",
                                entry.sid
                            ));
                        }
                        // Remove from registry regardless of SIGTERM result
                        // (dead or dying, we no longer track it).
                        reg.remove(&entry.sid);
                    }

                    // Persist updated registry (entries removed).
                    reg.save(&reg_path)?;

                    Ok((stopped, errors))
                })
            },
        )
        .await
        .map_err(|e| format!("pool_stop: task panicked: {e}"))?
        .map_err(|e| e.to_string())?;

        let (stopped, errors) = result;
        serde_json::to_string(&serde_json::json!({
            "stopped": stopped,
            "errors": errors,
        }))
        .map_err(|e| format!("pool_stop: serialize: {e}"))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::test_support::make_app_service_at;

    /// pool_stop_impl rejects pid=0 without delivering SIGTERM.
    ///
    /// A registry.json containing `"pid": 0` must be handled as an invalid
    /// POSIX target: the error is surfaced in the `errors` array, the entry is
    /// removed from the on-disk registry, and the test process itself survives
    /// (proving no SIGTERM was sent to the process group).
    #[tokio::test]
    #[cfg(unix)]
    async fn pool_stop_pid_zero_is_rejected() {
        // Arrange: build an AppService rooted at a tempdir so no real $HOME is
        // touched, then seed registry.json with a single pid=0 entry.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().to_path_buf();
        let svc = make_app_service_at(root.clone()).await;

        // The pool registry lives at {app_dir}/state/pool/registry.json.
        // AppDir::state_dir() resolves to {root}/state.
        let pool_reg_path = root.join("state").join("pool").join("registry.json");
        std::fs::create_dir_all(pool_reg_path.parent().unwrap()).expect("create pool dir");

        let seeded = serde_json::json!({
            "sessions": [{
                "sid": "zero-pid-session",
                "pid": 0u32,
                "sock": "/tmp/alc-pool/zero.sock",
                "version": "0.30.0",
                "created_at": "2026-01-01T00:00:00Z"
            }]
        });
        std::fs::write(&pool_reg_path, seeded.to_string()).expect("seed registry.json");

        // Act: stop all sessions.
        let json_str = svc.pool_stop_impl(None).await.expect("pool_stop_impl");
        let result: serde_json::Value =
            serde_json::from_str(&json_str).expect("response is valid JSON");

        // Assert (1): the error message contains "not a valid POSIX target pid".
        let errors = result["errors"].as_array().expect("errors array");
        assert!(
            !errors.is_empty(),
            "expected at least one error for pid=0 entry"
        );
        let err_msg = errors[0].as_str().unwrap_or("");
        assert!(
            err_msg.contains("not a valid POSIX target pid"),
            "unexpected error message: {err_msg}"
        );

        // Assert (2): stopped array is empty (no process was stopped).
        let stopped = result["stopped"].as_array().expect("stopped array");
        assert!(
            stopped.is_empty(),
            "pid=0 entry must not appear in stopped list"
        );

        // Assert (3): the entry is removed from the on-disk registry.
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&pool_reg_path).expect("read registry"))
                .expect("parse registry");
        let sessions = on_disk["sessions"].as_array().expect("sessions array");
        assert!(
            sessions.is_empty(),
            "pid=0 entry must be removed from on-disk registry"
        );

        // Assert (4): test process is still alive — trivially confirmed by
        // reaching this line without being killed by SIGTERM.
    }
}
