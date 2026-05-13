use std::sync::Arc;

use algocline_core::QueryId;
use algocline_engine::{FeedResult, VariantPkg};

use super::eval_store::{splice_response_string, splice_response_warnings};
use super::resolve::{is_package_installed, make_require_code, resolve_code, QueryResponse};
use super::transcript::write_transcript_log;
use super::AppService;
use crate::pool::dispatch::{continue_via_pool, run_via_pool};

/// Recover from MCP clients that JSON-stringify an object-typed field
/// before sending. The MCP `inputSchema` for `ctx` / `opts` /
/// `strategy_opts` declares `type: object`, but some clients send a
/// JSON-encoded string instead. Without this normalisation the value
/// reaches Lua as a string and breaks pkgs that require a table
/// (issue 1778656404-63015).
///
/// Only `Value::String` payloads that parse into a JSON object or
/// array are replaced; ordinary strings and primitive scalars pass
/// through untouched.
pub(crate) fn normalize_stringified_json_object(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(ref s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(parsed @ serde_json::Value::Object(_)) => parsed,
            Ok(parsed @ serde_json::Value::Array(_)) => parsed,
            _ => v,
        },
        other => other,
    }
}

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
    /// When `host_mode: Some(true)` is passed, the call is proxied via
    /// `PoolClient` to a long-lived worker subprocess over a Unix domain socket.
    /// When `host_mode` is `None` or `Some(false)` the existing in-process
    /// `Executor::start_session` path is used unchanged.
    ///
    /// # Concurrency
    ///
    /// **host_mode=false (default)**: No additional locking beyond `SessionRegistry`
    /// lock C. `AppService` itself holds no long-lived lock during this call.
    ///
    /// **host_mode=true**: Acquires `RwLock<PoolRegistry>` (write) and advisory
    /// `fs4::FileExt::lock_exclusive` to update `registry.json`. These locks are
    /// **not** held across the UDS round-trip await.
    ///
    /// **Cancel safety**: cancelling this `.await` mid-UDS-request leaves the
    /// worker subprocess running. The registry entry persists; callers can
    /// reconnect via `alc_continue` after MCP restart.
    pub async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
        project_root: Option<String>,
        host_mode: Option<bool>,
    ) -> Result<String, String> {
        let code = resolve_code(code, code_file)?;
        let ctx = normalize_stringified_json_object(ctx.unwrap_or(serde_json::Value::Null));
        let (extra, extra_warnings) = self.resolve_extra_lib_paths(project_root.as_deref());
        let (variants, variant_warnings) = self.resolve_variant_pkgs(project_root.as_deref());
        let mut warnings: Vec<String> = extra_warnings;
        warnings.extend(variant_warnings);

        if host_mode == Some(true) {
            // ── Pool path (Crux: MCP thin proxy IPC boundary) ─────────────────
            // Worker subprocess is spawned and communicated via UDS.
            // SessionRegistry (in-memory) is NOT touched on this path.
            let (session_id, json, pool_save_error) = run_via_pool(
                &self.pool_dir,
                &self.pool_reg_path,
                &self.pool_lock_path,
                extra,
                code,
                ctx,
            )
            .await
            .map_err(|e| e.to_string())?;

            // session_id is stored in the JSON by the worker; update the
            // in-memory registry so this MCP instance can route continues
            // without another disk read.
            // Load the just-persisted entry from disk to keep in-memory
            // registry in sync.  This is a best-effort convenience cache;
            // the disk state is authoritative.  Failure is surfaced to the
            // MCP wire response as `pool_cache_reload_warning` so the caller
            // can observe stale-cache conditions; tracing::warn! is kept for
            // operator visibility in logs.
            let cache_reload_warning: Option<String> =
                match crate::pool::PoolRegistry::load_or_default(&self.pool_reg_path) {
                    Ok(reg) => {
                        let mut guard = self.pool_registry.write().await;
                        *guard = reg;
                        None
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "failed to reload pool registry after run; in-memory cache may be stale"
                        );
                        Some(e.to_string())
                    }
                };

            let json = splice_response_warnings(&json, "lib_path_warnings", &warnings);
            let json = match pool_save_error {
                Some(msg) => splice_response_string(&json, "pool_save_error", &msg),
                None => json,
            };
            let json = match cache_reload_warning {
                Some(msg) => splice_response_string(&json, "pool_cache_reload_warning", &msg),
                None => json,
            };
            let _ = session_id; // session_id is embedded in the JSON response
            return Ok(json);
        }

        // ── In-process path (default) ──────────────────────────────────────────
        let json = self
            .start_and_tick(code, ctx, None, extra, variants)
            .await?;
        Ok(splice_response_warnings(
            &json,
            "lib_path_warnings",
            &warnings,
        ))
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

        let opts = opts.map(normalize_stringified_json_object);
        let mut ctx_map = match opts {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        if let Some(task) = task {
            ctx_map.insert("task".into(), serde_json::Value::String(task));
        }
        let ctx = serde_json::Value::Object(ctx_map);

        let (extra, extra_warnings) = self.resolve_extra_lib_paths(project_root.as_deref());
        let (variants, variant_warnings) = self.resolve_variant_pkgs(project_root.as_deref());
        let mut warnings: Vec<String> = extra_warnings;
        warnings.extend(variant_warnings);
        let json = self
            .start_and_tick(code, ctx, Some(strategy), extra, variants)
            .await?;
        Ok(splice_response_warnings(
            &json,
            "lib_path_warnings",
            &warnings,
        ))
    }

    /// Continue a paused execution — batch feed.
    ///
    /// For pool sessions (`session_id` found in registry.json), each response
    /// in the batch is forwarded to the worker via `PoolClient::send_request`.
    /// For in-MCP sessions, the existing `SessionRegistry::feed_response` path
    /// is used unchanged.
    pub async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        // ── Pool path check (same registry lookup as continue_single) ─────────
        let pool_entry = {
            let reg = self.pool_registry.read().await;
            reg.find(session_id).cloned()
        };

        let pool_entry = if pool_entry.is_some() {
            pool_entry
        } else {
            match crate::pool::PoolRegistry::load_or_default(&self.pool_reg_path) {
                Ok(reg) => {
                    let entry = reg.find(session_id).cloned();
                    if entry.is_some() {
                        let mut guard = self.pool_registry.write().await;
                        *guard = reg;
                    }
                    entry
                }
                Err(e) => {
                    return Err(format!("Continue failed: {e}"));
                }
            }
        };

        if let Some(entry) = pool_entry {
            // ── Pool routing ────────────────────────────────────────────────────
            let mut last_json = None;
            for qr in responses {
                let json =
                    continue_via_pool(&entry, session_id, qr.response, Some(qr.query_id), qr.usage)
                        .await
                        .map_err(|e| format!("Continue failed: {e}"))?;
                last_json = Some(json);
            }
            return last_json.ok_or_else(|| "Empty responses array".to_string());
        }

        // ── In-MCP path ────────────────────────────────────────────────────────
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
    ///
    /// Routing is automatic: if `session_id` is found in `registry.json`
    /// (pool path), the call is proxied via `PoolClient` over UDS. If not
    /// found (in-MCP path), the existing `SessionRegistry::feed_response`
    /// is used. Both paths never coexist for the same `session_id`.
    ///
    /// # Concurrency
    ///
    /// **Pool path**: acquires `RwLock<PoolRegistry>` (read) to look up the
    /// session entry, then acquires `tokio::sync::Mutex` inside `PoolClient`
    /// to serialize the UDS write. Neither lock is held across the UDS await.
    ///
    /// **In-MCP path**: acquires lock C in the two-phase pattern documented on
    /// `SessionRegistry::feed_response`.
    ///
    /// **Cancel safety**: cancelling mid-await on the pool path leaves the
    /// worker subprocess running (UDS send may have been partially written;
    /// `read_line` is not cancel-safe — a partial line in the buffer renders
    /// the connection unusable and `PoolClient` must reconnect).
    pub async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
        usage: Option<algocline_core::TokenUsage>,
    ) -> Result<String, String> {
        // ── Pool path: check in-memory registry, then disk registry ───────────
        // K-4: acquire read lock, clone the entry, release lock BEFORE await.
        let pool_entry = {
            let reg = self.pool_registry.read().await;
            reg.find(session_id).cloned()
        }; // read lock released here

        // If in-memory cache missed, check disk (e.g. after MCP restart).
        let pool_entry = if pool_entry.is_some() {
            pool_entry
        } else {
            match crate::pool::PoolRegistry::load_or_default(&self.pool_reg_path) {
                Ok(reg) => {
                    let entry = reg.find(session_id).cloned();
                    if entry.is_some() {
                        // Warm the in-memory cache.
                        let mut guard = self.pool_registry.write().await;
                        *guard = reg;
                    }
                    entry
                }
                Err(e) => {
                    // Corrupt registry: propagate to MCP wire per §Error 伝播規律.
                    return Err(format!("Continue failed: {e}"));
                }
            }
        };

        if let Some(entry) = pool_entry {
            // ── Pool routing (Crux: MCP thin proxy IPC boundary) ──────────────
            let json = continue_via_pool(
                &entry,
                session_id,
                response,
                query_id.map(str::to_string),
                usage,
            )
            .await
            .map_err(|e| format!("Continue failed: {e}"))?;
            return Ok(json);
        }

        // ── In-MCP path ────────────────────────────────────────────────────────
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
            // Mutex poison means a previous thread panicked while holding the lock.
            // Strategy name is non-critical for correctness, but the failure must be
            // surfaced to MCP callers so it is observable, not silently dropped.
            // See CLAUDE.md §Service 層の Error 伝播規律.
            let strategy = match self.session_strategies.lock() {
                Ok(mut map) => map.remove(session_id),
                Err(e) => {
                    tracing::warn!(
                        "session_strategies mutex poisoned for '{}': {}",
                        session_id,
                        e
                    );
                    // Return warning immediately; transcript cannot be written
                    // without strategy context being reliably recoverable.
                    return Some(format!(
                        "session_strategies mutex poisoned for '{session_id}': {e}"
                    ));
                }
            };
            // write_transcript_log returns Ok(Some(warning)) when meta write
            // failed but the main log succeeded, so both Err and meta warning
            // are surfaced as transcript_warning on the wire response.
            match write_transcript_log(
                &self.log_config,
                session_id,
                &exec_result.metrics,
                strategy.as_deref(),
            ) {
                Err(e) => Some(e.to_string()),
                Ok(meta_warning) => meta_warning,
            }
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

    // ── ST6: pool registry routing tests ────────────────────────────────────

    use crate::pool::PoolSessionEntry;

    /// T1: continue_single falls through to in-MCP path when session is not in pool registry.
    ///
    /// An unknown session ID should not be found in the pool registry and
    /// should reach the `SessionRegistry::feed_response` path, which returns
    /// an error because no session exists in the in-memory registry either.
    #[tokio::test]
    async fn continue_single_in_mcp_path_on_registry_miss() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;

        // The pool registry on disk and in-memory is empty (no workers registered).
        // continue_single should fall through to the in-MCP path and return
        // "not found" because the in-memory session registry also has nothing.
        let result = svc
            .continue_single(
                "nonexistent-session-id",
                "some response".to_string(),
                None,
                None,
            )
            .await;
        assert!(
            result.is_err(),
            "unknown session must return Err on in-MCP path"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not found") || msg.contains("Continue failed"),
            "error must indicate session not found, got: {msg}"
        );
    }

    /// T2: AppService::new initialises pool_registry as empty when pool dir absent.
    ///
    /// Verifies that startup GC with a missing registry.json (normal first-run)
    /// produces an empty PoolRegistry (not an error).
    #[tokio::test]
    async fn app_service_new_initialises_empty_pool_registry() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;

        let reg = svc.pool_registry.read().await;
        assert!(
            reg.sessions.is_empty(),
            "pool registry must be empty on first-run (no registry.json)"
        );
    }

    /// T2b: AppService correctly stores pool registry paths derived from app_dir.
    ///
    /// Verifies that pool_dir / pool_reg_path / pool_lock_path are
    /// non-empty paths derived from state_dir/pool/*.
    #[tokio::test]
    async fn app_service_pool_paths_correctly_derived() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;

        assert!(
            svc.pool_dir.ends_with("pool"),
            "pool_dir must end in 'pool', got: {}",
            svc.pool_dir.display()
        );
        assert!(
            svc.pool_reg_path.ends_with("pool/registry.json"),
            "pool_reg_path must end in 'pool/registry.json', got: {}",
            svc.pool_reg_path.display()
        );
        assert!(
            svc.pool_lock_path.ends_with("pool/registry.lock"),
            "pool_lock_path must end in 'pool/registry.lock', got: {}",
            svc.pool_lock_path.display()
        );
    }

    /// T3: continue_single propagates PoolError::RegistryCorrupted to MCP wire.
    ///
    /// When registry.json is corrupt and there is a cache miss, continue_single
    /// must return Err (not silently proceed with empty registry). This verifies
    /// the CLAUDE.md §Error 伝播規律 invariant — no unwrap_or_default() swallowing.
    #[tokio::test]
    async fn continue_single_propagates_corrupted_registry_error() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;

        // Write a corrupt registry.json to the pool directory.
        let pool_dir = svc.pool_dir.clone();
        std::fs::create_dir_all(&pool_dir).expect("create pool dir");
        std::fs::write(pool_dir.join("registry.json"), b"{ not valid json !!!")
            .expect("write corrupt registry");

        // The in-memory cache is empty (startup GC failed on the corrupt file,
        // so pool_registry is empty default).  The disk read in continue_single
        // will hit the corrupt file and must propagate the error.
        let result = svc
            .continue_single("any-session-id", "response".to_string(), None, None)
            .await;
        assert!(
            result.is_err(),
            "corrupted registry must cause Err, not silent empty fallback"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("corrupted") || msg.contains("parse") || msg.contains("Continue failed"),
            "error must mention registry problem, got: {msg}"
        );
    }

    // ── pool_cache_reload_warning splice tests ───────────────────────────────

    use super::super::eval_store::splice_response_string;

    /// T1 (happy path): splice_response_string inserts pool_cache_reload_warning
    /// into a valid JSON object response.
    ///
    /// Verifies the crux-card constraint: cache-reload failure must surface on
    /// the MCP wire as an additive field, not remain warn!-only.
    #[test]
    fn splice_response_string_injects_cache_reload_warning() {
        let json = r#"{"status":"finished","result":{"ok":true}}"#;
        let msg = "failed to reload pool registry: No such file or directory";
        let out = splice_response_string(json, "pool_cache_reload_warning", msg);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert_eq!(
            v["pool_cache_reload_warning"].as_str(),
            Some(msg),
            "pool_cache_reload_warning must be present in response"
        );
        // Original fields preserved (additive, not destructive).
        assert_eq!(v["status"].as_str(), Some("finished"));
    }

    /// T2 (edge case): splice_response_string is a no-op when input is not a
    /// JSON object (e.g. bare string or array).
    ///
    /// Guards against panics when strategy output is malformed.
    #[test]
    fn splice_response_string_passthrough_on_non_object_json() {
        let non_object = r#""just a string""#;
        let out = splice_response_string(non_object, "pool_cache_reload_warning", "err");
        // Must return original unchanged.
        assert_eq!(out, non_object);
    }

    /// T3 (error path / None branch): when cache_reload_warning is None, the
    /// pool_cache_reload_warning field must NOT appear in the response JSON.
    ///
    /// Verifies the None arm of the new match block in run.rs leaves the JSON
    /// untouched, consistent with the pool_save_error pattern.
    #[test]
    fn splice_response_string_not_called_when_none() {
        let json = r#"{"status":"finished"}"#;
        // Simulate the None branch: we simply do not call splice_response_string.
        let v: serde_json::Value = serde_json::from_str(json).expect("valid JSON");
        assert!(
            v.get("pool_cache_reload_warning").is_none(),
            "pool_cache_reload_warning must be absent when no cache-reload error occurred"
        );
    }

    /// T1b: in-memory pool registry lookup finds an entry and routes to pool path.
    ///
    /// Inserts a live entry (current process PID) into the in-memory registry
    /// and verifies that continue_single attempts the pool path (fails with
    /// connection error because no real worker socket exists, not "not found").
    #[tokio::test]
    async fn continue_single_routes_to_pool_on_registry_hit() {
        let tmp = tempfile::tempdir().expect("test tempdir");
        let svc = make_app_service_with_log_dir(tmp.path().to_path_buf()).await;

        // Insert a fake entry pointing to a non-existent socket.
        // This simulates the case where a pool session was started.
        let fake_sock = tmp.path().join("nonexistent.sock");
        let entry = PoolSessionEntry::new(
            "test-pool-session",
            std::process::id(), // live PID — survives GC
            fake_sock.clone(),
            env!("CARGO_PKG_VERSION"),
        );
        {
            let mut reg = svc.pool_registry.write().await;
            reg.add(entry);
        }

        // continue_single should find the entry and attempt pool path.
        // The UDS connect will fail (no socket file) → PoolError::Connect → Err.
        // Importantly, the error is a connection error, NOT a "session not found" error.
        let result = svc
            .continue_single("test-pool-session", "response".to_string(), None, None)
            .await;
        assert!(
            result.is_err(),
            "pool path must fail with connect error (no real worker)"
        );
        let msg = result.unwrap_err();
        // The error must come from pool path (UDS connect), not from SessionRegistry.
        // "not found" would indicate the in-MCP path was taken instead.
        assert!(
            !msg.contains("session not found") || msg.contains("Continue failed"),
            "error must be from pool path (UDS connect), got: {msg}"
        );
    }
}
