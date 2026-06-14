//! Pool worker registry lifecycle service backing `alc_pool_*` MCP tools.

use crate::pool::{registry::with_registry_lock, PoolError, PoolRegistry};

use super::AppService;

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
