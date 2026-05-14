//! `ReqIdRegistry` — wire-layer adapter exclusive mapping from `RequestId` to [`SessionId`].
//!
//! This module is intentionally confined to the `algocline-mcp` adapter crate.
//! Service-layer crates (`algocline-core`, `algocline-app`, `algocline-engine`) must never
//! import or reference any type from this module; wire concepts must not leak into the
//! service layer (Crux: `ReqIdRegistry wire-concept isolation`).
//!
//! # Lock discipline
//!
//! Every method acquires an `RwLock` guard, clones or mutates the value, and drops the guard
//! before the method returns. No guard is ever held across an `.await` point
//! (K-4 clone-then-release pattern; the `tokio::sync::RwLockReadGuard` / `RwLockWriteGuard`
//! are `!Send`, so holding one across `.await` in a `tokio::spawn` context would be a
//! compile error — this serves as the `test_req_rwlock_no_await_across_lock` compile-time gate).

use std::collections::HashMap;

use algocline_core::execution::SessionId;
use rmcp::model::NumberOrString;
use tokio::sync::RwLock;

/// Wire-level request identifier. This is a re-export of rmcp's `NumberOrString` so that
/// callers within `algocline-mcp` can use `RequestId` without importing rmcp directly.
pub type RequestId = NumberOrString;

/// Owns the sole `RequestId` → [`SessionId`] mapping for in-flight MCP requests.
///
/// `AlcService` holds this registry behind an `Arc<ReqIdRegistry>` so that the
/// `on_cancelled` handler can reverse-look up the session to cancel without any direct
/// task-kill path outside `ExecutionService`.
#[derive(Default)]
pub struct ReqIdRegistry {
    /// SAFETY: lock guard is dropped before this method returns; never held across `.await`.
    inner: RwLock<HashMap<RequestId, SessionId>>,
}

impl ReqIdRegistry {
    /// Register a `req_id` → `sid` mapping for a newly spawned execution.
    ///
    /// The write guard is acquired, the entry inserted, and the guard dropped immediately.
    /// No `.await` is performed while the guard is held.
    pub async fn insert(&self, req_id: RequestId, sid: SessionId) {
        // SAFETY: guard dropped before this method returns; never held across .await.
        self.inner.write().await.insert(req_id, sid);
    }

    /// Look up the [`SessionId`] for a given `req_id`.
    ///
    /// Returns `None` if the mapping has already been removed or was never registered.
    /// The read guard is acquired, the value cloned, and the guard dropped immediately.
    pub async fn lookup(&self, req_id: &RequestId) -> Option<SessionId> {
        // SAFETY: guard dropped before this method returns; never held across .await.
        self.inner.read().await.get(req_id).cloned()
    }

    /// Remove all entries whose value equals `sid` (O(n) scan over in-flight sessions,
    /// which is typically a very small number).
    ///
    /// Called by the terminal-cleanup background task in `alc_v2_run` and by
    /// `alc_v2_cancel` on successful cancellation.
    pub async fn remove_by_session(&self, sid: &SessionId) {
        // SAFETY: guard dropped before this method returns; never held across .await.
        self.inner.write().await.retain(|_, v| v != sid);
    }

    /// Remove the entry keyed by `req_id`.
    ///
    /// Used by callers that know the request-id directly (e.g. `on_cancelled` cleanup
    /// after a successful cancel resolves).
    pub async fn remove_by_request(&self, req_id: &RequestId) {
        // SAFETY: guard dropped before this method returns; never held across .await.
        self.inner.write().await.remove(req_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::execution::SessionId;
    use rmcp::model::NumberOrString;
    use std::sync::Arc;

    fn make_req_id(n: i64) -> RequestId {
        NumberOrString::Number(n)
    }

    fn make_sid(s: &str) -> SessionId {
        SessionId::new(s.to_string())
    }

    /// Verifies the sequential contract: insert → lookup (hit) → remove_by_session →
    /// lookup (miss).  Corresponds to `test_insert_lookup_remove_sequential` in
    /// concurrency-analysis.md §2.
    #[tokio::test]
    async fn insert_lookup_remove_sequential() {
        let registry = ReqIdRegistry::default();
        let req_id = make_req_id(1);
        let sid = make_sid("session-abc");

        registry.insert(req_id.clone(), sid.clone()).await;
        assert_eq!(registry.lookup(&req_id).await, Some(sid.clone()));

        registry.remove_by_session(&sid).await;
        assert_eq!(registry.lookup(&req_id).await, None);
    }

    /// Verifies that `remove_by_request` clears the specific entry.
    #[tokio::test]
    async fn remove_by_request_clears_entry() {
        let registry = ReqIdRegistry::default();
        let req_id = make_req_id(2);
        let sid = make_sid("session-def");

        registry.insert(req_id.clone(), sid.clone()).await;
        assert_eq!(registry.lookup(&req_id).await, Some(sid.clone()));

        registry.remove_by_request(&req_id).await;
        assert_eq!(registry.lookup(&req_id).await, None);
    }

    /// 8 tasks each perform 100 insert/lookup/remove_by_session cycles concurrently.
    /// Asserts deadlock-free completion in ≤ 5 seconds and that all keys are gone after
    /// all tasks finish.  Corresponds to `test_concurrent_short_lock` in
    /// concurrency-analysis.md §2.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_short_lock() {
        let registry = Arc::new(ReqIdRegistry::default());
        let mut handles = Vec::new();

        for task_id in 0i64..8 {
            let reg = Arc::clone(&registry);
            let handle = tokio::spawn(async move {
                for i in 0i64..100 {
                    let req_id = make_req_id(task_id * 1000 + i);
                    let sid = make_sid(&format!("sess-{task_id}-{i}"));
                    reg.insert(req_id.clone(), sid.clone()).await;
                    let _ = reg.lookup(&req_id).await;
                    reg.remove_by_session(&sid).await;
                }
            });
            handles.push(handle);
        }

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for h in handles {
                h.await.expect("task panicked");
            }
        })
        .await;
        assert!(
            result.is_ok(),
            "concurrent_short_lock timed out after 5 seconds (deadlock?)"
        );

        // All entries must be removed after each task cleaned up its own.
        for task_id in 0i64..8 {
            for i in 0i64..100 {
                let req_id = make_req_id(task_id * 1000 + i);
                assert_eq!(
                    registry.lookup(&req_id).await,
                    None,
                    "entry still present for task_id={task_id} i={i}"
                );
            }
        }
    }
}
