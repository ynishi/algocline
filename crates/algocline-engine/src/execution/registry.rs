//! `SessionRegistryV2` — the engine-level session lifecycle manager for the v2 path.
//!
//! Coexists with the legacy `SessionRegistry` (`session.rs`) without modifying it.
//! New callers (Subtask 3's `AppService::ExecutionService` impl) use this registry.
//!
//! # Design invariants
//!
//! - **Invariant 6**: `spawn_v2()` returns the `SessionId` immediately; execution
//!   runs in the background via `tokio::spawn(driver_loop(...))`.
//! - **Crux R1**: No `rmcp::*`, `progressToken`, `_meta`, `notifications/*`, or
//!   `mcp_`-prefixed identifiers appear anywhere in this module.
//! - **Crux R2**: Cancellation uses `CancellationToken::cancel()`; no
//!   `JoinHandle::abort()` or process kill path exists.
//! - **Crux R3**: `observe()` is a sync `fn` that calls `bus_tx.subscribe()` and
//!   returns a valid handle with zero pre-registered observers.
//! - **K-4**: The `sessions` `RwLock` is never held across `.await` points; the
//!   `clone-then-release` pattern is used throughout.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use algocline_core::execution::{
    AwaitError, CancelError, CancelReason, ExecutionState, ExecutionStateTag, ObserveError,
    ObserverHandle, PauseKind, ProgressEvent, ResumeError, ResumeOutcome, SessionId, SpawnError,
    StateError, TerminalOutcome,
};
use algocline_core::{ExecutionMetrics, ExecutionObserver, QueryId, TokenUsage};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use super::driver::{build_cancel_info, driver_loop, now_ms, transition_state, DriverContext};
use super::observer::BroadcastObserverHandle;
use super::record::{RespTxsMap, SessionRecord};
use crate::card::FileCardStore;
use crate::executor::Executor;
use crate::state::JsonFileStore;

// ---------------------------------------------------------------------------
// SessionRegistryV2
// ---------------------------------------------------------------------------

/// Registry that manages the lifecycle of v2 execution sessions.
///
/// `Clone` is cheap — the inner `Arc<RwLock<...>>` is reference-counted.
#[derive(Clone)]
pub struct SessionRegistryV2 {
    sessions: Arc<RwLock<HashMap<SessionId, Arc<SessionRecord>>>>,
    executor: Arc<Executor>,
    state_store: Arc<JsonFileStore>,
    card_store: Arc<FileCardStore>,
    scenarios_dir: std::path::PathBuf,
}

impl SessionRegistryV2 {
    /// Create a new empty registry backed by `executor`, with the storage paths
    /// that will be injected into each spawned VM session.
    ///
    /// The `state_store` / `card_store` / `scenarios_dir` mirror the legacy
    /// `AppService` resolution against the `AppConfig::app_dir()` layout, so a
    /// v2 caller produces the same on-disk side effects as a legacy caller.
    pub fn new(
        executor: Arc<Executor>,
        state_store: Arc<JsonFileStore>,
        card_store: Arc<FileCardStore>,
        scenarios_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            executor,
            state_store,
            card_store,
            scenarios_dir,
        }
    }

    // -----------------------------------------------------------------------
    // spawn_v2
    // -----------------------------------------------------------------------

    /// Start a new v2 execution session, returning the `SessionId` immediately.
    ///
    /// Execution proceeds in the background via `tokio::spawn(driver_loop(...))`.
    /// The caller receives the `SessionId` without waiting for execution to complete
    /// or for the first event (Invariant 6 / debt #40955).
    ///
    /// Only [`algocline_core::execution::SpecKind::Run`] is supported in this subtask.
    /// Other variants return [`SpawnError::InvalidSpec`].  Subtask 3 will extend this
    /// to handle `Advice` and `Eval` through the full `AppService` path.
    ///
    /// `card_run_enabled` is the resolved `[setting.card].run` gate value.  It is
    /// threaded through into the per-session bridge so `alc.card.create` /
    /// `alc.card.append` calls carrying a `run` field are no-op when the gate
    /// is off (Phase 1-B).
    ///
    /// # Errors
    /// - [`SpawnError::Engine`] — the executor failed to start the session.
    /// - [`SpawnError::InvalidSpec`] — the provided spec is malformed or uses an
    ///   unsupported kind.
    pub async fn spawn_v2(
        &self,
        spec: algocline_core::execution::SessionSpec,
        card_run_enabled: bool,
    ) -> Result<SessionId, SpawnError> {
        use algocline_core::execution::SpecKind;

        // Extract code from the spec kind.  Only Run is supported here.
        let code = match spec.kind {
            SpecKind::Run { code } => code,
            other => {
                return Err(SpawnError::InvalidSpec(format!(
                    "SessionRegistryV2::spawn_v2 only supports SpecKind::Run; got {:?}",
                    std::mem::discriminant(&other)
                )));
            }
        };

        if code.trim().is_empty() {
            return Err(SpawnError::InvalidSpec("code must not be empty".into()));
        }

        let ctx = spec.ctx.unwrap_or_else(|| serde_json::json!({}));

        // Start the per-session VM using the storage paths injected at
        // registry construction (mirrors legacy AppService::start_and_tick).
        let session = self
            .executor
            .start_session(
                code,
                ctx,
                vec![], // extra_lib_paths — populated by Advice/Eval kinds later
                vec![], // variant_pkgs   — populated by Advice/Eval kinds later
                Arc::clone(&self.state_store),
                Arc::clone(&self.card_store),
                self.scenarios_dir.clone(),
                card_run_enabled,
            )
            .await
            .map_err(SpawnError::Engine)?;

        let (exec_task, llm_rx, vm_driver, metrics) = session.into_driver_parts();

        // Build shared components — all constructed before spawning the task.
        let state: Arc<Mutex<ExecutionState>> = Arc::new(Mutex::new(ExecutionState::Running));
        let cancel_token = CancellationToken::new();
        let resp_txs: RespTxsMap = Arc::new(Mutex::new(HashMap::new()));
        // Wall-clock ms timestamp for idle-time GC (Crux #3 legacy parity).
        // Initialised to now_ms() so a session that is evicted before driver_loop
        // even starts is treated as "just spawned" rather than immediately expired.
        let last_active: Arc<AtomicI64> = Arc::new(AtomicI64::new(now_ms()));
        // Wrap metrics in Arc and clone into both DriverContext and SessionRecord
        // so both can access the same SessionStatus accumulator (K-4 clone-then-release).
        let metrics_arc: Arc<ExecutionMetrics> = Arc::new(metrics);

        // Crux R3 (sink-free): the receiver returned alongside `bus_tx` is
        // dropped immediately.  `bus_tx.send()` returns `Err(SendError)` when
        // 0 observers are subscribed, but every call site in `driver_loop`
        // uses `let _ = bus_tx.send(...)` to absorb the result — the caller
        // is never crashed by 0 observers.  See
        // `record::tests::bus_tx_does_not_crash_caller_with_zero_observers`.
        let (bus_tx, _) = tokio::sync::broadcast::channel::<ProgressEvent>(256);

        let session_id = SessionId::generate();

        // Bundle shared resources for driver_loop.
        let ctx = DriverContext {
            state: Arc::clone(&state),
            bus_tx: bus_tx.clone(),
            cancel_token: cancel_token.clone(),
            resp_txs: Arc::clone(&resp_txs),
            last_active: Arc::clone(&last_active),
            metrics: Arc::clone(&metrics_arc),
        };

        let join_handle = tokio::spawn(async move {
            // vm_driver must stay alive for the duration of the session.
            let _keep_driver = vm_driver;
            driver_loop(ctx, exec_task, llm_rx).await;
        });

        // Assemble the record with all shared fields.
        let record = Arc::new(SessionRecord {
            state,
            bus_tx,
            last_active,
            cancel_token,
            join_handle: Mutex::new(Some(join_handle)),
            resp_txs,
            first_cancel_info: Mutex::new(None),
            metrics: metrics_arc,
        });

        // Insert into registry.
        {
            let mut map = self.sessions.write().await;
            map.insert(session_id.clone(), record);
        }
        Ok(session_id)
    }

    // -----------------------------------------------------------------------
    // state
    // -----------------------------------------------------------------------

    /// Query the current [`ExecutionState`] of a session.
    ///
    /// # Errors
    /// - [`StateError::NotFound`] — no session with the given id exists.
    pub async fn state(&self, id: &SessionId) -> Result<ExecutionState, StateError> {
        let record = self
            .get_record(id)
            .await
            .ok_or_else(|| StateError::NotFound(id.clone()))?;
        let guard = record.state.lock().await;
        Ok(guard.clone())
    }

    // -----------------------------------------------------------------------
    // resume
    // -----------------------------------------------------------------------

    /// Resume a paused session by delivering LLM responses.
    ///
    /// # Errors
    /// - [`ResumeError::NotFound`] — no session with the given id exists.
    /// - [`ResumeError::NotPaused`] — the session is not in the `Paused` state.
    /// - [`ResumeError::AlreadyCancelled`] — the session is already cancelled.
    pub async fn resume(
        &self,
        id: &SessionId,
        payload: algocline_core::execution::ResumePayload,
    ) -> Result<ResumeOutcome, ResumeError> {
        use algocline_core::execution::ResumePayload;

        let record = self
            .get_record(id)
            .await
            .ok_or_else(|| ResumeError::NotFound(id.clone()))?;

        // checkpoint C: at resume entry
        // If the token is already cancelled, reject the resume immediately.
        if record.cancel_token.is_cancelled() {
            return Err(ResumeError::AlreadyCancelled);
        }

        // Verify the session is Paused (or Cancelled after the token check above).
        let (actual_tag, pause_kind) = {
            let guard = record.state.lock().await;
            let tag = guard.tag();
            let kind = if let ExecutionState::Paused(ref info) = *guard {
                info.kind
            } else {
                PauseKind::Single
            };
            (tag, kind)
        };

        match actual_tag {
            ExecutionStateTag::Cancelled => return Err(ResumeError::AlreadyCancelled),
            ExecutionStateTag::Paused => {} // continue
            _ => return Err(ResumeError::NotPaused { actual_tag }),
        }

        // Extract query responses from the payload, preserving per-response usage.
        let responses: Vec<(String, String, Option<TokenUsage>)> = match payload {
            ResumePayload::Single {
                query_id,
                response,
                usage,
            } => vec![(query_id, response, usage)],
            ResumePayload::Batch(batch) => batch
                .into_iter()
                .map(|r| (r.query_id, r.response, r.usage))
                .collect(),
        };

        // Deliver responses via the shared resp_txs map.
        {
            let mut txs = record.resp_txs.lock().await;
            for (qid_str, response, _usage) in &responses {
                let qid = QueryId::parse(qid_str);
                match txs.remove(&qid) {
                    Some(tx) => {
                        if let Err(_e) = tx.send(Ok(response.clone())) {
                            tracing::debug!(
                                "registry::resume: oneshot receiver already dropped for query {qid_str}"
                            );
                        }
                    }
                    None => {
                        tracing::debug!("registry::resume: no pending tx for query {qid_str}");
                    }
                }
            }
        }

        // Propagate per-response usage to the metrics observer (Crux 1: same Arc).
        // Observer call is outside the txs lock scope to keep cancel/lock paths intact.
        let observer = record.metrics.create_observer();
        for (qid_str, response, usage) in &responses {
            let qid = QueryId::parse(qid_str);
            observer.on_response_fed(&qid, response, usage.as_ref());
        }

        // Transition state from Paused → Running.
        {
            let guard = record.state.lock().await;
            if guard.tag() == ExecutionStateTag::Paused {
                drop(guard);
                transition_state(&record.state, &record.bus_tx, ExecutionState::Running).await;
                let _ = record.bus_tx.send(ProgressEvent::ResumeAccepted {
                    payload_kind: pause_kind,
                    at: now_ms(),
                });
            }
        }

        Ok(ResumeOutcome::Continued)
    }

    // -----------------------------------------------------------------------
    // cancel
    // -----------------------------------------------------------------------

    /// Request cooperative cancellation of a session.
    ///
    /// Idempotent: returns `Ok(())` for sessions already in a terminal state.
    ///
    /// # Errors
    /// - [`CancelError::NotFound`] — no session with the given id exists.
    pub async fn cancel(&self, id: &SessionId, reason: CancelReason) -> Result<(), CancelError> {
        let record = self
            .get_record(id)
            .await
            .ok_or_else(|| CancelError::NotFound(id.clone()))?;

        // Idempotency: already terminal → Ok.
        {
            let guard = record.state.lock().await;
            if matches!(
                guard.tag(),
                ExecutionStateTag::Done | ExecutionStateTag::Failed | ExecutionStateTag::Cancelled
            ) {
                return Ok(());
            }
        }

        // Store the first CancelInfo (idempotent: only set once).
        {
            let mut first = record.first_cancel_info.lock().await;
            if first.is_none() {
                let info = build_cancel_info(&record.state, reason).await;
                *first = Some(info);
            }
        }

        // Signal the driver (Crux R2: cooperative — no abort).
        record.cancel_token.cancel();

        // For Paused sessions, transition immediately: the driver is blocked
        // waiting for a resume and won't hit a checkpoint on its own.
        let should_transition = {
            let guard = record.state.lock().await;
            guard.tag() == ExecutionStateTag::Paused
        };
        if should_transition {
            let cancel_info_opt = {
                let first = record.first_cancel_info.lock().await;
                first.clone()
            };
            if let Some(info) = cancel_info_opt {
                transition_state(
                    &record.state,
                    &record.bus_tx,
                    ExecutionState::Cancelled(info),
                )
                .await;
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // observe  (sync fn — Crux R3)
    // -----------------------------------------------------------------------

    /// Subscribe to the progress event stream for a session.
    ///
    /// This is a **synchronous** `fn`: `broadcast::Sender::subscribe()` is
    /// synchronous and does not perform I/O.  Multiple concurrent subscribers
    /// each receive the full event stream independently (Crux R3).
    ///
    /// # Errors
    /// - [`ObserveError::NotFound`] — no session with the given id exists, **or**
    ///   `try_read()` experienced lock contention (write lock held by `spawn`).
    ///   The contention path emits `tracing::warn!(target = "session.observe", ...)`;
    ///   callers cannot distinguish it from a true absent-session result.
    pub fn observe(&self, id: &SessionId) -> Result<Box<dyn ObserverHandle>, ObserveError> {
        // Non-blocking read; the write lock is only held very briefly during spawn.
        match self.sessions.try_read() {
            Ok(map) => {
                let record = map
                    .get(id)
                    .ok_or_else(|| ObserveError::NotFound(id.clone()))?;
                Ok(Box::new(BroadcastObserverHandle::new(&record.bus_tx)))
            }
            Err(_) => {
                tracing::warn!(
                    target = "session.observe",
                    session_id = %id,
                    "try_read contention; surfacing as NotFound"
                );
                Err(ObserveError::NotFound(id.clone()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // await_terminal
    // -----------------------------------------------------------------------

    /// Await the terminal state of a session.
    ///
    /// Polls the shared state until it reaches a terminal variant (`Done`,
    /// `Cancelled`, or `Failed`).  The `JoinHandle` is never `.abort()`-ed
    /// (Crux R2).
    ///
    /// # Errors
    /// - [`AwaitError::NotFound`] — no session with the given id exists.
    pub async fn await_terminal(&self, id: &SessionId) -> Result<TerminalOutcome, AwaitError> {
        let record = self
            .get_record(id)
            .await
            .ok_or_else(|| AwaitError::NotFound(id.clone()))?;

        // Single-awaiter path: take the JoinHandle and await `driver_loop`
        // completion directly.  Replaces the previous `yield_now()` polling
        // loop that occupied a tokio worker slot scheduling-wise even though
        // it consumed no CPU.  The `driver_loop` guarantees a terminal
        // `transition_state` before returning, so once `handle.await` resolves
        // the state is guaranteed terminal.
        let handle_opt = {
            let mut guard = record.join_handle.lock().await;
            guard.take()
        };

        if let Some(handle) = handle_opt {
            handle
                .await
                .map_err(|e| AwaitError::Joined(format!("driver_loop join error: {e}")))?;
        }
        // (None branch: another caller has already taken the handle.  Either
        // they are still awaiting it — in which case the driver_loop has not
        // yet transitioned to terminal — or they have already finished, in
        // which case the state is terminal.  We fall through to a single
        // state read; the rare concurrent race returns `AwaitError::Joined`.)

        let guard = record.state.lock().await;
        match &*guard {
            ExecutionState::Done(result) => Ok(TerminalOutcome::Done(result.clone())),
            ExecutionState::Cancelled(info) => Ok(TerminalOutcome::Cancelled(info.clone())),
            ExecutionState::Failed(info) => Ok(TerminalOutcome::Failed(info.clone())),
            other => Err(AwaitError::Joined(format!(
                "await_terminal: driver_loop completed but state is {:?} (concurrent awaiter race)",
                other.tag()
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // spawn_gc_task
    // -----------------------------------------------------------------------

    /// Spawn a background GC task that periodically evicts idle, terminal sessions.
    ///
    /// Mirrors the legacy `SessionRegistry::spawn_gc_task` contract (Crux #3 legacy
    /// parity) with two extensions:
    ///
    /// 1. **Subscriber-count gate** (Crux #1): a session is only evicted when
    ///    `bus_tx.receiver_count() == 0` at the moment the write guard is held,
    ///    ensuring no use-after-eviction for active observers.
    /// 2. **Parameterised `interval`** (Crux #2): callers can supply a sub-second
    ///    interval for test determinism without requiring `tokio::time::pause`.
    ///
    /// The `JoinHandle` returned by `tokio::spawn` is intentionally dropped —
    /// the task runs until process exit (legacy fire-and-forget contract).
    ///
    /// # K-4 invariant
    ///
    /// The `sessions` write guard is acquired once per GC tick.  All operations
    /// inside the guard (`receiver_count()`, `AtomicI64::load`, `HashMap::remove`)
    /// are **synchronous** — no `.await` is called while the guard is held.
    pub fn spawn_gc_task(&self, ttl: Duration, interval: Duration) {
        let sessions = Arc::clone(&self.sessions);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                // Acquire the write guard once per tick.  All reads and removes
                // within this block are sync — no `.await` inside the guard (K-4).
                let mut map = sessions.write().await;
                let mut to_evict: Vec<SessionId> = Vec::new();
                for (id, record) in map.iter() {
                    // Crux #1: check subscriber count atomically with the guard held.
                    // `receiver_count()` is sync (no lock required on its own), but
                    // holding the write guard here means `observe()` cannot attach a
                    // new subscriber via `try_read()` concurrently — TOCTOU excluded.
                    let no_subscribers = record.bus_tx.receiver_count() == 0;
                    let last_ms = record.last_active.load(Ordering::Relaxed);
                    if no_subscribers && is_expired_v2(last_ms, ttl) {
                        to_evict.push(id.clone());
                    }
                }
                for id in &to_evict {
                    tracing::info!(session_id = %id, "GC: reaping expired v2 session");
                    map.remove(id);
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Clone-then-release lookup (K-4): the lock is dropped before returning.
    async fn get_record(&self, id: &SessionId) -> Option<Arc<SessionRecord>> {
        let map = self.sessions.read().await;
        map.get(id).cloned()
    }
}

// ---------------------------------------------------------------------------
// GC helpers (module-private)
// ---------------------------------------------------------------------------

/// Returns `true` when the session has been idle for at least `ttl`.
///
/// Uses wall-clock milliseconds matching the legacy `is_expired_impl` semantics:
/// `now_ms() - last_active_ms >= ttl.as_millis()`.
///
/// The legacy implementation uses `Instant` (monotonic) whereas this uses
/// `SystemTime` (wall-clock) — identical to the `now_ms()` helper in `driver.rs`
/// and to `Session.last_activity_ms` in the legacy codebase (Crux #3 parity).
fn is_expired_v2(last_active_ms: i64, ttl: Duration) -> bool {
    let now = super::driver::now_ms();
    let elapsed_ms = now.saturating_sub(last_active_ms);
    elapsed_ms >= ttl.as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::execution::{
        CancelCode, CancelReason, ExecutionState, SessionSpec, SpecKind,
    };
    use std::sync::Arc;

    async fn make_executor() -> Arc<Executor> {
        Arc::new(Executor::new(vec![]).await.expect("Executor::new"))
    }

    /// Construct a registry backed by per-test tempdir paths so the legacy
    /// AppConfig::app_dir() layout is approximated without touching the user's
    /// `~/.algocline` directory.
    fn make_registry(executor: Arc<Executor>) -> (SessionRegistryV2, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let state_store = Arc::new(JsonFileStore::new(tmp.path().join("state")));
        let card_store = Arc::new(FileCardStore::new(tmp.path().join("cards")));
        let scenarios_dir = tmp.path().join("scenarios");
        (
            SessionRegistryV2::new(executor, state_store, card_store, scenarios_dir),
            tmp,
        )
    }

    fn simple_spec(code: &str) -> SessionSpec {
        SessionSpec {
            kind: SpecKind::Run {
                code: code.to_owned(),
            },
            project_root: None,
            ctx: None,
        }
    }

    fn cancel_reason() -> CancelReason {
        CancelReason {
            code: CancelCode::User,
            detail: None,
            requested_at: now_ms(),
        }
    }

    // -----------------------------------------------------------------------
    // spawn_returns_session_id_immediately (debt #40955)
    // -----------------------------------------------------------------------

    /// `spawn_v2` must return `SessionId` without blocking on execution.
    #[tokio::test]
    async fn spawn_returns_session_id_immediately() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let start = std::time::Instant::now();
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            registry.spawn_v2(simple_spec("return 42"), false),
        )
        .await;

        assert!(result.is_ok(), "spawn_v2 must complete within 200ms");
        assert!(
            result.unwrap().is_ok(),
            "spawn_v2 must return Ok(SessionId)"
        );

        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(150),
            "spawn_v2 took too long: {elapsed:?}"
        );
    }

    // -----------------------------------------------------------------------
    // state_query_running
    // -----------------------------------------------------------------------

    /// Immediately after spawn, `state()` must return Running or Paused.
    #[tokio::test]
    async fn state_query_running() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        // Lua that pauses immediately so the session is observable.
        let sid = registry
            .spawn_v2(simple_spec(r#"return alc.llm("q")"#), false)
            .await
            .expect("spawn");

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let state = registry.state(&sid).await.expect("state");
        assert!(
            matches!(state, ExecutionState::Running | ExecutionState::Paused(_)),
            "state just after spawn must be Running or Paused, got: {:?}",
            state.tag()
        );
    }

    // -----------------------------------------------------------------------
    // cancel_at_checkpoint_c_at_resume_entry
    // -----------------------------------------------------------------------

    /// `resume()` on a cancelled session must return `AlreadyCancelled`.
    #[tokio::test]
    async fn cancel_at_checkpoint_c_at_resume_entry() {
        use algocline_core::execution::{ResumeError, ResumePayload};

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec(r#"return alc.llm("q")"#), false)
            .await
            .expect("spawn");

        // Wait for Paused.
        let mut retries = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if registry.state(&sid).await.expect("state").tag() == ExecutionStateTag::Paused {
                break;
            }
            retries += 1;
            assert!(retries < 50, "session did not reach Paused state");
        }

        registry
            .cancel(&sid, cancel_reason())
            .await
            .expect("cancel");

        // checkpoint C: at resume entry
        let result = registry
            .resume(
                &sid,
                ResumePayload::Single {
                    query_id: "q".into(),
                    response: "4".into(),
                    usage: None,
                },
            )
            .await;

        assert!(
            matches!(result, Err(ResumeError::AlreadyCancelled)),
            "resume on cancelled session must return AlreadyCancelled, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // cancel_idempotent
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn cancel_idempotent() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec("return 1"), false)
            .await
            .expect("spawn");

        registry
            .cancel(&sid, cancel_reason())
            .await
            .expect("first cancel");
        registry
            .cancel(&sid, cancel_reason())
            .await
            .expect("second cancel");
    }

    // -----------------------------------------------------------------------
    // await_terminal returns Done without busy-polling
    // -----------------------------------------------------------------------

    /// Regression for #2 (case A): `await_terminal` must complete by awaiting
    /// the `driver_loop` `JoinHandle` directly (single-awaiter `take` +
    /// `.await`) instead of polling `state` in a `yield_now()` loop.  We can't
    /// observe scheduler occupancy from a test, but we can verify the
    /// behavioural contract: (1) the call returns the correct `TerminalOutcome`,
    /// (2) it returns within a tight wall-clock budget without sleep, and
    /// (3) a second concurrent caller does not panic.
    #[tokio::test]
    async fn await_terminal_returns_done_for_trivial_script() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec("return 42"), false)
            .await
            .expect("spawn");

        let outcome = registry.await_terminal(&sid).await.expect("await_terminal");
        match outcome {
            TerminalOutcome::Done(result) => {
                assert_eq!(result.value, serde_json::json!(42));
            }
            other => panic!("expected Done, got: {other:?}"),
        }
    }

    /// Regression for #2 (case A) single-awaiter discipline: when two callers
    /// race on `await_terminal`, the second caller (which observes `None` after
    /// the first has taken the handle) must NOT panic.  It must either return
    /// the same terminal outcome (if the first has already finished) or an
    /// `AwaitError::Joined` (the documented race fallback).
    #[tokio::test]
    async fn await_terminal_does_not_panic_on_second_concurrent_caller() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec("return 99"), false)
            .await
            .expect("spawn");

        let r1 = registry.clone();
        let r2 = registry.clone();
        let s1 = sid.clone();
        let s2 = sid.clone();

        let h1 = tokio::spawn(async move { r1.await_terminal(&s1).await });
        let h2 = tokio::spawn(async move { r2.await_terminal(&s2).await });

        let out1 = h1.await.expect("h1 join");
        let out2 = h2.await.expect("h2 join");

        // First-caller path must succeed with the real outcome.
        let first_ok = matches!(&out1, Ok(TerminalOutcome::Done(_)))
            || matches!(&out2, Ok(TerminalOutcome::Done(_)));
        assert!(
            first_ok,
            "at least one caller must observe Done; got out1={out1:?}, out2={out2:?}"
        );
        // Second caller may have observed Joined (race) or Done; either is OK,
        // neither must panic — which we've already verified by the join above.
    }

    // -----------------------------------------------------------------------
    // observe_sink_free (Crux R3 — registry level)
    // -----------------------------------------------------------------------

    /// `observe()` must succeed and return a valid handle even with 0 prior observers.
    #[tokio::test]
    async fn observe_sink_free_registry() {
        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec(r#"return alc.llm("q")"#), false)
            .await
            .expect("spawn");

        // observe() before any subscriber exists must succeed.
        let handle = registry.observe(&sid);
        assert!(
            handle.is_ok(),
            "observe() must return Ok even with 0 prior observers"
        );
    }

    // -----------------------------------------------------------------------
    // observe_multi_subscriber_fan_out (Crux R3 — registry level)
    // -----------------------------------------------------------------------

    /// Multiple independent observers each get the same events.
    #[tokio::test]
    async fn observe_multi_subscriber_fan_out_registry() {
        use algocline_core::execution::ObserverRecvError;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        // A script that returns immediately — the driver will publish Done.
        let sid = registry
            .spawn_v2(simple_spec("return 99"), false)
            .await
            .expect("spawn");

        // Subscribe 3 observers.
        let mut h1 = registry.observe(&sid).expect("observe h1");
        let mut h2 = registry.observe(&sid).expect("observe h2");
        let mut h3 = registry.observe(&sid).expect("observe h3");

        // Wait for terminal so we know events have been published.
        let _ = registry.await_terminal(&sid).await;

        // Each observer must receive at least the terminal StateTransition.
        // Drain with idle-timeout: bus_tx is retained in SessionRecord for
        // sink-free late-subscribe (Crux R3), so Closed never fires while the
        // registry is alive.  A 100ms idle window after await_terminal() is
        // sufficient — all events are already buffered.
        use std::time::Duration;
        for (label, handle) in [("h1", &mut h1), ("h2", &mut h2), ("h3", &mut h3)] {
            let mut got_transition = false;
            loop {
                match tokio::time::timeout(Duration::from_millis(100), handle.recv()).await {
                    Ok(Ok(ProgressEvent::StateTransition { .. })) => got_transition = true,
                    Ok(Ok(_)) => {}
                    Ok(Err(ObserverRecvError::Closed)) => break,
                    Ok(Err(ObserverRecvError::Lagged(_))) => {}
                    Err(_) => break, // idle-timeout: no more events coming
                }
            }
            assert!(
                got_transition,
                "{label}: must receive at least one StateTransition event"
            );
        }
    }

    // -----------------------------------------------------------------------
    // AC#5a — gc_evicts_terminal_session_after_ttl
    // -----------------------------------------------------------------------

    /// GC must remove a terminal session (no subscribers) after TTL has elapsed
    /// and one full interval tick has fired.
    ///
    /// Covers: `tokio::time::interval` + `AtomicI64::load` + `RwLock::write` +
    /// `receiver_count == 0` (Crux #1 / concurrency-analysis §2 5a).
    #[tokio::test]
    async fn gc_evicts_terminal_session_after_ttl() {
        use algocline_core::execution::ObserveError;
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let ttl = Duration::from_millis(100);
        let interval = Duration::from_millis(50);

        let sid = registry
            .spawn_v2(simple_spec("return 1"), false)
            .await
            .expect("spawn");

        // Wait for the session to complete (terminal, no subscribers).
        registry.await_terminal(&sid).await.expect("await_terminal");

        // Sleep beyond one full GC interval + TTL + slack so the GC has had at
        // least one opportunity to evict (R4 fallback: interval + ttl + 50ms).
        tokio::time::sleep(interval + ttl + Duration::from_millis(50)).await;

        registry.spawn_gc_task(ttl, interval);

        // Sleep again to let the newly spawned GC run at least one tick.
        tokio::time::sleep(interval + Duration::from_millis(50)).await;

        // The session must now be gone.
        assert!(
            matches!(registry.observe(&sid), Err(ObserveError::NotFound(_))),
            "session must be evicted after TTL + interval"
        );
    }

    // -----------------------------------------------------------------------
    // AC#5b — gc_does_not_evict_session_with_active_subscriber
    // -----------------------------------------------------------------------

    /// GC must NOT evict a session that still has active subscribers, even after
    /// TTL has elapsed.  Once the subscriber is dropped, subsequent GC ticks must
    /// evict the session.
    ///
    /// Covers: `broadcast::Sender::receiver_count` > 0 path (Crux #1 /
    /// concurrency-analysis §2 5b).
    #[tokio::test]
    async fn gc_does_not_evict_session_with_active_subscriber() {
        use algocline_core::execution::ObserveError;
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let ttl = Duration::from_millis(100);
        let interval = Duration::from_millis(50);

        let sid = registry
            .spawn_v2(simple_spec("return 2"), false)
            .await
            .expect("spawn");

        // Acquire a subscriber *before* the session reaches terminal.
        let _handle = registry.observe(&sid).expect("observe");

        // Wait for terminal while subscriber is still held.
        registry.await_terminal(&sid).await.expect("await_terminal");

        // Start GC — session has receiver_count > 0, must NOT be evicted.
        registry.spawn_gc_task(ttl, interval);

        // Sleep well beyond TTL + interval.
        tokio::time::sleep(interval + ttl + Duration::from_millis(50)).await;

        // Session must still be present (subscriber is alive).
        assert!(
            registry.observe(&sid).is_ok(),
            "session must NOT be evicted while a subscriber is held"
        );

        // Drop the subscriber — now eviction is permitted.
        drop(_handle);

        // Sleep for another interval + slack so GC ticks again after the drop.
        tokio::time::sleep(interval + Duration::from_millis(50)).await;

        // Now the session should be evicted.
        assert!(
            matches!(registry.observe(&sid), Err(ObserveError::NotFound(_))),
            "session must be evicted after subscriber is dropped and GC ticks"
        );
    }

    // -----------------------------------------------------------------------
    // AC#5c — gc_respects_interval_no_immediate_eviction
    // -----------------------------------------------------------------------

    /// GC must NOT evict a terminal session before the interval has fired,
    /// even if TTL has already elapsed.
    ///
    /// Covers: `tokio::time::interval MissedTickBehavior::Burst` guard
    /// (R4 / concurrency-analysis §2 5c).
    #[tokio::test]
    async fn gc_respects_interval_no_immediate_eviction() {
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        // Use a long interval so we can assert the session is still present
        // after TTL has elapsed but before an interval tick fires.
        let ttl = Duration::from_millis(20);
        let interval = Duration::from_millis(500);

        let sid = registry
            .spawn_v2(simple_spec("return 3"), false)
            .await
            .expect("spawn");

        registry.await_terminal(&sid).await.expect("await_terminal");

        // Start GC after TTL has elapsed — the first tick fires up to `interval`
        // from now, so we check immediately (well before the first tick).
        tokio::time::sleep(ttl + Duration::from_millis(10)).await;
        registry.spawn_gc_task(ttl, interval);

        // Check immediately — no tick has fired yet.
        assert!(
            registry.observe(&sid).is_ok(),
            "session must NOT be evicted before first GC tick fires"
        );
    }

    // -----------------------------------------------------------------------
    // AC#5d — test_atomic_last_active_updated_by_driver_loop
    // -----------------------------------------------------------------------

    /// Concurrent writer (store) and reader (load) on `last_active` with
    /// Relaxed ordering must not panic or cause UB; final value must be > 0.
    ///
    /// Covers: `AtomicI64::store` + `AtomicI64::load` Relaxed ordering safety
    /// under concurrent access (concurrency-analysis §2 5d / Crux #3 invariant).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_atomic_last_active_updated_by_driver_loop() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::Arc;

        let last_active = Arc::new(AtomicI64::new(0));

        let writer_la = Arc::clone(&last_active);
        let writer = tokio::spawn(async move {
            for _ in 0..1000 {
                writer_la.store(now_ms(), Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
        });

        let reader_la = Arc::clone(&last_active);
        let reader = tokio::spawn(async move {
            for _ in 0..1000 {
                let _ = reader_la.load(Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
        });

        writer.await.expect("writer task must not panic");
        reader.await.expect("reader task must not panic");

        // After 1000 stores of now_ms() the value must be > 0.
        assert!(
            last_active.load(Ordering::Relaxed) > 0,
            "last_active must be updated to a non-zero wall-clock value"
        );
    }

    // -----------------------------------------------------------------------
    // AC#5e — test_concurrent_observe_during_gc_tick
    // -----------------------------------------------------------------------

    /// 8 concurrent tasks each calling `observe()` 100 times while GC is running
    /// must produce only `Ok` or `Err(NotFound)` — never a panic.
    ///
    /// Covers: `RwLock::try_read` vs `RwLock::write` mutual exclusion +
    /// `Arc<RwLock<HashMap>>` clone safety (concurrency-analysis §2 5e).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_observe_during_gc_tick() {
        use algocline_core::execution::ObserveError;
        use std::sync::Arc;
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);
        let registry = Arc::new(registry);

        let ttl = Duration::from_millis(10);
        let interval = Duration::from_millis(5);

        let sid = registry
            .spawn_v2(simple_spec("return 42"), false)
            .await
            .expect("spawn");

        registry.await_terminal(&sid).await.expect("await_terminal");
        registry.spawn_gc_task(ttl, interval);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let reg = Arc::clone(&registry);
            let id = sid.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    match reg.observe(&id) {
                        Ok(_) | Err(ObserveError::NotFound(_)) => {}
                    }
                    tokio::task::yield_now().await;
                }
            }));
        }

        for h in handles {
            h.await.expect("concurrent observe task must not panic");
        }
    }

    // -----------------------------------------------------------------------
    // AC#5f — test_gc_task_spawn_survives_handle_drop
    // -----------------------------------------------------------------------

    /// `spawn_gc_task` internally drops the `JoinHandle` (legacy fire-and-forget).
    /// Verify the GC loop continues running after `spawn_gc_task()` returns by
    /// asserting eviction occurs after the expected window.
    ///
    /// Covers: `tokio::task::spawn` "JoinHandle drop ≠ task abort" contract
    /// (concurrency-analysis §2 5f / Crux #2 legacy parity).
    #[tokio::test]
    async fn test_gc_task_spawn_survives_handle_drop() {
        use algocline_core::execution::ObserveError;
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let ttl = Duration::from_millis(100);
        let interval = Duration::from_millis(50);

        // Start GC first — handle is immediately dropped inside spawn_gc_task.
        registry.spawn_gc_task(ttl, interval);

        let sid = registry
            .spawn_v2(simple_spec("return 99"), false)
            .await
            .expect("spawn");

        registry.await_terminal(&sid).await.expect("await_terminal");

        // Sleep long enough for at least 2 GC ticks after TTL.
        tokio::time::sleep(ttl + interval * 2 + Duration::from_millis(50)).await;

        // The GC task (whose JoinHandle was dropped) must have continued running
        // and evicted the session.
        assert!(
            matches!(registry.observe(&sid), Err(ObserveError::NotFound(_))),
            "session must be evicted by the GC task even after its JoinHandle was dropped"
        );
    }

    // -----------------------------------------------------------------------
    // AC#5g — test_arc_rwlock_hashmap_shared_across_clones
    // -----------------------------------------------------------------------

    /// `SessionRegistryV2: Clone` shares the same underlying
    /// `Arc<RwLock<HashMap>>`.  A session spawned via one clone must be visible
    /// from another clone, and GC started on one clone must evict sessions
    /// visible from the other.
    ///
    /// Covers: `Arc<RwLock<HashMap>>` Send + Sync + Clone shared-state contract
    /// (concurrency-analysis §2 5g).
    #[tokio::test]
    async fn test_arc_rwlock_hashmap_shared_across_clones() {
        use algocline_core::execution::ObserveError;
        use std::time::Duration;

        let executor = make_executor().await;
        let (registry_a, _tmp) = make_registry(executor);
        let registry_b = registry_a.clone();

        let ttl = Duration::from_millis(100);
        let interval = Duration::from_millis(50);

        // Spawn via registry_a.
        let sid = registry_a
            .spawn_v2(simple_spec("return 7"), false)
            .await
            .expect("spawn via registry_a");

        // Session must be visible from registry_b (shared Arc<RwLock<HashMap>>).
        assert!(
            registry_b.observe(&sid).is_ok(),
            "session spawned via registry_a must be visible from registry_b"
        );

        // Wait for terminal.
        registry_a
            .await_terminal(&sid)
            .await
            .expect("await_terminal");

        // Start GC via registry_b.
        registry_b.spawn_gc_task(ttl, interval);

        // Sleep long enough for eviction.
        tokio::time::sleep(ttl + interval + Duration::from_millis(50)).await;

        // Session evicted via registry_b's GC must be invisible from registry_a too.
        assert!(
            matches!(registry_a.observe(&sid), Err(ObserveError::NotFound(_))),
            "session evicted by registry_b GC must be gone from registry_a too"
        );
    }

    // -----------------------------------------------------------------------
    // usage_aggregate_none_for_run_without_llm_calls (test (b))
    // -----------------------------------------------------------------------

    /// When no `alc.llm` call occurs, `Done.usage` must be `None`.
    /// Verifies that the `on_paused` wiring does not falsely activate when no
    /// LLM call occurs, and that `usage_aggregate()` gates on `llm_calls > 0`.
    #[tokio::test]
    async fn usage_aggregate_none_for_run_without_llm_calls() {
        use algocline_core::execution::TerminalOutcome;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec("return 42"), false)
            .await
            .expect("spawn");

        let outcome = registry.await_terminal(&sid).await.expect("await_terminal");
        match outcome {
            TerminalOutcome::Done(result) => {
                assert_eq!(
                    result.usage, None,
                    "Done.usage must be None when no alc.llm call occurred"
                );
            }
            other => panic!("expected Done, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // usage_aggregate_some_for_run_with_llm_call (test (a))
    // -----------------------------------------------------------------------

    /// When `alc.llm` is called and resumed with host-reported usage,
    /// `Done.usage` must be `Some(TokenUsage { prompt_tokens: Some(10), completion_tokens: Some(5) })`.
    /// Verifies both `on_paused` wiring and `on_response_fed` propagation.
    #[tokio::test]
    async fn usage_aggregate_some_for_run_with_llm_call() {
        use algocline_core::execution::{ResumePayload, TerminalOutcome};
        use algocline_core::TokenUsage;

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec(r#"return alc.llm("q")"#), false)
            .await
            .expect("spawn");

        // Wait for Paused state.
        let mut retries = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if registry.state(&sid).await.expect("state").tag() == ExecutionStateTag::Paused {
                break;
            }
            retries += 1;
            assert!(retries < 500, "session did not reach Paused state");
        }

        // Resume with host-reported usage.
        registry
            .resume(
                &sid,
                ResumePayload::Single {
                    query_id: "q-0".into(),
                    response: "answer".into(),
                    usage: Some(TokenUsage {
                        prompt_tokens: Some(10),
                        completion_tokens: Some(5),
                    }),
                },
            )
            .await
            .expect("resume");

        let outcome = registry.await_terminal(&sid).await.expect("await_terminal");
        match outcome {
            TerminalOutcome::Done(result) => {
                assert_eq!(
                    result.usage,
                    Some(TokenUsage {
                        prompt_tokens: Some(10),
                        completion_tokens: Some(5),
                    }),
                    "Done.usage must reflect host-reported token counts"
                );
            }
            other => panic!("expected Done, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // usage_aggregate_uses_estimates_when_usage_omitted (test (d))
    // -----------------------------------------------------------------------

    /// When `alc.llm` is called but resumed with `usage: None`, `Done.usage`
    /// must be `Some` with non-zero estimated values (Estimated source from
    /// prompt length heuristic in `MetricsObserver::on_paused`).
    #[tokio::test]
    async fn usage_aggregate_uses_estimates_when_usage_omitted() {
        use algocline_core::execution::{ResumePayload, TerminalOutcome};

        let executor = make_executor().await;
        let (registry, _tmp) = make_registry(executor);

        let sid = registry
            .spawn_v2(simple_spec(r#"return alc.llm("q")"#), false)
            .await
            .expect("spawn");

        // Wait for Paused state.
        let mut retries = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if registry.state(&sid).await.expect("state").tag() == ExecutionStateTag::Paused {
                break;
            }
            retries += 1;
            assert!(retries < 500, "session did not reach Paused state");
        }

        // Resume without host-reported usage (observer uses Estimated values).
        registry
            .resume(
                &sid,
                ResumePayload::Single {
                    query_id: "q-0".into(),
                    response: "answer".into(),
                    usage: None,
                },
            )
            .await
            .expect("resume");

        let outcome = registry.await_terminal(&sid).await.expect("await_terminal");
        match outcome {
            TerminalOutcome::Done(result) => {
                let usage = result
                    .usage
                    .expect("Done.usage must be Some when alc.llm was called");
                assert!(
                    usage.prompt_tokens.unwrap_or(0) > 0
                        || usage.completion_tokens.unwrap_or(0) > 0,
                    "Done.usage must have non-zero estimated tokens, got: {usage:?}"
                );
            }
            other => panic!("expected Done, got: {other:?}"),
        }
    }
}
