//! `algocline-engine::execution` — v2 execution module.
//!
//! Provides the engine-level session lifecycle machinery:
//!
//! - [`SessionRecord`] — per-session ownership bundle (state, broadcast tx,
//!   cancellation token, join handle, response senders).
//! - [`SessionRegistryV2`] — registry that spawns, tracks, and controls v2
//!   sessions (coexists with the legacy `SessionRegistry` in `session.rs`).
//! - [`driver_loop`] — background async fn that drives a session to completion,
//!   embedding the four cooperative cancellation checkpoints (Crux R2).
//! - [`BroadcastObserverHandle`] — concrete [`algocline_core::execution::ObserverHandle`]
//!   implementation backed by a `broadcast::Receiver<ProgressEvent>` (Crux R3).
//!
//! # Design invariants
//!
//! - **Crux R1**: No `rmcp::*`, `progressToken`, `_meta`, `notifications/*`, or
//!   `mcp_`-prefixed identifiers in this module tree.
//! - **Crux R2**: Cancellation uses `CancellationToken::cancel()` only;
//!   `JoinHandle::abort()` and process-kill paths are absent.
//! - **Crux R3**: `observe()` is sync and returns a valid handle even with zero
//!   pre-registered observers (sink-free fan-out).

mod driver;
mod observer;
mod record;
mod registry;

pub use observer::BroadcastObserverHandle;
pub use record::SessionRecord;
pub use registry::SessionRegistryV2;
