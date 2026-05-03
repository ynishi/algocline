//! Process-pool IPC layer for `host_mode=true` execution.
//!
//! This module provides the foundation types for the pool worker protocol:
//!
//! - [`error`] — [`PoolError`] enum (thiserror-derived)
//! - [`protocol`] — [`PoolRequest`] / [`PoolResponse`] wire types (serde JSON line)
//! - [`client`] — [`PoolClient`] UDS client (connect + handshake + send_request)
//!
//! The `registry` module (registry.json persistence + GC) is added in Subtask 4.

pub mod client;
pub mod error;
pub mod protocol;

pub use client::PoolClient;
pub use error::PoolError;
pub use protocol::{PoolRequest, PoolResponse, PoolResponseData};
