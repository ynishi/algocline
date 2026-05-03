//! Process-pool IPC layer for `host_mode=true` execution.
//!
//! This module provides the foundation types for the pool worker protocol:
//!
//! - [`error`] — [`PoolError`] enum (thiserror-derived)
//! - [`protocol`] — [`PoolRequest`] / [`PoolResponse`] wire types (serde JSON line)
//!
//! The `client` module (pool client / UDS connection handling) is added in Subtask 2.
//! The `registry` module (registry.json persistence + GC) is added in Subtask 4.

pub mod error;
pub mod protocol;

pub use error::PoolError;
pub use protocol::{PoolRequest, PoolResponse, PoolResponseData};
