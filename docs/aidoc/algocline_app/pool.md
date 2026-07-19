# algocline-app::pool

Process-pool IPC layer for `host_mode=true` execution.

This module provides the foundation types and dispatch helpers for the
pool worker protocol:

- [`error`] — [`PoolError`] enum (thiserror-derived)
- [`protocol`] — [`PoolRequest`] / [`PoolResponse`] wire types (serde JSON line)
- [`client`] — [`PoolClient`] UDS client (connect + handshake + send_request)
- [`registry`] — [`PoolRegistry`] persistence + GC (registry.json)
- [`dispatch`] — worker spawn + UDS proxy orchestration (ST6)

