# algocline-mcp::req_registry

`ReqIdRegistry` — wire-layer adapter exclusive mapping from `RequestId` to [`SessionId`].

This module is intentionally confined to the `algocline-mcp` adapter crate.
Service-layer crates (`algocline-core`, `algocline-app`, `algocline-engine`) must never
import or reference any type from this module; wire concepts must not leak into the
service layer (Crux: `ReqIdRegistry wire-concept isolation`).

# Lock discipline

Every method acquires an `RwLock` guard, clones or mutates the value, and drops the guard
before the method returns. No guard is ever held across an `.await` point
(K-4 clone-then-release pattern; the `tokio::sync::RwLockReadGuard` / `RwLockWriteGuard`
are `!Send`, so holding one across `.await` in a `tokio::spawn` context would be a
compile error — this serves as the `test_req_rwlock_no_await_across_lock` compile-time gate).

## Types

- `ReqIdRegistry` — Owns the sole `RequestId` → [`SessionId`] mapping for in-flight MCP requests.
- `RequestId` — Wire-level request identifier. This is a re-export of rmcp's `NumberOrString` so that

