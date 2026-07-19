# algocline-app::pool::dispatch

Host-mode dispatch helpers.

Provides [`run_via_pool`] and [`continue_via_pool`] which orchestrate
worker subprocess spawning, UDS connection, registry persistence, and
response forwarding.

## Crux invariants enforced here

- (MCP thin proxy IPC boundary): all host_mode=true calls reach a worker
  subprocess via [`PoolClient`] over a Unix domain socket.  The MCP-internal
  [`SessionRegistry`] is never touched on this path.
- (Registry reconnect across restarts): every successful [`run_via_pool`]
  call persists the session entry to `registry.json` via
  [`with_registry_lock`] before returning.  A restarted MCP process can
  rediscover live workers from that file.

## Functions

- `continue_via_pool` — Reconnect to an existing pool worker via its registry entry and forward a
- `run_via_pool` — Spawn a worker subprocess, connect via UDS, proxy a `Run` request,

