# algocline-app::pool::registry

Pool registry: persistent session-to-worker mapping.

[`PoolRegistry`] tracks live pool worker processes in
`~/.algocline/state/pool/registry.json`.  The file survives MCP-process
death so a restarted MCP can rediscover live worker sockets.

## Crux invariant (Registry reconnect across restarts)

`registry.json` is the **persistent source of truth**.  The in-memory
`PoolRegistry` value is a short-lived view — callers must reload from disk
after acquiring the advisory lock rather than caching across lock cycles.

## Functions

- `with_registry_lock` — Run `f` while holding an exclusive advisory lock on `lock_path`, using the

## Types

- `PoolRegistry` — In-memory view of `registry.json`.
- `PoolSessionEntry` — A single session entry in the pool registry.

