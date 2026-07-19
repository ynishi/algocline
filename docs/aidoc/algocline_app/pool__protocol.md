# algocline-app::pool::protocol

Wire protocol messages exchanged between the pool client (MCP
process) and pool workers over Unix domain sockets.

Requests carry an `"op"` discriminant and are serialised as single
JSON lines. Responses mirror the same shape and are correlated by
the request ordering on a per-connection basis.

## Types

- `PoolRequest` — A message sent from the MCP process (pool client) to a pool worker.
- `PoolResponse` — A message sent from a pool worker back to the pool client.
- `PoolResponseData` — Data payload carried by a successful [`PoolResponse`].

