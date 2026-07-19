# algocline-app::pool::client

UDS client for connecting to a pool worker process.

[`PoolClient`] is the MCP-side (AppService-side) handle that opens a Unix
domain socket connection to a worker subprocess and exchanges
[`PoolRequest`] / [`PoolResponse`] messages in JSON-line format.

## Types

- `PoolClient` — A thin UDS client for communicating with a pool worker process.

## Constants

- `POOL_PROTOCOL_VERSION` — Version string embedded in every handshake to prevent client/server skew.

