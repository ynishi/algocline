# algocline-app 0.45.0

Service layer sitting between `algocline-engine` and `algocline-mcp`.

Hosts [`AppConfig`], [`AppService`], and the process pool ([`pool`])
providing UDS-based worker isolation for MCP tool calls. Also owns
the hub-projection preset resolution (`load_hub_projection_config`
and related config types) that translates preset selectors into
runtime-ready projection contexts.

## Modules

- [`pool`](pool.md): Process-pool IPC layer for `host_mode=true` execution.
- [`pool::client`](pool__client.md): UDS client for connecting to a pool worker process.
- [`pool::dispatch`](pool__dispatch.md): Host-mode dispatch helpers.
- [`pool::error`](pool__error.md): Error types produced by the process-pool IPC layer.
- [`pool::protocol`](pool__protocol.md): Wire protocol messages exchanged between the pool client (MCP
- [`pool::registry`](pool__registry.md): Pool registry: persistent session-to-worker mapping.

