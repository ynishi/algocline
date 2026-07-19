# algocline-mcp::progress_forwarder

Progress forwarder task for the v2 MCP adapter.

Bridges the `ExecutionService` broadcast channel to MCP `ProgressNotification`
messages.  A forwarder is spawned **only** when `_meta.progressToken` is present
in the inbound request (Crux: ProgressToken-conditional forwarder spawn).

## Functions

- `spawn_progress_forwarder` — Spawn a background task that forwards execution progress events to the MCP client.

