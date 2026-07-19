# algocline-mcp 0.45.0

MCP (Model Context Protocol) server layer.

Exposes [`AlcService`] — the rmcp handler implementing MCP tools
(`alc_run` / `alc_continue` / `alc_status` / `alc_card_*` /
`alc_pkg_*` / `alc_hub_*` / `alc_eval_*` etc.), the [`PromptCatalog`]
and [`ResourceCatalog`], the request registry used for MCP sampling
continuations, and the [`progress_forwarder`] mapping engine
progress events to MCP progress notifications.

## Modules

- [`progress_forwarder`](progress_forwarder.md): Progress forwarder task for the v2 MCP adapter.
- [`prompts`](prompts.md): MCP Prompts capability — workflow-trigger prompts.
- [`req_registry`](req_registry.md): `ReqIdRegistry` — wire-layer adapter exclusive mapping from `RequestId` to [`SessionId`].
- [`resources`](resources.md): MCP Resources catalog for algocline.

