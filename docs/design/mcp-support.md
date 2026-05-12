# MCP Support — algocline design notes

algocline implements the Model Context Protocol (MCP) as a server. This document
records decisions about which MCP capabilities algocline adopts and the
non-standard choices we make within each. Per-capability reference material
(URIs, JSON shapes, error codes) lives in the capability-specific docs cited
inline; this file captures the design rationale only when it differs from a
straightforward reading of the spec.

## Coverage

Strict feature-by-feature support matrix against the MCP 2025-06-18 spec.
"Status" values:

- **Full** — every wire-level message, notification, and capability sub-flag
  defined by the spec for this feature is implemented.
- **Partial** — the feature is declared as a capability and the primary
  request/response works, but some sub-flag, notification, or branch is not
  implemented; specifics are noted in the row.
- **Not yet** — capability is not declared and no implementation exists.
- **Not applicable** — the spec topic does not apply to algocline's deployment
  (e.g. HTTP auth for a STDIO-only server).

### Base protocol

| Spec topic | Status | Notes |
|---|---|---|
| [Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle) (`initialize` request + `notifications/initialized`; shutdown is transport-level per spec) | Full | Delegated to `rmcp` ServerHandler default. |
| Messages (JSON-RPC 2.0 requests / responses / notifications) | Full | Delegated to `rmcp`. |
| Versioning (capability negotiation, spec version exchange) | Full | Delegated to `rmcp`. |
| Transports — STDIO | Full | `rmcp` `transport-io` feature. |
| Transports — Streamable HTTP / SSE | Not yet | algocline-mcp ships as a STDIO server only. |
| [Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization) (OAuth 2.1 for HTTP) | Not applicable | Specified for HTTP transports; STDIO server retrieves credentials from environment per spec. |
| `_meta` field passthrough | Partial | Standard `rmcp` types accept `_meta` where defined by the spec; algocline does not emit custom `_meta` of its own. |

### Server features

| Spec topic | Status | Notes |
|---|---|---|
| [Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — `tools/list`, `tools/call` | Full | All algocline operations are exposed as Tools (see `service.rs`). |
| Tools — `notifications/tools/list_changed` | Not yet | Tool list is static at server startup; no install/remove path mutates it. |
| Tools — `outputSchema` / structured content | Partial | Some tools return JSON in text content; explicit `outputSchema` declarations and `structuredContent` round-trip are not yet provided per tool. |
| [Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources) — `resources/list`, `resources/read` | Full | See [mcp-resources.md](../mcp-resources.md). |
| Resources — `resources/templates/list` | Full | Eight URI templates exposed. |
| Resources — `notifications/resources/list_changed` | Not yet | Static resource list at V1; capability sub-flag not declared. |
| Resources — `resources/subscribe` / `notifications/resources/updated` | Not yet | Capability sub-flag not declared; client-side polling against `resources/read` is the supported pattern. |
| Resources — embedded resources in Tool results | Partial | rmcp structurally supports `type: "resource"` and `type: "resource_link"` content variants, but no algocline tool currently emits them — all tools return `Result<String, String>` so the macro produces only `RawContent::Text`. |
| [Prompts](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts) — `prompts/list`, `prompts/get` | Full | Static workflow-trigger set (`advice`, `new_package`); see the Prompts section below for scope rationale. |
| Prompts — `notifications/prompts/list_changed` | Not supported (intentional) | The spec defines this notification as scoped to changes in the `prompts/list` output ("SHOULD send when the list of available prompts changes"). algocline's Prompts surface is a fixed set of kicker workflows (`advice`, `new_package`) with no mutation path, so the SHOULD condition never holds and the notification has no role here. The capability sub-flag is declared `false` (i.e. omitted) and no emit sites exist. **This notification is independent of `notifications/resources/list_changed`**, which is governed by the `resources.listChanged` sub-flag and is the correct hook for changes to the resource list. |
| Prompts — embedded resources in `PromptMessage.content` | Not yet | Blocked on upstream rmcp serialization bug — see [rust-sdk#842](https://github.com/modelcontextprotocol/rust-sdk/issues/842) / [#843](https://github.com/modelcontextprotocol/rust-sdk/pull/843). |

### Server utilities

| Spec topic | Status | Notes |
|---|---|---|
| [Completion](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion) — capability `completions: {}` | Full | Declared. |
| Completion — `ref/resource` | Partial | Handled for resource template arguments where the algocline side has a candidate source; other arg names return an empty result with `total: 0`, `hasMore: false`. |
| Completion — `ref/prompt` | Not yet | Returns an empty completion result; the workflow-trigger arguments are free-form strings without a managed candidate set. |
| [Pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination) (cursor on `*/list` responses) | Partial | algocline's list responses are small enough to fit in one page and do not emit `nextCursor`; the spec permits this. Cursor support is not implemented for any list endpoint. |
| [Logging](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/logging) (`logging/setLevel`, `notifications/message`) | Not yet | algocline uses `tracing` internally; no `logging` capability is declared and no `notifications/message` is emitted. |

### Base utilities

| Spec topic | Status | Notes |
|---|---|---|
| [Ping](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/ping) | Full | `rmcp` default handler. |
| [Cancellation](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/cancellation) (`notifications/cancelled`) | Partial | `rmcp` routes cancellation notifications; individual algocline tool handlers do not yet cooperatively check cancellation for long-running operations. |
| [Progress](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress) (`notifications/progress` with `progressToken`) | Not yet | algocline does not emit progress notifications. |

### Client features (server-initiated requests to the client)

| Spec topic | Status | Notes |
|---|---|---|
| [Sampling](https://modelcontextprotocol.io/specification/2025-06-18/client/sampling) (`sampling/createMessage`) | Not yet | Tracked under #1776105556-75720. Required before Prompts/Tools can drive recursive host-LLM interactions on the server side. |
| [Roots](https://modelcontextprotocol.io/specification/2025-06-18/client/roots) (`roots/list`, `notifications/roots/list_changed`) | Not yet | algocline does not currently request roots from the client. |
| [Elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) (`elicitation/create`) | Not yet | algocline does not currently solicit additional user input via the client. |

## Prompts

### Scope (decision)

algocline exposes Prompts as **user-kickable workflow triggers** only. Each
Prompt's `messages` body is an instruction directed at the host LLM telling it
to dispatch one or more algocline MCP **Tools** (e.g. `alc_advice`,
`alc_pkg_scaffold`, `alc_eval`) to complete the workflow. The Prompts surface
is therefore the user-side entry point into Tool-driven workflows; the actual
Lua-strategy execution remains a Tools-layer responsibility.

The first two Prompts shipped under this scope:

| Prompt          | Arguments                                | Body intent (summary)                                                                              |
|-----------------|------------------------------------------|----------------------------------------------------------------------------------------------------|
| `advice`        | `task` (required)                        | "Use `alc_advice` to select a package for the given task, then call `alc_run` and summarise."     |
| `new_package`   | `name` (required), `category` (optional) | "Drive `alc_pkg_scaffold` interactively for a new package, confirming shape and source with user." |

`explain_pkg` and `eval_strategy` are candidates for later expansion once the
workflow-trigger pattern is validated on the two above.

### Why we do not map bundled packages to Prompts

The MCP spec prescribes no particular mapping between server-side artefacts
and Prompts. Mapping each installed bundled package (`sc`, `cot`, `panel`, …)
to its own Prompt of the same name is rejected for two reasons rooted in
algocline's execution model:

1. **No actual algorithm execution.** A Prompt response is `messages` only;
   the host LLM interprets the text. Compute-heavy strategies (`sc`'s N-path
   majority vote, `ucb`'s explore/exploit state) cannot be honoured by an LLM
   reading a templated instruction — the algorithm value lives in the Lua
   runtime, reachable only via `alc_run` (Tools).
2. **Wrong control axis.** Per spec, Prompts are user-controlled; the user
   selects them deliberately. Package selection in algocline is contextual —
   the AI decides which strategy fits the task. That fits Tools (model-controlled)
   semantics, not Prompts.

Surfacing strategies as Prompts therefore inverts algocline's purpose of
providing deterministic, prompt-comprehension-independent execution.

### Embedded resources (forward-looking)

Prompts may embed algocline Resources (`alc://packages/{name}/narrative`,
`alc://cards/{id}`, …) using `PromptMessage.content.type = "resource"` to
surface documentation alongside the workflow instruction. This depends on the
upstream rmcp fix tracked under [rust-sdk#842](https://github.com/modelcontextprotocol/rust-sdk/issues/842)
/ [rust-sdk#843](https://github.com/modelcontextprotocol/rust-sdk/pull/843);
until that lands, embedded resources are not used in returned messages.

## Resources

See [mcp-resources.md](../mcp-resources.md). The only non-standard choice
worth noting at the design level: algocline keeps `resources/subscribe`
explicitly unimplemented in V1; client-side polling against `resources/read`
is the supported pattern, and `listChanged` is reserved for future use when
the underlying `alc.toml` / installed-packages set is reshaped.

## Tools

No non-standard structural choices. Tools follow the spec shape (`tools/list`,
`tools/call`, `inputSchema`). Service-layer errors propagate to the wire as
typed JSON-RPC errors per the project-wide error-propagation discipline
recorded in `CLAUDE.md`; this is enforcement of a Rust-idiomatic pattern, not
a deviation from the MCP spec.
