# MCP Support — algocline design notes

algocline implements the Model Context Protocol (MCP) as a server. This document
records decisions about which MCP capabilities algocline adopts and the
non-standard choices we make within each. Per-capability reference material
(URIs, JSON shapes, error codes) lives in the capability-specific docs cited
inline; this file captures the design rationale only when it differs from a
straightforward reading of the spec.

## Capability adoption

| Capability     | Status           | Reference                                  |
|----------------|------------------|--------------------------------------------|
| `tools`        | Adopted          | `crates/algocline-mcp/src/service.rs`      |
| `resources`    | Adopted          | [mcp-resources.md](../mcp-resources.md)    |
| `prompts`      | Adopted (Phase 2 rework, see below) | this document       |
| `sampling`     | Not adopted yet  | tracked under #1776105556-75720            |
| `elicitation`  | Not adopted yet  | —                                          |
| `roots`        | Not adopted yet  | —                                          |
| `completion`   | Partially (`ref/resource` only) | implementation in `resources.rs` |

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
