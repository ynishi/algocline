# algocline-mcp::prompts

MCP Prompts capability — workflow-trigger prompts.

algocline exposes a small, static set of Prompts that act as user-side
entry points into Tool-driven workflows. Each Prompt's `messages` body is
an instruction directed at the host LLM telling it to dispatch one or more
algocline MCP Tools (e.g. `alc_advice`, `alc_pkg_scaffold`) to complete
the workflow. See `docs/design/mcp-support.md` for the rationale behind
this scope (in particular, why per-package 1:1 mapping is not used).

## Types

- `PromptCatalog` — Catalog of workflow-trigger MCP prompts.

