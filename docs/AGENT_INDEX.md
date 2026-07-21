# Agent Index

Entry point for AI coding agents (Claude Code, Cursor, etc.) working on this
repository. Each line points to the canonical doc — read the linked file rather
than guessing from the 1-liner.

This file is the single curated index. The repo does **not** ship a `.claude/`
set; Claude Code spec / best-practices change frequently and we keep the
maintenance surface small. Pair this file with your own local `.claude/` if you
want one.

## Project Overview

- [README.md](../README.md) — what algocline is, install, quick start
- [CHANGELOG.md](../CHANGELOG.md) — release history; **BREAKING changes are
  documented inline here** (see `### Changed — **BREAKING**` headings)

## Development Workflow

- [justfile](../justfile) — `just ci` / `just ready` / `just test` / `just e2e`
  / `just check-invariants`. Recipes tagged `[group: 'agent']` are safe for
  agent invocation; `publish` is human-only
- [Cargo.toml](../Cargo.toml) — workspace layout (5 crates: `algocline-core` /
  `-engine` / `-app` / `-mcp` / root binary)

## Coding Conventions

- Service-layer error propagation — see `CLAUDE.md` §Service 層の Error 伝播規律
  (private; summary: `Result` must propagate to MCP wire, no `warn!`-and-drop)
- Doc language — all repo-committed docs are English (this file included)
- `.gitignore` — do not edit programmatically; the ignored local-infrastructure
  entries are mandatory and maintainer-managed

## MCP Surface

- [docs/design/mcp-support.md](design/mcp-support.md) — algocline-wide MCP
  capability adoption and non-standard design choices (Prompts scope etc.)
- [docs/mcp-resources.md](mcp-resources.md) — MCP resource catalog exposed by
  the server
- [docs/lua-stdlib.md](lua-stdlib.md) — `alc.*` Lua stdlib API reference
- [docs/state-management.md](state-management.md) — state primitive 2-layer
  split: Alc MCP namespace-generic CRUD + swarm-package domain methods;
  inherits 0.37.0 rollback constraints

## Package Authoring

- [docs/pkg-author-conventions.md](pkg-author-conventions.md) — naming, layout,
  required files, post-decommission convention (no `narrative.md`)
- [docs/bundled-packages-adoption-guide.md](bundled-packages-adoption-guide.md)
  — how bundled packages are sourced into `alc init`
- [docs/hub-gendoc-config.md](hub-gendoc-config.md) — Hub doc generation config

## Testing

- `tests/e2e.rs` — MCP tool E2E via `rmcp` client + child `alc` process. New
  MCP tools / params must add E2E cases here
- `cargo insta review` (or `just snapshots`) — review snapshot diffs

## Release

- Manual sequential `cargo publish` per crate (see `CLAUDE.md` §crates.io公開).
  `cargo release --execute` does **not** work due to internal workspace deps
  (cargo-release issue #691)
- `just publish <VERSION>` recipe encodes the order + 60s index-propagation
  sleep, but is **human-only**

## Out of Scope for This Index

- Internal protocols, accident logs, persona definitions, and workspace notes
  live in `workspace/` (gitignored) and `CLAUDE.md` (gitignored). Those are not
  part of the public contributor surface

## Maintenance

This file is verified by `just check-agent-index`. The recipe fails when:

1. A link target file does not exist
2. A file under `docs/*.md` is not referenced from this index (except
   `AGENT_INDEX.md` itself)

Fix either the link or add the missing doc to the appropriate section above.
