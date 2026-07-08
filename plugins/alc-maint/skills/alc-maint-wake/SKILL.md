---
name: alc-maint-wake
description: Load-only Skill that injects algocline core-maintenance status (bundled source drift snapshot, CHANGELOG Unreleased state, reflection SOP pointers) into the main thread. Run at the start of a maintenance session inside the algocline repository checkout. Does not spawn any Agent.
---

# /alc-maint-wake — algocline Maintainer Wake

A context primer to run before maintenance work **on the algocline repository
itself** (bundled source bumps, version bumps, release preparation). The
sibling of `/alc-wake`, which primes package-development sessions; this one
primes core-maintenance sessions. **Does not spawn any Agent.**

## What It Does

1. **Load the maintenance fundamentals into context** (expands the
   Fundamentals section of this Skill body into the main thread).
2. **Snapshot bundled source drift** — read the `BundledSource` entries in
   `src/init.rs` and run `git ls-remote --tags <url>` per source (read-only,
   no clone); present a current-vs-latest table. On remote failure report
   `unknown` for that row and continue.
3. **Check `CHANGELOG.md` head** — report whether an `## [Unreleased]`
   section exists and, if so, which subsections carry entries.
4. **Guide worker paths** — `@alc-bundled-sync` dispatch examples (below).

The maintainer starts the session with that dashboard in context. Detected
drift is consumed by dispatching `@alc-bundled-sync` (detect first, apply on
decision); everything irreversible (install into the running environment,
commit, publish) stays on the main thread with the maintainer's go sign.

## Fundamentals (prose injected verbatim into the main thread)

### Two bundled-source systems — do not mix them up

- **`BUNDLED_SOURCES`** (`src/init.rs`): tagged sources installed by CLI
  `alc init` / `alc update`. Tag bumps happen here.
- **`AUTO_INSTALL_SOURCES`** (`crates/algocline-app/src/service/resolve.rs`):
  untagged sources pinned to each repo's default branch, used by the MCP
  `pkg_install` auto-resolution path. Not touched during a bundled bump.

### Reflection SOP (what makes an edit take effect)

- **Rust code** (including `src/init.rs` tag bumps): `cargo install --path .`
  then restart the MCP session. Editing the file alone changes nothing at
  runtime — the table is compiled into the binary.
- **Agent / Skill markdown** (`plugins/*`): restart the Claude Code session,
  then spawn the agent for a real smoke check. `cargo install` is irrelevant
  for markdown.
- **After a bundled tag bump**: `alc update`, then `alc_pkg_doctor` — expect
  no `incomplete_pkg` / `unregistered_pkg` / `symlink_dangling` findings.
  Multi-file packages are exactly what `incomplete_pkg` detection exists for.

### Core version bump touch points (all four, in one pass)

1. `Cargo.toml` — `workspace.package.version`
2. `Cargo.toml` — the four internal crate versions under
   `workspace.dependencies` (core / engine / app / mcp)
3. `tests/snapshots/e2e__alc_info.snap` — the `version` field
   (`cargo test` flags the mismatch; fix via `cargo insta review` or edit)
4. `CHANGELOG.md` — move `[Unreleased]` content under the new version entry

### Ownership boundary

Edits and verification runs are agent/AI work. `git commit` waits for the
maintainer's go sign and goes through the repo's pre-commit gates. Publishing
(`cargo publish`, tag push, GitHub release) is exclusively maintainer-driven
and out of scope for every agent in this set.

## Worker query paths

- Drift check only:
  `@alc-bundled-sync` with `mode: detect`
- Apply a decided bump:
  `@alc-bundled-sync` with `mode: apply` (optionally `sources: [<repo-name>]`)
- After apply, back on the main thread:
  `cargo install --path .` → session restart → `alc update` →
  `alc_pkg_doctor` → commit on go sign.
