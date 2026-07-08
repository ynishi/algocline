# plugins/alc-maint — algocline Core Maintenance Set

A Set for maintaining **the algocline repository itself**: keeping bundled
source tags in sync with upstream releases, and priming maintenance sessions
with a status dashboard. It is the maintainer-facing sibling of
[`plugins/alc`](../alc/README.md), which is the user-facing package
development DX set — the two audiences are deliberately kept in separate
plugins.

This set is only useful inside an algocline repository checkout. algocline
users (people who write Lua packages on top of algocline) do not need it.

## Components

| Layer | Name | Role |
|---|---|---|
| Skill (load only) | `/alc-maint-wake` | Injects the maintenance dashboard: bundled source drift snapshot (`src/init.rs` vs upstream latest tags), `CHANGELOG.md` Unreleased state, reflection SOPs, and worker dispatch paths |
| Agent (sync worker) | `@alc-bundled-sync` | Detects tag drift between `BUNDLED_SOURCES` and upstream releases; in `apply` mode edits the tags and appends a CHANGELOG draft entry; always emits the verification checklist. Edit-only — never installs, commits, or pushes |

## Usage

1. Run `/alc-maint-wake` at the start of a maintenance session — the drift
   table and CHANGELOG state land in context.
2. If drift is shown, dispatch `@alc-bundled-sync` with `mode: detect` for
   the detailed report, then decide whether to take the bump.
3. Dispatch `@alc-bundled-sync` with `mode: apply` — the agent edits
   `src/init.rs` and `CHANGELOG.md`, then returns the checklist.
4. Execute the checklist on the main thread: `cargo install --path .` →
   session restart → `alc update` → `alc_pkg_doctor`.
5. Commit through the repo's pre-commit gates on the maintainer's go sign.

## Boundaries (shared by every component in this set)

- **Edit scope**: `src/init.rs`, `Cargo.toml`, `CHANGELOG.md`, and test
  snapshots only. `.gitignore`, CI workflows, and the root `justfile` are
  never touched.
- **No irreversible operations**: no `git commit`, no `git push`, no
  `cargo publish`, no GitHub release. Agents stop at "edits applied +
  checklist emitted".
- **`AUTO_INSTALL_SOURCES` is out of scope**: the untagged MCP-side source
  list in `resolve.rs` is a separate mechanism from `BUNDLED_SOURCES` and is
  not modified by this set.

## Reflection

Definitions in this directory are markdown, read at Claude Code session
startup. After editing them: restart the session, then spawn the agent for a
real smoke check. `cargo install` does not reflect markdown changes.

## Roadmap

- `@alc-release-prep` (planned): mechanize the four-touch-point core version
  bump (workspace version, internal dependency versions, e2e snapshot,
  CHANGELOG) with a `cargo test` consistency check, stopping before commit.
