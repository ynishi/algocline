# Bundled packages — narrative SSOT adoption guide

A handoff playbook for migrating an entire bundled-packages
repository (`algocline-bundled-packages`, currently 117 pkgs at
tag v0.21.0) onto the `M.docs` SSOT spec key shipped in algocline
v0.33.0.

This is the **operator-facing playbook**. For the per-pkg
authoring convention itself, see
[`pkg-author-conventions.md`](pkg-author-conventions.md). For the
runtime-side spec mechanism, see the cross-references in §6.

## Audience

You are reading this if you maintain
`algocline-bundled-packages` (or an equivalent collection repo)
and need to roll out `M.docs` declarations + Diátaxis-structured
narrative.md + rustdoc-flavoured docstrings across many pkgs in a
controlled way.

If you are authoring a single pkg from scratch, read
`pkg-author-conventions.md` instead — that doc tells you *what*
to write; this doc tells you *how to migrate an existing fleet*.

---

## 1. Why this matters

algocline v0.33.0 closed the **narrative SSOT vertical-slice**:
three layers that all key off the same `M.docs.narrative`
declaration in each pkg's `init.lua`.

```
   author writes              algocline ships               consumer sees
  ──────────────────         ────────────────────         ────────────────
   M.docs = {       ───►     pkg_resolve_      ───►       alc://packages/
     narrative =             narrative_path                 {name}/narrative
       "narrative.md"        (engine_api.rs)                (text/markdown)
   }                           │
   narrative.md                ├──►   resources.rs       ──►   IDE / agent
                               │      read_packages              text/markdown
                               │
                               └──►   alc_pkg_doctor    ──►   narrative_issues
                                      run_narrative_pass        bucket
                                                                (kind / severity)
```

Without a per-pkg `M.docs` declaration, all three layers fall
back to convention (`<pkg>/narrative.md` hard-coded path) and the
lint surface (`narrative_issues` bucket) flags the pkg as
`unmigrated` — a machine-readable signal that this guide is
designed to drive to zero.

The migration outcome is:

- Every bundled pkg has an explicit, machine-readable narrative
  declaration that downstream tools can trust without convention
  guessing.
- `alc_pkg_doctor narrative_issues` returns an empty array for
  every healthy pkg (no `unmigrated`, no `declared_missing`).
- Narrative content follows the Diátaxis 4-category structure so
  LLMs and humans get the same taxonomy when reading any pkg's
  reference.
- Docstrings inside `init.lua` follow the rustdoc-flavoured
  1-line summary discipline (arXiv 2510.26130 evidence:
  significant LLM code-gen pass-rate effect).

---

## 2. `M.docs` spec — field-by-field reference

The exact contract algocline 0.33.0 enforces.

### 2.1 Schema

```lua
M.docs = {
    narrative      = "narrative.md",  -- string?, pkg-dir-relative
    schema_version = 1,               -- number?, default 1
}
```

### 2.2 `narrative` (string, optional)

The relative path, **rooted at the pkg directory**
(`~/.algocline/packages/<name>/`), of the canonical narrative
markdown file.

**Constraints (enforced both at gendoc time and resource read
time)**:

- Must be a `string` if present (other types raise a typed error
  during gendoc PkgInfo extraction).
- Must **not** contain `..` (path-traversal guard).
- Must **not** start with `/` (absolute paths rejected).
- Empty string is treated as if `narrative` were absent — the
  resolver falls back to convention `narrative.md`.

**When omitted entirely** (no `M.docs` table, or `M.docs.narrative
= nil`): the resource resolver falls back to `<pkg>/narrative.md`
by convention. This is the path bundled-packages has used
historically; the migration is therefore additive and reversible.

**Recommended default**: `"narrative.md"` — keeps the file at the
same spot the convention fallback would look anyway, makes the
declaration explicit.

### 2.3 `schema_version` (number, optional, default 1)

Future-compat marker for the `M.docs` field set itself. The
current release accepts only `1` semantically; older algocline
versions that do not understand a future field can safely
`#[serde(default)]` past it. Always set `1` until the field set
is widened.

### 2.4 What `M.docs` is *not*

- **Not** a place for runtime contracts. Schema for
  `entries.run.{input, result}` belongs in `M.spec` — keep the
  separation clean (P3-4 design rationale).
- **Not** a place for Lua-internal multiline narrative. The
  narrative file is markdown on disk; embedding it in the Lua
  source was rejected during P3-4 design (`design.md §6` Option
  c).
- **Not** a place for category / description (those stay in
  `M.meta`).

---

## 3. Adoption procedure

The migration is split into three sub-steps that can be
independently scheduled and PR'd.

### 3.1 B-1 — Add `M.docs` declaration to every `init.lua`

The smallest, most mechanical step. Adds 3-5 lines per pkg.

**Insertion location**: after `M.spec` (or after `M.meta` when
the pkg has no `M.spec`), before `function M.run(...)`.

**Boilerplate** to insert:

```lua
M.docs = {
    narrative      = "narrative.md",
    schema_version = 1,
}
```

**PR strategy**:

- **Pilot**: pick 1-3 representative pkgs (e.g. `cot`, `sc`,
  `recipe_safe_panel`) and PR them first to validate the patch
  pattern + your CI behaviour with `alc_pkg_doctor narrative_issues`.
- **Sweep**: run a single batch PR over the remaining 114 pkgs
  with the boilerplate insert. A `sed` or `awk` one-liner is
  sufficient since the pattern is uniform.

**Validation after B-1**:

```bash
alc_pkg_doctor name=<pkg>
```

The pkg should now appear with `narrative_issues: []` (clean
state) **only if** the file `<pkg>/narrative.md` exists and
matches the declared name. Otherwise see §3.2 / §4.

### 3.2 B-2 — Restructure `narrative.md` content (Diátaxis sections)

The bigger, content-heavy step. Reformats existing
`narrative.md` files to follow the Diátaxis section convention.

**Required structure** (per `pkg-author-conventions.md §2`):

```markdown
# <pkg name>

<1–3 sentence overview, genre-agnostic>

## Tutorial      <!-- include only when applicable -->
## How-to        <!-- include only when applicable -->
## Reference     <!-- always recommended -->
## Explanation   <!-- include only when applicable -->
```

**Migration tactic**:

1. Cluster the 117 pkgs by archetype (orch / recipe / strategy /
   abm / etc.) so each cluster shares a content shape.
2. Per cluster, write a template that maps the pkg's existing
   narrative content onto the four section buckets.
3. PR cluster-by-cluster (5-15 pkgs per PR), since each cluster
   shares review concerns.

**Don't waste effort**:

- Small utility pkgs may only need `## Reference`. Don't
  manufacture `## Tutorial` content for pkgs that nobody learns
  by following along.
- The L-2 lint (heading violation detection) is **not** shipping
  in v0.33.0 — there is currently no machine validator for
  Diátaxis structure. Convention compliance is enforced by
  reviewer attention.

### 3.3 B-3 — Apply rustdoc-flavoured docstring discipline

Per-function: 1 imperative sentence ≤ 80 chars ending with a
period, blank `---` line, then optional body, then LuaCATS
annotations.

```lua
--- Compute the consensus answer from N sampled paths.
---
--- Generates n_paths independent reasoning chains, extracts the
--- answer from each, and returns the majority vote.
---
---@param ctx { task: string, n_paths: number? }
---@return { answer: string, votes: number }
function M.run(ctx) ... end
```

**Why this is a separate step**: docstring rewrites cannot be
mechanically batched (per-function attention required) and the
L-3 lint that would enforce this is also not in v0.33.0. The
payoff is documented but realised slowly: arXiv 2510.26130
shows significant LLM code-gen pass-rate uplift, materialising
gradually as the docstring corpus thickens.

**PR strategy**: opportunistic. Apply during any other PR that
touches the pkg, rather than scheduling a dedicated sweep.

---

## 4. Validation — `alc_pkg_doctor narrative_issues` bucket

The L-1 lint bucket added in algocline v0.33.0 (commit
`7b92d78`, issue #1778197805).

### 4.1 Output shape

```json
{
  "narrative_issues": [
    {
      "name": "cot",
      "kind": "unmigrated",
      "severity": "info",
      "resolved_path": "~/.algocline/packages/cot/narrative.md",
      "message": "convention narrative.md exists but M.docs is not declared (#1778197753 adoption candidate)",
      "suggestion": "Add M.docs = { narrative = \"narrative.md\", schema_version = 1 } to init.lua to make the SSOT explicit."
    }
  ]
}
```

### 4.2 Three classifications

| `kind` | `severity` | When fired | Action |
|---|---|---|---|
| `declared_missing` | `warn` | `M.docs.narrative` declared, file absent | Fix the path or create the file. Likely a typo in the declared name or a forgotten install of the narrative file. |
| `unmigrated` | `info` | `M.docs` not declared, convention `narrative.md` exists | Apply B-1: add the boilerplate `M.docs` block. |
| (silent — no entry) | — | Either both absent, or both present and matching | No action. |

### 4.3 Adoption progress measurement

Run with no `name` filter to enumerate all pkgs:

```bash
alc_pkg_doctor
```

Then count the `narrative_issues` array entries by `kind`:

```
unmigrated count       — pkgs still on convention fallback (B-1 backlog)
declared_missing count — pkgs broken by typo / missing file (must-fix before release)
```

The migration is **complete** when both counts hit zero across
the entire bundled fleet.

### 4.4 Lua executor lib_paths quirk to be aware of

`alc_pkg_doctor` runs a Lua eval per pkg via
`pkg_resolve_narrative_path` to extract `M.docs.narrative`. The
Lua executor's `package.path` is set at MCP server startup and
**does not refresh** after a mid-session `pkg_install`. This
means: if you `pkg_install` a fresh test pkg in the same MCP
session and immediately `pkg_doctor`, the narrative pass will
log `pkg load failed` and skip the pkg silently.

**Workaround for testing**: install the pkg before starting the
MCP server (the production scenario — `alc init` runs first,
then MCP starts). The `pre_install_pkg` helper in
`tests/e2e.rs` codifies this fixture pattern.

---

## 5. Edge cases

### 5.1 Pkgs that ship narrative under a non-default filename

If a pkg has historical reasons to keep narrative at, say,
`docs/intro.md` rather than `narrative.md`:

```lua
M.docs = {
    narrative      = "docs/intro.md",  -- pkg-dir-relative
    schema_version = 1,
}
```

Path components are joined by the resolver as
`<pkg dir>/docs/intro.md`. Subdirectories work. Just remember
the path-traversal guard: no `..`, no leading `/`.

### 5.2 Pkgs with no narrative content at all

For genuinely opaque pkgs that have no markdown narrative,
**omit `M.docs` entirely** (or omit just `narrative`):

```lua
M.docs = {
    schema_version = 1,
    -- narrative intentionally absent
}
```

Or simply do not declare `M.docs`. Both are accepted by the
spec.

`alc_pkg_doctor` will be silent (no `narrative_issues` entry)
because there is no `narrative.md` to flag and no declaration to
verify.

### 5.3 Author-edited narrative.md getting overwritten by `alc init --force`

`alc init --force` re-clones bundled-packages at the pinned tag
(`BUNDLED_SOURCES.tag` in `src/init.rs`) and copies
`<source>/docs/narrative/{name}.md` over the installed
`<dest>/{name}/narrative.md`. **Author edits in
`~/.algocline/packages/<name>/narrative.md` are not preserved.**

Edit narrative content in the bundled-packages source repo, not
in the installed cache. The installed cache is regenerable.

### 5.4 Partial adoption is safe

Adopting `M.docs` for some pkgs while leaving others on
convention is fully supported:

- Adopted pkgs: `narrative_issues` silent for them.
- Unadopted pkgs: `narrative_issues` includes them as
  `unmigrated/info`.
- The resource (`alc://packages/{name}/narrative`) works
  identically for both — adopted pkgs use the declared path,
  unadopted pkgs use convention fallback.

This means you can ship the migration in any order across the
117 pkgs without coordinated release.

### 5.5 Multi-file `M.docs` (future)

`schema_version = 1` reserves room for future fields (e.g.
`tutorial = "tutorial.md"` if Decide #2 in the P3-4 design ever
flips to Diátaxis-per-file split). Current implementation
ignores unknown fields when serde decoding via
`#[serde(default)]`, so a future bundled release that declares
`schema_version = 2` will degrade gracefully on older algocline
binaries.

---

## 6. Reference — algocline 本体 cross-links

| Spec / impl | File | Purpose |
|---|---|---|
| `EngineApi::pkg_resolve_narrative_path` trait method | `crates/algocline-core/src/engine_api.rs` | Public API surface bundled tools can rely on |
| `AppService::pkg_resolve_narrative_path` impl | `crates/algocline-app/src/service/pkg/read.rs` | Lua eval + path-traversal guard |
| `M.docs` spec extraction | `crates/algocline-app/src/service/lua/gendoc/docs/extract.lua` | Where `M.docs.narrative` is read at gendoc time |
| `Docs` entity schema | `crates/algocline-app/src/service/lua/gendoc/docs/entity_schemas.lua` | Strict schema enforced on PkgInfo.docs |
| `make_pkg_info` constructor | `crates/algocline-app/src/service/lua/gendoc/docs/pkg_info.lua` | 4-arg signature with `docs` |
| Resource path resolution | `crates/algocline-mcp/src/resources.rs` `read_packages` `[name, "narrative"]` arm | M.docs first, convention fallback |
| Doctor lint pass | `crates/algocline-app/src/service/pkg/doctor.rs` `run_narrative_pass` | `narrative_issues` bucket producer |
| `install_narrative_for` install hook | `src/init.rs` | `alc init --force` time copy from `<source>/docs/narrative/{name}.md` |
| Author convention spec | `docs/pkg-author-conventions.md` | What to write (this doc tells you how to migrate to that spec) |
| MCP resource catalog | `docs/mcp-resources.md` | Resource template surface incl. `alc://packages/{name}/narrative` |

### 6.1 Related issues (algocline repo unless noted)

- **#1778112139** (closed): the M.docs spec mechanism design (P3-4)
- **#1778197805** (closed): the L-1 lint mechanism (P3-5)
- **#1777052474** (closed): the bundled narrative resource (P3-3)
- **#1777032506** (closed): the related `alc init` types-distribution pattern this guide echoes
- **#1778197753** (open, project:algocline-bundled-packages): the bundled adoption work item this guide is the handoff for

---

## 7. Quick-start checklist

For the operator running the actual sweep:

- [ ] Read `pkg-author-conventions.md` §1 to internalise the
      pkg-shape contract
- [ ] Run `alc_pkg_doctor` against the current bundled install,
      capture the `unmigrated` count as the baseline
- [ ] B-1 pilot: PR `M.docs` block to `cot` / `sc` /
      `recipe_safe_panel` (3 pkgs)
- [ ] Verify pilot pkgs disappear from `narrative_issues`
- [ ] B-1 sweep: scripted batch PR for the remaining 114 pkgs
- [ ] B-2 cluster-by-cluster: rewrite `narrative.md` content per
      Diátaxis section convention
- [ ] B-3 opportunistic: apply docstring 1-line discipline as
      pkgs are touched
- [ ] Final validation: `alc_pkg_doctor` returns
      `narrative_issues: []` across the entire fleet
- [ ] Bump bundled-packages tag, document in CHANGELOG, update
      `BUNDLED_SOURCES.tag` in algocline next release cycle
