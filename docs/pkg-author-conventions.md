# Package author conventions

This document codifies the conventions algocline expects from package
authors. It is the single source of truth for the docstring convention,
the `M.meta` / `M.spec` / `M.docs` shape contract, and the lint rules
enforced by `alc_hub_gendoc`.

---

## 0. SSoT / Projection — Source vs generated artifacts

algocline pkg conventions enforce a strict separation between
**Source** (hand-written) and **Projection** (machine-generated):

- **Source (SSoT)**: `init.lua` — specifically `M.meta`, `M.spec`,
  and the leading `---` docstring block. This is the only file an
  author edits.
- **Projection**: every other artifact about the pkg, regenerated
  by `alc_hub_dist` from the Source.

Recognised projections (current set; the authoritative list lives in
`crates/algocline-app/src/service/lua/gendoc/docs/projections.lua`):

| Projection target            | `alc_hub_dist` projections key | Purpose                                  |
|------------------------------|--------------------------------|------------------------------------------|
| `docs/narrative/{name}.md`   | `narrative`                    | Rendered docstring + Parameters table    |
| `llms.txt` / `llms-full.txt` | `llms`                         | LLM-readable pkg catalog                 |
| `context7.json`              | `context7`                     | Context7 MCP integration                 |
| `.devin/wiki.json`           | `devin`                        | Devin AI wiki ingestion                  |
| `types/alc_pkgs.d.lua`       | `luacats`                      | LuaCATS type stubs for IDE               |

**Hand-writing or hand-editing any projection is forbidden.** A
discrepancy between Source and Projection always means either
(a) a stale projection that needs regeneration, or (b) a Source
write disguised as a projection edit. The lint pipeline assumes
Source is canonical.

The legacy `M.docs.narrative` field and standalone `narrative.md`
files were transitional violations of this principle — projections
that authors edited by hand. They are now fully generated.

---

## 1. Publish patterns and lint scope

Pkg authors fall into one of three publish patterns. Each pattern has
a different lint posture; the same lint codes (§5) are evaluated, but
the severity boundary that fails a build differs.

| Pattern       | Distribution                                                        | Lint mode                | Author expectation                                                                                |
|---------------|---------------------------------------------------------------------|--------------------------|---------------------------------------------------------------------------------------------------|
| **Bundled**   | Shipped via `algocline-bundled-packages` (the official curated set) | `lint_strict=true`       | Required + Recommended fields all populated. Every warning is treated as an error.                |
| **Community** | Personal repos, gists, hand-installed packages                      | `lint_strict=false`      | Required fields populated. Recommended fields strongly encouraged but warnings are non-blocking.  |
| **Private**   | Local-only experiments, never published                             | Lint not expected to run | No baseline. Authors may run lint at their discretion.                                            |

`lint_strict=true` (Bundled / CI gate) is the only mode that fails
the build on warnings. `lint_strict=false` (default) allows warnings
to surface in tooling output without blocking the build.

After lint passes, `alc_hub_dist` (the **DIST** stage) regenerates
every projection listed in §0.

The Required vs Recommended boundary is defined in §2 (`M.meta`
fields) and §3 (docstring shape). All conventions in this document
apply to all three patterns; only the lint posture differs.

---

## 2. Top-level pkg shape

Every algocline pkg is a Lua module that returns a table `M`. The canonical layout (taken from `cot/init.lua` and trimmed):

```lua
local S = require("alc_shapes")
local T = S.T

local M = {}

---@type AlcMeta
M.meta = {
    name        = "cot",
    version     = "0.1.0",
    description = "Iterative chain-of-thought — cumulative reasoning steps, then synthesis",
    category    = "reasoning",
}

---@type AlcSpec
M.spec = {
    entries = {
        run = {
            input = T.shape({
                task  = T.string:describe("The question or task to reason about"),
                depth = T.number:is_optional():describe("Number of reasoning steps (default: 3)"),
            }),
            result = T.shape({
                chain      = T.array_of(T.string):describe("Ordered insights, one per reasoning step"),
                conclusion = T.string:describe("Synthesized final answer"),
            }),
        },
    },
}

-- M.docs is optional; set schema_version for future compat if needed.
-- Do NOT set M.docs.narrative — that field has been removed.
M.docs = {
    schema_version = 1,
}

function M.run(ctx)
    -- ...
end

return M
```

Each field below is tagged with one of three status labels. The same labels are used throughout this document.

| Status          | Meaning                                                                                                                                                       |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Required**    | Must be present. Missing it is a lint error in every publish pattern (§1).                                                                                    |
| **Recommended** | Not required, but strongly encouraged. **Bundled** pkgs are expected to populate it; missing it is a warning by default and an error under `lint_strict=true`. |
| **Optional**    | May be omitted. No lint emission.                                                                                                                             |

### 2.1 `M.meta` — identity

| Field         | Status       | Type   | Notes                                                          |
|---------------|--------------|--------|----------------------------------------------------------------|
| `name`        | **Required** | string | Must match the pkg directory name (lint: `E_NAME_MISMATCH`).   |
| `version`     | **Required** | string | SemVer.                                                        |
| `description` | **Required** | string | One-line tagline (≤ 80 char). Projected into `llms.txt` entry. |
| `category`    | **Required** | string | Grouping key for `llms.txt` and hub search.                    |

(Lint codes: `E_META_MISSING_{NAME,VERSION,DESCRIPTION,CATEGORY}` — see §5.)

### 2.2 `M.spec` — runtime contract

`M.spec.entries.{entry}.{input, result}` declares each entry's
runtime contract using the `alc_shapes` T DSL. This is the
**SSoT for `## Parameters` projections** (§3.4) and for
`alc.run(<pkg>, ctx)` runtime validation.

| Field                                | Status          | Type                            | Notes                                                                                                                              |
|--------------------------------------|-----------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `M.spec.entries.{entry}.input`       | **Recommended** | `T.shape(...)`                  | Each field MUST carry `:describe("...")`. The `:describe` text becomes the `description` column of the `## Parameters` projection. |
| `M.spec.entries.{entry}.result`      | **Recommended** | `T.shape(...)` or `T.ref(name)` | Consumed by `alc_hub_dist projections=["luacats"]` for IDE type stubs.                                                             |

A pkg without `M.spec` is treated as **opaque**: downstream
generators emit "no declared shape" sections. This is acceptable
for **Community** and **Private** patterns but is rejected by
`lint_strict=true` for **Bundled** pkgs.

> **Note on naming**: the Lua-source location is
> `M.spec.entries.{entry}.input`. The name `input_shape` appears
> in `alc_hub_dist`'s JSON projection output (the field key in the
> generated JSON), but is **not** the source location. Do not
> write `M.meta.input_shape`; that phrasing is stale and the
> linter does not recognise it.

(Lint codes for V1: `E_META_MISSING_INPUT_SHAPE` / `E_PARAM_MISSING_DESCRIBE` — see §5.)

### 2.3 `M.docs` — optional container

`M.docs` is preserved as a container for future schema markers.
The only recognised field is `schema_version`. **`M.docs.narrative`
is removed** — do not set it. The linter will warn on unknown
`M.docs` fields in a future release.

---

## 3. Docstring-driven narrative

`init.lua` is the single source of truth for narrative content. The
gendoc pipeline (`extract.lua split_sections` + `projections.lua
narrative_md`) renders the docstring H2 sections into Markdown on
demand. The output is the `narrative` projection (§0).

**Removed conventions** (do not use):

- `M.docs.narrative = "narrative.md"` — field is gone
- Separate `narrative.md` files — not recognised; will be ignored

### 3.1 How the resource is served

When a client requests `alc://packages/{name}/narrative`:

1. The engine locates the pkg's `init.lua` (variant scope first, then global scope).
2. A fresh mlua VM runs `extract.build_pkg_info` + `projections.narrative_md`.
3. The rendered Markdown is returned as `text/markdown`.

No cache — every request re-renders from the current `init.lua`.

### 3.2 Recommended H2 sections

The H2 sections below map onto the four Diátaxis documentation
classes. Write only the sections relevant to the pkg — a simple
wrapper may need only `## Usage`; a paper implementation typically
needs `## Algorithm` + `## Theoretical foundations` + `## References`.

| Section                               | Diátaxis class | When to write                                                                                                                            |
|---------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `## Usage`                            | How-to         | Minimum working example. Write for almost every pkg.                                                                                     |
| `## When to use`                      | How-to         | When the pkg overlaps with siblings and the choice is non-obvious.                                                                       |
| `## Algorithm`                        | Explanation    | When the algorithm has 3+ named steps or a non-trivial invariant.                                                                        |
| `## Theoretical foundations`          | Explanation    | When correctness follows from a theorem or paper. State the theorem.                                                                     |
| `## Entry contract`                   | Reference      | When the pkg exposes multiple named entries.                                                                                             |
| `## Caveats`                          | Reference      | Pkg-wide rationale that does not fit a single parameter `:describe()` — known limitations, why certain knobs are hidden, edge cases.     |
| `## Empirical validation`             | Reference      | Bench data, sweep results, coverage observations.                                                                                        |
| `## Comparison with related packages` | Reference      | When a sibling pkg does something similar. One bullet per peer.                                                                          |
| `## References`                       | Reference      | Papers, arXiv IDs, books. Bullet list — see §4.4.                                                                                        |

`## Parameters` is **not** in the list above — it is generated from
`M.spec.entries.{entry}.input` (§2.2 / §4.3) and must not be
hand-written.

### 3.3 Canonical example

The reference docstring below demonstrates the required elements:

```lua
--- conformal_vote — split conformal prediction gate for multi-agent deliberation
---
--- Linear opinion pool + split conformal prediction post-hoc decision layer.
--- Emits a three-way decision (commit / escalate / anomaly) with a finite-sample
--- coverage guarantee `Pr[Y ∈ C(X)] ≥ 1-α` (Theorem 2). Calibration and online
--- rounds share aggregation weights so exchangeability is preserved.
---
--- ## Algorithm
---
--- Given N agents that each emit a verbalized probability distribution
--- π_i(y|x) over a fixed option set, the pkg performs:
---
--- ```math
--- P_social(y|x) = Σ_i w_i · π_i(y|x)        (linear opinion pool)
--- s_nc(x, y)    = 1 - P_social(y|x)          (nonconformity score)
--- q̂            = sorted[⌈(n+1)(1-α)⌉]        (finite-sample quantile, §4.3)
--- C(x)          = { y : P_social(y|x) ≥ 1 - q̂ }   (prediction set)
--- ```
---
--- ## Theoretical foundations
---
--- Theorem 2 guarantees `Pr[Y ∈ C(X)] ≥ 1-α` in finite samples whenever
--- calibration and online rounds share the same aggregation weights and
--- the data is exchangeable.
---
--- ## Entry contract
---
--- - `calibrate`   — pure, direct-args. returns `{ q_hat, tau, alpha, n, weights }`
--- - `aggregate`   — pure, direct-args. returns `{ [label] = p_social }`
--- - `predict_set` — pure, direct-args. returns `{ labels, top1, top1_prob, ... }`
--- - `decide`      — pure, direct-args. returns `{ action, selected }`
--- - `run`         — Strategy, ctx-threading. queries N agents via `alc.llm`
---
--- ## Comparison with related packages
---
--- Category: `validation` (alongside `sprt`, `eval_guard`, `inverse_u`).
---
--- ## References
---
--- Wang, Xie, Wang, Gao, Yang, Li, Qiu, Han, Qiu, Huang, Zhu, Woo (2026).
--- "From Debate to Decision: Conformal Social Choice for Safe Multi-Agent
--- Deliberation". arXiv:2604.07667.
```

---

## 4. Docstring style

### 4.1 1-line summary and abstract

Every docstring opens with a 1-line summary, followed by a 1–3 sentence abstract:

```lua
--- {PkgName} — {verb phrase}
---
--- {abstract: 1-3 sentences explaining the core capability}
---
--- ## {first H2 section}
```

**1-line summary** (line 1):

- One clause, ≤ 80 characters, of the form
  `{PkgName}({StyledName}) — {verb phrase}` when the pkg has a stylized
  name (typically a paper-cited abbreviation such as `CoT`, `UCB`,
  `MCTS`), or `{PkgName} — {verb phrase}` when the pkg directory name
  is already the canonical form
- `PkgName` is always the pkg directory name (lowercase, matches
  `M.meta.name`). `StyledName` is the conventional reading-aid form
  used in the literature; omit it when no such form exists.
- `—` is the em dash (UTF-8 `U+2014`)
- Becomes the H1 title in the rendered narrative and the entry in `llms.txt`
- `M.meta.description` (§2.1) carries the same (or slightly expanded) wording for JSON consumers

Good:

```lua
--- conformal_vote — split conformal prediction gate for multi-agent deliberation
--- cot(CoT) — iterative chain-of-thought reasoning
--- ucb(UCB) — upper confidence bound multi-armed bandit
```

Too vague:

```lua
--- conformal_vote — a useful voting package
```

**Abstract** (lines after the blank `---` separator, until the first `## ` heading):

- 1 to 3 sentences
- Plain prose only — no headings, no lists, no code fences
- Explains the core capability at a glance; rationale and algorithm details belong in the H2 sections (§3.2)

If the pkg is trivial enough that no H2 sections follow, the abstract may stand alone — the rendered narrative will then consist of H1 + abstract only.

### 4.2 Markdown syntax

Within the docstring body, follow these Markdown rules. Concrete examples follow each rule.

**Headings**:

- `#` (H1) — **forbidden**. The generator synthesises H1 from the 1-line summary (§4.1).
- `##` is the highest permitted level
- `###` is allowed as a subsection within a `##` block
- `####` and lower — forbidden
- Each heading must be followed by a blank `---` line before body text:

```lua
--- ## Algorithm
---
--- 1. step
```

**Code fences**:

- Use **explicit** triple backticks (`` ``` ``). 4-space-indent fences are forbidden
- Language hint is recommended (`` ```lua `` for Lua, `` ```math `` for GitHub MathJax-rendered equations)
- One snippet per fence. Multiple snippets must use independent fences
- Inline code uses single backticks (`` ` ``)

```lua
--- ## Usage
---
--- ```lua
--- local pkg = require("pkg")
--- return pkg.run(ctx)
--- ```
```

**Lists**:

- Bullet: `-` (hyphen) only. `*` and `+` are forbidden
- Numbered: `1.` / `2.` / `3.` style
- Letter-numbering (`a.` / `b.`) is renderer-dependent and forbidden
- Indentation: 2 spaces for sub-bullets under `-`, 3 spaces for sub-bullets under numbered lists
- Maximum nesting depth: 2 levels. 3+ levels of nesting are forbidden — renderer-dependent

**Links**:

- GitHub-style markdown links only: `[text](url)`
- Bare URLs are forbidden in body text; they are permitted **only** inside `## References` citations (§4.4)

```lua
--- See [the README](../README.md) for derivation.
```

**Encoding**:

- UTF-8 only
- em dash `—` (`U+2014`) and en dash `–` (`U+2013`) are allowed
- All docstrings are written in English. The rendered narrative and `llms.txt` are public artifacts; non-English docstrings break readability for downstream consumers.

### 4.3 Parameters — generated from `M.spec`

The `## Parameters` section in the rendered narrative is **machine-generated** from `M.spec.entries.{entry}.input` (§2.2). Authors do not write `## Parameters` in the docstring.

The generator emits one row per shape field, using:

- field name as `key`
- shape type as `type`
- `:is_optional()` as `required` (true / false)
- `:describe("...")` text as `description`

**`:describe()` requirements**:

Every shape field MUST carry `:describe("...")`. The describe text is the only source for the `description` column.

```lua
M.spec = {
    entries = {
        run = {
            input = T.shape({
                task  = T.string:describe("The question or task to reason about"),
                depth = T.number:is_optional():describe("Number of reasoning steps (default: 3)"),
            }),
        },
    },
}
```

In **Bundled** mode (`lint_strict=true`), missing `:describe()` is a hard error. In **Community** mode, missing `:describe()` is a warning (lint code `E_PARAM_MISSING_DESCRIBE` — see §5).

**Where to put the rationale** (two-tier rule):

| Scope of rationale                                                                                              | Where to write                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Per-parameter semantic rationale (e.g. *why* a numeric default was chosen, when truncation breaks invariants)   | Inside the parameter's `:describe("...")` text. The text may be long; the projection emits it verbatim into the `## Parameters` table.      |
| Pkg-wide rationale (e.g. *why* a token budget knob is hidden, why certain options are intentionally not exposed) | `## Caveats` H2 section (§3.2)                                                                                                              |

This separation ensures (a) parameter-specific rationale survives every projection (`## Parameters` table, `context7.json`, `llms-full.txt`, LuaCATS stubs), and (b) pkg-wide design rationale lives where readers look for caveats.

**Hand-written `## Parameters` is forbidden**:

Writing `## Parameters` in the docstring while `M.spec.entries.{entry}.input` is also declared raises lint error `E_PARAMETERS_CONFLICT`. The two are mutually exclusive: `M.spec` is the SSoT.

For pkgs without `M.spec` (opaque pkgs — Community / Private), the generator emits no `## Parameters` section. Authors do not need to write one.

### 4.4 References

The `## References` H2 section uses a flat bullet list. Do not wrap citations in code fences.

- Each citation is a single bullet starting with `-`
- Continuation lines indent by 2 spaces
- Bare URLs are permitted **only** inside this section (e.g. arXiv links, paper DOIs)

```lua
--- ## References
---
--- - Friedman, M. (1937). "The use of ranks to avoid the assumption of
---   normality ...," J. Am. Stat. Assoc. 32(200): 675–701.
--- - Wang, X. et al. (2026). "From Debate to Decision: Conformal Social
---   Choice for Safe Multi-Agent Deliberation". arXiv:2604.07667.
```

Inline citations in body text reference the bullet by surname or arXiv ID — no inline URL.

### 4.5 LuaCATS annotations

LuaCATS annotations (`---@type`, `---@param`, `---@return`, etc.) appear **after** the narrative docstring body. The generator stops narrative extraction at the first `---@` line.

```lua
--- cot(CoT) — iterative chain-of-thought reasoning
---
--- Builds a reasoning chain step by step, then synthesizes the chain
--- into a single coherent conclusion.
---
--- ## Usage
---
--- ```lua
--- local cot = require("cot")
--- return cot.run({ task = "Why is the sky blue?", depth = 3 })
--- ```

local S = require("alc_shapes")
local T = S.T

local M = {}

---@type AlcMeta
M.meta = {
    name        = "cot",
    version     = "0.1.0",
    description = "Iterative chain-of-thought — cumulative reasoning steps, then synthesis",
    category    = "reasoning",
}

---@type AlcSpec
M.spec = {
    entries = {
        run = {
            input = T.shape({
                task  = T.string:describe("The question or task to reason about"),
                depth = T.number:is_optional():describe("Number of reasoning steps (default: 3)"),
            }),
            result = T.shape({
                chain      = T.array_of(T.string):describe("Ordered insights, one per reasoning step"),
                conclusion = T.string:describe("Synthesized final answer"),
            }),
        },
    },
}

---@param ctx AlcCtx
---@return AlcCtx
function M.run(ctx) ... end

return M
```

Place them last so the narrative reads as a single coherent block when stripped of annotations. LuaCATS is consumed by `alc_hub_dist projections=["luacats"]` to generate IDE type stubs (§0).

### 4.6 Disallowed constructs (collected)

The constructs below are forbidden in any docstring. Each is also covered in the relevant subsection above; this table consolidates them with the corresponding lint code (§5).

| NG                                            | Reason                                                            | Lint code                  |
|-----------------------------------------------|-------------------------------------------------------------------|----------------------------|
| `#` (H1) heading                              | Generator synthesises H1 from the 1-line summary (§4.1)           | `E_H1_IN_DOCSTRING`        |
| `$...$` / `$$...$$` inline LaTeX              | Renders only on GitHub; breaks elsewhere. Use `` ```math `` fence | (no lint, manual review)   |
| 4-space-indent code fence                     | Use explicit `` ``` `` only                                       | (no lint, manual review)   |
| HTML tag (`<br>`, `<sub>`, etc.)              | Markdown only                                                     | (no lint, manual review)   |
| Emoji                                         | Breaks llms.txt and CRAN-style consumers                          | (no lint, manual review)   |
| 3+ levels of list nesting                     | Renderer-dependent                                                | (no lint, manual review)   |
| Hand-written `## Parameters` heading          | `M.spec.entries.{entry}.input` is the SSoT (§4.3)                 | `E_PARAMETERS_CONFLICT`    |

---

## 5. Lint rules

`alc_hub_gendoc` runs the lint pipeline and emits the codes below. `lint_strict=true` (Bundled / CI gate, §1) treats every `error` severity as a build failure; `lint_strict=false` (default) surfaces them as non-blocking diagnostics.

| Code                         | Severity        | Status       | Description                                                                                |
|------------------------------|-----------------|--------------|--------------------------------------------------------------------------------------------|
| `E_META_MISSING_NAME`        | error           | active       | `M.meta.name` missing                                                                      |
| `E_META_MISSING_VERSION`     | error           | active       | `M.meta.version` missing                                                                   |
| `E_META_MISSING_DESCRIPTION` | error           | active       | `M.meta.description` missing                                                               |
| `E_META_MISSING_CATEGORY`    | error           | active       | `M.meta.category` missing                                                                  |
| `E_NAME_MISMATCH`            | error           | active       | `M.meta.name` ≠ pkg directory name                                                         |
| `E_H1_IN_DOCSTRING`          | error           | active       | `#` heading present in docstring                                                           |
| `E_PARAMETERS_CONFLICT`      | error           | active       | `M.spec.entries.{entry}.input` declared and hand-written `## Parameters` both present      |
| `E_RESULT_CONFLICT`          | error           | active       | `M.spec.entries.{entry}.result` declared and hand-written `## Result` both present         |
| `E_META_MISSING_INPUT_SHAPE` | warning / error | planned (V1) | `M.spec.entries.{entry}.input` missing. `warning` in default mode, `error` under `lint_strict=true`. |
| `E_PARAM_MISSING_DESCRIBE`   | warning / error | planned (V1) | Shape field without `:describe(...)`. `warning` in default mode, `error` under `lint_strict=true`.   |
| `W_DESCRIPTION_MULTILINE`    | warning         | active       | `M.meta.description` contains a newline                                                    |
| `W_FAKE_LABEL`               | warning         | active       | `Usage:` / `Args:` style label — promote to `## Usage` etc.                                |
| `W_EMPTY_NARRATIVE`          | warning         | active       | No abstract and no H2 sections                                                             |

**Severity convention** (per `lint.lua`):

- `error` — `lint_strict=true` rejects the pkg; `lint_strict=false` reports non-blocking
- `warning` — never blocks the build, surfaces in tooling diagnostics
- Planned rules with split severity emit `warning` in default mode and **promote to `error`** under `lint_strict=true`

Active rules are implemented in `crates/algocline-app/src/service/lua/gendoc/docs/lint.lua`. Planned rules are not yet implemented.

---

## 6. Migration

### 6.1 Migration from `M.docs.narrative` / `narrative.md`

If an existing pkg uses the old (pre-narrative-decommission) convention:

1. Move narrative content into the `---` docstring block as H2 sections.
2. Remove `M.docs.narrative` from the `M.docs` table (or drop `M.docs` entirely if `schema_version` is also absent).
3. Delete the standalone `narrative.md` file.
4. Run `alc_hub_dist` with `projections=["narrative"]` to regenerate `docs/narrative/{name}.md`.
5. Verify `alc://packages/{name}/narrative` returns the expected Markdown.

### 6.2 Migration to V1 conventions

For pkgs predating the V1 conventions in this document:

1. **Move parameters into `M.spec`**: if the pkg declared `M.meta.input_shape` (a stale phrasing) or had a hand-written `## Parameters` H2, move the shape definition into `M.spec.entries.{entry}.input` (§2.2). Delete the `## Parameters` H2.
2. **Add `:describe()` to every shape field**: each field in `input` and `result` must carry a `:describe("...")` clause. Fields without describe text become empty rows in the projected `## Parameters` table.
3. **Move pkg-wide rationale to `## Caveats`**: prose explaining *why* certain knobs are hidden, why a token budget is fixed, etc., goes into the `## Caveats` H2 (§3.2). Per-parameter rationale stays inside `:describe()` (§4.3).
4. **Run lint**: `alc_hub_gendoc lint_strict=false` first to see warnings, then `lint_strict=true` if the pkg targets the Bundled distribution (§1).
5. **Regenerate projections**: `alc_hub_dist` to refresh `docs/narrative/{name}.md` and downstream artifacts (§0).

## 7. 1-pkg authors: publishing a single package to Hub

If your repository contains exactly one algocline package, follow these steps
to make it discoverable via `alc_hub_search`.

### Layout requirement

Place your package at `<repo>/<pkg_name>/init.lua` (nested), **not** at the
repository root. Example:

```
my-cool-pkg/
├── my_cool_pkg/
│   └── init.lua       # M.meta, M.spec, M.run
├── alc.toml
└── hub_index.json     # generated by `alc_hub_dist`
```

### Minimal `alc.toml`

```toml
[hub]
# Default values are sufficient for a single-package repo. Optional sections
# [hub.context7] and [hub.devin] customize projection targets — see
# docs/hub-gendoc-config.md.
```

### Publishing

From a Claude Code / rmcp MCP session in your repo root:

```
alc_hub_dist(
  source_dir = ".",
  output_path = "hub_index.json",
  out_dir = "docs",
  projections = ["hub", "narrative"],
  lint_strict = false
)
```

Then commit and push:

```sh
git add hub_index.json docs/
git commit -m "publish: regenerate hub_index"
git push
```

Consumers can now install via `alc_pkg_install({ url: "github.com/you/my-cool-pkg" })`
and your package will appear in `alc_hub_search` results.

---

## 8. Testing

algocline packages should ship tests under `<pkg>/spec/<file>_spec.lua`. Run
them via `mcp__algocline__alc_pkg_test`.

### Spec file layout

Place spec files at `<pkg_root>/spec/<name>_spec.lua`. Each file is a
self-contained lspec suite. The `lust` global (`describe`, `it`, `expect`,
`spy`, etc.) is pre-loaded automatically — no `require` needed.

```lua
-- <pkg_root>/spec/myfeature_spec.lua

local describe, it, expect = lust.describe, lust.it, lust.expect

describe('myfeature', function()
    it('does X correctly', function()
        local result = require('mypkg').do_x()
        expect(result).to.equal('expected_value')
    end)
end)
```

### Running tests

- `alc_pkg_test pkg="mypkg"` — run all `<pkg_root>/spec/*_spec.lua`
- `alc_pkg_test pkg="mypkg" filter="feature"` — run only specs whose stem
  contains `"feature"` (e.g. `feature_spec.lua`)
- `alc_pkg_test pkg="mypkg" spec_dir="tests"` — use a custom spec directory
- `alc_pkg_test code_file="<abs_path>"` — run a single file (escape hatch;
  use absolute paths in worktree environments)
- `alc_pkg_test code="<inline lua>"` — ad-hoc inline test

### Output shape

```json
{
  "passed": 3,
  "failed": 0,
  "pending": 0,
  "total": 3,
  "duration_ms": 42,
  "spec_files": [
    {
      "path": "/path/to/myfeature_spec.lua",
      "passed": 3,
      "failed": 0,
      "total": 3,
      "duration_ms": 40,
      "tests": [
        { "suite": "myfeature", "name": "does X correctly",
          "passed": true, "pending": false, "error": null }
      ]
    }
  ]
}
```

Per-spec-file Lua crashes increment `failed` and continue (execution is not
aborted). Setup failures (package not found, zero spec files) are returned as
a typed error on the MCP wire.

### Migration from `tests/test_<pkg>.lua`

Existing bundled-packages tests use a flat `tests/test_<pkg>.lua` layout and
continue to work via `mcp__lua-debugger__test_launch`. New packages should
adopt the `<pkg>/spec/<file>_spec.lua` layout and use `alc_pkg_test`.
