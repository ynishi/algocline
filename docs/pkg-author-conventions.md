# pkg-author-conventions: docstring-driven narrative

**Updated**: issue `#1778221491-39903` — `M.docs.narrative` and separate `narrative.md` files are removed.
**Applies to**: all bundled packages in `algocline-bundled-packages` and personal packages served via `alc://packages/{name}/narrative`.

The conventions are split into three sections:

1. **Top-level pkg shape** (`M.meta` / `M.spec` / `M.docs`)
2. **Docstring-driven narrative** (replaces `narrative.md` and `M.docs.narrative`)
3. **Docstring style** (rustdoc-flavoured 1-line summary + H2 body sections)

---

## 1. Top-level pkg shape

Every algocline pkg is a Lua module that returns a table `M`. The canonical layout:

```lua
local M = {}

M.meta = {
    name        = "cot",
    version     = "1.2.0",
    description = "Chain-of-thought reasoning",
    category    = "reasoning",
}

M.spec = {
    entries = {
        run = {
            input  = T.shape({ task = T.string }),
            result = T.shape({ reasoning = T.string, answer = T.string }),
        },
    },
}

-- M.docs is optional; set schema_version for future compat if needed.
-- Do NOT set M.docs.narrative — that field is removed (#1778221491-39903).
M.docs = {
    schema_version = 1,
}

function M.run(ctx)
    -- ...
end

return M
```

### 1.1 `M.meta` — identity

Required: `name`. Recommended: `version` (SemVer), `description` (≤ 80 char tagline), `category`.

### 1.2 `M.spec` — runtime contract

Schema for entries written in the `alc_shapes` T DSL. Used by `alc_hub_dist projections=["luacats"]` to generate `types/alc_pkgs.d.lua`, and by `alc.run(<pkg>, ctx)` runtime validation.

### 1.3 `M.docs` — optional container

`M.docs` is preserved as a container for future schema markers. The only recognised field is `schema_version`. **`M.docs.narrative` is removed** — do not set it. The linter will warn on unknown `M.docs` fields in a future release.

---

## 2. Docstring-driven narrative

`init.lua` is the single source of truth for narrative content. The gendoc pipeline (`extract.lua split_sections` + `projections.lua narrative_md`) renders the docstring H2 sections into Markdown on demand.

**Removed conventions** (do not use):

- `M.docs.narrative = "narrative.md"` — field is gone
- Separate `narrative.md` files — not recognised; will be ignored

### 2.1 How the resource is served

When a client requests `alc://packages/{name}/narrative`:

1. The engine locates the pkg's `init.lua` (variant scope first, then global scope).
2. A fresh mlua VM runs `extract.build_pkg_info` + `projections.narrative_md`.
3. The rendered Markdown is returned as `text/markdown`.

No cache — every request re-renders from the current `init.lua`.

### 2.2 Recommended H2 sections

From `algocline-bundled-packages/docs/docstring-convention.md §1.2`. Mapping to the four Diátaxis documentation classes:

| Section | Diátaxis class | When to write |
|---|---|---|
| `## Usage` | How-to | Minimum working example. Write for almost every pkg. |
| `## When to use` | How-to | When the pkg overlaps with siblings and the choice is non-obvious. |
| `## Algorithm` | Explanation | When the algorithm has 3+ named steps or a non-trivial invariant. |
| `## Theoretical foundations` | Explanation | When correctness follows from a theorem or paper. State the theorem. |
| `## Entry contract` | Reference | When the pkg exposes multiple named entries. |
| `## Caveats` | Reference | Known limitations, parameter sensitivity, edge cases. |
| `## Empirical validation` | Reference | Bench data, sweep results, coverage observations. |
| `## Comparison with related packages` | Reference | When a sibling pkg does something similar. One bullet per peer. |
| `## References` | Reference | Papers, arXiv IDs, books. Bullet list — see docstring-convention §5. |

Write only the sections relevant to the pkg. A simple wrapper may need only `## Usage`. A paper implementation typically needs Algorithm + Theoretical foundations + References.

### 2.3 Canonical example: `conformal_vote`

`conformal_vote/init.lua` in `algocline-bundled-packages` is the reference implementation. Its docstring demonstrates all required elements:

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

## 3. Docstring style

Lua docstrings (`---` comments) follow a rustdoc-flavoured convention: **1-line summary, blank line, H2 body sections**.

### 3.1 1-line summary

The first docstring line is the pkg's identity line:

```
{PkgName} — {verb phrase}
```

- One clause, ≤ 80 characters
- Becomes the H1 title in the rendered narrative and the `llms.txt` entry
- `M.meta.description` carries the same (or slightly expanded) content for JSON consumers

Good:

```lua
--- conformal_vote — split conformal prediction gate for multi-agent deliberation
```

Too vague:

```lua
--- conformal_vote — a useful voting package
```

### 3.2 Abstract

One to three sentences after the 1-line summary (separated by a blank `---` line). Runs until the first `## ` heading. Explains the core capability at a glance.

### 3.3 H2 body sections

- H1 (`#`) must not appear — the generator synthesises it from line 1
- H2 (`##`) is the highest permitted section level
- H3 (`###`) for subsections within H2 blocks
- Each heading must be followed by a blank `---` line before body text

### 3.4 Parameters — do not write by hand

Parameters are declared in `M.meta.input_shape` using the `alc_shapes` DSL. The generator synthesises a `## Parameters` table automatically. Manually writing `## Parameters` alongside `input_shape` triggers lint error `E_PARAMETERS_CONFLICT`.

### 3.5 LuaCATS annotations

`---@param`, `---@return`, `---@type`, etc. follow the narrative docstring body. Place them last so the prose reads as a single block when stripped of annotations. The generator stops narrative extraction at the first `---@` line.

---

## 4. Lint rules

Run `alc_hub_gendoc` with `lint_strict=true` before committing. Key codes (full list in `docstring-convention.md §13`):

| Code | Meaning |
|---|---|
| `E_H1_IN_DOCSTRING` | `#` heading in docstring — remove it |
| `E_PARAMETERS_CONFLICT` | `input_shape` + hand-written `## Parameters` both present |
| `W_EMPTY_NARRATIVE` | No abstract and no H2 sections |
| `E_META_MISSING_DESCRIPTION` | `M.meta.description` missing — required for llms.txt |
| `E_NAME_MISMATCH` | `meta.name` does not match the pkg directory name |

---

## 5. Migration from `M.docs.narrative` / `narrative.md`

If an existing pkg uses the old convention:

1. Move narrative content into the `---` docstring block as H2 sections.
2. Remove `M.docs.narrative` from the `M.docs` table (or drop `M.docs` entirely if `schema_version` is also absent).
3. Delete the standalone `narrative.md` file.
4. Run `alc_hub_dist` with `projections=["narrative"]` to regenerate `docs/narrative/{name}.md`.
5. Verify `alc://packages/{name}/narrative` returns the expected Markdown.

Bundled-packages adoption is tracked in issue `#1778197753-34288`.

---

## 6. Related documents

- `algocline-bundled-packages/docs/docstring-convention.md` — full syntax specification (V0 canonical)
- `algocline-bundled-packages/conformal_vote/init.lua` — canonical reference implementation
- `crates/algocline-app/src/service/lua/gendoc/docs/extract.lua` — narrative extraction engine
- `crates/algocline-app/src/service/lua/gendoc/docs/projections.lua` — `narrative_md` projection
