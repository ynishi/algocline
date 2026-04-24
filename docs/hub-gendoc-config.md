# `alc_hub_gendoc` / `alc_hub_dist` config schema

Configuration for the `context7` and `devin` projections can be supplied in
two ways:

1. **`alc.toml` (recommended)** — place `[hub]`, `[hub.context7]`, and/or
   `[hub.devin]` sections in the project root's `alc.toml`.  Picked up
   automatically when `config_path` is omitted.
2. **Explicit `config_path=<file>.toml`** — pass a TOML file containing flat
   `[context7]` / `[devin]` sections.  Retained for backward compatibility.

`.lua` config files are **no longer accepted** as of v0.26.  Passing a `.lua`
path returns an error:
`gendoc: config_path extension '.lua' is no longer supported; use .toml`

---

## `alc.toml` schema

```toml
[hub]
# Shared name/description — propagated to context7 and devin when their own
# fields are absent.
name = "my-project"
description = "Project overview"

[hub.context7]
# Optional: override shared name/description for this projection only.
name = "my-project"
description = "Context7-specific description"

# Rules (choose at most one of the three options):
extra_rules       = ["Append this rule to the default list"]
rules_override    = ["Replace ALL default rules with this list"]
rules_file        = "path/to/rules.txt"   # relative to project root

[hub.devin]
# Optional: override shared name/description for this projection only.
name = "my-project"
description = "Devin-specific description"

# Repo notes (choose at most one of the three options):
extra_repo_notes      = ["Append this note to the default list"]
repo_notes_override   = ["Replace ALL default repo notes with this list"]
repo_notes_file       = "path/to/notes.txt"  # relative to project root
```

All sections are optional.  Fields within each section are also optional and
fall back to core defaults when absent.

---

## Precedence chain

### `name` / `description`

| Priority | Source |
|---|---|
| 1 (highest) | `[hub.context7].name` / `.description` |
| 2 | `[hub].name` / `.description` |
| 3 (lowest) | Core default (`"algocline-hub"` / built-in description) |

The same chain applies to `[hub.devin]`.

### `rules` (context7)

| Priority | Source |
|---|---|
| 1 | `rules_file` — reads lines from the given file; blank lines and `#`-prefixed lines are ignored; **full replacement** |
| 2 | `rules_override` — uses the list as-is; **full replacement** |
| 3 | Core default rules **++** `extra_rules` |

`rules_file` and `rules_override` are **mutually exclusive**.  Setting both
returns:
`gendoc: rules_file and rules_override are mutually exclusive`

### `repo_notes` (devin)

Same three-stage chain as `rules`, using `repo_notes_file` / `repo_notes_override` /
`extra_repo_notes`.

---

## Backward-compatible flat `config_path=*.toml`

Callers that supply an explicit `config_path=<file>.toml` continue to work
unchanged.  The file uses the flat `[context7]` / `[devin]` top-level sections:

```toml
[context7]
projectTitle = "my project"
description = "optional description"
rules = []

[devin]
project_name = "my project"
```

Both top-level keys are optional.  TOML values are recursively converted to
Lua values for the embedded `gen_docs.lua` pipeline.

---

## Error reference

| Condition | Error message |
|---|---|
| `.lua` extension passed as `config_path` | `gendoc: config_path extension '.lua' is no longer supported; use .toml` |
| Unsupported extension (not `.toml`) | `gendoc: config_path '<path>' unsupported extension (expected .toml)` |
| `alc.toml` parse error | `gendoc: alc.toml parse failed: ...` |
| `config_path` TOML parse error | `gendoc: config_path '<path>' parse failed: ...` |
| `rules_file` / `rules_override` both set | `gendoc: rules_file and rules_override are mutually exclusive` |
| `repo_notes_file` / `repo_notes_override` both set | `gendoc: repo_notes_file and repo_notes_override are mutually exclusive` |
| `rules_file` not found | `gendoc: rules_file '<path>' load failed: ...` |
| Unknown projection token | `gendoc: unknown projection '<token>' (allowed: hub, context7, devin, lint, lint_only)` |

---

## `alc_hub_dist` presets (`preset`)

`alc_hub_dist` can expand a named preset into primitive `alc_hub_gendoc`
arguments.

Successful `alc_hub_dist` responses always include:

- `preset_catalog_version`: revision marker for the builtin preset dictionary
  bundled with the running `alc` binary.

When `preset` is provided, responses also include a `preset` object with the
resolved primitive args (`projections` / `config_path` / `lint_strict`) for
debuggability.

### Builtin `publish` (`Current`)

When `preset = "publish"` and `projections` is omitted, the builtin default is:

- `projections`: `["hub", "lint"]`
- `lint_strict`: `false`

This avoids requiring optional projection configs (`context7` / `devin`)
unless the caller (or `alc.toml`) explicitly opts in.

### Optional `alc.toml` dist overrides

In addition to the `[hub.context7]` / `[hub.devin]` projection sections above,
`alc.toml` supports dist-specific preset overrides:

```toml
[hub.dist]

[hub.dist.presets.publish]
projections = ["context7", "hub"]
config_path = "configs/gendoc.toml"
lint_strict = false
```

Merge order (strongest wins):

1. Explicit MCP arguments (`projections` / `config_path` / `lint_strict`)
2. `alc.toml` preset overrides (`[hub.dist.presets.<name>]`) — only fills **omitted** knobs
3. Builtin defaults for the selected preset
