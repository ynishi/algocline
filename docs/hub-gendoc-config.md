# `alc_hub_gendoc` / `alc_hub_dist` config schema

`config_path` accepts a TOML or Lua file (selected by extension) used only
when projections include `context7` and/or `devin`.

## Minimal schema (TOML)

```toml
[context7]
projectTitle = "my project"
description = "optional description"
rules = []

[devin]
project_name = "my project"
```

## Minimal schema (Lua)

```lua
return {
    context7 = {
        projectTitle = "my project",
        description = "optional description",
        rules = {},
    },
    devin = {
        project_name = "my project",
    },
}
```

## Rules

- Extension determines parser: `.toml` / `.TOML` → TOML, `.lua` / `.LUA` → Lua.
- Other extensions raise `gendoc: config_path '<path>' unsupported extension (expected .toml or .lua)`.
- Top-level keys are optional individually:
  - `context7` is required only when `projections` includes `"context7"`.
  - `devin` is required only when `projections` includes `"devin"`.
- When present, each projection key must be a table.
- TOML values are recursively converted to Lua values:
  - string → Lua string
  - integer → Lua integer
  - float → Lua number
  - boolean → Lua boolean
  - datetime → Lua string (RFC3339 text form)
  - array → Lua array-style table (1-based indexes)
  - table → Lua table
- Lua tables are used directly (no conversion layer).

## Error behavior

- Missing `config_path` while requesting `context7`/`devin`:
  - `gendoc: config_path is required when projections include context7 or devin`
- Unsupported file extension:
  - `gendoc: config_path '<path>' unsupported extension (expected .toml or .lua)`
- Parse error in TOML:
  - `gendoc: config_path '<path>' parse failed: ...`
- Parse error in Lua:
  - `gendoc: config_path '<path>' lua eval failed: ...`
- Lua return value is not a table:
  - `gendoc: config_path '<path>' must return a table, got <type>`
- Unknown projection token:
  - `gendoc: unknown projection '<token>' (allowed: hub, context7, devin, lint, lint_only)`

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

### Optional `alc.toml` overrides

In the resolved project root (`project_root`, or ancestor-discovered
`alc.toml`):

```toml
[hub.dist]

[hub.dist.presets.publish]
projections = ["context7", "hub"]
config_path = "configs/gendoc.toml"
lint_strict = false
```

Merge order (strongest wins):

1. explicit MCP arguments (`projections` / `config_path` / `lint_strict`)
2. `alc.toml` preset overrides (`[hub.dist.presets.<name>]`) — only fills **omitted** knobs
3. builtin defaults for the selected preset

# hub_gendoc config schema

`alc_hub_gendoc` and `alc_hub_dist` accept `config_path` as a TOML or Lua
file (selected by extension: `.toml` / `.lua`).

## Required only for projection targets

- `config_path` is required when `projections` includes `context7` or `devin`.
- If neither projection is used, `config_path` can be omitted.

## Schema

TOML form:

```toml
[context7]
projectTitle = "my project"
description = "optional description"
rules = [] # array

[devin]
project_name = "my project"
```

Lua form (wrapped shape — both top-level keys optional):

```lua
return {
    context7 = {
        projectTitle = "my project",
        description = "optional description",
        rules = {},
    },
    devin = {
        project_name = "my project",
    },
}
```

## Rules

- Top-level sections are optional individually:
  - `context7`
  - `devin`
- When present, each section must be a table.
- Values support TOML scalar/array/table types and are converted recursively to Lua tables for the embedded `gen_docs.lua` pipeline.
- Lua tables are used directly without conversion.

## Validation behavior

- Unknown projection values are rejected:
  - allowed: `hub`, `context7`, `devin`, `lint`, `lint_only`
- Invalid TOML syntax returns `gendoc: config_path '...' parse failed: ...`
- Invalid Lua syntax returns `gendoc: config_path '...' lua eval failed: ...`
- Unsupported extension returns `gendoc: config_path '...' unsupported extension (expected .toml or .lua)`
- Missing required config for `context7`/`devin` returns:
  - `gendoc: config_path is required when projections include context7 or devin`
