# `alc_hub_gendoc` / `alc_hub_dist` config schema

`config_path` is a TOML file used only when projections include `context7` and/or `devin`.

## Minimal schema

```toml
[context7]
projectTitle = "my project"
description = "optional description"
rules = []

[devin]
project_name = "my project"
```

## Rules

- Top-level keys are optional individually:
  - `[context7]` is required only when `projections` includes `"context7"`.
  - `[devin]` is required only when `projections` includes `"devin"`.
- When present, each projection key must be a TOML table.
- TOML values are recursively converted to Lua values:
  - string -> Lua string
  - integer -> Lua integer
  - float -> Lua number
  - boolean -> Lua boolean
  - datetime -> Lua string (RFC3339 text form)
  - array -> Lua array-style table (1-based indexes)
  - table -> Lua table

## Error behavior

- Missing `config_path` while requesting `context7`/`devin`:
  - `gendoc: config_path is required when projections include context7 or devin`
- Parse error in TOML:
  - `gendoc: config_path '<path>' parse failed: ...`
- Unknown projection token:
  - `gendoc: unknown projection '<token>' (allowed: hub, context7, devin, lint, lint_only)`

# hub_gendoc config TOML schema

`alc_hub_gendoc` and `alc_hub_dist` accept `config_path` as a TOML file.

## Required only for projection targets

- `config_path` is required when `projections` includes `context7` or `devin`.
- If neither projection is used, `config_path` can be omitted.

## Schema

```toml
[context7]
projectTitle = "my project"
description = "optional description"
rules = [] # array

[devin]
project_name = "my project"
```

## Rules

- Top-level sections are optional individually:
  - `[context7]`
  - `[devin]`
- When present, each section must be a TOML table.
- Values support TOML scalar/array/table types and are converted recursively to Lua tables for the embedded `gen_docs.lua` pipeline.

## Validation behavior

- Unknown projection values are rejected:
  - allowed: `hub`, `context7`, `devin`, `lint`, `lint_only`
- Invalid TOML syntax returns `gendoc: config_path '...' parse failed: ...`
- Missing required config for `context7`/`devin` returns:
  - `gendoc: config_path is required when projections include context7 or devin`

