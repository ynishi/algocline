# MCP Resources — algocline

algocline-mcp advertises an MCP `resources` capability alongside `tools`.
Service-layer read-only paths are projected as resources under the `alc://` scheme.

## Overview

**V1 scope:**

- Fixed resources (2): `alc://types/alc.d.lua`, `alc://types/alc_shapes.d.lua`
- Template resources (7): packages, cards, scenarios, evals, logs

**V2 candidates (out of scope for this release):**

- `alc://hub/index` — pending canonical AppDir path for `hub_reindex` default output
- `alc://packages/{name}/narrative` — `hub_gendoc` emits to an external `out_dir`, not AppDir
- `list_changed` notifications and `resources/subscribe` — V1 is static capability only

## Fixed Resources

Returned by `resources/list`. Always listed; a `read` for a missing file returns an error.

| URI | Name | Description | MIME | Source Path |
|---|---|---|---|---|
| `alc://types/alc.d.lua` | `alc.d.lua` | Lua type stubs for `alc.*` StdLib | `text/x-lua` | `$ALC_HOME/types/alc.d.lua` |
| `alc://types/alc_shapes.d.lua` | `alc_shapes.d.lua` | Lua type stubs for alc_shapes DSL | `text/x-lua` | `$ALC_HOME/types/alc_shapes.d.lua` |

## Resource Templates

Returned by `resources/templates/list`. URI templates follow RFC 6570 level-1 (`{var}`).

| URI Template | Description | MIME | Notes |
|---|---|---|---|
| `alc://packages/{name}/init.lua` | Package init.lua source | `text/x-lua` | Resolves global and variant-scoped packages |
| `alc://packages/{name}/meta` | Package metadata (M.meta table) | `application/json` | Extracted without running the Lua VM |
| `alc://cards/{card_id}` | Full Card JSON | `application/json` | card_id is a UUID |
| `alc://cards/{card_id}/samples` | Card samples JSONL | `application/json` | Supports `?offset=N&limit=M` |
| `alc://scenarios/{name}` | Scenario source | `text/x-lua` | Name without `.lua` extension |
| `alc://eval/{result_id}` | Eval result JSON | `application/json` | result_id is a UUID |
| `alc://logs/{session_id}` | Session log JSON | `application/json` | Supports `?limit=M&max_chars=C` |

### Template Examples

```
alc://packages/panel/init.lua
alc://packages/cot/meta
alc://cards/3f4b1a2c-0001-0000-0000-000000000000
alc://cards/3f4b1a2c-0001-0000-0000-000000000000/samples?offset=0&limit=20
alc://scenarios/my_scenario
alc://eval/9e8d7c6b-0001-0000-0000-000000000000
alc://logs/1a2b3c4d-0001-0000-0000-000000000000?limit=50&max_chars=10000

```

## Pagination

Pagination is expressed via URI query string parameters.

### `alc://cards/{card_id}/samples`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offset` | `usize` | `0` | Number of samples to skip |
| `limit` | `usize` | `100` | Maximum samples to return |

### `alc://logs/{session_id}`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | `usize` | `50` | Maximum log entries to return |
| `max_chars` | `usize` | `20000` | Maximum total characters in the response |

## MIME Types

| Extension / Format | MIME Type |
|---|---|
| `.lua` | `text/x-lua` |
| `.md` | `text/markdown` |
| JSON | `application/json` |

## Claude Code Integration

Claude Code v2.1.116+ supports `@server:protocol://path` resource mentions.
For algocline resources:

```
@alc:alc://types/alc.d.lua
@alc:alc://packages/panel/init.lua
@alc:alc://cards/3f4b1a2c-0001-0000-0000-000000000000
```

`resources/templates/list` is retrieved once at session start; `resources/list` is
retrieved as needed. Neither supports `list_changed` notifications in V1 (static capability).

## Error Semantics

| Condition | Error |
|---|---|
| Invalid scheme (not `alc://`) | `invalid_params` |
| Unknown service (e.g. `alc://foo/...`) | `invalid_params` |
| Path traversal segment (`..`) | `invalid_params` |
| Resource not found (file absent, card/eval/session missing) | `invalid_params` |
| Invalid query parameter value | `invalid_params` |
| EngineApi / I/O failure | `internal_error` |

Resources that appear in `resources/list` but whose backing file is absent at read-time
return `invalid_params` rather than a partial result. This matches MCP spec semantics
(the spec permits a listed resource to return an error on read).
