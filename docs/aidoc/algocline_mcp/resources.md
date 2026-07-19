# algocline-mcp::resources

MCP Resources catalog for algocline.

Implements a `ResourceCatalog` that dispatches `alc://<service>/<path>`
URIs to the appropriate backing store. Fixed resources (static files) are
fully implemented here; template dispatch stubs will be filled in by
Subtask 2.

## Functions

- `build_list_resources_result` — Build a `ListResourcesResult` from the catalog's fixed list.
- `build_list_templates_result` — Build a `ListResourceTemplatesResult` from the catalog's template list.
- `err_to_mcp` — Convert an `EngineApi` `Err(String)` to a `McpError`.
- `extract_template_vars` — Extract all RFC 6570 Level-1 variable names from a URI template string.
- `parse_uri` — Parse an `alc://<service>/<path>?<query>` URI.

## Types

- `CompletionCandidates` — Candidate result for a `completion/complete` request.
- `ParsedUri` — Parsed representation of an `alc://` URI.
- `ResourceCatalog` — Catalog that maps `alc://` URIs to MCP resource responses.
- `UriParseError` — Errors produced when parsing an `alc://` URI.

## Constants

- `DEFAULT_CARD_SAMPLES_LIMIT` — Default `limit` for `alc://cards/{id}/samples` when `?limit=` is absent.
- `DEFAULT_LOGS_LIMIT` — Default `limit` for `alc://logs/{session_id}` when `?limit=` is absent.
- `DEFAULT_LOGS_MAX_CHARS` — Default `max_chars` for `alc://logs/{session_id}` when `?max_chars=` is absent.
- `MAX_CARD_SAMPLES_LIMIT` — Hard cap for `?limit=` on `alc://cards/{id}/samples`.
- `MAX_CARD_SAMPLES_OFFSET` — Hard cap for `?offset=` on `alc://cards/{id}/samples`.
- `MAX_LOGS_LIMIT` — Hard cap for `?limit=` on `alc://logs/{session_id}`.
- `MAX_LOGS_MAX_CHARS` — Hard cap for `?max_chars=` on `alc://logs/{session_id}`.

