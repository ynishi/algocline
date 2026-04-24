//! MCP Resources catalog for algocline.
//!
//! Implements a `ResourceCatalog` that dispatches `alc://<service>/<path>`
//! URIs to the appropriate backing store. Fixed resources (static files) are
//! fully implemented here; template dispatch stubs will be filled in by
//! Subtask 2.

use std::collections::HashMap;
use std::sync::Arc;

use algocline_app::EngineApi;
use algocline_core::AppDir;
use rmcp::model::{
    Annotated, ListResourceTemplatesResult, ListResourcesResult, RawResource, RawResourceTemplate,
    ReadResourceResult, Resource, ResourceContents, ResourceTemplate,
};
use rmcp::ErrorData as McpError;

// ─── Known services ──────────────────────────────────────────────────────────

const KNOWN_SERVICES: &[&str] = &["types", "packages", "cards", "scenarios", "eval", "logs"];

// ─── URI parser ──────────────────────────────────────────────────────────────

/// Parsed representation of an `alc://` URI.
#[derive(Debug, PartialEq)]
pub struct ParsedUri {
    /// Service name (the first path component, e.g. `"types"`, `"cards"`).
    pub service: String,
    /// Remaining path segments after the service component.
    pub segments: Vec<String>,
    /// Parsed query parameters (key=value pairs, no URL-decoding needed for V1).
    pub query: HashMap<String, String>,
}

/// Errors produced when parsing an `alc://` URI.
#[derive(Debug, thiserror::Error)]
pub enum UriParseError {
    #[error("invalid scheme: expected alc://, got {0}")]
    Scheme(String),
    #[error("unknown service: {0}")]
    UnknownService(String),
    #[error("missing path segment in {uri}")]
    MissingSegment { uri: String },
    #[error("invalid query: {0}")]
    Query(String),
    #[error("path traversal segment rejected: {0}")]
    TraversalSegment(String),
}

/// Parse an `alc://<service>/<path>?<query>` URI.
///
/// V1 constraints:
/// - Scheme must be exactly `alc://`
/// - Service must be one of the known services
/// - At least one path segment after the service is required
/// - Query values are treated as raw strings (no URL-decoding)
pub fn parse_uri(s: &str) -> Result<ParsedUri, UriParseError> {
    // Strip the "alc://" scheme.
    let rest = s
        .strip_prefix("alc://")
        .ok_or_else(|| UriParseError::Scheme(s.to_string()))?;

    // Split path from optional query string.
    let (path_part, query_part) = match rest.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (rest, None),
    };

    // Split path into segments, filtering out empty strings produced by
    // trailing slashes (e.g. "cards/" → ["cards"]).
    let mut segments: Vec<String> = path_part
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    // Reject path-traversal segments (defense-in-depth for `read_types` and
    // future template dispatch that joins segments into filesystem paths).
    for seg in &segments {
        if seg == ".." || seg == "." {
            return Err(UriParseError::TraversalSegment(seg.clone()));
        }
    }

    // First segment is the service name.
    if segments.is_empty() {
        return Err(UriParseError::MissingSegment { uri: s.to_string() });
    }
    let service = segments.remove(0);

    // Validate service.
    if !KNOWN_SERVICES.contains(&service.as_str()) {
        return Err(UriParseError::UnknownService(service));
    }

    // After removing the service, at least one more segment is required.
    if segments.is_empty() {
        return Err(UriParseError::MissingSegment { uri: s.to_string() });
    }

    // Parse the query string (if present).
    let query = parse_query(query_part.unwrap_or(""), s)?;

    Ok(ParsedUri {
        service,
        segments,
        query,
    })
}

/// Parse a raw query string (`key=value&key2=value2`).
///
/// Empty query strings produce an empty map. A key without a value
/// (e.g. `?=bad`) is rejected as malformed.
fn parse_query(qs: &str, full_uri: &str) -> Result<HashMap<String, String>, UriParseError> {
    let mut map = HashMap::new();
    if qs.is_empty() {
        return Ok(map);
    }
    for pair in qs.split('&') {
        if pair.is_empty() {
            continue;
        }
        match pair.split_once('=') {
            None => {
                // key without '=' — treat as bare key with empty value
                map.insert(pair.to_string(), String::new());
            }
            Some(("", _)) => {
                return Err(UriParseError::Query(format!(
                    "empty key in query of {full_uri}"
                )));
            }
            Some((k, v)) => {
                if v.contains('=') {
                    return Err(UriParseError::Query(format!(
                        "duplicate '=' in query pair '{pair}' of {full_uri}"
                    )));
                }
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    Ok(map)
}

// ─── ResourceCatalog ─────────────────────────────────────────────────────────

/// Catalog that maps `alc://` URIs to MCP resource responses.
///
/// Fixed resources (e.g. `alc://types/alc.d.lua`) are backed by on-disk
/// files under `AppDir::types_dir()`. Template resources are dispatched via
/// the `EngineApi` trait and will be added in Subtask 2.
pub struct ResourceCatalog {
    app: Arc<dyn EngineApi>,
    app_dir: Arc<AppDir>,
}

impl ResourceCatalog {
    /// Construct a new catalog.
    ///
    /// `app` is retained for template dispatch (Subtask 2).
    /// `app_dir` is used for fixed file reads (types stubs).
    pub fn new(app: Arc<dyn EngineApi>, app_dir: Arc<AppDir>) -> Self {
        Self { app, app_dir }
    }

    /// Return the list of fixed (static) resources.
    ///
    /// Fixed resources are always listed even if the underlying file happens
    /// to be absent at list-time; a subsequent `read` for a missing file will
    /// return `McpError::invalid_params`. This matches MCP spec semantics.
    pub fn list_fixed(&self) -> Vec<Resource> {
        vec![
            make_resource(
                "alc://types/alc.d.lua",
                "alc.d.lua",
                "Lua type stubs for alc.* StdLib",
                "text/x-lua",
            ),
            make_resource(
                "alc://types/alc_shapes.d.lua",
                "alc_shapes.d.lua",
                "Lua type stubs for alc shapes",
                "text/x-lua",
            ),
        ]
    }

    /// Return the list of resource templates (URI template notation, RFC 6570 level 1).
    ///
    /// Returns the 7 approved V1 templates. `packages/{name}/narrative` is out
    /// of scope for V1.
    pub fn list_templates(&self) -> Vec<ResourceTemplate> {
        vec![
            make_template(
                "alc://packages/{name}/init.lua",
                "package-init-lua",
                "Lua source of an installed package",
                Some("text/x-lua"),
            ),
            make_template(
                "alc://packages/{name}/meta",
                "package-meta",
                "Package metadata JSON (description, category, alc_shapes_compat)",
                Some("application/json"),
            ),
            make_template(
                "alc://cards/{card_id}",
                "card",
                "Immutable Card snapshot",
                Some("application/json"),
            ),
            make_template(
                "alc://cards/{card_id}/samples",
                "card-samples",
                "Per-case sample rows (paginate with ?offset=N&limit=M)",
                Some("application/json"),
            ),
            make_template(
                "alc://scenarios/{name}",
                "scenario",
                "Scenario Lua source",
                Some("text/x-lua"),
            ),
            make_template(
                "alc://eval/{result_id}",
                "eval-result",
                "Eval result detail ({strategy}_{timestamp_secs} id)",
                Some("application/json"),
            ),
            make_template(
                "alc://logs/{session_id}",
                "session-log",
                "Session log (paginate with ?limit=N&max_chars=M)",
                Some("application/json"),
            ),
        ]
    }

    /// Read a resource by URI.
    ///
    /// Fixed resources (`alc://types/...`) are backed by `AppDir::types_dir()`.
    /// Template resources are dispatched via `EngineApi`.
    pub async fn read(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        let parsed = parse_uri(uri).map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match parsed.service.as_str() {
            "types" => self.read_types(uri, &parsed),
            "packages" => self.read_packages(uri, &parsed).await,
            "cards" => self.read_cards(uri, &parsed).await,
            "scenarios" => self.read_scenarios(uri, &parsed).await,
            "eval" => self.read_eval(uri, &parsed).await,
            "logs" => self.read_logs(uri, &parsed).await,
            other => Err(McpError::invalid_params(
                format!("unknown service: {other}"),
                None,
            )),
        }
    }

    // ── Private dispatch helpers ──────────────────────────────────────────

    fn read_types(&self, uri: &str, parsed: &ParsedUri) -> Result<ReadResourceResult, McpError> {
        if !parsed.query.is_empty() {
            return Err(McpError::invalid_params(
                format!("query params not supported on {uri}"),
                None,
            ));
        }
        let file_name = parsed.segments.join("/");
        let path = self.app_dir.types_dir().join(&file_name);

        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let contents = ResourceContents::TextResourceContents {
                    uri: uri.to_string(),
                    mime_type: Some("text/x-lua".to_string()),
                    text,
                    meta: None,
                };
                Ok(ReadResourceResult {
                    contents: vec![contents],
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    async fn read_packages(
        &self,
        uri: &str,
        parsed: &ParsedUri,
    ) -> Result<ReadResourceResult, McpError> {
        // Accepted: alc://packages/{name}/init.lua  or  alc://packages/{name}/meta
        match parsed.segments.as_slice() {
            [name, sub] if sub == "init.lua" => {
                if !parsed.query.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("query params not supported on {uri}"),
                        None,
                    ));
                }
                let text = self.app.pkg_read_init_lua(name).await.map_err(err_to_mcp)?;
                Ok(text_result(uri, text, "text/x-lua"))
            }
            [name, sub] if sub == "meta" => {
                if !parsed.query.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("query params not supported on {uri}"),
                        None,
                    ));
                }
                // Use pkg_list with a name filter to retrieve metadata for a
                // single package.  An empty result means the package is unknown.
                let filter = serde_json::json!({ "name": name });
                let json_str = self
                    .app
                    .pkg_list(
                        None,
                        None,
                        None,
                        Some(filter),
                        None,
                        Some("full".to_string()),
                    )
                    .await
                    .map_err(err_to_mcp)?;
                // Extract first entry from the `packages` array.
                let val: serde_json::Value =
                    serde_json::from_str(&json_str).map_err(|e| err_to_mcp(e.to_string()))?;
                let pkgs = val
                    .get("packages")
                    .and_then(|p| p.as_array())
                    .ok_or_else(|| {
                        err_to_mcp("pkg_list response missing 'packages' field".into())
                    })?;
                if pkgs.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("resource not found: {uri}"),
                        None,
                    ));
                }
                let entry_json =
                    serde_json::to_string(&pkgs[0]).map_err(|e| err_to_mcp(e.to_string()))?;
                Ok(text_result(uri, entry_json, "application/json"))
            }
            _ => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
        }
    }

    async fn read_cards(
        &self,
        uri: &str,
        parsed: &ParsedUri,
    ) -> Result<ReadResourceResult, McpError> {
        match parsed.segments.as_slice() {
            [card_id] => {
                // alc://cards/{card_id} — no query params allowed
                if !parsed.query.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("query params not supported on {uri}"),
                        None,
                    ));
                }
                let json_str = self.app.card_get(card_id).await.map_err(err_to_mcp)?;
                Ok(text_result(uri, json_str, "application/json"))
            }
            [card_id, sub] if sub == "samples" => {
                // alc://cards/{card_id}/samples?offset=N&limit=M
                let offset = parse_usize_param(&parsed.query, "offset", uri)?;
                let limit = parse_usize_param(&parsed.query, "limit", uri)?
                    .or(Some(DEFAULT_CARD_SAMPLES_LIMIT));
                // Reject unknown query keys (only offset and limit are valid)
                for key in parsed.query.keys() {
                    if key != "offset" && key != "limit" {
                        return Err(McpError::invalid_params(
                            format!("unsupported query param '{key}' on {uri}"),
                            None,
                        ));
                    }
                }
                let json_str = self
                    .app
                    .card_samples(card_id, offset, limit, None)
                    .await
                    .map_err(err_to_mcp)?;
                Ok(text_result(uri, json_str, "application/json"))
            }
            _ => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
        }
    }

    async fn read_scenarios(
        &self,
        uri: &str,
        parsed: &ParsedUri,
    ) -> Result<ReadResourceResult, McpError> {
        match parsed.segments.as_slice() {
            [name] => {
                if !parsed.query.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("query params not supported on {uri}"),
                        None,
                    ));
                }
                let text = self.app.scenario_show(name).await.map_err(err_to_mcp)?;
                Ok(text_result(uri, text, "text/x-lua"))
            }
            _ => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
        }
    }

    async fn read_eval(
        &self,
        uri: &str,
        parsed: &ParsedUri,
    ) -> Result<ReadResourceResult, McpError> {
        match parsed.segments.as_slice() {
            [result_id] => {
                if !parsed.query.is_empty() {
                    return Err(McpError::invalid_params(
                        format!("query params not supported on {uri}"),
                        None,
                    ));
                }
                // Minimal format validation: must contain at least one '_'
                // (strategy names themselves may contain underscores, so we
                // only verify the minimum requirement).
                if !result_id.contains('_') {
                    return Err(McpError::invalid_params(
                        format!(
                            "invalid eval_id: must be {{strategy}}_{{timestamp}}, got '{result_id}'"
                        ),
                        None,
                    ));
                }
                let json_str = self.app.eval_detail(result_id).await.map_err(err_to_mcp)?;
                Ok(text_result(uri, json_str, "application/json"))
            }
            _ => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
        }
    }

    async fn read_logs(
        &self,
        uri: &str,
        parsed: &ParsedUri,
    ) -> Result<ReadResourceResult, McpError> {
        match parsed.segments.as_slice() {
            [session_id] => {
                // alc://logs/{session_id}?limit=N&max_chars=M
                let limit =
                    parse_usize_param(&parsed.query, "limit", uri)?.or(Some(DEFAULT_LOGS_LIMIT));
                let max_chars = parse_usize_param(&parsed.query, "max_chars", uri)?
                    .or(Some(DEFAULT_LOGS_MAX_CHARS));
                // Reject unknown query keys (only limit and max_chars are valid)
                for key in parsed.query.keys() {
                    if key != "limit" && key != "max_chars" {
                        return Err(McpError::invalid_params(
                            format!("unsupported query param '{key}' on {uri}"),
                            None,
                        ));
                    }
                }
                let json_str = self
                    .app
                    .log_view(Some(session_id), limit, max_chars)
                    .await
                    .map_err(err_to_mcp)?;
                Ok(text_result(uri, json_str, "application/json"))
            }
            _ => Err(McpError::invalid_params(
                format!("resource not found: {uri}"),
                None,
            )),
        }
    }
}

// ─── Pagination defaults ──────────────────────────────────────────────────────

/// Default `limit` for `alc://cards/{id}/samples` when `?limit=` is absent.
pub const DEFAULT_CARD_SAMPLES_LIMIT: usize = 100;
/// Default `limit` for `alc://logs/{session_id}` when `?limit=` is absent.
pub const DEFAULT_LOGS_LIMIT: usize = 50;
/// Default `max_chars` for `alc://logs/{session_id}` when `?max_chars=` is absent.
pub const DEFAULT_LOGS_MAX_CHARS: usize = 20_000;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a `ListResourcesResult` from the catalog's fixed list.
pub fn build_list_resources_result(catalog: &ResourceCatalog) -> ListResourcesResult {
    ListResourcesResult::with_all_items(catalog.list_fixed())
}

/// Build a `ListResourceTemplatesResult` from the catalog's template list.
pub fn build_list_templates_result(catalog: &ResourceCatalog) -> ListResourceTemplatesResult {
    ListResourceTemplatesResult::with_all_items(catalog.list_templates())
}

/// Convert an `EngineApi` `Err(String)` to a `McpError`.
pub fn err_to_mcp(s: String) -> McpError {
    McpError::internal_error(s, None)
}

/// Parse a single usize query parameter by key.
///
/// Returns `Ok(None)` when the key is absent, `Err` when the value is present
/// but cannot be parsed as `usize`. Never silently coerces parse failures to 0.
fn parse_usize_param(
    query: &HashMap<String, String>,
    key: &str,
    uri: &str,
) -> Result<Option<usize>, McpError> {
    match query.get(key) {
        None => Ok(None),
        Some(s) => s.parse::<usize>().map(Some).map_err(|e| {
            McpError::invalid_params(
                format!("invalid query param '{key}={s}' on {uri}: {e}"),
                None,
            )
        }),
    }
}

/// Produce a single-content `ReadResourceResult` with text content.
fn text_result(uri: &str, text: String, mime_type: &str) -> ReadResourceResult {
    ReadResourceResult {
        contents: vec![ResourceContents::TextResourceContents {
            uri: uri.to_string(),
            mime_type: Some(mime_type.to_string()),
            text,
            meta: None,
        }],
    }
}

fn make_resource(uri: &str, name: &str, description: &str, mime_type: &str) -> Resource {
    let raw = RawResource {
        uri: uri.to_string(),
        name: name.to_string(),
        title: None,
        description: Some(description.to_string()),
        mime_type: Some(mime_type.to_string()),
        size: None,
        icons: None,
        meta: None,
    };
    Annotated::new(raw, None)
}

fn make_template(
    uri_template: &str,
    name: &str,
    description: &str,
    mime_type: Option<&str>,
) -> ResourceTemplate {
    let raw = RawResourceTemplate {
        uri_template: uri_template.to_string(),
        name: name.to_string(),
        title: None,
        description: Some(description.to_string()),
        mime_type: mime_type.map(|s| s.to_string()),
        icons: None,
    };
    Annotated::new(raw, None)
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── URI parser tests ──────────────────────────────────────────────────

    #[test]
    fn parse_valid_fixed_uri() {
        let parsed = parse_uri("alc://types/alc.d.lua").unwrap();
        assert_eq!(parsed.service, "types");
        assert_eq!(parsed.segments, vec!["alc.d.lua"]);
        assert!(parsed.query.is_empty());
    }

    #[test]
    fn parse_valid_template_uri() {
        let parsed = parse_uri("alc://cards/xyz-123").unwrap();
        assert_eq!(parsed.service, "cards");
        assert_eq!(parsed.segments, vec!["xyz-123"]);
        assert!(parsed.query.is_empty());
    }

    #[test]
    fn parse_with_query() {
        let parsed = parse_uri("alc://cards/xyz/samples?offset=10&limit=50").unwrap();
        assert_eq!(parsed.service, "cards");
        assert_eq!(parsed.segments, vec!["xyz", "samples"]);
        assert_eq!(parsed.query.get("offset").map(|s| s.as_str()), Some("10"));
        assert_eq!(parsed.query.get("limit").map(|s| s.as_str()), Some("50"));
    }

    #[test]
    fn parse_missing_scheme() {
        let err = parse_uri("types/alc.d.lua").unwrap_err();
        assert!(matches!(err, UriParseError::Scheme(_)));
    }

    #[test]
    fn parse_wrong_scheme() {
        let err = parse_uri("https://foo").unwrap_err();
        assert!(matches!(err, UriParseError::Scheme(_)));
    }

    #[test]
    fn parse_unknown_service() {
        let err = parse_uri("alc://unknown/x").unwrap_err();
        assert!(matches!(err, UriParseError::UnknownService(_)));
    }

    #[test]
    fn parse_missing_segment() {
        let err = parse_uri("alc://cards/").unwrap_err();
        assert!(matches!(err, UriParseError::MissingSegment { .. }));
    }

    #[test]
    fn parse_bad_query_empty_key() {
        let err = parse_uri("alc://cards/x?=bad").unwrap_err();
        assert!(matches!(err, UriParseError::Query(_)));
    }

    #[test]
    fn parse_empty_query_is_ok() {
        let parsed = parse_uri("alc://cards/x?").unwrap();
        assert!(parsed.query.is_empty());
    }

    #[test]
    fn parse_shapes_uri() {
        let parsed = parse_uri("alc://types/alc_shapes.d.lua").unwrap();
        assert_eq!(parsed.service, "types");
        assert_eq!(parsed.segments, vec!["alc_shapes.d.lua"]);
    }

    // ── ResourceCatalog helpers ───────────────────────────────────────────

    fn make_test_catalog(root: PathBuf) -> ResourceCatalog {
        let app_dir = Arc::new(AppDir::new(root));
        ResourceCatalog::new(Arc::new(NoopEngine), app_dir)
    }

    // Minimal no-op EngineApi implementation for unit tests.
    struct NoopEngine;

    #[async_trait::async_trait]
    impl EngineApi for NoopEngine {
        async fn run(
            &self,
            _code: Option<String>,
            _code_file: Option<String>,
            _ctx: Option<serde_json::Value>,
            _project_root: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn advice(
            &self,
            _strategy: &str,
            _task: Option<String>,
            _opts: Option<serde_json::Value>,
            _project_root: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn continue_single(
            &self,
            _session_id: &str,
            _response: String,
            _query_id: Option<&str>,
            _usage: Option<algocline_core::TokenUsage>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn continue_batch(
            &self,
            _session_id: &str,
            _responses: Vec<algocline_core::QueryResponse>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn status(
            &self,
            _session_id: Option<&str>,
            _pending_filter: Option<serde_json::Value>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval(
            &self,
            _scenario: Option<String>,
            _scenario_file: Option<String>,
            _scenario_name: Option<String>,
            _strategy: &str,
            _strategy_opts: Option<serde_json::Value>,
            _auto_card: bool,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval_history(
            &self,
            _strategy: Option<&str>,
            _limit: usize,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval_detail(&self, _eval_id: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval_compare(&self, _eval_id_a: &str, _eval_id_b: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn scenario_list(&self) -> Result<String, String> {
            Err("noop".into())
        }
        async fn scenario_show(&self, _name: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn scenario_install(&self, _url: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_link(
            &self,
            _path: String,
            _name: Option<String>,
            _force: Option<bool>,
            _scope: Option<String>,
            _project_root: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_list(
            &self,
            _project_root: Option<String>,
            _limit: Option<i32>,
            _sort: Option<String>,
            _filter: Option<serde_json::Value>,
            _fields: Option<Vec<String>>,
            _verbose: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_install(&self, _url: String, _name: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_unlink(&self, _name: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_remove(
            &self,
            _name: &str,
            _project_root: Option<String>,
            _version: Option<String>,
            _scope: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_repair(
            &self,
            _name: Option<String>,
            _project_root: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_doctor(
            &self,
            _name: Option<String>,
            _project_root: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn add_note(
            &self,
            _session_id: &str,
            _content: &str,
            _title: Option<&str>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn log_view(
            &self,
            _session_id: Option<&str>,
            _limit: Option<usize>,
            _max_chars: Option<usize>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn stats(
            &self,
            _strategy_filter: Option<&str>,
            _days: Option<u64>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn init(&self, _project_root: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn update(&self, _project_root: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn migrate(&self, _project_root: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_list(&self, _pkg: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_get(&self, _card_id: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_find(
            &self,
            _pkg: Option<String>,
            _where_: Option<serde_json::Value>,
            _order_by: Option<serde_json::Value>,
            _limit: Option<usize>,
            _offset: Option<usize>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_alias_list(&self, _pkg: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_get_by_alias(&self, _name: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_alias_set(
            &self,
            _name: &str,
            _card_id: &str,
            _pkg: Option<String>,
            _note: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_append(
            &self,
            _card_id: &str,
            _fields: serde_json::Value,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_install(&self, _url: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_samples(
            &self,
            _card_id: &str,
            _offset: Option<usize>,
            _limit: Option<usize>,
            _where_: Option<serde_json::Value>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_lineage(
            &self,
            _card_id: &str,
            _direction: Option<String>,
            _depth: Option<usize>,
            _include_stats: Option<bool>,
            _relation_filter: Option<Vec<String>>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_reindex(
            &self,
            _output_path: Option<String>,
            _source_dir: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_gendoc(
            &self,
            _source_dir: String,
            _out_dir: Option<String>,
            _projections: Option<Vec<String>>,
            _config_path: Option<String>,
            _lint_strict: Option<bool>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_dist(
            &self,
            _source_dir: String,
            _output_path: Option<String>,
            _out_dir: Option<String>,
            _preset: Option<String>,
            _project_root: Option<String>,
            _projections: Option<Vec<String>>,
            _config_path: Option<String>,
            _lint_strict: Option<bool>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_info(&self, _pkg: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_search(
            &self,
            _query: Option<String>,
            _category: Option<String>,
            _installed_only: Option<bool>,
            _limit: Option<i32>,
            _sort: Option<String>,
            _filter: Option<serde_json::Value>,
            _fields: Option<Vec<String>>,
            _verbose: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_scaffold(
            &self,
            _name: String,
            _target_dir: Option<String>,
            _category: Option<String>,
            _description: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_read_init_lua(&self, _name: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn info(&self) -> String {
            "noop".into()
        }
    }

    // ── ResourceCatalog read tests (types — sync-converted to async) ─────

    #[tokio::test]
    async fn read_types_alc_d_lua_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let types_dir = tmp.path().join("types");
        std::fs::create_dir_all(&types_dir).unwrap();
        std::fs::write(types_dir.join("alc.d.lua"), "-- stub content").unwrap();

        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let result = catalog.read("alc://types/alc.d.lua").await.unwrap();
        assert_eq!(result.contents.len(), 1);
        match &result.contents[0] {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => {
                assert_eq!(text, "-- stub content");
                assert_eq!(mime_type.as_deref(), Some("text/x-lua"));
            }
            _ => panic!("expected TextResourceContents"),
        }
    }

    #[tokio::test]
    async fn read_types_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let err = catalog.read("alc://types/alc.d.lua").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("resource not found"), "got: {msg}");
    }

    #[test]
    fn list_fixed_returns_two_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let fixed = catalog.list_fixed();
        assert_eq!(fixed.len(), 2);
        assert_eq!(fixed[0].raw.uri, "alc://types/alc.d.lua");
        assert_eq!(fixed[1].raw.uri, "alc://types/alc_shapes.d.lua");
    }

    #[test]
    fn list_templates_returns_7() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let templates = catalog.list_templates();
        assert_eq!(templates.len(), 7, "expected exactly 7 templates");
        // Spot-check uri_template and name fields.
        assert_eq!(
            templates[0].raw.uri_template,
            "alc://packages/{name}/init.lua"
        );
        assert_eq!(templates[6].raw.uri_template, "alc://logs/{session_id}");
    }

    // ── Template dispatch tests ───────────────────────────────────────────
    //
    // For these tests we use a custom `FakeEngine` that returns controlled
    // responses for the specific methods under test.

    struct FakeEngine {
        pkg_init_lua: Option<Result<String, String>>,
        pkg_list: Option<Result<String, String>>,
        card_get: Option<Result<String, String>>,
        card_samples: Option<(Option<usize>, Option<usize>, Result<String, String>)>,
        scenario_show: Option<Result<String, String>>,
        eval_detail: Option<Result<String, String>>,
        log_view: Option<(Option<usize>, Option<usize>, Result<String, String>)>,
    }

    impl FakeEngine {
        fn ok_init_lua(src: &str) -> Self {
            Self {
                pkg_init_lua: Some(Ok(src.to_string())),
                ..Self::noop()
            }
        }
        fn ok_pkg_list(json: &str) -> Self {
            Self {
                pkg_list: Some(Ok(json.to_string())),
                ..Self::noop()
            }
        }
        fn ok_card_get(json: &str) -> Self {
            Self {
                card_get: Some(Ok(json.to_string())),
                ..Self::noop()
            }
        }
        fn ok_card_samples(
            expected_offset: Option<usize>,
            expected_limit: Option<usize>,
            json: &str,
        ) -> Self {
            Self {
                card_samples: Some((expected_offset, expected_limit, Ok(json.to_string()))),
                ..Self::noop()
            }
        }
        fn ok_scenario_show(src: &str) -> Self {
            Self {
                scenario_show: Some(Ok(src.to_string())),
                ..Self::noop()
            }
        }
        fn ok_eval_detail(json: &str) -> Self {
            Self {
                eval_detail: Some(Ok(json.to_string())),
                ..Self::noop()
            }
        }
        fn ok_log_view(
            expected_limit: Option<usize>,
            expected_max_chars: Option<usize>,
            json: &str,
        ) -> Self {
            Self {
                log_view: Some((expected_limit, expected_max_chars, Ok(json.to_string()))),
                ..Self::noop()
            }
        }
        fn noop() -> Self {
            Self {
                pkg_init_lua: None,
                pkg_list: None,
                card_get: None,
                card_samples: None,
                scenario_show: None,
                eval_detail: None,
                log_view: None,
            }
        }
    }

    #[async_trait::async_trait]
    impl EngineApi for FakeEngine {
        async fn pkg_read_init_lua(&self, _name: &str) -> Result<String, String> {
            self.pkg_init_lua
                .clone()
                .unwrap_or(Err("not configured".into()))
        }
        async fn pkg_list(
            &self,
            _project_root: Option<String>,
            _limit: Option<i32>,
            _sort: Option<String>,
            _filter: Option<serde_json::Value>,
            _fields: Option<Vec<String>>,
            _verbose: Option<String>,
        ) -> Result<String, String> {
            self.pkg_list
                .clone()
                .unwrap_or(Err("not configured".into()))
        }
        async fn card_get(&self, _card_id: &str) -> Result<String, String> {
            self.card_get
                .clone()
                .unwrap_or(Err("not configured".into()))
        }
        async fn card_samples(
            &self,
            _card_id: &str,
            offset: Option<usize>,
            limit: Option<usize>,
            _where_: Option<serde_json::Value>,
        ) -> Result<String, String> {
            if let Some((exp_offset, exp_limit, ref result)) = self.card_samples {
                assert_eq!(offset, exp_offset, "card_samples offset mismatch");
                assert_eq!(limit, exp_limit, "card_samples limit mismatch");
                result.clone()
            } else {
                Err("not configured".into())
            }
        }
        async fn scenario_show(&self, _name: &str) -> Result<String, String> {
            self.scenario_show
                .clone()
                .unwrap_or(Err("not configured".into()))
        }
        async fn eval_detail(&self, _eval_id: &str) -> Result<String, String> {
            self.eval_detail
                .clone()
                .unwrap_or(Err("not configured".into()))
        }
        async fn log_view(
            &self,
            _session_id: Option<&str>,
            limit: Option<usize>,
            max_chars: Option<usize>,
        ) -> Result<String, String> {
            if let Some((exp_limit, exp_max_chars, ref result)) = self.log_view {
                assert_eq!(limit, exp_limit, "log_view limit mismatch");
                assert_eq!(max_chars, exp_max_chars, "log_view max_chars mismatch");
                result.clone()
            } else {
                Err("not configured".into())
            }
        }
        // ── All other methods are noop stubs ────────────────────────────────
        async fn run(
            &self,
            _: Option<String>,
            _: Option<String>,
            _: Option<serde_json::Value>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn advice(
            &self,
            _: &str,
            _: Option<String>,
            _: Option<serde_json::Value>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn continue_single(
            &self,
            _: &str,
            _: String,
            _: Option<&str>,
            _: Option<algocline_core::TokenUsage>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn continue_batch(
            &self,
            _: &str,
            _: Vec<algocline_core::QueryResponse>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn status(
            &self,
            _: Option<&str>,
            _: Option<serde_json::Value>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval(
            &self,
            _: Option<String>,
            _: Option<String>,
            _: Option<String>,
            _: &str,
            _: Option<serde_json::Value>,
            _: bool,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval_history(&self, _: Option<&str>, _: usize) -> Result<String, String> {
            Err("noop".into())
        }
        async fn eval_compare(&self, _: &str, _: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn scenario_list(&self) -> Result<String, String> {
            Err("noop".into())
        }
        async fn scenario_install(&self, _: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_link(
            &self,
            _: String,
            _: Option<String>,
            _: Option<bool>,
            _: Option<String>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_install(&self, _: String, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_unlink(&self, _: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_remove(
            &self,
            _: &str,
            _: Option<String>,
            _: Option<String>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_repair(&self, _: Option<String>, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_doctor(&self, _: Option<String>, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn add_note(&self, _: &str, _: &str, _: Option<&str>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn stats(&self, _: Option<&str>, _: Option<u64>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn init(&self, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn update(&self, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn migrate(&self, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_list(&self, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_find(
            &self,
            _: Option<String>,
            _: Option<serde_json::Value>,
            _: Option<serde_json::Value>,
            _: Option<usize>,
            _: Option<usize>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_alias_list(&self, _: Option<String>) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_get_by_alias(&self, _: &str) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_alias_set(
            &self,
            _: &str,
            _: &str,
            _: Option<String>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_append(&self, _: &str, _: serde_json::Value) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_install(&self, _: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn card_lineage(
            &self,
            _: &str,
            _: Option<String>,
            _: Option<usize>,
            _: Option<bool>,
            _: Option<Vec<String>>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_reindex(
            &self,
            _: Option<String>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_gendoc(
            &self,
            _: String,
            _: Option<String>,
            _: Option<Vec<String>>,
            _: Option<String>,
            _: Option<bool>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_dist(
            &self,
            _: String,
            _: Option<String>,
            _: Option<String>,
            _: Option<String>,
            _: Option<String>,
            _: Option<Vec<String>>,
            _: Option<String>,
            _: Option<bool>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_info(&self, _: String) -> Result<String, String> {
            Err("noop".into())
        }
        async fn hub_search(
            &self,
            _: Option<String>,
            _: Option<String>,
            _: Option<bool>,
            _: Option<i32>,
            _: Option<String>,
            _: Option<serde_json::Value>,
            _: Option<Vec<String>>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn pkg_scaffold(
            &self,
            _: String,
            _: Option<String>,
            _: Option<String>,
            _: Option<String>,
        ) -> Result<String, String> {
            Err("noop".into())
        }
        async fn info(&self) -> String {
            "fake".into()
        }
    }

    fn make_fake_catalog(engine: FakeEngine) -> ResourceCatalog {
        let tmp = tempfile::tempdir().unwrap();
        let app_dir = Arc::new(AppDir::new(tmp.path().to_path_buf()));
        // Leak the tempdir — tests are short-lived and OS reclaims on exit.
        std::mem::forget(tmp);
        ResourceCatalog::new(Arc::new(engine), app_dir)
    }

    // 1. read_pkg_init_lua_ok
    #[tokio::test]
    async fn read_pkg_init_lua_ok() {
        let cat = make_fake_catalog(FakeEngine::ok_init_lua("return 42"));
        let result = cat.read("alc://packages/mypkg/init.lua").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => {
                assert_eq!(text, "return 42");
                assert_eq!(mime_type.as_deref(), Some("text/x-lua"));
            }
            _ => panic!("expected text"),
        }
    }

    // 2. read_pkg_init_lua_not_found
    #[tokio::test]
    async fn read_pkg_init_lua_not_found() {
        struct NotFoundEngine;
        #[async_trait::async_trait]
        impl EngineApi for NotFoundEngine {
            async fn pkg_read_init_lua(&self, name: &str) -> Result<String, String> {
                Err(format!("pkg not found: {name}"))
            }
            async fn run(
                &self,
                _: Option<String>,
                _: Option<String>,
                _: Option<serde_json::Value>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn advice(
                &self,
                _: &str,
                _: Option<String>,
                _: Option<serde_json::Value>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn continue_single(
                &self,
                _: &str,
                _: String,
                _: Option<&str>,
                _: Option<algocline_core::TokenUsage>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn continue_batch(
                &self,
                _: &str,
                _: Vec<algocline_core::QueryResponse>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn status(
                &self,
                _: Option<&str>,
                _: Option<serde_json::Value>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn eval(
                &self,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
                _: &str,
                _: Option<serde_json::Value>,
                _: bool,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn eval_history(&self, _: Option<&str>, _: usize) -> Result<String, String> {
                Err("noop".into())
            }
            async fn eval_detail(&self, _: &str) -> Result<String, String> {
                Err("noop".into())
            }
            async fn eval_compare(&self, _: &str, _: &str) -> Result<String, String> {
                Err("noop".into())
            }
            async fn scenario_list(&self) -> Result<String, String> {
                Err("noop".into())
            }
            async fn scenario_show(&self, _: &str) -> Result<String, String> {
                Err("noop".into())
            }
            async fn scenario_install(&self, _: String) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_link(
                &self,
                _: String,
                _: Option<String>,
                _: Option<bool>,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_list(
                &self,
                _: Option<String>,
                _: Option<i32>,
                _: Option<String>,
                _: Option<serde_json::Value>,
                _: Option<Vec<String>>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_install(&self, _: String, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_unlink(&self, _: String) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_remove(
                &self,
                _: &str,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_repair(
                &self,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_doctor(
                &self,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn add_note(&self, _: &str, _: &str, _: Option<&str>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn log_view(
                &self,
                _: Option<&str>,
                _: Option<usize>,
                _: Option<usize>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn stats(&self, _: Option<&str>, _: Option<u64>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn init(&self, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn update(&self, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn migrate(&self, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_list(&self, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_get(&self, _: &str) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_find(
                &self,
                _: Option<String>,
                _: Option<serde_json::Value>,
                _: Option<serde_json::Value>,
                _: Option<usize>,
                _: Option<usize>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_alias_list(&self, _: Option<String>) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_get_by_alias(&self, _: &str) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_alias_set(
                &self,
                _: &str,
                _: &str,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_append(&self, _: &str, _: serde_json::Value) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_install(&self, _: String) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_samples(
                &self,
                _: &str,
                _: Option<usize>,
                _: Option<usize>,
                _: Option<serde_json::Value>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn card_lineage(
                &self,
                _: &str,
                _: Option<String>,
                _: Option<usize>,
                _: Option<bool>,
                _: Option<Vec<String>>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn hub_reindex(
                &self,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn hub_gendoc(
                &self,
                _: String,
                _: Option<String>,
                _: Option<Vec<String>>,
                _: Option<String>,
                _: Option<bool>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn hub_dist(
                &self,
                _: String,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
                _: Option<Vec<String>>,
                _: Option<String>,
                _: Option<bool>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn hub_info(&self, _: String) -> Result<String, String> {
                Err("noop".into())
            }
            async fn hub_search(
                &self,
                _: Option<String>,
                _: Option<String>,
                _: Option<bool>,
                _: Option<i32>,
                _: Option<String>,
                _: Option<serde_json::Value>,
                _: Option<Vec<String>>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn pkg_scaffold(
                &self,
                _: String,
                _: Option<String>,
                _: Option<String>,
                _: Option<String>,
            ) -> Result<String, String> {
                Err("noop".into())
            }
            async fn info(&self) -> String {
                "fake".into()
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let app_dir = Arc::new(AppDir::new(tmp.path().to_path_buf()));
        let cat = ResourceCatalog::new(Arc::new(NotFoundEngine), app_dir);
        let err = cat
            .read("alc://packages/missing/init.lua")
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("pkg not found") || msg.contains("internal"),
            "got: {msg}"
        );
    }

    // 3. read_pkg_meta_ok
    #[tokio::test]
    async fn read_pkg_meta_ok() {
        let list_json = r#"{"packages":[{"name":"foo","version":"1.0"}]}"#;
        let cat = make_fake_catalog(FakeEngine::ok_pkg_list(list_json));
        let result = cat.read("alc://packages/foo/meta").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => {
                assert!(text.contains("foo"));
                assert_eq!(mime_type.as_deref(), Some("application/json"));
            }
            _ => panic!("expected text"),
        }
    }

    // 4. read_pkg_meta_not_found
    #[tokio::test]
    async fn read_pkg_meta_not_found() {
        let list_json = r#"{"packages":[]}"#;
        let cat = make_fake_catalog(FakeEngine::ok_pkg_list(list_json));
        let err = cat.read("alc://packages/unknown/meta").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("resource not found"), "got: {msg}");
    }

    // 5. read_card_get_ok
    #[tokio::test]
    async fn read_card_get_ok() {
        let cat = make_fake_catalog(FakeEngine::ok_card_get(r#"{"id":"abc"}"#));
        let result = cat.read("alc://cards/abc").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => {
                assert!(text.contains("abc"));
                assert_eq!(mime_type.as_deref(), Some("application/json"));
            }
            _ => panic!("expected text"),
        }
    }

    // 6. read_card_get_not_found (engine returns error)
    #[tokio::test]
    async fn read_card_get_not_found() {
        let cat = make_fake_catalog(FakeEngine::noop());
        let err = cat.read("alc://cards/missing").await.unwrap_err();
        // NoopEngine returns "not configured" which becomes internal_error
        let msg = err.to_string();
        assert!(!msg.is_empty(), "expected non-empty error");
    }

    // 7. read_card_samples_with_pagination
    #[tokio::test]
    async fn read_card_samples_with_pagination() {
        let cat = make_fake_catalog(FakeEngine::ok_card_samples(
            Some(10),
            Some(50),
            r#"{"samples":[]}"#,
        ));
        let result = cat
            .read("alc://cards/abc/samples?offset=10&limit=50")
            .await
            .unwrap();
        assert_eq!(result.contents.len(), 1);
    }

    // 8. read_card_samples_default_limit
    #[tokio::test]
    async fn read_card_samples_default_limit() {
        // When no limit query param, default should be Some(DEFAULT_CARD_SAMPLES_LIMIT).
        let cat = make_fake_catalog(FakeEngine::ok_card_samples(
            None,
            Some(DEFAULT_CARD_SAMPLES_LIMIT),
            r#"{"samples":[]}"#,
        ));
        let result = cat.read("alc://cards/abc/samples").await.unwrap();
        assert_eq!(result.contents.len(), 1);
    }

    // 9. read_card_samples_rejects_unknown_query_param
    #[tokio::test]
    async fn read_card_samples_rejects_unknown_query_param() {
        // Unknown key "foo" must be rejected even when offset/limit are valid.
        let cat = make_fake_catalog(FakeEngine::noop());
        let err = cat
            .read("alc://cards/abc/samples?offset=0&limit=10&foo=bar")
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported query param") && msg.contains("foo"),
            "got: {msg}"
        );
    }

    // 10. read_scenario_ok
    #[tokio::test]
    async fn read_scenario_ok() {
        let cat = make_fake_catalog(FakeEngine::ok_scenario_show("-- scenario lua"));
        let result = cat.read("alc://scenarios/myscenario").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents {
                text, mime_type, ..
            } => {
                assert_eq!(text, "-- scenario lua");
                assert_eq!(mime_type.as_deref(), Some("text/x-lua"));
            }
            _ => panic!("expected text"),
        }
    }

    // 10. read_eval_ok
    #[tokio::test]
    async fn read_eval_ok() {
        let cat = make_fake_catalog(FakeEngine::ok_eval_detail(r#"{"id":"sc_1234"}"#));
        let result = cat.read("alc://eval/sc_1234").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents { mime_type, .. } => {
                assert_eq!(mime_type.as_deref(), Some("application/json"));
            }
            _ => panic!("expected text"),
        }
    }

    // 11. read_eval_bad_id_format
    #[tokio::test]
    async fn read_eval_bad_id_format() {
        let cat = make_fake_catalog(FakeEngine::noop());
        let err = cat.read("alc://eval/nounderscore").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid eval_id"), "got: {msg}");
    }

    // 12. read_logs_ok
    #[tokio::test]
    async fn read_logs_ok() {
        let cat = make_fake_catalog(FakeEngine::ok_log_view(
            Some(DEFAULT_LOGS_LIMIT),
            Some(DEFAULT_LOGS_MAX_CHARS),
            r#"{"entries":[]}"#,
        ));
        let result = cat.read("alc://logs/ses-abc123").await.unwrap();
        match &result.contents[0] {
            ResourceContents::TextResourceContents { mime_type, .. } => {
                assert_eq!(mime_type.as_deref(), Some("application/json"));
            }
            _ => panic!("expected text"),
        }
    }

    // 13. read_logs_with_pagination
    #[tokio::test]
    async fn read_logs_with_pagination() {
        let cat = make_fake_catalog(FakeEngine::ok_log_view(
            Some(20),
            Some(5000),
            r#"{"entries":[]}"#,
        ));
        let result = cat
            .read("alc://logs/ses-abc123?limit=20&max_chars=5000")
            .await
            .unwrap();
        assert_eq!(result.contents.len(), 1);
    }

    // 14. list_templates_returns_7 — already covered above
}
