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
    Annotated, ListResourceTemplatesResult, ListResourcesResult, RawResource, ReadResourceResult,
    Resource, ResourceContents, ResourceTemplate,
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
    #[allow(dead_code)]
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
    /// Template dispatch is implemented in Subtask 2. Returns an empty Vec
    /// as a placeholder.
    pub fn list_templates(&self) -> Vec<ResourceTemplate> {
        Vec::new()
    }

    /// Read a resource by URI.
    ///
    /// Fixed resources (`alc://types/...`) are backed by `AppDir::types_dir()`.
    /// Template resources are not yet implemented (Subtask 2) and will return
    /// `McpError::invalid_params` with a pending stub message.
    pub fn read(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        let parsed = parse_uri(uri).map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match parsed.service.as_str() {
            "types" => self.read_types(uri, &parsed),
            _ => Err(McpError::invalid_params(
                format!("template dispatch: pending subtask-2 (uri={uri})"),
                None,
            )),
        }
    }

    // ── Private dispatch helpers ──────────────────────────────────────────

    fn read_types(&self, uri: &str, parsed: &ParsedUri) -> Result<ReadResourceResult, McpError> {
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
}

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
///
/// Provided as a shared helper for Subtask 2's template dispatch.
pub fn err_to_mcp(s: String) -> McpError {
    McpError::internal_error(s, None)
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
        async fn info(&self) -> String {
            "noop".into()
        }
    }

    // ── ResourceCatalog read tests ────────────────────────────────────────

    #[test]
    fn read_types_alc_d_lua_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let types_dir = tmp.path().join("types");
        std::fs::create_dir_all(&types_dir).unwrap();
        std::fs::write(types_dir.join("alc.d.lua"), "-- stub content").unwrap();

        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let result = catalog.read("alc://types/alc.d.lua").unwrap();
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

    #[test]
    fn read_types_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = make_test_catalog(tmp.path().to_path_buf());
        let err = catalog.read("alc://types/alc.d.lua").unwrap_err();
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
    fn list_templates_is_empty_placeholder() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog = make_test_catalog(tmp.path().to_path_buf());
        assert!(catalog.list_templates().is_empty());
    }
}
