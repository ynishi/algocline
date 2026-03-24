//! E2E tests for the algocline MCP server.
//!
//! Uses rmcp client to spawn the `alc` binary as a child process,
//! communicate via stdio MCP protocol, and validate responses with
//! insta snapshots.

use std::borrow::Cow;

use rmcp::{model::CallToolRequestParams, transport::TokioChildProcess, ServiceExt};
use serde_json::{json, Map, Value};

// ─── Helpers ─────────────────────────────────────────────────────

/// Build `CallToolRequestParams` from a tool name and a JSON value.
fn call_params(name: &str, args: Value) -> CallToolRequestParams {
    let arguments = match args {
        Value::Object(map) => Some(map),
        _ => None,
    };
    CallToolRequestParams {
        name: Cow::Owned(name.to_string()),
        arguments,
        meta: None,
        task: None,
    }
}

/// Build `CallToolRequestParams` with no arguments.
fn call_params_empty(name: &str) -> CallToolRequestParams {
    CallToolRequestParams {
        name: Cow::Owned(name.to_string()),
        arguments: Some(Map::new()),
        meta: None,
        task: None,
    }
}

/// Connect to the `alc` binary as an MCP client.
async fn connect() -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let bin = std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")));
    let transport = TokioChildProcess::new(tokio::process::Command::new(bin))
        .expect("failed to spawn alc server");
    ().serve(transport)
        .await
        .expect("failed to initialize MCP session")
}

/// Extract the first text content from a CallToolResult.
fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .and_then(|c| c.raw.as_text())
        .map(|t| t.text.as_str())
        .unwrap_or("")
}

/// Redact UUIDs (session IDs) from text.
fn redact_uuids(text: &str) -> String {
    let re = regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        .expect("invalid regex");
    re.replace_all(text, "<UUID>").to_string()
}

/// Redact absolute paths from text (home directory portion).
fn redact_paths(text: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        text.replace(home.to_str().unwrap_or(""), "<HOME>")
    } else {
        text.to_string()
    }
}

/// Apply all redactions.
fn redact(text: &str) -> String {
    redact_paths(&redact_uuids(text))
}

// ─── Tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_tools() {
    let client = connect().await;

    let tools = client
        .list_all_tools()
        .await
        .expect("list_all_tools failed");
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    names.sort();

    insta::assert_json_snapshot!("list_tools", names);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_info() {
    let client = connect().await;

    let result = client
        .call_tool(call_params_empty("alc_info"))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    let redacted = redact(text);

    insta::assert_snapshot!("alc_info", redacted);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_empty() {
    let client = connect().await;

    let result = client
        .call_tool(call_params_empty("alc_status"))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    insta::assert_snapshot!("alc_status_empty", text);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_run_pure_lua() {
    let client = connect().await;

    let result = client
        .call_tool(call_params("alc_run", json!({ "code": "return 1 + 2" })))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    // Parse the JSON response to check the result value
    let parsed: Value = serde_json::from_str(text).expect("response should be JSON");

    assert_eq!(parsed["status"], "completed");
    assert_eq!(parsed["result"], 3);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_run_lua_error() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_run",
            json!({ "code": "error('intentional test error')" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    // The error message should contain our intentional error
    assert!(
        text.contains("intentional test error"),
        "expected error message in response, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_continue_invalid_session() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_continue",
            json!({
                "session_id": "00000000-0000-0000-0000-000000000000",
                "response": "test"
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    // Should indicate the session was not found
    assert!(
        text.to_lowercase().contains("not found")
            || text.to_lowercase().contains("no session")
            || text.to_lowercase().contains("unknown"),
        "expected 'not found' error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}
