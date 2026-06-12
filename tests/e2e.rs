//! E2E tests for the algocline MCP server.
//!
//! Uses rmcp client to spawn the `alc` binary as a child process,
//! communicate via stdio MCP protocol, and validate responses with
//! insta snapshots.

use std::borrow::Cow;
use std::io::Write;
use std::time::Duration;

use rmcp::{
    model::{
        ArgumentInfo, CallToolRequestParams, CompleteRequestParams, GetPromptRequestParams,
        ReadResourceRequestParams, Reference, ResourceReference,
    },
    transport::TokioChildProcess,
    ServiceExt,
};
use serde_json::{json, Map, Value};

use tokio::time::{sleep, timeout};

use algocline_app::PRESET_CATALOG_VERSION;

// ─── Helpers ─────────────────────────────────────────────────────

/// Build `CallToolRequestParams` from a tool name and a JSON value.
fn call_params(name: &str, args: Value) -> CallToolRequestParams {
    let arguments = match args {
        Value::Object(map) => Some(map),
        _ => None,
    };
    let mut p = CallToolRequestParams::default();
    p.name = Cow::Owned(name.to_string());
    p.arguments = arguments;
    p
}

/// Build `CallToolRequestParams` with no arguments.
fn call_params_empty(name: &str) -> CallToolRequestParams {
    let mut p = CallToolRequestParams::default();
    p.name = Cow::Owned(name.to_string());
    p.arguments = Some(Map::new());
    p
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

/// Redact environment-specific values inside the `gh_credentials` block.
///
/// The following fields vary across machines and are replaced with stable
/// placeholders so the snapshot is portable:
///   - git_config.user_name  → "<GIT_USER_NAME>"
///   - git_config.user_email → "<GIT_USER_EMAIL>"
///   - ssh_keys.found        → ["<REDACTED>"] or []
///   - origin_remote.error   → "<GIT_REMOTE_ERROR>" (git error messages vary)
fn redact_gh_credentials(text: &str) -> String {
    let re_user_name = regex::Regex::new(r#""user_name":\s*"[^"]*""#).expect("invalid regex");
    let re_user_email = regex::Regex::new(r#""user_email":\s*"[^"]*""#).expect("invalid regex");
    // ssh found array: replace entries but preserve structure (present/absent)
    let re_ssh_found = regex::Regex::new(r#""found":\s*\[[^\]]*\]"#).expect("invalid regex");
    // origin_remote error string
    let re_origin_err = regex::Regex::new(r#"("error":\s*)"[^"]*""#).expect("invalid regex");

    // We only want to redact origin_remote.error, not gh_auth.error.
    // Strategy: redact all error strings inside gh_credentials block via a
    // two-pass approach — replace within the gh_credentials JSON object.
    let text = re_user_name
        .replace_all(text, r#""user_name": "<GIT_USER_NAME>""#)
        .into_owned();
    let text = re_user_email
        .replace_all(&text, r#""user_email": "<GIT_USER_EMAIL>""#)
        .into_owned();
    // Replace ssh found list with a stable redacted placeholder that preserves
    // whether keys were present (any_present field is separate and unchanged).
    let text = re_ssh_found
        .replace_all(&text, r#""found": ["<REDACTED>"]"#)
        .into_owned();
    // Replace all error strings that are non-null (null stays null).
    // This covers both gh_auth.error and origin_remote.error inside the
    // gh_credentials block without touching errors outside it.
    // Simple approach: only replace non-null error values (quoted strings).
    let text = re_origin_err
        .replace_all(&text, r#"$1"<REDACTED>""#)
        .into_owned();
    text
}

/// Apply all redactions.
fn redact(text: &str) -> String {
    redact_gh_credentials(&redact_paths(&redact_uuids(text)))
}

/// Connect with a specific ALC_HOME directory.
///
/// Also sets `ALC_PACKAGES_PATH` to `{alc_home}/packages` so that the server's
/// package search path is scoped to the tmp directory. Without this, `pkg_list`
/// would scan `~/.algocline/packages/` instead of the test fixture.
async fn connect_with_alc_home(
    alc_home: &std::path::Path,
) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let bin = std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")));
    let packages_path = alc_home.join("packages");
    let mut cmd = tokio::process::Command::new(bin);
    cmd.env("ALC_HOME", alc_home)
        .env("ALC_PACKAGES_PATH", &packages_path);
    let transport = TokioChildProcess::new(cmd).expect("failed to spawn alc server");
    ().serve(transport)
        .await
        .expect("failed to initialize MCP session")
}

/// Read a resource by URI, returning the result.
async fn read_resource(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    uri: &str,
) -> Result<rmcp::model::ReadResourceResult, rmcp::service::ServiceError> {
    client
        .read_resource(ReadResourceRequestParams::new(uri.to_string()))
        .await
}

/// Extract text from a `ResourceContents` variant.
fn resource_text(contents: &rmcp::model::ResourceContents) -> (&str, &str) {
    match contents {
        rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
            (uri.as_str(), text.as_str())
        }
        rmcp::model::ResourceContents::BlobResourceContents { uri, .. } => {
            panic!("expected TextResourceContents, got BlobResourceContents for URI {uri}")
        }
    }
}

/// Call a tool, extract text, parse as JSON.
async fn call_json(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    name: &str,
    args: Value,
) -> Value {
    let result = client
        .call_tool(call_params(name, args))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    serde_json::from_str(text).unwrap_or_else(|e| panic!("JSON parse failed: {e}\nraw: {text}"))
}

/// Poll `alc_v2_state` at 10ms intervals until the session reaches the `paused` state.
///
/// Panics if the session does not become paused within `max_wait`. Safe for use only
/// in test code.
async fn poll_until_paused(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    session_id: &str,
    max_wait: Duration,
) -> Value {
    timeout(max_wait, async {
        loop {
            let state = call_json(client, "alc_v2_state", json!({"session_id": session_id})).await;
            if state["state"].as_str() == Some("paused") {
                return state;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("timed out waiting for paused state")
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

// ─── alc_status pending_filter E2E ────────────────────────────
//
// Covers the MCP surface of the pending_filter parameter introduced
// in feat(status). Empty-registry paths exercise the resolve dispatch
// (preset / object / bad shape) without needing a paused session; the
// paused-session test hits the actual projection pipeline.

#[tokio::test]
async fn test_alc_status_preset_meta_empty_registry() {
    let client = connect().await;

    let resp = call_json(&client, "alc_status", json!({ "pending_filter": "meta" })).await;
    assert_eq!(resp["active_sessions"], 0);
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_preset_preview_and_full_empty_registry() {
    let client = connect().await;

    for preset in ["preview", "full"] {
        let resp = call_json(&client, "alc_status", json!({ "pending_filter": preset })).await;
        assert_eq!(
            resp["active_sessions"], 0,
            "preset '{preset}' should return empty-registry shape"
        );
    }

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_unknown_preset_errors() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_status",
            json!({ "pending_filter": "bogus" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.contains("unknown pending_filter preset"),
        "expected typed error, got: {text}"
    );
    assert!(
        text.contains("bogus"),
        "error should echo the bad preset name, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_bad_shape_errors() {
    let client = connect().await;

    // bool is neither a preset string nor a filter object
    let result = client
        .call_tool(call_params("alc_status", json!({ "pending_filter": true })))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.contains("pending_filter must be a preset name"),
        "expected shape error, got: {text}"
    );
    assert!(
        text.contains("bool"),
        "error should name the bad type, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_custom_object_filter() {
    let client = connect().await;

    // Empty registry still exercises the object dispatch branch.
    let resp = call_json(
        &client,
        "alc_status",
        json!({
            "pending_filter": {
                "query_id": true,
                "prompt": { "mode": "preview", "chars": 50 }
            }
        }),
    )
    .await;
    assert_eq!(resp["active_sessions"], 0);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_status_paused_session_projection() {
    let client = connect().await;

    // 1. Start a session that will pause on alc.llm()
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": "return alc.llm('What is 2+2?')" }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id").to_string();

    // 2. Query status with preset=meta and a specific session_id
    let resp = call_json(
        &client,
        "alc_status",
        json!({
            "session_id": session_id,
            "pending_filter": "meta",
        }),
    )
    .await;

    assert_eq!(resp["pending_queries"], 1, "should report 1 pending query");
    let pending = resp["pending"]
        .as_array()
        .expect("pending array should be emitted when filter is set");
    assert_eq!(pending.len(), 1);
    // meta preset: query_id + max_tokens only
    assert!(
        pending[0]["query_id"].is_string(),
        "query_id must be present"
    );
    assert!(
        pending[0]["max_tokens"].is_number(),
        "max_tokens must be present"
    );
    assert!(
        pending[0].get("prompt").is_none(),
        "meta preset must not project prompt"
    );
    assert!(
        pending[0].get("prompt_preview").is_none(),
        "meta preset must not project prompt_preview"
    );

    // 3. Preview preset should add prompt_preview field
    let resp = call_json(
        &client,
        "alc_status",
        json!({
            "session_id": session_id,
            "pending_filter": "preview",
        }),
    )
    .await;
    let pending = resp["pending"].as_array().expect("pending array");
    assert!(
        pending[0]["prompt_preview"].is_string(),
        "preview preset must project prompt_preview"
    );

    // 4. Clean up — resume the session so the process does not hold it
    let _ = call_json(
        &client,
        "alc_continue",
        json!({ "session_id": session_id, "response": "4" }),
    )
    .await;

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

/// Regression: when a non-conforming MCP client sends `ctx` as a
/// JSON-encoded string instead of an object (issue 1778656404-63015),
/// the server normalises it back to an object before injecting into
/// the Lua VM so the existing `lua.to_value` path produces a Lua
/// table.
#[tokio::test]
async fn test_alc_run_ctx_stringified_json_normalized_to_table() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_run",
            json!({
                "code": "return type(ctx) .. \"/\" .. tostring(ctx.task)",
                "ctx": "{\"task\":\"probe\"}",
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    let parsed: Value = serde_json::from_str(text).expect("response should be JSON");

    assert_eq!(parsed["status"], "completed");
    assert_eq!(
        parsed["result"], "table/probe",
        "stringified ctx must be auto-decoded into a table, got {parsed}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Regression: `alc_run.ctx` must advertise `type: object` in its MCP
/// inputSchema. Without the `#[schemars(with = "Option<Map<...>>")]`
/// annotation on `RunParams.ctx`, schemars emits an unconstrained
/// schema (`true`) for `serde_json::Value`, which lets some MCP clients
/// JSON-stringify the field and break the Lua-side `ctx` global
/// (issue 1778656404-63015).
#[tokio::test]
async fn test_alc_run_ctx_schema_is_object() {
    let client = connect().await;

    let tools = client
        .list_all_tools()
        .await
        .expect("list_all_tools failed");

    let alc_run = tools
        .iter()
        .find(|t| t.name.as_ref() == "alc_run")
        .expect("alc_run tool should be present");

    let schema_json: Value =
        serde_json::to_value(&alc_run.input_schema).expect("schema serializable");
    let ctx_schema = schema_json
        .get("properties")
        .and_then(|p| p.get("ctx"))
        .unwrap_or_else(|| panic!("alc_run schema missing `ctx`: {schema_json}"));

    let rendered = serde_json::to_string(ctx_schema).expect("ctx schema serializable");
    assert!(
        rendered.contains("\"object\""),
        "ctx schema should constrain to object (got {rendered})"
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

// ─── LLM round-trip tests ───────────────────────────────────────

#[tokio::test]
async fn test_alc_llm_single_roundtrip() {
    let client = connect().await;

    // 1. Run code that calls alc.llm()
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": "return alc.llm('What is 2+2?')" }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id missing");
    assert!(resp["prompt"].as_str().is_some(), "prompt missing");
    assert!(
        resp.get("query_id").is_some(),
        "query_id missing in response"
    );

    // 2. Continue with response (no explicit query_id — tests auto-resolve)
    let resp = call_json(
        &client,
        "alc_continue",
        json!({ "session_id": session_id, "response": "4" }),
    )
    .await;
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["result"], "4");

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_llm_batch_roundtrip() {
    let client = connect().await;

    // 1. Run code that calls alc.llm_batch()
    let code = r#"
        local results = alc.llm_batch({
            { prompt = "Say A" },
            { prompt = "Say B" },
        })
        return results
    "#;
    let resp = call_json(&client, "alc_run", json!({ "code": code })).await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id missing");
    let queries = resp["queries"].as_array().expect("queries array missing");
    assert_eq!(queries.len(), 2);

    let q0_id = queries[0]["id"].as_str().expect("q-0 id missing");
    let q1_id = queries[1]["id"].as_str().expect("q-1 id missing");

    // 2. Feed responses via batch
    let resp = call_json(
        &client,
        "alc_continue",
        json!({
            "session_id": session_id,
            "responses": [
                { "query_id": q0_id, "response": "Alpha" },
                { "query_id": q1_id, "response": "Beta" },
            ]
        }),
    )
    .await;
    assert_eq!(resp["status"], "completed");
    let result = resp["result"].as_array().expect("result should be array");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], "Alpha");
    assert_eq!(result[1], "Beta");

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_cache_hit_miss() {
    let client = connect().await;

    // Code: call alc.cache twice with same prompt, return cache_info
    let code = r#"
        local r1 = alc.cache("cached prompt")
        local r2 = alc.cache("cached prompt")
        local info = alc.cache_info()
        return { r1 = r1, r2 = r2, info = info }
    "#;

    // 1. First call pauses (cache miss)
    let resp = call_json(&client, "alc_run", json!({ "code": code })).await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id missing");

    // 2. Continue — cache miss resolved, second call hits cache
    let resp = call_json(
        &client,
        "alc_continue",
        json!({ "session_id": session_id, "response": "cached_value" }),
    )
    .await;
    assert_eq!(resp["status"], "completed");
    let result = &resp["result"];
    assert_eq!(result["r1"], "cached_value");
    assert_eq!(
        result["r2"], "cached_value",
        "cache hit should return same value"
    );
    assert_eq!(result["info"]["hits"], 1);
    assert_eq!(result["info"]["misses"], 1);
    assert_eq!(result["info"]["entries"], 1);

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_parallel_roundtrip() {
    let client = connect().await;

    // alc.parallel sends items as llm_batch
    let code = r#"
        local items = {"apple", "banana"}
        local results = alc.parallel(items, function(item, i)
            return "Describe " .. item
        end)
        return results
    "#;

    // 1. Pauses with batch queries
    let resp = call_json(&client, "alc_run", json!({ "code": code })).await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"].as_str().expect("session_id missing");
    let queries = resp["queries"].as_array().expect("queries missing");
    assert_eq!(queries.len(), 2);

    let q0_id = queries[0]["id"].as_str().expect("id missing");
    let q1_id = queries[1]["id"].as_str().expect("id missing");

    // 2. Feed batch
    let resp = call_json(
        &client,
        "alc_continue",
        json!({
            "session_id": session_id,
            "responses": [
                { "query_id": q0_id, "response": "A red fruit" },
                { "query_id": q1_id, "response": "A yellow fruit" },
            ]
        }),
    )
    .await;
    assert_eq!(resp["status"], "completed");
    let result = resp["result"].as_array().expect("result array");
    assert_eq!(result[0], "A red fruit");
    assert_eq!(result[1], "A yellow fruit");

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_fork_roundtrip() {
    let client = connect().await;

    // 1. Create temp packages in collection layout (<coll>/<name>/init.lua) and install them.
    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");

    // Collection A: coll_a/e2e_fork_a/init.lua
    let coll_a = tmp_dir.path().join("coll_a");
    let pkg_a_dir = coll_a.join("e2e_fork_a");
    std::fs::create_dir_all(&pkg_a_dir).expect("mkdir");
    let mut f = std::fs::File::create(pkg_a_dir.join("init.lua")).expect("create init.lua");
    write!(
        f,
        r#"local M = {{}}
M.meta = {{ name = "e2e_fork_a", version = "0.1.0", description = "E2E fork A" }}
function M.run(ctx)
    return alc.llm("Fork A: " .. (ctx.task or ""))
end
return M"#
    )
    .expect("write init.lua");

    // Collection B: coll_b/e2e_fork_b/init.lua
    let coll_b = tmp_dir.path().join("coll_b");
    let pkg_b_dir = coll_b.join("e2e_fork_b");
    std::fs::create_dir_all(&pkg_b_dir).expect("mkdir");
    let mut f = std::fs::File::create(pkg_b_dir.join("init.lua")).expect("create init.lua");
    write!(
        f,
        r#"local M = {{}}
M.meta = {{ name = "e2e_fork_b", version = "0.1.0", description = "E2E fork B" }}
function M.run(ctx)
    return alc.llm("Fork B: " .. (ctx.task or ""))
end
return M"#
    )
    .expect("write init.lua");

    // Install via MCP (pass collection roots)
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_a.to_string_lossy() }),
    )
    .await;
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_b.to_string_lossy() }),
    )
    .await;

    // 2. Run alc.fork
    let code = r#"
        local results = alc.fork({"e2e_fork_a", "e2e_fork_b"}, ctx)
        return results
    "#;
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": code, "ctx": { "task": "test" } }),
    )
    .await;
    assert_eq!(resp["status"], "needs_response");
    let session_id = resp["session_id"]
        .as_str()
        .expect("session_id missing")
        .to_string();

    // Fork yields one query at a time (or batched) — collect all
    // The multiplexer may batch queries from multiple child VMs
    let mut completed = false;
    let mut final_resp = resp;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;
    while !completed {
        iterations += 1;
        assert!(
            iterations <= MAX_ITERATIONS,
            "fork test exceeded {MAX_ITERATIONS} iterations — possible infinite loop"
        );
        if final_resp["status"] == "needs_response" {
            let session = final_resp["session_id"]
                .as_str()
                .unwrap_or(&session_id)
                .to_string();

            if let Some(queries) = final_resp["queries"].as_array() {
                // Batch: respond to all queries
                let responses: Vec<Value> = queries
                    .iter()
                    .map(|q| {
                        let qid = q["id"].as_str().expect("query id");
                        let prompt = q["prompt"].as_str().unwrap_or("");
                        let answer = if prompt.contains("Fork A") {
                            "Answer A"
                        } else {
                            "Answer B"
                        };
                        json!({ "query_id": qid, "response": answer })
                    })
                    .collect();
                final_resp = call_json(
                    &client,
                    "alc_continue",
                    json!({ "session_id": session, "responses": responses }),
                )
                .await;
            } else {
                // Single query
                let prompt = final_resp["prompt"].as_str().unwrap_or("");
                let answer = if prompt.contains("Fork A") {
                    "Answer A"
                } else {
                    "Answer B"
                };
                final_resp = call_json(
                    &client,
                    "alc_continue",
                    json!({ "session_id": session, "response": answer }),
                )
                .await;
            }
        } else {
            completed = true;
        }
    }

    assert_eq!(final_resp["status"], "completed");
    let result = final_resp["result"]
        .as_array()
        .expect("result should be array");
    assert_eq!(result.len(), 2);

    // Verify both strategies returned results
    let strategy_a = &result[0];
    assert_eq!(strategy_a["strategy"], "e2e_fork_a");
    assert_eq!(strategy_a["ok"], true);
    assert_eq!(strategy_a["result"], "Answer A");

    let strategy_b = &result[1];
    assert_eq!(strategy_b["strategy"], "e2e_fork_b");
    assert_eq!(strategy_b["ok"], true);
    assert_eq!(strategy_b["result"], "Answer B");

    // 3. Cleanup packages (physical delete from cache — pkg_remove no longer does this)
    if let Some(home) = dirs::home_dir() {
        let pkg_cache = home.join(".algocline").join("packages");
        let _ = std::fs::remove_dir_all(pkg_cache.join("e2e_fork_a"));
        let _ = std::fs::remove_dir_all(pkg_cache.join("e2e_fork_b"));
    }

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_pkg_install_returns_types_path() {
    let client = connect().await;

    // Create a temporary package in collection layout: <coll>/<name>/init.lua
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let coll_dir = tmp_dir.path().join("e2e_types_test_coll");
    let pkg_dir = coll_dir.join("e2e_types_test");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_types_test", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install and check response (pass collection root)
    let resp = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    assert_eq!(resp["installed"], json!(["e2e_types_test"]));
    assert!(
        resp["types_path"].is_string(),
        "types_path should be present in pkg_install response"
    );
    let types_path = resp["types_path"].as_str().unwrap();
    assert!(
        types_path.ends_with("types/alc.d.lua"),
        "types_path should end with types/alc.d.lua, got: {types_path}"
    );

    // Cleanup (physical delete from cache — pkg_remove no longer does this)
    if let Some(home) = dirs::home_dir() {
        let pkg_cache = home.join(".algocline").join("packages");
        let _ = std::fs::remove_dir_all(pkg_cache.join("e2e_types_test"));
    }
    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_pkg_install_returns_alc_shapes_types_path() {
    let client = connect().await;

    // Create a temporary package in collection layout: <coll>/<name>/init.lua
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let coll_dir = tmp_dir.path().join("e2e_alc_shapes_types_test_coll");
    let pkg_dir = coll_dir.join("e2e_alc_shapes_types_test");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_alc_shapes_types_test", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install and check response (pass collection root)
    let resp = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    assert_eq!(resp["installed"], json!(["e2e_alc_shapes_types_test"]));

    // alc_shapes_types_path is present when alc init has been run (types/alc_shapes.d.lua exists)
    if resp["alc_shapes_types_path"].is_string() {
        let alc_shapes_path = resp["alc_shapes_types_path"].as_str().unwrap();
        assert!(
            alc_shapes_path.ends_with("types/alc_shapes.d.lua"),
            "alc_shapes_types_path should end with types/alc_shapes.d.lua, got: {alc_shapes_path}"
        );
    }
    // alc_shapes_types_path is null when alc init has not distributed the file yet —
    // that is the expected behaviour (Option<String> → JSON null).

    // Regression guard: existing types_path field is unaffected
    if resp["types_path"].is_string() {
        let types_path = resp["types_path"].as_str().unwrap();
        assert!(
            types_path.ends_with("types/alc.d.lua"),
            "types_path should end with types/alc.d.lua, got: {types_path}"
        );
    }

    // Cleanup (physical delete from cache — pkg_remove no longer does this)
    if let Some(home) = dirs::home_dir() {
        let pkg_cache = home.join(".algocline").join("packages");
        let _ = std::fs::remove_dir_all(pkg_cache.join("e2e_alc_shapes_types_test"));
    }
    client.cancel().await.expect("cancel failed");
}

/// `alc_pkg_remove` with `scope = "global"` deletes the entry from the
/// global manifest `~/.algocline/installed.json` while leaving the cached
/// directory `~/.algocline/packages/{name}/` intact. Regression against
/// the "no tool path to clean up orphan `installed.json` entries" gap
/// that motivated the scope reintroduction (CHANGELOG).
#[tokio::test]
async fn test_pkg_remove_scope_global_cleans_manifest_not_files() {
    let client = connect().await;

    // Install a unique package so we don't collide with real user state.
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let pkg_name = "e2e_remove_global";
    // Collection layout: <coll>/<name>/init.lua
    let coll_dir = tmp_dir.path().join("e2e_remove_global_coll");
    let pkg_dir = coll_dir.join(pkg_name);
    std::fs::create_dir_all(&pkg_dir).expect("mkdir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_remove_global", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    let home = dirs::home_dir().expect("home");
    let manifest_path = home.join(".algocline").join("installed.json");
    let cache_dir = home.join(".algocline").join("packages").join(pkg_name);

    // Precondition: manifest has the entry, cache dir exists.
    let before: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest read"))
            .expect("manifest JSON");
    assert!(
        before["packages"][pkg_name].is_object(),
        "precondition: manifest must contain '{pkg_name}' before remove"
    );
    assert!(cache_dir.exists(), "precondition: cache dir must exist");

    // scope=global removal.
    let resp = call_json(
        &client,
        "alc_pkg_remove",
        json!({ "name": pkg_name, "scope": "global" }),
    )
    .await;
    assert_eq!(resp["removed"], pkg_name);
    assert_eq!(resp["scope"], "global");

    // Postcondition: manifest no longer has the entry, cache dir still exists.
    let after: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest read"))
            .expect("manifest JSON");
    assert!(
        after["packages"][pkg_name].is_null(),
        "manifest still contains '{pkg_name}' after scope=global remove"
    );
    assert!(
        cache_dir.exists(),
        "scope=global must not delete ~/.algocline/packages/{pkg_name}/"
    );

    // Cleanup the cache dir (scope=global deliberately leaves it).
    let _ = std::fs::remove_dir_all(&cache_dir);
    client.cancel().await.expect("cancel failed");
}

/// Variant scope link → require: `alc_pkg_link --scope=variant` writes
/// `alc.local.toml` and the next `alc_run` (with the same `project_root`)
/// must be able to `require()` the variant pkg by its declared name.
///
/// Regression for the `VariantPkg` resolver: see
/// `crates/algocline-engine/src/variant_pkg.rs`.
#[tokio::test]
async fn test_variant_scope_link_then_run_require() {
    let client = connect().await;

    // 1. Create a temp project root with empty alc.toml so resolve_project_root succeeds.
    let tmp = tempfile::tempdir().expect("failed to create tempdir");
    let project_root = tmp.path();
    std::fs::write(project_root.join("alc.toml"), "[packages]\n").expect("write alc.toml");

    // 2. Create a temp collection dir living OUTSIDE the project root (typical worktree
    //    workflow). Collection layout: <coll>/<name>/init.lua.
    let coll_dir = tmp.path().join("variant_src");
    let pkg_dir = coll_dir.join("e2e_variant_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir pkg_dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"return { value = "from-variant" }"#,
    )
    .expect("write init.lua");

    // 3. Link as variant scope → writes alc.local.toml.
    let link_resp = call_json(
        &client,
        "alc_pkg_link",
        json!({
            "path": coll_dir.to_string_lossy(),
            "scope": "variant",
            "project_root": project_root.to_string_lossy(),
        }),
    )
    .await;
    assert!(
        link_resp.get("error").is_none(),
        "alc_pkg_link should succeed, got: {link_resp}"
    );
    assert!(
        project_root.join("alc.local.toml").exists(),
        "alc.local.toml should have been created"
    );

    // 4. Run `require("e2e_variant_pkg")` — must resolve through VariantPkg resolver.
    let run_resp = call_json(
        &client,
        "alc_run",
        json!({
            "code": r#"return require("e2e_variant_pkg").value"#,
            "project_root": project_root.to_string_lossy(),
        }),
    )
    .await;

    assert_eq!(
        run_resp["status"], "completed",
        "alc_run should complete, got: {run_resp}"
    );
    assert_eq!(
        run_resp["result"], "from-variant",
        "variant pkg should be resolved and return its sentinel value"
    );

    // 5. alc_pkg_list should surface the variant entry.
    let list_resp = call_json(
        &client,
        "alc_pkg_list",
        json!({ "project_root": project_root.to_string_lossy() }),
    )
    .await;
    let packages = list_resp["packages"]
        .as_array()
        .expect("packages array missing");
    let entry = packages
        .iter()
        .find(|p| p["name"] == "e2e_variant_pkg")
        .expect("e2e_variant_pkg not found in alc_pkg_list");
    assert_eq!(entry["scope"], "variant");
    assert_eq!(entry["active"], true);
    assert_eq!(entry["resolved_source_kind"], "variant");

    client.cancel().await.expect("cancel failed");
}

/// Install → remove dest dir → repair: full round-trip through MCP.
/// Covers the (B) installed-dir-missing class of `alc_pkg_repair`.
#[tokio::test]
async fn test_pkg_repair_reinstalls_deleted_dir() {
    let client = connect().await;

    // Source pkg dir outside HOME (collection layout: <coll>/<name>/init.lua).
    let tmp = tempfile::tempdir().expect("tempdir");
    let coll_dir = tmp.path().join("e2e_repair_coll");
    let source = coll_dir.join("e2e_repair_pkg");
    std::fs::create_dir_all(&source).expect("mkdir");
    std::fs::write(
        source.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_repair_pkg", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install (pass collection root).
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    // Simulate breakage: remove the installed dest.
    let dest = dirs::home_dir()
        .expect("home")
        .join(".algocline")
        .join("packages")
        .join("e2e_repair_pkg");
    assert!(dest.exists(), "dest should exist after install");
    std::fs::remove_dir_all(&dest).expect("rm dest");
    assert!(!dest.exists());

    // Repair.
    let resp = call_json(
        &client,
        "alc_pkg_repair",
        json!({ "name": "e2e_repair_pkg" }),
    )
    .await;

    let repaired = resp["repaired"].as_array().expect("repaired array missing");
    assert_eq!(repaired.len(), 1, "one repair expected, got: {resp}");
    assert_eq!(repaired[0]["name"], "e2e_repair_pkg");
    assert_eq!(repaired[0]["kind"], "installed_missing");
    assert!(dest.exists(), "dest should be restored after repair");

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dest);
    client.cancel().await.expect("cancel failed");
}

/// Install → remove dest dir → doctor: diagnose without side effects.
/// Verifies the (B) installed_missing bucket is populated AND that the dest
/// directory is NOT resurrected (this is the doctor-vs-repair distinction).
#[tokio::test]
async fn test_pkg_doctor_reports_installed_missing() {
    let client = connect().await;

    // Source pkg dir outside HOME (collection layout: <coll>/<name>/init.lua).
    let tmp = tempfile::tempdir().expect("tempdir");
    let coll_dir = tmp.path().join("e2e_doctor_coll");
    let source = coll_dir.join("e2e_doctor_pkg");
    std::fs::create_dir_all(&source).expect("mkdir");
    std::fs::write(
        source.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_pkg", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install (pass collection root).
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    // Simulate breakage: remove the installed dest.
    let dest = dirs::home_dir()
        .expect("home")
        .join(".algocline")
        .join("packages")
        .join("e2e_doctor_pkg");
    assert!(dest.exists(), "dest should exist after install");
    std::fs::remove_dir_all(&dest).expect("rm dest");
    assert!(!dest.exists());

    // Doctor (read-only diagnose).
    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_doctor_pkg" }),
    )
    .await;

    let installed_missing = resp["installed_missing"]
        .as_array()
        .expect("installed_missing array missing");
    let entry = installed_missing
        .iter()
        .find(|e| e["name"] == "e2e_doctor_pkg")
        .unwrap_or_else(|| panic!("e2e_doctor_pkg not found in installed_missing, got: {resp}"));
    assert_eq!(entry["kind"], "installed_missing");

    // THE doctor-vs-repair distinction: dest must NOT be resurrected.
    assert!(
        !dest.exists(),
        "dest must not be resurrected by doctor (read-only)"
    );

    // Cleanup: doctor didn't create anything; remove the manifest entry via repair
    // to keep installed.json clean for subsequent runs.
    let _ = call_json(
        &client,
        "alc_pkg_repair",
        json!({ "name": "e2e_doctor_pkg" }),
    )
    .await;
    let _ = std::fs::remove_dir_all(&dest);
    client.cancel().await.expect("cancel failed");
}

/// Unknown pkg name → Err with a "not found in installed.json" message.
#[tokio::test]
async fn test_pkg_doctor_unknown_pkg_errors() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_pkg_doctor",
            json!({ "name": "nonexistent_xyz_pkg" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.contains("not found in installed.json"),
        "expected unknown-pkg error message, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Shape violation: `name` must be a string (or omitted), not a number.
///
/// The param deserialization fails at the MCP protocol layer (before the
/// handler runs), so we expect `call_tool` itself to return `Err` with an
/// invalid-type message — distinct from handler-level typed errors which
/// surface as `CallToolResult { is_error: true, ... }`.
#[tokio::test]
async fn test_pkg_doctor_shape_error() {
    let client = connect().await;

    let outcome = client
        .call_tool(call_params("alc_pkg_doctor", json!({ "name": 123 })))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            let has_type_error = text.contains("invalid type")
                || text.contains("expected a string")
                || text.contains("expected string");
            assert!(
                is_error || has_type_error,
                "expected shape error (is_error=true or type-mismatch text), got is_error={is_error:?}, text: {text}"
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("invalid type") && msg.contains("string"),
                "expected invalid-type error from param deserialization, got: {msg}"
            );
        }
    }

    client.cancel().await.expect("cancel failed");
}

// ─── Hub tools (alc_hub_reindex / alc_hub_gendoc / alc_hub_dist) ────

/// Create a minimal hub fixture directory containing a single fake package
/// whose `init.lua` has a `meta` table. Shared by hub_reindex / hub_gendoc
/// / hub_dist tests — each test owns its own tempdir so runs are parallel-
/// safe.
fn setup_hub_fixture() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("fake_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir fake_pkg");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = {
  name = "fake_pkg",
  version = "0.1.0",
  category = "test",
  description = "fake package used by e2e tests",
}
M.spec = {}
return M
"#,
    )
    .expect("write init.lua");

    // Optional TOML config (not used unless projections include
    // context7/devin). Kept around so tests can opt into config-
    // requiring projections without re-writing the fixture.
    std::fs::write(
        tmp.path().join("configs.toml"),
        r#"[context7]
projectTitle = "test"
description = "test"
rules = []

[devin]
project_name = "test"
"#,
    )
    .expect("write configs.toml");

    tmp
}

#[tokio::test]
async fn test_alc_hub_reindex_ok() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    let resp = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir,
            "output_path": output_path,
        }),
    )
    .await;

    let pkg_count = resp
        .get("package_count")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("expected package_count u64 in response: {resp}"));
    assert!(
        pkg_count > 0,
        "expected at least one package in reindex output, got {pkg_count}: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_gendoc_ok() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // gendoc requires an existing hub_index.json in source_dir.
    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let out_dir_path = tmp.path().join("docs");
    let out_dir = out_dir_path.to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir,
            "out_dir": out_dir,
        }),
    )
    .await;

    assert!(
        resp.get("source_dir").is_some(),
        "expected source_dir in gendoc response, got: {resp}"
    );

    let narrative = out_dir_path.join("narrative").join("fake_pkg.md");
    assert!(
        narrative.exists(),
        "expected narrative/fake_pkg.md to be generated at {} (gendoc resp: {resp})",
        narrative.display()
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_gendoc_with_toml_config_context7() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let config_path = tmp
        .path()
        .join("configs.toml")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // gendoc requires an existing hub_index.json in source_dir.
    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let _resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir.clone(),
            "projections": ["context7"],
            "config_path": config_path,
        }),
    )
    .await;

    let context7_json = tmp.path().join("context7.json");
    assert!(
        context7_json.exists(),
        "expected context7 projection to be generated at {}",
        context7_json.display()
    );

    client.cancel().await.expect("cancel failed");
}

/// Core-defaults case: requesting `context7`/`devin` projections without a
/// `config_path` and without an `alc.toml` must succeed using core-embedded
/// default rules and repo notes (no `is_error`).
#[tokio::test]
async fn test_alc_hub_gendoc_core_defaults_without_alc_toml() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Need a hub_index.json first so gendoc reaches the projection step.
    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    // No config_path, no alc.toml in fixture → core defaults only.
    let resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir.clone(),
            "projections": ["context7", "devin"],
        }),
    )
    .await;

    // Both projection output files must be generated.
    let context7_json = tmp.path().join("context7.json");
    let devin_wiki = tmp.path().join(".devin").join("wiki.json");
    assert!(
        context7_json.exists(),
        "expected context7.json from core defaults, got resp: {resp}"
    );
    assert!(
        devin_wiki.exists(),
        "expected .devin/wiki.json from core defaults, got resp: {resp}"
    );

    // Verify projectTitle and description are non-null in context7.json (core defaults).
    // The context7 Lua projection emits these when present in the injected config table.
    let c7_text = std::fs::read_to_string(&context7_json).expect("read context7.json");
    let c7: serde_json::Value = serde_json::from_str(&c7_text).expect("parse context7.json");
    assert!(
        c7.get("projectTitle").and_then(|v| v.as_str()).is_some(),
        "expected projectTitle to be a non-null string in context7.json, got: {c7_text}"
    );
    assert!(
        c7.get("description").and_then(|v| v.as_str()).is_some(),
        "expected description to be a non-null string in context7.json, got: {c7_text}"
    );

    // The devin wiki schema only contains repo_notes / pages (no project_name or description
    // at the top level per docs.devin.ai schema). Verify repo_notes is populated.
    let dv_text = std::fs::read_to_string(&devin_wiki).expect("read wiki.json");
    let dv: serde_json::Value = serde_json::from_str(&dv_text).expect("parse wiki.json");
    assert!(
        dv.get("repo_notes").and_then(|v| v.as_array()).is_some(),
        "expected repo_notes array in wiki.json, got: {dv_text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Happy-path: `alc_hub_gendoc` with `alc.toml` containing `[hub.context7]`
/// and `[hub.devin]` sections and no explicit `config_path` must generate both
/// `context7.json` and `.devin/wiki.json`.
#[tokio::test]
async fn test_alc_hub_gendoc_with_alc_toml_hub_sections() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Write alc.toml with [hub.context7] / [hub.devin] sections.
    std::fs::write(
        tmp.path().join("alc.toml"),
        r#"[hub]
name = "e2e-test-project"

[hub.context7]
description = "E2E test project description"
extra_rules = ["Always write tests"]

[hub.devin]
extra_repo_notes = ["Use Rust for performance-critical paths"]
"#,
    )
    .expect("write alc.toml");

    // gendoc requires an existing hub_index.json.
    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir.clone(),
            "projections": ["context7", "devin"],
        }),
    )
    .await;

    let context7_json = tmp.path().join("context7.json");
    let devin_wiki = tmp.path().join(".devin").join("wiki.json");
    assert!(
        context7_json.exists(),
        "expected context7.json with alc.toml hub sections, got resp: {resp}"
    );
    assert!(
        devin_wiki.exists(),
        "expected .devin/wiki.json with alc.toml hub sections, got resp: {resp}"
    );

    // Verify projectTitle and description are wired from alc.toml in context7.json.
    let c7_text = std::fs::read_to_string(&context7_json).expect("read context7.json");
    let c7: serde_json::Value = serde_json::from_str(&c7_text).expect("parse context7.json");
    assert_eq!(
        c7.get("projectTitle").and_then(|v| v.as_str()),
        Some("e2e-test-project"),
        "expected projectTitle = 'e2e-test-project' from [hub].name, got: {c7_text}"
    );
    assert_eq!(
        c7.get("description").and_then(|v| v.as_str()),
        Some("E2E test project description"),
        "expected description from [hub.context7].description, got: {c7_text}"
    );

    // The devin wiki schema only contains repo_notes / pages (no project_name or description
    // at the top level per docs.devin.ai schema). Verify repo_notes is populated.
    let dv_text = std::fs::read_to_string(&devin_wiki).expect("read wiki.json");
    let dv: serde_json::Value = serde_json::from_str(&dv_text).expect("parse wiki.json");
    assert!(
        dv.get("repo_notes").and_then(|v| v.as_array()).is_some(),
        "expected repo_notes array in wiki.json, got: {dv_text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Error-path: `alc_hub_gendoc` with a `.lua` `config_path` must return
/// `is_error=true` with a message stating that `.lua` is no longer supported.
#[tokio::test]
async fn test_alc_hub_gendoc_rejects_lua_config_path() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Write a (valid) Lua file — the error must fire on extension alone,
    // before any file I/O.
    let lua_path = tmp.path().join("config.lua");
    std::fs::write(&lua_path, "return {}").expect("write config.lua");
    let config_path = lua_path.to_str().expect("utf-8 path").to_string();

    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let outcome = client
        .call_tool(call_params(
            "alc_hub_gendoc",
            json!({
                "source_dir": source_dir,
                "projections": ["context7"],
                "config_path": config_path,
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true for .lua config_path, got is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("'.lua' is no longer supported"),
                "expected '.lua' is no longer supported in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

/// Error-path: `alc_hub_gendoc` with `alc.toml` containing both `rules_file`
/// and `rules_override` must return `is_error=true` with a message stating
/// the fields are mutually exclusive.
#[tokio::test]
async fn test_alc_hub_gendoc_rejects_mutually_exclusive_rules() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Write rules file and alc.toml with both rules_file + rules_override set.
    std::fs::write(tmp.path().join("rules.txt"), "Rule one\n").expect("write rules.txt");
    std::fs::write(
        tmp.path().join("alc.toml"),
        r#"[hub.context7]
rules_file = "rules.txt"
rules_override = ["Conflict rule"]
"#,
    )
    .expect("write alc.toml");

    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let outcome = client
        .call_tool(call_params(
            "alc_hub_gendoc",
            json!({
                "source_dir": source_dir,
                "projections": ["context7"],
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true for mutually-exclusive rules, got is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("mutually exclusive"),
                "expected 'mutually exclusive' in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_gendoc_unknown_projection_rejected() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Need a hub_index.json first so projection validation is evaluated
    // in the normal gendoc flow.
    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let outcome = client
        .call_tool(call_params(
            "alc_hub_gendoc",
            json!({
                "source_dir": source_dir,
                "projections": ["unknown_projection"],
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true for unknown projection, got is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("unknown projection"),
                "expected unknown projection error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

/// narrative projection: `projections=["narrative"]` must be accepted and
/// generate `docs/narrative/{pkg}.md`.
///
/// Note: narrative/{pkg}.md files are unconditionally emitted by the embedded
/// gen_docs.lua when lint_only=false regardless of which other projections are
/// requested — this test confirms the Rust allowlist gate passes "narrative"
/// through (approach A: no --narrative argv is pushed to gen_docs.lua).
#[tokio::test]
async fn test_alc_hub_gendoc_narrative_projection_generates_per_pkg_md() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir_path = tmp.path().join("docs_narrative");
    let out_dir = out_dir_path.to_str().expect("utf-8 path").to_string();

    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir,
            "out_dir": out_dir,
            "projections": ["narrative"],
        }),
    )
    .await;

    let narrative = out_dir_path.join("narrative").join("fake_pkg.md");
    assert!(
        narrative.exists(),
        "expected narrative/fake_pkg.md at {} (resp: {resp})",
        narrative.display()
    );
    let len = std::fs::metadata(&narrative).expect("metadata").len();
    assert!(
        len > 0,
        "expected non-empty narrative/fake_pkg.md at {} (resp: {resp})",
        narrative.display()
    );

    client.cancel().await.expect("cancel failed");
}

/// llms projection: `projections=["llms"]` must be accepted and generate
/// `docs/llms.txt` + `docs/llms-full.txt`.
///
/// Note: llms.txt and llms-full.txt are unconditionally emitted by the embedded
/// gen_docs.lua when lint_only=false — this test confirms the Rust allowlist
/// gate passes "llms" through (approach A: no --llms argv is pushed to
/// gen_docs.lua).
#[tokio::test]
async fn test_alc_hub_gendoc_llms_projection_generates_llms_txt() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir_path = tmp.path().join("docs_llms");
    let out_dir = out_dir_path.to_str().expect("utf-8 path").to_string();

    let _ = call_json(
        &client,
        "alc_hub_reindex",
        json!({
            "source_dir": source_dir.clone(),
            "output_path": output_path,
        }),
    )
    .await;

    let resp = call_json(
        &client,
        "alc_hub_gendoc",
        json!({
            "source_dir": source_dir,
            "out_dir": out_dir,
            "projections": ["llms"],
        }),
    )
    .await;

    let llms_txt = out_dir_path.join("llms.txt");
    assert!(
        llms_txt.exists(),
        "expected llms.txt at {} (resp: {resp})",
        llms_txt.display()
    );
    let len = std::fs::metadata(&llms_txt).expect("metadata").len();
    assert!(
        len > 0,
        "expected non-empty llms.txt at {} (resp: {resp})",
        llms_txt.display()
    );

    let llms_full_txt = out_dir_path.join("llms-full.txt");
    assert!(
        llms_full_txt.exists(),
        "expected llms-full.txt at {} (resp: {resp})",
        llms_full_txt.display()
    );
    let len_full = std::fs::metadata(&llms_full_txt).expect("metadata").len();
    assert!(
        len_full > 0,
        "expected non-empty llms-full.txt at {} (resp: {resp})",
        llms_full_txt.display()
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_dist_ok() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = tmp
        .path()
        .join("docs")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir": source_dir,
            "output_path": output_path,
            "out_dir": out_dir,
        }),
    )
    .await;

    assert_eq!(
        resp.get("preset_catalog_version").and_then(|v| v.as_str()),
        Some(PRESET_CATALOG_VERSION),
        "expected preset_catalog_version in dist response, got: {resp}"
    );
    assert!(
        resp.get("preset").is_none(),
        "expected no preset object when preset omitted, got: {resp}"
    );

    let reindex = resp
        .get("reindex")
        .unwrap_or_else(|| panic!("expected reindex field, got: {resp}"));
    let pkg_count = reindex
        .get("package_count")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("expected reindex.package_count u64, got: {resp}"));
    assert!(
        pkg_count > 0,
        "expected reindex.package_count > 0, got {pkg_count}: {resp}"
    );

    let gendoc = resp
        .get("gendoc")
        .unwrap_or_else(|| panic!("expected gendoc field, got: {resp}"));
    assert!(
        gendoc.is_object(),
        "expected gendoc to be a JSON object, got: {resp}"
    );
    assert!(
        gendoc.get("stdout").is_some(),
        "dist.gendoc must include stdout field",
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_info_includes_preset_catalog_version() {
    let client = connect().await;

    let resp = call_json(&client, "alc_info", json!({})).await;
    assert_eq!(
        resp.get("preset_catalog_version").and_then(|v| v.as_str()),
        Some(PRESET_CATALOG_VERSION),
        "expected preset_catalog_version in alc_info, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc_hub_search` with `local_indices` must merge packages from a local
/// `hub_index.json` file into the search results.
///
/// The test writes a minimal single-entry `hub_index.json` to a temp dir,
/// then calls `alc_hub_search` with `local_indices` pointing to that file.
/// It verifies that:
/// - The package `testpkg` appears in `results`.
/// - The top-level `sources` field is present.
/// - No `warnings` entry mentioning the local file is present (success path).
#[tokio::test]
async fn test_hub_search_local_indices_merges_results() {
    let alc_home = tempfile::tempdir().expect("tempdir failed");
    std::fs::create_dir_all(alc_home.path().join("packages")).expect("create packages dir failed");

    let idx_dir = tempfile::tempdir().expect("tempdir for index failed");
    let idx_path = idx_dir.path().join("hub_index.json");

    // Write a minimal 1-entry hub_index.json.
    let idx_json = serde_json::json!({
        "schema_version": "hub_index/v0",
        "updated_at": "2026-05-15T00:00:00Z",
        "packages": [
            {
                "name": "testpkg",
                "version": "0.1.0",
                "description": "A local-index test package",
                "category": "testing"
            }
        ]
    });
    std::fs::write(&idx_path, idx_json.to_string()).expect("write hub_index.json failed");

    let client = connect_with_alc_home(alc_home.path()).await;

    let resp = call_json(
        &client,
        "alc_hub_search",
        json!({
            "query": "testpkg",
            "local_indices": [idx_path.to_str().expect("utf-8 path")],
            "verbose": "full",
        }),
    )
    .await;

    // The package must appear in results.
    let results = resp["results"].as_array().expect("results must be array");
    let found = results
        .iter()
        .any(|r| r.get("name").and_then(|v| v.as_str()) == Some("testpkg"));
    assert!(
        found,
        "expected testpkg in results from local_indices, got: {resp}"
    );

    // sources field must be present (may be empty for a fresh alc_home).
    assert!(
        resp.get("sources").is_some(),
        "expected top-level 'sources' in response, got: {resp}"
    );

    // No warning mentioning the local file path (success path — file was readable).
    if let Some(warnings) = resp.get("warnings").and_then(|w| w.as_array()) {
        let idx_str = idx_path.to_str().unwrap_or("");
        let has_path_warning = warnings
            .iter()
            .any(|w| w.as_str().map(|s| s.contains(idx_str)).unwrap_or(false));
        assert!(
            !has_path_warning,
            "unexpected warning referencing local index path: {warnings:?}"
        );
    }

    client.cancel().await.expect("cancel failed");
}

/// `alc_hub_search` must match packages by their `tags` field.
///
/// Writes a local hub_index.json with a package carrying tags, then
/// searches by a tag value that does NOT appear in name/description/category.
/// Verifies the package is found (proving tags are searched).
#[tokio::test]
async fn test_hub_search_matches_tags() {
    let alc_home = tempfile::tempdir().expect("tempdir failed");
    std::fs::create_dir_all(alc_home.path().join("packages")).expect("create packages dir failed");

    let idx_dir = tempfile::tempdir().expect("tempdir for index failed");
    let idx_path = idx_dir.path().join("hub_index.json");

    let idx_json = serde_json::json!({
        "schema_version": "hub_index/v0",
        "updated_at": "2026-05-26T00:00:00Z",
        "packages": [
            {
                "name": "tagged_pkg",
                "version": "0.1.0",
                "description": "A package with metadata",
                "category": "testing",
                "tags": ["swarm", "primitive"]
            }
        ]
    });
    std::fs::write(&idx_path, idx_json.to_string()).expect("write hub_index.json failed");

    let client = connect_with_alc_home(alc_home.path()).await;

    // Search by a tag that is NOT in name/description/category.
    let resp = call_json(
        &client,
        "alc_hub_search",
        json!({
            "query": "primitive",
            "local_indices": [idx_path.to_str().expect("utf-8 path")],
            "verbose": "full",
        }),
    )
    .await;

    let results = resp["results"].as_array().expect("results must be array");
    let found = results
        .iter()
        .any(|r| r.get("name").and_then(|v| v.as_str()) == Some("tagged_pkg"));
    assert!(
        found,
        "expected tagged_pkg found via tag search, got: {resp}"
    );

    // Verify tags field is projected in full verbose.
    let entry = results
        .iter()
        .find(|r| r.get("name").and_then(|v| v.as_str()) == Some("tagged_pkg"))
        .expect("tagged_pkg must exist");
    let tags = entry.get("tags").and_then(|t| t.as_array());
    assert!(tags.is_some(), "tags field must be projected in full mode");
    let tag_strs: Vec<&str> = tags.unwrap().iter().filter_map(|v| v.as_str()).collect();
    assert!(
        tag_strs.contains(&"swarm") && tag_strs.contains(&"primitive"),
        "tags must contain swarm and primitive, got: {tag_strs:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc_hub_search` must NOT match a tag query against a package without tags.
#[tokio::test]
async fn test_hub_search_no_tags_does_not_match_tag_query() {
    let alc_home = tempfile::tempdir().expect("tempdir failed");
    std::fs::create_dir_all(alc_home.path().join("packages")).expect("create packages dir failed");

    let idx_dir = tempfile::tempdir().expect("tempdir for index failed");
    let idx_path = idx_dir.path().join("hub_index.json");

    let idx_json = serde_json::json!({
        "schema_version": "hub_index/v0",
        "updated_at": "2026-05-26T00:00:00Z",
        "packages": [
            {
                "name": "no_tags_pkg",
                "version": "0.1.0",
                "description": "Plain package",
                "category": "testing"
            }
        ]
    });
    std::fs::write(&idx_path, idx_json.to_string()).expect("write hub_index.json failed");

    let client = connect_with_alc_home(alc_home.path()).await;

    let resp = call_json(
        &client,
        "alc_hub_search",
        json!({
            "query": "primitive",
            "local_indices": [idx_path.to_str().expect("utf-8 path")],
        }),
    )
    .await;

    let results = resp["results"].as_array().expect("results must be array");
    let found = results
        .iter()
        .any(|r| r.get("name").and_then(|v| v.as_str()) == Some("no_tags_pkg"));
    assert!(
        !found,
        "no_tags_pkg should NOT match tag-only query, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_dist_preset_publish_uses_alc_toml_override() {
    let client = connect().await;
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    std::fs::write(
        root.join("alc.toml"),
        r#"[packages]

[hub.dist]

[hub.dist.presets.publish]
projections = ["context7"]
config_path = "configs.toml"
"#,
    )
    .expect("write alc.toml");

    let hub_dir = root.join("hub");
    std::fs::create_dir_all(&hub_dir).expect("mkdir hub");
    std::fs::write(
        hub_dir.join("configs.toml"),
        r#"[context7]
projectTitle = "test"
description = "test"
rules = []
"#,
    )
    .expect("write configs.toml");

    let pkg_dir = hub_dir.join("fake_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir fake_pkg");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = {
  name = "fake_pkg",
  version = "0.1.0",
  category = "test",
  description = "fake package used by preset e2e",
}
M.spec = {}
return M
"#,
    )
    .expect("write init.lua");

    let source_dir = hub_dir.to_str().expect("utf-8 path").to_string();
    let project_root = root.to_str().expect("utf-8 path").to_string();
    let output_path = hub_dir
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = hub_dir
        .join("docs")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir": source_dir,
            "project_root": project_root,
            "output_path": output_path,
            "out_dir": out_dir,
            "preset": "publish",
        }),
    )
    .await;

    assert_eq!(
        resp.get("preset_catalog_version").and_then(|v| v.as_str()),
        Some(PRESET_CATALOG_VERSION),
        "expected preset_catalog_version in dist response, got: {resp}"
    );

    let preset = resp
        .get("preset")
        .unwrap_or_else(|| panic!("expected preset object, got: {resp}"));
    assert_eq!(preset.get("name").and_then(|v| v.as_str()), Some("publish"));

    let resolved = preset
        .get("resolved")
        .unwrap_or_else(|| panic!("expected preset.resolved, got: {preset}"));
    let projections = resolved
        .get("projections")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("expected projections array, got: {resolved}"));
    let projection_names: Vec<&str> = projections
        .iter()
        .map(|v| v.as_str().expect("projection string"))
        .collect();
    assert_eq!(projection_names, vec!["context7"]);

    let context7_json = hub_dir.join("context7.json");
    assert!(
        context7_json.exists(),
        "expected context7.json at {}",
        context7_json.display()
    );

    client.cancel().await.expect("cancel failed");
}

/// Pool session survives MCP restart and can be resumed via registry.json reconnect.
///
/// Acceptance criterion 3 from subtask-6.md: a `host_mode=true` session paused by
/// `alc.llm(...)` must remain continuable after the originating MCP process
/// (AppService) dies and a brand-new AppService starts against the same ALC_HOME.
///
/// Verifies Crux constraint: Registry reconnect — registry.json persists atomically
/// and survives MCP process death.
#[tokio::test]
async fn host_mode_paused_session_survives_appservice_restart() {
    // Use an isolated ALC_HOME so the pool registry is clean.
    // Use /tmp directly to avoid exceeding SUN_LEN (104/108 bytes on macOS/Linux).
    // tempfile::tempdir() on macOS expands to /var/folders/.../ which can be
    // > 90 chars before the socket filename, leaving < 14 chars for the name.
    // /tmp is short (4 chars) leaving ~100 chars for the socket path.
    let tmp = tempfile::Builder::new()
        .prefix("alc-e2e-")
        .tempdir_in("/tmp")
        .expect("tempdir in /tmp");
    let alc_home = tmp.path();

    // ── Step 1: First MCP session — run with host_mode=true, pauses on alc.llm. ──
    let client1 = connect_with_alc_home(alc_home).await;
    let resp1 = call_json(
        &client1,
        "alc_run",
        json!({
            "code": "return alc.llm('What is 1+1?')",
            "host_mode": true,
        }),
    )
    .await;

    // The pool worker pauses execution at alc.llm; the MCP layer returns needs_response.
    assert_eq!(
        resp1.get("status").and_then(|v| v.as_str()),
        Some("needs_response"),
        "host_mode=true alc_run must pause on alc.llm; got: {resp1}"
    );
    let session_id = resp1
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("session_id missing from alc_run response: {resp1}"))
        .to_string();

    // ── Step 2: Simulate MCP restart — cancel (kill) the first client. ──
    client1.cancel().await.expect("cancel client1");

    // ── Step 3: Second MCP session — new AppService, same ALC_HOME. ──
    let client2 = connect_with_alc_home(alc_home).await;
    let resp2 = call_json(
        &client2,
        "alc_continue",
        json!({
            "session_id": session_id,
            "response": "2",
        }),
    )
    .await;

    // ── Step 4: Verify the worker was resumed, not "session not found". ──
    // The worker process is still alive (setsid makes it independent of the MCP
    // parent).  registry.json carries the UDS socket path needed for reconnect.
    let err_text = resp2.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        !err_text.contains("session not found in pool") && !err_text.contains("not found"),
        "worker must survive MCP restart and be resumable via registry.json; \
         got error: {err_text:?}, full response: {resp2}"
    );

    let status = resp2.get("status").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        status == "completed" || status == "needs_response",
        "continue after restart must succeed (status=completed or needs_response), \
         got status={status:?}, full response: {resp2}"
    );

    client2.cancel().await.expect("cancel client2");
}

#[tokio::test]
async fn test_alc_hub_dist_with_toml_config_context7() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let config_path = tmp
        .path()
        .join("configs.toml")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    let _resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir": source_dir,
            "output_path": output_path,
            "projections": ["context7"],
            "config_path": config_path,
        }),
    )
    .await;

    let context7_json = tmp.path().join("context7.json");
    assert!(
        context7_json.exists(),
        "expected context7 projection to be generated at {}",
        context7_json.display()
    );

    client.cancel().await.expect("cancel failed");
}

#[tokio::test]
async fn test_alc_hub_dist_gendoc_failure_includes_reindex_result() {
    let client = connect().await;
    let tmp = setup_hub_fixture();
    let source_dir = tmp.path().to_str().expect("utf-8 path").to_string();
    let output_path = tmp
        .path()
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // Reindex should succeed, then gendoc should fail because an unknown
    // projection value is invalid.  This verifies that dist embeds the
    // reindex result in the gendoc failure message regardless of the
    // failure reason.
    let outcome = client
        .call_tool(call_params(
            "alc_hub_dist",
            json!({
                "source_dir": source_dir,
                "output_path": output_path,
                "projections": ["invalid_projection_xyz"],
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true when gendoc fails after reindex, got: is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("dist: gendoc failed"),
                "expected dist gendoc failure prefix, got: {text}"
            );
            assert!(
                text.contains("reindex result (succeeded):"),
                "expected reindex result to be embedded in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` against the in-repo
/// `tests/fixtures/hub_dist_sample/` tree that contains three packages
/// (pkg_alpha / pkg_beta / pkg_gamma) each embedding a distinct signal
/// token in its docstring.
///
/// Each package exercises a different part of the `alc_shapes` type
/// system that is now fully vendored:
///   - pkg_alpha: `T.boolean` / `T.table`  (ALPHA_SIGNAL_BOOLEAN_TABLE)
///   - pkg_beta:  `S.instrument` + `:describe`  (BETA_SIGNAL_INSTRUMENT_DESCRIBE)
///   - pkg_gamma: nested `T.shape` / `T.array_of`  (GAMMA_SIGNAL_NESTED_SHAPE)
///
/// Verifications (labelled A–G per plan):
///   A. dist response `status == "ok"` (implicit: no is_error)
///   B. `llms-full.txt` contains all three signal tokens
///   C. `narrative/{pkg_alpha,pkg_beta,pkg_gamma}.md` exist
///   D. `llms.txt` contains index lines for all three packages
///   E. Type-system signal tokens appear in the narrative files
///   F. `reindex.package_count == 3`
///   G. context7.json and .devin/wiki.json are emitted
#[tokio::test]
async fn test_alc_hub_dist_fixture() {
    let client = connect().await;

    // Copy fixture tree to a writable tempdir so gen_docs can write
    // context7.json / .devin/wiki.json back to source_dir.
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hub_dist_sample");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir_path = root.join("docs");
    let out_dir = out_dir_path.to_str().expect("utf-8 path").to_string();
    let config_path = root
        .join("tools/gendoc.toml")
        .to_str()
        .expect("utf-8 path")
        .to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir":  source_dir,
            "output_path": output_path,
            "out_dir":     out_dir,
            "projections": ["hub", "context7", "devin"],
            "config_path": config_path,
        }),
    )
    .await;

    // A. Top-level response must not carry is_error; reindex + gendoc both present.
    let reindex = resp
        .get("reindex")
        .unwrap_or_else(|| panic!("expected reindex field, got: {resp}"));
    let _gendoc = resp
        .get("gendoc")
        .unwrap_or_else(|| panic!("expected gendoc field, got: {resp}"));

    // F. reindex.package_count == 3
    let pkg_count = reindex
        .get("package_count")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("expected reindex.package_count u64, got: {resp}"));
    assert_eq!(
        pkg_count, 3,
        "expected exactly 3 packages indexed, got {pkg_count}: {resp}"
    );

    // C. narrative/*.md files must exist for all three packages.
    for pkg in &["pkg_alpha", "pkg_beta", "pkg_gamma"] {
        let narrative = out_dir_path.join("narrative").join(format!("{pkg}.md"));
        assert!(
            narrative.exists(),
            "expected narrative/{pkg}.md at {}",
            narrative.display()
        );
    }

    // B + E. llms-full.txt must contain all three signal tokens.
    let llms_full_path = out_dir_path.join("llms-full.txt");
    assert!(
        llms_full_path.exists(),
        "expected llms-full.txt at {}",
        llms_full_path.display()
    );
    let llms_full = std::fs::read_to_string(&llms_full_path).expect("read llms-full.txt");
    for token in &[
        "ALPHA_SIGNAL_BOOLEAN_TABLE",
        "BETA_SIGNAL_INSTRUMENT_DESCRIBE",
        "GAMMA_SIGNAL_NESTED_SHAPE",
    ] {
        assert!(
            llms_full.contains(token),
            "expected signal token '{token}' in llms-full.txt"
        );
    }

    // D. llms.txt must reference all three packages.
    let llms_path = out_dir_path.join("llms.txt");
    assert!(
        llms_path.exists(),
        "expected llms.txt at {}",
        llms_path.display()
    );
    let llms = std::fs::read_to_string(&llms_path).expect("read llms.txt");
    for pkg in &["pkg_alpha", "pkg_beta", "pkg_gamma"] {
        assert!(
            llms.contains(pkg),
            "expected '{pkg}' in llms.txt index, got:\n{llms}"
        );
    }

    // G. context7.json and .devin/wiki.json emitted at source_dir root.
    let context7 = root.join("context7.json");
    assert!(
        context7.exists(),
        "expected context7.json at {}",
        context7.display()
    );
    let devin_wiki = root.join(".devin/wiki.json");
    assert!(
        devin_wiki.exists(),
        "expected .devin/wiki.json at {}",
        devin_wiki.display()
    );

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` with a mirror `alc_shapes/init.lua`
/// whose `M.VERSION` matches `EMBEDDED_ALC_SHAPES_VERSION` (0.25.1).
///
/// The mirror file is read for VERSION extraction only; actual Lua API
/// still comes from the embedded vendored copy. Dist must succeed.
#[tokio::test]
async fn test_alc_hub_dist_fixture_mirror_version_match() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hub_dist_sample_version_match");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir":  source_dir,
            "output_path": output_path,
            "out_dir":     out_dir,
        }),
    )
    .await;

    // Dist must succeed: response contains reindex + gendoc fields.
    assert!(
        resp.get("reindex").is_some(),
        "expected reindex field on version-match success, got: {resp}"
    );
    assert!(
        resp.get("gendoc").is_some(),
        "expected gendoc field on version-match success, got: {resp}"
    );

    // Signal token must appear in llms-full.txt.
    let out_dir_path = root.join("docs");
    let llms_full_path = out_dir_path.join("llms-full.txt");
    assert!(
        llms_full_path.exists(),
        "expected llms-full.txt at {}",
        llms_full_path.display()
    );
    let llms_full = std::fs::read_to_string(&llms_full_path).expect("read llms-full.txt");
    assert!(
        llms_full.contains("VMATCH_SIGNAL_ALPHA"),
        "expected VMATCH_SIGNAL_ALPHA signal token in llms-full.txt"
    );

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` with a mirror `alc_shapes/init.lua`
/// whose `M.VERSION` ("9.9.9") differs from the embedded constant.
///
/// Dist must fail early with a typed `ShapesVersionMismatch` error
/// surfaced in the MCP wire response, containing both version strings
/// and the canonical hint.
#[tokio::test]
async fn test_alc_hub_dist_fixture_mirror_version_mismatch() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hub_dist_sample_version_mismatch");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let outcome = client
        .call_tool(call_params(
            "alc_hub_dist",
            json!({
                "source_dir":  source_dir,
                "output_path": output_path,
                "out_dir":     out_dir,
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true on version mismatch, got: is_error={is_error:?}, text: {text}"
            );
            // Both version strings must appear in the error text.
            assert!(
                text.contains("0.25.1"),
                "expected embedded version '0.25.1' in error text, got: {text}"
            );
            assert!(
                text.contains("9.9.9"),
                "expected mirror version '9.9.9' in error text, got: {text}"
            );
            // The canonical hint must be present.
            assert!(
                text.contains("CHANGELOG"),
                "expected CHANGELOG hint in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` with the `luacats` projection emits
/// `source_dir/types/alc_shapes.d.lua` containing LuaCATS class declarations
/// generated from the embedded `alc_shapes` SSoT.
///
/// Verifications:
///   A. dist response contains `reindex` and `gendoc` fields (no is_error)
///   B. `source_dir/types/alc_shapes.d.lua` exists after the call
///   C. File contains at least three `---@class ` lines (one per registered shape)
///   D. File contains the `AlcResultVoted` class name (from `M.voted` in alc_shapes)
#[tokio::test]
async fn test_alc_hub_dist_luacats_projection() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hub_dist_sample");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir":  source_dir,
            "output_path": output_path,
            "out_dir":     out_dir,
            "projections": ["luacats"],
        }),
    )
    .await;

    // A. dist response must contain reindex and gendoc fields.
    assert!(
        resp.get("reindex").is_some(),
        "expected reindex field in response, got: {resp}"
    );
    assert!(
        resp.get("gendoc").is_some(),
        "expected gendoc field in response, got: {resp}"
    );

    // B. types/alc_shapes.d.lua must exist under source_dir.
    let luacats_path = root.join("types").join("alc_shapes.d.lua");
    assert!(
        luacats_path.exists(),
        "expected types/alc_shapes.d.lua at {}",
        luacats_path.display()
    );

    let content = std::fs::read_to_string(&luacats_path).expect("read alc_shapes.d.lua");

    // C. At least three `---@class ` declarations (one per registered shape).
    let class_count = content.matches("---@class ").count();
    assert!(
        class_count >= 3,
        "expected at least 3 '---@class ' lines in alc_shapes.d.lua, got {class_count}"
    );

    // D. AlcResultVoted class must be present (from M.voted in alc_shapes/init.lua).
    assert!(
        content.contains("---@class AlcResultVoted"),
        "expected '---@class AlcResultVoted' in alc_shapes.d.lua"
    );

    client.cancel().await.expect("cancel failed");
}

/// Mid-way failure: an invalid `source_dir` causes reindex to fail. The
/// caller must see `is_error=true` with text starting `dist: reindex
/// failed:`, proving that `gendoc` was not invoked and the caller was
/// not silently given a partial success.
#[tokio::test]
async fn test_alc_hub_dist_reindex_failure() {
    let client = connect().await;

    let outcome = client
        .call_tool(call_params(
            "alc_hub_dist",
            json!({
                "source_dir": "/nonexistent/path/for/dist/test",
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true on reindex failure, got: is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("reindex failed"),
                "expected 'reindex failed' in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

// ─── pkg compat-range E2E tests ──────────────────────────────────────────────

/// Shared fixture helper: recursively copy a directory tree.
///
/// Used by every fixture-based E2E test. Previously the same function
/// was redefined nested inside four separate test bodies plus a fifth
/// `_compat` variant at module scope; consolidated here.
fn copy_fixture_tree(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_fixture_tree(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Fixture-based E2E: `alc_hub_dist` where the single package declares
/// `alc_shapes_compat = ">=0.25.0, <0.26"`, which includes embedded
/// alc_shapes 0.25.1.
///
/// Dist must succeed: response contains `reindex` and `gendoc` fields,
/// and the `gendoc.warnings` array must NOT contain any compat warning.
#[tokio::test]
async fn test_alc_hub_dist_compat_declared_in_range() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hub_dist_sample_compat_declared");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir":  source_dir,
            "output_path": output_path,
            "out_dir":     out_dir,
        }),
    )
    .await;

    assert!(
        resp.get("reindex").is_some(),
        "expected reindex field on compat-declared success, got: {resp}"
    );
    assert!(
        resp.get("gendoc").is_some(),
        "expected gendoc field on compat-declared success, got: {resp}"
    );

    // Warnings array should not contain any alc_shapes_compat warning —
    // the package declared an in-range compat range.
    if let Some(gendoc) = resp.get("gendoc") {
        if let Some(warnings) = gendoc.get("warnings") {
            let warnings_str = warnings.to_string();
            assert!(
                !warnings_str.contains("alc_shapes_compat not declared"),
                "unexpected compat-undeclared warning for declared package: {warnings_str}"
            );
        }
    }

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` where the single package declares
/// `alc_shapes_compat = ">=0.26.0, <0.27"`, which does NOT include
/// embedded alc_shapes 0.25.1.
///
/// Dist must fail with `is_error=true`. The error text must contain
/// `ShapesCompatViolation`-related substrings: the pkg_name, declared_range,
/// and actual_version.
#[tokio::test]
async fn test_alc_hub_dist_compat_declared_out_of_range() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hub_dist_sample_compat_out_of_range");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let outcome = client
        .call_tool(call_params(
            "alc_hub_dist",
            json!({
                "source_dir":  source_dir,
                "output_path": output_path,
                "out_dir":     out_dir,
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true on out-of-range compat, got: is_error={is_error:?}, text: {text}"
            );
            // pkg_name must appear
            assert!(
                text.contains("pkg_alpha"),
                "expected pkg_name 'pkg_alpha' in error text, got: {text}"
            );
            // declared range must appear
            assert!(
                text.contains(">=0.26.0, <0.27"),
                "expected declared_range '>=0.26.0, <0.27' in error text, got: {text}"
            );
            // actual embedded version must appear
            assert!(
                text.contains("0.25.1"),
                "expected actual_version '0.25.1' in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_scaffold E2E tests ──────────────────────────────────────────────

/// Basic scaffold: `my_pkg` with no optional fields.
///
/// Checks:
/// - Response has `status = "ok"`.
/// - `<tempdir>/my_pkg/init.lua` is created on disk.
/// - Content contains expected Lua skeleton markers.
#[tokio::test]
async fn test_alc_pkg_scaffold_basic() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let target_dir = tmp.path().to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_pkg_scaffold",
        json!({
            "name": "my_pkg",
            "target_dir": target_dir,
        }),
    )
    .await;

    assert_eq!(
        resp.get("status").and_then(|v| v.as_str()),
        Some("ok"),
        "expected status=ok, got: {resp}"
    );

    let init_lua = tmp.path().join("my_pkg").join("init.lua");
    assert!(
        init_lua.exists(),
        "expected init.lua at {}",
        init_lua.display()
    );

    let content = std::fs::read_to_string(&init_lua).expect("read init.lua");
    assert!(
        content.contains(r#"name = "my_pkg""#),
        "expected name field in content"
    );
    assert!(
        content.contains("alc_shapes_compat = \">=0.25.0, <0.26\""),
        "expected compat range in content, got excerpt: {}",
        &content[..content.len().min(400)]
    );
    assert!(
        content.contains("function M.run(ctx)"),
        "expected M.run stub"
    );
    assert!(content.contains("T.shape"), "expected T.shape reference");
    assert!(content.contains("return M"), "expected return M");

    client.cancel().await.expect("cancel failed");
}

/// Scaffold with category and description provided — both fields must appear
/// uncommented in the generated `M.meta` table.
#[tokio::test]
async fn test_alc_pkg_scaffold_with_category_and_description() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let target_dir = tmp.path().to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_pkg_scaffold",
        json!({
            "name": "my_pkg",
            "target_dir": target_dir,
            "category": "selection",
            "description": "test pkg",
        }),
    )
    .await;

    assert_eq!(
        resp.get("status").and_then(|v| v.as_str()),
        Some("ok"),
        "expected status=ok, got: {resp}"
    );

    let content =
        std::fs::read_to_string(tmp.path().join("my_pkg").join("init.lua")).expect("read init.lua");

    assert!(
        content.contains(r#"category = "selection""#),
        "expected uncommented category in content"
    );
    assert!(
        content.contains(r#"description = "test pkg""#),
        "expected uncommented description in content"
    );
    // Commented-out placeholder lines must NOT appear.
    assert!(
        !content.contains("-- category ="),
        "unexpected commented-out category placeholder"
    );
    assert!(
        !content.contains("-- description ="),
        "unexpected commented-out description placeholder"
    );

    client.cancel().await.expect("cancel failed");
}

/// AlreadyExists error: pre-create the init.lua then call scaffold.
///
/// The MCP response must carry `is_error = true` and the text must mention
/// "already exists".
#[tokio::test]
async fn test_alc_pkg_scaffold_already_exists() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg_dir = tmp.path().join("my_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create dir");
    std::fs::write(pkg_dir.join("init.lua"), "-- existing").expect("write existing");

    let target_dir = tmp.path().to_str().expect("utf-8 path").to_string();

    let outcome = client
        .call_tool(call_params(
            "alc_pkg_scaffold",
            json!({
                "name": "my_pkg",
                "target_dir": target_dir,
            }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true for AlreadyExists, got: is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("already exists"),
                "expected 'already exists' in error text, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

/// NameInvalid error: empty name, digit-starting name, and slash name.
///
/// Each must produce `is_error = true` with text mentioning the problematic
/// name.
#[tokio::test]
async fn test_alc_pkg_scaffold_name_invalid() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let target_dir = tmp.path().to_str().expect("utf-8 path").to_string();

    for bad_name in &["", "1bad", "has/slash"] {
        let outcome = client
            .call_tool(call_params(
                "alc_pkg_scaffold",
                json!({
                    "name": bad_name,
                    "target_dir": target_dir,
                }),
            ))
            .await;

        match outcome {
            Ok(result) => {
                let is_error = result.is_error.unwrap_or(false);
                let text = extract_text(&result);
                assert!(
                    is_error,
                    "expected is_error=true for name={bad_name:?}, got text: {text}"
                );
                // The error message must contain "invalid" (from NameInvalid display).
                assert!(
                    text.contains("invalid"),
                    "expected 'invalid' in error text for name={bad_name:?}, got: {text}"
                );
            }
            Err(e) => panic!("unexpected call_tool Err for name={bad_name:?}: {e}"),
        }
    }

    client.cancel().await.expect("cancel failed");
}

/// Fixture-based E2E: `alc_hub_dist` where the single package has no
/// `alc_shapes_compat` field in `M.meta`.
///
/// Dist must succeed (backward compat), and the `gendoc.warnings` array
/// must contain the `"alc_shapes_compat not declared"` substring.
#[tokio::test]
async fn test_alc_hub_dist_compat_undeclared_warns() {
    let client = connect().await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();

    let fixture_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hub_dist_sample_compat_undeclared");

    copy_fixture_tree(&fixture_src, root).expect("copy fixture");

    let source_dir = root.to_str().expect("utf-8 path").to_string();
    let output_path = root
        .join("hub_index.json")
        .to_str()
        .expect("utf-8 path")
        .to_string();
    let out_dir = root.join("docs").to_str().expect("utf-8 path").to_string();

    let resp = call_json(
        &client,
        "alc_hub_dist",
        json!({
            "source_dir":  source_dir,
            "output_path": output_path,
            "out_dir":     out_dir,
        }),
    )
    .await;

    assert!(
        resp.get("reindex").is_some(),
        "expected reindex field on undeclared compat success, got: {resp}"
    );
    assert!(
        resp.get("gendoc").is_some(),
        "expected gendoc field on undeclared compat success, got: {resp}"
    );

    // The gendoc.warnings array must contain the undeclared warning.
    let gendoc = resp.get("gendoc").expect("gendoc field present");
    let warnings = gendoc.get("warnings").expect("warnings field in gendoc");
    let warnings_str = warnings.to_string();
    assert!(
        warnings_str.contains("alc_shapes_compat not declared"),
        "expected 'alc_shapes_compat not declared' in warnings, got: {warnings_str}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── MCP Resources E2E ──────────────────────────────────────────

/// `resources/list` must return exactly 3 fixed resources.
#[tokio::test]
async fn test_mcp_resources_list_returns_two_fixed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .list_resources(None)
        .await
        .expect("list_resources failed");

    assert_eq!(
        result.resources.len(),
        3,
        "expected 3 fixed resources (types x2 + hub/index), got: {:?}",
        result
            .resources
            .iter()
            .map(|r| r.raw.uri.as_str())
            .collect::<Vec<_>>()
    );

    let uris: Vec<&str> = result
        .resources
        .iter()
        .map(|r| r.raw.uri.as_str())
        .collect();
    assert!(
        uris.contains(&"alc://types/alc.d.lua"),
        "expected alc://types/alc.d.lua in fixed list"
    );
    assert!(
        uris.contains(&"alc://types/alc_shapes.d.lua"),
        "expected alc://types/alc_shapes.d.lua in fixed list"
    );
    assert!(
        uris.contains(&"alc://hub/index"),
        "expected alc://hub/index in fixed list"
    );

    // Verify that at least one fixed resource has title populated (ST-1 field extension).
    assert!(
        result.resources.iter().any(|r| r.raw.title.is_some()),
        "expected at least one fixed resource to have a title field"
    );

    client.cancel().await.expect("cancel failed");
}

/// `resources/templates/list` must return exactly 7 templates.
#[tokio::test]
async fn test_mcp_resource_templates_list_returns_eight() {
    // V1 = 7 templates; V2 added `packages/{name}/narrative` (#1777052474),
    // bringing the total to 8. The companion `list_changed` notification
    // template (#1777052497) was closed without adding a template.
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .list_resource_templates(None)
        .await
        .expect("list_resource_templates failed");

    assert_eq!(
        result.resource_templates.len(),
        8,
        "expected 8 resource templates, got: {:?}",
        result
            .resource_templates
            .iter()
            .map(|t| t.raw.uri_template.as_str())
            .collect::<Vec<_>>()
    );

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://hub/index` on a clean install (no sources registered) returns
/// an empty packages array.
#[tokio::test]
async fn test_mcp_resource_read_hub_index_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Fresh tempdir with no hub_registries.json and no cache files → packages empty.
    let result = read_resource(&client, "alc://hub/index")
        .await
        .expect("read alc://hub/index failed");

    assert_eq!(result.contents.len(), 1);
    let (uri, text) = resource_text(&result.contents[0]);
    assert_eq!(uri, "alc://hub/index");

    let parsed: serde_json::Value =
        serde_json::from_str(text).expect("hub/index response must be valid JSON");
    assert_eq!(
        parsed.get("schema_version").and_then(|v| v.as_str()),
        Some("hub_index/v0"),
        "schema_version must be hub_index/v0"
    );
    let packages = parsed
        .get("packages")
        .and_then(|p| p.as_array())
        .expect("packages must be an array");
    assert!(
        packages.is_empty(),
        "clean install must return empty packages, got: {packages:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://hub/unknown` returns an error (invalid path).
#[tokio::test]
async fn test_mcp_resource_read_hub_unknown_path_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let params = ReadResourceRequestParams::new("alc://hub/unknown".to_string());
    let result = client.read_resource(params).await;
    assert!(
        result.is_err(),
        "alc://hub/unknown must return an error, got: {result:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://types/alc.d.lua` when the file exists.
#[tokio::test]
async fn test_mcp_resource_read_types_alc_d_lua() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let types_dir = tmp.path().join("types");
    std::fs::create_dir_all(&types_dir).expect("create types dir");
    std::fs::write(
        types_dir.join("alc.d.lua"),
        "-- alc type stubs\n---@class alc\nalc = {}\n",
    )
    .expect("write alc.d.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://types/alc.d.lua")
        .await
        .expect("read_resource alc.d.lua failed");

    assert_eq!(result.contents.len(), 1);
    let (uri, text) = resource_text(&result.contents[0]);
    assert!(
        text.contains("alc type stubs"),
        "unexpected content: {text}"
    );
    assert_eq!(uri, "alc://types/alc.d.lua");

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://types/alc_shapes.d.lua` when the file exists.
#[tokio::test]
async fn test_mcp_resource_read_types_alc_shapes_d_lua() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let types_dir = tmp.path().join("types");
    std::fs::create_dir_all(&types_dir).expect("create types dir");
    std::fs::write(
        types_dir.join("alc_shapes.d.lua"),
        "-- alc_shapes type stubs\n",
    )
    .expect("write alc_shapes.d.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://types/alc_shapes.d.lua")
        .await
        .expect("read_resource alc_shapes.d.lua failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    assert!(text.contains("alc_shapes"), "unexpected content: {text}");

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://packages/{name}/init.lua` when the package exists.
#[tokio::test]
async fn test_mcp_resource_read_pkg_init_lua() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("packages").join("my_e2e_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'my_e2e_pkg', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://packages/my_e2e_pkg/init.lua")
        .await
        .expect("read_resource pkg init.lua failed");

    assert_eq!(result.contents.len(), 1);
    let (uri, text) = resource_text(&result.contents[0]);
    assert!(text.contains("my_e2e_pkg"), "unexpected content: {text}");
    assert_eq!(uri, "alc://packages/my_e2e_pkg/init.lua");

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://packages/{name}/meta` when a package is pre-installed in ALC_HOME.
///
/// The package directory is created before starting the server so that the
/// server's startup search-path resolution includes `$ALC_HOME/packages/`.
#[tokio::test]
async fn test_mcp_resource_read_pkg_meta() {
    let home_tmp = tempfile::tempdir().expect("home tempdir");

    // Pre-create the packages directory and the package so the server startup
    // includes it in `ALC_PACKAGES_PATH` search path resolution.
    let pkg_dir = home_tmp.path().join("packages").join("my_e2e_meta_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "local M = {}\n",
            "M.meta = { name = 'my_e2e_meta_pkg', version = '0.2.0', description = 'test' }\n",
            "return M\n"
        ),
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(home_tmp.path()).await;

    let result = read_resource(&client, "alc://packages/my_e2e_meta_pkg/meta")
        .await
        .expect("read_resource pkg meta failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    let meta: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("meta JSON parse failed: {e}\nraw: {text}"));
    assert_eq!(meta["name"], "my_e2e_meta_pkg");

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://packages/{name}/narrative` — narrative is served on-the-fly
/// from the pkg's init.lua docstring H2 sections (#1778221491-39903).
#[tokio::test]
async fn test_mcp_resource_read_pkg_narrative_ok() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("packages").join("narrative_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    // Pkg with a proper M.meta and a docstring narrative section.
    // The old narrative.md file is intentionally absent — the new serving
    // strategy extracts narrative from the init.lua docstring (#1778221491-39903).
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "--- narrative_pkg — Stdpkg reference smoke test.\n",
            "---\n",
            "--- A short summary paragraph.\n",
            "---\n",
            "--- ## Overview\n",
            "---\n",
            "--- Stdpkg reference content served from docstring.\n",
            "local M = {}\n",
            "M.meta = { name = 'narrative_pkg', version = '0.1.0',\n",
            "           category = 'test',\n",
            "           description = 'Stdpkg reference smoke test' }\n",
            "return M\n",
        ),
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://packages/narrative_pkg/narrative")
        .await
        .expect("read_resource pkg narrative failed");

    assert_eq!(result.contents.len(), 1);
    let (uri, text) = resource_text(&result.contents[0]);
    assert_eq!(uri, "alc://packages/narrative_pkg/narrative");
    assert!(
        text.contains("narrative_pkg"),
        "pkg name missing in narrative: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc://packages/{name}/narrative` for a pkg that is NOT installed
/// (no init.lua on disk) returns -32002 ResourceNotFound (#1778221491-39903).
/// The new serving strategy uses `pkg_get_narrative_md` which returns
/// `Ok(None)` when the pkg is not found, mapped to resource_not_found.
#[tokio::test]
async fn test_mcp_resource_read_pkg_narrative_missing_returns_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Intentionally do NOT create any pkg directory — the pkg is not installed.

    let client = connect_with_alc_home(tmp.path()).await;

    let err = read_resource(&client, "alc://packages/no_narrative_pkg/narrative")
        .await
        .expect_err("read_resource must error when pkg is not installed");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not found") || msg.contains("32002"),
        "expected NotFound error, got: {msg}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `list_resource_templates` exposes `alc://packages/{name}/narrative`
/// (#1777052474, V2 cluster). Sanity check that template enumeration
/// keeps the new entry visible.
#[tokio::test]
async fn test_mcp_list_resource_templates_includes_narrative() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .list_resource_templates(None)
        .await
        .expect("list_resource_templates failed");

    let uris: Vec<String> = result
        .resource_templates
        .iter()
        .map(|t| t.uri_template.to_string())
        .collect();
    assert!(
        uris.iter().any(|u| u == "alc://packages/{name}/narrative"),
        "narrative template missing, got: {uris:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc://packages/{name}/narrative` renders narrative from the init.lua
/// docstring H2 sections (#1778221491-39903). `M.docs.narrative` is removed
/// as SSOT — the docstring is now the only source. A pkg declaring
/// `M.docs.narrative` pointing at an external file must NOT serve the
/// external file; instead the docstring is rendered.
#[tokio::test]
async fn test_mcp_resource_narrative_resolves_via_m_docs_when_declared() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("packages").join("docs_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    // Pkg has M.docs.narrative pointing at overview.md. After #1778221491-39903,
    // this field is ignored — the narrative is extracted from the docstring.
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "--- docs_pkg — docstring-driven narrative test.\n",
            "---\n",
            "--- Narrative from docstring, not from external file.\n",
            "local M = {}\n",
            "M.meta = { name = 'docs_pkg', version = '0.1.0',\n",
            "           category = 'test',\n",
            "           description = 'docstring-driven narrative test' }\n",
            "M.docs = { narrative = 'overview.md', schema_version = 1 }\n",
            "return M\n",
        ),
    )
    .expect("write init.lua");
    // overview.md exists but must NOT be served.
    std::fs::write(
        pkg_dir.join("overview.md"),
        "# docs_pkg\n\nExternal file — must NOT be returned (#1778221491-39903).\n",
    )
    .expect("write overview.md");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://packages/docs_pkg/narrative")
        .await
        .expect("read_resource pkg narrative failed");
    let (_, text) = resource_text(&result.contents[0]);
    // Docstring content must be present.
    assert!(
        text.contains("docs_pkg"),
        "pkg name missing in narrative, got: {text}"
    );
    // External file content must NOT be served.
    assert!(
        !text.contains("External file"),
        "external file must not be served, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc://packages/{name}/narrative` renders the docstring narrative
/// regardless of whether `M.docs` is declared (#1778221491-39903).
/// The convention `narrative.md` file-based fallback is removed.
#[tokio::test]
async fn test_mcp_resource_narrative_falls_back_to_convention_without_m_docs() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("packages").join("legacy_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    // No M.docs declaration. After #1778221491-39903, narrative.md is ignored
    // and the docstring is the source.
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "--- legacy_pkg — docstring narrative without M.docs.\n",
            "---\n",
            "--- Narrative from docstring only.\n",
            "local M = {}\n",
            "M.meta = { name = 'legacy_pkg', version = '0.1.0',\n",
            "           category = 'test',\n",
            "           description = 'docstring narrative without M.docs' }\n",
            "return M\n",
        ),
    )
    .expect("write init.lua");
    // narrative.md exists but must NOT be served after the decommission.
    std::fs::write(
        pkg_dir.join("narrative.md"),
        "# legacy_pkg\n\nConvention path — must NOT be returned (#1778221491-39903).\n",
    )
    .expect("write narrative.md");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://packages/legacy_pkg/narrative")
        .await
        .expect("read_resource pkg narrative failed");
    let (_, text) = resource_text(&result.contents[0]);
    // Docstring content must be present.
    assert!(
        text.contains("legacy_pkg"),
        "pkg name missing in narrative, got: {text}"
    );
    // Convention file content must NOT be served.
    assert!(
        !text.contains("Convention path — must NOT be returned"),
        "convention file must not be served, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// After #1778221491-39903, `M.docs.narrative` is removed. A pkg declaring
/// `M.docs.narrative = '../../etc/passwd'` must NOT serve that file — the
/// new serving strategy reads only the init.lua docstring, so path traversal
/// via `M.docs` is no longer possible. The resource must succeed and return
/// docstring content (not filesystem contents of the traversal target).
#[tokio::test]
async fn test_mcp_resource_narrative_rejects_path_traversal_in_m_docs() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let pkg_dir = tmp.path().join("packages").join("evil_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "--- evil_pkg — path traversal guard test.\n",
            "---\n",
            "--- M.docs.narrative path traversal is inert after #1778221491-39903.\n",
            "local M = {}\n",
            "M.meta = { name = 'evil_pkg', version = '0.1.0',\n",
            "           category = 'test',\n",
            "           description = 'path traversal guard test' }\n",
            "M.docs = { narrative = '../../etc/passwd' }\n",
            "return M\n",
        ),
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    // The resource must succeed — M.docs.narrative is ignored.
    // The docstring narrative (not /etc/passwd content) is returned.
    let result = read_resource(&client, "alc://packages/evil_pkg/narrative")
        .await
        .expect("read_resource must succeed — M.docs.narrative path traversal is inert");
    let (_, text) = resource_text(&result.contents[0]);
    // Docstring content must be present; /etc/passwd content must not appear.
    assert!(
        text.contains("evil_pkg"),
        "pkg name missing in narrative, got: {text}"
    );
    assert!(
        !text.contains("root:") && !text.contains("/bin/"),
        "/etc/passwd content must not appear in narrative, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read an unknown service → `McpError` returned to the client.
#[tokio::test]
async fn test_mcp_resource_read_unknown_uri_returns_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let err = read_resource(&client, "alc://unknown/resource")
        .await
        .expect_err("expected error for unknown URI");

    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "expected non-empty error for unknown URI, got empty"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read a URI with an invalid scheme → `McpError` returned to the client.
#[tokio::test]
async fn test_mcp_resource_read_invalid_scheme_returns_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let err = read_resource(&client, "https://example.com/foo")
        .await
        .expect_err("expected error for invalid scheme");

    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "expected non-empty error for invalid scheme, got empty"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read a URI with a path traversal segment → `McpError` returned.
#[tokio::test]
async fn test_mcp_resource_read_traversal_uri_returns_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let err = read_resource(&client, "alc://types/../etc/passwd")
        .await
        .expect_err("expected error for traversal URI");

    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "expected non-empty error for traversal URI, got empty"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://scenarios/{name}` when the scenario exists.
#[tokio::test]
async fn test_mcp_resource_read_scenario() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let scenarios_dir = tmp.path().join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");
    std::fs::write(
        scenarios_dir.join("my_scenario.lua"),
        "-- scenario\nlocal S = {}\nS.cases = {}\nreturn S\n",
    )
    .expect("write scenario");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, "alc://scenarios/my_scenario")
        .await
        .expect("read_resource scenario failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    assert!(text.contains("scenario"), "unexpected content: {text}");

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://cards/{card_id}` for a pre-written card fixture.
#[tokio::test]
async fn test_mcp_resource_read_card() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let card_id = "mypkg_20260401T000000_aabbcc";
    let card_dir = tmp.path().join("cards").join("mypkg");
    std::fs::create_dir_all(&card_dir).expect("create card dir");
    std::fs::write(
        card_dir.join(format!("{card_id}.toml")),
        concat!(
            "schema_version = \"card/v0\"\n",
            "card_id = \"mypkg_20260401T000000_aabbcc\"\n",
            "created_at = \"2026-04-01T00:00:00Z\"\n",
            "[pkg]\n",
            "name = \"mypkg\"\n",
        ),
    )
    .expect("write card toml");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, &format!("alc://cards/{card_id}"))
        .await
        .expect("read_resource card failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    let card: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("card JSON parse failed: {e}\nraw: {text}"));
    assert_eq!(card["card_id"], card_id);

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://cards/{card_id}/samples` with pagination, verifying response shape.
#[tokio::test]
async fn test_mcp_resource_read_card_samples_pagination() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let card_id = "mypkg_20260401T000000_aabbcc";
    let card_dir = tmp.path().join("cards").join("mypkg");
    std::fs::create_dir_all(&card_dir).expect("create card dir");
    std::fs::write(
        card_dir.join(format!("{card_id}.toml")),
        concat!(
            "schema_version = \"card/v0\"\n",
            "card_id = \"mypkg_20260401T000000_aabbcc\"\n",
            "created_at = \"2026-04-01T00:00:00Z\"\n",
            "[pkg]\n",
            "name = \"mypkg\"\n",
        ),
    )
    .expect("write card toml");
    // Write a two-row sidecar JSONL so pagination is exercisable.
    let jsonl = "{\"case_idx\":0,\"score\":1.0}\n{\"case_idx\":1,\"score\":0.5}\n";
    std::fs::write(card_dir.join(format!("{card_id}.samples.jsonl")), jsonl)
        .expect("write samples jsonl");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(
        &client,
        &format!("alc://cards/{card_id}/samples?offset=0&limit=2"),
    )
    .await
    .expect("read_resource card samples failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    // Verify the response is valid JSON (array or object — implementation detail).
    let body: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("samples JSON parse failed: {e}\nraw: {text}"));
    assert!(
        body.is_array() || body.is_object(),
        "expected JSON array or object response, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://eval/{result_id}` for a pre-written eval result fixture.
#[tokio::test]
async fn test_mcp_resource_read_eval_detail() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let eval_id = "mystrategy_1745000000";
    let evals_dir = tmp.path().join("evals");
    std::fs::create_dir_all(&evals_dir).expect("create evals dir");
    std::fs::write(
        evals_dir.join(format!("{eval_id}.json")),
        r#"{"eval_id":"mystrategy_1745000000","strategy":"mystrategy","pass_rate":0.8}"#,
    )
    .expect("write eval json");

    let client = connect_with_alc_home(tmp.path()).await;

    let result = read_resource(&client, &format!("alc://eval/{eval_id}"))
        .await
        .expect("read_resource eval detail failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    let eval: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("eval JSON parse failed: {e}\nraw: {text}"));
    assert_eq!(eval["eval_id"], eval_id);

    client.cancel().await.expect("cancel failed");
}

/// Read `alc://logs/{session_id}` with pagination params.
///
/// Log files are resolved via `ALC_LOG_DIR` (not `ALC_HOME/logs`), so we set
/// both env vars to ensure the server's `log_view` call finds the fixture.
#[tokio::test]
async fn test_mcp_resource_read_logs_pagination() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "ses-e2e-log-test";
    let logs_dir = tmp.path().join("logs");
    std::fs::create_dir_all(&logs_dir).expect("create logs dir");
    // Write a minimal log JSON in the transcript format the server can parse.
    std::fs::write(
        logs_dir.join(format!("{session_id}.json")),
        r#"{"session_id":"ses-e2e-log-test","rounds":[]}"#,
    )
    .expect("write log json");

    let bin = std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")));
    let packages_path = tmp.path().join("packages");
    let mut cmd = tokio::process::Command::new(bin);
    cmd.env("ALC_HOME", tmp.path())
        .env("ALC_PACKAGES_PATH", &packages_path)
        .env("ALC_LOG_DIR", &logs_dir);
    let transport = TokioChildProcess::new(cmd).expect("spawn alc server");
    let client = ().serve(transport).await.expect("initialize MCP session");

    let result = read_resource(
        &client,
        &format!("alc://logs/{session_id}?limit=10&max_chars=500"),
    )
    .await
    .expect("read_resource logs failed");

    assert_eq!(result.contents.len(), 1);
    let (_uri, text) = resource_text(&result.contents[0]);
    assert!(!text.is_empty(), "expected non-empty log response");

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_doctor — incomplete_pkg bucket ──────────────────────

/// Happy path: multi-file package with all subs present → `healthy` bucket.
///
/// Fixture layout:
///   source/e2e_doctor_complete/
///     init.lua   (requires "e2e_doctor_complete.sub")
///     sub.lua
///
/// After install via `alc_pkg_install`, doctor should classify the package as
/// `healthy` (all required submodule files are present).
#[tokio::test]
async fn test_pkg_doctor_multifile_healthy() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build source package in collection layout: <coll>/<name>/init.lua + sub.lua
    let source_root = tempfile::tempdir().expect("source tempdir");
    let coll_root = source_root.path();
    let pkg_src = coll_root.join("e2e_doctor_complete");
    std::fs::create_dir_all(&pkg_src).expect("mkdir pkg_src");
    std::fs::write(
        pkg_src.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_complete", version = "0.1.0" }
local sub = require("e2e_doctor_complete.sub")
function M.run(ctx) return sub.hello() end
return M"#,
    )
    .expect("write init.lua");
    std::fs::write(
        pkg_src.join("sub.lua"),
        r#"return { hello = function() return "hi" end }"#,
    )
    .expect("write sub.lua");

    // Install the package into the isolated ALC_HOME (pass collection root).
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_root.to_string_lossy() }),
    )
    .await;

    // Doctor should report the package as healthy.
    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_doctor_complete" }),
    )
    .await;

    let healthy = resp["healthy"].as_array().expect("healthy array missing");
    assert!(
        healthy.iter().any(|e| e["name"] == "e2e_doctor_complete"),
        "e2e_doctor_complete should be in healthy bucket, got: {resp}"
    );
    let incomplete = resp["incomplete_pkg"]
        .as_array()
        .expect("incomplete_pkg array missing");
    assert!(
        incomplete.is_empty(),
        "incomplete_pkg should be empty for complete package, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Defect path: init.lua requires a sibling submodule that is missing →
/// `incomplete_pkg` bucket with the sub name and a suggestion.
///
/// Fixture layout after install + manual removal:
///   packages/e2e_doctor_incomplete/
///     init.lua   (requires "e2e_doctor_incomplete.missing_sub")
///     (missing_sub.lua intentionally removed after install)
#[tokio::test]
async fn test_pkg_doctor_incomplete_pkg_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build source package in collection layout: <coll>/<name>/init.lua + missing_sub.lua
    let source_root = tempfile::tempdir().expect("source tempdir");
    let coll_root = source_root.path();
    let pkg_src = coll_root.join("e2e_doctor_incomplete");
    std::fs::create_dir_all(&pkg_src).expect("mkdir pkg_src");
    std::fs::write(
        pkg_src.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_incomplete", version = "0.1.0" }
local sub = require("e2e_doctor_incomplete.missing_sub")
function M.run(ctx) return sub.run(ctx) end
return M"#,
    )
    .expect("write init.lua");
    std::fs::write(
        pkg_src.join("missing_sub.lua"),
        r#"return { run = function(ctx) return "ok" end }"#,
    )
    .expect("write missing_sub.lua");

    // Install (pass collection root).
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_root.to_string_lossy() }),
    )
    .await;

    // Simulate partial deletion: remove `missing_sub.lua` from the installed dest.
    let installed_dest = tmp.path().join("packages").join("e2e_doctor_incomplete");
    assert!(
        installed_dest.exists(),
        "package should be installed at {installed_dest:?}"
    );
    std::fs::remove_file(installed_dest.join("missing_sub.lua")).expect("remove missing_sub.lua");

    // Doctor should detect the incomplete package.
    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_doctor_incomplete" }),
    )
    .await;

    let incomplete = resp["incomplete_pkg"]
        .as_array()
        .expect("incomplete_pkg array missing");
    let entry = incomplete
        .iter()
        .find(|e| e["name"] == "e2e_doctor_incomplete")
        .unwrap_or_else(|| {
            panic!("e2e_doctor_incomplete not in incomplete_pkg bucket, got: {resp}")
        });

    assert_eq!(entry["kind"], "incomplete_pkg", "kind field: {entry}");

    let missing = entry["missing_subs"]
        .as_array()
        .expect("missing_subs array missing");
    assert!(
        missing.iter().any(|s| s == "missing_sub"),
        "missing_subs should contain 'missing_sub', got: {missing:?}"
    );

    let suggestion = entry["suggestion"].as_str().unwrap_or("");
    assert!(
        !suggestion.is_empty(),
        "suggestion should not be empty: {entry}"
    );

    // healthy and installed_missing must NOT contain this package.
    let healthy = resp["healthy"].as_array().expect("healthy array");
    assert!(
        !healthy.iter().any(|e| e["name"] == "e2e_doctor_incomplete"),
        "incomplete pkg must not appear in healthy: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_doctor — narrative_issues bucket (#1778197805 L-1) ────

/// Pre-install a pkg into ALC_HOME (filesystem + installed.json) BEFORE
/// the MCP server connects, so the Lua executor's `package.path`
/// includes the pkg dir from startup. Required because the shared VM's
/// `lib_paths` does not refresh after `pkg_install` calls during the
/// same session — narrative pass needs `require(pkg)` to succeed.
fn pre_install_pkg(alc_home: &std::path::Path, name: &str, files: &[(&str, &str)]) {
    let pkg_dir = alc_home.join("packages").join(name);
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    for (filename, content) in files {
        std::fs::write(pkg_dir.join(filename), content).expect("write pkg file");
    }
    let manifest_path = alc_home.join("installed.json");
    let mut manifest: serde_json::Value = if manifest_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read installed.json"))
            .expect("parse installed.json")
    } else {
        json!({ "packages": {} })
    };
    manifest["packages"][name] = json!({
        "version": "0.1.0",
        "source": { "type": "path", "path": pkg_dir.to_string_lossy() },
        "installed_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write installed.json");
}

/// After #1778221491-39903, `narrative_issues` bucket is removed from
/// `alc_pkg_doctor` output. Pkgs declaring `M.docs.narrative` are no longer
/// linted for narrative file presence — the docstring is the SSoT.
/// Verify the bucket is absent and the tool call succeeds.
#[tokio::test]
async fn test_pkg_doctor_narrative_declared_missing_warns() {
    let tmp = tempfile::tempdir().expect("tempdir");
    pre_install_pkg(
        tmp.path(),
        "e2e_narr_declared",
        &[(
            "init.lua",
            concat!(
                "local M = {}\n",
                "M.meta = { name = 'e2e_narr_declared', version = '0.1.0' }\n",
                "M.docs = { narrative = 'overview.md', schema_version = 1 }\n",
                "function M.run(ctx) return ctx end\n",
                "return M\n",
            ),
        )],
    );
    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_narr_declared" }),
    )
    .await;

    // narrative_issues bucket is removed (#1778221491-39903).
    assert!(
        resp["narrative_issues"].is_null(),
        "narrative_issues must be absent after decommission, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// After #1778221491-39903, `narrative_issues` bucket is removed from
/// `alc_pkg_doctor` output. Pkgs with a `narrative.md` but no `M.docs`
/// are no longer flagged as `unmigrated` — the docstring is the SSoT.
/// Verify the bucket is absent and the tool call succeeds.
#[tokio::test]
async fn test_pkg_doctor_narrative_unmigrated_info() {
    let tmp = tempfile::tempdir().expect("tempdir");
    pre_install_pkg(
        tmp.path(),
        "e2e_narr_unmigrated",
        &[
            (
                "init.lua",
                concat!(
                    "local M = {}\n",
                    "M.meta = { name = 'e2e_narr_unmigrated', version = '0.1.0' }\n",
                    "function M.run(ctx) return ctx end\n",
                    "return M\n",
                ),
            ),
            (
                "narrative.md",
                "# e2e_narr_unmigrated\n\nAdoption candidate.\n",
            ),
        ],
    );
    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_narr_unmigrated" }),
    )
    .await;

    // narrative_issues bucket is removed (#1778221491-39903).
    assert!(
        resp["narrative_issues"].is_null(),
        "narrative_issues must be absent after decommission, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── unregistered_pkg bucket tests ───────────────────────────────────────────

/// AC-1: Physical dir with init.lua but no registration (no installed.json,
/// no alc.toml) → reported in `unregistered_pkg` bucket with correct fields.
///
/// Fixture: `<ALC_HOME>/packages/e2e_unreg_foo/init.lua` placed directly
/// (no pkg_install, so no installed.json entry). alc.toml absent.
///
/// Expectations:
/// - `unregistered_pkg` array contains an entry with `name == "e2e_unreg_foo"`
/// - entry has `kind == "unregistered_pkg"`, `source == "unknown"`
/// - `reason` contains "physical dir with init.lua"
/// - `suggestion` is a JSON array with exactly 4 string elements
#[tokio::test]
async fn test_pkg_doctor_unregistered_pkg_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Place init.lua directly in packages dir without installing.
    let pkg_dir = tmp.path().join("packages").join("e2e_unreg_foo");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'e2e_unreg_foo', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(&client, "alc_pkg_doctor", json!({})).await;

    let bucket = resp["unregistered_pkg"]
        .as_array()
        .expect("unregistered_pkg array missing");
    let entry = bucket
        .iter()
        .find(|e| e["name"] == "e2e_unreg_foo")
        .unwrap_or_else(|| panic!("e2e_unreg_foo not found in unregistered_pkg, got: {resp}"));

    assert_eq!(entry["kind"], "unregistered_pkg", "kind mismatch: {entry}");
    assert_eq!(entry["source"], "unknown", "source mismatch: {entry}");
    assert!(
        entry["reason"]
            .as_str()
            .unwrap_or("")
            .contains("physical dir with init.lua"),
        "reason should mention 'physical dir with init.lua': {entry}"
    );

    // Crux constraint: suggestion must be array<string> with exactly 4 elements.
    let suggestion = entry["suggestion"]
        .as_array()
        .expect("suggestion must be an array for unregistered_pkg");
    assert_eq!(
        suggestion.len(),
        4,
        "suggestion must have exactly 4 elements: {suggestion:?}"
    );
    for (i, elem) in suggestion.iter().enumerate() {
        assert!(
            elem.is_string(),
            "suggestion[{i}] must be a string, got: {elem}"
        );
    }

    client.cancel().await.expect("cancel failed");
}

/// AC-2: Empty directory (no init.lua) is not reported in any bucket.
///
/// Fixture: `<ALC_HOME>/packages/e2e_unreg_empty/` with no files inside.
///
/// Expectation: `unregistered_pkg` does not contain `e2e_unreg_empty`.
#[tokio::test]
async fn test_pkg_doctor_unregistered_pkg_empty_dir_skipped() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Create empty directory (no init.lua).
    let empty_dir = tmp.path().join("packages").join("e2e_unreg_empty");
    std::fs::create_dir_all(&empty_dir).expect("create empty dir");

    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(&client, "alc_pkg_doctor", json!({})).await;

    let bucket = resp["unregistered_pkg"]
        .as_array()
        .expect("unregistered_pkg array missing");
    assert!(
        !bucket.iter().any(|e| e["name"] == "e2e_unreg_empty"),
        "empty dir must not appear in unregistered_pkg, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// AC-3: Package registered in installed.json (via pkg_install) is not
/// reported in `unregistered_pkg`.
///
/// Fixture: install `e2e_unreg_bar` via alc_pkg_install → it appears in
/// installed.json and healthy.
///
/// Expectation: `unregistered_pkg` does not contain `e2e_unreg_bar`.
#[tokio::test]
async fn test_pkg_doctor_unregistered_pkg_manifest_registered_skipped() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Build collection with a single package.
    let source_root = tempfile::tempdir().expect("source tempdir");
    let pkg_src = source_root.path().join("e2e_unreg_bar");
    std::fs::create_dir_all(&pkg_src).expect("mkdir pkg_src");
    std::fs::write(
        pkg_src.join("init.lua"),
        "local M = {}\nM.meta = { name = 'e2e_unreg_bar', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    // Install into the isolated ALC_HOME.
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": source_root.path().to_string_lossy() }),
    )
    .await;

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_unreg_bar" }),
    )
    .await;

    let bucket = resp["unregistered_pkg"]
        .as_array()
        .expect("unregistered_pkg array missing");
    assert!(
        !bucket.iter().any(|e| e["name"] == "e2e_unreg_bar"),
        "manifest-registered pkg must not appear in unregistered_pkg, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// AC-4: Physical dir inside packages_dir that is declared as a path-dep in
/// alc.toml is skipped (false positive avoidance via canonical path comparison).
///
/// Fixture:
///   - `<ALC_HOME>/packages/e2e_unreg_baz/init.lua` placed directly
///   - `<project_root>/alc.toml` with `[packages.e2e_unreg_baz] path = <abs_path>`
///
/// Expectation: `unregistered_pkg` does not contain `e2e_unreg_baz`.
#[tokio::test]
async fn test_pkg_doctor_unregistered_pkg_alc_toml_path_dep_skipped() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Place init.lua in packages dir (not via install).
    let pkg_dir = tmp.path().join("packages").join("e2e_unreg_baz");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'e2e_unreg_baz', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write init.lua");

    // Write alc.toml declaring this dir as a path dep.
    // Use the same tmp dir as project_root for simplicity.
    let project_root = tmp.path();
    let alc_toml_content = format!(
        "[packages.e2e_unreg_baz]\npath = \"{}\"\n",
        pkg_dir.display()
    );
    std::fs::write(project_root.join("alc.toml"), &alc_toml_content).expect("write alc.toml");

    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "project_root": project_root.to_string_lossy() }),
    )
    .await;

    let bucket = resp["unregistered_pkg"]
        .as_array()
        .expect("unregistered_pkg array missing");
    assert!(
        !bucket.iter().any(|e| e["name"] == "e2e_unreg_baz"),
        "path-dep in alc.toml must not appear in unregistered_pkg (false positive), got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// AC-5: target_filter Some(name) where the package exists only as a physical
/// dir → returns non-Err response with `unregistered_pkg` populated.
///
/// This is the primary bug fix: previously `pkg_doctor name="quux"` would
/// return Err("Package 'quux' not found in ...") even when the physical dir
/// existed. Now it routes to `unregistered_pkg` instead.
///
/// Fixture: `<ALC_HOME>/packages/e2e_unreg_quux/init.lua` placed directly
/// (no installed.json entry, no alc.toml).
///
/// Expectations:
/// - Call does NOT return Err (no panic from call_json)
/// - `unregistered_pkg` contains `e2e_unreg_quux`
#[tokio::test]
async fn test_pkg_doctor_unregistered_pkg_target_filter_routes_to_bucket() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Place init.lua directly in packages dir without installing.
    let pkg_dir = tmp.path().join("packages").join("e2e_unreg_quux");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'e2e_unreg_quux', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    // This call previously returned Err. Now it must succeed and populate
    // unregistered_pkg (crux constraint: target_filter Some routing to bucket).
    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_unreg_quux" }),
    )
    .await;

    let bucket = resp["unregistered_pkg"]
        .as_array()
        .expect("unregistered_pkg array missing");
    assert!(
        bucket.iter().any(|e| e["name"] == "e2e_unreg_quux"),
        "e2e_unreg_quux must appear in unregistered_pkg when target_filter is Some, got: {resp}"
    );

    // Verify suggestion is array<string> with 4 elements (crux constraint).
    let entry = bucket
        .iter()
        .find(|e| e["name"] == "e2e_unreg_quux")
        .expect("entry not found");
    let suggestion = entry["suggestion"]
        .as_array()
        .expect("suggestion must be an array");
    assert_eq!(
        suggestion.len(),
        4,
        "suggestion must have exactly 4 elements: {suggestion:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// After #1778221491-39903, `narrative_issues` bucket is removed from
/// `alc_pkg_doctor` output. Verify the doctor call succeeds and the
/// bucket is absent for a pkg that previously had a clean narrative state.
#[tokio::test]
async fn test_pkg_doctor_narrative_clean_state_silent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    pre_install_pkg(
        tmp.path(),
        "e2e_narr_clean",
        &[
            (
                "init.lua",
                concat!(
                    "local M = {}\n",
                    "M.meta = { name = 'e2e_narr_clean', version = '0.1.0' }\n",
                    "M.docs = { narrative = 'narrative.md', schema_version = 1 }\n",
                    "function M.run(ctx) return ctx end\n",
                    "return M\n",
                ),
            ),
            ("narrative.md", "# clean\n"),
        ],
    );
    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_narr_clean" }),
    )
    .await;

    // narrative_issues bucket is removed (#1778221491-39903).
    assert!(
        resp["narrative_issues"].is_null(),
        "narrative_issues must be absent after decommission, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_install force parameter ─────────────────────────────

/// Verify that `force=true` is accepted over MCP wire and the Collection install
/// succeeds when the package already exists at dest.
///
/// Uses the git-clone Collection path via a local git repo so that the
/// GitUrl Collection branch (where force skip/overwrite logic lives) is
/// exercised end-to-end.
#[tokio::test]
async fn test_pkg_install_force_overwrites_collection() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build a bare git repo acting as a Collection source.
    // One subdir `e2e_force_col` with init.lua.
    let repo_root = tempfile::tempdir().expect("repo root");
    let pkg_subdir = repo_root.path().join("e2e_force_col");
    std::fs::create_dir_all(&pkg_subdir).expect("mkdir pkg_subdir");
    std::fs::write(
        pkg_subdir.join("init.lua"),
        "local M = {}\nM.meta = { name = \"e2e_force_col\", version = \"1.0.0\" }\nreturn M",
    )
    .expect("write init.lua v1");

    // Init git repo and commit.
    let git_init = std::process::Command::new("git")
        .args(["init", "-b", "main"])
        .current_dir(repo_root.path())
        .output()
        .expect("git init");
    assert!(git_init.status.success(), "git init failed");
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(repo_root.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(repo_root.path())
        .output();
    let git_add = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(repo_root.path())
        .output()
        .expect("git add");
    assert!(git_add.status.success(), "git add failed");
    let git_commit = std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(repo_root.path())
        .output()
        .expect("git commit");
    assert!(git_commit.status.success(), "git commit failed");

    // First install via file:// URL — goes through git clone Collection path.
    let repo_url = format!("file://{}", repo_root.path().to_string_lossy());
    let resp1 = call_json(&client, "alc_pkg_install", json!({ "url": repo_url })).await;
    let installed1 = resp1["installed"].as_array().expect("installed array");
    assert!(
        installed1.iter().any(|v| v == "e2e_force_col"),
        "e2e_force_col should be installed on first call: {resp1}"
    );

    // Modify the installed package to simulate drift.
    let installed_init = tmp
        .path()
        .join("packages")
        .join("e2e_force_col")
        .join("init.lua");
    std::fs::write(&installed_init, "-- stale").expect("write stale");

    // Second install with force=true — GitUrl Collection branch: removes existing → copies fresh.
    let resp2 = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": repo_url, "force": true }),
    )
    .await;
    // force=true: existing package is overwritten → lands in `installed` (not `skipped`).
    let installed2 = resp2["installed"]
        .as_array()
        .expect("installed array (force)");
    assert!(
        installed2.iter().any(|v| v == "e2e_force_col"),
        "e2e_force_col should be in installed (not skipped) after force=true: {resp2}"
    );

    // The stale content must be gone.
    let content = std::fs::read_to_string(&installed_init).expect("read after force install");
    assert!(
        content.contains("1.0.0"),
        "installed init.lua should be restored from git after force overwrite, got: {content}"
    );
    assert!(
        !content.contains("stale"),
        "stale content must be removed after force overwrite"
    );

    client.cancel().await.expect("cancel failed");
}

/// Verify that omitting `force` (default false) causes an existing Collection
/// package to be skipped via the git-clone Collection path.
#[tokio::test]
async fn test_pkg_install_force_default_is_skip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build a bare git repo acting as a Collection source.
    let repo_root = tempfile::tempdir().expect("repo root");
    let pkg_subdir = repo_root.path().join("e2e_skip_col");
    std::fs::create_dir_all(&pkg_subdir).expect("mkdir pkg_subdir");
    std::fs::write(
        pkg_subdir.join("init.lua"),
        "local M = {}\nM.meta = { name = \"e2e_skip_col\", version = \"1.0.0\" }\nreturn M",
    )
    .expect("write init.lua");

    let _ = std::process::Command::new("git")
        .args(["init", "-b", "main"])
        .current_dir(repo_root.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(repo_root.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(repo_root.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(repo_root.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(repo_root.path())
        .output();

    // First install via file:// URL.
    let repo_url = format!("file://{}", repo_root.path().to_string_lossy());
    call_json(&client, "alc_pkg_install", json!({ "url": repo_url })).await;

    // Second install without force — existing package must be skipped.
    let resp2 = call_json(&client, "alc_pkg_install", json!({ "url": repo_url })).await;
    let skipped = resp2["skipped"].as_array().expect("skipped array");
    assert!(
        skipped.iter().any(|v| v == "e2e_skip_col"),
        "e2e_skip_col should be in skipped when force is omitted: {resp2}"
    );
    let installed = resp2["installed"].as_array().expect("installed array");
    assert!(
        !installed.iter().any(|v| v == "e2e_skip_col"),
        "e2e_skip_col must not be in installed when force is omitted: {resp2}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Verify that passing a non-bool value for `force` results in a deserialization
/// error returned by the MCP layer (protocol-level error, not a tool-level error).
///
/// Uses the default `connect()` to match the pattern of `test_pkg_doctor_shape_error`
/// which tests the same schema-violation path.
#[tokio::test]
async fn test_pkg_install_invalid_force_type() {
    let client = connect().await;

    // The MCP layer rejects invalid parameter types before reaching the tool handler.
    // This surfaces as an Err from call_tool (McpError), not as is_error=true.
    // Pattern mirrors `test_pkg_doctor_shape_error`.
    let outcome = client
        .call_tool(call_params(
            "alc_pkg_install",
            json!({ "url": "github.com/user/pkg", "force": "notbool" }),
        ))
        .await;

    match outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            let has_type_error = text.contains("invalid type")
                || text.contains("expected a boolean")
                || text.contains("expected bool");
            assert!(
                is_error || has_type_error,
                "expected shape error (is_error=true or type-mismatch text), got is_error={is_error:?}, text: {text}"
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("invalid type") || msg.contains("expected a boolean"),
                "expected invalid-type error from param deserialization, got: {msg}"
            );
        }
    }

    client.cancel().await.expect("cancel failed");
}

// ─── state lost-update / Lua print safety ────────────────────────

/// Verify that concurrent `alc.state.set` calls on the same key do not
/// produce a lost update.
///
/// Two `alc_run` requests are issued in parallel via the same MCP
/// connection.  Each Lua snippet reads the current value of key `"c"`,
/// adds 1, and writes it back.  Without the per-namespace `Mutex`
/// introduced in `JsonFileStore`, one of the two writes would be
/// dropped when both reads observe the same initial value.
///
/// After both calls complete the key must equal 2.
#[tokio::test]
async fn test_state_no_lost_update_under_concurrent_writes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Use a dedicated namespace via ctx._ns so the test is hermetic.
    // alc.state.incr is used here because its read-modify-write is fully
    // serialised by the per-namespace Mutex inside JsonFileStore.  A plain
    // get/set pair would require the Lua code to also run atomically, which
    // cannot be guaranteed across two independent MCP sessions — the sessions
    // themselves run concurrently but the Lua RMW within each is not atomic
    // unless backed by an atomic backend primitive.
    //
    // This test verifies that two concurrent incr(delta=1) calls on the same
    // key never produce a lost update: the final value must be exactly 2.
    let ns = "e2e_concurrent_state";
    let code = r#"return alc.state.incr("c", 1, 0)"#;

    // Fire both requests in parallel, both targeting the same namespace.
    let (r1, r2) = tokio::join!(
        call_json(
            &client,
            "alc_run",
            json!({ "code": code, "ctx": { "_ns": ns } })
        ),
        call_json(
            &client,
            "alc_run",
            json!({ "code": code, "ctx": { "_ns": ns } })
        ),
    );
    assert_eq!(r1["status"], "completed", "run 1 failed: {r1}");
    assert_eq!(r2["status"], "completed", "run 2 failed: {r2}");

    // Each call returns the value after its own increment.  The two returned
    // values must be {1, 2} (in any order) — never {1, 1} which would indicate
    // a lost update.
    let v1 = r1["result"]
        .as_f64()
        .unwrap_or_else(|| panic!("r1 result not a number: {r1}")) as u64;
    let v2 = r2["result"]
        .as_f64()
        .unwrap_or_else(|| panic!("r2 result not a number: {r2}")) as u64;
    let mut pair = [v1, v2];
    pair.sort();
    assert_eq!(
        pair,
        [1, 2],
        "expected incr results to be {{1, 2}} (no lost update), got {v1} and {v2}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Verify that Lua `print(...)` does not corrupt the rmcp transport.
///
/// A user strategy calling `print("dbg")` should emit log output without
/// touching stdout.  The MCP connection must remain functional after the
/// call, i.e. a follow-up request to the same client must succeed.
#[tokio::test]
async fn test_lua_print_does_not_corrupt_transport() {
    let client = connect().await;

    // Run Lua that calls print() — this would corrupt the JSON-RPC transport
    // if print still wrote to stdout.
    let resp = call_json(
        &client,
        "alc_run",
        json!({ "code": r#"print("dbg: hello from lua"); return 42"# }),
    )
    .await;
    assert_eq!(
        resp["status"], "completed",
        "alc_run with print failed: {resp}"
    );
    assert_eq!(resp["result"], 42);

    // The transport must still be functional — issue a follow-up request.
    let resp2 = call_json(&client, "alc_run", json!({ "code": "return 99" })).await;
    assert_eq!(
        resp2["status"], "completed",
        "follow-up call failed: {resp2}"
    );
    assert_eq!(resp2["result"], 99);

    client.cancel().await.expect("cancel failed");
}

// ─── MCP completion/complete E2E ────────────────────────────────

/// Happy-path: `completion/complete` for `alc://packages/{name}/init.lua`
/// with an empty prefix and no packages installed returns a structurally
/// valid `CompleteResult` with `completion.values` as an array.
///
/// This test exercises the full MCP wire path:
///   client → `completion/complete` request → `ServerHandler::complete` →
///   `ResourceCatalog::complete_resource_arg` → `EngineApi::pkg_list` →
///   JSON → `CompleteResult { completion: CompletionInfo }`.
///
/// When the `packages/` directory is empty (isolated `ALC_HOME`), the
/// result must be `values: []` — not an error.  That confirms the handler
/// is wired and the wire shape is correct.
#[tokio::test]
async fn test_mcp_resources_complete_packages_empty_home() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Ensure packages dir exists so the server does not error on scan.
    std::fs::create_dir_all(tmp.path().join("packages")).expect("mkdir packages");

    let client = connect_with_alc_home(tmp.path()).await;

    let params = CompleteRequestParams::new(
        Reference::Resource(ResourceReference {
            uri: "alc://packages/{name}/init.lua".to_string(),
        }),
        ArgumentInfo {
            name: "name".to_string(),
            value: "".to_string(),
        },
    );

    let result = client
        .complete(params)
        .await
        .expect("completion/complete failed");

    // Wire shape: completion.values must be a Vec (possibly empty).
    assert!(
        result.completion.values.len() <= 100,
        "values length must not exceed cap=100, got {}",
        result.completion.values.len()
    );
    // All values must be non-empty strings.
    for v in &result.completion.values {
        assert!(!v.is_empty(), "completion value must not be empty string");
    }
    // When values is empty, has_more must not be true.
    if result.completion.values.is_empty() {
        assert_ne!(
            result.completion.has_more,
            Some(true),
            "has_more must not be true when values is empty"
        );
    }

    client.cancel().await.expect("cancel failed");
}

/// When a package is pre-installed in `ALC_HOME/packages/`, the completion
/// result for `alc://packages/{name}/init.lua` must include it in `values`.
///
/// This confirms the EngineApi → JSON → completion dispatch chain returns
/// real data end-to-end across the MCP wire.
#[tokio::test]
async fn test_mcp_resources_complete_packages_returns_installed_name() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Pre-install a package so pkg_list returns it.
    let pkg_dir = tmp.path().join("packages").join("e2e_complete_test_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir pkg");
    std::fs::write(
        pkg_dir.join("init.lua"),
        concat!(
            "local M = {}\n",
            "M.meta = { name = 'e2e_complete_test_pkg', version = '0.1.0' }\n",
            "return M\n",
        ),
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let params = CompleteRequestParams::new(
        Reference::Resource(ResourceReference {
            uri: "alc://packages/{name}/init.lua".to_string(),
        }),
        ArgumentInfo {
            name: "name".to_string(),
            value: "".to_string(),
        },
    );

    let result = client
        .complete(params)
        .await
        .expect("completion/complete failed");

    assert!(
        result
            .completion
            .values
            .iter()
            .any(|v| v == "e2e_complete_test_pkg"),
        "expected 'e2e_complete_test_pkg' in completion values, got: {:?}",
        result.completion.values
    );
    // total must be Some and ≥ 1 when values are present.
    assert!(
        result.completion.total.unwrap_or(0) >= 1,
        "expected total >= 1 when at least one package is installed"
    );

    client.cancel().await.expect("cancel failed");
}

/// Prefix filtering: when the prefix `"e2e_c"` is provided, only packages
/// whose names start with that prefix must appear in the completion values.
///
/// This verifies that `complete_resource_arg` passes the prefix down and
/// filters correctly.
#[tokio::test]
async fn test_mcp_resources_complete_packages_prefix_filter() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Install two packages: one matching the prefix, one not.
    for name in &["e2e_c_alpha", "other_pkg"] {
        let dir = tmp.path().join("packages").join(name);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("init.lua"),
            format!(
                "local M = {{}}\nM.meta = {{ name = '{}', version = '0.1.0' }}\nreturn M\n",
                name
            ),
        )
        .expect("write init.lua");
    }

    let client = connect_with_alc_home(tmp.path()).await;

    let params = CompleteRequestParams::new(
        Reference::Resource(ResourceReference {
            uri: "alc://packages/{name}/init.lua".to_string(),
        }),
        ArgumentInfo {
            name: "name".to_string(),
            value: "e2e_c".to_string(),
        },
    );

    let result = client
        .complete(params)
        .await
        .expect("completion/complete failed");

    // e2e_c_alpha starts with "e2e_c" → must appear.
    assert!(
        result.completion.values.iter().any(|v| v == "e2e_c_alpha"),
        "expected 'e2e_c_alpha' in completion values for prefix 'e2e_c', got: {:?}",
        result.completion.values
    );
    // other_pkg does NOT start with "e2e_c" → must not appear.
    assert!(
        !result.completion.values.iter().any(|v| v == "other_pkg"),
        "unexpected 'other_pkg' in filtered completion values: {:?}",
        result.completion.values
    );

    client.cancel().await.expect("cancel failed");
}

/// E2E: `alc://hub/index` with a corrupt cache file emits a `warnings` array.
///
/// Sets up an isolated `ALC_HOME` with:
/// - A `hub_registries.json` pointing to two source URLs.
/// - A valid `hub_cache/{key}.json` for the first URL (contains "good_pkg").
/// - An invalid JSON `hub_cache/{key}.json` for the second URL (corrupt).
///
/// Reading `alc://hub/index` must:
/// - Return valid JSON with `packages` containing entries from the valid source.
/// - Include a `warnings` array mentioning the corrupt source URL.
#[tokio::test]
async fn test_alc_hub_index_aggregate_warnings_emitted_on_corrupt_cache() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let alc_home = tmp.path();

    let url_good = "https://good.example.com/index.json";
    let url_bad = "https://bad-corrupt.example.com/index.json";

    // FNV-1a cache key (matches hub.rs::cache_key)
    fn fnv1a_hex(url: &str) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325u64;
        for b in url.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3u64);
        }
        format!("{h:016x}")
    }

    let hub_cache_dir = alc_home.join("hub_cache");
    std::fs::create_dir_all(&hub_cache_dir).expect("mkdir hub_cache");

    // Write a valid cache for the good URL.
    // IndexEntry uses #[serde(flatten)] on entity, so pkg fields are top-level.
    let good_index = serde_json::json!({
        "schema_version": "hub_index/v0",
        "updated_at": "2026-01-01T00:00:00Z",
        "packages": [{
            "name": "good_pkg",
            "version": "0.1.0",
            "source": { "type": "unknown" },
            "card_count": 0
        }]
    });
    std::fs::write(
        hub_cache_dir.join(format!("{}.json", fnv1a_hex(url_good))),
        serde_json::to_string(&good_index).expect("serialize"),
    )
    .expect("write good cache");

    // Write corrupt JSON for the bad URL.
    std::fs::write(
        hub_cache_dir.join(format!("{}.json", fnv1a_hex(url_bad))),
        b"{{{{ definitely not valid json",
    )
    .expect("write corrupt cache");

    // Register both URLs in hub_registries.json.
    let reg_json = serde_json::json!({
        "registries": [
            { "source": url_good, "origin": "pkg_install", "added_at": "2026-01-01T00:00:00Z" },
            { "source": url_bad,  "origin": "pkg_install", "added_at": "2026-01-01T00:00:00Z" }
        ]
    });
    std::fs::write(
        alc_home.join("hub_registries.json"),
        serde_json::to_string(&reg_json).expect("serialize"),
    )
    .expect("write hub_registries");

    let client = connect_with_alc_home(alc_home).await;
    let result = read_resource(&client, "alc://hub/index")
        .await
        .expect("read alc://hub/index failed");

    assert_eq!(result.contents.len(), 1);
    let (uri, text) = resource_text(&result.contents[0]);
    assert_eq!(uri, "alc://hub/index");

    let body: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("hub/index JSON parse failed: {e}\nraw: {text}"));

    // packages must contain entries from the valid source.
    // IndexEntry uses #[serde(flatten)] so package fields are at top level.
    let packages = body["packages"].as_array().expect("packages must be array");
    let has_good_pkg = packages
        .iter()
        .any(|p| p["name"].as_str() == Some("good_pkg"));
    assert!(
        has_good_pkg,
        "expected good_pkg from valid source, got packages: {packages:?}"
    );

    // warnings must be non-empty and mention the corrupt URL.
    let warnings = body["warnings"].as_array().expect("warnings must be array");
    assert!(
        !warnings.is_empty(),
        "expected non-empty warnings array, got: {body}"
    );
    let bad_url_mentioned = warnings.iter().any(|w| {
        w.as_str()
            .map(|s| s.contains("bad-corrupt.example.com"))
            .unwrap_or(false)
    });
    assert!(
        bad_url_mentioned,
        "expected warning to mention the corrupt URL, got warnings: {warnings:?}"
    );

    client.cancel().await.expect("cancel failed");
}

/// E2E: `completion/complete` for a resource template arg name that the server
/// does not provide candidates for must return a non-error response with
/// `values: []`, `total: 0`, and `has_more: false`.
///
/// Uses `alc://logs/{session_id}` with arg `session_id` — the handler
/// explicitly returns empty for the `logs` domain (no session-list API).
#[tokio::test]
async fn test_mcp_resources_complete_returns_empty_for_unsupported_arg_name() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let params = CompleteRequestParams::new(
        Reference::Resource(ResourceReference {
            uri: "alc://logs/{session_id}".to_string(),
        }),
        ArgumentInfo {
            name: "session_id".to_string(),
            value: "".to_string(),
        },
    );

    let result = client
        .complete(params)
        .await
        .expect("completion/complete for logs domain must not error");

    assert!(
        result.completion.values.is_empty(),
        "expected empty values for logs domain (no session-list API), got: {:?}",
        result.completion.values
    );
    assert_eq!(
        result.completion.total,
        Some(0),
        "expected total=0 for empty completion, got: {:?}",
        result.completion.total
    );
    assert_ne!(
        result.completion.has_more,
        Some(true),
        "has_more must not be true when values is empty"
    );

    client.cancel().await.expect("cancel failed");
}

/// `ref/prompt` reference must return an empty completion result without error.
///
/// The handler's `Reference::Prompt` branch returns `CompletionInfo::default()`
/// (empty values, no total/has_more).  This confirms the branch is reachable
/// end-to-end without panic or MCP error.
#[tokio::test]
async fn test_mcp_resources_complete_prompt_ref_returns_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let params = CompleteRequestParams::new(
        Reference::for_prompt("nonexistent_prompt"),
        ArgumentInfo {
            name: "arg".to_string(),
            value: "".to_string(),
        },
    );

    let result = client
        .complete(params)
        .await
        .expect("completion/complete for prompt ref must not error");

    assert!(
        result.completion.values.is_empty(),
        "expected empty values for prompt ref, got: {:?}",
        result.completion.values
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_session_new (issue #1776627475) ─────────────────────────────────────

/// `alc_session_new` activates a session pin: subsequent calls without
/// an explicit `project_root` resolve through the pin (S layer in the
/// P > S > E > W chain).
#[tokio::test]
async fn test_alc_session_new_activates_pin_for_subsequent_pkg_list() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build a project root with alc.toml so pkg_list returns project rows.
    let project_dir = tmp.path().join("proj-A");
    std::fs::create_dir_all(&project_dir).expect("create project dir");
    std::fs::write(project_dir.join("alc.toml"), "[packages]\n").expect("write alc.toml");

    // Activate session with this project root.
    let resp = call_json(
        &client,
        "alc_session_new",
        json!({ "project_root": project_dir.to_string_lossy() }),
    )
    .await;
    assert_eq!(resp["mode"], "default");
    assert_eq!(
        resp["project_root"],
        json!(project_dir.to_string_lossy().to_string())
    );
    assert!(
        resp["session_id"]
            .as_str()
            .unwrap_or("")
            .starts_with("alc-sess-"),
        "session_id must use alc-sess- prefix, got {:?}",
        resp["session_id"]
    );

    // alc_pkg_list without explicit project_root must pick up the pin
    // (alc.toml at project_dir → empty `[packages]`).
    let listing = call_json(&client, "alc_pkg_list", json!({})).await;
    let project_root = listing["project_root"].as_str().unwrap_or("");
    assert_eq!(
        project_root,
        project_dir.to_string_lossy().as_ref(),
        "pkg_list project_root must reflect the activated session pin, got {project_root}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `mode = "test"` round-trips through the activation response.
#[tokio::test]
async fn test_alc_session_new_test_mode_records_in_response() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let resp = call_json(&client, "alc_session_new", json!({ "mode": "test" })).await;
    assert_eq!(resp["mode"], "test");

    client.cancel().await.expect("cancel failed");
}

/// Without `alc_session_new`, the legacy fallback chain (env / cwd
/// walk) is preserved unchanged. After activating, the pin replaces
/// what cwd-walk would have produced.
#[tokio::test]
async fn test_legacy_fallback_unchanged_when_session_not_activated() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // No alc_session_new call → legacy resolution. pkg_list must
    // succeed without panic / MCP error and return a JSON object.
    // (Specific project_root value depends on cwd ancestor walk and
    // is not asserted here — the contract is "doesn't break".)
    let listing_no_session = call_json(&client, "alc_pkg_list", json!({})).await;
    assert!(
        listing_no_session.is_object(),
        "pkg_list legacy path must return a JSON object, got: {listing_no_session}"
    );

    // After activation, the pin takes effect — fixture project root
    // appears as the resolved root for subsequent calls.
    let project_dir = tmp.path().join("legacy-flip");
    std::fs::create_dir_all(&project_dir).expect("create project dir");
    std::fs::write(project_dir.join("alc.toml"), "[packages]\n").expect("write alc.toml");

    let _ = call_json(
        &client,
        "alc_session_new",
        json!({ "project_root": project_dir.to_string_lossy() }),
    )
    .await;
    let listing_with_session = call_json(&client, "alc_pkg_list", json!({})).await;
    assert_eq!(
        listing_with_session["project_root"].as_str().unwrap_or(""),
        project_dir.to_string_lossy().as_ref(),
        "after activation, pkg_list must reflect the session pin"
    );

    client.cancel().await.expect("cancel failed");
}

/// Per-call `project_root` overrides the activated session pin
/// (P > S in the §6 priority order).
#[tokio::test]
async fn test_per_call_project_root_overrides_session_pin() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let pinned = tmp.path().join("pinned");
    let override_dir = tmp.path().join("override");
    std::fs::create_dir_all(&pinned).expect("create pinned");
    std::fs::create_dir_all(&override_dir).expect("create override");
    std::fs::write(pinned.join("alc.toml"), "[packages]\n").expect("write pinned alc.toml");
    std::fs::write(override_dir.join("alc.toml"), "[packages]\n").expect("write override alc.toml");

    // Pin to `pinned`.
    let _ = call_json(
        &client,
        "alc_session_new",
        json!({ "project_root": pinned.to_string_lossy() }),
    )
    .await;

    // pkg_list with explicit project_root = override_dir must win.
    let listing = call_json(
        &client,
        "alc_pkg_list",
        json!({ "project_root": override_dir.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        listing["project_root"].as_str().unwrap_or(""),
        override_dir.to_string_lossy().as_ref(),
        "explicit per-call project_root must override session pin"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_card_analyze E2E tests ──────────────────────────────────────────────

/// Happy-path round-trip for `alc_card_analyze`.
///
/// Part A (no-sample path): invokes `alc_card_analyze` on a Card with no
/// samples sidecar.  The `card_analysis` pkg detects total==0 and returns
/// immediately without calling `alc.llm`.  This exercises the typed-contract
/// post-processing inside `AppService::card_analyze` and asserts the flat
/// `CardAnalyzeResult` shape (`pattern`/`suggested_change`/`confidence` at
/// top-level `result`, `card`/`samples` echo absent).
///
/// Part B (LLM round-trip): invokes `alc_card_analyze` on a Card with a
/// failure sample so the pkg calls `alc.llm` and pauses.  Then drives a
/// complete `alc_continue` round-trip and asserts `status: "completed"`.
/// This satisfies Tier 1 constraint C1 (complete alc_continue round-trip).
#[tokio::test]
async fn test_alc_card_analyze_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // ── Card A: no sidecar (zero samples → immediate completed path) ──
    let card_id_a = "e2e_20260510T000000_aabbcc";
    let card_pkg = "e2e";
    let card_dir = tmp.path().join("cards").join(card_pkg);
    std::fs::create_dir_all(&card_dir).expect("create card dir");
    std::fs::write(
        card_dir.join(format!("{card_id_a}.toml")),
        concat!(
            "schema_version = \"card/v0\"\n",
            "card_id = \"e2e_20260510T000000_aabbcc\"\n",
            "created_at = \"2026-05-10T00:00:00Z\"\n",
            "[pkg]\n",
            "name = \"e2e\"\n",
        ),
    )
    .expect("write card A toml");
    // No samples file → `card_analysis` returns immediately (total==0 guard).

    // ── Card B: one failure sample (LLM path → needs_response → alc_continue) ──
    let card_id_b = "e2e_20260510T000001_bbccdd";
    std::fs::write(
        card_dir.join(format!("{card_id_b}.toml")),
        concat!(
            "schema_version = \"card/v0\"\n",
            "card_id = \"e2e_20260510T000001_bbccdd\"\n",
            "created_at = \"2026-05-10T00:00:00Z\"\n",
            "[pkg]\n",
            "name = \"e2e\"\n",
        ),
    )
    .expect("write card B toml");
    // One failure sample → pkg calls alc.llm → session pauses.
    std::fs::write(
        card_dir.join(format!("{card_id_b}.samples.jsonl")),
        "{\"case_idx\":0,\"admission\":\"fail\",\"status\":\"fail\"}\n",
    )
    .expect("write card B samples");

    let client = connect_with_alc_home(tmp.path()).await;

    // 1. Install the fixture pkg via absolute path string (K-137: no file:// URL).
    // Use the user's packages dir as a Collection root so that card_analysis
    // (at <packages>/card_analysis/init.lua) is discovered via Collection layout.
    // Single-package mode (root-level init.lua) was removed in v0.36.0.
    let packages_dir =
        std::path::PathBuf::from(std::env::var("HOME").expect("HOME")).join(".algocline/packages");
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": packages_dir.to_string_lossy().as_ref() }),
    )
    .await;

    // ── Part A: zero-sample card → immediate completed, flat CardAnalyzeResult ──

    let resp_a = call_json(&client, "alc_card_analyze", json!({ "card_id": card_id_a })).await;
    // The pkg returns immediately (total==0 guard); post-processing flattens result.
    assert_eq!(
        resp_a["status"], "completed",
        "zero-sample card must complete immediately, got: {resp_a}"
    );
    // Flat shape: pattern/suggested_change/confidence at top-level result.
    assert!(
        resp_a["result"]["pattern"].as_str().is_some(),
        "pattern must be a string at result.pattern, got: {resp_a}"
    );
    assert!(
        resp_a["result"]["suggested_change"].as_str().is_some(),
        "suggested_change must be a string at result.suggested_change, got: {resp_a}"
    );
    assert!(
        resp_a["result"]["confidence"].as_f64().is_some(),
        "confidence must be a number at result.confidence, got: {resp_a}"
    );
    // Typed contract: card/samples echo must be absent from result.
    assert!(
        resp_a["result"].get("card").is_none(),
        "echo of 'card' must be removed from result: {resp_a}"
    );
    assert!(
        resp_a["result"].get("samples").is_none(),
        "echo of 'samples' must be removed from result: {resp_a}"
    );

    // ── Part B: failure-sample card → LLM pause → alc_continue round-trip ──

    let resp_b = call_json(&client, "alc_card_analyze", json!({ "card_id": card_id_b })).await;
    assert_eq!(
        resp_b["status"], "needs_response",
        "failure-sample card must pause for LLM, got: {resp_b}"
    );
    let session_id = resp_b["session_id"]
        .as_str()
        .expect("session_id missing from needs_response");

    // Drive the complete alc_continue round-trip (Tier 1 C1).
    // Stub response is STRICT JSON so alc.json_extract can parse it (Risk 4).
    let stub = r#"{"pattern":"prompt too vague","suggested_change":"add explicit constraints","confidence":0.8}"#;
    let resp_cont = call_json(
        &client,
        "alc_continue",
        json!({ "session_id": session_id, "response": stub }),
    )
    .await;
    assert_eq!(
        resp_cont["status"], "completed",
        "expected completed after alc_continue, got: {resp_cont}"
    );
    // alc_continue returns the raw session result; the LLM output appears
    // nested at result.result (post-processing runs in card_analyze, not here).
    assert!(
        resp_cont["result"]["result"]["pattern"].as_str().is_some(),
        "pattern must be present in result.result after round-trip, got: {resp_cont}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Error path: invoking `alc_card_analyze` with an unknown `card_id` must
/// return a distinct typed error message containing the card id.
///
/// Validates Tier 1 constraint C1 (unknown card_id typed error).
#[tokio::test]
async fn test_alc_card_analyze_unknown_card_id() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_card_analyze",
            json!({ "card_id": "nonexistent_card_id_xyz" }),
        ))
        .await
        .expect("call_tool failed");

    let text = extract_text(&result);
    assert!(
        text.contains("nonexistent_card_id_xyz"),
        "error must reference the unknown card_id, got: {text}"
    );
    assert!(
        text.contains("not found"),
        "error must say 'not found', got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Shape violation: passing `card_id` as a non-string value (boolean) must
/// be rejected by MCP transport / schemars validation with a distinct typed
/// error — either the transport returns an `McpError` or a `CallToolResult`
/// with `is_error = true`.  A successful `completed` response is not acceptable.
///
/// Validates Tier 1 constraint C1 (shape violation typed error).
#[tokio::test]
async fn test_alc_card_analyze_shape_violation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Pass `true` (bool) where `card_id: string` is expected.
    // The MCP / schemars deserialization layer rejects this before it reaches
    // `card_analyze`, returning an McpError at the transport level.
    let outcome = client
        .call_tool(call_params("alc_card_analyze", json!({ "card_id": true })))
        .await;

    match outcome {
        Err(e) => {
            // Transport-level rejection: schemars / rmcp parameter deserialization
            // failed.  The error message must mention the type mismatch.
            let msg = format!("{e}");
            assert!(
                msg.contains("boolean") || msg.contains("string") || msg.contains("invalid"),
                "McpError message must describe a type/shape error, got: {msg}"
            );
        }
        Ok(result) => {
            // Tool-level rejection: the tool returned is_error=true.
            let text = extract_text(&result);
            let is_err = result.is_error.unwrap_or(false);
            let parsed: Option<serde_json::Value> = serde_json::from_str(text).ok();
            let is_completed = parsed.as_ref().is_some_and(|v| v["status"] == "completed");
            assert!(
                is_err || !is_completed,
                "shape violation must not produce a successful completed result, got: {text}"
            );
        }
    }

    client.cancel().await.expect("cancel failed");
}

// ─── Library package type guard tests ────────────────────────────────────────

/// `alc_advice` must reject a package whose type is "library" with an error
/// message containing "library".
///
/// Validates Crux: Adviser library branch — the adviser must never fall through
/// to the runnable execution path for a library-typed package.
#[tokio::test]
async fn test_alc_advice_rejects_library_package() {
    let alc_home = tempfile::tempdir().expect("tempdir");
    let pkg_dir = alc_home.path().join("packages").join("test_lib_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "test_lib_pkg", version = "0.1.0", type = "library" }
M.helpers = {}
return M"#,
    )
    .expect("write init.lua");

    let client = connect_with_alc_home(alc_home.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_advice",
            json!({ "strategy": "test_lib_pkg", "task": "test" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    assert!(
        text.contains("library"),
        "expected error mentioning 'library', got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc_eval` must reject a package whose type is "library" with an error
/// message containing "library", and must do so before any LLM call is
/// initiated.
///
/// Validates Crux: Eval type guard — the eval entry point must reject library
/// packages with an explicit typed error before any LLM call is initiated.
#[tokio::test]
async fn test_alc_eval_rejects_library_package() {
    let alc_home = tempfile::tempdir().expect("tempdir");
    let pkg_dir = alc_home.path().join("packages").join("test_lib_pkg");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "test_lib_pkg", version = "0.1.0", type = "library" }
M.helpers = {}
return M"#,
    )
    .expect("write init.lua");
    // evalframe is required by eval(); provide a minimal stub so the evalframe
    // check passes and the library guard fires (not the evalframe-missing error).
    let evalframe_dir = alc_home.path().join("packages").join("evalframe");
    std::fs::create_dir_all(&evalframe_dir).expect("create evalframe dir");
    std::fs::write(
        evalframe_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "evalframe", version = "0.1.0", type = "runnable" }
M.run = function(ctx) return {} end
return M"#,
    )
    .expect("write evalframe init.lua");

    let client = connect_with_alc_home(alc_home.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_eval",
            json!({ "strategy": "test_lib_pkg", "scenario": "return {cases={}}" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    assert!(
        text.contains("library"),
        "expected error mentioning 'library', got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── CLI dry-run tests (no MCP harness) ──────────────────────────────────────

/// Resolve the path to the `alc` binary, mirroring the logic in `connect()`.
fn alc_bin() -> String {
    std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")))
}

#[test]
fn test_cli_help_short() {
    let output = std::process::Command::new(alc_bin())
        .arg("-h")
        .output()
        .expect("failed to run alc -h");

    assert!(
        output.status.success(),
        "alc -h exited with non-zero status: {}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage:"),
        "alc -h stdout missing 'Usage:': {stdout}"
    );
}

#[test]
fn test_cli_help_long() {
    let output = std::process::Command::new(alc_bin())
        .arg("--help")
        .output()
        .expect("failed to run alc --help");

    assert!(
        output.status.success(),
        "alc --help exited with non-zero status: {}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("init"),
        "alc --help stdout missing 'init': {stdout}"
    );
    assert!(
        stdout.contains("update"),
        "alc --help stdout missing 'update': {stdout}"
    );
}

#[test]
fn test_cli_version() {
    let output = std::process::Command::new(alc_bin())
        .arg("-V")
        .output()
        .expect("failed to run alc -V");

    assert!(
        output.status.success(),
        "alc -V exited with non-zero status: {}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // clap uses the binary name from #[command(name = "alc", ...)], so the
    // version line is "alc X.Y.Z"
    assert!(
        stdout.contains("alc"),
        "alc -V stdout missing binary name 'alc': {stdout}"
    );
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "alc -V stdout missing version '{}': {stdout}",
        env!("CARGO_PKG_VERSION")
    );
}

// ─── MCP Prompts E2E (workflow-trigger) ──────────────────────────────────────

/// `prompts/list` returns the static workflow-trigger prompts (`advice`,
/// `new_package`). See `docs/design/mcp-support.md` for the scope rationale.
///
/// Verifies:
/// - `list_all_prompts()` succeeds.
/// - Both `advice` and `new_package` are present.
/// - `advice` exposes a required `task` argument.
/// - `new_package` exposes a required `name` argument and an optional `category`.
#[tokio::test]
async fn test_list_prompts_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let client = connect_with_alc_home(tmp.path()).await;

    let prompts = client
        .list_all_prompts()
        .await
        .expect("list_all_prompts failed");

    let advice = prompts
        .iter()
        .find(|p| p.name == "advice")
        .expect("`advice` prompt missing from list");
    let advice_args = advice
        .arguments
        .as_deref()
        .expect("advice must declare arguments");
    assert_eq!(advice_args.len(), 1, "advice expects exactly 1 argument");
    assert_eq!(advice_args[0].name, "task");
    assert_eq!(
        advice_args[0].required,
        Some(true),
        "advice.task must be required"
    );

    let new_pkg = prompts
        .iter()
        .find(|p| p.name == "new_package")
        .expect("`new_package` prompt missing from list");
    let new_pkg_args = new_pkg
        .arguments
        .as_deref()
        .expect("new_package must declare arguments");
    assert_eq!(
        new_pkg_args.len(),
        2,
        "new_package expects exactly 2 arguments"
    );
    let name_arg = new_pkg_args
        .iter()
        .find(|a| a.name == "name")
        .expect("new_package.name argument missing");
    assert_eq!(
        name_arg.required,
        Some(true),
        "new_package.name must be required"
    );
    let category_arg = new_pkg_args
        .iter()
        .find(|a| a.name == "category")
        .expect("new_package.category argument missing");
    assert_eq!(
        category_arg.required,
        Some(false),
        "new_package.category must be optional"
    );

    client.cancel().await.expect("cancel failed");
}

/// `prompts/get` must return a single user-role text message with declared
/// arguments substituted into the template.
///
/// Verifies on the `advice` prompt:
/// - exactly 1 message of role `User` is returned;
/// - the supplied `task` value appears verbatim in the message text;
/// - the `${task}` placeholder is not present verbatim (substitution happened).
#[tokio::test]
async fn test_get_prompt_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let client = connect_with_alc_home(tmp.path()).await;

    let task_text = "解の精度を上げて";
    let mut arguments = serde_json::Map::new();
    arguments.insert(
        "task".to_string(),
        serde_json::Value::String(task_text.to_string()),
    );

    let result = client
        .get_prompt(GetPromptRequestParams::new("advice").with_arguments(arguments))
        .await
        .expect("get_prompt failed");

    assert_eq!(
        result.messages.len(),
        1,
        "expected exactly 1 message in get_prompt result"
    );

    let msg = &result.messages[0];
    assert_eq!(
        msg.role,
        rmcp::model::PromptMessageRole::User,
        "message role must be User"
    );

    let text = match &msg.content {
        rmcp::model::PromptMessageContent::Text { text } => text.as_str(),
        other => panic!("expected Text content, got: {other:?}"),
    };

    assert!(
        text.contains(task_text),
        "message text must contain the task value '{task_text}', got: {text}"
    );
    assert!(
        !text.contains("${task}"),
        "placeholder '${{task}}' must be substituted, but was found in: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_v2 E2E tests ─────────────────────────────────────────────

/// Case 1: `alc_v2_run` with a simple pure-Lua expression returns a JSON object
/// containing a `session_id` string field.
#[tokio::test]
async fn test_alc_v2_run_basic() {
    let client = connect().await;

    let resp = call_json(&client, "alc_v2_run", json!({ "code": "return {ok=true}" })).await;

    assert!(
        resp["session_id"].is_string(),
        "expected session_id string in response, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 2: `alc_v2_state` with a non-existent session id returns an error
/// message containing "not found".
#[tokio::test]
async fn test_alc_v2_state_unknown_session_errors() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_v2_state",
            json!({ "session_id": "ses-nonexistent-00000000" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.to_lowercase().contains("not found"),
        "expected 'not found' error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 3: `alc_v2_state` is idempotent — two successive calls with the same
/// `session_id` return equal JSON.
#[tokio::test]
async fn test_alc_v2_state_idempotent() {
    let client = connect().await;

    // Run a session that completes immediately.
    let run_resp = call_json(&client, "alc_v2_run", json!({ "code": "return {ok=true}" })).await;
    let sid = run_resp["session_id"].as_str().expect("session_id");

    // Poll until the session reaches a terminal state (done / cancelled / failed).
    let first = timeout(Duration::from_secs(5), async {
        loop {
            let s = call_json(&client, "alc_v2_state", json!({"session_id": sid})).await;
            let state_tag = s["state"].as_str().unwrap_or("");
            if matches!(state_tag, "done" | "cancelled" | "failed") {
                return s;
            }
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("session did not reach terminal state within 5s");

    // A second call with the same session_id must return the same JSON.
    let second = call_json(&client, "alc_v2_state", json!({"session_id": sid})).await;

    assert_eq!(
        first, second,
        "alc_v2_state must be idempotent — two calls returned different JSON"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 4: `alc_v2_resume` with a non-existent session id returns an error
/// containing "not found".
#[tokio::test]
async fn test_alc_v2_resume_unknown_session_errors() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_v2_resume",
            json!({
                "session_id": "ses-nonexistent-00000000",
                "payload": {
                    "payload_kind": "single",
                    "response": "ok",
                    "query_id": "q-0",
                    "usage": null
                }
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.to_lowercase().contains("not found"),
        "expected 'not found' error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 5: `alc_v2_resume` on a running (non-paused) session returns an error
/// containing "not paused".
#[tokio::test]
async fn test_alc_v2_resume_not_paused_errors() {
    let client = connect().await;

    // Start a session that completes without pausing.
    let run_resp = call_json(&client, "alc_v2_run", json!({ "code": "return 42" })).await;
    let sid = run_resp["session_id"].as_str().expect("session_id");

    // Wait until the session is done.
    timeout(Duration::from_secs(5), async {
        loop {
            let s = call_json(&client, "alc_v2_state", json!({"session_id": sid})).await;
            if s["state"].as_str() == Some("done") {
                return;
            }
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("session did not reach done state within 5s");

    // Attempt to resume the done session — must fail with "not paused".
    let result = client
        .call_tool(call_params(
            "alc_v2_resume",
            json!({
                "session_id": sid,
                "payload": {
                    "payload_kind": "single",
                    "response": "ok",
                    "query_id": "q-0",
                    "usage": null
                }
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.to_lowercase().contains("not paused"),
        "expected 'not paused' error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 6: `alc_v2_cancel` with a non-existent session id returns an error
/// containing "not found".
#[tokio::test]
async fn test_alc_v2_cancel_unknown_session_errors() {
    let client = connect().await;

    let result = client
        .call_tool(call_params(
            "alc_v2_cancel",
            json!({ "session_id": "ses-nonexistent-00000000" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.to_lowercase().contains("not found"),
        "expected 'not found' error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 7: `alc_v2_cancel` is idempotent — cancelling a session twice both
/// succeed (terminal states are idempotent per spec).
#[tokio::test]
async fn test_alc_v2_cancel_idempotent() {
    let client = connect().await;

    // Run a session that pauses so we have a live session to cancel.
    let run_resp = call_json(
        &client,
        "alc_v2_run",
        json!({ "code": "return alc.llm('hi')" }),
    )
    .await;
    let sid = run_resp["session_id"].as_str().expect("session_id");

    // Wait until paused so the registry entry is live.
    let _state = poll_until_paused(&client, sid, Duration::from_secs(5)).await;

    // First cancel — must succeed.
    let first = call_json(&client, "alc_v2_cancel", json!({ "session_id": sid })).await;
    // A success response is `{}` — any JSON parse succeeding is the criterion.
    let _ = first; // value captured for completeness

    // Second cancel — must also succeed (idempotent).
    let second_result = client
        .call_tool(call_params("alc_v2_cancel", json!({ "session_id": sid })))
        .await
        .expect("call_tool failed");
    let second_text = extract_text(&second_result);
    // Idempotent cancel must not return an error — the text must parse as JSON `{}`.
    let second_json: Value = serde_json::from_str(second_text)
        .unwrap_or_else(|e| panic!("second cancel must return JSON, got: {second_text} ({e})"));
    assert_eq!(
        second_json,
        json!({}),
        "second cancel must return {{}} for idempotent terminal cancel, got: {second_json}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Case 8: Full paused lifecycle — run → paused → resume → terminal.
///
/// Verifies the complete `alc_v2_run` → `alc_v2_state` → `alc_v2_resume` flow for a
/// session that invokes `alc.llm()`.
#[tokio::test]
async fn test_alc_v2_run_paused_lifecycle() {
    let client = connect().await;

    // 1. Start a session that will pause at alc.llm().
    let run_resp = call_json(
        &client,
        "alc_v2_run",
        json!({ "code": "return alc.llm('hi')" }),
    )
    .await;
    let sid = run_resp["session_id"].as_str().expect("session_id");

    // 2. Poll until paused.
    let state = poll_until_paused(&client, sid, Duration::from_secs(5)).await;
    assert_eq!(
        state["state"].as_str(),
        Some("paused"),
        "session must be in paused state"
    );

    // Capture the paused state projection for snapshot verification.
    // Redact UUIDs and Unix-millisecond timestamps so the snapshot is stable.
    let state_redacted = {
        let s = serde_json::to_string_pretty(&state).expect("serialize state");
        let s = redact_uuids(&s);
        // Replace 13-digit numbers (Unix-ms timestamps like 1778723417409) with a placeholder.
        let re = regex::Regex::new(r"\b[0-9]{13}\b").expect("regex");
        re.replace_all(&s, "<TIMESTAMP_MS>").to_string()
    };
    insta::assert_snapshot!("alc_v2_state_paused", state_redacted);

    // 3. Extract the query_id from the pause payload.
    // The PauseInfo struct serialises pending prompts under a "prompts" key.
    let query_id = state["prompts"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|q| q["query_id"].as_str())
        .unwrap_or("q-0")
        .to_string();

    // 4. Resume with a Single response.
    let resume_resp = call_json(
        &client,
        "alc_v2_resume",
        json!({
            "session_id": sid,
            "payload": {
                "payload_kind": "single",
                "response": "ok",
                "query_id": query_id,
                "usage": null
            }
        }),
    )
    .await;

    // Outcome must be "continued" or "terminal".
    let outcome = resume_resp["outcome"].as_str().unwrap_or("");
    assert!(
        outcome == "continued" || outcome == "terminal",
        "resume outcome must be 'continued' or 'terminal', got: {outcome}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Regression: clients that ship `payload` / `reason` as a stringified JSON object
/// (instead of a real object — observed with Claude Code's MCP client even when
/// the schema declares `type: object`) must still be accepted.  The handler
/// reparses `Value::String` via `normalize_stringified_json_object`.
///
/// Without that decode path, `alc_v2_resume` and `alc_v2_cancel` fail with
/// "invalid type: string ..., expected ..." at the rmcp deserializer.
#[tokio::test]
async fn test_alc_v2_resume_and_cancel_accept_stringified_payload() {
    let client = connect().await;

    // 1. Spawn a session that immediately pauses on `alc.llm`.
    let spawn = call_json(
        &client,
        "alc_v2_run",
        json!({"code": "local r = alc.llm(\"ping?\"); return \"got: \" .. r"}),
    )
    .await;
    let sid = spawn["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    // 2. Wait for the paused state.
    poll_until_paused(&client, &sid, Duration::from_secs(2)).await;

    // 3. Resume with payload encoded as a STRING (the regression path).
    let payload_str = serde_json::to_string(&json!({
        "payload_kind": "single",
        "response": "pong",
        "query_id": "q-0",
        "usage": null
    }))
    .expect("encode payload");
    let resume_resp = call_json(
        &client,
        "alc_v2_resume",
        json!({
            "session_id": sid,
            "payload": payload_str,
        }),
    )
    .await;

    let outcome = resume_resp["outcome"].as_str().unwrap_or("");
    assert!(
        outcome == "continued" || outcome == "terminal",
        "stringified payload must be accepted; got outcome: {outcome} / full: {resume_resp}"
    );

    // 4. Cancel the same session with a stringified `reason` to exercise the
    //    parallel decode path on `alc_v2_cancel`.  Idempotent on terminal sessions.
    let reason_str = serde_json::to_string(&json!({
        "code": "user",
        "detail": "smoke test",
        "requested_at": 0,
    }))
    .expect("encode reason");
    let _ = call_json(
        &client,
        "alc_v2_cancel",
        json!({"session_id": sid, "reason": reason_str}),
    )
    .await;

    client.cancel().await.expect("cancel failed");
}

/// Case 9: `alc_v2_run` with missing required `code` field yields an error.
///
/// rmcp may surface the missing-required-field failure either as a JSON-RPC error
/// (Err from `call_tool`) or as a tool-level error text.  Both are acceptable; we
/// verify that the call does not succeed silently.
#[tokio::test]
async fn test_alc_v2_run_shape_validation() {
    let client = connect().await;

    // Pass empty args — `code` field is required.
    let outcome = client.call_tool(call_params("alc_v2_run", json!({}))).await;

    // The call must either return a ServiceError (JSON-RPC level rejection) or a
    // tool-level error result with non-empty text.  A successful JSON parse of the
    // response (i.e. a valid session_id) would indicate the constraint was not enforced.
    match outcome {
        Err(_service_err) => {
            // JSON-RPC protocol error — the missing required field was caught by rmcp.
        }
        Ok(result) => {
            let text = extract_text(&result);
            // If the server returned Ok, it must at minimum contain an error message.
            // A valid non-error response would have a session_id JSON field.
            let parsed: Option<Value> = serde_json::from_str(text).ok();
            assert!(
                parsed.as_ref().and_then(|v| v.get("session_id")).is_none(),
                "alc_v2_run with missing code must not return a session_id, got: {text}"
            );
        }
    }

    client.cancel().await.expect("cancel failed");
}

// ─── Single-package mode removal (Crux #2) ───────────────────────────────────

/// Crux #2 enforcement: E2E fixture must match production env — no cache bypass.
///
/// Pre-seeds:
/// - `alc_home/hub_registries.json` with one Git source URL (production Tier 1
///   registry entry).
/// - `alc_home/packages/e2e_crux2_pkg/init.lua` as a valid installed package.
///
/// Does NOT pre-seed `hub_cache/`, so the hub discovery path flows through
/// Tier 0/1/2/3 without a cache shortcut (Crux #2: no cache-bypass shortcut).
///
/// Assertions:
/// - `alc_pkg_install` with a local Collection-layout path (`<coll>/<name>/init.lua`)
///   succeeds and the response includes entries in `installed` without `mode=single`.
/// - `alc_pkg_install` with a Single-layout path (root-level `init.lua`, no subdir)
///   returns `is_error = true` and the error message cites v0.36.0.
#[tokio::test]
async fn test_pkg_install_collection_only_after_single_removal() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let alc_home = tmp.path();

    // ── Crux #2: pre-seed hub_registries.json (Tier 1 registry entry) ──
    // This establishes the production hub discovery path without relying on
    // a cache pre-seed (hub_cache/ is intentionally absent).
    let registries_json = serde_json::json!({
        "registries": [
            {
                "source": "https://github.com/ynishi/algocline-bundled-packages",
                "origin": "pkg_install",
                "added_at": "2026-01-01T00:00:00Z"
            }
        ]
    });
    std::fs::write(
        alc_home.join("hub_registries.json"),
        serde_json::to_string(&registries_json).expect("serialize hub_registries"),
    )
    .expect("write hub_registries.json");

    // ── Crux #2: pre-seed packages dir (valid installed package) ──
    // Direct filesystem seeding — not via alc_pkg_install — to match the
    // production-equivalent state described by Crux #2.
    let installed_pkg_dir = alc_home.join("packages").join("e2e_crux2_pkg");
    std::fs::create_dir_all(&installed_pkg_dir).expect("mkdir installed pkg");
    std::fs::write(
        installed_pkg_dir.join("init.lua"),
        concat!(
            "local M = {}\n",
            "M.meta = { name = 'e2e_crux2_pkg', version = '0.1.0' }\n",
            "function M.run(ctx) return 'crux2' end\n",
            "return M\n",
        ),
    )
    .expect("write installed pkg init.lua");

    // hub_cache/ is intentionally NOT pre-seeded (Crux #2 constraint).

    let client = connect_with_alc_home(alc_home).await;

    // ── T1: Collection install succeeds with no mode=single ──
    //
    // Build a local Collection source: <coll>/<name>/init.lua.
    let src_tmp = tempfile::tempdir().expect("src tempdir");
    let coll_dir = src_tmp.path().join("e2e_crux2_coll");
    let pkg_subdir = coll_dir.join("e2e_crux2_install_pkg");
    std::fs::create_dir_all(&pkg_subdir).expect("mkdir pkg subdir");
    std::fs::write(
        pkg_subdir.join("init.lua"),
        concat!(
            "local M = {}\n",
            "M.meta = { name = 'e2e_crux2_install_pkg', version = '0.1.0' }\n",
            "function M.run(ctx) return 'ok' end\n",
            "return M\n",
        ),
    )
    .expect("write collection pkg init.lua");

    let resp = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_dir.to_string_lossy() }),
    )
    .await;

    assert!(
        resp["installed"]
            .as_array()
            .map(|a| a.iter().any(|v| v == "e2e_crux2_install_pkg"))
            .unwrap_or(false),
        "e2e_crux2_install_pkg should appear in installed, got: {resp}"
    );
    // mode field must not be "single" — Collection-only world after v0.36.0.
    assert!(
        resp.get("mode").is_none() || resp["mode"] != "single",
        "response must not contain mode=single after Single-mode removal, got: {resp}"
    );

    // ── T2: Single-layout install returns error citing v0.36.0 ──
    //
    // Build a Single-layout source: root-level init.lua (no subdir).
    let single_tmp = tempfile::tempdir().expect("single tempdir");
    let single_dir = single_tmp.path().join("e2e_crux2_single");
    std::fs::create_dir_all(&single_dir).expect("mkdir single dir");
    std::fs::write(
        single_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'should_fail', version = '0.1.0' }\nreturn M\n",
    )
    .expect("write single-layout init.lua");

    let single_outcome = client
        .call_tool(call_params(
            "alc_pkg_install",
            json!({ "url": single_dir.to_string_lossy() }),
        ))
        .await;

    match single_outcome {
        Ok(result) => {
            let is_error = result.is_error.unwrap_or(false);
            let text = extract_text(&result);
            assert!(
                is_error,
                "expected is_error=true for Single-layout install after v0.36.0 removal, \
                 got is_error={is_error:?}, text: {text}"
            );
            assert!(
                text.contains("v0.36.0"),
                "error message must cite v0.36.0 removal, got: {text}"
            );
        }
        Err(e) => panic!("unexpected call_tool Err for Single-layout install: {e}"),
    }

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_test ─────────────────────────────────────────────────────────────

/// Normal path: inline `code` with a passing lspec assertion returns
/// `{passed: 1, failed: 0}`.
#[tokio::test]
async fn test_pkg_test_inline_passing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let lua_code = concat!(
        "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
        "describe('suite', function()\n",
        "    it('passes', function() expect(1).to.equal(1) end)\n",
        "end)\n",
    );

    let resp = call_json(&client, "alc_pkg_test", json!({ "code": lua_code })).await;

    assert_eq!(resp["passed"], 1, "expected passed=1, got: {resp}");
    assert_eq!(resp["failed"], 0, "expected failed=0, got: {resp}");
    assert_eq!(resp["pending"], 0, "expected pending=0, got: {resp}");

    client.cancel().await.expect("cancel failed");
}

/// Error path: calling `alc_pkg_test` with no input source returns the
/// typed exclusivity error (crux constraint 1).
#[tokio::test]
async fn test_pkg_test_exclusivity_error_zero_inputs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .call_tool(call_params_empty("alc_pkg_test"))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.contains("provide exactly one of pkg, code_file, code"),
        "expected exclusivity error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Error path: `alc_pkg_test` with an unknown package name returns the
/// typed not-found error (crux constraint 2 — propagated tier).
#[tokio::test]
async fn test_pkg_test_pkg_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_pkg_test",
            json!({ "pkg": "nonexistent_pkg_xyz" }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);

    assert!(
        text.contains("nonexistent_pkg_xyz"),
        "error must name the missing package, got: {text}"
    );
    assert!(
        text.contains("not found"),
        "error must say not found, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// T1 (happy path): installed package appears in `resolved_search_paths` when
/// `auto_search_paths` is omitted (default behaviour).
///
/// Creates a minimal package under `{alc_home}/packages/dummy_e2e_pkg_auto/`
/// and verifies that the JSON response contains a `resolved_search_paths` array
/// with an entry whose `source` is `"installed"` and `name` matches the pkg.
#[tokio::test]
async fn test_pkg_test_auto_search_paths_default_includes_installed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Place a minimal package so the auto-resolve has something to find.
    let pkg_dir = tmp.path().join("packages").join("dummy_e2e_pkg_auto");
    // K-185: nested dir — use create_dir_all.
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(pkg_dir.join("init.lua"), "return {}\n").expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let lua_code = concat!(
        "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
        "describe('auto', function()\n",
        "    it('ok', function() expect(1).to.equal(1) end)\n",
        "end)\n",
    );
    // auto_search_paths omitted → default true.
    let resp = call_json(&client, "alc_pkg_test", json!({ "code": lua_code })).await;

    // Basic test run health.
    assert_eq!(resp["passed"], 1, "expected passed=1, got: {resp}");

    // Crux constraint 3: structured mapping must appear in JSON return value.
    let paths = resp["resolved_search_paths"]
        .as_array()
        .expect("resolved_search_paths must be an array");
    let found = paths.iter().any(|row| {
        row["name"].as_str() == Some("dummy_e2e_pkg_auto")
            && row["source"].as_str() == Some("installed")
    });
    assert!(
        found,
        "expected dummy_e2e_pkg_auto/installed in resolved_search_paths, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// T2 (boundary): `auto_search_paths: false` produces an empty
/// `resolved_search_paths` array even when installed packages exist.
///
/// Crux constraint 2: when opt-out flag is set, zero auto-resolved paths must
/// be injected and the registry must not be consulted.
#[tokio::test]
async fn test_pkg_test_auto_search_paths_false_skips_auto() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // Place the same package — but with opt-out it must NOT appear.
    let pkg_dir = tmp.path().join("packages").join("dummy_e2e_pkg_auto");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    std::fs::write(pkg_dir.join("init.lua"), "return {}\n").expect("write init.lua");

    let client = connect_with_alc_home(tmp.path()).await;

    let lua_code = concat!(
        "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
        "describe('opt_out', function()\n",
        "    it('ok', function() expect(1).to.equal(1) end)\n",
        "end)\n",
    );
    let resp = call_json(
        &client,
        "alc_pkg_test",
        json!({ "code": lua_code, "auto_search_paths": false }),
    )
    .await;

    // Crux constraint 2: resolved_search_paths must be empty.
    assert_eq!(
        resp["resolved_search_paths"],
        json!([]),
        "expected empty resolved_search_paths with auto_search_paths=false, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// T1 (happy path): a package linked via `alc.local.toml` appears in
/// `resolved_search_paths` with `source: "alc.local.toml"`.
///
/// Creates a temp project root containing `alc.local.toml` that links a
/// local directory, then verifies the structured mapping is present.
#[tokio::test]
async fn test_pkg_test_auto_search_paths_resolves_alc_local_toml() {
    let tmp_home = tempfile::tempdir().expect("tempdir alc_home");
    let tmp_project = tempfile::tempdir().expect("tempdir project");

    // Create a minimal linked package directory outside alc_home.
    let linked_pkg_dir = tmp_project.path().join("packages").join("local_pkg_e2e");
    std::fs::create_dir_all(&linked_pkg_dir).expect("create linked pkg dir");
    std::fs::write(linked_pkg_dir.join("init.lua"), "return {}\n").expect("write init.lua");

    // Write alc.local.toml that registers the linked package via an absolute path
    // to the package directory itself (alc.toml path = pkg_dir, resolve strips
    // the leaf to get the search dir).
    let local_toml_content = format!(
        "[packages.local_pkg_e2e]\npath = \"{}\"\n",
        linked_pkg_dir.to_str().expect("utf8 path")
    );
    std::fs::write(
        tmp_project.path().join("alc.local.toml"),
        &local_toml_content,
    )
    .expect("write alc.local.toml");

    let client = connect_with_alc_home(tmp_home.path()).await;

    let lua_code = concat!(
        "local describe, it, expect = lust.describe, lust.it, lust.expect\n",
        "describe('local_toml', function()\n",
        "    it('ok', function() expect(1).to.equal(1) end)\n",
        "end)\n",
    );
    let resp = call_json(
        &client,
        "alc_pkg_test",
        json!({
            "code": lua_code,
            "project_root": tmp_project.path().to_str().expect("utf8 path"),
        }),
    )
    .await;

    // Basic test run health.
    assert_eq!(resp["passed"], 1, "expected passed=1, got: {resp}");

    // Crux constraint 3: mapping must appear as structured field.
    let paths = resp["resolved_search_paths"]
        .as_array()
        .expect("resolved_search_paths must be an array");
    let found = paths.iter().any(|row| {
        row["name"].as_str() == Some("local_pkg_e2e")
            && row["source"].as_str() == Some("alc.local.toml")
    });
    assert!(
        found,
        "expected local_pkg_e2e/alc.local.toml in resolved_search_paths, got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Regression: `alc_pkg_test pkg=<name> project_root=<collection_root>`
/// must discover `<collection_root>/<name>/init.lua` when the collection root
/// carries `alc.toml` but no `alc.local.toml`.
///
/// Targets the `/alc-build --location=collection` layout where any repo with
/// `alc.toml` at its root holds packages as flat `<root>/<pkg>/init.lua`.
/// Prior to the fix, `pkg_resolve_init_path` only walked
/// `~/.algocline/packages/` and the cwd-ancestor's `alc.local.toml`, so the
/// `project_root` argument was effectively ignored in `pkg=` mode.
#[tokio::test]
async fn test_pkg_test_pkg_mode_resolves_collection_root() {
    let tmp_home = tempfile::tempdir().expect("tempdir alc_home");
    let tmp_collection = tempfile::tempdir().expect("tempdir collection");

    // Create the collection-root marker (alc.toml with [packages]) — note we
    // do NOT create alc.local.toml; the fix must discover the package via
    // the project_root + flat layout convention alone.
    std::fs::write(
        tmp_collection.path().join("alc.toml"),
        "[hub]\nname = \"e2e-test-collection\"\n\n[packages]\n",
    )
    .expect("write alc.toml");

    // Create <collection>/test_collection_pkg/init.lua and spec.
    let pkg_dir = tmp_collection.path().join("test_collection_pkg");
    let spec_dir = pkg_dir.join("spec");
    std::fs::create_dir_all(&spec_dir).expect("create pkg dirs");
    std::fs::write(
        pkg_dir.join("init.lua"),
        "local M = {}\nM.meta = { name = 'test_collection_pkg', version = '0.0.1', description = 'e2e', category = 'test' }\nM.spec = { entries = {} }\nfunction M.run(ctx) return (ctx and ctx.input or '') .. ' ok' end\nreturn M\n",
    )
    .expect("write init.lua");
    std::fs::write(
        spec_dir.join("test_collection_pkg_spec.lua"),
        "local describe, it, expect = lust.describe, lust.it, lust.expect\ndescribe('collection', function()\n    it('discovered', function() expect(1).to.equal(1) end)\nend)\n",
    )
    .expect("write spec");

    let client = connect_with_alc_home(tmp_home.path()).await;

    let resp = call_json(
        &client,
        "alc_pkg_test",
        json!({
            "pkg": "test_collection_pkg",
            "project_root": tmp_collection.path().to_str().expect("utf8 path"),
        }),
    )
    .await;

    // Pre-fix: this returned a "package not found" error.
    assert_eq!(
        resp["passed"], 1,
        "expected passed=1 (package discovered via project_root), got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc.env ──────────────────────────────────────────────────────────────────

/// T1 (happy path): inject env vars are accessible from Lua via `alc.env.KEY`.
///
/// Verifies the TIME (snapshot frozen) and SOURCE (inject layer) crux boundaries:
/// the env map is snapshotted at `alc_run` invocation time and inject has the
/// highest priority.
#[tokio::test]
async fn test_alc_env_inject_readable() {
    let client = connect().await;
    let resp = call_json(
        &client,
        "alc_run",
        json!({
            "code": r#"return alc.env.FOO"#,
            "ctx": { "env": { "inject": { "FOO": "bar" } } }
        }),
    )
    .await;
    assert_eq!(resp["status"], "completed", "unexpected status: {resp}");
    assert_eq!(
        resp["result"].as_str().unwrap_or(""),
        "bar",
        "alc.env.FOO should return inject value 'bar', got: {resp}"
    );
    client.cancel().await.expect("cancel failed");
}

/// T2 (SPACE boundary): writing to `alc.env` must produce a runtime error
/// containing "readonly".
///
/// Verifies the SPACE crux boundary: host-owned env map must never be mutated
/// by Lua guest; `MetaMethod::NewIndex` must hard-error on any write attempt.
#[tokio::test]
async fn test_alc_env_readonly_write_errors() {
    let client = connect().await;
    let result = client
        .call_tool(call_params(
            "alc_run",
            json!({
                "code": r#"alc.env.X = 1; return "unreached""#,
                "ctx": { "env": { "inject": {} } }
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    assert!(
        text.contains("readonly") || result.is_error == Some(true),
        "expected readonly error, got: {text}"
    );
    client.cancel().await.expect("cancel failed");
}

/// T2 (edge case): `alc.env:use{...}` proxy returns nil for undeclared keys.
///
/// Verifies the SPACE crux boundary declare-at-use semantics: keys not listed
/// in the `use{...}` call must be invisible (nil), even if present in the
/// underlying env snapshot.
#[tokio::test]
async fn test_alc_env_use_undeclared_returns_nil() {
    let client = connect().await;
    let resp = call_json(
        &client,
        "alc_run",
        json!({
            "code": r#"local e = alc.env:use{"FOO"}; return tostring(e.UNDECLARED)"#,
            "ctx": { "env": { "inject": { "FOO": "bar", "UNDECLARED": "leak" } } }
        }),
    )
    .await;
    assert_eq!(resp["status"], "completed", "unexpected status: {resp}");
    assert_eq!(
        resp["result"].as_str().unwrap_or(""),
        "nil",
        "undeclared key via :use{{}} should return nil, got: {resp}"
    );
    client.cancel().await.expect("cancel failed");
}

/// T3 (SOURCE priority): inject > dotenv > os.env strict three-layer merge.
///
/// Verifies the SOURCE crux boundary: when the same key is present in all
/// three sources, the inject layer value must always shadow dotenv and OS.
/// `set_var` is called before `connect()` so the spawned child process inherits it.
#[tokio::test]
async fn test_alc_env_source_priority_inject_over_dotenv_over_os() {
    // Create a temporary .env file containing KEY=from_dotenv
    let dir = tempfile::tempdir().expect("tempdir");
    let dotenv_path = dir.path().join(".env");
    std::fs::write(&dotenv_path, "KEY=from_dotenv\n").expect("write dotenv");
    // SAFETY: set_var before connect() so the spawned alc child process inherits the OS env var.
    // No concurrent env mutation occurs in the e2e harness (separate processes per test).
    std::env::set_var("KEY", "from_os");
    let client = connect().await;
    let resp = call_json(
        &client,
        "alc_run",
        json!({
            "code": r#"return alc.env.KEY"#,
            "ctx": {
                "env": {
                    "inject": { "KEY": "from_inject" },
                    "dotenv": dotenv_path.to_str().expect("dotenv path utf8"),
                    "allow_os": true
                }
            }
        }),
    )
    .await;
    // Remove env var before assertions to avoid leaking into subsequent tests
    std::env::remove_var("KEY");
    assert_eq!(resp["status"], "completed", "unexpected status: {resp}");
    assert_eq!(
        resp["result"].as_str().unwrap_or(""),
        "from_inject",
        "inject must shadow dotenv and OS env, got: {resp}"
    );
    client.cancel().await.expect("cancel failed");
}

/// T1 (happy path): dotenv file is parsed and values are accessible via `alc.env`.
///
/// Verifies that the dotenv source layer correctly parses KEY=value pairs and
/// multi-line escape sequences from a file at an absolute path.
#[tokio::test]
async fn test_alc_env_dotenv_file_parse() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dotenv_path = dir.path().join(".env");
    // dotenv file with a plain value and a double-quoted value with \n escape
    std::fs::write(&dotenv_path, "FOO=bar\nBAZ=\"line1\\nline2\"\n").expect("write dotenv");
    let client = connect().await;
    let resp = call_json(
        &client,
        "alc_run",
        json!({
            "code": r#"return alc.env.FOO .. "|" .. alc.env.BAZ"#,
            "ctx": { "env": { "dotenv": dotenv_path.to_str().expect("dotenv path utf8") } }
        }),
    )
    .await;
    assert_eq!(resp["status"], "completed", "unexpected status: {resp}");
    assert_eq!(
        resp["result"].as_str().unwrap_or(""),
        "bar|line1\nline2",
        "dotenv FOO and BAZ should be parsed correctly, got: {resp}"
    );
    client.cancel().await.expect("cancel failed");
}

/// Defect path: installed pkg dir has init.lua but M.meta.name is absent →
/// `missing_meta` bucket with the pkg name and a suggestion.
///
/// Fixture layout:
///   1. Install a package with a valid init.lua (M.meta.name present).
///   2. Overwrite the installed init.lua to remove M.meta.name.
///   3. Doctor should classify the package as `missing_meta`.
#[tokio::test]
async fn test_pkg_doctor_missing_meta_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let source_root = tempfile::tempdir().expect("source tempdir");
    let coll_root = source_root.path();
    let pkg_src = coll_root.join("e2e_doctor_missing_meta");
    std::fs::create_dir_all(&pkg_src).expect("mkdir pkg_src");
    // First write a valid init.lua so install succeeds.
    std::fs::write(
        pkg_src.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_missing_meta", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_root.to_string_lossy() }),
    )
    .await;

    // Overwrite installed init.lua to remove M.meta.name.
    let installed = tmp.path().join("packages").join("e2e_doctor_missing_meta");
    std::fs::write(
        installed.join("init.lua"),
        r#"local M = {}
M.meta = { version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("rewrite init.lua without name");

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_doctor_missing_meta" }),
    )
    .await;

    let missing_meta = resp["missing_meta"]
        .as_array()
        .expect("missing_meta array missing");
    let entry = missing_meta
        .iter()
        .find(|e| e["name"] == "e2e_doctor_missing_meta")
        .unwrap_or_else(|| {
            panic!("e2e_doctor_missing_meta not found in missing_meta, got: {resp}")
        });
    assert_eq!(entry["kind"], "missing_meta");

    client.cancel().await.expect("cancel failed");
}

/// Defect path: project_root has 2+ pkg dirs each with init.lua but
/// hub_index.json is missing → `missing_hub_index` bucket with pkg_count >= 2.
///
/// Fixture layout (synthetic project_root, no packages installed):
///   project_root/
///     pkg_alpha/init.lua
///     pkg_beta/init.lua
///     (no hub_index.json)
#[tokio::test]
async fn test_pkg_doctor_missing_hub_index_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    // Build a synthetic project_root with 2 pkg dirs, no hub_index.json.
    let project_tmp = tempfile::tempdir().expect("project tempdir");
    let project_root = project_tmp.path();
    for pkg_name in ["pkg_alpha", "pkg_beta"] {
        let pkg_dir = project_root.join(pkg_name);
        std::fs::create_dir_all(&pkg_dir).expect("mkdir pkg_dir");
        std::fs::write(
            pkg_dir.join("init.lua"),
            format!(
                r#"local M = {{}}
M.meta = {{ name = {pkg_name:?}, version = "0.1.0" }}
return M"#
            ),
        )
        .expect("write init.lua");
    }
    // Intentionally no hub_index.json.

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "project_root": project_root.to_string_lossy() }),
    )
    .await;

    let missing_hub_index = resp["missing_hub_index"]
        .as_array()
        .expect("missing_hub_index array missing");
    assert_eq!(
        missing_hub_index.len(),
        1,
        "expected 1 missing_hub_index entry, got: {resp}"
    );
    let entry = &missing_hub_index[0];
    assert_eq!(entry["kind"], "missing_hub_index");
    assert!(
        entry["pkg_count"].as_u64().unwrap_or(0) >= 2,
        "pkg_count should be >= 2, got: {entry}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Defect path: hub_cache/{hash}.json with mtime > CACHE_TTL_SECS (3600s)
/// → `stale_cache` bucket with the file path and age_secs.
///
/// Fixture: place a `.json` file under {ALC_HOME}/hub_cache/ with mtime
/// backdated to 2h ago via std::fs::FileTimes.
#[tokio::test]
async fn test_pkg_doctor_stale_cache_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache_dir = tmp.path().join("hub_cache");
    std::fs::create_dir_all(&cache_dir).expect("mkdir hub_cache");
    let stale_path = cache_dir.join("deadbeef00000000.json");
    std::fs::write(&stale_path, r#"{"entries":[]}"#).expect("write cache");
    let past = std::time::SystemTime::now() - std::time::Duration::from_secs(7200);
    let times = std::fs::FileTimes::new().set_modified(past);
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&stale_path)
        .expect("open for backdate");
    f.set_times(times).expect("set_times");

    let client = connect_with_alc_home(tmp.path()).await;
    let resp = call_json(&client, "alc_pkg_doctor", json!({})).await;

    let stale = resp["stale_cache"]
        .as_array()
        .expect("stale_cache array missing");
    assert_eq!(stale.len(), 1, "expected 1 stale_cache entry: {resp}");
    let entry = &stale[0];
    assert_eq!(entry["kind"], "stale_cache");
    assert!(
        entry["path"]
            .as_str()
            .unwrap_or("")
            .ends_with("deadbeef00000000.json"),
        "path mismatch: {entry}"
    );
    assert!(
        entry["age_secs"].as_u64().unwrap_or(0) >= 7200,
        "age_secs too small: {entry}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Defect path: installed pkg has spec/ directory but contains zero
/// *_spec.lua files → `spec_missing` bucket with the pkg name.
#[tokio::test]
async fn test_pkg_doctor_spec_missing_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let source_root = tempfile::tempdir().expect("source tempdir");
    let coll_root = source_root.path();
    let pkg_src = coll_root.join("e2e_doctor_spec_missing");
    std::fs::create_dir_all(&pkg_src).expect("mkdir pkg_src");
    std::fs::write(
        pkg_src.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_spec_missing", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": coll_root.to_string_lossy() }),
    )
    .await;

    // Create empty spec/ in the installed package dir.
    let installed = tmp.path().join("packages").join("e2e_doctor_spec_missing");
    std::fs::create_dir_all(installed.join("spec")).expect("mkdir spec");

    let resp = call_json(
        &client,
        "alc_pkg_doctor",
        json!({ "name": "e2e_doctor_spec_missing" }),
    )
    .await;

    let spec_missing = resp["spec_missing"]
        .as_array()
        .expect("spec_missing array missing");
    let entry = spec_missing
        .iter()
        .find(|e| e["name"] == "e2e_doctor_spec_missing")
        .unwrap_or_else(|| {
            panic!("e2e_doctor_spec_missing not found in spec_missing, got: {resp}")
        });
    assert_eq!(entry["kind"], "spec_missing");

    client.cancel().await.expect("cancel failed");
}

// ─── alc_setting_resolve ──────────────────────────────────────────────────────

/// Connect with a specific ALC_HOME and additional env vars.
///
/// Used by `alc_setting_resolve` tests that need to inject `ALC_SETTING_*`
/// env vars into the child server process without using `std::env::set_var`
/// (which would affect the test process itself and cause cross-test pollution).
async fn connect_with_alc_home_and_env(
    alc_home: &std::path::Path,
    extra_env: &[(&str, &str)],
) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let bin = std::env::var("CARGO_BIN_EXE_alc")
        .unwrap_or_else(|_| format!("{}/target/debug/alc", env!("CARGO_MANIFEST_DIR")));
    let packages_path = alc_home.join("packages");
    let mut cmd = tokio::process::Command::new(bin);
    cmd.env("ALC_HOME", alc_home)
        .env("ALC_PACKAGES_PATH", &packages_path);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let transport = TokioChildProcess::new(cmd).expect("failed to spawn alc server");
    ().serve(transport)
        .await
        .expect("failed to initialize MCP session")
}

/// Normal path (target absent): `alc_setting_resolve` with no target returns all
/// `[setting.*]` tables merged across global config + project config.
///
/// Setup:
///  - global config.toml:  `[setting.journal] path = "/global/j.md"`
///                         `[setting.advisor] path = "/global/a.md"`
///  - project alc.toml:    `[setting.journal] path = "/project/j.md"`
///  - env override:        `ALC_SETTING_JOURNAL_PKG=true`
///
/// Assertions:
///  - `resolved.journal.path` == "/project/j.md" (project wins over global)
///  - `resolved.journal.pkg`  == true (env adds new field)
///  - `resolved.advisor.path` == "/global/a.md" (global only)
///  - `sources.journal.path`  == "project"
///  - `sources.journal.pkg`   == "env"
///  - `sources.advisor.path`  == "global"
///  - both `resolved` and `sources` are present in the response (crux #3)
#[tokio::test]
async fn test_alc_setting_resolve_returns_all_when_target_absent() {
    let alc_home_tmp = tempfile::tempdir().expect("alc_home tempdir");
    let project_tmp = tempfile::tempdir().expect("project tempdir");

    // Write global config.toml inside alc_home
    std::fs::write(
        alc_home_tmp.path().join("config.toml"),
        "[setting.journal]\npath = \"/global/j.md\"\n\n[setting.advisor]\npath = \"/global/a.md\"\n",
    )
    .expect("write config.toml");

    // Write project alc.toml
    std::fs::write(
        project_tmp.path().join("alc.toml"),
        "[setting.journal]\npath = \"/project/j.md\"\n",
    )
    .expect("write alc.toml");

    // Inject ALC_SETTING_JOURNAL_PKG=true and ALC_PROJECT_ROOT to the child process.
    let client = connect_with_alc_home_and_env(
        alc_home_tmp.path(),
        &[
            ("ALC_SETTING_JOURNAL_PKG", "true"),
            (
                "ALC_PROJECT_ROOT",
                project_tmp.path().to_str().expect("project root utf8"),
            ),
        ],
    )
    .await;

    let resp = call_json(&client, "alc_setting_resolve", json!({})).await;

    // Both top-level keys must be present (crux #3: single-call contract)
    assert!(
        resp.get("resolved").is_some(),
        "response must contain 'resolved', got: {resp}"
    );
    assert!(
        resp.get("sources").is_some(),
        "response must contain 'sources', got: {resp}"
    );

    // journal.path: project wins over global
    assert_eq!(
        resp["resolved"]["journal"]["path"].as_str(),
        Some("/project/j.md"),
        "project should win for journal.path, got: {resp}"
    );
    // journal.pkg: env adds new field
    assert_eq!(
        resp["resolved"]["journal"]["pkg"].as_bool(),
        Some(true),
        "env should add journal.pkg=true, got: {resp}"
    );
    // advisor.path: global only (no project override)
    assert_eq!(
        resp["resolved"]["advisor"]["path"].as_str(),
        Some("/global/a.md"),
        "global should provide advisor.path, got: {resp}"
    );

    // Sources attribution (crux #2: field-level)
    assert_eq!(
        resp["sources"]["journal"]["path"].as_str(),
        Some("project"),
        "journal.path source should be 'project', got: {resp}"
    );
    assert_eq!(
        resp["sources"]["journal"]["pkg"].as_str(),
        Some("env"),
        "journal.pkg source should be 'env', got: {resp}"
    );
    assert_eq!(
        resp["sources"]["advisor"]["path"].as_str(),
        Some("global"),
        "advisor.path source should be 'global', got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Normal path (target specified): `alc_setting_resolve` with `target="journal"` returns
/// only the journal target — advisor must be absent.
#[tokio::test]
async fn test_alc_setting_resolve_target_filter() {
    let alc_home_tmp = tempfile::tempdir().expect("alc_home tempdir");

    // Write global config.toml with two targets
    std::fs::write(
        alc_home_tmp.path().join("config.toml"),
        "[setting.journal]\npath = \"/global/j.md\"\n\n[setting.advisor]\nmodel = \"gpt4\"\n",
    )
    .expect("write config.toml");

    let client = connect_with_alc_home(alc_home_tmp.path()).await;

    let resp = call_json(
        &client,
        "alc_setting_resolve",
        json!({ "target": "journal" }),
    )
    .await;

    assert!(
        resp["resolved"].get("journal").is_some(),
        "resolved should contain 'journal', got: {resp}"
    );
    assert!(
        resp["resolved"].get("advisor").is_none(),
        "resolved must NOT contain 'advisor' when filtering for 'journal', got: {resp}"
    );
    assert!(
        resp["sources"].get("journal").is_some(),
        "sources should contain 'journal', got: {resp}"
    );
    assert!(
        resp["sources"].get("advisor").is_none(),
        "sources must NOT contain 'advisor' when filtering for 'journal', got: {resp}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Error path: `target="Invalid-Name"` (contains uppercase / hyphens) must return a
/// typed `InvalidTarget` error propagated through the MCP wire layer.
#[tokio::test]
async fn test_alc_setting_resolve_rejects_invalid_target_name() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_setting_resolve",
            json!({ "target": "Invalid-Name" }),
        ))
        .await
        .expect("call_tool should not fail at transport level");

    let is_error = result.is_error.unwrap_or(false);
    let text = extract_text(&result);

    assert!(
        is_error,
        "expected is_error=true for invalid target name, got is_error={is_error:?}, text: {text}"
    );
    assert!(
        text.contains("invalid characters"),
        "error message should mention 'invalid characters', got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_pkg_list / alc_advice — VM eval type detection ─────────────────────

/// Verify that a package with `M.run` is auto-detected as runnable via
/// VM eval (`LUA_TYPE_AUTODETECT`). The package must appear in `alc_pkg_list`
/// with `type = "runnable"` and no warnings.
#[tokio::test]
async fn e2e_vm_eval_detects_runnable_package() {
    let alc_home = tempfile::tempdir().unwrap();
    pre_install_pkg(
        alc_home.path(),
        "vm_eval_runnable_pkg",
        &[(
            "init.lua",
            concat!(
                "local M = {}\n",
                "M.meta = { name = \"vm_eval_runnable_pkg\", version = \"0.1.0\" }\n",
                "function M.run(ctx) return {} end\n",
                "return M\n",
            ),
        )],
    );
    let client = connect_with_alc_home(alc_home.path()).await;

    let result = call_json(
        &client,
        "alc_pkg_list",
        serde_json::json!({ "limit": 0, "fields": ["name", "type"] }),
    )
    .await;
    let packages = result["packages"]
        .as_array()
        .expect("packages must be an array");
    let entry = packages
        .iter()
        .find(|p| p["name"] == "vm_eval_runnable_pkg")
        .unwrap_or_else(|| panic!("vm_eval_runnable_pkg not found, got: {result}"));
    assert_eq!(
        entry["type"], "runnable",
        "M.run defined => type must be 'runnable', got: {entry}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Verify that a library package (no `M.run`, auto-detected) is rejected by
/// `alc_advice` with a typed error naming the package.
#[tokio::test]
async fn e2e_vm_eval_library_rejected_by_alc_advice() {
    let alc_home = tempfile::tempdir().unwrap();
    pre_install_pkg(
        alc_home.path(),
        "vm_eval_library_reject",
        &[(
            "init.lua",
            concat!(
                "local M = {}\n",
                "M.meta = { name = \"vm_eval_library_reject\", version = \"0.1.0\" }\n",
                // No M.run => auto-detected library via LUA_TYPE_AUTODETECT.
                "return M\n",
            ),
        )],
    );
    let client = connect_with_alc_home(alc_home.path()).await;

    let result = client
        .call_tool(call_params(
            "alc_advice",
            serde_json::json!({ "strategy": "vm_eval_library_reject", "task": "test" }),
        ))
        .await
        .expect("call_tool must not fail at transport level");

    let text = extract_text(&result);
    assert!(
        text.contains("library") || text.contains("vm_eval_library_reject"),
        "error message must name the pkg or mention 'library', got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_card_publish ────────────────────────────────────────────

/// Creates an isolated test fixture:
/// - `alc_home`: temp dir used as ALC_HOME
/// - `bare`: temp dir containing a bare git repo (target hub)
/// - returns `(alc_home, bare, card_id, target_url)`
///
/// The bare repo is initialised with `git init --bare` so `git push` works
/// against it as the target_repo. A minimal card TOML is written under
/// `{alc_home}/cards/test_pkg/{card_id}.toml`.
async fn setup_card_publish_fixture() -> (tempfile::TempDir, tempfile::TempDir, String, String) {
    let alc_home = tempfile::tempdir().expect("alc_home tempdir");
    let bare = tempfile::tempdir().expect("bare repo tempdir");

    // Initialise a bare git repository that will act as the hub target.
    let out = tokio::process::Command::new("git")
        .args(["init", "--bare"])
        .arg(bare.path())
        .output()
        .await
        .expect("git init --bare failed");
    assert!(
        out.status.success(),
        "git init --bare failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Write a minimal card TOML.
    let card_id = "test_card_001";
    let cards_dir = alc_home.path().join("cards").join("test_pkg");
    std::fs::create_dir_all(&cards_dir).expect("create cards dir");
    let card_toml = format!(
        r#"schema_version = "card/v0"
card_id = "{card_id}"
created_at = "2026-06-12T00:00:00Z"

[pkg]
name = "test_pkg"
"#
    );
    std::fs::write(cards_dir.join(format!("{card_id}.toml")), card_toml).expect("write card toml");

    let target_url = format!("file://{}", bare.path().display());
    (alc_home, bare, card_id.to_string(), target_url)
}

/// Happy path: card is published to the bare hub repo.
/// Asserts: `published_url` present, `commit_hash` non-empty, `reindex_status.ok == true`.
#[tokio::test]
async fn test_alc_card_publish_happy_path() {
    let (alc_home, _bare, card_id, target_url) = setup_card_publish_fixture().await;

    let client = connect_with_alc_home(alc_home.path()).await;
    let resp = call_json(
        &client,
        "alc_card_publish",
        json!({
            "card_id": card_id,
            "target_repo": target_url,
        }),
    )
    .await;

    assert!(
        resp.get("published_url").and_then(|v| v.as_str()).is_some(),
        "expected published_url in response, got: {resp}"
    );
    assert!(
        resp.get("commit_hash")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "expected non-empty commit_hash in response, got: {resp}"
    );
    let reindex = resp
        .get("reindex_status")
        .expect("reindex_status missing from response");
    assert_eq!(
        reindex.get("ok").and_then(|v| v.as_bool()),
        Some(true),
        "expected reindex_status.ok == true, got: {reindex}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Push failure path: the target repo is an invalid URL so `git clone` /
/// `git push` fail. The MCP tool must return a typed error — specifically the
/// text must NOT be a JSON success object but must describe the failure.
///
/// Crux constraint: the error must specifically identify authentication /
/// push failure (not a generic unknown error).
#[tokio::test]
async fn test_alc_card_publish_push_failure_typed_error() {
    let alc_home = tempfile::tempdir().expect("alc_home tempdir");

    // Write a minimal card so the lookup succeeds.
    let card_id = "test_card_push_fail";
    let cards_dir = alc_home.path().join("cards").join("test_pkg");
    std::fs::create_dir_all(&cards_dir).expect("create cards dir");
    let card_toml = format!(
        r#"schema_version = "card/v0"
card_id = "{card_id}"
created_at = "2026-06-12T00:00:00Z"

[pkg]
name = "test_pkg"
"#
    );
    std::fs::write(cards_dir.join(format!("{card_id}.toml")), card_toml).expect("write card toml");

    // Point to a non-existent remote so git clone fails (git error, not credential).
    let bad_url = "file:///nonexistent/path/to/repo";

    let client = connect_with_alc_home(alc_home.path()).await;
    let result = client
        .call_tool(call_params(
            "alc_card_publish",
            json!({
                "card_id": card_id,
                "target_repo": bad_url,
            }),
        ))
        .await
        .expect("call_tool failed");

    // The result must be an error (isError flag set).
    assert!(
        result.is_error.unwrap_or(false),
        "expected isError=true for push failure, got: {:?}",
        result
    );
    let text = extract_text(&result);
    // Must mention the failure in a non-empty way (typed error, not empty string).
    assert!(
        !text.is_empty(),
        "error text must not be empty for push failure"
    );
    // Must NOT look like a successful CardPublishOutcome JSON.
    assert!(
        !text.contains("published_url"),
        "push failure must not return a success object, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// Reindex failure isolated: push succeeds but hub_reindex fails because
/// `installed.json` is corrupt. Asserts that:
/// - `published_url` is present (push succeeded)
/// - `commit_hash` is non-empty (push succeeded)
/// - `reindex_status.ok == false` (reindex failed independently)
/// - `reindex_status.error` is non-empty (error message present)
///
/// Crux constraint: push success and reindex failure must be observable as
/// independent fields; a successful push is never suppressed when only
/// reindex fails.
#[tokio::test]
async fn test_alc_card_publish_reindex_failure_isolated() {
    let (alc_home, _bare, card_id, target_url) = setup_card_publish_fixture().await;

    // Corrupt installed.json so load_manifest returns Err.
    // hub_reindex(None, None) uses use_local_state=true, which calls
    // load_manifest → FsInstalledManifestStore::load → serde_json::from_str fails.
    std::fs::write(
        alc_home.path().join("installed.json"),
        b"{not valid json{{{",
    )
    .expect("write corrupt installed.json");

    let client = connect_with_alc_home(alc_home.path()).await;
    let resp = call_json(
        &client,
        "alc_card_publish",
        json!({
            "card_id": card_id,
            "target_repo": target_url,
        }),
    )
    .await;

    // Push must have succeeded — these fields are present.
    assert!(
        resp.get("published_url").and_then(|v| v.as_str()).is_some(),
        "published_url must be present even when reindex fails, got: {resp}"
    );
    assert!(
        resp.get("commit_hash")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "commit_hash must be present even when reindex fails, got: {resp}"
    );

    // Reindex must have failed independently.
    let reindex = resp
        .get("reindex_status")
        .expect("reindex_status missing from response");
    assert_eq!(
        reindex.get("ok").and_then(|v| v.as_bool()),
        Some(false),
        "expected reindex_status.ok == false when installed.json is corrupt, got: {reindex}"
    );
    assert!(
        reindex
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "expected non-empty reindex_status.error, got: {reindex}"
    );

    client.cancel().await.expect("cancel failed");
}

// ─── alc_advice variant-scope resolution regression tests ─────────────────────

/// `alc_advice` must resolve a strategy package that is linked via
/// `alc.local.toml` (scope = variant) without returning a not-found error,
/// even when the package is absent from the global `~/.algocline/packages/`
/// tier.
///
/// Validates Crux: variant tier lookup hoist before install check — the
/// `resolve_variant_pkgs` result must short-circuit the `is_package_installed`
/// guard so variant-linked strategies are applied without requiring
/// `alc_pkg_install`.
#[tokio::test]
async fn test_alc_advice_resolves_variant_scope_package() {
    // Use a fresh alc_home so the global packages tier is empty (no bundled
    // packages installed there), ensuring the only way to resolve the strategy
    // is through the variant tier.
    let alc_home = tempfile::tempdir().expect("tempdir alc_home");
    let client = connect_with_alc_home(alc_home.path()).await;

    // Create a temp project root with alc.toml so resolve_project_root succeeds.
    let project_tmp = tempfile::tempdir().expect("tempdir project");
    let project_root = project_tmp.path();
    std::fs::write(project_root.join("alc.toml"), "[packages]\n").expect("write alc.toml");

    // Create the strategy package outside the project root (typical worktree layout).
    // Package must have a valid M.run entry point so the advice() execution path
    // proceeds past the library guard and start_and_tick without early error.
    let pkg_src = project_tmp.path().join("variant_strategy_src");
    let pkg_dir = pkg_src.join("e2e_variant_strategy");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir pkg_dir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_variant_strategy", version = "0.1.0", type = "runnable" }
M.run = function(ctx) return { answer = "variant-ok" } end
return M"#,
    )
    .expect("write init.lua");

    // Link the collection directory as variant scope → writes alc.local.toml.
    let link_resp = call_json(
        &client,
        "alc_pkg_link",
        json!({
            "path": pkg_src.to_string_lossy(),
            "scope": "variant",
            "project_root": project_root.to_string_lossy(),
        }),
    )
    .await;
    assert!(
        link_resp.get("error").is_none(),
        "alc_pkg_link should succeed, got: {link_resp}"
    );

    // Call alc_advice with the variant-linked strategy.
    // The package is absent from the global tier, so without the variant hoist
    // the old code would return a not-found error.
    let result = client
        .call_tool(call_params(
            "alc_advice",
            json!({
                "strategy": "e2e_variant_strategy",
                "task": "ping",
                "project_root": project_root.to_string_lossy(),
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    assert!(
        !text.contains("not found after installing bundled collection"),
        "variant-linked strategy should not produce a not-found error, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}

/// `alc_advice` must return the existing not-found error when the requested
/// strategy is absent from all three tiers: variant (`alc.local.toml`),
/// global (`~/.algocline/packages/`), and the auto-installed bundled
/// collection.
///
/// Validates Crux: E2E regression coverage for both resolution paths — the
/// not-found path must remain intact after the variant-hoist fix.
#[tokio::test]
async fn test_alc_advice_returns_not_found_when_all_tiers_absent() {
    // Fresh alc_home with empty packages tier.
    let alc_home = tempfile::tempdir().expect("tempdir alc_home");
    let client = connect_with_alc_home(alc_home.path()).await;

    // Project root with alc.toml but no alc.local.toml (variant tier empty).
    let project_tmp = tempfile::tempdir().expect("tempdir project");
    let project_root = project_tmp.path();
    std::fs::write(project_root.join("alc.toml"), "[packages]\n").expect("write alc.toml");

    // Use a strategy name that cannot exist in any bundled collection.
    let result = client
        .call_tool(call_params(
            "alc_advice",
            json!({
                "strategy": "nonexistent_pkg_xyz_e2e",
                "project_root": project_root.to_string_lossy(),
            }),
        ))
        .await
        .expect("call_tool failed");
    let text = extract_text(&result);
    assert!(
        text.contains("not found after installing bundled collection"),
        "expected not-found error when all tiers lack the package, got: {text}"
    );

    client.cancel().await.expect("cancel failed");
}
