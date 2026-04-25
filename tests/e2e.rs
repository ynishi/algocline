//! E2E tests for the algocline MCP server.
//!
//! Uses rmcp client to spawn the `alc` binary as a child process,
//! communicate via stdio MCP protocol, and validate responses with
//! insta snapshots.

use std::borrow::Cow;
use std::io::Write;

use rmcp::{
    model::{CallToolRequestParams, ReadResourceRequestParams},
    transport::TokioChildProcess,
    ServiceExt,
};
use serde_json::{json, Map, Value};

use algocline_app::PRESET_CATALOG_VERSION;

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
        .read_resource(ReadResourceRequestParams {
            uri: uri.to_string(),
            meta: None,
        })
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

    // 1. Create temp packages and install them
    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");

    let pkg_a_dir = tmp_dir.path().join("e2e_fork_a");
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

    let pkg_b_dir = tmp_dir.path().join("e2e_fork_b");
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

    // Install via MCP
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_a_dir.to_string_lossy() }),
    )
    .await;
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_b_dir.to_string_lossy() }),
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

    // Create a temporary package
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let pkg_dir = tmp_dir.path().join("e2e_types_test");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_types_test", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install and check response
    let resp = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_dir.to_string_lossy() }),
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

    // Create a temporary package
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let pkg_dir = tmp_dir.path().join("e2e_alc_shapes_types_test");
    std::fs::create_dir_all(&pkg_dir).expect("mkdir");
    std::fs::write(
        pkg_dir.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_alc_shapes_types_test", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install and check response
    let resp = call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_dir.to_string_lossy() }),
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
    let pkg_dir = tmp_dir.path().join(pkg_name);
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
        json!({ "url": pkg_dir.to_string_lossy() }),
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

    // 2. Create a temp pkg dir living OUTSIDE the project root (typical worktree workflow).
    let pkg_dir = tmp.path().join("variant_src").join("e2e_variant_pkg");
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
            "path": pkg_dir.to_string_lossy(),
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

    // Source pkg dir outside HOME.
    let tmp = tempfile::tempdir().expect("tempdir");
    let source = tmp.path().join("e2e_repair_pkg");
    std::fs::create_dir_all(&source).expect("mkdir");
    std::fs::write(
        source.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_repair_pkg", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install.
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": source.to_string_lossy() }),
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

    // Source pkg dir outside HOME.
    let tmp = tempfile::tempdir().expect("tempdir");
    let source = tmp.path().join("e2e_doctor_pkg");
    std::fs::create_dir_all(&source).expect("mkdir");
    std::fs::write(
        source.join("init.lua"),
        r#"local M = {}
M.meta = { name = "e2e_doctor_pkg", version = "0.1.0" }
function M.run(ctx) return "ok" end
return M"#,
    )
    .expect("write init.lua");

    // Install.
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": source.to_string_lossy() }),
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

/// `resources/list` must return exactly 2 fixed resources.
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
        2,
        "expected 2 fixed resources, got: {:?}",
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

    client.cancel().await.expect("cancel failed");
}

/// `resources/templates/list` must return exactly 7 templates.
#[tokio::test]
async fn test_mcp_resource_templates_list_returns_seven() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = connect_with_alc_home(tmp.path()).await;

    let result = client
        .list_resource_templates(None)
        .await
        .expect("list_resource_templates failed");

    assert_eq!(
        result.resource_templates.len(),
        7,
        "expected 7 resource templates, got: {:?}",
        result
            .resource_templates
            .iter()
            .map(|t| t.raw.uri_template.as_str())
            .collect::<Vec<_>>()
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

    // Build source package with init.lua + sub.lua.
    let source_root = tempfile::tempdir().expect("source tempdir");
    let pkg_src = source_root.path().join("e2e_doctor_complete");
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

    // Install the package into the isolated ALC_HOME.
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_src.to_string_lossy() }),
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

    // Build source package with init.lua + missing_sub.lua.
    let source_root = tempfile::tempdir().expect("source tempdir");
    let pkg_src = source_root.path().join("e2e_doctor_incomplete");
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

    // Install.
    call_json(
        &client,
        "alc_pkg_install",
        json!({ "url": pkg_src.to_string_lossy() }),
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
