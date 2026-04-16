//! E2E tests for the algocline MCP server.
//!
//! Uses rmcp client to spawn the `alc` binary as a child process,
//! communicate via stdio MCP protocol, and validate responses with
//! insta snapshots.

use std::borrow::Cow;
use std::io::Write;

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

    let repaired = resp["repaired"]
        .as_array()
        .expect("repaired array missing");
    assert_eq!(repaired.len(), 1, "one repair expected, got: {resp}");
    assert_eq!(repaired[0]["name"], "e2e_repair_pkg");
    assert_eq!(repaired[0]["kind"], "installed_missing");
    assert!(dest.exists(), "dest should be restored after repair");

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dest);
    client.cancel().await.expect("cancel failed");
}
