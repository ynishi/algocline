use algocline_core::{BudgetHandle, ProgressHandle, QueryId};
use mlua::prelude::*;

use crate::llm_bridge::{LlmRequest, QueryRequest};

/// Register `alc.llm(prompt, opts?)` — calls Host LLM via coroutine yield.
///
/// Registered as an async function so the Lua coroutine yields while
/// waiting for the LLM response, allowing other coroutines to progress.
///
/// Lua usage:
///   local response = alc.llm("What is 2+2?")
///   local response = alc.llm("Explain X", { system = "You are an expert.", max_tokens = 500 })
///   local response = alc.llm(prompt, { cache_breakpoint = "context" })
pub(super) fn register_llm(
    lua: &Lua,
    alc_table: &LuaTable,
    llm_tx: tokio::sync::mpsc::Sender<LlmRequest>,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let llm =
        lua.create_async_function(move |lua, (prompt, opts): (String, Option<LuaTable>)| {
            let tx = llm_tx.clone();
            let bh = budget.clone();
            async move {
                bh.check().map_err(LuaError::external)?;
                let system = opts.as_ref().and_then(|o| o.get::<String>("system").ok());
                let max_tokens = opts
                    .as_ref()
                    .and_then(|o| o.get::<u32>("max_tokens").ok())
                    .unwrap_or(1024);
                let grounded = opts
                    .as_ref()
                    .and_then(|o| o.get::<bool>("grounded").ok())
                    .unwrap_or(false);
                let underspecified = opts
                    .as_ref()
                    .and_then(|o| o.get::<bool>("underspecified").ok())
                    .unwrap_or(false);
                let cache_breakpoint = opts
                    .as_ref()
                    .and_then(|o| o.get::<String>("cache_breakpoint").ok());
                // Optional role hint (e.g. "grader"). Additive: absent when not set.
                // Other opts keys are ignored for forward compatibility.
                let role = opts.as_ref().and_then(|o| o.get::<String>("role").ok());

                // In-process route for role="nn": a caller with role="nn" and
                // model=<name> is asking an in-VM tiny model to answer. Dispatch
                // to the algocline-nn model registry synchronously — no yield,
                // no host round-trip. Everything else falls through to the normal
                // Host path unchanged.
                #[cfg(feature = "nn")]
                if role.as_deref() == Some("nn") {
                    let model_name = opts
                        .as_ref()
                        .and_then(|o| o.get::<String>("model").ok())
                        .ok_or_else(|| {
                            LuaError::external("alc.llm role=\"nn\": missing model = <name>")
                        })?;
                    let registry = lua
                        .app_data_ref::<algocline_nn::NnModelRegistry>()
                        .ok_or_else(|| {
                            LuaError::external(
                                "alc.llm role=\"nn\": alc.nn registry missing — has alc.nn been \
                             initialised on this VM?",
                            )
                        })?;
                    let forward = registry.get(&model_name).ok_or_else(|| {
                        LuaError::external(format!(
                            "alc.llm role=\"nn\": no model registered as {model_name:?}"
                        ))
                    })?;
                    // Drop the borrow before calling the closure so a re-entrant
                    // alc.llm inside the model closure would not deadlock the
                    // registry cell.
                    drop(registry);
                    return forward.call::<String>(prompt);
                }

                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();

                tx.send(LlmRequest {
                    queries: vec![QueryRequest {
                        id: QueryId::single(),
                        prompt,
                        system,
                        max_tokens,
                        grounded,
                        underspecified,
                        cache_breakpoint,
                        role,
                        resp_tx,
                    }],
                })
                .await
                .map_err(|e| LuaError::external(format!("LLM bridge send failed: {e}")))?;

                resp_rx
                    .await
                    .map_err(|e| LuaError::external(format!("LLM bridge recv failed: {e}")))?
                    .map_err(LuaError::external)
            }
        })?;

    alc_table.set("llm", llm)?;
    Ok(())
}

/// Register `alc.llm_batch(items)` — parallel LLM calls via coroutine yield.
///
/// All queries are sent as a single batch, then the coroutine yields
/// while awaiting all responses concurrently.
///
/// Lua usage:
///   local responses = alc.llm_batch({
///       { prompt = "Analyze A" },
///       { prompt = "Analyze B", system = "expert", max_tokens = 500 },
///   })
///   -- responses[1], responses[2] in same order as input
pub(super) fn register_llm_batch(
    lua: &Lua,
    alc_table: &LuaTable,
    llm_tx: tokio::sync::mpsc::Sender<LlmRequest>,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let llm_batch = lua.create_async_function(move |_, items: LuaTable| {
        let tx = llm_tx.clone();
        let bh = budget.clone();
        async move {
            bh.check().map_err(LuaError::external)?;
            let len = items.len()? as usize;
            if len == 0 {
                return Err(LuaError::external("alc.llm_batch: empty items array"));
            }

            let mut query_requests = Vec::with_capacity(len);
            let mut resp_rxs = Vec::with_capacity(len);

            for i in 1..=len {
                let item: LuaTable = items.get(i)?;
                let prompt: String = item.get("prompt")?;
                let system: Option<String> = item.get::<LuaValue>("system").ok().and_then(|v| {
                    if let LuaValue::String(s) = v {
                        Some(s.to_str().ok()?.to_string())
                    } else {
                        None
                    }
                });
                let max_tokens: u32 = item.get::<u32>("max_tokens").unwrap_or(1024);
                let grounded: bool = item.get::<bool>("grounded").unwrap_or(false);
                let underspecified: bool = item.get::<bool>("underspecified").unwrap_or(false);
                let cache_breakpoint: Option<String> =
                    item.get::<LuaValue>("cache_breakpoint").ok().and_then(|v| {
                        if let LuaValue::String(s) = v {
                            Some(s.to_str().ok()?.to_string())
                        } else {
                            None
                        }
                    });
                // Optional role hint per item (additive, forward-compat).
                let role: Option<String> = item.get::<LuaValue>("role").ok().and_then(|v| {
                    if let LuaValue::String(s) = v {
                        Some(s.to_str().ok()?.to_string())
                    } else {
                        None
                    }
                });

                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                resp_rxs.push(resp_rx);

                query_requests.push(QueryRequest {
                    id: QueryId::batch(i - 1), // 0-indexed
                    prompt,
                    system,
                    max_tokens,
                    grounded,
                    underspecified,
                    cache_breakpoint,
                    role,
                    resp_tx,
                });
            }

            // Send all queries as a single batch
            tx.send(LlmRequest {
                queries: query_requests,
            })
            .await
            .map_err(|e| LuaError::external(format!("LLM bridge send failed: {e}")))?;

            // Await all responses concurrently (order matches input)
            let mut responses = Vec::with_capacity(len);
            for (i, rx) in resp_rxs.into_iter().enumerate() {
                let resp = rx
                    .await
                    .map_err(|e| {
                        LuaError::external(format!("LLM bridge recv failed for q-{i}: {e}"))
                    })?
                    .map_err(LuaError::external)?;
                responses.push(resp);
            }

            Ok(responses)
        }
    })?;

    alc_table.set("llm_batch", llm_batch)?;
    Ok(())
}

/// Register `alc.budget_remaining()` — query remaining budget.
///
/// Lua return type:
/// - `nil` if no budget was set (ctx.budget absent)
/// - `{ llm_calls = N|nil, elapsed_ms = N|nil }` where each field is
///   present only if the corresponding limit was set. Values are
///   remaining capacity (saturating at 0).
pub(super) fn register_budget_remaining(
    lua: &Lua,
    alc_table: &LuaTable,
    budget: BudgetHandle,
) -> LuaResult<()> {
    let budget_fn = lua.create_function(move |lua, ()| {
        let remaining = budget.remaining();
        lua.to_value(&remaining)
    })?;
    alc_table.set("budget_remaining", budget_fn)?;
    Ok(())
}

/// Register `alc.progress(step, total, msg?)` — report structured progress.
///
/// Writes progress info into SessionStatus, readable via `alc_status` MCP tool.
/// Not all strategies need to call this — it is opt-in for strategies that
/// benefit from structured step tracking.
///
/// Lua usage:
///   alc.progress(1, 5, "Analyzing chunk 1")
///   alc.progress(2, 5)  -- message is optional
pub(super) fn register_progress(
    lua: &Lua,
    alc_table: &LuaTable,
    progress: ProgressHandle,
) -> LuaResult<()> {
    let progress_fn =
        lua.create_function(move |_, (step, total, msg): (u64, u64, Option<String>)| {
            progress.set(step, total, msg);
            Ok(())
        })?;
    alc_table.set("progress", progress_fn)?;
    Ok(())
}
