//! `alc.fork()` — parallel multi-VM strategy execution.
//!
//! Spawns N independent Lua VMs, each running one strategy with the same ctx.
//! LLM requests from all children are batched and sent through the parent's
//! llm_tx, achieving true LLM parallelism.

use std::path::PathBuf;

use algocline_core::{BudgetHandle, ExecutionMetrics, QueryId};
use mlua::prelude::*;
use mlua::LuaSerdeExt;
use mlua_isle::{AsyncIsle, AsyncIsleDriver};
use mlua_pkg::{resolvers::FsResolver, Registry};

use super::{register, BridgeConfig, PRELUDE};
use crate::llm_bridge::{LlmRequest, QueryRequest};
use crate::variant_pkg::{register_variant_pkgs, VariantPkg};

/// Event from a child VM during fork execution.
enum ForkEvent {
    /// Child VM emitted an LLM request.
    Request {
        vm_index: usize,
        queries: Vec<ForkQuery>,
    },
    /// Child VM completed execution.
    Completed {
        vm_index: usize,
        result: Result<String, String>,
    },
}

/// A single LLM query from a fork child, with the child's response channel.
struct ForkQuery {
    prompt: String,
    system: Option<String>,
    max_tokens: u32,
    grounded: bool,
    underspecified: bool,
    child_resp_tx: tokio::sync::oneshot::Sender<Result<String, String>>,
}

/// Register `alc.fork(strategies, ctx, opts?)` onto the given table.
///
/// Lua usage:
///   local results = alc.fork({"cot", "reflect", "cove"}, ctx)
///   -- results = { {strategy="cot", result=...}, {strategy="reflect", result=...}, ... }
///
///   local results = alc.fork({"cot", "reflect"}, ctx, { on_error = "skip" })
pub(crate) fn register_fork(
    lua: &Lua,
    alc_table: &LuaTable,
    llm_tx: tokio::sync::mpsc::Sender<LlmRequest>,
    budget: BudgetHandle,
    lib_paths: Vec<PathBuf>,
    variant_pkgs: Vec<VariantPkg>,
) -> LuaResult<()> {
    let fork_fn = lua.create_async_function(
        move |lua, (strategies, ctx, opts): (LuaTable, LuaTable, Option<LuaTable>)| {
            let parent_tx = llm_tx.clone();
            let bh = budget.clone();
            let paths = lib_paths.clone();
            let variants = variant_pkgs.clone();
            async move {
                let n = strategies.len()? as usize;
                if n == 0 {
                    return Err(LuaError::external(
                        "alc.fork: strategies must be a non-empty array",
                    ));
                }

                let on_error = opts
                    .as_ref()
                    .and_then(|o| o.get::<String>("on_error").ok())
                    .unwrap_or_else(|| "skip".into());

                // Collect strategy names (validated to prevent Lua injection)
                let mut strategy_names = Vec::with_capacity(n);
                for i in 1..=n {
                    let name: String = strategies.get(i)?;
                    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        return Err(LuaError::external(format!(
                            "alc.fork: invalid strategy name '{}' \
                             (only alphanumeric and underscore allowed)",
                            name
                        )));
                    }
                    strategy_names.push(name);
                }

                // Serialize ctx to JSON for child VMs
                let ctx_json: serde_json::Value = lua.from_value(LuaValue::Table(ctx))?;

                // Aggregated event channel
                let (event_tx, mut event_rx) =
                    tokio::sync::mpsc::channel::<ForkEvent>(16 * n.max(1));

                // Spawn child VMs and their event-forwarding tasks.
                // drivers must stay alive to keep child VMs running.
                let mut drivers: Vec<AsyncIsleDriver> = Vec::with_capacity(n);

                for (vm_idx, strategy) in strategy_names.iter().enumerate() {
                    let (child_llm_tx, mut child_llm_rx) =
                        tokio::sync::mpsc::channel::<LlmRequest>(16);

                    // Spawn child VM
                    let child_paths = paths.clone();
                    let child_variants = variants.clone();
                    let (child_isle, child_driver) = AsyncIsle::spawn(move |child_lua| {
                        let mut reg = Registry::new();
                        // Variant pkgs first (highest priority — alc.local.toml wins).
                        register_variant_pkgs(&mut reg, &child_variants);
                        for path in &child_paths {
                            match FsResolver::new(path) {
                                Ok(resolver) => {
                                    reg.add(resolver);
                                }
                                Err(e) => {
                                    eprintln!(
                                        "alc.fork: warn: FsResolver failed for {}: {e}",
                                        path.display()
                                    );
                                }
                            }
                        }
                        reg.install(child_lua)?;
                        Ok(())
                    })
                    .await
                    .map_err(|e| {
                        LuaError::external(format!(
                            "alc.fork: VM spawn failed for '{strategy}': {e}"
                        ))
                    })?;

                    // Setup child VM: register alc.*, set ctx, load prelude
                    let child_ctx = ctx_json.clone();
                    let child_metrics = ExecutionMetrics::new();
                    let child_config = BridgeConfig {
                        llm_tx: Some(child_llm_tx),
                        ns: format!("fork-{vm_idx}"),
                        custom_metrics: child_metrics.custom_metrics_handle(),
                        budget: bh.clone(),
                        progress: child_metrics.progress_handle(),
                        lib_paths: vec![],    // Children don't need to fork further
                        variant_pkgs: vec![], // Children don't need to fork further
                    };

                    child_isle
                        .exec(move |child_lua| {
                            let alc_table = child_lua.create_table()?;
                            register(child_lua, &alc_table, child_config)?;
                            child_lua.globals().set("alc", alc_table)?;

                            let ctx_value = child_lua.to_value(&child_ctx)?;
                            child_lua.globals().set("ctx", ctx_value)?;

                            child_lua.load(PRELUDE).exec().map_err(|e| {
                                mlua_isle::IsleError::Lua(format!("Prelude load failed: {e}"))
                            })?;

                            Ok("ok".to_string())
                        })
                        .await
                        .map_err(|e| {
                            LuaError::external(format!(
                                "alc.fork: setup failed for '{strategy}': {e}"
                            ))
                        })?;

                    // Execute strategy as coroutine
                    let code = format!(
                        "return alc.json_encode((function() \
                         local s = require('{}'); return s.run(ctx) \
                         end)())",
                        strategy
                    );
                    let exec_task = child_isle.spawn_coroutine_eval(&code);
                    drop(child_isle); // Release isle handle; driver keeps VM alive

                    drivers.push(child_driver);

                    // Spawn event-forwarding task for this child.
                    // Task terminates naturally when event_tx is dropped (channel close).
                    let evt_tx = event_tx.clone();
                    tokio::spawn(async move {
                        let mut exec_task = exec_task;
                        loop {
                            tokio::select! {
                                biased;
                                result = &mut exec_task => {
                                    let mapped = match result {
                                        Ok(json_str) => Ok(json_str),
                                        Err(e) => Err(e.to_string()),
                                    };
                                    let _ = evt_tx.send(ForkEvent::Completed {
                                        vm_index: vm_idx,
                                        result: mapped,
                                    }).await;
                                    return;
                                }
                                Some(req) = child_llm_rx.recv() => {
                                    let fork_queries = req.queries.into_iter().map(|qr| {
                                        ForkQuery {
                                            prompt: qr.prompt,
                                            system: qr.system,
                                            max_tokens: qr.max_tokens,
                                            grounded: qr.grounded,
                                            underspecified: qr.underspecified,
                                            child_resp_tx: qr.resp_tx,
                                        }
                                    }).collect();
                                    let _ = evt_tx.send(ForkEvent::Request {
                                        vm_index: vm_idx,
                                        queries: fork_queries,
                                    }).await;
                                }
                            }
                        }
                    });
                }
                drop(event_tx); // Only child tasks hold senders now

                // Multiplexer: collect child events, batch LLM requests, distribute responses
                let mut results: Vec<Option<Result<serde_json::Value, String>>> = vec![None; n];
                let mut seq_counter: Vec<usize> = vec![0; n];

                while results.iter().any(|r| r.is_none()) {
                    // Wait for first event
                    let first = match event_rx.recv().await {
                        Some(evt) => evt,
                        None => break, // All senders dropped
                    };

                    // Collect first + drain any immediately ready events
                    let mut events = vec![first];
                    while let Ok(evt) = event_rx.try_recv() {
                        events.push(evt);
                    }

                    // Process events: separate completions from requests
                    let mut batch_queries: Vec<QueryRequest> = Vec::new();
                    let mut parent_resp_rxs: Vec<
                        tokio::sync::oneshot::Receiver<Result<String, String>>,
                    > = Vec::new();
                    let mut child_resp_txs: Vec<
                        tokio::sync::oneshot::Sender<Result<String, String>>,
                    > = Vec::new();

                    for event in events {
                        match event {
                            ForkEvent::Completed { vm_index, result } => {
                                results[vm_index] = Some(match result {
                                    Ok(json_str) => serde_json::from_str(&json_str)
                                        .map_err(|e| format!("JSON parse: {e}")),
                                    Err(e) => Err(e),
                                });
                            }
                            ForkEvent::Request { vm_index, queries } => {
                                for fq in queries {
                                    let fork_id = QueryId::fork(vm_index, seq_counter[vm_index]);
                                    seq_counter[vm_index] += 1;

                                    let (parent_resp_tx, parent_resp_rx) =
                                        tokio::sync::oneshot::channel();

                                    parent_resp_rxs.push(parent_resp_rx);
                                    child_resp_txs.push(fq.child_resp_tx);

                                    batch_queries.push(QueryRequest {
                                        id: fork_id,
                                        prompt: fq.prompt,
                                        system: fq.system,
                                        max_tokens: fq.max_tokens,
                                        grounded: fq.grounded,
                                        underspecified: fq.underspecified,
                                        resp_tx: parent_resp_tx,
                                    });
                                }
                            }
                        }
                    }

                    if batch_queries.is_empty() {
                        continue;
                    }

                    // Send batch to parent session (causes parent to pause)
                    parent_tx
                        .send(LlmRequest {
                            queries: batch_queries,
                        })
                        .await
                        .map_err(|e| {
                            LuaError::external(format!("alc.fork: LLM bridge send failed: {e}"))
                        })?;

                    // Await all responses from host, forward to children
                    for (parent_rx, child_tx) in parent_resp_rxs.into_iter().zip(child_resp_txs) {
                        match parent_rx.await {
                            Ok(result) => {
                                let _ = child_tx.send(result);
                            }
                            Err(e) => {
                                let _ = child_tx.send(Err(format!("alc.fork: response lost: {e}")));
                            }
                        }
                    }
                }

                // Keep drivers alive until all results collected
                drop(drivers);

                // Build result table
                let result_table = lua.create_table()?;
                for (i, (name, result)) in
                    strategy_names.iter().zip(results.into_iter()).enumerate()
                {
                    let entry = lua.create_table()?;
                    entry.set("strategy", name.as_str())?;
                    match result {
                        Some(Ok(val)) => {
                            let lua_val = lua.to_value(&val)?;
                            entry.set("result", lua_val)?;
                            entry.set("ok", true)?;
                        }
                        Some(Err(err)) => {
                            if on_error == "abort" {
                                return Err(LuaError::external(format!(
                                    "alc.fork: strategy '{}' failed: {}",
                                    name, err
                                )));
                            }
                            entry.set("error", err)?;
                            entry.set("ok", false)?;
                        }
                        None => {
                            entry.set("error", "no result (channel closed)")?;
                            entry.set("ok", false)?;
                        }
                    }
                    result_table.set(i + 1, entry)?;
                }

                Ok(result_table)
            }
        },
    )?;

    alc_table.set("fork", fork_fn)?;
    Ok(())
}
