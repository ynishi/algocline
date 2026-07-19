use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use algocline_core::{CustomMetricsHandle, LogEntry, LogSink, StatsHandle};
use mlua::prelude::*;
use mlua::{LuaSerdeExt, SerializeOptions};

use crate::card::{self, FileCardStore};
use crate::state::{JsonFileStore, StateStore};

/// Converts a `serde_json::Value` to a Lua value with JSON null surfacing as
/// Lua nil (not the mlua default lightuserdata sentinel).
///
/// All `bridge::data` accessors that expose serde JSON to Lua funnel through
/// this helper so the null-handling contract is uniform across `alc.json_decode`,
/// `alc.state.*`, `alc.card.*`, `alc.stats.*`, and similar bridge surfaces.
/// JSON `null` always decodes to Lua `nil`, preventing `if obj.x then ...`
/// truthy checks from passing through nullable fields and crashing downstream
/// operations (e.g. `io.open(value)`).
///
/// See issues db041966 (originating fix for `alc.json_decode`) and ff6372af
/// (sweep that unified all 16 callsites within `bridge/data.rs`).
///
/// Generic over `T: serde::Serialize` so the same contract applies to
/// `serde_json::Value` callers (which can carry `Value::Null`) and to typed
/// structs that may contain `Option<_>` fields (e.g. `SinkBackfillReport`).
/// Plain types without any nullable shape (`Vec<String>`, etc.) are still
/// safe to funnel through this helper — the option toggles only affect
/// `None` / `()` paths.
fn to_lua_value<T: serde::Serialize + ?Sized>(lua: &Lua, value: &T) -> LuaResult<LuaValue> {
    let options = SerializeOptions::new()
        .serialize_none_to_null(false)
        .serialize_unit_to_null(false);
    lua.to_value_with(value, options)
}

pub(super) fn register_json(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let encode = lua.create_function(|lua, value: LuaValue| {
        let json: serde_json::Value = lua.from_value(value)?;
        serde_json::to_string(&json).map_err(LuaError::external)
    })?;

    // alc.json_decode funnels through `to_lua_value` for the canonical
    // JSON null -> Lua nil contract. See issue db041966.
    let decode = lua.create_function(|lua, s: String| {
        let value: serde_json::Value = serde_json::from_str(&s).map_err(LuaError::external)?;
        to_lua_value(lua, &value)
    })?;

    alc_table.set("json_encode", encode)?;
    alc_table.set("json_decode", decode)?;
    Ok(())
}

/// Register `alc.log(level, msg)` — routes Lua log calls to tracing and to the
/// per-session [`LogSink`] ring buffer.
///
/// # Arguments
///
/// - `lua` — The Lua VM.
/// - `alc_table` — The `alc` table to register the function on.
/// - `log_sink` — Shared ring buffer; the entry is pushed in addition to the
///   existing tracing output so stderr tail is unaffected.
///
/// # Errors
///
/// Returns `LuaError` only if function or table registration fails (mlua infra).
pub(super) fn register_log(lua: &Lua, alc_table: &LuaTable, log_sink: LogSink) -> LuaResult<()> {
    let log = lua.create_function(move |_, (level, msg): (String, String)| {
        // Existing tracing path — preserves stderr/log-file output.
        match level.as_str() {
            "error" => tracing::error!(target: "alc.log", "{}", msg),
            "warn" => tracing::warn!(target: "alc.log", "{}", msg),
            "info" => tracing::info!(target: "alc.log", "{}", msg),
            "debug" => tracing::debug!(target: "alc.log", "{}", msg),
            _ => tracing::info!(target: "alc.log", "{}", msg),
        }
        // Push to per-session ring buffer for alc_status recent_logs.
        log_sink.push(LogEntry::new(level.clone(), "alc.log", msg));
        Ok(())
    })?;

    alc_table.set("log", log)?;
    Ok(())
}

/// Register a Lua `print()` override that routes output to the per-session
/// [`LogSink`] ring buffer and to `tracing::info!(target: "alc.lua.print")`.
///
/// Behaviour mirrors the standard `print`:
/// - Multiple arguments are joined with `"\t"`.
/// - Each argument is coerced to a string.
/// - Trailing newlines are stripped before storing in the ring buffer.
///
/// The existing tracing path is preserved so operator `tail -f` workflows
/// are unaffected.  `io.write` is intentionally left unchanged.
///
/// # Arguments
///
/// - `lua` — The Lua VM.
/// - `log_sink` — Shared ring buffer for this session.
///
/// # Errors
///
/// Returns `LuaError` only if function or global registration fails (mlua infra).
pub(super) fn register_print(lua: &Lua, log_sink: LogSink) -> LuaResult<()> {
    let print_fn = lua.create_function(move |lua_inner, args: mlua::MultiValue| {
        let parts: Vec<String> = args
            .iter()
            .map(|v| match v {
                LuaValue::Nil => "nil".to_string(),
                LuaValue::Boolean(b) => b.to_string(),
                LuaValue::Integer(n) => n.to_string(),
                LuaValue::Number(n) => {
                    // Reproduce Lua's default float formatting: no trailing zeros
                    // for whole-number values.
                    if n.fract() == 0.0 && n.abs() < 1e15_f64 {
                        format!("{n:.1}")
                    } else {
                        format!("{n}")
                    }
                }
                other => lua_inner
                    .coerce_string(other.clone())
                    .ok()
                    .flatten()
                    .and_then(|s| s.to_str().ok().map(|r| r.to_string()))
                    .unwrap_or_else(|| format!("{other:?}")),
            })
            .collect();
        let line = parts.join("\t");
        // Emit to tracing — operator log-file / stderr path preserved.
        tracing::info!(target: "alc.lua.print", "{}", line);
        // Push trimmed message to per-session ring buffer.
        let message = line.trim_end_matches('\n').to_string();
        log_sink.push(LogEntry::new("info", "alc.lua.print", message));
        Ok(())
    })?;
    lua.globals().set("print", print_fn)?;
    Ok(())
}

/// Register `alc.state` table with get/set/keys/delete/has/set_nx/incr/list/show/reset.
///
/// Lua usage:
///   alc.state.set("score", 42)
///   local v = alc.state.get("score")       -- 42
///   local v = alc.state.get("missing", 0)  -- 0 (default)
///   local k = alc.state.keys()             -- {"score"}
///   alc.state.delete("score")
///   alc.state.has("score")                 -- false
///   alc.state.set_nx("score", 100)         -- true (set because absent)
///   alc.state.incr("counter")              -- 1 (init 0 + delta 1)
///   alc.state.incr("counter", 5)           -- 6
///   alc.state.incr("counter", 10, 100)     -- 16 (default ignored)
///   alc.state.list("my_ns")               -- {"task_a", "task_b"} (sorted)
///   alc.state.show("my_ns", "task_a")     -- full JSON table
///   alc.state.reset("my_ns", "task_a", {steps={"1b_X"}, fields={"x"}})
///                                          -- { ok=true, backup_path="...", steps_removed=1, fields_removed=1 }
pub(super) fn register_state(
    lua: &Lua,
    alc_table: &LuaTable,
    ns: String,
    state_store: Arc<JsonFileStore>,
) -> LuaResult<()> {
    let state_table = lua.create_table()?;

    // alc.state.get(key, default?)
    let ns_get = ns.clone();
    let store_get = Arc::clone(&state_store);
    let get =
        lua.create_function(
            move |lua, (key, default): (String, Option<LuaValue>)| match store_get
                .get(&ns_get, &key)
            {
                Ok(Some(v)) => to_lua_value(lua, &v),
                Ok(None) => Ok(default.unwrap_or(LuaValue::Nil)),
                Err(e) => Err(LuaError::external(e)),
            },
        )?;

    // alc.state.set(key, value)
    let ns_set = ns.clone();
    let store_set = Arc::clone(&state_store);
    let set = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        store_set
            .set(&ns_set, &key, json)
            .map_err(LuaError::external)
    })?;

    // alc.state.keys()
    let ns_keys = ns.clone();
    let store_keys = Arc::clone(&state_store);
    let keys = lua.create_function(move |lua, ()| {
        let k = store_keys.keys(&ns_keys).map_err(LuaError::external)?;
        to_lua_value(lua, &k)
    })?;

    // alc.state.delete(key)
    let ns_del = ns.clone();
    let store_del = Arc::clone(&state_store);
    let delete = lua.create_function(move |_, key: String| {
        store_del.delete(&ns_del, &key).map_err(LuaError::external)
    })?;

    // alc.state.has(key) -> bool
    let ns_has = ns.clone();
    let store_has = Arc::clone(&state_store);
    let has = lua.create_function(move |_, key: String| {
        store_has.has(&ns_has, &key).map_err(LuaError::external)
    })?;

    // alc.state.set_nx(key, value) -> bool
    let ns_snx = ns.clone();
    let store_snx = Arc::clone(&state_store);
    let set_nx = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        store_snx
            .set_nx(&ns_snx, &key, json)
            .map_err(LuaError::external)
    })?;

    // alc.state.incr(key, delta?, default?) -> number
    let ns_incr = ns;
    let store_incr = Arc::clone(&state_store);
    let incr = lua.create_function(
        move |_, (key, delta, default): (String, Option<f64>, Option<f64>)| {
            store_incr
                .incr(&ns_incr, &key, delta.unwrap_or(1.0), default.unwrap_or(0.0))
                .map_err(LuaError::external)
        },
    )?;

    // alc.state.list(namespace) -> string[]
    let store_list = Arc::clone(&state_store);
    let list = lua.create_function(move |lua, namespace: String| {
        let keys = store_list
            .list_dispatched(&namespace)
            .map_err(LuaError::external)?;
        to_lua_value(lua, &keys)
    })?;

    // alc.state.show(namespace, key) -> table
    let store_show = Arc::clone(&state_store);
    let show = lua.create_function(move |lua, (namespace, key): (String, String)| {
        let v = store_show
            .show_dispatched(&namespace, &key)
            .map_err(LuaError::external)?;
        to_lua_value(lua, &v)
    })?;

    // alc.state.reset(namespace, key, opts?) -> { ok, backup_path, steps_removed, fields_removed }
    let store_reset = Arc::clone(&state_store);
    let reset = lua.create_function(
        move |lua, (namespace, key, opts): (String, String, Option<LuaTable>)| {
            let (steps, fields) = match opts {
                Some(t) => {
                    let s = t.get::<Option<Vec<String>>>("steps")?.unwrap_or_default();
                    let f = t.get::<Option<Vec<String>>>("fields")?.unwrap_or_default();
                    (s, f)
                }
                None => (Vec::new(), Vec::new()),
            };
            let report = store_reset
                .reset_dispatched_with_backup(&namespace, &key, &steps, &fields)
                .map_err(LuaError::external)?;
            let ret = lua.create_table()?;
            ret.set("ok", true)?;
            ret.set(
                "backup_path",
                report.backup_path.to_string_lossy().to_string(),
            )?;
            ret.set("steps_removed", report.steps_removed)?;
            ret.set("fields_removed", report.fields_removed)?;
            Ok(ret)
        },
    )?;

    // alc.state.set_dispatched(namespace, key, value) -> nil  (explicit-namespace set)
    let store_set_dispatched = Arc::clone(&state_store);
    let set_dispatched = lua.create_function(
        move |lua, (namespace, key, value): (String, String, LuaValue)| {
            let json: serde_json::Value = lua.from_value(value)?;
            store_set_dispatched
                .set_dispatched(&namespace, &key, &json)
                .map_err(LuaError::external)
        },
    )?;

    // alc.state.delete_dispatched(namespace, key) -> bool  (explicit-namespace delete, existed flag)
    let store_delete_dispatched = Arc::clone(&state_store);
    let delete_dispatched = lua.create_function(move |_, (namespace, key): (String, String)| {
        store_delete_dispatched
            .delete_dispatched(&namespace, &key)
            .map_err(LuaError::external)
    })?;

    state_table.set("get", get)?;
    state_table.set("set", set)?;
    state_table.set("keys", keys)?;
    state_table.set("delete", delete)?;
    state_table.set("has", has)?;
    state_table.set("set_nx", set_nx)?;
    state_table.set("incr", incr)?;
    state_table.set("list", list)?;
    state_table.set("show", show)?;
    state_table.set("reset", reset)?;
    state_table.set("set_dispatched", set_dispatched)?;
    state_table.set("delete_dispatched", delete_dispatched)?;

    alc_table.set("state", state_table)?;
    Ok(())
}

/// Register `alc._dirs` — absolute paths that Lua prelude helpers
/// (`alc.eval` scenario resolution, etc.) need from the service layer.
///
/// Values are plain strings so Lua can concat/`io.open` them without
/// additional userdata binding.
pub(super) fn register_dirs(
    lua: &Lua,
    alc_table: &LuaTable,
    state_dir: &Path,
    cards_dir: &Path,
    scenarios_dir: &Path,
) -> LuaResult<()> {
    let dirs = lua.create_table()?;
    dirs.set("state", state_dir.to_string_lossy().into_owned())?;
    dirs.set("cards", cards_dir.to_string_lossy().into_owned())?;
    dirs.set("scenarios", scenarios_dir.to_string_lossy().into_owned())?;
    alc_table.set("_dirs", dirs)?;
    Ok(())
}

/// Register `alc.card` table with v0 P0+P1 API.
///
/// P0 (minimum viable): create / get / list
/// P1 (observation-driven additions): append / alias_set / alias_list / find
///
/// Lua usage:
///   local c = alc.card.create({ pkg = { name = "cot" }, model = {...}, stats = {...} })
///   local card = alc.card.get("cot_opus46_20260412_a3f9c1")
///   alc.card.list({ pkg = "cot" })
///   alc.card.append("cot_...", { caveats = { notes = "rescored" } })
///   alc.card.alias_set("best_on_gsm8k", "cot_...", { pkg = "cot", note = "..." })
///   alc.card.alias_list({ pkg = "cot" })
///   alc.card.find({
///       pkg = "cot",
///       where = {
///           scenario = { name = "gsm8k" },
///           stats = { pass_rate = { gte = 0.8 } },
///       },
///       order_by = "-stats.pass_rate",
///       limit = 5,
///   })
///   alc.card.get_by_alias("best_on_gsm8k")  -- resolve alias → full Card
///   alc.card.write_samples("cot_...", { {case="c0", passed=true}, ... })  -- write-once
///   alc.card.read_samples("cot_...", { offset = 0, limit = 100 })
pub(super) fn register_card(
    lua: &Lua,
    alc_table: &LuaTable,
    card_store: Arc<FileCardStore>,
    card_run_enabled: bool,
) -> LuaResult<()> {
    let card_table = lua.create_table()?;

    // alc.card.create(table) -> { card_id, path } | nil
    //
    // Validates any optional top-level `run` field before touching the
    // store — invalid statuses surface as Lua errors regardless of the
    // Enable gate.  When `run` is present and `card_run_enabled` is
    // false, the closure short-circuits with `nil` so no card file is
    // written and no `CardEvent::Created` is published (Phase 1-B gate).
    let store_create = Arc::clone(&card_store);
    let create = lua.create_function(move |lua, input: LuaValue| -> LuaResult<LuaValue> {
        let json: serde_json::Value = lua.from_value(input)?;
        let run_section = card::RunSection::from_json(&json).map_err(LuaError::external)?;
        if run_section.is_some() && !card_run_enabled {
            return Ok(LuaValue::Nil);
        }
        let (card_id, path) = store_create.create(json).map_err(LuaError::external)?;
        let ret = lua.create_table()?;
        ret.set("card_id", card_id)?;
        ret.set("path", path.to_string_lossy().to_string())?;
        Ok(LuaValue::Table(ret))
    })?;

    // alc.card.get(card_id) -> table | nil
    let store_get = Arc::clone(&card_store);
    let get = lua.create_function(move |lua, card_id: String| match store_get.get(&card_id) {
        Ok(Some(v)) => to_lua_value(lua, &v),
        Ok(None) => Ok(LuaValue::Nil),
        Err(e) => Err(LuaError::external(e)),
    })?;

    // alc.card.list(filter?) -> [summary]
    let store_list = Arc::clone(&card_store);
    let list = lua.create_function(move |lua, filter: Option<LuaTable>| {
        let pkg = match filter {
            Some(t) => t.get::<Option<String>>("pkg")?,
            None => None,
        };
        let rows = store_list
            .list(pkg.as_deref())
            .map_err(LuaError::external)?;
        to_lua_value(lua, &card::summaries_to_json(&rows))
    })?;

    // alc.card.append(card_id, fields) -> merged_card | nil
    //
    // Symmetric to `create`: any `run` field inside `fields` is validated
    // first, then gated by `card_run_enabled`.  A disabled gate causes
    // the append call to no-op with `nil` — no store rewrite, no
    // `CardEvent::Appended` publication.
    let store_append = Arc::clone(&card_store);
    let append = lua.create_function(
        move |lua, (card_id, fields): (String, LuaValue)| -> LuaResult<LuaValue> {
            let json: serde_json::Value = lua.from_value(fields)?;
            let run_section = card::RunSection::from_json(&json).map_err(LuaError::external)?;
            if run_section.is_some() && !card_run_enabled {
                return Ok(LuaValue::Nil);
            }
            let merged = store_append
                .append(&card_id, json)
                .map_err(LuaError::external)?;
            to_lua_value(lua, &merged)
        },
    )?;

    // alc.card.get_by_alias(name) -> table | nil
    let store_gba = Arc::clone(&card_store);
    let get_by_alias = lua.create_function(move |lua, name: String| {
        match store_gba.get_by_alias(&name).map_err(LuaError::external)? {
            Some(v) => to_lua_value(lua, &v),
            None => Ok(LuaValue::Nil),
        }
    })?;

    // alc.card.alias_set(name, card_id, opts?) -> alias
    let store_aset = Arc::clone(&card_store);
    let alias_set = lua.create_function(
        move |lua, (name, card_id, opts): (String, String, Option<LuaTable>)| {
            let (pkg, note) = match opts {
                Some(t) => (
                    t.get::<Option<String>>("pkg")?,
                    t.get::<Option<String>>("note")?,
                ),
                None => (None, None),
            };
            let a = store_aset
                .alias_set(&name, &card_id, pkg.as_deref(), note.as_deref())
                .map_err(LuaError::external)?;
            let arr = card::aliases_to_json(&[a]);
            let first = match arr {
                serde_json::Value::Array(mut v) if !v.is_empty() => v.remove(0),
                other => other,
            };
            to_lua_value(lua, &first)
        },
    )?;

    // alc.card.alias_list(filter?) -> [alias]
    let store_alist = Arc::clone(&card_store);
    let alias_list = lua.create_function(move |lua, filter: Option<LuaTable>| {
        let pkg = match filter {
            Some(t) => t.get::<Option<String>>("pkg")?,
            None => None,
        };
        let rows = store_alist
            .alias_list(pkg.as_deref())
            .map_err(LuaError::external)?;
        to_lua_value(lua, &card::aliases_to_json(&rows))
    })?;

    // alc.card.find(query?) -> [summary]
    //
    // Accepts a Prisma-style `where` DSL + dotted-path `order_by`.
    // See `card::parse_where` / `card::parse_order_by` for semantics.
    let store_find = Arc::clone(&card_store);
    let find = lua.create_function(move |lua, query: Option<LuaTable>| {
        let q = match query {
            Some(t) => {
                let pkg = t.get::<Option<String>>("pkg")?;
                let limit = t.get::<Option<usize>>("limit")?;
                let offset = t.get::<Option<usize>>("offset")?;

                let where_parsed = match t.get::<LuaValue>("where")? {
                    LuaValue::Nil => None,
                    v => {
                        let json: serde_json::Value = lua.from_value(v)?;
                        Some(card::parse_where(&json).map_err(LuaError::external)?)
                    }
                };
                let order_parsed = match t.get::<LuaValue>("order_by")? {
                    LuaValue::Nil => Vec::new(),
                    v => {
                        let json: serde_json::Value = lua.from_value(v)?;
                        card::parse_order_by(&json).map_err(LuaError::external)?
                    }
                };

                card::FindQuery {
                    pkg,
                    where_: where_parsed,
                    order_by: order_parsed,
                    limit,
                    offset,
                }
            }
            None => card::FindQuery::default(),
        };
        let rows = store_find.find(q).map_err(LuaError::external)?;
        to_lua_value(lua, &card::summaries_to_json(&rows))
    })?;

    // alc.card.write_samples(card_id, samples) -> { path, count }
    let store_ws = Arc::clone(&card_store);
    let write_samples =
        lua.create_function(move |lua, (card_id, samples): (String, LuaValue)| {
            let json: serde_json::Value = lua.from_value(samples)?;
            let arr = match json {
                serde_json::Value::Array(a) => a,
                _ => {
                    return Err(LuaError::external(
                        "alc.card.write_samples: samples must be an array",
                    ))
                }
            };
            let count = arr.len();
            let path = store_ws
                .write_samples(&card_id, arr)
                .map_err(LuaError::external)?;
            let ret = lua.create_table()?;
            ret.set("path", path.to_string_lossy().to_string())?;
            ret.set("count", count)?;
            Ok(ret)
        })?;

    // alc.card.read_samples(card_id, opts?) -> [sample]
    //
    // opts.where applies the Prisma-style DSL to each row; offset/limit
    // page the post-filter stream. See `card::parse_where`.
    let store_rs = Arc::clone(&card_store);
    let read_samples =
        lua.create_function(move |lua, (card_id, opts): (String, Option<LuaTable>)| {
            let (offset, limit, where_parsed) = match opts {
                Some(t) => {
                    let offset = t.get::<Option<usize>>("offset")?.unwrap_or(0);
                    let limit = t.get::<Option<usize>>("limit")?;
                    let where_parsed = match t.get::<LuaValue>("where")? {
                        LuaValue::Nil => None,
                        v => {
                            let json: serde_json::Value = lua.from_value(v)?;
                            Some(card::parse_where(&json).map_err(LuaError::external)?)
                        }
                    };
                    (offset, limit, where_parsed)
                }
                None => (0, None, None),
            };
            let q = card::SamplesQuery {
                offset,
                limit,
                where_: where_parsed,
            };
            let rows = store_rs
                .read_samples(&card_id, q)
                .map_err(LuaError::external)?;
            to_lua_value(lua, &serde_json::Value::Array(rows))
        })?;

    // alc.card.sink_backfill({ sink, dry_run }) -> report
    //
    // Backfill one subscriber with all cards from the primary store.
    // Drift-safe: existing cards on the subscriber are skipped.
    let store_sb = Arc::clone(&card_store);
    let sink_backfill = lua.create_function(move |lua, params: LuaTable| {
        let sink: String = params.get("sink")?;
        let dry_run: Option<bool> = params.get("dry_run")?;
        let report = store_sb
            .card_sink_backfill(&sink, dry_run.unwrap_or(false))
            .map_err(LuaError::external)?;
        to_lua_value(lua, &report)
    })?;

    // alc.card.lineage(query) -> { root, nodes, edges, truncated }
    //
    // Walks `metadata.prior_card_id` ancestors (default), descendants, or
    // both. Relation filter and depth cap are both optional.
    let store_lin = Arc::clone(&card_store);
    let lineage = lua.create_function(move |lua, query: LuaTable| {
        let card_id: String = query.get("card_id")?;
        let direction_str: Option<String> = query.get("direction")?;
        let direction = match direction_str.as_deref() {
            Some(s) => card::LineageDirection::parse(s).map_err(LuaError::external)?,
            None => card::LineageDirection::Up,
        };
        let depth: Option<usize> = query.get("depth")?;
        let include_stats: Option<bool> = query.get("include_stats")?;
        let relation_filter: Option<Vec<String>> = match query.get::<LuaValue>("relation_filter")? {
            LuaValue::Nil => None,
            v => Some(lua.from_value(v)?),
        };

        let q = card::LineageQuery {
            card_id,
            direction,
            depth,
            include_stats: include_stats.unwrap_or(true),
            relation_filter,
        };
        match store_lin.lineage(q).map_err(LuaError::external)? {
            Some(res) => to_lua_value(lua, &card::lineage_to_json(&res)),
            None => Ok(LuaValue::Nil),
        }
    })?;

    card_table.set("create", create)?;
    card_table.set("get", get)?;
    card_table.set("list", list)?;
    card_table.set("append", append)?;
    card_table.set("get_by_alias", get_by_alias)?;
    card_table.set("alias_set", alias_set)?;
    card_table.set("alias_list", alias_list)?;
    card_table.set("find", find)?;
    card_table.set("write_samples", write_samples)?;
    card_table.set("read_samples", read_samples)?;
    card_table.set("lineage", lineage)?;
    card_table.set("sink_backfill", sink_backfill)?;

    alc_table.set("card", card_table)?;
    Ok(())
}

/// Register `alc.stats` table with record/get + auto-counted llm_calls.
///
/// Lua usage:
///   alc.stats.record("accuracy", 0.95)
///   local v = alc.stats.get("accuracy")  -- 0.95
///   local n = alc.stats.llm_calls()      -- session-level cumulative count
///
/// `llm_calls()` reads the engine-maintained `SessionStatus.llm_calls`
/// counter (incremented on every paused-cycle complete in
/// `MetricsObserver`). Recipes / ingredients can compute scoped deltas
/// via `local before = alc.stats.llm_calls(); ... ; local n = alc.stats.llm_calls() - before`
/// without manually tracking calls per branch.
pub(super) fn register_stats(
    lua: &Lua,
    alc_table: &LuaTable,
    custom_metrics: CustomMetricsHandle,
    stats: StatsHandle,
) -> LuaResult<()> {
    let stats_table = lua.create_table()?;

    // alc.stats.record(key, value)
    let cm_record = custom_metrics.clone();
    let record = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        cm_record.record(key, json);
        Ok(())
    })?;

    // alc.stats.get(key)
    let cm_get = custom_metrics;
    let get = lua.create_function(move |lua, key: String| match cm_get.get(&key) {
        Some(v) => to_lua_value(lua, &v),
        None => Ok(LuaValue::Nil),
    })?;

    // alc.stats.llm_calls() — auto-counted session-level LLM call total
    let stats_handle = stats;
    let llm_calls = lua.create_function(move |_, ()| Ok(stats_handle.llm_calls()))?;

    stats_table.set("record", record)?;
    stats_table.set("get", get)?;
    stats_table.set("llm_calls", llm_calls)?;

    alc_table.set("stats", stats_table)?;
    Ok(())
}

// ─── alc.env ─────────────────────────────────────────────────────────────────

/// Read-only Lua UserData view of the frozen env snapshot.
///
/// The underlying `HashMap` is owned by the host (Rust) and wrapped in an
/// `Arc` so it can be shared across the parent session and fork children
/// without copying.  Guest Lua code may only read keys via `alc.env.KEY`
/// (`__index`) or `alc.env:get(key, default)`.  Any write attempt
/// (`alc.env.KEY = value`) returns a hard runtime error — this is the SPACE
/// boundary that keeps host env state immutable from the Lua side.
pub struct AlcEnv(pub Arc<HashMap<String, String>>);

impl mlua::UserData for AlcEnv {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // __index: read a key from the frozen snapshot.
        methods.add_meta_method(mlua::MetaMethod::Index, |_, this, key: String| {
            Ok(this.0.get(&key).cloned())
        });

        // __newindex: hard runtime error on any write attempt.
        // CRUX must_not_simplify: never silently ignore writes.
        methods.add_meta_method(
            mlua::MetaMethod::NewIndex,
            |_, _, (_k, _v): (mlua::Value, mlua::Value)| {
                Err::<(), _>(mlua::Error::external("alc.env is readonly"))
            },
        );

        // get(key [, default]) — explicit lookup with optional fallback.
        methods.add_method(
            "get",
            |_, this, (key, default): (String, Option<String>)| {
                Ok(this.0.get(&key).cloned().or(default))
            },
        );

        // use({key1, key2, ...}) — declare-at-use: returns a plain Lua table
        // containing only the declared keys that exist in the snapshot.
        // Undeclared keys are absent (nil when accessed).
        methods.add_method("use", |lua, this, declared: Vec<String>| {
            let proxy = lua.create_table()?;
            for k in &declared {
                if let Some(v) = this.0.get(k) {
                    proxy.set(k.clone(), v.clone())?;
                }
            }
            Ok(proxy)
        });
    }
}

/// Register `alc.env` on the given `alc` table and store the snapshot as
/// side-band app-data on `lua` so fork children can inherit it via
/// `lua.app_data_ref::<Arc<HashMap<String,String>>>()`.
///
/// This function is intentionally `pub` (not `pub(super)`) because it is
/// re-exported from `bridge::mod.rs` and called from `executor.rs` and
/// `fork.rs` outside this module.
pub fn register_env(
    lua: &mlua::Lua,
    alc_table: &mlua::Table,
    env_map: Arc<HashMap<String, String>>,
) -> mlua::Result<()> {
    alc_table.set("env", AlcEnv(Arc::clone(&env_map)))?;
    lua.set_app_data(env_map);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::ExecutionMetrics;

    /// Build a fresh [`BridgeConfig`] plus its owning state/card
    /// tempdir stores. Returned together so callers can re-use the
    /// store handles (e.g. for assertions / cleanup) after register.
    fn test_config_with(ns: &str) -> crate::bridge::BridgeConfig {
        let metrics = ExecutionMetrics::new();
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        crate::bridge::BridgeConfig {
            llm_tx: None,
            ns: ns.into(),
            custom_metrics: metrics.custom_metrics_handle(),
            stats: metrics.stats_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
            variant_pkgs: vec![],
            state_store: Arc::new(JsonFileStore::new(root.join("state"))),
            card_store: Arc::new(FileCardStore::new(root.join("cards"))),
            card_run_enabled: false,
            scenarios_dir: root.join("scenarios"),
            nn_dir: root.join("nn"),
            log_sink: None,
        }
    }

    fn test_config() -> crate::bridge::BridgeConfig {
        test_config_with("default")
    }

    fn test_config_with_ns(ns: &str) -> crate::bridge::BridgeConfig {
        test_config_with(ns)
    }

    #[test]
    fn json_roundtrip() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(r#"return alc.json_encode({hello = "world", n = 42})"#)
            .eval()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["hello"], "world");
        assert_eq!(parsed["n"], 42);
    }

    #[test]
    fn json_decode_encode() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        let result: String = lua
            .load(
                r#"
                local val = alc.json_decode('{"a":1,"b":"two"}')
                val.c = true
                return alc.json_encode(val)
            "#,
            )
            .eval()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["a"], 1);
        assert_eq!(parsed["b"], "two");
        assert_eq!(parsed["c"], true);
    }

    /// Regression: JSON `null` must decode to Lua `nil` (not an mlua
    /// lightuserdata sentinel that bypasses `if value then ...` truthy checks).
    /// Covers top-level null, nullable object fields, and array elements.
    /// See issue db041966.
    #[test]
    fn json_decode_null_yields_lua_nil() {
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Top-level null: must be Lua nil so `if v then ...` skips the branch.
        let top_level_truthy: bool = lua
            .load(r#"local v = alc.json_decode("null"); return v ~= nil"#)
            .eval()
            .unwrap();
        assert!(
            !top_level_truthy,
            "alc.json_decode(\"null\") should return Lua nil"
        );

        // Object field null: `obj.x` must be nil; `if obj.x then ...` skips.
        let field_truthy: bool = lua
            .load(
                r#"
                local obj = alc.json_decode('{"x": null, "y": 1}')
                return obj.x ~= nil
            "#,
            )
            .eval()
            .unwrap();
        assert!(
            !field_truthy,
            "Object field decoded from JSON null should be Lua nil"
        );

        // Object field null: type must be "nil", not "userdata" (sentinel).
        let field_type: String = lua
            .load(r#"return type(alc.json_decode('{"x": null}').x)"#)
            .eval()
            .unwrap();
        assert_eq!(
            field_type, "nil",
            "type() of null-decoded field must be 'nil', not 'userdata'"
        );

        // Array element null: observe how the table is shaped. Document the
        // resulting `#arr` length so consumers can rely on a stable contract.
        let arr_len: i64 = lua
            .load(r#"return #alc.json_decode('[1, null, 3]')"#)
            .eval()
            .unwrap();
        // mlua/Lua 5.4 length: nil holes inside the array part do not
        // truncate `#arr`; it returns the original JSON array length (3 for
        // `[1, null, 3]`). Indexed access still yields nil at the hole.
        // Consumers iterating with `for i = 1, #arr do ... if arr[i] then` are
        // safe; `ipairs()` will stop at the first nil. Document the contract
        // here so downstream packages can rely on it.
        assert_eq!(
            arr_len, 3,
            "JSON array length is preserved across null elements (mlua/Lua 5.4 array part)"
        );

        // Array element null: explicit indexed access surfaces nil for the
        // hole and the original values at the surrounding indices.
        let (a, b, c): (Option<i64>, Option<i64>, Option<i64>) = lua
            .load(
                r#"
                local arr = alc.json_decode('[1, null, 3]')
                return arr[1], arr[2], arr[3]
            "#,
            )
            .eval()
            .unwrap();
        assert_eq!(a, Some(1));
        assert_eq!(b, None, "Array element decoded from JSON null must be nil");
        assert_eq!(c, Some(3));
    }

    #[test]
    fn state_get_set() {
        // Each BridgeConfig comes with its own tempdir-rooted
        // JsonFileStore so no cross-test cleanup is needed.
        let ns = "_test_bridge_state";

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config_with_ns(ns)).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Set and get
        lua.load(r#"alc.state.set("x", 99)"#).exec().unwrap();
        let result: i64 = lua.load(r#"return alc.state.get("x")"#).eval().unwrap();
        assert_eq!(result, 99);

        // Default value
        let result: i64 = lua
            .load(r#"return alc.state.get("missing", 0)"#)
            .eval()
            .unwrap();
        assert_eq!(result, 0);

        // Nil for missing without default
        let result: LuaValue = lua
            .load(r#"return alc.state.get("missing")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    /// Sibling regression for the `to_lua_value` helper sweep (issue ff6372af):
    /// when state stores a value containing a JSON null field, retrieval must
    /// surface that field as Lua nil (not the mlua lightuserdata sentinel) so
    /// that `if v.x then ...` truthy checks behave correctly. Verifies the
    /// helper's null-handling contract via the `alc.state.get` call path,
    /// representative of the other accessors that funnel through the same
    /// helper (`alc.card.*`, `alc.stats.*`, etc.).
    #[test]
    fn state_get_object_with_null_field_yields_lua_nil() {
        let ns = "_test_bridge_state_null_field";

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config_with_ns(ns)).unwrap();
        lua.globals().set("alc", t).unwrap();

        // Set state to an object containing a null field via JSON decode round-trip.
        lua.load(r#"alc.state.set("obj", alc.json_decode('{"x": null, "y": 1}'))"#)
            .exec()
            .unwrap();

        // Retrieved object's `x` field must be Lua nil (not lightuserdata sentinel).
        let x_truthy: bool = lua
            .load(r#"local v = alc.state.get("obj"); return v.x ~= nil"#)
            .eval()
            .unwrap();
        assert!(
            !x_truthy,
            "state.get returned object's null field must be Lua nil (truthy check must skip)"
        );

        // type() of the null field must be "nil", not "userdata" (sentinel).
        let x_type: String = lua
            .load(r#"local v = alc.state.get("obj"); return type(v.x)"#)
            .eval()
            .unwrap();
        assert_eq!(
            x_type, "nil",
            "type() of state.get'd null field must be 'nil', not 'userdata'"
        );

        // Sibling field `y` should still surface as the original value.
        let y: i64 = lua.load(r#"return alc.state.get("obj").y"#).eval().unwrap();
        assert_eq!(y, 1, "Non-null sibling field must round-trip unchanged");
    }

    #[test]
    fn state_has_set_nx_incr() {
        let ns = "_test_bridge_state_t1";

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config_with_ns(ns)).unwrap();
        lua.globals().set("alc", t).unwrap();

        // has: false for missing key
        let h: bool = lua.load(r#"return alc.state.has("k")"#).eval().unwrap();
        assert!(!h);

        // set_nx: true when absent
        let ok: bool = lua
            .load(r#"return alc.state.set_nx("k", "first")"#)
            .eval()
            .unwrap();
        assert!(ok);

        // has: true after set
        let h: bool = lua.load(r#"return alc.state.has("k")"#).eval().unwrap();
        assert!(h);

        // set_nx: false when present
        let ok: bool = lua
            .load(r#"return alc.state.set_nx("k", "second")"#)
            .eval()
            .unwrap();
        assert!(!ok);

        // incr: init + delta
        let v: f64 = lua
            .load(r#"return alc.state.incr("counter")"#)
            .eval()
            .unwrap();
        assert!((v - 1.0).abs() < f64::EPSILON);

        // incr: with explicit delta
        let v: f64 = lua
            .load(r#"return alc.state.incr("counter", 5)"#)
            .eval()
            .unwrap();
        assert!((v - 6.0).abs() < f64::EPSILON);

        // incr: with custom default (ignored since key exists)
        let v: f64 = lua
            .load(r#"return alc.state.incr("counter", 10, 100)"#)
            .eval()
            .unwrap();
        assert!((v - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn card_create_get_list_from_lua() {
        // Use a unique pkg name per-run to avoid clobbering real cards.
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pkg = format!("_test_bridge_card_{ns}");

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(&lua, &t, test_config()).unwrap();
        lua.globals().set("alc", t).unwrap();

        // create
        let create_script = format!(
            r#"
            local r = alc.card.create({{
                pkg = {{ name = "{pkg}" }},
                model = {{ id = "claude-opus-4-6" }},
                stats = {{ pass_rate = 0.9 }},
            }})
            return r.card_id
        "#
        );
        let card_id: String = lua.load(&create_script).eval().unwrap();
        assert!(card_id.starts_with(&pkg));

        // get
        let get_script = format!(r#"return alc.card.get("{card_id}").stats.pass_rate"#);
        let rate: f64 = lua.load(&get_script).eval().unwrap();
        assert!((rate - 0.9).abs() < 1e-9);

        // list (filtered by pkg)
        let list_script = format!(
            r#"
            local rows = alc.card.list({{ pkg = "{pkg}" }})
            return #rows
        "#
        );
        let count: i64 = lua.load(&list_script).eval().unwrap();
        assert_eq!(count, 1);

        // No cleanup needed: the card_store is tempdir-rooted via test_config().
    }

    #[test]
    fn stats_record_get() {
        let metrics = ExecutionMetrics::new();
        let custom_handle = metrics.custom_metrics_handle();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        crate::bridge::register(
            &lua,
            &t,
            crate::bridge::BridgeConfig {
                llm_tx: None,
                ns: "default".into(),
                custom_metrics: custom_handle.clone(),
                stats: metrics.stats_handle(),
                budget: metrics.budget_handle(),
                progress: metrics.progress_handle(),
                lib_paths: vec![],
                variant_pkgs: vec![],
                state_store: Arc::new(JsonFileStore::new(root.join("state"))),
                card_store: Arc::new(FileCardStore::new(root.join("cards"))),
                card_run_enabled: false,
                scenarios_dir: root.join("scenarios"),
                nn_dir: root.join("nn"),
                log_sink: None,
            },
        )
        .unwrap();
        lua.globals().set("alc", t).unwrap();

        // Record from Lua
        lua.load(r#"alc.stats.record("score", 42)"#).exec().unwrap();
        let result: i64 = lua.load(r#"return alc.stats.get("score")"#).eval().unwrap();
        assert_eq!(result, 42);

        // Verify via Handle
        assert_eq!(custom_handle.get("score"), Some(serde_json::json!(42)));

        // Missing key returns nil
        let result: LuaValue = lua
            .load(r#"return alc.stats.get("missing")"#)
            .eval()
            .unwrap();
        assert!(result.is_nil());
    }

    /// `alc.stats.llm_calls()` reads the engine-maintained
    /// `SessionStatus.llm_calls` counter and returns 0 for a fresh session.
    /// After driving the counter via `MetricsObserver::on_paused`, the
    /// Lua-side function reflects the new value.
    #[test]
    fn stats_llm_calls_reads_session_status() {
        use crate::card::FileCardStore;
        use crate::state::JsonFileStore;
        use algocline_core::{ExecutionObserver, LlmQuery, QueryId};
        use std::sync::Arc;

        let metrics = ExecutionMetrics::new();
        let observer = metrics.create_observer();

        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        let tmp = tempfile::tempdir().expect("test tempdir");
        let root = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        crate::bridge::register(
            &lua,
            &t,
            crate::bridge::BridgeConfig {
                llm_tx: None,
                ns: "default".into(),
                custom_metrics: metrics.custom_metrics_handle(),
                stats: metrics.stats_handle(),
                budget: metrics.budget_handle(),
                progress: metrics.progress_handle(),
                lib_paths: vec![],
                variant_pkgs: vec![],
                state_store: Arc::new(JsonFileStore::new(root.join("state"))),
                card_store: Arc::new(FileCardStore::new(root.join("cards"))),
                card_run_enabled: false,
                scenarios_dir: root.join("scenarios"),
                nn_dir: root.join("nn"),
                log_sink: None,
            },
        )
        .unwrap();
        lua.globals().set("alc", t).unwrap();

        // Initial value: 0
        let initial: u64 = lua.load(r#"return alc.stats.llm_calls()"#).eval().unwrap();
        assert_eq!(initial, 0, "fresh session must report llm_calls() == 0");

        // Drive the observer to simulate a paused-cycle (one LLM call).
        observer.on_paused(&[LlmQuery {
            id: QueryId::parse("q-0"),
            prompt: "hi".to_string(),
            system: None,
            max_tokens: 0,
            grounded: false,
            underspecified: false,
            cache_breakpoint: None,
            role: None,
        }]);

        // Lua side now sees the increment.
        let after_one: u64 = lua.load(r#"return alc.stats.llm_calls()"#).eval().unwrap();
        assert_eq!(
            after_one, 1,
            "one paused query must increment llm_calls() to 1"
        );

        // Two more queries in a single paused-cycle.
        observer.on_paused(&[
            LlmQuery {
                id: QueryId::parse("q-1"),
                prompt: "a".to_string(),
                system: None,
                max_tokens: 0,
                grounded: false,
                underspecified: false,
                cache_breakpoint: None,
                role: None,
            },
            LlmQuery {
                id: QueryId::parse("q-2"),
                prompt: "b".to_string(),
                system: None,
                max_tokens: 0,
                grounded: false,
                underspecified: false,
                cache_breakpoint: None,
                role: None,
            },
        ]);

        let after_three: u64 = lua.load(r#"return alc.stats.llm_calls()"#).eval().unwrap();
        assert_eq!(
            after_three, 3,
            "two further paused queries (multi-query batch) must bring llm_calls() to 3"
        );
    }

    // ─── register_log tests ───

    // T1: alc.log routes entry to LogSink with correct fields
    #[test]
    fn register_log_pushes_to_log_sink() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        // Safety: unwrap is acceptable in test code.
        register_log(&lua, &t, sink.clone()).unwrap();
        lua.globals().set("alc", t).unwrap();

        lua.load(r#"alc.log("info", "hello-from-log")"#)
            .exec()
            // Safety: unwrap in test code — propagates Lua errors as test failure.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, "alc.log");
        assert_eq!(entries[0].level, "info");
        assert_eq!(entries[0].message, "hello-from-log");
    }

    // T2: alc.log with unknown level falls back to "info" entry source
    #[test]
    fn register_log_unknown_level_still_pushes() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        // Safety: unwrap in test code.
        register_log(&lua, &t, sink.clone()).unwrap();
        lua.globals().set("alc", t).unwrap();

        lua.load(r#"alc.log("custom", "edge-case")"#)
            .exec()
            // Safety: unwrap in test code.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, "alc.log");
        // level is passed through verbatim regardless of tracing fallback path
        assert_eq!(entries[0].level, "custom");
        assert_eq!(entries[0].message, "edge-case");
    }

    // T3: alc.log with empty message — edge case, should still push
    #[test]
    fn register_log_empty_message() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        // Safety: unwrap in test code.
        register_log(&lua, &t, sink.clone()).unwrap();
        lua.globals().set("alc", t).unwrap();

        lua.load(r#"alc.log("warn", "")"#)
            .exec()
            // Safety: unwrap in test code.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "");
    }

    // ─── register_print tests ───

    // T1: print() override pushes to LogSink with source alc.lua.print
    #[test]
    fn register_print_pushes_to_log_sink() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        // Safety: unwrap in test code.
        register_print(&lua, sink.clone()).unwrap();

        lua.load(r#"print("hello-print")"#)
            .exec()
            // Safety: unwrap in test code.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, "alc.lua.print");
        assert_eq!(entries[0].level, "info");
        assert_eq!(entries[0].message, "hello-print");
    }

    // T2: print() with multiple arguments joins with tab
    #[test]
    fn register_print_multiple_args_tab_joined() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        // Safety: unwrap in test code.
        register_print(&lua, sink.clone()).unwrap();

        lua.load(r#"print("a", "b", "c")"#)
            .exec()
            // Safety: unwrap in test code.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "a\tb\tc");
    }

    // T3: print() with nil/bool/number args — no panic, correct string coercion
    #[test]
    fn register_print_mixed_value_types() {
        use algocline_core::LogSink;

        let sink = LogSink::new();
        let lua = Lua::new();
        // Safety: unwrap in test code.
        register_print(&lua, sink.clone()).unwrap();

        lua.load(r#"print(nil, true, 42, 3.14)"#)
            .exec()
            // Safety: unwrap in test code.
            .unwrap();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        // nil → "nil", bool → "true", int → "42", float → formatted per Lua convention
        let msg = &entries[0].message;
        assert!(msg.starts_with("nil\ttrue\t42\t"), "got: {msg}");
    }

    // ─── alc.env unit tests ───────────────────────────────────────────────────

    fn make_env_lua(pairs: &[(&str, &str)]) -> (Lua, Arc<HashMap<String, String>>) {
        let mut map = HashMap::new();
        for (k, v) in pairs {
            map.insert(k.to_string(), v.to_string());
        }
        let env_map = Arc::new(map);
        let lua = Lua::new();
        let alc_table = lua.create_table().unwrap();
        register_env(&lua, &alc_table, Arc::clone(&env_map)).unwrap();
        lua.globals().set("alc", alc_table).unwrap();
        (lua, env_map)
    }

    #[test]
    fn env_index_reads_existing_key() {
        let (lua, _) = make_env_lua(&[("FOO", "bar")]);
        let val: Option<String> = lua.load(r#"return alc.env.FOO"#).eval().unwrap();
        assert_eq!(val, Some("bar".to_string()));
    }

    #[test]
    fn env_index_missing_key_returns_nil() {
        let (lua, _) = make_env_lua(&[("FOO", "bar")]);
        let val: LuaValue = lua.load(r#"return alc.env.MISSING"#).eval().unwrap();
        assert!(val.is_nil());
    }

    #[test]
    fn env_newindex_returns_error() {
        let (lua, _) = make_env_lua(&[("FOO", "bar")]);
        let result: Result<(), _> = lua.load(r#"alc.env.FOO = "x""#).exec();
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("alc.env is readonly"),
            "expected readonly error, got: {err}"
        );
    }

    #[test]
    fn env_get_with_default_returns_default_on_miss() {
        let (lua, _) = make_env_lua(&[]);
        let val: Option<String> = lua
            .load(r#"return alc.env:get("MISSING", "fallback")"#)
            .eval()
            .unwrap();
        assert_eq!(val, Some("fallback".to_string()));
    }

    #[test]
    fn env_get_returns_value_when_present() {
        let (lua, _) = make_env_lua(&[("KEY", "val")]);
        let val: Option<String> = lua
            .load(r#"return alc.env:get("KEY", "default")"#)
            .eval()
            .unwrap();
        assert_eq!(val, Some("val".to_string()));
    }

    #[test]
    fn env_use_returns_declared_keys_only() {
        let (lua, _) = make_env_lua(&[("FOO", "foo_val"), ("BAR", "bar_val"), ("SECRET", "s")]);
        let result: LuaValue = lua
            .load(
                r#"
                local e = alc.env:use{"FOO", "BAR"}
                return e
            "#,
            )
            .eval()
            .unwrap();
        let tbl = result.as_table().unwrap();
        assert_eq!(tbl.get::<String>("FOO").unwrap(), "foo_val");
        assert_eq!(tbl.get::<String>("BAR").unwrap(), "bar_val");
        // SECRET was not declared — should be absent (nil)
        let secret: LuaValue = tbl.get("SECRET").unwrap();
        assert!(secret.is_nil(), "SECRET should be nil in proxy");
    }

    #[test]
    fn env_use_undeclared_key_is_nil() {
        let (lua, _) = make_env_lua(&[("FOO", "foo_val")]);
        let val: LuaValue = lua
            .load(
                r#"
                local e = alc.env:use{"FOO"}
                return e.UNDECLARED
            "#,
            )
            .eval()
            .unwrap();
        assert!(val.is_nil());
    }

    #[test]
    fn register_env_sets_app_data() {
        let mut map = HashMap::new();
        map.insert("X".to_string(), "1".to_string());
        let env_map = Arc::new(map);
        let lua = Lua::new();
        let alc_table = lua.create_table().unwrap();
        register_env(&lua, &alc_table, Arc::clone(&env_map)).unwrap();
        // Verify app_data is set and accessible
        let retrieved = lua.app_data_ref::<Arc<HashMap<String, String>>>().unwrap();
        assert_eq!(retrieved.get("X").unwrap(), "1");
    }

    mod state_dispatched_lua {
        use super::*;
        use mlua::Lua;
        use std::sync::Arc;
        use tempfile::TempDir;

        fn setup() -> (Lua, Arc<JsonFileStore>, TempDir) {
            let tmp = tempfile::tempdir().unwrap();
            let store = Arc::new(JsonFileStore::new(tmp.path().to_path_buf()));
            let lua = Lua::new();
            let alc = lua.create_table().unwrap();
            register_state(&lua, &alc, "default".to_string(), Arc::clone(&store)).unwrap();
            lua.globals().set("alc", alc).unwrap();
            (lua, store, tmp)
        }

        #[test]
        fn list_returns_sorted_keys() {
            let (lua, _store, tmp) = setup();
            // Seed two files directly into the dispatched layout.
            std::fs::create_dir_all(tmp.path().join("testns")).unwrap();
            std::fs::write(
                tmp.path().join("testns/beta.json"),
                r#"{"data": {"completed_steps": [], "x": 1}}"#,
            )
            .unwrap();
            std::fs::write(
                tmp.path().join("testns/alpha.json"),
                r#"{"data": {"completed_steps": [], "y": 2}}"#,
            )
            .unwrap();
            lua.load(
                r#"
                    local result = alc.state.list("testns")
                    assert(#result == 2, "expected 2 keys, got " .. #result)
                    assert(result[1] == "alpha", "first key should be alpha, got " .. tostring(result[1]))
                    assert(result[2] == "beta", "second key should be beta, got " .. tostring(result[2]))
                "#,
            )
            .exec()
            .unwrap();
        }

        #[test]
        fn show_returns_full_table() {
            let (lua, _store, tmp) = setup();
            std::fs::create_dir_all(tmp.path().join("testns")).unwrap();
            std::fs::write(
                tmp.path().join("testns/alpha.json"),
                r#"{"data": {"completed_steps": ["a", "b", "c"], "x": 1, "y": 2}}"#,
            )
            .unwrap();
            lua.load(
                r#"
                    local result = alc.state.show("testns", "alpha")
                    assert(type(result) == "table", "expected table")
                    assert(type(result.data) == "table", "expected result.data to be a table")
                    assert(result.data.x == 1, "expected x=1")
                    assert(result.data.y == 2, "expected y=2")
                    assert(#result.data.completed_steps == 3, "expected 3 steps")
                "#,
            )
            .exec()
            .unwrap();
        }

        #[test]
        fn show_missing_returns_not_found_error() {
            let (lua, _store, _tmp) = setup();
            lua.load(
                r#"
                    local ok, err = pcall(alc.state.show, "testns", "missing")
                    assert(not ok, "expected error but got success")
                    local msg = tostring(err)
                    assert(string.find(msg, "not found"), "error message should contain 'not found', got: " .. msg)
                "#,
            )
            .exec()
            .unwrap();
        }

        #[test]
        fn reset_removes_steps_and_fields_with_backup() {
            let (lua, _store, tmp) = setup();
            std::fs::create_dir_all(tmp.path().join("testns")).unwrap();
            let file_path = tmp.path().join("testns/alpha.json");
            std::fs::write(
                &file_path,
                r#"{"data": {"completed_steps": ["a", "b", "c"], "x": 1, "y": 2}}"#,
            )
            .unwrap();
            // Store tmp path as a string for Lua assertions.
            let tmp_path_str = tmp.path().to_string_lossy().to_string();
            lua.globals().set("TMP_PATH", tmp_path_str.clone()).unwrap();
            lua.load(
                r#"
                    local r = alc.state.reset("testns", "alpha", {steps={"b"}, fields={"x"}})
                    assert(r.ok == true, "expected ok=true")
                    assert(type(r.backup_path) == "string", "backup_path should be a string")
                    assert(r.steps_removed == 1, "expected steps_removed=1, got " .. tostring(r.steps_removed))
                    assert(r.fields_removed == 1, "expected fields_removed=1, got " .. tostring(r.fields_removed))
                "#,
            )
            .exec()
            .unwrap();
            // Assert .bak exists with original content.
            let bak_path = tmp.path().join("testns/alpha.json.bak");
            assert!(
                bak_path.exists(),
                "backup file should exist at {:?}",
                bak_path
            );
            let bak_content = std::fs::read_to_string(&bak_path).unwrap();
            assert!(
                bak_content.contains("\"b\""),
                "backup should contain original 'b' step"
            );
            // Assert live file was mutated: "b" removed from steps, "x" removed from data.
            let live_content = std::fs::read_to_string(&file_path).unwrap();
            let live: serde_json::Value = serde_json::from_str(&live_content).unwrap();
            let steps = live["data"]["completed_steps"].as_array().unwrap();
            assert!(
                !steps.iter().any(|s| s.as_str() == Some("b")),
                "step 'b' should be removed from completed_steps"
            );
            assert!(
                live["data"]["x"].is_null() || live["data"].get("x").is_none(),
                "field 'x' should be removed from data"
            );
        }

        #[test]
        fn unsafe_namespace_rejected() {
            let (lua, _store, _tmp) = setup();
            lua.load(
                r#"
                    local ok, err = pcall(alc.state.list, "../evil")
                    assert(not ok, "expected error for unsafe namespace")
                    local msg = tostring(err)
                    assert(string.find(msg, "unsafe"), "error should contain 'unsafe', got: " .. msg)
                "#,
            )
            .exec()
            .unwrap();
            lua.load(
                r#"
                    local ok, err = pcall(alc.state.show, "../evil", "key")
                    assert(not ok, "expected error for unsafe namespace in show")
                    local msg = tostring(err)
                    assert(string.find(msg, "unsafe"), "error should contain 'unsafe', got: " .. msg)
                "#,
            )
            .exec()
            .unwrap();
            lua.load(
                r#"
                    local ok, err = pcall(alc.state.reset, "../evil", "key", {})
                    assert(not ok, "expected error for unsafe namespace in reset")
                    local msg = tostring(err)
                    assert(string.find(msg, "unsafe"), "error should contain 'unsafe', got: " .. msg)
                "#,
            )
            .exec()
            .unwrap();
        }
    }
}
