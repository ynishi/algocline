use algocline_core::CustomMetricsHandle;
use mlua::prelude::*;
use mlua::LuaSerdeExt;

use crate::card;
use crate::state;

pub(super) fn register_json(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let encode = lua.create_function(|lua, value: LuaValue| {
        let json: serde_json::Value = lua.from_value(value)?;
        serde_json::to_string(&json).map_err(LuaError::external)
    })?;

    let decode = lua.create_function(|lua, s: String| {
        let value: serde_json::Value = serde_json::from_str(&s).map_err(LuaError::external)?;
        lua.to_value(&value)
    })?;

    alc_table.set("json_encode", encode)?;
    alc_table.set("json_decode", decode)?;
    Ok(())
}

pub(super) fn register_log(_lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let log = _lua.create_function(|_, (level, msg): (String, String)| {
        match level.as_str() {
            "error" => tracing::error!("{}", msg),
            "warn" => tracing::warn!("{}", msg),
            "info" => tracing::info!("{}", msg),
            "debug" => tracing::debug!("{}", msg),
            _ => tracing::info!("{}", msg),
        }
        Ok(())
    })?;

    alc_table.set("log", log)?;
    Ok(())
}

/// Register `alc.state` table with get/set/keys/delete.
///
/// Lua usage:
///   alc.state.set("score", 42)
///   local v = alc.state.get("score")       -- 42
///   local v = alc.state.get("missing", 0)  -- 0 (default)
///   local k = alc.state.keys()             -- {"score"}
///   alc.state.delete("score")
pub(super) fn register_state(lua: &Lua, alc_table: &LuaTable, ns: String) -> LuaResult<()> {
    let state_table = lua.create_table()?;

    // alc.state.get(key, default?)
    let ns_get = ns.clone();
    let get =
        lua.create_function(
            move |lua, (key, default): (String, Option<LuaValue>)| match state::get(&ns_get, &key) {
                Ok(Some(v)) => lua.to_value(&v),
                Ok(None) => Ok(default.unwrap_or(LuaValue::Nil)),
                Err(e) => Err(LuaError::external(e)),
            },
        )?;

    // alc.state.set(key, value)
    let ns_set = ns.clone();
    let set = lua.create_function(move |lua, (key, value): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(value)?;
        state::set(&ns_set, &key, json).map_err(LuaError::external)
    })?;

    // alc.state.keys()
    let ns_keys = ns.clone();
    let keys = lua.create_function(move |lua, ()| {
        let k = state::keys(&ns_keys).map_err(LuaError::external)?;
        lua.to_value(&k)
    })?;

    // alc.state.delete(key)
    let ns_del = ns.clone();
    let delete = lua.create_function(move |_, key: String| {
        state::delete(&ns_del, &key).map_err(LuaError::external)
    })?;

    state_table.set("get", get)?;
    state_table.set("set", set)?;
    state_table.set("keys", keys)?;
    state_table.set("delete", delete)?;

    alc_table.set("state", state_table)?;
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
pub(super) fn register_card(lua: &Lua, alc_table: &LuaTable) -> LuaResult<()> {
    let card_table = lua.create_table()?;

    // alc.card.create(table) -> { card_id, path }
    let create = lua.create_function(|lua, input: LuaValue| {
        let json: serde_json::Value = lua.from_value(input)?;
        let (card_id, path) = card::create(json).map_err(LuaError::external)?;
        let ret = lua.create_table()?;
        ret.set("card_id", card_id)?;
        ret.set("path", path.to_string_lossy().to_string())?;
        Ok(ret)
    })?;

    // alc.card.get(card_id) -> table | nil
    let get = lua.create_function(|lua, card_id: String| match card::get(&card_id) {
        Ok(Some(v)) => lua.to_value(&v),
        Ok(None) => Ok(LuaValue::Nil),
        Err(e) => Err(LuaError::external(e)),
    })?;

    // alc.card.list(filter?) -> [summary]
    let list = lua.create_function(|lua, filter: Option<LuaTable>| {
        let pkg = match filter {
            Some(t) => t.get::<Option<String>>("pkg")?,
            None => None,
        };
        let rows = card::list(pkg.as_deref()).map_err(LuaError::external)?;
        lua.to_value(&card::summaries_to_json(&rows))
    })?;

    // alc.card.append(card_id, fields) -> merged_card
    let append = lua.create_function(|lua, (card_id, fields): (String, LuaValue)| {
        let json: serde_json::Value = lua.from_value(fields)?;
        let merged = card::append(&card_id, json).map_err(LuaError::external)?;
        lua.to_value(&merged)
    })?;

    // alc.card.get_by_alias(name) -> table | nil
    let get_by_alias = lua.create_function(|lua, name: String| {
        match card::get_by_alias(&name).map_err(LuaError::external)? {
            Some(v) => lua.to_value(&v),
            None => Ok(LuaValue::Nil),
        }
    })?;

    // alc.card.alias_set(name, card_id, opts?) -> alias
    let alias_set = lua.create_function(
        |lua, (name, card_id, opts): (String, String, Option<LuaTable>)| {
            let (pkg, note) = match opts {
                Some(t) => (
                    t.get::<Option<String>>("pkg")?,
                    t.get::<Option<String>>("note")?,
                ),
                None => (None, None),
            };
            let a = card::alias_set(&name, &card_id, pkg.as_deref(), note.as_deref())
                .map_err(LuaError::external)?;
            let arr = card::aliases_to_json(&[a]);
            let first = match arr {
                serde_json::Value::Array(mut v) if !v.is_empty() => v.remove(0),
                other => other,
            };
            lua.to_value(&first)
        },
    )?;

    // alc.card.alias_list(filter?) -> [alias]
    let alias_list = lua.create_function(|lua, filter: Option<LuaTable>| {
        let pkg = match filter {
            Some(t) => t.get::<Option<String>>("pkg")?,
            None => None,
        };
        let rows = card::alias_list(pkg.as_deref()).map_err(LuaError::external)?;
        lua.to_value(&card::aliases_to_json(&rows))
    })?;

    // alc.card.find(query?) -> [summary]
    //
    // Accepts a Prisma-style `where` DSL + dotted-path `order_by`.
    // See `card::parse_where` / `card::parse_order_by` for semantics.
    let find = lua.create_function(|lua, query: Option<LuaTable>| {
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
        let rows = card::find(q).map_err(LuaError::external)?;
        lua.to_value(&card::summaries_to_json(&rows))
    })?;

    // alc.card.write_samples(card_id, samples) -> { path, count }
    let write_samples = lua.create_function(|lua, (card_id, samples): (String, LuaValue)| {
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
        let path = card::write_samples(&card_id, arr).map_err(LuaError::external)?;
        let ret = lua.create_table()?;
        ret.set("path", path.to_string_lossy().to_string())?;
        ret.set("count", count)?;
        Ok(ret)
    })?;

    // alc.card.read_samples(card_id, opts?) -> [sample]
    //
    // opts.where applies the Prisma-style DSL to each row; offset/limit
    // page the post-filter stream. See `card::parse_where`.
    let read_samples =
        lua.create_function(|lua, (card_id, opts): (String, Option<LuaTable>)| {
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
            let rows = card::read_samples(&card_id, q).map_err(LuaError::external)?;
            lua.to_value(&serde_json::Value::Array(rows))
        })?;

    // alc.card.lineage(query) -> { root, nodes, edges, truncated }
    //
    // Walks `metadata.prior_card_id` ancestors (default), descendants, or
    // both. Relation filter and depth cap are both optional.
    let lineage = lua.create_function(|lua, query: LuaTable| {
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
        match card::lineage(q).map_err(LuaError::external)? {
            Some(res) => lua.to_value(&card::lineage_to_json(&res)),
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

    alc_table.set("card", card_table)?;
    Ok(())
}

/// Register `alc.stats` table with record/get.
///
/// Lua usage:
///   alc.stats.record("accuracy", 0.95)
///   local v = alc.stats.get("accuracy")  -- 0.95
pub(super) fn register_stats(
    lua: &Lua,
    alc_table: &LuaTable,
    custom_metrics: CustomMetricsHandle,
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
        Some(v) => lua.to_value(&v),
        None => Ok(LuaValue::Nil),
    })?;

    stats_table.set("record", record)?;
    stats_table.set("get", get)?;

    alc_table.set("stats", stats_table)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_core::ExecutionMetrics;

    fn test_config() -> crate::bridge::BridgeConfig {
        let metrics = ExecutionMetrics::new();
        crate::bridge::BridgeConfig {
            llm_tx: None,
            ns: "default".into(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
        }
    }

    fn test_config_with_ns(ns: &str) -> crate::bridge::BridgeConfig {
        let metrics = ExecutionMetrics::new();
        crate::bridge::BridgeConfig {
            llm_tx: None,
            ns: ns.into(),
            custom_metrics: metrics.custom_metrics_handle(),
            budget: metrics.budget_handle(),
            progress: metrics.progress_handle(),
            lib_paths: vec![],
        }
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

    #[test]
    fn state_get_set() {
        let ns = "_test_bridge_state";
        // Clean up
        let _ = crate::state::delete(ns, "x");

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

        // Clean up
        let _ = crate::state::delete(ns, "x");
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

        // Cleanup
        if let Some(home) = dirs::home_dir() {
            let _ = std::fs::remove_dir_all(home.join(".algocline").join("cards").join(&pkg));
        }
    }

    #[test]
    fn stats_record_get() {
        let metrics = ExecutionMetrics::new();
        let custom_handle = metrics.custom_metrics_handle();
        let lua = Lua::new();
        let t = lua.create_table().unwrap();
        crate::bridge::register(
            &lua,
            &t,
            crate::bridge::BridgeConfig {
                llm_tx: None,
                ns: "default".into(),
                custom_metrics: custom_handle.clone(),
                budget: metrics.budget_handle(),
                progress: metrics.progress_handle(),
                lib_paths: vec![],
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
}
