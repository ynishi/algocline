//! Card storage — immutable run-result snapshots.
//!
//! Storage: ~/.algocline/cards/{pkg}/{card_id}.toml
//!
//! v0 schema (card_schema_v0_draft.md):
//!   REQUIRED fields: schema_version, card_id, created_at, [pkg].name
//!   Everything else is OPTIONAL and auto-injected where possible.
//!
//! v0 P0 API (exposed to Lua as alc.card.*):
//!   create(table) -> { card_id, path }   — write new Card
//!   get(card_id)  -> table | nil          — read Card by id
//!   list(filter?) -> [summary]            — list Cards, optionally filtered by pkg

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value as Json};

pub const SCHEMA_VERSION: &str = "card/v0";

/// Resolve the cards root directory, creating it if needed.
fn cards_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let dir = home.join(".algocline").join("cards");
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("Failed to create cards dir: {e}"))?;
    }
    Ok(dir)
}

/// Per-pkg subdirectory. Validates pkg name to prevent path traversal.
fn pkg_dir(pkg: &str) -> Result<PathBuf, String> {
    validate_name(pkg, "pkg")?;
    let dir = cards_dir()?.join(pkg);
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("Failed to create pkg dir: {e}"))?;
    }
    Ok(dir)
}

fn validate_name(name: &str, kind: &str) -> Result<(), String> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains('\0')
    {
        return Err(format!("Invalid {kind} name: '{name}'"));
    }
    Ok(())
}

/// DJB2 hash, hex-encoded. Used for param_fingerprint and card_id hash segment.
fn djb2_hex(s: &str) -> String {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{h:016x}")
}

/// Short-hash: first 6 hex chars of djb2.
fn hash6(s: &str) -> String {
    djb2_hex(s).chars().take(6).collect()
}

/// Stable serialization of a JSON value for hashing (sorted keys).
fn stable_json(v: &Json) -> String {
    let mut buf = String::new();
    stable_json_into(v, &mut buf);
    buf
}
fn stable_json_into(v: &Json, buf: &mut String) {
    match v {
        Json::Null => buf.push_str("null"),
        Json::Bool(b) => buf.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => buf.push_str(&n.to_string()),
        Json::String(s) => {
            buf.push('"');
            buf.push_str(s);
            buf.push('"');
        }
        Json::Array(a) => {
            buf.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                stable_json_into(item, buf);
            }
            buf.push(']');
        }
        Json::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            buf.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                buf.push('"');
                buf.push_str(k);
                buf.push_str("\":");
                stable_json_into(&m[*k], buf);
            }
            buf.push('}');
        }
    }
}

/// Derive a short model id (e.g. "claude-opus-4-6" -> "opus46").
/// v0: best-effort. Falls back to "model" if input is empty.
fn short_model(id: &str) -> String {
    if id.is_empty() {
        return "model".into();
    }
    // Strip common vendor prefixes.
    let stripped = id
        .strip_prefix("claude-")
        .or_else(|| id.strip_prefix("gpt-"))
        .unwrap_or(id);
    // Keep alnum only.
    let s: String = stripped
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    if s.is_empty() {
        "model".into()
    } else {
        s
    }
}

/// RFC3339 UTC "YYYY-MM-DDTHH:MM:SSZ" from current system time.
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (y, mo, d) = civil_from_days(days);
    let hh = tod / 3600;
    let mm = (tod % 3600) / 60;
    let ss = tod % 60;
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// YYYYMMDD for current UTC date.
fn today_compact() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (y, mo, d) = civil_from_days(secs.div_euclid(86400));
    format!("{y:04}{mo:02}{d:02}")
}

/// Howard Hinnant's civil_from_days algorithm.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = y + if m <= 2 { 1 } else { 0 };
    (y as i32, m, d)
}

/// Converter: serde_json::Value -> toml::Value.
/// Nulls are dropped (TOML has no null). Mixed-type arrays are allowed in TOML 1.0+.
fn json_to_toml(v: Json) -> Result<toml::Value, String> {
    Ok(match v {
        Json::Null => return Err("TOML does not support null values".into()),
        Json::Bool(b) => toml::Value::Boolean(b),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                toml::Value::Float(f)
            } else {
                return Err(format!("Unsupported number: {n}"));
            }
        }
        Json::String(s) => toml::Value::String(s),
        Json::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for item in a {
                if !item.is_null() {
                    out.push(json_to_toml(item)?);
                }
            }
            toml::Value::Array(out)
        }
        Json::Object(m) => {
            let mut table = toml::map::Map::new();
            for (k, val) in m {
                if val.is_null() {
                    continue;
                }
                table.insert(k, json_to_toml(val)?);
            }
            toml::Value::Table(table)
        }
    })
}

/// Converter: toml::Value -> serde_json::Value (for alc.card.get()).
fn toml_to_json(v: toml::Value) -> Json {
    match v {
        toml::Value::String(s) => Json::String(s),
        toml::Value::Integer(i) => json!(i),
        toml::Value::Float(f) => json!(f),
        toml::Value::Boolean(b) => Json::Bool(b),
        toml::Value::Datetime(dt) => Json::String(dt.to_string()),
        toml::Value::Array(a) => Json::Array(a.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => {
            let mut m = serde_json::Map::new();
            for (k, v) in t {
                m.insert(k, toml_to_json(v));
            }
            Json::Object(m)
        }
    }
}

/// Extract [pkg].name from an input JSON object. REQUIRED.
fn require_pkg_name(input: &Json) -> Result<String, String> {
    let name = input
        .get("pkg")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .ok_or_else(|| "alc.card.create: pkg.name is required".to_string())?
        .to_string();
    validate_name(&name, "pkg")?;
    Ok(name)
}

/// Main create entry. Returns (card_id, absolute_path).
pub fn create(mut input: Json) -> Result<(String, PathBuf), String> {
    if !input.is_object() {
        return Err("alc.card.create: input must be a table".into());
    }
    let pkg_name = require_pkg_name(&input)?;
    let obj = input.as_object_mut().unwrap();

    // ─── Auto-inject REQUIRED fields ──────────────────────────
    obj.entry("schema_version".to_string())
        .or_insert_with(|| json!(SCHEMA_VERSION));
    obj.entry("created_at".to_string())
        .or_insert_with(|| json!(now_rfc3339()));
    obj.entry("created_by".to_string())
        .or_insert_with(|| json!(format!("alc@{}", env!("CARGO_PKG_VERSION"))));

    // ─── param_fingerprint (if [params] present) ──────────────
    if let Some(params) = obj.get("params").cloned() {
        if params.is_object() {
            let fp = djb2_hex(&stable_json(&params));
            obj.insert("param_fingerprint".to_string(), json!(fp));
        }
    }

    // ─── card_id generation (if absent) ───────────────────────
    let card_id = match obj.get("card_id").and_then(|v| v.as_str()) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            let model_id = obj
                .get("model")
                .and_then(|m| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let model_short = short_model(model_id);
            let date = today_compact();
            let fp_seed = stable_json(&Json::Object(obj.clone()));
            let h = hash6(&fp_seed);
            format!("{pkg_name}_{model_short}_{date}_{h}")
        }
    };
    validate_name(&card_id, "card_id")?;
    obj.insert("card_id".to_string(), json!(card_id.clone()));

    // ─── Write TOML atomically ────────────────────────────────
    let dir = pkg_dir(&pkg_name)?;
    let path = dir.join(format!("{card_id}.toml"));
    if path.exists() {
        return Err(format!(
            "alc.card.create: card '{card_id}' already exists (immutable)"
        ));
    }
    let toml_val = json_to_toml(input)?;
    let text = toml::to_string_pretty(&toml_val)
        .map_err(|e| format!("Failed to serialize card TOML: {e}"))?;
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, &text).map_err(|e| format!("Failed to write card tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename card file: {e}"))?;

    Ok((card_id, path))
}

/// Search cards dir for `{card_id}.toml`.
fn find_card_path(card_id: &str) -> Result<Option<PathBuf>, String> {
    validate_name(card_id, "card_id")?;
    let root = cards_dir()?;
    let entries = fs::read_dir(&root)
        .map_err(|e| format!("Failed to read cards dir: {e}"))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let candidate = p.join(format!("{card_id}.toml"));
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

/// Read a Card by id. Returns None if not found.
pub fn get(card_id: &str) -> Result<Option<Json>, String> {
    let path = match find_card_path(card_id)? {
        Some(p) => p,
        None => return Ok(None),
    };
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read card '{card_id}': {e}"))?;
    let val: toml::Value = toml::from_str(&text)
        .map_err(|e| format!("Failed to parse card '{card_id}': {e}"))?;
    Ok(Some(toml_to_json(val)))
}

/// Summary row for `alc.card.list()`.
#[derive(Debug, Clone)]
pub struct Summary {
    pub card_id: String,
    pub pkg: String,
    pub created_at: Option<String>,
    pub model: Option<String>,
    pub scenario: Option<String>,
    pub pass_rate: Option<f64>,
}

impl Summary {
    fn to_json(&self) -> Json {
        let mut m = serde_json::Map::new();
        m.insert("card_id".into(), json!(self.card_id));
        m.insert("pkg".into(), json!(self.pkg));
        if let Some(v) = &self.created_at {
            m.insert("created_at".into(), json!(v));
        }
        if let Some(v) = &self.model {
            m.insert("model".into(), json!(v));
        }
        if let Some(v) = &self.scenario {
            m.insert("scenario".into(), json!(v));
        }
        if let Some(v) = self.pass_rate {
            m.insert("pass_rate".into(), json!(v));
        }
        Json::Object(m)
    }
}

fn summarize(path: &std::path::Path, pkg: &str) -> Option<Summary> {
    let text = fs::read_to_string(path).ok()?;
    let val: toml::Value = toml::from_str(&text).ok()?;
    let card_id = val
        .get("card_id")
        .and_then(|v| v.as_str())
        .or_else(|| path.file_stem().and_then(|s| s.to_str()))?
        .to_string();
    let created_at = val
        .get("created_at")
        .and_then(|v| v.as_str())
        .map(String::from);
    let model = val
        .get("model")
        .and_then(|m| m.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let scenario = val
        .get("scenario")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let pass_rate = val
        .get("stats")
        .and_then(|s| s.get("pass_rate"))
        .and_then(|v| v.as_float());
    Some(Summary {
        card_id,
        pkg: pkg.to_string(),
        created_at,
        model,
        scenario,
        pass_rate,
    })
}

/// List cards. `pkg_filter = Some("name")` restricts to that pkg subdir.
pub fn list(pkg_filter: Option<&str>) -> Result<Vec<Summary>, String> {
    let root = cards_dir()?;
    let mut out = Vec::new();

    let pkg_dirs: Vec<PathBuf> = if let Some(p) = pkg_filter {
        validate_name(p, "pkg")?;
        let d = root.join(p);
        if d.is_dir() {
            vec![d]
        } else {
            vec![]
        }
    } else {
        fs::read_dir(&root)
            .map_err(|e| format!("Failed to read cards dir: {e}"))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect()
    };

    for pdir in pkg_dirs {
        let pkg = pdir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let entries = match fs::read_dir(&pdir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("toml") {
                continue;
            }
            if let Some(s) = summarize(&p, &pkg) {
                out.push(s);
            }
        }
    }

    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

pub fn summaries_to_json(rows: &[Summary]) -> Json {
    Json::Array(rows.iter().map(|s| s.to_json()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_pkg() -> String {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("_test_card_{ns}")
    }

    fn cleanup(pkg: &str) {
        if let Ok(d) = pkg_dir(pkg) {
            let _ = fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn minimum_valid_card() {
        let pkg = unique_pkg();
        let input = json!({ "pkg": { "name": pkg } });
        let (id, path) = create(input).unwrap();
        assert!(path.exists());
        assert!(id.starts_with(&pkg));

        let got = get(&id).unwrap().unwrap();
        assert_eq!(got["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(got["card_id"], json!(id));
        assert_eq!(got["pkg"]["name"], json!(pkg));
        assert!(got.get("created_at").is_some());
        assert!(got.get("created_by").is_some());

        cleanup(&pkg);
    }

    #[test]
    fn create_rejects_missing_pkg_name() {
        let err = create(json!({})).unwrap_err();
        assert!(err.contains("pkg.name"));
    }

    #[test]
    fn create_is_immutable() {
        let pkg = unique_pkg();
        let input = json!({
            "card_id": "fixed_id_001",
            "pkg": { "name": pkg }
        });
        create(input.clone()).unwrap();
        let err = create(input).unwrap_err();
        assert!(err.contains("already exists"));
        cleanup(&pkg);
    }

    #[test]
    fn create_injects_param_fingerprint() {
        let pkg = unique_pkg();
        let input = json!({
            "pkg": { "name": pkg },
            "params": { "depth": 3, "temperature": 0.0 }
        });
        let (id, _) = create(input).unwrap();
        let got = get(&id).unwrap().unwrap();
        assert!(got["param_fingerprint"].is_string());
        cleanup(&pkg);
    }

    #[test]
    fn list_returns_newest_first() {
        let pkg = unique_pkg();
        // First card
        let (id1, _) = create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
            "created_at": "2025-01-01T00:00:00Z"
        }))
        .unwrap();
        let (id2, _) = create(json!({
            "card_id": format!("{pkg}_b"),
            "pkg": { "name": pkg },
            "created_at": "2026-01-01T00:00:00Z"
        }))
        .unwrap();

        let rows = list(Some(&pkg)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].card_id, id2); // newer first
        assert_eq!(rows[1].card_id, id1);

        cleanup(&pkg);
    }

    #[test]
    fn list_extracts_summary_fields() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({
            "pkg": { "name": pkg },
            "model": { "id": "claude-opus-4-6" },
            "scenario": { "name": "gsm8k_sample100" },
            "stats": { "pass_rate": 0.82 }
        }))
        .unwrap();

        let rows = list(Some(&pkg)).unwrap();
        let row = rows.iter().find(|r| r.card_id == id).unwrap();
        assert_eq!(row.model.as_deref(), Some("claude-opus-4-6"));
        assert_eq!(row.scenario.as_deref(), Some("gsm8k_sample100"));
        assert_eq!(row.pass_rate, Some(0.82));

        cleanup(&pkg);
    }

    #[test]
    fn get_missing_returns_none() {
        assert!(get("does_not_exist_xyz").unwrap().is_none());
    }

    #[test]
    fn short_model_variants() {
        assert_eq!(short_model("claude-opus-4-6"), "opus46");
        assert_eq!(short_model("gpt-4o"), "4o");
        assert_eq!(short_model(""), "model");
    }
}
