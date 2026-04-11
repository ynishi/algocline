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

/// Short-hash: last 6 hex chars of djb2.
///
/// DJB2's high bits are dominated by the `5381 * 33^n` term (same for any
/// input of equal length), so the top hex digits collide easily for same-
/// length inputs that differ only in a few byte positions. The low bits,
/// driven by the most-recent bytes, mix well enough for short-hash use.
fn hash6(s: &str) -> String {
    let hex = djb2_hex(s);
    let start = hex.len().saturating_sub(6);
    hex[start..].to_string()
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

/// YYYYMMDDTHHMMSS for current UTC time (compact, sortable).
///
/// Used in card_id generation so that:
///   - rapid successive runs don't collide on the hash6 segment
///   - string sort of card_id = chronological order
fn now_compact() -> String {
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
    format!("{y:04}{mo:02}{d:02}T{hh:02}{mm:02}{ss:02}")
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
            let ts = now_compact();
            let fp_seed = stable_json(&Json::Object(obj.clone()));
            let h = hash6(&fp_seed);
            format!("{pkg_name}_{model_short}_{ts}_{h}")
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

    // Sort newest first. card_id embeds a compact UTC timestamp so it's
    // naturally chronological; we still prefer created_at when present
    // (some callers may override it), falling back to card_id.
    out.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| b.card_id.cmp(&a.card_id))
    });
    Ok(out)
}

pub fn summaries_to_json(rows: &[Summary]) -> Json {
    Json::Array(rows.iter().map(|s| s.to_json()).collect())
}

// ───────────────────────────────────────────────────────────────
// P1 API: append / alias_{set,list} / find
// ───────────────────────────────────────────────────────────────

/// Append new top-level fields to an existing Card.
///
/// Semantics: **additive only**. If any top-level key in `fields` already
/// exists in the Card, the call fails — Cards are immutable w.r.t. existing
/// data. New top-level keys are inserted and the Card file is rewritten
/// atomically.
///
/// Returns the merged Card JSON.
pub fn append(card_id: &str, fields: Json) -> Result<Json, String> {
    let path = find_card_path(card_id)?
        .ok_or_else(|| format!("alc.card.append: card '{card_id}' not found"))?;
    let fields_obj = match fields {
        Json::Object(m) => m,
        _ => return Err("alc.card.append: fields must be a table".into()),
    };

    let text = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read card '{card_id}': {e}"))?;
    let existing: toml::Value = toml::from_str(&text)
        .map_err(|e| format!("Failed to parse card '{card_id}': {e}"))?;
    let mut existing_json = toml_to_json(existing);
    let existing_obj = existing_json
        .as_object_mut()
        .ok_or_else(|| format!("Card '{card_id}' is not a table"))?;

    for (k, v) in fields_obj {
        if existing_obj.contains_key(&k) {
            return Err(format!(
                "alc.card.append: key '{k}' already set on card '{card_id}' (immutable)"
            ));
        }
        if !v.is_null() {
            existing_obj.insert(k, v);
        }
    }

    let toml_val = json_to_toml(existing_json.clone())?;
    let text = toml::to_string_pretty(&toml_val)
        .map_err(|e| format!("Failed to serialize card TOML: {e}"))?;
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, &text).map_err(|e| format!("Failed to write card tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename card file: {e}"))?;

    Ok(existing_json)
}

/// Path of the global alias table: `~/.algocline/cards/_aliases.toml`.
fn aliases_path() -> Result<PathBuf, String> {
    Ok(cards_dir()?.join("_aliases.toml"))
}

#[derive(Debug, Clone)]
pub struct Alias {
    pub name: String,
    pub card_id: String,
    pub pkg: Option<String>,
    pub set_at: String,
    pub note: Option<String>,
}

impl Alias {
    fn to_json(&self) -> Json {
        let mut m = serde_json::Map::new();
        m.insert("name".into(), json!(self.name));
        m.insert("card_id".into(), json!(self.card_id));
        if let Some(p) = &self.pkg {
            m.insert("pkg".into(), json!(p));
        }
        m.insert("set_at".into(), json!(self.set_at));
        if let Some(n) = &self.note {
            m.insert("note".into(), json!(n));
        }
        Json::Object(m)
    }
}

fn read_aliases() -> Result<Vec<Alias>, String> {
    let path = aliases_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read aliases file: {e}"))?;
    let val: toml::Value = toml::from_str(&text)
        .map_err(|e| format!("Failed to parse aliases file: {e}"))?;
    let arr = val
        .get("alias")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let t = match entry {
            toml::Value::Table(t) => t,
            _ => continue,
        };
        let name = match t.get("name").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let card_id = match t.get("card_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push(Alias {
            name,
            card_id,
            pkg: t.get("pkg").and_then(|v| v.as_str()).map(String::from),
            set_at: t
                .get("set_at")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_default(),
            note: t.get("note").and_then(|v| v.as_str()).map(String::from),
        });
    }
    Ok(out)
}

fn write_aliases(aliases: &[Alias]) -> Result<(), String> {
    let path = aliases_path()?;
    let mut arr = Vec::with_capacity(aliases.len());
    for a in aliases {
        let mut t = toml::map::Map::new();
        t.insert("name".into(), toml::Value::String(a.name.clone()));
        t.insert("card_id".into(), toml::Value::String(a.card_id.clone()));
        if let Some(p) = &a.pkg {
            t.insert("pkg".into(), toml::Value::String(p.clone()));
        }
        t.insert("set_at".into(), toml::Value::String(a.set_at.clone()));
        if let Some(n) = &a.note {
            t.insert("note".into(), toml::Value::String(n.clone()));
        }
        arr.push(toml::Value::Table(t));
    }
    let mut root = toml::map::Map::new();
    root.insert("alias".into(), toml::Value::Array(arr));
    let text = toml::to_string_pretty(&toml::Value::Table(root))
        .map_err(|e| format!("Failed to serialize aliases: {e}"))?;
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, &text).map_err(|e| format!("Failed to write aliases tmp: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename aliases file: {e}"))?;
    Ok(())
}

/// Bind (or rebind) an alias to a Card.
///
/// Validates that `card_id` exists. If an alias with the same `name` already
/// exists it is overwritten — the alias table is intentionally mutable even
/// though the Cards themselves are not.
pub fn alias_set(
    name: &str,
    card_id: &str,
    pkg: Option<&str>,
    note: Option<&str>,
) -> Result<Alias, String> {
    validate_name(name, "alias")?;
    if find_card_path(card_id)?.is_none() {
        return Err(format!("alc.card.alias_set: card '{card_id}' not found"));
    }
    let mut aliases = read_aliases()?;
    aliases.retain(|a| a.name != name);
    let entry = Alias {
        name: name.to_string(),
        card_id: card_id.to_string(),
        pkg: pkg.map(String::from),
        set_at: now_rfc3339(),
        note: note.map(String::from),
    };
    aliases.push(entry.clone());
    write_aliases(&aliases)?;
    Ok(entry)
}

/// List aliases, optionally filtered by pkg.
pub fn alias_list(pkg_filter: Option<&str>) -> Result<Vec<Alias>, String> {
    let mut aliases = read_aliases()?;
    if let Some(p) = pkg_filter {
        aliases.retain(|a| a.pkg.as_deref() == Some(p));
    }
    Ok(aliases)
}

pub fn aliases_to_json(rows: &[Alias]) -> Json {
    Json::Array(rows.iter().map(|a| a.to_json()).collect())
}

/// Query parameters for `find`. All filters are optional.
#[derive(Debug, Default, Clone)]
pub struct FindQuery {
    pub pkg: Option<String>,
    pub scenario: Option<String>,
    pub model: Option<String>,
    /// One of: `"pass_rate"` (desc), `"pass_rate_asc"`, `"created_at"` (desc, default).
    pub sort: Option<String>,
    pub limit: Option<usize>,
    pub min_pass_rate: Option<f64>,
}

/// Filter/sort Cards across the store.
///
/// Thin layer over `list`: loads all summaries (optionally restricted to
/// a pkg subdir), applies field filters, sorts, and truncates.
pub fn find(q: FindQuery) -> Result<Vec<Summary>, String> {
    let mut rows = list(q.pkg.as_deref())?;
    if let Some(s) = &q.scenario {
        rows.retain(|r| r.scenario.as_deref() == Some(s.as_str()));
    }
    if let Some(m) = &q.model {
        rows.retain(|r| r.model.as_deref() == Some(m.as_str()));
    }
    if let Some(min) = q.min_pass_rate {
        rows.retain(|r| r.pass_rate.is_some_and(|v| v >= min));
    }
    match q.sort.as_deref() {
        Some("pass_rate") => rows.sort_by(|a, b| {
            b.pass_rate
                .partial_cmp(&a.pass_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        Some("pass_rate_asc") => rows.sort_by(|a, b| {
            a.pass_rate
                .partial_cmp(&b.pass_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => {
            rows.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| b.card_id.cmp(&a.card_id))
            });
        }
    }
    if let Some(lim) = q.limit {
        rows.truncate(lim);
    }
    Ok(rows)
}

// ───────────────────────────────────────────────────────────────
// Samples sidecar: per-case detail written alongside a Card as
// `{pkg}/{card_id}.samples.jsonl`. Write-once to preserve Card
// immutability: once a Card has a samples file, it cannot be
// rewritten — mismatched per-case data would break auditability.
// ───────────────────────────────────────────────────────────────

/// Resolve the samples sidecar path for a Card.
///
/// Returns an error if the Card does not exist — samples without a
/// parent Card are meaningless and we refuse to create orphans.
fn samples_path(card_id: &str) -> Result<PathBuf, String> {
    let card_path = find_card_path(card_id)?
        .ok_or_else(|| format!("card '{card_id}' not found"))?;
    let dir = card_path
        .parent()
        .ok_or_else(|| format!("card '{card_id}' has no parent directory"))?;
    Ok(dir.join(format!("{card_id}.samples.jsonl")))
}

/// Write per-case samples to `{card_id}.samples.jsonl` (write-once).
///
/// Each `samples` entry is serialized as one compact JSON line.
/// Fails if a samples file already exists for this card — mirrors
/// the immutability guarantee of Cards themselves.
pub fn write_samples(card_id: &str, samples: Vec<Json>) -> Result<PathBuf, String> {
    let path = samples_path(card_id)?;
    if path.exists() {
        return Err(format!(
            "alc.card.write_samples: samples already exist for card '{card_id}' (write-once)"
        ));
    }
    let mut buf = String::new();
    for (idx, s) in samples.iter().enumerate() {
        let line = serde_json::to_string(s).map_err(|e| {
            format!("alc.card.write_samples: failed to serialize sample #{idx}: {e}")
        })?;
        buf.push_str(&line);
        buf.push('\n');
    }
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, &buf)
        .map_err(|e| format!("Failed to write samples tmp: {e}"))?;
    fs::rename(&tmp, &path)
        .map_err(|e| format!("Failed to rename samples file: {e}"))?;
    Ok(path)
}

/// Read per-case samples from `{card_id}.samples.jsonl`.
///
/// Returns an empty Vec if no samples file exists (Cards without
/// per-case details are the common case, not an error).
pub fn read_samples(
    card_id: &str,
    offset: usize,
    limit: Option<usize>,
) -> Result<Vec<Json>, String> {
    let path = samples_path(card_id)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read samples file: {e}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if i < offset {
            continue;
        }
        if let Some(lim) = limit {
            if out.len() >= lim {
                break;
            }
        }
        let val: Json = serde_json::from_str(line)
            .map_err(|e| format!("Failed to parse sample line {i}: {e}"))?;
        out.push(val);
    }
    Ok(out)
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
    fn card_id_embeds_compact_timestamp() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({ "pkg": { "name": pkg } })).unwrap();
        // Expect: {pkg}_{model}_{YYYYMMDDTHHMMSS}_{hash6}
        // After removing the pkg prefix, there should be a segment
        // containing 'T' separating date and time.
        let tail = id.strip_prefix(&format!("{pkg}_")).unwrap();
        let parts: Vec<&str> = tail.split('_').collect();
        // parts = [model_short, YYYYMMDDTHHMMSS, hash6]
        assert_eq!(parts.len(), 3, "unexpected card_id shape: {id}");
        let ts = parts[1];
        assert_eq!(ts.len(), 15, "timestamp segment wrong length: {ts}");
        assert!(ts.chars().nth(8) == Some('T'), "missing T separator: {ts}");
        cleanup(&pkg);
    }

    #[test]
    fn now_compact_format() {
        let s = now_compact();
        assert_eq!(s.len(), 15);
        assert_eq!(s.chars().nth(8), Some('T'));
        // All other positions are digits
        for (i, c) in s.chars().enumerate() {
            if i != 8 {
                assert!(c.is_ascii_digit(), "non-digit at pos {i}: {s}");
            }
        }
    }

    #[test]
    fn short_model_variants() {
        assert_eq!(short_model("claude-opus-4-6"), "opus46");
        assert_eq!(short_model("gpt-4o"), "4o");
        assert_eq!(short_model(""), "model");
    }

    #[test]
    fn two_cards_same_second_different_stats_get_distinct_ids() {
        let pkg = unique_pkg();
        let input1 = json!({
            "pkg": { "name": pkg },
            "scenario": { "name": "gsm8k" },
            "stats": { "pass_rate": 0.4 }
        });
        let input2 = json!({
            "pkg": { "name": pkg },
            "scenario": { "name": "gsm8k" },
            "stats": { "pass_rate": 0.9 }
        });
        let (id1, _) = create(input1).unwrap();
        let (id2, _) = create(input2).unwrap();
        assert_ne!(id1, id2, "distinct stats must yield distinct card_ids");
        cleanup(&pkg);
    }

    // ─── P1: append ────────────────────────────────────────────

    #[test]
    fn append_adds_new_fields() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 }
        }))
        .unwrap();

        let merged = append(
            &id,
            json!({
                "caveats": { "notes": "rescored after fix" },
                "metadata": { "reviewer": "yn" }
            }),
        )
        .unwrap();
        assert_eq!(merged["caveats"]["notes"], json!("rescored after fix"));
        assert_eq!(merged["metadata"]["reviewer"], json!("yn"));

        // Persisted
        let got = get(&id).unwrap().unwrap();
        assert_eq!(got["caveats"]["notes"], json!("rescored after fix"));
        // Existing field untouched
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));

        cleanup(&pkg);
    }

    #[test]
    fn append_rejects_existing_key() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 }
        }))
        .unwrap();

        let err = append(&id, json!({ "stats": { "pass_rate": 0.9 } })).unwrap_err();
        assert!(err.contains("already set"), "got: {err}");
        // Verify original value still there
        let got = get(&id).unwrap().unwrap();
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));

        cleanup(&pkg);
    }

    #[test]
    fn append_errors_on_missing_card() {
        let err = append("does_not_exist_xyz", json!({ "x": 1 })).unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── P1: alias_set / alias_list ────────────────────────────

    #[test]
    fn alias_set_and_list_roundtrip() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({ "pkg": { "name": pkg } })).unwrap();

        let alias_name = format!("test_alias_{}", &pkg);
        alias_set(&alias_name, &id, Some(&pkg), Some("smoke")).unwrap();

        let rows = alias_list(Some(&pkg)).unwrap();
        let a = rows.iter().find(|a| a.name == alias_name).unwrap();
        assert_eq!(a.card_id, id);
        assert_eq!(a.pkg.as_deref(), Some(pkg.as_str()));
        assert_eq!(a.note.as_deref(), Some("smoke"));
        assert!(!a.set_at.is_empty());

        // Rebind to a new card
        let (id2, _) = create(json!({
            "card_id": format!("{pkg}_b"),
            "pkg": { "name": pkg }
        }))
        .unwrap();
        alias_set(&alias_name, &id2, Some(&pkg), None).unwrap();
        let rows = alias_list(Some(&pkg)).unwrap();
        let matching: Vec<&Alias> = rows.iter().filter(|a| a.name == alias_name).collect();
        assert_eq!(matching.len(), 1, "alias should be unique by name");
        assert_eq!(matching[0].card_id, id2);

        // Cleanup: remove our alias from the file
        let remaining: Vec<Alias> = read_aliases()
            .unwrap()
            .into_iter()
            .filter(|a| a.name != alias_name)
            .collect();
        write_aliases(&remaining).unwrap();
        cleanup(&pkg);
    }

    #[test]
    fn alias_set_rejects_unknown_card() {
        let err = alias_set("x", "does_not_exist_xyz", None, None).unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── P1: find ──────────────────────────────────────────────

    #[test]
    fn find_filters_and_sorts_by_pass_rate() {
        let pkg = unique_pkg();
        create(json!({
            "card_id": format!("{pkg}_low"),
            "pkg": { "name": pkg },
            "scenario": { "name": "gsm8k" },
            "stats": { "pass_rate": 0.4 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_high"),
            "pkg": { "name": pkg },
            "scenario": { "name": "gsm8k" },
            "stats": { "pass_rate": 0.9 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_other"),
            "pkg": { "name": pkg },
            "scenario": { "name": "other" },
            "stats": { "pass_rate": 1.0 }
        }))
        .unwrap();

        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            scenario: Some("gsm8k".into()),
            sort: Some("pass_rate".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].pass_rate, Some(0.9));
        assert_eq!(rows[1].pass_rate, Some(0.4));

        // min_pass_rate filter
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            min_pass_rate: Some(0.8),
            sort: Some("pass_rate".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.pass_rate.unwrap() >= 0.8));

        // limit
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            sort: Some("pass_rate".into()),
            limit: Some(1),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pass_rate, Some(1.0));

        cleanup(&pkg);
    }

    // ─── samples sidecar ───────────────────────────────────────

    #[test]
    fn write_and_read_samples_roundtrip() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 }
        }))
        .unwrap();

        let samples = vec![
            json!({ "case": "c0", "passed": true, "score": 1.0 }),
            json!({ "case": "c1", "passed": false, "score": 0.0 }),
            json!({ "case": "c2", "passed": true, "score": 0.75 }),
        ];
        let path = write_samples(&id, samples.clone()).unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().ends_with(".samples.jsonl"));

        let got = read_samples(&id, 0, None).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["case"], json!("c0"));
        assert_eq!(got[2]["score"], json!(0.75));

        // offset + limit
        let slice = read_samples(&id, 1, Some(1)).unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0]["case"], json!("c1"));

        cleanup(&pkg);
    }

    #[test]
    fn write_samples_is_write_once() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({ "pkg": { "name": pkg } })).unwrap();
        write_samples(&id, vec![json!({ "x": 1 })]).unwrap();
        let err = write_samples(&id, vec![json!({ "x": 2 })]).unwrap_err();
        assert!(err.contains("already exist"), "got: {err}");
        cleanup(&pkg);
    }

    #[test]
    fn read_samples_empty_when_absent() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({ "pkg": { "name": pkg } })).unwrap();
        let got = read_samples(&id, 0, None).unwrap();
        assert!(got.is_empty());
        cleanup(&pkg);
    }

    #[test]
    fn samples_errors_on_missing_card() {
        let err = write_samples("does_not_exist_xyz_samples", vec![json!({})]).unwrap_err();
        assert!(err.contains("not found"));
    }
}
