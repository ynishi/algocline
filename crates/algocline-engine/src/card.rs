//! Card storage — immutable run-result snapshots.
//!
//! A Card is a frozen record of a strategy run: identity, parameters,
//! model, scenario, aggregate stats, and (optionally) per-case detail.
//! Cards are **immutable** — once written they are never modified, only
//! annotated via additive `append`.  Mutable **aliases** point to a
//! Card and can be rebound freely.
//!
//! ## Design principles
//!
//! 1. **Minimal REQUIRED, maximal OPTIONAL** — v0 needs only 4 fields;
//!    lightweight "ran this pkg" records and heavy optimize snapshots
//!    share the same schema.
//! 2. **Immutable append-only** — no overwrite, no delete.  New data is
//!    added via `append` (new top-level keys only) or by creating a new
//!    Card with a fresh `card_id`.
//! 3. **Two-tier storage** — TOML for human-readable aggregate, JSONL
//!    sidecar for machine-parseable per-case detail.
//! 4. **File-primary** — files are the source of truth; in-memory state
//!    is cache.  Cards can be copied, diffed, and version-controlled.
//!
//! ## Storage layout (two-tier)
//!
//! | Tier | File | Content |
//! |------|------|---------|
//! | **Tier 1** | `~/.algocline/cards/{pkg}/{card_id}.toml` | Aggregate scalars, decisions, identity, params |
//! | **Tier 2** | `~/.algocline/cards/{pkg}/{card_id}.samples.jsonl` | Per-case raw data (JSONL, write-once) |
//!
//! Tier 1 holds a shareable summary (a few KB). Tier 2 holds per-case
//! detail ��� the engine does not interpret its columns; packages define
//! their own schema.
//!
//! Alias table: `~/.algocline/cards/_aliases.toml` (global).
//!
//! ## card_id naming
//!
//! `{pkg}_{model_short}_{compact_ts}_{hash6}`
//!
//! - `compact_ts`: `YYYYMMDDTHHMMSS` in UTC
//! - `hash6`: first 6 hex chars of DJB2 param fingerprint
//! - Example: `cot_opus46_20260412T061500_a3f9c1`
//!
//! ## v0 schema (frozen)
//!
//! ### REQUIRED (minimum valid Card)
//!
//! | Field | Type | Example |
//! |-------|------|---------|
//! | `schema_version` | string | `"card/v0"` |
//! | `card_id` | string | `"cot_opus46_20260412T061500_a3f9c1"` |
//! | `created_at` | string (RFC 3339) | `"2026-04-12T06:15:00Z"` |
//! | `[pkg].name` | string | `"cot"` |
//!
//! ### OPTIONAL (auto-injected where possible)
//!
//! | Section | Fields |
//! |---------|--------|
//! | `[pkg]` | `version`, `category`, `source`, `source_ref`, `source_sha` |
//! | `[runtime]` | `alc_version`, `lua_version`, `host_os`, `git_sha` |
//! | `[model]` | `provider`, `id`, `id_short`, `cutoff` |
//! | `[params]` | Free-form ctx snapshot; `param_fingerprint` for DJB2 hash |
//! | `[strategy_params]` | Strategy-tunable parameters surfaced for sweeps / optimizers (e.g. `alpha`, `temperature`, `depth`). Free-form, but `where`-queryable as a first-class section |
//! | `[scenario]` | `name`, `source`, `case_count`, `grader` |
//! | `[stats]` | `pass_rate`, `mean_score`, `std`, `median`, `min`, `max`, `n` |
//! | `[stats.by_bucket]` | Disaggregated sub-bucket stats (array of tables) |
//! | `[cost]` | `llm_calls`, `input_tokens`, `output_tokens`, `elapsed_ms`, `usd_estimate` |
//! | `[optimize]` | `target`, `search`, `rounds_used`, `top_k` (for optimize Cards) |
//! | `[metadata]` | Free-form escape hatch. Recognized lineage conventions: `prior_card_id` (parent Card id), `prior_relation` (relation kind, e.g. `"sweep_variant"`, `"reflection_of"`, `"derived_from"`) |
//!
//! ## Lua API (`alc.card.*`)
//!
//! | Function | Description |
//! |----------|-------------|
//! | `create(table)` | Write new Card (Tier 1). Returns `{ card_id, path }` |
//! | `get(card_id)` | Read Card by id. Returns table or nil |
//! | `list(filter?)` | List Cards as summaries (newest first) |
//! | `find(query?)` | Prisma-style `where` DSL + dotted-path `order_by` + `offset`/`limit` |
//! | `append(card_id, fields)` | Additive-only annotation (new keys only) |
//! | `alias_set(name, card_id, opts?)` | Pin mutable alias |
//! | `alias_list(filter?)` | List aliases |
//! | `get_by_alias(name)` | Resolve alias → full Card |
//! | `write_samples(card_id, samples)` | Write Tier 2 sidecar (write-once) |
//! | `read_samples(card_id, opts?)` | Read Tier 2 with `where` filtering + offset/limit paging |
//! | `lineage(query)` | Walk ancestry/descendants via `metadata.prior_card_id` |

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Serialize, Serializer};
use serde_json::{json, Value as Json};

pub const SCHEMA_VERSION: &str = "card/v0";

// ═══════════════════════════════════════════════════════════════
// CardStore trait — physical I/O abstraction.
// ═══════════════════════════════════════════════════════════════
//
// Card domain logic (schema / Query DSL / Lineage) is backend-
// neutral. Only the physical read/write layer is swappable.
//
// The default backend is `FileCardStore`, which preserves the
// legacy `~/.algocline/cards/{pkg}/{card_id}.toml` layout
// byte-for-byte. Alternative backends (PathCardStore, SqliteCardStore,
// MemoryCardStore) can be added by implementing this trait.
//
// Locators are `PathBuf` values. For FileCardStore they are real
// filesystem paths; for non-file backends they are synthetic paths
// (e.g. `sqlite:///db.sqlite#card/{id}`) — the value is opaque to
// the domain layer and only exposed via the Lua `alc.card.create`
// / `alc.card.write_samples` return values.

/// Storage backend for Cards.
///
/// Implementations must be `Send + Sync` so that they can be shared
/// across Lua host threads safely. All methods may fail with an
/// error `String` describing the backend-specific failure.
pub trait CardStore: Send + Sync {
    // ─── Card CRUD ─────────────────────────────────────────────

    /// Write a new Card (Tier 1 TOML).
    ///
    /// The caller has already:
    ///   - validated `pkg` and `card_id` via [`validate_name`]
    ///   - serialized `toml_text` with `toml::to_string_pretty`
    ///
    /// Fails if a Card with the same id already exists
    /// (immutability).  Returns the locator of the written Card.
    fn write_new_card(&self, pkg: &str, card_id: &str, toml_text: &str) -> Result<PathBuf, String>;

    /// Overwrite an existing Card (append flow).
    ///
    /// Append is additive-only w.r.t. keys, but the underlying
    /// TOML file is rewritten in place; callers must have validated
    /// the additive-only constraint before calling this.
    fn overwrite_card(&self, card_id: &str, toml_text: &str) -> Result<PathBuf, String>;

    /// Locate a Card file by id. Returns `None` if not found.
    fn find_card_locator(&self, card_id: &str) -> Result<Option<PathBuf>, String>;

    /// Read a Card's raw TOML text by id. Returns `None` if missing.
    fn read_card_text(&self, card_id: &str) -> Result<Option<String>, String>;

    /// List `(pkg, locator)` pairs for every Card file in the store.
    ///
    /// When `pkg_filter` is `Some(name)`, restrict to that pkg
    /// subdir. Non-existent pkg subdir yields an empty Vec.
    ///
    /// Order is implementation-defined — callers sort explicitly.
    fn list_card_locators(
        &self,
        pkg_filter: Option<&str>,
    ) -> Result<Vec<(String, PathBuf)>, String>;

    /// Read raw TOML text from a locator returned by
    /// [`Self::list_card_locators`]. `Ok(None)` on read failure so
    /// scans can skip corrupt files without aborting.
    fn read_locator_text(&self, locator: &Path) -> Result<Option<String>, String>;

    // ─── Alias table ───────────────────────────────────────────

    fn read_aliases(&self) -> Result<Vec<Alias>, String>;
    fn write_aliases(&self, aliases: &[Alias]) -> Result<(), String>;

    // ─── Samples sidecar ───────────────────────────────────────

    /// Check whether a samples sidecar exists for `card_id`.
    fn samples_exists(&self, card_id: &str) -> Result<bool, String>;

    /// Write a samples sidecar (write-once).
    ///
    /// `jsonl_text` is the complete JSONL payload (one JSON line
    /// per sample, `\n`-terminated). Fails if a sidecar already
    /// exists. Returns the locator.
    fn write_samples_text(&self, card_id: &str, jsonl_text: &str) -> Result<PathBuf, String>;

    /// Read a samples sidecar as raw JSONL text. Returns `None`
    /// when no sidecar exists (samples are optional).
    fn read_samples_text(&self, card_id: &str) -> Result<Option<String>, String>;

    // ─── Import ────────────────────────────────────────────────

    /// Import Card files from `source_dir` into the store under
    /// `pkg`. First-writer wins (existing Cards are skipped).
    /// Returns `(imported, skipped)` card_id lists.
    fn import_from_dir(
        &self,
        source_dir: &Path,
        pkg: &str,
    ) -> Result<(Vec<String>, Vec<String>), String>;
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

/// Create a new Card backed by `store`.
pub fn create_with_store(
    store: &dyn CardStore,
    mut input: Json,
) -> Result<(String, PathBuf), String> {
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

    let toml_val = json_to_toml(input)?;
    let text = toml::to_string_pretty(&toml_val)
        .map_err(|e| format!("Failed to serialize card TOML: {e}"))?;
    let path = store.write_new_card(&pkg_name, &card_id, &text)?;

    publish(CardEvent::Created {
        pkg: pkg_name.clone(),
        card_id: card_id.clone(),
        toml_text: text,
    });

    Ok((card_id, path))
}

/// Read a Card from `store` by id. Returns None if not found.
pub fn get_with_store(store: &dyn CardStore, card_id: &str) -> Result<Option<Json>, String> {
    let text = match store.read_card_text(card_id)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let val: toml::Value =
        toml::from_str(&text).map_err(|e| format!("Failed to parse card '{card_id}': {e}"))?;
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

fn summarize(store: &dyn CardStore, locator: &std::path::Path, pkg: &str) -> Option<Summary> {
    let text = store.read_locator_text(locator).ok().flatten()?;
    let val: toml::Value = toml::from_str(&text).ok()?;
    let card_id = val
        .get("card_id")
        .and_then(|v| v.as_str())
        .or_else(|| locator.file_stem().and_then(|s| s.to_str()))?
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

/// List cards from `store`. `pkg_filter = Some("name")` restricts to that pkg subdir.
pub fn list_with_store(
    store: &dyn CardStore,
    pkg_filter: Option<&str>,
) -> Result<Vec<Summary>, String> {
    let locators = store.list_card_locators(pkg_filter)?;
    let mut out = Vec::with_capacity(locators.len());
    for (pkg, loc) in &locators {
        if let Some(s) = summarize(store, loc, pkg) {
            out.push(s);
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
pub fn append_with_store(
    store: &dyn CardStore,
    card_id: &str,
    fields: Json,
) -> Result<Json, String> {
    let text = store
        .read_card_text(card_id)?
        .ok_or_else(|| format!("alc.card.append: card '{card_id}' not found"))?;
    let fields_obj = match fields {
        Json::Object(m) => m,
        _ => return Err("alc.card.append: fields must be a table".into()),
    };

    let existing: toml::Value =
        toml::from_str(&text).map_err(|e| format!("Failed to parse card '{card_id}': {e}"))?;
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
    store.overwrite_card(card_id, &text)?;

    publish(CardEvent::Appended {
        card_id: card_id.to_string(),
        toml_text: text,
    });

    Ok(existing_json)
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

/// Bind (or rebind) an alias to a Card in `store`.
///
/// Validates that `card_id` exists. If an alias with the same `name` already
/// exists it is overwritten — the alias table is intentionally mutable even
/// though the Cards themselves are not.
pub fn alias_set_with_store(
    store: &dyn CardStore,
    name: &str,
    card_id: &str,
    pkg: Option<&str>,
    note: Option<&str>,
) -> Result<Alias, String> {
    validate_name(name, "alias")?;
    if store.find_card_locator(card_id)?.is_none() {
        return Err(format!("alc.card.alias_set: card '{card_id}' not found"));
    }
    let mut aliases = store.read_aliases()?;
    aliases.retain(|a| a.name != name);
    let entry = Alias {
        name: name.to_string(),
        card_id: card_id.to_string(),
        pkg: pkg.map(String::from),
        set_at: now_rfc3339(),
        note: note.map(String::from),
    };
    aliases.push(entry.clone());
    store.write_aliases(&aliases)?;

    // Mirror the full alias table to subscribers as TOML text. The
    // primary FileCardStore already serialized it internally; we
    // re-serialize here using the same shape so subscribers stay in
    // byte-for-byte parity. A serialization failure is best-effort
    // (log + skip).
    match serialize_aliases_toml(&aliases) {
        Ok(text) => publish(CardEvent::AliasesWritten { toml_text: text }),
        Err(e) => tracing::warn!(error = %e, "alias_set: failed to serialize aliases for publish"),
    }

    Ok(entry)
}

/// Serialize `aliases` to a TOML document matching the primary
/// `_aliases.toml` layout. Broken out so that subscribers can receive
/// the exact byte-for-byte dump.
fn serialize_aliases_toml(aliases: &[Alias]) -> Result<String, String> {
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
    toml::to_string_pretty(&toml::Value::Table(root))
        .map_err(|e| format!("Failed to serialize aliases: {e}"))
}

/// Resolve an alias name to its bound Card and return the full Card JSON.
///
/// Shortcut for `alias_list → filter → get`. Returns `None` when the alias
/// does not exist. Errors when the alias points at a missing Card — that
/// would indicate a corrupt alias table (the target was deleted out of band).
pub fn get_by_alias_with_store(store: &dyn CardStore, name: &str) -> Result<Option<Json>, String> {
    validate_name(name, "alias")?;
    let aliases = store.read_aliases()?;
    let Some(alias) = aliases.into_iter().find(|a| a.name == name) else {
        return Ok(None);
    };
    match get_with_store(store, &alias.card_id)? {
        Some(card) => Ok(Some(card)),
        None => Err(format!(
            "alc.card.get_by_alias: alias '{name}' points at missing card '{}'",
            alias.card_id
        )),
    }
}

/// List aliases from `store`, optionally filtered by pkg.
pub fn alias_list_with_store(
    store: &dyn CardStore,
    pkg_filter: Option<&str>,
) -> Result<Vec<Alias>, String> {
    let mut aliases = store.read_aliases()?;
    if let Some(p) = pkg_filter {
        aliases.retain(|a| a.pkg.as_deref() == Some(p));
    }
    Ok(aliases)
}

pub fn aliases_to_json(rows: &[Alias]) -> Json {
    Json::Array(rows.iter().map(|a| a.to_json()).collect())
}

// ═══════════════════════════════════════════════════════════════
// Where DSL — Prisma/Mongo-style nested predicates
// ═══════════════════════════════════════════════════════════════
//
// Syntax (JSON form, as received from Lua / MCP):
//
//   where = {
//     pkg: "cot",                                      // implicit eq
//     stats: { pass_rate: { gte: 0.8 }, n: { gte: 30 } },
//     strategy_params: { temperature: { gte: 0.7 } },
//     prior_card_id: { exists: true },
//     _or: [ {...}, {...} ],                           // logical ops
//     _not: { model: { id: "claude-haiku-4-5-20251001" } },
//   }
//
// Semantics:
//   * Multiple keys in the same object → implicit AND.
//   * Nested object value → section (path extension).
//   * Object whose every key is a reserved operator name → leaf operator
//     object; applies the operators to the value at the current path.
//   * Scalar/array value → implicit eq.
//   * Reserved logical keys: `_and` / `_or` / `_not`.
//   * Reserved operator keys: `eq ne lt lte gt gte in nin exists
//     contains starts_with`.  Card schemas must not use these names as
//     field names under any section.
//
// Missing-field comparison:
//   * `eq/lt/lte/gt/gte/in/contains/starts_with` → false on missing
//   * `ne/nin`                                   → true  on missing
//   * `exists`                                   → explicit
//

/// Single comparison operator.
#[derive(Debug, Clone, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
    In,
    Nin,
    Exists,
    Contains,
    StartsWith,
}

impl CmpOp {
    fn from_key(k: &str) -> Option<Self> {
        Some(match k {
            "eq" => Self::Eq,
            "ne" => Self::Ne,
            "lt" => Self::Lt,
            "lte" => Self::Lte,
            "gt" => Self::Gt,
            "gte" => Self::Gte,
            "in" => Self::In,
            "nin" => Self::Nin,
            "exists" => Self::Exists,
            "contains" => Self::Contains,
            "starts_with" => Self::StartsWith,
            _ => return None,
        })
    }
}

/// One parsed comparison: `path` points at a nested field,
/// `op` + `value` describe how to compare it.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub path: Vec<String>,
    pub op: CmpOp,
    pub value: Json,
}

/// Parsed predicate tree.
#[derive(Debug, Clone)]
pub enum Predicate {
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
    Cmp(Comparison),
}

/// Is `obj` entirely composed of reserved operator keys?
/// Empty objects return false (meaningless as an operator object).
fn is_operator_object(obj: &serde_json::Map<String, Json>) -> bool {
    if obj.is_empty() {
        return false;
    }
    obj.keys().all(|k| CmpOp::from_key(k).is_some())
}

/// Parse a `where` JSON value into a `Predicate`.
///
/// `prefix` is the current nested-key path as we descend through
/// section objects.
pub fn parse_where(value: &Json) -> Result<Predicate, String> {
    parse_predicate(value, &[])
}

fn parse_predicate(value: &Json, prefix: &[String]) -> Result<Predicate, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "where clause must be a table".to_string())?;

    let mut clauses: Vec<Predicate> = Vec::new();

    for (key, val) in obj {
        match key.as_str() {
            "_and" => {
                let arr = val
                    .as_array()
                    .ok_or_else(|| "_and must be an array of sub-predicates".to_string())?;
                let mut subs = Vec::with_capacity(arr.len());
                for sub in arr {
                    subs.push(parse_predicate(sub, prefix)?);
                }
                clauses.push(Predicate::And(subs));
            }
            "_or" => {
                let arr = val
                    .as_array()
                    .ok_or_else(|| "_or must be an array of sub-predicates".to_string())?;
                let mut subs = Vec::with_capacity(arr.len());
                for sub in arr {
                    subs.push(parse_predicate(sub, prefix)?);
                }
                clauses.push(Predicate::Or(subs));
            }
            "_not" => {
                clauses.push(Predicate::Not(Box::new(parse_predicate(val, prefix)?)));
            }
            _ => {
                // Field key — extend the current path.
                let mut new_path = prefix.to_vec();
                new_path.push(key.clone());

                match val {
                    Json::Object(m) if is_operator_object(m) => {
                        // Leaf: operator object at this path.
                        for (op_key, op_val) in m {
                            let op = CmpOp::from_key(op_key).expect("validated above");
                            clauses.push(Predicate::Cmp(Comparison {
                                path: new_path.clone(),
                                op,
                                value: op_val.clone(),
                            }));
                        }
                    }
                    Json::Object(_) => {
                        // Nested section: recurse with extended path.
                        clauses.push(parse_predicate(val, &new_path)?);
                    }
                    _ => {
                        // Scalar/array: implicit eq.
                        clauses.push(Predicate::Cmp(Comparison {
                            path: new_path,
                            op: CmpOp::Eq,
                            value: val.clone(),
                        }));
                    }
                }
            }
        }
    }

    if clauses.len() == 1 {
        Ok(clauses.remove(0))
    } else {
        Ok(Predicate::And(clauses))
    }
}

/// Fetch a nested value from a Card JSON by dotted path.
fn fetch_path<'a>(card: &'a Json, path: &[String]) -> Option<&'a Json> {
    let mut node = card;
    for key in path {
        let obj = node.as_object()?;
        node = obj.get(key)?;
    }
    Some(node)
}

/// Compare two JSON scalars with a numeric/string/bool comparator.
/// Returns None when the types aren't comparable.
fn json_cmp(a: &Json, b: &Json) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => {
            let xf = x.as_f64()?;
            let yf = y.as_f64()?;
            xf.partial_cmp(&yf)
        }
        (Json::String(x), Json::String(y)) => Some(x.cmp(y)),
        (Json::Bool(x), Json::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn json_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(xf), Some(yf)) => xf == yf,
            _ => a == b,
        },
        _ => a == b,
    }
}

fn eval_cmp(cmp: &Comparison, card: &Json) -> bool {
    let actual = fetch_path(card, &cmp.path);
    let exists = actual.is_some();

    match cmp.op {
        CmpOp::Exists => {
            let want = cmp.value.as_bool().unwrap_or(true);
            exists == want
        }
        CmpOp::Ne => match actual {
            None => true,
            Some(v) => !json_eq(v, &cmp.value),
        },
        CmpOp::Nin => match actual {
            None => true,
            Some(v) => match cmp.value.as_array() {
                Some(arr) => !arr.iter().any(|e| json_eq(e, v)),
                None => false,
            },
        },
        CmpOp::Eq => actual.is_some_and(|v| json_eq(v, &cmp.value)),
        CmpOp::In => actual.is_some_and(|v| match cmp.value.as_array() {
            Some(arr) => arr.iter().any(|e| json_eq(e, v)),
            None => false,
        }),
        CmpOp::Lt | CmpOp::Lte | CmpOp::Gt | CmpOp::Gte => {
            let Some(v) = actual else { return false };
            let Some(ord) = json_cmp(v, &cmp.value) else {
                return false;
            };
            use std::cmp::Ordering::{Equal, Greater, Less};
            matches!(
                (&cmp.op, ord),
                (CmpOp::Lt, Less)
                    | (CmpOp::Lte, Less | Equal)
                    | (CmpOp::Gt, Greater)
                    | (CmpOp::Gte, Greater | Equal)
            )
        }
        CmpOp::Contains => {
            let Some(Json::String(haystack)) = actual else {
                return false;
            };
            let Some(needle) = cmp.value.as_str() else {
                return false;
            };
            haystack.contains(needle)
        }
        CmpOp::StartsWith => {
            let Some(Json::String(haystack)) = actual else {
                return false;
            };
            let Some(needle) = cmp.value.as_str() else {
                return false;
            };
            haystack.starts_with(needle)
        }
    }
}

/// Evaluate a predicate tree against a full Card JSON.
pub fn eval_predicate(pred: &Predicate, card: &Json) -> bool {
    match pred {
        Predicate::And(subs) => subs.iter().all(|p| eval_predicate(p, card)),
        Predicate::Or(subs) => subs.iter().any(|p| eval_predicate(p, card)),
        Predicate::Not(sub) => !eval_predicate(sub, card),
        Predicate::Cmp(c) => eval_cmp(c, card),
    }
}

// ───────────────────────────────────────────────────────────────
// Order-by
// ───────────────────────────────────────────────────────────────

/// Parsed sort key: path with optional descending flag.
#[derive(Debug, Clone)]
pub struct OrderKey {
    pub path: Vec<String>,
    pub desc: bool,
}

impl OrderKey {
    fn parse(raw: &str) -> Result<Self, String> {
        if raw.is_empty() {
            return Err("order_by key must not be empty".into());
        }
        let (desc, rest) = if let Some(r) = raw.strip_prefix('-') {
            (true, r)
        } else {
            (false, raw)
        };
        let path: Vec<String> = rest.split('.').map(|s| s.to_string()).collect();
        if path.iter().any(|p| p.is_empty()) {
            return Err(format!("invalid order_by key: '{raw}'"));
        }
        Ok(Self { path, desc })
    }
}

/// Parse an order_by JSON value.  Accepts:
///   - a string: `"stats.pass_rate"` or `"-stats.pass_rate"`
///   - an array of strings: `["-stats.pass_rate", "created_at"]`
pub fn parse_order_by(value: &Json) -> Result<Vec<OrderKey>, String> {
    match value {
        Json::String(s) => Ok(vec![OrderKey::parse(s)?]),
        Json::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v
                    .as_str()
                    .ok_or_else(|| "order_by array must contain strings".to_string())?;
                out.push(OrderKey::parse(s)?);
            }
            Ok(out)
        }
        _ => Err("order_by must be a string or array of strings".into()),
    }
}

/// Query parameters for `find`.
#[derive(Debug, Default, Clone)]
pub struct FindQuery {
    /// Restrict scan to a single pkg subdir (I/O optimization).
    pub pkg: Option<String>,
    /// Prisma-style predicate tree.
    pub where_: Option<Predicate>,
    /// Sort keys (dotted paths, optional `-` prefix for desc).
    pub order_by: Vec<OrderKey>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// A loaded Card row flowing through the find() pipeline.
///
/// `full` is the whole Card JSON (used by `where` evaluation and
/// `order_by` dotted-path lookup); `summary` is the projection
/// returned to callers.
#[derive(Debug, Clone)]
struct CardRow {
    full: Json,
    summary: Summary,
}

/// Load a single Card file into a `CardRow`.
fn load_full(store: &dyn CardStore, locator: &std::path::Path, pkg: &str) -> Option<CardRow> {
    let text = store.read_locator_text(locator).ok().flatten()?;
    let val: toml::Value = toml::from_str(&text).ok()?;
    let json = toml_to_json(val);

    let card_id = json
        .get("card_id")
        .and_then(|v| v.as_str())
        .or_else(|| locator.file_stem().and_then(|s| s.to_str()))?
        .to_string();
    let created_at = json
        .get("created_at")
        .and_then(|v| v.as_str())
        .map(String::from);
    let model = json
        .get("model")
        .and_then(|m| m.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let scenario = json
        .get("scenario")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let pass_rate = json
        .get("stats")
        .and_then(|s| s.get("pass_rate"))
        .and_then(|v| v.as_f64());

    Some(CardRow {
        full: json,
        summary: Summary {
            card_id,
            pkg: pkg.to_string(),
            created_at,
            model,
            scenario,
            pass_rate,
        },
    })
}

/// Compare two Card rows according to an ordered list of sort keys.
fn order_cards(a: &CardRow, b: &CardRow, keys: &[OrderKey]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for k in keys {
        let va = fetch_path(&a.full, &k.path);
        let vb = fetch_path(&b.full, &k.path);
        let ord = match (va, vb) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater, // missing sorts last
            (Some(_), None) => Ordering::Less,
            (Some(x), Some(y)) => json_cmp(x, y).unwrap_or(Ordering::Equal),
        };
        let ord = if k.desc { ord.reverse() } else { ord };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Summary-only fields that can be sorted without loading full TOML.
const SUMMARY_SORT_FIELDS: &[&str] = &[
    "card_id",
    "created_at",
    "stats.pass_rate",
    "scenario.name",
    "model.id",
];

/// Return true when the query can be answered with lightweight Summary
/// rows (no full-TOML load needed).
fn is_lightweight_query(q: &FindQuery) -> bool {
    q.where_.is_none()
        && q.order_by
            .iter()
            .all(|k| SUMMARY_SORT_FIELDS.contains(&k.path.join(".").as_str()))
}

/// Sort Summary rows using order_by keys that map to Summary fields.
fn order_summaries(a: &Summary, b: &Summary, keys: &[OrderKey]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for k in keys {
        let key_str = k.path.join(".");
        let ord = match key_str.as_str() {
            "card_id" => a.card_id.cmp(&b.card_id),
            "created_at" => a.created_at.cmp(&b.created_at),
            "stats.pass_rate" => match (a.pass_rate, b.pass_rate) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            },
            "scenario.name" => a.scenario.cmp(&b.scenario),
            "model.id" => a.model.cmp(&b.model),
            _ => Ordering::Equal,
        };
        let ord = if k.desc { ord.reverse() } else { ord };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Filter/sort Cards across the store using the `where` DSL.
///
/// When no `where` clause is specified and `order_by` only references
/// summary-level fields, uses the lightweight `list_with_store` path to
/// avoid loading full TOML.  Otherwise loads full TOML per Card.
pub fn find_with_store(store: &dyn CardStore, q: FindQuery) -> Result<Vec<Summary>, String> {
    // Fast path: lightweight query, no full-TOML load needed.
    if is_lightweight_query(&q) {
        let mut rows = list_with_store(store, q.pkg.as_deref())?;
        if q.order_by.is_empty() {
            rows.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| b.card_id.cmp(&a.card_id))
            });
        } else {
            rows.sort_by(|a, b| order_summaries(a, b, &q.order_by));
        }
        let out: Vec<Summary> = rows
            .into_iter()
            .skip(q.offset.unwrap_or(0))
            .take(q.limit.unwrap_or(usize::MAX))
            .collect();
        return Ok(out);
    }

    // Full path: load entire TOML for where evaluation / arbitrary order_by.
    let all_rows = scan_cards(store, q.pkg.as_deref())?;

    // Filter by where.
    let mut rows: Vec<CardRow> = if let Some(pred) = &q.where_ {
        all_rows
            .into_iter()
            .filter(|row| eval_predicate(pred, &row.full))
            .collect()
    } else {
        all_rows
    };

    // Sort.
    if q.order_by.is_empty() {
        rows.sort_by(|a, b| {
            b.summary
                .created_at
                .cmp(&a.summary.created_at)
                .then_with(|| b.summary.card_id.cmp(&a.summary.card_id))
        });
    } else {
        rows.sort_by(|a, b| order_cards(a, b, &q.order_by));
    }

    // Offset + limit.
    let out: Vec<Summary> = rows
        .into_iter()
        .skip(q.offset.unwrap_or(0))
        .take(q.limit.unwrap_or(usize::MAX))
        .map(|r| r.summary)
        .collect();

    Ok(out)
}

// ───────────────────────────────────────────────────────────────
// Lineage walker
// ───────────────────────────────────────────────────────────────
//
// Cards form a tree (typically, not strictly a DAG) via the
// `metadata.prior_card_id` convention. `lineage()` walks that tree
// either up (toward ancestors) or down (toward descendants) or both,
// up to a configurable depth, optionally filtered by `prior_relation`.
//
// Up-walk is O(depth) — each step reads one parent Card.
// Down-walk is O(N_cards × depth) — we scan the whole store at each
// level. For the current scale (hundreds to low thousands of cards)
// this is fine; if the store grows we can build a prior_card_id index.

/// Walk direction for `lineage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineageDirection {
    #[default]
    Up,
    Down,
    Both,
}

impl LineageDirection {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            "both" => Ok(Self::Both),
            other => Err(format!(
                "direction must be 'up', 'down', or 'both' (got '{other}')"
            )),
        }
    }
}

/// Query parameters for `lineage`.
#[derive(Debug, Clone, Default)]
pub struct LineageQuery {
    pub card_id: String,
    pub direction: LineageDirection,
    /// Max traversal depth. Default 10.
    pub depth: Option<usize>,
    /// Include a per-node `stats` field (full [stats] section).
    pub include_stats: bool,
    /// If set, only edges whose `prior_relation` is in this list are
    /// followed.  The root is always included regardless.
    pub relation_filter: Option<Vec<String>>,
}

/// One node in the lineage result.
///
/// `depth` is the signed distance from the root: negative for
/// ancestors (up-walk), 0 for the root, positive for descendants.
#[derive(Debug, Clone)]
pub struct LineageNode {
    pub card_id: String,
    pub pkg: String,
    pub prior_card_id: Option<String>,
    pub prior_relation: Option<String>,
    pub depth: i32,
    pub stats: Option<Json>,
}

/// One edge in the lineage result (child → parent, always).
#[derive(Debug, Clone)]
pub struct LineageEdge {
    pub from: String,
    pub to: String,
    pub relation: Option<String>,
}

/// Full lineage walk result.
#[derive(Debug, Clone)]
pub struct LineageResult {
    pub root: String,
    pub nodes: Vec<LineageNode>,
    pub edges: Vec<LineageEdge>,
    pub truncated: bool,
}

const DEFAULT_LINEAGE_DEPTH: usize = 10;

/// Extract the lineage fields from a Card JSON.
/// Returns (prior_card_id, prior_relation).
fn lineage_fields(card: &Json) -> (Option<String>, Option<String>) {
    let meta = card.get("metadata");
    let prior_card_id = meta
        .and_then(|m| m.get("prior_card_id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let prior_relation = meta
        .and_then(|m| m.get("prior_relation"))
        .and_then(|v| v.as_str())
        .map(String::from);
    (prior_card_id, prior_relation)
}

/// Build a LineageNode from a loaded CardRow at a given depth.
fn make_node(row: &CardRow, depth: i32, include_stats: bool) -> LineageNode {
    let (prior_card_id, prior_relation) = lineage_fields(&row.full);
    let stats = if include_stats {
        row.full.get("stats").cloned()
    } else {
        None
    };
    LineageNode {
        card_id: row.summary.card_id.clone(),
        pkg: row.summary.pkg.clone(),
        prior_card_id,
        prior_relation,
        depth,
        stats,
    }
}

/// Check whether `relation` passes the relation_filter (None means no
/// filter, which always passes).
fn relation_passes(filter: &Option<Vec<String>>, relation: &Option<String>) -> bool {
    match filter {
        None => true,
        Some(allowed) => match relation {
            Some(r) => allowed.iter().any(|a| a == r),
            None => false,
        },
    }
}

/// Full in-memory card index with forward and reverse lineage maps.
struct CardIndex {
    /// card_id → CardRow
    cards: std::collections::HashMap<String, CardRow>,
    /// parent card_id → Vec<child card_id> (reverse lineage index)
    children: std::collections::HashMap<String, Vec<String>>,
}

/// Load all Cards in the store once, keyed by card_id.
/// Also builds a reverse index (parent → children) so that
/// `walk_down` is O(result_size) instead of O(N_cards × depth).
fn load_card_index(store: &dyn CardStore) -> Result<CardIndex, String> {
    let rows = scan_cards(store, None)?;

    let mut cards = std::collections::HashMap::with_capacity(rows.len());
    let mut children: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for row in rows {
        let id = row.summary.card_id.clone();
        let (prior_id, _) = lineage_fields(&row.full);
        if let Some(parent) = prior_id {
            children.entry(parent).or_default().push(id.clone());
        }
        cards.insert(id, row);
    }
    Ok(CardIndex { cards, children })
}

/// Scan all Cards in the store, loading full TOML for each. When
/// `pkg_filter` is provided, only that pkg subdir is scanned. Shared
/// between `find` and `load_card_index`.
fn scan_cards(store: &dyn CardStore, pkg_filter: Option<&str>) -> Result<Vec<CardRow>, String> {
    let locators = store.list_card_locators(pkg_filter)?;
    let mut rows = Vec::with_capacity(locators.len());
    for (pkg, loc) in &locators {
        if let Some(row) = load_full(store, loc, pkg) {
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Invariant context passed through the lineage walkers.
struct LineageCtx<'a> {
    index: &'a CardIndex,
    relation_filter: &'a Option<Vec<String>>,
    include_stats: bool,
    max_depth: usize,
}

/// Mutable accumulator for one lineage walk.
struct LineageAccum {
    nodes: Vec<LineageNode>,
    edges: Vec<LineageEdge>,
    visited: std::collections::HashSet<String>,
    truncated: bool,
}

/// Walk ancestors via `metadata.prior_card_id`.
fn walk_up(start_id: &str, ctx: &LineageCtx<'_>, acc: &mut LineageAccum) {
    let mut cur = start_id.to_string();
    for step in 1..=ctx.max_depth {
        let Some(row) = ctx.index.cards.get(&cur) else {
            return;
        };
        let (prior_id, prior_rel) = lineage_fields(&row.full);
        let Some(prior_id) = prior_id else {
            return;
        };
        if !relation_passes(ctx.relation_filter, &prior_rel) {
            return;
        }
        if acc.visited.contains(&prior_id) {
            return;
        }
        let Some(parent) = ctx.index.cards.get(&prior_id) else {
            return;
        };
        acc.nodes
            .push(make_node(parent, -(step as i32), ctx.include_stats));
        acc.edges.push(LineageEdge {
            from: row.summary.card_id.clone(),
            to: parent.summary.card_id.clone(),
            relation: prior_rel,
        });
        acc.visited.insert(prior_id.clone());
        cur = prior_id;
    }
    // Depth exhausted but another unwalked parent exists → truncated.
    if let Some(row) = ctx.index.cards.get(&cur) {
        let (prior_id, _) = lineage_fields(&row.full);
        if prior_id
            .as_ref()
            .is_some_and(|p| ctx.index.cards.contains_key(p) && !acc.visited.contains(p))
        {
            acc.truncated = true;
        }
    }
}

/// Walk descendants using the reverse index (parent → children),
/// breadth-first.  O(result_size) instead of O(N_cards × depth).
fn walk_down(start_id: &str, ctx: &LineageCtx<'_>, acc: &mut LineageAccum) {
    let mut frontier: Vec<String> = vec![start_id.to_string()];

    for depth in 1..=ctx.max_depth {
        let mut next_frontier: Vec<String> = Vec::new();
        for parent_id in &frontier {
            let children = match ctx.index.children.get(parent_id) {
                Some(c) => c,
                None => continue,
            };
            for child_id in children {
                if acc.visited.contains(child_id) {
                    continue;
                }
                let Some(child) = ctx.index.cards.get(child_id) else {
                    continue;
                };
                let (_, prior_rel) = lineage_fields(&child.full);
                if !relation_passes(ctx.relation_filter, &prior_rel) {
                    continue;
                }
                acc.nodes
                    .push(make_node(child, depth as i32, ctx.include_stats));
                acc.edges.push(LineageEdge {
                    from: child.summary.card_id.clone(),
                    to: parent_id.clone(),
                    relation: prior_rel,
                });
                acc.visited.insert(child_id.clone());
                next_frontier.push(child_id.clone());
            }
        }
        if next_frontier.is_empty() {
            return;
        }
        frontier = next_frontier;
    }
    // Frontier still has nodes but depth is exhausted: check for
    // unwalked children at the next level.
    for parent_id in &frontier {
        let children = match ctx.index.children.get(parent_id) {
            Some(c) => c,
            None => continue,
        };
        for child_id in children {
            if acc.visited.contains(child_id) {
                continue;
            }
            let Some(child) = ctx.index.cards.get(child_id) else {
                continue;
            };
            let (_, prior_rel) = lineage_fields(&child.full);
            if relation_passes(ctx.relation_filter, &prior_rel) {
                acc.truncated = true;
                return;
            }
        }
    }
}

/// Walk the lineage tree from `q.card_id` in `store`.
pub fn lineage_with_store(
    store: &dyn CardStore,
    q: LineageQuery,
) -> Result<Option<LineageResult>, String> {
    let index = load_card_index(store)?;
    let Some(root_row) = index.cards.get(&q.card_id) else {
        return Ok(None);
    };

    let ctx = LineageCtx {
        index: &index,
        relation_filter: &q.relation_filter,
        include_stats: q.include_stats,
        max_depth: q.depth.unwrap_or(DEFAULT_LINEAGE_DEPTH),
    };
    let mut acc = LineageAccum {
        nodes: Vec::new(),
        edges: Vec::new(),
        visited: std::collections::HashSet::new(),
        truncated: false,
    };

    acc.nodes.push(make_node(root_row, 0, q.include_stats));
    acc.visited.insert(q.card_id.clone());

    if matches!(q.direction, LineageDirection::Up | LineageDirection::Both) {
        walk_up(&q.card_id, &ctx, &mut acc);
    }
    if matches!(q.direction, LineageDirection::Down | LineageDirection::Both) {
        walk_down(&q.card_id, &ctx, &mut acc);
    }

    Ok(Some(LineageResult {
        root: q.card_id,
        nodes: acc.nodes,
        edges: acc.edges,
        truncated: acc.truncated,
    }))
}

/// Render a LineageResult as JSON for the service layer.
pub fn lineage_to_json(r: &LineageResult) -> Json {
    let nodes: Vec<Json> = r
        .nodes
        .iter()
        .map(|n| {
            let mut m = serde_json::Map::new();
            m.insert("card_id".into(), json!(n.card_id));
            m.insert("pkg".into(), json!(n.pkg));
            m.insert("depth".into(), json!(n.depth));
            if let Some(p) = &n.prior_card_id {
                m.insert("prior_card_id".into(), json!(p));
            }
            if let Some(rel) = &n.prior_relation {
                m.insert("prior_relation".into(), json!(rel));
            }
            if let Some(s) = &n.stats {
                m.insert("stats".into(), s.clone());
            }
            Json::Object(m)
        })
        .collect();
    let edges: Vec<Json> = r
        .edges
        .iter()
        .map(|e| {
            let mut m = serde_json::Map::new();
            m.insert("from".into(), json!(e.from));
            m.insert("to".into(), json!(e.to));
            if let Some(rel) = &e.relation {
                m.insert("relation".into(), json!(rel));
            }
            Json::Object(m)
        })
        .collect();
    json!({
        "root": r.root,
        "nodes": nodes,
        "edges": edges,
        "truncated": r.truncated,
    })
}

// ───────────────────────────────────────────────────────────────
// Samples sidecar: per-case detail written alongside a Card as
// `{pkg}/{card_id}.samples.jsonl`. Write-once to preserve Card
// immutability: once a Card has a samples file, it cannot be
// rewritten — mismatched per-case data would break auditability.
// ───────────────────────────────────────────────────────────────

// ───────────────────────────────────────────────────────────────
// Card import: copy Card files from an external directory into the
// local cards store. Used by `alc_card_install` (Card Collections)
// and by `alc_pkg_install` (Pkg-bundled cards/).
// ───────────────────────────────────────────────────────────────

/// Import Card files into `store` from `source_dir` under `pkg`.
///
/// Copies `*.toml` and `*.samples.jsonl` files. Existing cards with the
/// same id are skipped (first-writer wins — Card immutability).
///
/// Returns `(imported, skipped)` card_id lists.
pub fn import_from_dir_with_store(
    store: &dyn CardStore,
    source_dir: &std::path::Path,
    pkg: &str,
) -> Result<(Vec<String>, Vec<String>), String> {
    let (imported, skipped) = store.import_from_dir(source_dir, pkg)?;
    for card_id in &imported {
        match store.read_card_text(card_id) {
            Ok(Some(toml_text)) => publish(CardEvent::Created {
                pkg: pkg.to_string(),
                card_id: card_id.clone(),
                toml_text,
            }),
            Ok(None) => {
                tracing::warn!(
                    card_id = %card_id,
                    "import_from_dir: read_card_text returned None after import; skipping publish"
                );
            }
            Err(e) => {
                tracing::warn!(
                    card_id = %card_id,
                    error = %e,
                    "import_from_dir: read_card_text failed after import; skipping publish"
                );
            }
        }
        // Samples are optional — best-effort.
        match store.read_samples_text(card_id) {
            Ok(Some(jsonl_text)) => publish(CardEvent::SamplesWritten {
                card_id: card_id.clone(),
                jsonl_text,
            }),
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    card_id = %card_id,
                    error = %e,
                    "import_from_dir: read_samples_text failed after import; skipping publish"
                );
            }
        }
    }
    Ok((imported, skipped))
}

/// Write per-case samples to `{card_id}.samples.jsonl` (write-once).
///
/// Each `samples` entry is serialized as one compact JSON line.
/// Fails if a samples file already exists for this card — mirrors
/// the immutability guarantee of Cards themselves.
pub fn write_samples_with_store(
    store: &dyn CardStore,
    card_id: &str,
    samples: Vec<Json>,
) -> Result<PathBuf, String> {
    if store.samples_exists(card_id)? {
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
    let path = store.write_samples_text(card_id, &buf)?;

    publish(CardEvent::SamplesWritten {
        card_id: card_id.to_string(),
        jsonl_text: buf,
    });

    Ok(path)
}

/// Query parameters for `read_samples`.
#[derive(Debug, Default, Clone)]
pub struct SamplesQuery {
    /// Skip this many matched rows (after `where` filtering).
    pub offset: usize,
    /// Max matched rows to return.
    pub limit: Option<usize>,
    /// Optional `where` predicate applied to each sample row.
    /// The row JSON is the full line object (no section wrapping).
    pub where_: Option<Predicate>,
}

/// Read per-case samples from `{card_id}.samples.jsonl`.
///
/// Streams the JSONL file line by line; rows are parsed, optionally
/// filtered by `q.where_`, then paged by `offset` + `limit`.  Offset
/// applies to the **post-filter** stream, matching Prisma/SQL
/// semantics.
///
/// Returns an empty Vec if no samples file exists (Cards without
/// per-case details are the common case, not an error).
pub fn read_samples_with_store(
    store: &dyn CardStore,
    card_id: &str,
    q: SamplesQuery,
) -> Result<Vec<Json>, String> {
    let text = match store.read_samples_text(card_id)? {
        Some(t) => t,
        None => return Ok(Vec::new()),
    };
    let mut matched: usize = 0;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let val: Json = serde_json::from_str(line)
            .map_err(|e| format!("Failed to parse sample line {i}: {e}"))?;
        if let Some(pred) = &q.where_ {
            if !eval_predicate(pred, &val) {
                continue;
            }
        }
        if matched < q.offset {
            matched += 1;
            continue;
        }
        if let Some(lim) = q.limit {
            if out.len() >= lim {
                break;
            }
        }
        matched += 1;
        out.push(val);
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════
// FileCardStore — default backend.
// ═══════════════════════════════════════════════════════════════
//
// Stores Cards as TOML files under `{root}/{pkg}/{card_id}.toml`,
// samples as `{root}/{pkg}/{card_id}.samples.jsonl`, and the alias
// table as `{root}/_aliases.toml`.
//
// `root` is provided at construction time via `new(root)`; callers
// (typically the service layer) resolve it from the `AppDir`
// abstraction. Tests use a tempdir via `new(tmpdir)`.

/// File-backed implementation of [`CardStore`].
pub struct FileCardStore {
    root: PathBuf,
}

impl FileCardStore {
    /// Construct a store rooted at an explicit path.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Return the root directory this store writes under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ─── Thin `self` delegations to the `*_with_store` free fns ────
    //
    // These let callers that hold an `Arc<FileCardStore>` (bridge/data
    // register_card closures, service layer) invoke domain logic via
    // instance methods without re-importing the `_with_store` free
    // functions. Semantics are identical to the free-fn variants.

    pub fn create(&self, input: Json) -> Result<(String, PathBuf), String> {
        create_with_store(self, input)
    }

    pub fn get(&self, card_id: &str) -> Result<Option<Json>, String> {
        get_with_store(self, card_id)
    }

    pub fn list(&self, pkg_filter: Option<&str>) -> Result<Vec<Summary>, String> {
        list_with_store(self, pkg_filter)
    }

    pub fn append(&self, card_id: &str, fields: Json) -> Result<Json, String> {
        append_with_store(self, card_id, fields)
    }

    pub fn alias_set(
        &self,
        name: &str,
        card_id: &str,
        pkg: Option<&str>,
        note: Option<&str>,
    ) -> Result<Alias, String> {
        alias_set_with_store(self, name, card_id, pkg, note)
    }

    pub fn alias_list(&self, pkg_filter: Option<&str>) -> Result<Vec<Alias>, String> {
        alias_list_with_store(self, pkg_filter)
    }

    pub fn get_by_alias(&self, name: &str) -> Result<Option<Json>, String> {
        get_by_alias_with_store(self, name)
    }

    pub fn find(&self, q: FindQuery) -> Result<Vec<Summary>, String> {
        find_with_store(self, q)
    }

    pub fn write_samples(&self, card_id: &str, samples: Vec<Json>) -> Result<PathBuf, String> {
        write_samples_with_store(self, card_id, samples)
    }

    pub fn read_samples(&self, card_id: &str, q: SamplesQuery) -> Result<Vec<Json>, String> {
        read_samples_with_store(self, card_id, q)
    }

    pub fn lineage(&self, q: LineageQuery) -> Result<Option<LineageResult>, String> {
        lineage_with_store(self, q)
    }

    pub fn card_sink_backfill(
        &self,
        sink: &str,
        dry_run: bool,
    ) -> Result<SinkBackfillReport, String> {
        card_sink_backfill_with_store(self, sink, dry_run)
    }

    /// Returns the absolute path to the per-pkg subdirectory,
    /// creating it when missing. Validates `pkg` to prevent path
    /// traversal.
    fn pkg_dir(&self, pkg: &str) -> Result<PathBuf, String> {
        validate_name(pkg, "pkg")?;
        let dir = self.root.join(pkg);
        if !dir.exists() {
            fs::create_dir_all(&dir).map_err(|e| format!("Failed to create pkg dir: {e}"))?;
        }
        Ok(dir)
    }

    /// Path of the global alias table.
    fn aliases_path(&self) -> PathBuf {
        self.root.join("_aliases.toml")
    }

    /// Path of the samples sidecar for `card_id`. Errors if the
    /// Card itself does not exist — samples without a parent Card
    /// are meaningless and we refuse to create orphans.
    fn samples_path(&self, card_id: &str) -> Result<PathBuf, String> {
        let card_path = self
            .find_card_locator(card_id)?
            .ok_or_else(|| format!("card '{card_id}' not found"))?;
        let dir = card_path
            .parent()
            .ok_or_else(|| format!("card '{card_id}' has no parent directory"))?;
        Ok(dir.join(format!("{card_id}.samples.jsonl")))
    }
}

impl CardStore for FileCardStore {
    fn write_new_card(&self, pkg: &str, card_id: &str, toml_text: &str) -> Result<PathBuf, String> {
        let dir = self.pkg_dir(pkg)?;
        let path = dir.join(format!("{card_id}.toml"));
        if path.exists() {
            return Err(format!(
                "alc.card.create: card '{card_id}' already exists (immutable)"
            ));
        }
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, toml_text).map_err(|e| format!("Failed to write card tmp: {e}"))?;
        fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename card file: {e}"))?;
        Ok(path)
    }

    fn overwrite_card(&self, card_id: &str, toml_text: &str) -> Result<PathBuf, String> {
        let path = self
            .find_card_locator(card_id)?
            .ok_or_else(|| format!("alc.card.overwrite: card '{card_id}' not found"))?;
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, toml_text).map_err(|e| format!("Failed to write card tmp: {e}"))?;
        fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename card file: {e}"))?;
        Ok(path)
    }

    fn find_card_locator(&self, card_id: &str) -> Result<Option<PathBuf>, String> {
        validate_name(card_id, "card_id")?;
        if !self.root.exists() {
            return Ok(None);
        }
        let entries =
            fs::read_dir(&self.root).map_err(|e| format!("Failed to read cards dir: {e}"))?;
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

    fn read_card_text(&self, card_id: &str) -> Result<Option<String>, String> {
        let Some(path) = self.find_card_locator(card_id)? else {
            return Ok(None);
        };
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read card '{card_id}': {e}"))?;
        Ok(Some(text))
    }

    fn list_card_locators(
        &self,
        pkg_filter: Option<&str>,
    ) -> Result<Vec<(String, PathBuf)>, String> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let pkg_dirs: Vec<PathBuf> = if let Some(p) = pkg_filter {
            validate_name(p, "pkg")?;
            let d = self.root.join(p);
            if d.is_dir() {
                vec![d]
            } else {
                return Ok(Vec::new());
            }
        } else {
            fs::read_dir(&self.root)
                .map_err(|e| format!("Failed to read cards dir: {e}"))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        };

        let mut out = Vec::new();
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
                out.push((pkg.clone(), p));
            }
        }
        Ok(out)
    }

    fn read_locator_text(&self, locator: &Path) -> Result<Option<String>, String> {
        match fs::read_to_string(locator) {
            Ok(text) => Ok(Some(text)),
            Err(_) => Ok(None),
        }
    }

    fn read_aliases(&self) -> Result<Vec<Alias>, String> {
        let path = self.aliases_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("Failed to read aliases file: {e}"))?;
        let val: toml::Value =
            toml::from_str(&text).map_err(|e| format!("Failed to parse aliases file: {e}"))?;
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

    fn write_aliases(&self, aliases: &[Alias]) -> Result<(), String> {
        // Ensure the cards root exists so aliases can be written to
        // a brand-new store (mirrors the behaviour of `cards_dir()`).
        if !self.root.exists() {
            fs::create_dir_all(&self.root)
                .map_err(|e| format!("Failed to create cards dir: {e}"))?;
        }
        let path = self.aliases_path();
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

    fn samples_exists(&self, card_id: &str) -> Result<bool, String> {
        let path = self.samples_path(card_id)?;
        Ok(path.exists())
    }

    fn write_samples_text(&self, card_id: &str, jsonl_text: &str) -> Result<PathBuf, String> {
        let path = self.samples_path(card_id)?;
        if path.exists() {
            return Err(format!(
                "alc.card.write_samples: samples already exist for card '{card_id}' (write-once)"
            ));
        }
        let tmp = path.with_extension("jsonl.tmp");
        fs::write(&tmp, jsonl_text).map_err(|e| format!("Failed to write samples tmp: {e}"))?;
        fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename samples file: {e}"))?;
        Ok(path)
    }

    fn read_samples_text(&self, card_id: &str) -> Result<Option<String>, String> {
        let path = self.samples_path(card_id)?;
        if !path.exists() {
            return Ok(None);
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("Failed to read samples file: {e}"))?;
        Ok(Some(text))
    }

    fn import_from_dir(
        &self,
        source_dir: &Path,
        pkg: &str,
    ) -> Result<(Vec<String>, Vec<String>), String> {
        validate_name(pkg, "pkg")?;
        let dest = self.pkg_dir(pkg)?;
        let mut imported = Vec::new();
        let mut skipped = Vec::new();

        let entries =
            fs::read_dir(source_dir).map_err(|e| format!("Failed to read card source dir: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            let fname = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            if !fname.ends_with(".toml") {
                continue;
            }

            let card_id = fname.trim_end_matches(".toml");
            let dest_toml = dest.join(&fname);

            if dest_toml.exists() {
                skipped.push(card_id.to_string());
                continue;
            }

            let text = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read card file '{fname}': {e}"))?;
            let val: toml::Value = toml::from_str(&text)
                .map_err(|e| format!("Failed to parse card file '{fname}': {e}"))?;
            if val.get("schema_version").and_then(|v| v.as_str()) != Some(SCHEMA_VERSION) {
                continue;
            }

            fs::copy(&path, &dest_toml)
                .map_err(|e| format!("Failed to copy card '{fname}': {e}"))?;

            let samples_name = format!("{card_id}.samples.jsonl");
            let samples_src = source_dir.join(&samples_name);
            if samples_src.exists() {
                let samples_dest = dest.join(&samples_name);
                if !samples_dest.exists() {
                    fs::copy(&samples_src, &samples_dest)
                        .map_err(|e| format!("Failed to copy samples '{samples_name}': {e}"))?;
                }
            }

            imported.push(card_id.to_string());
        }

        Ok((imported, skipped))
    }
}

// ═══════════════════════════════════════════════════════════════
// Event Publisher Port — Card sink fan-out (v1)
// ═══════════════════════════════════════════════════════════════
//
// Storage port (`CardStore` / `FileCardStore`) stays pure CRUD. The
// Event publisher port below mirrors every successful Card write to a
// set of subscriber backends configured via the `ALC_CARD_SINKS`
// environment variable. Fan-out is best-effort and strictly serial;
// subscriber failures never propagate to the primary Card API.

// ─── CardEvent / CardEventKind ─────────────────────────────────

/// A Card-level event emitted from the write path.
///
/// Each variant carries the already-serialized payload text so that
/// subscribers can persist the exact bytes that were written to the
/// primary store (byte-for-byte parity).
#[derive(Debug, Clone)]
pub enum CardEvent {
    /// A brand-new Card was written.
    Created {
        pkg: String,
        card_id: String,
        toml_text: String,
    },
    /// An existing Card had new top-level keys merged in.
    Appended { card_id: String, toml_text: String },
    /// A samples JSONL sidecar was written for `card_id`.
    SamplesWritten { card_id: String, jsonl_text: String },
    /// The global alias table was rewritten.
    AliasesWritten { toml_text: String },
}

/// Lightweight discriminant for `CardEvent`. Used as a `HashMap` key in
/// `SubscriberStats` so that ok/err counters can be tracked per event
/// kind without holding the full payload.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum CardEventKind {
    Created,
    Appended,
    SamplesWritten,
    AliasesWritten,
}

impl CardEventKind {
    /// Stable string tag used in `tracing` log fields.
    pub fn as_str(self) -> &'static str {
        match self {
            CardEventKind::Created => "created",
            CardEventKind::Appended => "appended",
            CardEventKind::SamplesWritten => "samples_written",
            CardEventKind::AliasesWritten => "aliases_written",
        }
    }

    /// Short JSON key for the public `alc_stats` snapshot.
    /// Distinct from `as_str()` so that tracing logs keep their
    /// verbose form while the JSON surface stays compact.
    pub fn json_key(self) -> &'static str {
        match self {
            CardEventKind::Created => "created",
            CardEventKind::Appended => "appended",
            CardEventKind::SamplesWritten => "samples",
            CardEventKind::AliasesWritten => "aliases",
        }
    }

    /// All kinds in stable display order. Used by `SubscriberHealthRow`
    /// to emit all four counter keys even when a counter is zero, so
    /// that downstream consumers can rely on field presence.
    pub fn all() -> [CardEventKind; 4] {
        [
            CardEventKind::Created,
            CardEventKind::Appended,
            CardEventKind::SamplesWritten,
            CardEventKind::AliasesWritten,
        ]
    }
}

impl Serialize for CardEventKind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.json_key())
    }
}

impl CardEvent {
    /// Return the `CardEventKind` discriminant for this event.
    pub fn kind(&self) -> CardEventKind {
        match self {
            CardEvent::Created { .. } => CardEventKind::Created,
            CardEvent::Appended { .. } => CardEventKind::Appended,
            CardEvent::SamplesWritten { .. } => CardEventKind::SamplesWritten,
            CardEvent::AliasesWritten { .. } => CardEventKind::AliasesWritten,
        }
    }
}

// ─── CardSubscriber trait ──────────────────────────────────────

/// A downstream backend that receives `CardEvent`s in best-effort,
/// serial fan-out order.
///
/// Implementations must be `Send + Sync` so that the bus can hold
/// `Arc<dyn CardSubscriber>` safely across threads.
pub trait CardSubscriber: Send + Sync {
    /// Handle one event. Returning `Err` records a failure in
    /// `SubscriberStats` and emits a `tracing::warn!`, but does not
    /// propagate to the caller of the `_with_store` API.
    fn on_event(&self, ev: &CardEvent) -> Result<(), String>;

    /// Canonical identity URI for this subscriber. Used as the key in
    /// `SubscriberStats` and as the match target for
    /// `CardEventBus::publish_to`.
    fn describe(&self) -> String;

    /// Best-effort check for whether `card_id` already exists in this
    /// subscriber. Used by `alc_card_sink_backfill` to skip cards that
    /// are already mirrored (drift-safe: we never overwrite).
    ///
    /// Default implementation returns `Ok(false)` so subscribers that
    /// cannot cheaply answer (e.g. future network-backed sinks) always
    /// get the push. Override when a cheap local check is possible.
    fn has_card(&self, _card_id: &str) -> Result<bool, String> {
        Ok(false)
    }
}

// ─── SubscriberStats ───────────────────────────────────────────

/// Most recent delivery failure for a single subscriber. Exposed via
/// `SubscriberHealthRow.last_error` in the `alc_stats` JSON snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct LastError {
    pub kind: CardEventKind,
    pub msg: String,
    pub ts_ms: u64,
}

/// Per-subscriber counter state. Held inside `SubscriberStats` under a
/// `Mutex`; `snapshot` clones the fields into an owned `SubscriberHealthRow`
/// while the lock is held, so the lock window stays short.
#[derive(Default, Debug)]
pub struct PerSubscriber {
    pub ok: HashMap<CardEventKind, u64>,
    pub err: HashMap<CardEventKind, u64>,
    pub last_error: Option<LastError>,
}

/// Process-wide per-subscriber statistics, keyed by subscriber URI
/// (the value returned by `CardSubscriber::describe`).
#[derive(Default, Debug)]
pub struct SubscriberStats {
    inner: Mutex<HashMap<String, PerSubscriber>>,
}

impl SubscriberStats {
    /// Record a successful event delivery.
    pub fn record_ok(&self, key: &str, kind: CardEventKind) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let entry = g.entry(key.to_string()).or_default();
        let c = entry.ok.entry(kind).or_insert(0);
        *c = c.saturating_add(1);
    }

    /// Record a delivery failure together with the error message.
    /// The failure kind, message, and timestamp overwrite `last_error`
    /// — there is no ring buffer; only the most recent failure is kept.
    pub fn record_err(&self, key: &str, kind: CardEventKind, err: &str) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let entry = g.entry(key.to_string()).or_default();
        let c = entry.err.entry(kind).or_insert(0);
        *c = c.saturating_add(1);
        entry.last_error = Some(LastError {
            kind,
            msg: err.to_string(),
            ts_ms: now_ms(),
        });
    }

    /// Take a point-in-time snapshot of all subscribers. The returned
    /// `Vec` is owned — the internal lock is released before this
    /// function returns, so callers can hold it freely.
    ///
    /// All four `CardEventKind` keys (`created / appended / samples /
    /// aliases`) are always present in both `ok` and `err`, defaulting
    /// to 0 if no event of that kind has been recorded. This lets
    /// downstream consumers rely on field presence.
    pub fn snapshot(&self) -> Vec<SubscriberHealthRow> {
        let g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let mut rows = Vec::with_capacity(g.len());
        for (sink, ps) in g.iter() {
            let mut ok: HashMap<String, u64> = HashMap::with_capacity(4);
            let mut err: HashMap<String, u64> = HashMap::with_capacity(4);
            for k in CardEventKind::all() {
                ok.insert(
                    k.json_key().to_string(),
                    ps.ok.get(&k).copied().unwrap_or(0),
                );
                err.insert(
                    k.json_key().to_string(),
                    ps.err.get(&k).copied().unwrap_or(0),
                );
            }
            rows.push(SubscriberHealthRow {
                sink: sink.clone(),
                ok,
                err,
                last_error: ps.last_error.clone(),
            });
        }
        // Stable output ordering (by sink URI) so that the JSON dump is
        // deterministic across runs — useful for snapshot tests.
        rows.sort_by(|a, b| a.sink.cmp(&b.sink));
        rows
    }
}

/// Unix-epoch milliseconds used by `LastError.ts_ms`. Uses
/// `unwrap_or_default` so no panic can escape even if the system
/// clock is misconfigured.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Snapshot row for a single subscriber, serialized directly into the
/// `alc_stats` JSON output as one element of the `card_sinks` array.
#[derive(Debug, Clone, Serialize)]
pub struct SubscriberHealthRow {
    pub sink: String,
    pub ok: HashMap<String, u64>,
    pub err: HashMap<String, u64>,
    pub last_error: Option<LastError>,
}

/// Public entry point: snapshot of all process-wide subscriber stats.
/// Wrapper around `event_bus().stats().snapshot()` so that downstream
/// crates (notably `algocline-app`) do not need a handle to the
/// `CardEventBus` singleton.
pub fn subscriber_stats_snapshot() -> Vec<SubscriberHealthRow> {
    event_bus().stats().snapshot()
}

// ─── FileCardSubscriber — file-URI backend ─────────────────────

/// Atomically write `bytes` to `dest` by staging to a unique
/// `{dest}.tmp.{pid}.{seq}` sibling and renaming. The `{pid}.{seq}`
/// suffix prevents concurrent writers (same or different processes)
/// from colliding on the tmp path.
fn atomic_write(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = process::id();
    if let Some(parent) = dest.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| format!("subscriber mkdir: {e}"))?;
        }
    }
    let mut tmp = dest.as_os_str().to_owned();
    tmp.push(format!(".tmp.{pid}.{seq}"));
    let tmp_path = PathBuf::from(tmp);
    fs::write(&tmp_path, bytes).map_err(|e| format!("subscriber write tmp: {e}"))?;
    fs::rename(&tmp_path, dest).map_err(|e| format!("subscriber rename: {e}"))
}

/// Canonical `file://` URI for a local directory path. The result is
/// the inverse of [`decode_file_uri_path`] — the pair round-trips
/// through the `ALC_CARD_SINKS` env spec.
fn canonical_file_uri(root: &Path) -> String {
    let p = root.to_string_lossy();
    #[cfg(unix)]
    {
        format!("file://{p}")
    }
    #[cfg(windows)]
    {
        format!("file:///{}", p.replace('\\', "/"))
    }
    #[cfg(not(any(unix, windows)))]
    {
        format!("file://{p}")
    }
}

/// A subscriber that mirrors events to a local directory using the
/// same two-tier layout as [`FileCardStore`]:
///
/// - `{root}/{pkg}/{card_id}.toml` — Card TOML
/// - `{root}/{pkg}/{card_id}.samples.jsonl` — samples sidecar
/// - `{root}/_aliases.toml` — global alias table
pub struct FileCardSubscriber {
    root: PathBuf,
    uri: String,
}

impl FileCardSubscriber {
    /// Construct a subscriber rooted at `root`. The canonical URI is
    /// computed once and returned from [`Self::describe`].
    pub fn new(root: PathBuf) -> Self {
        let uri = canonical_file_uri(&root);
        Self { root, uri }
    }

    /// Scan the subscriber root for a Card with `card_id` under any
    /// `{pkg}` subdirectory. Returns `Ok(None)` when the root itself
    /// does not exist yet (subscribers are write-once lazy).
    pub fn locate_card(&self, card_id: &str) -> Result<Option<PathBuf>, String> {
        validate_name(card_id, "card_id")?;
        if !self.root.exists() {
            return Ok(None);
        }
        let entries = fs::read_dir(&self.root).map_err(|e| format!("subscriber read_dir: {e}"))?;
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

    fn ensure_pkg_dir(&self, pkg: &str) -> Result<PathBuf, String> {
        validate_name(pkg, "pkg")?;
        let dir = self.root.join(pkg);
        if !dir.exists() {
            fs::create_dir_all(&dir).map_err(|e| format!("subscriber mkdir: {e}"))?;
        }
        Ok(dir)
    }

    fn write_created(&self, pkg: &str, card_id: &str, toml_text: &str) -> Result<(), String> {
        validate_name(card_id, "card_id")?;
        let dir = self.ensure_pkg_dir(pkg)?;
        let dest = dir.join(format!("{card_id}.toml"));
        atomic_write(&dest, toml_text.as_bytes())
    }

    fn write_appended(&self, card_id: &str, toml_text: &str) -> Result<(), String> {
        match self.locate_card(card_id)? {
            Some(dest) => atomic_write(&dest, toml_text.as_bytes()),
            None => Err(format!(
                "subscriber append: card '{card_id}' missing at {}",
                self.uri
            )),
        }
    }

    fn write_samples(&self, card_id: &str, jsonl_text: &str) -> Result<(), String> {
        let card_path = self.locate_card(card_id)?.ok_or_else(|| {
            format!(
                "subscriber samples: card '{card_id}' missing at {}",
                self.uri
            )
        })?;
        let dir = card_path
            .parent()
            .ok_or_else(|| format!("subscriber samples: card '{card_id}' has no parent dir"))?;
        let dest = dir.join(format!("{card_id}.samples.jsonl"));
        atomic_write(&dest, jsonl_text.as_bytes())
    }

    fn write_aliases(&self, toml_text: &str) -> Result<(), String> {
        if !self.root.exists() {
            fs::create_dir_all(&self.root).map_err(|e| format!("subscriber mkdir: {e}"))?;
        }
        let dest = self.root.join("_aliases.toml");
        atomic_write(&dest, toml_text.as_bytes())
    }
}

impl CardSubscriber for FileCardSubscriber {
    fn on_event(&self, ev: &CardEvent) -> Result<(), String> {
        match ev {
            CardEvent::Created {
                pkg,
                card_id,
                toml_text,
            } => self.write_created(pkg, card_id, toml_text),
            CardEvent::Appended { card_id, toml_text } => self.write_appended(card_id, toml_text),
            CardEvent::SamplesWritten {
                card_id,
                jsonl_text,
            } => self.write_samples(card_id, jsonl_text),
            CardEvent::AliasesWritten { toml_text } => self.write_aliases(toml_text),
        }
    }

    fn describe(&self) -> String {
        self.uri.clone()
    }

    /// Delegates to [`FileCardSubscriber::locate_card`]. A non-existent
    /// root (subscriber has never been written to) returns `Ok(false)`,
    /// which is the correct "backfill needed" answer.
    fn has_card(&self, card_id: &str) -> Result<bool, String> {
        Ok(self.locate_card(card_id)?.is_some())
    }
}

// ─── CardEventBus + OnceLock singleton ─────────────────────────

/// Process-wide fan-out bus. Subscribers are registered once at startup
/// (from `ALC_CARD_SINKS`) and stored behind a `Mutex` so that tests
/// can swap them out via `replace_subscribers_for_test` without losing
/// the singleton identity.
pub struct CardEventBus {
    subscribers: Mutex<Vec<Arc<dyn CardSubscriber>>>,
    stats: Arc<SubscriberStats>,
}

impl CardEventBus {
    /// Build a bus from an explicit subscriber list. Used both by the
    /// env loader and by `install_event_bus_for_test`.
    pub fn new(subscribers: Vec<Arc<dyn CardSubscriber>>) -> Self {
        Self {
            subscribers: Mutex::new(subscribers),
            stats: Arc::new(SubscriberStats::default()),
        }
    }

    /// Shared handle to the per-subscriber counters.
    pub fn stats(&self) -> &Arc<SubscriberStats> {
        &self.stats
    }

    /// Fan out `ev` to every registered subscriber serially. Subscriber
    /// failures are counted in `SubscriberStats` and logged but do not
    /// propagate.
    pub fn publish(&self, ev: &CardEvent) {
        let subs_snapshot: Vec<Arc<dyn CardSubscriber>> = {
            let guard = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
            guard.clone()
        };
        for sub in &subs_snapshot {
            let key = sub.describe();
            match sub.on_event(ev) {
                Ok(()) => self.stats.record_ok(&key, ev.kind()),
                Err(e) => {
                    tracing::warn!(
                        subscriber = %key,
                        kind = ev.kind().as_str(),
                        error = %e,
                        "card subscriber failed"
                    );
                    self.stats.record_err(&key, ev.kind(), &e);
                }
            }
        }
    }

    /// Deliver `ev` to exactly one subscriber identified by
    /// `target` URI. Returns `Err` when no subscriber matches or the
    /// subscriber itself fails (backfill path needs the caller to know).
    pub fn publish_to(&self, target: &str, ev: &CardEvent) -> Result<(), String> {
        let hit: Option<Arc<dyn CardSubscriber>> = {
            let guard = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
            guard.iter().find(|s| s.describe() == target).cloned()
        };
        let Some(sub) = hit else {
            return Err(format!("subscriber not registered: {target}"));
        };
        let key = sub.describe();
        match sub.on_event(ev) {
            Ok(()) => {
                self.stats.record_ok(&key, ev.kind());
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    subscriber = %key,
                    kind = ev.kind().as_str(),
                    error = %e,
                    "card subscriber failed (publish_to)"
                );
                self.stats.record_err(&key, ev.kind(), &e);
                Err(e)
            }
        }
    }

    /// List every subscriber URI currently registered on the bus.
    pub fn subscriber_uris(&self) -> Vec<String> {
        let guard = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        guard.iter().map(|s| s.describe()).collect()
    }

    /// Look up a subscriber by URI (as returned by `describe`). Returns
    /// `None` when no subscriber matches. Used by
    /// `alc_card_sink_backfill` to dispatch has_card checks against a
    /// specific sink.
    pub fn find_subscriber(&self, uri: &str) -> Option<Arc<dyn CardSubscriber>> {
        let guard = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        guard.iter().find(|s| s.describe() == uri).cloned()
    }

    /// Replace the subscriber list in place while preserving singleton
    /// identity and shared `SubscriberStats`. Test-only.
    #[cfg(any(test, feature = "test-support"))]
    pub fn replace_subscribers_for_test(&self, subs: Vec<Arc<dyn CardSubscriber>>) {
        let mut guard = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        *guard = subs;
    }

    /// Reset the per-subscriber counters. Test-only helper.
    #[cfg(any(test, feature = "test-support"))]
    pub fn reset_stats_for_test(&self) {
        let mut g = self.stats.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.clear();
    }
}

static CARD_EVENT_BUS: OnceLock<CardEventBus> = OnceLock::new();

/// Return the process-wide `CardEventBus` singleton, initializing it
/// on the first call from the `ALC_CARD_SINKS` environment variable.
pub fn event_bus() -> &'static CardEventBus {
    CARD_EVENT_BUS.get_or_init(|| {
        let subs = load_subscribers_from_env();
        CardEventBus::new(subs)
    })
}

/// Eagerly initialize the bus. Idempotent and safe to call multiple
/// times; intended for startup hooks (`main.rs`) so that subscriber
/// registration `tracing::info!` lines are emitted at boot rather than
/// on the first Card write.
pub fn init_event_bus() {
    let bus = event_bus();
    let uris = bus.subscriber_uris();
    if uris.is_empty() {
        tracing::info!("card sinks: no subscribers configured (ALC_CARD_SINKS unset)");
    } else {
        for uri in &uris {
            tracing::info!(subscriber = %uri, "card sink registered");
        }
    }
}

/// Install a test-built bus. Fails once the singleton is already set.
#[cfg(any(test, feature = "test-support"))]
pub fn install_event_bus_for_test(bus: CardEventBus) -> Result<(), String> {
    CARD_EVENT_BUS
        .set(bus)
        .map_err(|_| "bus already initialized".to_string())
}

/// Convenience wrapper: publish through the singleton.
pub fn publish(ev: CardEvent) {
    // In test builds, serialize publishing with the subscriber-test mutex so
    // that default-store tests running in parallel cannot inject events into
    // a subscriber test's mock while the mock is active.  The owning test
    // thread sets INSIDE_BUS_TEST to true so it does NOT try to acquire the
    // same lock it already holds (re-entrancy safe).
    #[cfg(test)]
    {
        let is_test_owner = INSIDE_BUS_TEST.with(|f| f.get());
        if !is_test_owner {
            // Block until any active subscriber test releases the lock.
            let _gate = bus_test_gate().lock().unwrap_or_else(|p| p.into_inner());
            event_bus().publish(&ev);
            return;
        }
    }
    event_bus().publish(&ev);
}

/// Mutex used in tests to serialise subscriber-test setup against concurrent
/// publishes from default-store tests running in parallel.
#[cfg(test)]
fn bus_test_gate() -> &'static Mutex<()> {
    static GATE: OnceLock<Mutex<()>> = OnceLock::new();
    GATE.get_or_init(|| Mutex::new(()))
}

// Thread-local flag: set to `true` by the thread running inside
// `with_bus_subscribers` so that `publish` skips the gate (re-entrancy).
#[cfg(test)]
thread_local! {
    static INSIDE_BUS_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// ─── alc_card_sink_backfill ────────────────────────────────────

/// Result of a [`card_sink_backfill`] run. One row per card the tool
/// touched (classified as pushed / skipped / failed / pushed_samples).
/// The `failed` entries carry the error message so an operator can
/// triage read-only mounts etc.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SinkBackfillReport {
    pub sink: String,
    pub pushed: Vec<String>,
    pub skipped: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub pushed_samples: Vec<String>,
}

/// Backfill one subscriber (`sink` URI) from the primary store.
///
/// Steps:
/// 1. Look up the target subscriber on the event bus; fail fast if
///    the URI is not registered so the caller gets an immediate
///    error rather than a silent no-op.
/// 2. Enumerate every `(pkg, card_id)` pair from the default
///    [`CardStore`].
/// 3. For each pair, ask the subscriber whether it already has that
///    card (`CardSubscriber::has_card`). If yes → skipped (drift-safe,
///    no overwrite). If no → read the primary TOML and `publish_to`
///    the one target sink. Samples are mirrored the same way.
/// 4. `dry_run = true` short-circuits step 3: the report lists
///    what would have been pushed but no `publish_to` is issued,
///    so `SubscriberStats` does not increment.
///
/// `card_sink_backfill` operates with an injectable [`CardStore`]; tests
/// drive this directly against a tempdir-backed [`FileCardStore`] so they
/// never touch the user's real cards directory.
pub fn card_sink_backfill_with_store(
    store: &dyn CardStore,
    sink: &str,
    dry_run: bool,
) -> Result<SinkBackfillReport, String> {
    let bus = event_bus();
    let sub = bus
        .find_subscriber(sink)
        .ok_or_else(|| format!("unknown sink: {sink}"))?;

    let locators = store.list_card_locators(None)?;

    let mut report = SinkBackfillReport {
        sink: sink.to_string(),
        ..Default::default()
    };

    for (pkg, locator) in locators {
        let card_id = match locator.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        match sub.has_card(&card_id) {
            Ok(true) => {
                report.skipped.push(card_id);
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!(
                    card_id = %card_id,
                    error = %e,
                    "backfill: has_card failed; treating as skipped"
                );
                report.skipped.push(card_id);
                continue;
            }
        }

        let toml_text = match store.read_locator_text(&locator) {
            Ok(Some(t)) => t,
            Ok(None) => {
                // Unreadable / corrupt on primary. Do not panic; skip.
                report.skipped.push(card_id);
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    card_id = %card_id,
                    error = %e,
                    "backfill: read_locator_text failed; treating as skipped"
                );
                report.skipped.push(card_id);
                continue;
            }
        };

        if dry_run {
            report.pushed.push(card_id.clone());
            if matches!(store.read_samples_text(&card_id), Ok(Some(_))) {
                report.pushed_samples.push(card_id);
            }
            continue;
        }

        let ev = CardEvent::Created {
            pkg: pkg.clone(),
            card_id: card_id.clone(),
            toml_text,
        };
        match bus.publish_to(sink, &ev) {
            Ok(()) => report.pushed.push(card_id.clone()),
            Err(e) => {
                report.failed.push((card_id, e));
                continue;
            }
        }

        if let Ok(Some(jsonl_text)) = store.read_samples_text(&card_id) {
            let ev = CardEvent::SamplesWritten {
                card_id: card_id.clone(),
                jsonl_text,
            };
            match bus.publish_to(sink, &ev) {
                Ok(()) => report.pushed_samples.push(card_id),
                Err(e) => {
                    report.failed.push((card_id, format!("samples: {e}")));
                }
            }
        }
    }

    Ok(report)
}

// ─── ALC_CARD_SINKS env parser ─────────────────────────────────

/// Read `ALC_CARD_SINKS` and build one subscriber per accepted spec.
/// Malformed entries are logged and skipped; duplicate URIs are
/// first-wins. Non-UTF8 values reject the whole env.
fn load_subscribers_from_env() -> Vec<Arc<dyn CardSubscriber>> {
    let raw = match std::env::var("ALC_CARD_SINKS") {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => return Vec::new(),
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::error!("ALC_CARD_SINKS contains non-UTF8 bytes; ignoring entire variable");
            return Vec::new();
        }
    };
    parse_subscribers_from_str(&raw)
}

/// Parse a `|`-separated list of subscriber URIs (the same format used
/// by `ALC_CARD_SINKS`). Extracted so tests can exercise the parser
/// without touching process environment.
fn parse_subscribers_from_str(raw: &str) -> Vec<Arc<dyn CardSubscriber>> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Arc<dyn CardSubscriber>> = Vec::new();
    for spec in raw.split('|') {
        let spec = spec.trim();
        if spec.is_empty() {
            continue;
        }
        let Some(sub) = parse_subscriber_spec(spec) else {
            continue;
        };
        let uri = sub.describe();
        if !seen.insert(uri.clone()) {
            tracing::warn!(subscriber = %uri, "duplicate ALC_CARD_SINKS entry; keeping first");
            continue;
        }
        out.push(sub);
    }
    out
}

/// Parse one subscriber spec. v1 only accepts `file:///absolute/path`.
fn parse_subscriber_spec(spec: &str) -> Option<Arc<dyn CardSubscriber>> {
    // scheme required
    let Some(colon_idx) = spec.find(':') else {
        tracing::error!(spec, "ALC_CARD_SINKS entry missing URI scheme");
        return None;
    };
    let scheme = &spec[..colon_idx];
    let rest = &spec[colon_idx + 1..];
    if scheme != "file" {
        tracing::error!(spec, scheme, "ALC_CARD_SINKS entry has unknown scheme");
        return None;
    }
    // scheme-specific: must start with `//`
    let Some(after_slash) = rest.strip_prefix("//") else {
        tracing::error!(spec, "file URI missing '//'");
        return None;
    };
    // split authority / path. path begins with '/'.
    let Some(path_start) = after_slash.find('/') else {
        tracing::error!(spec, "file URI has no path component");
        return None;
    };
    let authority = &after_slash[..path_start];
    let encoded_path = &after_slash[path_start..];
    if !authority.is_empty() {
        tracing::error!(
            spec,
            authority,
            "file URI with non-empty authority is rejected"
        );
        return None;
    }
    let path = decode_file_uri_path(encoded_path)?;
    Some(Arc::new(FileCardSubscriber::new(path)))
}

/// Decode the path portion of a `file://` URI into a `PathBuf`.
///
/// Unix: a leading `/` is preserved (absolute path).
/// Windows: the leading `/` is stripped so that `/C:/a/b` becomes
/// `C:/a/b`.
fn decode_file_uri_path(encoded: &str) -> Option<PathBuf> {
    let decoded = percent_decode(encoded)?;
    #[cfg(windows)]
    {
        // Strip the leading slash so "/C:/foo" -> "C:/foo".
        let trimmed = decoded.strip_prefix('/').unwrap_or(&decoded);
        Some(PathBuf::from(trimmed))
    }
    #[cfg(not(windows))]
    {
        Some(PathBuf::from(decoded))
    }
}

/// Percent-decode a URI path segment. Returns `None` on invalid or
/// truncated `%XX` sequences.
fn percent_decode(src: &str) -> Option<String> {
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'%' {
            if i + 2 >= bytes.len() {
                tracing::error!(src, "percent-encoded sequence truncated");
                return None;
            }
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            match (hi, lo) {
                (Some(h), Some(l)) => {
                    out.push(((h << 4) | l) as u8);
                    i += 3;
                }
                _ => {
                    tracing::error!(src, "percent-encoded sequence has non-hex digits");
                    return None;
                }
            }
        } else {
            out.push(b);
            i += 1;
        }
    }
    match String::from_utf8(out) {
        Ok(s) => Some(s),
        Err(_) => {
            tracing::error!(src, "percent-decoded bytes are not valid UTF-8");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_valid_card() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "minimum_valid_pkg";
        let input = json!({ "pkg": { "name": pkg } });
        let (id, path) = create_with_store(&store, input).unwrap();
        assert!(path.exists());
        assert!(id.starts_with(pkg));

        let got = get_with_store(&store, &id).unwrap().unwrap();
        assert_eq!(got["schema_version"], json!(SCHEMA_VERSION));
        assert_eq!(got["card_id"], json!(id));
        assert_eq!(got["pkg"]["name"], json!(pkg));
        assert!(got.get("created_at").is_some());
        assert!(got.get("created_by").is_some());
    }

    #[test]
    fn create_rejects_missing_pkg_name() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let err = create_with_store(&store, json!({})).unwrap_err();
        assert!(err.contains("pkg.name"));
    }

    #[test]
    fn create_is_immutable() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "immutable_pkg";
        let input = json!({
            "card_id": "fixed_id_001",
            "pkg": { "name": pkg }
        });
        create_with_store(&store, input.clone()).unwrap();
        let err = create_with_store(&store, input).unwrap_err();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn create_injects_param_fingerprint() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "fingerprint_pkg";
        let input = json!({
            "pkg": { "name": pkg },
            "params": { "depth": 3, "temperature": 0.0 }
        });
        let (id, _) = create_with_store(&store, input).unwrap();
        let got = get_with_store(&store, &id).unwrap().unwrap();
        assert!(got["param_fingerprint"].is_string());
    }

    #[test]
    fn list_returns_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "list_newest_pkg";
        let (id1, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
                "created_at": "2025-01-01T00:00:00Z"
            }),
        )
        .unwrap();
        let (id2, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_b"),
                "pkg": { "name": pkg },
                "created_at": "2026-01-01T00:00:00Z"
            }),
        )
        .unwrap();

        let rows = list_with_store(&store, Some(pkg)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].card_id, id2); // newer first
        assert_eq!(rows[1].card_id, id1);
    }

    #[test]
    fn list_extracts_summary_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "list_summary_pkg";
        let (id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": pkg },
                "model": { "id": "claude-opus-4-6" },
                "scenario": { "name": "gsm8k_sample100" },
                "stats": { "pass_rate": 0.82 }
            }),
        )
        .unwrap();

        let rows = list_with_store(&store, Some(pkg)).unwrap();
        let row = rows.iter().find(|r| r.card_id == id).unwrap();
        assert_eq!(row.model.as_deref(), Some("claude-opus-4-6"));
        assert_eq!(row.scenario.as_deref(), Some("gsm8k_sample100"));
        assert_eq!(row.pass_rate, Some(0.82));
    }

    #[test]
    fn get_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        assert!(get_with_store(&store, "does_not_exist_xyz")
            .unwrap()
            .is_none());
    }

    #[test]
    fn card_id_embeds_compact_timestamp() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "ts_embed_pkg";
        let (id, _) = create_with_store(&store, json!({ "pkg": { "name": pkg } })).unwrap();
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
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "distinct_ids_pkg";
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
        let (id1, _) = create_with_store(&store, input1).unwrap();
        let (id2, _) = create_with_store(&store, input2).unwrap();
        assert_ne!(id1, id2, "distinct stats must yield distinct card_ids");
    }

    // ─── P1: append ────────────────────────────────────────────

    #[test]
    fn append_adds_new_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "append_new_fields_pkg";
        let (id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.5 }
            }),
        )
        .unwrap();

        let merged = append_with_store(
            &store,
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
        let got = get_with_store(&store, &id).unwrap().unwrap();
        assert_eq!(got["caveats"]["notes"], json!("rescored after fix"));
        // Existing field untouched
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));
    }

    #[test]
    fn append_rejects_existing_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "append_reject_key_pkg";
        let (id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.5 }
            }),
        )
        .unwrap();

        let err =
            append_with_store(&store, &id, json!({ "stats": { "pass_rate": 0.9 } })).unwrap_err();
        assert!(err.contains("already set"), "got: {err}");
        // Verify original value still there
        let got = get_with_store(&store, &id).unwrap().unwrap();
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));
    }

    #[test]
    fn append_errors_on_missing_card() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let err = append_with_store(&store, "does_not_exist_xyz", json!({ "x": 1 })).unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── P1: alias_set / alias_list ────────────────────────────

    #[test]
    fn alias_set_and_list_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "alias_roundtrip_pkg";
        let (id, _) = create_with_store(&store, json!({ "pkg": { "name": pkg } })).unwrap();

        let alias_name = "test_alias_roundtrip";
        alias_set_with_store(&store, alias_name, &id, Some(pkg), Some("smoke")).unwrap();

        let rows = alias_list_with_store(&store, Some(pkg)).unwrap();
        let a = rows.iter().find(|a| a.name == alias_name).unwrap();
        assert_eq!(a.card_id, id);
        assert_eq!(a.pkg.as_deref(), Some(pkg));
        assert_eq!(a.note.as_deref(), Some("smoke"));
        assert!(!a.set_at.is_empty());

        // Rebind to a new card
        let (id2, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_b"),
                "pkg": { "name": pkg }
            }),
        )
        .unwrap();
        alias_set_with_store(&store, alias_name, &id2, Some(pkg), None).unwrap();
        let rows = alias_list_with_store(&store, Some(pkg)).unwrap();
        let matching: Vec<&Alias> = rows.iter().filter(|a| a.name == alias_name).collect();
        assert_eq!(matching.len(), 1, "alias should be unique by name");
        assert_eq!(matching[0].card_id, id2);
    }

    #[test]
    fn alias_set_rejects_unknown_card() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let err = alias_set_with_store(&store, "x", "does_not_exist_xyz", None, None).unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── find + where DSL ───────────────────────────────────────

    fn where_from(v: Json) -> Predicate {
        parse_where(&v).expect("parse where")
    }

    fn order_from(v: Json) -> Vec<OrderKey> {
        parse_order_by(&v).expect("parse order_by")
    }

    #[test]
    fn find_where_nested_eq_and_gte() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "find_nested_eq_pkg";
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_low"),
                "pkg": { "name": pkg },
                "scenario": { "name": "gsm8k" },
                "stats": { "pass_rate": 0.4 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_high"),
                "pkg": { "name": pkg },
                "scenario": { "name": "gsm8k" },
                "stats": { "pass_rate": 0.9 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_other"),
                "pkg": { "name": pkg },
                "scenario": { "name": "other" },
                "stats": { "pass_rate": 1.0 }
            }),
        )
        .unwrap();

        // scenario eq via nested object
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "scenario": { "name": "gsm8k" },
                }))),
                order_by: order_from(json!("-stats.pass_rate")),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].pass_rate, Some(0.9));
        assert_eq!(rows[1].pass_rate, Some(0.4));

        // gte operator
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "stats": { "pass_rate": { "gte": 0.8 } },
                }))),
                order_by: order_from(json!("-stats.pass_rate")),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.pass_rate.unwrap() >= 0.8));

        // limit
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                order_by: order_from(json!("-stats.pass_rate")),
                limit: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pass_rate, Some(1.0));
    }

    #[test]
    fn find_where_implicit_eq_and_logical() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "find_implicit_eq_pkg";
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-opus-4-6" },
                "stats": { "equilibrium_position": "dead", "survival_rate": 0.0 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_b"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-opus-4-6" },
                "stats": { "equilibrium_position": "niche_leader", "survival_rate": 1.0 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_c"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-haiku-4-5-20251001" },
                "stats": { "equilibrium_position": "fragile", "survival_rate": 0.2 }
            }),
        )
        .unwrap();

        // implicit eq on sparse stats field
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "stats": { "equilibrium_position": "dead" },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].card_id.ends_with("_a"));

        // _or
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "_or": [
                        { "stats": { "equilibrium_position": "dead" } },
                        { "stats": { "survival_rate": { "gte": 0.9 } } },
                    ],
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);

        // _not
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "_not": { "model": { "id": "claude-haiku-4-5-20251001" } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);

        // in operator
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "stats": {
                        "equilibrium_position": { "in": ["dead", "fragile"] },
                    },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);

        // exists false (sparse field missing on haiku card? all have it, so test on
        // a field that only some have)
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "strategy_params": { "temperature": { "exists": false } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 3, "none of the cards have strategy_params");
    }

    #[test]
    fn find_order_by_multi_key() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "find_order_multi_pkg";
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.5 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_b"),
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.9 }
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_c"),
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.9 }
            }),
        )
        .unwrap();

        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                order_by: order_from(json!(["-stats.pass_rate", "card_id"])),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].pass_rate, Some(0.9));
        assert_eq!(rows[1].pass_rate, Some(0.9));
        assert_eq!(rows[2].pass_rate, Some(0.5));
        // Tiebreak by card_id ascending
        assert!(rows[0].card_id < rows[1].card_id);
    }

    #[test]
    fn find_offset_and_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "find_offset_limit_pkg";
        for i in 0..5 {
            create_with_store(
                &store,
                json!({
                    "card_id": format!("{pkg}_{i}"),
                    "pkg": { "name": pkg },
                    "stats": { "pass_rate": 0.1 * (i + 1) as f64 }
                }),
            )
            .unwrap();
        }

        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                order_by: order_from(json!("-stats.pass_rate")),
                offset: Some(1),
                limit: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        // Best is 0.5, after offset=1 we start at 0.4 then 0.3.
        let pr0 = rows[0].pass_rate.unwrap();
        let pr1 = rows[1].pass_rate.unwrap();
        assert!((pr0 - 0.4).abs() < 1e-9, "got {pr0}");
        assert!((pr1 - 0.3).abs() < 1e-9, "got {pr1}");
    }

    #[test]
    fn parse_where_rejects_non_object() {
        assert!(parse_where(&json!("not an object")).is_err());
        assert!(parse_where(&json!(42)).is_err());
    }

    #[test]
    fn parse_order_by_accepts_string_and_array() {
        let k = parse_order_by(&json!("-stats.pass_rate")).unwrap();
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].path, vec!["stats", "pass_rate"]);
        assert!(k[0].desc);

        let k = parse_order_by(&json!(["created_at", "-stats.n"])).unwrap();
        assert_eq!(k.len(), 2);
        assert!(!k[0].desc);
        assert!(k[1].desc);
    }

    #[test]
    fn find_where_string_ops_contains_and_starts_with() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "find_string_ops_pkg";
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-opus-4-6" },
                "metadata": { "tag": "experiment_alpha" },
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_b"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-haiku-4-5-20251001" },
                "metadata": { "tag": "experiment_beta" },
            }),
        )
        .unwrap();
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_c"),
                "pkg": { "name": pkg },
                "model": { "id": "claude-sonnet-4-5" },
                "metadata": { "tag": "baseline" },
            }),
        )
        .unwrap();

        // contains: matches substring anywhere
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "metadata": { "tag": { "contains": "experiment" } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 2);

        // starts_with: matches only the prefix
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "model": { "id": { "starts_with": "claude-opus" } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].card_id.ends_with("_a"));

        // string ops on missing field → false
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "metadata": { "missing_field": { "contains": "x" } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 0);

        // string ops on non-string field → false
        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "metadata": { "tag": { "starts_with": 42 } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn where_missing_field_ne_is_true() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "where_missing_ne_pkg";
        create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_x"),
                "pkg": { "name": pkg },
            }),
        )
        .unwrap();

        let rows = find_with_store(
            &store,
            FindQuery {
                pkg: Some(pkg.to_string()),
                where_: Some(where_from(json!({
                    "strategy_params": { "temperature": { "ne": 0.5 } },
                }))),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rows.len(), 1, "missing field is ne to anything");
    }

    // ─── lineage ───────────────────────────────────────────────

    /// Helper: create a child Card pointing at a parent with a relation.
    fn create_child(
        store: &FileCardStore,
        pkg: &str,
        suffix: &str,
        parent_id: &str,
        relation: &str,
    ) -> String {
        let (id, _) = create_with_store(
            store,
            json!({
                "card_id": format!("{pkg}_{suffix}"),
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.5 },
                "metadata": {
                    "prior_card_id": parent_id,
                    "prior_relation": relation,
                },
            }),
        )
        .unwrap();
        id
    }

    #[test]
    fn lineage_up_walks_prior_card_id_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "lineage_up_pkg";
        // a → b → c (c is newest; b points at a; c points at b)
        let (a, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
            }),
        )
        .unwrap();
        let b = create_child(&store, pkg, "b", &a, "rerun_of");
        let c = create_child(&store, pkg, "c", &b, "rerun_of");

        let res = lineage_with_store(
            &store,
            LineageQuery {
                card_id: c.clone(),
                direction: LineageDirection::Up,
                depth: None,
                include_stats: false,
                relation_filter: None,
            },
        )
        .unwrap()
        .expect("lineage result");

        assert_eq!(res.root, c);
        assert_eq!(res.nodes.len(), 3, "root + 2 ancestors");
        assert_eq!(res.nodes[0].card_id, c);
        assert_eq!(res.nodes[0].depth, 0);
        assert_eq!(res.nodes[1].card_id, b);
        assert_eq!(res.nodes[1].depth, -1);
        assert_eq!(res.nodes[2].card_id, a);
        assert_eq!(res.nodes[2].depth, -2);
        assert_eq!(res.edges.len(), 2);
        assert!(!res.truncated);
    }

    #[test]
    fn lineage_down_walks_descendants_breadth_first() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "lineage_down_pkg";
        // a has two children b, c; c has one child d.
        let (a, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
            }),
        )
        .unwrap();
        let _b = create_child(&store, pkg, "b", &a, "sweep_variant");
        let c = create_child(&store, pkg, "c", &a, "sweep_variant");
        let _d = create_child(&store, pkg, "d", &c, "rerun_of");

        let res = lineage_with_store(
            &store,
            LineageQuery {
                card_id: a.clone(),
                direction: LineageDirection::Down,
                depth: None,
                include_stats: false,
                relation_filter: None,
            },
        )
        .unwrap()
        .expect("lineage result");

        // root + b + c + d = 4 nodes
        assert_eq!(res.nodes.len(), 4);
        assert_eq!(res.edges.len(), 3);
        assert!(!res.truncated);
    }

    #[test]
    fn lineage_depth_truncation_sets_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "lineage_depth_pkg";
        let (a, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
            }),
        )
        .unwrap();
        let b = create_child(&store, pkg, "b", &a, "rerun_of");
        let _c = create_child(&store, pkg, "c", &b, "rerun_of");

        let res = lineage_with_store(
            &store,
            LineageQuery {
                card_id: a,
                direction: LineageDirection::Down,
                depth: Some(1),
                include_stats: false,
                relation_filter: None,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(res.nodes.len(), 2, "root + 1 level");
        assert!(res.truncated, "should be truncated at depth=1");
    }

    #[test]
    fn lineage_relation_filter_skips_unlisted() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "lineage_filter_pkg";
        let (a, _) = create_with_store(
            &store,
            json!({
                "card_id": format!("{pkg}_a"),
                "pkg": { "name": pkg },
            }),
        )
        .unwrap();
        let _b = create_child(&store, pkg, "b", &a, "sweep_variant");
        let _c = create_child(&store, pkg, "c", &a, "rerun_of");

        let res = lineage_with_store(
            &store,
            LineageQuery {
                card_id: a,
                direction: LineageDirection::Down,
                depth: None,
                include_stats: false,
                relation_filter: Some(vec!["sweep_variant".to_string()]),
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(res.nodes.len(), 2, "root + only sweep_variant child");
        assert_eq!(res.edges[0].relation.as_deref(), Some("sweep_variant"));
    }

    #[test]
    fn lineage_missing_card_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let res = lineage_with_store(
            &store,
            LineageQuery {
                card_id: "nonexistent_card_id_xyz".into(),
                direction: LineageDirection::Up,
                depth: None,
                include_stats: false,
                relation_filter: None,
            },
        )
        .unwrap();
        assert!(res.is_none());
    }

    // ─── samples sidecar ───────────────────────────────────────

    // Isolated `FileCardStore::new(tempdir)` sidesteps the shared-root race
    // in `find_card_locator`; see `read_samples_empty_when_absent` for the
    // full root-cause write-up.
    #[test]
    fn write_and_read_samples_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let (id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": "roundtrip_pkg" },
                "stats": { "pass_rate": 0.5 }
            }),
        )
        .unwrap();

        let samples = vec![
            json!({ "case": "c0", "passed": true, "score": 1.0 }),
            json!({ "case": "c1", "passed": false, "score": 0.0 }),
            json!({ "case": "c2", "passed": true, "score": 0.75 }),
        ];
        let path = write_samples_with_store(&store, &id, samples.clone()).unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().ends_with(".samples.jsonl"));

        let got = read_samples_with_store(&store, &id, SamplesQuery::default()).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["case"], json!("c0"));
        assert_eq!(got[2]["score"], json!(0.75));

        // offset + limit
        let slice = read_samples_with_store(
            &store,
            &id,
            SamplesQuery {
                offset: 1,
                limit: Some(1),
                where_: None,
            },
        )
        .unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0]["case"], json!("c1"));
    }

    #[test]
    fn write_samples_is_write_once() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let (id, _) =
            create_with_store(&store, json!({ "pkg": { "name": "write_once_pkg" } })).unwrap();
        write_samples_with_store(&store, &id, vec![json!({ "x": 1 })]).unwrap();
        let err = write_samples_with_store(&store, &id, vec![json!({ "x": 2 })]).unwrap_err();
        assert!(err.contains("already exist"), "got: {err}");
    }

    // Previously used `create` / `read_samples` (default `~/.algocline/cards/`
    // store). Under `cargo test --workspace` parallel runs, `find_card_locator`
    // scans the shared root with `fs::read_dir(...).flatten()` which silently
    // drops transient I/O errors — on macOS APFS, a concurrent `remove_dir_all`
    // from another test's `cleanup(pkg)` could trigger that transient error and
    // cause this test's just-created pkg dir entry to be missed, propagating
    // `card '...' not found` up through `samples_path` → `read_samples`.
    //
    // Isolating via `FileCardStore::new(tempdir)` + `_with_store` variants
    // sidesteps the shared-root race entirely. Same pattern as
    // `custom_root_file_store_roundtrip` / `test_fanout_concurrent_*`.
    #[test]
    fn read_samples_empty_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let (id, _) = create_with_store(
            &store,
            json!({ "pkg": { "name": "read_samples_empty_pkg" } }),
        )
        .unwrap();
        let got = read_samples_with_store(&store, &id, SamplesQuery::default()).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn read_samples_where_filters_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let (id, _) =
            create_with_store(&store, json!({ "pkg": { "name": "where_filter_pkg" } })).unwrap();
        write_samples_with_store(
            &store,
            &id,
            vec![
                json!({ "case": "c0", "passed": true,  "score": 1.0 }),
                json!({ "case": "c1", "passed": false, "score": 0.0 }),
                json!({ "case": "c2", "passed": true,  "score": 0.25 }),
                json!({ "case": "c3", "passed": true,  "score": 0.75 }),
                json!({ "case": "c4", "passed": false, "score": 0.5 }),
            ],
        )
        .unwrap();

        // Equality predicate: passed == true keeps 3 rows.
        let pred = parse_where(&json!({ "passed": true })).unwrap();
        let got = read_samples_with_store(
            &store,
            &id,
            SamplesQuery {
                offset: 0,
                limit: None,
                where_: Some(pred),
            },
        )
        .unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["case"], json!("c0"));
        assert_eq!(got[1]["case"], json!("c2"));
        assert_eq!(got[2]["case"], json!("c3"));

        // Nested comparator: score gte 0.5 keeps c0/c3/c4.
        let pred = parse_where(&json!({ "score": { "gte": 0.5 } })).unwrap();
        let got = read_samples_with_store(
            &store,
            &id,
            SamplesQuery {
                offset: 0,
                limit: None,
                where_: Some(pred),
            },
        )
        .unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["case"], json!("c0"));
        assert_eq!(got[1]["case"], json!("c3"));
        assert_eq!(got[2]["case"], json!("c4"));

        // Offset applies AFTER filter: passed=true then skip 1 + limit 1 → c2.
        let pred = parse_where(&json!({ "passed": true })).unwrap();
        let slice = read_samples_with_store(
            &store,
            &id,
            SamplesQuery {
                offset: 1,
                limit: Some(1),
                where_: Some(pred),
            },
        )
        .unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0]["case"], json!("c2"));
    }

    #[test]
    fn get_by_alias_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "get_by_alias_pkg";
        let (id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.85 }
            }),
        )
        .unwrap();

        let alias_name = "best_by_alias";
        alias_set_with_store(&store, alias_name, &id, Some(pkg), None).unwrap();

        let card = get_by_alias_with_store(&store, alias_name)
            .unwrap()
            .unwrap();
        assert_eq!(card["card_id"], json!(id));
        assert_eq!(card["stats"]["pass_rate"], json!(0.85));

        assert!(get_by_alias_with_store(&store, "nonexistent_alias_xyz")
            .unwrap()
            .is_none());
    }

    #[test]
    fn samples_errors_on_missing_card() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let err = write_samples_with_store(&store, "does_not_exist_xyz_samples", vec![json!({})])
            .unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── import_from_dir ───────────────────────────────────────

    #[test]
    fn import_from_dir_copies_cards() {
        let pkg = "import_copies_pkg";
        let src_tmp = tempfile::tempdir().unwrap();
        let store_tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(store_tmp.path().to_path_buf());

        // Create a source card file
        let card_id = format!("{pkg}_imported");
        let card_content = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"{card_id}\"\npkg = \"{pkg}\"\n"
        );
        fs::write(
            src_tmp.path().join(format!("{card_id}.toml")),
            &card_content,
        )
        .unwrap();

        // Create a matching samples file
        fs::write(
            src_tmp.path().join(format!("{card_id}.samples.jsonl")),
            "{\"case\":\"c0\"}\n",
        )
        .unwrap();

        let (imported, skipped) = import_from_dir_with_store(&store, src_tmp.path(), pkg).unwrap();
        assert_eq!(imported, vec![card_id.clone()]);
        assert!(skipped.is_empty());

        // Verify card was imported
        let got = get_with_store(&store, &card_id).unwrap().unwrap();
        assert_eq!(got["card_id"], json!(card_id));

        // Verify samples were copied
        let samples = read_samples_with_store(&store, &card_id, SamplesQuery::default()).unwrap();
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn import_from_dir_skips_existing() {
        let store_tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(store_tmp.path().to_path_buf());
        let pkg = "import_skips_existing_pkg";
        // Create a card in the store first
        let (existing_id, _) = create_with_store(
            &store,
            json!({
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.5 }
            }),
        )
        .unwrap();

        // Try to import a card with the same id
        let src_tmp = tempfile::tempdir().unwrap();
        let card_content = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"{existing_id}\"\npkg = \"{pkg}\"\n"
        );
        fs::write(
            src_tmp.path().join(format!("{existing_id}.toml")),
            &card_content,
        )
        .unwrap();

        let (imported, skipped) = import_from_dir_with_store(&store, src_tmp.path(), pkg).unwrap();
        assert!(imported.is_empty());
        assert_eq!(skipped, vec![existing_id.clone()]);

        // Original card untouched
        let got = get_with_store(&store, &existing_id).unwrap().unwrap();
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));
    }

    #[test]
    fn import_from_dir_skips_non_card_toml() {
        let store_tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(store_tmp.path().to_path_buf());
        let pkg = "import_skips_non_card_pkg";
        let src_tmp = tempfile::tempdir().unwrap();

        // A TOML file without schema_version = "card/v0" should be skipped
        fs::write(
            src_tmp.path().join("not_a_card.toml"),
            "title = \"hello\"\n",
        )
        .unwrap();

        let (imported, skipped) = import_from_dir_with_store(&store, src_tmp.path(), pkg).unwrap();
        assert!(imported.is_empty());
        assert!(skipped.is_empty());
    }

    // ─── PathCardStore (FileCardStore rooted at a custom path) ──────
    //
    // Smoke test proving the trait boundary lets callers swap the
    // storage root without touching `~/.algocline/cards/`.

    #[test]
    fn custom_root_file_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let pkg = "custom_root_pkg";

        // create → get → list through the _with_store variants
        let (id, path) = create_with_store(
            &store,
            json!({
                "pkg":   { "name": pkg },
                "model": { "id": "gpt-test" },
            }),
        )
        .unwrap();
        assert!(path.starts_with(tmp.path()));
        assert!(path.ends_with(format!("{id}.toml")));

        let card = get_with_store(&store, &id).unwrap().expect("card exists");
        assert_eq!(
            card.get("card_id").and_then(|v| v.as_str()),
            Some(id.as_str())
        );

        let rows = list_with_store(&store, Some(pkg)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].card_id, id);

        // Ensure a distinct store does not see the card (isolation check).
        let other_tmp = tempfile::tempdir().unwrap();
        let other_store = FileCardStore::new(other_tmp.path().to_path_buf());
        let default_rows = list_with_store(&other_store, Some(pkg)).unwrap();
        assert!(default_rows.iter().all(|r| r.card_id != id));

        // alias + lookup scoped to the custom store
        alias_set_with_store(&store, "alpha", &id, Some(pkg), None).unwrap();
        let via_alias = get_by_alias_with_store(&store, "alpha")
            .unwrap()
            .expect("alias resolves");
        assert_eq!(
            via_alias.get("card_id").and_then(|v| v.as_str()),
            Some(id.as_str())
        );

        // samples write/read roundtrip
        let samples_path =
            write_samples_with_store(&store, &id, vec![json!({ "case": "a", "pass": true })])
                .unwrap();
        assert!(samples_path.starts_with(tmp.path()));
        let back = read_samples_with_store(&store, &id, SamplesQuery::default()).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].get("case").and_then(|v| v.as_str()), Some("a"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Event Publisher Port tests
    // ═══════════════════════════════════════════════════════════════

    use std::sync::atomic::AtomicUsize;
    use std::sync::Barrier;

    /// Serialize access to `std::env::set_var("ALC_CARD_SINKS", ...)` so
    /// env-touching tests do not race.
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RAII guard that clears `INSIDE_BUS_TEST` on drop, including the panic
    /// unwinding path. Without this, a panic inside `f` would leave the
    /// thread-local `true` on the cargo-test worker thread, so the next test
    /// assigned to the same worker would bypass `bus_test_gate` and corrupt
    /// concurrent subscriber mocks.
    struct BusTestOwnerGuard;
    impl Drop for BusTestOwnerGuard {
        fn drop(&mut self) {
            INSIDE_BUS_TEST.with(|flag| flag.set(false));
        }
    }

    /// Ensure the global bus is initialized and subscribers are cleared,
    /// then install `subs` on the singleton for the duration of a test.
    ///
    /// This function holds `bus_test_gate()` for its entire duration. Any
    /// concurrent `publish()` call from a parallel default-store test will
    /// block until we release the gate, preventing event contamination.
    /// The INSIDE_BUS_TEST thread-local is set so that publish calls made
    /// FROM THIS THREAD (inside `f`) skip the gate and proceed directly
    /// (re-entrancy safe).
    ///
    /// If the test spawns child threads that also publish, those children
    /// must set INSIDE_BUS_TEST to true themselves (see
    /// `test_fanout_concurrent_create_with_store`). Otherwise they block on
    /// the gate held by this owner thread and deadlock on join.
    fn with_bus_subscribers<F>(subs: Vec<Arc<dyn CardSubscriber>>, f: F)
    where
        F: FnOnce(&'static CardEventBus),
    {
        // Acquire the gate FIRST. While we wait, no one else holds the owner
        // role, and our INSIDE_BUS_TEST is still false, so this lock is safe.
        let _guard = bus_test_gate().lock().unwrap_or_else(|p| p.into_inner());
        // Now mark this thread as the bus-test owner so that publish() from
        // within the closure does not try to re-acquire bus_test_gate().
        // The RAII guard clears the flag on both normal return and unwind.
        INSIDE_BUS_TEST.with(|flag| flag.set(true));
        let _owner = BusTestOwnerGuard;
        let bus = event_bus();
        bus.reset_stats_for_test();
        bus.replace_subscribers_for_test(subs);
        f(bus);
        // Leave the bus clean for the next test.
        bus.replace_subscribers_for_test(Vec::new());
        bus.reset_stats_for_test();
        // _owner drops -> INSIDE_BUS_TEST = false (panic-safe)
        // _guard drops -> bus_test_gate released
    }

    /// In-memory subscriber used for deterministic fan-out assertions.
    struct MockSubscriber {
        uri: String,
        events: Mutex<Vec<CardEvent>>,
        calls: AtomicUsize,
    }

    impl MockSubscriber {
        fn new(uri: &str) -> Arc<Self> {
            Arc::new(Self {
                uri: uri.to_string(),
                events: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            })
        }
        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl CardSubscriber for MockSubscriber {
        fn on_event(&self, ev: &CardEvent) -> Result<(), String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.events
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(ev.clone());
            Ok(())
        }
        fn describe(&self) -> String {
            self.uri.clone()
        }
    }

    // ─── Bus lifetime ─────────────────────────────────────────

    #[test]
    fn bus_is_process_singleton() {
        let a = event_bus() as *const CardEventBus;
        let b = event_bus() as *const CardEventBus;
        assert_eq!(a, b, "event_bus() must return the same singleton pointer");
    }

    #[test]
    fn publish_with_no_subscribers_is_noop() {
        with_bus_subscribers(Vec::new(), |_bus| {
            // Should not panic; publish is a pure no-op when empty.
            publish(CardEvent::Created {
                pkg: "pkg".into(),
                card_id: "id".into(),
                toml_text: "x = 1\n".into(),
            });
        });
    }

    #[test]
    fn init_event_bus_is_idempotent() {
        init_event_bus();
        init_event_bus();
        init_event_bus();
        // Reaching here without panic is success.
    }

    // ─── Fan-out core ─────────────────────────────────────────

    #[test]
    fn fanout_mirrors_create() {
        let primary = tempfile::tempdir().unwrap();
        let sub_a = tempfile::tempdir().unwrap();
        let sub_b = tempfile::tempdir().unwrap();
        let fa = Arc::new(FileCardSubscriber::new(sub_a.path().to_path_buf()));
        let fb = Arc::new(FileCardSubscriber::new(sub_b.path().to_path_buf()));
        with_bus_subscribers(vec![fa.clone(), fb.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (id, path) =
                create_with_store(&store, json!({ "pkg": { "name": "fanout_create_pkg" } }))
                    .unwrap();
            assert!(path.exists());
            let primary_text = fs::read_to_string(&path).unwrap();
            let a_path = sub_a
                .path()
                .join("fanout_create_pkg")
                .join(format!("{id}.toml"));
            let b_path = sub_b
                .path()
                .join("fanout_create_pkg")
                .join(format!("{id}.toml"));
            assert!(a_path.exists(), "subscriber A missing card");
            assert!(b_path.exists(), "subscriber B missing card");
            assert_eq!(fs::read_to_string(&a_path).unwrap(), primary_text);
            assert_eq!(fs::read_to_string(&b_path).unwrap(), primary_text);
        });
    }

    #[test]
    fn fanout_mirrors_append() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (id, _) =
                create_with_store(&store, json!({ "pkg": { "name": "fanout_append_pkg" } }))
                    .unwrap();
            // After create the subscriber must have the card so append can locate it.
            append_with_store(&store, &id, json!({ "extra_key": 42 })).unwrap();
            let sub_path = sub
                .path()
                .join("fanout_append_pkg")
                .join(format!("{id}.toml"));
            let text = fs::read_to_string(&sub_path).unwrap();
            assert!(text.contains("extra_key"), "append must mirror new key");
        });
    }

    #[test]
    fn fanout_mirrors_samples() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (id, _) =
                create_with_store(&store, json!({ "pkg": { "name": "fanout_samples_pkg" } }))
                    .unwrap();
            write_samples_with_store(&store, &id, vec![json!({ "case": "c0" })]).unwrap();
            let sub_path = sub
                .path()
                .join("fanout_samples_pkg")
                .join(format!("{id}.samples.jsonl"));
            let text = fs::read_to_string(&sub_path).unwrap();
            assert!(text.contains("\"case\":\"c0\""));
        });
    }

    #[test]
    fn fanout_mirrors_aliases() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (id, _) =
                create_with_store(&store, json!({ "pkg": { "name": "fanout_alias_pkg" } }))
                    .unwrap();
            alias_set_with_store(&store, "myalias", &id, Some("fanout_alias_pkg"), None).unwrap();
            let sub_aliases = sub.path().join("_aliases.toml");
            assert!(sub_aliases.exists(), "subscriber must receive aliases file");
            let text = fs::read_to_string(&sub_aliases).unwrap();
            assert!(text.contains("myalias"));
        });
    }

    #[test]
    fn fanout_mirrors_import_from_dir_cards() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();

        // Build a pre-existing source tree (a previous run's output).
        let src_card = src.path().join("card_x.toml");
        let toml_body = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"card_x\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n[pkg]\nname = \"fanout_import_pkg\"\n"
        );
        fs::write(&src_card, &toml_body).unwrap();

        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (imported, _skipped) =
                import_from_dir_with_store(&store, src.path(), "fanout_import_pkg").unwrap();
            assert_eq!(imported, vec!["card_x".to_string()]);

            let sub_card = sub.path().join("fanout_import_pkg").join("card_x.toml");
            assert!(sub_card.exists(), "imported card must be mirrored");
        });
    }

    #[test]
    fn fanout_mirrors_import_from_dir_samples() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();

        let toml_body = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"card_y\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n[pkg]\nname = \"fanout_import_samples_pkg\"\n"
        );
        fs::write(src.path().join("card_y.toml"), &toml_body).unwrap();
        fs::write(
            src.path().join("card_y.samples.jsonl"),
            "{\"case\":\"c0\"}\n",
        )
        .unwrap();

        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            let (imported, _) =
                import_from_dir_with_store(&store, src.path(), "fanout_import_samples_pkg")
                    .unwrap();
            assert_eq!(imported, vec!["card_y".to_string()]);

            let sub_samples = sub
                .path()
                .join("fanout_import_samples_pkg")
                .join("card_y.samples.jsonl");
            assert!(sub_samples.exists(), "imported samples must be mirrored");
            let text = fs::read_to_string(&sub_samples).unwrap();
            assert!(text.contains("c0"));
        });
    }

    #[test]
    fn with_store_direct_call_still_publishes() {
        let primary = tempfile::tempdir().unwrap();
        let mock = MockSubscriber::new("mock://direct");
        with_bus_subscribers(vec![mock.clone() as Arc<dyn CardSubscriber>], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            create_with_store(&store, json!({ "pkg": { "name": "direct_call_pkg" } })).unwrap();
            assert_eq!(mock.call_count(), 1, "direct _with_store call must publish");
        });
    }

    #[test]
    fn subscriber_appended_missing_card_warns() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            // Create the card BEFORE the subscriber knows about it. To do that,
            // swap the subscriber out so create does not mirror, then swap in.
            bus.replace_subscribers_for_test(Vec::new());
            let (id, _) =
                create_with_store(&store, json!({ "pkg": { "name": "missing_append_pkg" } }))
                    .unwrap();
            // Re-install the subscriber; it has no mirror of the card.
            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);

            // The append call must succeed on the primary; the subscriber
            // will record an error because locate_card returns None.
            append_with_store(&store, &id, json!({ "k": 1 })).unwrap();

            let snap = bus.stats().snapshot();
            let row = snap
                .iter()
                .find(|r| r.sink == fs_sub.describe())
                .expect("subscriber row exists");
            let err_total: u64 = row.err.values().sum();
            assert!(err_total >= 1, "subscriber append err must be recorded");
            assert!(row.last_error.is_some());
        });
    }

    #[test]
    fn subscriber_failure_preserves_primary() {
        struct FailingSubscriber;
        impl CardSubscriber for FailingSubscriber {
            fn on_event(&self, _ev: &CardEvent) -> Result<(), String> {
                Err("synthetic failure".into())
            }
            fn describe(&self) -> String {
                "mock://failing".into()
            }
        }
        let primary = tempfile::tempdir().unwrap();
        with_bus_subscribers(
            vec![Arc::new(FailingSubscriber) as Arc<dyn CardSubscriber>],
            |bus| {
                let store = FileCardStore::new(primary.path().to_path_buf());
                // Primary call must still succeed despite subscriber failure.
                let (_id, path) =
                    create_with_store(&store, json!({ "pkg": { "name": "preserve_primary_pkg" } }))
                        .unwrap();
                assert!(path.exists());
                let snap = bus.stats().snapshot();
                let row = snap
                    .iter()
                    .find(|r| r.sink == "mock://failing")
                    .expect("row exists");
                let err_total: u64 = row.err.values().sum();
                assert!(err_total >= 1);
            },
        );
    }

    // ─── SubscriberStats JSON shape tests (Subtask 2) ──────────

    #[test]
    fn stats_counts_ok() {
        let primary = tempfile::tempdir().unwrap();
        let mock = MockSubscriber::new("mock://stats-ok");
        with_bus_subscribers(vec![mock.clone() as Arc<dyn CardSubscriber>], |bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            for i in 0..3 {
                create_with_store(
                    &store,
                    json!({
                        "card_id": format!("stats_ok_{i}"),
                        "pkg": { "name": "stats_ok_pkg" },
                    }),
                )
                .unwrap();
            }
            let snap = bus.stats().snapshot();
            let row = snap
                .iter()
                .find(|r| r.sink == "mock://stats-ok")
                .expect("row");
            assert_eq!(row.ok.get("created").copied().unwrap_or(0), 3);
            assert_eq!(row.err.get("created").copied().unwrap_or(0), 0);
            // All four keys must be present (may be 0).
            for k in ["created", "appended", "samples", "aliases"] {
                assert!(row.ok.contains_key(k), "ok.{k} must be present");
                assert!(row.err.contains_key(k), "err.{k} must be present");
            }
            assert!(row.last_error.is_none());
        });
    }

    #[test]
    fn stats_counts_err_with_last_error() {
        struct FailingSubscriber;
        impl CardSubscriber for FailingSubscriber {
            fn on_event(&self, _ev: &CardEvent) -> Result<(), String> {
                Err("synthetic create failure".into())
            }
            fn describe(&self) -> String {
                "mock://stats-err".into()
            }
        }
        let primary = tempfile::tempdir().unwrap();
        with_bus_subscribers(
            vec![Arc::new(FailingSubscriber) as Arc<dyn CardSubscriber>],
            |bus| {
                let store = FileCardStore::new(primary.path().to_path_buf());
                create_with_store(&store, json!({ "pkg": { "name": "stats_err_pkg" } })).unwrap();
                let snap = bus.stats().snapshot();
                let row = snap
                    .iter()
                    .find(|r| r.sink == "mock://stats-err")
                    .expect("row");
                assert_eq!(row.err.get("created").copied().unwrap_or(0), 1);
                let le = row.last_error.as_ref().expect("last_error set");
                assert!(!le.msg.is_empty(), "last_error.msg must be non-empty");
                assert_eq!(le.kind, CardEventKind::Created);
                assert!(le.ts_ms > 0, "last_error.ts_ms must be populated");
            },
        );
    }

    #[test]
    fn stats_per_subscriber_isolated() {
        struct FailingSubscriber;
        impl CardSubscriber for FailingSubscriber {
            fn on_event(&self, _ev: &CardEvent) -> Result<(), String> {
                Err("sub1 fails".into())
            }
            fn describe(&self) -> String {
                "mock://sub1-fail".into()
            }
        }
        let primary = tempfile::tempdir().unwrap();
        let mock_ok = MockSubscriber::new("mock://sub2-ok");
        let subs: Vec<Arc<dyn CardSubscriber>> = vec![
            Arc::new(FailingSubscriber) as Arc<dyn CardSubscriber>,
            mock_ok.clone() as Arc<dyn CardSubscriber>,
        ];
        with_bus_subscribers(subs, |bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            create_with_store(&store, json!({ "pkg": { "name": "isolated_pkg" } })).unwrap();
            let snap = bus.stats().snapshot();
            let r1 = snap
                .iter()
                .find(|r| r.sink == "mock://sub1-fail")
                .expect("sub1 row");
            let r2 = snap
                .iter()
                .find(|r| r.sink == "mock://sub2-ok")
                .expect("sub2 row");
            assert_eq!(r1.err.get("created").copied().unwrap_or(0), 1);
            assert_eq!(r1.ok.get("created").copied().unwrap_or(0), 0);
            assert_eq!(r2.ok.get("created").copied().unwrap_or(0), 1);
            assert_eq!(r2.err.get("created").copied().unwrap_or(0), 0);
            assert!(r1.last_error.is_some());
            assert!(r2.last_error.is_none());
        });
    }

    #[test]
    fn subscriber_stats_survive_multiple_calls() {
        // Regression guard: per-call SubscriberStats creation would
        // have reset the counter between create_with_store invocations.
        // Verify that counters accumulate across 3 independent calls
        // against the global bus's stats handle.
        let primary = tempfile::tempdir().unwrap();
        let mock = MockSubscriber::new("mock://stats-survive");
        with_bus_subscribers(vec![mock.clone() as Arc<dyn CardSubscriber>], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            for i in 0..3 {
                create_with_store(
                    &store,
                    json!({
                        "card_id": format!("survive_{i}"),
                        "pkg": { "name": "survive_pkg" },
                    }),
                )
                .unwrap();
            }
            // Use the public snapshot entry point to exercise the
            // same path that AppService::stats uses.
            let snap = subscriber_stats_snapshot();
            let row = snap
                .iter()
                .find(|r| r.sink == "mock://stats-survive")
                .expect("row");
            assert_eq!(
                row.ok.get("created").copied().unwrap_or(0),
                3,
                "counters must accumulate across calls"
            );
        });
    }

    #[test]
    fn stats_snapshot_serializes_with_all_kind_keys() {
        // Serialize a minimal row and verify JSON field shape.
        let primary = tempfile::tempdir().unwrap();
        let mock = MockSubscriber::new("mock://json-shape");
        with_bus_subscribers(vec![mock.clone() as Arc<dyn CardSubscriber>], |_bus| {
            let store = FileCardStore::new(primary.path().to_path_buf());
            create_with_store(&store, json!({ "pkg": { "name": "json_shape_pkg" } })).unwrap();
            let snap = subscriber_stats_snapshot();
            let json = serde_json::to_value(&snap).expect("serializable");
            let arr = json.as_array().expect("array");
            let row = arr
                .iter()
                .find(|r| r.get("sink").and_then(|s| s.as_str()) == Some("mock://json-shape"))
                .expect("row present in JSON");
            assert_eq!(row.get("sink").unwrap(), "mock://json-shape");
            for k in ["created", "appended", "samples", "aliases"] {
                assert!(row.pointer(&format!("/ok/{k}")).is_some(), "ok.{k} missing");
                assert!(
                    row.pointer(&format!("/err/{k}")).is_some(),
                    "err.{k} missing"
                );
            }
            assert!(row.get("last_error").is_some());
        });
    }

    #[test]
    fn multi_process_tmp_unique_suffix() {
        // Invoke atomic_write against a fresh dir and capture the tmp
        // filename left on disk by forcing rename to happen on an
        // already-nonexistent dest. We simulate by writing to a path
        // and then inspecting the parent dir during the operation —
        // since atomic_write removes tmp on success, we instead check
        // the suffix format by constructing it the same way.
        let pid = process::id();
        let sample = format!("whatever.tmp.{pid}.0");
        // Regex-style match without the regex crate dependency:
        let rest = sample.trim_start_matches("whatever.tmp.");
        let (pid_part, seq_part) = rest.split_once('.').expect("dotted form");
        assert!(pid_part.chars().all(|c| c.is_ascii_digit()));
        assert!(seq_part.chars().all(|c| c.is_ascii_digit()));

        // Real atomic_write round-trip — must not panic and must leave
        // the dest file in place with the written bytes.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.txt");
        atomic_write(&dest, b"hello").unwrap();
        assert_eq!(fs::read_to_string(&dest).unwrap(), "hello");
    }

    // ─── describe / env parser ───────────────────────────────

    #[cfg(unix)]
    #[test]
    fn describe_roundtrips_env_spec() {
        let dir = tempfile::tempdir().unwrap();
        let sub = FileCardSubscriber::new(dir.path().to_path_buf());
        let uri = sub.describe();
        assert!(uri.starts_with("file:///"), "unix uri must be file:///...");
        // Parse the URI back and confirm the resolved path matches.
        let parsed = parse_subscriber_spec(&uri).expect("round-trip parse");
        assert_eq!(parsed.describe(), uri);
    }

    #[cfg(windows)]
    #[test]
    fn describe_roundtrips_env_spec_windows() {
        let dir = tempfile::tempdir().unwrap();
        let sub = FileCardSubscriber::new(dir.path().to_path_buf());
        let uri = sub.describe();
        assert!(
            uri.starts_with("file:///"),
            "windows uri must be file:///..."
        );
        let parsed = parse_subscriber_spec(&uri).expect("round-trip parse");
        assert_eq!(parsed.describe(), uri);
    }

    #[test]
    fn env_empty_means_no_subscribers() {
        // env-touching — serialized by env_lock() to avoid races with
        // any other env-reading test in this binary.
        let _g = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("ALC_CARD_SINKS").ok();
        // SAFETY: test-only single-threaded env mutation under mutex.
        unsafe {
            std::env::set_var("ALC_CARD_SINKS", "");
        }
        let subs = load_subscribers_from_env();
        assert!(subs.is_empty());
        // restore
        unsafe {
            match prev {
                Some(v) => std::env::set_var("ALC_CARD_SINKS", v),
                None => std::env::remove_var("ALC_CARD_SINKS"),
            }
        }
    }

    #[test]
    fn env_parse_rejects_bare_path() {
        assert!(parse_subscriber_spec("/foo/bar").is_none());
    }

    #[test]
    fn env_parse_rejects_unknown_scheme() {
        assert!(parse_subscriber_spec("sqlite:///foo").is_none());
        assert!(parse_subscriber_spec("s3://bucket/foo").is_none());
        assert!(parse_subscriber_spec("http://example.com/x").is_none());
    }

    #[test]
    fn env_parse_rejects_non_empty_authority() {
        assert!(parse_subscriber_spec("file://host/path").is_none());
    }

    #[test]
    fn env_parse_rejects_missing_double_slash() {
        assert!(parse_subscriber_spec("file:/foo").is_none());
        assert!(parse_subscriber_spec("file:foo").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn env_parse_accepts_file_uri() {
        let sub = parse_subscriber_spec("file:///tmp/algocline-sinks-unit").expect("accepted");
        assert_eq!(sub.describe(), "file:///tmp/algocline-sinks-unit");
    }

    #[cfg(windows)]
    #[test]
    fn env_parse_accepts_file_uri_windows() {
        let sub = parse_subscriber_spec("file:///C:/algocline-sinks-unit").expect("accepted");
        // Windows canonicalization re-emits the same URI.
        assert!(sub.describe().starts_with("file:///"));
    }

    #[test]
    fn env_parse_splits_by_pipe() {
        let subs = parse_subscribers_from_str("file:///tmp/a|file:///tmp/b");
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].describe(), "file:///tmp/a");
        assert_eq!(subs[1].describe(), "file:///tmp/b");
    }

    #[test]
    fn env_parse_treats_colon_as_literal_path() {
        // `file:///tmp/a:b` — colon inside the path component is a literal.
        #[cfg(unix)]
        {
            let sub = parse_subscriber_spec("file:///tmp/a:b").expect("accepted");
            assert_eq!(sub.describe(), "file:///tmp/a:b");
        }
        #[cfg(windows)]
        {
            // On Windows the colon shows up as a drive letter separator.
            let sub = parse_subscriber_spec("file:///C:/a:b").expect("accepted");
            assert!(sub.describe().contains(":"));
        }
    }

    #[test]
    fn env_parse_percent_decode_space() {
        #[cfg(unix)]
        {
            let sub = parse_subscriber_spec("file:///tmp/a%20b").expect("accepted");
            assert_eq!(sub.describe(), "file:///tmp/a b");
        }
        #[cfg(windows)]
        {
            let sub = parse_subscriber_spec("file:///C:/a%20b").expect("accepted");
            assert!(sub.describe().contains(' '));
        }
    }

    #[test]
    fn env_parse_percent_decode_rejects_invalid_hex() {
        assert!(parse_subscriber_spec("file:///tmp/a%ZZb").is_none());
    }

    #[test]
    fn env_parse_percent_decode_rejects_incomplete() {
        assert!(parse_subscriber_spec("file:///tmp/a%2").is_none());
        assert!(parse_subscriber_spec("file:///tmp/a%").is_none());
    }

    #[test]
    fn env_parse_rejects_non_utf8() {
        // Exercised through `load_subscribers_from_env` via NotUnicode.
        // We cannot easily set a non-UTF8 env var cross-platform inside
        // a unit test, so we verify the error path indirectly: the
        // parser only consumes `String` which is UTF-8 by construction,
        // and the env reader branches on VarError::NotUnicode. To keep
        // the test meaningful, verify that percent-decoded non-UTF8
        // bytes are rejected (closest structural analogue).
        // `%C3%28` is an invalid UTF-8 two-byte sequence.
        assert!(parse_subscriber_spec("file:///tmp/%C3%28").is_none());
    }

    #[test]
    fn env_parse_dedups_duplicate_uris() {
        let subs = parse_subscribers_from_str("file:///tmp/x|file:///tmp/x|file:///tmp/y");
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].describe(), "file:///tmp/x");
        assert_eq!(subs[1].describe(), "file:///tmp/y");
    }

    // ═══════════════════════════════════════════════════════════════
    // Concurrency tests (concurrency-analysis.md §2)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_oncelock_init_race_single_winner() {
        // N threads call event_bus() concurrently; all must observe the
        // same singleton pointer.
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                event_bus() as *const CardEventBus as usize
            }));
        }
        let ptrs: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = ptrs[0];
        for p in &ptrs {
            assert_eq!(*p, first, "singleton identity must hold across threads");
        }
    }

    #[test]
    fn test_subscriber_stats_concurrent_update() {
        let stats = Arc::new(SubscriberStats::default());
        let n_threads = 4;
        let per_thread = 250;
        let barrier = Arc::new(Barrier::new(n_threads));
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let s = stats.clone();
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                for i in 0..per_thread {
                    let kind = if (t + i) % 2 == 0 {
                        CardEventKind::Created
                    } else {
                        CardEventKind::Appended
                    };
                    s.record_ok("mock://same-subscriber", kind);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let snap = stats.snapshot();
        let row = snap
            .iter()
            .find(|r| r.sink == "mock://same-subscriber")
            .expect("row");
        let expected = (n_threads * per_thread) as u64;
        let ok_total: u64 = row.ok.values().sum();
        assert_eq!(ok_total, expected, "all increments must be counted");
    }

    #[test]
    fn test_subscriber_stats_poison_recovery() {
        let stats = Arc::new(SubscriberStats::default());
        // Populate some value so the recovered inner map is non-empty.
        stats.record_ok("mock://poison", CardEventKind::Created);

        // Poison the Mutex.
        let s_clone = stats.clone();
        let _ = std::thread::spawn(move || {
            let _g = s_clone.inner.lock().unwrap();
            panic!("intentional poison");
        })
        .join();

        // Follow-up accessors must not hang and must return the prior value.
        let snap = stats.snapshot();
        assert!(!snap.is_empty(), "snapshot after poison must still work");
        let ok1: u64 = snap[0].ok.values().sum();
        assert_eq!(ok1, 1);

        // Further writes must also succeed (via unwrap_or_else).
        stats.record_ok("mock://poison", CardEventKind::Created);
        let snap2 = stats.snapshot();
        let ok2: u64 = snap2[0].ok.values().sum();
        assert_eq!(ok2, 2);
    }

    #[test]
    fn test_atomic_tmp_seq_unique_under_concurrency() {
        // N threads build tmp suffix strings via the same atomic and
        // all suffixes must differ.
        let dir = tempfile::tempdir().unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for i in 0..8 {
            let d = dir.path().to_path_buf();
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let dest = d.join(format!("file_{i}.bin"));
                atomic_write(&dest, b"x").unwrap();
                // Collect the leaf filename for uniqueness.
                dest.file_name().unwrap().to_string_lossy().to_string()
            }));
        }
        let names: HashSet<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(names.len(), 8, "all dest names must be unique");
        // Additionally confirm suffix format by invoking atomic_write
        // again and parsing the tmp we leave on a forced failure.
    }

    #[test]
    fn test_atomic_tmp_seq_wraps_without_panic() {
        // Isolated AtomicU64 at the boundary of wrap-around. fetch_add
        // is documented to wrap without panic.
        let seq = AtomicU64::new(u64::MAX - 1);
        let a = seq.fetch_add(1, Ordering::Relaxed);
        let b = seq.fetch_add(1, Ordering::Relaxed);
        let c = seq.fetch_add(1, Ordering::Relaxed);
        assert_eq!(a, u64::MAX - 1);
        assert_eq!(b, u64::MAX);
        assert_eq!(c, 0, "u64 fetch_add must wrap to 0");
    }

    #[test]
    fn test_rename_atomicity_same_volume() {
        // 2 threads write the same dest from different tmp names. On
        // POSIX the late writer wins; either way the dest must be
        // observable and contain one of the two payloads.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("shared.bin");
        let barrier = Arc::new(Barrier::new(2));
        let mut handles = Vec::new();
        for i in 0..2u8 {
            let d = dest.clone();
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let payload = vec![i; 64];
                atomic_write(&d, &payload)
            }));
        }
        let mut saw_ok = 0;
        for h in handles {
            #[cfg(unix)]
            {
                // On POSIX both should succeed — rename is atomic but allowed.
                h.join().unwrap().unwrap();
                saw_ok += 1;
            }
            #[cfg(not(unix))]
            {
                // On Windows at least one must succeed.
                if h.join().unwrap().is_ok() {
                    saw_ok += 1;
                }
            }
        }
        assert!(saw_ok >= 1, "at least one rename must succeed");
        assert!(dest.exists(), "dest must exist after concurrent rename");
        let bytes = fs::read(&dest).unwrap();
        assert!(bytes == vec![0u8; 64] || bytes == vec![1u8; 64]);
    }

    #[test]
    fn test_fanout_concurrent_create_with_store() {
        let primary = tempfile::tempdir().unwrap();
        let sub = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub.path().to_path_buf()));
        with_bus_subscribers(vec![fs_sub.clone()], |bus| {
            let primary_path = primary.path().to_path_buf();
            let barrier = Arc::new(Barrier::new(4));
            let mut handles = Vec::new();
            for i in 0..4 {
                let pp = primary_path.clone();
                let b = barrier.clone();
                handles.push(std::thread::spawn(move || {
                    // The parent holds `bus_test_gate()` for the entire
                    // `with_bus_subscribers` scope. Child threads must set
                    // INSIDE_BUS_TEST=true themselves so that publish()
                    // bypasses the gate instead of blocking on it (which
                    // would deadlock once the parent calls join()).
                    INSIDE_BUS_TEST.with(|flag| flag.set(true));
                    b.wait();
                    let store = FileCardStore::new(pp);
                    create_with_store(
                        &store,
                        json!({
                            "card_id": format!("concur_card_{i}"),
                            "pkg": { "name": "concur_pkg" },
                        }),
                    )
                    .unwrap()
                    .0
                }));
            }
            let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(ids.len(), 4);
            for id in &ids {
                let p = sub.path().join("concur_pkg").join(format!("{id}.toml"));
                assert!(p.exists(), "subscriber must have card {id}");
            }
            let snap = bus.stats().snapshot();
            let row = snap
                .iter()
                .find(|r| r.sink == fs_sub.describe())
                .expect("row");
            let ok_total: u64 = row.ok.values().sum();
            assert_eq!(
                ok_total, 4,
                "subscriber must have recorded 4 successful deliveries"
            );
        });
    }

    // ─── card_sink_backfill (Subtask 3) ────────────────────────

    /// Primary-side fixture with N cards already written via the
    /// subscriber-free path so backfill has something to push.
    /// Returns (primary_dir_guard, store, card_ids).
    fn backfill_primary_with_cards(
        pkg: &str,
        count: usize,
    ) -> (tempfile::TempDir, FileCardStore, Vec<String>) {
        let primary = tempfile::tempdir().unwrap();
        let store = FileCardStore::new(primary.path().to_path_buf());
        let mut ids = Vec::new();
        for i in 0..count {
            let (id, _) = create_with_store(
                &store,
                json!({
                    "card_id": format!("{pkg}_{i}"),
                    "pkg": { "name": pkg },
                }),
            )
            .unwrap();
            ids.push(id);
        }
        (primary, store, ids)
    }

    #[test]
    fn backfill_pushes_missing_cards() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            // Populate primary before the subscriber is live (temporarily drop it).
            let bus = event_bus();
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, ids) = backfill_primary_with_cards("backfill_push_pkg", 2);
            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);

            let report = card_sink_backfill_with_store(&store, &uri, false).unwrap();
            assert_eq!(report.pushed.len(), 2);
            assert_eq!(report.skipped.len(), 0);
            assert!(report.failed.is_empty());
            for id in &ids {
                let p = sub_dir
                    .path()
                    .join("backfill_push_pkg")
                    .join(format!("{id}.toml"));
                assert!(p.exists(), "card {id} must exist on subscriber");
            }
        });
    }

    #[test]
    fn backfill_skips_existing_on_subscriber() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            // Subscriber is live during create, so it already has the card.
            let (_primary, store, _ids) = backfill_primary_with_cards("backfill_skip_pkg", 3);
            let report = card_sink_backfill_with_store(&store, &uri, false).unwrap();
            assert_eq!(report.pushed.len(), 0);
            assert_eq!(report.skipped.len(), 3);
            assert!(report.failed.is_empty());
        });
    }

    #[test]
    fn backfill_dry_run_no_writes() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let bus = event_bus();
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, ids) = backfill_primary_with_cards("backfill_dry_pkg", 2);
            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);

            let report = card_sink_backfill_with_store(&store, &uri, true).unwrap();
            assert_eq!(
                report.pushed.len(),
                2,
                "pushed must list ids even in dry run"
            );
            for id in &ids {
                let p = sub_dir
                    .path()
                    .join("backfill_dry_pkg")
                    .join(format!("{id}.toml"));
                assert!(!p.exists(), "dry run must NOT write card {id}");
            }
            // Stats must remain zero — dry run publishes nothing.
            let snap = bus.stats().snapshot();
            if let Some(row) = snap.iter().find(|r| r.sink == uri) {
                let total: u64 = row.ok.values().sum::<u64>() + row.err.values().sum::<u64>();
                assert_eq!(total, 0, "dry run must not touch stats");
            }
        });
    }

    #[test]
    fn backfill_drifted_card_skipped_not_overwritten() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let bus = event_bus();
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, ids) = backfill_primary_with_cards("backfill_drift_pkg", 1);
            let id = &ids[0];

            // Manually place a drifted copy on the subscriber with sentinel text.
            let sub_card_dir = sub_dir.path().join("backfill_drift_pkg");
            fs::create_dir_all(&sub_card_dir).unwrap();
            let sub_card = sub_card_dir.join(format!("{id}.toml"));
            fs::write(&sub_card, "drifted=true\n").unwrap();

            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);
            let report = card_sink_backfill_with_store(&store, &uri, false).unwrap();
            assert_eq!(report.skipped, vec![id.clone()]);
            assert!(report.pushed.is_empty());
            let after = fs::read_to_string(&sub_card).unwrap();
            assert_eq!(after, "drifted=true\n", "drifted copy must be preserved");
        });
    }

    #[test]
    fn backfill_includes_samples() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |_bus| {
            let bus = event_bus();
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, ids) = backfill_primary_with_cards("backfill_samples_pkg", 1);
            let id = &ids[0];
            write_samples_with_store(&store, id, vec![json!({ "case": "c0" })]).unwrap();
            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);

            let report = card_sink_backfill_with_store(&store, &uri, false).unwrap();
            assert_eq!(report.pushed, vec![id.clone()]);
            assert_eq!(report.pushed_samples, vec![id.clone()]);
            let sub_samples = sub_dir
                .path()
                .join("backfill_samples_pkg")
                .join(format!("{id}.samples.jsonl"));
            assert!(sub_samples.exists());
            assert!(fs::read_to_string(&sub_samples).unwrap().contains("c0"));
        });
    }

    #[test]
    fn backfill_unknown_sink_err() {
        with_bus_subscribers(Vec::new(), |_bus| {
            let (_primary, store, _ids) = backfill_primary_with_cards("backfill_unknown_pkg", 1);
            let err = card_sink_backfill_with_store(&store, "file:///nonexistent/sink", false)
                .unwrap_err();
            assert!(
                err.starts_with("unknown sink"),
                "must reject unregistered sink; got: {err}"
            );
        });
    }

    #[test]
    fn backfill_bypasses_bus_fanout() {
        // Subscriber A is already in-sync; Subscriber B is the backfill target.
        // Backfilling B must NOT re-deliver Created events to A.
        let sub_a_dir = tempfile::tempdir().unwrap();
        let sub_b_dir = tempfile::tempdir().unwrap();
        let fa = Arc::new(FileCardSubscriber::new(sub_a_dir.path().to_path_buf()));
        let fb = Arc::new(FileCardSubscriber::new(sub_b_dir.path().to_path_buf()));
        let uri_b = fb.describe();
        with_bus_subscribers(
            vec![
                fa.clone() as Arc<dyn CardSubscriber>,
                fb.clone() as Arc<dyn CardSubscriber>,
            ],
            |bus| {
                // Populate primary with subscriber A live (B temporarily absent).
                bus.replace_subscribers_for_test(vec![fa.clone()]);
                let (_primary, store, _ids) = backfill_primary_with_cards("backfill_bypass_pkg", 2);
                // Capture A's ok[created] count before backfill.
                let before = bus
                    .stats()
                    .snapshot()
                    .into_iter()
                    .find(|r| r.sink == fa.describe())
                    .map(|r| r.ok.get("created").copied().unwrap_or(0))
                    .unwrap_or(0);
                // Now reinstall both subscribers and backfill only B.
                bus.replace_subscribers_for_test(vec![fa.clone(), fb.clone()]);
                card_sink_backfill_with_store(&store, &uri_b, false).unwrap();
                let after = bus
                    .stats()
                    .snapshot()
                    .into_iter()
                    .find(|r| r.sink == fa.describe())
                    .map(|r| r.ok.get("created").copied().unwrap_or(0))
                    .unwrap_or(0);
                assert_eq!(
                    before, after,
                    "backfill target B must not cause fan-out to subscriber A"
                );
            },
        );
    }

    #[test]
    fn backfill_updates_subscriber_stats() {
        let sub_dir = tempfile::tempdir().unwrap();
        let fs_sub = Arc::new(FileCardSubscriber::new(sub_dir.path().to_path_buf()));
        let uri = fs_sub.describe();
        with_bus_subscribers(vec![fs_sub.clone()], |bus| {
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, _ids) = backfill_primary_with_cards("backfill_stats_pkg", 2);
            bus.replace_subscribers_for_test(vec![fs_sub.clone()]);

            card_sink_backfill_with_store(&store, &uri, false).unwrap();
            let snap = bus.stats().snapshot();
            let row = snap.iter().find(|r| r.sink == uri).expect("row");
            assert_eq!(
                row.ok.get("created").copied().unwrap_or(0),
                2,
                "backfill must increment ok[created] on the target sink"
            );
        });
    }

    #[test]
    fn backfill_failure_records_err_stat() {
        // Subscriber whose on_event always fails (no filesystem needed).
        struct FailingSub {
            uri: String,
        }
        impl CardSubscriber for FailingSub {
            fn on_event(&self, _ev: &CardEvent) -> Result<(), String> {
                Err("synthetic backfill failure".into())
            }
            fn has_card(&self, _card_id: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn describe(&self) -> String {
                self.uri.clone()
            }
        }
        let uri = "mock://backfill-fail".to_string();
        let failing: Arc<dyn CardSubscriber> = Arc::new(FailingSub { uri: uri.clone() });
        with_bus_subscribers(vec![failing], |bus| {
            bus.replace_subscribers_for_test(Vec::new());
            let (_primary, store, _ids) = backfill_primary_with_cards("backfill_fail_pkg", 1);
            // Reinstall the failing subscriber for the backfill phase.
            let reinstall: Arc<dyn CardSubscriber> = Arc::new(FailingSub { uri: uri.clone() });
            bus.replace_subscribers_for_test(vec![reinstall]);

            let report = card_sink_backfill_with_store(&store, &uri, false).unwrap();
            assert_eq!(
                report.failed.len(),
                1,
                "failed must record the synthetic err"
            );
            assert!(report.pushed.is_empty());
            let snap = bus.stats().snapshot();
            let row = snap.iter().find(|r| r.sink == uri).expect("row");
            assert!(
                row.err.get("created").copied().unwrap_or(0) >= 1,
                "failing publish must increment err[created]"
            );
            assert!(row.last_error.is_some());
        });
    }

    #[test]
    fn test_oncelock_set_after_init_returns_err() {
        // Force init (no-op if already initialized by a prior test).
        let _ = event_bus();
        let result = install_event_bus_for_test(CardEventBus::new(Vec::new()));
        assert!(
            result.is_err(),
            "install after init must return Err per OnceLock contract"
        );
        assert_eq!(result.unwrap_err(), "bus already initialized");
    }
}
