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

use std::fs;
use std::path::{Path, PathBuf};

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

/// Return the default backend (File-backed, `~/.algocline/cards/`).
fn default_store() -> Result<FileCardStore, String> {
    FileCardStore::from_home()
}

/// Resolve the cards root directory, creating it if needed.
fn cards_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let dir = home.join(".algocline").join("cards");
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| format!("Failed to create cards dir: {e}"))?;
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
pub fn create(input: Json) -> Result<(String, PathBuf), String> {
    create_with_store(&default_store()?, input)
}

/// Create a new Card backed by `store`. See [`create`] for the default-store variant.
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

    Ok((card_id, path))
}

/// Read a Card by id. Returns None if not found.
pub fn get(card_id: &str) -> Result<Option<Json>, String> {
    get_with_store(&default_store()?, card_id)
}

/// Read a Card from `store`. See [`get`] for the default-store variant.
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

/// List cards. `pkg_filter = Some("name")` restricts to that pkg subdir.
pub fn list(pkg_filter: Option<&str>) -> Result<Vec<Summary>, String> {
    list_with_store(&default_store()?, pkg_filter)
}

/// List cards from `store`. See [`list`] for the default-store variant.
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
pub fn append(card_id: &str, fields: Json) -> Result<Json, String> {
    append_with_store(&default_store()?, card_id, fields)
}

/// Append to a Card in `store`. See [`append`] for the default-store variant.
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
    alias_set_with_store(&default_store()?, name, card_id, pkg, note)
}

/// Bind an alias in `store`. See [`alias_set`] for the default-store variant.
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
    Ok(entry)
}

/// Resolve an alias name to its bound Card and return the full Card JSON.
///
/// Shortcut for `alias_list → filter → get`. Returns `None` when the alias
/// does not exist. Errors when the alias points at a missing Card — that
/// would indicate a corrupt alias table (the target was deleted out of band).
pub fn get_by_alias(name: &str) -> Result<Option<Json>, String> {
    get_by_alias_with_store(&default_store()?, name)
}

/// Resolve an alias in `store`. See [`get_by_alias`] for the default-store variant.
pub fn get_by_alias_with_store(
    store: &dyn CardStore,
    name: &str,
) -> Result<Option<Json>, String> {
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

/// List aliases, optionally filtered by pkg.
pub fn alias_list(pkg_filter: Option<&str>) -> Result<Vec<Alias>, String> {
    alias_list_with_store(&default_store()?, pkg_filter)
}

/// List aliases from `store`. See [`alias_list`] for the default-store variant.
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
// See `workspace/tasks/card-dsl/design.md` for the full spec.
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
/// summary-level fields, uses the lightweight `list()` path to avoid
/// loading full TOML.  Otherwise loads full TOML per Card.
pub fn find(q: FindQuery) -> Result<Vec<Summary>, String> {
    find_with_store(&default_store()?, q)
}

/// Filter/sort Cards from `store`. See [`find`] for the default-store variant.
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

/// Walk the lineage tree from `q.card_id`.
pub fn lineage(q: LineageQuery) -> Result<Option<LineageResult>, String> {
    lineage_with_store(&default_store()?, q)
}

/// Walk the lineage tree in `store`. See [`lineage`] for the default-store variant.
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

/// Import Card files from `source_dir` into `~/.algocline/cards/{pkg}/`.
///
/// Copies `*.toml` and `*.samples.jsonl` files. Existing cards with the
/// same id are skipped (first-writer wins — Card immutability).
///
/// Returns `(imported, skipped)` card_id lists.
pub fn import_from_dir(
    source_dir: &std::path::Path,
    pkg: &str,
) -> Result<(Vec<String>, Vec<String>), String> {
    import_from_dir_with_store(&default_store()?, source_dir, pkg)
}

/// Import Card files into `store`. See [`import_from_dir`] for the default-store variant.
pub fn import_from_dir_with_store(
    store: &dyn CardStore,
    source_dir: &std::path::Path,
    pkg: &str,
) -> Result<(Vec<String>, Vec<String>), String> {
    store.import_from_dir(source_dir, pkg)
}

/// Write per-case samples to `{card_id}.samples.jsonl` (write-once).
///
/// Each `samples` entry is serialized as one compact JSON line.
/// Fails if a samples file already exists for this card — mirrors
/// the immutability guarantee of Cards themselves.
pub fn write_samples(card_id: &str, samples: Vec<Json>) -> Result<PathBuf, String> {
    write_samples_with_store(&default_store()?, card_id, samples)
}

/// Write samples via `store`. See [`write_samples`] for the default-store variant.
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
    store.write_samples_text(card_id, &buf)
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
pub fn read_samples(card_id: &str, q: SamplesQuery) -> Result<Vec<Json>, String> {
    read_samples_with_store(&default_store()?, card_id, q)
}

/// Read samples from `store`. See [`read_samples`] for the default-store variant.
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
// `root` defaults to `~/.algocline/cards/` via `from_home()`. Tests
// may use `new(tmpdir)` to redirect storage to a scratch directory.

/// File-backed implementation of [`CardStore`].
pub struct FileCardStore {
    root: PathBuf,
}

impl FileCardStore {
    /// Construct a store rooted at an explicit path.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Construct the default store rooted at `~/.algocline/cards/`.
    pub fn from_home() -> Result<Self, String> {
        Ok(Self { root: cards_dir()? })
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

        let entries = fs::read_dir(source_dir)
            .map_err(|e| format!("Failed to read card source dir: {e}"))?;

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
        if let Ok(store) = default_store() {
            if let Ok(d) = store.pkg_dir(pkg) {
                let _ = fs::remove_dir_all(&d);
            }
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
        let store = default_store().unwrap();
        let remaining: Vec<Alias> = store
            .read_aliases()
            .unwrap()
            .into_iter()
            .filter(|a| a.name != alias_name)
            .collect();
        store.write_aliases(&remaining).unwrap();
        cleanup(&pkg);
    }

    #[test]
    fn alias_set_rejects_unknown_card() {
        let err = alias_set("x", "does_not_exist_xyz", None, None).unwrap_err();
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

        // scenario eq via nested object
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "scenario": { "name": "gsm8k" },
            }))),
            order_by: order_from(json!("-stats.pass_rate")),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].pass_rate, Some(0.9));
        assert_eq!(rows[1].pass_rate, Some(0.4));

        // gte operator
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "stats": { "pass_rate": { "gte": 0.8 } },
            }))),
            order_by: order_from(json!("-stats.pass_rate")),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.pass_rate.unwrap() >= 0.8));

        // limit
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            order_by: order_from(json!("-stats.pass_rate")),
            limit: Some(1),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pass_rate, Some(1.0));

        cleanup(&pkg);
    }

    #[test]
    fn find_where_implicit_eq_and_logical() {
        let pkg = unique_pkg();
        create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-opus-4-6" },
            "stats": { "equilibrium_position": "dead", "survival_rate": 0.0 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_b"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-opus-4-6" },
            "stats": { "equilibrium_position": "niche_leader", "survival_rate": 1.0 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_c"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-haiku-4-5-20251001" },
            "stats": { "equilibrium_position": "fragile", "survival_rate": 0.2 }
        }))
        .unwrap();

        // implicit eq on sparse stats field
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "stats": { "equilibrium_position": "dead" },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].card_id.ends_with("_a"));

        // _or
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "_or": [
                    { "stats": { "equilibrium_position": "dead" } },
                    { "stats": { "survival_rate": { "gte": 0.9 } } },
                ],
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);

        // _not
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "_not": { "model": { "id": "claude-haiku-4-5-20251001" } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);

        // in operator
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "stats": {
                    "equilibrium_position": { "in": ["dead", "fragile"] },
                },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);

        // exists false (sparse field missing on haiku card? all have it, so test on
        // a field that only some have)
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "strategy_params": { "temperature": { "exists": false } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 3, "none of the cards have strategy_params");

        cleanup(&pkg);
    }

    #[test]
    fn find_order_by_multi_key() {
        let pkg = unique_pkg();
        create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_b"),
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.9 }
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_c"),
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.9 }
        }))
        .unwrap();

        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            order_by: order_from(json!(["-stats.pass_rate", "card_id"])),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].pass_rate, Some(0.9));
        assert_eq!(rows[1].pass_rate, Some(0.9));
        assert_eq!(rows[2].pass_rate, Some(0.5));
        // Tiebreak by card_id ascending
        assert!(rows[0].card_id < rows[1].card_id);

        cleanup(&pkg);
    }

    #[test]
    fn find_offset_and_limit() {
        let pkg = unique_pkg();
        for i in 0..5 {
            create(json!({
                "card_id": format!("{pkg}_{i}"),
                "pkg": { "name": pkg },
                "stats": { "pass_rate": 0.1 * (i + 1) as f64 }
            }))
            .unwrap();
        }

        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            order_by: order_from(json!("-stats.pass_rate")),
            offset: Some(1),
            limit: Some(2),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);
        // Best is 0.5, after offset=1 we start at 0.4 then 0.3.
        let pr0 = rows[0].pass_rate.unwrap();
        let pr1 = rows[1].pass_rate.unwrap();
        assert!((pr0 - 0.4).abs() < 1e-9, "got {pr0}");
        assert!((pr1 - 0.3).abs() < 1e-9, "got {pr1}");

        cleanup(&pkg);
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
        let pkg = unique_pkg();
        create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-opus-4-6" },
            "metadata": { "tag": "experiment_alpha" },
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_b"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-haiku-4-5-20251001" },
            "metadata": { "tag": "experiment_beta" },
        }))
        .unwrap();
        create(json!({
            "card_id": format!("{pkg}_c"),
            "pkg": { "name": pkg },
            "model": { "id": "claude-sonnet-4-5" },
            "metadata": { "tag": "baseline" },
        }))
        .unwrap();

        // contains: matches substring anywhere
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "metadata": { "tag": { "contains": "experiment" } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 2);

        // starts_with: matches only the prefix
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "model": { "id": { "starts_with": "claude-opus" } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].card_id.ends_with("_a"));

        // string ops on missing field → false
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "metadata": { "missing_field": { "contains": "x" } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 0);

        // string ops on non-string field → false
        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "metadata": { "tag": { "starts_with": 42 } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 0);

        cleanup(&pkg);
    }

    #[test]
    fn where_missing_field_ne_is_true() {
        let pkg = unique_pkg();
        create(json!({
            "card_id": format!("{pkg}_x"),
            "pkg": { "name": pkg },
        }))
        .unwrap();

        let rows = find(FindQuery {
            pkg: Some(pkg.clone()),
            where_: Some(where_from(json!({
                "strategy_params": { "temperature": { "ne": 0.5 } },
            }))),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(rows.len(), 1, "missing field is ne to anything");

        cleanup(&pkg);
    }

    // ─── lineage ───────────────────────────────────────────────

    /// Helper: create a child Card pointing at a parent with a relation.
    fn create_child(pkg: &str, suffix: &str, parent_id: &str, relation: &str) -> String {
        let (id, _) = create(json!({
            "card_id": format!("{pkg}_{suffix}"),
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 },
            "metadata": {
                "prior_card_id": parent_id,
                "prior_relation": relation,
            },
        }))
        .unwrap();
        id
    }

    #[test]
    fn lineage_up_walks_prior_card_id_chain() {
        let pkg = unique_pkg();
        // a → b → c (c is newest; b points at a; c points at b)
        let (a, _) = create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
        }))
        .unwrap();
        let b = create_child(&pkg, "b", &a, "rerun_of");
        let c = create_child(&pkg, "c", &b, "rerun_of");

        let res = lineage(LineageQuery {
            card_id: c.clone(),
            direction: LineageDirection::Up,
            depth: None,
            include_stats: false,
            relation_filter: None,
        })
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

        cleanup(&pkg);
    }

    #[test]
    fn lineage_down_walks_descendants_breadth_first() {
        let pkg = unique_pkg();
        // a has two children b, c; c has one child d.
        let (a, _) = create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
        }))
        .unwrap();
        let _b = create_child(&pkg, "b", &a, "sweep_variant");
        let c = create_child(&pkg, "c", &a, "sweep_variant");
        let _d = create_child(&pkg, "d", &c, "rerun_of");

        let res = lineage(LineageQuery {
            card_id: a.clone(),
            direction: LineageDirection::Down,
            depth: None,
            include_stats: false,
            relation_filter: None,
        })
        .unwrap()
        .expect("lineage result");

        // root + b + c + d = 4 nodes
        assert_eq!(res.nodes.len(), 4);
        assert_eq!(res.edges.len(), 3);
        assert!(!res.truncated);

        cleanup(&pkg);
    }

    #[test]
    fn lineage_depth_truncation_sets_flag() {
        let pkg = unique_pkg();
        let (a, _) = create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
        }))
        .unwrap();
        let b = create_child(&pkg, "b", &a, "rerun_of");
        let _c = create_child(&pkg, "c", &b, "rerun_of");

        let res = lineage(LineageQuery {
            card_id: a,
            direction: LineageDirection::Down,
            depth: Some(1),
            include_stats: false,
            relation_filter: None,
        })
        .unwrap()
        .unwrap();
        assert_eq!(res.nodes.len(), 2, "root + 1 level");
        assert!(res.truncated, "should be truncated at depth=1");

        cleanup(&pkg);
    }

    #[test]
    fn lineage_relation_filter_skips_unlisted() {
        let pkg = unique_pkg();
        let (a, _) = create(json!({
            "card_id": format!("{pkg}_a"),
            "pkg": { "name": pkg },
        }))
        .unwrap();
        let _b = create_child(&pkg, "b", &a, "sweep_variant");
        let _c = create_child(&pkg, "c", &a, "rerun_of");

        let res = lineage(LineageQuery {
            card_id: a,
            direction: LineageDirection::Down,
            depth: None,
            include_stats: false,
            relation_filter: Some(vec!["sweep_variant".to_string()]),
        })
        .unwrap()
        .unwrap();
        assert_eq!(res.nodes.len(), 2, "root + only sweep_variant child");
        assert_eq!(res.edges[0].relation.as_deref(), Some("sweep_variant"));

        cleanup(&pkg);
    }

    #[test]
    fn lineage_missing_card_returns_none() {
        let res = lineage(LineageQuery {
            card_id: "nonexistent_card_id_xyz".into(),
            direction: LineageDirection::Up,
            depth: None,
            include_stats: false,
            relation_filter: None,
        })
        .unwrap();
        assert!(res.is_none());
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

        let got = read_samples(&id, SamplesQuery::default()).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["case"], json!("c0"));
        assert_eq!(got[2]["score"], json!(0.75));

        // offset + limit
        let slice = read_samples(
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
        let got = read_samples(&id, SamplesQuery::default()).unwrap();
        assert!(got.is_empty());
        cleanup(&pkg);
    }

    #[test]
    fn read_samples_where_filters_rows() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({ "pkg": { "name": pkg } })).unwrap();
        write_samples(
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
        let got = read_samples(
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
        let got = read_samples(
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
        let slice = read_samples(
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

        cleanup(&pkg);
    }

    #[test]
    fn get_by_alias_roundtrip() {
        let pkg = unique_pkg();
        let (id, _) = create(json!({
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.85 }
        }))
        .unwrap();

        let alias_name = format!("best_{pkg}");
        alias_set(&alias_name, &id, Some(&pkg), None).unwrap();

        let card = get_by_alias(&alias_name).unwrap().unwrap();
        assert_eq!(card["card_id"], json!(id));
        assert_eq!(card["stats"]["pass_rate"], json!(0.85));

        assert!(get_by_alias("nonexistent_alias_xyz").unwrap().is_none());

        cleanup(&pkg);
    }

    #[test]
    fn samples_errors_on_missing_card() {
        let err = write_samples("does_not_exist_xyz_samples", vec![json!({})]).unwrap_err();
        assert!(err.contains("not found"));
    }

    // ─── import_from_dir ───────────────────────────────────────

    #[test]
    fn import_from_dir_copies_cards() {
        let pkg = unique_pkg();
        let tmp = tempfile::tempdir().unwrap();

        // Create a source card file
        let card_id = format!("{pkg}_imported");
        let card_content = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"{card_id}\"\npkg = \"{pkg}\"\n"
        );
        fs::write(tmp.path().join(format!("{card_id}.toml")), &card_content).unwrap();

        // Create a matching samples file
        fs::write(
            tmp.path().join(format!("{card_id}.samples.jsonl")),
            "{\"case\":\"c0\"}\n",
        )
        .unwrap();

        let (imported, skipped) = import_from_dir(tmp.path(), &pkg).unwrap();
        assert_eq!(imported, vec![card_id.clone()]);
        assert!(skipped.is_empty());

        // Verify card was imported
        let got = get(&card_id).unwrap().unwrap();
        assert_eq!(got["card_id"], json!(card_id));

        // Verify samples were copied
        let samples = read_samples(&card_id, SamplesQuery::default()).unwrap();
        assert_eq!(samples.len(), 1);

        cleanup(&pkg);
    }

    #[test]
    fn import_from_dir_skips_existing() {
        let pkg = unique_pkg();
        // Create a card in the store first
        let (existing_id, _) = create(json!({
            "pkg": { "name": pkg },
            "stats": { "pass_rate": 0.5 }
        }))
        .unwrap();

        // Try to import a card with the same id
        let tmp = tempfile::tempdir().unwrap();
        let card_content = format!(
            "schema_version = \"{SCHEMA_VERSION}\"\ncard_id = \"{existing_id}\"\npkg = \"{pkg}\"\n"
        );
        fs::write(
            tmp.path().join(format!("{existing_id}.toml")),
            &card_content,
        )
        .unwrap();

        let (imported, skipped) = import_from_dir(tmp.path(), &pkg).unwrap();
        assert!(imported.is_empty());
        assert_eq!(skipped, vec![existing_id.clone()]);

        // Original card untouched
        let got = get(&existing_id).unwrap().unwrap();
        assert_eq!(got["stats"]["pass_rate"], json!(0.5));

        cleanup(&pkg);
    }

    #[test]
    fn import_from_dir_skips_non_card_toml() {
        let pkg = unique_pkg();
        let tmp = tempfile::tempdir().unwrap();

        // A TOML file without schema_version = "card/v0" should be skipped
        fs::write(tmp.path().join("not_a_card.toml"), "title = \"hello\"\n").unwrap();

        let (imported, skipped) = import_from_dir(tmp.path(), &pkg).unwrap();
        assert!(imported.is_empty());
        assert!(skipped.is_empty());

        cleanup(&pkg);
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
        assert_eq!(card.get("card_id").and_then(|v| v.as_str()), Some(id.as_str()));

        let rows = list_with_store(&store, Some(pkg)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].card_id, id);

        // Ensure the default store is not polluted.
        let default_rows = list(Some(pkg)).unwrap();
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
        let samples_path = write_samples_with_store(
            &store,
            &id,
            vec![json!({ "case": "a", "pass": true })],
        )
        .unwrap();
        assert!(samples_path.starts_with(tmp.path()));
        let back = read_samples_with_store(&store, &id, SamplesQuery::default()).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].get("case").and_then(|v| v.as_str()), Some("a"));
    }
}
