//! `alc.nn.card.*` bridge — Card layer for the alc.nn spike (feature `nn`).
//!
//! Sits on top of the existing `alc.nn` primitives (safetensors save/load
//! + model registry) already registered by `super::register_nn`.
//!
//! Provides three Lua-facing entries under the `alc.nn.card` sub-table:
//!
//! ```text
//! alc.nn.card.save(vars, name, meta) -> card_id
//! alc.nn.card.load(card_id)          -> vars_table
//! alc.nn.card.register(card_id, model_name)
//! ```
//!
//! Invariants:
//!
//! 1. Card id is used verbatim as the safetensors bundle name — this
//!    module enforces the 1:1 mapping by pre-generating a Card id and
//!    passing it to both `alc.nn.save` (safetensors bundle) and
//!    `FileCardStore::create` (Card TOML). A mismatch surfaces as a
//!    Lua error rather than silent divergence.
//! 2. Store write failures propagate loudly (`?` / `LuaError::external`)
//!    per algocline `.claude/CLAUDE.md §Service 層の Error 伝播規律` —
//!    no `warn!` / `let _ = ...` / `.ok()` swallowing.
//! 3. `register` is idempotent — re-registering the same `model_name`
//!    overwrites the existing entry via `NnModelRegistry`'s underlying
//!    `HashMap::insert`.
//! 4. `load` returns an error (never partial state) when the Card's
//!    `metadata.nn.candle.bundle_ref` cannot be resolved on disk.

use std::sync::Arc;

use algocline_nn::card::{NnCandleBranch, NnCardMeta, NnLineage};
use mlua::prelude::*;
use mlua::LuaSerdeExt;
use serde_json::{json, Value as Json};

use crate::card::FileCardStore;

/// Pkg name under which nn Cards are stored
/// (`<cards_root>/alc_nn/<card_id>.toml`).
const NN_PKG: &str = "alc_nn";

/// Register `alc.nn.card.*` onto the pre-existing `alc.nn` table.
///
/// Must be called after [`super::register_nn`]; assumes `alc_table.nn`
/// is already populated by [`algocline_nn::module`]. Accepts the shared
/// `Arc<FileCardStore>` from [`super::BridgeConfig::card_store`] so
/// Cards persist through the same store as `alc.card.*`.
pub(super) fn register_nn_card(
    lua: &Lua,
    alc_table: &LuaTable,
    card_store: Arc<FileCardStore>,
) -> LuaResult<()> {
    let nn_table: LuaTable = alc_table.get("nn")?;
    let card_ns = lua.create_table()?;

    let save_store = Arc::clone(&card_store);
    let save = lua.create_function(
        move |lua, (vars, name, meta): (LuaTable, String, LuaTable)| -> LuaResult<String> {
            save_impl(lua, save_store.as_ref(), vars, &name, meta)
        },
    )?;
    card_ns.set("save", save)?;

    let load_store = Arc::clone(&card_store);
    let load = lua.create_function(move |lua, card_id: String| -> LuaResult<LuaTable> {
        load_impl(lua, load_store.as_ref(), &card_id)
    })?;
    card_ns.set("load", load)?;

    let register_store = Arc::clone(&card_store);
    let register = lua.create_function(
        move |lua, (card_id, model_name): (String, String)| -> LuaResult<()> {
            register_impl(lua, register_store.as_ref(), &card_id, &model_name)
        },
    )?;
    card_ns.set("register", register)?;

    nn_table.set("card", card_ns)?;
    Ok(())
}

/// Save the given `vars` as a safetensors bundle and record a Card
/// describing it.
///
/// Delegates the actual tensor dump to `alc.nn.save(vars, card_id)` so
/// this module never touches the [`algocline_nn::NnStoreHandle`]
/// directly — that keeps the save path 1:1 with the existing alc.nn
/// spike (test invariant: same store, same on-disk layout).
fn save_impl(
    lua: &Lua,
    store: &FileCardStore,
    vars: LuaTable,
    name: &str,
    meta: LuaTable,
) -> LuaResult<String> {
    let meta_json: Json = lua.from_value(LuaValue::Table(meta))?;

    // Pre-generate card_id so bundle_ref = "nn/<card_id>" is known
    // before we build the Card (invariant #1).
    let card_id = generate_card_id(name);

    // Delegate safetensors serialization to the existing alc.nn.save
    // path. Any store-side write failure propagates loudly through the
    // call chain (invariant #2).
    let nn_save: LuaFunction = alc_nn_fn(lua, "save")?;
    nn_save.call::<()>((vars, card_id.clone()))?;

    // Assemble the Card create payload with pkg + card_id + full
    // [metadata.nn] block already populated.
    let payload = build_create_payload(&card_id, name, &meta_json)?;

    // Store write — propagate error loudly (invariant #2).
    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| LuaError::external(format!("alc.nn.card.save: {e}")))?;

    // Sanity: the pre-generated id must match what create returned. A
    // divergence would silently break the safetensors ↔ Card 1:1
    // mapping (invariant #1) — surface it instead.
    if returned_id != card_id {
        return Err(LuaError::external(format!(
            "alc.nn.card.save: card_id mismatch (expected {card_id}, got {returned_id})"
        )));
    }
    Ok(card_id)
}

/// Load the safetensors bundle referenced by a Card and return the
/// rehydrated Vars as a Lua table (keyed by the original save names).
///
/// Refuses partial state: a Card without a resolvable
/// `metadata.nn.candle.bundle_ref` or with a bundle-not-on-disk
/// surfaces as a Lua error (invariant #4).
fn load_impl(lua: &Lua, store: &FileCardStore, card_id: &str) -> LuaResult<LuaTable> {
    let card = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load: card '{card_id}' not found"))
        })?;

    // Extract bundle_ref and enforce the "nn/<card_id>" shape imposed
    // by save_impl. A mismatch means the Card was hand-edited or built
    // by a different pipeline — refuse rather than guess the bundle.
    let bundle_ref = card
        .get("metadata")
        .and_then(|m| m.get("nn"))
        .and_then(|n| n.get("candle"))
        .and_then(|c| c.get("bundle_ref"))
        .and_then(|b| b.as_str())
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load: card '{card_id}' missing metadata.nn.candle.bundle_ref"
            ))
        })?;
    let expected = format!("nn/{card_id}");
    if bundle_ref != expected {
        return Err(LuaError::external(format!(
            "alc.nn.card.load: bundle_ref '{bundle_ref}' does not match card_id \
             '{card_id}' (expected '{expected}')"
        )));
    }

    // Delegate safetensors read to the existing alc.nn.load path. It
    // reads via the installed `NnStoreHandle` and errors clearly if the
    // bundle is missing on disk (invariant #4).
    let nn_load: LuaFunction = alc_nn_fn(lua, "load")?;
    let vars: LuaTable = nn_load.call(card_id.to_string())?;
    Ok(vars)
}

/// Register (or overwrite) a card_id → model_name alias in the
/// per-VM [`algocline_nn::NnModelRegistry`].
///
/// The Card foundation installs a small placeholder forward closure
/// so `alc.llm(role="nn", model=model_name)` returns a tagged
/// response pointing at the card. A later trainer follow-up replaces
/// this closure with the actual architecture forward pass —
/// idempotency (invariant #3) is guaranteed by `HashMap::insert` in
/// `NnModelRegistry`.
fn register_impl(
    lua: &Lua,
    store: &FileCardStore,
    card_id: &str,
    model_name: &str,
) -> LuaResult<()> {
    // Confirm the Card exists — registering an unknown card_id is
    // almost certainly a typo, so surface it loudly rather than
    // silently plant a broken alias.
    let exists = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.register: {e}")))?
        .is_some();
    if !exists {
        return Err(LuaError::external(format!(
            "alc.nn.card.register: card '{card_id}' not found"
        )));
    }

    // Placeholder forward closure. A later trainer follow-up
    // overwrites this via the same `alc.nn.register` entry once real
    // architecture inference lands.
    let placeholder_id = card_id.to_string();
    let forward = lua.create_function(move |_, prompt: String| -> LuaResult<String> {
        Ok(format!("[nn card {placeholder_id}]:{prompt}"))
    })?;

    let nn_register: LuaFunction = alc_nn_fn(lua, "register")?;
    nn_register.call::<()>((model_name.to_string(), forward))?;
    Ok(())
}

/// Build the JSON payload for `FileCardStore::create`. Required
/// fields from `meta` are `training_path` and `architecture`; the
/// rest are optional pass-through.
fn build_create_payload(card_id: &str, name: &str, user_meta: &Json) -> LuaResult<Json> {
    let training_path = user_meta
        .get("training_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LuaError::external("alc.nn.card.save: meta.training_path is required"))?
        .to_string();
    let architecture = user_meta
        .get("architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LuaError::external("alc.nn.card.save: meta.architecture is required"))?
        .to_string();
    let task = user_meta
        .get("task")
        .and_then(|v| v.as_str())
        .map(String::from);

    let lineage = match user_meta.get("lineage").cloned() {
        Some(v) => serde_json::from_value::<NnLineage>(v).map_err(|e| {
            LuaError::external(format!("alc.nn.card.save: invalid meta.lineage: {e}"))
        })?,
        None => NnLineage::default(),
    };

    // Normalise empty Lua tables (which mlua serializes as `[]`) into
    // an empty JSON object so downstream TOML serialization renders a
    // `[metadata.nn.hyperparams]` section (not `hyperparams = []`).
    let hyperparams = normalise_object(user_meta.get("hyperparams").cloned());
    let metrics = normalise_object(user_meta.get("metrics").cloned());

    let candle_in = user_meta.get("candle");
    let candle = NnCandleBranch {
        bundle_ref: format!("nn/{card_id}"),
        device: candle_in
            .and_then(|c| c.get("device"))
            .and_then(|v| v.as_str())
            .map(String::from),
        dtype: candle_in
            .and_then(|c| c.get("dtype"))
            .and_then(|v| v.as_str())
            .map(String::from),
        // LoRA is populated by the LoRA follow-up; the Card
        // foundation always leaves this None.
        lora: None,
    };

    let nn_meta = NnCardMeta {
        name: name.to_string(),
        backend: "candle".into(),
        task,
        architecture,
        training_path,
        lineage,
        hyperparams,
        metrics,
        candle: Some(candle),
    };

    let nn_meta_json = serde_json::to_value(&nn_meta).map_err(|e| {
        LuaError::external(format!("alc.nn.card.save: serialize meta: {e}"))
    })?;

    Ok(json!({
        "pkg": { "name": NN_PKG },
        "card_id": card_id,
        "metadata": {
            "kind": "nn_model",
            "nn": nn_meta_json,
        }
    }))
}

fn normalise_object(v: Option<Json>) -> Json {
    match v {
        Some(Json::Object(m)) => Json::Object(m),
        // Empty Lua table → serde_json Array([]) via mlua. Treat as
        // empty object; a non-empty array is a caller mistake but is
        // preserved for observability (better a broken TOML than silent
        // data loss).
        Some(Json::Array(a)) if a.is_empty() => Json::Object(serde_json::Map::new()),
        Some(other) => other,
        None => Json::Object(serde_json::Map::new()),
    }
}

/// Deterministic Card id derived from `name` + wall-clock microseconds.
///
/// Format: `<sanitized_name>_<epoch_us>`. Satisfies both
/// `FileCardStore::validate_name` (no `/` / `\` / `..` / `\0`) and
/// [`algocline_nn::FsStore`]'s stricter `[A-Za-z0-9_.-]` alphabet.
fn generate_card_id(name: &str) -> String {
    let ts = compact_epoch_us();
    let sanitized = sanitize_name(name);
    format!("{sanitized}_{ts}")
}

fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "nn".into()
    } else {
        out
    }
}

fn compact_epoch_us() -> String {
    // Clock-skew corner (`SystemTime` < `UNIX_EPOCH`) collapses to
    // `Duration::ZERO`; id collision then surfaces loudly through
    // `FileCardStore::write_new_card`'s immutable-card guard, not
    // silently, so the safety net is downstream rather than in this
    // helper's signature.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    // Microsecond-resolution suffix; keeps rapid successive save() calls
    // unique without pulling in a UUID crate.
    format!("{}{:06}", d.as_secs(), d.subsec_micros())
}

/// Fetch a function from the (already-registered) `alc.nn.*` table.
///
/// The `alc` table is placed on `_G` by the outer `bridge::register`
/// caller once every primitive is wired, so by the time these closures
/// fire from user Lua code the lookup path is stable.
fn alc_nn_fn(lua: &Lua, key: &str) -> LuaResult<LuaFunction> {
    let alc: LuaTable = lua.globals().get("alc").map_err(|e| {
        LuaError::external(format!("alc.nn.card: `alc` global missing: {e}"))
    })?;
    let nn: LuaTable = alc
        .get("nn")
        .map_err(|e| LuaError::external(format!("alc.nn.card: `alc.nn` missing: {e}")))?;
    nn.get::<LuaFunction>(key).map_err(|e| {
        LuaError::external(format!("alc.nn.card: `alc.nn.{key}` missing: {e}"))
    })
}
