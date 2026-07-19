//! `alc.nn.card.*` bridge — Card layer for the alc.nn spike (feature `nn`).
//!
//! Sits on top of the existing `alc.nn` primitives (safetensors save/load
//! + model registry) already registered by `super::register_nn`.
//!
//! Provides four Lua-facing entries under the `alc.nn.card` sub-table:
//!
//! ```text
//! alc.nn.card.save(vars, name, meta)         -> card_id
//! alc.nn.card.load(card_id)                  -> vars_table
//! alc.nn.card.load_gpt2(card_id, base)       -> Gpt2Handle (LoRA cards)
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
//!    per the crate's Service-layer error-propagation discipline —
//!    no `warn!` / `let _ = ...` / `.ok()` swallowing.
//! 3. `register` is idempotent — re-registering the same `model_name`
//!    overwrites the existing entry via `NnModelRegistry`'s underlying
//!    `HashMap::insert`.
//! 4. `load` returns an error (never partial state) when the Card's
//!    `metadata.nn.candle.bundle_ref` cannot be resolved on disk.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use algocline_nn::arch::{Gpt2Config, Gpt2Model, LoraConfig};
use algocline_nn::card::{NnCandleBranch, NnCardMeta, NnLineage, NnLoraBranch};
use algocline_nn::tokenizer::HfTokenizer;
use algocline_nn::train::{
    run_distill, run_full_ft, run_lora_ft, Batch, CrossEntropyLoss, Dataset, DatasetOpts,
    DistillLossKind, DistillSpec, FullFtConfig, JsonlDataset, ParquetDataset, ScheduleKind,
    TokenizedDataset, TrainError, TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use mlua::prelude::*;
use mlua::LuaSerdeExt;
use serde_json::{json, Value as Json};

use crate::card::{FileCardStore, SamplesQuery};

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
    nn_dir: PathBuf,
) -> LuaResult<()> {
    let nn_table: LuaTable = alc_table.get("nn")?;
    register_preset_ns(lua, &nn_table, nn_dir.clone())?;
    register_data_ns(lua, &nn_table, Arc::clone(&card_store), nn_dir.clone())?;
    register_trainer_ns(lua, &nn_table, nn_dir.clone())?;
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

    let load_gpt2_store = Arc::clone(&card_store);
    let load_gpt2 = lua.create_function(
        move |_lua, (card_id, base_handle): (String, LuaAnyUserData)| -> LuaResult<Gpt2Handle> {
            load_gpt2_impl(load_gpt2_store.as_ref(), &card_id, &base_handle)
        },
    )?;
    card_ns.set("load_gpt2", load_gpt2)?;

    let register_store = Arc::clone(&card_store);
    let register = lua.create_function(
        move |lua, (card_id, model_name): (String, String)| -> LuaResult<()> {
            register_impl(lua, register_store.as_ref(), &card_id, &model_name)
        },
    )?;
    card_ns.set("register", register)?;

    nn_table.set("card", card_ns)?;
    // `nn_dir` was cloned into each register_* call above; drop the
    // outer binding explicitly so future edits that add new
    // `nn_dir`-consuming registrations trip a compiler error rather
    // than silently miss a use.
    let _ = nn_dir;
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

/// Reconstruct a LoRA-wrapped [`Gpt2Handle`] from a Card + a fresh
/// base handle.
///
/// The Card must carry a `[metadata.nn.candle.lora]` block populated
/// by [`alc.nn.trainer.lora`]; a card without one errors loudly
/// rather than silently returning the base handle unchanged. Callers
/// who want weight-only rehydration of a full-FT Card should call
/// `alc.nn.card.load` (returns the raw vars table) instead.
///
/// The base handle is **mutated in place**: `wrap_lora` replaces each
/// target `Linear` layer with a `LoraLinear` inside the shared
/// `Gpt2Model`. Callers must therefore pass a *fresh* base handle
/// (e.g. from `alc.nn.preset.gpt2(...)`) — re-using the same base
/// handle for two consecutive `load_gpt2` calls fails with the
/// "expected Plain, got Lora" error surfaced by
/// [`Gpt2Model::wrap_lora`].
///
/// Merge-equivalence invariant: a forward pass through the returned
/// handle matches `LoraLinear(base + Δ).forward` element-wise —
/// candle-nn's [`VarMap::load`] restores only the vars whose names
/// match those registered by `wrap_lora`, so the base parameters
/// stay bit-identical.
fn load_gpt2_impl(
    store: &FileCardStore,
    card_id: &str,
    base_handle: &LuaAnyUserData,
) -> LuaResult<Gpt2Handle> {
    // Fetch the Card and locate the [nn.candle.lora] block.
    let card = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load_gpt2: card '{card_id}' not found"))
        })?;
    let candle_json = card
        .get("metadata")
        .and_then(|m| m.get("nn"))
        .and_then(|n| n.get("candle"))
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load_gpt2: card '{card_id}' missing metadata.nn.candle"
            ))
        })?;
    let lora_json = candle_json.get("lora").ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_gpt2: card '{card_id}' has no metadata.nn.candle.lora block \
             (use alc.nn.card.load for weight-only reload of a non-LoRA card)"
        ))
    })?;

    // Schema validation happens first — deserialize the lora branch
    // and require `delta_path` + file existence before we touch the
    // base handle. This ordering matters for two reasons: (1) tests
    // that assert schema-level errors (missing `delta_path`, missing
    // delta file, etc.) can use a placeholder userdata for
    // `base_handle`; (2) real callers see the most specific error
    // first (schema gap vs base/architecture pairing).
    let lora_branch: NnLoraBranch = serde_json::from_value(lora_json.clone()).map_err(|e| {
        LuaError::external(format!(
            "alc.nn.card.load_gpt2: invalid metadata.nn.candle.lora: {e}"
        ))
    })?;
    let delta_path_str = lora_branch.delta_path.clone().ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_gpt2: card '{card_id}' metadata.nn.candle.lora is missing \
             delta_path (pre-ST-d cards do not record it; re-save via alc.nn.trainer.lora + \
             alc.nn.card.save to populate)"
        ))
    })?;
    let delta_path = PathBuf::from(&delta_path_str);
    if !delta_path.exists() {
        return Err(LuaError::external(format!(
            "alc.nn.card.load_gpt2: delta safetensors missing at {delta_path:?} \
             (expected the file produced by run_lora_ft; ckpt_dir may have been cleaned)"
        )));
    }

    let card_arch = card
        .get("metadata")
        .and_then(|m| m.get("nn"))
        .and_then(|n| n.get("architecture"))
        .and_then(|a| a.as_str())
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load_gpt2: card '{card_id}' missing metadata.nn.architecture"
            ))
        })?;
    let base_variant_snapshot = {
        let base_snap = base_handle
            .borrow::<Gpt2Handle>()
            .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: base handle: {e}")))?;
        base_snap.variant.clone()
    };
    let base_cfg_id = if base_variant_snapshot.starts_with("gpt2-") {
        base_variant_snapshot.clone()
    } else {
        format!("gpt2-{base_variant_snapshot}")
    };
    if card_arch != base_cfg_id && card_arch != base_variant_snapshot {
        return Err(LuaError::external(format!(
            "alc.nn.card.load_gpt2: architecture mismatch — card '{card_id}' was trained on \
             '{card_arch}' but base handle is '{base_variant_snapshot}'. Rebuild the base with \
             `alc.nn.preset.gpt2('{card_arch}', ...)` to match."
        )));
    }

    // Reconstruct the LoraConfig that produced this Δ. `target_modules`
    // + rank + alpha must match the training-time values verbatim
    // so `VarMap::load` finds every var by name. `dropout` is set for
    // provenance only — [`crate::arch::LoraLinear::forward`] does not
    // apply dropout at inference, so this field does not affect the
    // element-wise bit-exact merge-equivalence invariant.
    let mut lora_cfg = LoraConfig::with_targets(
        lora_branch.rank as usize,
        lora_branch.alpha as f32,
        lora_branch.target_modules.iter().cloned(),
    );
    lora_cfg.dropout = lora_branch.dropout;

    // Snapshot the base handle's identity fields before we borrow its
    // model for mutation — the returned Gpt2Handle carries the same
    // shape metadata (variant, layers, etc.) as the base.
    let base = base_handle
        .borrow::<Gpt2Handle>()
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: base handle: {e}")))?;
    let model_arc = base.model();
    let variant = base.variant.clone();
    let layers = base.layers;
    let heads = base.heads;
    let dim = base.dim;
    let ctx = base.ctx;
    let vocab = base.vocab;
    let device = base.device.clone();
    let dtype = base.dtype.clone();
    let pretrained = base.pretrained;
    drop(base);

    // Wrap the base in place. `wrap_lora` returns a fresh VarMap
    // holding only the LoRA A/B parameters; the base weights stay
    // frozen (invariant #1). Wrong `target_modules` (e.g. an entry
    // the model does not expose) surface as a Candle error prefixed
    // with the trainer stage rather than a silent no-op.
    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: model lock: {e}")))?;
    let mut lora_vm = model
        .wrap_lora(&lora_cfg)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: wrap_lora: {e}")))?;
    drop(model);

    // Restore the trained Δ into the freshly-wrapped LoRA vars. Names
    // registered by wrap_lora must match names in the safetensors
    // file; a mismatch surfaces via candle-nn's `VarMap::load` error
    // path (either missing-name or shape-mismatch).
    lora_vm.load(&delta_path).map_err(|e| {
        LuaError::external(format!(
            "alc.nn.card.load_gpt2: load delta {delta_path:?}: {e}"
        ))
    })?;

    Ok(Gpt2Handle {
        inner: model_arc,
        varmap: Some(Arc::new(lora_vm)),
        variant,
        layers,
        heads,
        dim,
        ctx,
        vocab,
        device,
        dtype,
        pretrained,
    })
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

    // LoRA sub-branch is optional: only `alc.nn.trainer.lora` populates
    // it via the returned Checkpoint table's `lora` field, which the
    // caller then threads back through `meta.candle.lora` when saving
    // the Card. `full_ft` / `distill` callers omit it (Card foundation
    // invariant: only lora-trained models carry `NnLoraBranch`).
    let lora = match candle_in.and_then(|c| c.get("lora")) {
        Some(v) if !v.is_null() => Some(
            serde_json::from_value::<NnLoraBranch>(v.clone()).map_err(|e| {
                LuaError::external(format!("alc.nn.card.save: invalid meta.candle.lora: {e}"))
            })?,
        ),
        _ => None,
    };

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
        lora,
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

    let nn_meta_json = serde_json::to_value(&nn_meta)
        .map_err(|e| LuaError::external(format!("alc.nn.card.save: serialize meta: {e}")))?;

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
    let alc: LuaTable = lua
        .globals()
        .get("alc")
        .map_err(|e| LuaError::external(format!("alc.nn.card: `alc` global missing: {e}")))?;
    let nn: LuaTable = alc
        .get("nn")
        .map_err(|e| LuaError::external(format!("alc.nn.card: `alc.nn` missing: {e}")))?;
    nn.get::<LuaFunction>(key)
        .map_err(|e| LuaError::external(format!("alc.nn.card: `alc.nn.{key}` missing: {e}")))
}

// ─── alc.nn.preset ────────────────────────────────────────────────

/// Opaque handle exposed to Lua for a constructed GPT-2 model.
///
/// Wrapped in `Arc<Mutex<...>>` so `#[cfg(feature = "nn")]` builds can
/// keep sending the handle across `mlua`'s `send`-required boundary.
/// A later trainer follow-up replaces the `Option` with a mutable
/// trainer wrap; this stage only needs read access for shape
/// assertions from Lua.
pub(super) struct Gpt2Handle {
    inner: Arc<Mutex<Gpt2Model>>,
    /// `VarMap` the model's parameters were registered against.
    ///
    /// Populated only for from-scratch handles (`pretrained = false`)
    /// because [`Gpt2Model::from_pretrained`] loads its weights through
    /// an mmap-backed [`candle_nn::VarBuilder`] that has no `VarMap`
    /// counterpart on the caller side. `full_ft` / `distill` bindings
    /// require this field and error out on `None`; `lora` binding does
    /// not — [`Gpt2Model::wrap_lora`] builds its own delta `VarMap`
    /// internally and leaves the base parameters frozen.
    ///
    /// Wrapped in `Arc<VarMap>` rather than `Arc<Mutex<VarMap>>`:
    /// `VarMap::all_vars` is `&self`, and the candle-nn optimizer
    /// (`AdamW`) only needs read access to build its parameter list.
    /// The concurrency guard is the [`TrainingLease`] one layer up,
    /// which structurally prevents overlapping training sessions
    /// against the same handle.
    varmap: Option<Arc<VarMap>>,
    variant: String,
    layers: usize,
    heads: usize,
    dim: usize,
    ctx: usize,
    vocab: usize,
    device: String,
    dtype: String,
    pretrained: bool,
}

impl mlua::UserData for Gpt2Handle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("variant", |_, this, ()| Ok(this.variant.clone()));
        methods.add_method("layers", |_, this, ()| Ok(this.layers));
        methods.add_method("heads", |_, this, ()| Ok(this.heads));
        methods.add_method("dim", |_, this, ()| Ok(this.dim));
        methods.add_method("ctx", |_, this, ()| Ok(this.ctx));
        methods.add_method("vocab", |_, this, ()| Ok(this.vocab));
        methods.add_method("device", |_, this, ()| Ok(this.device.clone()));
        methods.add_method("dtype", |_, this, ()| Ok(this.dtype.clone()));
        methods.add_method("pretrained", |_, this, ()| Ok(this.pretrained));
        methods.add_method("forward_shape", |_, this, (batch, seq): (usize, usize)| {
            Ok(vec![batch, seq, this.vocab])
        });
    }
}

impl Gpt2Handle {
    /// Shared handle to the underlying model. The trainer bindings
    /// lock this for the duration of a `run_full_ft` / `run_lora_ft` /
    /// `run_distill` call; concurrent access is barred one layer up by
    /// [`TrainingLease`].
    pub(super) fn model(&self) -> Arc<Mutex<Gpt2Model>> {
        Arc::clone(&self.inner)
    }

    /// Shared handle to the model's `VarMap`, if constructed from
    /// scratch (see the field-level rationale). Returns `None` for
    /// `pretrained = true` handles, in which case `full_ft` and
    /// `distill` bindings surface a clear Lua-side error rather than
    /// panic or silently no-op.
    pub(super) fn varmap(&self) -> Option<Arc<VarMap>> {
        self.varmap.as_ref().map(Arc::clone)
    }
}

fn register_preset_ns(lua: &Lua, nn_table: &LuaTable, nn_dir: PathBuf) -> LuaResult<()> {
    let preset = lua.create_table()?;

    let gpt2 = lua.create_function(
        move |_lua, (variant, opts): (String, Option<LuaTable>)| -> LuaResult<Gpt2Handle> {
            build_gpt2_handle(&variant, opts.as_ref(), &nn_dir)
        },
    )?;
    preset.set("gpt2", gpt2)?;
    nn_table.set("preset", preset)?;
    Ok(())
}

fn build_gpt2_handle(
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<Gpt2Handle> {
    let mut cfg = Gpt2Config::from_variant(variant).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.preset.gpt2: unknown variant '{variant}' (expected 'medium' or 'large')"
        ))
    })?;

    let device_str = opts
        .and_then(|t| t.get::<Option<String>>("device").ok().flatten())
        .unwrap_or_else(|| "cpu".to_string());
    let dtype_str = opts
        .and_then(|t| t.get::<Option<String>>("dtype").ok().flatten())
        .unwrap_or_else(|| {
            // Design §6.1 default: bf16 on CUDA, f32 on CPU.
            if device_str.starts_with("cuda") {
                "bf16".to_string()
            } else {
                "f32".to_string()
            }
        });
    let pretrained = opts
        .and_then(|t| t.get::<Option<bool>>("pretrained").ok().flatten())
        .unwrap_or(true);

    cfg.device = parse_device(&device_str)?;
    cfg.dtype = parse_dtype(&dtype_str)?;

    // Guard the "bf16 on CPU" combo: error early rather than let
    // candle emit an obscure kernel error downstream.
    if matches!(cfg.device, Device::Cpu) && matches!(cfg.dtype, DType::BF16) {
        return Err(LuaError::external(
            "alc.nn.preset.gpt2: bf16 dtype requires a CUDA device (use dtype='f32' on CPU)"
                .to_string(),
        ));
    }

    let (model, varmap) = if pretrained {
        let cache_dir = nn_dir.to_path_buf();
        let m = Gpt2Model::from_pretrained(variant, &cfg, &cache_dir)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.gpt2: {e}")))?;
        // `from_pretrained` loads through an mmap-backed VarBuilder that
        // has no VarMap counterpart on the caller side. Downstream
        // trainer bindings that need one (`full_ft` / `distill`) error
        // out cleanly when the handle carries `None` (§field docs).
        (m, None)
    } else {
        let vm = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let m = Gpt2Model::new(&cfg, vs)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.gpt2: {e}")))?;
        (m, Some(Arc::new(vm)))
    };

    Ok(Gpt2Handle {
        inner: Arc::new(Mutex::new(model)),
        varmap,
        variant: variant.to_string(),
        layers: cfg.layers,
        heads: cfg.heads,
        dim: cfg.dim,
        ctx: cfg.ctx,
        vocab: cfg.vocab,
        device: device_str,
        dtype: dtype_str,
        pretrained,
    })
}

fn parse_device(s: &str) -> LuaResult<Device> {
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let ord: usize = rest.parse().map_err(|e| {
            LuaError::external(format!(
                "alc.nn.preset.gpt2: invalid cuda ordinal '{rest}': {e}"
            ))
        })?;
        return Device::new_cuda(ord).map_err(|e| {
            LuaError::external(format!("alc.nn.preset.gpt2: cuda:{ord} unavailable: {e}"))
        });
    }
    if s == "cuda" {
        return Device::new_cuda(0)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.gpt2: cuda unavailable: {e}")));
    }
    Err(LuaError::external(format!(
        "alc.nn.preset.gpt2: unknown device '{s}' (expected 'cpu', 'cuda', or 'cuda:N')"
    )))
}

fn parse_dtype(s: &str) -> LuaResult<DType> {
    match s {
        "f32" | "fp32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        "f16" | "fp16" => Ok(DType::F16),
        other => Err(LuaError::external(format!(
            "alc.nn.preset.gpt2: unknown dtype '{other}' (expected 'f32', 'bf16', or 'f16')"
        ))),
    }
}

// ─── alc.nn.data ──────────────────────────────────────────────────

/// Lua userdata handle around a `Box<dyn Dataset>`.
///
/// The `Send` bound is satisfied by wrapping the `Box` in a `Mutex` so
/// mlua's `send` feature accepts the type across VM boundaries. The
/// trainer follow-up pulls batches out of this handle inside the
/// training loop.
pub(super) struct DatasetHandle {
    inner: Mutex<Box<dyn Dataset + Send>>,
    source: String,
    batch_size: usize,
    ctx_len: usize,
}

impl mlua::UserData for DatasetHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("source", |_, this, ()| Ok(this.source.clone()));
        methods.add_method("batch_size", |_, this, ()| Ok(this.batch_size));
        methods.add_method("ctx_len", |_, this, ()| Ok(this.ctx_len));
        methods.add_method("len_hint", |_, this, ()| {
            let ds = this.inner.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.data: dataset lock poisoned: {e}"))
            })?;
            Ok(ds.len_hint())
        });
        methods.add_method_mut(
            "next_batch",
            |lua, this, ()| -> LuaResult<Option<LuaTable>> {
                let mut ds = this.inner.lock().map_err(|e| {
                    LuaError::external(format!("alc.nn.data: dataset lock poisoned: {e}"))
                })?;
                match ds
                    .next_batch()
                    .map_err(|e| LuaError::external(format!("alc.nn.data.next_batch: {e}")))?
                {
                    Some(batch) => Ok(Some(batch_to_lua(lua, batch)?)),
                    None => Ok(None),
                }
            },
        );
    }
}

fn batch_to_lua(lua: &Lua, batch: Batch) -> LuaResult<LuaTable> {
    let out = lua.create_table()?;
    let rows = lua.create_table()?;
    for (i, row) in batch.input_ids.into_iter().enumerate() {
        let arr = lua.create_table()?;
        for (j, id) in row.into_iter().enumerate() {
            arr.set(j + 1, id)?;
        }
        rows.set(i + 1, arr)?;
    }
    out.set("input_ids", rows)?;
    out.set("is_last", batch.is_last)?;
    Ok(out)
}

fn register_data_ns(
    lua: &Lua,
    nn_table: &LuaTable,
    card_store: Arc<FileCardStore>,
    nn_dir: PathBuf,
) -> LuaResult<()> {
    let data = lua.create_table()?;

    // jsonl(path, opts) — streams a JSONL file, tokenizing each row.
    let jsonl_tok_dir = nn_dir.join("tokenizers");
    let jsonl = lua.create_function(
        move |_lua, (path, opts): (String, Option<LuaTable>)| -> LuaResult<DatasetHandle> {
            let dopts = extract_dataset_opts(opts.as_ref())?;
            let tokenizer_name = opts
                .as_ref()
                .and_then(|t| t.get::<Option<String>>("tokenizer").ok().flatten())
                .unwrap_or_else(|| "gpt2".to_string());
            let tok = HfTokenizer::load_cached(&tokenizer_name, &jsonl_tok_dir)
                .map_err(|e| LuaError::external(format!("alc.nn.data.jsonl: {e}")))?;
            let ds = JsonlDataset::new(std::path::Path::new(&path), dopts.clone(), tok)
                .map_err(|e| LuaError::external(format!("alc.nn.data.jsonl: {e}")))?;
            Ok(DatasetHandle {
                inner: Mutex::new(Box::new(ds)),
                source: format!("jsonl:{path}"),
                batch_size: dopts.batch_size,
                ctx_len: dopts.ctx_len,
            })
        },
    )?;
    data.set("jsonl", jsonl)?;

    // parquet(path, opts) — scaffold only; iteration surfaces
    // NotImplemented until a later stage wires the reader.
    let parquet = lua.create_function(
        move |_lua, (path, opts): (String, Option<LuaTable>)| -> LuaResult<DatasetHandle> {
            let dopts = extract_dataset_opts(opts.as_ref())?;
            let ds = ParquetDataset::new(std::path::Path::new(&path), dopts.clone());
            Ok(DatasetHandle {
                inner: Mutex::new(Box::new(ds)),
                source: format!("parquet:{path}"),
                batch_size: dopts.batch_size,
                ctx_len: dopts.ctx_len,
            })
        },
    )?;
    data.set("parquet", parquet)?;

    // from_card(card_id, opts) — read Card samples via
    // FileCardStore (invariant #5), tokenize `prompt` / `response`
    // pairs, and build an in-memory `TokenizedDataset`.
    let from_card_store = Arc::clone(&card_store);
    let from_card_tok_dir = nn_dir.join("tokenizers");
    let from_card = lua.create_function(
        move |_lua, (card_id, opts): (String, Option<LuaTable>)| -> LuaResult<DatasetHandle> {
            let dopts = extract_dataset_opts(opts.as_ref())?;
            let tokenizer_name = opts
                .as_ref()
                .and_then(|t| t.get::<Option<String>>("tokenizer").ok().flatten())
                .unwrap_or_else(|| "gpt2".to_string());
            let tok = HfTokenizer::load_cached(&tokenizer_name, &from_card_tok_dir)
                .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?;

            let samples = from_card_store
                .read_samples(&card_id, SamplesQuery::default())
                .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?;

            // Tokenize "prompt\nresponse" per sample. Empty rows are
            // skipped so an in-progress teacher log without complete
            // pairs still yields a usable dataset.
            let mut rows: Vec<Vec<u32>> = Vec::with_capacity(samples.len());
            for (idx, sample) in samples.into_iter().enumerate() {
                let prompt = sample
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let response = sample
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let text = if !prompt.is_empty() && !response.is_empty() {
                    format!("{prompt}\n{response}")
                } else if !prompt.is_empty() {
                    prompt.to_string()
                } else if !response.is_empty() {
                    response.to_string()
                } else {
                    continue;
                };
                let ids = tok.encode(&text).map_err(|e| {
                    LuaError::external(format!("alc.nn.data.from_card: sample {idx}: {e}"))
                })?;
                if !ids.is_empty() {
                    rows.push(ids);
                }
            }

            let ds = TokenizedDataset::new(rows, dopts.clone());
            Ok(DatasetHandle {
                inner: Mutex::new(Box::new(ds)),
                source: format!("card:{card_id}"),
                batch_size: dopts.batch_size,
                ctx_len: dopts.ctx_len,
            })
        },
    )?;
    data.set("from_card", from_card)?;

    // synthetic(rows, opts) — build a TokenizedDataset directly from
    // a Lua table of pre-tokenized u32 sequences, skipping the
    // tokenizer path. Purpose: enable CPU smoke examples
    // (`examples/nn_*_smoke.lua`) with the `tiny` preset, whose
    // vocab=64 has no matching HuggingFace tokenizer. Each row is a
    // Lua array of integers in `[0, vocab)` — out-of-range ids
    // surface as an index-select error inside the model forward
    // rather than here (this binding only enforces integer typing).
    let synthetic = lua.create_function(
        move |_lua, (rows_tbl, opts): (LuaTable, Option<LuaTable>)| -> LuaResult<DatasetHandle> {
            let dopts = extract_dataset_opts(opts.as_ref())?;
            let row_count = rows_tbl.raw_len();
            if row_count == 0 {
                return Err(LuaError::external(
                    "alc.nn.data.synthetic: rows must be a non-empty array of token id \
                     sequences (each row itself an array of u32)"
                        .to_string(),
                ));
            }
            let mut rows: Vec<Vec<u32>> = Vec::with_capacity(row_count);
            for i in 1..=row_count {
                let row: LuaTable = rows_tbl.get(i).map_err(|e| {
                    LuaError::external(format!("alc.nn.data.synthetic: row {i} not a table: {e}"))
                })?;
                let len = row.raw_len();
                if len == 0 {
                    return Err(LuaError::external(format!(
                        "alc.nn.data.synthetic: row {i} is empty (need at least 1 token)"
                    )));
                }
                let mut ids: Vec<u32> = Vec::with_capacity(len);
                for j in 1..=len {
                    let id: u32 = row.get(j).map_err(|e| {
                        LuaError::external(format!(
                            "alc.nn.data.synthetic: row {i} token {j} not a u32 integer: {e}"
                        ))
                    })?;
                    ids.push(id);
                }
                rows.push(ids);
            }
            let ds = TokenizedDataset::new(rows, dopts.clone());
            Ok(DatasetHandle {
                inner: Mutex::new(Box::new(ds)),
                source: format!("synthetic:{row_count}rows"),
                batch_size: dopts.batch_size,
                ctx_len: dopts.ctx_len,
            })
        },
    )?;
    data.set("synthetic", synthetic)?;

    nn_table.set("data", data)?;
    Ok(())
}

fn extract_dataset_opts(opts: Option<&LuaTable>) -> LuaResult<DatasetOpts> {
    let mut d = DatasetOpts::default();
    if let Some(t) = opts {
        if let Some(v) = t.get::<Option<usize>>("batch_size")? {
            d.batch_size = v;
        }
        if let Some(v) = t.get::<Option<usize>>("ctx_len")? {
            d.ctx_len = v;
        }
        if let Some(v) = t.get::<Option<bool>>("shuffle")? {
            d.shuffle = v;
        }
        if let Some(v) = t.get::<Option<u32>>("pad_id")? {
            d.pad_id = v;
        }
        if let Some(v) = t.get::<Option<String>>("text_field")? {
            d.text_field = v;
        }
    }
    if d.batch_size == 0 {
        return Err(LuaError::external(
            "alc.nn.data: batch_size must be >= 1".to_string(),
        ));
    }
    if d.ctx_len == 0 {
        return Err(LuaError::external(
            "alc.nn.data: ctx_len must be >= 1".to_string(),
        ));
    }
    Ok(d)
}

// ─── alc.nn.trainer ───────────────────────────────────────────────

/// Register `alc.nn.trainer.{full_ft, lora, distill}` onto the
/// pre-existing `alc.nn` table.
///
/// Symmetric to [`register_data_ns`]; the three bindings share:
///
/// - a single per-VM [`TrainingLease`] (design's "one training session
///   per VM" invariant — [`run_full_ft`] / [`run_lora_ft`] /
///   [`run_distill`] all `acquire()` it internally),
/// - the `FullFtConfig` opts-table extractor ([`extract_full_ft_opts`]),
/// - a common [`ckpt_from_lease_result`] converter that turns
///   [`TrainError`] into `mlua::Error` and the returned [`Checkpoint`]
///   into a Lua table with primitive fields plus a metrics sub-table.
///
/// The `lora` binding additionally attaches a `lora = { rank, alpha,
/// base_bundle_ref }` sub-table to the returned Checkpoint so callers
/// can thread it back through `alc.nn.card.save`'s `meta.candle.lora`
/// field (invariant: `NnCandleBranch::lora` populate is a lora-only
/// concern; `full_ft` and `distill` never emit it).
fn register_trainer_ns(lua: &Lua, nn_table: &LuaTable, nn_dir: PathBuf) -> LuaResult<()> {
    let trainer = lua.create_table()?;
    let lease = Arc::new(TrainingLease::new());

    let full_ft_lease = Arc::clone(&lease);
    let full_ft_dir = nn_dir.clone();
    let full_ft = lua.create_function(
        move |lua,
              (handle, dataset, opts): (LuaAnyUserData, LuaAnyUserData, Option<LuaTable>)|
              -> LuaResult<LuaTable> {
            full_ft_impl(
                lua,
                &handle,
                &dataset,
                opts.as_ref(),
                &full_ft_dir,
                Arc::clone(&full_ft_lease),
            )
        },
    )?;
    trainer.set("full_ft", full_ft)?;

    let lora_lease = Arc::clone(&lease);
    let lora_dir = nn_dir.clone();
    let lora = lua.create_function(
        move |lua,
              (handle, dataset, opts): (LuaAnyUserData, LuaAnyUserData, Option<LuaTable>)|
              -> LuaResult<LuaTable> {
            lora_impl(
                lua,
                &handle,
                &dataset,
                opts.as_ref(),
                &lora_dir,
                Arc::clone(&lora_lease),
            )
        },
    )?;
    trainer.set("lora", lora)?;

    let distill_lease = Arc::clone(&lease);
    let distill_dir = nn_dir;
    let distill = lua.create_function(
        move |lua,
              (handle, dataset, opts): (LuaAnyUserData, LuaAnyUserData, Option<LuaTable>)|
              -> LuaResult<LuaTable> {
            distill_impl(
                lua,
                &handle,
                &dataset,
                opts.as_ref(),
                &distill_dir,
                Arc::clone(&distill_lease),
            )
        },
    )?;
    trainer.set("distill", distill)?;

    nn_table.set("trainer", trainer)?;
    Ok(())
}

/// `alc.nn.trainer.full_ft(handle, dataset, opts?) -> Checkpoint`.
///
/// Requires a from-scratch handle (`opts.pretrained = false` at
/// [`build_gpt2_handle`] time) — the `VarMap` the optimizer needs
/// doesn't exist for pretrained handles today (see [`Gpt2Handle`]).
/// Surfaces that as a Lua-side error before touching the training
/// lease.
fn full_ft_impl(
    lua: &Lua,
    handle: &LuaAnyUserData,
    dataset: &LuaAnyUserData,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    lease: Arc<TrainingLease>,
) -> LuaResult<LuaTable> {
    let cfg = extract_full_ft_opts(opts)?;
    let (ckpt_dir, ckpt_prefix) = resolve_ckpt_dest(opts, nn_dir, "full_ft")?;
    let card_id = ckpt_prefix.clone();

    let gpt2 = handle.borrow::<Gpt2Handle>()?;
    let model_arc = gpt2.model();
    let vm_arc = gpt2.varmap().ok_or_else(|| {
        LuaError::external(
            "alc.nn.trainer.full_ft: handle was built with pretrained=true; \
             full-fine-tune requires a from-scratch handle (pretrained=false)"
                .to_string(),
        )
    })?;
    drop(gpt2);

    let ds_guard = dataset.borrow_mut::<DatasetHandle>()?;
    let mut ds_lock = ds_guard
        .inner
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.full_ft: dataset lock: {e}")))?;

    let loss_fn = CrossEntropyLoss::new();
    let model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.full_ft: model lock: {e}")))?;

    let result = run_full_ft(
        &model,
        &vm_arc,
        ds_lock.as_mut(),
        &cfg,
        &loss_fn,
        &ckpt_dir,
        &ckpt_prefix,
        lease,
    );
    drop(model);
    drop(ds_lock);
    drop(ds_guard);

    let ckpt = result.map_err(train_err_to_lua)?;
    checkpoint_to_lua(lua, &ckpt, &card_id, None)
}

/// `alc.nn.trainer.lora(handle, dataset, opts) -> Checkpoint`.
///
/// `opts.rank` and `opts.alpha` are required. `opts.target_modules`
/// defaults to the canonical GPT-2 set (`q_proj` / `k_proj` / `v_proj`
/// / `o_proj` / `up` / `down`). The returned Checkpoint carries a
/// `lora = { rank, alpha, base_bundle_ref }` sub-table so callers can
/// thread it through `alc.nn.card.save(vars, name, { candle = { lora
/// = ckpt.lora } })` without repeating the values.
///
/// **Not idempotent per handle**: [`Gpt2Model::wrap_lora`] rejects a
/// second wrap of an already-LoRA-wrapped block (double-wrap surfaces
/// as `TrainError::Candle`). Callers who want to run multiple LoRA
/// trainings should build a fresh [`Gpt2Handle`] per run.
fn lora_impl(
    lua: &Lua,
    handle: &LuaAnyUserData,
    dataset: &LuaAnyUserData,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    lease: Arc<TrainingLease>,
) -> LuaResult<LuaTable> {
    let train_cfg = extract_full_ft_opts(opts)?;
    let lora_cfg = extract_lora_cfg(opts)?;
    let (ckpt_dir, card_id) = resolve_ckpt_dest(opts, nn_dir, "lora")?;

    let gpt2 = handle.borrow::<Gpt2Handle>()?;
    let model_arc = gpt2.model();
    // Base bundle_ref recorded on the Card metadata later. For a
    // from-scratch handle the base is transient (no persisted bundle);
    // callers who want a durable base_bundle_ref should `alc.nn.card
    // .save` the base first, then run lora on top, and pass
    // `opts.base_bundle_ref = "nn/<base-card_id>"`. Wrong-type input
    // (e.g. `base_bundle_ref = 42`) surfaces as a Lua type-mismatch
    // error rather than silently falling back to the default.
    let base_bundle_ref = match opts {
        Some(t) => t
            .get::<Option<String>>("base_bundle_ref")?
            .unwrap_or_else(|| format!("nn/{}", gpt2.variant)),
        None => format!("nn/{}", gpt2.variant),
    };
    drop(gpt2);

    let ds_guard = dataset.borrow_mut::<DatasetHandle>()?;
    let mut ds_lock = ds_guard
        .inner
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.lora: dataset lock: {e}")))?;

    let loss_fn = CrossEntropyLoss::new();
    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.lora: model lock: {e}")))?;

    let result = run_lora_ft(
        &mut model,
        ds_lock.as_mut(),
        &lora_cfg,
        &train_cfg,
        &loss_fn,
        &ckpt_dir,
        &card_id,
        lease,
    );
    drop(model);
    drop(ds_lock);
    drop(ds_guard);

    let ckpt = result.map_err(train_err_to_lua)?;

    // Attach the LoRA branch descriptor. The Card foundation reads
    // this back through `meta.candle.lora` in `build_create_payload`.
    //
    // ST-d additions: `target_modules` / `dropout` / `delta_path` are
    // required by `alc.nn.card.load_gpt2` to reconstruct the same
    // `LoraConfig` on reload and locate the delta safetensors. The
    // delta lives at `<ckpt_dir>/nn/<ckpt.bundle_ref>` — `run_lora_ft`
    // writes `nn/lora-<card_id>.safetensors` under the caller's
    // `ckpt_dir`, and `ckpt.bundle_ref` is the trailing filename.
    let lora_tbl = lua.create_table()?;
    lora_tbl.set("rank", lora_cfg.rank as u32)?;
    lora_tbl.set("alpha", lora_cfg.alpha as u32)?;
    lora_tbl.set("base_bundle_ref", base_bundle_ref)?;
    let target_modules_tbl = lua.create_table()?;
    for (i, m) in lora_cfg.target_modules.iter().enumerate() {
        target_modules_tbl.set(i + 1, m.clone())?;
    }
    lora_tbl.set("target_modules", target_modules_tbl)?;
    lora_tbl.set("dropout", lora_cfg.dropout)?;
    let delta_path = ckpt_dir.join("nn").join(&ckpt.bundle_ref);
    lora_tbl.set("delta_path", delta_path.to_string_lossy().to_string())?;

    checkpoint_to_lua(lua, &ckpt, &card_id, Some(lora_tbl))
}

/// `alc.nn.trainer.distill(handle, dataset, opts?) -> Checkpoint`.
///
/// Currently supports only `loss_kind = "ce"` (hard-label CE, the
/// only variant [`DistillLossKind`] exposes). Unknown `loss_kind`
/// values error out rather than silently fall back to CE.
fn distill_impl(
    lua: &Lua,
    handle: &LuaAnyUserData,
    dataset: &LuaAnyUserData,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    lease: Arc<TrainingLease>,
) -> LuaResult<LuaTable> {
    let hyperparams = extract_full_ft_opts(opts)?;
    let loss_kind = extract_distill_loss_kind(opts)?;
    let spec = DistillSpec {
        hyperparams,
        loss_kind,
    };
    let (ckpt_dir, ckpt_prefix) = resolve_ckpt_dest(opts, nn_dir, "distill")?;
    let card_id = ckpt_prefix.clone();

    let gpt2 = handle.borrow::<Gpt2Handle>()?;
    let model_arc = gpt2.model();
    let vm_arc = gpt2.varmap().ok_or_else(|| {
        LuaError::external(
            "alc.nn.trainer.distill: handle was built with pretrained=true; \
             distillation requires a from-scratch student handle (pretrained=false)"
                .to_string(),
        )
    })?;
    drop(gpt2);

    let ds_guard = dataset.borrow_mut::<DatasetHandle>()?;
    let mut ds_lock = ds_guard
        .inner
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.distill: dataset lock: {e}")))?;

    let model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.distill: model lock: {e}")))?;

    let result = run_distill(
        &model,
        &vm_arc,
        ds_lock.as_mut(),
        &spec,
        &ckpt_dir,
        &ckpt_prefix,
        lease,
    );
    drop(model);
    drop(ds_lock);
    drop(ds_guard);

    let ckpt = result.map_err(train_err_to_lua)?;
    checkpoint_to_lua(lua, &ckpt, &card_id, None)
}

/// Extract [`FullFtConfig`] from an opts table, applying the crate's
/// defaults for any missing key. Rejects zero-sized values at the
/// boundary (matches the training-loop's own early exit shape).
fn extract_full_ft_opts(opts: Option<&LuaTable>) -> LuaResult<FullFtConfig> {
    let mut cfg = FullFtConfig::default();
    let Some(t) = opts else {
        return Ok(cfg);
    };
    if let Some(v) = t.get::<Option<f64>>("lr")? {
        cfg.lr = v;
    }
    if let Some(v) = t.get::<Option<usize>>("batch_size")? {
        cfg.batch_size = v;
    }
    if let Some(v) = t.get::<Option<usize>>("grad_accum")? {
        cfg.grad_accum = v;
    }
    if let Some(v) = t.get::<Option<usize>>("steps")? {
        cfg.steps = v;
    }
    if let Some(v) = t.get::<Option<usize>>("warmup")? {
        cfg.warmup = v;
    }
    if let Some(v) = t.get::<Option<String>>("schedule")? {
        cfg.schedule = parse_schedule(&v)?;
    }
    if let Some(v) = t.get::<Option<f64>>("weight_decay")? {
        cfg.weight_decay = v;
    }
    if let Some(v) = t.get::<Option<usize>>("ckpt_every")? {
        cfg.ckpt_every = v;
    }
    if let Some(v) = t.get::<Option<usize>>("ckpt_keep")? {
        cfg.ckpt_keep = v;
    }
    if cfg.batch_size == 0 {
        return Err(LuaError::external(
            "alc.nn.trainer: batch_size must be >= 1".to_string(),
        ));
    }
    Ok(cfg)
}

fn parse_schedule(s: &str) -> LuaResult<ScheduleKind> {
    match s {
        "cosine" | "cosine_with_warmup" => Ok(ScheduleKind::CosineWithWarmup),
        "constant" => Ok(ScheduleKind::Constant),
        other => Err(LuaError::external(format!(
            "alc.nn.trainer: unknown schedule '{other}' \
             (expected 'cosine' or 'constant')"
        ))),
    }
}

/// Extract [`LoraConfig`] from an opts table. `rank` and `alpha` are
/// required — omitting them is almost certainly a user error and
/// silently defaulting would hide a wrong training run.
fn extract_lora_cfg(opts: Option<&LuaTable>) -> LuaResult<LoraConfig> {
    let t = opts.ok_or_else(|| {
        LuaError::external(
            "alc.nn.trainer.lora: opts table is required (need at least rank and alpha)"
                .to_string(),
        )
    })?;
    let rank = t.get::<Option<usize>>("rank")?.ok_or_else(|| {
        LuaError::external("alc.nn.trainer.lora: opts.rank is required".to_string())
    })?;
    let alpha_raw = t.get::<Option<f32>>("alpha")?.ok_or_else(|| {
        LuaError::external("alc.nn.trainer.lora: opts.alpha is required".to_string())
    })?;
    if rank == 0 {
        return Err(LuaError::external(
            "alc.nn.trainer.lora: rank must be >= 1".to_string(),
        ));
    }
    if !alpha_raw.is_finite() || alpha_raw <= 0.0 {
        return Err(LuaError::external(
            "alc.nn.trainer.lora: alpha must be a positive finite number".to_string(),
        ));
    }
    let dropout = t.get::<Option<f32>>("dropout")?.unwrap_or(0.0);
    let target_modules = match t.get::<Option<LuaTable>>("target_modules")? {
        Some(list) => {
            let mut names = Vec::new();
            for pair in list.pairs::<LuaValue, String>() {
                let (_k, name) = pair?;
                names.push(name);
            }
            if names.is_empty() {
                return Err(LuaError::external(
                    "alc.nn.trainer.lora: opts.target_modules must not be empty".to_string(),
                ));
            }
            names
        }
        None => LoraConfig::default_targets(),
    };
    Ok(LoraConfig {
        rank,
        alpha: alpha_raw,
        target_modules,
        dropout,
    })
}

fn extract_distill_loss_kind(opts: Option<&LuaTable>) -> LuaResult<DistillLossKind> {
    // Wrong-type input (`loss_kind = 42`) must surface as a Lua
    // type-mismatch error rather than silently falling back to `"ce"`
    // — silent fallback would hide a misconfigured caller.
    let raw = match opts {
        Some(t) => t
            .get::<Option<String>>("loss_kind")?
            .unwrap_or_else(|| "ce".to_string()),
        None => "ce".to_string(),
    };
    match raw.as_str() {
        "ce" => Ok(DistillLossKind::Ce),
        other => Err(LuaError::external(format!(
            "alc.nn.trainer.distill: unknown loss_kind '{other}' (expected 'ce')"
        ))),
    }
}

/// Decide where checkpoints get written for this training run.
///
/// - `opts.ckpt_dir` overrides the default (`<nn_dir>/ckpt`) when the
///   caller wants a scenario-specific location.
/// - `opts.card_id` overrides the default `<path>_<epoch_us>` prefix
///   when the caller wants the ckpt filename to match a Card id they
///   already own (`run_lora_ft` in particular expects this to line up
///   with the `nn/lora-<card_id>.safetensors` bundle name).
fn resolve_ckpt_dest(
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    stage: &str,
) -> LuaResult<(PathBuf, String)> {
    // Both `ckpt_dir` and `card_id` are read strictly: wrong-type input
    // surfaces as a Lua error rather than silently falling back to the
    // default (which would write the checkpoint to an unexpected
    // location without diagnostic).
    let ckpt_dir = match opts {
        Some(t) => t
            .get::<Option<String>>("ckpt_dir")?
            .map(PathBuf::from)
            .unwrap_or_else(|| nn_dir.join("ckpt")),
        None => nn_dir.join("ckpt"),
    };
    std::fs::create_dir_all(&ckpt_dir).map_err(|e| {
        LuaError::external(format!(
            "alc.nn.trainer.{stage}: mkdir {:?}: {e}",
            ckpt_dir.display()
        ))
    })?;
    let ckpt_prefix = match opts {
        Some(t) => match t.get::<Option<String>>("card_id")? {
            Some(id) => sanitize_name(&id),
            None => generate_card_id(stage),
        },
        None => generate_card_id(stage),
    };
    Ok((ckpt_dir, ckpt_prefix))
}

/// Convert a [`TrainError`] into an `mlua::Error` with the training
/// stage prefixed onto the message.
fn train_err_to_lua(e: TrainError) -> LuaError {
    LuaError::external(format!("alc.nn.trainer: {e}"))
}

/// Convert a [`algocline_nn::train::Checkpoint`] into a Lua table.
///
/// Optional `lora_branch` sub-table is attached under `ckpt.lora` for
/// the LoRA binding; `full_ft` / `distill` pass `None`.
///
/// **On `ckpt.card_id`**: this is the *checkpoint filename prefix*
/// (matches the safetensors bundle `<prefix>.safetensors`), NOT the
/// Card store's card_id. `alc.nn.card.save` currently generates its
/// own Card id internally (`generate_card_id(name)`) rather than
/// accepting a caller-provided value, so `alc.nn.card.load(ckpt.card_id)`
/// will not resolve. Callers who need to correlate the trainer output
/// with a saved Card should use the return value of `alc.nn.card.save`
/// directly. Renaming this field to `ckpt_prefix` is deferred pending
/// a Phase 2 follow-up that threads the prefix through to `save_impl`.
fn checkpoint_to_lua(
    lua: &Lua,
    ckpt: &algocline_nn::train::Checkpoint,
    card_id: &str,
    lora_branch: Option<LuaTable>,
) -> LuaResult<LuaTable> {
    let out = lua.create_table()?;
    out.set("bundle_ref", ckpt.bundle_ref.clone())?;
    out.set("card_id", card_id.to_string())?;
    out.set("step", ckpt.step)?;
    out.set("train_loss", ckpt.train_loss)?;
    match ckpt.val_loss {
        Some(v) => out.set("val_loss", v)?,
        None => out.set("val_loss", LuaValue::Nil)?,
    }
    let metrics = lua.create_table()?;
    for (k, v) in &ckpt.metrics {
        metrics.set(k.as_str(), *v)?;
    }
    out.set("metrics", metrics)?;
    if let Some(lora) = lora_branch {
        out.set("lora", lora)?;
    }
    Ok(out)
}

#[cfg(test)]
mod trainer_tests {
    use super::*;
    use mlua::Lua;

    fn opts_from(lua: &Lua, pairs: &[(&str, LuaValue)]) -> LuaTable {
        let t = lua.create_table().expect("create opts table");
        for (k, v) in pairs {
            t.set(*k, v.clone()).expect("set opt field");
        }
        t
    }

    #[test]
    fn full_ft_opts_defaults_when_empty() {
        let lua = Lua::new();
        let cfg = extract_full_ft_opts(None).expect("None -> defaults");
        let default = FullFtConfig::default();
        assert_eq!(cfg.lr, default.lr);
        assert_eq!(cfg.batch_size, default.batch_size);
        assert_eq!(cfg.steps, default.steps);

        let empty = lua.create_table().unwrap();
        let cfg2 = extract_full_ft_opts(Some(&empty)).expect("empty table -> defaults");
        assert_eq!(cfg2.lr, default.lr);
        assert_eq!(cfg2.batch_size, default.batch_size);
    }

    #[test]
    fn full_ft_opts_partial_merges_with_defaults() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[
                ("lr", LuaValue::Number(1e-3)),
                ("steps", LuaValue::Integer(42)),
            ],
        );
        let cfg = extract_full_ft_opts(Some(&opts)).expect("partial merge");
        assert!((cfg.lr - 1e-3).abs() < 1e-12, "lr override");
        assert_eq!(cfg.steps, 42, "steps override");
        // Unset fields keep the crate default.
        let d = FullFtConfig::default();
        assert_eq!(cfg.batch_size, d.batch_size);
        assert_eq!(cfg.warmup, d.warmup);
    }

    #[test]
    fn full_ft_opts_reject_zero_batch_size() {
        let lua = Lua::new();
        let opts = opts_from(&lua, &[("batch_size", LuaValue::Integer(0))]);
        let err = extract_full_ft_opts(Some(&opts)).expect_err("zero batch_size");
        assert!(
            err.to_string().contains("batch_size must be >= 1"),
            "message: {err}"
        );
    }

    #[test]
    fn schedule_parser_accepts_known_and_rejects_unknown() {
        assert!(matches!(
            parse_schedule("cosine").unwrap(),
            ScheduleKind::CosineWithWarmup
        ));
        assert!(matches!(
            parse_schedule("cosine_with_warmup").unwrap(),
            ScheduleKind::CosineWithWarmup
        ));
        assert!(matches!(
            parse_schedule("constant").unwrap(),
            ScheduleKind::Constant
        ));
        let err = parse_schedule("linear").expect_err("unknown");
        assert!(err.to_string().contains("linear"), "message: {err}");
    }

    #[test]
    fn lora_cfg_requires_opts_table() {
        let err = extract_lora_cfg(None).expect_err("None opts");
        assert!(err.to_string().contains("opts table is required"));
    }

    #[test]
    fn lora_cfg_requires_rank_and_alpha() {
        let lua = Lua::new();
        let no_rank = opts_from(&lua, &[("alpha", LuaValue::Number(16.0))]);
        let err = extract_lora_cfg(Some(&no_rank)).expect_err("missing rank");
        assert!(err.to_string().contains("opts.rank is required"));

        let no_alpha = opts_from(&lua, &[("rank", LuaValue::Integer(8))]);
        let err = extract_lora_cfg(Some(&no_alpha)).expect_err("missing alpha");
        assert!(err.to_string().contains("opts.alpha is required"));
    }

    #[test]
    fn lora_cfg_rejects_zero_rank_and_nonpositive_alpha() {
        let lua = Lua::new();
        let zero_rank = opts_from(
            &lua,
            &[
                ("rank", LuaValue::Integer(0)),
                ("alpha", LuaValue::Number(1.0)),
            ],
        );
        let err = extract_lora_cfg(Some(&zero_rank)).expect_err("zero rank");
        assert!(err.to_string().contains("rank must be >= 1"));

        let neg_alpha = opts_from(
            &lua,
            &[
                ("rank", LuaValue::Integer(4)),
                ("alpha", LuaValue::Number(-1.0)),
            ],
        );
        let err = extract_lora_cfg(Some(&neg_alpha)).expect_err("negative alpha");
        assert!(err.to_string().contains("positive finite number"));
    }

    #[test]
    fn lora_cfg_defaults_target_modules_when_omitted() {
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[
                ("rank", LuaValue::Integer(8)),
                ("alpha", LuaValue::Number(16.0)),
            ],
        );
        let cfg = extract_lora_cfg(Some(&opts)).expect("defaults");
        let defaults = LoraConfig::default_targets();
        assert_eq!(cfg.target_modules, defaults);
        assert_eq!(cfg.rank, 8);
        assert!((cfg.alpha - 16.0).abs() < 1e-6);
        assert_eq!(cfg.dropout, 0.0);
    }

    #[test]
    fn lora_cfg_rejects_empty_target_modules_list() {
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("rank", LuaValue::Integer(4)).unwrap();
        opts.set("alpha", LuaValue::Number(8.0)).unwrap();
        let empty = lua.create_table().unwrap();
        opts.set("target_modules", empty).unwrap();
        let err = extract_lora_cfg(Some(&opts)).expect_err("empty targets");
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn lora_cfg_reads_custom_targets_dropout() {
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("rank", LuaValue::Integer(16)).unwrap();
        opts.set("alpha", LuaValue::Number(32.0)).unwrap();
        opts.set("dropout", LuaValue::Number(0.05)).unwrap();
        let targets = lua.create_table().unwrap();
        targets.set(1, "q_proj").unwrap();
        targets.set(2, "v_proj").unwrap();
        opts.set("target_modules", targets).unwrap();
        let cfg = extract_lora_cfg(Some(&opts)).expect("custom");
        assert_eq!(
            cfg.target_modules,
            vec!["q_proj".to_string(), "v_proj".into()]
        );
        assert!((cfg.dropout - 0.05).abs() < 1e-6);
    }

    #[test]
    fn distill_loss_kind_rejects_wrong_type_input() {
        let lua = Lua::new();
        // `loss_kind = true` (boolean) must surface as a Lua
        // type-mismatch error, not silently fall back to "ce".
        // (Integers coerce to strings in mlua, so booleans are used to
        // provoke the type check.)
        let opts = opts_from(&lua, &[("loss_kind", LuaValue::Boolean(true))]);
        let err = extract_distill_loss_kind(Some(&opts)).expect_err("wrong-type loss_kind");
        let msg = err.to_string();
        assert!(!msg.is_empty(), "type-mismatch error should have a message");
    }

    #[test]
    fn resolve_ckpt_dest_rejects_wrong_type_ckpt_dir() {
        let lua = Lua::new();
        let opts = opts_from(&lua, &[("ckpt_dir", LuaValue::Boolean(false))]);
        let tmp = std::env::temp_dir();
        let err = resolve_ckpt_dest(Some(&opts), &tmp, "full_ft").expect_err("wrong-type ckpt_dir");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn distill_loss_kind_defaults_to_ce_and_rejects_unknown() {
        assert!(matches!(
            extract_distill_loss_kind(None).unwrap(),
            DistillLossKind::Ce
        ));
        let lua = Lua::new();
        let opts = opts_from(
            &lua,
            &[(
                "loss_kind",
                LuaValue::String(lua.create_string("ce").unwrap()),
            )],
        );
        assert!(matches!(
            extract_distill_loss_kind(Some(&opts)).unwrap(),
            DistillLossKind::Ce
        ));
        let bad = opts_from(
            &lua,
            &[(
                "loss_kind",
                LuaValue::String(lua.create_string("kl_soft").unwrap()),
            )],
        );
        let err = extract_distill_loss_kind(Some(&bad)).expect_err("unknown loss");
        assert!(err.to_string().contains("kl_soft"));
    }

    #[test]
    fn build_create_payload_populates_lora_branch_when_meta_provides_it() {
        let user_meta = json!({
            "training_path": "lora",
            "architecture": "gpt2-medium",
            "candle": {
                "device": "cuda:0",
                "dtype": "bf16",
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "base_bundle_ref": "nn/base-gpt2-medium"
                }
            }
        });
        let payload =
            build_create_payload("card-abc", "my-model", &user_meta).expect("payload with lora");
        let lora = payload
            .pointer("/metadata/nn/candle/lora")
            .expect("lora sub-object");
        assert_eq!(lora.get("rank"), Some(&json!(8)));
        assert_eq!(lora.get("alpha"), Some(&json!(16)));
        assert_eq!(
            lora.get("base_bundle_ref"),
            Some(&json!("nn/base-gpt2-medium"))
        );
    }

    #[test]
    fn build_create_payload_omits_lora_when_meta_absent_or_null() {
        let no_candle = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
        });
        let p = build_create_payload("c1", "m", &no_candle).unwrap();
        let candle = p.pointer("/metadata/nn/candle").expect("candle present");
        assert!(
            candle.get("lora").is_none() || candle.get("lora") == Some(&Json::Null),
            "lora must be absent: {candle}"
        );

        let candle_only = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
            "candle": { "device": "cpu" }
        });
        let p2 = build_create_payload("c2", "m", &candle_only).unwrap();
        let candle2 = p2.pointer("/metadata/nn/candle").expect("candle present");
        assert!(candle2.get("lora").is_none() || candle2.get("lora") == Some(&Json::Null));

        let explicit_null = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
            "candle": { "lora": Json::Null }
        });
        let p3 = build_create_payload("c3", "m", &explicit_null).unwrap();
        let candle3 = p3.pointer("/metadata/nn/candle").expect("candle present");
        assert!(candle3.get("lora").is_none() || candle3.get("lora") == Some(&Json::Null));
    }

    #[test]
    fn build_create_payload_preserves_lora_target_modules_dropout_delta_path() {
        // ST-d extension: the extra fields `target_modules` /
        // `dropout` / `delta_path` populated by
        // `alc.nn.trainer.lora` must round-trip verbatim through the
        // Card payload so `alc.nn.card.load_gpt2` can rebuild the
        // same LoraConfig and locate the delta.
        let user_meta = json!({
            "training_path": "lora",
            "architecture": "gpt2-medium",
            "candle": {
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "base_bundle_ref": "nn/base-gpt2-medium",
                    "target_modules": ["q_proj", "v_proj"],
                    // 0.0625 is exactly representable in f32, so the
                    // json → serde f32 → serde json roundtrip does
                    // not perturb the value.
                    "dropout": 0.0625,
                    "delta_path": "/tmp/ckpt/nn/lora-run.safetensors"
                }
            }
        });
        let payload = build_create_payload("card-lora", "m", &user_meta)
            .expect("payload with full lora branch");
        let lora = payload
            .pointer("/metadata/nn/candle/lora")
            .expect("lora sub-object");
        assert_eq!(
            lora.get("target_modules"),
            Some(&json!(["q_proj", "v_proj"]))
        );
        assert_eq!(lora.get("dropout"), Some(&json!(0.0625)));
        assert_eq!(
            lora.get("delta_path"),
            Some(&json!("/tmp/ckpt/nn/lora-run.safetensors"))
        );
    }

    #[test]
    fn build_create_payload_defaults_lora_target_modules_when_meta_omits_them() {
        // Backwards compat: a caller providing only the pre-ST-d
        // trio (rank / alpha / base_bundle_ref) still produces a
        // valid Card — the payload fills `target_modules` with the
        // canonical six and `dropout` with 0.0.
        let user_meta = json!({
            "training_path": "lora",
            "architecture": "gpt2-medium",
            "candle": {
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "base_bundle_ref": "nn/base-gpt2-medium"
                }
            }
        });
        let payload =
            build_create_payload("card-legacy", "m", &user_meta).expect("payload with legacy lora");
        let lora = payload
            .pointer("/metadata/nn/candle/lora")
            .expect("lora sub-object");
        // NnLoraBranch::target_modules defaults via serde to the
        // canonical six on deserialize; re-serialization emits the
        // full list.
        let targets = lora.get("target_modules").expect("target_modules");
        let arr = targets.as_array().expect("array");
        assert_eq!(arr.len(), 6);
        assert_eq!(lora.get("dropout"), Some(&json!(0.0)));
        // delta_path is Option<String> with `skip_serializing_if =
        // "Option::is_none"` — a card written without it stays
        // without it (the load_gpt2 path errors on this shape).
        assert!(
            lora.get("delta_path").is_none() || lora.get("delta_path") == Some(&Json::Null),
            "delta_path must be omitted when absent: {lora}"
        );
    }

    #[test]
    fn load_gpt2_impl_errors_when_card_missing() {
        // Store lookup miss surfaces as a clear error — no partial
        // handle returned, no silent fallback. Verifies the error
        // path before base_handle is touched (`base_handle` is unused
        // in this branch, so we pass a fresh empty VM's app-level
        // Value we can create; but easier: bail before that point via
        // an obviously-nonexistent card_id).
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let store = FileCardStore::new(tmp.path().to_path_buf());
        // Use a Lua VM only to synthesise a placeholder UserData so
        // the function signature is satisfied — the error path is
        // hit before the userdata is borrowed.
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).expect("placeholder userdata");
        let msg = match load_gpt2_impl(&store, "does-not-exist", &placeholder) {
            Ok(_) => panic!("card missing must error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("does-not-exist"),
            "error must name the missing card: {msg}"
        );
    }

    /// Test helper: write a Card with a caller-controlled
    /// `metadata.nn` shape into a fresh [`FileCardStore`] and return
    /// the generated `card_id`. Uses `create_with_store`'s auto id
    /// generation so tests don't have to know the exact format.
    fn write_test_card(store: &FileCardStore, nn_meta: serde_json::Value) -> String {
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "metadata": {
                "kind": "nn_model",
                "nn": nn_meta,
            }
        });
        let (card_id, _path) = store.create(payload).expect("create test card");
        card_id
    }

    #[test]
    fn load_gpt2_impl_errors_when_metadata_nn_candle_missing() {
        // (M-2 fix) A Card with `metadata.nn` but no `candle`
        // sub-block cannot be a LoRA reload — surface the specific
        // gap rather than a generic downstream failure. Errors
        // before the base handle is inspected.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let card_id = write_test_card(
            &store,
            json!({
                "name": "no-candle",
                "backend": "endpoint",
                "architecture": "gpt2-medium",
                "training_path": "lora",
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).expect("placeholder userdata");
        let msg = match load_gpt2_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("missing candle must error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("missing metadata.nn.candle"),
            "error must name the missing candle branch: {msg}"
        );
    }

    #[test]
    fn load_gpt2_impl_errors_when_lora_branch_lacks_delta_path() {
        // (M-2 fix) A pre-ST-d LoRA card (or one saved via a legacy
        // path that omitted `delta_path`) cannot be reloaded — the
        // load path errors loudly rather than picking a default
        // location that would silently pick the wrong bundle.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let card_id = write_test_card(
            &store,
            json!({
                "name": "legacy-lora",
                "backend": "candle",
                "architecture": "gpt2-medium",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 8,
                        "alpha": 16,
                        "base_bundle_ref": "nn/base-gpt2-medium",
                    }
                }
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).expect("placeholder userdata");
        let msg = match load_gpt2_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("missing delta_path must error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("delta_path"),
            "error must name the missing delta_path field: {msg}"
        );
    }

    #[test]
    fn load_gpt2_impl_errors_when_delta_path_file_missing() {
        // (M-2 fix) When `delta_path` is recorded but the
        // safetensors file has been cleaned (ckpt_dir wiped between
        // runs), we surface a filesystem-specific error rather than
        // a downstream `VarMap::load` message that could be
        // mistaken for a corrupted delta.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let store = FileCardStore::new(tmp.path().to_path_buf());
        let card_id = write_test_card(
            &store,
            json!({
                "name": "missing-delta-file",
                "backend": "candle",
                "architecture": "gpt2-medium",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 8,
                        "alpha": 16,
                        "base_bundle_ref": "nn/base-gpt2-medium",
                        "target_modules": ["q_proj", "v_proj"],
                        "dropout": 0.0,
                        "delta_path": "/nonexistent/path/to/lora-run.safetensors"
                    }
                }
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).expect("placeholder userdata");
        let msg = match load_gpt2_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("missing delta file must error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("delta safetensors missing")
                || msg.contains("/nonexistent/path/to/lora-run.safetensors"),
            "error must name the missing delta file: {msg}"
        );
    }

    #[test]
    fn build_create_payload_reports_invalid_lora_shape() {
        // Missing required NnLoraBranch fields (`rank` / `alpha` /
        // `base_bundle_ref`) surfaces as a clear error rather than a
        // silently half-populated Card.
        let bad = json!({
            "training_path": "lora",
            "architecture": "gpt2-medium",
            "candle": { "lora": { "rank": 8 } }
        });
        let err = build_create_payload("cx", "m", &bad).expect_err("invalid lora");
        let msg = err.to_string();
        assert!(msg.contains("invalid meta.candle.lora"), "message: {msg}");
    }
}
