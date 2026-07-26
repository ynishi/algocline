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

use algocline_nn::arch::adapter::{LlamaAdapter, LlamaAdapterConfig};
use algocline_nn::arch::{
    Activation, Gpt2Config, Gpt2Custom, Gpt2Model, LoraConfig, MoeConfig, NormKind, NormPlacement,
    PosKind, ResidualKind, TinyLlamaConfig, TinyLlamaModel,
};
use algocline_nn::card::{
    validate_architecture, NnCandleBranch, NnCardMeta, NnLineage, NnLoraBranch,
};
use algocline_nn::merged::{export_merged, MergeError, MergedProvenance};
use algocline_nn::tokenizer::HfTokenizer;
use algocline_nn::train::{
    run_distill, run_full_ft, run_lora_ft, Batch, CrossEntropyLoss, Dataset, DatasetOpts,
    DistillLossKind, DistillSpec, FullFtConfig, JsonlDataset, ParquetDataset, ScheduleKind,
    TeacherCardDataset, TokenizedDataset, TrainError, TrainingLease,
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

    // Legacy raw-vars loader (returns a Lua table of tensor Vars).
    // Layer 4b S4 renames the entry from `load` to `load_vars` so
    // the neutral handle-returning `load` entry can occupy the
    // `load` slot. The old `load` name stays as a deprecated
    // alias for one release cycle (Layer 4b §8 open question).
    let load_vars_store = Arc::clone(&card_store);
    let load_vars = lua.create_function(move |lua, card_id: String| -> LuaResult<LuaTable> {
        load_impl(lua, load_vars_store.as_ref(), &card_id)
    })?;
    card_ns.set("load_vars", load_vars.clone())?;
    // Deprecated: `alc.nn.card.load` still returns raw vars for
    // backward compat. NEW callers should use `load_vars` for the
    // raw-vars path, and the new handle-returning `alc.nn.card.load`
    // will be added in a future minor release once the deprecation
    // window closes. In the interim, the neutral handle-returning
    // path is registered as `load_handle` (Layer 4b S4).
    card_ns.set("load", load_vars)?;

    // New arch-neutral handle-returning entry (Layer 4b §Q3-A).
    // Named `load_handle` during the deprecation window; the
    // shorter `load` slot flips over to this once the raw-vars
    // deprecation cycle completes.
    let load_handle_store = Arc::clone(&card_store);
    let load_handle_nn_dir = nn_dir.clone();
    let load_handle = lua.create_function(move |_lua, card_id: String| -> LuaResult<NnHandle> {
        load_handle_impl(load_handle_store.as_ref(), &card_id, &load_handle_nn_dir)
    })?;
    card_ns.set("load_handle", load_handle)?;

    let load_gpt2_store = Arc::clone(&card_store);
    let load_gpt2 = lua.create_function(
        move |_lua, (card_id, base_handle): (String, LuaAnyUserData)| -> LuaResult<Gpt2Handle> {
            load_gpt2_impl(load_gpt2_store.as_ref(), &card_id, &base_handle)
        },
    )?;
    card_ns.set("load_gpt2", load_gpt2)?;

    // Layer 4b §Q3-A arch-neutral LoRA card loader. Accepts either
    // NnHandle (from `alc.nn.preset("gpt2"/"tinyllama", …)`) or a
    // typed Handle (from `alc.nn.preset.gpt2` / `.tinyllama`) as
    // the base. Refuses self-contained cards (full_ft / merged /
    // distillation) with a directional error pointing at
    // `alc.nn.card.load_handle`.
    let load_wrap_store = Arc::clone(&card_store);
    let load_wrap = lua.create_function(
        move |_lua, (card_id, base_handle): (String, LuaAnyUserData)| -> LuaResult<NnHandle> {
            load_wrap_impl(load_wrap_store.as_ref(), &card_id, &base_handle)
        },
    )?;
    card_ns.set("load_wrap", load_wrap)?;

    let register_store = Arc::clone(&card_store);
    let register = lua.create_function(
        move |lua, (card_id, model_name): (String, String)| -> LuaResult<()> {
            register_impl(lua, register_store.as_ref(), &card_id, &model_name)
        },
    )?;
    card_ns.set("register", register)?;

    // Layer 5a: arch-neutral LoRA merge entry. Consumes a
    // LoRA-wrapped NnHandle (or typed Gpt2Handle / TinyLlamaHandle
    // for backward compat) + opts { name, lora_card }, writes a
    // merged safetensors bundle under nn_dir/nn/<id>.safetensors,
    // records a Card with training_path="merged", returns the new
    // card_id.
    let merge_lora_store = Arc::clone(&card_store);
    let merge_lora_nn_dir = nn_dir.clone();
    let merge_lora = lua.create_function(
        move |_lua, (base_handle, opts): (LuaAnyUserData, LuaTable)| -> LuaResult<String> {
            merge_lora_impl(
                merge_lora_store.as_ref(),
                &merge_lora_nn_dir,
                &base_handle,
                opts,
            )
        },
    )?;
    card_ns.set("merge_lora", merge_lora)?;

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

/// Arch-neutral self-contained card loader (Layer 4b §Q3-A `load`).
///
/// Reads the Card, extracts `architecture` + `training_path`,
/// resolves the arch's [`ArchOps::build_from_safetensors`], and
/// dispatches. Refuses LoRA cards with a directional error
/// pointing at `alc.nn.card.load_wrap` (Layer 4b §Q3-A invariant
/// #2). Refuses arches whose `build_from_safetensors` slot is
/// `None` (currently `llama-adapter`).
fn load_handle_impl(
    store: &FileCardStore,
    card_id: &str,
    nn_dir: &std::path::Path,
) -> LuaResult<NnHandle> {
    let card = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_handle: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}' not found"
            ))
        })?;

    let meta_json = card
        .get("metadata")
        .and_then(|m| m.get("nn"))
        .cloned()
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}' missing metadata.nn"
            ))
        })?;
    let meta: NnCardMeta = serde_json::from_value(meta_json).map_err(|e| {
        LuaError::external(format!(
            "alc.nn.card.load_handle: card '{card_id}' invalid metadata.nn: {e}"
        ))
    })?;

    // training_path 分岐: self-contained (full_ft / merged /
    // distillation) のみ受け付ける。 lora card は load_wrap 側に
    // 誘導 (Layer 4b §Q3-A invariant #2)。
    match meta.training_path.as_str() {
        "full_ft" | "merged" | "distillation" => {}
        "lora" => {
            return Err(LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}' has training_path=\"lora\"; \
                 LoRA cards need a base handle — call `alc.nn.card.load_wrap(card_id, base)` \
                 instead"
            )));
        }
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}' has unknown training_path \
                 {other:?} (expected one of full_ft / lora / merged / distillation)"
            )));
        }
    }

    // Enforce the bundle_ref = "nn/<card_id>" invariant that
    // load_impl also asserts (save_impl writes this shape).
    let bundle_ref = meta
        .candle
        .as_ref()
        .map(|c| c.bundle_ref.as_str())
        .ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}' missing metadata.nn.candle"
            ))
        })?;
    let expected = format!("nn/{card_id}");
    if bundle_ref != expected {
        return Err(LuaError::external(format!(
            "alc.nn.card.load_handle: bundle_ref '{bundle_ref}' does not match card_id \
             '{card_id}' (expected '{expected}')"
        )));
    }

    let ops = resolve_arch_ops(&meta.architecture).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_handle: card '{card_id}' architecture {:?} \
             has no bridge dispatch (expected one of {})",
            meta.architecture,
            registered_arch_names().join(" / ")
        ))
    })?;
    let build = ops.build_from_safetensors.ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_handle: card '{card_id}' architecture {:?} \
             does not support self-contained card load (adapter-style archs \
             need a different entry point — Layer 4b §8 carry)",
            meta.architecture
        ))
    })?;

    let path = nn_dir.join(format!("{card_id}.safetensors"));
    if !path.exists() {
        return Err(LuaError::external(format!(
            "alc.nn.card.load_handle: bundle missing at {path:?} for card '{card_id}'"
        )));
    }
    build(&meta, &path)
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
    // Delegate to the shared LoRA-wrap core so the arch-neutral
    // load_wrap path (Layer 4b §Q3-A) and this backward-compat
    // typed shortcut share a single body. The shortcut asserts
    // `base` is a typed `Gpt2Handle` up front; the neutral path
    // downcasts through `NnHandle::as_gpt2` for the same effect.
    //
    // Ordering: schema + delta-file precheck happens BEFORE
    // borrowing `base_handle` so callers hitting a schema gap
    // (missing candle / missing delta_path / missing delta file)
    // see the specific error rather than a generic "base handle
    // is not a Gpt2Handle" fallback — this ordering is asserted
    // by `trainer_tests::load_gpt2_impl_errors_*`.
    let card = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load_gpt2: card '{card_id}' not found"))
        })?;
    let meta = extract_nn_card_meta("alc.nn.card.load_gpt2", card_id, &card)?;
    precheck_lora_card_meta("alc.nn.card.load_gpt2", card_id, &meta)?;
    let base = base_handle
        .borrow::<Gpt2Handle>()
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: base handle: {e}")))?;
    wrap_gpt2_lora_from_meta("alc.nn.card.load_gpt2", card_id, &meta, &base)
}

/// Schema + delta-file precheck for a LoRA card, run before the
/// base handle is borrowed so callers hitting a schema gap see the
/// specific error rather than a generic "base handle wrong type"
/// downstream failure. Called by both `load_gpt2_impl` (typed
/// shortcut) and `load_wrap_impl` (arch-neutral).
fn precheck_lora_card_meta(ctx: &str, card_id: &str, meta: &NnCardMeta) -> LuaResult<()> {
    let candle = meta.candle.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' missing metadata.nn.candle"
        ))
    })?;
    let lora_branch = candle.lora.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' has no metadata.nn.candle.lora block \
             (use alc.nn.card.load / load_handle for weight-only reload of a non-LoRA card)"
        ))
    })?;
    let delta_path_str = lora_branch.delta_path.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' metadata.nn.candle.lora is missing delta_path \
             (pre-ST-d cards do not record it; re-save via alc.nn.trainer.lora + \
             alc.nn.card.save to populate)"
        ))
    })?;
    let delta_path = std::path::Path::new(delta_path_str);
    if !delta_path.exists() {
        return Err(LuaError::external(format!(
            "{ctx}: delta safetensors missing at {delta_path:?} \
             (expected the file produced by run_lora_ft; ckpt_dir may have been cleaned)"
        )));
    }
    Ok(())
}

/// Extract + deserialise `metadata.nn` from a Card JSON. Shared by
/// `load_gpt2_impl` / `load_handle_impl` / `load_wrap_impl` so
/// they surface identical errors on schema shape violations.
fn extract_nn_card_meta(ctx: &str, card_id: &str, card: &Json) -> LuaResult<NnCardMeta> {
    let meta_json = card
        .get("metadata")
        .and_then(|m| m.get("nn"))
        .cloned()
        .ok_or_else(|| {
            LuaError::external(format!("{ctx}: card '{card_id}' missing metadata.nn"))
        })?;
    serde_json::from_value(meta_json).map_err(|e| {
        LuaError::external(format!("{ctx}: card '{card_id}' invalid metadata.nn: {e}"))
    })
}

/// Core LoRA wrap for GPT-2. Consumes an already-parsed
/// `NnCardMeta` + a `Gpt2Handle` base; enforces the same schema +
/// arch-match invariants as the historical `load_gpt2_impl` body.
/// Shared between the arch-neutral `load_wrap` path
/// (via `wrap_gpt2_from_card`) and the backward-compat
/// `load_gpt2` shortcut (via `load_gpt2_impl`).
fn wrap_gpt2_lora_from_meta(
    ctx: &str,
    card_id: &str,
    meta: &NnCardMeta,
    base: &Gpt2Handle,
) -> LuaResult<Gpt2Handle> {
    let candle = meta.candle.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' missing metadata.nn.candle"
        ))
    })?;
    let lora_branch = candle.lora.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' has no metadata.nn.candle.lora block \
             (use alc.nn.card.load / load_handle for weight-only reload of a non-LoRA card)"
        ))
    })?;

    let delta_path_str = lora_branch.delta_path.clone().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' metadata.nn.candle.lora is missing delta_path \
             (pre-ST-d cards do not record it; re-save via alc.nn.trainer.lora + \
             alc.nn.card.save to populate)"
        ))
    })?;
    let delta_path = PathBuf::from(&delta_path_str);
    if !delta_path.exists() {
        return Err(LuaError::external(format!(
            "{ctx}: delta safetensors missing at {delta_path:?} \
             (expected the file produced by run_lora_ft; ckpt_dir may have been cleaned)"
        )));
    }

    let card_arch = &meta.architecture;
    let base_variant = &base.variant;
    let base_cfg_id = if base_variant.starts_with("gpt2-") {
        base_variant.clone()
    } else {
        format!("gpt2-{base_variant}")
    };
    if card_arch != &base_cfg_id && card_arch != base_variant {
        return Err(LuaError::external(format!(
            "{ctx}: architecture mismatch — card '{card_id}' was trained on \
             '{card_arch}' but base handle is '{base_variant}'. Rebuild the base with \
             `alc.nn.preset.gpt2('{card_arch}', ...)` (or the neutral \
             `alc.nn.preset('gpt2', '{card_arch}', ...)`) to match."
        )));
    }

    let mut lora_cfg = LoraConfig::with_targets(
        lora_branch.rank as usize,
        lora_branch.alpha as f32,
        lora_branch.target_modules.iter().cloned(),
    );
    lora_cfg.dropout = lora_branch.dropout;

    let model_arc = base.model();
    let variant = base.variant.clone();
    let layers = base.layers;
    let heads = base.heads;
    let dim = base.dim;
    let ctx_len = base.ctx;
    let vocab = base.vocab;
    let device = base.device.clone();
    let dtype = base.dtype.clone();
    let pretrained = base.pretrained;

    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("{ctx}: model lock: {e}")))?;
    let mut lora_vm = model
        .wrap_lora(&lora_cfg)
        .map_err(|e| LuaError::external(format!("{ctx}: wrap_lora: {e}")))?;
    drop(model);

    lora_vm
        .load(&delta_path)
        .map_err(|e| LuaError::external(format!("{ctx}: load delta {delta_path:?}: {e}")))?;

    Ok(Gpt2Handle {
        inner: model_arc,
        varmap: Some(Arc::new(lora_vm)),
        variant,
        layers,
        heads,
        dim,
        ctx: ctx_len,
        vocab,
        device,
        dtype,
        pretrained,
        has_lora: true,
    })
}

/// Core LoRA wrap for TinyLlama. Mirrors
/// [`wrap_gpt2_lora_from_meta`]; the arch-specific parts are the
/// `TinyLlamaHandle` field set + the arch prefix used in the
/// mismatch error message.
fn wrap_tinyllama_lora_from_meta(
    ctx: &str,
    card_id: &str,
    meta: &NnCardMeta,
    base: &TinyLlamaHandle,
) -> LuaResult<TinyLlamaHandle> {
    let candle = meta.candle.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' missing metadata.nn.candle"
        ))
    })?;
    let lora_branch = candle.lora.as_ref().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' has no metadata.nn.candle.lora block \
             (use alc.nn.card.load / load_handle for weight-only reload of a non-LoRA card)"
        ))
    })?;

    let delta_path_str = lora_branch.delta_path.clone().ok_or_else(|| {
        LuaError::external(format!(
            "{ctx}: card '{card_id}' metadata.nn.candle.lora is missing delta_path"
        ))
    })?;
    let delta_path = PathBuf::from(&delta_path_str);
    if !delta_path.exists() {
        return Err(LuaError::external(format!(
            "{ctx}: delta safetensors missing at {delta_path:?}"
        )));
    }

    let card_arch = &meta.architecture;
    let base_variant = &base.variant;
    let base_cfg_id = if base_variant.starts_with("tinyllama-") {
        base_variant.clone()
    } else {
        format!("tinyllama-{base_variant}")
    };
    if card_arch != &base_cfg_id && card_arch != base_variant {
        return Err(LuaError::external(format!(
            "{ctx}: architecture mismatch — card '{card_id}' was trained on \
             '{card_arch}' but base handle is '{base_variant}'. Rebuild the base with \
             `alc.nn.preset.tinyllama('{card_arch}', ...)` (or the neutral \
             `alc.nn.preset('tinyllama', '{card_arch}', ...)`) to match."
        )));
    }

    let mut lora_cfg = LoraConfig::with_targets(
        lora_branch.rank as usize,
        lora_branch.alpha as f32,
        lora_branch.target_modules.iter().cloned(),
    );
    lora_cfg.dropout = lora_branch.dropout;

    let model_arc = base.model();
    let variant = base.variant.clone();
    let layers = base.layers;
    let heads = base.heads;
    let kv_heads = base.kv_heads;
    let dim = base.dim;
    let ctx_len = base.ctx;
    let vocab = base.vocab;
    let device = base.device.clone();
    let dtype = base.dtype.clone();
    let pretrained = base.pretrained;

    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("{ctx}: model lock: {e}")))?;
    let mut lora_vm = model
        .wrap_lora(&lora_cfg)
        .map_err(|e| LuaError::external(format!("{ctx}: wrap_lora: {e}")))?;
    drop(model);

    lora_vm
        .load(&delta_path)
        .map_err(|e| LuaError::external(format!("{ctx}: load delta {delta_path:?}: {e}")))?;

    Ok(TinyLlamaHandle {
        inner: model_arc,
        varmap: Some(Arc::new(lora_vm)),
        variant,
        layers,
        heads,
        kv_heads,
        dim,
        ctx: ctx_len,
        vocab,
        device,
        dtype,
        pretrained,
        has_lora: true,
    })
}

/// Wrap a `Gpt2Handle` with LoRA using the given `LoraConfig` and
/// produce a fresh handle whose `varmap` slot now holds the newly
/// allocated LoRA delta [`VarMap`].
///
/// Consumed by the L5b-S1 `nn_wrap.rs` bridge (`alc.nn.wrap_lora`).
/// Added as a `pub(super)` helper (rather than widening
/// [`Gpt2Handle`]'s private fields) so external callers still cannot
/// build a `has_lora=true` handle without going through the actual
/// [`Gpt2Model::wrap_lora`] call — the invariant that
/// `has_lora=true` implies the underlying model has been
/// LoRA-wrapped in place stays enforced at construction.
///
/// The `inner` `Arc<Mutex<Gpt2Model>>` is shared with the caller's
/// base handle by design: `Gpt2Model::wrap_lora` mutates the model
/// in place (each block's linears are moved into `LoraLinear`), so
/// after this call the base handle's `inner` also points at the
/// (now-wrapped) model. Callers that need the pre-wrap base to
/// remain untouched must clone the model first — matching the
/// existing [`wrap_gpt2_lora_from_meta`] discipline.
pub(super) fn wrap_gpt2_lora_bridge(base: &Gpt2Handle, cfg: &LoraConfig) -> LuaResult<Gpt2Handle> {
    let model_arc = base.model();
    let variant = base.variant.clone();
    let layers = base.layers;
    let heads = base.heads;
    let dim = base.dim;
    let ctx_len = base.ctx;
    let vocab = base.vocab;
    let device = base.device.clone();
    let dtype = base.dtype.clone();
    let pretrained = base.pretrained;

    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.wrap_lora: model lock: {e}")))?;
    let lora_vm = model
        .wrap_lora(cfg)
        .map_err(|e| LuaError::external(format!("alc.nn.wrap_lora: candle: {e}")))?;
    drop(model);

    Ok(Gpt2Handle {
        inner: model_arc,
        varmap: Some(Arc::new(lora_vm)),
        variant,
        layers,
        heads,
        dim,
        ctx: ctx_len,
        vocab,
        device,
        dtype,
        pretrained,
        has_lora: true,
    })
}

/// Wrap a `TinyLlamaHandle` with LoRA using the given `LoraConfig`.
/// Mirrors [`wrap_gpt2_lora_bridge`]; consumed by the L5b-S1
/// `nn_wrap.rs` bridge (`alc.nn.wrap_lora`).
pub(super) fn wrap_tinyllama_lora_bridge(
    base: &TinyLlamaHandle,
    cfg: &LoraConfig,
) -> LuaResult<TinyLlamaHandle> {
    let model_arc = base.model();
    let variant = base.variant.clone();
    let layers = base.layers;
    let heads = base.heads;
    let kv_heads = base.kv_heads;
    let dim = base.dim;
    let ctx_len = base.ctx;
    let vocab = base.vocab;
    let device = base.device.clone();
    let dtype = base.dtype.clone();
    let pretrained = base.pretrained;

    let mut model = model_arc
        .lock()
        .map_err(|e| LuaError::external(format!("alc.nn.wrap_lora: model lock: {e}")))?;
    let lora_vm = model
        .wrap_lora(cfg)
        .map_err(|e| LuaError::external(format!("alc.nn.wrap_lora: candle: {e}")))?;
    drop(model);

    Ok(TinyLlamaHandle {
        inner: model_arc,
        varmap: Some(Arc::new(lora_vm)),
        variant,
        layers,
        heads,
        kv_heads,
        dim,
        ctx: ctx_len,
        vocab,
        device,
        dtype,
        pretrained,
        has_lora: true,
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
    validate_architecture(&architecture)
        .map_err(|e| LuaError::external(format!("alc.nn.card.save: {e}")))?;
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

/// Assemble the Card create payload directly from an already-built
/// [`NnCardMeta`], skipping the user-JSON parse + validate round
/// trip that [`build_create_payload`] performs.
///
/// Used by Layer 5a `alc.nn.card.merge_lora`: after
/// [`MergedProvenance::to_card_meta`] returns a fully-typed
/// [`NnCardMeta`] there is no user-facing JSON to re-validate.
/// The Card envelope shape (pkg / card_id / metadata.kind /
/// metadata.nn) is identical to [`build_create_payload`]'s output —
/// only the input side differs (typed struct vs. raw JSON).
///
/// Widened to `pub(super)` for L5b-S2 `nn_trainer.rs::run_lora_ft_impl`
/// which also builds a typed [`NnCardMeta`] (LoRA branch) and needs
/// the same envelope constructor — no user-JSON re-validation to
/// perform, so re-using [`build_create_payload`] would force an
/// unnecessary JSON round-trip.
pub(super) fn build_create_payload_from_meta(card_id: &str, meta: &NnCardMeta) -> LuaResult<Json> {
    // Defensive re-validation: even though the caller passes a
    // fully-typed struct, the architecture field still must match
    // the canonical family list (a mis-constructed MergedProvenance
    // could carry a stray value). Same guard as build_create_payload
    // §architecture check.
    validate_architecture(&meta.architecture)
        .map_err(|e| LuaError::external(format!("alc.nn.card.merge_lora: {e}")))?;

    let nn_meta_json = serde_json::to_value(meta)
        .map_err(|e| LuaError::external(format!("alc.nn.card.merge_lora: serialize meta: {e}")))?;

    Ok(json!({
        "pkg": { "name": NN_PKG },
        "card_id": card_id,
        "metadata": {
            "kind": "nn_model",
            "nn": nn_meta_json,
        }
    }))
}

/// Translate an [`algocline_nn::merged::MergeError`] into a
/// `merge_lora`-prefixed Lua error. Kept as a single translation
/// site to mirror the existing `wrap_gpt2_lora_from_meta` error
/// wall (§Layer 4b bridge discipline).
fn merge_error_to_lua(err: MergeError) -> LuaError {
    let msg = match err {
        MergeError::Provenance(inner) => format!("alc.nn.card.merge_lora: provenance: {inner}"),
        MergeError::Merge(inner) => format!("alc.nn.card.merge_lora: merge: {inner}"),
        MergeError::Io(inner) => format!("alc.nn.card.merge_lora: io: {inner}"),
        MergeError::Serialize(inner) => format!("alc.nn.card.merge_lora: serialize: {inner}"),
    };
    LuaError::external(msg)
}

/// Layer 5a `alc.nn.card.merge_lora` core.
///
/// Accepts a LoRA-wrapped [`NnHandle`] (or a typed handle for
/// backward-compat with `alc.nn.preset.gpt2` / `.tinyllama`
/// wrap-side flows) plus a Lua options table carrying `name` +
/// `lora_card`, and:
///
/// 1. Refuses base (non-LoRA) handles with a directional error
///    pointing at `alc.nn.card.load_wrap`.
/// 2. Pre-generates the merged card_id (mirrors [`save_impl`]).
/// 3. Builds [`MergedProvenance`] with `arch` derived from the
///    handle (never caller-supplied) and `bundle_ref` fixed to
///    `"nn/<merged_card_id>"` (invariant #1 from `save_impl`).
/// 4. Dispatches [`export_merged`] on the underlying typed model,
///    which writes the safetensors bundle under
///    `<nn_dir>/nn/<merged_card_id>.safetensors` and returns the
///    projected [`NnCardMeta`].
/// 5. Persists the Card via [`build_create_payload_from_meta`] +
///    `FileCardStore::create`, asserting the returned id matches
///    the pre-generated one.
///
/// Returns the freshly-minted merged card_id string.
fn merge_lora_impl(
    store: &FileCardStore,
    nn_dir: &std::path::Path,
    base_handle: &LuaAnyUserData,
    opts: LuaTable,
) -> LuaResult<String> {
    // 1. Extract + validate opts (name / lora_card).
    let name: Option<String> = opts.get("name")?;
    let name = name.filter(|s| !s.is_empty()).ok_or_else(|| {
        LuaError::external("alc.nn.card.merge_lora: opts.name must be a non-empty string")
    })?;

    let lora_card: Option<String> = opts.get("lora_card")?;
    let lora_card = lora_card.filter(|s| !s.is_empty()).ok_or_else(|| {
        LuaError::external("alc.nn.card.merge_lora: opts.lora_card must be a non-empty string")
    })?;

    // 2. Borrow the wrapped handle. Accept NnHandle (arch-neutral
    //    return from card.load_wrap / preset(family, ...)) or a
    //    typed Handle (from the preset.gpt2 / preset.tinyllama
    //    entry points once a future L5b wrap adds a Lua-side wrap
    //    that returns typed). Base / non-wrapped handles are
    //    refused with a directional error.
    let handle: NnHandle = if let Ok(nn) = base_handle.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = base_handle.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = base_handle.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = base_handle.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.card.merge_lora: base handle is not a recognised NnHandle / \
             Gpt2Handle / TinyLlamaHandle / LlamaHandle",
        ));
    };

    if !handle.is_lora_wrapped() {
        return Err(LuaError::external(format!(
            "alc.nn.card.merge_lora: handle is not LoRA-wrapped (arch={:?}); \
             use `alc.nn.card.load_wrap(lora_card_id, base_handle)` to obtain a \
             wrapped handle before calling merge_lora",
            handle.arch()
        )));
    }

    // 3. Pre-generate merged_card_id + derive arch + bundle_ref.
    let merged_card_id = generate_card_id(&name);
    let arch = handle.arch_family_variant();
    let bundle_ref = format!("nn/{merged_card_id}");

    let provenance = MergedProvenance {
        lora_card,
        arch,
        bundle_ref,
    };

    // 4. Compute out_path and dispatch export_merged per arch.
    //    export_merged handles parent-dir mkdir + safetensors save
    //    + NnCardMeta projection internally. The on-disk layout
    //    matches load_handle_impl's resolution — `nn_dir/<id>
    //    .safetensors` directly (the "nn/" bundle_ref prefix is a
    //    logical Card reference, not a filesystem subdir).
    let out_path = nn_dir.join(format!("{merged_card_id}.safetensors"));

    let (_bytes, meta) = match &handle {
        NnHandle::Gpt2(gpt2) => {
            let model_arc = gpt2.model();
            let model_guard = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.card.merge_lora: model lock: {e}"))
            })?;
            export_merged(&*model_guard, &provenance, &out_path).map_err(merge_error_to_lua)?
        }
        NnHandle::TinyLlama(tll) => {
            let model_arc = tll.model();
            let model_guard = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.card.merge_lora: model lock: {e}"))
            })?;
            export_merged(&*model_guard, &provenance, &out_path).map_err(merge_error_to_lua)?
        }
        NnHandle::Llama(_) => {
            // Defensive: is_lora_wrapped returns false for Llama
            // already, so this arm is unreachable in practice.
            return Err(LuaError::external(
                "alc.nn.card.merge_lora: llama adapter path does not support LoRA merge",
            ));
        }
    };

    // 5. Overwrite meta.name to the caller-supplied name (the
    //    projection defaults to the bundle file stem; keep the
    //    user-visible name for the Card record instead).
    let mut meta = meta;
    meta.name = name.clone();

    let payload = build_create_payload_from_meta(&merged_card_id, &meta)?;

    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| LuaError::external(format!("alc.nn.card.merge_lora: card store: {e}")))?;

    if returned_id != merged_card_id {
        return Err(LuaError::external(format!(
            "alc.nn.card.merge_lora: card_id mismatch (expected {merged_card_id}, got {returned_id})"
        )));
    }

    Ok(merged_card_id)
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

// Widened to `pub(super)` for L5b-S2 `nn_trainer.rs::run_lora_ft_impl`
// which generates its LoRA card_id via the same
// `<sanitized_name>_<epoch_us>` convention (mirrors save_impl /
// merge_lora_impl). Kept module-private otherwise.
pub(super) fn sanitize_name(name: &str) -> String {
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

// Widened to `pub(super)` for L5b-S2 `nn_trainer.rs::run_lora_ft_impl`
// which generates its LoRA card_id via the same
// `<sanitized_name>_<epoch_us>` convention (mirrors save_impl /
// merge_lora_impl). Kept module-private otherwise.
pub(super) fn compact_epoch_us() -> String {
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
#[derive(Clone)]
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
    /// True iff this handle's underlying [`Gpt2Model`] has been
    /// LoRA-wrapped (via [`wrap_gpt2_lora_from_meta`] or an equivalent
    /// upcoming Layer 5b Lua-side wrap entry). Base handles from
    /// `preset.gpt2` / `from_safetensors` carry `false`.
    ///
    /// [`NnHandle::is_lora_wrapped`] and the Layer 5a
    /// `alc.nn.card.merge_lora` bridge consult this to refuse
    /// merge_lora on a base handle with a directional error.
    pub(super) has_lora: bool,
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

    /// Test-only: strip the `VarMap` off a from-scratch handle so it
    /// mimics the varmap-less state of a `pretrained = true` handle
    /// without the HF hub download that [`Gpt2Model::from_pretrained`]
    /// requires. Lets sibling-module tests (`nn_trainer.rs`) exercise
    /// the trainer bindings' pretrained-refusal guards through the
    /// full dispatch path. Same cross-module test-constructor pattern
    /// as [`DatasetHandle::for_test`].
    #[cfg(test)]
    pub(super) fn for_test_pretrained_like(mut self) -> Self {
        self.varmap = None;
        self.pretrained = true;
        self
    }
}

fn register_preset_ns(lua: &Lua, nn_table: &LuaTable, nn_dir: PathBuf) -> LuaResult<()> {
    let preset = lua.create_table()?;

    // Typed aliases (kept as backward-compat entry points; Layer 4b
    // §Q2-A). Each returns its typed Handle directly so existing
    // callers + the trainer bindings that borrow the typed Handle
    // continue to work unchanged.
    let gpt2_nn_dir = nn_dir.clone();
    let gpt2 = lua.create_function(
        move |_lua, (variant, opts): (String, Option<LuaTable>)| -> LuaResult<Gpt2Handle> {
            build_gpt2_handle(&variant, opts.as_ref(), &gpt2_nn_dir)
        },
    )?;
    preset.set("gpt2", gpt2)?;

    // TinyLlama-family trainable preset (Layer 4b S2). Mirrors
    // `alc.nn.preset.gpt2` — returns a `TinyLlamaHandle` directly
    // for callers that want an arch-pinned entry point.
    let tinyllama_nn_dir = nn_dir.clone();
    let tinyllama = lua.create_function(
        move |_lua, (variant, opts): (String, Option<LuaTable>)| -> LuaResult<TinyLlamaHandle> {
            build_tinyllama_handle(&variant, opts.as_ref(), &tinyllama_nn_dir)
        },
    )?;
    preset.set("tinyllama", tinyllama)?;

    // Llama-family inference preset (GH #9 Layer 2). Wraps
    // `candle_transformers::models::llama` through
    // `algocline_nn::arch::adapter::LlamaAdapter` and hands Lua a
    // `LlamaHandle` UserData mirroring the `Gpt2Handle` shape. This
    // path is inference-only: `LlamaHandle` deliberately does not
    // carry a `VarMap`, so `alc.nn.trainer.*` refuses it up front
    // rather than silently producing a no-op training loop.
    let llama = lua.create_function(
        move |_lua, (variant, opts): (String, Option<LuaTable>)| -> LuaResult<LlamaHandle> {
            build_llama_handle(&variant, opts.as_ref())
        },
    )?;
    preset.set("llama", llama)?;

    // Arch-neutral entry (Layer 4b §Q2-A): `alc.nn.preset(arch,
    // variant, opts)` — dispatches to the arch's build helper +
    // wraps the result in `NnHandle` for uniform Lua-side surface.
    // Implemented via a `__call` metamethod on the preset table so
    // `alc.nn.preset("gpt2", "medium")` reads naturally alongside
    // `alc.nn.preset.gpt2("medium")` (both live under the same
    // name). Callers that need the typed handle keep using the
    // typed alias; callers that want arch-agnostic code get an
    // `NnHandle` back.
    let neutral_nn_dir = nn_dir.clone();
    let preset_call = lua.create_function(
        move |_lua,
              (_self, arch, variant, opts): (LuaTable, String, String, Option<LuaTable>)|
              -> LuaResult<NnHandle> {
            build_neutral_preset(&arch, &variant, opts.as_ref(), &neutral_nn_dir)
        },
    )?;
    let preset_meta = lua.create_table()?;
    preset_meta.set("__call", preset_call)?;
    preset.set_metatable(Some(preset_meta))?;

    nn_table.set("preset", preset)?;
    Ok(())
}

/// Arch-neutral preset entry — dispatches via `ARCH_OPS` and wraps
/// the resulting typed handle in `NnHandle`. Layer 4b §Q2-A / §Q4-A.
fn build_neutral_preset(
    arch: &str,
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<NnHandle> {
    let ops = resolve_arch_ops(arch).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.preset: arch '{arch}' not registered \
             (expected one of {}); qwen2 / phi / gemma are declared in \
             SUPPORTED_ARCHITECTURE_FAMILIES but do not yet have a \
             bridge preset entry",
            registered_arch_names().join(" / ")
        ))
    })?;
    (ops.build_preset)(variant, opts, nn_dir)
}

/// Per-arch bridge operations dispatched at runtime by arch family
/// prefix. Layer 4b §Q4-A: a static `&[(family_prefix, ArchOps)]`
/// table lives in this file; each entry is a set of function
/// pointers the neutral Lua entries call.
///
/// Adding a new arch = new tuple in [`ARCH_OPS`] + new
/// `build_<arch>_handle` (preset) + new `load_<arch>_from_card`
/// (S4 self-contained) + new `wrap_<arch>_lora_from_card` (S5
/// wrap). All function pointers are `fn`-typed (not `Fn` closures)
/// so registration stays purely `const` — no dynamic allocation
/// per invocation.
///
/// The `build_from_safetensors` / `build_from_wrap` slots are
/// `None` for arches without a card-load path yet (currently
/// `llama-adapter` is preset-only; TinyLlama full card-load
/// wiring lands in S4/S5).
struct ArchOps {
    build_preset: fn(&str, Option<&LuaTable>, &std::path::Path) -> LuaResult<NnHandle>,
    build_from_safetensors: Option<fn(&NnCardMeta, &std::path::Path) -> LuaResult<NnHandle>>,
    #[allow(dead_code)]
    build_from_wrap: Option<fn(&NnCardMeta, &NnHandle) -> LuaResult<NnHandle>>,
}

const ARCH_OPS: &[(&str, ArchOps)] = &[
    (
        "gpt2",
        ArchOps {
            build_preset: preset_gpt2_neutral,
            build_from_safetensors: Some(gpt2_from_safetensors),
            build_from_wrap: Some(wrap_gpt2_from_card),
        },
    ),
    (
        "tinyllama",
        ArchOps {
            build_preset: preset_tinyllama_neutral,
            build_from_safetensors: Some(tinyllama_from_safetensors),
            build_from_wrap: Some(wrap_tinyllama_from_card),
        },
    ),
    (
        "llama",
        ArchOps {
            // Adapter-style inference arch: card load path uses
            // GGUF / sharded safetensors + a different construction
            // flow than the trainable arches; no
            // `build_from_safetensors` slot until that lands (§8
            // carry).
            build_preset: preset_llama_neutral,
            build_from_safetensors: None,
            build_from_wrap: None,
        },
    ),
];

/// Resolve `arch` (e.g. `"gpt2"` or `"gpt2-medium"`) to its
/// [`ArchOps`] entry via family-prefix match against [`ARCH_OPS`].
/// Layer 4b §Q4-A. Returns `None` for arches not registered on the
/// bridge yet.
///
/// Matching rules mirror
/// [`algocline_nn::card::validate_architecture`]: bare family name
/// or `<family>-<variant>` (a `-` boundary) both count. A longer
/// identifier that merely starts with a family name (e.g.
/// `"gpt2experimental"`) does NOT match — the namespace stays
/// partitioned.
fn resolve_arch_ops(arch: &str) -> Option<&'static ArchOps> {
    for (family, ops) in ARCH_OPS {
        if arch == *family {
            return Some(ops);
        }
        if let Some(rest) = arch.strip_prefix(*family) {
            if rest.starts_with('-') {
                return Some(ops);
            }
        }
    }
    None
}

/// Registered arch family names (for error messages).
fn registered_arch_names() -> Vec<&'static str> {
    ARCH_OPS.iter().map(|(name, _)| *name).collect()
}

// ─── ArchOps.build_preset adapters ────────────────────────────────
// Each adapter converts the arch-specific typed-handle builder into
// the uniform `fn(&str, Option<&LuaTable>, &Path) -> LuaResult<NnHandle>`
// signature the static table expects.

fn preset_gpt2_neutral(
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<NnHandle> {
    build_gpt2_handle(variant, opts, nn_dir).map(NnHandle::Gpt2)
}

fn preset_tinyllama_neutral(
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<NnHandle> {
    build_tinyllama_handle(variant, opts, nn_dir).map(NnHandle::TinyLlama)
}

fn preset_llama_neutral(
    variant: &str,
    opts: Option<&LuaTable>,
    _nn_dir: &std::path::Path,
) -> LuaResult<NnHandle> {
    // Llama adapter does not consume `nn_dir` (its weights are
    // resolved from `opts.weights` — see `build_llama_handle`).
    // Accepted here for signature uniformity.
    build_llama_handle(variant, opts).map(NnHandle::Llama)
}

// ─── ArchOps.build_from_safetensors adapters (Layer 4b S4) ────────
// Self-contained card load: read the card's architecture +
// candle-branch overrides, resolve the bundle path (bridge already
// resolved it and passes as `&Path`), call the arch's
// `from_safetensors_file`, wrap the result in a typed Handle +
// NnHandle. `VarMap` is always `None` on this path — the file is
// loaded through an mmap-backed VarBuilder that has no VarMap
// counterpart (matches `Gpt2Model::from_pretrained` today).

fn gpt2_from_safetensors(meta: &NnCardMeta, path: &std::path::Path) -> LuaResult<NnHandle> {
    // `Gpt2Config::from_variant` accepts both bare ("medium") and
    // "gpt2-medium" forms — pass the card's architecture string
    // directly.
    let mut cfg = Gpt2Config::from_variant(&meta.architecture).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load: unknown gpt2 variant {:?} on card {:?}",
            meta.architecture, meta.name
        ))
    })?;
    apply_candle_branch_device_dtype("alc.nn.card.load", meta, &mut cfg.device, &mut cfg.dtype)?;
    guard_device_dtype_matrix("alc.nn.card.load", &cfg.device, cfg.dtype)?;

    let model = Gpt2Model::from_safetensors_file(&cfg, path)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load: {e}")))?;
    let (device_str, dtype_str) = candle_branch_device_dtype_strings(meta, &cfg.device, cfg.dtype);
    Ok(NnHandle::Gpt2(Gpt2Handle {
        inner: Arc::new(Mutex::new(model)),
        varmap: None,
        variant: meta.architecture.clone(),
        layers: cfg.layers,
        heads: cfg.heads,
        dim: cfg.dim,
        ctx: cfg.ctx,
        vocab: cfg.vocab,
        device: device_str,
        dtype: dtype_str,
        // mmap-backed load = weights come from the file, treat as
        // "pretrained" from the trainer's perspective (VarMap
        // absent means trainer bindings refuse cleanly).
        pretrained: true,
        has_lora: false,
    }))
}

fn tinyllama_from_safetensors(meta: &NnCardMeta, path: &std::path::Path) -> LuaResult<NnHandle> {
    let mut cfg = TinyLlamaConfig::from_variant(&meta.architecture).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load: unknown tinyllama variant {:?} on card {:?}",
            meta.architecture, meta.name
        ))
    })?;
    apply_candle_branch_device_dtype("alc.nn.card.load", meta, &mut cfg.device, &mut cfg.dtype)?;
    guard_device_dtype_matrix("alc.nn.card.load", &cfg.device, cfg.dtype)?;

    let model = TinyLlamaModel::from_safetensors_file(&cfg, path)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load: {e}")))?;
    let (device_str, dtype_str) = candle_branch_device_dtype_strings(meta, &cfg.device, cfg.dtype);
    Ok(NnHandle::TinyLlama(TinyLlamaHandle {
        inner: Arc::new(Mutex::new(model)),
        varmap: None,
        variant: meta.architecture.clone(),
        layers: cfg.layers,
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        dim: cfg.dim,
        ctx: cfg.ctx,
        vocab: cfg.vocab,
        device: device_str,
        dtype: dtype_str,
        pretrained: true,
        has_lora: false,
    }))
}

/// Apply `meta.candle.device` / `meta.candle.dtype` overrides on
/// top of the arch's default device/dtype. Absent fields leave the
/// arch default unchanged.
fn apply_candle_branch_device_dtype(
    ctx: &str,
    meta: &NnCardMeta,
    device: &mut Device,
    dtype: &mut DType,
) -> LuaResult<()> {
    if let Some(candle) = &meta.candle {
        if let Some(device_str) = &candle.device {
            *device = parse_device_for(ctx, device_str)?;
        }
        if let Some(dtype_str) = &candle.dtype {
            *dtype = parse_dtype_for(ctx, dtype_str)?;
        }
    }
    Ok(())
}

/// Resolve the string-form device / dtype for a Handle's metadata
/// slots. Prefers explicit values from the card's candle branch;
/// falls back to the effective device / dtype used for load.
fn candle_branch_device_dtype_strings(
    meta: &NnCardMeta,
    effective_device: &Device,
    effective_dtype: DType,
) -> (String, String) {
    let device_str = meta
        .candle
        .as_ref()
        .and_then(|c| c.device.clone())
        .unwrap_or_else(|| device_display(effective_device));
    let dtype_str = meta
        .candle
        .as_ref()
        .and_then(|c| c.dtype.clone())
        .unwrap_or_else(|| dtype_display(effective_dtype));
    (device_str, dtype_str)
}

fn device_display(d: &Device) -> String {
    match d {
        Device::Cpu => "cpu".into(),
        Device::Cuda(_) => "cuda".into(),
        Device::Metal(_) => "metal".into(),
    }
}

// ─── ArchOps.build_from_wrap adapters (Layer 4b S5) ───────────────
// Arch-neutral LoRA wrap load: downcast NnHandle to the typed
// handle, delegate to the shared `wrap_<arch>_lora_from_meta`
// core, wrap the result back in NnHandle.

fn wrap_gpt2_from_card(meta: &NnCardMeta, base: &NnHandle) -> LuaResult<NnHandle> {
    let gpt2 = base.as_gpt2().ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_wrap: gpt2 card requires a gpt2 base handle; got '{}'",
            base.arch()
        ))
    })?;
    // Card id is not carried through the ArchOps signature; the
    // caller (load_wrap_impl) is responsible for surfacing it in
    // error context via the meta.name field if it wants a
    // card-id-specific message.
    let card_id = meta.name.as_str();
    let wrapped = wrap_gpt2_lora_from_meta("alc.nn.card.load_wrap", card_id, meta, gpt2)?;
    Ok(NnHandle::Gpt2(wrapped))
}

fn wrap_tinyllama_from_card(meta: &NnCardMeta, base: &NnHandle) -> LuaResult<NnHandle> {
    let tll = base.as_tinyllama().ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_wrap: tinyllama card requires a tinyllama base handle; got '{}'",
            base.arch()
        ))
    })?;
    let card_id = meta.name.as_str();
    let wrapped = wrap_tinyllama_lora_from_meta("alc.nn.card.load_wrap", card_id, meta, tll)?;
    Ok(NnHandle::TinyLlama(wrapped))
}

/// Arch-neutral LoRA card loader (Layer 4b §Q3-A `load_wrap`).
/// Refuses non-LoRA cards with a directional error pointing at
/// `alc.nn.card.load_handle`.
///
/// `base_handle` accepts either a typed handle (`Gpt2Handle` /
/// `TinyLlamaHandle` — backward compat from existing typed
/// preset entries) or an `NnHandle` (new arch-neutral preset).
/// The typed-handle path is lifted into `NnHandle` via Clone.
///
/// Widened to `pub(super)` so L5b-S2
/// `nn_trainer::run_lora_ft_bridge_tests` (C3 / D1 / D2) can invoke
/// the arch-neutral loader on a freshly-written LoRA Card. Kept
/// module-private to callers otherwise (production callers reach it
/// via the `alc.nn.card.load_wrap` Lua closure registered in
/// `register_nn_card`).
pub(super) fn load_wrap_impl(
    store: &FileCardStore,
    card_id: &str,
    base_handle: &LuaAnyUserData,
) -> LuaResult<NnHandle> {
    let card = store
        .get(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_wrap: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load_wrap: card '{card_id}' not found"))
        })?;
    let mut meta = extract_nn_card_meta("alc.nn.card.load_wrap", card_id, &card)?;
    // Overwrite meta.name with card_id so downstream error
    // messages from the wrap core reference the caller-visible
    // id (the meta.name field is user-set at save time and can
    // diverge from card_id).
    meta.name = card_id.to_string();

    // training_path 分岐: lora のみ受け付ける。 self-contained
    // (full_ft / merged / distillation) は load_handle 側に誘導。
    match meta.training_path.as_str() {
        "lora" => {}
        "full_ft" | "merged" | "distillation" => {
            return Err(LuaError::external(format!(
                "alc.nn.card.load_wrap: card '{card_id}' has training_path=\"{}\"; \
                 self-contained cards do not need a base handle — call \
                 `alc.nn.card.load_handle(card_id)` instead",
                meta.training_path
            )));
        }
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.card.load_wrap: card '{card_id}' has unknown training_path \
                 {other:?} (expected one of full_ft / lora / merged / distillation)"
            )));
        }
    }

    // Schema + delta-file precheck BEFORE inspecting base_handle so
    // schema gaps surface as specific errors (parity with the
    // trainer_tests::load_gpt2_impl_errors_* discipline that also
    // guards `load_gpt2_impl`).
    precheck_lora_card_meta("alc.nn.card.load_wrap", card_id, &meta)?;

    let ops = resolve_arch_ops(&meta.architecture).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_wrap: card '{card_id}' architecture {:?} \
             has no bridge dispatch (expected one of {})",
            meta.architecture,
            registered_arch_names().join(" / ")
        ))
    })?;
    let wrap = ops.build_from_wrap.ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.card.load_wrap: card '{card_id}' architecture {:?} \
             does not support LoRA wrap load",
            meta.architecture
        ))
    })?;

    // Try to borrow base_handle as NnHandle first (new arch-neutral
    // preset return); fall back to typed Handle borrow for backward
    // compat with `alc.nn.preset.gpt2` / `alc.nn.preset.tinyllama`
    // callers.
    let base_nn: NnHandle = if let Ok(nn) = base_handle.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = base_handle.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = base_handle.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = base_handle.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.card.load_wrap: base handle is not a recognised NnHandle / \
             Gpt2Handle / TinyLlamaHandle / LlamaHandle",
        ));
    };

    wrap(&meta, &base_nn)
}

fn dtype_display(d: DType) -> String {
    match d {
        DType::F32 => "f32".into(),
        DType::F16 => "f16".into(),
        DType::BF16 => "bf16".into(),
        DType::U8 => "u8".into(),
        DType::U32 => "u32".into(),
        DType::I64 => "i64".into(),
        DType::F64 => "f64".into(),
        // candle_core::DType is `#[non_exhaustive]`; unknown
        // future variants surface with a placeholder rather than
        // panicking. The trainable arches only exercise the
        // f32/f16/bf16 subset today.
        _ => format!("{d:?}"),
    }
}

// ─── alc.nn.preset.llama ──────────────────────────────────────────

/// Opaque handle exposed to Lua for a constructed Llama-family model.
///
/// Inference-only counterpart of [`Gpt2Handle`]: wraps a
/// [`LlamaAdapter`] (which itself wraps
/// `candle_transformers::models::llama::Llama`) and carries the same
/// metadata fields the caller inspects (`variant` / `layers` /
/// `heads` / `dim` / `ctx` / `vocab` / `device` / `dtype`). No
/// `VarMap` is carried because the adapter loads its parameters from
/// a `VarBuilder::from_mmaped_safetensors` reader or a plain
/// `VarBuilder::from_varmap` for the smoke `tiny` variant; the
/// training bindings check the absent `VarMap` on the sibling
/// [`Gpt2Handle`] to refuse from-pretrained handles, and the same
/// invariant applies to Llama by construction (no `varmap()`
/// accessor is exposed).
#[derive(Clone)]
pub(super) struct LlamaHandle {
    inner: Arc<LlamaAdapter>,
    variant: String,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    ctx: usize,
    vocab: usize,
    device: String,
    dtype: String,
}

impl mlua::UserData for LlamaHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("variant", |_, this, ()| Ok(this.variant.clone()));
        methods.add_method("layers", |_, this, ()| Ok(this.layers));
        methods.add_method("heads", |_, this, ()| Ok(this.heads));
        methods.add_method("kv_heads", |_, this, ()| Ok(this.kv_heads));
        methods.add_method("dim", |_, this, ()| Ok(this.dim));
        methods.add_method("ctx", |_, this, ()| Ok(this.ctx));
        methods.add_method("vocab", |_, this, ()| Ok(this.vocab));
        methods.add_method("device", |_, this, ()| Ok(this.device.clone()));
        methods.add_method("dtype", |_, this, ()| Ok(this.dtype.clone()));
        methods.add_method("forward_shape", |_, this, (batch, _seq): (usize, usize)| {
            // Llama forward slices the last-token logits before
            // returning, so the caller-visible output shape is
            // `[batch, vocab]` regardless of the input sequence
            // length.
            Ok(vec![batch, this.vocab])
        });
    }
}

impl LlamaHandle {
    /// Shared handle to the underlying adapter, for callers who want
    /// to drive `forward` from Rust-side helper code.
    #[allow(dead_code)]
    pub(super) fn adapter(&self) -> Arc<LlamaAdapter> {
        Arc::clone(&self.inner)
    }
}

/// Trainable Lua-side handle for [`TinyLlamaModel`].
///
/// Mirrors [`Gpt2Handle`]'s shape (fields + UserData methods) — the
/// only structural difference is the extra `kv_heads` field for GQA.
/// `varmap` semantics are identical: `Some` for `pretrained=false`
/// (from-scratch build, trainer bindings can use it), `None` for
/// `pretrained=true` (mmap-backed VarBuilder load, trainer bindings
/// that need a `VarMap` error out cleanly).
#[derive(Clone)]
pub(super) struct TinyLlamaHandle {
    inner: Arc<Mutex<TinyLlamaModel>>,
    varmap: Option<Arc<VarMap>>,
    variant: String,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    ctx: usize,
    vocab: usize,
    device: String,
    dtype: String,
    pretrained: bool,
    /// True iff this handle's underlying [`TinyLlamaModel`] has been
    /// LoRA-wrapped (via [`wrap_tinyllama_lora_from_meta`]). Base
    /// handles from `preset.tinyllama` / `from_safetensors` carry
    /// `false`. See [`Gpt2Handle::has_lora`] field docs for the full
    /// rationale (mirrored discipline).
    pub(super) has_lora: bool,
}

impl mlua::UserData for TinyLlamaHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("variant", |_, this, ()| Ok(this.variant.clone()));
        methods.add_method("layers", |_, this, ()| Ok(this.layers));
        methods.add_method("heads", |_, this, ()| Ok(this.heads));
        methods.add_method("kv_heads", |_, this, ()| Ok(this.kv_heads));
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

impl TinyLlamaHandle {
    /// Shared handle to the underlying model; mirrors
    /// [`Gpt2Handle::model`].
    #[allow(dead_code)]
    pub(super) fn model(&self) -> Arc<Mutex<TinyLlamaModel>> {
        Arc::clone(&self.inner)
    }

    /// Shared handle to the model's `VarMap`, if constructed from
    /// scratch. Returns `None` for `pretrained = true` handles;
    /// mirrors [`Gpt2Handle::varmap`].
    #[allow(dead_code)]
    pub(super) fn varmap(&self) -> Option<Arc<VarMap>> {
        self.varmap.as_ref().map(Arc::clone)
    }

    /// Test-only: strip the `VarMap` off a from-scratch handle so it
    /// mimics the varmap-less state of a `pretrained = true` handle
    /// without the HF hub download that a real TinyLlama
    /// `from_pretrained` load requires. Lets sibling-module tests
    /// (`nn_trainer.rs`) exercise the trainer bindings'
    /// pretrained-refusal guards through the full dispatch path on the
    /// TinyLlama arm. Mirror of [`Gpt2Handle::for_test_pretrained_like`]
    /// — same cross-module test-constructor pattern as
    /// [`DatasetHandle::for_test`].
    #[cfg(test)]
    pub(super) fn for_test_pretrained_like(mut self) -> Self {
        self.varmap = None;
        self.pretrained = true;
        self
    }
}

fn build_llama_handle(variant: &str, opts: Option<&LuaTable>) -> LuaResult<LlamaHandle> {
    // Resolve the variant with the caller-visible `flash_attn` opt so
    // an `--features flash-attn` GPU build can enable fused attention
    // without a second call site. Defaults to `false` because
    // `candle-transformers`'s flash-attn path requires the extra
    // Cargo feature to be enabled.
    let flash_attn = opts
        .and_then(|t| t.get::<Option<bool>>("flash_attn").ok().flatten())
        .unwrap_or(false);
    let mut cfg = LlamaAdapterConfig::from_variant(variant, flash_attn).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.preset.llama: unknown variant '{variant}' \
             (expected 'tiny' / '7b-v1' / '7b-v2', or one of their 'llama-*' aliases)"
        ))
    })?;

    let device_str = opts
        .and_then(|t| t.get::<Option<String>>("device").ok().flatten())
        .unwrap_or_else(|| "cpu".to_string());
    let dtype_str = opts
        .and_then(|t| t.get::<Option<String>>("dtype").ok().flatten())
        .unwrap_or_else(|| default_dtype_for_device(&device_str).to_string());
    let use_kv_cache = opts
        .and_then(|t| t.get::<Option<bool>>("use_kv_cache").ok().flatten())
        .unwrap_or(true);

    cfg.device = parse_llama_device(&device_str)?;
    cfg.dtype = parse_llama_dtype(&dtype_str)?;
    cfg.use_kv_cache = use_kv_cache;

    guard_device_dtype_matrix("alc.nn.preset.llama", &cfg.device, cfg.dtype)?;

    // Weight source: `opts.weights` is either a single string path or
    // a Lua array of strings for a sharded safetensors bundle. Absent
    // → random VarMap-backed adapter (only useful for the `tiny`
    // smoke variant; caller can still `forward` for shape assertions).
    let weights_paths = extract_weights_paths(opts)?;

    let (adapter, cfg_snapshot) = if let Some(paths) = weights_paths {
        let cfg_snapshot = cfg.clone();
        let adapter = LlamaAdapter::from_safetensors_files(&paths, cfg)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.llama: {e}")))?;
        (adapter, cfg_snapshot)
    } else {
        let cfg_snapshot = cfg.clone();
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.llama: {e}")))?;
        // Drop `vm` on purpose: the adapter is inference-only and
        // never expects a trainable VarMap; the surrounding scope
        // guarantees the mmap-less handle stays valid because the
        // adapter now owns the tensor snapshots.
        drop(vm);
        (adapter, cfg_snapshot)
    };

    Ok(LlamaHandle {
        inner: Arc::new(adapter),
        variant: variant.to_string(),
        layers: cfg_snapshot.config.num_hidden_layers,
        heads: cfg_snapshot.config.num_attention_heads,
        kv_heads: cfg_snapshot.config.num_key_value_heads,
        dim: cfg_snapshot.config.hidden_size,
        ctx: cfg_snapshot.config.max_position_embeddings,
        vocab: cfg_snapshot.config.vocab_size,
        device: device_str,
        dtype: dtype_str,
    })
}

fn extract_weights_paths(opts: Option<&LuaTable>) -> LuaResult<Option<Vec<PathBuf>>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Some(raw) = opts.get::<Option<mlua::Value>>("weights").ok().flatten() else {
        return Ok(None);
    };
    match raw {
        mlua::Value::String(s) => Ok(Some(vec![PathBuf::from(s.to_str()?.to_string())])),
        mlua::Value::Table(tbl) => {
            let mut out = Vec::new();
            for pair in tbl.sequence_values::<String>() {
                out.push(PathBuf::from(pair?));
            }
            if out.is_empty() {
                return Err(LuaError::external(
                    "alc.nn.preset.llama: opts.weights is empty; provide at least one path",
                ));
            }
            Ok(Some(out))
        }
        _ => Err(LuaError::external(
            "alc.nn.preset.llama: opts.weights must be a string or an array of strings",
        )),
    }
}

fn parse_llama_device(s: &str) -> LuaResult<Device> {
    parse_device_for("alc.nn.preset.llama", s)
}

fn parse_llama_dtype(s: &str) -> LuaResult<DType> {
    parse_dtype_for("alc.nn.preset.llama", s)
}

// Widened to `pub(super)` for L5b-S1 `nn_wrap.rs` test scaffolding
// (`setup_gpt2_base_scaffold` builds a base handle in-place). Kept
// module-private otherwise; no production caller outside this module
// consumes it.
pub(super) fn build_gpt2_handle(
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<Gpt2Handle> {
    let mut cfg = if variant == "custom" || variant == "gpt2-custom" {
        build_custom_gpt2_config(opts)?
    } else {
        if let Some(t) = opts {
            reject_custom_only_keys(variant, t)?;
        }
        Gpt2Config::from_variant(variant).ok_or_else(|| {
            LuaError::external(format!(
                "alc.nn.preset.gpt2: unknown variant '{variant}' \
                 (expected 'medium', 'large', 'tiny', or 'custom')"
            ))
        })?
    };

    let device_str = opts
        .and_then(|t| t.get::<Option<String>>("device").ok().flatten())
        .unwrap_or_else(|| "cpu".to_string());
    let dtype_str = opts
        .and_then(|t| t.get::<Option<String>>("dtype").ok().flatten())
        .unwrap_or_else(|| default_dtype_for_device(&device_str).to_string());
    let pretrained = opts
        .and_then(|t| t.get::<Option<bool>>("pretrained").ok().flatten())
        .unwrap_or(true);

    cfg.device = parse_device(&device_str)?;
    cfg.dtype = parse_dtype(&dtype_str)?;

    guard_device_dtype_matrix("alc.nn.preset.gpt2", &cfg.device, cfg.dtype)?;

    // Custom specs are random-init only (same guard family as MoE —
    // `Gpt2Model::from_pretrained` also refuses downstream, but the
    // bridge-level check turns it into an actionable message instead
    // of a failed hub lookup for a bundle that cannot exist).
    if cfg.custom.is_some() && pretrained {
        return Err(LuaError::external(
            "alc.nn.preset.gpt2('custom'): custom architectures are random-init \
             only (no pretrained bundle exists for a customized GPT-2) — pass \
             pretrained = false",
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
        has_lora: false,
    })
}

fn parse_device(s: &str) -> LuaResult<Device> {
    parse_device_for("alc.nn.preset.gpt2", s)
}

/// Option keys that only make sense on the `custom` variant. On any
/// other variant they would silently not take effect (the stock
/// presets never set `Gpt2Config::custom` / `moe` and their shape is
/// fixed), so [`build_gpt2_handle`] rejects them up front instead of
/// shipping a config that looks customized but is not.
const GPT2_CUSTOM_ONLY_KEYS: &[&str] = &[
    "act",
    "norm",
    "residual",
    "mlp_ratio",
    "placement",
    "pos",
    "kv_heads",
    "window",
    "untied_head",
    "moe",
    "layers",
    "heads",
    "dim",
    "ctx",
    "vocab",
];

fn reject_custom_only_keys(variant: &str, t: &LuaTable) -> LuaResult<()> {
    for key in GPT2_CUSTOM_ONLY_KEYS {
        if t.contains_key(*key)? {
            return Err(LuaError::external(format!(
                "alc.nn.preset.gpt2: option '{key}' only applies to the 'custom' \
                 variant (got '{variant}') and would silently not take effect \
                 here — use alc.nn.preset.gpt2('custom', {{ ... }}) for \
                 architecture experiments"
            )));
        }
    }
    Ok(())
}

/// Typed option getter for the `custom` opts table. Unlike the legacy
/// `device` / `dtype` reads (which predate the Service-layer error
/// discipline and swallow type errors via `.ok()`), a present key of
/// the wrong Lua type is a hard, actionable error — a silently
/// ignored `mlp_ratio = "3"` would build the reference MLP while the
/// caller believes they are running a ratio-3 experiment.
fn custom_opt<T: mlua::FromLua>(t: &LuaTable, key: &str, expected: &str) -> LuaResult<Option<T>> {
    t.get::<Option<T>>(key).map_err(|_| {
        LuaError::external(format!(
            "alc.nn.preset.gpt2('custom'): option '{key}' must be {expected}"
        ))
    })
}

fn custom_bad_value(key: &str, got: &str, expected: &str) -> LuaError {
    LuaError::external(format!(
        "alc.nn.preset.gpt2('custom'): unknown {key} '{got}' (expected {expected})"
    ))
}

/// Parse the flat `custom` opts table into a [`Gpt2Config`] carrying
/// `custom: Some(spec)` (and optionally `moe`). Base shape is
/// [`Gpt2Config::tiny`]; `layers` / `heads` / `dim` / `ctx` / `vocab`
/// override it so an experiment can match a real tokenizer's vocab or
/// the arch_probe scale. Axis semantics and cross-axis validation
/// (PostLN×Parallel, GQA divisibility, RoPE even head_dim, MoE
/// dense-knob combination) live Rust-side in `Gpt2Custom::validate` /
/// `Gpt2Model::new`; their messages propagate to Lua verbatim.
fn build_custom_gpt2_config(opts: Option<&LuaTable>) -> LuaResult<Gpt2Config> {
    let mut cfg = Gpt2Config::tiny();
    let mut spec = Gpt2Custom::default();
    let Some(t) = opts else {
        cfg.custom = Some(spec);
        return Ok(cfg);
    };

    if let Some(v) = custom_opt::<usize>(t, "layers", "an integer")? {
        cfg.layers = v;
    }
    if let Some(v) = custom_opt::<usize>(t, "heads", "an integer")? {
        cfg.heads = v;
    }
    if let Some(v) = custom_opt::<usize>(t, "dim", "an integer")? {
        cfg.dim = v;
    }
    if let Some(v) = custom_opt::<usize>(t, "ctx", "an integer")? {
        cfg.ctx = v;
    }
    if let Some(v) = custom_opt::<usize>(t, "vocab", "an integer")? {
        cfg.vocab = v;
    }

    if let Some(s) = custom_opt::<String>(t, "act", "a string")? {
        spec.act = match s.as_str() {
            "gelu" => Activation::Gelu,
            "relu" => Activation::Relu,
            "silu" => Activation::Silu,
            "swiglu" => Activation::SwiGlu,
            "geglu" => Activation::GeGlu,
            other => {
                return Err(custom_bad_value(
                    "act",
                    other,
                    "'gelu' / 'relu' / 'silu' / 'swiglu' / 'geglu'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(t, "norm", "a string")? {
        spec.norm = match s.as_str() {
            "layernorm" => NormKind::LayerNorm,
            "rmsnorm" => NormKind::RmsNorm,
            other => return Err(custom_bad_value("norm", other, "'layernorm' / 'rmsnorm'")),
        };
    }
    if let Some(s) = custom_opt::<String>(t, "residual", "a string")? {
        spec.residual = match s.as_str() {
            "sequential" => ResidualKind::Sequential,
            "parallel" => ResidualKind::Parallel,
            other => {
                return Err(custom_bad_value(
                    "residual",
                    other,
                    "'sequential' / 'parallel'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(t, "placement", "a string")? {
        spec.placement = match s.as_str() {
            "preln" => NormPlacement::PreLn,
            "postln" => NormPlacement::PostLn,
            other => return Err(custom_bad_value("placement", other, "'preln' / 'postln'")),
        };
    }
    if let Some(s) = custom_opt::<String>(t, "pos", "a string")? {
        spec.pos = match s.as_str() {
            "learned" => PosKind::Learned,
            "rope" => PosKind::Rope,
            "alibi" => PosKind::Alibi,
            "nope" => PosKind::NoPos,
            other => {
                return Err(custom_bad_value(
                    "pos",
                    other,
                    "'learned' / 'rope' / 'alibi' / 'nope'",
                ))
            }
        };
    }
    if let Some(v) = custom_opt::<usize>(t, "mlp_ratio", "an integer")? {
        spec.mlp_ratio = v;
    }
    if let Some(v) = custom_opt::<usize>(t, "kv_heads", "an integer")? {
        spec.kv_heads = Some(v);
    }
    if let Some(v) = custom_opt::<usize>(t, "window", "an integer")? {
        spec.window = Some(v);
    }
    if let Some(b) = custom_opt::<bool>(t, "untied_head", "a boolean")? {
        spec.untied_head = b;
    }

    cfg.moe = parse_custom_moe(t)?;
    cfg.custom = Some(spec);
    Ok(cfg)
}

/// Parse the optional nested `moe = { n_experts, top_k?, alpha? }`
/// table. Defaults mirror [`MoeConfig::new`] (Mixtral top-2 routing,
/// Switch α = 0.01); `MoeConfig::validate` runs at build time in
/// `Gpt2Model::new`.
fn parse_custom_moe(t: &LuaTable) -> LuaResult<Option<MoeConfig>> {
    let Some(m) = custom_opt::<LuaTable>(t, "moe", "a table")? else {
        return Ok(None);
    };
    let n_experts = custom_opt::<usize>(&m, "n_experts", "an integer")?.ok_or_else(|| {
        LuaError::external("alc.nn.preset.gpt2('custom'): moe.n_experts is required (integer ≥ 1)")
    })?;
    let mut moe = MoeConfig::new(n_experts);
    if let Some(k) = custom_opt::<usize>(&m, "top_k", "an integer")? {
        moe.top_k = k;
    }
    if let Some(a) = custom_opt::<f64>(&m, "alpha", "a number")? {
        moe.alpha = a;
    }
    Ok(Some(moe))
}

/// Build a trainable TinyLlama handle. Mirrors
/// [`build_gpt2_handle`] — same option shape (`device` / `dtype` /
/// `pretrained`) and the same VarMap `Some` / `None` semantics
/// carry over. `pretrained=true` requires a variant with an HF
/// repo (`tinyllama-1.1b`); the `tinyllama-tiny` smoke variant
/// only supports `pretrained=false` (mirrors GPT-2's `tiny` case).
// Widened to `pub(super)` for L5b-S1 `nn_wrap.rs` test scaffolding
// (`setup_tinyllama_base_scaffold` builds a base handle in-place).
// Kept module-private otherwise; no production caller outside this
// module consumes it.
pub(super) fn build_tinyllama_handle(
    variant: &str,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
) -> LuaResult<TinyLlamaHandle> {
    let mut cfg = TinyLlamaConfig::from_variant(variant).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.preset.tinyllama: unknown variant '{variant}' \
             (expected 'tinyllama-1.1b' / '1.1b' / 'tinyllama-tiny' / 'tiny')"
        ))
    })?;

    let device_str = opts
        .and_then(|t| t.get::<Option<String>>("device").ok().flatten())
        .unwrap_or_else(|| "cpu".to_string());
    let dtype_str = opts
        .and_then(|t| t.get::<Option<String>>("dtype").ok().flatten())
        .unwrap_or_else(|| default_dtype_for_device(&device_str).to_string());
    let pretrained = opts
        .and_then(|t| t.get::<Option<bool>>("pretrained").ok().flatten())
        .unwrap_or(true);

    cfg.device = parse_device_for("alc.nn.preset.tinyllama", &device_str)?;
    cfg.dtype = parse_dtype_for("alc.nn.preset.tinyllama", &dtype_str)?;

    guard_device_dtype_matrix("alc.nn.preset.tinyllama", &cfg.device, cfg.dtype)?;

    let (model, varmap) = if pretrained {
        let cache_dir = nn_dir.to_path_buf();
        let m = TinyLlamaModel::from_pretrained(variant, &cfg, &cache_dir)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.tinyllama: {e}")))?;
        (m, None)
    } else {
        let vm = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let m = TinyLlamaModel::new(&cfg, vs)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.tinyllama: {e}")))?;
        (m, Some(Arc::new(vm)))
    };

    Ok(TinyLlamaHandle {
        inner: Arc::new(Mutex::new(model)),
        varmap,
        variant: variant.to_string(),
        layers: cfg.layers,
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        dim: cfg.dim,
        ctx: cfg.ctx,
        vocab: cfg.vocab,
        device: device_str,
        dtype: dtype_str,
        pretrained,
        has_lora: false,
    })
}

/// Resolve the default dtype for a caller-provided device string when
/// the caller does not supply `opts.dtype` explicitly.
///
/// Matrix (GH #9 Layer 3):
///
/// - `"cuda"` / `"cuda:N"` — `"bf16"` (the design §6.1 GPU default).
/// - `"metal"` / `"metal:N"` — `"f16"` (Metal has no bf16 kernels;
///   f16 keeps the same memory footprint benefit).
/// - Everything else (including unrecognized strings — they will fail
///   parsing downstream in [`parse_device_for`]) — `"f32"`.
///
/// The returned string is fed straight into [`parse_dtype_for`], so
/// keeping this a `&'static str` avoids an allocation on the common
/// path.
fn default_dtype_for_device(device: &str) -> &'static str {
    if device.starts_with("cuda") {
        "bf16"
    } else if device.starts_with("metal") {
        "f16"
    } else {
        "f32"
    }
}

fn parse_dtype(s: &str) -> LuaResult<DType> {
    parse_dtype_for("alc.nn.preset.gpt2", s)
}

/// Shared device string parser used by both `alc.nn.preset.gpt2` and
/// `alc.nn.preset.llama`. `preset` is the caller-facing tag used in
/// error messages so a Lua-side error still points at the right
/// preset call site.
///
/// Accepted device strings (GH #9 Layer 3 dtype / device matrix):
///
/// - `"cpu"` — always available.
/// - `"cuda"` / `"cuda:N"` — enabled only in a `--features nn-cuda`
///   build; the runtime `Device::new_cuda(...)` call errors otherwise.
/// - `"metal"` / `"metal:N"` — enabled only in a `--features nn-metal`
///   build; the runtime `Device::new_metal(...)` call errors otherwise.
fn parse_device_for(preset: &str, s: &str) -> LuaResult<Device> {
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let ord: usize = rest.parse().map_err(|e| {
            LuaError::external(format!("{preset}: invalid cuda ordinal '{rest}': {e}"))
        })?;
        return Device::new_cuda(ord)
            .map_err(|e| LuaError::external(format!("{preset}: cuda:{ord} unavailable: {e}")));
    }
    if s == "cuda" {
        return Device::new_cuda(0)
            .map_err(|e| LuaError::external(format!("{preset}: cuda unavailable: {e}")));
    }
    if let Some(rest) = s.strip_prefix("metal:") {
        let ord: usize = rest.parse().map_err(|e| {
            LuaError::external(format!("{preset}: invalid metal ordinal '{rest}': {e}"))
        })?;
        return Device::new_metal(ord)
            .map_err(|e| LuaError::external(format!("{preset}: metal:{ord} unavailable: {e}")));
    }
    if s == "metal" {
        return Device::new_metal(0)
            .map_err(|e| LuaError::external(format!("{preset}: metal unavailable: {e}")));
    }
    Err(LuaError::external(format!(
        "{preset}: unknown device '{s}' (expected 'cpu', 'cuda', 'cuda:N', 'metal', or 'metal:N')"
    )))
}

/// Shared dtype string parser. See [`parse_device_for`] for the sibling
/// dtype / device matrix rationale. Downstream device / dtype
/// combinability is validated by [`guard_device_dtype_matrix`].
fn parse_dtype_for(preset: &str, s: &str) -> LuaResult<DType> {
    match s {
        "f32" | "fp32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        "f16" | "fp16" => Ok(DType::F16),
        other => Err(LuaError::external(format!(
            "{preset}: unknown dtype '{other}' (expected 'f32', 'bf16', or 'f16')"
        ))),
    }
}

/// Validate a resolved (device, dtype) pair against candle 0.11's
/// backend support matrix. Called by both `alc.nn.preset.gpt2` and
/// `alc.nn.preset.llama` so the two presets have identical failure
/// modes for unsupported combinations.
///
/// Matrix (GH #9 Layer 3):
///
/// | device | f32 | bf16 | f16 |
/// |--------|-----|------|-----|
/// | CPU    | ok  | ERR  | ok  |
/// | CUDA   | ok  | ok   | ok  |
/// | Metal  | ok  | ERR  | ok  |
///
/// `bf16` is rejected up front on CPU and Metal because candle-nn 0.11
/// only ships bf16 kernels for the CUDA backend; the CPU / Metal
/// fallback would otherwise fail deep inside a forward pass with an
/// opaque kernel-not-found error. Every other combination is passed
/// through to candle — if a specific kernel is unimplemented for the
/// chosen shape (rare in the shipping presets) the caller still gets a
/// candle-side error surfaced by the preset's `map_err`.
fn guard_device_dtype_matrix(preset: &str, device: &Device, dtype: DType) -> LuaResult<()> {
    if dtype != DType::BF16 {
        return Ok(());
    }
    match device {
        Device::Cpu => Err(LuaError::external(format!(
            "{preset}: bf16 dtype requires a CUDA device (use dtype='f32' on CPU, \
             or dtype='f16' on Metal)"
        ))),
        Device::Metal(_) => Err(LuaError::external(format!(
            "{preset}: bf16 dtype is not supported on Metal (use dtype='f16' or 'f32' on Metal, \
             or move to CUDA for bf16)"
        ))),
        _ => Ok(()),
    }
}

/// Reject bf16 base handles at the trainer entrypoints.
///
/// Sibling to [`guard_device_dtype_matrix`] — that guard rejects bf16
/// at preset build time on non-CUDA devices; this guard rejects bf16
/// at trainer entry regardless of device (bf16 is inference-only on
/// the current trainer path). Called from the four L5b/L5c trainer
/// impl fns (`wrap_lora_impl` / `run_lora_ft_impl` / `run_full_ft_impl`
/// / `run_distill_impl`) between the Llama refusal (step 4) and the
/// opts / dataset processing (step 5), so a bf16 base surfaces a
/// directional Lua error rather than an opaque candle
/// `unexpected dtype, expected: F32, got: BF16` raised deep inside a
/// backward pass.
pub(super) fn guard_base_dtype_for_training(fn_name: &str, handle: &NnHandle) -> LuaResult<()> {
    let dtype = match handle {
        NnHandle::Gpt2(h) => h.dtype.as_str(),
        NnHandle::TinyLlama(h) => h.dtype.as_str(),
        NnHandle::Llama(h) => h.dtype.as_str(),
    };
    if dtype.eq_ignore_ascii_case("bf16") {
        return Err(LuaError::external(format!(
            "{fn_name}: training requires an f32 base (got bf16); \
             build the preset with dtype=\"f32\" (bf16 base is \
             inference-only, not supported by the trainer path)"
        )));
    }
    Ok(())
}

/// Test-only helper: swap the recorded `dtype` string on a
/// [`Gpt2Handle`] without rebuilding the underlying model. Used by the
/// L5b/L5c bridge tests to exercise the bf16 handle-time guard
/// (`guard_base_dtype_for_training`) without needing a working bf16
/// build path on CPU (which [`guard_device_dtype_matrix`] would refuse
/// up front at preset construction time).
#[cfg(test)]
pub(super) fn gpt2_handle_with_dtype(mut base: Gpt2Handle, dtype: &str) -> Gpt2Handle {
    base.dtype = dtype.to_string();
    base
}

/// Test-only helper: sibling of [`gpt2_handle_with_dtype`] for
/// [`TinyLlamaHandle`].
#[cfg(test)]
pub(super) fn tinyllama_handle_with_dtype(
    mut base: TinyLlamaHandle,
    dtype: &str,
) -> TinyLlamaHandle {
    base.dtype = dtype.to_string();
    base
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

impl DatasetHandle {
    /// Lock the inner dataset for the duration of a training call.
    ///
    /// Widened accessor for the L5b-S2 sibling
    /// [`super::nn_trainer::run_lora_ft_impl`], which cannot reach
    /// the module-private `inner` field directly. In-module callers
    /// (`full_ft_impl` / `lora_impl` / `distill_impl`) still use
    /// `self.inner.lock()` inline; wrapping is unnecessary for
    /// them.
    pub(super) fn inner_lock(
        &self,
    ) -> LuaResult<std::sync::MutexGuard<'_, Box<dyn Dataset + Send>>> {
        self.inner.lock().map_err(|e| {
            LuaError::external(format!(
                "alc.nn.trainer.run_lora_ft: dataset lock poisoned: {e}"
            ))
        })
    }

    /// Construct a [`DatasetHandle`] from a boxed [`Dataset`] plus
    /// caller-supplied source / batch / ctx metadata. Test-only
    /// helper for the L5b-S2 bridge test module, which builds a
    /// [`TokenizedDataset`] in-process rather than routing through
    /// the public `alc.nn.data.jsonl` / `.synthetic` Lua entries.
    #[cfg(test)]
    pub(super) fn for_test(
        inner: Box<dyn Dataset + Send>,
        source: String,
        batch_size: usize,
        ctx_len: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(inner),
            source,
            batch_size,
            ctx_len,
        }
    }
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

/// Loss-mask mode declared by a teacher-log Card under `[metadata]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LossMaskDecl {
    /// Mask out the `prompt` region; score only the `response` region.
    Response,
}

/// Read the `metadata.loss_mask` declaration from a Tier 1 Card JSON.
///
/// `Ok(None)` = the Card carries no declaration (legacy, mask-free path).
/// `Err(_)`   = a declaration is present but not recognized.
pub(super) fn loss_mask_decl_from_card(card: &Json) -> Result<Option<LossMaskDecl>, String> {
    let value = match card
        .get("metadata")
        .and_then(|m| m.as_object())
        .and_then(|m| m.get("loss_mask"))
    {
        Some(v) => v,
        None => return Ok(None),
    };
    match value.as_str() {
        Some("response") => Ok(Some(LossMaskDecl::Response)),
        Some(other) => Err(format!(
            "unknown metadata.loss_mask value {other:?} (expected \"response\")"
        )),
        None => Err(format!(
            "metadata.loss_mask must be a string (expected \"response\"), got {value}"
        )),
    }
}

/// One `(input_ids, loss_mask)` row of a mask-carrying teacher dataset,
/// shaped for [`TeacherCardDataset::from_rows`].
pub(super) type TeacherRow = (Vec<u32>, Vec<f32>);

/// Build `(input_ids, loss_mask)` rows for a mask-declaring teacher Card.
///
/// `encode` is injected so unit tests can exercise the boundary rules
/// without constructing a tokenizer (and therefore without network
/// access). Row `i` of the result pairs the joint `"{prompt}\n{response}"`
/// token ids with a same-length mask that is `0.0` across the prompt
/// tokens and `1.0` across the response tokens.
///
/// The prompt token boundary is derived by encoding the `prompt` field
/// on its own and verifying that its ids are a prefix of the joint
/// encoding — never by a character offset or a delimiter search over
/// the concatenated token stream.
pub(super) fn build_teacher_rows<F>(samples: &[Json], encode: F) -> Result<Vec<TeacherRow>, String>
where
    F: Fn(&str) -> Result<Vec<u32>, String>,
{
    let mut rows: Vec<TeacherRow> = Vec::with_capacity(samples.len());
    for (idx, sample) in samples.iter().enumerate() {
        let prompt = match sample.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return Err(format!(
                    "sample {idx}: metadata.loss_mask is declared but 'prompt' is missing or empty"
                ))
            }
        };
        let response = match sample.get("response").and_then(|v| v.as_str()) {
            Some(r) if !r.is_empty() => r,
            _ => {
                return Err(format!(
                    "sample {idx}: metadata.loss_mask is declared but 'response' is missing or \
                     empty"
                ))
            }
        };

        let prompt_ids = encode(prompt).map_err(|e| format!("sample {idx}: {e}"))?;
        if prompt_ids.is_empty() {
            return Err(format!("sample {idx}: prompt encodes to zero tokens"));
        }
        let joint_ids =
            encode(&format!("{prompt}\n{response}")).map_err(|e| format!("sample {idx}: {e}"))?;
        if joint_ids.len() <= prompt_ids.len() {
            return Err(format!(
                "sample {idx}: no response tokens after the prompt boundary (prompt={} joint={})",
                prompt_ids.len(),
                joint_ids.len()
            ));
        }
        if joint_ids[..prompt_ids.len()] != prompt_ids[..] {
            return Err(format!(
                "sample {idx}: prompt token prefix mismatch (prompt={} joint={}) — the tokenizer \
                 merged the prompt/response boundary",
                prompt_ids.len(),
                joint_ids.len()
            ));
        }

        let mut mask = vec![0.0f32; prompt_ids.len()];
        mask.resize(joint_ids.len(), 1.0f32);
        rows.push((joint_ids, mask));
    }
    Ok(rows)
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

            // Tier 1 `[metadata] loss_mask = "response"` switches this
            // producer to a mask-carrying teacher dataset. Absent
            // declaration falls through to the legacy path below with
            // bit-identical token ids.
            let decl = from_card_store
                .get(&card_id)
                .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?
                .as_ref()
                .map(loss_mask_decl_from_card)
                .transpose()
                .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?
                .flatten();

            if decl == Some(LossMaskDecl::Response) {
                let samples = from_card_store
                    .read_samples(&card_id, SamplesQuery::default())
                    .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?;
                let rows = build_teacher_rows(&samples, |text| {
                    tok.encode(text).map_err(|e| e.to_string())
                })
                .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?;
                let ds = TeacherCardDataset::from_rows(rows, dopts.clone())
                    .map_err(|e| LuaError::external(format!("alc.nn.data.from_card: {e}")))?;
                return Ok(DatasetHandle {
                    inner: Mutex::new(Box::new(ds)),
                    source: format!("card:{card_id}"),
                    batch_size: dopts.batch_size,
                    ctx_len: dopts.ctx_len,
                });
            }

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

    // `run_full_ft` is now generic over `M: Module + DeviceView`; the
    // trait bounds are not satisfied by `MutexGuard<Gpt2Model>` itself
    // (they are on the guarded `Gpt2Model`), so deref through the
    // guard explicitly before handing the reference off.
    let result = run_full_ft(
        &*model,
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

    // `run_lora_ft` is now generic over `M: Module + DeviceView +
    // LoraWrappable`; deref through the guard explicitly so the trait
    // bounds resolve against `Gpt2Model`, not `MutexGuard<Gpt2Model>`.
    let result = run_lora_ft(
        &mut *model,
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
        &*model,
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
///
/// Widened to `pub(super)` for L5b-S2 `nn_trainer.rs::run_lora_ft_impl`
/// which reuses the same train-opts extractor (`lr` / `batch_size` /
/// `steps` / `warmup` / `schedule` / etc.) and layers the `run_lora_ft`
/// bind's stricter validation (steps > 0, batch > 0, lr > 0) on top.
/// Do NOT reimplement the extractor there — keeping one SoT prevents
/// field-set drift between the pre-existing `alc.nn.trainer.lora`
/// / `full_ft` / `distill` entries and the new `run_lora_ft`.
pub(super) fn extract_full_ft_opts(opts: Option<&LuaTable>) -> LuaResult<FullFtConfig> {
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

/// Arch-neutral handle union (Layer 4b §Q1-A).
///
/// Wraps the three per-arch typed handles under a single UserData
/// so `alc.nn.preset(arch, variant, opts)` /
/// `alc.nn.card.load(card_id)` / `alc.nn.card.load_wrap(card_id,
/// base)` can hand Lua a uniform handle regardless of the
/// underlying arch. Method dispatch inside
/// `impl mlua::UserData for NnHandle` fans out to the wrapped
/// typed handle's accessor.
///
/// Existing typed entries (`alc.nn.preset.gpt2` /
/// `alc.nn.card.load_gpt2`) continue to return the typed handle
/// directly for backward compat — the trainer bindings still
/// borrow those typed handles from `LuaAnyUserData` and their
/// migration to accept `NnHandle` is a follow-up (Layer 4b §8).
///
/// New arch = add an enum variant + a match arm to every dispatch
/// method + register the arch in `ARCH_OPS` (§Q4-A) + add a
/// per-arch `build_*_handle` helper. Three grep-able edit sites.
#[allow(dead_code)]
#[derive(Clone)]
pub(super) enum NnHandle {
    /// GPT-2 (trainable, MHA).
    Gpt2(Gpt2Handle),
    /// TinyLlama (trainable, GQA).
    TinyLlama(TinyLlamaHandle),
    /// Llama adapter (inference-only, GQA — different execution
    /// model, held here for symmetry per §1 non-goal / §8 carry).
    Llama(LlamaHandle),
}

impl NnHandle {
    /// Architecture family prefix (`"gpt2"` / `"tinyllama"` /
    /// `"llama"`) matching the first-column entries in
    /// [`algocline_nn::card::SUPPORTED_ARCHITECTURE_FAMILIES`].
    #[allow(dead_code)]
    pub(super) fn arch(&self) -> &'static str {
        match self {
            Self::Gpt2(_) => "gpt2",
            Self::TinyLlama(_) => "tinyllama",
            Self::Llama(_) => "llama",
        }
    }

    /// Typed downcast to the underlying [`Gpt2Handle`], if this
    /// variant is `Gpt2`. Bridge fns that need to reach a
    /// trainer-compatible typed handle use this.
    #[allow(dead_code)]
    pub(super) fn as_gpt2(&self) -> Option<&Gpt2Handle> {
        match self {
            Self::Gpt2(h) => Some(h),
            _ => None,
        }
    }

    /// Typed downcast to the underlying [`TinyLlamaHandle`].
    #[allow(dead_code)]
    pub(super) fn as_tinyllama(&self) -> Option<&TinyLlamaHandle> {
        match self {
            Self::TinyLlama(h) => Some(h),
            _ => None,
        }
    }

    /// Typed downcast to the underlying [`LlamaHandle`].
    #[allow(dead_code)]
    pub(super) fn as_llama(&self) -> Option<&LlamaHandle> {
        match self {
            Self::Llama(h) => Some(h),
            _ => None,
        }
    }

    /// Architecture family + variant identifier suitable for
    /// [`algocline_nn::MergedProvenance::arch`] and the projected
    /// [`algocline_nn::NnCardMeta::architecture`] field.
    ///
    /// The underlying handle's `variant` field is stored in one of
    /// two conventions depending on the construction path:
    ///
    /// - `preset.<arch>(variant, ...)` stores the raw `variant`
    ///   string (e.g. `"medium"` / `"1.1b"`).
    /// - `card.load_handle` (from a saved card) stores the full
    ///   `family-variant` form (e.g. `"gpt2-medium"` /
    ///   `"tinyllama-1.1b"`) because the card's
    ///   `NnCardMeta.architecture` is already in that shape.
    ///
    /// This helper normalises both into the full `family-variant`
    /// form so the Layer 5a `merge_lora` bridge can hand a
    /// consistent value to [`MergedProvenance`] regardless of how
    /// the wrapped handle was obtained. The prefix-strip guard
    /// matches [`wrap_gpt2_lora_from_meta`]'s existing
    /// `base_cfg_id` logic (§Layer 4b invariant carry).
    #[allow(dead_code)]
    pub(super) fn arch_family_variant(&self) -> String {
        let (family, variant) = match self {
            Self::Gpt2(h) => ("gpt2", h.variant.as_str()),
            Self::TinyLlama(h) => ("tinyllama", h.variant.as_str()),
            Self::Llama(h) => ("llama", h.variant.as_str()),
        };
        let prefix = format!("{family}-");
        if variant.starts_with(&prefix) {
            variant.to_string()
        } else {
            format!("{prefix}{variant}")
        }
    }

    /// True iff the underlying model has been LoRA-wrapped.
    ///
    /// Layer 5a `alc.nn.card.merge_lora` consults this to refuse a
    /// base (non-wrapped) handle with a directional error before
    /// invoking [`algocline_nn::export_merged`]. A base handle
    /// would produce a "merged" bundle byte-identical to the base,
    /// which would silently mis-describe the resulting card's
    /// provenance — refusing early keeps the Card store honest.
    ///
    /// `Llama(_)` always returns `false`: the adapter path does not
    /// support LoRA wrap in the current codebase (§Layer 2 non-goal,
    /// carried through Layer 4b).
    #[allow(dead_code)]
    pub(super) fn is_lora_wrapped(&self) -> bool {
        match self {
            Self::Gpt2(h) => h.has_lora,
            Self::TinyLlama(h) => h.has_lora,
            Self::Llama(_) => false,
        }
    }
}

impl mlua::UserData for NnHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("arch", |_, this, ()| Ok(this.arch()));

        methods.add_method("variant", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.variant.clone()),
            NnHandle::TinyLlama(h) => Ok(h.variant.clone()),
            NnHandle::Llama(h) => Ok(h.variant.clone()),
        });
        methods.add_method("layers", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.layers),
            NnHandle::TinyLlama(h) => Ok(h.layers),
            NnHandle::Llama(h) => Ok(h.layers),
        });
        methods.add_method("heads", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.heads),
            NnHandle::TinyLlama(h) => Ok(h.heads),
            NnHandle::Llama(h) => Ok(h.heads),
        });
        // `kv_heads` — GPT-2 is MHA, so `kv_heads == heads`; the two
        // GQA variants (TinyLlama, Llama) carry the real value.
        // Returning `heads` for the GPT-2 arm keeps Lua callers from
        // having to arch-branch when they only want the KV group
        // count for a shape assertion.
        methods.add_method("kv_heads", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.heads),
            NnHandle::TinyLlama(h) => Ok(h.kv_heads),
            NnHandle::Llama(h) => Ok(h.kv_heads),
        });
        methods.add_method("dim", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.dim),
            NnHandle::TinyLlama(h) => Ok(h.dim),
            NnHandle::Llama(h) => Ok(h.dim),
        });
        methods.add_method("ctx", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.ctx),
            NnHandle::TinyLlama(h) => Ok(h.ctx),
            NnHandle::Llama(h) => Ok(h.ctx),
        });
        methods.add_method("vocab", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.vocab),
            NnHandle::TinyLlama(h) => Ok(h.vocab),
            NnHandle::Llama(h) => Ok(h.vocab),
        });
        methods.add_method("device", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.device.clone()),
            NnHandle::TinyLlama(h) => Ok(h.device.clone()),
            NnHandle::Llama(h) => Ok(h.device.clone()),
        });
        methods.add_method("dtype", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.dtype.clone()),
            NnHandle::TinyLlama(h) => Ok(h.dtype.clone()),
            NnHandle::Llama(h) => Ok(h.dtype.clone()),
        });
        // `pretrained` — Gpt2 / TinyLlama carry the flag; Llama
        // adapter is inference-only and always loads pretrained
        // weights, so `true` is the honest default.
        methods.add_method("pretrained", |_, this, ()| match this {
            NnHandle::Gpt2(h) => Ok(h.pretrained),
            NnHandle::TinyLlama(h) => Ok(h.pretrained),
            NnHandle::Llama(_) => Ok(true),
        });
        // `forward_shape` — trainable arches (Gpt2 / TinyLlama)
        // return `[batch, seq, vocab]`; the Llama adapter slices
        // the last-token logits and returns `[batch, vocab]`.
        // Shape difference is arch-visible; Lua callers that care
        // can consult `handle:arch()` first.
        methods.add_method(
            "forward_shape",
            |_, this, (batch, seq): (usize, usize)| match this {
                NnHandle::Gpt2(h) => Ok(vec![batch, seq, h.vocab]),
                NnHandle::TinyLlama(h) => Ok(vec![batch, seq, h.vocab]),
                NnHandle::Llama(h) => Ok(vec![batch, h.vocab]),
            },
        );
    }
}

#[cfg(test)]
mod arch_ops_tests {
    use super::*;
    use algocline_nn::card::SUPPORTED_ARCHITECTURE_FAMILIES;

    /// Every entry in [`ARCH_OPS`] must appear in
    /// [`SUPPORTED_ARCHITECTURE_FAMILIES`] — the bridge cannot
    /// register an arch the crate-side canonical list does not
    /// recognise. Layer 4b §Q4-A invariant #6.
    #[test]
    fn arch_ops_entries_are_all_canonical_families() {
        for (name, _) in ARCH_OPS {
            assert!(
                SUPPORTED_ARCHITECTURE_FAMILIES.contains(name),
                "ARCH_OPS entry {name:?} is not in SUPPORTED_ARCHITECTURE_FAMILIES {SUPPORTED_ARCHITECTURE_FAMILIES:?}"
            );
        }
    }

    /// `resolve_arch_ops` accepts both the bare family name
    /// (`"gpt2"`) and the `<family>-<variant>` form
    /// (`"gpt2-medium"`), matching
    /// [`algocline_nn::card::validate_architecture`]'s discipline.
    /// A longer identifier that merely shares a prefix
    /// (`"gpt2experimental"`) does NOT match — the namespace stays
    /// partitioned.
    #[test]
    fn resolve_arch_ops_matches_bare_and_variant_forms() {
        assert!(resolve_arch_ops("gpt2").is_some());
        assert!(resolve_arch_ops("gpt2-medium").is_some());
        assert!(resolve_arch_ops("tinyllama").is_some());
        assert!(resolve_arch_ops("tinyllama-1.1b").is_some());
        assert!(resolve_arch_ops("llama").is_some());
        assert!(resolve_arch_ops("llama-7b").is_some());
    }

    #[test]
    fn resolve_arch_ops_rejects_prefix_only_matches() {
        assert!(resolve_arch_ops("gpt2experimental").is_none());
        assert!(resolve_arch_ops("tinyllamafork").is_none());
    }

    #[test]
    fn resolve_arch_ops_rejects_unregistered_families() {
        // Declared in SUPPORTED_ARCHITECTURE_FAMILIES but no
        // bridge preset yet — these return None until per-arch
        // handlers land.
        assert!(resolve_arch_ops("qwen2").is_none());
        assert!(resolve_arch_ops("phi").is_none());
        assert!(resolve_arch_ops("gemma").is_none());
    }

    #[test]
    fn registered_arch_names_reports_gpt2_tinyllama_llama_today() {
        let names = registered_arch_names();
        assert_eq!(names, vec!["gpt2", "tinyllama", "llama"]);
    }
}

#[cfg(test)]
mod nn_handle_helper_tests {
    //! Layer 5a S1 — `NnHandle::arch_family_variant` /
    //! `is_lora_wrapped` unit coverage.
    //!
    //! `arch_family_variant` is tested via a synthetic handle built
    //! from a from-scratch `preset.gpt2("tiny", ...)` /
    //! `preset.tinyllama("tinyllama-tiny", ...)` path so no HF
    //! download is required. `is_lora_wrapped` returns `false` for
    //! all base handles built via `preset.*`; the wrap-side `true`
    //! path is covered by the Layer 5a integration test
    //! (`bridge_nn_card_merge_lora_test`) since it requires a live
    //! LoRA card + `load_wrap` round-trip.
    use super::*;
    use mlua::Lua;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn arch_family_variant_prepends_family_prefix_for_bare_variant() {
        let dir = tempdir();
        let lua = Lua::new();
        // Build a base Gpt2Handle via the bridge helper directly
        // (bypasses the Lua-facing preset registration; same output
        // shape).
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        let gpt2 = build_gpt2_handle("tiny", Some(&opts), dir.path()).expect("build gpt2");
        let handle = NnHandle::Gpt2(gpt2);
        assert_eq!(handle.arch_family_variant(), "gpt2-tiny");
        assert!(!handle.is_lora_wrapped());
    }

    #[test]
    fn arch_family_variant_passes_through_prefixed_variant() {
        // Simulate the `card.load_handle` path where `variant`
        // already carries the full "family-variant" string
        // (mmap-backed load uses `meta.architecture` verbatim).
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        let mut gpt2 = build_gpt2_handle("tiny", Some(&opts), dir.path()).expect("build gpt2");
        gpt2.variant = "gpt2-tiny".to_string();
        let handle = NnHandle::Gpt2(gpt2);
        assert_eq!(handle.arch_family_variant(), "gpt2-tiny");
    }

    #[test]
    fn arch_family_variant_prepends_tinyllama_prefix() {
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        let tll =
            build_tinyllama_handle("tinyllama-tiny", Some(&opts), dir.path()).expect("build tll");
        let handle = NnHandle::TinyLlama(tll);
        // `preset.tinyllama("tinyllama-tiny", ...)` stores the
        // already-prefixed string verbatim (`variant.to_string()`
        // in build_tinyllama_handle), so pass-through is expected.
        assert_eq!(handle.arch_family_variant(), "tinyllama-tiny");
        assert!(!handle.is_lora_wrapped());
    }

    #[test]
    fn is_lora_wrapped_returns_false_for_base_handles() {
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        let gpt2 = build_gpt2_handle("tiny", Some(&opts), dir.path()).expect("build gpt2");
        assert!(!NnHandle::Gpt2(gpt2).is_lora_wrapped());

        let tll =
            build_tinyllama_handle("tinyllama-tiny", Some(&opts), dir.path()).expect("build tll");
        assert!(!NnHandle::TinyLlama(tll).is_lora_wrapped());
    }

    #[test]
    fn is_lora_wrapped_returns_true_when_has_lora_flag_set() {
        // Direct field-level assertion — the wrap-time flow that
        // flips `has_lora` to `true` is exercised end-to-end by
        // the Layer 5a integration test; this unit test guards the
        // NnHandle::is_lora_wrapped dispatch itself.
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        let mut gpt2 = build_gpt2_handle("tiny", Some(&opts), dir.path()).expect("build gpt2");
        gpt2.has_lora = true;
        assert!(NnHandle::Gpt2(gpt2).is_lora_wrapped());

        let mut tll =
            build_tinyllama_handle("tinyllama-tiny", Some(&opts), dir.path()).expect("build tll");
        tll.has_lora = true;
        assert!(NnHandle::TinyLlama(tll).is_lora_wrapped());
    }
}

#[cfg(test)]
mod build_create_payload_from_meta_tests {
    //! Layer 5a S2 — `build_create_payload_from_meta` unit coverage.
    //!
    //! The envelope shape (`pkg.name` / `card_id` /
    //! `metadata.kind` / `metadata.nn`) must be byte-identical to
    //! what `build_create_payload` produces from the equivalent
    //! user-JSON input so `FileCardStore::create` treats both
    //! entry points uniformly.
    use super::*;
    use algocline_nn::card::{NnCandleBranch, NnCardMeta, NnLineage};
    use serde_json::json;

    fn sample_merged_meta() -> NnCardMeta {
        NnCardMeta {
            name: "my-merged".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "merged".into(),
            lineage: NnLineage {
                parent: Some("cards/lora-src-001".into()),
                ..NnLineage::default()
            },
            hyperparams: json!({}),
            metrics: json!({}),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/my-merged-1".into(),
                device: None,
                dtype: None,
                lora: None,
            }),
        }
    }

    #[test]
    fn envelope_shape_matches_build_create_payload() {
        let meta = sample_merged_meta();
        let payload = build_create_payload_from_meta("my-merged-1", &meta).expect("build payload");

        assert_eq!(payload["pkg"]["name"], NN_PKG);
        assert_eq!(payload["card_id"], "my-merged-1");
        assert_eq!(payload["metadata"]["kind"], "nn_model");

        let nn = &payload["metadata"]["nn"];
        assert_eq!(nn["name"], "my-merged");
        assert_eq!(nn["architecture"], "gpt2-medium");
        assert_eq!(nn["training_path"], "merged");
        assert_eq!(nn["lineage"]["parent"], "cards/lora-src-001");
        assert_eq!(nn["candle"]["bundle_ref"], "nn/my-merged-1");
    }

    #[test]
    fn refuses_unknown_architecture_family() {
        let mut meta = sample_merged_meta();
        meta.architecture = "nonexistent-arch".into();
        let err = build_create_payload_from_meta("my-merged-1", &meta)
            .expect_err("should reject unknown arch");
        assert!(
            err.to_string().contains("alc.nn.card.merge_lora"),
            "expected merge_lora-prefixed error, got: {err}"
        );
    }
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

/// Layer 4b S6 integration tests — arch-neutral preset + card
/// load dispatch. These target the arch-neutral entry
/// (`build_neutral_preset`, `load_handle_impl`, `load_wrap_impl`)
/// plus `NnHandle` dispatch, and the directional errors that keep
/// the two load surfaces (`load_handle` / `load_wrap`) from silently
/// swallowing each other's cards.
///
/// Scope note: end-to-end tests that need a physical bundle at
/// `<nn_dir>/<card_id>.safetensors` matching the auto-generated
/// card_id are deferred until a `FileCardStore::update` (or Lua
/// bridge `card.save`) test hook lands — the current
/// `FileCardStore::create` returns the id post-hoc, so a
/// pre-write bundle can't pin the correct filename. The tests
/// here exercise the dispatch decisions + error paths that
/// don't require this.
#[cfg(test)]
mod load_dispatch_tests {
    use super::*;
    use algocline_nn::arch::{Gpt2Config, Gpt2Model, LoraConfig};
    use candle_nn::{VarBuilder, VarMap};
    use mlua::Lua;
    use serde_json::json;

    fn write_test_card(store: &FileCardStore, nn_meta: serde_json::Value) -> String {
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "metadata": { "kind": "nn_model", "nn": nn_meta }
        });
        let (card_id, _path) = store.create(payload).expect("create test card");
        card_id
    }

    // ── arch-neutral preset dispatch ─────────────────────────

    #[test]
    fn neutral_preset_gpt2_returns_gpt2_nn_handle() {
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let opts_val = lua.to_value(&json!({ "pretrained": false })).unwrap();
        let opts_tbl = match opts_val {
            LuaValue::Table(t) => t,
            _ => unreachable!(),
        };
        let h = build_neutral_preset("gpt2", "tiny", Some(&opts_tbl), tmp.path())
            .expect("neutral preset gpt2 tiny");
        assert_eq!(h.arch(), "gpt2");
        assert!(h.as_gpt2().is_some());
        assert!(h.as_tinyllama().is_none());
    }

    #[test]
    fn neutral_preset_tinyllama_returns_tinyllama_nn_handle() {
        let tmp = tempfile::TempDir::new().unwrap();
        // pretrained=false to skip HF hub.
        let lua = Lua::new();
        let opts_val = lua.to_value(&json!({ "pretrained": false })).unwrap();
        let opts_tbl = match opts_val {
            LuaValue::Table(t) => t,
            _ => unreachable!(),
        };
        let h = build_neutral_preset("tinyllama", "tinyllama-tiny", Some(&opts_tbl), tmp.path())
            .expect("neutral preset tinyllama-tiny");
        assert_eq!(h.arch(), "tinyllama");
        assert!(h.as_tinyllama().is_some());
    }

    #[test]
    fn neutral_preset_rejects_unregistered_arch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let msg = match build_neutral_preset("qwen2", "1.5b", None, tmp.path()) {
            Ok(_) => panic!("qwen2 must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("qwen2") && msg.contains("not registered"),
            "message: {msg}"
        );
    }

    // ── load_handle_impl directional errors ──────────────────

    #[test]
    fn load_handle_refuses_lora_card_with_directional_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-lora",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 4, "alpha": 8, "base_bundle_ref": "nn/base",
                        "target_modules": ["q_proj"], "dropout": 0.0,
                        "delta_path": "/nonexistent/lora.safetensors"
                    }
                }
            }),
        );
        let msg = match load_handle_impl(&store, &card_id, &nn_dir) {
            Ok(_) => panic!("lora card must be refused by load_handle"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("training_path=\"lora\"") && msg.contains("load_wrap"),
            "directional error must point at load_wrap: {msg}"
        );
    }

    #[test]
    fn load_handle_refuses_unknown_training_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-bogus",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "quantized_awq",
                "candle": { "bundle_ref": "nn/placeholder" }
            }),
        );
        let msg = match load_handle_impl(&store, &card_id, &nn_dir) {
            Ok(_) => panic!("unknown training_path must error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("unknown training_path") || msg.contains("quantized_awq"),
            "message: {msg}"
        );
    }

    // ── load_wrap_impl directional errors + arch match ───────

    #[test]
    fn load_wrap_refuses_merged_card_with_directional_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-merged",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "merged",
                "candle": { "bundle_ref": "nn/placeholder" }
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).unwrap();
        let msg = match load_wrap_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("merged card must be refused by load_wrap"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("training_path=\"merged\"") && msg.contains("load_handle"),
            "directional error must point at load_handle: {msg}"
        );
    }

    #[test]
    fn load_wrap_refuses_full_ft_card_with_directional_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-fullft",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "full_ft",
                "candle": { "bundle_ref": "nn/placeholder" }
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).unwrap();
        let msg = match load_wrap_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("full_ft card must be refused by load_wrap"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("training_path=\"full_ft\"") && msg.contains("load_handle"),
            "message: {msg}"
        );
    }

    #[test]
    fn load_wrap_rejects_arch_mismatched_base_handle() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));

        // Build a valid LoRA delta on gpt2-tiny so precheck passes.
        let cfg = Gpt2Config::from_variant("tiny").unwrap();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();
        let lora_cfg = LoraConfig::new(4, 8.0);
        let lora_vm = model.wrap_lora(&lora_cfg).unwrap();
        let delta_path = tmp.path().join("gpt2-lora.safetensors");
        lora_vm.save(&delta_path).unwrap();

        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-lora",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 4, "alpha": 8, "base_bundle_ref": "nn/base-gpt2",
                        "target_modules": ["q_proj", "k_proj", "v_proj"],
                        "dropout": 0.0,
                        "delta_path": delta_path.to_str().unwrap()
                    }
                }
            }),
        );

        // Build a tinyllama base handle — arch mismatch.
        let tll_nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let opts_val = lua.to_value(&json!({ "pretrained": false })).unwrap();
        let opts_tbl = match opts_val {
            LuaValue::Table(t) => t,
            _ => unreachable!(),
        };
        let tll_handle =
            build_tinyllama_handle("tinyllama-tiny", Some(&opts_tbl), &tll_nn_dir).unwrap();
        let tll_ud = lua.create_userdata(tll_handle).unwrap();

        let msg = match load_wrap_impl(&store, &card_id, &tll_ud) {
            Ok(_) => panic!("arch mismatch must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("gpt2 card requires a gpt2 base handle") && msg.contains("tinyllama"),
            "arch mismatch message must name both sides: {msg}"
        );
    }

    // ── legacy shims still work ──────────────────────────────

    #[test]
    fn legacy_load_gpt2_shim_delegates_to_shared_core() {
        // The 3 existing trainer_tests::load_gpt2_impl_errors_*
        // tests exercise the pre-borrow schema-precheck path
        // (candle / lora / delta_path); this test guards that the
        // shim continues to route through the shared core after
        // the S5 refactor by asserting the same delta_path error
        // shape survives.
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
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
                        "rank": 8, "alpha": 16,
                        "base_bundle_ref": "nn/base-gpt2-medium"
                    }
                }
            }),
        );
        let lua = Lua::new();
        let placeholder = lua.create_any_userdata(0u32).unwrap();
        let msg = match load_gpt2_impl(&store, &card_id, &placeholder) {
            Ok(_) => panic!("missing delta_path must error via shim"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("delta_path"), "message: {msg}");
    }
}

#[cfg(test)]
mod merge_lora_bridge_tests {
    //! Layer 5a S4 — bridge integration tests for
    //! `alc.nn.card.merge_lora`.
    //!
    //! Exercised end-to-end from the LoRA card write → base handle
    //! build → `load_wrap_impl` → `merge_lora_impl` chain, then
    //! re-opens the merged card via `load_handle_impl` to verify
    //! shape + on-disk artifacts.
    //!
    //! All tests use the CPU/F32 `gpt2-tiny` / `tinyllama-tiny`
    //! micro shapes so they stay under `cargo test` friendly (no HF
    //! hub download, no >1s train step).
    use super::*;
    use algocline_nn::arch::{Gpt2Config, Gpt2Model, LoraConfig, TinyLlamaConfig, TinyLlamaModel};
    use candle_nn::{VarBuilder, VarMap};
    use mlua::Lua;
    use serde_json::json;

    fn write_test_card(store: &FileCardStore, nn_meta: serde_json::Value) -> String {
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "metadata": { "kind": "nn_model", "nn": nn_meta }
        });
        let (card_id, _path) = store.create(payload).expect("create test card");
        card_id
    }

    fn opts_table(lua: &Lua, v: serde_json::Value) -> LuaTable {
        let val = lua.to_value(&v).expect("to_value");
        match val {
            LuaValue::Table(t) => t,
            _ => unreachable!("json object must serialise to Lua table"),
        }
    }

    /// Build a `gpt2-tiny` base handle, wrap-and-save a LoRA delta,
    /// then write a `training_path=lora` Card pointing at the delta.
    /// Returns `(store, nn_dir, lora_card_id, lua)`; the caller keeps
    /// the returned tempdir alive for the duration of the test.
    fn setup_gpt2_lora_scaffold() -> (tempfile::TempDir, FileCardStore, PathBuf, String, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");

        let cfg = Gpt2Config::from_variant("tiny").unwrap();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();
        let lora_cfg = LoraConfig::new(4, 8.0);
        let lora_vm = model.wrap_lora(&lora_cfg).unwrap();
        let delta_path = tmp.path().join("gpt2-lora-delta.safetensors");
        lora_vm.save(&delta_path).unwrap();

        let lora_card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-lora-src",
                "backend": "candle",
                "architecture": "gpt2-tiny",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 4, "alpha": 8,
                        "base_bundle_ref": "nn/base-gpt2-tiny",
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up", "down"],
                        "dropout": 0.0,
                        "delta_path": delta_path.to_str().unwrap()
                    }
                }
            }),
        );

        let lua = Lua::new();
        (tmp, store, nn_dir, lora_card_id, lua)
    }

    fn setup_tinyllama_lora_scaffold() -> (tempfile::TempDir, FileCardStore, PathBuf, String, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");

        let cfg = TinyLlamaConfig::from_variant("tinyllama-tiny").unwrap();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();
        let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
        let lora_vm = model.wrap_lora(&lora_cfg).unwrap();
        let delta_path = tmp.path().join("tinyllama-lora-delta.safetensors");
        lora_vm.save(&delta_path).unwrap();

        let lora_card_id = write_test_card(
            &store,
            json!({
                "name": "tinyllama-lora-src",
                "backend": "candle",
                "architecture": "tinyllama-tiny",
                "training_path": "lora",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "lora": {
                        "rank": 4, "alpha": 8,
                        "base_bundle_ref": "nn/base-tinyllama-tiny",
                        "target_modules": TinyLlamaModel::default_lora_targets(),
                        "dropout": 0.0,
                        "delta_path": delta_path.to_str().unwrap()
                    }
                }
            }),
        );

        let lua = Lua::new();
        (tmp, store, nn_dir, lora_card_id, lua)
    }

    #[test]
    fn merge_lora_gpt2_happy_path_produces_merged_card_and_bundle() {
        let (_tmp, store, nn_dir, lora_card_id, lua) = setup_gpt2_lora_scaffold();

        // Build a matching base handle + wrap it.
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let gpt2_base = build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).unwrap();
        let gpt2_ud = lua.create_userdata(gpt2_base).unwrap();
        let wrapped_nn = load_wrap_impl(&store, &lora_card_id, &gpt2_ud).unwrap();
        assert!(wrapped_nn.is_lora_wrapped(), "wrap must set has_lora=true");
        let wrapped_ud = lua.create_userdata(wrapped_nn).unwrap();

        // Merge.
        let merge_opts = opts_table(
            &lua,
            json!({
                "name": "my-merged-gpt2",
                "lora_card": lora_card_id.clone(),
            }),
        );
        let merged_card_id =
            merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts).expect("merge_lora");

        // Verify: safetensors bundle exists on disk.
        let bundle_path = nn_dir.join(format!("{merged_card_id}.safetensors"));
        assert!(
            bundle_path.exists(),
            "merged safetensors must exist at {bundle_path:?}"
        );

        // Verify: Card metadata (training_path / architecture /
        // lineage / bundle_ref / name).
        let card = store.get(&merged_card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(nn.get("training_path").unwrap().as_str().unwrap(), "merged");
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "gpt2-tiny"
        );
        assert_eq!(
            nn.get("lineage")
                .and_then(|l| l.get("parent"))
                .and_then(|p| p.as_str())
                .unwrap(),
            lora_card_id
        );
        assert_eq!(
            nn.get("candle")
                .and_then(|c| c.get("bundle_ref"))
                .and_then(|b| b.as_str())
                .unwrap(),
            format!("nn/{merged_card_id}")
        );
        assert_eq!(nn.get("name").unwrap().as_str().unwrap(), "my-merged-gpt2");

        // Verify: the merged card is self-contained (loadable via
        // load_handle, no wrap needed).
        let merged_handle = load_handle_impl(&store, &merged_card_id, &nn_dir).unwrap();
        assert_eq!(merged_handle.arch(), "gpt2");
        assert!(!merged_handle.is_lora_wrapped());
    }

    #[test]
    fn merge_lora_tinyllama_happy_path_produces_merged_card_and_bundle() {
        let (_tmp, store, nn_dir, lora_card_id, lua) = setup_tinyllama_lora_scaffold();

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let tll_base = build_tinyllama_handle("tinyllama-tiny", Some(&base_opts), &nn_dir).unwrap();
        let tll_ud = lua.create_userdata(tll_base).unwrap();
        let wrapped_nn = load_wrap_impl(&store, &lora_card_id, &tll_ud).unwrap();
        assert!(wrapped_nn.is_lora_wrapped());
        let wrapped_ud = lua.create_userdata(wrapped_nn).unwrap();

        let merge_opts = opts_table(
            &lua,
            json!({
                "name": "my-merged-tinyllama",
                "lora_card": lora_card_id.clone(),
            }),
        );
        let merged_card_id =
            merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts).expect("merge_lora");

        let bundle_path = nn_dir.join(format!("{merged_card_id}.safetensors"));
        assert!(bundle_path.exists());

        let card = store.get(&merged_card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(nn.get("training_path").unwrap().as_str().unwrap(), "merged");
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "tinyllama-tiny"
        );

        // Self-contained load.
        let merged_handle = load_handle_impl(&store, &merged_card_id, &nn_dir).unwrap();
        assert_eq!(merged_handle.arch(), "tinyllama");
        assert!(!merged_handle.is_lora_wrapped());
    }

    #[test]
    fn merge_lora_refuses_unwrapped_base_handle_with_directional_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");

        let lua = Lua::new();
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let gpt2_base = build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).unwrap();
        // Do NOT wrap. Feed the base NnHandle directly.
        let base_nn = NnHandle::Gpt2(gpt2_base);
        let base_ud = lua.create_userdata(base_nn).unwrap();

        let merge_opts = opts_table(
            &lua,
            json!({ "name": "should-fail", "lora_card": "cards/whatever" }),
        );
        let msg = match merge_lora_impl(&store, &nn_dir, &base_ud, merge_opts) {
            Ok(id) => panic!("base handle must be refused; got merged card {id:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("is not LoRA-wrapped") && msg.contains("load_wrap"),
            "directional error must mention load_wrap: {msg}"
        );
    }

    #[test]
    fn merge_lora_refuses_missing_opts_name() {
        let (_tmp, store, nn_dir, lora_card_id, lua) = setup_gpt2_lora_scaffold();

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let gpt2_base = build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).unwrap();
        let gpt2_ud = lua.create_userdata(gpt2_base).unwrap();
        let wrapped_nn = load_wrap_impl(&store, &lora_card_id, &gpt2_ud).unwrap();
        let wrapped_ud = lua.create_userdata(wrapped_nn).unwrap();

        // opts.name absent.
        let merge_opts = opts_table(&lua, json!({ "lora_card": lora_card_id.clone() }));
        let msg = match merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts) {
            Ok(_) => panic!("missing name must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("opts.name must be a non-empty string"),
            "message: {msg}"
        );
    }

    #[test]
    fn merge_lora_refuses_missing_opts_lora_card() {
        let (_tmp, store, nn_dir, lora_card_id, lua) = setup_gpt2_lora_scaffold();

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let gpt2_base = build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).unwrap();
        let gpt2_ud = lua.create_userdata(gpt2_base).unwrap();
        let wrapped_nn = load_wrap_impl(&store, &lora_card_id, &gpt2_ud).unwrap();
        let wrapped_ud = lua.create_userdata(wrapped_nn).unwrap();

        let merge_opts = opts_table(&lua, json!({ "name": "no-lora-card" }));
        let msg = match merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts) {
            Ok(_) => panic!("missing lora_card must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("opts.lora_card must be a non-empty string"),
            "message: {msg}"
        );
    }

    #[test]
    fn merge_lora_refuses_empty_string_opts() {
        let (_tmp, store, nn_dir, lora_card_id, lua) = setup_gpt2_lora_scaffold();

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let gpt2_base = build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).unwrap();
        let gpt2_ud = lua.create_userdata(gpt2_base).unwrap();
        let wrapped_nn = load_wrap_impl(&store, &lora_card_id, &gpt2_ud).unwrap();
        let wrapped_ud = lua.create_userdata(wrapped_nn).unwrap();

        // Empty name.
        let merge_opts = opts_table(
            &lua,
            json!({ "name": "", "lora_card": lora_card_id.clone() }),
        );
        let msg = match merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts) {
            Ok(_) => panic!("empty name must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("opts.name must be a non-empty string"),
            "message: {msg}"
        );

        // Empty lora_card. Need a fresh wrapped_ud since the
        // previous userdata was consumed by borrow inside the
        // failing call (userdata is still owned by lua so we can
        // reuse it).
        let merge_opts_2 = opts_table(&lua, json!({ "name": "ok", "lora_card": "" }));
        let msg2 = match merge_lora_impl(&store, &nn_dir, &wrapped_ud, merge_opts_2) {
            Ok(_) => panic!("empty lora_card must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg2.contains("opts.lora_card must be a non-empty string"),
            "message: {msg2}"
        );
    }
}

#[cfg(test)]
mod loss_mask_from_card_tests {
    //! Metadata-gated loss-mask branch of `alc.nn.data.from_card`.
    //!
    //! Two layers, both network-free:
    //!
    //! 1. Unit tests for `loss_mask_decl_from_card` / `build_teacher_rows`
    //!    with an injected encoder (no tokenizer object at all).
    //! 2. Bridge tests that drive the real `from_card` closure through
    //!    `register_data_ns`. A hand-authored WordLevel tokenizer is
    //!    seeded at `<nn_dir>/tokenizers/gpt2.json` first, so
    //!    `HfTokenizer::load_cached` takes its cache-hit branch and
    //!    never reaches the HuggingFace hub.
    use super::*;
    use mlua::Lua;
    use serde_json::json;

    /// Deterministic whitespace-splitting encoder used by the unit
    /// tests: every known word maps to a fixed id, unknown words to 0.
    fn split_encode(text: &str) -> Result<Vec<u32>, String> {
        Ok(text
            .split_whitespace()
            .map(|w| match w {
                "alpha" => 1,
                "beta" => 2,
                "gamma" => 3,
                "delta" => 4,
                "epsilon" => 5,
                _ => 0,
            })
            .collect())
    }

    // ─── Unit: declaration parsing ────────────────────────────────

    #[test]
    fn loss_mask_decl_absent_is_none() {
        let no_metadata = json!({ "card_id": "c1" });
        assert_eq!(loss_mask_decl_from_card(&no_metadata), Ok(None));

        let empty_metadata = json!({ "card_id": "c1", "metadata": {} });
        assert_eq!(loss_mask_decl_from_card(&empty_metadata), Ok(None));

        let unrelated = json!({ "metadata": { "prior_card_id": "c0" } });
        assert_eq!(loss_mask_decl_from_card(&unrelated), Ok(None));
    }

    #[test]
    fn loss_mask_decl_response_is_parsed() {
        let card = json!({ "metadata": { "loss_mask": "response" } });
        assert_eq!(
            loss_mask_decl_from_card(&card),
            Ok(Some(LossMaskDecl::Response))
        );
    }

    #[test]
    fn loss_mask_decl_unknown_value_errors() {
        let wrong_string = json!({ "metadata": { "loss_mask": "prompt" } });
        let msg = loss_mask_decl_from_card(&wrong_string).expect_err("unknown value must error");
        assert!(msg.contains("metadata.loss_mask"), "message: {msg}");

        let wrong_type = json!({ "metadata": { "loss_mask": 42 } });
        let msg2 = loss_mask_decl_from_card(&wrong_type).expect_err("non-string must error");
        assert!(msg2.contains("metadata.loss_mask"), "message: {msg2}");
    }

    // ─── Unit: row + mask construction ────────────────────────────

    #[test]
    fn build_teacher_rows_zeroes_prompt_region() {
        let samples = vec![json!({ "prompt": "alpha beta", "response": "gamma delta" })];
        let rows = build_teacher_rows(&samples, split_encode).expect("rows");
        assert_eq!(rows.len(), 1);
        let (ids, mask) = &rows[0];
        assert_eq!(ids, &vec![1, 2, 3, 4]);
        assert_eq!(ids.len(), mask.len());
        assert_eq!(mask, &vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn build_teacher_rows_missing_prompt_errors() {
        let samples = vec![
            json!({ "prompt": "alpha", "response": "beta" }),
            json!({ "response": "gamma" }),
        ];
        let msg = build_teacher_rows(&samples, split_encode).expect_err("missing prompt errors");
        assert!(msg.contains("sample 1"), "message: {msg}");
        assert!(msg.contains("prompt"), "message: {msg}");
    }

    #[test]
    fn build_teacher_rows_missing_response_errors() {
        let samples = vec![
            json!({ "prompt": "alpha", "response": "beta" }),
            json!({ "prompt": "gamma", "response": "" }),
        ];
        let msg = build_teacher_rows(&samples, split_encode).expect_err("missing response errors");
        assert!(msg.contains("sample 1"), "message: {msg}");
        assert!(msg.contains("response"), "message: {msg}");
    }

    #[test]
    fn build_teacher_rows_prefix_mismatch_errors() {
        // Joint encoding does not start with the prompt encoding —
        // the boundary cannot be trusted, so the row must fail loudly
        // instead of being masked by guess.
        let encode = |text: &str| -> Result<Vec<u32>, String> {
            if text.contains('\n') {
                Ok(vec![7, 8, 9])
            } else {
                Ok(vec![1])
            }
        };
        let samples = vec![json!({ "prompt": "alpha", "response": "beta" })];
        let msg = build_teacher_rows(&samples, encode).expect_err("prefix mismatch must error");
        assert!(msg.contains("prompt token prefix"), "message: {msg}");
        assert!(msg.contains("sample 0"), "message: {msg}");
    }

    #[test]
    fn build_teacher_rows_empty_samples_is_ok() {
        let rows = build_teacher_rows(&[], split_encode).expect("empty rows");
        assert!(rows.is_empty());
    }

    // ─── Bridge: full `from_card` closure, no network ─────────────

    /// Hand-authored `tokenizers` fixture (WordLevel + WhitespaceSplit).
    /// Seeded on disk so `HfTokenizer::load_cached("gpt2", ..)` finds a
    /// cache hit and never calls the hub.
    const FIXTURE_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
 "added_tokens":[{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
 "normalizer":null,
 "pre_tokenizer":{"type":"WhitespaceSplit"},
 "post_processor":null,"decoder":null,
 "model":{"type":"WordLevel","vocab":{"[UNK]":0,"alpha":1,"beta":2,"gamma":3,"delta":4,"epsilon":5},"unk_token":"[UNK]"}}"#;

    fn opts_table(lua: &Lua, v: serde_json::Value) -> LuaTable {
        let val = lua.to_value(&v).expect("to_value");
        match val {
            LuaValue::Table(t) => t,
            _ => unreachable!("json object must serialise to Lua table"),
        }
    }

    /// Build a tempdir-backed store + nn_dir with the fixture tokenizer
    /// already on disk, create a Card (optionally declaring the mask)
    /// with one teacher-log sample, and return the registered Lua VM.
    fn setup_from_card(declare_mask: bool) -> (tempfile::TempDir, Lua, LuaTable, String) {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        std::fs::create_dir_all(nn_dir.join("tokenizers")).unwrap();
        std::fs::write(nn_dir.join("tokenizers/gpt2.json"), FIXTURE_TOKENIZER).unwrap();

        let store = Arc::new(FileCardStore::new(tmp.path().join("cards")));
        let mut metadata = serde_json::Map::new();
        metadata.insert("kind".to_string(), json!("teacher_log"));
        if declare_mask {
            metadata.insert("loss_mask".to_string(), json!("response"));
        }
        let (card_id, _path) = store
            .create(json!({
                "pkg": { "name": "alc_nn" },
                "metadata": Json::Object(metadata),
            }))
            .expect("create card");
        store
            .write_samples(
                &card_id,
                vec![json!({ "prompt": "alpha beta", "response": "gamma delta" })],
            )
            .expect("write samples");

        let lua = Lua::new();
        let nn_table = lua.create_table().unwrap();
        register_data_ns(&lua, &nn_table, Arc::clone(&store), nn_dir).expect("register data ns");
        (tmp, lua, nn_table, card_id)
    }

    fn call_from_card(lua: &Lua, nn_table: &LuaTable, card_id: &str) -> LuaAnyUserData {
        let data: LuaTable = nn_table.get("data").expect("data ns");
        let from_card: LuaFunction = data.get("from_card").expect("from_card entry");
        let opts = opts_table(
            lua,
            json!({ "batch_size": 1, "ctx_len": 8, "pad_id": 0, "tokenizer": "gpt2" }),
        );
        from_card
            .call::<LuaAnyUserData>((card_id.to_string(), opts))
            .expect("from_card call")
    }

    #[test]
    fn from_card_with_loss_mask_declaration_returns_masked_batch() {
        let (_tmp, lua, nn_table, card_id) = setup_from_card(true);
        let ud = call_from_card(&lua, &nn_table, &card_id);
        let handle = ud.borrow::<DatasetHandle>().expect("dataset handle");
        assert_eq!(handle.source, format!("card:{card_id}"));

        let mut ds = handle.inner_lock().expect("lock");
        let batch = ds
            .next_batch()
            .expect("next_batch")
            .expect("one batch available");
        assert_eq!(batch.input_ids, vec![vec![1, 2, 3, 4, 0, 0, 0, 0]]);
        let mask = batch.loss_mask.expect("declared card carries a loss mask");
        assert_eq!(mask, vec![vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]);
    }

    #[test]
    fn from_card_without_declaration_matches_legacy_ids() {
        let (_tmp_masked, lua_masked, nn_masked, masked_id) = setup_from_card(true);
        let masked_ud = call_from_card(&lua_masked, &nn_masked, &masked_id);
        let masked_ids = {
            let handle = masked_ud.borrow::<DatasetHandle>().expect("handle");
            let mut ds = handle.inner_lock().expect("lock");
            ds.next_batch()
                .expect("next_batch")
                .expect("batch")
                .input_ids
        };

        let (_tmp, lua, nn_table, card_id) = setup_from_card(false);
        let ud = call_from_card(&lua, &nn_table, &card_id);
        let handle = ud.borrow::<DatasetHandle>().expect("dataset handle");
        let mut ds = handle.inner_lock().expect("lock");
        let batch = ds.next_batch().expect("next_batch").expect("batch");

        // Invariant: identical token ids, mask-free legacy path.
        assert_eq!(batch.input_ids, masked_ids);
        assert!(batch.loss_mask.is_none());
    }

    #[test]
    fn from_card_unknown_declaration_errors_with_surface_prefix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        std::fs::create_dir_all(nn_dir.join("tokenizers")).unwrap();
        std::fs::write(nn_dir.join("tokenizers/gpt2.json"), FIXTURE_TOKENIZER).unwrap();

        let store = Arc::new(FileCardStore::new(tmp.path().join("cards")));
        let (card_id, _path) = store
            .create(json!({
                "pkg": { "name": "alc_nn" },
                "metadata": { "loss_mask": "resp" },
            }))
            .expect("create card");

        let lua = Lua::new();
        let nn_table = lua.create_table().unwrap();
        register_data_ns(&lua, &nn_table, Arc::clone(&store), nn_dir).expect("register data ns");

        let data: LuaTable = nn_table.get("data").expect("data ns");
        let from_card: LuaFunction = data.get("from_card").expect("from_card entry");
        let opts = opts_table(&lua, json!({ "batch_size": 1, "ctx_len": 8 }));
        let msg = match from_card.call::<LuaAnyUserData>((card_id, opts)) {
            Ok(_) => panic!("unknown declaration must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("alc.nn.data.from_card:"), "message: {msg}");
        assert!(msg.contains("metadata.loss_mask"), "message: {msg}");
    }
}
