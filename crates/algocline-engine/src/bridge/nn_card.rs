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
//! alc.nn.card.load_ckpt(path, spec)          -> NnHandle (Cardless)
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

use algocline_nn::arch::adapter::{
    InferenceAdapter, LlamaAdapter, LlamaAdapterConfig, LogitsShape,
};
use algocline_nn::arch::{
    Activation, Gpt2Config, Gpt2Custom, Gpt2Model, LoraConfig, MoeConfig, NormKind, NormPlacement,
    PosKind, ResidualKind, TinyLlamaConfig, TinyLlamaModel,
};
use algocline_nn::card::{
    bundle_ref_for, sanitize_stem, unique_stem, validate_training_path, CardId, NnCandleBranch,
    NnCardMeta, NnCustomBranch, NnLineage, NnLoraBranch, NnModelCard,
};
use algocline_nn::merged::{export_merged, MergeError, MergedProvenance};
use algocline_nn::tokenizer::HfTokenizer;
use algocline_nn::train::{
    run_distill, run_full_ft, run_lora_ft, Batch, CrossEntropyLoss, Dataset, DatasetOpts,
    DistillSpec, JsonlDataset, ParquetDataset, TeacherCardDataset, TokenizedDataset, TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use mlua::prelude::*;
use mlua::LuaSerdeExt;
use serde_json::Value as Json;

use super::nn_opts::{
    extract_distill_loss_kind, extract_full_ft_opts, extract_on_ckpt_hook, train_err_to_lua,
};
use crate::card::nn::persist;
use crate::card::{FileCardStore, SamplesQuery};

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
    // `alc.nn.tokenize` / `alc.nn.detokenize` — the token <-> text edge
    // of the generation path owned by `nn_gen.rs` (the rest of that
    // surface hangs off the Llama handle rather than the `alc.nn`
    // table).
    super::nn_gen::register_gen_ns(lua, &nn_table, nn_dir.clone())?;
    // `alc.nn.sampler.*` / `alc.nn.constraint.*` — the token-choosing
    // half of the same generation path (Sampler plan Layer 3).
    super::nn_sampler::register_sampler_ns(lua, &nn_table, nn_dir.clone())?;
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

    // Cardless sibling of `load_handle`: rebuild a handle from a raw
    // safetensors path plus a caller-supplied arch spec. The entry an
    // `alc.nn.trainer.*` `on_ckpt` hook uses to evaluate a mid-run
    // checkpoint, which has no Card to read the architecture from.
    // Captures nothing — no card store, no `nn_dir` — because the
    // caller names the file outright.
    let load_ckpt = lua.create_function(
        move |_lua, (path, spec): (String, LuaTable)| -> LuaResult<NnHandle> {
            load_ckpt_impl(&path, &spec)
        },
    )?;
    card_ns.set("load_ckpt", load_ckpt)?;

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

    // Pre-mint the id so bundle_ref = "nn/<card_id>" is known before
    // we build the Card (invariant #1) — the aggregate derives it
    // from this id at construction.
    let card_id = CardId::mint(name);

    // Delegate safetensors serialization to the existing alc.nn.save
    // path. Any store-side write failure propagates loudly through the
    // call chain (invariant #2).
    let nn_save: LuaFunction = alc_nn_fn(lua, "save")?;
    nn_save.call::<()>((vars, card_id.as_str().to_string()))?;

    // Parse the user-facing meta into the typed schema, then let the
    // aggregate constructor enforce the architecture / training_path
    // / bundle_ref invariants (invariant #1, now at build time).
    let nn_meta = build_nn_meta(&card_id, name, &meta_json)?;
    let card = NnModelCard::new(card_id, nn_meta)
        .map_err(|e| LuaError::external(format!("alc.nn.card.save: {e}")))?;

    // Envelope assembly + store write + returned-id coherence live in
    // `crate::card::nn::persist` — propagate errors loudly
    // (invariant #2).
    persist(store, &card).map_err(|e| LuaError::external(format!("alc.nn.card.save: {e}")))
}

/// Load the safetensors bundle referenced by a Card and return the
/// rehydrated Vars as a Lua table (keyed by the original save names).
///
/// Refuses partial state: a Card without a resolvable
/// `metadata.nn.candle.bundle_ref` or with a bundle-not-on-disk
/// surfaces as a Lua error (invariant #4).
fn load_impl(lua: &Lua, store: &FileCardStore, card_id: &str) -> LuaResult<LuaTable> {
    let card_id =
        CardId::parse(card_id).map_err(|e| LuaError::external(format!("alc.nn.card.load: {e}")))?;
    let card = store
        .get(card_id.as_str())
        .map_err(|e| LuaError::external(format!("alc.nn.card.load: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load: card '{card_id}' not found"))
        })?;

    // Extract bundle_ref and enforce the "nn/<card_id>" shape the
    // aggregate guarantees at save time. A mismatch means the Card
    // was hand-edited or built by a different pipeline — refuse
    // rather than guess the bundle.
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
    let expected = card_id.bundle_ref();
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
    let vars: LuaTable = nn_load.call(card_id.as_str().to_string())?;
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
    let card_id = CardId::parse(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_handle: {e}")))?;
    let card = store
        .get(card_id.as_str())
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
    // 誘導 (Layer 4b §Q3-A invariant #2)。 未知値の accepted list は
    // `validate_training_path` (crate::card) が単一 SoT。
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
            let msg = match validate_training_path(other) {
                Err(e) => e,
                // A value the schema accepts but this dispatcher has
                // no route for — a new SUPPORTED_TRAINING_PATHS entry
                // whose load arm has not landed yet.
                Ok(()) => format!("training_path {other:?} has no load_handle route"),
            };
            return Err(LuaError::external(format!(
                "alc.nn.card.load_handle: card '{card_id}': {msg}"
            )));
        }
    }

    // Enforce the bundle_ref = "nn/<card_id>" invariant that the
    // NnModelCard aggregate guarantees at save time.
    assert_bundle_ref_matches("alc.nn.card.load_handle", &card_id, &meta)?;

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

/// Error prefix for the `alc.nn.card.load_ckpt` surface.
const LOAD_CKPT_ERR_PREFIX: &str = "alc.nn.card.load_ckpt";

/// Spec keys that describe a custom GPT-2 shape. Presence of any one
/// of them switches [`load_ckpt_impl`] into synthesising a
/// [`NnCustomBranch`]; the list mirrors what
/// [`build_custom_gpt2_config`] reads, so a key added there without a
/// line here would be silently ignored by the load path.
const LOAD_CKPT_CUSTOM_KEYS: &[&str] = &[
    "layers",
    "heads",
    "dim",
    "ctx",
    "vocab",
    "act",
    "norm",
    "residual",
    "placement",
    "pos",
    "mlp_ratio",
    "kv_heads",
    "window",
    "untied_head",
    "moe",
];

/// Cardless self-contained loader — `alc.nn.card.load_ckpt(path, spec)`.
///
/// Rebuilds a handle straight from a raw safetensors file, with the
/// architecture supplied by the caller instead of read off a Card.
/// This is what lets an `on_ckpt` hook turn the `info.ckpt_path` it
/// receives mid-run into a handle the metric surfaces can consume:
/// mid-run checkpoints have no Card, so [`load_handle_impl`] cannot
/// reach them.
///
/// ```text
/// local handle = alc.nn.card.load_ckpt(info.ckpt_path, {
///     arch = "gpt2-tiny",   -- required; same vocabulary as load_handle
///     device = "cpu",       -- optional, default "cpu"
///     dtype = "f32",        -- optional, default "f32"
/// })
/// ```
///
/// A custom-shape run (`arch = "gpt2-custom"`) additionally passes the
/// shape keys it built the preset with (`vocab` / `ctx` / `layers` /
/// `heads` / `dim` / `act` / `norm` / `residual` / `placement` / `pos`
/// / `mlp_ratio` / `kv_heads` / `window` / `untied_head` / `moe`) —
/// the architecture string pins nothing for that variant, so the shape
/// has to come from the caller the same way it does on the build side.
///
/// # Checkpoint paths are volatile
///
/// `info.ckpt_path` names a rotating file: the trainer keeps only the
/// last `ckpt_keep` checkpoints and unlinks the rest. Loading it
/// **inside the hook body** is the only safe use — storing the path and
/// loading it after the run has moved on races the rotation, and the
/// failure mode is a missing file (or, worse, a different step's
/// weights at the same name). Callers who need a durable artifact
/// should let the run finish and load the Card it persists.
///
/// # Cost
///
/// The hook fires while the trainer holds the model mutex and the
/// dataset lock, so a `load_ckpt` from inside it constructs a *second*
/// full model while training is paused. That is free at `gpt2-tiny`
/// scale, but a large architecture combined with a small `ckpt_every`
/// pays for a whole model build per checkpoint out of the run's
/// wall-clock.
///
/// # Locking
///
/// This path never touches an existing handle's `Mutex` or `VarMap` —
/// it only builds a new model from the file. That is what makes it
/// callable from inside the hook at all: any route that re-read the
/// training handle's config (e.g. [`custom_branch_of_gpt2`]) would
/// deadlock against the mutex the trainer is already holding.
///
/// # Errors
///
/// Two prefixes surface here, by design:
///
/// - This function's own validation — missing / mistyped `spec.arch`,
///   a `path` that is not on disk, an architecture with no bridge
///   dispatch, and inference-only architectures (`llama`) whose
///   `build_from_safetensors` slot is `None` — carries
///   `alc.nn.card.load_ckpt:`.
/// - Everything raised while reading the bundle (unknown variant,
///   custom+MoE refusal, device / dtype parsing, shape mismatch)
///   keeps the `alc.nn.card.load:` prefix its shared implementation
///   already spells. Re-prefixing those would mean threading a context
///   argument through the `ArchOps` function-pointer signature, which
///   this iteration does not change.
fn load_ckpt_impl(path: &str, spec: &LuaTable) -> LuaResult<NnHandle> {
    let arch: String = spec
        .get::<Option<String>>("arch")
        .map_err(|_| {
            LuaError::external(format!(
                "{LOAD_CKPT_ERR_PREFIX}: spec.arch must be a string"
            ))
        })?
        .ok_or_else(|| {
            LuaError::external(format!(
                "{LOAD_CKPT_ERR_PREFIX}: spec.arch is required (e.g. 'gpt2-tiny'); \
                 it uses the same architecture vocabulary as alc.nn.card.load_handle"
            ))
        })?;

    // `device` / `dtype` default to the CPU / f32 pair every shipped
    // preset builds with, so a hook that only knows its arch stays a
    // one-liner. Their *values* are validated downstream by the shared
    // card-load path (see the error-prefix note above).
    let device = spec
        .get::<Option<String>>("device")
        .map_err(|_| {
            LuaError::external(format!(
                "{LOAD_CKPT_ERR_PREFIX}: spec.device must be a string"
            ))
        })?
        .unwrap_or_else(|| "cpu".to_string());
    let dtype = spec
        .get::<Option<String>>("dtype")
        .map_err(|_| {
            LuaError::external(format!(
                "{LOAD_CKPT_ERR_PREFIX}: spec.dtype must be a string"
            ))
        })?
        .unwrap_or_else(|| "f32".to_string());

    let path = std::path::Path::new(path);
    if !path.exists() {
        return Err(LuaError::external(format!(
            "{LOAD_CKPT_ERR_PREFIX}: no safetensors file at {path:?} \
             (mid-run checkpoints rotate — load info.ckpt_path inside the \
             on_ckpt hook body, not after the run)"
        )));
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| {
            LuaError::external(format!(
                "{LOAD_CKPT_ERR_PREFIX}: path {path:?} has no file stem to name the handle after"
            ))
        })?;

    // Synthesise the `NnCardMeta` the shared load core reads. Every
    // field is spelled out: the struct derives no `Default`, and an
    // implicit one would let a future field land here unnoticed.
    //
    // The candle branch is always present — leaving it `None` would
    // drop `device` / `dtype` back to the architecture default and
    // silently ignore what the caller asked for. `bundle_ref` is not
    // meaningful for a Cardless load (there is no `nn/<card_id>`
    // bundle), so it carries the `"ckpt/<stem>"` marker; the
    // `bundle_ref == "nn/<card_id>"` invariant belongs to the Card
    // surfaces and `assert_bundle_ref_matches` is deliberately not
    // called here.
    let meta = NnCardMeta {
        name: stem.clone(),
        backend: "candle".to_string(),
        task: None,
        architecture: arch.clone(),
        // Raw checkpoints are written by the full-fine-tune loop; the
        // LoRA path records a delta instead and reloads through
        // `load_wrap`.
        training_path: "full_ft".to_string(),
        lineage: NnLineage::default(),
        hyperparams: Json::Object(serde_json::Map::new()),
        metrics: Json::Object(serde_json::Map::new()),
        candle: Some(NnCandleBranch {
            bundle_ref: format!("ckpt/{stem}"),
            device: Some(device),
            dtype: Some(dtype),
            lora: None,
            custom: custom_branch_from_spec(spec)?,
        }),
    };

    let ops = resolve_arch_ops(&arch).ok_or_else(|| {
        LuaError::external(format!(
            "{LOAD_CKPT_ERR_PREFIX}: spec.arch {arch:?} has no bridge dispatch \
             (expected one of {})",
            registered_arch_names().join(" / ")
        ))
    })?;
    let build = ops.build_from_safetensors.ok_or_else(|| {
        LuaError::external(format!(
            "{LOAD_CKPT_ERR_PREFIX}: spec.arch {arch:?} does not support loading a \
             self-contained safetensors checkpoint (adapter-style architectures are \
             inference-only and never produce one)"
        ))
    })?;
    build(&meta, path)
}

/// Project the custom-shape keys of a `load_ckpt` spec onto a
/// [`NnCustomBranch`], or `None` when the spec names none of them.
///
/// `None` is the common case (`arch = "gpt2-tiny"` and friends): a
/// named variant rebuilds its shape from the architecture string alone,
/// and [`gpt2_config_for_card`] ignores the branch for those anyway.
/// Synthesising one unconditionally would therefore be dead weight for
/// every named arch and misleading metadata on the handle.
fn custom_branch_from_spec(spec: &LuaTable) -> LuaResult<Option<NnCustomBranch>> {
    let mut names_a_shape = false;
    for key in LOAD_CKPT_CUSTOM_KEYS {
        if spec.contains_key(*key)? {
            names_a_shape = true;
            break;
        }
    }
    if !names_a_shape {
        return Ok(None);
    }
    // Same parser the build side uses, so a spec that repeats the
    // `alc.nn.preset.gpt2("custom", ...)` opts verbatim reconstructs
    // the exact config that run was trained with.
    let cfg = build_custom_gpt2_config(LOAD_CKPT_ERR_PREFIX, Some(spec))?;
    Ok(NnCustomBranch::from_gpt2_config(&cfg))
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
    let card_id = CardId::parse(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: {e}")))?;
    let card = store
        .get(card_id.as_str())
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load_gpt2: card '{card_id}' not found"))
        })?;
    let meta = extract_nn_card_meta("alc.nn.card.load_gpt2", card_id.as_str(), &card)?;
    precheck_lora_card_meta("alc.nn.card.load_gpt2", card_id.as_str(), &meta)?;
    assert_bundle_ref_matches("alc.nn.card.load_gpt2", &card_id, &meta)?;
    let base = base_handle
        .borrow::<Gpt2Handle>()
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_gpt2: base handle: {e}")))?;
    wrap_gpt2_lora_from_meta("alc.nn.card.load_gpt2", card_id.as_str(), &meta, &base)
}

/// Enforce the `bundle_ref == "nn/<card_id>"` invariant on a loaded
/// card (design §5). Save paths cannot write a divergent shape any
/// more (the [`NnModelCard`] aggregate derives `bundle_ref` from the
/// id at construction), so a mismatch here means the Card was
/// hand-edited or written by a foreign pipeline — refuse rather than
/// guess. Shared by every card-load surface.
fn assert_bundle_ref_matches(ctx: &str, card_id: &CardId, meta: &NnCardMeta) -> LuaResult<()> {
    let bundle_ref = meta
        .candle
        .as_ref()
        .map(|c| c.bundle_ref.as_str())
        .ok_or_else(|| {
            LuaError::external(format!(
                "{ctx}: card '{card_id}' missing metadata.nn.candle"
            ))
        })?;
    let expected = card_id.bundle_ref();
    if bundle_ref != expected {
        return Err(LuaError::external(format!(
            "{ctx}: bundle_ref '{bundle_ref}' does not match card_id \
             '{card_id}' (expected '{expected}')"
        )));
    }
    Ok(())
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

    Ok(Gpt2Handle {
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

    Ok(Gpt2Handle {
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
fn build_nn_meta(card_id: &CardId, name: &str, user_meta: &Json) -> LuaResult<NnCardMeta> {
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
        bundle_ref: card_id.bundle_ref(),
        device: candle_in
            .and_then(|c| c.get("device"))
            .and_then(|v| v.as_str())
            .map(String::from),
        dtype: candle_in
            .and_then(|c| c.get("dtype"))
            .and_then(|v| v.as_str())
            .map(String::from),
        lora,
        // The custom-architecture branch is derived from the trained
        // config, not from caller-supplied metadata, so this save path
        // (which projects a user `meta` table) leaves it absent. The
        // trainer entry points record it through
        // `NnModelCard::from_training`.
        custom: None,
    };

    Ok(NnCardMeta {
        name: name.to_string(),
        backend: "candle".into(),
        task,
        architecture,
        training_path,
        lineage,
        hyperparams,
        metrics,
        candle: Some(candle),
    })
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
/// 5. Persists the Card via [`NnModelCard::from_merge`] +
///    [`crate::card::nn::persist`], which re-checks the
///    bundle_ref ↔ id coherence and asserts the store echoes the
///    pre-minted id back.
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

    // 3. Pre-mint merged card id + derive arch; bundle_ref is derived
    //    from the id (invariant #1 — same guarantee the aggregate
    //    re-checks at construction below).
    let merged_card_id = CardId::mint(&name);
    let arch = handle.arch_family_variant();
    let bundle_ref = merged_card_id.bundle_ref();

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
    let out_path = nn_dir.join(format!("{}.safetensors", merged_card_id.as_str()));

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

    // 5. Wrap into the typed aggregate — `from_merge` overrides the
    //    projection's default name (bundle file stem) with the
    //    caller-supplied one and re-checks the bundle_ref ↔ id
    //    coherence — then persist (envelope + returned-id check live
    //    in `crate::card::nn::persist`).
    let card = NnModelCard::from_merge(merged_card_id, name, meta)
        .map_err(|e| LuaError::external(format!("alc.nn.card.merge_lora: {e}")))?;

    persist(store, &card).map_err(|e| LuaError::external(format!("alc.nn.card.merge_lora: {e}")))
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

/// Arch-neutral description of a Lua-facing handle.
///
/// Every per-arch handle in this module (`Gpt2Handle` /
/// `TinyLlamaHandle` / `LlamaHandle`) answers `meta()` with this shape,
/// and [`NnHandle`] fans out to the wrapped handle's implementation.
/// The Lua-facing accessors then read fields off the returned value
/// instead of each one carrying its own three-arm `match`.
///
/// Sibling of `algocline_nn::arch::adapter::AdapterMeta`, which is the
/// crate-side equivalent an [`InferenceAdapter`] produces. The two are
/// deliberately separate types: this one carries the *Lua projection*
/// (`device` / `dtype` as the caller-supplied strings, plus the
/// bridge-only `pretrained` / `lora_wrapped` flags), while `AdapterMeta`
/// carries candle's own `Device` / `DType`. `build_llama_handle`
/// converts one into the other at construction time.
///
/// Borrows from the handle (`&'a str` rather than `String`) so an
/// accessor as cheap as `handle:layers()` does not allocate two strings
/// just to read a `usize`.
pub(super) struct HandleMeta<'a> {
    /// Architecture family prefix, matching an entry in
    /// [`algocline_nn::card::SUPPORTED_ARCHITECTURE_FAMILIES`].
    family: &'static str,
    /// Caller-facing variant id. May be the bare variant (`"medium"`)
    /// or the full `family-variant` form depending on the construction
    /// path — see [`HandleMeta::arch_family_variant`].
    variant: &'a str,
    layers: usize,
    heads: usize,
    /// Equal to `heads` for multi-head arches (GPT-2); the real
    /// grouped-query count for TinyLlama / Llama.
    kv_heads: usize,
    dim: usize,
    ctx: usize,
    vocab: usize,
    device: &'a str,
    dtype: &'a str,
    /// Whether the weights came from a pretrained bundle. The
    /// inference-only adapter arches are always `true` by construction.
    pretrained: bool,
    /// Whether the underlying model has been LoRA-wrapped. Always
    /// `false` for adapter arches, which do not support wrapping.
    lora_wrapped: bool,
    /// Caller-visible logits shape, which drives `forward_shape`.
    logits: LogitsShape,
}

impl HandleMeta<'_> {
    /// Dimensions a forward over `[batch, seq]` token ids produces.
    ///
    /// Trainable arches return `[batch, seq, vocab]`; the Llama adapter
    /// slices the final position and returns `[batch, vocab]`. The
    /// difference is carried by [`Self::logits`] rather than by an
    /// arch-specific branch here.
    fn forward_shape(&self, batch: usize, seq: usize) -> Vec<usize> {
        self.logits.dims(batch, seq, self.vocab)
    }

    /// Full `family-variant` identifier suitable for
    /// [`algocline_nn::MergedProvenance::arch`] and the projected
    /// [`algocline_nn::NnCardMeta::architecture`] field.
    ///
    /// The underlying handle's `variant` is stored in one of two
    /// conventions depending on the construction path:
    ///
    /// - `preset.<arch>(variant, ...)` stores the raw `variant` string
    ///   (e.g. `"medium"` / `"1.1b"`).
    /// - `card.load_handle` (from a saved card) stores the full
    ///   `family-variant` form (e.g. `"gpt2-medium"`) because the card's
    ///   `NnCardMeta.architecture` is already in that shape.
    ///
    /// This normalises both into the full form so the Layer 5a
    /// `merge_lora` bridge hands a consistent value to
    /// [`algocline_nn::MergedProvenance`] regardless of how the handle
    /// was obtained. The prefix-strip guard matches
    /// [`wrap_gpt2_lora_from_meta`]'s `base_cfg_id` logic (§Layer 4b
    /// invariant carry).
    fn arch_family_variant(&self) -> String {
        let prefix = format!("{}-", self.family);
        if self.variant.starts_with(&prefix) {
            self.variant.to_string()
        } else {
            format!("{prefix}{}", self.variant)
        }
    }
}

/// Register the shared Lua-facing accessors every `alc.nn` handle
/// exposes, reading each value off [`HandleMeta`].
///
/// Called from `impl UserData` for each per-arch handle and for
/// [`NnHandle`], so the accessor surface stays identical across all of
/// them by construction — adding an accessor here reaches every handle
/// at once. Previously each handle spelled out its own nine to eleven
/// `add_method` closures, and `NnHandle` additionally carried a
/// three-arm `match` inside every one of them.
fn add_meta_methods<T, M>(methods: &mut M, meta: fn(&T) -> HandleMeta<'_>)
where
    // `T: 'static` is what `mlua::UserData` already requires of every
    // handle type registered here; naming it explicitly lets the
    // borrow checker see that a `HandleMeta<'_>` borrowed from `this`
    // never outlives the method call.
    T: 'static,
    M: mlua::UserDataMethods<T>,
{
    methods.add_method("variant", move |_, this, ()| {
        Ok(meta(this).variant.to_string())
    });
    methods.add_method("layers", move |_, this, ()| Ok(meta(this).layers));
    methods.add_method("heads", move |_, this, ()| Ok(meta(this).heads));
    methods.add_method("kv_heads", move |_, this, ()| Ok(meta(this).kv_heads));
    methods.add_method("dim", move |_, this, ()| Ok(meta(this).dim));
    methods.add_method("ctx", move |_, this, ()| Ok(meta(this).ctx));
    methods.add_method("vocab", move |_, this, ()| Ok(meta(this).vocab));
    methods.add_method("device", move |_, this, ()| {
        Ok(meta(this).device.to_string())
    });
    methods.add_method("dtype", move |_, this, ()| Ok(meta(this).dtype.to_string()));
    methods.add_method("pretrained", move |_, this, ()| Ok(meta(this).pretrained));
    methods.add_method(
        "forward_shape",
        move |_, this, (batch, seq): (usize, usize)| Ok(meta(this).forward_shape(batch, seq)),
    );
}

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
    /// KV head count for GQA-aware accessors.
    ///
    /// MHA — the reference and every named preset — stores
    /// `kv_heads == heads`. A `custom = Some(spec)` build that opts
    /// into GQA (`spec.kv_heads = Some(k)`) stores the explicit `k`.
    /// This is populated at construction time from
    /// [`Gpt2Config::effective_kv_heads`] and read back by
    /// [`Self::meta`]; the two sides must stay in lockstep — mirroring
    /// `heads` in the accessor here (instead of storing the config's
    /// value) silently misreports GQA models to Lua callers even
    /// though the forward pass uses the correct value.
    kv_heads: usize,
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
        add_meta_methods(methods, Gpt2Handle::meta);
        // `handle:generate_session(prompt)` — stateless full-history
        // backend (no KV cache on the trainable arch); see nn_gen's
        // module doc §"Sessions over trainable arches".
        super::nn_gen::add_gpt2_generate_session_method(methods);
    }
}

impl Gpt2Handle {
    /// Arch-neutral projection of this handle. See [`HandleMeta`].
    ///
    /// `kv_heads` reports the value the handle was built with
    /// ([`Gpt2Config::effective_kv_heads`]): MHA (the reference and
    /// every named preset) reports `heads`; a `custom` build that opts
    /// into GQA (`custom.kv_heads = Some(k)`) reports the explicit `k`.
    /// The accessor previously mirrored `self.heads` unconditionally,
    /// which silently misreported GQA custom models to Lua callers
    /// even though the internal forward pass used the correct value.
    fn meta(&self) -> HandleMeta<'_> {
        HandleMeta {
            family: "gpt2",
            variant: &self.variant,
            layers: self.layers,
            heads: self.heads,
            kv_heads: self.kv_heads,
            dim: self.dim,
            ctx: self.ctx,
            vocab: self.vocab,
            device: &self.device,
            dtype: &self.dtype,
            pretrained: self.pretrained,
            lora_wrapped: self.has_lora,
            logits: LogitsShape::FullSeq,
        }
    }

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

    /// Vocabulary size, for sibling modules (`nn_gen`) that bound-check
    /// token ids without borrowing a full [`HandleMeta`].
    pub(super) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Context window, for sibling modules (`nn_gen`) that cap the
    /// stateless session history.
    pub(super) fn ctx(&self) -> usize {
        self.ctx
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

/// Rebuild the [`Gpt2Config`] a card's bundle was written under.
///
/// Two shapes of card reach here:
///
/// - **Named variant** (`"gpt2-medium"` / `"gpt2-tiny"` / ...) —
///   [`Gpt2Config::from_variant`] is the single source of the shape,
///   which is why those cards record no shape block.
/// - **Custom variant** (`"gpt2-custom"`, written by
///   `alc.nn.preset.gpt2("custom", ...)` plus a
///   [`super::nn_trainer`] entry point) — the architecture string pins
///   nothing, so the shape comes from `meta.candle.custom`
///   ([`NnCustomBranch`], recorded by
///   [`custom_branch_of_gpt2`]).
///
/// `device` / `dtype` are deliberately left at the base config's
/// values: they are load-time choices that
/// [`apply_candle_branch_device_dtype`] layers on top from
/// `meta.candle`.
///
/// # Errors
///
/// - The `custom` branch carries an MoE block. The bundle's per-block
///   expert Vars have no load path ([`Gpt2Model::from_safetensors_file`]
///   refuses MoE configs), and rebuilding the config *without* `moe`
///   would hand back a plain dense-MLP model under the card's name —
///   a silent architecture swap. Refuse loudly instead.
/// - The architecture is not a known variant and there is no `custom`
///   branch to fall back on (an unknown variant, or a custom-variant
///   card written before the shape block existed).
fn gpt2_config_for_card(meta: &NnCardMeta) -> LuaResult<Gpt2Config> {
    // `Gpt2Config::from_variant` accepts both bare ("medium") and
    // "gpt2-medium" forms — pass the card's architecture string
    // directly.
    if let Some(cfg) = Gpt2Config::from_variant(&meta.architecture) {
        return Ok(cfg);
    }

    let Some(branch) = meta.candle.as_ref().and_then(|c| c.custom.as_ref()) else {
        return Err(LuaError::external(format!(
            "alc.nn.card.load: unknown gpt2 variant {:?} on card {:?}; if this is a \
             custom-variant card it predates custom-shape metadata \
             (metadata.nn.candle.custom is absent, so the trained shape cannot be \
             recovered — retrain with a current build to make it reloadable)",
            meta.architecture, meta.name
        )));
    };

    if branch.moe.is_some() {
        return Err(LuaError::external(format!(
            "alc.nn.card.load: card {:?} is a custom+MoE model; MoE safetensors reload \
             is not supported yet (the bundle's per-block expert Vars have no load \
             path), and loading it as a dense model would silently change the \
             architecture — keep using the handle from the session that trained it",
            meta.name
        )));
    }

    Ok(Gpt2Config {
        vocab: branch.vocab,
        ctx: branch.ctx,
        layers: branch.layers,
        heads: branch.heads,
        dim: branch.dim,
        custom: Some(branch.spec.clone()),
        // Refused above; restated so a future MoE load path has to
        // revisit this arm rather than inheriting a stale `None`.
        moe: None,
        // `eps` is not reachable from the Lua `custom` opts table, so
        // every custom config was built on the `tiny` base (see
        // `build_custom_gpt2_config`). Spreading that base keeps the
        // two sides sharing one epsilon instead of a literal here.
        ..Gpt2Config::tiny()
    })
}

fn gpt2_from_safetensors(meta: &NnCardMeta, path: &std::path::Path) -> LuaResult<NnHandle> {
    let mut cfg = gpt2_config_for_card(meta)?;
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
        // Recovers `custom.kv_heads` for GQA cards; MHA cards keep
        // `kv_heads == heads`. Uses the same SoT the build path does
        // so a train → save → reload round-trip preserves the GQA
        // shape all the way through to the Lua accessor — the
        // previous mirror-of-`heads` in `Gpt2Handle::meta` silently
        // hid the reloaded value from Lua callers.
        kv_heads: cfg.effective_kv_heads(),
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
    let card_id = CardId::parse(card_id)
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_wrap: {e}")))?;
    let card = store
        .get(card_id.as_str())
        .map_err(|e| LuaError::external(format!("alc.nn.card.load_wrap: {e}")))?
        .ok_or_else(|| {
            LuaError::external(format!("alc.nn.card.load_wrap: card '{card_id}' not found"))
        })?;
    let mut meta = extract_nn_card_meta("alc.nn.card.load_wrap", card_id.as_str(), &card)?;
    // Overwrite meta.name with card_id so downstream error
    // messages from the wrap core reference the caller-visible
    // id (the meta.name field is user-set at save time and can
    // diverge from card_id).
    meta.name = card_id.as_str().to_string();

    // training_path 分岐: lora のみ受け付ける。 self-contained
    // (full_ft / merged / distillation) は load_handle 側に誘導。
    // 未知値の accepted list は `validate_training_path`
    // (crate::card) が単一 SoT。
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
            let msg = match validate_training_path(other) {
                Err(e) => e,
                Ok(()) => format!("training_path {other:?} has no load_wrap route"),
            };
            return Err(LuaError::external(format!(
                "alc.nn.card.load_wrap: card '{card_id}': {msg}"
            )));
        }
    }

    // Schema + delta-file precheck BEFORE inspecting base_handle so
    // schema gaps surface as specific errors (parity with the
    // trainer_tests::load_gpt2_impl_errors_* discipline that also
    // guards `load_gpt2_impl`).
    precheck_lora_card_meta("alc.nn.card.load_wrap", card_id.as_str(), &meta)?;

    // bundle_ref ↔ id coherence (design §5) — previously unchecked on
    // this surface; the write side can no longer produce a divergent
    // shape, so a mismatch means hand-editing.
    assert_bundle_ref_matches("alc.nn.card.load_wrap", &card_id, &meta)?;

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
    /// Caller-visible logits shape, copied from the adapter's
    /// [`InferenceAdapter::meta`] at construction. Carried as a field
    /// rather than hard-coded in [`Self::meta`] so a future adapter
    /// arch with a different convention drops in without editing this
    /// handle.
    logits: LogitsShape,
}

impl mlua::UserData for LlamaHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        add_meta_methods(methods, LlamaHandle::meta);
        // Inference-only extra: `handle:generate_session(prompt)`. The
        // adapter's own `forward` stays unbound — see `nn_gen.rs` for
        // why a session (with its own KV cache) is the only decode
        // entry point exposed to Lua.
        super::nn_gen::add_generate_session_method(methods);
    }
}

impl LlamaHandle {
    /// Arch-neutral projection of this handle. See [`HandleMeta`].
    ///
    /// `pretrained` is `true` unconditionally: the adapter path is
    /// inference-only and its weights always come from a bundle the
    /// caller supplied (the random-init `tiny` smoke build exists only
    /// to assert shapes and never reaches a Card). `lora_wrapped` is
    /// `false` unconditionally: the adapter path does not support LoRA
    /// wrapping (§Layer 2 non-goal, carried through Layer 4b).
    ///
    /// The shape numbers come from the adapter's
    /// [`InferenceAdapter::meta`] at construction time, so this handle
    /// never re-reads the upstream `candle_transformers` config.
    fn meta(&self) -> HandleMeta<'_> {
        HandleMeta {
            family: "llama",
            variant: &self.variant,
            layers: self.layers,
            heads: self.heads,
            kv_heads: self.kv_heads,
            dim: self.dim,
            ctx: self.ctx,
            vocab: self.vocab,
            device: &self.device,
            dtype: &self.dtype,
            pretrained: true,
            lora_wrapped: false,
            logits: self.logits,
        }
    }

    /// Shared handle to the underlying adapter, for callers who want
    /// to drive `forward` from Rust-side helper code.
    ///
    /// Consumed by `nn_gen::add_generate_session_method`, which clones
    /// the `Arc` into each generation session: the weights are shared,
    /// the KV cache is not.
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
        add_meta_methods(methods, TinyLlamaHandle::meta);
        // Stateless-session mirror of the Gpt2Handle registration.
        super::nn_gen::add_tinyllama_generate_session_method(methods);
    }
}

impl TinyLlamaHandle {
    /// Arch-neutral projection of this handle. See [`HandleMeta`].
    fn meta(&self) -> HandleMeta<'_> {
        HandleMeta {
            family: "tinyllama",
            variant: &self.variant,
            layers: self.layers,
            heads: self.heads,
            kv_heads: self.kv_heads,
            dim: self.dim,
            ctx: self.ctx,
            vocab: self.vocab,
            device: &self.device,
            dtype: &self.dtype,
            pretrained: self.pretrained,
            lora_wrapped: self.has_lora,
            logits: LogitsShape::FullSeq,
        }
    }

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

    /// Vocabulary size; mirrors [`Gpt2Handle::vocab`].
    pub(super) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Context window; mirrors [`Gpt2Handle::ctx`].
    pub(super) fn ctx(&self) -> usize {
        self.ctx
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

    let adapter = if let Some(paths) = weights_paths {
        LlamaAdapter::from_safetensors_files(&paths, cfg)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.llama: {e}")))?
    } else {
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg)
            .map_err(|e| LuaError::external(format!("alc.nn.preset.llama: {e}")))?;
        // Drop `vm` on purpose: the adapter is inference-only and
        // never expects a trainable VarMap; the surrounding scope
        // guarantees the mmap-less handle stays valid because the
        // adapter now owns the tensor snapshots.
        drop(vm);
        adapter
    };

    // Shape parameters come from the adapter's own `InferenceAdapter`
    // projection rather than from a clone of the config we just moved
    // into it. That keeps this bridge from re-reading
    // `candle_transformers`' `Config` field names, so a future adapter
    // arch needs no new field-by-field transcription here.
    let meta = InferenceAdapter::meta(&adapter);

    Ok(LlamaHandle {
        inner: Arc::new(adapter),
        // The caller-supplied `variant` is kept verbatim rather than
        // taking `meta.variant`: `LlamaAdapterConfig::from_variant`
        // canonicalises aliases (`"tiny"` becomes `"llama-tiny"`), and
        // the Lua-visible `handle:variant()` has always echoed back
        // exactly what the caller passed.
        variant: variant.to_string(),
        layers: meta.layers,
        heads: meta.heads,
        kv_heads: meta.kv_heads,
        dim: meta.dim,
        ctx: meta.ctx,
        vocab: meta.vocab,
        device: device_str,
        dtype: dtype_str,
        logits: meta.logits,
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
        build_custom_gpt2_config(GPT2_CUSTOM_PRESET_ERR_PREFIX, opts)?
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
        // SoT for both the internal `Block` builder and this Lua-facing
        // handle field — see `Gpt2Config::effective_kv_heads` for the
        // MHA / GQA resolution rule.
        kv_heads: cfg.effective_kv_heads(),
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
/// Error prefix for the custom-shape parser when it serves the
/// `alc.nn.preset.gpt2("custom", ...)` build side.
const GPT2_CUSTOM_PRESET_ERR_PREFIX: &str = "alc.nn.preset.gpt2('custom')";

fn custom_opt<T: mlua::FromLua>(
    ctx: &str,
    t: &LuaTable,
    key: &str,
    expected: &str,
) -> LuaResult<Option<T>> {
    t.get::<Option<T>>(key)
        .map_err(|_| LuaError::external(format!("{ctx}: option '{key}' must be {expected}")))
}

fn custom_bad_value(ctx: &str, key: &str, got: &str, expected: &str) -> LuaError {
    LuaError::external(format!(
        "{ctx}: unknown {key} '{got}' (expected {expected})"
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
///
/// `ctx` is the caller-facing tag every option error is prefixed with,
/// so the same parser can serve `alc.nn.preset.gpt2('custom')` (build
/// side) and `alc.nn.card.load_ckpt` (raw-checkpoint load side) without
/// pointing a load-time typo at the preset entry.
fn build_custom_gpt2_config(ctx: &str, opts: Option<&LuaTable>) -> LuaResult<Gpt2Config> {
    let mut cfg = Gpt2Config::tiny();
    let mut spec = Gpt2Custom::default();
    let Some(t) = opts else {
        cfg.custom = Some(spec);
        return Ok(cfg);
    };

    if let Some(v) = custom_opt::<usize>(ctx, t, "layers", "an integer")? {
        cfg.layers = v;
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "heads", "an integer")? {
        cfg.heads = v;
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "dim", "an integer")? {
        cfg.dim = v;
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "ctx", "an integer")? {
        cfg.ctx = v;
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "vocab", "an integer")? {
        cfg.vocab = v;
    }

    if let Some(s) = custom_opt::<String>(ctx, t, "act", "a string")? {
        spec.act = match s.as_str() {
            "gelu" => Activation::Gelu,
            "relu" => Activation::Relu,
            "silu" => Activation::Silu,
            "swiglu" => Activation::SwiGlu,
            "geglu" => Activation::GeGlu,
            other => {
                return Err(custom_bad_value(
                    ctx,
                    "act",
                    other,
                    "'gelu' / 'relu' / 'silu' / 'swiglu' / 'geglu'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(ctx, t, "norm", "a string")? {
        spec.norm = match s.as_str() {
            "layernorm" => NormKind::LayerNorm,
            "rmsnorm" => NormKind::RmsNorm,
            other => {
                return Err(custom_bad_value(
                    ctx,
                    "norm",
                    other,
                    "'layernorm' / 'rmsnorm'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(ctx, t, "residual", "a string")? {
        spec.residual = match s.as_str() {
            "sequential" => ResidualKind::Sequential,
            "parallel" => ResidualKind::Parallel,
            other => {
                return Err(custom_bad_value(
                    ctx,
                    "residual",
                    other,
                    "'sequential' / 'parallel'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(ctx, t, "placement", "a string")? {
        spec.placement = match s.as_str() {
            "preln" => NormPlacement::PreLn,
            "postln" => NormPlacement::PostLn,
            other => {
                return Err(custom_bad_value(
                    ctx,
                    "placement",
                    other,
                    "'preln' / 'postln'",
                ))
            }
        };
    }
    if let Some(s) = custom_opt::<String>(ctx, t, "pos", "a string")? {
        spec.pos = match s.as_str() {
            "learned" => PosKind::Learned,
            "rope" => PosKind::Rope,
            "alibi" => PosKind::Alibi,
            "nope" => PosKind::NoPos,
            other => {
                return Err(custom_bad_value(
                    ctx,
                    "pos",
                    other,
                    "'learned' / 'rope' / 'alibi' / 'nope'",
                ))
            }
        };
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "mlp_ratio", "an integer")? {
        spec.mlp_ratio = v;
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "kv_heads", "an integer")? {
        spec.kv_heads = Some(v);
    }
    if let Some(v) = custom_opt::<usize>(ctx, t, "window", "an integer")? {
        spec.window = Some(v);
    }
    if let Some(b) = custom_opt::<bool>(ctx, t, "untied_head", "a boolean")? {
        spec.untied_head = b;
    }

    cfg.moe = parse_custom_moe(ctx, t)?;
    cfg.custom = Some(spec);
    Ok(cfg)
}

/// Parse the optional nested `moe = { n_experts, top_k?, alpha? }`
/// table. Defaults mirror [`MoeConfig::new`] (Mixtral top-2 routing,
/// Switch α = 0.01); `MoeConfig::validate` runs at build time in
/// `Gpt2Model::new`.
fn parse_custom_moe(ctx: &str, t: &LuaTable) -> LuaResult<Option<MoeConfig>> {
    let Some(m) = custom_opt::<LuaTable>(ctx, t, "moe", "a table")? else {
        return Ok(None);
    };
    let n_experts = custom_opt::<usize>(ctx, &m, "n_experts", "an integer")?.ok_or_else(|| {
        LuaError::external(format!("{ctx}: moe.n_experts is required (integer ≥ 1)"))
    })?;
    let mut moe = MoeConfig::new(n_experts);
    if let Some(k) = custom_opt::<usize>(ctx, &m, "top_k", "an integer")? {
        moe.top_k = k;
    }
    if let Some(a) = custom_opt::<f64>(ctx, &m, "alpha", "a number")? {
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

/// Reject f16 base handles at the trainer entrypoints.
///
/// Sibling to [`guard_device_dtype_matrix`] — that guard rejects bf16
/// at preset build time on non-CUDA devices; this guard rejects f16
/// at trainer entry regardless of device. F32 trains through the
/// stock AdamW and BF16 through the FP32-master `MixedAdamW`
/// (`algocline_nn::train::MixedAdamW`, design §7.1); F16 has no path
/// because its 5-bit exponent needs loss scaling that does not ship,
/// and accepting it would train into silent gradient underflow.
/// Called from the four L5b/L5c trainer impl fns (`wrap_lora_impl` /
/// `run_lora_ft_impl` / `run_full_ft_impl` / `run_distill_impl`)
/// between the Llama refusal (step 4) and the opts / dataset
/// processing (step 5), so an f16 base surfaces a directional Lua
/// error up front rather than the trainer-core refusal deep inside
/// the run.
pub(super) fn guard_base_dtype_for_training(fn_name: &str, handle: &NnHandle) -> LuaResult<()> {
    let dtype = match handle {
        NnHandle::Gpt2(h) => h.dtype.as_str(),
        NnHandle::TinyLlama(h) => h.dtype.as_str(),
        NnHandle::Llama(h) => h.dtype.as_str(),
    };
    if dtype.eq_ignore_ascii_case("f16") {
        return Err(LuaError::external(format!(
            "{fn_name}: training does not support an f16 base (f16 needs \
             loss scaling, which is not implemented); build the preset \
             with dtype=\"bf16\" (CUDA) or dtype=\"f32\""
        )));
    }
    Ok(())
}

/// Project the handle's architecture-customization spec onto a Card
/// branch, for the trainer entry points that write a Card
/// (`run_lora_ft` / `run_full_ft` / `run_distill` in
/// [`super::nn_trainer`]).
///
/// Returns `None` — meaning "the Card needs no shape block" — for
/// every handle whose config is the GPT-2 reference: the two
/// non-GPT-2 arches (whose shape is pinned by their own variant
/// presets) and a GPT-2 handle built from a named preset. Only a
/// `preset.gpt2("custom", ...)` handle yields `Some`, because
/// `architecture = "gpt2-custom"` does not pin a shape and the load
/// path has nothing else to rebuild the config from.
///
/// # Lock discipline
///
/// The [`Gpt2Config`] is the only thing behind the model mutex that
/// this reads, and [`NnCustomBranch::from_gpt2_config`] copies out of
/// it, so the guard is released before returning. Callers invoke this
/// *before* taking the training lock (alongside
/// [`NnHandle::arch_family_variant`]) so the short read never nests
/// inside the training critical section.
///
/// # Errors
///
/// A poisoned model mutex propagates as a loud [`LuaError::external`]
/// under the caller's `fn_name` prefix — a previous panic while the
/// model was locked means the shape on the Card cannot be trusted, so
/// the run refuses rather than writing a Card with the branch absent
/// (which the load path would read as "reference architecture").
pub(super) fn custom_branch_of_gpt2(
    fn_name: &str,
    handle: &NnHandle,
) -> LuaResult<Option<NnCustomBranch>> {
    let Some(gpt2) = handle.as_gpt2() else {
        return Ok(None);
    };
    let model_arc = gpt2.model();
    let model = model_arc.lock().map_err(|e| {
        LuaError::external(format!(
            "{fn_name}: model lock while reading the custom architecture spec: {e}"
        ))
    })?;
    let branch = NnCustomBranch::from_gpt2_config(model.config());
    drop(model);
    Ok(branch)
}

/// Project the training-time device / dtype off a handle for
/// [`NnModelCard::from_training`]. Both slots come back as
/// `Some(&str)`-derived owned strings so the load path
/// (`apply_candle_branch_device_dtype`) restores the exact target
/// instead of silently falling back to the arch default (which is
/// CPU / f32 for `Gpt2Config::tiny()`). All shipped handles record
/// device / dtype as strings already (`Gpt2Handle::device` /
/// `TinyLlamaHandle::device` / …), so the projection is exact — no
/// stringification, no `Device::*` variant to translate here.
pub(super) fn candle_branch_device_dtype_of(handle: &NnHandle) -> (Option<String>, Option<String>) {
    let meta = handle.meta();
    (Some(meta.device.to_string()), Some(meta.dtype.to_string()))
}

/// Test-only helper: swap the recorded `dtype` string on a
/// [`Gpt2Handle`] without rebuilding the underlying model. Used by the
/// L5b/L5c bridge tests to exercise the f16 handle-time guard
/// (`guard_base_dtype_for_training`) without building an f16 model.
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

    // parquet(path, opts) — reads the `text_field` column via the
    // parquet row API, tokenizing each row. Same opts shape as jsonl
    // (`tokenizer` defaults to "gpt2").
    let parquet_tok_dir = nn_dir.join("tokenizers");
    let parquet = lua.create_function(
        move |_lua, (path, opts): (String, Option<LuaTable>)| -> LuaResult<DatasetHandle> {
            let dopts = extract_dataset_opts(opts.as_ref())?;
            let tokenizer_name = opts
                .as_ref()
                .and_then(|t| t.get::<Option<String>>("tokenizer").ok().flatten())
                .unwrap_or_else(|| "gpt2".to_string());
            let tok = HfTokenizer::load_cached(&tokenizer_name, &parquet_tok_dir)
                .map_err(|e| LuaError::external(format!("alc.nn.data.parquet: {e}")))?;
            let ds = ParquetDataset::new(std::path::Path::new(&path), dopts.clone(), tok)
                .map_err(|e| LuaError::external(format!("alc.nn.data.parquet: {e}")))?;
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
/// - the `FullFtConfig` opts-table extractor ([`extract_full_ft_opts`],
///   shared with the `run_*` surfaces via [`super::nn_opts`]),
/// - the [`super::nn_opts::train_err_to_lua`] converter that turns
///   `algocline_nn::train::TrainError` into `mlua::Error`, plus
///   [`checkpoint_to_lua`] which turns the returned `Checkpoint` into a
///   Lua table with primitive fields and a metrics sub-table.
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
    let cfg = extract_full_ft_opts(TRAINER_ERR_PREFIX, opts)?;
    // Extract the optional `on_ckpt` hook alongside the config. The
    // hook is a separate independent arg to `run_full_ft` (see
    // `CkptHook` docs: the callback can't sit inside `FullFtConfig`
    // without breaking its derive cascade).
    let hook = extract_on_ckpt_hook(TRAINER_ERR_PREFIX, lua, opts)?;
    let (ckpt_dir, ckpt_prefix) = resolve_ckpt_dest(opts, nn_dir, "full_ft")?;

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
        hook,
    );
    drop(model);
    drop(ds_lock);
    drop(ds_guard);

    let ckpt = result.map_err(|e| train_err_to_lua(TRAINER_ERR_PREFIX, e))?;
    checkpoint_to_lua(lua, &ckpt, &ckpt_prefix, None)
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
    let train_cfg = extract_full_ft_opts(TRAINER_ERR_PREFIX, opts)?;
    let lora_cfg = extract_lora_cfg(opts)?;
    let (ckpt_dir, ckpt_prefix) = resolve_ckpt_dest(opts, nn_dir, "lora")?;

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
            .unwrap_or_else(|| bundle_ref_for(&gpt2.variant)),
        None => bundle_ref_for(&gpt2.variant),
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
        &ckpt_prefix,
        lease,
    );
    drop(model);
    drop(ds_lock);
    drop(ds_guard);

    let ckpt = result.map_err(|e| train_err_to_lua(TRAINER_ERR_PREFIX, e))?;

    // Attach the LoRA branch descriptor. The Card foundation reads
    // this back through `meta.candle.lora` in `build_nn_meta`.
    //
    // ST-d additions: `target_modules` / `dropout` / `delta_path` are
    // required by `alc.nn.card.load_gpt2` to reconstruct the same
    // `LoraConfig` on reload and locate the delta safetensors. The
    // delta lives at `<ckpt_dir>/nn/<ckpt.bundle_ref>` — `run_lora_ft`
    // writes `nn/lora-<ckpt_prefix>.safetensors` under the caller's
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

    checkpoint_to_lua(lua, &ckpt, &ckpt_prefix, Some(lora_tbl))
}

/// `alc.nn.trainer.distill(handle, dataset, opts?) -> Checkpoint`.
///
/// Currently supports only `loss_kind = "ce"` (hard-label CE, the only
/// variant [`algocline_nn::train::DistillLossKind`] exposes). Unknown
/// `loss_kind` values error out rather than silently fall back to CE
/// ([`super::nn_opts::extract_distill_loss_kind`]).
fn distill_impl(
    lua: &Lua,
    handle: &LuaAnyUserData,
    dataset: &LuaAnyUserData,
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    lease: Arc<TrainingLease>,
) -> LuaResult<LuaTable> {
    let hyperparams = extract_full_ft_opts(TRAINER_ERR_PREFIX, opts)?;
    // This surface spells its own prefix (`alc.nn.trainer.distill`,
    // one level deeper than `TRAINER_ERR_PREFIX`) — keeping the
    // literal preserves the pre-consolidation message byte for byte.
    let loss_kind = extract_distill_loss_kind("alc.nn.trainer.distill", opts)?;
    let spec = DistillSpec {
        hyperparams,
        loss_kind,
    };
    let (ckpt_dir, ckpt_prefix) = resolve_ckpt_dest(opts, nn_dir, "distill")?;

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

    let ckpt = result.map_err(|e| train_err_to_lua(TRAINER_ERR_PREFIX, e))?;
    checkpoint_to_lua(lua, &ckpt, &ckpt_prefix, None)
}

/// Error prefix shared by the three `alc.nn.trainer.{full_ft, lora,
/// distill}` entries registered here.
///
/// Threaded into the shared [`super::nn_opts`] extractors /
/// [`super::nn_opts::train_err_to_lua`] so those keep emitting this
/// surface's prefix while holding a single implementation.
const TRAINER_ERR_PREFIX: &str = "alc.nn.trainer";

/// Extract [`LoraConfig`] from an opts table. `rank` and `alpha` are
/// required — omitting them is almost certainly a user error and
/// silently defaulting would hide a wrong training run.
///
/// This is the `alc.nn.trainer.lora` contract, which is *not* the one
/// the `alc.nn.wrap_lora` / `alc.nn.trainer.run_lora_ft` surfaces use
/// ([`super::nn_opts::extract_lora_cfg`]): here the architecture is
/// not consulted, `dropout` is taken verbatim (no range check),
/// `alpha` must additionally be finite, and `target_modules` is read
/// with `pairs` so a map-shaped table is accepted. Those are
/// Lua-visible differences, so the two extractors stay apart.
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

/// Decide where checkpoints get written for this training run.
///
/// - `opts.ckpt_dir` overrides the default (`<nn_dir>/ckpt`) when the
///   caller wants a scenario-specific location.
/// - `opts.ckpt_prefix` overrides the default `<path>_<epoch_us>`
///   prefix when the caller wants the ckpt filename to line up with a
///   name they already own (`run_lora_ft` in particular expects this
///   to match the `nn/lora-<prefix>.safetensors` bundle name).
///   `opts.card_id` is accepted as a deprecated alias of the same
///   knob (pre-refactor name; the value never was a Card store id).
fn resolve_ckpt_dest(
    opts: Option<&LuaTable>,
    nn_dir: &std::path::Path,
    stage: &str,
) -> LuaResult<(PathBuf, String)> {
    // Both `ckpt_dir` and `ckpt_prefix` are read strictly: wrong-type
    // input surfaces as a Lua error rather than silently falling back
    // to the default (which would write the checkpoint to an
    // unexpected location without diagnostic).
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
    // `opts.ckpt_prefix` is the canonical override key; `opts.card_id`
    // is accepted as a deprecated alias (the value was never a Card
    // store id — it is the checkpoint filename stem).
    let ckpt_prefix = match opts {
        Some(t) => {
            let explicit = match t.get::<Option<String>>("ckpt_prefix")? {
                Some(p) => Some(p),
                None => t.get::<Option<String>>("card_id")?,
            };
            match explicit {
                Some(p) => sanitize_stem(&p),
                None => unique_stem(stage),
            }
        }
        None => unique_stem(stage),
    };
    Ok((ckpt_dir, ckpt_prefix))
}

/// Convert a [`algocline_nn::train::Checkpoint`] into a Lua table.
///
/// Optional `lora_branch` sub-table is attached under `ckpt.lora` for
/// the LoRA binding; `full_ft` / `distill` pass `None`.
///
/// **On `ckpt.ckpt_prefix`**: this is the *checkpoint filename
/// prefix* (matches the safetensors bundle `<prefix>.safetensors`),
/// NOT a Card store id — `alc.nn.card.save` mints its own Card id
/// ([`CardId::mint`]). Callers who need a loadable Card id should use
/// the return value of `alc.nn.card.save` (or the `run_*` trainer
/// surfaces, which persist the Card themselves). The field was named
/// `card_id` before the 2026-07-30 Card-domain refactor; the rename
/// closes the eval-iter confusion where the prefix was mistaken for a
/// loadable Card id.
fn checkpoint_to_lua(
    lua: &Lua,
    ckpt: &algocline_nn::train::Checkpoint,
    ckpt_prefix: &str,
    lora_branch: Option<LuaTable>,
) -> LuaResult<LuaTable> {
    let out = lua.create_table()?;
    out.set("bundle_ref", ckpt.bundle_ref.clone())?;
    out.set("ckpt_prefix", ckpt_prefix.to_string())?;
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
    /// Arch-neutral projection of the wrapped handle.
    ///
    /// This is the enum's single dispatch point: every other accessor
    /// here and every Lua-facing method reads fields off the returned
    /// [`HandleMeta`], so adding an architecture means adding one arm
    /// to this method rather than one arm to each of a dozen.
    fn meta(&self) -> HandleMeta<'_> {
        match self {
            Self::Gpt2(h) => h.meta(),
            Self::TinyLlama(h) => h.meta(),
            Self::Llama(h) => h.meta(),
        }
    }

    /// Architecture family prefix (`"gpt2"` / `"tinyllama"` /
    /// `"llama"`) matching the first-column entries in
    /// [`algocline_nn::card::SUPPORTED_ARCHITECTURE_FAMILIES`].
    #[allow(dead_code)]
    pub(super) fn arch(&self) -> &'static str {
        self.meta().family
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
    /// Thin delegate to [`HandleMeta::arch_family_variant`], which
    /// documents the two storage conventions this normalises between.
    #[allow(dead_code)]
    pub(super) fn arch_family_variant(&self) -> String {
        self.meta().arch_family_variant()
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
    /// `Llama` always reports `false`: the adapter path does not
    /// support LoRA wrap in the current codebase (§Layer 2 non-goal,
    /// carried through Layer 4b).
    #[allow(dead_code)]
    pub(super) fn is_lora_wrapped(&self) -> bool {
        self.meta().lora_wrapped
    }
}

impl mlua::UserData for NnHandle {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // `arch` is the one accessor unique to the union — a typed
        // handle's caller already knows its architecture statically.
        methods.add_method("arch", |_, this, ()| Ok(this.arch()));
        // Everything else is the shared surface, resolved through
        // `NnHandle::meta`'s single three-arm match.
        add_meta_methods(methods, NnHandle::meta);
        // `handle:generate_session(prompt)` on the union — this is what
        // lets a Card reloaded via `alc.nn.card.load_handle` generate.
        super::nn_gen::add_nn_handle_generate_session_method(methods);
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
mod custom_spec_vocabulary_tests {
    //! Vocabulary parity pin for the `custom` spec axes (issue
    //! 467e6630).
    //!
    //! Two independent definitions spell the same axis values:
    //! [`build_custom_gpt2_config`]'s `match` arms decide what
    //! `alc.nn.preset.gpt2("custom", ...)` accepts, and the `serde`
    //! attributes on [`Gpt2Custom`] decide what a Card records. Both
    //! matter to one caller: a Card is meant to read back the way it
    //! was written, and [`gpt2_config_for_card`] deserializes the Card
    //! straight into a spec to rebuild the config. If the two drift, a
    //! Card either reports an axis under the wrong name or reloads as
    //! a different model. This walks every accepted value of every
    //! string axis in both directions (Lua -> spec -> JSON, and
    //! JSON -> spec) and asserts they agree.
    use super::*;
    use mlua::Lua;

    /// Every accepted value of every string-valued axis. A value added
    /// to [`build_custom_gpt2_config`] without a line here is an
    /// unpinned spelling.
    const STRING_AXES: &[(&str, &[&str])] = &[
        ("act", &["gelu", "relu", "silu", "swiglu", "geglu"]),
        ("norm", &["layernorm", "rmsnorm"]),
        ("residual", &["sequential", "parallel"]),
        ("placement", &["preln", "postln"]),
        ("pos", &["learned", "rope", "alibi", "nope"]),
    ];

    #[test]
    fn lua_and_card_axis_spellings_agree() {
        let lua = Lua::new();
        for (axis, values) in STRING_AXES {
            for value in *values {
                // One axis per table: some values are mutually
                // exclusive at build time (Post-LN excludes Parallel),
                // and the parse step under test is per-axis anyway.
                let opts = lua.create_table().expect("opts table");
                opts.set(*axis, *value).expect("set axis");
                let from_lua = build_custom_gpt2_config(GPT2_CUSTOM_PRESET_ERR_PREFIX, Some(&opts))
                    .unwrap_or_else(|e| panic!("Lua {axis} = {value:?} must parse: {e}"))
                    .custom
                    .expect("custom spec");

                // Forward: the Card must name the axis the way the
                // caller wrote it.
                let json = serde_json::to_value(&from_lua).expect("serialize spec");
                assert_eq!(
                    json.get(*axis),
                    Some(&serde_json::json!(*value)),
                    "Card spelling for {axis} = {value:?} drifted from the Lua \
                     vocabulary: {json}"
                );

                // Reverse: reading that spelling back must select the
                // same variant the Lua string did.
                let mut card_spec = serde_json::Map::new();
                card_spec.insert((*axis).to_string(), serde_json::json!(*value));
                let card_json = serde_json::Value::Object(card_spec);
                let from_card: Gpt2Custom = serde_json::from_value(card_json)
                    .unwrap_or_else(|e| panic!("Card {axis} = {value:?} must deserialize: {e}"));
                assert_eq!(
                    serde_json::to_value(&from_card).expect("re-serialize"),
                    json,
                    "{axis} = {value:?} deserializes to a different spec than the \
                     Lua parse produces"
                );
            }
        }
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

    /// Save side of the custom-shape contract (issue 467e6630): a
    /// `preset.gpt2("custom", ...)` handle projects its live
    /// `Gpt2Config` onto the branch the trainers record. The values
    /// have to be the ones the caller asked for — the load path
    /// rebuilds the config from exactly these.
    #[test]
    fn custom_branch_of_gpt2_projects_the_custom_preset_shape() {
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();
        opts.set("act", "swiglu").unwrap();
        opts.set("norm", "rmsnorm").unwrap();
        opts.set("pos", "rope").unwrap();
        opts.set("kv_heads", 1).unwrap();
        opts.set("untied_head", true).unwrap();
        opts.set("vocab", 96).unwrap();
        opts.set("ctx", 32).unwrap();
        let gpt2 = build_gpt2_handle("custom", Some(&opts), dir.path()).expect("build custom gpt2");
        let handle = NnHandle::Gpt2(gpt2);

        let branch = custom_branch_of_gpt2("test", &handle)
            .expect("projection must succeed")
            .expect("custom handle must yield a branch");
        assert_eq!(branch.vocab, 96);
        assert_eq!(branch.ctx, 32);
        assert_eq!(branch.layers, Gpt2Config::tiny().layers);
        assert_eq!(branch.heads, Gpt2Config::tiny().heads);
        assert_eq!(branch.dim, Gpt2Config::tiny().dim);
        assert_eq!(branch.spec.act, Activation::SwiGlu);
        assert_eq!(branch.spec.norm, NormKind::RmsNorm);
        assert_eq!(branch.spec.pos, PosKind::Rope);
        assert_eq!(branch.spec.kv_heads, Some(1));
        assert!(branch.spec.untied_head);
        assert!(branch.moe.is_none());
    }

    /// Named-variant and non-GPT-2 handles record no shape block:
    /// their `architecture` string already pins the shape, so adding
    /// one would only fatten the Card.
    #[test]
    fn custom_branch_of_gpt2_is_none_for_named_and_non_gpt2_handles() {
        let dir = tempdir();
        let lua = Lua::new();
        let opts = lua.create_table().unwrap();
        opts.set("pretrained", false).unwrap();

        let gpt2 = build_gpt2_handle("tiny", Some(&opts), dir.path()).expect("build gpt2");
        assert!(custom_branch_of_gpt2("test", &NnHandle::Gpt2(gpt2))
            .expect("projection must succeed")
            .is_none());

        let tll =
            build_tinyllama_handle("tinyllama-tiny", Some(&opts), dir.path()).expect("build tll");
        assert!(custom_branch_of_gpt2("test", &NnHandle::TinyLlama(tll))
            .expect("projection must succeed")
            .is_none());
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
mod nn_model_card_persist_tests {
    //! Layer 5a S2 successor — the former
    //! `build_create_payload_from_meta` unit coverage, retargeted at
    //! the [`NnModelCard`] aggregate + [`crate::card::nn::persist`]
    //! pair that replaced it. The envelope shape (`pkg.name` /
    //! `card_id` / `metadata.kind` / `metadata.nn`) is now assembled
    //! in exactly one place (`persist`), so the shape assertion runs
    //! against what actually lands in the store.
    use super::*;
    use crate::card::nn::NN_PKG;
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
                custom: None,
            }),
        }
    }

    #[test]
    fn persisted_envelope_shape_is_canonical() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileCardStore::new(dir.path().to_path_buf());
        let id = CardId::parse("my-merged-1").expect("valid id");
        let card = NnModelCard::new(id, sample_merged_meta()).expect("aggregate");

        let returned = persist(&store, &card).expect("persist");
        assert_eq!(returned, "my-merged-1");

        let payload = store
            .get("my-merged-1")
            .expect("get")
            .expect("card present");
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
    fn aggregate_refuses_unknown_architecture_family() {
        let mut meta = sample_merged_meta();
        meta.architecture = "nonexistent-arch".into();
        let err = NnModelCard::new(CardId::parse("my-merged-1").expect("valid id"), meta)
            .expect_err("should reject unknown arch");
        assert!(
            err.contains("unknown architecture family"),
            "expected architecture-family error, got: {err}"
        );
    }

    #[test]
    fn aggregate_refuses_bundle_ref_id_mismatch() {
        let err = NnModelCard::new(
            CardId::parse("someone-else").expect("valid id"),
            sample_merged_meta(),
        )
        .expect_err("bundle_ref/id divergence must be rejected");
        assert!(
            err.contains("does not match card_id"),
            "expected coherence error, got: {err}"
        );
    }
}

#[cfg(test)]
mod trainer_tests {
    use super::*;
    use mlua::Lua;
    use serde_json::json;

    /// Parse a literal test id (all fixtures use store-safe names).
    fn cid(s: &str) -> CardId {
        CardId::parse(s).expect("test card id")
    }

    /// Serialize a typed meta back to JSON so the shape assertions
    /// keep exercising exactly what `persist` will serialize.
    fn meta_json(meta: &NnCardMeta) -> Json {
        serde_json::to_value(meta).expect("serialize meta")
    }

    fn opts_from(lua: &Lua, pairs: &[(&str, LuaValue)]) -> LuaTable {
        let t = lua.create_table().expect("create opts table");
        for (k, v) in pairs {
            t.set(*k, v.clone()).expect("set opt field");
        }
        t
    }

    // The training-config extractor / schedule-parser and
    // distillation-loss-selector unit tests live next to their
    // implementation in `super::super::nn_opts` (moved there together
    // with the functions). The `lora_cfg_*` tests below stay here
    // because `extract_lora_cfg` is this surface's own contract.

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
    fn resolve_ckpt_dest_rejects_wrong_type_ckpt_dir() {
        let lua = Lua::new();
        let opts = opts_from(&lua, &[("ckpt_dir", LuaValue::Boolean(false))]);
        let tmp = std::env::temp_dir();
        let err = resolve_ckpt_dest(Some(&opts), &tmp, "full_ft").expect_err("wrong-type ckpt_dir");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn build_nn_meta_populates_lora_branch_when_meta_provides_it() {
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
        let meta = build_nn_meta(&cid("card-abc"), "my-model", &user_meta).expect("meta with lora");
        let nn = meta_json(&meta);
        let lora = nn.pointer("/candle/lora").expect("lora sub-object");
        assert_eq!(lora.get("rank"), Some(&json!(8)));
        assert_eq!(lora.get("alpha"), Some(&json!(16)));
        assert_eq!(
            lora.get("base_bundle_ref"),
            Some(&json!("nn/base-gpt2-medium"))
        );
    }

    #[test]
    fn build_nn_meta_omits_lora_when_meta_absent_or_null() {
        let no_candle = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
        });
        let p = meta_json(&build_nn_meta(&cid("c1"), "m", &no_candle).unwrap());
        let candle = p.pointer("/candle").expect("candle present");
        assert!(
            candle.get("lora").is_none() || candle.get("lora") == Some(&Json::Null),
            "lora must be absent: {candle}"
        );

        let candle_only = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
            "candle": { "device": "cpu" }
        });
        let p2 = meta_json(&build_nn_meta(&cid("c2"), "m", &candle_only).unwrap());
        let candle2 = p2.pointer("/candle").expect("candle present");
        assert!(candle2.get("lora").is_none() || candle2.get("lora") == Some(&Json::Null));

        let explicit_null = json!({
            "training_path": "full_ft",
            "architecture": "gpt2-medium",
            "candle": { "lora": Json::Null }
        });
        let p3 = meta_json(&build_nn_meta(&cid("c3"), "m", &explicit_null).unwrap());
        let candle3 = p3.pointer("/candle").expect("candle present");
        assert!(candle3.get("lora").is_none() || candle3.get("lora") == Some(&Json::Null));
    }

    #[test]
    fn build_nn_meta_preserves_lora_target_modules_dropout_delta_path() {
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
        let meta =
            build_nn_meta(&cid("card-lora"), "m", &user_meta).expect("meta with full lora branch");
        let nn = meta_json(&meta);
        let lora = nn.pointer("/candle/lora").expect("lora sub-object");
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
    fn build_nn_meta_defaults_lora_target_modules_when_meta_omits_them() {
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
        let meta =
            build_nn_meta(&cid("card-legacy"), "m", &user_meta).expect("meta with legacy lora");
        let nn = meta_json(&meta);
        let lora = nn.pointer("/candle/lora").expect("lora sub-object");
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
    /// its `card_id`.
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
    fn build_nn_meta_reports_invalid_lora_shape() {
        // Missing required NnLoraBranch fields (`rank` / `alpha` /
        // `base_bundle_ref`) surfaces as a clear error rather than a
        // silently half-populated Card.
        let bad = json!({
            "training_path": "lora",
            "architecture": "gpt2-medium",
            "candle": { "lora": { "rank": 8 } }
        });
        let err = build_nn_meta(&cid("cx"), "m", &bad).expect_err("invalid lora");
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

    fn write_test_card(store: &FileCardStore, mut nn_meta: serde_json::Value) -> String {
        // Mint the id up front and rewrite `candle.bundle_ref` (when
        // present) to the matching "nn/<id>" — the invariant every
        // production writer now guarantees via `NnModelCard`, and
        // which the load surfaces assert since the S-refactor.
        let card_id = unique_stem("testcard");
        if let Some(candle) = nn_meta.get_mut("candle") {
            if candle.is_object() {
                candle["bundle_ref"] = json!(bundle_ref_for(&card_id));
            }
        }
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "card_id": card_id,
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

    // ── gpt2-custom shape restore (issue 467e6630) ───────────

    /// Seed an empty bundle at the path `load_handle_impl` resolves so
    /// the pre-flight existence check passes. The two refusals below
    /// fire while rebuilding the config, before any safetensors byte
    /// is read, so the file's contents do not matter — but its absence
    /// would mask the error under test.
    fn touch_bundle(nn_dir: &std::path::Path, card_id: &str) {
        std::fs::create_dir_all(nn_dir).expect("create nn dir");
        std::fs::write(nn_dir.join(format!("{card_id}.safetensors")), b"")
            .expect("seed placeholder bundle");
    }

    /// A `"gpt2-custom"` card with no shape block is unrecoverable:
    /// the architecture string pins nothing and there is no
    /// `candle.custom` to rebuild from. That is the state of cards
    /// trained before the shape block existed, so the error has to say
    /// so rather than reading as "you typed the variant wrong".
    #[test]
    fn load_handle_refuses_gpt2_custom_card_without_custom_branch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-custom-legacy",
                "backend": "candle",
                "architecture": "gpt2-custom",
                "training_path": "full_ft",
                "candle": { "bundle_ref": "nn/placeholder" }
            }),
        );
        touch_bundle(&nn_dir, &card_id);
        let msg = match load_handle_impl(&store, &card_id, &nn_dir) {
            Ok(_) => panic!("custom card without a shape block must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("gpt2-custom") && msg.contains("predates custom-shape metadata"),
            "error must explain the missing shape block: {msg}"
        );
        assert!(
            msg.contains("retrain"),
            "error must name the recovery action: {msg}"
        );
    }

    /// A custom+MoE card has no reload path: the bundle carries
    /// per-block expert Vars that `from_safetensors_file` refuses.
    /// Rebuilding the config with `moe: None` would load a dense-MLP
    /// model under the card's name, so the loader must refuse instead
    /// of silently swapping the architecture.
    #[test]
    fn load_handle_refuses_custom_moe_card() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let card_id = write_test_card(
            &store,
            json!({
                "name": "gpt2-custom-moe",
                "backend": "candle",
                "architecture": "gpt2-custom",
                "training_path": "full_ft",
                "candle": {
                    "bundle_ref": "nn/placeholder",
                    "custom": {
                        "vocab": 64, "ctx": 16, "layers": 2, "heads": 2, "dim": 32,
                        "spec": { "norm": "rmsnorm" },
                        "moe": { "n_experts": 4, "top_k": 2, "alpha": 0.01 }
                    }
                }
            }),
        );
        touch_bundle(&nn_dir, &card_id);
        let msg = match load_handle_impl(&store, &card_id, &nn_dir) {
            Ok(_) => panic!("custom+MoE card must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("custom+MoE") && msg.contains("not supported yet"),
            "error must name the unsupported combination: {msg}"
        );
    }

    /// The happy path of the same rebuild, at the level of the config
    /// projection: every shape scalar comes off the branch, the spec
    /// is restored verbatim, and `eps` matches the `tiny` base the
    /// build side starts from (`build_custom_gpt2_config`). The
    /// end-to-end version (train -> Card -> reload -> generate) lives
    /// in `tests/nn_bridge_smoke.rs`, which can write a real bundle.
    #[test]
    fn gpt2_config_for_card_rebuilds_custom_shape() {
        let meta: NnCardMeta = serde_json::from_value(json!({
            "name": "gpt2-custom-run",
            "backend": "candle",
            "architecture": "gpt2-custom",
            "training_path": "full_ft",
            "candle": {
                "bundle_ref": "nn/gpt2-custom-run",
                "custom": {
                    "vocab": 96, "ctx": 32, "layers": 3, "heads": 4, "dim": 64,
                    "spec": {
                        "act": "swiglu",
                        "norm": "rmsnorm",
                        "pos": "rope",
                        "mlp_ratio": 3,
                        "kv_heads": 1,
                        "untied_head": true
                    }
                }
            }
        }))
        .expect("meta deserialize");

        let cfg = gpt2_config_for_card(&meta).expect("custom config rebuild");
        assert_eq!(cfg.vocab, 96);
        assert_eq!(cfg.ctx, 32);
        assert_eq!(cfg.layers, 3);
        assert_eq!(cfg.heads, 4);
        assert_eq!(cfg.dim, 64);
        assert_eq!(cfg.eps, Gpt2Config::tiny().eps);
        assert!(cfg.moe.is_none());
        let spec = cfg.custom.expect("custom spec restored");
        assert_eq!(spec.act, Activation::SwiGlu);
        assert_eq!(spec.norm, NormKind::RmsNorm);
        assert_eq!(spec.pos, PosKind::Rope);
        assert_eq!(spec.mlp_ratio, 3);
        assert_eq!(spec.kv_heads, Some(1));
        assert!(spec.untied_head);
        // Axes the card left unset stay at the reference.
        assert_eq!(spec.residual, ResidualKind::Sequential);
        assert_eq!(spec.placement, NormPlacement::PreLn);
        assert_eq!(spec.window, None);
    }

    /// A named variant must not consult the shape block at all — its
    /// preset is the single source of the shape, and a stray `custom`
    /// key on such a card must not silently reshape the model.
    #[test]
    fn gpt2_config_for_card_prefers_named_variant_over_custom_branch() {
        let meta: NnCardMeta = serde_json::from_value(json!({
            "name": "gpt2-tiny-run",
            "backend": "candle",
            "architecture": "gpt2-tiny",
            "training_path": "full_ft",
            "candle": {
                "bundle_ref": "nn/gpt2-tiny-run",
                "custom": {
                    "vocab": 999, "ctx": 999, "layers": 9, "heads": 9, "dim": 99,
                    "spec": { "act": "swiglu" }
                }
            }
        }))
        .expect("meta deserialize");

        let cfg = gpt2_config_for_card(&meta).expect("named variant rebuild");
        assert_eq!(cfg.vocab, Gpt2Config::tiny().vocab);
        assert_eq!(cfg.layers, Gpt2Config::tiny().layers);
        assert!(cfg.custom.is_none(), "named variant is the reference shape");
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

/// ST-B integration tests — `alc.nn.card.load_ckpt`.
///
/// The Cardless loader is what an `on_ckpt` hook uses to turn the raw
/// `info.ckpt_path` it receives mid-run into a handle, so the axes
/// here are:
///
/// - **hook round-trip**: a checkpoint written mid-run loads *from
///   inside the hook body* (the trainer holds the model mutex and the
///   dataset lock there — a load path that touched either would
///   deadlock) and the resulting handle generates.
/// - **restore equivalence**: with `steps % ckpt_every == 0` the last
///   mid-run checkpoint holds the weights the run finalises with, so
///   loading it must agree logit-for-logit with loading the Card the
///   Card-writing trainer persists. The premise is pinned in the test
///   body: a run whose step count is not a multiple of `ckpt_every`
///   keeps training after the last hook and would diverge for a
///   legitimate reason.
/// - **arch coverage**: GPT-2 (named + custom shape) and TinyLlama,
///   confirming the spec's `arch` vocabulary matches `load_handle`'s.
/// - **refusals**: missing file, missing / unknown `spec.arch`,
///   inference-only architectures, and custom-shape typos.
#[cfg(test)]
mod load_ckpt_tests {
    use super::*;
    use candle_core::Tensor;
    use mlua::Lua;
    use serde_json::json;

    fn opts_table(lua: &Lua, v: serde_json::Value) -> LuaTable {
        match lua.to_value(&v).expect("json to Lua value") {
            LuaValue::Table(t) => t,
            _ => unreachable!("json object must serialise to a Lua table"),
        }
    }

    /// Training row that fits `gpt2-tiny` (ctx = 16). Mirrors the
    /// sibling scaffolding in [`super::super::nn_trainer`]'s tests.
    fn overfit_row() -> Vec<u32> {
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    }

    fn make_dataset_handle(lua: &Lua, n: usize) -> LuaAnyUserData {
        let rows: Vec<Vec<u32>> = std::iter::repeat_with(overfit_row).take(n).collect();
        let ds = TokenizedDataset::new(
            rows,
            DatasetOpts {
                batch_size: 1,
                ctx_len: 16,
                shuffle: false,
                pad_id: 0,
                text_field: "text".into(),
            },
        );
        lua.create_userdata(DatasetHandle::for_test(
            Box::new(ds),
            "test-synthetic".into(),
            1,
            16,
        ))
        .expect("dataset userdata")
    }

    /// Full-fine-tune `gpt2-tiny` with mid-run checkpointing, running
    /// `on_hook` for every checkpoint the trainer writes.
    ///
    /// Returns the `info.ckpt_path` values the hook observed, in fire
    /// order. `ckpt_keep` is generous so every path handed to the hook
    /// is still on disk once the run returns.
    fn train_with_ckpts(
        lua: &Lua,
        nn_dir: &std::path::Path,
        prefix: &str,
        steps: usize,
        ckpt_every: usize,
        mut on_hook: impl FnMut(&Lua, &str) + Send + 'static,
    ) -> Vec<String> {
        let base_opts = opts_table(lua, json!({ "pretrained": false }));
        let base = build_gpt2_handle("tiny", Some(&base_opts), nn_dir).expect("gpt2 tiny base");
        let base_ud = lua.create_userdata(base).expect("handle userdata");
        let ds_ud = make_dataset_handle(lua, 20);

        let opts = opts_table(
            lua,
            json!({
                "lr": 5e-3,
                "batch_size": 1,
                "steps": steps,
                "warmup": 0,
                "schedule": "constant",
                "ckpt_every": ckpt_every,
                "ckpt_keep": 8,
                "ckpt_prefix": prefix,
            }),
        );

        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let on_ckpt = lua
            .create_function_mut(move |lua, info: LuaTable| -> LuaResult<()> {
                let path: String = info.get("ckpt_path")?;
                on_hook(lua, &path);
                sink.lock().expect("hook sink").push(path);
                Ok(())
            })
            .expect("create on_ckpt");
        opts.set("on_ckpt", on_ckpt).expect("set on_ckpt");

        full_ft_impl(
            lua,
            &base_ud,
            &ds_ud,
            Some(&opts),
            nn_dir,
            Arc::new(TrainingLease::new()),
        )
        .expect("full_ft with on_ckpt");

        let observed = seen.lock().expect("hook sink").clone();
        observed
    }

    /// Next-token logits over `prompt`, read straight off the model so
    /// the comparison does not depend on the generation session's
    /// bookkeeping.
    fn logits_of(handle: &NnHandle, prompt: &[u32]) -> Vec<f32> {
        let gpt2 = handle.as_gpt2().expect("gpt2 handle");
        let model = gpt2.model();
        let guard = model.lock().expect("model lock");
        let input =
            Tensor::from_slice(prompt, (1, prompt.len()), &Device::Cpu).expect("input tensor");
        guard
            .forward(&input)
            .expect("forward")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("logits row")
    }

    /// Write a Card pointing at `nn/<card_id>` so `load_handle` accepts
    /// it, and return the id (the caller drops the matching bundle at
    /// `<nn_dir>/<card_id>.safetensors`).
    fn write_full_ft_card(store: &FileCardStore, architecture: &str) -> String {
        let card_id = unique_stem("ckptcard");
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "card_id": card_id,
            "metadata": {
                "kind": "nn_model",
                "nn": {
                    "name": "ckpt-equivalence",
                    "backend": "candle",
                    "architecture": architecture,
                    "training_path": "full_ft",
                    "candle": {
                        "bundle_ref": bundle_ref_for(&card_id),
                        "device": "cpu",
                        "dtype": "f32"
                    }
                }
            }
        });
        let (card_id, _path) = store.create(payload).expect("create card");
        card_id
    }

    /// A file that exists but is never read — the refusals below fire
    /// before a single safetensors byte is touched, and the path check
    /// would otherwise mask the error under test.
    fn touch(path: &std::path::Path) {
        std::fs::write(path, b"").expect("seed placeholder file");
    }

    // ── hook round-trip ──────────────────────────────────────────

    #[test]
    fn load_ckpt_loads_a_mid_run_checkpoint_from_inside_the_hook() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();

        // Loading from inside the hook body is the production shape:
        // the trainer is holding the model mutex + dataset lock while
        // this runs, so a successful load here is the deadlock-freedom
        // assertion (a hang, not a failed assert, is the failure mode
        // this guards against).
        let loaded_in_hook: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let counter = Arc::clone(&loaded_in_hook);
        let observed = train_with_ckpts(&lua, &nn_dir, "hookload", 4, 2, move |lua, path| {
            let spec = opts_table(lua, json!({ "arch": "gpt2-tiny" }));
            let handle = load_ckpt_impl(path, &spec).expect("load_ckpt inside on_ckpt");
            assert_eq!(handle.arch(), "gpt2");
            *counter.lock().expect("counter") += 1;
        });

        assert_eq!(
            observed.len(),
            2,
            "steps=4 / ckpt_every=2 must fire the hook twice: {observed:?}"
        );
        assert_eq!(
            *loaded_in_hook.lock().expect("counter"),
            2,
            "every hook firing must have completed a load_ckpt"
        );

        // The handle the hook could have kept still generates after the
        // run returns (`ckpt_keep` kept the file).
        let last = observed.last().expect("at least one checkpoint");
        let spec = opts_table(&lua, json!({ "arch": "gpt2-tiny" }));
        let handle = load_ckpt_impl(last, &spec).expect("load_ckpt after the run");
        let vocab = handle.as_gpt2().expect("gpt2 handle").vocab;
        lua.globals()
            .set("h", lua.create_userdata(handle).expect("handle userdata"))
            .expect("set global");
        let argmax: u32 = lua
            .load(
                r#"
                local s = h:generate_session({ 1, 2, 3 })
                return s:next_logits():argmax()
                "#,
            )
            .eval()
            .expect("generate_session over a load_ckpt handle");
        assert!(
            (argmax as usize) < vocab,
            "argmax {argmax} must be a token id under vocab {vocab}"
        );
    }

    // ── restore equivalence ──────────────────────────────────────

    #[test]
    fn load_ckpt_of_the_final_checkpoint_matches_the_card_load() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();

        // Premise (design F13): `steps % ckpt_every == 0`. Only then is
        // the last mid-run checkpoint the same weight set the run
        // finalises with — with a remainder the trainer keeps stepping
        // after the last hook and the two loads diverge legitimately.
        let steps = 4;
        let ckpt_every = 2;
        assert_eq!(steps % ckpt_every, 0, "test premise");
        let observed = train_with_ckpts(&lua, &nn_dir, "equiv", steps, ckpt_every, |_, _| {});
        let last = observed.last().expect("at least one checkpoint");

        // Card side: `save_final` writes `<prefix>.safetensors`, which
        // is exactly what a Card-writing trainer persists as the
        // Card's bundle.
        let final_bundle = nn_dir.join("ckpt").join("equiv.safetensors");
        assert!(final_bundle.exists(), "final save at {final_bundle:?}");
        let card_id = write_full_ft_card(&store, "gpt2-tiny");
        std::fs::create_dir_all(&nn_dir).expect("nn dir");
        std::fs::copy(&final_bundle, nn_dir.join(format!("{card_id}.safetensors")))
            .expect("place the card bundle");

        let via_card = load_handle_impl(&store, &card_id, &nn_dir).expect("load_handle");
        let spec = opts_table(&lua, json!({ "arch": "gpt2-tiny" }));
        let via_ckpt = load_ckpt_impl(last, &spec).expect("load_ckpt");

        let prompt = [1u32, 2, 3];
        let from_ckpt = logits_of(&via_ckpt, &prompt);
        let from_card = logits_of(&via_card, &prompt);
        assert_eq!(
            from_ckpt.len(),
            from_card.len(),
            "both loads must produce a vocab-shaped row"
        );
        // Same weights through the same forward: the rows agree to
        // f32 round-off. A tolerance rather than bitwise equality
        // keeps the assertion honest about float summation order
        // without weakening it — a wrong config would be off by
        // orders of magnitude, not 1e-6.
        let worst = from_ckpt
            .iter()
            .zip(&from_card)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-6,
            "load_ckpt and load_handle disagree by {worst} on the same weights"
        );
    }

    // ── arch coverage ────────────────────────────────────────────

    #[test]
    fn load_ckpt_restores_a_tinyllama_bundle() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base = build_tinyllama_handle(
            "tinyllama-tiny",
            Some(&opts_table(&lua, json!({ "pretrained": false }))),
            &nn_dir,
        )
        .expect("tinyllama-tiny base");

        // Written in the HF Llama key layout, which is what
        // `TinyLlamaModel::from_safetensors_file` reads (see the
        // raw-VarMap test below for the layout that does not load).
        let path = tmp.path().join("tinyllama-bundle.safetensors");
        {
            let model = base.model();
            let guard = model.lock().expect("model lock");
            export_merged(
                &*guard,
                &MergedProvenance {
                    lora_card: "cards/none".into(),
                    arch: "tinyllama-tiny".into(),
                    bundle_ref: "nn/tinyllama-bundle".into(),
                },
                &path,
            )
            .expect("write tinyllama bundle");
        }

        let spec = opts_table(&lua, json!({ "arch": "tinyllama-tiny" }));
        let handle =
            load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec).expect("load_ckpt tinyllama");
        assert_eq!(handle.arch(), "tinyllama");
        assert!(handle.as_tinyllama().is_some());
    }

    /// KNOWN LIMITATION (pre-existing, not introduced by `load_ckpt`).
    ///
    /// A from-scratch TinyLlama handle registers its Vars at the model
    /// root (`TinyLlamaModel::new` hands the same `VarBuilder` to both
    /// halves of the split), so a raw `VarMap` dump — which is exactly
    /// what the trainer's `CheckpointStore` writes — has no `model.`
    /// prefix. `TinyLlamaModel::from_safetensors_file` reads the HF
    /// layout (`model.*` plus a top-level `lm_head.weight`), so the two
    /// do not meet. The same asymmetry blocks the TinyLlama
    /// `run_full_ft` -> Card -> `load_handle` round-trip; GPT-2 is
    /// unaffected (one flat namespace on both sides).
    ///
    /// This test pins the current failure so the gap is visible and a
    /// future fix has a place to flip. Until then, `load_ckpt` on a
    /// TinyLlama mid-run checkpoint fails loudly rather than silently
    /// returning a randomly-initialised model.
    #[test]
    fn load_ckpt_of_a_raw_tinyllama_varmap_dump_is_refused_today() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base = build_tinyllama_handle(
            "tinyllama-tiny",
            Some(&opts_table(&lua, json!({ "pretrained": false }))),
            &nn_dir,
        )
        .expect("tinyllama-tiny base");
        let path = tmp.path().join("tinyllama-varmap.safetensors");
        base.varmap()
            .expect("from-scratch handle carries a VarMap")
            .save(&path)
            .expect("write raw varmap dump");

        let spec = opts_table(&lua, json!({ "arch": "tinyllama-tiny" }));
        let msg = match load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!(
                "a raw VarMap dump does not carry the HF key layout; \
                 loading it must not silently succeed"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("cannot find tensor model."),
            "the failure must name the missing HF-prefixed tensor: {msg}"
        );
    }

    #[test]
    fn load_ckpt_rebuilds_a_custom_gpt2_shape_from_the_spec() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let shape = json!({
            "vocab": 48,
            "ctx": 16,
            "layers": 2,
            "heads": 2,
            "dim": 32,
            "act": "silu",
            "norm": "rmsnorm",
            "pos": "rope",
            "untied_head": true,
        });

        let mut build_opts = shape.clone();
        build_opts["pretrained"] = json!(false);
        let base = build_gpt2_handle("custom", Some(&opts_table(&lua, build_opts)), &nn_dir)
            .expect("custom gpt2 base");
        let path = tmp.path().join("custom-run.safetensors");
        base.varmap()
            .expect("from-scratch handle carries a VarMap")
            .save(&path)
            .expect("write checkpoint");

        // `"gpt2-custom"` pins no shape, so the spec repeats the same
        // keys the preset was built with — the load side has no other
        // source for them (a Card would carry them in
        // `candle.custom`).
        let mut spec = shape;
        spec["arch"] = json!("gpt2-custom");
        let handle = load_ckpt_impl(path.to_str().expect("utf-8 path"), &opts_table(&lua, spec))
            .expect("load_ckpt custom");
        let gpt2 = handle.as_gpt2().expect("gpt2 handle");
        assert_eq!(gpt2.vocab, 48);
        assert_eq!(gpt2.ctx, 16);
        assert_eq!(gpt2.layers, 2);
        assert_eq!(gpt2.dim, 32);
    }

    // ── refusals ─────────────────────────────────────────────────

    #[test]
    fn load_ckpt_refuses_a_path_that_is_not_on_disk() {
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let spec = opts_table(&lua, json!({ "arch": "gpt2-tiny" }));
        let missing = tmp.path().join("rotated-away.safetensors");
        let msg = match load_ckpt_impl(missing.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!("a missing checkpoint must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("alc.nn.card.load_ckpt:") && msg.contains("no safetensors file"),
            "message: {msg}"
        );
        assert!(
            msg.contains("on_ckpt"),
            "the message must name the rotation trap: {msg}"
        );
    }

    #[test]
    fn load_ckpt_requires_spec_arch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let path = tmp.path().join("run.safetensors");
        touch(&path);
        let spec = opts_table(&lua, json!({ "device": "cpu" }));
        let msg = match load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!("a spec without arch must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("alc.nn.card.load_ckpt:") && msg.contains("spec.arch is required"),
            "message: {msg}"
        );
    }

    #[test]
    fn load_ckpt_refuses_an_unregistered_arch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let path = tmp.path().join("run.safetensors");
        touch(&path);
        let spec = opts_table(&lua, json!({ "arch": "qwen2-1.5b" }));
        let msg = match load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!("an unregistered arch must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("alc.nn.card.load_ckpt:") && msg.contains("no bridge dispatch"),
            "message: {msg}"
        );
        assert!(
            msg.contains("gpt2"),
            "the message must list the registered families: {msg}"
        );
    }

    #[test]
    fn load_ckpt_refuses_an_inference_only_arch() {
        // `llama` resolves to an ArchOps entry, but its
        // `build_from_safetensors` slot is `None` — adapter-style
        // architectures never write a self-contained checkpoint. The
        // refusal has to say that rather than read as "unknown arch".
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let path = tmp.path().join("run.safetensors");
        touch(&path);
        let spec = opts_table(&lua, json!({ "arch": "llama-tiny" }));
        let msg = match load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!("an inference-only arch must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("alc.nn.card.load_ckpt:")
                && msg.contains("does not support loading a self-contained safetensors checkpoint"),
            "message: {msg}"
        );
    }

    #[test]
    fn load_ckpt_custom_spec_typos_carry_the_load_ckpt_prefix() {
        // The custom-shape parser is shared with
        // `alc.nn.preset.gpt2("custom", ...)`; a load-time typo must
        // point at the load entry, not at the preset call the caller
        // is not making here.
        let tmp = tempfile::TempDir::new().unwrap();
        let lua = Lua::new();
        let path = tmp.path().join("run.safetensors");
        touch(&path);
        let spec = opts_table(&lua, json!({ "arch": "gpt2-custom", "norm": "rmsnrm" }));
        let msg = match load_ckpt_impl(path.to_str().expect("utf-8 path"), &spec) {
            Ok(_) => panic!("an unknown norm must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("alc.nn.card.load_ckpt:") && msg.contains("unknown norm 'rmsnrm'"),
            "message: {msg}"
        );
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

    fn write_test_card(store: &FileCardStore, mut nn_meta: serde_json::Value) -> String {
        // Mint the id up front and rewrite `candle.bundle_ref` (when
        // present) to the matching "nn/<id>" — the invariant every
        // production writer now guarantees via `NnModelCard`, and
        // which the load surfaces assert since the S-refactor.
        let card_id = unique_stem("testcard");
        if let Some(candle) = nn_meta.get_mut("candle") {
            if candle.is_object() {
                candle["bundle_ref"] = json!(bundle_ref_for(&card_id));
            }
        }
        let payload = json!({
            "pkg": { "name": "alc_nn" },
            "card_id": card_id,
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
