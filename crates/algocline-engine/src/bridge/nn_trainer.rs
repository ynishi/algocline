//! `alc.nn.trainer.{run_lora_ft, run_full_ft, run_distill}` bridge —
//! Layer 5b S2, Layer 5c S1 and Layer 5c S2 Lua binds for the
//! one-call fine-tuning surfaces (train + save safetensors + write a
//! Card, all in a single call).
//!
//! Extends the pre-existing `alc.nn.trainer` sub-table (populated by
//! [`super::nn_card::register_nn_card`] with `full_ft` / `lora` /
//! `distill`, which return a raw Checkpoint table) by adding three
//! sibling entries that return a `card_id` string instead:
//!
//! ```text
//! alc.nn.trainer.run_lora_ft(base_handle, dataset, opts) -> lora_card_id
//! alc.nn.trainer.run_full_ft(base_handle, dataset, opts) -> card_id, candidates
//! alc.nn.trainer.run_distill(student_handle, dataset, opts) -> card_id
//! ```
//!
//! Sibling of Layer 5b S1 [`super::nn_wrap`], which registered the
//! wrap-only surface (`alc.nn.wrap_lora`). `run_lora_ft` consumes the
//! LoRA opts schema (`rank` / `alpha` / `target_modules` / `dropout`)
//! and layers a training-config schema (`lr` / `batch` / `steps` /
//! `warmup` / `schedule`) on top. `run_full_ft` consumes only the
//! training-config schema (no LoRA fields) and requires a
//! `pretrained = false` handle (full-fine-tune needs a `VarMap`; a
//! mmapped pretrained base does not carry one).
//!
//! # Invariants
//!
//! All errors surface as loud [`LuaError::external`] with prefix
//! `alc.nn.trainer.run_lora_ft:` — no silent fallback, no `warn!`
//! swallow (per `CLAUDE.md §Service 層の Error 伝播規律`). Every shape
//! listed in the Layer 5b design doc §2.2 has a matching branch in
//! this module.
//!
//! 1. Base handle validation refuses (a) non-userdata / non-`NnHandle`
//!    userdata, (b) already-wrapped handles (LoRA wrap happens inside
//!    [`algocline_nn::train::run_lora_ft`], so passing a pre-wrapped
//!    handle would double-wrap), (c) inference-only architectures
//!    (Llama).
//! 2. Dataset validation downcasts to the same [`DatasetHandle`]
//!    userdata that the existing [`super::nn_card`] trainer bindings
//!    (`full_ft` / `lora` / `distill`) consume. Do NOT reimplement
//!    the dataset accessor.
//! 3. Train opts pre-flight: `lr > 0`, `batch > 0`, `steps > 0`,
//!    `warmup >= 0`. `schedule` must be `"CosineWithWarmup"` or
//!    `"Constant"`; anything else is refused with the caller-supplied
//!    value in the error. `steps == 0` is caught here so the caller
//!    sees a Lua-shaped error rather than a candle back-trace via
//!    [`algocline_nn::train::TrainError::ZeroSteps`].
//! 4. `grad_accum` / `weight_decay` / `ckpt_every` / `ckpt_keep` are
//!    accepted as pass-through overrides — the same field set the
//!    `full_ft`-side surface (`alc.nn.trainer.{full_ft, lora,
//!    distill}`) accepts, reached through the shared
//!    [`super::nn_opts::extract_run_train_cfg`]. A missing key keeps
//!    the [`algocline_nn::train::FullFtConfig`] crate default and no
//!    surface-specific validation is layered on top. `grad_accum > 1`
//!    is honoured natively by [`algocline_nn::train::run_full_ft`] /
//!    `run_lora_ft` / `run_distill`: `N` micro-batches are summed via
//!    `GradStore::extend` (with a canonical `1 / N` pre-backward loss
//!    scale) and a single optimizer step is applied per effective
//!    batch of `batch * grad_accum` rows. Only `grad_accum = 0` is
//!    refused up front with
//!    [`algocline_nn::train::TrainError::ZeroGradAccum`].
//! 5. [`TrainingLease`] — a fresh
//!    `Arc::new(TrainingLease::new())` is constructed per-call. This
//!    is a documented limitation: `run_lora_ft` does NOT share a
//!    lease with the sibling `full_ft` / `lora` / `distill` entries
//!    in [`super::nn_card::register_nn_card`] (those share one lease
//!    per VM lifetime via [`super::nn_card::register_nn_card`]).
//!    Sharing across Lua calls is out of scope per design §0.
//! 6. Card write (design §3.2 step 7-8):
//!    - `training_path = "lora"`
//!    - `candle.bundle_ref = "nn/<lora_card_id>"`
//!    - `candle.lora = { rank, alpha, target_modules, dropout,
//!                       delta_path = <absolute>, base_bundle_ref }`
//!    - `base_bundle_ref` derived from the base handle via
//!      [`NnHandle::arch_family_variant`] as `"nn/<family>-<variant>"`.
//!    - `store.create(payload)` — the returned id must equal the
//!      pre-generated `lora_card_id`; a divergence surfaces as a
//!      loud Lua error (mirrors the `save_impl` / `merge_lora_impl`
//!      invariant guard).
//!    - `candle.custom` — present iff the base handle is a
//!      `preset.gpt2("custom", ...)` model, projected off its live
//!      `Gpt2Config` by
//!      [`super::nn_card::custom_branch_of_gpt2`]. `architecture =
//!      "gpt2-custom"` pins no shape, so this branch is what lets
//!      `alc.nn.card.load_handle` rebuild the trained config; named
//!      variants leave it absent. All three entry points below record
//!      it identically.
//! 7. Δ safetensors path is fixed by the Rust surface convention:
//!    [`run_lora_ft`] writes to
//!    `<ckpt_dir>/nn/lora-<card_id>.safetensors`. This bridge passes
//!    `ckpt_dir = nn_dir` so the delta lands at
//!    `<nn_dir>/nn/lora-<lora_card_id>.safetensors`.
//! 8. Feature gate: file body is `#[cfg(feature = "nn")]` at the
//!    module level via `bridge/mod.rs`; the default build never
//!    links this file.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_nn::card::{
    bundle_ref_for, CardId, NnCustomBranch, NnLoraBranch, NnModelCard, TrainingPath,
};
use algocline_nn::train::{
    run_allowed_ft, run_conditioned_ft, run_distill, run_full_ft, run_lora_ft, CrossEntropyLoss,
    DistillLossKind, DistillSpec, TrainingLease,
};
use mlua::prelude::*;

use crate::card::nn::persist;
use crate::card::FileCardStore;

use super::nn_card::{
    candle_branch_device_dtype_of, custom_branch_of_gpt2, guard_base_dtype_for_training,
    DatasetHandle, Gpt2Handle, LlamaHandle, NnHandle, TinyLlamaHandle,
};
use super::nn_opts::{
    extract_distill_loss_kind, extract_lora_cfg, extract_on_ckpt_hook, extract_run_train_cfg,
    train_err_to_lua,
};

/// Error prefix for the `alc.nn.trainer.run_lora_ft` surface.
const RUN_LORA_FT_ERR_PREFIX: &str = "alc.nn.trainer.run_lora_ft";
/// Error prefix for the `alc.nn.trainer.run_full_ft` surface.
const RUN_FULL_FT_ERR_PREFIX: &str = "alc.nn.trainer.run_full_ft";
/// Error prefix for the `alc.nn.trainer.run_distill` surface.
const RUN_DISTILL_ERR_PREFIX: &str = "alc.nn.trainer.run_distill";

/// Register `alc.nn.trainer.{run_lora_ft, run_full_ft, run_distill}`
/// onto the pre-existing `alc.nn.trainer` sub-table.
///
/// Must be called after [`super::nn_card::register_nn_card`] (which
/// creates the `alc.nn.trainer` sub-table populated with `full_ft` /
/// `lora` / `distill`) — this call extends that sub-table with three
/// sibling entries rather than creating a fresh one.
///
/// Signature mirrors the sibling [`super::nn_wrap::register_nn_wrap`]:
/// take the outer `alc` table and reach into the `nn.trainer`
/// sub-tables from inside.
pub(super) fn register_nn_trainer(
    lua: &Lua,
    alc_table: &LuaTable,
    card_store: Arc<FileCardStore>,
    nn_dir: PathBuf,
) -> LuaResult<()> {
    let nn_table: LuaTable = alc_table.get("nn")?;
    let trainer: LuaTable = nn_table.get("trainer")?;

    let store = Arc::clone(&card_store);
    let dir = nn_dir.clone();
    let run_lora_ft = lua.create_function(
        move |_lua, (base, dataset, opts): (LuaValue, LuaValue, LuaTable)| -> LuaResult<String> {
            run_lora_ft_impl(&store, &dir, &base, &dataset, opts)
        },
    )?;
    trainer.set("run_lora_ft", run_lora_ft)?;

    // L5c S1 sibling: same shape as run_lora_ft, but no LoRA opts and
    // returns a `training_path="full_ft"` Card. Fresh per-call
    // TrainingLease per L5b design §0 discipline (single-lease-per-call
    // contract; sharing across Lua calls is out of scope).
    let store_ff = Arc::clone(&card_store);
    let dir_ff = nn_dir.clone();
    let run_full_ft = lua.create_function(
        move |lua,
              (base, dataset, opts): (LuaValue, LuaValue, LuaTable)|
              -> LuaResult<(String, LuaTable)> {
            run_full_ft_impl(&store_ff, &dir_ff, lua, &base, &dataset, opts)
        },
    )?;
    trainer.set("run_full_ft", run_full_ft)?;

    // L5c S2 sibling: distillation surface. Same shape as run_full_ft
    // (no LoRA opts, requires a from-scratch student handle) plus the
    // `loss_kind` opts field; writes a `training_path="distillation"`
    // Card. Fresh per-call TrainingLease, same as the siblings above.
    let store_rd = Arc::clone(&card_store);
    let dir_rd = nn_dir;
    let run_distill = lua.create_function(
        move |_lua,
              (student, dataset, opts): (LuaValue, LuaValue, LuaTable)|
              -> LuaResult<String> {
            run_distill_impl(&store_rd, &dir_rd, &student, &dataset, opts)
        },
    )?;
    trainer.set("run_distill", run_distill)?;

    Ok(())
}

fn run_lora_ft_impl(
    store: &FileCardStore,
    nn_dir: &std::path::Path,
    base: &LuaValue,
    dataset: &LuaValue,
    opts: LuaTable,
) -> LuaResult<String> {
    // 1. Reject non-userdata base up front — design §2.2 inherits
    //    §2.1 row 1 ("expected NnHandle, got <type>").
    let base_ud = match base {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_lora_ft: expected NnHandle, got {}",
                base.type_name()
            )));
        }
    };

    // 2. Downcast to NnHandle (arch-neutral) or one of the typed
    //    Handles (backward-compat for callers that reach for
    //    `alc.nn.preset.gpt2` / `.tinyllama` directly). Same
    //    discipline as `nn_card::merge_lora_impl` /
    //    `nn_wrap::wrap_lora_impl`.
    let handle: NnHandle = if let Ok(nn) = base_ud.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = base_ud.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = base_ud.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = base_ud.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.trainer.run_lora_ft: expected NnHandle, got unknown userdata \
             (Gpt2Handle / TinyLlamaHandle / LlamaHandle also accepted)",
        ));
    };

    // 3. Refuse already-wrapped handles with the design §2.2
    //    directional error (points at "drop the wrap first" —
    //    `run_lora_ft` performs its own wrap inside the training
    //    loop, so passing a wrapped handle would double-wrap and
    //    fail cryptically deep in candle).
    if handle.is_lora_wrapped() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_lora_ft: expected base (unwrapped) NnHandle; \
             drop the wrap first",
        ));
    }

    // 4. Refuse inference-only architectures (Llama adapter). Done
    //    before opts validation so a Llama caller sees the
    //    directional arch error rather than a rank/alpha schema
    //    error that also applies.
    if let NnHandle::Llama(_) = handle {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_lora_ft: architecture {} is not LoRA-wrappable \
             (only gpt2 / tinyllama families are supported)",
            handle.arch()
        )));
    }

    // 4.5. Reject f16 base handles (no loss scaling ships; f32 and
    //      bf16 both train — bf16 through the FP32-master MixedAdamW).
    guard_base_dtype_for_training("alc.nn.trainer.run_lora_ft", &handle)?;

    // 5. Downcast dataset to the shared DatasetHandle userdata that
    //    the sibling trainer bindings (`full_ft` / `lora` /
    //    `distill`) consume. Refuse anything else with a clear
    //    directional error rather than the raw mlua downcast
    //    failure.
    let dataset_ud = match dataset {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_lora_ft: dataset must be an alc.nn.dataset \
                 (got {})",
                dataset.type_name()
            )));
        }
    };
    if dataset_ud.borrow::<DatasetHandle>().is_err() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_lora_ft: dataset must be an alc.nn.dataset \
             (got unknown userdata)",
        ));
    }

    // 6. Extract + validate LoRA + train opts (pre-flight so a
    //    misconfigured caller sees a Lua-shaped error rather than a
    //    candle back-trace).
    let arch = handle.arch();
    let lora_cfg = extract_lora_cfg(RUN_LORA_FT_ERR_PREFIX, arch, &opts)?;
    let train_cfg = extract_run_train_cfg(RUN_LORA_FT_ERR_PREFIX, &opts)?;

    // 7. Pre-mint the Card id (mirrors save_impl / merge_lora_impl).
    //    Minted before training because the id doubles as the delta
    //    checkpoint filename stem (`lora-<id>.safetensors`).
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_lora_ft");
    let lora_card_id = CardId::mint(name_base);

    // 8. Derive base_bundle_ref BEFORE any lock is acquired —
    //    `arch_family_variant()` needs only immutable handle
    //    fields, and threading it through here keeps the field
    //    lookup outside the training-loop critical section.
    let base_bundle_ref = bundle_ref_for(&handle.arch_family_variant());
    let architecture = handle.arch_family_variant();
    let (device_str, dtype_str) = candle_branch_device_dtype_of(&handle);
    // Same "before any lock" rule: reading the custom spec takes the
    // model mutex briefly (see `custom_branch_of_gpt2`), so it happens
    // here rather than nested inside the training critical section.
    let custom = custom_branch_of_gpt2(RUN_LORA_FT_ERR_PREFIX, &handle)?;

    // 9. Fresh per-call TrainingLease. Design §0: single-lease-per
    //    -call is the L5b contract; sharing across Lua calls is out
    //    of scope. This lease is NOT the same instance as the one
    //    held by `register_trainer_ns` for `full_ft` / `lora` /
    //    `distill`; a caller mixing `run_lora_ft` with those
    //    entries in parallel would see two independent leases (a
    //    concurrent `full_ft` under the old entry would not block
    //    `run_lora_ft` and vice versa). Real concurrency needs
    //    external coordination — not L5b's problem.
    let lease = Arc::new(TrainingLease::new());

    // 10. Dispatch on NnHandle variant into the generic
    //     `run_lora_ft::<M>` (Layer 3 S3 surface). Delta safetensors
    //     is written by the training loop at
    //     `<ckpt_dir>/nn/lora-<card_id>.safetensors` (Rust surface
    //     convention — do not override).
    let ckpt = match &handle {
        NnHandle::Gpt2(gpt2) => {
            let model_arc = gpt2.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let loss_fn = CrossEntropyLoss::new();
            let mut model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_lora_ft: model lock: {e}"))
            })?;

            let result = run_lora_ft(
                &mut *model,
                ds_lock.as_mut(),
                &lora_cfg,
                &train_cfg,
                &loss_fn,
                nn_dir,
                lora_card_id.as_str(),
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_LORA_FT_ERR_PREFIX, e))?
        }
        NnHandle::TinyLlama(tll) => {
            let model_arc = tll.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let loss_fn = CrossEntropyLoss::new();
            let mut model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_lora_ft: model lock: {e}"))
            })?;

            let result = run_lora_ft(
                &mut *model,
                ds_lock.as_mut(),
                &lora_cfg,
                &train_cfg,
                &loss_fn,
                nn_dir,
                lora_card_id.as_str(),
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_LORA_FT_ERR_PREFIX, e))?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the
            // assertion rather than silently invoking a non-existent
            // train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed Card aggregate and persist it.
    //     `ckpt.bundle_ref` is the trailing filename
    //     `"lora-<card_id>.safetensors"` (Layer 3 convention); the
    //     absolute path is `<nn_dir>/nn/<ckpt.bundle_ref>`. Envelope
    //     assembly + returned-id coherence live in
    //     [`crate::card::nn::persist`]; the aggregate constructor
    //     enforces bundle_ref = "nn/<id>" at build time.
    let delta_path = nn_dir.join("nn").join(&ckpt.bundle_ref);
    let delta_path_string = delta_path.to_string_lossy().to_string();

    // LoRA branch values come from the validated `LoraConfig` +
    // resolved `delta_path` + derived `base_bundle_ref`. `rank` /
    // `alpha` are `u32` on `NnLoraBranch` while `LoraConfig` stores
    // them as `usize` / `f32`; convert defensively.
    let lora_branch = NnLoraBranch {
        rank: lora_cfg.rank as u32,
        alpha: lora_cfg.alpha as u32,
        base_bundle_ref,
        target_modules: lora_cfg.target_modules.clone(),
        dropout: lora_cfg.dropout,
        delta_path: Some(delta_path_string),
    };

    let card = NnModelCard::from_training(
        lora_card_id,
        name_base,
        architecture,
        TrainingPath::Lora(lora_branch),
        &ckpt,
        &train_cfg,
        custom,
        device_str,
        dtype_str,
    )
    .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_lora_ft: {e}")))?;

    persist(store, &card)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_lora_ft: {e}")))
}

// ─── Layer 5c S1 — `alc.nn.trainer.run_full_ft` ─────────────────────
//
// Sibling of [`run_lora_ft_impl`] above. Shares:
//
// - The `NnHandle` dispatch shape (Gpt2 / TinyLlama; Llama refused
//   as inference-only).
// - The `DatasetHandle` downcast + lock discipline.
// - The `training_path` field + `NnCardMeta` envelope discipline;
//   diverges only on the `training_path` value (`"full_ft"` here vs
//   `"lora"` above) and the presence of the `NnLoraBranch`
//   sub-table (absent here; present above).
//
// Diverges from [`run_lora_ft_impl`] on:
//
// - `VarMap` requirement — full-fine-tune drives AdamW against the
//   full parameter list, which needs the base handle's original
//   `VarMap`. `pretrained = true` handles are mmap-backed and carry
//   no `VarMap`; they surface as a loud Lua-side error rather than
//   silently falling back.
// - No LoRA-config extraction — the `rank` / `alpha` /
//   `target_modules` / `dropout` opts fields do not apply. The
//   config schema shrinks to `lr` / `batch` / `steps` / `warmup` /
//   `schedule`. Extra LoRA-shaped keys in `opts` are silently
//   ignored (Lua's untyped table semantics — matches
//   `run_lora_ft_impl`'s treatment of stray keys).
// - No `already-wrapped` refusal — a LoRA-wrapped handle carries
//   only the LoRA delta VarMap on the underlying model, but the
//   `full_ft` semantics train the *base* parameters. Passing a
//   wrapped handle would surprise the caller; refuse it up front
//   with a directional error pointing at
//   [`super::nn_wrap::register_nn_wrap`] (unwrap first).
// - Error prefix — every `LuaError::external` emitted from this impl
//   carries the prefix `alc.nn.trainer.run_full_ft:`
//   (`RUN_FULL_FT_ERR_PREFIX`), threaded into the shared
//   `super::nn_opts` extractor / error converter so the loud-error
//   contract (one prefix per surface) holds off one implementation.
// - Channel routing — a GPT-2 handle built with an optional input
//   channel (`cond_slots` / `allowed_input` on the `custom` preset)
//   goes through `run_conditioned_ft` / `run_allowed_ft` instead of
//   `run_full_ft`. The decision reads the shape the handle carries
//   rather than an opts key ([`trained_channel`]): a model holding a
//   channel table has no forward pass that ignores it, so a key would
//   only give the caller a way to disagree with the model. TinyLlama
//   models carry no such table and keep the plain entry point.
// - `opts.on_ckpt` — the checkpoint hook, mirrored from the Layer 5b
//   sibling `alc.nn.trainer.full_ft` (`super::nn_card`) through the
//   shared [`super::nn_opts::extract_on_ckpt_hook`]. Full-fine-tune is
//   the only surface carrying it: `run_lora_ft` / `run_distill` keep
//   passing `None` (Layer 5b design decision, revisited only if a
//   caller needs it). Requires `ckpt_every > 0`, otherwise the hook
//   could never fire and the extractor refuses the pair.

/// Which optional input channel a model was built with, and therefore
/// which training entry point it has to go through.
///
/// The three are mutually exclusive by construction: a spec setting
/// both channels is refused at build time
/// (`algocline_nn::arch::Gpt2Custom::validate`) because no forward pass
/// delivers both, so the model that carried both would silently lose
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainedChannel {
    /// No channel table — the plain `Module::forward` path.
    None,
    /// A conditioning table, fed one condition per row.
    Conditioning,
    /// An allowed-id table, fed one id set per position.
    Allowed,
}

/// Read the channel off the Card branch projected from the handle's
/// live config.
///
/// The branch is the same value the Card records, which is what keeps
/// the routing decision and the written declaration from disagreeing:
/// a model routed through the conditioned entry point is exactly the
/// one whose Card says `cond_slots`, and the load path checks that
/// declaration against the bundle
/// (`algocline_nn::card::NnCardMeta::verify_channel_tensors`).
///
/// `None` for a reference-architecture run, which has no branch.
fn trained_channel(custom: Option<&NnCustomBranch>) -> TrainedChannel {
    let Some(spec) = custom.map(|branch| &branch.spec) else {
        return TrainedChannel::None;
    };
    if spec.cond_slots.is_some() {
        TrainedChannel::Conditioning
    } else if spec.allowed_input {
        TrainedChannel::Allowed
    } else {
        TrainedChannel::None
    }
}

/// L5c S1 core. Mirrors [`run_lora_ft_impl`] structurally; see the
/// section header above for the design divergence.
///
/// Takes `lua` (unlike the LoRA / distillation siblings) because the
/// optional `on_ckpt` hook holds the Lua callback through a `WeakLua`
/// — see [`extract_on_ckpt_hook`].
fn run_full_ft_impl(
    store: &FileCardStore,
    nn_dir: &std::path::Path,
    lua: &Lua,
    base: &LuaValue,
    dataset: &LuaValue,
    opts: LuaTable,
) -> LuaResult<(String, LuaTable)> {
    // 1. Reject non-userdata base up front.
    let base_ud = match base {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_full_ft: expected NnHandle, got {}",
                base.type_name()
            )));
        }
    };

    // 2. Downcast to NnHandle (arch-neutral) or a typed Handle. Same
    //    discipline as run_lora_ft_impl / merge_lora_impl.
    let handle: NnHandle = if let Ok(nn) = base_ud.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = base_ud.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = base_ud.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = base_ud.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.trainer.run_full_ft: expected NnHandle, got unknown userdata \
             (Gpt2Handle / TinyLlamaHandle / LlamaHandle also accepted)",
        ));
    };

    // 3. Refuse already-wrapped handles — full-fine-tune trains the
    //    base parameters; a wrapped handle would silently disagree
    //    with the caller's intent (design divergence bullet above).
    if handle.is_lora_wrapped() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_full_ft: expected base (unwrapped) NnHandle; \
             drop the wrap first (a LoRA-wrapped handle is a LoRA-training \
             target, not a full-fine-tune target)",
        ));
    }

    // 4. Refuse Llama (inference-only). Directional error, same as
    //    run_lora_ft_impl.
    if let NnHandle::Llama(_) = handle {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_full_ft: architecture {} is not trainable \
             (only gpt2 / tinyllama families are supported)",
            handle.arch()
        )));
    }

    // 4.5. Reject f16 base handles (see `run_lora_ft_impl` for
    //      rationale — f32 / bf16 train, f16 has no loss scaler).
    guard_base_dtype_for_training("alc.nn.trainer.run_full_ft", &handle)?;

    // 5. Dataset downcast to the shared DatasetHandle userdata.
    let dataset_ud = match dataset {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_full_ft: dataset must be an alc.nn.dataset \
                 (got {})",
                dataset.type_name()
            )));
        }
    };
    if dataset_ud.borrow::<DatasetHandle>().is_err() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_full_ft: dataset must be an alc.nn.dataset \
             (got unknown userdata)",
        ));
    }

    // 6. Extract + validate train opts (pre-flight so a misconfigured
    //    caller sees a Lua-shaped error rather than a candle back-trace).
    let train_cfg = extract_run_train_cfg(RUN_FULL_FT_ERR_PREFIX, &opts)?;

    // 6.5. Extract the optional `on_ckpt` hook alongside the config
    //      (shared helper with the Layer 5b sibling
    //      `alc.nn.trainer.full_ft`). The extractor also refuses an
    //      `on_ckpt` paired with `ckpt_every = 0` / absent, which would
    //      register a hook that can never fire.
    let hook = extract_on_ckpt_hook(RUN_FULL_FT_ERR_PREFIX, lua, Some(&opts))?;

    // 7. Pre-mint the Card id (mirrors run_lora_ft_impl step 7).
    //    Minted before training because the id doubles as the
    //    checkpoint filename stem (`<nn_dir>/<id>.safetensors`).
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_full_ft");
    let card_id = CardId::mint(name_base);

    // 8. Derive architecture BEFORE any lock (immutable handle field
    //    lookup outside the training-loop critical section). The
    //    custom-architecture branch rides along here for the same
    //    reason (it takes the model mutex briefly — see
    //    `custom_branch_of_gpt2`).
    let architecture = handle.arch_family_variant();
    let custom = custom_branch_of_gpt2(RUN_FULL_FT_ERR_PREFIX, &handle)?;
    // Which entry point the run takes follows from the shape the
    // handle was built with, not from an opts key. A model carrying a
    // channel table has no forward pass that ignores it, so a key
    // would only offer the caller a way to disagree with the model.
    let channel = trained_channel(custom.as_ref());
    let (device_str, dtype_str) = candle_branch_device_dtype_of(&handle);

    // 9. Fresh per-call TrainingLease (design §0, matches
    //    run_lora_ft_impl step 9).
    let lease = Arc::new(TrainingLease::new());

    // 10. Dispatch on NnHandle variant into the generic
    //     `run_full_ft::<M>` (Layer 3 S3 surface).
    //     `save_final` writes to `<ckpt_dir>/<ckpt_prefix>.safetensors`,
    //     which lines up with `alc.nn.load` (which resolves
    //     `<nn_dir>/<card_id>.safetensors`).
    let ckpt = match &handle {
        NnHandle::Gpt2(gpt2) => {
            let vm_arc = gpt2.varmap().ok_or_else(|| {
                LuaError::external(
                    "alc.nn.trainer.run_full_ft: handle was built with \
                     pretrained=true; full-fine-tune requires a from-scratch \
                     handle (pretrained=false)",
                )
            })?;
            let model_arc = gpt2.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let loss_fn = CrossEntropyLoss::new();
            let model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_full_ft: model lock: {e}"))
            })?;

            // Optional `on_ckpt` hook (step 6.5). `None` when the
            // caller omitted the key, which keeps the pre-hook
            // behaviour bit-identical. `CkptHook` is not `Clone`, but
            // the arms below are exclusive so the move is fine.
            let result = match channel {
                TrainedChannel::None => run_full_ft(
                    &*model,
                    &vm_arc,
                    ds_lock.as_mut(),
                    &train_cfg,
                    &loss_fn,
                    nn_dir,
                    card_id.as_str(),
                    Arc::clone(&lease),
                    hook,
                ),
                TrainedChannel::Conditioning => run_conditioned_ft(
                    &*model,
                    &vm_arc,
                    ds_lock.as_mut(),
                    &train_cfg,
                    &loss_fn,
                    nn_dir,
                    card_id.as_str(),
                    Arc::clone(&lease),
                    hook,
                ),
                TrainedChannel::Allowed => run_allowed_ft(
                    &*model,
                    &vm_arc,
                    ds_lock.as_mut(),
                    &train_cfg,
                    &loss_fn,
                    nn_dir,
                    card_id.as_str(),
                    Arc::clone(&lease),
                    hook,
                ),
            };
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_FULL_FT_ERR_PREFIX, e))?
        }
        NnHandle::TinyLlama(tll) => {
            let vm_arc = tll.varmap().ok_or_else(|| {
                LuaError::external(
                    "alc.nn.trainer.run_full_ft: handle was built with \
                     pretrained=true; full-fine-tune requires a from-scratch \
                     handle (pretrained=false)",
                )
            })?;
            let model_arc = tll.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let loss_fn = CrossEntropyLoss::new();
            let model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_full_ft: model lock: {e}"))
            })?;

            let result = run_full_ft(
                &*model,
                &vm_arc,
                ds_lock.as_mut(),
                &train_cfg,
                &loss_fn,
                nn_dir,
                card_id.as_str(),
                Arc::clone(&lease),
                // Same as above: optional `on_ckpt` hook from step 6.5.
                hook,
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_FULL_FT_ERR_PREFIX, e))?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the assertion
            // rather than silently invoking a non-existent train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed Card aggregate and persist it.
    //     Full-fine-tune carries NO LoRA branch — the aggregate's
    //     `TrainingPath::FullFt` arm leaves that field `None`,
    //     matching `training_path="full_ft"` cards written by the
    //     sibling `alc.nn.trainer.full_ft` + `alc.nn.card.save` flow.
    //     It DOES carry the custom branch when the base was a
    //     `preset.gpt2("custom", ...)` handle: `architecture =
    //     "gpt2-custom"` pins no shape, so `alc.nn.card.load_handle`
    //     rebuilds the config from that branch.
    let card = NnModelCard::from_training(
        card_id,
        name_base,
        architecture,
        TrainingPath::FullFt,
        &ckpt,
        &train_cfg,
        custom,
        device_str,
        dtype_str,
    )
    .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_full_ft: {e}")))?;

    // The `ckpt.bundle_ref` field is left unused on purpose: L5c S1
    // consumers only see the Card id, and the on-disk safetensors
    // path is fixed by the `<nn_dir>/<card_id>.safetensors`
    // convention (validated by
    // `run_full_ft_bridge_tests::run_full_ft_gpt2_happy_path_writes_full_ft_card`
    // below).
    let _ = ckpt.bundle_ref;

    // Second return value: the checkpoints the hook asked to hold.
    // A caller that binds one variable (`local id = run_full_ft(…)`)
    // is unaffected — Lua drops the extra value — so this is additive
    // for every consumer that predates the keep surface.
    let candidates = crate::bridge::nn_opts::candidates_to_lua(lua, &ckpt.candidates)?;

    let card_id = persist(store, &card)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_full_ft: {e}")))?;
    Ok((card_id, candidates))
}

// ─── Layer 5c S2 — `alc.nn.trainer.run_distill` ─────────────────────
//
// Sibling of [`run_full_ft_impl`] above. Distillation IS a full
// fine-tune under a distillation loss (the Rust surface
// [`algocline_nn::train::run_distill`] forwards to `run_full_ft`
// with the loss selected by [`DistillSpec::loss_kind`]; the teacher
// signal lives in the dataset, not in a second model instance), so
// every validation step mirrors `run_full_ft_impl`:
//
// - `VarMap` requirement — a `pretrained = true` student is
//   mmap-backed and carries no `VarMap`; refused with a directional
//   error.
// - LoRA-wrapped refusal — distillation trains the *base*
//   parameters; a wrapped student would silently disagree with the
//   caller's intent.
// - Llama refusal — inference-only architecture.
//
// Diverges from [`run_full_ft_impl`] on:
//
// - `opts.loss_kind` — the distillation-loss selector (`"ce"` today,
//   the only variant [`DistillLossKind`] exposes). Unknown values
//   are refused rather than silently falling back to CE.
// - `training_path = "distillation"` on the written Card (one of
//   [`algocline_nn::card::SUPPORTED_TRAINING_PATHS`]).
// - Error prefix `alc.nn.trainer.run_distill:`
//   (`RUN_DISTILL_ERR_PREFIX`) — one prefix per surface per design's
//   loud-error contract, carried into the shared `super::nn_opts`
//   entries as an argument.

/// L5c S2 core. Mirrors [`run_full_ft_impl`] structurally; see the
/// section header above for the design divergence.
fn run_distill_impl(
    store: &FileCardStore,
    nn_dir: &std::path::Path,
    student: &LuaValue,
    dataset: &LuaValue,
    opts: LuaTable,
) -> LuaResult<String> {
    // 1. Reject non-userdata student up front.
    let student_ud = match student {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_distill: expected NnHandle, got {}",
                student.type_name()
            )));
        }
    };

    // 2. Downcast to NnHandle (arch-neutral) or a typed Handle. Same
    //    discipline as run_full_ft_impl / run_lora_ft_impl.
    let handle: NnHandle = if let Ok(nn) = student_ud.borrow::<NnHandle>() {
        (*nn).clone()
    } else if let Ok(g) = student_ud.borrow::<Gpt2Handle>() {
        NnHandle::Gpt2(g.clone())
    } else if let Ok(t) = student_ud.borrow::<TinyLlamaHandle>() {
        NnHandle::TinyLlama(t.clone())
    } else if let Ok(l) = student_ud.borrow::<LlamaHandle>() {
        NnHandle::Llama(l.clone())
    } else {
        return Err(LuaError::external(
            "alc.nn.trainer.run_distill: expected NnHandle, got unknown userdata \
             (Gpt2Handle / TinyLlamaHandle / LlamaHandle also accepted)",
        ));
    };

    // 3. Refuse already-wrapped handles — distillation trains the
    //    base parameters (design divergence bullet above).
    if handle.is_lora_wrapped() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_distill: expected base (unwrapped) NnHandle; \
             drop the wrap first (a LoRA-wrapped handle is a LoRA-training \
             target, not a distillation student)",
        ));
    }

    // 4. Refuse Llama (inference-only). Directional error, same as
    //    the siblings.
    if let NnHandle::Llama(_) = handle {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_distill: architecture {} is not trainable \
             (only gpt2 / tinyllama families are supported)",
            handle.arch()
        )));
    }

    // 4.5. Reject f16 student handles (see `run_lora_ft_impl` for
    //      rationale — f32 / bf16 train, f16 has no loss scaler).
    guard_base_dtype_for_training("alc.nn.trainer.run_distill", &handle)?;

    // 5. Dataset downcast to the shared DatasetHandle userdata.
    let dataset_ud = match dataset {
        LuaValue::UserData(u) => u,
        _ => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_distill: dataset must be an alc.nn.dataset \
                 (got {})",
                dataset.type_name()
            )));
        }
    };
    if dataset_ud.borrow::<DatasetHandle>().is_err() {
        return Err(LuaError::external(
            "alc.nn.trainer.run_distill: dataset must be an alc.nn.dataset \
             (got unknown userdata)",
        ));
    }

    // 6. Extract + validate train opts and the distillation-loss
    //    selector (pre-flight so a misconfigured caller sees a
    //    Lua-shaped error rather than a candle back-trace).
    let train_cfg = extract_run_train_cfg(RUN_DISTILL_ERR_PREFIX, &opts)?;
    let loss_kind = extract_distill_loss_kind(RUN_DISTILL_ERR_PREFIX, Some(&opts))?;
    let spec = DistillSpec {
        hyperparams: train_cfg,
        loss_kind,
    };

    // 7. Pre-mint the Card id (mirrors run_full_ft_impl step 7).
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_distill");
    let card_id = CardId::mint(name_base);

    // 8. Derive architecture BEFORE any lock (immutable handle field
    //    lookup outside the training-loop critical section), plus the
    //    student's custom-architecture branch (brief model-mutex read,
    //    same rationale as the siblings).
    let architecture = handle.arch_family_variant();
    let custom = custom_branch_of_gpt2(RUN_DISTILL_ERR_PREFIX, &handle)?;
    let (device_str, dtype_str) = candle_branch_device_dtype_of(&handle);

    // 9. Fresh per-call TrainingLease (design §0, matches the
    //    siblings).
    let lease = Arc::new(TrainingLease::new());

    // 10. Dispatch on NnHandle variant into the generic
    //     `run_distill::<M>`. `ckpt_prefix = card_id` pins the final
    //     safetensors to `<nn_dir>/<card_id>.safetensors`, same as
    //     run_full_ft (the underlying loop IS run_full_ft).
    let ckpt = match &handle {
        NnHandle::Gpt2(gpt2) => {
            let vm_arc = gpt2.varmap().ok_or_else(|| {
                LuaError::external(
                    "alc.nn.trainer.run_distill: handle was built with \
                     pretrained=true; distillation requires a from-scratch \
                     student handle (pretrained=false)",
                )
            })?;
            let model_arc = gpt2.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_distill: model lock: {e}"))
            })?;

            let result = run_distill(
                &*model,
                &vm_arc,
                ds_lock.as_mut(),
                &spec,
                nn_dir,
                card_id.as_str(),
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_DISTILL_ERR_PREFIX, e))?
        }
        NnHandle::TinyLlama(tll) => {
            let vm_arc = tll.varmap().ok_or_else(|| {
                LuaError::external(
                    "alc.nn.trainer.run_distill: handle was built with \
                     pretrained=true; distillation requires a from-scratch \
                     student handle (pretrained=false)",
                )
            })?;
            let model_arc = tll.model();
            let ds_handle = dataset_ud.borrow_mut::<DatasetHandle>()?;
            let mut ds_lock = ds_handle.inner_lock()?;

            let model = model_arc.lock().map_err(|e| {
                LuaError::external(format!("alc.nn.trainer.run_distill: model lock: {e}"))
            })?;

            let result = run_distill(
                &*model,
                &vm_arc,
                ds_lock.as_mut(),
                &spec,
                nn_dir,
                card_id.as_str(),
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(|e| train_err_to_lua(RUN_DISTILL_ERR_PREFIX, e))?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the assertion
            // rather than silently invoking a non-existent train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed Card aggregate and persist it.
    //     Distillation carries NO LoRA branch; `loss_kind` rides on
    //     `hyperparams` (via `TrainingPath::Distillation`) so the
    //     run's loss selection is auditable from the Card.
    let loss_kind_str = match spec.loss_kind {
        DistillLossKind::Ce => "ce",
    };

    let card = NnModelCard::from_training(
        card_id,
        name_base,
        architecture,
        TrainingPath::Distillation {
            loss_kind: loss_kind_str.into(),
        },
        &ckpt,
        &spec.hyperparams,
        custom,
        device_str,
        dtype_str,
    )
    .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_distill: {e}")))?;

    // `ckpt.bundle_ref` unused on purpose — same rationale as
    // run_full_ft_impl (consumers only see the Card id; the on-disk
    // path is fixed by the `<nn_dir>/<card_id>.safetensors`
    // convention).
    let _ = ckpt.bundle_ref;

    persist(store, &card)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_distill: {e}")))
}

#[cfg(test)]
mod run_ft_bridge_tests {
    //! Layer 5b S2 + Layer 5c S1/S2 — bridge integration tests for
    //! `alc.nn.trainer.{run_lora_ft, run_full_ft, run_distill}`.
    //!
    //! run_lora_ft coverage (Axes A3/A4/B6/B7/C1-C3/D1/D2):
    //! GPT-2 + TinyLlama happy paths (arch dispatch + on-disk Card +
    //! Δ safetensors + payload shape). Axis B6/B7 exercise config
    //! schema refusals unique to the trainer (steps / schedule).
    //! Axis C1-C3 exercise state invariants that only become
    //! verifiable post-train (base-freeze, Δ var-count, cross-surface
    //! `load_wrap` consumption). Axis D1/D2 exercise the Card + Δ
    //! round-trip through `alc.nn.card.load_wrap`.
    //!
    //! All tests use the CPU/F32 `gpt2-tiny` / `tinyllama-tiny`
    //! micro shapes — same discipline as `nn_card::merge_lora_bridge_tests`
    //! (no HF hub download, no >1s train step).
    use super::super::nn_card::{
        build_gpt2_handle, build_tinyllama_handle, gpt2_handle_with_dtype, load_wrap_impl,
        tinyllama_handle_with_dtype,
    };
    use super::*;
    use algocline_nn::train::{DatasetOpts, TokenizedDataset};
    use candle_nn::VarMap;
    use mlua::Lua;
    use serde_json::json;
    use std::sync::Mutex;

    /// Training row that fits inside both `gpt2-tiny` (ctx=16,
    /// vocab=64) and `tinyllama-tiny` (ctx=16, vocab=32). Values
    /// stay inside `vocab = 32`. Mirrors
    /// `algocline-nn/tests/tinyllama_lora_ft.rs::overfit_row`
    /// (15-line copy, per subtask spec).
    fn overfit_row() -> Vec<u32> {
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    }

    fn opts_table(lua: &Lua, v: serde_json::Value) -> LuaTable {
        use mlua::LuaSerdeExt;
        let val = lua.to_value(&v).expect("to_value");
        match val {
            LuaValue::Table(t) => t,
            _ => unreachable!("json object must serialise to Lua table"),
        }
    }

    /// Build a DatasetHandle userdata backed by an in-memory
    /// [`TokenizedDataset`] of the given row repeated `n` times.
    /// Constructed inline (rather than routing through the public
    /// `alc.nn.data.synthetic` Lua entry) so the test module stays
    /// self-contained and does not require the full bridge to be
    /// registered.
    fn make_dataset_handle(lua: &Lua, row: Vec<u32>, n: usize) -> LuaAnyUserData {
        let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(n).collect();
        let dopts = DatasetOpts {
            batch_size: 1,
            ctx_len: 16,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        };
        let ds = TokenizedDataset::new(rows, dopts);
        let handle = DatasetHandle::for_test(Box::new(ds), "test-synthetic".into(), 1, 16);
        lua.create_userdata(handle).expect("dataset userdata")
    }

    fn setup_gpt2_scaffold() -> (tempfile::TempDir, FileCardStore, PathBuf, Gpt2Handle, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("build gpt2 tiny base");
        (tmp, store, nn_dir, base, lua)
    }

    fn setup_tinyllama_scaffold() -> (
        tempfile::TempDir,
        FileCardStore,
        PathBuf,
        TinyLlamaHandle,
        Lua,
    ) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let base = build_tinyllama_handle("tinyllama-tiny", Some(&base_opts), &nn_dir)
            .expect("build tinyllama-tiny base");
        (tmp, store, nn_dir, base, lua)
    }

    fn snapshot_varmap(vm: &VarMap) -> Vec<Vec<f32>> {
        vm.all_vars()
            .iter()
            .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
            .collect()
    }

    fn base_train_opts() -> serde_json::Value {
        json!({
            "rank": 4,
            "alpha": 8.0,
            "lr": 5e-3,
            "batch": 1,
            "steps": 3,
            "warmup": 0,
            "schedule": "CosineWithWarmup",
        })
    }

    // ─── Axis A — happy paths ─────────────────────────────────────

    #[test]
    fn run_lora_ft_gpt2_happy_path_writes_lora_card_and_delta() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        // Δ safetensors exists on disk at the expected path.
        let delta_path = nn_dir
            .join("nn")
            .join(format!("lora-{card_id}.safetensors"));
        assert!(
            delta_path.exists(),
            "Δ safetensors must exist at {delta_path:?}"
        );

        // Card metadata: training_path / architecture / lora branch.
        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(nn.get("training_path").unwrap().as_str().unwrap(), "lora");
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "gpt2-tiny"
        );
        assert_eq!(
            nn.get("candle")
                .and_then(|c| c.get("bundle_ref"))
                .and_then(|b| b.as_str())
                .unwrap(),
            format!("nn/{card_id}")
        );
        let lora = nn.get("candle").and_then(|c| c.get("lora")).unwrap();
        assert_eq!(lora.get("rank").unwrap().as_u64().unwrap(), 4);
        assert_eq!(lora.get("alpha").unwrap().as_u64().unwrap(), 8);
        assert_eq!(
            lora.get("base_bundle_ref").unwrap().as_str().unwrap(),
            "nn/gpt2-tiny"
        );
        assert_eq!(
            lora.get("delta_path").unwrap().as_str().unwrap(),
            delta_path.to_string_lossy()
        );
    }

    #[test]
    fn run_lora_ft_tinyllama_happy_path_writes_lora_card_and_delta() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();

        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        let delta_path = nn_dir
            .join("nn")
            .join(format!("lora-{card_id}.safetensors"));
        assert!(delta_path.exists());

        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(nn.get("training_path").unwrap().as_str().unwrap(), "lora");
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "tinyllama-tiny"
        );
        let lora = nn.get("candle").and_then(|c| c.get("lora")).unwrap();
        assert_eq!(
            lora.get("base_bundle_ref").unwrap().as_str().unwrap(),
            "nn/tinyllama-tiny"
        );
    }

    // ─── Axis B — config schema refusals ──────────────────────────

    fn expect_err<T: std::fmt::Debug>(result: LuaResult<T>) -> String {
        match result {
            Ok(ok) => panic!("expected the run to fail; got {ok:?}"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn run_lora_ft_refuses_zero_steps() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let mut o = base_train_opts();
        o["steps"] = json!(0);
        let opts = opts_table(&lua, o);
        let msg = expect_err(run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_lora_ft:") && msg.contains("opts.steps"),
            "expected steps error, got: {msg}"
        );
    }

    #[test]
    fn run_lora_ft_refuses_unknown_schedule() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let mut o = base_train_opts();
        o["schedule"] = json!("Adam");
        let opts = opts_table(&lua, o);
        let msg = expect_err(run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_lora_ft:")
                && msg.contains("opts.schedule")
                && msg.contains("Adam"),
            "expected schedule error, got: {msg}"
        );
    }

    // ─── Axis C — state invariants ────────────────────────────────

    #[test]
    fn run_lora_ft_leaves_base_vars_bit_identical() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let base_vm = base.varmap().expect("from-scratch base carries VarMap");
        let before = snapshot_varmap(&base_vm);
        let base_var_count = base_vm.all_vars().len();

        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let _card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        let after = snapshot_varmap(&base_vm);
        assert_eq!(
            base_vm.all_vars().len(),
            base_var_count,
            "base VarMap var count changed"
        );
        assert_eq!(before.len(), after.len(), "base VarMap length changed");
        for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            assert_eq!(
                b, a,
                "base VarMap tensor #{i} drifted through run_lora_ft (must stay frozen)"
            );
        }
    }

    #[test]
    fn run_lora_ft_produces_delta_of_expected_var_count() {
        // TinyLlama tiny: 2 layers × 7 targets × 2 (A + B) = 28
        // tensors. Mirrors
        // `tinyllama_lora_ft.rs::run_lora_ft_tinyllama_saves_delta_with_expected_var_count`.
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        let delta_path = nn_dir
            .join("nn")
            .join(format!("lora-{card_id}.safetensors"));
        let tensors = candle_core::safetensors::load(&delta_path, &candle_core::Device::Cpu)
            .expect("delta safetensors must load");
        assert_eq!(
            tensors.len(),
            28,
            "delta safetensors must contain exactly 28 tensors (2 layers × 7 targets × 2), \
             got {}: keys={:?}",
            tensors.len(),
            tensors.keys().collect::<Vec<_>>()
        );
        for name in tensors.keys() {
            assert!(
                name.ends_with(".lora_a.weight") || name.ends_with(".lora_b.weight"),
                "unexpected non-LoRA key in delta bundle: {name}"
            );
        }
    }

    #[test]
    fn run_lora_ft_wraps_have_lora_true() {
        // Cross-surface: the Card + Δ written by run_lora_ft must
        // be consumable by `alc.nn.card.load_wrap` (L4b), which
        // returns a handle with `is_lora_wrapped()==true`.
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        // Build a fresh base handle and consume the freshly-written
        // LoRA card via load_wrap.
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let fresh_base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("fresh gpt2 base");
        let fresh_ud = lua.create_userdata(fresh_base).unwrap();
        let wrapped = load_wrap_impl(&store, &card_id, &fresh_ud).expect("load_wrap");
        assert!(
            wrapped.is_lora_wrapped(),
            "load_wrap of run_lora_ft output must set has_lora=true"
        );
    }

    // ─── Axis D — Card + Δ round-trip through load_wrap ───────────

    #[test]
    fn run_lora_ft_card_roundtrips_through_load_wrap_gpt2() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let fresh_base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("fresh gpt2 base");
        let fresh_ud = lua.create_userdata(fresh_base).unwrap();
        let wrapped = load_wrap_impl(&store, &card_id, &fresh_ud).expect("load_wrap");
        assert_eq!(wrapped.arch(), "gpt2");
        assert!(wrapped.is_lora_wrapped());
    }

    #[test]
    fn run_lora_ft_card_roundtrips_through_load_wrap_tinyllama() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_lora_ft");

        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let fresh_base = build_tinyllama_handle("tinyllama-tiny", Some(&base_opts), &nn_dir)
            .expect("fresh tinyllama base");
        let fresh_ud = lua.create_userdata(fresh_base).unwrap();
        let wrapped = load_wrap_impl(&store, &card_id, &fresh_ud).expect("load_wrap");
        assert_eq!(wrapped.arch(), "tinyllama");
        assert!(wrapped.is_lora_wrapped());
    }

    // ─── L5c S1 — `alc.nn.trainer.run_full_ft` bridge tests ────────
    //
    // Axis mirrors the run_lora_ft coverage above:
    // - A1/A2: GPT-2 + TinyLlama happy paths (arch dispatch,
    //   on-disk safetensors, Card `training_path="full_ft"` shape).
    // - B1: config schema refusal unique to run_full_ft
    //   (`steps == 0`). LoRA-specific refusals do not apply.
    // - B2: refuse pretrained=true handle (LoRA path is silent
    //   about this because it wraps the base regardless; full-FT
    //   needs the base VarMap).
    // - C1: refuse LoRA-wrapped handle (design divergence
    //   bullet — a wrapped handle would silently disagree with
    //   the caller's intent).
    // - C2: refuse Llama (inference-only, symmetric to run_lora_ft
    //   Axis handling).

    fn base_full_ft_opts() -> serde_json::Value {
        json!({
            "lr": 5e-3,
            "batch": 1,
            "steps": 3,
            "warmup": 0,
            "schedule": "CosineWithWarmup",
        })
    }

    #[test]
    fn run_full_ft_gpt2_happy_path_writes_full_ft_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let (card_id, _candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft");

        // Safetensors on disk at `<nn_dir>/<card_id>.safetensors`
        // (the `save_final` convention pinned by `run_full_ft`'s
        // `ckpt_prefix = card_id`).
        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(
            ckpt_path.exists(),
            "full-ft safetensors must exist at {ckpt_path:?}"
        );

        // Card metadata: training_path / architecture / bundle_ref /
        // no LoRA branch.
        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(
            nn.get("training_path").unwrap().as_str().unwrap(),
            "full_ft"
        );
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "gpt2-tiny"
        );
        let candle = nn.get("candle").unwrap();
        assert_eq!(
            candle.get("bundle_ref").unwrap().as_str().unwrap(),
            format!("nn/{card_id}")
        );
        // `candle.lora` is either absent (skipped by serde) or `null`
        // — full-fine-tune must not emit a LoRA branch.
        let lora = candle.get("lora");
        assert!(
            lora.is_none() || lora.unwrap().is_null(),
            "full-ft Card must not carry a LoRA branch; got: {lora:?}"
        );
    }

    // ─── Channel routing ──────────────────────────────────────────

    /// Build a custom `gpt2` handle carrying `cond_slots` rows of a
    /// conditioning table, on the same tiny shape the scaffolds use.
    fn setup_conditioned_gpt2_scaffold(
        cond_slots: usize,
    ) -> (tempfile::TempDir, FileCardStore, PathBuf, Gpt2Handle, Lua) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = FileCardStore::new(tmp.path().join("cards"));
        let nn_dir = tmp.path().join("nn");
        let lua = Lua::new();
        let base_opts = opts_table(
            &lua,
            json!({ "pretrained": false, "cond_slots": cond_slots }),
        );
        let base = build_gpt2_handle("custom", Some(&base_opts), &nn_dir)
            .expect("build conditioned gpt2 custom base");
        (tmp, store, nn_dir, base, lua)
    }

    /// A dataset whose rows each carry one condition, cycling through
    /// `cond_slots` so no single row's condition explains the corpus.
    fn make_conditioned_dataset_handle(
        lua: &Lua,
        row: Vec<u32>,
        n: usize,
        cond_slots: usize,
    ) -> LuaAnyUserData {
        let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(n).collect();
        let conds: Vec<algocline_nn::arch::CondIndex> = (0..n)
            .map(|i| {
                algocline_nn::arch::CondIndex::new((i % cond_slots) as u32, cond_slots)
                    .expect("slot index inside the table")
            })
            .collect();
        let dopts = DatasetOpts {
            batch_size: 1,
            ctx_len: 16,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        };
        let ds = TokenizedDataset::new(rows, dopts)
            .with_conditions(conds)
            .expect("attach conditions");
        let handle = DatasetHandle::for_test(Box::new(ds), "test-conditioned".into(), 1, 16);
        lua.create_userdata(handle).expect("dataset userdata")
    }

    /// A model built with a conditioning table has no forward pass
    /// that ignores it, so `run_full_ft` has to take the conditioned
    /// entry point. The bundle carrying `cond_wte.weight` is what
    /// shows it did: the plain entry point would have refused the
    /// batch outright, and a model built without the table would have
    /// written a bundle without one.
    #[test]
    fn run_full_ft_routes_a_conditioned_model_through_the_conditioned_entry() {
        let (_tmp, store, nn_dir, base, lua) = setup_conditioned_gpt2_scaffold(2);
        let ds_ud = make_conditioned_dataset_handle(&lua, overfit_row(), 20, 2);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let (card_id, _candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft over a conditioned model");

        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(ckpt_path.exists(), "bundle must exist at {ckpt_path:?}");

        // The Card declares the channel, and the bundle carries its
        // table — the pair the load path checks against each other.
        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        let spec = nn
            .get("candle")
            .and_then(|c| c.get("custom"))
            .and_then(|c| c.get("spec"))
            .expect("custom spec on the card");
        assert_eq!(spec.get("cond_slots").unwrap().as_u64().unwrap(), 2);

        let meta: algocline_nn::card::NnCardMeta =
            serde_json::from_value(nn.clone()).expect("card metadata");
        meta.verify_channel_tensors_in_bundle(&ckpt_path)
            .expect("the written bundle must agree with the written declaration");
    }

    /// The same model against a corpus that carries no conditions.
    /// Refused rather than trained unconditioned, which would write a
    /// checkpoint labelled as conditioned.
    #[test]
    fn run_full_ft_refuses_a_conditioned_model_over_an_unconditioned_corpus() {
        let (_tmp, store, nn_dir, base, lua) = setup_conditioned_gpt2_scaffold(2);
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let err = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect_err("a conditioned model over an unconditioned corpus");
        let msg = err.to_string();
        assert!(
            msg.contains(RUN_FULL_FT_ERR_PREFIX) && msg.contains("carried no conditions"),
            "message must name the surface and the disagreement: {msg}"
        );
    }

    /// The routing decision reads the same branch the Card records, so
    /// the two cannot disagree. A reference-architecture run has no
    /// branch at all and keeps the plain entry point.
    #[test]
    fn trained_channel_reads_the_recorded_branch() {
        assert_eq!(trained_channel(None), TrainedChannel::None);

        let branch = |spec: serde_json::Value| -> NnCustomBranch {
            serde_json::from_value(json!({
                "vocab": 64, "ctx": 16, "layers": 2, "heads": 2, "dim": 32,
                "spec": spec
            }))
            .expect("custom branch")
        };
        assert_eq!(
            trained_channel(Some(&branch(json!({})))),
            TrainedChannel::None
        );
        assert_eq!(
            trained_channel(Some(&branch(json!({ "cond_slots": 3 })))),
            TrainedChannel::Conditioning
        );
        assert_eq!(
            trained_channel(Some(&branch(json!({ "allowed_input": true })))),
            TrainedChannel::Allowed
        );
    }

    #[test]
    fn run_full_ft_tinyllama_happy_path_writes_full_ft_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let (card_id, _candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft");

        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(ckpt_path.exists());

        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(
            nn.get("training_path").unwrap().as_str().unwrap(),
            "full_ft"
        );
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "tinyllama-tiny"
        );
    }

    #[test]
    fn run_full_ft_refuses_zero_steps() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let mut o = base_full_ft_opts();
        o["steps"] = json!(0);
        let opts = opts_table(&lua, o);
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("opts.steps"),
            "expected steps error, got: {msg}"
        );
    }

    #[test]
    fn run_full_ft_refuses_lora_wrapped_handle() {
        // Wrap a base first via run_lora_ft (the natural way to get
        // a wrapped handle round-tripped through the bridge). Then
        // reload as a wrapped handle via load_wrap and confirm
        // run_full_ft refuses it with the directional error.
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let lora_opts = opts_table(&lua, base_train_opts());
        let lora_card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            lora_opts,
        )
        .expect("run_lora_ft");

        // Reload the LoRA card as a wrapped handle.
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let fresh_base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("fresh gpt2 base");
        let fresh_ud = lua.create_userdata(fresh_base).unwrap();
        let wrapped = load_wrap_impl(&store, &lora_card_id, &fresh_ud).expect("load_wrap");
        assert!(wrapped.is_lora_wrapped());

        // Feed the wrapped handle into run_full_ft; must refuse.
        let wrapped_ud = lua.create_userdata(wrapped).unwrap();
        let ds_ud2 = make_dataset_handle(&lua, overfit_row(), 5);
        let ff_opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(wrapped_ud),
            &LuaValue::UserData(ds_ud2),
            ff_opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("drop the wrap first"),
            "expected wrapped-handle refusal, got: {msg}"
        );
    }

    #[test]
    fn run_full_ft_refuses_pretrained_handle() {
        // Axis B2 — a varmap-less handle (the state a
        // `pretrained = true` build lands in) must be refused with
        // the directional error through the full dispatch path.
        // `for_test_pretrained_like` strips the VarMap off a
        // from-scratch handle so no HF hub download is needed.
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let pretrained_like = base.for_test_pretrained_like();
        let base_ud = lua
            .create_userdata(NnHandle::Gpt2(pretrained_like))
            .unwrap();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("pretrained=true"),
            "expected pretrained refusal, got: {msg}"
        );
    }

    // ─── L5c S1 — `on_ckpt` hook mirror ───────────────────────────
    //
    // The hook first shipped on the Layer 5b sibling
    // `alc.nn.trainer.full_ft` (nn_card.rs) and is mirrored here so the
    // Card-writing surface can drive per-checkpoint evaluation. Axes:
    // - fire path: hook runs at each `ckpt_every` boundary and the
    //   `info.ckpt_path` it receives points at a real file.
    // - early break: `"break"` stops the run and the Card is still
    //   persisted (the trainer finalizes through `save_final`).
    // - error path: a Lua-side `error(...)` surfaces as a loud Lua
    //   error rather than a silent PASS.
    // - no-op guard: `on_ckpt` without a positive `ckpt_every` is
    //   refused up front (the hook could never fire).
    // The `hook = None` regression (a run without `on_ckpt` behaves as
    // before) is carried by the happy-path tests above.

    #[test]
    fn run_full_ft_fires_on_ckpt_hook_at_each_boundary() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let mut o = base_full_ft_opts();
        o["steps"] = json!(4);
        o["ckpt_every"] = json!(2);
        // Keep every rotating checkpoint so the paths handed to the
        // hook are still on disk when the assertions run.
        o["ckpt_keep"] = json!(5);
        let opts = opts_table(&lua, o);

        let fires: Arc<Mutex<Vec<(i64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&fires);
        let on_ckpt = lua
            .create_function(move |_, info: LuaTable| -> LuaResult<()> {
                let step: i64 = info.get("step")?;
                let path: String = info.get("ckpt_path")?;
                // Returning nil maps to `CkptControl::Continue`.
                sink.lock().expect("hook sink").push((step, path));
                Ok(())
            })
            .expect("create on_ckpt");
        opts.set("on_ckpt", on_ckpt).unwrap();

        let (card_id, _candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft with on_ckpt");

        let fired = fires.lock().expect("hook sink");
        let steps: Vec<i64> = fired.iter().map(|(s, _)| *s).collect();
        assert_eq!(
            steps,
            vec![2, 4],
            "hook must fire once per ckpt_every boundary (steps=4, ckpt_every=2)"
        );
        for (step, path) in fired.iter() {
            assert!(
                std::path::Path::new(path).exists(),
                "info.ckpt_path for step {step} must point at a written file: {path}"
            );
        }

        // The Card is still written after a full (non-break) run.
        assert!(
            store.get(&card_id).unwrap().is_some(),
            "run_full_ft must persist the Card even with a hook attached"
        );
    }

    /// The Lua surface end to end: a hook that returns `"keep"` gets
    /// its checkpoint held out of the rotation, and the second return
    /// value describes what was held.
    ///
    /// `ckpt_keep = 1` is the point — the step-2 file is what a
    /// rotation at step 4 would otherwise delete, so a caller reading
    /// `candidates[1].ckpt_path` would be holding a dead path.
    #[test]
    fn run_full_ft_keep_returns_a_candidate_whose_file_survives() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let mut o = base_full_ft_opts();
        o["steps"] = json!(4);
        o["ckpt_every"] = json!(2);
        o["ckpt_keep"] = json!(1);
        let opts = opts_table(&lua, o);

        // Keep the first boundary (step 2) and let step 4 pass.
        let on_ckpt = lua
            .create_function(move |_, info: LuaTable| -> LuaResult<Option<String>> {
                let step: i64 = info.get("step")?;
                Ok((step == 2).then(|| "keep".to_string()))
            })
            .expect("create on_ckpt");
        opts.set("on_ckpt", on_ckpt).unwrap();

        let (card_id, candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft with a keeping on_ckpt");

        assert_eq!(candidates.raw_len(), 1, "one boundary asked to be kept");
        let entry: LuaTable = candidates.get(1).unwrap();
        assert_eq!(entry.get::<i64>("step").unwrap(), 2);
        let path: String = entry.get("ckpt_path").unwrap();
        assert!(
            std::path::Path::new(&path).exists(),
            "the kept checkpoint must survive the step-4 rotation at ckpt_keep=1: {path}"
        );
        assert!(entry.get::<f32>("train_loss").unwrap().is_finite());
        assert_eq!(
            entry.get::<Option<String>>("reason").unwrap(),
            None,
            "a bare \"keep\" carries no reason"
        );

        assert!(
            store.get(&card_id).unwrap().is_some(),
            "keeping a checkpoint must not disturb the terminal Card"
        );
    }

    /// A run whose hook never keeps anything returns an empty
    /// candidates table — `local id = run_full_ft(…)` keeps working
    /// untouched.
    #[test]
    fn run_full_ft_without_keeps_returns_an_empty_candidate_list() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let mut o = base_full_ft_opts();
        o["steps"] = json!(2);
        let opts = opts_table(&lua, o);

        let (card_id, candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft without a hook");

        assert_eq!(candidates.raw_len(), 0);
        assert!(store.get(&card_id).unwrap().is_some());
    }

    #[test]
    fn run_full_ft_on_ckpt_break_stops_early_and_persists_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let mut o = base_full_ft_opts();
        o["steps"] = json!(6);
        o["ckpt_every"] = json!(2);
        o["ckpt_keep"] = json!(5);
        let opts = opts_table(&lua, o);

        let fires: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let counter = Arc::clone(&fires);
        let on_ckpt = lua
            .create_function(move |lua, _info: LuaTable| -> LuaResult<LuaValue> {
                let n = {
                    let mut g = counter.lock().expect("hook counter");
                    *g += 1;
                    *g
                };
                if n >= 2 {
                    Ok(LuaValue::String(lua.create_string("break")?))
                } else {
                    Ok(LuaValue::Nil)
                }
            })
            .expect("create on_ckpt");
        opts.set("on_ckpt", on_ckpt).unwrap();

        let (card_id, _candidates) = run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_full_ft with early-breaking on_ckpt");

        assert_eq!(
            *fires.lock().expect("hook counter"),
            2,
            "hook must fire twice (steps 2 and 4) and then break"
        );

        // Terminal safetensors + Card are written on the early-break
        // path too (the trainer finalizes through `save_final`).
        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(
            ckpt_path.exists(),
            "early break must still write {ckpt_path:?}"
        );
        let card = store
            .get(&card_id)
            .unwrap()
            .expect("Card must be persisted");
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(
            nn.get("training_path").unwrap().as_str().unwrap(),
            "full_ft"
        );
        // The recorded step is the break boundary (4), not the
        // requested `steps` (6) — proof the run stopped early.
        assert_eq!(
            nn.get("metrics")
                .and_then(|m| m.get("step"))
                .and_then(|s| s.as_u64())
                .unwrap(),
            4,
            "Card must record the break step, not the requested step count"
        );
    }

    #[test]
    fn run_full_ft_on_ckpt_error_propagates_as_lua_error() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let mut o = base_full_ft_opts();
        o["steps"] = json!(4);
        o["ckpt_every"] = json!(2);
        let opts = opts_table(&lua, o);

        // Lua-side `error(...)` (the shape a real callback raises)
        // rather than a Rust-constructed LuaError.
        let on_ckpt: LuaFunction = lua
            .load(r#"return function(_info) error("hook exploded") end"#)
            .eval()
            .expect("load on_ckpt");
        opts.set("on_ckpt", on_ckpt).unwrap();

        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:")
                && msg.contains("on_ckpt callback failed")
                && msg.contains("hook exploded"),
            "hook error must surface loudly with the surface prefix, got: {msg}"
        );
    }

    #[test]
    fn run_full_ft_refuses_on_ckpt_without_ckpt_every() {
        // Silent-no-op guard: `ckpt_every` defaults to 0 (mid-run
        // checkpoints disabled), so a hook supplied without it would
        // never fire. Refused before training starts.
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let on_ckpt = lua
            .create_function(|_, _info: LuaTable| -> LuaResult<()> { Ok(()) })
            .expect("create on_ckpt");
        opts.set("on_ckpt", on_ckpt).unwrap();

        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("opts.ckpt_every > 0"),
            "expected on_ckpt/ckpt_every pairing refusal, got: {msg}"
        );
    }

    // ─── L5c S2 — `alc.nn.trainer.run_distill` bridge tests ────────
    //
    // Axis mirrors the run_full_ft coverage above:
    // - A1/A2: GPT-2 + TinyLlama happy paths (arch dispatch,
    //   on-disk safetensors, Card `training_path="distillation"`
    //   shape + `loss_kind` in hyperparams).
    // - B1: refuse unknown `loss_kind` (the schema field unique to
    //   this surface).
    // - B2: refuse pretrained=true (varmap-less) student handle.

    #[test]
    fn run_distill_gpt2_happy_path_writes_distillation_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let card_id = run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_distill");

        // Safetensors on disk at `<nn_dir>/<card_id>.safetensors`
        // (run_distill forwards to run_full_ft, same `save_final`
        // convention).
        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(
            ckpt_path.exists(),
            "distill safetensors must exist at {ckpt_path:?}"
        );

        // Card metadata: training_path / architecture / loss_kind /
        // no LoRA branch.
        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(
            nn.get("training_path").unwrap().as_str().unwrap(),
            "distillation"
        );
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "gpt2-tiny"
        );
        assert_eq!(
            nn.get("hyperparams")
                .and_then(|h| h.get("loss_kind"))
                .and_then(|l| l.as_str())
                .unwrap(),
            "ce"
        );
        let candle = nn.get("candle").unwrap();
        assert_eq!(
            candle.get("bundle_ref").unwrap().as_str().unwrap(),
            format!("nn/{card_id}")
        );
        let lora = candle.get("lora");
        assert!(
            lora.is_none() || lora.unwrap().is_null(),
            "distillation Card must not carry a LoRA branch; got: {lora:?}"
        );
    }

    #[test]
    fn run_distill_tinyllama_happy_path_writes_distillation_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let card_id = run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        )
        .expect("run_distill");

        let ckpt_path = nn_dir.join(format!("{card_id}.safetensors"));
        assert!(ckpt_path.exists());

        let card = store.get(&card_id).unwrap().unwrap();
        let nn = card.get("metadata").and_then(|m| m.get("nn")).unwrap();
        assert_eq!(
            nn.get("training_path").unwrap().as_str().unwrap(),
            "distillation"
        );
        assert_eq!(
            nn.get("architecture").unwrap().as_str().unwrap(),
            "tinyllama-tiny"
        );
    }

    #[test]
    fn run_distill_refuses_unknown_loss_kind() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let mut o = base_full_ft_opts();
        o["loss_kind"] = json!("kl");
        let opts = opts_table(&lua, o);
        let msg = expect_err(run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_distill:")
                && msg.contains("loss_kind")
                && msg.contains("kl"),
            "expected loss_kind error, got: {msg}"
        );
    }

    #[test]
    fn run_distill_refuses_pretrained_handle() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let pretrained_like = base.for_test_pretrained_like();
        let base_ud = lua
            .create_userdata(NnHandle::Gpt2(pretrained_like))
            .unwrap();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_distill:") && msg.contains("pretrained=true"),
            "expected pretrained refusal, got: {msg}"
        );
    }

    // ─── L5c S3 — trainer bind test hardening ──────────────────────
    //
    // Closes two coverage gaps carried from S2:
    // - run_distill's LoRA-wrapped refusal guard (nn_trainer.rs
    //   §3) had no test; the run_full_ft sibling
    //   (`run_full_ft_refuses_lora_wrapped_handle`) was already
    //   covered.
    // - pretrained refusal was only exercised on the GPT-2 arm; the
    //   TinyLlama arm hits the same guard but lacked a test
    //   constructor (`TinyLlamaHandle::for_test_pretrained_like`).

    #[test]
    fn run_distill_refuses_lora_wrapped_handle() {
        // Mirror of `run_full_ft_refuses_lora_wrapped_handle` swapped
        // onto the run_distill dispatch: distillation trains the base
        // parameters, so a LoRA-wrapped handle must be refused with the
        // directional error.
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base)).unwrap();
        let lora_opts = opts_table(&lua, base_train_opts());
        let lora_card_id = run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            lora_opts,
        )
        .expect("run_lora_ft");

        // Reload the LoRA card as a wrapped handle.
        let base_opts = opts_table(&lua, json!({ "pretrained": false }));
        let fresh_base =
            build_gpt2_handle("tiny", Some(&base_opts), &nn_dir).expect("fresh gpt2 base");
        let fresh_ud = lua.create_userdata(fresh_base).unwrap();
        let wrapped = load_wrap_impl(&store, &lora_card_id, &fresh_ud).expect("load_wrap");
        assert!(wrapped.is_lora_wrapped());

        // Feed the wrapped handle into run_distill; must refuse.
        let wrapped_ud = lua.create_userdata(wrapped).unwrap();
        let ds_ud2 = make_dataset_handle(&lua, overfit_row(), 5);
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(wrapped_ud),
            &LuaValue::UserData(ds_ud2),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_distill:") && msg.contains("drop the wrap first"),
            "expected wrapped-handle refusal, got: {msg}"
        );
    }

    #[test]
    fn run_full_ft_refuses_pretrained_handle_tinyllama() {
        // TinyLlama mirror of `run_full_ft_refuses_pretrained_handle`
        // (Axis B2): a varmap-less handle (the state a
        // `pretrained = true` build lands in) must be refused with the
        // directional error through the full dispatch path.
        // `for_test_pretrained_like` strips the VarMap off a
        // from-scratch handle so no HF hub download is needed.
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let pretrained_like = base.for_test_pretrained_like();
        let base_ud = lua
            .create_userdata(NnHandle::TinyLlama(pretrained_like))
            .unwrap();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("pretrained=true"),
            "expected pretrained refusal, got: {msg}"
        );
    }

    #[test]
    fn run_distill_refuses_pretrained_handle_tinyllama() {
        // TinyLlama mirror of `run_distill_refuses_pretrained_handle`.
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let pretrained_like = base.for_test_pretrained_like();
        let base_ud = lua
            .create_userdata(NnHandle::TinyLlama(pretrained_like))
            .unwrap();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 5);
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_distill:") && msg.contains("pretrained=true"),
            "expected pretrained refusal, got: {msg}"
        );
    }

    // ─── f16 base handle guard (step 4.5) ─────────────────────────
    //
    // Sibling coverage to `wrap_lora_bridge_tests::wrap_lora_rejects_f16_base_*`.
    // The f16 guard fires between the Llama refusal (step 4) and
    // the dataset downcast (step 5), so these tests pass
    // `LuaValue::Nil` for the dataset — the guard errors before the
    // dataset is even inspected. BF16 passes this guard (the mixed-
    // precision trainer path handles it) — pinned by the bf16 test
    // below, which must fail *later* than the dtype guard.

    fn assert_f16_refusal(prefix: &str, msg: &str) {
        assert!(
            msg.contains(prefix)
                && msg.contains("training does not support an f16 base")
                && msg.contains(r#"dtype="bf16""#),
            "expected f16 refusal from {prefix}, got: {msg}"
        );
    }

    #[test]
    fn run_lora_ft_rejects_f16_base() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let base_f16 = gpt2_handle_with_dtype(base, "f16");
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base_f16)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let msg = expect_err(run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::Nil,
            opts,
        ));
        assert_f16_refusal("alc.nn.trainer.run_lora_ft:", &msg);
    }

    #[test]
    fn run_lora_ft_rejects_f16_base_case_insensitive_tinyllama() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        // Upstream may stringify f16 as `"F16"` (uppercase). The
        // guard must fire regardless of case.
        let base_f16 = tinyllama_handle_with_dtype(base, "F16");
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base_f16)).unwrap();
        let opts = opts_table(&lua, base_train_opts());
        let msg = expect_err(run_lora_ft_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::Nil,
            opts,
        ));
        assert_f16_refusal("alc.nn.trainer.run_lora_ft:", &msg);
    }

    #[test]
    fn run_full_ft_rejects_f16_base() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let base_f16 = gpt2_handle_with_dtype(base, "f16");
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base_f16)).unwrap();
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::Nil,
            opts,
        ));
        assert_f16_refusal("alc.nn.trainer.run_full_ft:", &msg);
    }

    #[test]
    fn run_distill_rejects_f16_base() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let base_f16 = gpt2_handle_with_dtype(base, "f16");
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base_f16)).unwrap();
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_distill_impl(
            &store,
            &nn_dir,
            &LuaValue::UserData(base_ud),
            &LuaValue::Nil,
            opts,
        ));
        assert_f16_refusal("alc.nn.trainer.run_distill:", &msg);
    }

    /// BF16 must pass the dtype guard: with a `Nil` dataset the run
    /// proceeds past step 4.5 and fails at the dataset downcast
    /// (step 5) instead — proving the guard no longer fires on bf16.
    #[test]
    fn run_full_ft_bf16_base_passes_dtype_guard() {
        let (_tmp, store, nn_dir, base, lua) = setup_gpt2_scaffold();
        let base_bf16 = gpt2_handle_with_dtype(base, "bf16");
        let base_ud = lua.create_userdata(NnHandle::Gpt2(base_bf16)).unwrap();
        let opts = opts_table(&lua, base_full_ft_opts());
        let msg = expect_err(run_full_ft_impl(
            &store,
            &nn_dir,
            &lua,
            &LuaValue::UserData(base_ud),
            &LuaValue::Nil,
            opts,
        ));
        assert!(
            !msg.contains("f16 base") && !msg.contains("f32 base"),
            "bf16 must not trip the dtype guard, got: {msg}"
        );
    }
}
