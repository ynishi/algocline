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
//! alc.nn.trainer.run_full_ft(base_handle, dataset, opts) -> card_id
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
//! 4. `grad_accum_steps` / `ckpt_every` are NOT exposed and remain
//!    pinned to their crate defaults (design §1.2 explicit
//!    non-exposure — multi-step accumulation returns
//!    `GradAccumUnsupported` today; rotating checkpoint is out of
//!    scope for the Lua bridge).
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

use algocline_nn::arch::{LoraConfig, TinyLlamaModel};
use algocline_nn::card::{NnCandleBranch, NnCardMeta, NnLineage, NnLoraBranch};
use algocline_nn::train::{
    run_distill, run_full_ft, run_lora_ft, CrossEntropyLoss, DistillLossKind, DistillSpec,
    FullFtConfig, TrainError, TrainingLease,
};
use mlua::prelude::*;
use serde_json::json;

use crate::card::FileCardStore;

use super::nn_card::{
    build_create_payload_from_meta, compact_epoch_us, extract_full_ft_opts, sanitize_name,
    DatasetHandle, Gpt2Handle, LlamaHandle, NnHandle, TinyLlamaHandle,
};

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
        move |_lua, (base, dataset, opts): (LuaValue, LuaValue, LuaTable)| -> LuaResult<String> {
            run_full_ft_impl(&store_ff, &dir_ff, &base, &dataset, opts)
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
    let lora_cfg = extract_lora_cfg(&opts, arch)?;
    let train_cfg = extract_train_cfg(&opts)?;

    // 7. Pre-generate lora_card_id (mirrors save_impl /
    //    merge_lora_impl). Sanitize the user-supplied name if
    //    provided; otherwise use "run_lora_ft" as the base.
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_lora_ft");
    let lora_card_id = format!("{}_{}", sanitize_name(name_base), compact_epoch_us());

    // 8. Derive base_bundle_ref BEFORE any lock is acquired —
    //    `arch_family_variant()` needs only immutable handle
    //    fields, and threading it through here keeps the field
    //    lookup outside the training-loop critical section.
    let base_bundle_ref = format!("nn/{}", handle.arch_family_variant());
    let architecture = handle.arch_family_variant();

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
                &lora_card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua)?
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
                &lora_card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua)?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the
            // assertion rather than silently invoking a non-existent
            // train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed NnCardMeta payload and persist the Card.
    //     `ckpt.bundle_ref` is the trailing filename
    //     `"lora-<card_id>.safetensors"` (Layer 3 convention); the
    //     absolute path is `<nn_dir>/nn/<ckpt.bundle_ref>`.
    let delta_path = nn_dir.join("nn").join(&ckpt.bundle_ref);
    let delta_path_string = delta_path.to_string_lossy().to_string();

    // LoRA branch values come from the validated `LoraConfig` +
    // resolved `delta_path` + derived `base_bundle_ref`. `rank` /
    // `alpha` are `u32` on `NnLoraBranch` while `LoraConfig` stores
    // them as `usize` / `f32`; convert defensively.
    let lora_branch = NnLoraBranch {
        rank: lora_cfg.rank as u32,
        alpha: lora_cfg.alpha as u32,
        base_bundle_ref: base_bundle_ref.clone(),
        target_modules: lora_cfg.target_modules.clone(),
        dropout: lora_cfg.dropout,
        delta_path: Some(delta_path_string.clone()),
    };

    let candle = NnCandleBranch {
        bundle_ref: format!("nn/{lora_card_id}"),
        device: None,
        dtype: None,
        lora: Some(lora_branch),
    };

    let meta = NnCardMeta {
        name: name_base.to_string(),
        backend: "candle".into(),
        task: None,
        architecture,
        training_path: "lora".into(),
        lineage: NnLineage::default(),
        hyperparams: json!({
            "lr": train_cfg.lr,
            "batch": train_cfg.batch_size,
            "steps": train_cfg.steps,
            "warmup": train_cfg.warmup,
        }),
        metrics: json!({
            "train_loss": ckpt.train_loss,
            "step": ckpt.step,
        }),
        candle: Some(candle),
    };

    let payload = build_create_payload_from_meta(&lora_card_id, &meta)?;

    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_lora_ft: card store: {e}")))?;

    if returned_id != lora_card_id {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_lora_ft: card_id mismatch (expected \
             {lora_card_id}, got {returned_id})"
        )));
    }

    Ok(lora_card_id)
}

/// Extract a validated [`LoraConfig`] from `opts`. Mirrors the S1
/// wrap-side validation (`nn_wrap.rs`) but under the trainer-side
/// error prefix — sharing the extractor across surfaces would tie
/// them to a single prefix, which defeats the loud-error contract.
fn extract_lora_cfg(opts: &LuaTable, arch: &str) -> LuaResult<LoraConfig> {
    let rank: Option<i64> = opts.get("rank")?;
    let rank = rank.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_lora_ft: opts.rank must be a positive integer")
    })? as usize;

    let alpha: Option<f64> = opts.get("alpha")?;
    let alpha = alpha.filter(|v| *v > 0.0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_lora_ft: opts.alpha must be a positive number")
    })? as f32;

    let dropout: Option<f64> = opts.get("dropout")?;
    let dropout = dropout.unwrap_or(0.0);
    if !(0.0..1.0).contains(&dropout) {
        return Err(LuaError::external(
            "alc.nn.trainer.run_lora_ft: opts.dropout must be in [0.0, 1.0)",
        ));
    }
    let dropout = dropout as f32;

    // Per-arch canonical target set (SoT lives in the algocline-nn
    // crate; the bridge only routes, per L5b design §1.3). Same
    // logic as `nn_wrap::canonical_targets_for` — duplicated 4-line
    // match keeps this file self-contained.
    let known = canonical_targets_for(arch).ok_or_else(|| {
        LuaError::external(format!(
            "alc.nn.trainer.run_lora_ft: architecture {arch} is not LoRA-wrappable \
             (only gpt2 / tinyllama families are supported)"
        ))
    })?;

    let raw: LuaValue = opts.get("target_modules")?;
    let target_modules: Vec<String> = match raw {
        LuaValue::Nil => known.clone(),
        LuaValue::Table(tbl) => {
            let entries: Vec<String> = tbl
                .sequence_values::<String>()
                .collect::<LuaResult<Vec<_>>>()?;
            if entries.is_empty() {
                return Err(LuaError::external(
                    "alc.nn.trainer.run_lora_ft: opts.target_modules must be non-empty \
                     (or nil for the per-arch default)",
                ));
            }
            for entry in &entries {
                if !known.iter().any(|k| k == entry) {
                    let known_list = known.join(", ");
                    return Err(LuaError::external(format!(
                        "alc.nn.trainer.run_lora_ft: unknown target module {entry:?} \
                         for arch {arch} (known: [{known_list}])"
                    )));
                }
            }
            entries
        }
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_lora_ft: opts.target_modules must be an array of strings \
                 (or nil for the per-arch default); got {}",
                other.type_name()
            )));
        }
    };

    let mut cfg = LoraConfig::with_targets(rank, alpha, target_modules);
    cfg.dropout = dropout;
    Ok(cfg)
}

/// Extract a validated [`FullFtConfig`] from `opts`.
///
/// Design §1.2 pins the exposed field set to
/// `lr / batch / steps / warmup / schedule`. Validation is
/// pre-flight (`lr > 0`, `batch > 0`, `steps > 0`, `warmup >= 0`,
/// `schedule` in `{CosineWithWarmup, Constant}`) so a misconfigured
/// caller sees a Lua-shaped error before the training loop is
/// entered. `grad_accum_steps` and `ckpt_every` are NOT exposed
/// (they stay at [`FullFtConfig::default`] values: `grad_accum = 1`,
/// `ckpt_every = 0`).
///
/// After validation the design §1.2 field names are canonicalised
/// into the sibling extractor's naming convention
/// (`batch` → `batch_size`, `"CosineWithWarmup"` →
/// `"cosine_with_warmup"`, `"Constant"` → `"constant"`) and the
/// shared [`extract_full_ft_opts`] does the final `FullFtConfig`
/// assembly. That keeps a single SoT for the training-config
/// schema — a future field addition to `FullFtConfig` gets picked
/// up here automatically via the shared extractor.
fn extract_train_cfg(opts: &LuaTable) -> LuaResult<FullFtConfig> {
    let lr: Option<f64> = opts.get("lr")?;
    let lr = lr.filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_lora_ft: opts.lr must be a positive number")
    })?;

    let batch: Option<i64> = opts.get("batch")?;
    let batch = batch.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_lora_ft: opts.batch must be a positive integer")
    })? as usize;

    let steps: Option<i64> = opts.get("steps")?;
    let steps = steps.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_lora_ft: opts.steps must be a positive integer")
    })? as usize;

    // Optional warmup — must be >= 0 (integer). i64 lets a negative
    // value surface the design §2.2 loud error rather than silently
    // clamp to 0.
    let warmup = match opts.get::<Option<i64>>("warmup")? {
        Some(v) if v < 0 => {
            return Err(LuaError::external(
                "alc.nn.trainer.run_lora_ft: opts.warmup must be >= 0",
            ));
        }
        Some(v) => v as usize,
        None => 0,
    };

    // Optional schedule — design §1.2 default "CosineWithWarmup".
    // Anything outside the two-value enum is refused with the
    // caller-supplied value in the error message so a typo is
    // debuggable at the surface.
    let schedule_canonical = match opts
        .get::<Option<String>>("schedule")?
        .as_deref()
        .unwrap_or("CosineWithWarmup")
    {
        "CosineWithWarmup" => "cosine_with_warmup",
        "Constant" => "constant",
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_lora_ft: opts.schedule must be one of \
                 \"CosineWithWarmup\" / \"Constant\" (got {other:?})"
            )));
        }
    };

    // Canonicalise design §1.2 field names into the shared
    // extractor's naming convention, then delegate. Mutating
    // `opts` in place is safe: the caller-supplied table is a
    // per-invocation value, no other closure retains it.
    opts.set("lr", lr)?;
    opts.set("batch_size", batch as i64)?;
    opts.set("steps", steps as i64)?;
    opts.set("warmup", warmup as i64)?;
    opts.set("schedule", schedule_canonical.to_string())?;

    extract_full_ft_opts(Some(opts))
}

/// Return the arch's canonical LoRA target-module set. Same source
/// as [`super::nn_wrap`] — the Rust `algocline-nn` crate owns the
/// SoT; the bridge only routes.
fn canonical_targets_for(arch: &str) -> Option<Vec<String>> {
    match arch {
        "gpt2" => Some(LoraConfig::default_targets()),
        "tinyllama" => Some(TinyLlamaModel::default_lora_targets()),
        _ => None,
    }
}

/// Translate a [`TrainError`] into a `run_lora_ft`-prefixed Lua
/// error. Mirrors the shape spelled out in design §2.2.
fn train_err_to_lua(e: TrainError) -> LuaError {
    let msg = match e {
        TrainError::ZeroSteps => "alc.nn.trainer.run_lora_ft: zero steps".to_string(),
        TrainError::LeaseHeld => {
            "alc.nn.trainer.run_lora_ft: training lease already active on this VM".to_string()
        }
        TrainError::DatasetExhausted { seen, requested } => format!(
            "alc.nn.trainer.run_lora_ft: dataset exhausted after {seen} steps \
             (requested {requested})"
        ),
        TrainError::Ckpt(inner) => {
            format!("alc.nn.trainer.run_lora_ft: checkpoint: {inner}")
        }
        TrainError::Candle(inner) => {
            format!("alc.nn.trainer.run_lora_ft: candle: {inner}")
        }
        // The `TrainError` enum is `#[non_exhaustive]`-adjacent (may
        // gain variants); route unknown variants through Display so
        // the caller still sees a loud error rather than a compile
        // break.
        other => format!("alc.nn.trainer.run_lora_ft: {other}"),
    };
    LuaError::external(msg)
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
// - Error prefix — every [`LuaError::external`] emitted from this
//   impl carries the prefix `alc.nn.trainer.run_full_ft:`. Do NOT
//   share the extractor with [`extract_train_cfg`] above:
//   design's loud-error contract pins one prefix per surface.

/// L5c S1 core. Mirrors [`run_lora_ft_impl`] structurally; see the
/// section header above for the design divergence.
fn run_full_ft_impl(
    store: &FileCardStore,
    nn_dir: &std::path::Path,
    base: &LuaValue,
    dataset: &LuaValue,
    opts: LuaTable,
) -> LuaResult<String> {
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
    let train_cfg = extract_train_cfg_ff(&opts)?;

    // 7. Pre-generate card_id (mirrors run_lora_ft_impl step 7).
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_full_ft");
    let card_id = format!("{}_{}", sanitize_name(name_base), compact_epoch_us());

    // 8. Derive architecture BEFORE any lock (immutable handle field
    //    lookup outside the training-loop critical section).
    let architecture = handle.arch_family_variant();

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

            let result = run_full_ft(
                &*model,
                &vm_arc,
                ds_lock.as_mut(),
                &train_cfg,
                &loss_fn,
                nn_dir,
                &card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua_ff)?
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
                &card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua_ff)?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the assertion
            // rather than silently invoking a non-existent train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed NnCardMeta payload and persist the Card.
    //     Full-fine-tune carries NO LoRA branch — that field stays
    //     `None`, matching `training_path="full_ft"` cards written
    //     by the sibling `alc.nn.trainer.full_ft` + `alc.nn.card.save`
    //     flow.
    let candle = NnCandleBranch {
        bundle_ref: format!("nn/{card_id}"),
        device: None,
        dtype: None,
        lora: None,
    };

    let meta = NnCardMeta {
        name: name_base.to_string(),
        backend: "candle".into(),
        task: None,
        architecture,
        training_path: "full_ft".into(),
        lineage: NnLineage::default(),
        hyperparams: json!({
            "lr": train_cfg.lr,
            "batch": train_cfg.batch_size,
            "steps": train_cfg.steps,
            "warmup": train_cfg.warmup,
        }),
        metrics: json!({
            "train_loss": ckpt.train_loss,
            "step": ckpt.step,
        }),
        candle: Some(candle),
    };

    let payload = build_create_payload_from_meta(&card_id, &meta)?;

    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_full_ft: card store: {e}")))?;

    if returned_id != card_id {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_full_ft: card_id mismatch (expected \
             {card_id}, got {returned_id})"
        )));
    }

    // The `ckpt.bundle_ref` field is left unused on purpose: L5c S1
    // consumers only see the Card id, and the on-disk safetensors
    // path is fixed by the `<nn_dir>/<card_id>.safetensors`
    // convention (validated by
    // `run_full_ft_bridge_tests::run_full_ft_gpt2_happy_path_writes_full_ft_card`
    // below).
    let _ = ckpt.bundle_ref;

    Ok(card_id)
}

/// Extract a validated [`FullFtConfig`] from `opts` for the
/// `run_full_ft` surface.
///
/// Same field-set as [`extract_train_cfg`] (`lr` / `batch` /
/// `steps` / `warmup` / `schedule`) with the same validation rules,
/// but every error carries the `alc.nn.trainer.run_full_ft:` prefix
/// per design's loud-error contract (one prefix per surface). The
/// two extractors are intentionally duplicated 60-line-for-60-line
/// so a future divergence of the training-config schema between
/// LoRA and full-fine-tune surfaces stays local to one extractor.
fn extract_train_cfg_ff(opts: &LuaTable) -> LuaResult<FullFtConfig> {
    let lr: Option<f64> = opts.get("lr")?;
    let lr = lr.filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_full_ft: opts.lr must be a positive number")
    })?;

    let batch: Option<i64> = opts.get("batch")?;
    let batch = batch.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_full_ft: opts.batch must be a positive integer")
    })? as usize;

    let steps: Option<i64> = opts.get("steps")?;
    let steps = steps.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_full_ft: opts.steps must be a positive integer")
    })? as usize;

    let warmup = match opts.get::<Option<i64>>("warmup")? {
        Some(v) if v < 0 => {
            return Err(LuaError::external(
                "alc.nn.trainer.run_full_ft: opts.warmup must be >= 0",
            ));
        }
        Some(v) => v as usize,
        None => 0,
    };

    let schedule_canonical = match opts
        .get::<Option<String>>("schedule")?
        .as_deref()
        .unwrap_or("CosineWithWarmup")
    {
        "CosineWithWarmup" => "cosine_with_warmup",
        "Constant" => "constant",
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_full_ft: opts.schedule must be one of \
                 \"CosineWithWarmup\" / \"Constant\" (got {other:?})"
            )));
        }
    };

    // Canonicalise design §1.2 field names into the shared
    // extractor's naming convention, then delegate. Mutating
    // `opts` in place is safe: the caller-supplied table is a
    // per-invocation value.
    opts.set("lr", lr)?;
    opts.set("batch_size", batch as i64)?;
    opts.set("steps", steps as i64)?;
    opts.set("warmup", warmup as i64)?;
    opts.set("schedule", schedule_canonical.to_string())?;

    extract_full_ft_opts(Some(opts))
}

/// Translate a [`TrainError`] into a `run_full_ft`-prefixed Lua
/// error. Mirrors [`train_err_to_lua`] structurally; diverges only
/// on the surface prefix per design's loud-error contract.
fn train_err_to_lua_ff(e: TrainError) -> LuaError {
    let msg = match e {
        TrainError::ZeroSteps => "alc.nn.trainer.run_full_ft: zero steps".to_string(),
        TrainError::LeaseHeld => {
            "alc.nn.trainer.run_full_ft: training lease already active on this VM".to_string()
        }
        TrainError::DatasetExhausted { seen, requested } => format!(
            "alc.nn.trainer.run_full_ft: dataset exhausted after {seen} steps \
             (requested {requested})"
        ),
        TrainError::Ckpt(inner) => {
            format!("alc.nn.trainer.run_full_ft: checkpoint: {inner}")
        }
        TrainError::Candle(inner) => {
            format!("alc.nn.trainer.run_full_ft: candle: {inner}")
        }
        other => format!("alc.nn.trainer.run_full_ft: {other}"),
    };
    LuaError::external(msg)
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
// - Error prefix `alc.nn.trainer.run_distill:` — one prefix per
//   surface per design's loud-error contract; extractors are NOT
//   shared with the siblings above.

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
    let train_cfg = extract_train_cfg_rd(&opts)?;
    let loss_kind = extract_distill_loss_kind_rd(&opts)?;
    let spec = DistillSpec {
        hyperparams: train_cfg,
        loss_kind,
    };

    // 7. Pre-generate card_id (mirrors run_full_ft_impl step 7).
    let name: Option<String> = opts.get("name")?;
    let name_base = name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("run_distill");
    let card_id = format!("{}_{}", sanitize_name(name_base), compact_epoch_us());

    // 8. Derive architecture BEFORE any lock (immutable handle field
    //    lookup outside the training-loop critical section).
    let architecture = handle.arch_family_variant();

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
                &card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua_rd)?
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
                &card_id,
                Arc::clone(&lease),
            );
            drop(model);
            drop(ds_lock);
            drop(ds_handle);

            result.map_err(train_err_to_lua_rd)?
        }
        NnHandle::Llama(_) => {
            // Guarded at step 4 above; kept as `unreachable!` so a
            // future refactor that reorders steps trips the assertion
            // rather than silently invoking a non-existent train path.
            unreachable!("Llama variant guarded above")
        }
    };

    // 11. Build the typed NnCardMeta payload and persist the Card.
    //     Distillation carries NO LoRA branch; `loss_kind` rides on
    //     `hyperparams` so the run's loss selection is auditable
    //     from the Card.
    let candle = NnCandleBranch {
        bundle_ref: format!("nn/{card_id}"),
        device: None,
        dtype: None,
        lora: None,
    };

    let loss_kind_str = match spec.loss_kind {
        DistillLossKind::Ce => "ce",
    };

    let meta = NnCardMeta {
        name: name_base.to_string(),
        backend: "candle".into(),
        task: None,
        architecture,
        training_path: "distillation".into(),
        lineage: NnLineage::default(),
        hyperparams: json!({
            "lr": spec.hyperparams.lr,
            "batch": spec.hyperparams.batch_size,
            "steps": spec.hyperparams.steps,
            "warmup": spec.hyperparams.warmup,
            "loss_kind": loss_kind_str,
        }),
        metrics: json!({
            "train_loss": ckpt.train_loss,
            "step": ckpt.step,
        }),
        candle: Some(candle),
    };

    let payload = build_create_payload_from_meta(&card_id, &meta)?;

    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| LuaError::external(format!("alc.nn.trainer.run_distill: card store: {e}")))?;

    if returned_id != card_id {
        return Err(LuaError::external(format!(
            "alc.nn.trainer.run_distill: card_id mismatch (expected \
             {card_id}, got {returned_id})"
        )));
    }

    // `ckpt.bundle_ref` unused on purpose — same rationale as
    // run_full_ft_impl (consumers only see the Card id; the on-disk
    // path is fixed by the `<nn_dir>/<card_id>.safetensors`
    // convention).
    let _ = ckpt.bundle_ref;

    Ok(card_id)
}

/// Extract a validated [`FullFtConfig`] from `opts` for the
/// `run_distill` surface.
///
/// Same field-set and validation rules as [`extract_train_cfg_ff`],
/// but every error carries the `alc.nn.trainer.run_distill:` prefix
/// per design's loud-error contract (one prefix per surface). The
/// extractors are intentionally duplicated so a future divergence of
/// the training-config schema between surfaces stays local to one
/// extractor.
fn extract_train_cfg_rd(opts: &LuaTable) -> LuaResult<FullFtConfig> {
    let lr: Option<f64> = opts.get("lr")?;
    let lr = lr.filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_distill: opts.lr must be a positive number")
    })?;

    let batch: Option<i64> = opts.get("batch")?;
    let batch = batch.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_distill: opts.batch must be a positive integer")
    })? as usize;

    let steps: Option<i64> = opts.get("steps")?;
    let steps = steps.filter(|v| *v > 0).ok_or_else(|| {
        LuaError::external("alc.nn.trainer.run_distill: opts.steps must be a positive integer")
    })? as usize;

    let warmup = match opts.get::<Option<i64>>("warmup")? {
        Some(v) if v < 0 => {
            return Err(LuaError::external(
                "alc.nn.trainer.run_distill: opts.warmup must be >= 0",
            ));
        }
        Some(v) => v as usize,
        None => 0,
    };

    let schedule_canonical = match opts
        .get::<Option<String>>("schedule")?
        .as_deref()
        .unwrap_or("CosineWithWarmup")
    {
        "CosineWithWarmup" => "cosine_with_warmup",
        "Constant" => "constant",
        other => {
            return Err(LuaError::external(format!(
                "alc.nn.trainer.run_distill: opts.schedule must be one of \
                 \"CosineWithWarmup\" / \"Constant\" (got {other:?})"
            )));
        }
    };

    // Canonicalise design §1.2 field names into the shared
    // extractor's naming convention, then delegate. Mutating
    // `opts` in place is safe: the caller-supplied table is a
    // per-invocation value.
    opts.set("lr", lr)?;
    opts.set("batch_size", batch as i64)?;
    opts.set("steps", steps as i64)?;
    opts.set("warmup", warmup as i64)?;
    opts.set("schedule", schedule_canonical.to_string())?;

    extract_full_ft_opts(Some(opts))
}

/// Extract the distillation-loss selector from `opts.loss_kind`.
///
/// Mirrors the pre-existing `alc.nn.trainer.distill` extractor in
/// [`super::nn_card`] but under the `run_distill` error prefix (one
/// prefix per surface). Missing key defaults to `"ce"`; unknown
/// values are refused rather than silently falling back — silent
/// fallback would hide a misconfigured caller.
fn extract_distill_loss_kind_rd(opts: &LuaTable) -> LuaResult<DistillLossKind> {
    let raw = opts
        .get::<Option<String>>("loss_kind")?
        .unwrap_or_else(|| "ce".to_string());
    match raw.as_str() {
        "ce" => Ok(DistillLossKind::Ce),
        other => Err(LuaError::external(format!(
            "alc.nn.trainer.run_distill: unknown loss_kind '{other}' (expected 'ce')"
        ))),
    }
}

/// Translate a [`TrainError`] into a `run_distill`-prefixed Lua
/// error. Mirrors [`train_err_to_lua_ff`] structurally; diverges only
/// on the surface prefix per design's loud-error contract.
fn train_err_to_lua_rd(e: TrainError) -> LuaError {
    let msg = match e {
        TrainError::ZeroSteps => "alc.nn.trainer.run_distill: zero steps".to_string(),
        TrainError::LeaseHeld => {
            "alc.nn.trainer.run_distill: training lease already active on this VM".to_string()
        }
        TrainError::DatasetExhausted { seen, requested } => format!(
            "alc.nn.trainer.run_distill: dataset exhausted after {seen} steps \
             (requested {requested})"
        ),
        TrainError::Ckpt(inner) => {
            format!("alc.nn.trainer.run_distill: checkpoint: {inner}")
        }
        TrainError::Candle(inner) => {
            format!("alc.nn.trainer.run_distill: candle: {inner}")
        }
        other => format!("alc.nn.trainer.run_distill: {other}"),
    };
    LuaError::external(msg)
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
    use super::super::nn_card::{build_gpt2_handle, build_tinyllama_handle, load_wrap_impl};
    use super::*;
    use algocline_nn::train::{DatasetOpts, TokenizedDataset};
    use candle_nn::VarMap;
    use mlua::Lua;
    use serde_json::json;

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

    fn expect_err(result: LuaResult<String>) -> String {
        match result {
            Ok(id) => panic!("expected run_lora_ft_impl to fail; got card_id {id:?}"),
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
        let card_id = run_full_ft_impl(
            &store,
            &nn_dir,
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

    #[test]
    fn run_full_ft_tinyllama_happy_path_writes_full_ft_card() {
        let (_tmp, store, nn_dir, base, lua) = setup_tinyllama_scaffold();
        let ds_ud = make_dataset_handle(&lua, overfit_row(), 20);
        let base_ud = lua.create_userdata(NnHandle::TinyLlama(base)).unwrap();

        let opts = opts_table(&lua, base_full_ft_opts());
        let card_id = run_full_ft_impl(
            &store,
            &nn_dir,
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
            &LuaValue::UserData(base_ud),
            &LuaValue::UserData(ds_ud),
            opts,
        ));
        assert!(
            msg.contains("alc.nn.trainer.run_full_ft:") && msg.contains("pretrained=true"),
            "expected pretrained refusal, got: {msg}"
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
}
