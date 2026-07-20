//! `alc.nn.trainer.run_lora_ft` bridge — Layer 5b S2 Lua bind for the
//! one-call LoRA fine-tuning surface: wrap the base + train + save Δ
//! safetensors + write a `training_path="lora"` Card, all in a single
//! call.
//!
//! Extends the pre-existing `alc.nn.trainer` sub-table (populated by
//! [`super::nn_card::register_nn_card`] with `full_ft` / `lora` /
//! `distill`) by adding a fourth entry alongside them:
//!
//! ```text
//! alc.nn.trainer.run_lora_ft(base_handle, dataset, opts) -> lora_card_id
//! ```
//!
//! Sibling of Layer 5b S1 [`super::nn_wrap`], which registered the
//! wrap-only surface (`alc.nn.wrap_lora`). S2 consumes the same LoRA
//! opts schema (`rank` / `alpha` / `target_modules` / `dropout`) and
//! layers a training-config schema (`lr` / `batch` / `steps` /
//! `warmup` / `schedule`) on top.
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
use algocline_nn::train::{run_lora_ft, CrossEntropyLoss, FullFtConfig, TrainError, TrainingLease};
use mlua::prelude::*;
use serde_json::json;

use crate::card::FileCardStore;

use super::nn_card::{
    build_create_payload_from_meta, compact_epoch_us, extract_full_ft_opts, sanitize_name,
    DatasetHandle, Gpt2Handle, LlamaHandle, NnHandle, TinyLlamaHandle,
};

/// Register `alc.nn.trainer.run_lora_ft` onto the pre-existing
/// `alc.nn.trainer` sub-table.
///
/// Must be called after [`super::nn_card::register_nn_card`] (which
/// creates the `alc.nn.trainer` sub-table populated with `full_ft` /
/// `lora` / `distill`) — this call extends that sub-table with a
/// fourth entry rather than creating a fresh one, so subsequent
/// `run_full_ft` / `run_distill` binds (L5c) can land alongside
/// without a namespace rename.
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

#[cfg(test)]
mod run_lora_ft_bridge_tests {
    //! Layer 5b S2 — bridge integration tests for
    //! `alc.nn.trainer.run_lora_ft`.
    //!
    //! Axis A3/A4 exercise the GPT-2 + TinyLlama happy paths (arch
    //! dispatch + on-disk Card + Δ safetensors + payload shape).
    //! Axis B6/B7 exercise config schema refusals unique to the
    //! trainer (steps / schedule). Axis C1-C3 exercise state
    //! invariants that only become verifiable post-train
    //! (base-freeze, Δ var-count, cross-surface `load_wrap`
    //! consumption). Axis D1/D2 exercise the Card + Δ round-trip
    //! through `alc.nn.card.load_wrap`.
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
}
