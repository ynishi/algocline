//! Full FT training loop.
//!
//! Ties [`crate::arch::gpt2::Gpt2Model`], a [`crate::train::data::Dataset`]
//! source, an AdamW optimizer, a learning-rate [`Scheduler`], and a
//! rotating [`CheckpointStore`] together into a single
//! [`run_full_ft`] entry point.
//!
//! The loop is intentionally CPU-friendly: the tests build a
//! 2-layer / 2-head / 16-dim model and overfit a synthetic 4-token
//! sequence in ~100 steps. On a real GPU the same code path scales to
//! the full 355M / 774M presets without further changes because every
//! candle operation used here already dispatches on the device the
//! `VarMap` was built with.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarMap};

use crate::arch::{LoraConfig, LoraWrappable};
use crate::train::ckpt::{checkpoint_from_path, CheckpointStore};
use crate::train::data::{Batch, Dataset, DatasetError};
use crate::train::loss::Loss;
use crate::train::mixed::MixedAdamW;
use crate::train::scheduler::{ScheduleKind, Scheduler};
use crate::train::Checkpoint;
use crate::train::DeviceView;

/// Optimizer flavour selected by the parameter dtype (design §7.1).
///
/// - All-F32 vars → the stock [`candle_nn::AdamW`], keeping the
///   established baseline bit-identical.
/// - All-BF16 vars → [`MixedAdamW`] (FP32 master weights + FP32
///   moments; gradients upcast per step).
/// - Anything else (F16, F64, a mixed set) is a loud
///   [`TrainError::Candle`]: stock AdamW on BF16 keeps its moments in
///   BF16 and stalls silently, and F16 needs a loss scaler that does
///   not ship here.
enum FtOptimizer {
    Stock(AdamW),
    Mixed(MixedAdamW),
}

impl FtOptimizer {
    fn for_vars(vars: Vec<candle_core::Var>, params: ParamsAdamW) -> Result<Self, TrainError> {
        let mut dtypes: Vec<DType> = vars.iter().map(|v| v.dtype()).collect();
        dtypes.sort_by_key(|d| format!("{d:?}"));
        dtypes.dedup();
        match dtypes.as_slice() {
            [DType::F32] => Ok(Self::Stock(AdamW::new(vars, params)?)),
            [DType::BF16] => Ok(Self::Mixed(MixedAdamW::new(vars, params)?)),
            [DType::F16] => Err(TrainError::Candle(
                "run_ft_core: f16 parameters need loss scaling, which is not \
                 implemented — build the model with dtype bf16 (CUDA) or f32"
                    .into(),
            )),
            other => Err(TrainError::Candle(format!(
                "run_ft_core: unsupported parameter dtype set {other:?} \
                 (expected all-f32 or all-bf16)"
            ))),
        }
    }

    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::Stock(o) => o.set_learning_rate(lr),
            Self::Mixed(o) => o.set_learning_rate(lr),
        }
    }

    fn backward_step(&mut self, loss: &Tensor) -> CandleResult<()> {
        match self {
            Self::Stock(o) => o.backward_step(loss),
            Self::Mixed(o) => o.backward_step(loss),
        }
    }
}

/// Hyperparameters for [`run_full_ft`].
///
/// Every field has a sensible default so callers coming through the
/// Lua bridge with a partial opts table can still get a runnable
/// config.
#[derive(Debug, Clone)]
pub struct FullFtConfig {
    /// Peak learning rate.
    pub lr: f64,
    /// Rows per micro-batch. The dataset is expected to yield batches
    /// of at least this size.
    pub batch_size: usize,
    /// Number of micro-batches summed per optimizer step.
    ///
    /// Only `grad_accum == 1` is honoured by the MVP loop; higher
    /// values are recorded on the config for downstream reporting but
    /// currently trigger a clear error at loop start rather than
    /// silently degrading. Multi-step accumulation is scheduled for a
    /// follow-up once the internal `GradStore` API surface is finalised.
    pub grad_accum: usize,
    /// Total optimizer steps to run.
    pub steps: usize,
    /// Warmup steps for the cosine schedule.
    pub warmup: usize,
    /// Schedule variant.
    pub schedule: ScheduleKind,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Save a rotating checkpoint every N steps. Set to 0 to disable
    /// mid-run checkpoints (the final `<prefix>.safetensors` is still
    /// written at the end).
    pub ckpt_every: usize,
    /// Number of rotating checkpoints kept (clamped to at least 1
    /// inside [`CheckpointStore`]).
    pub ckpt_keep: usize,
}

impl Default for FullFtConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            batch_size: 8,
            grad_accum: 1,
            steps: 100,
            warmup: 10,
            schedule: ScheduleKind::CosineWithWarmup,
            weight_decay: 0.1,
            ckpt_every: 0,
            ckpt_keep: 3,
        }
    }
}

/// One-time guard preventing two Full FT loops from running against
/// the same in-process `NnModelRegistry`.
///
/// The design's "one training session per VM" constraint sits here.
/// Concurrent inference (calls to the registry from a Lua strategy)
/// is unaffected; the guard only refuses a *second* trainer entry
/// while a first one is still holding a lease.
#[derive(Debug, Default)]
pub struct TrainingLease {
    active: AtomicBool,
}

impl TrainingLease {
    /// Build an idle lease.
    pub fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
        }
    }

    /// Try to acquire the lease. Returns `None` when another training
    /// session already holds it.
    pub fn acquire(self: &Arc<Self>) -> Option<TrainingLeaseGuard> {
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            Some(TrainingLeaseGuard {
                lease: self.clone(),
            })
        } else {
            None
        }
    }

    /// Report whether a lease is currently held. Intended for
    /// diagnostics / tests.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
}

/// RAII guard that releases the training lease on drop.
#[must_use = "the lease is released as soon as this guard is dropped"]
pub struct TrainingLeaseGuard {
    lease: Arc<TrainingLease>,
}

impl Drop for TrainingLeaseGuard {
    fn drop(&mut self) {
        self.lease.active.store(false, Ordering::Release);
    }
}

/// Errors surfaced by the training loop.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// The dataset ran out of batches before the requested step count.
    #[error("dataset exhausted after {seen} steps (requested {requested})")]
    DatasetExhausted {
        /// Number of successful step iterations before exhaustion.
        seen: usize,
        /// Configured `cfg.steps` the loop was aiming for.
        requested: usize,
    },
    /// The dataset returned an error mid-iteration.
    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
    /// candle-side failure (forward / backward / optimizer / save).
    #[error("candle: {0}")]
    Candle(String),
    /// Checkpoint I/O failure.
    #[error("checkpoint io: {0}")]
    Ckpt(String),
    /// The caller requested `grad_accum > 1` but the MVP loop does
    /// not implement it yet.
    #[error(
        "grad_accum > 1 is not implemented in the MVP loop; \
             please pass grad_accum = 1 or wait for the follow-up"
    )]
    GradAccumUnsupported,
    /// Config asked for zero training steps.
    #[error("`steps` must be at least 1")]
    ZeroSteps,
    /// Another training session already holds the lease.
    #[error("another training session is already active on this VM")]
    LeaseHeld,
}

impl From<candle_core::Error> for TrainError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e.to_string())
    }
}

/// Run Full FT training and return the final checkpoint record.
///
/// The caller supplies both the `model` (holding forward-pass weights)
/// and the `varmap` those weights were registered against; the loop
/// pulls the parameter list out of the varmap and hands it to AdamW.
/// The dataset is consumed batch-by-batch; when it drains before
/// `cfg.steps`, an explicit error surfaces rather than a silent
/// short-run.
///
/// `ckpt_dir` is the directory the rotating checkpoints live in. A
/// dedicated `<ckpt_prefix>` keeps concurrent (or historical) runs
/// from colliding on filenames.
#[allow(clippy::too_many_arguments)]
pub fn run_full_ft<M>(
    model: &M,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView,
{
    // `run_full_ft` optimises every variable registered against
    // `varmap` — the full-fine-tune baseline. It shares its inner
    // step/save loop with `run_lora_ft` via `run_ft_core`; the only
    // difference is which VarMap the optimizer holds and which VarMap
    // the checkpoint store saves.
    run_ft_core(
        model,
        varmap,
        varmap,
        dataset,
        cfg,
        loss_fn,
        ckpt_dir,
        ckpt_prefix,
        lease,
    )
}

/// Shared inner training loop.
///
/// - `opt_vm` — VarMap whose variables get optimizer updates. In a
///   Full FT run this is the same map as the model was constructed
///   against; in a LoRA run it is the fresh LoRA-only map returned by
///   [`crate::arch::Gpt2Model::wrap_lora`] so the base parameters stay
///   frozen.
/// - `save_vm` — VarMap whose contents get written to disk. Same as
///   `opt_vm` for both current callers, but kept as a distinct
///   parameter so a future full-vs-delta save-side split can flip
///   independently.
#[allow(clippy::too_many_arguments)]
fn run_ft_core<M>(
    model: &M,
    opt_vm: &VarMap,
    save_vm: &VarMap,
    dataset: &mut dyn Dataset,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView,
{
    if cfg.steps == 0 {
        return Err(TrainError::ZeroSteps);
    }
    if cfg.grad_accum != 1 {
        return Err(TrainError::GradAccumUnsupported);
    }

    let _lease = lease.acquire().ok_or(TrainError::LeaseHeld)?;

    let vars = opt_vm.all_vars();
    if vars.is_empty() {
        return Err(TrainError::Candle(
            "run_ft_core: optimizer VarMap has no trainable variables".into(),
        ));
    }

    // AdamW picks up its `lr` from the config once and then follows
    // `set_learning_rate` at each step. The optimizer flavour is
    // driven by the parameter dtype (design §7.1): F32 keeps the
    // stock candle-nn AdamW (bit-identical baseline), BF16 routes
    // through the FP32-master [`MixedAdamW`]. Anything else is a
    // loud error — stock AdamW on BF16 vars would keep its moments
    // in BF16 and stall silently, and F16 needs a loss scaler that
    // does not ship here.
    let adamw_params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..Default::default()
    };
    let mut opt = FtOptimizer::for_vars(vars, adamw_params)?;

    let scheduler = Scheduler::new(cfg.schedule, cfg.lr, 0.0, cfg.warmup, cfg.steps);

    // The store is always constructed: even without mid-run
    // checkpoints (`ckpt_every == 0`) the loop still writes the
    // terminal `<prefix>.safetensors` file through it.
    let ckpt_store = CheckpointStore::new(ckpt_dir, ckpt_prefix.to_string(), cfg.ckpt_keep)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;

    let device = model.device().clone();
    let mut last_train_loss = f32::NAN;
    let mut running_min_loss = f32::INFINITY;

    for step in 0..cfg.steps {
        let lr = scheduler.lr_at(step);
        opt.set_learning_rate(lr);

        let batch = dataset.next_batch()?.ok_or(TrainError::DatasetExhausted {
            seen: step,
            requested: cfg.steps,
        })?;

        let (inputs, targets, mask) = batch_to_input_target(&batch, &device)?;
        let logits = model.forward(&inputs)?;
        // Mixed precision: the loss (log_softmax + NLL reduction) is
        // always scored in F32 — BF16's 8 mantissa bits are too coarse
        // for a mean over thousands of log-probs. `to_dtype` is
        // differentiable, so the backward pass crosses back into the
        // model's dtype at this boundary. F32 logits pass through
        // untouched.
        let logits = if logits.dtype() == DType::F32 {
            logits
        } else {
            logits.to_dtype(DType::F32)?
        };
        let loss = loss_fn.compute(&logits, &targets, mask.as_ref())?;

        let loss_val: f32 = loss.to_scalar()?;
        last_train_loss = loss_val;
        if loss_val < running_min_loss {
            running_min_loss = loss_val;
        }

        // Per-step observability. Emit through `tracing` so downstream
        // subscribers (RUST_LOG=algocline_nn=info) can collect the loss
        // trajectory without changing the return shape. The training
        // loop itself stays a closed function: this line is the only
        // window a caller has into intermediate loss values.
        tracing::info!(step = step, loss = loss_val, lr = lr, "train_step");

        opt.backward_step(&loss)?;

        if cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0 {
            ckpt_store
                .save_step(save_vm, step + 1)
                .map_err(|e| TrainError::Ckpt(e.to_string()))?;
        }
    }

    // Terminal save under the stable `<prefix>.safetensors` filename.
    let final_path = ckpt_store
        .save_final(save_vm)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;

    let mut metrics: HashMap<String, f32> = HashMap::new();
    metrics.insert("min_train_loss".into(), running_min_loss);
    metrics.insert("final_lr".into(), scheduler.lr_at(cfg.steps - 1) as f32);

    checkpoint_from_path(&final_path, cfg.steps, last_train_loss, None, metrics)
        .map_err(TrainError::Ckpt)
}

/// Run LoRA fine-tuning and return the final Δ-only checkpoint record.
///
/// The base model's `Linear` projections are wrapped in-place with
/// [`LoraLinear`] instances (attention Q/K/V/O and MLP up/down per
/// [`LoraConfig::target_modules`]). Only the freshly-created LoRA A/B
/// matrices are handed to AdamW, so the base weights registered
/// against the model's original `VarMap` are guaranteed
/// bit-identical before and after training (LoRA invariant:
/// base parameters are frozen).
///
/// The Δ checkpoint is written to
/// `<ckpt_dir>/nn/lora-<ckpt_stem>.safetensors` — a filename
/// convention that keeps LoRA bundles clearly separated from
/// full-model bundles on disk. The `nn/` subdirectory is created if
/// missing. `ckpt_stem` is a filename stem only — callers that also
/// record a Card conventionally pass the Card id here, but this loop
/// has no Card concept.
///
/// # Errors
///
/// - [`TrainError::ZeroSteps`] / [`TrainError::GradAccumUnsupported`]
///   / [`TrainError::LeaseHeld`] mirror the Full FT path.
/// - Any error raised by [`LoraWrappable::wrap_lora`] surfaces as
///   [`TrainError::Candle`] (unknown `target_modules`, oversized
///   rank, etc.).
/// - Checkpoint I/O failures surface as [`TrainError::Ckpt`].
#[allow(clippy::too_many_arguments)]
pub fn run_lora_ft<M>(
    base: &mut M,
    dataset: &mut dyn Dataset,
    lora_cfg: &LoraConfig,
    train_cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_stem: &str,
    lease: Arc<TrainingLease>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView + LoraWrappable,
{
    if ckpt_stem.is_empty() {
        return Err(TrainError::Candle("run_lora_ft: ckpt_stem is empty".into()));
    }

    // Wrap first so we surface `LoraConfig` validation errors (unknown
    // target module, oversized rank) before the lease is acquired.
    // `M: LoraWrappable` routes this call through the trait; on a
    // concrete `Gpt2Model` / `TinyLlamaModel` the trait impl delegates
    // to the inherent `wrap_lora` method.
    let lora_vm = base.wrap_lora(lora_cfg)?;

    let nn_dir = ckpt_dir.join("nn");
    std::fs::create_dir_all(&nn_dir)
        .map_err(|e| TrainError::Ckpt(format!("run_lora_ft: mkdir {:?}: {e}", nn_dir.display())))?;
    let ckpt_prefix = format!("lora-{ckpt_stem}");

    // `run_ft_core` uses `lora_vm` for both the optimizer and the
    // checkpoint save: the optimizer only sees LoRA A/B parameters
    // (so base weights are structurally frozen — the base varmap is
    // never handed to AdamW) and the saved safetensors bundle
    // contains only those same LoRA A/B tensors (so the Δ file stays
    // small — invariant #3). On GPT-2 medium the empirical Δ size at
    // rank 16 depends on the target set: ~9.5 MB for attention-only
    // wrap (Q/K/V fused + O = 4 wraps × 24 layers), and ~24 MB when
    // the canonical 6-target set also wraps the two MLP linears
    // (add ~14.5 MB from the 4× MLP widening). The previous
    // "< 20 MB" figure only held for the attention-only variant and
    // has been corrected to reflect both cases.
    run_ft_core(
        base,
        &lora_vm,
        &lora_vm,
        dataset,
        train_cfg,
        loss_fn,
        &nn_dir,
        &ckpt_prefix,
        lease,
    )
}

/// Which distillation loss the caller wants for [`run_distill`].
///
/// Only the hard-label cross-entropy variant ships today; a KL-soft
/// variant (needing teacher log-probs) is scheduled for a later
/// stage. Callers stay forward-compatible by matching on the enum.
#[derive(Debug, Clone, Copy)]
pub enum DistillLossKind {
    /// Hard-label cross-entropy on the teacher-emitted tokens
    /// ([`crate::train::HardLabelDistillLoss`]).
    Ce,
}

/// Distillation-run configuration.
///
/// A thin wrapper around a `FullFtConfig` plus the loss variant to
/// use. The actual dataset (a `TeacherCardDataset` in practice) is
/// passed separately to [`run_distill`] so a caller can reuse a
/// pre-built dataset across multiple distillation runs without
/// reconstructing it.
#[derive(Debug, Clone)]
pub struct DistillSpec {
    /// Training hyperparameters (learning rate, steps, schedule, etc.).
    pub hyperparams: FullFtConfig,
    /// Which distillation loss to use.
    pub loss_kind: DistillLossKind,
}

impl DistillSpec {
    /// Build a spec with the default cross-entropy loss and the
    /// supplied hyperparams.
    pub fn ce(hyperparams: FullFtConfig) -> Self {
        Self {
            hyperparams,
            loss_kind: DistillLossKind::Ce,
        }
    }
}

/// Run a distillation training loop.
///
/// Wraps [`run_full_ft`] with the loss selected by `spec.loss_kind`
/// and the caller-supplied `dataset` (which is expected to carry a
/// `Batch::loss_mask` so the loss is scored only on the response
/// region of each teacher log).
///
/// Everything else — checkpoint rotation, scheduler, lease — behaves
/// exactly the same as a Full FT run because that is precisely the
/// underlying loop. The named entry exists so downstream callers
/// (Card metadata, Lua bridge) can encode "this run was a
/// distillation" without inspecting the training config.
///
/// Generic over the student architecture with the same bound as
/// [`run_full_ft`] (`Module + DeviceView`): distillation places no
/// extra requirement on the model — the teacher signal lives in the
/// dataset, not in a second model instance.
#[allow(clippy::too_many_arguments)]
pub fn run_distill<M>(
    student: &M,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    spec: &DistillSpec,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView,
{
    match spec.loss_kind {
        DistillLossKind::Ce => {
            let loss = crate::train::HardLabelDistillLoss::new();
            run_full_ft(
                student,
                varmap,
                dataset,
                &spec.hyperparams,
                &loss,
                ckpt_dir,
                ckpt_prefix,
                lease,
            )
        }
    }
}

/// Break a [`Batch`] into `(inputs, targets, mask)` tensors on the
/// model's device.
///
/// Inputs are `[batch, seq-1]`, targets are `[batch, seq-1]` and are
/// simply the inputs shifted by one position. This matches the
/// standard next-token-prediction training setup: for a sequence
/// `[a, b, c, d]` the model consumes `[a, b, c]` and predicts
/// `[b, c, d]`.
///
/// When the batch carries a `loss_mask` (teacher-log style datasets),
/// the mask is likewise shifted by one so it lines up with the target
/// positions: mask position `k` gates the loss contribution of target
/// token `input_ids[k+1]`. Batches without a mask return `Ok((.., ..,
/// None))` and the caller passes `None` through to `Loss::compute`.
fn batch_to_input_target(
    batch: &Batch,
    device: &Device,
) -> CandleResult<(Tensor, Tensor, Option<Tensor>)> {
    let batch_size = batch.input_ids.len();
    if batch_size == 0 {
        return Err(candle_core::Error::Msg(
            "batch_to_input_target: empty batch".into(),
        ));
    }
    let seq = batch.input_ids[0].len();
    if seq < 2 {
        return Err(candle_core::Error::Msg(format!(
            "batch_to_input_target: seq={seq} is too short (need >= 2)"
        )));
    }
    // Every row must be the same length after padding.
    for (i, row) in batch.input_ids.iter().enumerate() {
        if row.len() != seq {
            return Err(candle_core::Error::Msg(format!(
                "batch_to_input_target: row {i} has length {} (expected {seq})",
                row.len()
            )));
        }
    }

    // Build the [B, S] tensor first, then slice into inputs / targets.
    let mut flat: Vec<u32> = Vec::with_capacity(batch_size * seq);
    for row in &batch.input_ids {
        flat.extend_from_slice(row);
    }
    let full = Tensor::from_vec(flat, (batch_size, seq), device)?;
    let inputs = full
        .narrow(1, 0, seq - 1)?
        .to_dtype(DType::U32)?
        .contiguous()?;
    let targets = full
        .narrow(1, 1, seq - 1)?
        .to_dtype(DType::U32)?
        .contiguous()?;

    // Slice the mask in lockstep with the target shift so mask
    // position `k` gates target token position `k` (== input position
    // `k + 1`).
    let mask = if let Some(mask_rows) = batch.loss_mask.as_ref() {
        if mask_rows.len() != batch_size {
            return Err(candle_core::Error::Msg(format!(
                "batch_to_input_target: loss_mask row count {} != batch size {}",
                mask_rows.len(),
                batch_size
            )));
        }
        for (i, row) in mask_rows.iter().enumerate() {
            if row.len() != seq {
                return Err(candle_core::Error::Msg(format!(
                    "batch_to_input_target: loss_mask row {i} has length {} (expected {seq})",
                    row.len()
                )));
            }
        }
        let mut mflat: Vec<f32> = Vec::with_capacity(batch_size * seq);
        for row in mask_rows {
            mflat.extend_from_slice(row);
        }
        let full_mask = Tensor::from_vec(mflat, (batch_size, seq), device)?;
        Some(
            full_mask
                .narrow(1, 1, seq - 1)?
                .to_dtype(DType::F32)?
                .contiguous()?,
        )
    } else {
        None
    };

    Ok((inputs, targets, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::Gpt2Config;
    use crate::arch::Gpt2Model;
    use crate::train::data::{DatasetOpts, TokenizedDataset};
    use crate::train::loss::CrossEntropyLoss;
    use candle_nn::VarBuilder;
    use tempfile::TempDir;

    fn tiny_cfg_and_model() -> (Gpt2Config, VarMap, Gpt2Model) {
        let cfg = Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vb).unwrap();
        (cfg, vm, model)
    }

    fn overfit_dataset() -> TokenizedDataset {
        // A single 8-token sequence repeated over and over. That is
        // enough to prove the loss trend without spinning up a real
        // corpus.
        let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(400).collect();
        TokenizedDataset::new(
            rows,
            DatasetOpts {
                batch_size: 1,
                ctx_len: 8,
                shuffle: false,
                pad_id: 0,
                text_field: "text".into(),
            },
        )
    }

    #[test]
    fn lease_rejects_second_concurrent_acquire() {
        let lease = Arc::new(TrainingLease::new());
        let guard1 = lease.acquire().expect("first acquire must succeed");
        assert!(lease.is_active());
        assert!(lease.acquire().is_none(), "second acquire must fail");
        drop(guard1);
        assert!(!lease.is_active());
        assert!(lease.acquire().is_some(), "acquire after drop must succeed");
    }

    #[test]
    fn zero_steps_errors_up_front() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 0,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let err =
            run_full_ft(&model, &vm, &mut ds, &cfg, &loss, tmp.path(), "z", lease).unwrap_err();
        assert!(matches!(err, TrainError::ZeroSteps));
    }

    #[test]
    fn grad_accum_gt_one_errors_up_front() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            grad_accum: 4,
            steps: 5,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let err =
            run_full_ft(&model, &vm, &mut ds, &cfg, &loss, tmp.path(), "g", lease).unwrap_err();
        assert!(matches!(err, TrainError::GradAccumUnsupported));
    }

    #[test]
    fn tiny_overfit_reduces_loss() {
        // Small enough that the whole test finishes in ~5s on CPU.
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 8e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 150,
            warmup: 5,
            schedule: ScheduleKind::CosineWithWarmup,
            weight_decay: 0.0,
            ckpt_every: 0,
            ckpt_keep: 1,
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        // First few steps: measure baseline loss for a reference.
        let baseline = {
            // Snapshot the current LM output on the same input the
            // training loop will see and compute a scalar loss.
            let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
            let inputs = Tensor::from_vec(row.clone(), (1, 8), &Device::Cpu)
                .unwrap()
                .narrow(1, 0, 7)
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .contiguous()
                .unwrap();
            let targets = Tensor::from_vec(row, (1, 8), &Device::Cpu)
                .unwrap()
                .narrow(1, 1, 7)
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .contiguous()
                .unwrap();
            let logits = model.forward(&inputs).unwrap();
            let l = loss.compute(&logits, &targets, None).unwrap();
            l.to_scalar::<f32>().unwrap()
        };

        let ckpt = run_full_ft(&model, &vm, &mut ds, &cfg, &loss, tmp.path(), "tiny", lease)
            .expect("training must complete");

        // Sanity: min recorded loss must be materially better than the
        // baseline captured before training kicked in. The threshold
        // (~30% relative reduction) is what the tiny 2-layer / 2-head
        // model reliably achieves on this 8-token repeated corpus in
        // 150 steps on CPU — enough to prove the loop is actually
        // updating parameters without demanding a hero training run
        // inside the test suite.
        let min_loss = *ckpt.metrics.get("min_train_loss").expect("min_train_loss");
        assert!(
            min_loss < baseline * 0.7,
            "expected min_train_loss ({min_loss}) < 0.7 * baseline ({baseline})"
        );
        assert!(
            ckpt.train_loss.is_finite(),
            "final train loss must be finite: {}",
            ckpt.train_loss
        );

        // Terminal file exists at `<prefix>.safetensors`.
        let final_path = tmp.path().join("tiny.safetensors");
        assert!(final_path.exists());
    }

    #[test]
    fn ckpt_every_writes_intermediate_files() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 10,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 2,
            ckpt_keep: 3,
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let _ckpt =
            run_full_ft(&model, &vm, &mut ds, &cfg, &loss, tmp.path(), "rot", lease).unwrap();

        // We should have ckpt-step<N>.safetensors files, capped at
        // `ckpt_keep`.
        let step_files: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.starts_with("rot-step") && n.ends_with(".safetensors")
            })
            .collect();
        assert!(!step_files.is_empty(), "at least one step ckpt must exist");
        assert!(step_files.len() <= cfg.ckpt_keep);
    }

    /// Both trainable arch types must implement [`DeviceView`] so the
    /// generic `run_ft_core` / `run_full_ft` / `run_lora_ft` can pull
    /// the target device out uniformly. Compile-time bound check
    /// (the `let _: &Device = ...` line rejects a missing impl at
    /// compile time) plus a runtime sanity check on the returned
    /// device value.
    #[test]
    fn gpt2_and_tinyllama_impl_device_view() {
        use crate::arch::{TinyLlamaConfig, TinyLlamaModel};

        let (_gpt_cfg, _gpt_vm, gpt_model) = tiny_cfg_and_model();
        let gpt_dev: &Device = DeviceView::device(&gpt_model);
        assert!(matches!(gpt_dev, Device::Cpu));

        let tl_cfg = TinyLlamaConfig::tiny();
        let tl_vm = VarMap::new();
        let tl_vs = VarBuilder::from_varmap(&tl_vm, tl_cfg.dtype, &tl_cfg.device);
        let tl_model = TinyLlamaModel::new(&tl_cfg, tl_vs).unwrap();
        let tl_dev: &Device = DeviceView::device(&tl_model);
        assert!(matches!(tl_dev, Device::Cpu));
    }

    /// Both trainable arch types must implement `candle_nn::Module`
    /// (via a delegate to the inherent `forward`). Force the trait
    /// dispatch by binding through `&dyn Module` — if the impl were
    /// missing, the coercion would fail at compile time; if the impl
    /// diverged from the inherent forward, output shape or values
    /// would drift.
    #[test]
    fn gpt2_and_tinyllama_impl_module_forward() {
        use crate::arch::{TinyLlamaConfig, TinyLlamaModel};

        // GPT-2 path.
        let (gpt_cfg, _gpt_vm, gpt_model) = tiny_cfg_and_model();
        let gpt_ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &gpt_cfg.device).unwrap();
        let gpt_inherent = gpt_model.forward(&gpt_ids).unwrap();
        let gpt_module: &dyn Module = &gpt_model;
        let gpt_via_trait = gpt_module.forward(&gpt_ids).unwrap();
        assert_eq!(gpt_inherent.dims(), gpt_via_trait.dims());
        assert_eq!(gpt_inherent.dims(), &[1, 4, gpt_cfg.vocab]);
        let gpt_inh_vec: Vec<f32> = gpt_inherent.flatten_all().unwrap().to_vec1().unwrap();
        let gpt_trait_vec: Vec<f32> = gpt_via_trait.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            gpt_inh_vec, gpt_trait_vec,
            "Gpt2Model Module impl must byte-match inherent forward"
        );

        // TinyLlama path.
        let tl_cfg = TinyLlamaConfig::tiny();
        let tl_vm = VarMap::new();
        let tl_vs = VarBuilder::from_varmap(&tl_vm, tl_cfg.dtype, &tl_cfg.device);
        let tl_model = TinyLlamaModel::new(&tl_cfg, tl_vs).unwrap();
        let tl_ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &tl_cfg.device).unwrap();
        let tl_inherent = tl_model.forward(&tl_ids).unwrap();
        let tl_module: &dyn Module = &tl_model;
        let tl_via_trait = tl_module.forward(&tl_ids).unwrap();
        assert_eq!(tl_inherent.dims(), tl_via_trait.dims());
        assert_eq!(tl_inherent.dims(), &[1, 4, tl_cfg.vocab]);
        let tl_inh_vec: Vec<f32> = tl_inherent.flatten_all().unwrap().to_vec1().unwrap();
        let tl_trait_vec: Vec<f32> = tl_via_trait.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            tl_inh_vec, tl_trait_vec,
            "TinyLlamaModel Module impl must byte-match inherent forward"
        );
    }

    /// Both trainable arch types must implement [`LoraWrappable`] so
    /// the generic `run_lora_ft` can call `wrap_lora` uniformly.
    /// Bind through `&mut dyn LoraWrappable` to force trait dispatch;
    /// verify the returned `VarMap` carries `layers × targets × 2` new
    /// LoRA vars (matches the freeze invariant test — but here we're
    /// asserting the trait dispatch itself, not the wrap semantics).
    #[test]
    fn gpt2_and_tinyllama_impl_lora_wrappable() {
        use crate::arch::{TinyLlamaConfig, TinyLlamaModel};

        // GPT-2: 2 layers × 4 wraps × 2 (A+B) = 16 LoRA vars.
        // Note: the 6 canonical GPT-2 target names (q_proj, k_proj,
        // v_proj, o_proj, up, down) collapse into 4 physical wraps
        // because q/k/v share the fused `c_attn` linear (see
        // `Gpt2Block::wrap_lora`), so any of q/k/v in target_modules
        // triggers exactly one c_attn wrap.
        let (_gpt_cfg, _gpt_vm, mut gpt_model) = tiny_cfg_and_model();
        let gpt_lora_cfg = LoraConfig::new(2, 4.0);
        let gpt_dyn: &mut dyn LoraWrappable = &mut gpt_model;
        let gpt_lora_vm = gpt_dyn.wrap_lora(&gpt_lora_cfg).unwrap();
        assert_eq!(
            gpt_lora_vm.all_vars().len(),
            2 * 4 * 2,
            "GPT-2 wrap_lora via LoraWrappable must register 2 layers × 4 wraps × 2 = 16 vars"
        );

        // TinyLlama: 2 layers × 7 canonical targets × 2 = 28 LoRA vars.
        let tl_cfg = TinyLlamaConfig::tiny();
        let tl_vm = VarMap::new();
        let tl_vs = VarBuilder::from_varmap(&tl_vm, tl_cfg.dtype, &tl_cfg.device);
        let mut tl_model = TinyLlamaModel::new(&tl_cfg, tl_vs).unwrap();
        let tl_lora_cfg = LoraConfig::with_targets(2, 4.0, TinyLlamaModel::default_lora_targets());
        let tl_dyn: &mut dyn LoraWrappable = &mut tl_model;
        let tl_lora_vm = tl_dyn.wrap_lora(&tl_lora_cfg).unwrap();
        assert_eq!(
            tl_lora_vm.all_vars().len(),
            2 * 7 * 2,
            "TinyLlama wrap_lora via LoraWrappable must register 2 layers × 7 targets × 2 = 28 vars"
        );
    }
}
