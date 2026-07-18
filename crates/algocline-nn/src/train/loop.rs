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
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::arch::Gpt2Model;
use crate::train::ckpt::{checkpoint_from_path, CheckpointStore};
use crate::train::data::{Batch, Dataset, DatasetError};
use crate::train::loss::Loss;
use crate::train::scheduler::{ScheduleKind, Scheduler};
use crate::train::Checkpoint;

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
/// dedicated `<card_id>` prefix keeps concurrent (or historical) runs
/// from colliding on filenames.
#[allow(clippy::too_many_arguments)]
pub fn run_full_ft(
    model: &Gpt2Model,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
) -> Result<Checkpoint, TrainError> {
    if cfg.steps == 0 {
        return Err(TrainError::ZeroSteps);
    }
    if cfg.grad_accum != 1 {
        return Err(TrainError::GradAccumUnsupported);
    }

    let _lease = lease.acquire().ok_or(TrainError::LeaseHeld)?;

    // AdamW picks up its `lr` from the config once and then follows
    // `set_learning_rate` at each step.
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            ..Default::default()
        },
    )?;

    let scheduler = Scheduler::new(cfg.schedule, cfg.lr, 0.0, cfg.warmup, cfg.steps);

    // The store is always constructed: even without mid-run
    // checkpoints (`ckpt_every == 0`) the loop still writes the
    // terminal `<prefix>.safetensors` file through it.
    let ckpt_store = CheckpointStore::new(ckpt_dir, ckpt_prefix.to_string(), cfg.ckpt_keep)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;

    let device = model.config().device.clone();
    let mut last_train_loss = f32::NAN;
    let mut running_min_loss = f32::INFINITY;

    for step in 0..cfg.steps {
        let lr = scheduler.lr_at(step);
        opt.set_learning_rate(lr);

        let batch = dataset.next_batch()?.ok_or(TrainError::DatasetExhausted {
            seen: step,
            requested: cfg.steps,
        })?;

        let (inputs, targets) = batch_to_input_target(&batch, &device)?;
        let logits = model.forward(&inputs)?;
        let loss = loss_fn.compute(&logits, &targets, None)?;

        let loss_val: f32 = loss.to_scalar()?;
        last_train_loss = loss_val;
        if loss_val < running_min_loss {
            running_min_loss = loss_val;
        }

        opt.backward_step(&loss)?;

        if cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0 {
            ckpt_store
                .save_step(varmap, step + 1)
                .map_err(|e| TrainError::Ckpt(e.to_string()))?;
        }
    }

    // Terminal save under the stable `<prefix>.safetensors` filename.
    let final_path = ckpt_store
        .save_final(varmap)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;

    let mut metrics: HashMap<String, f32> = HashMap::new();
    metrics.insert("min_train_loss".into(), running_min_loss);
    metrics.insert("final_lr".into(), scheduler.lr_at(cfg.steps - 1) as f32);

    checkpoint_from_path(&final_path, cfg.steps, last_train_loss, None, metrics)
        .map_err(TrainError::Ckpt)
}

/// Break a [`Batch`] into `(inputs, targets)` tensors on the model's
/// device.
///
/// Inputs are `[batch, seq-1]`, targets are `[batch, seq-1]` and are
/// simply the inputs shifted by one position. This matches the
/// standard next-token-prediction training setup: for a sequence
/// `[a, b, c, d]` the model consumes `[a, b, c]` and predicts
/// `[b, c, d]`.
fn batch_to_input_target(batch: &Batch, device: &Device) -> CandleResult<(Tensor, Tensor)> {
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
    Ok((inputs, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::Gpt2Config;
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
}
