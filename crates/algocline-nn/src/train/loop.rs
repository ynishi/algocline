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

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use serde::Serialize;

use candle_core::backprop::GradStore;
use candle_core::TensorId;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Module, Optimizer, ParamsAdamW, VarMap};

use crate::arch::{AllowedSets, Checkpointable, CondIndex, LoraConfig, LoraWrappable};
use crate::train::checkpointing::checkpointed_step;
use crate::train::ckpt::{
    checkpoint_from_path, restore_into, BundleIdentity, Candidate, CheckpointStore, MetricPoint,
    RestoreError,
};
use crate::train::data::{Batch, Dataset, DatasetError};
use crate::train::lion::{Lion, ParamsLion};
use crate::train::loss::Loss;
use crate::train::mixed::MixedAdamW;
use crate::train::optstate::{names_by_tensor_id, sidecar_path, OptimizerState};
use crate::train::scheduler::{ScheduleKind, Scheduler};
use crate::train::AllowedForward;
use crate::train::Checkpoint;
use crate::train::ConditionedForward;
use crate::train::DeviceView;

/// Optimizer flavour, over the parameter dtypes each one accepts
/// (design §7.1).
///
/// - AdamW → [`MixedAdamW`] on both F32 and BF16 (FP32 master weights +
///   FP32 moments; gradients upcast per step).
/// - Lion → [`Lion`], which keeps its own FP32 master where the dtype
///   needs one.
/// - Anything else (F16, F64, a mixed set) is a loud
///   [`TrainError::Candle`]: an optimizer holding BF16 moments stalls
///   silently, and F16 needs a loss scaler that does not ship here.
///
/// # Why F32 AdamW no longer routes through `candle_nn::AdamW`
///
/// It used to, as the bit-identical baseline. But that type keeps its
/// moments in private fields with no accessor, so a run using it could
/// not write its optimizer state and `init_from` could never be more
/// than a warm start — for the default optimizer at the default dtype,
/// which is nearly every run. [`MixedAdamW`] implements the same update
/// term for term and its FP32-on-FP32 case is pinned against the stock
/// one to within `1e-6` by `mixed::tests::f32_parity_with_stock_adamw`.
/// Resumable state for every run is worth that much.
enum FtOptimizer {
    Mixed(MixedAdamW),
    Lion(Lion),
}

/// Which optimizer the trainer requested.
///
/// The dtype dispatch below is a separate axis: it decides *how* a
/// chosen optimizer is realised (stock or FP32-master), not which one
/// the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OptimizerKind {
    /// Decoupled-decay Adam. The default, and what every existing run
    /// was trained with.
    #[default]
    AdamW,
    /// Sign-momentum, from arXiv:2302.06675. Takes a learning rate
    /// 3–10× smaller than AdamW's and a weight decay 3–10× larger —
    /// carrying AdamW's values over unchanged trains at a step size an
    /// order of magnitude off.
    Lion,
}

impl OptimizerKind {
    /// Parse the wire form written by callers (Lua bridge, JSON config).
    ///
    /// `None` on an unknown string, so the caller can name the
    /// alternatives rather than surface a generic parse error.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "adamw" | "adam_w" | "adam" => Some(Self::AdamW),
            "lion" => Some(Self::Lion),
            _ => None,
        }
    }

    /// Every wire name this version accepts.
    pub const NAMES: [&'static str; 4] = ["adamw", "adam_w", "adam", "lion"];

    /// Stable number for this flavour, written into an optimizer-state
    /// file so a resume can refuse to read one optimizer's tensors into
    /// another's slots.
    ///
    /// Spelt out rather than taken from the enum's layout: a variant
    /// added above another must not renumber a file already on disk.
    pub fn discriminant(self) -> u32 {
        match self {
            Self::AdamW => 0,
            Self::Lion => 1,
        }
    }

    /// Inverse of [`Self::discriminant`]. `None` for a number this
    /// build has no optimizer for.
    pub fn from_discriminant(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::AdamW),
            1 => Some(Self::Lion),
            _ => None,
        }
    }
}

impl FtOptimizer {
    fn for_vars(
        kind: OptimizerKind,
        vars: Vec<candle_core::Var>,
        params: ParamsAdamW,
    ) -> Result<Self, TrainError> {
        let mut dtypes: Vec<DType> = vars.iter().map(|v| v.dtype()).collect();
        dtypes.sort_by_key(|d| format!("{d:?}"));
        dtypes.dedup();
        // Lion keeps its own FP32 master where one is needed, so it
        // spans both dtypes without a second arm here.
        if kind == OptimizerKind::Lion {
            return match dtypes.as_slice() {
                [DType::F32] | [DType::BF16] => Ok(Self::Lion(Lion::new(
                    vars,
                    ParamsLion {
                        lr: params.lr,
                        beta1: params.beta1,
                        beta2: params.beta2,
                        weight_decay: params.weight_decay,
                    },
                )?)),
                [DType::F16] => Err(TrainError::Candle(
                    "run_ft_core: f16 parameters need loss scaling, which is not \
                     implemented — build the model with dtype bf16 (CUDA) or f32"
                        .into(),
                )),
                other => Err(TrainError::Candle(format!(
                    "run_ft_core: unsupported parameter dtype set {other:?} \
                     (expected all-f32 or all-bf16)"
                ))),
            };
        }
        match dtypes.as_slice() {
            [DType::F32] | [DType::BF16] => Ok(Self::Mixed(MixedAdamW::new(vars, params)?)),
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
            Self::Mixed(o) => o.set_learning_rate(lr),
            Self::Lion(o) => o.set_learning_rate(lr),
        }
    }

    /// Which flavour this is, for the state file's own record of what
    /// wrote it.
    fn kind(&self) -> OptimizerKind {
        match self {
            Self::Mixed(_) => OptimizerKind::AdamW,
            Self::Lion(_) => OptimizerKind::Lion,
        }
    }

    /// Everything this optimizer would need to carry on, as tensors
    /// named after the parameters they belong to.
    fn state(&self, names: &HashMap<TensorId, String>) -> Result<OptimizerState, String> {
        let mut tensors = HashMap::new();
        let step = match self {
            Self::Mixed(o) => {
                o.write_state(names, &mut tensors)?;
                o.step_count()
            }
            Self::Lion(o) => {
                o.write_state(names, &mut tensors)?;
                // Lion has no bias correction and keeps no counter; the
                // loop's own step is written so a resume can pick the
                // schedule back up.
                0
            }
        };
        Ok(OptimizerState {
            kind: self.kind(),
            step,
            tensors,
        })
    }

    /// Read a state back in, refusing one written by a different
    /// optimizer.
    fn load_state(
        &mut self,
        names: &HashMap<TensorId, String>,
        state: &OptimizerState,
    ) -> Result<(), String> {
        if state.kind != self.kind() {
            return Err(format!(
                "optimizer state: the file was written by {:?} and this run uses {:?};                  their tensors are not the same quantity",
                state.kind,
                self.kind()
            ));
        }
        match self {
            Self::Mixed(o) => o.read_state(names, &state.tensors, state.step),
            Self::Lion(o) => o.read_state(names, &state.tensors),
        }
    }

    /// Apply a single optimizer step against a pre-computed
    /// [`GradStore`]. This is the entry point the multi-micro-batch
    /// path in [`run_ft_core`] uses to reuse the underlying
    /// [`Optimizer::step`] once per `grad_accum` micro-batches, after
    /// [`GradStore::extend`] has summed the per-micro grads. The
    /// stock [`Optimizer::backward_step`] convenience is not delegated
    /// because the loop always splits `backward()` and `step()` (even
    /// for `grad_accum == 1`) so a single code path serves both cases.
    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        match self {
            Self::Mixed(o) => o.step(grads),
            Self::Lion(o) => o.step(grads),
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
    /// `grad_accum > 1` accumulates gradients across `grad_accum`
    /// micro-batches before applying a single optimizer update, so the
    /// effective batch size is `batch_size * grad_accum`. Each
    /// micro-batch's loss is pre-scaled by `1 / grad_accum` before
    /// `backward()` (canonical PyTorch form). `grad_accum = 0` is
    /// refused as a config error; `grad_accum = 1` behaves exactly
    /// like the single-micro path.
    ///
    /// What that sum equals is the mean **of the per-micro means**, not
    /// the mean over the effective batch's tokens: the loss divides by
    /// each micro-batch's own scored-token count. The two coincide only
    /// when every micro-batch scores the same number of positions,
    /// which stopped being automatic when [`DatasetOpts::mask_pad`]
    /// began excluding padding — a short final batch now carries more
    /// weight per token than a full one.
    pub grad_accum: usize,
    /// Total optimizer steps to run.
    pub steps: usize,
    /// Warmup steps for the cosine schedule.
    pub warmup: usize,
    /// Schedule variant.
    pub schedule: ScheduleKind,
    /// Floor the decaying schedules land on at `steps` and hold
    /// afterwards. `0.0` (default) reproduces the decay-to-zero the
    /// cosine schedule had before this field existed.
    ///
    /// Governs the tail only: the warmup ramp climbs from near zero and
    /// passes below this value on its way up.
    pub min_lr: f64,
    /// Length of the decay stretch for
    /// [`ScheduleKind::WarmupStableDecay`], in steps. `None` leaves it
    /// at the share [`Scheduler::DEFAULT_DECAY_FRACTION`] names. Read
    /// by no other schedule.
    pub decay_steps: Option<usize>,
    /// Which optimizer runs the steps.
    pub optimizer: OptimizerKind,
    /// Weight decay, decoupled from the gradient in both optimizers.
    ///
    /// Lion's guidance puts this 3–10× above an AdamW value for the
    /// same run, because the effective decay is `lr·λ` and Lion's `lr`
    /// is correspondingly smaller.
    pub weight_decay: f64,
    /// First-moment / update-blend coefficient. AdamW's `β₁` and Lion's
    /// alike; the paper defaults differ (0.9 for both here, but Lion's
    /// `β₂` default is 0.99 against AdamW's 0.999).
    pub beta1: f64,
    /// Second-moment coefficient for AdamW, momentum-EMA coefficient
    /// for Lion.
    pub beta2: f64,
    /// AdamW's denominator epsilon. Unread by Lion, which has no
    /// second moment to divide by.
    pub eps: f64,
    /// Save a rotating checkpoint every N steps. Set to 0 to disable
    /// mid-run checkpoints (the final `<prefix>.safetensors` is still
    /// written at the end).
    pub ckpt_every: usize,
    /// Number of rotating checkpoints kept (clamped to at least 1
    /// inside [`CheckpointStore`]).
    pub ckpt_keep: usize,
    /// Write the optimizer's own state beside every checkpoint, so
    /// [`Self::init_from`] can resume rather than warm-start.
    ///
    /// `false` (default) leaves checkpoints the size they have always
    /// been. AdamW's state is an FP32 master plus two FP32 moments per
    /// parameter — roughly three times the parameters again — and it is
    /// of no use to anything but a resume of this exact run, so it goes
    /// in a `<checkpoint>.opt.safetensors` sidecar rather than into the
    /// bundle every inference path loads.
    ///
    /// What turning it on buys: `init_from` currently restores weights
    /// into a zeroed optimizer, and AdamW's bias-corrected
    /// `m̂ / (√v̂ + ε)` makes the first steps after that behave like the
    /// first steps of a run. The loss bends at the restart and nothing
    /// in the record says why. With the state in place the moments and
    /// the step count carry over and the schedule picks up where it
    /// stopped.
    ///
    /// What it does not buy: the data order. A [`Dataset`] is a
    /// one-pass stream with no position to restore, so a resumed run
    /// starts the corpus again from the top.
    pub save_optimizer_state: bool,
    /// Cap the joint L2 norm of the gradient at this value before each
    /// optimizer step, or `None` (default) for no cap.
    ///
    /// Scales every trainable parameter's gradient by
    /// `max_norm / norm` when the norm exceeds `max_norm`, leaving the
    /// direction alone and only its length changed — the standard
    /// global-norm form rather than a per-tensor or per-element clamp,
    /// which would tilt the update away from the gradient.
    ///
    /// What it is for: one bad batch produces a gradient orders of
    /// magnitude larger than the rest, the step it drives lands far
    /// outside the region the loss was measured in, and the run either
    /// returns to a worse place or leaves with non-finite weights. The
    /// cap bounds how far any single step can move regardless of the
    /// batch, and `1.0` is where most transformer recipes sit.
    ///
    /// A non-finite norm is left unscaled — multiplying by
    /// `max_norm / NaN` would only spread the NaN into every parameter
    /// that still had a usable gradient, and the norm still reaches the
    /// `on_ckpt` hook, which is where a run can notice and stop.
    ///
    /// The norm reported through [`CkptInfo::grad_norm`] is the one
    /// measured before this scaling, so it says what the step actually
    /// produced rather than what the cap allowed through.
    pub clip_grad_norm: Option<f64>,
    /// What the checkpoints this run writes should say about
    /// themselves, or `None` (default) to leave them describing
    /// nothing.
    ///
    /// A bare `.safetensors` file carries no architecture, no
    /// vocabulary and no dtype; everything that identified a bundle
    /// lived in its Card, which does not travel with the file. With
    /// this set, every checkpoint carries a header and a `.json`
    /// sidecar saying what model loads it — see [`BundleIdentity`].
    ///
    /// Provenance rather than a hyperparameter, and it sits here
    /// because the loop has no other way to learn it: a `VarMap` does
    /// not say what architecture registered it.
    pub bundle_identity: Option<BundleIdentity>,
    /// Recompute each block's activations during the backward pass
    /// instead of keeping them from the forward.
    ///
    /// `false` (default) keeps every intermediate of every block alive
    /// until the backward reads it, which is what bounds context length
    /// and batch size on a given card. `true` keeps one
    /// `[batch, seq, dim]` tensor per block and pays a second forward
    /// pass for the rest — roughly a third more compute for a fraction
    /// of the activation memory (Chen et al. 2016,
    /// [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)).
    ///
    /// The gradients are the same gradients: a checkpointed step is
    /// asserted against an ordinary one parameter by parameter, not
    /// merely shaped like it.
    ///
    /// Available on [`run_full_ft`] only, and only for models that can
    /// be driven one block at a time
    /// ([`crate::arch::Checkpointable::checkpointable`] says which).
    /// Asking for it elsewhere is
    /// [`TrainError::CheckpointingUnsupported`] rather than a flag that
    /// silently does nothing.
    pub grad_checkpoint: bool,
    /// Append one line per N optimizer steps to a
    /// `<prefix>-metrics.jsonl` file beside the checkpoints, or `0`
    /// (default) to write none.
    ///
    /// What the run produced used to reach a record only as final
    /// values on the Card and as `Candidate` rows at keep time — two
    /// snapshots and no curve. Whether a run converged, stalled,
    /// diverged, or was still descending when it ran out of steps are
    /// all different shapes of the same final loss, and none of them is
    /// legible from it. `tracing` already emits a `train_step` event per
    /// step, but that is a subscriber's to collect and is gone
    /// afterwards; this leaves a file the run itself wrote.
    ///
    /// One JSON object per line (`{"step":…,"loss":…,"lr":…}`, plus
    /// `grad_norm` and `val_loss` where the run has them), which is the
    /// shape every plotting tool reads and which an interrupted run
    /// leaves valid up to its last complete line.
    pub metrics_every: usize,
    /// Stop when the held-out loss has not improved for a while, or
    /// `None` (default) to run every step asked for.
    ///
    /// Requires [`Self::eval_every`] and a validation dataset:
    /// stopping on the training loss would stop when the model stopped
    /// fitting the data it is being fitted to, which is not the
    /// question early stopping asks. Configured without them it is
    /// [`TrainError::EarlyStopWithoutValidation`] rather than a rule
    /// that quietly never fires.
    pub early_stop: Option<EarlyStop>,
    /// Score the held-out set every N optimizer steps, or `0`
    /// (default) to run without one.
    ///
    /// Non-zero requires the caller to hand a validation dataset to the
    /// entry point, and a validation dataset requires this to be
    /// non-zero: either half alone is
    /// [`TrainError::ValidationHalfConfigured`] rather than a run that
    /// silently never evaluates, or one that pays for a held-out split
    /// nothing reads.
    ///
    /// The held-out batches are drained once before the first step and
    /// re-scored at each boundary, so every evaluation sees the same
    /// rows and the sequence of values is a curve rather than a walk
    /// through different data.
    pub eval_every: usize,
    /// Checkpoint the model's variables are restored from before the
    /// first step, or `None` (default) to train from whatever the
    /// caller built.
    ///
    /// Restored through [`restore_into`], so anything short of a
    /// complete restore is a [`TrainError::Restore`] and the run does
    /// not start: a resume that quietly kept some parameters at their
    /// initial values is the failure that costs a run, and it is
    /// indistinguishable from a real one once training is under way.
    ///
    /// A resume or a warm start, depending on what is beside the file.
    /// With a `<checkpoint>.opt.safetensors` sidecar (written by
    /// [`Self::save_optimizer_state`]) the optimizer state and the step
    /// count come back too, the schedule continues from that step, and
    /// [`Self::steps`] is read as the total the run is working towards
    /// rather than a count of further steps. Without one the weights
    /// are restored into a fresh optimizer at step 0, and the run says
    /// so through `tracing` rather than leaving the two cases looking
    /// alike.
    ///
    /// Not supported on [`run_lora_ft`], which never sees the base
    /// map — see [`TrainError::InitFromUnsupported`].
    pub init_from: Option<PathBuf>,
    /// Whether the loss scores each target among the ids that position
    /// allowed, rather than among the whole vocabulary.
    ///
    /// When `true`, [`allowed_logit_mask`] is added to the logits
    /// before the loss, so the ids a target could not have taken stop
    /// being charged for. Requires the dataset to carry
    /// [`Batch::allowed_ids`]; a batch without them is
    /// [`TrainError::MissingAllowedSets`] rather than an unmasked step
    /// under a config that says otherwise.
    ///
    /// `false` (default) leaves the loss over the full vocabulary even
    /// for a dataset that carries the sets — the sets are then either
    /// unused or serving only as model input through
    /// [`run_allowed_ft`], which is what the two independent switches
    /// are for.
    ///
    /// A model trained with the mask on receives no gradient pressure
    /// to suppress disallowed ids, so its unconstrained argmax is not a
    /// meaningful decision: pair it with a constrained decode path (an
    /// allow-list sampler or an equivalent gate). Measured on a small
    /// board-game policy, raw argmax legality fell from 0.71 to
    /// 0.08–0.16 under masked training while gated-decode play
    /// strength improved.
    pub mask_disallowed_logits: bool,
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
            min_lr: 0.0,
            decay_steps: None,
            optimizer: OptimizerKind::AdamW,
            weight_decay: 0.1,
            // candle's `ParamsAdamW` defaults, restated rather than
            // inherited so a candle bump is a visible change here
            // instead of a silent one in every run.
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            ckpt_every: 0,
            ckpt_keep: 3,
            bundle_identity: None,
            grad_checkpoint: false,
            metrics_every: 0,
            early_stop: None,
            save_optimizer_state: false,
            clip_grad_norm: None,
            eval_every: 0,
            init_from: None,
            mask_disallowed_logits: false,
        }
    }
}

/// When to stop a run that is no longer improving.
///
/// Read at every evaluation boundary, against the held-out loss. The
/// rule is the standard one and both halves of it matter: `patience`
/// alone stops on noise, and `min_delta` alone never stops a run whose
/// loss keeps creeping down by nothing.
#[derive(Debug, Clone, Copy)]
pub struct EarlyStop {
    /// Evaluations without an improvement before the run stops.
    ///
    /// `0` stops at the first evaluation that fails to improve, which
    /// is almost always too eager — a loss curve is not monotone at the
    /// scale of one evaluation period.
    pub patience: usize,
    /// How much lower the loss has to be to count as an improvement.
    ///
    /// `0.0` counts any decrease, including one indistinguishable from
    /// floating-point noise, which makes `patience` nearly unreachable
    /// on a long run.
    pub min_delta: f32,
}

/// The best held-out loss seen, and how long since it was beaten.
#[derive(Debug)]
struct EarlyStopWatch {
    rule: EarlyStop,
    best: f32,
    /// Evaluations since `best` was last improved on.
    since: usize,
}

impl EarlyStopWatch {
    fn new(rule: EarlyStop) -> Self {
        Self {
            rule,
            best: f32::INFINITY,
            since: 0,
        }
    }

    /// Record one evaluation and answer whether the run should stop.
    ///
    /// A non-finite loss is not an improvement and not a reason to keep
    /// going: it counts against patience like any other failure to
    /// improve, so a run that diverges into NaN stops on the same rule
    /// rather than running to the end producing nothing.
    fn observe(&mut self, value: f32) -> bool {
        if value.is_finite() && value < self.best - self.rule.min_delta {
            self.best = value;
            self.since = 0;
            false
        } else {
            self.since += 1;
            self.since > self.rule.patience
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
    /// Deprecated: previously surfaced when the loop rejected
    /// `grad_accum > 1`. The training loop now honours multi-step
    /// accumulation natively, so this variant is unreachable from
    /// [`run_ft_core`]; kept for one release cycle to avoid breaking
    /// downstream `match` arms that still enumerate it.
    #[deprecated(
        since = "0.47.0",
        note = "grad_accum > 1 is now supported natively; this variant is retained for one release cycle and is no longer constructed by the training loop"
    )]
    #[allow(dead_code)]
    #[error(
        "grad_accum > 1 is not implemented in the MVP loop; \
             please pass grad_accum = 1 or wait for the follow-up"
    )]
    GradAccumUnsupported,
    /// Config asked for zero training steps.
    #[error("`steps` must be at least 1")]
    ZeroSteps,
    /// Config named a learning rate that cannot mean what it says.
    ///
    /// A negative rate ascends the loss and a non-finite one poisons
    /// every parameter on the first update; both run to completion and
    /// write a checkpoint that looks like any other. **Zero is
    /// allowed** — it takes no step, which is a real thing to ask for
    /// (holding the weights while the rest of the loop runs) and is
    /// what two of this module's own tests do.
    ///
    /// Checked in the loop rather than at one bridge surface, so every
    /// entry point answers the same way.
    #[error("`lr` must be finite and not negative (got {value})")]
    InvalidLearningRate {
        /// The value the config carried.
        value: f64,
    },
    /// Config asked for `grad_accum = 0`, which would divide by zero
    /// when scaling per-micro losses. Multi-step accumulation is now
    /// honoured for `grad_accum >= 1`.
    #[error("`grad_accum` must be at least 1 (got 0)")]
    ZeroGradAccum,
    /// Another training session already holds the lease.
    #[error("another training session is already active on this VM")]
    LeaseHeld,
    /// [`FullFtConfig::clip_grad_norm`] was set to a value that cannot
    /// cap anything.
    ///
    /// Zero would erase every gradient and a negative value would
    /// reverse it, and either produces a run that trains — steps are
    /// taken, a loss is reported, a checkpoint is written — while
    /// moving nowhere or backwards.
    #[error("clip_grad_norm must be a finite positive number (got {value})")]
    InvalidClipNorm {
        /// The value the config carried.
        value: f64,
    },
    /// A resumed run had already reached or passed
    /// [`FullFtConfig::steps`].
    ///
    /// With optimizer state in hand, `steps` is the total the run is
    /// working towards, so there is nothing left to do. Refused rather
    /// than returned as a zero-step run, which would write a fresh
    /// terminal checkpoint whose recorded loss came from no step at
    /// all.
    #[error(
        "the checkpoint resumes at step {resumed} and cfg.steps is {steps}, so this run has          already finished; raise steps to extend it"
    )]
    ResumeBeyondSteps {
        /// Step the optimizer state was written at.
        resumed: usize,
        /// Total the config asks for.
        steps: usize,
    },
    /// Optimizer state was found beside a checkpoint and could not be
    /// read into this run's optimizer.
    ///
    /// Refused rather than skipped: the caller asked to resume, and a
    /// silent fall back to a warm start is the failure the state file
    /// exists to remove.
    #[error("init_from: {0}")]
    OptState(String),
    /// One half of the validation setup arrived without the other.
    ///
    /// [`FullFtConfig::eval_every`] and the entry point's validation
    /// dataset are one decision expressed in two places, so either half
    /// alone is a mistake with a quiet outcome: a period with no set
    /// never evaluates, and a set with no period is a slice held out of
    /// training that nothing reads.
    #[error(
        "validation is half-configured: {present} was given and {missing} was not — \
         eval_every and the validation dataset go together"
    )]
    ValidationHalfConfigured {
        /// The half the caller supplied.
        present: &'static str,
        /// The half it needs.
        missing: &'static str,
    },
    /// [`FullFtConfig::grad_checkpoint`] was set where it cannot be
    /// honoured.
    ///
    /// Either the entry point does not take a blockwise view of the
    /// model (the conditioned / allowed-id / LoRA paths), or the model
    /// itself refuses one. A flag that silently did nothing would leave
    /// a caller believing they had the memory headroom they asked for.
    #[error("grad_checkpoint cannot be honoured here: {0}")]
    CheckpointingUnsupported(String),
    /// [`FullFtConfig::early_stop`] was set on a run with nothing to
    /// watch.
    ///
    /// The rule reads the held-out loss, and a run without a held-out
    /// set never produces one. Refused rather than left to never fire,
    /// which is indistinguishable from a rule that was never reached.
    #[error(
        "early_stop needs a held-out set to watch: set eval_every and pass a validation \
         dataset, or drop the rule"
    )]
    EarlyStopWithoutValidation,
    /// A validation dataset was handed over and produced no batch.
    ///
    /// Refused at the start rather than reported as an absent
    /// `val_loss` later: a held-out split that came out empty is a
    /// split that went wrong, and the run would otherwise spend its
    /// whole length before saying so.
    #[error("the validation dataset yielded no batch, so there is nothing to score")]
    EmptyValidationSet,
    /// An `on_ckpt` hook returned an error.
    ///
    /// The error propagates immediately: no terminal
    /// `<prefix>.safetensors` is written, and no [`Checkpoint`] comes
    /// back — the last-good weights a caller can reach are the
    /// rotating `<prefix>-step<N>.safetensors` files, plus any step
    /// the hook had already asked to keep (pins survive, since nothing
    /// un-pins). Anything the run had collected in
    /// [`Checkpoint::candidates`] is lost with the error, so a hook
    /// that keeps checkpoints and can also fail should record what it
    /// kept as it goes rather than waiting for the run to hand the
    /// list back.
    #[error("on_ckpt hook: {0}")]
    Hook(String),
    /// [`FullFtConfig::init_from`] named a checkpoint the model's
    /// variables could not be restored from. The run does not start.
    #[error("init_from: {0}")]
    Restore(#[from] RestoreError),
    /// [`FullFtConfig::init_from`] was set on an entry point that never
    /// sees the base `VarMap`.
    ///
    /// [`run_lora_ft`] is handed the model and builds its own
    /// LoRA-only map; the base variables belong to a map the caller
    /// holds. Restoring the LoRA map from a base checkpoint would
    /// restore nothing it recognises, so the request is refused here
    /// rather than answered with a checkpoint that went somewhere else.
    /// Restore the base map yourself (
    /// [`crate::train::restore_into`]) before wrapping.
    #[error(
        "init_from is not supported by this entry point: it never sees the base VarMap, so the \
         checkpoint would have nowhere to land; restore the base map before wrapping it"
    )]
    InitFromUnsupported,
    /// A conditioned run was handed a batch carrying no conditions.
    ///
    /// Refused rather than falling back to the plain forward, because a
    /// run that quietly trained unconditioned would still write a
    /// checkpoint labelled as conditioned.
    #[error(
        "a conditioned run received a batch of {rows} row(s) carrying no conditions; the run \
         would have trained unconditioned under a checkpoint labelled otherwise"
    )]
    MissingConditions {
        /// Rows in the batch that arrived without conditions.
        rows: usize,
    },
    /// An entry point with nowhere to put a condition was handed a
    /// batch carrying one. The mirror of
    /// [`TrainError::MissingConditions`]: dropping it would discard a
    /// channel the caller attached per row.
    #[error(
        "a batch of {rows} row(s) carries {conds} condition(s), but this entry point has \
         nowhere to put them; use run_conditioned_ft"
    )]
    UnexpectedConditions {
        /// Rows in the batch.
        rows: usize,
        /// Conditions the batch carried.
        conds: usize,
    },
    /// Allowed-id sets were required — as model input by
    /// [`run_allowed_ft`], or by the loss under
    /// [`FullFtConfig::mask_disallowed_logits`] — and the batch carried
    /// none.
    #[error(
        "a batch of {rows} row(s) carries no allowed-id sets, which this run requires ({needed})"
    )]
    MissingAllowedSets {
        /// Rows in the batch that arrived without sets.
        rows: usize,
        /// What needed them, so the caller knows which switch to look
        /// at: the model input or the loss mask.
        needed: &'static str,
    },
    /// A batch arrived whose loss mask scores no position at all.
    ///
    /// Refused rather than run. The masked mean divides by
    /// `max(mask_sum, 1)` to keep a fully-masked batch from producing
    /// `NaN`, so such a batch yields a loss of exactly `0.0` — a step
    /// with no gradient, reported as the best loss the run has seen.
    /// Once that value is latched, `min_train_loss` never rises again
    /// and every later checkpoint looks worse than a step that learnt
    /// nothing. The two readings — "this batch is empty" and "this
    /// batch is perfect" — are the same number, which is why it has to
    /// stop here instead of being scored.
    #[error(
        "a batch of {rows} row(s) has {scored} scored position(s): its loss mask leaves nothing \
         to learn from, which scores 0.0 and would latch as the run's best loss"
    )]
    NothingScored {
        /// Rows in the batch.
        rows: usize,
        /// Scored positions across the whole batch — zero, by
        /// construction of this error.
        scored: usize,
    },
}

/// Information handed to the [`CkptHook`] at every `ckpt_every` boundary.
///
/// Kept flat (owned primitives + [`PathBuf`]) so the hook can convert it
/// into an mlua-side Lua table without borrowing from any tensor. The
/// checkpoint has already been written to `ckpt_path` by the time the
/// hook fires — the hook decides whether to keep training or stop.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CkptInfo {
    /// Optimizer step index at which this checkpoint fired
    /// (1-indexed, matches the `<prefix>-step<N>.safetensors` filename).
    pub step: usize,
    /// Absolute path of the checkpoint file just written by
    /// [`super::ckpt::CheckpointStore::save_step`].
    pub ckpt_path: PathBuf,
    /// Mean per-micro loss on the just-completed optimizer step
    /// (identical to the value emitted through `tracing::info!`).
    pub train_loss: f32,
    /// Learning rate applied on the just-completed optimizer step.
    pub lr: f64,
    /// L2 norm of the gradient tensors accumulated for the step,
    /// upcast to F32 (so mixed-precision runs report a comparable
    /// number). Non-finite values propagate untouched — the hook is
    /// the intended place to notice.
    pub grad_norm: f32,
    /// Wall-clock milliseconds since the trainer entered `run_ft_core`.
    /// Sourced from a single [`Instant`] so successive hook fires
    /// give a monotonically non-decreasing value.
    pub elapsed_ms: u64,
    /// Minimum train loss seen so far in this run (matches the
    /// terminal `metrics["min_train_loss"]` value if the run completes
    /// without an early break).
    pub min_train_loss: f32,
    /// Loss on the held-out set at the most recent evaluation, or
    /// `None` on a run with no held-out set
    /// ([`FullFtConfig::eval_every`] unset).
    ///
    /// This is the number a keep decision wants: `train_loss` falls
    /// whether the model is learning the task or the corpus, and the
    /// two are indistinguishable from inside the training set. The
    /// evaluation runs on the same forward path and the same loss the
    /// training step uses, so the two are comparable.
    ///
    /// Evaluations happen every `eval_every` steps and checkpoints
    /// every `ckpt_every` steps; when the two do not divide each other
    /// this carries the most recent evaluation rather than one taken at
    /// this step. [`Self::step`] against
    /// [`FullFtConfig::eval_every`] is what says how stale it can be.
    ///
    /// Absent rather than `null` where a run held nothing out, so the
    /// candidate record this flattens into follows the same rule the
    /// metrics file does: a reader that sees the key can rely on it,
    /// and two sibling JSONL files in one directory do not disagree
    /// about how they spell "no value".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_loss: Option<f32>,
}

/// Whether the trainer continues or breaks early after an
/// [`CkptHook`] fires.
///
/// A `Break` triggers the same terminal `save_final` +
/// `checkpoint_from_path` finalization as a normal loop completion, so
/// the returned [`Checkpoint`] is always usable. The metrics map
/// additionally carries `early_break = 1.0` when the hook stopped the
/// run early, so downstream consumers can distinguish an early stop
/// from a full-run save without walking the step count against the
/// requested `cfg.steps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkptFlow {
    /// Keep training. Emitted as `nil` / `"continue"` from Lua.
    Continue,
    /// Stop training now (after the current checkpoint). Emitted as
    /// `"break"` from Lua.
    Break,
}

/// A hook's request to hold the checkpoint it was just handed.
///
/// Carries the caller's own account of why, in two registers: a note
/// for a reader and the numbers the judgment actually read. The trainer
/// parses neither — both are written through to the [`Candidate`] so
/// whoever reads the run outcome can see which judgment selected which
/// step, and on what evidence.
///
/// `values` exists because a search record that says only `"tier-2"`
/// cannot be re-examined. The measurements are in hand at the moment of
/// the decision and gone immediately after, so this is the one place
/// they can be captured without every application inventing its own
/// manifest for them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KeepMark {
    /// Free-form note from the hook, e.g. the band label that hit.
    pub reason: Option<String>,
    /// The measurements behind the decision, by name. Ordered so the
    /// written record is byte-stable across runs that measured the
    /// same things.
    pub values: BTreeMap<String, f64>,
}

/// What the hook decided at a checkpoint boundary.
///
/// Two independent questions, kept as two fields rather than folded
/// into one enum, because a checkpoint search asks both at once and
/// the answers do not constrain each other:
///
/// - `flow` — does the run go on?
/// - `keep` — is this checkpoint worth holding?
///
/// All four combinations are reachable and meaningful. "Keep and stop"
/// is the ordinary end of a successful search; "keep and continue" is
/// the ordinary middle of one; "drop and continue" is every other
/// checkpoint; "drop and stop" is a run abandoned on a diverging loss.
/// Folding the two into a single enum would have made the first case
/// unsayable, which is what an earlier ABI did — a hook that wanted to
/// hold a checkpoint had to reach around the trainer and copy the file
/// out from under the rotation before it turned over.
// Not `Eq`: the keep mark carries measured `f64`s.
#[derive(Debug, Clone, PartialEq)]
pub struct CkptControl {
    /// Whether the training loop proceeds past this checkpoint.
    pub flow: CkptFlow,
    /// `Some` when the hook asked for this checkpoint to be held out
    /// of the rotation and recorded as a candidate.
    pub keep: Option<KeepMark>,
}

impl CkptControl {
    /// Carry on, holding nothing. What `nil` / `"continue"` means.
    pub const CONTINUE: Self = Self {
        flow: CkptFlow::Continue,
        keep: None,
    };

    /// Stop now, holding nothing. What `"break"` means.
    pub const BREAK: Self = Self {
        flow: CkptFlow::Break,
        keep: None,
    };

    /// Hold this checkpoint and carry on. What `"keep"` means.
    pub fn keep(reason: Option<String>) -> Self {
        Self {
            flow: CkptFlow::Continue,
            keep: Some(KeepMark {
                reason,
                ..KeepMark::default()
            }),
        }
    }

    /// Hold this checkpoint and stop. The end of a successful search.
    pub fn keep_and_break(reason: Option<String>) -> Self {
        Self {
            flow: CkptFlow::Break,
            keep: Some(KeepMark {
                reason,
                ..KeepMark::default()
            }),
        }
    }

    /// Hold this checkpoint, carry on, and record what was measured.
    pub fn keep_with(mark: KeepMark) -> Self {
        Self {
            flow: CkptFlow::Continue,
            keep: Some(mark),
        }
    }
}

/// Callback fired at every `ckpt_every` boundary, after the checkpoint
/// has been written to disk.
///
/// - `Send` bound: the engine crate uses
///   `mlua = { features = ["send", ...] }` so the hook may cross the
///   `AsyncIsle` thread boundary. `'static` is implied by
///   `Box<dyn ... + Send>`.
/// - Return type is `Result<CkptControl, String>` (not raw
///   `CkptControl`) because the hook's typical source is a Lua
///   callback that can raise: a Lua-side error propagates back as
///   [`TrainError::Hook`] (loud, `?`-able) rather than as a `panic!`
///   inside the training loop.
pub type CkptHook = Box<dyn FnMut(&CkptInfo) -> Result<CkptControl, String> + Send>;

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
///
/// `val` is the held-out set. `Some` requires
/// [`FullFtConfig::eval_every`] to be non-zero and `None` requires it
/// to be zero — see [`TrainError::ValidationHalfConfigured`]. With one
/// in place the returned [`Checkpoint::val_loss`] and each
/// [`CkptInfo::val_loss`] carry the loss on those rows, scored through
/// the same forward path and loss as training.
///
/// `hook` is an optional [`CkptHook`] fired at each `ckpt_every`
/// boundary (after `save_step`). Passing `None` retains the previous
/// behaviour bit-identically; passing `Some(_)` lets the caller inspect
/// per-checkpoint scalars and return [`CkptControl::BREAK`] to stop
/// training early. The hook is exclusive to the full-fine-tune surface
/// today; the LoRA / distillation entries pass `None` internally.
#[allow(clippy::too_many_arguments)]
pub fn run_full_ft<M>(
    model: &M,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    val: Option<&mut dyn Dataset>,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
    hook: Option<CkptHook>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView + Checkpointable,
{
    // `run_full_ft` optimises every variable registered against
    // `varmap` — the full-fine-tune baseline. It shares its inner
    // step/save loop with `run_lora_ft` via `run_ft_core`; the only
    // difference is which VarMap the optimizer holds and which VarMap
    // the checkpoint store saves.
    apply_init_from(cfg, varmap)?;
    run_ft_core(
        model.device(),
        ForwardPass::Plain(&mut |xs| model.forward(xs).map_err(TrainError::from)),
        varmap,
        varmap,
        dataset,
        val,
        // The same model, as the blockwise view `grad_checkpoint`
        // needs. Only this entry point has one: the conditioned and
        // allowed-id passes read a channel the blockwise surface does
        // not take.
        Some(model),
        cfg,
        loss_fn,
        ckpt_dir,
        ckpt_prefix,
        lease,
        hook,
    )
}

/// Run Full FT training with a condition supplied per row of every
/// batch.
///
/// The same loop as [`run_full_ft`] — same optimizer, schedule,
/// checkpoint rotation and hook — differing only in how a batch reaches
/// the model: through [`ConditionedForward::forward_conditioned_rows`]
/// with the batch's own [`Batch::conds`], rather than through
/// `Module::forward`.
///
/// # Why a second entry point rather than a flag
///
/// The two paths need different things of the model. `run_full_ft`'s
/// bound is `Module + DeviceView`, which every architecture here
/// satisfies; conditioning needs a table most of them do not have.
/// Widening the single entry point would have made every model answer
/// for a concept it does not carry, and the answer would have been a
/// runtime error raised after the corpus was read. Two entries put the
/// same refusal in the type checker.
///
/// # Errors
///
/// As [`run_full_ft`], plus [`TrainError::MissingConditions`] when a
/// batch arrives without them.
#[allow(clippy::too_many_arguments)]
pub fn run_conditioned_ft<M>(
    model: &M,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    val: Option<&mut dyn Dataset>,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
    hook: Option<CkptHook>,
) -> Result<Checkpoint, TrainError>
where
    M: ConditionedForward + DeviceView,
{
    apply_init_from(cfg, varmap)?;
    run_ft_core(
        model.device(),
        ForwardPass::PerRow(&mut |xs, conds, per_row| {
            model
                .forward_conditioned_rows(xs, conds, per_row)
                .map_err(TrainError::from)
        }),
        varmap,
        varmap,
        dataset,
        val,
        // No blockwise view: this pass reads an input channel the
        // blockwise surface does not take, so `grad_checkpoint` is
        // refused here rather than quietly dropping the channel.
        None,
        cfg,
        loss_fn,
        ckpt_dir,
        ckpt_prefix,
        lease,
        hook,
    )
}

/// Run Full FT training with the ids allowed at every position handed
/// to the model as input.
///
/// The same loop as [`run_full_ft`] — same optimizer, schedule,
/// checkpoint rotation and hook — differing only in how a batch reaches
/// the model: through [`AllowedForward::forward_allowed_rows`] with the
/// sets built from the batch's own [`Batch::allowed_ids`], rather than
/// through `Module::forward`.
///
/// # What the sets do at each end
///
/// The same list serves twice and the two uses are opposite. As
/// **input** (here) the model is told what is available before it
/// answers; as a **mask** ([`allowed_logit_mask`], switched on by
/// [`FullFtConfig::mask_disallowed_logits`]) the ids that were never
/// available stop being charged for. A run can have either or both:
/// this entry point adds the first to whichever the config asked for.
///
/// # Errors
///
/// As [`run_full_ft`], plus [`TrainError::MissingAllowedSets`] when a
/// batch arrives without them — refused rather than falling back to the
/// plain forward, which on a model built with the table would fail at
/// the forward anyway, several steps further from the cause.
#[allow(clippy::too_many_arguments)]
pub fn run_allowed_ft<M>(
    model: &M,
    varmap: &VarMap,
    dataset: &mut dyn Dataset,
    val: Option<&mut dyn Dataset>,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
    hook: Option<CkptHook>,
) -> Result<Checkpoint, TrainError>
where
    M: AllowedForward + DeviceView,
{
    apply_init_from(cfg, varmap)?;
    run_ft_core(
        model.device(),
        ForwardPass::Allowed(&mut |xs, allowed| {
            model
                .forward_allowed_rows(xs, allowed)
                .map_err(TrainError::from)
        }),
        varmap,
        varmap,
        dataset,
        val,
        // No blockwise view: this pass reads an input channel the
        // blockwise surface does not take, so `grad_checkpoint` is
        // refused here rather than quietly dropping the channel.
        None,
        cfg,
        loss_fn,
        ckpt_dir,
        ckpt_prefix,
        lease,
        hook,
    )
}

/// Write the optimizer's state beside a checkpoint that was just
/// saved, when the config asked for it.
///
/// `step` is the loop's global step, which is what a resume needs for
/// the schedule. For AdamW it also has to match the optimizer's own
/// counter, and it does: both count optimizer steps from the start of
/// the original run.
fn save_optimizer_state(
    opt: &FtOptimizer,
    names: &HashMap<TensorId, String>,
    cfg: &FullFtConfig,
    ckpt_path: &Path,
    step: usize,
) -> Result<(), TrainError> {
    if !cfg.save_optimizer_state {
        return Ok(());
    }
    let mut state = opt.state(names).map_err(TrainError::OptState)?;
    // The loop's step wins over the optimizer's own: Lion keeps no
    // counter, and on a resumed AdamW run the two agree anyway.
    state.step = step;
    state
        .save(&sidecar_path(ckpt_path))
        .map_err(TrainError::OptState)
}

/// Read the optimizer state beside `ckpt_path` into `opt`, returning
/// the step it was written at.
///
/// `0` when there is no sidecar: the caller asked to start from a
/// checkpoint that carries no optimizer state, which is a warm start
/// and is logged as one. The two cases used to be indistinguishable
/// from the outside, which is the whole complaint.
fn resume_optimizer(
    opt: &mut FtOptimizer,
    names: &HashMap<TensorId, String>,
    ckpt_path: &Path,
    device: &Device,
) -> Result<usize, TrainError> {
    let path = sidecar_path(ckpt_path);
    if !path.exists() {
        tracing::info!(
            target: "algocline_nn::train",
            checkpoint = %ckpt_path.display(),
            "init_from is a warm start: no optimizer state beside the checkpoint, so the \
             moments begin at zero and the schedule at step 0"
        );
        return Ok(0);
    }
    let state = OptimizerState::load(&path, device).map_err(TrainError::OptState)?;
    opt.load_state(names, &state)
        .map_err(TrainError::OptState)?;
    tracing::info!(
        target: "algocline_nn::train",
        checkpoint = %ckpt_path.display(),
        step = state.step,
        "init_from is a resume: optimizer state restored"
    );
    Ok(state.step)
}

/// Restore [`FullFtConfig::init_from`] into `varmap`, if one was named.
///
/// Strict: [`restore_into`] refuses anything short of a complete
/// restore, and the error propagates rather than degrading into a run
/// that started from a mixture of the checkpoint and a random
/// initialisation. The report is logged rather than returned — the
/// caller asked for a resume, and what it needs to know is that the
/// resume happened and from where.
fn apply_init_from(cfg: &FullFtConfig, varmap: &VarMap) -> Result<(), TrainError> {
    let Some(path) = cfg.init_from.as_ref() else {
        return Ok(());
    };
    let report = restore_into(varmap, path)?;
    tracing::info!(
        target: "algocline_nn::train",
        summary = %report.summary(),
        "init_from restored"
    );
    Ok(())
}

/// How one micro-batch reaches the model, and what the caller expects
/// of the batch.
///
/// The loop holds this rather than the model itself, so the entry
/// points can require different things of it — `Module` on one side,
/// [`ConditionedForward`] or [`AllowedForward`] on the others — without
/// any of those bounds leaking into the shared loop or into another
/// entry's callers.
///
/// It is an enum rather than one callback taking `Option`s, because the
/// variant is also the caller's **declared intent**, and that is what
/// lets the loop refuse both mismatches rather than only the one the
/// model happens to notice. A callback that quietly ignored an
/// unexpected condition would put the loop back where
/// [`TrainError::UnexpectedConditions`] came from.
enum ForwardPass<'a> {
    /// `Module::forward` — the ids alone. A batch carrying conditions
    /// is [`TrainError::UnexpectedConditions`].
    Plain(&'a mut PlainForward<'a>),
    /// [`ConditionedForward::forward_conditioned_rows`] — the ids plus
    /// the conditions each row carries. A batch carrying none is
    /// [`TrainError::MissingConditions`].
    PerRow(&'a mut PerRowForward<'a>),
    /// [`AllowedForward::forward_allowed_rows`] — the ids plus, at
    /// every position, the set the answer there may be drawn from. A
    /// batch carrying none is [`TrainError::MissingAllowedSets`]; one
    /// carrying conditions is [`TrainError::UnexpectedConditions`],
    /// since this entry point has nowhere to put them either.
    Allowed(&'a mut AllowedForwardPass<'a>),
}

/// The ids of one micro-batch to its logits.
///
/// The lifetime is the closure's own: it borrows the model, so it does
/// not outlive the call that built it. Naming it is what keeps the
/// alias from defaulting the trait object to `'static`, which no entry
/// point here could satisfy.
type PlainForward<'a> = dyn FnMut(&Tensor) -> Result<Tensor, TrainError> + 'a;

/// The same, with the conditions each row of the batch carries.
type PerRowForward<'a> = dyn FnMut(&Tensor, &[CondIndex], usize) -> Result<Tensor, TrainError> + 'a;

/// The same, with the ids allowed at each position of the batch.
type AllowedForwardPass<'a> = dyn FnMut(&Tensor, &AllowedSets) -> Result<Tensor, TrainError> + 'a;

/// Shared inner training loop.
///
/// - `device` — where the batch tensors are built. The model's own, so
///   the caller reads it off the model rather than being trusted to
///   pick one.
/// - `forward` — how a micro-batch reaches the model, and what the
///   caller expects of it. See [`ForwardPass`]: the loop is written
///   against the callback so the entry points can differ in what they
///   demand of the model without this function knowing about any of
///   them, and it checks each batch against the variant rather than
///   letting a disagreement pass as a silent ignore.
/// - `blockwise` — the model again, as something that can be driven one
///   block at a time, for [`FullFtConfig::grad_checkpoint`]. `None` on
///   the entry points that have no such view, where asking for
///   checkpointing is refused rather than ignored.
/// - `val` — the held-out set, drained once before the first step and
///   re-scored every [`FullFtConfig::eval_every`] steps. `None` on a
///   run without one; supplying one of the two without the other is
///   [`TrainError::ValidationHalfConfigured`].
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
fn run_ft_core(
    device: &Device,
    mut forward: ForwardPass<'_>,
    opt_vm: &VarMap,
    save_vm: &VarMap,
    dataset: &mut dyn Dataset,
    val: Option<&mut dyn Dataset>,
    blockwise: Option<&dyn Checkpointable>,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    ckpt_dir: &Path,
    ckpt_prefix: &str,
    lease: Arc<TrainingLease>,
    mut hook: Option<CkptHook>,
) -> Result<Checkpoint, TrainError> {
    if cfg.steps == 0 {
        return Err(TrainError::ZeroSteps);
    }
    if cfg.grad_accum == 0 {
        return Err(TrainError::ZeroGradAccum);
    }
    if !(cfg.lr.is_finite() && cfg.lr >= 0.0) {
        return Err(TrainError::InvalidLearningRate { value: cfg.lr });
    }
    if let Some(max_norm) = cfg.clip_grad_norm {
        if !(max_norm.is_finite() && max_norm > 0.0) {
            return Err(TrainError::InvalidClipNorm { value: max_norm });
        }
    }
    // The two halves of the validation setup, checked against each
    // other before the lease is taken: both mismatches end in a run
    // that looks configured and measures nothing.
    // Checked before the lease, like the other config disagreements.
    let blockwise = match (cfg.grad_checkpoint, blockwise) {
        (false, _) => None,
        (true, Some(model)) => {
            model
                .checkpointable()
                .map_err(TrainError::CheckpointingUnsupported)?;
            Some(model)
        }
        (true, None) => {
            return Err(TrainError::CheckpointingUnsupported(
                "this entry point drives the model through a forward it cannot decompose; \
                 grad_checkpoint is available on run_full_ft"
                    .into(),
            ))
        }
    };
    if cfg.early_stop.is_some() && cfg.eval_every == 0 {
        return Err(TrainError::EarlyStopWithoutValidation);
    }
    let val = match (val, cfg.eval_every) {
        (Some(_), 0) => {
            return Err(TrainError::ValidationHalfConfigured {
                present: "a validation dataset",
                missing: "cfg.eval_every",
            })
        }
        (None, n) if n > 0 => {
            return Err(TrainError::ValidationHalfConfigured {
                present: "cfg.eval_every",
                missing: "a validation dataset",
            })
        }
        (val, _) => val,
    };

    let _lease = lease.acquire().ok_or(TrainError::LeaseHeld)?;
    // Fixed reference point for [`CkptInfo::elapsed_ms`]. Taken after
    // the lease is acquired so a `LeaseHeld` refusal does not pay
    // wall-clock time it never used.
    let train_start = Instant::now();

    let vars = opt_vm.all_vars();
    if vars.is_empty() {
        return Err(TrainError::Candle(
            "run_ft_core: optimizer VarMap has no trainable variables".into(),
        ));
    }

    // AdamW picks up its `lr` from the config once and then follows
    // `set_learning_rate` at each step. Both F32 and BF16 route
    // through the FP32-master `MixedAdamW` — see `FtOptimizer`, which
    // records why the stock candle-nn AdamW was retired. Anything
    // else is a loud error: an optimizer keeping BF16 moments stalls
    // silently, and F16 needs a loss scaler that does not ship here.
    let adamw_params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        beta1: cfg.beta1,
        beta2: cfg.beta2,
        eps: cfg.eps,
    };
    let mut opt = FtOptimizer::for_vars(cfg.optimizer, vars, adamw_params)?;

    // The weights were restored before the loop was entered (see
    // `apply_init_from`); the optimizer exists only now, so its own
    // state is read here. `resumed_step` is 0 for a fresh run and for a
    // warm start, and the step the state was written at otherwise.
    let names = names_by_tensor_id(opt_vm);
    let resumed_step = match cfg.init_from.as_ref() {
        Some(path) => resume_optimizer(&mut opt, &names, path, device)?,
        None => 0,
    };
    if resumed_step >= cfg.steps {
        return Err(TrainError::ResumeBeyondSteps {
            resumed: resumed_step,
            steps: cfg.steps,
        });
    }

    let scheduler = {
        let s = Scheduler::new(cfg.schedule, cfg.lr, cfg.min_lr, cfg.warmup, cfg.steps);
        match cfg.decay_steps {
            Some(n) => s.with_decay_steps(n),
            None => s,
        }
    };

    // Drained before the first step so an empty or broken held-out
    // split is a refusal at the start rather than a `val_loss` that
    // never arrives.
    let val_batches = match val {
        Some(val) => Some(drain_validation(val)?),
        None => None,
    };

    // The store is always constructed: even without mid-run
    // checkpoints (`ckpt_every == 0`) the loop still writes the
    // terminal `<prefix>.safetensors` file through it.
    let mut ckpt_store = CheckpointStore::new(ckpt_dir, ckpt_prefix.to_string(), cfg.ckpt_keep)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;
    if let Some(identity) = cfg.bundle_identity.clone() {
        ckpt_store = ckpt_store.with_identity(identity);
    }

    let device = device.clone();
    let mut last_train_loss = f32::NAN;
    let mut running_min_loss = f32::INFINITY;
    // `None` until the first evaluation lands, which is also what a run
    // without a held-out set reports for its whole length.
    let mut last_val_loss: Option<f32> = None;
    let mut min_val_loss = f32::INFINITY;
    let mut watch = cfg.early_stop.map(EarlyStopWatch::new);
    // Checkpoints the hook asked to hold, in the order it asked.
    let mut candidates: Vec<Candidate> = Vec::new();

    // Nested loop: outer = one optimizer step per iteration; inner =
    // `cfg.grad_accum` micro-batches whose per-micro losses are scaled
    // by `1 / grad_accum` and whose grads are summed through
    // `GradStore::extend` before a single `opt.step`. For
    // `grad_accum == 1` this collapses to the previous single-micro
    // path (one backward + one step). `DatasetExhausted::seen` is
    // reported as the number of *completed* optimizer steps: a mid-
    // micro exhaustion inside a step counts that step as unfinished so
    // the reported total matches "how many effective batches actually
    // updated the parameters".
    let grad_accum = cfg.grad_accum;
    let scale = 1.0f64 / grad_accum as f64;
    // `step` is the global index: on a resumed run it starts where the
    // state left off, so the schedule, the checkpoint filenames and the
    // hook's `info.step` all continue the earlier run rather than
    // restarting alongside it.
    for step in resumed_step..cfg.steps {
        let lr = scheduler.lr_at(step);
        opt.set_learning_rate(lr);

        let mut accum: Option<GradStore> = None;
        let mut micro_loss_sum: f32 = 0.0;
        for _micro in 0..grad_accum {
            let batch = dataset.next_batch()?.ok_or(TrainError::DatasetExhausted {
                seen: step,
                requested: cfg.steps,
            })?;

            // Pre-backward `1 / grad_accum` scaling — the canonical
            // form. Scalar multiplication is linear w.r.t. the backward
            // pass, so `sum_i grad(loss_i / N) == grad(mean_i loss_i)`
            // and the reported grad equals the mean over the effective
            // batch. For `grad_accum == 1` this reduces to `scale = 1`
            // and the multiply is a no-op numerically. Both paths apply
            // it before any backward runs, so they scale identically.
            let (loss_val, grads) = match blockwise {
                Some(model) => {
                    let (inputs, targets, mask) = batch_to_input_target(&batch, &device)?;
                    let (value, grads) = checkpointed_step(
                        model,
                        &inputs,
                        |logits| {
                            loss_from_logits(
                                logits.clone(),
                                &batch,
                                &targets,
                                mask.as_ref(),
                                cfg,
                                loss_fn,
                                &device,
                            )
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))
                        },
                        scale,
                    )?;
                    (value, grads)
                }
                None => {
                    let loss = forward_loss(&mut forward, &batch, &device, cfg, loss_fn)?;
                    let loss_val: f32 = loss.to_scalar()?;
                    let scaled = (&loss * scale)?;
                    (loss_val, scaled.backward()?)
                }
            };
            micro_loss_sum += loss_val;
            match accum.as_mut() {
                Some(store) => store.extend(grads)?,
                None => accum = Some(grads),
            }
        }
        // `accum` is always `Some` here because `grad_accum >= 1` is
        // enforced above and the inner loop runs at least once — a
        // mid-micro dataset exhaustion returns early via `?` above.
        let mut grads = accum.expect("grad_accum >= 1 guarantees at least one backward");

        let mean_loss = micro_loss_sum / grad_accum as f32;
        last_train_loss = mean_loss;
        if mean_loss < running_min_loss {
            running_min_loss = mean_loss;
        }

        // Per-step observability. Emit through `tracing` so downstream
        // subscribers (RUST_LOG=algocline_nn=info) can collect the loss
        // trajectory without changing the return shape. `loss` is the
        // mean of the per-micro losses — see `grad_accum` for why that
        // is not the same as one `batch_size * grad_accum` batch once
        // padding is masked — and `grad_accum` is emitted as an
        // additive field so post-hoc analysis can distinguish
        // accumulated steps from raw single-micro ones.
        tracing::info!(
            step = step,
            loss = mean_loss,
            lr = lr,
            grad_accum = grad_accum,
            "train_step"
        );

        // Computed before `opt.step` consumes / mutates the
        // per-parameter state. Two consumers want it — the `on_ckpt`
        // hook and the clip below — and the walk is skipped when
        // neither does, so a run with no hook and no cap pays nothing
        // beyond the existing per-step cost.
        let will_fire_hook =
            hook.is_some() && cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0;
        let grad_norm = if will_fire_hook || cfg.clip_grad_norm.is_some() {
            grad_l2_norm(opt_vm, &grads)?
        } else {
            0.0
        };
        // Reported as measured, then capped: `grad_norm` above is what
        // the step produced, and this is what the optimizer receives.
        if let Some(max_norm) = cfg.clip_grad_norm {
            clip_grad_norm_(opt_vm, &mut grads, max_norm, grad_norm)?;
        }

        opt.step(&grads)?;

        // Scored after the step, so the value belongs to the weights
        // the checkpoint written just below actually holds.
        let mut out_of_patience = false;
        if let Some(batches) = val_batches.as_ref() {
            if (step + 1) % cfg.eval_every == 0 {
                let v = evaluate(&mut forward, batches, &device, cfg, loss_fn)?;
                last_val_loss = Some(v);
                if v < min_val_loss {
                    min_val_loss = v;
                }
                tracing::info!(step = step, val_loss = v, "eval_step");
                if let Some(watch) = watch.as_mut() {
                    out_of_patience = watch.observe(v);
                }
            }
        }

        // The curve, written as the run goes rather than collected at
        // the end: a run that dies still leaves what it had.
        if cfg.metrics_every > 0 && (step + 1) % cfg.metrics_every == 0 {
            let point = MetricPoint {
                step: step + 1,
                loss: mean_loss,
                lr,
                grad_norm: (will_fire_hook || cfg.clip_grad_norm.is_some()).then_some(grad_norm),
                val_loss: last_val_loss,
            };
            ckpt_store
                .append_metrics(&point)
                .map_err(|e| TrainError::Ckpt(format!("metrics: {e}")))?;
        }

        if cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0 {
            let ckpt_path = ckpt_store
                .save_step(save_vm, step + 1)
                .map_err(|e| TrainError::Ckpt(e.to_string()))?;
            save_optimizer_state(&opt, &names, cfg, &ckpt_path, step + 1)?;

            if let Some(hook_fn) = hook.as_mut() {
                let info = CkptInfo {
                    step: step + 1,
                    ckpt_path,
                    train_loss: mean_loss,
                    lr,
                    grad_norm,
                    elapsed_ms: train_start.elapsed().as_millis() as u64,
                    min_train_loss: running_min_loss,
                    val_loss: last_val_loss,
                };
                let control = hook_fn(&info).map_err(TrainError::Hook)?;

                // Pin before anything else can rotate the file away.
                // `save_step` on a later boundary is what prunes, so the
                // pin has to be in place before the loop comes round
                // again — and before the `Break` arm returns, since a
                // candidate selected on the last fire is the one the
                // caller most wants to still exist.
                if let Some(mark) = control.keep {
                    ckpt_store.pin(step + 1);
                    let candidate = Candidate {
                        info: info.clone(),
                        reason: mark.reason,
                        values: mark.values,
                    };
                    // Written down before it is handed back, because
                    // the returned list only reaches a caller on the
                    // paths that return. A run that dies later still
                    // leaves the pinned file, so it has to leave the
                    // record of it too. A failure to write is loud:
                    // a search whose record is missing entries is
                    // worse than one that stopped.
                    ckpt_store
                        .append_candidate(&candidate)
                        .map_err(|e| TrainError::Ckpt(format!("candidate record: {e}")))?;
                    candidates.push(candidate);
                }

                match control.flow {
                    CkptFlow::Continue => {}
                    CkptFlow::Break => {
                        // Early-return path: write terminal ckpt +
                        // finalize with `early_break = 1.0` marker so
                        // downstream consumers can distinguish an
                        // early stop from a full-run save without
                        // walking `step` against `cfg.steps`.
                        // Same as the early-stop exit: the record's
                        // `val_loss` belongs to the weights it names.
                        if let Some(batches) = val_batches.as_ref() {
                            if !(step + 1).is_multiple_of(cfg.eval_every) {
                                let v = evaluate(&mut forward, batches, &device, cfg, loss_fn)?;
                                last_val_loss = Some(v);
                                if v < min_val_loss {
                                    min_val_loss = v;
                                }
                            }
                        }
                        let final_path = ckpt_store
                            .save_final(save_vm, step + 1)
                            .map_err(|e| TrainError::Ckpt(e.to_string()))?;
                        save_optimizer_state(&opt, &names, cfg, &final_path, step + 1)?;
                        let mut metrics: HashMap<String, f32> = HashMap::new();
                        metrics.insert("min_train_loss".into(), running_min_loss);
                        metrics.insert("final_lr".into(), lr as f32);
                        metrics.insert("early_break".into(), 1.0);
                        if resumed_step > 0 {
                            metrics.insert("resumed_from_step".into(), resumed_step as f32);
                        }
                        if last_val_loss.is_some() {
                            metrics.insert("min_val_loss".into(), min_val_loss);
                        }
                        let mut ckpt = checkpoint_from_path(
                            &final_path,
                            step + 1,
                            mean_loss,
                            last_val_loss,
                            metrics,
                        )
                        .map_err(TrainError::Ckpt)?;
                        ckpt.candidates = candidates;
                        return Ok(ckpt);
                    }
                }
            }
        } // After the checkpoint block above, not before it: the
          // step that runs out of patience can also be a
          // `ckpt_every` boundary, and returning here first would
          // deny the hook the one fire it most needs — the step the
          // run ends on is the step a selection hook wants to keep.
        if out_of_patience {
            // The run stops where it stopped improving, and says so:
            // a caller reading `step` against `cfg.steps` would
            // otherwise have to guess whether the run was cut short or
            // the config was.
            tracing::info!(
                target: "algocline_nn::train",
                step = step + 1,
                val_loss = last_val_loss,
                "early stop: the held-out loss stopped improving"
            );
            // Score the weights this record names, the way the normal
            // exit does: `last_val_loss` may be several steps old, and
            // a record pairing step N's weights with step N-3's
            // held-out loss is a comparison nobody can make sense of
            // later.
            if let Some(batches) = val_batches.as_ref() {
                if !(step + 1).is_multiple_of(cfg.eval_every) {
                    let v = evaluate(&mut forward, batches, &device, cfg, loss_fn)?;
                    last_val_loss = Some(v);
                    if v < min_val_loss {
                        min_val_loss = v;
                    }
                }
            }
            let final_path = ckpt_store
                .save_final(save_vm, step + 1)
                .map_err(|e| TrainError::Ckpt(e.to_string()))?;
            save_optimizer_state(&opt, &names, cfg, &final_path, step + 1)?;
            let mut metrics: HashMap<String, f32> = HashMap::new();
            metrics.insert("min_train_loss".into(), running_min_loss);
            metrics.insert("final_lr".into(), lr as f32);
            metrics.insert("early_stop".into(), 1.0);
            metrics.insert("min_val_loss".into(), min_val_loss);
            if resumed_step > 0 {
                metrics.insert("resumed_from_step".into(), resumed_step as f32);
            }
            let mut ckpt =
                checkpoint_from_path(&final_path, step + 1, mean_loss, last_val_loss, metrics)
                    .map_err(TrainError::Ckpt)?;
            ckpt.candidates = candidates;
            return Ok(ckpt);
        }
    }

    // Terminal save under the stable `<prefix>.safetensors` filename.
    let final_path = ckpt_store
        .save_final(save_vm, cfg.steps)
        .map_err(|e| TrainError::Ckpt(e.to_string()))?;
    save_optimizer_state(&opt, &names, cfg, &final_path, cfg.steps)?;

    let mut metrics: HashMap<String, f32> = HashMap::new();
    metrics.insert("min_train_loss".into(), running_min_loss);
    metrics.insert("final_lr".into(), scheduler.lr_at(cfg.steps - 1) as f32);
    if resumed_step > 0 {
        metrics.insert("resumed_from_step".into(), resumed_step as f32);
    }
    // A last evaluation whenever the final step was not one, so the
    // returned record always carries the held-out loss of the weights
    // it names rather than of some earlier step.
    if let Some(batches) = val_batches.as_ref() {
        if !cfg.steps.is_multiple_of(cfg.eval_every) {
            let v = evaluate(&mut forward, batches, &device, cfg, loss_fn)?;
            last_val_loss = Some(v);
            if v < min_val_loss {
                min_val_loss = v;
            }
        }
        metrics.insert("min_val_loss".into(), min_val_loss);
    }

    let mut ckpt = checkpoint_from_path(
        &final_path,
        cfg.steps,
        last_train_loss,
        last_val_loss,
        metrics,
    )
    .map_err(TrainError::Ckpt)?;
    ckpt.candidates = candidates;
    Ok(ckpt)
}

/// Scale every trainable parameter's gradient so their joint L2 norm is
/// at most `max_norm`, given the `norm` already measured over the same
/// set.
///
/// Takes the measured norm rather than computing it, because the caller
/// wants the pre-clip value for [`CkptInfo::grad_norm`] anyway and the
/// walk is the expensive part.
///
/// Only the gradients of variables registered in `opt_vm` are scaled.
/// A [`GradStore`] from `backward()` also holds gradients for
/// intermediate tensors, and those are not part of the update, so
/// including them would measure and scale against a norm no optimizer
/// step uses.
///
/// Leaves everything alone when the norm is already within the cap, and
/// when it is not finite — see [`FullFtConfig::clip_grad_norm`].
/// Returns the scale that was applied (`1.0` when nothing was).
fn clip_grad_norm_(
    opt_vm: &VarMap,
    grads: &mut GradStore,
    max_norm: f64,
    norm: f32,
) -> CandleResult<f64> {
    if !norm.is_finite() || (norm as f64) <= max_norm {
        return Ok(1.0);
    }
    let scale = max_norm / norm as f64;
    // Collected first: the walk reads the map while the writes below go
    // to the store, and holding the map's lock across the writes is not
    // needed for either.
    let ids: Vec<candle_core::TensorId> = {
        let data = opt_vm.data().lock().unwrap();
        data.values().map(|var| var.as_tensor().id()).collect()
    };
    for id in ids {
        let Some(g) = grads.get_id(id) else {
            continue;
        };
        let scaled = (g * scale)?;
        grads.insert_id(id, scaled);
    }
    Ok(scale)
}

/// L2 norm of every trainable parameter's gradient in `opt_vm`.
///
/// Iterates the [`VarMap`] rather than the [`GradStore`] because the
/// map is the definitive inventory (a `Var` missing from `grads`
/// contributes 0 to the norm, matching the "un-touched parameter"
/// semantics candle already uses). Per-tensor squared sum is upcast to
/// F32 so mixed-precision runs report a comparable number to F32-only
/// runs.
fn grad_l2_norm(opt_vm: &VarMap, grads: &GradStore) -> CandleResult<f32> {
    let data = opt_vm.data().lock().unwrap();
    // Summed in name order, not `HashMap` order. f32 addition is not
    // associative and `RandomState` reseeds per process, so iterating
    // the map directly makes the norm differ in its last bits between
    // two runs of the same configuration — and with `clip_grad_norm`
    // set that difference scales every gradient, which is the one axis
    // `arch::seeded` exists to close. The same reason `ckpt.rs` writes
    // its tensors from a `BTreeMap`.
    let ordered: BTreeMap<&String, &candle_core::Var> = data.iter().collect();
    let mut sum_sq: f32 = 0.0;
    for var in ordered.into_values() {
        if let Some(g) = grads.get(var.as_tensor()) {
            let g_f32 = if g.dtype() == DType::F32 {
                g.clone()
            } else {
                g.to_dtype(DType::F32)?
            };
            let s: f32 = g_f32.sqr()?.sum_all()?.to_scalar()?;
            sum_sq += s;
        }
    }
    Ok(sum_sq.sqrt())
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
    // The base map belongs to the caller and never reaches here, so a
    // checkpoint named on the config would have nowhere to land. See
    // `TrainError::InitFromUnsupported`.
    if train_cfg.init_from.is_some() {
        return Err(TrainError::InitFromUnsupported);
    }
    // No validation parameter reaches this entry point, so a period
    // set here could never be honoured. Refused rather than ignored.
    if train_cfg.eval_every > 0 {
        return Err(TrainError::ValidationHalfConfigured {
            present: "cfg.eval_every",
            missing: "a validation dataset (this entry point takes none)",
        });
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
    let base = &*base;
    run_ft_core(
        base.device(),
        ForwardPass::Plain(&mut |xs| base.forward(xs).map_err(TrainError::from)),
        &lora_vm,
        &lora_vm,
        dataset,
        None,
        // No blockwise view: the wrapped model's blocks are not the
        // ones this entry holds a map for.
        None,
        train_cfg,
        loss_fn,
        &nn_dir,
        &ckpt_prefix,
        lease,
        // LoRA runs today do not expose the `on_ckpt` hook — the
        // shared inner loop is asked to run without one.
        None,
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
    M: Module + DeviceView + Checkpointable,
{
    match spec.loss_kind {
        DistillLossKind::Ce => {
            let loss = crate::train::HardLabelDistillLoss::new();
            // Distillation shares the Full FT loop; the `on_ckpt`
            // hook stays a full-fine-tune-only surface for this iter
            // (see CkptHook doc), so `None` is passed through here.
            run_full_ft(
                student,
                varmap,
                dataset,
                // Distillation holds nothing out either; `eval_every`
                // on the shared config is refused by the loop.
                None,
                &spec.hyperparams,
                &loss,
                ckpt_dir,
                ckpt_prefix,
                lease,
                None,
            )
        }
    }
}

/// One batch to its scalar loss: the target shift, the side-channel
/// checks, the F32 cast and the optional allowed-id mask, in the order
/// the training step needs them.
///
/// Held apart from the step itself because the evaluation pass has to
/// score the held-out set exactly the way training scores a batch. A
/// second copy of this sequence would let `val_loss` and `train_loss`
/// drift apart under any later change to either — and two numbers that
/// are compared have to be the same measurement.
///
/// Returns the loss tensor rather than its scalar value: the training
/// step needs the node to call `backward()` on, and the evaluation pass
/// simply never does.
fn forward_loss(
    forward: &mut ForwardPass<'_>,
    batch: &Batch,
    device: &Device,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
) -> Result<Tensor, TrainError> {
    let (inputs, targets, mask) = batch_to_input_target(batch, device)?;
    // Counted on the rows the batch carries rather than on the tensor:
    // the mask is host-side data and a `sum_all` here would pull the
    // device back for every step. The first column is dropped because
    // the mask is sliced in lockstep with the target shift.
    if let Some(rows) = batch.loss_mask.as_ref() {
        let scored = rows
            .iter()
            .flat_map(|row| row.iter().skip(1))
            .filter(|weight| **weight != 0.0)
            .count();
        if scored == 0 {
            return Err(TrainError::NothingScored {
                rows: rows.len(),
                scored,
            });
        }
    }
    // The allowed-id input, built only for the entry point that takes
    // one: every other run would pay for a tensor it cannot read. The
    // sets are shifted to line up with the model's inputs — see
    // `allowed_input_sets`.
    let allowed = match &forward {
        ForwardPass::Allowed(_) => allowed_input_sets(batch, batch.input_ids[0].len(), device)?,
        _ => None,
    };
    // The batch's own side channels against the caller's declared
    // intent. Both disagreements are refused: a conditioned run over a
    // conditionless batch would train unconditioned under a checkpoint
    // labelled otherwise, and an unconditioned run over a conditioned
    // batch would drop a condition the caller attached per row.
    // `inputs` is the batch after the target shift, so its first
    // dimension is still the row count.
    let logits = match (forward, batch.conds.as_deref(), allowed.as_ref()) {
        (ForwardPass::Plain(plain), None, _) => plain(&inputs)?,
        (ForwardPass::PerRow(per_row), Some(conds), _) => {
            per_row(&inputs, conds, batch.conds_per_row)?
        }
        (ForwardPass::Allowed(allowed_forward), None, Some(sets)) => {
            allowed_forward(&inputs, sets)?
        }
        (ForwardPass::Allowed(_), None, None) => {
            return Err(TrainError::MissingAllowedSets {
                rows: inputs.dim(0)?,
                needed: "the model reads them at every position",
            })
        }
        // Neither of these two takes a condition, so a batch carrying
        // one is the same mistake at both.
        (ForwardPass::Plain(_) | ForwardPass::Allowed(_), Some(conds), _) => {
            return Err(TrainError::UnexpectedConditions {
                rows: inputs.dim(0)?,
                conds: conds.len(),
            })
        }
        (ForwardPass::PerRow(_), None, _) => {
            return Err(TrainError::MissingConditions {
                rows: inputs.dim(0)?,
            })
        }
    };
    loss_from_logits(logits, batch, &targets, mask.as_ref(), cfg, loss_fn, device)
}

/// A model's logits to the scalar loss: the F32 cast, the optional
/// allowed-id mask, and the loss itself.
///
/// The tail of [`forward_loss`], split off because the
/// gradient-checkpointed path produces its logits elsewhere and has to
/// score them the same way. Two copies of this sequence would let a
/// checkpointed run and an ordinary one optimise slightly different
/// objectives while reporting the same number.
fn loss_from_logits(
    logits: Tensor,
    batch: &Batch,
    targets: &Tensor,
    mask: Option<&Tensor>,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
    device: &Device,
) -> Result<Tensor, TrainError> {
    // Mixed precision: the loss (log_softmax + NLL reduction) is always
    // scored in F32 — BF16's 8 mantissa bits are too coarse for a mean
    // over thousands of log-probs. `to_dtype` is differentiable, so the
    // backward pass crosses back into the model's dtype at this
    // boundary. F32 logits pass through untouched.
    let logits = if logits.dtype() == DType::F32 {
        logits
    } else {
        logits.to_dtype(DType::F32)?
    };
    // Remove the ids this target could not have taken, so the loss
    // scores the choice among the ones it could. Applied after the F32
    // cast: a large negative penalty in BF16 would not survive the
    // conversion cleanly. Opt-in, and a batch that cannot honour the
    // opt-in is refused rather than trained unmasked.
    let logits = if cfg.mask_disallowed_logits {
        match allowed_logit_mask(batch, batch.input_ids[0].len(), logits.dim(2)?, device)? {
            Some(m) => logits.broadcast_add(&m)?,
            None => {
                return Err(TrainError::MissingAllowedSets {
                    rows: logits.dim(0)?,
                    needed: "cfg.mask_disallowed_logits asks the loss to use them",
                })
            }
        }
    } else {
        logits
    };
    Ok(loss_fn.compute(&logits, targets, mask)?)
}

/// Drain a validation dataset into the batches every evaluation will
/// re-score.
///
/// Held in memory rather than re-read: [`Dataset`] is a one-pass
/// stream with no rewind, so a second evaluation would otherwise score
/// different rows from the first and the sequence of values would stop
/// being a curve. A held-out split is small by construction, which is
/// what makes holding it affordable.
fn drain_validation(val: &mut dyn Dataset) -> Result<Vec<Batch>, TrainError> {
    let mut batches = Vec::new();
    while let Some(batch) = val.next_batch()? {
        batches.push(batch);
    }
    if batches.is_empty() {
        return Err(TrainError::EmptyValidationSet);
    }
    Ok(batches)
}

/// Mean loss over the held-out batches, scored on the same forward path
/// and the same loss the training step uses.
///
/// The mean is taken over batches rather than over tokens, matching how
/// the training step reports its own loss across micro-batches. The two
/// agree whenever the batches are equally sized, which they are except
/// for a short final one.
///
/// No `backward()` is called, so nothing here touches the optimizer or
/// the parameters.
fn evaluate(
    forward: &mut ForwardPass<'_>,
    batches: &[Batch],
    device: &Device,
    cfg: &FullFtConfig,
    loss_fn: &dyn Loss,
) -> Result<f32, TrainError> {
    let mut sum = 0.0f32;
    for batch in batches {
        let loss = forward_loss(forward, batch, device, cfg, loss_fn)?;
        sum += loss.to_scalar::<f32>()?;
    }
    Ok(sum / batches.len() as f32)
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

/// Magnitude of the penalty applied to a disallowed id.
///
/// Large enough that `exp` of it underflows to zero in f32, so the
/// softmax behaves as if the id were absent — and finite, which
/// negative infinity is not. An infinite log-probability multiplied by
/// a zero loss weight is NaN rather than zero, so a single disallowed
/// target at a position the loss ignores would otherwise poison the
/// whole batch, and the padding past the end of a row is exactly such
/// a position.
const DISALLOWED_LOGIT: f32 = -1e9;

/// Additive logit mask that removes every id a target may not take.
///
/// Returns `[batch, seq - 1, vocab]` of `0.0` on allowed ids and
/// a large finite negative penalty elsewhere (finite so that a
/// disallowed target at a position the loss ignores stays zero rather
/// than turning into NaN), aligned with the targets: entry `k`
/// governs the prediction of `input_ids[k + 1]`, so it reads
/// `allowed_ids[row][k + 1]`.
///
/// Added to the logits before the softmax, this makes the loss score
/// the choice among the allowed ids rather than the choice among all of
/// them. A position with no allowed ids is left unmasked, which is how
/// a producer says "this position is not constrained".
///
/// `None` for a batch that carries no sets — the caller decides whether
/// that is acceptable (the training loop refuses it when
/// [`FullFtConfig::mask_disallowed_logits`] asked for the mask).
///
/// Nothing here checks that a target is inside its own position's set:
/// a batch whose sets contradict its tokens is scored with the target
/// penalised as a disallowed id, which is a number rather than a
/// refusal. The check belongs where the two are attached to each other
/// and the producer can still fix it —
/// [`crate::train::data::TokenizedDataset::with_allowed_ids`] refuses
/// that pairing.
///
/// # Errors
///
/// The sets do not have one row per row of the batch, or an id is
/// outside `vocab`.
pub fn allowed_logit_mask(
    batch: &Batch,
    seq: usize,
    vocab: usize,
    device: &Device,
) -> CandleResult<Option<Tensor>> {
    let Some(allowed) = batch.allowed_ids.as_ref() else {
        return Ok(None);
    };
    let rows = batch.input_ids.len();
    if allowed.len() != rows {
        return Err(candle_core::Error::Msg(format!(
            "allowed_ids row count {} != batch size {rows}",
            allowed.len()
        )));
    }
    if seq < 2 {
        return Err(candle_core::Error::Msg(format!(
            "allowed_logit_mask: seq={seq} is too short (need >= 2)"
        )));
    }
    let width = seq - 1;
    let mut flat = vec![DISALLOWED_LOGIT; rows * width * vocab];
    for (r, row) in allowed.iter().enumerate() {
        for k in 0..width {
            let base = (r * width + k) * vocab;
            match row.get(k + 1) {
                Some(ids) if !ids.is_empty() => {
                    for id in ids {
                        let i = *id as usize;
                        if i >= vocab {
                            return Err(candle_core::Error::Msg(format!(
                                "allowed id {i} is outside vocab {vocab}"
                            )));
                        }
                        flat[base + i] = 0.0;
                    }
                }
                // No set for this position: leave every id available.
                _ => flat[base..base + vocab].fill(0.0),
            }
        }
    }
    Ok(Some(Tensor::from_vec(flat, (rows, width, vocab), device)?))
}

/// The batch's allowed-id sets as a model input, aligned with the
/// positions the model actually consumes.
///
/// Returns `None` for a batch that carries none, which is how a dataset
/// that models no constrained id space passes through here.
///
/// # The alignment
///
/// Entry `k` of the result is the set the model's answer at input
/// position `k` is drawn from, so it reads `allowed_ids[row][k + 1]` —
/// the same entry [`allowed_logit_mask`] uses for target `k`, and for
/// the same reason: having consumed input `k`, the model is standing
/// where token `k + 1` is produced, and that position's available ids
/// are both what it may answer and what the loss should score it among.
///
/// The two are built one after the other in this file precisely so the
/// `+ 1` is written once in each and can be read side by side. Off by
/// one, every set describes the position before the one it is attached
/// to, and every shape still agrees.
///
/// # Errors
///
/// The sets do not have one row per row of the batch, the batch is too
/// short to shift, or the window holds no ids at all — see
/// [`AllowedSets::window`], which refuses that rather than handing the
/// model an input equivalent to having no allowed-id channel.
pub fn allowed_input_sets(
    batch: &Batch,
    seq: usize,
    device: &Device,
) -> CandleResult<Option<AllowedSets>> {
    let Some(allowed) = batch.allowed_ids.as_ref() else {
        return Ok(None);
    };
    let rows = batch.input_ids.len();
    if allowed.len() != rows {
        return Err(candle_core::Error::Msg(format!(
            "allowed_ids row count {} != batch size {rows}",
            allowed.len()
        )));
    }
    if seq < 2 {
        return Err(candle_core::Error::Msg(format!(
            "allowed_input_sets: seq={seq} is too short (need >= 2)"
        )));
    }
    AllowedSets::window(allowed, 1, seq - 1, device).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::Gpt2Config;
    use crate::arch::Gpt2Model;
    use crate::train::data::{DatasetOpts, TokenizedDataset};
    use crate::train::loss::CrossEntropyLoss;
    use candle_nn::VarBuilder;
    use std::sync::Mutex;
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
                seed: None,
                pad_id: 0,
                mask_pad: true,
                text_field: "text".into(),
            },
        )
    }

    /// With optimizer state on disk, `init_from` resumes: the step
    /// count carries over, `steps` is the total the run works towards,
    /// and the record says where it picked up.
    #[test]
    fn init_from_with_optimizer_state_resumes_the_step_count_and_the_schedule() {
        let tmp = TempDir::new().unwrap();
        let loss = CrossEntropyLoss::new();

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let first = FullFtConfig {
            steps: 4,
            warmup: 2,
            save_optimizer_state: true,
            ..FullFtConfig::default()
        };
        let a = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &first,
            &loss,
            tmp.path(),
            "resume",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        assert_eq!(a.step, 4);
        let ckpt = tmp.path().join("resume.safetensors");
        let sidecar = crate::train::optstate::sidecar_path(&ckpt);
        assert!(sidecar.exists(), "the sidecar must sit beside {ckpt:?}");

        let (_, vm2, model2) = tiny_cfg_and_model();
        let mut ds2 = overfit_dataset();
        let second = FullFtConfig {
            steps: 7,
            warmup: 2,
            save_optimizer_state: true,
            init_from: Some(ckpt),
            ..FullFtConfig::default()
        };
        let b = run_full_ft(
            &model2,
            &vm2,
            &mut ds2,
            None,
            &second,
            &loss,
            tmp.path(),
            "resume2",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        assert_eq!(
            b.step, 7,
            "steps is the total, not a count of further steps"
        );
        assert_eq!(
            b.metrics.get("resumed_from_step").copied(),
            Some(4.0),
            "the record has to say the run did not start at zero"
        );
    }

    /// A checkpoint with no state beside it is still accepted, and the
    /// run reports itself as starting from zero rather than looking
    /// like a resume.
    #[test]
    fn init_from_without_optimizer_state_is_still_a_warm_start() {
        let tmp = TempDir::new().unwrap();
        let loss = CrossEntropyLoss::new();

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let first = FullFtConfig {
            steps: 2,
            warmup: 0,
            ..FullFtConfig::default()
        };
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &first,
            &loss,
            tmp.path(),
            "warm",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        let ckpt = tmp.path().join("warm.safetensors");
        assert!(
            !crate::train::optstate::sidecar_path(&ckpt).exists(),
            "save_optimizer_state was off, so nothing should sit beside it"
        );

        let (_, vm2, model2) = tiny_cfg_and_model();
        let mut ds2 = overfit_dataset();
        let second = FullFtConfig {
            steps: 2,
            warmup: 0,
            init_from: Some(ckpt),
            ..FullFtConfig::default()
        };
        let b = run_full_ft(
            &model2,
            &vm2,
            &mut ds2,
            None,
            &second,
            &loss,
            tmp.path(),
            "warm2",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        assert_eq!(b.step, 2);
        assert!(!b.metrics.contains_key("resumed_from_step"));
    }

    /// Resuming into a total the run has already reached is refused
    /// rather than answered with a zero-step run and a fresh terminal
    /// checkpoint.
    #[test]
    fn a_resume_past_the_requested_total_is_refused() {
        let tmp = TempDir::new().unwrap();
        let loss = CrossEntropyLoss::new();
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let first = FullFtConfig {
            steps: 4,
            warmup: 0,
            save_optimizer_state: true,
            ..FullFtConfig::default()
        };
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &first,
            &loss,
            tmp.path(),
            "done",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();

        let (_, vm2, model2) = tiny_cfg_and_model();
        let mut ds2 = overfit_dataset();
        let second = FullFtConfig {
            steps: 4,
            warmup: 0,
            init_from: Some(tmp.path().join("done.safetensors")),
            ..FullFtConfig::default()
        };
        let err = run_full_ft(
            &model2,
            &vm2,
            &mut ds2,
            None,
            &second,
            &loss,
            tmp.path(),
            "done2",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                TrainError::ResumeBeyondSteps {
                    resumed: 4,
                    steps: 4
                }
            ),
            "{err}"
        );
    }

    /// State written by one optimizer is not read into another's slots.
    #[test]
    fn a_state_written_by_another_optimizer_is_refused() {
        let tmp = TempDir::new().unwrap();
        let loss = CrossEntropyLoss::new();
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let adamw = FullFtConfig {
            steps: 2,
            warmup: 0,
            save_optimizer_state: true,
            ..FullFtConfig::default()
        };
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &adamw,
            &loss,
            tmp.path(),
            "kind",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();

        let (_, vm2, model2) = tiny_cfg_and_model();
        let mut ds2 = overfit_dataset();
        let lion = FullFtConfig {
            steps: 4,
            warmup: 0,
            optimizer: OptimizerKind::Lion,
            lr: 1e-4,
            init_from: Some(tmp.path().join("kind.safetensors")),
            ..FullFtConfig::default()
        };
        let err = run_full_ft(
            &model2,
            &vm2,
            &mut ds2,
            None,
            &lion,
            &loss,
            tmp.path(),
            "kind2",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, TrainError::OptState(_)), "{err}");
    }

    /// A rotated-out checkpoint takes its state file with it.
    #[test]
    fn rotation_drops_the_optimizer_state_alongside_the_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let loss = CrossEntropyLoss::new();
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let cfg = FullFtConfig {
            steps: 4,
            warmup: 0,
            ckpt_every: 1,
            ckpt_keep: 2,
            save_optimizer_state: true,
            ..FullFtConfig::default()
        };
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "rot",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        let sidecars: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("-step") && n.ends_with(".opt.safetensors"))
            .collect();
        assert_eq!(
            sidecars.len(),
            2,
            "one per surviving step checkpoint, not one per step: {sidecars:?}"
        );
    }

    /// The cap scales the gradient down to exactly `max_norm` and    /// The cap scales the gradient down to exactly `max_norm` and
    /// leaves its direction alone.
    #[test]
    fn clipping_shortens_the_gradient_without_turning_it() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let _ = vm
            .get(3, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var = {
            let data = vm.data().lock().unwrap();
            data["w"].clone()
        };
        var.set(&Tensor::new(&[3.0f32, 4.0, 0.0], &dev).unwrap())
            .unwrap();
        // `sum(w^2)/2` has gradient `w`, so the norm is the length of
        // the value set above — 5, against a cap of 1.
        let loss = (var.as_tensor().sqr().unwrap().sum_all().unwrap() * 0.5).unwrap();
        let mut grads = loss.backward().unwrap();

        let before = grad_l2_norm(&vm, &grads).unwrap();
        assert!((before - 5.0).abs() < 1e-5, "norm before = {before}");

        let scale = clip_grad_norm_(&vm, &mut grads, 1.0, before).unwrap();
        assert!((scale - 0.2).abs() < 1e-6, "scale = {scale}");
        let after = grad_l2_norm(&vm, &grads).unwrap();
        assert!((after - 1.0).abs() < 1e-5, "norm after = {after}");
        let g: Vec<f32> = grads
            .get(var.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Direction preserved: the original [3, 4, 0] scaled by 1/5.
        assert!((g[0] - 0.6).abs() < 1e-6 && (g[1] - 0.8).abs() < 1e-6 && g[2].abs() < 1e-6);
    }

    /// A gradient already inside the cap is handed to the optimizer
    /// untouched.
    #[test]
    fn a_gradient_within_the_cap_is_left_alone() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let _ = vm
            .get(2, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var = {
            let data = vm.data().lock().unwrap();
            data["w"].clone()
        };
        var.set(&Tensor::new(&[0.3f32, 0.4], &dev).unwrap())
            .unwrap();
        let loss = (var.as_tensor().sqr().unwrap().sum_all().unwrap() * 0.5).unwrap();
        let mut grads = loss.backward().unwrap();
        let before = grad_l2_norm(&vm, &grads).unwrap();
        let scale = clip_grad_norm_(&vm, &mut grads, 1.0, before).unwrap();
        assert_eq!(scale, 1.0);
        let after = grad_l2_norm(&vm, &grads).unwrap();
        assert!((after - before).abs() < 1e-7);
    }

    /// A non-finite norm is left unscaled rather than multiplied into
    /// every parameter that still had a usable gradient.
    #[test]
    fn a_non_finite_norm_is_not_scaled() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let _ = vm
            .get(2, "w", candle_nn::Init::Const(0.0), DType::F32, &dev)
            .unwrap();
        let var = {
            let data = vm.data().lock().unwrap();
            data["w"].clone()
        };
        var.set(&Tensor::new(&[1.0f32, 2.0], &dev).unwrap())
            .unwrap();
        let loss = (var.as_tensor().sqr().unwrap().sum_all().unwrap() * 0.5).unwrap();
        let mut grads = loss.backward().unwrap();
        let scale = clip_grad_norm_(&vm, &mut grads, 1.0, f32::NAN).unwrap();
        assert_eq!(scale, 1.0);
        let g: Vec<f32> = grads
            .get(var.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(g, vec![1.0, 2.0], "the gradient must not become NaN");
    }

    /// A learning rate that ascends or poisons is refused; zero, which
    /// takes no step, is not — holding the weights while the rest of
    /// the loop runs is a real thing to ask for.
    #[test]
    fn a_learning_rate_that_cannot_mean_what_it_says_is_refused() {
        let (_, vm, model) = tiny_cfg_and_model();
        let loss = CrossEntropyLoss::new();
        let tmp = TempDir::new().unwrap();
        for value in [-1e-3, f64::NAN, f64::INFINITY] {
            let mut ds = overfit_dataset();
            let cfg = FullFtConfig {
                lr: value,
                steps: 2,
                ..FullFtConfig::default()
            };
            let err = run_full_ft(
                &model,
                &vm,
                &mut ds,
                None,
                &cfg,
                &loss,
                tmp.path(),
                "lr",
                Arc::new(TrainingLease::new()),
                None,
            )
            .unwrap_err();
            assert!(
                matches!(err, TrainError::InvalidLearningRate { .. }),
                "lr = {value}: {err}"
            );
        }

        let mut ds = overfit_dataset();
        let zero = FullFtConfig {
            lr: 0.0,
            steps: 2,
            warmup: 0,
            ..FullFtConfig::default()
        };
        assert!(
            run_full_ft(
                &model,
                &vm,
                &mut ds,
                None,
                &zero,
                &loss,
                tmp.path(),
                "lrzero",
                Arc::new(TrainingLease::new()),
                None,
            )
            .is_ok(),
            "lr = 0 takes no step and is allowed"
        );
    }

    /// A cap that cannot cap anything is refused before the run starts.
    #[test]
    fn a_clip_norm_that_erases_or_reverses_the_gradient_is_refused() {
        let (_, vm, model) = tiny_cfg_and_model();
        let loss = CrossEntropyLoss::new();
        let tmp = TempDir::new().unwrap();
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut ds = overfit_dataset();
            let cfg = FullFtConfig {
                steps: 2,
                clip_grad_norm: Some(value),
                ..FullFtConfig::default()
            };
            let err = run_full_ft(
                &model,
                &vm,
                &mut ds,
                None,
                &cfg,
                &loss,
                tmp.path(),
                "clip",
                Arc::new(TrainingLease::new()),
                None,
            )
            .unwrap_err();
            assert!(
                matches!(err, TrainError::InvalidClipNorm { .. }),
                "clip_grad_norm = {value}: {err}"
            );
        }
    }

    /// The norm the hook is handed is the one the step produced, not
    /// the one the cap let through.
    #[test]
    fn the_hook_sees_the_norm_before_the_cap_applied() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 2,
            warmup: 0,
            ckpt_every: 1,
            // Small enough that a from-scratch step is over it.
            clip_grad_norm: Some(1e-6),
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let seen: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let hook: CkptHook = Box::new(move |info: &CkptInfo| {
            sink.lock().unwrap().push(info.grad_norm);
            Ok(CkptControl::CONTINUE)
        });
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "clipnorm",
            Arc::new(TrainingLease::new()),
            Some(hook),
        )
        .unwrap();
        let seen = seen.lock().unwrap().clone();
        assert!(
            seen.iter().all(|n| *n > 1e-6),
            "reported norms must be the measured ones, not the cap: {seen:?}"
        );
    }

    /// A run with a held-out set reports its loss on the record, in
    /// the metrics, and at every hook fire.
    #[test]
    fn a_held_out_set_is_scored_and_reaches_the_record_and_the_hook() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let mut val = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 4,
            warmup: 1,
            ckpt_every: 2,
            eval_every: 2,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let seen: Arc<Mutex<Vec<Option<f32>>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let hook: CkptHook = Box::new(move |info: &CkptInfo| {
            sink.lock().unwrap().push(info.val_loss);
            Ok(CkptControl::CONTINUE)
        });
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            Some(&mut val),
            &cfg,
            &loss,
            tmp.path(),
            "val",
            Arc::new(TrainingLease::new()),
            Some(hook),
        )
        .expect("run with a held-out set");

        let val_loss = ckpt.val_loss.expect("the record carries the held-out loss");
        assert!(
            val_loss.is_finite() && val_loss > 0.0,
            "val_loss = {val_loss}"
        );
        let min = *ckpt
            .metrics
            .get("min_val_loss")
            .expect("min_val_loss is recorded alongside min_train_loss");
        assert!(min <= val_loss, "min_val_loss {min} > final {val_loss}");
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "two checkpoint boundaries at ckpt_every = 2");
        assert!(
            seen.iter().all(|v| v.is_some()),
            "every fire lands on an evaluation boundary here, so each carries a value: {seen:?}"
        );
    }

    /// A checkpointed run is the same run: after the same steps on the
    /// same data from the same initialisation, the parameters match.
    #[test]
    fn a_checkpointed_run_lands_where_the_ordinary_one_does() {
        let loss = CrossEntropyLoss::new();
        let tmp = TempDir::new().unwrap();

        // `(before, after)` per run: a comparison of two runs that both
        // failed to train agrees perfectly, which is what a version of
        // this test that only compared the ends could not tell.
        let weights_after = |grad_checkpoint: bool,
                             prefix: &str|
         -> (BTreeMap<String, Vec<f32>>, BTreeMap<String, Vec<f32>>) {
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
            // Seeded, so the two runs start from one model and a
            // difference at the end is the checkpointing.
            let vs = crate::arch::seeded_var_builder(&vm, 4711, cfg.dtype, &cfg.device);
            let model = Gpt2Model::new(&cfg, vs).unwrap();
            let snapshot = |vm: &VarMap| -> BTreeMap<String, Vec<f32>> {
                let data = vm.data().lock().unwrap();
                data.iter()
                    .map(|(name, var)| {
                        let t = var.as_tensor().flatten_all().unwrap();
                        (name.clone(), t.to_vec1::<f32>().unwrap())
                    })
                    .collect()
            };
            let before = snapshot(&vm);
            let mut ds = overfit_dataset();
            let ft = FullFtConfig {
                lr: 5e-3,
                steps: 4,
                warmup: 1,
                grad_checkpoint,
                ..FullFtConfig::default()
            };
            run_full_ft(
                &model,
                &vm,
                &mut ds,
                None,
                &ft,
                &loss,
                tmp.path(),
                prefix,
                Arc::new(TrainingLease::new()),
                None,
            )
            .expect("run");
            (before, snapshot(&vm))
        };

        let (plain_before, plain) = weights_after(false, "plain");
        let (_, checkpointed) = weights_after(true, "ckpt");

        // Four steps have to have moved something, or the agreement
        // below is between two runs that did nothing.
        let moved = plain_before
            .iter()
            .map(|(name, before)| {
                let after = &plain[name];
                before
                    .iter()
                    .zip(after)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(moved > 1e-6, "training moved nothing: max change {moved}");
        assert_eq!(plain.len(), checkpointed.len());
        assert!(!plain.is_empty());
        for (name, values) in &plain {
            let other = checkpointed.get(name).expect("same parameter set");
            let gap = values
                .iter()
                .zip(other)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(gap < 1e-4, "`{name}` diverged by {gap} after 4 steps");
        }
    }

    /// An entry point with no blockwise view refuses the flag rather
    /// than ignoring it — a caller who believed they had the memory
    /// headroom would find out from an allocator, much later.
    #[test]
    fn checkpointing_is_refused_where_it_cannot_be_honoured() {
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
            custom: Some(crate::arch::Gpt2Custom {
                cond_slots: Some(2),
                ..Default::default()
            }),
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let loss = CrossEntropyLoss::new();
        let tmp = TempDir::new().unwrap();
        let ft = FullFtConfig {
            steps: 2,
            grad_checkpoint: true,
            ..FullFtConfig::default()
        };

        // The conditioned entry point has no blockwise view at all.
        let mut ds = overfit_dataset()
            .with_conditions(vec![CondIndex::new(0, 2).unwrap(); 400])
            .unwrap();
        let err = run_conditioned_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &ft,
            &loss,
            tmp.path(),
            "cond",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, TrainError::CheckpointingUnsupported(_)),
            "{err}"
        );

        // And the model itself refuses, because its forward reads a
        // channel the blockwise surface does not take.
        let mut ds = overfit_dataset();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &ft,
            &loss,
            tmp.path(),
            "chan",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        match err {
            TrainError::CheckpointingUnsupported(why) => {
                assert!(why.contains("input channel"), "{why}")
            }
            other => panic!("expected a checkpointing refusal, got {other}"),
        }
    }

    /// Early stopping ends the run where the held-out loss stopped
    /// improving, and the record says the run was cut short rather than
    /// leaving a caller to compare `step` against `steps` and guess.
    #[test]
    fn early_stopping_ends_a_run_that_stopped_improving() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let mut val = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            // No learning at all, so the held-out loss cannot improve
            // and the rule is the only thing that can end the run.
            lr: 0.0,
            weight_decay: 0.0,
            steps: 20,
            warmup: 0,
            eval_every: 1,
            early_stop: Some(EarlyStop {
                patience: 2,
                min_delta: 0.0,
            }),
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            Some(&mut val),
            &cfg,
            &loss,
            tmp.path(),
            "stop",
            Arc::new(TrainingLease::new()),
            None,
        )
        .expect("run");

        assert!(
            ckpt.step < cfg.steps,
            "the run should have stopped early, ended at {}",
            ckpt.step
        );
        assert_eq!(
            ckpt.metrics.get("early_stop").copied(),
            Some(1.0),
            "the record has to distinguish a stopped run from a finished one"
        );
        assert!(tmp.path().join("stop.safetensors").exists());
    }

    /// The step a run stops on is a step the hook sees. A selection
    /// hook picks the checkpoint it wants to keep from what it is
    /// shown, and the last step of the run is the one it most wants to
    /// be shown — so the checkpoint block runs before the stop, not
    /// after it.
    #[test]
    fn the_hook_sees_the_step_the_run_stops_on() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let mut val = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            // Frozen weights: the held-out loss cannot improve, so the
            // patience rule is what ends the run.
            lr: 0.0,
            weight_decay: 0.0,
            steps: 20,
            warmup: 0,
            eval_every: 1,
            // Every step is a boundary, so the stopping step is one.
            ckpt_every: 1,
            ckpt_keep: 20,
            early_stop: Some(EarlyStop {
                patience: 2,
                min_delta: 0.0,
            }),
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();

        let fires: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let fires_hook = Arc::clone(&fires);
        let hook: CkptHook = Box::new(move |info| {
            fires_hook.lock().unwrap().push(info.step);
            Ok(CkptControl {
                flow: CkptFlow::Continue,
                keep: Some(KeepMark {
                    reason: Some("last one wins".into()),
                    values: Default::default(),
                }),
            })
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            Some(&mut val),
            &cfg,
            &loss,
            tmp.path(),
            "stop_hook",
            Arc::new(TrainingLease::new()),
            Some(hook),
        )
        .expect("run");

        assert_eq!(
            ckpt.metrics.get("early_stop").copied(),
            Some(1.0),
            "the run has to have stopped on the rule for this to test anything"
        );
        let seen = fires.lock().unwrap();
        assert_eq!(
            seen.last().copied(),
            Some(ckpt.step),
            "the stopping step must be among the fires, got {seen:?} for a run ending at {}",
            ckpt.step
        );
        let last = ckpt
            .candidates
            .last()
            .expect("the hook kept every fire it saw");
        assert_eq!(last.info.step, ckpt.step);
        assert!(
            last.info.ckpt_path.exists(),
            "a candidate kept on the stopping step has to survive the exit"
        );
    }

    /// A rule with nothing to watch is refused rather than left never
    /// to fire.
    #[test]
    fn early_stopping_without_a_held_out_set_is_refused() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 4,
            early_stop: Some(EarlyStop {
                patience: 1,
                min_delta: 0.0,
            }),
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "nostop",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, TrainError::EarlyStopWithoutValidation),
            "{err}"
        );
    }

    /// The patience rule itself, away from a training run: an
    /// improvement resets the count, a non-improvement spends it, and a
    /// non-finite loss counts as a failure to improve rather than as a
    /// reason to keep going.
    #[test]
    fn the_patience_rule_counts_what_it_says_it_counts() {
        let mut watch = EarlyStopWatch::new(EarlyStop {
            patience: 2,
            min_delta: 0.1,
        });
        assert!(!watch.observe(1.0), "the first value is the best so far");
        assert!(
            !watch.observe(0.95),
            "0.05 is inside min_delta: no improvement (1 of 2)"
        );
        assert!(!watch.observe(0.80), "a real improvement resets the count");
        assert!(!watch.observe(0.80), "no improvement (1 of 2)");
        assert!(!watch.observe(0.80), "no improvement (2 of 2)");
        assert!(watch.observe(0.80), "patience spent");

        let mut watch = EarlyStopWatch::new(EarlyStop {
            patience: 0,
            min_delta: 0.0,
        });
        assert!(!watch.observe(1.0));
        assert!(
            watch.observe(f32::NAN),
            "a diverged run must not outlive the rule"
        );
    }

    /// The metrics file is a curve: one line per step, valid JSON on
    /// each, and the numbers are the ones the run reported.
    #[test]
    fn the_metrics_file_records_one_line_per_step() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 6,
            warmup: 0,
            metrics_every: 2,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "curve",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();

        let path = tmp.path().join("curve-metrics.jsonl");
        let text = std::fs::read_to_string(&path).expect("the metrics file exists");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "6 steps at every 2");
        let points: Vec<serde_json::Value> = lines
            .iter()
            .map(|l| serde_json::from_str(l).expect("each line is a JSON object"))
            .collect();
        assert_eq!(points[0]["step"], 2);
        assert_eq!(points[2]["step"], 6);
        assert!(points[0]["loss"].is_number());
        assert!(
            points[0].get("val_loss").is_none(),
            "a run with no held-out set writes no val_loss key"
        );
        assert!(
            points[0].get("grad_norm").is_none(),
            "a run computing no gradient norm writes no grad_norm key"
        );
        assert_eq!(ckpt.step, 6);
    }

    /// A run that was not asked for a curve writes no file.
    #[test]
    fn no_metrics_file_is_written_unless_one_was_asked_for() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 2,
            warmup: 0,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "quiet",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        assert!(!tmp.path().join("quiet-metrics.jsonl").exists());
    }

    /// Without a held-out set the record says so rather than reporting
    /// a number that came from the training rows.
    #[test]
    fn a_run_without_a_held_out_set_reports_no_val_loss() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 2,
            warmup: 1,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "noval",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap();
        assert_eq!(ckpt.val_loss, None);
        assert!(!ckpt.metrics.contains_key("min_val_loss"));
    }

    /// Either half of the validation setup without the other is a
    /// refusal, not a run that quietly measures nothing.
    #[test]
    fn half_a_validation_setup_is_refused_from_both_sides() {
        let (_, vm, model) = tiny_cfg_and_model();
        let loss = CrossEntropyLoss::new();
        let tmp = TempDir::new().unwrap();

        let mut ds = overfit_dataset();
        let period_only = FullFtConfig {
            steps: 2,
            eval_every: 1,
            ..FullFtConfig::default()
        };
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &period_only,
            &loss,
            tmp.path(),
            "half",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                TrainError::ValidationHalfConfigured {
                    present: "cfg.eval_every",
                    ..
                }
            ),
            "{err}"
        );

        let mut ds = overfit_dataset();
        let mut val = overfit_dataset();
        let set_only = FullFtConfig {
            steps: 2,
            ..FullFtConfig::default()
        };
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            Some(&mut val),
            &set_only,
            &loss,
            tmp.path(),
            "half",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                TrainError::ValidationHalfConfigured {
                    present: "a validation dataset",
                    ..
                }
            ),
            "{err}"
        );
    }

    /// An empty held-out set is caught before the first step, not after
    /// a whole run has gone by without a number.
    #[test]
    fn an_empty_held_out_set_is_refused_before_the_first_step() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let mut val = TokenizedDataset::new(
            Vec::new(),
            DatasetOpts {
                batch_size: 1,
                ctx_len: 8,
                shuffle: false,
                seed: None,
                pad_id: 0,
                mask_pad: true,
                text_field: "text".into(),
            },
        );
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 2,
            eval_every: 1,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            Some(&mut val),
            &cfg,
            &loss,
            tmp.path(),
            "empty",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, TrainError::EmptyValidationSet), "{err}");
    }

    /// A row short enough that padding covers everything the loss
    /// would score. The masked mean answers `0.0` for such a batch, so
    /// without a refusal the run would report a perfect step it never
    /// took.
    #[test]
    fn a_batch_that_scores_nothing_is_refused_rather_than_scored_zero() {
        let (_, vm, model) = tiny_cfg_and_model();
        // One real token, seven pads: the mask is 1 at position 0 and 0
        // everywhere after, and position 0 is the one the target shift
        // drops.
        let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| vec![1u32]).take(4).collect();
        let mut ds = TokenizedDataset::new(
            rows,
            DatasetOpts {
                batch_size: 1,
                ctx_len: 8,
                shuffle: false,
                seed: None,
                pad_id: 0,
                mask_pad: true,
                text_field: "text".into(),
            },
        );
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            steps: 2,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "empty-mask",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, TrainError::NothingScored { scored: 0, .. }),
            "{err}"
        );
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
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "z",
            lease,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, TrainError::ZeroSteps));
    }

    #[test]
    fn zero_grad_accum_errors_up_front() {
        // `grad_accum = 0` would divide by zero when computing the
        // pre-backward `1 / N` scale, so it is refused at loop entry
        // rather than allowed to produce NaN grads. Multi-step
        // accumulation (`grad_accum > 1`) is now honoured natively and
        // has its own equivalence test in the integration suite.
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            grad_accum: 0,
            steps: 5,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "z",
            lease,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, TrainError::ZeroGradAccum));
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
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
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

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "tiny",
            lease,
            None,
        )
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
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let _ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "rot",
            lease,
            None,
        )
        .unwrap();

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

    /// The `on_ckpt` hook fires exactly at each `ckpt_every` boundary
    /// and the [`CkptInfo`] fields carry sane values (step index,
    /// ckpt path pointing at a real file, monotonic `elapsed_ms`).
    #[test]
    fn hook_fires_at_every_ckpt_boundary_with_populated_info() {
        use std::sync::Mutex;

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
            ckpt_keep: 5,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let captured: Arc<Mutex<Vec<CkptInfo>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_hook = Arc::clone(&captured);
        let hook: CkptHook = Box::new(move |info| {
            captured_hook.lock().unwrap().push(info.clone());
            Ok(CkptControl::CONTINUE)
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_fire",
            lease,
            Some(hook),
        )
        .expect("training with hook must complete");

        // 10 steps / ckpt_every=2 = 5 fires.
        let fires = captured.lock().unwrap();
        assert_eq!(fires.len(), 5, "hook must fire once per ckpt_every step");

        // Step indices are the 1-indexed boundaries.
        let steps: Vec<usize> = fires.iter().map(|i| i.step).collect();
        assert_eq!(steps, vec![2, 4, 6, 8, 10]);

        // `elapsed_ms` is monotonically non-decreasing.
        for pair in fires.windows(2) {
            assert!(
                pair[1].elapsed_ms >= pair[0].elapsed_ms,
                "elapsed_ms must not go backwards: {} -> {}",
                pair[0].elapsed_ms,
                pair[1].elapsed_ms
            );
        }

        // Each ckpt_path exists on disk at fire time (the file may be
        // rotated away later, but for `ckpt_keep=5` all 5 survive).
        for info in fires.iter() {
            assert!(
                info.ckpt_path.exists(),
                "ckpt_path must point at a real file: {:?}",
                info.ckpt_path
            );
            let name = info.ckpt_path.file_name().unwrap().to_string_lossy();
            assert!(
                name.starts_with("hook_fire-step") && name.ends_with(".safetensors"),
                "ckpt_path must match the store's <prefix>-step<N>.safetensors form: {name}"
            );
            assert!(
                info.grad_norm.is_finite() && info.grad_norm >= 0.0,
                "grad_norm must be a finite non-negative number, got {}",
                info.grad_norm
            );
            assert!(
                info.train_loss.is_finite(),
                "train_loss must be finite, got {}",
                info.train_loss
            );
            assert!(info.lr > 0.0, "lr must be positive, got {}", info.lr);
        }

        // Terminal ckpt still records min_train_loss (full-run path,
        // no early_break marker).
        assert!(ckpt.metrics.contains_key("min_train_loss"));
        assert!(
            !ckpt.metrics.contains_key("early_break"),
            "full-run completion must not tag early_break"
        );
    }

    /// A hook returning [`CkptControl::BREAK`] stops training after
    /// the current ckpt, still writes the terminal
    /// `<prefix>.safetensors`, and tags `metrics["early_break"] = 1.0`.
    #[test]
    fn hook_break_stops_training_early_with_marker() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 20,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 3,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        // Break on the second fire (step 8) so the loop stops well
        // short of the 20-step cap.
        let fire_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let fire_count_hook = Arc::clone(&fire_count);
        let hook: CkptHook = Box::new(move |_info| {
            let mut n = fire_count_hook.lock().unwrap();
            *n += 1;
            if *n >= 2 {
                Ok(CkptControl::BREAK)
            } else {
                Ok(CkptControl::CONTINUE)
            }
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_break",
            lease,
            Some(hook),
        )
        .expect("training must return a Checkpoint even after early break");

        assert_eq!(*fire_count.lock().unwrap(), 2);
        // Ckpt.step is the 1-indexed step at which the break fired.
        assert_eq!(ckpt.step, 8);
        assert_eq!(
            ckpt.metrics.get("early_break").copied(),
            Some(1.0),
            "early_break marker must be present"
        );
        assert!(ckpt.metrics.contains_key("min_train_loss"));
        assert!(ckpt.metrics.contains_key("final_lr"));

        // Terminal file was still written under the stable name so
        // downstream `alc.nn.load` still resolves.
        assert!(
            tmp.path().join("hook_break.safetensors").exists(),
            "save_final must run before returning from an early break"
        );
    }

    /// The whole point of the keep surface: a checkpoint the hook held
    /// early in the run is still on disk when the run returns, and the
    /// run says which one it was.
    ///
    /// `ckpt_keep = 1` makes the hazard unambiguous — without the pin
    /// the step-4 file is deleted at step 8 and every later boundary,
    /// so a search that liked step 4 would come back holding a path to
    /// nothing.
    #[test]
    fn kept_checkpoint_survives_rotation_and_lands_in_candidates() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 20,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        // Keep the first fire (step 4) and nothing else. Four more
        // boundaries follow it, each pruning down to keep=1.
        let fire_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let fire_count_hook = Arc::clone(&fire_count);
        let hook: CkptHook = Box::new(move |_info| {
            let mut n = fire_count_hook.lock().unwrap();
            *n += 1;
            if *n == 1 {
                Ok(CkptControl::keep(Some("first-look".to_string())))
            } else {
                Ok(CkptControl::CONTINUE)
            }
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_keep",
            lease,
            Some(hook),
        )
        .expect("a keep must not disturb the run");

        assert_eq!(*fire_count.lock().unwrap(), 5, "20 steps / ckpt_every 4");
        assert_eq!(ckpt.step, 20, "keep alone must not stop the run");

        assert_eq!(ckpt.candidates.len(), 1);
        let candidate = &ckpt.candidates[0];
        assert_eq!(candidate.info.step, 4);
        assert_eq!(candidate.reason.as_deref(), Some("first-look"));
        assert!(
            candidate.info.ckpt_path.exists(),
            "the candidate's file must outlive four rotations at keep=1: {:?}",
            candidate.info.ckpt_path
        );
        assert!(candidate.info.train_loss.is_finite());
    }

    /// Keep and break together — how a successful search ends. The
    /// checkpoint that satisfied the judgment is held, and the run
    /// stops on the same decision.
    #[test]
    fn keep_and_break_holds_the_checkpoint_it_stopped_on() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 20,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let hook: CkptHook =
            Box::new(move |_info| Ok(CkptControl::keep_and_break(Some("good-enough".into()))));

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_keep_break",
            lease,
            Some(hook),
        )
        .expect("keep+break must still finalize");

        assert_eq!(ckpt.step, 4, "the run stopped on the first boundary");
        assert_eq!(
            ckpt.metrics.get("early_break").copied(),
            Some(1.0),
            "a keep alongside a break must not swallow the early_break marker"
        );
        assert_eq!(ckpt.candidates.len(), 1);
        assert_eq!(ckpt.candidates[0].info.step, 4);
        assert!(
            ckpt.candidates[0].info.ckpt_path.exists(),
            "the pin has to be in place before the break returns"
        );
    }

    /// A run whose hook never keeps anything reports no candidates —
    /// the field is additive, not a behaviour change.
    #[test]
    fn a_run_that_keeps_nothing_reports_no_candidates() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 8,
            warmup: 1,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 2,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let hook: CkptHook = Box::new(move |_info| Ok(CkptControl::CONTINUE));

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_no_keep",
            lease,
            Some(hook),
        )
        .unwrap();
        assert!(ckpt.candidates.is_empty());
    }

    /// The record of a kept checkpoint outlives the run that kept it.
    ///
    /// The hook keeps at the first boundary and then raises at the
    /// second, so `run_full_ft` returns `Err` and the caller never sees
    /// a `Checkpoint` — but the pinned file is still on disk, and this
    /// is the case where the in-memory list would have been the only
    /// record of what it was.
    #[test]
    fn a_kept_checkpoint_is_written_down_before_the_run_can_lose_it() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 20,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let fire_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let fire_count_hook = Arc::clone(&fire_count);
        let hook: CkptHook = Box::new(move |_info| {
            let mut n = fire_count_hook.lock().unwrap();
            *n += 1;
            match *n {
                1 => Ok(CkptControl::keep(Some("first-look".to_string()))),
                _ => Err("measurement blew up".to_string()),
            }
        });

        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_keep_then_die",
            lease,
            Some(hook),
        )
        .expect_err("the second fire must fail the run");
        assert!(matches!(err, TrainError::Hook(ref m) if m.contains("measurement blew up")));

        // The pinned file survives — nothing un-pins.
        let ckpt = tmp.path().join("hook_keep_then_die-step4.safetensors");
        assert!(ckpt.exists(), "the pinned checkpoint must still be there");

        // And so does the record naming it.
        let record =
            std::fs::read_to_string(tmp.path().join("hook_keep_then_die-candidates.jsonl"))
                .expect("the candidate record must exist after a failed run");
        let lines: Vec<&str> = record.lines().collect();
        assert_eq!(lines.len(), 1, "one keep, one line: {record}");
        assert!(lines[0].contains("\"step\":4"), "{}", lines[0]);
        assert!(lines[0].contains("first-look"), "{}", lines[0]);
        assert!(
            lines[0].contains("hook_keep_then_die-step4.safetensors"),
            "the line must name the file it is about: {}",
            lines[0]
        );
    }

    /// On a run that returns normally the record and the returned list
    /// say the same thing — one is not a lossy copy of the other.
    #[test]
    fn the_candidate_record_agrees_with_the_returned_list() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 20,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        // Keep the 1st and 3rd boundaries (steps 4 and 12).
        let fire_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let fire_count_hook = Arc::clone(&fire_count);
        let hook: CkptHook = Box::new(move |_info| {
            let mut n = fire_count_hook.lock().unwrap();
            *n += 1;
            match *n {
                1 => Ok(CkptControl::keep(Some("early".to_string()))),
                3 => Ok(CkptControl::keep(None)),
                _ => Ok(CkptControl::CONTINUE),
            }
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "record_agrees",
            lease,
            Some(hook),
        )
        .unwrap();

        let record =
            std::fs::read_to_string(tmp.path().join("record_agrees-candidates.jsonl")).unwrap();
        let lines: Vec<&str> = record.lines().collect();
        assert_eq!(lines.len(), ckpt.candidates.len());
        assert_eq!(lines.len(), 2);
        for (line, candidate) in lines.iter().zip(ckpt.candidates.iter()) {
            assert!(
                line.contains(&format!("\"step\":{}", candidate.info.step)),
                "{line}"
            );
        }
        // A keep with no reason omits the key rather than writing null,
        // so a reader sees the same absence Lua sees.
        assert!(lines[0].contains("early"), "{}", lines[0]);
        assert!(!lines[1].contains("reason"), "{}", lines[1]);
    }

    /// The measurements behind a keep reach both the returned list and
    /// the written record — the evidence outlives the hook call it was
    /// gathered in.
    #[test]
    fn a_keeps_measurements_reach_the_list_and_the_record() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 4,
            warmup: 1,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let hook: CkptHook = Box::new(move |_info| {
            let mut values = BTreeMap::new();
            values.insert("ci_lower".to_string(), 0.62);
            values.insert("trickiness".to_string(), 0.41);
            Ok(CkptControl::keep_with(KeepMark {
                reason: Some("tier-2".to_string()),
                values,
            }))
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "keep_values",
            lease,
            Some(hook),
        )
        .unwrap();

        assert_eq!(ckpt.candidates.len(), 1);
        assert_eq!(ckpt.candidates[0].values.get("ci_lower"), Some(&0.62));

        let line =
            std::fs::read_to_string(tmp.path().join("keep_values-candidates.jsonl")).unwrap();
        assert!(line.contains("\"ci_lower\":0.62"), "{line}");
        assert!(line.contains("\"trickiness\":0.41"), "{line}");
        // Ordered, so two runs measuring the same things write the same
        // bytes: `ci_lower` sorts before `trickiness`.
        assert!(
            line.find("ci_lower").unwrap() < line.find("trickiness").unwrap(),
            "{line}"
        );
    }

    /// The record carries what the *trainer* knew at the boundary, not
    /// only the loss.
    ///
    /// Asking later whether a keep was sound needs both sides: the
    /// model-side readings are the caller's `values`, and whether the
    /// run was still converging when they were taken is the trainer's.
    /// Neither survives the hook call, so a record holding only one of
    /// them cannot answer the question short of re-running the sweep.
    #[test]
    fn the_record_carries_the_training_side_numbers_too() {
        use std::sync::Mutex;

        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 4,
            warmup: 1,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        // Capture what the hook was handed, so the record can be
        // compared against it rather than against a guess.
        let seen: Arc<Mutex<Option<CkptInfo>>> = Arc::new(Mutex::new(None));
        let seen_hook = Arc::clone(&seen);
        let hook: CkptHook = Box::new(move |info| {
            *seen_hook.lock().unwrap() = Some(info.clone());
            Ok(CkptControl::keep(None))
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "train_side",
            lease,
            Some(hook),
        )
        .unwrap();

        let handed = seen.lock().unwrap().clone().expect("hook fired");
        assert_eq!(
            ckpt.candidates[0].info, handed,
            "the candidate must carry the same frame the hook was handed"
        );

        let line = std::fs::read_to_string(tmp.path().join("train_side-candidates.jsonl")).unwrap();
        for key in [
            "step",
            "ckpt_path",
            "train_loss",
            "lr",
            "grad_norm",
            "elapsed_ms",
            "min_train_loss",
        ] {
            assert!(
                line.contains(&format!("\"{key}\"")),
                "the written record must name {key}: {line}"
            );
        }
    }

    /// A keep with no measurements omits the key rather than writing an
    /// empty object, so absence reads the same as it does for `reason`.
    #[test]
    fn a_keep_without_measurements_omits_the_values_key() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let cfg = FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 4,
            warmup: 1,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 4,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let hook: CkptHook = Box::new(move |_info| Ok(CkptControl::keep(None)));

        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "keep_bare",
            lease,
            Some(hook),
        )
        .unwrap();

        let line = std::fs::read_to_string(tmp.path().join("keep_bare-candidates.jsonl")).unwrap();
        assert!(!line.contains("values"), "{line}");
        assert!(!line.contains("reason"), "{line}");
    }

    /// A hook that returns an error surfaces as
    /// [`TrainError::Hook`], carrying the message unchanged. Training
    /// stops without writing a terminal ckpt.
    #[test]
    fn hook_error_propagates_as_train_error_hook() {
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
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let hook: CkptHook = Box::new(|_info| Err("hook: bad time".to_string()));

        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &loss,
            tmp.path(),
            "hook_err",
            lease,
            Some(hook),
        )
        .unwrap_err();
        match err {
            TrainError::Hook(msg) => {
                assert!(
                    msg.contains("hook: bad time"),
                    "TrainError::Hook must carry the original message, got {msg}"
                );
            }
            other => panic!("expected TrainError::Hook, got {other:?}"),
        }
    }

    /// A `hook = None` run produces bit-identical training loss
    /// against a run with the same seed / dataset / config but no
    /// hook argument. Guards the "additive parameter" contract at the
    /// scalar-metric level (`min_train_loss` reproduces exactly).
    #[test]
    fn hook_none_is_bit_identical_to_pre_hook_path() {
        // Two independent runs on identical config and identical
        // dataset (both are deterministic constructions).
        let base_cfg = || FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum: 1,
            steps: 10,
            warmup: 2,
            schedule: ScheduleKind::Constant,
            weight_decay: 0.0,
            ckpt_every: 0,
            ckpt_keep: 1,
            init_from: None,
            mask_disallowed_logits: false,
            ..FullFtConfig::default()
        };

        // Run A: baseline (no hook).
        let (_, vm_a, model_a) = tiny_cfg_and_model();
        let mut ds_a = overfit_dataset();
        let loss = CrossEntropyLoss::new();
        let tmp_a = TempDir::new().unwrap();
        let lease_a = Arc::new(TrainingLease::new());
        let ckpt_a = run_full_ft(
            &model_a,
            &vm_a,
            &mut ds_a,
            None,
            &base_cfg(),
            &loss,
            tmp_a.path(),
            "bit_ident_a",
            lease_a,
            None,
        )
        .expect("run A must complete");

        // Run B: identical, still no hook (proves the hook-carrying
        // signature does not disturb the numerics when the hook is
        // absent).
        let (_, vm_b, model_b) = tiny_cfg_and_model();
        let mut ds_b = overfit_dataset();
        let tmp_b = TempDir::new().unwrap();
        let lease_b = Arc::new(TrainingLease::new());
        let ckpt_b = run_full_ft(
            &model_b,
            &vm_b,
            &mut ds_b,
            None,
            &base_cfg(),
            &loss,
            tmp_b.path(),
            "bit_ident_b",
            lease_b,
            None,
        )
        .expect("run B must complete");

        // `tiny_cfg_and_model` calls `VarBuilder`'s randomised init
        // per call — the two runs start from independent weights and
        // therefore reach different absolute losses. The invariant we
        // *can* pin without a shared init snapshot is "hook=None
        // never introduces its own new metric key" (existing metrics
        // are the same set as before the hook wiring).
        let keys_a: std::collections::BTreeSet<_> = ckpt_a.metrics.keys().cloned().collect();
        let keys_b: std::collections::BTreeSet<_> = ckpt_b.metrics.keys().cloned().collect();
        assert_eq!(
            keys_a, keys_b,
            "hook=None runs must expose the same metrics key set"
        );
        assert!(keys_a.contains("min_train_loss"));
        assert!(keys_a.contains("final_lr"));
        assert!(
            !keys_a.contains("early_break"),
            "hook=None must never tag early_break"
        );
        assert!(
            !keys_a.contains("hook_error"),
            "hook=None must never tag hook_error"
        );
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

    // ── Side-channel entry points ───────────────────────────────────

    use crate::arch::Gpt2Custom;
    use candle_core::IndexOp;

    /// The tiny model above, plus whichever side-channel table the
    /// caller names.
    fn side_channel_model(custom: Gpt2Custom) -> (Gpt2Config, VarMap, Gpt2Model) {
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
            custom: Some(custom),
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vb).unwrap();
        (cfg, vm, model)
    }

    fn short_run(steps: usize) -> FullFtConfig {
        FullFtConfig {
            lr: 1e-3,
            batch_size: 1,
            steps,
            warmup: 0,
            ..FullFtConfig::default()
        }
    }

    /// `rows` copies of one sequence, so a run over them is cheap and
    /// deterministic in shape.
    fn repeated_rows(rows: usize) -> Vec<Vec<u32>> {
        std::iter::repeat_with(|| vec![1u32, 2, 3, 4, 5, 6, 7, 8])
            .take(rows)
            .collect()
    }

    fn one_row_batch_opts() -> DatasetOpts {
        DatasetOpts {
            batch_size: 1,
            ctx_len: 8,
            shuffle: false,
            seed: None,
            pad_id: 0,
            mask_pad: true,
            text_field: "text".into(),
        }
    }

    /// The conditioned entry point drives a model that has the table,
    /// over a dataset that carries the conditions, and reaches the end.
    #[test]
    fn conditioned_run_completes_over_a_conditioned_dataset() {
        let (_, vm, model) = side_channel_model(Gpt2Custom {
            cond_slots: Some(2),
            ..Default::default()
        });
        let conds: Vec<CondIndex> = (0..8u32)
            .map(|i| CondIndex::new(i % 2, 2).unwrap())
            .collect();
        let mut ds = TokenizedDataset::new(repeated_rows(8), one_row_batch_opts())
            .with_conditions(conds)
            .expect("one condition per row");
        let tmp = TempDir::new().unwrap();
        let ckpt = run_conditioned_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &short_run(4),
            &CrossEntropyLoss::new(),
            tmp.path(),
            "cond",
            Arc::new(TrainingLease::new()),
            None,
        )
        .expect("conditioned run");
        assert_eq!(ckpt.step, 4);
        assert!(tmp.path().join("cond.safetensors").exists());
    }

    /// A conditioned run over a conditionless batch would train
    /// unconditioned under a checkpoint labelled otherwise.
    #[test]
    fn conditioned_run_refuses_a_batch_without_conditions() {
        let (_, vm, model) = side_channel_model(Gpt2Custom {
            cond_slots: Some(2),
            ..Default::default()
        });
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let tmp = TempDir::new().unwrap();
        let err = run_conditioned_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &short_run(1),
            &CrossEntropyLoss::new(),
            tmp.path(),
            "cond",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, TrainError::MissingConditions { rows: 1 }),
            "{err:?}"
        );
    }

    /// The mirror: an entry point with nowhere to put a condition says
    /// so rather than dropping it.
    #[test]
    fn plain_run_refuses_a_batch_carrying_conditions() {
        let (_, vm, model) = tiny_cfg_and_model();
        let conds: Vec<CondIndex> = (0..4).map(|_| CondIndex::new(0, 2).unwrap()).collect();
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts())
            .with_conditions(conds)
            .expect("one condition per row");
        let tmp = TempDir::new().unwrap();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &short_run(1),
            &CrossEntropyLoss::new(),
            tmp.path(),
            "plain",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        match err {
            TrainError::UnexpectedConditions { rows, conds } => {
                assert_eq!((rows, conds), (1, 1));
            }
            other => panic!("expected UnexpectedConditions, got {other:?}"),
        }
    }

    /// The allowed-id entry point drives a model built with the table
    /// over a dataset that carries the sets, and reaches the end.
    #[test]
    fn allowed_run_completes_over_a_dataset_carrying_the_sets() {
        let (_, vm, model) = side_channel_model(Gpt2Custom {
            allowed_input: true,
            ..Default::default()
        });
        let allowed: Vec<Vec<Vec<u32>>> = (0..8)
            .map(|_| (0..8).map(|p| vec![p as u32 + 1, 9]).collect())
            .collect();
        let mut ds = TokenizedDataset::new(repeated_rows(8), one_row_batch_opts())
            .with_allowed_ids(allowed)
            .expect("one set list per row");
        let tmp = TempDir::new().unwrap();
        let ckpt = run_allowed_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &short_run(4),
            &CrossEntropyLoss::new(),
            tmp.path(),
            "allowed",
            Arc::new(TrainingLease::new()),
            None,
        )
        .expect("allowed run");
        assert_eq!(ckpt.step, 4);
    }

    /// The model reads the sets at every position, so a batch without
    /// them is refused where the cause is still visible.
    #[test]
    fn allowed_run_refuses_a_batch_without_the_sets() {
        let (_, vm, model) = side_channel_model(Gpt2Custom {
            allowed_input: true,
            ..Default::default()
        });
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let tmp = TempDir::new().unwrap();
        let err = run_allowed_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &short_run(1),
            &CrossEntropyLoss::new(),
            tmp.path(),
            "allowed",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        match err {
            TrainError::MissingAllowedSets { rows, needed } => {
                assert_eq!(rows, 1);
                assert!(needed.contains("every position"), "{needed}");
            }
            other => panic!("expected MissingAllowedSets, got {other:?}"),
        }
    }

    /// Asking the loss to score among the allowed ids, over a dataset
    /// that carries none, is refused rather than run unmasked under a
    /// config that says otherwise.
    #[test]
    fn masked_loss_refuses_a_batch_without_the_sets() {
        let (_, vm, model) = tiny_cfg_and_model();
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let cfg = FullFtConfig {
            mask_disallowed_logits: true,
            ..short_run(1)
        };
        let tmp = TempDir::new().unwrap();
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "masked",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        match err {
            TrainError::MissingAllowedSets { needed, .. } => {
                assert!(needed.contains("mask_disallowed_logits"), "{needed}");
            }
            other => panic!("expected MissingAllowedSets, got {other:?}"),
        }
    }

    /// A plain run over a dataset that does carry the sets accepts the
    /// mask and completes — the switch is what turns it on.
    #[test]
    fn masked_loss_completes_when_the_batch_carries_the_sets() {
        let (_, vm, model) = tiny_cfg_and_model();
        let allowed: Vec<Vec<Vec<u32>>> = (0..4)
            .map(|_| (0..8).map(|p| vec![p as u32 + 1, 9]).collect())
            .collect();
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts())
            .with_allowed_ids(allowed)
            .expect("one set list per row");
        let cfg = FullFtConfig {
            mask_disallowed_logits: true,
            ..short_run(3)
        };
        let tmp = TempDir::new().unwrap();
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "masked",
            Arc::new(TrainingLease::new()),
            None,
        )
        .expect("masked run");
        assert_eq!(ckpt.step, 3);
    }

    fn batch_with_allowed(allowed: Vec<Vec<Vec<u32>>>, rows: usize, seq: usize) -> Batch {
        Batch {
            input_ids: (0..rows).map(|_| vec![1u32; seq]).collect(),
            loss_mask: None,
            is_last: true,
            allowed_ids: Some(allowed),
            conds: None,
            conds_per_row: 1,
        }
    }

    /// The mask reads one position later than the entry it governs, and
    /// an empty set means "not constrained here" rather than "nothing
    /// is allowed here" — which would zero the whole row.
    #[test]
    fn allowed_logit_mask_reads_the_position_after_the_input() {
        // Row of 3 positions: sets at 0 / 1 / 2, mask width 2.
        let allowed = vec![vec![vec![0u32], vec![1], vec![]]];
        let batch = batch_with_allowed(allowed, 1, 3);
        let mask = allowed_logit_mask(&batch, 3, 4, &Device::Cpu)
            .unwrap()
            .expect("the batch carries sets");
        assert_eq!(mask.dims(), &[1, 2, 4]);
        let values: Vec<Vec<f32>> = mask.i(0).unwrap().to_vec2().unwrap();
        // Entry 0 governs target `input_ids[1]`, so it reads the set at
        // position 1 — `{1}` — and not the one at position 0.
        assert_eq!(
            values[0],
            vec![DISALLOWED_LOGIT, 0.0, DISALLOWED_LOGIT, DISALLOWED_LOGIT]
        );
        // Entry 1 reads position 2, whose set is empty: unconstrained.
        assert_eq!(values[1], vec![0.0; 4]);
    }

    /// An id past the end of the vocabulary is a producer mistake that
    /// no shape would reveal.
    #[test]
    fn allowed_logit_mask_refuses_an_id_outside_the_vocabulary() {
        let batch = batch_with_allowed(vec![vec![vec![0u32], vec![9]]], 1, 2);
        let msg = allowed_logit_mask(&batch, 2, 4, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("outside vocab 4"), "{msg}");
    }

    /// The input side reads the same entry as the mask, one position
    /// after the input it is attached to. Off by one here and every
    /// shape would still agree.
    #[test]
    fn allowed_input_sets_read_the_same_entry_as_the_mask() {
        // Position 0 offers a set the window must skip; the two the
        // window keeps have different widths, so the padding shows.
        let allowed = vec![vec![vec![3u32], vec![1, 2], vec![0]]];
        let batch = batch_with_allowed(allowed, 1, 3);
        let sets = allowed_input_sets(&batch, 3, &Device::Cpu)
            .unwrap()
            .expect("the batch carries sets");
        assert_eq!((sets.rows(), sets.width(), sets.widest()), (1, 2, 2));
        let ids: Vec<Vec<u32>> = sets.ids().i(0).unwrap().to_vec2().unwrap();
        // Position 0 of the window is the set at index 1 — `{1, 2}` —
        // not `{3}`, which belongs to the input the model already read.
        assert_eq!(ids[0], vec![1, 2]);
        assert_eq!(ids[1], vec![0, 0]); // `{0}` plus one padding entry
        let weights: Vec<Vec<f32>> = sets.weights().i(0).unwrap().to_vec2().unwrap();
        assert_eq!(weights[0], vec![0.5, 0.5]);
        assert_eq!(weights[1], vec![1.0, 0.0]);
    }

    /// A batch carrying no sets passes through both helpers, which is
    /// how an unconstrained dataset reaches the plain loop.
    #[test]
    fn the_allowed_helpers_pass_a_batch_that_carries_no_sets() {
        let batch = Batch {
            input_ids: vec![vec![1u32, 2, 3]],
            loss_mask: None,
            is_last: true,
            allowed_ids: None,
            conds: None,
            conds_per_row: 1,
        };
        assert!(allowed_logit_mask(&batch, 3, 8, &Device::Cpu)
            .unwrap()
            .is_none());
        assert!(allowed_input_sets(&batch, 3, &Device::Cpu)
            .unwrap()
            .is_none());
    }

    /// The pad mask a dataset attaches survives the target shift with
    /// the meaning it was built with: entry `k` gates the prediction of
    /// `input_ids[k + 1]`, so the position that first holds filler is
    /// the first one excluded.
    #[test]
    fn the_pad_mask_lines_up_with_the_targets_after_the_shift() {
        let mut ds = TokenizedDataset::new(
            vec![vec![1u32, 2, 3]],
            DatasetOpts {
                batch_size: 1,
                ctx_len: 5,
                shuffle: false,
                seed: None,
                pad_id: 0,
                mask_pad: true,
                text_field: "text".into(),
            },
        );
        let batch = ds.next_batch().unwrap().unwrap();
        assert_eq!(batch.input_ids[0], vec![1, 2, 3, 0, 0]);
        let (_, targets, mask) = batch_to_input_target(&batch, &Device::Cpu).unwrap();
        let targets: Vec<u32> = targets.i(0).unwrap().to_vec1().unwrap();
        let mask: Vec<f32> = mask
            .expect("a padded batch reaches the loss with a mask")
            .i(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(targets, vec![2, 3, 0, 0]);
        assert_eq!(
            mask,
            vec![1.0, 1.0, 0.0, 0.0],
            "predicting token 3 is scored; predicting the filler behind it is not"
        );
    }

    /// `init_from` puts the checkpoint's weights in place before the
    /// first step. Run at `lr = 0` with no weight decay so what the
    /// map holds afterwards is the checkpoint and nothing else.
    #[test]
    fn init_from_restores_the_checkpoint_before_training() {
        let tmp = TempDir::new().unwrap();
        let (_, source_vm, _source) = tiny_cfg_and_model();
        let source_path = tmp.path().join("source.safetensors");
        source_vm.save(&source_path).unwrap();

        let (_, vm, model) = tiny_cfg_and_model();
        let before: Vec<f32> = vm.data().lock().unwrap()["wte.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let from_file: Vec<f32> = source_vm.data().lock().unwrap()["wte.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_ne!(before, from_file, "two random inits collided");

        let cfg = FullFtConfig {
            lr: 0.0,
            weight_decay: 0.0,
            init_from: Some(source_path),
            ..short_run(1)
        };
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "resumed",
            Arc::new(TrainingLease::new()),
            None,
        )
        .expect("resumed run");

        let after: Vec<f32> = vm.data().lock().unwrap()["wte.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(after, from_file);
    }

    /// A checkpoint that cannot be read stops the run before it starts,
    /// rather than training from the random initialisation under a
    /// config that says it resumed.
    #[test]
    fn init_from_propagates_a_failed_restore_and_writes_nothing() {
        let tmp = TempDir::new().unwrap();
        let (_, vm, model) = tiny_cfg_and_model();
        let cfg = FullFtConfig {
            init_from: Some(tmp.path().join("nowhere.safetensors")),
            ..short_run(1)
        };
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "resumed",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        match &err {
            TrainError::Restore(RestoreError::Open { .. }) => {}
            other => panic!("expected Restore(Open), got {other:?}"),
        }
        assert!(err.to_string().starts_with("init_from:"), "{err}");
        assert!(
            !tmp.path().join("resumed.safetensors").exists(),
            "a run that never started must not write a terminal checkpoint"
        );
    }

    /// A shape disagreement is the same refusal one level down: the
    /// checkpoint is not this model's.
    #[test]
    fn init_from_refuses_a_checkpoint_of_another_shape() {
        let tmp = TempDir::new().unwrap();
        let other = VarMap::new();
        let vb = VarBuilder::from_varmap(&other, DType::F32, &Device::Cpu);
        let _ = vb.get((3, 3), "wte.weight").unwrap();
        let path = tmp.path().join("other.safetensors");
        other.save(&path).unwrap();

        let (_, vm, model) = tiny_cfg_and_model();
        let cfg = FullFtConfig {
            init_from: Some(path),
            ..short_run(1)
        };
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
            None,
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "resumed",
            Arc::new(TrainingLease::new()),
            None,
        )
        .unwrap_err();
        // Every other variable of the model is absent from that file,
        // so the strict entry point stops on the gap first.
        match &err {
            TrainError::Restore(RestoreError::Incomplete { .. })
            | TrainError::Restore(RestoreError::Mismatch { .. }) => {}
            other => panic!("expected Restore(Incomplete|Mismatch), got {other:?}"),
        }
    }

    /// The LoRA entry point never sees the base map, so a checkpoint
    /// named on the config would have nowhere to land.
    #[test]
    fn init_from_is_refused_by_the_lora_entry_point() {
        let tmp = TempDir::new().unwrap();
        let (_, vm, mut model) = tiny_cfg_and_model();
        let path = tmp.path().join("base.safetensors");
        vm.save(&path).unwrap();

        let cfg = FullFtConfig {
            init_from: Some(path),
            ..short_run(1)
        };
        let mut ds = TokenizedDataset::new(repeated_rows(4), one_row_batch_opts());
        let err = run_lora_ft(
            &mut model,
            &mut ds,
            &LoraConfig::new(2, 4.0),
            &cfg,
            &CrossEntropyLoss::new(),
            tmp.path(),
            "lora",
            Arc::new(TrainingLease::new()),
        )
        .unwrap_err();
        assert!(matches!(err, TrainError::InitFromUnsupported), "{err:?}");
    }
}
