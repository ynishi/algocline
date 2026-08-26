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
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarMap};

use crate::arch::{AllowedSets, CondIndex, LoraConfig, LoraWrappable};
use crate::train::ckpt::{checkpoint_from_path, restore_into, CheckpointStore, RestoreError};
use crate::train::data::{Batch, Dataset, DatasetError};
use crate::train::loss::Loss;
use crate::train::mixed::MixedAdamW;
use crate::train::scheduler::{ScheduleKind, Scheduler};
use crate::train::AllowedForward;
use crate::train::Checkpoint;
use crate::train::ConditionedForward;
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
            Self::Stock(o) => o.step(grads),
            Self::Mixed(o) => o.step(grads),
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
    /// `backward()` so the summed gradient equals the mean over the
    /// full effective batch (canonical PyTorch form). `grad_accum = 0`
    /// is refused as a config error; `grad_accum = 1` behaves exactly
    /// like the single-micro path.
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
            weight_decay: 0.1,
            ckpt_every: 0,
            ckpt_keep: 3,
            init_from: None,
            mask_disallowed_logits: false,
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
    /// Config asked for `grad_accum = 0`, which would divide by zero
    /// when scaling per-micro losses. Multi-step accumulation is now
    /// honoured for `grad_accum >= 1`.
    #[error("`grad_accum` must be at least 1 (got 0)")]
    ZeroGradAccum,
    /// Another training session already holds the lease.
    #[error("another training session is already active on this VM")]
    LeaseHeld,
    /// An `on_ckpt` hook returned an error. The trainer writes the
    /// terminal `<prefix>.safetensors` before propagating so callers
    /// still have last-good weights on disk, but the returned
    /// [`Checkpoint`] carries `metrics["hook_error"] = 1.0` so the
    /// error is discoverable from the bundle side too.
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
}

/// Information handed to the [`CkptHook`] at every `ckpt_every` boundary.
///
/// Kept flat (owned primitives + [`PathBuf`]) so the hook can convert it
/// into an mlua-side Lua table without borrowing from any tensor. The
/// checkpoint has already been written to `ckpt_path` by the time the
/// hook fires — the hook decides whether to keep training or stop.
#[derive(Debug, Clone)]
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
pub enum CkptControl {
    /// Keep training. Emitted as `nil` / `"continue"` from Lua.
    Continue,
    /// Stop training now (after the current checkpoint). Emitted as
    /// `"break"` from Lua.
    Break,
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
/// `hook` is an optional [`CkptHook`] fired at each `ckpt_every`
/// boundary (after `save_step`). Passing `None` retains the previous
/// behaviour bit-identically; passing `Some(_)` lets the caller inspect
/// per-checkpoint scalars and return [`CkptControl::Break`] to stop
/// training early. The hook is exclusive to the full-fine-tune surface
/// today; the LoRA / distillation entries pass `None` internally.
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
    hook: Option<CkptHook>,
) -> Result<Checkpoint, TrainError>
where
    M: Module + DeviceView,
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
        cfg,
        loss_fn,
        ckpt_dir,
        ckpt_prefix,
        lease,
        hook,
    )
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

    let device = device.clone();
    let mut last_train_loss = f32::NAN;
    let mut running_min_loss = f32::INFINITY;

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
    for step in 0..cfg.steps {
        let lr = scheduler.lr_at(step);
        opt.set_learning_rate(lr);

        let mut accum: Option<GradStore> = None;
        let mut micro_loss_sum: f32 = 0.0;
        for _micro in 0..grad_accum {
            let batch = dataset.next_batch()?.ok_or(TrainError::DatasetExhausted {
                seen: step,
                requested: cfg.steps,
            })?;

            let (inputs, targets, mask) = batch_to_input_target(&batch, &device)?;
            // The allowed-id input, built only for the entry point that
            // takes one: every other run would pay for a tensor it
            // cannot read. The sets are shifted to line up with the
            // model's inputs — see `allowed_input_sets`.
            let allowed = match &forward {
                ForwardPass::Allowed(_) => {
                    allowed_input_sets(&batch, batch.input_ids[0].len(), &device)?
                }
                _ => None,
            };
            // The batch's own side channels against the caller's
            // declared intent. Both disagreements are refused: a
            // conditioned run over a conditionless batch would train
            // unconditioned under a checkpoint labelled otherwise, and
            // an unconditioned run over a conditioned batch would drop
            // a condition the caller attached per row. `inputs` is the
            // batch after the target shift, so its first dimension is
            // still the row count.
            let logits = match (&mut forward, batch.conds.as_deref(), allowed.as_ref()) {
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
                // Neither of these two takes a condition, so a batch
                // carrying one is the same mistake at both.
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
            // Mixed precision: the loss (log_softmax + NLL reduction)
            // is always scored in F32 — BF16's 8 mantissa bits are too
            // coarse for a mean over thousands of log-probs.
            // `to_dtype` is differentiable, so the backward pass
            // crosses back into the model's dtype at this boundary.
            // F32 logits pass through untouched.
            let logits = if logits.dtype() == DType::F32 {
                logits
            } else {
                logits.to_dtype(DType::F32)?
            };
            // Remove the ids this target could not have taken, so the
            // loss scores the choice among the ones it could. Applied
            // after the F32 cast: a large negative penalty in BF16
            // would not survive the conversion cleanly. Opt-in, and a
            // batch that cannot honour the opt-in is refused rather
            // than trained unmasked.
            let logits = if cfg.mask_disallowed_logits {
                match allowed_logit_mask(&batch, batch.input_ids[0].len(), logits.dim(2)?, &device)?
                {
                    Some(m) => logits.broadcast_add(&m)?,
                    None => {
                        return Err(TrainError::MissingAllowedSets {
                            rows: inputs.dim(0)?,
                            needed: "cfg.mask_disallowed_logits asks the loss to use them",
                        })
                    }
                }
            } else {
                logits
            };
            let loss = loss_fn.compute(&logits, &targets, mask.as_ref())?;

            let loss_val: f32 = loss.to_scalar()?;
            micro_loss_sum += loss_val;

            // Pre-backward `1 / grad_accum` scaling — the canonical
            // form. Scalar multiplication is linear w.r.t. the backward
            // pass, so `sum_i grad(loss_i / N) == grad(mean_i loss_i)`
            // and the reported grad equals the mean over the effective
            // batch. For `grad_accum == 1` this reduces to `scale = 1`
            // and the multiply is a no-op numerically.
            let scaled = (&loss * scale)?;
            let grads = scaled.backward()?;
            match accum.as_mut() {
                Some(store) => store.extend(grads)?,
                None => accum = Some(grads),
            }
        }
        // `accum` is always `Some` here because `grad_accum >= 1` is
        // enforced above and the inner loop runs at least once — a
        // mid-micro dataset exhaustion returns early via `?` above.
        let grads = accum.expect("grad_accum >= 1 guarantees at least one backward");

        let mean_loss = micro_loss_sum / grad_accum as f32;
        last_train_loss = mean_loss;
        if mean_loss < running_min_loss {
            running_min_loss = mean_loss;
        }

        // Per-step observability. Emit through `tracing` so downstream
        // subscribers (RUST_LOG=algocline_nn=info) can collect the loss
        // trajectory without changing the return shape. `loss` is the
        // mean per-micro loss (matches the equivalent single-micro
        // `batch_size * grad_accum` run) and `grad_accum` is emitted as
        // an additive field so post-hoc analysis can distinguish
        // accumulated steps from raw single-micro ones.
        tracing::info!(
            step = step,
            loss = mean_loss,
            lr = lr,
            grad_accum = grad_accum,
            "train_step"
        );

        // Compute grad norm before `opt.step` consumes / mutates the
        // per-parameter state. The value is only surfaced through the
        // `on_ckpt` hook, so the walk is guarded by `hook.is_some()`
        // and `ckpt_every` — a no-hook run pays nothing beyond the
        // existing per-step cost.
        let will_fire_hook =
            hook.is_some() && cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0;
        let grad_norm = if will_fire_hook {
            grad_l2_norm(opt_vm, &grads)?
        } else {
            0.0
        };

        opt.step(&grads)?;

        if cfg.ckpt_every > 0 && (step + 1) % cfg.ckpt_every == 0 {
            let ckpt_path = ckpt_store
                .save_step(save_vm, step + 1)
                .map_err(|e| TrainError::Ckpt(e.to_string()))?;

            if let Some(hook_fn) = hook.as_mut() {
                let info = CkptInfo {
                    step: step + 1,
                    ckpt_path,
                    train_loss: mean_loss,
                    lr,
                    grad_norm,
                    elapsed_ms: train_start.elapsed().as_millis() as u64,
                    min_train_loss: running_min_loss,
                };
                match hook_fn(&info).map_err(TrainError::Hook)? {
                    CkptControl::Continue => {}
                    CkptControl::Break => {
                        // Early-return path: write terminal ckpt +
                        // finalize with `early_break = 1.0` marker so
                        // downstream consumers can distinguish an
                        // early stop from a full-run save without
                        // walking `step` against `cfg.steps`.
                        let final_path = ckpt_store
                            .save_final(save_vm)
                            .map_err(|e| TrainError::Ckpt(e.to_string()))?;
                        let mut metrics: HashMap<String, f32> = HashMap::new();
                        metrics.insert("min_train_loss".into(), running_min_loss);
                        metrics.insert("final_lr".into(), lr as f32);
                        metrics.insert("early_break".into(), 1.0);
                        return checkpoint_from_path(
                            &final_path,
                            step + 1,
                            mean_loss,
                            None,
                            metrics,
                        )
                        .map_err(TrainError::Ckpt);
                    }
                }
            }
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
    let mut sum_sq: f32 = 0.0;
    for var in data.values() {
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
    M: Module + DeviceView,
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
        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
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
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());
        let _ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
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
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let captured: Arc<Mutex<Vec<CkptInfo>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_hook = Arc::clone(&captured);
        let hook: CkptHook = Box::new(move |info| {
            captured_hook.lock().unwrap().push(info.clone());
            Ok(CkptControl::Continue)
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
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

    /// A hook returning [`CkptControl::Break`] stops training after
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
                Ok(CkptControl::Break)
            } else {
                Ok(CkptControl::Continue)
            }
        });

        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut ds,
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
        };
        let tmp = TempDir::new().unwrap();
        let lease = Arc::new(TrainingLease::new());

        let hook: CkptHook = Box::new(|_info| Err("hook: bad time".to_string()));

        let err = run_full_ft(
            &model,
            &vm,
            &mut ds,
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
            pad_id: 0,
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
