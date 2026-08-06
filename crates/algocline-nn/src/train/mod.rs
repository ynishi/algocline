//! Training-side scaffolding.
//!
//! This module owns four building blocks the trainer entry uses:
//!
//! - [`data`] — streaming batch abstraction (`Dataset` trait + JSONL /
//!   Parquet / in-memory implementations).
//! - [`loss`] — [`loss::Loss`] trait + [`loss::CrossEntropyLoss`], the
//!   default used by Full FT. A distillation follow-up plugs in the
//!   same trait.
//! - [`scheduler`] — cosine-with-warmup learning-rate schedule.
//! - [`ckpt`] — rotating safetensors [`ckpt::CheckpointStore`], the
//!   [`Checkpoint`] record type used by the caller, and
//!   [`ckpt::restore_into`], which puts a checkpoint back into a
//!   trainable `VarMap` after checking that it describes the same model.
//! - [`fullft`] — [`fullft::run_full_ft`] entry point: `forward → loss →
//!   backward → optimizer step`, with per-step LR from the scheduler
//!   and rotating checkpoints from the store.
//!
//! The Lua bridge only reaches for the top-level re-exports; internal
//! callers can still pull individual submodule items when needed.

use std::collections::HashMap;

pub mod ckpt;
pub mod data;
pub mod loss;
pub mod mixed;
pub mod scheduler;

#[path = "loop.rs"]
pub mod fullft;

/// Snapshot of a completed training run.
///
/// Returned by [`fullft::run_full_ft`] and consumed by the Lua bridge
/// when it converts the run outcome into the Card metadata block. Kept
/// deliberately flat (owned strings + primitives + `HashMap`) so it
/// crosses the mlua boundary without borrowing gymnastics.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Path of the terminal `<prefix>.safetensors` file, relative to
    /// the checkpoint directory (e.g. `"tinystories.safetensors"`).
    pub bundle_ref: String,
    /// Step index when training terminated. Equal to the requested
    /// `cfg.steps` on a full run.
    pub step: usize,
    /// Loss on the last-seen training batch. Not necessarily the
    /// minimum — use `metrics["min_train_loss"]` for that.
    pub train_loss: f32,
    /// Optional validation loss, populated by callers that hold out a
    /// slice of the dataset.
    pub val_loss: Option<f32>,
    /// Per-run scalar metrics keyed by name.
    pub metrics: HashMap<String, f32>,
}

/// Access to the [`candle_core::Device`] a trainable model was built
/// against.
///
/// Pairs with `candle_nn::Module` to give a training loop everything
/// it needs to drive any architecture: `Module::forward` for the pass
/// itself, `DeviceView::device` for input-tensor placement (the loop
/// needs to know the target device before it can move a `Batch`'s
/// input tensor onto it).
///
/// Implemented for every trainable model in `arch::` (currently
/// [`crate::arch::Gpt2Model`] and [`crate::arch::TinyLlamaModel`]) as
/// a thin delegate to the underlying `Config::device` field. New
/// architectures pick up training for free once they implement this
/// plus `candle_nn::Module`.
pub trait DeviceView {
    /// The device this model's parameters were built against. All
    /// input tensors handed to `Module::forward` must reside on the
    /// same device.
    fn device(&self) -> &candle_core::Device;
}

/// A model that can be told which condition each row of a batch
/// carries.
///
/// Sits beside [`DeviceView`] for the same reason that one does: the
/// training loop needs something `candle_nn::Module` does not offer.
/// `Module::forward` takes the ids and nothing else, so a loop driving
/// a model through it has nowhere to put a condition — which is why
/// [`fullft::run_conditioned_ft`] is generic over this instead.
///
/// A separate trait rather than a widened `Module`, because most
/// architectures have no condition to be told about.
/// [`crate::arch::TinyLlamaModel`] does not implement this and grows
/// nothing from its existence; the compiler keeps it out of the
/// conditioned entry point, which is a better place for that fact than
/// a runtime error raised half-way through a run.
pub trait ConditionedForward {
    /// Forward `xs` (`[batch, seq]` token ids) with one condition per
    /// row, returning `[batch, seq, vocab]` logits.
    ///
    /// One index **per row**, because a batch mixes conditions: a
    /// single index for the whole batch would condition most rows on
    /// some other row's band, and every shape downstream would still
    /// line up.
    ///
    /// The argument is a [`crate::arch::CondIndex`] rather than a
    /// number because the numbering a caller holds is usually a
    /// different one — a token id, most often — and the two ranges
    /// overlap without meaning the same thing. Its documentation has
    /// the arithmetic.
    ///
    /// # Errors
    ///
    /// Implementation-defined, but `conds.len() != batch` and an index
    /// outside the implementation's table are expected to be refused
    /// rather than absorbed.
    fn forward_conditioned_rows(
        &self,
        xs: &candle_core::Tensor,
        conds: &[crate::arch::CondIndex],
    ) -> candle_core::Result<candle_core::Tensor>;
}

pub use ckpt::{
    checkpoint_from_path, restore_into, restore_into_partial, ApplyStage, CheckpointStore,
    RestoreError, RestoreReport, TensorMismatch,
};
pub use data::{
    Batch, Dataset, DatasetError, DatasetOpts, JsonlDataset, ParquetDataset, TeacherCardDataset,
    TokenizedDataset,
};
pub use fullft::{
    allowed_logit_mask, run_conditioned_ft, run_distill, run_full_ft, run_lora_ft, CkptControl,
    CkptHook, CkptInfo, DistillLossKind, DistillSpec, FullFtConfig, TrainError, TrainingLease,
    TrainingLeaseGuard,
};
pub use loss::{CrossEntropyLoss, HardLabelDistillLoss, Loss, Reduction};
pub use mixed::MixedAdamW;
pub use scheduler::{ScheduleKind, Scheduler};
