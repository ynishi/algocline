//! Training-side scaffolding.
//!
//! This module owns four building blocks the trainer entry uses:
//!
//! - [`data`] — streaming batch abstraction (`Dataset` trait + JSONL /
//!   Parquet / in-memory implementations).
//! - [`corpus`] — the pre-tokenized corpus file format (its module
//!   documentation is the spec) plus the loader and the round-robin
//!   merge that turn a set of such files into rows a
//!   [`data::TokenizedDataset`] takes.
//! - [`loss`] — [`loss::Loss`] trait + [`loss::CrossEntropyLoss`], the
//!   default used by Full FT. A distillation follow-up plugs in the
//!   same trait.
//! - [`scheduler`] — cosine-with-warmup learning-rate schedule.
//! - [`ckpt`] — rotating safetensors [`ckpt::CheckpointStore`], the
//!   [`Checkpoint`] record type used by the caller, and the restore
//!   side ([`ckpt::restore_into`]) that reads a checkpoint back into a
//!   live `VarMap`.
//! - [`fullft`] — [`fullft::run_full_ft`] entry point: `forward → loss →
//!   backward → optimizer step`, with per-step LR from the scheduler
//!   and rotating checkpoints from the store, plus the two entry points
//!   that hand the model a side channel per batch
//!   ([`fullft::run_conditioned_ft`] / [`fullft::run_allowed_ft`]).
//!
//! The Lua bridge mostly reaches for the top-level re-exports; internal
//! callers can still pull individual submodule items when needed, as
//! the bridge does for [`corpus::interleave`].

use std::collections::HashMap;

pub mod ckpt;
pub mod corpus;
pub mod data;
pub mod lion;
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
    /// Mid-run checkpoints the `on_ckpt` hook asked to hold, in the
    /// order it asked. Each one's file is pinned out of the rotation,
    /// so every path here still exists when the run returns.
    ///
    /// Empty on a run with no hook, and on a run whose hook never
    /// asked to keep anything — which is every run that predates the
    /// keep surface.
    pub candidates: Vec<fullft::Candidate>,
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

/// A model that can be told, once per row, which condition that row was
/// recorded under.
///
/// Sits beside [`DeviceView`] for the same reason that one does: the
/// training loop needs something `candle_nn::Module` does not offer.
/// `Module::forward` takes the ids and nothing else, so a loop driving
/// a model through it has nowhere to put a condition — which is why
/// [`fullft::run_conditioned_ft`] is generic over this instead.
///
/// A separate trait rather than a widened `Module`, because most
/// architectures have no condition to be told about. The compiler keeps
/// them out of the conditioned entry point, which is a better place for
/// that fact than a runtime error raised half-way through a run.
pub trait ConditionedForward {
    /// Forward `xs` (`[batch, seq]` token ids) with `per_row`
    /// conditions per row, row-major, returning `[batch, seq, vocab]`
    /// logits.
    ///
    /// Indices **per row**, because a batch mixes conditions: a single
    /// index for the whole batch would condition most rows on some
    /// other row's condition, and every shape downstream would still
    /// line up. `per_row` is `1` for every single-slot dataset and the
    /// slot count for a multi-slot one
    /// ([`data::Batch::conds_per_row`] carries it alongside the list it
    /// describes).
    ///
    /// The argument is a [`crate::arch::CondIndex`] rather than a
    /// number because the numbering a caller holds is usually a
    /// different one — a token id, most often — and the two ranges
    /// overlap without meaning the same thing.
    ///
    /// # Errors
    ///
    /// Implementation-defined, but `conds.len() != batch * per_row`,
    /// `per_row` of zero and an index outside the implementation's
    /// table are expected to be refused rather than absorbed.
    fn forward_conditioned_rows(
        &self,
        xs: &candle_core::Tensor,
        conds: &[crate::arch::CondIndex],
        per_row: usize,
    ) -> candle_core::Result<candle_core::Tensor>;
}

/// A model that can be told, at every position, which ids the answer
/// there may take.
///
/// Sibling of [`ConditionedForward`], and a separate trait for the same
/// reason: most architectures model no constrained id space and would
/// have to answer for a concept they do not carry. The compiler keeps
/// them out of [`fullft::run_allowed_ft`].
///
/// The argument is a [`crate::arch::AllowedSets`] rather than nested
/// `Vec`s because the padding and the per-position counts have to agree
/// with each other, and that agreement is the constructor's business
/// rather than every caller's.
pub trait AllowedForward {
    /// Forward `xs` (`[batch, seq]` token ids) with one set of allowed
    /// ids per position, returning `[batch, seq, vocab]` logits.
    ///
    /// # Errors
    ///
    /// Implementation-defined, but sets that do not cover exactly
    /// `[batch, seq]`, and ids outside the vocabulary, are expected to
    /// be refused rather than absorbed.
    fn forward_allowed_rows(
        &self,
        xs: &candle_core::Tensor,
        allowed: &crate::arch::AllowedSets,
    ) -> candle_core::Result<candle_core::Tensor>;
}

pub use ckpt::{
    checkpoint_from_path, restore_into, restore_into_partial, ApplyStage, CheckpointStore,
    RestoreError, RestoreReport, TensorMismatch,
};
// The types carry their noun and are re-exported; `interleave` /
// `interleave_labelled` do not, and `train::interleave` would read as
// "interleave datasets" rather than "interleave corpora", so they stay
// at `train::corpus::`.
pub use corpus::{CorpusError, CorpusFile, InterleavedRow};
pub use data::{
    Batch, Dataset, DatasetError, DatasetOpts, JsonlDataset, ParquetDataset, TeacherCardDataset,
    TokenizedDataset,
};
pub use fullft::{
    allowed_input_sets, allowed_logit_mask, run_allowed_ft, run_conditioned_ft, run_distill,
    run_full_ft, run_lora_ft, Candidate, CkptControl, CkptFlow, CkptHook, CkptInfo,
    DistillLossKind, DistillSpec, FullFtConfig, KeepMark, OptimizerKind, TrainError, TrainingLease,
    TrainingLeaseGuard,
};
pub use lion::{Lion, ParamsLion};
pub use loss::{CrossEntropyLoss, HardLabelDistillLoss, Loss, Reduction};
pub use mixed::MixedAdamW;
pub use scheduler::{ScheduleKind, Scheduler};
