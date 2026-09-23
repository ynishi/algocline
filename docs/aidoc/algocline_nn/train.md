# algocline-nn::train

Training-side scaffolding.

This module owns four building blocks the trainer entry uses:

- [`data`] — streaming batch abstraction (`Dataset` trait + JSONL /
  Parquet / in-memory implementations).
- [`corpus`] — the pre-tokenized corpus file format (its module
  documentation is the spec) plus the loader and the round-robin
  merge that turn a set of such files into rows a
  [`data::TokenizedDataset`] takes.
- [`loss`] — [`loss::Loss`] trait + [`loss::CrossEntropyLoss`], the
  default used by Full FT. A distillation follow-up plugs in the
  same trait.
- [`scheduler`] — cosine-with-warmup learning-rate schedule.
- [`ckpt`] — rotating safetensors [`ckpt::CheckpointStore`], the
  [`Checkpoint`] record type used by the caller, and the restore
  side ([`ckpt::restore_into`]) that reads a checkpoint back into a
  live `VarMap`.
- [`fullft`] — [`fullft::run_full_ft`] entry point: `forward → loss →
  backward → optimizer step`, with per-step LR from the scheduler
  and rotating checkpoints from the store, plus the two entry points
  that hand the model a side channel per batch
  ([`fullft::run_conditioned_ft`] / [`fullft::run_allowed_ft`]).

The Lua bridge mostly reaches for the top-level re-exports; internal
callers can still pull individual submodule items when needed, as
the bridge does for [`corpus::interleave`].

## Types

- `Checkpoint` — Snapshot of a completed training run.

## Traits

- `AllowedForward` — A model that can be told, at every position, which ids the answer
- `ConditionedForward` — A model that can be told, once per row, which condition that row was
- `DeviceView` — Access to the [`candle_core::Device`] a trainable model was built

