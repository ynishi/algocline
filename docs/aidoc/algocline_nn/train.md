# algocline-nn::train

Training-side scaffolding.

This module owns four building blocks the trainer entry uses:

- [`data`] — streaming batch abstraction (`Dataset` trait + JSONL /
  Parquet / in-memory implementations).
- [`loss`] — [`loss::Loss`] trait + [`loss::CrossEntropyLoss`], the
  default used by Full FT. A distillation follow-up plugs in the
  same trait.
- [`scheduler`] — cosine-with-warmup learning-rate schedule.
- [`ckpt`] — rotating safetensors [`ckpt::CheckpointStore`] and the
  [`Checkpoint`] record type used by the caller.
- [`fullft`] — [`fullft::run_full_ft`] entry point: `forward → loss →
  backward → optimizer step`, with per-step LR from the scheduler
  and rotating checkpoints from the store.

The Lua bridge only reaches for the top-level re-exports; internal
callers can still pull individual submodule items when needed.

## Types

- `Checkpoint` — Snapshot of a completed training run.

