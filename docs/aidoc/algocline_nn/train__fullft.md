# algocline-nn::train::fullft

Full FT training loop.

Ties [`crate::arch::gpt2::Gpt2Model`], a [`crate::train::data::Dataset`]
source, an AdamW optimizer, a learning-rate [`Scheduler`], and a
rotating [`CheckpointStore`] together into a single
[`run_full_ft`] entry point.

The loop is intentionally CPU-friendly: the tests build a
2-layer / 2-head / 16-dim model and overfit a synthetic 4-token
sequence in ~100 steps. On a real GPU the same code path scales to
the full 355M / 774M presets without further changes because every
candle operation used here already dispatches on the device the
`VarMap` was built with.

## Functions

- `run_distill` — Run a distillation training loop.
- `run_full_ft` — Run Full FT training and return the final checkpoint record.
- `run_lora_ft` — Run LoRA fine-tuning and return the final Δ-only checkpoint record.

## Types

- `DistillLossKind` — Which distillation loss the caller wants for [`run_distill`].
- `DistillSpec` — Distillation-run configuration.
- `FullFtConfig` — Hyperparameters for [`run_full_ft`].
- `TrainError` — Errors surfaced by the training loop.
- `TrainingLease` — One-time guard preventing two Full FT loops from running against
- `TrainingLeaseGuard` — RAII guard that releases the training lease on drop.

