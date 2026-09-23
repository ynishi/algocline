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

- `allowed_input_sets` — The batch's allowed-id sets as a model input, aligned with the
- `allowed_logit_mask` — Additive logit mask that removes every id a target may not take.
- `run_allowed_ft` — Run Full FT training with the ids allowed at every position handed
- `run_conditioned_ft` — Run Full FT training with a condition supplied per row of every
- `run_distill` — Run a distillation training loop.
- `run_full_ft` — Run Full FT training and return the final checkpoint record.
- `run_lora_ft` — Run LoRA fine-tuning and return the final Δ-only checkpoint record.

## Types

- `CkptControl` — What the hook decided at a checkpoint boundary.
- `CkptFlow` — Whether the trainer continues or breaks early after an
- `CkptHook` — Callback fired at every `ckpt_every` boundary, after the checkpoint
- `CkptInfo` — Information handed to the [`CkptHook`] at every `ckpt_every` boundary.
- `DistillLossKind` — Which distillation loss the caller wants for [`run_distill`].
- `DistillSpec` — Distillation-run configuration.
- `EarlyStop` — When to stop a run that is no longer improving.
- `FullFtConfig` — Hyperparameters for [`run_full_ft`].
- `KeepMark` — A hook's request to hold the checkpoint it was just handed.
- `OptimizerKind` — Which optimizer the trainer requested.
- `TrainError` — Errors surfaced by the training loop.
- `TrainingLease` — One-time guard preventing two Full FT loops from running against
- `TrainingLeaseGuard` — RAII guard that releases the training lease on drop.

