# algocline-nn::train::ckpt

Rotating safetensors checkpoint writer.

During Full FT the trainer writes an intermediate checkpoint every
`ckpt_every` steps and keeps only the most recent `ckpt_keep`
files on disk. The older files are dropped by modification time
rather than by step number so a manual `touch` cannot hide the
trainer's own bookkeeping from `ls -t`.

## Functions

- `checkpoint_from_path` — Build a [`Checkpoint`] record from a save path and per-run metrics.

## Types

- `CheckpointStore` — Rotating checkpoint writer.

