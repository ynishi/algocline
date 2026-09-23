# algocline-nn::train::ckpt

Rotating safetensors checkpoint writer, and the restore side that
reads one back into a live `VarMap`.

During Full FT the trainer writes an intermediate checkpoint every
`ckpt_every` steps and keeps only the most recent `ckpt_keep`
files on disk. The older files are dropped by modification time
rather than by step number so a manual `touch` cannot hide the
trainer's own bookkeeping from `ls -t`.

A step can be [pinned](CheckpointStore::pin), which lifts it out of
that rotation entirely: pinned files neither count against
`ckpt_keep` nor get dropped by it. That is what lets a checkpoint
search hold on to a candidate it liked at step 40 while the window
keeps turning over around it — without a pin, the file the search
selected is deleted as soon as `ckpt_keep` newer ones are written.

[`restore_into`] / [`restore_into_partial`] are the other direction:
a checkpoint back into the variables a model was built against, so a
run can continue from weights an earlier run produced. Both verify
the whole map before writing anything and report what they did, name
by name — see [`restore_into`] for what that adds over
`candle_nn::VarMap::load`.

## Functions

- `checkpoint_from_path` — Build a [`Checkpoint`] record from a save path and per-run metrics.
- `identity_sidecar_path` — Path of the sidecar describing the bundle at `path`.
- `producer` — The value written under [`PRODUCER_KEY`]: `algocline <version>`.
- `read_bundle_header` — Read the header a bundle carries.
- `restore_into` — Load a checkpoint into a live `VarMap`, refusing anything short of a
- `restore_into_partial` — Load a checkpoint into a live `VarMap`, accepting that some

## Types

- `ApplyStage` — Which half of the apply pass failed, for [`RestoreError::Apply`].
- `BundleIdentity` — What a checkpoint says about itself.
- `Candidate` — A checkpoint the hook asked to hold, as it stood when it was held.
- `CheckpointStore` — Rotating checkpoint writer.
- `MetricPoint` — One step of a run, as the metrics file records it.
- `RestoreError` — Why a restore refused to run, or stopped part-way.
- `RestoreReport` — What a restore did, name by name.
- `TensorMismatch` — One name whose tensor in the checkpoint does not describe the same

## Constants

- `ARCHITECTURE_KEY` — Key holding the architecture preset id.
- `BUNDLE_SCHEMA` — Value of [`SCHEMA_KEY`] for the key set this version writes.
- `CARD_ID_KEY` — Key holding the Card id the weights belong to.
- `FORMAT_KEY` — The one `__metadata__` key safetensors readers share.
- `FORMAT_PT` — The de-facto value of [`FORMAT_KEY`]: PyTorch's tensor layout,
- `KIND_CHECKPOINT` — Value of [`KIND_KEY`] in a trainer checkpoint's header.
- `KIND_KEY` — Key naming which of algocline's files this is. [`SCHEMA_KEY`] says
- `PRODUCER_KEY` — Key naming the program that wrote the file ([`producer`]).
- `SCHEMA_KEY` — Key whose presence marks a header as algocline's. Everything else

