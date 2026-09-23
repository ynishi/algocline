# algocline-nn::train::optstate

Optimizer state that outlives the run that produced it.

A checkpoint written by [`CheckpointStore`](super::ckpt::CheckpointStore)
is `varmap.save()` and nothing else: the parameters, and no trace of
the optimizer that shaped them. AdamW's two moments and Lion's
momentum live in memory for the length of a run and are gone when it
ends, so `init_from` restores weights into a freshly-zeroed
optimizer — a warm start, not a resume.

The difference shows up immediately and quietly. AdamW's update is
`m̂ / (√v̂ + ε)` with both moments bias-corrected against the step
count; starting them at zero makes the first steps after a restart
behave like the first steps of a run, which is exactly when the
update is least like the one the schedule assumes. The loss curve
bends at the restart and nothing in the record says why.

This module writes that state to a sidecar beside the checkpoint,
reads it back, and refuses anything short of a complete restore —
the same stance [`restore_into`](super::ckpt::restore_into) takes
for the weights, and for the same reason: a resume that silently
kept half the state is indistinguishable from one that worked until
the run is over.

# Why a sidecar rather than more tensors in the checkpoint

The checkpoint is the artifact everything downstream loads — the
Card's `bundle_ref`, `alc.nn.load`, an export. Optimizer state is
roughly three times the size of the parameters for AdamW (an FP32
master plus two FP32 moments), and it is of no use to anything that
is not resuming this exact run. Keeping it in `<name>.opt` leaves
the inference path loading what it loaded before.

## Functions

- `names_by_tensor_id` — Which parameter each tensor id belongs to, by the name the `VarMap`
- `param_name` — The name a parameter's state is stored under, or a refusal naming
- `sidecar_path` — Path of the optimizer-state file belonging to `ckpt`.
- `slot_tensor` — Look one parameter's slot tensor up by name, with the refusal spelt

## Types

- `OptimizerState` — Everything one optimizer needs to carry on where it left off.

## Constants

- `OPT_SIDECAR_SUFFIX` — Suffix of the file holding a checkpoint's optimizer state.

