//! Optimizer state that outlives the run that produced it.
//!
//! A checkpoint written by [`CheckpointStore`](super::ckpt::CheckpointStore)
//! is `varmap.save()` and nothing else: the parameters, and no trace of
//! the optimizer that shaped them. AdamW's two moments and Lion's
//! momentum live in memory for the length of a run and are gone when it
//! ends, so `init_from` restores weights into a freshly-zeroed
//! optimizer — a warm start, not a resume.
//!
//! The difference shows up immediately and quietly. AdamW's update is
//! `m̂ / (√v̂ + ε)` with both moments bias-corrected against the step
//! count; starting them at zero makes the first steps after a restart
//! behave like the first steps of a run, which is exactly when the
//! update is least like the one the schedule assumes. The loss curve
//! bends at the restart and nothing in the record says why.
//!
//! This module writes that state to a sidecar beside the checkpoint,
//! reads it back, and refuses anything short of a complete restore —
//! the same stance [`restore_into`](super::ckpt::restore_into) takes
//! for the weights, and for the same reason: a resume that silently
//! kept half the state is indistinguishable from one that worked until
//! the run is over.
//!
//! # Why a sidecar rather than more tensors in the checkpoint
//!
//! The checkpoint is the artifact everything downstream loads — the
//! Card's `bundle_ref`, `alc.nn.load`, an export. Optimizer state is
//! roughly three times the size of the parameters for AdamW (an FP32
//! master plus two FP32 moments), and it is of no use to anything that
//! is not resuming this exact run. Keeping it in `<name>.opt` leaves
//! the inference path loading what it loaded before.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, TensorId};
use candle_nn::VarMap;

use super::fullft::OptimizerKind;

/// Suffix of the file holding a checkpoint's optimizer state.
///
/// Public because the checkpoint store has to be able to tell a sidecar
/// from a checkpoint: both end in `.safetensors` and both start with
/// the run's prefix, so the rotation would otherwise count sidecars
/// towards its window and delete live checkpoints to make room for
/// them.
pub const OPT_SIDECAR_SUFFIX: &str = ".opt.safetensors";

/// Key under which the optimizer's own step counter is stored.
///
/// A tensor rather than a header field: candle's safetensors writer
/// emits no `__metadata__`, and a one-element tensor is the format's
/// own way of carrying a number. The leading underscores keep it out of
/// the parameter-name space, which never starts with one.
const STEP_KEY: &str = "__step__";

/// Key under which the optimizer flavour is stored, as its
/// [`OptimizerKind`] discriminant.
const KIND_KEY: &str = "__kind__";

/// Path of the optimizer-state file belonging to `ckpt`.
///
/// `run-step40.safetensors` → `run-step40.opt.safetensors`, so the two
/// sort together and a stray sidecar is obvious next to the checkpoint
/// it belongs to.
pub fn sidecar_path(ckpt: &Path) -> PathBuf {
    let mut name = ckpt
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.push_str(OPT_SIDECAR_SUFFIX);
    ckpt.with_file_name(name)
}

/// Which parameter each tensor id belongs to, by the name the `VarMap`
/// registered it under.
///
/// The optimizers hold `Var`s and not names — `VarMap::all_vars()`
/// hands over the values alone — so the naming has to come back from
/// the map. Tensor identity is the join: a `Var` derefs to the tensor
/// whose id the [`GradStore`](candle_core::backprop::GradStore) is also
/// keyed by.
pub fn names_by_tensor_id(vm: &VarMap) -> HashMap<TensorId, String> {
    let data = vm.data().lock().unwrap();
    data.iter()
        .map(|(name, var)| (var.as_tensor().id(), name.clone()))
        .collect()
}

/// Everything one optimizer needs to carry on where it left off.
pub struct OptimizerState {
    /// Which optimizer wrote it. A resume into a different flavour is
    /// refused rather than answered with tensors that mean something
    /// else — Lion's momentum and AdamW's first moment are both an EMA
    /// of the gradient and neither is the other.
    pub kind: OptimizerKind,
    /// Optimizer steps applied before this state was written. AdamW's
    /// bias correction reads it; the training loop resumes its schedule
    /// from it.
    pub step: usize,
    /// Per-parameter tensors, keyed `<parameter name>.<slot>` — the
    /// slot names being whichever the optimizer writes (`master` / `m`
    /// / `v` for AdamW, `master` / `momentum` for Lion).
    pub tensors: HashMap<String, Tensor>,
}

impl OptimizerState {
    /// Write the state to `path` as safetensors.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut out = self.tensors.clone();
        let device = self
            .tensors
            .values()
            .next()
            .map(|t| t.device().clone())
            .unwrap_or(Device::Cpu);
        out.insert(
            STEP_KEY.into(),
            Tensor::new(&[self.step as u32], &device)
                .map_err(|e| format!("optimizer state: step tensor: {e}"))?,
        );
        out.insert(
            KIND_KEY.into(),
            Tensor::new(&[self.kind.discriminant()], &device)
                .map_err(|e| format!("optimizer state: kind tensor: {e}"))?,
        );
        candle_core::safetensors::save(&out, path)
            .map_err(|e| format!("optimizer state: write {}: {e}", path.display()))
    }

    /// Read a state written by [`Self::save`].
    pub fn load(path: &Path, device: &Device) -> Result<Self, String> {
        let mut tensors = candle_core::safetensors::load(path, device)
            .map_err(|e| format!("optimizer state: read {}: {e}", path.display()))?;
        let step = take_scalar(&mut tensors, STEP_KEY, path)? as usize;
        let kind = OptimizerKind::from_discriminant(take_scalar(&mut tensors, KIND_KEY, path)?)
            .ok_or_else(|| {
                format!(
                    "optimizer state: {} names an optimizer this build does not have",
                    path.display()
                )
            })?;
        Ok(Self {
            kind,
            step,
            tensors,
        })
    }
}

/// Pull a one-element `u32` tensor out of a loaded map.
fn take_scalar(
    tensors: &mut HashMap<String, Tensor>,
    key: &str,
    path: &Path,
) -> Result<u32, String> {
    let t = tensors.remove(key).ok_or_else(|| {
        format!(
            "optimizer state: {} carries no `{key}`, so it was not written by this crate",
            path.display()
        )
    })?;
    t.to_dtype(DType::U32)
        .and_then(|t| t.reshape(1))
        .and_then(|t| t.to_vec1::<u32>())
        .map(|v| v[0])
        .map_err(|e| format!("optimizer state: `{key}` in {}: {e}", path.display()))
}

/// Look one parameter's slot tensor up by name, with the refusal spelt
/// out where it happens.
///
/// Shared by both optimizers' restore paths so the two give the same
/// account of a state file that does not fit the model in hand — the
/// common causes being a checkpoint from a different architecture and a
/// vocabulary size that has since changed.
pub fn slot_tensor(
    tensors: &HashMap<String, Tensor>,
    param: &str,
    slot: &str,
    expect: &Tensor,
) -> Result<Tensor, String> {
    let key = format!("{param}.{slot}");
    let t = tensors.get(&key).ok_or_else(|| {
        format!("optimizer state: no `{key}` — the state does not cover `{param}`")
    })?;
    if t.shape() != expect.shape() {
        return Err(format!(
            "optimizer state: `{key}` is {:?} and the parameter is {:?}",
            t.shape(),
            expect.shape()
        ));
    }
    t.to_dtype(expect.dtype())
        .and_then(|t| t.to_device(expect.device()))
        .map_err(|e| format!("optimizer state: `{key}`: {e}"))
}

/// The name a parameter's state is stored under, or a refusal naming
/// the parameter that is missing from the map.
///
/// A `Var` the optimizer holds but the `VarMap` does not name cannot be
/// written or read back, and a state file that silently skipped it
/// would restore into a run whose optimizer is part fresh.
pub fn param_name(names: &HashMap<TensorId, String>, id: TensorId) -> Result<&String, String> {
    names.get(&id).ok_or_else(|| {
        "optimizer state: a parameter the optimizer holds is not registered in the VarMap, \
         so its state has no name to be stored under"
            .to_string()
    })
}
