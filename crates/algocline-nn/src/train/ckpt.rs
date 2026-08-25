//! Rotating safetensors checkpoint writer, and the restore side that
//! reads one back into a live `VarMap`.
//!
//! During Full FT the trainer writes an intermediate checkpoint every
//! `ckpt_every` steps and keeps only the most recent `ckpt_keep`
//! files on disk. The older files are dropped by modification time
//! rather than by step number so a manual `touch` cannot hide the
//! trainer's own bookkeeping from `ls -t`.
//!
//! [`restore_into`] / [`restore_into_partial`] are the other direction:
//! a checkpoint back into the variables a model was built against, so a
//! run can continue from weights an earlier run produced. Both verify
//! the whole map before writing anything and report what they did, name
//! by name — see [`restore_into`] for what that adds over
//! `candle_nn::VarMap::load`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::DType;
use candle_nn::VarMap;

use super::Checkpoint;

/// Rotating checkpoint writer.
///
/// Filenames follow the shape
/// `<prefix>-step<N>.safetensors` under `ckpt_dir`. The prefix defaults
/// to `"ckpt"` but the trainer passes in a card id so multiple
/// concurrent runs (should they ever be allowed) do not collide.
#[derive(Debug, Clone)]
pub struct CheckpointStore {
    dir: PathBuf,
    prefix: String,
    keep: usize,
}

impl CheckpointStore {
    /// Build a checkpoint writer.
    ///
    /// `dir` is created if it does not exist. `keep` clamps to at least
    /// 1 so at least one checkpoint always survives rotation.
    pub fn new(
        dir: impl AsRef<Path>,
        prefix: impl Into<String>,
        keep: usize,
    ) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            prefix: prefix.into(),
            keep: keep.max(1),
        })
    }

    /// Directory the writer is targeting.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename prefix in use.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Number of checkpoints kept.
    pub fn keep(&self) -> usize {
        self.keep
    }

    /// Path a `step` checkpoint would be written to.
    pub fn path_for_step(&self, step: usize) -> PathBuf {
        self.dir
            .join(format!("{}-step{}.safetensors", self.prefix, step))
    }

    /// Dump the `varmap` contents as a safetensors file, then prune old
    /// files down to `keep` entries.
    pub fn save_step(&self, varmap: &VarMap, step: usize) -> candle_core::Result<PathBuf> {
        let path = self.path_for_step(step);
        varmap.save(&path)?;
        self.prune()
            .map_err(|e| candle_core::Error::Msg(format!("ckpt prune: {e}")))?;
        Ok(path)
    }

    /// Save the final checkpoint as `<prefix>.safetensors` (no step
    /// suffix). The trainer calls this at the end of a run so the
    /// last-good weights sit under a stable filename that downstream
    /// consumers can reference without knowing the step count.
    pub fn save_final(&self, varmap: &VarMap) -> candle_core::Result<PathBuf> {
        let path = self.dir.join(format!("{}.safetensors", self.prefix));
        varmap.save(&path)?;
        Ok(path)
    }

    /// Enumerate the step checkpoints currently on disk, oldest first.
    pub fn list(&self) -> std::io::Result<Vec<PathBuf>> {
        let mut entries: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        let step_prefix = format!("{}-step", self.prefix);
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.starts_with(&step_prefix) || !name.ends_with(".safetensors") {
                continue;
            }
            let mtime = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
            entries.push((path, mtime));
        }
        entries.sort_by_key(|(_, m)| *m);
        Ok(entries.into_iter().map(|(p, _)| p).collect())
    }

    fn prune(&self) -> std::io::Result<()> {
        let files = self.list()?;
        if files.len() <= self.keep {
            return Ok(());
        }
        let drop_count = files.len() - self.keep;
        for path in files.into_iter().take(drop_count) {
            fs::remove_file(&path)?;
        }
        Ok(())
    }
}

/// Build a [`Checkpoint`] record from a save path and per-run metrics.
///
/// The trainer calls this after `save_step` / `save_final` so the
/// returned struct carries the actual bundle path plus the metrics
/// snapshot the caller will attach to the Card.
pub fn checkpoint_from_path(
    bundle_path: &Path,
    step: usize,
    train_loss: f32,
    val_loss: Option<f32>,
    metrics: std::collections::HashMap<String, f32>,
) -> Result<Checkpoint, String> {
    let bundle_ref = bundle_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(String::from)
        .ok_or_else(|| {
            format!(
                "checkpoint_from_path: bundle path has no valid UTF-8 file name: {bundle_path:?}"
            )
        })?;
    Ok(Checkpoint {
        bundle_ref,
        step,
        train_loss,
        val_loss,
        metrics,
    })
}

/// One name whose tensor in the checkpoint does not describe the same
/// parameter as the one registered under that name in the `VarMap`.
///
/// Both halves are carried so the error message can say what was
/// expected and what was found, which is the difference between "the
/// restore failed" and "this checkpoint was written at one vocabulary
/// size and the map wants another".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMismatch {
    /// Tensor name, identical on both sides — only shape or dtype differ.
    pub name: String,
    /// Shape of the variable registered in the `VarMap`.
    pub expected_shape: Vec<usize>,
    /// Shape of the tensor stored in the checkpoint.
    pub found_shape: Vec<usize>,
    /// Dtype of the variable registered in the `VarMap`.
    pub expected_dtype: DType,
    /// Dtype of the tensor stored in the checkpoint.
    pub found_dtype: DType,
}

impl std::fmt::Display for TensorMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: map holds {:?}{:?}, checkpoint holds {:?}{:?}",
            self.name, self.expected_dtype, self.expected_shape, self.found_dtype, self.found_shape
        )
    }
}

/// Which half of the apply pass failed, for [`RestoreError::Apply`].
///
/// Both leave the same half-written map; they differ in whether the
/// checkpoint or the variable is the thing to look at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyStage {
    /// Reading the tensor out of the checkpoint.
    Load,
    /// Writing the loaded tensor into its `Var`.
    Set,
}

impl std::fmt::Display for ApplyStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyStage::Load => write!(f, "loading"),
            ApplyStage::Set => write!(f, "writing"),
        }
    }
}

/// Why a restore refused to run, or stopped part-way.
///
/// Every variant names the checkpoint and, where one exists, the
/// offending tensor: a failed restore that cannot say *which* parameter
/// disagreed leaves the caller no better off than the silent
/// half-restore this type exists to prevent.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    /// The file could not be opened or is not a readable safetensors
    /// container.
    #[error("restore from {path:?}: cannot read the checkpoint: {message}")]
    Open {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// Underlying candle / IO message.
        message: String,
    },

    /// The `VarMap` mutex was left poisoned by a panicking holder, so
    /// the registered variables cannot be trusted to be consistent.
    #[error("restore from {path:?}: the VarMap lock is poisoned; a previous holder panicked")]
    Poisoned {
        /// Checkpoint that was being read.
        path: PathBuf,
    },

    /// The map has nothing registered. Restoring into it would report a
    /// flawless zero-of-zero result, which is exactly the false success
    /// this API exists to rule out — most often it means the model was
    /// built against a different `VarMap` than the one passed here.
    #[error(
        "restore from {path:?}: the VarMap has no registered variables, \
         so a restore would silently do nothing"
    )]
    NoRegisteredVars {
        /// Checkpoint that was being read.
        path: PathBuf,
    },

    /// Under the strict entry point, some registered variable had no
    /// counterpart in the file and would have kept its initial value.
    #[error(
        "restore from {path:?}: {} of {registered} registered variables are absent from the \
         checkpoint ({}); training would have continued from their initial values. Use \
         `restore_into_partial` if that is intended",
        .absent.len(),
        preview(.absent),
    )]
    Incomplete {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// How many variables the `VarMap` holds in total.
        registered: usize,
        /// Names registered in the map but absent from the file.
        absent: Vec<String>,
    },

    /// A name present on both sides describes a different tensor in
    /// each. Never tolerated: a shape or dtype disagreement means the
    /// two models are not the same model.
    #[error(
        "restore from {path:?}: {} tensor(s) disagree with the checkpoint: {}",
        .mismatches.len(),
        preview_mismatches(.mismatches),
    )]
    Mismatch {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// Every disagreement found, not just the first.
        mismatches: Vec<TensorMismatch>,
    },

    /// The map and the file have no name in common. Writing nothing and
    /// returning success would be the same false resume as
    /// [`RestoreError::NoRegisteredVars`], one step further along: the
    /// map holds variables, they are simply not these. A LoRA-wrapped
    /// map pointed at another run's checkpoint, or a file whose names
    /// are `model.*` where the map registers `base.*`, both land here.
    #[error(
        "restore from {path:?}: none of the {registered} registered variables appear among the \
         checkpoint's {in_file} tensor(s), so the restore would have written nothing; \
         the map and the file are not two views of the same model"
    )]
    NothingRestored {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// How many variables the `VarMap` holds.
        registered: usize,
        /// How many tensors the checkpoint holds.
        in_file: usize,
    },

    /// A tensor could not be read while *verifying*, or carries a dtype
    /// that does not map onto a `Var`. Nothing has been written when
    /// this is raised — the mid-apply counterpart is
    /// [`RestoreError::Apply`].
    #[error("restore from {path:?}: reading tensor {name:?}: {message}")]
    Read {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// Tensor that failed to load.
        name: String,
        /// Underlying candle message.
        message: String,
    },

    /// A tensor that verification accepted could not be applied. Unlike
    /// every verdict above, this one fires part-way through writing:
    /// the variables in `written` already hold checkpoint values while
    /// the rest hold what they had, so the map is a mixture of two
    /// models and should be rebuilt rather than trained on.
    ///
    /// An empty `written` is not by itself a promise that the map
    /// survived. It is one under [`ApplyStage::Load`], which reads the
    /// checkpoint and touches no variable; under [`ApplyStage::Set`]
    /// the failing write was already under way, and candle reports no
    /// boundary between "refused" and "copied half of it". The
    /// [`std::fmt::Display`] text distinguishes the two.
    #[error(
        "restore from {path:?}: {stage} tensor {name:?} failed while applying the checkpoint; \
         {}: {message}",
        applied_state(.stage, .written),
    )]
    Apply {
        /// Checkpoint that was being read.
        path: PathBuf,
        /// Which half of the apply step failed.
        stage: ApplyStage,
        /// Tensor the restore stopped on.
        name: String,
        /// Variables already written, in the order they were applied.
        written: Vec<String>,
        /// Underlying candle message.
        message: String,
    },
}

/// What a restore did, name by name.
///
/// Returned by both entry points, including the permissive one — a
/// caller that opts into a partial restore still has to be able to see
/// which variables were skipped, otherwise "partial" degrades back into
/// the silence this module is built to avoid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    /// Checkpoint the values came from.
    pub path: PathBuf,
    /// Names found in both the file and the map, whose variables now
    /// hold checkpoint values. Sorted.
    pub restored: Vec<String>,
    /// Names registered in the map but absent from the file. These keep
    /// whatever value they were initialised with. Always empty after a
    /// successful [`restore_into`]. Sorted.
    pub absent_from_file: Vec<String>,
    /// Names present in the file but not registered in the map. Nothing
    /// was done with them. Sorted.
    pub unused_from_file: Vec<String>,
}

impl RestoreReport {
    /// How many variables the `VarMap` had registered.
    pub fn registered_count(&self) -> usize {
        self.restored.len() + self.absent_from_file.len()
    }

    /// Number of variables that took a value from the checkpoint.
    pub fn restored_count(&self) -> usize {
        self.restored.len()
    }

    /// Whether every registered variable was restored. Always true for
    /// a successful [`restore_into`]; worth checking after
    /// [`restore_into_partial`].
    pub fn is_complete(&self) -> bool {
        self.absent_from_file.is_empty()
    }

    /// One-line summary suitable for a training log.
    pub fn summary(&self) -> String {
        format!(
            "{} of {} variables restored from {:?} ({} left at their initial value, \
             {} tensors in the file unused)",
            self.restored.len(),
            self.registered_count(),
            self.path,
            self.absent_from_file.len(),
            self.unused_from_file.len(),
        )
    }
}

/// How much of the map a restore is allowed to leave untouched.
///
/// Kept private and reached through two named entry points rather than
/// exposed as a parameter: the permissive mode is a deliberate,
/// narrow-purpose choice (a base checkpoint into a map that legitimately
/// holds more than the base), and a call site that wants it should have
/// to say so by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Every registered variable must be present in the file.
    Complete,
    /// Variables the file does not carry keep their current value.
    AllowAbsent,
}

/// Load a checkpoint into a live `VarMap`, refusing anything short of a
/// complete restore.
///
/// The variables stay `Var`s, so the map remains trainable: this is the
/// supported way to start a run from previously trained weights,
/// whether to resume an interrupted run or to stage a curriculum across
/// several runs.
///
/// # What this adds over `candle_nn::VarMap::load`
///
/// `VarMap::load` is loud about most disagreements. It iterates the
/// *registered* variables, so a variable the file does not carry stops
/// it with `CannotFindTensor`, and a shape that differs stops it with
/// the mismatch `Var::set` raises. Two things it does pass over in
/// silence:
///
/// - **an empty map.** The loop body never runs, `Ok(())` comes back,
///   and the caller has a fully random initialisation reported as a
///   resume. This is the one that costs a run: it is what a model built
///   against a different `VarMap` than the one handed to the loader
///   looks like, and nothing downstream distinguishes it from a real
///   restore. [`RestoreError::NoRegisteredVars`] names it instead;
/// - **tensors in the file that the map does not register.** Dropped
///   without a word. Often legitimate — a full-model checkpoint read
///   into a map that holds part of it — but the caller should still be
///   able to see it, so they arrive on the report as
///   [`RestoreReport::unused_from_file`].
///
/// The rest is a matter of degree rather than of silence: the map is
/// borrowed shared rather than mutably, every disagreement is collected
/// instead of only the first, and all of them are checked before any
/// variable is written, so a refusal leaves the map as it was rather
/// than half-loaded. The rules:
///
/// - a name registered in the map but absent from the file is an error
///   here (see [`restore_into_partial`] for the one case where it is
///   not);
/// - a name in both, carrying a different shape or dtype, is an error
///   in either mode;
/// - a name in the file that the map does not know is reported, not
///   rejected;
/// - a map that shares no name at all with the file is an error in
///   either mode, since a restore that writes nothing has not resumed
///   anything.
///
/// Every verdict above is reached before the first write, so a rejected
/// restore leaves the map exactly as it was. The single exception is
/// [`RestoreError::Apply`], which fires part-way through writing and
/// says which variables were already updated.
///
/// # Caller obligation
///
/// The checkpoint is mmapped for the duration of the call, so the
/// caller must ensure nothing modifies the file while it runs. A write
/// underneath a live mapping is undefined behaviour, not merely a
/// changed value, and no check inside this function can detect one.
/// Checkpoints this crate writes are created once and unlinked by
/// rotation, which satisfies the requirement for a path that came out
/// of [`CheckpointStore`] — but that is a property of those files and
/// not of the argument.
///
/// # Errors
///
/// Every [`RestoreError`] variant; see each for what it means.
pub fn restore_into(varmap: &VarMap, path: &Path) -> Result<RestoreReport, RestoreError> {
    restore(varmap, path, Scope::Complete)
}

/// Load a checkpoint into a live `VarMap`, accepting that some
/// registered variables have no counterpart in the file.
///
/// This is for restoring a base model into a map that legitimately
/// holds more than the base: the LoRA legs of a wrapped model have
/// never been trained, so no base checkpoint can carry them, and
/// insisting on a complete restore would make "LoRA on weights you
/// trained yourself" impossible to express.
///
/// Everything else is as strict as [`restore_into`]: a name present in
/// both with a different shape or dtype is still an error, a map that
/// overlaps the file nowhere is still an error, and the returned
/// [`RestoreReport`] still lists every skipped name so the caller can
/// check the skips are the ones it expected.
///
/// The mmap obligation described on [`restore_into`] applies here
/// unchanged: the caller warrants that nothing writes to `path` while
/// this runs.
///
/// # Errors
///
/// Every [`RestoreError`] variant except [`RestoreError::Incomplete`],
/// which is precisely what this entry point permits.
pub fn restore_into_partial(varmap: &VarMap, path: &Path) -> Result<RestoreReport, RestoreError> {
    restore(varmap, path, Scope::AllowAbsent)
}

fn restore(varmap: &VarMap, path: &Path, scope: Scope) -> Result<RestoreReport, RestoreError> {
    // SAFETY: the mapping is unsound if the file changes underneath it,
    // and `path` is the caller's. The requirement is therefore stated
    // as a caller obligation on the two public entry points; nothing
    // observable from here would catch a concurrent writer.
    let file = unsafe { MmapedSafetensors::new(path) }.map_err(|e| RestoreError::Open {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    // The lock is held across the whole restore. Verifying under one
    // lock and writing under another would let a concurrent
    // registration slip a variable in between the two, and the check
    // would then be describing a map that no longer exists.
    let registered = varmap.data().lock().map_err(|_| RestoreError::Poisoned {
        path: path.to_path_buf(),
    })?;
    if registered.is_empty() {
        return Err(RestoreError::NoRegisteredVars {
            path: path.to_path_buf(),
        });
    }

    let in_file: BTreeSet<String> = file.tensors().into_iter().map(|(name, _)| name).collect();

    // Pass one: partition the names and compare metadata. Nothing is
    // written here, so any verdict below can still refuse cleanly.
    let mut restored: Vec<String> = Vec::new();
    let mut absent_from_file: Vec<String> = Vec::new();
    let mut mismatches: Vec<TensorMismatch> = Vec::new();
    for (name, var) in registered.iter() {
        if !in_file.contains(name) {
            absent_from_file.push(name.clone());
            continue;
        }
        let view = file.get(name).map_err(|e| RestoreError::Read {
            path: path.to_path_buf(),
            name: name.clone(),
            message: e.to_string(),
        })?;
        // A dtype with no `DType` counterpart is a read failure rather
        // than a mismatch: there is no `found_dtype` to report.
        let found_dtype = DType::try_from(view.dtype()).map_err(|e| RestoreError::Read {
            path: path.to_path_buf(),
            name: name.clone(),
            message: format!("checkpoint dtype does not map onto a Var dtype: {e}"),
        })?;
        if view.shape() != var.shape().dims() || found_dtype != var.dtype() {
            mismatches.push(TensorMismatch {
                name: name.clone(),
                expected_shape: var.shape().dims().to_vec(),
                found_shape: view.shape().to_vec(),
                expected_dtype: var.dtype(),
                found_dtype,
            });
            continue;
        }
        restored.push(name.clone());
    }
    // `HashMap` iteration order is arbitrary; sorting makes the report
    // and the error messages reproducible between runs.
    restored.sort();
    absent_from_file.sort();
    mismatches.sort_by(|a, b| a.name.cmp(&b.name));

    if !mismatches.is_empty() {
        return Err(RestoreError::Mismatch {
            path: path.to_path_buf(),
            mismatches,
        });
    }
    if scope == Scope::Complete && !absent_from_file.is_empty() {
        return Err(RestoreError::Incomplete {
            path: path.to_path_buf(),
            registered: registered.len(),
            absent: absent_from_file,
        });
    }
    // Checked after the two verdicts above so their precedence is
    // untouched: under `Scope::Complete` a zero overlap is already an
    // `Incomplete`, and this is what catches the same thing under
    // `Scope::AllowAbsent`, where every name would otherwise be excused
    // one at a time into a restore that writes nothing.
    if restored.is_empty() {
        return Err(RestoreError::NothingRestored {
            path: path.to_path_buf(),
            registered: registered.len(),
            in_file: in_file.len(),
        });
    }

    // Pass two: everything agreed, so write. Loading one tensor at a
    // time keeps the transient copy to a single parameter rather than a
    // second full model.
    //
    // `written` accumulates what has actually landed, so a failure here
    // can say how far it got. Everything before this point could still
    // refuse without touching the map; from here on it cannot.
    let mut written: Vec<String> = Vec::with_capacity(restored.len());
    for (name, var) in registered.iter() {
        if !in_file.contains(name) {
            continue;
        }
        let tensor = file
            .load(name, var.device())
            .map_err(|e| RestoreError::Apply {
                path: path.to_path_buf(),
                stage: ApplyStage::Load,
                name: name.clone(),
                written: written.clone(),
                message: e.to_string(),
            })?;
        var.set(&tensor).map_err(|e| RestoreError::Apply {
            path: path.to_path_buf(),
            stage: ApplyStage::Set,
            name: name.clone(),
            written: written.clone(),
            message: e.to_string(),
        })?;
        written.push(name.clone());
    }

    let unused_from_file: Vec<String> = in_file
        .into_iter()
        .filter(|name| !registered.contains_key(name))
        .collect();

    Ok(RestoreReport {
        path: path.to_path_buf(),
        restored,
        absent_from_file,
        unused_from_file,
    })
}

/// Render at most a handful of names for an error message.
///
/// A vocabulary-sized mismatch can name every tensor in the model; the
/// first few plus a count is enough to identify which model was handed
/// over, and the full list is on the report for callers that want it.
fn preview(names: &[String]) -> String {
    const MAX: usize = 6;
    let head = names
        .iter()
        .take(MAX)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    match names.len().checked_sub(MAX) {
        Some(rest) if rest > 0 => format!("{head}, and {rest} more"),
        _ => head,
    }
}

/// The same truncation for the mismatch list, which needs shapes as
/// well as names to be readable.
fn preview_mismatches(mismatches: &[TensorMismatch]) -> String {
    const MAX: usize = 3;
    let head = mismatches
        .iter()
        .take(MAX)
        .map(TensorMismatch::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    match mismatches.len().checked_sub(MAX) {
        Some(rest) if rest > 0 => format!("{head}; and {rest} more"),
        _ => head,
    }
}

/// Say what a mid-apply failure had already done to the map.
///
/// Three cases, because the count alone does not settle it.
///
/// Failing on the first tensor while *reading* it leaves the map
/// untouched — the load builds a tensor and touches no `Var` — and
/// telling that caller to rebuild would send them chasing damage that
/// never happened.
///
/// Failing on the first tensor while *writing* it is not the same
/// thing. Pass one has already checked the shape and the dtype, which
/// makes the early returns in `Var::set` unreachable for anything it
/// verified, so a failure that survives to here comes out of the copy
/// itself, and candle does not say whether the destination was left
/// alone. The honest word is "may".
///
/// Failing later, with variables behind it, leaves a model that is part
/// one checkpoint and part another, which no amount of further training
/// makes coherent.
fn applied_state(stage: &ApplyStage, written: &[String]) -> String {
    match (stage, written.is_empty()) {
        (ApplyStage::Load, true) => {
            "no variable had been written, and reading the checkpoint touches none, so the map \
             still holds exactly what it did"
                .to_string()
        }
        (ApplyStage::Set, true) => {
            "no variable had been written in full, but this one was already being written and \
             candle does not report how far the copy got, so it may hold part of the checkpoint; \
             the map should be rebuilt rather than trained on"
                .to_string()
        }
        (_, false) => format!(
            "{} variable(s) had already been written ({}), so the map now mixes this checkpoint \
             with what it held before and should be rebuilt rather than trained on",
            written.len(),
            preview(written),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use tempfile::TempDir;

    fn small_varmap() -> VarMap {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        // Register two tiny parameters via a linear layer.
        let _ = candle_nn::linear(2, 2, vb.pp("ln")).unwrap();
        vm
    }

    #[test]
    fn save_step_creates_file_and_updates_list() {
        let tmp = TempDir::new().unwrap();
        let store = CheckpointStore::new(tmp.path(), "run", 3).unwrap();
        let vm = small_varmap();
        let path = store.save_step(&vm, 0).unwrap();
        assert!(path.exists());
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn rotation_keeps_only_the_last_n() {
        let tmp = TempDir::new().unwrap();
        let store = CheckpointStore::new(tmp.path(), "run", 2).unwrap();
        let vm = small_varmap();
        for step in [10, 20, 30, 40] {
            store.save_step(&vm, step).unwrap();
            // Give the mtimes a chance to differ on filesystems with
            // coarse-grained timestamps.
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 2, "keep=2 must retain exactly 2 files");
        // The two survivors must be the two most recent step numbers.
        let names: Vec<&str> = listed
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(names.iter().any(|n| n.contains("step30")));
        assert!(names.iter().any(|n| n.contains("step40")));
    }

    #[test]
    fn keep_clamps_to_at_least_one() {
        let tmp = TempDir::new().unwrap();
        let store = CheckpointStore::new(tmp.path(), "run", 0).unwrap();
        assert_eq!(store.keep(), 1);
        let vm = small_varmap();
        for step in 0..3 {
            store.save_step(&vm, step).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn save_final_writes_stable_filename_without_step() {
        let tmp = TempDir::new().unwrap();
        let store = CheckpointStore::new(tmp.path(), "run", 3).unwrap();
        let vm = small_varmap();
        let path = store.save_final(&vm).unwrap();
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "run.safetensors"
        );
        assert!(path.exists());
    }

    /// A map holding `ln` plus whatever extra layers the caller names.
    fn varmap_with(extra: &[(&str, usize, usize)]) -> VarMap {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let _ = candle_nn::linear(2, 2, vb.pp("ln")).unwrap();
        for (name, in_dim, out_dim) in extra {
            let _ = candle_nn::linear(*in_dim, *out_dim, vb.pp(*name)).unwrap();
        }
        vm
    }

    /// Every value in the map, flattened, keyed by name. Enough to tell
    /// "the variable now holds the checkpoint" from "the variable holds
    /// what it was initialised with".
    fn flat_values(vm: &VarMap) -> std::collections::BTreeMap<String, Vec<f32>> {
        vm.data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| {
                (
                    name.clone(),
                    var.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                )
            })
            .collect()
    }

    /// The whole point: after a strict restore the map holds the
    /// checkpoint's numbers, and the report says so name by name.
    #[test]
    fn restore_into_writes_every_variable_and_reports_them() {
        let tmp = TempDir::new().unwrap();
        let source = varmap_with(&[]);
        let path = tmp.path().join("src.safetensors");
        source.save(&path).unwrap();

        let target = varmap_with(&[]);
        assert_ne!(
            flat_values(&source),
            flat_values(&target),
            "two random inits collided; the restore below would prove nothing"
        );

        let report = restore_into(&target, &path).expect("complete restore");
        assert!(report.is_complete());
        assert_eq!(report.restored_count(), 2); // ln.weight + ln.bias
        assert_eq!(report.registered_count(), 2);
        assert_eq!(report.restored, vec!["ln.bias", "ln.weight"]);
        assert!(report.absent_from_file.is_empty());
        assert!(report.unused_from_file.is_empty());
        assert_eq!(flat_values(&source), flat_values(&target));
        assert!(report.summary().contains("2 of 2 variables restored"));
    }

    /// A variable the file does not carry would keep its initial value,
    /// which is a resume that silently did not happen.
    #[test]
    fn restore_into_refuses_when_a_variable_is_absent_from_the_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        varmap_with(&[]).save(&path).unwrap();

        let target = varmap_with(&[("head", 2, 3)]);
        let before = flat_values(&target);
        let err = restore_into(&target, &path).unwrap_err();
        match &err {
            RestoreError::Incomplete {
                registered, absent, ..
            } => {
                assert_eq!(*registered, 4);
                assert_eq!(absent, &vec!["head.bias".to_string(), "head.weight".into()]);
            }
            other => panic!("expected Incomplete, got {other:?}"),
        }
        assert!(err.to_string().contains("restore_into_partial"));
        assert_eq!(
            before,
            flat_values(&target),
            "a refused restore must leave the map exactly as it was"
        );
    }

    /// The permissive entry point accepts the same map, and still says
    /// which variables it skipped.
    #[test]
    fn restore_into_partial_accepts_the_gap_and_names_it() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        let source = varmap_with(&[]);
        source.save(&path).unwrap();

        let target = varmap_with(&[("head", 2, 3)]);
        let report = restore_into_partial(&target, &path).expect("partial restore");
        assert!(!report.is_complete());
        assert_eq!(report.restored, vec!["ln.bias", "ln.weight"]);
        assert_eq!(report.absent_from_file, vec!["head.bias", "head.weight"]);
        assert_eq!(report.registered_count(), 4);
        assert!(report.summary().contains("2 of 4 variables restored"));

        // The two that were present really did take the file's values.
        let after = flat_values(&target);
        let from_file = flat_values(&source);
        assert_eq!(after["ln.weight"], from_file["ln.weight"]);
        assert_eq!(after["ln.bias"], from_file["ln.bias"]);
    }

    /// A name that means a different tensor on each side is refused in
    /// either mode: the two maps are not the same model.
    #[test]
    fn restore_refuses_a_shape_disagreement_in_either_mode() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        varmap_with(&[("head", 2, 3)]).save(&path).unwrap();

        // Same names, different width on `head`.
        let target = varmap_with(&[("head", 2, 5)]);
        let before = flat_values(&target);
        for err in [
            restore_into(&target, &path).unwrap_err(),
            restore_into_partial(&target, &path).unwrap_err(),
        ] {
            match &err {
                RestoreError::Mismatch { mismatches, .. } => {
                    assert_eq!(mismatches.len(), 2); // head.weight + head.bias
                    let head = &mismatches[0];
                    assert_eq!(head.name, "head.bias");
                    assert_eq!(head.expected_shape, vec![5]);
                    assert_eq!(head.found_shape, vec![3]);
                    assert_eq!(head.expected_dtype, DType::F32);
                    assert_eq!(head.found_dtype, DType::F32);
                }
                other => panic!("expected Mismatch, got {other:?}"),
            }
            assert!(err.to_string().contains("head.bias"), "{err}");
        }
        assert_eq!(
            before,
            flat_values(&target),
            "a refused restore must leave the map exactly as it was"
        );
    }

    /// An empty map is the failure `VarMap::load` reports as success.
    #[test]
    fn restore_refuses_an_empty_varmap() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        varmap_with(&[]).save(&path).unwrap();

        let err = restore_into(&VarMap::new(), &path).unwrap_err();
        assert!(
            matches!(err, RestoreError::NoRegisteredVars { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("silently do nothing"));
    }

    /// A map that overlaps the file nowhere has not resumed anything,
    /// under either scope.
    #[test]
    fn restore_refuses_a_map_that_shares_no_name_with_the_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        varmap_with(&[]).save(&path).unwrap();

        let other = VarMap::new();
        let vb = VarBuilder::from_varmap(&other, DType::F32, &Device::Cpu);
        let _ = candle_nn::linear(2, 2, vb.pp("elsewhere")).unwrap();

        let err = restore_into_partial(&other, &path).unwrap_err();
        match &err {
            RestoreError::NothingRestored {
                registered,
                in_file,
                ..
            } => {
                assert_eq!(*registered, 2);
                assert_eq!(*in_file, 2);
            }
            other => panic!("expected NothingRestored, got {other:?}"),
        }
        assert!(err.to_string().contains("not two views of the same model"));
    }

    /// Tensors the map does not register are reported rather than
    /// rejected — legitimate, but the caller should be able to see it.
    #[test]
    fn restore_reports_tensors_the_map_does_not_register() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("src.safetensors");
        varmap_with(&[("head", 2, 3)]).save(&path).unwrap();

        let target = varmap_with(&[]);
        let report = restore_into(&target, &path).expect("the map's own names all matched");
        assert!(report.is_complete());
        assert_eq!(report.unused_from_file, vec!["head.bias", "head.weight"]);
        assert!(report.summary().contains("2 tensors in the file unused"));
    }

    /// A path with no checkpoint behind it names the file it could not
    /// read rather than surfacing a bare IO message.
    #[test]
    fn restore_refuses_a_path_that_is_not_a_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nowhere.safetensors");
        let err = restore_into(&varmap_with(&[]), &missing).unwrap_err();
        assert!(matches!(err, RestoreError::Open { .. }), "{err:?}");
        assert!(err.to_string().contains("nowhere.safetensors"));
    }

    /// Both halves of a mid-apply failure describe what the map holds,
    /// and they do not describe it the same way.
    #[test]
    fn apply_stage_text_distinguishes_a_read_from_a_half_written_copy() {
        let untouched = applied_state(&ApplyStage::Load, &[]);
        assert!(untouched.contains("still holds exactly what it did"));
        let maybe = applied_state(&ApplyStage::Set, &[]);
        assert!(maybe.contains("may hold part of the checkpoint"));
        let mixed = applied_state(&ApplyStage::Set, &["ln.weight".to_string()]);
        assert!(mixed.contains("1 variable(s) had already been written"));
        assert!(mixed.contains("ln.weight"));
    }

    /// The preview truncates rather than naming a whole model.
    #[test]
    fn preview_truncates_long_name_lists() {
        let names: Vec<String> = (0..9).map(|i| format!("var{i}")).collect();
        let text = preview(&names);
        assert!(text.contains("var0"));
        assert!(text.contains("and 3 more"), "{text}");
        assert!(!text.contains("var8"));

        let mismatches: Vec<TensorMismatch> = (0..5)
            .map(|i| TensorMismatch {
                name: format!("var{i}"),
                expected_shape: vec![2],
                found_shape: vec![3],
                expected_dtype: DType::F32,
                found_dtype: DType::F32,
            })
            .collect();
        let text = preview_mismatches(&mismatches);
        assert!(text.contains("and 2 more"), "{text}");
    }

    /// The restored values survive a device-local round trip through a
    /// tensor the test builds itself, so the assertion above is not
    /// comparing two copies of the same uninspected bytes.
    #[test]
    fn restore_writes_the_exact_values_the_file_holds() {
        let tmp = TempDir::new().unwrap();
        let source = VarMap::new();
        let vb = VarBuilder::from_varmap(&source, DType::F32, &Device::Cpu);
        let _ = vb.get((2, 2), "w").unwrap();
        let known = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        source.data().lock().unwrap()["w"].set(&known).unwrap();
        let path = tmp.path().join("src.safetensors");
        source.save(&path).unwrap();

        let target = VarMap::new();
        let vb = VarBuilder::from_varmap(&target, DType::F32, &Device::Cpu);
        let _ = vb.get((2, 2), "w").unwrap();
        restore_into(&target, &path).expect("restore");
        assert_eq!(flat_values(&target)["w"], vec![1.0, 2.0, 3.0, 4.0]);
    }
}
