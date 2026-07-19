//! Rotating safetensors checkpoint writer.
//!
//! During Full FT the trainer writes an intermediate checkpoint every
//! `ckpt_every` steps and keeps only the most recent `ckpt_keep`
//! files on disk. The older files are dropped by modification time
//! rather than by step number so a manual `touch` cannot hide the
//! trainer's own bookkeeping from `ls -t`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
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
}
