use std::path::{Path, PathBuf};

// ─── Helpers ────────────────────────────────────────────────────

/// Recursively copy a directory tree (follows symlinks).
pub(crate) fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        // Use metadata() (follows symlinks) instead of file_type() (does not)
        let meta = entry.metadata()?;
        let dest_path = dst.join(entry.file_name());
        if meta.is_dir() {
            copy_dir(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

// ─── Path safety ────────────────────────────────────────────────

/// A path verified to reside within a base directory.
///
/// Constructed via [`ContainedPath::child`], which rejects path traversal
/// (`..`, absolute paths, symlink escapes). Once constructed, the inner path
/// is safe for filesystem operations within the base directory.
#[derive(Debug)]
pub(crate) struct ContainedPath(PathBuf);

impl ContainedPath {
    /// Resolve `name` as a child of `base`, rejecting traversal attempts.
    ///
    /// Validates that every component in `name` is [`Component::Normal`].
    /// If the resulting path already exists on disk, additionally verifies
    /// via `canonicalize` that symlinks do not escape `base`.
    pub(crate) fn child(base: &Path, name: &str) -> Result<Self, String> {
        for comp in Path::new(name).components() {
            if !matches!(comp, std::path::Component::Normal(_)) {
                return Err(format!(
                    "Invalid path component in '{name}': path traversal detected"
                ));
            }
        }
        let path = base.join(name);
        if path.exists() {
            let canonical = path
                .canonicalize()
                .map_err(|e| format!("Path resolution failed: {e}"))?;
            let base_canonical = base
                .canonicalize()
                .map_err(|e| format!("Base path resolution failed: {e}"))?;
            if !canonical.starts_with(&base_canonical) {
                return Err(format!("Path '{name}' escapes base directory"));
            }
        }
        Ok(Self(path))
    }
}

impl std::ops::Deref for ContainedPath {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for ContainedPath {
    fn as_ref(&self) -> &Path {
        self
    }
}
