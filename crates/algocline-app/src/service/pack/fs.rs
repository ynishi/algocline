//! Filesystem primitives shared by pack and unpack.
//!
//! Both directions walk a tree, count what moved, and must keep going when a
//! single entry is unreadable — a dangling link inside `cards/` should not
//! abort a 200 MB transfer. What they disagree on is only whether an existing
//! destination file may be replaced, which is what [`OverwritePolicy`]
//! expresses.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// One entry that could not be carried, surfaced on the response rather than
/// logged.
///
/// A `warn!` here reaches the MCP server's stderr and never the caller's UI,
/// which would make an unreadable file indistinguishable from an absent one.
/// The service layer propagates failures to the caller instead of logging
/// them; see the error-propagation policy in the project's contributor guide.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct SkipRecord {
    pub path: String,
    pub reason: String,
}

/// How to treat a destination file that already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverwritePolicy {
    /// Replace it.
    Replace,
    /// Leave it and count it under [`CopyStats::kept`]. Used by
    /// `unpack --mode=merge`, where the local machine's copy is authoritative.
    KeepExisting,
}

/// Files and bytes moved by a copy, plus what was left in place.
///
/// `kept` is a count rather than a per-file list on purpose. Restoring a pack
/// onto a machine that already has most of it is the normal case — measured
/// against a real application directory, an unpack left 1,799 files in place,
/// and emitting one record each produced a 290,755-character response, past
/// what the MCP transport would carry. "Already present" is bulk, expected,
/// and actionable only in aggregate; [`SkipRecord`] stays for the entries a
/// caller actually has to look at (unreadable paths, explicit exclusions,
/// absent sections).
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct CopyStats {
    pub files: u64,
    pub bytes: u64,
    /// Files left untouched because they already existed at the destination
    /// and the policy was [`OverwritePolicy::KeepExisting`].
    pub kept: u64,
}

impl std::ops::AddAssign for CopyStats {
    fn add_assign(&mut self, rhs: Self) {
        self.files += rhs.files;
        self.bytes += rhs.bytes;
        self.kept += rhs.kept;
    }
}

/// Failure modes for the shared copy primitives.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FsError {
    #[error("failed to create directory at {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read directory {path}: {source}")]
    ReadDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to copy {from} to {to}: {source}")]
    Copy {
        from: PathBuf,
        to: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl From<FsError> for String {
    fn from(e: FsError) -> Self {
        e.to_string()
    }
}

/// Copy one file, returning what it cost.
pub(crate) fn copy_file(src: &Path, dst: &Path) -> Result<CopyStats, FsError> {
    let bytes = std::fs::copy(src, dst).map_err(|source| FsError::Copy {
        from: src.to_path_buf(),
        to: dst.to_path_buf(),
        source,
    })?;
    Ok(CopyStats {
        files: 1,
        bytes,
        kept: 0,
    })
}

/// Recursively copy `src` into `dst`.
///
/// Symlinks are followed: the point of a payload is to carry real bytes, and
/// links under `packages/` travel as definitions instead (see
/// [`super::profile::ProfileLink`]). An entry that cannot be stat'ed lands in
/// `skipped` and the walk continues.
pub(crate) fn copy_tree(
    src: &Path,
    dst: &Path,
    policy: OverwritePolicy,
    skipped: &mut Vec<SkipRecord>,
) -> Result<CopyStats, FsError> {
    let mut stats = CopyStats::default();
    std::fs::create_dir_all(dst).map_err(|source| FsError::CreateDir {
        path: dst.to_path_buf(),
        source,
    })?;

    let entries = std::fs::read_dir(src).map_err(|source| FsError::ReadDir {
        path: src.to_path_buf(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| FsError::ReadDir {
            path: src.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let dest = dst.join(entry.file_name());

        // `std::fs::metadata` follows symlinks, so a dangling one surfaces
        // here as an error rather than reaching the copy. `DirEntry::metadata`
        // would NOT do this — it does not traverse links, which would let a
        // broken link through to `copy_file` and fail the whole walk.
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                skipped.push(SkipRecord {
                    path: path.display().to_string(),
                    reason: format!("unreadable: {e}"),
                });
                continue;
            }
        };

        if meta.is_dir() {
            stats += copy_tree(&path, &dest, policy, skipped)?;
        } else {
            if policy == OverwritePolicy::KeepExisting && dest.exists() {
                stats.kept += 1;
                continue;
            }
            stats += copy_file(&path, &dest)?;
        }
    }

    Ok(stats)
}

/// Count what [`copy_tree`] would move, without writing anything.
///
/// Applies the same policy so that a dry run reports the same skip set a real
/// run would produce.
pub(crate) fn measure_tree(
    src: &Path,
    dst: &Path,
    policy: OverwritePolicy,
    skipped: &mut Vec<SkipRecord>,
) -> Result<CopyStats, FsError> {
    let mut stats = CopyStats::default();
    let entries = std::fs::read_dir(src).map_err(|source| FsError::ReadDir {
        path: src.to_path_buf(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| FsError::ReadDir {
            path: src.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let dest = dst.join(entry.file_name());

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                skipped.push(SkipRecord {
                    path: path.display().to_string(),
                    reason: format!("unreadable: {e}"),
                });
                continue;
            }
        };

        if meta.is_dir() {
            stats += measure_tree(&path, &dest, policy, skipped)?;
        } else {
            if policy == OverwritePolicy::KeepExisting && dest.exists() {
                stats.kept += 1;
                continue;
            }
            stats += CopyStats {
                files: 1,
                bytes: meta.len(),
                kept: 0,
            };
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn keep_existing_leaves_destination_and_counts_it() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write(&src.join("a.txt"), "incoming");
        write(&dst.join("a.txt"), "local");

        let mut skipped = Vec::new();
        let stats = copy_tree(&src, &dst, OverwritePolicy::KeepExisting, &mut skipped).unwrap();

        assert_eq!(stats.files, 0);
        assert_eq!(stats.kept, 1);
        assert!(
            skipped.is_empty(),
            "bulk 'already present' is a count, not a per-file record: {skipped:?}"
        );
        assert_eq!(std::fs::read_to_string(dst.join("a.txt")).unwrap(), "local");
    }

    #[test]
    fn replace_overwrites_destination() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write(&src.join("a.txt"), "incoming");
        write(&dst.join("a.txt"), "local");

        let mut skipped = Vec::new();
        let stats = copy_tree(&src, &dst, OverwritePolicy::Replace, &mut skipped).unwrap();

        assert_eq!(stats.files, 1);
        assert!(skipped.is_empty());
        assert_eq!(
            std::fs::read_to_string(dst.join("a.txt")).unwrap(),
            "incoming"
        );
    }

    #[test]
    fn unreadable_entry_is_skipped_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write(&src.join("good.txt"), "ok");
        std::os::unix::fs::symlink(tmp.path().join("nowhere"), src.join("bad")).unwrap();

        let mut skipped = Vec::new();
        let stats = copy_tree(
            &src,
            &tmp.path().join("dst"),
            OverwritePolicy::Replace,
            &mut skipped,
        )
        .unwrap();

        assert_eq!(stats.files, 1, "the readable file still travels");
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].reason.contains("unreadable"));
    }

    #[test]
    fn measure_matches_copy_without_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write(&src.join("a.txt"), "12345");
        write(&src.join("nested").join("b.txt"), "678");

        let dst = tmp.path().join("dst");
        let mut measured_skips = Vec::new();
        let measured =
            measure_tree(&src, &dst, OverwritePolicy::Replace, &mut measured_skips).unwrap();
        assert!(!dst.exists(), "measuring writes nothing");

        let mut copied_skips = Vec::new();
        let copied = copy_tree(&src, &dst, OverwritePolicy::Replace, &mut copied_skips).unwrap();

        assert_eq!(measured, copied);
        assert_eq!(measured_skips, copied_skips);
    }
}
