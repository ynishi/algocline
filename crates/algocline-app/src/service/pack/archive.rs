//! Freezing a pack into one verifiable file.
//!
//! # Why a pack is an archive and not a directory
//!
//! A pack exists to be carried across time and machines. Between the moment it
//! is written and the moment it is restored there is usually a gap — long
//! enough that "is this still what I packed?" stops being answerable by
//! looking at it. A directory cannot answer that question: anything with
//! filesystem access can add, remove or truncate a file inside it, and nothing
//! about the result says so.
//!
//! A single `.tgz` plus its SHA-256 makes the snapshot checkable. The tar is
//! the atomic unit — one file, one mtime, one size — and the digest turns
//! "probably fine" into a comparison. `unpack` verifies before it expands, so
//! a truncated transfer or an edited payload is refused rather than restored
//! halfway.
//!
//! Compression is incidental here. The reason to reach for tar is the freeze,
//! not the bytes saved.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Failure modes for writing / reading a pack archive.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ArchiveError {
    #[error("failed to create archive at {path}: {source}")]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to append {dir} to archive: {source}")]
    Append {
        dir: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to finish archive at {path}: {source}")]
    Finish {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read archive at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to unpack archive {path}: {source}")]
    Unpack {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write checksum at {path}: {source}")]
    WriteChecksum {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "archive {path} does not match its checksum: recorded {expected}, \
         computed {actual} — the file changed after it was packed \
         (truncated transfer, edited payload, or the wrong .sha256 alongside it)"
    )]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error(
        "checksum file {path} is malformed: expected a 64-character hex digest, \
         got {found:?}"
    )]
    MalformedChecksum { path: PathBuf, found: String },
}

impl From<ArchiveError> for String {
    fn from(e: ArchiveError) -> Self {
        e.to_string()
    }
}

/// Sidecar path holding the archive's digest.
pub(crate) fn checksum_path(archive: &Path) -> PathBuf {
    let mut s = archive.as_os_str().to_os_string();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Whether a caller-supplied path names an archive rather than a directory.
///
/// Accepted spellings are `.tgz` and `.tar.gz`. Used by `unpack` so that a
/// directory produced before this became an archive still restores.
pub(crate) fn is_archive_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    name.ends_with(".tgz") || name.ends_with(".tar.gz")
}

/// Give `path` an archive extension if it does not already have one.
///
/// A caller that names a destination without a suffix gets `.tgz`; one that
/// spells it out keeps their spelling.
pub(crate) fn with_archive_extension(path: &Path) -> PathBuf {
    if is_archive_path(path) {
        path.to_path_buf()
    } else {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tgz");
        PathBuf::from(s)
    }
}

/// Compress `dir` into `archive` (gzip'd tar), then write `<archive>.sha256`.
///
/// The tar carries `dir`'s contents under a single top-level entry named after
/// the archive's stem, so expanding it produces one directory rather than
/// scattering `profile.toml` and `payload/` into the current directory.
///
/// Returns the digest and the archive's size in bytes.
pub(crate) fn write_archive(
    dir: &Path,
    archive: &Path,
    root_name: &str,
) -> Result<(String, u64), ArchiveError> {
    let file = std::fs::File::create(archive).map_err(|source| ArchiveError::Create {
        path: archive.to_path_buf(),
        source,
    })?;
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut builder = tar::Builder::new(encoder);

    builder
        .append_dir_all(root_name, dir)
        .map_err(|source| ArchiveError::Append {
            dir: dir.to_path_buf(),
            source,
        })?;

    // `into_inner` on both layers so the gzip trailer is flushed before the
    // digest is taken — hashing a half-written file would record a checksum
    // for something that never existed on disk.
    let encoder = builder
        .into_inner()
        .map_err(|source| ArchiveError::Finish {
            path: archive.to_path_buf(),
            source,
        })?;
    let file = encoder.finish().map_err(|source| ArchiveError::Finish {
        path: archive.to_path_buf(),
        source,
    })?;
    drop(file);

    let digest = digest_of(archive)?;
    let size = std::fs::metadata(archive)
        .map_err(|source| ArchiveError::Read {
            path: archive.to_path_buf(),
            source,
        })?
        .len();

    let sidecar = checksum_path(archive);
    let name = archive
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    // `sha256sum`-compatible so the check does not require algocline.
    std::fs::write(&sidecar, format!("{digest}  {name}\n")).map_err(|source| {
        ArchiveError::WriteChecksum {
            path: sidecar.clone(),
            source,
        }
    })?;

    Ok((digest, size))
}

/// SHA-256 of a file, streamed rather than buffered whole.
pub(crate) fn digest_of(path: &Path) -> Result<String, ArchiveError> {
    let mut file = std::fs::File::open(path).map_err(|source| ArchiveError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|source| ArchiveError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare an archive against its `.sha256` sidecar.
///
/// A missing sidecar is not an error: an operator may have carried only the
/// archive. Returns `Ok(None)` in that case so the caller can report that the
/// snapshot went unverified instead of implying it passed.
pub(crate) fn verify_archive(archive: &Path) -> Result<Option<String>, ArchiveError> {
    let sidecar = checksum_path(archive);
    if !sidecar.exists() {
        return Ok(None);
    }

    let recorded = std::fs::read_to_string(&sidecar).map_err(|source| ArchiveError::Read {
        path: sidecar.clone(),
        source,
    })?;
    let expected = recorded
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ArchiveError::MalformedChecksum {
            path: sidecar,
            found: recorded.trim().chars().take(80).collect(),
        });
    }

    let actual = digest_of(archive)?;
    if actual != expected {
        return Err(ArchiveError::ChecksumMismatch {
            path: archive.to_path_buf(),
            expected,
            actual,
        });
    }
    Ok(Some(actual))
}

/// Expand `archive` into `dest`, returning the single directory it contained.
///
/// The caller is expected to have verified the digest first; this only handles
/// the expansion.
pub(crate) fn extract_archive(archive: &Path, dest: &Path) -> Result<PathBuf, ArchiveError> {
    let file = std::fs::File::open(archive).map_err(|source| ArchiveError::Read {
        path: archive.to_path_buf(),
        source,
    })?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(dest).map_err(|source| ArchiveError::Unpack {
        path: archive.to_path_buf(),
        source,
    })?;

    // The archive carries one top-level directory (see `write_archive`). Find
    // it rather than assuming the name, so an archive rolled by hand with a
    // different stem still restores.
    let mut entries = std::fs::read_dir(dest)
        .map_err(|source| ArchiveError::Read {
            path: dest.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir());

    match entries.next() {
        Some(dir) => Ok(dir),
        // An archive whose root holds `profile.toml` directly rather than a
        // wrapping directory: restore from where it landed.
        None => Ok(dest.to_path_buf()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    fn fixture(root: &Path) {
        write(&root.join("profile.toml"), "[pack]\nformat_version = 1\n");
        write(&root.join("payload/core/config.toml"), "x = 1\n");
        write(&root.join("payload/cards/c.toml"), "y = 2\n");
    }

    #[test]
    fn round_trips_through_an_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");

        let (digest, size) = write_archive(&src, &archive, "snap").unwrap();
        assert_eq!(digest.len(), 64);
        assert!(size > 0);
        assert!(archive.exists());
        assert!(checksum_path(&archive).exists());

        let dest = tmp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        let restored = extract_archive(&archive, &dest).unwrap();

        assert_eq!(
            std::fs::read_to_string(restored.join("payload/core/config.toml")).unwrap(),
            "x = 1\n"
        );
        assert!(restored.join("profile.toml").exists());
    }

    #[test]
    fn verify_accepts_an_untouched_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        let (digest, _) = write_archive(&src, &archive, "snap").unwrap();

        assert_eq!(verify_archive(&archive).unwrap(), Some(digest));
    }

    /// The whole point of the freeze: an archive edited after packing is
    /// refused rather than restored.
    #[test]
    fn verify_refuses_an_edited_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        write_archive(&src, &archive, "snap").unwrap();

        let mut bytes = std::fs::read(&archive).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&archive, &bytes).unwrap();

        let err = verify_archive(&archive).unwrap_err();
        assert!(
            matches!(err, ArchiveError::ChecksumMismatch { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("changed after it was packed"));
    }

    /// A truncated transfer is the same failure, and must not read as success.
    #[test]
    fn verify_refuses_a_truncated_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        write_archive(&src, &archive, "snap").unwrap();

        let bytes = std::fs::read(&archive).unwrap();
        std::fs::write(&archive, &bytes[..bytes.len() / 2]).unwrap();

        assert!(matches!(
            verify_archive(&archive).unwrap_err(),
            ArchiveError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn verify_reports_absence_rather_than_passing() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        write_archive(&src, &archive, "snap").unwrap();
        std::fs::remove_file(checksum_path(&archive)).unwrap();

        assert_eq!(
            verify_archive(&archive).unwrap(),
            None,
            "a missing sidecar is 'unverified', not 'verified ok'"
        );
    }

    #[test]
    fn malformed_checksum_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        write_archive(&src, &archive, "snap").unwrap();
        std::fs::write(checksum_path(&archive), "not-a-digest\n").unwrap();

        assert!(matches!(
            verify_archive(&archive).unwrap_err(),
            ArchiveError::MalformedChecksum { .. }
        ));
    }

    #[test]
    fn sidecar_is_sha256sum_compatible() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        fixture(&src);
        let archive = tmp.path().join("snap.tgz");
        let (digest, _) = write_archive(&src, &archive, "snap").unwrap();

        let body = std::fs::read_to_string(checksum_path(&archive)).unwrap();
        assert_eq!(body, format!("{digest}  snap.tgz\n"));
    }

    #[test]
    fn archive_extension_is_added_only_when_missing() {
        assert_eq!(
            with_archive_extension(Path::new("/tmp/x")),
            PathBuf::from("/tmp/x.tgz")
        );
        assert_eq!(
            with_archive_extension(Path::new("/tmp/x.tgz")),
            PathBuf::from("/tmp/x.tgz")
        );
        assert_eq!(
            with_archive_extension(Path::new("/tmp/x.tar.gz")),
            PathBuf::from("/tmp/x.tar.gz")
        );
    }

    #[test]
    fn directory_paths_are_not_mistaken_for_archives() {
        assert!(!is_archive_path(Path::new("/tmp/snap")));
        assert!(is_archive_path(Path::new("/tmp/snap.tgz")));
        assert!(is_archive_path(Path::new("/tmp/snap.TAR.GZ")));
    }
}
