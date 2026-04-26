//! Advisory cross-process file locks.
//!
//! Wraps `fs4::fs_std::FileExt::lock_exclusive` so callers can serialize
//! load → modify → save sequences on a shared file without inlining the
//! open/lock/drop boilerplate.
//!
//! Two call sites use this today:
//! - `manifest::with_manifest_lock` — guards `~/.algocline/installed.json`
//!   against concurrent `pkg_install` writes.
//! - `pkg::install::with_project_files_lock` — guards the
//!   `alc.toml` / `alc.lock` load→modify→save in
//!   `update_project_files_for_install` against concurrent installs targeting
//!   the same project root.
//!
//! Locks are advisory `flock(2)` on Unix and `LockFileEx` on Windows (via
//! fs4). They serialize only processes that also call this function —
//! external editors writing `installed.json` by hand are not blocked.

use std::path::Path;

use fs4::fs_std::FileExt;

/// Run `f` while holding an exclusive advisory lock on `lock_path`.
///
/// The file at `lock_path` is created if it does not already exist. The
/// lock is released when the underlying `File` is dropped, which happens
/// on every exit from this function — including panic in `f`, since stack
/// unwinding runs the `File` destructor and that destructor calls
/// `close(2)`, which releases the advisory `flock`.
///
/// The error type `E` must implement `From<LockError>` so that lock
/// acquisition failures are injected into the same typed error channel as
/// the closure's own failures. Wire boundaries that still need `String`
/// errors can use `E = String` with `LockError: Into<String>` satisfied by
/// the blanket `impl From<LockError> for String` in [`LockError`].
pub(crate) fn with_exclusive_lock<F, R, E>(lock_path: &Path, f: F) -> Result<R, E>
where
    F: FnOnce() -> Result<R, E>,
    E: From<LockError>,
{
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| LockError::CreateDir {
                path: parent.to_path_buf(),
                source: e,
            })
            .map_err(E::from)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|e| LockError::Open {
            path: lock_path.to_path_buf(),
            source: e,
        })
        .map_err(E::from)?;
    FileExt::lock_exclusive(&file)
        .map_err(|e| LockError::Acquire {
            path: lock_path.to_path_buf(),
            source: e,
        })
        .map_err(E::from)?;

    let result = f();
    // Ordering insurance for the success path — the drop impl would run
    // regardless when the function returns or unwinds, but making it
    // explicit documents that the lock is tied to this lexical scope.
    drop(file);
    result
}

/// Errors that can occur when acquiring or operating the advisory file lock.
///
/// Callers that still surface `String` errors can derive `From<LockError> for String`
/// via the blanket impl below. Callers with typed service errors can absorb
/// `LockError` through a `#[from]`-annotated variant.
#[derive(Debug, thiserror::Error)]
pub(crate) enum LockError {
    /// Failed to create the lock file's parent directory.
    #[error("failed to create lock dir {}: {source}", path.display())]
    CreateDir {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Failed to open (or create) the lock file.
    #[error("failed to open lock file {}: {source}", path.display())]
    Open {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Failed to acquire the exclusive lock.
    #[error("failed to acquire lock on {}: {source}", path.display())]
    Acquire {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl From<LockError> for String {
    fn from(e: LockError) -> Self {
        e.to_string()
    }
}
