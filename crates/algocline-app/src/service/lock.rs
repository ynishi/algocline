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
pub(crate) fn with_exclusive_lock<F, R>(lock_path: &Path, f: F) -> Result<R, String>
where
    F: FnOnce() -> Result<R, String>,
{
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create lock dir {}: {e}", parent.display()))?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|e| format!("Failed to open lock file {}: {e}", lock_path.display()))?;
    FileExt::lock_exclusive(&file)
        .map_err(|e| format!("Failed to acquire lock on {}: {e}", lock_path.display()))?;

    let result = f();
    // Ordering insurance for the success path — the drop impl would run
    // regardless when the function returns or unwinds, but making it
    // explicit documents that the lock is tied to this lexical scope.
    drop(file);
    result
}
