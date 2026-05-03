//! Pool registry: persistent session-to-worker mapping.
//!
//! [`PoolRegistry`] tracks live pool worker processes in
//! `~/.algocline/state/pool/registry.json`.  The file survives MCP-process
//! death so a restarted MCP can rediscover live worker sockets.
//!
//! ## Crux invariant (Registry reconnect across restarts)
//!
//! `registry.json` is the **persistent source of truth**.  The in-memory
//! `PoolRegistry` value is a short-lived view — callers must reload from disk
//! after acquiring the advisory lock rather than caching across lock cycles.

use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::pool::PoolError;
use crate::service::lock::LockError;
use crate::service::manifest::now_iso8601;

// ─── Entry ────────────────────────────────────────────────────────────────────

/// A single session entry in the pool registry.
///
/// # Fields
///
/// - `sid` — session ID string (UUID or similar).
/// - `pid` — OS process-ID of the worker; used for liveness checks via
///   `kill -0`.
/// - `sock` — absolute path to the Unix-domain socket owned by the worker.
/// - `version` — crate version at the time the worker was spawned; used in
///   version-handshake validation.
/// - `created_at` — ISO 8601 timestamp of worker creation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoolSessionEntry {
    /// Session identifier.
    pub sid: String,
    /// Worker process ID (`u32` — never zero or negative on POSIX).
    pub pid: u32,
    /// Absolute path of the worker's Unix-domain socket.
    pub sock: PathBuf,
    /// Crate version that spawned the worker (for handshake validation).
    pub version: String,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
}

impl PoolSessionEntry {
    /// Create a new entry stamped with the current time.
    ///
    /// # Arguments
    ///
    /// * `sid` — session identifier.
    /// * `pid` — worker process ID.
    /// * `sock` — absolute path to the worker's UDS socket.
    /// * `version` — crate version string (e.g. `env!("CARGO_PKG_VERSION")`).
    ///
    /// # Returns
    ///
    /// A new `PoolSessionEntry` with `created_at` set to the current UTC time.
    pub fn new(
        sid: impl Into<String>,
        pid: u32,
        sock: PathBuf,
        version: impl Into<String>,
    ) -> Self {
        Self {
            sid: sid.into(),
            pid,
            sock,
            version: version.into(),
            created_at: now_iso8601(),
        }
    }
}

// ─── Registry ─────────────────────────────────────────────────────────────────

/// In-memory view of `registry.json`.
///
/// This struct must **always** be loaded from and saved to disk within a
/// single advisory-lock region (see [`with_registry_lock`]).  Do not hold
/// a `PoolRegistry` value across lock boundaries.
///
/// ## Crux: registry.json is the persistent source of truth
///
/// MCP processes must not rely on any in-memory state to discover live
/// workers after a restart.  Every mutation path must call [`save`] before
/// dropping the lock.
///
/// [`save`]: PoolRegistry::save
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PoolRegistry {
    /// All currently-registered worker sessions.
    pub sessions: Vec<PoolSessionEntry>,
}

impl PoolRegistry {
    /// Load `registry.json` from disk, returning an empty registry if the
    /// file does not exist.
    ///
    /// # Arguments
    ///
    /// * `path` — absolute path to `registry.json`.
    ///
    /// # Returns
    ///
    /// `Ok(PoolRegistry)` — either the parsed on-disk state or an empty
    /// registry when the file is absent.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::RegistryCorrupted(reason)` if the file exists but
    /// cannot be parsed as valid JSON.  **Never** falls back to an empty
    /// registry on parse failure — callers must handle the error explicitly
    /// and propagate it to the MCP wire layer.
    ///
    /// # Concurrency
    ///
    /// This is a synchronous file read.  The caller must hold the advisory
    /// `fs4::fs_std::FileExt::lock_exclusive` on `registry.lock` before
    /// calling this method to prevent concurrent read-modify-write races
    /// between multiple MCP processes.
    pub fn load_or_default(path: &Path) -> Result<Self, PoolError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path).map_err(|e| {
            PoolError::RegistryCorrupted(format!("failed to read {}: {e}", path.display()))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            PoolError::RegistryCorrupted(format!("failed to parse {}: {e}", path.display()))
        })
    }

    /// Atomically persist the registry to `registry.json` via
    /// `tempfile::NamedTempFile::persist` (POSIX `rename(2)`).
    ///
    /// # Arguments
    ///
    /// * `path` — absolute path to `registry.json`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::RegistryCorrupted` if parent-directory creation,
    /// serialization, temp-file creation/write/fsync, or the atomic rename
    /// fails.
    ///
    /// # Atomicity
    ///
    /// `NamedTempFile::persist` is atomic on modern Linux filesystems and
    /// macOS.  It is **not** guaranteed atomic on all platforms.
    ///
    /// # Concurrency
    ///
    /// Callers must hold the advisory `fs4::fs_std::FileExt::lock_exclusive`
    /// on `registry.lock` for the entire read-modify-write cycle
    /// (load → mutate → save) to prevent last-writer-wins data loss when
    /// multiple MCP processes write concurrently.
    pub fn save(&self, path: &Path) -> Result<(), PoolError> {
        let parent = path.parent().ok_or_else(|| {
            PoolError::RegistryCorrupted(format!(
                "registry path has no parent directory: {}",
                path.display()
            ))
        })?;

        std::fs::create_dir_all(parent).map_err(|e| {
            PoolError::RegistryCorrupted(format!(
                "failed to create registry directory {}: {e}",
                parent.display()
            ))
        })?;

        let content = serde_json::to_string_pretty(self).map_err(|e| {
            PoolError::RegistryCorrupted(format!("failed to serialize registry: {e}"))
        })?;

        let mut tmp = tempfile::NamedTempFile::new_in(parent).map_err(|e| {
            PoolError::RegistryCorrupted(format!(
                "failed to create temp file in {}: {e}",
                parent.display()
            ))
        })?;

        {
            use std::io::Write;
            tmp.write_all(content.as_bytes()).map_err(|e| {
                PoolError::RegistryCorrupted(format!("failed to write registry temp file: {e}"))
            })?;
            tmp.as_file().sync_all().map_err(|e| {
                PoolError::RegistryCorrupted(format!("failed to fsync registry temp file: {e}"))
            })?;
        }

        tmp.persist(path).map_err(|e| {
            PoolError::RegistryCorrupted(format!(
                "failed to atomically replace {} with temp file: {e}",
                path.display()
            ))
        })?;

        Ok(())
    }

    /// Scan all registered sessions and remove entries whose worker process
    /// is no longer alive, returning the surviving (live) entries.
    ///
    /// Liveness is tested with `kill(pid, 0)` (POSIX signal 0 — does not
    /// send a signal, only checks whether the process exists).  An `ESRCH`
    /// return value means the process does not exist and the entry is pruned.
    ///
    /// # Arguments
    ///
    /// None — operates on `&mut self` in place.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<PoolSessionEntry>)` — the subset of sessions that survived GC
    /// (i.e. whose worker process is still alive).
    ///
    /// # Errors
    ///
    /// Currently infallible on POSIX; the `Result` wrapper is kept for future
    /// extension.
    ///
    /// # Platform support
    ///
    /// On non-Unix targets the liveness check is omitted and all entries are
    /// assumed live (conservative).
    pub fn scan_and_gc(&mut self) -> Result<Vec<PoolSessionEntry>, PoolError> {
        let before_len = self.sessions.len();

        #[cfg(unix)]
        self.sessions.retain(|entry| {
            // Guard u32 → i32 (pid_t) conversion: pids above i32::MAX would
            // produce a negative value and send the signal to an unintended
            // process group (K-52).  Treat overflow as "dead" and prune.
            let pid_t = match i32::try_from(entry.pid) {
                Ok(p) => p,
                Err(_) => {
                    tracing::warn!(
                        pid = entry.pid,
                        sid = %entry.sid,
                        "pid exceeds i32::MAX, treating as dead (K-52)"
                    );
                    return false;
                }
            };
            // SAFETY: libc::kill(pid, sig) is a thin syscall wrapper.
            // pid > 0 sends to the specific process (never to a group).
            // sig == 0 performs an existence check without delivering a signal.
            // pid fits in i32, verified by try_from above.
            // Return 0 → process exists (live).
            // Return -1 with errno ESRCH → no such process (dead / orphan).
            let result = unsafe { libc::kill(pid_t, 0) };
            result == 0
        });

        // On non-Unix platforms: retain all entries (conservative fallback).
        #[cfg(not(unix))]
        let _ = before_len;

        let _ = before_len; // suppress unused warning on non-unix
        Ok(self.sessions.clone())
    }

    /// Add a session entry to the registry.
    ///
    /// # Arguments
    ///
    /// * `entry` — the `PoolSessionEntry` to insert.
    ///
    /// # Notes
    ///
    /// Does not persist to disk.  Call [`save`](PoolRegistry::save) after
    /// mutating to ensure durability.
    pub fn add(&mut self, entry: PoolSessionEntry) {
        self.sessions.push(entry);
    }

    /// Remove the entry with the given session ID.
    ///
    /// # Arguments
    ///
    /// * `sid` — session ID to remove.
    ///
    /// # Returns
    ///
    /// `true` if an entry was found and removed, `false` if no matching entry
    /// existed.
    ///
    /// # Notes
    ///
    /// Does not persist to disk.  Call [`save`](PoolRegistry::save) after
    /// mutating to ensure durability.
    pub fn remove(&mut self, sid: &str) -> bool {
        let before = self.sessions.len();
        self.sessions.retain(|e| e.sid != sid);
        self.sessions.len() < before
    }

    /// Look up a session entry by ID.
    ///
    /// # Arguments
    ///
    /// * `sid` — session ID to search for.
    ///
    /// # Returns
    ///
    /// `Some(&PoolSessionEntry)` if found, `None` otherwise.
    pub fn find(&self, sid: &str) -> Option<&PoolSessionEntry> {
        self.sessions.iter().find(|e| e.sid == sid)
    }
}

// ─── Advisory-lock helper ─────────────────────────────────────────────────────

/// Run `f` while holding an exclusive advisory lock on `lock_path`, using the
/// same `fs4`-backed mechanism as `service::lock::with_exclusive_lock`.
///
/// Callers should pass the `registry.lock` sentinel path (e.g.
/// `app_dir.root().join("pool/registry.lock")`).
///
/// # Arguments
///
/// * `lock_path` — path to the advisory lock file (created if absent).
/// * `f` — closure to run under the lock.
///
/// # Returns
///
/// Propagates the return value of `f`.
///
/// # Errors
///
/// Returns `PoolError::RegistryCorrupted` if the lock file cannot be created
/// or the exclusive lock cannot be acquired.
///
/// # Concurrency
///
/// The lock is released when the underlying `File` is dropped, which occurs
/// on all exit paths from this function including panics (RAII / drop).
pub fn with_registry_lock<F, R>(lock_path: &Path, f: F) -> Result<R, PoolError>
where
    F: FnOnce() -> Result<R, PoolError>,
{
    crate::service::lock::with_exclusive_lock(lock_path, f)
}

/// Bridge so that `lock::with_exclusive_lock` generic `E: From<LockError>`
/// constraint is satisfied when `E = PoolError`.
impl From<LockError> for PoolError {
    fn from(e: LockError) -> Self {
        PoolError::RegistryCorrupted(e.to_string())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_entry(sid: &str, pid: u32) -> PoolSessionEntry {
        PoolSessionEntry::new(
            sid,
            pid,
            PathBuf::from(format!("/tmp/alc-pool/{sid}.sock")),
            "0.30.0",
        )
    }

    // ── T1: happy path ────────────────────────────────────────────────────────

    /// T1 — load_or_default returns empty registry when the file is absent.
    #[test]
    fn load_default_when_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("registry.json");

        let reg = PoolRegistry::load_or_default(&path).expect("load_or_default");
        assert!(reg.sessions.is_empty(), "expected empty registry");
    }

    // ── T2: boundary / edge ───────────────────────────────────────────────────

    /// T2 — scan_and_gc removes the dead-PID entry and retains the live one.
    ///
    /// Uses the current process PID as the "live" entry (guaranteed to exist)
    /// and pid=999999 as the "dead" entry (virtually certain to be absent).
    #[test]
    fn scan_and_gc_removes_dead_pid() {
        // SAFETY: std::process::id() returns the current PID, which is alive.
        let live_pid = std::process::id();

        let mut reg = PoolRegistry::default();
        reg.add(make_entry("live-session", live_pid));
        reg.add(make_entry("dead-session", 999_999));

        let survivors = reg.scan_and_gc().expect("scan_and_gc");

        assert_eq!(survivors.len(), 1, "expected 1 survivor");
        assert_eq!(survivors[0].sid, "live-session");
        assert_eq!(
            reg.sessions.len(),
            1,
            "in-place mutation must prune dead entry"
        );
        assert!(
            reg.find("dead-session").is_none(),
            "dead entry must be gone"
        );
        assert!(reg.find("live-session").is_some(), "live entry must remain");
    }

    // ── T3: error path ────────────────────────────────────────────────────────

    /// T3 — load_or_default returns PoolError::RegistryCorrupted for bad JSON.
    ///
    /// Verifies that parse failures are NOT silently swallowed as empty
    /// registries — CLAUDE.md 2026-04-22 事故と同じパターンの再発防止。
    #[test]
    fn load_corrupted_returns_pool_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("registry.json");

        // Write intentionally broken JSON.
        std::fs::write(&path, b"{ not valid json !!!").expect("write");

        let result = PoolRegistry::load_or_default(&path);
        match result {
            Err(PoolError::RegistryCorrupted(msg)) => {
                assert!(!msg.is_empty(), "error message must not be empty");
            }
            other => panic!("expected RegistryCorrupted, got {other:?}"),
        }
    }

    // ── T4: concurrent writers (advisory lock prevents entry loss) ────────────

    /// T4 — two concurrent tasks each perform 50 add→save cycles under the
    /// advisory lock; the final registry must contain all entries.
    ///
    /// Uses `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` as
    /// required by the concurrency-analysis spec.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_writers_no_entry_loss() {
        let dir = Arc::new(tempfile::tempdir().expect("tempdir"));
        let reg_path = Arc::new(dir.path().join("registry.json"));
        let lock_path = Arc::new(dir.path().join("registry.lock"));

        let n_per_task: u32 = 50;
        let n_tasks: u32 = 2;

        let mut handles = Vec::new();
        for task_id in 0..n_tasks {
            let reg_path = Arc::clone(&reg_path);
            let lock_path = Arc::clone(&lock_path);

            let handle = tokio::task::spawn_blocking(move || {
                for i in 0..n_per_task {
                    let sid = format!("t{task_id}-s{i}");
                    // SAFETY: std::process::id() is the live PID of this process.
                    let entry = make_entry(&sid, std::process::id());

                    with_registry_lock(&lock_path, || {
                        let mut reg = PoolRegistry::load_or_default(&reg_path)?;
                        reg.add(entry);
                        reg.save(&reg_path)
                    })
                    .expect("lock + save must not fail");
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.await.expect("task did not panic");
        }

        // Final verification: load without lock (no concurrent writers left).
        let final_reg = PoolRegistry::load_or_default(&reg_path).expect("final load_or_default");
        let expected = (n_per_task * n_tasks) as usize;
        assert_eq!(
            final_reg.sessions.len(),
            expected,
            "all {expected} entries must be present (no last-writer-wins loss)"
        );
    }
}
