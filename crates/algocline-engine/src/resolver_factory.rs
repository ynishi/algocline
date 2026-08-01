//! Shared factory for `mlua_pkg::FsResolver` with algocline's sandbox policy.
//!
//! Three execution sites need the same resolver configuration:
//! - [`crate::executor::Executor`] — session and eval_simple VMs
//! - [`crate::variant_pkg`] — variant-scoped package submodules
//! - [`crate::bridge::fork`] — per-child fork VMs
//!
//! Prior to this factory, each site inlined the symlink-aware default vs.
//! `ALC_PKG_STRICT=1` strict-`FsResolver::new` choice independently, and
//! `bridge::fork` diverged by always using the strict path. Spinning up a
//! single `make_resolver` keeps the three in lock-step so behaviour is the
//! same regardless of where the VM was spawned.

use std::path::{Path, PathBuf};

use mlua_pkg::{
    resolvers::FsResolver,
    sandbox::{FileContent, InitError, ReadError, SandboxedFs},
};

/// Symlink-aware sandbox that re-scans `root` for symlinks on every read.
///
/// Upstream `mlua_pkg::sandbox::SymlinkAwareSandbox` snapshots the canonical
/// targets of symlinks found under `root` once at construction time. That
/// snapshot goes stale on long-lived VMs: the shared `eval_simple` VM lives
/// for the whole MCP server process, so packages linked with
/// `alc_pkg_link --scope=global` *after* server startup were rejected with
/// `path traversal blocked` even though the symlink is sitting in the
/// packages dir (issue bda23871 — broke `alc_eval` / `alc_advice` /
/// `alc_pkg_list` meta loads until a server restart).
///
/// This implementation keeps the same boundary semantics — a canonical path
/// is readable iff it lives under `root` or under the target of a symlink
/// currently present directly under `root` — but resolves the symlink set at
/// read time. The packages dir holds O(100) entries and `require` results
/// are cached in `package.loaded`, so the extra `read_dir` per module load
/// is negligible.
struct RescanSymlinkSandbox {
    /// Canonicalized sandbox root.
    root: PathBuf,
}

impl RescanSymlinkSandbox {
    fn new(root: impl Into<PathBuf>) -> Result<Self, InitError> {
        let raw = root.into();
        let canonical = match raw.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(InitError::RootNotFound { path: raw });
            }
            Err(e) => {
                return Err(InitError::Io {
                    path: raw,
                    source: e,
                });
            }
        };
        Ok(Self { root: canonical })
    }
}

impl SandboxedFs for RescanSymlinkSandbox {
    fn read(&self, relative: &Path) -> Result<Option<FileContent>, ReadError> {
        let path = self.root.join(relative);
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(ReadError::Io { path, source: e }),
        };

        // Normal case: resolved path stayed under root (no symlink escape).
        if canonical.starts_with(&self.root) {
            return read_file(&canonical);
        }

        // The canonical path escaped root, which is legitimate exactly when it
        // was reached through a symlink currently present under root. Scan the
        // live directory state instead of a construction-time snapshot so
        // links created after VM startup are honoured (and removed links are
        // revoked).
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let is_symlink = entry
                    .path()
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false);
                if !is_symlink {
                    continue;
                }
                if let Ok(target) = entry.path().canonicalize() {
                    if canonical.starts_with(&target) {
                        return read_file(&canonical);
                    }
                }
            }
        }

        Err(ReadError::Traversal {
            attempted: canonical,
        })
    }
}

/// Shared file reading logic (mirrors upstream sandbox behaviour:
/// `NotFound` is represented as `Ok(None)`, not an error).
fn read_file(canonical: &Path) -> Result<Option<FileContent>, ReadError> {
    match std::fs::read_to_string(canonical) {
        Ok(content) => Ok(Some(FileContent {
            content,
            resolved_path: canonical.to_path_buf(),
        })),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(ReadError::Io {
            path: canonical.to_path_buf(),
            source: e,
        }),
    }
}

/// Build an `FsResolver` for `path` honouring algocline's sandbox policy.
///
/// # Policy
///
/// - Default: [`RescanSymlinkSandbox`], so symlinks created by
///   `alc_pkg_link --scope=global` resolve to their real targets even when
///   those targets live outside `path`, including links created **after**
///   the VM (and its resolver) was constructed.
/// - `ALC_PKG_STRICT=1` (or `true`, case-insensitive): plain `FsResolver::new`,
///   which rejects symlinks pointing outside `path`. Useful for hermetic
///   builds and regression tests where symlink escape should be a hard error.
///
/// Returns `None` when the resolver cannot be constructed (typically because
/// `path` does not exist or is not a directory).
pub(crate) fn make_resolver(path: &Path) -> Option<FsResolver> {
    let strict = std::env::var("ALC_PKG_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if strict {
        FsResolver::new(path).ok()
    } else {
        RescanSymlinkSandbox::new(path)
            .ok()
            .map(FsResolver::with_sandbox)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Regular file under root reads fine.
    #[test]
    fn regular_file_under_root_reads() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("foo.lua"), "return 1").unwrap();

        let sandbox = RescanSymlinkSandbox::new(tmp.path()).unwrap();
        let got = sandbox.read(Path::new("foo.lua")).unwrap().unwrap();
        assert_eq!(got.content, "return 1");
    }

    /// Missing file is `Ok(None)`, mirroring upstream sandbox semantics.
    #[test]
    fn missing_file_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = RescanSymlinkSandbox::new(tmp.path()).unwrap();
        assert!(sandbox.read(Path::new("nope.lua")).unwrap().is_none());
    }

    /// Escaping root without a symlink stays blocked.
    #[test]
    fn escape_without_symlink_is_blocked() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(tmp.path().join("evil.lua"), "return 666").unwrap();

        let sandbox = RescanSymlinkSandbox::new(&root).unwrap();
        assert_traversal(sandbox.read(Path::new("../evil.lua")));
    }

    /// `FileContent` has no `Debug` impl, so `unwrap_err()` is unavailable —
    /// assert the Traversal variant via match instead.
    fn assert_traversal(result: Result<Option<FileContent>, ReadError>) {
        match result {
            Err(ReadError::Traversal { .. }) => {}
            Err(other) => panic!("expected Traversal, got: {other:?}"),
            Ok(_) => panic!("expected Traversal, got Ok"),
        }
    }

    /// Core regression for issue bda23871: a symlink created **after** the
    /// sandbox was constructed must be readable (upstream
    /// `SymlinkAwareSandbox` snapshots targets at construction time and
    /// rejects this).
    #[cfg(unix)]
    #[test]
    fn symlink_created_after_construction_is_readable() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("packages");
        fs::create_dir_all(&root).unwrap();

        // Sandbox constructed while the packages dir is still empty.
        let sandbox = RescanSymlinkSandbox::new(&root).unwrap();

        // Package source lives outside root; linked in afterwards.
        let external = tmp.path().join("worktree").join("late_pkg");
        fs::create_dir_all(&external).unwrap();
        fs::write(external.join("init.lua"), "return { value = 42 }").unwrap();
        std::os::unix::fs::symlink(&external, root.join("late_pkg")).unwrap();

        let got = sandbox
            .read(Path::new("late_pkg/init.lua"))
            .unwrap()
            .unwrap();
        assert_eq!(got.content, "return { value = 42 }");
    }

    /// Removing the symlink revokes access on the next read (`Ok(None)`
    /// because the joined path no longer resolves).
    #[cfg(unix)]
    #[test]
    fn removed_symlink_revokes_access() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("packages");
        fs::create_dir_all(&root).unwrap();
        let external = tmp.path().join("src_pkg");
        fs::create_dir_all(&external).unwrap();
        fs::write(external.join("init.lua"), "return 1").unwrap();
        std::os::unix::fs::symlink(&external, root.join("src_pkg")).unwrap();

        let sandbox = RescanSymlinkSandbox::new(&root).unwrap();
        assert!(sandbox
            .read(Path::new("src_pkg/init.lua"))
            .unwrap()
            .is_some());

        fs::remove_file(root.join("src_pkg")).unwrap();
        assert!(sandbox
            .read(Path::new("src_pkg/init.lua"))
            .unwrap()
            .is_none());
    }

    /// A file reachable only through a symlink that belongs to a *different*
    /// directory (not under root) is still blocked: only symlinks directly
    /// under root whitelist their targets.
    #[cfg(unix)]
    #[test]
    fn unrelated_external_path_stays_blocked() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("packages");
        fs::create_dir_all(&root).unwrap();

        // One legitimate linked pkg.
        let linked = tmp.path().join("linked_pkg");
        fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(&linked, root.join("linked_pkg")).unwrap();

        // An unrelated external file, reached via `..` escape.
        fs::write(tmp.path().join("evil.lua"), "return 666").unwrap();

        let sandbox = RescanSymlinkSandbox::new(&root).unwrap();
        assert_traversal(sandbox.read(Path::new("../evil.lua")));
    }
}
