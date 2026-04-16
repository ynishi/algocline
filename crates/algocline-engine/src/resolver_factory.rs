//! Shared factory for `mlua_pkg::FsResolver` with algocline's sandbox policy.
//!
//! Three execution sites need the same resolver configuration:
//! - [`crate::executor::Executor`] — session and eval_simple VMs
//! - [`crate::variant_pkg`] — variant-scoped package submodules
//! - [`crate::bridge::fork`] — per-child fork VMs
//!
//! Prior to this factory, each site inlined the `SymlinkAwareSandbox` default
//! vs. `ALC_PKG_STRICT=1` strict-`FsResolver::new` choice independently, and
//! `bridge::fork` diverged by always using the strict path. Spinning up a
//! single `make_resolver` keeps the three in lock-step so behaviour is the
//! same regardless of where the VM was spawned.

use std::path::Path;

use mlua_pkg::{resolvers::FsResolver, sandbox::SymlinkAwareSandbox};

/// Build an `FsResolver` for `path` honouring algocline's sandbox policy.
///
/// # Policy
///
/// - Default: `SymlinkAwareSandbox`, so symlinks created by
///   `alc_pkg_link --scope=global` resolve to their real targets even when
///   those targets live outside `path`.
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
        SymlinkAwareSandbox::new(path)
            .ok()
            .map(FsResolver::with_sandbox)
    }
}
