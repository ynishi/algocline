//! Package management — install, list, remove, and project-local resolution.
//!
//! # Package resolution model
//!
//! algocline resolves `require("name")` through an ordered chain of
//! `mlua_isle::FsResolver`s. When `alc_run` / `alc_advice` starts a session,
//! the executor registers resolvers in the following priority (high → low):
//!
//! 1. **Project-local** (`alc.toml` path entries, resolved via `alc.lock`) —
//!    zero-copy references to on-disk directories. Resolved via
//!    [`super::lockfile::resolve_local_dir_paths`].
//! 2. **`ALC_PACKAGES_PATH`** — colon-separated search paths from the env var.
//! 3. **Global default** (`~/.algocline/packages/`) — packages installed by
//!    `alc_pkg_install`.
//!
//! A higher-priority package **shadows** a lower-priority one with the same
//! name. `pkg_list` reports both, marking the effective one as `active: true`.
//!
//! # Scope concept
//!
//! Each package belongs to a scope:
//!
//! - **`"global"`** — installed in `~/.algocline/packages/` via `pkg_install`.
//!   Physical directory managed by algocline (clone/copy).
//! - **`"project"`** — declared in `alc.toml` via `pkg_install` or manual entry.
//!   Recorded in `alc.lock`. algocline never copies or deletes the source files.
//!
//! # `alc.toml` / `alc.lock` lifecycle
//!
//! `alc.toml` is the package declaration file (user-authored).
//! `alc.lock` is the resolved lockfile (tool-managed).
//!
//! - `pkg_install` updates both `alc.toml` (adds entry) and `alc.lock` (records version/source).
//! - `pkg_remove` removes the entry from both `alc.toml` and `alc.lock`.
//! - Physical files in `~/.algocline/packages/` are never deleted by `pkg_remove`.
//!
//! `alc.lock` is updated atomically via temp-file + rename.
//! Read at session start to build extra `FsResolver` entries.
//!
//! # Project root resolution
//!
//! See [`super::project`] for the 3-tier priority (explicit arg →
//! `ALC_PROJECT_ROOT` env → ancestor walk from cwd looking for `alc.toml`).
//!
//! # Security invariants
//!
//! - Package names are whitelist-validated (`[a-zA-Z0-9_-]`) before Lua
//!   interpolation in `pkg_list` meta evaluation.
//! - Lockfile writes are serialized in-process (`SAVE_GUARD` mutex)
//!   and use atomic rename to prevent half-written reads.
//!
//! # Submodules
//!
//! - [`list`] — `pkg_list` and its intermediate DTOs.
//! - [`install`] — `pkg_install`, local-path variant, and bundled auto-install.
//! - [`remove`] — `pkg_remove` (alc.toml + alc.lock deletion).

mod install;
mod list;
mod remove;
mod repair;

#[cfg(test)]
mod tests;
