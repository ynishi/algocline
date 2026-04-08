//! Package management — install, list, remove, and project-local resolution.
//!
//! # Package resolution model
//!
//! algocline resolves `require("name")` through an ordered chain of
//! `mlua_isle::FsResolver`s. When `alc_run` / `alc_advice` starts a session,
//! the executor registers resolvers in the following priority (high → low):
//!
//! 1. **Project-local** (`alc.lock` `local_dir` entries) — zero-copy links to
//!    on-disk directories within the project tree. Resolved via
//!    [`super::lockfile::resolve_local_dir_paths`].
//! 2. **`ALC_PACKAGES_PATH`** — colon-separated search paths from the env var.
//! 3. **Global default** (`~/.algocline/packages/`) — packages installed by
//!    `alc_pkg_install` or `alc init`.
//!
//! A higher-priority package **shadows** a lower-priority one with the same
//! name. `pkg_list` reports both, marking the effective one as `active: true`.
//!
//! # Scope concept
//!
//! Each package belongs to a scope:
//!
//! - **`"global"`** — installed in `~/.algocline/packages/` via `pkg_install`.
//!   Physical directory managed by algocline (clone/copy/delete).
//! - **`"project"`** — declared in `alc.lock` via [`super::pkg_link`].
//!   Only a path reference is stored; algocline never copies or deletes the
//!   source files.
//!
//! `pkg_remove` with `scope: "project"` removes the `alc.lock` entry only.
//! `scope: "global"` deletes the directory from `~/.algocline/packages/`.
//! When omitted, project scope is tried first, then global fallback.
//!
//! # `alc.lock` lifecycle
//!
//! `alc.lock` is a TOML file at the project root (see [`super::lockfile`]).
//! Created on first `pkg_link` call. Updated atomically via temp-file + rename.
//! Read at session start to build extra `FsResolver` entries.
//!
//! Path entries are **containment-checked**: canonicalized paths that escape
//! the project root are rejected at both write time (`pkg_link`) and read
//! time (`resolve_local_dir_paths`).
//!
//! # Project root resolution
//!
//! See [`super::project`] for the 3-tier priority (explicit arg →
//! `ALC_PROJECT_ROOT` env → ancestor walk from cwd).
//!
//! # Security invariants
//!
//! - Package names are whitelist-validated (`[a-zA-Z0-9_-]`) before Lua
//!   interpolation in `pkg_list` meta evaluation.
//! - `LocalDir` paths are canonicalized and containment-checked against
//!   `project_root` to prevent path traversal via hand-edited `alc.lock`.
//! - Lockfile writes are serialized in-process (`SAVE_GUARD` mutex)
//!   and use atomic rename to prevent half-written reads.
//!
//! # Submodules
//!
//! - [`list`] — `pkg_list` and its intermediate DTOs.
//! - [`install`] — `pkg_install`, local-path variant, and bundled auto-install.
//! - [`remove`] — `pkg_remove` and scope resolution.

mod install;
mod list;
mod remove;

#[cfg(test)]
mod tests;
