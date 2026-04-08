//! Package management: install, list, remove.
//!
//! Split across submodules by operation to keep each file focused:
//! - [`list`] — `pkg_list` and its intermediate DTOs.
//! - [`install`] — `pkg_install`, local-path variant, and bundled auto-install.
//! - [`remove`] — `pkg_remove` and scope resolution.

mod install;
mod list;
mod remove;

#[cfg(test)]
mod tests;
