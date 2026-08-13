//! Portable snapshots of `~/.algocline` (`alc_pack` / `alc_unpack`).
//!
//! A pack is a directory, not an archive:
//!
//! ```text
//! <name>.alcpack/
//! ├── profile.toml     declarative half — what to re-fetch, what to re-link
//! └── payload/         byte half — what cannot be reproduced
//! ```
//!
//! Compression is left to the caller (`tar czf pack.tgz <dir>`) so that the
//! crate takes no archive-format dependency and the on-disk shape stays
//! inspectable with ordinary tools.
//!
//! See [`profile`] for the reproducible / irreproducible / referential split
//! that the format is built around.

pub(crate) mod create;
pub(crate) mod fs;
pub(crate) mod profile;
pub(crate) mod restore;

pub use create::PackOptions;
pub use restore::{UnpackMode, UnpackOptions};
