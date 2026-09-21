//! Which [`CardBackend`] a process runs with, resolved from
//! `[setting.card]`.
//!
//! There are two construction sites — [`AppService::new`] and the pool
//! worker's `main` — and they must agree, or a Card opened over MCP
//! would be closed into a different store. So the choice is resolved
//! once here and both sites call it.
//!
//! [`AppService::new`]: crate::AppService::new

use std::path::PathBuf;
use std::sync::Arc;

use algocline_core::AppDir;
use algocline_engine::{CardBackend, FileCardStore};

use super::cardbox_store::CardboxStore;
use super::setting::resolve_setting;

/// `[setting.card].backend` value selecting the file backend. Default.
const BACKEND_FILE: &str = "file";
/// `[setting.card].backend` value selecting the cardbox backend.
const BACKEND_CARDBOX: &str = "cardbox";

/// The resolved card backend, before it is built.
///
/// Kept as a description rather than only as the built `Arc` so that
/// `alc info` can report what it resolved to without downcasting a
/// trait object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardBackendChoice {
    /// TOML files under `~/.algocline/cards`.
    File { root: PathBuf },
    /// A cardbox root, driven through the `cardbox` binary.
    Cardbox { root: PathBuf },
}

impl CardBackendChoice {
    /// Resolve `[setting.card].backend` / `.cardbox_root` through the
    /// same layered resolver as `[setting.card].run`.
    ///
    /// An unrecognised `backend` value falls back to the file backend
    /// with a `tracing::warn`: this runs inside `AppService::new`, which
    /// has no way to return an error, and a typo that silently switched
    /// stores would be worse than one that kept the default. The value
    /// is visible in `alc info` either way.
    pub fn resolve(app_dir: &AppDir) -> Self {
        let file = || Self::File {
            root: app_dir.cards_dir(),
        };
        let resolved = match resolve_setting(app_dir, None, Some("card")) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("card backend: failed to resolve [setting.card] ({e}) — using file");
                return file();
            }
        };
        match resolved.get_str("card", "backend").unwrap_or(BACKEND_FILE) {
            BACKEND_FILE => file(),
            BACKEND_CARDBOX => Self::Cardbox {
                root: resolved
                    .get_str("card", "cardbox_root")
                    .map(expand_home)
                    .unwrap_or_else(|| default_cardbox_root(app_dir)),
            },
            other => {
                tracing::warn!(
                    "card backend: [setting.card].backend = '{other}' is not one of \
                     '{BACKEND_FILE}' / '{BACKEND_CARDBOX}' — using '{BACKEND_FILE}'"
                );
                file()
            }
        }
    }

    /// Build the backend this choice names.
    pub fn build(&self) -> Arc<dyn CardBackend> {
        match self {
            Self::File { root } => Arc::new(FileCardStore::new(root.clone())),
            Self::Cardbox { root } => Arc::new(CardboxStore::with_default_bin(root.clone())),
        }
    }

    /// The `backend` token, as it is written in `alc.toml`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::File { .. } => BACKEND_FILE,
            Self::Cardbox { .. } => BACKEND_CARDBOX,
        }
    }

    /// The store root, whichever backend this is.
    pub fn root(&self) -> &std::path::Path {
        match self {
            Self::File { root } | Self::Cardbox { root } => root,
        }
    }

    /// The `card_backend` block `alc info` reports.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind(),
            "root": self.root().display().to_string(),
        })
    }
}

/// `~/.algocline/cardbox` — beside `cards`, not inside it.
fn default_cardbox_root(app_dir: &AppDir) -> PathBuf {
    app_dir.root().join("cardbox")
}

/// Expand a leading `~/` against the home directory.
///
/// A config file is somewhere a `~` gets typed, and without this the
/// store would quietly root itself in a directory literally named `~`
/// next to wherever the process started — a wrong path that looks like
/// an empty one. A `~` anywhere else is left alone; it is a legal
/// character in a directory name.
fn expand_home(raw: &str) -> PathBuf {
    match raw.strip_prefix("~/") {
        Some(rest) => match crate::service::config::AppConfig::resolve_home() {
            Some(home) => home.join(rest),
            None => PathBuf::from(raw),
        },
        None => PathBuf::from(raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_file_backend_rooted_at_cards_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app_dir = AppDir::new(dir.path().to_path_buf());
        let choice = CardBackendChoice::resolve(&app_dir);
        assert_eq!(
            choice,
            CardBackendChoice::File {
                root: app_dir.cards_dir()
            }
        );
        assert_eq!(choice.kind(), "file");
    }

    #[test]
    fn cardbox_root_defaults_beside_cards_not_inside_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app_dir = AppDir::new(dir.path().to_path_buf());
        assert_eq!(
            default_cardbox_root(&app_dir),
            dir.path().join("cardbox"),
            "the cardbox store is its own root, not a subdirectory of the file store"
        );
        assert_ne!(default_cardbox_root(&app_dir), app_dir.cards_dir());
    }

    /// A `~` that stayed literal would root the store in a directory
    /// named `~` beside the cwd, which reads as "the store is empty".
    #[test]
    fn a_leading_tilde_is_expanded_not_taken_literally() {
        let expanded = expand_home("~/.algocline/cardbox");
        assert!(!expanded.starts_with("~"), "{}", expanded.display());
        if let Some(home) = crate::service::config::AppConfig::resolve_home() {
            assert_eq!(expanded, home.join(".algocline/cardbox"));
        }
        // Absolute paths and a `~` elsewhere are left alone.
        assert_eq!(expand_home("/srv/cardbox"), PathBuf::from("/srv/cardbox"));
        assert_eq!(expand_home("/srv/~odd"), PathBuf::from("/srv/~odd"));
    }

    #[test]
    fn info_block_names_the_kind_and_the_root() {
        let choice = CardBackendChoice::Cardbox {
            root: PathBuf::from("/tmp/cb"),
        };
        assert_eq!(
            choice.to_json(),
            serde_json::json!({ "kind": "cardbox", "root": "/tmp/cb" })
        );
    }
}
