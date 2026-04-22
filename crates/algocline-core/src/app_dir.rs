//! Application directory layout for algocline.
//!
//! [`AppDir`] encapsulates the root directory (`~/.algocline/` by default,
//! overridable via the `ALC_HOME` environment variable) and provides typed
//! path accessors for each subsystem directory. Construction is expected to
//! happen in a single resolution point (`AppConfig::from_env` in the app
//! crate); downstream Service-layer code reaches these paths through the
//! accessors here rather than reading `HOME` / `ALC_HOME` directly.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Application root directory.
///
/// The inner `root` is stored behind an [`Arc`] so `AppDir::clone` is
/// `O(1)` (refcount bump, no path allocation). Downstream services call
/// `clone()` on `AppDir` freely as part of per-request delegate
/// construction (see `FsInstalledManifestStore` and siblings in `algocline-app`);
/// keeping clone cheap avoids the per-call `PathBuf` allocation that an
/// owned `PathBuf` field would incur.
#[derive(Clone, Debug)]
pub struct AppDir {
    root: Arc<PathBuf>,
}

impl AppDir {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root: Arc::new(root),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn packages_dir(&self) -> PathBuf {
        self.root.join("packages")
    }

    pub fn cards_dir(&self) -> PathBuf {
        self.root.join("cards")
    }

    pub fn state_dir(&self) -> PathBuf {
        self.root.join("state")
    }

    pub fn evals_dir(&self) -> PathBuf {
        self.root.join("evals")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn scenarios_dir(&self) -> PathBuf {
        self.root.join("scenarios")
    }

    pub fn types_dir(&self) -> PathBuf {
        self.root.join("types")
    }

    pub fn hub_cache_dir(&self) -> PathBuf {
        self.root.join("hub_cache")
    }

    pub fn installed_json(&self) -> PathBuf {
        self.root.join("installed.json")
    }

    pub fn hub_registries_json(&self) -> PathBuf {
        self.root.join("hub_registries.json")
    }

    pub fn config_toml(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn init_lua(&self) -> PathBuf {
        self.root.join("init.lua")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_each_subdir_under_root() {
        let dir = AppDir::new(PathBuf::from("/tmp/alc-root"));
        assert_eq!(dir.root(), Path::new("/tmp/alc-root"));
        assert_eq!(dir.packages_dir(), PathBuf::from("/tmp/alc-root/packages"));
        assert_eq!(dir.cards_dir(), PathBuf::from("/tmp/alc-root/cards"));
        assert_eq!(dir.state_dir(), PathBuf::from("/tmp/alc-root/state"));
        assert_eq!(dir.evals_dir(), PathBuf::from("/tmp/alc-root/evals"));
        assert_eq!(dir.logs_dir(), PathBuf::from("/tmp/alc-root/logs"));
        assert_eq!(
            dir.scenarios_dir(),
            PathBuf::from("/tmp/alc-root/scenarios")
        );
        assert_eq!(dir.types_dir(), PathBuf::from("/tmp/alc-root/types"));
        assert_eq!(
            dir.hub_cache_dir(),
            PathBuf::from("/tmp/alc-root/hub_cache")
        );
        assert_eq!(
            dir.installed_json(),
            PathBuf::from("/tmp/alc-root/installed.json")
        );
        assert_eq!(
            dir.hub_registries_json(),
            PathBuf::from("/tmp/alc-root/hub_registries.json")
        );
        assert_eq!(
            dir.config_toml(),
            PathBuf::from("/tmp/alc-root/config.toml")
        );
        assert_eq!(dir.init_lua(), PathBuf::from("/tmp/alc-root/init.lua"));
    }

    #[test]
    fn clone_is_independent() {
        let a = AppDir::new(PathBuf::from("/a"));
        let b = a.clone();
        assert_eq!(a.root(), b.root());
    }
}
