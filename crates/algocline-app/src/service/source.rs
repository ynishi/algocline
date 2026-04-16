//! Package source classification for algocline packages.
//!
//! Defines the `PackageSource` enum which describes how a package was obtained.
//! Also provides `infer_from_legacy_source_string` for backward-compatible
//! interpretation of the legacy `ManifestEntry.source: String` field.
//!
//! Note: `ManifestEntry.source` is **not** modified. Callers use this function
//! to interpret the string value without touching the existing manifest schema.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Describes the origin of an algocline package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum PackageSource {
    /// Package installed into `~/.algocline/packages/` (from Git or local copy).
    Installed,
    /// Package linked directly from a local path (no copy).
    /// Changes to files in `path` are reflected immediately on next `alc_run`.
    Path { path: String },
    /// Package cloned/fetched from a Git repository.
    Git { url: String, rev: Option<String> },
    /// Bundled package shipped with algocline.
    Bundled { collection: Option<String> },
}

/// Infer a `PackageSource` from a legacy `ManifestEntry.source` string.
///
/// Heuristics (evaluated in order, **syntactic only**):
/// 1. `"bundled"` → `Bundled { collection: None }`
/// 2. Absolute path (by shape, not existence) → `Installed`
/// 3. Anything else → `Git { url, rev: None }`
///
/// The classification deliberately does **not** touch the filesystem. A prior
/// version used `p.is_dir()`, which produced a symptom identical to the one
/// `pkg_install` → `classify_install_url` previously had: if the original
/// local source directory was deleted after install (e.g. a `tempfile::tempdir`
/// source that e2e tests drop at end-of-scope), the string fell through to
/// the `Git` arm and `normalize_git_url("/abs/path")` produced
/// `"https:///abs/path"`, which git rejects as
/// `unable to find remote helper for 'https'`. Classifying by absolute-path
/// shape alone lets `pkg_repair` route the entry to the `LocalPath` installer,
/// whose error message ("Failed to read source dir: ... No such file or directory")
/// is at least diagnostic.
pub(crate) fn infer_from_legacy_source_string(s: &str) -> PackageSource {
    if s == "bundled" {
        return PackageSource::Bundled { collection: None };
    }

    let p = Path::new(s);
    if p.is_absolute() {
        return PackageSource::Installed;
    }

    PackageSource::Git {
        url: s.to_string(),
        rev: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_git_url() {
        let result =
            infer_from_legacy_source_string("https://github.com/ynishi/algocline-bundled-packages");
        assert_eq!(
            result,
            PackageSource::Git {
                url: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
                rev: None,
            }
        );
    }

    #[test]
    fn infer_local_copy() {
        // Use a directory that is guaranteed to exist on any system.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();

        let result = infer_from_legacy_source_string(&path);
        assert_eq!(result, PackageSource::Installed);
    }

    #[test]
    fn infer_bundled() {
        let result = infer_from_legacy_source_string("bundled");
        assert_eq!(result, PackageSource::Bundled { collection: None });
    }

    #[test]
    fn infer_non_existent_absolute_path_is_installed() {
        // An absolute path that does NOT exist still classifies as Installed —
        // classification is now syntactic (shape, not existence). The caller
        // (pkg_repair) then routes to the LocalPath installer, whose copy-dir
        // error is more diagnostic than `git clone https:///abs/...`.
        let result =
            infer_from_legacy_source_string("/nonexistent/path/that/should/never/exist_xyz");
        assert_eq!(result, PackageSource::Installed);
    }

    #[test]
    fn infer_relative_path_is_git() {
        // Relative paths are NOT Installed — they fall through to the Git arm.
        // (Callers never record relative paths as `ManifestEntry.source`, but
        // keep this branch for backward compatibility with older manifests.)
        let result = infer_from_legacy_source_string("relative/path/pkg");
        assert!(matches!(result, PackageSource::Git { .. }));
    }
}
