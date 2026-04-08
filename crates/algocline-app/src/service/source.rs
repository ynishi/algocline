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
    /// Package cloned/fetched from a Git repository.
    Git { url: String, rev: Option<String> },
    /// Package installed by copying from a local path.
    /// The directory is a snapshot copy; changes to the original are not reflected.
    LocalCopy { path: String },
    /// Package linked directly from a local directory (no copy).
    /// Changes to files in `path` are reflected immediately on next `alc_run`.
    LocalDir { path: String },
    /// Bundled package shipped with algocline.
    Bundled { collection: Option<String> },
}

/// Infer a `PackageSource` from a legacy `ManifestEntry.source` string.
///
/// Heuristics (evaluated in order):
/// 1. `"bundled"` → `Bundled { collection: None }`
/// 2. Absolute path that exists as a directory → `LocalCopy { path }`
/// 3. Anything else → `Git { url, rev: None }`
// Used in Subtask 3 (alc_pkg_list/remove project-awareness).
#[allow(dead_code)]
pub(crate) fn infer_from_legacy_source_string(s: &str) -> PackageSource {
    if s == "bundled" {
        return PackageSource::Bundled { collection: None };
    }

    let p = Path::new(s);
    if p.is_absolute() && p.is_dir() {
        return PackageSource::LocalCopy {
            path: s.to_string(),
        };
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
        assert_eq!(result, PackageSource::LocalCopy { path });
    }

    #[test]
    fn infer_bundled() {
        let result = infer_from_legacy_source_string("bundled");
        assert_eq!(result, PackageSource::Bundled { collection: None });
    }

    #[test]
    fn infer_non_existent_absolute_path_is_git() {
        // An absolute path that does NOT exist → falls through to Git.
        let result =
            infer_from_legacy_source_string("/nonexistent/path/that/should/never/exist_xyz");
        assert!(matches!(result, PackageSource::Git { .. }));
    }
}
