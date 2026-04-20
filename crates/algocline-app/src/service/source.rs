//! Package source classification for algocline packages.
//!
//! Defines the `PackageSource` typed enum that describes how a package was
//! obtained. This is the canonical representation used throughout the
//! manifest (`installed.json`), hub index (`hub_index.json`), and project
//! lockfile. Serialization uses a tagged form (`{"type": "...", ...}`) on
//! write; deserialization transparently accepts **two** wire shapes to
//! preserve backward compatibility with pre-typed manifests:
//!
//! 1. **Typed form** (new, always written): `{"type": "git", "url": "..."}`.
//! 2. **Legacy string form** (read-only compatibility): a bare string like
//!    `""`, `"bundled"`, `"https://..."`, or `"/abs/path"`. Coerced through
//!    [`infer_from_legacy_source_string`] on load.
//!
//! Any write path produces the typed form. Once a manifest or index entry
//! is rewritten (e.g. by the first successful `pkg_install` or
//! `alc_hub_reindex` after upgrade), the legacy representation is gone and
//! future reads deserialize directly as typed data.
//!
//! `PackageSource::Unknown` is the landing site for legacy `""` entries
//! and for `Default::default()` fallbacks (e.g. `hub.rs::build_index` when
//! the manifest has no record for a directory). See plan §6 for why a
//! dedicated `Unknown` variant is preferred over `Option<PackageSource>`.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Describes the origin of an algocline package.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    from = "PackageSourceRepr",
    into = "PackageSourceRepr"
)]
pub(crate) enum PackageSource {
    /// Source is not tracked (e.g. legacy `hub_index.json` with
    /// `source: ""`) or not yet known (manifest lookup miss during
    /// `build_index`). Carries no structural meaning beyond "we do not
    /// know where this came from yet — reindex to resolve".
    #[default]
    Unknown,
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

/// On-the-wire representation. Accepts either a typed tagged object
/// (`{"type": "...", ...}`) or a bare legacy string (`""`, `"bundled"`,
/// URL, absolute path). Always serializes as the typed tagged object so
/// subsequent reads are trivially typed.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PackageSourceRepr {
    /// Tagged form. The field order matches `PackageSource` so the derived
    /// `Serialize` round-trips cleanly.
    Typed(PackageSourceTyped),
    /// Legacy bare string (pre-typed manifest / hub_index entries).
    Legacy(String),
}

/// Private typed mirror of `PackageSource` used purely for serde
/// round-tripping through `PackageSourceRepr`. Keeping this separate from
/// the public enum lets us use `from` / `into` on `PackageSource` without
/// creating an infinite derive loop.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PackageSourceTyped {
    Unknown,
    Installed,
    Path { path: String },
    Git { url: String, rev: Option<String> },
    Bundled { collection: Option<String> },
}

impl From<PackageSourceTyped> for PackageSource {
    fn from(t: PackageSourceTyped) -> Self {
        match t {
            PackageSourceTyped::Unknown => PackageSource::Unknown,
            PackageSourceTyped::Installed => PackageSource::Installed,
            PackageSourceTyped::Path { path } => PackageSource::Path { path },
            PackageSourceTyped::Git { url, rev } => PackageSource::Git { url, rev },
            PackageSourceTyped::Bundled { collection } => PackageSource::Bundled { collection },
        }
    }
}

impl From<PackageSource> for PackageSourceTyped {
    fn from(s: PackageSource) -> Self {
        match s {
            PackageSource::Unknown => PackageSourceTyped::Unknown,
            PackageSource::Installed => PackageSourceTyped::Installed,
            PackageSource::Path { path } => PackageSourceTyped::Path { path },
            PackageSource::Git { url, rev } => PackageSourceTyped::Git { url, rev },
            PackageSource::Bundled { collection } => PackageSourceTyped::Bundled { collection },
        }
    }
}

impl From<PackageSourceRepr> for PackageSource {
    fn from(r: PackageSourceRepr) -> Self {
        match r {
            PackageSourceRepr::Typed(t) => t.into(),
            PackageSourceRepr::Legacy(s) => infer_from_legacy_source_string(&s),
        }
    }
}

impl From<PackageSource> for PackageSourceRepr {
    fn from(s: PackageSource) -> Self {
        PackageSourceRepr::Typed(s.into())
    }
}

impl PackageSource {
    /// Return the underlying Git URL when the source is a Git repo.
    ///
    /// Used by hub index-URL discovery (`discover_index_urls` in
    /// `hub.rs`) where only Git-backed entries can host a remote
    /// `hub_index.json`. All other variants return `None`; callers
    /// `filter_map` on the result.
    pub(crate) fn git_url(&self) -> Option<&str> {
        match self {
            PackageSource::Git { url, .. } => Some(url.as_str()),
            _ => None,
        }
    }

    /// Return a human-readable one-line description of this source.
    ///
    /// Used by JSON output payloads (e.g. `pkg_list.install_source`,
    /// `pkg_repair.RepairOutcome.source`) where consumers expect a single
    /// string rather than a tagged object. Prefer emitting the typed
    /// `PackageSource` directly where the downstream schema allows it;
    /// this helper is the fallback for legacy string fields.
    ///
    /// Mapping:
    /// - `Unknown`  → `""`  (preserves the legacy "no source recorded" sentinel)
    /// - `Installed` → `"installed"`
    /// - `Path { path }` → `path.clone()`
    /// - `Git { url, .. }` → `url.clone()`
    /// - `Bundled { collection: Some(c) }` → `format!("bundled:{c}")`
    /// - `Bundled { collection: None }` → `"bundled"`
    pub(crate) fn display_string(&self) -> String {
        match self {
            PackageSource::Unknown => String::new(),
            PackageSource::Installed => "installed".to_string(),
            PackageSource::Path { path } => path.clone(),
            PackageSource::Git { url, .. } => url.clone(),
            PackageSource::Bundled {
                collection: Some(c),
            } => format!("bundled:{c}"),
            PackageSource::Bundled { collection: None } => "bundled".to_string(),
        }
    }
}

/// Infer a `PackageSource` from a legacy `source: String` manifest / index
/// field. Used as the deserialization fallback when the wire format is a
/// bare string rather than the tagged object.
///
/// Heuristics (evaluated in order, **syntactic only**):
/// 1. `""` (empty string) → `Unknown` — legacy "no source recorded"
///    entries in `hub_index.json` and pre-typed `installed.json`.
/// 2. `"bundled"` → `Bundled { collection: None }`
/// 3. Absolute path (by shape, not existence) → `Installed`
/// 4. Anything else → `Git { url, rev: None }`
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
    if s.is_empty() {
        return PackageSource::Unknown;
    }
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
    fn infer_empty_string_is_unknown() {
        // Regression: previously `""` fell through to the `Git` arm and
        // produced `Git { url: "", rev: None }`, which is nonsense. After
        // the typed migration, `""` must map to `Unknown` (the legacy
        // "source not recorded" sentinel) — both on the
        // `infer_from_legacy_source_string` path and via the `PackageSource`
        // serde deserialize path (see `deserialize_empty_string_is_unknown`).
        let result = infer_from_legacy_source_string("");
        assert_eq!(result, PackageSource::Unknown);
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

    // ─── serde: legacy string ⇄ typed round trip ─────────────────

    #[test]
    fn deserialize_empty_string_is_unknown() {
        // Legacy `hub_index.json` entries emit `"source": ""`; they must
        // round-trip through the serde deserialize path (not just the
        // `infer_*` helper) and land on `PackageSource::Unknown`.
        let src: PackageSource = serde_json::from_str("\"\"").unwrap();
        assert_eq!(src, PackageSource::Unknown);
    }

    #[test]
    fn deserialize_legacy_bundled_string() {
        let src: PackageSource = serde_json::from_str("\"bundled\"").unwrap();
        assert_eq!(src, PackageSource::Bundled { collection: None });
    }

    #[test]
    fn deserialize_legacy_git_url_string() {
        let src: PackageSource = serde_json::from_str("\"https://github.com/a/b\"").unwrap();
        assert_eq!(
            src,
            PackageSource::Git {
                url: "https://github.com/a/b".to_string(),
                rev: None,
            }
        );
    }

    #[test]
    fn deserialize_legacy_absolute_path_string() {
        // Syntactically absolute → Installed, even for a path that does
        // not currently exist (mirrors `infer_non_existent_absolute_path_is_installed`).
        let src: PackageSource = serde_json::from_str("\"/abs/nonexistent/path\"").unwrap();
        assert_eq!(src, PackageSource::Installed);
    }

    #[test]
    fn deserialize_typed_form() {
        let src: PackageSource =
            serde_json::from_str(r#"{"type":"git","url":"https://x/y","rev":null}"#).unwrap();
        assert_eq!(
            src,
            PackageSource::Git {
                url: "https://x/y".to_string(),
                rev: None,
            }
        );

        let src: PackageSource = serde_json::from_str(r#"{"type":"unknown"}"#).unwrap();
        assert_eq!(src, PackageSource::Unknown);

        let src: PackageSource = serde_json::from_str(r#"{"type":"installed"}"#).unwrap();
        assert_eq!(src, PackageSource::Installed);

        let src: PackageSource =
            serde_json::from_str(r#"{"type":"bundled","collection":"pkg-set"}"#).unwrap();
        assert_eq!(
            src,
            PackageSource::Bundled {
                collection: Some("pkg-set".to_string())
            }
        );
    }

    #[test]
    fn serialize_always_typed_form() {
        // Forward-migration contract: write path is always tagged, never
        // legacy-string. After one save, future loads take the fast path.
        let src = PackageSource::Git {
            url: "https://x/y".to_string(),
            rev: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            json.contains("\"type\":\"git\""),
            "write must emit tagged form: {json}"
        );
        assert!(json.contains("\"url\":\"https://x/y\""), "{json}");

        let unk = PackageSource::Unknown;
        let json = serde_json::to_string(&unk).unwrap();
        assert_eq!(json, r#"{"type":"unknown"}"#);
    }

    #[test]
    fn round_trip_typed() {
        let cases = [
            PackageSource::Unknown,
            PackageSource::Installed,
            PackageSource::Path {
                path: "/some/path".to_string(),
            },
            PackageSource::Git {
                url: "https://x/y".to_string(),
                rev: Some("abc123".to_string()),
            },
            PackageSource::Bundled {
                collection: Some("main".to_string()),
            },
        ];
        for src in cases {
            let json = serde_json::to_string(&src).unwrap();
            let back: PackageSource = serde_json::from_str(&json).unwrap();
            assert_eq!(back, src, "round trip failed: {json}");
        }
    }

    #[test]
    fn default_is_unknown() {
        assert_eq!(PackageSource::default(), PackageSource::Unknown);
    }

    // ─── serde: malformed-form rejection (no silent fallback) ─────
    //
    // `PackageSourceRepr` is `#[serde(untagged)]` with two arms:
    //   1. Typed   — tagged object `{"type": "...", ...}`
    //   2. Legacy  — bare string
    //
    // A worry with `untagged` is that a malformed tagged object (e.g. an
    // unknown `type` tag, or a tagged shape missing a required field) could
    // silently fall through to the Legacy arm and be coerced into a
    // `Git { url: "" }` or `Unknown`. These tests pin the actual behavior:
    // object-shaped inputs cannot match the Legacy `String` arm, so serde
    // correctly surfaces an error when the Typed arm fails. Regression guard
    // for the code review MEDIUM finding (malformed tagged → no silent coerce).

    #[test]
    fn deserialize_unknown_type_tag_is_error() {
        // Unknown variant name in the `type` field. The Typed arm fails
        // (no matching variant); the Legacy arm fails (input is an object,
        // not a string). Net result: serde error, not a silent `Unknown`.
        let result: Result<PackageSource, _> =
            serde_json::from_str(r#"{"type":"unknown_variant","path":"/x"}"#);
        assert!(
            result.is_err(),
            "malformed tagged form must be rejected, got {result:?}"
        );
    }

    #[test]
    fn deserialize_git_missing_url_is_error() {
        // `type: "git"` without the required `url` field. Typed arm fails
        // (missing field); Legacy arm fails (object ≠ string). Must error.
        let result: Result<PackageSource, _> = serde_json::from_str(r#"{"type":"git"}"#);
        assert!(
            result.is_err(),
            "git variant with missing url must be rejected, got {result:?}"
        );
    }

    #[test]
    fn deserialize_path_missing_path_is_error() {
        // `type: "path"` without the required `path` field.
        let result: Result<PackageSource, _> = serde_json::from_str(r#"{"type":"path"}"#);
        assert!(
            result.is_err(),
            "path variant with missing path must be rejected, got {result:?}"
        );
    }

    #[test]
    fn deserialize_object_without_type_key_is_error() {
        // Arbitrary object without a `type` discriminator at all. Neither
        // arm accepts it.
        let result: Result<PackageSource, _> = serde_json::from_str(r#"{"foo":"bar"}"#);
        assert!(
            result.is_err(),
            "object without type tag must be rejected, got {result:?}"
        );
    }

    #[test]
    fn deserialize_non_string_non_object_scalar_is_error() {
        // Numbers, booleans, null are all invalid shapes. Neither arm
        // accepts them (Legacy requires String, Typed requires object).
        for json in ["null", "42", "true", "[]"] {
            let result: Result<PackageSource, _> = serde_json::from_str(json);
            assert!(
                result.is_err(),
                "scalar/array {json} must be rejected, got {result:?}"
            );
        }
    }

    #[test]
    fn git_url_accessor_returns_url_only_for_git() {
        assert_eq!(
            PackageSource::Git {
                url: "https://a/b".to_string(),
                rev: None
            }
            .git_url(),
            Some("https://a/b")
        );
        assert_eq!(PackageSource::Unknown.git_url(), None);
        assert_eq!(PackageSource::Installed.git_url(), None);
        assert_eq!(
            PackageSource::Path {
                path: "/x".to_string()
            }
            .git_url(),
            None
        );
        assert_eq!(PackageSource::Bundled { collection: None }.git_url(), None);
    }
}
