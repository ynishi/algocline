//! Pack profile (`<name>.alcpack/profile.toml`) — the declarative half of a
//! portable `~/.algocline` snapshot.
//!
//! # Why a profile and not a plain archive
//!
//! A `~/.algocline` tree mixes three kinds of state:
//!
//! 1. **Reproducible** — packages installed from Git / bundled collections.
//!    Carrying the bytes is wasteful and pins the destination to whatever
//!    revision the source machine happened to hold. The profile records the
//!    [`PackageSource`] and `unpack` re-fetches through `pkg_install`.
//! 2. **Irreproducible** — packages whose origin is a local path, packages
//!    present on disk but absent from `installed.json`, plus the accumulated
//!    cards / evals / checkpoints. These must travel as bytes.
//! 3. **Referential** — symlinks created by `pkg_link`, which point outside
//!    `~/.algocline` entirely. Whether the target exists on the destination
//!    machine cannot be known at pack time, so only the link definition
//!    travels and `unpack` reports what it could not resolve.
//!
//! Note that `installed.json` is *not* the source of truth for
//! `~/.algocline/packages/`: `pkg_link` records nothing there, so a manifest-only
//! snapshot silently drops every linked and every unregistered package. The
//! filesystem walk is the primary input; the manifest only classifies what the
//! walk finds.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use algocline_core::AppDir;
use serde::{Deserialize, Serialize};

use crate::service::source::PackageSource;

/// Current `profile.toml` schema version.
///
/// `unpack` refuses profiles carrying a higher version rather than guessing
/// at unknown fields — a newer pack may encode sections this build cannot
/// place on disk.
pub(crate) const FORMAT_VERSION: u32 = 1;

// ─── Sections ───────────────────────────────────────────────────

/// A selectable slice of the application directory.
///
/// Sections are named after [`AppDir`] accessors rather than raw path
/// fragments so that a layout change in `AppDir` cannot silently desync the
/// pack surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PackSection {
    /// `config.toml`, `hub_registries.json`, `installed.json`, `scenarios/`,
    /// and the payload packages. Always present; a pack without it cannot be
    /// unpacked at all.
    Core,
    /// `cards/` — accumulated card store.
    Cards,
    /// `evals/` — eval result history.
    Evals,
    /// `nn/` — safetensors checkpoints. Large (hundreds of MB) and mostly
    /// intermediate experiment output, hence opt-in.
    Nn,
    /// `state/` — orchestration / session state.
    State,
    /// `logs/` — execution logs. Excluded even under `all`: the volume is
    /// large and the migration value is low, so carrying it must be a
    /// deliberate `include`.
    Logs,
    /// `hub_cache/` — refetchable, excluded from `all`.
    HubCache,
    /// `types/` — generated `.d.lua`, excluded from `all`.
    Types,
}

impl PackSection {
    /// Every section, in a stable order.
    pub(crate) const ALL: [PackSection; 8] = [
        PackSection::Core,
        PackSection::Cards,
        PackSection::Evals,
        PackSection::Nn,
        PackSection::State,
        PackSection::Logs,
        PackSection::HubCache,
        PackSection::Types,
    ];

    /// Wire / CLI name, matching the serde representation.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            PackSection::Core => "core",
            PackSection::Cards => "cards",
            PackSection::Evals => "evals",
            PackSection::Nn => "nn",
            PackSection::State => "state",
            PackSection::Logs => "logs",
            PackSection::HubCache => "hub_cache",
            PackSection::Types => "types",
        }
    }

    /// Parse a caller-supplied section name.
    pub(crate) fn parse(name: &str) -> Result<Self, PackProfileError> {
        PackSection::ALL
            .into_iter()
            .find(|s| s.as_str() == name)
            .ok_or_else(|| PackProfileError::UnknownSection {
                name: name.to_string(),
                known: PackSection::ALL
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// Included when the caller passes no selection at all.
    pub(crate) fn on_by_default(&self) -> bool {
        matches!(
            self,
            PackSection::Core | PackSection::Cards | PackSection::Evals
        )
    }

    /// Included by `all: true`.
    ///
    /// `logs` / `hub_cache` / `types` are deliberately asymmetric — they are
    /// reachable only through an explicit `include`.
    pub(crate) fn in_all(&self) -> bool {
        !matches!(
            self,
            PackSection::Logs | PackSection::HubCache | PackSection::Types
        )
    }

    /// The single directory this section copies, if it is directory-shaped.
    ///
    /// [`PackSection::Core`] spans several files plus a filtered subset of
    /// `packages/`, so it has no single directory and returns `None`; its
    /// copy logic lives in the pack service.
    pub(crate) fn dir(&self, app_dir: &AppDir) -> Option<PathBuf> {
        match self {
            PackSection::Core => None,
            PackSection::Cards => Some(app_dir.cards_dir()),
            PackSection::Evals => Some(app_dir.evals_dir()),
            PackSection::Nn => Some(app_dir.nn_dir()),
            PackSection::State => Some(app_dir.state_dir()),
            PackSection::Logs => Some(app_dir.logs_dir()),
            PackSection::HubCache => Some(app_dir.hub_cache_dir()),
            PackSection::Types => Some(app_dir.types_dir()),
        }
    }
}

/// Resolve the effective section set from the caller's three knobs.
///
/// Precedence is: start from `all` (or the defaults), add every `include`,
/// then subtract every `exclude`. `all` and `include` compose as a union, so
/// `all: true, include: ["logs"]` is the documented way to take everything.
/// [`PackSection::Core`] is force-added last and may not be excluded.
pub(crate) fn resolve_sections(
    all: bool,
    include: &[String],
    exclude: &[String],
) -> Result<BTreeSet<PackSection>, PackProfileError> {
    let mut selected: BTreeSet<PackSection> = PackSection::ALL
        .into_iter()
        .filter(|s| if all { s.in_all() } else { s.on_by_default() })
        .collect();

    for name in include {
        selected.insert(PackSection::parse(name)?);
    }

    for name in exclude {
        let section = PackSection::parse(name)?;
        if section == PackSection::Core {
            return Err(PackProfileError::CoreExcluded);
        }
        selected.remove(&section);
    }

    // Core is structural, not optional: without it there is no profile to
    // unpack from.
    selected.insert(PackSection::Core);
    Ok(selected)
}

// ─── Profile document ───────────────────────────────────────────

/// Top-level `profile.toml` document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackProfile {
    pub pack: PackMeta,
    /// Packages restored by re-fetching from their source (unpack phase 1).
    #[serde(default)]
    pub packages: Vec<ProfilePackage>,
    /// Packages restored by copying bytes out of `payload/` (unpack phase 2).
    #[serde(default)]
    pub payloads: Vec<ProfilePayload>,
    /// Symlinks re-created against the destination filesystem (unpack phase 3).
    #[serde(default)]
    pub links: Vec<ProfileLink>,
}

/// Pack-level metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackMeta {
    pub format_version: u32,
    pub created_at: String,
    /// `algocline` version that produced the pack, for diagnosing shape drift.
    pub alc_version: String,
    /// Sections actually present in `payload/` — not what the caller asked
    /// for. An empty source directory is recorded as absent, so `unpack` can
    /// distinguish "not carried" from "carried but empty".
    pub sections: Vec<PackSection>,
    /// Absolute application directory on the source machine. Kept for
    /// diagnostics: link targets are absolute, and seeing the origin root
    /// explains why they may not resolve here.
    pub app_dir: String,
}

/// A package that unpack re-fetches rather than copies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProfilePackage {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub source: PackageSource,
}

/// A package whose bytes travel inside `payload/packages/`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProfilePayload {
    pub name: String,
    pub reason: PayloadReason,
}

/// Why a package could not be reduced to a source declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PayloadReason {
    /// On disk under `packages/` but absent from `installed.json`,
    /// `alc.toml`, and `alc.local.toml` — the `unregistered_pkg` class that
    /// `alc_pkg_doctor` reports.
    Unregistered,
    /// Manifest records `source = { type = "path" }`; the origin directory
    /// is outside `~/.algocline` and is not assumed to exist on the
    /// destination.
    SourcePath,
    /// Manifest knows the package but not a re-fetchable origin —
    /// [`PackageSource::Installed`] (copied in from somewhere no longer
    /// recorded) or [`PackageSource::Unknown`] (legacy blank entry). There is
    /// nothing to re-fetch *from*, so the bytes travel.
    Opaque,
    /// Reproducible in principle, but the caller asked for bytes anyway.
    Forced,
}

/// A symlink under `packages/`, recorded by definition rather than by target
/// contents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProfileLink {
    pub name: String,
    /// Link target as read from the source machine, verbatim.
    pub target: String,
    pub status_at_pack: LinkStatus,
}

/// Whether a link resolved on the source machine at pack time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LinkStatus {
    /// Target existed when the pack was written.
    Live,
    /// Target was already missing on the source machine. Recorded rather
    /// than dropped so the destination sees the same inventory instead of a
    /// silently shorter one.
    Dangling,
}

// ─── Typed error ────────────────────────────────────────────────

/// Failure modes for reading, writing, and validating a pack profile.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PackProfileError {
    #[error("unknown section '{name}' (known sections: {known})")]
    UnknownSection { name: String, known: String },

    #[error("section 'core' cannot be excluded: a pack without it has no profile to unpack from")]
    CoreExcluded,

    #[error("failed to read pack profile at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse pack profile at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to serialize pack profile: {source}")]
    Serialize {
        #[source]
        source: toml::ser::Error,
    },

    #[error("failed to write pack profile at {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "pack profile at {path} declares format_version {found}, \
         but this build supports up to {supported}: upgrade algocline to unpack it"
    )]
    UnsupportedFormatVersion {
        path: PathBuf,
        found: u32,
        supported: u32,
    },
}

/// Bridge for the service layer's `Result<_, String>` surfaces. One-way: the
/// variant identity is lost, but a future typed service error can absorb this
/// through `#[from]` without churning call sites.
impl From<PackProfileError> for String {
    fn from(e: PackProfileError) -> Self {
        e.to_string()
    }
}

// ─── IO ─────────────────────────────────────────────────────────

/// `profile.toml` inside a pack directory.
pub(crate) fn profile_path(pack_dir: &Path) -> PathBuf {
    pack_dir.join("profile.toml")
}

/// `payload/` inside a pack directory.
pub(crate) fn payload_dir(pack_dir: &Path) -> PathBuf {
    pack_dir.join("payload")
}

impl PackProfile {
    /// Read and validate `profile.toml` from a pack directory.
    ///
    /// Rejects a future `format_version` outright instead of parsing what it
    /// recognises — a partially understood profile would unpack a partially
    /// correct tree, which is worse than refusing.
    pub(crate) fn load(pack_dir: &Path) -> Result<Self, PackProfileError> {
        let path = profile_path(pack_dir);
        let text = std::fs::read_to_string(&path).map_err(|source| PackProfileError::Read {
            path: path.clone(),
            source,
        })?;
        let profile: PackProfile =
            toml::from_str(&text).map_err(|source| PackProfileError::Parse {
                path: path.clone(),
                source,
            })?;

        if profile.pack.format_version > FORMAT_VERSION {
            return Err(PackProfileError::UnsupportedFormatVersion {
                path,
                found: profile.pack.format_version,
                supported: FORMAT_VERSION,
            });
        }
        Ok(profile)
    }

    /// Serialize to `profile.toml` inside a pack directory.
    ///
    /// The parent directory is expected to exist; the pack service creates
    /// the whole tree before this is called.
    pub(crate) fn save(&self, pack_dir: &Path) -> Result<(), PackProfileError> {
        let path = profile_path(pack_dir);
        let text = toml::to_string_pretty(self)
            .map_err(|source| PackProfileError::Serialize { source })?;
        std::fs::write(&path, text).map_err(|source| PackProfileError::Write { path, source })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(set: &BTreeSet<PackSection>) -> Vec<&'static str> {
        set.iter().map(|s| s.as_str()).collect()
    }

    #[test]
    fn default_selection_is_core_cards_evals() {
        let sections = resolve_sections(false, &[], &[]).unwrap();
        assert_eq!(names(&sections), vec!["core", "cards", "evals"]);
    }

    #[test]
    fn all_excludes_logs_hub_cache_and_types() {
        let sections = resolve_sections(true, &[], &[]).unwrap();
        assert_eq!(
            names(&sections),
            vec!["core", "cards", "evals", "nn", "state"]
        );
    }

    #[test]
    fn include_composes_with_all_as_a_union() {
        // The developer-machine case: everything, plus the two sections that
        // `all` deliberately withholds.
        let sections =
            resolve_sections(true, &["logs".to_string(), "types".to_string()], &[]).unwrap();
        assert_eq!(
            names(&sections),
            vec!["core", "cards", "evals", "nn", "state", "logs", "types"]
        );
    }

    #[test]
    fn include_opts_into_heavy_sections_without_all() {
        let sections = resolve_sections(false, &["state".to_string()], &[]).unwrap();
        assert_eq!(names(&sections), vec!["core", "cards", "evals", "state"]);
    }

    #[test]
    fn exclude_wins_over_all() {
        let sections = resolve_sections(true, &[], &["nn".to_string()]).unwrap();
        assert_eq!(names(&sections), vec!["core", "cards", "evals", "state"]);
    }

    #[test]
    fn exclude_wins_over_include() {
        let sections = resolve_sections(false, &["nn".to_string()], &["nn".to_string()]).unwrap();
        assert_eq!(names(&sections), vec!["core", "cards", "evals"]);
    }

    #[test]
    fn core_cannot_be_excluded() {
        let err = resolve_sections(false, &[], &["core".to_string()]).unwrap_err();
        assert!(matches!(err, PackProfileError::CoreExcluded));
    }

    #[test]
    fn unknown_section_is_rejected_with_the_known_list() {
        let err = resolve_sections(false, &["carads".to_string()], &[]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("carads"), "message names the bad input: {msg}");
        assert!(msg.contains("cards"), "message lists valid sections: {msg}");
    }

    #[test]
    fn every_section_except_core_maps_to_a_directory() {
        let app_dir = AppDir::new(PathBuf::from("/tmp/alc-home"));
        for section in PackSection::ALL {
            match section {
                PackSection::Core => assert!(section.dir(&app_dir).is_none()),
                other => {
                    let dir = other.dir(&app_dir).expect("directory-shaped section");
                    assert!(dir.starts_with("/tmp/alc-home"), "{dir:?}");
                }
            }
        }
    }

    #[test]
    fn profile_round_trips_through_toml() {
        let profile = PackProfile {
            pack: PackMeta {
                format_version: FORMAT_VERSION,
                created_at: "2026-08-13T09:12:44Z".to_string(),
                alc_version: "0.46.1".to_string(),
                sections: vec![PackSection::Core, PackSection::Cards],
                app_dir: "/home/x/.algocline".to_string(),
            },
            packages: vec![ProfilePackage {
                name: "ab_mcts".to_string(),
                version: Some("0.3.0".to_string()),
                source: PackageSource::Git {
                    url: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
                    rev: None,
                },
            }],
            payloads: vec![ProfilePayload {
                name: "biz_kernel".to_string(),
                reason: PayloadReason::Unregistered,
            }],
            links: vec![ProfileLink {
                name: "agent_primitive".to_string(),
                target: "/home/x/projects/coding-pipeline/packages/agent_primitive".to_string(),
                status_at_pack: LinkStatus::Live,
            }],
        };

        let dir = tempfile::tempdir().unwrap();
        profile.save(dir.path()).unwrap();
        let loaded = PackProfile::load(dir.path()).unwrap();
        assert_eq!(loaded, profile);
    }

    #[test]
    fn future_format_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let profile = PackProfile {
            pack: PackMeta {
                format_version: FORMAT_VERSION + 1,
                created_at: "2026-08-13T09:12:44Z".to_string(),
                alc_version: "9.9.9".to_string(),
                sections: vec![PackSection::Core],
                app_dir: "/home/x/.algocline".to_string(),
            },
            packages: vec![],
            payloads: vec![],
            links: vec![],
        };
        profile.save(dir.path()).unwrap();

        let err = PackProfile::load(dir.path()).unwrap_err();
        assert!(matches!(
            err,
            PackProfileError::UnsupportedFormatVersion { .. }
        ));
    }

    #[test]
    fn missing_profile_reports_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let err = PackProfile::load(dir.path()).unwrap_err();
        assert!(matches!(err, PackProfileError::Read { .. }));
        assert!(err.to_string().contains("profile.toml"));
    }
}
