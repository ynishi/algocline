//! `AppService::unpack` — lay a pack down onto this machine.
//!
//! Three phases, in order:
//!
//! 1. **Re-fetch** every `[[packages]]` declaration through `pkg_install`.
//! 2. **Expand** `payload/` into the application directory.
//! 3. **Re-link** every `[[links]]` definition against local paths.
//!
//! Phase 3 is the one that routinely cannot complete: link targets are
//! absolute paths into the source machine's checkout layout, and there is no
//! reason for the destination to mirror it. That is reported, not treated as
//! failure — `unresolved` entries name the target so the operator can clone
//! it or re-point the link with `alc_pkg_link`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use algocline_core::AppDir;
use serde::Serialize;

use super::archive::{extract_archive, is_archive_path, verify_archive, ArchiveError};
use super::fs::{
    copy_file, copy_tree, measure_tree, CopyStats, FsError, OverwritePolicy, SkipRecord,
};
use super::profile::{
    payload_dir, PackProfile, PackProfileError, PackSection, ProfileLink, ProfilePackage,
};
use crate::service::manifest::{
    merge_manifest, InstalledManifestStoreError, Manifest, ManifestMergeOutcome,
};
use crate::service::pkg::install::InstallSource;
use crate::service::resolve::packages_dir;
use crate::service::source::PackageSource;
use crate::service::AppService;

// ─── Options ────────────────────────────────────────────────────

/// What to do when the destination already holds something.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnpackMode {
    /// Keep whatever is already here; the local machine wins. Everything not
    /// applied is reported under `skipped`.
    #[default]
    Merge,
    /// The pack wins. `installed.json` is still merged entry by entry rather
    /// than replaced, so packages that exist only here keep their tracking.
    Overwrite,
    /// Write nothing. Runs the same walks and the same link probes, so the
    /// report shows what a real run would do — including which links would
    /// fail to resolve.
    DryRun,
}

/// Parsed from the wire as a string so the MCP surface stays free of this
/// crate's error type; the rejection message names the accepted values.
impl std::str::FromStr for UnpackMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "merge" => Ok(UnpackMode::Merge),
            "overwrite" => Ok(UnpackMode::Overwrite),
            "dry-run" | "dry_run" => Ok(UnpackMode::DryRun),
            other => Err(format!(
                "unknown mode '{other}' (expected: merge, overwrite, dry-run)"
            )),
        }
    }
}

impl UnpackMode {
    fn is_dry_run(self) -> bool {
        self == UnpackMode::DryRun
    }

    /// `DryRun` inherits `Merge`'s conflict handling: a dry run should predict
    /// the conservative default, not a destructive one.
    fn policy(self) -> OverwritePolicy {
        match self {
            UnpackMode::Overwrite => OverwritePolicy::Replace,
            UnpackMode::Merge | UnpackMode::DryRun => OverwritePolicy::KeepExisting,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            UnpackMode::Merge => "merge",
            UnpackMode::Overwrite => "overwrite",
            UnpackMode::DryRun => "dry-run",
        }
    }
}

/// Caller-supplied unpack selection.
#[derive(Debug, Clone, Default)]
pub struct UnpackOptions {
    pub mode: UnpackMode,
    /// When non-empty, expand only these sections (out of those the pack
    /// actually carries). `core` is always expanded.
    pub include: Vec<String>,
    /// Sections to leave in the pack.
    pub exclude: Vec<String>,
}

// ─── Typed error ────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub(crate) enum UnpackError {
    #[error("failed to create symlink {link} -> {target}: {source}")]
    Symlink {
        link: PathBuf,
        target: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to remove existing entry at {path}: {source}")]
    Remove {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read manifest carried in the pack at {path}: {source}")]
    ReadPackedManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse manifest carried in the pack at {path}: {source}")]
    ParsePackedManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to merge installed manifest: {source}")]
    Manifest {
        #[source]
        source: InstalledManifestStoreError,
    },

    #[error(transparent)]
    Fs {
        #[from]
        source: FsError,
    },

    #[error(transparent)]
    Archive {
        #[from]
        source: ArchiveError,
    },

    #[error(transparent)]
    Profile {
        #[from]
        source: PackProfileError,
    },
}

impl From<InstalledManifestStoreError> for UnpackError {
    fn from(source: InstalledManifestStoreError) -> Self {
        UnpackError::Manifest { source }
    }
}

impl From<UnpackError> for String {
    fn from(e: UnpackError) -> Self {
        e.to_string()
    }
}

// ─── Result shape ───────────────────────────────────────────────

/// Something the pack asked for that this machine could not supply.
///
/// Carried on the response, never only logged: an unresolved link is the
/// expected outcome of moving between machines with different checkout
/// layouts, and the operator needs the target path to act on it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct Unresolved {
    /// `link_target_missing` | `source_not_refetchable` | `source_fetch_failed`
    pub kind: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Where a restore's contents came from, when they came from an archive.
///
/// `verified` is `None` when the archive arrived without its `.sha256`
/// sidecar. That is reported as unverified rather than silently treated as
/// checked — "we did not look" and "we looked and it was fine" are different
/// answers to the only question an old snapshot raises.
#[derive(Debug, Clone)]
struct ArchiveOrigin<'a> {
    path: &'a Path,
    verified: Option<String>,
}

#[derive(Debug, Default)]
struct UnpackReport {
    from_source: usize,
    from_payload: usize,
    links: usize,
    stats: CopyStats,
    manifest: Option<ManifestMergeOutcome>,
    /// Packages already installed here, left alone.
    kept_packages: usize,
    /// Links whose destination was already occupied, left alone.
    kept_links: usize,
    /// Entries a caller has to look at: unreadable paths, absent sections.
    /// Bulk "already present" is counted, not listed — see [`CopyStats::kept`].
    skipped: Vec<SkipRecord>,
    unresolved: Vec<Unresolved>,
}

// ─── Service ────────────────────────────────────────────────────

impl AppService {
    /// Restore a pack into this machine's application directory.
    ///
    /// `pack` is the `.tgz` written by `alc_pack`. Its `.sha256` sidecar is
    /// checked before anything is expanded, so an archive that changed after
    /// it was packed is refused rather than half-restored. A directory is also
    /// accepted, for packs written before the archive format.
    ///
    /// Returns a JSON report; `status` is `"partial"` whenever anything landed
    /// in `unresolved`.
    pub async fn unpack(&self, pack: String, opts: UnpackOptions) -> Result<String, String> {
        let app_dir = self.log_config.app_dir();
        Ok(self.unpack_at(&app_dir, Path::new(&pack), &opts).await?)
    }

    /// Resolve `pack` to a directory — verifying and expanding it first when it
    /// names an archive — then restore from it.
    ///
    /// The expansion goes to a temp dir that lives until this returns: the
    /// snapshot is never left lying around unarchived, which is what keeps
    /// "what was in the pack" answerable after the fact.
    async fn unpack_at(
        &self,
        app_dir: &AppDir,
        pack: &Path,
        opts: &UnpackOptions,
    ) -> Result<String, UnpackError> {
        if !is_archive_path(pack) {
            return self.unpack_inner(app_dir, pack, opts, None).await;
        }

        let verified = verify_archive(pack)?;
        let staging = tempfile::tempdir().map_err(|source| FsError::CreateDir {
            path: PathBuf::from("<temp>"),
            source,
        })?;
        let dir = extract_archive(pack, staging.path())?;
        self.unpack_inner(
            app_dir,
            &dir,
            opts,
            Some(ArchiveOrigin {
                path: pack,
                verified,
            }),
        )
        .await
    }

    async fn unpack_inner(
        &self,
        app_dir: &AppDir,
        pack_dir: &Path,
        opts: &UnpackOptions,
        origin: Option<ArchiveOrigin<'_>>,
    ) -> Result<String, UnpackError> {
        let profile = PackProfile::load(pack_dir)?;
        let sections = select_sections(&profile.pack.sections, &opts.include, &opts.exclude)?;
        let mut report = UnpackReport::default();

        // Phase 1 — re-fetch. Needs `self`, so it stays inline.
        self.refetch_declared(app_dir, &profile.packages, opts.mode, &mut report)
            .await;

        // Phase 2 — expand payload.
        expand_payload(app_dir, pack_dir, &sections, opts.mode, &mut report)?;

        // Phase 3 — re-link.
        relink(app_dir, &profile.links, opts.mode, &mut report)?;

        Ok(report_json(pack_dir, origin, opts.mode, &sections, &report))
    }

    /// Install each declared package from its recorded origin.
    ///
    /// A failure here is per-package: one unreachable remote must not abandon
    /// the other hundred, so failures accumulate in `unresolved` and the walk
    /// continues.
    async fn refetch_declared(
        &self,
        app_dir: &AppDir,
        declared: &[ProfilePackage],
        mode: UnpackMode,
        report: &mut UnpackReport,
    ) {
        let pkg_dir = packages_dir(app_dir);
        for pkg in declared {
            if mode != UnpackMode::Overwrite && pkg_dir.join(&pkg.name).exists() {
                report.kept_packages += 1;
                continue;
            }

            let Some(source) = refetch_source(&pkg.source) else {
                report.unresolved.push(Unresolved {
                    kind: "source_not_refetchable".to_string(),
                    name: pkg.name.clone(),
                    target: None,
                    detail: Some(
                        "declaration carries no fetchable origin; re-pack with \
                         packages_only to carry its bytes instead"
                            .to_string(),
                    ),
                });
                continue;
            };

            if mode.is_dry_run() {
                report.from_source += 1;
                continue;
            }

            match self
                .pkg_install_typed(source, Some(pkg.name.clone()), Some(true))
                .await
            {
                Ok(_) => report.from_source += 1,
                Err(detail) => report.unresolved.push(Unresolved {
                    kind: "source_fetch_failed".to_string(),
                    name: pkg.name.clone(),
                    target: None,
                    detail: Some(detail),
                }),
            }
        }
    }
}

/// Which fetchable origin, if any, a recorded source maps to.
fn refetch_source(source: &PackageSource) -> Option<InstallSource> {
    match source {
        PackageSource::Git { url, .. } => Some(InstallSource::GitUrl(url.clone())),
        PackageSource::Bundled {
            collection: Some(url),
        } => Some(InstallSource::GitUrl(url.clone())),
        // A bundled package with no collection URL, a bare `installed`, an
        // `unknown`, or a `path` into the source machine: nothing to fetch.
        _ => None,
    }
}

/// Intersect what the pack carries with what the caller asked for.
///
/// `core` is unconditional — it holds the manifest and the payload packages,
/// without which the rest is incoherent.
fn select_sections(
    carried: &[PackSection],
    include: &[String],
    exclude: &[String],
) -> Result<BTreeSet<PackSection>, PackProfileError> {
    let carried: BTreeSet<PackSection> = carried.iter().copied().collect();

    let mut selected = if include.is_empty() {
        carried.clone()
    } else {
        let mut wanted = BTreeSet::new();
        for name in include {
            let section = PackSection::parse(name)?;
            // Asking for a section the pack does not carry is not an error;
            // it is reported as absent by the expansion walk.
            if carried.contains(&section) {
                wanted.insert(section);
            }
        }
        wanted
    };

    for name in exclude {
        let section = PackSection::parse(name)?;
        selected.remove(&section);
    }

    selected.insert(PackSection::Core);
    Ok(selected)
}

/// Copy `payload/` into the application directory.
fn expand_payload(
    app_dir: &AppDir,
    pack_dir: &Path,
    sections: &BTreeSet<PackSection>,
    mode: UnpackMode,
    report: &mut UnpackReport,
) -> Result<(), UnpackError> {
    let payload = payload_dir(pack_dir);
    let policy = mode.policy();

    // Core: loose files, scenarios, payload packages, manifest merge.
    let core = payload.join("core");
    for (file, dest) in [
        ("config.toml", app_dir.config_toml()),
        ("hub_registries.json", app_dir.hub_registries_json()),
    ] {
        let src = core.join(file);
        if !src.exists() {
            continue;
        }
        if policy == OverwritePolicy::KeepExisting && dest.exists() {
            report.stats.kept += 1;
            continue;
        }
        if mode.is_dry_run() {
            report.stats += CopyStats {
                files: 1,
                bytes: 0,
                kept: 0,
            };
            continue;
        }
        ensure_parent(&dest)?;
        report.stats += copy_file(&src, &dest)?;
    }

    // The manifest is grafted entry by entry, never swapped wholesale.
    let packed_manifest = core.join("installed.json");
    if packed_manifest.exists() {
        let text = std::fs::read_to_string(&packed_manifest).map_err(|source| {
            UnpackError::ReadPackedManifest {
                path: packed_manifest.clone(),
                source,
            }
        })?;
        let incoming: Manifest =
            serde_json::from_str(&text).map_err(|source| UnpackError::ParsePackedManifest {
                path: packed_manifest.clone(),
                source,
            })?;
        if !mode.is_dry_run() {
            report.manifest = Some(merge_manifest(
                app_dir,
                &incoming,
                mode == UnpackMode::Overwrite,
            )?);
        }
    }

    let copy_or_measure = |src: &Path, dst: &Path, report: &mut UnpackReport| {
        if !src.exists() {
            return Ok(());
        }
        let stats = if mode.is_dry_run() {
            measure_tree(src, dst, policy, &mut report.skipped)?
        } else {
            copy_tree(src, dst, policy, &mut report.skipped)?
        };
        report.stats += stats;
        Ok::<(), UnpackError>(())
    };

    copy_or_measure(&payload.join("scenarios"), &app_dir.scenarios_dir(), report)?;

    let packed_packages = payload.join("packages");
    if packed_packages.exists() {
        let pkg_dir = packages_dir(app_dir);
        for entry in std::fs::read_dir(&packed_packages).map_err(|source| FsError::ReadDir {
            path: packed_packages.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| FsError::ReadDir {
                path: packed_packages.clone(),
                source,
            })?;
            let name = entry.file_name();
            let dest = pkg_dir.join(&name);
            let existed = dest.exists();
            copy_or_measure(&entry.path(), &dest, report)?;
            // A package counts as restored when the pack supplied it, whether
            // or not every file inside was new.
            if !existed || mode == UnpackMode::Overwrite {
                report.from_payload += 1;
            }
        }
    }

    // Directory-shaped sections.
    for section in sections {
        let Some(dest) = section.dir(app_dir) else {
            continue;
        };
        copy_or_measure(&payload.join(section.as_str()), &dest, report)?;
    }

    Ok(())
}

/// Re-create the packed symlinks, reporting the ones whose target is absent.
fn relink(
    app_dir: &AppDir,
    links: &[ProfileLink],
    mode: UnpackMode,
    report: &mut UnpackReport,
) -> Result<(), UnpackError> {
    if links.is_empty() {
        return Ok(());
    }
    let pkg_dir = packages_dir(app_dir);
    if !mode.is_dry_run() {
        std::fs::create_dir_all(&pkg_dir).map_err(|source| FsError::CreateDir {
            path: pkg_dir.clone(),
            source,
        })?;
    }

    for link in links {
        let dest = pkg_dir.join(&link.name);
        let target = Path::new(&link.target);

        if !target.exists() {
            report.unresolved.push(Unresolved {
                kind: "link_target_missing".to_string(),
                name: link.name.clone(),
                target: Some(link.target.clone()),
                detail: Some(
                    "clone the repository to this path, or re-point the link with alc_pkg_link"
                        .to_string(),
                ),
            });
            continue;
        }

        // symlink_metadata so an existing dangling link counts as occupied.
        let occupied = dest.symlink_metadata().is_ok();
        if occupied && mode != UnpackMode::Overwrite {
            report.kept_links += 1;
            continue;
        }

        if mode.is_dry_run() {
            report.links += 1;
            continue;
        }

        if occupied {
            remove_path(&dest)?;
        }
        std::os::unix::fs::symlink(target, &dest).map_err(|source| UnpackError::Symlink {
            link: dest.clone(),
            target: link.target.clone(),
            source,
        })?;
        report.links += 1;
    }

    Ok(())
}

fn remove_path(path: &Path) -> Result<(), UnpackError> {
    let meta = path
        .symlink_metadata()
        .map_err(|source| UnpackError::Remove {
            path: path.to_path_buf(),
            source,
        })?;
    let result = if meta.file_type().is_symlink() || !meta.is_dir() {
        std::fs::remove_file(path)
    } else {
        std::fs::remove_dir_all(path)
    };
    result.map_err(|source| UnpackError::Remove {
        path: path.to_path_buf(),
        source,
    })
}

fn ensure_parent(path: &Path) -> Result<(), UnpackError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| FsError::CreateDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn report_json(
    pack_dir: &Path,
    origin: Option<ArchiveOrigin<'_>>,
    mode: UnpackMode,
    sections: &BTreeSet<PackSection>,
    report: &UnpackReport,
) -> String {
    let manifest = report.manifest.as_ref().map(|m| {
        serde_json::json!({
            "added": m.added.len(),
            "updated": m.updated.len(),
            "kept": m.kept.len(),
        })
    });

    // An archive restore reports what it verified. `verified: false` means the
    // `.sha256` was absent, not that the check failed — a failed check does not
    // reach here at all.
    let source = match &origin {
        Some(o) => serde_json::json!({
            "kind": "archive",
            "path": o.path.display().to_string(),
            "verified": o.verified.is_some(),
            "sha256": o.verified,
        }),
        None => serde_json::json!({
            "kind": "directory",
            "path": pack_dir.display().to_string(),
            "verified": false,
        }),
    };

    serde_json::json!({
        "status": if report.unresolved.is_empty() { "ok" } else { "partial" },
        "mode": mode.as_str(),
        "dry_run": mode.is_dry_run(),
        "source": source,
        "sections": sections.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "restored": {
            "from_source": report.from_source,
            "from_payload": report.from_payload,
            "links": report.links,
        },
        // Counted, not listed: restoring onto a machine that already holds
        // most of the pack is normal, and one record per file overflows the
        // response (1,799 files produced 290,755 characters against a real
        // application directory).
        "kept_existing": {
            "files": report.stats.kept,
            "packages": report.kept_packages,
            "links": report.kept_links,
        },
        "manifest": manifest,
        "total_files": report.stats.files,
        "total_bytes": report.stats.bytes,
        "skipped": report.skipped,
        "unresolved": report.unresolved,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::pack::profile::{LinkStatus, PackMeta, FORMAT_VERSION};

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// A pack carrying: one payload package, one card, a manifest entry, and
    /// two links (one resolvable here, one not).
    fn packed(tmp: &Path) -> (PathBuf, PathBuf) {
        let pack = tmp.join("in.alcpack");
        let payload = pack.join("payload");

        write(
            &payload.join("core/config.toml"),
            "[setting.card]\nrun = true\n",
        );
        write(
            &payload.join("core/installed.json"),
            r#"{"packages":{"packed_pkg":{"version":"2.0.0",
               "source":{"type":"installed"},
               "installed_at":"2026-01-01T00:00:00Z",
               "updated_at":"2026-01-01T00:00:00Z"}}}"#,
        );
        write(&payload.join("packages/packed_pkg/init.lua"), "-- packed\n");
        write(&payload.join("cards/c.toml"), "x = 1\n");

        // Exists on this machine, so the link is resolvable.
        let here = tmp.join("checkout/live_pkg");
        write(&here.join("init.lua"), "-- live\n");

        let profile = PackProfile {
            pack: PackMeta {
                format_version: FORMAT_VERSION,
                created_at: "2026-08-13T00:00:00Z".to_string(),
                alc_version: "0.46.1".to_string(),
                sections: vec![PackSection::Core, PackSection::Cards],
                app_dir: "/elsewhere/.algocline".to_string(),
            },
            packages: vec![],
            payloads: vec![],
            links: vec![
                ProfileLink {
                    name: "live_link".to_string(),
                    target: here.display().to_string(),
                    status_at_pack: LinkStatus::Live,
                },
                ProfileLink {
                    name: "absent_link".to_string(),
                    target: tmp.join("never/cloned").display().to_string(),
                    status_at_pack: LinkStatus::Live,
                },
            ],
        };
        profile.save(&pack).unwrap();
        (pack, tmp.join("home"))
    }

    async fn run(pack: &Path, home: &Path, opts: UnpackOptions) -> serde_json::Value {
        use crate::service::AppConfig;
        use std::sync::Arc;

        let executor = Arc::new(
            algocline_engine::Executor::new(vec![home.join("packages")])
                .await
                .unwrap(),
        );
        let config = AppConfig::default()
            .with_app_dir(home.to_path_buf())
            .with_log_disabled();
        let service = AppService::new(executor, config, vec![]);

        let app_dir = AppDir::new(home.to_path_buf());
        // Through `unpack_at`, so archive paths take the verify + expand route
        // and directories take the legacy one — the same dispatch the MCP tool
        // hits.
        let json = service
            .unpack_at(&app_dir, pack, &opts)
            .await
            .expect("unpack");
        serde_json::from_str(&json).unwrap()
    }

    /// The path a real migration takes: pack writes an archive, unpack
    /// verifies it against its digest and restores from the expansion.
    #[tokio::test]
    async fn restores_from_a_verified_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        let archive = tmp.path().join("snap.tgz");
        let (digest, _) =
            crate::service::pack::archive::write_archive(&pack, &archive, "snap").unwrap();

        let v = run(&archive, &home, UnpackOptions::default()).await;

        assert_eq!(v["source"]["kind"], "archive");
        assert_eq!(v["source"]["verified"], true);
        assert_eq!(v["source"]["sha256"], digest);
        assert!(home.join("packages/packed_pkg/init.lua").exists());
        assert!(home.join("cards/c.toml").exists());
    }

    /// An archive that changed after packing is refused before anything is
    /// written — a half-restored application directory is worse than none.
    #[tokio::test]
    async fn refuses_an_archive_that_changed_after_packing() {
        use crate::service::AppConfig;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        let archive = tmp.path().join("snap.tgz");
        crate::service::pack::archive::write_archive(&pack, &archive, "snap").unwrap();

        let mut bytes = std::fs::read(&archive).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&archive, &bytes).unwrap();

        let executor = Arc::new(
            algocline_engine::Executor::new(vec![home.join("packages")])
                .await
                .unwrap(),
        );
        let config = AppConfig::default()
            .with_app_dir(home.to_path_buf())
            .with_log_disabled();
        let service = AppService::new(executor, config, vec![]);
        let app_dir = AppDir::new(home.to_path_buf());

        let err = service
            .unpack_at(&app_dir, &archive, &UnpackOptions::default())
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("changed after it was packed"),
            "{err}"
        );
        assert!(
            !home.join("packages/packed_pkg").exists(),
            "nothing may land from an archive that failed verification"
        );
    }

    /// An archive carried without its sidecar restores, but says it was not
    /// verified. "We did not check" must not read as "we checked and it passed".
    #[tokio::test]
    async fn archive_without_a_sidecar_reports_unverified() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        let archive = tmp.path().join("snap.tgz");
        crate::service::pack::archive::write_archive(&pack, &archive, "snap").unwrap();
        std::fs::remove_file(crate::service::pack::archive::checksum_path(&archive)).unwrap();

        let v = run(&archive, &home, UnpackOptions::default()).await;

        assert_eq!(v["source"]["kind"], "archive");
        assert_eq!(v["source"]["verified"], false);
        assert!(home.join("packages/packed_pkg/init.lua").exists());
    }

    /// Packs written before the archive format still restore.
    #[tokio::test]
    async fn a_directory_pack_still_restores() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());

        let v = run(&pack, &home, UnpackOptions::default()).await;

        assert_eq!(v["source"]["kind"], "directory");
        assert_eq!(v["source"]["verified"], false);
        assert!(home.join("packages/packed_pkg/init.lua").exists());
    }

    #[tokio::test]
    async fn missing_link_target_is_reported_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());

        let v = run(&pack, &home, UnpackOptions::default()).await;

        assert_eq!(v["status"], "partial");
        assert_eq!(v["restored"]["links"], 1, "the resolvable link is created");
        let unresolved = v["unresolved"].as_array().unwrap();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0]["kind"], "link_target_missing");
        assert_eq!(unresolved[0]["name"], "absent_link");
        assert!(
            unresolved[0]["target"]
                .as_str()
                .unwrap()
                .contains("never/cloned"),
            "the report names the path to clone"
        );

        // The one that could be linked really is a symlink.
        let live = home.join("packages/live_link");
        assert!(live.symlink_metadata().unwrap().file_type().is_symlink());
        assert!(live.join("init.lua").exists());
    }

    #[tokio::test]
    async fn payload_lands_and_manifest_is_grafted() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        // A package known only to this machine must survive the merge.
        write(
            &home.join("installed.json"),
            r#"{"packages":{"local_only":{"version":"1.0.0",
               "source":{"type":"installed"},
               "installed_at":"2026-01-01T00:00:00Z",
               "updated_at":"2026-01-01T00:00:00Z"}}}"#,
        );

        let v = run(&pack, &home, UnpackOptions::default()).await;

        assert!(home.join("packages/packed_pkg/init.lua").exists());
        assert!(home.join("cards/c.toml").exists());
        assert_eq!(v["restored"]["from_payload"], 1);
        assert_eq!(v["manifest"]["added"], 1);

        // Read back through the store, not the file: Inv-4 keeps
        // `installed.json` IO inside `service::manifest`.
        let manifest = crate::service::manifest::load_manifest(&AppDir::new(home.clone()))
            .expect("load manifest");
        assert!(
            manifest.packages.contains_key("local_only"),
            "a wholesale replace would have dropped this"
        );
        assert!(manifest.packages.contains_key("packed_pkg"));
    }

    #[tokio::test]
    async fn merge_mode_keeps_local_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        write(&home.join("cards/c.toml"), "x = 99\n");

        let v = run(&pack, &home, UnpackOptions::default()).await;

        assert_eq!(
            std::fs::read_to_string(home.join("cards/c.toml")).unwrap(),
            "x = 99\n"
        );
        assert_eq!(v["kept_existing"]["files"], 1, "report: {v}");
        // Regression guard: listing every already-present file overflowed the
        // MCP response against a real application directory (1,799 files,
        // 290,755 characters). Bulk skips must stay a count.
        assert!(
            v["skipped"].as_array().unwrap().is_empty(),
            "'already present' must not be listed per file: {v}"
        );
    }

    #[tokio::test]
    async fn overwrite_mode_replaces_local_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());
        write(&home.join("cards/c.toml"), "x = 99\n");

        run(
            &pack,
            &home,
            UnpackOptions {
                mode: UnpackMode::Overwrite,
                ..Default::default()
            },
        )
        .await;

        assert_eq!(
            std::fs::read_to_string(home.join("cards/c.toml")).unwrap(),
            "x = 1\n"
        );
    }

    #[tokio::test]
    async fn dry_run_writes_nothing_but_still_probes_links() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());

        let v = run(
            &pack,
            &home,
            UnpackOptions {
                mode: UnpackMode::DryRun,
                ..Default::default()
            },
        )
        .await;

        assert_eq!(v["dry_run"], true);
        assert_eq!(v["status"], "partial", "the absent link is still detected");
        assert_eq!(v["restored"]["links"], 1);
        assert!(v["total_files"].as_u64().unwrap() > 0, "it counts the work");

        assert!(!home.join("packages/packed_pkg").exists());
        assert!(!home.join("cards/c.toml").exists());
        assert!(!home.join("packages/live_link").exists());
    }

    #[tokio::test]
    async fn exclude_leaves_a_section_in_the_pack() {
        let tmp = tempfile::tempdir().unwrap();
        let (pack, home) = packed(tmp.path());

        let v = run(
            &pack,
            &home,
            UnpackOptions {
                exclude: vec!["cards".to_string()],
                ..Default::default()
            },
        )
        .await;

        assert!(!home.join("cards/c.toml").exists());
        // Core still lands.
        assert!(home.join("packages/packed_pkg/init.lua").exists());
        let sections: Vec<&str> = v["sections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert_eq!(sections, vec!["core"]);
    }

    #[test]
    fn core_survives_an_include_that_omits_it() {
        let carried = vec![PackSection::Core, PackSection::Cards, PackSection::Nn];
        let selected = select_sections(&carried, &["cards".to_string()], &[]).unwrap();
        assert!(selected.contains(&PackSection::Core));
        assert!(selected.contains(&PackSection::Cards));
        assert!(!selected.contains(&PackSection::Nn));
    }

    #[test]
    fn including_a_section_the_pack_lacks_is_not_an_error() {
        let carried = vec![PackSection::Core];
        let selected = select_sections(&carried, &["nn".to_string()], &[]).unwrap();
        assert_eq!(
            selected.into_iter().collect::<Vec<_>>(),
            vec![PackSection::Core]
        );
    }

    #[test]
    fn only_git_and_bundled_urls_are_refetchable() {
        assert!(matches!(
            refetch_source(&PackageSource::Git {
                url: "https://example.invalid/r".to_string(),
                rev: None
            }),
            Some(InstallSource::GitUrl(_))
        ));
        assert!(matches!(
            refetch_source(&PackageSource::Bundled {
                collection: Some("https://example.invalid/c".to_string())
            }),
            Some(InstallSource::GitUrl(_))
        ));
        // Nothing to fetch from: these must travel as payload instead.
        assert!(refetch_source(&PackageSource::Bundled { collection: None }).is_none());
        assert!(refetch_source(&PackageSource::Installed).is_none());
        assert!(refetch_source(&PackageSource::Unknown).is_none());
        assert!(refetch_source(&PackageSource::Path {
            path: "/somewhere".to_string()
        })
        .is_none());
    }

    #[test]
    fn mode_parse_rejects_unknown_values() {
        use std::str::FromStr;

        assert_eq!(UnpackMode::from_str("merge").unwrap(), UnpackMode::Merge);
        assert_eq!(UnpackMode::from_str("dry-run").unwrap(), UnpackMode::DryRun);
        assert_eq!(UnpackMode::from_str("dry_run").unwrap(), UnpackMode::DryRun);

        let err = UnpackMode::from_str("force").unwrap_err();
        assert!(err.contains("force"), "names the bad input: {err}");
        assert!(err.contains("overwrite"), "lists valid modes: {err}");
    }
}
