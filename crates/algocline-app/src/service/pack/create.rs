//! `AppService::pack` — write a portable snapshot of `~/.algocline`.
//!
//! The filesystem walk over `packages/` is the primary input, not
//! `installed.json`. `pkg_link` records nothing in the manifest, and packages
//! can land on disk by other routes (the `unregistered_pkg` class reported by
//! `alc_pkg_doctor`), so a manifest-driven pack silently drops them. The
//! manifest is consulted only to decide *how* each directory found on disk
//! should travel — as a re-fetchable declaration or as bytes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use algocline_core::AppDir;

use super::fs::{copy_file, copy_tree, CopyStats, FsError, OverwritePolicy, SkipRecord};
use super::profile::{
    payload_dir, resolve_sections, LinkStatus, PackMeta, PackProfile, PackProfileError,
    PackSection, PayloadReason, ProfileLink, ProfilePackage, ProfilePayload, FORMAT_VERSION,
};
use crate::service::manifest::{load_manifest, now_iso8601, InstalledManifestStoreError, Manifest};
use crate::service::resolve::packages_dir;
use crate::service::source::PackageSource;
use crate::service::AppService;

// ─── Options ────────────────────────────────────────────────────

/// Caller-supplied pack selection.
///
/// Section selection (`all` / `include` / `exclude`) and package selection
/// (`packages_only` / `packages_exclude`) are separate axes and deliberately
/// carry separate names — one picks directories of the application dir, the
/// other picks entries inside `packages/`.
#[derive(Debug, Clone, Default)]
pub struct PackOptions {
    pub all: bool,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    /// When non-empty, only these package names are considered at all.
    pub packages_only: Vec<String>,
    /// Package names dropped from the pack entirely.
    pub packages_exclude: Vec<String>,
}

// ─── Typed error ────────────────────────────────────────────────

/// Failure modes for writing a pack.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PackError {
    #[error(
        "pack destination {path} already exists: choose a fresh path \
         (packing never overwrites an existing directory)"
    )]
    OutDirExists { path: PathBuf },

    #[error("failed to read link {path}: {source}")]
    ReadLink {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read directory {path}: {source}")]
    ReadDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to load installed manifest: {source}")]
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
    Profile {
        #[from]
        source: PackProfileError,
    },
}

impl From<InstalledManifestStoreError> for PackError {
    fn from(source: InstalledManifestStoreError) -> Self {
        PackError::Manifest { source }
    }
}

impl From<PackError> for String {
    fn from(e: PackError) -> Self {
        e.to_string()
    }
}

/// Outcome of classifying everything under `packages/`.
#[derive(Debug, Default)]
struct Classified {
    declared: Vec<ProfilePackage>,
    payloads: Vec<ProfilePayload>,
    links: Vec<ProfileLink>,
    skipped: Vec<SkipRecord>,
}

// ─── Service ────────────────────────────────────────────────────

impl AppService {
    /// Write a pack directory describing (and partially containing) the
    /// current application directory.
    ///
    /// Returns a JSON summary: section list, per-class package counts, link
    /// liveness, byte totals, and everything skipped.
    pub async fn pack(&self, out_dir: String, opts: PackOptions) -> Result<String, String> {
        let app_dir = self.log_config.app_dir();
        Ok(pack_into(&app_dir, Path::new(&out_dir), &opts)?)
    }
}

fn pack_into(app_dir: &AppDir, out_dir: &Path, opts: &PackOptions) -> Result<String, PackError> {
    let sections = resolve_sections(opts.all, &opts.include, &opts.exclude)?;

    // Refuse to write into an existing directory. Packing is a bulk copy; an
    // accidental merge into an unrelated tree is not something the caller can
    // easily undo.
    if out_dir.exists() {
        return Err(PackError::OutDirExists {
            path: out_dir.to_path_buf(),
        });
    }
    let payload = payload_dir(out_dir);
    std::fs::create_dir_all(&payload).map_err(|source| FsError::CreateDir {
        path: payload.clone(),
        source,
    })?;

    let manifest = load_manifest(app_dir)?;
    let mut classified = classify_packages(&packages_dir(app_dir), &manifest, opts)?;

    let mut stats_by_section: Vec<(PackSection, CopyStats)> = Vec::new();
    let mut present: Vec<PackSection> = Vec::new();

    // Core: the loose files plus the payload packages.
    let core_stats = copy_core(
        app_dir,
        &payload,
        &classified.payloads,
        &mut classified.skipped,
    )?;
    stats_by_section.push((PackSection::Core, core_stats));
    present.push(PackSection::Core);

    // Directory-shaped sections.
    for section in sections.iter().copied() {
        let Some(src) = section.dir(app_dir) else {
            continue; // Core, handled above.
        };
        if !src.exists() {
            classified.skipped.push(SkipRecord {
                path: src.display().to_string(),
                reason: format!("section '{}' absent on source machine", section.as_str()),
            });
            continue;
        }
        let dst = payload.join(section.as_str());
        let stats = copy_tree(
            &src,
            &dst,
            OverwritePolicy::Replace,
            &mut classified.skipped,
        )?;
        stats_by_section.push((section, stats));
        present.push(section);
    }

    let profile = PackProfile {
        pack: PackMeta {
            format_version: FORMAT_VERSION,
            created_at: now_iso8601(),
            alc_version: env!("CARGO_PKG_VERSION").to_string(),
            sections: present.clone(),
            app_dir: app_dir.root().display().to_string(),
        },
        packages: classified.declared.clone(),
        payloads: classified.payloads.clone(),
        links: classified.links.clone(),
    };
    profile.save(out_dir)?;

    Ok(summary_json(
        out_dir,
        &present,
        &classified,
        &stats_by_section,
    ))
}

/// Sort every entry under `packages/` into declaration, payload, or link.
fn classify_packages(
    pkg_dir: &Path,
    manifest: &Manifest,
    opts: &PackOptions,
) -> Result<Classified, PackError> {
    let mut out = Classified::default();
    if !pkg_dir.exists() {
        return Ok(out);
    }

    let only: BTreeSet<&str> = opts.packages_only.iter().map(String::as_str).collect();
    let excluded: BTreeSet<&str> = opts.packages_exclude.iter().map(String::as_str).collect();

    let entries = std::fs::read_dir(pkg_dir).map_err(|source| PackError::ReadDir {
        path: pkg_dir.to_path_buf(),
        source,
    })?;

    let mut names: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| PackError::ReadDir {
            path: pkg_dir.to_path_buf(),
            source,
        })?;
        let name = entry.file_name().to_string_lossy().to_string();
        names.push((name, entry.path()));
    }
    // Stable output regardless of directory iteration order.
    names.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, path) in names {
        if !only.is_empty() && !only.contains(name.as_str()) {
            continue;
        }
        if excluded.contains(name.as_str()) {
            out.skipped.push(SkipRecord {
                path: path.display().to_string(),
                reason: "excluded by packages_exclude".to_string(),
            });
            continue;
        }

        // symlink_metadata does not follow, so a dangling link is still
        // detected as a link rather than vanishing.
        let meta = match path.symlink_metadata() {
            Ok(m) => m,
            Err(e) => {
                out.skipped.push(SkipRecord {
                    path: path.display().to_string(),
                    reason: format!("cannot stat: {e}"),
                });
                continue;
            }
        };

        if meta.file_type().is_symlink() {
            let target = path.read_link().map_err(|source| PackError::ReadLink {
                path: path.clone(),
                source,
            })?;
            out.links.push(ProfileLink {
                name,
                target: target.display().to_string(),
                // `exists()` follows the link: false means the target is
                // already gone on the source machine.
                status_at_pack: if path.exists() {
                    LinkStatus::Live
                } else {
                    LinkStatus::Dangling
                },
            });
            continue;
        }

        if !meta.is_dir() {
            out.skipped.push(SkipRecord {
                path: path.display().to_string(),
                reason: "not a directory or symlink".to_string(),
            });
            continue;
        }

        match manifest.packages.get(&name) {
            // Re-fetchable: carry the declaration only.
            Some(entry)
                if matches!(
                    entry.source,
                    PackageSource::Git { .. } | PackageSource::Bundled { .. }
                ) =>
            {
                out.declared.push(ProfilePackage {
                    name,
                    version: entry.version.clone(),
                    source: entry.source.clone(),
                });
            }
            // Known to the manifest, but with no origin to re-fetch from.
            Some(entry) => {
                let reason = match entry.source {
                    PackageSource::Path { .. } => PayloadReason::SourcePath,
                    _ => PayloadReason::Opaque,
                };
                out.payloads.push(ProfilePayload { name, reason });
            }
            // On disk, unknown to the manifest.
            None => out.payloads.push(ProfilePayload {
                name,
                reason: PayloadReason::Unregistered,
            }),
        }
    }

    Ok(out)
}

/// Copy the core section: the three loose files, `scenarios/`, and the
/// packages that must travel as bytes.
fn copy_core(
    app_dir: &AppDir,
    payload: &Path,
    payloads: &[ProfilePayload],
    skipped: &mut Vec<SkipRecord>,
) -> Result<CopyStats, PackError> {
    let mut stats = CopyStats::default();
    let core = payload.join("core");
    std::fs::create_dir_all(&core).map_err(|source| FsError::CreateDir {
        path: core.clone(),
        source,
    })?;

    for src in [
        app_dir.config_toml(),
        app_dir.hub_registries_json(),
        app_dir.installed_json(),
    ] {
        let Some(file_name) = src.file_name() else {
            continue;
        };
        if !src.exists() {
            skipped.push(SkipRecord {
                path: src.display().to_string(),
                reason: "absent on source machine".to_string(),
            });
            continue;
        }
        stats += copy_file(&src, &core.join(file_name))?;
    }

    let scenarios = app_dir.scenarios_dir();
    if scenarios.exists() {
        stats += copy_tree(
            &scenarios,
            &payload.join("scenarios"),
            OverwritePolicy::Replace,
            skipped,
        )?;
    }

    if !payloads.is_empty() {
        let pkg_root = payload.join("packages");
        std::fs::create_dir_all(&pkg_root).map_err(|source| FsError::CreateDir {
            path: pkg_root.clone(),
            source,
        })?;
        let src_root = packages_dir(app_dir);
        for entry in payloads {
            stats += copy_tree(
                &src_root.join(&entry.name),
                &pkg_root.join(&entry.name),
                OverwritePolicy::Replace,
                skipped,
            )?;
        }
    }

    Ok(stats)
}

fn summary_json(
    out_dir: &Path,
    sections: &[PackSection],
    classified: &Classified,
    stats: &[(PackSection, CopyStats)],
) -> String {
    let live = classified
        .links
        .iter()
        .filter(|l| l.status_at_pack == LinkStatus::Live)
        .count();
    let dangling = classified.links.len() - live;

    let mut by_section = serde_json::Map::new();
    let mut total = CopyStats::default();
    for (section, s) in stats {
        by_section.insert(
            section.as_str().to_string(),
            serde_json::json!({ "files": s.files, "bytes": s.bytes }),
        );
        total += *s;
    }

    serde_json::json!({
        "pack_dir": out_dir.display().to_string(),
        "format_version": FORMAT_VERSION,
        "sections": sections.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "packages": {
            "declared": classified.declared.len(),
            "payload": classified.payloads.len(),
            "linked": classified.links.len(),
        },
        "links": { "live": live, "dangling": dangling },
        "bytes_by_section": serde_json::Value::Object(by_section),
        "total_files": total.files,
        "total_bytes": total.bytes,
        "skipped": classified.skipped,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::manifest::ManifestEntry;
    use std::collections::BTreeMap;

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn entry(source: PackageSource) -> ManifestEntry {
        ManifestEntry {
            version: Some("1.0.0".to_string()),
            source,
            installed_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            pkg_type: None,
        }
    }

    /// Build an application dir holding one package of each classification.
    fn fixture() -> (tempfile::TempDir, AppDir, Manifest) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("home");
        let app_dir = AppDir::new(root.clone());
        let pkgs = root.join("packages");

        for name in ["from_git", "from_path", "opaque", "unregistered"] {
            write(&pkgs.join(name).join("init.lua"), "-- pkg\n");
        }
        write(&root.join("config.toml"), "[setting.card]\nrun = true\n");
        write(&root.join("installed.json"), "{\"packages\":{}}");
        write(&root.join("scenarios").join("s.lua"), "-- scenario\n");
        write(&root.join("cards").join("c.toml"), "x = 1\n");
        write(&root.join("logs").join("l.txt"), "log line\n");

        // A live link and a dangling one.
        let outside = tmp.path().join("outside").join("linked_pkg");
        write(&outside.join("init.lua"), "-- linked\n");
        std::os::unix::fs::symlink(&outside, pkgs.join("live_link")).unwrap();
        std::os::unix::fs::symlink(tmp.path().join("gone"), pkgs.join("dead_link")).unwrap();

        let mut packages = BTreeMap::new();
        packages.insert(
            "from_git".to_string(),
            entry(PackageSource::Git {
                url: "https://example.invalid/repo".to_string(),
                rev: None,
            }),
        );
        packages.insert(
            "from_path".to_string(),
            entry(PackageSource::Path {
                path: "/somewhere/else".to_string(),
            }),
        );
        packages.insert("opaque".to_string(), entry(PackageSource::Installed));

        (tmp, app_dir, Manifest { packages })
    }

    #[test]
    fn classification_splits_by_source_kind() {
        let (_tmp, app_dir, manifest) = fixture();
        let c =
            classify_packages(&packages_dir(&app_dir), &manifest, &PackOptions::default()).unwrap();

        assert_eq!(
            c.declared
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["from_git"]
        );
        let payloads: Vec<_> = c
            .payloads
            .iter()
            .map(|p| (p.name.as_str(), p.reason))
            .collect();
        assert_eq!(
            payloads,
            vec![
                ("from_path", PayloadReason::SourcePath),
                ("opaque", PayloadReason::Opaque),
                ("unregistered", PayloadReason::Unregistered),
            ]
        );
    }

    #[test]
    fn dangling_links_are_recorded_not_dropped() {
        let (_tmp, app_dir, manifest) = fixture();
        let c =
            classify_packages(&packages_dir(&app_dir), &manifest, &PackOptions::default()).unwrap();

        let statuses: Vec<_> = c
            .links
            .iter()
            .map(|l| (l.name.as_str(), l.status_at_pack))
            .collect();
        assert_eq!(
            statuses,
            vec![
                ("dead_link", LinkStatus::Dangling),
                ("live_link", LinkStatus::Live),
            ]
        );
    }

    #[test]
    fn packages_exclude_drops_and_records() {
        let (_tmp, app_dir, manifest) = fixture();
        let opts = PackOptions {
            packages_exclude: vec!["unregistered".to_string()],
            ..Default::default()
        };
        let c = classify_packages(&packages_dir(&app_dir), &manifest, &opts).unwrap();

        assert!(c.payloads.iter().all(|p| p.name != "unregistered"));
        assert_eq!(c.skipped.len(), 1);
        assert!(c.skipped[0].reason.contains("packages_exclude"));
    }

    #[test]
    fn packages_only_narrows_every_class() {
        let (_tmp, app_dir, manifest) = fixture();
        let opts = PackOptions {
            packages_only: vec!["from_git".to_string(), "live_link".to_string()],
            ..Default::default()
        };
        let c = classify_packages(&packages_dir(&app_dir), &manifest, &opts).unwrap();

        assert_eq!(c.declared.len(), 1);
        assert_eq!(c.payloads.len(), 0);
        assert_eq!(c.links.len(), 1);
    }

    #[test]
    fn default_pack_carries_core_and_cards_but_not_logs() {
        let (tmp, app_dir, _) = fixture();
        // The real manifest file is what pack_into reads.
        let manifest_json = r#"{"packages":{"from_git":{"version":"1.0.0",
            "source":{"type":"git","url":"https://example.invalid/repo","rev":null},
            "installed_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}}}"#;
        write(&app_dir.installed_json(), manifest_json);

        let out = tmp.path().join("out.alcpack");
        let json = pack_into(&app_dir, &out, &PackOptions::default()).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(v["packages"]["declared"], 1);
        assert_eq!(v["packages"]["linked"], 2);
        assert_eq!(v["links"]["dangling"], 1);

        assert!(out.join("profile.toml").exists());
        assert!(out.join("payload/core/config.toml").exists());
        assert!(out.join("payload/scenarios/s.lua").exists());
        assert!(out.join("payload/cards/c.toml").exists());
        assert!(
            !out.join("payload/logs").exists(),
            "logs must not travel by default"
        );
        // Payload packages carry bytes; declared ones do not.
        assert!(out.join("payload/packages/unregistered/init.lua").exists());
        assert!(!out.join("payload/packages/from_git").exists());
    }

    #[test]
    fn logs_travel_only_when_explicitly_included() {
        let (tmp, app_dir, _) = fixture();
        write(&app_dir.installed_json(), r#"{"packages":{}}"#);

        let out = tmp.path().join("with-logs.alcpack");
        let opts = PackOptions {
            all: true,
            include: vec!["logs".to_string()],
            ..Default::default()
        };
        pack_into(&app_dir, &out, &opts).unwrap();

        assert!(out.join("payload/logs/l.txt").exists());
    }

    #[test]
    fn existing_destination_is_refused() {
        let (tmp, app_dir, _) = fixture();
        write(&app_dir.installed_json(), r#"{"packages":{}}"#);

        let out = tmp.path().join("taken.alcpack");
        std::fs::create_dir_all(&out).unwrap();

        let err = pack_into(&app_dir, &out, &PackOptions::default()).unwrap_err();
        assert!(matches!(err, PackError::OutDirExists { .. }));
    }

    #[test]
    fn absent_section_is_reported_as_skipped() {
        let (tmp, app_dir, _) = fixture();
        write(&app_dir.installed_json(), r#"{"packages":{}}"#);

        // `evals/` was never created by the fixture.
        let out = tmp.path().join("no-evals.alcpack");
        let json = pack_into(&app_dir, &out, &PackOptions::default()).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        let skipped = v["skipped"].as_array().unwrap();
        assert!(
            skipped
                .iter()
                .any(|s| s["reason"].as_str().unwrap_or("").contains("evals")),
            "absent section is surfaced on the response: {skipped:?}"
        );
        // And it is not claimed as present in the profile.
        let sections: Vec<&str> = v["sections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert!(!sections.contains(&"evals"));
    }
}
