//! Installed-packages manifest (`~/.algocline/installed.json`).
//!
//! Records package name, version, source, and install/update timestamps.
//! Written on `pkg_install` success, pruned on `pkg_remove`.
//! Read by `pkg_list` to display version tracking info.

use std::collections::BTreeMap;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::Mutex;

use algocline_core::AppDir;
use serde::{Deserialize, Serialize};

use super::source::PackageSource;

/// Per-package record in the manifest.
///
/// The `source` field is typed ([`PackageSource`]). On-disk compatibility
/// with pre-typed manifests (where `source` was a bare string such as
/// `""`, `"bundled"`, a Git URL, or an absolute path) is handled by
/// `PackageSource`'s serde shim — see
/// [`super::source::infer_from_legacy_source_string`] for the exact
/// mapping. Writes always emit the tagged form, so each install / update
/// migrates the file forward one entry at a time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ManifestEntry {
    /// Package version from `M.meta.version` (if available).
    pub version: Option<String>,
    /// How the package was installed (Git URL, local path, bundled, etc.).
    #[serde(default)]
    pub source: PackageSource,
    /// ISO 8601 timestamp of first install.
    pub installed_at: String,
    /// ISO 8601 timestamp of last update (same as installed_at if never updated).
    pub updated_at: String,
}

/// Top-level manifest structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub(crate) struct Manifest {
    pub packages: BTreeMap<String, ManifestEntry>,
}

// ─── Typed error ──────────────────────────────────────────────
//
// `InstalledManifestStoreError` classifies the ways `installed.json`
// CRUD can fail. The variants carry the path and the underlying cause
// (`std::io::Error` / `serde_json::Error` / lock subsystem message) so
// callers — and, via the `From` bridge below, the MCP wire response —
// can tell a missing-file case (which we normalise to an empty
// `Manifest` at the `load` layer) from a corrupt-JSON case from a
// locked-file case.
//
// The `From<InstalledManifestStoreError> for String` bridge keeps the
// existing service-layer `Result<_, String>` surfaces compiling; each
// caller's `?` runs through this conversion at the call site. A future
// upgrade to a typed service-layer `Result<_, ServiceError>` can absorb
// `InstalledManifestStoreError` through `#[from]` without a call-site
// churn.

/// CRUD failure modes for the installed-packages manifest.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InstalledManifestStoreError {
    /// Failed to read `installed.json`. Distinct from "file absent" —
    /// that case is normalised to an empty `Manifest` at the `load`
    /// boundary and never reaches this variant.
    #[error("failed to read installed manifest at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Failed to parse `installed.json` as JSON. Indicates on-disk
    /// corruption or a hand-edit that violated the schema.
    #[error("failed to parse installed manifest at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// Failed to serialise the in-memory `Manifest` to JSON. Indicates
    /// a programming error (non-representable value) — the serde model
    /// is total under normal use.
    #[error("failed to serialize installed manifest: {source}")]
    Serialize {
        #[source]
        source: serde_json::Error,
    },
    /// Failed to create the parent directory for `installed.json`.
    #[error("failed to create installed manifest directory at {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Failed to write the temp file that precedes the atomic rename.
    #[error("failed to write installed manifest temp file at {path}: {source}")]
    WriteTmp {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Failed to atomically rename the temp file into `installed.json`.
    #[error("failed to atomically rename installed manifest onto {path}: {source}")]
    Rename {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The advisory lock subsystem (`service::lock::with_exclusive_lock`)
    /// returned an error. We carry its message verbatim until the lock
    /// subsystem is also typed.
    #[error("installed manifest lock failed at {path}: {message}")]
    Lock { path: PathBuf, message: String },
}

/// Bridge for callers still on `Result<_, String>`. Each service-layer
/// caller's `?` runs through this conversion, so a corrupt
/// `installed.json` surfaces as a populated error string on the MCP
/// wire response rather than degrading silently. The conversion is one
/// way: the service layer loses variant identity, but a future upgrade
/// to a typed service error can absorb `InstalledManifestStoreError`
/// through `#[from]` without a call-site churn.
impl From<InstalledManifestStoreError> for String {
    fn from(e: InstalledManifestStoreError) -> Self {
        e.to_string()
    }
}

// ─── Store trait ──────────────────────────────────────────────
//
// `InstalledManifestStore` abstracts the `installed.json` CRUD surface so
// unit tests can substitute an in-memory impl for the fs-backed one. The
// store owns the IO — every `std::fs::read_to_string(installed.json)` /
// `std::fs::write(installed.json)` in this module is confined to the
// `FsInstalledManifestStore` impl block, which is what Inv-4 greps for.
//
// Naming note (intentional): this is a `Store`, not a `Repo`. It is a
// 1:1 adapter around a single physical file (`installed.json`) — the
// infrastructure-layer `Store` shape mirrors the engine-crate
// `JsonFileStore` / `FileCardStore` split. The service-layer
// `Repository` seat — a proper Aggregate Repo on top of a `Pkg` domain
// entity that orchestrates `installed.json` + `alc.toml` + `alc.lock` +
// the `packages/{name}/` fs layout + symlink state — is deliberately
// left vacant. Schema-proximate naming (`ManifestRepo`) would have
// pre-committed that seat to a 1-physical-file adapter and obscured the
// Aggregate boundary.

/// CRUD surface for the installed-packages manifest (`installed.json`).
///
/// `record_*` are responsible for their own locking (`with_lock`); they
/// do not expose the lock handle to callers.
///
/// The `Send + Sync` bounds are preemptive: no `Arc<dyn InstalledManifestStore>`
/// exists in-tree yet (callers still reach `FsInstalledManifestStore` via
/// the free-fn delegates at the bottom of this module), but the trait is
/// designed to slot into that shape once a proper `PkgRepository`
/// aggregate is introduced and takes ownership of this store alongside
/// the other per-file stores (`alc.toml` / `alc.lock` / fs layout /
/// symlink registry).
pub(crate) trait InstalledManifestStore: Send + Sync {
    /// Load the manifest. Returns an empty manifest if the backing store
    /// is missing; I/O / parse failures are returned as typed errors.
    fn load(&self) -> Result<Manifest, InstalledManifestStoreError>;

    /// Record a successful install / update (see `record_install` free-fn
    /// docs for semantics).
    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError>;

    /// Batch variant of `record_install`.
    fn record_install_batch(
        &self,
        names: &[String],
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError>;

    /// Remove a package from the manifest (used by `pkg_remove` scope
    /// `"global"` / `"all"`).
    fn record_remove(&self, name: &str) -> Result<(), InstalledManifestStoreError>;
}

// ─── FS-backed impl ───────────────────────────────────────────

/// `installed.json` on disk, rooted at `{app_dir}/installed.json`.
///
/// Holds `AppDir` by value rather than `Arc<AppDir>` because `AppDir`'s
/// internal `Arc<PathBuf>` already makes `clone` `O(1)`, and the struct
/// is constructed per-call as a short-lived delegate (see the free-fn
/// section at the bottom of this module). An additional outer `Arc`
/// would introduce a per-call heap allocation with no sharing benefit.
#[derive(Clone)]
pub(crate) struct FsInstalledManifestStore {
    app_dir: AppDir,
}

impl FsInstalledManifestStore {
    pub(crate) fn new(app_dir: AppDir) -> Self {
        Self { app_dir }
    }

    fn manifest_path(&self) -> PathBuf {
        self.app_dir.installed_json()
    }

    fn manifest_lock_path(&self) -> PathBuf {
        self.app_dir.installed_json().with_extension("json.lock")
    }

    /// Run `f` under an exclusive advisory file lock on the companion
    /// `.lock` path. Wraps the lock subsystem's stringly-typed errors
    /// into the typed `Lock` variant; `f`'s own errors (already typed)
    /// are passed through.
    fn with_lock<F, R>(&self, f: F) -> Result<R, InstalledManifestStoreError>
    where
        F: FnOnce() -> Result<R, InstalledManifestStoreError>,
    {
        let lock_path = self.manifest_lock_path();
        // `with_exclusive_lock` encodes both its own acquisition errors
        // and the inner closure's error as `String`. We cannot tell
        // them apart after the fact, so we return the whole string
        // under the `Lock` variant. Once the lock subsystem is typed,
        // this merging can be replaced with `#[from]` absorption.
        crate::service::lock::with_exclusive_lock(&lock_path, || f().map_err(|e| e.to_string()))
            .map_err(|message| InstalledManifestStoreError::Lock {
                path: lock_path.clone(),
                message,
            })
    }

    fn save(&self, manifest: &Manifest) -> Result<(), InstalledManifestStoreError> {
        let path = self.manifest_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                InstalledManifestStoreError::CreateDir {
                    path: parent.to_path_buf(),
                    source,
                }
            })?;
        }
        let content = serde_json::to_string_pretty(manifest)
            .map_err(|source| InstalledManifestStoreError::Serialize { source })?;

        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &content).map_err(|source| InstalledManifestStoreError::WriteTmp {
            path: tmp.clone(),
            source,
        })?;
        std::fs::rename(&tmp, &path).map_err(|source| {
            // Best-effort cleanup: the tmp file's presence is not a
            // correctness hazard (next `save` overwrites it) but
            // leaving it behind clutters the app dir.
            let _ = std::fs::remove_file(&tmp);
            InstalledManifestStoreError::Rename {
                path: path.clone(),
                source,
            }
        })
    }
}

impl InstalledManifestStore for FsInstalledManifestStore {
    fn load(&self) -> Result<Manifest, InstalledManifestStoreError> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(Manifest::default());
        }
        let content =
            std::fs::read_to_string(&path).map_err(|source| InstalledManifestStoreError::Read {
                path: path.clone(),
                source,
            })?;
        serde_json::from_str(&content)
            .map_err(|source| InstalledManifestStoreError::Parse { path, source })
    }

    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError> {
        self.with_lock(|| {
            let mut manifest = self.load()?;
            let now = now_iso8601();

            manifest
                .packages
                .entry(name.to_string())
                .and_modify(|e| {
                    if let Some(v) = version {
                        e.version = Some(v.to_string());
                    }
                    e.source = source.clone();
                    e.updated_at = now.clone();
                })
                .or_insert_with(|| ManifestEntry {
                    version: version.map(String::from),
                    source: source.clone(),
                    installed_at: now.clone(),
                    updated_at: now,
                });

            self.save(&manifest)
        })
    }

    fn record_install_batch(
        &self,
        names: &[String],
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError> {
        if names.is_empty() {
            return Ok(());
        }
        self.with_lock(|| {
            let mut manifest = self.load()?;
            let now = now_iso8601();

            for name in names {
                manifest
                    .packages
                    .entry(name.clone())
                    .and_modify(|e| {
                        e.source = source.clone();
                        e.updated_at = now.clone();
                    })
                    .or_insert_with(|| ManifestEntry {
                        version: None,
                        source: source.clone(),
                        installed_at: now.clone(),
                        updated_at: now.clone(),
                    });
            }

            self.save(&manifest)
        })
    }

    fn record_remove(&self, name: &str) -> Result<(), InstalledManifestStoreError> {
        self.with_lock(|| {
            let mut manifest = self.load()?;
            manifest.packages.remove(name);
            self.save(&manifest)
        })
    }
}

// ─── In-memory mock (tests only) ──────────────────────────────

/// In-memory manifest backed by a `Mutex<Manifest>`. Used by tests that
/// need parallel-safe isolation without touching the filesystem.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct InMemoryInstalledManifestStore {
    data: Mutex<Manifest>,
}

#[cfg(test)]
impl InstalledManifestStore for InMemoryInstalledManifestStore {
    fn load(&self) -> Result<Manifest, InstalledManifestStoreError> {
        Ok(self.data.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError> {
        let mut guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_iso8601();
        guard
            .packages
            .entry(name.to_string())
            .and_modify(|e| {
                if let Some(v) = version {
                    e.version = Some(v.to_string());
                }
                e.source = source.clone();
                e.updated_at = now.clone();
            })
            .or_insert_with(|| ManifestEntry {
                version: version.map(String::from),
                source: source.clone(),
                installed_at: now.clone(),
                updated_at: now,
            });
        Ok(())
    }

    fn record_install_batch(
        &self,
        names: &[String],
        source: PackageSource,
    ) -> Result<(), InstalledManifestStoreError> {
        if names.is_empty() {
            return Ok(());
        }
        let mut guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_iso8601();
        for name in names {
            guard
                .packages
                .entry(name.clone())
                .and_modify(|e| {
                    e.source = source.clone();
                    e.updated_at = now.clone();
                })
                .or_insert_with(|| ManifestEntry {
                    version: None,
                    source: source.clone(),
                    installed_at: now.clone(),
                    updated_at: now.clone(),
                });
        }
        Ok(())
    }

    fn record_remove(&self, name: &str) -> Result<(), InstalledManifestStoreError> {
        let mut guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
        guard.packages.remove(name);
        Ok(())
    }
}

// ─── Free-fn delegates (call-site compatibility) ──────────────
//
// Existing callers (`pkg/install.rs`, `pkg/list.rs`, `pkg/doctor.rs`,
// `pkg/repair.rs`, `pkg/remove.rs`, `hub.rs`) still hold `&AppDir`
// rather than an `Arc<dyn InstalledManifestStore>`, so the free functions are
// retained as thin delegates onto a per-call `FsInstalledManifestStore`. This
// keeps Inv-4 (all filesystem writes confined to the `FsInstalledManifestStore`
// impl block) intact while avoiding a crate-wide `Arc<dyn>` plumbing
// refactor in this commit.

/// Load the manifest from disk. Returns empty manifest if file is missing.
pub(crate) fn load_manifest(app_dir: &AppDir) -> Result<Manifest, InstalledManifestStoreError> {
    FsInstalledManifestStore::new(app_dir.clone()).load()
}

// ─── Operations ────────────────────────────────────────────────

pub(crate) fn now_iso8601() -> String {
    // Use SystemTime for a simple UTC timestamp without extra dependencies.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Manual formatting: YYYY-MM-DDTHH:MM:SSZ
    let s = secs as i64;
    let days = s / 86400;
    let time_of_day = s % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let sec = time_of_day % 60;

    // Days since epoch to Y-M-D (simplified Gregorian)
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{sec:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    // Algorithm from Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Record a successful install/update in the manifest.
///
/// - If the package already exists:
///   - `version` is overwritten **only when the caller provides `Some(_)`**;
///     `None` preserves whatever version was stored previously. Rationale:
///     the git-clone single-pkg path in `pkg_install` always passes `None`
///     here (it fetches the version later via `update_project_files_for_install`
///     for `alc.lock`, and does not write it back into this manifest).
///     Blindly clobbering with `None` would erase the version displayed by
///     `pkg_list` on every re-install.
///   - `source` and `updated_at` are always refreshed.
/// - If new, sets both `installed_at` and `updated_at` to now, and records
///   `version` verbatim (may be `None`).
///
/// `version` is extracted from the package's `M.meta.version` field if provided.
pub(crate) fn record_install(
    app_dir: &AppDir,
    name: &str,
    version: Option<&str>,
    source: PackageSource,
) -> Result<(), InstalledManifestStoreError> {
    FsInstalledManifestStore::new(app_dir.clone()).record_install(name, version, source)
}

/// Record a batch of installs (e.g. collection mode).
pub(crate) fn record_install_batch(
    app_dir: &AppDir,
    names: &[String],
    source: PackageSource,
) -> Result<(), InstalledManifestStoreError> {
    FsInstalledManifestStore::new(app_dir.clone()).record_install_batch(names, source)
}

/// Remove a package from the manifest (`installed.json`). Used by
/// `pkg_remove` scope `"global"` / `"all"`.
pub(crate) fn record_remove(
    app_dir: &AppDir,
    name: &str,
) -> Result<(), InstalledManifestStoreError> {
    FsInstalledManifestStore::new(app_dir.clone()).record_remove(name)
}

/// Load manifest for test with custom path.
#[cfg(test)]
pub(crate) fn load_manifest_from(
    path: &std::path::Path,
) -> Result<Manifest, InstalledManifestStoreError> {
    if !path.exists() {
        return Ok(Manifest::default());
    }
    let content =
        std::fs::read_to_string(path).map_err(|source| InstalledManifestStoreError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    serde_json::from_str(&content).map_err(|source| InstalledManifestStoreError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_installed_manifest_store_roundtrip() {
        // Demonstrates that `InMemoryInstalledManifestStore` satisfies the trait
        // end-to-end without touching the filesystem, so future unit
        // tests can exercise caller logic in parallel without the
        // `FakeHome` + `HOME_MUTEX` serialisation the FS-backed repo
        // currently forces.
        let repo = InMemoryInstalledManifestStore::default();
        let git_src = || PackageSource::Git {
            url: "https://example.test/mock".to_string(),
            rev: None,
        };

        repo.record_install("alpha", Some("1.0.0"), git_src())
            .unwrap();
        repo.record_install("alpha", None, git_src()).unwrap();
        repo.record_install_batch(&["beta".to_string(), "gamma".to_string()], git_src())
            .unwrap();
        repo.record_remove("beta").unwrap();

        let loaded = repo.load().unwrap();
        assert_eq!(
            loaded.packages.get("alpha").unwrap().version.as_deref(),
            Some("1.0.0"),
            "version=None should preserve existing entry"
        );
        assert!(loaded.packages.contains_key("gamma"));
        assert!(!loaded.packages.contains_key("beta"));
    }

    #[test]
    fn days_to_ymd_epoch() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2024-01-01 = day 19723
        assert_eq!(days_to_ymd(19723), (2024, 1, 1));
    }

    #[test]
    fn manifest_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("installed.json");

        let mut manifest = Manifest::default();
        manifest.packages.insert(
            "cot".to_string(),
            ManifestEntry {
                version: Some("0.1.0".to_string()),
                source: PackageSource::Git {
                    url: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
                    rev: None,
                },
                installed_at: "2024-01-01T00:00:00Z".to_string(),
                updated_at: "2024-01-01T00:00:00Z".to_string(),
            },
        );

        let content = serde_json::to_string_pretty(&manifest).unwrap();
        std::fs::write(&path, &content).unwrap();

        let loaded = load_manifest_from(&path).unwrap();
        assert_eq!(loaded, manifest);
    }

    /// Regression: legacy `installed.json` files written before the
    /// typed-source migration carried `source` as a bare string (e.g.
    /// `"https://..."`). The `PackageSource` serde shim must accept that
    /// shape and coerce it through `infer_from_legacy_source_string` to
    /// the appropriate tagged variant. If this test ever fails, existing
    /// user manifests on disk become unparseable — a breaking change
    /// masquerading as a schema update.
    #[test]
    fn manifest_backward_compat_legacy_string_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("installed.json");

        // A hand-written legacy manifest. Four entries covering the
        // distinct legacy shapes:
        //   - git URL  → PackageSource::Git
        //   - bundled  → PackageSource::Bundled
        //   - abs path → PackageSource::Installed
        //   - ""       → PackageSource::Unknown
        let legacy_json = r#"{
  "packages": {
    "cot": {
      "version": "0.1.0",
      "source": "https://github.com/ynishi/algocline-bundled-packages",
      "installed_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    },
    "ucb": {
      "version": null,
      "source": "bundled",
      "installed_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    },
    "local_pkg": {
      "version": null,
      "source": "/abs/local/pkg",
      "installed_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    },
    "legacy_empty": {
      "version": null,
      "source": "",
      "installed_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  }
}"#;
        std::fs::write(&path, legacy_json).unwrap();

        let loaded = load_manifest_from(&path).expect("must parse legacy manifest");
        assert_eq!(
            loaded.packages.get("cot").unwrap().source,
            PackageSource::Git {
                url: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
                rev: None,
            },
        );
        assert_eq!(
            loaded.packages.get("ucb").unwrap().source,
            PackageSource::Bundled { collection: None },
        );
        assert_eq!(
            loaded.packages.get("local_pkg").unwrap().source,
            PackageSource::Installed,
        );
        assert_eq!(
            loaded.packages.get("legacy_empty").unwrap().source,
            PackageSource::Unknown,
        );
    }

    #[test]
    fn manifest_empty_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let loaded = load_manifest_from(&path).unwrap();
        assert!(loaded.packages.is_empty());
    }

    #[test]
    fn record_install_none_preserves_existing_version() {
        // Regression: prior to `and_modify { e.version = version.map(..) }` →
        // conditional assignment, re-installing a git-cloned single pkg
        // (which passes `version = None`) silently erased the stored version
        // displayed by `pkg_list`. Guarantee: `None` preserves; `Some` overwrites.
        //
        // Uses a tempdir-rooted `AppDir` directly so the test does not rely on
        // the `FakeHome` HOME_MUTEX (軸 A defer).
        let tmp = tempfile::tempdir().unwrap();
        let app_dir = AppDir::new(tmp.path().to_path_buf());

        let git_src = || PackageSource::Git {
            url: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
            rev: None,
        };

        // Seed: insert an entry with a known version.
        record_install(&app_dir, "cot", Some("0.1.0"), git_src()).unwrap();
        let before = load_manifest(&app_dir).unwrap();
        assert_eq!(
            before.packages.get("cot").unwrap().version.as_deref(),
            Some("0.1.0")
        );

        // Re-install with `None` — should keep "0.1.0", not clobber.
        record_install(&app_dir, "cot", None, git_src()).unwrap();
        let after_none = load_manifest(&app_dir).unwrap();
        assert_eq!(
            after_none.packages.get("cot").unwrap().version.as_deref(),
            Some("0.1.0"),
            "version=None must preserve existing version"
        );

        // Re-install with `Some("0.2.0")` — should overwrite.
        record_install(&app_dir, "cot", Some("0.2.0"), git_src()).unwrap();
        let after_some = load_manifest(&app_dir).unwrap();
        assert_eq!(
            after_some.packages.get("cot").unwrap().version.as_deref(),
            Some("0.2.0"),
            "version=Some(_) must overwrite"
        );
    }
}
