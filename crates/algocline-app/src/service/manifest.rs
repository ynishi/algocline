//! Installed-packages manifest (`~/.algocline/installed.json`).
//!
//! Records package name, version, source, and install/update timestamps.
//! Written on `pkg_install` success, pruned on `pkg_remove`.
//! Read by `pkg_list` to display version tracking info.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
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

// ─── Repo trait (Subtask 3a) ──────────────────────────────────
//
// `ManifestRepo` abstracts the `installed.json` CRUD surface so unit
// tests can substitute an in-memory impl for the fs-backed one. The
// repo owns the IO — every `std::fs::read_to_string(installed.json)` /
// `std::fs::write(installed.json)` in this module is confined to the
// `FsManifestRepo` impl block, which is what Inv-4 (Subtask 3b) greps
// for.

/// CRUD surface for the installed-packages manifest.
///
/// `record_*` are responsible for their own locking (`with_lock`); they
/// do not expose the lock handle to callers.
pub(crate) trait ManifestRepo: Send + Sync {
    /// Load the manifest. Returns an empty manifest if the backing store
    /// is missing.
    fn load(&self) -> Result<Manifest, String>;

    /// Record a successful install / update (see `record_install` free-fn
    /// docs for semantics).
    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), String>;

    /// Batch variant of `record_install`.
    fn record_install_batch(&self, names: &[String], source: PackageSource) -> Result<(), String>;

    /// Remove a package from the manifest (used by `pkg_remove` scope
    /// `"global"` / `"all"`).
    fn record_remove(&self, name: &str) -> Result<(), String>;
}

// ─── FS-backed impl ───────────────────────────────────────────

/// `installed.json` on disk, rooted at `{app_dir}/installed.json`.
#[derive(Clone)]
pub(crate) struct FsManifestRepo {
    app_dir: Arc<AppDir>,
}

impl FsManifestRepo {
    pub(crate) fn new(app_dir: Arc<AppDir>) -> Self {
        Self { app_dir }
    }

    fn manifest_path(&self) -> PathBuf {
        self.app_dir.installed_json()
    }

    fn manifest_lock_path(&self) -> PathBuf {
        self.app_dir.installed_json().with_extension("json.lock")
    }

    fn with_lock<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce() -> Result<R, String>,
    {
        let lock_path = self.manifest_lock_path();
        crate::service::lock::with_exclusive_lock(&lock_path, f)
    }

    fn save(&self, manifest: &Manifest) -> Result<(), String> {
        let path = self.manifest_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create manifest dir: {e}"))?;
        }
        let content = serde_json::to_string_pretty(manifest)
            .map_err(|e| format!("Failed to serialize manifest: {e}"))?;

        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &content)
            .map_err(|e| format!("Failed to write manifest temp file {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!(
                "Failed to atomically rename manifest temp onto {}: {e}",
                path.display()
            )
        })
    }
}

impl ManifestRepo for FsManifestRepo {
    fn load(&self) -> Result<Manifest, String> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(Manifest::default());
        }
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("Failed to read manifest: {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {e}"))
    }

    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), String> {
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

    fn record_install_batch(&self, names: &[String], source: PackageSource) -> Result<(), String> {
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

    fn record_remove(&self, name: &str) -> Result<(), String> {
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
pub(crate) struct InMemoryManifestRepo {
    data: Mutex<Manifest>,
}

#[cfg(test)]
impl ManifestRepo for InMemoryManifestRepo {
    fn load(&self) -> Result<Manifest, String> {
        Ok(self.data.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    fn record_install(
        &self,
        name: &str,
        version: Option<&str>,
        source: PackageSource,
    ) -> Result<(), String> {
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

    fn record_install_batch(&self, names: &[String], source: PackageSource) -> Result<(), String> {
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

    fn record_remove(&self, name: &str) -> Result<(), String> {
        let mut guard = self.data.lock().unwrap_or_else(|e| e.into_inner());
        guard.packages.remove(name);
        Ok(())
    }
}

// ─── Free-fn delegates (call-site compatibility) ──────────────
//
// Existing callers (`pkg/install.rs`, `pkg/list.rs`, `pkg/doctor.rs`,
// `pkg/repair.rs`, `pkg/remove.rs`, `hub.rs`) still hold `&AppDir`
// rather than an `Arc<dyn ManifestRepo>`, so the free functions are
// retained as thin delegates onto a per-call `FsManifestRepo`. This
// keeps Inv-4 (all filesystem writes confined to the `FsManifestRepo`
// impl block) intact while avoiding a crate-wide `Arc<dyn>` plumbing
// refactor in this commit.

/// Load the manifest from disk. Returns empty manifest if file is missing.
pub(crate) fn load_manifest(app_dir: &AppDir) -> Result<Manifest, String> {
    FsManifestRepo::new(Arc::new(app_dir.clone())).load()
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
) -> Result<(), String> {
    FsManifestRepo::new(Arc::new(app_dir.clone())).record_install(name, version, source)
}

/// Record a batch of installs (e.g. collection mode).
pub(crate) fn record_install_batch(
    app_dir: &AppDir,
    names: &[String],
    source: PackageSource,
) -> Result<(), String> {
    FsManifestRepo::new(Arc::new(app_dir.clone())).record_install_batch(names, source)
}

/// Remove a package from the manifest (`installed.json`). Used by
/// `pkg_remove` scope `"global"` / `"all"`.
pub(crate) fn record_remove(app_dir: &AppDir, name: &str) -> Result<(), String> {
    FsManifestRepo::new(Arc::new(app_dir.clone())).record_remove(name)
}

/// Load manifest for test with custom path.
#[cfg(test)]
pub(crate) fn load_manifest_from(path: &std::path::Path) -> Result<Manifest, String> {
    if !path.exists() {
        return Ok(Manifest::default());
    }
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read manifest: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_manifest_repo_roundtrip() {
        // Demonstrates that `InMemoryManifestRepo` satisfies the trait
        // end-to-end without touching the filesystem, so future unit
        // tests can exercise caller logic in parallel without the
        // `FakeHome` + `HOME_MUTEX` serialisation the FS-backed repo
        // currently forces.
        let repo = InMemoryManifestRepo::default();
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
