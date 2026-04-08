//! Installed-packages manifest (`~/.algocline/installed.json`).
//!
//! Records package name, version, source, and install/update timestamps.
//! Written on `pkg_install` success, pruned on `pkg_remove`.
//! Read by `pkg_list` to display version tracking info.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Per-package record in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ManifestEntry {
    /// Package version from `M.meta.version` (if available).
    pub version: Option<String>,
    /// How the package was installed (git URL, local path, or "bundled").
    pub source: String,
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

// ─── Paths ─────────────────────────────────────────────────────

fn manifest_path() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("installed.json"))
}

// ─── Read / Write ──────────────────────────────────────────────

/// Load the manifest from disk. Returns empty manifest if file is missing.
pub(crate) fn load_manifest() -> Result<Manifest, String> {
    let path = manifest_path()?;
    if !path.exists() {
        return Ok(Manifest::default());
    }
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read manifest: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {e}"))
}

/// Save the manifest to disk (pretty-printed for human readability).
fn save_manifest(manifest: &Manifest) -> Result<(), String> {
    let path = manifest_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create manifest dir: {e}"))?;
    }
    let content = serde_json::to_string_pretty(manifest)
        .map_err(|e| format!("Failed to serialize manifest: {e}"))?;
    std::fs::write(&path, content).map_err(|e| format!("Failed to write manifest: {e}"))
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
/// - If the package already exists, updates `version`, `source`, and `updated_at`.
/// - If new, sets both `installed_at` and `updated_at` to now.
///
/// `version` is extracted from the package's `M.meta.version` field if provided.
pub(crate) fn record_install(
    name: &str,
    version: Option<&str>,
    source: &str,
) -> Result<(), String> {
    let mut manifest = load_manifest()?;
    let now = now_iso8601();

    let entry = manifest
        .packages
        .entry(name.to_string())
        .and_modify(|e| {
            e.version = version.map(String::from);
            e.source = source.to_string();
            e.updated_at = now.clone();
        })
        .or_insert_with(|| ManifestEntry {
            version: version.map(String::from),
            source: source.to_string(),
            installed_at: now.clone(),
            updated_at: now,
        });
    let _ = entry; // silence unused binding

    save_manifest(&manifest)
}

/// Record a batch of installs (e.g. collection mode).
pub(crate) fn record_install_batch(names: &[String], source: &str) -> Result<(), String> {
    if names.is_empty() {
        return Ok(());
    }
    let mut manifest = load_manifest()?;
    let now = now_iso8601();

    for name in names {
        manifest
            .packages
            .entry(name.clone())
            .and_modify(|e| {
                e.source = source.to_string();
                e.updated_at = now.clone();
            })
            .or_insert_with(|| ManifestEntry {
                version: None, // batch installs don't have per-package version info readily
                source: source.to_string(),
                installed_at: now.clone(),
                updated_at: now.clone(),
            });
    }

    save_manifest(&manifest)
}

/// Remove a package from the manifest.
pub(crate) fn record_remove(name: &str) -> Result<(), String> {
    let mut manifest = load_manifest()?;
    manifest.packages.remove(name);
    save_manifest(&manifest)
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
                source: "https://github.com/ynishi/algocline-bundled-packages".to_string(),
                installed_at: "2024-01-01T00:00:00Z".to_string(),
                updated_at: "2024-01-01T00:00:00Z".to_string(),
            },
        );

        let content = serde_json::to_string_pretty(&manifest).unwrap();
        std::fs::write(&path, &content).unwrap();

        let loaded = load_manifest_from(&path).unwrap();
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn manifest_empty_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let loaded = load_manifest_from(&path).unwrap();
        assert!(loaded.packages.is_empty());
    }
}
