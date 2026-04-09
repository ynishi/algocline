//! `alc_pkg_link` — link a local directory as a package via symlink.
//!
//! Creates a symlink in `~/.algocline/packages/{name}` pointing to the source
//! directory. Equivalent to `npm link`. No files are copied; no alc.lock is
//! written. Changes in the source directory are reflected immediately on the
//! next `alc_run`.

#[cfg(unix)]
use std::os::unix::fs::symlink;

use std::path::{Path, PathBuf};

use super::alc_toml::validate_package_name;
use super::resolve::packages_dir;
use super::AppService;

impl AppService {
    /// Link a local directory as a package by creating a symlink in the cache.
    ///
    /// `path`: source directory to link (absolute or cwd-relative).
    /// `name`: optional package name override (single package mode only).
    /// `force`: if `true`, overwrite an existing real directory at the destination.
    ///          Existing symlinks (including dangling) are always overwritten.
    pub async fn pkg_link(
        &self,
        path: String,
        name: Option<String>,
        force: Option<bool>,
    ) -> Result<String, String> {
        #[cfg(not(unix))]
        {
            let _ = (path, name, force);
            return Err("pkg_link is not supported on non-Unix platforms".to_string());
        }

        #[cfg(unix)]
        {
            let force = force.unwrap_or(false);

            // 1. Resolve source path (absolute: use as-is; relative: join with cwd).
            let raw = Path::new(&path);
            let source: PathBuf = if raw.is_absolute() {
                raw.to_path_buf()
            } else {
                std::env::current_dir()
                    .map_err(|e| format!("Cannot determine cwd: {e}"))?
                    .join(raw)
            };

            if !source.is_dir() {
                return Err(format!("Path is not a directory: {}", source.display()));
            }

            // 2. Detect mode: single package (init.lua at root) or collection.
            let mode = detect_mode(&source)?;

            // 3. Get packages_dir.
            let pkgs = packages_dir()?;
            std::fs::create_dir_all(&pkgs)
                .map_err(|e| format!("Cannot create packages dir {}: {e}", pkgs.display()))?;

            // 4. Link packages.
            let mode_str;
            let mut linked_names: Vec<String> = Vec::new();
            let mut targets: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

            match mode {
                PackageMode::Single => {
                    mode_str = "single";
                    let pkg_name = if let Some(n) = name {
                        n
                    } else {
                        source
                            .file_name()
                            .ok_or_else(|| {
                                format!("Cannot determine package name from: {}", source.display())
                            })?
                            .to_string_lossy()
                            .to_string()
                    };
                    validate_package_name(&pkg_name)?;

                    let dest = pkgs.join(&pkg_name);
                    create_symlink(&source, &dest, force)?;

                    targets.insert(
                        pkg_name.clone(),
                        serde_json::Value::String(source.display().to_string()),
                    );
                    linked_names.push(pkg_name);
                }
                PackageMode::Collection => {
                    mode_str = "collection";
                    let entries = std::fs::read_dir(&source).map_err(|e| {
                        format!("Failed to read directory {}: {e}", source.display())
                    })?;

                    for entry in entries {
                        let entry =
                            entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
                        let pkg_path = entry.path();
                        // Skip non-dirs and dirs without init.lua.
                        if !pkg_path.is_dir() || !pkg_path.join("init.lua").exists() {
                            continue;
                        }
                        let pkg_name = entry.file_name().to_string_lossy().to_string();
                        validate_package_name(&pkg_name)?;

                        let dest = pkgs.join(&pkg_name);
                        create_symlink(&pkg_path, &dest, force)?;

                        targets.insert(
                            pkg_name.clone(),
                            serde_json::Value::String(pkg_path.display().to_string()),
                        );
                        linked_names.push(pkg_name);
                    }

                    if linked_names.is_empty() {
                        return Err(format!(
                            "No init.lua found in any subdirectory of: {}",
                            source.display()
                        ));
                    }

                    linked_names.sort();
                }
            }

            Ok(serde_json::json!({
                "linked": linked_names,
                "mode": mode_str,
                "targets": targets,
            })
            .to_string())
        }
    }
}

// ─── Internal helpers ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum PackageMode {
    Single,
    Collection,
}

/// Determine whether `path` is a single package or a collection.
fn detect_mode(path: &Path) -> Result<PackageMode, String> {
    if path.join("init.lua").exists() {
        return Ok(PackageMode::Single);
    }

    let entries = std::fs::read_dir(path).map_err(|e| format!("Failed to read directory: {e}"))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
        let sub = entry.path();
        if sub.is_dir() && sub.join("init.lua").exists() {
            return Ok(PackageMode::Collection);
        }
    }

    Err(format!(
        "No init.lua found in {} or any of its subdirectories",
        path.display()
    ))
}

/// Create a symlink at `dest` pointing to `source`.
///
/// - If `dest` is a symlink (including dangling): always overwrite (remove + recreate).
/// - If `dest` is a real directory: require `force == true`; remove with `remove_dir_all`.
/// - If `dest` does not exist: create directly.
#[cfg(unix)]
fn create_symlink(source: &Path, dest: &Path, force: bool) -> Result<(), String> {
    // Check symlink status first (symlink_metadata does not follow symlinks).
    let meta = dest.symlink_metadata();

    if let Ok(m) = meta {
        if m.file_type().is_symlink() {
            // Existing symlink (live or dangling) — always overwrite.
            std::fs::remove_file(dest).map_err(|e| {
                format!("Failed to remove existing symlink {}: {e}", dest.display())
            })?;
        } else if m.is_dir() {
            // Real directory — require force.
            if !force {
                return Err(format!(
                    "Destination '{}' is a real directory. Use force=true to overwrite.",
                    dest.display()
                ));
            }
            std::fs::remove_dir_all(dest)
                .map_err(|e| format!("Failed to remove directory {}: {e}", dest.display()))?;
        } else {
            // Regular file or other — overwrite regardless.
            std::fs::remove_file(dest)
                .map_err(|e| format!("Failed to remove {}: {e}", dest.display()))?;
        }
    }

    symlink(source, dest).map_err(|e| {
        format!(
            "Failed to create symlink {} -> {}: {e}",
            dest.display(),
            source.display()
        )
    })
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::service::test_support::{make_app_service, FakeHome};

    #[tokio::test]
    async fn pkg_link_single_creates_symlink() {
        let env = FakeHome::new();
        let home = &env.home;

        let src = home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(src.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["mode"], "single");
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));
        assert_eq!(json["targets"]["my_pkg"], src.to_string_lossy().as_ref());

        let dest = home.join(".algocline").join("packages").join("my_pkg");
        assert!(dest.symlink_metadata().unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read_link(&dest).unwrap(), src);
    }

    #[tokio::test]
    async fn pkg_link_collection_creates_symlinks() {
        let env = FakeHome::new();
        let home = &env.home;

        let coll = home.join("collection");
        std::fs::create_dir_all(coll.join("pkg_a")).unwrap();
        std::fs::create_dir_all(coll.join("pkg_b")).unwrap();
        std::fs::write(coll.join("pkg_a").join("init.lua"), "return {}").unwrap();
        std::fs::write(coll.join("pkg_b").join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(coll.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["mode"], "collection");

        let linked = json["linked"].as_array().unwrap();
        let mut names: Vec<&str> = linked.iter().map(|v| v.as_str().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["pkg_a", "pkg_b"]);

        let pkgs = home.join(".algocline").join("packages");
        assert!(pkgs
            .join("pkg_a")
            .symlink_metadata()
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(pkgs
            .join("pkg_b")
            .symlink_metadata()
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[tokio::test]
    async fn pkg_link_overwrites_existing_symlink() {
        let env = FakeHome::new();
        let home = &env.home;

        let src = home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let pkgs = home.join(".algocline").join("packages");
        std::fs::create_dir_all(&pkgs).unwrap();
        let dest = pkgs.join("my_pkg");
        symlink(&src, &dest).unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(src.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));
        assert!(dest.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[tokio::test]
    async fn pkg_link_real_dir_requires_force() {
        let env = FakeHome::new();
        let home = &env.home;

        let src = home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let pkgs = home.join(".algocline").join("packages");
        let dest = pkgs.join("my_pkg");
        std::fs::create_dir_all(&dest).unwrap();

        let svc = make_app_service().await;

        let err = svc
            .pkg_link(src.to_string_lossy().to_string(), None, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("real directory"),
            "expected real directory error, got: {err}"
        );

        let result = svc
            .pkg_link(src.to_string_lossy().to_string(), None, Some(true))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));
        assert!(dest.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[tokio::test]
    async fn pkg_link_dangling_symlink_overwritten() {
        let env = FakeHome::new();
        let home = &env.home;

        let src = home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let pkgs = home.join(".algocline").join("packages");
        std::fs::create_dir_all(&pkgs).unwrap();
        let dest = pkgs.join("my_pkg");
        symlink(home.join("nonexistent"), &dest).unwrap();
        assert!(!dest.exists()); // dangling

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(src.to_string_lossy().to_string(), None, None)
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));
        assert!(dest.symlink_metadata().unwrap().file_type().is_symlink());
        assert!(dest.exists()); // no longer dangling
    }

    #[tokio::test]
    async fn pkg_link_path_not_found_returns_error() {
        let env = FakeHome::new();
        let nonexistent = env.home.join("does_not_exist");

        let svc = make_app_service().await;
        let err = svc
            .pkg_link(nonexistent.to_string_lossy().to_string(), None, None)
            .await
            .unwrap_err();
        assert!(err.contains("not a directory"), "got: {err}");
    }
}
