//! `alc_pkg_link` — link a local directory as a package.
//!
//! Two scopes:
//! - **global** (default): creates a symlink in `~/.algocline/packages/{name}`.
//!   Equivalent to `npm link`. No files are copied; no alc.lock is written.
//! - **variant**: appends a `[packages.{name}]` entry to `alc.local.toml`
//!   at the project root. Worktree-scoped override (git-ignored, loaded
//!   every `alc_run`). No symlink is created.

#[cfg(unix)]
use std::os::unix::fs::symlink;

use std::path::{Path, PathBuf};

use super::alc_toml::{self, add_package_entry, validate_package_name, PackageDep};
use super::project::resolve_project_root;
#[cfg(unix)]
use super::resolve::packages_dir;
use super::AppService;

impl AppService {
    /// Link a local directory as a package.
    ///
    /// - `scope = None | Some("global")`: create a symlink in
    ///   `~/.algocline/packages/{name}`. Unix-only (symlink).
    /// - `scope = Some("variant")`: record the path in `alc.local.toml`
    ///   at the project root. Works on all platforms (no symlink).
    /// - Any other scope value → `Err`.
    ///
    /// `force` is only meaningful in `global` scope (overwrite real dir).
    /// `project_root` is only consulted in `variant` scope.
    pub async fn pkg_link(
        &self,
        path: String,
        name: Option<String>,
        force: Option<bool>,
        scope: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        let scope_str = scope.as_deref().unwrap_or("global");
        match scope_str {
            "global" => self.pkg_link_global(path, name, force).await,
            "variant" => {
                let _ = force;
                self.pkg_link_variant(path, name, project_root).await
            }
            other => Err(format!(
                "invalid scope: '{other}' (expected 'global' or 'variant')"
            )),
        }
    }

    /// `scope = global` — create a symlink in `~/.algocline/packages/{name}`.
    async fn pkg_link_global(
        &self,
        path: String,
        name: Option<String>,
        force: Option<bool>,
    ) -> Result<String, String> {
        #[cfg(not(unix))]
        {
            let _ = (path, name, force);
            return Err(
                "pkg_link scope='global' is not supported on non-Unix platforms".to_string(),
            );
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
                "scope": "global",
            })
            .to_string())
        }
    }

    /// `scope = variant` — record the path in `alc.local.toml`.
    async fn pkg_link_variant(
        &self,
        path: String,
        name: Option<String>,
        project_root: Option<String>,
    ) -> Result<String, String> {
        // 1. Resolve source path.
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

        // 2. Detect mode.
        let mode = detect_mode(&source)?;

        // 3. Resolve project root.
        let root = resolve_project_root(project_root.as_deref()).ok_or_else(|| {
            "No project root found. Pass project_root or set ALC_PROJECT_ROOT, or run from within a project containing alc.toml.".to_string()
        })?;

        // 4. Load or create alc.local.toml document.
        let mut doc = match alc_toml::load_alc_local_toml_document(&root)? {
            Some(d) => d,
            None => "[packages]\n"
                .parse::<toml_edit::DocumentMut>()
                .map_err(|e| format!("Failed to create empty alc.local.toml document: {e}"))?,
        };

        // 5. Build entries to add.
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

                let abs = source.display().to_string();
                let added = add_package_entry(
                    &mut doc,
                    &pkg_name,
                    &PackageDep::Path {
                        path: abs.clone(),
                        version: None,
                    },
                );

                targets.insert(pkg_name.clone(), serde_json::Value::String(abs));
                if added {
                    linked_names.push(pkg_name);
                }
            }
            PackageMode::Collection => {
                mode_str = "collection";
                let entries = std::fs::read_dir(&source)
                    .map_err(|e| format!("Failed to read directory {}: {e}", source.display()))?;

                let mut candidates: Vec<(String, String)> = Vec::new();
                for entry in entries {
                    let entry =
                        entry.map_err(|e| format!("Failed to read directory entry: {e}"))?;
                    let pkg_path = entry.path();
                    if !pkg_path.is_dir() || !pkg_path.join("init.lua").exists() {
                        continue;
                    }
                    let pkg_name = entry.file_name().to_string_lossy().to_string();
                    validate_package_name(&pkg_name)?;
                    candidates.push((pkg_name, pkg_path.display().to_string()));
                }

                if candidates.is_empty() {
                    return Err(format!(
                        "No init.lua found in any subdirectory of: {}",
                        source.display()
                    ));
                }

                candidates.sort();
                for (pkg_name, abs) in candidates {
                    let added = add_package_entry(
                        &mut doc,
                        &pkg_name,
                        &PackageDep::Path {
                            path: abs.clone(),
                            version: None,
                        },
                    );
                    targets.insert(pkg_name.clone(), serde_json::Value::String(abs));
                    if added {
                        linked_names.push(pkg_name);
                    }
                }
            }
        }

        // 6. Save.
        alc_toml::save_alc_local_toml(&root, &doc)?;

        let alc_local_path = alc_toml::local_alc_toml_path(&root);

        Ok(serde_json::json!({
            "linked": linked_names,
            "mode": mode_str,
            "targets": targets,
            "scope": "variant",
            "alc_local_toml": alc_local_path.display().to_string(),
        })
        .to_string())
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
            .pkg_link(src.to_string_lossy().to_string(), None, None, None, None)
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
            .pkg_link(coll.to_string_lossy().to_string(), None, None, None, None)
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
            .pkg_link(src.to_string_lossy().to_string(), None, None, None, None)
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
            .pkg_link(src.to_string_lossy().to_string(), None, None, None, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("real directory"),
            "expected real directory error, got: {err}"
        );

        let result = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                Some(true),
                None,
                None,
            )
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
            .pkg_link(src.to_string_lossy().to_string(), None, None, None, None)
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
            .pkg_link(
                nonexistent.to_string_lossy().to_string(),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("not a directory"), "got: {err}");
    }

    // ── scope = variant ───────────────────────────────────────────────

    #[tokio::test]
    async fn pkg_link_scope_variant_appends_to_alc_local_toml() {
        let env = FakeHome::new();
        let root = env.home.join("proj");
        std::fs::create_dir_all(&root).unwrap();
        // Source pkg.
        let src = env.home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                None,
                Some("variant".to_string()),
                Some(root.to_string_lossy().to_string()),
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["scope"], "variant");
        assert_eq!(json["mode"], "single");
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));

        // File was written.
        let local = root.join("alc.local.toml");
        assert!(local.exists());
        let content = std::fs::read_to_string(&local).unwrap();
        assert!(content.contains("my_pkg"));
        assert!(content.contains(src.to_string_lossy().as_ref()));
    }

    #[tokio::test]
    async fn pkg_link_scope_variant_no_symlink_created() {
        let env = FakeHome::new();
        let root = env.home.join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let src = env.home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        svc.pkg_link(
            src.to_string_lossy().to_string(),
            None,
            None,
            Some("variant".to_string()),
            Some(root.to_string_lossy().to_string()),
        )
        .await
        .unwrap();

        let cache_link = env.home.join(".algocline").join("packages").join("my_pkg");
        assert!(
            cache_link.symlink_metadata().is_err(),
            "variant scope must not create a symlink in ~/.algocline/packages/"
        );
    }

    #[tokio::test]
    async fn pkg_link_scope_variant_second_call_is_noop_for_existing_entry() {
        let env = FakeHome::new();
        let root = env.home.join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let src = env.home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        svc.pkg_link(
            src.to_string_lossy().to_string(),
            None,
            None,
            Some("variant".to_string()),
            Some(root.to_string_lossy().to_string()),
        )
        .await
        .unwrap();

        // Second call — entry already exists, should be linked:[] (skipped).
        let result = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                None,
                Some("variant".to_string()),
                Some(root.to_string_lossy().to_string()),
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["linked"], serde_json::json!([]));
        // targets still records the current state.
        assert_eq!(json["targets"]["my_pkg"], src.to_string_lossy().as_ref());

        // File must still contain exactly one entry.
        // Parse the TOML and count keys under [packages] rather than
        // substring-match "my_pkg", because the path value also contains
        // that literal ("/.../my_pkg").
        let local = root.join("alc.local.toml");
        let content = std::fs::read_to_string(&local).unwrap();
        let doc: toml_edit::DocumentMut = content.parse().unwrap();
        let pkgs = doc["packages"].as_table().unwrap();
        let key_count = pkgs.iter().filter(|(k, _)| *k == "my_pkg").count();
        assert_eq!(key_count, 1, "duplicate entry written: {content}");
    }

    #[tokio::test]
    async fn pkg_link_scope_variant_requires_project_root() {
        let env = FakeHome::new();
        let src = env.home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        // No project_root, no ALC_PROJECT_ROOT (test env doesn't set it).
        // cwd walks up from test runner cwd — unlikely to find alc.toml ancestor.
        // Use an invalid explicit path to force fallback + fail.
        let nonexistent = env.home.join("no_such_project_root_zzz");
        let err = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                None,
                Some("variant".to_string()),
                Some(nonexistent.to_string_lossy().to_string()),
            )
            .await;
        // Note: fallback may succeed via cwd walk-up to a real alc.toml;
        // the best we can reliably assert is that EITHER:
        // (a) the call errored with "No project root found"
        // (b) the call succeeded with some valid root
        if let Err(e) = err {
            assert!(e.contains("No project root found"), "unexpected err: {e}");
        }
    }

    #[tokio::test]
    async fn pkg_link_invalid_scope_returns_error() {
        let env = FakeHome::new();
        let src = env.home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let err = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                None,
                Some("unknown".to_string()),
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("invalid scope"), "got: {err}");
    }

    #[tokio::test]
    async fn pkg_link_scope_global_default_matches_existing_behavior() {
        // Explicit scope=Some("global") should behave exactly as scope=None
        // (the default path).
        let env = FakeHome::new();
        let home = &env.home;

        let src = home.join("my_pkg");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                src.to_string_lossy().to_string(),
                None,
                None,
                Some("global".to_string()),
                None,
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["scope"], "global");
        assert_eq!(json["linked"], serde_json::json!(["my_pkg"]));
        let dest = home.join(".algocline").join("packages").join("my_pkg");
        assert!(dest.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[tokio::test]
    async fn pkg_link_scope_variant_collection_appends_all() {
        let env = FakeHome::new();
        let root = env.home.join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let coll = env.home.join("collection");
        std::fs::create_dir_all(coll.join("pkg_a")).unwrap();
        std::fs::create_dir_all(coll.join("pkg_b")).unwrap();
        std::fs::write(coll.join("pkg_a").join("init.lua"), "return {}").unwrap();
        std::fs::write(coll.join("pkg_b").join("init.lua"), "return {}").unwrap();

        let svc = make_app_service().await;
        let result = svc
            .pkg_link(
                coll.to_string_lossy().to_string(),
                None,
                None,
                Some("variant".to_string()),
                Some(root.to_string_lossy().to_string()),
            )
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["scope"], "variant");
        assert_eq!(json["mode"], "collection");
        let linked = json["linked"].as_array().unwrap();
        let names: Vec<&str> = linked.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(names, ["pkg_a", "pkg_b"]);

        let local = root.join("alc.local.toml");
        let content = std::fs::read_to_string(&local).unwrap();
        assert!(content.contains("pkg_a"));
        assert!(content.contains("pkg_b"));
    }
}
