//! `pkg_install` — install a package from a Git URL or local path.

use std::path::{Path, PathBuf};

use super::super::alc_toml::{
    add_package_entry, load_alc_toml_document, save_alc_toml, PackageDep,
};
use super::super::hub;
use super::super::lockfile::{load_lockfile, save_lockfile, LockFile, LockPackage};
use super::super::manifest;
use super::super::path::{copy_dir, ContainedPath};
use super::super::project::resolve_project_root;
use super::super::resolve::{
    install_scenarios_from_dir, packages_dir, scenarios_dir, DirEntryFailures, AUTO_INSTALL_SOURCES,
};
use super::super::source::PackageSource;
use super::super::AppService;

/// Explicit install dispatch. Carries exactly the information `pkg_install`
/// needs after classification so that downstream code does not re-classify
/// a string (which is racy: the local directory may disappear between the
/// caller's check and the installer's check).
#[derive(Debug, Clone)]
pub(crate) enum InstallSource {
    /// Copy from a local directory (absolute path).
    LocalPath(PathBuf),
    /// Clone from a Git URL (already normalized with scheme or `git@`).
    GitUrl(String),
}

/// Classify a caller-provided `url` string into an [`InstallSource`].
///
/// Must stay consistent with [`super::super::source::infer_from_legacy_source_string`]:
/// an absolute-path-*shaped* string maps to [`InstallSource::LocalPath`]
/// (matching `PackageSource::Installed`), everything else maps to a normalized
/// Git URL. Classification is deliberately syntactic — no filesystem probes.
/// Rationale: a dir that is_absolute but currently missing used to fall through
/// to the Git arm and produce `https:///abs/path`, which git rejects with
/// `unable to find remote helper for 'https'`. Keeping the classification
/// syntactic gives `install_from_local_path` a chance to surface a diagnostic
/// "Failed to read source dir" error instead.
fn classify_install_url(url: &str) -> InstallSource {
    let local_path = Path::new(url);
    if local_path.is_absolute() {
        return InstallSource::LocalPath(local_path.to_path_buf());
    }

    InstallSource::GitUrl(prefix_git_scheme_if_missing(url))
}

/// Prepend `https://` to a Git remote-style string that lacks a scheme.
///
/// Accepts `http://`, `https://`, `file://`, and `git@` prefixes as-is; any
/// other input (e.g. bare `github.com/a/b`) is prefixed with `https://`.
/// Shared between `classify_install_url` (install path) and
/// `pkg::repair::normalize_git_url` (repair path) — both need the same
/// normalization when routing an already-decided Git URL through `git clone`.
pub(super) fn prefix_git_scheme_if_missing(url: &str) -> String {
    if url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("file://")
        || url.starts_with("git@")
    {
        url.to_string()
    } else {
        format!("https://{url}")
    }
}

impl AppService {
    /// Install a package from a Git URL or local path (string-typed, public MCP API).
    ///
    /// Classifies `url` via [`classify_install_url`] then delegates to
    /// [`AppService::pkg_install_typed`]. Callers that already hold a
    /// classified [`InstallSource`] (e.g. `pkg_repair`) should call the
    /// typed API directly to avoid re-classifying a stale string.
    pub async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        let source = classify_install_url(&url);
        self.pkg_install_typed(source, name).await
    }

    /// Typed install dispatch. Does no string re-classification; branches
    /// explicitly on the already-classified [`InstallSource`].
    pub(crate) async fn pkg_install_typed(
        &self,
        source: InstallSource,
        name: Option<String>,
    ) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let _ = std::fs::create_dir_all(&pkg_dir);

        let git_url = match source {
            InstallSource::LocalPath(path) => {
                return self.install_from_local_path(&path, &pkg_dir, name).await;
            }
            InstallSource::GitUrl(u) => u,
        };
        // `url` is the recorded form used for manifest/hub. Normalization
        // happens in `classify_install_url`, so this is already the
        // scheme-prefixed form (e.g. `https://github.com/x`).
        let url = git_url.clone();

        // Clone to temp directory first to detect single vs collection
        let staging = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {e}"))?;

        // Bound `git clone` wall time. Without this a misconfigured remote
        // (auth prompt, unreachable host, slow network) can block the MCP
        // tool call indefinitely. 60s covers normal shallow clones of our
        // bundled-packages-sized repos with margin.
        let clone_future = tokio::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                &git_url,
                &staging.path().to_string_lossy(),
            ])
            .output();
        let output = tokio::time::timeout(std::time::Duration::from_secs(60), clone_future)
            .await
            .map_err(|_| format!("git clone timed out after 60s: {git_url}"))?
            .map_err(|e| format!("Failed to run git: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {stderr}"));
        }

        // Remove .git dir from staging (best-effort; absent .git would be
        // surprising but not fatal).
        if let Err(e) = std::fs::remove_dir_all(staging.path().join(".git")) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    "pkg_install: failed to strip .git from staging {}: {e}",
                    staging.path().display()
                );
            }
        }

        // Detect: single package (init.lua at root) vs collection (subdirs with init.lua)
        if staging.path().join("init.lua").exists() {
            // Single package mode
            let name = name.unwrap_or_else(|| {
                url.trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .trim_end_matches(".git")
                    .to_string()
            });

            let dest = ContainedPath::child(&pkg_dir, &name)?;
            if dest.as_ref().exists() {
                return Err(format!(
                    "Package '{name}' already exists at {}. Remove it first.",
                    dest.as_ref().display()
                ));
            }

            copy_dir(staging.path(), dest.as_ref())
                .map_err(|e| format!("Failed to copy package: {e}"))?;

            // Record in manifest (best-effort; install itself already succeeded)
            let _ = manifest::record_install(
                &name,
                None,
                super::super::source::PackageSource::Git {
                    url: url.clone(),
                    rev: None,
                },
            );
            hub::register_source(&url, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(std::slice::from_ref(&name))
                .await;

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "single",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        } else {
            // Collection mode: scan for subdirs containing init.lua
            if name.is_some() {
                // name parameter is only meaningful for single-package repos
                return Err(
                    "The 'name' parameter is only supported for single-package repos (init.lua at root). \
                     This repository is a collection (subdirs with init.lua)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut skipped = Vec::new();

            let entries = std::fs::read_dir(staging.path())
                .map_err(|e| format!("Failed to read staging dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                // Go through ContainedPath::child to block path traversal from
                // a malicious subdir name (`..`, `foo/../bar`) — the staging
                // dir is untrusted input in the general case.
                let dest = ContainedPath::child(&pkg_dir, &pkg_name)?;
                if dest.as_ref().exists() {
                    skipped.push(pkg_name);
                    continue;
                }
                copy_dir(&path, dest.as_ref())
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                installed.push(pkg_name);
            }

            // Import bundled cards from each package's cards/ subdirectory.
            let mut cards_installed: Vec<String> = Vec::new();
            for pkg_name in installed.iter().chain(skipped.iter()) {
                let cards_subdir = staging.path().join(pkg_name).join("cards");
                if cards_subdir.is_dir() {
                    let imported = self.import_pkg_bundled_cards(pkg_name, &cards_subdir);
                    cards_installed.extend(imported);
                }
            }

            // Install bundled scenarios only when an explicit `scenarios/` subdir exists.
            let scenarios_subdir = staging.path().join("scenarios");
            let mut scenarios_installed: Vec<String> = Vec::new();
            let mut scenarios_failures: DirEntryFailures = Vec::new();
            if scenarios_subdir.is_dir() {
                if let Ok(sc_dir) = scenarios_dir() {
                    std::fs::create_dir_all(&sc_dir)
                        .map_err(|e| format!("Failed to create scenarios dir: {e}"))?;
                    if let Ok(result) = install_scenarios_from_dir(&scenarios_subdir, &sc_dir) {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                            if let Some(arr) = parsed.get("installed").and_then(|v| v.as_array()) {
                                scenarios_installed = arr
                                    .iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect();
                            }
                            if let Some(arr) = parsed.get("failures").and_then(|v| v.as_array()) {
                                scenarios_failures = arr
                                    .iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect();
                            }
                        }
                    }
                }
            }

            if installed.is_empty() && skipped.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            // Record in manifest (best-effort)
            let _ = manifest::record_install_batch(
                &installed,
                super::super::source::PackageSource::Git {
                    url: url.clone(),
                    rev: None,
                },
            );
            hub::register_source(&url, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(&installed).await;

            let mut response = serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "cards_installed": cards_installed,
                "scenarios_installed": scenarios_installed,
                "scenarios_failures": scenarios_failures,
                "mode": "collection",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        }
    }

    /// Install from a local directory path (supports dirty/uncommitted files).
    async fn install_from_local_path(
        &self,
        source: &Path,
        pkg_dir: &Path,
        name: Option<String>,
    ) -> Result<String, String> {
        // Reject a missing source dir up front. Without this check, a missing
        // path falls through to the Collection branch (since `init.lua` isn't
        // present) and surfaces as the misleading "'name' parameter is only
        // supported for single-package dirs" error when `name` is provided —
        // which hides the real failure mode (source gone) from the caller.
        if !source.exists() {
            return Err(format!(
                "Source directory does not exist: {}",
                source.display()
            ));
        }
        if source.join("init.lua").exists() {
            // Single package
            let name = name.unwrap_or_else(|| {
                source
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });

            let dest = ContainedPath::child(pkg_dir, &name)?;
            if dest.as_ref().exists() {
                // Overwrite for local installs (dev workflow). Log failures —
                // silent `let _ =` used to hide Permission Denied / Busy
                // errors and surfaced later as a confusing "File exists" from
                // copy_dir.
                if let Err(e) = std::fs::remove_dir_all(&dest) {
                    tracing::warn!(
                        "pkg_install: failed to remove existing dest {} before overwrite: {e}",
                        dest.as_ref().display()
                    );
                }
            }

            copy_dir(source, dest.as_ref()).map_err(|e| format!("Failed to copy package: {e}"))?;
            // Remove .git if copied (best-effort; absent .git is the common case).
            if let Err(e) = std::fs::remove_dir_all(dest.as_ref().join(".git")) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        "pkg_install: failed to strip .git from {}: {e}",
                        dest.as_ref().display()
                    );
                }
            }

            // Record in manifest (best-effort). Local-path installs are
            // recorded as `Path { path }` so the original source location
            // is preserved in the typed form — this keeps `pkg_repair` able
            // to re-copy from the same source, and `pkg_list` can show
            // where the bytes came from. (Pre-typed manifests stored the
            // path as a bare string; `infer_from_legacy_source_string`
            // coerced it to `Installed`, which lost the path — the typed
            // form fixes that regression by carrying `path` explicitly.)
            let source_str_local = source.display().to_string();
            let _ = manifest::record_install(
                &name,
                None,
                super::super::source::PackageSource::Path {
                    path: source_str_local.clone(),
                },
            );
            hub::register_source(&source_str_local, "pkg_install");

            // Update alc.toml + alc.lock if project root is found
            self.update_project_files_for_install(std::slice::from_ref(&name))
                .await;

            let mut response = serde_json::json!({
                "installed": [name],
                "mode": "local_single",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        } else {
            // Collection mode
            if name.is_some() {
                return Err(
                    "The 'name' parameter is only supported for single-package dirs (init.lua at root)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut updated = Vec::new();

            let entries =
                std::fs::read_dir(source).map_err(|e| format!("Failed to read source dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() || !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                // Guard against traversal-shaped subdir names from an
                // untrusted source tree, matching the git-clone branch.
                let dest = ContainedPath::child(pkg_dir, &pkg_name)?;
                let existed = dest.as_ref().exists();
                if existed {
                    if let Err(e) = std::fs::remove_dir_all(dest.as_ref()) {
                        tracing::warn!(
                            "pkg_install: failed to remove existing dest {} before overwrite: {e}",
                            dest.as_ref().display()
                        );
                    }
                }
                copy_dir(&path, dest.as_ref())
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                if let Err(e) = std::fs::remove_dir_all(dest.as_ref().join(".git")) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            "pkg_install: failed to strip .git from {}: {e}",
                            dest.as_ref().display()
                        );
                    }
                }
                if existed {
                    updated.push(pkg_name);
                } else {
                    installed.push(pkg_name);
                }
            }

            if installed.is_empty() && updated.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            // Import bundled cards from each package's cards/ subdirectory.
            let mut cards_installed: Vec<String> = Vec::new();
            for pkg_name in installed.iter().chain(updated.iter()) {
                let cards_subdir = source.join(pkg_name).join("cards");
                if cards_subdir.is_dir() {
                    let imported = self.import_pkg_bundled_cards(pkg_name, &cards_subdir);
                    cards_installed.extend(imported);
                }
            }

            // Record in manifest (best-effort). Batch local-path installs
            // use `Path { path }` for the same reason as single-install
            // (preserve the source path in the typed form).
            let source_str = source.display().to_string();
            let all_names: Vec<String> = installed.iter().chain(updated.iter()).cloned().collect();
            let _ = manifest::record_install_batch(
                &all_names,
                super::super::source::PackageSource::Path {
                    path: source_str.clone(),
                },
            );
            hub::register_source(&source_str, "pkg_install");

            // Update alc.toml + alc.lock for newly installed packages
            self.update_project_files_for_install(&installed).await;

            let mut response = serde_json::json!({
                "installed": installed,
                "updated": updated,
                "cards_installed": cards_installed,
                "mode": "local_collection",
            });
            if let Some(tp) = super::super::resolve::types_stub_path() {
                response["types_path"] = serde_json::Value::String(tp);
            }
            Ok(response.to_string())
        }
    }

    /// After a successful cache install, update `alc.toml` and `alc.lock` if a project
    /// root (containing `alc.toml`) is found.  Failures are logged but not propagated —
    /// the install itself already succeeded.
    async fn update_project_files_for_install(&self, names: &[String]) {
        let root = match resolve_project_root(None) {
            Some(r) => r,
            None => return, // No project root → skip (current-compat)
        };

        // Resolve per-package versions *before* taking the lock, so the
        // lock-held critical section contains only synchronous I/O
        // (load → mutate → save). `fetch_pkg_version` dispatches into the
        // shared Lua executor and may await arbitrarily long.
        let mut resolved: Vec<(String, Option<String>)> = Vec::with_capacity(names.len());
        for name in names {
            let version = self.fetch_pkg_version(name).await;
            resolved.push((name.clone(), version));
        }

        // Guard the alc.toml / alc.lock load→modify→save against overlapping
        // `pkg_install` calls that target the same project root. Without this
        // advisory lock two concurrent installs can each load the old state,
        // apply their own mutation, and race to save — the later writer
        // silently overwrites the earlier's entry.
        let lock_path = project_files_lock_path(&root);
        let lock_result = super::super::lock::with_exclusive_lock(&lock_path, move || {
            // Load alc.toml document (preserving comments/formatting).
            let mut doc = match load_alc_toml_document(&root) {
                Ok(Some(d)) => d,
                Ok(None) => return Ok(()), // alc.toml not found → skip
                Err(e) => {
                    tracing::warn!("pkg_install: failed to load alc.toml: {e}");
                    return Ok(());
                }
            };

            // Load or create alc.lock.
            let mut lock = match load_lockfile(&root) {
                Ok(Some(l)) => l,
                Ok(None) => LockFile {
                    version: 1,
                    packages: Vec::new(),
                },
                Err(e) => {
                    tracing::warn!("pkg_install: failed to load alc.lock: {e}");
                    return Ok(());
                }
            };

            for (name, version) in &resolved {
                // Add to alc.toml (no-op if already present).
                add_package_entry(&mut doc, name, &PackageDep::Version("*".to_string()));
                // Upsert into alc.lock with the pre-resolved version.
                upsert_lock_entry(
                    &mut lock,
                    name.clone(),
                    version.clone(),
                    PackageSource::Installed,
                );
            }

            if let Err(e) = save_alc_toml(&root, &doc) {
                tracing::warn!("pkg_install: failed to save alc.toml: {e}");
            }
            if let Err(e) = save_lockfile(&root, &lock) {
                tracing::warn!("pkg_install: failed to save alc.lock: {e}");
            }
            Ok(())
        });

        if let Err(e) = lock_result {
            tracing::warn!("pkg_install: project files lock failed: {e}");
        }
    }

    /// Fetch package version via `eval_simple` (best-effort; returns `None` on failure).
    async fn fetch_pkg_version(&self, name: &str) -> Option<String> {
        if !is_safe_pkg_name(name) {
            return None;
        }
        let code = format!(
            r#"package.loaded["{name}"] = nil
local pkg = require("{name}")
return (pkg.meta or {{}}).version"#
        );
        match self.executor.eval_simple(code).await {
            Ok(serde_json::Value::String(v)) if !v.is_empty() => Some(v),
            _ => None,
        }
    }

    /// Install all bundled sources (collections + single packages).
    pub(in crate::service) async fn auto_install_bundled_packages(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();
        for url in AUTO_INSTALL_SOURCES {
            tracing::info!("auto-installing from {url}");
            if let Err(e) = self.pkg_install(url.to_string(), None).await {
                tracing::warn!("failed to auto-install from {url}: {e}");
                errors.push(format!("{url}: {e}"));
            }
        }
        // Fail only if ALL sources failed
        if errors.len() == AUTO_INSTALL_SOURCES.len() {
            return Err(format!(
                "Failed to auto-install bundled packages: {}",
                errors.join("; ")
            ));
        }
        Ok(())
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Path to the advisory lock file guarding `alc.toml` + `alc.lock` updates
/// within a project root. The lock file sits alongside the project files so
/// two processes working in the same checkout serialize on the same path.
///
/// The filename is deliberately distinct from `alc.lock` itself — the latter
/// is the dependency lockfile users read, while `.alc-install.lock` is an
/// internal flock companion. Consumers who share a project tree via `.gitignore`
/// should ignore it alongside other temp files; algocline does not add it
/// automatically today.
fn project_files_lock_path(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".alc-install.lock")
}

/// Returns `true` iff `name` is safe to interpolate into a Lua source string.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Insert or update a `LockPackage` entry in the lockfile.
fn upsert_lock_entry(
    lock: &mut LockFile,
    name: String,
    version: Option<String>,
    source: PackageSource,
) {
    if let Some(existing) = lock.packages.iter_mut().find(|p| p.name == name) {
        existing.version = version;
        existing.source = source;
    } else {
        lock.packages.push(LockPackage {
            name,
            version,
            source,
        });
    }
}
