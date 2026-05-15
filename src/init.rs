//! `alc init` / `alc update` — Install and update bundled packages.
//!
//! Clones packages from multiple Git sources and installs them into
//! `~/.algocline/packages/`.
//!
//! All sources use the **Collection** layout: the repo contains subdirectories,
//! each with `init.lua` (e.g. algocline-bundled-packages with ucb/, cove/, etc.).
//! Package authors must publish a `hub_index.json` at the repo root so that
//! `alc_hub_search` can discover their packages via `repo_to_index_url`.
//!
//! Sources are defined in [`BUNDLED_SOURCES`] and processed in order.
//!
//! Fallback: if git clone fails, looks for a sibling directory with
//! the same repo name on disk (development workflow).
//!
//! Usage:
//!   alc init             — Install new packages (skip existing)
//!   alc init --force     — Overwrite all packages
//!   alc init --dev       — Force local source (development)
//!   alc update           — Alias for `alc init --force`

use anyhow::Context;
use std::path::{Path, PathBuf};

/// A bundled package source: Git URL and tag.
///
/// All bundled sources use the Collection layout — the repo contains
/// subdirectories each with `init.lua`. The repo must also have a
/// `hub_index.json` at root so that `alc_hub_search` can discover it.
#[derive(Debug)]
struct BundledSource {
    url: &'static str,
    tag: &'static str,
}

/// All bundled sources, processed in order during `alc init`.
///
/// To add a new source: append an entry here. The repo must use the
/// Collection layout (`<repo>/<name>/init.lua`) and publish a
/// `hub_index.json` so `alc_hub_search` can discover its packages.
const BUNDLED_SOURCES: &[BundledSource] = &[
    BundledSource {
        url: "https://github.com/ynishi/algocline-bundled-packages",
        tag: "v0.23.0",
    },
    BundledSource {
        url: "https://github.com/ynishi/evalframe",
        tag: "v0.4.0",
    },
];

fn packages_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".algocline").join("packages"))
}

fn types_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".algocline").join("types"))
}

/// LuaCats type definitions for editor completion (alc.d.lua).
/// Embedded at compile time, distributed to ~/.algocline/types/ on init.
const ALC_TYPE_STUB: &str = include_str!("../types/alc.d.lua");

/// LuaCats type definitions for alc_shapes editor completion (alc_shapes.d.lua).
/// Embedded at compile time, distributed to ~/.algocline/types/ on init.
const ALC_SHAPES_TYPE_STUB: &str = include_str!("../types/alc_shapes.d.lua");

/// Paths to the installed type stub files distributed by [`distribute_types`].
#[derive(Debug)]
pub struct DistributedTypes {
    pub alc: PathBuf,
    pub alc_shapes: PathBuf,
}

/// Distribute alc.d.lua and alc_shapes.d.lua type stubs to ~/.algocline/types/.
/// Always overwrites (version upgrade support).
/// Returns the paths to both installed type stub files.
pub fn distribute_types() -> anyhow::Result<DistributedTypes> {
    let dir = types_dir()?;
    std::fs::create_dir_all(&dir)?;
    let alc = dir.join("alc.d.lua");
    std::fs::write(&alc, ALC_TYPE_STUB)?;
    let alc_shapes = dir.join("alc_shapes.d.lua");
    std::fs::write(&alc_shapes, ALC_SHAPES_TYPE_STUB)?;
    Ok(DistributedTypes { alc, alc_shapes })
}

/// Print .luarc.json setup guidance if not present in current directory.
fn print_luarc_guidance(types_path: &Path) {
    let luarc = std::env::current_dir().map(|d| d.join(".luarc.json")).ok();
    if luarc.as_ref().is_some_and(|p| p.exists()) {
        return;
    }
    let types_dir = types_path.parent().unwrap_or(types_path);
    eprintln!();
    eprintln!("Tip: To enable editor completion, create .luarc.json with:");
    eprintln!(
        r#"  {{ "workspace": {{ "library": ["{}"] }} }}"#,
        types_dir.display()
    );
}

/// Distribute type stubs and print guidance. Non-fatal: warnings only on error.
fn finalize_init() {
    match distribute_types() {
        Ok(DistributedTypes { alc, alc_shapes }) => {
            eprintln!("installed: {}", alc.display());
            eprintln!("installed: {}", alc_shapes.display());
            print_luarc_guidance(&alc);
        }
        Err(e) => {
            eprintln!("Warning: failed to install type stubs: {e}");
        }
    }
}

/// Discover package directories in a source directory.
///
/// Returns sorted list of (name, path) for each subdirectory containing `init.lua`.
/// Names must be valid Lua module identifiers (alphanumeric + underscore).
fn discover_packages(source: &Path) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let mut packages = Vec::new();

    let entries = std::fs::read_dir(source)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if !path.join("init.lua").exists() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        // Skip hidden dirs and non-Lua-identifier names
        if name.starts_with('.') || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        packages.push((name, path));
    }

    packages.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(packages)
}

/// Extract repo name from a Git URL (e.g. "https://github.com/user/evalframe" → "evalframe").
fn repo_name(url: &str) -> &str {
    url.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("unknown")
}

/// Find a local sibling directory for a given repo name (development).
///
/// Searches for `../{repo_name}/` relative to CWD or the binary location.
/// This supports the development workflow where repositories are checked out side by side.
fn find_local_source(name: &str) -> Option<PathBuf> {
    // Check CWD/../{name}/
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(parent) = cwd.parent() {
            let sibling = parent.join(name);
            if sibling.is_dir() {
                return Some(sibling);
            }
        }
    }

    // Check relative to binary
    if let Ok(exe) = std::env::current_exe() {
        let dev_pkg = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join(name));
        if let Some(path) = dev_pkg {
            if path.is_dir() {
                return Some(path);
            }
        }
    }

    None
}

/// Copy a package directory tree to dest.
///
/// Copies the entire directory tree (all files, not just `init.lua`) so that
/// multi-file Collection packages install completely into `dest_root/{name}/`.
///
/// Zombie detection: compares the top-level entry names of the source and dest
/// trees. If all source entries are present in dest (src ⊆ dest), the install
/// is considered healthy and skipped. A missing entry triggers repair.
///
/// # Arguments
/// * `name` - Package name used as the destination directory basename.
/// * `pkg_source` - Path to the source package directory (must exist).
/// * `dest_root` - Parent directory under which `{name}/` will be created.
/// * `force` - When true, unconditionally replaces any existing dest tree.
///
/// # Returns
/// * `Ok(true)` — Package was installed or updated.
/// * `Ok(false)` — Package already exists and is healthy; skipped.
///
/// # Errors
/// Returns an error if `pkg_source` does not exist, or if any I/O operation fails.
fn copy_package(
    name: &str,
    pkg_source: &Path,
    dest_root: &Path,
    force: bool,
) -> anyhow::Result<bool> {
    if !pkg_source.exists() {
        anyhow::bail!("Source not found: {}", pkg_source.display());
    }

    let dest_dir = dest_root.join(name);

    if dest_dir.exists() && !force {
        // Zombie detection: collect top-level entry names from src and dest.
        // If all src entries are present in dest (src ⊆ dest), the tree is
        // healthy and we skip. Any missing src entry triggers repair.
        let src_names = top_level_entry_names(pkg_source)?;
        let dest_names = top_level_entry_names(&dest_dir)?;
        let all_present = src_names.iter().all(|n| dest_names.contains(n));
        if all_present {
            return Ok(false); // Healthy tree, skip
        }
        // Missing entries → zombie. Fall through to overwrite.
        eprintln!("    (repairing incomplete package for {name})");
    }

    if dest_dir.exists() {
        std::fs::remove_dir_all(&dest_dir)?;
    }
    copy_dir(pkg_source, &dest_dir)?;
    // Remove .git if present (best-effort, not an error if absent)
    let _ = std::fs::remove_dir_all(dest_dir.join(".git"));

    Ok(true)
}

/// Collect the top-level entry names of a directory.
///
/// # Arguments
/// * `dir` - Directory to read.
///
/// # Returns
/// A set of file/directory name strings (OsString lossy-converted).
///
/// # Errors
/// Returns an error if the directory cannot be read.
fn top_level_entry_names(dir: &Path) -> std::io::Result<std::collections::HashSet<String>> {
    let mut names = std::collections::HashSet::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        names.insert(entry.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

/// Recursively copy a directory tree.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        let dest_path = dst.join(entry.file_name());
        if meta.is_dir() {
            copy_dir(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

/// Clone a single source and install its packages.
async fn install_source_from_git(
    source: &BundledSource,
    dest: &Path,
    force: bool,
) -> anyhow::Result<()> {
    eprintln!("Cloning {} ({})...", source.url, source.tag);

    let staging = tempfile::tempdir()?;

    let output = tokio::process::Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--branch",
            source.tag,
            source.url,
            &staging.path().to_string_lossy(),
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git clone failed (tag {}): {stderr}", source.tag);
    }

    install_from_local(staging.path(), dest, force)
}

/// Clone all bundled sources and install.
async fn install_from_git(dest: &Path, force: bool) -> anyhow::Result<()> {
    let mut errors: Vec<String> = Vec::new();

    for source in BUNDLED_SOURCES {
        if let Err(e) = install_source_from_git(source, dest, force).await {
            eprintln!("  ! Failed to install from {}: {e}", source.url);
            errors.push(format!("{}: {e}", source.url));
        }
    }

    if errors.len() == BUNDLED_SOURCES.len() {
        // All sources failed
        anyhow::bail!(
            "All bundled sources failed to install: {}",
            errors.join("; ")
        );
    }

    if !errors.is_empty() {
        eprintln!(
            "Warning: {} of {} sources failed (non-fatal)",
            errors.len(),
            BUNDLED_SOURCES.len()
        );
    }

    Ok(())
}

/// Copy `<source>/docs/narrative/{name}.md` to `<dest>/{name}/narrative.md`
/// when present. Used by `install_from_local` to surface bundled-packages
/// narrative docs as the `alc://packages/{name}/narrative` MCP resource
/// (#1777052474). Always overwrites the destination so a `--force` install
/// or a tag bump propagates updated narrative without manual cleanup.
///
/// Returns `Ok(())` when the source narrative does not exist (silent skip)
/// or when the copy succeeds. Returns `Err` only on filesystem failure
/// while the source file *does* exist — so non-bundled installs (which do
/// not follow the `docs/narrative/` convention) never raise.
fn install_narrative_for(source: &Path, dest: &Path, name: &str) -> anyhow::Result<()> {
    let src = source
        .join("docs")
        .join("narrative")
        .join(format!("{name}.md"));
    if !src.is_file() {
        return Ok(());
    }
    let dest_pkg = dest.join(name);
    if !dest_pkg.is_dir() {
        // copy_package failed earlier — skip narrative copy rather than
        // creating an orphan dir. The earlier error has already been
        // reported by the caller.
        return Ok(());
    }
    let dest_file = dest_pkg.join("narrative.md");
    std::fs::copy(&src, &dest_file).with_context(|| {
        format!(
            "copy narrative from {} to {}",
            src.display(),
            dest_file.display()
        )
    })?;
    Ok(())
}

/// Install from local packages directory.
///
/// Dynamically discovers all subdirectories with `init.lua` and installs them.
fn install_from_local(source: &Path, dest: &Path, force: bool) -> anyhow::Result<()> {
    eprintln!("Installing packages from {}...", source.display());

    let packages = discover_packages(source)?;

    if packages.is_empty() {
        anyhow::bail!(
            "No packages found in {}. Expected subdirectories with init.lua.",
            source.display()
        );
    }

    let mut installed = 0;
    let mut updated = 0;
    let mut skipped = 0;
    let mut failures: Vec<String> = Vec::new();

    for (name, pkg_path) in &packages {
        let existed = dest.join(name).join("init.lua").exists();
        match copy_package(name, pkg_path, dest, force) {
            Ok(true) => {
                if existed {
                    eprintln!("  ~ {name} (updated)");
                    updated += 1;
                } else {
                    eprintln!("  + {name}");
                    installed += 1;
                }
            }
            Ok(false) => {
                eprintln!("  = {name} (already installed, use --force to overwrite)");
                skipped += 1;
            }
            Err(e) => {
                eprintln!("  ! {name}: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
        // Stdpkg reference (#1777052474): bundled-packages stores per-pkg
        // narrative under `<source>/docs/narrative/{name}.md` outside the
        // pkg subdir, so `copy_package` (which only copies the pkg subdir)
        // does not pick it up. Copy the narrative file (when present) into
        // `<dest>/{name}/narrative.md` so the `alc://packages/{name}/narrative`
        // MCP resource can serve it from a stable path. Silent skip when the
        // source has no narrative for this pkg — non-bundled local sources
        // simply do not have the docs/narrative/ convention.
        if let Err(e) = install_narrative_for(source, dest, name) {
            eprintln!("  ! {name}: narrative copy failed: {e}");
        }
    }

    eprintln!(
        "Done: {installed} installed, {updated} updated, {skipped} skipped. ({} packages total)",
        packages.len()
    );

    if !failures.is_empty() {
        anyhow::bail!(
            "{} package(s) failed to install: {}",
            failures.len(),
            failures.join(", ")
        );
    }

    Ok(())
}

pub async fn run(args: &[String], force_override: bool) -> anyhow::Result<()> {
    let force = force_override || args.iter().any(|a| a == "--force");
    let dev = args.iter().any(|a| a == "--dev");

    let dest = packages_dir()?;
    std::fs::create_dir_all(&dest)?;

    if dev {
        // --dev: install from local sibling directories for all sources
        let mut found_any = false;
        for source in BUNDLED_SOURCES {
            let name = repo_name(source.url);
            if let Some(local) = find_local_source(name) {
                found_any = true;
                install_from_local(&local, &dest, force)?;
            } else {
                eprintln!("  ? {name}: local directory not found, skipping");
            }
        }
        if !found_any {
            anyhow::bail!("No local source directories found for any bundled source");
        }
        finalize_init();
        return Ok(());
    }

    // Try git clone first, fall back to local for failed sources
    install_from_git(&dest, force).await?;
    finalize_init();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_source_tags_are_valid_semver() {
        for source in BUNDLED_SOURCES {
            let version = source.tag.strip_prefix('v').unwrap_or(source.tag);
            assert!(
                version.split('.').all(|p| p.parse::<u32>().is_ok()),
                "Invalid semver tag '{}' for source {}",
                source.tag,
                source.url
            );
        }
    }

    #[test]
    fn discover_packages_finds_subdirs_with_init_lua() {
        let source = tempfile::tempdir().unwrap();

        // Valid package
        let pkg_a = source.path().join("alpha");
        std::fs::create_dir(&pkg_a).unwrap();
        std::fs::write(pkg_a.join("init.lua"), "return {}").unwrap();

        // Valid package
        let pkg_b = source.path().join("beta");
        std::fs::create_dir(&pkg_b).unwrap();
        std::fs::write(pkg_b.join("init.lua"), "return {}").unwrap();

        // Dir without init.lua — skipped
        let no_init = source.path().join("nomod");
        std::fs::create_dir(&no_init).unwrap();

        // Hidden dir — skipped
        let hidden = source.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("init.lua"), "return {}").unwrap();

        // Regular file — skipped
        std::fs::write(source.path().join("README.md"), "# hi").unwrap();

        let packages = discover_packages(source.path()).unwrap();
        let names: Vec<&str> = packages.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn discover_packages_skips_invalid_names() {
        let source = tempfile::tempdir().unwrap();

        // Invalid: contains hyphen
        let bad = source.path().join("my-pkg");
        std::fs::create_dir(&bad).unwrap();
        std::fs::write(bad.join("init.lua"), "return {}").unwrap();

        // Valid: underscore OK
        let good = source.path().join("my_pkg");
        std::fs::create_dir(&good).unwrap();
        std::fs::write(good.join("init.lua"), "return {}").unwrap();

        let packages = discover_packages(source.path()).unwrap();
        let names: Vec<&str> = packages.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["my_pkg"]);
    }

    #[test]
    fn discover_packages_returns_sorted() {
        let source = tempfile::tempdir().unwrap();

        for name in &["zeta", "alpha", "mid"] {
            let dir = source.path().join(name);
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join("init.lua"), "return {}").unwrap();
        }

        let packages = discover_packages(source.path()).unwrap();
        let names: Vec<&str> = packages.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn copy_package_creates_init_lua() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Create a source package
        let pkg_dir = source.path().join("mypkg");
        std::fs::create_dir(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let installed = copy_package("mypkg", &pkg_dir, dest.path(), false).unwrap();
        assert!(installed);
        assert!(dest.path().join("mypkg/init.lua").exists());
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {}"
        );
    }

    #[test]
    fn copy_package_copies_subfiles() {
        // T1 (happy path): multi-file source dir installs all sub-files to dest.
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {}").unwrap();
        std::fs::write(src_pkg.join("mc.lua"), "return {}").unwrap();
        std::fs::write(src_pkg.join("stats.lua"), "return {}").unwrap();

        let installed = copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();
        assert!(installed, "fresh install should return Ok(true)");
        assert!(
            dest.path().join("mypkg/init.lua").exists(),
            "init.lua must be installed"
        );
        assert!(
            dest.path().join("mypkg/mc.lua").exists(),
            "mc.lua must be installed"
        );
        assert!(
            dest.path().join("mypkg/stats.lua").exists(),
            "stats.lua must be installed"
        );
    }

    #[test]
    fn copy_package_skips_existing_same_size() {
        // Healthy tree: src top-level entries ⊆ dest top-level entries → skip.
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {v=2}").unwrap();

        // Dest has the same top-level entry (init.lua) → healthy, skip
        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "return {v=1}").unwrap();

        let installed = copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();
        assert!(!installed, "healthy tree should be skipped");
        // Original dest content must be preserved (not overwritten)
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {v=1}"
        );
    }

    #[test]
    fn copy_package_repairs_zombie_file() {
        // Zombie: dest is missing one or more src top-level entries.
        // Setup: src has init.lua + mc.lua, dest has only init.lua.
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {complete=true}").unwrap();
        std::fs::write(src_pkg.join("mc.lua"), "return {}").unwrap();

        // Dest is missing mc.lua — incomplete (zombie) tree
        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "return {old=true}").unwrap();

        // Without force: missing entry triggers repair
        let installed = copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();
        assert!(installed, "zombie should be repaired even without --force");
        // After repair both files must be present with src content
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {complete=true}"
        );
        assert!(dest.path().join("mypkg/mc.lua").exists());
    }

    #[test]
    fn copy_package_no_tmp_file_on_success() {
        // Tree copy leaves no stale .tmp files and installs all expected files.
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {}").unwrap();
        std::fs::write(src_pkg.join("helper.lua"), "return {}").unwrap();

        copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();

        // No stale temp files
        assert!(!dest.path().join("mypkg/init.lua.tmp").exists());
        // Tree was written correctly
        assert!(dest.path().join("mypkg/init.lua").exists());
        assert!(dest.path().join("mypkg/helper.lua").exists());
    }

    #[test]
    fn copy_package_force_overwrites() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {new=true}").unwrap();

        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "return {old=true}").unwrap();

        let installed = copy_package("mypkg", &src_pkg, dest.path(), true).unwrap();
        assert!(installed);
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {new=true}"
        );
    }

    #[test]
    fn copy_package_missing_source_errors() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let empty = source.path().join("nonexistent");
        let result = copy_package("nonexistent", &empty, dest.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn install_from_local_discovers_and_installs() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        for name in &["pkg_a", "pkg_b", "pkg_c"] {
            let dir = source.path().join(name);
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join("init.lua"), format!("return {{name=\"{name}\"}}")).unwrap();
        }

        install_from_local(source.path(), dest.path(), false).unwrap();

        assert!(dest.path().join("pkg_a/init.lua").exists());
        assert!(dest.path().join("pkg_b/init.lua").exists());
        assert!(dest.path().join("pkg_c/init.lua").exists());
    }

    #[test]
    fn install_from_local_update_mode() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Initial install
        let pkg = source.path().join("mypkg");
        std::fs::create_dir(&pkg).unwrap();
        std::fs::write(pkg.join("init.lua"), "return {v=1}").unwrap();
        install_from_local(source.path(), dest.path(), false).unwrap();

        // Update source
        std::fs::write(pkg.join("init.lua"), "return {v=2}").unwrap();

        // Without force: skipped
        install_from_local(source.path(), dest.path(), false).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {v=1}"
        );

        // With force: updated
        install_from_local(source.path(), dest.path(), true).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {v=2}"
        );
    }

    #[test]
    fn install_from_local_reports_partial_failure() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Valid package
        let good = source.path().join("good_pkg");
        std::fs::create_dir(&good).unwrap();
        std::fs::write(good.join("init.lua"), "return {}").unwrap();

        // Package dir exists but init.lua is missing (will fail copy_package)
        let bad = source.path().join("bad_pkg");
        std::fs::create_dir(&bad).unwrap();
        std::fs::write(bad.join("init.lua"), "return {}").unwrap();

        // First install succeeds
        install_from_local(source.path(), dest.path(), false).unwrap();

        // Remove source init.lua for bad_pkg to simulate copy failure on force update
        std::fs::remove_file(bad.join("init.lua")).unwrap();

        // Force update: good_pkg succeeds, bad_pkg no longer discovered (no init.lua)
        // Instead, test with a read-only dest to trigger fs::copy failure
        let source2 = tempfile::tempdir().unwrap();
        let dest2 = tempfile::tempdir().unwrap();

        let pkg = source2.path().join("test_pkg");
        std::fs::create_dir(&pkg).unwrap();
        std::fs::write(pkg.join("init.lua"), "return {}").unwrap();

        // Make dest read-only to force fs::create_dir_all failure
        let dest_pkg = dest2.path().join("test_pkg");
        std::fs::create_dir(&dest_pkg).unwrap();
        // Create a file where init.lua dir would go, blocking create_dir_all
        // Actually, just verify the error path by using a non-writable directory
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dest2.path(), std::fs::Permissions::from_mode(0o444)).unwrap();

            let result = install_from_local(source2.path(), dest2.path(), true);
            assert!(result.is_err(), "should report partial failure");
            let err_msg = result.unwrap_err().to_string();
            assert!(
                err_msg.contains("failed to install"),
                "error should mention failure: {err_msg}"
            );

            // Restore permissions for cleanup
            std::fs::set_permissions(dest2.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    #[test]
    fn alc_type_stub_starts_with_meta() {
        assert!(
            ALC_TYPE_STUB.starts_with("---@meta"),
            "ALC_TYPE_STUB should start with ---@meta (LuaCats format)"
        );
    }

    #[test]
    fn alc_type_stub_contains_llm_function() {
        assert!(
            ALC_TYPE_STUB.contains("function alc.llm"),
            "ALC_TYPE_STUB should contain function alc.llm definition"
        );
    }
}
