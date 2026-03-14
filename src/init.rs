//! `alc init` — Install bundled packages from algocline-bundled-packages.
//!
//! Clones the bundled package collection from GitHub (tag-based)
//! and installs all packages into `~/.algocline/packages/`.
//!
//! Sources (checked in order):
//! 1. Git clone with tag `v{BUNDLED_VERSION}` (production)
//! 2. Local packages directory (development fallback)
//!
//! Usage:
//!   alc init           — Install bundled packages
//!   alc init --force   — Overwrite existing packages
//!   alc init --dev     — Force local source (development)

use std::path::{Path, PathBuf};

/// Supported bundled packages version.
///
/// Independent of algocline's own CARGO_PKG_VERSION.
/// Updated when a new bundled-packages release is validated.
const BUNDLED_VERSION: &str = "0.1.0";

const BUNDLED_PACKAGES_URL: &str = "https://github.com/ynishi/algocline-bundled-packages";

/// Bundled package names shipped with algocline.
const BUNDLED_PACKAGES: &[&str] = &[
    // Reasoning
    "cot",
    "maieutic",
    "reflect",
    "calibrate",
    // Selection
    "sc",
    "rank",
    "triad",
    "ucb",
    // Generation
    "sot",
    "decompose",
    // Extraction
    "distill",
    "cod",
    // Validation / Analysis
    "cove",
    "factscore",
    // Synthesis
    "panel",
];

fn packages_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".algocline").join("packages"))
}

/// Find a local packages source directory (development).
///
/// Searches for a sibling `algocline-bundled-packages/` directory relative to CWD
/// or the binary location. This supports the development workflow where
/// both repositories are checked out side by side.
fn find_local_packages() -> Option<PathBuf> {
    // Check CWD/../algocline-bundled-packages/
    let cwd = std::env::current_dir().ok()?;
    let sibling = cwd.parent()?.join("algocline-bundled-packages");
    if sibling.is_dir() {
        return Some(sibling);
    }

    // Check relative to binary
    if let Ok(exe) = std::env::current_exe() {
        let dev_pkg = exe
            .parent()?
            .parent()?
            .parent()?
            .parent()?
            .join("algocline-bundled-packages");
        if dev_pkg.is_dir() {
            return Some(dev_pkg);
        }
    }

    None
}

/// Copy a package from source directory to packages directory.
fn copy_package(name: &str, source: &Path, dest_root: &Path, force: bool) -> anyhow::Result<bool> {
    let src = source.join(name).join("init.lua");
    if !src.exists() {
        anyhow::bail!("Source not found: {}", src.display());
    }

    let dest_dir = dest_root.join(name);
    let dest_file = dest_dir.join("init.lua");

    if dest_file.exists() && !force {
        return Ok(false); // Already installed, skip
    }

    std::fs::create_dir_all(&dest_dir)?;
    std::fs::copy(&src, &dest_file)?;
    Ok(true)
}

/// Clone the bundled packages repo at a specific tag and install.
async fn install_from_git(dest: &Path, force: bool) -> anyhow::Result<()> {
    let tag = format!("v{BUNDLED_VERSION}");

    eprintln!("Cloning bundled packages {tag} from {BUNDLED_PACKAGES_URL}...");

    let staging = tempfile::tempdir()?;

    let output = tokio::process::Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--branch",
            &tag,
            BUNDLED_PACKAGES_URL,
            &staging.path().to_string_lossy(),
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git clone failed (tag {tag}): {stderr}");
    }

    install_from_local(staging.path(), dest, force)
}

/// Install from local packages directory.
fn install_from_local(source: &Path, dest: &Path, force: bool) -> anyhow::Result<()> {
    eprintln!("Installing packages from {}...", source.display());

    let mut installed = 0;
    let mut skipped = 0;

    for name in BUNDLED_PACKAGES {
        match copy_package(name, source, dest, force) {
            Ok(true) => {
                eprintln!("  + {name}");
                installed += 1;
            }
            Ok(false) => {
                eprintln!("  = {name} (already installed, use --force to overwrite)");
                skipped += 1;
            }
            Err(e) => {
                eprintln!("  ! {name}: {e}");
            }
        }
    }

    eprintln!("Installed {installed}, skipped {skipped}.");
    Ok(())
}

pub async fn run(args: &[String]) -> anyhow::Result<()> {
    let force = args.iter().any(|a| a == "--force");
    let dev = args.iter().any(|a| a == "--dev");

    let dest = packages_dir()?;
    std::fs::create_dir_all(&dest)?;

    if dev {
        // --dev: force local packages directory
        let source = find_local_packages().ok_or_else(|| {
            anyhow::anyhow!("No local algocline-bundled-packages/ directory found")
        })?;
        return install_from_local(&source, &dest, force);
    }

    // Try git clone first, fall back to local
    match install_from_git(&dest, force).await {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Git clone failed: {e}");
            if let Some(source) = find_local_packages() {
                eprintln!("Falling back to local algocline-bundled-packages/...");
                install_from_local(&source, &dest, force)
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_version_is_valid_semver() {
        assert!(
            BUNDLED_VERSION.split('.').all(|p| p.parse::<u32>().is_ok()),
            "BUNDLED_VERSION must be valid semver: {BUNDLED_VERSION}"
        );
    }

    #[test]
    fn bundled_packages_is_non_empty() {
        assert!(!BUNDLED_PACKAGES.is_empty());
    }

    #[test]
    fn bundled_packages_have_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for name in BUNDLED_PACKAGES {
            assert!(seen.insert(name), "duplicate package: {name}");
        }
    }

    #[test]
    fn bundled_packages_names_are_valid() {
        for name in BUNDLED_PACKAGES {
            assert!(!name.is_empty(), "empty package name");
            assert!(
                !name.contains('/') && !name.contains('\\') && !name.contains(".."),
                "invalid package name: {name}"
            );
            // Must be alphanumeric + underscore (valid Lua module names)
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "non-alphanumeric package name: {name}"
            );
        }
    }

    #[test]
    fn copy_package_creates_init_lua() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Create a source package
        let pkg_dir = source.path().join("mypkg");
        std::fs::create_dir(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let installed = copy_package("mypkg", source.path(), dest.path(), false).unwrap();
        assert!(installed);
        assert!(dest.path().join("mypkg/init.lua").exists());
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {}"
        );
    }

    #[test]
    fn copy_package_skips_existing() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Create source and dest
        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {}").unwrap();

        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "return {old=true}").unwrap();

        let installed = copy_package("mypkg", source.path(), dest.path(), false).unwrap();
        assert!(!installed); // Should skip
                             // Content should NOT be overwritten
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {old=true}"
        );
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

        let installed = copy_package("mypkg", source.path(), dest.path(), true).unwrap();
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

        let result = copy_package("nonexistent", source.path(), dest.path(), false);
        assert!(result.is_err());
    }
}
