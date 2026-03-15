//! `alc init` / `alc update` — Install and update bundled packages.
//!
//! Clones the bundled package collection from GitHub (tag-based)
//! and installs all packages into `~/.algocline/packages/`.
//!
//! Packages are discovered dynamically: any subdirectory containing
//! `init.lua` is treated as a package. No hardcoded package list.
//!
//! Sources (checked in order):
//! 1. Git clone with tag `v{BUNDLED_VERSION}` (production)
//! 2. Local packages directory (development fallback)
//!
//! Usage:
//!   alc init             — Install new packages (skip existing)
//!   alc init --force     — Overwrite all packages
//!   alc init --dev       — Force local source (development)
//!   alc update           — Alias for `alc init --force`

use std::path::{Path, PathBuf};

/// Supported bundled packages version.
///
/// Independent of algocline's own CARGO_PKG_VERSION.
/// Updated when a new bundled-packages release is validated.
const BUNDLED_VERSION: &str = "0.2.0";

const BUNDLED_PACKAGES_URL: &str = "https://github.com/ynishi/algocline-bundled-packages";

fn packages_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".algocline").join("packages"))
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

/// Copy a single package directory to dest.
///
/// Uses atomic write (copy to temp → rename) to prevent truncated zombie files.
/// Detects existing zombie files via size mismatch and repairs them on force.
fn copy_package(
    name: &str,
    pkg_source: &Path,
    dest_root: &Path,
    force: bool,
) -> anyhow::Result<bool> {
    let src = pkg_source.join("init.lua");
    if !src.exists() {
        anyhow::bail!("Source not found: {}", src.display());
    }

    let dest_dir = dest_root.join(name);
    let dest_file = dest_dir.join("init.lua");

    if dest_file.exists() && !force {
        // Zombie detection: if dest exists but size mismatches source,
        // it's likely a truncated leftover from a previous failed copy.
        let src_len = std::fs::metadata(&src)?.len();
        let dest_len = std::fs::metadata(&dest_file)?.len();
        if src_len == dest_len {
            return Ok(false); // Healthy file, skip
        }
        // Size mismatch → zombie. Fall through to overwrite.
        eprintln!("    (repairing truncated file for {name})");
    }

    std::fs::create_dir_all(&dest_dir)?;

    // Atomic write: copy to temp file in same directory, then rename.
    // rename() on the same filesystem is atomic on POSIX.
    let tmp_file = dest_dir.join("init.lua.tmp");
    match std::fs::copy(&src, &tmp_file) {
        Ok(_) => {
            std::fs::rename(&tmp_file, &dest_file)?;
        }
        Err(e) => {
            // Clean up partial temp file
            let _ = std::fs::remove_file(&tmp_file);
            return Err(e.into());
        }
    }

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
    fn copy_package_skips_existing_same_size() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        // Same size content — should skip (not detected as zombie)
        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {v=2}").unwrap();

        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "return {v=1}").unwrap();

        let installed = copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();
        assert!(!installed, "same-size file should be skipped");
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {v=1}"
        );
    }

    #[test]
    fn copy_package_repairs_zombie_file() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {complete=true}").unwrap();

        // Create a zombie (truncated) dest file — size mismatch
        let dst_pkg = dest.path().join("mypkg");
        std::fs::create_dir(&dst_pkg).unwrap();
        std::fs::write(dst_pkg.join("init.lua"), "ret").unwrap(); // truncated

        // Without force: zombie is detected and repaired via size mismatch
        let installed = copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();
        assert!(installed, "zombie should be repaired even without --force");
        assert_eq!(
            std::fs::read_to_string(dest.path().join("mypkg/init.lua")).unwrap(),
            "return {complete=true}"
        );
    }

    #[test]
    fn copy_package_no_tmp_file_on_success() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let src_pkg = source.path().join("mypkg");
        std::fs::create_dir(&src_pkg).unwrap();
        std::fs::write(src_pkg.join("init.lua"), "return {}").unwrap();

        copy_package("mypkg", &src_pkg, dest.path(), false).unwrap();

        // Temp file should not remain after successful install
        assert!(!dest.path().join("mypkg/init.lua.tmp").exists());
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
}
