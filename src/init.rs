//! `alc init` — Install Layer 2 first-party packages.
//!
//! Copies the `std/` packages (explore, panel, chain, ensemble, verify)
//! into `~/.algocline/packages/` where the executor's package resolver finds them.
//!
//! Sources (checked in order):
//! 1. GitHub Releases asset (production): downloads alc-std-{version}.tar.gz
//! 2. Local std/ directory (development): copies directly
//!
//! Usage:
//!   alc init           — Install std packages (from release asset or local)
//!   alc init --force   — Overwrite existing std packages
//!   alc init --dev     — Force local std/ source (development)

use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Standard package names. These are the packages shipped with algocline.
const STD_PACKAGES: &[&str] = &["explore", "panel", "chain", "ensemble", "verify"];

fn packages_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".algocline").join("packages"))
}

/// Find the local std/ directory (development).
fn find_local_std() -> Option<PathBuf> {
    // Check CWD/std/
    let cwd = std::env::current_dir().ok()?;
    let cwd_std = cwd.join("std");
    if cwd_std.is_dir() {
        return Some(cwd_std);
    }

    // Check relative to binary: target/debug/../../std/
    if let Ok(exe) = std::env::current_exe() {
        let dev_std = exe.parent()?.parent()?.parent()?.join("std");
        if dev_std.is_dir() {
            return Some(dev_std);
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

/// Download std packages from GitHub Releases.
async fn install_from_release(dest: &Path, _force: bool) -> anyhow::Result<()> {
    let url = format!(
        "https://github.com/yutakanishimura/algocline/releases/download/v{VERSION}/alc-std-{VERSION}.tar.gz"
    );

    eprintln!("Downloading alc-std v{VERSION} from GitHub Releases...");

    let output = tokio::process::Command::new("curl")
        .args(["-fsSL", &url])
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!(
            "Failed to download {url}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Extract tarball: pipe curl stdout into tar stdin
    let mut tar_child = tokio::process::Command::new("tar")
        .args(["xzf", "-", "-C", &dest.to_string_lossy()])
        .stdin(std::process::Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = tar_child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin.write_all(&output.stdout).await?;
        // Drop stdin to close pipe and signal EOF to tar
    }

    let tar_status = tar_child.wait().await?;
    if !tar_status.success() {
        anyhow::bail!("Failed to extract tarball");
    }

    // Report
    let mut count = 0;
    for name in STD_PACKAGES {
        let pkg = dest.join(name).join("init.lua");
        if pkg.exists() {
            count += 1;
            eprintln!("  + {name}");
        }
    }
    eprintln!("Installed {count} std packages.");

    // _force: tarball extraction always overwrites. install_from_local
    // handles the skip-if-exists logic when --force is absent.
    Ok(())
}

/// Install from local std/ directory.
fn install_from_local(source: &Path, dest: &Path, force: bool) -> anyhow::Result<()> {
    eprintln!("Installing std packages from {}...", source.display());

    let mut installed = 0;
    let mut skipped = 0;

    for name in STD_PACKAGES {
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
        // --dev: force local std/
        let source =
            find_local_std().ok_or_else(|| anyhow::anyhow!("No local std/ directory found"))?;
        return install_from_local(&source, &dest, force);
    }

    // Try GitHub Releases first, fall back to local
    match install_from_release(&dest, force).await {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Release download failed: {e}");
            if let Some(source) = find_local_std() {
                eprintln!("Falling back to local std/...");
                install_from_local(&source, &dest, force)
            } else {
                Err(e)
            }
        }
    }
}
