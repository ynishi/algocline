//! `alc init` — Install official packages from algocline-packages.
//!
//! Downloads the official package collection from GitHub Releases
//! and installs all packages into `~/.algocline/packages/`.
//!
//! Sources (checked in order):
//! 1. GitHub Releases asset (production): downloads alc-packages-{version}.tar.gz
//! 2. Local packages directory (development): copies directly
//!
//! Usage:
//!   alc init           — Install official packages (from release asset or local)
//!   alc init --force   — Overwrite existing packages
//!   alc init --dev     — Force local source (development)

use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Official package names shipped with algocline.
const OFFICIAL_PACKAGES: &[&str] = &[
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
    "review",
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
/// Searches for a sibling `algocline-packages/` directory relative to CWD
/// or the binary location. This supports the development workflow where
/// both repositories are checked out side by side.
fn find_local_packages() -> Option<PathBuf> {
    // Check CWD/../algocline-packages/
    let cwd = std::env::current_dir().ok()?;
    let sibling = cwd.parent()?.join("algocline-packages");
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
            .join("algocline-packages");
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

/// Download official packages from GitHub Releases.
async fn install_from_release(dest: &Path, _force: bool) -> anyhow::Result<()> {
    let url = format!(
        "https://github.com/yutakanishimura/algocline-packages/releases/download/v{VERSION}/alc-packages-{VERSION}.tar.gz"
    );

    eprintln!("Downloading algocline-packages v{VERSION} from GitHub Releases...");

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
    for name in OFFICIAL_PACKAGES {
        let pkg = dest.join(name).join("init.lua");
        if pkg.exists() {
            count += 1;
            eprintln!("  + {name}");
        }
    }
    eprintln!("Installed {count} packages.");

    Ok(())
}

/// Install from local packages directory.
fn install_from_local(source: &Path, dest: &Path, force: bool) -> anyhow::Result<()> {
    eprintln!("Installing packages from {}...", source.display());

    let mut installed = 0;
    let mut skipped = 0;

    for name in OFFICIAL_PACKAGES {
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
        let source = find_local_packages()
            .ok_or_else(|| anyhow::anyhow!("No local algocline-packages/ directory found"))?;
        return install_from_local(&source, &dest, force);
    }

    // Try GitHub Releases first, fall back to local
    match install_from_release(&dest, force).await {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Release download failed: {e}");
            if let Some(source) = find_local_packages() {
                eprintln!("Falling back to local algocline-packages/...");
                install_from_local(&source, &dest, force)
            } else {
                Err(e)
            }
        }
    }
}
