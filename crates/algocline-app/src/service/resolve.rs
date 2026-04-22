use std::path::{Path, PathBuf};

use algocline_core::AppDir;

use super::path::ContainedPath;

// ─── Search path (package resolution chain) ─────────────────────

/// Source of a package search path entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchPathSource {
    /// From `ALC_PACKAGES_PATH` environment variable.
    Env,
    /// Default `~/.algocline/packages/`.
    Default,
}

impl std::fmt::Display for SearchPathSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Env => write!(f, "ALC_PACKAGES_PATH"),
            Self::Default => write!(f, "default"),
        }
    }
}

/// A package search path with its origin, ordered by priority (first = highest).
#[derive(Clone, Debug)]
pub struct SearchPath {
    pub path: PathBuf,
    pub source: SearchPathSource,
}

impl SearchPath {
    pub fn env(path: PathBuf) -> Self {
        Self {
            path,
            source: SearchPathSource::Env,
        }
    }

    pub fn default_global(path: PathBuf) -> Self {
        Self {
            path,
            source: SearchPathSource::Default,
        }
    }
}

// Re-export from core for backward compatibility.
pub use algocline_core::QueryResponse;

// ─── Code resolution ────────────────────────────────────────────

pub(crate) fn resolve_code(
    code: Option<String>,
    code_file: Option<String>,
) -> Result<String, String> {
    match (code, code_file) {
        (Some(c), None) => Ok(c),
        (None, Some(path)) => std::fs::read_to_string(Path::new(&path))
            .map_err(|e| format!("Failed to read {path}: {e}")),
        (Some(_), Some(_)) => Err("Provide either `code` or `code_file`, not both.".into()),
        (None, None) => Err("Either `code` or `code_file` must be provided.".into()),
    }
}

/// Build Lua code that loads a package by name and calls `pkg.run(ctx)`.
///
/// # Security: `name` is not sanitized
///
/// `name` is interpolated directly into a Lua `require()` call without
/// sanitization. This is intentional in the current architecture:
///
/// - algocline is a **local development/execution tool** that runs Lua in
///   the user's own environment via mlua (not a multi-tenant service).
/// - The same caller has access to `alc_run`, which executes **arbitrary
///   Lua code**. Sanitizing `name` here would not reduce the attack surface.
/// - The MCP trust boundary lies at the **host/client** level — the host
///   decides whether to invoke `alc_advice` at all.
///
/// If algocline is extended to a shared backend (e.g. a package registry
/// server accepting untrusted strategy names), `name` **must** be validated
/// (allowlist of `[a-zA-Z0-9_-]` or equivalent) before interpolation.
///
/// References:
/// - [MCP Security Best Practices — Local MCP Server Compromise](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
/// - [OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html)
pub(crate) fn make_require_code(name: &str) -> String {
    format!(
        r#"local pkg = require("{name}")
return pkg.run(ctx)"#
    )
}

pub(crate) fn types_stub_path(app_dir: &AppDir) -> Option<String> {
    let p = app_dir.types_dir().join("alc.d.lua");
    if p.exists() {
        Some(p.display().to_string())
    } else {
        None
    }
}

pub(crate) fn packages_dir(app_dir: &AppDir) -> PathBuf {
    app_dir.packages_dir()
}

pub(crate) fn scenarios_dir(app_dir: &AppDir) -> PathBuf {
    app_dir.scenarios_dir()
}

/// Resolve scenario code from one of three mutually exclusive sources:
/// inline code, file path, or scenario name (looked up in `{app_dir}/scenarios/`).
pub(crate) fn resolve_scenario_code(
    app_dir: &AppDir,
    scenario: Option<String>,
    scenario_file: Option<String>,
    scenario_name: Option<String>,
) -> Result<String, String> {
    match (scenario, scenario_file, scenario_name) {
        (Some(c), None, None) => Ok(c),
        (None, Some(path), None) => std::fs::read_to_string(Path::new(&path))
            .map_err(|e| format!("Failed to read {path}: {e}")),
        (None, None, Some(name)) => {
            let dir = scenarios_dir(app_dir);
            let path = ContainedPath::child(&dir, &format!("{name}.lua"))
                .map_err(|e| format!("Invalid scenario name: {e}"))?;
            if !path.as_ref().exists() {
                return Err(format!(
                    "Scenario '{name}' not found at {}",
                    path.as_ref().display()
                ));
            }
            std::fs::read_to_string(path.as_ref())
                .map_err(|e| format!("Failed to read scenario '{name}': {e}"))
        }
        (None, None, None) => {
            Err("Provide one of: scenario, scenario_file, or scenario_name.".into())
        }
        _ => Err(
            "Provide only one of: scenario, scenario_file, or scenario_name (not multiple).".into(),
        ),
    }
}

/// Git URLs for auto-installation. Collection repos contain multiple packages
/// as subdirectories; single repos have init.lua at root.
pub(super) const AUTO_INSTALL_SOURCES: &[&str] = &[
    "https://github.com/ynishi/algocline-bundled-packages",
    "https://github.com/ynishi/evalframe",
];

/// System packages: installed alongside user packages but not user-facing strategies.
/// Excluded from `pkg_list` and not loaded via `require` for meta extraction.
const SYSTEM_PACKAGES: &[&str] = &["evalframe"];

/// Check whether a package is a system (non-user-facing) package.
pub(super) fn is_system_package(name: &str) -> bool {
    SYSTEM_PACKAGES.contains(&name)
}

/// Check whether a package is installed (has `init.lua`).
pub(super) fn is_package_installed(app_dir: &AppDir, name: &str) -> bool {
    packages_dir(app_dir).join(name).join("init.lua").exists()
}

/// Per-entry I/O failures collected during resilient batch operations.
///
/// **Resilience pattern:** Directory iteration and file operations may encounter
/// per-entry I/O errors (permission denied, broken symlinks, etc.) that should
/// not abort the entire operation. Failures are collected and returned alongside
/// successful results so the caller has both the available data and diagnostics.
///
/// Included in JSON responses as `"failures": [...]`.
pub(super) type DirEntryFailures = Vec<String>;

/// Extract a display name from a path: file_stem if available, otherwise file_name.
pub(super) fn display_name(path: &Path, file_name: &str) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(String::from)
        .unwrap_or_else(|| file_name.to_string())
}

/// Determine the scenario source directory within a cloned/downloaded tree.
///
/// Prefers a `scenarios/` subdirectory when present, falling back to the root.
///
/// # `.git` and other non-Lua entries
///
/// When falling back to the root, the directory may contain `.git/`, `README.md`,
/// `LICENSE`, etc. This is safe because [`install_scenarios_from_dir`] applies two
/// filters: `is_file()` (excludes `.git/` and other subdirectories) and
/// `.lua` extension check (excludes non-Lua files). No explicit `.git` exclusion
/// is needed.
pub(super) fn resolve_scenario_source(clone_root: &Path) -> PathBuf {
    let subdir = clone_root.join("scenarios");
    if subdir.is_dir() {
        subdir
    } else {
        clone_root.to_path_buf()
    }
}

/// Copy all `.lua` files from `source` directory into `dest` (scenarios dir).
/// Skips files that already exist. Collects per-entry I/O errors as `failures`
/// rather than aborting.
pub(super) fn install_scenarios_from_dir(source: &Path, dest: &Path) -> Result<String, String> {
    let entries =
        std::fs::read_dir(source).map_err(|e| format!("Failed to read source dir: {e}"))?;

    let mut installed = Vec::new();
    let mut skipped = Vec::new();
    let mut failures: DirEntryFailures = Vec::new();

    for entry_result in entries {
        let entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                failures.push(format!("readdir entry: {e}"));
                continue;
            }
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str());
        if ext != Some("lua") {
            continue;
        }
        let file_name = entry.file_name().to_string_lossy().to_string();
        let dest_path = match ContainedPath::child(dest, &file_name) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let name = display_name(&path, &file_name);
        if dest_path.as_ref().exists() {
            skipped.push(name);
            continue;
        }
        match std::fs::copy(&path, dest_path.as_ref()) {
            Ok(_) => installed.push(name),
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }

    if installed.is_empty() && skipped.is_empty() && failures.is_empty() {
        return Err("No .lua scenario files found in source.".into());
    }

    Ok(serde_json::json!({
        "installed": installed,
        "skipped": skipped,
        "failures": failures,
    })
    .to_string())
}
