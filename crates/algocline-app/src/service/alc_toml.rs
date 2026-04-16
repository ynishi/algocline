//! `alc.toml` — project package declaration file.
//!
//! ## File location
//! `alc.toml` lives at the project root.
//!
//! ## Schema example
//! ```toml
//! [packages]
//! coding_orch = "*"
//! flow_design = "0.2.0"
//!
//! [packages.head_agent]
//! path = "packages/head_agent"
//!
//! [packages.my_pkg]
//! git = "https://github.com/user/my-pkg"
//! rev = "abc123"
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ─── Types ─────────────────────────────────────────────────────────────────

/// Top-level structure of `alc.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct AlcToml {
    #[serde(default)]
    pub packages: BTreeMap<String, PackageDep>,
}

/// A single package dependency declaration.
///
/// Uses `#[serde(untagged)]` — `Version` must come first so that a plain
/// string is matched before the struct variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum PackageDep {
    /// `"*"` or `"0.2.0"` — resolve from installed cache.
    Version(String),
    /// `{ path = "..." }` — local directory.
    Path {
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    /// `{ git = "..." }` — Git source (future).
    Git {
        git: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        rev: Option<String>,
    },
}

// ─── Paths ──────────────────────────────────────────────────────────────────

pub(crate) fn alc_toml_path(project_root: &Path) -> PathBuf {
    project_root.join("alc.toml")
}

/// `alc.local.toml` — worktree-scoped override file (Wave 1).
///
/// Physical filename uses `local` per the dotenv convention signal
/// (git-ignored), but the logical scope is "variant" (sub-unit of repo).
pub(crate) fn local_alc_toml_path(project_root: &Path) -> PathBuf {
    project_root.join("alc.local.toml")
}

// ─── Read ────────────────────────────────────────────────────────────────────

/// Load and parse `alc.toml` using serde.
///
/// Returns `Ok(None)` if the file does not exist.
pub(crate) fn load_alc_toml(project_root: &Path) -> Result<Option<AlcToml>, String> {
    let path = alc_toml_path(project_root);
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read alc.toml at {}: {e}", path.display()))?;

    let parsed: AlcToml = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse alc.toml at {}: {e}", path.display()))?;

    Ok(Some(parsed))
}

/// Load `alc.toml` as a raw `toml_edit::DocumentMut` (preserves comments/formatting).
///
/// Returns `Ok(None)` if the file does not exist.
pub(crate) fn load_alc_toml_document(
    project_root: &Path,
) -> Result<Option<toml_edit::DocumentMut>, String> {
    let path = alc_toml_path(project_root);
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read alc.toml at {}: {e}", path.display()))?;

    let doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|e| format!("Failed to parse alc.toml at {}: {e}", path.display()))?;

    Ok(Some(doc))
}

/// Load and parse `alc.local.toml` using serde.
///
/// Returns `Ok(None)` if the file does not exist.
pub(crate) fn load_alc_local_toml(project_root: &Path) -> Result<Option<AlcToml>, String> {
    let path = local_alc_toml_path(project_root);
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read alc.local.toml at {}: {e}", path.display()))?;

    let parsed: AlcToml = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse alc.local.toml at {}: {e}", path.display()))?;

    Ok(Some(parsed))
}

/// Load `alc.local.toml` as a raw `toml_edit::DocumentMut` (preserves comments/formatting).
///
/// Returns `Ok(None)` if the file does not exist.
pub(crate) fn load_alc_local_toml_document(
    project_root: &Path,
) -> Result<Option<toml_edit::DocumentMut>, String> {
    let path = local_alc_toml_path(project_root);
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read alc.local.toml at {}: {e}", path.display()))?;

    let doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|e| format!("Failed to parse alc.local.toml at {}: {e}", path.display()))?;

    Ok(Some(doc))
}

// ─── Write ───────────────────────────────────────────────────────────────────

/// Internal helper: write a `toml_edit::DocumentMut` to an arbitrary path.
///
/// Shared by `save_alc_toml` and `save_alc_local_toml`.
fn save_alc_toml_at(path: &Path, doc: &toml_edit::DocumentMut) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Cannot determine parent directory for {}",
            path.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create directory for {}: {e}", path.display()))?;

    std::fs::write(path, doc.to_string())
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;

    Ok(())
}

/// Write a `toml_edit::DocumentMut` back to `alc.toml` (comment-preserving).
pub(crate) fn save_alc_toml(
    project_root: &Path,
    doc: &toml_edit::DocumentMut,
) -> Result<(), String> {
    save_alc_toml_at(&alc_toml_path(project_root), doc)
}

/// Write a `toml_edit::DocumentMut` back to `alc.local.toml` (comment-preserving).
pub(crate) fn save_alc_local_toml(
    project_root: &Path,
    doc: &toml_edit::DocumentMut,
) -> Result<(), String> {
    save_alc_toml_at(&local_alc_toml_path(project_root), doc)
}

// ─── Local-path entry resolution ─────────────────────────────────────────────

/// Resolve `[packages.*] path = "..."` entries from `alc.local.toml` to
/// absolute paths.
///
/// - Only `PackageDep::Path` variants contribute; `Version` / `Git` are
///   silently skipped (reserved for future use).
/// - Relative paths are joined onto `project_root`; absolute paths are
///   taken as-is.
/// - Entries whose resolved path does not exist are skipped with a
///   `tracing::warn!`.
/// - Iteration order follows `BTreeMap` (= package-name ascending).
pub(crate) fn resolve_local_path_entries(
    project_root: &Path,
    local: &AlcToml,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for (name, dep) in &local.packages {
        let raw = match dep {
            PackageDep::Path { path, .. } => path,
            _ => continue,
        };

        let resolved = {
            let p = Path::new(raw);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                project_root.join(p)
            }
        };

        if !resolved.exists() {
            tracing::warn!(
                "alc.local.toml: path entry for '{}' does not exist, skipping: {}",
                name,
                resolved.display()
            );
            continue;
        }

        paths.push(resolved);
    }

    paths
}

// ─── Entry manipulation ──────────────────────────────────────────────────────

/// Add a package entry to `[packages]` in the document.
///
/// Returns `true` if the entry was added, `false` if it already existed (skip).
pub(crate) fn add_package_entry(
    doc: &mut toml_edit::DocumentMut,
    name: &str,
    dep: &PackageDep,
) -> bool {
    use toml_edit::{value, Item, Table};

    // Ensure [packages] table exists.
    if doc.get("packages").is_none() {
        doc.insert("packages", Item::Table(Table::new()));
    }

    let packages = match doc["packages"].as_table_mut() {
        Some(t) => t,
        None => return false,
    };

    // Already exists — skip.
    if packages.contains_key(name) {
        return false;
    }

    match dep {
        PackageDep::Version(v) => {
            packages.insert(name, value(v.as_str()));
        }
        PackageDep::Path { path, version: ver } => {
            let mut tbl = toml_edit::InlineTable::new();
            tbl.insert("path", path.as_str().into());
            if let Some(v) = ver {
                tbl.insert("version", v.as_str().into());
            }
            packages.insert(name, Item::Value(toml_edit::Value::InlineTable(tbl)));
        }
        PackageDep::Git { git, rev } => {
            let mut tbl = toml_edit::InlineTable::new();
            tbl.insert("git", git.as_str().into());
            if let Some(r) = rev {
                tbl.insert("rev", r.as_str().into());
            }
            packages.insert(name, Item::Value(toml_edit::Value::InlineTable(tbl)));
        }
    }

    true
}

/// Remove a package entry from `[packages]` in the document.
///
/// Returns `true` if the entry was removed, `false` if it did not exist.
pub(crate) fn remove_package_entry(doc: &mut toml_edit::DocumentMut, name: &str) -> bool {
    let packages = match doc.get_mut("packages").and_then(|i| i.as_table_mut()) {
        Some(t) => t,
        None => return false,
    };
    packages.remove(name).is_some()
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate a package name: must match `[a-zA-Z][a-zA-Z0-9_-]*`.
pub(crate) fn validate_package_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("package name must not be empty".to_string());
    }

    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() {
        return Err(format!(
            "package name must start with a letter, got '{first}'"
        ));
    }

    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' && c != '-' {
            return Err(format!(
                "package name contains invalid character '{c}': only [a-zA-Z0-9_-] allowed"
            ));
        }
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse tests ─────────────────────────────────────────────────

    #[test]
    fn parse_version_dep() {
        let toml = r#"
[packages]
cot = "*"
flow = "0.2.0"
"#;
        let parsed: AlcToml = toml::from_str(toml).unwrap();
        assert_eq!(parsed.packages["cot"], PackageDep::Version("*".to_string()));
        assert_eq!(
            parsed.packages["flow"],
            PackageDep::Version("0.2.0".to_string())
        );
    }

    #[test]
    fn parse_path_dep() {
        let toml = r#"
[packages.head_agent]
path = "packages/head_agent"
"#;
        let parsed: AlcToml = toml::from_str(toml).unwrap();
        assert_eq!(
            parsed.packages["head_agent"],
            PackageDep::Path {
                path: "packages/head_agent".to_string(),
                version: None,
            }
        );
    }

    #[test]
    fn parse_git_dep() {
        let toml = r#"
[packages.my_pkg]
git = "https://github.com/user/my-pkg"
rev = "abc123"
"#;
        let parsed: AlcToml = toml::from_str(toml).unwrap();
        assert_eq!(
            parsed.packages["my_pkg"],
            PackageDep::Git {
                git: "https://github.com/user/my-pkg".to_string(),
                rev: Some("abc123".to_string()),
            }
        );
    }

    #[test]
    fn parse_mixed() {
        let toml = r#"
[packages]
cot = "*"

[packages.head_agent]
path = "packages/head_agent"
version = "0.3.0"

[packages.my_pkg]
git = "https://github.com/user/my-pkg"
"#;
        let parsed: AlcToml = toml::from_str(toml).unwrap();
        assert_eq!(parsed.packages.len(), 3);
        assert_eq!(parsed.packages["cot"], PackageDep::Version("*".to_string()));
        assert_eq!(
            parsed.packages["head_agent"],
            PackageDep::Path {
                path: "packages/head_agent".to_string(),
                version: Some("0.3.0".to_string()),
            }
        );
        assert_eq!(
            parsed.packages["my_pkg"],
            PackageDep::Git {
                git: "https://github.com/user/my-pkg".to_string(),
                rev: None,
            }
        );
    }

    #[test]
    fn parse_invalid_format() {
        let toml = r#"
[packages]
invalid_key = 42
"#;
        let result: Result<AlcToml, _> = toml::from_str(toml);
        assert!(result.is_err());
    }

    // ── load/save roundtrip ──────────────────────────────────────────

    #[test]
    fn load_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_alc_toml(tmp.path()).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn load_and_parse() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alc.toml"), "[packages]\ncot = \"*\"\n").unwrap();

        let parsed = load_alc_toml(tmp.path()).unwrap().unwrap();
        assert_eq!(parsed.packages["cot"], PackageDep::Version("*".to_string()));
    }

    #[test]
    fn document_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let content = "# comment\n[packages]\ncot = \"*\"\n";
        std::fs::write(tmp.path().join("alc.toml"), content).unwrap();

        let doc = load_alc_toml_document(tmp.path()).unwrap().unwrap();
        save_alc_toml(tmp.path(), &doc).unwrap();

        let after = std::fs::read_to_string(tmp.path().join("alc.toml")).unwrap();
        // comment must be preserved
        assert!(after.contains("# comment"), "comment was lost: {after}");
        assert!(after.contains("cot"), "cot entry was lost");
    }

    // ── add/remove entry ─────────────────────────────────────────────

    #[test]
    fn add_version_entry() {
        let mut doc: toml_edit::DocumentMut = "[packages]\n".parse().unwrap();
        let added = add_package_entry(&mut doc, "cot", &PackageDep::Version("*".to_string()));
        assert!(added);
        assert!(doc.to_string().contains("cot"));
    }

    #[test]
    fn add_path_entry() {
        let mut doc: toml_edit::DocumentMut = "[packages]\n".parse().unwrap();
        let added = add_package_entry(
            &mut doc,
            "head",
            &PackageDep::Path {
                path: "packages/head".to_string(),
                version: None,
            },
        );
        assert!(added);
        assert!(doc.to_string().contains("head"));
        assert!(doc.to_string().contains("packages/head"));
    }

    #[test]
    fn add_skips_existing() {
        let mut doc: toml_edit::DocumentMut = "[packages]\ncot = \"*\"\n".parse().unwrap();
        let added = add_package_entry(&mut doc, "cot", &PackageDep::Version("0.1.0".to_string()));
        assert!(!added, "should skip existing entry");
        // value unchanged
        assert!(doc.to_string().contains("\"*\""));
    }

    #[test]
    fn add_creates_packages_table() {
        let mut doc: toml_edit::DocumentMut = "".parse().unwrap();
        let added = add_package_entry(&mut doc, "cot", &PackageDep::Version("*".to_string()));
        assert!(added);
        assert!(doc.to_string().contains("[packages]"));
    }

    #[test]
    fn remove_entry() {
        let mut doc: toml_edit::DocumentMut = "[packages]\ncot = \"*\"\n".parse().unwrap();
        let removed = remove_package_entry(&mut doc, "cot");
        assert!(removed);
        assert!(!doc.to_string().contains("cot"));
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut doc: toml_edit::DocumentMut = "[packages]\n".parse().unwrap();
        let removed = remove_package_entry(&mut doc, "nonexistent");
        assert!(!removed);
    }

    // ── validate_package_name ────────────────────────────────────────

    #[test]
    fn valid_names() {
        assert!(validate_package_name("cot").is_ok());
        assert!(validate_package_name("head_agent").is_ok());
        assert!(validate_package_name("my-pkg").is_ok());
        assert!(validate_package_name("A123").is_ok());
    }

    #[test]
    fn invalid_names() {
        assert!(validate_package_name("").is_err());
        assert!(validate_package_name("1start").is_err());
        assert!(validate_package_name("_start").is_err());
        assert!(validate_package_name("has space").is_err());
        assert!(validate_package_name("has.dot").is_err());
    }

    // ── alc.local.toml: path / load / save ────────────────────────────

    #[test]
    fn local_alc_toml_path_joins_filename() {
        let root = Path::new("/tmp/some/project");
        assert_eq!(
            local_alc_toml_path(root),
            PathBuf::from("/tmp/some/project/alc.local.toml")
        );
    }

    #[test]
    fn load_alc_local_toml_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_alc_local_toml(tmp.path()).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn load_alc_local_toml_parses_path_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("target-dir");
        std::fs::create_dir_all(&target).unwrap();
        let content = format!(
            "[packages.cot]\npath = \"{}\"\n",
            target.display()
        );
        std::fs::write(tmp.path().join("alc.local.toml"), &content).unwrap();

        let parsed = load_alc_local_toml(tmp.path()).unwrap().unwrap();
        assert_eq!(
            parsed.packages["cot"],
            PackageDep::Path {
                path: target.display().to_string(),
                version: None,
            }
        );
    }

    #[test]
    fn save_alc_local_toml_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut doc: toml_edit::DocumentMut = "[packages]\n".parse().unwrap();
        add_package_entry(
            &mut doc,
            "foo",
            &PackageDep::Path {
                path: "/abs/path".to_string(),
                version: None,
            },
        );
        save_alc_local_toml(tmp.path(), &doc).unwrap();

        let written = std::fs::read_to_string(tmp.path().join("alc.local.toml")).unwrap();
        assert!(written.contains("foo"));
        assert!(written.contains("/abs/path"));
    }

    #[test]
    fn save_alc_local_toml_preserves_comments() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("alc.local.toml"),
            "# worktree override\n[packages.foo]\npath = \"/abs\"\n",
        )
        .unwrap();

        let doc = load_alc_local_toml_document(tmp.path()).unwrap().unwrap();
        save_alc_local_toml(tmp.path(), &doc).unwrap();

        let after = std::fs::read_to_string(tmp.path().join("alc.local.toml")).unwrap();
        assert!(after.contains("# worktree override"));
        assert!(after.contains("foo"));
    }

    // ── resolve_local_path_entries ────────────────────────────────────

    #[test]
    fn resolve_local_path_entries_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let abs = tmp.path().join("dev").join("cot");
        std::fs::create_dir_all(&abs).unwrap();

        let mut packages = BTreeMap::new();
        packages.insert(
            "cot".to_string(),
            PackageDep::Path {
                path: abs.display().to_string(),
                version: None,
            },
        );
        let local = AlcToml { packages };

        let resolved = resolve_local_path_entries(tmp.path(), &local);
        assert_eq!(resolved, vec![abs]);
    }

    #[test]
    fn resolve_local_path_entries_relative_path() {
        let tmp = tempfile::tempdir().unwrap();
        let rel = PathBuf::from("rel/sub");
        let abs = tmp.path().join(&rel);
        std::fs::create_dir_all(&abs).unwrap();

        let mut packages = BTreeMap::new();
        packages.insert(
            "sub".to_string(),
            PackageDep::Path {
                path: "rel/sub".to_string(),
                version: None,
            },
        );
        let local = AlcToml { packages };

        let resolved = resolve_local_path_entries(tmp.path(), &local);
        assert_eq!(resolved, vec![abs]);
    }

    #[test]
    fn resolve_local_path_entries_missing_path_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let mut packages = BTreeMap::new();
        packages.insert(
            "ghost".to_string(),
            PackageDep::Path {
                path: "/definitely/does/not/exist/zzz".to_string(),
                version: None,
            },
        );
        let local = AlcToml { packages };

        let resolved = resolve_local_path_entries(tmp.path(), &local);
        assert!(resolved.is_empty());
    }

    #[test]
    fn resolve_local_path_entries_ignores_version_and_git() {
        let tmp = tempfile::tempdir().unwrap();
        let abs = tmp.path().join("real");
        std::fs::create_dir_all(&abs).unwrap();

        let mut packages = BTreeMap::new();
        packages.insert("v".to_string(), PackageDep::Version("*".to_string()));
        packages.insert(
            "g".to_string(),
            PackageDep::Git {
                git: "https://example/repo".to_string(),
                rev: None,
            },
        );
        packages.insert(
            "p".to_string(),
            PackageDep::Path {
                path: abs.display().to_string(),
                version: None,
            },
        );
        let local = AlcToml { packages };

        let resolved = resolve_local_path_entries(tmp.path(), &local);
        assert_eq!(resolved, vec![abs]);
    }

    #[test]
    fn resolve_local_path_entries_ordering_is_name_ascending() {
        let tmp = tempfile::tempdir().unwrap();
        let ucb = tmp.path().join("ucb");
        let cot = tmp.path().join("cot");
        std::fs::create_dir_all(&ucb).unwrap();
        std::fs::create_dir_all(&cot).unwrap();

        let mut packages = BTreeMap::new();
        packages.insert(
            "ucb".to_string(),
            PackageDep::Path {
                path: ucb.display().to_string(),
                version: None,
            },
        );
        packages.insert(
            "cot".to_string(),
            PackageDep::Path {
                path: cot.display().to_string(),
                version: None,
            },
        );
        let local = AlcToml { packages };

        let resolved = resolve_local_path_entries(tmp.path(), &local);
        // BTreeMap iterates keys in ascending order → cot first, ucb second.
        assert_eq!(resolved, vec![cot, ucb]);
    }
}
