//! `alc.toml` — project package declaration file.
// Functions here are used in subtask 2+ (pkg_install, alc_init, etc.)
#![allow(dead_code)]
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

// ─── Write ───────────────────────────────────────────────────────────────────

/// Write a `toml_edit::DocumentMut` back to `alc.toml` (comment-preserving).
pub(crate) fn save_alc_toml(
    project_root: &Path,
    doc: &toml_edit::DocumentMut,
) -> Result<(), String> {
    let path = alc_toml_path(project_root);
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Cannot determine parent directory for alc.toml at {}",
            path.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create directory for alc.toml: {e}"))?;

    std::fs::write(&path, doc.to_string())
        .map_err(|e| format!("Failed to write alc.toml at {}: {e}", path.display()))?;

    Ok(())
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
}
