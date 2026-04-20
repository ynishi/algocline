//! Canonical projection of a Lua package's `M.meta` block.
//!
//! `PkgEntity` captures the identity portion of an algocline package: the
//! fields users rely on to discover, categorize, and version-track a package.
//! It is the single source of truth for "what is this package?" and is
//! flattened into higher-level records (`IndexEntry`, `SearchResult`,
//! `hub_info` responses) so the JSON wire shape stays consistent across the
//! Hub, the manifest, and project lockfiles.
//!
//! ## Parsing contract
//!
//! [`PkgEntity::parse_from_init_lua`] is a non-Lua-VM best-effort parser over
//! the `M.meta = { ... }` block of an `init.lua`. It deliberately only
//! supports flat key–value pairs with (possibly concatenated) string
//! literals; nested tables (e.g. `tags = { ... }`) are skipped via
//! brace-depth tracking. When `M.meta.name` is absent or empty the parser
//! returns `None` — this is the **inclusion gate** for hub indexing. The
//! caller (`build_index` in `algocline-app::service::hub`) is expected to
//! drop `None` directories silently so "draft" directories like
//! `alc_shapes/` (a type DSL library, not an algocline package) do not
//! pollute the hub index.
//!
//! ## Wire format
//!
//! `Option` fields use `#[serde(default)]` but deliberately do **not** use
//! `skip_serializing_if`. A missing field deserializes as `None` and
//! serializes back as `null`. This preserves the key-presence guarantee of
//! the current `hub_index.json` consumers (Bundled-side doc generation,
//! `README.md` package-count scripts) so they do not break on field
//! absence.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Canonical projection of a Lua package's `M.meta` block.
///
/// `name` is required (= hub-index inclusion gate). Other fields are
/// optional and degrade UI / discoverability when absent, following the
/// BP convention of Cargo / JSR / npm.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PkgEntity {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub docstring: Option<String>,
}

impl PkgEntity {
    /// Parse `M.meta` + leading `---` docstring from an `init.lua`.
    ///
    /// Returns `None` when the file cannot be read, `M.meta` is absent, or
    /// `M.meta.name` is empty. Callers treat `None` as "not a package" and
    /// drop the directory silently from the hub index.
    ///
    /// The parser is **not** a full Lua evaluator:
    ///
    /// - Only flat key–value pairs inside `M.meta` are extracted.
    /// - Nested tables (e.g. `tags = { ... }`) are skipped via brace-depth
    ///   tracking; their keys are not reachable from here.
    /// - Values must be string literals (`"..."`), optionally joined by `..`
    ///   concatenation with whitespace between operators.
    /// - Occurrences of `M.meta` inside single-line comments (`-- ...`)
    ///   are ignored, so docstrings mentioning the key do not hijack the
    ///   search.
    pub fn parse_from_init_lua(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let (name, version, description, category) = parse_meta(&content)?;
        let docstring = extract_docstring_from(&content);
        Some(PkgEntity {
            name,
            version: option_from_str(version),
            description: option_from_str(description),
            category: option_from_str(category),
            docstring: option_from_str(docstring),
        })
    }
}

/// Return `None` for empty strings, `Some(s)` otherwise. Kept inline with
/// `parse_from_init_lua` so the "empty field = absent" projection rule is
/// applied uniformly to every optional column.
fn option_from_str(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Extract leading `---` doc-comment lines from an init.lua source. Blank
/// lines within the block are tolerated; the first non-doc content line
/// terminates the block.
fn extract_docstring_from(content: &str) -> String {
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("---") {
            lines.push(rest.trim().to_string());
        } else if trimmed.is_empty() {
            continue;
        } else {
            break;
        }
    }
    lines.join("\n")
}

/// Parse `M.meta = { ... }` out of `content`. Returns
/// `(name, version, description, category)`. `None` if the block is
/// missing, unparseable, or `name` is empty.
fn parse_meta(content: &str) -> Option<(String, String, String, String)> {
    let head = content;

    // Find M.meta = { ... } block (with brace-depth tracking).
    // Skip occurrences inside Lua line comments (`-- ...`) so that
    // docstrings mentioning "M.meta" do not hijack the search.
    let mut search_from = 0;
    let meta_start = loop {
        let rel = head[search_from..].find("M.meta")?;
        let pos = search_from + rel;
        let line_start = head[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        if !head[line_start..pos].contains("--") {
            break pos;
        }
        search_from = pos + "M.meta".len();
    };
    let brace_start = head[meta_start..].find('{')? + meta_start;

    // Track brace depth so nested tables do not terminate the block.
    let mut depth = 0;
    let mut brace_end = None;
    for (i, ch) in head[brace_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    brace_end = Some(brace_start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let brace_end = brace_end?;
    let block = &head[brace_start + 1..brace_end];

    let extract = |field: &str| -> String {
        // Match: field = "value" [.. "value" ...] with word-boundary check.
        // Walk through all occurrences of `field`, skipping matches inside
        // longer identifiers (e.g. "short_description"). On the first valid
        // occurrence, collect one or more `"..."` string literals joined by
        // `..` concatenation operators.
        let mut search_from = 0;
        while let Some(rel) = block[search_from..].find(field) {
            let pos = search_from + rel;
            let word_boundary = pos == 0 || {
                let prev = block.as_bytes()[pos - 1];
                !(prev.is_ascii_alphanumeric() || prev == b'_')
            };
            if word_boundary {
                let after = &block[pos + field.len()..];
                let mut collected = String::new();
                let mut cursor = 0usize;
                let mut found_any = false;
                loop {
                    let rest = &after[cursor..];
                    let Some(q_start_rel) = rest.find('"') else {
                        break;
                    };
                    if found_any {
                        // Between the prior closing quote and this opening
                        // quote, only whitespace and a single `..` operator
                        // are allowed. Anything else (comma, another field,
                        // etc.) ends the value.
                        let between = &rest[..q_start_rel];
                        if between.trim() != ".." {
                            break;
                        }
                    }
                    let lit_start = cursor + q_start_rel + 1;
                    let Some(q_end_rel) = after[lit_start..].find('"') else {
                        break;
                    };
                    collected.push_str(&after[lit_start..lit_start + q_end_rel]);
                    cursor = lit_start + q_end_rel + 1;
                    found_any = true;
                }
                if found_any {
                    return collected;
                }
            }
            search_from = pos + field.len();
        }
        String::new()
    };

    let name = extract("name");
    if name.is_empty() {
        return None;
    }
    Some((
        name,
        extract("version"),
        extract("description"),
        extract("category"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_init_lua(dir: &Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("init.lua");
        fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn parse_flat_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "my_pkg",
    version = "1.0.0",
    description = "A test package",
    category = "reasoning",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "my_pkg");
        assert_eq!(pkg.version.as_deref(), Some("1.0.0"));
        assert_eq!(pkg.description.as_deref(), Some("A test package"));
        assert_eq!(pkg.category.as_deref(), Some("reasoning"));
    }

    #[test]
    fn parse_nested_table_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "nested_pkg",
    tags = { "a", "b" },
    description = "After nested",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "nested_pkg");
        assert_eq!(pkg.description.as_deref(), Some("After nested"));
    }

    #[test]
    fn parse_concat_string_literals() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "concat_pkg",
    version = "0.1.0",
    description = "foo "
        .. "bar "
        .. "baz",
    category = "reasoning",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.description.as_deref(), Some("foo bar baz"));
    }

    #[test]
    fn parse_word_boundary_for_description() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "wb_pkg",
    short_description = "should not match",
    description = "correct one",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "wb_pkg");
        assert_eq!(pkg.description.as_deref(), Some("correct one"));
    }

    #[test]
    fn parse_meta_large_leading_docstring() {
        let tmp = tempfile::tempdir().unwrap();
        let mut content = String::new();
        for i in 0..120 {
            content.push_str(&format!("--- line {i}: long doc comment\n"));
        }
        content.push_str(
            r#"
local M = {}
M.meta = {
    name = "late_meta_pkg",
    version = "0.2.0",
    description = "Located past 2KB",
    category = "test",
}
return M
"#,
        );
        assert!(content.len() > 2048, "fixture should exceed 2KB");
        let path = write_init_lua(tmp.path(), &content);

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "late_meta_pkg");
        assert_eq!(pkg.version.as_deref(), Some("0.2.0"));
        assert_eq!(pkg.description.as_deref(), Some("Located past 2KB"));
        assert_eq!(pkg.category.as_deref(), Some("test"));
    }

    #[test]
    fn parse_returns_none_without_meta_block() {
        // Mirrors the alc_shapes case: an init.lua with no M.meta block at
        // all. This is the **silent exclusion gate** — the caller drops
        // these directories from the hub index without warning.
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
--- alc_shapes — type DSL (not a package)
local M = {}
return M
"#,
        );

        assert!(PkgEntity::parse_from_init_lua(&path).is_none());
    }

    #[test]
    fn parse_returns_none_when_name_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "",
    version = "1.0.0",
}
return M
"#,
        );

        assert!(PkgEntity::parse_from_init_lua(&path).is_none());
    }

    #[test]
    fn parse_returns_none_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.lua");
        assert!(PkgEntity::parse_from_init_lua(&path).is_none());
    }

    #[test]
    fn extracts_docstring_and_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"--- cascade — Multi-level routing with confidence gating
--- Based on: "FrugalGPT" (Chen et al., 2023)

local M = {}
M.meta = {
    name = "cascade",
    version = "0.1.0",
    description = "Multi-level routing",
    category = "meta",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "cascade");
        let doc = pkg.docstring.expect("docstring should be present");
        assert!(doc.contains("FrugalGPT"));
        assert!(doc.contains("Multi-level"));
        assert!(!doc.contains("local M"));
    }

    #[test]
    fn docstring_absent_when_no_leading_comments() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"local M = {}
M.meta = { name = "nodoc" }
return M
"#,
        );
        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert!(pkg.docstring.is_none());
    }

    #[test]
    fn m_dot_meta_inside_comment_is_ignored() {
        // A `M.meta` reference inside a `--` comment must not hijack the
        // parser. The real block below it should still be found.
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
-- example: M.meta = { name = "decoy" }
local M = {}
M.meta = {
    name = "real",
}
return M
"#,
        );
        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "real");
    }

    #[test]
    fn serde_round_trip_preserves_none_vs_empty() {
        // Wire format contract: None is serialized as null; empty string
        // deserializes as Some("") (not None). Keep these separable so the
        // consumer can distinguish "field absent" from "field present but
        // empty".
        let pkg = PkgEntity {
            name: "p".into(),
            version: None,
            description: Some(String::new()),
            category: Some("meta".into()),
            docstring: None,
        };
        let json = serde_json::to_string(&pkg).unwrap();
        assert!(json.contains("\"version\":null"), "version null: {json}");
        assert!(
            json.contains("\"description\":\"\""),
            "description empty string: {json}"
        );
        assert!(
            json.contains("\"docstring\":null"),
            "docstring null: {json}"
        );

        let back: PkgEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pkg);
    }

    #[test]
    fn serde_deserialize_accepts_missing_optional_fields() {
        // Legacy hub_index.json entries may omit every optional field;
        // they must deserialize as None (not error).
        let json = r#"{"name":"minimal"}"#;
        let pkg: PkgEntity = serde_json::from_str(json).unwrap();
        assert_eq!(pkg.name, "minimal");
        assert!(pkg.version.is_none());
        assert!(pkg.description.is_none());
        assert!(pkg.category.is_none());
        assert!(pkg.docstring.is_none());
    }
}
