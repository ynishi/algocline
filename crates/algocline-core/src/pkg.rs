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
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Package type discriminator: runnable (has `M.run`) or library (API surface only).
///
/// Used by the adviser and eval entry points to route packages correctly:
/// - `Runnable` packages are executed via `M.run(ctx)`.
/// - `Library` packages expose an API surface and must not be executed via `M.run`.
///
/// Wire format: `"runnable"` / `"library"` (lowercase via `#[serde(rename_all = "lowercase")]`).
///
/// Auto-detection: when `M.meta.type` is absent, the Lua VM path uses
/// `type(pkg.run) == "function"` at runtime (canonical), while the offline
/// `parse_from_init_lua` path uses `detect_has_run` (text-scan mirror).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PkgType {
    /// Package has a `M.run(ctx)` entry point — can be executed via `alc_advice`.
    Runnable,
    /// Package exposes an API surface only — must not be invoked via `M.run`.
    Library,
}

impl std::fmt::Display for PkgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Runnable => write!(f, "runnable"),
            Self::Library => write!(f, "library"),
        }
    }
}

impl FromStr for PkgType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "runnable" => Ok(Self::Runnable),
            "library" => Ok(Self::Library),
            other => Err(format!("unknown package type: {other:?}")),
        }
    }
}

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
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Package type (`"runnable"` or `"library"`). Serialized as `"type"` on
    /// the wire. `None` means the type was not determined (legacy entries).
    #[serde(default, rename = "type")]
    pub pkg_type: Option<PkgType>,
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
        let parsed = parse_meta(&content)?;
        let docstring = extract_docstring_from(&content);
        // Resolve type: use explicit M.meta.type if present; otherwise fall
        // back to Rust text-scan (build_index / offline batch path).
        let pkg_type = parsed.pkg_type.or_else(|| {
            Some(if detect_has_run(&content) {
                PkgType::Runnable
            } else {
                PkgType::Library
            })
        });
        Some(PkgEntity {
            name: parsed.name,
            version: option_from_str(parsed.version),
            description: option_from_str(parsed.description),
            category: option_from_str(parsed.category),
            docstring: option_from_str(docstring),
            tags: if parsed.tags.is_empty() {
                None
            } else {
                Some(parsed.tags)
            },
            pkg_type,
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

/// Intermediate result of parsing `M.meta = { ... }` from an `init.lua`.
/// All fields are owned. `pkg_type` is `None` when the `type` key is absent
/// or has an unrecognized value (caller applies auto-detect fallback).
struct ParsedMeta {
    name: String,
    version: String,
    description: String,
    category: String,
    tags: Vec<String>,
    pkg_type: Option<PkgType>,
}

/// Detect whether the init.lua source declares `M.run`.
///
/// Used by `parse_from_init_lua` as the offline (non-VM) fallback for
/// `build_index` when `M.meta.type` is absent. The Lua VM path (`pkg_list`,
/// `resolve_pkg_type_lua`) uses `type(pkg.run) == "function"` at runtime and
/// is the canonical source; this function is the Rust mirror for offline batch.
///
/// Comment exclusion: for each line, only the portion before the first `--`
/// is considered, so `-- M.run = ...` or `-- function M.run(ctx)` do not
/// trigger a match.
fn detect_has_run(content: &str) -> bool {
    for line in content.lines() {
        // Strip inline comment suffix.
        let effective = line.split("--").next().unwrap_or(line);
        if effective.contains("M.run") {
            return true;
        }
    }
    false
}

/// Parse `M.meta = { ... }` out of `content`. Returns `None` if the block is
/// missing, unparseable, or `name` is empty.
fn parse_meta(content: &str) -> Option<ParsedMeta> {
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
    let tags = extract_string_array(block, "tags");
    // Parse `type` field; unrecognized values fall back to None (auto-detect).
    let pkg_type = {
        let raw = extract("type");
        if raw.is_empty() {
            None
        } else {
            raw.parse::<PkgType>().ok()
        }
    };
    Some(ParsedMeta {
        name,
        version: extract("version"),
        description: extract("description"),
        category: extract("category"),
        tags,
        pkg_type,
    })
}

/// Extract a string array from a nested table like `tags = { "a", "b" }`.
/// Returns an empty Vec if the field is absent or has no string elements.
fn extract_string_array(block: &str, field: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = block[search_from..].find(field) {
        let pos = search_from + rel;
        let word_boundary = pos == 0 || {
            let prev = block.as_bytes()[pos - 1];
            !(prev.is_ascii_alphanumeric() || prev == b'_')
        };
        if word_boundary {
            let after = &block[pos + field.len()..];
            if let Some(brace_start) = after.find('{') {
                let inner_start = brace_start + 1;
                let mut depth = 1;
                let mut brace_end = None;
                for (i, ch) in after[inner_start..].char_indices() {
                    match ch {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                brace_end = Some(inner_start + i);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(end) = brace_end {
                    let inner = &after[inner_start..end];
                    let mut cursor = 0;
                    while let Some(q_start) = inner[cursor..].find('"') {
                        let lit_start = cursor + q_start + 1;
                        if let Some(q_end) = inner[lit_start..].find('"') {
                            let s = &inner[lit_start..lit_start + q_end];
                            if !s.is_empty() {
                                result.push(s.to_string());
                            }
                            cursor = lit_start + q_end + 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            break;
        }
        search_from = pos + field.len();
    }
    result
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
    fn parse_tags_from_nested_table() {
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
        assert_eq!(
            pkg.tags.as_deref(),
            Some(vec!["a".to_string(), "b".to_string()].as_slice())
        );
    }

    #[test]
    fn parse_tags_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "no_tags_pkg",
    description = "No tags",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "no_tags_pkg");
        assert!(pkg.tags.is_none());
    }

    #[test]
    fn parse_tags_empty_array() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "empty_tags_pkg",
    tags = {},
    description = "Empty tags",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "empty_tags_pkg");
        assert!(pkg.tags.is_none());
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
            tags: None,
            pkg_type: None,
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

    // ─── PkgType / pkg_type field tests ──────────────────────────

    #[test]
    fn parse_type_from_meta() {
        // Acceptance criterion 5: M.meta.type = "library" → PkgType::Library
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "lib_pkg",
    type = "library",
    version = "1.0.0",
}
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.name, "lib_pkg");
        assert_eq!(pkg.pkg_type, Some(PkgType::Library));
    }

    #[test]
    fn parse_type_runnable_explicit() {
        // Acceptance criterion 6: M.meta.type = "runnable" → PkgType::Runnable
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "run_pkg",
    type = "runnable",
}
function M.run(ctx) end
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.pkg_type, Some(PkgType::Runnable));
    }

    #[test]
    fn auto_detect_type_from_m_run() {
        // Acceptance criterion 7: no M.meta.type, but M.run is present → Runnable
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "auto_run_pkg",
}
function M.run(ctx)
    return alc.llm("hello")
end
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(
            pkg.pkg_type,
            Some(PkgType::Runnable),
            "M.run present → Runnable"
        );
    }

    #[test]
    fn auto_detect_type_library_no_m_run() {
        // Acceptance criterion 8: no M.meta.type, no M.run → Library
        let tmp = tempfile::tempdir().unwrap();
        let path = write_init_lua(
            tmp.path(),
            r#"
local M = {}
M.meta = {
    name = "auto_lib_pkg",
    description = "A pure library with no run entry point",
}
function M.create(opts)
    return {}
end
return M
"#,
        );

        let pkg = PkgEntity::parse_from_init_lua(&path).expect("should parse");
        assert_eq!(pkg.pkg_type, Some(PkgType::Library), "no M.run → Library");
    }

    #[test]
    fn detect_has_run_ignores_comments() {
        // Acceptance criterion 9: M.run in a comment must not trigger detection.
        assert!(
            !detect_has_run("-- M.run = function(ctx) end\nlocal M = {}\n"),
            "commented-out M.run should not be detected"
        );
        // But a real M.run assignment on the same line after a comment is still
        // in the non-comment portion before '--', so it IS detected.
        assert!(
            detect_has_run("local M = {}\nM.run = function(ctx) end\n"),
            "real M.run should be detected"
        );
        // Edge case: M.run after inline comment on same line — not detected
        // because the effective portion before '--' does not contain M.run.
        assert!(
            !detect_has_run("local x = 1 -- M.run is described here\n"),
            "M.run inside inline comment should not be detected"
        );
    }

    #[test]
    fn serde_round_trip_with_pkg_type() {
        // Acceptance criterion 10: PkgType::Library survives a JSON round-trip.
        let pkg = PkgEntity {
            name: "lib".into(),
            version: Some("0.1.0".into()),
            description: None,
            category: None,
            docstring: None,
            tags: None,
            pkg_type: Some(PkgType::Library),
        };
        let json = serde_json::to_string(&pkg).unwrap();
        assert!(
            json.contains("\"type\":\"library\""),
            "wire key must be 'type': {json}"
        );
        let back: PkgEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pkg_type, Some(PkgType::Library));
        assert_eq!(back, pkg);
    }

    #[test]
    fn serde_deserialize_missing_pkg_type() {
        // Acceptance criterion 11: JSON without "type" key → pkg_type = None
        let json = r#"{"name":"legacy_pkg","version":"1.0.0"}"#;
        let pkg: PkgEntity = serde_json::from_str(json).unwrap();
        assert_eq!(pkg.name, "legacy_pkg");
        assert!(
            pkg.pkg_type.is_none(),
            "missing 'type' key must deserialize as None"
        );
    }
}
