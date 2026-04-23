//! `alc_pkg_scaffold` — generate a minimal package skeleton.
//!
//! Writes a single `init.lua` file to `<target_dir>/<name>/init.lua` with
//! `M.meta` / `M.spec.entries.run` / `M.run` template.  The
//! `alc_shapes_compat` range is derived automatically from
//! [`EMBEDDED_ALC_SHAPES_VERSION`].
//!
//! Per the project-level Error propagation rule
//! (`CLAUDE.md §Service 層 Error 伝播規律`), every error variant is
//! propagated via `?` to the MCP wire layer — no `warn!` drops, no
//! `unwrap_or_default`, no silent `Err(_) =>` branches.

use std::path::PathBuf;

use semver::Version;
use thiserror::Error;

use super::gendoc::EMBEDDED_ALC_SHAPES_VERSION;
use super::AppService;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum PkgScaffoldError {
    #[error("invalid package name {name:?}: {reason}")]
    NameInvalid { name: String, reason: &'static str },

    #[error("package skeleton already exists at {}", path.display())]
    AlreadyExists { path: PathBuf },

    #[error("I/O error at {}: {cause}", path.display())]
    IoError { path: PathBuf, cause: String },
}

// ── Result type ──────────────────────────────────────────────────────────────

/// Successful scaffold result.
#[derive(Debug)]
pub struct ScaffoldResult {
    pub path: PathBuf,
    pub bytes_written: usize,
}

// ── Template ─────────────────────────────────────────────────────────────────

/// Base template — markers `{{NAME}}`, `{{COMPAT}}`, `{{HEADER_LINE}}`,
/// `{{CATEGORY_LINE}}`, `{{DESCRIPTION_LINE}}` are substituted at render time.
const TEMPLATE: &str = r#"--- {{NAME}} — {{HEADER_LINE}}.

local S = require("alc_shapes")
local T = S.T

local M = {
    meta = {
        name = "{{NAME}}",
        version = "0.1.0",
        alc_shapes_compat = "{{COMPAT}}",
{{CATEGORY_LINE}}{{DESCRIPTION_LINE}}    },
    spec = {
        entries = {
            run = {
                -- TODO: declare input / result via alc_shapes.t combinators.
                -- input  = T.shape({ ... }),
                -- result = T.shape({ ... }),
            },
        },
    },
}

function M.run(ctx)
    -- TODO: implement. Use alc.llm(prompt) for LLM calls
    -- (pauses execution; host resumes via alc_continue).
    local answer = alc.llm("example prompt for " .. tostring(ctx.task))
    return { answer = answer }
end

return M
"#;

// ── Name validation ───────────────────────────────────────────────────────────

/// Validate a package name.
///
/// Rules (hand-rolled, no regex):
/// - Non-empty, length ≤ 64.
/// - First character: `a-z`.
/// - Remaining characters: `a-z`, `0-9`, `_`.
fn validate_name(name: &str) -> Result<(), PkgScaffoldError> {
    if name.is_empty() {
        return Err(PkgScaffoldError::NameInvalid {
            name: name.to_string(),
            reason: "name must not be empty",
        });
    }
    if name.len() > 64 {
        return Err(PkgScaffoldError::NameInvalid {
            name: name.to_string(),
            reason: "name must be 64 characters or fewer",
        });
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_lowercase() {
        return Err(PkgScaffoldError::NameInvalid {
            name: name.to_string(),
            reason: "name must start with a lowercase ASCII letter (a-z)",
        });
    }
    for ch in chars {
        if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '_' {
            return Err(PkgScaffoldError::NameInvalid {
                name: name.to_string(),
                reason: "name may only contain lowercase ASCII letters, digits, and underscores",
            });
        }
    }
    Ok(())
}

// ── Default compat range ─────────────────────────────────────────────────────

/// Compute the default `alc_shapes_compat` range from [`EMBEDDED_ALC_SHAPES_VERSION`].
///
/// `"0.25.1"` → `">=0.25.0, <0.26"`.
/// `"1.2.3"`  → `">=1.2.0, <1.3"`.
///
/// On parse failure (should never happen — the constant is well-formed) falls
/// back to a literal and emits `tracing::warn!`.
fn default_compat_range() -> String {
    match Version::parse(EMBEDDED_ALC_SHAPES_VERSION) {
        Ok(v) => {
            let major = v.major;
            let minor = v.minor;
            format!(">={major}.{minor}.0, <{major}.{}", minor + 1)
        }
        Err(e) => {
            tracing::warn!(
                embedded = EMBEDDED_ALC_SHAPES_VERSION,
                error = %e,
                "pkg_scaffold: failed to parse EMBEDDED_ALC_SHAPES_VERSION; \
                 falling back to hardcoded compat range"
            );
            ">=0.25.0, <0.26".to_string()
        }
    }
}

// ── Template rendering ────────────────────────────────────────────────────────

/// Escape a Rust `&str` for safe embedding inside a Lua double-quoted
/// string literal.
///
/// Without this, a `category` / `description` value containing a bare
/// `"` or `\n` would break out of the string literal in the generated
/// `init.lua` and, in the worst case, let a caller inject arbitrary
/// Lua code. We escape `\`, `"`, `\n`, `\r`, and `\0` — the minimal
/// set required by Lua's lexer.
///
/// The `header_line` substitution lands in a `---` comment rather than
/// a string literal, but may still contain newlines; callers should
/// collapse newlines before passing a description used as a header.
fn escape_lua_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            c => out.push(c),
        }
    }
    out
}

/// Collapse any CR/LF in a header substitution (lands in a `---` comment).
/// Comment lines terminate at `\n`, so an unescaped newline in the header
/// would silently break the doc comment structure.
fn sanitize_header_line(s: &str) -> String {
    s.replace(['\r', '\n'], " ")
}

fn render_template(
    name: &str,
    compat: &str,
    category: Option<&str>,
    description: Option<&str>,
) -> String {
    let header_line = match description {
        Some(d) => sanitize_header_line(d),
        None => "TODO: one-line description".to_string(),
    };

    // Category line: uncommented when provided, commented-out placeholder otherwise.
    let category_line = match category {
        Some(cat) => format!("        category = \"{}\",\n", escape_lua_string(cat)),
        None => {
            "        -- category = \"<category>\",       -- uncomment if provided\n".to_string()
        }
    };

    // Description line: uncommented when provided, commented-out placeholder otherwise.
    let description_line = match description {
        Some(desc) => format!("        description = \"{}\",\n", escape_lua_string(desc)),
        None => {
            "        -- description = \"<description>\", -- uncomment if provided\n".to_string()
        }
    };

    // `name` and `compat` are validated separately (name: strict charset;
    // compat: produced internally from EMBEDDED_ALC_SHAPES_VERSION), so
    // they cannot contain characters requiring escaping. Still escape
    // defensively to keep the template rendering layer self-contained.
    TEMPLATE
        .replace("{{NAME}}", &escape_lua_string(name))
        .replace("{{COMPAT}}", &escape_lua_string(compat))
        .replace("{{HEADER_LINE}}", &header_line)
        .replace("{{CATEGORY_LINE}}", &category_line)
        .replace("{{DESCRIPTION_LINE}}", &description_line)
}

// ── Core function ─────────────────────────────────────────────────────────────

/// Generate a minimal package skeleton at `<target_dir>/<name>/init.lua`.
///
/// Errors:
/// - [`PkgScaffoldError::NameInvalid`] — name fails validation.
/// - [`PkgScaffoldError::AlreadyExists`] — `init.lua` already present.
/// - [`PkgScaffoldError::IoError`] — filesystem operation failed.
pub fn scaffold_pkg(
    name: &str,
    target_dir: &str,
    category: Option<&str>,
    description: Option<&str>,
) -> Result<ScaffoldResult, PkgScaffoldError> {
    validate_name(name)?;

    let pkg_dir = std::path::Path::new(target_dir).join(name);
    let init_lua = pkg_dir.join("init.lua");

    if init_lua.exists() {
        return Err(PkgScaffoldError::AlreadyExists { path: init_lua });
    }

    std::fs::create_dir_all(&pkg_dir).map_err(|e| PkgScaffoldError::IoError {
        path: pkg_dir.clone(),
        cause: e.to_string(),
    })?;

    let compat = default_compat_range();
    let content = render_template(name, &compat, category, description);
    let bytes_written = content.len();

    std::fs::write(&init_lua, &content).map_err(|e| PkgScaffoldError::IoError {
        path: init_lua.clone(),
        cause: e.to_string(),
    })?;

    Ok(ScaffoldResult {
        path: init_lua,
        bytes_written,
    })
}

// ── AppService method ─────────────────────────────────────────────────────────

impl AppService {
    /// Generate a minimal package skeleton at `<target_dir>/<name>/init.lua`.
    ///
    /// Returns a JSON string `{ "status": "ok", "path": "...", "bytes_written": N }`.
    /// Typed errors are forwarded as `Err(String)` to the MCP wire layer.
    pub fn pkg_scaffold(
        &self,
        name: &str,
        target_dir: Option<&str>,
        category: Option<&str>,
        description: Option<&str>,
    ) -> Result<String, String> {
        let dir = target_dir.unwrap_or(".");

        let result = scaffold_pkg(name, dir, category, description).map_err(|e| e.to_string())?;

        serde_json::to_string(&serde_json::json!({
            "status": "ok",
            "path": result.path.to_string_lossy(),
            "bytes_written": result.bytes_written,
        }))
        .map_err(|e| format!("pkg_scaffold: JSON serialization error: {e}"))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_name ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_name_ok() {
        assert!(validate_name("my_pkg").is_ok());
        assert!(validate_name("a").is_ok());
        assert!(validate_name("pkg123").is_ok());
        assert!(validate_name("a_b_c").is_ok());
        // exactly 64 chars
        let long = "a".repeat(64);
        assert!(validate_name(&long).is_ok());
    }

    #[test]
    fn test_validate_name_empty() {
        let err = validate_name("").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
        assert!(err.to_string().contains("not be empty"));
    }

    #[test]
    fn test_validate_name_too_long() {
        let name = "a".repeat(65);
        let err = validate_name(&name).unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
        assert!(err.to_string().contains("64 characters"));
    }

    #[test]
    fn test_validate_name_starts_with_digit() {
        let err = validate_name("1bad").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
        assert!(err.to_string().contains("start with a lowercase"));
    }

    #[test]
    fn test_validate_name_starts_with_upper() {
        let err = validate_name("Bad").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
    }

    #[test]
    fn test_validate_name_contains_slash() {
        let err = validate_name("has/slash").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
        assert!(err.to_string().contains("only contain"));
    }

    #[test]
    fn test_validate_name_contains_hyphen() {
        let err = validate_name("with-hyphen").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
    }

    #[test]
    fn test_validate_name_uppercase_mid() {
        let err = validate_name("myPkg").unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
    }

    // ── default_compat_range ─────────────────────────────────────────────────

    #[test]
    fn test_default_compat_range_format() {
        let range = default_compat_range();
        // Must contain the expected format for current version.
        assert!(
            range.starts_with(">="),
            "expected range to start with '>=' got: {range}"
        );
        assert!(
            range.contains(", <"),
            "expected range to contain ', <' got: {range}"
        );
    }

    #[test]
    fn test_default_compat_range_current_version() {
        // "0.25.1" → ">=0.25.0, <0.26"
        let range = default_compat_range();
        assert_eq!(range, ">=0.25.0, <0.26");
    }

    // ── escape_lua_string ────────────────────────────────────────────────────

    #[test]
    fn test_escape_lua_string_passes_through_plain() {
        assert_eq!(escape_lua_string("plain text"), "plain text");
    }

    #[test]
    fn test_escape_lua_string_escapes_quote_and_backslash() {
        assert_eq!(
            escape_lua_string(r#"he said "hi" \n"#),
            r#"he said \"hi\" \\n"#
        );
    }

    #[test]
    fn test_escape_lua_string_escapes_newline_cr_nul() {
        assert_eq!(escape_lua_string("a\nb\rc\0d"), "a\\nb\\rc\\0d");
    }

    #[test]
    fn test_render_template_escapes_injection_payload() {
        // Attempted breakout via closing quote + injected Lua code.
        let payload = r#"x",injected=os.execute("rm -rf /"),y=""#;
        let out = render_template("my_pkg", ">=0.25.0, <0.26", Some(payload), Some(payload));
        // Every `"` in the payload must be emitted as `\"` in the
        // rendered Lua string literal. The payload contains 4 quotes;
        // they must all be preceded by a backslash.
        let expected_escaped = r#"x\",injected=os.execute(\"rm -rf /\"),y=\""#;
        assert!(
            out.contains(expected_escaped),
            "payload must be fully escaped; render was:\n{out}"
        );
        // The `category` / `description` Lua string literals must
        // contain the escaped form only. Scan the rendered lines that
        // begin the field declaration.
        for line in out.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("category = \"") || trimmed.starts_with("description = \"") {
                assert!(
                    line.contains(expected_escaped),
                    "field line must contain escaped payload: {line}"
                );
                assert!(
                    !line.contains(payload),
                    "field line must not contain raw payload: {line}"
                );
            }
        }
    }

    // ── render_template ──────────────────────────────────────────────────────

    #[test]
    fn test_render_template_basic() {
        let out = render_template("my_pkg", ">=0.25.0, <0.26", None, None);
        assert!(out.contains(r#"name = "my_pkg""#));
        assert!(out.contains(r#"alc_shapes_compat = ">=0.25.0, <0.26""#));
        assert!(out.contains("-- category = \"<category>\","));
        assert!(out.contains("-- description = \"<description>\","));
        assert!(out.contains("TODO: one-line description"));
        assert!(out.contains("function M.run(ctx)"));
        assert!(out.contains("T.shape"));
        assert!(out.contains("return M"));
    }

    #[test]
    fn test_render_template_with_category_and_description() {
        let out = render_template(
            "my_pkg",
            ">=0.25.0, <0.26",
            Some("selection"),
            Some("test pkg"),
        );
        assert!(out.contains(r#"category = "selection""#));
        assert!(out.contains(r#"description = "test pkg""#));
        // Commented-out placeholders must NOT appear.
        assert!(!out.contains("-- category ="));
        assert!(!out.contains("-- description ="));
        // Header line uses description value.
        assert!(out.contains("test pkg"));
    }

    #[test]
    fn test_render_template_with_category_only() {
        let out = render_template("my_pkg", ">=0.25.0, <0.26", Some("reasoning"), None);
        assert!(out.contains(r#"category = "reasoning""#));
        // description placeholder is commented out.
        assert!(out.contains("-- description ="));
    }

    // ── scaffold_pkg ─────────────────────────────────────────────────────────

    #[test]
    fn test_scaffold_pkg_creates_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let result =
            scaffold_pkg("my_pkg", tmp.path().to_str().unwrap(), None, None).expect("scaffold ok");

        let expected_path = tmp.path().join("my_pkg").join("init.lua");
        assert_eq!(result.path, expected_path);
        assert!(expected_path.exists(), "init.lua must exist");

        let content = std::fs::read_to_string(&expected_path).expect("read init.lua");
        assert!(content.contains(r#"name = "my_pkg""#));
        assert!(content.contains("alc_shapes_compat"));
        assert!(result.bytes_written > 0);
        assert_eq!(result.bytes_written, content.len());
    }

    #[test]
    fn test_scaffold_pkg_already_exists() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let pkg_dir = tmp.path().join("my_pkg");
        std::fs::create_dir_all(&pkg_dir).expect("create dir");
        std::fs::write(pkg_dir.join("init.lua"), "-- existing").expect("write existing");

        let err = scaffold_pkg("my_pkg", tmp.path().to_str().unwrap(), None, None).unwrap_err();
        assert!(matches!(err, PkgScaffoldError::AlreadyExists { .. }));
    }

    #[test]
    fn test_scaffold_pkg_invalid_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = scaffold_pkg("1bad", tmp.path().to_str().unwrap(), None, None).unwrap_err();
        assert!(matches!(err, PkgScaffoldError::NameInvalid { .. }));
    }

    #[test]
    fn test_scaffold_pkg_with_category_and_description() {
        let tmp = tempfile::tempdir().expect("tempdir");
        scaffold_pkg(
            "my_pkg",
            tmp.path().to_str().unwrap(),
            Some("selection"),
            Some("test pkg"),
        )
        .expect("scaffold ok");

        let content = std::fs::read_to_string(tmp.path().join("my_pkg").join("init.lua")).unwrap();
        assert!(content.contains(r#"category = "selection""#));
        assert!(content.contains(r#"description = "test pkg""#));
        assert!(!content.contains("-- category ="));
        assert!(!content.contains("-- description ="));
    }
}
