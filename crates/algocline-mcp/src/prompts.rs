//! MCP Prompts capability — dynamic package enumeration.
//!
//! Each installed package is exposed 1:1 as a named MCP prompt.
//! `list_prompts` enumerates packages by calling `EngineApi::pkg_list` on every
//! request (no compile-time or startup-time static list).
//! `get_prompt` substitutes the caller-supplied `task` argument into the message
//! text at runtime before returning the single-message result.

use std::sync::Arc;

use algocline_app::EngineApi;
use algocline_core::AppDir;
use rmcp::{
    model::{GetPromptResult, Prompt, PromptArgument, PromptMessage, PromptMessageRole},
    ErrorData as McpError,
};
use tracing::warn;

// ─── Title-case helper ───────────────────────────────────────────

/// Convert the first character of `s` to uppercase and keep the rest unchanged.
///
/// # Arguments
///
/// * `s` — the input string (typically a package name such as `"cot"`)
///
/// # Returns
///
/// A new `String` with the first character uppercased. Returns an empty string
/// when `s` is empty.
fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

// ─── PromptCatalog ───────────────────────────────────────────────

/// Catalog of MCP prompts derived from installed packages.
///
/// Each installed package is exposed as one prompt with name equal to the
/// package name, a title derived by uppercasing the first character, and a
/// single optional `task` argument.
///
/// Mirroring `ResourceCatalog`, this struct holds an `Arc<dyn EngineApi>` for
/// runtime package enumeration and an `Arc<AppDir>` reserved for future
/// extensions (e.g. `notifications/prompts/list_changed` in Phase 1.x).
pub struct PromptCatalog {
    app: Arc<dyn EngineApi>,
    #[allow(dead_code)]
    app_dir: Arc<AppDir>,
}

impl PromptCatalog {
    /// Create a new `PromptCatalog`.
    ///
    /// # Arguments
    ///
    /// * `app` — engine API used to enumerate installed packages on each request
    /// * `app_dir` — reserved for Phase 1.x list-changed notification support
    pub fn new(app: Arc<dyn EngineApi>, app_dir: Arc<AppDir>) -> Self {
        Self { app, app_dir }
    }

    /// Enumerate all installed packages and return them as MCP `Prompt` objects.
    ///
    /// Calls `EngineApi::pkg_list` on every invocation so that the list always
    /// reflects the current state of `alc.toml` + `~/.algocline/packages/`.
    /// No compile-time or startup-time caching is performed (Crux #1 constraint).
    ///
    /// # Returns
    ///
    /// `Ok(Vec<Prompt>)` — one entry per installed package, ordered as returned
    /// by `pkg_list` (newest-first by default).
    ///
    /// # Errors
    ///
    /// Returns `Err(McpError)` with code `-32603` (internal error) when
    /// `pkg_list` fails or the JSON response cannot be parsed.
    pub async fn list_prompts(&self) -> Result<Vec<Prompt>, McpError> {
        let json_str = self
            .app
            .pkg_list(None, Some(0), None, None, None, None)
            .await
            .map_err(|e| McpError::internal_error(format!("pkg enumeration failed: {e}"), None))?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
            McpError::internal_error(format!("pkg enumeration failed: invalid JSON: {e}"), None)
        })?;

        let packages = parsed
            .get("packages")
            .and_then(|p| p.as_array())
            .ok_or_else(|| {
                McpError::internal_error(
                    "pkg enumeration failed: missing `packages` array".to_string(),
                    None,
                )
            })?;

        let task_arg = PromptArgument::new("task")
            .with_description("Task to apply this strategy to")
            .with_required(false);

        let prompts = packages
            .iter()
            .filter_map(|entry| {
                let name = match entry.get("name").and_then(|n| n.as_str()) {
                    Some(n) => n,
                    None => {
                        warn!("pkg_list entry missing `name` field, skipping");
                        return None;
                    }
                };
                let description = entry
                    .get("description")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string());
                let title = title_case(name);
                Some(
                    Prompt::new(name, description.as_deref(), Some(vec![task_arg.clone()]))
                        .with_title(title),
                )
            })
            .collect();

        Ok(prompts)
    }

    /// Return the prompt message for a named package, substituting `task` at runtime.
    ///
    /// Enumerates packages via `pkg_list` on every call to verify the name is
    /// valid and to fetch the current description (Crux #1 constraint — no cache).
    /// The `task` argument value is substituted into the message template at call
    /// time (Crux #2 constraint — no pre-baked static text).
    ///
    /// # Arguments
    ///
    /// * `name` — the prompt name (must equal an installed package name)
    /// * `arguments` — optional map containing the `task` key; if absent or if
    ///   `task` is not present, an empty string is used (template still valid)
    ///
    /// # Returns
    ///
    /// `Ok(GetPromptResult)` with one `User`-role text message containing the
    /// interpolated template.
    ///
    /// # Errors
    ///
    /// * `-32602` (invalid params) when `name` does not match any installed package
    /// * `-32603` (internal error) when `pkg_list` fails or the JSON is malformed
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<GetPromptResult, McpError> {
        let json_str = self
            .app
            .pkg_list(None, Some(0), None, None, None, None)
            .await
            .map_err(|e| McpError::internal_error(format!("pkg enumeration failed: {e}"), None))?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
            McpError::internal_error(format!("pkg enumeration failed: invalid JSON: {e}"), None)
        })?;

        let packages = parsed
            .get("packages")
            .and_then(|p| p.as_array())
            .ok_or_else(|| {
                McpError::internal_error(
                    "pkg enumeration failed: missing `packages` array".to_string(),
                    None,
                )
            })?;

        // Find the matching package entry to confirm it exists and get description.
        let pkg_entry = packages
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some(name))
            .ok_or_else(|| McpError::invalid_params(format!("unknown prompt: {name}"), None))?;

        let description = pkg_entry
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());

        // Extract `task` argument; absent or null → empty string (Crux #2).
        let task_value = arguments
            .and_then(|m| m.get("task"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Build the message text and substitute ${task} at runtime (Crux #2).
        let template = format!(
            "{name} の戦略を以下の task に適用して。task が空なら一般的な使い方を示して。\n\ntask: ${{task}}"
        );
        let body = template.replace("${task}", task_value);

        let message = PromptMessage::new_text(PromptMessageRole::User, body);

        let result = if let Some(desc) = description {
            GetPromptResult::new(vec![message]).with_description(desc)
        } else {
            GetPromptResult::new(vec![message])
        };

        Ok(result)
    }
}

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // T1: title_case on typical package names
    #[test]
    fn title_case_lowercases_first_char_upper() {
        assert_eq!(title_case("cot"), "Cot");
        assert_eq!(title_case("sc"), "Sc");
        assert_eq!(title_case("ucb"), "Ucb");
    }

    // T2: title_case edge cases — empty / single char / already-upper
    #[test]
    fn title_case_edge_cases() {
        assert_eq!(title_case(""), "");
        assert_eq!(title_case("x"), "X");
        assert_eq!(title_case("A"), "A");
        assert_eq!(title_case("ABC"), "ABC");
    }

    // T3: title_case preserves multi-char tail unchanged
    #[test]
    fn title_case_preserves_tail() {
        assert_eq!(
            title_case("review_and_investigate"),
            "Review_and_investigate"
        );
    }
}
