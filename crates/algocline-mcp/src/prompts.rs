//! MCP Prompts capability — workflow-trigger prompts.
//!
//! algocline exposes a small, static set of Prompts that act as user-side
//! entry points into Tool-driven workflows. Each Prompt's `messages` body is
//! an instruction directed at the host LLM telling it to dispatch one or more
//! algocline MCP Tools (e.g. `alc_advice`, `alc_pkg_scaffold`) to complete
//! the workflow. See `docs/design/mcp-support.md` for the rationale behind
//! this scope (in particular, why per-package 1:1 mapping is not used).

use std::sync::Arc;

use algocline_app::EngineApi;
use algocline_core::AppDir;
use rmcp::{
    model::{GetPromptResult, Prompt, PromptArgument, PromptMessage, PromptMessageRole},
    ErrorData as McpError,
};

// ─── Static prompt definitions ───────────────────────────────────

/// Definition of a workflow-trigger Prompt: identity, arguments, and the
/// instruction template returned by `prompts/get`.
struct PromptDef {
    name: &'static str,
    title: &'static str,
    description: &'static str,
    arguments: &'static [PromptArgDef],
    template: &'static str,
}

/// Definition of a single Prompt argument. `required` mirrors the MCP spec
/// field; the host UI is expected to surface it as such.
struct PromptArgDef {
    name: &'static str,
    description: &'static str,
    required: bool,
}

const PROMPTS: &[PromptDef] = &[
    PromptDef {
        name: "advice",
        title: "Advice",
        description: "Pick an appropriate algocline strategy for a task and run it.",
        arguments: &[PromptArgDef {
            name: "task",
            description: "The task to apply an algocline strategy to.",
            required: true,
        }],
        template: "User task: ${task}\n\n\
Use the `alc_advice` MCP tool to choose the most appropriate installed \
algocline package for this task. State the chosen package and a one-sentence \
rationale, then invoke `alc_run` with that package and the task above. \
Summarise the result for the user.",
    },
    PromptDef {
        name: "new_package",
        title: "New Package",
        description: "Scaffold a new algocline package interactively.",
        arguments: &[
            PromptArgDef {
                name: "name",
                description: "Package name (snake_case, unique across installed packages).",
                required: true,
            },
            PromptArgDef {
                name: "category",
                description: "Optional category hint (e.g. aggregation, reasoning, eval).",
                required: false,
            },
        ],
        template: "User wants to scaffold a new algocline package named \
`${name}` (category hint: `${category}`).\n\n\
Drive `alc_pkg_scaffold` to create the initial package layout. Ask the user \
for any missing required fields (shape, source, dependencies) before \
finalising, and present the resulting `init.lua` summary for confirmation.",
    },
];

// ─── PromptCatalog ───────────────────────────────────────────────

/// Catalog of workflow-trigger MCP prompts.
///
/// The prompt list is static (see `PROMPTS`). `Arc<dyn EngineApi>` and
/// `Arc<AppDir>` are retained on the struct for parity with `ResourceCatalog`
/// and to support future workflows that may need to enumerate or read engine
/// state at `prompts/get` time.
pub struct PromptCatalog {
    #[allow(dead_code)]
    app: Arc<dyn EngineApi>,
    #[allow(dead_code)]
    app_dir: Arc<AppDir>,
}

impl PromptCatalog {
    /// Create a new `PromptCatalog`.
    ///
    /// # Arguments
    ///
    /// * `app` — engine API handle (retained for future workflow needs).
    /// * `app_dir` — application directory handle (same rationale).
    pub fn new(app: Arc<dyn EngineApi>, app_dir: Arc<AppDir>) -> Self {
        Self { app, app_dir }
    }

    /// Return the static workflow-trigger prompts.
    ///
    /// The list does not change during the lifetime of the server, so
    /// `notifications/prompts/list_changed` is not fired by this catalog.
    pub async fn list_prompts(&self) -> Result<Vec<Prompt>, McpError> {
        let prompts = PROMPTS
            .iter()
            .map(|def| {
                let args: Vec<PromptArgument> = def
                    .arguments
                    .iter()
                    .map(|a| {
                        PromptArgument::new(a.name)
                            .with_description(a.description)
                            .with_required(a.required)
                    })
                    .collect();
                Prompt::new(def.name, Some(def.description), Some(args)).with_title(def.title)
            })
            .collect();
        Ok(prompts)
    }

    /// Return the prompt messages for a named workflow trigger.
    ///
    /// Substitutes the caller-supplied arguments into the prompt's template at
    /// call time. Missing arguments are substituted as the empty string;
    /// validation of required arguments is deliberately deferred to the
    /// downstream Tool calls invoked by the host LLM.
    ///
    /// # Errors
    ///
    /// * `-32602` (invalid params) when `name` does not match a known prompt.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<GetPromptResult, McpError> {
        let def = PROMPTS
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| McpError::invalid_params(format!("unknown prompt: {name}"), None))?;

        let body = substitute(def.template, def.arguments, arguments);
        let message = PromptMessage::new_text(PromptMessageRole::User, body);
        Ok(GetPromptResult::new(vec![message]).with_description(def.description))
    }
}

/// Substitute `${arg_name}` placeholders in `template` using values from
/// `arguments`. Each declared argument is looked up by name; missing or
/// non-string values are replaced with the empty string.
fn substitute(
    template: &str,
    declared: &[PromptArgDef],
    arguments: Option<&serde_json::Map<String, serde_json::Value>>,
) -> String {
    let mut out = template.to_string();
    for arg in declared {
        let needle = format!("${{{}}}", arg.name);
        let value = arguments
            .and_then(|m| m.get(arg.name))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        out = out.replace(&needle, value);
    }
    out
}

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_constant_names_are_unique() {
        let mut names: Vec<&str> = PROMPTS.iter().map(|d| d.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len, "prompt names must be unique");
    }

    #[test]
    fn substitute_replaces_declared_arguments() {
        let decl = &[
            PromptArgDef {
                name: "task",
                description: "",
                required: true,
            },
            PromptArgDef {
                name: "category",
                description: "",
                required: false,
            },
        ];
        let mut args = serde_json::Map::new();
        args.insert(
            "task".to_string(),
            serde_json::Value::String("refactor".into()),
        );
        args.insert(
            "category".to_string(),
            serde_json::Value::String("reasoning".into()),
        );

        let out = substitute("t=${task} c=${category}", decl, Some(&args));
        assert_eq!(out, "t=refactor c=reasoning");
    }

    #[test]
    fn substitute_uses_empty_string_for_missing_arguments() {
        let decl = &[PromptArgDef {
            name: "task",
            description: "",
            required: true,
        }];
        let out = substitute("t=${task}", decl, None);
        assert_eq!(out, "t=");
    }

    #[test]
    fn substitute_ignores_undeclared_keys() {
        let decl = &[PromptArgDef {
            name: "task",
            description: "",
            required: true,
        }];
        let mut args = serde_json::Map::new();
        args.insert("task".to_string(), serde_json::Value::String("x".into()));
        args.insert(
            "extra".to_string(),
            serde_json::Value::String("ignored".into()),
        );
        let out = substitute("t=${task} e=${extra}", decl, Some(&args));
        // `extra` is not declared, so its placeholder is left intact.
        assert_eq!(out, "t=x e=${extra}");
    }
}
