//! Session specification types for the `ExecutionService` layer.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::path::PathBuf;

/// Complete specification required to spawn a new execution session.
///
/// `SessionSpec` is a pure value type passed to [`crate::execution::ExecutionService::spawn`].
/// It carries no runtime handles and is safe to clone and serialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSpec {
    /// The kind of execution to perform.
    pub kind: SpecKind,
    /// Optional project root directory.  When `None`, the engine chooses a default.
    pub project_root: Option<PathBuf>,
    /// Arbitrary JSON context forwarded to the execution environment.
    pub ctx: Option<JsonValue>,
}

/// Discriminant for the execution kind carried by [`SessionSpec`].
///
/// Each variant corresponds to one of the primary operations the engine can perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecKind {
    /// Execute a Lua code snippet directly.
    Run {
        /// Lua source code to execute.
        code: String,
    },
    /// Apply an advice package strategy to an optional task.
    Advice {
        /// Name of the strategy package to apply.
        strategy: String,
        /// Optional task description passed as context.
        task: Option<String>,
        /// Additional JSON options for the strategy.
        opts: Option<JsonValue>,
    },
    /// Evaluate a scenario against a strategy.
    Eval {
        /// Reference to the scenario to evaluate.
        scenario: ScenarioRef,
        /// Name of the strategy to evaluate.
        strategy: String,
        /// Optional JSON options for the evaluation.
        opts: Option<JsonValue>,
        /// Whether to automatically create a card from the evaluation result.
        auto_card: bool,
    },
}

/// Reference to a scenario used in [`SpecKind::Eval`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "ref_kind", rename_all = "snake_case")]
pub enum ScenarioRef {
    /// Inline scenario content as a string.
    Inline(String),
    /// Name of an installed scenario (looked up from `~/.algocline/scenarios/`).
    Name(String),
    /// Path to a scenario file on the local file system.
    File(PathBuf),
}
