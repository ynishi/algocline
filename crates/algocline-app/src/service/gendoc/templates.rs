//! Core-embedded default rules, descriptions, and repo_notes for
//! `alc_hub_gendoc` context7 / devin projections.
//!
//! These are compile-time constants (`&'static str` / `&'static [&'static str]`).
//! They are **not** executed as Lua — no `include_str!` or `@embedded:` prefix
//! applies here. The constants are merged into `HubProjectionConfig` by
//! `load_hub_projection_config` before being converted to `toml::Value` and
//! injected into the Lua preload registry via `inject_config_subtable`.

/// Fallback project name when no `name` field is supplied in `[hub]` or
/// `[hub.context7]` / `[hub.devin]` sections and the hub index basename
/// cannot be determined.
pub const DEFAULT_NAME_FALLBACK: &str = "algocline-hub";

/// Default description for context7 projections.
pub const DEFAULT_C7_DESCRIPTION: &str =
    "A collection of packages and tools for the algocline ecosystem.";

/// Default rules injected into context7 projections when neither
/// `rules_file` nor `rules_override` is specified in `[hub.context7]`.
/// Extra rules from `extra_rules` are appended after these defaults.
pub const DEFAULT_C7_RULES: &[&str] = &[
    "Follow the package documentation and examples closely.",
    "Prefer explicit configuration over implicit defaults.",
    "Check the hub index for the latest available packages before suggesting alternatives.",
    "When referencing package APIs, use the version documented in the hub index.",
    "Report any discrepancies between documentation and actual behaviour.",
];

/// Default repo notes injected into Devin wiki projections when neither
/// `repo_notes_file` nor `repo_notes_override` is specified in `[hub.devin]`.
/// Extra notes from `extra_repo_notes` are appended after these defaults.
///
/// These strings are stored as plain `&str` in-memory. When converted to
/// `toml::Value` via `HubProjectionConfig::to_devin_toml`, each entry is
/// wrapped into `{content = "<str>"}` inline-table form to satisfy the
/// `validate_note` contract in `projections.lua:664-670`.
pub const DEFAULT_DEVIN_REPO_NOTES: &[&str] = &[
    "This repository is managed with algocline. Use `alc` commands for package operations.",
    "Prefer `alc_hub_gendoc` to regenerate documentation after significant changes.",
    "Package metadata is defined in `alc.toml`. Consult it before modifying package structure.",
];
