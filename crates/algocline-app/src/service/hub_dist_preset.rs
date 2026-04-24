//! `alc_hub_dist` preset expansion (`Current` builtin recipes + optional
//! `alc.toml` overrides).
//!
//! Design goals:
//! - Keep `hub_gendoc` a primitive — presets are expanded only in `hub_dist`.
//! - `preset_catalog_version` is a human-oriented revision marker for the
//!   builtin recipe dictionary (not semver for individual presets).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::json;

use super::gendoc::templates;
use super::project::resolve_project_root;

/// Revision marker for the builtin preset dictionary bundled with this
/// binary. Bump when builtin recipes change (even if `CARGO_PKG_VERSION`
/// does not).
pub const PRESET_CATALOG_VERSION: &str = "preset-catalog@2026-04-23";

#[derive(Debug, Clone)]
pub struct HubDistPresetResolution {
    pub projections: Option<Vec<String>>,
    pub config_path: Option<String>,
    pub lint_strict: Option<bool>,
    pub catalog_version: String,
    pub preset_name: Option<String>,
    pub overrides_source: Vec<String>,
    pub resolved_project_root: Option<PathBuf>,
}

// ── TOML deserialization structs ──────────────────────────────────────

#[derive(Debug, Deserialize)]
struct HubDistToml {
    hub: Option<HubSection>,
}

/// Top-level `[hub]` section in `alc.toml`.
///
/// All fields use `#[serde(default)]` so that existing consumers that only
/// define `[hub.dist]` continue to deserialize without error when the new
/// `context7` / `devin` / `name` / `description` fields are absent.
#[derive(Debug, Deserialize, Default)]
struct HubSection {
    dist: Option<HubDistSection>,
    /// Shared project name propagated to context7/devin when their own
    /// `name` field is absent.
    #[serde(default)]
    name: Option<String>,
    /// Shared project description propagated to context7/devin when their
    /// own `description` field is absent.
    #[serde(default)]
    description: Option<String>,
    /// Context7 projection configuration (`[hub.context7]`).
    #[serde(default)]
    context7: Option<HubContext7Config>,
    /// Devin wiki projection configuration (`[hub.devin]`).
    #[serde(default)]
    devin: Option<HubDevinConfig>,
}

#[derive(Debug, Deserialize)]
struct HubDistSection {
    preset_catalog_version: Option<String>,
    presets: Option<BTreeMap<String, HubDistPresetOverride>>,
}

#[derive(Debug, Deserialize)]
struct HubDistPresetOverride {
    projections: Option<Vec<String>>,
    config_path: Option<String>,
    lint_strict: Option<bool>,
}

/// `[hub.context7]` TOML section — context7 projection overrides.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct HubContext7Config {
    /// Override project name for the context7 projection.
    #[serde(default)]
    pub name: Option<String>,
    /// Override project description for the context7 projection.
    #[serde(default)]
    pub description: Option<String>,
    /// Fully replace the default rules with this list (mutually exclusive
    /// with `rules_file`).
    #[serde(default)]
    pub rules_override: Option<Vec<String>>,
    /// Path to a file whose lines become the rules list (one rule per line,
    /// blank lines and lines starting with `#` are ignored).  Mutually
    /// exclusive with `rules_override`.
    #[serde(default)]
    pub rules_file: Option<String>,
    /// Rules appended after the default (or overridden) list.
    #[serde(default)]
    pub extra_rules: Option<Vec<String>>,
}

/// `[hub.devin]` TOML section — Devin wiki projection overrides.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct HubDevinConfig {
    /// Override project name for the Devin wiki projection.
    #[serde(default)]
    pub name: Option<String>,
    /// Override project description for the Devin wiki projection.
    #[serde(default)]
    pub description: Option<String>,
    /// Fully replace the default repo notes with this list (mutually
    /// exclusive with `repo_notes_file`).
    #[serde(default)]
    pub repo_notes_override: Option<Vec<String>>,
    /// Path to a file whose lines become repo notes (one note per line,
    /// blank lines and lines starting with `#` are ignored).  Mutually
    /// exclusive with `repo_notes_override`.
    #[serde(default)]
    pub repo_notes_file: Option<String>,
    /// Repo notes appended after the default (or overridden) list.
    #[serde(default)]
    pub extra_repo_notes: Option<Vec<String>>,
}

// ── Resolved output types ─────────────────────────────────────────────

/// Resolved, merged configuration for a single context7 projection.
#[derive(Debug, Clone)]
pub struct ResolvedContext7 {
    /// Project name to surface in the context7 output.
    pub name: String,
    /// Project description to surface in the context7 output.
    pub description: String,
    /// Merged rules list (default + extras, or overridden/file-sourced).
    pub rules: Vec<String>,
}

/// Resolved, merged configuration for a single Devin wiki projection.
#[derive(Debug, Clone)]
pub struct ResolvedDevin {
    /// Project name to surface in the Devin wiki output.
    pub name: String,
    /// Project description to surface in the Devin wiki output.
    pub description: String,
    /// Merged repo notes list stored as plain strings in memory.
    ///
    /// **Important**: when converting to `toml::Value` via
    /// [`ResolvedDevin::to_devin_toml`], each entry is wrapped into a
    /// `{content = "<str>"}` inline table to satisfy the `validate_note`
    /// contract in `projections.lua:664-670`.  Passing plain strings
    /// produces a Lua runtime error at projection time.
    pub repo_notes: Vec<String>,
}

/// Combined resolved configuration passed to `inject_config_subtable` for
/// both the context7 and Devin projections.
#[derive(Debug, Clone)]
pub struct HubProjectionConfig {
    pub context7: ResolvedContext7,
    pub devin: ResolvedDevin,
}

impl HubProjectionConfig {
    /// Convert the context7 configuration into a `toml::Value::Table`
    /// suitable for passing to `inject_config_subtable` as the
    /// `tools.docs.context7_config` preload.
    ///
    /// Shape: `{ projectTitle = "...", description = "...", rules = ["...", ...] }`
    pub fn to_context7_toml(&self) -> toml::Value {
        let mut map = toml::value::Table::new();
        map.insert(
            "projectTitle".to_string(),
            toml::Value::String(self.context7.name.clone()),
        );
        map.insert(
            "description".to_string(),
            toml::Value::String(self.context7.description.clone()),
        );
        let rules: Vec<toml::Value> = self
            .context7
            .rules
            .iter()
            .map(|r| toml::Value::String(r.clone()))
            .collect();
        map.insert("rules".to_string(), toml::Value::Array(rules));
        toml::Value::Table(map)
    }

    /// Convert the Devin configuration into a `toml::Value::Table`
    /// suitable for passing to `inject_config_subtable` as the
    /// `tools.docs.devin_wiki_config` preload.
    ///
    /// Each `repo_notes` string is wrapped into a `{content = "<str>"}` inline
    /// table to satisfy `projections.lua:664-670` `validate_note`, which
    /// requires `type(note) == "table"` with a `note.content` string field.
    /// Passing plain strings produces a Lua runtime error at projection time.
    ///
    /// Shape: `{ project_name = "...", description = "...", repo_notes = [{content = "..."}, ...] }`
    pub fn to_devin_toml(&self) -> toml::Value {
        let mut map = toml::value::Table::new();
        map.insert(
            "project_name".to_string(),
            toml::Value::String(self.devin.name.clone()),
        );
        map.insert(
            "description".to_string(),
            toml::Value::String(self.devin.description.clone()),
        );
        // Wrap each plain string into {content = "<str>"} inline table.
        let repo_notes: Vec<toml::Value> = self
            .devin
            .repo_notes
            .iter()
            .map(|s| {
                let mut t = toml::value::Table::new();
                t.insert("content".to_string(), toml::Value::String(s.clone()));
                toml::Value::Table(t)
            })
            .collect();
        map.insert("repo_notes".to_string(), toml::Value::Array(repo_notes));
        toml::Value::Table(map)
    }
}

// ── Helper: load + resolve hub projection config ──────────────────────

/// Load `alc.toml` from `project_root` (if available) and resolve the
/// `[hub.context7]` / `[hub.devin]` sections into a [`HubProjectionConfig`]
/// using a three-stage precedence chain:
///
/// **name / description**:
/// `[hub.context7].name` / `.description` > `[hub].name` / `.description` > core default
///
/// **rules** (context7):
/// `rules_file` (full replacement) > `rules_override` (full replacement) >
/// `core_default_rules ++ extra_rules`
///
/// **repo_notes** (devin):
/// `repo_notes_file` (full replacement) > `repo_notes_override` (full replacement) >
/// `core_default_repo_notes ++ extra_repo_notes`
///
/// `rules_file` and `rules_override` are mutually exclusive; likewise for
/// `repo_notes_file` and `repo_notes_override`.  Both violations return a
/// typed `Err` that propagates to the MCP wire layer.
///
/// When `project_root` is `None` or `alc.toml` is absent, the function
/// returns a config built from core defaults only (not an error).
pub fn load_hub_projection_config(
    project_root: Option<&Path>,
) -> Result<HubProjectionConfig, String> {
    // 1. Try to read alc.toml; absence is legitimate (defaults apply).
    let hub_section: Option<HubSection> = if let Some(root) = project_root {
        let alc_path = root.join("alc.toml");
        if alc_path.is_file() {
            let raw = std::fs::read_to_string(&alc_path)
                .map_err(|e| format!("gendoc: failed to read {}: {e}", alc_path.display()))?;
            let parsed: HubDistToml =
                toml::from_str(&raw).map_err(|e| format!("gendoc: alc.toml parse failed: {e}"))?;
            parsed.hub
        } else {
            None
        }
    } else {
        None
    };

    let hub = hub_section.as_ref();
    let shared_name = hub.and_then(|h| h.name.as_deref());
    let shared_description = hub.and_then(|h| h.description.as_deref());
    let c7_cfg = hub.and_then(|h| h.context7.as_ref());
    let dv_cfg = hub.and_then(|h| h.devin.as_ref());

    // 2. Validate mutually exclusive fields.
    if let Some(c7) = c7_cfg {
        if c7.rules_file.is_some() && c7.rules_override.is_some() {
            return Err("gendoc: rules_file and rules_override are mutually exclusive".to_string());
        }
    }
    if let Some(dv) = dv_cfg {
        if dv.repo_notes_file.is_some() && dv.repo_notes_override.is_some() {
            return Err(
                "gendoc: repo_notes_file and repo_notes_override are mutually exclusive"
                    .to_string(),
            );
        }
    }

    // 3. Resolve context7.
    let c7_name = c7_cfg
        .and_then(|c| c.name.as_deref())
        .or(shared_name)
        .unwrap_or(templates::DEFAULT_NAME_FALLBACK)
        .to_string();

    let c7_description = c7_cfg
        .and_then(|c| c.description.as_deref())
        .or(shared_description)
        .unwrap_or(templates::DEFAULT_C7_DESCRIPTION)
        .to_string();

    let c7_rules = resolve_rules(
        c7_cfg.and_then(|c| c.rules_file.as_deref()),
        c7_cfg.and_then(|c| c.rules_override.as_deref()),
        c7_cfg.and_then(|c| c.extra_rules.as_deref()),
        templates::DEFAULT_C7_RULES,
        project_root,
    )?;

    // 4. Resolve devin.
    let dv_name = dv_cfg
        .and_then(|d| d.name.as_deref())
        .or(shared_name)
        .unwrap_or(templates::DEFAULT_NAME_FALLBACK)
        .to_string();

    let dv_description = dv_cfg
        .and_then(|d| d.description.as_deref())
        .or(shared_description)
        .unwrap_or(templates::DEFAULT_DEVIN_DESCRIPTION)
        .to_string();

    let dv_repo_notes = resolve_rules(
        dv_cfg.and_then(|d| d.repo_notes_file.as_deref()),
        dv_cfg.and_then(|d| d.repo_notes_override.as_deref()),
        dv_cfg.and_then(|d| d.extra_repo_notes.as_deref()),
        templates::DEFAULT_DEVIN_REPO_NOTES,
        project_root,
    )?;

    Ok(HubProjectionConfig {
        context7: ResolvedContext7 {
            name: c7_name,
            description: c7_description,
            rules: c7_rules,
        },
        devin: ResolvedDevin {
            name: dv_name,
            description: dv_description,
            repo_notes: dv_repo_notes,
        },
    })
}

/// Resolve a `Vec<String>` list (rules or repo_notes) using the three-stage
/// precedence chain:
///
/// 1. `file_path` — read file relative to `project_root`, split by lines,
///    strip blanks and `#`-comments; full replacement.
/// 2. `override_list` — use as-is; full replacement.
/// 3. `default_list ++ extra` — core defaults concatenated with extras.
fn resolve_rules(
    file_path: Option<&str>,
    override_list: Option<&[String]>,
    extra: Option<&[String]>,
    default_list: &[&str],
    project_root: Option<&Path>,
) -> Result<Vec<String>, String> {
    if let Some(rel_path) = file_path {
        // Resolve relative path against project_root; fall back to cwd.
        let abs_path = if let Some(root) = project_root {
            root.join(rel_path)
        } else {
            Path::new(rel_path).to_path_buf()
        };
        let content = std::fs::read_to_string(&abs_path).map_err(|e| {
            format!(
                "gendoc: rules_file '{}' load failed: {e}",
                abs_path.display()
            )
        })?;
        let lines: Vec<String> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        return Ok(lines);
    }

    if let Some(ov) = override_list {
        return Ok(ov.to_vec());
    }

    // Default + extras.
    let mut result: Vec<String> = default_list.iter().map(|s| s.to_string()).collect();
    if let Some(ex) = extra {
        result.extend(ex.iter().cloned());
    }
    Ok(result)
}

// ── Existing preset resolution (unchanged) ────────────────────────────

pub fn resolve_hub_dist_preset(
    preset: Option<&str>,
    project_root: Option<&str>,
    source_dir: &str,
    projections: Option<&[String]>,
    config_path: Option<&str>,
    lint_strict: Option<bool>,
) -> Result<HubDistPresetResolution, String> {
    let mut overrides_source: Vec<String> = Vec::new();

    let resolved_root = resolve_project_root(project_root);
    if resolved_root.is_some() {
        overrides_source.push("project_root".to_string());
    }

    let preset_name = preset.map(|s| s.trim()).filter(|s| !s.is_empty());

    // Start from explicit caller knobs.
    let caller_projections = projections.map(|p| p.to_vec());
    let caller_config_path = config_path.map(|s| s.to_string());
    let caller_lint_strict = lint_strict;

    let mut eff_projections = caller_projections.clone();
    let mut eff_config_path = caller_config_path.clone();
    let mut eff_lint_strict = caller_lint_strict;

    if let Some(name) = preset_name {
        if name != "publish" {
            return Err(format!(
                "dist: unknown preset '{name}' (allowed: publish); bump {PRESET_CATALOG_VERSION} if adding presets"
            ));
        }
    }

    // Optional overrides from `alc.toml` at the resolved project root.
    if let Some(root) = resolved_root.as_deref() {
        let alc_path = root.join("alc.toml");
        if alc_path.is_file() {
            let raw = std::fs::read_to_string(&alc_path)
                .map_err(|e| format!("dist: failed to read {}: {e}", alc_path.display()))?;
            let parsed: HubDistToml =
                toml::from_str(&raw).map_err(|e| format!("dist: failed to parse alc.toml: {e}"))?;

            if let Some(hub) = parsed.hub.as_ref() {
                if let Some(dist) = hub.dist.as_ref() {
                    if let Some(v) = dist.preset_catalog_version.as_deref() {
                        if !v.trim().is_empty() && v.trim() != PRESET_CATALOG_VERSION {
                            // Doc-only marker today; still surface mismatches loudly so
                            // hub repos don't silently assume a different catalog.
                            return Err(format!(
                                "dist: alc.toml hub.dist.preset_catalog_version={v:?} does not match builtin {PRESET_CATALOG_VERSION}"
                            ));
                        }
                    }

                    if let Some(name) = preset_name {
                        if let Some(map) = dist.presets.as_ref() {
                            if let Some(ov) = map.get(name) {
                                overrides_source.push("alc.toml".to_string());

                                // `alc.toml` may refine defaults, but explicit MCP args win.
                                if caller_projections.is_none() {
                                    if let Some(p) = ov.projections.as_ref() {
                                        eff_projections = Some(p.clone());
                                    }
                                }

                                if caller_config_path.is_none() {
                                    eff_config_path = ov.config_path.clone();
                                }

                                if caller_lint_strict.is_none() {
                                    eff_lint_strict = ov.lint_strict;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply builtin preset defaults after optional `alc.toml` refinement.
    if preset_name.is_some() {
        overrides_source.push("builtin".to_string());

        if eff_projections.is_none() {
            // Safe default: machine-readable hub entries + lint pass, without
            // requiring optional projection configs (context7/devin).
            eff_projections = Some(vec!["hub".to_string(), "lint".to_string()]);
        }
        if eff_lint_strict.is_none() {
            eff_lint_strict = Some(false);
        }
    }

    // Resolve relative config_path against project root (preferred) or the
    // hub source directory (fallback for hub-only repos without alc.toml).
    if let Some(p) = eff_config_path.as_deref() {
        let path = Path::new(p);
        if !path.is_absolute() {
            let source_base = Path::new(source_dir);
            let candidate_source = source_base.join(path);
            let candidate_project = resolved_root.as_deref().map(|root| root.join(path));

            let chosen = if candidate_source.is_file() {
                candidate_source
            } else if let Some(c) = candidate_project {
                if c.is_file() {
                    c
                } else {
                    candidate_source
                }
            } else {
                candidate_source
            };

            eff_config_path = Some(chosen.to_string_lossy().to_string());
        }
    }

    Ok(HubDistPresetResolution {
        projections: eff_projections,
        config_path: eff_config_path,
        lint_strict: eff_lint_strict,
        catalog_version: PRESET_CATALOG_VERSION.to_string(),
        preset_name: preset_name.map(|s| s.to_string()),
        overrides_source,
        resolved_project_root: resolved_root,
    })
}

pub fn preset_meta_value(resolution: &HubDistPresetResolution) -> serde_json::Value {
    json!({
        "name": resolution.preset_name.as_deref(),
        "catalog_version": resolution.catalog_version,
        "resolved": {
            "projections": resolution.projections,
            "config_path": resolution.config_path,
            "lint_strict": resolution.lint_strict,
            "project_root": resolution.resolved_project_root.as_ref().map(|p| p.display().to_string()),
            "overrides_source": resolution.overrides_source,
            "preset_ref": serde_json::Value::Null,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing preset tests (unchanged) ─────────────────────────────

    #[test]
    fn publish_defaults_to_hub_and_lint_when_projections_omitted() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Minimal alc.toml so resolve_project_root can find a root even when
        // we pass an explicit project_root below.
        std::fs::write(root.join("alc.toml"), "[packages]\n").expect("write alc.toml");

        let source_dir = root.join("src");
        std::fs::create_dir_all(&source_dir).expect("mkdir");

        let res = resolve_hub_dist_preset(
            Some("publish"),
            Some(root.to_str().unwrap()),
            source_dir.to_str().unwrap(),
            None,
            None,
            None,
        )
        .expect("resolve");

        assert_eq!(
            res.projections,
            Some(vec!["hub".to_string(), "lint".to_string()])
        );
        assert_eq!(res.lint_strict, Some(false));
        assert!(res.config_path.is_none());
    }

    #[test]
    fn alc_toml_preset_section_overrides_projections() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("alc.toml"),
            r#"[packages]

[hub.dist]

[hub.dist.presets.publish]
projections = ["context7"]
config_path = "configs.toml"
"#,
        )
        .expect("write alc.toml");

        let source_dir = root.join("hub");
        std::fs::create_dir_all(&source_dir).expect("mkdir");
        std::fs::write(
            root.join("configs.toml"),
            "[context7]\nprojectTitle=\"x\"\nrules=[]\n",
        )
        .expect("write configs");

        let res = resolve_hub_dist_preset(
            Some("publish"),
            Some(root.to_str().unwrap()),
            source_dir.to_str().unwrap(),
            None,
            None,
            None,
        )
        .expect("resolve");

        assert_eq!(res.projections, Some(vec!["context7".to_string()]));
        assert_eq!(
            res.config_path.as_deref(),
            Some(root.join("configs.toml").to_str().unwrap())
        );
    }

    // ── New projection config tests ───────────────────────────────────

    #[test]
    fn load_projection_config_default_only() {
        // No project root → templates-only config.
        let cfg = load_hub_projection_config(None).expect("load");

        assert_eq!(cfg.context7.name, templates::DEFAULT_NAME_FALLBACK);
        assert_eq!(cfg.context7.description, templates::DEFAULT_C7_DESCRIPTION);
        assert_eq!(
            cfg.context7.rules,
            templates::DEFAULT_C7_RULES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );

        assert_eq!(cfg.devin.name, templates::DEFAULT_NAME_FALLBACK);
        assert_eq!(cfg.devin.description, templates::DEFAULT_DEVIN_DESCRIPTION);
        assert_eq!(
            cfg.devin.repo_notes,
            templates::DEFAULT_DEVIN_REPO_NOTES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn load_projection_config_name_only_override() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub]
name = "my-project"
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load");

        // [hub].name propagates to both c7 and devin.
        assert_eq!(cfg.context7.name, "my-project");
        assert_eq!(cfg.devin.name, "my-project");
        // Descriptions fall back to defaults.
        assert_eq!(cfg.context7.description, templates::DEFAULT_C7_DESCRIPTION);
        assert_eq!(cfg.devin.description, templates::DEFAULT_DEVIN_DESCRIPTION);
    }

    #[test]
    fn load_projection_config_extra_rules_append() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub.context7]
extra_rules = ["Custom rule A", "Custom rule B"]
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load");

        let mut expected: Vec<String> = templates::DEFAULT_C7_RULES
            .iter()
            .map(|s| s.to_string())
            .collect();
        expected.push("Custom rule A".to_string());
        expected.push("Custom rule B".to_string());

        assert_eq!(cfg.context7.rules, expected);
    }

    #[test]
    fn load_projection_config_rules_override_replaces() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub.context7]
rules_override = ["Only this rule"]
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load");

        assert_eq!(cfg.context7.rules, vec!["Only this rule".to_string()]);
    }

    #[test]
    fn load_projection_config_rules_file_reads() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Write a rules file with blanks and a comment.
        std::fs::write(
            root.join("my_rules.txt"),
            "Rule one\n# ignored comment\n\nRule two\n",
        )
        .expect("write rules file");

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub.context7]
rules_file = "my_rules.txt"
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load");

        assert_eq!(
            cfg.context7.rules,
            vec!["Rule one".to_string(), "Rule two".to_string()]
        );
    }

    #[test]
    fn load_projection_config_mutually_exclusive_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(root.join("rules.txt"), "Rule\n").expect("write rules file");

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub.context7]
rules_file = "rules.txt"
rules_override = ["Also a rule"]
"#,
        )
        .expect("write alc.toml");

        let err = load_hub_projection_config(Some(root)).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutually-exclusive error, got: {err}"
        );
    }

    #[test]
    fn load_projection_config_devin_equivalent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Test extra_repo_notes path.
        std::fs::write(
            root.join("alc.toml"),
            r#"[hub.devin]
extra_repo_notes = ["Extra note"]
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load extra");
        let mut expected: Vec<String> = templates::DEFAULT_DEVIN_REPO_NOTES
            .iter()
            .map(|s| s.to_string())
            .collect();
        expected.push("Extra note".to_string());
        assert_eq!(cfg.devin.repo_notes, expected);

        // Test repo_notes_override path.
        let tmp2 = tempfile::tempdir().expect("tempdir2");
        let root2 = tmp2.path();
        std::fs::write(
            root2.join("alc.toml"),
            r#"[hub.devin]
repo_notes_override = ["Only note"]
"#,
        )
        .expect("write alc.toml");

        let cfg2 = load_hub_projection_config(Some(root2)).expect("load override");
        assert_eq!(cfg2.devin.repo_notes, vec!["Only note".to_string()]);

        // Test repo_notes_file path.
        let tmp3 = tempfile::tempdir().expect("tempdir3");
        let root3 = tmp3.path();
        std::fs::write(root3.join("notes.txt"), "Note A\nNote B\n").expect("write notes");
        std::fs::write(
            root3.join("alc.toml"),
            r#"[hub.devin]
repo_notes_file = "notes.txt"
"#,
        )
        .expect("write alc.toml");

        let cfg3 = load_hub_projection_config(Some(root3)).expect("load file");
        assert_eq!(
            cfg3.devin.repo_notes,
            vec!["Note A".to_string(), "Note B".to_string()]
        );

        // Test mutually-exclusive error for devin.
        let tmp4 = tempfile::tempdir().expect("tempdir4");
        let root4 = tmp4.path();
        std::fs::write(root4.join("notes.txt"), "Note\n").expect("write notes");
        std::fs::write(
            root4.join("alc.toml"),
            r#"[hub.devin]
repo_notes_file = "notes.txt"
repo_notes_override = ["conflict"]
"#,
        )
        .expect("write alc.toml");

        let err = load_hub_projection_config(Some(root4)).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected devin mutually-exclusive error, got: {err}"
        );
    }

    #[test]
    fn hub_section_backward_compat_dist_only() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // alc.toml with only [hub.dist] — new fields absent.
        std::fs::write(
            root.join("alc.toml"),
            r#"[packages]

[hub.dist]

[hub.dist.presets.publish]
projections = ["hub", "lint"]
"#,
        )
        .expect("write alc.toml");

        let source_dir = root.join("hub");
        std::fs::create_dir_all(&source_dir).expect("mkdir");

        // resolve_hub_dist_preset must still work correctly.
        let res = resolve_hub_dist_preset(
            Some("publish"),
            Some(root.to_str().unwrap()),
            source_dir.to_str().unwrap(),
            None,
            None,
            None,
        )
        .expect("resolve");

        assert_eq!(
            res.projections,
            Some(vec!["hub".to_string(), "lint".to_string()])
        );

        // load_hub_projection_config also works: no context7/devin sections
        // → falls back to defaults.
        let cfg = load_hub_projection_config(Some(root)).expect("load projection");
        assert_eq!(cfg.context7.name, templates::DEFAULT_NAME_FALLBACK);
        assert_eq!(cfg.devin.name, templates::DEFAULT_NAME_FALLBACK);
    }

    #[test]
    fn to_devin_toml_wraps_repo_notes_as_content_table() {
        let resolved = ResolvedDevin {
            name: "test".to_string(),
            description: "desc".to_string(),
            repo_notes: vec!["a".to_string(), "b".to_string()],
        };
        let cfg = HubProjectionConfig {
            context7: ResolvedContext7 {
                name: "test".to_string(),
                description: "desc".to_string(),
                rules: vec![],
            },
            devin: resolved,
        };

        let val = cfg.to_devin_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        let repo_notes = match table.get("repo_notes") {
            Some(toml::Value::Array(arr)) => arr,
            _ => panic!("expected repo_notes array"),
        };

        assert_eq!(repo_notes.len(), 2);

        for (item, expected_content) in repo_notes.iter().zip(["a", "b"].iter()) {
            match item {
                toml::Value::Table(t) => {
                    let content = t.get("content").expect("missing content key");
                    assert_eq!(
                        content,
                        &toml::Value::String(expected_content.to_string()),
                        "content mismatch for note"
                    );
                    // Must not have extra keys beyond content (no author).
                    assert_eq!(t.len(), 1, "unexpected extra keys in note table");
                }
                _ => panic!("expected each repo_note to be a Table, got: {item:?}"),
            }
        }
    }

    #[test]
    fn to_context7_toml_wires_project_title_from_hub_name() {
        let cfg = HubProjectionConfig {
            context7: ResolvedContext7 {
                name: "my-project".to_string(),
                description: "A description".to_string(),
                rules: vec!["Rule 1".to_string()],
            },
            devin: ResolvedDevin {
                name: "my-project".to_string(),
                description: "desc".to_string(),
                repo_notes: vec![],
            },
        };

        let val = cfg.to_context7_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        // Key must be "projectTitle", not "name".
        assert!(
            table.get("name").is_none(),
            "unexpected 'name' key in context7 output"
        );
        assert_eq!(
            table.get("projectTitle"),
            Some(&toml::Value::String("my-project".to_string())),
            "expected projectTitle = 'my-project'"
        );
        assert_eq!(
            table.get("description"),
            Some(&toml::Value::String("A description".to_string())),
            "expected description to be present"
        );
    }

    #[test]
    fn to_devin_toml_wires_project_name_from_hub_name() {
        let cfg = HubProjectionConfig {
            context7: ResolvedContext7 {
                name: "my-project".to_string(),
                description: "desc".to_string(),
                rules: vec![],
            },
            devin: ResolvedDevin {
                name: "my-project".to_string(),
                description: "Devin description".to_string(),
                repo_notes: vec![],
            },
        };

        let val = cfg.to_devin_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        // Key must be "project_name", not "name".
        assert!(
            table.get("name").is_none(),
            "unexpected 'name' key in devin output"
        );
        assert_eq!(
            table.get("project_name"),
            Some(&toml::Value::String("my-project".to_string())),
            "expected project_name = 'my-project'"
        );
        assert_eq!(
            table.get("description"),
            Some(&toml::Value::String("Devin description".to_string())),
            "expected description to be present"
        );
    }

    #[test]
    fn to_devin_toml_wires_description_core_default() {
        // No alc.toml → load_hub_projection_config uses DEFAULT_DEVIN_DESCRIPTION.
        let cfg = load_hub_projection_config(None).expect("load");

        let val = cfg.to_devin_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        assert_eq!(
            table.get("description"),
            Some(&toml::Value::String(
                templates::DEFAULT_DEVIN_DESCRIPTION.to_string()
            )),
            "expected DEFAULT_DEVIN_DESCRIPTION in devin output"
        );
    }

    #[test]
    fn to_context7_toml_uses_default_name_fallback_when_no_name_configured() {
        // No alc.toml → load_hub_projection_config uses DEFAULT_NAME_FALLBACK.
        let cfg = load_hub_projection_config(None).expect("load");

        let val = cfg.to_context7_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        assert_eq!(
            table.get("projectTitle"),
            Some(&toml::Value::String(
                templates::DEFAULT_NAME_FALLBACK.to_string()
            )),
            "expected DEFAULT_NAME_FALLBACK as projectTitle when no name configured"
        );
    }

    #[test]
    fn to_devin_toml_uses_default_name_fallback_when_no_name_configured() {
        // No alc.toml → load_hub_projection_config uses DEFAULT_NAME_FALLBACK.
        let cfg = load_hub_projection_config(None).expect("load");

        let val = cfg.to_devin_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        assert_eq!(
            table.get("project_name"),
            Some(&toml::Value::String(
                templates::DEFAULT_NAME_FALLBACK.to_string()
            )),
            "expected DEFAULT_NAME_FALLBACK as project_name when no name configured"
        );
    }

    #[test]
    fn to_context7_toml_propagates_hub_name_via_load() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("alc.toml"),
            r#"[hub]
name = "test-hub"
"#,
        )
        .expect("write alc.toml");

        let cfg = load_hub_projection_config(Some(root)).expect("load");

        let val = cfg.to_context7_toml();
        let table = match &val {
            toml::Value::Table(t) => t,
            _ => panic!("expected Table"),
        };

        assert_eq!(
            table.get("projectTitle"),
            Some(&toml::Value::String("test-hub".to_string())),
            "expected [hub].name to propagate to projectTitle"
        );
    }
}
