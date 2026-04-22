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

#[derive(Debug, Deserialize)]
struct HubDistToml {
    hub: Option<HubSection>,
}

#[derive(Debug, Deserialize)]
struct HubSection {
    dist: Option<HubDistSection>,
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
}
