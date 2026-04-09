//! `alc_migrate` — convert legacy `alc.lock` to `alc.toml` + new `alc.lock`.

use super::alc_toml::{alc_toml_path, save_alc_toml};
use super::lockfile::lockfile_path;
use super::AppService;

impl AppService {
    pub async fn migrate(&self, project_root: Option<String>) -> Result<String, String> {
        // Resolve root: explicit → ALC_PROJECT_ROOT → cwd.
        // walk_up is not used because alc.toml may not exist yet.
        let root = if let Some(s) = project_root.as_deref() {
            std::path::PathBuf::from(s)
        } else if let Ok(env) = std::env::var("ALC_PROJECT_ROOT") {
            if !env.is_empty() {
                std::path::PathBuf::from(env)
            } else {
                std::env::current_dir().map_err(|e| format!("Cannot determine cwd: {e}"))?
            }
        } else {
            std::env::current_dir().map_err(|e| format!("Cannot determine cwd: {e}"))?
        };

        let lock_path = lockfile_path(&root);
        if !lock_path.exists() {
            return Ok(serde_json::json!({
                "status": "nothing to migrate",
                "reason": "alc.lock not found"
            })
            .to_string());
        }

        let content = std::fs::read_to_string(&lock_path)
            .map_err(|e| format!("Failed to read alc.lock: {e}"))?;

        // Detect legacy format by parsing TOML and checking for structural
        // markers: `linked_at` fields on [[package]] entries or source
        // `type = "local_dir"`.  String-contains was previously used but
        // could false-positive on package names containing those strings.
        let is_legacy = detect_legacy_format(&content);
        if !is_legacy {
            return Ok(serde_json::json!({
                "status": "nothing to migrate",
                "reason": "already new format or no local_dir entries"
            })
            .to_string());
        }

        let toml_path = alc_toml_path(&root);
        if toml_path.exists() {
            return Err(format!(
                "alc.toml already exists at {}. Remove it first or migrate manually.",
                toml_path.display()
            ));
        }

        // Parse legacy TOML line-by-line to extract local_dir entries.
        let mut doc: toml_edit::DocumentMut = "[packages]\n"
            .parse()
            .map_err(|e: toml_edit::TomlError| format!("Internal error: {e}"))?;

        {
            let mut current_name: Option<String> = None;
            let mut current_path: Option<String> = None;
            let mut in_local_dir = false;

            let flush =
                |doc: &mut toml_edit::DocumentMut, name: Option<String>, path: Option<String>| {
                    if let (Some(n), Some(p)) = (name, path) {
                        if let Some(tbl) = doc["packages"].as_table_mut() {
                            let mut inline = toml_edit::InlineTable::new();
                            inline.insert("path", p.as_str().into());
                            tbl.insert(
                                &n,
                                toml_edit::Item::Value(toml_edit::Value::InlineTable(inline)),
                            );
                        }
                    }
                };

            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed == "[[package]]" {
                    if in_local_dir {
                        flush(&mut doc, current_name.take(), current_path.take());
                    }
                    current_name = None;
                    current_path = None;
                    in_local_dir = false;
                } else if let Some(v) = trimmed.strip_prefix("name = ") {
                    current_name = Some(v.trim_matches('"').to_string());
                } else if trimmed.contains("local_dir") {
                    in_local_dir = true;
                } else if in_local_dir {
                    if let Some(v) = trimmed.strip_prefix("path = ") {
                        current_path = Some(v.trim_matches('"').to_string());
                    }
                }
            }
            if in_local_dir {
                flush(&mut doc, current_name, current_path);
            }
        }

        // Atomic: write alc.toml first, then rename alc.lock to backup.
        save_alc_toml(&root, &doc)?;

        let bak_path = lock_path.with_extension("lock.bak");
        std::fs::rename(&lock_path, &bak_path)
            .map_err(|e| format!("Failed to rename alc.lock to alc.lock.bak: {e}"))?;

        let result = serde_json::json!({
            "migrated": true,
            "alc_toml": toml_path.display().to_string(),
            "backup": bak_path.display().to_string(),
            "note": "Run alc_update to generate new alc.lock"
        });
        Ok(result.to_string())
    }
}

/// Detect whether `alc.lock` content uses the legacy format.
///
/// Legacy indicators (checked via TOML parse, not string search):
/// - Any `[[package]]` entry has a `linked_at` key
/// - Any `[package.source]` has `type = "local_dir"`
fn detect_legacy_format(content: &str) -> bool {
    let parsed: toml::Value = match toml::from_str(content) {
        Ok(v) => v,
        Err(_) => return false,
    };

    let packages = match parsed.get("package").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return false,
    };

    for pkg in packages {
        let tbl = match pkg.as_table() {
            Some(t) => t,
            None => continue,
        };

        if tbl.contains_key("linked_at") {
            return true;
        }

        if let Some(source) = tbl.get("source").and_then(|s| s.as_table()) {
            if source.get("type").and_then(|t| t.as_str()) == Some("local_dir") {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use crate::service::test_support::make_app_service as make_service;

    const LEGACY_LOCK: &str = r#"version = 1

[[package]]
name = "my_pkg"
linked_at = "2024-01-01T00:00:00Z"

[package.source]
type = "local_dir"
path = "/some/path/my_pkg"
"#;

    #[tokio::test]
    async fn migrate_when_no_lock_returns_nothing_to_migrate() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service().await;
        let result = svc
            .migrate(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        assert!(result.contains("nothing to migrate"), "{result}");
        assert!(result.contains("alc.lock not found"), "{result}");
    }

    #[tokio::test]
    async fn migrate_new_format_lock_returns_nothing_to_migrate() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("alc.lock"),
            "version = 1\n\n[[package]]\nname = \"cot\"\n\n[package.source]\ntype = \"installed\"\n",
        )
        .unwrap();
        let svc = make_service().await;
        let result = svc
            .migrate(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        assert!(result.contains("nothing to migrate"), "{result}");
    }

    #[tokio::test]
    async fn migrate_legacy_lock_creates_alc_toml_and_backup() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alc.lock"), LEGACY_LOCK).unwrap();
        let svc = make_service().await;
        let result = svc
            .migrate(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        assert!(result.contains("\"migrated\":true"), "{result}");
        assert!(tmp.path().join("alc.toml").exists());
        assert!(tmp.path().join("alc.lock.bak").exists());
        assert!(!tmp.path().join("alc.lock").exists());

        let toml_content = std::fs::read_to_string(tmp.path().join("alc.toml")).unwrap();
        assert!(toml_content.contains("[packages]"), "{toml_content}");
        assert!(toml_content.contains("my_pkg"), "{toml_content}");
    }

    #[tokio::test]
    async fn migrate_fails_if_alc_toml_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alc.lock"), LEGACY_LOCK).unwrap();
        std::fs::write(tmp.path().join("alc.toml"), "[packages]\n").unwrap();
        let svc = make_service().await;
        let err = svc
            .migrate(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap_err();
        assert!(err.contains("already exists"), "{err}");
    }
}
