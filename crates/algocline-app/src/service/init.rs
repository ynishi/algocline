//! `alc_init` — initialize `alc.toml` in a project root.

use super::alc_toml::{alc_toml_path, save_alc_toml};
use super::project::resolve_project_root;
use super::AppService;

impl AppService {
    pub async fn init(&self, project_root: Option<String>) -> Result<String, String> {
        // resolve: explicit → ALC_PROJECT_ROOT → walk_up (None if alc.toml absent) → cwd
        let root = match resolve_project_root(project_root.as_deref()) {
            Some(r) => r,
            None => std::env::current_dir().map_err(|e| format!("Cannot determine cwd: {e}"))?,
        };

        let path = alc_toml_path(&root);
        if path.exists() {
            return Err(format!("alc.toml already exists at {}", path.display()));
        }

        let doc: toml_edit::DocumentMut = "[packages]\n"
            .parse()
            .map_err(|e: toml_edit::TomlError| format!("Internal error: {e}"))?;
        save_alc_toml(&root, &doc)?;

        let result = serde_json::json!({ "created": path.display().to_string() });
        Ok(result.to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::service::test_support::make_app_service as make_service;

    #[tokio::test]
    async fn init_creates_alc_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service().await;
        let result = svc
            .init(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        assert!(result.contains("created"));
        assert!(tmp.path().join("alc.toml").exists());

        let content = std::fs::read_to_string(tmp.path().join("alc.toml")).unwrap();
        assert!(content.contains("[packages]"));
    }

    #[tokio::test]
    async fn init_fails_if_alc_toml_exists() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("alc.toml"), "[packages]\n").unwrap();
        let svc = make_service().await;
        let err = svc
            .init(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap_err();
        assert!(err.contains("already exists"));
    }
}
