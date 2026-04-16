//! `alc_init` — initialize `alc.toml` in a project root.

use std::path::Path;

use super::alc_toml::{alc_toml_path, save_alc_toml};
use super::project::resolve_project_root;
use super::AppService;

/// Entry to ensure is present in `.gitignore` after `alc_init`.
///
/// Matches the physical filename chosen in `decisions.md` Q1. The logical
/// scope name (`variant`) and the physical filename (`alc.local.toml`) are
/// intentionally asymmetric — the filename follows the dotenv `.env.local`
/// convention so "gitignored" is read at a glance.
const GITIGNORE_ENTRY: &str = "alc.local.toml";

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

        // Best-effort .gitignore append. Failures are surfaced to the caller
        // rather than swallowed — the whole point of `alc_init` is to set up
        // a reproducible project shape, and a silent gitignore failure
        // would leak `alc.local.toml` into VCS later.
        let gitignore_path = root.join(".gitignore");
        let gitignore_updated = update_gitignore(&root, GITIGNORE_ENTRY)?;

        let result = serde_json::json!({
            "created": path.display().to_string(),
            "gitignore_path": gitignore_path.display().to_string(),
            "gitignore_updated": gitignore_updated,
        });
        Ok(result.to_string())
    }
}

/// Ensure `entry` appears as a line in `{root}/.gitignore`.
///
/// - Missing file → create with just `entry\n`.
/// - Present, entry already on its own line (ignoring surrounding whitespace)
///   → no-op.
/// - Present but entry absent → append `entry\n`, inserting a leading newline
///   if the existing file does not end in one.
///
/// Returns `Ok(true)` when the file was written, `Ok(false)` when the entry
/// was already present. Comment-style matches (`# alc.local.toml`) are not
/// treated as existing entries — they're comments, not patterns.
pub(crate) fn update_gitignore(root: &Path, entry: &str) -> Result<bool, String> {
    let path = root.join(".gitignore");

    if !path.exists() {
        std::fs::write(&path, format!("{entry}\n"))
            .map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
        return Ok(true);
    }

    let existing = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let already_present = existing.lines().any(|line| {
        let trimmed = line.trim();
        !trimmed.starts_with('#') && trimmed == entry
    });

    if already_present {
        return Ok(false);
    }

    let mut new_content = existing;
    if !new_content.is_empty() && !new_content.ends_with('\n') {
        new_content.push('\n');
    }
    new_content.push_str(entry);
    new_content.push('\n');

    std::fs::write(&path, new_content)
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::update_gitignore;
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

    #[tokio::test]
    async fn init_creates_gitignore_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_service().await;
        let raw = svc
            .init(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["gitignore_updated"], true);

        let gi = tmp.path().join(".gitignore");
        assert!(gi.exists());
        let content = std::fs::read_to_string(&gi).unwrap();
        assert_eq!(content, "alc.local.toml\n");
    }

    #[tokio::test]
    async fn init_appends_to_existing_gitignore() {
        let tmp = tempfile::tempdir().unwrap();
        let gi = tmp.path().join(".gitignore");
        std::fs::write(&gi, "target\nworkspace\n").unwrap();

        let svc = make_service().await;
        let raw = svc
            .init(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["gitignore_updated"], true);

        let content = std::fs::read_to_string(&gi).unwrap();
        assert_eq!(content, "target\nworkspace\nalc.local.toml\n");
    }

    #[tokio::test]
    async fn init_is_idempotent_on_gitignore_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let gi = tmp.path().join(".gitignore");
        std::fs::write(&gi, "target\nalc.local.toml\nworkspace\n").unwrap();

        let svc = make_service().await;
        let raw = svc
            .init(Some(tmp.path().to_str().unwrap().to_string()))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(json["gitignore_updated"], false);

        // File unchanged.
        let content = std::fs::read_to_string(&gi).unwrap();
        assert_eq!(content, "target\nalc.local.toml\nworkspace\n");
    }

    #[tokio::test]
    async fn update_gitignore_adds_trailing_newline_if_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let gi = tmp.path().join(".gitignore");
        std::fs::write(&gi, "target").unwrap(); // no trailing \n

        let updated = update_gitignore(tmp.path(), "alc.local.toml").unwrap();
        assert!(updated);

        let content = std::fs::read_to_string(&gi).unwrap();
        assert_eq!(content, "target\nalc.local.toml\n");
    }

    #[tokio::test]
    async fn update_gitignore_does_not_match_commented_line() {
        // A commented-out `# alc.local.toml` must not be mistaken for an
        // existing entry — the entry is still absent.
        let tmp = tempfile::tempdir().unwrap();
        let gi = tmp.path().join(".gitignore");
        std::fs::write(&gi, "# alc.local.toml\ntarget\n").unwrap();

        let updated = update_gitignore(tmp.path(), "alc.local.toml").unwrap();
        assert!(updated);

        let content = std::fs::read_to_string(&gi).unwrap();
        assert_eq!(content, "# alc.local.toml\ntarget\nalc.local.toml\n");
    }
}
