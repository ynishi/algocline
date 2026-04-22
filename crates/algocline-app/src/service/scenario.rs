use std::path::Path;

use super::path::ContainedPath;
use super::resolve::{
    install_scenarios_from_dir, resolve_scenario_source, scenarios_dir, DirEntryFailures,
};
use super::AppService;

impl AppService {
    /// List available scenarios in `~/.algocline/scenarios/`.
    ///
    /// Per-entry I/O errors are collected in `"failures"` rather than aborting.
    pub fn scenario_list(&self) -> Result<String, String> {
        let dir = scenarios_dir(&self.log_config.app_dir());
        if !dir.exists() {
            return Ok(serde_json::json!({ "scenarios": [], "failures": [] }).to_string());
        }

        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("Failed to read scenarios dir: {e}"))?;

        let mut scenarios: Vec<serde_json::Value> = Vec::new();
        let mut failures: DirEntryFailures = Vec::new();
        for entry_result in entries {
            let entry = match entry_result {
                Ok(e) => e,
                Err(e) => {
                    failures.push(format!("readdir entry: {e}"));
                    continue;
                }
            };
            let path = entry.path();
            let name = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let ext = path.extension().and_then(|s| s.to_str());
            if ext != Some("lua") {
                continue;
            }
            let metadata = std::fs::metadata(&path);
            let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
            scenarios.push(serde_json::json!({
                "name": name,
                "path": path.to_string_lossy(),
                "size_bytes": size_bytes,
            }));
        }

        scenarios.sort_by(|a, b| {
            a.get("name")
                .and_then(|v| v.as_str())
                .cmp(&b.get("name").and_then(|v| v.as_str()))
        });

        Ok(serde_json::json!({
            "scenarios": scenarios,
            "failures": failures,
        })
        .to_string())
    }

    /// Show the content of a named scenario.
    pub fn scenario_show(&self, name: &str) -> Result<String, String> {
        let dir = scenarios_dir(&self.log_config.app_dir());
        let path = ContainedPath::child(&dir, &format!("{name}.lua"))
            .map_err(|e| format!("Invalid scenario name: {e}"))?;
        if !path.as_ref().exists() {
            return Err(format!("Scenario '{name}' not found"));
        }
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read scenario '{name}': {e}"))?;
        Ok(serde_json::json!({
            "name": name,
            "path": path.as_ref().to_string_lossy(),
            "content": content,
        })
        .to_string())
    }

    /// Install scenarios from a Git URL or local path into `~/.algocline/scenarios/`.
    ///
    /// Expects the source to contain `.lua` files (at root or in a `scenarios/` subdirectory).
    pub async fn scenario_install(&self, url: String) -> Result<String, String> {
        let dest_dir = scenarios_dir(&self.log_config.app_dir());
        std::fs::create_dir_all(&dest_dir)
            .map_err(|e| format!("Failed to create scenarios dir: {e}"))?;

        // Local path: copy .lua files directly
        let local_path = Path::new(&url);
        if local_path.is_absolute() && local_path.is_dir() {
            return install_scenarios_from_dir(local_path, &dest_dir);
        }

        // Normalize URL
        let git_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
            || url.starts_with("git@")
        {
            url.clone()
        } else {
            format!("https://{url}")
        };

        let staging = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let output = tokio::process::Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                &git_url,
                &staging.path().to_string_lossy(),
            ])
            .output()
            .await
            .map_err(|e| format!("Failed to run git: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {stderr}"));
        }

        let source = resolve_scenario_source(staging.path());
        install_scenarios_from_dir(&source, &dest_dir)
    }
}
