//! Card service layer — MCP-facing read/write operations.
//!
//! Thin adapter between MCP tool handlers and [`algocline_engine::card`].
//! All data flows through the engine; this layer handles JSON
//! serialization for the MCP transport.
//!
//! For Card schema, storage layout, and design principles, see
//! [`algocline_engine::card`] module documentation.

use std::path::Path;

use algocline_engine::card;

use super::hub;
use super::AppService;

impl AppService {
    /// List Cards as JSON summaries, optionally filtered by package.
    pub fn card_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = card::list(pkg)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Fetch full Card body (Tier 1) by id.
    pub fn card_get(&self, card_id: &str) -> Result<String, String> {
        match card::get(card_id)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("card '{card_id}' not found")),
        }
    }

    /// Query Cards with sort, filter, and limit.
    #[allow(clippy::too_many_arguments)]
    pub fn card_find(
        &self,
        pkg: Option<String>,
        scenario: Option<String>,
        model: Option<String>,
        sort: Option<String>,
        limit: Option<usize>,
        min_pass_rate: Option<f64>,
    ) -> Result<String, String> {
        let q = card::FindQuery {
            pkg,
            scenario,
            model,
            sort,
            limit,
            min_pass_rate,
        };
        let rows = card::find(q)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Resolve alias then fetch the full Card.
    pub fn card_get_by_alias(&self, name: &str) -> Result<String, String> {
        match card::get_by_alias(name)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("alias '{name}' not found")),
        }
    }

    /// List aliases, optionally filtered by package.
    pub fn card_alias_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = card::alias_list(pkg)?;
        Ok(card::aliases_to_json(&rows).to_string())
    }

    /// Pin or rebind a mutable alias to a Card.
    pub fn card_alias_set(
        &self,
        name: &str,
        card_id: &str,
        pkg: Option<&str>,
        note: Option<&str>,
    ) -> Result<String, String> {
        let alias = card::alias_set(name, card_id, pkg, note)?;
        let arr = card::aliases_to_json(std::slice::from_ref(&alias));
        let single = arr
            .as_array()
            .and_then(|a| a.first().cloned())
            .unwrap_or(serde_json::Value::Null);
        Ok(single.to_string())
    }

    /// Additive-only annotation — new top-level keys only.
    pub fn card_append(&self, card_id: &str, fields: serde_json::Value) -> Result<String, String> {
        let merged = card::append(card_id, fields)?;
        Ok(merged.to_string())
    }

    /// Install Cards from a Card Collection repo (Git URL or local path).
    ///
    /// A Card Collection is identified by `alc_cards.toml` at the repo root.
    /// Each subdirectory is treated as a package name, and `*.toml` card files
    /// within are imported into `~/.algocline/cards/{pkg}/`.
    pub async fn card_install(&self, url: String) -> Result<String, String> {
        // Local path: import directly
        let local_path = Path::new(&url);
        if local_path.is_absolute() && local_path.is_dir() {
            return self.card_install_from_dir(local_path, &url);
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

        // Clone to temp directory
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

        self.card_install_from_dir(staging.path(), &url)
    }

    /// Import Cards from a local directory (Card Collection or bare cards dir).
    fn card_install_from_dir(&self, root: &Path, source: &str) -> Result<String, String> {
        // Verify this is a Card Collection (alc_cards.toml present)
        let manifest_path = root.join("alc_cards.toml");
        if !manifest_path.exists() {
            return Err("Not a Card Collection: alc_cards.toml not found at root. \
                 Card Collections must have an alc_cards.toml manifest."
                .into());
        }

        let mut all_imported: Vec<String> = Vec::new();
        let mut all_skipped: Vec<String> = Vec::new();
        let mut packages: Vec<String> = Vec::new();

        let entries =
            std::fs::read_dir(root).map_err(|e| format!("Failed to read source dir: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let pkg_name = match entry.file_name().to_str() {
                Some(n) if !n.starts_with('_') && !n.starts_with('.') => n.to_string(),
                _ => continue,
            };

            // Check if dir has any .toml files (cards)
            let has_toml = std::fs::read_dir(&path)
                .map(|entries| {
                    entries
                        .flatten()
                        .any(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
                })
                .unwrap_or(false);

            if !has_toml {
                continue;
            }

            let (imported, skipped) = card::import_from_dir(&path, &pkg_name)?;
            if !imported.is_empty() || !skipped.is_empty() {
                packages.push(pkg_name);
            }
            all_imported.extend(imported);
            all_skipped.extend(skipped);
        }

        if all_imported.is_empty() && all_skipped.is_empty() {
            return Err("No Card files found in any subdirectory.".into());
        }

        // Register source for Hub index discovery
        hub::register_source(source, "card_install");

        let response = serde_json::json!({
            "installed_cards": all_imported,
            "skipped_cards": all_skipped,
            "packages": packages,
            "source": source,
            "mode": "card_collection",
        });
        Ok(response.to_string())
    }

    /// Import bundled Cards from a package's `cards/` subdirectory.
    ///
    /// Called by `pkg_install` when a package contains a `cards/` dir.
    /// Returns imported card_ids (may be empty if all were skipped).
    pub(crate) fn import_pkg_bundled_cards(pkg_name: &str, cards_dir: &Path) -> Vec<String> {
        match card::import_from_dir(cards_dir, pkg_name) {
            Ok((imported, _)) => imported,
            Err(e) => {
                tracing::warn!("Failed to import bundled cards for '{pkg_name}': {e}");
                Vec::new()
            }
        }
    }

    /// Read per-case sidecar rows (Tier 2) with offset/limit paging.
    pub fn card_samples(
        &self,
        card_id: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<String, String> {
        let rows = card::read_samples(card_id, offset, limit)?;
        Ok(serde_json::Value::Array(rows).to_string())
    }
}
