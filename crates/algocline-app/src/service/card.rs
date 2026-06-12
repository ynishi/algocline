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
use serde::{Deserialize, Serialize};

use super::error::CardPublishError;
use super::hub;
use super::AppService;

/// Input shape for [`AppService::card_sink_backfill`]. Deserialized from
/// the Lua/MCP table argument `{ sink, dry_run }`.
#[derive(Debug, Deserialize)]
pub struct SinkBackfillParams {
    pub sink: String,
    #[serde(default)]
    pub dry_run: bool,
}

/// Typed contract for the output produced by a Card analyzer package.
///
/// Host-side validation: after `advice()` returns `status == "completed"`,
/// the `result.result` nested value is deserialized into this struct before
/// being placed at the top level of the MCP response.  Any package that
/// cannot produce all required fields (`pattern`, `suggested_change`,
/// `confidence`) will cause `card_analyze` to return a typed error rather
/// than passing freeform JSON to the caller.
///
/// `failure_count` and `sample_count` are optional so that future analyzer
/// packages may omit them without breaking the typed contract.
#[derive(Debug, Serialize, Deserialize)]
pub struct CardAnalyzeResult {
    /// One-line summary of the dominant failure pattern.
    pub pattern: String,
    /// Concrete improvement suggestion (prompt wording, Lua change, etc.).
    pub suggested_change: String,
    /// Analyzer confidence in the finding, clamped to `0.0..=1.0`.
    pub confidence: f64,
    /// Number of failure samples detected (optional).
    #[serde(default)]
    pub failure_count: Option<u64>,
    /// Total number of samples evaluated (optional).
    #[serde(default)]
    pub sample_count: Option<u64>,
}

/// Default analyzer package name dispatched from
/// [`AppService::card_analyze`] when the caller omits `pkg`.
///
/// This is an **IF promise**, not a bundled hard dependency: any pkg
/// (bundled, project-local, or user-installed) named `card_analysis`
/// that exposes `M.run(ctx) -> ctx` will satisfy it. Not having a pkg
/// of this name installed surfaces as a normal "package not found"
/// error from the underlying `advice` dispatch.
pub const DEFAULT_CARD_ANALYZE_PKG: &str = "card_analysis";

impl AppService {
    /// List Cards as JSON summaries, optionally filtered by package.
    pub fn card_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = self.card_store.list(pkg)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Fetch full Card body (Tier 1) by id.
    pub fn card_get(&self, card_id: &str) -> Result<String, String> {
        match self.card_store.get(card_id)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("card '{card_id}' not found")),
        }
    }

    /// Query Cards using the `where` DSL + `order_by` / limit / offset.
    pub fn card_find(
        &self,
        pkg: Option<String>,
        where_: Option<serde_json::Value>,
        order_by: Option<serde_json::Value>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<String, String> {
        let where_parsed = match where_ {
            Some(v) => Some(card::parse_where(&v)?),
            None => None,
        };
        let order_parsed = match order_by {
            Some(v) => card::parse_order_by(&v)?,
            None => Vec::new(),
        };
        let q = card::FindQuery {
            pkg,
            where_: where_parsed,
            order_by: order_parsed,
            limit,
            offset,
        };
        let rows = self.card_store.find(q)?;
        Ok(card::summaries_to_json(&rows).to_string())
    }

    /// Resolve alias then fetch the full Card.
    pub fn card_get_by_alias(&self, name: &str) -> Result<String, String> {
        match self.card_store.get_by_alias(name)? {
            Some(v) => Ok(v.to_string()),
            None => Err(format!("alias '{name}' not found")),
        }
    }

    /// List aliases, optionally filtered by package.
    pub fn card_alias_list(&self, pkg: Option<&str>) -> Result<String, String> {
        let rows = self.card_store.alias_list(pkg)?;
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
        let alias = self.card_store.alias_set(name, card_id, pkg, note)?;
        let arr = card::aliases_to_json(std::slice::from_ref(&alias));
        let single = arr
            .as_array()
            .and_then(|a| a.first().cloned())
            .unwrap_or(serde_json::Value::Null);
        Ok(single.to_string())
    }

    /// Additive-only annotation — new top-level keys only.
    pub fn card_append(&self, card_id: &str, fields: serde_json::Value) -> Result<String, String> {
        let merged = self.card_store.append(card_id, fields)?;
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

            let (imported, skipped) =
                card::import_from_dir_with_store(&*self.card_store, &path, &pkg_name)?;
            if !imported.is_empty() || !skipped.is_empty() {
                packages.push(pkg_name);
            }
            all_imported.extend(imported);
            all_skipped.extend(skipped);
        }

        if all_imported.is_empty() && all_skipped.is_empty() {
            return Err("No Card files found in any subdirectory.".into());
        }

        // Register source for Hub index discovery. Storage failure here
        // surfaces as `storage_warnings` rather than aborting the
        // import — the Cards themselves are already on disk.
        let mut storage_warnings: Vec<String> = Vec::new();
        if let Err(e) = hub::register_source(&self.log_config.app_dir(), source, "card_install") {
            storage_warnings.push(format!("hub register_source: {e}"));
        }

        let mut response = serde_json::json!({
            "installed_cards": all_imported,
            "skipped_cards": all_skipped,
            "packages": packages,
            "source": source,
            "mode": "card_collection",
        });
        if !storage_warnings.is_empty() {
            response["storage_warnings"] = serde_json::json!(storage_warnings);
        }
        Ok(response.to_string())
    }

    /// Import bundled Cards from a package's `cards/` subdirectory.
    ///
    /// Called by `pkg_install` when a package contains a `cards/` dir.
    /// Returns imported card_ids (may be empty if all were skipped).
    pub(crate) fn import_pkg_bundled_cards(&self, pkg_name: &str, cards_dir: &Path) -> Vec<String> {
        match card::import_from_dir_with_store(&*self.card_store, cards_dir, pkg_name) {
            Ok((imported, _)) => imported,
            Err(e) => {
                tracing::warn!("Failed to import bundled cards for '{pkg_name}': {e}");
                Vec::new()
            }
        }
    }

    /// Read per-case sidecar rows (Tier 2) with `where` filtering and paging.
    pub fn card_samples(
        &self,
        card_id: &str,
        offset: usize,
        limit: Option<usize>,
        where_: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let where_parsed = match where_ {
            Some(v) => Some(card::parse_where(&v)?),
            None => None,
        };
        let q = card::SamplesQuery {
            offset,
            limit,
            where_: where_parsed,
        };
        let rows = self.card_store.read_samples(card_id, q)?;
        Ok(serde_json::Value::Array(rows).to_string())
    }

    /// Walk a Card's lineage tree via `metadata.prior_card_id`.
    pub fn card_lineage(
        &self,
        card_id: &str,
        direction: Option<&str>,
        depth: Option<usize>,
        include_stats: Option<bool>,
        relation_filter: Option<Vec<String>>,
    ) -> Result<String, String> {
        let dir = match direction {
            Some(s) => card::LineageDirection::parse(s)?,
            None => card::LineageDirection::Up,
        };
        let q = card::LineageQuery {
            card_id: card_id.to_string(),
            direction: dir,
            depth,
            include_stats: include_stats.unwrap_or(true),
            relation_filter,
        };
        match self.card_store.lineage(q)? {
            Some(res) => Ok(card::lineage_to_json(&res).to_string()),
            None => Err(format!("card '{card_id}' not found")),
        }
    }

    /// Backfill one subscriber (`sink` URI) with all cards from the
    /// primary store. Drift-safe: existing cards on the subscriber
    /// are skipped, never overwritten. Returns the
    /// [`card::SinkBackfillReport`] serialized as JSON for MCP
    /// transport.
    pub fn card_sink_backfill(&self, params: SinkBackfillParams) -> Result<String, String> {
        let report = self
            .card_store
            .card_sink_backfill(&params.sink, params.dry_run)?;
        serde_json::to_string(&report)
            .map_err(|e| format!("failed to serialize SinkBackfillReport: {e}"))
    }

    /// Load a Card + its samples sidecar and dispatch them to a Lua
    /// analyzer package.
    ///
    /// The host owns Card-schema parsing (Tier 1 body + Tier 2
    /// `samples.jsonl`) so the analyzer pkg gets a ready-to-use ctx
    /// shape. The pkg owns prompt construction + `alc.llm` + hint
    /// formatting.
    ///
    /// `pkg` defaults to [`DEFAULT_CARD_ANALYZE_PKG`] when omitted —
    /// an IF promise, not a bundled hard dependency. The call delegates
    /// to [`AppService::advice`], so all of `advice`'s machinery
    /// (auto-install bundled fallback, `start_and_tick`, response
    /// warning splicing) applies.
    ///
    /// ctx shape passed to the pkg's `M.run(ctx)`:
    /// ```jsonc
    /// {
    ///   "card_id": "<id>",
    ///   "card":    <full Card body, same shape as alc_card_get>,
    ///   "samples": [<sidecar rows, same shape as alc_card_samples>]
    /// }
    /// ```
    /// The pkg is responsible for filtering failures, building prompts,
    /// and shaping the result.
    pub async fn card_analyze(&self, card_id: &str, pkg: Option<String>) -> Result<String, String> {
        // Tier 1: Card body
        let card_value = match self.card_store.get(card_id)? {
            Some(v) => v,
            None => return Err(format!("card '{card_id}' not found")),
        };

        // Tier 2: samples sidecar (full read; analyzer pkg filters failures)
        let samples = self
            .card_store
            .read_samples(card_id, card::SamplesQuery::default())?;

        let mut opts = serde_json::Map::new();
        opts.insert(
            "card_id".into(),
            serde_json::Value::String(card_id.to_string()),
        );
        opts.insert("card".into(), card_value);
        opts.insert("samples".into(), serde_json::Value::Array(samples));

        let pkg_name = pkg.as_deref().unwrap_or(DEFAULT_CARD_ANALYZE_PKG);
        let raw = self
            .advice(pkg_name, None, Some(serde_json::Value::Object(opts)), None)
            .await?;

        // Post-process only the final `completed` envelope.  All other
        // statuses (`needs_response`, `error`, `cancelled`) pass through
        // unchanged so that the `alc_continue` round-trip is not broken.
        let mut envelope: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("card_analyze: response is not valid JSON: {e}"))?;

        if envelope.get("status").and_then(serde_json::Value::as_str) == Some("completed") {
            // Extract `result.result` (the pkg-set ctx field) and validate it
            // against the typed contract before promoting it to top-level.
            let inner = envelope
                .get_mut("result")
                .ok_or_else(|| {
                    "card_analyze: completed response missing top-level 'result' field".to_string()
                })?
                .get_mut("result")
                .ok_or_else(|| {
                    "card_analyze: pkg response missing 'result.result' field".to_string()
                })?
                .take();

            let typed: CardAnalyzeResult = serde_json::from_value(inner)
                .map_err(|e| format!("card_analyze: pkg returned invalid result shape: {e}"))?;

            envelope["result"] = serde_json::to_value(&typed).map_err(|e| {
                format!("card_analyze: failed to re-serialize CardAnalyzeResult: {e}")
            })?;
        }

        serde_json::to_string(&envelope)
            .map_err(|e| format!("card_analyze: failed to serialize response: {e}"))
    }

    /// Publish a Card to a hub repository.
    ///
    /// Validates `target_repo` as a URL, clones it to a staging directory,
    /// copies the card files, commits, and pushes.  On push success, calls
    /// `hub_reindex` and returns the outcome as a JSON string including
    /// `published_url`, `commit_hash`, and `reindex_status`.
    ///
    /// Push failures due to credential issues return a typed
    /// `CardPublishError::MissingCredentials` with actionable guidance.
    /// Reindex failures are reported in the response JSON, not as errors,
    /// so a successful push is never rolled back.
    pub async fn card_publish(
        &self,
        card_id: &str,
        target_repo: &str,
        commit_message: Option<&str>,
    ) -> Result<String, String> {
        self.card_publish_inner(card_id, target_repo, commit_message)
            .await
            .map_err(|e| e.to_string())
    }

    async fn card_publish_inner(
        &self,
        card_id: &str,
        target_repo: &str,
        commit_message: Option<&str>,
    ) -> Result<String, CardPublishError> {
        // 1. Validate target_repo is a URL (pkg slug not yet supported)
        if !is_supported_target(target_repo) {
            return Err(CardPublishError::InvalidTarget(format!(
                "{target_repo} — must be a URL (http/https/file/git@/ssh). \
                pkg slug resolution is not yet supported; see issue #1.",
            )));
        }

        // 2. Resolve card from store
        let card_value = self
            .card_store
            .get(card_id)
            .map_err(|e| CardPublishError::GitCommand {
                cmd: "card_store.get".into(),
                stderr: e,
            })?
            .ok_or_else(|| CardPublishError::CardNotFound(card_id.to_string()))?;

        // 3. Extract pkg name from card value
        let pkg_name = card_value
            .get("pkg")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // 4. Clone target_repo to staging
        let staging = tempfile::tempdir()?;
        let staging_str = staging
            .path()
            .to_str()
            .ok_or_else(|| {
                CardPublishError::InvalidTarget("staging path is not valid UTF-8".into())
            })?
            .to_string();

        if let Err((stderr, is_credential)) =
            run_git_command(&["clone", "--depth", "1", target_repo, &staging_str], None).await
        {
            if is_credential {
                let app_dir_path = self.log_config.app_dir().root().to_owned();
                let report = tokio::task::spawn_blocking(move || {
                    crate::service::gh_credentials::diagnose(&app_dir_path)
                })
                .await
                .map_err(|e| CardPublishError::GitCommand {
                    cmd: "spawn_blocking(diagnose)".into(),
                    stderr: e.to_string(),
                })?;
                let guidance = crate::service::gh_credentials::build_guidance(&report);
                return Err(CardPublishError::MissingCredentials { guidance });
            } else {
                return Err(CardPublishError::GitCommand {
                    cmd: "clone".into(),
                    stderr,
                });
            }
        }

        // 5. Copy card files into staging/cards/{pkg}/
        let dest_dir = staging.path().join("cards").join(&pkg_name);
        std::fs::create_dir_all(&dest_dir)?;

        // Collect card files: {card_id}.toml and optionally {card_id}.samples.jsonl
        // card_store.root() is already the cards dir (e.g. ~/.algocline/cards)
        let cards_root = self.card_store.root().join(&pkg_name);
        let card_toml = cards_root.join(format!("{card_id}.toml"));
        let card_samples = cards_root.join(format!("{card_id}.samples.jsonl"));

        if card_toml.exists() {
            std::fs::copy(&card_toml, dest_dir.join(format!("{card_id}.toml")))?;
        } else {
            return Err(CardPublishError::CardNotFound(card_id.to_string()));
        }
        if card_samples.exists() {
            std::fs::copy(
                &card_samples,
                dest_dir.join(format!("{card_id}.samples.jsonl")),
            )?;
        }

        // 6. git add
        run_git_command(&["add", "."], Some(staging.path()))
            .await
            .map_err(|(stderr, _)| CardPublishError::GitCommand {
                cmd: "add".into(),
                stderr,
            })?;

        // 7. git commit
        let msg = commit_message
            .map(String::from)
            .unwrap_or_else(|| format!("publish card {card_id}"));
        run_git_command(&["commit", "-m", &msg], Some(staging.path()))
            .await
            .map_err(|(stderr, _)| CardPublishError::GitCommand {
                cmd: "commit".into(),
                stderr,
            })?;

        // 8. git rev-parse HEAD — get commit hash
        let commit_hash = run_git_output(&["rev-parse", "HEAD"], Some(staging.path()))
            .await
            .map_err(|stderr| CardPublishError::GitCommand {
                cmd: "rev-parse HEAD".into(),
                stderr,
            })?
            .trim()
            .to_string();

        // 9. git push — detect credential failures here.
        //    diagnose() uses sync std::process::Command internally (logging.rs
        //    compatibility), so wrap in spawn_blocking to avoid blocking the
        //    tokio worker thread.
        if let Err((stderr, is_credential)) =
            run_git_command(&["push", "origin", "HEAD"], Some(staging.path())).await
        {
            if is_credential {
                let app_dir_path = self.log_config.app_dir().root().to_owned();
                let report = tokio::task::spawn_blocking(move || {
                    crate::service::gh_credentials::diagnose(&app_dir_path)
                })
                .await
                .map_err(|e| CardPublishError::GitCommand {
                    cmd: "spawn_blocking(diagnose)".into(),
                    stderr: e.to_string(),
                })?;
                let guidance = crate::service::gh_credentials::build_guidance(&report);
                return Err(CardPublishError::MissingCredentials { guidance });
            } else {
                return Err(CardPublishError::GitCommand {
                    cmd: "push".into(),
                    stderr,
                });
            }
        }

        // 10. hub_reindex — failure absorbed into reindex_status (not bubbled as Err)
        let reindex_status = match self.hub_reindex(None, None).await {
            Ok(out) => ReindexStatus {
                ok: true,
                output: Some(out),
                error: None,
            },
            Err(e) => ReindexStatus {
                ok: false,
                output: None,
                error: Some(e),
            },
        };

        // 11. Build response
        let outcome = CardPublishOutcome {
            published_url: target_repo.to_string(),
            commit_hash,
            reindex_status,
        };
        serde_json::to_string(&outcome).map_err(|e| CardPublishError::GitCommand {
            cmd: "serialize response".into(),
            stderr: e.to_string(),
        })
    }
}

// ─── Private response types ───────────────────────────────────────

#[derive(Debug, Serialize)]
struct CardPublishOutcome {
    published_url: String,
    commit_hash: String,
    reindex_status: ReindexStatus,
}

#[derive(Debug, Serialize)]
struct ReindexStatus {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ─── Private helpers ──────────────────────────────────────────────

/// Returns `true` when `target_repo` starts with a recognized URL scheme.
fn is_supported_target(target_repo: &str) -> bool {
    target_repo.starts_with("http://")
        || target_repo.starts_with("https://")
        || target_repo.starts_with("file://")
        || target_repo.starts_with("git@")
        || target_repo.starts_with("ssh://")
}

/// Returns `true` when `stderr` matches known credential-failure patterns.
fn detect_credential_error(stderr: &str) -> bool {
    let patterns = [
        "Permission denied (publickey)",
        "Authentication failed",
        "remote: Permission to",
        "could not read Username",
        "terminal prompts disabled",
        "gh auth login",
    ];
    patterns.iter().any(|p| stderr.contains(p))
}

/// Run a git command optionally in a working directory.
///
/// Returns `Ok(())` on success.
/// Returns `Err((stderr, is_credential_error))` on failure.
async fn run_git_command(args: &[&str], cwd: Option<&Path>) -> Result<(), (String, bool)> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(args).env("GIT_TERMINAL_PROMPT", "0");
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let output = cmd.output().await.map_err(|e| (e.to_string(), false))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let is_cred = detect_credential_error(&stderr);
        Err((stderr, is_cred))
    }
}

/// Run a git command in a working directory and capture stdout.
///
/// Returns `Ok(stdout)` on success, `Err(stderr)` on failure.
async fn run_git_output(args: &[&str], cwd: Option<&Path>) -> Result<String, String> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(args).env("GIT_TERMINAL_PROMPT", "0");
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let output = cmd.output().await.map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_credential_error_matches_publickey_denied() {
        assert!(detect_credential_error(
            "git@github.com: Permission denied (publickey).\nfatal: Could not read from remote repository."
        ));
    }

    #[test]
    fn detect_credential_error_matches_authentication_failed() {
        assert!(detect_credential_error(
            "remote: Authentication failed for 'https://github.com/user/repo.git'"
        ));
    }

    #[test]
    fn detect_credential_error_returns_false_for_unrelated_stderr() {
        assert!(!detect_credential_error(
            "fatal: pathspec 'cards/cot/foo.toml' did not match any files known to git"
        ));
    }

    #[test]
    fn is_supported_target_accepts_https() {
        assert!(is_supported_target("https://github.com/user/repo.git"));
    }

    #[test]
    fn is_supported_target_accepts_git_at() {
        assert!(is_supported_target("git@github.com:user/repo.git"));
    }

    #[test]
    fn is_supported_target_accepts_file_url() {
        assert!(is_supported_target("file:///tmp/bare-repo"));
    }

    #[test]
    fn is_supported_target_rejects_bare_slug() {
        assert!(!is_supported_target("cot"));
        assert!(!is_supported_target("my-pkg"));
    }
}
