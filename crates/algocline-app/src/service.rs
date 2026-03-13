use std::path::{Path, PathBuf};
use std::sync::Arc;

use algocline_core::QueryId;
use algocline_engine::{Executor, SessionRegistry};

// ─── Helpers ────────────────────────────────────────────────────

/// Recursively copy a directory tree (follows symlinks).
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        // Use metadata() (follows symlinks) instead of file_type() (does not)
        let meta = entry.metadata()?;
        let dest_path = dst.join(entry.file_name());
        if meta.is_dir() {
            copy_dir(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

// ─── Parameter types (MCP-independent) ──────────────────────────

/// A single query response in a batch feed.
#[derive(Debug)]
pub struct QueryResponse {
    /// Query ID (e.g. "q-0", "q-1").
    pub query_id: String,
    /// The host LLM's response for this query.
    pub response: String,
}

// ─── Code resolution ────────────────────────────────────────────

pub(crate) fn resolve_code(
    code: Option<String>,
    code_file: Option<String>,
) -> Result<String, String> {
    match (code, code_file) {
        (Some(c), None) => Ok(c),
        (None, Some(path)) => std::fs::read_to_string(Path::new(&path))
            .map_err(|e| format!("Failed to read {path}: {e}")),
        (Some(_), Some(_)) => Err("Provide either `code` or `code_file`, not both.".into()),
        (None, None) => Err("Either `code` or `code_file` must be provided.".into()),
    }
}

/// Build Lua code that loads a package by name and calls `pkg.run(ctx)`.
///
/// # Security: `name` is not sanitized
///
/// `name` is interpolated directly into a Lua `require()` call without
/// sanitization. This is intentional in the current architecture:
///
/// - algocline is a **local development/execution tool** that runs Lua in
///   the user's own environment via mlua (not a multi-tenant service).
/// - The same caller has access to `alc_run`, which executes **arbitrary
///   Lua code**. Sanitizing `name` here would not reduce the attack surface.
/// - The MCP trust boundary lies at the **host/client** level — the host
///   decides whether to invoke `alc_advice` at all.
///
/// If algocline is extended to a shared backend (e.g. a package registry
/// server accepting untrusted strategy names), `name` **must** be validated
/// (allowlist of `[a-zA-Z0-9_-]` or equivalent) before interpolation.
///
/// References:
/// - [MCP Security Best Practices — Local MCP Server Compromise](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
/// - [OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html)
pub(crate) fn make_require_code(name: &str) -> String {
    format!(
        r#"local pkg = require("{name}")
return pkg.run(ctx)"#
    )
}

pub(crate) fn packages_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    Ok(home.join(".algocline").join("packages"))
}

// ─── Application Service ────────────────────────────────────────

#[derive(Clone)]
pub struct AppService {
    executor: Arc<Executor>,
    registry: Arc<SessionRegistry>,
}

impl AppService {
    pub fn new(executor: Arc<Executor>) -> Self {
        Self {
            executor,
            registry: Arc::new(SessionRegistry::new()),
        }
    }

    /// Execute Lua code with optional JSON context.
    pub async fn run(
        &self,
        code: Option<String>,
        code_file: Option<String>,
        ctx: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let code = resolve_code(code, code_file)?;
        let ctx = ctx.unwrap_or(serde_json::Value::Null);
        self.start_and_tick(code, ctx).await
    }

    /// Apply a built-in strategy to a task.
    pub async fn advice(
        &self,
        strategy: &str,
        task: String,
        opts: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let code = make_require_code(strategy);

        let mut ctx_map = match opts {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        ctx_map.insert("task".into(), serde_json::Value::String(task));
        let ctx = serde_json::Value::Object(ctx_map);

        self.start_and_tick(code, ctx).await
    }

    /// Continue a paused execution — batch feed.
    pub async fn continue_batch(
        &self,
        session_id: &str,
        responses: Vec<QueryResponse>,
    ) -> Result<String, String> {
        let mut last_result = None;
        for qr in responses {
            let qid = QueryId::parse(&qr.query_id);
            let result = self
                .registry
                .feed_response(session_id, &qid, qr.response)
                .await
                .map_err(|e| format!("Continue failed: {e}"))?;
            last_result = Some(result);
        }
        let result = last_result.ok_or("Empty responses array")?;
        Ok(result.to_json(session_id).to_string())
    }

    /// Continue a paused execution — single response (with optional query_id).
    pub async fn continue_single(
        &self,
        session_id: &str,
        response: String,
        query_id: Option<&str>,
    ) -> Result<String, String> {
        let query_id = match query_id {
            Some(qid) => QueryId::parse(qid),
            None => QueryId::single(),
        };

        let result = self
            .registry
            .feed_response(session_id, &query_id, response)
            .await
            .map_err(|e| format!("Continue failed: {e}"))?;

        Ok(result.to_json(session_id).to_string())
    }

    // ─── Package Management ─────────────────────────────────────

    /// List installed packages with metadata.
    pub async fn pkg_list(&self) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        if !pkg_dir.is_dir() {
            return Ok(serde_json::json!({ "packages": [] }).to_string());
        }

        let mut packages = Vec::new();
        let entries =
            std::fs::read_dir(&pkg_dir).map_err(|e| format!("Failed to read packages dir: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let init_lua = path.join("init.lua");
            if !init_lua.exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let code = format!(
                r#"local pkg = require("{name}")
return pkg.meta or {{ name = "{name}" }}"#
            );
            match self.executor.eval_simple(code).await {
                Ok(meta) => packages.push(meta),
                Err(_) => {
                    packages
                        .push(serde_json::json!({ "name": name, "error": "failed to load meta" }));
                }
            }
        }

        Ok(serde_json::json!({ "packages": packages }).to_string())
    }

    /// Install a package from a Git URL or local path.
    pub async fn pkg_install(&self, url: String, name: Option<String>) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let _ = std::fs::create_dir_all(&pkg_dir);

        // Normalize URL: add https:// only for bare domain-style URLs
        let git_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
            || url.starts_with("git@")
            || url.starts_with('/')
        {
            url.clone()
        } else {
            format!("https://{url}")
        };

        // Clone to temp directory first to detect single vs collection
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

        // Remove .git dir from staging
        let _ = std::fs::remove_dir_all(staging.path().join(".git"));

        // Detect: single package (init.lua at root) vs collection (subdirs with init.lua)
        if staging.path().join("init.lua").exists() {
            // Single package mode
            let name = name.unwrap_or_else(|| {
                url.trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .trim_end_matches(".git")
                    .to_string()
            });

            let dest = pkg_dir.join(&name);
            if dest.exists() {
                return Err(format!(
                    "Package '{name}' already exists at {}. Remove it first.",
                    dest.display()
                ));
            }

            copy_dir(staging.path(), &dest).map_err(|e| format!("Failed to copy package: {e}"))?;

            Ok(serde_json::json!({
                "installed": [name],
                "mode": "single",
            })
            .to_string())
        } else {
            // Collection mode: scan for subdirs containing init.lua
            if name.is_some() {
                // name parameter is only meaningful for single-package repos
                return Err(
                    "The 'name' parameter is only supported for single-package repos (init.lua at root). \
                     This repository is a collection (subdirs with init.lua)."
                        .to_string(),
                );
            }

            let mut installed = Vec::new();
            let mut skipped = Vec::new();

            let entries = std::fs::read_dir(staging.path())
                .map_err(|e| format!("Failed to read staging dir: {e}"))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("init.lua").exists() {
                    continue;
                }
                let pkg_name = entry.file_name().to_string_lossy().to_string();
                let dest = pkg_dir.join(&pkg_name);
                if dest.exists() {
                    skipped.push(pkg_name);
                    continue;
                }
                copy_dir(&path, &dest)
                    .map_err(|e| format!("Failed to copy package '{pkg_name}': {e}"))?;
                installed.push(pkg_name);
            }

            if installed.is_empty() && skipped.is_empty() {
                return Err(
                    "No packages found. Expected init.lua at root (single) or */init.lua (collection)."
                        .to_string(),
                );
            }

            Ok(serde_json::json!({
                "installed": installed,
                "skipped": skipped,
                "mode": "collection",
            })
            .to_string())
        }
    }

    /// Remove an installed package.
    pub async fn pkg_remove(&self, name: &str) -> Result<String, String> {
        let pkg_dir = packages_dir()?;
        let dest = pkg_dir.join(name);

        if !dest.exists() {
            return Err(format!("Package '{name}' not found"));
        }

        // Safety: only remove within ~/.algocline/packages/
        let canonical = dest
            .canonicalize()
            .map_err(|e| format!("Path error: {e}"))?;
        let pkg_canonical = pkg_dir
            .canonicalize()
            .map_err(|e| format!("Path error: {e}"))?;
        if !canonical.starts_with(&pkg_canonical) {
            return Err("Path traversal detected".to_string());
        }

        std::fs::remove_dir_all(&dest).map_err(|e| format!("Failed to remove '{name}': {e}"))?;

        Ok(serde_json::json!({ "removed": name }).to_string())
    }

    // ─── Internal ───────────────────────────────────────────────

    async fn start_and_tick(&self, code: String, ctx: serde_json::Value) -> Result<String, String> {
        let session = self.executor.start_session(code, ctx).await?;
        let (session_id, result) = self
            .registry
            .start_execution(session)
            .await
            .map_err(|e| format!("Execution failed: {e}"))?;
        Ok(result.to_json(&session_id).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ─── resolve_code tests ───

    #[test]
    fn resolve_code_inline() {
        let result = resolve_code(Some("return 1".into()), None);
        assert_eq!(result.unwrap(), "return 1");
    }

    #[test]
    fn resolve_code_from_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "return 42").unwrap();

        let result = resolve_code(None, Some(tmp.path().to_string_lossy().into()));
        assert_eq!(result.unwrap(), "return 42");
    }

    #[test]
    fn resolve_code_both_provided_error() {
        let result = resolve_code(Some("code".into()), Some("file.lua".into()));
        let err = result.unwrap_err();
        assert!(err.contains("not both"), "error: {err}");
    }

    #[test]
    fn resolve_code_neither_provided_error() {
        let result = resolve_code(None, None);
        let err = result.unwrap_err();
        assert!(err.contains("must be provided"), "error: {err}");
    }

    #[test]
    fn resolve_code_nonexistent_file_error() {
        let result = resolve_code(
            None,
            Some("/tmp/algocline_nonexistent_test_file.lua".into()),
        );
        assert!(result.is_err());
    }

    // ─── make_require_code tests ───

    #[test]
    fn make_require_code_basic() {
        let code = make_require_code("ucb");
        assert!(code.contains(r#"require("ucb")"#), "code: {code}");
        assert!(code.contains("pkg.run(ctx)"), "code: {code}");
    }

    #[test]
    fn make_require_code_different_names() {
        for name in &["panel", "cot", "sc", "cove", "reflect", "calibrate"] {
            let code = make_require_code(name);
            assert!(
                code.contains(&format!(r#"require("{name}")"#)),
                "code for {name}: {code}"
            );
        }
    }

    // ─── packages_dir tests ───

    #[test]
    fn packages_dir_ends_with_expected_path() {
        let dir = packages_dir().unwrap();
        assert!(
            dir.ends_with(".algocline/packages"),
            "dir: {}",
            dir.display()
        );
    }
}
