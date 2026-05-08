//! `pkg_read_init_lua` — read the `init.lua` source of an installed package.
//!
//! Searches variant (`alc.local.toml` path entries, project-root-scoped) and
//! then global (`~/.algocline/packages/<name>/init.lua`) scope in priority
//! order: variant wins.

use std::path::{Path, PathBuf};

use super::super::alc_toml;
use super::super::AppService;

impl AppService {
    /// Return the raw Lua source of `<name>/init.lua`.
    ///
    /// Search order (highest priority first):
    /// 1. Variant entries from `alc.local.toml` (worktree-scoped, gitignored).
    ///    When `project_root` is `Some`, that path is used directly.
    ///    When `None`, falls back to `resolve_project_root(None)` (env / cwd walk).
    /// 2. Global packages under `~/.algocline/packages/<name>/init.lua`.
    ///
    /// Returns `Err` when the package is not found in any scope, when an
    /// I/O error prevents reading the file, or when `alc.local.toml` is malformed
    /// (corruption is an error, not a silent fallthrough).
    pub(crate) fn pkg_read_init_lua(
        &self,
        name: &str,
        project_root: Option<&Path>,
    ) -> Result<String, String> {
        // ── 1. Variant scope: alc.local.toml ──────────────────────────────
        //
        // Use the caller-supplied project_root when available; fall back to
        // self.resolve_root(None) (P > S > E > W) for MCP callers that do not
        // pass one. A missing file is non-fatal (fall through to global scope).
        // A malformed file is a hard error — corruption must reach the caller,
        // not be silently swallowed.
        let resolved_root = project_root
            .map(|p| p.to_path_buf())
            .or_else(|| self.resolve_root(None));
        if let Some(root) = resolved_root {
            match alc_toml::load_alc_local_toml(&root) {
                Ok(Some(local)) => {
                    for vp in alc_toml::resolve_local_variant_pkgs(&root, &local) {
                        if vp.name == name {
                            let init_lua = vp.pkg_dir.join("init.lua");
                            return std::fs::read_to_string(&init_lua).map_err(|e| {
                                format!(
                                    "pkg_read_init_lua: failed to read {}: {e}",
                                    init_lua.display()
                                )
                            });
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(format!(
                        "pkg_read_init_lua: malformed alc.local.toml at {}: {e}",
                        root.display()
                    ));
                }
            }
        }

        // ── 2. Global scope: ~/.algocline/packages/<name>/init.lua ─────────
        let global_init_lua: PathBuf = self
            .log_config
            .app_dir()
            .packages_dir()
            .join(name)
            .join("init.lua");

        match std::fs::metadata(&global_init_lua) {
            Ok(_) => std::fs::read_to_string(&global_init_lua).map_err(|e| {
                format!(
                    "pkg_read_init_lua: failed to read {}: {e}",
                    global_init_lua.display()
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(format!("pkg not found: {name}"))
            }
            Err(e) => Err(format!(
                "pkg_read_init_lua: I/O error for {}: {e}",
                global_init_lua.display()
            )),
        }
    }

    /// Resolve the per-pkg narrative file path declared via the
    /// `M.docs.narrative` SSOT spec key (#1778112139).
    ///
    /// Returns `Ok(Some(rel_path))` when the pkg declares
    /// `M.docs.narrative = "<path>"`. Returns `Ok(None)` when the
    /// declaration is absent, signalling the caller should fall
    /// back to the convention path (`narrative.md`). Returns `Err`
    /// only when the pkg cannot be loaded at all.
    ///
    /// `..` and leading `/` are rejected here as a path-traversal
    /// guard so callers can join the result against the pkg dir
    /// without re-validating.
    pub(crate) async fn pkg_resolve_narrative_path(
        &self,
        name: &str,
    ) -> Result<Option<String>, String> {
        if !is_safe_pkg_name(name) {
            return Err(format!(
                "pkg_resolve_narrative_path: invalid package name '{name}'"
            ));
        }
        let code = format!(
            r#"package.loaded["{name}"] = nil
local pkg = require("{name}")
if type(pkg.docs) == "table" and type(pkg.docs.narrative) == "string" then
    return pkg.docs.narrative
else
    return nil
end"#
        );
        let value =
            self.executor.eval_simple(code).await.map_err(|e| {
                format!("pkg_resolve_narrative_path: pkg '{name}' load failed: {e}")
            })?;
        match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(s) => {
                if s.contains("..") || s.starts_with('/') {
                    Err(format!(
                        "pkg_resolve_narrative_path: pkg '{name}' M.docs.narrative \
                         must be a pkg-dir-relative path without '..' or leading '/' (got '{s}')"
                    ))
                } else if s.is_empty() {
                    // Empty string is treated as "no declaration" rather than
                    // a valid path, so the convention fallback kicks in.
                    Ok(None)
                } else {
                    Ok(Some(s))
                }
            }
            other => Err(format!(
                "pkg_resolve_narrative_path: pkg '{name}' M.docs.narrative must be a string \
                 (got {other:?})"
            )),
        }
    }
}

/// Validate a Lua module identifier — alphanumeric + underscore only.
/// Mirrors `pkg/list.rs::is_safe_pkg_name` but kept module-local to
/// avoid creating a cross-module dependency just for this guard.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::test_support::make_app_service_at;

    #[tokio::test]
    async fn read_with_malformed_local_toml_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        // Write malformed alc.local.toml at the project root.
        std::fs::write(tmp.path().join("alc.local.toml"), "not valid toml ][[[").unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;
        // Pass the tempdir as project_root directly — no env var needed.
        let err = svc
            .pkg_read_init_lua("mypkg", Some(tmp.path()))
            .unwrap_err();
        assert!(
            err.contains("malformed alc.local.toml"),
            "expected malformed error, got: {err}"
        );
    }

    #[tokio::test]
    async fn read_global_pkg_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let pkg_dir = tmp.path().join("packages").join("mypkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let result = svc.pkg_read_init_lua("mypkg", None).unwrap();
        assert_eq!(result, "return {}");
    }

    #[tokio::test]
    async fn read_missing_pkg_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc.pkg_read_init_lua("nonexistent", None).unwrap_err();
        assert!(err.contains("pkg not found"), "got: {err}");
    }
}
