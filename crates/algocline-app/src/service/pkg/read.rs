//! `pkg_read_init_lua` — read the `init.lua` source of an installed package.
//!
//! Searches variant (`alc.local.toml` path entries, project-root-scoped) and
//! then global (`~/.algocline/packages/<name>/init.lua`) scope in priority
//! order: variant wins.

use std::path::PathBuf;

use super::super::alc_toml;
use super::super::project::resolve_project_root;
use super::super::AppService;

impl AppService {
    /// Return the raw Lua source of `<name>/init.lua`.
    ///
    /// Search order (highest priority first):
    /// 1. Variant entries from `alc.local.toml` (worktree-scoped, gitignored).
    ///    Uses `resolve_project_root(None)` for auto-detection (same as `pkg_list`).
    /// 2. Global packages under `~/.algocline/packages/<name>/init.lua`.
    ///
    /// Returns `Err` when the package is not found in any scope or when an
    /// I/O error prevents reading the file.
    pub(crate) fn pkg_read_init_lua(&self, name: &str) -> Result<String, String> {
        // ── 1. Variant scope: alc.local.toml ──────────────────────────────
        //
        // Auto-detect project root the same way `pkg_list` does when called
        // without an explicit project_root. Missing or malformed alc.local.toml
        // is non-fatal — we fall through to global scope.
        if let Some(root) = resolve_project_root(None) {
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
                    // Non-fatal: malformed alc.local.toml is a warn, not a hard error.
                    // We continue to global scope so a broken local file doesn't
                    // shadow a healthy global package.
                    tracing::warn!("pkg_read_init_lua: skipping malformed alc.local.toml: {e}");
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
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::test_support::make_app_service_at;

    #[tokio::test]
    async fn read_global_pkg_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let pkg_dir = tmp.path().join("packages").join("mypkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(pkg_dir.join("init.lua"), "return {}").unwrap();

        let result = svc.pkg_read_init_lua("mypkg").unwrap();
        assert_eq!(result, "return {}");
    }

    #[tokio::test]
    async fn read_missing_pkg_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc.pkg_read_init_lua("nonexistent").unwrap_err();
        assert!(err.contains("pkg not found"), "got: {err}");
    }
}
