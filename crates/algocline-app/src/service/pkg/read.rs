//! `pkg_read_init_lua` / `pkg_get_narrative_md` — read init.lua source and
//! render narrative markdown for an installed package.
//!
//! Searches variant (`alc.local.toml` path entries, project-root-scoped) and
//! then global (`~/.algocline/packages/<name>/init.lua`) scope in priority
//! order: variant wins.

use std::path::{Path, PathBuf};

use mlua::Lua;

use super::super::alc_toml;
use super::super::gendoc::register_preloads_pub;
use super::super::AppService;

/// Validate a Lua module identifier — alphanumeric + underscore only.
///
/// Used as a path-traversal / injection guard before interpolating a
/// package name into Lua code or file paths.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Inline Lua snippet used by `pkg_get_narrative_md`.
///
/// Requires `tools.docs.extract`, `tools.docs.projections`, and a
/// `package.preload` entry for `_alc_pkg_name` to be set before execution.
/// The `_alc_pkg_name` and `_alc_init_path` globals are injected
/// by the Rust caller.
const NARRATIVE_RENDER_LUA: &str = r#"
local extract     = require("tools.docs.extract")
local projections = require("tools.docs.projections")
local pi = extract.build_pkg_info(_alc_pkg_name, _alc_init_path, _alc_pkg_name .. "/init.lua")
return projections.narrative_md(pi)
"#;

/// Lua snippet that registers a pkg's init.lua content as a `package.preload`
/// entry. Used to inject a pkg's Lua source before `extract.build_pkg_info`
/// calls `require(pkg_name)`.
///
/// Variables consumed: `_alc_pkg_name` (string), `_alc_init_src` (string).
const INJECT_PKG_PRELOAD_LUA: &str = r#"
local src = _alc_init_src
local name = _alc_pkg_name
package.preload[name] = function()
    return load(src, "@" .. name .. "/init.lua")()
end
"#;

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

    /// Resolve the `init.lua` path for a package.
    ///
    /// Returns `Ok(Some(init_lua_path))` when found.
    /// Returns `Ok(None)` when the package is not installed.
    /// Returns `Err` only for I/O errors or malformed `alc.local.toml`.
    fn pkg_resolve_init_path(&self, name: &str) -> Result<Option<PathBuf>, String> {
        // ── 1. Variant scope: alc.local.toml ──────────────────────────────
        let resolved_root = self.resolve_root(None);
        if let Some(root) = resolved_root {
            match alc_toml::load_alc_local_toml(&root) {
                Ok(Some(local)) => {
                    for vp in alc_toml::resolve_local_variant_pkgs(&root, &local) {
                        if vp.name == name {
                            return Ok(Some(vp.pkg_dir.join("init.lua")));
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(format!(
                        "pkg_resolve_init_path: malformed alc.local.toml: {e}"
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
            Ok(_) => Ok(Some(global_init_lua)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!(
                "pkg_resolve_init_path: I/O error for {}: {e}",
                global_init_lua.display()
            )),
        }
    }

    /// Render the narrative markdown for a package on-the-fly.
    ///
    /// Extracts the init.lua docstring H2/H3 sections via the embedded
    /// gendoc pipeline (`extract.build_pkg_info` + `projections.narrative_md`)
    /// and returns the rendered markdown string.
    ///
    /// Returns `Ok(Some(markdown))` when the pkg is found and loadable.
    /// Returns `Ok(None)` when the pkg is not installed.
    /// Returns `Err(...)` when the gendoc pipeline fails (e.g. malformed Lua).
    pub(crate) async fn pkg_get_narrative_md(&self, name: &str) -> Result<Option<String>, String> {
        if !is_safe_pkg_name(name) {
            return Err(format!(
                "pkg_get_narrative_md: invalid package name '{name}'"
            ));
        }

        let init_lua_path = match self.pkg_resolve_init_path(name)? {
            Some(p) => p,
            None => return Ok(None),
        };

        // Read the init.lua source up-front in async context.
        let init_src = std::fs::read_to_string(&init_lua_path).map_err(|e| {
            format!(
                "pkg_get_narrative_md: failed to read {}: {e}",
                init_lua_path.display()
            )
        })?;

        let init_path_str = init_lua_path.to_string_lossy().into_owned();
        let name_owned = name.to_string();

        // Run the Lua pipeline synchronously on a blocking thread.
        // mlua::Lua is not Send, so it must be constructed inside the task.
        tokio::task::spawn_blocking(move || {
            let lua = Lua::new();

            // Register gendoc preloads (tools.docs.*, alc_shapes*).
            register_preloads_pub(&lua)?;

            // Inject the pkg's init.lua as a package.preload entry so that
            // `require(pkg_name)` resolves it without filesystem access.
            // This handles variant pkgs where the on-disk directory name may
            // differ from the Lua module name.
            lua.globals()
                .set("_alc_pkg_name", name_owned.clone())
                .map_err(|e| {
                    format!("pkg_get_narrative_md: globals set _alc_pkg_name failed: {e}")
                })?;
            lua.globals().set("_alc_init_src", init_src).map_err(|e| {
                format!("pkg_get_narrative_md: globals set _alc_init_src failed: {e}")
            })?;
            lua.load(INJECT_PKG_PRELOAD_LUA)
                .set_name("@embedded:pkg_get_narrative_md/preload")
                .exec()
                .map_err(|e| {
                    format!(
                        "pkg_get_narrative_md: pkg preload inject failed for '{name_owned}': {e}"
                    )
                })?;

            // Inject globals consumed by NARRATIVE_RENDER_LUA.
            lua.globals()
                .set("_alc_init_path", init_path_str.clone())
                .map_err(|e| {
                    format!("pkg_get_narrative_md: globals set _alc_init_path failed: {e}")
                })?;

            // Execute the render snippet.
            let markdown: String = lua
                .load(NARRATIVE_RENDER_LUA)
                .set_name("@embedded:pkg_get_narrative_md/render")
                .eval()
                .map_err(|e| {
                    format!("pkg_get_narrative_md: gendoc pipeline failed for '{name_owned}': {e}")
                })?;

            Ok(Some(markdown))
        })
        .await
        .map_err(|e| format!("pkg_get_narrative_md: blocking task panicked: {e}"))?
    }
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

    // ── pkg_get_narrative_md ──────────────────────────────────────────────

    #[tokio::test]
    async fn narrative_md_missing_pkg_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let result = svc.pkg_get_narrative_md("nonexistent").await.unwrap();
        assert!(
            result.is_none(),
            "expected None for missing pkg, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn narrative_md_invalid_name_returns_err() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        let err = svc.pkg_get_narrative_md("bad-name!").await.unwrap_err();
        assert!(
            err.contains("invalid package name"),
            "expected invalid name error, got: {err}"
        );
    }

    #[tokio::test]
    async fn narrative_md_renders_docstring_sections() {
        let tmp = tempfile::tempdir().unwrap();
        let svc = make_app_service_at(tmp.path().to_path_buf()).await;

        // Minimal pkg with H2 sections in the docstring.
        let pkg_dir = tmp.path().join("packages").join("mypkg");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(
            pkg_dir.join("init.lua"),
            r#"--- mypkg — a test package.
---
--- A short summary.
---
--- ## Algorithm
---
--- This is the algorithm section.
---
--- ## References
---
--- No references.

local M = {}
M.meta = { name = "mypkg", version = "0.1.0", category = "test", description = "A test pkg" }
return M
"#,
        )
        .unwrap();

        let result = svc.pkg_get_narrative_md("mypkg").await.unwrap();
        let markdown = result.expect("expected Some(markdown) for installed pkg");

        assert!(markdown.contains("mypkg"), "title missing: {markdown}");
        assert!(
            markdown.contains("Algorithm"),
            "Algorithm section missing: {markdown}"
        );
        assert!(
            markdown.contains("References"),
            "References section missing: {markdown}"
        );
    }
}
