use std::path::{Path, PathBuf};

use mlua_pkg::{
    resolvers::{FsResolver, PrefixResolver},
    sandbox::SymlinkAwareSandbox,
    Registry, Resolver,
};

/// A variant-scoped package pinned to an explicit `(name, pkg_dir)` mapping.
///
/// Sourced from `alc.local.toml` (worktree-scoped, gitignored). Unlike the
/// global `~/.algocline/packages/` layout — where `FsResolver(parent_dir)`
/// implicitly maps `require("X")` to `<parent>/X/init.lua` via directory
/// names — variant entries declare the require name explicitly so the on-disk
/// directory name does not have to match.
///
/// Resolution rules (built by [`register_variant_pkgs`]):
/// - `require("{name}")`        → `{pkg_dir}/init.lua`
/// - `require("{name}.{sub}")`  → `{pkg_dir}/{sub}.lua` or `{pkg_dir}/{sub}/init.lua`
#[derive(Clone, Debug)]
pub struct VariantPkg {
    pub name: String,
    pub pkg_dir: PathBuf,
}

impl VariantPkg {
    pub fn new(name: impl Into<String>, pkg_dir: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            pkg_dir: pkg_dir.into(),
        }
    }
}

/// Resolver for `require("{name}")` — the package root of a variant pkg.
///
/// Reads `{init_path}` lazily on each resolve so that an in-place edit of
/// the variant pkg's `init.lua` is visible on the next `alc_run` (each
/// `alc_run` builds a fresh session VM, but `package.loaded` cache only
/// applies within a single session). Submodule lookups (`{name}.{sub}`)
/// are delegated to a separate `PrefixResolver(name, FsResolver(pkg_dir))`.
struct VariantRootResolver {
    name: String,
    init_path: PathBuf,
}

impl Resolver for VariantRootResolver {
    fn resolve(&self, lua: &mlua::Lua, req: &str) -> Option<mlua::Result<mlua::Value>> {
        if req != self.name {
            return None;
        }
        let content = match std::fs::read_to_string(&self.init_path) {
            Ok(c) => c,
            Err(e) => {
                return Some(Err(mlua::Error::external(format!(
                    "variant pkg '{}': failed to read {}: {e}",
                    self.name,
                    self.init_path.display()
                ))));
            }
        };
        Some(
            lua.load(content)
                .set_name(self.init_path.display().to_string())
                .eval(),
        )
    }
}

/// Build a sandboxed `FsResolver` for a variant pkg's submodule lookups.
///
/// Mirrors the behaviour of `Executor`'s global path resolver factory:
/// `SymlinkAwareSandbox` by default so `alc_pkg_link` symlinks inside the
/// variant pkg are followed; `ALC_PKG_STRICT=1` falls back to plain
/// `FsResolver::new` (rejects all symlinks pointing outside the root).
fn make_submodule_resolver(pkg_dir: &Path) -> Option<FsResolver> {
    let strict = std::env::var("ALC_PKG_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if strict {
        FsResolver::new(pkg_dir).ok()
    } else {
        SymlinkAwareSandbox::new(pkg_dir)
            .ok()
            .map(FsResolver::with_sandbox)
    }
}

/// Register both the root resolver and the submodule prefix resolver for
/// each variant pkg. Variant pkgs should be inserted before any global
/// library resolvers so that `alc.local.toml` overrides win over
/// `~/.algocline/packages/`.
pub(crate) fn register_variant_pkgs(reg: &mut Registry, variant_pkgs: &[VariantPkg]) {
    for vp in variant_pkgs {
        let init_path = vp.pkg_dir.join("init.lua");
        reg.add(VariantRootResolver {
            name: vp.name.clone(),
            init_path,
        });
        match make_submodule_resolver(&vp.pkg_dir) {
            Some(inner) => {
                reg.add(PrefixResolver::new(vp.name.clone(), inner));
            }
            None => {
                tracing::warn!(
                    "variant pkg '{}': sandbox init failed for {}, submodules disabled",
                    vp.name,
                    vp.pkg_dir.display()
                );
            }
        }
    }
}
