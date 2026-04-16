use std::path::{Path, PathBuf};

use mlua_pkg::{
    resolvers::{FsResolver, PrefixResolver},
    Registry, Resolver,
};

use crate::resolver_factory::make_resolver;

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
/// `resolve()` reads `{init_path}` from disk each time it fires. Within a
/// single Lua session, `package.loaded` caches the first result, so this
/// resolver runs at most once per name per session. Freshness across edits
/// comes from the fact that each `alc_run` builds a fresh session VM: the
/// disk read at resolve time guarantees the new session sees the current
/// `init.lua` content. Submodule lookups (`{name}.{sub}`) are delegated to
/// a separate `PrefixResolver(name, FsResolver(pkg_dir))`.
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
/// Delegates to the crate-level [`make_resolver`] factory so the sandbox
/// policy (default `SymlinkAwareSandbox`, strict under `ALC_PKG_STRICT=1`)
/// stays identical to the resolvers `Executor` and `alc.fork` construct.
fn make_submodule_resolver(pkg_dir: &Path) -> Option<FsResolver> {
    make_resolver(pkg_dir)
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
