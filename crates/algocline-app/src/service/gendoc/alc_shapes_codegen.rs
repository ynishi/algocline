//! Helper for generating `types/alc_shapes.d.lua` from the embedded
//! `alc_shapes` Lua source via an in-process `mlua` VM.
//!
//! Exposes a single `pub fn gen_alc_shapes_dlua_contents()` that is
//! consumed by the `gen_alc_shapes_dlua` example binary (manual regen)
//! and the `alc_shapes_dlua_drift` integration test (drift-check CI guard).
//!
//! The `pub` items here are re-exported from `lib.rs` via the visibility
//! chain opened in Subtask 2 (`pub(crate) mod gendoc` in `service/mod.rs`).

use mlua::Lua;

// File-local copies of the vendored alc_shapes Lua sources.
// These paths are relative to *this file's directory* (gendoc/), which
// matches the layout used in gendoc.rs for the EMBEDDED_TOOL_PRELOADS
// constants.
const LUA_ALC_SHAPES_T: &str = include_str!("alc_shapes/t.lua");
const LUA_ALC_SHAPES_REFLECT: &str = include_str!("alc_shapes/reflect.lua");
const LUA_ALC_SHAPES_CHECK: &str = include_str!("alc_shapes/check.lua");
const LUA_ALC_SHAPES_LUACATS: &str = include_str!("alc_shapes/luacats.lua");
const LUA_ALC_SHAPES_SPEC_RESOLVER: &str = include_str!("alc_shapes/spec_resolver.lua");
const LUA_ALC_SHAPES_INSTRUMENT: &str = include_str!("alc_shapes/instrument.lua");
const LUA_ALC_SHAPES_INIT: &str = include_str!("alc_shapes/init.lua");

/// Preload order for the alc_shapes module family.
///
/// Registration order mirrors `EMBEDDED_TOOL_PRELOADS` in `gendoc.rs`:
/// sub-modules before the top-level `alc_shapes` that `require`s them.
const ALC_SHAPES_PRELOADS: &[(&str, &str)] = &[
    ("alc_shapes.t", LUA_ALC_SHAPES_T),
    ("alc_shapes.reflect", LUA_ALC_SHAPES_REFLECT),
    ("alc_shapes.check", LUA_ALC_SHAPES_CHECK),
    ("alc_shapes.luacats", LUA_ALC_SHAPES_LUACATS),
    ("alc_shapes.spec_resolver", LUA_ALC_SHAPES_SPEC_RESOLVER),
    ("alc_shapes.instrument", LUA_ALC_SHAPES_INSTRUMENT),
    ("alc_shapes", LUA_ALC_SHAPES_INIT),
];

/// Generate the full contents of `types/alc_shapes.d.lua` by running
/// `alc_shapes.LuaCats.gen(S)` in a fresh `mlua` Lua 5.4 VM.
///
/// The output always ends with a newline (guaranteed by `luacats.gen`
/// at `alc_shapes/luacats.lua:227`).
///
/// # Errors
///
/// Returns `Err` (propagated via `?`) if the Lua VM fails to initialise,
/// any `require` fails, or `gen` raises a Lua error.  No warnings are
/// swallowed — all error paths surface as `anyhow::Error`.
pub fn gen_alc_shapes_dlua_contents() -> anyhow::Result<String> {
    let lua = Lua::new();

    // Register all alc_shapes modules onto package.preload so that
    // `require("alc_shapes")` and its sub-modules resolve from the
    // embedded sources without touching the filesystem.
    {
        let package: mlua::Table = lua.globals().get("package")?;
        let preload: mlua::Table = package.get("preload")?;
        for (mod_name, src) in ALC_SHAPES_PRELOADS.iter().copied() {
            let chunk_name = format!("@embedded:gendoc/{mod_name}.lua");
            let loader = lua.create_function(move |lua, ()| {
                lua.load(src)
                    .set_name(chunk_name.clone())
                    .eval::<mlua::Value>()
            })?;
            preload.set(mod_name, loader)?;
        }
    }

    // Load the top-level module and invoke `LuaCats.gen(shapes)`.
    let shapes: mlua::Table = lua.load(r#"return require("alc_shapes")"#).eval()?;
    let luacats: mlua::Table = shapes.get("LuaCats")?;
    let gen: mlua::Function = luacats.get("gen")?;
    // Pass `shapes` as the first argument; `class_prefix` is nil → default "AlcResult".
    let contents: String = gen.call(shapes.clone())?;
    Ok(contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: `gen_alc_shapes_dlua_contents()` succeeds and the
    /// output ends with a newline (guaranteed by `luacats.gen:227`).
    #[test]
    fn gen_returns_nonempty_with_trailing_newline() {
        let contents = gen_alc_shapes_dlua_contents().expect("generation failed");
        assert!(!contents.is_empty(), "generated output should not be empty");
        assert!(
            contents.ends_with('\n'),
            "generated output should end with newline"
        );
    }

    /// Smoke test: two consecutive calls produce identical output
    /// (verifies `luacats.gen` determinism within the same process).
    #[test]
    fn gen_is_deterministic() {
        let first = gen_alc_shapes_dlua_contents().expect("first call failed");
        let second = gen_alc_shapes_dlua_contents().expect("second call failed");
        assert_eq!(first, second, "generation must be deterministic");
    }

    /// When the `ALC_REGENERATE` environment variable is set to `1`, write
    /// the generated output to `types/alc_shapes.d.lua` in the workspace
    /// root.  This is equivalent to running the `gen_alc_shapes_dlua`
    /// example and is provided as a fallback for ST1 where the example
    /// binary cannot yet be built (visibility not yet exported via lib.rs).
    #[test]
    fn regenerate_if_env_set() {
        if std::env::var("ALC_REGENERATE").as_deref() != Ok("1") {
            return;
        }
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir).join("../..");
        let out_path = workspace_root.join("types/alc_shapes.d.lua");
        let contents = gen_alc_shapes_dlua_contents().expect("generation failed");
        std::fs::write(&out_path, &contents).expect("failed to write types/alc_shapes.d.lua");
        println!("Written {} bytes to {}", contents.len(), out_path.display());
    }
}
