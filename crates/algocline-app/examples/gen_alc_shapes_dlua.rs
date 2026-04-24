//! Regenerate `types/alc_shapes.d.lua` from the vendored `alc_shapes`
//! Lua source.
//!
//! Run with:
//! ```
//! cargo run -p algocline-app --example gen_alc_shapes_dlua
//! ```
//!
//! The output file path is `<workspace-root>/types/alc_shapes.d.lua`,
//! resolved relative to this crate's `CARGO_MANIFEST_DIR`.

use algocline_app::service::gendoc::alc_shapes_codegen::gen_alc_shapes_dlua_contents;

fn main() -> anyhow::Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // .../crates/algocline-app
    let workspace_root = std::path::Path::new(manifest_dir)
        .join("../..")
        .canonicalize()?;
    let out_path = workspace_root.join("types/alc_shapes.d.lua");

    let contents = gen_alc_shapes_dlua_contents()?;
    std::fs::write(&out_path, &contents)?;

    println!("Written {} bytes to {}", contents.len(), out_path.display());
    Ok(())
}
