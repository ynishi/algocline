//! Drift-check integration test for `types/alc_shapes.d.lua`.
//!
//! Verifies that the committed `types/alc_shapes.d.lua` matches the
//! output of `gen_alc_shapes_dlua_contents()`.  When the embedded
//! `alc_shapes` Lua source changes, this test fails until the file is
//! regenerated with:
//!
//! ```
//! cargo run -p algocline-app --example gen_alc_shapes_dlua
//! ```

use algocline_app::service::gendoc::alc_shapes_codegen::gen_alc_shapes_dlua_contents;

#[test]
fn alc_shapes_dlua_not_drifted() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // .../crates/algocline-app
    let workspace_root = std::path::Path::new(manifest_dir).join("../..");
    let on_disk_path = workspace_root.join("types/alc_shapes.d.lua");

    let on_disk = std::fs::read_to_string(&on_disk_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {}: {}. Run `cargo run -p algocline-app --example gen_alc_shapes_dlua` to generate it.",
            on_disk_path.display(),
            e
        )
    });

    let generated = gen_alc_shapes_dlua_contents().expect("gen_alc_shapes_dlua_contents() failed");

    assert_eq!(
        on_disk, generated,
        "types/alc_shapes.d.lua is out of date. Run:\n  cargo run -p algocline-app --example gen_alc_shapes_dlua"
    );
}

/// Smoke test: running the generator twice produces identical output
/// (verifies determinism of `luacats.gen`).
#[test]
fn alc_shapes_dlua_generation_is_deterministic() {
    let first =
        gen_alc_shapes_dlua_contents().expect("first gen_alc_shapes_dlua_contents() call failed");
    let second =
        gen_alc_shapes_dlua_contents().expect("second gen_alc_shapes_dlua_contents() call failed");
    assert_eq!(
        first, second,
        "gen_alc_shapes_dlua_contents() is non-deterministic: two runs produced different output"
    );
}
