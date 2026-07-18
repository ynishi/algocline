#![cfg(feature = "nn")]
//! Engine-level smoke test for the `alc.nn` bridge (feature `nn`).
//!
//! The `algocline-nn` crate tests exercise the candle wrapper in isolation. This
//! test closes the remaining gap: it confirms the full engine registration path
//! (`bridge::register` -> `register_nn`) actually places `alc.nn` on the `alc`
//! table of a real engine VM and that a tensor op round-trips through Lua.
//!
//! It installs via `install_for_pkg_test`, the same path the pkg-test sandbox
//! uses, so with `nn` enabled the `alc.nn` surface is also present in the sandbox
//! (keeping the `production ⊆ sandbox` parity invariant intact).
//!
//! Only compiled with `--features nn`; the default build has no `alc.nn` and no
//! candle link.

use algocline_engine::bridge;
use mlua::Lua;

fn nn_vm() -> Lua {
    let lua = Lua::new();
    bridge::install_for_pkg_test(&lua).expect("install_for_pkg_test");
    lua
}

#[test]
fn alc_nn_tensor_add_roundtrips_through_engine_bridge() {
    let lua = nn_vm();
    let out: Vec<f32> = lua
        .load(
            r#"
            local a = alc.nn.tensor({ 1, 2, 3 })
            local b = alc.nn.tensor({ 10, 20, 30 })
            return a:add(b):to_vec()
        "#,
        )
        .eval()
        .expect("alc.nn tensor add roundtrip");
    assert_eq!(out, vec![11.0, 22.0, 33.0]);
}

#[test]
fn alc_nn_tensor_dims_reachable_through_engine_bridge() {
    let lua = nn_vm();
    let dims: Vec<usize> = lua
        .load("return alc.nn.tensor({ 1, 2, 3, 4 }):dims()")
        .eval()
        .expect("alc.nn.tensor(...):dims()");
    assert_eq!(dims, vec![4]);
}
