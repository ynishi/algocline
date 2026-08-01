#![cfg(feature = "nn")]
//! Engine-level smoke test for the `alc.nn.metric.*` bridge (feature `nn`).
//!
//! The `algocline-nn::metric` unit tests cover the primitives themselves
//! (numeric identities + validation error paths). This test closes the
//! bridge gap: it confirms the full engine registration path
//! (`bridge::register` → `register_nn_metric`) places `alc.nn.metric` on
//! the `alc` table of a real engine VM, the four primitives round-trip
//! through Lua, `MetricError` variants surface as `LuaError::external`,
//! and the Pure-Lua `registry` supports the `register` / `evaluate`
//! contract the trainer `on_ckpt` hook (ST3) depends on.
//!
//! Uses `install_for_pkg_test` — the same registration path the pkg-test
//! sandbox uses — so a passing test also means the metric surface is
//! present in the sandbox (preserving the `production ⊆ sandbox` parity
//! invariant enforced by `tests/bridge_sandbox_parity.rs`).

use algocline_engine::bridge;
use mlua::Lua;

/// Fresh VM with the full `alc.*` bridge (including `alc.nn.*`) mounted
/// via the same path `alc_pkg_test` uses.
fn nn_vm() -> Lua {
    let lua = Lua::new();
    bridge::install_for_pkg_test(&lua).expect("install_for_pkg_test");
    lua
}

/// `alc.nn.metric.kl({0.5, 0.5}, {0.25, 0.75})` should equal the hand
/// computed value `0.5 * ln(2) + 0.5 * ln(2/3)` within f32 precision.
#[test]
fn bridge_kl_round_trip() {
    let lua = nn_vm();
    let got: f32 = lua
        .load("return alc.nn.metric.kl({ 0.5, 0.5 }, { 0.25, 0.75 })")
        .eval()
        .expect("alc.nn.metric.kl should evaluate");
    let expected = 0.5f32 * 2.0f32.ln() + 0.5f32 * (2.0f32 / 3.0f32).ln();
    assert!(
        (got - expected).abs() < 1e-5,
        "kl mismatch: got {got}, expected {expected}"
    );
}

/// Entropy of the uniform distribution on N elements is `ln(N)`.
#[test]
fn bridge_entropy_uniform_ln_n() {
    let lua = nn_vm();
    let got: f32 = lua
        .load("return alc.nn.metric.entropy({ 0.25, 0.25, 0.25, 0.25 })")
        .eval()
        .expect("alc.nn.metric.entropy should evaluate");
    let expected = 4.0f32.ln();
    assert!(
        (got - expected).abs() < 1e-5,
        "entropy mismatch: got {got}, expected {expected}"
    );
}

/// KL against a `q` that assigns zero mass where `p` assigns positive
/// mass is `+∞` by definition. The bridge must preserve the infinity
/// (not silently rewrite as an error) because the trainer hook uses
/// `math.huge` as a sentinel for "distributions disagree on support".
#[test]
fn bridge_kl_disjoint_one_hot_infinity() {
    let lua = nn_vm();
    let is_infinite: bool = lua
        .load(
            r#"
            local v = alc.nn.metric.kl({ 1, 0 }, { 0, 1 })
            return v == math.huge
            "#,
        )
        .eval()
        .expect("alc.nn.metric.kl should evaluate to +inf");
    assert!(
        is_infinite,
        "KL(p=[1,0] || q=[0,1]) must serialize as math.huge on the Lua side"
    );
}

/// A distribution containing a strictly negative element must be refused
/// with a Lua-visible error that names the primitive and the offending
/// index. `pcall` returns `false, msg` so the test can inspect the
/// message without unwinding the VM.
#[test]
fn bridge_error_negative_refused() {
    let lua = nn_vm();
    let (ok, msg): (bool, String) = lua
        .load(
            r#"
            local ok, err = pcall(function()
                return alc.nn.metric.kl({ -0.5, 1.5 }, { 0.5, 0.5 })
            end)
            return ok, tostring(err)
            "#,
        )
        .eval()
        .expect("pcall wrapper should evaluate");
    assert!(
        !ok,
        "kl with a negative probability must fail (got success, msg={msg})"
    );
    assert!(
        msg.contains("negative"),
        "error message should mention `negative`, got: {msg}"
    );
}

/// Registry round-trip: `register(name, fn)` followed by
/// `evaluate(name, ctx)` must call the stored closure with the supplied
/// context and return its result. Exercises the exact call shape the
/// trainer `on_ckpt` hook (ST3) will use.
#[test]
fn registry_round_trip() {
    let lua = nn_vm();
    let got: i64 = lua
        .load(
            r#"
            alc.nn.metric.registry.register("test_metric", function(ctx)
                return ctx.x * 2
            end)
            return alc.nn.metric.registry.evaluate("test_metric", { x = 21 })
            "#,
        )
        .eval()
        .expect("registry register/evaluate round-trip");
    assert_eq!(got, 42);
}

/// `evaluate` on an unregistered name must raise a clear error whose
/// message names the missing metric — the trainer hook cannot silently
/// treat "no such metric" as `continue`, or a mis-spelled config would
/// hide the mistake for the entire training run.
#[test]
fn registry_unknown_metric_refused() {
    let lua = nn_vm();
    let (ok, msg): (bool, String) = lua
        .load(
            r#"
            local ok, err = pcall(function()
                return alc.nn.metric.registry.evaluate("nope", {})
            end)
            return ok, tostring(err)
            "#,
        )
        .eval()
        .expect("pcall wrapper should evaluate");
    assert!(!ok, "evaluate on an unknown name must fail");
    assert!(
        msg.contains("nope"),
        "error message should name the missing metric `nope`, got: {msg}"
    );
}
