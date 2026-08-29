#![cfg(feature = "nn")]
//! Engine-level smoke test for the `alc.nn.metric.*` bridge (feature `nn`).
//!
//! The `algocline-nn::metric` unit tests cover the primitives themselves
//! (numeric identities + validation error paths). This test closes the
//! bridge gap: it confirms the full engine registration path
//! (`bridge::register` → `register_nn_metric`) places `alc.nn.metric` on
//! the `alc` table of a real engine VM, the four primitives round-trip
//! through Lua, `MetricError` variants surface as `LuaError::external`,
//! and the cluster bootstrap honours its seed and refuses an empty
//! sample.
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

// ─── Cluster bootstrap ────────────────────────────────────────────

/// Every cluster holding the same value leaves nothing for resampling
/// to vary, so the interval collapses onto the point estimate. That is
/// the one case whose bounds can be asserted exactly, which makes it
/// the check that the clusters reached the tally at all — a bridge that
/// dropped them would have no observations to average and would error
/// instead.
#[test]
fn bootstrap_ci_degenerates_to_a_point_when_every_cluster_agrees() {
    let lua = nn_vm();
    let (point, low, high, clusters): (f64, f64, f64, i64) = lua
        .load(
            r#"
            local ci = alc.nn.metric.bootstrap_ci(
                { { 1.0, 1.0 }, { 1.0 }, { 1.0, 1.0, 1.0 } },
                { draws = 64, seed = 7 })
            return ci.point, ci.low, ci.high, ci.clusters
            "#,
        )
        .eval()
        .expect("bootstrap_ci should evaluate");
    assert!((point - 1.0).abs() < 1e-12, "point: {point}");
    assert!((low - 1.0).abs() < 1e-12, "low: {low}");
    assert!((high - 1.0).abs() < 1e-12, "high: {high}");
    assert_eq!(clusters, 3);
}

/// The same seed over the same sample has to reproduce the interval
/// exactly — an interval that moved between two identical calls would
/// not be a measurement of anything. The point estimate is the sample
/// as walked, so it is the mean over every observation (`6/10` here)
/// and does not depend on the draws at all.
///
/// The bounds themselves are not asserted: they are order statistics of
/// the resample distribution and pinning them here would pin the
/// resampler's internals rather than this bridge.
#[test]
fn bootstrap_ci_is_reproducible_and_reports_its_inputs() {
    let lua = nn_vm();
    let (same, point, low, high, draws, seed): (bool, f64, f64, f64, i64, i64) = lua
        .load(
            r#"
            local sample = { { 1, 0, 1 }, { 0, 0 }, { 1, 1, 1 }, { 0, 1 } }
            local a = alc.nn.metric.bootstrap_ci(sample, { draws = 200, seed = 11 })
            local b = alc.nn.metric.bootstrap_ci(sample, { draws = 200, seed = 11 })
            return (a.low == b.low and a.high == b.high and a.point == b.point),
                   a.point, a.low, a.high, a.draws, a.seed
            "#,
        )
        .eval()
        .expect("bootstrap_ci should evaluate");
    assert!(same, "the same seed must reproduce the interval");
    assert!(
        (point - 0.6).abs() < 1e-12,
        "point must be the mean over every observation, got {point}"
    );
    assert!(
        low <= point && point <= high,
        "the point estimate must lie inside its own interval: {low} .. {point} .. {high}"
    );
    assert_eq!(draws, 200, "usable draws are reported back");
    assert_eq!(seed, 11, "the seed is carried back with the interval");
}

/// The seed is required rather than defaulted: an interval nothing can
/// reproduce would look exactly like one that can.
#[test]
fn bootstrap_ci_requires_a_seed() {
    let lua = nn_vm();
    let (ok, msg): (bool, String) = lua
        .load(
            r#"
            local ok, err = pcall(function()
                return alc.nn.metric.bootstrap_ci({ { 1, 0 }, { 1 } }, { draws = 16 })
            end)
            return ok, tostring(err)
            "#,
        )
        .eval()
        .expect("pcall wrapper should evaluate");
    assert!(!ok, "a bootstrap without a seed must fail");
    assert!(
        msg.contains("opts.seed"),
        "error should name the missing option, got: {msg}"
    );
}

/// An empty cluster list has no resampling unit at all.
#[test]
fn bootstrap_ci_refuses_an_empty_sample() {
    let lua = nn_vm();
    let (ok, msg): (bool, String) = lua
        .load(
            r#"
            local ok, err = pcall(function()
                return alc.nn.metric.bootstrap_ci({}, { seed = 1 })
            end)
            return ok, tostring(err)
            "#,
        )
        .eval()
        .expect("pcall wrapper should evaluate");
    assert!(!ok, "an empty sample must fail");
    assert!(
        msg.contains("non-empty"),
        "error should say the sample is empty, got: {msg}"
    );
}
