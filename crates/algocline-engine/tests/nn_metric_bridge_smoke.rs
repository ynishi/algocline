#![cfg(feature = "nn")]
//! Engine-level smoke test for the `alc.nn.metric.*` bridge (feature `nn`).
//!
//! One surface is left on this namespace — the cluster bootstrap — and
//! this test closes the bridge gap for it: the full engine registration
//! path (`bridge::register` → `register_nn_metric`) puts it on the `alc`
//! table of a real engine VM, and it honours its seed and refuses an
//! empty sample through Lua.
//!
//! The four distribution primitives that used to be tested here moved to
//! `alc.math` (mlua-mathlib owns them and their tests now).
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

/// A sample whose clusters all report the same value has nothing for
/// the resampler to move, so the interval collapses onto the point.
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
