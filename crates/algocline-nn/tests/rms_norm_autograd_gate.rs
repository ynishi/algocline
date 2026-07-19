//! Autograd gate for `candle_nn::RmsNorm` (fast path) and
//! `candle_nn::ops::rms_norm_slow` (slow path).
//!
//! Purpose: before wiring `RmsNorm` into the TinyLlama trainable
//! architecture (GH #10 Layer 1), verify whether candle-nn 0.11's
//! `RmsNorm::forward` (which delegates to `ops::rms_norm`) preserves
//! the gradient chain back to its `weight` parameter. GPT-2's
//! `LayerNorm` shipped a `CustomOp` autograd break in candle-nn 0.11
//! that the 0.46 release patched with an `apply_slow_layer_norm` shim
//! calling `ops::layer_norm_slow`. `rms_norm` is the same class of
//! `CustomOp`, so this test acts as the regression gate that says
//! "either the fast path is safe → use `RmsNorm` directly in Layer 1,
//! or the fast path is broken → route through `ops::rms_norm_slow`
//! via an `apply_slow_rms_norm` shim".
//!
//! Verified on 2026-07-20: the fast path is broken in candle-nn 0.11.
//! `RmsNorm::forward` produces a `GradStore` with no entry for the
//! `weight` Var — identical to the LayerNorm cliff patched in 0.46.
//! Layer 1 will therefore route RMSNorm through `ops::rms_norm_slow`
//! via an `apply_slow_rms_norm` shim (same shape as the existing
//! `apply_slow_layer_norm` in `arch::gpt2`).
//!
//! Both cases are asserted:
//!   1. `RmsNorm::forward` (fast path) — currently BROKEN. Marked
//!      `#[should_panic]`. If a future candle-nn bump restores the
//!      backward, this test flips to a real failure, telling us the
//!      shim can be dropped (mirrors GH #8 for the LayerNorm side).
//!   2. `ops::rms_norm_slow` — the fallback that must always work,
//!      confirming the slow-path shim would restore gradients.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

const DIM: usize = 8;
const EPS: f64 = 1e-5;

/// Build a random 3-D input `[batch, seq, dim]` on CPU with a fixed
/// shape. F32 keeps the gradient checks robust across platforms.
fn make_input(device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, (2, 4, DIM), device)
}

/// Sum of absolute values of a tensor as `f32`. Used to prove a
/// gradient is not the all-zero tensor (an autograd break can produce
/// either a missing entry in the `GradStore` or a zero tensor).
fn grad_magnitude(g: &Tensor) -> Result<f32> {
    g.abs()?.sum_all()?.to_scalar::<f32>()
}

/// Confirms that the candle-nn 0.11 fast path IS BROKEN. Uses
/// `should_panic` so the test passes today (documenting the drift)
/// and flips to a failure the moment a future candle-nn release
/// restores the backward — that failure is the signal to remove the
/// `apply_slow_rms_norm` shim.
#[test]
#[should_panic(expected = "fast-path autograd break")]
fn rms_norm_fast_path_is_currently_broken() {
    rms_norm_fast_path_backprops_to_weight().unwrap();
}

fn rms_norm_fast_path_backprops_to_weight() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rn = candle_nn::rms_norm(DIM, EPS, vb)?;

    let x = make_input(&device)?;
    let y = rn.forward(&x)?;
    let loss = y.sqr()?.mean_all()?;
    let grads = loss.backward()?;

    let vars = varmap.all_vars();
    assert!(
        !vars.is_empty(),
        "expected candle_nn::rms_norm to register at least one Var in the VarMap"
    );

    for var in &vars {
        let grad = grads.get(var.as_tensor()).unwrap_or_else(|| {
            panic!(
                "fast-path autograd break: weight Var has no entry in GradStore \
                 (this matches the LayerNorm CustomOp cliff patched in 0.46; \
                 route Layer 1 RMSNorm through ops::rms_norm_slow via a shim)"
            )
        });
        let mag = grad_magnitude(grad)?;
        assert!(
            mag > 0.0,
            "fast-path autograd break: weight grad is all-zero \
             (sum(|g|) = 0). Same remediation as the missing-entry case."
        );
    }

    Ok(())
}

#[test]
fn rms_norm_slow_path_backprops_to_weight() -> Result<()> {
    // Control: `ops::rms_norm_slow` is the pure-op decomposition
    // (`sqr / mean / sqrt / div / mul`). Every op has a proper
    // backward, so gradients must flow. If this test ever fails, the
    // slow-path fallback strategy itself is broken and Layer 1 needs a
    // hand-rolled decomposition (2026-07-20 verified path).
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let weight = vb.get_with_hints((DIM,), "weight", candle_nn::Init::Const(1.0))?;

    let x = make_input(&device)?;
    let y = candle_nn::ops::rms_norm_slow(&x, &weight, EPS as f32)?;
    let loss = y.sqr()?.mean_all()?;
    let grads = loss.backward()?;

    let vars = varmap.all_vars();
    assert_eq!(
        vars.len(),
        1,
        "expected exactly one Var (weight) in the VarMap"
    );
    let var = &vars[0];
    let grad = grads
        .get(var.as_tensor())
        .expect("slow-path autograd break: weight Var missing from GradStore");
    let mag = grad_magnitude(grad)?;
    assert!(
        mag > 0.0,
        "slow-path autograd break: weight grad is all-zero (sum(|g|) = 0)"
    );

    Ok(())
}
