//! Autograd gate for `candle_nn::candle_nn::rotary_emb::rope` (fast path) and
//! `candle_nn::candle_nn::rotary_emb::rope_slow` (slow path).
//!
//! Purpose: sibling of `rms_norm_autograd_gate`, this test decides
//! whether Layer 1 RoPE for TinyLlama (GH #10) can use candle-nn's
//! fast-path helper or must route through the slow-path fallback.
//!
//! Verified on 2026-07-20: the fast path is broken by design in
//! candle-nn 0.11. `candle_nn::rotary_emb::rope` calls `apply_op3_no_bwd` (see
//! candle-nn 0.11 `src/rotary_emb.rs`:583), which explicitly declares
//! no backward for the `RotaryEmb` `CustomOp3`. Any `Var` that would
//! reach RoPE loses its gradient chain. Layer 1 will therefore route
//! RoPE through `candle_nn::rotary_emb::rope_slow`, a pure-op decomposition
//! (`broadcast_mul` + `rotate_half`) that ships in the same crate —
//! no hand-rolled implementation required.
//!
//! Both cases are asserted:
//!   1. `candle_nn::rotary_emb::rope` (fast path) — currently BROKEN by design.
//!      Marked `#[should_panic]`. If a future candle-nn release
//!      wires a real backward for `RotaryEmb`, this test flips to a
//!      real failure, telling us the shim can be dropped (parallel
//!      to GH #8 for the LayerNorm side).
//!   2. `candle_nn::rotary_emb::rope_slow` — the fallback that must always work.
//!
//! Shape follows the candle-nn contract for `rope`:
//!   - `xs`   : `[batch, n_head, seq_len, head_dim]`, contiguous
//!   - `cos`  : `[seq_max, head_dim / 2]`, contiguous
//!   - `sin`  : `[seq_max, head_dim / 2]`, contiguous
//!
//! `head_dim` must be even; `cos_n_embd * 2 == head_dim`.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

const BATCH: usize = 2;
const HEADS: usize = 4;
const SEQ: usize = 8;
const HEAD_DIM: usize = 16;

/// Build a canonical `[seq_max, head_dim/2]` cos/sin cache. Values
/// are precomputed constants (no gradient), so any reasonable float
/// tensor works for the autograd probe.
fn make_cos_sin(device: &Device) -> Result<(Tensor, Tensor)> {
    let cos = Tensor::randn(0f32, 1f32, (SEQ, HEAD_DIM / 2), device)?.contiguous()?;
    let sin = Tensor::randn(0f32, 1f32, (SEQ, HEAD_DIM / 2), device)?.contiguous()?;
    Ok((cos, sin))
}

/// Build a random 4-D input `[batch, heads, seq, head_dim]`.
fn make_input(device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, (BATCH, HEADS, SEQ, HEAD_DIM), device)?.contiguous()
}

fn grad_magnitude(g: &Tensor) -> Result<f32> {
    g.abs()?.sum_all()?.to_scalar::<f32>()
}

/// Confirms that the candle-nn 0.11 fast path IS BROKEN (by design —
/// `apply_op3_no_bwd`). Uses `should_panic` so the test passes today
/// (documenting the drift) and flips to a real failure the moment a
/// future candle-nn release wires a backward for `RotaryEmb`.
#[test]
#[should_panic(expected = "fast-path autograd break")]
fn rope_fast_path_is_currently_broken() {
    rope_fast_path_backprops_to_scale_weight().unwrap();
}

/// RoPE has no trainable parameter of its own, so we probe the
/// gradient chain by scaling the input with a trainable `weight`
/// broadcast along `head_dim`. If the RoPE op preserves grads, the
/// chain reaches `weight`; if not, `weight` disappears from the
/// `GradStore` (autograd break).
fn rope_fast_path_backprops_to_scale_weight() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let weight = vb.get_with_hints((HEAD_DIM,), "scale", candle_nn::Init::Const(1.0))?;

    let base = make_input(&device)?;
    let scaled = base
        .broadcast_mul(&weight.reshape((1, 1, 1, HEAD_DIM))?)?
        .contiguous()?;

    let (cos, sin) = make_cos_sin(&device)?;
    let y = candle_nn::rotary_emb::rope(&scaled, &cos, &sin)?;
    let loss = y.sqr()?.mean_all()?;
    let grads = loss.backward()?;

    let vars = varmap.all_vars();
    assert_eq!(vars.len(), 1, "expected exactly one Var (scale) in VarMap");
    let var = &vars[0];
    let grad = grads.get(var.as_tensor()).unwrap_or_else(|| {
        panic!(
            "fast-path autograd break: scale Var has no entry in GradStore \
             (candle_nn::rotary_emb::rope uses apply_op3_no_bwd; route Layer 1 RoPE through \
             candle_nn::rotary_emb::rope_slow via a shim)"
        )
    });
    let mag = grad_magnitude(grad)?;
    assert!(
        mag > 0.0,
        "fast-path autograd break: scale grad is all-zero (sum(|g|) = 0)"
    );

    Ok(())
}

#[test]
fn rope_slow_path_backprops_to_scale_weight() -> Result<()> {
    // Control: `candle_nn::rotary_emb::rope_slow` is a pure decomposition
    // (`broadcast_mul` + `rotate_half`). Every op has a proper
    // backward, so gradients must flow. If this test ever fails, the
    // slow-path fallback strategy itself is broken and Layer 1 needs
    // a hand-rolled implementation (2026-07-20 verified path).
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let weight = vb.get_with_hints((HEAD_DIM,), "scale", candle_nn::Init::Const(1.0))?;

    let base = make_input(&device)?;
    let scaled = base
        .broadcast_mul(&weight.reshape((1, 1, 1, HEAD_DIM))?)?
        .contiguous()?;

    let (cos, sin) = make_cos_sin(&device)?;
    let y = candle_nn::rotary_emb::rope_slow(&scaled, &cos, &sin)?;
    let loss = y.sqr()?.mean_all()?;
    let grads = loss.backward()?;

    let vars = varmap.all_vars();
    assert_eq!(vars.len(), 1, "expected exactly one Var (scale) in VarMap");
    let var = &vars[0];
    let grad = grads
        .get(var.as_tensor())
        .expect("slow-path autograd break: scale Var missing from GradStore");
    let mag = grad_magnitude(grad)?;
    assert!(
        mag > 0.0,
        "slow-path autograd break: scale grad is all-zero (sum(|g|) = 0)"
    );

    Ok(())
}
