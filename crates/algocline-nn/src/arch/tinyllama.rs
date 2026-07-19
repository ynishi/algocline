//! TinyLlama-1.1B trainable architecture — Layer 1a primitives.
//!
//! This file provides the backward-safe primitives that a full
//! `TinyLlamaModel` (Layer 1b) will compose:
//!
//! - [`apply_slow_rms_norm`] — RMSNorm forward via
//!   [`candle_nn::ops::rms_norm_slow`] to keep the autograd chain
//!   intact (candle-nn 0.11 `RmsNorm::forward` is a `CustomOp3` with no
//!   backward, see `tests/rms_norm_autograd_gate.rs`).
//! - [`apply_rope`] — Rotary embedding forward via
//!   [`candle_nn::rotary_emb::rope_slow`] for the same reason
//!   (`rotary_emb::rope` uses `apply_op3_no_bwd`).
//! - [`build_rope_cache`] — precomputed cos / sin cache using the
//!   canonical Llama-family frequency formula
//!   `theta_i = base^{-2i/head_dim}` for `i ∈ [0, head_dim/2)`.
//! - [`repeat_kv`] — grouped-query-attention KV expansion:
//!   `[B, H_kv, S, head_dim] → [B, H, S, head_dim]` via `repeat`.
//!
//! Full `TinyLlamaConfig` / `TinyLlamaBlock` / `TinyLlamaModel` land
//! in Layer 1b. Keeping this file primitive-only makes each helper
//! individually testable and forces the shim layer to stabilize
//! before the model wiring depends on it.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{ops, rotary_emb, RmsNorm};

/// RMSNorm forward that always uses the backward-safe basic-op path
/// (`rms_norm_slow`).
///
/// Rationale: candle-nn 0.11's `RmsNorm::forward` calls
/// `crate::ops::rms_norm`, a `CustomOp3` whose backward is not
/// implemented — a `Var` upstream of any RMSNorm loses its gradient
/// chain. This shim keeps the chain intact by routing through
/// [`ops::rms_norm_slow`], a pure-op decomposition
/// (`sqr` / `mean_keepdim` / `add` / `sqrt` / `div` / `broadcast_mul`)
/// with a proper backward implementation.
///
/// The affine `weight` is read via [`RmsNorm::inner`] and forwarded to
/// the slow path. Mirrors the shape of
/// [`super::gpt2::apply_slow_layer_norm`]; the two shims will
/// eventually merge into a shared `nn::slow_ops` module once TinyLlama
/// lands (tracked as a follow-up under GH #8-adjacent refactor).
///
/// See also: `crates/algocline-nn/tests/rms_norm_autograd_gate.rs`.
pub fn apply_slow_rms_norm(rn: &RmsNorm, x: &Tensor) -> CandleResult<Tensor> {
    ops::rms_norm_slow(x, rn.weight(), rn.eps() as f32)
}

/// Rotary Position Embedding forward using the backward-safe
/// [`rotary_emb::rope_slow`] helper.
///
/// Rationale: candle-nn 0.11's `rotary_emb::rope` is registered via
/// `apply_op3_no_bwd`, so any `Var` upstream of RoPE loses its
/// gradient chain. `rope_slow` is a 6-line pure-op decomposition
/// (`broadcast_mul` + `rotate_half` via `cat`/`narrow`/`neg`) that
/// preserves the chain.
///
/// Contract mirrors `rope_slow`:
///
/// - `xs`  : `[batch, n_head, seq_len, head_dim]`, contiguous.
/// - `cos` : `[seq_len_max, head_dim / 2]`, contiguous.
/// - `sin` : `[seq_len_max, head_dim / 2]`, contiguous.
///
/// `head_dim` must be even (the slow path bisects the last dimension).
/// The caller is responsible for narrowing `cos` / `sin` to the
/// active sequence prefix; `rope_slow` will do the seq narrow
/// internally.
///
/// See also: `crates/algocline-nn/tests/rope_autograd_gate.rs`.
pub fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    rotary_emb::rope_slow(xs, cos, sin)
}

/// Build a canonical Llama-family RoPE cos / sin cache.
///
/// Frequencies follow the standard formula:
///
/// ```text
/// theta_i = base^{-2 i / head_dim}   for i in [0, head_dim / 2)
/// cos[p, i] = cos(p * theta_i)
/// sin[p, i] = sin(p * theta_i)
/// ```
///
/// The output shape is `[max_seq, head_dim / 2]`, matching the
/// contract of [`apply_rope`]. `base` is TinyLlama's `rope_theta`
/// (10000 in the reference config). Values are computed in `f32` and
/// cast to the requested `dtype` at the end so downstream matmul
/// dtype-matches the model params.
///
/// `head_dim` must be even; the function returns an error otherwise.
pub fn build_rope_cache(
    max_seq: usize,
    head_dim: usize,
    base: f32,
    dtype: DType,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    if !head_dim.is_multiple_of(2) {
        return Err(candle_core::Error::Msg(format!(
            "build_rope_cache: head_dim must be even (got {head_dim})"
        )));
    }
    let half = head_dim / 2;

    // theta_i = base^{-2 i / head_dim}
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| base.powf(-((2 * i) as f32) / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?; // [1, half]

    // positions [max_seq, 1]
    let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq, 1), device)?; // [max_seq, 1]

    // angles [max_seq, half]  =  positions ⨂ inv_freq
    let angles = positions.broadcast_mul(&inv_freq)?;
    let cos = angles.cos()?.to_dtype(dtype)?.contiguous()?;
    let sin = angles.sin()?.to_dtype(dtype)?.contiguous()?;
    Ok((cos, sin))
}

/// Grouped-query-attention KV expansion.
///
/// Repeats the KV heads so that each of the `n_kv` KV heads is shared
/// across `n_rep` query heads. Input / output shapes:
///
/// - `xs` : `[batch, n_kv, seq, head_dim]`
/// - out  : `[batch, n_kv * n_rep, seq, head_dim]`
///
/// `n_rep == 1` is a fast pass-through. This is the same shape as the
/// standard `repeat_kv` used in HuggingFace's Llama implementation and
/// in `candle-transformers::models::llama`.
pub fn repeat_kv(xs: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, n_kv, s, d) = xs.dims4()?;
    // [b, n_kv, 1, s, d]  →  broadcast to [b, n_kv, n_rep, s, d]
    //                    →  reshape to [b, n_kv * n_rep, s, d]
    xs.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use candle_nn::{rms_norm, VarBuilder, VarMap};

    /// `apply_slow_rms_norm` produces the same numerical result as the
    /// canonical `RmsNorm::forward` (mean-square normalization with the
    /// affine weight).  Numerical tolerance is generous because the
    /// slow path composes several float ops in a different order than
    /// the fast path.
    #[test]
    fn apply_slow_rms_norm_matches_fast_path_numerically() -> CandleResult<()> {
        use candle_nn::Module;

        let device = Device::Cpu;
        let dim = 8;
        let eps = 1e-5;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let rn = rms_norm(dim, eps, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 4, dim), &device)?;
        let y_fast = rn.forward(&x)?;
        let y_slow = apply_slow_rms_norm(&rn, &x)?;

        let diff: f32 = (y_fast - y_slow)?.abs()?.max(0)?.max(0)?.max(0)?.to_scalar()?;
        assert!(
            diff < 1e-4,
            "slow-path RMSNorm diverges from fast path: max_abs_diff = {diff}"
        );
        Ok(())
    }

    /// `apply_rope` is a thin wrapper around `rope_slow`; verify the
    /// output shape matches the input shape.
    #[test]
    fn apply_rope_preserves_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let (b, h, s, d) = (2, 4, 8, 16);
        let x = Tensor::randn(0f32, 1f32, (b, h, s, d), &device)?.contiguous()?;
        let (cos, sin) = build_rope_cache(s, d, 10_000.0, DType::F32, &device)?;
        let y = apply_rope(&x, &cos, &sin)?;
        assert_eq!(y.dims4()?, (b, h, s, d));
        Ok(())
    }

    /// `build_rope_cache` returns cos / sin with `[max_seq, head_dim/2]`
    /// and values in `[-1, 1]`.
    #[test]
    fn build_rope_cache_shape_and_range() -> CandleResult<()> {
        let device = Device::Cpu;
        let (max_seq, head_dim, base) = (32, 64, 10_000.0);
        let (cos, sin) = build_rope_cache(max_seq, head_dim, base, DType::F32, &device)?;

        assert_eq!(cos.dims(), &[max_seq, head_dim / 2]);
        assert_eq!(sin.dims(), &[max_seq, head_dim / 2]);

        let cos_max: f32 = cos.max(0)?.max(0)?.to_scalar()?;
        let cos_min: f32 = cos.min(0)?.min(0)?.to_scalar()?;
        assert!(cos_max <= 1.0 + 1e-6 && cos_min >= -1.0 - 1e-6);

        // Position 0 → all angles are 0 → cos = 1, sin = 0.
        let cos_row0 = cos.i(0)?;
        let sin_row0 = sin.i(0)?;
        let cos0_min: f32 = cos_row0.min(0)?.to_scalar()?;
        let sin0_abs_max: f32 = sin_row0.abs()?.max(0)?.to_scalar()?;
        assert!((cos0_min - 1.0).abs() < 1e-5, "cos[0] should be all 1");
        assert!(sin0_abs_max < 1e-5, "sin[0] should be all 0");
        Ok(())
    }

    /// `build_rope_cache` rejects odd `head_dim`.
    #[test]
    fn build_rope_cache_rejects_odd_head_dim() {
        let device = Device::Cpu;
        let err = build_rope_cache(8, 15, 10_000.0, DType::F32, &device).unwrap_err();
        assert!(
            err.to_string().contains("head_dim must be even"),
            "expected head_dim error, got: {err}"
        );
    }

    /// `repeat_kv` with `n_rep = 1` returns an equivalent tensor.
    #[test]
    fn repeat_kv_identity_when_n_rep_is_1() -> CandleResult<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (2, 4, 8, 16), &device)?;
        let y = repeat_kv(&x, 1)?;
        assert_eq!(y.dims4()?, x.dims4()?);
        let diff: f32 = (&x - &y)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar()?;
        assert_eq!(diff, 0.0);
        Ok(())
    }

    /// `repeat_kv` expansion matches TinyLlama-1.1B's `H_kv=4, H=32`
    /// case (`n_rep = 8`) and repeats each KV head consecutively.
    #[test]
    fn repeat_kv_expands_to_full_head_count() -> CandleResult<()> {
        let device = Device::Cpu;
        let (b, n_kv, s, d) = (1, 4, 3, 8);
        let n_rep = 8;
        let x = Tensor::randn(0f32, 1f32, (b, n_kv, s, d), &device)?;
        let y = repeat_kv(&x, n_rep)?;
        assert_eq!(y.dims4()?, (b, n_kv * n_rep, s, d));

        // Every consecutive block of `n_rep` heads on the query axis
        // must equal the corresponding KV head.
        for kv_idx in 0..n_kv {
            let kv_slice = x.i((.., kv_idx))?; // [b, s, d]
            for rep in 0..n_rep {
                let q_idx = kv_idx * n_rep + rep;
                let q_slice = y.i((.., q_idx))?;
                let diff: f32 = (&q_slice - &kv_slice)?
                    .abs()?
                    .max(0)?
                    .max(0)?
                    .max(0)?
                    .to_scalar()?;
                assert_eq!(
                    diff, 0.0,
                    "kv head {kv_idx} rep {rep} (q head {q_idx}) does not equal source"
                );
            }
        }
        Ok(())
    }
}
