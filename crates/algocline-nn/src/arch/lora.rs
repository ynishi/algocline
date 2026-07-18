//! Low-rank adaptation ("LoRA") wrap for a candle-nn `Linear`.
//!
//! Wraps a frozen base linear layer with two thin trainable matrices
//! (`lora_a` shaped `[rank, in_features]`, `lora_b` shaped
//! `[out_features, rank]`) and a scaling factor of `alpha / rank`.
//! The forward pass computes
//!
//! ```text
//! y = base.forward(x) + scaling * (lora_b(lora_a(x)))
//! ```
//!
//! The base parameters are held as a `Linear` value (so
//! `.weight()` / `.bias()` are still accessible for merge equivalence
//! checks) but the caller is expected to keep them out of the
//! optimizer's parameter list — only `lora_a` and `lora_b` should be
//! trainable during LoRA fine-tuning.
//!
//! [`LoraLinear::merged_weight`] materialises the equivalent
//! `base.weight() + scaling * (lora_b.weight() @ lora_a.weight())`
//! matrix so a caller can construct a plain `Linear` that produces
//! identical outputs for the same input. This is what the merge-
//! equivalence integration test asserts within 1e-4 element-wise.

use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// LoRA rank + scaling + wrap-target configuration.
///
/// `alpha` and `rank` together determine the scaling factor
/// `alpha / rank`; the caller sets both explicitly rather than passing
/// a pre-computed factor so a saved LoRA can be re-derived from the
/// two integer fields alone.
///
/// `target_modules` is consumed by [`crate::arch::Gpt2Model::wrap_lora`]
/// — the low-level [`LoraLinear::wrap`] path itself never inspects it.
/// `dropout` is reserved for a future stage (per-forward LoRA dropout,
/// design §7.2) and is currently ignored by both `LoraLinear::forward`
/// and the training loops; it lives on the config today so the on-disk
/// LoRA schema does not need a breaking migration when dropout ships.
///
/// The struct is `Clone` (not `Copy`) because `target_modules` is
/// heap-allocated.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition. Typical values are 8, 16,
    /// 32; the invariants below assume `rank > 0` and `rank <= min(in,
    /// out)`.
    pub rank: usize,
    /// Alpha scaling numerator (see the paper's `alpha / r` factor).
    pub alpha: f32,
    /// Module names to wrap when `Gpt2Model::wrap_lora` walks the
    /// model. Canonical GPT-2 vocabulary: `q_proj`, `k_proj`, `v_proj`,
    /// `o_proj`, `up`, `down`.
    pub target_modules: Vec<String>,
    /// Dropout probability applied on the low-rank leg during training
    /// (reserved; currently unused — see the struct-level docs).
    pub dropout: f32,
}

impl LoraConfig {
    /// Build a config with the given rank and alpha and the full
    /// canonical target-module set (attention Q/K/V/O + MLP up/down).
    ///
    /// Callers who want a narrower wrap should either mutate
    /// `target_modules` after construction or use
    /// [`LoraConfig::with_targets`].
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            target_modules: Self::default_targets(),
            dropout: 0.0,
        }
    }

    /// Build a config with an explicit target-module list.
    pub fn with_targets<I, S>(rank: usize, alpha: f32, targets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            rank,
            alpha,
            target_modules: targets.into_iter().map(Into::into).collect(),
            dropout: 0.0,
        }
    }

    /// The canonical GPT-2 target-module set: attention Q/K/V/O plus
    /// MLP up/down projections.
    pub fn default_targets() -> Vec<String> {
        vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into(),
            "up".into(),
            "down".into(),
        ]
    }

    /// The scaling factor applied to the low-rank product before it is
    /// summed with the frozen base output.
    pub fn scaling(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }
}

/// A `Linear` layer wrapped with a low-rank additive update.
///
/// Constructed via [`LoraLinear::wrap`] once the frozen base linear
/// has been built (e.g. via `candle_nn::linear` on the loaded base
/// checkpoint's `VarBuilder`). The LoRA matrices come from a distinct
/// `VarBuilder` so the caller can register them under a different
/// `VarMap` and hand *that* map to the optimizer, keeping the base
/// weights frozen.
pub struct LoraLinear {
    base: Linear,
    lora_a: Linear,
    lora_b: Linear,
    scaling: f32,
}

impl LoraLinear {
    /// Wrap `base` with newly-registered LoRA matrices.
    ///
    /// `vs` is scoped to a namespace unique to this layer (e.g.
    /// `vs.pp("c_attn").pp("lora")`) so the two `Linear`s registered
    /// below do not clash with the base parameter names.
    ///
    /// The base is expected to have already been built against a
    /// separate `VarBuilder`; passing an untrained base here is
    /// legal but the merge-equivalence guarantee holds against
    /// whatever weights `base` currently carries.
    pub fn wrap(base: Linear, cfg: LoraConfig, vs: VarBuilder) -> CandleResult<Self> {
        if cfg.rank == 0 {
            return Err(candle_core::Error::Msg(
                "LoraLinear::wrap: rank must be > 0".into(),
            ));
        }
        let in_features = base.weight().dim(1)?;
        let out_features = base.weight().dim(0)?;
        if cfg.rank > in_features || cfg.rank > out_features {
            return Err(candle_core::Error::Msg(format!(
                "LoraLinear::wrap: rank {} must be <= min(in_features, \
                 out_features) = min({in_features}, {out_features})",
                cfg.rank
            )));
        }
        // No biases on the LoRA legs — the additive contribution
        // conceptually shifts only the weight of the base linear, not
        // its bias.
        let lora_a = candle_nn::linear_no_bias(in_features, cfg.rank, vs.pp("lora_a"))?;
        let lora_b = candle_nn::linear_no_bias(cfg.rank, out_features, vs.pp("lora_b"))?;
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling: cfg.scaling(),
        })
    }

    /// Directly compose a `LoraLinear` from a pre-built base plus two
    /// caller-owned LoRA legs. Useful in tests where the caller wants
    /// deterministic weights.
    pub fn from_parts(base: Linear, lora_a: Linear, lora_b: Linear, scaling: f32) -> Self {
        Self {
            base,
            lora_a,
            lora_b,
            scaling,
        }
    }

    /// Access the frozen base linear.
    pub fn base(&self) -> &Linear {
        &self.base
    }

    /// Access the LoRA down-projection (rank × in_features).
    pub fn lora_a(&self) -> &Linear {
        &self.lora_a
    }

    /// Access the LoRA up-projection (out_features × rank).
    pub fn lora_b(&self) -> &Linear {
        &self.lora_b
    }

    /// Scaling factor applied to the low-rank product.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Compose the merged weight matrix
    /// `base.weight + scaling * (lora_b.weight @ lora_a.weight)`.
    ///
    /// Downstream code can build a plain `Linear` from the returned
    /// tensor for inference paths that do not want to carry the two
    /// low-rank matrices around.
    pub fn merged_weight(&self) -> CandleResult<Tensor> {
        let a = self.lora_a.weight(); // [rank, in]
        let b = self.lora_b.weight(); // [out, rank]
        let delta = b.matmul(a)?; // [out, in]
        let scaled = delta.affine(self.scaling as f64, 0.0)?;
        self.base.weight().broadcast_add(&scaled)
    }
}

impl Module for LoraLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let base_out = self.base.forward(xs)?;
        // (x @ Aᵀ) @ Bᵀ path — cheaper than materialising the full
        // (b @ a) matrix per forward call.
        let a_out = self.lora_a.forward(xs)?; // [..., rank]
        let b_out = self.lora_b.forward(&a_out)?; // [..., out]
        let scaled = b_out.affine(self.scaling as f64, 0.0)?;
        base_out.broadcast_add(&scaled)
    }
}

/// Snapshot two tensors as flat f32 vectors and return the maximum
/// element-wise absolute difference. Small utility so tests and
/// downstream callers can share the same tolerance check.
///
/// Uses `f32` because our merge-equivalence test runs on `DType::F32`
/// weights.
pub fn max_abs_diff_f32(a: &Tensor, b: &Tensor) -> CandleResult<f32> {
    if a.dims() != b.dims() {
        return Err(candle_core::Error::Msg(format!(
            "max_abs_diff_f32: shape mismatch {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let av: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let bv: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut m = 0.0_f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_base() -> Linear {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &device);
        candle_nn::linear(4, 3, vs.pp("base")).unwrap()
    }

    #[test]
    fn scaling_matches_alpha_over_rank() {
        let cfg = LoraConfig::new(16, 32.0);
        let s = cfg.scaling();
        assert!((s - 2.0).abs() < 1e-6);
    }

    #[test]
    fn scaling_of_zero_rank_config_is_zero() {
        let cfg = LoraConfig::new(0, 32.0);
        assert_eq!(cfg.scaling(), 0.0);
    }

    #[test]
    fn wrap_rejects_zero_rank() {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let base = tiny_base();
        let err = match LoraLinear::wrap(base, LoraConfig::new(0, 8.0), vs.pp("l0")) {
            Ok(_) => panic!("expected zero-rank rejection"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("rank"), "unexpected error: {err}");
    }

    #[test]
    fn wrap_rejects_rank_larger_than_layer_dims() {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let base = tiny_base();
        // base is [3, 4] — rank must be <= 3.
        let err = match LoraLinear::wrap(base, LoraConfig::new(8, 8.0), vs.pp("lbad")) {
            Ok(_) => panic!("expected oversized-rank rejection"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("rank"), "unexpected error: {err}");
    }

    #[test]
    fn merged_weight_matches_forward_within_tolerance() {
        let device = Device::Cpu;
        // Base parameters — deterministic values so the test doesn't
        // depend on VarBuilder RNG behaviour.
        let base_w = Tensor::from_slice(
            &[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            ],
            (3, 4),
            &device,
        )
        .unwrap();
        let base_b = Tensor::from_slice(&[0.05f32, -0.05, 0.10], (3,), &device).unwrap();
        let base = Linear::new(base_w, Some(base_b));

        // LoRA legs at rank = 2.
        let lora_a_w = Tensor::from_slice(
            &[0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            (2, 4),
            &device,
        )
        .unwrap();
        let lora_b_w =
            Tensor::from_slice(&[0.09f32, 0.10, 0.11, 0.12, 0.13, 0.14], (3, 2), &device).unwrap();
        let lora_a = Linear::new(lora_a_w, None);
        let lora_b = Linear::new(lora_b_w, None);
        let scaling = 0.5f32;

        let lora = LoraLinear::from_parts(base, lora_a, lora_b, scaling);

        // Reference: build a plain Linear from the merged weight and
        // compare its forward output against LoraLinear's forward.
        let merged_w = lora.merged_weight().unwrap();
        let merged_linear = Linear::new(merged_w, lora.base().bias().cloned());

        let xs = Tensor::from_slice(
            &[1.0f32, -1.0, 0.5, 0.25, 2.0, -2.0, 0.0, 3.0],
            (2, 4),
            &device,
        )
        .unwrap();

        let y_lora = lora.forward(&xs).unwrap();
        let y_merged = merged_linear.forward(&xs).unwrap();
        let diff = max_abs_diff_f32(&y_lora, &y_merged).unwrap();
        assert!(
            diff < 1e-4,
            "merge equivalence failed: max abs diff = {diff}"
        );
    }

    #[test]
    fn forward_shape_matches_base_output_shape() {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let base = candle_nn::linear(8, 5, vs.pp("base")).unwrap();
        let lora = LoraLinear::wrap(base, LoraConfig::new(2, 4.0), vs.pp("lora")).unwrap();
        let xs = Tensor::zeros((3, 8), DType::F32, &device).unwrap();
        let out = lora.forward(&xs).unwrap();
        assert_eq!(out.dims(), &[3, 5]);
    }
}
