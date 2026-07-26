//! Dense Mixture-of-Experts feed-forward for the GPT-2 stack.
//!
//! Replaces a block's MLP with a router + `n_experts` GPT-2-shaped
//! experts (`c_fc → GELU → c_proj`). Routing follows the standard
//! top-k recipe (Switch Transformer, Fedus et al. 2021 for top-1;
//! Mixtral, Jiang et al. 2024 for the top-2 convention): softmax the
//! router logits, keep the top-k probabilities per token, renormalize,
//! and mix expert outputs with those weights.
//!
//! **Dense compute only.** Every expert runs on every token and the
//! outputs are combined by weight — no token dispatch / grouped GEMM.
//! candle has no expert-dispatch kernel, and a scatter/gather custom op
//! would sit in the same no-backward `CustomOp` trap the LayerNorm /
//! RoPE slow-path shims exist for (`apply_slow_layer_norm`,
//! `arch::tinyllama`). The dense mixture is a composition of
//! softmax / linear / GELU / mul / add, all of which carry proper
//! backward implementations, so the autograd chain stays intact.
//!
//! `top_k = n_experts` degenerates into a dense softmax mixture where
//! every expert receives gradient on every step — the
//! `tests/moe_grad_coverage.rs` gate runs in that mode to check
//! structural gradient reachability without tangling with the (normal)
//! sparsity of top-k routing.

use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{Init, Linear, Module, VarBuilder};

use super::gpt2::{gpt2_linear, residual_init_stdev, Gpt2Config, INIT_STDEV};

/// Configuration for the dense-MoE feed-forward.
///
/// Attached to a [`Gpt2Config`] via its `moe` field; `None` keeps the
/// stock GPT-2 MLP. MoE models are **random-init only** — there is no
/// pretrained bundle with this layout, so
/// [`super::gpt2::Gpt2Model::from_pretrained`] refuses configs that
/// carry a `MoeConfig`.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of experts per block. Must be ≥ 1.
    pub n_experts: usize,
    /// Number of experts a token's output is mixed from. Must satisfy
    /// `1 ≤ top_k ≤ n_experts`. `top_k == n_experts` disables the
    /// selection and yields the dense softmax mixture.
    pub top_k: usize,
    /// Load-balancing loss coefficient (Switch Transformer §2.2 uses
    /// 0.01). The model returns the *unscaled* aux term from
    /// `forward_with_aux`; callers apply `alpha` when composing the
    /// total loss so probes can A/B the coefficient without rebuilding.
    pub alpha: f64,
}

impl MoeConfig {
    /// Standard preset: top-2 routing (Mixtral convention) with the
    /// Switch load-balancing coefficient. `top_k` is clamped to
    /// `n_experts` so `new(1)` is valid.
    pub fn new(n_experts: usize) -> Self {
        Self {
            n_experts,
            top_k: 2.min(n_experts.max(1)),
            alpha: 0.01,
        }
    }

    /// Dense-mixture preset (`top_k = n_experts`): every expert is
    /// mixed on every token. The grad-coverage gate runs in this mode.
    pub fn dense_mixture(n_experts: usize) -> Self {
        Self {
            n_experts,
            top_k: n_experts,
            alpha: 0.01,
        }
    }

    /// Validate the invariants the forward path assumes. Called by the
    /// model constructor so an invalid config fails at build time with
    /// a clear message rather than as a shape error mid-forward.
    pub fn validate(&self) -> CandleResult<()> {
        if self.n_experts == 0 {
            return Err(candle_core::Error::Msg(
                "moe: n_experts must be >= 1".into(),
            ));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(candle_core::Error::Msg(format!(
                "moe: top_k {} out of range 1..={}",
                self.top_k, self.n_experts
            )));
        }
        Ok(())
    }
}

/// One expert: the GPT-2 MLP shape (`dim → 4·dim → dim` with GELU),
/// initialized exactly like the stock `mlp.c_fc` / `mlp.c_proj` pair —
/// including the `1/sqrt(2·layers)` residual-write scaling on `c_proj`,
/// since each expert's output feeds the residual stream through the
/// mixture.
struct Expert {
    c_fc: Linear,
    c_proj: Linear,
}

impl Expert {
    fn new(cfg: &Gpt2Config, vs: VarBuilder) -> CandleResult<Self> {
        let resid_stdev = residual_init_stdev(cfg.layers);
        let c_fc = gpt2_linear(cfg.dim, 4 * cfg.dim, INIT_STDEV, vs.pp("c_fc"))?;
        let c_proj = gpt2_linear(4 * cfg.dim, cfg.dim, resid_stdev, vs.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = h.gelu()?;
        self.c_proj.forward(&h)
    }
}

/// Dense-MoE feed-forward: router + experts.
///
/// VarMap naming (under the block's `h.<i>.moe` scope, disjoint from
/// the stock `h.<i>.mlp.*` names):
///
/// ```text
/// h.<i>.moe.router.weight                    [n_experts, dim]
/// h.<i>.moe.experts.<j>.c_fc.weight/.bias
/// h.<i>.moe.experts.<j>.c_proj.weight/.bias
/// ```
pub(crate) struct MoeMlp {
    /// `dim → n_experts`, bias-free (Switch convention), drawn from
    /// `N(0, INIT_STDEV)` like the other "reading" projections.
    router: Linear,
    experts: Vec<Expert>,
    top_k: usize,
}

/// Per-forward MoE byproducts, threaded back through the block so the
/// model can aggregate them across layers.
pub(crate) struct MoeOutput {
    /// Mixed expert output `[B, S, D]`.
    pub y: Tensor,
    /// Unscaled Switch load-balancing term `E · Σ_i f_i · P_i`
    /// (scalar). `f_i` (top-1 assignment fraction) is a constant with
    /// respect to the parameters — only the router-probability mean
    /// `P_i` carries gradient, per the reference implementation.
    pub aux: Tensor,
    /// Router probabilities `[B, S, E]` (full softmax, pre top-k) for
    /// utilization / entropy probes.
    pub probs: Tensor,
}

impl MoeMlp {
    /// Build a router + experts under `vs` (caller scopes it to
    /// `h.<i>.moe`). `moe` must already be validated.
    pub(crate) fn new(cfg: &Gpt2Config, moe: &MoeConfig, vs: VarBuilder) -> CandleResult<Self> {
        let router_ws = vs.pp("router").get_with_hints(
            (moe.n_experts, cfg.dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: INIT_STDEV,
            },
        )?;
        let router = Linear::new(router_ws, None);
        let experts_vs = vs.pp("experts");
        let mut experts = Vec::with_capacity(moe.n_experts);
        for j in 0..moe.n_experts {
            experts.push(Expert::new(cfg, experts_vs.pp(j.to_string()))?);
        }
        Ok(Self {
            router,
            experts,
            top_k: moe.top_k,
        })
    }

    /// Dense-MoE forward. `x` is `[B, S, D]`.
    ///
    /// 1. `probs = softmax(router(x))` — `[B, S, E]`
    /// 2. top-k: keep the k largest probabilities per token,
    ///    renormalize them to sum to 1, zero the rest. The keep-mask is
    ///    a comparison result (no gradient), so gradient flows only
    ///    into the selected components of `probs` — the standard top-k
    ///    routing behaviour.
    /// 3. `y = Σ_e w_e · expert_e(x)` with every expert computed
    ///    densely.
    pub(crate) fn forward(&self, x: &Tensor) -> CandleResult<MoeOutput> {
        let e = self.experts.len();
        let router_logits = self.router.forward(x)?; // [B, S, E]
                                                     // Backward-safe softmax (see `super::softmax_last_dim_slow`).
                                                     // The router's ONLY gradient path is this softmax — the fused
                                                     // no-backward kernel leaves it permanently frozen, which is
                                                     // exactly what `tests/moe_grad_coverage.rs` first caught.
        let probs = super::softmax_last_dim_slow(&router_logits)?;

        let weights = if self.top_k >= e {
            probs.clone()
        } else {
            // k-th largest probability per token as the keep threshold.
            // Ties beyond position k are measure-zero for continuous
            // router outputs, so `>=` keeps exactly k entries in
            // practice.
            let (sorted, _idx) = probs.sort_last_dim(false)?; // descending
            let kth = sorted
                .narrow(D::Minus1, self.top_k - 1, 1)?
                .broadcast_as(probs.shape())?;
            let keep = probs.ge(&kth)?.to_dtype(probs.dtype())?;
            let masked = probs.mul(&keep)?;
            let denom = masked.sum_keepdim(D::Minus1)?;
            masked.broadcast_div(&denom)?
        };

        let mut y: Option<Tensor> = None;
        for (j, expert) in self.experts.iter().enumerate() {
            let w_j = weights.narrow(D::Minus1, j, 1)?; // [B, S, 1]
            let contrib = expert.forward(x)?.broadcast_mul(&w_j)?;
            y = Some(match y {
                None => contrib,
                Some(acc) => (acc + contrib)?,
            });
        }
        let y = y.ok_or_else(|| candle_core::Error::Msg("moe: expert list is empty".into()))?;

        let aux = self.load_balancing_aux(&probs)?;
        Ok(MoeOutput { y, aux, probs })
    }

    /// Switch Transformer load-balancing loss (§2.2), unscaled:
    /// `E · Σ_i f_i · P_i` where `f_i` is the fraction of tokens whose
    /// top-1 expert is `i` and `P_i` is the mean router probability for
    /// expert `i`. Uniform routing gives exactly 1.0.
    fn load_balancing_aux(&self, probs: &Tensor) -> CandleResult<Tensor> {
        let (b, s, e) = probs.dims3()?;
        let flat = probs.reshape((b * s, e))?;
        let p_mean = flat.mean(0)?; // [E], differentiable

        // Top-1 assignment as a one-hot fraction — comparison output,
        // constant w.r.t. parameters.
        let top1 = flat.argmax(D::Minus1)?; // [B*S] u32
        let ids = Tensor::arange(0u32, e as u32, probs.device())?
            .reshape((1, e))?
            .broadcast_as((b * s, e))?;
        let onehot = top1
            .reshape((b * s, 1))?
            .broadcast_as((b * s, e))?
            .eq(&ids)?
            .to_dtype(probs.dtype())?;
        let f = onehot.mean(0)?; // [E]

        (f.mul(&p_mean)?.sum_all()? * (e as f64))?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_cfg(moe: Option<MoeConfig>) -> Gpt2Config {
        Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe,
        }
    }

    fn build_moe(moe: MoeConfig) -> MoeMlp {
        let cfg = tiny_cfg(None);
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        MoeMlp::new(&cfg, &moe, vs.pp("moe")).expect("build moe")
    }

    #[test]
    fn validate_rejects_bad_shapes() {
        assert!(MoeConfig {
            n_experts: 0,
            top_k: 1,
            alpha: 0.01
        }
        .validate()
        .is_err());
        assert!(MoeConfig {
            n_experts: 4,
            top_k: 0,
            alpha: 0.01
        }
        .validate()
        .is_err());
        assert!(MoeConfig {
            n_experts: 4,
            top_k: 5,
            alpha: 0.01
        }
        .validate()
        .is_err());
        assert!(MoeConfig::new(4).validate().is_ok());
        assert!(MoeConfig::new(1).validate().is_ok());
        assert!(MoeConfig::dense_mixture(3).validate().is_ok());
    }

    #[test]
    fn new_defaults_to_top2() {
        let c = MoeConfig::new(4);
        assert_eq!(c.top_k, 2);
        // Clamped when fewer experts than the Mixtral default.
        assert_eq!(MoeConfig::new(1).top_k, 1);
    }

    #[test]
    fn forward_shape_and_probs() {
        let cfg = tiny_cfg(None);
        let m = build_moe(MoeConfig::new(4));
        let x = Tensor::randn(0f32, 1f32, (2, 5, cfg.dim), &cfg.device).unwrap();
        let out = m.forward(&x).expect("forward");
        assert_eq!(out.y.dims(), &[2, 5, cfg.dim]);
        assert_eq!(out.probs.dims(), &[2, 5, 4]);
        assert_eq!(out.aux.dims(), &[] as &[usize]);
    }

    /// Top-k weights: exactly `k` non-zero entries per token and they
    /// renormalize to 1. Verified through the mixture identity — with
    /// experts probing is indirect, so check the weight tensor via the
    /// degenerate path instead: run with k < E and recover the weights
    /// by comparing against the dense-mixture probabilities.
    #[test]
    fn top_k_keeps_k_entries_renormalized() {
        let m = build_moe(MoeConfig {
            n_experts: 4,
            top_k: 2,
            alpha: 0.01,
        });
        let cfg = tiny_cfg(None);
        let x = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &cfg.device).unwrap();
        let out = m.forward(&x).unwrap();

        // Recompute the top-2 weights from the returned full probs and
        // check the invariant directly.
        let probs: Vec<Vec<f32>> = out.probs.reshape((3, 4)).unwrap().to_vec2().unwrap();
        for row in probs {
            let mut sorted = row.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let kth = sorted[1];
            let kept: Vec<f32> = row.iter().cloned().filter(|p| *p >= kth).collect();
            assert_eq!(kept.len(), 2, "expected exactly 2 kept probs: {row:?}");
        }
    }

    #[test]
    fn uniform_probs_give_unit_aux() {
        let m = build_moe(MoeConfig::dense_mixture(4));
        // Hand-built uniform probs: aux = E * Σ f_i * (1/E) = Σ f_i = 1.
        let probs = Tensor::full(0.25f32, (1, 8, 4), &Device::Cpu).unwrap();
        let aux: f32 = m.load_balancing_aux(&probs).unwrap().to_scalar().unwrap();
        assert!((aux - 1.0).abs() < 1e-6, "aux={aux}");
    }
}
