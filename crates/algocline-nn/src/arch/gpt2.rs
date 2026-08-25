//! GPT-2 architecture builder.
//!
//! Implements the two variants shipped in Phase 1 (design §6.1):
//!
//! - `gpt2-medium` — 24 layers, 16 heads, 1024 dim, 1024 ctx, 50257 vocab
//! - `gpt2-large`  — 36 layers, 20 heads, 1280 dim, 1024 ctx, 50257 vocab
//!
//! Architecture components (nanoGPT / HuggingFace `openai-community/gpt2`
//! reference layout):
//!
//! - `wte` — token embedding (`vocab × dim`)
//! - `wpe` — learned positional embedding (`ctx × dim`)
//! - `h.<i>.ln_1` / `ln_2` — pre-LayerNorm (dim)
//! - `h.<i>.attn.c_attn` — fused Q/K/V projection (`dim → 3·dim`)
//! - `h.<i>.attn.c_proj` — attention output projection (`dim → dim`)
//! - `h.<i>.mlp.c_fc` / `mlp.c_proj` — 4× expansion MLP with GELU
//! - `ln_f` — final LayerNorm (dim)
//! - LM head weights are tied to `wte` (shared matrix)
//!
//! Forward output shape is `[batch, seq, vocab]` per subtask invariant
//! #1. Attention uses a causal (lower-triangular) mask.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{
    layer_norm, rms_norm, Embedding, Init, LayerNorm, Module, RmsNorm, VarBuilder, VarMap,
};

use super::custom::{Activation, Gpt2Custom, NormKind, NormPlacement, PosKind, ResidualKind};
use super::tinyllama::{apply_rope, apply_slow_rms_norm, build_rope_cache, repeat_kv};

// `LoraLinear` is imported for the intra-doc links (`[`LoraLinear`]`) in the
// wrap_lora docs below; the wrap helper itself lives in `arch::lora` since
// TinyLlama needs the same swap-in-place idiom.
#[allow(unused_imports)]
use super::lora::{wrap_variant_in_place, LinearVariant, LoraConfig, LoraLinear};
use super::moe::{MoeConfig, MoeMlp};

/// Which projections inside a [`Block`] the caller wants LoRA-wrapped.
#[derive(Debug, Clone, Copy)]
struct WrapFlags {
    /// Wrap the fused Q/K/V projection (`c_attn`). Any of `q_proj`,
    /// `k_proj`, `v_proj` in the target list flips this on because
    /// candle-nn's GPT-2 layout keeps the three projections fused.
    qkv: bool,
    /// Wrap the attention output projection (`c_proj`).
    o: bool,
    /// Wrap the MLP up-projection (`mlp.c_fc`).
    up: bool,
    /// Wrap the MLP down-projection (`mlp.c_proj`).
    down: bool,
}

/// Canonical GPT-2 target-module names accepted by
/// [`Gpt2Model::wrap_lora`]. Any name outside this list triggers an
/// error at wrap time so a typo does not silently degrade to "no-op".
const KNOWN_TARGET_MODULES: [&str; 6] = ["q_proj", "k_proj", "v_proj", "o_proj", "up", "down"];

/// Standard deviation of the normal distribution the GPT-2 reference
/// implementation draws `wte` / `wpe` from.
///
/// candle-nn's [`candle_nn::embedding`] helper defaults to
/// `Randn { mean: 0.0, stdev: 1.0 }`, which is 50× too wide here. It
/// matters more for GPT-2 than for a model with an untied head because
/// [`Gpt2Model::forward`] reuses `wte` as the LM head: the logit scale
/// is `sqrt(dim) * stdev(wte)`, so a `stdev = 1.0` draw puts
/// `gpt2-medium` logits at `std ~= 32` and a from-scratch masked
/// cross-entropy at ~140 instead of the `ln(vocab) ~= 10.82` a uniform
/// softmax gives. Training from there saturates the softmax rather than
/// descending. Measured on `examples/init_loss_probe.rs`.
pub(crate) const INIT_STDEV: f64 = 0.02;

/// Overwrite every registered parameter with values a seed decides, so
/// a test can assert on the numbers a forward pass actually produces.
///
/// candle has no `Device::set_seed` on this version and this crate does
/// not seed its initialisers, so a model built by [`Gpt2Model::new`] is
/// different on every run. A test that thresholds the difference
/// between two such forwards is thresholding a random draw; this makes
/// the draw the test's own.
///
/// Norm weights are held at one and every bias at zero rather than
/// randomised: a norm scaled by a number near zero would make the model
/// degenerate in a way no real initialisation is.
#[cfg(test)]
pub(crate) fn fill_deterministic(vm: &VarMap, seed: u64) -> CandleResult<()> {
    let data = vm
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("fill_deterministic: VarMap lock poisoned".into()))?;
    // `HashMap` iteration order is arbitrary and the generator is
    // sequential, so the names have to be walked in a fixed order for
    // the fill to be reproducible.
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // [-0.05, 0.05), the order of magnitude the real initialiser
        // draws at (INIT_STDEV = 0.02).
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
    };
    for name in names {
        let var = &data[name];
        let n = var.shape().elem_count();
        let values: Vec<f32> = if name.ends_with(".bias") {
            vec![0.0; n]
        } else if name.contains("ln_") {
            vec![1.0; n]
        } else {
            (0..n).map(|_| next()).collect()
        };
        let t =
            Tensor::from_vec(values, var.shape().clone(), var.device())?.to_dtype(var.dtype())?;
        var.set(&t)?;
    }
    Ok(())
}

/// Embedding table drawn from `N(0, INIT_STDEV)` instead of candle-nn's
/// `N(0, 1)` default.
///
/// Only the random-init path is affected: when `vs` is backed by a
/// safetensors mmap ([`Gpt2Model::from_pretrained`] /
/// [`Gpt2Model::from_safetensors_file`]) the hint is ignored and the
/// stored tensor is loaded as-is.
fn gpt2_embedding(rows: usize, dim: usize, vs: VarBuilder) -> CandleResult<Embedding> {
    let ws = vs.get_with_hints(
        (rows, dim),
        "weight",
        Init::Randn {
            mean: 0.0,
            stdev: INIT_STDEV,
        },
    )?;
    Ok(Embedding::new(ws, dim))
}

/// Linear projection drawn from `N(0, stdev)` with a zero bias, per the
/// GPT-2 reference, instead of candle-nn's Kaiming weight + uniform
/// bias default.
///
/// `stdev` is [`INIT_STDEV`] for the two "reading" projections
/// (`c_attn`, `mlp.c_fc`) and [`residual_init_stdev`] for the two that
/// write back into the residual stream (`attn.c_proj`, `mlp.c_proj`).
///
/// Like [`gpt2_embedding`], the hint only takes effect on the
/// random-init path; a safetensors-backed `vs` loads stored weights.
pub(crate) fn gpt2_linear(
    in_dim: usize,
    out_dim: usize,
    stdev: f64,
    vs: VarBuilder,
) -> CandleResult<candle_nn::Linear> {
    let ws = vs.get_with_hints(
        (out_dim, in_dim),
        "weight",
        Init::Randn { mean: 0.0, stdev },
    )?;
    let bs = vs.get_with_hints(out_dim, "bias", Init::Const(0.0))?;
    Ok(candle_nn::Linear::new(ws, Some(bs)))
}

/// Standard deviation for the projections that write into the residual
/// stream (`attn.c_proj` and `mlp.c_proj`).
///
/// GPT-2 scales these by `1/sqrt(2 * n_layer)` — two residual writes per
/// block — so the variance the stream accumulates stays independent of
/// depth instead of growing with it. Without the scaling a 24-layer
/// `gpt2-medium` enters `ln_f` with a residual whose magnitude is set by
/// depth rather than by content.
pub(crate) fn residual_init_stdev(layers: usize) -> f64 {
    INIT_STDEV / ((2 * layers.max(1)) as f64).sqrt()
}

/// LayerNorm forward that always uses the backward-safe basic-op path
/// (`layer_norm_slow`).
///
/// Rationale: candle-nn 0.11's `LayerNorm::forward` fast path calls
/// `crate::ops::layer_norm`, a `CustomOp3` registered via
/// `apply_op3_no_bwd`. That path yields a tensor with
/// `BackpropOp::none()`, which **severs the autograd graph** at every
/// LayerNorm: any parameter upstream of a LayerNorm (i.e. every
/// transformer-block Var — the tied `wte` LM head is the only exception
/// because it participates only downstream of the final LN) receives no
/// gradient, so full FT and LoRA both silently fail to learn while
/// forward + loss + optimizer.step still run without error.
/// See <https://github.com/huggingface/candle/blob/candle-nn-v0.11.0/candle-nn/src/ops.rs>
/// `apply_op3_no_bwd`.
///
/// The slow path (`layer_norm_slow`) computes the same numerical result
/// via `sub` / `sqr` / `sum_keepdim` / `div` / `sqrt` / `mul` / `add`,
/// each of which has a proper backward implementation, keeping the
/// gradient chain intact from loss back to every trainable Var.
///
/// The affine parameters (`weight`, `bias`) are supplied by the caller
/// via `ln.weight()` / `ln.bias()`; only the `remove_mean = true` case
/// with `bias.is_some()` is exercised in this crate today, matching the
/// GPT-2 LayerNorm topology. A None-bias fallback (RmsNorm-style) is
/// out of scope until a caller needs it.
fn apply_slow_layer_norm(ln: &LayerNorm, x: &Tensor) -> CandleResult<Tensor> {
    let bias = ln.bias().ok_or_else(|| {
        candle_core::Error::Msg(
            "apply_slow_layer_norm: LayerNorm without bias is not supported yet".into(),
        )
    })?;
    candle_nn::ops::layer_norm_slow(x, ln.weight(), bias, ln.eps() as f32)
}

/// Immutable configuration for a GPT-2 preset.
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    /// Number of transformer blocks.
    pub layers: usize,
    /// Number of attention heads. `dim` must be divisible by `heads`.
    pub heads: usize,
    /// Model hidden size.
    pub dim: usize,
    /// Maximum context length (positional embedding size).
    pub ctx: usize,
    /// Vocabulary size (matches the tokenizer, 50257 for GPT-2).
    pub vocab: usize,
    /// Weight precision.
    pub dtype: DType,
    /// Device the parameters live on.
    pub device: Device,
    /// LayerNorm epsilon (HuggingFace GPT-2 default = 1e-5).
    pub eps: f64,
    /// Dense-MoE feed-forward configuration. `None` (all shipped
    /// presets) keeps the stock GPT-2 MLP; `Some` swaps every block's
    /// MLP for a router + experts mixture ([`MoeConfig`]). MoE models
    /// are random-init only — the pretrained loaders reject configs
    /// that set this.
    pub moe: Option<MoeConfig>,
    /// Architecture customization spec ([`Gpt2Custom`]: activation /
    /// norm kind + placement / residual topology / MLP ratio /
    /// position / GQA / sliding window / untied head). `None` (all
    /// shipped presets) is the GPT-2 reference. Custom models cannot
    /// warm-start from a HuggingFace bundle
    /// ([`Gpt2Model::from_pretrained`] rejects them), but a bundle this
    /// crate wrote for the same spec reloads through
    /// [`Gpt2Model::from_safetensors_file`]. Combining `custom` with `moe` is allowed as
    /// long as the dense-MLP knobs (`act` / `mlp_ratio`) stay at the
    /// reference — those two do not apply to the experts and a
    /// non-default value would silently not take effect, so the build
    /// rejects that combination.
    pub custom: Option<Gpt2Custom>,
}

impl Gpt2Config {
    /// `gpt2-medium` preset (355M params).
    pub fn medium() -> Self {
        Self {
            layers: 24,
            heads: 16,
            dim: 1024,
            ctx: 1024,
            vocab: 50257,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        }
    }

    /// `gpt2-large` preset (774M params).
    pub fn large() -> Self {
        Self {
            layers: 36,
            heads: 20,
            dim: 1280,
            ctx: 1024,
            vocab: 50257,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        }
    }

    /// `tiny` preset — a 2-layer / 2-head / dim-32 / ctx-16 / vocab-64
    /// GPT-2 built for CPU smoke tests.
    ///
    /// Shape lets a from-scratch forward pass complete in milliseconds
    /// on CPU (integration tests in `tests/{full_ft,distill}_synthetic.rs`
    /// use a comparable configuration inline) so the Lua-facing
    /// `examples/nn_*_smoke.lua` scripts run in a couple of seconds
    /// without downloading pretrained weights.
    ///
    /// [`Self::hf_repo`] returns `None` for this shape — there is no
    /// HuggingFace bundle at this size, and
    /// [`Gpt2Model::from_pretrained`] therefore refuses `pretrained =
    /// true` on the tiny variant. Callers of `alc.nn.preset.gpt2(
    /// "tiny", ...)` must pass `pretrained = false` to build a
    /// from-scratch handle whose `VarMap` the trainer bindings can
    /// then optimise.
    pub fn tiny() -> Self {
        Self {
            layers: 2,
            heads: 2,
            dim: 32,
            ctx: 16,
            vocab: 64,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        }
    }

    /// Resolve a variant name (`"medium"` / `"large"` / `"tiny"`) to
    /// the matching preset. Returns `None` for unknown names.
    pub fn from_variant(variant: &str) -> Option<Self> {
        match variant {
            "medium" | "gpt2-medium" => Some(Self::medium()),
            "large" | "gpt2-large" => Some(Self::large()),
            "tiny" | "gpt2-tiny" => Some(Self::tiny()),
            _ => None,
        }
    }

    /// HuggingFace repository id for warm-start weight download
    /// (design §12 Q5). Returns `None` for a config not built from a
    /// standard preset (including [`Self::tiny`], which has no
    /// pretrained bundle).
    pub fn hf_repo(&self) -> Option<&'static str> {
        match (self.layers, self.heads, self.dim) {
            (24, 16, 1024) => Some("openai-community/gpt2-medium"),
            (36, 20, 1280) => Some("openai-community/gpt2-large"),
            _ => None,
        }
    }

    /// Effective KV-head count for GQA-aware accessors and shape math.
    ///
    /// MHA — the reference and every named preset (`custom = None`) —
    /// reports `heads`, so every query head has its own key/value head.
    /// A custom spec that opts into GQA (`custom.kv_heads = Some(k)`)
    /// reports the explicit `k`; `Some(1)` is MQA.
    ///
    /// This is the single source of truth for the KV-head count both
    /// the [`Block`] builder and the Lua-facing `Gpt2Handle::meta`
    /// accessor read from — the bridge previously mirrored `heads`
    /// here, which silently misreported GQA models to Lua callers even
    /// though the internal forward pass used the correct value.
    pub fn effective_kv_heads(&self) -> usize {
        self.custom
            .as_ref()
            .and_then(|c| c.kv_heads)
            .unwrap_or(self.heads)
    }
}

/// A single GPT-2 transformer block.
///
/// Pre-LN topology (LN → Attention → residual → LN → MLP → residual)
/// matches nanoGPT and HF GPT-2.
///
/// The four linear projections (`c_attn`, `c_proj`, `mlp_c_fc`,
/// `mlp_c_proj`) are held as [`LinearVariant`] so a subsequent
/// [`Gpt2Model::wrap_lora`] call can replace individual layers with a
/// LoRA wrap without changing the surrounding forward code path.
/// Block / final normalization — the seam [`Gpt2Custom::norm`]
/// switches. Both variants keep the `ln_*` VarBuilder scope, so the
/// weight name is stable (`ln_1.weight` etc.); RMSNorm simply has no
/// bias entry. Both apply through the backward-safe slow shims.
enum Norm {
    Ln(LayerNorm),
    Rms(RmsNorm),
}

impl Norm {
    fn new(kind: NormKind, dim: usize, eps: f64, vs: VarBuilder) -> CandleResult<Self> {
        Ok(match kind {
            NormKind::LayerNorm => Self::Ln(layer_norm(dim, eps, vs)?),
            NormKind::RmsNorm => Self::Rms(rms_norm(dim, eps, vs)?),
        })
    }

    fn apply(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Ln(ln) => apply_slow_layer_norm(ln, x),
            Self::Rms(rn) => apply_slow_rms_norm(rn, x),
        }
    }

    /// LayerNorm accessor for the HF-layout export path (which only
    /// runs on non-custom models — RMSNorm has no HF-GPT-2 layout).
    fn as_layer_norm(&self) -> CandleResult<&LayerNorm> {
        match self {
            Self::Ln(ln) => Ok(ln),
            Self::Rms(_) => Err(candle_core::Error::Msg(
                "export_merged: RMSNorm reached the HF-layout export path (guard bug)".into(),
            )),
        }
    }
}

struct Block {
    ln_1: Norm,
    c_attn: LinearVariant,
    c_proj: LinearVariant,
    ln_2: Norm,
    ff: FeedForward,
    residual: ResidualKind,
    placement: NormPlacement,
    heads: usize,
    /// KV heads (= `heads` for MHA; fewer for GQA / MQA).
    kv_heads: usize,
    head_dim: usize,
}

/// Per-forward positional context threaded from [`Gpt2Model`] into
/// every block: the RoPE cos/sin cache and/or the ALiBi score bias.
/// Both are `None` on the reference (learned `wpe`) and NoPos paths.
struct PosContext<'a> {
    /// `[ctx, head_dim/2]` cos / sin tables ([`build_rope_cache`]).
    rope: Option<(&'a Tensor, &'a Tensor)>,
    /// `[heads, t, t]` additive score bias (already negated).
    alibi: Option<&'a Tensor>,
}

/// The block's feed-forward half — the seam [`Gpt2Config::moe`] /
/// [`Gpt2Custom::act`] switch. `Dense` is the GPT-2-shaped MLP with a
/// configurable activation (`c_fc → act → c_proj`; gated activations
/// add the plain `c_gate` projection and compute
/// `act(c_gate(x)) * c_fc(x)`); `Moe` is the dense mixture-of-experts
/// ([`MoeMlp`], random-init only, LoRA out of scope). LoRA `up` /
/// `down` wrap `c_fc` / `c_proj` on any Dense variant; `c_gate` stays
/// plain (not a LoRA target).
enum FeedForward {
    Dense {
        c_fc: LinearVariant,
        c_gate: Option<candle_nn::Linear>,
        c_proj: LinearVariant,
        act: Activation,
    },
    Moe(MoeMlp),
}

impl Block {
    fn new(cfg: &Gpt2Config, vs: VarBuilder) -> CandleResult<Self> {
        let head_dim = cfg.dim / cfg.heads;
        let custom = cfg.custom.clone().unwrap_or_default();
        let norm_kind = custom.norm;
        let kv_heads = cfg.effective_kv_heads();
        let ln_1 = Norm::new(norm_kind, cfg.dim, cfg.eps, vs.pp("ln_1"))?;
        let resid_stdev = residual_init_stdev(cfg.layers);
        let attn_vs = vs.pp("attn");
        // Fused QKV: `[dim + 2·kv·head_dim, dim]`. MHA (`kv == heads`)
        // reduces to the reference `[3·dim, dim]`; the VarMap name
        // stays `attn.c_attn` either way.
        let qkv_out = cfg.dim + 2 * kv_heads * head_dim;
        let c_attn = gpt2_linear(cfg.dim, qkv_out, INIT_STDEV, attn_vs.pp("c_attn"))?;
        let c_proj = gpt2_linear(cfg.dim, cfg.dim, resid_stdev, attn_vs.pp("c_proj"))?;
        let ln_2 = Norm::new(norm_kind, cfg.dim, cfg.eps, vs.pp("ln_2"))?;
        let ff = match &cfg.moe {
            None => {
                let hidden = custom.mlp_hidden(cfg.dim);
                let mlp_vs = vs.pp("mlp");
                let mlp_c_fc = gpt2_linear(cfg.dim, hidden, INIT_STDEV, mlp_vs.pp("c_fc"))?;
                let c_gate = if custom.act.is_gated() {
                    Some(gpt2_linear(
                        cfg.dim,
                        hidden,
                        INIT_STDEV,
                        mlp_vs.pp("c_gate"),
                    )?)
                } else {
                    None
                };
                let mlp_c_proj = gpt2_linear(hidden, cfg.dim, resid_stdev, mlp_vs.pp("c_proj"))?;
                FeedForward::Dense {
                    c_fc: LinearVariant::Plain(mlp_c_fc),
                    c_gate,
                    c_proj: LinearVariant::Plain(mlp_c_proj),
                    act: custom.act,
                }
            }
            Some(moe) => FeedForward::Moe(MoeMlp::new(cfg, moe, vs.pp("moe"))?),
        };
        Ok(Self {
            ln_1,
            c_attn: LinearVariant::Plain(c_attn),
            c_proj: LinearVariant::Plain(c_proj),
            ln_2,
            ff,
            residual: custom.residual,
            placement: custom.placement,
            heads: cfg.heads,
            kv_heads,
            head_dim,
        })
    }

    /// Replace this block's `Plain` linear projections with LoRA-wrapped
    /// counterparts according to `flags`. Idempotency: a layer already
    /// in the `Lora` variant is left untouched (double-wrap error).
    ///
    /// Callers pass the per-block `VarBuilder` scoped so LoRA parameter
    /// names line up with the block index, e.g. `h.<i>.attn.lora.c_attn`.
    fn wrap_lora(
        &mut self,
        cfg: &LoraConfig,
        flags: WrapFlags,
        vs: VarBuilder,
    ) -> CandleResult<()> {
        let attn_vs = vs.pp("attn");
        if flags.qkv {
            wrap_variant_in_place(&mut self.c_attn, cfg, attn_vs.pp("c_attn"))?;
        }
        if flags.o {
            wrap_variant_in_place(&mut self.c_proj, cfg, attn_vs.pp("c_proj"))?;
        }
        if flags.up || flags.down {
            match &mut self.ff {
                FeedForward::Dense { c_fc, c_proj, .. } => {
                    let mlp_vs = vs.pp("mlp");
                    if flags.up {
                        wrap_variant_in_place(c_fc, cfg, mlp_vs.pp("c_fc"))?;
                    }
                    if flags.down {
                        wrap_variant_in_place(c_proj, cfg, mlp_vs.pp("c_proj"))?;
                    }
                }
                FeedForward::Moe(_) => {
                    // Refuse rather than silently skipping: a caller who
                    // asked for `up` / `down` on a MoE model would
                    // otherwise train fewer parameters than requested.
                    return Err(candle_core::Error::Msg(
                        "wrap_lora: `up` / `down` target the dense MLP; the MoE feed-forward \
                         is out of LoRA scope (attention targets remain available)"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn attention(&self, x: &Tensor, mask: &Tensor, pos: &PosContext<'_>) -> CandleResult<Tensor> {
        // x: [B, T, D]
        let (b, t, _d) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?; // [B, T, D + 2·kv·Dh]
                                           // Split into Q [B,T,H·Dh] / K,V [B,T,kv·Dh]. For MHA
                                           // (kv == heads) this is byte-identical to the reference
                                           // `reshape(b,t,3,H,Dh)` + index split.
        let d = self.heads * self.head_dim;
        let kv_d = self.kv_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, kv_d)?;
        let v = qkv.narrow(D::Minus1, d + kv_d, kv_d)?;
        // [B, T, H, Dh] → [B, H, T, Dh].
        let q = q
            .reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // RoPE rotates Q and K in-place (backward-safe slow shim; the
        // cache is narrowed to the active prefix inside `rope_slow`).
        let (q, k) = match pos.rope {
            Some((cos, sin)) => (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?),
            None => (q, k),
        };

        // GQA: share each KV head across `heads / kv_heads` query
        // heads. `n_rep == 1` (MHA) is a pass-through.
        let n_rep = self.heads / self.kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Scaled dot-product: [B, H, T, T].
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        scores = (scores / scale)?;

        // ALiBi: constant additive bias, applied before the mask (the
        // masked positions get overwritten with -inf below anyway).
        if let Some(bias) = pos.alibi {
            scores = scores.broadcast_add(&bias.unsqueeze(0)?)?;
        }

        // Causal mask: keep positions j <= i (banded when the sliding
        // window is on — see `build_causal_mask`).
        let mask = mask.i((..t, ..t))?; // [T, T]
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?
            .to_dtype(scores.dtype())?
            .broadcast_as(scores.shape())?;
        let mask4 = mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(scores.shape())?;
        scores = mask4.where_cond(&scores, &neg_inf)?;
        // Backward-safe softmax (see `super::softmax_last_dim_slow`):
        // the fused kernel severs the autograd graph, silently zeroing
        // the Q/K gradient (the V path masks it on the fused c_attn).
        let probs = super::softmax_last_dim_slow(&scores)?;

        // [B, H, T, Dh] then merge back to [B, T, D].
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.c_proj.forward(&ctx)
    }

    /// Feed-forward half. Dense returns no byproducts; MoE returns the
    /// unscaled load-balancing aux term and, when `probs_sink` is
    /// supplied, pushes the router probabilities for probing.
    fn feed_forward(
        &self,
        x: &Tensor,
        probs_sink: Option<&mut Vec<Tensor>>,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        match &self.ff {
            FeedForward::Dense {
                c_fc,
                c_gate,
                c_proj,
                act,
            } => {
                // Non-gated: `act(c_fc(x))`. Gated (Shazeer 2020):
                // `act(c_gate(x)) * c_fc(x)` — the activated branch is
                // `c_gate`, the linear branch keeps the `c_fc` name
                // (Llama's `up_proj`). GELU stays the exact form; the
                // HF approximate variant is within 1e-3 for our
                // purposes.
                let h = match (act, c_gate) {
                    (Activation::Gelu, None) => c_fc.forward(x)?.gelu()?,
                    (Activation::Relu, None) => c_fc.forward(x)?.relu()?,
                    (Activation::Silu, None) => c_fc.forward(x)?.silu()?,
                    (Activation::SwiGlu, Some(gate)) => {
                        gate.forward(x)?.silu()?.mul(&c_fc.forward(x)?)?
                    }
                    (Activation::GeGlu, Some(gate)) => {
                        gate.forward(x)?.gelu()?.mul(&c_fc.forward(x)?)?
                    }
                    (act, gate) => {
                        return Err(candle_core::Error::Msg(format!(
                            "gpt2 mlp: activation {act:?} / gate presence {} mismatch \
                             (builder bug)",
                            gate.is_some()
                        )))
                    }
                };
                Ok((c_proj.forward(&h)?, None))
            }
            FeedForward::Moe(moe) => {
                let out = moe.forward(x)?;
                if let Some(sink) = probs_sink {
                    sink.push(out.probs);
                }
                Ok((out.y, Some(out.aux)))
            }
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        pos: &PosContext<'_>,
        probs_sink: Option<&mut Vec<Tensor>>,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        match (self.placement, self.residual) {
            // GPT-2 reference: norm the input, two sequential residual
            // writes.
            (NormPlacement::PreLn, ResidualKind::Sequential) => {
                let n = self.ln_1.apply(x)?;
                let a = self.attention(&n, mask, pos)?;
                let x = (x + a)?;
                let n = self.ln_2.apply(&x)?;
                let (m, aux) = self.feed_forward(&n, probs_sink)?;
                Ok(((x + m)?, aux))
            }
            // GPT-J / PaLM: both halves read the block input; one
            // combined residual write.
            (NormPlacement::PreLn, ResidualKind::Parallel) => {
                let a = self.attention(&self.ln_1.apply(x)?, mask, pos)?;
                let (m, aux) = self.feed_forward(&self.ln_2.apply(x)?, probs_sink)?;
                Ok((((x + a)? + m)?, aux))
            }
            // Original Transformer: norm the residual sum after each
            // sublayer.
            (NormPlacement::PostLn, ResidualKind::Sequential) => {
                let a = self.attention(x, mask, pos)?;
                let x = self.ln_1.apply(&(x + a)?)?;
                let (m, aux) = self.feed_forward(&x, probs_sink)?;
                Ok((self.ln_2.apply(&(x + m)?)?, aux))
            }
            // Rejected in `Gpt2Custom::validate`; reaching it here is a
            // builder bug, not a user error.
            (NormPlacement::PostLn, ResidualKind::Parallel) => Err(candle_core::Error::Msg(
                "gpt2 block: Post-LN × parallel residual survived validation (builder bug)".into(),
            )),
        }
    }
}

/// Why a table row could not be wrapped in a [`CondIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CondIndexError {
    /// The row is outside the table the caller named.
    #[error(
        "condition row {row} is outside a {slots}-row conditioning table; \
         rows are numbered 0..{slots}"
    )]
    OutOfRange {
        /// Row the caller asked for.
        row: u32,
        /// Rows the table holds.
        slots: usize,
    },
    /// The caller named a table with no rows, which selects nothing.
    #[error("a conditioning table of zero rows has no row to select")]
    NoSlots,
}

/// How a forward pass names the condition it adds to the residual
/// stream.
///
/// Internal to the forward path: the public entry points
/// ([`Gpt2Model::forward_conditioned`],
/// [`Gpt2Model::forward_conditioned_groups`],
/// [`Gpt2Model::forward_cond_weighted`]) each build one variant, so a
/// caller never chooses between them by passing a tag.
enum CondInput<'a> {
    /// Table rows, `per_row` of them for each row of the batch, summed
    /// before the addition.
    Rows {
        /// `batch * per_row` indices, row-major.
        conds: &'a [CondIndex],
        /// How many of them belong to each row of the batch.
        per_row: usize,
    },
    /// One coefficient per table row; the combination they describe is
    /// added to every row of the batch.
    Weights(&'a [f32]),
}

/// A row of a model's conditioning table.
///
/// The point of the wrapper is that the number inside cannot be
/// supplied casually. Rows are `0..cond_slots`, and a condition that
/// means something to a caller almost always has a *second* numbering —
/// most obviously a token id, when the condition also appears in the
/// sequence. The two ranges are different and they overlap, so an id
/// passed where a row is wanted selects a real but wrong row: every
/// shape agrees, the forward pass succeeds, and the model is
/// conditioned on something else with nothing downstream able to report
/// it.
///
/// So the constructor takes the table size as well as the row and
/// checks one against the other. A caller holding an id from some other
/// numbering does not have a slot count that makes it valid, and the
/// mistake stops here rather than at a number that happens to be in
/// range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondIndex(u32);

impl CondIndex {
    /// Wrap `row` after checking it against a table of `slots` rows.
    ///
    /// `slots` is [`Gpt2Custom::cond_slots`] of the model the index is
    /// destined for. Passing it is what makes the check possible, and
    /// asking for it is what makes a caller state which table they mean.
    ///
    /// # Errors
    ///
    /// [`CondIndexError::NoSlots`] when `slots` is zero, and
    /// [`CondIndexError::OutOfRange`] when `row >= slots`.
    pub fn new(row: u32, slots: usize) -> Result<Self, CondIndexError> {
        if slots == 0 {
            return Err(CondIndexError::NoSlots);
        }
        if row as usize >= slots {
            return Err(CondIndexError::OutOfRange { row, slots });
        }
        Ok(Self(row))
    }

    /// The row this index selects.
    pub fn row(self) -> u32 {
        self.0
    }
}

/// The ids a model may pick from at each position of a batch, in the
/// form [`Gpt2Model::forward_allowed`] reads them.
///
/// # What the indices mean
///
/// `sets[r][p]` is the set the model's prediction **for input position
/// `p` of row `r`** is drawn from — what is available once it has
/// consumed input position `p`, not what was available when position
/// `p` itself was produced. Under the usual next-token shift that is
/// the same set the loss mask uses for target `p`
/// ([`crate::train::allowed_logit_mask`]), which is why
/// [`crate::train::allowed_input_sets`] builds both from one list at
/// one offset: given as input on one side, used to strike out the
/// alternatives on the other.
///
/// Getting that offset wrong shifts every set by one position and
/// leaves every shape agreeing, so the in-crate producer is the
/// training-side helper rather than this constructor. A caller that
/// builds its own — one recovering the sets from the state it is
/// standing on — passes it here.
///
/// # Layout
///
/// The lists are padded to the longest set **in this batch** rather
/// than to any constant: a constant would have to be a bound on how
/// many ids a position can offer, and one chosen slightly too small
/// truncates a set with nothing to say so. The padding carries no
/// weight, so the width it reaches changes the arithmetic nowhere.
///
/// The cost is paid in the forward pass rather than here: the ids are
/// `rows × width × k` u32, but the gather they drive materialises
/// `rows × width × k × dim` floats before the mean collapses it.
#[derive(Debug, Clone)]
pub struct AllowedSets {
    /// `[rows, width, k]` u32 — each position's ids, padded with `0`.
    ids: Tensor,
    /// `[rows, width, k]` f32 — `1 / count` on a real entry and `0` on
    /// padding, so the mean is a weighted sum and the division by a
    /// count that could be zero never happens.
    weights: Tensor,
    /// Largest id in `ids`, kept from construction so the range check
    /// in the forward pass costs no device round trip. `0` when every
    /// id present is `0`, which is in range for every vocabulary this
    /// builds a model over.
    max_id: u32,
}

impl AllowedSets {
    /// Build from one set per position of every row.
    ///
    /// Every row must hold the same number of positions, which is the
    /// model's input width. See the type's documentation for what
    /// `sets[r][p]` has to be.
    ///
    /// # Errors
    ///
    /// As [`Self::window`], which this delegates to with the full width.
    pub fn new(sets: &[Vec<Vec<u32>>], device: &Device) -> CandleResult<Self> {
        let width = sets.first().map(Vec::len).unwrap_or(0);
        Self::window(sets, 0, width, device)
    }

    /// Build from a window of each row's sets: position `p` of the
    /// result reads `sets[r][from + p]`.
    ///
    /// # The empty set, per position and per batch
    ///
    /// A **position** past the end of its row gets the empty set, and
    /// that is a legitimate thing for a producer to say: it is the
    /// padding past the end of a row, and the forward pass reads it as
    /// the zero vector rather than dividing by a count of nothing.
    ///
    /// A **batch** in which every position of every row is empty is a
    /// different statement and is refused here. Such an input would add
    /// the zero vector everywhere, so the model handed it would answer
    /// exactly as the same model with this channel deleted — a
    /// checkpoint trained with the channel, scored as though it had
    /// none, every number well-formed.
    ///
    /// The guard in [`Gpt2Model::forward_allowed`] tests that an
    /// `AllowedSets` was supplied rather than that it carries ids, so
    /// it would let that through and return numbers a caller cannot
    /// tell from a correct run. The refusal therefore lives here, where
    /// every producer passes.
    ///
    /// # Errors
    ///
    /// `sets` is empty, `width` is zero, a row holds a different number
    /// of positions than row 0 **before** the window is applied (rows
    /// of differing lengths mean the batch and its sets were built from
    /// different row lists, which no windowing makes safe), or no
    /// position in the window holds an id.
    pub fn window(
        sets: &[Vec<Vec<u32>>],
        from: usize,
        width: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let rows = sets.len();
        if rows == 0 {
            return Err(candle_core::Error::Msg(
                "allowed sets: no rows, so there is nothing to say is allowed".into(),
            ));
        }
        if width == 0 {
            return Err(candle_core::Error::Msg(
                "allowed sets: zero positions per row".into(),
            ));
        }
        let full = sets[0].len();
        for (r, row) in sets.iter().enumerate() {
            if row.len() != full {
                return Err(candle_core::Error::Msg(format!(
                    "allowed sets: row {r} holds {} position(s) and row 0 holds {full}",
                    row.len()
                )));
            }
        }

        // The longest set in the window. Zero means no position of any
        // row holds an id, which is refused rather than built: every
        // weight would be zero, the channel would contribute the
        // additive identity everywhere, and the model would answer
        // exactly as the same model with no table — the one outcome
        // nothing further along can tell from a correct run.
        let k = sets
            .iter()
            .flat_map(|row| row.iter().skip(from).take(width))
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        if k == 0 {
            return Err(candle_core::Error::Msg(format!(
                "allowed sets: no position of any row holds an id ({rows} row(s), {width} \
                 position(s) from {from}), so this input would add nothing anywhere and the \
                 model would answer as though it had no allowed-id channel at all; a producer \
                 with nothing to say about a batch cannot say it here"
            )));
        }

        let mut ids = vec![0u32; rows * width * k];
        let mut weights = vec![0f32; rows * width * k];
        let mut max_id = 0u32;
        for (r, row) in sets.iter().enumerate() {
            for p in 0..width {
                let Some(set) = row.get(from + p) else {
                    continue;
                };
                if set.is_empty() {
                    continue;
                }
                let base = (r * width + p) * k;
                // Every real entry carries the same share, so the
                // weighted sum below is the mean over the true count
                // and the padding — weight zero — is not counted.
                let share = 1.0 / set.len() as f32;
                for (j, id) in set.iter().enumerate() {
                    ids[base + j] = *id;
                    weights[base + j] = share;
                    max_id = max_id.max(*id);
                }
            }
        }

        Ok(Self {
            ids: Tensor::from_vec(ids, (rows, width, k), device)?,
            weights: Tensor::from_vec(weights, (rows, width, k), device)?,
            max_id,
        })
    }

    /// Rows this covers.
    pub fn rows(&self) -> usize {
        self.ids.dims()[0]
    }

    /// Positions per row.
    pub fn width(&self) -> usize {
        self.ids.dims()[1]
    }

    /// Entries the padded lists were widened to — the longest set in
    /// the batch, and never zero, since a batch holding no ids at all
    /// is refused by [`Self::window`].
    pub fn widest(&self) -> usize {
        self.ids.dims()[2]
    }

    /// The padded id table, `[rows, width, widest]`. Test-only: the
    /// padding convention is part of what the mean arithmetic rests on,
    /// and asserting on it directly is cheaper than inferring it from a
    /// forward pass.
    #[cfg(test)]
    pub(crate) fn ids(&self) -> &Tensor {
        &self.ids
    }

    /// The per-entry weights, `[rows, width, widest]` — `1 / count` on
    /// a real entry, `0` on padding. Test-only, for the reason
    /// [`Self::ids`] is.
    #[cfg(test)]
    pub(crate) fn weights(&self) -> &Tensor {
        &self.weights
    }
}

/// GPT-2 forward-only model.
///
/// Constructed via [`Gpt2Model::new`] (random init from a [`VarBuilder`])
/// or [`Gpt2Model::from_pretrained`] (HuggingFace warm-start, design
/// §12 Q5). Training loops live in a follow-up stage; this stage
/// ships the forward path only.
pub struct Gpt2Model {
    wte: Embedding,
    /// Learned positional embedding. `None` when [`Gpt2Custom::pos`]
    /// is a non-learned kind (RoPE / ALiBi / NoPos) — the `wpe` Var
    /// then never exists in the VarMap.
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: Norm,
    /// Cached causal mask (`1` kept, `0` masked) sized to
    /// [`Gpt2Config::ctx`] so per-forward mask allocation is avoided.
    /// Banded when the sliding window is on.
    causal_mask: Tensor,
    /// RoPE cos / sin cache (`[ctx, head_dim/2]` each), present iff
    /// `pos == Rope`.
    rope: Option<(Tensor, Tensor)>,
    /// ALiBi constants, present iff `pos == Alibi`: per-head slopes
    /// `[heads, 1, 1]` and the signed distance table `i - j`
    /// `[ctx, ctx]`. Combined into the `[heads, t, t]` score bias per
    /// forward (both are constants — no Vars, no gradient tracking).
    alibi: Option<(Tensor, Tensor)>,
    /// Untied LM head weight (`[vocab, dim]`, VarMap name
    /// `lm_head.weight`), present iff `untied_head`. `None` = head
    /// tied to `wte` (reference).
    lm_head: Option<Tensor>,
    /// Conditioning table (`[cond_slots, dim]`, VarMap name
    /// `cond_wte.weight`), present iff [`Gpt2Custom::cond_slots`] is
    /// set. Read only by [`Self::forward_conditioned`] and its grouped
    /// sibling.
    cond_wte: Option<Embedding>,
    /// Allowed-id table (`[vocab, dim]`, VarMap name
    /// `allowed_wte.weight`), present iff [`Gpt2Custom::allowed_input`].
    /// Read only by [`Self::forward_allowed`].
    allowed_wte: Option<Embedding>,
    cfg: Gpt2Config,
}

impl Gpt2Model {
    /// Build a fresh GPT-2 model (random parameters) from a
    /// [`VarBuilder`]. Weight naming matches the HF `openai-community/gpt2*`
    /// convention so a subsequent [`Gpt2Model::from_pretrained`] can
    /// load into the same shape.
    pub fn new(cfg: &Gpt2Config, vs: VarBuilder) -> CandleResult<Self> {
        if !cfg.dim.is_multiple_of(cfg.heads) {
            return Err(candle_core::Error::Msg(format!(
                "gpt2: dim {} must be divisible by heads {}",
                cfg.dim, cfg.heads
            )));
        }
        if let Some(moe) = &cfg.moe {
            moe.validate()?;
        }
        let custom = cfg.custom.clone().unwrap_or_default();
        if let Some(spec) = &cfg.custom {
            spec.validate()?;
            if cfg.moe.is_some()
                && (spec.act != Activation::Gelu || !matches!(spec.mlp_ratio, 0 | 4))
            {
                // Every other custom axis (norm / placement / residual /
                // pos / GQA / window / untied head) composes with MoE;
                // only the dense-MLP knobs address a module the MoE
                // seam replaces.
                return Err(candle_core::Error::Msg(
                    "gpt2: `custom.act` / `custom.mlp_ratio` do not apply to the MoE \
                     experts — keep them at the reference (Gelu / 4) when combining \
                     `custom` with `moe`"
                        .into(),
                ));
            }
            if let Some(kv) = spec.kv_heads {
                if !cfg.heads.is_multiple_of(kv) {
                    return Err(candle_core::Error::Msg(format!(
                        "gpt2: heads {} must be divisible by kv_heads {kv}",
                        cfg.heads
                    )));
                }
            }
            if spec.pos == PosKind::Rope && !(cfg.dim / cfg.heads).is_multiple_of(2) {
                return Err(candle_core::Error::Msg(format!(
                    "gpt2: RoPE needs an even head_dim (got {})",
                    cfg.dim / cfg.heads
                )));
            }
        }
        let wte = gpt2_embedding(cfg.vocab, cfg.dim, vs.pp("wte"))?;
        let wpe = match custom.pos {
            PosKind::Learned => Some(gpt2_embedding(cfg.ctx, cfg.dim, vs.pp("wpe"))?),
            PosKind::Rope | PosKind::Alibi | PosKind::NoPos => None,
        };
        let rope = match custom.pos {
            // TinyLlama's canonical cache; base 10000 is the standard
            // GPT-NeoX / Llama rope_theta and is not a knob here (a
            // theta sweep would be a trainer-side probe, not an arch
            // axis).
            PosKind::Rope => Some(build_rope_cache(
                cfg.ctx,
                cfg.dim / cfg.heads,
                10_000.0,
                cfg.dtype,
                &cfg.device,
            )?),
            _ => None,
        };
        let alibi = match custom.pos {
            PosKind::Alibi => Some(build_alibi_consts(
                cfg.heads,
                cfg.ctx,
                cfg.dtype,
                &cfg.device,
            )?),
            _ => None,
        };
        let h_vs = vs.pp("h");
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block::new(cfg, h_vs.pp(i.to_string()))?);
        }
        let ln_f = Norm::new(custom.norm, cfg.dim, cfg.eps, vs.pp("ln_f"))?;
        let lm_head = if custom.untied_head {
            Some(vs.pp("lm_head").get_with_hints(
                (cfg.vocab, cfg.dim),
                "weight",
                Init::Randn {
                    mean: 0.0,
                    stdev: INIT_STDEV,
                },
            )?)
        } else {
            None
        };
        // Conditioning table. Small (`slots × dim`) and read by nothing
        // but the conditioning addition, which is what keeps it out of
        // the tug-of-war `wte` would be in — see
        // [`Gpt2Custom::cond_slots`].
        let cond_wte = match custom.cond_slots {
            Some(slots) => Some(gpt2_embedding(
                slots,
                cfg.dim,
                vs.pp(super::custom::COND_TABLE_PREFIX),
            )?),
            None => None,
        };
        // Allowed-id table. One row per vocabulary entry, because its
        // entries are ids — and separate from `wte` for the reason
        // [`Gpt2Custom::allowed_input`] records: the set added here
        // holds the id that is about to be the target.
        let allowed_wte = if custom.allowed_input {
            Some(gpt2_embedding(
                cfg.vocab,
                cfg.dim,
                vs.pp(super::custom::ALLOWED_TABLE_PREFIX),
            )?)
        } else {
            None
        };
        let causal_mask = build_causal_mask(cfg.ctx, custom.window, &cfg.device, cfg.dtype)?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            causal_mask,
            rope,
            alibi,
            lm_head,
            cond_wte,
            allowed_wte,
            cfg: cfg.clone(),
        })
    }

    /// Return the configuration this model was built from.
    pub fn config(&self) -> &Gpt2Config {
        &self.cfg
    }

    /// Load model weights from an on-disk safetensors bundle whose
    /// key layout matches the Var names `cfg` declares — the HF GPT-2
    /// convention for a reference config (same layout that
    /// [`Gpt2Model::from_pretrained`] downloads and that
    /// [`super::lora::MergeableLora::export_merged`] emits), and the
    /// spec-dependent superset for a `custom` one.
    ///
    /// This is the plain-load path used by (a) the merged-bundle
    /// parity oracle in `tests/merged_export_parity_gpt2.rs`,
    /// (b) load-side integration that recognises
    /// `training_path == "merged"` and dispatches here instead of
    /// re-wrapping the model, and (c) reloading a Card-store bundle.
    ///
    /// `custom` configs are accepted here — unlike in
    /// [`Gpt2Model::from_pretrained`]. Card-store bundles are written
    /// by `VarMap::save`, so they carry exactly the Vars the config
    /// declares (gated `mlp.c_gate`, an untied `lm_head.weight`, no
    /// `wpe` for a non-learned position kind, no LayerNorm biases
    /// under RMSNorm — all named by the same `get_with_hints` calls
    /// [`Self::new`] makes). RoPE / ALiBi / causal-mask tables are
    /// recomputed by [`Self::new`] and never live in the bundle.
    /// The HF-hub path still refuses `custom` because a hub bundle
    /// really does carry only the reference layout.
    ///
    /// # Errors
    ///
    /// `PretrainedError::Load` on safetensors parse failure or
    /// weight-name mismatch against the model shape, and for MoE
    /// configs, whose `h.<i>.moe.*` layout no bundle currently
    /// carries.
    pub fn from_safetensors_file(
        cfg: &Gpt2Config,
        path: &std::path::Path,
    ) -> Result<Self, PretrainedError> {
        if cfg.moe.is_some() {
            return Err(PretrainedError::Load(
                "MoE configs are random-init only; no safetensors bundle carries the \
                 h.<i>.moe.* layout"
                    .into(),
            ));
        }
        // SAFETY: same discipline as `from_pretrained` — the file
        // must not be concurrently truncated while the mmap is
        // active. Callers hold the mmap for the lifetime of this
        // call.
        let vs = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&path.to_path_buf()),
                cfg.dtype,
                &cfg.device,
            )
            .map_err(|e| PretrainedError::Load(e.to_string()))?
        };
        Self::new(cfg, vs).map_err(|e| PretrainedError::Load(e.to_string()))
    }

    /// Load pretrained GPT-2 weights from HuggingFace on first use and
    /// cache the safetensors bundle at `cache_dir/base/<preset>.safetensors`.
    ///
    /// The config selects the HF repo via [`Gpt2Config::hf_repo`]; a
    /// custom config that is not one of the shipped presets returns an
    /// error rather than downloading a mismatched bundle.
    ///
    /// # Errors
    ///
    /// Fails on: unknown preset (no HF repo), network / cache IO error,
    /// safetensors parse error, or a weight-name mismatch between the
    /// downloaded bundle and the model shape.
    pub fn from_pretrained(
        variant: &str,
        cfg: &Gpt2Config,
        cache_dir: &std::path::Path,
    ) -> Result<Self, PretrainedError> {
        if cfg.moe.is_some() {
            return Err(PretrainedError::Load(
                "MoE configs are random-init only; pretrained GPT-2 bundles have no \
                 router / expert weights"
                    .into(),
            ));
        }
        if cfg.custom.is_some() {
            return Err(PretrainedError::Load(
                "custom configs are random-init only; pretrained GPT-2 bundles carry \
                 the reference architecture"
                    .into(),
            ));
        }
        let repo = cfg
            .hf_repo()
            .ok_or_else(|| PretrainedError::UnknownPreset(variant.to_string()))?;

        // Cache dir: `<cache_dir>/base/<repo-basename>.safetensors`.
        let base_dir = cache_dir.join("base");
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| PretrainedError::CacheIo(format!("mkdir {:?}: {e}", base_dir)))?;
        let repo_leaf = repo.rsplit('/').next().unwrap_or(repo);
        let cache_path = base_dir.join(format!("{repo_leaf}.safetensors"));

        if !cache_path.exists() {
            tracing::info!(
                target: "algocline_nn::arch::gpt2",
                repo,
                cache = %cache_path.display(),
                "downloading gpt-2 pretrained weights"
            );
            crate::hub::download_to(repo, "model.safetensors", &cache_path)
                .map_err(|e| PretrainedError::Download(e.to_string()))?;
        }

        // Load through candle's mmap-safetensors VarBuilder.
        // SAFETY: candle exposes this constructor as unsafe because the
        // caller must ensure the mmap-backed file is not concurrently
        // truncated. The cache path is only written once above under a
        // first-use guard; subsequent readers hold the mmap for the
        // lifetime of this call.
        let vs = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&cache_path),
                cfg.dtype,
                &cfg.device,
            )
            .map_err(|e| PretrainedError::Load(e.to_string()))?
        };
        Self::new(cfg, vs).map_err(|e| PretrainedError::Load(e.to_string()))
    }

    /// Forward pass. Input `xs` is `[batch, seq]` of `u32` token ids;
    /// output is `[batch, seq, vocab]` — the raw logits (softmax is left
    /// to the training loss / sampling caller).
    ///
    /// # Errors
    ///
    /// `seq > ctx`, and — since a model built with
    /// [`Gpt2Custom::allowed_input`] reads an allowed-id set at every
    /// position — a model carrying `allowed_wte` handed none. See
    /// [`Self::forward_allowed`] for why that is refused rather than
    /// run with the channel absent.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_inner(xs, None, None, None)
            .map(|(logits, _)| logits)
    }

    /// Forward pass whose condition is a **weighted combination** of the
    /// whole conditioning table rather than one of its rows.
    ///
    /// `weights` holds one coefficient per row of `cond_wte`; the vector
    /// `Σ weights[i] · cond_wte[i]` is added at every position, at the
    /// same point [`Self::forward_conditioned`] adds a single row. One
    /// combination covers the whole batch — the weights describe *what
    /// to condition on*, and a batch conditioned row by row is what
    /// [`Self::forward_conditioned`] already expresses.
    ///
    /// A one-hot `weights` reproduces [`Self::forward_conditioned`] on
    /// the row it selects, so this is a widening of that entry point
    /// rather than a second convention. The two reach that vector by
    /// different kernels — the whole table gathered, scaled and reduced,
    /// against the one row gathered — so they agree to within
    /// floating-point reassociation rather than bit for bit.
    ///
    /// # Why the weights are not normalised
    ///
    /// Rescaling them here would silently change what the caller asked
    /// for. Coefficients summing to more than one are a deliberate use
    /// of this operation (an amplified condition), coefficients summing
    /// to less are a damped one, and neither is distinguishable from a
    /// caller error by anything this function can see. So the numbers
    /// are used as given and their meaning stays with whoever chose
    /// them.
    ///
    /// The one combination that is refused is all-zero: it adds the zero
    /// vector, which is exactly what a model carrying this table does
    /// when nobody feeds it — the state the channel exists to keep the
    /// model out of. Refusing it here means "condition on nothing" has
    /// to be said with [`Self::forward`], where it is visible.
    ///
    /// # Training
    ///
    /// The training entry points take table rows; nothing trains a
    /// fractional combination. What the weights interpolate is therefore
    /// the *embedding* space, and whether the model's behaviour
    /// interpolates with it is a question about the trained model, not a
    /// property of this call.
    ///
    /// # Errors
    ///
    /// - The model was built without a conditioning table.
    /// - `weights.len()` is not the table's row count.
    /// - Any weight is not finite **once narrowed to the model's
    ///   dtype**: a NaN or infinity spreads through the sum to every
    ///   position of the residual stream, and a weight finite as an
    ///   `f32` can still be an infinity as an `f16`.
    /// - Every weight is zero once narrowed (see above) — which
    ///   coefficients far below the model dtype's smallest subnormal
    ///   also reach.
    /// - Anything [`Self::forward`] rejects, `seq > ctx` included.
    pub fn forward_cond_weighted(&self, xs: &Tensor, weights: &[f32]) -> CandleResult<Tensor> {
        self.forward_inner(xs, Some(CondInput::Weights(weights)), None, None)
            .map(|(logits, _)| logits)
    }

    /// Forward pass with a condition embedding added at **every**
    /// position rather than carried by a token at the front of the row.
    ///
    /// `conds` holds one [`CondIndex`] per row of `xs`;
    /// `cond_wte[conds[i]]` is added to row `i` at every position, next
    /// to where the learned positional embedding is added. Output shape
    /// is unchanged. Requires [`Gpt2Custom::cond_slots`] to have been
    /// set when the model was built.
    ///
    /// # Why the condition is an argument
    ///
    /// The alternative — read it out of a fixed position in `xs` — does
    /// not survive a row longer than `ctx`: whatever windowing the
    /// caller applies can move an ordinary token into that slot, and
    /// the model would condition on it with no observable difference
    /// from a real condition. As an argument there is no position for
    /// windowing to disturb.
    ///
    /// # Errors
    ///
    /// - The model was built without a conditioning table.
    /// - `conds.len()` is not `batch`.
    /// - Any index is outside `[0, cond_slots)` — reachable when the
    ///   index was built against a larger table than this model has,
    ///   which is a crossed pair rather than a typo.
    /// - Anything [`Self::forward`] rejects, `seq > ctx` included.
    pub fn forward_conditioned(&self, xs: &Tensor, conds: &[CondIndex]) -> CandleResult<Tensor> {
        self.forward_inner(xs, Some(CondInput::Rows { conds, per_row: 1 }), None, None)
            .map(|(logits, _)| logits)
    }

    /// [`Self::forward_conditioned`] with `per_row` conditions per row,
    /// whose table rows are **summed** before the addition.
    ///
    /// `conds` holds `batch * per_row` indices, row-major: row `i`'s
    /// conditions are `conds[i*per_row .. (i+1)*per_row]`. With
    /// `per_row == 1` this is exactly [`Self::forward_conditioned`].
    ///
    /// # Why a sum
    ///
    /// The composition being expressed is additive: each slot
    /// contributes its own vector to the residual stream at every
    /// position, next to where a single condition's vector already
    /// goes, and nothing downstream distinguishes "one vector that is a
    /// sum" from "two additions". That makes the operation
    /// permutation-invariant in the slots — `[a, b]` and `[b, a]` are
    /// the same forward — which is a property of the mechanism and not
    /// a claim that the *model* treats the slots symmetrically: what
    /// each row's vector comes to mean is the training corpus's
    /// business.
    ///
    /// What this is **not**: a per-slot table. Every index still points
    /// into the one `cond_wte`, so the caller owns the convention of
    /// which rows belong to which slot. Two indices of the same slot
    /// are not refused here — the forward cannot know the grouping —
    /// and a caller that wants that refusal puts it where the grouping
    /// is known.
    ///
    /// # Errors
    ///
    /// - `per_row` is zero (pass [`Self::forward`] for an
    ///   unconditioned batch instead of an empty condition list).
    /// - `conds.len()` is not `batch * per_row`.
    /// - Anything [`Self::forward_conditioned`] refuses.
    pub fn forward_conditioned_groups(
        &self,
        xs: &Tensor,
        conds: &[CondIndex],
        per_row: usize,
    ) -> CandleResult<Tensor> {
        if per_row == 0 {
            return Err(candle_core::Error::Msg(
                "gpt2 forward: per_row must be ≥ 1; an unconditioned batch goes through \
                 `forward` rather than through an empty condition list"
                    .into(),
            ));
        }
        self.forward_inner(xs, Some(CondInput::Rows { conds, per_row }), None, None)
            .map(|(logits, _)| logits)
    }

    /// Forward pass with the ids allowed at each position added to the
    /// residual stream there.
    ///
    /// `allowed` holds one set per position of every row; the mean of
    /// `allowed_wte` over that set is added where the positional
    /// embedding is added. Output shape is unchanged. Requires
    /// [`Gpt2Custom::allowed_input`] to have been set when the model
    /// was built.
    ///
    /// # What this is for
    ///
    /// A model trained on sequences alone has to infer the rules of a
    /// constrained id space from them, and a decoder that walks its
    /// ranking against the allowed set discards every bit of that work.
    /// Handing the set over spends that share of the objective on the
    /// question that survives to inference — which of the available ids
    /// to pick.
    ///
    /// # The mean, and the empty set
    ///
    /// A set is summarised by the mean of its rows rather than the sum,
    /// so a position offering forty ids and one offering four arrive at
    /// the same scale. Exactly so in `F32`. In a lower-precision dtype
    /// it is approximate: the shares are cast to the model's dtype
    /// here rather than held in it, and `BF16` carries 8 significand
    /// bits, so `1 / count` need not be representable — it is for a
    /// count that is a power of two and rounds by up to about `2^-8`
    /// otherwise. Nothing here corrects for that; a run that cares
    /// would have to.
    ///
    /// A set with **no** ids contributes the zero vector: the mean over
    /// zero elements is undefined, and rather than clamp a count the
    /// weights are simply zero and the sum is empty, so the undefined
    /// division never happens. That case is not hypothetical — it is
    /// every padded position past the end of a row, and a NaN there
    /// would take the whole batch with it. This much is exact in every
    /// dtype: zero times anything is zero.
    ///
    /// The zero vector is the additive identity of the thing being
    /// added, so such a position sees this channel contribute nothing,
    /// which is the closest available reading of "nothing is known to
    /// be available here". A whole input of such positions is a
    /// different matter, and [`AllowedSets::window`] refuses to build
    /// one.
    ///
    /// # Errors
    ///
    /// - The model was built without an allowed-id table.
    /// - `allowed` does not cover exactly `[batch, seq]`.
    /// - Any id is outside the vocabulary.
    /// - Anything [`Self::forward`] rejects, `seq > ctx` included.
    pub fn forward_allowed(&self, xs: &Tensor, allowed: &AllowedSets) -> CandleResult<Tensor> {
        self.forward_inner(xs, None, Some(allowed), None)
            .map(|(logits, _)| logits)
    }

    /// Forward pass that also returns the summed MoE load-balancing
    /// aux term (unscaled — multiply by [`MoeConfig::alpha`] when
    /// composing the total loss). `None` on a dense (non-MoE) model,
    /// so existing callers of [`Self::forward`] see no change.
    pub fn forward_with_aux(&self, xs: &Tensor) -> CandleResult<(Tensor, Option<Tensor>)> {
        self.forward_inner(xs, None, None, None)
    }

    /// Probe variant of [`Self::forward_with_aux`] that additionally
    /// returns each MoE layer's router probabilities `[B, S, E]`
    /// (bottom layer first; empty on a dense model). Backing for
    /// `examples/moe_router_probe.rs`'s utilization / entropy
    /// observations — not a training API.
    pub fn forward_with_router_probs(
        &self,
        xs: &Tensor,
    ) -> CandleResult<(Tensor, Option<Tensor>, Vec<Tensor>)> {
        let mut probs = Vec::new();
        let (logits, aux) = self.forward_inner(xs, None, None, Some(&mut probs))?;
        Ok((logits, aux, probs))
    }

    fn forward_inner(
        &self,
        xs: &Tensor,
        cond: Option<CondInput<'_>>,
        allowed: Option<&AllowedSets>,
        mut probs_sink: Option<&mut Vec<Tensor>>,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (b, t) = xs.dims2()?;
        if t > self.cfg.ctx {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: seq {t} exceeds ctx {}",
                self.cfg.ctx
            )));
        }
        let tok_emb = self.wte.forward(xs)?; // [B, T, D]
        let mut h = match &self.wpe {
            Some(wpe) => {
                let pos_ids = Tensor::arange(0u32, t as u32, xs.device())?; // [T]
                let pos_emb = wpe.forward(&pos_ids)?; // [T, D]
                let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(tok_emb.shape())?;
                (tok_emb + pos_emb)?
            }
            // RoPE / ALiBi position information enters inside the
            // attention; NoPos has none by design.
            None => tok_emb,
        };
        // The condition, if the caller passed one. Added here, beside
        // the positional embedding, so it is present at every position
        // instead of decaying with distance from a token at the front.
        if let Some(cond) = cond {
            // `[B, 1, D]` from a row selection, `[1, 1, D]` from a
            // weighted combination — both broadcast over the sequence,
            // and the second over the batch as well.
            let cond_emb = match cond {
                CondInput::Rows { conds, per_row } => {
                    self.condition_embedding(conds, b, per_row)?
                }
                CondInput::Weights(weights) => self.weighted_condition_embedding(weights)?,
            };
            h = h.broadcast_add(&cond_emb)?;
        }
        // The allowed-id input, beside the condition and for the same
        // reason: it belongs at every position, because every position
        // has its own set of available ids.
        //
        // Both disagreements are refused. A model that carries the
        // table was trained with this channel at every position, so
        // running it without one runs a model in a state it never
        // trained in — and its output would look no different. The
        // mirror, sets handed to a model with no table, is a caller
        // that believes it is doing something it is not.
        match (&self.allowed_wte, allowed) {
            (Some(table), Some(sets)) => {
                let allowed_emb = self.allowed_embedding(table, sets, b, t)?; // [B, T, D]
                h = (h + allowed_emb)?;
            }
            (Some(_), None) => {
                return Err(candle_core::Error::Msg(
                    "gpt2 forward: this model was built with an allowed-id table and reads one \
                     at every position; running it without one drops that channel entirely — \
                     pass the allowed ids to forward_allowed"
                        .into(),
                ))
            }
            (None, Some(_)) => {
                return Err(candle_core::Error::Msg(
                    "gpt2 forward: allowed ids were supplied to a model with no allowed-id \
                     table; build it with `custom.allowed_input = true` to use forward_allowed"
                        .into(),
                ))
            }
            (None, None) => {}
        }
        // ALiBi score bias for the active prefix: [H, t, t] =
        // -slopes ⊙ (i - j). Constant tensors — no gradient tracking.
        let alibi_bias = match &self.alibi {
            Some((slopes, dist)) => {
                let dist_t = dist.i((..t, ..t))?.unsqueeze(0)?; // [1, t, t]
                Some(slopes.broadcast_mul(&dist_t)?.neg()?)
            }
            None => None,
        };
        let pos_ctx = PosContext {
            rope: self.rope.as_ref().map(|(c, s)| (c, s)),
            alibi: alibi_bias.as_ref(),
        };
        let mut aux_sum: Option<Tensor> = None;
        for block in &self.blocks {
            let (next, aux) =
                block.forward(&h, &self.causal_mask, &pos_ctx, probs_sink.as_deref_mut())?;
            h = next;
            if let Some(a) = aux {
                aux_sum = Some(match aux_sum {
                    None => a,
                    Some(acc) => (acc + a)?,
                });
            }
        }
        let h = self.ln_f.apply(&h)?; // [B, T, D]
                                      // LM head: tied reuses wte; untied has its own Var.
        let w = match &self.lm_head {
            Some(w) => w,
            None => self.wte.embeddings(), // [V, D]
        };
        let logits = h.broadcast_matmul(&w.t()?)?; // [B, T, V]
        debug_assert_eq!(logits.dims(), &[b, t, self.cfg.vocab]);
        Ok((logits, aux_sum))
    }

    /// The per-row condition vectors, shaped `[batch, 1, dim]` so they
    /// broadcast across the sequence.
    ///
    /// The indices arrive as a host slice rather than as a `Tensor`:
    /// every caller has them on the host anyway, and it makes the arity
    /// and range checks free. Range in particular has to be checked
    /// here — candle's CPU backend rejects an out-of-range
    /// `index_select`, but leaving it to the backend would mean the
    /// accelerator builds decide whether a bad index is an error or a
    /// silently conditioned run.
    fn condition_embedding(
        &self,
        conds: &[CondIndex],
        batch: usize,
        per_row: usize,
    ) -> CandleResult<Tensor> {
        let table = self.cond_wte.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "gpt2 forward: this model has no conditioning table; build it with \
                 `custom.cond_slots = Some(n)` to use forward_conditioned"
                    .into(),
            )
        })?;
        if conds.len() != batch * per_row {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: {} condition index/indices for a batch of {batch} at {per_row} \
                 per row; pass exactly {}",
                conds.len(),
                batch * per_row
            )));
        }
        let slots = self
            .cfg
            .custom
            .as_ref()
            .and_then(|c| c.cond_slots)
            .unwrap_or(0);
        if let Some(bad) = conds.iter().find(|i| i.row() as usize >= slots) {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: condition index {} is outside the {slots}-row conditioning table",
                bad.row()
            )));
        }
        let rows: Vec<u32> = conds.iter().map(|i| i.row()).collect();
        let ids = Tensor::from_vec(rows, (batch, per_row), &self.cfg.device)?;
        // [B, k, D] summed over k — one addition per slot, an identity
        // when k is 1.
        table.forward(&ids)?.sum(1)?.unsqueeze(1) // [B, 1, D]
    }

    /// The combination `Σ weights[i] · cond_wte[i]`, shaped
    /// `[1, 1, dim]` so it broadcasts across both the batch and the
    /// sequence.
    ///
    /// The whole table is gathered and scaled rather than only the rows
    /// with a non-zero weight: which rows those are is data, and a
    /// gather whose shape depends on the values would make two calls
    /// with the same arity run different kernels. The table is
    /// `cond_slots × dim` — the smallest tensor in the model — so the
    /// scaled-and-summed form costs nothing worth branching for.
    ///
    /// Validation is here rather than at the entry point because this is
    /// where the table is in hand: the row count the weights have to
    /// match is the table's, and a model built without one has no count
    /// to check against.
    fn weighted_condition_embedding(&self, weights: &[f32]) -> CandleResult<Tensor> {
        let table = self.cond_wte.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "gpt2 forward: this model has no conditioning table; build it with \
                 `custom.cond_slots = Some(n)` to use forward_cond_weighted"
                    .into(),
            )
        })?;
        let slots = self
            .cfg
            .custom
            .as_ref()
            .and_then(|c| c.cond_slots)
            .unwrap_or(0);
        if weights.len() != slots {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: {} condition weight(s) for a {slots}-row conditioning table; \
                 pass exactly one weight per row",
                weights.len()
            )));
        }
        // The coefficients are checked *after* the cast, because what
        // multiplies the table is the cast tensor: an `f32` weight of
        // 1e5 is finite as given and an infinity once narrowed to `f16`,
        // and one of 1e-8 is non-zero as given and exactly zero once
        // narrowed — the two states this validation exists to refuse.
        // The check itself is done in `f32`, which every model dtype
        // here widens into losslessly.
        let shares = Tensor::from_slice(weights, (1, slots, 1), &self.cfg.device)?
            .to_dtype(self.cfg.dtype)?;
        let narrowed: Vec<f32> = shares.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        if let Some((i, w)) = narrowed
            .iter()
            .enumerate()
            .find(|(_, w)| !w.is_finite())
            .map(|(i, w)| (i, *w))
        {
            let dtype = self.cfg.dtype;
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: condition weight {i} is {w} as {dtype:?} (given as {given}); a \
                 non-finite coefficient reaches every position of the residual stream",
                given = weights[i]
            )));
        }
        if narrowed.iter().all(|w| *w == 0.0) {
            let dtype = self.cfg.dtype;
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: every condition weight is zero as {dtype:?}, which adds the zero \
                 vector — the same state a model carrying this table runs in when nobody feeds \
                 it; an unconditioned run goes through `forward`"
            )));
        }
        let rows: Vec<u32> = (0..slots as u32).collect();
        let ids = Tensor::from_vec(rows, (1, slots), &self.cfg.device)?;
        let table_rows = table.forward(&ids)?; // [1, slots, D]
        table_rows.broadcast_mul(&shares)?.sum(1)?.unsqueeze(1) // [1, 1, D]
    }

    /// The per-position allowed-id vectors, shaped `[batch, seq, dim]`.
    ///
    /// The mean over each position's ids, computed as a weighted sum so
    /// that padding (weight zero) costs nothing and an empty set — the
    /// padding past the end of a row — yields the zero vector without a
    /// division ever being attempted. See [`Self::forward_allowed`].
    ///
    /// The gather is the expensive step: `[batch, seq, k, dim]` floats
    /// exist between the lookup and the sum, where `k` is the longest
    /// set in the batch. That is what pays for not fixing a maximum.
    ///
    /// Range is checked here for the reason
    /// [`Self::condition_embedding`] checks it: candle's CPU backend
    /// refuses an out-of-range `index_select`, and leaving it to the
    /// backend would let the accelerator builds decide whether a bad id
    /// is an error or a quietly different vector.
    fn allowed_embedding(
        &self,
        table: &Embedding,
        allowed: &AllowedSets,
        batch: usize,
        seq: usize,
    ) -> CandleResult<Tensor> {
        if allowed.rows() != batch || allowed.width() != seq {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: allowed ids cover {}x{} (rows x positions) for an input of \
                 {batch}x{seq}; one set per position of every row",
                allowed.rows(),
                allowed.width()
            )));
        }
        if allowed.max_id as usize >= self.cfg.vocab {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: allowed id {} is outside the {}-entry vocabulary",
                allowed.max_id, self.cfg.vocab
            )));
        }
        let k = allowed.widest();
        let rows = table.forward(&allowed.ids)?; // [B, T, K, D]
        let shares = allowed
            .weights
            .to_dtype(self.cfg.dtype)?
            .reshape((batch, seq, k, 1))?;
        rows.broadcast_mul(&shares)?.sum(2) // [B, T, D]
    }

    /// Wrap the model's per-block linear projections with LoRA
    /// low-rank updates.
    ///
    /// The base parameters (already registered against the model's
    /// original `VarMap`) are held frozen inside each new [`LoraLinear`];
    /// only the freshly-created `lora_a` / `lora_b` matrices are
    /// registered against the returned [`VarMap`]. Callers pass that
    /// map to the optimizer so gradients flow through the LoRA legs
    /// only — the design invariant "base parameters bit-identical
    /// before / after training" holds automatically because the base
    /// varmap is never handed to AdamW.
    ///
    /// # Errors
    ///
    /// - Any `cfg.target_modules` entry outside the canonical GPT-2 set
    ///   (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up`, `down`) is
    ///   rejected with a clear message. Empty `target_modules` errors
    ///   too rather than silently no-op'ing.
    /// - `cfg.rank == 0` or `cfg.rank > min(in, out)` for any wrapped
    ///   layer propagates the underlying `LoraLinear::wrap` error.
    /// - candle-side allocation failure for the LoRA parameters.
    ///
    /// # Notes
    ///
    /// The three `q_proj` / `k_proj` / `v_proj` names all map to
    /// wrapping the fused `c_attn` linear because GPT-2 keeps the
    /// three attention projections combined; a caller who requests
    /// only `q_proj` still gets Q, K, V wrapped as a single Δ (this
    /// matches the PEFT reference behaviour).
    pub fn wrap_lora(&mut self, cfg: &LoraConfig) -> CandleResult<VarMap> {
        if cfg.target_modules.is_empty() {
            return Err(candle_core::Error::Msg(
                "wrap_lora: target_modules is empty; nothing to wrap".into(),
            ));
        }
        for m in &cfg.target_modules {
            if !KNOWN_TARGET_MODULES.contains(&m.as_str()) {
                return Err(candle_core::Error::Msg(format!(
                    "wrap_lora: unknown target module {m:?} (known: {KNOWN_TARGET_MODULES:?})"
                )));
            }
        }
        let flags = WrapFlags {
            qkv: cfg
                .target_modules
                .iter()
                .any(|m| matches!(m.as_str(), "q_proj" | "k_proj" | "v_proj")),
            o: cfg.target_modules.iter().any(|m| m == "o_proj"),
            up: cfg.target_modules.iter().any(|m| m == "up"),
            down: cfg.target_modules.iter().any(|m| m == "down"),
        };

        // Clone the device and dtype up front so `VarBuilder` borrows
        // local values rather than a slice of `self` — otherwise the
        // simultaneous `self.blocks.iter_mut()` borrow below would
        // conflict with the `&self.cfg.device` inside the VarBuilder.
        let device = self.cfg.device.clone();
        let dtype = self.cfg.dtype;
        let lora_vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&lora_vm, dtype, &device);
        let h_vs = vs.pp("h");
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.wrap_lora(cfg, flags, h_vs.pp(i.to_string()))?;
        }
        Ok(lora_vm)
    }
}

/// Delegate to the inherent [`Gpt2Model::forward`] so the training
/// loop can drive any `M: candle_nn::Module` uniformly.
impl Module for Gpt2Model {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        Gpt2Model::forward(self, xs)
    }
}

/// Delegate to the inherent [`Gpt2Model::forward_conditioned_groups`],
/// so the conditioned training entry point can drive this model without
/// naming it.
impl crate::train::ConditionedForward for Gpt2Model {
    fn forward_conditioned_rows(
        &self,
        xs: &Tensor,
        conds: &[CondIndex],
        per_row: usize,
    ) -> CandleResult<Tensor> {
        self.forward_conditioned_groups(xs, conds, per_row)
    }
}

/// Delegate to the inherent [`Gpt2Model::forward_allowed`], so the
/// allowed-id training entry point can drive this model without naming
/// it.
impl crate::train::AllowedForward for Gpt2Model {
    fn forward_allowed_rows(&self, xs: &Tensor, allowed: &AllowedSets) -> CandleResult<Tensor> {
        self.forward_allowed(xs, allowed)
    }
}

impl crate::train::DeviceView for Gpt2Model {
    fn device(&self) -> &Device {
        &self.cfg.device
    }
}

/// Delegate to the inherent [`Gpt2Model::wrap_lora`] so the generic
/// [`crate::train::run_lora_ft`] loop can drive LoRA fine-tuning on any
/// `M: candle_nn::Module + crate::train::DeviceView + LoraWrappable`.
impl super::lora::LoraWrappable for Gpt2Model {
    fn wrap_lora(&mut self, cfg: &LoraConfig) -> CandleResult<VarMap> {
        Gpt2Model::wrap_lora(self, cfg)
    }
}

/// Emit a merged inference-ready weight bundle keyed by HF GPT-2
/// safetensors names. Every `LinearVariant::Lora` in a block is
/// collapsed via [`LoraLinear::merged_weight`]; every `Plain`
/// projection passes through unchanged. The emitted key layout
/// matches exactly what [`Gpt2Model::from_pretrained`] reads, so the
/// bundle is a drop-in base for the same `Gpt2Config`.
///
/// Layer 4a §3 Q2 — HF-native layout keys.
impl super::lora::MergeableLora for Gpt2Model {
    fn export_merged(&self) -> CandleResult<std::collections::HashMap<String, Tensor>> {
        // Custom architectures have no HF-layout equivalent — the
        // merged bundle contract is "drop-in base for from_pretrained",
        // which a custom model can never satisfy (random-init only).
        if self.cfg.custom.is_some() {
            return Err(candle_core::Error::Msg(
                "export_merged: custom architectures have no HF-layout merged-bundle \
                 representation (custom is random-init only)"
                    .into(),
            ));
        }
        let mut out: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

        // Top-level: token + positional embeddings (LM head is tied
        // to `wte`, so no separate `lm_head.weight` — matches
        // Gpt2Model::new which uses wte for both). `wpe` is only
        // `None` on custom (non-learned-pos) models, which the guard
        // above already rejected.
        let wpe = self.wpe.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "export_merged: wpe missing on a non-custom model (guard bug)".into(),
            )
        })?;
        out.insert("wte.weight".into(), self.wte.embeddings().clone());
        out.insert("wpe.weight".into(), wpe.embeddings().clone());

        // Per-block: ln_1, attn.c_attn, attn.c_proj, ln_2,
        // mlp.c_fc, mlp.c_proj. Naming mirrors Block::new's
        // VarBuilder path (`h.<i>.<field>`).
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("h.{i}");

            // LayerNorms carry both weight and bias.
            let ln_1 = block.ln_1.as_layer_norm()?;
            out.insert(format!("{prefix}.ln_1.weight"), ln_1.weight().clone());
            if let Some(b) = ln_1.bias() {
                out.insert(format!("{prefix}.ln_1.bias"), b.clone());
            }
            let ln_2 = block.ln_2.as_layer_norm()?;
            out.insert(format!("{prefix}.ln_2.weight"), ln_2.weight().clone());
            if let Some(b) = ln_2.bias() {
                out.insert(format!("{prefix}.ln_2.bias"), b.clone());
            }

            // Attention linears (potentially LoRA-wrapped).
            let c_attn_w = block.c_attn.merged_weight()?;
            out.insert(format!("{prefix}.attn.c_attn.weight"), c_attn_w);
            if let Some(b) = block.c_attn.bias() {
                out.insert(format!("{prefix}.attn.c_attn.bias"), b.clone());
            }
            let c_proj_w = block.c_proj.merged_weight()?;
            out.insert(format!("{prefix}.attn.c_proj.weight"), c_proj_w);
            if let Some(b) = block.c_proj.bias() {
                out.insert(format!("{prefix}.attn.c_proj.bias"), b.clone());
            }

            // MLP linears (potentially LoRA-wrapped). MoE blocks have
            // no HF-layout equivalent — the merged bundle contract is
            // "drop-in base for from_pretrained", which a MoE model can
            // never satisfy (random-init only), so exporting one is an
            // error rather than a silently incomplete bundle.
            match &block.ff {
                FeedForward::Dense { c_fc, c_proj, .. } => {
                    let mlp_c_fc_w = c_fc.merged_weight()?;
                    out.insert(format!("{prefix}.mlp.c_fc.weight"), mlp_c_fc_w);
                    if let Some(b) = c_fc.bias() {
                        out.insert(format!("{prefix}.mlp.c_fc.bias"), b.clone());
                    }
                    let mlp_c_proj_w = c_proj.merged_weight()?;
                    out.insert(format!("{prefix}.mlp.c_proj.weight"), mlp_c_proj_w);
                    if let Some(b) = c_proj.bias() {
                        out.insert(format!("{prefix}.mlp.c_proj.bias"), b.clone());
                    }
                }
                FeedForward::Moe(_) => {
                    return Err(candle_core::Error::Msg(
                        "export_merged: MoE blocks have no HF-layout merged-bundle \
                         representation (MoE is random-init only)"
                            .into(),
                    ));
                }
            }
        }

        // Final LayerNorm.
        let ln_f = self.ln_f.as_layer_norm()?;
        out.insert("ln_f.weight".into(), ln_f.weight().clone());
        if let Some(b) = ln_f.bias() {
            out.insert("ln_f.bias".into(), b.clone());
        }

        Ok(out)
    }
}

/// Errors from [`Gpt2Model::from_pretrained`].
///
/// Explicit variants so the caller (Lua bridge) can surface an
/// actionable error string per the crate's Service-layer
/// error-propagation discipline — no silent fallback.
#[derive(Debug, thiserror::Error)]
pub enum PretrainedError {
    /// Requested variant has no known HuggingFace mapping.
    #[error("unknown pretrained preset: {0}")]
    UnknownPreset(String),
    /// Weight download failure.
    #[error("hub download: {0}")]
    Download(String),
    /// Local cache IO failure.
    #[error("cache io: {0}")]
    CacheIo(String),
    /// safetensors / candle loading failure.
    #[error("load: {0}")]
    Load(String),
}

/// Build a causal (lower-triangular) mask `[ctx, ctx]` where valid
/// (kept) positions are `1` and masked-out are `0`.
///
/// `window = Some(w)` bands the triangle: position `i` keeps only
/// `j ∈ (i - w, i]` (the last `w` positions including itself —
/// Mistral's sliding-window convention). `w ≥ ctx` is the full
/// triangle again.
///
/// Returned as `u8` because candle's `Tensor::where_cond` only accepts
/// unsigned-integer condition tensors; the attention path never uses
/// the mask as a numeric multiplier (it drives `where_cond` between
/// the scaled scores and `-inf`), so the concrete `dtype` of the model
/// weights is irrelevant here.
fn build_causal_mask(
    ctx: usize,
    window: Option<usize>,
    device: &Device,
    _dtype: DType,
) -> CandleResult<Tensor> {
    let mut data = vec![0u8; ctx * ctx];
    for i in 0..ctx {
        let lo = match window {
            Some(w) => i.saturating_sub(w - 1),
            None => 0,
        };
        for j in lo..=i {
            data[i * ctx + j] = 1;
        }
    }
    Tensor::from_vec(data, (ctx, ctx), device)
}

/// Build the ALiBi constants: per-head slopes `[heads, 1, 1]` and the
/// signed distance table `dist[i, j] = i - j` `[ctx, ctx]`.
///
/// Slopes follow Press 2022's geometric sequence
/// `m_h = 2^(-8·(h+1)/H)` for `h ∈ [0, H)`. The paper's power-of-two
/// head counts get the exact published values; other counts get the
/// same closed form (the paper's interpolation scheme reduces to it
/// for our purposes). The forward pass combines the two into the
/// additive score bias `-m_h · (i - j)`; positions `j > i` produce a
/// positive bias there but are overwritten by the causal `-inf` mask,
/// so only the `j ≤ i` half is ever read.
fn build_alibi_consts(
    heads: usize,
    ctx: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let slopes: Vec<f32> = (0..heads)
        .map(|h| 2f32.powf(-8.0 * (h + 1) as f32 / heads as f32))
        .collect();
    let slopes = Tensor::from_vec(slopes, (heads, 1, 1), device)?.to_dtype(dtype)?;
    let mut dist = vec![0f32; ctx * ctx];
    for i in 0..ctx {
        for j in 0..ctx {
            dist[i * ctx + j] = i as f32 - j as f32;
        }
    }
    let dist = Tensor::from_vec(dist, (ctx, ctx), device)?.to_dtype(dtype)?;
    Ok((slopes, dist))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::LoraConfig;
    use candle_nn::VarMap;

    #[test]
    fn medium_preset_shape() {
        let cfg = Gpt2Config::medium();
        assert_eq!(cfg.layers, 24);
        assert_eq!(cfg.heads, 16);
        assert_eq!(cfg.dim, 1024);
        assert_eq!(cfg.ctx, 1024);
        assert_eq!(cfg.vocab, 50257);
        assert_eq!(cfg.hf_repo(), Some("openai-community/gpt2-medium"));
    }

    #[test]
    fn large_preset_shape() {
        let cfg = Gpt2Config::large();
        assert_eq!(cfg.layers, 36);
        assert_eq!(cfg.heads, 20);
        assert_eq!(cfg.dim, 1280);
        assert_eq!(cfg.hf_repo(), Some("openai-community/gpt2-large"));
    }

    #[test]
    fn from_variant_recognizes_aliases() {
        assert!(Gpt2Config::from_variant("medium").is_some());
        assert!(Gpt2Config::from_variant("gpt2-medium").is_some());
        assert!(Gpt2Config::from_variant("large").is_some());
        assert!(Gpt2Config::from_variant("gpt2-large").is_some());
        assert!(Gpt2Config::from_variant("small").is_none());
    }

    #[test]
    fn rejects_dim_not_divisible_by_heads() {
        let cfg = Gpt2Config {
            layers: 2,
            heads: 3, // 8 % 3 != 0
            dim: 8,
            ctx: 4,
            vocab: 10,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let msg = match Gpt2Model::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("divisible"));
    }

    /// Tiny (2-layer, 2-head, vocab 32) forward on CPU. Confirms the
    /// exact `[batch, seq, vocab]` output shape (subtask invariant #1).
    #[test]
    fn tiny_forward_shape() {
        let cfg = Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 32]);
    }

    #[test]
    fn tiny_forward_batch_shape() {
        let cfg = Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 8,
            ctx: 4,
            vocab: 16,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8], (2, 4), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[2, 4, 16]);
    }

    #[test]
    fn forward_rejects_seq_over_ctx() {
        let cfg = Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 8,
            ctx: 4,
            vocab: 16,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        // seq = 5 > ctx = 4
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let msg = match model.forward(&ids) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("exceeds ctx"));
    }

    fn tiny_cfg() -> Gpt2Config {
        Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        }
    }

    #[test]
    fn wrap_lora_rejects_empty_target_modules() {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();
        let lora = LoraConfig::with_targets(4, 8.0, Vec::<String>::new());
        let msg = match model.wrap_lora(&lora) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("empty"), "unexpected error: {msg}");
    }

    #[test]
    fn wrap_lora_rejects_unknown_target_module() {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();
        let lora = LoraConfig::with_targets(4, 8.0, vec!["typo_proj"]);
        let msg = match model.wrap_lora(&lora) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("unknown"), "unexpected error: {msg}");
    }

    #[test]
    fn wrap_lora_populates_lora_vm_and_freezes_base() {
        // Snapshot every base parameter before wrap + a single training-
        // like `opt.backward_step` on the LoRA legs. The base vars must
        // stay bit-identical; only lora_a / lora_b tensors should have
        // been created and any updates land there.
        let cfg = tiny_cfg();
        let base_vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();

        // Base tensor bytes before wrap.
        let base_before: Vec<Vec<f32>> = base_vm
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
            .collect();
        let base_var_count = base_vm.all_vars().len();

        let lora_cfg = LoraConfig::new(4, 8.0);
        let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

        // 2 layers * (c_attn + c_proj + mlp_c_fc + mlp_c_proj) * (a + b)
        // = 2 * 4 * 2 = 16 new Vars.
        assert_eq!(lora_vm.all_vars().len(), 16);
        // Base var count is unchanged (no new registrations on the base
        // varmap).
        assert_eq!(base_vm.all_vars().len(), base_var_count);

        // Base tensor bytes after wrap must match the snapshot.
        let base_after: Vec<Vec<f32>> = base_vm
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
            .collect();
        for (before, after) in base_before.iter().zip(base_after.iter()) {
            assert_eq!(before, after, "wrap_lora perturbed a base tensor");
        }

        // Forward still runs and returns the right shape.
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);
    }

    #[test]
    fn wrap_lora_with_narrow_targets_only_wraps_selected_layers() {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = Gpt2Model::new(&cfg, vs).unwrap();

        // Only wrap MLP down-projection (`mlp_c_proj`). Attention and
        // MLP-up stay `Plain`.
        let lora_cfg = LoraConfig::with_targets(2, 4.0, vec!["down"]);
        let lora_vm = model.wrap_lora(&lora_cfg).unwrap();
        // 2 layers * 1 wrapped linear * 2 vars (a + b) = 4.
        assert_eq!(lora_vm.all_vars().len(), 4);
    }

    fn tiny_moe_cfg() -> Gpt2Config {
        Gpt2Config {
            moe: Some(MoeConfig::new(2)),
            ..tiny_cfg()
        }
    }

    fn build_moe_model() -> (VarMap, Gpt2Model) {
        let cfg = tiny_moe_cfg();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        (vm, model)
    }

    #[test]
    fn moe_forward_shape_and_aux() {
        let cfg = tiny_moe_cfg();
        let (_vm, model) = build_moe_model();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();

        // Plain forward keeps the [B, S, V] contract.
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);

        // forward_with_aux returns a finite scalar aux on the MoE path.
        let (logits, aux) = model.forward_with_aux(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);
        let aux: f32 = aux.expect("moe aux").to_scalar().unwrap();
        assert!(aux.is_finite() && aux > 0.0, "aux={aux}");

        // Probe variant exposes one probs tensor per layer.
        let (_, _, probs) = model.forward_with_router_probs(&ids).unwrap();
        assert_eq!(probs.len(), cfg.layers);
        assert_eq!(probs[0].dims(), &[1, 5, 2]);
    }

    #[test]
    fn dense_forward_with_aux_returns_none() {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let (_, aux) = model.forward_with_aux(&ids).unwrap();
        assert!(aux.is_none());
        let (_, _, probs) = model.forward_with_router_probs(&ids).unwrap();
        assert!(probs.is_empty());
    }

    #[test]
    fn moe_varmap_uses_disjoint_names() {
        let (vm, _model) = build_moe_model();
        let data = vm.data().lock().unwrap();
        let names: Vec<&String> = data.keys().collect();
        assert!(
            names.iter().any(|n| *n == "h.0.moe.router.weight"),
            "missing router name in {names:?}"
        );
        assert!(
            names.iter().any(|n| *n == "h.1.moe.experts.1.c_proj.bias"),
            "missing expert name in {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains(".mlp.")),
            "stock mlp names must not appear on the MoE path: {names:?}"
        );
    }

    #[test]
    fn moe_wrap_lora_rejects_mlp_targets_but_allows_attention() {
        let (_vm, mut model) = build_moe_model();
        let msg = match model.wrap_lora(&LoraConfig::with_targets(2, 4.0, vec!["down"])) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("out of LoRA scope"), "unexpected error: {msg}");

        let (_vm2, mut model2) = build_moe_model();
        let lora_vm = model2
            .wrap_lora(&LoraConfig::with_targets(2, 4.0, vec!["q_proj"]))
            .unwrap();
        // 2 layers * c_attn * (a + b) = 4.
        assert_eq!(lora_vm.all_vars().len(), 4);
    }

    #[test]
    fn moe_rejects_pretrained_load() {
        let mut cfg = Gpt2Config::medium();
        cfg.moe = Some(MoeConfig::new(4));
        let msg = match Gpt2Model::from_pretrained("medium", &cfg, &std::env::temp_dir()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("random-init"), "unexpected error: {msg}");
        let msg = match Gpt2Model::from_safetensors_file(&cfg, std::path::Path::new("/nonexistent"))
        {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("random-init"), "unexpected error: {msg}");
    }

    #[test]
    fn moe_rejects_merged_export() {
        use crate::arch::lora::MergeableLora;
        let (_vm, model) = build_moe_model();
        let msg = match model.export_merged() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("export_merged"), "unexpected error: {msg}");
    }

    #[test]
    fn custom_default_matches_reference_varmap_names() {
        // `custom: Some(default)` must register exactly the same Var
        // names as `custom: None` — the spec's defaults are the
        // reference architecture.
        let names = |custom: Option<Gpt2Custom>| -> Vec<String> {
            let cfg = Gpt2Config {
                custom,
                ..tiny_cfg()
            };
            let vm = VarMap::new();
            let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
            let _model = Gpt2Model::new(&cfg, vs).unwrap();
            let data = vm.data().lock().unwrap();
            let mut n: Vec<String> = data.keys().cloned().collect();
            n.sort();
            n
        };
        assert_eq!(names(None), names(Some(Gpt2Custom::default())));
    }

    #[test]
    fn custom_forward_shape_across_activations() {
        for act in [
            Activation::Gelu,
            Activation::Relu,
            Activation::Silu,
            Activation::SwiGlu,
            Activation::GeGlu,
        ] {
            let cfg = Gpt2Config {
                custom: Some(Gpt2Custom {
                    act,
                    ..Default::default()
                }),
                ..tiny_cfg()
            };
            let vm = VarMap::new();
            let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
            let model = Gpt2Model::new(&cfg, vs).unwrap();
            let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
            let logits = model.forward(&ids).unwrap();
            assert_eq!(logits.dims(), &[1, 5, cfg.vocab], "act={act:?}");

            let data = vm.data().lock().unwrap();
            let has_gate = data.keys().any(|n| n.contains(".c_gate."));
            assert_eq!(has_gate, act.is_gated(), "act={act:?}");
        }
    }

    #[test]
    fn custom_rmsnorm_parallel_ratio_forward_shape() {
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                norm: NormKind::RmsNorm,
                residual: ResidualKind::Parallel,
                mlp_ratio: 2,
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);

        // RMSNorm keeps the ln_* names but registers no biases.
        let data = vm.data().lock().unwrap();
        assert!(data.keys().any(|n| n == "h.0.ln_1.weight"));
        assert!(!data.keys().any(|n| n.starts_with("h.0.ln_1.bias")));
        assert!(!data.keys().any(|n| n == "ln_f.bias"));
    }

    #[test]
    fn custom_rejects_pretrained_load_and_export() {
        use crate::arch::lora::MergeableLora;
        let mut cfg = Gpt2Config::medium();
        cfg.custom = Some(Gpt2Custom::default());
        let msg = match Gpt2Model::from_pretrained("medium", &cfg, &std::env::temp_dir()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("random-init"), "unexpected error: {msg}");

        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom::default()),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let msg = match model.export_merged() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("export_merged"), "unexpected error: {msg}");
    }

    /// A bundle this crate wrote for a custom config must reload
    /// through [`Gpt2Model::from_safetensors_file`] and reproduce the
    /// same logits. `VarMap::save` writes exactly the Vars the spec
    /// declares (gated `mlp.c_gate`, untied `lm_head.weight`, no `wpe`
    /// under RoPE, no LayerNorm biases under RMSNorm) and `new`
    /// requests those same names, so the round-trip is lossless; the
    /// RoPE cache and causal mask are recomputed, not stored.
    #[test]
    fn custom_bundle_reloads_from_safetensors_with_identical_logits() {
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                act: Activation::SwiGlu,
                norm: NormKind::RmsNorm,
                pos: PosKind::Rope,
                kv_heads: Some(1),
                untied_head: true,
                ..Default::default()
            }),
            ..tiny_cfg()
        };

        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let trained = Gpt2Model::new(&cfg, vs).expect("build custom model");
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).expect("ids");
        let expected = trained.forward(&ids).expect("forward before save");

        // The spec-dependent Vars must actually be in the bundle —
        // otherwise the reload below would pass vacuously.
        {
            let data = vm.data().lock().expect("varmap lock");
            assert!(data.keys().any(|n| n.contains(".c_gate.")), "gated MLP Var");
            assert!(
                data.keys().any(|n| n == "lm_head.weight"),
                "untied head Var"
            );
            assert!(!data.keys().any(|n| n == "wpe.weight"), "no wpe under RoPE");
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("custom.safetensors");
        vm.save(&path).expect("varmap save");

        let reloaded = Gpt2Model::from_safetensors_file(&cfg, &path).expect("reload custom bundle");
        let actual = reloaded.forward(&ids).expect("forward after reload");
        assert_eq!(actual.dims(), expected.dims());
        let diff = crate::arch::max_abs_diff_f32(&expected, &actual).expect("diff");
        assert!(diff < 1e-6, "logits diverged after reload: {diff}");
    }

    #[test]
    fn custom_moe_combination_rejects_dense_mlp_knobs_only() {
        // Non-default dense-MLP knobs address a module the MoE seam
        // replaces — rejected.
        for custom in [
            Gpt2Custom {
                act: Activation::SwiGlu,
                ..Default::default()
            },
            Gpt2Custom {
                mlp_ratio: 2,
                ..Default::default()
            },
        ] {
            let cfg = Gpt2Config {
                moe: Some(MoeConfig::new(2)),
                custom: Some(custom),
                ..tiny_cfg()
            };
            let vm = VarMap::new();
            let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
            let msg = match Gpt2Model::new(&cfg, vs) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected an error"),
            };
            assert!(msg.contains("experts"), "unexpected error: {msg}");
        }

        // Every other axis composes with MoE (Phase 2 integration —
        // grad coverage of the composition lives in
        // tests/custom_grad_coverage.rs).
        let cfg = Gpt2Config {
            moe: Some(MoeConfig::new(2)),
            custom: Some(Gpt2Custom {
                norm: NormKind::RmsNorm,
                pos: PosKind::Rope,
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        Gpt2Model::new(&cfg, vs).expect("custom (non-MLP axes) + moe builds");
    }

    #[test]
    fn custom_rejects_gqa_indivisible_and_odd_rope_head_dim() {
        // heads 2 % kv_heads 0/indivisible.
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                kv_heads: Some(3), // 2 % 3 != 0
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let msg = match Gpt2Model::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("kv_heads"), "unexpected error: {msg}");

        // dim 6 / heads 2 → head_dim 3 (odd) — RoPE bisects the head.
        let cfg = Gpt2Config {
            dim: 6,
            custom: Some(Gpt2Custom {
                pos: PosKind::Rope,
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let msg = match Gpt2Model::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("even head_dim"), "unexpected error: {msg}");
    }

    #[test]
    fn custom_pos_variants_forward_shape_and_wpe_presence() {
        for pos in [
            PosKind::Learned,
            PosKind::Rope,
            PosKind::Alibi,
            PosKind::NoPos,
        ] {
            let cfg = Gpt2Config {
                custom: Some(Gpt2Custom {
                    pos,
                    ..Default::default()
                }),
                ..tiny_cfg()
            };
            let vm = VarMap::new();
            let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
            let model = Gpt2Model::new(&cfg, vs).unwrap();
            let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
            let logits = model.forward(&ids).unwrap();
            assert_eq!(logits.dims(), &[1, 5, cfg.vocab], "pos={pos:?}");

            let data = vm.data().lock().unwrap();
            let has_wpe = data.keys().any(|n| n.starts_with("wpe."));
            assert_eq!(has_wpe, pos == PosKind::Learned, "pos={pos:?}");
        }
    }

    #[test]
    fn custom_untied_head_registers_lm_head_var() {
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                untied_head: true,
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab]);
        let data = vm.data().lock().unwrap();
        assert!(data.keys().any(|n| n == "lm_head.weight"));
    }

    #[test]
    fn sliding_window_mask_bands_the_triangle() {
        // ctx 5, window 2: row i keeps j ∈ {max(0, i-1), .., i}.
        let mask = build_causal_mask(5, Some(2), &Device::Cpu, DType::F32).unwrap();
        let rows: Vec<Vec<u8>> = mask.to_vec2().unwrap();
        assert_eq!(rows[0], vec![1, 0, 0, 0, 0]);
        assert_eq!(rows[1], vec![1, 1, 0, 0, 0]);
        assert_eq!(rows[2], vec![0, 1, 1, 0, 0]);
        assert_eq!(rows[4], vec![0, 0, 0, 1, 1]);
        // window ≥ ctx degenerates to the full triangle.
        let full = build_causal_mask(3, Some(8), &Device::Cpu, DType::F32).unwrap();
        let rows: Vec<Vec<u8>> = full.to_vec2().unwrap();
        assert_eq!(rows[2], vec![1, 1, 1]);
    }

    #[test]
    fn alibi_slopes_match_press_2022_for_8_heads() {
        // H = 8 → m_h = 2^-(h+1) (the paper's canonical example).
        let (slopes, dist) = build_alibi_consts(8, 4, DType::F32, &Device::Cpu).unwrap();
        let s: Vec<f32> = slopes.flatten_all().unwrap().to_vec1().unwrap();
        for (h, v) in s.iter().enumerate() {
            let expected = 2f32.powi(-(h as i32 + 1));
            assert!((v - expected).abs() < 1e-7, "h={h}: {v} vs {expected}");
        }
        let d: Vec<Vec<f32>> = dist.to_vec2().unwrap();
        assert_eq!(d[2][0], 2.0);
        assert_eq!(d[2][2], 0.0);
    }

    #[test]
    fn effective_kv_heads_mha_reports_full_head_count() {
        // Reference / named preset: `custom = None` → MHA, kv == heads.
        let cfg = tiny_cfg();
        assert_eq!(cfg.effective_kv_heads(), cfg.heads);
        assert_eq!(cfg.effective_kv_heads(), 2);
    }

    #[test]
    fn effective_kv_heads_custom_default_still_reports_full_head_count() {
        // `custom = Some(default)` leaves `kv_heads = None`, which is
        // spec-equivalent to MHA — must still report `heads`.
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom::default()),
            ..tiny_cfg()
        };
        assert_eq!(cfg.effective_kv_heads(), cfg.heads);
    }

    #[test]
    fn effective_kv_heads_gqa_reports_explicit_value() {
        // `custom.kv_heads = Some(1)` (MQA on a 2-head base) — the
        // accessor must report 1, not silently mirror `heads`.
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                kv_heads: Some(1),
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        assert_eq!(cfg.effective_kv_heads(), 1);
    }

    #[test]
    fn effective_kv_heads_ignores_unrelated_custom_axes() {
        // Axes orthogonal to attention (`mlp_ratio`, `untied_head`)
        // must not perturb the KV-head count — only `kv_heads`
        // influences the result.
        let cfg = Gpt2Config {
            custom: Some(Gpt2Custom {
                mlp_ratio: 3,
                untied_head: true,
                kv_heads: Some(2),
                ..Default::default()
            }),
            ..tiny_cfg()
        };
        assert_eq!(cfg.effective_kv_heads(), 2);
    }

    #[test]
    fn moe_rejects_invalid_config_at_build() {
        let cfg = Gpt2Config {
            moe: Some(MoeConfig {
                n_experts: 2,
                top_k: 3,
                alpha: 0.01,
            }),
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let msg = match Gpt2Model::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("top_k"), "unexpected error: {msg}");
    }

    // ── Conditioning ────────────────────────────────────────────────

    /// A model wide enough in context that "far from the front of the
    /// row" means something, carrying `slots` conditioning rows.
    fn conditioning_cfg(slots: Option<usize>) -> Gpt2Config {
        Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 16,
            ctx: 128,
            vocab: 64,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: Some(Gpt2Custom {
                cond_slots: slots,
                ..Default::default()
            }),
        }
    }

    /// A model whose weights a seed decides, so the assertions below
    /// are on numbers rather than on a draw.
    fn seeded_model(cfg: &Gpt2Config) -> Gpt2Model {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(cfg, vs).unwrap();
        fill_deterministic(&varmap, 0x0824_2026).unwrap();
        model
    }

    fn cond(row: u32, slots: usize) -> CondIndex {
        CondIndex::new(row, slots).expect("row inside the table")
    }

    /// The claim is comparative. "The condition changes the output far
    /// from the front of the row" is true of a token at the front too —
    /// decayed, but non-zero — so on its own it does not distinguish
    /// the two mechanisms. What is specific to conditioning at every
    /// position is that the effect does not fall off with distance.
    ///
    /// So the test measures both, on the same model and the same two
    /// positions: (a) two conditions passed as arguments, and (b) two
    /// rows differing only in the token at position 1, which is the
    /// prefix mechanism this replaces. The counterfactual is run rather
    /// than asserted in a comment, which is what makes the separation a
    /// measured margin instead of a chosen one.
    #[test]
    fn a_condition_does_not_decay_with_distance_the_way_a_prefix_does() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);

        let seq: Vec<u32> = (0..120u32).map(|i| 8 + i % 40).collect();
        let far = seq.len() - 1;
        let near = 1;
        let ids = Tensor::from_slice(&seq, (1, seq.len()), &cfg.device).unwrap();
        let at = |out: &Tensor, pos: usize| out.i((0, pos)).unwrap();

        let plain = model.forward(&ids).unwrap();
        let low = model.forward_conditioned(&ids, &[cond(0, 4)]).unwrap();
        let high = model.forward_conditioned(&ids, &[cond(1, 4)]).unwrap();

        let plain_vs_low = crate::arch::max_abs_diff_f32(&at(&plain, far), &at(&low, far)).unwrap();
        assert!(
            plain_vs_low > 1e-4,
            "a condition that changes nothing at position {far} is not a condition \
             (max abs diff {plain_vs_low})"
        );

        let far_gap = crate::arch::max_abs_diff_f32(&at(&low, far), &at(&high, far)).unwrap();
        assert!(
            far_gap > 1e-4,
            "two conditions produced the same logits at position {far} (gap {far_gap})"
        );
        let near_gap = crate::arch::max_abs_diff_f32(&at(&low, near), &at(&high, near)).unwrap();
        let per_position = far_gap / near_gap;

        // The counterfactual: the same information carried by a token
        // at position 1.
        let mut prefix_a = seq.clone();
        prefix_a[near] = 2;
        let mut prefix_b = seq.clone();
        prefix_b[near] = 3;
        let out_a = model
            .forward(&Tensor::from_slice(&prefix_a, (1, seq.len()), &cfg.device).unwrap())
            .unwrap();
        let out_b = model
            .forward(&Tensor::from_slice(&prefix_b, (1, seq.len()), &cfg.device).unwrap())
            .unwrap();
        let prefix_far = crate::arch::max_abs_diff_f32(&at(&out_a, far), &at(&out_b, far)).unwrap();
        let prefix_near =
            crate::arch::max_abs_diff_f32(&at(&out_a, near), &at(&out_b, near)).unwrap();
        let prefix_ratio = prefix_far / prefix_near;

        // The two mechanisms have to be *separated*, not merely on
        // opposite sides of a boundary: bounding one below 0.5 and the
        // other above it passes at 0.499 against 0.501, which is two
        // indistinguishable mechanisms. So the assertion is on the
        // ratio between them. The margin belongs to this model — one
        // layer at dim 16, untrained, where the prefix's far/near is
        // mostly dilution over the sequence — and the claim being made
        // is ordinal (an argument does not fade where a prefix does),
        // not a prediction of the size of the gap.
        let separation = per_position / prefix_ratio;
        assert!(
            separation > 10.0,
            "the two mechanisms are {separation}x apart: argument {per_position} \
             (near {near_gap}, far {far_gap}) against prefix {prefix_ratio} \
             (near {prefix_near}, far {prefix_far}); below a factor of ten they are not \
             telling us apart from what they replace"
        );
    }

    /// With one condition per row the grouped entry point is exactly
    /// the single-condition one — the sum over one row is that row.
    #[test]
    fn a_group_of_one_is_the_single_condition_forward() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &cfg.device).unwrap();
        let conds = [cond(0, 4), cond(2, 4)];
        let single = model.forward_conditioned(&ids, &conds).unwrap();
        let grouped = model.forward_conditioned_groups(&ids, &conds, 1).unwrap();
        let gap = crate::arch::max_abs_diff_f32(&single, &grouped).unwrap();
        assert!(gap < 1e-6, "the two entry points diverged by {gap}");
    }

    /// The sum makes the slots order-free, and makes the pair a third
    /// condition rather than either of its halves.
    #[test]
    fn grouped_conditions_are_order_free_and_not_either_alone() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let (a, b) = (cond(0, 4), cond(1, 4));
        let ab = model.forward_conditioned_groups(&ids, &[a, b], 2).unwrap();
        let ba = model.forward_conditioned_groups(&ids, &[b, a], 2).unwrap();
        assert!(crate::arch::max_abs_diff_f32(&ab, &ba).unwrap() < 1e-6);

        let alone = model.forward_conditioned(&ids, &[a]).unwrap();
        assert!(
            crate::arch::max_abs_diff_f32(&ab, &alone).unwrap() > 1e-4,
            "the pair read as one of its halves"
        );
    }

    /// One index per row, or the caller does not know which row it
    /// meant.
    #[test]
    fn conditioned_forward_rejects_a_count_that_is_not_one_per_row() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned(&ids, &[cond(0, 4)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("per row"), "{msg}");

        let three = [cond(0, 4), cond(1, 4), cond(2, 4)];
        let msg = match model.forward_conditioned_groups(&ids, &three, 2) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("pass exactly 4"), "{msg}");
    }

    /// Zero conditions per row is a different request — the plain
    /// forward — and is refused rather than read as an empty sum.
    #[test]
    fn grouped_conditions_reject_a_zero_group() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned_groups(&ids, &[], 0) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("per_row"), "{msg}");
    }

    /// A model built without the table cannot be conditioned, and says
    /// so rather than ignoring the argument.
    #[test]
    fn conditioned_forward_needs_a_table() {
        let cfg = conditioning_cfg(None);
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned(&ids, &[cond(0, 4)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("no conditioning table"), "{msg}");
    }

    /// An index built against a larger table than this model carries is
    /// a crossed pair, and no downstream reading would reveal it.
    #[test]
    fn conditioned_forward_rejects_an_index_outside_this_models_table() {
        let cfg = conditioning_cfg(Some(2));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        // Valid against a 4-row table, which this model does not have.
        let msg = match model.forward_conditioned(&ids, &[cond(3, 4)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("conditioning table"), "{msg}");
    }

    // ─── Weighted conditions ──────────────────────────────────────

    /// A one-hot combination is the row it selects. The two paths reach
    /// the same vector by different kernels — a gather of the whole
    /// table, scaled and reduced, against a gather of the one row — so
    /// the agreement is asserted within a tolerance rather than
    /// bit-for-bit.
    #[test]
    fn one_hot_weights_are_the_single_row_forward() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();

        for row in 0..4u32 {
            let mut weights = [0.0f32; 4];
            weights[row as usize] = 1.0;
            let selected = model.forward_conditioned(&ids, &[cond(row, 4)]).unwrap();
            let weighted = model.forward_cond_weighted(&ids, &weights).unwrap();
            let gap = crate::arch::max_abs_diff_f32(&selected, &weighted).unwrap();
            assert!(
                gap < 1e-6,
                "one-hot weights on row {row} diverged from the row selection by {gap}"
            );
        }
    }

    /// Equal weights on two rows put the added vector exactly halfway
    /// between the two rows' own vectors.
    ///
    /// The claim is about what is added to the residual stream, so it is
    /// checked there. The logits are not a linear function of that
    /// vector, and asserting a midpoint on them would be asserting
    /// something the model does not promise — the second half of this
    /// test only pins that the interpolation reaches the forward pass at
    /// all.
    #[test]
    fn equal_weights_land_halfway_between_the_two_rows() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);

        let first = model
            .weighted_condition_embedding(&[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let second = model
            .weighted_condition_embedding(&[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        let middle = model
            .weighted_condition_embedding(&[0.5, 0.5, 0.0, 0.0])
            .unwrap();
        let average = (&first + &second).unwrap().affine(0.5, 0.0).unwrap();
        let gap = crate::arch::max_abs_diff_f32(&middle, &average).unwrap();
        assert!(
            gap < 1e-6,
            "the half-and-half vector was {gap} off the mean"
        );
        // Non-vacuity: if the two rows held the same vector, every
        // combination of them would agree and the assertion above would
        // hold for a table that carries no distinction at all.
        let rows_differ = crate::arch::max_abs_diff_f32(&first, &second).unwrap();
        assert!(
            rows_differ > 1e-4,
            "the two table rows are {rows_differ} apart — too close to tell a midpoint from an \
             endpoint"
        );

        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let out_first = model
            .forward_cond_weighted(&ids, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let out_middle = model
            .forward_cond_weighted(&ids, &[0.5, 0.5, 0.0, 0.0])
            .unwrap();
        let logit_gap = crate::arch::max_abs_diff_f32(&out_first, &out_middle).unwrap();
        assert!(
            logit_gap > 1e-4,
            "the interpolated condition changed the logits by {logit_gap}; it never reached the \
             forward pass"
        );
    }

    /// The weights are used as given. A scaled combination is a
    /// different condition, not the same one renormalised.
    #[test]
    fn weights_are_not_normalised() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let plain = model
            .forward_cond_weighted(&ids, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let amplified = model
            .forward_cond_weighted(&ids, &[2.0, 0.0, 0.0, 0.0])
            .unwrap();
        let gap = crate::arch::max_abs_diff_f32(&plain, &amplified).unwrap();
        assert!(
            gap > 1e-4,
            "doubling the coefficient changed nothing (gap {gap}); the weights were rescaled \
             behind the caller's back"
        );
    }

    /// One weight per row, or the caller and the table disagree about
    /// which rows the numbers refer to.
    #[test]
    fn weighted_forward_rejects_a_count_that_is_not_one_per_row() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_cond_weighted(&ids, &[1.0, 0.0]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("one weight per row"), "{msg}");
    }

    /// A non-finite coefficient would reach every position, and the
    /// output would be uniformly unusable with nothing naming the cause.
    #[test]
    fn weighted_forward_rejects_a_non_finite_weight() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let msg = match model.forward_cond_weighted(&ids, &[0.5, bad, 0.0, 0.0]) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected an error for {bad}"),
            };
            assert!(msg.contains("condition weight 1"), "{msg}");
        }
    }

    /// All-zero weights add the zero vector, which is the unfed state
    /// the channel exists to keep the model out of.
    #[test]
    fn weighted_forward_rejects_an_all_zero_combination() {
        let cfg = conditioning_cfg(Some(4));
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_cond_weighted(&ids, &[0.0, 0.0, 0.0, 0.0]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("every condition weight is zero"), "{msg}");
    }

    /// The two refusals hold in the model's dtype rather than in the
    /// `f32` the caller wrote, because the cast tensor is what
    /// multiplies the table. `1e5` is finite as an `f32` and `+inf` as
    /// an `f16` (which tops out at 65504); `1e-8` is non-zero as an
    /// `f32` and exactly zero as an `f16` (whose smallest subnormal is
    /// ≈ 6e-8), which is the zero-vector state the all-zero refusal
    /// exists for. The `f32` model is run on the same weights as the
    /// control: it accepts both, so the assertions above are about the
    /// dtype and not about the numbers.
    #[test]
    fn weighted_forward_checks_the_weights_in_the_models_dtype() {
        let mut half = conditioning_cfg(Some(4));
        half.dtype = DType::F16;
        let model = seeded_model(&half);

        let msg = match model.weighted_condition_embedding(&[1.0e5, 0.0, 0.0, 0.0]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            msg.contains("condition weight 0") && msg.contains("F16"),
            "{msg}"
        );

        let msg = match model.weighted_condition_embedding(&[1e-8; 4]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("every condition weight is zero"), "{msg}");

        let full = seeded_model(&conditioning_cfg(Some(4)));
        full.weighted_condition_embedding(&[1.0e5, 0.0, 0.0, 0.0])
            .expect("1e5 is an ordinary f32 coefficient");
        full.weighted_condition_embedding(&[1e-8; 4])
            .expect("1e-8 is non-zero as an f32");
    }

    /// A model built without the table has no rows to combine, and says
    /// so rather than ignoring the argument.
    #[test]
    fn weighted_forward_needs_a_table() {
        let cfg = conditioning_cfg(None);
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_cond_weighted(&ids, &[1.0]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            msg.contains("no conditioning table") && msg.contains("forward_cond_weighted"),
            "{msg}"
        );
    }

    /// The constructor is where a row from some other numbering stops.
    #[test]
    fn cond_index_checks_the_row_against_the_table_it_names() {
        assert_eq!(CondIndex::new(0, 1).unwrap().row(), 0);
        assert_eq!(CondIndex::new(3, 4).unwrap().row(), 3);
        assert_eq!(
            CondIndex::new(4, 4).unwrap_err(),
            CondIndexError::OutOfRange { row: 4, slots: 4 }
        );
        assert_eq!(CondIndex::new(0, 0).unwrap_err(), CondIndexError::NoSlots);
        assert!(CondIndex::new(4, 4)
            .unwrap_err()
            .to_string()
            .contains("outside a 4-row"));
    }

    /// The table is a parameter of its own, registered under a name
    /// nothing else reads. Its absence from an unconditioned model is
    /// what makes a checkpoint of one convention fail to restore into
    /// the other.
    #[test]
    fn the_conditioning_table_is_a_parameter_of_its_own() {
        let cfg = conditioning_cfg(Some(3));
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let _ = Gpt2Model::new(&cfg, vs).unwrap();
        let names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        assert!(
            names.iter().any(|n| n == "cond_wte.weight"),
            "got: {names:?}"
        );
        let shape = varmap.data().lock().unwrap()["cond_wte.weight"]
            .shape()
            .dims()
            .to_vec();
        assert_eq!(shape, vec![3, cfg.dim]);

        let plain = conditioning_cfg(None);
        let plain_vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&plain_vm, plain.dtype, &plain.device);
        let _ = Gpt2Model::new(&plain, vs).unwrap();
        assert!(!plain_vm
            .data()
            .lock()
            .unwrap()
            .contains_key("cond_wte.weight"));
    }

    // ── Allowed-id input ────────────────────────────────────────────

    fn allowed_cfg() -> Gpt2Config {
        Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 16,
            ctx: 16,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: Some(Gpt2Custom {
                allowed_input: true,
                ..Default::default()
            }),
        }
    }

    /// The set is summarised by its mean, so a set that names the same
    /// id twice is the same input as the set that names it once. That
    /// identity is exact, and it is the arithmetic the channel rests
    /// on: a position offering many ids and one offering few arrive at
    /// the same scale.
    #[test]
    fn a_repeated_id_averages_back_to_the_single_id() {
        let cfg = allowed_cfg();
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();

        let once = AllowedSets::new(&[vec![vec![5u32], vec![5], vec![5]]], &cfg.device).unwrap();
        let twice =
            AllowedSets::new(&[vec![vec![5u32, 5], vec![5, 5], vec![5, 5]]], &cfg.device).unwrap();
        assert_eq!((once.widest(), twice.widest()), (1, 2));

        let a = model.forward_allowed(&ids, &once).unwrap();
        let b = model.forward_allowed(&ids, &twice).unwrap();
        let gap = crate::arch::max_abs_diff_f32(&a, &b).unwrap();
        assert!(gap < 1e-5, "the mean over a repeated id moved by {gap}");

        // A different set is a different input, or the channel does
        // nothing.
        let other = AllowedSets::new(&[vec![vec![6u32], vec![6], vec![6]]], &cfg.device).unwrap();
        let c = model.forward_allowed(&ids, &other).unwrap();
        assert!(crate::arch::max_abs_diff_f32(&a, &c).unwrap() > 1e-4);
    }

    /// The weights are `1 / count` on a real entry and zero on padding,
    /// so an empty position adds the zero vector without a division by
    /// a count of nothing ever being attempted.
    #[test]
    fn an_empty_position_carries_no_weight_at_all() {
        let device = Device::Cpu;
        let sets = AllowedSets::new(&[vec![vec![1u32, 2, 3], vec![], vec![7]]], &device).unwrap();
        assert_eq!((sets.rows(), sets.width(), sets.widest()), (1, 3, 3));
        let weights: Vec<Vec<f32>> = sets.weights().i(0).unwrap().to_vec2().unwrap();
        let third = 1.0f32 / 3.0;
        assert_eq!(weights[0], vec![third, third, third]);
        assert_eq!(weights[1], vec![0.0, 0.0, 0.0]);
        assert_eq!(weights[2], vec![1.0, 0.0, 0.0]);
        let ids: Vec<Vec<u32>> = sets.ids().i(0).unwrap().to_vec2().unwrap();
        assert_eq!(ids[1], vec![0, 0, 0], "padding is the zero id, weight zero");
    }

    /// An empty position changes nothing at that position and nothing
    /// before it, and a non-empty one does — which is how the zero
    /// vector differs from a set.
    #[test]
    fn an_empty_position_leaves_that_position_to_the_sequence_alone() {
        let cfg = allowed_cfg();
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2], (1, 2), &cfg.device).unwrap();
        let empty_tail = AllowedSets::new(&[vec![vec![5u32], vec![]]], &cfg.device).unwrap();
        let filled_tail = AllowedSets::new(&[vec![vec![5u32], vec![9]]], &cfg.device).unwrap();

        let a = model.forward_allowed(&ids, &empty_tail).unwrap();
        let b = model.forward_allowed(&ids, &filled_tail).unwrap();
        let at = |out: &Tensor, pos: usize| out.i((0, pos)).unwrap();
        assert!(
            crate::arch::max_abs_diff_f32(&at(&a, 0), &at(&b, 0)).unwrap() < 1e-6,
            "a later position changed an earlier one"
        );
        assert!(
            crate::arch::max_abs_diff_f32(&at(&a, 1), &at(&b, 1)).unwrap() > 1e-4,
            "an empty set read the same as a set naming an id"
        );
    }

    /// A batch in which no position of any row holds an id would add
    /// the zero vector everywhere, leaving the model answering exactly
    /// as one built without the table.
    #[test]
    fn allowed_sets_refuse_a_batch_with_no_ids_anywhere() {
        let device = Device::Cpu;
        let msg = AllowedSets::new(&[vec![vec![], vec![]]], &device)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("no position of any row holds an id"), "{msg}");

        // Zero rows and zero positions are refused by name as well.
        assert!(AllowedSets::new(&[], &device)
            .unwrap_err()
            .to_string()
            .contains("no rows"));
        assert!(AllowedSets::window(&[vec![vec![1u32]]], 0, 0, &device)
            .unwrap_err()
            .to_string()
            .contains("zero positions"));
    }

    /// Rows of differing lengths mean the batch and its sets were built
    /// from different row lists, which no windowing makes safe.
    #[test]
    fn allowed_sets_refuse_rows_of_differing_lengths() {
        let device = Device::Cpu;
        let msg = AllowedSets::new(&[vec![vec![1u32], vec![2]], vec![vec![3u32]]], &device)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("row 1 holds 1 position(s)"), "{msg}");
    }

    /// The window skips the positions it was told to skip, and reaches
    /// past the end of a row as the empty set rather than as an error.
    #[test]
    fn the_window_selects_positions_and_pads_past_the_end() {
        let device = Device::Cpu;
        let sets =
            AllowedSets::window(&[vec![vec![1u32], vec![2, 3], vec![4]]], 1, 3, &device).unwrap();
        assert_eq!((sets.rows(), sets.width(), sets.widest()), (1, 3, 2));
        let ids: Vec<Vec<u32>> = sets.ids().i(0).unwrap().to_vec2().unwrap();
        assert_eq!(ids[0], vec![2, 3]);
        assert_eq!(ids[1], vec![4, 0]);
        assert_eq!(ids[2], vec![0, 0], "past the end of the row");
    }

    /// The two disagreements are refused in both directions: a model
    /// that reads the channel run without one, and sets handed to a
    /// model that has no table.
    #[test]
    fn the_allowed_channel_refuses_both_disagreements() {
        let cfg = allowed_cfg();
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2], (1, 2), &cfg.device).unwrap();

        let msg = match model.forward(&ids) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("drops that channel entirely"), "{msg}");

        let plain_cfg = Gpt2Config {
            custom: None,
            ..allowed_cfg()
        };
        let plain = seeded_model(&plain_cfg);
        let sets = AllowedSets::new(&[vec![vec![5u32], vec![6]]], &cfg.device).unwrap();
        let msg = match plain.forward_allowed(&ids, &sets) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("no allowed-id table"), "{msg}");
    }

    /// Sets that do not cover exactly the input, and ids past the end
    /// of the vocabulary, are the two mistakes a shape check alone
    /// would not catch on the device.
    #[test]
    fn allowed_forward_checks_the_cover_and_the_vocabulary() {
        let cfg = allowed_cfg();
        let model = seeded_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2], (1, 2), &cfg.device).unwrap();

        let short = AllowedSets::new(&[vec![vec![5u32]]], &cfg.device).unwrap();
        let msg = match model.forward_allowed(&ids, &short) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("1x1 (rows x positions)"), "{msg}");

        let too_big =
            AllowedSets::new(&[vec![vec![5u32], vec![cfg.vocab as u32]]], &cfg.device).unwrap();
        let msg = match model.forward_allowed(&ids, &too_big) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("outside the 32-entry vocabulary"), "{msg}");
    }

    /// The allowed-id table is its own parameter, sized by the
    /// vocabulary, and absent from a model built without the axis.
    #[test]
    fn the_allowed_table_is_a_parameter_of_its_own() {
        let cfg = allowed_cfg();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let _ = Gpt2Model::new(&cfg, vs).unwrap();
        let shape = varmap.data().lock().unwrap()["allowed_wte.weight"]
            .shape()
            .dims()
            .to_vec();
        assert_eq!(shape, vec![cfg.vocab, cfg.dim]);

        let plain_cfg = Gpt2Config {
            custom: None,
            ..allowed_cfg()
        };
        let plain_vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&plain_vm, plain_cfg.dtype, &plain_cfg.device);
        let _ = Gpt2Model::new(&plain_cfg, vs).unwrap();
        assert!(!plain_vm
            .data()
            .lock()
            .unwrap()
            .contains_key("allowed_wte.weight"));
    }
}
