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
    /// shipped presets) is the GPT-2 reference. Custom models are
    /// random-init only. Combining `custom` with `moe` is allowed as
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
        let kv_heads = custom.kv_heads.unwrap_or(cfg.heads);
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
            cfg: cfg.clone(),
        })
    }

    /// Return the configuration this model was built from.
    pub fn config(&self) -> &Gpt2Config {
        &self.cfg
    }

    /// Load model weights from an on-disk safetensors bundle whose
    /// key layout matches the HF GPT-2 convention (same layout that
    /// [`Gpt2Model::from_pretrained`] downloads and that
    /// [`super::lora::MergeableLora::export_merged`] emits).
    ///
    /// This is the plain-load path used by (a) the merged-bundle
    /// parity oracle in `tests/merged_export_parity_gpt2.rs` and
    /// (b) future load-side integration that recognises
    /// `training_path == "merged"` and dispatches here instead of
    /// re-wrapping the model.
    ///
    /// # Errors
    ///
    /// `PretrainedError::Load` on safetensors parse failure or
    /// weight-name mismatch against the model shape.
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
        if cfg.custom.is_some() {
            return Err(PretrainedError::Load(
                "custom configs are random-init only; safetensors bundles carry the \
                 GPT-2 reference layout"
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
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_inner(xs, None).map(|(logits, _)| logits)
    }

    /// Forward pass that also returns the summed MoE load-balancing
    /// aux term (unscaled — multiply by [`MoeConfig::alpha`] when
    /// composing the total loss). `None` on a dense (non-MoE) model,
    /// so existing callers of [`Self::forward`] see no change.
    pub fn forward_with_aux(&self, xs: &Tensor) -> CandleResult<(Tensor, Option<Tensor>)> {
        self.forward_inner(xs, None)
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
        let (logits, aux) = self.forward_inner(xs, Some(&mut probs))?;
        Ok((logits, aux, probs))
    }

    fn forward_inner(
        &self,
        xs: &Tensor,
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
}
