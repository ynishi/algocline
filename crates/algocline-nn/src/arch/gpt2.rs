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
//! - `cond_wte` — optional conditioning table (`cond_slots × dim`),
//!   added at every position by [`Gpt2Model::forward_conditioned`]
//! - `legal_wte` — optional legality table (`vocab × dim`), whose mean
//!   over the ids allowed at a position is added there by
//!   [`Gpt2Model::forward_legal`]
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
/// a test can assert on numbers a forward pass actually produces.
///
/// candle has no `Device::set_seed` on this version and the crate does
/// not seed its initialisers, so a model built by [`Gpt2Model::new`] is
/// different on every run. A test that thresholds a difference between
/// two such forwards is thresholding a random draw; this makes the
/// draw the test's own.
///
/// Norm weights are held at one and every bias at zero rather than
/// being randomised, because a norm scaled by a number near zero would
/// make the model degenerate in a way no real initialisation is.
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

/// A row of a model's conditioning table.
///
/// The point of the wrapper is that the number inside cannot be
/// supplied by a caller. Rows are `0..cond_slots`, and a condition
/// that means something to the caller almost always has a *second*
/// numbering — most obviously a token id, when the condition also
/// appears in the sequence. The two ranges are different and they
/// overlap, so an id passed where a row is wanted selects a real but
/// wrong row: the range check passes, the forward pass succeeds, and
/// the model is conditioned on something else with nothing downstream
/// able to report it.
///
/// So the constructor is crate-private, and a caller obtains one from
/// whatever holds the mapping — for the chess models in this crate,
/// [`crate::chess::ModelShape::band_index`], whose documentation has
/// the arithmetic of the overlap. Every consumer outside this crate,
/// the chess examples included, goes through such a producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondIndex(u32);

impl CondIndex {
    /// Wrap a table row.
    ///
    /// Named for what the argument has to be rather than `new`,
    /// because `new` reads as acceptable at a call site holding the
    /// wrong number and `from_table_row` does not. Crate-private, so
    /// no consumer outside this crate can reach it at all — but the
    /// in-crate training path is exactly the code holding both
    /// numberings, and there a name is the whole of the guard.
    pub(crate) fn from_table_row(row: u32) -> Self {
        Self(row)
    }

    /// The row this index selects.
    pub fn row(self) -> u32 {
        self.0
    }
}

/// The ids a model may pick from at each position of a batch, in the
/// form [`Gpt2Model::forward_legal`] reads them.
///
/// # What the indices mean
///
/// `sets[r][p]` is the set the model's prediction **for input position
/// `p` of row `r`** is drawn from — the moves available once it has
/// consumed input position `p`, not the ones that were available when
/// position `p` itself was played. Under the usual next-token shift
/// that is the same set the loss mask uses for target `p`
/// ([`crate::train::allowed_logit_mask`]), which is why
/// [`crate::train::legal_input_sets`] builds both from one list at one
/// offset: given as input on one side, used to strike out the
/// alternatives on the other.
///
/// Getting that offset wrong shifts every set by one position and
/// leaves every shape agreeing, so the in-crate producer is the
/// training-side helper rather than this constructor. A reader that
/// builds its own — the players, which generate the legal set from the
/// board they are standing on — passes it here.
///
/// # Layout
///
/// The lists are padded to the longest set **in this batch** rather
/// than to any constant: a constant would have to be a bound on the
/// number of moves a position can offer, and one chosen slightly too
/// small truncates a legal set with nothing to say so. The padding
/// carries no weight, so the width it reaches changes the arithmetic
/// nowhere.
///
/// The cost is paid in the forward pass rather than here: the ids are
/// `rows × width × k` u32 (7 MB at 64 × 128 × 218), but the gather they
/// drive materialises `rows × width × k × dim` floats before the mean
/// collapses it.
#[derive(Debug, Clone)]
pub struct LegalSets {
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

impl LegalSets {
    /// Build from one set per position of every row.
    ///
    /// Every row must be the same length, which is the model's input
    /// width. See the type's documentation for what `sets[r][p]` has to
    /// be.
    ///
    /// # Errors
    ///
    /// No rows, no positions, or rows of differing lengths.
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
    /// padding past the end of a game, and the forward pass reads it as
    /// the zero vector rather than dividing by a count of nothing.
    ///
    /// A **batch** in which every position of every row is empty is a
    /// different statement and is refused here. Such an input would add
    /// the zero vector everywhere, so the model handed it would answer
    /// exactly as the same model with the legality term deleted — a
    /// checkpoint trained with this channel, scored as though it had
    /// none, every number well-formed.
    ///
    /// Nothing else in the workspace catches that. The sidecar axis and
    /// the readers' refusals act on what a checkpoint *declares*, and
    /// this is a property of the input handed to it; the guard in
    /// [`Gpt2Model::forward_legal`] tests that a `LegalSets` was
    /// supplied rather than that it carries ids, so it lets this
    /// through and returns numbers a caller cannot tell from a correct
    /// run.
    ///
    /// The two producers differ on whether they can build one.
    /// [`crate::train::legal_input_sets`] windows a batch's own
    /// `allowed_ids`, and the batches the chess dataset builds fill
    /// theirs from a board replay that stops at the first token the
    /// position cannot play — so the empty positions are a tail and the
    /// batch carries ids, which
    /// `tests/chess_legal_input_bake.rs` asserts on a real corpus. A
    /// reader recovering sets by replaying history has the empty set
    /// available as a fallback for the positions a window dropped the
    /// history for, and taking it at every position is exactly the case
    /// above. So the refusal lives here, where both producers pass,
    /// rather than in an assertion the second would have to remember.
    ///
    /// # Errors
    ///
    /// `sets` is empty, `width` is zero, a row is shorter than the rest
    /// **before** the window is applied (rows of differing lengths mean
    /// the batch and its legal sets were built from different row
    /// lists, which no windowing makes safe), or no position in the
    /// window holds an id.
    pub fn window(
        sets: &[Vec<Vec<u32>>],
        from: usize,
        width: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let rows = sets.len();
        if rows == 0 {
            return Err(candle_core::Error::Msg(
                "legal sets: no rows, so there is nothing to say is legal".into(),
            ));
        }
        if width == 0 {
            return Err(candle_core::Error::Msg(
                "legal sets: zero positions per row".into(),
            ));
        }
        let full = sets[0].len();
        for (r, row) in sets.iter().enumerate() {
            if row.len() != full {
                return Err(candle_core::Error::Msg(format!(
                    "legal sets: row {r} holds {} position(s) and row 0 holds {full}",
                    row.len()
                )));
            }
        }

        // The longest set in the window. Zero means no position of any
        // row holds an id, which is refused rather than built: every
        // weight would be zero, the channel would contribute the
        // additive identity everywhere, and the model would answer
        // exactly as the same model with no legality table — the one
        // outcome nothing further along can tell from a correct run.
        // See this function's documentation for which caller can
        // produce it.
        let k = sets
            .iter()
            .flat_map(|row| row.iter().skip(from).take(width))
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        if k == 0 {
            return Err(candle_core::Error::Msg(format!(
                "legal sets: no position of any row holds an id ({rows} row(s), {width} \
                 position(s) from {from}), so this input would add nothing anywhere and the \
                 model would answer as though it had no legality channel at all; a producer \
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
    /// set. Read only by [`Self::forward_conditioned`].
    cond_wte: Option<Embedding>,
    /// Legality table (`[vocab, dim]`, VarMap name
    /// `legal_wte.weight`), present iff [`Gpt2Custom::legal_input`].
    /// Read only by [`Self::forward_legal`].
    legal_wte: Option<Embedding>,
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
        // but `forward_conditioned`, which is what keeps it out of the
        // tug-of-war `wte` would be in — see `Gpt2Custom::cond_slots`.
        let cond_wte = match custom.cond_slots {
            Some(slots) => Some(gpt2_embedding(slots, cfg.dim, vs.pp("cond_wte"))?),
            None => None,
        };
        // Legality table. One row per vocabulary entry, because its
        // entries are ids — and separate from `wte` for the reason
        // `Gpt2Custom::legal_input` records: the set added here holds
        // the token that is about to be the target.
        let legal_wte = if custom.legal_input {
            Some(gpt2_embedding(cfg.vocab, cfg.dim, vs.pp("legal_wte"))?)
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
            legal_wte,
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
    /// [`Gpt2Custom::legal_input`] reads a legality input at every
    /// position — a model carrying `legal_wte` handed none. See
    /// [`Self::forward_legal`] for why that is refused rather than run
    /// with the channel absent.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_inner(xs, None, None, None)
            .map(|(logits, _)| logits)
    }

    /// Forward pass with the ids allowed at each position added to the
    /// residual stream there.
    ///
    /// `legal` holds one set per position of every row; the mean of
    /// `legal_wte` over that set is added where the positional
    /// embedding is added. Output shape is unchanged. Requires
    /// [`Gpt2Custom::legal_input`] to have been set when the model was
    /// built.
    ///
    /// # What this is for
    ///
    /// A model trained on sequences alone has to infer the rules of the
    /// action space from them, and on chess that inference is most of
    /// the objective: 1.59 of 4.52 nats measured here went on keeping
    /// mass off moves that did not exist in the position, and the
    /// decoder discards all of it, because it walks the ranking against
    /// the legal set regardless of what the model believed. Handing the
    /// set over spends that share on the question that survives to
    /// inference — which of the available moves to play.
    ///
    /// # The mean, and the empty set
    ///
    /// A set is summarised by the mean of its rows rather than the sum,
    /// so a position offering forty moves and one offering four arrive
    /// at the same scale. Exactly so in `F32`, which is what the chess
    /// runs use. In a lower-precision dtype it is approximate: the
    /// shares are cast to the model's dtype here rather than held in
    /// it, and `BF16` carries 8 significand bits, so `1/count` need not
    /// be representable — it is for a count that is a power of two, and
    /// rounds by up to about `2^-8` otherwise — and neither the shares
    /// nor the sum they drive is exact. Nothing here corrects for that;
    /// a run that cares would have to.
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
    /// be legal here". It is not distinguishable from a set whose rows
    /// happen to average to zero; nothing here relies on telling those
    /// two apart.
    ///
    /// A whole input of such positions is a different matter, and
    /// [`LegalSets::window`] refuses to build one: it would leave this
    /// call returning what the same model with no table returns, which
    /// is the failure the surrounding machinery exists to catch.
    ///
    /// # Errors
    ///
    /// - The model was built without a legality table.
    /// - `legal` does not cover exactly `[batch, seq]`.
    /// - Any id is outside the vocabulary.
    /// - Anything [`Self::forward`] rejects, `seq > ctx` included.
    pub fn forward_legal(&self, xs: &Tensor, legal: &LegalSets) -> CandleResult<Tensor> {
        self.forward_inner(xs, None, Some(legal), None)
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
    /// The alternative — read it out of a fixed position in `xs` — is
    /// what the chess readers used to assume, and it does not survive a
    /// row longer than `ctx`: whatever windowing the caller applies can
    /// move an ordinary token into that slot, and the model would
    /// condition on it without any observable difference from a real
    /// condition. As an argument there is no position for windowing to
    /// disturb.
    ///
    /// The table's rows are numbered from zero and the band tokens are
    /// not, so the two numberings overlap without meaning the same
    /// thing; [`CondIndex`] is what keeps a caller from passing one for
    /// the other, and its documentation has the arithmetic.
    ///
    /// # Why a table of its own
    ///
    /// The band ids are vocabulary entries, so reusing `wte` was the
    /// tempting choice: it adds no tensor, which would leave the
    /// safetensors key set identical between this convention and the
    /// prefix one. It is also wrong on the reference topology, where
    /// the LM head is tied to `wte`
    /// ([`Gpt2Custom::untied_head`] `== false`). A `wte` row added to
    /// the residual stream at every position raises that token's logit
    /// at every position; the band token is a target nowhere but the
    /// front of the row, so a full-vocabulary cross-entropy pushes it
    /// straight back down. The model can comply by shrinking the
    /// vector or by having the blocks subtract it out of the stream,
    /// and both destroy the signal the caller is trying to measure —
    /// more so here than under a prefix, since here it happens at every
    /// position rather than one, which would make an arm difference
    /// attributable to weight tying rather than to where the condition
    /// attaches. `cond_wte` is read by nothing but this addition.
    ///
    /// The cost is that the two conventions are no longer identical on
    /// disk: this one carries `cond_wte.weight`, `cond_slots × dim`
    /// parameters. That makes [`crate::chess::ModelShape::encoding`] a
    /// second line of defence rather than the only one, and it is a
    /// difference in parameter count between arms, which belongs in the
    /// confound table of whatever compares them.
    ///
    /// # Errors
    ///
    /// - The model was built without a conditioning table.
    /// - `conds.len()` is not `batch`.
    /// - Any index is outside `[0, cond_slots)` — reachable when the
    ///   index came from a `ModelShape` with more bands than the model
    ///   has rows, which is a crossed pair rather than a typo.
    /// - Anything [`Self::forward`] rejects, `seq > ctx` included.
    pub fn forward_conditioned(&self, xs: &Tensor, conds: &[CondIndex]) -> CandleResult<Tensor> {
        self.forward_inner(xs, Some((conds, 1)), None, None)
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
    /// The composition being tested is additive: each slot contributes
    /// its own vector to the residual stream at every position, next to
    /// where a single condition's vector already goes, and nothing
    /// downstream distinguishes "one vector that is a sum" from "two
    /// additions". That makes the operation permutation-invariant in
    /// the slots — `[a, b]` and `[b, a]` are the same forward — which
    /// is a property of the mechanism and not a claim that the *model*
    /// treats the slots symmetrically: what each row's vector comes to
    /// mean is the training corpus's business.
    ///
    /// What this is **not**: a per-slot table. Every index still points
    /// into the one `cond_wte`, so the caller owns the convention of
    /// which rows belong to which slot (a shape records it as its group
    /// sizes). Two indices of the same slot are not refused here — the
    /// forward cannot know the grouping — and a caller that wants that
    /// refusal puts it where the grouping is known.
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
        self.forward_inner(xs, Some((conds, per_row)), None, None)
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
        conds: Option<(&[CondIndex], usize)>,
        legal: Option<&LegalSets>,
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
        if let Some((conds, per_row)) = conds {
            let cond_emb = self.condition_embedding(conds, b, per_row)?; // [B, 1, D]
            h = h.broadcast_add(&cond_emb)?;
        }
        // The legality input, beside the condition and for the same
        // reason: it belongs at every position, because every position
        // is a different set of available moves.
        //
        // Both disagreements are refused. A model that carries the
        // table was trained with this channel at every position, so
        // running it without one runs a model in a state it never
        // trained in — and its moves would look no different, which is
        // the same silence `run_conditioned_ft` refuses from the other
        // side. The mirror, sets handed to a model with no table, is a
        // caller that believes it is doing something it is not.
        match (&self.legal_wte, legal) {
            (Some(table), Some(sets)) => {
                let legal_emb = self.legality_embedding(table, sets, b, t)?; // [B, T, D]
                h = (h + legal_emb)?;
            }
            (Some(_), None) => {
                return Err(candle_core::Error::Msg(
                    "gpt2 forward: this model was built with a legality table and reads one at \
                     every position; running it without one drops that channel entirely — pass \
                     the legal ids to forward_legal"
                        .into(),
                ))
            }
            (None, Some(_)) => {
                return Err(candle_core::Error::Msg(
                    "gpt2 forward: legal ids were supplied to a model with no legality table; \
                     build it with `custom.legal_input = true` to use forward_legal"
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
    /// every caller has them on the host anyway, and it makes the
    /// arity and range checks free. Range in particular has to be
    /// checked here — candle's CPU backend rejects an out-of-range
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
                "gpt2 forward: condition index {} is outside the {slots}-row \
                 conditioning table",
                bad.row()
            )));
        }
        let rows: Vec<u32> = conds.iter().map(|i| i.row()).collect();
        let ids = Tensor::from_vec(rows, (batch, per_row), &self.cfg.device)?;
        // [B, k, D] summed over k — one addition per slot, an identity
        // when k is 1.
        table.forward(&ids)?.sum(1)?.unsqueeze(1) // [B, 1, D]
    }

    /// The per-position legality vectors, shaped `[batch, seq, dim]`.
    ///
    /// The mean over each position's ids, computed as a weighted sum so
    /// that padding (weight zero) costs nothing and an empty set — the
    /// padding past the end of a row — yields the zero vector without a
    /// division ever being attempted. See [`Self::forward_legal`].
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
    fn legality_embedding(
        &self,
        table: &Embedding,
        legal: &LegalSets,
        batch: usize,
        seq: usize,
    ) -> CandleResult<Tensor> {
        if legal.rows() != batch || legal.width() != seq {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: legal ids cover {}x{} (rows x positions) for an input of \
                 {batch}x{seq}; one set per position of every row",
                legal.rows(),
                legal.width()
            )));
        }
        if legal.max_id as usize >= self.cfg.vocab {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: legal id {} is outside the {}-entry vocabulary",
                legal.max_id, self.cfg.vocab
            )));
        }
        let k = legal.widest();
        let rows = table.forward(&legal.ids)?; // [B, T, K, D]
        let shares = legal
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

impl crate::train::DeviceView for Gpt2Model {
    fn device(&self) -> &Device {
        &self.cfg.device
    }
}

/// Delegate to the inherent [`Gpt2Model::forward_conditioned`] so the
/// conditioned training entry ([`crate::train::run_conditioned_ft`]) can
/// drive this model without naming it.
///
/// The trait method is spelled differently from the inherent one on
/// purpose. Sharing the name would leave every call site resolving to
/// whichever the compiler prefers — the inherent one — which is right
/// here and silently wrong the day an implementor writes the delegation
/// the other way round and recurses.
impl crate::train::ConditionedForward for Gpt2Model {
    fn forward_conditioned_rows(
        &self,
        xs: &Tensor,
        conds: &[CondIndex],
        per_row: usize,
    ) -> CandleResult<Tensor> {
        Gpt2Model::forward_conditioned_groups(self, xs, conds, per_row)
    }
}

/// Delegate to the inherent [`Gpt2Model::forward_legal`] so the
/// legality-input training entry ([`crate::train::run_legal_ft`]) can
/// drive this model without naming it.
///
/// Spelled differently from the inherent method for the reason
/// [`crate::train::ConditionedForward`] gives: a shared name leaves the
/// delegation one edit away from recursing into itself.
impl crate::train::LegalForward for Gpt2Model {
    fn forward_legal_rows(&self, xs: &Tensor, legal: &LegalSets) -> CandleResult<Tensor> {
        Gpt2Model::forward_legal(self, xs, legal)
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
    use crate::arch::{max_abs_diff_f32, LoraConfig};
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

    /// A model wide enough in `ctx` to put a position far from the
    /// front of the row, which is where the condition has to arrive,
    /// and carrying a conditioning table.
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

    /// A conditioned model whose weights a seed decides, so the
    /// assertions below are on numbers rather than on a draw.
    fn conditioning_model(cfg: &Gpt2Config) -> Gpt2Model {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(cfg, vs).unwrap();
        fill_deterministic(&varmap, 0x0801_2026).unwrap();
        model
    }

    /// Phase 0-1, and the claim being tested is a comparative one.
    ///
    /// "The condition changes the output far from the front of the row"
    /// is true of a prefix token too — decayed, but non-zero — so on
    /// its own it does not distinguish the two mechanisms. What is
    /// specific to conditioning at every position is that the effect
    /// does not fall off with distance.
    ///
    /// So the test measures both. It takes the same model and the same
    /// two positions, and compares (a) two conditions passed as
    /// arguments against (b) two rows differing only in the token at
    /// position 1, which is the prefix mechanism this replaces. The
    /// per-position ratio has to sit inside a band around 1 and the
    /// prefix ratio has to sit below it — the counterfactual is run
    /// rather than asserted in a comment, which is what makes the band
    /// a measured margin instead of a chosen one.
    #[test]
    fn conditioned_forward_does_not_decay_with_distance() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);

        let seq: Vec<u32> = (0..120u32).map(|i| 8 + i % 40).collect();
        let far = seq.len() - 1;
        let near = 1;
        let ids = Tensor::from_slice(&seq, (1, seq.len()), &cfg.device).unwrap();
        let at = |out: &Tensor, pos: usize| out.i((0, pos)).unwrap();

        let plain = model.forward(&ids).unwrap();
        let low = model
            .forward_conditioned(&ids, &[CondIndex::from_table_row(0)])
            .unwrap();
        let high = model
            .forward_conditioned(&ids, &[CondIndex::from_table_row(1)])
            .unwrap();

        let plain_vs_low = max_abs_diff_f32(&at(&plain, far), &at(&low, far)).unwrap();
        assert!(
            plain_vs_low > 1e-4,
            "a condition that changes nothing at position {far} is not a condition \
             (max abs diff {plain_vs_low})"
        );

        let far_gap = max_abs_diff_f32(&at(&low, far), &at(&high, far)).unwrap();
        assert!(
            far_gap > 1e-4,
            "two conditions produced the same logits at position {far} (gap {far_gap})"
        );
        let near_gap = max_abs_diff_f32(&at(&low, near), &at(&high, near)).unwrap();
        let per_position = far_gap / near_gap;

        // The counterfactual: the condition carried by a token at
        // position 1, which is what the rest of this work is measuring
        // the decay of.
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
        let prefix_far = max_abs_diff_f32(&at(&out_a, far), &at(&out_b, far)).unwrap();
        let prefix_near = max_abs_diff_f32(&at(&out_a, near), &at(&out_b, near)).unwrap();
        let prefix_ratio = prefix_far / prefix_near;

        assert!(
            (0.5..=2.0).contains(&per_position),
            "the condition's effect at position {far} was {per_position}x its effect at \
             position {near} (near {near_gap}, far {far_gap}); a mechanism that fades \
             with distance is the one this replaces"
        );
        // The two mechanisms have to be *separated*, not merely on
        // opposite sides of a boundary: bounding one below 0.5 and the
        // other above it passes at 0.499 against 0.501, which is two
        // indistinguishable mechanisms. So the assertion is on the
        // ratio between them.
        //
        // Observed 0.840 against 0.0033, a separation of some 250x, so
        // a floor of 10 is nowhere near either side.
        //
        // That 250 belongs to *this* model and does not travel. It is
        // one layer at dim 16, untrained, so attention is near uniform
        // and the prefix's `far/near` is mostly dilution over the
        // sequence — roughly `1/T`. The trained 4-layer prefix decay
        // measured on held-out play is 0.0272 -> 0.0021, a ratio of
        // about 0.077, some 23x larger than the 0.0033 here. The claim
        // this test makes is ordinal (an argument does not fade where
        // a prefix does), not a prediction of the margin.
        let separation = per_position / prefix_ratio;
        assert!(
            separation > 10.0,
            "the two mechanisms are {separation}x apart: argument {per_position} \
             (near {near_gap}, far {far_gap}) against prefix {prefix_ratio} \
             (near {prefix_near}, far {prefix_far}); below a factor of ten they are not \
             telling us apart from what they replace"
        );
    }

    /// One index per row, or the caller does not know which row it
    /// meant.
    #[test]
    fn conditioned_forward_rejects_a_condition_per_batch_mismatch() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned(&ids, &[CondIndex::from_table_row(0)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("per row"), "{msg}");
    }

    /// With one condition per row the grouped entry point is exactly
    /// the single-condition one — the sum over one row is that row.
    #[test]
    fn a_group_of_one_is_the_single_condition_forward() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &cfg.device).unwrap();
        let conds = [CondIndex::from_table_row(0), CondIndex::from_table_row(2)];
        let single = model.forward_conditioned(&ids, &conds).unwrap();
        let grouped = model.forward_conditioned_groups(&ids, &conds, 1).unwrap();
        let gap = max_abs_diff_f32(&single, &grouped).unwrap();
        assert!(gap < 1e-6, "the two entry points diverged by {gap}");
    }

    /// The sum makes the slots order-free, and makes the pair a third
    /// condition rather than either alone.
    #[test]
    fn grouped_conditions_are_order_free_and_not_either_alone() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let a = CondIndex::from_table_row(0);
        let b = CondIndex::from_table_row(1);
        let ab = model.forward_conditioned_groups(&ids, &[a, b], 2).unwrap();
        let ba = model.forward_conditioned_groups(&ids, &[b, a], 2).unwrap();
        assert!(max_abs_diff_f32(&ab, &ba).unwrap() < 1e-6);

        let alone = model.forward_conditioned(&ids, &[a]).unwrap();
        assert!(
            max_abs_diff_f32(&ab, &alone).unwrap() > 1e-4,
            "the pair read as one of its halves"
        );
    }

    /// Zero conditions per row is a different request — the plain
    /// forward — and is refused rather than read as an empty sum.
    #[test]
    fn grouped_conditions_reject_a_zero_group() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned_groups(&ids, &[], 0) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("per_row"), "{msg}");
    }

    /// A count that is not `batch * per_row` cannot be attributed to
    /// rows, and is refused with the expected count named.
    #[test]
    fn grouped_conditions_reject_a_count_mismatch() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &cfg.device).unwrap();
        let three = [
            CondIndex::from_table_row(0),
            CondIndex::from_table_row(1),
            CondIndex::from_table_row(2),
        ];
        let msg = match model.forward_conditioned_groups(&ids, &three, 2) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("pass exactly 4"), "{msg}");
    }

    /// An index past the end of the table is a caller mistake that no
    /// downstream reading would reveal, so it is refused by name.
    #[test]
    fn conditioned_forward_rejects_an_index_outside_the_table() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned(&ids, &[CondIndex::from_table_row(4)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("conditioning table"), "{msg}");
    }

    /// A model built without the table cannot be conditioned, and says
    /// so rather than ignoring the argument.
    #[test]
    fn conditioned_forward_needs_a_table() {
        let cfg = conditioning_cfg(None);
        let model = conditioning_model(&cfg);
        let ids = Tensor::from_slice(&[1u32, 2, 3], (1, 3), &cfg.device).unwrap();
        let msg = match model.forward_conditioned(&ids, &[CondIndex::from_table_row(0)]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("no conditioning table"), "{msg}");
    }

    /// The table is not tied to the LM head, so it is not registered
    /// under a name any other part of the model reads. Its absence
    /// from an unconditioned model is what makes a checkpoint of one
    /// convention fail to restore into the other.
    #[test]
    fn the_conditioning_table_is_a_parameter_of_its_own() {
        let cfg = conditioning_cfg(Some(3));
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let _ = Gpt2Model::new(&cfg, vs).unwrap();
        let data = varmap.data().lock().unwrap();
        let table = data
            .get("cond_wte.weight")
            .expect("a conditioned model registers cond_wte.weight");
        assert_eq!(table.shape().dims(), &[3, cfg.dim]);

        let plain_cfg = conditioning_cfg(None);
        let plain_map = VarMap::new();
        let plain_vs = VarBuilder::from_varmap(&plain_map, plain_cfg.dtype, &plain_cfg.device);
        let _ = Gpt2Model::new(&plain_cfg, plain_vs).unwrap();
        assert!(!plain_map
            .data()
            .lock()
            .unwrap()
            .contains_key("cond_wte.weight"));
    }

    /// Each row takes its own condition: two rows with the same tokens
    /// and different indices must not come out the same.
    #[test]
    fn conditioned_forward_applies_a_condition_per_row() {
        let cfg = conditioning_cfg(Some(4));
        let model = conditioning_model(&cfg);
        let row: Vec<u32> = (0..40u32).map(|i| 8 + i % 20).collect();
        let mut both = row.clone();
        both.extend_from_slice(&row);
        let ids = Tensor::from_slice(&both, (2, row.len()), &cfg.device).unwrap();
        let out = model
            .forward_conditioned(
                &ids,
                &[CondIndex::from_table_row(0), CondIndex::from_table_row(1)],
            )
            .unwrap();
        let last = row.len() - 1;
        let gap = max_abs_diff_f32(&out.i((0, last)).unwrap(), &out.i((1, last)).unwrap()).unwrap();
        assert!(gap > 1e-4, "both rows were conditioned the same way");
    }

    /// A model carrying the legality table, at a size small enough that
    /// a set can cover a visible share of the vocabulary.
    fn legality_cfg() -> Gpt2Config {
        Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 16,
            ctx: 32,
            vocab: 24,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: Some(Gpt2Custom {
                legal_input: true,
                ..Default::default()
            }),
        }
    }

    fn legality_model(cfg: &Gpt2Config) -> (VarMap, Gpt2Model) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(cfg, vs).unwrap();
        fill_deterministic(&varmap, 0x0807_2026).unwrap();
        (varmap, model)
    }

    /// One row of `width` sets, all the same.
    fn sets_of(width: usize, ids: &[u32]) -> Vec<Vec<Vec<u32>>> {
        vec![vec![ids.to_vec(); width]]
    }

    /// The table is a parameter of its own, sized to the vocabulary,
    /// and absent from a model that was not asked for one.
    #[test]
    fn the_legality_table_is_a_parameter_of_its_own() {
        let cfg = legality_cfg();
        let (varmap, _model) = legality_model(&cfg);
        let data = varmap.data().lock().unwrap();
        let table = data
            .get("legal_wte.weight")
            .expect("a legality-input model registers legal_wte.weight");
        assert_eq!(table.shape().dims(), &[cfg.vocab, cfg.dim]);
        // Not the same tensor as `wte`, whose values the same fill
        // produced under a different name. Sharing them is what the
        // separate table exists to avoid, and the shapes are equal, so
        // nothing else here would notice.
        let wte: Vec<f32> = data["wte.weight"]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let legal: Vec<f32> = table.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(wte, legal, "legal_wte must not be an alias of wte");

        let plain = VarMap::new();
        let vs = VarBuilder::from_varmap(&plain, cfg.dtype, &cfg.device);
        let mut without = cfg.clone();
        without.custom = None;
        let _ = Gpt2Model::new(&without, vs).unwrap();
        assert!(!plain
            .data()
            .lock()
            .unwrap()
            .contains_key("legal_wte.weight"));
    }

    /// What the input does at all: the same tokens under two different
    /// legal sets come out differently, at every position rather than
    /// at one.
    #[test]
    fn the_legal_set_changes_what_the_model_says() {
        let cfg = legality_cfg();
        let (_vm, model) = legality_model(&cfg);
        let row: Vec<u32> = (0..20u32).map(|i| 2 + i % 18).collect();
        let ids = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();

        let a = LegalSets::new(&sets_of(row.len(), &[3, 4, 5]), &cfg.device).unwrap();
        let b = LegalSets::new(&sets_of(row.len(), &[9, 10]), &cfg.device).unwrap();
        let out_a = model.forward_legal(&ids, &a).unwrap();
        let out_b = model.forward_legal(&ids, &b).unwrap();

        for pos in [0usize, 1, row.len() / 2, row.len() - 1] {
            let gap =
                max_abs_diff_f32(&out_a.i((0, pos)).unwrap(), &out_b.i((0, pos)).unwrap()).unwrap();
            assert!(
                gap > 1e-5,
                "position {pos} said the same thing under two different legal sets (gap {gap})"
            );
        }
    }

    /// The summary is the mean over the true count, not the sum: the
    /// same id repeated is the same vector, where a sum would double
    /// it.
    ///
    /// This is what makes a position offering forty moves and one
    /// offering four arrive at the same scale, and it is the property
    /// the padding has to leave alone.
    #[test]
    fn a_set_is_summarised_by_its_mean() {
        let cfg = legality_cfg();
        let (_vm, model) = legality_model(&cfg);
        let row = vec![2u32, 3, 4, 5];
        let ids = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();

        let once = LegalSets::new(&sets_of(row.len(), &[7]), &cfg.device).unwrap();
        let twice = LegalSets::new(&sets_of(row.len(), &[7, 7]), &cfg.device).unwrap();
        assert_eq!(once.widest(), 1);
        assert_eq!(twice.widest(), 2);

        let a = model.forward_legal(&ids, &once).unwrap();
        let b = model.forward_legal(&ids, &twice).unwrap();
        let gap = max_abs_diff_f32(&a, &b).unwrap();
        assert!(
            gap < 1e-5,
            "the same id listed twice moved the output by {gap}, so the set is being summed \
             rather than averaged"
        );
    }

    /// Padding to the longest set in the batch changes nothing for the
    /// shorter rows.
    ///
    /// The property that lets the width follow the batch instead of a
    /// constant: a row scored beside a row with a much larger set has
    /// to come out exactly as it does alone.
    #[test]
    fn padding_to_the_longest_set_leaves_the_shorter_rows_alone() {
        let cfg = legality_cfg();
        let (_vm, model) = legality_model(&cfg);
        let row: Vec<u32> = vec![2, 3, 4, 5];
        let both: Vec<u32> = row.iter().chain(row.iter()).copied().collect();

        // Row 0 offers two ids at every position; row 1 offers seven,
        // so the batch pads row 0's lists out to seven.
        let wide: Vec<u32> = (6..13).collect();
        let mixed = LegalSets::new(
            &[vec![vec![7u32, 8]; row.len()], vec![wide; row.len()]],
            &cfg.device,
        )
        .unwrap();
        assert_eq!(mixed.widest(), 7);
        let ids = Tensor::from_slice(&both, (2, row.len()), &cfg.device).unwrap();
        let padded = model.forward_legal(&ids, &mixed).unwrap();

        // The same row on its own, where nothing is padded.
        let alone = LegalSets::new(&sets_of(row.len(), &[7, 8]), &cfg.device).unwrap();
        assert_eq!(alone.widest(), 2);
        let ids_alone = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();
        let unpadded = model.forward_legal(&ids_alone, &alone).unwrap();

        let gap = max_abs_diff_f32(
            &padded.i(0).unwrap().contiguous().unwrap(),
            &unpadded.i(0).unwrap().contiguous().unwrap(),
        )
        .unwrap();
        assert!(
            gap < 1e-5,
            "padding a row's lists out to a wider neighbour moved its output by {gap}"
        );
    }

    /// An empty position contributes the zero vector, and a batch that
    /// holds one stays finite.
    ///
    /// The case is not hypothetical: it is every padded position past
    /// the end of a row. A mean over zero elements would be NaN and
    /// would take the batch with it.
    ///
    /// The equality is asserted against the plain forward of a model
    /// built **from the same `VarMap`** without the table, so every
    /// shared tensor is the same tensor and the only difference between
    /// the two runs is the legality term. Attention is causal, so
    /// position 0 depends on position 0 alone: where the set there is
    /// empty, the two have to agree bit for bit.
    #[test]
    fn an_empty_position_contributes_nothing_rather_than_a_nan() {
        let cfg = legality_cfg();
        let (vm, model) = legality_model(&cfg);
        let row = vec![2u32, 3, 4, 5];
        let ids = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();

        // Empty at every position but one — the shape of every batch
        // whose rows end before the window does.
        let sets =
            LegalSets::new(&[vec![vec![], vec![6u32], vec![], vec![]]], &cfg.device).unwrap();
        let out = model.forward_legal(&ids, &sets).unwrap();
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "got {values:?}");

        let mut plain_cfg = cfg.clone();
        plain_cfg.custom = None;
        let plain = Gpt2Model::new(
            &plain_cfg,
            VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device),
        )
        .unwrap();
        let bare = plain.forward(&ids).unwrap();
        let at = |t: &Tensor, p: usize| t.i((0, p)).unwrap();

        let gap = max_abs_diff_f32(&at(&out, 0), &at(&bare, 0)).unwrap();
        assert!(
            gap == 0.0,
            "an empty set at position 0 did not contribute the zero vector (gap {gap})"
        );
        // And position 1, which does carry an id, has to differ —
        // otherwise the equality above would hold for a channel that
        // does nothing at all.
        let gap = max_abs_diff_f32(&at(&out, 1), &at(&bare, 1)).unwrap();
        assert!(
            gap > 1e-5,
            "an id at position 1 changed nothing (gap {gap})"
        );
    }

    /// A batch with no ids anywhere is refused at construction.
    ///
    /// It is the one input that would leave [`Gpt2Model::forward_legal`]
    /// returning what the plain forward returns, so a checkpoint
    /// trained with this channel could be scored without it and every
    /// number would be well-formed. The sidecar axis and the readers'
    /// refusals do not cover it: they act on what the checkpoint says,
    /// and this is a property of the input handed to it.
    #[test]
    fn a_batch_with_no_ids_anywhere_is_refused() {
        let msg = match LegalSets::new(&sets_of(4, &[]), &Device::Cpu) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an all-empty batch must not build"),
        };
        assert!(msg.contains("no position of any row holds an id"), "{msg}");

        // And through the windowing constructor, which is the one the
        // training path calls.
        let rows = vec![vec![vec![], vec![], vec![]], vec![vec![], vec![], vec![]]];
        assert!(LegalSets::window(&rows, 1, 2, &Device::Cpu).is_err());

        // A single id anywhere in the window is enough: the refusal is
        // about the batch, not about a position.
        let rows = vec![
            vec![vec![], vec![], vec![]],
            vec![vec![], vec![], vec![9u32]],
        ];
        let sets = LegalSets::window(&rows, 1, 2, &Device::Cpu).expect("one id is enough");
        assert_eq!(sets.widest(), 1);
    }

    /// A model built with the table refuses to run without the input,
    /// and a model built without one refuses the input. Both
    /// directions, because either would be a run in a state the model
    /// was not trained in and neither would look wrong afterwards.
    #[test]
    fn the_legality_channel_refuses_both_mismatches() {
        let cfg = legality_cfg();
        let (_vm, model) = legality_model(&cfg);
        let row = vec![2u32, 3, 4];
        let ids = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();

        let msg = match model.forward(&ids) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a model with a legality table must not run without one"),
        };
        assert!(msg.contains("legality table"), "{msg}");

        let mut plain_cfg = cfg.clone();
        plain_cfg.custom = None;
        let (_vm, plain) = legality_model(&plain_cfg);
        let sets = LegalSets::new(&sets_of(row.len(), &[5]), &cfg.device).unwrap();
        let msg = match plain.forward_legal(&ids, &sets) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a model with no legality table must not accept sets"),
        };
        assert!(msg.contains("no legality table"), "{msg}");
    }

    /// Sets that do not cover the input, and ids the vocabulary does
    /// not hold, are refused by name rather than left to the backend.
    #[test]
    fn the_legality_input_has_to_describe_this_batch() {
        let cfg = legality_cfg();
        let (_vm, model) = legality_model(&cfg);
        let row = vec![2u32, 3, 4, 5];
        let ids = Tensor::from_slice(&row, (1, row.len()), &cfg.device).unwrap();

        let short = LegalSets::new(&sets_of(row.len() - 1, &[5]), &cfg.device).unwrap();
        let msg = match model.forward_legal(&ids, &short) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a coverage error"),
        };
        assert!(msg.contains("one set per position"), "{msg}");

        let outside =
            LegalSets::new(&sets_of(row.len(), &[cfg.vocab as u32]), &cfg.device).unwrap();
        let msg = match model.forward_legal(&ids, &outside) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a vocabulary error"),
        };
        assert!(msg.contains("outside the"), "{msg}");
    }

    /// Rows of differing lengths mean the batch and its legal sets came
    /// from different row lists, which no windowing makes safe.
    #[test]
    fn legal_sets_refuse_rows_of_differing_lengths() {
        let msg = match LegalSets::new(&[vec![vec![1u32]; 3], vec![vec![1u32]; 2]], &Device::Cpu) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a row-length error"),
        };
        assert!(msg.contains("position(s)"), "{msg}");
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
}
