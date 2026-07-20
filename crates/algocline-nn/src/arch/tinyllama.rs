//! TinyLlama-1.1B trainable architecture.
//!
//! Layer 1a — primitives (backward-safe shims + RoPE cache + GQA
//! `repeat_kv`):
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
//!   `[B, H_kv, S, head_dim] → [B, H, S, head_dim]` via `expand`.
//!
//! Layer 1b — model:
//!
//! - [`TinyLlamaConfig`] with presets `tinyllama-1.1b` and
//!   `tinyllama-tiny` (CPU-friendly smoke shape).
//! - [`TinyLlamaModel`] — pre-RMSNorm decoder stack with GQA + RoPE +
//!   SwiGLU MLP, mirroring the HuggingFace `LlamaModel` weight layout
//!   so a downloaded `TinyLlama-1.1B-*` safetensors bundle loads
//!   through [`candle_nn::VarBuilder::from_mmaped_safetensors`] without
//!   any renaming.
//!
//! HF weight naming (matches
//! `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`):
//!
//! ```text
//! model.embed_tokens.weight                     [vocab, dim]
//! model.layers.<i>.input_layernorm.weight       [dim]
//! model.layers.<i>.self_attn.q_proj.weight      [heads   *head_dim, dim]
//! model.layers.<i>.self_attn.k_proj.weight      [kv_heads*head_dim, dim]
//! model.layers.<i>.self_attn.v_proj.weight      [kv_heads*head_dim, dim]
//! model.layers.<i>.self_attn.o_proj.weight      [dim, heads*head_dim]
//! model.layers.<i>.post_attention_layernorm.weight  [dim]
//! model.layers.<i>.mlp.gate_proj.weight         [hidden, dim]
//! model.layers.<i>.mlp.up_proj.weight           [hidden, dim]
//! model.layers.<i>.mlp.down_proj.weight         [dim, hidden]
//! model.norm.weight                             [dim]
//! lm_head.weight                                [vocab, dim]
//! ```
//!
//! Forward output shape is `[batch, seq, vocab]`. Attention uses a
//! causal (lower-triangular) mask cached to `ctx`.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias, ops, rms_norm, rotary_emb, Embedding, Linear, Module, RmsNorm,
    VarBuilder, VarMap,
};

// `LoraLinear` is imported for the intra-doc links in `wrap_lora` /
// `Block::wrap_lora`; the wrap helper itself lives in `arch::lora` since
// GPT-2 already uses the same swap-in-place idiom.
#[allow(unused_imports)]
use super::lora::{wrap_variant_in_place, LinearVariant, LoraConfig, LoraLinear};

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

// ---------------------------------------------------------------------
// Layer 1b — Config / Block / Model / from_pretrained
// ---------------------------------------------------------------------

/// Immutable configuration for a TinyLlama preset.
///
/// Field naming mirrors the HuggingFace `LlamaConfig` so a future
/// generic `from_hf_config_json` helper stays a mechanical translation.
#[derive(Debug, Clone)]
pub struct TinyLlamaConfig {
    /// Number of transformer blocks.
    pub layers: usize,
    /// Number of query attention heads. `dim % heads == 0` required.
    pub heads: usize,
    /// Number of key / value heads (GQA). `heads % kv_heads == 0`
    /// required; `heads / kv_heads` is the `n_rep` passed to
    /// [`repeat_kv`].
    pub kv_heads: usize,
    /// Model hidden size (`d_model`).
    pub dim: usize,
    /// SwiGLU MLP intermediate size.
    pub hidden_dim: usize,
    /// Maximum context length (RoPE cache size).
    pub ctx: usize,
    /// Vocabulary size (matches the SentencePiece tokenizer, 32000
    /// for TinyLlama).
    pub vocab: usize,
    /// RoPE base (theta_0).
    pub rope_theta: f32,
    /// RMSNorm epsilon (HF `rms_norm_eps`, TinyLlama = 1e-5).
    pub eps: f64,
    /// Weight precision.
    pub dtype: DType,
    /// Device the parameters live on.
    pub device: Device,
}

impl TinyLlamaConfig {
    /// `tinyllama-1.1b` preset (~1.1B params, 22 layers, 32 heads,
    /// 4 KV heads, dim 2048, hidden 5632, ctx 2048, vocab 32000).
    ///
    /// Numbers match
    /// `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/config.json`.
    pub fn tinyllama_1_1b() -> Self {
        Self {
            layers: 22,
            heads: 32,
            kv_heads: 4,
            dim: 2048,
            hidden_dim: 5632,
            ctx: 2048,
            vocab: 32000,
            rope_theta: 10_000.0,
            eps: 1e-5,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// `tinyllama-tiny` — a 2-layer / 2-head / 1-KV-head / dim-64 /
    /// hidden-128 / ctx-16 / vocab-32 shape for CPU smoke tests.
    ///
    /// `head_dim = 32` keeps RoPE cache small; `heads / kv_heads == 2`
    /// exercises the GQA `repeat_kv` path with `n_rep > 1`. There is
    /// no HuggingFace bundle at this size, so
    /// [`TinyLlamaModel::from_pretrained`] refuses this variant
    /// ([`Self::hf_repo`] returns `None`).
    pub fn tiny() -> Self {
        Self {
            layers: 2,
            heads: 2,
            kv_heads: 1,
            dim: 64,
            hidden_dim: 128,
            ctx: 16,
            vocab: 32,
            rope_theta: 10_000.0,
            eps: 1e-5,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Resolve a variant name (`"tinyllama-1.1b"` / `"1.1b"` /
    /// `"tinyllama-tiny"` / `"tiny"`) to the matching preset. Returns
    /// `None` for unknown names.
    pub fn from_variant(variant: &str) -> Option<Self> {
        match variant {
            "tinyllama-1.1b" | "1.1b" => Some(Self::tinyllama_1_1b()),
            "tinyllama-tiny" | "tiny" => Some(Self::tiny()),
            _ => None,
        }
    }

    /// HuggingFace repository id for warm-start weight download.
    /// Returns `None` for a config not built from a shipped preset,
    /// including [`Self::tiny`].
    pub fn hf_repo(&self) -> Option<&'static str> {
        match (self.layers, self.heads, self.kv_heads, self.dim) {
            (22, 32, 4, 2048) => Some("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
            _ => None,
        }
    }

    /// Convenience: `dim / heads` per-query head dimension.
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

/// Errors from [`TinyLlamaModel::from_pretrained`].
///
/// Mirror of [`super::gpt2::PretrainedError`] — kept per-arch so a
/// caller matching on the enum can produce arch-specific messages
/// without an intermediate `Box<dyn Error>`.
#[derive(Debug, thiserror::Error)]
pub enum PretrainedError {
    /// Requested variant has no known HuggingFace mapping.
    #[error("unknown pretrained preset: {0}")]
    UnknownPreset(String),
    /// hf-hub client construction failure.
    #[error("hf-hub api: {0}")]
    HubApi(String),
    /// Weight download failure.
    #[error("hf-hub download: {0}")]
    Download(String),
    /// Local cache IO failure.
    #[error("cache io: {0}")]
    CacheIo(String),
    /// safetensors / candle loading failure.
    #[error("load: {0}")]
    Load(String),
}

/// A single TinyLlama transformer block.
///
/// Pre-RMSNorm topology:
///
/// ```text
/// x' = x + attn(input_layernorm(x))
/// y  = x' + mlp(post_attention_layernorm(x'))
/// ```
///
/// `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
/// `down_proj` are all held as [`LinearVariant`] (bias-less base linears
/// under `Plain`). A subsequent [`TinyLlamaModel::wrap_lora`] call can
/// replace individual layers with a LoRA-wrapped variant without
/// touching the surrounding forward code path (the `Module` impl on
/// [`LinearVariant`] dispatches to `Plain` or `Lora` internally).
struct Block {
    input_layernorm: RmsNorm,
    q_proj: LinearVariant,
    k_proj: LinearVariant,
    v_proj: LinearVariant,
    o_proj: LinearVariant,
    post_attention_layernorm: RmsNorm,
    gate_proj: LinearVariant,
    up_proj: LinearVariant,
    down_proj: LinearVariant,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
}

/// Which per-block projections the caller wants LoRA-wrapped.
///
/// Unlike GPT-2 (fused `c_attn` linear, single `qkv` flag), TinyLlama
/// keeps `q_proj` / `k_proj` / `v_proj` split, so each projection
/// carries its own bool. Mirrors the shape of the HF PEFT reference
/// (`target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj",
/// "up_proj","down_proj"]`).
#[derive(Debug, Clone, Copy)]
struct TinyLlamaWrapFlags {
    q: bool,
    k: bool,
    v: bool,
    o: bool,
    gate: bool,
    up: bool,
    down: bool,
}

/// Canonical TinyLlama / HuggingFace-Llama target-module names accepted
/// by [`TinyLlamaModel::wrap_lora`]. Any name outside this list triggers
/// an error at wrap time so a typo does not silently degrade to "no-op".
const KNOWN_TARGET_MODULES_TINYLLAMA: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

impl Block {
    fn new(cfg: &TinyLlamaConfig, vs: VarBuilder) -> CandleResult<Self> {
        let head_dim = cfg.head_dim();
        let n_rep = cfg.heads / cfg.kv_heads;

        let input_layernorm = rms_norm(cfg.dim, cfg.eps, vs.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.dim, cfg.eps, vs.pp("post_attention_layernorm"))?;

        let attn_vs = vs.pp("self_attn");
        let q_proj = linear_no_bias(cfg.dim, cfg.heads * head_dim, attn_vs.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.dim, cfg.kv_heads * head_dim, attn_vs.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.dim, cfg.kv_heads * head_dim, attn_vs.pp("v_proj"))?;
        let o_proj = linear_no_bias(cfg.heads * head_dim, cfg.dim, attn_vs.pp("o_proj"))?;

        let mlp_vs = vs.pp("mlp");
        let gate_proj = linear_no_bias(cfg.dim, cfg.hidden_dim, mlp_vs.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.dim, cfg.hidden_dim, mlp_vs.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.hidden_dim, cfg.dim, mlp_vs.pp("down_proj"))?;

        Ok(Self {
            input_layernorm,
            q_proj: LinearVariant::Plain(q_proj),
            k_proj: LinearVariant::Plain(k_proj),
            v_proj: LinearVariant::Plain(v_proj),
            o_proj: LinearVariant::Plain(o_proj),
            post_attention_layernorm,
            gate_proj: LinearVariant::Plain(gate_proj),
            up_proj: LinearVariant::Plain(up_proj),
            down_proj: LinearVariant::Plain(down_proj),
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim,
            n_rep,
        })
    }

    /// Replace this block's `Plain` linear projections with LoRA-wrapped
    /// counterparts according to `flags`. Idempotency: a layer already
    /// in the `Lora` variant errors out (double-wrap is a caller bug).
    ///
    /// Callers pass the per-block `VarBuilder` scoped so LoRA parameter
    /// names line up with the block index, e.g.
    /// `h.<i>.self_attn.q_proj.lora_a.weight`.
    fn wrap_lora(
        &mut self,
        cfg: &LoraConfig,
        flags: TinyLlamaWrapFlags,
        vs: VarBuilder,
    ) -> CandleResult<()> {
        let attn_vs = vs.pp("self_attn");
        let mlp_vs = vs.pp("mlp");
        if flags.q {
            wrap_variant_in_place(&mut self.q_proj, cfg, attn_vs.pp("q_proj"))?;
        }
        if flags.k {
            wrap_variant_in_place(&mut self.k_proj, cfg, attn_vs.pp("k_proj"))?;
        }
        if flags.v {
            wrap_variant_in_place(&mut self.v_proj, cfg, attn_vs.pp("v_proj"))?;
        }
        if flags.o {
            wrap_variant_in_place(&mut self.o_proj, cfg, attn_vs.pp("o_proj"))?;
        }
        if flags.gate {
            wrap_variant_in_place(&mut self.gate_proj, cfg, mlp_vs.pp("gate_proj"))?;
        }
        if flags.up {
            wrap_variant_in_place(&mut self.up_proj, cfg, mlp_vs.pp("up_proj"))?;
        }
        if flags.down {
            wrap_variant_in_place(&mut self.down_proj, cfg, mlp_vs.pp("down_proj"))?;
        }
        Ok(())
    }

    fn attention(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CandleResult<Tensor> {
        let (b, t, _d) = x.dims3()?;

        // Project to Q / K / V.
        let q = self.q_proj.forward(x)?; // [B, T, H  *Dh]
        let k = self.k_proj.forward(x)?; // [B, T, Hkv*Dh]
        let v = self.v_proj.forward(x)?; // [B, T, Hkv*Dh]

        // Reshape + transpose to [B, H_?, T, Dh].
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

        // Rotary embeddings on Q and K. `apply_rope` calls the
        // backward-safe `rope_slow` path, so gradients flow through
        // Q / K back to `q_proj` / `k_proj`.
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: broadcast the KV heads across the query groups.
        let k = repeat_kv(&k, self.n_rep)?; // [B, H, T, Dh]
        let v = repeat_kv(&v, self.n_rep)?; // [B, H, T, Dh]

        // Scaled dot-product [B, H, T, T].
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        scores = (scores / scale)?;

        // Causal mask: keep positions j <= i.
        let mask = mask.i((..t, ..t))?; // [T, T]
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?
            .to_dtype(scores.dtype())?
            .broadcast_as(scores.shape())?;
        let mask4 = mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(scores.shape())?;
        scores = mask4.where_cond(&scores, &neg_inf)?;
        let probs = ops::softmax_last_dim(&scores)?;

        // Weighted values → merge heads back to [B, T, D].
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o_proj.forward(&ctx)
    }

    /// SwiGLU MLP: `down_proj( silu(gate_proj(x)) * up_proj(x) )`.
    fn mlp(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let h = (gate * up)?;
        self.down_proj.forward(&h)
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CandleResult<Tensor> {
        let n = apply_slow_rms_norm(&self.input_layernorm, x)?;
        let a = self.attention(&n, cos, sin, mask)?;
        let x = (x + a)?;
        let n = apply_slow_rms_norm(&self.post_attention_layernorm, &x)?;
        let m = self.mlp(&n)?;
        x + m
    }
}

/// TinyLlama forward-only model.
///
/// Constructed via [`TinyLlamaModel::new`] (random init from a
/// [`VarBuilder`]) or [`TinyLlamaModel::from_pretrained`] (HuggingFace
/// warm-start). Training loops are added in Layer 3.
pub struct TinyLlamaModel {
    embed_tokens: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    /// Precomputed RoPE cos cache, `[ctx, head_dim / 2]`.
    rope_cos: Tensor,
    /// Precomputed RoPE sin cache, `[ctx, head_dim / 2]`.
    rope_sin: Tensor,
    /// Cached causal mask (`u8`, `1` on / below diagonal) sized to
    /// [`TinyLlamaConfig::ctx`].
    causal_mask: Tensor,
    cfg: TinyLlamaConfig,
}

impl TinyLlamaModel {
    /// Build a fresh TinyLlama model (random parameters) from a
    /// [`VarBuilder`]. The `VarBuilder` is expected to be scoped to the
    /// model root — i.e. `embed_tokens` etc. are registered directly
    /// under `vs`. To match the HuggingFace safetensors layout (which
    /// nests everything under `model.` except `lm_head`),
    /// [`Self::from_pretrained`] internally calls
    /// `Self::new_from_model_root(vs.pp("model"), lm_head_vs)` — see
    /// that method for the split VarBuilder path.
    pub fn new(cfg: &TinyLlamaConfig, vs: VarBuilder) -> CandleResult<Self> {
        Self::new_from_split(cfg, vs.clone(), vs)
    }

    /// Build a model where the `model.*` weights come from one
    /// [`VarBuilder`] and `lm_head.weight` from another. This matches
    /// the HF safetensors bundle layout: `model.<all>` + top-level
    /// `lm_head.weight`.
    ///
    /// Callers that construct a fresh `VarMap` (training / init) can
    /// pass the same builder for both arguments; the pretrained loader
    /// splits the root builder via `pp("model")` for the first.
    fn new_from_split(
        cfg: &TinyLlamaConfig,
        model_vs: VarBuilder,
        lm_head_vs: VarBuilder,
    ) -> CandleResult<Self> {
        if !cfg.dim.is_multiple_of(cfg.heads) {
            return Err(candle_core::Error::Msg(format!(
                "tinyllama: dim {} must be divisible by heads {}",
                cfg.dim, cfg.heads
            )));
        }
        if !cfg.heads.is_multiple_of(cfg.kv_heads) {
            return Err(candle_core::Error::Msg(format!(
                "tinyllama: heads {} must be divisible by kv_heads {}",
                cfg.heads, cfg.kv_heads
            )));
        }
        if cfg.kv_heads == 0 {
            return Err(candle_core::Error::Msg(
                "tinyllama: kv_heads must be >= 1".into(),
            ));
        }

        let head_dim = cfg.head_dim();
        let embed_tokens = embedding(cfg.vocab, cfg.dim, model_vs.pp("embed_tokens"))?;
        let layers_vs = model_vs.pp("layers");
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block::new(cfg, layers_vs.pp(i.to_string()))?);
        }
        let norm = rms_norm(cfg.dim, cfg.eps, model_vs.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.dim, cfg.vocab, lm_head_vs.pp("lm_head"))?;

        let (rope_cos, rope_sin) =
            build_rope_cache(cfg.ctx, head_dim, cfg.rope_theta, cfg.dtype, &cfg.device)?;
        let causal_mask = build_causal_mask(cfg.ctx, &cfg.device)?;

        Ok(Self {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope_cos,
            rope_sin,
            causal_mask,
            cfg: cfg.clone(),
        })
    }

    /// Return the configuration this model was built from.
    pub fn config(&self) -> &TinyLlamaConfig {
        &self.cfg
    }

    /// Load model weights from an on-disk safetensors bundle whose
    /// key layout matches the HF Llama convention (`model.*` prefix
    /// plus top-level `lm_head.weight` — the same layout that
    /// [`TinyLlamaModel::from_pretrained`] downloads and that
    /// [`super::lora::MergeableLora::export_merged`] emits).
    ///
    /// This is the plain-load path used by (a) the merged-bundle
    /// parity oracle in `tests/merged_export_parity_tinyllama.rs`
    /// and (b) future load-side integration that recognises
    /// `training_path == "merged"` and dispatches here instead of
    /// re-wrapping the model.
    ///
    /// # Errors
    ///
    /// `PretrainedError::Load` on safetensors parse failure or
    /// weight-name mismatch against the model shape.
    pub fn from_safetensors_file(
        cfg: &TinyLlamaConfig,
        path: &std::path::Path,
    ) -> Result<Self, PretrainedError> {
        // SAFETY: same discipline as `from_pretrained` — the file
        // must not be concurrently truncated while the mmap is
        // active. Callers hold the mmap for the lifetime of this
        // call.
        let root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&path.to_path_buf()),
                cfg.dtype,
                &cfg.device,
            )
            .map_err(|e| PretrainedError::Load(e.to_string()))?
        };
        Self::new_from_split(cfg, root.pp("model"), root)
            .map_err(|e| PretrainedError::Load(e.to_string()))
    }

    /// Load pretrained TinyLlama weights from HuggingFace on first use
    /// and cache the safetensors bundle at
    /// `cache_dir/base/<repo-basename>.safetensors`.
    ///
    /// Mirrors [`super::gpt2::Gpt2Model::from_pretrained`] — same
    /// first-use guard, same mmap-safetensors path, same error surface.
    pub fn from_pretrained(
        variant: &str,
        cfg: &TinyLlamaConfig,
        cache_dir: &std::path::Path,
    ) -> Result<Self, PretrainedError> {
        let repo = cfg
            .hf_repo()
            .ok_or_else(|| PretrainedError::UnknownPreset(variant.to_string()))?;

        let base_dir = cache_dir.join("base");
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| PretrainedError::CacheIo(format!("mkdir {:?}: {e}", base_dir)))?;
        let repo_leaf = repo.rsplit('/').next().unwrap_or(repo);
        let cache_path = base_dir.join(format!("{repo_leaf}.safetensors"));

        if !cache_path.exists() {
            tracing::info!(
                target: "algocline_nn::arch::tinyllama",
                repo,
                cache = %cache_path.display(),
                "downloading tinyllama pretrained weights"
            );
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| PretrainedError::HubApi(e.to_string()))?;
            let downloaded = api
                .model(repo.to_string())
                .get("model.safetensors")
                .map_err(|e| PretrainedError::Download(e.to_string()))?;
            std::fs::copy(&downloaded, &cache_path).map_err(|e| {
                PretrainedError::CacheIo(format!("copy {:?} -> {:?}: {e}", downloaded, cache_path))
            })?;
        }

        // SAFETY: candle exposes `from_mmaped_safetensors` as unsafe
        // because the caller must ensure the mmap-backed file is not
        // concurrently truncated. The cache path is only written once
        // above under a first-use guard; subsequent readers hold the
        // mmap for the lifetime of this call.
        let root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&cache_path),
                cfg.dtype,
                &cfg.device,
            )
            .map_err(|e| PretrainedError::Load(e.to_string()))?
        };
        // HF layout: `model.<...>` + top-level `lm_head.weight`.
        Self::new_from_split(cfg, root.pp("model"), root)
            .map_err(|e| PretrainedError::Load(e.to_string()))
    }

    /// Forward pass. Input `xs` is `[batch, seq]` of `u32` token ids;
    /// output is `[batch, seq, vocab]` — the raw logits.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (b, t) = xs.dims2()?;
        if t > self.cfg.ctx {
            return Err(candle_core::Error::Msg(format!(
                "tinyllama forward: seq {t} exceeds ctx {}",
                self.cfg.ctx
            )));
        }
        let mut h = self.embed_tokens.forward(xs)?; // [B, T, D]
        for block in &self.blocks {
            h = block.forward(&h, &self.rope_cos, &self.rope_sin, &self.causal_mask)?;
        }
        let h = apply_slow_rms_norm(&self.norm, &h)?;
        let logits = self.lm_head.forward(&h)?;
        debug_assert_eq!(logits.dims(), &[b, t, self.cfg.vocab]);
        Ok(logits)
    }

    /// Canonical LoRA target-module set for TinyLlama: attention
    /// Q/K/V/O plus MLP gate/up/down projections. Matches the
    /// HuggingFace / PEFT convention.
    ///
    /// Callers who want a narrower wrap should either pass their own
    /// list to [`super::lora::LoraConfig::with_targets`] or mutate the
    /// returned vector before feeding it in.
    pub fn default_lora_targets() -> Vec<String> {
        KNOWN_TARGET_MODULES_TINYLLAMA
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Wrap the model's per-block linear projections with LoRA
    /// low-rank updates. Mirrors [`super::gpt2::Gpt2Model::wrap_lora`]:
    /// the base parameters (already registered against the model's
    /// original `VarMap`) are held frozen inside each new
    /// [`LoraLinear`]; only the freshly-created `lora_a` / `lora_b`
    /// matrices are registered against the returned [`VarMap`], which
    /// the caller hands to the optimizer.
    ///
    /// # Errors
    ///
    /// - `cfg.target_modules` empty → error (mirrors GPT-2 shape).
    /// - Any entry outside the canonical TinyLlama set
    ///   (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
    ///   `up_proj`, `down_proj`) is rejected with the full list in the
    ///   error message.
    /// - `cfg.rank == 0` or `cfg.rank > min(in, out)` for any wrapped
    ///   layer propagates the underlying [`LoraLinear::wrap`] error.
    ///   GQA makes `k_proj` / `v_proj` narrower on the output side
    ///   (`kv_heads * head_dim`); on the 1.1B preset the minimum is
    ///   256, so any `rank <= 256` config is safe across all 7 targets.
    ///
    /// # LoRA VarMap layout
    ///
    /// Parameters register under
    /// `h.<i>.self_attn.q_proj.lora_a.weight`,
    /// `h.<i>.mlp.gate_proj.lora_b.weight`, etc.
    /// The `h.<i>` block prefix mirrors
    /// [`super::gpt2::Gpt2Model::wrap_lora`] so downstream tooling that
    /// iterates a LoRA VarMap can treat both architectures uniformly.
    /// The base [`VarMap`] passed to [`Self::new`] is byte-identical
    /// before / after this call by construction — the returned
    /// [`VarMap`] is freshly allocated and only `lora_a` / `lora_b`
    /// tensors register against it, while each base [`Linear`] is
    /// moved by value into [`LoraLinear::base`] without touching any
    /// [`VarBuilder`]. End-to-end verification lives in the
    /// `tinyllama_lora_merge_equivalence` integration test.
    pub fn wrap_lora(&mut self, cfg: &LoraConfig) -> CandleResult<VarMap> {
        if cfg.target_modules.is_empty() {
            return Err(candle_core::Error::Msg(
                "wrap_lora: target_modules is empty; nothing to wrap".into(),
            ));
        }
        for m in &cfg.target_modules {
            if !KNOWN_TARGET_MODULES_TINYLLAMA.contains(&m.as_str()) {
                return Err(candle_core::Error::Msg(format!(
                    "wrap_lora: unknown target module {m:?} \
                     (known: {KNOWN_TARGET_MODULES_TINYLLAMA:?})"
                )));
            }
        }
        let flags = TinyLlamaWrapFlags {
            q: cfg.target_modules.iter().any(|m| m == "q_proj"),
            k: cfg.target_modules.iter().any(|m| m == "k_proj"),
            v: cfg.target_modules.iter().any(|m| m == "v_proj"),
            o: cfg.target_modules.iter().any(|m| m == "o_proj"),
            gate: cfg.target_modules.iter().any(|m| m == "gate_proj"),
            up: cfg.target_modules.iter().any(|m| m == "up_proj"),
            down: cfg.target_modules.iter().any(|m| m == "down_proj"),
        };

        // Clone device / dtype up front so the VarBuilder borrows
        // locals rather than a slice of `self` (would collide with
        // `self.blocks.iter_mut()` below — same idiom as GPT-2).
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

/// Delegate to the inherent [`TinyLlamaModel::forward`] so the training
/// loop can drive any `M: candle_nn::Module` uniformly.
impl Module for TinyLlamaModel {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        TinyLlamaModel::forward(self, xs)
    }
}

impl crate::train::DeviceView for TinyLlamaModel {
    fn device(&self) -> &Device {
        &self.cfg.device
    }
}

/// Delegate to the inherent [`TinyLlamaModel::wrap_lora`] so the generic
/// [`crate::train::run_lora_ft`] loop can drive LoRA fine-tuning on any
/// `M: candle_nn::Module + crate::train::DeviceView + LoraWrappable`.
impl super::lora::LoraWrappable for TinyLlamaModel {
    fn wrap_lora(&mut self, cfg: &LoraConfig) -> CandleResult<VarMap> {
        TinyLlamaModel::wrap_lora(self, cfg)
    }
}

/// Emit a merged inference-ready weight bundle keyed by HF Llama
/// safetensors names. Every `LinearVariant::Lora` in a block is
/// collapsed via [`LoraLinear::merged_weight`]; every `Plain`
/// projection passes through unchanged. GQA shape is preserved:
/// `k_proj` / `v_proj` output dimension stays `kv_heads * head_dim`
/// (`256` on the 1.1B preset), never broadcast up to Q's shape.
///
/// The emitted key layout matches exactly what
/// [`TinyLlamaModel::from_pretrained`] reads (HF Llama layout with
/// `model.` prefix and `layers.<i>` per-block indexing), so the
/// bundle is a drop-in base for the same `TinyLlamaConfig`.
///
/// Layer 4a §3 Q2 — HF-native layout keys.
impl super::lora::MergeableLora for TinyLlamaModel {
    fn export_merged(&self) -> CandleResult<std::collections::HashMap<String, Tensor>> {
        let mut out: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

        // Top-level: token embedding + final norm + lm_head.
        // Naming mirrors `from_pretrained`'s HF Llama layout —
        // everything under `model.*` except `lm_head.weight`.
        out.insert(
            "model.embed_tokens.weight".into(),
            self.embed_tokens.embeddings().clone(),
        );

        // Per-block: input_layernorm, self_attn.{q,k,v,o}_proj,
        // post_attention_layernorm, mlp.{gate,up,down}_proj.
        // RmsNorm carries weight only (no bias); attention/MLP
        // linears are bias-less (`linear_no_bias`).
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("model.layers.{i}");

            out.insert(
                format!("{prefix}.input_layernorm.weight"),
                block.input_layernorm.weight().clone(),
            );
            out.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                block.post_attention_layernorm.weight().clone(),
            );

            // Attention projections (potentially LoRA-wrapped).
            let q = block.q_proj.merged_weight()?;
            out.insert(format!("{prefix}.self_attn.q_proj.weight"), q);
            let k = block.k_proj.merged_weight()?;
            out.insert(format!("{prefix}.self_attn.k_proj.weight"), k);
            let v = block.v_proj.merged_weight()?;
            out.insert(format!("{prefix}.self_attn.v_proj.weight"), v);
            let o = block.o_proj.merged_weight()?;
            out.insert(format!("{prefix}.self_attn.o_proj.weight"), o);

            // MLP projections (potentially LoRA-wrapped).
            let gate = block.gate_proj.merged_weight()?;
            out.insert(format!("{prefix}.mlp.gate_proj.weight"), gate);
            let up = block.up_proj.merged_weight()?;
            out.insert(format!("{prefix}.mlp.up_proj.weight"), up);
            let down = block.down_proj.merged_weight()?;
            out.insert(format!("{prefix}.mlp.down_proj.weight"), down);
        }

        // Final RmsNorm + lm_head (lm_head is at the top level,
        // outside the `model.*` subtree).
        out.insert("model.norm.weight".into(), self.norm.weight().clone());
        out.insert("lm_head.weight".into(), self.lm_head.weight().clone());

        Ok(out)
    }
}

/// Build a causal (lower-triangular) mask `[ctx, ctx]` where valid
/// (kept) positions are `1` and masked-out are `0`.
///
/// Returned as `u8` because candle's `Tensor::where_cond` only accepts
/// unsigned-integer condition tensors.
fn build_causal_mask(ctx: usize, device: &Device) -> CandleResult<Tensor> {
    let mut data = vec![0u8; ctx * ctx];
    for i in 0..ctx {
        for j in 0..=i {
            data[i * ctx + j] = 1;
        }
    }
    Tensor::from_vec(data, (ctx, ctx), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

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

        let diff: f32 = (y_fast - y_slow)?
            .abs()?
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar()?;
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
        let diff: f32 = (&x - &y)?
            .abs()?
            .max(0)?
            .max(0)?
            .max(0)?
            .max(0)?
            .to_scalar()?;
        assert_eq!(diff, 0.0);
        Ok(())
    }

    // -----------------------------------------------------------------
    // Layer 1b tests
    // -----------------------------------------------------------------

    #[test]
    fn tinyllama_1_1b_preset_shape() {
        let cfg = TinyLlamaConfig::tinyllama_1_1b();
        assert_eq!(cfg.layers, 22);
        assert_eq!(cfg.heads, 32);
        assert_eq!(cfg.kv_heads, 4);
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.hidden_dim, 5632);
        assert_eq!(cfg.ctx, 2048);
        assert_eq!(cfg.vocab, 32000);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(
            cfg.hf_repo(),
            Some("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        );
    }

    #[test]
    fn tinyllama_tiny_preset_has_no_hf_repo() {
        let cfg = TinyLlamaConfig::tiny();
        assert_eq!(cfg.layers, 2);
        assert_eq!(cfg.heads, 2);
        assert_eq!(cfg.kv_heads, 1);
        assert_eq!(cfg.dim, 64);
        assert_eq!(cfg.head_dim(), 32);
        // Tiny has no pretrained bundle.
        assert!(cfg.hf_repo().is_none());
    }

    #[test]
    fn tinyllama_from_variant_recognizes_aliases() {
        assert!(TinyLlamaConfig::from_variant("tinyllama-1.1b").is_some());
        assert!(TinyLlamaConfig::from_variant("1.1b").is_some());
        assert!(TinyLlamaConfig::from_variant("tinyllama-tiny").is_some());
        assert!(TinyLlamaConfig::from_variant("tiny").is_some());
        assert!(TinyLlamaConfig::from_variant("llama-2").is_none());
    }

    #[test]
    fn rejects_dim_not_divisible_by_heads() {
        let cfg = TinyLlamaConfig {
            layers: 2,
            heads: 3, // 8 % 3 != 0
            kv_heads: 1,
            dim: 8,
            hidden_dim: 16,
            ctx: 4,
            vocab: 10,
            rope_theta: 10_000.0,
            eps: 1e-5,
            dtype: DType::F32,
            device: Device::Cpu,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let msg = match TinyLlamaModel::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("divisible"), "unexpected error: {msg}");
    }

    #[test]
    fn rejects_heads_not_divisible_by_kv_heads() {
        let cfg = TinyLlamaConfig {
            layers: 2,
            heads: 4,
            kv_heads: 3, // 4 % 3 != 0
            dim: 32,
            hidden_dim: 64,
            ctx: 4,
            vocab: 16,
            rope_theta: 10_000.0,
            eps: 1e-5,
            dtype: DType::F32,
            device: Device::Cpu,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let msg = match TinyLlamaModel::new(&cfg, vs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            msg.contains("kv_heads"),
            "expected kv_heads error, got: {msg}"
        );
    }

    /// Tiny preset forward on CPU. Confirms the exact
    /// `[batch, seq, vocab]` output shape.
    #[test]
    fn tiny_forward_shape() {
        let cfg = TinyLlamaConfig::tiny();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = TinyLlamaModel::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);
    }

    #[test]
    fn tiny_forward_batch_shape() {
        let cfg = TinyLlamaConfig {
            layers: 1,
            heads: 2,
            kv_heads: 1,
            dim: 16,
            hidden_dim: 32,
            ctx: 4,
            vocab: 8,
            rope_theta: 10_000.0,
            eps: 1e-5,
            dtype: DType::F32,
            device: Device::Cpu,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = TinyLlamaModel::new(&cfg, vs).unwrap();
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 0], (2, 4), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[2, 4, 8]);
    }

    #[test]
    fn forward_rejects_seq_over_ctx() {
        let cfg = TinyLlamaConfig::tiny();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = TinyLlamaModel::new(&cfg, vs).unwrap();
        // ctx = 16, seq = 17
        let ids: Vec<u32> = (0u32..17).collect();
        let seq_len = ids.len();
        let ids = Tensor::from_vec(ids, (1, seq_len), &cfg.device).unwrap();
        let msg = match model.forward(&ids) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("exceeds ctx"), "unexpected error: {msg}");
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

    // ------------------------------------------------------------------
    // Layer 2 — LoRA wrap machinery
    //
    // Tests mirror the GPT-2 counterparts in `crates/algocline-nn/src/
    // arch/gpt2.rs` (`wrap_lora_rejects_empty_target_modules`,
    // `wrap_lora_rejects_unknown_target_module`,
    // `wrap_lora_populates_lora_vm_and_freezes_base`,
    // `wrap_lora_with_narrow_targets_only_wraps_selected_layers`) so
    // the two architectures stay behaviourally symmetric. The 5th test
    // (`default_lora_targets_matches_canonical_seven`) is TinyLlama-
    // specific: GPT-2 uses `LoraConfig::default_targets()` for its
    // canonical set, TinyLlama exposes it per-model instead so callers
    // don't accidentally cross the GPT-2 / Llama vocabulary boundary.
    // ------------------------------------------------------------------

    #[test]
    fn wrap_lora_rejects_empty_target_modules() {
        let cfg = TinyLlamaConfig::tiny();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();
        let lora = LoraConfig::with_targets(4, 8.0, Vec::<String>::new());
        let msg = match model.wrap_lora(&lora) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("empty"), "unexpected error: {msg}");
    }

    #[test]
    fn wrap_lora_rejects_unknown_target_module() {
        let cfg = TinyLlamaConfig::tiny();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();
        let lora = LoraConfig::with_targets(4, 8.0, vec!["typo_proj"]);
        let msg = match model.wrap_lora(&lora) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(msg.contains("unknown"), "unexpected error: {msg}");
        // The error should surface the full canonical list so a
        // mistyping caller sees the correct spellings.
        for name in KNOWN_TARGET_MODULES_TINYLLAMA {
            assert!(
                msg.contains(name),
                "error msg does not mention canonical target {name:?}: {msg}"
            );
        }
    }

    #[test]
    fn wrap_lora_populates_lora_vm_and_freezes_base() {
        let cfg = TinyLlamaConfig::tiny();
        let base_vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
        let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

        // Snapshot every base tensor before wrap.
        let base_before: Vec<Vec<f32>> = base_vm
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
            .collect();
        let base_var_count = base_vm.all_vars().len();

        // Tiny preset has k/v output dim = kv_heads * head_dim = 1*32 = 32,
        // so rank must be <= 32 across all 7 targets. Rank 4 is safely
        // within that on every projection.
        let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
        let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

        // 2 layers * 7 wrapped projections * (a + b) = 28 new Vars.
        assert_eq!(lora_vm.all_vars().len(), 28);
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

        // Forward still runs and returns the right shape after wrap.
        let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
        let logits = model.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab]);
    }

    #[test]
    fn wrap_lora_with_narrow_targets_only_wraps_selected_layers() {
        let cfg = TinyLlamaConfig::tiny();
        let vm = VarMap::new();
        let vs = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

        // Only wrap the MLP down-projection. Attention Q/K/V/O and
        // MLP gate/up all stay `Plain`.
        let lora_cfg = LoraConfig::with_targets(2, 4.0, vec!["down_proj"]);
        let lora_vm = model.wrap_lora(&lora_cfg).unwrap();
        // 2 layers * 1 wrapped linear * 2 vars (a + b) = 4.
        assert_eq!(lora_vm.all_vars().len(), 4);
    }

    #[test]
    fn default_lora_targets_matches_canonical_seven() {
        let targets = TinyLlamaModel::default_lora_targets();
        assert_eq!(targets.len(), 7);
        assert_eq!(
            targets,
            vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ]
        );
    }
}
