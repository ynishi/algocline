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
    embedding, layer_norm, linear, ops, Embedding, LayerNorm, Module, VarBuilder, VarMap,
};

// `LoraLinear` is imported for the intra-doc links (`[`LoraLinear`]`) in the
// wrap_lora docs below; the wrap helper itself lives in `arch::lora` since
// TinyLlama needs the same swap-in-place idiom.
#[allow(unused_imports)]
use super::lora::{wrap_variant_in_place, LinearVariant, LoraConfig, LoraLinear};

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
struct Block {
    ln_1: LayerNorm,
    c_attn: LinearVariant,
    c_proj: LinearVariant,
    ln_2: LayerNorm,
    mlp_c_fc: LinearVariant,
    mlp_c_proj: LinearVariant,
    heads: usize,
    head_dim: usize,
}

impl Block {
    fn new(cfg: &Gpt2Config, vs: VarBuilder) -> CandleResult<Self> {
        let head_dim = cfg.dim / cfg.heads;
        let ln_1 = layer_norm(cfg.dim, cfg.eps, vs.pp("ln_1"))?;
        let attn_vs = vs.pp("attn");
        let c_attn = linear(cfg.dim, 3 * cfg.dim, attn_vs.pp("c_attn"))?;
        let c_proj = linear(cfg.dim, cfg.dim, attn_vs.pp("c_proj"))?;
        let ln_2 = layer_norm(cfg.dim, cfg.eps, vs.pp("ln_2"))?;
        let mlp_vs = vs.pp("mlp");
        let mlp_c_fc = linear(cfg.dim, 4 * cfg.dim, mlp_vs.pp("c_fc"))?;
        let mlp_c_proj = linear(4 * cfg.dim, cfg.dim, mlp_vs.pp("c_proj"))?;
        Ok(Self {
            ln_1,
            c_attn: LinearVariant::Plain(c_attn),
            c_proj: LinearVariant::Plain(c_proj),
            ln_2,
            mlp_c_fc: LinearVariant::Plain(mlp_c_fc),
            mlp_c_proj: LinearVariant::Plain(mlp_c_proj),
            heads: cfg.heads,
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
        let mlp_vs = vs.pp("mlp");
        if flags.qkv {
            wrap_variant_in_place(&mut self.c_attn, cfg, attn_vs.pp("c_attn"))?;
        }
        if flags.o {
            wrap_variant_in_place(&mut self.c_proj, cfg, attn_vs.pp("c_proj"))?;
        }
        if flags.up {
            wrap_variant_in_place(&mut self.mlp_c_fc, cfg, mlp_vs.pp("c_fc"))?;
        }
        if flags.down {
            wrap_variant_in_place(&mut self.mlp_c_proj, cfg, mlp_vs.pp("c_proj"))?;
        }
        Ok(())
    }

    fn attention(&self, x: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
        // x: [B, T, D]
        let (b, t, _d) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?; // [B, T, 3D]
        let qkv = qkv.reshape((b, t, 3, self.heads, self.head_dim))?;
        // Split into Q/K/V — each [B, T, H, Dh]. Then transpose to [B, H, T, Dh].
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product: [B, H, T, T].
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

        // [B, H, T, Dh] then merge back to [B, T, D].
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.c_proj.forward(&ctx)
    }

    fn mlp(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.mlp_c_fc.forward(x)?;
        // GELU (approximate variant matches HF GPT-2). `gelu` is the
        // exact form; both are within 1e-3 for our purposes.
        let h = h.gelu()?;
        self.mlp_c_proj.forward(&h)
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
        let n = apply_slow_layer_norm(&self.ln_1, x)?;
        let a = self.attention(&n, mask)?;
        let x = (x + a)?;
        let n = apply_slow_layer_norm(&self.ln_2, &x)?;
        let m = self.mlp(&n)?;
        x + m
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
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    /// Cached causal mask (`1.0` below the diagonal, `0.0` above) sized
    /// to [`Gpt2Config::ctx`] so per-forward mask allocation is avoided.
    causal_mask: Tensor,
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
        let wte = embedding(cfg.vocab, cfg.dim, vs.pp("wte"))?;
        let wpe = embedding(cfg.ctx, cfg.dim, vs.pp("wpe"))?;
        let h_vs = vs.pp("h");
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block::new(cfg, h_vs.pp(i.to_string()))?);
        }
        let ln_f = layer_norm(cfg.dim, cfg.eps, vs.pp("ln_f"))?;
        let causal_mask = build_causal_mask(cfg.ctx, &cfg.device, cfg.dtype)?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            causal_mask,
            cfg: cfg.clone(),
        })
    }

    /// Return the configuration this model was built from.
    pub fn config(&self) -> &Gpt2Config {
        &self.cfg
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

    pub fn from_pretrained(
        variant: &str,
        cfg: &Gpt2Config,
        cache_dir: &std::path::Path,
    ) -> Result<Self, PretrainedError> {
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
        let (b, t) = xs.dims2()?;
        if t > self.cfg.ctx {
            return Err(candle_core::Error::Msg(format!(
                "gpt2 forward: seq {t} exceeds ctx {}",
                self.cfg.ctx
            )));
        }
        let tok_emb = self.wte.forward(xs)?; // [B, T, D]
        let pos_ids = Tensor::arange(0u32, t as u32, xs.device())?; // [T]
        let pos_emb = self.wpe.forward(&pos_ids)?; // [T, D]
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(tok_emb.shape())?;
        let mut h = (tok_emb + pos_emb)?;
        for block in &self.blocks {
            h = block.forward(&h, &self.causal_mask)?;
        }
        let h = apply_slow_layer_norm(&self.ln_f, &h)?; // [B, T, D]
                                                        // Tied LM head: logits = h @ wte.weight^T.
        let w = self.wte.embeddings(); // [V, D]
        let logits = h.broadcast_matmul(&w.t()?)?; // [B, T, V]
        debug_assert_eq!(logits.dims(), &[b, t, self.cfg.vocab]);
        Ok(logits)
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
        let mut out: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Top-level: token + positional embeddings (LM head is tied
        // to `wte`, so no separate `lm_head.weight` — matches
        // Gpt2Model::new which uses wte for both).
        out.insert("wte.weight".into(), self.wte.embeddings().clone());
        out.insert("wpe.weight".into(), self.wpe.embeddings().clone());

        // Per-block: ln_1, attn.c_attn, attn.c_proj, ln_2,
        // mlp.c_fc, mlp.c_proj. Naming mirrors Block::new's
        // VarBuilder path (`h.<i>.<field>`).
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("h.{i}");

            // LayerNorms carry both weight and bias.
            out.insert(format!("{prefix}.ln_1.weight"), block.ln_1.weight().clone());
            if let Some(b) = block.ln_1.bias() {
                out.insert(format!("{prefix}.ln_1.bias"), b.clone());
            }
            out.insert(format!("{prefix}.ln_2.weight"), block.ln_2.weight().clone());
            if let Some(b) = block.ln_2.bias() {
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

            // MLP linears (potentially LoRA-wrapped).
            let mlp_c_fc_w = block.mlp_c_fc.merged_weight()?;
            out.insert(format!("{prefix}.mlp.c_fc.weight"), mlp_c_fc_w);
            if let Some(b) = block.mlp_c_fc.bias() {
                out.insert(format!("{prefix}.mlp.c_fc.bias"), b.clone());
            }
            let mlp_c_proj_w = block.mlp_c_proj.merged_weight()?;
            out.insert(format!("{prefix}.mlp.c_proj.weight"), mlp_c_proj_w);
            if let Some(b) = block.mlp_c_proj.bias() {
                out.insert(format!("{prefix}.mlp.c_proj.bias"), b.clone());
            }
        }

        // Final LayerNorm.
        out.insert("ln_f.weight".into(), self.ln_f.weight().clone());
        if let Some(b) = self.ln_f.bias() {
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

/// Build a causal (lower-triangular) mask `[ctx, ctx]` where valid
/// (kept) positions are `1` and masked-out are `0`.
///
/// Returned as `u8` because candle's `Tensor::where_cond` only accepts
/// unsigned-integer condition tensors; the attention path never uses
/// the mask as a numeric multiplier (it drives `where_cond` between
/// the scaled scores and `-inf`), so the concrete `dtype` of the model
/// weights is irrelevant here.
fn build_causal_mask(ctx: usize, device: &Device, _dtype: DType) -> CandleResult<Tensor> {
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
}
