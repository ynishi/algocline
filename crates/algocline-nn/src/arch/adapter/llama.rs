//! Llama-family inference adapter over `candle_transformers::models::llama`.
//!
//! Wraps the upstream `Llama` stack (RMSNorm + RoPE + GQA + SwiGLU) as
//! an inference handle at the `alc.nn` layer boundary. The upstream
//! implementation is loaded from a `VarBuilder` — no `VarMap` — so
//! this adapter deliberately does not participate in the training loop
//! (`alc.nn.trainer.*` refuses handles without a `VarMap`).
//!
//! The adapter owns the mutable KV [`Cache`] alongside the stack so a
//! caller who threads through the same handle across successive
//! `forward` calls (e.g. token-by-token generation) reuses the same
//! cache without exposing candle-transformers types to Lua.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};

use super::{AdapterMeta, InferenceAdapter, LogitsShape};

/// Build-time configuration for a [`LlamaAdapter`].
///
/// `variant` is the caller-facing preset id (e.g. `"tinyllama-1.1b"`,
/// `"llama-3.2-1b"`, `"7b-v2"`); it is carried unmodified so the
/// engine bridge can echo it back through the `Gpt2Handle`-shaped
/// metadata block. `config` is the upstream `candle-transformers`
/// `Config` — either a preset builder (`Config::config_7b_v2`) or a
/// hand-authored struct for architectures without a shipped preset.
#[derive(Debug, Clone)]
pub struct LlamaAdapterConfig {
    /// Preset id echoed back to the caller.
    pub variant: String,
    /// Device the parameters and cache tensors live on.
    pub device: Device,
    /// Weight precision.
    pub dtype: DType,
    /// Upstream Llama configuration (layer / head / dim / rope / rms
    /// parameters). Owned by value so the adapter never borrows across
    /// the mlua boundary.
    pub config: Config,
    /// Whether the KV cache is populated across successive `forward`
    /// calls. `true` for token-by-token generation; set to `false` for
    /// one-shot forward benchmarks that never revisit a position.
    ///
    /// Every constructor here defaults this to `true`, matching the
    /// engine bridge's `opts.use_kv_cache` default
    /// (`bridge/nn_card.rs`, `build_llama_handle`). The two used to
    /// disagree — the bridge defaulted to `true` while `from_variant` /
    /// [`Self::tiny`] returned `false` — so a Rust caller constructing
    /// a config directly got different behaviour from a Lua caller
    /// reaching the same adapter through the preset. Keep both defaults
    /// in lockstep when changing either.
    pub use_kv_cache: bool,
}

impl LlamaAdapterConfig {
    /// Resolve a variant name to the matching config, returning
    /// `None` for unknown names.
    ///
    /// Accepted names:
    /// - `"tiny"` / `"llama-tiny"` — [`Self::tiny`], the CPU smoke
    ///   config (2 layers / 2 heads / hidden 32).
    /// - `"7b-v1"` / `"llama-7b-v1"` — the upstream
    ///   `candle_transformers::models::llama::Config::config_7b_v1`
    ///   preset (32 layers / 32 heads / hidden 4096, LLaMA 1 dtype
    ///   norms).
    /// - `"7b-v2"` / `"llama-7b-v2"` — the upstream
    ///   `Config::config_7b_v2` preset (LLaMA 2 dtype norms).
    ///
    /// `flash_attn` is threaded through so callers on GPU builds with
    /// the `flash-attn` feature enabled can opt into fused attention.
    pub fn from_variant(variant: &str, flash_attn: bool) -> Option<Self> {
        match variant {
            "tiny" | "llama-tiny" => Some(Self::tiny()),
            "7b-v1" | "llama-7b-v1" => Some(Self {
                variant: variant.to_string(),
                device: Device::Cpu,
                dtype: DType::F32,
                config: Config::config_7b_v1(flash_attn),
                use_kv_cache: true,
            }),
            "7b-v2" | "llama-7b-v2" => Some(Self {
                variant: variant.to_string(),
                device: Device::Cpu,
                dtype: DType::F32,
                config: Config::config_7b_v2(flash_attn),
                use_kv_cache: true,
            }),
            _ => None,
        }
    }

    /// Build a synthetic `tiny` config suitable for CPU smoke tests.
    ///
    /// Mirrors the [`super::super::gpt2::Gpt2Config::tiny`] convention
    /// so the arch::adapter::* modules can be exercised without a real
    /// pretrained bundle. Shape: 2 layers, 2 heads, hidden 32, vocab
    /// 64, ctx 16, `f32` on CPU, no flash-attn.
    pub fn tiny() -> Self {
        let config = Config {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 16,
            tie_word_embeddings: false,
        };
        Self {
            variant: "llama-tiny".into(),
            device: Device::Cpu,
            dtype: DType::F32,
            config,
            use_kv_cache: true,
        }
    }
}

/// Inference-only Llama handle.
///
/// Holds the upstream `Llama` stack, the associated mutable KV
/// [`Cache`], and the metadata a caller needs to identify the loaded
/// weights.
///
/// The `Mutex` around the cache exists solely to satisfy the mlua
/// `send` feature: candle's tensors are already `Send` + `Sync`, but
/// the cache is a mutable candle-transformers value that
/// [`Llama::forward`] takes by `&mut`. Because the underlying VM is
/// single-threaded there is no real cross-thread traffic through the
/// mutex.
pub struct LlamaAdapter {
    model: Llama,
    cache: Mutex<Cache>,
    config: Config,
    variant: String,
    device: Device,
    dtype: DType,
}

impl LlamaAdapter {
    /// Load the Llama stack from a caller-provided [`VarBuilder`].
    ///
    /// Mirrors the trainable [`super::super::gpt2::Gpt2Model::new`]
    /// shape so the engine bridge can construct both adapters through
    /// a uniform path.
    pub fn load(vb: VarBuilder, cfg: LlamaAdapterConfig) -> CandleResult<Self> {
        let LlamaAdapterConfig {
            variant,
            device,
            dtype,
            config,
            use_kv_cache,
        } = cfg;
        let cache = Cache::new(use_kv_cache, dtype, &config, &device)?;
        let model = Llama::load(vb, &config)?;
        Ok(Self {
            model,
            cache: Mutex::new(cache),
            config,
            variant,
            device,
            dtype,
        })
    }

    /// Load the Llama stack from a list of `.safetensors` file paths.
    ///
    /// Each path is mmap'd through
    /// [`VarBuilder::from_mmaped_safetensors`]; the caller is
    /// responsible for enumerating every shard of a sharded bundle
    /// (e.g. `model-00001-of-00002.safetensors` +
    /// `model-00002-of-00002.safetensors`). The dtype used to build the
    /// VarBuilder is the adapter's own `dtype`, so the safetensors
    /// tensors are up-/down-cast on load when they differ.
    pub fn from_safetensors_files(
        paths: &[impl AsRef<Path>],
        cfg: LlamaAdapterConfig,
    ) -> CandleResult<Self> {
        // SAFETY: `VarBuilder::from_mmaped_safetensors` is `unsafe`
        // because it mmaps the caller's files; the caller warrants
        // that the files are not concurrently mutated. We forward the
        // safety obligation to the adapter's public contract (documented
        // in the caller-facing preset binding).
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, cfg.dtype, &cfg.device)? };
        Self::load(vb, cfg)
    }

    /// Forward `tokens` (shape `[batch, seq]`) through the Llama stack.
    ///
    /// `index_pos` is the position of the *first* token in this batch
    /// within the ongoing generation (`0` for the initial prompt,
    /// growing by `seq` after each call). Returns logits of shape
    /// `[batch, vocab]` — the upstream `Llama::forward` slices the
    /// last-token logits before returning, so callers doing
    /// token-by-token generation get the next-token distribution
    /// directly.
    pub fn forward(&self, tokens: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let mut cache = self.cache.lock().map_err(|e| {
            candle_core::Error::Msg(format!("alc.nn adapter::llama: cache lock poisoned: {e}"))
        })?;
        self.model.forward(tokens, index_pos, &mut cache)
    }

    /// Caller-facing preset id (e.g. `"tinyllama-1.1b"`).
    pub fn variant(&self) -> &str {
        &self.variant
    }

    /// Number of transformer blocks.
    pub fn layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    /// Number of query heads.
    pub fn heads(&self) -> usize {
        self.config.num_attention_heads
    }

    /// Number of key/value heads (equals `heads` when the config does
    /// not use grouped-query attention).
    pub fn kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    /// Hidden dimension.
    pub fn dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Vocabulary size.
    pub fn vocab(&self) -> usize {
        self.config.vocab_size
    }

    /// Maximum sequence length (positional context).
    pub fn ctx(&self) -> usize {
        self.config.max_position_embeddings
    }

    /// Device the model parameters live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Weight precision.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl InferenceAdapter for LlamaAdapter {
    /// Describe the loaded stack for the engine bridge.
    ///
    /// Every field is derived from the upstream `Config` captured at
    /// load time, so this is the single place the bridge needs to read
    /// to build its Lua-facing handle — it never reaches into
    /// `candle_transformers`' config itself.
    ///
    /// `logits` is [`LogitsShape::LastToken`]: upstream `Llama::forward`
    /// slices the final position before returning, so a `[batch, seq]`
    /// input yields `[batch, vocab]`.
    fn meta(&self) -> AdapterMeta {
        AdapterMeta {
            family: "llama",
            variant: self.variant.clone(),
            layers: self.layers(),
            heads: self.heads(),
            kv_heads: self.kv_heads(),
            dim: self.dim(),
            ctx: self.ctx(),
            vocab: self.vocab(),
            device: self.device.clone(),
            dtype: self.dtype,
            logits: LogitsShape::LastToken,
        }
    }

    /// Delegates to the inherent [`LlamaAdapter::forward`], spelled with
    /// the fully-qualified form so the call cannot be read as recursing
    /// through this trait method.
    fn forward(&self, tokens: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        LlamaAdapter::forward(self, tokens, index_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use candle_nn::{VarBuilder, VarMap};

    /// Build a tiny Llama over random weights and confirm the adapter
    /// wires the upstream `Llama::forward` correctly. This is the
    /// smallest test that exercises `load` + `forward` end-to-end
    /// without requiring a pretrained bundle on disk.
    #[test]
    fn tiny_load_and_forward_returns_batch_vocab_logits() {
        let cfg = LlamaAdapterConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg).expect("load tiny Llama");

        // Feed a 1x4 batch of token ids and confirm the returned
        // logits are shaped [batch, vocab].
        let tokens = Tensor::new(&[[1u32, 2, 3, 4]], adapter.device()).unwrap();
        let logits = adapter.forward(&tokens, 0).expect("forward");
        assert_eq!(logits.dims(), &[1, adapter.vocab()]);
        assert_eq!(adapter.layers(), 2);
        assert_eq!(adapter.heads(), 2);
        assert_eq!(adapter.kv_heads(), 2);
        assert_eq!(adapter.dim(), 32);
        assert_eq!(adapter.vocab(), 64);
        assert_eq!(adapter.ctx(), 16);
        assert_eq!(adapter.variant(), "llama-tiny");
    }

    /// The adapter records but does not enforce `use_kv_cache = true`:
    /// two successive `forward` calls with disjoint index positions
    /// must accumulate into the cache without panicking or shape
    /// errors.
    #[test]
    fn cached_forward_advances_index_pos_without_error() {
        let mut cfg = LlamaAdapterConfig::tiny();
        cfg.use_kv_cache = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg).expect("load tiny Llama");

        let prompt = Tensor::new(&[[1u32, 2, 3]], adapter.device()).unwrap();
        let out1 = adapter.forward(&prompt, 0).expect("first forward");
        assert_eq!(out1.dims(), &[1, adapter.vocab()]);

        let next = Tensor::new(&[[4u32]], adapter.device()).unwrap();
        let out2 = adapter.forward(&next, 3).expect("cached second forward");
        assert_eq!(out2.dims(), &[1, adapter.vocab()]);
    }

    /// [`InferenceAdapter::meta`] must agree with the inherent
    /// accessors it is derived from. The bridge builds its Lua-facing
    /// handle from `meta()` alone, so a drift between the two would
    /// silently mis-report shape parameters to a Lua caller.
    #[test]
    fn meta_agrees_with_inherent_accessors() {
        let cfg = LlamaAdapterConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg).expect("load tiny Llama");

        let meta = InferenceAdapter::meta(&adapter);
        assert_eq!(meta.family, "llama");
        assert_eq!(meta.variant, adapter.variant());
        assert_eq!(meta.layers, adapter.layers());
        assert_eq!(meta.heads, adapter.heads());
        assert_eq!(meta.kv_heads, adapter.kv_heads());
        assert_eq!(meta.dim, adapter.dim());
        assert_eq!(meta.ctx, adapter.ctx());
        assert_eq!(meta.vocab, adapter.vocab());
        assert_eq!(meta.dtype, adapter.dtype());
        assert_eq!(meta.logits, LogitsShape::LastToken);
    }

    /// The declared [`LogitsShape`] must predict the real `forward`
    /// output dims. This is the invariant the bridge's `forward_shape`
    /// binding relies on once it computes shapes from `meta()` instead
    /// of an arch-specific `match`.
    #[test]
    fn declared_logits_shape_predicts_forward_dims() {
        let cfg = LlamaAdapterConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let adapter = LlamaAdapter::load(vb, cfg).expect("load tiny Llama");

        let (batch, seq) = (1usize, 4usize);
        let tokens = Tensor::new(&[[1u32, 2, 3, 4]], adapter.device()).unwrap();
        let logits = InferenceAdapter::forward(&adapter, &tokens, 0).expect("forward");

        let meta = InferenceAdapter::meta(&adapter);
        assert_eq!(logits.dims(), meta.logits.dims(batch, seq, meta.vocab));
    }

    /// Every constructor defaults `use_kv_cache` to `true` so a Rust
    /// caller matches the engine bridge's `opts.use_kv_cache` default.
    /// The two disagreed before (`false` here, `true` on the bridge).
    #[test]
    fn kv_cache_defaults_to_enabled_across_constructors() {
        assert!(LlamaAdapterConfig::tiny().use_kv_cache);
        for variant in ["tiny", "7b-v1", "7b-v2"] {
            let cfg = LlamaAdapterConfig::from_variant(variant, false)
                .unwrap_or_else(|| panic!("variant {variant} should resolve"));
            assert!(
                cfg.use_kv_cache,
                "variant {variant} must default use_kv_cache to true"
            );
        }
    }
}
