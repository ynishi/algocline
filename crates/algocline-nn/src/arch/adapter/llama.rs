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
//!
//! # Built-in cache vs caller-owned cache
//!
//! Two forward entries exist, and the difference is a concurrency
//! invariant rather than a convenience:
//!
//! - [`LlamaAdapter::forward`] drives the adapter's **built-in** cache.
//!   Correct for exactly one generation loop at a time — two loops
//!   sharing the handle would interleave their keys and values into the
//!   same cache and silently produce cross-contaminated logits.
//! - [`LlamaAdapter::forward_with_cache`] drives a **caller-owned**
//!   cache obtained from [`LlamaAdapter::new_cache`]. Each generation
//!   session holds its own cache, so concurrent sessions over one
//!   `Arc<LlamaAdapter>` (weights are read-only and shared) cannot mix
//!   state at all — the mixing is prevented structurally rather than by
//!   a caller-side convention.
//!
//! The built-in path is kept as the single-loop legacy entry so existing
//! callers keep working unchanged; new multi-session callers (the engine
//! bridge's Lua-facing generation session) use the caller-owned path.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama};

use super::{AdapterMeta, InferenceAdapter, LogitsShape};

/// The upstream KV cache type, re-exported so a downstream crate can
/// hold a caller-owned cache (see [`LlamaAdapter::new_cache`]) without
/// taking its own direct `candle-transformers` dependency just to name
/// the type.
pub use candle_transformers::models::llama::Cache as LlamaCache;

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
    /// Cache mode the built-in cache was constructed with, retained so
    /// [`Self::new_cache`] can hand out caches that behave identically
    /// to the built-in one. Without it a session-owned cache would
    /// silently differ from the adapter's own when a caller built the
    /// adapter with `use_kv_cache = false`.
    use_kv_cache: bool,
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
            use_kv_cache,
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

    /// Forward `tokens` (shape `[batch, seq]`) through the Llama stack
    /// using the adapter's **built-in** KV cache.
    ///
    /// `index_pos` is the position of the *first* token in this batch
    /// within the ongoing generation (`0` for the initial prompt,
    /// growing by `seq` after each call). Returns logits of shape
    /// `[batch, vocab]` — the upstream `Llama::forward` slices the
    /// last-token logits before returning, so callers doing
    /// token-by-token generation get the next-token distribution
    /// directly.
    ///
    /// This is the single-loop legacy path: the built-in cache is one
    /// piece of mutable state shared by every caller of this method, so
    /// two generation loops running against the same adapter would
    /// corrupt each other's context. A caller that needs more than one
    /// concurrent loop takes a cache of its own from
    /// [`Self::new_cache`] and drives [`Self::forward_with_cache`].
    pub fn forward(&self, tokens: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let mut cache = self.cache.lock().map_err(|e| {
            candle_core::Error::Msg(format!("alc.nn adapter::llama: cache lock poisoned: {e}"))
        })?;
        self.forward_with_cache(tokens, index_pos, &mut cache)
    }

    /// Build a fresh, empty KV cache for a new generation session.
    ///
    /// Constructed with the same `use_kv_cache` / dtype / config /
    /// device the adapter itself was built with, so a session-owned
    /// cache is behaviourally identical to the built-in one — it only
    /// differs in *who owns it*.
    ///
    /// Ownership is the point: weights stay shared (an
    /// `Arc<LlamaAdapter>` is read-only during inference) while the
    /// per-session mutable state is separate, which is what makes two
    /// concurrent generation loops structurally unable to mix contexts.
    pub fn new_cache(&self) -> CandleResult<Cache> {
        Cache::new(self.use_kv_cache, self.dtype, &self.config, &self.device)
    }

    /// Forward `tokens` (shape `[batch, seq]`) through the stack using a
    /// caller-owned `cache` instead of the adapter's built-in one.
    ///
    /// Same contract as [`Self::forward`] for `index_pos` and the
    /// returned `[batch, vocab]` logits; the only difference is which
    /// cache accumulates the keys and values. `index_pos` must be the
    /// number of tokens already forwarded *through this cache* — mixing
    /// positions from another session's stream is what the per-session
    /// cache exists to prevent.
    pub fn forward_with_cache(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> CandleResult<Tensor> {
        self.model.forward(tokens, index_pos, cache)
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

    // ─── caller-owned cache (generation sessions) ─────────────────

    /// Tiny adapter over random weights. The weights are fixed for the
    /// lifetime of the returned adapter, which is what lets the
    /// generation tests below compare token streams for equality.
    fn tiny_adapter() -> LlamaAdapter {
        let cfg = LlamaAdapterConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        LlamaAdapter::load(vb, cfg).expect("load tiny Llama")
    }

    /// Argmax over a `[1, vocab]` logits row — a greedy decode step,
    /// deterministic so two runs over the same weights must agree.
    fn greedy(logits: &Tensor) -> u32 {
        let row: Vec<f32> = logits.squeeze(0).unwrap().to_vec1().unwrap();
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        u32::try_from(best).unwrap()
    }

    /// One in-flight greedy generation over a caller-owned cache,
    /// mirroring what the engine bridge's session does.
    struct Stream<'a> {
        adapter: &'a LlamaAdapter,
        cache: Cache,
        tokens: Vec<u32>,
        forwarded: usize,
    }

    impl<'a> Stream<'a> {
        fn new(adapter: &'a LlamaAdapter, prompt: &[u32]) -> Self {
            Self {
                adapter,
                cache: adapter.new_cache().expect("new_cache"),
                tokens: prompt.to_vec(),
                forwarded: 0,
            }
        }

        /// Advance one greedy step, returning both the chosen token and
        /// the logits row it came from. The row is what the mixing test
        /// compares: two prompts can argmax to the same id under
        /// random-init weights, but their full distributions cannot
        /// coincide, so the row keeps the comparison non-vacuous.
        fn step(&mut self) -> (u32, Vec<f32>) {
            let pending = &self.tokens[self.forwarded..];
            let input =
                Tensor::from_slice(pending, (1, pending.len()), self.adapter.device()).unwrap();
            let logits = self
                .adapter
                .forward_with_cache(&input, self.forwarded, &mut self.cache)
                .expect("forward_with_cache");
            self.forwarded = self.tokens.len();
            let next = greedy(&logits);
            self.tokens.push(next);
            (next, logits.squeeze(0).unwrap().to_vec1().unwrap())
        }
    }

    /// Largest absolute element-wise gap between two logits streams.
    fn max_gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len(), "stream length mismatch");
        let mut worst = 0.0f32;
        for (ra, rb) in a.iter().zip(b) {
            assert_eq!(ra.len(), rb.len(), "vocab mismatch");
            for (x, y) in ra.iter().zip(rb) {
                worst = worst.max((x - y).abs());
            }
        }
        worst
    }

    /// A caller-owned cache reproduces the built-in cache's generation
    /// exactly: same weights, same prompt, same greedy decisions. Both
    /// caches start empty, so any divergence would mean
    /// `forward_with_cache` is not driving the same computation (or that
    /// the fresh cache inherited built-in state).
    #[test]
    fn caller_owned_cache_reproduces_builtin_cache_generation() {
        let adapter = tiny_adapter();
        let prompt = [1u32, 2, 3];

        // Built-in cache path (the legacy single-loop entry).
        let mut tokens = prompt.to_vec();
        let mut forwarded = 0usize;
        let mut builtin = Vec::new();
        for _ in 0..4 {
            let pending = &tokens[forwarded..];
            let input = Tensor::from_slice(pending, (1, pending.len()), adapter.device()).unwrap();
            let logits = adapter.forward(&input, forwarded).expect("builtin forward");
            assert_eq!(logits.dims(), &[1, adapter.vocab()]);
            forwarded = tokens.len();
            let next = greedy(&logits);
            tokens.push(next);
            builtin.push(next);
        }

        // Caller-owned cache path over the same adapter.
        let mut stream = Stream::new(&adapter, &prompt);
        let owned: Vec<u32> = (0..4).map(|_| stream.step().0).collect();

        assert_eq!(owned, builtin);
    }

    /// Two caller-owned caches advanced in lockstep must produce exactly
    /// what each produces on its own. This is the structural claim the
    /// session-owned cache exists to make: interleaving two generation
    /// loops over one set of weights cannot contaminate either context.
    #[test]
    fn interleaved_caches_match_their_solo_runs() {
        let adapter = tiny_adapter();
        let prompt_a = [1u32, 2, 3];
        let prompt_b = [10u32, 11, 12];

        let solo_a: Vec<(u32, Vec<f32>)> = {
            let mut s = Stream::new(&adapter, &prompt_a);
            (0..4).map(|_| s.step()).collect()
        };
        let solo_b: Vec<(u32, Vec<f32>)> = {
            let mut s = Stream::new(&adapter, &prompt_b);
            (0..4).map(|_| s.step()).collect()
        };

        let mut a = Stream::new(&adapter, &prompt_a);
        let mut b = Stream::new(&adapter, &prompt_b);
        let mut mixed_a = Vec::new();
        let mut mixed_b = Vec::new();
        for _ in 0..4 {
            mixed_a.push(a.step());
            mixed_b.push(b.step());
        }

        let tokens = |run: &[(u32, Vec<f32>)]| run.iter().map(|(t, _)| *t).collect::<Vec<_>>();
        let rows = |run: &[(u32, Vec<f32>)]| run.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>();

        assert_eq!(
            tokens(&mixed_a),
            tokens(&solo_a),
            "stream A drifted when interleaved with B"
        );
        assert_eq!(
            tokens(&mixed_b),
            tokens(&solo_b),
            "stream B drifted when interleaved with A"
        );
        assert!(
            max_gap(&rows(&mixed_a), &rows(&solo_a)) < 1e-5,
            "stream A logits drifted when interleaved with B"
        );
        assert!(
            max_gap(&rows(&mixed_b), &rows(&solo_b)) < 1e-5,
            "stream B logits drifted when interleaved with A"
        );
        // Guard against the assertions above holding vacuously: if both
        // prompts produced the same distributions, contamination would
        // be undetectable and the test would prove nothing.
        assert!(
            max_gap(&rows(&solo_a), &rows(&solo_b)) > 1e-4,
            "test prompts must produce different logits for the mixing claim to mean anything"
        );
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
