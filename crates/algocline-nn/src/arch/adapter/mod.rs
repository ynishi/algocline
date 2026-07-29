//! Inference-only adapters over `candle-transformers` model
//! implementations.
//!
//! The Card foundation trained architectures (currently GPT-2) live in
//! sibling `arch::*` modules and expose a `VarMap` so
//! `algocline_nn::train::*` can drive optimizer steps against them.
//! This module holds the **inference-only** counterpart: thin wrappers
//! around `candle_transformers::models::*` that answer at the same layer
//! boundary as the trainable handles but decline to expose a `VarMap`
//! (candle-transformers stacks are loaded from a
//! `VarBuilder::from_mmaped_safetensors` reader, not a `VarMap`).
//!
//! Callers reach these adapters through the engine bridge presets
//! (`alc.nn.preset.llama` / `alc.nn.preset.qwen` / ...), which build a
//! `<Arch>Handle` `UserData` and expose the same registry / `alc.llm`
//! `role="nn"` fast path that GPT-2 handles use. Because the adapters
//! do not own a `VarMap`, the trainer bindings (`alc.nn.trainer.*`)
//! surface a clear Lua-side error rather than attempting a training
//! loop.
//!
//! # Adapter contract
//!
//! Every adapter here implements [`InferenceAdapter`], which fixes the
//! two members the engine bridge dispatches on:
//!
//! - [`InferenceAdapter::meta`] — an [`AdapterMeta`] describing the
//!   loaded stack (family / variant / shape parameters / device /
//!   dtype / logits shape). The bridge builds its Lua-facing handle
//!   from this value instead of re-reading the upstream config, so
//!   adding an arch does not mean copying nine accessors into a new
//!   handle struct.
//! - [`InferenceAdapter::forward`] — `(tokens, index_pos) -> logits`,
//!   with the caller-visible output shape declared by
//!   [`AdapterMeta::logits`] rather than left to per-arch prose.
//!
//! Construction stays **outside** the trait. Each adapter keeps its own
//! `Config` struct, a `load(vb, config)` constructor, and a
//! `from_safetensors_files(paths, config)` constructor, because the
//! `VarBuilder` origin differs per arch (single file / sharded mmap /
//! GGUF) and a `Self: Sized` constructor in the trait would rule out
//! holding adapters as `dyn InferenceAdapter`. The engine bridge's
//! `ARCH_OPS` table owns that per-arch construction dispatch; this
//! trait owns behaviour only.
//!
//! # Future additions
//!
//! `arch::adapter::qwen`, `arch::adapter::phi`, and
//! `arch::adapter::gemma` will land as sibling modules on the same
//! shape once a caller needs each. Each is an `InferenceAdapter` impl
//! plus one `ARCH_OPS` entry on the bridge side. HuggingFace-hub
//! warm-start (`from_pretrained`) is deliberately deferred out of the
//! initial module cut so a downstream caller can drive the adapter from
//! local files (or an already-cached HF snapshot) without pulling
//! network into the unit-test path.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub mod llama;

pub use llama::{LlamaAdapter, LlamaAdapterConfig};

/// Caller-visible shape of the logits an adapter's `forward` returns.
///
/// Upstream stacks disagree here: `candle_transformers`'s Llama slices
/// the last-token row before returning, while the trainable `arch::*`
/// models return logits for every position. Carrying the difference as
/// a value (rather than as per-arch prose plus a `match` on the bridge
/// side) lets a caller compute the expected output shape without
/// branching on the architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsShape {
    /// `[batch, vocab]` — only the final position's distribution, ready
    /// to sample the next token from directly.
    LastToken,
    /// `[batch, seq, vocab]` — one distribution per input position, as
    /// a training loss needs.
    FullSeq,
}

impl LogitsShape {
    /// Dimensions a forward over `[batch, seq]` token ids produces for
    /// a model with the given vocabulary size.
    pub fn dims(self, batch: usize, seq: usize, vocab: usize) -> Vec<usize> {
        match self {
            Self::LastToken => vec![batch, vocab],
            Self::FullSeq => vec![batch, seq, vocab],
        }
    }
}

/// Architecture-neutral description of a loaded model.
///
/// Produced by [`InferenceAdapter::meta`] and consumed by the engine
/// bridge when it builds the Lua-facing handle. Owned (not borrowed)
/// because the upstream configs an adapter wraps are private
/// implementation detail — an adapter is free to synthesise these
/// numbers rather than hold a config struct.
#[derive(Debug, Clone)]
pub struct AdapterMeta {
    /// Architecture family, matching an entry in
    /// [`crate::card::SUPPORTED_ARCHITECTURE_FAMILIES`] (e.g.
    /// `"llama"`).
    pub family: &'static str,
    /// Caller-facing preset id (e.g. `"7b-v2"` / `"llama-tiny"`).
    pub variant: String,
    /// Number of transformer blocks.
    pub layers: usize,
    /// Number of query heads.
    pub heads: usize,
    /// Number of key/value heads. Equal to `heads` for multi-head
    /// attention; smaller for grouped-query / multi-query.
    pub kv_heads: usize,
    /// Hidden dimension.
    pub dim: usize,
    /// Maximum sequence length.
    pub ctx: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Device the parameters live on.
    pub device: Device,
    /// Weight precision.
    pub dtype: DType,
    /// Shape of the tensor [`InferenceAdapter::forward`] returns.
    pub logits: LogitsShape,
}

/// An inference-only model handle the engine bridge can drive without
/// knowing the architecture.
///
/// See the module-level "Adapter contract" section for why construction
/// is deliberately excluded from this trait.
///
/// # Invariants (mirrored per adapter impl)
///
/// - [`Self::meta`] is stable for the lifetime of the adapter: the
///   returned values describe the weights loaded at construction time
///   and never change as a side effect of `forward`.
/// - [`Self::forward`] returns a tensor whose dims equal
///   `meta().logits.dims(batch, seq, meta().vocab)` for a `[batch,
///   seq]` input.
/// - `index_pos` is the position of the first token of `tokens` within
///   the ongoing generation. An adapter that maintains a KV cache
///   advances it by `seq` per call; one that does not may ignore the
///   argument entirely, but must still accept it.
pub trait InferenceAdapter: Send + Sync {
    /// Describe the loaded stack. See [`AdapterMeta`].
    fn meta(&self) -> AdapterMeta;

    /// Forward `tokens` (shape `[batch, seq]`) and return logits shaped
    /// per [`AdapterMeta::logits`].
    fn forward(&self, tokens: &Tensor, index_pos: usize) -> CandleResult<Tensor>;
}
