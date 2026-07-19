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
//! Every adapter here follows the same shape so the engine bridge can
//! stay uniform:
//!
//! - A concrete `Config` struct (or re-export of an upstream config)
//!   describing the family / variant / device / dtype.
//! - A `load(vb, config, ...)` constructor that takes a
//!   `VarBuilder`, mirroring the trainable-side constructors.
//! - A `from_safetensors_files(paths, config, ...)` constructor that
//!   mmaps a slice of `.safetensors` files, builds a `VarBuilder`, and
//!   forwards to `load`.
//! - A `forward(tokens, index_pos)` method returning logits shaped for
//!   the caller (e.g. `[batch, vocab]` or `[batch, seq, vocab]` per
//!   the upstream model's convention).
//!
//! # Future additions
//!
//! `arch::adapter::qwen`, `arch::adapter::phi`, and
//! `arch::adapter::gemma` will land as sibling modules on the same
//! shape once a caller needs each. HuggingFace-hub warm-start
//! (`from_pretrained`) is deliberately deferred out of the initial
//! module cut so a downstream caller can drive the adapter from local
//! files (or an already-cached HF snapshot) without pulling network
//! into the unit-test path.

pub mod llama;

pub use llama::{LlamaAdapter, LlamaAdapterConfig};
