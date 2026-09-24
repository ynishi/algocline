# algocline-nn 0.49.0

algocline-nn — thin candle wrapper for the `alc.nn` Lua surface.

# Architecture

This crate is the Host(Rust) side of the alc.nn layer boundary. The design
intent is:

- **Host owns the heavy state**: tensors, the autograd graph, parameters
  (`candle_nn::VarMap`), optimizer state, and the `GradStore` all live in
  Rust. Lua never holds a `Var` lifetime — only opaque handles.
- **Lua owns composition and loops**: model assembly, the training loop, lr
  schedule, and batching are written in Lua. Rust exposes only thin wraps of
  individual candle ops; it does not embed loop / schedule / batching logic.
- **core stays clean**: `algocline-core` never depends on candle or tensor
  types. This crate is an optional, feature-gated dependency of the engine
  (`nn` feature, default off) so the default MCP build stays light.

# L1 spike scope

Phase L1 is a spike: it validates the riskiest unknowns (candle link
interference, GradStore key access, and `mlua::UserData` tensor exposure)
and lands a minimal primitive. It is not the full op set.

In Step 1 this crate only links `candle-core` (CPU) to confirm there is no
link interference with the mlua-vendored workspace. Later steps add
`candle-nn` (VarMap / autograd / optimizer) and the `mlua` UserData surface.

## Modules

- [`arch`](arch.md): Architecture presets for `alc.nn.preset.*`.
- [`arch::adapter`](arch__adapter.md): Inference-only adapters over `candle-transformers` model
- [`arch::adapter::llama`](arch__adapter__llama.md): Llama-family inference adapter over `candle_transformers::models::llama`.
- [`arch::blockwise`](arch__blockwise.md): Driving a model one block at a time.
- [`arch::custom`](arch__custom.md): Spec-driven customization points for the GPT-2 stack (Phase 1).
- [`arch::gpt2`](arch__gpt2.md): GPT-2 architecture builder.
- [`arch::kv`](arch__kv.md): Keys and values a session has already computed.
- [`arch::lora`](arch__lora.md): Low-rank adaptation ("LoRA") wrap for a candle-nn `Linear`.
- [`arch::moe`](arch__moe.md): Dense Mixture-of-Experts feed-forward for the GPT-2 stack.
- [`arch::seeded`](arch__seeded.md): Parameter initialisation a seed decides.
- [`arch::tinyllama`](arch__tinyllama.md): TinyLlama-1.1B trainable architecture.
- [`card`](card.md): Card metadata schema for `alc.nn.card.*`.
- [`export`](export.md): Exporting a Card in the vocabulary other tools read.
- [`export::mapping`](export__mapping.md): A Card's fields in the vocabulary other tools read.
- [`gguf`](gguf.md): Writing a model out as GGUF.
- [`merged`](merged.md): Merged inference checkpoint export (Layer 4a of GH #10).
- [`pooling`](pooling.md): One vector for a sequence.
- [`sampling`](sampling.md): Next-token samplers for the inference path.
- [`sampling::beam`](sampling__beam.md): Beam search.
- [`sampling::constraint`](sampling__constraint.md): Layer 2 of the sampler plan: constraints that mask logits before a
- [`sampling::json_schema`](sampling__json_schema.md): JSON-schema-constrained decoding: a [`Constraint`] that admits only
- [`sampling::penalty`](sampling__penalty.md): Penalties that read what a generation has already produced.
- [`tokenizer`](tokenizer.md): HuggingFace `tokenizers` wrap with first-use download cache.
- [`train`](train.md): Training-side scaffolding.
- [`train::checkpointing`](train__checkpointing.md): The forward and backward of a gradient-checkpointed step.
- [`train::ckpt`](train__ckpt.md): Rotating safetensors checkpoint writer, and the restore side that
- [`train::corpus`](train__corpus.md): Corpus files: pre-tokenized training rows on disk.
- [`train::data`](train__data.md): Dataset iterator abstraction.
- [`train::lion`](train__lion.md): Lion — the sign-momentum optimizer from *Symbolic Discovery of
- [`train::loss`](train__loss.md): Loss functions used by the training loop.
- [`train::mixed`](train__mixed.md): Mixed-precision AdamW (design §7.1: BF16 weights / activations +
- [`train::optstate`](train__optstate.md): Optimizer state that outlives the run that produced it.
- [`train::scheduler`](train__scheduler.md): Learning-rate schedules for the training loop.
- [`train::fullft`](train__fullft.md): Full FT training loop.

