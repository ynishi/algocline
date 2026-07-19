# algocline-nn 0.45.0

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
- [`arch::gpt2`](arch__gpt2.md): GPT-2 architecture builder.
- [`arch::lora`](arch__lora.md): Low-rank adaptation ("LoRA") wrap for a candle-nn `Linear`.
- [`card`](card.md): Card metadata schema for `alc.nn.card.*`.
- [`tokenizer`](tokenizer.md): HuggingFace `tokenizers` wrap with first-use download cache.
- [`train`](train.md): Training-side scaffolding.
- [`train::ckpt`](train__ckpt.md): Rotating safetensors checkpoint writer.
- [`train::data`](train__data.md): Dataset iterator abstraction.
- [`train::loss`](train__loss.md): Loss functions used by the training loop.
- [`train::scheduler`](train__scheduler.md): Learning-rate schedules for the training loop.
- [`train::fullft`](train__fullft.md): Full FT training loop.

