# algocline-nn::arch::adapter

Inference-only adapters over `candle-transformers` model
implementations.

The Card foundation trained architectures (currently GPT-2) live in
sibling `arch::*` modules and expose a `VarMap` so
`algocline_nn::train::*` can drive optimizer steps against them.
This module holds the **inference-only** counterpart: thin wrappers
around `candle_transformers::models::*` that answer at the same layer
boundary as the trainable handles but decline to expose a `VarMap`
(candle-transformers stacks are loaded from a
`VarBuilder::from_mmaped_safetensors` reader, not a `VarMap`).

Callers reach these adapters through the engine bridge presets
(`alc.nn.preset.llama` / `alc.nn.preset.qwen` / ...), which build a
`<Arch>Handle` `UserData` and expose the same registry / `alc.llm`
`role="nn"` fast path that GPT-2 handles use. Because the adapters
do not own a `VarMap`, the trainer bindings (`alc.nn.trainer.*`)
surface a clear Lua-side error rather than attempting a training
loop.

# Adapter contract

Every adapter here implements [`InferenceAdapter`], which fixes the
two members the engine bridge dispatches on:

- [`InferenceAdapter::meta`] — an [`AdapterMeta`] describing the
  loaded stack (family / variant / shape parameters / device /
  dtype / logits shape). The bridge builds its Lua-facing handle
  from this value instead of re-reading the upstream config, so
  adding an arch does not mean copying nine accessors into a new
  handle struct.
- [`InferenceAdapter::forward`] — `(tokens, index_pos) -> logits`,
  with the caller-visible output shape declared by
  [`AdapterMeta::logits`] rather than left to per-arch prose.

Construction stays **outside** the trait. Each adapter keeps its own
`Config` struct, a `load(vb, config)` constructor, and a
`from_safetensors_files(paths, config)` constructor, because the
`VarBuilder` origin differs per arch (single file / sharded mmap /
GGUF) and a `Self: Sized` constructor in the trait would rule out
holding adapters as `dyn InferenceAdapter`. The engine bridge's
`ARCH_OPS` table owns that per-arch construction dispatch; this
trait owns behaviour only.

# Future additions

`arch::adapter::qwen`, `arch::adapter::phi`, and
`arch::adapter::gemma` will land as sibling modules on the same
shape once a caller needs each. Each is an `InferenceAdapter` impl
plus one `ARCH_OPS` entry on the bridge side. HuggingFace-hub
warm-start (`from_pretrained`) is deliberately deferred out of the
initial module cut so a downstream caller can drive the adapter from
local files (or an already-cached HF snapshot) without pulling
network into the unit-test path.

## Types

- `AdapterMeta` — Architecture-neutral description of a loaded model.
- `LogitsShape` — Caller-visible shape of the logits an adapter's `forward` returns.

## Traits

- `InferenceAdapter` — An inference-only model handle the engine bridge can drive without

