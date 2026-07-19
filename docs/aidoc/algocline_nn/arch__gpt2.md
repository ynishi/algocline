# algocline-nn::arch::gpt2

GPT-2 architecture builder.

Implements the two variants shipped in Phase 1 (design §6.1):

- `gpt2-medium` — 24 layers, 16 heads, 1024 dim, 1024 ctx, 50257 vocab
- `gpt2-large`  — 36 layers, 20 heads, 1280 dim, 1024 ctx, 50257 vocab

Architecture components (nanoGPT / HuggingFace `openai-community/gpt2`
reference layout):

- `wte` — token embedding (`vocab × dim`)
- `wpe` — learned positional embedding (`ctx × dim`)
- `h.<i>.ln_1` / `ln_2` — pre-LayerNorm (dim)
- `h.<i>.attn.c_attn` — fused Q/K/V projection (`dim → 3·dim`)
- `h.<i>.attn.c_proj` — attention output projection (`dim → dim`)
- `h.<i>.mlp.c_fc` / `mlp.c_proj` — 4× expansion MLP with GELU
- `ln_f` — final LayerNorm (dim)
- LM head weights are tied to `wte` (shared matrix)

Forward output shape is `[batch, seq, vocab]` per subtask invariant
#1. Attention uses a causal (lower-triangular) mask.

## Types

- `Gpt2Config` — Immutable configuration for a GPT-2 preset.
- `Gpt2Model` — GPT-2 forward-only model.
- `PretrainedError` — Errors from [`Gpt2Model::from_pretrained`].

