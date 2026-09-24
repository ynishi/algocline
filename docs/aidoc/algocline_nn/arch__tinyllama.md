# algocline-nn::arch::tinyllama

TinyLlama-1.1B trainable architecture.

Layer 1a — primitives (backward-safe shims + RoPE cache + GQA
`repeat_kv`):

- [`apply_slow_rms_norm`] — RMSNorm forward via
  [`candle_nn::ops::rms_norm_slow`] to keep the autograd chain
  intact (candle-nn 0.11 `RmsNorm::forward` is a `CustomOp3` with no
  backward, see `tests/rms_norm_autograd_gate.rs`).
- [`apply_rope`] — Rotary embedding forward via
  [`candle_nn::rotary_emb::rope_slow`] for the same reason
  (`rotary_emb::rope` uses `apply_op3_no_bwd`).
- [`build_rope_cache`] — precomputed cos / sin cache using the
  canonical Llama-family frequency formula
  `theta_i = base^{-2i/head_dim}` for `i ∈ [0, head_dim/2)`.
- [`repeat_kv`] — grouped-query-attention KV expansion:
  `[B, H_kv, S, head_dim] → [B, H, S, head_dim]` via `expand`.

Layer 1b — model:

- [`TinyLlamaConfig`] with presets `tinyllama-1.1b` and
  `tinyllama-tiny` (CPU-friendly smoke shape).
- [`TinyLlamaModel`] — pre-RMSNorm decoder stack with GQA + RoPE +
  SwiGLU MLP, mirroring the HuggingFace `LlamaModel` weight layout
  so a downloaded `TinyLlama-1.1B-*` safetensors bundle loads
  through [`candle_nn::VarBuilder::from_mmaped_safetensors`] without
  any renaming.

HF weight naming (matches
`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`):

```text
model.embed_tokens.weight                     [vocab, dim]
model.layers.<i>.input_layernorm.weight       [dim]
model.layers.<i>.self_attn.q_proj.weight      [heads   *head_dim, dim]
model.layers.<i>.self_attn.k_proj.weight      [kv_heads*head_dim, dim]
model.layers.<i>.self_attn.v_proj.weight      [kv_heads*head_dim, dim]
model.layers.<i>.self_attn.o_proj.weight      [dim, heads*head_dim]
model.layers.<i>.post_attention_layernorm.weight  [dim]
model.layers.<i>.mlp.gate_proj.weight         [hidden, dim]
model.layers.<i>.mlp.up_proj.weight           [hidden, dim]
model.layers.<i>.mlp.down_proj.weight         [dim, hidden]
model.norm.weight                             [dim]
lm_head.weight                                [vocab, dim]
```

Forward output shape is `[batch, seq, vocab]`. Attention uses a
causal (lower-triangular) mask cached to `ctx`.

## Functions

- `apply_rope` — Rotary Position Embedding forward using the backward-safe
- `apply_slow_rms_norm` — RMSNorm forward that always uses the backward-safe basic-op path
- `build_rope_cache` — Build a canonical Llama-family RoPE cos / sin cache.
- `repeat_kv` — Grouped-query-attention KV expansion.

## Types

- `PretrainedError` — Errors from [`TinyLlamaModel::from_pretrained`].
- `TinyLlamaConfig` — Immutable configuration for a TinyLlama preset.
- `TinyLlamaModel` — TinyLlama forward-only model.

