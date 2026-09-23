# algocline-nn::arch::adapter::llama

Llama-family inference adapter over `candle_transformers::models::llama`.

Wraps the upstream `Llama` stack (RMSNorm + RoPE + GQA + SwiGLU) as
an inference handle at the `alc.nn` layer boundary. The upstream
implementation is loaded from a `VarBuilder` — no `VarMap` — so
this adapter deliberately does not participate in the training loop
(`alc.nn.trainer.*` refuses handles without a `VarMap`).

The adapter owns the mutable KV [`Cache`] alongside the stack so a
caller who threads through the same handle across successive
`forward` calls (e.g. token-by-token generation) reuses the same
cache without exposing candle-transformers types to Lua.

# Built-in cache vs caller-owned cache

Two forward entries exist, and the difference is a concurrency
invariant rather than a convenience:

- [`LlamaAdapter::forward`] drives the adapter's **built-in** cache.
  Correct for exactly one generation loop at a time — two loops
  sharing the handle would interleave their keys and values into the
  same cache and silently produce cross-contaminated logits.
- [`LlamaAdapter::forward_with_cache`] drives a **caller-owned**
  cache obtained from [`LlamaAdapter::new_cache`]. Each generation
  session holds its own cache, so concurrent sessions over one
  `Arc<LlamaAdapter>` (weights are read-only and shared) cannot mix
  state at all — the mixing is prevented structurally rather than by
  a caller-side convention.

The built-in path is kept as the single-loop legacy entry so existing
callers keep working unchanged; new multi-session callers (the engine
bridge's Lua-facing generation session) use the caller-owned path.

## Types

- `LlamaAdapter` — Inference-only Llama handle.
- `LlamaAdapterConfig` — Build-time configuration for a [`LlamaAdapter`].

