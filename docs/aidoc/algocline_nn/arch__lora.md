# algocline-nn::arch::lora

Low-rank adaptation ("LoRA") wrap for a candle-nn `Linear`.

Wraps a frozen base linear layer with two thin trainable matrices
(`lora_a` shaped `[rank, in_features]`, `lora_b` shaped
`[out_features, rank]`) and a scaling factor of `alpha / rank`.
The forward pass computes

```text
y = base.forward(x) + scaling * (lora_b(lora_a(x)))
```

The base parameters are held as a `Linear` value (so
`.weight()` / `.bias()` are still accessible for merge equivalence
checks) but the caller is expected to keep them out of the
optimizer's parameter list — only `lora_a` and `lora_b` should be
trainable during LoRA fine-tuning.

[`LoraLinear::merged_weight`] materialises the equivalent
`base.weight() + scaling * (lora_b.weight() @ lora_a.weight())`
matrix so a caller can construct a plain `Linear` that produces
identical outputs for the same input. This is what the merge-
equivalence integration test asserts within 1e-4 element-wise.

# Initialisation

[`LoraLinear::wrap`] follows the canonical LoRA init from
Hu et al. 2021 §4.1: `lora_a` gets candle-nn's default (Kaiming
uniform) random weights, while `lora_b` is initialised to **zero**.
Because `ΔW = scaling * (B · A)` and `B = 0` at `t=0`, the wrap is
effectively an identity map at construction — `wrap(base).forward(x)`
equals `base.forward(x)` bit-for-bit. Only training moves `B` off
zero, at which point `ΔW` starts contributing.

## Functions

- `max_abs_diff_f32` — Snapshot two tensors as flat f32 vectors and return the maximum

## Types

- `LoraConfig` — LoRA rank + scaling + wrap-target configuration.
- `LoraLinear` — A `Linear` layer wrapped with a low-rank additive update.

## Traits

- `LoraWrappable` — A trainable model that can be wrapped with LoRA
- `MergeableLora` — A model that can export a merged inference-ready weight bundle.

