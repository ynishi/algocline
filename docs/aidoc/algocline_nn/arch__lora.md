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

## Functions

- `max_abs_diff_f32` — Snapshot two tensors as flat f32 vectors and return the maximum

## Types

- `LoraConfig` — LoRA rank + scaling + wrap-target configuration.
- `LoraLinear` — A `Linear` layer wrapped with a low-rank additive update.

