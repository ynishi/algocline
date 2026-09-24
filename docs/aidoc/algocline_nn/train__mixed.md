# algocline-nn::train::mixed

Mixed-precision AdamW (design §7.1: BF16 weights / activations +
FP32 optimizer states).

candle-nn's stock [`candle_nn::AdamW`] keeps its first/second
moments in the *parameter's* dtype, so handing it BF16 `Var`s
silently runs the whole optimizer state in BF16 — the classic
mixed-precision failure mode where small updates round to zero at
the parameter's magnitude and training stalls without erroring.
[`MixedAdamW`] implements the standard master-weights recipe
instead:

- the model's `Var`s stay in their low precision (BF16) and keep
  driving forward / backward,
- a private FP32 master copy of every parameter plus FP32 moment
  tensors receive the AdamW update (gradients are upcast per step),
- the updated master is cast back down and written into the `Var`
  so the next forward sees the new weights.

The update math mirrors `candle_nn::AdamW::step` term for term
(decoupled weight decay applied as `θ · (1 − lr·λ)`, bias-corrected
moments) so the FP32-on-FP32 special case is numerically identical
to the stock optimizer — pinned by the parity test below.

F16 is deliberately not accepted: its 5-bit exponent needs loss
scaling to keep gradients from flushing to zero, and no scaler
ships here. BF16 shares FP32's 8-bit exponent, which is why it
trains without a scaler.

## Types

- `MixedAdamW` — AdamW with FP32 master weights over low-precision parameters.

