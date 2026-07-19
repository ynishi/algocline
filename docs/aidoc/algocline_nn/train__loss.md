# algocline-nn::train::loss

Loss functions used by the training loop.

Introduces a small [`Loss`] trait so the loop stays agnostic to the
specific reduction and can accept plain cross-entropy (Full FT), a
hard-label distillation variant, or a future KL-soft variant without
changing the loop signature.

The optional `loss_mask` on `Loss::compute` is what lets a
distillation caller zero out prompt-region tokens: for Full FT the
caller passes `None`; distillation passes a per-token 0/1 mask so
only the response region contributes to the loss.

## Types

- `CrossEntropyLoss` — Standard token-level cross-entropy against a hard target.
- `HardLabelDistillLoss` — Hard-label distillation loss.
- `Reduction` — Reduction strategy applied inside a `Loss::compute` call.

## Traits

- `Loss` — Loss function on `[batch, seq, vocab]` logits.

