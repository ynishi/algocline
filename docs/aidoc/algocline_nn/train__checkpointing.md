# algocline-nn::train::checkpointing

The forward and backward of a gradient-checkpointed step.

See [`crate::arch::blockwise`] for what checkpointing is and which
models can be driven this way. This module is the other half: the
two-phase pass the training loop runs in place of
`forward → loss → backward`.

# The two phases

**Forward.** Embed, then walk the blocks, detaching each block's
output before it becomes the next block's input. The detach is what
frees the memory: a detached tensor carries no operation, so the
graph that produced it — every intermediate inside that block —
becomes unreachable and is dropped. What is kept is one
`[batch, seq, dim]` tensor per block, the inputs, which is the
checkpoint. The head and the loss run on the last one, tracked
normally.

**Backward.** Run the loss's own backward, which reaches the head's
parameters and stops at the last checkpoint — leaving, in the grad
store, the gradient with respect to that checkpoint. Then walk the
blocks backwards: re-run one block on its stored input (rebuilding
exactly the intermediates the forward threw away), and push the
incoming gradient through it.

# Pushing a gradient into a sub-graph

candle's `Tensor::backward` seeds the graph with ones, so it answers
`∂(Σ y)/∂θ` and there is no argument for "start from this gradient
instead". The surrogate `Σ(y ⊙ g)` supplies it:

```text
∂/∂θ Σᵢ yᵢ·gᵢ  =  Σᵢ gᵢ · ∂yᵢ/∂θ  =  Jᵀg
```

which is exactly the chained gradient, for `g` treated as a
constant — and it is, having come out of a grad store already
detached. The same backward also writes `∂/∂x Σ(y ⊙ g)` for the
block's input `x`, which is the gradient the next block down
receives.

# Why the checkpoint is re-entered as a `Var`

Because a plain detached tensor does not work, and the difference is
silent. candle's graph walk marks a node as tracking gradients only
if a `Var` is reachable below it, and skips the rest entirely — so
every operation computed from a non-`Var` leaf alone is never
visited by the backward pass. A gradient does land on such a leaf
when its immediate consumer is tracked, which makes the arrangement
look workable on a one-operation example; it stops there. Measured:
`layer_norm_slow` over a non-`Var` input leaves that input with no
gradient at all, because the mean and variance it computes first
depend on nothing else.

Each checkpoint is therefore re-entered as a temporary
[`Var`](candle_core::Var) during the backward phase. A `Var` is a
tracked leaf, so the block's recomputation tracks gradients
throughout and its own gradient — the one the next block down
receives — is written where it can be read. The `Var` lives for one
block's recomputation and is dropped with it; it is never registered
in any `VarMap` and no optimizer ever sees it.

Both halves of that are asserted in this module's tests rather than
inferred, because a checkpointed run whose gradients quietly stop
partway produces a loss curve that looks like a slow one.

## Functions

- `checkpointed_step` — Run one checkpointed step and return `(loss value, gradients)`.
- `grad_coverage` — How many of `vm`'s variables `grads` holds a gradient for.
- `max_grad_gap` — Sum of `|a - b|`'s maximum over every variable both stores hold, for

