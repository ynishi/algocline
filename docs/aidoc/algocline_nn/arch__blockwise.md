# algocline-nn::arch::blockwise

Driving a model one block at a time.

The ordinary forward hands back logits and keeps, inside the
autograd graph, every intermediate that produced them — for a
transformer that is on the order of ten tensors per block, all of
them `[batch, seq, ·]`, held until the backward pass reads them.
Context length and batch size are bounded by that, not by the
parameters.

Gradient checkpointing trades it back: keep only each block's
**input**, and recompute the block's insides during the backward
pass, one block at a time. Memory falls from every intermediate of
every block to one tensor per block plus the intermediates of
whichever block is currently being recomputed; the cost is a second
forward pass, so roughly a third more compute for a fraction of the
activation memory. The technique is Chen et al. 2016,
[*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).

This trait is what lets the training loop do that: a model that
implements it can be run embed → block → block → … → head under the
loop's control instead of in one call.

# Why the loop and not the model

Recomputation has to be interleaved with the backward pass, and the
backward pass belongs to the training loop. A model that tried to
own both would have to own the loss as well.

# What cannot be checkpointed here

[`Checkpointable::checkpointable`] answers for each model. The two
refusals today are both about a forward that carries more than its
output:

- **A mixture-of-experts block** returns a load-balancing term
  alongside its activations, and that term is part of the loss. A
  blockwise driver that dropped it would train without the balancing
  objective while still reporting a loss — and the routing would
  collapse onto a few experts with nothing saying so.
- **An input channel** (a conditioning table, an allowed-id set) is
  read at the embedding and again at every position; the blockwise
  surface here takes ids alone. Refused rather than run with the
  channel dropped, which is the same silent failure the forward path
  already refuses.

## Traits

- `Checkpointable` — A model the training loop can drive one block at a time.

