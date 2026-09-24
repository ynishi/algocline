# algocline-nn::arch::kv

Keys and values a session has already computed.

A decode loop over a trainable architecture used to re-forward its
whole history at every step: generating `n` tokens ran the model
`1 + 2 + … + n` positions instead of `n`, because each step recomputed
every earlier position's keys and values from weights that had not
changed. The arithmetic is quadratic in the length and the work is
redundant — attention at position `p` reads the same `K`/`V` at every
earlier position it read the step before.

A cache holds those tensors per layer and grows by the positions each
step adds. The model then forwards only the new tokens and attends
over the whole history, which is the same computation the
full-sequence pass performs at the last row and the standard decode
arrangement everywhere it appears.

# What is cached, and what is not

The entries are the **post-rotation, pre-broadcast** `K` and `V`:
`[batch, kv_heads, position, head_dim]`. Post-rotation because RoPE
is a function of the absolute position, which does not change as the
sequence grows — rotating once is correct and rotating again would
not be. Pre-broadcast because grouped-query attention repeats each
KV head across its query group, and storing the repeats would hold
`heads / kv_heads` copies of every tensor for no gain.

Queries are not cached: a query belongs to the position being
answered and is never read again.

# A cache belongs to one sequence

Entries are positions of one particular sequence, so a cache carries
the batch it was filled at and refuses a forward at another
([`KvError::BatchChanged`]). Two generations sharing a cache would
each attend over the other's history with every shape still
agreeing, which is the failure this refusal exists for — and the
same reason [`crate::arch::adapter::LlamaAdapter`] hands out a cache
per session rather than holding one.

## Types

- `KvCache` — Per-layer keys and values for the positions already forwarded.
- `KvError` — A cache used in a way it cannot answer for.

