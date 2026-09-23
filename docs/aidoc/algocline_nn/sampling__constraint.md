# algocline-nn::sampling::constraint

Layer 2 of the sampler plan: constraints that mask logits before a
Layer 1 [`Sampler`] picks a token.

A [`Constraint`] answers two questions per generation step, both from
the generated-token prefix alone: *which tokens may come next*
([`Constraint::mask`]) and *should generation stop*
([`Constraint::is_terminal`]). [`ConstrainedSampler`] wires a
constraint to an arbitrary inner sampler and is itself a [`Sampler`],
so constrained decoding composes with every Layer 1 impl and with
future Layer 3 schedules without either side knowing about the other.

# Sparse masks

[`TokenMask`] is deliberately sparse. A dense `Vec<bool>` of vocab
length (32k–256k entries) would be rebuilt on every single token even
when the constraint has nothing to say. [`TokenMask::AllowAll`] is the
common case for prefix-agnostic constraints (stop tokens, most of a
grammar's interior) and costs *nothing*: the logits tensor is handed
to the inner sampler untouched, with no device round-trip.

# Failure is loud

A mask that leaves zero candidate tokens is a caller programming
error, not a situation to paper over. Softmax over an all-`-inf` row
yields NaNs, and silently falling back to argmax on the *unmasked*
logits would emit a token the constraint explicitly forbade — the one
outcome constrained decoding exists to prevent. Both cases return
`Err` instead.

# Constraints that have landed

[`StopTokensConstraint`] is the termination-only case: it masks
nothing and only answers [`Constraint::is_terminal`].
[`AllowListConstraint`] is the masking-only case: a fixed legal set,
the same at every position. [`RegexConstraint`] is the first
structural one — it drives an
anchored DFA over the tokenizer's surface strings so every sampled
token keeps the output on a path towards a full pattern match. JSON
schema and GBNF grammars are future additions behind the same trait.

## Types

- `AllowListConstraint` — Restrict every position to a fixed set of legal token ids.
- `ConstrainedSampler` — A [`Sampler`] that masks logits through a [`Constraint`] before
- `RegexConstraint` — Restrict generation to token sequences that spell a full match of a
- `StopTokensConstraint` — Terminate generation when one of a fixed set of token ids is emitted.
- `TokenBitset` — A token mask as one bit per token id.
- `TokenMask` — Sparse per-step token mask produced by a [`Constraint`].

## Traits

- `Constraint` — Per-step restriction on the next token.

