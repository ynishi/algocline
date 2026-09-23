# algocline-nn::sampling::penalty

Penalties that read what a generation has already produced.

Every sampler in [`super`] reads one logits row and nothing else, so
none of them can tell a token the model has emitted six times from
one it has never emitted. Left alone, a model that starts repeating
keeps repeating — the state that would break the loop is exactly the
history the sampler cannot see.

Three penalties are in general use and they are not variants of one
another:

- **Repetition** ([`Penalties::repetition`]) scales the logit of any
  token already seen, by a factor rather than a subtraction. From
  CTRL ([Keskar et al. 2019](https://arxiv.org/abs/1909.05858) §4.3,
  which reports `1.2` as a working value), and it divides positive
  logits while multiplying negative ones — the same rule on both
  sides would *raise* a negative logit and reward the repeat it was
  meant to discourage.
- **Frequency** ([`Penalties::frequency`]) subtracts in proportion to
  how many times the token appeared, so pressure accumulates with
  each repeat.
- **Presence** ([`Penalties::presence`]) subtracts a flat amount from
  every token that appeared at all, which pushes towards new
  vocabulary rather than away from repetition as such.

The last two are OpenAI's, additive and independent of scale; the
first is multiplicative and interacts with temperature. They compose,
and the order is fixed here: repetition scales, then the two
subtractions land on the result.

# Where the history comes from

[`PenalizedSampler`] accumulates the tokens it returns, the same way
[`ConstrainedSampler`](super::ConstrainedSampler) accumulates its
prefix. A prompt is not part of that by default — see
[`PenalizedSampler::with_history`] — because whether the prompt
counts is a real choice and not one this crate should make silently:
penalising it discourages a summary from reusing the words it was
given, which is sometimes exactly wrong and sometimes exactly right.

## Functions

- `apply_penalties` — Apply the penalties to a `[vocab]` logits row against `history`.

## Types

- `PenalizedSampler` — A sampler that pushes down what the generation has already produced.
- `Penalties` — How hard an already-seen token is pushed down.

