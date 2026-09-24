# algocline-nn::sampling::beam

Beam search.

Every sampler in [`super`] commits to one token and never
reconsiders: the sequence it produces is the product of a chain of
local choices, and a high-probability continuation reachable only
through a mediocre first token is unreachable. Beam search keeps `k`
partial sequences alive, extends all of them, and keeps the `k` best
of the result — so a token that looked second-best can still lead.

It is not a better sampler; it is a different objective. Sampling
draws from the model's distribution, and beam search approximates
the *most likely sequence* under it. That is what you want for a
translation, a constrained field, a short structured answer — and
not what you want for open text, where the most likely sequence is
famously bland and repetitive.

# Scoring

A beam's score is the sum of its tokens' log-probabilities. Summing
logs rather than multiplying probabilities is not a convenience: the
product of a few hundred probabilities underflows f32 long before a
generation ends, and every beam would score zero.

Longer sequences score lower for being longer, since every
additional term is negative. [`BeamOptions::length_penalty`] divides
by `length^α`, the normalisation from
[Wu et al. 2016](https://arxiv.org/abs/1609.08144) §7 — `0.0` leaves
raw sums and prefers short answers, `1.0` is the mean log-probability
per token, and the values in use sit near `0.6`–`1.0`.

# What this does not do

Sampling and beam search do not compose: the second explores, and a
stochastic step would make the exploration unrepeatable and the
"best" beam a draw. This search is greedy over the top-`k`
continuations and takes no RNG.

## Functions

- `beam_search` — Search for the most likely continuations of `prompt`.

## Types

- `Beam` — One finished or surviving sequence.
- `BeamOptions` — How the search runs.

## Traits

- `BeamModel` — A model a beam search can advance.

