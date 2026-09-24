# algocline-nn::pooling

One vector for a sequence.

A model's hidden state is `[batch, seq, dim]` — one vector per
position — and an embedding is one vector per sequence. Pooling is
the step between, and which pooling is a real choice rather than a
detail: the three below disagree about what a sequence's meaning
sits in, and the right answer depends on how the model was trained.

- [`Pooling::Mean`] averages every position. The usual default for a
  sequence embedding, and what Sentence-BERT's own guidance
  recommends for models with no pooling head of their own.
- [`Pooling::Last`] takes the final position. What a decoder-only
  model's own objective builds: that position is the only one that
  has read the whole sequence, which is why it is the one the
  language-model head is asked from.
- [`Pooling::Max`] takes the elementwise maximum, which keeps the
  strongest activation of each feature rather than its average.

# Padding

`lengths` says how many positions of each row are real. Without it
the padded tail is pooled along with the content: a mean over a
half-padded row is half an embedding of the padding, and a "last"
pool lands on filler. It is optional because a batch of one — the
common case for an embedding call — has nothing to pad.

## Functions

- `pool` — Pool a `[batch, seq, dim]` hidden state into `[batch, dim]`.

## Types

- `Pooling` — How a sequence's positions become one vector.

