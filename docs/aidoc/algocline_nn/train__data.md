# algocline-nn::train::data

Dataset iterator abstraction.

Design §8.4: `alc.nn.data.jsonl`, `alc.nn.data.parquet`, and
`alc.nn.data.from_card` all return an opaque handle that the
trainer follow-up entries consume batch-by-batch. Iteration is Rust-side
(no per-batch Lua callback) so the trainer stays on the hot path
without VM cross-calls.

Subtask invariants:

- Batches produce `[batch_size, ctx_len]` `u32` token ids. The last
  batch may be short (`Batch::is_last == true`).
- JSONL adapter reads each line as `{ "text": "..." }` (default
  field name; overridable via [`DatasetOpts::text_field`]) and
  tokenizes with the caller-supplied [`crate::tokenizer::HfTokenizer`].
- `from_card` reads Card sample rows via `FileCardStore` (invariant
  #5); the `algocline-nn` crate does no filesystem access itself.
  The bridge builds a [`TokenizedDataset`] from the pre-tokenized
  rows and hands it back to the trainer.
- Parquet reading is scaffolded (see [`ParquetDataset`]); a later
  stage picks up the concrete reader implementation. Constructing the
  handle is legal, `next_batch` returns an error so no silent
  empty iterator is exposed (Rust exception-free discipline).

## Types

- `Batch` — One training batch.
- `DatasetError` — Errors surfaced by dataset iterators.
- `DatasetOpts` — Iterator config shared across dataset kinds.
- `JsonlDataset` — JSONL-backed dataset.
- `ParquetDataset` — Parquet-backed dataset (scaffold).
- `TeacherCardDataset` — In-memory dataset for hard-label distillation.
- `TokenizedDataset` — In-memory dataset built from a `Vec<Vec<u32>>` of pre-tokenized

## Traits

- `Dataset` — Streaming batch iterator.

