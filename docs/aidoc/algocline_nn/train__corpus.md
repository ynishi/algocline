# algocline-nn::train::corpus

Corpus files: pre-tokenized training rows on disk.

A producer that has already tokenized its material — a simulator, an
export from another tool, an earlier stage of the same pipeline —
has no text left for a tokenizer to read, so the JSONL and Parquet
adapters in [`crate::train::data`] do not apply to it. This module
is the format such a producer writes and the loader that reads it
back.

# Format

This documentation is the format's specification. One file is one
JSON object:

```json
{
  "meta": { "ctx_len": 48, "vocab_size": 46 },
  "rows": [[3, 7, 1], [2, 9]]
}
```

- **`meta.ctx_len`** — the sequence length the rows were written
  for. Both dimensions are non-zero, and no row is longer than it
  ([`CorpusError::RowTooLong`]).
- **`meta.vocab_size`** — the id space the rows are drawn from.
  Every id in `rows` is checked against it at load
  ([`CorpusError::TokenOutOfRange`]); an id at or past it would
  otherwise surface several layers away as an out-of-range embedding
  lookup.
- **`meta.requires`** — optional array of `meta` field names a
  reader has to understand for the file to mean what its producer
  intended. An entry this version does not implement is refused by
  name ([`CorpusError::UnsupportedRequirement`]) rather than loaded
  without it. This version understands `ctx_len`, `vocab_size` and
  `per_row_allowed`.
- **`meta.<anything else>`** — ignored unless `meta.requires` lists
  it. A producer records more than a trainer reads, and a file
  carrying a bookkeeping field this version has never heard of still
  loads.
- **`rows`** — token id sequences, at least one, none of them empty,
  none longer than `meta.ctx_len`. Rows are *not* padded here:
  padding to `ctx_len` is [`crate::train::TokenizedDataset`]'s
  business, per batch.
- **`allowed`** — optional, and opt-in through
  `meta.requires: ["per_row_allowed"]`: which ids each position of
  each row was allowed to take. See *Allowed-id sets* below.

# Allowed-id sets

A corpus may state what was available at each position, which a run
reads either as a mask on the loss or as an input to the model. The
field is opt-in and announces itself:

```json
{
  "meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["per_row_allowed"] },
  "rows": [[3, 7, 1, 5]],
  "allowed": [{ "2": [7, 8], "4": [5] }]
}
```

`allowed` is parallel to `rows`, one entry per row, and each entry is
sparse: it maps a **1-based** position to the ids available there. A
position nobody listed is unconstrained, which is what an empty set
means downstream too — so the padding past the end of a row is left
out rather than spelled.

The requirement and the field are checked against each other in both
directions ([`CorpusError::AllowedMissing`] /
[`CorpusError::AllowedUnannounced`]). Either way round the run would
otherwise train on rows that do not mean what the producer wrote and
report the numbers of a well-formed one: `allowed` without the
requirement lets a reader that does not implement the field train the
same rows unconstrained, and the requirement without `allowed` is a
producer that meant to write sets and did not.

Rows are left at their own listed width here. Widening them to the
common width the model's allowed-id input takes belongs with the
merge, because a width can only be settled once every source has been
read.

# Extending the format

`meta.requires` is the mechanism, and it is the reason the format
needs no version number. A producer adding a field that only records
how the file was made writes it into `meta` and older readers ignore
it. A producer adding a field the file's *meaning* depends on — a
per-row constraint, say — lists that field in `meta.requires`, so a
reader that does not implement it refuses the file instead of
training on a reading its producer did not intend. Unknown-and-
ignored is therefore a choice the writer makes per field rather than
a property of every field a reader has not heard of.

# What this module does not do

The loader's responsibility ends at "file → validated rows".
Batching, padding, the per-row side channels and any repetition of
the row list belong to [`crate::train::TokenizedDataset`], which
already implements them; duplicating any of it here would give a
corpus-backed run different semantics from every other dataset for
no reason a caller could see.

# Combining several files

[`interleave`] merges sources round-robin rather than concatenating
them. Concatenation makes the source a function of how far the run
has got, which a run binding something per source cannot separate
from the thing it is supposed to be binding. Sources of unequal
length drop out as they are exhausted, so the tail is whatever the
largest ones have left; nothing is duplicated to even the rotation
out, because that would change the mixture the caller named.

## Functions

- `interleave` — [`interleave_labelled`] for a caller that attaches nothing per
- `interleave_labelled` — Merge several corpora round-robin, keeping each row's source.

## Types

- `CorpusError` — Errors surfaced while reading a corpus file or combining several.
- `CorpusFile` — One loaded corpus file: the format's two `meta` dimensions plus the
- `InterleavedRow` — One row of an interleaved corpus, together with the source it came

