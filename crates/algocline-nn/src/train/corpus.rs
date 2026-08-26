//! Corpus files: pre-tokenized training rows on disk.
//!
//! A producer that has already tokenized its material — a simulator, an
//! export from another tool, an earlier stage of the same pipeline —
//! has no text left for a tokenizer to read, so the JSONL and Parquet
//! adapters in [`crate::train::data`] do not apply to it. This module
//! is the format such a producer writes and the loader that reads it
//! back.
//!
//! # Format
//!
//! This documentation is the format's specification. One file is one
//! JSON object:
//!
//! ```json
//! {
//!   "meta": { "ctx_len": 48, "vocab_size": 46 },
//!   "rows": [[3, 7, 1], [2, 9]]
//! }
//! ```
//!
//! - **`meta.ctx_len`** — the sequence length the rows were written
//!   for. Both dimensions are non-zero, and no row is longer than it
//!   ([`CorpusError::RowTooLong`]).
//! - **`meta.vocab_size`** — the id space the rows are drawn from.
//!   Every id in `rows` is checked against it at load
//!   ([`CorpusError::TokenOutOfRange`]); an id at or past it would
//!   otherwise surface several layers away as an out-of-range embedding
//!   lookup.
//! - **`meta.requires`** — optional array of `meta` field names a
//!   reader has to understand for the file to mean what its producer
//!   intended. An entry this version does not implement is refused by
//!   name ([`CorpusError::UnsupportedRequirement`]) rather than loaded
//!   without it. This version understands `ctx_len`, `vocab_size` and
//!   `per_row_allowed`.
//! - **`meta.<anything else>`** — ignored unless `meta.requires` lists
//!   it. A producer records more than a trainer reads, and a file
//!   carrying a bookkeeping field this version has never heard of still
//!   loads.
//! - **`rows`** — token id sequences, at least one, none of them empty,
//!   none longer than `meta.ctx_len`. Rows are *not* padded here:
//!   padding to `ctx_len` is [`crate::train::TokenizedDataset`]'s
//!   business, per batch.
//! - **`allowed`** — optional, and opt-in through
//!   `meta.requires: ["per_row_allowed"]`: which ids each position of
//!   each row was allowed to take. See *Allowed-id sets* below.
//!
//! # Allowed-id sets
//!
//! A corpus may state what was available at each position, which a run
//! reads either as a mask on the loss or as an input to the model. The
//! field is opt-in and announces itself:
//!
//! ```json
//! {
//!   "meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["per_row_allowed"] },
//!   "rows": [[3, 7, 1, 5]],
//!   "allowed": [{ "2": [7, 8], "4": [5] }]
//! }
//! ```
//!
//! `allowed` is parallel to `rows`, one entry per row, and each entry is
//! sparse: it maps a **1-based** position to the ids available there. A
//! position nobody listed is unconstrained, which is what an empty set
//! means downstream too — so the padding past the end of a row is left
//! out rather than spelled.
//!
//! The requirement and the field are checked against each other in both
//! directions ([`CorpusError::AllowedMissing`] /
//! [`CorpusError::AllowedUnannounced`]). Either way round the run would
//! otherwise train on rows that do not mean what the producer wrote and
//! report the numbers of a well-formed one: `allowed` without the
//! requirement lets a reader that does not implement the field train the
//! same rows unconstrained, and the requirement without `allowed` is a
//! producer that meant to write sets and did not.
//!
//! Rows are left at their own listed width here. Widening them to the
//! common width the model's allowed-id input takes belongs with the
//! merge, because a width can only be settled once every source has been
//! read.
//!
//! # Extending the format
//!
//! `meta.requires` is the mechanism, and it is the reason the format
//! needs no version number. A producer adding a field that only records
//! how the file was made writes it into `meta` and older readers ignore
//! it. A producer adding a field the file's *meaning* depends on — a
//! per-row constraint, say — lists that field in `meta.requires`, so a
//! reader that does not implement it refuses the file instead of
//! training on a reading its producer did not intend. Unknown-and-
//! ignored is therefore a choice the writer makes per field rather than
//! a property of every field a reader has not heard of.
//!
//! # What this module does not do
//!
//! The loader's responsibility ends at "file → validated rows".
//! Batching, padding, the per-row side channels and any repetition of
//! the row list belong to [`crate::train::TokenizedDataset`], which
//! already implements them; duplicating any of it here would give a
//! corpus-backed run different semantics from every other dataset for
//! no reason a caller could see.
//!
//! # Combining several files
//!
//! [`interleave`] merges sources round-robin rather than concatenating
//! them. Concatenation makes the source a function of how far the run
//! has got, which a run binding something per source cannot separate
//! from the thing it is supposed to be binding. Sources of unequal
//! length drop out as they are exhausted, so the tail is whatever the
//! largest ones have left; nothing is duplicated to even the rotation
//! out, because that would change the mixture the caller named.

use std::path::{Path, PathBuf};

use serde_json::Value as Json;

/// Errors surfaced while reading a corpus file or combining several.
///
/// Every variant names the file it came from, and the row / position
/// ones name those too: a corpus is written by a program, so the reader
/// of this message is looking for the line of that program that emitted
/// the offending row.
///
/// Marked `#[non_exhaustive]`: this enum belongs to a format that is
/// meant to grow a check at a time, and every such check would
/// otherwise break a downstream `match`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CorpusError {
    /// The file could not be read.
    #[error("corpus {}: {message}", .path.display())]
    Io {
        /// File the loader was pointed at.
        path: PathBuf,
        /// What the filesystem reported.
        message: String,
    },
    /// The bytes are not a corpus JSON object, or one of its parts is
    /// not the shape the format states.
    #[error("corpus {}: {message}", .path.display())]
    Parse {
        /// File the loader was reading.
        path: PathBuf,
        /// Which part disagreed with the format, and how.
        message: String,
    },
    /// A `meta` field the format requires is absent.
    ///
    /// Separate from [`Self::Parse`] because absence and malformation
    /// have different fixes: a missing `meta.vocab_size` means the
    /// producer never recorded the id space, and no reader can infer
    /// it from the rows (an id space larger than the ids that happen to
    /// occur is the normal case).
    #[error(
        "corpus {}: `{field}` is missing — the format takes \
         {{\"meta\":{{\"ctx_len\":N,\"vocab_size\":V}},\"rows\":[[id,…]]}}",
        .path.display()
    )]
    MetaMissing {
        /// File the loader was reading.
        path: PathBuf,
        /// Dotted name of the absent field (`meta`, `meta.ctx_len`, …).
        field: String,
    },
    /// A dimension the format requires to be non-zero is zero: either
    /// `meta` dimension, the row list itself, or one row's length.
    ///
    /// Refused rather than carried: a corpus contributing no tokens
    /// still changes the mixture it was named in, and a row of no
    /// tokens trains on padding alone.
    #[error("corpus {}: {what} is empty", .path.display())]
    ZeroDim {
        /// File the loader was reading.
        path: PathBuf,
        /// Which dimension came out zero (`meta.ctx_len`, `rows`,
        /// `row 3`, …).
        what: String,
    },
    /// A row holds an id at or past the `meta.vocab_size` its own file
    /// declares.
    #[error(
        "corpus {}: row {row} position {position} holds token id {token}, outside the \
         meta.vocab_size {vocab_size} the file declares",
        .path.display()
    )]
    TokenOutOfRange {
        /// File the loader was reading.
        path: PathBuf,
        /// 0-based row index within `rows`.
        row: usize,
        /// 0-based position within that row.
        position: usize,
        /// The id that was found.
        token: u64,
        /// The id space the file declares.
        vocab_size: usize,
    },
    /// A row is longer than the `meta.ctx_len` its own file declares.
    ///
    /// Refused rather than windowed: the file contradicts itself, and
    /// the tail past `ctx_len` would otherwise be dropped per batch
    /// with nothing said about it. The usual cause is a `ctx_len` that
    /// changed after the writer was last run.
    #[error(
        "corpus {}: row {row} holds {len} ids, past the meta.ctx_len {ctx_len} the file \
         declares",
        .path.display()
    )]
    RowTooLong {
        /// File the loader was reading.
        path: PathBuf,
        /// 0-based row index within `rows`.
        row: usize,
        /// How many ids that row holds.
        len: usize,
        /// The width the file declares.
        ctx_len: usize,
    },
    /// `meta.requires` names a `meta` field this reader does not
    /// implement.
    ///
    /// Refused rather than loaded without it: the producer listed the
    /// field precisely because the rows do not mean what they say
    /// without it, so ignoring it would train on a reading the file was
    /// not written for.
    #[error(
        "corpus {}: meta.requires names `{field}`, which this reader does not implement — \
         the file's producer marked it as load-bearing, so the rows do not mean what they \
         say without it",
        .path.display()
    )]
    UnsupportedRequirement {
        /// File the loader was reading.
        path: PathBuf,
        /// The `meta` field the file requires a reader to understand.
        field: String,
    },
    /// `meta.requires` lists `per_row_allowed` and the file carries no
    /// top-level `allowed` array.
    ///
    /// The producer marked the sets as load-bearing and then did not
    /// write them, so the rows on their own are not the corpus it meant
    /// to hand over.
    #[error(
        "corpus {}: meta.requires lists `per_row_allowed` and there is no top-level \
         `allowed` array — the requirement says the rows are not the whole corpus, and \
         the sets it points at are absent",
        .path.display()
    )]
    AllowedMissing {
        /// File the loader was reading.
        path: PathBuf,
    },
    /// The file carries a top-level `allowed` array without listing
    /// `per_row_allowed` in `meta.requires`.
    ///
    /// Refused rather than read: unannounced, a reader that does not
    /// implement the field trains the same rows unconstrained and
    /// reports the same numbers, so the two runs cannot be told apart
    /// afterwards.
    #[error(
        "corpus {}: a top-level `allowed` array is present without `per_row_allowed` in \
         meta.requires — unannounced, a reader that does not implement the field trains \
         these rows unconstrained and reports the same numbers",
        .path.display()
    )]
    AllowedUnannounced {
        /// File the loader was reading.
        path: PathBuf,
    },
    /// `allowed` and `rows` are not the same length.
    #[error(
        "corpus {}: `allowed` holds {entries} entr(ies) for {rows} row(s) — the two are \
         parallel, so it takes exactly one entry per row",
        .path.display()
    )]
    AllowedCountMismatch {
        /// File the loader was reading.
        path: PathBuf,
        /// How many entries `allowed` holds.
        entries: usize,
        /// How many rows the file holds.
        rows: usize,
    },
    /// An `allowed` entry is keyed by something that is not a 1-based
    /// position.
    #[error(
        "corpus {}: allowed[{row}] is keyed by {key:?}, which is not a 1-based position \
         ({why})",
        .path.display()
    )]
    AllowedPositionKey {
        /// File the loader was reading.
        path: PathBuf,
        /// 0-based row index within `rows`.
        row: usize,
        /// The key as it was written.
        key: String,
        /// Why it could not be read as a position.
        why: String,
    },
    /// An `allowed` set names an id at or past the `meta.vocab_size` its
    /// own file declares.
    #[error(
        "corpus {}: allowed[{row}] position {position} allows token id {token}, outside \
         the meta.vocab_size {vocab_size} the file declares",
        .path.display()
    )]
    AllowedIdOutOfRange {
        /// File the loader was reading.
        path: PathBuf,
        /// 0-based row index within `rows`.
        row: usize,
        /// 1-based position the set was listed at.
        position: usize,
        /// The id that was found.
        token: u64,
        /// The id space the file declares.
        vocab_size: usize,
    },
    /// Files being combined disagree about whether they carry allowed-id
    /// sets at all.
    ///
    /// Refused rather than merged: the rows of the source without sets
    /// would train unconstrained inside a run the caller set up as a
    /// constrained one, and nothing downstream distinguishes "this row
    /// had no constraint recorded" from "this row was unconstrained".
    #[error(
        "corpus {} {} allowed-id sets and {} {} — a merge of the two would train part of \
         the rows unconstrained inside a run set up as a constrained one",
        .first.display(),
        if *.first_has { "carries" } else { "carries no" },
        .other.display(),
        if *.first_has { "does not" } else { "does" }
    )]
    AllowedPresenceMismatch {
        /// First file of the combination (the one setting the shape).
        first: PathBuf,
        /// Whether that file carries the sets.
        first_has: bool,
        /// File that disagreed.
        other: PathBuf,
    },
    /// Two files being combined disagree about the shape their rows
    /// were written for.
    ///
    /// Refused rather than resolved to either side: one model is
    /// trained on the merged rows, and it has exactly one context
    /// length and one id space.
    #[error(
        "corpus {} declares ctx_len {first_ctx_len} / vocab_size {first_vocab_size} and \
         {} declares ctx_len {ctx_len} / vocab_size {vocab_size} — one model cannot be \
         trained on both",
        .first.display(), .other.display()
    )]
    ShapeMismatch {
        /// First file of the combination (the one setting the shape).
        first: PathBuf,
        /// Context length that file declares.
        first_ctx_len: usize,
        /// Id space that file declares.
        first_vocab_size: usize,
        /// File that disagreed.
        other: PathBuf,
        /// Context length the disagreeing file declares.
        ctx_len: usize,
        /// Id space the disagreeing file declares.
        vocab_size: usize,
    },
    /// [`interleave`] was handed no sources at all.
    ///
    /// Refused rather than answered with an empty row list: an empty
    /// dataset fails later, at the training loop, as a step count the
    /// rows could not cover — which reads as a run that was too short
    /// rather than as a call that named nothing.
    #[error("interleaving takes at least one corpus, and none were given")]
    NoSources,
}

/// One loaded corpus file: the format's two `meta` dimensions plus the
/// validated rows.
///
/// [`CorpusFile::load`] is the only way to build one, so every value of
/// this type has been through the format's checks: both dimensions are
/// non-zero, there is at least one row, no row is empty, no row is
/// longer than `ctx_len`, and every id is below `vocab_size`. The
/// consumers in this crate rely on that — [`interleave_labelled`]
/// checks that the sources agree with each other, not that each is
/// well-formed — which is why the fields are read through accessors
/// rather than assembled by a caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusFile {
    path: PathBuf,
    ctx_len: usize,
    vocab_size: usize,
    rows: Vec<Vec<u32>>,
    allowed: Option<Vec<Vec<Vec<u32>>>>,
}

impl CorpusFile {
    /// Where the file was read from.
    ///
    /// Kept so the errors raised while *combining* files can name which
    /// file they mean — see [`CorpusError::ShapeMismatch`].
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sequence length the rows were written for (`meta.ctx_len`),
    /// non-zero and at least as large as the longest row.
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    /// Id space the rows are drawn from (`meta.vocab_size`), non-zero
    /// and above every id in [`Self::rows`].
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Token id sequences, unpadded, in file order: at least one, none
    /// of them empty.
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// The allowed-id sets, dense by 0-based position and parallel to
    /// [`Self::rows`], when the file carries them — `None` when it does
    /// not, which is the majority of corpora.
    ///
    /// Each row runs to its own last listed position and holds the empty
    /// set, meaning unconstrained, everywhere nobody listed. Rows are
    /// deliberately left ragged: the common width the model's allowed-id
    /// input takes cannot be settled until every source of a merge has
    /// been read, so widening belongs with the merge rather than here.
    pub fn allowed(&self) -> Option<&[Vec<Vec<u32>>]> {
        self.allowed.as_deref()
    }

    /// Read and validate the corpus file at `path`.
    ///
    /// # Errors
    ///
    /// [`CorpusError::Io`] when the file cannot be read,
    /// [`CorpusError::Parse`] when the bytes are not the JSON object
    /// the format states, [`CorpusError::MetaMissing`] when a required
    /// `meta` field is absent, [`CorpusError::ZeroDim`] when a
    /// dimension the format requires to be non-zero is zero,
    /// [`CorpusError::TokenOutOfRange`] when a row holds an id at or
    /// past the declared `meta.vocab_size`,
    /// [`CorpusError::RowTooLong`] when a row is longer than the
    /// declared `meta.ctx_len`, and
    /// [`CorpusError::UnsupportedRequirement`] when `meta.requires`
    /// names something this reader does not implement.
    ///
    /// For a file carrying allowed-id sets, additionally
    /// [`CorpusError::AllowedMissing`] /
    /// [`CorpusError::AllowedUnannounced`] when the requirement and the
    /// field disagree, [`CorpusError::AllowedCountMismatch`] when
    /// `allowed` is not parallel to `rows`,
    /// [`CorpusError::AllowedPositionKey`] when an entry is keyed by
    /// something that is not a 1-based position, and
    /// [`CorpusError::AllowedIdOutOfRange`] when a set names an id at or
    /// past the declared `meta.vocab_size`.
    pub fn load(path: &Path) -> Result<Self, CorpusError> {
        let text = std::fs::read_to_string(path).map_err(|e| CorpusError::Io {
            path: path.to_path_buf(),
            message: format!("could not be read: {e}"),
        })?;
        Self::parse(&text, path)
    }

    /// Parse already-read bytes, which is [`Self::load`] minus the
    /// filesystem; `path` is carried purely so the errors can name the
    /// source.
    fn parse(text: &str, path: &Path) -> Result<Self, CorpusError> {
        let parse_err = |message: String| CorpusError::Parse {
            path: path.to_path_buf(),
            message,
        };
        let doc: Json = serde_json::from_str(text).map_err(|e| {
            parse_err(format!(
                "is not JSON ({e}) — the format takes \
                 {{\"meta\":{{\"ctx_len\":N,\"vocab_size\":V}},\"rows\":[[id,…]]}}"
            ))
        })?;
        let obj = doc
            .as_object()
            .ok_or_else(|| parse_err(format!("is a JSON {}, not an object", kind(&doc))))?;

        let meta = obj.get("meta").ok_or_else(|| CorpusError::MetaMissing {
            path: path.to_path_buf(),
            field: "meta".to_string(),
        })?;
        let meta = meta
            .as_object()
            .ok_or_else(|| parse_err(format!("meta is a JSON {}, not an object", kind(meta))))?;
        let ctx_len = read_dim(path, meta, "ctx_len")?;
        let vocab_size = read_dim(path, meta, "vocab_size")?;
        let requires_allowed = check_requirements(path, meta)?;

        let rows_val = obj
            .get("rows")
            .ok_or_else(|| parse_err("has no `rows` array".to_string()))?;
        let rows_val = rows_val
            .as_array()
            .ok_or_else(|| parse_err(format!("rows is a JSON {}, not an array", kind(rows_val))))?;
        if rows_val.is_empty() {
            return Err(CorpusError::ZeroDim {
                path: path.to_path_buf(),
                what: "rows".to_string(),
            });
        }

        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(rows_val.len());
        for (row, row_val) in rows_val.iter().enumerate() {
            let ids_val = row_val.as_array().ok_or_else(|| {
                parse_err(format!(
                    "row {row} is a JSON {}, not an array of token ids",
                    kind(row_val)
                ))
            })?;
            if ids_val.is_empty() {
                return Err(CorpusError::ZeroDim {
                    path: path.to_path_buf(),
                    what: format!("row {row}"),
                });
            }
            if ids_val.len() > ctx_len {
                return Err(CorpusError::RowTooLong {
                    path: path.to_path_buf(),
                    row,
                    len: ids_val.len(),
                    ctx_len,
                });
            }
            let mut ids: Vec<u32> = Vec::with_capacity(ids_val.len());
            for (position, id_val) in ids_val.iter().enumerate() {
                let id = id_val.as_u64().ok_or_else(|| {
                    parse_err(format!(
                        "row {row} position {position} is a JSON {}, not a non-negative integer",
                        kind(id_val)
                    ))
                })?;
                if id >= vocab_size as u64 {
                    return Err(CorpusError::TokenOutOfRange {
                        path: path.to_path_buf(),
                        row,
                        position,
                        token: id,
                        vocab_size,
                    });
                }
                // The bound above already places the id below
                // `vocab_size`; this only fails for a declared id space
                // wider than a token id can be, which no model reads.
                ids.push(u32::try_from(id).map_err(|_| {
                    parse_err(format!(
                        "row {row} position {position} holds token id {id}, wider than a \
                         32-bit token id"
                    ))
                })?);
            }
            rows.push(ids);
        }

        // The requirement and the field are checked against each other
        // before either is read, so a file that disagrees with itself is
        // named for the disagreement rather than for whatever the sets
        // then fail on.
        let allowed = match (requires_allowed, obj.get("allowed")) {
            (true, None) => {
                return Err(CorpusError::AllowedMissing {
                    path: path.to_path_buf(),
                })
            }
            (false, Some(_)) => {
                return Err(CorpusError::AllowedUnannounced {
                    path: path.to_path_buf(),
                })
            }
            (false, None) => None,
            (true, Some(value)) => Some(read_allowed(path, value, &rows, vocab_size)?),
        };

        Ok(Self {
            path: path.to_path_buf(),
            ctx_len,
            vocab_size,
            rows,
            allowed,
        })
    }
}

/// Read the top-level `allowed` array into the dense `[row][position]`
/// form, checking it against the rows it is parallel to.
///
/// The keys are 1-based positions and the entries are sparse, so a row's
/// dense list runs to its own last listed position and holds the empty
/// set — unconstrained — everywhere nobody listed.
fn read_allowed(
    path: &Path,
    value: &Json,
    rows: &[Vec<u32>],
    vocab_size: usize,
) -> Result<Vec<Vec<Vec<u32>>>, CorpusError> {
    let parse_err = |message: String| CorpusError::Parse {
        path: path.to_path_buf(),
        message,
    };
    let entries = value
        .as_array()
        .ok_or_else(|| parse_err(format!("allowed is a JSON {}, not an array", kind(value))))?;
    if entries.len() != rows.len() {
        return Err(CorpusError::AllowedCountMismatch {
            path: path.to_path_buf(),
            entries: entries.len(),
            rows: rows.len(),
        });
    }

    let mut dense = Vec::with_capacity(entries.len());
    for (row, entry) in entries.iter().enumerate() {
        let map = entry.as_object().ok_or_else(|| {
            parse_err(format!(
                "allowed[{row}] is a JSON {}, not an object keyed by 1-based position",
                kind(entry)
            ))
        })?;

        // Collected before the row is sized: the width is the largest
        // position anybody listed, which is not known until the last key
        // has been read.
        let mut listed: Vec<(usize, Vec<u32>)> = Vec::with_capacity(map.len());
        for (key, ids_val) in map {
            let position = match key.trim().parse::<usize>() {
                Ok(0) => {
                    return Err(CorpusError::AllowedPositionKey {
                        path: path.to_path_buf(),
                        row,
                        key: key.clone(),
                        why: "the positions are 1-based".to_string(),
                    })
                }
                Ok(position) => position,
                Err(e) => {
                    return Err(CorpusError::AllowedPositionKey {
                        path: path.to_path_buf(),
                        row,
                        key: key.clone(),
                        why: e.to_string(),
                    })
                }
            };
            let ids_val = ids_val.as_array().ok_or_else(|| {
                parse_err(format!(
                    "allowed[{row}] position {position} is a JSON {}, not an array of \
                     token ids",
                    kind(ids_val)
                ))
            })?;
            let mut ids = Vec::with_capacity(ids_val.len());
            for id_val in ids_val {
                let id = id_val.as_u64().ok_or_else(|| {
                    parse_err(format!(
                        "allowed[{row}] position {position} names a JSON {}, not a \
                         non-negative token id",
                        kind(id_val)
                    ))
                })?;
                if id >= vocab_size as u64 {
                    return Err(CorpusError::AllowedIdOutOfRange {
                        path: path.to_path_buf(),
                        row,
                        position,
                        token: id,
                        vocab_size,
                    });
                }
                // The bound above already places the id below
                // `vocab_size`; this only fails for a declared id space
                // wider than a token id can be, which no model reads.
                ids.push(u32::try_from(id).map_err(|_| {
                    parse_err(format!(
                        "allowed[{row}] position {position} names token id {id}, wider \
                         than a 32-bit token id"
                    ))
                })?);
            }
            listed.push((position, ids));
        }

        let width = listed.iter().map(|(p, _)| *p).max().unwrap_or(0);
        let mut positions = vec![Vec::new(); width];
        for (position, ids) in listed {
            positions[position - 1] = ids;
        }
        dense.push(positions);
    }
    Ok(dense)
}

/// The fields this version acts on, and so the entries it accepts in
/// `meta.requires`.
///
/// `per_row_allowed` names the top-level `allowed` array rather than a
/// `meta` field: what the list carries is the name of the thing a reader
/// has to implement, and a per-row constraint has nowhere to live inside
/// `meta`.
const UNDERSTOOD_REQUIREMENTS: [&str; 3] = ["ctx_len", "vocab_size", "per_row_allowed"];

/// Check `meta.requires`, the format's forward-compatibility list, and
/// report whether it asks for the per-row allowed-id sets.
///
/// Absent or empty is the older behaviour — every unknown `meta` field
/// is ignored. An entry outside [`UNDERSTOOD_REQUIREMENTS`] is refused
/// by name rather than ignored: the producer listed it because the rows
/// do not mean what they say without it.
fn check_requirements(
    path: &Path,
    meta: &serde_json::Map<String, Json>,
) -> Result<bool, CorpusError> {
    let Some(value) = meta.get("requires") else {
        return Ok(false);
    };
    let listed = value.as_array().ok_or_else(|| CorpusError::Parse {
        path: path.to_path_buf(),
        message: format!(
            "meta.requires is a JSON {}, not an array of meta field names",
            kind(value)
        ),
    })?;
    let mut per_row_allowed = false;
    for (index, entry) in listed.iter().enumerate() {
        let field = entry.as_str().ok_or_else(|| CorpusError::Parse {
            path: path.to_path_buf(),
            message: format!(
                "meta.requires[{index}] is a JSON {}, not a meta field name",
                kind(entry)
            ),
        })?;
        if !UNDERSTOOD_REQUIREMENTS.contains(&field) {
            return Err(CorpusError::UnsupportedRequirement {
                path: path.to_path_buf(),
                field: field.to_string(),
            });
        }
        per_row_allowed |= field == "per_row_allowed";
    }
    Ok(per_row_allowed)
}

/// Read one required non-zero `meta` dimension.
fn read_dim(
    path: &Path,
    meta: &serde_json::Map<String, Json>,
    field: &str,
) -> Result<usize, CorpusError> {
    let value = meta.get(field).ok_or_else(|| CorpusError::MetaMissing {
        path: path.to_path_buf(),
        field: format!("meta.{field}"),
    })?;
    let raw = value.as_u64().ok_or_else(|| CorpusError::Parse {
        path: path.to_path_buf(),
        message: format!(
            "meta.{field} is a JSON {}, not a non-negative integer",
            kind(value)
        ),
    })?;
    let dim = usize::try_from(raw).map_err(|_| CorpusError::Parse {
        path: path.to_path_buf(),
        message: format!("meta.{field} = {raw} does not fit in this platform's usize"),
    })?;
    if dim == 0 {
        return Err(CorpusError::ZeroDim {
            path: path.to_path_buf(),
            what: format!("meta.{field}"),
        });
    }
    Ok(dim)
}

/// Short kind label for a JSON value, for error messages. The value
/// itself is deliberately not printed: a corpus row can be long, and
/// what a reader needs is which part disagreed rather than its content.
fn kind(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Number(_) => "number",
        Json::String(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

/// One row of an interleaved corpus, together with the source it came
/// from.
///
/// The source index is what lets a caller attach something per corpus —
/// a condition, most often — to the merged rows: the merge re-orders
/// them, so the pairing cannot be reconstructed from the row list
/// afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InterleavedRow {
    /// Index into the `sources` slice this row was taken from.
    pub source: usize,
    /// The row's token ids.
    pub ids: Vec<u32>,
    /// The row's allowed-id sets, dense by 0-based position, when the
    /// sources carry them.
    ///
    /// Ragged across rows: each runs to its own last listed position.
    /// The common width the model's allowed-id input takes is applied by
    /// whoever builds the dataset, which is also where the truncation to
    /// `ctx_len` happens.
    pub allowed: Option<Vec<Vec<u32>>>,
}

/// Merge several corpora round-robin, keeping each row's source.
///
/// Rows come out in the order source 0 row 0, source 1 row 0, …,
/// source 0 row 1, … — deterministic, and with sources dropping out of
/// the rotation as they are exhausted. See the module documentation for
/// why the merge is a rotation rather than a concatenation.
///
/// # Errors
///
/// [`CorpusError::NoSources`] when `sources` is empty,
/// [`CorpusError::ShapeMismatch`] when the files disagree about
/// `ctx_len` / `vocab_size`, and
/// [`CorpusError::AllowedPresenceMismatch`] when they disagree about
/// whether they carry allowed-id sets.
pub fn interleave_labelled(sources: &[&CorpusFile]) -> Result<Vec<InterleavedRow>, CorpusError> {
    let Some(first) = sources.first() else {
        return Err(CorpusError::NoSources);
    };
    let first_has_allowed = first.allowed.is_some();
    for other in sources.iter().skip(1) {
        if other.ctx_len != first.ctx_len || other.vocab_size != first.vocab_size {
            return Err(CorpusError::ShapeMismatch {
                first: first.path.clone(),
                first_ctx_len: first.ctx_len,
                first_vocab_size: first.vocab_size,
                other: other.path.clone(),
                ctx_len: other.ctx_len,
                vocab_size: other.vocab_size,
            });
        }
        // A merge of a constrained source with an unconstrained one puts
        // rows nobody recorded a constraint for into a run set up as a
        // constrained one, and downstream an absent set and an
        // unconstrained position are the same value.
        if other.allowed.is_some() != first_has_allowed {
            return Err(CorpusError::AllowedPresenceMismatch {
                first: first.path.clone(),
                first_has: first_has_allowed,
                other: other.path.clone(),
            });
        }
    }

    let longest = sources.iter().fold(0, |acc, s| acc.max(s.rows.len()));
    let total: usize = sources.iter().map(|s| s.rows.len()).sum();
    let mut out = Vec::with_capacity(total);
    for position in 0..longest {
        for (source, corpus) in sources.iter().enumerate() {
            if let Some(ids) = corpus.rows.get(position) {
                out.push(InterleavedRow {
                    source,
                    ids: ids.clone(),
                    // Read at the same index as the row, in the same
                    // iteration, so the rotation cannot shift one
                    // against the other. The index is in range because
                    // the loader refuses an `allowed` that is not
                    // parallel to `rows`, and the row above is present.
                    allowed: corpus.allowed.as_ref().map(|sets| sets[position].clone()),
                });
            }
        }
    }
    Ok(out)
}

/// [`interleave_labelled`] for a caller that attaches nothing per
/// source: the merged rows alone, in the same order.
///
/// # Errors
///
/// As [`interleave_labelled`].
pub fn interleave(sources: &[&CorpusFile]) -> Result<Vec<Vec<u32>>, CorpusError> {
    Ok(interleave_labelled(sources)?
        .into_iter()
        .map(|row| row.ids)
        .collect())
}

#[cfg(test)]
mod tests {
    //! What is worth pinning here is the refusals: a corpus is written
    //! by a program, and every one of these failures is a program that
    //! wrote something its own `meta` contradicts. The happy path
    //! covers the one property a producer relies on — that a field this
    //! version has never heard of does not stop the file loading.

    use super::*;
    use tempfile::TempDir;

    /// Write `body` into `dir` under `name` and return its path.
    fn write(dir: &TempDir, name: &str, body: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, body).expect("write corpus fixture");
        path
    }

    #[test]
    fn a_well_formed_corpus_loads_and_unknown_meta_fields_are_ignored() {
        let dir = TempDir::new().expect("tempdir");
        let path = write(
            &dir,
            "a.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10, "written_by": "some producer",
                          "seed": 7 },
                 "rows": [[1, 2, 3], [4, 5]] }"#,
        );
        let corpus = CorpusFile::load(&path).expect("a well-formed corpus loads");
        assert_eq!(corpus.ctx_len(), 4);
        assert_eq!(corpus.vocab_size(), 10);
        assert_eq!(corpus.rows(), [vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(corpus.path(), path);
    }

    /// `meta.requires` is what a producer uses to mark a `meta` field as
    /// load-bearing, so the two cases that must not change are the ones
    /// where it asks for nothing.
    #[test]
    fn a_requirement_this_reader_does_not_implement_is_refused_by_name() {
        let dir = TempDir::new().expect("tempdir");

        let empty = write(
            &dir,
            "empty_requires.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10, "requires": [] },
                 "rows": [[1]] }"#,
        );
        CorpusFile::load(&empty).expect("an empty requirement list asks for nothing");

        let understood = write(
            &dir,
            "understood.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["vocab_size"],
                          "seed": 7 },
                 "rows": [[1]] }"#,
        );
        CorpusFile::load(&understood).expect("a field this version reads is understood");

        let unknown = write(
            &dir,
            "unknown.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["allowed_ids"],
                          "allowed_ids": [[1, 2]] },
                 "rows": [[1]] }"#,
        );
        let err = CorpusFile::load(&unknown).expect_err("a requirement from a later writer");
        match &err {
            CorpusError::UnsupportedRequirement { field, path } => {
                assert_eq!(field, "allowed_ids");
                assert_eq!(path, &unknown);
            }
            other => panic!("expected UnsupportedRequirement, got {other:?}"),
        }
        assert!(err.to_string().contains("allowed_ids"), "{err}");

        let malformed = write(
            &dir,
            "malformed_requires.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10, "requires": "allowed_ids" },
                 "rows": [[1]] }"#,
        );
        let err = CorpusFile::load(&malformed).expect_err("requires is not an array");
        assert!(
            err.to_string().contains("meta.requires is a JSON string"),
            "{err}"
        );
    }

    /// A row past the width its own file declares is the same producer
    /// bug as an id past the declared id space: the file contradicts
    /// itself, and the tail would otherwise be dropped per batch.
    #[test]
    fn a_row_longer_than_the_declared_ctx_len_is_refused_by_row() {
        let dir = TempDir::new().expect("tempdir");
        let path = write(
            &dir,
            "long_row.json",
            r#"{ "meta": { "ctx_len": 2, "vocab_size": 10 }, "rows": [[1, 2], [3, 4, 5]] }"#,
        );
        let err = CorpusFile::load(&path).expect_err("a row of 3 ids at ctx_len 2");
        match &err {
            CorpusError::RowTooLong {
                row, len, ctx_len, ..
            } => assert_eq!((*row, *len, *ctx_len), (1, 3, 2)),
            other => panic!("expected RowTooLong, got {other:?}"),
        }
        let text = err.to_string();
        assert!(text.contains("long_row.json"), "{text}");
        assert!(text.contains("row 1"), "{text}");
    }

    #[test]
    fn a_missing_file_names_the_path_it_was_pointed_at() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("absent.json");
        let err = CorpusFile::load(&path).expect_err("no such file");
        assert!(matches!(err, CorpusError::Io { .. }), "{err:?}");
        assert!(err.to_string().contains("absent.json"), "{err}");
    }

    #[test]
    fn a_missing_meta_field_is_refused_by_name() {
        let dir = TempDir::new().expect("tempdir");

        let no_meta = write(&dir, "no_meta.json", r#"{ "rows": [[1]] }"#);
        let err = CorpusFile::load(&no_meta).expect_err("no meta block");
        match &err {
            CorpusError::MetaMissing { field, .. } => assert_eq!(field, "meta"),
            other => panic!("expected MetaMissing, got {other:?}"),
        }

        let no_vocab = write(
            &dir,
            "no_vocab.json",
            r#"{ "meta": { "ctx_len": 4 }, "rows": [[1]] }"#,
        );
        let err = CorpusFile::load(&no_vocab).expect_err("no vocab_size");
        match &err {
            CorpusError::MetaMissing { field, path } => {
                assert_eq!(field, "meta.vocab_size");
                assert_eq!(path, &no_vocab);
            }
            other => panic!("expected MetaMissing, got {other:?}"),
        }
        assert!(err.to_string().contains("no_vocab.json"), "{err}");
    }

    #[test]
    fn a_token_outside_the_declared_id_space_names_the_file_and_the_position() {
        let dir = TempDir::new().expect("tempdir");
        let path = write(
            &dir,
            "wide.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 3 }, "rows": [[1, 2], [0, 9]] }"#,
        );
        let err = CorpusFile::load(&path).expect_err("id 9 is outside a 3-id space");
        match &err {
            CorpusError::TokenOutOfRange {
                row,
                position,
                token,
                vocab_size,
                ..
            } => assert_eq!((*row, *position, *token, *vocab_size), (1, 1, 9, 3)),
            other => panic!("expected TokenOutOfRange, got {other:?}"),
        }
        let text = err.to_string();
        assert!(text.contains("wide.json"), "{text}");
        assert!(text.contains("row 1 position 1"), "{text}");
    }

    #[test]
    fn every_zero_dimension_is_refused_and_named() {
        let dir = TempDir::new().expect("tempdir");

        let zero_ctx = write(
            &dir,
            "zero_ctx.json",
            r#"{ "meta": { "ctx_len": 0, "vocab_size": 3 }, "rows": [[1]] }"#,
        );
        let err = CorpusFile::load(&zero_ctx).expect_err("ctx_len 0");
        match &err {
            CorpusError::ZeroDim { what, .. } => assert_eq!(what, "meta.ctx_len"),
            other => panic!("expected ZeroDim, got {other:?}"),
        }

        let no_rows = write(
            &dir,
            "no_rows.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 3 }, "rows": [] }"#,
        );
        let err = CorpusFile::load(&no_rows).expect_err("no rows");
        match &err {
            CorpusError::ZeroDim { what, .. } => assert_eq!(what, "rows"),
            other => panic!("expected ZeroDim, got {other:?}"),
        }

        let empty_row = write(
            &dir,
            "empty_row.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 3 }, "rows": [[1], []] }"#,
        );
        let err = CorpusFile::load(&empty_row).expect_err("an empty row");
        match &err {
            CorpusError::ZeroDim { what, .. } => assert_eq!(what, "row 1"),
            other => panic!("expected ZeroDim, got {other:?}"),
        }
    }

    #[test]
    fn a_malformed_document_is_refused_at_the_part_that_disagrees() {
        let dir = TempDir::new().expect("tempdir");

        let not_json = write(&dir, "not_json.json", "rows = [[1]]");
        let err = CorpusFile::load(&not_json).expect_err("not JSON");
        assert!(matches!(err, CorpusError::Parse { .. }), "{err:?}");
        assert!(err.to_string().contains("is not JSON"), "{err}");

        let rows_object = write(
            &dir,
            "rows_object.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 3 }, "rows": { "0": [1] } }"#,
        );
        let err = CorpusFile::load(&rows_object).expect_err("rows is not an array");
        assert!(err.to_string().contains("rows is a JSON object"), "{err}");

        let float_id = write(
            &dir,
            "float_id.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 3 }, "rows": [[1.5]] }"#,
        );
        let err = CorpusFile::load(&float_id).expect_err("a fractional token id");
        assert!(
            err.to_string()
                .contains("row 0 position 0 is a JSON number"),
            "{err}"
        );

        let text_dim = write(
            &dir,
            "text_dim.json",
            r#"{ "meta": { "ctx_len": "four", "vocab_size": 3 }, "rows": [[1]] }"#,
        );
        let err = CorpusFile::load(&text_dim).expect_err("a textual dimension");
        assert!(
            err.to_string().contains("meta.ctx_len is a JSON string"),
            "{err}"
        );
    }

    /// The merge order is part of the format's contract: a caller
    /// binding something per source depends on the rotation, and a
    /// caller reading the rows back depends on it being reproducible.
    #[test]
    fn interleaving_rotates_between_sources_and_keeps_their_labels() {
        let dir = TempDir::new().expect("tempdir");
        let a = CorpusFile::load(&write(
            &dir,
            "a.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10 }, "rows": [[1], [2], [3]] }"#,
        ))
        .expect("corpus a");
        let b = CorpusFile::load(&write(
            &dir,
            "b.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10 }, "rows": [[7]] }"#,
        ))
        .expect("corpus b");

        // The shorter source drops out of the rotation once drained;
        // the longer one's surplus trails at the end in its own order.
        let rows = interleave(&[&a, &b]).expect("two agreeing corpora");
        assert_eq!(rows, vec![vec![1], vec![7], vec![2], vec![3]]);

        let labelled = interleave_labelled(&[&a, &b]).expect("two agreeing corpora");
        let sources: Vec<usize> = labelled.iter().map(|r| r.source).collect();
        assert_eq!(sources, vec![0, 1, 0, 0]);
        let ids: Vec<Vec<u32>> = labelled.into_iter().map(|r| r.ids).collect();
        assert_eq!(ids, rows, "both entry points merge in the same order");

        // Order follows the argument order, not the file names.
        let reversed = interleave(&[&b, &a]).expect("two agreeing corpora");
        assert_eq!(reversed, vec![vec![7], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn corpora_that_disagree_about_their_shape_are_refused() {
        let dir = TempDir::new().expect("tempdir");
        let a = CorpusFile::load(&write(
            &dir,
            "a.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10 }, "rows": [[1]] }"#,
        ))
        .expect("corpus a");
        let narrow = CorpusFile::load(&write(
            &dir,
            "narrow.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 6 }, "rows": [[1]] }"#,
        ))
        .expect("corpus narrow");
        let short = CorpusFile::load(&write(
            &dir,
            "short.json",
            r#"{ "meta": { "ctx_len": 2, "vocab_size": 10 }, "rows": [[1]] }"#,
        ))
        .expect("corpus short");

        let err = interleave(&[&a, &narrow]).expect_err("disagreeing id spaces");
        match &err {
            CorpusError::ShapeMismatch {
                first_vocab_size,
                vocab_size,
                ..
            } => assert_eq!((*first_vocab_size, *vocab_size), (10, 6)),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        let text = err.to_string();
        assert!(
            text.contains("a.json") && text.contains("narrow.json"),
            "{text}"
        );

        let err = interleave(&[&a, &short]).expect_err("disagreeing context lengths");
        assert!(matches!(err, CorpusError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn interleaving_nothing_is_refused_rather_than_answered_with_no_rows() {
        let err = interleave(&[]).expect_err("no sources");
        assert!(matches!(err, CorpusError::NoSources), "{err:?}");
    }

    #[test]
    fn allowed_sets_are_read_sparse_and_come_back_dense_at_their_own_width() {
        let dir = TempDir::new().expect("tempdir");
        let path = write(
            &dir,
            "allowed.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10,
                          "requires": ["per_row_allowed"] },
                 "rows": [[3, 7, 1, 5], [2, 9]],
                 "allowed": [{ "2": [7, 8], "4": [5] }, { "1": [2] }] }"#,
        );
        let corpus = CorpusFile::load(&path).expect("a corpus carrying allowed-id sets loads");
        let allowed = corpus.allowed().expect("the sets are read");

        // Sparse in, dense out: a position nobody listed holds the empty
        // set, and each row runs to its own last listed position rather
        // than to a width shared with the other rows.
        assert_eq!(
            allowed,
            [vec![vec![], vec![7, 8], vec![], vec![5]], vec![vec![2]],]
        );
    }

    /// The requirement and the field have to agree in both directions:
    /// each half alone produces a run that reports the numbers of a
    /// well-formed one while training on something else.
    #[test]
    fn the_requirement_and_the_allowed_field_are_checked_against_each_other() {
        let dir = TempDir::new().expect("tempdir");

        let announced = write(
            &dir,
            "announced.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10,
                          "requires": ["per_row_allowed"] },
                 "rows": [[1, 2]] }"#,
        );
        let err = CorpusFile::load(&announced).expect_err("announced but absent");
        assert!(matches!(err, CorpusError::AllowedMissing { .. }), "{err:?}");

        let unannounced = write(
            &dir,
            "unannounced.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10 },
                 "rows": [[1, 2]],
                 "allowed": [{ "1": [1] }] }"#,
        );
        let err = CorpusFile::load(&unannounced).expect_err("present but unannounced");
        assert!(
            matches!(err, CorpusError::AllowedUnannounced { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_malformed_allowed_array_is_refused_at_the_part_that_disagrees() {
        let dir = TempDir::new().expect("tempdir");
        let load = |name: &str, body: &str| {
            CorpusFile::load(&write(&dir, name, body)).expect_err("malformed allowed")
        };
        let head = r#""meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["per_row_allowed"] }"#;

        // One entry per row, no more and no fewer: the two are parallel,
        // so a length disagreement pairs sets with the wrong rows.
        let err = load(
            "count.json",
            &format!(r#"{{ {head}, "rows": [[1], [2]], "allowed": [{{ "1": [1] }}] }}"#),
        );
        match &err {
            CorpusError::AllowedCountMismatch { entries, rows, .. } => {
                assert_eq!((*entries, *rows), (1, 2))
            }
            other => panic!("expected AllowedCountMismatch, got {other:?}"),
        }

        // The positions are 1-based, so 0 is a producer that wrote
        // 0-based indices and would have every set off by one.
        let err = load(
            "zero.json",
            &format!(r#"{{ {head}, "rows": [[1]], "allowed": [{{ "0": [1] }}] }}"#),
        );
        match &err {
            CorpusError::AllowedPositionKey { key, why, .. } => {
                assert_eq!(key, "0");
                assert!(why.contains("1-based"), "{why}");
            }
            other => panic!("expected AllowedPositionKey, got {other:?}"),
        }

        let err = load(
            "key.json",
            &format!(r#"{{ {head}, "rows": [[1]], "allowed": [{{ "first": [1] }}] }}"#),
        );
        assert!(
            matches!(err, CorpusError::AllowedPositionKey { .. }),
            "{err:?}"
        );

        // An id past the declared space is the same fault as one in
        // `rows`, and is caught in the same place rather than several
        // layers away at an embedding lookup.
        let err = load(
            "id.json",
            &format!(r#"{{ {head}, "rows": [[1]], "allowed": [{{ "1": [1, 42] }}] }}"#),
        );
        match &err {
            CorpusError::AllowedIdOutOfRange {
                position,
                token,
                vocab_size,
                ..
            } => assert_eq!((*position, *token, *vocab_size), (1, 42, 10)),
            other => panic!("expected AllowedIdOutOfRange, got {other:?}"),
        }

        let err = load(
            "shape.json",
            &format!(r#"{{ {head}, "rows": [[1]], "allowed": [[1]] }}"#),
        );
        match &err {
            CorpusError::Parse { message, .. } => {
                assert!(message.contains("allowed[0] is a JSON array"), "{message}")
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn merging_keeps_each_row_with_its_own_sets() {
        let dir = TempDir::new().expect("tempdir");
        let head = r#""meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["per_row_allowed"] }"#;
        let a = CorpusFile::load(&write(
            &dir,
            "a.json",
            &format!(
                r#"{{ {head}, "rows": [[1], [2]], "allowed": [{{ "1": [1] }}, {{ "1": [2] }}] }}"#
            ),
        ))
        .expect("corpus a");
        let b = CorpusFile::load(&write(
            &dir,
            "b.json",
            &format!(r#"{{ {head}, "rows": [[7]], "allowed": [{{ "1": [7] }}] }}"#),
        ))
        .expect("corpus b");

        /// One merged row, flattened to what this test compares: the
        /// ids and the sets that travelled with them.
        type Paired = (Vec<u32>, Option<Vec<Vec<u32>>>);

        let merged = interleave_labelled(&[&a, &b]).expect("two agreeing corpora");
        let paired: Vec<Paired> = merged
            .into_iter()
            .map(|row| (row.ids, row.allowed))
            .collect();
        assert_eq!(
            paired,
            vec![
                (vec![1], Some(vec![vec![1]])),
                (vec![7], Some(vec![vec![7]])),
                (vec![2], Some(vec![vec![2]])),
            ],
            "the rotation moves the row and its sets together"
        );
    }

    #[test]
    fn merging_a_constrained_corpus_with_an_unconstrained_one_is_refused() {
        let dir = TempDir::new().expect("tempdir");
        let constrained = CorpusFile::load(&write(
            &dir,
            "constrained.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10,
                          "requires": ["per_row_allowed"] },
                 "rows": [[1]], "allowed": [{ "1": [1] }] }"#,
        ))
        .expect("constrained corpus");
        let plain = CorpusFile::load(&write(
            &dir,
            "plain.json",
            r#"{ "meta": { "ctx_len": 4, "vocab_size": 10 }, "rows": [[2]] }"#,
        ))
        .expect("plain corpus");

        let err = interleave(&[&constrained, &plain]).expect_err("a mixed merge");
        match &err {
            CorpusError::AllowedPresenceMismatch { first_has, .. } => assert!(*first_has),
            other => panic!("expected AllowedPresenceMismatch, got {other:?}"),
        }
        let text = err.to_string();
        assert!(
            text.contains("constrained.json") && text.contains("plain.json"),
            "{text}"
        );

        // Refused from either side, so the answer does not depend on
        // which file the caller happened to name first.
        let err = interleave(&[&plain, &constrained]).expect_err("a mixed merge, reversed");
        match &err {
            CorpusError::AllowedPresenceMismatch { first_has, .. } => assert!(!*first_has),
            other => panic!("expected AllowedPresenceMismatch, got {other:?}"),
        }
    }
}
