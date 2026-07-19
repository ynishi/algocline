//! Dataset iterator abstraction.
//!
//! Design §8.4: `alc.nn.data.jsonl`, `alc.nn.data.parquet`, and
//! `alc.nn.data.from_card` all return an opaque handle that the
//! trainer follow-up entries consume batch-by-batch. Iteration is Rust-side
//! (no per-batch Lua callback) so the trainer stays on the hot path
//! without VM cross-calls.
//!
//! Subtask invariants:
//!
//! - Batches produce `[batch_size, ctx_len]` `u32` token ids. The last
//!   batch may be short (`Batch::is_last == true`).
//! - JSONL adapter reads each line as `{ "text": "..." }` (default
//!   field name; overridable via [`DatasetOpts::text_field`]) and
//!   tokenizes with the caller-supplied [`crate::tokenizer::HfTokenizer`].
//! - `from_card` reads Card sample rows via `FileCardStore` (invariant
//!   #5); the `algocline-nn` crate does no filesystem access itself.
//!   The bridge builds a [`TokenizedDataset`] from the pre-tokenized
//!   rows and hands it back to the trainer.
//! - Parquet reading is scaffolded (see [`ParquetDataset`]); a later
//!   stage picks up the concrete reader implementation. Constructing the
//!   handle is legal, `next_batch` returns an error so no silent
//!   empty iterator is exposed (Rust exception-free discipline).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::tokenizer::{HfTokenizer, TokenizerError};

/// Errors surfaced by dataset iterators.
///
/// All variants surface loudly through `next_batch` so caller Lua sees
/// a real error rather than a silent end-of-stream.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// Filesystem IO failure while reading a JSONL / Parquet source.
    #[error("io: {0}")]
    Io(String),
    /// JSON parse failure on a JSONL row.
    #[error("json: {0}")]
    Json(String),
    /// Required text field was missing on a source row.
    #[error("missing text field '{field}' on record {index}")]
    MissingField {
        /// Field name the loader was looking for.
        field: String,
        /// 0-based record index within the source.
        index: usize,
    },
    /// Tokenizer failure while encoding a row.
    #[error("tokenize: {0}")]
    Tokenize(#[from] TokenizerError),
    /// Feature not implemented yet (e.g. Parquet body).
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

/// Iterator config shared across dataset kinds.
///
/// Every field has a sensible default; unspecified fields via the Lua
/// bridge collapse to these values.
#[derive(Debug, Clone)]
pub struct DatasetOpts {
    /// Rows per batch.
    pub batch_size: usize,
    /// Fixed sequence length per row; longer rows truncated, shorter
    /// rows padded with `pad_id`.
    pub ctx_len: usize,
    /// Randomly shuffle rows before iteration (in-memory shuffle;
    /// large corpora should stream separately — v2 carry).
    pub shuffle: bool,
    /// Pad token id used to fill short rows to `ctx_len`.
    /// Defaults to `0` which is the GPT-2 `<|endoftext|>` id and
    /// matches the nanoGPT convention.
    pub pad_id: u32,
    /// JSONL / Parquet source field to tokenize. Defaults to `"text"`.
    pub text_field: String,
}

impl Default for DatasetOpts {
    fn default() -> Self {
        Self {
            batch_size: 8,
            ctx_len: 128,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        }
    }
}

/// One training batch.
#[derive(Debug, Clone)]
pub struct Batch {
    /// `batch_size × ctx_len` token ids row-major (each inner `Vec`
    /// is exactly `ctx_len` long after padding).
    pub input_ids: Vec<Vec<u32>>,
    /// Optional per-token loss mask, shape identical to `input_ids`
    /// (each inner `Vec` of `f32`, `1.0` = counted / `0.0` = ignored).
    ///
    /// `None` = uniform mask (every token position contributes to
    /// the loss). Populated by teacher-log style datasets so a
    /// downstream `Loss::compute` call receives the mask that zeroes
    /// out prompt-region tokens, leaving only the response region to
    /// drive the gradient. See `TeacherCardDataset`.
    #[allow(clippy::type_complexity)]
    pub loss_mask: Option<Vec<Vec<f32>>>,
    /// `true` when this is the final batch of the source and the
    /// caller may want to skip a gradient step (or scale the loss).
    pub is_last: bool,
}

/// Streaming batch iterator.
///
/// Kept small on purpose so the trainer follow-up can hold a `Box<dyn Dataset>` in the
/// trainer state without a generic parameter explosion.
pub trait Dataset {
    /// Return the next batch or `None` once the source is drained.
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError>;

    /// Approximate row count (before batching), when cheaply known.
    /// `None` for streaming sources that cannot count without a full
    /// pass.
    fn len_hint(&self) -> Option<usize>;
}

/// In-memory dataset built from a `Vec<Vec<u32>>` of pre-tokenized
/// rows.
///
/// Used by the bridge's `from_card` path (invariant #5) and by tests
/// that want to feed deterministic token sequences without a JSONL
/// / Parquet fixture on disk.
pub struct TokenizedDataset {
    rows: Vec<Vec<u32>>,
    opts: DatasetOpts,
    cursor: usize,
}

impl TokenizedDataset {
    /// Construct from an owned rows vector (each row is an unpadded
    /// token id sequence; padding / truncation happens per batch).
    pub fn new(rows: Vec<Vec<u32>>, opts: DatasetOpts) -> Self {
        let mut this = Self {
            rows,
            opts,
            cursor: 0,
        };
        if this.opts.shuffle {
            this.rows.reverse(); // deterministic re-order for now; a later stage wires an RNG
        }
        this
    }

    fn take_next_batch(&mut self) -> Option<Batch> {
        if self.cursor >= self.rows.len() {
            return None;
        }
        let start = self.cursor;
        let end = (start + self.opts.batch_size).min(self.rows.len());
        self.cursor = end;
        let ctx = self.opts.ctx_len;
        let pad = self.opts.pad_id;
        let input_ids: Vec<Vec<u32>> = self.rows[start..end]
            .iter()
            .map(|row| pad_or_truncate(row, ctx, pad))
            .collect();
        Some(Batch {
            input_ids,
            loss_mask: None,
            is_last: end == self.rows.len(),
        })
    }
}

impl Dataset for TokenizedDataset {
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError> {
        Ok(self.take_next_batch())
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.rows.len())
    }
}

/// JSONL-backed dataset.
///
/// Each line is parsed as a JSON object; the field named
/// [`DatasetOpts::text_field`] is tokenized. Lines are loaded lazily
/// (line-by-line) unless [`DatasetOpts::shuffle`] is set, in which
/// case the loader materialises all rows first.
pub struct JsonlDataset {
    reader: Option<BufReader<File>>,
    buffer: Vec<Vec<u32>>,
    buffer_cursor: usize,
    opts: DatasetOpts,
    tokenizer: HfTokenizer,
    path: PathBuf,
    line_index: usize,
    total_rows: Option<usize>,
}

impl JsonlDataset {
    /// Open `path` for iteration. When `opts.shuffle` is `true`, the
    /// full file is parsed and tokenized eagerly so rows can be
    /// re-ordered; otherwise a streaming reader is retained.
    pub fn new(
        path: &Path,
        opts: DatasetOpts,
        tokenizer: HfTokenizer,
    ) -> Result<Self, DatasetError> {
        let file =
            File::open(path).map_err(|e| DatasetError::Io(format!("open {:?}: {e}", path)))?;
        let reader = BufReader::new(file);
        let mut this = Self {
            reader: Some(reader),
            buffer: Vec::new(),
            buffer_cursor: 0,
            opts,
            tokenizer,
            path: path.to_path_buf(),
            line_index: 0,
            total_rows: None,
        };
        if this.opts.shuffle {
            this.materialize_all()?;
        }
        Ok(this)
    }

    fn materialize_all(&mut self) -> Result<(), DatasetError> {
        let Some(reader) = self.reader.take() else {
            return Ok(());
        };
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| DatasetError::Io(format!("read {:?}: {e}", self.path)))?;
            if line.trim().is_empty() {
                continue;
            }
            let ids = self.tokenize_line(&line, idx)?;
            rows.push(ids);
        }
        // Reverse ordering as a deterministic "shuffle" placeholder;
        // A later stage wires a real RNG seed once the trainer opts land.
        rows.reverse();
        self.total_rows = Some(rows.len());
        self.buffer = rows;
        self.buffer_cursor = 0;
        Ok(())
    }

    fn tokenize_line(&self, line: &str, idx: usize) -> Result<Vec<u32>, DatasetError> {
        let json: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| DatasetError::Json(format!("record {idx}: {e}")))?;
        let text = json
            .get(&self.opts.text_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| DatasetError::MissingField {
                field: self.opts.text_field.clone(),
                index: idx,
            })?;
        Ok(self.tokenizer.encode(text)?)
    }

    fn fill_batch_streaming(&mut self) -> Result<Vec<Vec<u32>>, DatasetError> {
        let mut rows = Vec::with_capacity(self.opts.batch_size);
        let reader = match self.reader.as_mut() {
            Some(r) => r,
            None => return Ok(rows),
        };
        let mut line = String::new();
        while rows.len() < self.opts.batch_size {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| DatasetError::Io(format!("read {:?}: {e}", self.path)))?;
            if n == 0 {
                break;
            }
            if line.trim().is_empty() {
                self.line_index += 1;
                continue;
            }
            let idx = self.line_index;
            self.line_index += 1;
            let json: serde_json::Value = serde_json::from_str(line.trim())
                .map_err(|e| DatasetError::Json(format!("record {idx}: {e}")))?;
            let text = json
                .get(&self.opts.text_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::MissingField {
                    field: self.opts.text_field.clone(),
                    index: idx,
                })?;
            let ids = self.tokenizer.encode(text)?;
            rows.push(ids);
        }
        Ok(rows)
    }
}

impl Dataset for JsonlDataset {
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError> {
        // Shuffle path — everything is in `buffer`.
        if self.opts.shuffle {
            if self.buffer_cursor >= self.buffer.len() {
                return Ok(None);
            }
            let start = self.buffer_cursor;
            let end = (start + self.opts.batch_size).min(self.buffer.len());
            self.buffer_cursor = end;
            let ctx = self.opts.ctx_len;
            let pad = self.opts.pad_id;
            let input_ids = self.buffer[start..end]
                .iter()
                .map(|row| pad_or_truncate(row, ctx, pad))
                .collect();
            return Ok(Some(Batch {
                input_ids,
                loss_mask: None,
                is_last: end == self.buffer.len(),
            }));
        }

        // Streaming path — pull the next `batch_size` lines.
        let rows = self.fill_batch_streaming()?;
        if rows.is_empty() {
            return Ok(None);
        }
        let ctx = self.opts.ctx_len;
        let pad = self.opts.pad_id;
        let short_batch = rows.len() < self.opts.batch_size;
        let input_ids = rows
            .iter()
            .map(|row| pad_or_truncate(row, ctx, pad))
            .collect();
        Ok(Some(Batch {
            input_ids,
            loss_mask: None,
            is_last: short_batch,
        }))
    }

    fn len_hint(&self) -> Option<usize> {
        self.total_rows
    }
}

/// Parquet-backed dataset (scaffold).
///
/// A later stage lands the concrete Apache Arrow / Parquet reader
/// wiring. The current stage exposes the constructor so the Lua
/// bridge surface is stable — a
/// `next_batch` call surfaces [`DatasetError::NotImplemented`] instead
/// of silently returning an empty iterator (Rust exception-free
/// discipline — no silent drop, per the crate's Service-layer
/// error-propagation discipline).
pub struct ParquetDataset {
    path: PathBuf,
    opts: DatasetOpts,
}

impl ParquetDataset {
    /// Record the source path + iteration opts; defer the reader open
    /// to a later stage.
    pub fn new(path: &Path, opts: DatasetOpts) -> Self {
        Self {
            path: path.to_path_buf(),
            opts,
        }
    }

    /// Requested batch size (accessor for the bridge surface).
    pub fn batch_size(&self) -> usize {
        self.opts.batch_size
    }

    /// Source path (accessor for the bridge surface).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Dataset for ParquetDataset {
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError> {
        Err(DatasetError::NotImplemented(format!(
            "parquet reader for {:?} — deferred to a later stage",
            self.path
        )))
    }

    fn len_hint(&self) -> Option<usize> {
        None
    }
}

/// Pad `row` up to `ctx` with `pad`, or truncate to `ctx` when longer.
fn pad_or_truncate(row: &[u32], ctx: usize, pad: u32) -> Vec<u32> {
    if row.len() >= ctx {
        row[..ctx].to_vec()
    } else {
        let mut out = Vec::with_capacity(ctx);
        out.extend_from_slice(row);
        out.resize(ctx, pad);
        out
    }
}

/// Pad `mask` up to `ctx` with `0.0` (positions past the real content
/// contribute nothing to the loss), or truncate to `ctx` when longer.
fn pad_or_truncate_mask(mask: &[f32], ctx: usize) -> Vec<f32> {
    if mask.len() >= ctx {
        mask[..ctx].to_vec()
    } else {
        let mut out = Vec::with_capacity(ctx);
        out.extend_from_slice(mask);
        out.resize(ctx, 0.0);
        out
    }
}

/// In-memory dataset for hard-label distillation.
///
/// Each row carries the concatenated `[prompt || sep || response]`
/// token ids alongside a per-token float mask picking out the
/// response region. The mask is padded / truncated in lockstep with
/// the token ids so a downstream `Loss::compute` call receives shape-
/// aligned inputs.
///
/// The trainer plumbing (bridge / Lua caller) is responsible for
/// reading the actual Card sample rows via a `FileCardStore` and
/// tokenising them before constructing this dataset — the
/// `algocline-nn` crate never touches the filesystem itself. That
/// keeps the dataset trivially testable with hand-picked deterministic
/// rows here, and lets the bridge decide the exact prompt / response
/// split rule.
pub struct TeacherCardDataset {
    rows: Vec<Vec<u32>>,
    masks: Vec<Vec<f32>>,
    opts: DatasetOpts,
    cursor: usize,
}

impl TeacherCardDataset {
    /// Construct from a paired vector of `(input_ids, loss_mask)` rows.
    ///
    /// Every mask row must have the same length as its paired
    /// `input_ids` row — otherwise the caller has broken the
    /// per-position-mask invariant and the constructor refuses rather
    /// than silently truncating.
    pub fn from_rows(
        rows: Vec<(Vec<u32>, Vec<f32>)>,
        opts: DatasetOpts,
    ) -> Result<Self, DatasetError> {
        let mut ids = Vec::with_capacity(rows.len());
        let mut masks = Vec::with_capacity(rows.len());
        for (idx, (row_ids, row_mask)) in rows.into_iter().enumerate() {
            if row_ids.len() != row_mask.len() {
                return Err(DatasetError::MissingField {
                    field: format!(
                        "loss_mask length {} != input_ids length {}",
                        row_mask.len(),
                        row_ids.len()
                    ),
                    index: idx,
                });
            }
            ids.push(row_ids);
            masks.push(row_mask);
        }
        Ok(Self {
            rows: ids,
            masks,
            opts,
            cursor: 0,
        })
    }

    /// Rows currently held by this dataset.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

impl Dataset for TeacherCardDataset {
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError> {
        if self.cursor >= self.rows.len() {
            return Ok(None);
        }
        let start = self.cursor;
        let end = (start + self.opts.batch_size).min(self.rows.len());
        self.cursor = end;
        let ctx = self.opts.ctx_len;
        let pad = self.opts.pad_id;
        let input_ids: Vec<Vec<u32>> = self.rows[start..end]
            .iter()
            .map(|row| pad_or_truncate(row, ctx, pad))
            .collect();
        let loss_mask: Vec<Vec<f32>> = self.masks[start..end]
            .iter()
            .map(|m| pad_or_truncate_mask(m, ctx))
            .collect();
        Ok(Some(Batch {
            input_ids,
            loss_mask: Some(loss_mask),
            is_last: end == self.rows.len(),
        }))
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.rows.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenized_dataset_yields_padded_batches() {
        let rows = vec![vec![1u32, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let opts = DatasetOpts {
            batch_size: 2,
            ctx_len: 4,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        };
        let mut ds = TokenizedDataset::new(rows, opts);

        let b1 = ds.next_batch().unwrap().unwrap();
        assert_eq!(b1.input_ids.len(), 2);
        assert_eq!(b1.input_ids[0], vec![1, 2, 3, 0]); // padded
        assert_eq!(b1.input_ids[1], vec![4, 5, 0, 0]); // padded
        assert!(!b1.is_last);

        let b2 = ds.next_batch().unwrap().unwrap();
        assert_eq!(b2.input_ids.len(), 1); // short last batch
        assert_eq!(b2.input_ids[0], vec![6, 7, 8, 9]); // truncated to ctx
        assert!(b2.is_last);

        assert!(ds.next_batch().unwrap().is_none());
    }

    #[test]
    fn tokenized_dataset_truncates_row_over_ctx() {
        let rows = vec![vec![1u32, 2, 3, 4, 5, 6]];
        let opts = DatasetOpts {
            batch_size: 1,
            ctx_len: 3,
            ..DatasetOpts::default()
        };
        let mut ds = TokenizedDataset::new(rows, opts);
        let b = ds.next_batch().unwrap().unwrap();
        assert_eq!(b.input_ids[0], vec![1, 2, 3]);
    }

    #[test]
    fn parquet_scaffold_errors_on_iteration() {
        let mut ds = ParquetDataset::new(
            Path::new("/does/not/matter.parquet"),
            DatasetOpts::default(),
        );
        let err = ds.next_batch().unwrap_err();
        assert!(matches!(err, DatasetError::NotImplemented(_)));
        assert!(ds.len_hint().is_none());
    }

    #[test]
    fn pad_or_truncate_pads_short_rows() {
        let padded = pad_or_truncate(&[1, 2, 3], 6, 99);
        assert_eq!(padded, vec![1, 2, 3, 99, 99, 99]);
    }

    #[test]
    fn pad_or_truncate_trims_long_rows() {
        let trimmed = pad_or_truncate(&[1, 2, 3, 4, 5], 3, 0);
        assert_eq!(trimmed, vec![1, 2, 3]);
    }
}
