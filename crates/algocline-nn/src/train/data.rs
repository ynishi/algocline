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
//! - Parquet adapter (see [`ParquetDataset`]) reads the column named
//!   [`DatasetOpts::text_field`] through the `parquet` crate's row API
//!   (no arrow dependency) and tokenizes it exactly like the JSONL
//!   adapter — same field-name convention, same shuffle semantics.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::arch::CondIndex;
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
    /// Parquet schema / decode failure on a Parquet source (open,
    /// footer parse, row decode, missing or non-string text column).
    #[error("parquet: {0}")]
    Parquet(String),
    /// A teacher row's loss mask retains no scored position once the
    /// row is truncated to `ctx_len` and shifted against the targets
    /// (mask position 0 gates no target token). Training on such a row
    /// is a silent no-op: the loss is exactly 0.0, no gradient flows,
    /// and `min_train_loss` records 0.0 as if learning had completed.
    #[error(
        "row {index}: loss_mask has no scored position within ctx_len {ctx_len} \
         (mask length {row_len}) — the response region was fully truncated or \
         the mask is all-zero"
    )]
    FullyMaskedRow {
        /// 0-based row index within the input rows.
        index: usize,
        /// Configured sequence length the row is truncated to.
        ctx_len: usize,
        /// Untruncated mask length of the offending row.
        row_len: usize,
    },
    /// A per-row conditioning list was attached to a dataset holding a
    /// different number of rows.
    ///
    /// Refused rather than zipped to the shorter of the two: the pairing
    /// is positional, so a length disagreement means some row would be
    /// conditioned on another row's band, and the batch shapes would
    /// still line up.
    #[error(
        "{conds} condition(s) for {rows} row(s) at {per_row} per row — the pairing is \
         positional, so it takes exactly rows × per-row"
    )]
    ConditionCountMismatch {
        /// Conditions handed over.
        conds: usize,
        /// Rows the dataset holds.
        rows: usize,
        /// Conditions each row carries.
        per_row: usize,
    },
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
    /// Per-position sets of token ids the target is allowed to take,
    /// indexed `[row][position]`.
    ///
    /// `None` (every dataset that does not model a constrained action
    /// space) leaves the loss ranging over the whole vocabulary.
    /// `Some` restricts it: the training loop drives every id outside
    /// the set to negative infinity before the softmax, so the model is
    /// scored only on choosing among the moves that exist.
    ///
    /// This matters wherever decoding already enforces the constraint.
    /// A chess model measured here spent 1.59 of its 4.52 nats keeping
    /// mass off illegal moves — a third of the objective — and the
    /// decoder throws that work away, because it walks the ranking
    /// against the legal set no matter what the model believed.
    #[allow(clippy::type_complexity)]
    pub allowed_ids: Option<Vec<Vec<Vec<u32>>>>,
    /// Which condition each row of this batch was recorded under, one
    /// entry per row of `input_ids`.
    ///
    /// `None` for every dataset that models no condition, which leaves
    /// the training loop driving the model through `Module::forward`
    /// exactly as before. `Some` is what
    /// [`crate::train::run_conditioned_ft`] hands to
    /// [`crate::train::ConditionedForward`].
    ///
    /// The two entry points refuse the pairing they cannot serve rather
    /// than ignoring this field: `Some` at [`crate::train::run_full_ft`]
    /// is [`crate::train::TrainError::UnexpectedConditions`], and `None`
    /// at `run_conditioned_ft` is `MissingConditions`. A run that
    /// silently dropped a condition attached per row, or silently
    /// trained without one, would still write a checkpoint labelled the
    /// way the caller meant it.
    ///
    /// Per row rather than per batch because a batch mixes conditions:
    /// the corpus is shuffled, so consecutive rows are unrelated games
    /// and one index for the batch would condition most of them on some
    /// other row's band.
    ///
    /// A [`CondIndex`] rather than the condition's token id. The two
    /// numberings overlap — band tokens start at vocabulary id 2 while
    /// the conditioning table's rows start at 0 — so an id passed here
    /// would select a real but wrong row, and nothing downstream could
    /// report it. The type keeps that substitution from being
    /// expressible; producing one goes through whatever holds the
    /// mapping, which for the chess models is
    /// [`crate::chess::ModelShape::band_index`].
    pub conds: Option<Vec<CondIndex>>,
    /// How many of [`Self::conds`]'s entries belong to each row —
    /// `conds` is row-major, `conds_per_row` entries per row of
    /// `input_ids`, the layout
    /// [`crate::arch::Gpt2Model::forward_conditioned_groups`] reads.
    ///
    /// `1` wherever `conds` is `None` or one-per-row, so every dataset
    /// that predates condition groups keeps its meaning without
    /// stating anything.
    pub conds_per_row: usize,
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
            allowed_ids: None,
            conds: None,
            conds_per_row: 1,
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
                allowed_ids: None,
                conds: None,
                conds_per_row: 1,
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
            allowed_ids: None,
            conds: None,
            conds_per_row: 1,
        }))
    }

    fn len_hint(&self) -> Option<usize> {
        self.total_rows
    }
}

/// Parquet-backed dataset.
///
/// Reads the column named [`DatasetOpts::text_field`] via the
/// `parquet` crate's row API (`SerializedFileReader` → `RowIter`, no
/// arrow dependency) and tokenizes it like [`JsonlDataset`]. Rows are
/// streamed row-group by row-group unless [`DatasetOpts::shuffle`] is
/// set, in which case all rows are materialised first (same semantics
/// as the JSONL adapter, including its deterministic reverse-order
/// shuffle placeholder). The text column must exist at the top level
/// of the file schema — that is checked at construction so a wrong
/// `text_field` fails before the first training step, not mid-epoch.
pub struct ParquetDataset {
    rows_iter: Option<parquet::record::reader::RowIter<'static>>,
    buffer: Vec<Vec<u32>>,
    buffer_cursor: usize,
    opts: DatasetOpts,
    tokenizer: HfTokenizer,
    path: PathBuf,
    row_index: usize,
    total_rows: usize,
}

impl std::fmt::Debug for ParquetDataset {
    // Manual impl: `RowIter` / `HfTokenizer` carry no `Debug`, and the
    // useful state for a failure message is the source + progress.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetDataset")
            .field("path", &self.path)
            .field("total_rows", &self.total_rows)
            .field("row_index", &self.row_index)
            .field("shuffle", &self.opts.shuffle)
            .finish_non_exhaustive()
    }
}

impl ParquetDataset {
    /// Open `path`, verify the schema carries `opts.text_field` as a
    /// top-level field, and set up row iteration. When `opts.shuffle`
    /// is `true` the full file is parsed and tokenized eagerly so rows
    /// can be re-ordered; otherwise the row iterator is retained and
    /// batches stream.
    pub fn new(
        path: &Path,
        opts: DatasetOpts,
        tokenizer: HfTokenizer,
    ) -> Result<Self, DatasetError> {
        use parquet::file::reader::FileReader;

        let file =
            File::open(path).map_err(|e| DatasetError::Io(format!("open {:?}: {e}", path)))?;
        let reader = parquet::file::serialized_reader::SerializedFileReader::new(file)
            .map_err(|e| DatasetError::Parquet(format!("open {:?}: {e}", path)))?;

        let file_meta = reader.metadata().file_metadata();
        let total_rows = usize::try_from(file_meta.num_rows()).unwrap_or(0);
        let schema = file_meta.schema();
        if !schema
            .get_fields()
            .iter()
            .any(|f| f.name() == opts.text_field)
        {
            let available: Vec<&str> = schema.get_fields().iter().map(|f| f.name()).collect();
            return Err(DatasetError::Parquet(format!(
                "text field '{}' not found in the schema of {:?} (top-level fields: {})",
                opts.text_field,
                path,
                available.join(", ")
            )));
        }

        let rows_iter = parquet::record::reader::RowIter::from_file_into(Box::new(reader));
        let mut this = Self {
            rows_iter: Some(rows_iter),
            buffer: Vec::new(),
            buffer_cursor: 0,
            opts,
            tokenizer,
            path: path.to_path_buf(),
            row_index: 0,
            total_rows,
        };
        if this.opts.shuffle {
            this.materialize_all()?;
        }
        Ok(this)
    }

    fn materialize_all(&mut self) -> Result<(), DatasetError> {
        let Some(iter) = self.rows_iter.take() else {
            return Ok(());
        };
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for (idx, row) in iter.enumerate() {
            let row = row.map_err(|e| {
                DatasetError::Parquet(format!("record {idx} in {:?}: {e}", self.path))
            })?;
            let text = row_text(&row, &self.opts.text_field, idx)?;
            rows.push(self.tokenizer.encode(text)?);
        }
        // Reverse ordering as a deterministic "shuffle" placeholder;
        // a later stage wires a real RNG seed once the trainer opts
        // land (same placeholder as `JsonlDataset::materialize_all`).
        rows.reverse();
        self.total_rows = rows.len();
        self.buffer = rows;
        self.buffer_cursor = 0;
        Ok(())
    }

    fn fill_batch_streaming(&mut self) -> Result<Vec<Vec<u32>>, DatasetError> {
        let mut rows = Vec::with_capacity(self.opts.batch_size);
        let Some(iter) = self.rows_iter.as_mut() else {
            return Ok(rows);
        };
        while rows.len() < self.opts.batch_size {
            match iter.next() {
                None => break,
                Some(Err(e)) => {
                    return Err(DatasetError::Parquet(format!(
                        "record {} in {:?}: {e}",
                        self.row_index, self.path
                    )))
                }
                Some(Ok(row)) => {
                    let idx = self.row_index;
                    self.row_index += 1;
                    let text = row_text(&row, &self.opts.text_field, idx)?;
                    rows.push(self.tokenizer.encode(text)?);
                }
            }
        }
        Ok(rows)
    }
}

/// Extract the text column value from a decoded row. Field lookup is
/// by name (row column order follows the file schema, not
/// `DatasetOpts`); a present-but-non-string value is a schema error,
/// reported by kind so a multi-megabyte binary cell never lands in the
/// error string.
fn row_text<'a>(
    row: &'a parquet::record::Row,
    text_field: &str,
    idx: usize,
) -> Result<&'a str, DatasetError> {
    for (name, field) in row.get_column_iter() {
        if name == text_field {
            return match field {
                parquet::record::Field::Str(s) => Ok(s.as_str()),
                parquet::record::Field::Null => Err(DatasetError::Parquet(format!(
                    "record {idx}: column '{text_field}' is null"
                ))),
                other => Err(DatasetError::Parquet(format!(
                    "record {idx}: column '{text_field}' is not a UTF-8 string \
                     (found {})",
                    field_kind(other)
                ))),
            };
        }
    }
    // The constructor verified the schema, so a decoded row missing the
    // column indicates file corruption rather than caller error.
    Err(DatasetError::MissingField {
        field: text_field.to_string(),
        index: idx,
    })
}

/// Short kind label for a decoded Parquet field, for error messages.
fn field_kind(field: &parquet::record::Field) -> &'static str {
    use parquet::record::Field;
    match field {
        Field::Null => "null",
        Field::Bool(_) => "bool",
        Field::Byte(_) | Field::Short(_) | Field::Int(_) | Field::Long(_) => "int",
        Field::UByte(_) | Field::UShort(_) | Field::UInt(_) | Field::ULong(_) => "uint",
        Field::Float16(_) | Field::Float(_) | Field::Double(_) => "float",
        Field::Decimal(_) => "decimal",
        Field::Str(_) => "string",
        Field::Bytes(_) => "bytes",
        Field::Date(_)
        | Field::TimeMillis(_)
        | Field::TimeMicros(_)
        | Field::TimestampMillis(_)
        | Field::TimestampMicros(_) => "timestamp",
        Field::Group(_) => "group",
        Field::ListInternal(_) => "list",
        Field::MapInternal(_) => "map",
    }
}

impl Dataset for ParquetDataset {
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
                allowed_ids: None,
                conds: None,
                conds_per_row: 1,
            }));
        }

        // Streaming path — pull the next `batch_size` rows.
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
            allowed_ids: None,
            conds: None,
            conds_per_row: 1,
        }))
    }

    fn len_hint(&self) -> Option<usize> {
        // Row count comes from the file footer at open time, so it is
        // known even on the streaming path (unlike JSONL).
        Some(self.total_rows)
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
///
/// Truncation here can only trim a *partially* scored tail:
/// [`TeacherCardDataset::from_rows`] has already refused any row whose
/// mask would come out of this truncation with no scored position left.
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
#[derive(Debug)]
pub struct TeacherCardDataset {
    rows: Vec<Vec<u32>>,
    masks: Vec<Vec<f32>>,
    /// Conditions row-major at [`Self::conds_per_row`] per row, or
    /// `None` for an unconditioned dataset. See
    /// [`TeacherCardDataset::with_conditions`] and
    /// [`TeacherCardDataset::with_condition_groups`].
    conds: Option<Vec<CondIndex>>,
    conds_per_row: usize,
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
    ///
    /// Additionally, every row must keep at least one scored (non-zero)
    /// mask position after truncation to `opts.ctx_len` and the training
    /// loop's input/target shift (mask position 0 gates no target).
    /// A row failing this — a response fully cut off by `ctx_len`, or an
    /// all-zero mask — surfaces as [`DatasetError::FullyMaskedRow`]
    /// instead of training as a silent zero-loss no-op step.
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
            // Refuse rows that would train as a silent no-op. `next_batch`
            // truncates to `ctx_len` and the loop's input/target shift drops
            // mask position 0 (it gates no target token); if the surviving
            // mask is all-zero the loss is exactly 0.0, no gradient flows,
            // and the step still advances — `min_train_loss` would record
            // 0.0 as if the model had learned perfectly.
            let scored = row_mask
                .iter()
                .take(opts.ctx_len)
                .skip(1)
                .filter(|m| **m != 0.0)
                .count();
            if scored == 0 {
                return Err(DatasetError::FullyMaskedRow {
                    index: idx,
                    ctx_len: opts.ctx_len,
                    row_len: row_mask.len(),
                });
            }
            ids.push(row_ids);
            masks.push(row_mask);
        }
        Ok(Self {
            rows: ids,
            masks,
            conds: None,
            conds_per_row: 1,
            opts,
            cursor: 0,
        })
    }

    /// Attach the condition each row was recorded under, in row order.
    ///
    /// Every batch then carries the slice of these belonging to its own
    /// rows, which is what [`crate::train::run_conditioned_ft`] passes
    /// to the model.
    ///
    /// Applied to the built dataset rather than taken by
    /// [`Self::from_rows`] so the unconditioned constructor keeps its
    /// signature — most callers have no condition to give. The pairing
    /// is positional and stays that way: this dataset walks its rows in
    /// the order it was handed them and never reorders, so a caller
    /// that shuffles has to shuffle the row and its condition as one
    /// unit before either reaches here.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ConditionCountMismatch`] when the list is not
    /// exactly one entry per row.
    pub fn with_conditions(self, conds: Vec<CondIndex>) -> Result<Self, DatasetError> {
        self.with_condition_groups(conds, 1)
    }

    /// [`Self::with_conditions`] with `per_row` conditions per row,
    /// row-major — the layout
    /// [`crate::chess::train::row_conditions`] emits for a multi-slot
    /// corpus and
    /// [`crate::arch::Gpt2Model::forward_conditioned_groups`] reads.
    ///
    /// Explicit rather than inferred from the list's length: a count
    /// that happens to divide evenly is exactly the mistake an
    /// inference would wave through.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ConditionCountMismatch`] when the list is not
    /// exactly `per_row` entries per row, `per_row` of zero included.
    pub fn with_condition_groups(
        mut self,
        conds: Vec<CondIndex>,
        per_row: usize,
    ) -> Result<Self, DatasetError> {
        if per_row == 0 || conds.len() != self.rows.len() * per_row {
            return Err(DatasetError::ConditionCountMismatch {
                conds: conds.len(),
                rows: self.rows.len(),
                per_row,
            });
        }
        self.conds = Some(conds);
        self.conds_per_row = per_row;
        Ok(self)
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
            allowed_ids: None,
            // The same `start..end` the rows were taken with, scaled by
            // the per-row count, so a row and its conditions cannot
            // come apart here.
            conds: self
                .conds
                .as_ref()
                .map(|c| c[start * self.conds_per_row..end * self.conds_per_row].to_vec()),
            conds_per_row: self.conds_per_row,
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

    // ParquetDataset coverage lives in `tests/dataset_iter.rs`, which
    // writes real parquet fixtures + a WordLevel tokenizer fixture and
    // exercises the full read → tokenize → batch path offline.

    fn teacher_opts(ctx_len: usize) -> DatasetOpts {
        DatasetOpts {
            batch_size: 1,
            ctx_len,
            ..DatasetOpts::default()
        }
    }

    #[test]
    fn teacher_rows_reject_mask_fully_truncated_by_ctx() {
        // The response region (mask = 1) lives at positions 6..8;
        // ctx_len = 4 cuts it off entirely, leaving a row that would
        // train as a silent zero-loss no-op.
        let rows = vec![(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8],
            vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        )];
        let err = TeacherCardDataset::from_rows(rows, teacher_opts(4)).unwrap_err();
        assert!(
            matches!(
                err,
                DatasetError::FullyMaskedRow {
                    index: 0,
                    ctx_len: 4,
                    row_len: 8
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn teacher_rows_reject_all_zero_mask() {
        let rows = vec![(vec![1u32, 2, 3, 4], vec![0.0f32, 0.0, 0.0, 0.0])];
        let err = TeacherCardDataset::from_rows(rows, teacher_opts(4)).unwrap_err();
        assert!(
            matches!(err, DatasetError::FullyMaskedRow { index: 0, .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn teacher_rows_reject_mask_scored_only_at_position_zero() {
        // Mask position 0 gates no target after the input/target shift
        // (`batch_to_input_target` drops it), so a row scored only there
        // is a no-op too.
        let rows = vec![(vec![1u32, 2, 3, 4], vec![1.0f32, 0.0, 0.0, 0.0])];
        let err = TeacherCardDataset::from_rows(rows, teacher_opts(4)).unwrap_err();
        assert!(
            matches!(err, DatasetError::FullyMaskedRow { index: 0, .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn teacher_rows_accept_partially_truncated_response() {
        // ctx_len = 5 keeps scored positions 3..5; trimming the rest of
        // the response tail is allowed as long as at least one scored
        // position survives, and the emitted batch mask reflects the
        // lockstep truncation.
        let rows = vec![(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8],
            vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )];
        let mut ds = TeacherCardDataset::from_rows(rows, teacher_opts(5)).expect("row accepted");
        let batch = ds.next_batch().unwrap().unwrap();
        assert_eq!(batch.input_ids, vec![vec![1, 2, 3, 4, 5]]);
        assert_eq!(batch.loss_mask, Some(vec![vec![0.0, 0.0, 0.0, 1.0, 1.0]]));
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
