//! Integration tests for the `Dataset` trait and its impls.
//!
//! These exercise the crate boundary without a network or GPU:
//!
//! - `TokenizedDataset` — in-memory rows, verify batch shape / padding /
//!   truncation / `is_last` flag / `len_hint`.
//! - `ParquetDataset` — real parquet fixtures written with the same
//!   `parquet` crate (row-API writer) + a WordLevel tokenizer fixture,
//!   covering batch shape, row-group streaming, shuffle, schema /
//!   type / IO error propagation. No network.
//! - `DatasetOpts` defaults — the Lua bridge relies on these when the
//!   caller omits fields.
//!
//! `JsonlDataset` needs a real (downloaded) tokenizer and is covered by
//! the inline unit tests plus the `#[ignore]`-gated tokenizer roundtrip;
//! this integration file stays offline.

use std::path::PathBuf;

use algocline_nn::tokenizer::HfTokenizer;
use algocline_nn::train::{
    Batch, Dataset, DatasetError, DatasetOpts, ParquetDataset, TokenizedDataset,
};

fn opts_with_batch_ctx(batch_size: usize, ctx_len: usize) -> DatasetOpts {
    DatasetOpts {
        batch_size,
        ctx_len,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    }
}

#[test]
fn tokenized_dataset_produces_padded_first_batch() {
    let rows = vec![vec![1u32, 2, 3], vec![4, 5]];
    let mut ds = TokenizedDataset::new(rows, opts_with_batch_ctx(2, 4));
    let batch = ds
        .next_batch()
        .expect("next_batch must not error")
        .expect("first batch must be present");
    assert_eq!(batch.input_ids.len(), 2);
    assert_eq!(batch.input_ids[0], vec![1, 2, 3, 0]);
    assert_eq!(batch.input_ids[1], vec![4, 5, 0, 0]);
    assert!(batch.is_last, "two rows exactly fill one batch of two");
}

#[test]
fn tokenized_dataset_marks_short_final_batch() {
    let rows = vec![vec![1u32], vec![2], vec![3]];
    let mut ds = TokenizedDataset::new(rows, opts_with_batch_ctx(2, 2));
    let first: Batch = ds.next_batch().unwrap().unwrap();
    assert!(!first.is_last, "first batch should not be marked last");
    assert_eq!(first.input_ids.len(), 2);

    let last: Batch = ds.next_batch().unwrap().unwrap();
    assert!(last.is_last, "third row should be flagged is_last");
    assert_eq!(last.input_ids.len(), 1);
    assert_eq!(last.input_ids[0], vec![3, 0]);

    assert!(
        ds.next_batch().unwrap().is_none(),
        "drained dataset yields None"
    );
}

#[test]
fn tokenized_dataset_truncates_row_over_ctx() {
    let rows = vec![vec![1u32, 2, 3, 4, 5]];
    let mut ds = TokenizedDataset::new(rows, opts_with_batch_ctx(1, 3));
    let batch = ds.next_batch().unwrap().unwrap();
    assert_eq!(batch.input_ids[0], vec![1, 2, 3]);
}

#[test]
fn tokenized_dataset_len_hint_matches_row_count() {
    let rows = vec![vec![1u32], vec![2], vec![3], vec![4]];
    let ds = TokenizedDataset::new(rows, DatasetOpts::default());
    assert_eq!(ds.len_hint(), Some(4));
}

// ─── ParquetDataset (offline fixtures) ────────────────────────────
//
// A real parquet file is written with the same `parquet` crate the
// reader uses (row-API writer, no arrow), and tokenization goes
// through a hand-written WordLevel tokenizer fixture so the whole
// read → tokenize → batch path runs without network.

/// Minimal WordLevel tokenizer: "w0".."w9" map to ids 0..9,
/// whitespace pre-tokenization. Loaded via the same
/// `HfTokenizer::load_from_file` entry the fixture-driven E2E uses.
fn fixture_tokenizer(dir: &std::path::Path) -> HfTokenizer {
    let mut vocab = String::new();
    for i in 0..10 {
        vocab.push_str(&format!("\"w{i}\": {i}, "));
    }
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "WordLevel",
                "vocab": {{ {vocab} "[UNK]": 10 }},
                "unk_token": "[UNK]"
            }}
        }}"#
    );
    let path = dir.join("wordlevel.json");
    std::fs::write(&path, json).expect("write tokenizer fixture");
    HfTokenizer::load_from_file("wordlevel", &path).expect("load tokenizer fixture")
}

/// Write `row_groups` of UTF-8 `texts` into a single-column parquet
/// file (`column_name` as `required binary ... (UTF8)`), one row group
/// per outer slice. Uncompressed, so no codec feature is exercised.
fn write_parquet_fixture(path: &std::path::Path, column_name: &str, row_groups: &[&[&str]]) {
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::sync::Arc;

    let schema = parse_message_type(&format!(
        "message schema {{ required binary {column_name} (UTF8); }}"
    ))
    .expect("parse fixture schema");
    let file = std::fs::File::create(path).expect("create fixture file");
    let mut writer =
        SerializedFileWriter::new(file, Arc::new(schema), Arc::new(WriterProperties::new()))
            .expect("open fixture writer");
    for group in row_groups {
        let mut rg = writer.next_row_group().expect("next_row_group");
        let mut col = rg
            .next_column()
            .expect("next_column")
            .expect("one column expected");
        let values: Vec<ByteArray> = group.iter().map(|s| ByteArray::from(*s)).collect();
        col.typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .expect("write_batch");
        col.close().expect("close column");
        rg.close().expect("close row group");
    }
    writer.close().expect("close writer");
}

#[test]
fn parquet_dataset_reads_tokenizes_and_batches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    write_parquet_fixture(&path, "text", &[&["w1 w2 w3", "w4 w5", "w6 w7 w8 w9"]]);
    let tok = fixture_tokenizer(dir.path());

    let mut ds =
        ParquetDataset::new(&path, opts_with_batch_ctx(2, 4), tok).expect("open parquet dataset");
    assert_eq!(ds.len_hint(), Some(3), "row count comes from the footer");

    let b1 = ds.next_batch().unwrap().expect("first batch");
    assert_eq!(b1.input_ids.len(), 2);
    assert_eq!(b1.input_ids[0], vec![1, 2, 3, 0]); // padded
    assert_eq!(b1.input_ids[1], vec![4, 5, 0, 0]); // padded
    assert!(!b1.is_last);

    let b2 = ds.next_batch().unwrap().expect("second batch");
    assert_eq!(b2.input_ids.len(), 1); // short last batch
    assert_eq!(b2.input_ids[0], vec![6, 7, 8, 9]);
    assert!(b2.is_last);

    assert!(ds.next_batch().unwrap().is_none());
}

#[test]
fn parquet_dataset_streams_across_row_groups() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    // A batch of 2 must span the row-group boundary transparently.
    write_parquet_fixture(&path, "text", &[&["w0"], &["w1", "w2"]]);
    let tok = fixture_tokenizer(dir.path());

    let mut ds =
        ParquetDataset::new(&path, opts_with_batch_ctx(2, 2), tok).expect("open parquet dataset");
    let b1 = ds.next_batch().unwrap().expect("first batch");
    assert_eq!(b1.input_ids, vec![vec![0, 0], vec![1, 0]]);
    let b2 = ds.next_batch().unwrap().expect("second batch");
    assert_eq!(b2.input_ids, vec![vec![2, 0]]);
    assert!(b2.is_last);
}

#[test]
fn parquet_dataset_shuffle_materializes_all_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    write_parquet_fixture(&path, "text", &[&["w1", "w2", "w3"]]);
    let tok = fixture_tokenizer(dir.path());

    let opts = DatasetOpts {
        shuffle: true,
        ..opts_with_batch_ctx(3, 1)
    };
    let mut ds = ParquetDataset::new(&path, opts, tok).expect("open parquet dataset");
    let batch = ds.next_batch().unwrap().expect("single batch");
    // Deterministic reverse-order "shuffle" placeholder (same as the
    // JSONL adapter).
    assert_eq!(batch.input_ids, vec![vec![3], vec![2], vec![1]]);
    assert!(batch.is_last);
}

#[test]
fn parquet_dataset_rejects_missing_text_field_at_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    write_parquet_fixture(&path, "content", &[&["w1"]]);
    let tok = fixture_tokenizer(dir.path());

    let err = ParquetDataset::new(&path, opts_with_batch_ctx(1, 2), tok)
        .expect_err("wrong text_field must fail at construction");
    match err {
        DatasetError::Parquet(msg) => {
            assert!(
                msg.contains("text field 'text' not found") && msg.contains("content"),
                "message should name the missing field and the available ones; got: {msg}"
            );
        }
        other => panic!("expected Parquet error, got {other:?}"),
    }
}

#[test]
fn parquet_dataset_rejects_non_string_text_column() {
    use parquet::data_type::Int64Type;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    let schema =
        parse_message_type("message schema { required int64 text; }").expect("parse schema");
    let file = std::fs::File::create(&path).expect("create fixture file");
    let mut writer =
        SerializedFileWriter::new(file, Arc::new(schema), Arc::new(WriterProperties::new()))
            .expect("open writer");
    let mut rg = writer.next_row_group().expect("next_row_group");
    let mut col = rg.next_column().expect("next_column").expect("one column");
    col.typed::<Int64Type>()
        .write_batch(&[42i64], None, None)
        .expect("write_batch");
    col.close().expect("close column");
    rg.close().expect("close row group");
    writer.close().expect("close writer");

    let tok = fixture_tokenizer(dir.path());
    let mut ds =
        ParquetDataset::new(&path, opts_with_batch_ctx(1, 2), tok).expect("schema has 'text'");
    let err = ds
        .next_batch()
        .expect_err("non-string text column must error, not silently skip");
    match err {
        DatasetError::Parquet(msg) => {
            assert!(
                msg.contains("not a UTF-8 string") && msg.contains("int"),
                "message should report the column kind; got: {msg}"
            );
        }
        other => panic!("expected Parquet error, got {other:?}"),
    }
}

#[test]
fn parquet_dataset_missing_file_is_io_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tok = fixture_tokenizer(dir.path());
    let err = ParquetDataset::new(
        &PathBuf::from("/nowhere/data.parquet"),
        DatasetOpts::default(),
        tok,
    )
    .expect_err("missing file must fail at construction");
    assert!(matches!(err, DatasetError::Io(_)), "got {err:?}");
}

#[test]
fn dataset_opts_default_values_match_bridge_contract() {
    let opts = DatasetOpts::default();
    // Lua bridge relies on these when the caller omits fields.
    assert_eq!(opts.batch_size, 8);
    assert_eq!(opts.ctx_len, 128);
    assert!(!opts.shuffle);
    assert_eq!(opts.pad_id, 0);
    assert_eq!(opts.text_field, "text");
}
