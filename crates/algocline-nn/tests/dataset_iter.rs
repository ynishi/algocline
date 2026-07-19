//! Integration tests for the `Dataset` trait and its impls.
//!
//! These exercise the crate boundary without a network or GPU:
//!
//! - `TokenizedDataset` — in-memory rows, verify batch shape / padding /
//!   truncation / `is_last` flag / `len_hint`.
//! - `ParquetDataset` — scaffold surface must surface
//!   `DatasetError::NotImplemented` rather than silently returning
//!   `None`, per algocline's Service-layer error propagation discipline.
//! - `DatasetOpts` defaults — the Lua bridge relies on these when the
//!   caller omits fields.
//!
//! `JsonlDataset` needs a real tokenizer (network) and is covered by
//! the inline unit tests plus the `#[ignore]`-gated tokenizer roundtrip;
//! this integration file stays offline.

use std::path::PathBuf;

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

#[test]
fn parquet_scaffold_errors_on_iteration_and_reports_no_len_hint() {
    let mut ds = ParquetDataset::new(
        &PathBuf::from("/nowhere/data.parquet"),
        DatasetOpts::default(),
    );
    let err = ds
        .next_batch()
        .expect_err("parquet scaffold must not silently return None");
    match err {
        DatasetError::NotImplemented(msg) => {
            assert!(
                msg.contains("parquet"),
                "message should mention parquet; got: {msg}"
            );
            assert!(
                msg.contains("deferred to a later stage"),
                "message should note the reader is deferred; got: {msg}"
            );
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    assert!(ds.len_hint().is_none());
    assert_eq!(ds.batch_size(), DatasetOpts::default().batch_size);
    assert_eq!(ds.path(), PathBuf::from("/nowhere/data.parquet"));
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
