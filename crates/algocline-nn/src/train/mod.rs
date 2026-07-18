//! Training-side scaffolding.
//!
//! This stage ships the data-loader abstraction only ([`data`]).
//! Optimizer / schedule / gradient-accumulation helpers land in a
//! follow-up alongside the Full FT loop, and LoRA-specific plumbing
//! comes with the LoRA follow-up. Keeping this module tree in place
//! now avoids reshaping `lib.rs` when those additions land.

pub mod data;

pub use data::{
    Batch, Dataset, DatasetError, DatasetOpts, JsonlDataset, ParquetDataset, TokenizedDataset,
};
