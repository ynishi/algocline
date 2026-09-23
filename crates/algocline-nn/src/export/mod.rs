//! Exporting a Card in the vocabulary other tools read.
//!
//! A model trained here is described by its Card, which lives in a card
//! store that does not travel with the weights. The readers outside
//! algocline each look in their own place: the Hugging Face Hub reads a
//! model card's YAML front matter, GGUF runtimes read `general.*`, and
//! a safetensors reader has `__metadata__` — whose only shared key is
//! `format`. This module maps one Card onto all of them
//! ([`CardExport`], table in [`mapping`]) and writes the weights with a
//! byte-stable header that carries a digest of the tensors
//! ([`rewrite_with_metadata`]).
//!
//! The engine's `alc.nn.card.export` drives both; nothing here touches
//! a card store.

mod bundle;
pub mod mapping;

pub use bundle::{publish_new, rewrite_with_metadata, temp_path_for, TENSOR_SHA256_KEY};
pub use mapping::{base_model_relation, is_hub_repo_id, CardExport, TAGS};
