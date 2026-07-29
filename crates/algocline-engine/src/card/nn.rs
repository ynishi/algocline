//! `nn_model` Card persistence — the single projection point from the
//! typed [`NnModelCard`] aggregate onto the generic card store.
//!
//! The Lua bridge (`bridge/nn_card.rs` / `bridge/nn_trainer.rs`)
//! constructs an [`NnModelCard`] (whose invariants — architecture /
//! training_path validity and `bundle_ref == "nn/<id>"` — are enforced
//! at construction) and hands it here. This module owns the Card
//! envelope shape (`pkg` / `card_id` / `metadata.kind` /
//! `metadata.nn`) and the create + returned-id coherence check; no
//! other code path may assemble an `nn_model` create payload.

use serde_json::json;

use algocline_nn::card::NnModelCard;

use super::FileCardStore;

/// Card package name under which `nn_model` Cards are stored
/// (`~/.algocline/cards/alc_nn/...`).
pub const NN_PKG: &str = "alc_nn";

/// Persist `card` into `store` and return the store-confirmed id.
///
/// Errors carry no surface prefix — callers (the Lua bridge) wrap the
/// message under their own `alc.nn.*` prefix so the loud-error
/// contract per surface is preserved.
pub fn persist(store: &FileCardStore, card: &NnModelCard) -> Result<String, String> {
    let nn_meta_json =
        serde_json::to_value(card.meta()).map_err(|e| format!("serialize meta: {e}"))?;

    let card_id = card.id().as_str();
    let payload = json!({
        "pkg": { "name": NN_PKG },
        "card_id": card_id,
        "metadata": {
            "kind": "nn_model",
            "nn": nn_meta_json,
        }
    });

    let (returned_id, _path) = store
        .create(payload)
        .map_err(|e| format!("card store: {e}"))?;

    // The store must echo the aggregate's id back. A divergence would
    // silently break the safetensors ↔ Card 1:1 mapping — surface it.
    if returned_id != card_id {
        return Err(format!(
            "card_id mismatch (expected {card_id}, got {returned_id})"
        ));
    }
    Ok(returned_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use algocline_nn::card::{CardId, NnModelCard, TrainingPath};
    use algocline_nn::train::{Checkpoint, FullFtConfig};

    fn test_card(name: &str) -> NnModelCard {
        let ckpt = Checkpoint {
            bundle_ref: "x.safetensors".into(),
            step: 3,
            train_loss: 0.5,
            val_loss: None,
            metrics: std::collections::HashMap::new(),
        };
        NnModelCard::from_training(
            CardId::mint(name),
            name,
            "gpt2-medium".into(),
            TrainingPath::FullFt,
            &ckpt,
            &FullFtConfig::default(),
        )
        .expect("test card")
    }

    #[test]
    fn persist_writes_envelope_and_returns_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileCardStore::new(dir.path().to_path_buf());
        let card = test_card("persist-me");

        let id = persist(&store, &card).expect("persist");
        assert_eq!(id, card.id().as_str());

        let stored = store
            .get(&id)
            .expect("get")
            .expect("card present after persist");
        assert_eq!(stored["pkg"]["name"], NN_PKG);
        assert_eq!(stored["metadata"]["kind"], "nn_model");
        assert_eq!(
            stored["metadata"]["nn"]["candle"]["bundle_ref"],
            card.id().bundle_ref()
        );
    }

    #[test]
    fn persist_is_refused_for_duplicate_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileCardStore::new(dir.path().to_path_buf());
        let card = test_card("dup");

        persist(&store, &card).expect("first persist");
        let err = persist(&store, &card).expect_err("second persist must fail (immutable cards)");
        assert!(err.contains("card store:"), "got: {err}");
    }

    #[test]
    fn card_id_is_a_valid_store_key_by_construction() {
        // CardId::mint sanitizes; the store must accept every minted id.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileCardStore::new(dir.path().to_path_buf());
        let card = test_card("weird name/with:chars!");
        persist(&store, &card).expect("sanitized id accepted by store");
        let _ = CardId::parse(card.id().as_str()).expect("round-trips");
    }
}
