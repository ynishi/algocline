//! Integration test — Layer 4a Q0 pattern contract for merged
//! provenance (Model-side struct + `to_card_meta` projection).
//!
//! This file guards two properties:
//!
//! 1. The [`MergedProvenance`] → [`NnCardMeta`] projection uses only
//!    existing Card schema fields — no new struct / no new field on
//!    `NnCardMeta` / `NnCandleBranch` / `NnLineage`. If a future
//!    change adds a `merged` sub-branch or a new lineage slot to
//!    absorb provenance, the SoT stops being the Model-side struct
//!    and these tests will catch the regression.
//!
//! 2. The projection maps into the specific fields the design
//!    committed to (lineage.parent for `lora_card`, architecture
//!    for `arch`, candle.bundle_ref for `bundle_ref`,
//!    training_path = "merged"). A silent field-mapping change
//!    would break downstream consumers filtering by these paths.

use algocline_nn::card::{
    validate_training_path, NnCardMeta, SUPPORTED_TRAINING_PATHS,
};
use algocline_nn::merged::MergedProvenance;

fn sample_provenance() -> MergedProvenance {
    MergedProvenance {
        lora_card: "cards/domain-lora-042".into(),
        arch: "tinyllama-1.1b".into(),
        bundle_ref: "nn/merged-001".into(),
    }
}

/// The projected card has `lineage.parent == Some(lora_card)`,
/// `architecture == arch`, `candle.bundle_ref == bundle_ref`,
/// and `training_path == "merged"`.
#[test]
fn to_card_meta_maps_lora_card_to_lineage_parent() {
    let provenance = sample_provenance();
    let card = provenance.to_card_meta("merged-001".into());

    assert_eq!(card.name, "merged-001");
    assert_eq!(card.architecture, "tinyllama-1.1b");
    assert_eq!(card.training_path, "merged");
    assert_eq!(
        card.lineage.parent.as_deref(),
        Some("cards/domain-lora-042")
    );
    let candle = card.candle.as_ref().expect("merged card carries candle");
    assert_eq!(candle.bundle_ref, "nn/merged-001");
}

/// A merged card does not populate the `teacher` / `training_data` /
/// `tokenizer` lineage slots and does not carry a `lora` sub-branch.
/// The absence is deliberate: a wide 4b load path recognises
/// merged-vs-lora by `training_path` + `candle.lora.is_none()`.
#[test]
fn to_card_meta_leaves_lora_and_teacher_lineage_slots_empty() {
    let provenance = sample_provenance();
    let card = provenance.to_card_meta("merged-001".into());

    assert!(card.lineage.teacher.is_none());
    assert!(card.lineage.training_data.is_none());
    assert!(card.lineage.tokenizer.is_none());

    let candle = card.candle.as_ref().expect("candle present");
    assert!(
        candle.lora.is_none(),
        "merged card must not carry a LoRA sub-branch"
    );
}

/// The projected card round-trips through the same
/// `serde_json::from_str::<NnCardMeta>` deserialiser that
/// pre-Layer-4 callers use — proof that no Card schema struct was
/// extended (a struct addition would show up as a required field or
/// a new variant that the pre-Layer-4 deserialiser could not accept
/// without `#[serde(default)]` coverage).
#[test]
fn merged_card_json_roundtrip_uses_existing_schema() {
    let provenance = sample_provenance();
    let card = provenance.to_card_meta("merged-001".into());

    let json = serde_json::to_string(&card).expect("serialize");
    let back: NnCardMeta = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.name, card.name);
    assert_eq!(back.architecture, card.architecture);
    assert_eq!(back.training_path, card.training_path);
    assert_eq!(back.lineage.parent, card.lineage.parent);
    let (a, b) = (
        card.candle.as_ref().expect("orig candle"),
        back.candle.as_ref().expect("back candle"),
    );
    assert_eq!(a.bundle_ref, b.bundle_ref);
    assert!(b.lora.is_none());
}

/// The `SUPPORTED_TRAINING_PATHS` const includes `"merged"` and the
/// opt-in `validate_training_path` fn accepts it (and rejects a
/// typo). Guards S4's promise that "merged" is a first-class
/// accepted value alongside full_ft / lora / distillation.
#[test]
fn training_path_merged_accepted_by_validator() {
    assert!(SUPPORTED_TRAINING_PATHS.contains(&"merged"));
    assert!(validate_training_path("merged").is_ok());
    assert!(validate_training_path("full_ft").is_ok());
    assert!(validate_training_path("lora").is_ok());
    assert!(validate_training_path("distillation").is_ok());
    assert!(validate_training_path("merged_typo").is_err());
    assert!(validate_training_path("").is_err());
}

/// `MergedProvenance::validate` rejects any empty field with a
/// message naming the field. The `to_card_meta` projection assumes
/// non-empty inputs; a caller writing an empty field would produce
/// a card that is unloadable in a meaningful way.
#[test]
fn merged_provenance_validate_rejects_empty_fields() {
    let base = sample_provenance();

    let empty_lora = MergedProvenance {
        lora_card: "".into(),
        ..base.clone()
    };
    let err = empty_lora.validate().unwrap_err();
    assert!(err.contains("lora_card"), "err was: {err}");

    let empty_arch = MergedProvenance {
        arch: "".into(),
        ..base.clone()
    };
    let err = empty_arch.validate().unwrap_err();
    assert!(err.contains("arch"), "err was: {err}");

    let empty_bundle = MergedProvenance {
        bundle_ref: "".into(),
        ..base
    };
    let err = empty_bundle.validate().unwrap_err();
    assert!(err.contains("bundle_ref"), "err was: {err}");
}
