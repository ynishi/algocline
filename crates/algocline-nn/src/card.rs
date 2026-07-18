//! Card metadata schema for `alc.nn.card.*`.
//!
//! Mirrors the `[metadata.nn]` TOML block written by the engine bridge
//! (`bridge/nn_card.rs`) when it assembles the Card create payload.
//! Downstream training paths (Full FT / LoRA / Distillation) populate
//! `hyperparams` / `metrics` / `lineage` uniformly through this schema.
//!
//! `hyperparams` and `metrics` are free-form JSON pass-through so trainer
//! subtasks can extend without reshaping this crate.
//!
//! The Card foundation leaves `NnCandleBranch::lora` as `None`. A
//! later LoRA follow-up populates it via the [`NnLoraBranch`]
//! sub-struct without breaking foundation serialization
//! (`skip_serializing_if = "Option::is_none"`).

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

/// Content of `[metadata.nn]`.
///
/// `training_path` and `architecture` are required; everything else is
/// optional with a sensible default so trainer subtasks can populate
/// incrementally.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnCardMeta {
    /// Logical model name (user-facing, e.g. `"domain-classifier-v3"`).
    pub name: String,

    /// Runtime backend of record. `"candle"` for v1; `"endpoint"` /
    /// `"hosted"` / `"adapter"` are v2+ carry.
    pub backend: String,

    /// Informational task label (free-form, e.g. `"classification"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,

    /// Architecture preset id (e.g. `"gpt2-medium"`).
    pub architecture: String,

    /// Training path taken: `"full_ft"` / `"lora"` / `"distillation"`.
    pub training_path: String,

    /// Lineage back-references (parent / teacher / data / tokenizer).
    #[serde(default)]
    pub lineage: NnLineage,

    /// Free-form hyperparameter table (trainer subtasks populate).
    ///
    /// Stored as a JSON object; empty object `{}` means "no hyperparams
    /// recorded" and serializes to an empty TOML sub-table.
    #[serde(default = "empty_object")]
    pub hyperparams: Json,

    /// Free-form training metrics (trainer subtasks populate). Same
    /// semantics as [`Self::hyperparams`].
    #[serde(default = "empty_object")]
    pub metrics: Json,

    /// Backend-specific candle branch. Present when
    /// `backend == "candle"` (v1 default); absent for `endpoint` /
    /// `hosted` cards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle: Option<NnCandleBranch>,
}

/// Content of `[metadata.nn.lineage]`.
///
/// Every field is optional so a fresh full-FT card (no parent, no
/// teacher) can omit the block entirely without breaking the schema.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct NnLineage {
    /// Prior card id when this model iterates on a predecessor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,

    /// Distillation teacher card ref (e.g. `"cards/haiku-run-042"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<String>,

    /// Training data card ref (e.g. `"cards/prompts-corpus-v1"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<String>,

    /// Tokenizer preset id (e.g. `"gpt2"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

/// Content of `[metadata.nn.candle]`.
///
/// `bundle_ref` is required and always equal to `"nn/<card_id>"` per
/// design §5 (1:1 mapping between Card id and safetensors bundle name).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnCandleBranch {
    /// Bundle reference `"nn/<card_id>"` — the engine bridge fills this
    /// deterministically from the Card id at save time.
    pub bundle_ref: String,

    /// Device (e.g. `"cuda:0"` / `"cpu"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,

    /// Weights precision (`"bf16"` / `"fp16"` / `"fp32"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// LoRA branch — populated by the LoRA follow-up. The Card
    /// foundation leaves this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<NnLoraBranch>,
}

/// Content of `[metadata.nn.candle.lora]`.
///
/// Present only when `training_path == "lora"`. The LoRA follow-up
/// populates all three fields; the Card foundation does not
/// construct this struct.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnLoraBranch {
    /// Low-rank decomposition rank (typical: 8 / 16 / 32).
    pub rank: u32,

    /// LoRA scaling factor (`alpha / rank` becomes the effective LR
    /// scaler applied to the delta).
    pub alpha: u32,

    /// Bundle reference for the frozen base model
    /// (e.g. `"nn/base-gpt2-medium"`).
    pub base_bundle_ref: String,
}

/// Default value for `hyperparams` / `metrics` — an empty JSON object.
fn empty_object() -> Json {
    Json::Object(serde_json::Map::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_ft_meta_roundtrips_through_json() {
        let meta = NnCardMeta {
            name: "domain-classifier-v3".into(),
            backend: "candle".into(),
            task: Some("classification".into()),
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage {
                tokenizer: Some("gpt2".into()),
                ..Default::default()
            },
            hyperparams: serde_json::json!({ "lr": 3e-4, "steps": 20000 }),
            metrics: serde_json::json!({ "val_loss_final": 0.51 }),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/abc123".into(),
                device: Some("cuda:0".into()),
                dtype: Some("bf16".into()),
                lora: None,
            }),
        };

        let json = serde_json::to_value(&meta).expect("serialize");
        let back: NnCardMeta = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(back.name, meta.name);
        assert_eq!(back.training_path, "full_ft");
        assert_eq!(back.candle.as_ref().unwrap().bundle_ref, "nn/abc123");
        assert!(back.candle.as_ref().unwrap().lora.is_none());
        assert_eq!(back.lineage.tokenizer.as_deref(), Some("gpt2"));

        // Round-trip preserves every field verbatim.
        let json2 = serde_json::to_value(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn lora_branch_serializes_when_present() {
        let meta = NnCardMeta {
            name: "student-lora".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "lora".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/student-lora".into(),
                device: None,
                dtype: None,
                lora: Some(NnLoraBranch {
                    rank: 16,
                    alpha: 32,
                    base_bundle_ref: "nn/base-gpt2-medium".into(),
                }),
            }),
        };

        let json = serde_json::to_value(&meta).expect("serialize");
        let candle = json.get("candle").expect("candle branch present");
        let lora = candle.get("lora").expect("lora sub-object present");
        assert_eq!(lora.get("rank"), Some(&serde_json::json!(16)));
        assert_eq!(lora.get("alpha"), Some(&serde_json::json!(32)));
        assert_eq!(
            lora.get("base_bundle_ref"),
            Some(&serde_json::json!("nn/base-gpt2-medium"))
        );
    }

    #[test]
    fn absent_lora_is_omitted_not_null() {
        let meta = NnCardMeta {
            name: "no-lora".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/x".into(),
                device: None,
                dtype: None,
                lora: None,
            }),
        };
        let json = serde_json::to_value(&meta).expect("serialize");
        let candle = json.get("candle").expect("candle branch");
        assert!(
            candle.get("lora").is_none(),
            "lora must be omitted (skip_serializing_if), got: {candle}"
        );
    }

    #[test]
    fn deserialize_tolerates_absent_optional_fields() {
        let minimal = serde_json::json!({
            "name": "m",
            "backend": "candle",
            "architecture": "gpt2-medium",
            "training_path": "full_ft",
        });
        let meta: NnCardMeta = serde_json::from_value(minimal).expect("deserialize minimal");
        assert!(meta.task.is_none());
        assert!(meta.candle.is_none());
        assert!(meta.lineage.parent.is_none());
        assert_eq!(meta.hyperparams, empty_object());
        assert_eq!(meta.metrics, empty_object());
    }
}
