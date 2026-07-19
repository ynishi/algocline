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
/// Present only when `training_path == "lora"`. `rank` / `alpha` /
/// `base_bundle_ref` are always populated by the trainer; the extra
/// `target_modules` / `dropout` / `delta_path` fields carry defaults
/// for backwards compatibility with cards written before ST-d landed
/// (a foundation-era card without these fields still deserializes,
/// but `alc.nn.card.load_gpt2` errors when `delta_path` is missing —
/// the load path cannot locate the delta without it).
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

    /// LoRA target modules (subset of
    /// `["q_proj", "k_proj", "v_proj", "o_proj", "up", "down"]`).
    /// Recorded so the load-with-merge path can rebuild the same
    /// `LoraConfig` that produced the delta and wrap the base model
    /// identically.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// LoRA dropout probability applied during training. Currently
    /// held for provenance only — the shipped [`crate::arch::LoraLinear`]
    /// forward does not apply dropout at inference; the field is
    /// preserved so a future dropout-at-train-only variant can be
    /// distinguished from cards trained without it.
    #[serde(default)]
    pub dropout: f32,

    /// Absolute path of the delta safetensors file emitted by
    /// [`crate::train::run_lora_ft`] at
    /// `<ckpt_dir>/nn/lora-<ckpt_prefix>.safetensors`. Recorded so
    /// `alc.nn.card.load_gpt2` can locate the delta without knowing
    /// the caller's `ckpt_dir` / `ckpt_prefix` conventions.
    ///
    /// Absent for hand-authored cards; the load path errors loudly
    /// when a lora card lacks `delta_path` rather than guessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_path: Option<String>,
}

/// Default target modules when a LoRA card was written before ST-d
/// added the field. Matches
/// [`crate::arch::LoraConfig::default_targets`].
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
        "up".into(),
        "down".into(),
    ]
}

/// Default value for `hyperparams` / `metrics` — an empty JSON object.
fn empty_object() -> Json {
    Json::Object(serde_json::Map::new())
}

/// Architecture family prefixes accepted for [`NnCardMeta::architecture`].
///
/// Values written by the engine bridge use the `<family>-<variant>` shape
/// (e.g. `"gpt2-medium"`, `"llama-3.2-1b"`, `"tinyllama-1.1b"`). At save
/// time the bridge validates the prefix against this list so a typo does
/// not silently write an unloadable Card, and downstream loaders can
/// route by family prefix without a second lookup table.
///
/// Extend this list when a new trainable or inference-only architecture
/// lands; the corresponding `alc.nn.preset.<family>` binding is expected
/// to populate the same prefix.
pub const SUPPORTED_ARCHITECTURE_FAMILIES: &[&str] =
    &["gpt2", "llama", "tinyllama", "qwen2", "phi", "gemma"];

/// Validate that `arch` starts with a known family prefix from
/// [`SUPPORTED_ARCHITECTURE_FAMILIES`].
///
/// Accepts either the bare family name (`"gpt2"`) or the
/// `<family>-<variant>` form (`"gpt2-medium"`). The prefix must be
/// followed by end-of-string or `-`; a longer identifier that happens to
/// start with a family name (`"gpt2experimental"`) is rejected so the
/// namespace stays partitioned.
pub fn validate_architecture(arch: &str) -> Result<(), String> {
    if arch.is_empty() {
        return Err("architecture must not be empty".into());
    }
    for family in SUPPORTED_ARCHITECTURE_FAMILIES {
        if arch == *family {
            return Ok(());
        }
        if let Some(rest) = arch.strip_prefix(family) {
            if rest.starts_with('-') {
                return Ok(());
            }
        }
    }
    Err(format!(
        "unknown architecture family for {arch:?} (expected one of {SUPPORTED_ARCHITECTURE_FAMILIES:?}, \
         optionally followed by '-<variant>')"
    ))
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
                    target_modules: default_target_modules(),
                    dropout: 0.0,
                    delta_path: Some(
                        "/var/algocline/nn/ckpt/nn/lora-student-lora.safetensors".into(),
                    ),
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
        // Post-ST-d extension: target_modules + dropout serialize
        // alongside the trio.
        assert_eq!(
            lora.get("target_modules"),
            Some(&serde_json::json!(default_target_modules()))
        );
        assert_eq!(lora.get("dropout"), Some(&serde_json::json!(0.0)));
        assert_eq!(
            lora.get("delta_path"),
            Some(&serde_json::json!(
                "/var/algocline/nn/ckpt/nn/lora-student-lora.safetensors"
            ))
        );
    }

    /// A pre-ST-d card (only `rank` / `alpha` / `base_bundle_ref` in
    /// the lora sub-table) must still deserialize. `target_modules`
    /// falls back to the canonical six, `dropout` to 0.0, and
    /// `delta_path` to `None` — the load path then errors when it
    /// tries to locate the delta.
    #[test]
    fn deserialize_backwards_compat_lora_without_new_fields() {
        let legacy = serde_json::json!({
            "name": "legacy-lora",
            "backend": "candle",
            "architecture": "gpt2-medium",
            "training_path": "lora",
            "candle": {
                "bundle_ref": "nn/legacy-lora",
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "base_bundle_ref": "nn/base-gpt2-medium",
                }
            }
        });
        let meta: NnCardMeta = serde_json::from_value(legacy).expect("legacy lora deserialize");
        let lora = meta
            .candle
            .as_ref()
            .and_then(|c| c.lora.as_ref())
            .expect("lora branch present");
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16);
        assert_eq!(lora.target_modules, default_target_modules());
        assert_eq!(lora.dropout, 0.0);
        assert!(
            lora.delta_path.is_none(),
            "delta_path defaults to None for pre-ST-d cards"
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

    #[test]
    fn validate_architecture_accepts_known_families() {
        for arch in [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-tiny",
            "llama",
            "llama-3.2-1b",
            "tinyllama",
            "tinyllama-1.1b",
            "tinyllama-tiny",
            "qwen2",
            "qwen2-0.5b",
            "phi",
            "phi-1.5",
            "gemma",
            "gemma-2b",
        ] {
            assert!(
                validate_architecture(arch).is_ok(),
                "expected {arch} to be accepted"
            );
        }
    }

    #[test]
    fn validate_architecture_rejects_unknown_and_typos() {
        for arch in ["", "gpt3", "mistral-7b", "gpt2experimental", " gpt2-medium"] {
            assert!(
                validate_architecture(arch).is_err(),
                "expected {arch:?} to be rejected"
            );
        }
    }
}
