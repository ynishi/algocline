//! Merged inference checkpoint export (Layer 4a of GH #10).
//!
//! After a LoRA-wrapped model has been trained (via
//! [`crate::train::run_lora_ft`]), the on-disk delta bundle plus the
//! base model can be composed into a single safetensors bundle that
//! downstream consumers load as if it were a plain pretrained model.
//! This module ships the export half only — a caller-supplied
//! [`MergedProvenance`] plus a wrapped model produce a safetensors
//! file on disk (via [`export_merged`]) and a matching
//! [`NnCardMeta`] describing the merged bundle's provenance.
//!
//! # Design pattern (§Q0)
//!
//! [`MergedProvenance`] is the Model-side SoT for what the Card
//! metadata layer will record. The `to_card_meta` projection maps
//! into existing [`NnCardMeta`] fields only — the Card schema
//! itself (`crate::card`) is not modified beyond adding `"merged"`
//! to the accepted `training_path` values (Layer 4a S4). Future
//! branches (LoRA / distillation / full-FT hyperparams) can adopt
//! the same pattern — Model-side struct + `to_card_*` projection —
//! so the "which Card field do we invent?" negotiation stays local
//! to each branch's Model-side struct.
//!
//! See `workspace/tasks/alc-nn-tinyllama/layer-4-merged-ckpt-design.md`
//! §3 Q0 for the full pattern rationale.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::arch::lora::MergeableLora;
use crate::card::{NnCandleBranch, NnCardMeta, NnLineage};

/// Provenance for a merged inference bundle. This is the Model-side
/// SoT (per §Q0 pattern); the [`Self::to_card_meta`] projection maps
/// it onto existing Card schema fields without extending
/// [`NnCardMeta`] / [`NnCandleBranch`] / [`NnLineage`].
///
/// # Field discipline
///
/// - `lora_card` is the sole provenance handle recorded. The base
///   model reference is transitively reachable via the LoRA card's
///   `NnLoraBranch.base_bundle_ref` — denormalising it onto the
///   merged card would create a second SoT that could drift.
/// - LoRA hyperparams (`rank` / `alpha` / `target_modules`) are
///   NOT copied here. The LoRA card that `lora_card` points at
///   remains the SoT for those values.
#[derive(Debug, Clone)]
pub struct MergedProvenance {
    /// Card id of the LoRA training run this merge is derived from
    /// (e.g. `"cards/domain-lora-042"`). Recorded in
    /// `NnLineage.parent` at projection time.
    pub lora_card: String,

    /// Architecture family+variant identifier (e.g.
    /// `"tinyllama-1.1b"` / `"gpt2-medium"`). Recorded in
    /// `NnCardMeta.architecture`.
    pub arch: String,

    /// Bundle reference — conventionally `"nn/<merged-card-id>"`
    /// to match the existing `NnCandleBranch.bundle_ref` shape.
    pub bundle_ref: String,
}

impl MergedProvenance {
    /// Validate that every provenance field is populated.
    ///
    /// The projection to `NnCardMeta` requires non-empty values for
    /// `lora_card` / `arch` / `bundle_ref`; a merged card missing
    /// any of them is unloadable in a meaningful way, so `validate`
    /// rejects at construction time rather than deferring to
    /// downstream deserialisation.
    pub fn validate(&self) -> Result<(), String> {
        if self.lora_card.is_empty() {
            return Err("MergedProvenance.lora_card must not be empty".into());
        }
        if self.arch.is_empty() {
            return Err("MergedProvenance.arch must not be empty".into());
        }
        if self.bundle_ref.is_empty() {
            return Err("MergedProvenance.bundle_ref must not be empty".into());
        }
        Ok(())
    }

    /// Project into an `NnCardMeta` using only existing Card schema
    /// fields.
    ///
    /// Mapping:
    ///
    /// - `NnCardMeta.name` ← `name` argument
    /// - `NnCardMeta.backend` ← `"candle"` (Layer 4a is Rust-side candle only)
    /// - `NnCardMeta.architecture` ← `self.arch`
    /// - `NnCardMeta.training_path` ← `"merged"` (accepted after S4 lands)
    /// - `NnCardMeta.lineage.parent` ← `Some(self.lora_card)`
    /// - `NnCandleBranch.bundle_ref` ← `self.bundle_ref`
    /// - `NnCandleBranch.lora` ← `None` (merged card no longer needs a wrap
    ///   block)
    /// - `hyperparams` / `metrics` ← empty JSON objects (SoT remains on the
    ///   referenced LoRA card)
    pub fn to_card_meta(&self, name: String) -> NnCardMeta {
        NnCardMeta {
            name,
            backend: "candle".into(),
            task: None,
            architecture: self.arch.clone(),
            training_path: "merged".into(),
            lineage: NnLineage {
                parent: Some(self.lora_card.clone()),
                ..NnLineage::default()
            },
            hyperparams: serde_json::Value::Object(serde_json::Map::new()),
            metrics: serde_json::Value::Object(serde_json::Map::new()),
            candle: Some(NnCandleBranch {
                bundle_ref: self.bundle_ref.clone(),
                device: None,
                dtype: None,
                lora: None,
            }),
        }
    }
}

/// Errors surfaced by [`export_merged`].
///
/// Explicit variants so the Lua bridge / caller can surface an
/// actionable error string — no silent fallback (matches the
/// Service-layer error-propagation discipline in
/// `.claude/CLAUDE.md`).
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// Provenance validation failed (empty required field).
    #[error("provenance: {0}")]
    Provenance(String),

    /// The arch's `export_merged` walker failed (e.g. LoRA merge
    /// math error, tensor device mismatch).
    #[error("merge: {0}")]
    Merge(candle_core::Error),

    /// Filesystem or serialisation IO failed.
    #[error("io: {0}")]
    Io(String),

    /// safetensors save failure.
    #[error("serialize: {0}")]
    Serialize(String),
}

/// Export a merged inference-ready safetensors bundle for a
/// LoRA-wrapped model.
///
/// The output bundle is a drop-in base model: it can be loaded
/// through the same arch's `from_pretrained` /
/// `VarBuilder::from_mmaped_safetensors` path with the same
/// `<Arch>Config` that produced it, and its forward output matches
/// the wrapped model's forward within f32 tolerance (parity
/// verified by `tests/merged_export_parity_*`).
///
/// Returns the number of bytes written to `out_path` plus the
/// projected `NnCardMeta` that the caller can serialise alongside
/// the bundle (e.g. via the engine bridge's Card store).
///
/// # Errors
///
/// - `MergeError::Provenance` if any [`MergedProvenance`] field is
///   empty.
/// - `MergeError::Merge` if the walker fails (arithmetic /
///   tensor-shape error).
/// - `MergeError::Io` if the parent directory cannot be created or
///   the final file size cannot be measured.
/// - `MergeError::Serialize` if `candle_core::safetensors::save`
///   rejects the tensor map.
pub fn export_merged<M: MergeableLora>(
    model: &M,
    provenance: &MergedProvenance,
    out_path: &Path,
) -> Result<(usize, NnCardMeta), MergeError> {
    provenance.validate().map_err(MergeError::Provenance)?;

    // Walk the model on its live device (§3 Q4-A).
    let device_map = model.export_merged().map_err(MergeError::Merge)?;

    // Move every tensor to CPU before handing to safetensors.
    // safetensors::save expects host-resident data.
    let cpu = Device::Cpu;
    let mut cpu_map: HashMap<String, Tensor> = HashMap::with_capacity(device_map.len());
    for (k, t) in device_map {
        let t_cpu = t.to_device(&cpu).map_err(MergeError::Merge)?;
        cpu_map.insert(k, t_cpu);
    }

    // Ensure the parent dir exists so the caller doesn't have to
    // mkdir separately.
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| MergeError::Io(format!("mkdir {:?}: {e}", parent)))?;
        }
    }

    candle_core::safetensors::save(&cpu_map, out_path)
        .map_err(|e| MergeError::Serialize(e.to_string()))?;

    let bytes = std::fs::metadata(out_path)
        .map_err(|e| MergeError::Io(format!("stat {:?}: {e}", out_path)))?
        .len() as usize;

    let card = provenance.to_card_meta(
        out_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("merged")
            .to_string(),
    );

    Ok((bytes, card))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::lora::MergeableLora;
    use candle_core::Tensor;
    use std::collections::HashMap;

    /// Minimal `MergeableLora` stub: emits two toy tensors so the
    /// unit tests can drive `export_merged` without spinning up a
    /// full arch model.
    struct StubModel;

    impl MergeableLora for StubModel {
        fn export_merged(&self) -> candle_core::Result<HashMap<String, Tensor>> {
            let mut out = HashMap::new();
            out.insert(
                "unit.weight".into(),
                Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?,
            );
            out.insert(
                "unit.bias".into(),
                Tensor::from_slice(&[0.5f32, 0.5], 2, &Device::Cpu)?,
            );
            Ok(out)
        }
    }

    #[test]
    fn validate_rejects_empty_fields() {
        let base = MergedProvenance {
            lora_card: "cards/x".into(),
            arch: "tinyllama-1.1b".into(),
            bundle_ref: "nn/merged-001".into(),
        };
        assert!(base.validate().is_ok());

        let empty_lora = MergedProvenance {
            lora_card: "".into(),
            ..base.clone()
        };
        assert!(empty_lora.validate().is_err());

        let empty_arch = MergedProvenance {
            arch: "".into(),
            ..base.clone()
        };
        assert!(empty_arch.validate().is_err());

        let empty_bundle = MergedProvenance {
            bundle_ref: "".into(),
            ..base
        };
        assert!(empty_bundle.validate().is_err());
    }

    #[test]
    fn to_card_meta_maps_to_existing_fields() {
        let provenance = MergedProvenance {
            lora_card: "cards/domain-lora-042".into(),
            arch: "tinyllama-1.1b".into(),
            bundle_ref: "nn/merged-001".into(),
        };
        let card = provenance.to_card_meta("merged-001".into());

        assert_eq!(card.name, "merged-001");
        assert_eq!(card.backend, "candle");
        assert_eq!(card.architecture, "tinyllama-1.1b");
        assert_eq!(card.training_path, "merged");
        assert_eq!(
            card.lineage.parent.as_deref(),
            Some("cards/domain-lora-042")
        );
        assert!(card.lineage.teacher.is_none());
        assert!(card.lineage.training_data.is_none());
        assert!(card.lineage.tokenizer.is_none());
        let candle = card.candle.expect("merged card carries candle branch");
        assert_eq!(candle.bundle_ref, "nn/merged-001");
        assert!(candle.lora.is_none());
    }

    #[test]
    fn export_merged_writes_readable_safetensors_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nn").join("stub.safetensors");
        let provenance = MergedProvenance {
            lora_card: "cards/stub-lora".into(),
            arch: "stub-arch".into(),
            bundle_ref: "nn/stub".into(),
        };
        let (bytes, card) = export_merged(&StubModel, &provenance, &path).unwrap();
        assert!(bytes > 0);
        assert!(path.exists());
        assert_eq!(card.training_path, "merged");

        // Round-trip: load back and confirm the tensor set matches.
        let loaded = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("unit.weight"));
        assert!(loaded.contains_key("unit.bias"));
    }

    #[test]
    fn export_merged_reports_bytes_written() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("stub.safetensors");
        let provenance = MergedProvenance {
            lora_card: "cards/stub-lora".into(),
            arch: "stub-arch".into(),
            bundle_ref: "nn/stub".into(),
        };
        let (reported_bytes, _card) = export_merged(&StubModel, &provenance, &path).unwrap();
        let actual = std::fs::metadata(&path).unwrap().len() as usize;
        assert_eq!(reported_bytes, actual);
    }
}
