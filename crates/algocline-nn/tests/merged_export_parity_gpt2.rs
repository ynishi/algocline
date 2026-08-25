//! Integration test — Layer 4a merged export parity for GPT-2.
//!
//! After training a LoRA-wrapped [`Gpt2Model`] briefly, exporting the
//! merged bundle via [`export_merged`] and reloading it as a plain
//! `Gpt2Model` (through the same `VarBuilder::from_mmaped_safetensors`
//! path used by `from_pretrained`) must produce a model whose forward
//! output matches the wrapped model's forward within f32 tolerance.
//!
//! Complementary invariants:
//!
//! - The emitted key set matches exactly the tensor names
//!   `Gpt2Model::new` reads under a `VarBuilder` (HF GPT-2 layout).
//! - The base `VarMap` is bit-identical before / after
//!   `export_merged` — merge is a pure read.
//!
//! These are the properties that let a downstream inference stack
//! drop `wrap_lora` + delta-load entirely once it reads a merged
//! bundle; the wide 4b load-side integration then only needs to
//! recognise `training_path == "merged"` and route to the plain
//! `Gpt2Model::from_safetensors_*` path.
//!
//! The test uses `TinyLlamaConfig::tiny`-style micro presets on CPU
//! F32 so it stays under `cargo test` friendly.

use algocline_nn::arch::lora::MergeableLora;
use algocline_nn::arch::{max_abs_diff_f32, Gpt2Config, Gpt2Model, LoraConfig};
use algocline_nn::merged::{export_merged, MergedProvenance};
use algocline_nn::train::{
    run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
    TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;
use tempfile::TempDir;

fn tiny_gpt2_cfg() -> Gpt2Config {
    Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab: 32,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: None,
    }
}

/// Build a fresh GPT-2, wrap + briefly train LoRA so the wrap is
/// non-identity, return the trained wrapped model + a shared base
/// `VarMap` snapshot (for the base-freeze assertion).
fn wrap_and_train_briefly() -> (Gpt2Model, VarMap, Gpt2Config) {
    let cfg = tiny_gpt2_cfg();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = Gpt2Model::new(&cfg, vs).unwrap();

    let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(30).collect();
    let mut ds = TokenizedDataset::new(
        rows,
        DatasetOpts {
            batch_size: 1,
            ctx_len: 8,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );
    let loss = CrossEntropyLoss::new();
    let train_cfg = FullFtConfig {
        lr: 5e-3,
        batch_size: 1,
        grad_accum: 1,
        steps: 20,
        warmup: 2,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
        init_from: None,
        mask_disallowed_logits: false,
    };
    let lora_cfg = LoraConfig::new(4, 8.0);
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let _ = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss,
        tmp.path(),
        "merge-parity",
        lease,
    )
    .expect("run_lora_ft must succeed");
    (model, base_vm, cfg)
}

/// Wrapped forward parity: export a merged bundle, reload it as a
/// plain `Gpt2Model`, and assert both models produce (near-)
/// identical logits on a fixed input.
#[test]
fn merged_bundle_forward_matches_wrapped_forward() {
    let (wrapped_model, _base_vm, cfg) = wrap_and_train_briefly();

    let input = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 0], (1, 8), &cfg.device).unwrap();
    let wrapped_out = wrapped_model.forward(&input).unwrap();

    let tmp = TempDir::new().unwrap();
    let out_path = tmp.path().join("nn").join("merged-gpt2.safetensors");
    let provenance = MergedProvenance {
        lora_card: "cards/gpt2-lora-parity".into(),
        arch: "gpt2-tiny".into(),
        bundle_ref: "nn/merged-gpt2".into(),
    };
    let (bytes, card) =
        export_merged(&wrapped_model, &provenance, &out_path).expect("export_merged");
    assert!(bytes > 0);
    assert_eq!(card.training_path, "merged");

    // Reload the merged file as if it were a plain pretrained base.
    let loaded_model =
        Gpt2Model::from_safetensors_file(&cfg, &out_path).expect("reload merged as plain gpt2");
    let loaded_out = loaded_model.forward(&input).unwrap();

    let diff = max_abs_diff_f32(&wrapped_out, &loaded_out).unwrap();
    assert!(
        diff < 1e-4,
        "wrapped vs merged forward diverged by {diff} (>= 1e-4)"
    );
}

/// Emitted key set matches exactly the tensor names `Gpt2Model::new`
/// consumes under a `VarBuilder`. Divergence is a naming bug.
#[test]
fn merged_bundle_key_layout_matches_from_pretrained() {
    let (model, _base_vm, cfg) = wrap_and_train_briefly();
    let map = model.export_merged().expect("export_merged");

    // Build the set of keys we expect based on the model shape.
    let mut expected: std::collections::HashSet<String> = std::collections::HashSet::new();
    expected.insert("wte.weight".into());
    expected.insert("wpe.weight".into());
    for i in 0..cfg.layers {
        let p = format!("h.{i}");
        // LayerNorms with weight + bias.
        expected.insert(format!("{p}.ln_1.weight"));
        expected.insert(format!("{p}.ln_1.bias"));
        expected.insert(format!("{p}.ln_2.weight"));
        expected.insert(format!("{p}.ln_2.bias"));
        // Attention + MLP linears with weight + bias.
        expected.insert(format!("{p}.attn.c_attn.weight"));
        expected.insert(format!("{p}.attn.c_attn.bias"));
        expected.insert(format!("{p}.attn.c_proj.weight"));
        expected.insert(format!("{p}.attn.c_proj.bias"));
        expected.insert(format!("{p}.mlp.c_fc.weight"));
        expected.insert(format!("{p}.mlp.c_fc.bias"));
        expected.insert(format!("{p}.mlp.c_proj.weight"));
        expected.insert(format!("{p}.mlp.c_proj.bias"));
    }
    expected.insert("ln_f.weight".into());
    expected.insert("ln_f.bias".into());

    let got: std::collections::HashSet<String> = map.keys().cloned().collect();
    assert_eq!(got, expected, "merged bundle key set differs from expected");
}

/// Base parameters bit-identical before / after `export_merged`.
/// Merge is a pure read; any drift is a bug.
#[test]
fn merged_bundle_leaves_base_bit_identical() {
    let (model, base_vm, _cfg) = wrap_and_train_briefly();

    let base_before: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();

    // Run export twice to strengthen the "pure read" claim.
    let tmp = TempDir::new().unwrap();
    for i in 0..2 {
        let out_path = tmp
            .path()
            .join("nn")
            .join(format!("merged-gpt2-{i}.safetensors"));
        let provenance = MergedProvenance {
            lora_card: "cards/gpt2-lora-parity".into(),
            arch: "gpt2-tiny".into(),
            bundle_ref: format!("nn/merged-gpt2-{i}"),
        };
        let _ = export_merged(&model, &provenance, &out_path).expect("export_merged");
    }

    let base_after: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    for (idx, (before, after)) in base_before.iter().zip(base_after.iter()).enumerate() {
        assert_eq!(
            before, after,
            "base var index {idx} drifted through export_merged"
        );
    }
}
