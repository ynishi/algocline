//! Integration test — Layer 4a merged export parity for TinyLlama.
//!
//! Mirror of `tests/merged_export_parity_gpt2.rs`, applied to
//! `TinyLlamaModel` on the `TinyLlamaConfig::tiny()` preset (2 layers,
//! 2 heads, 1 KV head so GQA `n_rep == 2`, head_dim 32, dim 64,
//! hidden_dim 128).
//!
//! Additional TinyLlama-specific coverage:
//!
//! - GQA K/V projection shape preservation — `k_proj.weight` /
//!   `v_proj.weight` output dim stays `kv_heads * head_dim = 32` on
//!   the `tiny` preset (never broadcast up to Q's `heads * head_dim
//!   = 64`).

use algocline_nn::arch::lora::MergeableLora;
use algocline_nn::arch::{max_abs_diff_f32, LoraConfig, TinyLlamaConfig, TinyLlamaModel};
use algocline_nn::merged::{export_merged, MergedProvenance};
use algocline_nn::train::{
    run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
    TrainingLease,
};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;
use tempfile::TempDir;

/// Build a fresh TinyLlama on `tiny`, wrap + briefly train LoRA so
/// the wrap is non-identity, return `(model, base_vm, cfg)`.
fn wrap_and_train_briefly() -> (TinyLlamaModel, VarMap, TinyLlamaConfig) {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

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
    };
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let _ = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss,
        tmp.path(),
        "merge-parity-tll",
        lease,
    )
    .expect("run_lora_ft must succeed");
    (model, base_vm, cfg)
}

/// Wrapped forward parity: reload the merged bundle as a plain
/// TinyLlama and assert forward output matches within f32 tolerance.
#[test]
fn merged_bundle_tinyllama_forward_matches_wrapped_forward() {
    let (wrapped_model, _base_vm, cfg) = wrap_and_train_briefly();

    let input = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 0], (1, 8), &cfg.device).unwrap();
    let wrapped_out = wrapped_model.forward(&input).unwrap();

    let tmp = TempDir::new().unwrap();
    let out_path = tmp.path().join("nn").join("merged-tll.safetensors");
    let provenance = MergedProvenance {
        lora_card: "cards/tll-lora-parity".into(),
        arch: "tinyllama-tiny".into(),
        bundle_ref: "nn/merged-tll".into(),
    };
    let (bytes, card) =
        export_merged(&wrapped_model, &provenance, &out_path).expect("export_merged");
    assert!(bytes > 0);
    assert_eq!(card.training_path, "merged");
    assert_eq!(card.architecture, "tinyllama-tiny");

    // Reload merged as plain TinyLlama.
    let loaded_model =
        TinyLlamaModel::from_safetensors_file(&cfg, &out_path).expect("reload merged as plain tll");
    let loaded_out = loaded_model.forward(&input).unwrap();

    let diff = max_abs_diff_f32(&wrapped_out, &loaded_out).unwrap();
    assert!(
        diff < 1e-4,
        "wrapped vs merged tinyllama forward diverged by {diff} (>= 1e-4)"
    );
}

/// Emitted key set matches exactly the HF Llama layout that
/// `TinyLlamaModel::new` consumes under a `VarBuilder` rooted at the
/// model root (`model.*` plus top-level `lm_head.weight`).
#[test]
fn merged_bundle_tinyllama_key_layout_matches_from_pretrained() {
    let (model, _base_vm, cfg) = wrap_and_train_briefly();
    let map = model.export_merged().expect("export_merged");

    let mut expected: std::collections::HashSet<String> = std::collections::HashSet::new();
    expected.insert("model.embed_tokens.weight".into());
    expected.insert("model.norm.weight".into());
    expected.insert("lm_head.weight".into());
    for i in 0..cfg.layers {
        let p = format!("model.layers.{i}");
        expected.insert(format!("{p}.input_layernorm.weight"));
        expected.insert(format!("{p}.post_attention_layernorm.weight"));
        expected.insert(format!("{p}.self_attn.q_proj.weight"));
        expected.insert(format!("{p}.self_attn.k_proj.weight"));
        expected.insert(format!("{p}.self_attn.v_proj.weight"));
        expected.insert(format!("{p}.self_attn.o_proj.weight"));
        expected.insert(format!("{p}.mlp.gate_proj.weight"));
        expected.insert(format!("{p}.mlp.up_proj.weight"));
        expected.insert(format!("{p}.mlp.down_proj.weight"));
    }

    let got: std::collections::HashSet<String> = map.keys().cloned().collect();
    assert_eq!(
        got, expected,
        "merged bundle key set differs from expected HF Llama layout"
    );
}

/// GQA K/V shape preservation. On `tiny`, `kv_heads = 1`, `head_dim
/// = 32`, `heads = 2`, so K/V projections have output shape
/// `[kv_heads * head_dim, dim] = [32, 64]`, while Q has
/// `[heads * head_dim, dim] = [64, 64]`. A walker bug that
/// broadcasts K/V to Q's shape would surface here.
#[test]
fn merged_bundle_tinyllama_preserves_gqa_kv_shape() {
    let (model, _base_vm, cfg) = wrap_and_train_briefly();
    let map = model.export_merged().expect("export_merged");

    let head_dim = cfg.dim / cfg.heads;
    let q_expected = (cfg.heads * head_dim, cfg.dim);
    let kv_expected = (cfg.kv_heads * head_dim, cfg.dim);

    for i in 0..cfg.layers {
        let q = &map[&format!("model.layers.{i}.self_attn.q_proj.weight")];
        let k = &map[&format!("model.layers.{i}.self_attn.k_proj.weight")];
        let v = &map[&format!("model.layers.{i}.self_attn.v_proj.weight")];
        assert_eq!(
            q.dims(),
            [q_expected.0, q_expected.1],
            "layer {i} q_proj shape mismatch"
        );
        assert_eq!(
            k.dims(),
            [kv_expected.0, kv_expected.1],
            "layer {i} k_proj shape mismatch (GQA collapse expected: [{}, {}], got {:?})",
            kv_expected.0,
            kv_expected.1,
            k.dims()
        );
        assert_eq!(
            v.dims(),
            [kv_expected.0, kv_expected.1],
            "layer {i} v_proj shape mismatch (GQA collapse expected: [{}, {}], got {:?})",
            kv_expected.0,
            kv_expected.1,
            v.dims()
        );
    }
}

/// Base parameters bit-identical before / after `export_merged`.
#[test]
fn merged_bundle_tinyllama_leaves_base_bit_identical() {
    let (model, base_vm, _cfg) = wrap_and_train_briefly();

    let base_before: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();

    let tmp = TempDir::new().unwrap();
    for i in 0..2 {
        let out_path = tmp
            .path()
            .join("nn")
            .join(format!("merged-tll-{i}.safetensors"));
        let provenance = MergedProvenance {
            lora_card: "cards/tll-lora-parity".into(),
            arch: "tinyllama-tiny".into(),
            bundle_ref: format!("nn/merged-tll-{i}"),
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
