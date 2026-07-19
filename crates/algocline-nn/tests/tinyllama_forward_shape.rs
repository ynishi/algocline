//! Integration tests for the TinyLlama forward path (Layer 1b).
//!
//! Exercises the crate boundary: build a `TinyLlamaModel` on CPU via
//! `TinyLlamaModel::new`, run `forward`, and assert the output shape is
//! `[batch, seq, vocab]` (spike-status.md Layer 1b invariant).
//!
//! Uses tiny configs so the tests run in milliseconds without touching
//! the network or a GPU.

use algocline_nn::arch::{TinyLlamaConfig, TinyLlamaModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn tiny_cfg(
    layers: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    hidden_dim: usize,
    ctx: usize,
    vocab: usize,
) -> TinyLlamaConfig {
    TinyLlamaConfig {
        layers,
        heads,
        kv_heads,
        dim,
        hidden_dim,
        ctx,
        vocab,
        rope_theta: 10_000.0,
        eps: 1e-5,
        dtype: DType::F32,
        device: Device::Cpu,
    }
}

fn build_random(cfg: &TinyLlamaConfig) -> TinyLlamaModel {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
    TinyLlamaModel::new(cfg, vs).expect("build random tinyllama model")
}

#[test]
fn forward_single_row_matches_batch_seq_vocab() {
    // heads = 2, kv_heads = 1 → n_rep = 2, exercising the GQA path.
    let cfg = tiny_cfg(1, 2, 1, 16, 32, 4, 8);
    let model = build_random(&cfg);
    let ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &cfg.device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.dims(), &[1, 4, 8]);
}

#[test]
fn forward_batch_shape_scales_across_first_axis() {
    // heads = 4, kv_heads = 2 → n_rep = 2. head_dim = 32.
    let cfg = tiny_cfg(2, 4, 2, 128, 256, 8, 32);
    let model = build_random(&cfg);
    let ids = Tensor::from_slice(
        &[1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        (2, 8),
        &cfg.device,
    )
    .unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.dims(), &[2, 8, 32]);
}

#[test]
fn forward_rejects_seq_over_ctx() {
    // ctx = 3 but seq = 5 → the model must reject rather than silently
    // trim / wrap. Matches the causal-mask contract.
    let cfg = tiny_cfg(1, 2, 1, 16, 32, 3, 8);
    let model = build_random(&cfg);
    let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
    let err = model
        .forward(&ids)
        .expect_err("expected ctx overflow error");
    let msg = err.to_string();
    assert!(msg.contains("exceeds ctx"), "unexpected error: {msg}");
}

#[test]
fn forward_rejects_dim_not_divisible_by_heads() {
    let cfg = TinyLlamaConfig {
        layers: 1,
        heads: 3, // 8 % 3 != 0
        kv_heads: 1,
        dim: 8,
        hidden_dim: 16,
        ctx: 4,
        vocab: 8,
        rope_theta: 10_000.0,
        eps: 1e-5,
        dtype: DType::F32,
        device: Device::Cpu,
    };
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
    let msg = match TinyLlamaModel::new(&cfg, vs) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected an error"),
    };
    assert!(msg.contains("divisible"), "unexpected error: {msg}");
}

#[test]
fn forward_rejects_heads_not_divisible_by_kv_heads() {
    let cfg = TinyLlamaConfig {
        layers: 1,
        heads: 4,
        kv_heads: 3, // 4 % 3 != 0
        dim: 32,
        hidden_dim: 64,
        ctx: 4,
        vocab: 8,
        rope_theta: 10_000.0,
        eps: 1e-5,
        dtype: DType::F32,
        device: Device::Cpu,
    };
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
    let msg = match TinyLlamaModel::new(&cfg, vs) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected an error"),
    };
    assert!(
        msg.contains("kv_heads"),
        "expected kv_heads error, got: {msg}"
    );
}

#[test]
fn config_1_1b_exposes_expected_shapes() {
    let c = TinyLlamaConfig::tinyllama_1_1b();
    assert_eq!(
        (
            c.layers,
            c.heads,
            c.kv_heads,
            c.dim,
            c.hidden_dim,
            c.ctx,
            c.vocab
        ),
        (22, 32, 4, 2048, 5632, 2048, 32000)
    );
    assert_eq!(c.head_dim(), 64);
    assert_eq!(
        c.hf_repo(),
        Some("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    );
}

#[test]
fn config_from_variant_resolves_aliases_and_rejects_unknown() {
    assert!(TinyLlamaConfig::from_variant("tinyllama-1.1b").is_some());
    assert!(TinyLlamaConfig::from_variant("1.1b").is_some());
    assert!(TinyLlamaConfig::from_variant("tinyllama-tiny").is_some());
    assert!(TinyLlamaConfig::from_variant("tiny").is_some());
    assert!(TinyLlamaConfig::from_variant("llama-2").is_none());
    assert!(TinyLlamaConfig::from_variant("").is_none());
}

#[test]
fn config_tiny_exposes_expected_shapes_and_no_hf_repo() {
    let t = TinyLlamaConfig::tiny();
    assert_eq!(
        (
            t.layers,
            t.heads,
            t.kv_heads,
            t.dim,
            t.hidden_dim,
            t.ctx,
            t.vocab
        ),
        (2, 2, 1, 64, 128, 16, 32)
    );
    // No HuggingFace bundle at this size.
    assert_eq!(t.hf_repo(), None);
}

#[test]
fn model_config_accessor_returns_matching_cfg() {
    let cfg = tiny_cfg(2, 2, 1, 16, 32, 4, 8);
    let model = build_random(&cfg);
    let got = model.config();
    assert_eq!(got.layers, 2);
    assert_eq!(got.heads, 2);
    assert_eq!(got.kv_heads, 1);
    assert_eq!(got.dim, 16);
    assert_eq!(got.hidden_dim, 32);
    assert_eq!(got.ctx, 4);
    assert_eq!(got.vocab, 8);
}
