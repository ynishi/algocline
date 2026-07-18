//! Integration tests for the GPT-2 forward path.
//!
//! Exercises the crate boundary: build a `Gpt2Model` on CPU via
//! `Gpt2Model::new`, run `forward`, and assert the output shape is
//! `[batch, seq, vocab]` (subtask invariant #1).
//!
//! Uses tiny configs so the tests run in milliseconds without touching
//! the network or a GPU.

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn tiny_cfg(layers: usize, heads: usize, dim: usize, ctx: usize, vocab: usize) -> Gpt2Config {
    Gpt2Config {
        layers,
        heads,
        dim,
        ctx,
        vocab,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
    }
}

fn build_random(cfg: &Gpt2Config) -> Gpt2Model {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
    Gpt2Model::new(cfg, vs).expect("build random gpt-2 model")
}

#[test]
fn forward_single_row_matches_batch_seq_vocab() {
    let cfg = tiny_cfg(1, 2, 8, 4, 16);
    let model = build_random(&cfg);
    let ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &cfg.device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.dims(), &[1, 4, 16]);
}

#[test]
fn forward_batch_shape_scales_across_first_axis() {
    let cfg = tiny_cfg(2, 2, 16, 8, 32);
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
    // trim / wrap. Matches the design's causal-mask contract.
    let cfg = tiny_cfg(1, 2, 8, 3, 16);
    let model = build_random(&cfg);
    let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
    let err = model
        .forward(&ids)
        .expect_err("expected ctx overflow error");
    let msg = err.to_string();
    assert!(msg.contains("exceeds ctx"), "unexpected error: {msg}");
}

#[test]
fn config_medium_and_large_expose_expected_shapes() {
    let m = Gpt2Config::medium();
    assert_eq!(
        (m.layers, m.heads, m.dim, m.ctx, m.vocab),
        (24, 16, 1024, 1024, 50257)
    );
    assert_eq!(m.hf_repo(), Some("openai-community/gpt2-medium"));

    let l = Gpt2Config::large();
    assert_eq!(
        (l.layers, l.heads, l.dim, l.ctx, l.vocab),
        (36, 20, 1280, 1024, 50257)
    );
    assert_eq!(l.hf_repo(), Some("openai-community/gpt2-large"));
}

#[test]
fn config_from_variant_resolves_aliases_and_rejects_unknown() {
    assert!(Gpt2Config::from_variant("medium").is_some());
    assert!(Gpt2Config::from_variant("gpt2-medium").is_some());
    assert!(Gpt2Config::from_variant("large").is_some());
    assert!(Gpt2Config::from_variant("gpt2-large").is_some());
    assert!(Gpt2Config::from_variant("small").is_none());
    assert!(Gpt2Config::from_variant("").is_none());
}

#[test]
fn model_config_accessor_returns_matching_cfg() {
    let cfg = tiny_cfg(1, 2, 8, 4, 16);
    let model = build_random(&cfg);
    let got = model.config();
    assert_eq!(got.layers, 1);
    assert_eq!(got.heads, 2);
    assert_eq!(got.dim, 8);
    assert_eq!(got.ctx, 4);
    assert_eq!(got.vocab, 16);
}
