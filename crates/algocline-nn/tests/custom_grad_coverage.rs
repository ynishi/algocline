//! Gradient-coverage gate for the customized GPT-2 path — the
//! "kitchen sink" configuration.
//!
//! Rather than enumerating every axis combination, one config turns
//! every Phase 1 customization away from the reference at once
//! (SwiGLU + RMSNorm + parallel residual + non-default MLP ratio) and
//! requires a single masked-CE backward to reach every Var. This
//! checks structural graph connectivity of all the custom wiring —
//! including the paths a new axis is most likely to sever (the gated
//! `c_gate` branch, the RMSNorm slow shim, the parallel-residual
//! reconvergence) — in one pinned inventory.

mod common;

use algocline_nn::arch::{Activation, Gpt2Config, Gpt2Custom, Gpt2Model, NormKind, ResidualKind};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

#[test]
fn kitchen_sink_backward_reaches_every_base_var() {
    let cfg = Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab: 32,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: Some(Gpt2Custom {
            act: Activation::SwiGlu,
            norm: NormKind::RmsNorm,
            residual: ResidualKind::Parallel,
            mlp_ratio: 2,
        }),
    };
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build kitchen-sink gpt-2");

    // Same shift + mask discipline as the sibling gates: scored
    // positions at the causal tail so every parameter participates.
    let row: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    let mask: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let seq = row.len();
    let inputs = Tensor::from_slice(&row[..seq - 1], (1, seq - 1), &cfg.device).unwrap();
    let targets = Tensor::from_slice(&row[1..], (1, seq - 1), &cfg.device).unwrap();
    let mask_t = Tensor::from_slice(&mask[1..], (1, seq - 1), &cfg.device).unwrap();

    let logits = model.forward(&inputs).expect("forward");
    let loss = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask_t))
        .expect("loss");
    let grads = loss.backward().expect("backward");

    // Inventory: wte + wpe + 2 blocks × (ln_1.weight [RMS, no bias] +
    // c_attn w/b + c_proj w/b + ln_2.weight + mlp.c_fc w/b +
    // mlp.c_gate w/b + mlp.c_proj w/b = 12) + ln_f.weight = 27 Vars.
    common::assert_full_grad_coverage(&vm, &grads, 27);
}
