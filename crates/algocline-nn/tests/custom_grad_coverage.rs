//! Gradient-coverage gates for the customized GPT-2 path — the
//! "kitchen sink" configurations.
//!
//! Rather than enumerating every axis combination, a config turns
//! every customization it legally can away from the reference at once
//! and requires a single masked-CE backward to reach every Var. This
//! checks structural graph connectivity of all the custom wiring in
//! one pinned inventory.
//!
//! Two sinks because Post-LN × parallel residual is rejected by
//! design (no canonical wiring), so no single config can flip every
//! axis:
//!
//! - **sink A** — SwiGLU + RMSNorm + parallel residual + ratio 2 +
//!   RoPE + MQA (`kv_heads = 1`) + sliding window + untied head
//!   (Pre-LN). Exercises the gated `c_gate` branch, the RMSNorm slow
//!   shim, the parallel-residual reconvergence, the RoPE Q/K rotation,
//!   `repeat_kv`, the banded mask, and the independent `lm_head` Var.
//! - **sink B** — Post-LN + ALiBi (sequential residual, LayerNorm,
//!   GELU). Exercises the post-norm wiring and the additive score
//!   bias, plus the tied-head path on a wpe-less model.
//!
//! A third gate covers the custom × MoE composition (norm / pos / GQA
//! axes over the MoE feed-forward) now that the build accepts it as
//! long as the dense-MLP knobs stay at the reference.

mod common;

use algocline_nn::arch::{
    Activation, Gpt2Config, Gpt2Custom, Gpt2Model, MoeConfig, NormKind, NormPlacement, PosKind,
    ResidualKind,
};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn tiny_cfg(moe: Option<MoeConfig>, custom: Gpt2Custom) -> Gpt2Config {
    Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16, // head_dim 8 — even, RoPE-compatible
        ctx: 8,
        vocab: 32,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe,
        custom: Some(custom),
    }
}

/// Same shift + mask discipline as the sibling gates: scored
/// positions at the causal tail so every parameter participates.
fn masked_backward(cfg: &Gpt2Config) -> (VarMap, candle_core::backprop::GradStore) {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(cfg, vb).expect("build custom gpt-2");

    let row: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    let mask: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let seq = row.len();
    let inputs = Tensor::from_slice(&row[..seq - 1], (1, seq - 1), &cfg.device).unwrap();
    let targets = Tensor::from_slice(&row[1..], (1, seq - 1), &cfg.device).unwrap();
    let mask_t = Tensor::from_slice(&mask[1..], (1, seq - 1), &cfg.device).unwrap();

    let (logits, aux) = model.forward_with_aux(&inputs).expect("forward");
    let ce = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask_t))
        .expect("loss");
    let total = match (aux, &cfg.moe) {
        // The aux term joins the loss so the router's P_i path stays
        // part of the certified surface (same rationale as the MoE
        // gate).
        (Some(aux), Some(moe)) => (ce + (aux * moe.alpha).unwrap()).expect("total loss"),
        (None, None) => ce,
        (aux, moe) => panic!(
            "aux presence {} / moe config presence {} mismatch",
            aux.is_some(),
            moe.is_some()
        ),
    };
    let grads = total.backward().expect("backward");
    (vm, grads)
}

#[test]
fn kitchen_sink_a_backward_reaches_every_base_var() {
    let cfg = tiny_cfg(
        None,
        Gpt2Custom {
            act: Activation::SwiGlu,
            norm: NormKind::RmsNorm,
            residual: ResidualKind::Parallel,
            mlp_ratio: 2,
            placement: NormPlacement::PreLn,
            pos: PosKind::Rope,
            kv_heads: Some(1), // MQA — c_attn out = 16 + 2·8 = 32
            window: Some(4),
            untied_head: true,
            cond_slots: None,
            allowed_input: false,
        },
    );
    let (vm, grads) = masked_backward(&cfg);

    // Inventory: wte + lm_head (untied) + NO wpe (RoPE) + 2 blocks ×
    // (ln_1.weight [RMS, no bias] + c_attn w/b + c_proj w/b +
    // ln_2.weight + mlp.c_fc w/b + mlp.c_gate w/b + mlp.c_proj w/b
    // = 12) + ln_f.weight = 27 Vars.
    common::assert_full_grad_coverage(&vm, &grads, 27);
}

#[test]
fn kitchen_sink_b_post_ln_alibi_backward_reaches_every_base_var() {
    let cfg = tiny_cfg(
        None,
        Gpt2Custom {
            placement: NormPlacement::PostLn,
            pos: PosKind::Alibi,
            ..Default::default()
        },
    );
    let (vm, grads) = masked_backward(&cfg);

    // Inventory: wte + NO wpe (ALiBi) + 2 blocks × (ln_1 w/b +
    // c_attn w/b + c_proj w/b + ln_2 w/b + mlp.c_fc w/b +
    // mlp.c_proj w/b = 12) + ln_f w/b = 27 Vars.
    common::assert_full_grad_coverage(&vm, &grads, 27);
}

#[test]
fn custom_moe_composition_backward_reaches_every_base_var() {
    let cfg = tiny_cfg(
        Some(MoeConfig::dense_mixture(2)),
        Gpt2Custom {
            norm: NormKind::RmsNorm,
            pos: PosKind::Rope,
            kv_heads: Some(1),
            ..Default::default() // act Gelu / ratio reference — required with MoE
        },
    );
    let (vm, grads) = masked_backward(&cfg);

    // Inventory: wte + NO wpe (RoPE) + 2 blocks × (ln_1.weight [RMS]
    // + c_attn w/b + c_proj w/b + ln_2.weight + router.weight +
    // 2 experts × (c_fc w/b + c_proj w/b) = 8 → 15) + ln_f.weight
    // = 32 Vars.
    common::assert_full_grad_coverage(&vm, &grads, 32);
}
