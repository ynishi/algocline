//! Gradient-coverage gate for the TinyLlama trainable path.
//!
//! Sibling of `gpt2_grad_coverage.rs` with one architecture-specific
//! reason to exist: TinyLlama keeps `q_proj` / `k_proj` as standalone
//! parameters whose ONLY gradient path runs through the attention
//! softmax. GPT-2's fused `c_attn` masks a severed scores path (the V
//! slice keeps the fused weight's gradient non-zero), but here a
//! no-backward softmax kernel shows up directly as `q_proj` / `k_proj`
//! missing from the GradStore. This is exactly how the candle-nn 0.11
//! `softmax_last_dim` cliff (`apply_op1_no_bwd` — same family as the
//! LayerNorm / RMSNorm fast paths) stayed invisible until the MoE
//! router hit it; the gate pins the fix (`arch::softmax_last_dim_slow`)
//! against future candle bumps and forward refactors.

use algocline_nn::arch::{TinyLlamaConfig, TinyLlamaModel};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};

#[test]
fn masked_backward_reaches_every_base_var() {
    let cfg = TinyLlamaConfig::tiny();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = TinyLlamaModel::new(&cfg, vb).expect("build tiny tinyllama");

    // Same shift + mask discipline as the GPT-2 gate: scored positions
    // at the causal tail so every parameter participates.
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

    // Parameter inventory: embed_tokens + lm_head + final norm +
    // 2 blocks × (input_layernorm + post_attention_layernorm +
    // q/k/v/o_proj + gate/up/down_proj = 9) = 21 Vars. Pinned so the
    // per-Var loop below cannot go vacuous if the arch registration
    // changes shape.
    let data = vm.data().lock().unwrap();
    assert_eq!(
        data.len(),
        21,
        "TinyLlama tiny VarMap inventory drifted; update the count in this test"
    );

    let mut missing: Vec<String> = Vec::new();
    let mut zero: Vec<String> = Vec::new();
    for (name, var) in data.iter() {
        match grads.get(var.as_tensor()) {
            None => missing.push(name.clone()),
            Some(g) => {
                let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
                if mag.is_nan() || mag <= 0.0 {
                    zero.push(format!("{name} (sum|g|={mag})"));
                }
            }
        }
    }
    missing.sort();
    zero.sort();
    assert!(
        missing.is_empty() && zero.is_empty(),
        "autograd coverage hole — the loss can still descend while these \
         parameters never learn.\n  missing from GradStore: {missing:?}\n  \
         zero/NaN gradient: {zero:?}"
    );
}
