//! Gradient-coverage gate for the GPT-2 trainable path.
//!
//! The loss-threshold tests (`tiny_overfit_reduces_loss`,
//! `distill_run_reduces_masked_loss_on_teacher_corpus`, the teacher-card
//! E2E) assert that training descends, but on a tied-head model descent
//! alone cannot localize *which* parameters learned: if the autograd
//! graph is severed inside the blocks (the candle-nn 0.11 LayerNorm
//! fast-path cliff that `apply_slow_layer_norm` shims around), the tied
//! `wte` LM head still receives gradient through the final logits matmul
//! and can memorize a repeated corpus on its own, keeping those tests
//! green while every block parameter stays frozen.
//!
//! This test closes that hole mechanically: one masked-CE backward pass
//! on a tiny from-scratch GPT-2 must produce a non-zero gradient for
//! EVERY Var registered in the model's VarMap — both embeddings, every
//! per-block projection and LayerNorm affine, and the final LayerNorm.
//! A future candle bump that reintroduces a no-backward `CustomOp` (or a
//! forward refactor that routes around the slow-path shim) fails here
//! with the offending parameter names, instead of surfacing as a
//! silently weaker training run. Sibling in spirit to
//! `rms_norm_autograd_gate.rs` (TinyLlama side), but scoped to the whole
//! GPT-2 parameter inventory rather than a single op.

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

#[test]
fn masked_backward_reaches_every_base_var() {
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
    };
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build tiny gpt-2");

    // Same shift + mask discipline as the training loop: inputs are the
    // row minus its last token, targets the row minus its first, and the
    // shifted mask scores the last four targets (a distill-shaped
    // prompt/response split). Scored positions sit at the causal tail,
    // so every earlier position — and therefore every parameter —
    // participates in the scored logits.
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

    // Parameter inventory: wte + wpe + 2 blocks × (ln_1 w/b + c_attn w/b
    // + c_proj w/b + ln_2 w/b + mlp.c_fc w/b + mlp.c_proj w/b = 12) +
    // ln_f w/b = 28 Vars. Pinned so the per-Var loop below cannot go
    // vacuous if the arch registration changes shape.
    let data = vm.data().lock().unwrap();
    assert_eq!(
        data.len(),
        28,
        "GPT-2 tiny VarMap inventory drifted; update the count in this test"
    );

    let mut missing: Vec<String> = Vec::new();
    let mut zero: Vec<String> = Vec::new();
    for (name, var) in data.iter() {
        match grads.get(var.as_tensor()) {
            None => missing.push(name.clone()),
            Some(g) => {
                let mag: f32 = g.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
                // The explicit NaN arm keeps a NaN gradient from slipping
                // through the `<=` comparison.
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
        "autograd coverage hole — the loss can still descend head-only while \
         these parameters never learn.\n  missing from GradStore: {missing:?}\n  \
         zero/NaN gradient: {zero:?}"
    );

    // Fused-projection blind spot: the whole-Var check above cannot see
    // a severed attention-scores path, because the V slice of the fused
    // `c_attn` keeps the Var's total gradient non-zero even when Q and
    // K receive nothing (the candle-nn 0.11 `softmax_last_dim`
    // no-backward cliff hid here until the MoE router — whose only
    // gradient path is its softmax — exposed it). Row layout of
    // `c_attn.weight` is `[3·dim, dim]` = Q rows, K rows, V rows, so
    // slice-level magnitudes pin the scores path directly.
    let dim = 16;
    for (name, var) in data.iter() {
        if !name.ends_with("attn.c_attn.weight") {
            continue;
        }
        let g = grads.get(var.as_tensor()).expect("checked above");
        for (slice, label) in [(0usize, "Q"), (1, "K")] {
            let mag: f32 = g
                .narrow(0, slice * dim, dim)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(
                mag > 0.0 && mag.is_finite(),
                "{name}: {label} rows of the fused c_attn gradient are zero \
                 (sum|g|={mag}) — the attention-scores softmax path is severed"
            );
        }
    }
}
