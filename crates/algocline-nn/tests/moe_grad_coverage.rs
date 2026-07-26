//! Gradient-coverage gate for the dense-MoE GPT-2 path.
//!
//! MoE version of `gpt2_grad_coverage.rs` (same rationale: a tied-head
//! model can descend head-only while block parameters stay frozen, so
//! descent alone cannot certify the autograd graph). One backward pass
//! over `masked-CE + α·aux` must produce a non-zero gradient for EVERY
//! Var in the model — the router and every expert included.
//!
//! Runs in **dense-mixture mode** (`top_k = n_experts`): with top-k
//! routing an expert that is never selected in a step legitimately
//! receives no gradient, which is sparsity, not a severed graph. The
//! gate's responsibility is structural reachability — "can gradient
//! reach this parameter at all" — so it disables the selection; whether
//! top-k routing actually exercises every expert over a training run is
//! the utilization observation in `examples/moe_router_probe.rs`, not a
//! pass/fail here.
//!
//! The aux term joins the loss because the router's `P_i` path (mean
//! router probability) is part of the trainable surface the
//! load-balancing loss exists to shape; leaving it out would let a
//! future refactor sever it silently.

use algocline_nn::arch::{Gpt2Config, Gpt2Model, MoeConfig};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

#[test]
fn moe_masked_backward_reaches_every_base_var() {
    let n_experts = 2;
    let cfg = Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab: 32,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: Some(MoeConfig::dense_mixture(n_experts)),
    };
    let alpha = cfg.moe.as_ref().unwrap().alpha;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build tiny moe gpt-2");

    // Same shift + mask discipline as the dense gate: scored positions
    // sit at the causal tail so every earlier position — and therefore
    // every parameter — participates in the scored logits.
    let row: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    let mask: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let seq = row.len();
    let inputs = Tensor::from_slice(&row[..seq - 1], (1, seq - 1), &cfg.device).unwrap();
    let targets = Tensor::from_slice(&row[1..], (1, seq - 1), &cfg.device).unwrap();
    let mask_t = Tensor::from_slice(&mask[1..], (1, seq - 1), &cfg.device).unwrap();

    let (logits, aux) = model.forward_with_aux(&inputs).expect("forward");
    let aux = aux.expect("MoE model must return an aux term");
    let ce = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask_t))
        .expect("loss");
    let total = (ce + (aux * alpha).unwrap()).expect("total loss");
    let grads = total.backward().expect("backward");

    // Parameter inventory: wte + wpe + 2 blocks × (ln_1 w/b + c_attn w/b
    // + c_proj w/b + ln_2 w/b = 8, router.weight = 1, 2 experts ×
    // (c_fc w/b + c_proj w/b) = 8 → 17) + ln_f w/b = 38 Vars. Pinned so
    // the per-Var loop below cannot go vacuous if the arch registration
    // changes shape.
    let data = vm.data().lock().unwrap();
    assert_eq!(
        data.len(),
        38,
        "MoE tiny VarMap inventory drifted; update the count in this test"
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
}
