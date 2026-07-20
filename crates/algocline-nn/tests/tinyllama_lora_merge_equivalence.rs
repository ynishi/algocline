//! Integration test — TinyLlama LoRA wrap freeze + dispatch.
//!
//! Design source: `workspace/tasks/alc-nn-tinyllama/layer-2-lora-design.md`
//! §6.1. Two deviations from that spec are called out below.
//!
//! **Deviation 1** — Design step 6 ("forward with fresh `lora_a=0` init
//! matches pre-wrap snapshot") has the LoRA init direction reversed.
//! Canonical LoRA per Hu et al. 2021 §4.1 zeros **B**, not A, so
//! `ΔW = B·A = 0` at t=0. `LoraLinear::wrap` currently uses candle-nn's
//! default (`xavier` / `kaiming`) for both legs, so neither is zero at
//! init and no natural "wrap is identity at init" property exists to
//! verify from integration scope. That property lives one of two ways
//! as a follow-up:
//!   (a) change `LoraLinear::wrap` to zero-init `lora_b` per the paper
//!       (scope expansion — affects GPT-2 too), or
//!   (b) expose a public accessor so a test can reach the wrapped
//!       `LoraLinear` legs and manually zero `lora_b` post-wrap.
//! Neither is landed here; the freeze invariant + dispatch divergence
//! together cover the wrap semantics that Layer 2 promises.
//!
//! **Deviation 2** — Design step 7 ("manually set `lora_a` = nonzero →
//! forward diverges") requires reaching into private `Block` fields.
//! Achieved here without private-state access by observing that
//! `LoraLinear::wrap`'s default init already leaves A and B non-zero,
//! so `wrap_lora` → `forward` diverges from the baseline forward at
//! initialisation. That covers the underlying hypothesis
//! (`LinearVariant::Lora` actually dispatches through the low-rank leg
//! — not a silent no-op) via a bare `wrap → forward` round-trip.

use algocline_nn::arch::{LoraConfig, TinyLlamaConfig, TinyLlamaModel};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};

/// Steps 1, 3, 4, 5 of design §6.1: build tiny model, wrap, assert
/// registration count = 2 layers × 7 wrapped × (a + b) = 28, and
/// prove the base [`VarMap`] is byte-identical before and after wrap.
#[test]
fn wrap_lora_registers_28_vars_and_freezes_base() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    // Snapshot every base tensor as f32 BEFORE wrap.
    let base_before: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let base_var_count = base_vm.all_vars().len();

    // rank = 4 is safely within min(in, out) on every projection of the
    // tiny preset (k/v_proj is the narrowest at out = kv_heads *
    // head_dim = 1 * 32 = 32).
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora must succeed");

    // Registration count.
    assert_eq!(
        lora_vm.all_vars().len(),
        28,
        "expected 2 layers * 7 wrapped * 2 = 28 new LoRA Vars"
    );

    // Base var count unchanged (no new registrations on the base
    // varmap; the LoRA VarBuilder is scoped to `lora_vm`).
    assert_eq!(base_vm.all_vars().len(), base_var_count);

    // Base tensor bytes bit-identical (freeze invariant — the same
    // property that lets Layer 3's training loop hand ONLY `lora_vm`
    // to AdamW and be sure the base weights don't drift).
    let base_after: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    for (idx, (before, after)) in base_before.iter().zip(base_after.iter()).enumerate() {
        assert_eq!(before, after, "wrap_lora perturbed base var index {idx}");
    }
}

/// Design §6.1 steps 2 + 7 (adapted, see deviation 2 in the module
/// doc): capture the baseline forward, apply `wrap_lora` with default
/// init (non-zero A and B), then re-forward on the same input. The
/// wrapped output MUST differ from the baseline — otherwise
/// `LinearVariant::Lora`'s `Module::forward` is silently dispatching
/// only to the base and the low-rank leg never contributes.
#[test]
fn wrap_lora_alters_forward_through_linear_variant_dispatch() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();

    // Baseline forward BEFORE wrap.
    let y_baseline: Vec<f32> = model
        .forward(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Wrap and re-forward on the same input.
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let _lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

    let y_wrapped: Vec<f32> = model
        .forward(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Shape unchanged: `[1, 5, vocab]` flattened.
    assert_eq!(y_baseline.len(), y_wrapped.len());
    assert_eq!(y_baseline.len(), 5 * cfg.vocab);

    // Something must differ. If every element matches bit-for-bit, either
    // (a) the wrap is not being threaded through Block::attention / mlp
    //     (silent no-op via a LinearVariant Module dispatch bug), or
    // (b) candle-nn's default `Linear` init produced exactly-zero weights
    //     (impossible with kaiming / xavier).
    let mut max_abs_diff = 0.0_f32;
    for (a, b) in y_baseline.iter().zip(y_wrapped.iter()) {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff > 1e-6,
        "wrapped forward matched baseline exactly (max_abs_diff = {max_abs_diff}) \
         — LoRA dispatch through LinearVariant::Lora may be a silent no-op or \
         default init is unexpectedly zero"
    );
}
