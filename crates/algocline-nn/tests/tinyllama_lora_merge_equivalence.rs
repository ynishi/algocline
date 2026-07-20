//! Integration test — TinyLlama LoRA wrap freeze + dispatch.
//!
//! Design source: TinyLlama LoRA Layer 2 design §6.1, plus the Layer 3
//! prep S0 fix that adopted canonical Hu et al. 2021 §4.1 init in
//! `LoraLinear::wrap` (`lora_a` random, `lora_b` zero).
//!
//! Under canonical init `ΔW = scaling * (B · A) = 0` at construction,
//! so the wrap is an identity map at `t=0`. Two properties get their
//! own test:
//!
//!   1. **Identity at init** — `wrap → forward` matches the pre-wrap
//!      baseline forward bit-for-bit (design §6.1 step 6, now with the
//!      init direction restored to the paper).
//!   2. **Dispatch reaches the LoRA leg** — after `wrap`, manually
//!      overwrite every `lora_b.weight` Var in the returned LoRA
//!      `VarMap` with a small non-zero constant, re-forward, and
//!      assert the output diverges from the baseline (design §6.1
//!      step 7). Poking through `VarMap::data()` keeps the test
//!      independent of `Block`'s private field layout — the mechanism
//!      is the same one Layer 3's training loop will use (both walk
//!      the LoRA `VarMap`).

use algocline_nn::arch::{LoraConfig, TinyLlamaConfig, TinyLlamaModel};
use candle_core::{DType, Device, Tensor};
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

/// Design §6.1 step 6 — canonical property post-S0: with `lora_b`
/// zero-init per Hu et al. 2021 §4.1, `ΔW = B · A = 0` at construction
/// so `wrap → forward` must match the pre-wrap baseline bit-for-bit.
/// A regression here would either mean (a) `lora_b` init drifted off
/// zero, or (b) `LinearVariant::Lora::forward` mixes in something other
/// than `base(x) + scaling * B(A(x))` even with `B = 0`.
#[test]
fn wrap_lora_initial_delta_is_zero_matches_baseline() {
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

    // Wrap. `lora_a` gets Kaiming random weights, `lora_b` is zero.
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let _lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

    let y_wrapped: Vec<f32> = model
        .forward(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(y_baseline.len(), y_wrapped.len());
    assert_eq!(y_baseline.len(), 5 * cfg.vocab);
    assert_eq!(
        y_baseline, y_wrapped,
        "canonical zero-init B must make wrapped forward bit-identical \
         to baseline; observed a divergence — either lora_b drifted off \
         zero or LinearVariant::Lora::forward has an extra additive term"
    );
}

/// Design §6.1 step 7 — dispatch verification. With `B = 0` at init the
/// wrap is silent, so we can't tell from a bare `wrap → forward` round-
/// trip whether `LinearVariant::Lora::forward` actually threads through
/// the LoRA leg or is a hidden no-op that just calls the base. Force
/// the question by overwriting every `lora_b.weight` Var in the LoRA
/// `VarMap` with a small non-zero constant (this is exactly the
/// mechanism Layer 3's optimizer will use) and asserting the re-
/// forwarded output diverges. The mutation is deterministic (0.01·1)
/// so a bit-for-bit match here can only mean dispatch is broken.
#[test]
fn wrap_lora_dispatch_reaches_lora_leg_via_lora_b_poke() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    let ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5], (1, 5), &cfg.device).unwrap();
    let y_baseline: Vec<f32> = model
        .forward(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

    // Overwrite every lora_b.weight Var with 0.01. All other LoRA
    // vars (lora_a.weight) keep their random init.
    let data = lora_vm.data();
    let guard = data.lock().unwrap();
    let mut poked = 0usize;
    for (name, var) in guard.iter() {
        if name.ends_with(".lora_b.weight") || name == "lora_b.weight" {
            let shape = var.as_tensor().shape().clone();
            let nonzero = Tensor::ones(shape, DType::F32, &Device::Cpu)
                .unwrap()
                .affine(0.01, 0.0)
                .unwrap();
            var.set(&nonzero).expect("Var::set on lora_b must succeed");
            poked += 1;
        }
    }
    drop(guard);
    // 2 layers * 7 wrapped modules = 14 lora_b Vars expected.
    assert_eq!(
        poked, 14,
        "expected to overwrite 14 lora_b.weight Vars (2 layers * 7 targets); \
         name filter drift?"
    );

    let y_poked: Vec<f32> = model
        .forward(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(y_baseline.len(), y_poked.len());
    let mut max_abs_diff = 0.0_f32;
    for (a, b) in y_baseline.iter().zip(y_poked.iter()) {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff > 1e-6,
        "post-poke wrapped forward matched baseline exactly \
         (max_abs_diff = {max_abs_diff}) — LinearVariant::Lora::forward \
         is not routing through the low-rank leg"
    );
}
