//! Integration test — LoRA merge equivalence.
//!
//! The subtask invariant for LoRA is that a wrapped forward
//!
//! ```text
//! y_lora = base(x) + scaling * lora_b(lora_a(x))
//! ```
//!
//! is element-wise close to a "merged" forward
//!
//! ```text
//! y_merged = Linear(base_weight + scaling * lora_b_weight @ lora_a_weight, base_bias)(x)
//! ```
//!
//! within 1e-4 across a small batch. That guarantee is what lets an
//! inference-only downstream consumer collapse the two low-rank
//! matrices back into a single Linear without behaviour drift.
//!
//! The test uses hand-set weights (rather than a `VarBuilder`
//! initialiser) so its outcome is fully deterministic regardless of
//! candle's RNG state.

use algocline_nn::arch::{max_abs_diff_f32, Gpt2Config, Gpt2Model, LoraConfig, LoraLinear};
use algocline_nn::train::{
    run_full_ft, run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, Loss, ScheduleKind,
    TokenizedDataset, TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use std::sync::Arc;
use tempfile::TempDir;

fn tensor2(vals: &[f32], rows: usize, cols: usize, dev: &Device) -> Tensor {
    Tensor::from_slice(vals, (rows, cols), dev).unwrap()
}

#[test]
fn wrapped_forward_matches_merged_linear_within_tolerance() {
    let device = Device::Cpu;

    // Base linear: 5 -> 4, deterministic values.
    let base_w = tensor2(
        &[
            0.10f32, 0.20, 0.30, 0.40, 0.50, //
            0.60, 0.70, 0.80, 0.90, 1.00, //
            1.10, 1.20, 1.30, 1.40, 1.50, //
            1.60, 1.70, 1.80, 1.90, 2.00, //
        ],
        4,
        5,
        &device,
    );
    let base_b = Tensor::from_slice(&[0.01f32, -0.02, 0.03, -0.04], (4,), &device).unwrap();
    let base = Linear::new(base_w, Some(base_b));

    // LoRA legs at rank = 2 (2 <= min(4, 5) so wrap must succeed).
    let a_w = tensor2(
        &[
            0.01f32, 0.02, 0.03, 0.04, 0.05, //
            0.06, 0.07, 0.08, 0.09, 0.10, //
        ],
        2,
        5,
        &device,
    );
    let b_w = tensor2(
        &[
            0.11f32, 0.12, //
            0.13, 0.14, //
            0.15, 0.16, //
            0.17, 0.18, //
        ],
        4,
        2,
        &device,
    );
    let lora_a = Linear::new(a_w, None);
    let lora_b = Linear::new(b_w, None);
    let scaling = LoraConfig::new(2, 4.0).scaling();

    let lora = LoraLinear::from_parts(base, lora_a, lora_b, scaling);

    // Build a plain merged Linear and compare its forward against the
    // LoRA-wrapped forward across a batch of inputs.
    let merged_w = lora.merged_weight().unwrap();
    let merged = Linear::new(merged_w, lora.base().bias().cloned());

    let xs = tensor2(
        &[
            0.5f32, -0.5, 1.5, -1.5, 2.5, //
            -2.5, 0.0, 3.0, -3.0, 0.25, //
            0.75, -0.75, 1.25, -1.25, 0.10, //
        ],
        3,
        5,
        &device,
    );

    let y_lora = lora.forward(&xs).unwrap();
    let y_merged = merged.forward(&xs).unwrap();
    assert_eq!(y_lora.dims(), y_merged.dims());
    let diff = max_abs_diff_f32(&y_lora, &y_merged).unwrap();
    assert!(
        diff < 1e-4,
        "LoRA / merged forwards diverged by {diff} (tolerance = 1e-4)"
    );
}

#[test]
fn merged_weight_shape_matches_base() {
    let device = Device::Cpu;
    let base_w = Tensor::zeros((6, 8), DType::F32, &device).unwrap();
    let base = Linear::new(base_w, None);
    let a_w = Tensor::zeros((3, 8), DType::F32, &device).unwrap();
    let b_w = Tensor::zeros((6, 3), DType::F32, &device).unwrap();
    let lora = LoraLinear::from_parts(
        base,
        Linear::new(a_w, None),
        Linear::new(b_w, None),
        LoraConfig::new(3, 6.0).scaling(),
    );
    let merged = lora.merged_weight().unwrap();
    assert_eq!(merged.dims(), &[6, 8]);
}

/// Base-frozen invariant:
/// running [`run_lora_ft`] must leave the base parameters bit-identical
/// before and after training. Only the freshly-created LoRA A/B legs
/// are handed to AdamW; the base varmap never sees the optimizer.
///
/// We snapshot every base tensor as an `f32` vector before the run and
/// re-read them after the run — any drift indicates the frozen-base
/// contract has broken.
#[test]
fn run_lora_ft_leaves_base_weights_bit_identical() {
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
        custom: None,
    };
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = Gpt2Model::new(&cfg, vs).unwrap();

    // Snapshot base tensors before training.
    let base_before: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let base_var_count = base_vm.all_vars().len();

    let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(50).collect();
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
        steps: 10,
        warmup: 2,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
        init_from: None,
        mask_disallowed_logits: false,
        ..FullFtConfig::default()
    };
    let lora_cfg = LoraConfig::new(4, 8.0);
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let ckpt = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss,
        tmp.path(),
        "invariant-1",
        lease,
    )
    .expect("run_lora_ft must succeed");

    // Terminal Δ file lives under `<ckpt_dir>/nn/lora-<card>.safetensors`.
    let delta_path = tmp.path().join("nn").join("lora-invariant-1.safetensors");
    assert!(
        delta_path.exists(),
        "expected Δ ckpt at {}",
        delta_path.display()
    );
    // Δ file is tiny (16 vars × ~1 KB each on this micro model). The
    // 20 MB size cap from invariant #3 is exercised on the full
    // GPT-2 medium build; here we just guard that the file is not
    // accidentally storing the full base model.
    let delta_size = std::fs::metadata(&delta_path).unwrap().len();
    assert!(
        delta_size < 100_000,
        "Δ ckpt is unexpectedly large: {delta_size} bytes"
    );
    assert!(ckpt.train_loss.is_finite());

    // Base var count unchanged.
    assert_eq!(base_vm.all_vars().len(), base_var_count);

    // Every base tensor bit-identical to its pre-training snapshot.
    let base_after: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    for (idx, (before, after)) in base_before.iter().zip(base_after.iter()).enumerate() {
        assert_eq!(
            before, after,
            "base var index {idx} drifted through run_lora_ft"
        );
    }
}

/// Learning invariant #2 (LoRA loss reduction):
/// [`run_lora_ft`] must reduce loss on an overfit corpus. Mirrors
/// [`crate::train::run_full_ft`]'s `tiny_overfit_reduces_loss` so the
/// LoRA path is covered by the same shape of evidence: baseline loss
/// captured *before* wrap+train, then `min_train_loss` from the
/// returned [`Checkpoint`] must fall to `< 0.75 * baseline`.
///
/// A "form runs, learning zero" bug would leave `min_train_loss` at
/// baseline (no update path through the LoRA legs). This test catches
/// exactly that failure mode.
#[test]
fn run_lora_ft_reduces_loss_on_overfit_corpus() {
    // `dim = 64` (rather than the 16 the sibling tests in this file use)
    // is load-bearing: GPT-2 ties its LM head to `wte`, which `run_lora_ft`
    // keeps frozen, so the adapters can only steer the hidden state into
    // the right embedding row. At `dim = 16` that steering saturates ~8%
    // above the entropy floor and the assertion below cannot separate
    // "learning" from "not learning".
    let cfg = Gpt2Config {
        layers: 2,
        heads: 4,
        dim: 64,
        ctx: 8,
        vocab: 32,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: None,
    };
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = Gpt2Model::new(&cfg, vs).unwrap();

    // Baseline: forward loss on the exact input the training loop will
    // see, captured BEFORE wrap_lora + training.
    let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let loss_fn = CrossEntropyLoss::new();
    let baseline = {
        let inputs = Tensor::from_vec(row.clone(), (1, 8), &cfg.device)
            .unwrap()
            .narrow(1, 0, 7)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .contiguous()
            .unwrap();
        let targets = Tensor::from_vec(row.clone(), (1, 8), &cfg.device)
            .unwrap()
            .narrow(1, 1, 7)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .contiguous()
            .unwrap();
        let logits = model.forward(&inputs).unwrap();
        loss_fn
            .compute(&logits, &targets, None)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };

    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(400).collect();
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
    let lora_cfg = LoraConfig::new(4, 8.0);
    let train_cfg = FullFtConfig {
        lr: 8e-3,
        batch_size: 1,
        grad_accum: 1,
        steps: 150,
        warmup: 5,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
        init_from: None,
        mask_disallowed_logits: false,
        ..FullFtConfig::default()
    };
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let ckpt = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss_fn,
        tmp.path(),
        "loss-reduce",
        lease,
    )
    .expect("run_lora_ft must succeed");

    // Threshold calibration: `baseline` sits at the vocabulary's entropy
    // floor (`ln(32) ~= 3.47`) because `Gpt2Model::new` draws `wte` from
    // `N(0, 0.02)`, so the assertion measures real learning rather than
    // the decay of an over-wide init. LoRA on this shape plateaus at
    // `min/baseline ~= 0.62-0.67` (10 draws, 150 and 300 steps alike —
    // more steps buy nothing), so 0.75 keeps a working margin while
    // still failing loudly if the adapters stop receiving gradient.
    let min_loss = *ckpt.metrics.get("min_train_loss").expect("min_train_loss");
    assert!(
        min_loss < baseline * 0.75,
        "run_lora_ft did not reduce loss: min_train_loss={min_loss}, baseline={baseline}, \
         threshold=0.75*baseline={} (LoRA training may not be updating params)",
        baseline * 0.75
    );
    assert!(
        ckpt.train_loss.is_finite(),
        "final train loss must be finite: {}",
        ckpt.train_loss
    );
}

/// Learning invariant #3 (LoRA weight movement):
/// [`Gpt2Model::wrap_lora`] + training via [`run_full_ft`] on the
/// returned LoRA `VarMap` must materially update at least one LoRA
/// tensor. Complements the loss-reduction test by asserting the
/// mechanism (Var updates) rather than the outcome (loss down).
///
/// Uses `run_full_ft(&model, &lora_vm, ...)` after manual `wrap_lora`
/// so the test can snapshot `lora_vm.all_vars()` before/after. This
/// exercises the same `run_ft_core` code path `run_lora_ft` takes
/// internally (both dispatch to `run_ft_core` with `opt_vm == save_vm
/// == lora_vm`), just with the wrap step surfaced so before/after
/// state is inspectable.
///
/// A "backward step doesn't reach LoRA vars" bug leaves
/// `lora_before == lora_after` for every tensor; that condition is
/// exactly what this test asserts against.
#[test]
fn run_lora_ft_updates_lora_weights() {
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
        custom: None,
    };
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = Gpt2Model::new(&cfg, vs).unwrap();

    let lora_cfg = LoraConfig::new(4, 8.0);
    let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

    // Snapshot LoRA tensors BEFORE training (read straight off the Var
    // storage — this is what AdamW updates).
    let lora_before: Vec<Vec<f32>> = lora_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    assert!(
        !lora_before.is_empty(),
        "wrap_lora should have registered LoRA vars"
    );

    let row: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(100).collect();
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
    let loss_fn = CrossEntropyLoss::new();
    let train_cfg = FullFtConfig {
        lr: 8e-3,
        batch_size: 1,
        grad_accum: 1,
        steps: 30,
        warmup: 2,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
        init_from: None,
        mask_disallowed_logits: false,
        ..FullFtConfig::default()
    };
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());

    // Drive training directly against `lora_vm`. Equivalent to what
    // `run_lora_ft` does internally (`run_ft_core(base, &lora_vm,
    // &lora_vm, ...)`) minus the wrap step (already done above).
    let _ckpt = run_full_ft(
        &model,
        &lora_vm,
        &mut ds,
        &train_cfg,
        &loss_fn,
        tmp.path(),
        "weight-update",
        lease,
        None,
    )
    .expect("run_full_ft on lora_vm must succeed");

    let lora_after: Vec<Vec<f32>> = lora_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();

    // At least one tensor must have moved. `before == after` for every
    // tensor is exactly the "199 backward steps updated nothing" signal
    // observed on GPU (Phase 2 verify report).
    let mut changed_count = 0usize;
    let mut inspected_count = 0usize;
    for (before, after) in lora_before.iter().zip(lora_after.iter()) {
        inspected_count += 1;
        if before != after {
            changed_count += 1;
        }
    }
    assert!(
        changed_count > 0,
        "run_full_ft on lora_vm left ALL {inspected_count} LoRA tensors bit-identical \
         before/after training — AdamW is not updating LoRA Vars (backward path or \
         Var/Tensor identity is broken)"
    );
}

#[test]
fn zero_lora_deltas_leave_base_output_unchanged() {
    // If both LoRA matrices are zero, the wrap must be a no-op:
    // `y = base(x) + scaling * 0 = base(x)`.
    let device = Device::Cpu;
    let base_w = tensor2(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &device);
    let base_b = Tensor::from_slice(&[0.1f32, 0.2], (2,), &device).unwrap();
    let base = Linear::new(base_w, Some(base_b));
    let base_ref = Linear::new(base.weight().clone(), base.bias().cloned());

    let a_w = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
    let b_w = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
    let lora = LoraLinear::from_parts(
        base,
        Linear::new(a_w, None),
        Linear::new(b_w, None),
        LoraConfig::new(2, 4.0).scaling(),
    );

    let xs = tensor2(&[1.0f32, -1.0, 0.5, 0.5, 0.25, -0.25], 2, 3, &device);
    let y_lora = lora.forward(&xs).unwrap();
    let y_base = base_ref.forward(&xs).unwrap();
    let diff = max_abs_diff_f32(&y_lora, &y_base).unwrap();
    assert!(diff < 1e-6, "zero-delta LoRA changed base output by {diff}");
}
