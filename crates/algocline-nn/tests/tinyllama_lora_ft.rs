//! Integration test — TinyLlama LoRA fine-tune loop end-to-end.
//!
//! Empirical validation that the Layer 3 generic training loop
//! (S1-S3: `Module` + `DeviceView` + `LoraWrappable` bounds on
//! `run_ft_core` / `run_full_ft` / `run_lora_ft`) actually drives a
//! second architecture — TinyLlama — through the same trainer entry
//! points that ship for GPT-2. Every test here mirrors a counterpart
//! in `tests/lora_merge_equivalence.rs` (the GPT-2 side); a green
//! `cargo test` here is the "Layer 3 works on Llama-family" proof.
//!
//! Design source: TinyLlama Layer 3 training-loop design §5.1
//! (integration) + §6 S4 — the generic `run_ft_core<M>` /
//! `run_full_ft<M>` / `run_lora_ft<M>` entry points introduced by
//! Layer 3 S1-S3 with `M: Module + DeviceView [+ LoraWrappable]`.
//!
//! All tests use `TinyLlamaConfig::tiny()` (2 layers, 2 heads,
//! 1 kv_head, dim 64, hidden 128, ctx 16, vocab 32, F32/CPU) so the
//! runtime stays friendly to `cargo test`.

use algocline_nn::arch::{LoraConfig, TinyLlamaConfig, TinyLlamaModel};
use algocline_nn::train::{
    run_full_ft, run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, Loss, ScheduleKind,
    TokenizedDataset, TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;
use tempfile::TempDir;

/// Training row that fits inside `ctx = 16` (post-input/target slice
/// window is 15). Values stay inside `vocab = 32`.
fn overfit_row() -> Vec<u32> {
    vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
}

/// Learning invariant #1 (LoRA freeze — TinyLlama):
/// [`run_lora_ft`] must leave every tensor in the base `VarMap`
/// bit-identical. The wrap machinery hands only the LoRA-only VarMap
/// to AdamW (freeze is structural, not a runtime `requires_grad`
/// check), so a byte-compare on the base map before / after training
/// is the direct assertion.
///
/// A regression here would mean either (a) `TinyLlamaModel::wrap_lora`
/// silently registered a Var on the base map, or (b) the generic
/// `run_ft_core` path leaked a gradient into the base optimizer.
#[test]
fn run_lora_ft_tinyllama_leaves_base_weights_bit_identical() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    let base_before: Vec<Vec<f32>> = base_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let base_var_count = base_vm.all_vars().len();

    let row = overfit_row();
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(50).collect();
    let mut ds = TokenizedDataset::new(
        rows,
        DatasetOpts {
            batch_size: 1,
            ctx_len: 16,
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
    };
    // rank = 4 is safely inside `min(in, out)` for every projection on
    // the tiny preset (k/v_proj is narrowest at kv_heads * head_dim =
    // 1 * 32 = 32).
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let ckpt = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss,
        tmp.path(),
        "tinyllama-freeze",
        lease,
    )
    .expect("run_lora_ft must succeed");

    // Delta checkpoint lives at <ckpt_dir>/nn/lora-<card_id>.safetensors
    // — same convention as the GPT-2 path (no arch tag in filename;
    // Card metadata carries the arch).
    let delta_path = tmp
        .path()
        .join("nn")
        .join("lora-tinyllama-freeze.safetensors");
    assert!(
        delta_path.exists(),
        "expected Δ ckpt at {}",
        delta_path.display()
    );
    let delta_size = std::fs::metadata(&delta_path).unwrap().len();
    // Tiny preset: 28 LoRA vars, each a few hundred bytes. Well under
    // 100 KB; anything larger implies the base model leaked into the
    // Δ file.
    assert!(
        delta_size < 200_000,
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
            "base var index {idx} drifted through run_lora_ft (TinyLlama)"
        );
    }
}

/// Learning invariant #2 (LoRA loss reduction — TinyLlama):
/// [`run_lora_ft`] must reduce loss on an overfit corpus. Mirror of
/// the GPT-2 `run_lora_ft_reduces_loss_on_overfit_corpus` test.
/// Baseline loss is captured *before* wrap+train on the same input
/// the training loop will see, then `min_train_loss` from the returned
/// [`Checkpoint`] must fall below `0.7 * baseline`.
///
/// A "form runs, learning zero" bug (backward path broken, LoRA vars
/// never move) would leave `min_train_loss` around baseline.
#[test]
fn run_lora_ft_tinyllama_reduces_loss_on_overfit_corpus() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    // Baseline: forward loss on the exact input the training loop will
    // see, captured BEFORE wrap_lora + training.
    let row = overfit_row();
    let loss_fn = CrossEntropyLoss::new();
    let baseline = {
        let inputs = Tensor::from_vec(row.clone(), (1, 16), &cfg.device)
            .unwrap()
            .narrow(1, 0, 15)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .contiguous()
            .unwrap();
        let targets = Tensor::from_vec(row.clone(), (1, 16), &cfg.device)
            .unwrap()
            .narrow(1, 1, 15)
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
            ctx_len: 16,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
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
        "tinyllama-loss-reduce",
        lease,
    )
    .expect("run_lora_ft must succeed");

    let min_loss = *ckpt.metrics.get("min_train_loss").expect("min_train_loss");
    assert!(
        min_loss < baseline * 0.7,
        "run_lora_ft did not reduce loss on TinyLlama: \
         min_train_loss={min_loss}, baseline={baseline}, \
         threshold=0.7*baseline={} (LoRA training may not be updating \
         params via the generic loop)",
        baseline * 0.7
    );
    assert!(
        ckpt.train_loss.is_finite(),
        "final train loss must be finite: {}",
        ckpt.train_loss
    );
}

/// Learning invariant #3 (LoRA weight movement — TinyLlama):
/// [`TinyLlamaModel::wrap_lora`] + training via [`run_full_ft`] on
/// the returned LoRA `VarMap` must materially update at least one
/// LoRA tensor. Mirror of the GPT-2 `run_lora_ft_updates_lora_weights`
/// test; the assertion is deliberately "at least one" (not "all")
/// because under S0's canonical zero-init B, gradient reaches only B
/// at step 1 (A moves at step 2+) — 30 steps is plenty for at least
/// one to move.
///
/// Uses `run_full_ft(&model, &lora_vm, ...)` after manual `wrap_lora`
/// so the test can snapshot `lora_vm.all_vars()` before/after. This
/// exercises the same `run_ft_core` code path `run_lora_ft` takes
/// internally.
#[test]
fn run_lora_ft_tinyllama_updates_lora_weights() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let lora_vm = model.wrap_lora(&lora_cfg).expect("wrap_lora ok");

    let lora_before: Vec<Vec<f32>> = lora_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    assert!(
        !lora_before.is_empty(),
        "wrap_lora should have registered LoRA vars"
    );

    let row = overfit_row();
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(100).collect();
    let mut ds = TokenizedDataset::new(
        rows,
        DatasetOpts {
            batch_size: 1,
            ctx_len: 16,
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
    };
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());

    // Drive training directly against `lora_vm` — equivalent to what
    // `run_lora_ft` does internally (`run_ft_core(base, &lora_vm,
    // &lora_vm, ...)`) minus the wrap step. `run_full_ft` here
    // monomorphises to `run_full_ft::<TinyLlamaModel>` under S3's
    // generic bounds.
    let _ckpt = run_full_ft(
        &model,
        &lora_vm,
        &mut ds,
        &train_cfg,
        &loss_fn,
        tmp.path(),
        "tinyllama-weight-update",
        lease,
        None,
    )
    .expect("run_full_ft on lora_vm must succeed");

    let lora_after: Vec<Vec<f32>> = lora_vm
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().flatten_all().unwrap().to_vec1().unwrap())
        .collect();

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
         before/after training on TinyLlama — AdamW is not updating LoRA Vars via the \
         generic Module/DeviceView path"
    );
}

/// Layer 3 specific — checkpoint shape invariant:
/// The saved delta safetensors bundle at
/// `<ckpt_dir>/nn/lora-<card_id>.safetensors` must contain exactly
/// `2 layers × 7 targets × 2 (A + B) = 28` tensors on the tiny
/// preset. Guards against a future refactor accidentally widening
/// `save_vm` scope (e.g. saving the base VarMap alongside LoRA
/// tensors), which would break the "Δ file is LoRA-only" invariant
/// documented in the trait doc for `LoraWrappable`.
#[test]
fn run_lora_ft_tinyllama_saves_delta_with_expected_var_count() {
    let cfg = TinyLlamaConfig::tiny();
    let base_vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vs).unwrap();

    let row = overfit_row();
    let rows: Vec<Vec<u32>> = std::iter::repeat_with(|| row.clone()).take(20).collect();
    let mut ds = TokenizedDataset::new(
        rows,
        DatasetOpts {
            batch_size: 1,
            ctx_len: 16,
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
        steps: 5,
        warmup: 1,
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
    };
    let lora_cfg = LoraConfig::with_targets(4, 8.0, TinyLlamaModel::default_lora_targets());
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());
    let _ckpt = run_lora_ft(
        &mut model,
        &mut ds,
        &lora_cfg,
        &train_cfg,
        &loss,
        tmp.path(),
        "tinyllama-delta-shape",
        lease,
    )
    .expect("run_lora_ft must succeed");

    let delta_path = tmp
        .path()
        .join("nn")
        .join("lora-tinyllama-delta-shape.safetensors");
    let tensors = candle_core::safetensors::load(&delta_path, &Device::Cpu)
        .expect("delta safetensors must load");
    // 2 layers × 7 target modules × 2 (A + B) = 28.
    assert_eq!(
        tensors.len(),
        28,
        "delta safetensors must contain exactly 28 tensors \
         (2 layers × 7 targets × 2), got {}: keys={:?}",
        tensors.len(),
        tensors.keys().collect::<Vec<_>>()
    );

    // Every tensor name should end in `.lora_a.weight` or
    // `.lora_b.weight` — if a base tensor leaked into the Δ file it
    // would not match this pattern.
    for name in tensors.keys() {
        assert!(
            name.ends_with(".lora_a.weight") || name.ends_with(".lora_b.weight"),
            "unexpected non-LoRA key in delta bundle: {name}"
        );
    }
}
