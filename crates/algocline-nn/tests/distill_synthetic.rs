//! Integration test — synthetic distillation run.
//!
//! Exercises `run_distill` end-to-end through the public crate surface
//! on a tiny CPU model + hand-picked teacher rows. The dataset carries
//! a per-token loss mask that zeroes out the "prompt" prefix of each
//! row; the assertion is that training still drives the min loss under
//! a fraction of the untrained baseline, proving the masked cross-
//! entropy actually flows gradients through the response region and
//! updates the model. A companion test checks that a fully-masked
//! batch (mask = 0 everywhere) yields an exactly-zero loss so downstream
//! callers can trust the "no positions counted" edge case.

use std::sync::Arc;

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{
    run_distill, CrossEntropyLoss, DatasetOpts, DistillLossKind, DistillSpec, FullFtConfig,
    HardLabelDistillLoss, Loss, ScheduleKind, TeacherCardDataset, TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tempfile::TempDir;

/// Build a tiny CPU GPT-2 model and its owning VarMap.
fn tiny_model() -> (Gpt2Config, VarMap, Gpt2Model) {
    let cfg = Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab: 24,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: None,
    };
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).expect("build tiny gpt-2");
    (cfg, vm, model)
}

/// Build a repeating teacher corpus: a fixed 8-token sequence whose
/// first 4 tokens act as the "prompt" (mask=0) and last 4 as the
/// "response" (mask=1). Every token id stays < vocab (24).
fn teacher_corpus(rows: usize) -> Vec<(Vec<u32>, Vec<f32>)> {
    let ids: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    let mask: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    std::iter::repeat_with(|| (ids.clone(), mask.clone()))
        .take(rows)
        .collect()
}

fn opts_for_seq(ctx_len: usize) -> DatasetOpts {
    DatasetOpts {
        batch_size: 1,
        ctx_len,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    }
}

/// Baseline forward-only loss on the first teacher row using the same
/// masked cross-entropy the training loop will see. Used to gate the
/// "training actually descended" assertion.
fn baseline_masked_loss(model: &Gpt2Model, ids: &[u32], mask: &[f32]) -> f32 {
    let device = &model.config().device;
    let seq = ids.len();
    let full_ids = Tensor::from_slice(ids, (1, seq), device).unwrap();
    let full_mask = Tensor::from_slice(mask, (1, seq), device).unwrap();
    let inputs = full_ids
        .narrow(1, 0, seq - 1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    let targets = full_ids
        .narrow(1, 1, seq - 1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    // Shift the mask so mask position k gates target token k
    // (= input position k + 1) — the same rule the loop uses.
    let mask_shifted = full_mask
        .narrow(1, 1, seq - 1)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap();
    let logits = model.forward(&inputs).unwrap();
    let loss = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask_shifted))
        .unwrap();
    loss.to_scalar::<f32>().unwrap()
}

#[test]
fn distill_run_reduces_masked_loss_on_teacher_corpus() {
    let (_cfg, vm, model) = tiny_model();
    let corpus = teacher_corpus(500);
    let baseline = baseline_masked_loss(&model, &corpus[0].0, &corpus[0].1);
    let mut dataset =
        TeacherCardDataset::from_rows(corpus, opts_for_seq(8)).expect("build TeacherCardDataset");
    let spec = DistillSpec::ce(FullFtConfig {
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
    });
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());

    let ckpt = run_distill(
        &model,
        &vm,
        &mut dataset,
        &spec,
        tmp.path(),
        "distill",
        lease,
    )
    .expect("distillation run must complete");

    let min_loss = *ckpt
        .metrics
        .get("min_train_loss")
        .expect("min_train_loss metric must be recorded");
    assert!(
        min_loss < baseline * 0.75,
        "expected training to drive the masked loss under 0.75 * baseline; \
         got min_train_loss={min_loss}, baseline={baseline}"
    );
    // Terminal bundle recorded under the stable filename.
    let final_path = tmp.path().join("distill.safetensors");
    assert!(final_path.exists(), "expected {final_path:?} to exist");
    assert_eq!(ckpt.bundle_ref, "distill.safetensors");
}

#[test]
fn distill_loss_kind_is_ce_by_default() {
    // Sanity: the `ce` builder produces the Ce loss variant so the
    // Card metadata layer knows which loss was used without inspecting
    // the loss object itself.
    let spec = DistillSpec::ce(FullFtConfig::default());
    match spec.loss_kind {
        DistillLossKind::Ce => {}
    }
}

#[test]
fn fully_masked_teacher_batch_yields_zero_loss() {
    // If every mask position is zero, the loss must be exactly zero
    // regardless of the model's raw logits (the numerator is zero
    // and the denominator is clamped to 1 inside CrossEntropyLoss).
    let device = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let cfg = Gpt2Config {
        layers: 1,
        heads: 2,
        dim: 8,
        ctx: 4,
        vocab: 16,
        dtype: DType::F32,
        device: Device::Cpu,
        eps: 1e-5,
        moe: None,
        custom: None,
    };
    let model = Gpt2Model::new(&cfg, vb).unwrap();
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let mask: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let full_ids = Tensor::from_slice(&ids, (1, 4), &device).unwrap();
    let full_mask = Tensor::from_slice(&mask, (1, 4), &device).unwrap();
    let inputs = full_ids
        .narrow(1, 0, 3)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    let targets = full_ids
        .narrow(1, 1, 3)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    let mask_shifted = full_mask
        .narrow(1, 1, 3)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap();
    let logits = model.forward(&inputs).unwrap();
    let loss = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask_shifted))
        .unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert_eq!(val, 0.0);

    // The plain CE variant, called through the same path, must agree.
    let plain = CrossEntropyLoss::new()
        .compute(&logits, &targets, Some(&mask_shifted))
        .unwrap();
    assert_eq!(plain.to_scalar::<f32>().unwrap(), 0.0);
}
