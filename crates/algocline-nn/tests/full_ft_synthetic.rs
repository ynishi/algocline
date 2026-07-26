//! Integration test for the Full FT training loop.
//!
//! Drives `run_full_ft` from outside the crate on a synthetic corpus
//! big enough to exercise the loop end-to-end but small enough to run
//! in a couple of seconds on CPU. This complements the inline unit
//! tests: the goal here is to prove that the public crate surface
//! (`Gpt2Model` + `TokenizedDataset` + `CrossEntropyLoss` +
//! `run_full_ft`) composes correctly through re-exports without
//! reaching into private items.

use std::path::PathBuf;
use std::sync::Arc;

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{
    run_full_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
    TrainError, TrainingLease,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tempfile::TempDir;

/// Build a tiny GPT-2 model on CPU together with the VarMap that owns
/// its parameters.
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

/// A repeating 8-token sequence — enough for a 2-layer / 2-head model
/// to overfit meaningfully in ~150 steps.
fn synthetic_corpus(rows: usize) -> Vec<Vec<u32>> {
    // Every token id must stay below `vocab = 24` — the model's vocab
    // size is intentionally small so a forward pass on an
    // out-of-range id would panic at index-select.
    let base: Vec<u32> = vec![1, 4, 9, 16, 22, 12, 7, 3];
    std::iter::repeat_with(|| base.clone()).take(rows).collect()
}

fn dataset_opts_for_seq(ctx_len: usize) -> DatasetOpts {
    DatasetOpts {
        batch_size: 1,
        ctx_len,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    }
}

/// Recompute the first-batch cross-entropy loss on the untrained
/// model. Used to compare against the training-loop's minimum loss.
fn baseline_loss(model: &Gpt2Model, corpus_row: &[u32]) -> f32 {
    let device = &model.config().device;
    let full = Tensor::from_slice(corpus_row, (1, corpus_row.len()), device).unwrap();
    let inputs = full
        .narrow(1, 0, corpus_row.len() - 1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    let targets = full
        .narrow(1, 1, corpus_row.len() - 1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .contiguous()
        .unwrap();
    let logits = model.forward(&inputs).unwrap();
    let loss = CrossEntropyLoss::new()
        .compute_via_trait(&logits, &targets)
        .unwrap();
    loss.to_scalar::<f32>().unwrap()
}

/// Small helper trait so `baseline_loss` can call `Loss::compute`
/// through a stable name without pulling the trait into every scope.
trait LossExt {
    fn compute_via_trait(&self, logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor>;
}

impl LossExt for CrossEntropyLoss {
    fn compute_via_trait(&self, logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
        use algocline_nn::train::Loss;
        self.compute(logits, targets, None)
    }
}

#[test]
fn synthetic_run_reduces_loss_and_saves_final_bundle() {
    let (_cfg, vm, model) = tiny_model();
    let corpus = synthetic_corpus(500);
    let baseline = baseline_loss(&model, &corpus[0]);
    let mut dataset = TokenizedDataset::new(corpus, dataset_opts_for_seq(8));
    let loss = CrossEntropyLoss::new();

    let ft_cfg = FullFtConfig {
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

    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "synthetic",
        lease,
    )
    .expect("training must complete");

    // Loss trend — the minimum recorded loss should be materially
    // lower than the baseline forward-pass loss on the same input.
    let min_loss = *ckpt
        .metrics
        .get("min_train_loss")
        .expect("min_train_loss metric must be recorded");
    assert!(
        min_loss < baseline * 0.75,
        "expected training to drive loss under 0.75 * baseline; got \
         min_train_loss={min_loss}, baseline={baseline}"
    );
    assert!(ckpt.train_loss.is_finite());

    // File contract — the terminal bundle name is `<prefix>.safetensors`.
    let final_path: PathBuf = tmp.path().join("synthetic.safetensors");
    assert!(final_path.exists(), "expected {final_path:?} to exist");
    assert_eq!(ckpt.bundle_ref, "synthetic.safetensors");

    // Metrics — the final LR must be recorded for downstream logging.
    let final_lr = *ckpt
        .metrics
        .get("final_lr")
        .expect("final_lr metric must be recorded");
    assert!(final_lr.is_finite());
    assert!(final_lr >= 0.0);
}

#[test]
fn dataset_exhaustion_surfaces_error() {
    let (_cfg, vm, model) = tiny_model();
    // 3 rows total — the loop asks for many more steps.
    let dataset_rows = synthetic_corpus(3);
    let mut dataset = TokenizedDataset::new(dataset_rows, dataset_opts_for_seq(8));
    let loss = CrossEntropyLoss::new();
    let ft_cfg = FullFtConfig {
        steps: 20,
        warmup: 1,
        lr: 1e-3,
        ..FullFtConfig::default()
    };
    let tmp = TempDir::new().unwrap();
    let lease = Arc::new(TrainingLease::new());

    let err = match run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "short",
        lease,
    ) {
        Ok(_) => panic!("expected DatasetExhausted"),
        Err(e) => e,
    };
    match err {
        TrainError::DatasetExhausted { seen, requested } => {
            assert_eq!(requested, 20);
            assert!(seen < 20);
            assert!(seen <= 3);
        }
        other => panic!("expected DatasetExhausted, got {other:?}"),
    }
}

#[test]
fn concurrency_lease_prevents_double_training() {
    // Simulate a running trainer by holding a lease guard, then try to
    // start a second run and expect the LeaseHeld error.
    let (_cfg, vm, model) = tiny_model();
    let mut dataset = TokenizedDataset::new(synthetic_corpus(50), dataset_opts_for_seq(8));
    let loss = CrossEntropyLoss::new();
    let ft_cfg = FullFtConfig {
        steps: 5,
        warmup: 1,
        lr: 1e-3,
        ..FullFtConfig::default()
    };
    let tmp = TempDir::new().unwrap();

    // Shared lease.
    let lease = Arc::new(TrainingLease::new());
    let _held_guard = lease.acquire().expect("initial acquire must succeed");
    assert!(lease.is_active());

    // A second run against the same lease must be rejected up front.
    let err = match run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "second",
        lease.clone(),
    ) {
        Ok(_) => panic!("expected LeaseHeld"),
        Err(e) => e,
    };
    assert!(matches!(err, TrainError::LeaseHeld));

    // No terminal bundle should have been written because the run
    // never started.
    assert!(!tmp.path().join("second.safetensors").exists());
}
