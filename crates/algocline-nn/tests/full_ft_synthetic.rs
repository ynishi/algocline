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
    tiny_model_with_dtype(DType::F32)
}

/// Same tiny shape with a caller-picked parameter dtype. Note that
/// candle 0.11's CPU backend has no BF16 matmul, so a BF16 GPT-2 only
/// forwards on CUDA (which is what the bridge's device/dtype matrix
/// guard enforces at the preset entrypoints); the F16 shape below is
/// buildable on CPU and exists to exercise the trainer's up-front
/// refusal.
fn tiny_model_with_dtype(dtype: DType) -> (Gpt2Config, VarMap, Gpt2Model) {
    let cfg = Gpt2Config {
        layers: 2,
        heads: 2,
        dim: 16,
        ctx: 8,
        vocab: 24,
        dtype,
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
    dataset_opts_for_seq_batched(ctx_len, 1)
}

fn dataset_opts_for_seq_batched(ctx_len: usize, batch_size: usize) -> DatasetOpts {
    DatasetOpts {
        batch_size,
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
        init_from: None,
        mask_disallowed_logits: false,
        ..FullFtConfig::default()
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
        None,
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

/// Toy learnable-unigram model for the offline BF16 loop fence.
///
/// candle 0.11's CPU backend has no BF16 matmul, so a real GPT-2
/// cannot forward in BF16 off-GPU (the A40 smoke runbook covers that
/// half). What CAN be fenced offline is everything the mixed path
/// adds around the model: the dtype-driven optimizer dispatch in
/// `run_ft_core`, the F32 loss cast, `MixedAdamW`'s master-weight
/// updates through a real `loss.backward()`, and the BF16 checkpoint
/// save. This model produces `[b, t, vocab]` logits by broadcasting a
/// single BF16 bias vector — `broadcast_add` / sum-reduction are BF16-
/// supported on CPU — and overfits toward the corpus marginal.
struct UnigramModel {
    /// `[vocab]` bias, registered against the test's VarMap (BF16).
    bias: Tensor,
    vocab: usize,
    device: Device,
}

impl candle_nn::Module for UnigramModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t) = xs.dims2()?;
        let zeros = Tensor::zeros((b, t, self.vocab), self.bias.dtype(), &self.device)?;
        zeros.broadcast_add(&self.bias)
    }
}

impl algocline_nn::train::DeviceView for UnigramModel {
    fn device(&self) -> &Device {
        &self.device
    }
}

/// Mixed-precision loop fence (design §7.1): BF16 parameters train
/// through `run_full_ft` → `MixedAdamW` on CPU. Loss must drop toward
/// the corpus marginal entropy, the parameter must stay BF16, and the
/// terminal bundle must carry BF16 tensors.
#[test]
fn bf16_synthetic_run_reduces_loss_through_mixed_adamw() {
    let device = Device::Cpu;
    let vocab = 24usize;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::BF16, &device);
    let bias = vb
        .get_with_hints(vocab, "bias", candle_nn::Init::Const(0.0))
        .expect("register bf16 bias");
    let model = UnigramModel {
        bias,
        vocab,
        device: device.clone(),
    };

    // Uniform logits at init: CE = ln(vocab) ≈ 3.178. The repeated
    // corpus row has 7 distinct target tokens uniformly, so the
    // optimum sits near ln(7) ≈ 1.946 — a big, assertable gap.
    let baseline = (vocab as f32).ln();

    let mut dataset = TokenizedDataset::new(synthetic_corpus(500), dataset_opts_for_seq(8));
    let loss = CrossEntropyLoss::new();
    let ft_cfg = FullFtConfig {
        lr: 5e-2,
        batch_size: 1,
        grad_accum: 1,
        steps: 200,
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

    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "bf16_synthetic",
        lease,
        None,
    )
    .expect("bf16 training must complete");

    let min_loss = *ckpt
        .metrics
        .get("min_train_loss")
        .expect("min_train_loss metric must be recorded");
    assert!(
        min_loss < baseline * 0.8,
        "expected bf16 training to drive loss under 0.8 * ln(vocab); got \
         min_train_loss={min_loss}, baseline={baseline}"
    );

    // Parameters stayed BF16 (MixedAdamW writes the downcast master
    // back), and the saved bundle carries BF16 tensors.
    for var in vm.all_vars() {
        assert_eq!(var.dtype(), DType::BF16, "var drifted off bf16");
    }
    let final_path = tmp.path().join("bf16_synthetic.safetensors");
    let loaded =
        candle_core::safetensors::load(&final_path, &Device::Cpu).expect("load terminal bundle");
    let (name, tensor) = loaded.iter().next().expect("bundle has tensors");
    assert_eq!(
        tensor.dtype(),
        DType::BF16,
        "bundle tensor '{name}' must be bf16"
    );
}

/// F16 parameters are refused before the first forward pass — no loss
/// scaler ships, so accepting them would train into silent gradient
/// underflow instead of erroring.
#[test]
fn f16_params_refused_before_training_starts() {
    let (_cfg, vm, model) = tiny_model_with_dtype(DType::F16);
    let mut dataset = TokenizedDataset::new(synthetic_corpus(10), dataset_opts_for_seq(8));
    let loss = CrossEntropyLoss::new();
    let ft_cfg = FullFtConfig {
        steps: 5,
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
        "f16_refused",
        lease,
        None,
    ) {
        Ok(_) => panic!("expected f16 refusal"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("f16") && msg.contains("loss scaling"),
        "unexpected error: {msg}"
    );
    assert!(!tmp.path().join("f16_refused.safetensors").exists());
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
        None,
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
        None,
    ) {
        Ok(_) => panic!("expected LeaseHeld"),
        Err(e) => e,
    };
    assert!(matches!(err, TrainError::LeaseHeld));

    // No terminal bundle should have been written because the run
    // never started.
    assert!(!tmp.path().join("second.safetensors").exists());
}

/// `grad_accum > 1` must be numerically equivalent to a single-micro run
/// with `batch_size * grad_accum` samples. Both paths start from the
/// **same random initial weights** — a fresh `VarMap` is created once,
/// snapshotted to safetensors before either run touches it, and Path B
/// reloads from that snapshot into a rebuilt model. Without this shared
/// snapshot the paths would diverge from independent inits and the
/// relative drift would swamp the equivalence signal.
///
/// The corpus is deterministic (`synthetic_corpus` repeats a single
/// 8-token row and `shuffle=false` in `DatasetOpts`), so both runs
/// consume identical token content. `CrossEntropyLoss` uses mean
/// reduction, so on identical rows the mean-of-per-micro losses equals
/// the per-batch loss, and the pre-backward `1/N` scaling makes the
/// summed gradient equal to the mean gradient. The remaining drift is
/// F32 rounding and scalar-multiply ordering — ~10% relative tolerance
/// is tight enough to catch a real regression but forgiving of the
/// last-bit noise that CPU stochastic ordering introduces.
#[test]
fn grad_accum_matches_equivalent_batch() {
    let steps = 40usize;
    let init_tmp = TempDir::new().unwrap();
    let init_snapshot = init_tmp.path().join("init.safetensors");

    // Path A: single-micro path with batch_size = 4. Snapshot the
    // initial weights before training so Path B can restart from the
    // same point in weight space.
    let min_loss_a = {
        let (_cfg, vm, model) = tiny_model();
        vm.save(&init_snapshot).expect("snapshot initial weights");
        let corpus = synthetic_corpus(steps * 4 + 8);
        let mut dataset = TokenizedDataset::new(corpus, dataset_opts_for_seq_batched(8, 4));
        let loss = CrossEntropyLoss::new();
        let ft_cfg = FullFtConfig {
            lr: 8e-3,
            batch_size: 4,
            grad_accum: 1,
            steps,
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
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut dataset,
            &ft_cfg,
            &loss,
            tmp.path(),
            "ga_ref",
            lease,
            None,
        )
        .expect("single-micro training must complete");
        *ckpt
            .metrics
            .get("min_train_loss")
            .expect("min_train_loss must be recorded")
    };

    // Path B: grad_accum = 4 with batch_size = 1. Reload the same
    // initial weights so the equivalence claim tests the loop shape,
    // not the two independent random inits.
    let min_loss_b = {
        let (_cfg, mut vm, model) = tiny_model();
        vm.load(&init_snapshot).expect("restore initial weights");
        let corpus = synthetic_corpus(steps * 4 + 8);
        let mut dataset = TokenizedDataset::new(corpus, dataset_opts_for_seq_batched(8, 1));
        let loss = CrossEntropyLoss::new();
        let ft_cfg = FullFtConfig {
            lr: 8e-3,
            batch_size: 1,
            grad_accum: 4,
            steps,
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
        let ckpt = run_full_ft(
            &model,
            &vm,
            &mut dataset,
            &ft_cfg,
            &loss,
            tmp.path(),
            "ga_accum",
            lease,
            None,
        )
        .expect("grad_accum=4 training must complete");
        *ckpt
            .metrics
            .get("min_train_loss")
            .expect("min_train_loss must be recorded")
    };

    let rel = (min_loss_a - min_loss_b).abs() / min_loss_a.abs().max(1e-6);
    assert!(
        rel < 0.10,
        "expected grad_accum=4 to match single-micro path within 10% \
         relative tolerance (both paths start from the same snapshot); \
         got min_loss_a={min_loss_a}, min_loss_b={min_loss_b}, rel={rel}"
    );
}

/// End-to-end sanity check: `grad_accum = 4` on its own must reduce the
/// training loss materially below the baseline forward-pass loss. This
/// is the same shape as `synthetic_run_reduces_loss_and_saves_final_bundle`
/// but drives the multi-micro path.
#[test]
fn grad_accum_gt_one_reduces_loss() {
    let (_cfg, vm, model) = tiny_model();
    let corpus = synthetic_corpus(800);
    let baseline = baseline_loss(&model, &corpus[0]);
    let mut dataset = TokenizedDataset::new(corpus, dataset_opts_for_seq_batched(8, 1));
    let loss = CrossEntropyLoss::new();

    let ft_cfg = FullFtConfig {
        lr: 8e-3,
        batch_size: 1,
        grad_accum: 4,
        steps: 60,
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

    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "ga_solo",
        lease,
        None,
    )
    .expect("grad_accum training must complete");

    let min_loss = *ckpt
        .metrics
        .get("min_train_loss")
        .expect("min_train_loss must be recorded");
    assert!(
        min_loss < baseline * 0.75,
        "expected grad_accum=4 training to drive loss under 0.75 * baseline; \
         got min_train_loss={min_loss}, baseline={baseline}"
    );
    assert!(ckpt.train_loss.is_finite());
}

/// BF16 variant of the multi-micro sanity check. Combines
/// `MixedAdamW`'s FP32-master update with `GradStore::extend`-based
/// gradient accumulation to fence the drift path where BF16 grads are
/// summed N times per optimizer step. Same shape and threshold as
/// `bf16_synthetic_run_reduces_loss_through_mixed_adamw`.
#[test]
fn bf16_grad_accum_synthetic_run_reduces_loss_through_mixed_adamw() {
    let device = Device::Cpu;
    let vocab = 24usize;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::BF16, &device);
    let bias = vb
        .get_with_hints(vocab, "bias", candle_nn::Init::Const(0.0))
        .expect("register bf16 bias");
    let model = UnigramModel {
        bias,
        vocab,
        device: device.clone(),
    };

    let baseline = (vocab as f32).ln();

    let mut dataset =
        TokenizedDataset::new(synthetic_corpus(800), dataset_opts_for_seq_batched(8, 1));
    let loss = CrossEntropyLoss::new();
    let ft_cfg = FullFtConfig {
        lr: 5e-2,
        batch_size: 1,
        grad_accum: 2,
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

    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        tmp.path(),
        "bf16_ga",
        lease,
        None,
    )
    .expect("bf16 grad_accum training must complete");

    let min_loss = *ckpt
        .metrics
        .get("min_train_loss")
        .expect("min_train_loss must be recorded");
    assert!(
        min_loss < baseline * 0.8,
        "expected bf16 grad_accum=2 training to drive loss under 0.8 * ln(vocab); \
         got min_train_loss={min_loss}, baseline={baseline}"
    );

    // Parameters stayed BF16 across the accumulated updates.
    for var in vm.all_vars() {
        assert_eq!(
            var.dtype(),
            DType::BF16,
            "var drifted off bf16 under grad_accum"
        );
    }
}
