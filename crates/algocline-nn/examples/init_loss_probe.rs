//! Initialization-scale probe for the from-scratch GPT-2 student.
//!
//! Backs the `INIT_STDEV` constant in `arch::gpt2` with a reproducible
//! measurement instead of an assertion in a doc comment. Reports, for
//! two token-embedding init scales:
//!
//!   - stdev 0.02 — the GPT-2 reference init (what the crate uses)
//!   - stdev 1.0  — candle-nn's `embedding()` default
//!
//! the logit standard deviation, the step-0 masked cross-entropy, and
//! the 8-step loss trajectory under the teacher-card E2E's
//! hyperparameters. The reference scale starts at `ln(vocab)` and
//! descends; candle-nn's default starts ~13x higher and collapses into
//! a saturated softmax (`loss == 0.0` with the logits still drifting on
//! optimizer momentum).
//!
//! Phase 1 (forward only) uses the real gpt2-medium shape. Phase 2
//! (8 AdamW steps) keeps vocab/dim — hence the logit scale — but drops
//! to 2 layers so the backward pass finishes in seconds.
//!
//! Run: cargo run -p algocline-nn --example init_loss_probe --release

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{HardLabelDistillLoss, Loss};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

fn stats(t: &Tensor) -> (f32, f32, f32) {
    let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let max = v.iter().cloned().fold(f32::MIN, f32::max);
    (mean, var.sqrt(), max)
}

/// Build a model, optionally re-drawing the tied token embedding (= the
/// LM head) and the positional embedding at `wte_stdev`.
fn build(cfg: &Gpt2Config, wte_stdev: Option<f64>) -> (VarMap, Gpt2Model) {
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(cfg, vb).expect("build model");
    if let Some(sd) = wte_stdev {
        // `Var::set` writes through the shared storage, so the
        // already-built model sees the new values.
        let w = Tensor::randn(0f32, sd as f32, (cfg.vocab, cfg.dim), &cfg.device).unwrap();
        vm.set_one("wte.weight", &w).expect("set wte");
        let p = Tensor::randn(0f32, sd as f32, (cfg.ctx, cfg.dim), &cfg.device).unwrap();
        vm.set_one("wpe.weight", &p).expect("set wpe");
    }
    (vm, model)
}

/// A single teacher row: `seq` ids whose last `resp` positions are the
/// scored "response" region.
fn row(cfg: &Gpt2Config, seq: usize, resp: usize) -> (Vec<u32>, Vec<f32>) {
    let ids: Vec<u32> = (0..seq as u32)
        .map(|i| (i * 977) % cfg.vocab as u32)
        .collect();
    let mut mask = vec![0.0f32; seq];
    for m in mask.iter_mut().skip(seq - resp) {
        *m = 1.0;
    }
    (ids, mask)
}

fn init_loss(label: &str, cfg: &Gpt2Config, wte_stdev: Option<f64>, seq: usize, resp: usize) {
    let (_vm, model) = build(cfg, wte_stdev);
    let (ids, mask) = row(cfg, seq, resp);
    let inputs = Tensor::from_slice(&ids[..seq - 1], (1, seq - 1), &cfg.device).unwrap();
    let targets = Tensor::from_slice(&ids[1..seq], (1, seq - 1), &cfg.device).unwrap();
    let mask = Tensor::from_slice(&mask[1..seq], (1, seq - 1), &cfg.device).unwrap();

    let logits = model.forward(&inputs).expect("forward");
    let (mean, std, max) = stats(&logits);
    let val: f32 = HardLabelDistillLoss::new()
        .compute(&logits, &targets, Some(&mask))
        .expect("loss")
        .to_scalar()
        .unwrap();
    println!("{label:<26} logit mean={mean:+.3} std={std:8.3} max={max:9.3}  masked_CE={val:9.4}");
}

/// Hand-rolled equivalent of `run_ft_core`'s inner loop, instrumented so
/// each step reports what the loop's single `loss` scalar hides.
fn trajectory(label: &str, cfg: &Gpt2Config, wte_stdev: Option<f64>, seq: usize, resp: usize) {
    let (vm, model) = build(cfg, wte_stdev);
    let (ids, mask) = row(cfg, seq, resp);
    let inputs = Tensor::from_slice(&ids[..seq - 1], (1, seq - 1), &cfg.device).unwrap();
    let targets = Tensor::from_slice(&ids[1..seq], (1, seq - 1), &cfg.device).unwrap();
    let mask_t = Tensor::from_slice(&mask[1..seq], (1, seq - 1), &cfg.device).unwrap();

    let mut opt = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 3e-4,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    println!("--- {label}: 8 steps @ lr 3e-4 ---");
    for step in 0..8 {
        let logits = model.forward(&inputs).unwrap();
        let (_, lstd, lmax) = stats(&logits);
        let loss = HardLabelDistillLoss::new()
            .compute(&logits, &targets, Some(&mask_t))
            .unwrap();
        let val: f32 = loss.to_scalar().unwrap();

        // Per-position NLL at the masked (scored) positions, computed
        // outside the Loss impl so a degenerate reduction is visible.
        let flat = logits.reshape((seq - 1, cfg.vocab)).unwrap();
        let lp = candle_nn::ops::log_softmax(&flat, 1).unwrap();
        let nll: Vec<f32> = lp
            .gather(&targets.reshape((seq - 1, 1)).unwrap(), 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .neg()
            .unwrap()
            .to_vec1()
            .unwrap();
        let scored: Vec<f32> = nll[seq - 1 - resp..].to_vec();

        let wte = vm.data().lock().unwrap()["wte.weight"].as_tensor().clone();
        let (wmean, wstd, wmax) = stats(&wte);

        println!(
            "  step {step}: loss={val:>12.4}  logit std={lstd:>9.3} max={lmax:>10.3}  \
             wte(mean={wmean:+.4} std={wstd:.4} max={wmax:.3})  scored_nll={scored:?}"
        );

        opt.backward_step(&loss).unwrap();
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "algocline_nn=info".into()),
        )
        .init();

    let mut cfg = Gpt2Config::medium();
    cfg.device = Device::Cpu;
    cfg.dtype = DType::F32;

    let seq = 32;
    let resp = 5;
    println!(
        "gpt2-medium from scratch (layers={} dim={} vocab={}), seq={seq}, masked tail={resp}",
        cfg.layers, cfg.dim, cfg.vocab
    );
    println!("ln(vocab) = {:.4}\n", (cfg.vocab as f32).ln());

    init_loss("wte stdev=0.02 (crate)", &cfg, None, seq, resp);
    init_loss("wte stdev=1.0 (candle def)", &cfg, Some(1.0), seq, resp);

    // Phase 2 — same vocab/dim (identical logit scale), 2 layers so the
    // backward pass is affordable on CPU.
    let mut small = cfg.clone();
    small.layers = 2;
    small.ctx = 64;
    println!("\n[trajectory] layers=2 (vocab/dim unchanged), E2E hyperparams\n");
    trajectory("wte stdev=0.02 (crate)", &small, None, seq, resp);
    trajectory("wte stdev=1.0 (candle def)", &small, Some(1.0), seq, resp);
}
