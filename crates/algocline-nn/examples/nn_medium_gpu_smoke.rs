//! GPU smoke test — GPT-2 medium (355M) training loop on CUDA.
//!
//! This is a "did the CUDA path light up at all?" example, not a
//! full acceptance run. It builds a randomly-initialised GPT-2 medium
//! (24 layers / 16 heads / 1024 dim / 50257 vocab), trains for a
//! configurable number of steps on a hand-picked repeating token
//! sequence, and prints the loss trajectory + wall clock. On an A40
//! this loop should complete in a few minutes; on CPU it is slow
//! enough to be curiosity-only (each forward+backward on medium takes
//! seconds).
//!
//! Usage:
//!
//! ```bash
//! # CUDA (requires nvcc + candle-core cuda feature at compile time)
//! cargo run --release --features nn-cuda --example nn_medium_gpu_smoke
//!
//! # CPU fallback (compile check on dev host)
//! cargo run --release --example nn_medium_gpu_smoke
//! ```
//!
//! Env vars:
//!
//! - `NN_SMOKE_STEPS` (default `50`) — number of optimizer steps.
//! - `NN_SMOKE_BATCH` (default `2`)  — batch size.
//! - `NN_SMOKE_CTX`   (default `64`) — sequence length per row.
//! - `NN_SMOKE_LR`    (default `3e-4`) — peak learning rate.
//! - `NN_SMOKE_CKPT`  (default `/tmp/nn-medium-smoke.safetensors`)
//!   — where the final `<prefix>.safetensors` bundle is written.
//!
//! Exit 0 = training loop completed without a Rust panic (loss might
//! not have descended meaningfully — that is expected for a random
//! init on a repeating corpus at this step budget).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use algocline_nn::arch::{Gpt2Config, Gpt2Model};
use algocline_nn::train::{
    run_full_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
    TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn resolve_device() -> Device {
    #[cfg(feature = "nn-cuda")]
    {
        match Device::new_cuda(0) {
            Ok(dev) => {
                eprintln!("[smoke] using CUDA device 0");
                return dev;
            }
            Err(e) => {
                eprintln!("[smoke] cuda unavailable ({e}); falling back to CPU");
            }
        }
    }
    eprintln!("[smoke] using CPU device (this will be slow on medium)");
    Device::Cpu
}

/// Build a repeating corpus of `rows` sequences, each `ctx` tokens
/// long, drawn from a small palette of ids that stay comfortably
/// under the GPT-2 vocab size.
fn synthetic_corpus(rows: usize, ctx: usize) -> Vec<Vec<u32>> {
    // Palette of ~50 ids so the model sees enough variety to actually
    // move gradients away from a uniform-prior baseline.
    let palette: Vec<u32> = (100..150).collect();
    let mut corpus = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let mut row = Vec::with_capacity(ctx);
        for pos in 0..ctx {
            let idx = (row_idx + pos * 7) % palette.len();
            row.push(palette[idx]);
        }
        corpus.push(row);
    }
    corpus
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = env_usize("NN_SMOKE_STEPS", 50);
    let batch = env_usize("NN_SMOKE_BATCH", 2);
    let ctx = env_usize("NN_SMOKE_CTX", 64);
    let lr = env_f64("NN_SMOKE_LR", 3e-4);
    let ckpt_path: PathBuf = std::env::var("NN_SMOKE_CKPT")
        .unwrap_or_else(|_| "/tmp/nn-medium-smoke.safetensors".into())
        .into();

    eprintln!(
        "[smoke] config: steps={steps} batch={batch} ctx={ctx} lr={lr} \
         ckpt={ckpt_path:?}"
    );

    let device = resolve_device();

    // Full-size GPT-2 medium — 355M params, 24 layers, 16 heads,
    // 1024 dim, 1024 ctx (we only fill `ctx` positions), 50257 vocab.
    let mut cfg = Gpt2Config::medium();
    cfg.device = device.clone();
    cfg.dtype = DType::F32;

    eprintln!(
        "[smoke] building model layers={} heads={} dim={} vocab={} device={:?}",
        cfg.layers, cfg.heads, cfg.dim, cfg.vocab, cfg.device
    );
    let build_t0 = Instant::now();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb)?;
    eprintln!("[smoke] model built in {:.2?}", build_t0.elapsed());

    // The dataset row count is a safety margin so the loop never
    // trips `TrainError::DatasetExhausted` before `steps` completes.
    let rows_needed = steps.saturating_mul(batch).saturating_add(batch);
    let corpus = synthetic_corpus(rows_needed, ctx);
    let mut dataset = TokenizedDataset::new(
        corpus,
        DatasetOpts {
            batch_size: batch,
            ctx_len: ctx,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );

    let ft_cfg = FullFtConfig {
        lr,
        batch_size: batch,
        grad_accum: 1,
        steps,
        warmup: steps.min(5),
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
        init_from: None,
        mask_disallowed_logits: false,
        ..FullFtConfig::default()
    };

    let ckpt_dir = ckpt_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let ckpt_prefix = ckpt_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("nn-medium-smoke");
    let lease = Arc::new(TrainingLease::new());
    let loss = CrossEntropyLoss::new();

    eprintln!("[smoke] starting training…");
    let train_t0 = Instant::now();
    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &loss,
        &ckpt_dir,
        ckpt_prefix,
        lease,
        None,
    )?;
    let elapsed = train_t0.elapsed();

    let min_loss = ckpt
        .metrics
        .get("min_train_loss")
        .copied()
        .unwrap_or(f32::NAN);
    let final_lr = ckpt.metrics.get("final_lr").copied().unwrap_or(f32::NAN);
    eprintln!(
        "[smoke] done in {:.2?} ({:.2}s/step avg): final_loss={:.4} \
         min_loss={:.4} final_lr={:.6} bundle={:?}",
        elapsed,
        elapsed.as_secs_f64() / steps.max(1) as f64,
        ckpt.train_loss,
        min_loss,
        final_lr,
        ckpt_dir.join(format!("{ckpt_prefix}.safetensors"))
    );

    Ok(())
}
