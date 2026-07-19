//! GPU smoke test — GPT-2 medium (355M) LoRA fine-tuning loop on CUDA.
//!
//! Sibling of `nn_medium_gpu_smoke.rs`. Same corpus / device / config
//! plumbing, but swaps `run_full_ft` for `run_lora_ft` so the CUDA
//! path exercises `Gpt2Model::wrap_lora` + LoRA-only optimizer step +
//! Δ-only checkpoint save. Asserts the emitted Δ bundle is under
//! 20 MB (invariant #3 for GPT-2 medium at rank 16).
//!
//! Usage:
//!
//! ```bash
//! # CUDA (requires nvcc + candle-core/candle-nn cuda features)
//! cargo run --release --features nn-cuda --example nn_medium_lora_gpu_smoke
//!
//! # CPU fallback (compile check on dev host)
//! cargo run --release --example nn_medium_lora_gpu_smoke
//! ```
//!
//! Env vars:
//!
//! - `NN_SMOKE_STEPS`      (default `50`) — number of optimizer steps.
//! - `NN_SMOKE_BATCH`      (default `2`)  — batch size.
//! - `NN_SMOKE_CTX`        (default `64`) — sequence length per row.
//! - `NN_SMOKE_LR`         (default `3e-4`) — peak learning rate.
//! - `NN_SMOKE_CKPT_DIR`   (default `/tmp`) — directory that will
//!   receive `nn/lora-<card_id>.safetensors`.
//! - `NN_SMOKE_CARD_ID`    (default `smoke-lora-medium`) — Δ bundle
//!   suffix (`<ckpt_dir>/nn/lora-<card_id>.safetensors`).
//! - `NN_SMOKE_LORA_RANK`  (default `16`) — LoRA rank.
//! - `NN_SMOKE_LORA_ALPHA` (default `32`) — LoRA alpha scaling.
//! - `NN_SMOKE_DELTA_MAX_BYTES` (default `20 * 1024 * 1024`) — hard
//!   ceiling for the Δ bundle size assertion.
//!
//! Exit 0 = training loop completed, Δ bundle was written, and its
//! size stayed under the ceiling.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use algocline_nn::arch::{Gpt2Config, Gpt2Model, LoraConfig};
use algocline_nn::train::{
    run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
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

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
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

/// Repeating corpus of `rows` sequences drawn from a small palette,
/// mirroring `nn_medium_gpu_smoke.rs` so the two runs are directly
/// comparable.
fn synthetic_corpus(rows: usize, ctx: usize) -> Vec<Vec<u32>> {
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
    let ckpt_dir: PathBuf = std::env::var("NN_SMOKE_CKPT_DIR")
        .unwrap_or_else(|_| "/tmp".into())
        .into();
    let card_id =
        std::env::var("NN_SMOKE_CARD_ID").unwrap_or_else(|_| "smoke-lora-medium".into());
    let lora_rank = env_usize("NN_SMOKE_LORA_RANK", 16);
    let lora_alpha = env_f32("NN_SMOKE_LORA_ALPHA", 32.0);
    let delta_max_bytes = env_u64("NN_SMOKE_DELTA_MAX_BYTES", 20 * 1024 * 1024);

    eprintln!(
        "[smoke] config: steps={steps} batch={batch} ctx={ctx} lr={lr} \
         ckpt_dir={ckpt_dir:?} card_id={card_id} rank={lora_rank} alpha={lora_alpha}"
    );

    let device = resolve_device();

    let mut cfg = Gpt2Config::medium();
    cfg.device = device.clone();
    cfg.dtype = DType::F32;

    eprintln!(
        "[smoke] building model layers={} heads={} dim={} vocab={} device={:?}",
        cfg.layers, cfg.heads, cfg.dim, cfg.vocab, cfg.device
    );
    let build_t0 = Instant::now();
    let base_vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = Gpt2Model::new(&cfg, vb)?;
    eprintln!("[smoke] model built in {:.2?}", build_t0.elapsed());

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
    };

    let lora_cfg = LoraConfig::new(lora_rank, lora_alpha);
    let lease = Arc::new(TrainingLease::new());
    let loss = CrossEntropyLoss::new();

    eprintln!(
        "[smoke] starting LoRA training… target_modules={:?}",
        lora_cfg.target_modules
    );
    let train_t0 = Instant::now();
    let ckpt = run_lora_ft(
        &mut model,
        &mut dataset,
        &lora_cfg,
        &ft_cfg,
        &loss,
        &ckpt_dir,
        &card_id,
        lease,
    )?;
    let elapsed = train_t0.elapsed();

    let min_loss = ckpt
        .metrics
        .get("min_train_loss")
        .copied()
        .unwrap_or(f32::NAN);
    let final_lr = ckpt.metrics.get("final_lr").copied().unwrap_or(f32::NAN);

    let delta_path = ckpt_dir
        .join("nn")
        .join(format!("lora-{card_id}.safetensors"));
    let delta_bytes = std::fs::metadata(&delta_path)
        .map_err(|e| format!("stat {:?}: {e}", delta_path.display()))?
        .len();

    eprintln!(
        "[smoke] done in {:.2?} ({:.2}s/step avg): final_loss={:.4} \
         min_loss={:.4} final_lr={:.6} delta={:?} delta_bytes={} \
         (<= {} = {:.2} MB ceiling)",
        elapsed,
        elapsed.as_secs_f64() / steps.max(1) as f64,
        ckpt.train_loss,
        min_loss,
        final_lr,
        delta_path,
        delta_bytes,
        delta_max_bytes,
        delta_max_bytes as f64 / (1024.0 * 1024.0),
    );

    if delta_bytes > delta_max_bytes {
        return Err(format!(
            "LoRA Δ bundle size {} B exceeds ceiling {} B (path={:?}). \
             Rank {} × alpha {} on GPT-2 medium should stay well under 20 MB; \
             something is off with the wrap or the saved varmap.",
            delta_bytes, delta_max_bytes, delta_path, lora_rank, lora_alpha
        )
        .into());
    }

    Ok(())
}
