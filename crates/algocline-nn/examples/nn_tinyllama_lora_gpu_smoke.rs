//! GPU smoke test — TinyLlama-1.1B LoRA fine-tuning loop on CUDA.
//!
//! Sibling of `nn_medium_lora_gpu_smoke.rs` (GPT-2 medium LoRA).
//! Same corpus / device / config plumbing, but swaps `Gpt2Model`
//! for `TinyLlamaModel` and picks the `tinyllama-1.1b` preset
//! (~1.1B params, 22 layers, 32 heads, 4 KV heads, dim 2048, hidden
//! 5632, ctx 2048, vocab 32000). Exercises `TinyLlamaModel::wrap_lora`
//! + LoRA-only optimizer step + Δ-only checkpoint save on CUDA.
//!
//! In addition to the run itself the driver verifies three invariants
//! that matter for "did LoRA actually learn?":
//!
//! 1. **Base weights frozen** — the base `VarMap` is dumped to
//!    safetensors before `wrap_lora` and again after training; the
//!    two files must be byte-identical, so `run_lora_ft` provably
//!    never routed a gradient into the base parameters.
//! 2. **Δ bundle is LoRA-only + right size** — the emitted
//!    `<ckpt_dir>/nn/lora-<card_id>.safetensors` size must fall
//!    below the configured ceiling (default `64 MB`, sized for
//!    TinyLlama-1.1B at rank 16 wrapping the canonical 7 targets =
//!    Q/K/V/O attention + gate/up/down MLP). GQA collapses K/V
//!    to `kv_heads * head_dim` so K/V Δ tensors stay narrower than
//!    Q/O, but the MLP triple (`gate_proj` / `up_proj` /
//!    `down_proj` at hidden_dim 5632) dominates the total footprint
//!    versus GPT-2 medium's `up` / `down` MLP pair.
//! 3. **Per-step loss trajectory is observable** — the loop emits a
//!    `tracing::info!("train_step", step, loss, lr)` event per step;
//!    the driver installs a stderr subscriber so the caller can
//!    inspect the trajectory via `RUST_LOG=algocline_nn=info`.
//!
//! Usage:
//!
//! ```bash
//! # CUDA (requires nvcc + candle-core/candle-nn cuda features)
//! RUST_LOG=algocline_nn=info \
//!   cargo run --release --features nn-cuda --example nn_tinyllama_lora_gpu_smoke
//!
//! # CPU fallback (compile check on dev host)
//! cargo run --release --example nn_tinyllama_lora_gpu_smoke
//! ```
//!
//! Env vars:
//!
//! - `NN_SMOKE_STEPS`         (default `50`)  — number of optimizer steps.
//! - `NN_SMOKE_BATCH`         (default `2`)   — batch size.
//! - `NN_SMOKE_CTX`           (default `64`)  — sequence length per row.
//! - `NN_SMOKE_LR`            (default `3e-4`) — peak learning rate.
//! - `NN_SMOKE_CKPT_DIR`      (default `/tmp`) — directory that will
//!   receive `nn/lora-<card_id>.safetensors` plus the base pre / post
//!   dumps.
//! - `NN_SMOKE_CARD_ID`       (default `smoke-lora-tinyllama`) — Δ bundle
//!   suffix (`<ckpt_dir>/nn/lora-<card_id>.safetensors`).
//! - `NN_SMOKE_VARIANT`       (default `tinyllama-1.1b`) — TinyLlama
//!   variant name accepted by `TinyLlamaConfig::from_variant`.
//! - `NN_SMOKE_LORA_RANK`     (default `16`)  — LoRA rank.
//! - `NN_SMOKE_LORA_ALPHA`    (default `32`)  — LoRA alpha scaling.
//! - `NN_SMOKE_LORA_TARGETS`  (default `full`) — `full` = canonical 7
//!   TinyLlama targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
//!   `gate_proj`, `up_proj`, `down_proj`), `attn` = attention only
//!   (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
//! - `NN_SMOKE_DELTA_MAX_BYTES` (default `67108864` = 64 MB) — hard
//!   ceiling for the Δ bundle size assertion. Larger than the GPT-2
//!   medium ceiling (32 MB) to accommodate TinyLlama's 3-way MLP
//!   wrap at hidden_dim 5632.
//! - `NN_SMOKE_BASE_FROZEN_CHECK` (default `1`) — set to `0` to skip
//!   the pre/post base-VarMap byte-comparison. The check adds
//!   ~4.4 GB of temp file I/O for TinyLlama-1.1B (F32).
//!
//! Exit 0 = training loop completed, base weights stayed frozen, and
//! the Δ bundle was written under the ceiling.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use algocline_nn::arch::{LoraConfig, TinyLlamaConfig, TinyLlamaModel};
use algocline_nn::train::{
    run_lora_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TokenizedDataset,
    TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tracing_subscriber::EnvFilter;

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

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key).ok().as_deref() {
        Some("0") | Some("false") | Some("no") | Some("off") => false,
        Some("1") | Some("true") | Some("yes") | Some("on") => true,
        Some(_) | None => default,
    }
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
    eprintln!("[smoke] using CPU device (this will be slow on tinyllama-1.1b)");
    Device::Cpu
}

/// Repeating corpus of `rows` sequences drawn from a small palette,
/// mirroring `nn_tinyllama_gpu_smoke.rs` so the base FT and LoRA FT
/// runs are directly comparable.
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

/// Parse the target-module preset from env: `attn` (Q/K/V/O only) or
/// `full` (canonical 7 for TinyLlama). Any unknown value falls
/// through to `full` with a warning line — the run should still
/// produce a valid result but the caller sees that the requested
/// preset was not honoured.
fn resolve_targets() -> Vec<String> {
    match std::env::var("NN_SMOKE_LORA_TARGETS")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "attn" => vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into(),
        ],
        "full" | "" => TinyLlamaModel::default_lora_targets(),
        other => {
            eprintln!(
                "[smoke] unknown NN_SMOKE_LORA_TARGETS={other:?}; falling back to full 7-target set"
            );
            TinyLlamaModel::default_lora_targets()
        }
    }
}

/// Byte-exact compare of two files, streaming a shared buffer instead
/// of loading both into memory (TinyLlama-1.1B base ≈ 4.4 GB each in
/// F32).
fn files_bytes_equal(a: &Path, b: &Path) -> std::io::Result<bool> {
    use std::io::Read;

    let mut fa = std::fs::File::open(a)?;
    let mut fb = std::fs::File::open(b)?;
    let la = fa.metadata()?.len();
    let lb = fb.metadata()?.len();
    if la != lb {
        return Ok(false);
    }

    let mut ba = vec![0u8; 64 * 1024];
    let mut bb = vec![0u8; 64 * 1024];
    loop {
        let na = fa.read(&mut ba)?;
        let nb = fb.read(&mut bb)?;
        if na != nb {
            return Ok(false);
        }
        if na == 0 {
            return Ok(true);
        }
        if ba[..na] != bb[..nb] {
            return Ok(false);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Stderr subscriber so the per-step `tracing::info!("train_step", …)`
    // events emitted by `run_ft_core` land next to the driver's own
    // `eprintln!` progress lines. Default filter (`info` for the
    // library) surfaces the trajectory without pulling in unrelated
    // spans; the caller can widen it with e.g. `RUST_LOG=debug`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("algocline_nn=info")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let steps = env_usize("NN_SMOKE_STEPS", 50);
    let batch = env_usize("NN_SMOKE_BATCH", 2);
    let ctx = env_usize("NN_SMOKE_CTX", 64);
    let lr = env_f64("NN_SMOKE_LR", 3e-4);
    let ckpt_dir: PathBuf = std::env::var("NN_SMOKE_CKPT_DIR")
        .unwrap_or_else(|_| "/tmp".into())
        .into();
    let card_id =
        std::env::var("NN_SMOKE_CARD_ID").unwrap_or_else(|_| "smoke-lora-tinyllama".into());
    let variant = std::env::var("NN_SMOKE_VARIANT").unwrap_or_else(|_| "tinyllama-1.1b".into());
    let lora_rank = env_usize("NN_SMOKE_LORA_RANK", 16);
    let lora_alpha = env_f32("NN_SMOKE_LORA_ALPHA", 32.0);
    let delta_max_bytes = env_u64("NN_SMOKE_DELTA_MAX_BYTES", 64 * 1024 * 1024);
    let base_frozen_check = env_bool("NN_SMOKE_BASE_FROZEN_CHECK", true);
    let targets = resolve_targets();

    eprintln!(
        "[smoke] config: variant={variant} steps={steps} batch={batch} ctx={ctx} lr={lr} \
         ckpt_dir={ckpt_dir:?} card_id={card_id} rank={lora_rank} alpha={lora_alpha} \
         targets={targets:?} delta_max_bytes={delta_max_bytes} \
         base_frozen_check={base_frozen_check}"
    );

    let device = resolve_device();

    let mut cfg = TinyLlamaConfig::from_variant(&variant)
        .ok_or_else(|| format!("unknown TinyLlama variant: {variant:?}"))?;
    cfg.device = device.clone();
    cfg.dtype = DType::F32;

    eprintln!(
        "[smoke] building model layers={} heads={} kv_heads={} dim={} hidden={} vocab={} device={:?}",
        cfg.layers, cfg.heads, cfg.kv_heads, cfg.dim, cfg.hidden_dim, cfg.vocab, cfg.device
    );
    let build_t0 = Instant::now();
    let base_vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&base_vm, cfg.dtype, &cfg.device);
    let mut model = TinyLlamaModel::new(&cfg, vb)?;
    eprintln!("[smoke] model built in {:.2?}", build_t0.elapsed());

    // Optional pre/post byte-exact frozen check. The base VarMap is
    // captured before `wrap_lora` (which only inserts new LoRA vars —
    // it does not touch the existing base tensors) and again after
    // `run_lora_ft` returns. Both files must be byte-identical.
    let frozen_dir = ckpt_dir.join("nn").join("base-frozen-check");
    let base_pre_path = frozen_dir.join(format!("base-pre-{card_id}.safetensors"));
    let base_post_path = frozen_dir.join(format!("base-post-{card_id}.safetensors"));
    if base_frozen_check {
        std::fs::create_dir_all(&frozen_dir)
            .map_err(|e| format!("mkdir {:?}: {e}", frozen_dir.display()))?;
        eprintln!(
            "[smoke] dumping base VarMap pre-train → {:?}",
            base_pre_path
        );
        base_vm.save(&base_pre_path)?;
        let sz = std::fs::metadata(&base_pre_path)?.len();
        eprintln!(
            "[smoke] base pre-dump = {sz} bytes ({:.2} MB)",
            sz as f64 / (1024.0 * 1024.0)
        );
    }

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
    };

    let lora_cfg = LoraConfig::with_targets(lora_rank, lora_alpha, targets.iter().cloned());
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

    // Assertions run in order: size ceiling first (cheap), then base
    // frozen (expensive dump + compare). Both surface as a non-zero
    // exit so the caller sees a hard fail rather than a warning.
    if delta_bytes > delta_max_bytes {
        return Err(format!(
            "LoRA Δ bundle size {} B exceeds ceiling {} B (path={:?}). \
             Rank {} × alpha {} × {} targets on TinyLlama {}. \
             Set NN_SMOKE_DELTA_MAX_BYTES to raise the ceiling.",
            delta_bytes,
            delta_max_bytes,
            delta_path,
            lora_rank,
            lora_alpha,
            targets.len(),
            variant,
        )
        .into());
    }

    if base_frozen_check {
        eprintln!(
            "[smoke] dumping base VarMap post-train → {:?}",
            base_post_path
        );
        base_vm.save(&base_post_path)?;
        let sz = std::fs::metadata(&base_post_path)?.len();
        eprintln!(
            "[smoke] base post-dump = {sz} bytes ({:.2} MB)",
            sz as f64 / (1024.0 * 1024.0)
        );

        let cmp_t0 = Instant::now();
        let equal = files_bytes_equal(&base_pre_path, &base_post_path)?;
        eprintln!(
            "[smoke] base pre/post byte-compare in {:.2?}: equal={}",
            cmp_t0.elapsed(),
            equal
        );
        if !equal {
            return Err(format!(
                "Base VarMap changed during LoRA training. \
                 pre={:?} post={:?} — LoRA invariant #1 (base weights frozen) violated.",
                base_pre_path, base_post_path
            )
            .into());
        }
    } else {
        eprintln!("[smoke] base-frozen check skipped (NN_SMOKE_BASE_FROZEN_CHECK=0)");
    }

    Ok(())
}
