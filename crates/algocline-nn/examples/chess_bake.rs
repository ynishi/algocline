//! Train one model on a rating-banded chess corpus.
//!
//! Reads PGN, filters to a band, encodes, and runs a full fine-tune
//! from scratch, writing a checkpoint the player example reloads.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_bake -- <path.pgn> <min_elo> <max_elo> <max_rows> <steps> <side>
//! ```
//!
//! One run is meant to finish in tens of seconds on a laptop CPU.
//! `steps * batch` is what decides that, and the corpus needs at least
//! that many rows, so raising either raises the wall clock directly.
//! Measure before scaling: the earlier Othello work spent an hour of
//! someone else's machine finding that out.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::corpus::{
    build_rows, ConditionBand, ConditionSpec, CorpusOptions, ScoredSide,
};
use algocline_nn::chess::filter::GameFilter;
use algocline_nn::chess::pgn::PgnReader;
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::ModelShape;
use algocline_nn::train::{
    run_full_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, ScheduleKind, TeacherCardDataset,
    TrainingLease,
};

/// Read a `usize` from the environment, falling back to `default`.
///
/// The shape and the training dials are environment-driven so a GPU
/// run can be scaled without editing the source the pod cloned.
fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_bake: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(
        "usage: chess_bake <path.pgn> [min_elo] [max_elo] [max_rows] [steps] [side] [ckpt_dir]",
    )?;
    let min_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1600);
    let max_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1799);
    let max_rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4000);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let side = match args.next().as_deref() {
        None | Some("both") => ScoredSide::Both,
        Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    let ckpt_dir: PathBuf = args.next().unwrap_or_else(|| "/tmp".into()).into();

    let band_token = format!("<elo:{min_elo}-{max_elo}>");
    let vocab = MoveVocab::new(std::slice::from_ref(&band_token))?;

    let mut shape = ModelShape::compact(vocab.model_vocab_size(), Some(band_token.clone()));
    shape.layers = env_usize("CHESS_LAYERS", shape.layers);
    shape.heads = env_usize("CHESS_HEADS", shape.heads);
    shape.dim = env_usize("CHESS_DIM", shape.dim);
    shape.ctx = env_usize("CHESS_CTX", shape.ctx);
    let batch = env_usize("CHESS_BATCH", 32);

    let opts = CorpusOptions {
        filter: GameFilter::accept_all()
            .with_rating_band(min_elo, max_elo)
            .decided_on_the_board()
            .with_min_base_seconds(180)
            .with_ply_bounds(10, None),
        max_rows,
        max_len: Some(shape.ctx),
        condition: Some(ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![ConditionBand {
                min: min_elo,
                max: max_elo,
                token: band_token.clone(),
            }],
        }),
        scored_side: side,
        ..Default::default()
    };

    eprintln!("[bake] reading {path}");
    let t0 = Instant::now();
    let mut reader = PgnReader::new(BufReader::new(File::open(&path)?));
    let corpus = build_rows(&mut reader, &vocab, &opts)?;
    eprintln!(
        "[bake] corpus: {} rows from {} games in {:.2?} ({} tokens)",
        corpus.stats.rows,
        corpus.stats.games_read,
        t0.elapsed(),
        corpus.stats.tokens
    );

    // The loop consumes `steps * batch` rows and refuses to run short.
    let needed = steps * batch + batch;
    if corpus.stats.rows < needed {
        return Err(format!(
            "corpus has {} rows but {steps} steps at batch {batch} need {needed}; \
             read a longer slice or lower steps",
            corpus.stats.rows
        )
        .into());
    }

    let vocab_size = vocab.model_vocab_size();
    let mut dataset = TeacherCardDataset::from_rows(
        corpus.into_teacher_rows(),
        DatasetOpts {
            batch_size: batch,
            ctx_len: shape.ctx,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    )?;

    // CUDA when the build has it, so the same example serves the pod.
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let cfg = shape.config(device, DType::F32);
    eprintln!(
        "[bake] model layers={} heads={} dim={} ctx={} vocab={} side={side:?}",
        cfg.layers, cfg.heads, cfg.dim, cfg.ctx, cfg.vocab
    );
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb)?;

    let ft_cfg = FullFtConfig {
        lr: 3e-3,
        batch_size: batch,
        grad_accum: 1,
        steps,
        warmup: steps.min(10),
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: 0,
        ckpt_keep: 1,
    };

    let prefix = format!("chess-{min_elo}-{max_elo}-{side:?}").to_lowercase();
    eprintln!("[bake] training {steps} steps at batch {batch}…");
    let t0 = Instant::now();
    let ckpt = run_full_ft(
        &model,
        &vm,
        &mut dataset,
        &ft_cfg,
        &CrossEntropyLoss::new(),
        &ckpt_dir,
        &prefix,
        Arc::new(TrainingLease::new()),
        None,
    )?;
    let elapsed = t0.elapsed();

    let min_loss = ckpt
        .metrics
        .get("min_train_loss")
        .copied()
        .unwrap_or(f32::NAN);
    // A model that learned nothing sits at the uniform-draw loss, so
    // that is the floor a run has to beat to have done anything.
    let uniform = (vocab_size as f32).ln();
    eprintln!(
        "[bake] done in {:.2?} ({:.2}s/step): final_loss={:.4} min_loss={:.4} \
         uniform_baseline={uniform:.4}",
        elapsed,
        elapsed.as_secs_f64() / steps.max(1) as f64,
        ckpt.train_loss,
        min_loss
    );
    // The shape rides with the checkpoint: the weights alone do not
    // say how many layers produced them, and a reader that guesses
    // wrong either fails on a tensor name or, worse, does not.
    let ckpt_path = ckpt_dir.join(format!("{prefix}.safetensors"));
    let shape_path = shape.save(&ckpt_path)?;
    eprintln!("[bake] shape written to {}", shape_path.display());
    println!("{}", ckpt_path.display());
    Ok(())
}
