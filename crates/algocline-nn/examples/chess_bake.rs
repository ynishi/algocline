//! Train one model on a rating-banded chess corpus.
//!
//! Reads PGN, filters to a band, encodes, and runs a full fine-tune
//! from scratch, writing a checkpoint the player example reloads.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_bake -- <path.pgn> <bands> <max_rows> <steps> <side> [ckpt_dir]
//! ```
//!
//! `bands` is one or more rating ranges: `1600-1799`, or
//! `1100-1299,1900-2099` to train one model that can be asked for
//! either. Each band gets a condition token prefixed to its games, and
//! a game outside every band is dropped.
//!
//! One run is meant to finish in tens of seconds on a laptop CPU.
//! `steps * batch` is what decides that, and the corpus needs at least
//! that many rows, so raising either raises the wall clock directly.
//! Measure before scaling: the earlier Othello work spent an hour of
//! someone else's machine finding that out.
//!
//! # Validation
//!
//! Pass a second PGN and the run holds nothing back from it — it is
//! already held back, being a different month. Rows are built from it
//! under the same bands and the same filter, and every periodic
//! checkpoint is scored against them after training, giving a loss
//! curve on data the run never touched.
//!
//! Without that curve, training loss is the only reading available,
//! and training loss cannot say whether a run stopped because it
//! converged or because it ran out of rows. Earlier runs here stopped
//! for the second reason and were reported as though the first had
//! happened.
//!
//! The validation month must be a *different* month rather than a
//! longer prefix of the training one: Lichess archives are a single
//! zstd frame, so reading further into the same file returns games the
//! training slice may already have consumed.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
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
    run_full_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig, Loss, ScheduleKind,
    TeacherCardDataset, TrainingLease,
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

/// Read an `f64` from the environment, falling back to `default`.
fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic xorshift, used only to order rows.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Fisher-Yates, so consecutive batches are not consecutive games.
///
/// Rows arrive in the order the archive lists them, which is the order
/// the games were played — so an unshuffled batch is a few dozen games
/// that started within the same second. Nothing forces those to be
/// alike, but nothing prevents it either, and the fix is four lines.
fn shuffle<T>(rows: &mut [T], rng: &mut Rng) {
    for i in (1..rows.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        rows.swap(i, j);
    }
}

/// Mean masked cross-entropy over a set of rows, without training.
///
/// The same loss the trainer optimises, so the two curves are directly
/// comparable.
fn eval_loss(
    model: &Gpt2Model,
    rows: &[(Vec<u32>, Vec<f32>)],
    ctx: usize,
    batch: usize,
    device: &Device,
) -> Result<f32, Box<dyn std::error::Error>> {
    let loss = CrossEntropyLoss::new();
    let mut total = 0f64;
    let mut counted = 0usize;
    for chunk in rows.chunks(batch) {
        let width = ctx;
        let mut ids: Vec<u32> = Vec::with_capacity(chunk.len() * width);
        let mut mask: Vec<f32> = Vec::with_capacity(chunk.len() * width);
        for (row, m) in chunk {
            for i in 0..width {
                ids.push(row.get(i).copied().unwrap_or(0));
                mask.push(m.get(i).copied().unwrap_or(0.0));
            }
        }
        let full = Tensor::from_vec(ids, (chunk.len(), width), device)?;
        let full_mask = Tensor::from_vec(mask, (chunk.len(), width), device)?;
        // Same shift the training loop applies: inputs are all but the
        // last token, targets and mask are all but the first.
        let inputs = full.narrow(1, 0, width - 1)?.contiguous()?;
        let targets = full.narrow(1, 1, width - 1)?.contiguous()?;
        let m = full_mask.narrow(1, 1, width - 1)?.contiguous()?;
        let logits = model.forward(&inputs)?;
        let l = loss
            .compute(&logits, &targets, Some(&m))?
            .to_scalar::<f32>()?;
        total += l as f64 * chunk.len() as f64;
        counted += chunk.len();
    }
    if counted == 0 {
        return Err("validation set is empty".into());
    }
    Ok((total / counted as f64) as f32)
}

/// Parse `1100-1299,1900-2099` into condition bands.
///
/// Several bands in one corpus is the interesting case: the model sees
/// which band it is playing as, so one checkpoint can be asked for
/// either. Baking a model per band answers a different and weaker
/// question, since two separately trained models differ for every
/// reason at once.
fn parse_bands(spec: &str) -> Result<Vec<ConditionBand>, String> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let (lo, hi) = part
            .split_once('-')
            .ok_or_else(|| format!("band {part:?} is not a min-max range"))?;
        let min: i64 = lo
            .trim()
            .parse()
            .map_err(|_| format!("band {part:?} has an unreadable minimum"))?;
        let max: i64 = hi
            .trim()
            .parse()
            .map_err(|_| format!("band {part:?} has an unreadable maximum"))?;
        if min > max {
            return Err(format!("band {part:?} is inverted"));
        }
        out.push(ConditionBand {
            min,
            max,
            token: format!("<elo:{min}-{max}>"),
        });
    }
    if out.is_empty() {
        return Err("no bands given".into());
    }
    Ok(out)
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
        "usage: chess_bake <path.pgn> <bands> [max_rows] [steps] [side] [ckpt_dir]\n\
         bands is comma-separated ranges, e.g. 1600-1799 or 1100-1299,1900-2099",
    )?;
    let bands = parse_bands(&args.next().unwrap_or_else(|| "1600-1799".into()))?;
    let max_rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4000);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let side = match args.next().as_deref() {
        None | Some("both") => ScoredSide::Both,
        Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    let ckpt_dir: PathBuf = args.next().unwrap_or_else(|| "/tmp".into()).into();
    let val_pgn = args.next();

    let tokens: Vec<String> = bands.iter().map(|b| b.token.clone()).collect();
    let vocab = MoveVocab::new(&tokens)?;

    let mut shape = ModelShape::compact(vocab.model_vocab_size(), bands.clone());
    shape.layers = env_usize("CHESS_LAYERS", shape.layers);
    shape.heads = env_usize("CHESS_HEADS", shape.heads);
    shape.dim = env_usize("CHESS_DIM", shape.dim);
    shape.ctx = env_usize("CHESS_CTX", shape.ctx);
    let batch = env_usize("CHESS_BATCH", 32);
    let lr = env_f64("CHESS_LR", 3e-3);
    // More than one pass over the rows is what lets a run continue past
    // the point the corpus runs out. Whether that helps or overfits is
    // exactly what the validation curve is for.
    let epochs = env_usize("CHESS_EPOCHS", 1).max(1);
    let seed = env_usize("CHESS_SEED", 20260805) as u64;
    let eval_every = env_usize("CHESS_EVAL_EVERY", 0);
    let val_rows_cap = env_usize("CHESS_VAL_ROWS", 3000);

    // The band is selected by the condition rather than by the filter:
    // a game outside every band is rejected when its token is resolved,
    // which is one code path instead of keeping a filter and a band
    // list in agreement.
    let opts = CorpusOptions {
        filter: GameFilter::accept_all()
            .decided_on_the_board()
            .with_min_base_seconds(180)
            .with_ply_bounds(10, None),
        max_rows,
        max_len: Some(shape.ctx),
        condition: Some(ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: bands.clone(),
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

    // Shuffle, then lay down one shuffled copy per epoch. The dataset
    // walks its rows once in order, so repetition has to be materialised
    // here; re-shuffling per epoch keeps the second pass from replaying
    // the first batch for batch.
    let base = corpus.into_teacher_rows();
    let mut rng = Rng::new(seed);
    let mut rows: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(base.len() * epochs);
    for _ in 0..epochs {
        let mut pass = base.clone();
        shuffle(&mut pass, &mut rng);
        rows.extend(pass);
    }
    eprintln!(
        "[bake] rows {} ({} unique x {epochs} epoch(s), shuffled, seed {seed})",
        rows.len(),
        base.len()
    );

    // The loop consumes `steps * batch` rows and refuses to run short.
    let needed = steps * batch + batch;
    if rows.len() < needed {
        return Err(format!(
            "have {} rows but {steps} steps at batch {batch} need {needed}; \
             read a longer slice, raise CHESS_EPOCHS, or lower steps",
            rows.len()
        )
        .into());
    }

    let vocab_size = vocab.model_vocab_size();
    let mut dataset = TeacherCardDataset::from_rows(
        rows,
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

    // Validation rows, built from a month the training slice cannot
    // contain. Read before training so a bad path fails in seconds
    // rather than after the run.
    let val_rows: Option<Vec<(Vec<u32>, Vec<f32>)>> = match &val_pgn {
        Some(p) => {
            let val_opts = CorpusOptions {
                max_rows: val_rows_cap,
                ..opts.clone()
            };
            let mut r = PgnReader::new(BufReader::new(File::open(p)?));
            let c = build_rows(&mut r, &vocab, &val_opts)?;
            eprintln!(
                "[bake] validation: {} rows from {} games of {p}",
                c.stats.rows, c.stats.games_read
            );
            Some(c.into_teacher_rows())
        }
        None => {
            eprintln!(
                "[bake] validation: none (pass a holdout PGN as the 7th argument). \
                 Training loss alone cannot tell convergence from running out of rows."
            );
            None
        }
    };

    let ft_cfg = FullFtConfig {
        lr,
        batch_size: batch,
        grad_accum: 1,
        steps,
        warmup: steps.min(10),
        schedule: ScheduleKind::CosineWithWarmup,
        weight_decay: 0.0,
        ckpt_every: eval_every,
        // Every periodic checkpoint is scored afterwards, so none of
        // them may be rotated away. The hook cannot do the scoring
        // itself: `CkptHook` is `'static`, and the model is not.
        ckpt_keep: steps.checked_div(eval_every).map_or(1, |n| n + 2),
    };

    let band_label = bands
        .iter()
        .map(|b| format!("{}-{}", b.min, b.max))
        .collect::<Vec<_>>()
        .join("_");
    let prefix = format!("chess-{band_label}-{side:?}").to_lowercase();
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

    // The validation curve. Scored after the fact from the periodic
    // checkpoints, which is what says whether the run stopped because
    // it converged or because it ran out of rows.
    if let Some(val) = &val_rows {
        eprintln!(
            "[bake] scoring {} validation rows against each checkpoint…",
            val.len()
        );
        let final_val = eval_loss(&model, val, shape.ctx, batch, &cfg.device)?;
        let mut curve: Vec<(usize, f32)> = Vec::new();
        if eval_every > 0 {
            let mut step = eval_every;
            while step <= steps {
                let p = ckpt_dir.join(format!("{prefix}-step{step}.safetensors"));
                if p.exists() {
                    let m = Gpt2Model::from_safetensors_file(&cfg, &p)?;
                    curve.push((step, eval_loss(&m, val, shape.ctx, batch, &cfg.device)?));
                }
                step += eval_every;
            }
        }
        println!("step,val_loss");
        for (s, v) in &curve {
            println!("{s},{v:.4}");
        }
        println!("{steps},{final_val:.4}");
        let best = curve
            .iter()
            .copied()
            .chain(std::iter::once((steps, final_val)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((s, v)) = best {
            eprintln!(
                "[bake] val_loss final={final_val:.4}  best={v:.4} at step {s}{}",
                if s == steps {
                    " (still improving at the end — undertrained)"
                } else {
                    " (past its best — later steps overfit)"
                }
            );
        }
    }
    eprintln!("[bake] shape written to {}", shape_path.display());
    println!("{}", ckpt_path.display());
    Ok(())
}
