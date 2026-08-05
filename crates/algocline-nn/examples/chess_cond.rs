//! Measure whether a condition token changes how the model plays.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_cond -- <ckpt.safetensors> <holdout.pgn> [side] [max_positions] [walk_band]
//! ```
//!
//! # Why not move match
//!
//! Top-1 move match is the wrong instrument for this question. On a
//! 236,800-row two-band checkpoint it read 0.2088 asked as the low band
//! against low-band games and 0.2080 asked as the high band — a
//! difference of 0.0008 — while the same model, in the Open Sicilian
//! after 3...cxd4, put 0.116 on Qxd4 as the low band and 0.018 as the
//! high one. Both bands answer Nxd4 first, so top-1 sees nothing; the
//! 6x change sits underneath it.
//!
//! What separates the bands is the shape of the distribution, so that
//! is what gets measured: the Jensen-Shannon divergence between the
//! model's legal-move distributions under each token, at the same
//! position. In bits, so 0 means identical and 1 means disjoint.
//!
//! # The reference it is read against
//!
//! A divergence has no meaning alone. Each band is also compared to a
//! uniform draw over the same legal moves, which is what the model
//! would look like knowing nothing. A band-to-band divergence that is
//! a small fraction of the band-to-uniform one says the token moves
//! the distribution slightly; one of the same order says it moves it
//! as much as knowing chess does.
//!
//! Divergences are also split by whether the two bands agree on the
//! top move, since that is exactly the split move match is blind to.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use cozy_chess::Board;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::corpus::ScoredSide;
use algocline_nn::chess::filter::GameFilter;
use algocline_nn::chess::pgn::{resolve_san, san_tokens, uci_standard, PgnReader};
use algocline_nn::chess::vocab::{MoveVocab, BOS};
use algocline_nn::chess::ModelShape;
use algocline_nn::metric::{self, MetricError};

/// Jensen-Shannon divergence in bits, via [`algocline_nn::metric::js`].
///
/// The crate's implementation reports nats and is already covered by
/// its own tests; bits are what a reader wants here, since the bound
/// becomes `[0, 1]` — zero when the two distributions agree everywhere,
/// one when they put their mass on disjoint moves.
fn js_bits(p: &[f32], q: &[f32]) -> Result<f64, MetricError> {
    Ok(metric::js(p, q)? as f64 / std::f64::consts::LN_2)
}

/// Renormalise a slice to sum to one, or fall back to uniform when it
/// carries no mass at all.
///
/// `metric::js` validates that its inputs are proper distributions, so
/// the renormalisation is a precondition rather than a nicety: the raw
/// softmax over the legal subset sums to whatever mass the model left
/// there.
fn normalise(v: &[f32]) -> Vec<f32> {
    let total: f32 = v.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / v.len() as f32; v.len()];
    }
    v.iter().map(|x| x / total).collect()
}

/// Running mean, kept as a pair so an empty set reads as absent rather
/// than as zero.
#[derive(Default)]
struct Mean {
    sum: f64,
    n: usize,
}

impl Mean {
    fn push(&mut self, x: f64) {
        self.sum += x;
        self.n += 1;
    }
    fn value(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_cond: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let ckpt: PathBuf = args
        .next()
        .ok_or("usage: chess_cond <ckpt> <holdout.pgn> [side] [max_positions] [walk_band]")?
        .into();
    let pgn = args.next().ok_or("missing holdout pgn path")?;
    let side = match args.next().as_deref() {
        None | Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some("both") => ScoredSide::Both,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    // 600 keeps one run inside a minute on a laptop CPU. Every position
    // costs one forward pass per band.
    let max_positions: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(600);
    let walk_band = args.next();

    let shape = ModelShape::load(&ckpt)?;
    if shape.bands.len() < 2 {
        return Err(format!(
            "this checkpoint carries {} band(s); comparing conditions needs at least 2",
            shape.bands.len()
        )
        .into());
    }
    let vocab = MoveVocab::new(&shape.band_tokens())?;
    let band_ids: Vec<u32> = shape
        .bands
        .iter()
        .map(|b| {
            vocab
                .id_of(&b.token)
                .ok_or_else(|| format!("band {} missing", b.token))
        })
        .collect::<Result<_, _>>()?;
    let n_bands = band_ids.len();

    let cfg = shape.config(Device::Cpu, DType::F32);
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;

    // Which games the positions are drawn from. Left open by default:
    // the question is how the token changes play, not how well it
    // matches any particular population.
    let mut filter = GameFilter::accept_all()
        .decided_on_the_board()
        .with_min_base_seconds(180)
        .with_ply_bounds(10, None);
    if let Some(arg) = &walk_band {
        let token = if arg.starts_with('<') {
            arg.clone()
        } else {
            format!("<elo:{arg}>")
        };
        let band = shape
            .band(&token)
            .ok_or_else(|| format!("checkpoint has no band {token}"))?;
        filter = filter.with_rating_band(band.min, band.max);
    }

    let mut reader = PgnReader::new(BufReader::new(File::open(&pgn)?));
    let t0 = Instant::now();
    let mut positions = 0usize;
    let mut games = 0usize;
    let mut top1_differs = 0usize;

    // Pairwise band-to-band, plus each band against a uniform draw.
    let mut pair_js: Vec<Mean> = (0..n_bands * n_bands).map(|_| Mean::default()).collect();
    let mut pair_js_agree: Vec<Mean> = (0..n_bands * n_bands).map(|_| Mean::default()).collect();
    let mut pair_js_differ: Vec<Mean> = (0..n_bands * n_bands).map(|_| Mean::default()).collect();
    let mut uniform_js: Vec<Mean> = (0..n_bands).map(|_| Mean::default()).collect();
    let mut all_pair_js: Vec<f64> = Vec::new();
    // The condition is one token at the far left of the row. If its
    // influence is carried by attention across the whole sequence, a
    // decay with depth is what a weak channel looks like — Maia-2 does
    // not prepend at all, it injects skill into the network
    // (arXiv:2409.20553), which is the shape of the remedy if this
    // decays.
    let ply_edges = [0usize, 10, 20, 30, 40, usize::MAX];
    let mut by_ply: Vec<Mean> = (0..ply_edges.len() - 1).map(|_| Mean::default()).collect();

    while positions < max_positions {
        let Some(game) = reader.next_game()? else {
            break;
        };
        if !filter.accepts_tags(&game) {
            continue;
        }
        let tokens = san_tokens(&game.movetext);
        if tokens.len() < 10 {
            continue;
        }
        games += 1;

        let mut board = Board::default();
        let mut moves_so_far: Vec<u32> = Vec::new();
        for (ply, token) in tokens.iter().enumerate() {
            let Ok(mv) = resolve_san(&board, token) else {
                break;
            };
            let played = uci_standard(&board, mv);

            let scored = match side {
                ScoredSide::Both => true,
                ScoredSide::White => ply.is_multiple_of(2),
                ScoredSide::Black => !ply.is_multiple_of(2),
            };
            if scored && positions < max_positions {
                let legal_ids = legal_move_ids(&board, &vocab);
                if legal_ids.len() >= 2 {
                    // One forward for all bands: the rows differ only in
                    // the condition token, so they batch cleanly.
                    let rows: Vec<Vec<u32>> = band_ids
                        .iter()
                        .map(|id| {
                            let mut row = vec![BOS, *id];
                            row.extend_from_slice(&moves_so_far);
                            let start = row.len().saturating_sub(shape.ctx);
                            row[start..].to_vec()
                        })
                        .collect();
                    let width = rows[0].len();
                    let flat: Vec<u32> = rows.concat();
                    let input = Tensor::from_vec(flat, (n_bands, width), &cfg.device)?;
                    let logits = model.forward(&input)?;

                    let mut dists: Vec<Vec<f32>> = Vec::with_capacity(n_bands);
                    for b in 0..n_bands {
                        let last = logits.i((b, width - 1))?;
                        let probs = candle_nn::ops::softmax(&last, 0)?.to_vec1::<f32>()?;
                        let over_legal: Vec<f32> =
                            legal_ids.iter().map(|id| probs[*id as usize]).collect();
                        dists.push(normalise(&over_legal));
                    }

                    let uniform = vec![1.0f32 / legal_ids.len() as f32; legal_ids.len()];
                    for (b, d) in dists.iter().enumerate() {
                        uniform_js[b].push(js_bits(d, &uniform)?);
                    }

                    let argmax = |d: &Vec<f32>| {
                        d.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    };
                    let tops: Vec<usize> = dists.iter().map(argmax).collect();

                    for i in 0..n_bands {
                        for j in (i + 1)..n_bands {
                            let d = js_bits(&dists[i], &dists[j])?;
                            pair_js[i * n_bands + j].push(d);
                            if tops[i] == tops[j] {
                                pair_js_agree[i * n_bands + j].push(d);
                            } else {
                                pair_js_differ[i * n_bands + j].push(d);
                            }
                            if i == 0 && j == n_bands - 1 {
                                // The widest pair is the one with the
                                // clearest signal, so depth is read off
                                // that one.
                                all_pair_js.push(d);
                                let bucket = ply_edges
                                    .windows(2)
                                    .position(|w| ply >= w[0] && ply < w[1])
                                    .unwrap_or(0);
                                by_ply[bucket].push(d);
                            }
                        }
                    }
                    if tops.iter().any(|t| *t != tops[0]) {
                        top1_differs += 1;
                    }
                    positions += 1;
                }
            }

            board.play_unchecked(mv);
            let Some(id) = vocab.id_of(&played) else {
                break;
            };
            moves_so_far.push(id);
        }
    }

    if positions == 0 {
        return Err("no positions measured".into());
    }
    let n = positions as f64;

    println!("ckpt       {}", ckpt.display());
    println!("holdout    {pgn}");
    println!(
        "walk       {} positions from {games} games, side {side:?}{}",
        positions,
        walk_band.map(|b| format!(", band {b}")).unwrap_or_default()
    );
    println!("elapsed    {:.1?}", t0.elapsed());
    println!();
    println!("JS divergence in bits (0 = identical, 1 = disjoint)");
    for i in 0..n_bands {
        for j in (i + 1)..n_bands {
            let m = &pair_js[i * n_bands + j];
            println!(
                "  {} vs {}   mean {:.4}",
                shape.bands[i].token,
                shape.bands[j].token,
                m.value().unwrap_or(f64::NAN)
            );
            if let Some(v) = pair_js_agree[i * n_bands + j].value() {
                println!(
                    "    where top-1 agrees   {:.4}  (n={})",
                    v,
                    pair_js_agree[i * n_bands + j].n
                );
            }
            if let Some(v) = pair_js_differ[i * n_bands + j].value() {
                println!(
                    "    where top-1 differs  {:.4}  (n={})",
                    v,
                    pair_js_differ[i * n_bands + j].n
                );
            }
        }
    }
    println!();
    println!("reference: each band against a uniform draw over the same legal moves");
    for (b, m) in uniform_js.iter().enumerate() {
        println!(
            "  {} vs uniform   {:.4}",
            shape.bands[b].token,
            m.value().unwrap_or(f64::NAN)
        );
    }
    println!();
    println!(
        "top-1 differs between bands: {:.2}% of positions",
        100.0 * top1_differs as f64 / n
    );
    println!();
    println!(
        "widest pair ({} vs {}) by depth:",
        shape.bands[0].token,
        shape.bands[n_bands - 1].token
    );
    for (b, m) in by_ply.iter().enumerate() {
        let hi = ply_edges[b + 1];
        let label = if hi == usize::MAX {
            format!("ply {}+", ply_edges[b])
        } else {
            format!("ply {}-{}", ply_edges[b], hi - 1)
        };
        match m.value() {
            Some(v) => println!("  {label:<12} {v:.4}  (n={})", m.n),
            None => println!("  {label:<12} (no positions)"),
        }
    }
    println!();
    if !all_pair_js.is_empty() {
        all_pair_js.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f64| all_pair_js[((all_pair_js.len() - 1) as f64 * q) as usize];
        println!(
            "first pair spread: p50 {:.4}  p90 {:.4}  p99 {:.4}  max {:.4}",
            at(0.50),
            at(0.90),
            at(0.99),
            all_pair_js[all_pair_js.len() - 1]
        );
    }
    Ok(())
}

/// Vocabulary ids of every legal move in a position.
fn legal_move_ids(board: &Board, vocab: &MoveVocab) -> Vec<u32> {
    let mut ids = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            if let Some(id) = vocab.id_of(&uci_standard(board, mv)) {
                ids.push(id);
            }
        }
        false
    });
    ids
}
