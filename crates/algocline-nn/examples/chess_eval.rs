//! Score a baked chess model against games it never trained on.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_eval -- <ckpt.safetensors> <holdout.pgn> [min_elo] [max_elo] [side] [max_positions]
//! ```
//!
//! The holdout must come from a slice the bake did not read — a
//! different month, not a longer prefix of the same one.
//!
//! # What is measured
//!
//! **Move match** is the headline: at each position where the modelled
//! side is to move, does the model's pick equal what the human played?
//! It is the metric Maia established for this question, and it has a
//! ceiling well under 100% — the same player does not repeat the same
//! move in the same position — so the number is only meaningful next
//! to another model measured the same way.
//!
//! **Legal mass** and **raw legality** say something different: how
//! much of the model's raw distribution lands on moves that exist here
//! at all. The move actually played is legal by construction because
//! the ranking is walked against the legal set, so these are the only
//! readings that describe the model rather than the gate.
//!
//! **Uniform match** is the floor. Picking at random among the legal
//! moves scores it, so a model has to beat it to have learned
//! anything about the position rather than about the game.

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
use algocline_nn::chess::{model_config, CTX};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_eval: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let ckpt: PathBuf = args
        .next()
        .ok_or("usage: chess_eval <ckpt> <holdout.pgn> [min_elo] [max_elo] [side] [max_positions]")?
        .into();
    let pgn = args.next().ok_or("missing holdout pgn path")?;
    let min_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1600);
    let max_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1799);
    let side = match args.next().as_deref() {
        None | Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some("both") => ScoredSide::Both,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    let max_positions: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    let band_token = format!("<elo:{min_elo}-{max_elo}>");
    let vocab = MoveVocab::new(std::slice::from_ref(&band_token))?;
    let band_id = vocab.id_of(&band_token).ok_or("band token missing")?;
    let cfg = model_config(vocab.model_vocab_size(), Device::Cpu, DType::F32);
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;

    let filter = GameFilter::accept_all()
        .with_rating_band(min_elo, max_elo)
        .decided_on_the_board()
        .with_min_base_seconds(180)
        .with_ply_bounds(10, None);

    let mut reader = PgnReader::new(BufReader::new(File::open(&pgn)?));
    let t0 = Instant::now();
    let mut games = 0usize;
    let mut positions = 0usize;
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut legal_mass_sum = 0f64;
    let mut raw_legal = 0usize;
    let mut uniform_sum = 0f64;

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
        let mut row: Vec<u32> = vec![BOS, band_id];
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
            if scored && positions < max_positions && row.len() >= 2 {
                let window: Vec<u32> = row.iter().rev().take(CTX).rev().copied().collect();
                let input = Tensor::from_vec(window.clone(), (1, window.len()), &cfg.device)?;
                let logits = model.forward(&input)?;
                let last = logits.i((0, window.len() - 1))?;
                let probs = candle_nn::ops::softmax(&last, 0)?.to_vec1::<f32>()?;

                let mut legal: Vec<(String, f32)> = Vec::new();
                board.generate_moves(|moves| {
                    for m in moves {
                        let uci = uci_standard(&board, m);
                        let p = vocab
                            .id_of(&uci)
                            .and_then(|id| probs.get(id as usize).copied())
                            .unwrap_or(0.0);
                        legal.push((uci, p));
                    }
                    false
                });
                if !legal.is_empty() {
                    legal.sort_by(|a, b| b.1.total_cmp(&a.1));
                    legal_mass_sum += legal.iter().map(|(_, p)| *p as f64).sum::<f64>();
                    uniform_sum += 1.0 / legal.len() as f64;
                    if legal[0].0 == played {
                        top1 += 1;
                    }
                    if legal.iter().take(5).any(|(u, _)| *u == played) {
                        top5 += 1;
                    }
                    let raw_id = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                    if let Some(tok) = vocab.token_of(raw_id) {
                        if legal.iter().any(|(u, _)| u == tok) {
                            raw_legal += 1;
                        }
                    }
                    positions += 1;
                }
            }

            board.play_unchecked(mv);
            let Some(id) = vocab.id_of(&played) else {
                break;
            };
            row.push(id);
        }
    }

    if positions == 0 {
        return Err("no positions scored — is the holdout in this band?".into());
    }
    let n = positions as f64;
    println!("ckpt       {}", ckpt.display());
    println!("holdout    {pgn}");
    println!("side       {side:?}  band {min_elo}-{max_elo}");
    println!(
        "games      {games}   positions {positions}   in {:.1?}",
        t0.elapsed()
    );
    println!(
        "move match top1 {:.4}   top5 {:.4}",
        top1 as f64 / n,
        top5 as f64 / n
    );
    println!(
        "uniform    top1 {:.4}  (floor: random among legal)",
        uniform_sum / n
    );
    println!("legal mass {:.4}", legal_mass_sum / n);
    println!("raw legal  {:.4}", raw_legal as f64 / n);
    Ok(())
}
