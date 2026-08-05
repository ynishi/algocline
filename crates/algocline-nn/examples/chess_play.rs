//! Play a baked chess model: give it the moves so far, get its reply.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_play -- <ckpt.safetensors> <min_elo> <max_elo> [moves...]
//! ```
//!
//! Moves may be UCI (`e2e4`) or SAN (`Nf3`); both are resolved against
//! the position, so a game can be typed the way it reads. With no
//! moves the model is on move one as White.
//!
//! The report is the model's pick plus the five most likely legal
//! moves with their share of the legal mass, and — separately — how
//! much mass the raw distribution put on illegal moves. That last
//! number is the one that says something about the model: the move
//! actually played is legal by construction, because the ranking is
//! walked against the legal set before anything is chosen.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::{DType, Device, IndexOp, Tensor};
use cozy_chess::Board;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::pgn::{move_from_uci_standard, resolve_san, uci_standard};
use algocline_nn::chess::vocab::{MoveVocab, BOS};
use algocline_nn::chess::{model_config, CTX};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_play: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let ckpt: PathBuf = args
        .next()
        .ok_or("usage: chess_play <ckpt.safetensors> [min_elo] [max_elo] [moves...]")?
        .into();
    let min_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1600);
    let max_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1799);
    let played: Vec<String> = args.collect();

    let band_token = format!("<elo:{min_elo}-{max_elo}>");
    let vocab = MoveVocab::new(std::slice::from_ref(&band_token))?;

    // Replay what has been played so far, accepting either notation.
    let mut board = Board::default();
    let mut row: Vec<u32> = vec![BOS, vocab.id_of(&band_token).ok_or("band token missing")?];
    let mut history: Vec<String> = Vec::new();
    for token in &played {
        let mv = match move_from_uci_standard(&board, token) {
            Some(mv) => mv,
            None => resolve_san(&board, token)
                .map_err(|e| format!("cannot read move {token:?}: {e}"))?,
        };
        let uci = uci_standard(&board, mv);
        board.play_unchecked(mv);
        let id = vocab
            .id_of(&uci)
            .ok_or_else(|| format!("move {uci} is not in the vocabulary"))?;
        row.push(id);
        history.push(uci);
    }

    let cfg = model_config(vocab.model_vocab_size(), Device::Cpu, DType::F32);
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;

    // Only the last CTX tokens fit; the model sees the tail.
    let window: Vec<u32> = row.iter().rev().take(CTX).rev().copied().collect();
    let input = Tensor::from_vec(window.clone(), (1, window.len()), &cfg.device)?;
    let logits = model.forward(&input)?;
    let last = logits.i((0, window.len() - 1))?;
    let probs = candle_nn::ops::softmax(&last, 0)?.to_vec1::<f32>()?;

    // Rank the legal moves by the model's mass, and note how much mass
    // fell outside the legal set.
    let mut legal: Vec<(String, f32)> = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            let uci = uci_standard(&board, mv);
            let p = vocab
                .id_of(&uci)
                .and_then(|id| probs.get(id as usize).copied())
                .unwrap_or(0.0);
            legal.push((uci, p));
        }
        false
    });
    if legal.is_empty() {
        println!("status     {:?} — no legal moves", board.status());
        return Ok(());
    }
    legal.sort_by(|a, b| b.1.total_cmp(&a.1));
    let legal_mass: f32 = legal.iter().map(|(_, p)| p).sum();

    // What the raw argmax would have played, legal or not.
    let raw_id = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let raw_token = vocab.token_of(raw_id).unwrap_or("(none)");
    let raw_legal = legal.iter().any(|(uci, _)| uci == raw_token);

    println!("ply        {}", history.len());
    println!("to move    {:?}", board.side_to_move());
    println!("history    {}", history.join(" "));
    println!("legal      {} moves", legal.len());
    println!(
        "raw argmax {raw_token} ({})",
        if raw_legal { "legal" } else { "ILLEGAL" }
    );
    println!("legal mass {:.4} of the distribution", legal_mass);
    println!("top 5 legal:");
    for (uci, p) in legal.iter().take(5) {
        let share = if legal_mass > 0.0 {
            p / legal_mass
        } else {
            0.0
        };
        println!("  {uci}  p={p:.4}  share={share:.3}");
    }
    println!("PLAYS      {}", legal[0].0);
    Ok(())
}
