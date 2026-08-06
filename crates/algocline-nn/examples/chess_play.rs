//! Play a baked chess model: give it the moves so far, get its reply.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_play -- <ckpt.safetensors> <band|-> [moves...]
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
use algocline_nn::chess::guide::{guide_logits, mean_logits};
use algocline_nn::chess::pgn::{move_from_uci_standard, resolve_san, uci_standard};
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::window::play_row;
use algocline_nn::chess::{CondEncoding, ModelShape};

/// Pick which band the model plays as.
///
/// `-` means unconditioned, which is only valid for a checkpoint that
/// was trained without bands. Anything else must name a band the
/// checkpoint carries: asking for one it was never trained on would
/// index a token it has no meaning for.
fn resolve_band(shape: &ModelShape, arg: &str) -> Result<Option<String>, String> {
    if arg == "-" {
        return if shape.bands.is_empty() {
            Ok(None)
        } else {
            Err(format!(
                "this checkpoint is conditional; pass one of {:?}",
                shape.band_tokens()
            ))
        };
    }
    // Accept either the token or the bare range.
    let token = if arg.starts_with('<') {
        arg.to_string()
    } else {
        format!("<elo:{arg}>")
    };
    match shape.band(&token) {
        Some(_) => Ok(Some(token)),
        None => Err(format!(
            "checkpoint has no band {token}; it carries {:?}",
            shape.band_tokens()
        )),
    }
}

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
        .ok_or("usage: chess_play <ckpt.safetensors> <band|-> [moves...]")?
        .into();
    // The band is played by writing it into the row, so a checkpoint
    // conditioned the other way is refused rather than played with its
    // condition ignored — the moves would look ordinary either way.
    let shape = ModelShape::load_as(&ckpt, CondEncoding::Prefix)?;
    let band_arg = args.next().unwrap_or_else(|| "-".into());
    let played: Vec<String> = args.collect();

    // The vocabulary is rebuilt from the checkpoint's own band list, so
    // the ids line up with what it was trained on rather than with
    // whatever the caller happened to type.
    let tokens = shape.band_tokens();
    let vocab = MoveVocab::new(&tokens)?;
    let band_token = resolve_band(&shape, &band_arg)?;

    // Replay what has been played so far, accepting either notation.
    let mut board = Board::default();
    let band_id = match &band_token {
        Some(token) => Some(vocab.id_of(token).ok_or("band token missing")?),
        None => None,
    };
    let mut moves: Vec<u32> = Vec::new();
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
        moves.push(id);
        history.push(uci);
    }

    let cfg = shape.config(Device::Cpu, DType::F32);
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;

    // Guidance strength. Read from the environment because the move
    // list is variadic and cannot share the tail of the argument list.
    // One is the unguided model; above one the band token's effect is
    // amplified (see `chess::guide`).
    let gamma: f32 = env::var("CHESS_GAMMA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    // A game longer than the context window is cut down to fit, keeping
    // `[BOS, band]` and windowing the moves. 3.8% of games — about one
    // in 26 — do not fit whole, with a ply p90 of 109 and a maximum of
    // 276, and the tail slice this replaces dropped exactly the two
    // tokens that are not moves, so from that ply on the model played
    // unconditioned without saying so.
    //
    // `play_row` decides the prefix length from the band itself, so the
    // window cannot disagree with the guidance rewrite below about
    // which position the condition is at.
    let window = play_row(band_id, &moves, shape.ctx)?;
    let width = window.len();
    let guided_band = band_token.as_ref().filter(|_| gamma != 1.0);
    let probs = if let Some(asked) = guided_band {
        // Guidance needs the other bands to build the reference it
        // extrapolates away from, so every band is run at this
        // position and the requested one is pushed away from their
        // mean.
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(shape.bands.len());
        let mut asked_at = 0usize;
        for (i, band) in shape.bands.iter().enumerate() {
            let id = vocab.id_of(&band.token).ok_or("band token missing")?;
            // Lands on the condition, or refuses: `Window` knows
            // whether it has one. The same write against a tail-sliced
            // `Vec` put a band id on top of a real move of the position
            // being evaluated (issue 8f9a96df).
            rows.push(window.with_band(id)?.into_tokens());
            if &band.token == asked {
                asked_at = i;
            }
        }
        let n = rows.len();
        let input = Tensor::from_vec(rows.concat(), (n, width), &cfg.device)?;
        let out = model.forward(&input)?;
        let per_band: Vec<Vec<f32>> = (0..n)
            .map(|b| out.i((b, width - 1))?.to_vec1::<f32>())
            .collect::<Result<_, _>>()?;
        let reference = mean_logits(&per_band);
        let guided = guide_logits(&per_band[asked_at], &reference, gamma);
        let t = Tensor::from_vec(guided, (vocab.model_vocab_size(),), &cfg.device)?;
        candle_nn::ops::softmax(&t, 0)?.to_vec1::<f32>()?
    } else {
        let input = Tensor::from_vec(window.tokens().to_vec(), (1, width), &cfg.device)?;
        let logits = model.forward(&input)?;
        let last = logits.i((0, width - 1))?;
        candle_nn::ops::softmax(&last, 0)?.to_vec1::<f32>()?
    };

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

    println!("gamma      {gamma}");
    println!("ply        {}", history.len());
    println!("to move    {:?}", board.side_to_move());
    println!("history    {}", history.join(" "));
    println!("legal      {} moves", legal.len());
    println!(
        "raw argmax {raw_token} ({})",
        if raw_legal { "legal" } else { "ILLEGAL" }
    );
    println!("legal mass {:.4} of the distribution", legal_mass);
    // How many of the ranked moves to show. Five is enough to see what
    // the model wants to play; asking a question about a specific move
    // — does this band fall for this trap? — needs the whole ranking,
    // because the interesting move is usually the one that did not make
    // the cut.
    let shown: usize = env::var("CHESS_SHOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
        .min(legal.len());
    println!("top {shown} legal:");
    for (uci, p) in legal.iter().take(shown) {
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
