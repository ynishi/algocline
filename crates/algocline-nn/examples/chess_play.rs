//! Play a baked chess model: give it the moves so far, get its reply.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_play -- <ckpt.safetensors> <band|cell|-> [moves...]
//! ```
//!
//! A multi-slot checkpoint is played under a **cell** — one band token
//! per condition group, comma-separated (`'<eco:B>,<elo:1100-1299>'`).
//! Its rows carry no condition token, so the conditions travel as
//! forward arguments and the printed header names every slot's table
//! row: playing two cells by hand is the observation plan 09's result
//! calls for, and it is worth nothing if the two runs silently played
//! the same conditions.
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
//!
//! # Both conditioning conventions
//!
//! This reader accepts either. It used to refuse a per-position
//! checkpoint, which meant the one step that asks whether the model is
//! any good as an *opponent* — playing it, by hand, as each band — could
//! not be run against the per-position arm at all. That is the step this
//! project has twice skipped and twice regretted: a boss that passed
//! every indicator and lost to one repeated action, and a chess model
//! that declined a free knight while its numbers looked fine.
//!
//! The row is the same under both conventions, `[BOS, band] + moves`.
//! What differs is that the per-position arm is *also* handed the band
//! as an index into a conditioning table of its own. Both facts live in
//! [`algocline_nn::chess::batch::BandBatch`], which pairs each row with
//! the conditioning that belongs to it and owns the only forward — in
//! the library rather than here, so that the construction this file
//! depends on is under test and shared with `chess_cond` instead of
//! written out twice.
//!
//! # And a checkpoint that was handed the rules
//!
//! It accepts those too. Such a checkpoint reads the ids legal at each
//! position, so it needs a set for every position of the row rather than
//! only for the board in front of it — and the replay this program does
//! in order to reach the position is what supplies them, mapped onto the
//! row by [`algocline_nn::chess::window::Window::legal_sets`]. Past the
//! ply where a game outgrows the context that mapping is not the
//! identity, which is where its documentation starts.
//!
//! # The regime this program is the one to reach
//!
//! A human game passes 126 plies about one time in 26 — 96.2% of games
//! fit whole in a 128-token context — and unlike the measurement path
//! this one has no bound at all. Past that ply the row is windowed, and
//! the tail slice this replaced dropped `BOS` and the band first, so the
//! model played unconditioned without saying so. `play_row` keeps the
//! prefix; under per-position the band also arrives as an argument,
//! which no amount of windowing can touch.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::{DType, Device, Tensor};
use cozy_chess::Board;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::batch::BandBatch;
use algocline_nn::chess::guide::{guide_logits, mean_logits};
use algocline_nn::chess::pgn::{move_from_uci_standard, resolve_san, uci_standard};
use algocline_nn::chess::vocab::{legal_ids, MoveVocab};
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
    if shape.effective_cond_groups().len() > 1 {
        // A multi-slot checkpoint is played under a cell — one token
        // per group — and this resolver answers about one band. Refused
        // rather than answered for the first group, which would play
        // the other attribute on whatever row happened to be zero.
        return Err(format!(
            "this checkpoint conditions in {} slots; name one band per slot, \
             comma-separated (e.g. '<eco:B>,<elo:1100-1299>'), from {:?}",
            shape.effective_cond_groups().len(),
            shape.band_tokens()
        ));
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
        .ok_or(
            "usage: chess_play <ckpt.safetensors> <band|cell|-> [moves...]\n\
             a multi-slot checkpoint takes a cell: '<eco:B>,<elo:1100-1299>'",
        )?
        .into();
    // Either conditioning convention. This used to be
    // `load_as(.., Prefix)`, which refused a per-position checkpoint —
    // and this is the program the plan's last step plays the adopted arm
    // with, so refusing here meant the adoption decision would rest on
    // indicators alone for one of the two arms. `Batch` below carries
    // the branch; nothing else in this file consults the encoding.
    //
    // The other axis too, which this reader used to be refused on. A
    // checkpoint trained with the legal moves supplied at every position
    // needs them supplied here as well — for every position of the row,
    // not only for the board in front of it — and past the ply where a
    // game outgrows the context the row no longer carries the history
    // the earlier ones would come from. They are taken from the replay
    // below, which this file performs anyway in order to reach the
    // position, and mapped onto the row by `Window::legal_sets`.
    //
    // `BandBatch` is where the requirement is enforced: it takes the
    // sets as an argument and refuses a legality checkpoint that arrives
    // without them, so opening the shape without a gate skips nothing.
    let shape = ModelShape::load_any(&ckpt)?;
    let band_arg = args.next().unwrap_or_else(|| "-".into());
    let played: Vec<String> = args.collect();

    // The vocabulary is rebuilt from the checkpoint's own band list, so
    // the ids line up with what it was trained on rather than with
    // whatever the caller happened to type.
    let tokens = shape.band_tokens();
    let vocab = MoveVocab::new(&tokens)?;
    // A multi-slot checkpoint is played under a cell: one band token
    // per condition group, comma-separated. Its rows carry no condition
    // token, so nothing below writes one into the window and the
    // conditions travel as forward arguments.
    let multi = shape.effective_cond_groups().len() > 1;
    let cell: Vec<String> = if multi {
        let cell: Vec<String> = band_arg
            .split(',')
            .map(|part| part.trim().to_string())
            .collect();
        for token in &cell {
            if shape.band(token).is_none() {
                return Err(format!(
                    "checkpoint has no band {token}; it carries {:?}",
                    shape.band_tokens()
                )
                .into());
            }
        }
        cell
    } else {
        Vec::new()
    };
    let band_token = if multi {
        None
    } else {
        resolve_band(&shape, &band_arg)?
    };

    // Replay what has been played so far, accepting either notation.
    let mut board = Board::default();
    let band_id = match &band_token {
        Some(token) => Some(vocab.id_of(token).ok_or("band token missing")?),
        None => None,
    };
    let mut moves: Vec<u32> = Vec::new();
    let mut history: Vec<String> = Vec::new();
    // The ids legal at each ply, entry `m` being the position reached
    // after `m` moves — collected only for a checkpoint that reads them,
    // so an ordinary one replays exactly as it did.
    let wants_legal = shape.legal_input;
    let mut at_ply: Vec<Vec<u32>> = Vec::new();
    for token in &played {
        let mv = match move_from_uci_standard(&board, token) {
            Some(mv) => mv,
            None => resolve_san(&board, token)
                .map_err(|e| format!("cannot read move {token:?}: {e}"))?,
        };
        let uci = uci_standard(&board, mv);
        // Before the move is played: this is the set `uci` was drawn
        // from, which is what the position it sits at was choosing among.
        if wants_legal {
            at_ply.push(legal_ids(&board, &vocab));
        }
        board.play_unchecked(mv);
        let id = vocab
            .id_of(&uci)
            .ok_or_else(|| format!("move {uci} is not in the vocabulary"))?;
        moves.push(id);
        history.push(uci);
    }
    // And the position on the board now, which is the set the move this
    // program is about to choose is drawn from.
    if wants_legal {
        at_ply.push(legal_ids(&board, &vocab));
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
    // One list for the row, whichever branch below builds the batch: the
    // per-band rows differ only at the band token, which holds no move.
    // On a windowed row the row's first move is the game's `start`-th,
    // and mapping that is `legal_sets`'s business rather than this
    // file's.
    //
    // `moves.len()` is how many plies `at_ply` is supposed to cover, and
    // it is passed rather than left to be inferred: `legal_sets` aligns
    // by taking the tail, so a history that covered more plies than this
    // row was built from would map every position later in the game than
    // the token it describes, with every length still agreeing. The two
    // come from the same replay above, one entry per move plus the
    // position on the board, and this is what makes that a stated
    // contract rather than a property of where the pushes sit.
    let sets = match wants_legal {
        true => Some(window.legal_sets(&at_ply, moves.len())?),
        false => None,
    };
    let guided_band = band_token.as_ref().filter(|_| gamma != 1.0);
    let logits: Vec<f32> = if multi {
        // Guidance is not defined for a cell here: its reference is the
        // mean over the band rows, and with two slots that mean is over
        // combinations rather than bands. Playing a cell is what plan
        // 09 asks for, and inventing a reference for it would be a
        // measurement nobody registered — so gamma is refused rather
        // than silently ignored.
        if gamma != 1.0 {
            return Err(
                "guidance is not defined for a multi-slot cell; run at gamma 1 (CHESS_GAMMA=1)"
                    .into(),
            );
        }
        BandBatch::single_combo(&window, &shape, &cell, sets.as_deref())?
            .logits(&model, &cfg.device)?
            .into_iter()
            .next()
            .ok_or("the single-row batch came back empty")?
    } else if let Some(asked) = guided_band {
        // Guidance needs the other bands to build the reference it
        // extrapolates away from, so every band is run at this position
        // and the requested one is pushed away from their mean.
        //
        // The reference is the same object under both conventions, and
        // deliberately so. `mean_logits` averages the model's output
        // under each band it carries — the population the bands were
        // split out of — and nothing in that description mentions how
        // the band reached the model.
        //
        // Under per-position there is a second candidate: run the batch
        // without its conditioning indices. That is not an unconditional
        // branch either. It removes one of the two channels the band
        // arrives through and leaves a row that still carries the band
        // token at position 1, so what comes back is a partially
        // conditioned model — and one trained with neither channel
        // missing, since nothing here uses condition dropout. Off
        // distribution in a way the band mean is not, and on top of that
        // a different reference on one arm than on the other would make
        // the two arms' guided numbers incomparable, which is the one
        // thing the comparison exists to avoid.
        let batch = BandBatch::over_bands(&window, &shape, &vocab, sets.as_deref())?;
        let per_band = batch.logits(&model, &cfg.device)?;
        let asked_at = shape
            .bands
            .iter()
            .position(|b| &b.token == asked)
            .ok_or_else(|| format!("band {asked} is not among this checkpoint's"))?;
        let conditioned = per_band
            .get(asked_at)
            .ok_or("the band batch came back short")?;
        let reference = mean_logits(&per_band);
        guide_logits(conditioned, &reference, gamma)
    } else {
        let batch = BandBatch::single(&window, &shape, band_token.as_deref(), sets.as_deref())?;
        batch
            .logits(&model, &cfg.device)?
            .into_iter()
            .next()
            .ok_or("the single-row batch came back empty")?
    };
    let width = logits.len();
    let t = Tensor::from_vec(logits, (width,), &cfg.device)?;
    let probs = candle_nn::ops::softmax(&t, 0)?.to_vec1::<f32>()?;

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

    // Printed because this is the step where a person plays the model by
    // hand and has to be able to tell that the band actually reached it.
    // "The same position gives different moves under the two bands" is
    // the check, and it is worth nothing if the two runs silently played
    // as the same band.
    println!(
        "band       {}  ({} conditioning{})",
        match multi {
            true => cell.join("+"),
            false => band_token.as_deref().unwrap_or("(none)").to_string(),
        },
        shape.encoding,
        match (multi, shape.encoding) {
            // Every slot's row, so a person playing two cells by hand
            // can see that both attributes reached the model and which
            // rows they took.
            (true, _) => format!(
                ", table rows {}",
                cell.iter()
                    .filter_map(|t| shape.band_index(t))
                    .map(|c| c.row().to_string())
                    .collect::<Vec<_>>()
                    .join("+")
            ),
            (false, CondEncoding::Prefix) => String::new(),
            (false, CondEncoding::EveryPosition) => format!(
                ", table row {}",
                band_token
                    .as_deref()
                    .and_then(|t| shape.band_index(t))
                    .map(|c| c.row().to_string())
                    .unwrap_or_else(|| "-".into())
            ),
        }
    );
    // Only for a checkpoint that has the axis, so playing an ordinary
    // one prints what it always printed. For one that does, this says
    // the channel it was trained with is being fed — a person playing
    // the model by hand has no other sight of it, and a row scored
    // without it would produce entirely ordinary-looking moves.
    if wants_legal {
        println!(
            "legality   supplied at every position of a {}-token row",
            window.len()
        );
    }
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
