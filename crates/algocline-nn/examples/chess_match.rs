//! Play condition tokens against each other and against random play.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_match -- <ckpt> <pairs> [gamma] [seed] [max_plies]
//! ```
//!
//! # The question
//!
//! Everything measured so far about the condition token describes the
//! shape of its move distribution: 0.0110 bits of divergence between
//! the 1100 and 1900 conditions, ordered by how far apart the bands
//! are, amplified 12.5x by guidance. None of that says the conditions
//! differ in *strength*. Two policies can disagree about which move to
//! prefer and be equally good at chess.
//!
//! That distinction matters because the claim this line of work is
//! following — Allie (arXiv:2410.03893) conditioning a trajectory
//! model on Elo and getting playing strength that tracks the target —
//! is about strength, and strength has never been measured here. This
//! example measures it the only way it can be measured: by playing.
//!
//! # Design
//!
//! Two arms. **Band against band** tests the claim directly. **Each
//! band against a uniform random legal mover** anchors it: an even
//! result in the first arm cannot distinguish two equally strong
//! policies from two equally weak ones, and the anchor separates
//! those.
//!
//! Confounds are controlled by construction rather than by argument:
//!
//! - **Colour.** Every opening is played twice with the seats swapped,
//!   and the pair is one sample. White's advantage cancels.
//! - **Opening.** Games would otherwise be identical, because decoding
//!   is greedy. Each pair starts from `k` uniformly random legal plies,
//!   `k` drawn from 4..=8, and both arms see the same openings from the
//!   same seed.
//! - **Games that never end.** Capped, then adjudicated on material.
//!   Material is an objective reading of who stands better even from
//!   policies that demonstrably do not understand it.
//! - **A broken harness.** Random plays random as a control. Anything
//!   other than roughly even there invalidates the rest.
//!
//! Greedy decoding rather than sampling is deliberate: temperature
//! would add variance without adding signal, and the openings already
//! supply the diversity.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use cozy_chess::{Board, Color, GameStatus, Move, Piece};

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::guide::{guide_logits, mean_logits};
use algocline_nn::chess::pgn::uci_standard;
use algocline_nn::chess::vocab::{MoveVocab, BOS};
use algocline_nn::chess::{CondEncoding, ModelShape};

/// A seat: either a band of the model, or uniform random legal play.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Player {
    /// Index into the checkpoint's band list.
    Band(usize),
    /// Uniform draw over the legal moves.
    Random,
}

impl Player {
    fn label(&self, shape: &ModelShape) -> String {
        match self {
            Player::Band(i) => shape.bands[*i].token.clone(),
            Player::Random => "random".to_string(),
        }
    }
}

/// How a game ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ending {
    Checkmate,
    Stalemate,
    /// Fifty-move rule, as reported by the rules crate.
    FiftyMove,
    /// Same position three times.
    Repetition,
    /// Hit the ply cap and was adjudicated on material.
    Adjudicated,
}

/// Deterministic xorshift, so a seed reproduces an opening set exactly.
///
/// The standard library has no seeded RNG and the crate's `rand`
/// dependency is for sampling inside training; a five-line generator
/// keeps the opening set reproducible without either.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Zero is a fixed point of xorshift, so it must not be a seed.
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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Every legal move in a position.
fn legal_moves(board: &Board) -> Vec<Move> {
    let mut out = Vec::new();
    board.generate_moves(|moves| {
        out.extend(moves);
        false
    });
    out
}

/// Material balance from White's point of view, in pawns.
///
/// Used only to adjudicate a game that hit the cap. Kings are omitted
/// because both sides always have exactly one.
fn material(board: &Board) -> i32 {
    let value = |p: Piece| match p {
        Piece::Pawn => 1,
        Piece::Knight | Piece::Bishop => 3,
        Piece::Rook => 5,
        Piece::Queen => 9,
        Piece::King => 0,
    };
    let mut score = 0;
    for sq in cozy_chess::Square::ALL {
        if let (Some(piece), Some(colour)) = (board.piece_on(sq), board.color_on(sq)) {
            let v = value(piece);
            score += if colour == Color::White { v } else { -v };
        }
    }
    score
}

/// One seat's move chooser, holding everything the model needs.
struct Engine<'a> {
    model: &'a Gpt2Model,
    vocab: &'a MoveVocab,
    shape: &'a ModelShape,
    device: Device,
    gamma: f32,
}

impl Engine<'_> {
    /// Pick a move for `player` in `board`, given the move ids so far.
    ///
    /// The returned move is always legal: the model's ranking is walked
    /// against the legal set rather than trusted.
    fn choose(
        &self,
        player: Player,
        board: &Board,
        history: &[u32],
        rng: &mut Rng,
    ) -> Result<Move, Box<dyn std::error::Error>> {
        let legal = legal_moves(board);
        if legal.is_empty() {
            return Err("no legal moves".into());
        }
        let band = match player {
            Player::Random => return Ok(legal[rng.below(legal.len())]),
            Player::Band(i) => i,
        };

        // Rows differ only in the condition token, so guidance gets its
        // reference from the same forward.
        let need_all = self.gamma != 1.0;
        let bands: Vec<usize> = if need_all {
            (0..self.shape.bands.len()).collect()
        } else {
            vec![band]
        };
        let rows: Vec<Vec<u32>> = bands
            .iter()
            .map(|b| {
                let id = self
                    .vocab
                    .id_of(&self.shape.bands[*b].token)
                    .expect("band token in vocabulary");
                let mut row = vec![BOS, id];
                row.extend_from_slice(history);
                let start = row.len().saturating_sub(self.shape.ctx);
                row[start..].to_vec()
            })
            .collect();
        let width = rows[0].len();
        let input = Tensor::from_vec(rows.concat(), (rows.len(), width), &self.device)?;
        let out = self.model.forward(&input)?;
        let per_band: Vec<Vec<f32>> = (0..rows.len())
            .map(|b| out.i((b, width - 1))?.to_vec1::<f32>())
            .collect::<Result<_, _>>()?;

        let logits = if need_all {
            let reference = mean_logits(&per_band);
            guide_logits(&per_band[band], &reference, self.gamma)
        } else {
            per_band[0].clone()
        };

        // Highest-scoring legal move. Ties break on the first seen,
        // which is deterministic because move generation is.
        let mut best = legal[0];
        let mut best_score = f32::NEG_INFINITY;
        for mv in legal {
            let score = self
                .vocab
                .id_of(&uci_standard(board, mv))
                .and_then(|id| logits.get(id as usize).copied())
                .unwrap_or(f32::NEG_INFINITY);
            if score > best_score {
                best_score = score;
                best = mv;
            }
        }
        Ok(best)
    }
}

/// Result of one game, from White's point of view.
struct GameResult {
    /// 1.0 White won, 0.5 drawn, 0.0 Black won.
    white_score: f64,
    ending: Ending,
    plies: usize,
}

/// Play one game from `opening`, with the given seats.
fn play_game(
    engine: &Engine,
    opening: &[Move],
    white: Player,
    black: Player,
    max_plies: usize,
    rng: &mut Rng,
) -> Result<GameResult, Box<dyn std::error::Error>> {
    let mut board = Board::default();
    let mut history: Vec<u32> = Vec::new();
    let mut seen: HashMap<u64, u32> = HashMap::new();

    // The opening is shared by both seats and is not attributed to
    // either: it is the position the measurement starts from.
    for mv in opening {
        let id = engine
            .vocab
            .id_of(&uci_standard(&board, *mv))
            .ok_or("opening move not in vocabulary")?;
        board.play_unchecked(*mv);
        history.push(id);
    }
    *seen.entry(board.hash()).or_insert(0) += 1;

    let mut plies = opening.len();
    loop {
        match board.status() {
            GameStatus::Won => {
                // The side to move is checkmated, so the other won.
                let white_score = if board.side_to_move() == Color::White {
                    0.0
                } else {
                    1.0
                };
                return Ok(GameResult {
                    white_score,
                    ending: Ending::Checkmate,
                    plies,
                });
            }
            GameStatus::Drawn => {
                // cozy-chess reports stalemate and the fifty-move rule
                // through the same variant; they are separated here by
                // whether a move exists.
                let ending = if legal_moves(&board).is_empty() {
                    Ending::Stalemate
                } else {
                    Ending::FiftyMove
                };
                return Ok(GameResult {
                    white_score: 0.5,
                    ending,
                    plies,
                });
            }
            GameStatus::Ongoing => {}
        }
        if plies >= max_plies {
            let m = material(&board);
            let white_score = match m.cmp(&0) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Less => 0.0,
                std::cmp::Ordering::Equal => 0.5,
            };
            return Ok(GameResult {
                white_score,
                ending: Ending::Adjudicated,
                plies,
            });
        }

        let to_move = if board.side_to_move() == Color::White {
            white
        } else {
            black
        };
        let mv = engine.choose(to_move, &board, &history, rng)?;
        let id = engine
            .vocab
            .id_of(&uci_standard(&board, mv))
            .ok_or("chosen move not in vocabulary")?;
        board.play_unchecked(mv);
        history.push(id);
        plies += 1;

        let count = seen.entry(board.hash()).or_insert(0);
        *count += 1;
        if *count >= 3 {
            return Ok(GameResult {
                white_score: 0.5,
                ending: Ending::Repetition,
                plies,
            });
        }
    }
}

/// Tally for one arm.
#[derive(Default)]
struct Arm {
    /// Score for the first player, summed over pairs.
    score: f64,
    pairs: usize,
    wins: usize,
    draws: usize,
    losses: usize,
    /// Score for the first player when it had White.
    as_white: f64,
    as_black: f64,
    endings: HashMap<&'static str, usize>,
    plies: usize,
}

impl Arm {
    fn record(&mut self, r: &GameResult, first_is_white: bool) {
        let s = if first_is_white {
            r.white_score
        } else {
            1.0 - r.white_score
        };
        self.score += s;
        if s > 0.5 {
            self.wins += 1;
        } else if s < 0.5 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }
        if first_is_white {
            self.as_white += s;
        } else {
            self.as_black += s;
        }
        let name = match r.ending {
            Ending::Checkmate => "checkmate",
            Ending::Stalemate => "stalemate",
            Ending::FiftyMove => "fifty-move",
            Ending::Repetition => "repetition",
            Ending::Adjudicated => "adjudicated",
        };
        *self.endings.entry(name).or_insert(0) += 1;
        self.plies += r.plies;
    }

    /// Mean score per game, and its standard error.
    ///
    /// The error is the binomial one for `games` independent draws,
    /// which is conservative here: the two games of a pair share an
    /// opening and are therefore correlated.
    fn summary(&self) -> (f64, f64, usize) {
        let games = self.pairs * 2;
        if games == 0 {
            return (f64::NAN, f64::NAN, 0);
        }
        let mean = self.score / games as f64;
        let se = (mean * (1.0 - mean) / games as f64).sqrt();
        (mean, se, games)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_match: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let ckpt: PathBuf = args
        .next()
        .ok_or("usage: chess_match <ckpt> [pairs] [gamma] [seed] [max_plies]")?
        .into();
    let pairs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let gamma: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20260805);
    let max_plies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    // Which arms to run. The control costs nothing because it never
    // touches the model, so it can be run alone at a sample size that
    // actually settles whether the harness is fair.
    let only: Option<Vec<String>> = args
        .next()
        .map(|s| s.split(',').map(|p| p.trim().to_string()).collect());

    // Prefix-conditioned only: each seat's band is written into the row
    // in `Engine::choose`.
    //
    // That row is still tail-sliced, so past ply 126 both seats play
    // unconditioned and the arms measure one policy against itself.
    // Guidance goes with it: with every band's row identical,
    // `guide_logits(c, r, γ) = r + γ(c − r)` has `c == r` and returns
    // `c` whatever γ says, so a run asked for γ=4 is silently unguided
    // from that ply on.
    //
    // Games here run to the 200-ply cap, so this is not a rare regime —
    // it is left because the head-to-head arm was already retired on
    // its own evidence (86-88% of games ended in repetition) and no
    // conclusion in the current plan rests on this program.
    let shape = ModelShape::load_as(&ckpt, CondEncoding::Prefix)?;
    if shape.bands.len() < 2 {
        return Err("this checkpoint carries fewer than two bands".into());
    }
    let vocab = MoveVocab::new(&shape.band_tokens())?;
    let cfg = shape.config(
        Device::cuda_if_available(0).unwrap_or(Device::Cpu),
        DType::F32,
    );
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;
    let engine = Engine {
        model: &model,
        vocab: &vocab,
        shape: &shape,
        device: cfg.device.clone(),
        gamma,
    };

    let low = Player::Band(0);
    let high = Player::Band(shape.bands.len() - 1);

    // One opening set, shared by every arm so the arms are comparable.
    let mut opening_rng = Rng::new(seed);
    let mut openings: Vec<Vec<Move>> = Vec::with_capacity(pairs);
    while openings.len() < pairs {
        let mut board = Board::default();
        let k = 4 + opening_rng.below(5); // 4..=8
        let mut moves = Vec::with_capacity(k);
        let mut ok = true;
        for _ in 0..k {
            let legal = legal_moves(&board);
            if legal.is_empty() {
                ok = false;
                break;
            }
            let mv = legal[opening_rng.below(legal.len())];
            moves.push(mv);
            board.play_unchecked(mv);
        }
        // A random opening that already ended is not a starting point.
        if ok && board.status() == GameStatus::Ongoing {
            openings.push(moves);
        }
    }

    let arms: Vec<(&str, Player, Player)> = vec![
        ("high vs low", high, low),
        ("high vs random", high, Player::Random),
        ("low vs random", low, Player::Random),
        ("random vs random", Player::Random, Player::Random),
    ];

    println!("ckpt       {}", ckpt.display());
    println!("bands      {:?}", shape.band_tokens());
    println!(
        "setup      {pairs} pairs ({} games) per arm, gamma {gamma}, seed {seed}, cap {max_plies} plies",
        pairs * 2
    );
    println!("device     {:?}", cfg.device);
    println!();

    for (name, first, second) in arms {
        if let Some(want) = &only {
            if !want.iter().any(|w| name.contains(w.as_str())) {
                continue;
            }
        }
        let t0 = Instant::now();
        let mut arm = Arm::default();
        // A separate stream per arm, reset per pair, so a random seat's
        // draws do not depend on how many moves the other seat took.
        for (i, opening) in openings.iter().enumerate() {
            let mut rng = Rng::new(seed ^ ((i as u64 + 1) << 32));
            let g1 = play_game(&engine, opening, first, second, max_plies, &mut rng)?;
            arm.record(&g1, true);
            let mut rng = Rng::new(seed ^ ((i as u64 + 1) << 33));
            let g2 = play_game(&engine, opening, second, first, max_plies, &mut rng)?;
            arm.record(&g2, false);
            arm.pairs += 1;
        }
        let (mean, se, games) = arm.summary();
        let mut endings: Vec<(&str, usize)> = arm.endings.iter().map(|(k, v)| (*k, *v)).collect();
        endings.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        println!(
            "{name}  ({} vs {})",
            first.label(&shape),
            second.label(&shape)
        );
        println!(
            "  score {mean:.4} +/- {se:.4}   W/D/L {}/{}/{}   games {games}",
            arm.wins, arm.draws, arm.losses
        );
        println!(
            "  as white {:.4}   as black {:.4}   mean length {:.0} plies",
            arm.as_white / arm.pairs as f64,
            arm.as_black / arm.pairs as f64,
            arm.plies as f64 / games as f64
        );
        println!(
            "  endings {}",
            endings
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("  elapsed {:.1?}", t0.elapsed());
        println!();
    }

    println!("reading: 0.5 is even. The pre-registered threshold is 0.5 +/- 2 SE;");
    println!("inside it the null is not rejected rather than confirmed.");
    Ok(())
}
