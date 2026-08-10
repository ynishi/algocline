//! Does the model take a piece handed to it for nothing?
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_freebie -- \
//!     <holdout.pgn> <positions> <band> <gamma> <role>=<ckpt> [<role>=<ckpt> ...]
//! ```
//!
//! # The criterion is this project's own, and it was never measured
//!
//! `gpu-run-2026-08-05.md:71` set it:
//!
//! > 駒をタダで渡しても取らない相手は、どんな指標が良くても相手として
//! > 成立しない
//!
//! The evidence was three positions from one hand-played game against a
//! checkpoint trained on 236,800 rows. The verdict was then carried
//! through five days of documents and applied to checkpoints trained on
//! 600,000 rows under a different objective, without being re-measured
//! once. This program measures it.
//!
//! # Why a rate and not more positions played by hand
//!
//! Positions reached by hand depend on what the person playing chose to
//! hang, and that person is the one reading the result. The position
//! set here comes from a held-out month: every position where White is
//! to move and **some** capture is free, whoever left it hanging.
//!
//! # Why `chance` is an arm
//!
//! A take rate of 0.3 says nothing on its own. A policy picking
//! uniformly among legal moves takes free pieces at
//! `mean(n_free / n_legal)`, and that is the number the model has to
//! beat before "it takes them" means anything. It is computed from the
//! same positions rather than assumed, so a position set that happens
//! to be full of forced captures cannot flatter the model.
//!
//! # What "free" means
//!
//! [`algocline_nn::chess::freebie`] — a capture no legal move can
//! recapture on. Narrow on purpose: decidable from the board, no
//! evaluation function, and matching the narrowness of the criterion it
//! stands in for. It is not "winning".
//!
//! # What this does not measure
//!
//! Whether the model **hangs** its own pieces, which is a different
//! quantity and a different program. Whether it plays well: taking free
//! material is a floor, not a ceiling.

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device};
use cozy_chess::Board;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::batch::BandBatch;
use algocline_nn::chess::freebie::{judged_free_captures, JUDGED_VALUE};
use algocline_nn::chess::guide::{guide_logits, mean_logits};
use algocline_nn::chess::pgn::{resolve_san, san_tokens, uci_standard, PgnReader};
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::window::play_row;
use algocline_nn::chess::ModelShape;
use algocline_nn::metric::bootstrap::{cluster_bootstrap, ClusterTally, Interval, CONFIDENCE};

/// Bootstrap draws, fixed as everywhere else in this project: a draw
/// count chosen per invocation is one that can be raised until an
/// interval lands where someone wanted it.
const DRAWS: usize = 2000;

/// Default seed for the resampling.
const DEFAULT_SEED: u64 = 0x0806_2026;

const USAGE: &str = "usage: chess_freebie <holdout.pgn> <positions> <band> <gamma> \
                     <role>=<ckpt> [<role>=<ckpt> ...]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_freebie: {e}");
            ExitCode::FAILURE
        }
    }
}

/// One checkpoint, opened and ready to score.
struct Arm {
    role: String,
    path: PathBuf,
    shape: ModelShape,
    model: Gpt2Model,
    device: Device,
}

/// What one qualifying position contributed.
struct Position {
    game: usize,
    legal: usize,
    free_judged: usize,
    free_pawn: usize,
    /// Per arm, in the order the arms were given: did its top legal
    /// move take a judged free piece?
    took: Vec<bool>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let pgn: PathBuf = args.next().ok_or(USAGE)?.into();
    let max_positions: usize = args.next().ok_or(USAGE)?.trim().parse()?;
    let band = args.next().ok_or(USAGE)?;
    let gamma: f32 = args.next().ok_or(USAGE)?.trim().parse()?;

    let mut arms: Vec<Arm> = Vec::new();
    for spec in args {
        let (role, path) = spec.split_once('=').ok_or(
            "each arm is <role>=<ckpt>, so that a report names the arms rather than the paths",
        )?;
        let path = PathBuf::from(path);
        let shape = algocline_nn::chess::open_reader_shape(&path)?;
        let device = Device::Cpu;
        let cfg = shape.config(device.clone(), DType::F32);
        let model = Gpt2Model::from_safetensors_file(&cfg, &path)?;
        arms.push(Arm {
            role: role.to_string(),
            path,
            shape,
            model,
            device,
        });
    }
    if arms.is_empty() {
        return Err(USAGE.into());
    }

    let t0 = Instant::now();
    // The vocabulary carries the band tokens, so it is built from a
    // shape rather than from nothing. Every arm has to declare the same
    // bands: two arms whose band lists differ are two arms whose move
    // ids mean different things, and every rate below would be computed
    // against a different vocabulary while looking comparable.
    let bands = arms[0].shape.band_tokens();
    for arm in &arms[1..] {
        if arm.shape.band_tokens() != bands {
            return Err(format!(
                "arm {} declares bands {:?} where arm {} declares {:?}; their move ids are not \
                 the same ids",
                arm.role,
                arm.shape.band_tokens(),
                arms[0].role,
                bands
            )
            .into());
        }
    }
    let vocab = MoveVocab::new(&bands)?;
    let band_id = vocab
        .id_of(&band)
        .ok_or_else(|| format!("band {band} is not in the vocabulary"))?;

    println!("arms");
    for arm in &arms {
        println!(
            "  {:<8} {} / {} conditioning / ctx {}",
            arm.role,
            arm.path.display(),
            arm.shape.encoding,
            arm.shape.ctx
        );
    }
    println!();
    println!("holdout    {}", pgn.display());
    println!(
        "criterion  the top LEGAL move takes a piece worth {JUDGED_VALUE}+ that no legal move \
         can recapture"
    );
    println!("band       {band} / gamma {gamma}");

    let mut positions: Vec<Position> = Vec::new();
    let mut games = 0usize;
    let mut scanned = 0usize;
    let mut reader = PgnReader::new(BufReader::new(File::open(&pgn)?));
    'games: while let Some(game) = reader.next_game()? {
        let tokens = san_tokens(&game.movetext);
        if tokens.is_empty() {
            continue;
        }
        let game_index = games;
        games += 1;

        let mut board = Board::default();
        let mut moves_so_far: Vec<u32> = Vec::new();
        for (ply, token) in tokens.iter().enumerate() {
            let Ok(mv) = resolve_san(&board, token) else {
                break;
            };
            // White to move only: every checkpoint here was scored on
            // White, so a Black position is one no arm was trained to
            // answer.
            if ply.is_multiple_of(2) {
                scanned += 1;
                let free = judged_free_captures(&board);
                let legal = legal_ucis(&board, &vocab);
                if !free.is_empty() && legal.len() >= 2 {
                    let free_ucis: Vec<String> = free.iter().map(|c| c.mv.to_string()).collect();
                    let mut took = Vec::with_capacity(arms.len());
                    for arm in &arms {
                        let best = top_legal(arm, &vocab, band_id, gamma, &moves_so_far, &legal)?;
                        took.push(free_ucis.contains(&best));
                    }
                    positions.push(Position {
                        game: game_index,
                        legal: legal.len(),
                        free_judged: free.len(),
                        free_pawn: algocline_nn::chess::freebie::free_captures(&board)
                            .iter()
                            .filter(|c| c.value < JUDGED_VALUE)
                            .count(),
                        took,
                    });
                    if positions.len() >= max_positions {
                        break 'games;
                    }
                }
            }

            let played = uci_standard(&board, mv);
            let Some(id) = vocab.id_of(&played) else {
                break;
            };
            moves_so_far.push(id);
            board.play_unchecked(mv);
        }
    }

    if positions.is_empty() {
        return Err("no position in this holdout offered a free capture".into());
    }
    // Games are the bootstrap's clusters, as everywhere else: positions
    // inside one game share an opening and a pair of players.
    let mut cluster_of: BTreeMap<usize, usize> = BTreeMap::new();
    for p in &positions {
        let next = cluster_of.len();
        cluster_of.entry(p.game).or_insert(next);
    }
    let clusters = cluster_of.len();

    println!();
    println!(
        "scanned    {scanned} White position(s) over {games} game(s); {} of them offered a free \
         capture, from {clusters} game(s)",
        positions.len()
    );
    let pawn_only: usize = positions.iter().filter(|p| p.free_judged == 0).count();
    let free_pawns: usize = positions.iter().map(|p| p.free_pawn).sum();
    println!(
        "           {free_pawns} free pawn(s) sit in those positions and are reported here only \
         — the judged set is {JUDGED_VALUE}+ ({pawn_only} position(s) would be pawn-only)"
    );

    // `chance` is what a uniform pick among the legal moves takes, on
    // these positions. Bootstrapped over the same draws as the arms so
    // that a difference against it is a difference on one resample
    // rather than two intervals eyeballed for overlap.
    let mut chance = ClusterTally::new(clusters);
    for p in &positions {
        chance.push(cluster_of[&p.game], p.free_judged as f64 / p.legal as f64)?;
    }
    let mut rates: Vec<ClusterTally> = (0..arms.len())
        .map(|_| ClusterTally::new(clusters))
        .collect();
    for p in &positions {
        for (ix, took) in p.took.iter().enumerate() {
            rates[ix].push(cluster_of[&p.game], f64::from(u8::from(*took)))?;
        }
    }

    println!();
    println!("take rate at gamma {gamma} — point estimate and interval");
    println!("  arm      share of positions whose top legal move is a free capture");
    let chance_interval =
        cluster_bootstrap(clusters, DRAWS, DEFAULT_SEED, |draw| chance.mean_over(draw))?;
    println!(
        "  {:<8} {}   <- uniform pick among legal moves",
        "chance", chance_interval
    );
    for (arm, tally) in arms.iter().zip(&rates) {
        let interval =
            cluster_bootstrap(clusters, DRAWS, DEFAULT_SEED, |draw| tally.mean_over(draw))?;
        println!("  {:<8} {}", arm.role, interval);
        report_dropped_draws(&arm.role, &interval);
    }

    // H26, when the arms it names are present. Everything is recomputed
    // inside one draw, as in `h22`.
    let index_of = |role: &str| arms.iter().position(|a| a.role == role);
    if let (Some(p), Some(p_b), Some(a), Some(a_b)) = (
        index_of("P"),
        index_of("P-b"),
        index_of("A"),
        index_of("A-b"),
    ) {
        let margin_and_floor = |draw: &[usize]| -> Option<(f64, f64)> {
            let rp = rates[p].mean_over(draw)?;
            let rp_b = rates[p_b].mean_over(draw)?;
            let ra = rates[a].mean_over(draw)?;
            let ra_b = rates[a_b].mean_over(draw)?;
            let c = chance.mean_over(draw)?;
            Some((rp - c, (rp - rp_b).abs().max((ra - ra_b).abs())))
        };
        let confirm = cluster_bootstrap(clusters, DRAWS, DEFAULT_SEED, |draw| {
            margin_and_floor(draw).map(|(m, f)| m - f)
        })?;
        let refute = cluster_bootstrap(clusters, DRAWS, DEFAULT_SEED, |draw| {
            margin_and_floor(draw).map(|(m, f)| m + f)
        })?;
        let whole: Vec<usize> = (0..clusters).collect();
        let (margin, floor) = margin_and_floor(&whole).ok_or("the whole sample is undefined")?;

        println!();
        println!("H26 — does the adopted arm take a piece handed to it?");
        println!("  M = take(P) - chance                  {margin:+.6}");
        println!("  F = max of the two seed gaps          {floor:+.6}");
        println!("  confirm on M - F above zero           {confirm}");
        println!("  refute  on M + F below zero           {refute}");
        report_dropped_draws("H26 confirm", &confirm);
        report_dropped_draws("H26 refute", &refute);
        let verdict = match (
            confirm.excludes_zero_from_above(),
            refute.excludes_zero_from_below(),
        ) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        };
        println!("  verdict on this month                 {verdict}");
    } else {
        println!();
        println!(
            "H26 needs arms named P, P-b, A and A-b; the rates above are reported without a \
             verdict."
        );
    }

    // H27 carries no verdict by design: the old checkpoint differs in
    // rows, objective and conditioning at once, so a difference cannot
    // be attributed. It is printed because it is the number the
    // 2026-08-05 judgement was made from.
    if let (Some(p), Some(old)) = (index_of("P"), index_of("old")) {
        let difference = cluster_bootstrap(clusters, DRAWS, DEFAULT_SEED, |draw| {
            Some(rates[p].mean_over(draw)? - rates[old].mean_over(draw)?)
        })?;
        println!();
        println!("H27 — against the checkpoint the 2026-08-05 verdict was about (no verdict)");
        println!("  take(P) - take(old)                   {difference}");
        println!(
            "  Rows, objective and conditioning all differ, so a difference here names no cause. \
             It is printed because it is the quantity that judgement rested on."
        );
    }

    println!();
    println!(
        "resampling {DRAWS} draws of {clusters} game(s), seed {DEFAULT_SEED}, {:.0}% percentile \
         interval",
        CONFIDENCE * 100.0
    );
    println!("elapsed    {:.1?}", t0.elapsed());
    Ok(())
}

/// The legal moves of a position, in the vocabulary's spelling.
fn legal_ucis(board: &Board, vocab: &MoveVocab) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            let uci = uci_standard(board, mv);
            if let Some(id) = vocab.id_of(&uci) {
                out.push((uci, id));
            }
        }
        false
    });
    out
}

/// The move an arm would play: the highest-scoring **legal** move,
/// which is what `chess_play` decodes to.
///
/// The band reaches the model the way its own encoding says, through
/// `BandBatch`, so a prefix arm and a per-position arm are each asked
/// the question they were trained to answer.
fn top_legal(
    arm: &Arm,
    vocab: &MoveVocab,
    band_id: u32,
    gamma: f32,
    moves_so_far: &[u32],
    legal: &[(String, u32)],
) -> Result<String, Box<dyn std::error::Error>> {
    let window = play_row(Some(band_id), moves_so_far, arm.shape.ctx)?;
    let batch = BandBatch::over_bands(&window, &arm.shape, vocab, None)?;
    let band_logits = batch.logits(&arm.model, &arm.device)?;
    // The row was built at `band_id`, and `over_bands` puts the bands in
    // the shape's order, so the row for this band is the one at its
    // index. A shape with one band has one row and the index is zero.
    let ix = arm
        .shape
        .band_tokens()
        .iter()
        .position(|b| vocab.id_of(b) == Some(band_id))
        .unwrap_or(0);
    // The same guidance path `chess_cond` and `chess_play` take. At
    // gamma 1 this returns the band's own logits, so it is not a
    // special case to skip — writing it out is what keeps a gamma
    // argument from being accepted and then ignored.
    let reference = mean_logits(&band_logits);
    let logits = band_logits
        .get(ix)
        .ok_or("the band index is outside the rows the batch returned")?;
    let guided = guide_logits(logits, &reference, gamma);
    let best = legal
        .iter()
        .max_by(|a, b| {
            guided[a.1 as usize]
                .partial_cmp(&guided[b.1 as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or("a position with no legal move reached the scorer")?;
    Ok(best.0.clone())
}

/// Say so when draws were thrown away.
fn report_dropped_draws(label: &str, interval: &Interval) {
    if interval.undefined_draws > 0 {
        println!(
            "  note: {label} dropped {} of {} draw(s) as undefined; the interval rests on the \
             remaining {}",
            interval.undefined_draws,
            interval.undefined_draws + interval.draws,
            interval.draws
        );
    }
}
