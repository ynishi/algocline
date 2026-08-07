//! Read several arms' per-position records and put error bars on the
//! differences between them.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_stats -- \
//!     <perpos.jsonl> <prefix.jsonl> <perpos-b.jsonl> [gamma] [seed] [draws]
//! ```
//!
//! The three files are the three arms, **in role order**, and the roles
//! are not interchangeable: the second is the arm the first is being
//! compared against, and the third is a second run of the first, whose
//! distance from it is what a between-arm margin has to beat.
//!
//! A swap between the first two would produce a full set of well-formed
//! numbers meaning something else, so it is refused before anything
//! prints: each arm's records carry the conditioning its checkpoint was
//! trained under, and a prefix-conditioned file in the per-position slot
//! is caught there. Passing the same weights as two arms is refused too
//! — their records are identical, so every difference between them is
//! zero in every draw, and if the pair is the two per-position runs that
//! silently removes the floor the margin has to beat.
//!
//! **What is not checked**, because nothing in a record could: which of
//! the two per-position files is which. They are two runs of one arm
//! differing only in a shuffle seed. Exchanging them leaves the same-arm
//! gap untouched, since it is symmetric in the pair, and makes the
//! margin the other replicate's — two equally good readings of one
//! experiment rather than a right one and a wrong one.
//!
//! The role-to-file mapping is still printed, but as something to read
//! rather than as the check.
//!
//! # Why this is a separate program from `chess_cond`
//!
//! Each arm is a separate checkpoint scored in a separate process, and
//! every hypothesis here is a difference **across arms scored on the
//! same positions**. Two summaries cannot be subtracted with an error
//! bar on the result: by the time each run has printed a mean, the
//! positions it came from are gone.
//!
//! # Why the error bars resample games
//!
//! Three thousand positions come from roughly ninety-five games, and
//! consecutive positions in one game share an opening and a pair of
//! players. Treating them as independent draws would narrow every
//! interval by roughly the square root of the positions per game.
//! Everything below is a cluster bootstrap over games — resample games
//! with replacement, recompute, take a percentile interval — and the
//! game count is printed beside every number for the same reason.
//!
//! Each interval names the seed it came from, so a reported figure can
//! be reproduced exactly.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::records::{AlignedArms, Walk};
use algocline_nn::chess::steerability::{self, DRAWS, PERPOS, PERPOS_B, PREFIX};
use algocline_nn::metric::bootstrap::{Interval, CONFIDENCE};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_stats: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Default seed for the resampling.
///
/// Named rather than drawn from the clock: an interval whose draws
/// cannot be reproduced is an interval nobody can check. Overridable so
/// a second seed can be run as a stability check, which is a different
/// thing from running seeds until one of them clears zero.
const DEFAULT_SEED: u64 = 0x0806_2026;

const USAGE: &str = "usage: chess_stats <perpos.jsonl> <prefix.jsonl> <perpos-b.jsonl> \
                     [gamma] [seed] [draws]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let perpos: PathBuf = args.next().ok_or(USAGE)?.into();
    let prefix: PathBuf = args.next().ok_or(USAGE)?.into();
    let perpos_b: PathBuf = args.next().ok_or(USAGE)?.into();
    let gamma: f32 = match args.next() {
        Some(text) => text.trim().parse()?,
        None => 1.0,
    };
    let seed: u64 = match args.next() {
        Some(text) => text.trim().parse()?,
        None => DEFAULT_SEED,
    };
    let draws: usize = match args.next() {
        Some(text) => text.trim().parse()?,
        None => DRAWS,
    };

    let t0 = Instant::now();
    let named = [
        (PERPOS.to_string(), perpos),
        (PREFIX.to_string(), prefix),
        (PERPOS_B.to_string(), perpos_b),
    ];
    let arms = AlignedArms::read(&named)?;
    // Before anything prints. The validity gate below is a difference
    // against the prefix arm, so under a swap it would be computed with
    // the per-position arm as the baseline — and a full, plausible,
    // wrong-baseline table would land on stdout with the refusal
    // arriving on stderr behind it. Refused rather than merely
    // displayed means refused first.
    steerability::check_roles(&arms)?;
    let read_elapsed = t0.elapsed();

    println!("arms (role -> file, and the checkpoint each was scored from)");
    for (role, path) in &named {
        let walk: &Walk = arms.walk(role)?;
        println!("  {:<9} {}", role, path.display());
        println!(
            "  {:<9} ckpt {} / {} conditioning / holdout {} / side {}",
            "", walk.header.ckpt, walk.header.encoding, walk.header.holdout, walk.header.side
        );
    }
    println!();
    println!(
        "stream     {} position(s) from {} game(s), ctx {}, bands {:?}",
        arms.positions(),
        arms.games(),
        arms.ctx(),
        arms.walk(PERPOS)?.header.bands,
    );
    println!(
        "resampling {draws} draws of {} game(s), seed {seed}, {:.0}% percentile interval",
        arms.games(),
        CONFIDENCE * 100.0,
    );
    println!("gamma      {gamma}");

    // Per-arm headline figures, each with its own interval.
    println!();
    println!("per arm at gamma {gamma} — point estimate and interval");
    println!(
        "  {:<9} {:<34} {:<34} legal mass",
        "arm", "flip rate", "top-1 match (mean of bands)"
    );
    for (role, _) in &named {
        let flip = steerability::flip_rate(&arms, role, gamma, draws, seed)?;
        let top1 = steerability::top1_match(&arms, role, gamma, draws, seed)?;
        let mass = steerability::legal_mass(&arms, role, gamma, draws, seed)?;
        println!("  {role:<9} {flip:<34} {top1:<34} {mass}");
        // Only `top1` can drop a draw — a position whose played move is
        // not in the vocabulary carries no top-1 value — but all three
        // are checked, because the cost is a comparison and the failure
        // it prevents is a figure resting on half its draws reading
        // exactly like one resting on all of them.
        report_dropped_draws(&format!("{role} flip rate"), &flip);
        report_dropped_draws(&format!("{role} top-1 match"), &top1);
        report_dropped_draws(&format!("{role} legal mass"), &mass);
    }

    // The validity gate, computed rather than left to the eye. Both are
    // stated as differences against the prefix arm, so they are
    // bootstrapped as differences over the same resampled games —
    // reading them off whether the three per-arm intervals above overlap
    // would be the one inference this machinery exists to rule out.
    println!();
    println!("validity gate at gamma {gamma} — recorded, not optimised");
    println!(
        "  gate 2  mean top-1 match within {:.2} of the prefix arm's",
        steerability::TOP1_GATE
    );
    println!(
        "  gate 3  legal mass within {:.0}% of the prefix arm's",
        steerability::LEGAL_MASS_GATE * 100.0
    );
    println!(
        "  {:<9} {:<34} {:<10} {:<34} gate 3",
        "arm", "top-1 delta vs prefix", "gate 2", "legal mass, relative"
    );
    for (role, _) in &named {
        let top1 = steerability::gate_top1(&arms, role, gamma, draws, seed)?;
        let mass = steerability::gate_legal_mass(&arms, role, gamma, draws, seed)?;
        println!(
            "  {role:<9} {:<34} {:<10} {:<34} {}",
            top1.interval,
            top1.verdict(),
            mass.interval,
            mass.verdict()
        );
        report_dropped_draws(&format!("{role} gate 2"), &top1.interval);
        report_dropped_draws(&format!("{role} gate 3"), &mass.interval);
    }
    println!(
        "  gate 1 (ce_legal < ce_uniform) is printed only in chess_cond's gamma=1 branch, so it \
         is a gamma=1 statement and is read off that summary rather than from here."
    );

    let h14 = steerability::h14(&arms, gamma, draws, seed)?;
    println!();
    println!("H14 — does conditioning at every position change the played move more often?");
    println!(
        "  M = flip(perpos) - flip(prefix)       {:+.6}   ({:.6} - {:.6})",
        h14.margin, h14.flip_perpos, h14.flip_prefix
    );
    println!(
        "  G = |flip(perpos) - flip(perpos-b)|   {:+.6}   (|{:.6} - {:.6}|)",
        h14.gap, h14.flip_perpos, h14.flip_perpos_b
    );
    println!("  confirm on M - G above zero           {}", h14.confirm);
    println!("  refute  on M + G below zero           {}", h14.refute);
    println!(
        "  verdict on this month                 {}   ({} position(s), {} game(s))",
        h14.verdict(),
        h14.positions,
        h14.games
    );
    report_dropped_draws("H14 confirm", &h14.confirm);
    report_dropped_draws("H14 refute", &h14.refute);

    let h15 = steerability::h15(&arms, gamma, draws, seed)?;
    println!();
    println!("H15 — does it also reach further down the game?");
    println!(
        "  shallow bucket  ply {}-{}   {} position(s) from {} game(s)",
        h15.shallow.low, h15.shallow.high, h15.shallow.positions, h15.shallow.games
    );
    println!(
        "  deep bucket     ply {}-{}  {} position(s) from {} game(s)",
        h15.deep.low, h15.deep.high, h15.deep.positions, h15.deep.games
    );
    if h15.deep.games < h15.shallow.games {
        println!(
            "                  the deep bucket rests on {} of the {} game(s) in the walk — \
             only those long enough to reach ply {}",
            h15.deep.games,
            arms.games(),
            h15.deep.low
        );
    }
    println!(
        "  ratio(perpos) = JS(deep)/JS(shallow)  {:.6}",
        h15.ratio_perpos
    );
    println!(
        "  ratio(prefix)                         {:.6}",
        h15.ratio_prefix
    );
    println!("  difference                            {}", h15.difference);
    println!("  verdict on this month                 {}", h15.verdict());
    report_dropped_draws("H15", &h15.difference);

    println!();
    println!(
        "Neither verdict is a decision. Both hypotheses are judged on two held-out months, \
         and only when the two months agree."
    );
    println!(
        "A confirmed H14 reads \"the margin exceeds one observed same-arm gap\": the gap is \
         one realisation from one pair of runs."
    );
    println!();
    println!(
        "elapsed    {:.1?} total, of which {:.1?} reading the records",
        t0.elapsed(),
        read_elapsed
    );
    Ok(())
}

/// Say so when draws were thrown away.
///
/// A draw whose statistic is undefined — an empty depth bucket, a zero
/// denominator — is left out of the percentiles, and leaving draws out
/// biases the interval. Silence about it would let a figure resting on
/// half its draws read exactly like one resting on all of them.
fn report_dropped_draws(label: &str, interval: &Interval) {
    // `label` is a `&str` rather than `&'static str` because the gate
    // rows name the arm they belong to.
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
