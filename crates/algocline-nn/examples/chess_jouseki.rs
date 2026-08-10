//! Read the jouseki experiment's four walks — two families, two seeds
//! each — and ask whether an opening-family token steers toward its own
//! family's games.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_jouseki -- \
//!     <J-famB.jsonl> <J-b-famB.jsonl> <J-famC.jsonl> <J-b-famC.jsonl> \
//!     [gamma] [seed] [draws]
//! ```
//!
//! The first pair are the primary seed's and the replicate's walks over
//! the **first family's** games, the second pair over the second
//! family's. Which family each pair is is not taken from the command
//! line: it is read from the records' own `games_of` header (format
//! version 5), and a pair whose header names another family — or names
//! nothing — is refused before anything prints. That is the swap this
//! experiment is exposed to: the arms differ on no checkpoint axis, so
//! whose games were walked is the only thing a mix-up could change,
//! and it reverses the sign of the cost with every figure well-formed.
//!
//! # What H29 asks
//!
//! Plan 06's primary hypothesis. The cost of mis-conditioning —
//! `top1(correct family token) - top1(wrong one)` on the walk's own
//! family's games — cleared its seed floor on **both** families, or
//! not. One steering family is not a confirmation: the min-logic sits
//! at the verdict layer, and `undetermined` is the outcome that says
//! the token carries no direction, which is the reading plan 06
//! registers in advance.
//!
//! # The strata carry no verdict
//!
//! An opening family is a property of the early plies, so the cost is
//! printed for ply 0-19 and 20+ as description. Fading with depth is
//! expected, not disqualifying, and no stratum confirms or refutes
//! anything.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::records::{AlignedArms, Walk};
use algocline_nn::chess::steerability::{
    self, ARM_J, ARM_J_B, DRAWS, JOUSEKI_SHALLOW, TOP1_GATE_JOUSEKI,
};
use algocline_nn::metric::bootstrap::{Interval, CONFIDENCE};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_jouseki: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Default seed for the resampling, named as `chess_survival`'s is.
const DEFAULT_SEED: u64 = 0x0811_2026;

const USAGE: &str = "usage: chess_jouseki <J-famB.jsonl> <J-b-famB.jsonl> <J-famC.jsonl> \
                     <J-b-famC.jsonl> [gamma] [seed] [draws]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut next_path =
        || -> Result<PathBuf, &'static str> { Ok(args.next().ok_or(USAGE)?.into()) };
    let first = [next_path()?, next_path()?];
    let second = [next_path()?, next_path()?];
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
    let mut pairs = Vec::new();
    for paths in [&first, &second] {
        let arms = AlignedArms::read(&[
            (ARM_J.to_string(), paths[0].clone()),
            (ARM_J_B.to_string(), paths[1].clone()),
        ])?;
        // The family is what the walks themselves claim, and the claim
        // has to be a single token: the judge below re-checks it per
        // pair, but the name is needed here to ask the question at all.
        let family = match &arms.walk(ARM_J)?.header.games_of {
            Some(tokens) if tokens.len() == 1 => tokens[0].clone(),
            other => {
                return Err(format!(
                    "the walk at {} does not claim exactly one family (games_of = {other:?}); \
                     a jouseki pair is one family's games, stated in the record",
                    paths[0].display()
                )
                .into())
            }
        };
        pairs.push((arms, family, [paths[0].clone(), paths[1].clone()]));
    }
    if pairs[0].1 == pairs[1].1 {
        return Err(format!(
            "both pairs claim the same family {}; H29 needs the two families' walks",
            pairs[0].1
        )
        .into());
    }
    let read_elapsed = t0.elapsed();

    println!("arms (family / role -> file, and the checkpoint each was scored from)");
    for (arms, family, paths) in &pairs {
        for (role, path) in [ARM_J, ARM_J_B].iter().zip(paths) {
            let walk: &Walk = arms.walk(role)?;
            println!("  {family} {role:<4} {}", path.display());
            println!(
                "  {:<7} {:<4} ckpt {} / {} conditioning / games of {:?}",
                "",
                "",
                walk.header.ckpt,
                walk.header.encoding,
                walk.header.games_of.as_deref().unwrap_or_default(),
            );
        }
        println!(
            "  {family} stream: {} position(s) from {} game(s), bands {:?}",
            arms.positions(),
            arms.games(),
            arms.walk(ARM_J)?.header.bands,
        );
    }
    println!(
        "resampling {draws} draws per family, seed {seed}, {:.0}% percentile interval, \
         gamma {gamma}",
        CONFIDENCE * 100.0,
    );

    // Admission, before the hypothesis. The reference is the seed
    // replicate — the pair differ in nothing but the shuffle seed, so
    // the only thing this catches is a run defect in one slot.
    println!();
    println!("admission at gamma {gamma} — recorded, not optimised");
    println!("  mean top-1 within {TOP1_GATE_JOUSEKI} of the pair's primary seed");
    let mut admitted = true;
    for (arms, family, _) in &pairs {
        let gate = steerability::gate_top1_jouseki(arms, family, gamma, draws, seed)?;
        println!("  {family} {:<34} {}", gate.interval, gate.verdict());
        report_dropped_draws(&format!("{family} top-1 gate"), &gate.interval);
        admitted &= gate.passes();
    }

    println!();
    println!("H29 — does a family token steer toward its own family's games?");
    let (arms_a, family_a, _) = &pairs[0];
    let (arms_b, family_b, _) = &pairs[1];
    let h29 = steerability::h29(
        [(arms_a, family_a.as_str()), (arms_b, family_b.as_str())],
        gamma,
        draws,
        seed,
    )?;
    for family in &h29.families {
        println!("  family {}", family.family);
        println!(
            "    cost = top1(correct) - top1(wrong)  {:+.6}   (replicate {:+.6})",
            family.cost_j, family.cost_j_b
        );
        println!(
            "    F = |cost(J) - cost(J-b)|           {:+.6}",
            family.floor
        );
        println!("    confirm on cost - F above zero      {}", family.confirm);
        println!("    refute  on cost + F below zero      {}", family.refute);
        report_dropped_draws(&format!("{} confirm", family.family), &family.confirm);
        report_dropped_draws(&format!("{} refute", family.family), &family.refute);
        println!(
            "    strata (description, no verdict)    ply {}-{}: {}   ply {}+: {}",
            JOUSEKI_SHALLOW.0,
            JOUSEKI_SHALLOW.1 - 1,
            match family.shallow_cost_j {
                Some(v) => format!("{v:+.6}"),
                None => "(no positions)".into(),
            },
            JOUSEKI_SHALLOW.1,
            match family.deep_cost_j {
                Some(v) => format!("{v:+.6}"),
                None => "(no positions)".into(),
            },
        );
        println!(
            "    ({} position(s), {} game(s))",
            family.positions, family.games
        );
    }
    println!();
    println!(
        "  verdict on this month                 {}   (confirmed only when both families \
         confirm; one steering family is what the min exists to keep out)",
        h29.verdict()
    );

    if !admitted {
        println!();
        println!(
            "NOTE: an admission gate did not pass, so the verdict above is descriptive and not \
             adopted — the treatment plan 03's arms received, for the same reason."
        );
    }

    println!();
    println!("Judged on two held-out months and only when the two agree.");
    println!(
        "elapsed    {:.1?} total, of which {:.1?} reading the records",
        t0.elapsed(),
        read_elapsed
    );
    Ok(())
}

/// Say so when draws were thrown away, as every judge here does.
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
