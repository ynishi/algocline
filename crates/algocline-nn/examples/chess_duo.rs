//! Read the duo experiment's walks — one arm pair per cell — and ask
//! whether both condition slots are read, each in its own attribute's
//! direction.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_duo -- \
//!     <K-cell1> <K-b-cell1> [<K-cell2> <K-b-cell2> ...] \
//!     [--gamma G] [--seed S] [--draws D]
//! ```
//!
//! Walk files come in pairs: the primary seed's and the replicate's
//! walks over one cell's games. Which cell each pair is is read from
//! the records' own `games_of` (two tokens, format version 5), and a
//! pair whose header claims no known cell — or whose two walks claim
//! different cells — is refused before anything prints. Plan 08
//! registers **four** cells for a verdict; this program judges the
//! cells it is fed and prints how many that was, so a run over fewer
//! reads as what it is rather than as the registered grid.
//!
//! # What H30 asks
//!
//! For each cell (e, b), the cost of falsifying each slot alone:
//! `top1(e,b) - top1(e',b)` and `top1(e,b) - top1(e,b')`, judged over
//! ply 0-19 with the pair's seed gap as the floor. The month confirms
//! only when **every** (cell, slot) confirms — a slot that only works
//! in some cells, and a cell where only one slot works, are both what
//! the min-logic exists to keep out of a confirmation.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::records::AlignedArms;
use algocline_nn::chess::steerability::{
    self, ARM_K, ARM_K_B, DRAWS, JOUSEKI_SHALLOW, TOP1_GATE_JOUSEKI,
};
use algocline_nn::metric::bootstrap::{Interval, CONFIDENCE};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_duo: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Default seed for the resampling, named as the sibling judges' are.
const DEFAULT_SEED: u64 = 0x0812_2026;

const USAGE: &str = "usage: chess_duo <K-cell1> <K-b-cell1> [<K-cell2> <K-b-cell2> ...] \
                     [--gamma G] [--seed S] [--draws D]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut gamma: f32 = 1.0;
    let mut seed: u64 = DEFAULT_SEED;
    let mut draws: usize = DRAWS;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gamma" => gamma = args.next().ok_or(USAGE)?.trim().parse()?,
            "--seed" => seed = args.next().ok_or(USAGE)?.trim().parse()?,
            "--draws" => draws = args.next().ok_or(USAGE)?.trim().parse()?,
            _ => paths.push(arg.into()),
        }
    }
    if paths.is_empty() || !paths.len().is_multiple_of(2) {
        return Err(USAGE.into());
    }

    let t0 = Instant::now();
    let mut pairs = Vec::new();
    for pair in paths.chunks(2) {
        let arms = AlignedArms::read(&[
            (ARM_K.to_string(), pair[0].clone()),
            (ARM_K_B.to_string(), pair[1].clone()),
        ])?;
        // Resolved before anything prints; the same check the judge
        // repeats, run here so a swapped file is named next to its
        // path.
        let cell = steerability::check_duo_roles(&arms)?;
        pairs.push((arms, cell, [pair[0].clone(), pair[1].clone()]));
    }
    let read_elapsed = t0.elapsed();

    println!("cells (cell / role -> file)");
    for (arms, cell, paths) in &pairs {
        for (role, path) in [ARM_K, ARM_K_B].iter().zip(paths) {
            println!("  {} {role:<4} {}", cell.label, path.display());
        }
        println!(
            "  {} stream: {} position(s) from {} game(s)",
            cell.label,
            arms.positions(),
            arms.games(),
        );
    }
    println!(
        "resampling {draws} draws per cell, seed {seed}, {:.0}% percentile interval, \
         gamma {gamma}, judged over ply {}-{}",
        CONFIDENCE * 100.0,
        JOUSEKI_SHALLOW.0,
        JOUSEKI_SHALLOW.1 - 1,
    );

    println!();
    println!("admission at gamma {gamma} — recorded, not optimised");
    println!("  mean top-1 within {TOP1_GATE_JOUSEKI} of the pair's primary seed");
    let mut admitted = true;
    for (arms, cell, _) in &pairs {
        let gate = steerability::gate_top1_duo(arms, gamma, draws, seed)?;
        println!("  {} {:<34} {}", cell.label, gate.interval, gate.verdict());
        report_dropped_draws(&format!("{} top-1 gate", cell.label), &gate.interval);
        admitted &= gate.passes();
    }

    println!();
    println!("H30 — are both slots read, each in its own attribute's direction?");
    let arms_refs: Vec<&AlignedArms> = pairs.iter().map(|(arms, _, _)| arms).collect();
    let h30 = steerability::h30(&arms_refs, gamma, draws, seed)?;
    for (label, slots) in &h30.cells {
        println!("  cell {label}");
        for slot in slots {
            println!("    {}", slot.family);
            println!(
                "      cost {:+.6} (replicate {:+.6})   F {:+.6}",
                slot.cost_j, slot.cost_j_b, slot.floor
            );
            println!("      confirm on cost - F above zero    {}", slot.confirm);
            println!("      refute  on cost + F below zero    {}", slot.refute);
            report_dropped_draws(&format!("{} confirm", slot.family), &slot.confirm);
            report_dropped_draws(&format!("{} refute", slot.family), &slot.refute);
            println!(
                "      ({} position(s), {} game(s) in the judged stratum)",
                slot.positions, slot.games
            );
        }
    }
    println!();
    println!(
        "  verdict on this month over {} cell(s)   {}   (confirmed only when every \
         (cell, slot) confirms; plan 08 registers four cells)",
        h30.cells.len(),
        h30.verdict()
    );

    if !admitted {
        println!();
        println!(
            "NOTE: an admission gate did not pass, so the verdict above is descriptive and \
             not adopted."
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
