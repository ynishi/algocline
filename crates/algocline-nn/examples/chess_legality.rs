//! Read the legality experiment's five arms together and put error
//! bars on the differences between them.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_legality -- \
//!     <control.jsonl> <A.jsonl> <A-b.jsonl> <AB.jsonl> <AB-b.jsonl> \
//!     [gamma] [seed] [draws]
//! ```
//!
//! The five files are the five arms, **in role order**: the control
//! (the arm plan 02 already baked), the two runs of `A` — loss over the
//! legal moves, legality not supplied as an input — and the two runs of
//! `AB`, which also hand the legal ids to the forward pass.
//!
//! # Why this is a separate program from `chess_stats`
//!
//! Not because five arguments are more than three. The two programs
//! judge different arm sets under different pre-registered criteria:
//! `chess_stats` reads `perpos` / `prefix` / `perpos-b` against plan
//! 02's top-1 tolerance and its role check, and this reads `prefix` /
//! `A` / `A-b` / `AB` / `AB-b` against `§5.1`'s wider one and a role
//! check that looks at a second axis. Folding both into one program
//! means branching on how many paths were passed, and a branch is
//! exactly where a check gets skipped for the arm set that took the
//! other one. The six walks plan 02 was judged on also keep the program
//! that produced their figures, unchanged.
//!
//! # What is refused before anything prints
//!
//! Every arm here is prefix-conditioned (`§3.1` makes it a constraint,
//! not a choice), so the conditioning separates none of them and the
//! axis that tells them apart is whether the legal set reached the
//! model. Exchange `A` and `AB` on this command line and
//! `D = ce_legal(A) - ce_legal(AB)` changes sign while every printed
//! figure stays well-formed. So the role check runs on the arms as
//! read, before the first line of output — plan 02's failure was a
//! full, plausible table reaching the reader ahead of the first
//! refusal. A walk written before record format version 4 could not
//! state the axis at all, and is refused as unverifiable rather than
//! believed on the `false` it defaults to.
//!
//! **What is not checked**, because nothing in a record could: which of
//! `A` and `A-b` is which, or of `AB` and `AB-b`. They are two runs of
//! one recipe differing only in a shuffle seed. Exchanging a pair
//! leaves the floor untouched — it is symmetric in the pair — and makes
//! the difference the other replicate's, which is a second reading of
//! one experiment rather than a wrong one.
//!
//! # Why the floor is printed beside the verdict
//!
//! The walk is sized in positions and the game count is an outcome, so
//! the resolution a run actually bought is not the one `§5.3` estimated
//! for it. An undetermined H21 at a floor of 0.09 nats and one at 0.04
//! are different findings — the first says the instrument was blunt,
//! the second that the effect is small — and the two are
//! indistinguishable if only the verdict is reported. The floor here is
//! read off the games this run resampled.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::records::{AlignedArms, Walk};
use algocline_nn::chess::steerability::{
    self, Resolution, ARM_A, ARM_AB, ARM_AB_B, ARM_A_B, CONTROL, DRAWS, LEGALITY_ARMS,
};
use algocline_nn::metric::bootstrap::{Interval, CONFIDENCE};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_legality: {e}");
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

const USAGE: &str = "usage: chess_legality <control.jsonl> <A.jsonl> <A-b.jsonl> <AB.jsonl> \
                     <AB-b.jsonl> [gamma] [seed] [draws]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut next_path =
        || -> Result<PathBuf, &'static str> { Ok(args.next().ok_or(USAGE)?.into()) };
    let control = next_path()?;
    let plain = next_path()?;
    let plain_b = next_path()?;
    let legal = next_path()?;
    let legal_b = next_path()?;
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
        (CONTROL.to_string(), control),
        (ARM_A.to_string(), plain),
        (ARM_A_B.to_string(), plain_b),
        (ARM_AB.to_string(), legal),
        (ARM_AB_B.to_string(), legal_b),
    ];
    let arms = AlignedArms::read(&named)?;
    // Before anything prints, and in this order. Every arm is
    // prefix-conditioned, so a swap of `A` and `AB` leaves every figure
    // below well-formed while reversing what the difference means; the
    // legality axis is the only thing that refuses it, and a table that
    // reached the reader first would be a table nobody had reason to
    // doubt. The second check asks the format version rather than the
    // fields, because a stale walk's absent cost would otherwise read
    // as a walk of positions nothing could score.
    steerability::check_legality_roles(&arms)?;
    steerability::check_scoreable(&arms, &LEGALITY_ARMS)?;
    let read_elapsed = t0.elapsed();

    // Unreachable from a built `AlignedArms`, which refuses a set with
    // no positions; total rather than unwrapped.
    let resolution = Resolution::at(arms.games()).ok_or("a walk of no games has no resolution")?;

    println!("arms (role -> file, and the checkpoint each was scored from)");
    for (role, path) in &named {
        let walk: &Walk = arms.walk(role)?;
        println!("  {:<9} {}", role, path.display());
        println!(
            "  {:<9} ckpt {} / {} conditioning / legality as input {} / holdout {} / side {}",
            "",
            walk.header.ckpt,
            walk.header.encoding,
            walk.header.legal_input,
            walk.header.holdout,
            walk.header.side
        );
    }
    println!();
    println!(
        "stream     {} position(s) from {} game(s), ctx {}, bands {:?}",
        arms.positions(),
        arms.games(),
        arms.ctx(),
        arms.walk(ARM_A)?.header.bands,
    );
    println!(
        "resampling {draws} draws of {} game(s), seed {seed}, {:.0}% percentile interval",
        arms.games(),
        CONFIDENCE * 100.0,
    );
    println!("gamma      {gamma}");
    println!(
        "floor      {:.4} nats on cost, at the {} game(s) this walk resampled \
         ({:.4} at the {} the plan is anchored on)",
        resolution.effect, resolution.games, resolution.planned_effect, resolution.planned_games
    );
    println!(
        "           it is a lower bound: the curve models neither the t-quantile at small n nor \
         drift in positions per game, and both push the same way."
    );

    // Per-arm headline figures, each with its own interval. Reported
    // for a reader; no hypothesis below subtracts two of them, since
    // two separately bootstrapped quantities carry no joint
    // distribution.
    println!();
    println!("per arm at gamma {gamma} — point estimate and interval");
    println!(
        "  {:<9} {:<34} {:<34} {:<34} legal mass",
        "arm", "ce_legal (nats)", "flip rate", "top-1 match (mean of bands)"
    );
    for (role, _) in &named {
        let cost = steerability::ce_legal(&arms, role, gamma, draws, seed)?;
        let flip = steerability::flip_rate(&arms, role, gamma, draws, seed)?;
        let top1 = steerability::top1_match(&arms, role, gamma, draws, seed)?;
        let mass = steerability::legal_mass(&arms, role, gamma, draws, seed)?;
        println!("  {role:<9} {cost:<34} {flip:<34} {top1:<34} {mass}");
        for (what, interval) in [
            ("ce_legal", &cost),
            ("flip rate", &flip),
            ("top-1 match", &top1),
            ("legal mass", &mass),
        ] {
            report_dropped_draws(&format!("{role} {what}"), interval);
        }
    }

    // The admission gate, computed rather than left to the eye: a
    // difference over the same resampled games, not two intervals to be
    // checked for overlap.
    println!();
    println!("admission at gamma {gamma} — recorded, not optimised");
    println!(
        "  mean top-1 match within {:.2} of the control's — wider than plan 02's {:.2} because \
         the same-recipe seed gap on top-1 was measured at 0.0146 and 0.0093",
        steerability::TOP1_GATE_LEGALITY,
        steerability::TOP1_GATE
    );
    println!("  {:<9} {:<34} verdict", "arm", "top-1 delta vs control");
    for (role, _) in &named {
        let gate = steerability::gate_top1_legality(&arms, role, gamma, draws, seed)?;
        println!("  {role:<9} {:<34} {}", gate.interval, gate.verdict());
        report_dropped_draws(&format!("{role} top-1 gate"), &gate.interval);
    }
    println!(
        "  Competence (ce_legal < ce_uniform) is deliberately absent: it fails on all six of \
         plan 02's arm-months, it tests calibration while being read as competence, and any \
         full-vocabulary objective fails it structurally — so it would exclude the control."
    );
    println!(
        "  Legal mass is absent for a different reason: with legality as an input the model can \
         put all its mass on legal moves trivially, so the quantity stops discriminating."
    );

    let h21 = steerability::h21(&arms, gamma, draws, seed)?;
    println!();
    println!("H21 — does handing the model the legal set add what masking the loss does not?");
    println!(
        "  D = ce_legal(A) - ce_legal(AB)        {:+.6}   ({:.6} - {:.6})",
        h21.difference, h21.ce_a, h21.ce_ab
    );
    println!(
        "  F = max of the two seed gaps          {:+.6}   (|{:.6} - {:.6}|, |{:.6} - {:.6}|)",
        h21.floor, h21.ce_ab, h21.ce_ab_b, h21.ce_a, h21.ce_a_b
    );
    println!("  confirm on D - F above zero           {}", h21.confirm);
    println!("  refute  on D + F below zero           {}", h21.refute);
    println!(
        "  resolution at {} game(s)              {:.6} nats — {}",
        h21.resolution.games,
        h21.resolution.effect,
        if h21.resolved() {
            "the effect is large enough for this walk to separate"
        } else {
            "the effect is under it, so the verdict is undetermined however clean the interval"
        }
    );
    println!(
        "  verdict on this month                 {}   ({} position(s), {} game(s))",
        h21.verdict(),
        h21.positions,
        h21.games
    );
    report_dropped_draws("H21 confirm", &h21.confirm);
    report_dropped_draws("H21 refute", &h21.refute);

    let h20 = steerability::h20(&arms, gamma, draws, seed)?;
    // Destructured rather than indexed: the cuts are a fixed-size
    // array, so this names all three and carries no index at all.
    let [first_cut, second_cut, third_cut] = h20.cuts;
    println!();
    println!("H20 — does handing over the legal set cost steerability?");
    println!(
        "  Stratified on the two compared arms' top-two margin, pooled, cut at the quartiles \
         {first_cut:.6} / {second_cut:.6} / {third_cut:.6}. The bands hold different positions \
         for the two arms, which is the design: the cuts are common, so a uniformly sharper arm \
         fills the high bands."
    );
    println!(
        "  {:<5} {:<22} {:<10} {:<10} {:<10} {:<34} {:<10} {:<10} verdict",
        "band",
        "margin",
        "flip(A)",
        "flip(AB)",
        "F_flip",
        "flip(AB) - flip(A) + F_flip",
        "level",
        "Holm"
    );
    for band in &h20.strata {
        println!(
            "  {:<5} {:<22} {:<10.6} {:<10.6} {:<10.6} {:<34} {:<10.6} {:<10.6} {}",
            band.index,
            band_range(band.low, band.high),
            band.flip_a,
            band.flip_ab,
            band.floor,
            band.interval,
            band.p_below_zero,
            band.holm_threshold,
            match (band.confirmed, band.refuted) {
                (true, _) => "confirms",
                (_, true) => "refutes",
                _ => "neither",
            }
        );
        println!(
            "        {} position(s) of A, {} of AB",
            band.positions_a, band.positions_ab
        );
        report_dropped_draws(&format!("H20 band {}", band.index), &band.interval);
    }
    println!(
        "  A refutation is read off the level after a Holm correction across the four bands, at \
         alpha {:.5} uncorrected. The confirmation is not corrected, and that is not an \
         oversight: it needs every band to clear zero, so the four are combined by intersection \
         and the joint claim is already at the level each part is tested at.",
        h20.alpha
    );
    println!(
        "  The floor above is on cost in nats and does not govern this comparison: H20's own \
         tolerance is F_flip, which widens it — a noisier replicate pair makes \"steerability \
         survives\" easier to declare, so a confirm here is the weaker of the plan's two claims."
    );
    println!(
        "  verdict on this month                 {}   (refuting bands {:?})",
        h20.verdict(),
        h20.refuting()
    );

    let h19 = steerability::h19(&arms, gamma, draws, seed)?;
    println!();
    println!("H19 — the manipulation check, which carries no verdict");
    println!(
        "  ce_legal(control) - ce_legal(A)       {:+.6}   ({:.6} - {:.6})",
        h19.difference, h19.ce_control, h19.ce_a
    );
    println!("  interval                              {}", h19.interval);
    println!(
        "  A optimises this quantity and the control does not, so a difference is expected by \
         construction and its arriving in that direction confirms nothing. handoff.md:52 \
         measured it at 0.48 nats; a figure far from that is a signal that something is wrong \
         with the run rather than a finding."
    );
    report_dropped_draws("H19", &h19.interval);

    println!();
    println!(
        "No verdict here is a decision. H21 and H20 are judged on two held-out months and only \
         when the two months agree — and for H20 that means the *same* band refuting in both, \
         which is why the bands are reported one by one rather than summarised."
    );
    println!(
        "A confirmed H21 reads \"the margin exceeds both observed same-recipe gaps\", never \
         \"clears the seed floor\": F is a max over two, and under exchangeability a third run \
         exceeds it about a third of the time."
    );
    println!();
    println!(
        "elapsed    {:.1?} total, of which {:.1?} reading the records",
        t0.elapsed(),
        read_elapsed
    );
    Ok(())
}

/// One band's margin range, with the open ends spelled rather than
/// filled in with a bound the band does not have.
fn band_range(low: Option<f64>, high: Option<f64>) -> String {
    let side = |edge: Option<f64>| match edge {
        Some(value) => format!("{value:.6}"),
        None => "..".to_string(),
    };
    format!("[{}, {})", side(low), side(high))
}

/// Say so when draws were thrown away.
///
/// A draw whose statistic is undefined — a resample that caught no
/// position of one arm, a zero denominator — is left out of the
/// percentiles, and leaving draws out biases the interval. Silence
/// about it would let a figure resting on half its draws read exactly
/// like one resting on all of them.
fn report_dropped_draws(label: &str, interval: &Interval) {
    // `label` is a `&str` rather than `&'static str` because the rows
    // name the arm or the band they belong to.
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
