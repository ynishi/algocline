//! Read the survival experiment's four arms together and ask whether the
//! conditioning advantage outlives the loss mask.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_survival -- \
//!     <A.jsonl> <A-b.jsonl> <P.jsonl> <P-b.jsonl> \
//!     [gamma] [seed] [draws]
//! ```
//!
//! The four files are the arms **in role order**: the two prefix runs
//! (plan 03's `A` and `A-b`, reused) and the two per-position runs.
//!
//! # What plan 04 is for
//!
//! Plan 02 measured per-position against prefix at flip 49.0% / 52.7%
//! versus 15.2% / 17.3% — **with the loss taken over the whole
//! vocabulary**. On that objective the model ranks legal moves at about
//! the uniform level (`ce_legal` 3.364 / 3.480 against a legal-uniform
//! level near 3.23), and a nearly flat ranking is cheap to reorder. Plan
//! 03 took the loss over the legal moves and the flip rate fell to
//! 0.089 / 0.088 — but every arm there was prefix-conditioned, so that
//! fall is not a measurement of the per-position arm.
//!
//! Nothing forced that. [`algocline_nn::arch::Gpt2Custom::validate`]
//! refuses a model carrying both the conditioning table and the legality
//! table; the **loss** mask carries no table and was never in that
//! constraint. This arm set is the combination neither plan baked.
//!
//! # What is refused before anything prints
//!
//! Here the conditioning is what separates the arms, which is the mirror
//! of `chess_legality`. Exchange `A` and `P` on this command line and
//! `M = flip(P) - flip(A)` changes sign while every printed figure stays
//! well-formed, so the role check runs on the arms as read, ahead of the
//! first line of output. The legality axis is checked too, against
//! `false` for all four: plan 04 holds legality-as-input out of the
//! experiment, so a legality checkpoint in any slot is an unregistered
//! third condition rather than a mislabelled arm.
//!
//! **What is not checked**, because nothing in a record could: which of
//! `A` and `A-b` is which, or of `P` and `P-b`. Exchanging within a pair
//! leaves the floor alone and makes the margin the other replicate's.
//!
//! # Why `undetermined` is the outcome to read
//!
//! `F` is a positive distance between two real checkpoints, so the
//! refute criterion `M + F < 0` needs the per-position arm to flip
//! *less* than the prefix arm by more than both seed gaps — not the
//! shape "the axis died" would take. Plan 04 §3 registers, before any
//! arm was baked, that the axis failing to survive shows up as
//! **undetermined**. That is written down so it cannot be presented
//! afterwards as a near miss.
//!
//! # H23 is read only when H22 confirms
//!
//! Plan 04 registers the depth reach as secondary and gates it on the
//! primary, so this prints it only in that case and says so otherwise.
//! Printing it either way would make "registered but not read" look the
//! same as "read and unremarkable".
//!
//! Its floor is plan 04's rather than plan 02's. `h15` draws none,
//! because only one side of that comparison was replicated; both sides
//! are replicated here. The plan's own text asked for "the same way of
//! drawing the floor as plan 02's H15", which names no rule — plan 04
//! §2 records the disambiguation, written after H22's verdict and
//! before H23 was computed, on the conservative side.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::records::{AlignedArms, Walk};
use algocline_nn::chess::steerability::{
    self, ARM_A, ARM_A_B, ARM_P, ARM_P_B, DRAWS, SURVIVAL_ARMS,
};
use algocline_nn::metric::bootstrap::{Interval, CONFIDENCE};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_survival: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Default seed for the resampling.
///
/// Named rather than drawn from the clock, as `chess_legality`'s is.
const DEFAULT_SEED: u64 = 0x0806_2026;

/// The cost every arm here has to be under for the mask to have been
/// applied at all.
///
/// Plan 04's H24, which carries no verdict. The mask-free control
/// measured 3.364 / 3.480 nats and the masked arms 2.717 / 2.773, so
/// anything near the former means the switch did not take and the run
/// is a defect rather than a result. Deliberately loose: it separates
/// "the mask ran" from "the mask did not", and is not a quality bar.
const MASK_APPLIED_CEILING: f64 = 3.0;

const USAGE: &str =
    "usage: chess_survival <A.jsonl> <A-b.jsonl> <P.jsonl> <P-b.jsonl> [gamma] [seed] [draws]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut next_path =
        || -> Result<PathBuf, &'static str> { Ok(args.next().ok_or(USAGE)?.into()) };
    let prefix = next_path()?;
    let prefix_b = next_path()?;
    let perpos = next_path()?;
    let perpos_b = next_path()?;
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
        (ARM_A.to_string(), prefix),
        (ARM_A_B.to_string(), prefix_b),
        (ARM_P.to_string(), perpos),
        (ARM_P_B.to_string(), perpos_b),
    ];
    let arms = AlignedArms::read(&named)?;
    // Before anything prints. A swap of `A` and `P` reverses the sign of
    // the margin while leaving every figure below well-formed, and a
    // table that reached the reader first would be one nobody had reason
    // to doubt — plan 02's failure exactly.
    steerability::check_survival_roles(&arms)?;
    steerability::check_scoreable(&arms, &SURVIVAL_ARMS)?;
    let read_elapsed = t0.elapsed();

    println!("arms (role -> file, and the checkpoint each was scored from)");
    for (role, path) in &named {
        let walk: &Walk = arms.walk(role)?;
        println!("  {:<5} {}", role, path.display());
        println!(
            "  {:<5} ckpt {} / {} conditioning / legality as input {} / holdout {} / side {}",
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

    println!();
    println!("per arm at gamma {gamma} — point estimate and interval");
    println!(
        "  {:<5} {:<34} {:<34} top-1 match (mean of bands)",
        "arm", "flip rate", "ce_legal (nats)"
    );
    for (role, _) in &named {
        let flip = steerability::flip_rate(&arms, role, gamma, draws, seed)?;
        let cost = steerability::ce_legal(&arms, role, gamma, draws, seed)?;
        let top1 = steerability::top1_match(&arms, role, gamma, draws, seed)?;
        println!("  {role:<5} {flip:<34} {cost:<34} {top1}");
        for (what, interval) in [
            ("flip rate", &flip),
            ("ce_legal", &cost),
            ("top-1 match", &top1),
        ] {
            report_dropped_draws(&format!("{role} {what}"), interval);
        }
    }

    // Admission, before the hypothesis. The reference is `A` — the arm
    // being compared — and not a control on a different objective,
    // which is the correction plan 03's gate needed: there the tolerance
    // was sized from the shuffle seed while the treatment's intended
    // effect was twice it, so every treatment arm was excluded before a
    // single record was read.
    println!();
    println!("admission at gamma {gamma} — recorded, not optimised");
    println!(
        "  mean top-1 match within {} of {ARM_A}'s. The reference is the compared arm: \
         per-position conditioning is not designed to move top-1 (plan 02 measured \
         +0.006 / +0.004, intervals spanning zero), and the same-recipe seed gap on this \
         pair is 0.0123 / 0.0112 — so passing and failing are both reachable.",
        steerability::TOP1_GATE_SURVIVAL
    );
    println!("  {:<5} {:<34} verdict", "arm", "top-1 delta vs A");
    let mut admitted = Vec::new();
    for (role, _) in &named {
        let gate = steerability::gate_top1_survival(&arms, role, gamma, draws, seed)?;
        println!("  {role:<5} {:<34} {}", gate.interval, gate.verdict());
        report_dropped_draws(&format!("{role} top-1 gate"), &gate.interval);
        if gate.passes() {
            admitted.push(role.as_str());
        }
    }

    // The manipulation check, printed before the verdict it does not
    // carry: a run where the mask did not take is a defect, and a
    // reader should see that before reading a margin off it.
    println!();
    println!("H24 — did the mask apply at all? (no verdict)");
    let mut mask_ok = true;
    for (role, _) in &named {
        let cost = steerability::ce_legal(&arms, role, gamma, draws, seed)?;
        let applied = cost.point < MASK_APPLIED_CEILING;
        mask_ok &= applied;
        println!(
            "  {role:<5} ce_legal {:.6} {} {MASK_APPLIED_CEILING}   {}",
            cost.point,
            if applied { "<" } else { ">=" },
            if applied {
                "mask applied"
            } else {
                "MASK DID NOT APPLY"
            }
        );
    }
    if !mask_ok {
        println!(
            "  the mask-free control measured 3.364 / 3.480 nats. An arm up there did not train \
             on the masked objective, so this run is a defect and not a result."
        );
    }

    println!();
    println!("H22 — does the conditioning advantage survive the loss mask?");
    let h22 = steerability::h22(&arms, gamma, draws, seed)?;
    println!(
        "  M = flip(P) - flip(A)                 {:+.6}   ({:.6} - {:.6})",
        h22.margin, h22.flip_p, h22.flip_a
    );
    println!(
        "  F = max of the two seed gaps          {:+.6}   (|{:.6} - {:.6}|, |{:.6} - {:.6}|)",
        h22.floor, h22.flip_p, h22.flip_p_b, h22.flip_a, h22.flip_a_b
    );
    println!("  confirm on M - F above zero           {}", h22.confirm);
    println!("  refute  on M + F below zero           {}", h22.refute);
    report_dropped_draws("H22 confirm", &h22.confirm);
    report_dropped_draws("H22 refute", &h22.refute);
    println!(
        "  verdict on this month                 {}   ({} position(s), {} game(s))",
        h22.verdict(),
        h22.positions,
        h22.games
    );
    println!(
        "  F is a distance between checkpoints, so walking more of the holdout pins it down \
         without shrinking it, and a third seed would widen it. The floor here is not a \
         sample-size problem and more positions do not help."
    );
    println!(
        "  A refute needs P to flip LESS than A by more than both seed gaps, which is not the \
         shape the axis dying would take. Plan 04 §3 registers, before any arm was baked, that \
         the axis failing to survive reads as `undetermined`."
    );

    // Read only when H22 confirms, which the plan fixed before any arm
    // was baked. Printing it either way would make "registered but not
    // read" indistinguishable from "read and unremarkable".
    if h22.confirmed() {
        println!();
        println!("H23 — does the advantage still reach the deep positions?");
        let h23 = steerability::h23(&arms, gamma, draws, seed)?;
        println!(
            "  buckets                               ply {}-{} ({} position(s), {} game(s)) over \
             ply {}-{} ({} position(s), {} game(s))",
            h23.deep.low,
            h23.deep.high,
            h23.deep.positions,
            h23.deep.games,
            h23.shallow.low,
            h23.shallow.high,
            h23.shallow.positions,
            h23.shallow.games
        );
        println!(
            "  R = ratio(P) - ratio(A)               {:+.6}   ({:.6} - {:.6})",
            h23.margin, h23.ratio_p, h23.ratio_a
        );
        println!(
            "  F_R = max of the two seed gaps        {:+.6}   (|{:.6} - {:.6}|, |{:.6} - {:.6}|)",
            h23.floor, h23.ratio_p, h23.ratio_p_b, h23.ratio_a, h23.ratio_a_b
        );
        println!("  confirm on R - F_R above zero         {}", h23.confirm);
        println!("  refute  on R + F_R below zero         {}", h23.refute);
        report_dropped_draws("H23 confirm", &h23.confirm);
        report_dropped_draws("H23 refute", &h23.refute);
        println!("  verdict on this month                 {}", h23.verdict());
        println!(
            "  The floor is plan 04's, not plan 02's: `h15` draws none because only one side of \
             that comparison was replicated. The plan's text did not name a rule, and §2 records \
             the disambiguation — written after H22's verdict and before this was computed, on \
             the conservative side."
        );
    } else {
        println!();
        println!(
            "H23 is registered and is read only if H22 confirms. It did not, so it was not \
             computed."
        );
    }

    if !admitted.contains(&ARM_P) {
        println!();
        println!(
            "NOTE: {ARM_P} did not pass admission, so the verdict above is descriptive and not \
             adopted — the same treatment plan 03's arms received, and for the same reason."
        );
    }

    println!();
    println!(
        "Judged on two held-out months and only when the two agree. H23 (depth reach) is \
         registered and read only when H22 confirms."
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
/// A draw whose statistic is undefined is left out of the percentiles,
/// and leaving draws out biases the interval. Silence about it would let
/// a figure resting on half its draws read like one resting on all.
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
