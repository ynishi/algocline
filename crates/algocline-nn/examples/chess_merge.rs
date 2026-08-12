//! Assemble a multi-slot checkpoint out of single-slot ones.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_merge -- <out.safetensors> <src1> <src2> [src3...]
//! ```
//!
//! Sources are single-slot per-position checkpoints, in the slot order
//! the merged model should carry: the first source's bands become
//! group 0. The body tensors are averaged, the conditioning tables are
//! stacked, and the merged sidecar is written beside the weights — so
//! what comes out is indistinguishable from a trained multi-slot
//! checkpoint to every reader downstream, which is what plan 10 needs
//! in order to score it with the judge that scored the trained arm.
//!
//! What this does not do is check that the sources were trained on the
//! same corpus with the same recipe. Nothing in a checkpoint records
//! it; the plan's provenance is the operator's.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use algocline_nn::chess::merge::{merge_slots, merge_task_arithmetic_scaled};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_merge: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: chess_merge [--base <base.safetensors>] [--scale <s>] \
                     <out.safetensors> <src1> <src2> [src3...]\n\
                     with --base the bodies combine as base + s*sum(source - base) \
                     (task arithmetic, s defaults to 1); without it, as their mean.\n\
                     --scale requires --base: the mean has no differences to scale";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut base: Option<PathBuf> = None;
    let mut scale: Option<f64> = None;
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = Some(args.next().ok_or(USAGE)?.into()),
            "--scale" => {
                let raw = args.next().ok_or(USAGE)?;
                scale = Some(
                    raw.parse()
                        .map_err(|_| format!("--scale {raw:?} is not a number"))?,
                );
            }
            _ => positional.push(arg.into()),
        }
    }
    // Refused rather than ignored: a run that passed --scale and got
    // the mean would produce a file named for a coefficient that had
    // no part in making it.
    if scale.is_some() && base.is_none() {
        return Err("--scale requires --base; the mean has no differences to scale".into());
    }
    if positional.len() < 3 {
        return Err(USAGE.into());
    }
    let out = positional[0].clone();
    let sources: Vec<PathBuf> = positional[1..].to_vec();
    let refs: Vec<&Path> = sources.iter().map(PathBuf::as_path).collect();

    let scale = scale.unwrap_or(1.0);
    let shape = match &base {
        Some(base) => merge_task_arithmetic_scaled(base, &refs, scale, &out)?,
        None => merge_slots(&refs, &out)?,
    };
    println!(
        "merged     {} source(s) {} -> {}",
        refs.len(),
        match &base {
            Some(b) => format!("as base + {scale}*sum(differences) from {}", b.display()),
            None => "as their mean".to_string(),
        },
        out.display()
    );
    for (slot, size) in shape.cond_groups.iter().enumerate() {
        let start: usize = shape.cond_groups[..slot].iter().sum();
        let tokens: Vec<&str> = shape.bands[start..start + size]
            .iter()
            .map(|b| b.token.as_str())
            .collect();
        println!("  slot {slot}: {tokens:?}");
    }
    println!(
        "shape      layers={} heads={} dim={} ctx={} vocab={} groups={:?}",
        shape.layers, shape.heads, shape.dim, shape.ctx, shape.vocab, shape.cond_groups
    );
    println!(
        "  the sidecar is written beside the weights; score this with the same judge the \
         trained arm goes through"
    );
    Ok(())
}
