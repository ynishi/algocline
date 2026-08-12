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

use algocline_nn::chess::merge::{merge_slots, merge_task_arithmetic};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_merge: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: chess_merge [--base <base.safetensors>] <out.safetensors> \
                     <src1> <src2> [src3...]\n\
                     with --base the bodies combine as base + sum(source - base) \
                     (task arithmetic); without it, as their mean";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut base: Option<PathBuf> = None;
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = Some(args.next().ok_or(USAGE)?.into()),
            _ => positional.push(arg.into()),
        }
    }
    if positional.len() < 3 {
        return Err(USAGE.into());
    }
    let out = positional[0].clone();
    let sources: Vec<PathBuf> = positional[1..].to_vec();
    let refs: Vec<&Path> = sources.iter().map(PathBuf::as_path).collect();

    let shape = match &base {
        Some(base) => merge_task_arithmetic(base, &refs, &out)?,
        None => merge_slots(&refs, &out)?,
    };
    println!(
        "merged     {} source(s) {} -> {}",
        refs.len(),
        match &base {
            Some(b) => format!("as base + sum(differences) from {}", b.display()),
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
