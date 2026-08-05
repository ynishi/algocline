//! Verify the PGN reader against a real corpus, and size what it feeds.
//!
//! Replays every game in a PGN file from the start position, resolving
//! each SAN token against the legal move set. A correct reader finds
//! exactly one legal move per token, so any "no match" or "ambiguous"
//! result is a bug in the reader — this is the self-check described in
//! `src/pgn.rs`.
//!
//! The same pass reports what a tokenizer would face: how many distinct
//! move tokens the corpus contains in SAN and in UCI, how concentrated
//! they are, and how long the games run. Those three numbers decide the
//! vocabulary size and the context length, so they are measured on the
//! corpus rather than assumed.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example pgn_san_check -- <path.pgn> [max_failures_to_print]
//! ```
//!
//! The input is a plain (already decompressed) PGN file. Lichess ships
//! monthly zstd archives as a single frame, so a prefix slice is the
//! practical way to get a working sample:
//!
//! ```text
//! curl -s -r 0-8000000 https://database.lichess.org/standard/<file>.pgn.zst -o head.zst
//! zstd -dc head.zst | head -n 480000 > sample.pgn
//! ```

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::pgn::{game_to_uci, san_tokens, PgnReader};

/// Report how large a vocabulary is and how much of the corpus its
/// most common entries cover.
///
/// Coverage is what says whether a vocabulary can be truncated: a
/// long tail that carries a negligible share of plies can be dropped,
/// while a flat distribution cannot.
fn report_vocab(label: &str, counts: &HashMap<String, usize>) {
    let total: usize = counts.values().sum();
    if total == 0 {
        println!("{label:<10} (no tokens)");
        return;
    }
    let mut sorted: Vec<usize> = counts.values().copied().collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let once = sorted.iter().filter(|c| **c == 1).count();
    println!("{label:<10} {} distinct", counts.len());
    let mut acc = 0usize;
    let mut next_mark = 0usize;
    let marks = [64usize, 128, 256, 512, 1024, 2048, 4096];
    for (i, c) in sorted.iter().enumerate() {
        acc += c;
        while next_mark < marks.len() && i + 1 == marks[next_mark] {
            println!(
                "           top {:>5}: {:>7.3}% of plies",
                marks[next_mark],
                100.0 * acc as f64 / total as f64
            );
            next_mark += 1;
        }
    }
    println!(
        "           seen once: {once} ({:.1}% of vocab, {:.4}% of plies)",
        100.0 * once as f64 / counts.len() as f64,
        100.0 * once as f64 / total as f64
    );
}

/// Print the ply-length quantiles that bound a usable context window.
fn report_lengths(lengths: &mut [usize]) {
    if lengths.is_empty() {
        return;
    }
    lengths.sort_unstable();
    let at = |q: f64| lengths[((lengths.len() - 1) as f64 * q) as usize];
    let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    println!(
        "ply/game   mean {mean:.1}  p50 {}  p90 {}  p95 {}  p99 {}  max {}",
        at(0.50),
        at(0.90),
        at(0.95),
        at(0.99),
        lengths[lengths.len() - 1]
    );
    for ctx in [64usize, 96, 128, 192, 256] {
        let fits = lengths.iter().filter(|l| **l <= ctx).count();
        println!(
            "           ctx {ctx:>4}: {:>6.2}% of games fit whole",
            100.0 * fits as f64 / lengths.len() as f64
        );
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: pgn_san_check <path.pgn> [max_failures_to_print]");
        return ExitCode::FAILURE;
    };
    let max_print: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut reader = PgnReader::new(BufReader::new(file));

    let start = Instant::now();
    let mut games = 0usize;
    let mut plies = 0usize;
    let mut failed = 0usize;
    let mut printed = 0usize;
    let mut san_counts: HashMap<String, usize> = HashMap::new();
    let mut uci_counts: HashMap<String, usize> = HashMap::new();
    let mut lengths: Vec<usize> = Vec::new();

    loop {
        let game = match reader.next_game() {
            Ok(Some(g)) => g,
            Ok(None) => break,
            Err(e) => {
                eprintln!("pgn read failed after {games} games: {e}");
                return ExitCode::FAILURE;
            }
        };
        games += 1;
        for token in san_tokens(&game.movetext) {
            *san_counts.entry(token).or_insert(0) += 1;
        }
        match game_to_uci(&game.movetext) {
            Ok(moves) => {
                plies += moves.len();
                lengths.push(moves.len());
                for mv in moves {
                    *uci_counts.entry(mv).or_insert(0) += 1;
                }
            }
            Err(e) => {
                failed += 1;
                if printed < max_print {
                    printed += 1;
                    let site = game.tag("Site").unwrap_or("(no Site tag)");
                    eprintln!("FAIL game {games} {site}: {e}");
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let rate = if elapsed.as_secs_f64() > 0.0 {
        plies as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!("games      {games}");
    println!("plies      {plies}");
    println!("failed     {failed} games");
    println!(
        "elapsed    {:.2}s  ({:.0}k ply/s)",
        elapsed.as_secs_f64(),
        rate / 1000.0
    );
    report_lengths(&mut lengths);
    report_vocab("SAN", &san_counts);
    report_vocab("UCI", &uci_counts);

    if failed == 0 {
        println!("VERDICT    OK — every SAN token resolved to exactly one legal move");
        ExitCode::SUCCESS
    } else {
        println!("VERDICT    FAILED — {failed} games did not replay");
        ExitCode::FAILURE
    }
}
