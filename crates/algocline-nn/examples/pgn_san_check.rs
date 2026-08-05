//! Verify the PGN reader against a real corpus.
//!
//! Replays every game in a PGN file from the start position, resolving
//! each SAN token against the legal move set. A correct reader finds
//! exactly one legal move per token, so any "no match" or "ambiguous"
//! result is a bug in the reader — this is the self-check described in
//! `src/pgn.rs`.
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

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::pgn::{game_to_uci, PgnReader};

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
    let mut longest = 0usize;

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
        match game_to_uci(&game.movetext) {
            Ok(moves) => {
                plies += moves.len();
                longest = longest.max(moves.len());
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
    println!("longest    {longest} plies");
    println!("failed     {failed} games");
    println!(
        "elapsed    {:.2}s  ({:.0}k ply/s)",
        elapsed.as_secs_f64(),
        rate / 1000.0
    );

    if failed == 0 {
        println!("VERDICT    OK — every SAN token resolved to exactly one legal move");
        ExitCode::SUCCESS
    } else {
        println!("VERDICT    FAILED — {failed} games did not replay");
        ExitCode::FAILURE
    }
}
