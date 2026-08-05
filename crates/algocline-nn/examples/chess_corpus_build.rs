//! Build a rating-banded chess corpus from a PGN file and report what
//! the filter did with the input.
//!
//! This is the end of the ingestion path: a public archive goes in, the
//! token rows a model trains on come out, and every game read is
//! accounted for as a row or as a named rejection.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_corpus_build -- <path.pgn> [min_elo] [max_elo] [max_rows]
//! ```
//!
//! Defaults to the 1600-1799 band and 20,000 rows. The band is applied
//! to both players, only games decided on the board are kept, bullet
//! time controls are excluded, and rows longer than 128 tokens are
//! dropped rather than cut.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;
use std::time::Instant;

use algocline_nn::chess::corpus::{build_rows, ConditionBand, ConditionSpec, CorpusOptions};
use algocline_nn::chess::filter::GameFilter;
use algocline_nn::chess::pgn::PgnReader;
use algocline_nn::chess::vocab::MoveVocab;

/// Context window the rows are built against.
///
/// Measured on a 2026-06 slice: 128 tokens hold 96.2% of games whole.
const MAX_LEN: usize = 128;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: chess_corpus_build <path.pgn> [min_elo] [max_elo] [max_rows]");
        return ExitCode::FAILURE;
    };
    let min_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1600);
    let max_elo: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1799);
    let max_rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20_000);

    let band_token = format!("<elo:{min_elo}-{max_elo}>");
    let vocab = match MoveVocab::new(std::slice::from_ref(&band_token)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot build vocabulary: {e}");
            return ExitCode::FAILURE;
        }
    };

    let opts = CorpusOptions {
        filter: GameFilter::accept_all()
            .with_rating_band(min_elo, max_elo)
            .decided_on_the_board()
            .with_min_base_seconds(180)
            .with_ply_bounds(10, None),
        max_rows,
        max_len: Some(MAX_LEN),
        condition: Some(ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![ConditionBand {
                min: min_elo,
                max: max_elo,
                token: band_token.clone(),
            }],
        }),
        ..Default::default()
    };

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut reader = PgnReader::new(BufReader::new(file));

    let start = Instant::now();
    let (rows, stats) = match build_rows(&mut reader, &vocab, &opts) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("corpus build failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = start.elapsed();

    println!("band       {min_elo}-{max_elo}  (token {band_token})");
    println!(
        "vocab      {} ids, model size {}",
        vocab.len(),
        vocab.model_vocab_size()
    );
    println!("elapsed    {:.2}s", elapsed.as_secs_f64());
    println!();
    println!("games read           {}", stats.games_read);
    println!("  rejected by tags   {}", stats.rejected_by_tags);
    println!("  rejected by length {}", stats.rejected_by_length);
    println!("  outside every band {}", stats.rejected_by_condition);
    println!("  dropped overlong   {}", stats.dropped_overlong);
    println!("  replay failures    {}", stats.replay_failures);
    println!("  rows kept          {}", stats.rows);
    if let Some(first) = &stats.first_replay_failure {
        println!("  first failure      {first}");
    }
    println!();
    if stats.games_read > 0 {
        println!(
            "yield      {:.2}% of games read",
            100.0 * stats.rows as f64 / stats.games_read as f64
        );
    }
    println!("tokens     {}", stats.tokens);
    if stats.rows > 0 {
        let mut lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        lens.sort_unstable();
        let mean = stats.tokens as f64 / stats.rows as f64;
        let at = |q: f64| lens[((lens.len() - 1) as f64 * q) as usize];
        println!(
            "row len    mean {mean:.1}  p50 {}  p90 {}  max {}",
            at(0.50),
            at(0.90),
            lens[lens.len() - 1]
        );
        let head: Vec<&str> = rows[0]
            .iter()
            .take(8)
            .filter_map(|id| vocab.token_of(*id))
            .collect();
        println!("row[0]     {}", head.join(" "));
    }

    // Every game read must land in exactly one bucket. A mismatch here
    // means a path through the builder that neither counts nor keeps.
    let accounted = stats.rows
        + stats.rejected_by_tags
        + stats.rejected_by_length
        + stats.rejected_by_condition
        + stats.dropped_overlong
        + stats.replay_failures;
    if accounted != stats.games_read {
        println!(
            "VERDICT    FAILED — {accounted} games accounted for, {} read",
            stats.games_read
        );
        return ExitCode::FAILURE;
    }
    println!("VERDICT    OK — every game read is accounted for");
    ExitCode::SUCCESS
}
