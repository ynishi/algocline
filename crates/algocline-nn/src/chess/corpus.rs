//! Building token rows from a PGN stream.
//!
//! One game becomes one row: `BOS`, an optional condition token, then
//! the game's moves as vocabulary ids. Rows go to
//! [`crate::train::data::TokenizedDataset`], which pads and batches
//! them.
//!
//! # The order of the stages
//!
//! ```text
//! read game -> test tags -> replay moves -> test length -> encode
//! ```
//!
//! Tags are tested before the replay because the replay is the
//! expensive stage and a narrow filter rejects most of an archive.
//! On a 2026-06 slice, "both players in a 200-point band, decided on
//! the board, blitz or slower, at least 10 plies" keeps 4-8% of games,
//! so replaying first would spend an order of magnitude more work than
//! the corpus needs.
//!
//! # What is counted rather than dropped
//!
//! Games that fail to replay are counted in [`CorpusStats`] and the
//! first failure is kept, rather than being skipped in silence. A
//! corpus that quietly lost 3% of its input reads exactly like one
//! that did not, and the difference only shows up as a model that
//! trained on less than it was told to.

use std::io::BufRead;

use thiserror::Error;

use crate::chess::filter::GameFilter;
use crate::chess::pgn::{game_to_uci, PgnError, PgnReader};
use crate::chess::vocab::{MoveVocab, BOS};

/// What to do with a game longer than the row limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overlong {
    /// Drop the game.
    ///
    /// The default, because a truncated game ends in a position no one
    /// resigned or was mated in, and the model reads that ending as a
    /// place where play stops.
    #[default]
    Drop,
    /// Keep the opening and cut the tail.
    Truncate,
}

/// A range of an integer tag mapped to a condition token.
#[derive(Debug, Clone)]
pub struct ConditionBand {
    /// Inclusive lower bound.
    pub min: i64,
    /// Inclusive upper bound.
    pub max: i64,
    /// Token emitted for values in the range. Must exist in the
    /// vocabulary's condition block.
    pub token: String,
}

/// How a game's condition token is derived from its tags.
///
/// A conditional model reads the band it should play as from the front
/// of the sequence, so the band has to be written into the row. This
/// is the mapping from a tag value to that token.
#[derive(Debug, Clone)]
pub struct ConditionSpec {
    /// Tag to read, e.g. `WhiteElo`.
    pub key: String,
    /// Bands, tested in order.
    pub bands: Vec<ConditionBand>,
}

impl ConditionSpec {
    /// Find the token for a game, or `None` when the tag is absent,
    /// unparsable, or outside every band.
    fn token_for(&self, game: &crate::chess::pgn::PgnGame) -> Option<&str> {
        let value = game.tag_i64(&self.key)?;
        self.bands
            .iter()
            .find(|b| value >= b.min && value <= b.max)
            .map(|b| b.token.as_str())
    }
}

/// How a corpus is built.
#[derive(Debug, Clone)]
pub struct CorpusOptions {
    /// Which games to keep.
    pub filter: GameFilter,
    /// Stop once this many rows exist.
    ///
    /// Reading stops here rather than at the end of the stream, which
    /// is what makes a monthly archive usable: the first slice that
    /// yields enough rows is read and the rest is never fetched.
    pub max_rows: usize,
    /// Longest row in tokens, `BOS` and the condition token included,
    /// or `None` for unbounded.
    pub max_len: Option<usize>,
    /// What to do when a row exceeds `max_len`.
    pub overlong: Overlong,
    /// Condition token derivation, or `None` for an unconditional
    /// corpus.
    pub condition: Option<ConditionSpec>,
}

impl Default for CorpusOptions {
    fn default() -> Self {
        Self {
            filter: GameFilter::accept_all(),
            max_rows: 20_000,
            max_len: None,
            overlong: Overlong::default(),
            condition: None,
        }
    }
}

/// What a corpus build did with its input.
///
/// Every game read lands in exactly one of the rejection counters or
/// in `rows`, so the numbers account for the whole stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CorpusStats {
    /// Games read off the stream.
    pub games_read: usize,
    /// Rejected by a tag predicate.
    pub rejected_by_tags: usize,
    /// Rejected by the ply bounds.
    pub rejected_by_length: usize,
    /// Rejected because no condition band covered the game.
    pub rejected_by_condition: usize,
    /// Dropped because the row exceeded `max_len`.
    pub dropped_overlong: usize,
    /// Kept but cut down to `max_len`.
    pub truncated: usize,
    /// Games whose moves could not be replayed.
    ///
    /// Non-zero means the reader met something it could not parse.
    /// `first_replay_failure` carries the first one.
    pub replay_failures: usize,
    /// The first replay failure, for diagnosis.
    pub first_replay_failure: Option<String>,
    /// Rows produced.
    pub rows: usize,
    /// Token ids across all rows.
    pub tokens: usize,
}

/// Failure while building a corpus.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// The PGN stream could not be read.
    #[error("corpus: {0}")]
    Pgn(#[from] PgnError),
    /// A token produced by the reader has no id in the vocabulary.
    ///
    /// For a move this means the vocabulary's enumeration has a hole;
    /// for a condition token it means the vocabulary was built without
    /// the bands the options ask for.
    #[error("corpus: {kind} token {token:?} is not in the vocabulary")]
    UnknownToken {
        /// `"move"` or `"condition"`.
        kind: &'static str,
        /// The token with no id.
        token: String,
    },
}

/// Read games and encode the ones that pass the filter.
///
/// Returns the rows and an account of what happened to everything
/// else.
pub fn build_rows<R: BufRead>(
    reader: &mut PgnReader<R>,
    vocab: &MoveVocab,
    opts: &CorpusOptions,
) -> Result<(Vec<Vec<u32>>, CorpusStats), CorpusError> {
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut stats = CorpusStats::default();

    while rows.len() < opts.max_rows {
        let Some(game) = reader.next_game()? else {
            break;
        };
        stats.games_read += 1;

        if !opts.filter.accepts_tags(&game) {
            stats.rejected_by_tags += 1;
            continue;
        }

        // Resolve the condition before replaying: a game with no band
        // cannot enter a conditional corpus, and finding that out
        // costs one tag lookup instead of a full replay.
        let condition_id = match &opts.condition {
            Some(spec) => match spec.token_for(&game) {
                Some(token) => {
                    let id = vocab
                        .id_of(token)
                        .ok_or_else(|| CorpusError::UnknownToken {
                            kind: "condition",
                            token: token.to_string(),
                        })?;
                    Some(id)
                }
                None => {
                    stats.rejected_by_condition += 1;
                    continue;
                }
            },
            None => None,
        };

        let moves = match game_to_uci(&game.movetext) {
            Ok(moves) => moves,
            Err(e) => {
                stats.replay_failures += 1;
                if stats.first_replay_failure.is_none() {
                    let site = game.tag("Site").unwrap_or("(no Site tag)");
                    stats.first_replay_failure = Some(format!("{site}: {e}"));
                }
                continue;
            }
        };

        if !opts.filter.accepts_length(moves.len()) {
            stats.rejected_by_length += 1;
            continue;
        }

        let mut row = Vec::with_capacity(moves.len() + 2);
        row.push(BOS);
        if let Some(id) = condition_id {
            row.push(id);
        }
        for mv in &moves {
            let id = vocab.id_of(mv).ok_or_else(|| CorpusError::UnknownToken {
                kind: "move",
                token: mv.clone(),
            })?;
            row.push(id);
        }

        if let Some(max_len) = opts.max_len {
            if row.len() > max_len {
                match opts.overlong {
                    Overlong::Drop => {
                        stats.dropped_overlong += 1;
                        continue;
                    }
                    Overlong::Truncate => {
                        row.truncate(max_len);
                        stats.truncated += 1;
                    }
                }
            }
        }

        stats.tokens += row.len();
        rows.push(row);
    }

    stats.rows = rows.len();
    Ok((rows, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::filter::GameFilter;
    use std::io::Cursor;

    fn pgn(games: &[(&str, &str, &str)]) -> String {
        let mut out = String::new();
        for (white_elo, termination, moves) in games {
            out.push_str(&format!("[WhiteElo \"{white_elo}\"]\n"));
            out.push_str(&format!("[BlackElo \"{white_elo}\"]\n"));
            out.push_str(&format!("[Termination \"{termination}\"]\n"));
            out.push_str(&format!("\n{moves}\n\n"));
        }
        out
    }

    fn vocab_with_bands() -> MoveVocab {
        MoveVocab::new(&["<elo:low>".to_string(), "<elo:high>".to_string()]).unwrap()
    }

    #[test]
    fn a_row_is_bos_then_the_moves() {
        let text = pgn(&[("1500", "Normal", "1. e4 e5 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let (rows, stats) = build_rows(&mut reader, &vocab, &CorpusOptions::default()).unwrap();
        assert_eq!(stats.rows, 1);
        assert_eq!(rows[0].len(), 3);
        assert_eq!(rows[0][0], BOS);
        assert_eq!(rows[0][1], vocab.id_of("e2e4").unwrap());
        assert_eq!(rows[0][2], vocab.id_of("e7e5").unwrap());
        assert_eq!(stats.tokens, 3);
    }

    #[test]
    fn tags_are_tested_before_the_replay() {
        // The second game's movetext is nonsense; it must never be
        // replayed, because its tags are rejected first.
        let text = pgn(&[
            ("1500", "Normal", "1. e4 e5 1-0"),
            ("1500", "Time forfeit", "1. Qz9 1-0"),
        ]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let opts = CorpusOptions {
            filter: GameFilter::accept_all().decided_on_the_board(),
            ..Default::default()
        };
        let (rows, stats) = build_rows(&mut reader, &vocab, &opts).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(stats.rejected_by_tags, 1);
        assert_eq!(stats.replay_failures, 0);
    }

    #[test]
    fn a_replay_failure_is_counted_and_kept() {
        let text = pgn(&[("1500", "Normal", "1. Qh6 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let (rows, stats) = build_rows(&mut reader, &vocab, &CorpusOptions::default()).unwrap();
        assert!(rows.is_empty());
        assert_eq!(stats.replay_failures, 1);
        assert!(stats.first_replay_failure.is_some());
    }

    #[test]
    fn the_condition_token_sits_right_after_bos() {
        let text = pgn(&[("1500", "Normal", "1. e4 e5 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = vocab_with_bands();
        let opts = CorpusOptions {
            condition: Some(ConditionSpec {
                key: "WhiteElo".to_string(),
                bands: vec![
                    ConditionBand {
                        min: 0,
                        max: 1599,
                        token: "<elo:low>".to_string(),
                    },
                    ConditionBand {
                        min: 1600,
                        max: 4000,
                        token: "<elo:high>".to_string(),
                    },
                ],
            }),
            ..Default::default()
        };
        let (rows, _) = build_rows(&mut reader, &vocab, &opts).unwrap();
        assert_eq!(rows[0][0], BOS);
        assert_eq!(rows[0][1], vocab.id_of("<elo:low>").unwrap());
        assert_eq!(rows[0][2], vocab.id_of("e2e4").unwrap());
    }

    #[test]
    fn a_game_outside_every_band_is_rejected() {
        let text = pgn(&[("2500", "Normal", "1. e4 e5 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = vocab_with_bands();
        let opts = CorpusOptions {
            condition: Some(ConditionSpec {
                key: "WhiteElo".to_string(),
                bands: vec![ConditionBand {
                    min: 0,
                    max: 1599,
                    token: "<elo:low>".to_string(),
                }],
            }),
            ..Default::default()
        };
        let (rows, stats) = build_rows(&mut reader, &vocab, &opts).unwrap();
        assert!(rows.is_empty());
        assert_eq!(stats.rejected_by_condition, 1);
    }

    #[test]
    fn a_band_the_vocabulary_does_not_carry_is_an_error() {
        let text = pgn(&[("1500", "Normal", "1. e4 e5 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let opts = CorpusOptions {
            condition: Some(ConditionSpec {
                key: "WhiteElo".to_string(),
                bands: vec![ConditionBand {
                    min: 0,
                    max: 4000,
                    token: "<elo:low>".to_string(),
                }],
            }),
            ..Default::default()
        };
        let err = build_rows(&mut reader, &vocab, &opts).unwrap_err();
        assert!(matches!(
            err,
            CorpusError::UnknownToken {
                kind: "condition",
                ..
            }
        ));
    }

    #[test]
    fn overlong_rows_are_dropped_by_default_and_truncated_on_request() {
        let text = pgn(&[("1500", "Normal", "1. e4 e5 2. Nf3 Nc6 1-0")]);
        let vocab = MoveVocab::new(&[]).unwrap();

        let drop_opts = CorpusOptions {
            max_len: Some(3),
            ..Default::default()
        };
        let mut reader = PgnReader::new(Cursor::new(text.clone()));
        let (rows, stats) = build_rows(&mut reader, &vocab, &drop_opts).unwrap();
        assert!(rows.is_empty());
        assert_eq!(stats.dropped_overlong, 1);

        let truncate_opts = CorpusOptions {
            max_len: Some(3),
            overlong: Overlong::Truncate,
            ..Default::default()
        };
        let mut reader = PgnReader::new(Cursor::new(text));
        let (rows, stats) = build_rows(&mut reader, &vocab, &truncate_opts).unwrap();
        assert_eq!(rows[0].len(), 3);
        assert_eq!(stats.truncated, 1);
    }

    #[test]
    fn reading_stops_at_max_rows() {
        let text = pgn(&[
            ("1500", "Normal", "1. e4 e5 1-0"),
            ("1500", "Normal", "1. d4 d5 1-0"),
            ("1500", "Normal", "1. c4 c5 1-0"),
        ]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let opts = CorpusOptions {
            max_rows: 2,
            ..Default::default()
        };
        let (rows, stats) = build_rows(&mut reader, &vocab, &opts).unwrap();
        assert_eq!(rows.len(), 2);
        // The third game was never read.
        assert_eq!(stats.games_read, 2);
    }

    #[test]
    fn every_game_read_is_accounted_for() {
        let text = pgn(&[
            ("1500", "Normal", "1. e4 e5 1-0"),
            ("1500", "Time forfeit", "1. d4 d5 1-0"),
            ("1500", "Normal", "1. Qh6 1-0"),
        ]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let opts = CorpusOptions {
            filter: GameFilter::accept_all().decided_on_the_board(),
            ..Default::default()
        };
        let (_, s) = build_rows(&mut reader, &vocab, &opts).unwrap();
        let accounted = s.rows
            + s.rejected_by_tags
            + s.rejected_by_length
            + s.rejected_by_condition
            + s.dropped_overlong
            + s.replay_failures;
        assert_eq!(accounted, s.games_read);
    }
}
