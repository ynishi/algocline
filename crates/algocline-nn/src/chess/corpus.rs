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

use cozy_chess::{Board, Move};
use thiserror::Error;

use crate::chess::filter::GameFilter;
use crate::chess::pgn::{game_to_uci, uci_standard, PgnError, PgnReader};
use crate::chess::vocab::{MoveVocab, BOS};
use crate::train::{Batch, Dataset, DatasetError, DatasetOpts};

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

/// Whose moves the loss is scored on.
///
/// A game is one row holding both players' moves, so without a mask a
/// model trained on it learns to play both sides. That is fine while
/// the corpus is symmetric — a band filter that puts both players in
/// the same range makes the two sides the same population — and wrong
/// as soon as it is not. "White is a 2000, Black is anyone" is a
/// perfectly good corpus, and every Black move in it is evidence about
/// a player who is not being modelled.
///
/// The mask is per position: `1.0` where the row holds a move by the
/// scored side, `0.0` elsewhere. `Loss::compute` averages over the
/// scored positions only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoredSide {
    /// Score every move. No mask is emitted.
    #[default]
    Both,
    /// Score White's moves only.
    White,
    /// Score Black's moves only.
    Black,
}

impl ScoredSide {
    /// Whether the move at `move_index` (0-based, White to move on
    /// even indices) is scored.
    fn scores(&self, move_index: usize) -> bool {
        match self {
            ScoredSide::Both => true,
            ScoredSide::White => move_index.is_multiple_of(2),
            ScoredSide::Black => !move_index.is_multiple_of(2),
        }
    }
}

/// A range of an integer tag mapped to a condition token.
///
/// Comparable because the band list is the part of a model's identity
/// that its tensors do not carry: the tokens occupy ids `2..2+n`
/// whatever the strings say, and the vocabulary rounds up to the same
/// power of two for any plausible number of them. A resume that wants
/// to know whether a checkpoint was trained on the bands it is being
/// asked for has to compare the bands themselves.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// Whose moves the loss is scored on.
    pub scored_side: ScoredSide,
}

impl Default for CorpusOptions {
    fn default() -> Self {
        Self {
            filter: GameFilter::accept_all(),
            max_rows: 20_000,
            max_len: None,
            overlong: Overlong::default(),
            condition: None,
            scored_side: ScoredSide::default(),
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
    /// Dropped because the scored side had no move left in the row.
    ///
    /// A row the loss cannot score trains as a zero-gradient no-op
    /// while still advancing the step, so it is rejected here rather
    /// than passed on.
    pub rejected_unscorable: usize,
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

/// The rows a build produced, and what it did with everything else.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// Token id rows, one per kept game.
    pub rows: Vec<Vec<u32>>,
    /// Per-position loss masks, aligned row for row with [`rows`].
    ///
    /// `None` when [`ScoredSide::Both`] was asked for, since then every
    /// position is scored and a mask of all ones carries no
    /// information.
    ///
    /// [`rows`]: Corpus::rows
    pub masks: Option<Vec<Vec<f32>>>,
    /// What happened to every game read.
    pub stats: CorpusStats,
}

impl Corpus {
    /// Pair rows with their masks for
    /// [`crate::train::data::TeacherCardDataset::from_rows`].
    ///
    /// When no mask was produced, every position is scored, so an
    /// all-ones mask is synthesised — the dataset takes pairs, and a
    /// uniformly scored row is exactly what all ones means.
    pub fn into_teacher_rows(self) -> Vec<(Vec<u32>, Vec<f32>)> {
        match self.masks {
            Some(masks) => self.rows.into_iter().zip(masks).collect(),
            None => self
                .rows
                .into_iter()
                .map(|row| {
                    let mask = vec![1.0f32; row.len()];
                    (row, mask)
                })
                .collect(),
        }
    }
}

/// Read games and encode the ones that pass the filter.
pub fn build_rows<R: BufRead>(
    reader: &mut PgnReader<R>,
    vocab: &MoveVocab,
    opts: &CorpusOptions,
) -> Result<Corpus, CorpusError> {
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut masks: Vec<Vec<f32>> = Vec::new();
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
        let mut mask = Vec::with_capacity(moves.len() + 2);
        row.push(BOS);
        mask.push(0.0f32);
        if let Some(id) = condition_id {
            row.push(id);
            // The condition token is given, not predicted: scoring it
            // would train the model to guess which band it is playing
            // as.
            mask.push(0.0);
        }
        for (i, mv) in moves.iter().enumerate() {
            let id = vocab.id_of(mv).ok_or_else(|| CorpusError::UnknownToken {
                kind: "move",
                token: mv.clone(),
            })?;
            row.push(id);
            mask.push(if opts.scored_side.scores(i) { 1.0 } else { 0.0 });
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
                        mask.truncate(max_len);
                        stats.truncated += 1;
                    }
                }
            }
        }

        // Position 0 gates no target — the training loop shifts the
        // mask alongside the targets — so a row whose only scored
        // positions are there scores nothing. Such a row would train
        // as a zero-gradient step that still counts as a step.
        if opts.scored_side != ScoredSide::Both && !mask.iter().skip(1).any(|m| *m != 0.0) {
            stats.rejected_unscorable += 1;
            continue;
        }

        stats.tokens += row.len();
        rows.push(row);
        masks.push(mask);
    }

    stats.rows = rows.len();
    let masks = match opts.scored_side {
        ScoredSide::Both => None,
        _ => Some(masks),
    };
    Ok(Corpus { rows, masks, stats })
}

/// Rows plus, for every position, the moves that were legal there.
///
/// # Why this exists
///
/// Cross-entropy over the whole vocabulary makes the model spend
/// capacity learning which moves are illegal. Measured on a trained
/// checkpoint: 1.59 of 4.52 nats — a third of the objective — went on
/// keeping mass off moves that did not exist in the position. Decoding
/// then throws that work away, because it walks the ranking against
/// the legal set regardless of what the model believed.
///
/// Restricting the loss to the legal moves spends that third on the
/// question that survives to inference: which of the available moves
/// to play.
///
/// # Cost
///
/// The legal sets are not stored. Holding them for a 600,000-row
/// corpus would take gigabytes, so each batch replays its own rows —
/// about 66 move generations per row, which is microseconds against a
/// step that takes a hundred milliseconds.
pub struct LegalMaskedDataset {
    rows: Vec<(Vec<u32>, Vec<f32>)>,
    /// Vocabulary id of each move, in the enumeration order a board
    /// replay produces them.
    vocab: MoveVocab,
    /// Tokens that are not moves and can never be a legal answer.
    prefix_len: usize,
    opts: DatasetOpts,
    cursor: usize,
}

impl LegalMaskedDataset {
    /// Wrap rows that begin with `prefix_len` non-move tokens.
    ///
    /// `prefix_len` is 1 for `[BOS, moves..]` and 2 when a condition
    /// token follows the BOS: those positions hold no move, so no legal
    /// set applies to them and they are left unrestricted.
    pub fn new(
        rows: Vec<(Vec<u32>, Vec<f32>)>,
        vocab: MoveVocab,
        prefix_len: usize,
        opts: DatasetOpts,
    ) -> Self {
        Self {
            rows,
            vocab,
            prefix_len,
            opts,
            cursor: 0,
        }
    }

    /// Rows held.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Replay one row, collecting the legal ids at every position.
    ///
    /// A position whose token is not a move it can reach — the prefix,
    /// the padding past the game's end, or anything after a token the
    /// board cannot play — gets an empty set, which the training loop
    /// reads as "do not restrict this position".
    fn legal_sets(&self, row: &[u32]) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); row.len()];
        let mut board = Board::default();
        for pos in self.prefix_len..row.len() {
            let mut ids = Vec::new();
            let mut played: Option<Move> = None;
            board.generate_moves(|moves| {
                for mv in moves {
                    if let Some(id) = self.vocab.id_of(&uci_standard(&board, mv)) {
                        if id == row[pos] {
                            played = Some(mv);
                        }
                        ids.push(id);
                    }
                }
                false
            });
            match played {
                // Only a position whose own token is one of the legal
                // moves may be restricted. Restricting a position whose
                // target is not in the set — padding past the end of
                // the game — would score the target at negative
                // infinity, and an infinity multiplied by that
                // position's zero loss weight is NaN, not zero.
                Some(mv) => {
                    out[pos] = ids;
                    board.play_unchecked(mv);
                }
                // Padding, or a token this position cannot produce.
                // Everything after is unreachable, so the sets stop
                // here rather than being invented.
                None => break,
            }
        }
        out
    }
}

impl Dataset for LegalMaskedDataset {
    fn next_batch(&mut self) -> Result<Option<Batch>, DatasetError> {
        if self.cursor >= self.rows.len() {
            return Ok(None);
        }
        let start = self.cursor;
        let end = (start + self.opts.batch_size).min(self.rows.len());
        self.cursor = end;
        let ctx = self.opts.ctx_len;
        let pad = self.opts.pad_id;

        let mut input_ids = Vec::with_capacity(end - start);
        let mut loss_mask = Vec::with_capacity(end - start);
        let mut allowed = Vec::with_capacity(end - start);
        for (row, mask) in &self.rows[start..end] {
            let padded: Vec<u32> = row
                .iter()
                .copied()
                .chain(std::iter::repeat(pad))
                .take(ctx)
                .collect();
            let padded_mask: Vec<f32> = mask
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .take(ctx)
                .collect();
            allowed.push(self.legal_sets(&padded));
            input_ids.push(padded);
            loss_mask.push(padded_mask);
        }

        Ok(Some(Batch {
            input_ids,
            loss_mask: Some(loss_mask),
            is_last: end == self.rows.len(),
            allowed_ids: Some(allowed),
        }))
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.rows.len())
    }
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

    fn build(text: String, vocab: &MoveVocab, opts: &CorpusOptions) -> Corpus {
        let mut reader = PgnReader::new(Cursor::new(text));
        build_rows(&mut reader, vocab, opts).unwrap()
    }

    #[test]
    fn a_row_is_bos_then_the_moves() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        assert_eq!(c.stats.rows, 1);
        assert_eq!(c.rows[0].len(), 3);
        assert_eq!(c.rows[0][0], BOS);
        assert_eq!(c.rows[0][1], vocab.id_of("e2e4").unwrap());
        assert_eq!(c.rows[0][2], vocab.id_of("e7e5").unwrap());
        assert_eq!(c.stats.tokens, 3);
        // Scoring both sides needs no mask.
        assert!(c.masks.is_none());
    }

    #[test]
    fn scoring_one_side_marks_only_that_sides_moves() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let text = pgn(&[("1500", "Normal", "1. e4 e5 2. Nf3 Nc6 1-0")]);

        let white = build(
            text.clone(),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::White,
                ..Default::default()
            },
        );
        // [BOS, e2e4, e7e5, g1f3, b8c6]
        assert_eq!(white.masks.unwrap()[0], vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let black = build(
            text,
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::Black,
                ..Default::default()
            },
        );
        assert_eq!(black.masks.unwrap()[0], vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn the_condition_token_is_given_not_scored() {
        let vocab = vocab_with_bands();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::White,
                condition: Some(ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![ConditionBand {
                        min: 0,
                        max: 1599,
                        token: "<elo:low>".to_string(),
                    }],
                }),
                ..Default::default()
            },
        );
        // [BOS, <elo:low>, e2e4, e7e5]
        assert_eq!(c.masks.unwrap()[0], vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_row_the_loss_cannot_score_is_rejected() {
        // One ply: White moves, Black never does, so scoring Black
        // leaves nothing. min_plies is relaxed so the row reaches the
        // scorability check rather than the length filter.
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 1-0")]),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::Black,
                ..Default::default()
            },
        );
        assert!(c.rows.is_empty());
        assert_eq!(c.stats.rejected_unscorable, 1);
    }

    #[test]
    fn teacher_rows_pair_ids_with_masks() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::White,
                ..Default::default()
            },
        );
        let paired = c.into_teacher_rows();
        assert_eq!(paired.len(), 1);
        assert_eq!(paired[0].0.len(), paired[0].1.len());

        // Without a mask every position is scored, so all ones.
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        let paired = c.into_teacher_rows();
        assert_eq!(paired[0].1, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn tags_are_tested_before_the_replay() {
        // The second game's movetext is nonsense; it must never be
        // replayed, because its tags are rejected first.
        let text = pgn(&[
            ("1500", "Normal", "1. e4 e5 1-0"),
            ("1500", "Time forfeit", "1. Qz9 1-0"),
        ]);
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            text,
            &vocab,
            &CorpusOptions {
                filter: GameFilter::accept_all().decided_on_the_board(),
                ..Default::default()
            },
        );
        assert_eq!(c.rows.len(), 1);
        assert_eq!(c.stats.rejected_by_tags, 1);
        assert_eq!(c.stats.replay_failures, 0);
    }

    #[test]
    fn a_replay_failure_is_counted_and_kept() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. Qh6 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        assert!(c.rows.is_empty());
        assert_eq!(c.stats.replay_failures, 1);
        assert!(c.stats.first_replay_failure.is_some());
    }

    #[test]
    fn the_condition_token_sits_right_after_bos() {
        let vocab = vocab_with_bands();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
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
            },
        );
        assert_eq!(c.rows[0][0], BOS);
        assert_eq!(c.rows[0][1], vocab.id_of("<elo:low>").unwrap());
        assert_eq!(c.rows[0][2], vocab.id_of("e2e4").unwrap());
    }

    #[test]
    fn a_game_outside_every_band_is_rejected() {
        let vocab = vocab_with_bands();
        let c = build(
            pgn(&[("2500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
                condition: Some(ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![ConditionBand {
                        min: 0,
                        max: 1599,
                        token: "<elo:low>".to_string(),
                    }],
                }),
                ..Default::default()
            },
        );
        assert!(c.rows.is_empty());
        assert_eq!(c.stats.rejected_by_condition, 1);
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

        let dropped = build(
            text.clone(),
            &vocab,
            &CorpusOptions {
                max_len: Some(3),
                ..Default::default()
            },
        );
        assert!(dropped.rows.is_empty());
        assert_eq!(dropped.stats.dropped_overlong, 1);

        let cut = build(
            text,
            &vocab,
            &CorpusOptions {
                max_len: Some(3),
                overlong: Overlong::Truncate,
                ..Default::default()
            },
        );
        assert_eq!(cut.rows[0].len(), 3);
        assert_eq!(cut.stats.truncated, 1);
    }

    #[test]
    fn a_truncated_row_keeps_its_mask_the_same_length() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 2. Nf3 Nc6 1-0")]),
            &vocab,
            &CorpusOptions {
                max_len: Some(3),
                overlong: Overlong::Truncate,
                scored_side: ScoredSide::White,
                ..Default::default()
            },
        );
        let masks = c.masks.unwrap();
        assert_eq!(c.rows[0].len(), masks[0].len());
    }

    #[test]
    fn reading_stops_at_max_rows() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[
                ("1500", "Normal", "1. e4 e5 1-0"),
                ("1500", "Normal", "1. d4 d5 1-0"),
                ("1500", "Normal", "1. c4 c5 1-0"),
            ]),
            &vocab,
            &CorpusOptions {
                max_rows: 2,
                ..Default::default()
            },
        );
        assert_eq!(c.rows.len(), 2);
        // The third game was never read.
        assert_eq!(c.stats.games_read, 2);
    }

    #[test]
    fn legal_sets_hold_the_move_that_was_played_and_only_legal_ones() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 2. Nf3 Nc6 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        let rows = c.into_teacher_rows();
        let ds = LegalMaskedDataset::new(
            rows.clone(),
            MoveVocab::new(&[]).unwrap(),
            1, // [BOS, moves..]
            DatasetOpts {
                batch_size: 1,
                ctx_len: 8,
                ..Default::default()
            },
        );
        let sets = ds.legal_sets(&{
            let mut r = rows[0].0.clone();
            r.resize(8, 0);
            r
        });

        // Position 0 is BOS: no move belongs there.
        assert!(sets[0].is_empty());
        // Position 1 is White's first move: twenty legal moves exist,
        // and the one actually played is among them.
        assert_eq!(sets[1].len(), 20);
        assert!(sets[1].contains(&vocab.id_of("e2e4").unwrap()));
        // A move that is not legal from the start position is absent.
        assert!(!sets[1].contains(&vocab.id_of("e2e5").unwrap()));
        // Position 2 is Black's reply, from a different position, so
        // the set differs.
        assert!(sets[2].contains(&vocab.id_of("e7e5").unwrap()));
        assert!(!sets[2].contains(&vocab.id_of("e2e4").unwrap()));
    }

    #[test]
    fn the_batch_carries_one_legal_set_per_position() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 2. Nf3 Nc6 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        let mut ds = LegalMaskedDataset::new(
            c.into_teacher_rows(),
            MoveVocab::new(&[]).unwrap(),
            1,
            DatasetOpts {
                batch_size: 4,
                ctx_len: 8,
                ..Default::default()
            },
        );
        let batch = ds.next_batch().unwrap().expect("a batch");
        let allowed = batch.allowed_ids.expect("legal sets");
        assert_eq!(allowed.len(), batch.input_ids.len());
        assert_eq!(allowed[0].len(), 8);
        // Past the end of the game the row is padding, which no legal
        // set covers.
        assert!(allowed[0][7].is_empty());
    }

    #[test]
    fn every_game_read_is_accounted_for() {
        let text = pgn(&[
            ("1500", "Normal", "1. e4 e5 1-0"),
            ("1500", "Time forfeit", "1. d4 d5 1-0"),
            ("1500", "Normal", "1. Qh6 1-0"),
        ]);
        let vocab = MoveVocab::new(&[]).unwrap();
        let s = build(
            text,
            &vocab,
            &CorpusOptions {
                filter: GameFilter::accept_all().decided_on_the_board(),
                ..Default::default()
            },
        )
        .stats;
        let accounted = s.rows
            + s.rejected_by_tags
            + s.rejected_by_length
            + s.rejected_by_condition
            + s.dropped_overlong
            + s.rejected_unscorable
            + s.replay_failures;
        assert_eq!(accounted, s.games_read);
    }
}
