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

use crate::arch::CondIndex;
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
///
/// [`ScoredSide::Both`] scores every *move* — which is not the same
/// thing as scoring every position, and the difference is why a mask is
/// emitted for it too. `BOS` and the condition token are not moves
/// under any side, and the position before the condition token asks the
/// model to predict which band it is about to be told it is playing as.
/// An all-ones mask scores that, over the whole vocabulary; the model
/// can only fit the band's marginal from a prefix token, unless it is
/// also being conditioned at every position, in which case it can fit
/// it exactly — and does so by pulling the conditioning vector towards
/// the band token's own embedding, which is the coupling the separate
/// `cond_wte` table exists to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoredSide {
    /// Score every move — which is not every position; see the type
    /// doc.
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

/// How a game is matched to a condition band.
///
/// Two kinds because the two attributes conditioned on so far read
/// their tags differently: a rating is an integer inside a range, an
/// ECO family is the leading letter of a code. A closed enum rather
/// than a predicate so the matcher can be written into a shape sidecar
/// and compared across checkpoints, which a function cannot be.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConditionMatcher {
    /// The tag named by [`ConditionSpec::key`] parses as an integer
    /// inside this inclusive range.
    IntRange {
        /// Inclusive lower bound.
        min: i64,
        /// Inclusive upper bound.
        max: i64,
    },
    /// The named tag's value starts with this string.
    ///
    /// Carries its own tag name where [`ConditionMatcher::IntRange`]
    /// reads [`ConditionSpec::key`], because the two axes it serves
    /// are different tags on the same game — a corpus conditioned on
    /// ECO families still derives its integer bands, if any, from a
    /// rating tag.
    TagPrefix {
        /// PGN tag to read, e.g. `ECO`.
        key: String,
        /// The value's required leading characters, e.g. `B`.
        prefix: String,
    },
}

/// A tag condition mapped to a condition token.
///
/// Comparable because the band list is the part of a model's identity
/// that its tensors do not carry: the tokens occupy ids `2..2+n`
/// whatever the strings say, and the vocabulary rounds up to the same
/// power of two for any plausible number of them. A resume that wants
/// to know whether a checkpoint was trained on the bands it is being
/// asked for has to compare the bands themselves.
///
/// # Sidecar compatibility
///
/// A shape written before [`ConditionMatcher`] existed carries a band
/// as `{"min", "max", "token"}`, and those files are evidence for
/// settled results, so they keep parsing (into
/// [`ConditionMatcher::IntRange`]). A band that *is* an integer range
/// is still **written** in that legacy form, so conditioning on
/// ratings produces sidecars an older build reads. A
/// [`ConditionMatcher::TagPrefix`] band has no legacy form; it is
/// written as `{"matcher", "token"}`, which an older build refuses
/// with a parse error — loudly, rather than by scoring a checkpoint
/// whose condition axis it cannot represent.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "BandRepr", into = "BandRepr")]
pub struct ConditionBand {
    /// How a game is matched to this band.
    pub matcher: ConditionMatcher,
    /// Token emitted for games the matcher accepts. Must exist in the
    /// vocabulary's condition block.
    pub token: String,
}

impl ConditionBand {
    /// A band matching an integer tag range — the rating form.
    pub fn rating(min: i64, max: i64, token: impl Into<String>) -> Self {
        Self {
            matcher: ConditionMatcher::IntRange { min, max },
            token: token.into(),
        }
    }

    /// A band matching a tag's leading characters — the ECO form.
    pub fn tag_prefix(
        key: impl Into<String>,
        prefix: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        Self {
            matcher: ConditionMatcher::TagPrefix {
                key: key.into(),
                prefix: prefix.into(),
            },
            token: token.into(),
        }
    }

    /// Narrow a game filter to this band — the walk-side reading.
    ///
    /// Walks filter their games by the band they are scoring so the
    /// fidelity columns mean something, and before this method each
    /// walk derived the predicate itself, which only ever knew the
    /// rating form. One place, next to the matcher, so a band kind
    /// added here is added to every walk at once.
    ///
    /// For an integer range this requires **both** players inside the
    /// range, which is what every walk has asked since
    /// `match-design.md` — a game is only evidence about a band if
    /// both sides were playing at it. That is deliberately stricter
    /// than [`ConditionSpec::band_for`], which reads the one tag named
    /// by the spec: the corpus question is "which token does this row
    /// carry", the walk question is "which games are this band's".
    pub fn narrow(
        &self,
        filter: crate::chess::filter::GameFilter,
    ) -> crate::chess::filter::GameFilter {
        use crate::chess::filter::{TagPredicate, TagRule};
        match &self.matcher {
            ConditionMatcher::IntRange { min, max } => filter.with_rating_band(*min, *max),
            ConditionMatcher::TagPrefix { key, prefix } => {
                let mut filter = filter;
                filter.tags.push(TagPredicate::new(
                    key.clone(),
                    TagRule::StartsWith(prefix.clone()),
                ));
                filter
            }
        }
    }
}

/// The on-disk forms of [`ConditionBand`], for the compatibility
/// argument that type documents.
///
/// `untagged`, and each variant denies unknown fields — without that,
/// an object carrying `min`, `max` **and** `matcher` (a hand-edited
/// sidecar, half-converted) would parse as `Legacy` and silently drop
/// the matcher, with the variant order deciding which half of the
/// edit won. Denied, the hybrid matches neither variant and the file
/// is refused, which is the only reading that does not guess.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum BandRepr {
    /// The shape every sidecar carried before matchers existed.
    Legacy(LegacyBandRepr),
    /// The general shape.
    Matched(MatchedBandRepr),
}

/// `deny_unknown_fields` is a container attribute, so each form is a
/// struct of its own rather than a struct variant.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyBandRepr {
    min: i64,
    max: i64,
    token: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchedBandRepr {
    matcher: ConditionMatcher,
    token: String,
}

impl From<BandRepr> for ConditionBand {
    fn from(repr: BandRepr) -> Self {
        match repr {
            BandRepr::Legacy(LegacyBandRepr { min, max, token }) => ConditionBand {
                matcher: ConditionMatcher::IntRange { min, max },
                token,
            },
            BandRepr::Matched(MatchedBandRepr { matcher, token }) => {
                ConditionBand { matcher, token }
            }
        }
    }
}

impl From<ConditionBand> for BandRepr {
    fn from(band: ConditionBand) -> Self {
        match band.matcher {
            ConditionMatcher::IntRange { min, max } => BandRepr::Legacy(LegacyBandRepr {
                min,
                max,
                token: band.token,
            }),
            matcher @ ConditionMatcher::TagPrefix { .. } => BandRepr::Matched(MatchedBandRepr {
                matcher,
                token: band.token,
            }),
        }
    }
}

/// How a game's condition token is derived from its tags.
///
/// A conditional model reads the band it should play as from the front
/// of the sequence, so the band has to be written into the row. This
/// is the mapping from a tag value to that token.
#[derive(Debug, Clone)]
pub struct ConditionSpec {
    /// Tag the [`ConditionMatcher::IntRange`] bands read, e.g.
    /// `WhiteElo`. A [`ConditionMatcher::TagPrefix`] band names its
    /// own tag and does not consult this.
    pub key: String,
    /// Bands, tested in order.
    pub bands: Vec<ConditionBand>,
}

impl ConditionSpec {
    /// Find the band a game falls in, as a position in [`Self::bands`],
    /// or `None` when no band's matcher accepts the game — the tag
    /// absent, unparsable, and outside-every-range cases all land
    /// there, since none of them is a band.
    ///
    /// The position rather than the token, because both are wanted: the
    /// token goes into the row, and the position is what a model shape
    /// turns into a conditioning-table row. Deriving one from the other
    /// later would mean searching the band list by string for every
    /// game.
    fn band_for(&self, game: &crate::chess::pgn::PgnGame) -> Option<usize> {
        self.bands.iter().position(|b| match &b.matcher {
            ConditionMatcher::IntRange { min, max } => game
                .tag_i64(&self.key)
                .is_some_and(|value| value >= *min && value <= *max),
            ConditionMatcher::TagPrefix { key, prefix } => game
                .tag(key)
                .is_some_and(|value| value.starts_with(prefix.as_str())),
        })
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
    /// Condition derivations, one per slot. Empty for an unconditional
    /// corpus.
    ///
    /// With **one** spec the corpus is what it always was: the band's
    /// token sits in the row after `BOS`, so a prefix-encoded model
    /// reads it there and a per-position one keeps it for row-shape
    /// comparability with prefix arms (see the [`ScoredSide`] doc).
    ///
    /// With **two or more**, no condition token enters the row at all
    /// — the rows are `BOS` then moves, and the conditions reach the
    /// model only as forward arguments
    /// ([`crate::arch::Gpt2Model::forward_conditioned_groups`]). There
    /// is no prefix arm to stay comparable with (a prefix cannot carry
    /// two slots without inventing an order for them), and keeping the
    /// tokens out of the rows is what keeps the multi-slot corpus out
    /// of the prefix-length arithmetic every windowed reader does.
    pub conditions: Vec<ConditionSpec>,
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
            conditions: Vec::new(),
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
    /// Dropped because the row holds no position the loss can score.
    ///
    /// A row the loss cannot score trains as a zero-gradient no-op
    /// while still advancing the step, so it is rejected here rather
    /// than passed on.
    ///
    /// Usually a one-sided [`ScoredSide`] over a row with no move by
    /// that side; under [`ScoredSide::Both`] it takes a row with no
    /// moves at all, since the prefix is unscored under every side.
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
    /// The per-row lists a [`Corpus`] holds are not the same length.
    ///
    /// The three are positional, so a shorter one means some row would
    /// be joined to another game's mask or band — or, if the join
    /// invented a substitute, to one that never existed. Refused rather
    /// than truncated to the shortest, which would drop rows from the
    /// end without saying so.
    #[error(
        "corpus: {rows} row(s), {masks} mask(s), {bands:?} band(s) — the three lists are \
         positional, so a row is nothing without its own"
    )]
    RaggedCorpus {
        /// Token rows held.
        rows: usize,
        /// Loss masks held.
        masks: usize,
        /// Band ordinals held, or `None` for an unconditional corpus.
        bands: Option<usize>,
    },
    /// One row and its mask are not the same length.
    ///
    /// The sibling of [`Self::RaggedCorpus`] one level in: the lists
    /// can hold the right number of entries while an entry is the wrong
    /// shape, and the mask is positional against its own row, so a
    /// disagreement scores the wrong positions.
    ///
    /// Refused here because only one of the two paths downstream would
    /// catch it: [`crate::train::TeacherCardDataset::from_rows`] does,
    /// while [`LegalMaskedDataset`] pads and truncates the ids and the
    /// mask to the context window independently — a short mask becomes
    /// unscored positions and a long one loses its tail, both in
    /// silence, on the `CHESS_LEGAL_MASK=1` path.
    #[error(
        "corpus row {index}: {ids} token(s) but {mask} mask position(s) — a mask is \
         positional against its own row, so one of a different length scores the wrong \
         positions"
    )]
    RaggedRow {
        /// 0-based row index.
        index: usize,
        /// Token ids in the row.
        ids: usize,
        /// Mask positions paired with it.
        mask: usize,
    },
}

/// The rows a build produced, and what it did with everything else.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// Token id rows, one per kept game.
    pub rows: Vec<Vec<u32>>,
    /// Per-position loss masks, aligned row for row with [`rows`].
    ///
    /// Always present, [`ScoredSide::Both`] included. That side scores
    /// every move, and the prefix positions — `BOS` and the condition
    /// token — hold no move under any side, so "all ones" was never a
    /// correct summary of it. See [`ScoredSide`] for what an all-ones
    /// mask costs a conditioned run.
    ///
    /// [`rows`]: Corpus::rows
    pub masks: Vec<Vec<f32>>,
    /// Which band each row fell in — one ordinal per condition slot,
    /// each a position in its own [`ConditionSpec::bands`] — aligned
    /// row for row with [`rows`].
    ///
    /// `None` for an unconditional corpus. The inner length is the
    /// number of specs the corpus was built with, the same for every
    /// row.
    ///
    /// Recorded here because here is where it is known for certain: the
    /// bands are what the build resolved from the game's own tags. Any
    /// later reader has to recover them from a position in the row
    /// instead, and that position is not dependable — a row longer than
    /// the context window is cut, and the tail slice moves an ordinary
    /// move into it; a multi-slot row does not carry them at all.
    ///
    /// [`rows`]: Corpus::rows
    pub bands: Option<Vec<Vec<usize>>>,
    /// What happened to every game read.
    pub stats: CorpusStats,
}

/// One row of a corpus, with everything the trainer needs about it.
///
/// The three travel as one value because the caller shuffles them and
/// lays down several epochs, and a row separated from its band or its
/// mask by a reordering would be trained under another game's, with
/// every shape still lining up.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherRow {
    /// Token ids: `BOS`, the band token if the corpus is conditional,
    /// then the moves.
    pub ids: Vec<u32>,
    /// Per-position loss mask, the same length as [`Self::ids`].
    pub mask: Vec<f32>,
    /// This row's band per condition slot — each a position in its own
    /// [`ConditionSpec::bands`] — or empty for an unconditional corpus.
    ///
    /// Ordinals into band lists, not token ids and not
    /// conditioning-table rows. A token id would select the wrong table
    /// row if it were ever handed over as one (band tokens start at
    /// vocabulary id 2, table rows at 0), and the table row belongs to
    /// the model's [`crate::chess::ModelShape`], which the corpus does
    /// not have. [`crate::chess::train::cond_table`] does that
    /// conversion, in one place.
    pub bands: Vec<usize>,
}

impl Corpus {
    /// Join the three per-row lists into [`TeacherRow`] values.
    ///
    /// # Errors
    ///
    /// [`CorpusError::RaggedCorpus`] when the lists are not the same
    /// length, and [`CorpusError::RaggedRow`] when a row and its mask
    /// are not. [`build_rows`] cannot produce either state — it appends
    /// to all three under one condition and truncates a row and its
    /// mask together — but the fields are public, and the join is where
    /// the guarantee that a row travels with its own mask and band is
    /// actually established, [`TeacherRow::mask`]'s own documented
    /// length included.
    ///
    /// Substituting for a missing entry is what would make the failure
    /// silent: an invented all-ones mask scores the prefix positions
    /// this module deliberately zeroed, and reads afterwards exactly
    /// like a mask that was meant.
    pub fn into_teacher_rows(self) -> Result<Vec<TeacherRow>, CorpusError> {
        let bands_len = self.bands.as_ref().map(|b| b.len());
        if self.masks.len() != self.rows.len() || bands_len.is_some_and(|n| n != self.rows.len()) {
            return Err(CorpusError::RaggedCorpus {
                rows: self.rows.len(),
                masks: self.masks.len(),
                bands: bands_len,
            });
        }
        let mut bands = self.bands.unwrap_or_default().into_iter();
        self.rows
            .into_iter()
            .zip(self.masks)
            .enumerate()
            .map(|(index, (ids, mask))| {
                if ids.len() != mask.len() {
                    return Err(CorpusError::RaggedRow {
                        index,
                        ids: ids.len(),
                        mask: mask.len(),
                    });
                }
                Ok(TeacherRow {
                    ids,
                    mask,
                    bands: bands.next().unwrap_or_default(),
                })
            })
            .collect()
    }
}

impl TeacherRow {
    /// The `(ids, mask)` pairs the datasets take, in row order.
    ///
    /// The bands are dropped here rather than being forgotten silently:
    /// [`crate::chess::train::row_conditions`] resolves them first, and
    /// what it returns is positional against the same order.
    pub fn into_pairs(rows: Vec<TeacherRow>) -> Vec<(Vec<u32>, Vec<f32>)> {
        rows.into_iter().map(|r| (r.ids, r.mask)).collect()
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
    let mut bands: Vec<Vec<usize>> = Vec::new();
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

        // Resolve every slot's condition before replaying: a game
        // outside any slot's bands cannot enter a conditional corpus,
        // and finding that out costs tag lookups instead of a full
        // replay. The vocabulary check runs for every slot even though
        // only a single-slot corpus writes a token into the row — the
        // token is the band's identity everywhere downstream (the
        // shape, the walks, the judges), and a vocabulary that cannot
        // name it would surface as a confusing failure much later.
        let mut ordinals: Vec<usize> = Vec::with_capacity(opts.conditions.len());
        let mut single_slot_id: Option<u32> = None;
        let mut rejected = false;
        for spec in &opts.conditions {
            match spec.band_for(&game) {
                Some(band) => {
                    let token = spec.bands[band].token.as_str();
                    let id = vocab
                        .id_of(token)
                        .ok_or_else(|| CorpusError::UnknownToken {
                            kind: "condition",
                            token: token.to_string(),
                        })?;
                    if opts.conditions.len() == 1 {
                        single_slot_id = Some(id);
                    }
                    ordinals.push(band);
                }
                None => {
                    rejected = true;
                    break;
                }
            }
        }
        if rejected {
            stats.rejected_by_condition += 1;
            continue;
        }

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
        // Set only by a single-slot corpus; a multi-slot one keeps its
        // conditions out of the row entirely (see
        // [`CorpusOptions::conditions`]).
        if let Some(id) = single_slot_id {
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
        //
        // Checked for every side, not only the one-sided ones. Under
        // `Both` the mask used to be discarded, so the check could not
        // have meant anything; now that it is kept, a row of no moves
        // at all — reachable when the filter sets no lower ply bound —
        // is unscorable under `Both` in exactly the same way, and is
        // rejected here rather than a few stages later where it would
        // abort the whole run.
        if !mask.iter().skip(1).any(|m| *m != 0.0) {
            stats.rejected_unscorable += 1;
            continue;
        }

        stats.tokens += row.len();
        rows.push(row);
        masks.push(mask);
        // Pushed only here, after every rejection above, so the band
        // list stays index-for-index with the rows rather than with the
        // games read.
        if !opts.conditions.is_empty() {
            bands.push(ordinals);
        }
    }

    stats.rows = rows.len();
    let bands = (!opts.conditions.is_empty()).then_some(bands);
    Ok(Corpus {
        rows,
        masks,
        bands,
        stats,
    })
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
    /// Conditions row-major at `conds_per_row` per row, or `None` for
    /// an unconditioned dataset. See
    /// [`LegalMaskedDataset::with_conditions`] and
    /// [`LegalMaskedDataset::with_condition_groups`].
    conds: Option<Vec<CondIndex>>,
    conds_per_row: usize,
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
            conds: None,
            conds_per_row: 1,
            opts,
            cursor: 0,
        }
    }

    /// Attach the condition each row was recorded under, in row order.
    ///
    /// The counterpart of
    /// [`crate::train::TeacherCardDataset::with_conditions`], with the
    /// same positional pairing and the same reason for being applied
    /// after construction rather than taken by the constructor.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ConditionCountMismatch`] when the list is not
    /// exactly one entry per row.
    pub fn with_conditions(self, conds: Vec<CondIndex>) -> Result<Self, DatasetError> {
        self.with_condition_groups(conds, 1)
    }

    /// [`Self::with_conditions`] with `per_row` conditions per row,
    /// row-major — the counterpart of
    /// [`crate::train::TeacherCardDataset::with_condition_groups`],
    /// explicit for the same reason: a count that happens to divide
    /// evenly is exactly the mistake an inference would wave through.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ConditionCountMismatch`] when the list is not
    /// exactly `per_row` entries per row, `per_row` of zero included.
    pub fn with_condition_groups(
        mut self,
        conds: Vec<CondIndex>,
        per_row: usize,
    ) -> Result<Self, DatasetError> {
        if per_row == 0 || conds.len() != self.rows.len() * per_row {
            return Err(DatasetError::ConditionCountMismatch {
                conds: conds.len(),
                rows: self.rows.len(),
                per_row,
            });
        }
        self.conds = Some(conds);
        self.conds_per_row = per_row;
        Ok(self)
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
            // Sliced with the same `start..end` as the rows, scaled by
            // the per-row count, so a row and its conditions cannot
            // come apart here.
            conds: self
                .conds
                .as_ref()
                .map(|c| c[start * self.conds_per_row..end * self.conds_per_row].to_vec()),
            conds_per_row: self.conds_per_row,
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
        // `Both` scores every move, and `BOS` is not one.
        assert_eq!(c.masks[0], vec![0.0, 1.0, 1.0]);
    }

    /// The prefix is unscored under every side, so the position that
    /// asks the model to predict the condition token it is about to be
    /// given never enters the loss.
    ///
    /// `Both` used to discard its mask, and the all-ones mask
    /// synthesised in its place scored exactly that position. It is the
    /// one place a conditioned model can fit the band token itself,
    /// which pulls the conditioning vector towards that token's
    /// embedding — the coupling `cond_wte` was separated from `wte` to
    /// avoid.
    #[test]
    fn the_prefix_is_unscored_under_both_sides_too() {
        let vocab = vocab_with_bands();
        let spec = ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![ConditionBand::rating(0, 1599, "<elo:low>")],
        };
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::Both,
                conditions: vec![spec],
                ..Default::default()
            },
        );
        // [BOS, <elo:low>, e2e4, e7e5]
        assert_eq!(c.masks[0], vec![0.0, 0.0, 1.0, 1.0]);
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
        assert_eq!(white.masks[0], vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let black = build(
            text,
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::Black,
                ..Default::default()
            },
        );
        assert_eq!(black.masks[0], vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn the_condition_token_is_given_not_scored() {
        let vocab = vocab_with_bands();
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions {
                scored_side: ScoredSide::White,
                conditions: vec![ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![ConditionBand::rating(0, 1599, "<elo:low>")],
                }],
                ..Default::default()
            },
        );
        // [BOS, <elo:low>, e2e4, e7e5]
        assert_eq!(c.masks[0], vec![0.0, 0.0, 1.0, 0.0]);
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
        let paired = c.into_teacher_rows().expect("the lists agree");
        assert_eq!(paired.len(), 1);
        assert_eq!(paired[0].ids.len(), paired[0].mask.len());
        // An unconditional corpus has no band to carry.
        assert!(paired[0].bands.is_empty());

        // `Both` scores every move and neither prefix token.
        let c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        let paired = c.into_teacher_rows().expect("the lists agree");
        assert_eq!(paired[0].mask, vec![0.0, 1.0, 1.0]);
    }

    /// The join is where a row is tied to its own mask and band, so a
    /// disagreement between the three lists is refused there rather
    /// than papered over with a substitute.
    ///
    /// `build_rows` cannot produce this state; the fields are public,
    /// so the join cannot assume it did.
    #[test]
    fn a_corpus_whose_lists_disagree_is_refused() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let mut c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        c.masks.clear();
        let err = c.into_teacher_rows().unwrap_err();
        assert!(
            matches!(
                err,
                CorpusError::RaggedCorpus {
                    rows: 1,
                    masks: 0,
                    bands: None
                }
            ),
            "{err:?}"
        );
    }

    /// And the same one level in: the lists can hold the right number
    /// of entries while a mask is the wrong length for its own row.
    ///
    /// Only one of the two dataset paths would have caught this.
    /// `LegalMaskedDataset` pads and truncates the ids and the mask to
    /// the context window independently, so a short mask turns into
    /// unscored positions without a word.
    #[test]
    fn a_row_whose_mask_is_the_wrong_length_is_refused() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let mut c = build(
            pgn(&[("1500", "Normal", "1. e4 e5 1-0")]),
            &vocab,
            &CorpusOptions::default(),
        );
        c.masks[0].pop();
        let err = c.into_teacher_rows().unwrap_err();
        assert!(
            matches!(
                err,
                CorpusError::RaggedRow {
                    index: 0,
                    ids: 3,
                    mask: 2
                }
            ),
            "{err:?}"
        );
    }

    /// The band a row was built under travels with the row, so nothing
    /// downstream has to read it back out of a position the context
    /// window can move.
    #[test]
    fn a_conditional_corpus_records_which_band_each_row_fell_in() {
        let vocab = vocab_with_bands();
        let spec = ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![
                ConditionBand::rating(0, 1599, "<elo:low>"),
                ConditionBand::rating(1600, 4000, "<elo:high>"),
            ],
        };
        let c = build(
            pgn(&[
                ("1800", "Normal", "1. e4 e5 1-0"),
                ("1500", "Normal", "1. d4 d5 1-0"),
                // Rejected by the replay, so it must not shift the
                // bands of the rows that follow it.
                ("1500", "Normal", "1. Qh6 1-0"),
                ("1800", "Normal", "1. c4 c5 1-0"),
            ]),
            &vocab,
            &CorpusOptions {
                conditions: vec![spec],
                ..Default::default()
            },
        );
        assert_eq!(c.stats.replay_failures, 1);
        assert_eq!(
            c.bands.as_deref(),
            Some([vec![1usize], vec![0], vec![1]].as_slice())
        );

        // And the ordinal names the band whose token the row carries.
        let rows = c.into_teacher_rows().expect("the lists agree");
        let tokens = ["<elo:low>", "<elo:high>"];
        for row in &rows {
            let [band] = row.bands.as_slice() else {
                panic!("a single-slot corpus bands every row once");
            };
            assert_eq!(row.ids[1], vocab.id_of(tokens[*band]).unwrap());
        }
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
                conditions: vec![ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![
                        ConditionBand::rating(0, 1599, "<elo:low>"),
                        ConditionBand::rating(1600, 4000, "<elo:high>"),
                    ],
                }],
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
                conditions: vec![ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![ConditionBand::rating(0, 1599, "<elo:low>")],
                }],
                ..Default::default()
            },
        );
        assert!(c.rows.is_empty());
        assert_eq!(c.stats.rejected_by_condition, 1);
    }

    /// A prefix band reads the tag its own matcher names, not the
    /// spec's key: the spec here says `WhiteElo` and the bands are
    /// derived from `ECO` anyway.
    #[test]
    fn a_prefix_band_reads_the_tag_its_matcher_names() {
        let vocab = MoveVocab::new(&["<eco:B>".to_string(), "<eco:C>".to_string()]).unwrap();
        let text = "[WhiteElo \"1500\"]\n[ECO \"B20\"]\n\n1. e4 e5 1-0\n\n\
                    [WhiteElo \"1500\"]\n[ECO \"C50\"]\n\n1. d4 d5 1-0\n\n\
                    [WhiteElo \"1500\"]\n[ECO \"A00\"]\n\n1. c4 c5 1-0\n\n"
            .to_string();
        let c = build(
            text,
            &vocab,
            &CorpusOptions {
                conditions: vec![ConditionSpec {
                    key: "WhiteElo".to_string(),
                    bands: vec![
                        ConditionBand::tag_prefix("ECO", "B", "<eco:B>"),
                        ConditionBand::tag_prefix("ECO", "C", "<eco:C>"),
                    ],
                }],
                ..Default::default()
            },
        );
        assert_eq!(c.bands.as_deref(), Some([vec![0usize], vec![1]].as_slice()));
        assert_eq!(c.stats.rejected_by_condition, 1);
    }

    /// A two-slot corpus keeps its conditions out of the rows — `BOS`
    /// then moves — and records one ordinal per slot instead, each
    /// against its own spec. A game either slot cannot band is
    /// rejected once.
    #[test]
    fn a_multi_slot_corpus_bands_rows_without_tokens_in_them() {
        let vocab = MoveVocab::new(&[
            "<eco:B>".to_string(),
            "<eco:C>".to_string(),
            "<elo:low>".to_string(),
            "<elo:high>".to_string(),
        ])
        .unwrap();
        let eco = ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![
                ConditionBand::tag_prefix("ECO", "B", "<eco:B>"),
                ConditionBand::tag_prefix("ECO", "C", "<eco:C>"),
            ],
        };
        let elo = ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: vec![
                ConditionBand::rating(0, 1599, "<elo:low>"),
                ConditionBand::rating(1600, 4000, "<elo:high>"),
            ],
        };
        let text = "[WhiteElo \"1800\"]\n[ECO \"C50\"]\n\n1. e4 e5 1-0\n\n\
                    [WhiteElo \"1500\"]\n[ECO \"B20\"]\n\n1. d4 d5 1-0\n\n\
                    [WhiteElo \"1500\"]\n[ECO \"A00\"]\n\n1. c4 c5 1-0\n\n"
            .to_string();
        let c = build(
            text,
            &vocab,
            &CorpusOptions {
                conditions: vec![eco, elo],
                ..Default::default()
            },
        );
        // The A00 game bands in the elo slot but not the eco one.
        assert_eq!(c.stats.rejected_by_condition, 1);
        assert_eq!(
            c.bands.as_deref(),
            Some([vec![1usize, 1], vec![0, 0]].as_slice())
        );
        // No condition token after BOS: the first id past it is a move.
        assert_eq!(c.rows[0][0], BOS);
        assert_eq!(c.rows[0][1], vocab.id_of("e2e4").unwrap());
        assert_eq!(c.masks[0], vec![0.0, 1.0, 1.0]);
    }

    /// The sidecar compatibility split the type documents: a legacy
    /// `{min, max, token}` object parses into a rating band, a rating
    /// band writes back exactly those keys, and a prefix band writes
    /// the `matcher` form without them — which the pre-matcher
    /// `ConditionBand`, whose `min` and `max` carried no default,
    /// could not have parsed into anything.
    #[test]
    fn sidecar_bands_keep_the_legacy_form_exactly_for_ratings_and_only_for_ratings() {
        let legacy = serde_json::json!({"min": 0, "max": 1599, "token": "<elo:low>"});
        let parsed: ConditionBand = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(parsed, ConditionBand::rating(0, 1599, "<elo:low>"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), legacy);

        let prefix = ConditionBand::tag_prefix("ECO", "B", "<eco:B>");
        let written = serde_json::to_value(&prefix).unwrap();
        assert!(written.get("min").is_none() && written.get("max").is_none());
        assert!(written.get("matcher").is_some());
        let back: ConditionBand = serde_json::from_value(written).unwrap();
        assert_eq!(back, prefix);
    }

    /// A hand-edited sidecar carrying both the legacy fields and a
    /// matcher is refused rather than read as whichever half the
    /// variant order favours.
    #[test]
    fn a_hybrid_band_object_is_refused_not_guessed() {
        let hybrid = serde_json::json!({
            "min": 0, "max": 1599, "token": "<elo:low>",
            "matcher": {"TagPrefix": {"key": "ECO", "prefix": "B"}}
        });
        assert!(serde_json::from_value::<ConditionBand>(hybrid).is_err());
    }

    /// The walk-side reading: a rating band requires both players
    /// inside the range, a prefix band requires the named tag.
    #[test]
    fn narrow_is_the_walk_side_reading_of_a_band() {
        let mut reader = PgnReader::new(Cursor::new(
            "[WhiteElo \"1650\"]\n[BlackElo \"1900\"]\n[ECO \"B20\"]\n\n1. e4 e5 1-0\n\n",
        ));
        let game = reader.next_game().unwrap().unwrap();

        let rating = ConditionBand::rating(1600, 1799, "<elo:mid>");
        assert!(!rating.narrow(GameFilter::accept_all()).accepts_tags(&game));

        let eco = ConditionBand::tag_prefix("ECO", "B", "<eco:B>");
        assert!(eco.narrow(GameFilter::accept_all()).accepts_tags(&game));
        let other = ConditionBand::tag_prefix("ECO", "C", "<eco:C>");
        assert!(!other.narrow(GameFilter::accept_all()).accepts_tags(&game));
    }

    #[test]
    fn a_band_the_vocabulary_does_not_carry_is_an_error() {
        let text = pgn(&[("1500", "Normal", "1. e4 e5 1-0")]);
        let mut reader = PgnReader::new(Cursor::new(text));
        let vocab = MoveVocab::new(&[]).unwrap();
        let opts = CorpusOptions {
            conditions: vec![ConditionSpec {
                key: "WhiteElo".to_string(),
                bands: vec![ConditionBand::rating(0, 4000, "<elo:low>")],
            }],
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
        assert_eq!(c.rows[0].len(), c.masks[0].len());
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
        let rows = c.into_teacher_rows().expect("the lists agree");
        let first = rows[0].ids.clone();
        let ds = LegalMaskedDataset::new(
            TeacherRow::into_pairs(rows),
            MoveVocab::new(&[]).unwrap(),
            1, // [BOS, moves..]
            DatasetOpts {
                batch_size: 1,
                ctx_len: 8,
                ..Default::default()
            },
        );
        let sets = ds.legal_sets(&{
            let mut r = first;
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
            TeacherRow::into_pairs(c.into_teacher_rows().expect("the lists agree")),
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
