//! What one walk saw, position by position, so that several walks can
//! be read together afterwards.
//!
//! # Why a file
//!
//! `chess_cond` prints a summary of a walk, and a summary is enough to
//! read a run by. It is not enough to *judge* one. The hypotheses in the
//! plan are differences **across arms scored on the same positions** —
//! `flip(perpos) - flip(prefix)`, and a decay ratio one arm's against
//! another's — and each arm is a separate checkpoint scored in a
//! separate process. Two summaries cannot be subtracted with an error
//! bar on the result, because by then the positions each number came
//! from have been averaged away.
//!
//! So the walk also writes out what it saw at each position, and the
//! statistics are assembled from several of those files
//! ([`crate::chess::steerability`]).
//!
//! # Why the game index is on every record
//!
//! Those 3,000 positions come from roughly 95 games, and positions
//! inside one game are not independent draws. Every error bar in the
//! plan is therefore a bootstrap that resamples **games**, which is only
//! expressible if each position remembers the game it came from. Nothing
//! else in the walk retained that.
//!
//! # Why the arms can be trusted to line up
//!
//! `chess_cond`'s walk is model-independent: it steps through the PGN,
//! and a position enters the sample when the board offers at least two
//! legal moves. Nothing in that decision consults the checkpoint. Two
//! arms walked over the same holdout with the same side and the same
//! cap therefore visit the same positions in the same order.
//!
//! That is an argument, not a guarantee — a different holdout file, a
//! different cap, or a checkpoint with a different band list would all
//! produce files that still parse. [`AlignedArms`] checks the sequence
//! of `(game, ply)` pairs rather than trusting it, and refuses the set
//! outright rather than joining on whatever prefix happens to match.
//!
//! # Format
//!
//! JSON Lines: the first line is a [`WalkHeader`], every line after it a
//! [`PositionRecord`]. One line per position keeps the file greppable
//! and lets a partial read say which position it stopped at; the header
//! carries what a reader needs before it can interpret any of them,
//! including the context window that fixes the depth buckets.
//!
//! The file is written whole at the end of a walk rather than streamed,
//! so the counts in the header are the counts in the file and a run that
//! died half way leaves no file to be mistaken for a short one.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::chess::CondEncoding;

/// Format version carried in every header, and the version this build
/// writes.
///
/// Version 2 added [`GammaRecord::top2_margin`]. Version 3 added
/// [`GammaRecord::ce`] and [`PositionRecord::n_legal`]. Version 4 added
/// [`WalkHeader::legal_input`]. Version 5 added
/// [`WalkHeader::games_of`]. No bump has changed a field that
/// already existed. A reader accepts [`MIN_READABLE_VERSION`] through
/// this and refuses anything outside that range rather than
/// interpreting unfamiliar fields as absent, which is the same stance
/// [`crate::chess::ModelShape`] takes for the same reason.
///
/// This is also the constant a reader consults to ask **whether a
/// walk was in a position to write a given field**. The fields added
/// after version 1 to a *record* are all `Option`, and their `None`
/// says what it says about the position rather than about the format;
/// the version is where the format question is answered. See
/// [`GammaRecord::ce`]. Version 4's addition is to the *header* and is
/// a plain `bool`, for a reason that field documents: there the default
/// is not a stand-in for an unanswered question. Version 5's is a
/// header `Option`, because for it the two absences are different
/// facts again — [`WalkHeader::games_of`] has the arithmetic.
pub const FORMAT_VERSION: u32 = 5;

/// Oldest version this build still reads.
///
/// A version 1 record carries no `top2_margin`, no `ce` and no
/// `n_legal`, and all three default to `None`, so a version 1 file
/// parses into exactly what it said rather than into a fabricated
/// number. Its header carries no `legal_input` either, and that one
/// defaults to `false` — a value rather than an absence, which is
/// sound here only because of the argument
/// [`WalkHeader::legal_input`] makes and not as a general licence.
/// That matters concretely: the walks Phase 5 was confirmed on are
/// version 1 files, and refusing them to gain fields nothing reads yet
/// would discard the evidence for a settled result.
///
/// It is a constant of its own rather than a `1` written into the
/// comparison because it has to be **raised by hand** the first time a
/// bump is not purely additive. Accepting a range is sound only while
/// every version in it is the same shape plus fields that default: a
/// change that reinterprets an existing field leaves an old file
/// parsing cleanly into something it never meant, and has to move this
/// floor in the same commit that makes it. Nothing here enforces that.
/// It is a rule whoever bumps [`FORMAT_VERSION`] has to follow.
pub const MIN_READABLE_VERSION: u32 = 1;

/// What a walk was, so its records can be interpreted and compared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WalkHeader {
    /// [`FORMAT_VERSION`] at the time of writing.
    pub version: u32,
    /// Checkpoint that was scored.
    pub ckpt: String,
    /// Holdout PGN that was walked.
    pub holdout: String,
    /// Which side's positions were scored, as
    /// [`crate::chess::corpus::ScoredSide`] renders it.
    pub side: String,
    /// How the checkpoint was conditioned.
    pub encoding: CondEncoding,
    /// Whether the checkpoint was handed the ids allowed at each
    /// position, as [`crate::chess::ModelShape::legal_input`] records
    /// it.
    ///
    /// The other axis of [`crate::chess::ShapeKind`], and here for the
    /// reason [`Self::encoding`] is: an arm's role in a comparison is a
    /// claim about how its checkpoint was trained, and a header that
    /// cannot state the claim cannot be checked against it. The next
    /// experiment's five arms are **all** prefix-conditioned and differ
    /// only on this axis, so without it two walks that differ in the
    /// only way that matters produce headers that cannot be told apart
    /// — and [`crate::chess::steerability::check_legality_roles`],
    /// which exists to catch an arm in the wrong slot, would be blind
    /// to the swap that reverses what its margin means.
    ///
    /// # Why a plain `bool` rather than an `Option`
    ///
    /// `false` is what a record written before version 4 *means*, not a
    /// stand-in for something it failed to say. No reader in this crate
    /// would open a legality checkpoint before `4080715`, so no walk of
    /// one can have been written at version 1, 2 or 3. That is the
    /// difference from [`GammaRecord::ce`], where the absence of the
    /// field and the absence of a value are two different facts and an
    /// `Option` has to carry both.
    ///
    /// The argument leaves one gap, and this does not try to close it
    /// with a third state. A walk written from `4080715` itself — after
    /// the readers opened and before this field existed — reads back
    /// `false` whatever its checkpoint was. Such a walk is refused
    /// rather than believed, and refused on its **version** rather than
    /// on this value:
    /// [`crate::chess::steerability::check_legality_roles`] requires
    /// [`crate::chess::steerability::LEGALITY_AXIS_VERSION`] of every
    /// arm it checks. Reading the value alone would not do, and that is
    /// the whole reason the version is read: it refuses only the slots
    /// that call for `true`, so the same stale walk of the same
    /// legality checkpoint would pass wherever `false` is what the role
    /// wanted. The default is honest where it is right; the version is
    /// what makes it loud where it is not.
    #[serde(default)]
    pub legal_input: bool,
    /// Context window the rows were cut to.
    ///
    /// Here because it fixes a depth bucket edge: a conditioned row is
    /// `[BOS, band] + moves`, so a row stops fitting at ply `ctx - 1`
    /// and the deep bucket has to end at `ctx - 2` to keep the windowed
    /// regime out of it. A reader that assumed 128 would silently mix
    /// the two on a run with another context size.
    pub ctx: usize,
    /// Condition tokens, in vocabulary order. Flip rate is defined
    /// against the first of them, so two arms with different band lists
    /// are not measuring the same quantity.
    pub bands: Vec<String>,
    /// Guidance strengths swept, in the order the per-position records
    /// carry them.
    pub gammas: Vec<f32>,
    /// Positions in the file.
    pub positions: usize,
    /// Games the walk accepted, including any that contributed no
    /// position. [`AlignedArms::games`] reports the clustering figure,
    /// which counts only games that did contribute.
    pub games: usize,
    /// Condition tokens of the filter that selected the walk's games,
    /// or `None` on a file written before version 5.
    ///
    /// The last axis of a walk's identity the header did not state.
    /// [`Self::encoding`] and [`Self::legal_input`] record how the
    /// **checkpoint** was trained; this records which games the
    /// **walk** scored. Before it, a walk over one band's games and a
    /// walk over another's produced headers that could not be told
    /// apart, and any per-band comparison that depends on which games
    /// were walked — "does the correct band predict its own games
    /// best" is one — could have its sign silently reversed by feeding
    /// it the other file. Plan 04 had to establish the walked band
    /// from run logs and from [`AlignedArms`]' position-stream match
    /// instead of from the file itself.
    ///
    /// A list rather than a single token because a walk may be
    /// narrowed on more than one axis at once (an opening family and
    /// a rating band), and `Some(vec![])` is itself a statement: the
    /// walk was deliberately unfiltered. That is why this is an
    /// `Option` where [`Self::legal_input`] is a plain `bool` — there
    /// `false` is what every pre-4 walk meant, here an old file's
    /// absence and a new file's "no filter" are two different facts
    /// and both occur, which is the [`GammaRecord::ce`] situation
    /// rather than the `legal_input` one. A reader whose question
    /// needs the filter refuses `None` on the **version**, the same
    /// move [`crate::chess::steerability::check_legality_roles`]
    /// makes for the legality axis.
    #[serde(default)]
    pub games_of: Option<Vec<String>>,
}

/// One position, under every guidance strength the walk swept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionRecord {
    /// Index of the game this position came from, counting accepted
    /// games from zero. The bootstrap's cluster.
    pub game: usize,
    /// Ply within the game, counting from zero. The depth bucket.
    pub ply: usize,
    /// How many legal moves the position offered.
    ///
    /// Here and not on [`GammaRecord`] because the legal set is a
    /// property of the board: guidance changes how the mass is spread
    /// over those moves and not which moves they are. Per gamma it
    /// would be the same number written once per sweep entry, which
    /// invites a reader to wonder which of them to trust.
    ///
    /// Recorded because a cross-entropy is read against the cost of a
    /// uniform draw over the same moves, and that bar is `ln(n_legal)`.
    /// Without it a reader of [`GammaRecord::ce`] can say a position
    /// was expensive but not whether it was expensive for its size.
    ///
    /// `None` on a record written before format version 3. Whether a
    /// walk was in a position to write it is a question about the
    /// format, so ask it of [`WalkHeader::version`] rather than of this
    /// field — the same reading [`GammaRecord::ce`] documents at
    /// length.
    #[serde(default)]
    pub n_legal: Option<usize>,
    /// One entry per gamma in [`WalkHeader::gammas`], in that order.
    pub at: Vec<GammaRecord>,
}

/// What one position looked like at one guidance strength.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GammaRecord {
    /// Whether any band's top legal move differed from the first
    /// band's. The per-position term the flip rate is the mean of.
    pub flipped: bool,
    /// Jensen-Shannon divergence in bits between the first and last
    /// band's legal-move distributions.
    pub widest_js: f64,
    /// Softmax mass landing on legal moves, meaned over the bands.
    /// Recorded because `§5.1`'s third validity gate is checked at every
    /// gamma and is therefore a per-gamma quantity like the others.
    pub legal_mass: f64,
    /// Whether each band's top legal move equalled the human's, one
    /// entry per band.
    ///
    /// `None` where the move actually played is not in the vocabulary,
    /// so there is nothing to match against. Absent rather than false:
    /// counting an unscoreable position as a miss would drag the top-1
    /// figure down by however many of them there were.
    pub top1: Option<Vec<bool>>,
    /// What the move actually played cost each band, in nats, at this
    /// gamma.
    ///
    /// The negative natural log of the probability that band gives the
    /// played move, once its distribution is restricted to the legal
    /// moves and renormalised. [`ce_over_legal`] is the computation.
    /// One entry per band, in [`WalkHeader::bands`] order, as
    /// [`GammaRecord::top1`] is.
    ///
    /// Per gamma because guidance changes the distribution, and per
    /// band for the reason `top1` is: the two answer the same question
    /// at different resolutions — which band ranked the human's move
    /// first, and how much mass each of them put on it.
    ///
    /// Renormalised over the legal moves rather than read off the full
    /// vocabulary, so that a band which spends more of its mass on
    /// illegal moves is not charged for that here. The full-vocabulary
    /// figure is a different quantity and `chess_cond` prints it
    /// beside this one in its summary.
    ///
    /// # `None`
    ///
    /// The position could not be scored: the move actually played is
    /// not in the vocabulary, so there is no entry to take a log of.
    /// The same condition that makes `top1` absent, and in `chess_cond`
    /// the same lookup — both come from the played move's index in the
    /// legal list, and the rows this is computed over are built from
    /// that same list, so the index is in range whenever it exists.
    /// That is an argument about the writer and not a property of this
    /// type, which will hold whatever a caller puts in it.
    ///
    /// **Not** a statement about the format. A record written before
    /// version 3 has no `ce` at all and reads back as `None` too, and
    /// the two are indistinguishable here by design: the field says
    /// what the walk found, and a reader that needs to know whether a
    /// walk was in a position to look asks [`WalkHeader::version`] for
    /// `>= 3` instead. Collapsing both readings into one `Option` is
    /// what leaves [`GammaRecord::top2_margin`]'s absence ambiguous;
    /// the ambiguity is settled here by moving the format question out
    /// of the field rather than by giving the field a third state.
    ///
    /// # Why it is here
    ///
    /// The next experiment's power calculation needs the per-game
    /// variance of this quantity, and no walk has ever recorded it —
    /// the fields beside it say where the mass went and how the moves
    /// ranked, not what the human's move cost. Nothing reads it yet.
    #[serde(default)]
    pub ce: Option<Vec<f64>>,
    /// The reference band's top-two margin among legal moves, at this
    /// gamma.
    ///
    /// Take the first band's distribution, restrict it to the legal
    /// moves, renormalise it to sum to one, and subtract the second
    /// largest entry from the largest. [`top2_margin`] is the
    /// computation, and its edge case is documented there.
    ///
    /// The first band because that is the band [`GammaRecord::flipped`]
    /// is defined against — a flip is any band's top legal move
    /// differing from the *first* band's — so this says how contestable
    /// the reference ranking was.
    ///
    /// Not a threshold anything crosses. The band that flips has its own
    /// guided distribution and its argmax is taken there, not against
    /// this number; a wide gap here can still be flipped by a band that
    /// ranks the position differently throughout. What it measures is
    /// how near the reference came to ranking differently by itself,
    /// which is the quantity that makes flip rates comparable or not.
    ///
    /// Renormalised because the raw distribution puts most of its mass
    /// on illegal moves and that share varies by arm; a margin read off
    /// the raw distribution would carry the legal mass with it.
    ///
    /// Per gamma because guidance changes the distribution, so how
    /// contestable the ranking is, is a per-gamma quantity like the
    /// others.
    ///
    /// # Why it is here
    ///
    /// Whether a flip happens depends on how close the top two legal
    /// moves already were, so a more confident model flips less for the
    /// same conditioning strength. Within one experiment that is
    /// harmless — the arms share a training setup and therefore a
    /// confidence regime, and the comparison between them is fair. It
    /// stops being harmless the moment two *formulations* are compared,
    /// a legal mask on or off or legality supplied as an input, because
    /// those change how sharp the distribution is and flip rate would
    /// then be measuring sharpness as much as steerability. This is
    /// recorded so that the comparison can be made when it arrives;
    /// nothing reads it yet.
    ///
    /// `None` on a record written at format version 1, before the field
    /// existed: a version 1 file is still read
    /// ([`MIN_READABLE_VERSION`]), and it has nothing to say here.
    ///
    /// That is why the field is optional, and no writer of a real walk
    /// produces `None` at version 2 — `chess_cond` is the only one, and
    /// it always writes `Some`. It is not an invariant of the type
    /// though: test fixtures in [`crate::chess::steerability`] build
    /// version 2 records with `None`, because nothing reads the field
    /// and filling it would be inventing a number to satisfy a shape.
    /// A reader that starts using it has to decide what `None` means to
    /// it rather than assume the version tells it.
    ///
    /// Whether a walk was **in a position** to write the field is the
    /// one part of that a reader can settle without guessing:
    /// [`WalkHeader::version`] of `>= 2` says the format had it. The
    /// same route works for the fields added since, and is the only
    /// route [`GammaRecord::ce`] offers.
    #[serde(default)]
    pub top2_margin: Option<f64>,
}

/// Cross-entropy at the move that was played, over the legal moves.
///
/// `legal_logits` holds one logit per legal move, in the order the
/// legal moves are listed, and `played` is the played move's index in
/// that list. The result is the negative natural log of the
/// probability that move receives once the distribution is restricted
/// to those moves and renormalised: the log-sum-exp of the row, less
/// the played move's own logit.
///
/// Shift-invariant, because it is a renormalised quantity — adding a
/// constant to every legal logit leaves the answer alone. A row of
/// equal logits returns `ln(n)`, the cost of a uniform draw over `n`
/// moves, which is the bar `chess_cond` reads its loss decomposition
/// against.
///
/// # Why from logits rather than from the renormalised probabilities
///
/// They are the same number, until an entry underflows. The walk's
/// distributions are `f32` softmaxes, and a legal move far enough
/// below the row's maximum lands on exactly zero there; `-ln(0)` is an
/// infinity, `serde_json` writes a non-finite float as `null`, and a
/// `Vec<f64>` does not read `null` back — so a walk that recorded one
/// would write a file its own reader refuses. In log space the same
/// position reads as a large finite number, which is what it is.
///
/// # `None`
///
/// `played` is past the end of the row. A caller whose index came from
/// searching the same list the row was built from will not reach it.
/// One that pairs a row with another position's index reaches it only
/// when that position had more legal moves; otherwise the index lands
/// inside the row and this returns a number for the wrong move. The
/// check is a bound, not an agreement between the two.
pub fn ce_over_legal(legal_logits: &[f32], played: usize) -> Option<f64> {
    let target = f64::from(*legal_logits.get(played)?);
    // Subtracting the maximum before exponentiating, for the reason
    // any softmax does: a row of large logits overflows otherwise, and
    // the shift cancels in the result.
    let max = f64::from(
        legal_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    );
    let total: f64 = legal_logits
        .iter()
        .map(|l| (f64::from(*l) - max).exp())
        .sum();
    Some(max + total.ln() - target)
}

/// Largest entry of a legal-move distribution, less the second largest.
///
/// `over_legal` is expected to be renormalised over the legal moves
/// already — the same array the top move is chosen from, so that the
/// margin and the flip it belongs to cannot come from two different
/// distributions.
///
/// # Fewer than two entries
///
/// A distribution over one move has no second largest. Returns `1.0`
/// there: the decision is maximally forced and no flip is possible at
/// it. `None` is the other option and is worse — a forced position
/// excluded from the margin while still counting in the flip-rate
/// denominator would make the two quantities describe different
/// samples.
///
/// The empty slice takes the same branch. It is not a distribution and
/// has no margin at all; `1.0` is returned rather than a panic because
/// this function is total, not because the value means anything there.
///
/// Neither case arises from the walk that feeds this. `chess_cond`
/// admits a position only when it offers at least two legal moves that
/// are in the vocabulary (`if legal.len() >= 2`), so both are filtered
/// out before any distribution is built.
pub fn top2_margin(over_legal: &[f32]) -> f64 {
    let mut sorted = over_legal.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    match (sorted.first(), sorted.get(1)) {
        (Some(first), Some(second)) => (first - second) as f64,
        _ => 1.0,
    }
}

/// Just enough of a header to decide whether this build can read the
/// rest of it.
///
/// Unknown fields are allowed here and denied on [`WalkHeader`], which
/// is the whole point: a file from a later format has to be able to
/// announce its version before this build objects to anything else
/// about it.
#[derive(Debug, Deserialize)]
struct VersionProbe {
    version: u32,
}

/// One walk: its header and every position it saw.
#[derive(Debug, Clone, PartialEq)]
pub struct Walk {
    /// What the walk was.
    pub header: WalkHeader,
    /// What it saw, in visit order.
    pub records: Vec<PositionRecord>,
}

/// Why a record file could not be written or read back.
#[derive(Debug, Error)]
pub enum RecordError {
    /// The file could not be opened, written or read.
    #[error("record file {path}: {message}")]
    Io {
        /// Path involved.
        path: String,
        /// Underlying IO message.
        message: String,
    },

    /// A line is not the JSON this format expects.
    #[error("record file {path}, line {line}: {message}")]
    Parse {
        /// Path involved.
        path: String,
        /// One-based line number.
        line: usize,
        /// Underlying serde message.
        message: String,
    },

    /// The file holds nothing, so not even the header is there.
    #[error("record file {path} is empty; a walk writes a header even when it saw no positions")]
    Empty {
        /// Path involved.
        path: String,
    },

    /// The header names a version outside the range this build reads.
    #[error(
        "record file {path} is format version {found}, and this build reads versions \
         {readable_from} through {expected}; outside that range a field may have changed \
         meaning, so it is refused rather than read as though it had not"
    )]
    Version {
        /// Path involved.
        path: String,
        /// Newest version this build reads, which is also the one it
        /// writes ([`FORMAT_VERSION`]).
        expected: u32,
        /// Oldest version this build reads ([`MIN_READABLE_VERSION`]).
        readable_from: u32,
        /// Version the file declares.
        found: u32,
    },

    /// The header's position count and the number of record lines
    /// disagree, so the file is not the whole walk.
    #[error(
        "record file {path} declares {declared} position(s) and carries {found}; \
         the run that wrote it did not finish"
    )]
    Truncated {
        /// Path involved.
        path: String,
        /// What the header claims.
        declared: usize,
        /// What the file holds.
        found: usize,
    },

    /// A record carries a different number of gammas than the header
    /// swept, so the two cannot be zipped.
    #[error(
        "record file {path}, position {index}: the header swept {expected} gamma(s) and this \
         record carries {found}"
    )]
    GammaCount {
        /// Path involved.
        path: String,
        /// Zero-based position index.
        index: usize,
        /// Gammas the header declares.
        expected: usize,
        /// Gammas the record carries.
        found: usize,
    },
}

impl Walk {
    /// Write the walk as JSON Lines.
    ///
    /// # Errors
    ///
    /// The file could not be created or written, or a record could not
    /// be serialised.
    pub fn write_jsonl(&self, path: &Path) -> Result<(), RecordError> {
        let io = |e: std::io::Error| RecordError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        };
        let file = File::create(path).map_err(io)?;
        let mut out = BufWriter::new(file);
        let header = serde_json::to_string(&self.header).map_err(|e| RecordError::Parse {
            path: path.display().to_string(),
            line: 1,
            message: e.to_string(),
        })?;
        writeln!(out, "{header}").map_err(io)?;
        for (index, record) in self.records.iter().enumerate() {
            let line = serde_json::to_string(record).map_err(|e| RecordError::Parse {
                path: path.display().to_string(),
                line: index + 2,
                message: e.to_string(),
            })?;
            writeln!(out, "{line}").map_err(io)?;
        }
        // Flushed explicitly: a `BufWriter` dropped with buffered bytes
        // discards the write error, which would leave a short file
        // reported as a complete one.
        out.flush().map_err(io)?;
        Ok(())
    }

    /// Read a walk back.
    ///
    /// # Errors
    ///
    /// Any of [`RecordError`]. In particular the file is refused when
    /// the header's counts do not match its contents, rather than being
    /// read as a shorter walk.
    pub fn read_jsonl(path: &Path) -> Result<Self, RecordError> {
        let display = path.display().to_string();
        let file = File::open(path).map_err(|e| RecordError::Io {
            path: display.clone(),
            message: e.to_string(),
        })?;
        let mut lines = BufReader::new(file).lines();

        let first = match lines.next() {
            Some(line) => line.map_err(|e| RecordError::Io {
                path: display.clone(),
                message: e.to_string(),
            })?,
            None => return Err(RecordError::Empty { path: display }),
        };
        // The version is read on its own first, from a probe that allows
        // unknown fields. `WalkHeader` denies them, so a file from a
        // later version that merely *added* one to the header would fail
        // as an unreadable line with a serde message and never reach the
        // clear refusal that exists for exactly that case: the field
        // meant to catch a future format would be defeated by the
        // strictness sitting beside it.
        let probe: VersionProbe = serde_json::from_str(&first).map_err(|e| RecordError::Parse {
            path: display.clone(),
            line: 1,
            message: e.to_string(),
        })?;
        // A range rather than an equality, so that the walks written at
        // version 1 stay readable. What makes that sound is that every
        // version in the range is this shape with later fields absent,
        // and `MIN_READABLE_VERSION` documents whose job it is to keep
        // that true.
        if probe.version < MIN_READABLE_VERSION || probe.version > FORMAT_VERSION {
            return Err(RecordError::Version {
                path: display,
                expected: FORMAT_VERSION,
                readable_from: MIN_READABLE_VERSION,
                found: probe.version,
            });
        }
        let header: WalkHeader = serde_json::from_str(&first).map_err(|e| RecordError::Parse {
            path: display.clone(),
            line: 1,
            message: e.to_string(),
        })?;

        let mut records = Vec::with_capacity(header.positions);
        for (offset, line) in lines.enumerate() {
            let line = line.map_err(|e| RecordError::Io {
                path: display.clone(),
                message: e.to_string(),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record: PositionRecord =
                serde_json::from_str(&line).map_err(|e| RecordError::Parse {
                    path: display.clone(),
                    line: offset + 2,
                    message: e.to_string(),
                })?;
            if record.at.len() != header.gammas.len() {
                return Err(RecordError::GammaCount {
                    path: display,
                    index: offset,
                    expected: header.gammas.len(),
                    found: record.at.len(),
                });
            }
            records.push(record);
        }
        if records.len() != header.positions {
            return Err(RecordError::Truncated {
                path: display,
                declared: header.positions,
                found: records.len(),
            });
        }
        Ok(Self { header, records })
    }

    /// Index of `gamma` in this walk's sweep.
    ///
    /// Compared by bit pattern rather than by tolerance: the value came
    /// out of the same parse of the same string on both sides, so an
    /// exact match is the honest test and a tolerance would let 1.0 and
    /// 1.0000001 be treated as the same sweep entry.
    pub fn gamma_index(&self, gamma: f32) -> Option<usize> {
        self.header.gammas.iter().position(|g| *g == gamma)
    }
}

/// Why several walks could not be read as one position stream.
#[derive(Debug, Error, PartialEq)]
pub enum AlignError {
    /// Fewer arms than the comparison needs.
    #[error("comparing arms needs at least {needed}, and {found} were given")]
    TooFewArms {
        /// How many the caller supplied.
        found: usize,
        /// How many are required.
        needed: usize,
    },

    /// Two arms carry the same name, so a role lookup would be
    /// ambiguous.
    #[error("two arms are both named {name:?}")]
    DuplicateArm {
        /// The repeated name.
        name: String,
    },

    /// The arms swept different band lists, so their flip rates are not
    /// the same quantity: the rate counts a disagreement with the
    /// *first* band across however many bands there are.
    #[error(
        "arm {arm:?} carries bands {found:?} and arm {reference:?} carries {expected:?}; \
         flip rate is defined against the first band over all of them, so these are not \
         comparable numbers"
    )]
    BandsDiffer {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Bands the reference carries.
        expected: Vec<String>,
        /// Bands this arm carries.
        found: Vec<String>,
    },

    /// The arms swept different guidance strengths.
    #[error("arm {arm:?} swept gammas {found:?} and arm {reference:?} swept {expected:?}")]
    GammasDiffer {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Gammas the reference swept.
        expected: Vec<f32>,
        /// Gammas this arm swept.
        found: Vec<f32>,
    },

    /// The arms were cut to different context windows, so the depth
    /// buckets do not line up.
    #[error(
        "arm {arm:?} was walked at ctx {found} and arm {reference:?} at {expected}; \
         the deep bucket ends at ctx - 2, so these arms bucket depth differently"
    )]
    CtxDiffers {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Context window of the reference.
        expected: usize,
        /// Context window of this arm.
        found: usize,
    },

    /// The arms hold different numbers of positions.
    #[error(
        "arm {arm:?} holds {found} position(s) and arm {reference:?} holds {expected}; \
         these arms were not scored on the same walk"
    )]
    PositionCountDiffers {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Positions in the reference.
        expected: usize,
        /// Positions in this arm.
        found: usize,
    },

    /// The arms hold the same number of positions, and they are not the
    /// same positions.
    #[error(
        "arm {arm:?} has game {found_game} ply {found_ply} at index {index}, where arm \
         {reference:?} has game {expected_game} ply {expected_ply}; joining these would \
         subtract two arms' numbers at positions that are not the same position"
    )]
    PositionDiffers {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Where in the stream they parted.
        index: usize,
        /// Game the reference has there.
        expected_game: usize,
        /// Ply the reference has there.
        expected_ply: usize,
        /// Game this arm has there.
        found_game: usize,
        /// Ply this arm has there.
        found_ply: usize,
    },

    /// A named arm is not among those supplied.
    #[error("no arm named {name:?}; the arms given are {available:?}")]
    UnknownArm {
        /// Name that was asked for.
        name: String,
        /// Names that are present.
        available: Vec<String>,
    },

    /// Two arms were scored from the same checkpoint.
    ///
    /// Every statistic downstream still comes out well-formed, which is
    /// the problem. Two arms of one checkpoint have identical records,
    /// so their difference is identically zero **in every bootstrap
    /// draw** — and if the pair is the two per-position runs, the
    /// same-arm gap that the two-arm margin has to beat vanishes and the
    /// confirm criterion silently becomes an interval on the margin
    /// alone. The floor is the one thing the plan calls undroppable, and
    /// its only visible trace would be a printed `G = 0.000000`.
    #[error(
        "arms {a:?} and {b:?} were both scored from checkpoint {ckpt}; their records are \
         identical, so every difference between them is zero in every draw — and if these are \
         the two runs whose gap is the floor, the floor has silently gone to zero"
    )]
    SameCheckpoint {
        /// First arm sharing the checkpoint.
        a: String,
        /// Second arm sharing the checkpoint.
        b: String,
        /// The checkpoint both name.
        ckpt: String,
    },

    /// Two arms produced byte-for-byte the same records.
    ///
    /// The check [`AlignError::SameCheckpoint`] should have made and
    /// cannot: that one compares the path a caller typed, and the same
    /// weights reach two arms under two spellings all the time — a
    /// relative path against an absolute one, a symlink, or a
    /// `perpos-b` directory that is a copy of `perpos` because an `scp`
    /// went wrong. Distinct strings, identical encoding, identical
    /// positions, and a difference of zero in every draw.
    ///
    /// Identical weights produce identical records, so this compares the
    /// thing that matters rather than a name for it. The comparison is
    /// by value; a record holding a `NaN` would compare unequal to
    /// itself and slip through, but a `NaN` in a divergence or a legal
    /// mass is a defect that shows up long before this.
    ///
    /// This is the failure that already ran undetected once: the same
    /// checkpoint was passed as two arms, the same-arm gap came out
    /// identically zero, and the confirm criterion `M - G` silently
    /// became an interval on `M` alone — the floor gone, with the output
    /// still complete and the refute criterion still behaving.
    #[error(
        "arms {a:?} and {b:?} produced identical records over all {positions} position(s), so \
         they were scored from the same weights however they were named; every difference \
         between them is zero in every draw, and if these are the two runs whose gap is the \
         floor, the floor has silently gone to zero"
    )]
    IdenticalRecords {
        /// First arm of the pair.
        a: String,
        /// Second arm of the pair.
        b: String,
        /// Positions they agree on.
        positions: usize,
    },

    /// The arms were not walked over the same held-out material.
    ///
    /// Distinct from [`AlignError::PositionDiffers`], which catches the
    /// same thing later and less clearly: a different month usually
    /// yields a different `(game, ply)` sequence, but nothing guarantees
    /// it does, and "these are two months" is a better message than
    /// "position 1,417 disagrees".
    #[error(
        "arm {arm:?} was walked over {field} {found:?} and arm {reference:?} over {expected:?}; \
         a difference between arms is only a difference if both were scored on the same material"
    )]
    HoldoutDiffers {
        /// Arm that disagreed.
        arm: String,
        /// Arm it was compared against.
        reference: String,
        /// Which header field differed: `holdout` or `side`.
        field: &'static str,
        /// What the reference carries.
        expected: String,
        /// What this arm carries.
        found: String,
    },

    /// The gamma asked for is not one the arms swept.
    ///
    /// Its own variant because gamma is a command-line argument and a
    /// typo is a likely operator error; reporting it as a disagreement
    /// between two arms — one of them fabricated from the argument —
    /// sends the reader looking for a mismatch between files that agree.
    #[error("gamma {gamma} was not swept; these arms carry {swept:?}")]
    GammaNotSwept {
        /// Gamma the caller asked for.
        gamma: f32,
        /// Gammas the walk actually swept.
        swept: Vec<f32>,
    },

    /// The walks carry no positions at all.
    #[error("the arms carry no positions, so there is nothing to compare")]
    NoPositions,
}

/// Several arms, checked to describe one position stream.
///
/// Construction is the check. Everything downstream can then index the
/// arms by position without asking again whether index *i* means the
/// same position in each of them.
#[derive(Debug, Clone)]
pub struct AlignedArms {
    names: Vec<String>,
    walks: Vec<Walk>,
    /// Dense cluster index per position, shared by every arm.
    clusters: Vec<usize>,
    /// How many distinct games contributed a position.
    games: usize,
}

impl AlignedArms {
    /// Check that `named` arms describe the same positions, and keep
    /// them.
    ///
    /// The first arm is the reference every other is compared against;
    /// which one holds that role does not affect the verdict, only which
    /// side of a disagreement the message calls expected.
    ///
    /// # Errors
    ///
    /// Any of [`AlignError`]. Nothing is joined partially: a
    /// disagreement at position 2,999 of 3,000 refuses the whole set.
    pub fn new(named: Vec<(String, Walk)>) -> Result<Self, AlignError> {
        if named.is_empty() {
            return Err(AlignError::TooFewArms {
                found: 0,
                needed: 1,
            });
        }
        let mut names: Vec<String> = Vec::with_capacity(named.len());
        let mut walks: Vec<Walk> = Vec::with_capacity(named.len());
        for (name, walk) in named {
            if names.contains(&name) {
                return Err(AlignError::DuplicateArm { name });
            }
            names.push(name);
            walks.push(walk);
        }

        // `names` and `walks` are pushed in lockstep above and both are
        // non-empty, so these two cannot miss; the `ok_or` keeps the
        // function free of indexing panics regardless.
        let reference_name = names.first().cloned().ok_or(AlignError::TooFewArms {
            found: 0,
            needed: 1,
        })?;
        let reference = walks.first().ok_or(AlignError::TooFewArms {
            found: 0,
            needed: 1,
        })?;

        // Provenance first, so that a whole-file disagreement is
        // reported as one rather than as a mismatch at some position
        // deep inside it.
        for (i, (name_a, walk_a)) in names.iter().zip(walks.iter()).enumerate() {
            for (name_b, walk_b) in names.iter().zip(walks.iter()).skip(i + 1) {
                if walk_a.header.ckpt == walk_b.header.ckpt {
                    return Err(AlignError::SameCheckpoint {
                        a: name_a.clone(),
                        b: name_b.clone(),
                        ckpt: walk_a.header.ckpt.clone(),
                    });
                }
            }
        }

        for (name, walk) in names.iter().zip(walks.iter()).skip(1) {
            for (field, expected, found) in [
                ("holdout", &reference.header.holdout, &walk.header.holdout),
                ("side", &reference.header.side, &walk.header.side),
            ] {
                if expected != found {
                    return Err(AlignError::HoldoutDiffers {
                        arm: name.clone(),
                        reference: reference_name.clone(),
                        field,
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
            }
            if walk.header.bands != reference.header.bands {
                return Err(AlignError::BandsDiffer {
                    arm: name.clone(),
                    reference: reference_name.clone(),
                    expected: reference.header.bands.clone(),
                    found: walk.header.bands.clone(),
                });
            }
            if walk.header.gammas != reference.header.gammas {
                return Err(AlignError::GammasDiffer {
                    arm: name.clone(),
                    reference: reference_name.clone(),
                    expected: reference.header.gammas.clone(),
                    found: walk.header.gammas.clone(),
                });
            }
            if walk.header.ctx != reference.header.ctx {
                return Err(AlignError::CtxDiffers {
                    arm: name.clone(),
                    reference: reference_name.clone(),
                    expected: reference.header.ctx,
                    found: walk.header.ctx,
                });
            }
            if walk.records.len() != reference.records.len() {
                return Err(AlignError::PositionCountDiffers {
                    arm: name.clone(),
                    reference: reference_name.clone(),
                    expected: reference.records.len(),
                    found: walk.records.len(),
                });
            }
            for (index, (want, got)) in reference
                .records
                .iter()
                .zip(walk.records.iter())
                .enumerate()
            {
                if want.game != got.game || want.ply != got.ply {
                    return Err(AlignError::PositionDiffers {
                        arm: name.clone(),
                        reference: reference_name.clone(),
                        index,
                        expected_game: want.game,
                        expected_ply: want.ply,
                        found_game: got.game,
                        found_ply: got.ply,
                    });
                }
            }
        }

        if reference.records.is_empty() {
            return Err(AlignError::NoPositions);
        }

        // After the sequence check, because until it has passed a
        // difference between two arms' records could be a difference of
        // position sets rather than of weights, and the message for that
        // is the other one. By here every arm holds the same positions
        // in the same order, so records that still agree everywhere came
        // from the same model.
        for (i, (name_a, walk_a)) in names.iter().zip(walks.iter()).enumerate() {
            for (name_b, walk_b) in names.iter().zip(walks.iter()).skip(i + 1) {
                if walk_a.records == walk_b.records {
                    return Err(AlignError::IdenticalRecords {
                        a: name_a.clone(),
                        b: name_b.clone(),
                        positions: walk_a.records.len(),
                    });
                }
            }
        }

        // Game ids are the walk's own numbering, which skips any game
        // that was accepted and then contributed nothing. The bootstrap
        // resamples `0..games`, so they are renumbered densely here —
        // over the games that actually carry positions, which is the
        // cluster set `§5.2` describes.
        let mut dense: BTreeMap<usize, usize> = BTreeMap::new();
        let mut clusters = Vec::with_capacity(reference.records.len());
        for record in &reference.records {
            let next = dense.len();
            let ix = *dense.entry(record.game).or_insert(next);
            clusters.push(ix);
        }
        let games = dense.len();

        Ok(Self {
            names,
            walks,
            clusters,
            games,
        })
    }

    /// Read each `(name, path)` and align what comes back.
    ///
    /// # Errors
    ///
    /// A file could not be read ([`RecordError`]), or the arms do not
    /// describe one position stream ([`AlignError`]).
    pub fn read(named: &[(String, PathBuf)]) -> Result<Self, ArmsError> {
        let mut walks = Vec::with_capacity(named.len());
        for (name, path) in named {
            walks.push((name.clone(), Walk::read_jsonl(path)?));
        }
        Ok(Self::new(walks)?)
    }

    /// Arm names, in the order they were supplied.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The walk an arm holds.
    ///
    /// # Errors
    ///
    /// No arm carries that name.
    pub fn walk(&self, name: &str) -> Result<&Walk, AlignError> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|ix| self.walks.get(ix))
            .ok_or_else(|| AlignError::UnknownArm {
                name: name.to_string(),
                available: self.names.clone(),
            })
    }

    /// Positions in the shared stream.
    pub fn positions(&self) -> usize {
        self.clusters.len()
    }

    /// Games that contributed at least one position — the cluster count
    /// every bootstrap here resamples over.
    pub fn games(&self) -> usize {
        self.games
    }

    /// Cluster index of each position, in stream order.
    pub fn clusters(&self) -> &[usize] {
        &self.clusters
    }

    /// Context window the arms were walked at.
    ///
    /// Equal across arms by construction; [`AlignedArms::new`] refuses
    /// the set otherwise.
    pub fn ctx(&self) -> usize {
        self.walks.first().map(|w| w.header.ctx).unwrap_or_default()
    }

    /// Index of `gamma` in the shared sweep.
    ///
    /// # Errors
    ///
    /// No arm swept that gamma.
    pub fn gamma_index(&self, gamma: f32) -> Result<usize, AlignError> {
        let reference = self.walks.first().ok_or(AlignError::TooFewArms {
            found: 0,
            needed: 1,
        })?;
        reference
            .gamma_index(gamma)
            .ok_or_else(|| AlignError::GammaNotSwept {
                gamma,
                swept: reference.header.gammas.clone(),
            })
    }
}

/// Either half of "read these files and line them up".
#[derive(Debug, Error)]
pub enum ArmsError {
    /// A file could not be read.
    #[error(transparent)]
    Record(#[from] RecordError),
    /// The files do not describe one position stream.
    #[error(transparent)]
    Align(#[from] AlignError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    pub(super) fn header(positions: usize, games: usize) -> WalkHeader {
        WalkHeader {
            version: FORMAT_VERSION,
            ckpt: "/root/ckpt/perpos/run.safetensors".into(),
            holdout: "holdout-2026-05.pgn".into(),
            side: "White".into(),
            encoding: CondEncoding::Prefix,
            legal_input: false,
            ctx: 128,
            bands: vec![
                "<elo:1100-1299>".into(),
                "<elo:1500-1699>".into(),
                "<elo:1900-2099>".into(),
            ],
            gammas: vec![1.0, 4.0],
            positions,
            games,
            games_of: Some(vec!["<elo:1100-1299>".into()]),
        }
    }

    pub(super) fn gamma_record(flipped: bool, js: f64) -> GammaRecord {
        GammaRecord {
            flipped,
            widest_js: js,
            legal_mass: 0.5,
            top1: Some(vec![flipped, false, true]),
            ce: Some(vec![1.5, 2.0, 2.5]),
            top2_margin: Some(0.125),
        }
    }

    /// Label a walk as an arm, giving it a checkpoint of its own.
    ///
    /// Every arm needs a distinct one: two arms scored from the same
    /// checkpoint have identical records, so every difference between
    /// them is zero in every draw, and `AlignedArms` refuses the pair
    /// rather than let a floor collapse quietly.
    pub(super) fn arm(name: &str, mut walk: Walk) -> (String, Walk) {
        walk.header.ckpt = format!("/root/ckpt/{name}/run.safetensors");
        (name.to_string(), walk)
    }

    /// A walk of `games` games, each holding `per_game` positions, with
    /// every gamma entry the same.
    pub(super) fn walk(games: usize, per_game: usize, flipped: bool) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ply in 0..per_game {
                records.push(PositionRecord {
                    game,
                    ply: ply * 2,
                    n_legal: Some(31),
                    at: vec![gamma_record(flipped, 0.02), gamma_record(flipped, 0.05)],
                });
            }
        }
        Walk {
            header: header(records.len(), games),
            records,
        }
    }

    #[test]
    fn a_walk_survives_a_round_trip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("perpos.jsonl");
        let original = walk(3, 4, true);
        original.write_jsonl(&path).unwrap();
        let back = Walk::read_jsonl(&path).unwrap();
        assert_eq!(back, original);
    }

    /// One line per position, plus the header, so the file can be
    /// counted and grepped.
    #[test]
    fn the_file_is_one_line_per_position_behind_a_header() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("perpos.jsonl");
        let original = walk(2, 5, false);
        original.write_jsonl(&path).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body.lines().count(), 1 + 10);
    }

    #[test]
    fn a_file_from_another_format_version_is_refused() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("old.jsonl");
        let mut original = walk(1, 1, false);
        original.header.version = FORMAT_VERSION + 1;
        original.write_jsonl(&path).unwrap();
        assert!(matches!(
            Walk::read_jsonl(&path),
            Err(RecordError::Version { .. })
        ));
    }

    /// And the other end of the range, which the test above cannot
    /// reach: a version below the floor is refused just as a version
    /// above the ceiling is. The range is bounded on both sides rather
    /// than being "anything up to what I write", which is what will
    /// matter the day [`MIN_READABLE_VERSION`] is raised — that is the
    /// moment files which used to read have to stop reading.
    #[test]
    fn a_version_below_the_floor_is_refused() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v0.jsonl");
        let mut original = walk(1, 1, false);
        original.header.version = MIN_READABLE_VERSION - 1;
        original.write_jsonl(&path).unwrap();
        let err = Walk::read_jsonl(&path);
        assert!(
            matches!(
                err,
                Err(RecordError::Version {
                    found: 0,
                    readable_from: MIN_READABLE_VERSION,
                    expected: FORMAT_VERSION,
                    ..
                })
            ),
            "{err:?}"
        );
    }

    /// The whole reason the version check became a range: six walks
    /// written at version 1 are the evidence for a confirmed result, and
    /// a bump that made them unreadable would throw it away. A version 1
    /// record carries none of the fields added since and reads them back
    /// as `None` — the absence it actually is, not a fabricated number.
    ///
    /// Written as raw lines rather than through `write_jsonl`, because
    /// this build cannot serialise a record without those fields.
    #[test]
    fn a_version_1_file_still_reads_with_the_later_fields_absent() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v1.jsonl");
        let header = r#"{"version": 1, "ckpt": "c", "holdout": "h", "side": "White", "encoding": "prefix", "ctx": 128, "bands": ["<lo>", "<hi>"], "gammas": [1.0], "positions": 1, "games": 1}"#;
        let record = r#"{"game": 0, "ply": 4, "at": [{"flipped": true, "widest_js": 0.02, "legal_mass": 0.9, "top1": [true, false]}]}"#;
        std::fs::write(&path, format!("{header}\n{record}\n")).unwrap();
        let walk = Walk::read_jsonl(&path).expect("a version 1 walk is still readable");
        assert_eq!(walk.header.version, 1);
        assert_eq!(walk.records.len(), 1);
        assert!(walk.records[0].at[0].flipped);
        assert_eq!(walk.records[0].at[0].top2_margin, None);
        assert_eq!(walk.records[0].at[0].ce, None);
        assert_eq!(walk.records[0].n_legal, None);
        // The header's version 4 field, which defaults to a value
        // rather than to an absence. `false` is what this walk means:
        // no reader would open a legality checkpoint when it was
        // written.
        assert!(!walk.header.legal_input);
    }

    /// The version 4 field, through the file and back, asserted on the
    /// text as well as on the parse so that a field which round-tripped
    /// only because both ends dropped it would still fail here.
    #[test]
    fn a_current_header_round_trips_its_legality_axis() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v4.jsonl");
        let mut original = walk(1, 1, true);
        original.header.legal_input = true;
        original.write_jsonl(&path).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"legal_input\":true"), "{body}");
        let back = Walk::read_jsonl(&path).unwrap();
        assert_eq!(back.header.version, FORMAT_VERSION);
        assert!(back.header.legal_input);
    }

    /// Two walks that differ only on the legality axis are two
    /// different headers, which is the whole point of the field: before
    /// version 4 these two parsed into the same `WalkHeader`.
    #[test]
    fn the_legality_axis_makes_two_otherwise_identical_headers_differ() {
        let plain = header(0, 0);
        let mut legal = header(0, 0);
        legal.legal_input = true;
        assert_ne!(plain, legal);
        // And on the same axis the encoding already occupied, so the
        // two are recorded independently rather than one standing in
        // for the other.
        assert_eq!(plain.encoding, legal.encoding);
    }

    /// The gap the field's doc names, as behaviour: a header written
    /// before the field existed reads back `false` whatever its
    /// checkpoint was. Recorded here so that the claim
    /// `steerability::check_legality_roles` is built on — that such a
    /// walk arrives as `false` and has to be refused rather than
    /// believed — rests on a test and not on a paragraph.
    #[test]
    fn a_header_written_before_the_field_reads_back_false() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v3.jsonl");
        let header = r#"{"version": 3, "ckpt": "c", "holdout": "h", "side": "White", "encoding": "prefix", "ctx": 128, "bands": ["<lo>", "<hi>"], "gammas": [1.0], "positions": 0, "games": 0}"#;
        std::fs::write(&path, format!("{header}\n")).unwrap();
        let walk = Walk::read_jsonl(&path).expect("a version 3 walk is still readable");
        assert_eq!(walk.header.version, 3);
        assert!(!walk.header.legal_input);
        // The version 5 field reads back as the absence it is on a
        // pre-5 file, not as an empty filter — the two are different
        // facts, and `games_of`'s doc leans on this distinction.
        assert_eq!(walk.header.games_of, None);
    }

    /// The two statements a version 5 header can make about its games
    /// — narrowed to these tokens, or deliberately left open — both
    /// survive the file, and neither collapses into the pre-5 `None`.
    #[test]
    fn the_walked_games_statement_round_trips_including_deliberately_open() {
        let tmp = TempDir::new().unwrap();
        for games_of in [
            Some(vec!["<eco:B>".to_string(), "<elo:1100-1299>".to_string()]),
            Some(vec![]),
        ] {
            let path = tmp.path().join("walk.jsonl");
            let mut h = header(0, 0);
            h.games_of = games_of.clone();
            Walk {
                header: h,
                records: vec![],
            }
            .write_jsonl(&path)
            .unwrap();
            let back = Walk::read_jsonl(&path).unwrap();
            assert_eq!(back.header.games_of, games_of);
        }
    }

    /// And a record written now carries the margin through the file and
    /// back, rather than the field existing only in memory.
    #[test]
    fn a_current_record_round_trips_its_margin() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v2.jsonl");
        let mut original = walk(1, 1, true);
        original.records[0].at[0].top2_margin = Some(0.375);
        original.write_jsonl(&path).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"top2_margin\":0.375"), "{body}");
        let back = Walk::read_jsonl(&path).unwrap();
        assert_eq!(back.header.version, FORMAT_VERSION);
        assert_eq!(back.records[0].at[0].top2_margin, Some(0.375));
    }

    /// The version 3 pair, through the file and back. Asserted on the
    /// text as well as on the parse, so that a field which round-tripped
    /// only because both ends dropped it would still fail here.
    #[test]
    fn a_current_record_round_trips_its_cost_and_its_legal_count() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v3.jsonl");
        let mut original = walk(1, 1, true);
        original.records[0].n_legal = Some(29);
        original.records[0].at[0].ce = Some(vec![0.5, 1.25, 2.0]);
        original.write_jsonl(&path).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"n_legal\":29"), "{body}");
        assert!(body.contains("\"ce\":[0.5,1.25,2.0]"), "{body}");
        let back = Walk::read_jsonl(&path).unwrap();
        assert_eq!(back.header.version, FORMAT_VERSION);
        assert_eq!(back.records[0].n_legal, Some(29));
        assert_eq!(back.records[0].at[0].ce, Some(vec![0.5, 1.25, 2.0]));
    }

    /// A position the walk could not score carries neither per-band
    /// figure: the played move is not in the vocabulary, so there is no
    /// top move to match it against and no probability to take a log of.
    /// Both absences survive the round trip, rather than one of them
    /// coming back as a number.
    #[test]
    fn an_unscoreable_position_keeps_both_of_its_absences() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("unscoreable.jsonl");
        let mut original = walk(1, 1, true);
        original.records[0].at[0].top1 = None;
        original.records[0].at[0].ce = None;
        original.write_jsonl(&path).unwrap();
        let back = Walk::read_jsonl(&path).unwrap();
        assert_eq!(back.records[0].at[0].top1, None);
        assert_eq!(back.records[0].at[0].ce, None);
    }

    /// What the doc says, on a distribution whose answer is arithmetic.
    /// Two legal moves at logits `0` and `ln 3` are a 1:3 split, so the
    /// first costs `ln 4` and the second `ln(4/3)`.
    #[test]
    fn the_cost_is_the_negative_log_of_the_renormalised_probability() {
        // Spelled as `ln 3` rather than as 1.0986, so that the row and
        // the expected answers below are the same arithmetic written
        // twice rather than a constant and a hope.
        let row = [0.0f32, 3.0f32.ln()];
        let want_first = 4.0f64.ln();
        let want_second = (4.0f64 / 3.0).ln();
        let first = ce_over_legal(&row, 0).expect("index 0 is in the row");
        let second = ce_over_legal(&row, 1).expect("index 1 is in the row");
        assert!((first - want_first).abs() < 1e-6, "{first} vs {want_first}");
        assert!(
            (second - want_second).abs() < 1e-6,
            "{second} vs {want_second}"
        );
    }

    /// It is a renormalised quantity, so only the gaps between the
    /// logits matter: the same row shifted bodily costs the same. The
    /// shift is 800, which is also what makes this the test that the
    /// log-sum-exp subtracts its maximum — `exp(800)` is an infinity in
    /// `f64`, so a row summed without that step returns one too.
    ///
    /// Every logit here is a dyadic rational, so the shifted row is the
    /// unshifted one to the bit. An earlier version of this test used
    /// `ln 3` and shifted by 80, where the `f32` sum rounds the
    /// fraction away and the two rows are no longer the same shape —
    /// it failed by 1.7e-6 on a property that holds.
    #[test]
    fn the_cost_ignores_a_constant_added_to_every_legal_logit() {
        let low = [0.0f32, 0.5, -2.0];
        let high: Vec<f32> = low.iter().map(|l| l + 800.0).collect();
        for played in 0..low.len() {
            let a = ce_over_legal(&low, played).expect("in range");
            let b = ce_over_legal(&high, played).expect("in range");
            assert!(b.is_finite(), "played {played}: {b}");
            assert!((a - b).abs() < 1e-9, "played {played}: {a} vs {b}");
        }
    }

    /// A row of equal logits is a uniform draw, and costs `ln(n)` — the
    /// bar `chess_cond` reads its loss decomposition against, and the
    /// reason `n_legal` is recorded beside the cost.
    #[test]
    fn a_flat_row_costs_the_uniform_draw() {
        for n in 1..6usize {
            let row = vec![0.4f32; n];
            let got = ce_over_legal(&row, 0).expect("in range");
            let want = (n as f64).ln();
            assert!((got - want).abs() < 1e-9, "n = {n}: {got} vs {want}");
        }
    }

    /// Out of range is the one absence the computation itself reports,
    /// and the empty row is the same case rather than a special one.
    #[test]
    fn a_played_index_past_the_row_has_no_cost() {
        assert_eq!(ce_over_legal(&[0.0, 1.0], 2), None);
        assert_eq!(ce_over_legal(&[], 0), None);
    }

    /// Why the cost is computed in log space rather than as `-ln(p)` off
    /// the renormalised probabilities: an underflowed entry would make
    /// it infinite, and this is what an infinite entry does to the file.
    /// `serde_json` writes a non-finite float as `null`, and reading a
    /// `Vec<f64>` back from `null` fails — a walk would write a file its
    /// own reader refuses.
    #[test]
    fn an_infinite_cost_would_write_a_file_the_reader_refuses() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("infinite.jsonl");
        let mut original = walk(1, 1, true);
        original.records[0].at[0].ce = Some(vec![f64::INFINITY, 1.0, 2.0]);
        original.write_jsonl(&path).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"ce\":[null,1.0,2.0]"), "{body}");
        assert!(matches!(
            Walk::read_jsonl(&path),
            Err(RecordError::Parse { line: 2, .. })
        ));
    }

    /// The margin is the first minus the second, over the distribution
    /// as given. Hand-built so the answer is arithmetic rather than
    /// whatever the code happens to produce: the two largest entries are
    /// 0.5 and 0.3, in neither the first nor the last position.
    #[test]
    fn the_margin_is_the_gap_between_the_top_two() {
        let d = [0.1f32, 0.5, 0.05, 0.3, 0.05];
        assert!((top2_margin(&d) - 0.2).abs() < 1e-6, "{}", top2_margin(&d));
    }

    /// A tie at the top is a margin of zero, which is the case the
    /// figure exists to expose: a flip counted there crossed nothing.
    #[test]
    fn a_tie_at_the_top_is_a_margin_of_zero() {
        let d = [0.4f32, 0.4, 0.2];
        assert_eq!(top2_margin(&d), 0.0);
    }

    /// One legal move is a forced decision with no second place. It
    /// reads as `1.0` rather than as absent, so that a forced position
    /// counts in the margin and in the flip-rate denominator alike.
    #[test]
    fn a_single_legal_move_is_a_margin_of_one() {
        // 0.7 rather than 1.0: a lone entry of 1.0 cannot tell the
        // constant apart from an implementation that returned the
        // largest entry, so the assertion would have passed either way
        // and the name would not have been earned.
        assert_eq!(top2_margin(&[0.7f32]), 1.0);
        assert_eq!(top2_margin(&[1.0f32]), 1.0);
        // The empty slice takes the same branch. Documented as total
        // rather than meaningful: the walk cannot produce either.
        assert_eq!(top2_margin(&[]), 1.0);
    }

    /// The case the version probe exists for, which the test above
    /// cannot reach: a later format that merely **adds** a field.
    ///
    /// `WalkHeader` denies unknown fields, so without the probe this
    /// arrives as an unreadable line carrying a serde message about
    /// `cond_scale`, and the clear refusal that exists for exactly this
    /// is never reached. Written as a raw line rather than through
    /// `write_jsonl`, because this build cannot serialise a field it
    /// does not have.
    #[test]
    fn a_later_format_that_adds_a_field_is_refused_as_a_version() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v2.jsonl");
        let body = format!(
            r#"{{"version": {}, "ckpt": "c", "holdout": "h", "side": "White", "encoding": "prefix", "ctx": 128, "bands": [], "gammas": [1.0], "positions": 0, "games": 0, "cond_scale": 2.0}}"#,
            FORMAT_VERSION + 1
        );
        std::fs::write(&path, format!("{body}\n")).unwrap();
        let err = Walk::read_jsonl(&path);
        assert!(
            matches!(
                err,
                Err(RecordError::Version {
                    expected: FORMAT_VERSION,
                    ..
                })
            ),
            "an added field must not hide the version: {err:?}"
        );
    }

    /// And a file of *this* version with an unknown field is still a
    /// parse failure, so the probe has not turned the strictness off —
    /// it has only let the version be read first.
    #[test]
    fn an_unknown_field_at_the_current_version_is_still_a_parse_failure() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("odd.jsonl");
        let body = format!(
            r#"{{"version": {FORMAT_VERSION}, "ckpt": "c", "holdout": "h", "side": "White", "encoding": "prefix", "ctx": 128, "bands": [], "gammas": [1.0], "positions": 0, "games": 0, "cond_scale": 2.0}}"#
        );
        std::fs::write(&path, format!("{body}\n")).unwrap();
        assert!(matches!(
            Walk::read_jsonl(&path),
            Err(RecordError::Parse { line: 1, .. })
        ));
    }

    /// A run that died half way leaves a header claiming more positions
    /// than the file holds. Refused rather than read as a short walk,
    /// which would silently change the sample every arm is compared on.
    #[test]
    fn a_short_file_is_refused_rather_than_read_as_a_short_walk() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("short.jsonl");
        let mut original = walk(2, 3, false);
        original.header.positions = 99;
        original.write_jsonl(&path).unwrap();
        assert!(matches!(
            Walk::read_jsonl(&path),
            Err(RecordError::Truncated {
                declared: 99,
                found: 6,
                ..
            })
        ));
    }

    #[test]
    fn an_empty_file_is_refused() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        assert!(matches!(
            Walk::read_jsonl(&path),
            Err(RecordError::Empty { .. })
        ));
    }

    #[test]
    fn arms_over_the_same_positions_align() {
        let arms = AlignedArms::new(vec![
            arm("perpos", walk(4, 5, true)),
            arm("prefix", walk(4, 5, false)),
        ])
        .unwrap();
        assert_eq!(arms.positions(), 20);
        assert_eq!(arms.games(), 4);
        assert_eq!(arms.ctx(), 128);
        assert_eq!(arms.clusters().first(), Some(&0));
        assert_eq!(arms.clusters().last(), Some(&3));
        assert_eq!(arms.gamma_index(4.0).unwrap(), 1);
        // And a gamma nobody swept is reported as that, not as a
        // disagreement between two arms one of which does not exist.
        assert!(matches!(
            arms.gamma_index(2.0),
            Err(AlignError::GammaNotSwept { gamma: 2.0, .. })
        ));
    }

    /// The refusal this type exists for: two arms with the same number
    /// of positions that are not the same positions.
    #[test]
    fn arms_scored_on_different_positions_are_refused() {
        let mut other = walk(4, 5, false);
        // Same shape, one position from a different ply — a walk over a
        // different holdout, or with a different filter, looks like this.
        other.records[7].ply = 999;
        let err = AlignedArms::new(vec![arm("perpos", walk(4, 5, true)), arm("prefix", other)])
            .unwrap_err();
        assert!(
            matches!(
                err,
                AlignError::PositionDiffers {
                    index: 7,
                    expected_ply: 4,
                    found_ply: 999,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn arms_of_different_lengths_are_refused() {
        let err = AlignedArms::new(vec![
            arm("perpos", walk(4, 5, true)),
            arm("prefix", walk(4, 4, false)),
        ])
        .unwrap_err();
        assert!(
            matches!(
                err,
                AlignError::PositionCountDiffers {
                    expected: 20,
                    found: 16,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Flip rate counts a disagreement with the first band across all of
    /// them, so a differing band list is a differing statistic even when
    /// the positions match.
    #[test]
    fn arms_with_different_bands_are_refused() {
        let mut other = walk(2, 2, false);
        other.header.bands.pop();
        let err = AlignedArms::new(vec![arm("a", walk(2, 2, true)), arm("b", other)]).unwrap_err();
        assert!(matches!(err, AlignError::BandsDiffer { .. }), "{err:?}");
    }

    #[test]
    fn arms_with_different_gammas_are_refused() {
        let mut other = walk(2, 2, false);
        other.header.gammas = vec![1.0, 2.0];
        let err = AlignedArms::new(vec![arm("a", walk(2, 2, true)), arm("b", other)]).unwrap_err();
        assert!(matches!(err, AlignError::GammasDiffer { .. }), "{err:?}");
    }

    /// The deep bucket ends at `ctx - 2`, so two arms walked at
    /// different context sizes do not bucket depth the same way.
    #[test]
    fn arms_walked_at_different_contexts_are_refused() {
        let mut other = walk(2, 2, false);
        other.header.ctx = 192;
        let err = AlignedArms::new(vec![arm("a", walk(2, 2, true)), arm("b", other)]).unwrap_err();
        assert!(matches!(err, AlignError::CtxDiffers { .. }), "{err:?}");
    }

    /// The one that a whole end-to-end run walked straight past: pass
    /// the same checkpoint twice and every number is well-formed, while
    /// the difference between those two arms is zero in every draw. If
    /// the pair is the two per-position runs, the same-arm gap the
    /// margin has to beat is gone and the confirm criterion has quietly
    /// become an interval on the margin alone.
    #[test]
    fn two_arms_from_the_same_checkpoint_are_refused() {
        let err = AlignedArms::new(vec![
            ("perpos".to_string(), walk(3, 4, true)),
            ("perpos-b".to_string(), walk(3, 4, true)),
        ])
        .unwrap_err();
        assert!(
            matches!(
                err,
                AlignError::SameCheckpoint {
                    ref a,
                    ref b,
                    ..
                } if a == "perpos" && b == "perpos-b"
            ),
            "{err:?}"
        );
    }

    /// The guard `SameCheckpoint` cannot be: the same weights under two
    /// spellings — a relative path, a symlink, a directory copied by a
    /// bad `scp` — pass the string check and still have a gap of zero in
    /// every draw.
    ///
    /// This is the failure that ran undetected once, so the fixture is
    /// the shape it took: two arms whose records agree everywhere.
    #[test]
    fn two_arms_with_identical_records_are_refused_however_they_were_named() {
        let err = AlignedArms::new(vec![
            arm("perpos", walk(3, 4, true)),
            // A different path, the same weights.
            arm("perpos-b", walk(3, 4, true)),
        ])
        .unwrap_err();
        assert!(
            matches!(
                err,
                AlignError::IdenticalRecords {
                    ref a,
                    ref b,
                    positions: 12,
                } if a == "perpos" && b == "perpos-b"
            ),
            "{err:?}"
        );
    }

    /// And one position of difference is enough to pass it, so the
    /// guard is not refusing two genuinely distinct checkpoints that
    /// happen to agree often.
    #[test]
    fn one_differing_position_is_enough_to_be_two_arms() {
        let mut other = walk(3, 4, true);
        other.records[5].at[0].flipped = false;
        AlignedArms::new(vec![
            arm("perpos", walk(3, 4, true)),
            arm("perpos-b", other),
        ])
        .expect("two arms that differ anywhere are two arms");
    }

    /// A difference between arms is only a difference if both were
    /// scored on the same material. Caught on the header rather than
    /// forty positions into the sequence.
    #[test]
    fn arms_walked_over_different_holdouts_are_refused() {
        let mut other = walk(2, 2, false);
        other.header.holdout = "holdout-2026-04.pgn".into();
        let err = AlignedArms::new(vec![arm("a", walk(2, 2, true)), arm("b", other)]).unwrap_err();
        assert!(
            matches!(
                err,
                AlignError::HoldoutDiffers {
                    field: "holdout",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn arms_walked_over_different_sides_are_refused() {
        let mut other = walk(2, 2, false);
        other.header.side = "Black".into();
        let err = AlignedArms::new(vec![arm("a", walk(2, 2, true)), arm("b", other)]).unwrap_err();
        assert!(
            matches!(err, AlignError::HoldoutDiffers { field: "side", .. }),
            "{err:?}"
        );
    }

    #[test]
    fn two_arms_of_the_same_name_are_refused() {
        let err = AlignedArms::new(vec![
            ("perpos".into(), walk(2, 2, true)),
            ("perpos".into(), walk(2, 2, false)),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            AlignError::DuplicateArm {
                name: "perpos".into()
            }
        );
    }

    /// Game ids skip whenever an accepted game contributed nothing, and
    /// the bootstrap needs a dense `0..games`.
    #[test]
    fn sparse_game_ids_are_renumbered_densely() {
        let mut w = walk(1, 1, false);
        w.records = vec![
            PositionRecord {
                game: 3,
                ply: 0,
                n_legal: Some(31),
                at: vec![gamma_record(false, 0.0), gamma_record(false, 0.0)],
            },
            PositionRecord {
                game: 3,
                ply: 2,
                n_legal: Some(31),
                at: vec![gamma_record(false, 0.0), gamma_record(false, 0.0)],
            },
            PositionRecord {
                game: 9,
                ply: 0,
                n_legal: Some(31),
                at: vec![gamma_record(false, 0.0), gamma_record(false, 0.0)],
            },
        ];
        w.header.positions = 3;
        let arms = AlignedArms::new(vec![arm("a", w)]).unwrap();
        assert_eq!(arms.clusters(), &[0, 0, 1]);
        assert_eq!(arms.games(), 2);
    }
}
