//! Chess corpus ingestion.
//!
//! Turns recorded human play — public PGN archives such as the Lichess
//! open database — into the token rows a language model trains on.
//! The point of reading real games is that the alternative is writing a
//! teacher and synthesising its play, which caps the model at whatever
//! that teacher already does and puts the burden of "what should this
//! player look like" on hand-written evaluation weights.
//!
//! The pieces:
//!
//! - [`pgn`] reads games and resolves SAN into UCI moves.
//! - [`vocab`] fixes the token alphabet from the board's geometry, so
//!   the same move always carries the same id no matter which slice of
//!   an archive was read.
//! - [`filter`] decides which games belong in a corpus. This is also
//!   where a playing style comes from: imitating players in a rating
//!   band is the one mechanism known to reproduce how they move.
//! - [`corpus`] runs the stages in order — tags, replay, length,
//!   encode — and reports what happened to every game it read.
//! - [`train`] hands a banded corpus to the conditioned training loop,
//!   turning each row's band into the conditioning-table row the model
//!   indexes by.
//!
//! Reading a checkpoint has one:
//!
//! - [`batch`] pairs the rows fed to the model with the conditioning
//!   that belongs to them, and owns the forward. The band reaches a
//!   per-position model through two channels at once, and a reader that
//!   supplied one and forgot the other would produce ordinary-looking
//!   moves from a model that was never told which band it is.
//!
//! Measurement has two of its own:
//!
//! - [`records`] writes out what one scoring walk saw at each position,
//!   with the game it came from, so that several checkpoints scored in
//!   separate processes can be read together afterwards.
//! - [`steerability`] assembles the pre-registered hypotheses from
//!   those records, with error bars that resample games rather than
//!   positions.

pub mod batch;
pub mod corpus;
pub mod filter;
pub mod freebie;
pub mod guide;
pub mod pgn;
pub mod records;
pub mod steerability;
pub mod train;
pub mod vocab;
pub mod window;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::arch::{CondIndex, Gpt2Config, Gpt2Custom};
use crate::chess::corpus::ConditionBand;

/// Context window, in tokens.
///
/// Measured on a 23,989-game slice of the Lichess 2026-06 database:
/// games run 66.9 plies on average and 128 tokens hold 96.2% of them
/// whole. Going to 192 would hold 99.9%, at the cost of attention that
/// grows with the square of the window.
pub const CTX: usize = 128;

/// Transformer blocks.
///
/// Two, because the earlier 6x6 Othello work measured depth going the
/// wrong way — 2 layers beat 4 beat 8 on both loss and legal-move rate
/// at equal steps — and the comparable published 6x6 study used three.
pub const LAYERS: usize = 2;

/// Attention heads.
pub const HEADS: usize = 4;

/// Hidden size.
///
/// Width was the dimension that mattered in the published Othello
/// ablations, where 512 at one layer beat narrower models at four.
pub const DIM: usize = 128;

/// The tensor that tells the two conditioning conventions apart on
/// disk: the table [`crate::arch::Gpt2Model::forward_conditioned`]
/// indexes into.
///
/// A per-position checkpoint carries it and a prefix one does not, which
/// is what makes the recorded [`CondEncoding`] checkable against the
/// weights themselves rather than only against a file sitting next to
/// them. The name is the one [`crate::arch::Gpt2Model::new`] registers,
/// so the two move together or the load fails on the name.
pub const COND_TABLE_TENSOR: &str = "cond_wte.weight";

/// The tensor that says a checkpoint was trained with the ids allowed
/// at each position handed to it as input: the table
/// [`crate::arch::Gpt2Model::forward_legal`] reads.
///
/// [`COND_TABLE_TENSOR`]'s counterpart on the other axis, and checked
/// the same way — the presence of this name in the weights against what
/// [`ModelShape::legal_input`] records. The two axes are independent on
/// disk even though no forward pass in this build reads both
/// (`Gpt2Custom::validate` refuses a model asking for the pair), so the
/// cross-check treats them as two questions rather than one.
pub const LEGAL_TABLE_TENSOR: &str = "legal_wte.weight";

/// How a checkpoint was told which band it is playing as.
///
/// The two conventions are close enough on disk that a reader which
/// guesses gets a plausible distribution out of the wrong one rather
/// than an error. They differ by one tensor — see
/// [`CondEncoding::EveryPosition`] — which is what
/// [`ModelShape::load_any`] cross-checks the recorded value against, so
/// that a sidecar describing other weights is caught rather than
/// believed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CondEncoding {
    /// The band token sits at position 1 of the row, behind `BOS`, and
    /// the model reads it as an ordinary token. The default, because
    /// every checkpoint written before this field existed was trained
    /// this way and their sidecars carry no `encoding`.
    #[default]
    Prefix,
    /// The band is passed to the forward pass as an argument — an index
    /// into a conditioning table of its own, `cond_wte` — and that
    /// vector is added at every position
    /// ([`crate::arch::Gpt2Model::forward_conditioned`]).
    ///
    /// The row is **unchanged**: it still begins `[BOS, band]`. The
    /// row's copy of the band is redundant under this encoding, and it
    /// is kept deliberately, so that the two arms train on the same
    /// corpus, the same row lengths and the same token counts — the
    /// alternative would add "the rows differed" to the list of things
    /// an arm difference could be attributed to. It also leaves the
    /// band embedded through `wte` at position 1 in both arms, so the
    /// LM-head tie at that one position is a property they share
    /// rather than a difference between them.
    EveryPosition,
}

impl std::fmt::Display for CondEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CondEncoding::Prefix => write!(f, "prefix"),
            CondEncoding::EveryPosition => write!(f, "every-position"),
        }
    }
}

/// The axes a checkpoint's shape file has to be told apart by, and
/// therefore what its name is built from.
///
/// Both axes change what the model reads at every position, and on
/// both the same direction of misreading is silent: a sidecar that does
/// not mention the axis builds a model without that table, the tensor
/// sits in the file unrequested, and every number that comes out looks
/// ordinary. That is what the naming is for.
///
/// What happens in the other direction is not the same on the two. A
/// model built with `cond_wte` and driven through the plain forward
/// runs, and its moves look no different — the whole reason
/// [`CondEncoding`] is recorded. A model built with `legal_wte` and
/// handed no sets is refused by the forward pass. The sidecar name is
/// the outer guard either way — see
/// [`ModelShape::path_for_kind`] — and the tensors are the inner one:
/// [`COND_TABLE_TENSOR`] and [`LEGAL_TABLE_TENSOR`] are each present
/// for exactly one value of their axis, which is what
/// [`ModelShape::load_any`] checks the sidecar against.
///
/// Four combinations exist here even though `Gpt2Custom::validate`
/// refuses to build a model for one of them (conditioning together with
/// a legality input). [`ModelShape::save`] will write that name anyway
/// if a shape asks for it — it validates nothing — so enumerating the
/// product rather than the buildable subset is what keeps the sweep
/// there and the ambiguity check in [`ModelShape::load_any`] total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeKind {
    /// How the band reaches the model.
    pub encoding: CondEncoding,
    /// Whether the model is handed the ids allowed at each position.
    pub legal_input: bool,
}

impl ShapeKind {
    /// Every combination, in a fixed order.
    pub const ALL: [ShapeKind; 4] = [
        ShapeKind {
            encoding: CondEncoding::Prefix,
            legal_input: false,
        },
        ShapeKind {
            encoding: CondEncoding::EveryPosition,
            legal_input: false,
        },
        ShapeKind {
            encoding: CondEncoding::Prefix,
            legal_input: true,
        },
        ShapeKind {
            encoding: CondEncoding::EveryPosition,
            legal_input: true,
        },
    ];

    /// The file extension a shape of this kind is written under.
    ///
    /// `shape.json` is the name every build of this program, past and
    /// present, looks for; the rest are deliberately names an older
    /// build does not look for. See [`ModelShape::path_for_kind`].
    pub fn suffix(self) -> &'static str {
        match (self.encoding, self.legal_input) {
            (CondEncoding::Prefix, false) => "shape.json",
            (CondEncoding::EveryPosition, false) => "shape2.json",
            (CondEncoding::Prefix, true) => "shape-legal.json",
            (CondEncoding::EveryPosition, true) => "shape2-legal.json",
        }
    }

    /// The kinds this one is not.
    pub fn others(self) -> Vec<ShapeKind> {
        Self::ALL.iter().copied().filter(|k| *k != self).collect()
    }
}

impl std::fmt::Display for ShapeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.encoding)?;
        if self.legal_input {
            write!(f, " with a legality input")?;
        }
        Ok(())
    }
}

/// The shape a checkpoint was trained at, written alongside it.
///
/// A checkpoint is a bag of tensors; nothing in it says how many
/// layers or how wide they were. Rebuild it at the wrong shape and the
/// load either fails on a tensor name or, worse, succeeds against a
/// model that is not the one that was trained. Carrying the shape next
/// to the weights is what lets a bake change size without every reader
/// having to be changed with it.
///
/// The band token is here for the same reason: it decides the
/// vocabulary layout, so a player that guesses it wrong indexes into
/// the wrong ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
// A field this build does not know is a field written by a build that
// knew something this one does not, and the shape file is precisely
// where that matters. Refusing is not retrospective — it cannot help
// `encoding`, whose absence is what an older file looks like — but it
// stops the next axis from being read as its default.
#[serde(deny_unknown_fields)]
pub struct ModelShape {
    /// Transformer blocks.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Hidden size.
    pub dim: usize,
    /// Context window in tokens.
    pub ctx: usize,
    /// Vocabulary size the model was built with.
    pub vocab: usize,
    /// Condition bands the corpus was built with, in vocabulary order.
    ///
    /// Empty for an unconditional model. More than one is the point of
    /// the mechanism: a single checkpoint that plays as whichever band
    /// is prefixed to the sequence, which is how Allie reproduced a
    /// rating scale from one model rather than one model per band.
    #[serde(default)]
    pub bands: Vec<ConditionBand>,
    /// How the band reached the model during training.
    ///
    /// Defaulted rather than required, so a checkpoint written before
    /// this field existed still loads — and loads as
    /// [`CondEncoding::Prefix`], which is what those runs did.
    ///
    /// It is recorded because the alternative is finding out late or
    /// not at all: a per-position checkpoint differs from a prefix one
    /// by a single small tensor, and scoring one under the other's
    /// convention returns numbers that look like measurements.
    #[serde(default)]
    pub encoding: CondEncoding,
    /// Whether the model was handed the ids allowed at each position
    /// while it trained
    /// ([`crate::arch::Gpt2Model::forward_legal`]).
    ///
    /// Recorded for the reason [`Self::encoding`] is: the difference on
    /// disk is one tensor, and a reader that scored such a checkpoint
    /// without supplying the input would be running a model in a state
    /// it never trained in, with every number well-formed.
    ///
    /// Omitted from the file when false, so a checkpoint of the
    /// ordinary kind is written exactly as it was before this field
    /// existed — and so a build that predates the field meets the key
    /// only on the checkpoints it must not read. `deny_unknown_fields`
    /// then makes that a refusal, for builds new enough to carry it;
    /// [`Self::path_for_kind`] is what covers the older ones.
    #[serde(default, skip_serializing_if = "is_false")]
    pub legal_input: bool,
    /// Sizes of the condition groups [`Self::bands`] is partitioned
    /// into, in band-list order — `[2, 2]` says the first two bands are
    /// one slot and the next two another, and every forward passes one
    /// row per group ([`crate::arch::Gpt2Model::forward_conditioned_groups`]).
    ///
    /// Empty for a single-slot model, and **omitted from the file**
    /// then, so every shape written before this field existed and every
    /// single-slot shape written after are the same bytes — an older
    /// build keeps reading them. A multi-slot shape writes the field,
    /// and the same older build refuses it on `deny_unknown_fields`:
    /// the loud outcome, since scoring a multi-slot checkpoint under
    /// single-slot conventions would condition every row on one group
    /// and read the rest as moves. The split
    /// [`crate::chess::corpus::ConditionBand`]'s sidecar forms make for
    /// the matcher axis, made here for the grouping axis.
    ///
    /// The sizes must sum to `bands.len()`; [`Self::load_any`] refuses
    /// a file where they do not, because every consumer below it would
    /// otherwise partition the band list at well-formed but wrong
    /// boundaries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cond_groups: Vec<usize>,
}

/// `skip_serializing_if` for a plain `bool` field.
fn is_false(b: &bool) -> bool {
    !*b
}

/// Failure while reading or writing a shape file.
#[derive(Debug, Error)]
pub enum ShapeError {
    /// The file could not be read or written.
    #[error("shape file {path}: {message}")]
    Io {
        /// Path involved.
        path: String,
        /// Underlying message.
        message: String,
    },
    /// The file is not valid shape JSON.
    #[error("shape file {path} is not readable as a shape: {message}")]
    Parse {
        /// Path involved.
        path: String,
        /// Underlying message.
        message: String,
    },
    /// The shape's group sizes do not sum to its band count.
    ///
    /// Refused at load rather than left to consumers, because every
    /// reader below this would partition the band list at well-formed
    /// but wrong boundaries — a slot reading its neighbour's bands,
    /// with every number downstream still printable.
    #[error(
        "shape file {path} partitions {bands} band(s) into groups of {groups:?}, which do not \
         sum to it; a reader following those boundaries would put bands in the wrong slots"
    )]
    GroupsContradictBands {
        /// Sidecar involved.
        path: String,
        /// The group sizes it declares.
        groups: Vec<usize>,
        /// How many bands it carries.
        bands: usize,
    },

    /// The checkpoint was conditioned one way and the caller is set up
    /// for the other.
    ///
    /// An error rather than a warning. A reader that carried on would
    /// produce a full set of plausible numbers from a model that was
    /// never told which band it is playing as. The conventions differ
    /// by one tensor, so a strict weight load would eventually refuse
    /// the pair too — but only in the direction where the table is
    /// missing, and only after the file has been read; every other
    /// reading downstream of that is silent.
    #[error(
        "checkpoint {path} was trained with {found} conditioning, but this reader is set up for \
         {want}. Everything except where the band enters the forward pass is the same, so \
         nothing further along would have reported a wrong number as wrong"
    )]
    EncodingMismatch {
        /// Checkpoint involved.
        path: String,
        /// What the caller can read.
        want: CondEncoding,
        /// What the shape file records.
        found: CondEncoding,
    },
    /// The shape was written, and the other convention's sidecar could
    /// not be removed afterwards.
    ///
    /// Distinct from [`ShapeError::Io`] on the write because the two
    /// leave opposite states and a caller cleaning up has to know
    /// which it is in: here the file named by `written` is correct and
    /// current, and the one named by `stale` is what has to go. A
    /// caller that treated this like a failed write and removed
    /// `written` would leave the stale sidecar alone beside the new
    /// weights, which is a checkpoint describing itself as the
    /// convention it is not. [`ModelShape::load_any`] refuses that
    /// pairing on the weights themselves, so it is caught rather than
    /// scored — but what it leaves an operator with is a contradiction
    /// to work out, where keeping `written` and losing `stale` leaves a
    /// checkpoint that simply loads.
    ///
    /// The paths are `PathBuf` rather than strings because this is the
    /// one variant a caller is expected to act on.
    #[error(
        "the shape was written to {}, but the {} sidecar at {} could not be removed ({message}); \
         until it is deleted every reader will refuse this checkpoint as ambiguous",
        .written.display(),
        .stale_kind,
        .stale.display(),
    )]
    SweepFailed {
        /// Sidecar this save wrote. Correct and current.
        written: PathBuf,
        /// Sidecar of another kind, left behind. The first that could
        /// not be removed; the sweep tries every other name before
        /// returning, so a second survivor is possible and the reader
        /// meets it as [`ShapeError::AmbiguousSidecar`].
        stale: PathBuf,
        /// Convention the stale file describes.
        stale_kind: ShapeKind,
        /// Underlying IO message.
        message: String,
    },
    /// The weights themselves could not be read, so the recorded
    /// encoding could not be checked against them.
    ///
    /// Distinct from [`ShapeError::Io`], which is about the sidecar. A
    /// shape read beside weights that are not there describes nothing,
    /// and every caller of [`ModelShape::load_any`] goes on to load
    /// those weights — so this refuses at the point where the path is
    /// still in hand rather than several steps later.
    #[error(
        "checkpoint {path}: the weights could not be read to check the shape against ({message})"
    )]
    WeightsUnreadable {
        /// Checkpoint involved.
        path: String,
        /// Underlying message.
        message: String,
    },
    /// The sidecar and the weights disagree about how the checkpoint was
    /// conditioned.
    ///
    /// This is the one route the sidecar scheme cannot see on its own. A
    /// per-position checkpoint that ends up beside only a stale
    /// `*.shape.json` — hand-copied, or an `scp` that took one sidecar
    /// and not the other — is not ambiguous and is not a mismatch
    /// against the caller: it reads as prefix, and a prefix reader
    /// agrees with it. Nothing further along notices, because the
    /// readers load weights through a mmapped `VarBuilder`, which asks
    /// only for the names its model wants and never sees
    /// [`COND_TABLE_TENSOR`] sitting in the file unrequested.
    ///
    /// Both directions are real. A per-position checkpoint read as
    /// prefix is scored with the condition delivered through one channel
    /// instead of two; a prefix checkpoint read as per-position asks for
    /// a table that is not there, which the weight load would refuse —
    /// but only after the shape has already decided how the rows are
    /// built.
    #[error(
        "checkpoint {path} has a sidecar recording {declared} conditioning, but its weights are \
         {implied}: `{tensor}` is present in a per-position checkpoint and in no other. One of \
         the two belongs to a different run — a sidecar left behind by a copy is the usual \
         cause — and reading it anyway scores these weights under the wrong convention with \
         every number well-formed"
    )]
    EncodingContradictsWeights {
        /// Checkpoint involved.
        path: String,
        /// What the sidecar records.
        declared: CondEncoding,
        /// What the presence or absence of [`COND_TABLE_TENSOR`] says.
        implied: CondEncoding,
        /// The tensor the two were compared over.
        tensor: &'static str,
    },
    /// The sidecar and the weights disagree about whether the model
    /// was handed the ids allowed at each position.
    ///
    /// [`Self::EncodingContradictsWeights`] on the other axis, with the
    /// same origin (a sidecar left behind by a copy) and the same
    /// consequence: [`LEGAL_TABLE_TENSOR`] sits in the file unrequested
    /// and the readers, which ask a mmapped `VarBuilder` only for the
    /// names their model wants, never see it.
    ///
    /// The two directions differ in how far they get. Weights with the
    /// table, read as though without it, build a model with no
    /// `legal_wte` and score it — every number well-formed and every
    /// one produced without the channel the run trained under. Weights
    /// without it, read as though with it, ask for a tensor that is not
    /// there and the weight load refuses — but only after the shape has
    /// decided what the reader is going to do.
    #[error(
        "checkpoint {path}: its sidecar records a legality input of {declared} while its \
         weights say {implied} — `{tensor}` is present in a checkpoint trained with one and in \
         no other. One of the two belongs to a different run, a sidecar left behind by a copy \
         being the usual cause"
    )]
    LegalInputContradictsWeights {
        /// Checkpoint involved.
        path: String,
        /// What the sidecar records.
        declared: bool,
        /// What the presence or absence of [`LEGAL_TABLE_TENSOR`] says.
        implied: bool,
        /// The tensor the two were compared over.
        tensor: &'static str,
    },
    /// The checkpoint was trained with a legality input and the caller
    /// cannot supply one.
    ///
    /// Every reader generates the legal moves for the position in front
    /// of it — it is how they rank what the model produced — but handing
    /// them to the forward pass means having them for **every** position
    /// of the row, including the ones a windowed row no longer carries
    /// the history for. `chess_cond` and `chess_play` recover those by
    /// replaying the game and mapping it onto the row
    /// ([`crate::chess::window::Window::legal_sets`]), so they no longer
    /// come here. `chess_eval` and `chess_match` do not, and this is
    /// what they say instead of scoring the model with the channel
    /// absent from every position.
    #[error(
        "checkpoint {path} was trained with the legal ids supplied at every position, and this \
         reader does not supply them; scoring it anyway would run the model in a state it never \
         trained in and every number would still look ordinary"
    )]
    LegalInputUnsupported {
        /// Checkpoint involved.
        path: String,
    },
    /// One checkpoint, more than one shape file.
    ///
    /// Only one of them can describe the weights that are actually
    /// there, and nothing in any of them says which. Reading one anyway
    /// is how a per-position checkpoint gets scored as a prefix one, so
    /// this is refused instead of resolved by precedence.
    #[error(
        "checkpoint has {} shape files ({}), written under different conventions; only one can \
         describe these weights and nothing here says which, so delete the ones that do not \
         belong to this run",
        .found.len(),
        .found.join(", "),
    )]
    AmbiguousSidecar {
        /// Every sidecar found beside the checkpoint.
        found: Vec<String>,
    },
}

/// Which conventions the weights on disk were built under, read from
/// the tensor names rather than from anything written beside them.
///
/// The safetensors header is a name-to-descriptor map at the front of
/// the file, so this reads the header and nothing else: no tensor is
/// materialised and the mapping is dropped on return.
///
/// This is deliberately not the reader the models load through.
/// `VarBuilder` answers "is the tensor I asked for here", which cannot
/// see a tensor nobody asked for — and an unrequested
/// [`COND_TABLE_TENSOR`] is exactly the state a stale prefix sidecar
/// leaves behind.
fn axes_of_weights(ckpt: &Path) -> Result<ShapeKind, ShapeError> {
    // SAFETY: the same discipline every other safetensors load in this
    // crate follows — the file must not be truncated while the mapping
    // is alive. It is created and dropped inside this call, and the
    // checkpoints this reads are written once and then only read.
    let weights =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(ckpt) }.map_err(|e| {
            ShapeError::WeightsUnreadable {
                path: ckpt.display().to_string(),
                message: e.to_string(),
            }
        })?;
    Ok(ShapeKind {
        encoding: match weights.get(COND_TABLE_TENSOR).is_ok() {
            true => CondEncoding::EveryPosition,
            false => CondEncoding::Prefix,
        },
        legal_input: weights.get(LEGAL_TABLE_TENSOR).is_ok(),
    })
}

impl ModelShape {
    /// The shape the local measurements sized this path for.
    pub fn compact(vocab: usize, bands: Vec<ConditionBand>) -> Self {
        Self {
            layers: LAYERS,
            heads: HEADS,
            dim: DIM,
            ctx: CTX,
            vocab,
            bands,
            encoding: CondEncoding::default(),
            legal_input: false,
            cond_groups: Vec::new(),
        }
    }

    /// The pair of axes this shape's sidecar name is built from.
    pub fn kind(&self) -> ShapeKind {
        ShapeKind {
            encoding: self.encoding,
            legal_input: self.legal_input,
        }
    }

    /// The effective group sizes: [`Self::cond_groups`] when stated,
    /// otherwise the whole band list as one group.
    ///
    /// One accessor rather than reading the field, so "empty means one
    /// group" is decided in one place. Empty bands give an empty list —
    /// an unconditioned model has no groups, not one group of nothing.
    pub fn effective_cond_groups(&self) -> Vec<usize> {
        if !self.cond_groups.is_empty() {
            self.cond_groups.clone()
        } else if self.bands.is_empty() {
            Vec::new()
        } else {
            vec![self.bands.len()]
        }
    }

    /// Find a band by its token.
    pub fn band(&self, token: &str) -> Option<&ConditionBand> {
        self.bands.iter().find(|b| b.token == token)
    }

    /// The condition tokens, in vocabulary order.
    pub fn band_tokens(&self) -> Vec<String> {
        self.bands.iter().map(|b| b.token.clone()).collect()
    }

    /// The conditioning-table row a band token stands for, for a model
    /// built under [`CondEncoding::EveryPosition`].
    ///
    /// This is the only way to obtain a [`CondIndex`] from outside the
    /// crate, and it exists because the obvious thing to reach for is
    /// wrong: a caller building a row already holds
    /// `vocab.id_of(&band.token)`, and those ids start at 2 (`PAD`,
    /// `BOS`, then the condition tokens) while the table's rows start
    /// at 0. Handing the id over would select a different band — the
    /// first band's id 2 picks row 2, the third band — and every check
    /// downstream would pass.
    ///
    /// # What holds the correspondence
    ///
    /// The rows here follow `bands` order, and [`Self::config`] sizes
    /// the table to `bands.len()`. That is a **count**, not a mapping:
    /// nothing in this file makes row *i* mean `bands[i]`. What makes
    /// it true is the training loop, which has to condition each row
    /// on the index this function returns for that row's band — and if
    /// it ever indexed by something else, every band would be
    /// misattributed with no error anywhere.
    ///
    /// So the two ends are pinned by a test
    /// (`every_band_maps_to_its_own_table_row`) rather than by the
    /// type system, and this paragraph is here so the next person to
    /// write the training side knows which end they are holding.
    pub fn band_index(&self, token: &str) -> Option<CondIndex> {
        self.bands
            .iter()
            .position(|b| b.token == token)
            .map(|i| CondIndex::from_table_row(i as u32))
    }

    /// Build the model config this shape describes.
    ///
    /// Under [`CondEncoding::EveryPosition`] the config asks for a
    /// conditioning table with one row per band, which is what
    /// `Gpt2Model::forward_conditioned` indexes into; under
    /// [`Self::legal_input`] it asks for the legality table
    /// `Gpt2Model::forward_legal` reads. A shape asking for both
    /// describes a model `Gpt2Custom::validate` refuses to build, and
    /// the refusal is left there rather than repeated here — this
    /// function has no way to report one.
    pub fn config(&self, device: Device, dtype: DType) -> Gpt2Config {
        let custom = match (self.encoding, self.legal_input) {
            (CondEncoding::Prefix, false) => None,
            (encoding, legal_input) => Some(Gpt2Custom {
                cond_slots: match encoding {
                    CondEncoding::Prefix => None,
                    CondEncoding::EveryPosition => Some(self.bands.len()),
                },
                legal_input,
                ..Default::default()
            }),
        };
        Gpt2Config {
            layers: self.layers,
            heads: self.heads,
            dim: self.dim,
            ctx: self.ctx,
            vocab: self.vocab,
            dtype,
            device,
            eps: 1e-5,
            moe: None,
            custom,
        }
    }

    /// The path a prefix-conditioned checkpoint's shape file sits at.
    ///
    /// Kept as the bare name because it is the one every build of this
    /// program, past and present, looks for.
    pub fn path_for(ckpt: &Path) -> PathBuf {
        ckpt.with_extension("shape.json")
    }

    /// The path a checkpoint's shape file sits at, for a given pair of
    /// axes.
    ///
    /// Every kind except "prefix-conditioned, no legality input" is
    /// written to a name older builds do not look for. That is
    /// deliberate and it is the only protection available in that
    /// direction: `serde` ignores nothing here any more, but a build
    /// that predates a field would read the sidecar, see the fields it
    /// knows, and score the checkpoint as whatever those fields say.
    /// Writing somewhere else turns that into "no shape file", which
    /// every reader in this crate already refuses.
    ///
    /// So a name per combination rather than per axis. `shape3.json`
    /// would have done as well and says less; the legality names carry
    /// the word.
    pub fn path_for_kind(ckpt: &Path, kind: ShapeKind) -> PathBuf {
        ckpt.with_extension(kind.suffix())
    }

    /// Write the shape beside a checkpoint.
    ///
    /// Every other kind's name is removed afterwards. A checkpoint is
    /// written under one set of conventions at a time, so a sidecar at
    /// another name is an earlier run's, describing weights that are no
    /// longer there — and two sidecars beside one checkpoint is a state
    /// [`Self::load_any`] refuses outright, since nothing in either
    /// file says which of them the weights belong to.
    ///
    /// The removal's outcome rides on the return value rather than
    /// being dropped: a write that succeeded and a sweep that failed
    /// leaves exactly the ambiguity this is here to prevent, and saying
    /// so is the difference between a checkpoint an operator can fix in
    /// one command and one that reads as unloadable for no visible
    /// reason. That case returns [`ShapeError::SweepFailed`] rather
    /// than [`ShapeError::Io`], because the two leave opposite states
    /// and a caller cleaning up after this has to be able to tell them
    /// apart — this one has already written a correct sidecar.
    pub fn save(&self, ckpt: &Path) -> Result<PathBuf, ShapeError> {
        let kind = self.kind();
        let path = Self::path_for_kind(ckpt, kind);
        let body = serde_json::to_string_pretty(self).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        std::fs::write(&path, body).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        // Every other name is swept, and a failure on one does not stop
        // the rest: leaving a second stale sidecar behind because the
        // first could not be removed would turn a fixable mispairing
        // into a longer one, and the error can only name one of them.
        let mut failed: Option<ShapeError> = None;
        for stale_kind in kind.others() {
            let stale = Self::path_for_kind(ckpt, stale_kind);
            match std::fs::remove_file(&stale) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    failed.get_or_insert_with(|| ShapeError::SweepFailed {
                        written: path.clone(),
                        stale,
                        stale_kind,
                        message: e.to_string(),
                    });
                }
            }
        }
        match failed {
            Some(e) => Err(e),
            None => Ok(path),
        }
    }

    /// Read the shape written beside a checkpoint, whichever
    /// conditioning convention it was written under.
    ///
    /// Named for what it does rather than for being the default, so
    /// that [`Self::load_as`] is the shorter thing to reach for. The
    /// callers that want it can all genuinely handle either convention:
    ///
    /// - `chess_bake`'s resume check, which has to read a checkpoint of
    ///   either convention in order to compare it against the run's own,
    ///   and which is not a reader — it goes on to train, so the
    ///   legality gate in [`crate::chess::open_reader_shape`] would be
    ///   wrong for it;
    /// - `chess_cond` and `chess_play`, which handle either conditioning
    ///   convention — the row is the same `[BOS, band] + moves` under
    ///   both, and the branch lives in
    ///   [`crate::chess::batch::BandBatch`] rather than at the call site
    ///   — and either value of [`Self::legal_input`], since they recover
    ///   the sets a legality checkpoint reads. `BandBatch` requires
    ///   those of them, so the gate they skip here is one they meet
    ///   there;
    /// - [`crate::chess::open_reader_shape`], which is this plus that
    ///   gate, for the readers that cannot supply the sets.
    ///
    /// It is not a loophole around [`Self::load_as`]. A caller that
    /// supports one convention has to say which, because the failure it
    /// prevents is silent: the two conventions produce a complete set of
    /// plausible numbers from each other's weights. Reach for this only
    /// when the code after it branches on [`Self::encoding`].
    ///
    /// # Why the weights are opened here
    ///
    /// Everything above rests on the sidecar being the one that belongs
    /// to these weights, and a single sidecar cannot say so. A stale
    /// `*.shape.json` beside per-position weights is not ambiguous and
    /// not a mismatch — it reads as prefix and a prefix reader agrees
    /// with it. So the recorded conventions are checked against the
    /// weights themselves: [`COND_TABLE_TENSOR`] is present exactly for
    /// [`CondEncoding::EveryPosition`] and [`LEGAL_TABLE_TENSOR`]
    /// exactly for [`Self::legal_input`], and either disagreement is
    /// [`ShapeError::EncodingContradictsWeights`] or
    /// [`ShapeError::LegalInputContradictsWeights`].
    ///
    /// It sits here rather than in a verification function each reader
    /// calls because this is already the single funnel —
    /// [`Self::load_as`] and [`crate::chess::open_reader_shape`] both go
    /// through it, so every reader in the workspace is covered by one
    /// call site rather than by five that each have to remember.
    ///
    /// That is a narrowed surface, not an impossibility, and the
    /// difference is worth stating because an earlier version of this
    /// paragraph claimed the stronger thing. `ModelShape` is `pub` and
    /// derives `Deserialize`, and [`Self::path_for`] and
    /// [`Self::path_for_kind`] are `pub`, so a reader that reaches
    /// for `serde_json::from_str` on a sidecar path never arrives here
    /// and is checked by nothing. Closing that would mean the shape can
    /// only be built by a constructor that has seen the weights —
    /// sealing the `Deserialize` behind a private wire type. Until then
    /// this covers the readers that exist, and the way past it is a
    /// route someone has to take deliberately.
    ///
    /// The cost is one extra open of the checkpoint, header only: the
    /// mapping made here is dropped before this returns and no tensor is
    /// materialised, and it happens once per checkpoint per run.
    ///
    /// The corollary is that the weights have to exist. A shape read
    /// beside a checkpoint that is not there describes nothing, and every
    /// caller of this loads those weights within a few lines; the one
    /// that does not, `chess_bake`'s resume check, has already tested the
    /// path with `is_file`.
    ///
    /// # Errors
    ///
    /// More than one of the four sidecar names is present, the one that
    /// is there is not readable as a shape, the weights cannot be
    /// opened, or the weights disagree with the sidecar on either axis.
    pub fn load_any(ckpt: &Path) -> Result<Self, ShapeError> {
        // More than one present is refused rather than resolved.
        // Preferring any of them would answer "which conventions is
        // this checkpoint" from a file that may have been written for
        // different weights, and the prefix-preferring version of this
        // was worse than the single-name scheme it replaced: it
        // returned a stale prefix shape for a per-position checkpoint,
        // so `require_encoding` compared the caller against the wrong
        // file and agreed with it. Nothing downstream catches that
        // either — the readers go through a mmapped `VarBuilder`, which
        // asks for the names the model wants and never notices
        // `cond_wte.weight` sitting in the file unrequested, which is
        // why the cross-check below has to look at the name list
        // itself.
        let present: Vec<PathBuf> = ShapeKind::ALL
            .iter()
            .map(|kind| Self::path_for_kind(ckpt, *kind))
            .filter(|path| path.is_file())
            .collect();
        let path = match present.as_slice() {
            // None found. The bare name is the one to name in the
            // error: it is what a reader expects and what an operator
            // will look for.
            [] => Self::path_for(ckpt),
            [only] => only.clone(),
            many => {
                return Err(ShapeError::AmbiguousSidecar {
                    found: many.iter().map(|p| p.display().to_string()).collect(),
                })
            }
        };
        let body = std::fs::read_to_string(&path).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let shape: Self = serde_json::from_str(&body).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        if !shape.cond_groups.is_empty()
            && shape.cond_groups.iter().sum::<usize>() != shape.bands.len()
        {
            return Err(ShapeError::GroupsContradictBands {
                path: path.display().to_string(),
                groups: shape.cond_groups.clone(),
                bands: shape.bands.len(),
            });
        }
        let implied = axes_of_weights(ckpt)?;
        if implied.encoding != shape.encoding {
            return Err(ShapeError::EncodingContradictsWeights {
                path: ckpt.display().to_string(),
                declared: shape.encoding,
                implied: implied.encoding,
                tensor: COND_TABLE_TENSOR,
            });
        }
        if implied.legal_input != shape.legal_input {
            return Err(ShapeError::LegalInputContradictsWeights {
                path: ckpt.display().to_string(),
                declared: shape.legal_input,
                implied: implied.legal_input,
                tensor: LEGAL_TABLE_TENSOR,
            });
        }
        Ok(shape)
    }

    /// Read the shape and refuse it unless it was conditioned the way
    /// the caller can read.
    ///
    /// It cannot be called without naming what the caller supports,
    /// which is the property [`Self::load_any`] gives up. It says
    /// nothing about the legality axis, so a reader wants
    /// [`crate::chess::open_reader_shape_as`], which is this plus that
    /// gate; what is left here is the encoding check on its own, for
    /// the tests that exercise it and for a caller that is not a reader.
    pub fn load_as(ckpt: &Path, want: CondEncoding) -> Result<Self, ShapeError> {
        let shape = Self::load_any(ckpt)?;
        shape.require_encoding(ckpt, want)?;
        Ok(shape)
    }

    /// Fail unless this checkpoint was conditioned the way `want` says.
    pub fn require_encoding(&self, ckpt: &Path, want: CondEncoding) -> Result<(), ShapeError> {
        if self.encoding == want {
            return Ok(());
        }
        Err(ShapeError::EncodingMismatch {
            path: ckpt.display().to_string(),
            want,
            found: self.encoding,
        })
    }

    /// Fail if this checkpoint expects the legal ids at every position
    /// and the caller has no way to supply them.
    ///
    /// The sibling of [`Self::require_encoding`] on the other axis, in
    /// the one direction that has a caller: a reader with no legality
    /// input to give. There is no `require_legal_input` to pair with
    /// it because nothing here asserts the opposite — a reader that can
    /// supply the sets hands them to
    /// [`crate::chess::batch::BandBatch`], which requires them exactly
    /// when [`Self::legal_input`] is set and refuses them otherwise.
    ///
    /// The readers that cannot supply the sets reach this through
    /// [`crate::chess::open_reader_shape`] rather than calling it
    /// themselves, so that the refusal is part of opening a checkpoint
    /// rather than a second line beside it.
    pub fn require_no_legal_input(&self, ckpt: &Path) -> Result<(), ShapeError> {
        if self.legal_input {
            return Err(ShapeError::LegalInputUnsupported {
                path: ckpt.display().to_string(),
            });
        }
        Ok(())
    }
}

/// Open the shape beside a checkpoint a reader that supplies **no**
/// legality input is about to score or play, refusing the kinds it
/// cannot handle.
///
/// The gate this adds is [`ModelShape::require_no_legal_input`]. A
/// checkpoint trained with the legal ids at every position has to be
/// scored with them supplied, which means having a set for every
/// position of the row rather than only for the position in front of
/// the reader — a windowed row no longer carries the history the rest
/// would be recovered from.
///
/// Two readers recover it, by replaying the game and mapping it onto
/// the row: `chess_cond` and `chess_play` go through
/// [`ModelShape::load_any`] and hand the sets to
/// [`crate::chess::batch::BandBatch`], which is where the requirement
/// is enforced for them — it takes the sets as a constructor argument
/// and refuses a legality checkpoint without them. `chess_eval` and
/// `chess_match` do not, and come through here.
///
/// # Why this exists rather than a call beside each reader
///
/// The refusal was a call site per reader, each a line the author had to
/// remember to write after loading the shape — and deleting any one of
/// them left the workspace compiling and every test passing. So it
/// belongs on the way in rather than beside it: a reader that gets its
/// shape here cannot skip the gate, because there is no second call to
/// omit.
///
/// It is a narrowing rather than a closure, and the distinction matters
/// the same way it does on [`ModelShape::load_any`]. That function
/// stays `pub` — `chess_bake`'s resume path is not a reader, and the two
/// readers that supply the sets need it — so a reader **can** be written
/// against it and be checked by nothing here. If it builds its rows with
/// [`crate::chess::batch::BandBatch`], what it meets there is a
/// constructor that will not take a legality shape without the sets; if
/// it calls a forward pass directly, the model refuses one. What no
/// reader can do is come through this function and forget.
///
/// # Errors
///
/// As [`ModelShape::load_any`], plus
/// [`ShapeError::LegalInputUnsupported`].
pub fn open_reader_shape(ckpt: &Path) -> Result<ModelShape, ShapeError> {
    let shape = ModelShape::load_any(ckpt)?;
    shape.require_no_legal_input(ckpt)?;
    Ok(shape)
}

/// [`open_reader_shape`], for a reader that handles one conditioning
/// convention rather than branching on both.
///
/// The legality gate runs first, so a checkpoint that is wrong on both
/// axes reports the legality refusal. Either message names a checkpoint
/// this reader must not score, and the legality one is the axis with no
/// reader at all.
///
/// # Errors
///
/// As [`open_reader_shape`], plus [`ShapeError::EncodingMismatch`].
pub fn open_reader_shape_as(ckpt: &Path, want: CondEncoding) -> Result<ModelShape, ShapeError> {
    let shape = open_reader_shape(ckpt)?;
    shape.require_encoding(ckpt, want)?;
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn a_shape(encoding: CondEncoding) -> ModelShape {
        let mut shape = ModelShape::compact(
            2048,
            vec![
                ConditionBand::rating(1100, 1299, "<elo:1100-1299>"),
                ConditionBand::rating(1900, 2099, "<elo:1900-2099>"),
            ],
        );
        shape.encoding = encoding;
        shape
    }

    /// Weights beside a checkpoint, under the convention `encoding`
    /// names.
    ///
    /// Two tensors is enough: nothing here builds a model, and what the
    /// cross-check reads is the name list in the safetensors header. The
    /// per-position file carries [`COND_TABLE_TENSOR`] and the prefix
    /// one does not, which is the whole of the difference on disk.
    fn write_weights(ckpt: &Path, encoding: CondEncoding) {
        write_weights_of(
            ckpt,
            ShapeKind {
                encoding,
                legal_input: false,
            },
        );
    }

    /// The same, for a checkpoint of any kind.
    fn write_weights_of(ckpt: &Path, kind: ShapeKind) {
        use candle_core::Tensor;
        use std::collections::HashMap;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let zeros = |rows: usize| Tensor::zeros((rows, 4), DType::F32, &Device::Cpu).unwrap();
        tensors.insert("wte.weight".into(), zeros(8));
        if kind.encoding == CondEncoding::EveryPosition {
            tensors.insert(COND_TABLE_TENSOR.into(), zeros(2));
        }
        if kind.legal_input {
            tensors.insert(LEGAL_TABLE_TENSOR.into(), zeros(8));
        }
        candle_core::safetensors::save(&tensors, ckpt).expect("weights on disk");
    }

    /// A checkpoint and the sidecar that belongs to it.
    fn a_checkpoint(dir: &Path, encoding: CondEncoding) -> PathBuf {
        let ckpt = dir.join("run.safetensors");
        write_weights(&ckpt, encoding);
        a_shape(encoding).save(&ckpt).unwrap();
        ckpt
    }

    #[test]
    fn compact_defaults_to_the_prefix_convention() {
        assert_eq!(
            a_shape(CondEncoding::default()).encoding,
            CondEncoding::Prefix
        );
    }

    #[test]
    fn the_encoding_survives_a_round_trip() {
        let tmp = TempDir::new().unwrap();
        let ckpt = a_checkpoint(tmp.path(), CondEncoding::EveryPosition);
        let back = ModelShape::load_any(&ckpt).unwrap();
        assert_eq!(back.encoding, CondEncoding::EveryPosition);
    }

    /// A build that predates the `encoding` field looks for
    /// `*.shape.json` and would read anything it found there as
    /// prefix-conditioned. A per-position run writes somewhere else, so
    /// what such a build finds is nothing.
    #[test]
    fn a_per_position_sidecar_is_not_where_an_older_build_looks() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let written = a_shape(CondEncoding::EveryPosition).save(&ckpt).unwrap();
        assert_eq!(
            written,
            ModelShape::path_for_kind(
                &ckpt,
                ShapeKind {
                    encoding: CondEncoding::EveryPosition,
                    legal_input: false,
                }
            )
        );
        assert!(!ModelShape::path_for(&ckpt).exists());
    }

    /// The scenario plan step 0-7 runs: two bakes on the same tiny
    /// corpus under the two encodings land on the same checkpoint
    /// filename, so the second `save` writes the second sidecar beside
    /// the first. The cross-encoding read has to be refused, and it is
    /// the second `save` sweeping the first name that makes it so —
    /// while both files existed, a prefix reader found the stale
    /// prefix sidecar, agreed with itself, and scored a per-position
    /// checkpoint.
    #[test]
    fn a_second_bake_under_the_other_encoding_is_not_readable_as_the_first() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::Prefix);
        a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
        // The second bake replaces the weights as well as the sidecar.
        write_weights(&ckpt, CondEncoding::EveryPosition);
        a_shape(CondEncoding::EveryPosition).save(&ckpt).unwrap();
        assert!(
            !ModelShape::path_for(&ckpt).exists(),
            "the prefix sidecar describes weights that are no longer there"
        );
        let err = ModelShape::load_as(&ckpt, CondEncoding::Prefix);
        assert!(
            matches!(err, Err(ShapeError::EncodingMismatch { .. })),
            "{err:?}"
        );
    }

    /// And if a sweep is skipped or fails, the ambiguity is refused
    /// rather than resolved by precedence. Written by hand here
    /// because `save` no longer produces this state.
    #[test]
    fn two_sidecars_for_one_checkpoint_are_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let prefix = serde_json::to_string(&a_shape(CondEncoding::Prefix)).unwrap();
        let perpos = serde_json::to_string(&a_shape(CondEncoding::EveryPosition)).unwrap();
        std::fs::write(ModelShape::path_for(&ckpt), prefix).unwrap();
        std::fs::write(
            ModelShape::path_for_kind(
                &ckpt,
                ShapeKind {
                    encoding: CondEncoding::EveryPosition,
                    legal_input: false,
                },
            ),
            perpos,
        )
        .unwrap();
        let err = ModelShape::load_any(&ckpt);
        assert!(
            matches!(err, Err(ShapeError::AmbiguousSidecar { .. })),
            "{err:?}"
        );
    }

    /// And a prefix run still writes exactly where it always did, so
    /// this change does not strand the checkpoints already on disk.
    #[test]
    fn a_prefix_sidecar_keeps_the_name_it_always_had() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let written = a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
        assert_eq!(written, ModelShape::path_for(&ckpt));
        assert_eq!(written.file_name().unwrap(), "run.shape.json");
    }

    /// A field this build does not know means the file was written by
    /// one that knew more, and reading it as though it had not is the
    /// failure mode `encoding` itself could not be protected from.
    #[test]
    fn a_sidecar_with_an_unknown_field_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let body = r#"{
            "layers": 4, "heads": 6, "dim": 384, "ctx": 128, "vocab": 2048,
            "bands": [], "encoding": "prefix", "cond_scale": 2.0
        }"#;
        std::fs::write(ModelShape::path_for(&ckpt), body).unwrap();
        let err = ModelShape::load_any(&ckpt);
        assert!(matches!(err, Err(ShapeError::Parse { .. })), "{err:?}");
    }

    /// The rows a band resolves to are the table's, not the
    /// vocabulary's. The two overlap numerically — band tokens start at
    /// vocabulary id 2 — which is the whole reason a caller cannot
    /// supply the number itself.
    #[test]
    fn a_band_resolves_to_its_row_in_the_table() {
        let shape = a_shape(CondEncoding::EveryPosition);
        assert_eq!(
            shape.band_index("<elo:1100-1299>").map(|i| i.row()),
            Some(0)
        );
        assert_eq!(
            shape.band_index("<elo:1900-2099>").map(|i| i.row()),
            Some(1)
        );
        assert!(shape.band_index("<elo:1500-1699>").is_none());
    }

    /// The contract nothing in the code enforces: every band has a row,
    /// the rows are `0..bands.len()` in band order, and the table is
    /// sized to hold exactly them.
    ///
    /// `config` passes a count, so a training loop that indexed by
    /// anything other than `band_index` would misattribute every band
    /// with no error anywhere. This is the assertion that says which
    /// numbering is the right one; the training side has to match it.
    #[test]
    fn every_band_maps_to_its_own_table_row() {
        use candle_core::{DType, Device};
        let shape = a_shape(CondEncoding::EveryPosition);
        for (i, token) in shape.band_tokens().iter().enumerate() {
            assert_eq!(
                shape.band_index(token).map(|x| x.row()),
                Some(i as u32),
                "band {token} should occupy row {i}"
            );
        }
        let slots = shape
            .config(Device::Cpu, DType::F32)
            .custom
            .and_then(|c| c.cond_slots);
        assert_eq!(
            slots,
            Some(shape.bands.len()),
            "the table has to hold exactly the rows the bands resolve to"
        );
    }

    /// The conditioning table is sized from the band list, and only
    /// exists under the encoding that indexes into it.
    #[test]
    fn the_config_carries_a_table_only_for_per_position() {
        use candle_core::{DType, Device};
        let prefix = a_shape(CondEncoding::Prefix).config(Device::Cpu, DType::F32);
        assert!(prefix.custom.is_none());
        let perpos = a_shape(CondEncoding::EveryPosition).config(Device::Cpu, DType::F32);
        assert_eq!(
            perpos.custom.and_then(|c| c.cond_slots),
            Some(2),
            "one row per band"
        );
    }

    /// Phase 0-2. The two arms leave identical weights, so this refusal
    /// is the only thing standing between a per-position checkpoint and
    /// a prefix reader's full set of plausible numbers.
    #[test]
    fn a_per_position_checkpoint_is_refused_by_a_prefix_reader() {
        let tmp = TempDir::new().unwrap();
        let ckpt = a_checkpoint(tmp.path(), CondEncoding::EveryPosition);
        let msg = match ModelShape::load_as(&ckpt, CondEncoding::Prefix) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected the mismatch to be refused"),
        };
        assert!(msg.contains("every-position"), "{msg}");
        assert!(msg.contains("prefix"), "{msg}");
    }

    /// And the other way round, which is the case that arrives once a
    /// per-position reader exists and is pointed at an older file.
    #[test]
    fn a_prefix_checkpoint_is_refused_by_a_per_position_reader() {
        let tmp = TempDir::new().unwrap();
        let ckpt = a_checkpoint(tmp.path(), CondEncoding::Prefix);
        let err = ModelShape::load_as(&ckpt, CondEncoding::EveryPosition);
        assert!(matches!(err, Err(ShapeError::EncodingMismatch { .. })));
    }

    /// The route the sidecar scheme cannot see: weights that condition
    /// every position, beside a prefix sidecar left over from another
    /// run.
    ///
    /// Nothing else refuses this. There is one sidecar, so it is not
    /// ambiguous; it says prefix, so a prefix reader agrees with it; and
    /// the weight load asks only for the tensors its model wants, so
    /// `cond_wte.weight` sits there unrequested and unnoticed. Scored,
    /// it delivers the condition through one channel instead of two and
    /// every number that comes out is well-formed.
    #[test]
    fn a_stale_prefix_sidecar_beside_per_position_weights_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::EveryPosition);
        let body = serde_json::to_string(&a_shape(CondEncoding::Prefix)).unwrap();
        std::fs::write(ModelShape::path_for(&ckpt), body).unwrap();

        let err = ModelShape::load_any(&ckpt).unwrap_err();
        assert!(
            matches!(
                err,
                ShapeError::EncodingContradictsWeights {
                    declared: CondEncoding::Prefix,
                    implied: CondEncoding::EveryPosition,
                    ..
                }
            ),
            "{err:?}"
        );
        // And through the entry point a reader actually uses, where the
        // caller and the sidecar agree with each other and are both
        // wrong about these weights.
        assert!(ModelShape::load_as(&ckpt, CondEncoding::Prefix).is_err());
    }

    /// The other direction: prefix weights beside a per-position
    /// sidecar, which is what a copy that took `run.shape2.json` and
    /// the wrong checkpoint leaves.
    ///
    /// The weight load would refuse this one eventually — it asks for a
    /// table that is not there — but only after the shape has already
    /// decided how the rows are built and what the forward is called
    /// with.
    #[test]
    fn a_per_position_sidecar_beside_prefix_weights_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::Prefix);
        let body = serde_json::to_string(&a_shape(CondEncoding::EveryPosition)).unwrap();
        std::fs::write(
            ModelShape::path_for_kind(
                &ckpt,
                ShapeKind {
                    encoding: CondEncoding::EveryPosition,
                    legal_input: false,
                },
            ),
            body,
        )
        .unwrap();

        let err = ModelShape::load_any(&ckpt).unwrap_err();
        assert!(
            matches!(
                err,
                ShapeError::EncodingContradictsWeights {
                    declared: CondEncoding::EveryPosition,
                    implied: CondEncoding::Prefix,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// A sidecar with no weights beside it describes nothing, and says
    /// so here rather than a few lines later in whichever reader asked.
    #[test]
    fn a_shape_beside_absent_weights_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
        let err = ModelShape::load_any(&ckpt).unwrap_err();
        assert!(
            matches!(err, ShapeError::WeightsUnreadable { .. }),
            "{err:?}"
        );
    }

    /// A legality-input shape rides the same round trip, at a name of
    /// its own.
    #[test]
    fn a_legality_input_checkpoint_records_and_reloads_it() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let kind = ShapeKind {
            encoding: CondEncoding::Prefix,
            legal_input: true,
        };
        write_weights_of(&ckpt, kind);
        let mut shape = a_shape(CondEncoding::Prefix);
        shape.legal_input = true;
        let written = shape.save(&ckpt).unwrap();

        assert_eq!(written.file_name().unwrap(), "run.shape-legal.json");
        assert!(
            !ModelShape::path_for(&ckpt).exists(),
            "a build that predates this axis looks at the bare name, and must find nothing"
        );
        let back = ModelShape::load_any(&ckpt).unwrap();
        assert!(back.legal_input);
        assert_eq!(back.encoding, CondEncoding::Prefix);
    }

    /// An ordinary checkpoint's sidecar is written exactly as it was
    /// before the axis existed, so nothing already on disk is stranded
    /// and nothing new is refused by a build that predates it.
    #[test]
    fn the_legality_field_is_absent_from_an_ordinary_sidecar() {
        let json = serde_json::to_value(a_shape(CondEncoding::Prefix)).unwrap();
        assert!(json.get("legal_input").is_none(), "got {json}");

        let mut legal = a_shape(CondEncoding::Prefix);
        legal.legal_input = true;
        let json = serde_json::to_value(legal).unwrap();
        assert_eq!(json.get("legal_input"), Some(&serde_json::json!(true)));
    }

    /// The route the sidecar naming cannot cover on its own, on this
    /// axis: weights carrying the legality table beside a sidecar that
    /// does not mention it.
    ///
    /// Nothing else refuses it. There is one sidecar, so it is not
    /// ambiguous; it says nothing about legality, so a reader with no
    /// legality input agrees with it; and the weight load asks only for
    /// the tensors its model wants, so `legal_wte.weight` sits there
    /// unrequested.
    #[test]
    fn a_sidecar_that_omits_the_legality_input_beside_weights_that_have_it_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights_of(
            &ckpt,
            ShapeKind {
                encoding: CondEncoding::Prefix,
                legal_input: true,
            },
        );
        let body = serde_json::to_string(&a_shape(CondEncoding::Prefix)).unwrap();
        std::fs::write(ModelShape::path_for(&ckpt), body).unwrap();

        let err = ModelShape::load_any(&ckpt).unwrap_err();
        assert!(
            matches!(
                err,
                ShapeError::LegalInputContradictsWeights {
                    declared: false,
                    implied: true,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// And the other direction, which is what a copy that took the
    /// legality sidecar and the wrong checkpoint leaves.
    #[test]
    fn a_legality_sidecar_beside_weights_without_the_table_is_refused() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::Prefix);
        let mut shape = a_shape(CondEncoding::Prefix);
        shape.legal_input = true;
        let body = serde_json::to_string(&shape).unwrap();
        std::fs::write(ModelShape::path_for_kind(&ckpt, shape.kind()), body.clone()).unwrap();

        let err = ModelShape::load_any(&ckpt).unwrap_err();
        assert!(
            matches!(
                err,
                ShapeError::LegalInputContradictsWeights {
                    declared: true,
                    implied: false,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// A reader with no legality input refuses such a checkpoint rather
    /// than scoring the model with that channel absent everywhere.
    #[test]
    fn a_reader_without_a_legality_input_refuses_a_checkpoint_that_wants_one() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        let mut shape = a_shape(CondEncoding::Prefix);
        shape.legal_input = true;
        let err = shape.require_no_legal_input(&ckpt).unwrap_err();
        assert!(
            matches!(err, ShapeError::LegalInputUnsupported { .. }),
            "{err:?}"
        );
        // And an ordinary checkpoint passes the same gate.
        a_shape(CondEncoding::Prefix)
            .require_no_legal_input(&ckpt)
            .expect("a checkpoint that wants no legality input is readable");
    }

    /// A second bake under the other objective lands on the same
    /// checkpoint name, and the sweep is what keeps the first bake's
    /// sidecar from describing the second bake's weights.
    #[test]
    fn a_bake_with_a_legality_input_sweeps_the_ordinary_sidecar() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::Prefix);
        a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
        assert!(ModelShape::path_for(&ckpt).is_file());

        let kind = ShapeKind {
            encoding: CondEncoding::Prefix,
            legal_input: true,
        };
        write_weights_of(&ckpt, kind);
        let mut legal = a_shape(CondEncoding::Prefix);
        legal.legal_input = true;
        legal.save(&ckpt).unwrap();

        assert!(
            !ModelShape::path_for(&ckpt).exists(),
            "the ordinary sidecar describes weights that are no longer there"
        );
        assert!(ModelShape::load_any(&ckpt).unwrap().legal_input);
    }

    /// The table only exists under the axis that reads it, and the two
    /// axes do not describe a model together — `Gpt2Custom::validate`
    /// refuses the pair, which is asserted here through the config this
    /// shape hands it.
    #[test]
    fn the_config_carries_a_legality_table_only_when_the_shape_says_so() {
        use candle_core::{DType, Device};
        let plain = a_shape(CondEncoding::Prefix).config(Device::Cpu, DType::F32);
        assert!(plain.custom.is_none());

        let mut shape = a_shape(CondEncoding::Prefix);
        shape.legal_input = true;
        let custom = shape
            .config(Device::Cpu, DType::F32)
            .custom
            .expect("a legality input needs a custom spec");
        assert!(custom.legal_input);
        assert_eq!(custom.cond_slots, None);
        custom.validate().expect("legality alone is buildable");

        let mut both = a_shape(CondEncoding::EveryPosition);
        both.legal_input = true;
        let custom = both
            .config(Device::Cpu, DType::F32)
            .custom
            .expect("both axes need a custom spec");
        assert!(
            custom.validate().is_err(),
            "no forward pass reads both channels, so the model must not build"
        );
    }

    /// Every checkpoint baked before the field existed has a sidecar
    /// without it, and every one of those was prefix-conditioned.
    #[test]
    fn a_sidecar_with_no_encoding_field_loads_as_prefix() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
        write_weights(&ckpt, CondEncoding::Prefix);
        let body = r#"{
            "layers": 4,
            "heads": 6,
            "dim": 384,
            "ctx": 128,
            "vocab": 2048,
            "bands": [{"min": 1100, "max": 1299, "token": "<elo:1100-1299>"}]
        }"#;
        std::fs::write(ModelShape::path_for(&ckpt), body).unwrap();
        let shape = ModelShape::load_as(&ckpt, CondEncoding::Prefix).unwrap();
        assert_eq!(shape.encoding, CondEncoding::Prefix);
        assert_eq!(shape.layers, 4);
    }
}
