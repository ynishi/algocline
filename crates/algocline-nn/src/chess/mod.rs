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

pub mod corpus;
pub mod filter;
pub mod guide;
pub mod pgn;
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

/// How a checkpoint was told which band it is playing as.
///
/// The two conventions are close enough on disk that a reader which
/// guesses gets a plausible distribution out of the wrong one rather
/// than an error. They differ by one tensor — see
/// [`CondEncoding::EveryPosition`] — so a strict weight load catches a
/// crossed pair as well, but only once it is loaded and only in one
/// direction. This field is what makes the refusal happen first and in
/// both directions.
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

impl CondEncoding {
    /// The convention this one is not.
    ///
    /// Two variants, so "the other one" is well defined; a third would
    /// have to make the sidecar sweep in [`ModelShape::save`] and the
    /// ambiguity check in [`ModelShape::load_any`] enumerate instead.
    pub fn other(self) -> Self {
        match self {
            CondEncoding::Prefix => CondEncoding::EveryPosition,
            CondEncoding::EveryPosition => CondEncoding::Prefix,
        }
    }
}

impl std::fmt::Display for CondEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CondEncoding::Prefix => write!(f, "prefix"),
            CondEncoding::EveryPosition => write!(f, "every-position"),
        }
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
    /// weights — a checkpoint that reads as the wrong convention,
    /// which is worse than one that does not read at all.
    ///
    /// The paths are `PathBuf` rather than strings because this is the
    /// one variant a caller is expected to act on.
    #[error(
        "the shape was written to {}, but the {} sidecar at {} could not be removed ({message}); \
         until it is deleted every reader will refuse this checkpoint as ambiguous",
        .written.display(),
        .stale_encoding,
        .stale.display(),
    )]
    SweepFailed {
        /// Sidecar this save wrote. Correct and current.
        written: PathBuf,
        /// Sidecar of the other convention, left behind.
        stale: PathBuf,
        /// Convention the stale file describes.
        stale_encoding: CondEncoding,
        /// Underlying IO message.
        message: String,
    },
    /// One checkpoint, two shape files, one per conditioning
    /// convention.
    ///
    /// Only one of them can describe the weights that are actually
    /// there, and nothing in either says which. Reading one anyway is
    /// how a per-position checkpoint gets scored as a prefix one, so
    /// this is refused instead of resolved by precedence.
    #[error(
        "checkpoint has two shape files, {first} and {second}, written under different \
         conditioning conventions; only one can describe these weights and nothing here says \
         which, so delete the one that does not belong to this run"
    )]
    AmbiguousSidecar {
        /// Prefix-convention sidecar.
        first: String,
        /// Per-position sidecar.
        second: String,
    },
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
    /// `Gpt2Model::forward_conditioned` indexes into.
    pub fn config(&self, device: Device, dtype: DType) -> Gpt2Config {
        let custom = match self.encoding {
            CondEncoding::Prefix => None,
            CondEncoding::EveryPosition => Some(Gpt2Custom {
                cond_slots: Some(self.bands.len()),
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

    /// The path a checkpoint's shape file sits at under a given
    /// conditioning convention.
    ///
    /// Anything other than [`CondEncoding::Prefix`] is written to a
    /// name older builds do not look for. That is deliberate and it is
    /// the only protection available in that direction: `serde` ignores
    /// nothing here any more, but a build that predates the `encoding`
    /// field would read a per-position sidecar, see fields it knows,
    /// and score the checkpoint as prefix-conditioned. Writing
    /// somewhere else turns that into "no shape file", which every
    /// reader in this crate already refuses.
    pub fn path_for_encoding(ckpt: &Path, encoding: CondEncoding) -> PathBuf {
        match encoding {
            CondEncoding::Prefix => Self::path_for(ckpt),
            CondEncoding::EveryPosition => ckpt.with_extension("shape2.json"),
        }
    }

    /// Write the shape beside a checkpoint.
    ///
    /// The other convention's name is removed afterwards. A checkpoint
    /// is written under one convention at a time, so a sidecar at the
    /// other name is an earlier run's, describing weights that are no
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
        let path = Self::path_for_encoding(ckpt, self.encoding);
        let body = serde_json::to_string_pretty(self).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        std::fs::write(&path, body).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let stale = Self::path_for_encoding(ckpt, self.encoding.other());
        match std::fs::remove_file(&stale) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(ShapeError::SweepFailed {
                    written: path,
                    stale,
                    stale_encoding: self.encoding.other(),
                    message: e.to_string(),
                })
            }
        }
        Ok(path)
    }

    /// Read the shape written beside a checkpoint, whichever
    /// conditioning convention it was written under.
    ///
    /// Named for what it does rather than for being the default, so
    /// that [`Self::load_as`] is the shorter thing to reach for. The
    /// one caller that wants this is `chess_bake`'s resume check, which
    /// has to be able to read a checkpoint of either convention in
    /// order to compare it against the run's own.
    pub fn load_any(ckpt: &Path) -> Result<Self, ShapeError> {
        let prefix_path = Self::path_for_encoding(ckpt, CondEncoding::Prefix);
        let other_path = Self::path_for_encoding(ckpt, CondEncoding::EveryPosition);
        // Both present is refused rather than resolved. Preferring
        // either one would answer "which convention is this checkpoint"
        // from a file that may have been written for different weights,
        // and the prefix-preferring version of this was worse than the
        // single-name scheme it replaced: it returned a stale prefix
        // shape for a per-position checkpoint, so `require_encoding`
        // compared the caller against the wrong file and agreed with
        // it. The weight load does not catch that either — the readers
        // go through a mmapped `VarBuilder`, which asks for the names
        // the model wants and never notices `cond_wte.weight` sitting
        // in the file unrequested.
        let path = match (prefix_path.is_file(), other_path.is_file()) {
            (true, true) => {
                return Err(ShapeError::AmbiguousSidecar {
                    first: prefix_path.display().to_string(),
                    second: other_path.display().to_string(),
                })
            }
            (true, false) => prefix_path,
            (false, _) => other_path,
        };
        let body = std::fs::read_to_string(&path).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        serde_json::from_str(&body).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    /// Read the shape and refuse it unless it was conditioned the way
    /// the caller can read.
    ///
    /// This is the entry point for a reader. It cannot be called
    /// without naming what the caller supports, which is the property
    /// [`Self::load_any`] gives up.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn a_shape(encoding: CondEncoding) -> ModelShape {
        let mut shape = ModelShape::compact(
            2048,
            vec![
                ConditionBand {
                    min: 1100,
                    max: 1299,
                    token: "<elo:1100-1299>".into(),
                },
                ConditionBand {
                    min: 1900,
                    max: 2099,
                    token: "<elo:1900-2099>".into(),
                },
            ],
        );
        shape.encoding = encoding;
        shape
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
        let ckpt = tmp.path().join("run.safetensors");
        a_shape(CondEncoding::EveryPosition).save(&ckpt).unwrap();
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
            ModelShape::path_for_encoding(&ckpt, CondEncoding::EveryPosition)
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
        a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
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
            ModelShape::path_for_encoding(&ckpt, CondEncoding::EveryPosition),
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
        let ckpt = tmp.path().join("run.safetensors");
        a_shape(CondEncoding::EveryPosition).save(&ckpt).unwrap();
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
        let ckpt = tmp.path().join("run.safetensors");
        a_shape(CondEncoding::Prefix).save(&ckpt).unwrap();
        let err = ModelShape::load_as(&ckpt, CondEncoding::EveryPosition);
        assert!(matches!(err, Err(ShapeError::EncodingMismatch { .. })));
    }

    /// Every checkpoint baked before the field existed has a sidecar
    /// without it, and every one of those was prefix-conditioned.
    #[test]
    fn a_sidecar_with_no_encoding_field_loads_as_prefix() {
        let tmp = TempDir::new().unwrap();
        let ckpt = tmp.path().join("run.safetensors");
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
