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

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::arch::Gpt2Config;
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

    /// Build the model config this shape describes.
    pub fn config(&self, device: Device, dtype: DType) -> Gpt2Config {
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
            custom: None,
        }
    }

    /// The path a checkpoint's shape file sits at.
    pub fn path_for(ckpt: &Path) -> PathBuf {
        ckpt.with_extension("shape.json")
    }

    /// Write the shape beside a checkpoint.
    pub fn save(&self, ckpt: &Path) -> Result<PathBuf, ShapeError> {
        let path = Self::path_for(ckpt);
        let body = serde_json::to_string_pretty(self).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        std::fs::write(&path, body).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Ok(path)
    }

    /// Read the shape written beside a checkpoint.
    pub fn load(ckpt: &Path) -> Result<Self, ShapeError> {
        let path = Self::path_for(ckpt);
        let body = std::fs::read_to_string(&path).map_err(|e| ShapeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        serde_json::from_str(&body).map_err(|e| ShapeError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }
}
