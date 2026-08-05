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
pub mod pgn;
pub mod vocab;

use candle_core::{DType, Device};

use crate::arch::Gpt2Config;

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

/// The model shape this ingestion path is sized for.
///
/// Both the bake and the player must build the same shape or the
/// checkpoint reloads into a different model, so the sizing lives here
/// rather than being repeated at each call site.
pub fn model_config(vocab_size: usize, device: Device, dtype: DType) -> Gpt2Config {
    Gpt2Config {
        layers: LAYERS,
        heads: HEADS,
        dim: DIM,
        ctx: CTX,
        vocab: vocab_size,
        dtype,
        device,
        eps: 1e-5,
        moe: None,
        custom: None,
    }
}
