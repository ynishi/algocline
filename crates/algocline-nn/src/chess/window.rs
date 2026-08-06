//! Fitting a game into the context window without losing what is in
//! front of it.
//!
//! A conditioned row is `[BOS, band] + moves`, and a game that runs
//! past the context window has to be cut down before it can be fed to
//! the model. Cutting it to the **last** `ctx` tokens is the obvious
//! way and the wrong one: the two tokens it drops first are exactly the
//! two that are not moves. What the model then sees is a row that never
//! occurred in training — no `BOS`, no band — and every band sees the
//! same row, so the condition provably does nothing at those positions.
//!
//! Worse, the row's shape is unchanged, so a caller that believes
//! position 1 holds the band goes on believing it. `chess_play`'s
//! guidance path wrote a band id there at any gamma other than one,
//! which after a tail slice overwrites a real move in the position
//! being evaluated (issue `8f9a96df`).
//!
//! So the prefix is kept and the moves are windowed:
//! `[BOS, band] + tail(ctx - 2)`.
//!
//! How often this matters is a property of the caller, not of the
//! corpus. Training drops overlong games outright, and held-out replay
//! reaches the regime in 1 position of 3,000 on 2026-05 and 27 of 3,000
//! on 2026-04. The playing path has no bound at all: 3.8% of games —
//! about one in 26 — do not fit whole in a 128-token context, with a
//! ply p90 of 109 and a maximum of 276.
//!
//! The unconditioned case is the same repair with a one-token prefix,
//! since `BOS` is no more optional than the band is.
//!
//! [`Window`] exists so that the prefix kind travels with the tokens.
//! A bare `Vec<u32>` cannot say whether position 1 is a band or a move,
//! which is the ambiguity the original defect lived in — a function
//! that writes a band into "position 1" of an unconditioned row
//! reproduces the bug inside a name that claims to prevent it.

use thiserror::Error;

use crate::chess::vocab::BOS;

/// Where the condition token sits in a conditioned row, behind `BOS`.
pub const BAND_POS: usize = 1;

/// Tokens a conditioned row carries before its first move: `BOS` and
/// the band.
pub const COND_PREFIX_LEN: usize = 2;

/// Tokens an unconditioned row carries before its first move: `BOS`.
pub const PLAIN_PREFIX_LEN: usize = 1;

/// Why a row could not be windowed, or a window could not be rewritten.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WindowError {
    /// The row is shorter than the prefix it is supposed to carry, so
    /// the caller and this function disagree about what is in it.
    #[error("row of {len} token(s) is shorter than its {prefix_len}-token prefix")]
    RowShorterThanPrefix {
        /// Length of the row handed over.
        len: usize,
        /// Prefix the caller declared.
        prefix_len: usize,
    },
    /// The context window has no room for a move once the prefix is
    /// kept. Returned rather than silently dropping the prefix, which
    /// is the behaviour this module exists to remove.
    #[error("ctx {ctx} leaves no room for a move beside a {prefix_len}-token prefix")]
    CtxTooSmall {
        /// Context window in tokens.
        ctx: usize,
        /// Prefix the caller declared.
        prefix_len: usize,
    },
    /// A band was written into a window that carries no condition. The
    /// slot it would have taken holds a move of the position being
    /// evaluated.
    #[error(
        "this window carries a {prefix_len}-token prefix, so it has no condition to \
         replace: position {BAND_POS} holds a move"
    )]
    NotConditioned {
        /// Prefix the window was built with.
        prefix_len: usize,
    },
}

/// A row cut to the context window, carrying how much of its front is
/// prefix rather than moves.
///
/// The pairing is the point. Every operation that addresses the prefix
/// by position — reading the band, replacing it — is a method here, so
/// it is checked against the row it is actually applied to rather than
/// against a `prefix_len` the call site worked out somewhere else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    tokens: Vec<u32>,
    prefix_len: usize,
}

impl Window {
    /// The tokens to feed the model.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Length in tokens, never more than the `ctx` it was built for.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the window is empty. Never true for a window this module
    /// builds — every prefix is at least `BOS` — but `clippy` asks for
    /// it beside `len`.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens before the first move.
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Whether this row carries a condition at all.
    pub fn is_conditioned(&self) -> bool {
        self.prefix_len == COND_PREFIX_LEN
    }

    /// The condition this row carries, if it carries one.
    pub fn band(&self) -> Option<u32> {
        self.is_conditioned()
            .then(|| self.tokens.get(BAND_POS).copied())
            .flatten()
    }

    /// Replace the condition, leaving every move untouched.
    ///
    /// This is what guidance does: it runs the same position under
    /// every band to build the reference it extrapolates away from, and
    /// the rows differ only here.
    ///
    /// # Errors
    ///
    /// The window carries no condition, so there is nothing to replace
    /// and the write would land on a move.
    pub fn set_band(&mut self, band_id: u32) -> Result<(), WindowError> {
        if !self.is_conditioned() {
            return Err(WindowError::NotConditioned {
                prefix_len: self.prefix_len,
            });
        }
        match self.tokens.get_mut(BAND_POS) {
            Some(slot) => {
                *slot = band_id;
                Ok(())
            }
            // Unreachable through this module's constructors, which
            // never build a conditioned window shorter than its
            // prefix. Returned rather than indexed so that a future
            // constructor cannot turn it into a panic.
            None => Err(WindowError::NotConditioned {
                prefix_len: self.prefix_len,
            }),
        }
    }

    /// A copy of this window under a different condition.
    ///
    /// # Errors
    ///
    /// As [`Self::set_band`].
    pub fn with_band(&self, band_id: u32) -> Result<Self, WindowError> {
        let mut out = self.clone();
        out.set_band(band_id)?;
        Ok(out)
    }

    /// Take the tokens, for a caller that needs to own them.
    pub fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }
}

/// Cut `row` down to `ctx` tokens, keeping its first `prefix_len`.
///
/// A row that already fits is returned unchanged. Anything longer keeps
/// its prefix and takes the **last** `ctx - prefix_len` moves, so the
/// model reads the most recent play with the condition still attached.
///
/// Crate-private, with [`play_row`] the only public constructor. The
/// `prefix_len` here is an untyped `usize` that nothing checks against
/// the row: passing [`COND_PREFIX_LEN`] for a row that carries no band
/// mints a `Window` claiming a condition it does not have, and
/// [`Window::set_band`] would then write over a move — the exact
/// ambiguity the type exists to remove, one call away. `play_row`
/// derives the length from the band itself, so it cannot be reached
/// that way.
///
/// A raw form can come back if something needs it, as
/// `Window::conditioned` / `Window::plain` rather than as a number.
///
/// # Errors
///
/// - The row is shorter than `prefix_len`.
/// - `ctx` leaves no room for a move once the prefix is kept.
pub(crate) fn window_row(
    row: &[u32],
    ctx: usize,
    prefix_len: usize,
) -> Result<Window, WindowError> {
    if row.len() < prefix_len {
        return Err(WindowError::RowShorterThanPrefix {
            len: row.len(),
            prefix_len,
        });
    }
    if ctx <= prefix_len {
        return Err(WindowError::CtxTooSmall { ctx, prefix_len });
    }
    if row.len() <= ctx {
        return Ok(Window {
            tokens: row.to_vec(),
            prefix_len,
        });
    }
    let keep_moves = ctx - prefix_len;
    let mut tokens = Vec::with_capacity(ctx);
    tokens.extend_from_slice(&row[..prefix_len]);
    tokens.extend_from_slice(&row[row.len() - keep_moves..]);
    Ok(Window { tokens, prefix_len })
}

/// Build the row a model plays from, and cut it to `ctx`.
///
/// `[BOS] + band? + moves`, windowed. The presence of the band decides
/// the prefix length, so the two cannot disagree — which they could
/// while each call site worked the length out for itself.
///
/// `BOS` is not a parameter: there is one, it is
/// [`crate::chess::vocab::BOS`], and passing it in would only create
/// the opportunity to pass something else.
///
/// # Errors
///
/// As [`window_row`]: `ctx` has to leave room for at least one move.
pub fn play_row(band: Option<u32>, moves: &[u32], ctx: usize) -> Result<Window, WindowError> {
    let mut row = Vec::with_capacity(moves.len() + COND_PREFIX_LEN);
    row.push(BOS);
    let prefix_len = match band {
        Some(id) => {
            row.push(id);
            COND_PREFIX_LEN
        }
        None => PLAIN_PREFIX_LEN,
    };
    row.extend_from_slice(moves);
    window_row(&row, ctx, prefix_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::fill_deterministic;
    use crate::arch::{max_abs_diff_f32, Gpt2Config, Gpt2Model};
    use crate::chess::pgn::{move_from_uci_standard, uci_standard};
    use crate::chess::vocab::MoveVocab;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use cozy_chess::Board;

    const CTX: usize = 128;
    const BANDS: [&str; 2] = ["<elo:1100-1299>", "<elo:1900-2099>"];

    fn vocab() -> MoveVocab {
        MoveVocab::new(&BANDS.map(String::from)).unwrap()
    }

    /// A real game of `plies` moves, as vocabulary ids.
    ///
    /// Both sides walk a knight out and back, which is legal from the
    /// opening position for as long as anyone cares to continue — the
    /// point is to reach ply 130 through actual move generation rather
    /// than to invent ids that happen to be in range.
    fn knight_shuffle(plies: usize, vocab: &MoveVocab) -> Vec<u32> {
        let cycle = ["b1c3", "b8c6", "c3b1", "c6b8"];
        let mut board = Board::default();
        let mut ids = Vec::with_capacity(plies);
        for ply in 0..plies {
            let uci = cycle[ply % cycle.len()];
            let mv = move_from_uci_standard(&board, uci)
                .unwrap_or_else(|| panic!("{uci} is not legal at ply {ply}"));
            let played = uci_standard(&board, mv);
            board.play_unchecked(mv);
            ids.push(vocab.id_of(&played).expect("move in the vocabulary"));
        }
        ids
    }

    fn band_id(token: &str, vocab: &MoveVocab) -> u32 {
        vocab.id_of(token).expect("band in the vocabulary")
    }

    /// `[BOS, band] + moves`, unwindowed, for the assertions that
    /// compare against what a tail slice would have produced.
    fn conditioned_row(band: &str, moves: &[u32], vocab: &MoveVocab) -> Vec<u32> {
        let mut row = vec![BOS, band_id(band, vocab)];
        row.extend_from_slice(moves);
        row
    }

    #[test]
    fn a_row_that_fits_is_returned_whole() {
        let row = vec![BOS, 7, 100, 101, 102];
        let window = window_row(&row, CTX, COND_PREFIX_LEN).unwrap();
        assert_eq!(window.tokens(), row);
        assert_eq!(window.band(), Some(7));
    }

    #[test]
    fn a_long_row_keeps_its_prefix_and_its_most_recent_moves() {
        let mut row = vec![BOS, 7];
        row.extend(1000u32..1200);
        let window = window_row(&row, CTX, COND_PREFIX_LEN).unwrap();
        assert_eq!(window.len(), CTX);
        assert_eq!(window.tokens()[0], BOS);
        assert_eq!(window.band(), Some(7));
        assert_eq!(
            window.tokens()[2..],
            row[row.len() - (CTX - COND_PREFIX_LEN)..]
        );
    }

    #[test]
    fn an_unconditioned_row_keeps_bos_alone() {
        let mut row = vec![BOS];
        row.extend(1000u32..1200);
        let window = window_row(&row, CTX, PLAIN_PREFIX_LEN).unwrap();
        assert_eq!(window.len(), CTX);
        assert_eq!(window.tokens()[0], BOS);
        assert_eq!(window.band(), None);
        assert_eq!(
            window.tokens()[1..],
            row[row.len() - (CTX - PLAIN_PREFIX_LEN)..]
        );
    }

    #[test]
    fn a_ctx_with_no_room_for_a_move_is_refused() {
        let row = vec![BOS, 7, 100];
        assert_eq!(
            window_row(&row, COND_PREFIX_LEN, COND_PREFIX_LEN),
            Err(WindowError::CtxTooSmall {
                ctx: COND_PREFIX_LEN,
                prefix_len: COND_PREFIX_LEN,
            })
        );
    }

    #[test]
    fn a_row_shorter_than_its_prefix_is_refused() {
        assert_eq!(
            window_row(&[BOS], CTX, COND_PREFIX_LEN),
            Err(WindowError::RowShorterThanPrefix {
                len: 1,
                prefix_len: COND_PREFIX_LEN,
            })
        );
    }

    /// `play_row`'s conditioned branch, which is what `chess_play` and
    /// `chess_cond` both go through.
    #[test]
    fn play_row_conditioned_carries_the_band() {
        let moves: Vec<u32> = (1000..1004).collect();
        let window = play_row(Some(7), &moves, CTX).unwrap();
        assert_eq!(window.tokens(), [BOS, 7, 1000, 1001, 1002, 1003]);
        assert!(window.is_conditioned());
        assert_eq!(window.band(), Some(7));
    }

    /// And its unconditioned branch, which is the one no test reached
    /// while the choice lived at the call site.
    #[test]
    fn play_row_unconditioned_has_no_band_to_set() {
        let moves: Vec<u32> = (1000..1004).collect();
        let mut window = play_row(None, &moves, CTX).unwrap();
        assert_eq!(window.tokens(), [BOS, 1000, 1001, 1002, 1003]);
        assert!(!window.is_conditioned());
        assert_eq!(window.band(), None);
        // The regression this type exists for: on an unconditioned row
        // position 1 is a move, and writing a band there corrupts the
        // position rather than conditioning it.
        assert_eq!(
            window.set_band(7),
            Err(WindowError::NotConditioned {
                prefix_len: PLAIN_PREFIX_LEN
            })
        );
        assert_eq!(window.tokens()[BAND_POS], 1000, "the move must survive");
    }

    #[test]
    fn play_row_windows_a_long_game_on_either_branch() {
        let moves: Vec<u32> = (1000..1200).collect();
        let conditioned = play_row(Some(7), &moves, CTX).unwrap();
        assert_eq!(conditioned.len(), CTX);
        assert_eq!(conditioned.band(), Some(7));
        let plain = play_row(None, &moves, CTX).unwrap();
        assert_eq!(plain.len(), CTX);
        assert_eq!(plain.tokens()[0], BOS);
    }

    /// Phase 0-3, at ply 130 — past the 126 plies at which a
    /// conditioned row outgrows the window.
    ///
    /// The second assertion is the regression: under the tail slice
    /// this replaces, position 1 of the window held a *move*, so the
    /// two bands' rows were identical and nothing about the condition
    /// could have been measured there.
    #[test]
    fn the_band_survives_a_game_of_130_plies() {
        let vocab = vocab();
        let moves = knight_shuffle(130, &vocab);
        let low = band_id(BANDS[0], &vocab);
        let row = conditioned_row(BANDS[0], &moves, &vocab);
        assert!(row.len() > CTX, "a 130-ply game should not fit in {CTX}");

        let window = play_row(Some(low), &moves, CTX).unwrap();
        assert_eq!(window.len(), CTX);
        assert_eq!(window.tokens()[0], BOS);
        assert_eq!(window.band(), Some(low));

        let tail = &row[row.len() - CTX..];
        assert_ne!(
            tail[BAND_POS], low,
            "the tail slice this replaces is supposed to have lost the band"
        );
    }

    /// The other half of the same defect: with the band back at
    /// position 1, guidance's rewrite lands on the condition instead of
    /// eating a move out of the position being evaluated.
    #[test]
    fn setting_the_band_lands_on_the_condition_not_on_a_move() {
        let vocab = vocab();
        let moves = knight_shuffle(130, &vocab);
        let window = play_row(Some(band_id(BANDS[0], &vocab)), &moves, CTX).unwrap();

        let high = band_id(BANDS[1], &vocab);
        let rewritten = window.with_band(high).unwrap();
        assert_eq!(rewritten.band(), Some(high));
        assert_eq!(
            rewritten.tokens()[2..],
            window.tokens()[2..],
            "the moves must be untouched by a band rewrite"
        );

        // Against the tail slice: the id the same write would have
        // destroyed is a move the position depends on.
        let row = conditioned_row(BANDS[0], &moves, &vocab);
        let tail = &row[row.len() - CTX..];
        let clobbered = vocab.token_of(tail[BAND_POS]).unwrap();
        assert!(
            !BANDS.contains(&clobbered),
            "the tail slice held {clobbered} at the condition slot, which the rewrite overwrote"
        );
    }

    /// The measurement this makes possible: at ply 130 the two bands
    /// now produce different logits, because their rows now differ.
    /// Before the repair the two rows were the same 128 tokens and the
    /// divergence was zero by construction.
    #[test]
    fn two_bands_differ_at_ply_130() {
        let vocab = vocab();
        let moves = knight_shuffle(130, &vocab);
        let low = play_row(Some(band_id(BANDS[0], &vocab)), &moves, CTX).unwrap();
        let high = low.with_band(band_id(BANDS[1], &vocab)).unwrap();
        assert_ne!(low, high, "the two rows must differ at the band");

        let cfg = Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 16,
            ctx: CTX,
            vocab: vocab.model_vocab_size(),
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        // Seeded, so the threshold below is a property of these weights
        // rather than of whatever the initialiser happened to draw.
        fill_deterministic(&varmap, 0x0803_2026).unwrap();

        let mut rows = low.into_tokens();
        rows.extend(high.into_tokens());
        let input = Tensor::from_vec(rows, (2, CTX), &cfg.device).unwrap();
        let out = model.forward(&input).unwrap();
        let spread =
            max_abs_diff_f32(&out.i((0, CTX - 1)).unwrap(), &out.i((1, CTX - 1)).unwrap()).unwrap();
        assert!(
            spread > 1e-5,
            "the bands produced the same logits at ply 130 (spread {spread})"
        );
    }
}
