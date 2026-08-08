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
//! The same cut is what makes the legal moves at each of a row's
//! positions hard to recover: the row no longer carries the history the
//! dropped plies would be replayed from. [`Window::legal_sets`] maps a
//! replayed game onto the positions this row does hold, for a reader
//! feeding a checkpoint that was trained with those sets as input.
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
    /// The replayed history does not reach back to the first move this
    /// window kept.
    ///
    /// The window holds the **last** moves of the game, so recovering
    /// the set each of them was drawn from needs the game replayed at
    /// least that far. Aligning a shorter list to whatever it does hold
    /// would attach every set to a different ply than the one it belongs
    /// to.
    #[error(
        "this window keeps {moves_in_window} move(s) and the history covers {plies} ply/plies, \
         so it does not reach the first move the window kept"
    )]
    HistoryTooShort {
        /// Moves this window kept.
        moves_in_window: usize,
        /// Plies the caller said its history covers.
        plies: usize,
    },
    /// The history does not hold one entry per ply plus the position
    /// they lead to, so it is not the history of the game it is said to
    /// be.
    ///
    /// [`Window::legal_sets`] maps by taking the **tail** of the list,
    /// which cannot see a list that is too long: every set would then
    /// describe a position later in the game than the token it is
    /// attached to, and every length downstream would still agree. This
    /// is the check that sees it.
    #[error(
        "a history of {plies} ply/plies has to hold {} legal set(s) — one per ply, plus the \
         position they lead to — and {given} were supplied",
        plies + 1
    )]
    HistoryLengthMismatch {
        /// Plies the caller said its history covers.
        plies: usize,
        /// Entries the caller supplied.
        given: usize,
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

    /// The ids legal at each of this row's positions, in the convention
    /// the training path records them in.
    ///
    /// `at_ply[m]` is the set of ids legal in the position reached after
    /// `m` moves of the game — equivalently, the set the game's `m`-th
    /// move was drawn from — and its last entry is the set the move
    /// about to be played is drawn from. A caller that replayed the game
    /// to the position it is scoring holds exactly that list, one entry
    /// per ply plus the position it is standing on.
    ///
    /// The result has `len() + 1` entries: one per token of the row,
    /// plus one for the position past its end, which is the one the
    /// model answers at. Entry `p` is the set the token at row position
    /// `p` was drawn from, and the prefix positions get the empty set —
    /// they hold no move, which is what
    /// [`crate::chess::corpus::LegalMaskedDataset`] writes there and
    /// what the forward pass reads as "nothing is known to be legal
    /// here".
    ///
    /// # The mapping, and where the off-by-one hides
    ///
    /// A windowed row is `[prefix] + moves[start..]`, so entry `p` is
    /// `at_ply[start + p - prefix_len]` — not `at_ply[p]`, and not
    /// `at_ply[p - prefix_len]`. `start` is `plies` minus the moves this
    /// window kept, so it is zero exactly when the row still holds every
    /// move the history covers. A game that fits whole therefore cannot
    /// tell a correct mapping from one that indexes by row position: on
    /// such a row the two are the same function.
    ///
    /// Nor does `start > 0` make them always disagree. The two read
    /// different plies, and two plies can offer the same moves — a
    /// knight walked out and back reaches the opening position again
    /// every four plies, which is how the first draft of the test for
    /// this passed against a mapping that was wrong.
    ///
    /// What the forward pass reads at a position is this list shifted by
    /// one — [`crate::chess::batch::BandBatch`] applies the same offset
    /// [`crate::train::legal_input_sets`] applies to the same
    /// convention, because the set the model may answer from at position
    /// `p` is the set the token at `p + 1` was drawn from. So on a
    /// conditioned row the **band**, at position 1, is handed entry 2 —
    /// `at_ply[start]`, the board after `start` moves have been played,
    /// which is the opening position only when `start` is zero. Off by
    /// one in either direction the sets are still legal-move sets of
    /// real positions and every length still agrees.
    ///
    /// # Why the ply count is an argument
    ///
    /// The alignment is a tail: `start` is counted back from the end of
    /// `at_ply`. That refuses a history too **short** and would accept
    /// one too **long** — mapping every entry to a position later in the
    /// game than the token it is attached to. Nothing downstream can see
    /// it. The lengths agree, the width check in
    /// [`crate::chess::batch::BandBatch`] passes, the ids are real moves
    /// of real positions, and the model answers.
    ///
    /// So the length is not inferred from `at_ply`. The caller states how
    /// many plies its history is supposed to cover, from its own move
    /// list — the same list the row was built from — and this checks the
    /// two against each other. A caller that accumulates `at_ply` over a
    /// whole game and then scores an earlier position of it gets
    /// [`WindowError::HistoryLengthMismatch`] rather than a wrong set at
    /// every position of the row.
    ///
    /// Passing `at_ply.len() - 1` restores the hole, since that is
    /// exactly the number being checked.
    ///
    /// # Errors
    ///
    /// The history does not hold one entry per ply plus the position they
    /// lead to, or it covers fewer plies than the moves this window kept.
    pub fn legal_sets(
        &self,
        at_ply: &[Vec<u32>],
        plies: usize,
    ) -> Result<Vec<Vec<u32>>, WindowError> {
        // One entry per ply, plus the position they lead to.
        if at_ply.len() != plies + 1 {
            return Err(WindowError::HistoryLengthMismatch {
                plies,
                given: at_ply.len(),
            });
        }
        let moves_in_window = self.tokens.len().saturating_sub(self.prefix_len);
        if plies < moves_in_window {
            return Err(WindowError::HistoryTooShort {
                moves_in_window,
                plies,
            });
        }
        let start = plies - moves_in_window;
        let mut out = vec![Vec::new(); self.prefix_len];
        out.extend(at_ply[start..].iter().cloned());
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
    use crate::chess::corpus::ConditionBand;
    use crate::chess::pgn::{move_from_uci_standard, uci_standard};
    use crate::chess::vocab::MoveVocab;
    use crate::chess::{CondEncoding, ModelShape};
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

    /// One entry per ply, carrying the ply number, so that a mapping
    /// error reads as the ply it landed on rather than as a set of
    /// chess moves that looks like any other.
    fn ply_marks(plies: usize) -> Vec<Vec<u32>> {
        (0..=plies).map(|m| vec![m as u32]).collect()
    }

    /// A row that fits whole: every move keeps the set it was drawn
    /// from, and the prefix holds none.
    #[test]
    fn an_unwindowed_row_maps_each_move_to_its_own_ply() {
        let moves: Vec<u32> = (1000..1010).collect();
        let window = play_row(Some(7), &moves, CTX).unwrap();
        let sets = window
            .legal_sets(&ply_marks(moves.len()), moves.len())
            .unwrap();

        assert_eq!(
            sets.len(),
            window.len() + 1,
            "one per token, plus the answer"
        );
        assert!(sets[0].is_empty(), "BOS holds no move");
        assert!(sets[BAND_POS].is_empty(), "the band holds no move");
        for (i, _) in moves.iter().enumerate() {
            assert_eq!(
                sets[COND_PREFIX_LEN + i],
                vec![i as u32],
                "move {i} should carry the set it was drawn from"
            );
        }
        assert_eq!(
            sets[window.len()],
            vec![moves.len() as u32],
            "the entry past the row is the position the model answers at"
        );
    }

    /// The same on a row the window cut, which is the case the identity
    /// mapping gets wrong: the row's first move is not the game's.
    ///
    /// 130 plies at `ctx` 128 keeps `128 - 2 = 126` of them, so the row
    /// starts at ply 4. A reader that indexed `at_ply` by row position
    /// would hand every position a set four plies stale — still a set of
    /// chess moves, still the right shape, still scored without a word.
    #[test]
    fn a_windowed_row_starts_at_the_ply_the_window_starts_at() {
        let vocab = vocab();
        let moves = knight_shuffle(130, &vocab);
        let window = play_row(Some(band_id(BANDS[0], &vocab)), &moves, CTX).unwrap();
        assert_eq!(window.len(), CTX, "a 130-ply game should be windowed");

        let sets = window
            .legal_sets(&ply_marks(moves.len()), moves.len())
            .unwrap();
        let start = moves.len() - (CTX - COND_PREFIX_LEN);
        assert_eq!(start, 4);
        assert_eq!(
            sets[COND_PREFIX_LEN],
            vec![start as u32],
            "the row's first move was drawn from the board after {start} moves"
        );
        assert_ne!(
            sets[COND_PREFIX_LEN],
            vec![0],
            "indexing by row position would have put the opening here"
        );
        assert_eq!(
            sets[window.len()],
            vec![moves.len() as u32],
            "and the last entry is still the position on the board"
        );
    }

    /// A history that does not reach the first move the window kept is
    /// refused rather than aligned to whatever it does hold.
    #[test]
    fn a_history_shorter_than_the_window_is_refused() {
        let moves: Vec<u32> = (1000..1010).collect();
        let window = play_row(Some(7), &moves, CTX).unwrap();
        let short = moves.len() - 1;
        assert_eq!(
            window.legal_sets(&ply_marks(short), short),
            Err(WindowError::HistoryTooShort {
                moves_in_window: 10,
                plies: 9,
            })
        );
        // And the exact length is enough: 10 moves need 11 entries.
        assert!(window
            .legal_sets(&ply_marks(moves.len()), moves.len())
            .is_ok());
    }

    /// The other direction, which the tail alignment cannot see on its
    /// own: a history covering **more** plies than the row was built
    /// from.
    ///
    /// This is the failure the mapping exists to prevent and the one
    /// nothing downstream can catch — the lengths still agree, the sets
    /// are still real positions', and the model still answers. The
    /// obvious refactor produces it: accumulate `at_ply` for a whole
    /// game once, then score each position out of the same list.
    #[test]
    fn a_history_longer_than_the_row_was_built_from_is_refused() {
        let moves: Vec<u32> = (1000..1010).collect();
        let window = play_row(Some(7), &moves, CTX).unwrap();
        assert_eq!(
            window.legal_sets(&ply_marks(20), moves.len()),
            Err(WindowError::HistoryLengthMismatch {
                plies: 10,
                given: 21,
            })
        );
    }

    /// And what that refusal stands in front of, which is why the ply
    /// count has to come from the caller's own move list.
    ///
    /// The same over-long history, accepted by a caller that also
    /// mis-states the count: the row's first move is handed the board
    /// after ten moves rather than after none, every length agrees, and
    /// nothing says a word.
    #[test]
    fn a_mis_stated_ply_count_maps_the_row_to_the_wrong_plies() {
        let moves: Vec<u32> = (1000..1010).collect();
        let window = play_row(Some(7), &moves, CTX).unwrap();
        let honest = window
            .legal_sets(&ply_marks(moves.len()), moves.len())
            .unwrap();
        let mis_stated = window.legal_sets(&ply_marks(20), 20).unwrap();
        assert_eq!(
            honest.len(),
            mis_stated.len(),
            "the lengths agree, which is why nothing downstream sees this"
        );
        assert_eq!(honest[COND_PREFIX_LEN], vec![0]);
        assert_eq!(mis_stated[COND_PREFIX_LEN], vec![10]);
    }

    /// An unconditioned row has one prefix token rather than two, and
    /// the mapping is derived from the window's own prefix rather than
    /// from a constant.
    #[test]
    fn an_unconditioned_row_maps_from_its_own_prefix() {
        let moves: Vec<u32> = (1000..1010).collect();
        let window = play_row(None, &moves, CTX).unwrap();
        let sets = window
            .legal_sets(&ply_marks(moves.len()), moves.len())
            .unwrap();
        assert_eq!(sets.len(), window.len() + 1);
        assert!(sets[0].is_empty(), "BOS holds no move");
        assert_eq!(
            sets[PLAIN_PREFIX_LEN],
            vec![0],
            "the first move sits one token in, not two"
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

    /// The same question under per-position conditioning, at the same
    /// ply, because the interactive path reaches this regime and has no
    /// bound at all — about one human game in 26 passes 126 plies.
    ///
    /// Two assertions:
    ///
    /// - the **same** windowed row under two different conditioning rows
    ///   gives different logits, so the band arrives through a channel
    ///   windowing cannot touch. Under the prefix convention the band is
    ///   token 1 and a tail slice can eat it; here it is an argument,
    ///   and there is no position for a window to disturb;
    /// - and dropping the condition changes the answer, so that a reader
    ///   dropping it is detectable at all.
    ///
    /// **What this does not test** is any reader. It builds a model in
    /// place and calls the two forwards directly, so reverting an
    /// example to a plain `forward` leaves it green. The construction a
    /// reader performs is tested in [`crate::chess::batch`], which is
    /// where that construction now lives for exactly this reason.
    ///
    /// Built through `ModelShape::config` and `ModelShape::band_index`
    /// rather than by hand, so it exercises the same two producers the
    /// readers use: a table sized from the band list, and the only way
    /// to obtain a `CondIndex` from outside the model code.
    #[test]
    fn the_condition_reaches_a_per_position_model_at_ply_130() {
        let vocab = vocab();
        let moves = knight_shuffle(130, &vocab);
        let low_id = band_id(BANDS[0], &vocab);
        let window = play_row(Some(low_id), &moves, CTX).unwrap();
        assert_eq!(window.len(), CTX, "a 130-ply game should be windowed");
        assert_eq!(window.band(), Some(low_id));

        let mut shape = ModelShape::compact(
            vocab.model_vocab_size(),
            BANDS
                .iter()
                .map(|token| ConditionBand {
                    min: 0,
                    max: 0,
                    token: (*token).to_string(),
                })
                .collect(),
        );
        shape.encoding = CondEncoding::EveryPosition;
        let cfg = shape.config(Device::Cpu, DType::F32);
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).unwrap();
        // Seeded, so the thresholds below are a property of these
        // weights rather than of whatever the initialiser drew.
        fill_deterministic(&varmap, 0x0806_2026).unwrap();

        // The same row twice. Only the conditioning argument differs, so
        // whatever separates the two outputs came through it.
        let low = shape.band_index(BANDS[0]).expect("a row for the low band");
        let high = shape.band_index(BANDS[1]).expect("a row for the high band");
        assert_ne!(low.row(), high.row(), "two bands, two table rows");
        let mut rows = window.tokens().to_vec();
        rows.extend_from_slice(window.tokens());
        let input = Tensor::from_vec(rows, (2, CTX), &cfg.device).unwrap();

        let conditioned = model.forward_conditioned(&input, &[low, high]).unwrap();
        let spread = max_abs_diff_f32(
            &conditioned.i((0, CTX - 1)).unwrap(),
            &conditioned.i((1, CTX - 1)).unwrap(),
        )
        .unwrap();
        assert!(
            spread > 1e-5,
            "the conditioning argument did nothing at ply 130 (spread {spread})"
        );

        let plain = model.forward(&input).unwrap();
        let dropped = max_abs_diff_f32(
            &conditioned.i((0, CTX - 1)).unwrap(),
            &plain.i((0, CTX - 1)).unwrap(),
        )
        .unwrap();
        assert!(
            dropped > 1e-5,
            "dropping the condition changed nothing, so nothing here could catch a reader \
             that drops it (spread {dropped})"
        );
    }
}
