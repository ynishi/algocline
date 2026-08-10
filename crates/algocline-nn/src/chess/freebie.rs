//! Captures that cannot be recaptured.
//!
//! Plan 05 measures a criterion this project set for itself on
//! 2026-08-05 and then carried for five days without measuring:
//!
//! > 駒をタダで渡しても取らない相手は、どんな指標が良くても相手として
//! > 成立しない
//!
//! The evidence behind it was three positions from one hand-played game
//! against a checkpoint trained on 236,800 rows. This module turns the
//! criterion into a rate over a position set, so that a later
//! checkpoint can be held to it rather than inheriting the verdict.
//!
//! # What "free" means here, exactly
//!
//! A capture is free when **no legal move in the resulting position
//! lands on the square that was captured on**. Recapture is asked as a
//! question about legality rather than about attack maps, which is the
//! stronger form for this purpose:
//!
//! - a pinned defender does not make a capture unfree, because it
//!   cannot legally recapture
//! - a king does not defend a square it may not legally move to
//!
//! Both fall out of the legal move generator rather than needing a
//! special case.
//!
//! # What it does not mean
//!
//! Not "winning", and not static exchange evaluation. A capture that is
//! free by this definition can still lose to a bigger reply — a
//! discovered attack on the capturing piece, a mate threat, a knight
//! fork the capture walks into. The definition is narrow on purpose: it
//! is decidable from the board with no evaluation function, and the
//! criterion it is standing in for ("takes a piece handed over for
//! nothing") is narrow in the same way.
//!
//! A model measured as taking these is not thereby a good player. It is
//! a model that clears the specific bar of 2026-08-05.

use cozy_chess::{Board, Move, Piece};

/// Material value of a piece, on the scale `match-design.md §4` fixes
/// for adjudicating a game at the ply cap: `Q=9 R=5 B=N=3 P=1`.
///
/// The king has no value here — it is never a capture victim in a legal
/// position, and giving it one would let a bug produce a number instead
/// of an absurdity.
#[must_use]
pub fn piece_value(piece: Piece) -> u32 {
    match piece {
        Piece::Pawn => 1,
        Piece::Knight | Piece::Bishop => 3,
        Piece::Rook => 5,
        Piece::Queen => 9,
        Piece::King => 0,
    }
}

/// Lowest victim value plan 05 judges on.
///
/// Minor pieces and up. A free pawn is common and one point is a margin
/// a policy can decline on taste — declining a whole knight is the
/// thing 2026-08-05 was about. Pawn captures are still returned, and
/// the caller reports them separately.
pub const JUDGED_VALUE: u32 = 3;

/// A capture whose square nothing can legally recapture on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeCapture {
    /// The move, in the UCI spelling the vocabulary uses.
    pub mv: Move,
    /// What it takes.
    pub victim: Piece,
    /// [`piece_value`] of the victim.
    pub value: u32,
}

/// Every free capture available to the side to move.
///
/// One legal move generation for the position, and one more per
/// capture. Captures are a small fraction of a position's moves, so
/// this costs a small multiple of a single movegen rather than a
/// multiple of the branching factor.
///
/// # En passant
///
/// Detected by shape — a pawn changing file onto an empty square — and
/// reported with a pawn victim. It never reaches the judged set
/// ([`JUDGED_VALUE`]), but leaving it out would understate the pawn
/// tally that is reported beside it.
///
/// The freeness test asks about `mv.to`, which for en passant is the
/// square the capturing pawn ends on rather than the one the captured
/// pawn stood on. That is the square a recapture would have to reach,
/// so it is the right one.
#[must_use]
pub fn free_captures(board: &Board) -> Vec<FreeCapture> {
    let mut out = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            let Some(victim) = victim_of(board, mv) else {
                continue;
            };
            let mut next = board.clone();
            next.play_unchecked(mv);
            if !recapturable(&next, mv.to) {
                out.push(FreeCapture {
                    mv,
                    victim,
                    value: piece_value(victim),
                });
            }
        }
        false
    });
    out
}

/// What a move captures, or `None` when it captures nothing.
fn victim_of(board: &Board, mv: Move) -> Option<Piece> {
    if let Some(piece) = board.piece_on(mv.to) {
        return Some(piece);
    }
    // An empty destination is still a capture when a pawn changes file:
    // the only way that happens is en passant.
    let moved = board.piece_on(mv.from)?;
    if moved == Piece::Pawn && mv.from.file() != mv.to.file() {
        return Some(Piece::Pawn);
    }
    None
}

/// Whether the side to move has any legal move onto `square`.
///
/// Legality rather than attack: a pinned piece attacks the square and
/// cannot go there, and that is exactly the case where the capture is
/// free despite looking defended.
fn recapturable(board: &Board, square: cozy_chess::Square) -> bool {
    let mut found = false;
    board.generate_moves(|moves| {
        for mv in moves {
            if mv.to == square {
                found = true;
                return true;
            }
        }
        false
    });
    found
}

/// The free captures worth at least [`JUDGED_VALUE`].
#[must_use]
pub fn judged_free_captures(board: &Board) -> Vec<FreeCapture> {
    free_captures(board)
        .into_iter()
        .filter(|c| c.value >= JUDGED_VALUE)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn board(fen: &str) -> Board {
        Board::from_str(fen).expect("a legal fen")
    }

    fn ucis(captures: &[FreeCapture]) -> Vec<String> {
        let mut out: Vec<String> = captures.iter().map(|c| c.mv.to_string()).collect();
        out.sort();
        out
    }

    /// A knight standing on a square nothing covers is free.
    #[test]
    fn an_undefended_knight_is_a_free_capture() {
        // Black knight on e5, White pawn on d4 able to take it, and
        // nothing of Black's covering e5.
        let b = board("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1");
        let free = judged_free_captures(&b);
        assert_eq!(ucis(&free), vec!["d4e5"], "{free:?}");
        assert_eq!(free[0].victim, Piece::Knight);
        assert_eq!(free[0].value, 3);
    }

    /// The same knight, defended, is not.
    #[test]
    fn a_defended_knight_is_not_a_free_capture() {
        // Black pawn on f6 recaptures on e5.
        let b = board("4k3/8/5p2/4n3/3P4/8/8/4K3 w - - 0 1");
        assert!(judged_free_captures(&b).is_empty());
    }

    /// A defender that is pinned to its own king cannot recapture, so
    /// the capture is free. This is the case an attack-map test gets
    /// wrong and a legality test gets right.
    #[test]
    fn a_pinned_defender_does_not_make_a_capture_unfree() {
        // Black bishop g5 is defended by the knight on e6 — and that
        // knight is pinned to the king on e8 by the rook on e1, so it
        // cannot recapture. `Bxg5` is therefore free although g5 is
        // attacked.
        //
        // `Rxe6` is free as well and the assertion says so: the knight
        // is covered by nothing, since g5's diagonals run f6-e7-d8 and
        // f4-e3, never e6. Asserting the whole set rather than a
        // membership keeps a second free capture from appearing later
        // without the test noticing.
        let b = board("4k3/8/4n3/6b1/7B/8/8/4R1K1 w - - 0 1");
        let free = judged_free_captures(&b);
        assert_eq!(ucis(&free), vec!["e1e6", "h4g5"], "{free:?}");
        let pinned_case = free
            .iter()
            .find(|c| c.mv.to_string() == "h4g5")
            .expect("the capture the pin makes free");
        assert_eq!(pinned_case.victim, Piece::Bishop);
    }

    /// A king does not defend a square it cannot legally move to.
    #[test]
    fn a_king_that_may_not_recapture_does_not_defend() {
        // The black rook on d7 is defended by its king alone. After
        // either capture the king may not recapture, because the other
        // white piece covers d7 — the rook along the file, the bishop
        // along h3-d7. Both captures are therefore free.
        //
        // The earlier version of this test used the rook alone and
        // asserted the same thing, which was wrong: the capturing rook
        // is the piece standing on d7, so `Kxd7` takes it and walks
        // into nothing.
        let b = board("4k3/3r4/8/8/8/7B/8/3RK3 w - - 0 1");
        let free = judged_free_captures(&b);
        assert_eq!(ucis(&free), vec!["d1d7", "h3d7"], "{free:?}");
        for capture in &free {
            assert_eq!(capture.victim, Piece::Rook);
            assert_eq!(capture.value, 5);
        }
    }

    /// The opening position has no captures at all, free or otherwise.
    #[test]
    fn the_opening_position_has_none() {
        assert!(free_captures(&Board::default()).is_empty());
        assert!(judged_free_captures(&Board::default()).is_empty());
    }

    /// Pawns are found but do not reach the judged set.
    #[test]
    fn a_free_pawn_is_found_and_not_judged() {
        let b = board("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1");
        let all = free_captures(&b);
        assert_eq!(ucis(&all), vec!["d4e5"], "{all:?}");
        assert_eq!(all[0].victim, Piece::Pawn);
        assert_eq!(all[0].value, 1);
        assert!(judged_free_captures(&b).is_empty());
    }

    /// The king carries no value, so a bug that let it be a victim
    /// would produce a zero rather than a number in the material scale.
    #[test]
    fn the_king_has_no_material_value() {
        assert_eq!(piece_value(Piece::King), 0);
        assert!(piece_value(Piece::King) < JUDGED_VALUE);
    }
}
