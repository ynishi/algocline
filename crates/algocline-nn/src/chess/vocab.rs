//! The token alphabet for chess move sequences.
//!
//! # Why the alphabet is generated, not collected
//!
//! The obvious way to build a vocabulary is to read a corpus and keep
//! what appears in it. That makes the alphabet a property of the slice
//! that happened to be read: a different month, or a longer prefix,
//! shifts every token id, and a checkpoint trained under one slice
//! silently mismatches a corpus built from another.
//!
//! Chess does not need that. A move in UCI is a pair of squares plus an
//! optional promotion piece, and which pairs can ever occur is fixed by
//! the board — a queen ray or a knight jump. Enumerating that gives an
//! alphabet that depends on nothing but the geometry, so token ids are
//! stable across corpora and across runs.
//!
//! Measured against a 23,989-game slice of the Lichess 2026-06
//! database, the games there use 1,879 distinct UCI moves, of which
//! only 17 appear exactly once. The alphabet is effectively saturated
//! at that corpus size — see `examples/pgn_san_check.rs`.
//!
//! # Layout
//!
//! Ids are assigned in blocks so a caller can reason about them without
//! consulting the table:
//!
//! ```text
//! 0            PAD
//! 1            BOS
//! 2 ..         caller-supplied condition tokens (rating band, ...)
//! .. len()     moves, in a fixed enumeration order
//! ```
//!
//! Condition tokens sit in front of the moves because that is where a
//! conditional model reads them: a band token prefixed to the game
//! makes the rest of the sequence conditional on it.

use std::collections::HashMap;

use thiserror::Error;

/// Padding id. Rows shorter than the context window are filled with it.
pub const PAD: u32 = 0;
/// Beginning-of-game id, emitted before the first move of every game.
pub const BOS: u32 = 1;
/// Number of ids reserved before the condition block.
const RESERVED: u32 = 2;

/// Failure while building a vocabulary.
#[derive(Debug, Error)]
pub enum VocabError {
    /// The same condition token was supplied twice.
    #[error("vocab: duplicate condition token {token:?}")]
    DuplicateCondition {
        /// The repeated token.
        token: String,
    },
    /// A condition token collides with a move token.
    #[error("vocab: condition token {token:?} is also a move")]
    ConditionIsMove {
        /// The colliding token.
        token: String,
    },
}

/// A fixed mapping between move strings and token ids.
#[derive(Debug, Clone)]
pub struct MoveVocab {
    tokens: Vec<String>,
    index: HashMap<String, u32>,
    condition_count: usize,
}

impl MoveVocab {
    /// Build the alphabet, with `conditions` occupying the block
    /// between the reserved ids and the moves.
    ///
    /// Pass an empty slice for an unconditional model.
    pub fn new(conditions: &[String]) -> Result<Self, VocabError> {
        let mut tokens = vec!["<pad>".to_string(), "<bos>".to_string()];
        let mut index = HashMap::new();
        index.insert(tokens[0].clone(), PAD);
        index.insert(tokens[1].clone(), BOS);

        for c in conditions {
            if index.contains_key(c) {
                return Err(VocabError::DuplicateCondition { token: c.clone() });
            }
            index.insert(c.clone(), tokens.len() as u32);
            tokens.push(c.clone());
        }

        for mv in enumerate_moves() {
            if let Some(existing) = index.get(&mv) {
                if *existing >= RESERVED && (*existing as usize) < tokens.len() {
                    return Err(VocabError::ConditionIsMove { token: mv });
                }
            }
            index.insert(mv.clone(), tokens.len() as u32);
            tokens.push(mv);
        }

        Ok(Self {
            tokens,
            index,
            condition_count: conditions.len(),
        })
    }

    /// Total number of ids, moves and reserved slots included.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Always `false`; the alphabet always carries at least PAD and BOS.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Number of condition tokens between the reserved ids and the
    /// moves.
    pub fn condition_count(&self) -> usize {
        self.condition_count
    }

    /// Smallest power of two that holds the alphabet.
    ///
    /// Model presets take a vocabulary size; rounding up keeps that
    /// number stable when a condition token is added later.
    pub fn model_vocab_size(&self) -> usize {
        self.tokens.len().next_power_of_two()
    }

    /// Look up a token's id.
    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.index.get(token).copied()
    }

    /// Look up an id's token.
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}

/// Every UCI move the board's geometry permits, in a fixed order.
///
/// A move is a from-square and a to-square that a queen or a knight
/// could connect, since no piece moves any other way. Steps onto the
/// far rank that a pawn could make additionally carry the four
/// promotion pieces, because UCI requires the suffix on a promotion.
/// Those steps keep their bare form too — the same geometry is an
/// ordinary rook, queen, bishop or king move.
fn enumerate_moves() -> Vec<String> {
    let mut out = Vec::new();
    for from in 0..64u8 {
        let (ff, fr) = (from % 8, from / 8);
        for to in 0..64u8 {
            if from == to {
                continue;
            }
            let (tf, tr) = (to % 8, to / 8);
            let df = (tf as i8 - ff as i8).abs();
            let dr = (tr as i8 - fr as i8).abs();
            let queen_ray = df == 0 || dr == 0 || df == dr;
            let knight_jump = (df == 1 && dr == 2) || (df == 2 && dr == 1);
            if !queen_ray && !knight_jump {
                continue;
            }
            let base = format!("{}{}", square_name(from), square_name(to));
            // The suffixed forms are additional, not a replacement.
            // A pawn reaching the far rank must promote, but the same
            // step is also an ordinary rook, queen, bishop or king
            // move — `a7a8` is a rook lift as often as it is a pawn
            // promoting, and dropping the bare form loses it.
            if is_promotion_step(ff, fr, tf, tr) {
                for piece in ['q', 'r', 'b', 'n'] {
                    out.push(format!("{base}{piece}"));
                }
            }
            out.push(base);
        }
    }
    out
}

/// `true` when a from/to pair is a pawn step onto the far rank.
///
/// One rank forward (from rank 7 to 8 for White, 2 to 1 for Black),
/// staying on the file or stepping one file across to capture.
fn is_promotion_step(ff: u8, fr: u8, tf: u8, tr: u8) -> bool {
    let file_step = (tf as i8 - ff as i8).abs();
    if file_step > 1 {
        return false;
    }
    (fr == 6 && tr == 7) || (fr == 1 && tr == 0)
}

/// Render a 0..64 square index as its algebraic name.
fn square_name(sq: u8) -> String {
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{file}{rank}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_ids_are_where_they_are_documented() {
        let v = MoveVocab::new(&[]).unwrap();
        assert_eq!(v.token_of(PAD), Some("<pad>"));
        assert_eq!(v.token_of(BOS), Some("<bos>"));
    }

    #[test]
    fn conditions_sit_between_reserved_ids_and_moves() {
        let conditions = vec!["<elo:1500>".to_string(), "<elo:1900>".to_string()];
        let v = MoveVocab::new(&conditions).unwrap();
        assert_eq!(v.id_of("<elo:1500>"), Some(2));
        assert_eq!(v.id_of("<elo:1900>"), Some(3));
        assert_eq!(v.condition_count(), 2);
        // The move block starts after the conditions.
        assert!(v.id_of("e2e4").unwrap() > 3);
    }

    #[test]
    fn a_repeated_condition_is_refused() {
        let conditions = vec!["<x>".to_string(), "<x>".to_string()];
        assert!(matches!(
            MoveVocab::new(&conditions),
            Err(VocabError::DuplicateCondition { .. })
        ));
    }

    #[test]
    fn ordinary_moves_are_present_and_bare() {
        let v = MoveVocab::new(&[]).unwrap();
        for mv in ["e2e4", "g1f3", "e1g1", "a1a8", "h8a1", "b1c3"] {
            assert!(v.id_of(mv).is_some(), "{mv} missing");
        }
    }

    #[test]
    fn promotions_carry_a_suffix() {
        let v = MoveVocab::new(&[]).unwrap();
        for mv in ["e7e8q", "e7e8r", "e7e8b", "e7e8n", "b2a1q", "h2g1n"] {
            assert!(v.id_of(mv).is_some(), "{mv} missing");
        }
    }

    #[test]
    fn a_step_onto_the_far_rank_keeps_its_bare_form() {
        // The same geometry a pawn promotes through is an ordinary
        // move for every other piece: a7a8 is a rook lift, e2f1 a king
        // step. Dropping the bare form loses those.
        let v = MoveVocab::new(&[]).unwrap();
        for mv in ["a7a8", "g7g8", "e2f1", "b2a1", "h2h1", "d7c8"] {
            assert!(v.id_of(mv).is_some(), "{mv} missing");
        }
    }

    #[test]
    fn geometrically_impossible_pairs_are_absent() {
        let v = MoveVocab::new(&[]).unwrap();
        // Neither a queen ray nor a knight jump.
        assert_eq!(v.id_of("a1c4"), None);
        assert_eq!(v.id_of("e2h6"), None);
    }

    #[test]
    fn ids_round_trip() {
        let v = MoveVocab::new(&["<elo:1500>".to_string()]).unwrap();
        for id in 0..v.len() as u32 {
            let token = v.token_of(id).expect("token for id");
            assert_eq!(v.id_of(token), Some(id));
        }
    }
}
