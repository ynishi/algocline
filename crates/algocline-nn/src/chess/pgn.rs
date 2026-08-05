//! PGN ingestion: read chess games recorded in Standard Algebraic
//! Notation and turn them into move sequences a tokenizer can consume.
//!
//! # Why this module exists
//!
//! Public chess play data is distributed as PGN (the Lichess open
//! database is monthly zstd-compressed PGN under CC0). PGN records
//! moves in SAN — `Nf3`, `exd5`, `O-O` — which names the destination
//! square and leaves the origin implicit: recovering `Nf3` means asking
//! which knight can legally reach f3 in this position. That is a legal
//! move generation problem, so the SAN reader here sits on top of
//! `cozy_chess` rather than pattern-matching strings.
//!
//! # What SAN resolution buys
//!
//! Resolving every SAN token against the legal move set is also the
//! self-check for this module. A correct reader finds *exactly one*
//! legal move per token; zero matches or two matches mean either the
//! parse or the replay is wrong. Running a real corpus through
//! [`game_to_uci`] therefore verifies the implementation against the
//! data itself — see `examples/pgn_san_check.rs`.
//!
//! # Notation on the way out
//!
//! Moves come back as UCI (`e2e4`, `e7e8q`). Unlike SAN, UCI is
//! position-independent and its alphabet is bounded by the board rather
//! than by the corpus, which is what a fixed-size token vocabulary
//! needs.
//!
//! Castling is normalised on the way out. `cozy_chess` encodes castling
//! the Chess960 way — the king moves onto its own rook — so kingside
//! castling from the standard start position is `e1h1` internally.
//! [`uci_standard`] rewrites that to the `e1g1` form that standard-chess
//! tooling expects.

use std::collections::HashMap;
use std::io::BufRead;

use cozy_chess::{Board, Move, Piece, Square};
use thiserror::Error;

/// Failure while reading a PGN stream.
#[derive(Debug, Error)]
pub enum PgnError {
    /// The underlying reader failed.
    #[error("pgn: read failed at line {line}: {source_msg}")]
    Io {
        /// 1-based line number reached when the read failed.
        line: usize,
        /// Message from the underlying I/O error.
        source_msg: String,
    },
    /// A line started with `[` but is not a well-formed tag pair.
    #[error("pgn: line {line} is not a well-formed tag pair: {raw:?}")]
    Tag {
        /// 1-based line number of the offending line.
        line: usize,
        /// The line as read.
        raw: String,
    },
}

/// Failure while turning a SAN token into a move.
#[derive(Debug, Error)]
pub enum SanError {
    /// No legal move in this position matches the token.
    ///
    /// Either the token was mis-parsed or the replay drifted from the
    /// position the game was actually in.
    #[error("san {san:?}: no legal move matches")]
    NoMatch {
        /// The SAN token as it appeared in the movetext.
        san: String,
    },
    /// More than one legal move matches, so the token is ambiguous.
    ///
    /// A well-formed PGN never produces this: the writer is required to
    /// add a file or rank hint whenever two pieces can reach the same
    /// square.
    #[error("san {san:?}: {count} legal moves match, expected exactly 1")]
    Ambiguous {
        /// The SAN token as it appeared in the movetext.
        san: String,
        /// Number of legal moves that matched.
        count: usize,
    },
    /// The token does not have the shape of a SAN move at all.
    #[error("san {san:?}: not parsable as SAN ({reason})")]
    Unparsable {
        /// The SAN token as it appeared in the movetext.
        san: String,
        /// What about the token could not be read.
        reason: &'static str,
    },
    /// The trailing two characters do not name a square.
    #[error("san {san:?}: {square:?} is not a square")]
    BadSquare {
        /// The SAN token as it appeared in the movetext.
        san: String,
        /// The substring that was read as a destination square.
        square: String,
    },
}

/// Failure while replaying a whole game.
#[derive(Debug, Error)]
pub enum ReplayError {
    /// A SAN token could not be resolved.
    ///
    /// Carries the ply index so a failure in a large corpus can be
    /// located without re-running the replay.
    #[error("ply {ply}: {source}")]
    San {
        /// 0-based index of the ply within the game.
        ply: usize,
        /// The SAN resolution failure.
        #[source]
        source: SanError,
    },
}

/// One game read off a PGN stream.
#[derive(Debug, Clone, Default)]
pub struct PgnGame {
    /// Tag pairs from the game's header, keyed by tag name.
    ///
    /// Lichess games carry `Result`, `WhiteElo`, `BlackElo`, `ECO`,
    /// `Opening`, `TimeControl` and `Termination` on every game; this
    /// map is what a corpus filter reads.
    pub tags: HashMap<String, String>,
    /// The movetext, comments and move numbers included, exactly as
    /// read. [`san_tokens`] strips it down to the moves.
    pub movetext: String,
}

impl PgnGame {
    /// Read a tag, returning `None` when it is absent.
    pub fn tag(&self, name: &str) -> Option<&str> {
        self.tags.get(name).map(String::as_str)
    }

    /// Read a tag and parse it as an integer, returning `None` when the
    /// tag is absent or does not parse.
    ///
    /// Ratings are the intended use: Lichess writes `?` for unrated
    /// accounts, so an absent-or-unparsable reading is expected rather
    /// than exceptional.
    pub fn tag_i64(&self, name: &str) -> Option<i64> {
        self.tag(name)?.parse().ok()
    }
}

/// Streaming PGN reader.
///
/// Games are yielded one at a time; nothing is held beyond the current
/// game, so a multi-gigabyte stream can be filtered without being
/// materialised. Note that the Lichess monthly archives are a single
/// zstd frame, so the stream can only be read from the beginning — a
/// prefix is a time-ordered slice, not a random sample.
pub struct PgnReader<R: BufRead> {
    reader: R,
    /// A line already read but belonging to the next game.
    pushback: Option<String>,
    line_no: usize,
}

impl<R: BufRead> PgnReader<R> {
    /// Wrap a buffered reader positioned at the start of a PGN stream.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            pushback: None,
            line_no: 0,
        }
    }

    /// Number of input lines consumed so far.
    pub fn line_no(&self) -> usize {
        self.line_no
    }

    fn read_line(&mut self) -> Result<Option<String>, PgnError> {
        if let Some(line) = self.pushback.take() {
            return Ok(Some(line));
        }
        let mut buf = String::new();
        let n = self.reader.read_line(&mut buf).map_err(|e| PgnError::Io {
            line: self.line_no + 1,
            source_msg: e.to_string(),
        })?;
        if n == 0 {
            return Ok(None);
        }
        self.line_no += 1;
        while buf.ends_with('\n') || buf.ends_with('\r') {
            buf.pop();
        }
        Ok(Some(buf))
    }

    /// Read the next game, or `None` once the stream is drained.
    pub fn next_game(&mut self) -> Result<Option<PgnGame>, PgnError> {
        let mut game = PgnGame::default();
        let mut seen_header = false;

        // Header: a run of tag pairs, preceded by any number of blanks.
        loop {
            let Some(line) = self.read_line()? else {
                return Ok(None);
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix('[') {
                let (key, value) = parse_tag(rest).ok_or_else(|| PgnError::Tag {
                    line: self.line_no,
                    raw: line.clone(),
                })?;
                game.tags.insert(key, value);
                seen_header = true;
                continue;
            }
            // First movetext line.
            self.pushback = Some(line);
            break;
        }

        // Movetext: everything up to the next blank line, the next
        // header, or end of stream.
        while let Some(line) = self.read_line()? {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.starts_with('[') {
                self.pushback = Some(line);
                break;
            }
            if !game.movetext.is_empty() {
                game.movetext.push(' ');
            }
            game.movetext.push_str(trimmed);
        }

        if !seen_header && game.movetext.is_empty() {
            return Ok(None);
        }
        Ok(Some(game))
    }
}

/// Split `[Key "Value"]` (with the leading `[` already removed).
fn parse_tag(rest: &str) -> Option<(String, String)> {
    let rest = rest.strip_suffix(']')?;
    let (key, remainder) = rest.split_once(' ')?;
    let value = remainder.trim().strip_prefix('"')?.strip_suffix('"')?;
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), value.to_string()))
}

/// Game terminators that appear inside the movetext.
const RESULT_TOKENS: [&str; 4] = ["1-0", "0-1", "1/2-1/2", "*"];

/// Strip a movetext down to its SAN tokens.
///
/// Removes `{...}` comments (Lichess carries a clock reading on 99.9%
/// of plies and a Stockfish evaluation on about 11%), `(...)`
/// variations, `$n` numeric annotation glyphs, move numbers and the
/// result token.
pub fn san_tokens(movetext: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut comment_depth = 0usize;
    let mut variation_depth = 0usize;

    let flush = |current: &mut String, out: &mut Vec<String>| {
        if current.is_empty() {
            return;
        }
        let token = std::mem::take(current);
        let token = strip_move_number(&token);
        if token.is_empty() || RESULT_TOKENS.contains(&token) || token.starts_with('$') {
            return;
        }
        out.push(token.to_string());
    };

    for ch in movetext.chars() {
        match ch {
            '{' => {
                flush(&mut current, &mut out);
                comment_depth += 1;
            }
            '}' => comment_depth = comment_depth.saturating_sub(1),
            '(' if comment_depth == 0 => {
                flush(&mut current, &mut out);
                variation_depth += 1;
            }
            ')' if comment_depth == 0 => variation_depth = variation_depth.saturating_sub(1),
            _ if comment_depth > 0 || variation_depth > 0 => {}
            c if c.is_whitespace() => flush(&mut current, &mut out),
            c => current.push(c),
        }
    }
    flush(&mut current, &mut out);
    out
}

/// Drop a leading move number (`12.`, `12...`, or a bare `12.` glued to
/// the move as in `12.e4`).
fn strip_move_number(token: &str) -> &str {
    let bytes = token.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        // No digits, or digits only (a bare move number or a result
        // fragment) — leave it to the caller's filter.
        return if i == 0 { token } else { "" };
    }
    if bytes[i] != b'.' {
        return token;
    }
    while i < bytes.len() && bytes[i] == b'.' {
        i += 1;
    }
    &token[i..]
}

/// Resolve one SAN token against a position.
///
/// Returns the unique legal move the token names. Both "no match" and
/// "more than one match" are errors: a PGN writer is required to
/// disambiguate, so either outcome indicates a bug rather than an
/// acceptable reading.
pub fn resolve_san(board: &Board, san: &str) -> Result<Move, SanError> {
    let core = strip_annotations(san);
    if core.is_empty() {
        return Err(SanError::Unparsable {
            san: san.to_string(),
            reason: "empty after stripping annotations",
        });
    }

    if let Some(kingside) = castling_side(core) {
        return resolve_castling(board, san, kingside);
    }

    // Promotion suffix: `=Q`, or the older `Q` glued to the square.
    let (body, promotion) = split_promotion(core, san)?;

    // Leading piece letter, or a pawn move.
    let (piece, rest) = match body.chars().next() {
        Some(c @ ('K' | 'Q' | 'R' | 'B' | 'N')) => (piece_of(c), &body[c.len_utf8()..]),
        Some(_) => (Piece::Pawn, body),
        None => {
            return Err(SanError::Unparsable {
                san: san.to_string(),
                reason: "no move body",
            })
        }
    };

    let rest: String = rest.chars().filter(|c| *c != 'x').collect();
    if rest.len() < 2 {
        return Err(SanError::Unparsable {
            san: san.to_string(),
            reason: "no destination square",
        });
    }
    let split = rest.len() - 2;
    let dest_str = &rest[split..];
    let dest: Square = dest_str.parse().map_err(|_| SanError::BadSquare {
        san: san.to_string(),
        square: dest_str.to_string(),
    })?;

    // What remains in front of the destination is the disambiguation:
    // a file letter, a rank digit, or both.
    let hint = &rest[..split];
    let mut hint_file: Option<u8> = None;
    let mut hint_rank: Option<u8> = None;
    for b in hint.bytes() {
        match b {
            b'a'..=b'h' => hint_file = Some(b),
            b'1'..=b'8' => hint_rank = Some(b),
            _ => {
                return Err(SanError::Unparsable {
                    san: san.to_string(),
                    reason: "unreadable disambiguation",
                })
            }
        }
    }

    let mut matched: Option<Move> = None;
    let mut count = 0usize;
    board.generate_moves(|moves| {
        if moves.piece != piece {
            return false;
        }
        for mv in moves {
            if mv.to != dest || mv.promotion != promotion {
                continue;
            }
            if hint_file.is_some() || hint_rank.is_some() {
                let from = mv.from.to_string();
                let from = from.as_bytes();
                if hint_file.is_some_and(|f| from[0] != f) {
                    continue;
                }
                if hint_rank.is_some_and(|r| from[1] != r) {
                    continue;
                }
            }
            count += 1;
            if matched.is_none() {
                matched = Some(mv);
            }
        }
        false
    });

    match (matched, count) {
        (Some(mv), 1) => Ok(mv),
        (_, 0) => Err(SanError::NoMatch {
            san: san.to_string(),
        }),
        (_, n) => Err(SanError::Ambiguous {
            san: san.to_string(),
            count: n,
        }),
    }
}

/// Drop check / mate markers and the `!?` annotation family.
fn strip_annotations(san: &str) -> &str {
    san.trim_end_matches(['+', '#', '!', '?'])
}

/// `Some(true)` for kingside castling, `Some(false)` for queenside.
fn castling_side(core: &str) -> Option<bool> {
    match core {
        "O-O" | "0-0" => Some(true),
        "O-O-O" | "0-0-0" => Some(false),
        _ => None,
    }
}

/// Find the castling move, which `cozy_chess` encodes as the king
/// moving onto its own rook.
fn resolve_castling(board: &Board, san: &str, kingside: bool) -> Result<Move, SanError> {
    let stm = board.side_to_move();
    let mut matched: Option<Move> = None;
    let mut count = 0usize;
    board.generate_moves(|moves| {
        if moves.piece != Piece::King {
            return false;
        }
        for mv in moves {
            if board.color_on(mv.to) != Some(stm) {
                continue; // a plain king move, not castling
            }
            let from = mv.from.to_string();
            let to = mv.to.to_string();
            let is_kingside = to.as_bytes()[0] > from.as_bytes()[0];
            if is_kingside != kingside {
                continue;
            }
            count += 1;
            if matched.is_none() {
                matched = Some(mv);
            }
        }
        false
    });
    match (matched, count) {
        (Some(mv), 1) => Ok(mv),
        (_, 0) => Err(SanError::NoMatch {
            san: san.to_string(),
        }),
        (_, n) => Err(SanError::Ambiguous {
            san: san.to_string(),
            count: n,
        }),
    }
}

/// Split a promotion suffix off the move body.
fn split_promotion<'a>(core: &'a str, san: &str) -> Result<(&'a str, Option<Piece>), SanError> {
    if let Some((body, promo)) = core.split_once('=') {
        let c = promo.chars().next().ok_or_else(|| SanError::Unparsable {
            san: san.to_string(),
            reason: "promotion marker with no piece",
        })?;
        if !matches!(c, 'Q' | 'R' | 'B' | 'N') {
            return Err(SanError::Unparsable {
                san: san.to_string(),
                reason: "promotion to an impossible piece",
            });
        }
        return Ok((body, Some(piece_of(c))));
    }
    // Older PGN writers glue the promotion piece to the square (`e8Q`).
    let bytes = core.as_bytes();
    if bytes.len() >= 3 {
        let last = bytes[bytes.len() - 1];
        if matches!(last, b'Q' | b'R' | b'B' | b'N') && bytes[bytes.len() - 2].is_ascii_digit() {
            return Ok((&core[..core.len() - 1], Some(piece_of(last as char))));
        }
    }
    Ok((core, None))
}

/// Map a SAN piece letter to a piece.
///
/// Only called with letters already matched against `KQRBN`.
fn piece_of(c: char) -> Piece {
    match c {
        'K' => Piece::King,
        'Q' => Piece::Queen,
        'R' => Piece::Rook,
        'B' => Piece::Bishop,
        _ => Piece::Knight,
    }
}

/// Render a move as standard-chess UCI.
///
/// `cozy_chess` encodes castling as the king capturing its own rook
/// (`e1h1`), which is the Chess960 convention. Standard-chess tooling
/// expects the king's two-square destination (`e1g1`), so castling is
/// rewritten here. Every other move is already in UCI form.
///
/// `board` must be the position the move is played from.
pub fn uci_standard(board: &Board, mv: Move) -> String {
    if board.color_on(mv.to) != Some(board.side_to_move()) {
        return mv.to_string();
    }
    let from = mv.from.to_string();
    let to = mv.to.to_string();
    let kingside = to.as_bytes()[0] > from.as_bytes()[0];
    let file = if kingside { 'g' } else { 'c' };
    let rank = from.as_bytes()[1] as char;
    format!("{from}{file}{rank}")
}

/// Replay a movetext from the standard start position and return the
/// moves in UCI.
///
/// Every SAN token must resolve to exactly one legal move; the first
/// token that does not stops the replay with the ply index attached.
pub fn game_to_uci(movetext: &str) -> Result<Vec<String>, ReplayError> {
    let tokens = san_tokens(movetext);
    let mut board = Board::default();
    let mut out = Vec::with_capacity(tokens.len());
    for (ply, token) in tokens.iter().enumerate() {
        let mv = resolve_san(&board, token).map_err(|source| ReplayError::San { ply, source })?;
        out.push(uci_standard(&board, mv));
        board.play_unchecked(mv);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = "[Event \"Rated Blitz game\"]\n\
[White \"a\"]\n\
[Black \"b\"]\n\
[Result \"1-0\"]\n\
[WhiteElo \"1523\"]\n\
\n\
1. e4 { [%clk 0:03:00] } 1... c5 { [%clk 0:03:00] } 2. Nf3 1-0\n\
\n";

    #[test]
    fn reads_tags_and_movetext() {
        let mut reader = PgnReader::new(Cursor::new(SAMPLE));
        let game = reader.next_game().unwrap().expect("a game");
        assert_eq!(game.tag("Result"), Some("1-0"));
        assert_eq!(game.tag_i64("WhiteElo"), Some(1523));
        assert_eq!(game.tag("Missing"), None);
        assert!(reader.next_game().unwrap().is_none());
    }

    #[test]
    fn strips_comments_numbers_and_result() {
        let mut reader = PgnReader::new(Cursor::new(SAMPLE));
        let game = reader.next_game().unwrap().unwrap();
        assert_eq!(san_tokens(&game.movetext), vec!["e4", "c5", "Nf3"]);
    }

    #[test]
    fn resolves_an_opening() {
        let uci = game_to_uci("1. e4 c5 2. Nf3 1-0").unwrap();
        assert_eq!(uci, vec!["e2e4", "c7c5", "g1f3"]);
    }

    #[test]
    fn castling_uses_the_standard_king_destination() {
        let uci = game_to_uci("1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. O-O O-O").unwrap();
        assert_eq!(uci.last(), Some(&"e8g8".to_string()));
        assert_eq!(uci[6], "e1g1");
    }

    #[test]
    fn disambiguates_by_file_and_rank() {
        // Both knights can reach d2; SAN must say which.
        let uci = game_to_uci("1. Nf3 d5 2. Nc3 d4 3. Nfe5").unwrap();
        assert_eq!(uci.last(), Some(&"f3e5".to_string()));
    }

    #[test]
    fn reads_promotion() {
        let uci = game_to_uci("1. e4 d5 2. exd5 c6 3. dxc6 Nf6 4. cxb7 Bg4 5. bxa8=Q").unwrap();
        assert_eq!(uci.last(), Some(&"b7a8q".to_string()));
    }

    #[test]
    fn rejects_a_move_that_is_not_legal_here() {
        let err = game_to_uci("1. e4 e5 2. Qh6").unwrap_err();
        let ReplayError::San { ply, source } = err;
        assert_eq!(ply, 2);
        assert!(matches!(source, SanError::NoMatch { .. }));
    }
}
