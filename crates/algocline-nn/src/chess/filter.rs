//! Selecting which recorded games enter a corpus.
//!
//! # Why filtering is the interesting part
//!
//! A public archive is not a training set. The Lichess 2026-06
//! standard database holds on the order of 84 million games spanning
//! every rating from 400 to 3100, a third of them decided on the clock
//! rather than on the board. What a corpus should contain is a
//! decision about which of those games represent the player being
//! modelled, and that decision is made here.
//!
//! This is also where a style comes from. The one mechanism known to
//! put a recognisable playing style into a model is imitating players
//! who have it — Maia's result is that depth-limited engines match a
//! target rating without matching how players at that rating actually
//! move. With real games available, "play like a 1500" is a filter on
//! the rating tags, not a hand-tuned evaluation function.
//!
//! # Why the predicates are declarative
//!
//! A filter could be a callback. At archive scale that means crossing
//! the Rust/Lua boundary tens of millions of times, once per game,
//! before anything has been decided about the game. A predicate list
//! is evaluated in Rust while the stream is being read, and it is
//! still writable as a plain table from Lua.
//!
//! # Order matters
//!
//! Tags are checked before the moves are replayed. Replaying is the
//! expensive half — every ply resolved against generated legal moves —
//! and a rating-band filter keeps only a few percent of an archive, so
//! testing tags first is what makes a narrow corpus cheap to build.

use crate::chess::pgn::PgnGame;

/// A condition a single tag must satisfy.
#[derive(Debug, Clone)]
pub enum TagRule {
    /// The tag exists, whatever its value.
    Present,
    /// The tag's value is one of these.
    ///
    /// Written as a set rather than a single value because the useful
    /// cases are sets: `Termination` in `{Normal}`, `Result` in
    /// `{1-0, 0-1}`.
    OneOf(Vec<String>),
    /// The tag parses as an integer inside an inclusive range.
    ///
    /// Either bound may be open. A value that does not parse fails the
    /// rule rather than passing it — Lichess writes `?` for unrated
    /// players, and an unrated game is not in a rating band.
    IntRange {
        /// Inclusive lower bound, or `None` for open.
        min: Option<i64>,
        /// Inclusive upper bound, or `None` for open.
        max: Option<i64>,
    },
    /// The integer before the `+` in a PGN time control is at least
    /// this many seconds.
    ///
    /// `TimeControl` is written `base+increment`, so an ordinary
    /// integer range cannot read it. Bullet games are the majority of
    /// a Lichess archive and are the ones most shaped by the clock
    /// rather than by the position, which is the usual reason to set
    /// this.
    BaseSecondsAtLeast(i64),
    /// The tag's value starts with this string.
    ///
    /// Here for tags whose leading characters are a classification of
    /// their own: an ECO code is a letter and two digits, and the
    /// letter alone names the opening family. [`TagRule::OneOf`] could
    /// express "family B" as the hundred codes `B00`..`B99`, but that
    /// states the encoding of the family rather than the family, and a
    /// reader of the filter would have to count the list to learn what
    /// it means.
    StartsWith(String),
}

impl TagRule {
    /// Test the rule against a tag value.
    fn accepts(&self, value: &str) -> bool {
        match self {
            TagRule::Present => true,
            TagRule::OneOf(set) => set.iter().any(|v| v == value),
            TagRule::IntRange { min, max } => match value.parse::<i64>() {
                Ok(n) => min.is_none_or(|lo| n >= lo) && max.is_none_or(|hi| n <= hi),
                Err(_) => false,
            },
            TagRule::BaseSecondsAtLeast(min) => {
                let base = value.split('+').next().unwrap_or("");
                base.parse::<i64>().map(|n| n >= *min).unwrap_or(false)
            }
            TagRule::StartsWith(prefix) => value.starts_with(prefix.as_str()),
        }
    }
}

/// One tag and the rule it must satisfy.
#[derive(Debug, Clone)]
pub struct TagPredicate {
    /// PGN tag name, e.g. `WhiteElo`.
    pub key: String,
    /// The condition placed on it.
    pub rule: TagRule,
}

impl TagPredicate {
    /// Build a predicate.
    pub fn new(key: impl Into<String>, rule: TagRule) -> Self {
        Self {
            key: key.into(),
            rule,
        }
    }
}

/// Which games enter a corpus.
///
/// Every predicate must hold; an absent tag fails its predicate. The
/// ply bounds are applied after the moves are replayed, since the tags
/// do not carry a game's length.
#[derive(Debug, Clone, Default)]
pub struct GameFilter {
    /// Conditions on the header tags, all of which must hold.
    pub tags: Vec<TagPredicate>,
    /// Reject games shorter than this many plies.
    ///
    /// Games abandoned in the opening carry almost no signal and are a
    /// measurable share of an archive: about 0.2% of the games in a
    /// 2026-06 slice terminate as `Abandoned` with no moves at all.
    pub min_plies: usize,
    /// Reject games longer than this many plies, or `None` to accept
    /// any length.
    pub max_plies: Option<usize>,
}

impl GameFilter {
    /// A filter that accepts everything.
    pub fn accept_all() -> Self {
        Self::default()
    }

    /// Test the header tags. Cheap, and runs before the replay.
    pub fn accepts_tags(&self, game: &PgnGame) -> bool {
        self.tags.iter().all(|p| match game.tag(&p.key) {
            Some(value) => p.rule.accepts(value),
            None => false,
        })
    }

    /// Test a replayed game's length.
    pub fn accepts_length(&self, plies: usize) -> bool {
        if plies < self.min_plies {
            return false;
        }
        self.max_plies.is_none_or(|max| plies <= max)
    }

    /// Require both players to sit inside a rating band.
    ///
    /// The common case, and the one that stands in for "which player
    /// is being modelled": a game is only evidence about a band if
    /// both sides were playing at it.
    pub fn with_rating_band(mut self, min: i64, max: i64) -> Self {
        for key in ["WhiteElo", "BlackElo"] {
            self.tags.push(TagPredicate::new(
                key,
                TagRule::IntRange {
                    min: Some(min),
                    max: Some(max),
                },
            ));
        }
        self
    }

    /// Keep only games that ended on the board.
    ///
    /// A third of a Lichess archive ends in `Time forfeit`, where the
    /// final position says nothing about what either player judged to
    /// be a good move.
    pub fn decided_on_the_board(mut self) -> Self {
        self.tags.push(TagPredicate::new(
            "Termination",
            TagRule::OneOf(vec!["Normal".to_string()]),
        ));
        self
    }

    /// Keep only games with at least this much base time.
    pub fn with_min_base_seconds(mut self, seconds: i64) -> Self {
        self.tags.push(TagPredicate::new(
            "TimeControl",
            TagRule::BaseSecondsAtLeast(seconds),
        ));
        self
    }

    /// Keep only games whose ECO code starts with this prefix.
    ///
    /// A one-letter prefix selects an opening family (`"B"` is the
    /// Sicilian-to-Caro-Kann range), a longer one narrows it (`"B2"`,
    /// `"B27"`). Lichess writes the tag on every game in the slices
    /// measured, but this predicate does not assume that: a game
    /// without the tag fails the predicate like any other absent tag.
    pub fn with_eco_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tags
            .push(TagPredicate::new("ECO", TagRule::StartsWith(prefix.into())));
        self
    }

    /// Set the accepted ply range.
    pub fn with_ply_bounds(mut self, min: usize, max: Option<usize>) -> Self {
        self.min_plies = min;
        self.max_plies = max;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::pgn::PgnReader;
    use std::io::Cursor;

    fn game(tags: &[(&str, &str)]) -> PgnGame {
        let mut text = String::new();
        for (k, v) in tags {
            text.push_str(&format!("[{k} \"{v}\"]\n"));
        }
        text.push_str("\n1. e4 e5 1-0\n\n");
        PgnReader::new(Cursor::new(text))
            .next_game()
            .unwrap()
            .unwrap()
    }

    #[test]
    fn a_rating_band_needs_both_players_inside_it() {
        let f = GameFilter::accept_all().with_rating_band(1600, 1799);
        assert!(f.accepts_tags(&game(&[("WhiteElo", "1650"), ("BlackElo", "1700")])));
        assert!(!f.accepts_tags(&game(&[("WhiteElo", "1650"), ("BlackElo", "1900")])));
    }

    #[test]
    fn an_unrated_player_is_not_in_any_band() {
        let f = GameFilter::accept_all().with_rating_band(1600, 1799);
        assert!(!f.accepts_tags(&game(&[("WhiteElo", "?"), ("BlackElo", "1700")])));
    }

    #[test]
    fn a_missing_tag_fails_its_predicate() {
        let f = GameFilter::accept_all().with_rating_band(1600, 1799);
        assert!(!f.accepts_tags(&game(&[("WhiteElo", "1650")])));
    }

    #[test]
    fn time_control_is_read_as_base_plus_increment() {
        let f = GameFilter::accept_all().with_min_base_seconds(180);
        assert!(f.accepts_tags(&game(&[("TimeControl", "180+2")])));
        assert!(f.accepts_tags(&game(&[("TimeControl", "600+0")])));
        assert!(!f.accepts_tags(&game(&[("TimeControl", "60+0")])));
        assert!(!f.accepts_tags(&game(&[("TimeControl", "-")])));
    }

    #[test]
    fn termination_is_a_set_membership_test() {
        let f = GameFilter::accept_all().decided_on_the_board();
        assert!(f.accepts_tags(&game(&[("Termination", "Normal")])));
        assert!(!f.accepts_tags(&game(&[("Termination", "Time forfeit")])));
    }

    #[test]
    fn an_eco_prefix_selects_a_family_and_narrows_with_length() {
        let family = GameFilter::accept_all().with_eco_prefix("B");
        assert!(family.accepts_tags(&game(&[("ECO", "B20")])));
        assert!(family.accepts_tags(&game(&[("ECO", "B99")])));
        assert!(!family.accepts_tags(&game(&[("ECO", "C20")])));

        let narrowed = GameFilter::accept_all().with_eco_prefix("B2");
        assert!(narrowed.accepts_tags(&game(&[("ECO", "B27")])));
        assert!(!narrowed.accepts_tags(&game(&[("ECO", "B30")])));
    }

    #[test]
    fn a_game_without_an_eco_tag_fails_the_eco_predicate() {
        let f = GameFilter::accept_all().with_eco_prefix("B");
        assert!(!f.accepts_tags(&game(&[("WhiteElo", "1650")])));
    }

    #[test]
    fn length_bounds_are_inclusive() {
        let f = GameFilter::accept_all().with_ply_bounds(10, Some(128));
        assert!(!f.accepts_length(9));
        assert!(f.accepts_length(10));
        assert!(f.accepts_length(128));
        assert!(!f.accepts_length(129));
    }

    #[test]
    fn an_empty_filter_accepts_everything() {
        let f = GameFilter::accept_all();
        assert!(f.accepts_tags(&game(&[])));
        assert!(f.accepts_length(0));
    }
}
