//! Handing a banded chess corpus to the conditioned training loop.
//!
//! Two numberings meet here and they are not the same one.
//!
//! - A corpus row records its band as a position in the
//!   [`ConditionSpec`] list it was built against ([`TeacherRow::band`]).
//! - A model's conditioning table is indexed by a
//!   [`CondIndex`], whose rows are `0..cond_slots` in
//!   [`ModelShape::bands`] order.
//!
//! They agree today, because `chess_bake` builds both lists from the
//! same command line — and nothing anywhere makes that so. A run that
//! resumed from a checkpoint whose bands were listed in another order,
//! or a corpus built against a subset, would line up numerically and
//! mean something else. So the ordinal is never used as a row: it is
//! resolved through the band's **token**, which is the one thing both
//! lists agree on, by [`ModelShape::band_index`] — the only producer of
//! a [`CondIndex`] outside this crate's model code.
//!
//! There is a third numbering close enough to be dangerous: the band's
//! vocabulary id, which is what actually sits in the row. Ids start at
//! 2 (`PAD`, `BOS`, then the condition tokens) while table rows start at
//! 0, so handing an id over as a row selects a real but different band,
//! and every check downstream passes. Nothing in this module produces a
//! [`CondIndex`] from an id.

use thiserror::Error;

use crate::arch::CondIndex;
use crate::chess::corpus::{ConditionSpec, TeacherRow};
use crate::chess::{CondEncoding, ModelShape};

/// Why a corpus could not be conditioned against a model shape.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConditionError {
    /// The shape does not describe a model with a conditioning table.
    ///
    /// [`ModelShape::config`] only asks for one under
    /// [`CondEncoding::EveryPosition`], so a table built against any
    /// other encoding names rows of a table the model will not have,
    /// and the run fails at its first forward pass instead — which on a
    /// real corpus is minutes after start-up, all of them spent reading
    /// PGN. Refused here, where the shape is available seconds in, for
    /// the same reason `chess_bake` resolves `CHESS_INIT_FROM` before
    /// the corpus rather than after it.
    #[error(
        "this shape is conditioned by {encoding}, so the model it describes has no \
         conditioning table to index into; per-row conditioning needs a shape written for \
         every-position conditioning"
    )]
    NotConditioned {
        /// Encoding the shape records.
        encoding: CondEncoding,
    },
    /// The corpus was built for a band the model was not.
    ///
    /// Reachable when a resume points at a checkpoint trained on other
    /// bands, or when a corpus is rebuilt with the list edited. Refused
    /// rather than resolved by position: the two lists would still line
    /// up as numbers.
    #[error(
        "the corpus is banded by {token:?}, which is not one of the model's bands ({model_bands}); \
         a row of that band has no row in the conditioning table"
    )]
    UnknownBand {
        /// Band token the corpus carries.
        token: String,
        /// The model's band tokens, for the message.
        model_bands: String,
    },
    /// A row carries no band in a corpus being trained as conditioned.
    ///
    /// A corpus is conditional or it is not, so a row without a band in
    /// one that has them means the two came apart somewhere between the
    /// build and here.
    #[error(
        "row {index} carries no band, but this run conditions every row; \
         the corpus and its band list have come apart"
    )]
    RowWithoutBand {
        /// 0-based row index.
        index: usize,
    },
    /// A row names a band the list it was built against does not hold.
    ///
    /// Same origin as [`Self::RowWithoutBand`] and the same reading: a
    /// row and its band list that no longer describe each other.
    #[error(
        "row {index} names band {ordinal} of a list holding {len}; \
         the corpus and its band list have come apart"
    )]
    BandOutOfRange {
        /// 0-based row index.
        index: usize,
        /// Ordinal the row carries.
        ordinal: usize,
        /// Length of the band list it was resolved against.
        len: usize,
    },
}

/// The conditioning-table row each band of one [`ConditionSpec`]
/// occupies, in that spec's order.
///
/// A named type rather than a bare `Vec<CondIndex>` because its indices
/// are only meaningful against the spec it was built from.
/// [`TeacherRow::band`] is an ordinal into the spec a corpus was
/// **built** with; a table built from some other list would answer
/// every lookup, with every length still agreeing, and mean another
/// band. Requiring this type at [`row_conditions`] is what stops a
/// third list from being passed there, and taking the spec — rather
/// than a band slice — is what lets a caller name the same value in
/// both places.
///
/// What remains the caller's obligation, since a row does not carry the
/// spec it came from: that `rows` were built against the spec this
/// table was built from. `chess_bake` holds one `ConditionSpec` value
/// and uses it for both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondTable {
    rows: Vec<CondIndex>,
}

impl CondTable {
    /// Bands the table covers.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table covers no band at all. Unreachable through
    /// [`cond_table`] for a spec with bands, and asked for by `clippy`
    /// beside `len`.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The conditioning-table row band `ordinal` of the spec occupies.
    pub fn row_for(&self, ordinal: usize) -> Option<CondIndex> {
        self.rows.get(ordinal).copied()
    }
}

/// Resolve every band of `spec` to the row it occupies in `shape`'s
/// conditioning table.
///
/// Built once per run rather than per row, which is also what keeps the
/// token lookup off the hot path.
///
/// # Errors
///
/// [`ConditionError::NotConditioned`] when the shape describes a model
/// with no conditioning table at all, and
/// [`ConditionError::UnknownBand`] when the model has no band by that
/// token. The alternative to the second — falling through to the
/// ordinal — is exactly the substitution this module exists to prevent.
pub fn cond_table(shape: &ModelShape, spec: &ConditionSpec) -> Result<CondTable, ConditionError> {
    if shape.encoding != CondEncoding::EveryPosition {
        return Err(ConditionError::NotConditioned {
            encoding: shape.encoding,
        });
    }
    let rows = spec
        .bands
        .iter()
        .map(|band| {
            shape
                .band_index(&band.token)
                .ok_or_else(|| ConditionError::UnknownBand {
                    token: band.token.clone(),
                    model_bands: shape.band_tokens().join(", "),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CondTable { rows })
}

/// One [`CondIndex`] per row, in row order.
///
/// The result is positional against `rows`, which is what the datasets'
/// `with_conditions` takes. Built from the rows as they stand, so a
/// caller that shuffled or replicated them for several epochs gets the
/// conditions of the rows it is actually about to train on.
///
/// # Errors
///
/// [`ConditionError::RowWithoutBand`] or
/// [`ConditionError::BandOutOfRange`] when a row and the band list have
/// come apart. Both are refused rather than skipped: dropping a row's
/// condition here would leave the remaining ones shifted by one, and
/// every row after it conditioned on its neighbour's band.
pub fn row_conditions(
    rows: &[TeacherRow],
    table: &CondTable,
) -> Result<Vec<CondIndex>, ConditionError> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let ordinal = row.band.ok_or(ConditionError::RowWithoutBand { index })?;
            table
                .row_for(ordinal)
                .ok_or(ConditionError::BandOutOfRange {
                    index,
                    ordinal,
                    len: table.len(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::corpus::ConditionBand;
    use crate::chess::CondEncoding;

    fn bands() -> Vec<ConditionBand> {
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
        ]
    }

    /// The spec a corpus of these two bands is built with.
    fn spec_of(bands: Vec<ConditionBand>) -> ConditionSpec {
        ConditionSpec {
            key: "WhiteElo".into(),
            bands,
        }
    }

    fn spec() -> ConditionSpec {
        spec_of(bands())
    }

    fn shape() -> ModelShape {
        let mut shape = ModelShape::compact(2048, bands());
        shape.encoding = CondEncoding::EveryPosition;
        shape
    }

    fn row(band: Option<usize>) -> TeacherRow {
        TeacherRow {
            ids: vec![1, 2, 100],
            mask: vec![0.0, 0.0, 1.0],
            band,
        }
    }

    /// Every band's row, as the table reports it.
    fn table_rows(table: &CondTable) -> Vec<u32> {
        (0..table.len())
            .map(|i| table.row_for(i).expect("in range").row())
            .collect()
    }

    #[test]
    fn each_band_resolves_to_its_own_table_row() {
        let table = cond_table(&shape(), &spec()).expect("both bands are the model's");
        assert_eq!(table_rows(&table), [0, 1]);
    }

    /// The band list a corpus was built against may be a permutation of
    /// the model's. The ordinals would line up as numbers and mean the
    /// other band; the tokens are what settle it.
    #[test]
    fn a_reordered_corpus_band_list_resolves_by_token_not_by_position() {
        let mut reordered = bands();
        reordered.reverse();
        let table = cond_table(&shape(), &spec_of(reordered)).expect("same bands, other order");
        assert_eq!(
            table_rows(&table),
            [1, 0],
            "corpus band 0 is the model's band 1 here"
        );
    }

    /// A prefix-conditioned shape has no table to index into, and the
    /// model built from it has no `cond_wte`. Caught here rather than
    /// at the first forward pass, which on a real corpus is minutes of
    /// PGN reading later.
    #[test]
    fn a_shape_that_conditions_by_prefix_has_no_table() {
        let mut prefix = shape();
        prefix.encoding = CondEncoding::Prefix;
        let err = cond_table(&prefix, &spec()).unwrap_err();
        assert_eq!(
            err,
            ConditionError::NotConditioned {
                encoding: CondEncoding::Prefix
            }
        );
    }

    #[test]
    fn a_band_the_model_does_not_have_is_refused() {
        let mut other = bands();
        other[1].token = "<elo:1500-1699>".into();
        let err = cond_table(&shape(), &spec_of(other)).unwrap_err();
        assert!(
            matches!(&err, ConditionError::UnknownBand { token, .. } if token == "<elo:1500-1699>"),
            "{err:?}"
        );
    }

    /// The band token's own vocabulary id is not its table row, and the
    /// two are both small integers in overlapping ranges. This is the
    /// substitution `CondIndex` exists to make unspellable.
    #[test]
    fn a_table_row_is_not_the_bands_vocabulary_id() {
        use crate::chess::vocab::MoveVocab;
        let vocab = MoveVocab::new(&bands().iter().map(|b| b.token.clone()).collect::<Vec<_>>())
            .expect("vocabulary");
        let table = cond_table(&shape(), &spec()).expect("table");
        for (ordinal, band) in bands().iter().enumerate() {
            let id = vocab.id_of(&band.token).expect("band in the vocabulary");
            let row = table.row_for(ordinal).expect("in range").row();
            assert_ne!(
                id, row,
                "band {} has id {id} and row {row} — a test that cannot tell them apart \
                 proves nothing",
                band.token
            );
        }
    }

    #[test]
    fn every_row_gets_the_condition_of_its_own_band() {
        let table = cond_table(&shape(), &spec()).expect("table");
        let rows = vec![row(Some(1)), row(Some(0)), row(Some(1))];
        let conds = row_conditions(&rows, &table).expect("every row has a band");
        assert_eq!(conds.iter().map(|c| c.row()).collect::<Vec<_>>(), [1, 0, 1]);
    }

    #[test]
    fn a_row_without_a_band_is_refused() {
        let table = cond_table(&shape(), &spec()).expect("table");
        let err = row_conditions(&[row(Some(0)), row(None)], &table).unwrap_err();
        assert_eq!(err, ConditionError::RowWithoutBand { index: 1 });
    }

    #[test]
    fn a_row_naming_a_band_outside_the_list_is_refused() {
        let table = cond_table(&shape(), &spec()).expect("table");
        let err = row_conditions(&[row(Some(7))], &table).unwrap_err();
        assert_eq!(
            err,
            ConditionError::BandOutOfRange {
                index: 0,
                ordinal: 7,
                len: 2,
            }
        );
    }
}
