//! Delivering the condition to the model, whichever way the checkpoint
//! expects it.
//!
//! [`super::window::Window`] exists because a bare `Vec<u32>` cannot say
//! whether position 1 is a band or a move. This is the same argument one
//! level up: a bare `Tensor` cannot say whether the model was told which
//! band it is playing as.
//!
//! # The two channels
//!
//! A conditioned row is `[BOS, band] + moves` under **both**
//! conventions. The per-position arm keeps the band token deliberately,
//! so that the two arms train on the same corpus with the same row
//! lengths; what it adds is a second channel, an index into a
//! conditioning table of its own, supplied as an argument to the forward
//! pass ([`crate::arch::Gpt2Model::forward_conditioned`]).
//!
//! So a reader has to do two things right at once, and the two are far
//! apart: put the band at position 1 of every row, and hand the matching
//! index to the forward. Doing the first and not the second runs a
//! conditioned model with no condition — and the moves that come out
//! look entirely ordinary, which is the whole difficulty. Nothing fails,
//! nothing warns, and the numbers are plausible.
//!
//! [`BandBatch`] builds both together, in one pass over one list, and
//! owns the only forward. A caller cannot construct the rows and forget
//! the indices, because it does not construct either; and the two cannot
//! come to name different bands, because they are pushed in lockstep
//! from the same band list.
//!
//! # Why this is not left to the call sites
//!
//! It was, and there were two of them in one file — the single row a
//! player scores, and the one-row-per-band batch guidance needs for its
//! reference. Two call sites each choosing a forward is two places to
//! forget. Here the choice is made once, from the checkpoint's own
//! recorded encoding, and both readers go through it.
//!
//! # The third channel
//!
//! A checkpoint trained with [`ModelShape::legal_input`] reads the ids
//! allowed at each position as well, and the same argument applies a
//! second time: the sets are supplied to the forward pass, so a reader
//! that built the rows and not the sets would be asking a model to run
//! in a state it never trained in. Here the two are one argument. A
//! shape that wants the sets cannot be batched without them and a shape
//! that does not cannot be batched with them, so a mismatch is caught
//! where the rows are built rather than at the forward — which refuses
//! both directions from its own side too, and is the backstop for a
//! caller that reaches [`crate::arch::Gpt2Model`] without coming through
//! here.
//!
//! What the sets have to be is [`Window::legal_sets`]'s business: the
//! mapping from a replayed game onto a row the context window may have
//! cut is where an off-by-one hides, and it hides quietly.

use candle_core::{Device, IndexOp, Tensor};
use thiserror::Error;

use crate::arch::{CondIndex, Gpt2Model, LegalSets};
use crate::chess::vocab::MoveVocab;
use crate::chess::window::{Window, WindowError};
use crate::chess::{CondEncoding, ModelShape};

/// Why a batch could not be built or scored.
#[derive(Debug, Error)]
pub enum BatchError {
    /// A band could not be written into a row.
    #[error(transparent)]
    Window(#[from] WindowError),

    /// [`BandBatch::over_combos`] was asked for a checkpoint that does
    /// not partition its bands into several groups.
    ///
    /// One group is [`BandBatch::over_bands`]'s case; answering it here
    /// would quietly run the same comparison through a second path.
    #[error(
        "this checkpoint carries {groups} condition group(s); combination rows need at least \
         two — a single group's comparison is `over_bands`"
    )]
    NotMultiSlot {
        /// Groups the shape partitions its bands into.
        groups: usize,
    },

    /// A multi-slot walk of a prefix-encoded checkpoint was asked for.
    ///
    /// No such checkpoint exists to walk: a prefix cannot carry two
    /// slots without inventing an order for them, which is why a
    /// multi-slot corpus keeps its conditions out of the rows entirely.
    /// Reaching this means a shape whose `cond_groups` and `encoding`
    /// contradict each other.
    #[error(
        "a multi-slot walk conditions through forward arguments, and this checkpoint is \
         prefix-encoded; its cond_groups and encoding contradict each other"
    )]
    MultiSlotNeedsEveryPosition,

    /// The forward pass failed.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    /// A band the shape lists is not in the vocabulary the caller built.
    ///
    /// The two are meant to come from the same place — a reader builds
    /// its vocabulary from `ModelShape::band_tokens` — so this means
    /// they did not.
    #[error("band {token:?} is not in this vocabulary; it and the checkpoint disagree")]
    BandMissing {
        /// Band token that could not be resolved.
        token: String,
    },

    /// A per-position checkpoint was asked to run without a band.
    ///
    /// Refused rather than run with the condition dropped: under this
    /// convention the band is an argument to the forward pass, and
    /// there is no value that means "no band" — dropping it would call
    /// the plain forward, which is a different model whose moves look
    /// no different.
    ///
    /// `bands` reports what the checkpoint does carry, which is what a
    /// caller needs in order to name one. It may be empty: a shape with
    /// no bands still sizes its table from the list
    /// ([`ModelShape::config`] passes `Some(0)`), so a zero-band
    /// per-position checkpoint is constructible. Nothing in this file
    /// rules it out and the refusal does not depend on ruling it out —
    /// such a checkpoint cannot be run at all, which is what this error
    /// then says.
    #[error(
        "this checkpoint conditions every position, so it cannot be run without a band; \
         it carries {bands:?}"
    )]
    NoBand {
        /// Bands the checkpoint does carry.
        bands: Vec<String>,
    },

    /// A band has no row in the conditioning table.
    ///
    /// Reachable when a shape and a model disagree about the band list,
    /// which is a crossed pair rather than a typo.
    #[error("band {token:?} has no row in this checkpoint's conditioning table")]
    NoConditionRow {
        /// Band token that could not be resolved to a row.
        token: String,
    },

    /// There is nothing to score.
    #[error("this batch holds no rows, so there is nothing to score")]
    Empty,

    /// The checkpoint reads the ids allowed at each position and none
    /// were supplied.
    ///
    /// Refused here rather than at the forward pass, which refuses it
    /// too. Both refusals are worth having: this one names the batch a
    /// caller is building and can say what the sets have to be, and the
    /// one in the model covers a caller that never came through here.
    #[error(
        "this checkpoint was trained with the ids allowed at each position supplied to it, so \
         a batch built from it has to carry them; `Window::legal_sets` maps a replayed game \
         onto the row's positions"
    )]
    LegalSetsMissing,

    /// Sets were supplied for a checkpoint that has no table to read
    /// them with.
    ///
    /// A caller that believes it is delivering a channel the model does
    /// not have. The mirror of [`Self::LegalSetsMissing`], and refused
    /// for the same reason: the pairing is a property of the checkpoint,
    /// not a choice at the call site.
    #[error(
        "this checkpoint was not trained with a legality input, so it has no table to read one \
         with; the sets supplied here would reach nothing"
    )]
    LegalSetsUnexpected,

    /// The sets do not cover this row.
    ///
    /// One per token plus one for the position the model answers at.
    /// A list of another length was built against a different row —
    /// a window from another position, or a history mapped by hand —
    /// and every entry of it would describe the wrong position.
    #[error(
        "the legal sets cover {found} position(s) and this row needs {want}: one per token, \
         plus the position past its end that the model answers at"
    )]
    LegalSetsWidth {
        /// Entries this row needs.
        want: usize,
        /// Entries supplied.
        found: usize,
    },

    /// The checkpoint asks for the condition at every position **and**
    /// for a legality input.
    ///
    /// No forward pass in this build reads both, and
    /// `Gpt2Custom::validate` refuses to build a model that asks for the
    /// pair — so a shape of this kind describes a model a reader cannot
    /// construct, whatever it does about the batch. The refusal is
    /// repeated here because this type would otherwise have to choose
    /// one of the two channels and drop the other, silently.
    #[error(
        "this checkpoint asks for the condition at every position and for a legality input, \
         and no forward pass in this build reads both"
    )]
    BothChannels,
}

/// Rows to feed the model, and the conditioning that travels with them.
///
/// Built by [`BandBatch::single`] or [`BandBatch::over_bands`]; there is
/// no other constructor, and the fields are private, so the invariants
/// below cannot be assembled wrongly from outside this module.
///
/// **Invariant**: `conds` is `Some` exactly when the checkpoint
/// conditions every position, and when it is, it holds one index per
/// row, naming the same band that row carries at position 1.
///
/// **Invariant**: `legal` is `Some` exactly when the checkpoint reads
/// the ids allowed at each position, and when it is, it holds one list
/// per row of `row.len() + 1` sets in [`Window::legal_sets`]'s
/// convention — entry `p` is the set the token at position `p` was
/// drawn from, and the last is the set the model's answer is drawn
/// from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandBatch {
    rows: Vec<Vec<u32>>,
    conds: Option<Vec<CondIndex>>,
    conds_per_row: usize,
    legal: Option<Vec<Vec<Vec<u32>>>>,
}

/// Every combination of one global table row per group, earlier groups
/// varying slower.
///
/// Global rows rather than within-group ordinals, so a consumer indexes
/// the flat band list and the conditioning table without re-deriving
/// the offsets — the two mistakes that arithmetic invites (an ordinal
/// used as a row, an offset applied twice) never leave this function.
fn combo_indices(groups: &[usize]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![Vec::new()];
    let mut offset = 0u32;
    for size in groups {
        let mut next = Vec::with_capacity(out.len() * size);
        for prefix in &out {
            for member in 0..*size as u32 {
                let mut combo = prefix.clone();
                combo.push(offset + member);
                next.push(combo);
            }
        }
        out = next;
        offset += *size as u32;
    }
    out
}

/// The conditioning row a band token occupies, when the checkpoint asks
/// for one.
///
/// Resolved through the token rather than through anything numeric. A
/// band's vocabulary id starts at 2 and a table row starts at 0, so
/// handing the id over would select a real but different band and every
/// check after it would pass.
fn cond_for(shape: &ModelShape, band: Option<&str>) -> Result<Option<CondIndex>, BatchError> {
    match shape.encoding {
        CondEncoding::Prefix => Ok(None),
        CondEncoding::EveryPosition => {
            let token = band.ok_or_else(|| BatchError::NoBand {
                bands: shape.band_tokens(),
            })?;
            shape
                .band_index(token)
                .map(Some)
                .ok_or_else(|| BatchError::NoConditionRow {
                    token: token.to_string(),
                })
        }
    }
}

/// The sets one row carries, checked against what the checkpoint asks
/// for and against the row they are supposed to describe.
///
/// The two axes are checked together because a checkpoint asking for
/// both is not buildable — see [`BatchError::BothChannels`] — and this
/// is where a batch would otherwise be built for one, dropping the
/// other.
fn legal_for(
    shape: &ModelShape,
    window: &Window,
    legal: Option<&[Vec<u32>]>,
) -> Result<Option<Vec<Vec<u32>>>, BatchError> {
    if shape.legal_input && shape.encoding == CondEncoding::EveryPosition {
        return Err(BatchError::BothChannels);
    }
    match (shape.legal_input, legal) {
        (false, None) => Ok(None),
        (true, None) => Err(BatchError::LegalSetsMissing),
        (false, Some(_)) => Err(BatchError::LegalSetsUnexpected),
        (true, Some(sets)) => {
            // One per token, plus the position past the end of the row
            // — the one the model is being asked to answer at.
            let want = window.len() + 1;
            if sets.len() != want {
                return Err(BatchError::LegalSetsWidth {
                    want,
                    found: sets.len(),
                });
            }
            Ok(Some(sets.to_vec()))
        }
    }
}

impl BandBatch {
    /// One row: the position as it stands, under the band it already
    /// carries.
    ///
    /// `band` names the token in the window, and is what the
    /// conditioning index is resolved from. It is a separate argument
    /// rather than read back out of the window because the window holds
    /// a vocabulary id and the table wants a row, and those are the two
    /// numberings [`ModelShape::band_index`] exists to keep apart.
    ///
    /// `legal` is the row's legal sets from [`Window::legal_sets`], and
    /// is required exactly when the checkpoint records
    /// [`ModelShape::legal_input`]. It is an argument rather than
    /// something added afterwards so that the requirement is met at
    /// construction or not at all.
    ///
    /// # Errors
    ///
    /// The checkpoint conditions every position and no band was named,
    /// the band has no row in its table, or the legal sets and the
    /// checkpoint disagree about whether there are any — see
    /// [`BatchError::LegalSetsMissing`] and its siblings.
    ///
    /// Checked in the order [`Self::over_bands`] checks them, so that
    /// the two enforce one pairing rather than two: a shape asking for
    /// both channels is refused as [`BatchError::BothChannels`] from
    /// either constructor, whatever else is wrong with the call. The two
    /// used to report different errors for the same shape — this one
    /// resolved the band first, so a shape asking for both with no band
    /// named came back as [`BatchError::NoBand`], which names a
    /// repairable mistake for a checkpoint that cannot be run at all.
    pub fn single(
        window: &Window,
        shape: &ModelShape,
        band: Option<&str>,
        legal: Option<&[Vec<u32>]>,
    ) -> Result<Self, BatchError> {
        let sets = legal_for(shape, window, legal)?;
        let cond = cond_for(shape, band)?;
        Ok(Self {
            rows: vec![window.tokens().to_vec()],
            conds: cond.map(|cond| vec![cond]),
            conds_per_row: 1,
            legal: sets.map(|sets| vec![sets]),
        })
    }

    /// One row per band the checkpoint carries.
    ///
    /// This is what a band-to-band comparison needs, and what guidance
    /// needs to build the reference it extrapolates away from: the same
    /// position, run under every band, so the rows differ in nothing
    /// else.
    ///
    /// `legal` is the same list [`Self::single`] takes, and it is one
    /// list rather than one per band because the rows differ only at
    /// the band token — which holds no move, so no set applies to it.
    /// Every row is handed a copy.
    ///
    /// # Errors
    ///
    /// The window carries no condition to replace, a band is missing
    /// from `vocab`, a band has no row in the conditioning table, or the
    /// legal sets and the checkpoint disagree about whether there are
    /// any.
    pub fn over_bands(
        window: &Window,
        shape: &ModelShape,
        vocab: &MoveVocab,
        legal: Option<&[Vec<u32>]>,
    ) -> Result<Self, BatchError> {
        let row_legal = legal_for(shape, window, legal)?;
        let mut rows = Vec::with_capacity(shape.bands.len());
        let mut conds = Vec::with_capacity(shape.bands.len());
        for band in &shape.bands {
            let id = vocab
                .id_of(&band.token)
                .ok_or_else(|| BatchError::BandMissing {
                    token: band.token.clone(),
                })?;
            // Lands on the condition, or refuses: `Window` knows whether
            // it has one. The same write against a tail-sliced `Vec` put
            // a band id on top of a real move of the position being
            // evaluated (issue 8f9a96df).
            rows.push(window.with_band(id)?.into_tokens());
            if let Some(cond) = cond_for(shape, Some(&band.token))? {
                conds.push(cond);
            }
        }
        // Derived from the encoding rather than from whether the loop
        // happened to push anything, so an empty band list cannot turn a
        // per-position checkpoint into an unconditioned forward.
        let conds = match shape.encoding {
            CondEncoding::Prefix => None,
            CondEncoding::EveryPosition => Some(conds),
        };
        // One copy per row, from the same list, for the same reason the
        // indices are pushed in lockstep above: two rows cannot come to
        // describe two different games.
        let legal = row_legal.map(|sets| vec![sets; rows.len()]);
        Ok(Self {
            rows,
            conds,
            conds_per_row: 1,
            legal,
        })
    }

    /// One row per **combination** of the checkpoint's condition
    /// groups — the multi-slot counterpart of [`Self::over_bands`].
    ///
    /// The rows are identical copies of the window: a multi-slot corpus
    /// carries no condition token in its rows, so nothing is written
    /// into them and the combinations differ only in the indices each
    /// row's forward receives. Combinations iterate with the earlier
    /// group varying **slower** — for groups `[B, C] × [low, high]`
    /// that is `B+low, B+high, C+low, C+high` — and
    /// [`Self::combo_labels`] renders the same order, which is what
    /// ties a row of logits to the name a reader prints for it.
    ///
    /// # Errors
    ///
    /// The shape is not multi-slot (one group is [`Self::over_bands`]'s
    /// case, and answering it here would quietly duplicate that path),
    /// or it is prefix-encoded — a prefix cannot carry two slots
    /// without inventing an order for them, so no such checkpoint
    /// exists to walk — or the legal sets and the checkpoint disagree
    /// as in [`Self::single`].
    pub fn over_combos(
        window: &Window,
        shape: &ModelShape,
        legal: Option<&[Vec<u32>]>,
    ) -> Result<Self, BatchError> {
        let row_legal = legal_for(shape, window, legal)?;
        let groups = shape.effective_cond_groups();
        if groups.len() < 2 {
            return Err(BatchError::NotMultiSlot {
                groups: groups.len(),
            });
        }
        if shape.encoding != CondEncoding::EveryPosition {
            return Err(BatchError::MultiSlotNeedsEveryPosition);
        }
        let combos = combo_indices(&groups);
        let per_row = groups.len();
        let rows = vec![window.tokens().to_vec(); combos.len()];
        let conds: Vec<CondIndex> = combos
            .iter()
            .flat_map(|combo| combo.iter().map(|row| CondIndex::from_table_row(*row)))
            .collect();
        let legal = row_legal.map(|sets| vec![sets; rows.len()]);
        Ok(Self {
            rows,
            conds: Some(conds),
            conds_per_row: per_row,
            legal,
        })
    }

    /// The name of each combination row [`Self::over_combos`] builds,
    /// in the same order, each a `+`-join of one band token per group.
    pub fn combo_labels(shape: &ModelShape) -> Vec<String> {
        let groups = shape.effective_cond_groups();
        combo_indices(&groups)
            .iter()
            .map(|combo| {
                combo
                    .iter()
                    .map(|row| shape.bands[*row as usize].token.as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect()
    }

    /// How many rows this batch holds.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether it holds none. Never true for a batch these constructors
    /// build against a checkpoint with bands, but `clippy` asks for it
    /// beside `len`.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether the condition is being delivered as a forward argument.
    ///
    /// Exposed so a caller can report it. Not so a caller can act on it:
    /// [`Self::logits`] already has.
    pub fn is_conditioned(&self) -> bool {
        self.conds.is_some()
    }

    /// Whether the ids allowed at each position travel with these rows.
    ///
    /// Exposed for the same reason, and with the same caveat.
    pub fn has_legal_sets(&self) -> bool {
        self.legal.is_some()
    }

    /// The rows, for a caller that wants to inspect them.
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// Last-position logits, one row per entry.
    ///
    /// The only forward in this path. Which one to call is a property of
    /// the checkpoint, recorded on its shape, and is decided here rather
    /// than at each call site — a plain forward on a per-position
    /// checkpoint is not a shortcut to the same answer, it is a
    /// different model whose moves look no different.
    ///
    /// # The offset the legal sets are read at
    ///
    /// The sets are held in the convention the training path records
    /// them in — entry `p` is the set the token at position `p` was
    /// drawn from — and the model reads, at each input position, the set
    /// its **answer** there is drawn from. So the window taken here
    /// starts at 1, which is the offset
    /// [`crate::train::legal_input_sets`] applies to the same lists on
    /// the training side. Off by one and every set describes the
    /// position before the one it is attached to, with every shape still
    /// agreeing.
    ///
    /// # Errors
    ///
    /// The batch is empty, or the forward pass failed — including the
    /// case where the model has no conditioning table but this batch
    /// carries indices, which is a shape and a checkpoint that do not
    /// describe each other.
    pub fn logits(&self, model: &Gpt2Model, device: &Device) -> Result<Vec<Vec<f32>>, BatchError> {
        let width = match self.rows.first() {
            Some(row) if !row.is_empty() => row.len(),
            _ => return Err(BatchError::Empty),
        };
        let n = self.rows.len();
        let input = Tensor::from_vec(self.rows.concat(), (n, width), device)?;
        let out = match (&self.conds, &self.legal) {
            (Some(conds), None) => {
                model.forward_conditioned_groups(&input, conds, self.conds_per_row)?
            }
            (None, Some(sets)) => {
                let legal = LegalSets::window(sets, 1, width, device)?;
                model.forward_legal(&input, &legal)?
            }
            (None, None) => model.forward(&input)?,
            // The constructors refuse this pair, and a checkpoint that
            // asks for it describes a model `Gpt2Custom::validate` will
            // not build. What is left is a batch assembled field by
            // field, which the private fields confine to this module and
            // its tests.
            (Some(_), Some(_)) => return Err(BatchError::BothChannels),
        };
        (0..n)
            .map(|b| Ok(out.i((b, width - 1))?.to_vec1::<f32>()?))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::gpt2::fill_deterministic;
    use crate::arch::max_abs_diff_f32;
    use crate::chess::corpus::ConditionBand;
    use crate::chess::pgn::{move_from_uci_standard, uci_standard};
    use crate::chess::vocab::{legal_ids, BOS};
    use crate::chess::window::{play_row, COND_PREFIX_LEN};
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use cozy_chess::Board;

    const CTX: usize = 128;
    const BANDS: [&str; 2] = ["<elo:1100-1299>", "<elo:1900-2099>"];
    /// Plies every fixture here is taken at — past 126, where a
    /// conditioned row outgrows the context. Named because the history
    /// and the ply count handed to `legal_sets` have to be the same
    /// number, and writing it twice is how they come to differ.
    const DEEP_PLIES: usize = 130;

    fn vocab() -> MoveVocab {
        MoveVocab::new(&BANDS.map(String::from)).expect("a vocabulary over these bands")
    }

    fn shape(encoding: CondEncoding, vocab: &MoveVocab) -> ModelShape {
        let mut shape = ModelShape::compact(
            vocab.model_vocab_size(),
            BANDS
                .iter()
                .map(|token| ConditionBand::rating(0, 0, *token))
                .collect(),
        );
        shape.encoding = encoding;
        shape
    }

    /// A prefix checkpoint that reads the ids allowed at each position.
    ///
    /// Prefix because that is the only conditioning it composes with —
    /// no forward pass reads both channels.
    fn legal_shape(vocab: &MoveVocab) -> ModelShape {
        let mut shape = shape(CondEncoding::Prefix, vocab);
        shape.legal_input = true;
        shape
    }

    /// A real game of `plies` moves, as vocabulary ids.
    ///
    /// Both sides walk a knight out and back, which is legal from the
    /// opening for as long as anyone continues — the point is to reach
    /// ply 130 through actual move generation rather than to invent ids
    /// that happen to be in range.
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

    /// A toy model at the shape's own size, seeded so thresholds are a
    /// property of these weights rather than of the initialiser's draw.
    /// A 2×2 multi-slot shape: two opening families and two rating
    /// bands, per-position encoded, groups `[2, 2]`.
    fn combo_shape(vocab: &MoveVocab) -> ModelShape {
        let mut shape = ModelShape::compact(
            vocab.model_vocab_size(),
            vec![
                ConditionBand::tag_prefix("ECO", "B", "<eco:B>"),
                ConditionBand::tag_prefix("ECO", "C", "<eco:C>"),
                ConditionBand::rating(0, 1599, "<lo>"),
                ConditionBand::rating(1600, 4000, "<hi>"),
            ],
        );
        shape.encoding = CondEncoding::EveryPosition;
        shape.cond_groups = vec![2, 2];
        shape
    }

    /// The combination order is the contract a reader's labels rest on:
    /// earlier groups vary slower, and the labels render the same walk.
    #[test]
    fn combo_rows_pair_each_label_with_its_own_index_pair() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let shape = combo_shape(&vocab);
        assert_eq!(
            BandBatch::combo_labels(&shape),
            [
                "<eco:B>+<lo>",
                "<eco:B>+<hi>",
                "<eco:C>+<lo>",
                "<eco:C>+<hi>"
            ]
        );

        let moves = knight_shuffle(6, &vocab);
        let window = play_row(None, &moves, CTX).unwrap();
        let batch = BandBatch::over_combos(&window, &shape, None).unwrap();
        assert_eq!(batch.len(), 4);
        // Identical rows: nothing is written into a multi-slot row.
        assert!(batch.rows().iter().all(|r| r == &batch.rows()[0]));

        let (model, device) = model_for(&shape);
        let logits = batch.logits(&model, &device).unwrap();
        assert_eq!(logits.len(), 4);
        // The combinations are distinct conditions: the two rows that
        // share no group member must differ.
        let gap = logits[0]
            .iter()
            .zip(&logits[3])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(gap > 1e-5, "B+lo and C+hi produced the same logits");
    }

    /// The two shapes `over_combos` refuses, by name: a single group
    /// (that comparison is `over_bands`) and a prefix encoding (no such
    /// checkpoint exists to walk).
    #[test]
    fn combos_refuse_a_single_group_and_a_prefix_encoding() {
        let vocab = MoveVocab::new(&[]).unwrap();
        let moves = knight_shuffle(6, &vocab);
        let window = play_row(None, &moves, CTX).unwrap();

        let single = shape(CondEncoding::EveryPosition, &vocab);
        assert!(matches!(
            BandBatch::over_combos(&window, &single, None),
            Err(BatchError::NotMultiSlot { groups: 1 })
        ));

        let mut prefix = combo_shape(&vocab);
        prefix.encoding = CondEncoding::Prefix;
        assert!(matches!(
            BandBatch::over_combos(&window, &prefix, None),
            Err(BatchError::MultiSlotNeedsEveryPosition)
        ));
    }

    fn model_for(shape: &ModelShape) -> (Gpt2Model, Device) {
        let cfg = shape.config(Device::Cpu, DType::F32);
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).expect("a model at this shape");
        fill_deterministic(&varmap, 0x0806_2026).expect("deterministic weights");
        (model, cfg.device)
    }

    /// The same game's legal ids at every ply, plus the position it
    /// ends in — what a reader that replayed the history holds.
    ///
    /// **These sets repeat with period four.** Both sides walk a knight
    /// out and back, so ply `m` and ply `m + 4` are the same position
    /// and offer the same moves. A mapping error of four plies — which
    /// is exactly what indexing this fixture's history by row position
    /// produces, since its window starts at ply 4 — leaves this list
    /// byte-identical and cannot be detected against it at all. That is
    /// how the first draft of the windowing test passed against a wrong
    /// mapping; `the_knight_shuffle_hides_a_four_ply_mapping_error`
    /// holds the property so a later test cannot inherit it unwarned. A
    /// test that needs two plies to differ needs a different game —
    /// `a_pawn_walk` in `tests/chess_legal_input_bake.rs` is one.
    fn knight_at_ply(plies: usize, vocab: &MoveVocab) -> Vec<Vec<u32>> {
        let cycle = ["b1c3", "b8c6", "c3b1", "c6b8"];
        let mut board = Board::default();
        let mut out = Vec::with_capacity(plies + 1);
        for ply in 0..plies {
            out.push(legal_ids(&board, vocab));
            let mv = move_from_uci_standard(&board, cycle[ply % cycle.len()])
                .unwrap_or_else(|| panic!("not legal at ply {ply}"));
            board.play_unchecked(mv);
        }
        out.push(legal_ids(&board, vocab));
        out
    }

    /// A window at ply 130, past the point where a conditioned row
    /// outgrows the context — the regime the interactive path reaches
    /// and the measurement path barely does.
    fn deep_window(vocab: &MoveVocab) -> Window {
        let moves = knight_shuffle(DEEP_PLIES, vocab);
        let low = vocab.id_of(BANDS[0]).expect("the low band");
        let window = play_row(Some(low), &moves, CTX).expect("a windowed row");
        assert_eq!(window.len(), CTX, "a 130-ply game should be windowed");
        assert_eq!(window.band(), Some(low), "the band must survive windowing");
        window
    }

    /// The sets a reader recovers for [`deep_window`], from the same
    /// game it was built from.
    ///
    /// The history and the ply count are produced here together, since
    /// `legal_sets` checks one against the other and a fixture that got
    /// them from two places would be testing its own bookkeeping.
    fn deep_sets(window: &Window, vocab: &MoveVocab) -> Vec<Vec<u32>> {
        window
            .legal_sets(&knight_at_ply(DEEP_PLIES, vocab), DEEP_PLIES)
            .expect("the history covers this window")
    }

    /// The prefix convention carries no indices, so the batch runs the
    /// plain forward.
    #[test]
    fn a_prefix_batch_carries_no_conditioning_argument() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let window = deep_window(&vocab);
        let batch = BandBatch::over_bands(&window, &shape, &vocab, None).unwrap();
        assert_eq!(batch.len(), BANDS.len());
        assert!(!batch.is_conditioned());
        // The rows still differ, because the band is token 1 of each —
        // and they differ *only* there, which is what makes a band-to-
        // band comparison a comparison of bands.
        assert_ne!(batch.rows()[0], batch.rows()[1]);
        assert_eq!(batch.rows()[0][0], batch.rows()[1][0], "both begin at BOS");
        assert_ne!(batch.rows()[0][1], batch.rows()[1][1], "differing bands");
        assert_eq!(
            batch.rows()[0][2..],
            batch.rows()[1][2..],
            "every move must be the same move"
        );
    }

    /// The per-position convention carries one index per row, naming the
    /// band that row also carries as a token.
    #[test]
    fn a_per_position_batch_pairs_each_row_with_its_own_table_row() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let window = deep_window(&vocab);
        let batch = BandBatch::over_bands(&window, &shape, &vocab, None).unwrap();
        assert!(batch.is_conditioned());
        let conds = batch.conds.as_ref().expect("indices");
        assert_eq!(conds.len(), batch.len(), "one index per row");
        for (row, cond) in batch.rows().iter().zip(conds) {
            // The two channels have to name the same band: the token at
            // position 1 and the table row handed to the forward.
            let token = vocab.token_of(row[1]).expect("a band token at position 1");
            assert_eq!(
                shape.band_index(token),
                Some(*cond),
                "row and index disagree about which band this is"
            );
        }
    }

    /// The construction a reader actually performs, at ply 130, giving
    /// band-dependent logits through a channel windowing cannot touch.
    ///
    /// And the comparison that makes it mean something: the same batch
    /// with its conditioning removed scores differently, so a reader
    /// that dropped the indices would not be producing the same numbers
    /// by another route.
    #[test]
    fn dropping_the_conditioning_argument_changes_what_the_model_plays() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let (model, device) = model_for(&shape);
        let window = deep_window(&vocab);

        let batch = BandBatch::over_bands(&window, &shape, &vocab, None).unwrap();
        let conditioned = batch.logits(&model, &device).unwrap();
        assert_eq!(conditioned.len(), BANDS.len());

        // Same rows, no indices — what a reader calling the plain
        // forward would have produced.
        let unconditioned = BandBatch {
            rows: batch.rows().to_vec(),
            conds: None,
            conds_per_row: 1,
            legal: None,
        };
        let plain = unconditioned.logits(&model, &device).unwrap();

        let device_tensor = |v: &Vec<f32>| Tensor::from_vec(v.clone(), (v.len(),), &device);
        let spread = max_abs_diff_f32(
            &device_tensor(&conditioned[0]).unwrap(),
            &device_tensor(&conditioned[1]).unwrap(),
        )
        .unwrap();
        assert!(
            spread > 1e-5,
            "the two bands produced the same logits at ply 130 (spread {spread})"
        );

        let dropped = max_abs_diff_f32(
            &device_tensor(&conditioned[0]).unwrap(),
            &device_tensor(&plain[0]).unwrap(),
        )
        .unwrap();
        assert!(
            dropped > 1e-5,
            "dropping the conditioning argument changed nothing, so this test could not \
             catch a reader that drops it (spread {dropped})"
        );
    }

    /// The prefix arm, at the depth where it used to stop working.
    ///
    /// `two_bands_differ_at_ply_130` over in [`crate::chess::window`]
    /// shows the two *rows* differ there. This shows the model reads
    /// that difference, which is a separate claim: a row can differ at
    /// a token the model then ignores, and on this arm that token is
    /// the band's only channel — there is no argument to fall back on.
    ///
    /// The second half is what makes this a fence rather than an
    /// observation. Under the tail slice `play_row` replaced,
    /// `[BOS, band] + moves` cut to its last `ctx` tokens loses both
    /// prefix tokens first, so every band yields the same row and the
    /// condition can do nothing at all. That construction is rebuilt
    /// here and asserted dead, so the half above is attributable to the
    /// windowing repair rather than to these particular weights.
    ///
    /// This arm is the one to fence: three of the four rows in the
    /// plan's decision table adopt prefix as the operating point, so it
    /// is most likely what a person ends up playing against.
    #[test]
    fn the_prefix_band_still_reaches_the_model_at_ply_130() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let (model, device) = model_for(&shape);
        let window = deep_window(&vocab);

        let batch = BandBatch::over_bands(&window, &shape, &vocab, None).unwrap();
        assert!(!batch.is_conditioned(), "the prefix arm takes no argument");
        let logits = batch.logits(&model, &device).unwrap();

        let device_tensor = |v: &Vec<f32>| Tensor::from_vec(v.clone(), (v.len(),), &device);
        let spread = max_abs_diff_f32(
            &device_tensor(&logits[0]).unwrap(),
            &device_tensor(&logits[1]).unwrap(),
        )
        .unwrap();
        assert!(
            spread > 1e-5,
            "the two bands produced the same logits at ply 130, so the band in the row \
             reached nothing (spread {spread})"
        );

        // The construction this window replaced: keep the last `ctx`
        // tokens of the whole row and let the prefix fall off the front.
        let moves = knight_shuffle(DEEP_PLIES, &vocab);
        let tail_rows: Vec<Vec<u32>> = BANDS
            .iter()
            .map(|token| {
                let band = vocab.id_of(token).expect("a band token");
                let mut row = vec![BOS, band];
                row.extend_from_slice(&moves);
                row[row.len() - CTX..].to_vec()
            })
            .collect();
        assert_eq!(
            tail_rows[0], tail_rows[1],
            "a tail slice at this depth should leave the bands indistinguishable; if it \
             does not, this game is too short to exercise the regime and the assertion \
             below proves nothing"
        );

        let naive = BandBatch {
            rows: tail_rows,
            conds: None,
            conds_per_row: 1,
            legal: None,
        };
        let naive_logits = naive.logits(&model, &device).unwrap();
        let naive_spread = max_abs_diff_f32(
            &device_tensor(&naive_logits[0]).unwrap(),
            &device_tensor(&naive_logits[1]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            naive_spread, 0.0,
            "identical rows must give identical logits; a difference here means the \
             comparison above is measuring something other than the band"
        );
    }

    /// The single-row form, which is what a player scores a position
    /// with, gets the same treatment.
    #[test]
    fn a_single_row_is_conditioned_too() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let (model, device) = model_for(&shape);
        let window = deep_window(&vocab);

        let low = BandBatch::single(&window, &shape, Some(BANDS[0]), None).unwrap();
        assert!(low.is_conditioned());
        assert_eq!(low.len(), 1);

        // Same row, other band: only the index differs, so anything that
        // separates the two came through it.
        let high = BandBatch::single(&window, &shape, Some(BANDS[1]), None).unwrap();
        assert_eq!(low.rows(), high.rows(), "the rows are the same row");
        assert_ne!(low.conds, high.conds);

        let a = low.logits(&model, &device).unwrap();
        let b = high.logits(&model, &device).unwrap();
        let t = |v: &Vec<f32>| Tensor::from_vec(v.clone(), (v.len(),), &device);
        let spread = max_abs_diff_f32(&t(&a[0]).unwrap(), &t(&b[0]).unwrap()).unwrap();
        assert!(
            spread > 1e-5,
            "the conditioning argument alone did nothing (spread {spread})"
        );
    }

    /// A per-position checkpoint cannot be run without a band, and says
    /// so rather than running with the condition dropped.
    #[test]
    fn a_per_position_checkpoint_refuses_to_run_unconditioned() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let window = deep_window(&vocab);
        let err = BandBatch::single(&window, &shape, None, None).unwrap_err();
        assert!(matches!(err, BatchError::NoBand { .. }), "{err:?}");
    }

    /// A prefix checkpoint may be run without one: there is no table,
    /// and the band it carries is in the row.
    #[test]
    fn a_prefix_checkpoint_needs_no_band_argument() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let window = deep_window(&vocab);
        let batch = BandBatch::single(&window, &shape, None, None).unwrap();
        assert!(!batch.is_conditioned());
    }

    /// A band the checkpoint does not carry has no row, and is refused
    /// rather than resolved to a neighbouring one.
    #[test]
    fn a_band_outside_the_checkpoints_list_is_refused() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let window = deep_window(&vocab);
        let err = BandBatch::single(&window, &shape, Some("<elo:1500-1699>"), None).unwrap_err();
        assert!(matches!(err, BatchError::NoConditionRow { .. }), "{err:?}");
    }

    /// An unconditioned window has no band slot to write into, so the
    /// per-band form refuses rather than putting a band id on a move.
    #[test]
    fn a_window_with_no_condition_cannot_be_run_over_bands() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let moves = knight_shuffle(20, &vocab);
        let window = play_row(None, &moves, CTX).unwrap();
        let err = BandBatch::over_bands(&window, &shape, &vocab, None).unwrap_err();
        assert!(
            matches!(err, BatchError::Window(WindowError::NotConditioned { .. })),
            "{err:?}"
        );
    }

    /// A legality checkpoint runs, at ply 130, on sets recovered from a
    /// replayed game — the regime the readers reach and the one a naive
    /// mapping gets wrong.
    ///
    /// The second half is what makes it a fence. The same rows under the
    /// sets moved one position along the row give different logits, so
    /// the channel is live: a model that ignored the sets would score
    /// the two the same. It says nothing about which mapping is right;
    /// that is [`crate::chess::window::Window::legal_sets`]'s tests and
    /// the cross-check against the dataset in
    /// `tests/chess_legal_input_bake.rs`.
    ///
    /// Nor is it the perturbation a naive reader would make. Indexing
    /// this fixture's history by row position moves every set four
    /// plies, and `knight_at_ply` repeats with period four, so that
    /// mutation leaves the list byte-identical and could not be
    /// detected here — see
    /// [`the_knight_shuffle_hides_a_four_ply_mapping_error`].
    #[test]
    fn a_legality_batch_runs_on_sets_recovered_from_the_history() {
        let vocab = vocab();
        let shape = legal_shape(&vocab);
        let (model, device) = model_for(&shape);
        let window = deep_window(&vocab);

        let sets = deep_sets(&window, &vocab);
        let batch = BandBatch::over_bands(&window, &shape, &vocab, Some(&sets)).unwrap();
        assert!(batch.has_legal_sets());
        assert!(!batch.is_conditioned(), "prefix carries no argument");
        let with_sets = batch.logits(&model, &device).unwrap();
        assert_eq!(with_sets.len(), BANDS.len());

        // The same rows under the sets moved one position along: every
        // position handed the one before it.
        let mut shifted = sets.clone();
        shifted.pop();
        shifted.insert(0, Vec::new());
        let off_by_one = BandBatch::over_bands(&window, &shape, &vocab, Some(&shifted)).unwrap();
        let other = off_by_one.logits(&model, &device).unwrap();

        let t = |v: &Vec<f32>| Tensor::from_vec(v.clone(), (v.len(),), &device);
        let spread = max_abs_diff_f32(&t(&with_sets[0]).unwrap(), &t(&other[0]).unwrap()).unwrap();
        assert!(
            spread > 1e-5,
            "moving the legal sets one position along changed nothing, so nothing here could \
             catch a mapping that is off by one (spread {spread})"
        );
    }

    /// The trap in [`knight_at_ply`], constructed rather than asserted
    /// in prose: this fixture cannot see a four-ply mapping error, which
    /// is the size of the error a windowed row invites.
    ///
    /// Two facts, and the second is the one that matters. The sets
    /// repeat with period four; and the list a reader that indexed the
    /// history by row position would build is therefore the same list,
    /// entry for entry, as the correct mapping. Any test that perturbs a
    /// mapping on this game has to know that.
    #[test]
    fn the_knight_shuffle_hides_a_four_ply_mapping_error() {
        let vocab = vocab();
        let at_ply = knight_at_ply(DEEP_PLIES, &vocab);
        for m in 0..at_ply.len() - 4 {
            assert_eq!(at_ply[m], at_ply[m + 4], "ply {m} against ply {}", m + 4);
        }
        // And not because every ply carries the same set, which would
        // make the loop above hold of anything.
        assert_ne!(at_ply[0], at_ply[1], "the two sides offer different moves");

        // What a reader indexing by row position builds: the prefix
        // empty, then the history from its own beginning.
        let window = deep_window(&vocab);
        let correct = deep_sets(&window, &vocab);
        let mut by_row_position = vec![Vec::new(); COND_PREFIX_LEN];
        by_row_position.extend(at_ply[..window.len() + 1 - COND_PREFIX_LEN].iter().cloned());
        assert_eq!(by_row_position.len(), correct.len());
        assert_eq!(
            by_row_position, correct,
            "this fixture can see a row-position mapping after all, and the note on \
             `knight_at_ply` is wrong"
        );
    }

    /// The single-row form carries them too, and the sets reach the
    /// model: two different set lists on one row give two answers.
    #[test]
    fn a_single_legality_row_reads_its_sets() {
        let vocab = vocab();
        let shape = legal_shape(&vocab);
        let (model, device) = model_for(&shape);
        let window = deep_window(&vocab);
        let sets = deep_sets(&window, &vocab);

        let batch = BandBatch::single(&window, &shape, Some(BANDS[0]), Some(&sets)).unwrap();
        assert!(batch.has_legal_sets());
        let a = batch.logits(&model, &device).unwrap();

        // One position's set narrowed to a single move. Nothing else
        // differs, so what separates the two came through the channel.
        let mut narrowed = sets.clone();
        let last = narrowed.len() - 1;
        narrowed[last] = vec![narrowed[last][0]];
        let b = BandBatch::single(&window, &shape, Some(BANDS[0]), Some(&narrowed))
            .unwrap()
            .logits(&model, &device)
            .unwrap();

        let t = |v: &Vec<f32>| Tensor::from_vec(v.clone(), (v.len(),), &device);
        let spread = max_abs_diff_f32(&t(&a[0]).unwrap(), &t(&b[0]).unwrap()).unwrap();
        assert!(spread > 1e-5, "the legality input did nothing ({spread})");
    }

    /// A checkpoint that reads the sets cannot be batched without them.
    #[test]
    fn a_legality_checkpoint_without_sets_is_refused() {
        let vocab = vocab();
        let shape = legal_shape(&vocab);
        let window = deep_window(&vocab);
        assert!(matches!(
            BandBatch::single(&window, &shape, Some(BANDS[0]), None),
            Err(BatchError::LegalSetsMissing)
        ));
        assert!(matches!(
            BandBatch::over_bands(&window, &shape, &vocab, None),
            Err(BatchError::LegalSetsMissing)
        ));
    }

    /// And an ordinary checkpoint cannot be batched with them: the sets
    /// would reach a model with no table to read them.
    #[test]
    fn sets_for_a_checkpoint_without_the_table_are_refused() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let window = deep_window(&vocab);
        let sets = deep_sets(&window, &vocab);
        assert!(matches!(
            BandBatch::single(&window, &shape, None, Some(&sets)),
            Err(BatchError::LegalSetsUnexpected)
        ));
    }

    /// Sets built against another row are refused rather than windowed
    /// into place, since every entry of them describes another position.
    #[test]
    fn sets_that_do_not_cover_the_row_are_refused() {
        let vocab = vocab();
        let shape = legal_shape(&vocab);
        let window = deep_window(&vocab);
        let mut sets = deep_sets(&window, &vocab);
        sets.pop();
        let err = BandBatch::single(&window, &shape, Some(BANDS[0]), Some(&sets)).unwrap_err();
        assert!(
            matches!(
                err,
                BatchError::LegalSetsWidth {
                    want,
                    found,
                } if want == CTX + 1 && found == CTX
            ),
            "{err:?}"
        );
    }

    /// The pair no forward pass reads. `Gpt2Custom::validate` refuses to
    /// build such a model, so this is a shape that cannot be loaded —
    /// and a batch built for it would have to drop one of the two.
    ///
    /// Both constructors, and with the rest of the call right or wrong,
    /// because the refusal is a property of the shape. It reads the same
    /// from either side: the constructors check the channels in the same
    /// order, which they did not while `single` resolved the band first
    /// and answered `NoBand` for a checkpoint that cannot be run at all.
    #[test]
    fn a_shape_asking_for_both_channels_is_refused() {
        let vocab = vocab();
        let mut shape = shape(CondEncoding::EveryPosition, &vocab);
        shape.legal_input = true;
        let window = deep_window(&vocab);
        let sets = deep_sets(&window, &vocab);
        assert!(matches!(
            BandBatch::single(&window, &shape, Some(BANDS[0]), Some(&sets)),
            Err(BatchError::BothChannels)
        ));
        assert!(matches!(
            BandBatch::single(&window, &shape, None, Some(&sets)),
            Err(BatchError::BothChannels)
        ));
        assert!(matches!(
            BandBatch::single(&window, &shape, None, None),
            Err(BatchError::BothChannels)
        ));
        assert!(matches!(
            BandBatch::over_bands(&window, &shape, &vocab, Some(&sets)),
            Err(BatchError::BothChannels)
        ));
        assert!(matches!(
            BandBatch::over_bands(&window, &shape, &vocab, None),
            Err(BatchError::BothChannels)
        ));
    }
}
