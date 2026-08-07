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

use candle_core::{Device, IndexOp, Tensor};
use thiserror::Error;

use crate::arch::{CondIndex, Gpt2Model};
use crate::chess::vocab::MoveVocab;
use crate::chess::window::{Window, WindowError};
use crate::chess::{CondEncoding, ModelShape};

/// Why a batch could not be built or scored.
#[derive(Debug, Error)]
pub enum BatchError {
    /// A band could not be written into a row.
    #[error(transparent)]
    Window(#[from] WindowError),

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
}

/// Rows to feed the model, and the conditioning that travels with them.
///
/// Built by [`BandBatch::single`] or [`BandBatch::over_bands`]; there is
/// no other constructor, and the fields are private, so the invariant
/// below cannot be assembled wrongly from outside.
///
/// **Invariant**: `conds` is `Some` exactly when the checkpoint
/// conditions every position, and when it is, it holds one index per
/// row, naming the same band that row carries at position 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandBatch {
    rows: Vec<Vec<u32>>,
    conds: Option<Vec<CondIndex>>,
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
    /// # Errors
    ///
    /// The checkpoint conditions every position and no band was named,
    /// or the band has no row in its table.
    pub fn single(
        window: &Window,
        shape: &ModelShape,
        band: Option<&str>,
    ) -> Result<Self, BatchError> {
        Ok(Self {
            rows: vec![window.tokens().to_vec()],
            conds: cond_for(shape, band)?.map(|cond| vec![cond]),
        })
    }

    /// One row per band the checkpoint carries.
    ///
    /// This is what a band-to-band comparison needs, and what guidance
    /// needs to build the reference it extrapolates away from: the same
    /// position, run under every band, so the rows differ in nothing
    /// else.
    ///
    /// # Errors
    ///
    /// The window carries no condition to replace, a band is missing
    /// from `vocab`, or a band has no row in the conditioning table.
    pub fn over_bands(
        window: &Window,
        shape: &ModelShape,
        vocab: &MoveVocab,
    ) -> Result<Self, BatchError> {
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
        Ok(Self { rows, conds })
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
        let out = match &self.conds {
            Some(conds) => model.forward_conditioned(&input, conds)?,
            None => model.forward(&input)?,
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
    use crate::chess::vocab::BOS;
    use crate::chess::window::play_row;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use cozy_chess::Board;

    const CTX: usize = 128;
    const BANDS: [&str; 2] = ["<elo:1100-1299>", "<elo:1900-2099>"];

    fn vocab() -> MoveVocab {
        MoveVocab::new(&BANDS.map(String::from)).expect("a vocabulary over these bands")
    }

    fn shape(encoding: CondEncoding, vocab: &MoveVocab) -> ModelShape {
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
        shape.encoding = encoding;
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
    fn model_for(shape: &ModelShape) -> (Gpt2Model, Device) {
        let cfg = shape.config(Device::Cpu, DType::F32);
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, cfg.dtype, &cfg.device);
        let model = Gpt2Model::new(&cfg, vs).expect("a model at this shape");
        fill_deterministic(&varmap, 0x0806_2026).expect("deterministic weights");
        (model, cfg.device)
    }

    /// A window at ply 130, past the point where a conditioned row
    /// outgrows the context — the regime the interactive path reaches
    /// and the measurement path barely does.
    fn deep_window(vocab: &MoveVocab) -> Window {
        let moves = knight_shuffle(130, vocab);
        let low = vocab.id_of(BANDS[0]).expect("the low band");
        let window = play_row(Some(low), &moves, CTX).expect("a windowed row");
        assert_eq!(window.len(), CTX, "a 130-ply game should be windowed");
        assert_eq!(window.band(), Some(low), "the band must survive windowing");
        window
    }

    /// The prefix convention carries no indices, so the batch runs the
    /// plain forward.
    #[test]
    fn a_prefix_batch_carries_no_conditioning_argument() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let window = deep_window(&vocab);
        let batch = BandBatch::over_bands(&window, &shape, &vocab).unwrap();
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
        let batch = BandBatch::over_bands(&window, &shape, &vocab).unwrap();
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

        let batch = BandBatch::over_bands(&window, &shape, &vocab).unwrap();
        let conditioned = batch.logits(&model, &device).unwrap();
        assert_eq!(conditioned.len(), BANDS.len());

        // Same rows, no indices — what a reader calling the plain
        // forward would have produced.
        let unconditioned = BandBatch {
            rows: batch.rows().to_vec(),
            conds: None,
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

        let batch = BandBatch::over_bands(&window, &shape, &vocab).unwrap();
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
        let moves = knight_shuffle(130, &vocab);
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

        let low = BandBatch::single(&window, &shape, Some(BANDS[0])).unwrap();
        assert!(low.is_conditioned());
        assert_eq!(low.len(), 1);

        // Same row, other band: only the index differs, so anything that
        // separates the two came through it.
        let high = BandBatch::single(&window, &shape, Some(BANDS[1])).unwrap();
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
        let err = BandBatch::single(&window, &shape, None).unwrap_err();
        assert!(matches!(err, BatchError::NoBand { .. }), "{err:?}");
    }

    /// A prefix checkpoint may be run without one: there is no table,
    /// and the band it carries is in the row.
    #[test]
    fn a_prefix_checkpoint_needs_no_band_argument() {
        let vocab = vocab();
        let shape = shape(CondEncoding::Prefix, &vocab);
        let window = deep_window(&vocab);
        let batch = BandBatch::single(&window, &shape, None).unwrap();
        assert!(!batch.is_conditioned());
    }

    /// A band the checkpoint does not carry has no row, and is refused
    /// rather than resolved to a neighbouring one.
    #[test]
    fn a_band_outside_the_checkpoints_list_is_refused() {
        let vocab = vocab();
        let shape = shape(CondEncoding::EveryPosition, &vocab);
        let window = deep_window(&vocab);
        let err = BandBatch::single(&window, &shape, Some("<elo:1500-1699>")).unwrap_err();
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
        let err = BandBatch::over_bands(&window, &shape, &vocab).unwrap_err();
        assert!(
            matches!(err, BatchError::Window(WindowError::NotConditioned { .. })),
            "{err:?}"
        );
    }
}
