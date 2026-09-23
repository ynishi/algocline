//! One vector for a sequence.
//!
//! A model's hidden state is `[batch, seq, dim]` — one vector per
//! position — and an embedding is one vector per sequence. Pooling is
//! the step between, and which pooling is a real choice rather than a
//! detail: the three below disagree about what a sequence's meaning
//! sits in, and the right answer depends on how the model was trained.
//!
//! - [`Pooling::Mean`] averages every position. The usual default for a
//!   sequence embedding, and what Sentence-BERT's own guidance
//!   recommends for models with no pooling head of their own.
//! - [`Pooling::Last`] takes the final position. What a decoder-only
//!   model's own objective builds: that position is the only one that
//!   has read the whole sequence, which is why it is the one the
//!   language-model head is asked from.
//! - [`Pooling::Max`] takes the elementwise maximum, which keeps the
//!   strongest activation of each feature rather than its average.
//!
//! # Padding
//!
//! `lengths` says how many positions of each row are real. Without it
//! the padded tail is pooled along with the content: a mean over a
//! half-padded row is half an embedding of the padding, and a "last"
//! pool lands on filler. It is optional because a batch of one — the
//! common case for an embedding call — has nothing to pad.

use candle_core::{Result as CandleResult, Tensor};

/// How a sequence's positions become one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// Mean over the real positions.
    #[default]
    Mean,
    /// The last real position.
    Last,
    /// Elementwise maximum over the real positions.
    Max,
}

impl Pooling {
    /// Parse the wire form written by callers (Lua bridge, JSON config).
    ///
    /// `None` on an unknown name, so the caller can list the
    /// alternatives rather than surface a generic parse error.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "mean" | "avg" | "average" => Some(Self::Mean),
            "last" => Some(Self::Last),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// Every wire name this version accepts.
    pub const NAMES: [&'static str; 5] = ["mean", "avg", "average", "last", "max"];
}

/// Pool a `[batch, seq, dim]` hidden state into `[batch, dim]`.
///
/// `lengths`, when given, holds one real-position count per row; a
/// count of zero, or one past the sequence, is refused rather than
/// clamped — both mean the caller's bookkeeping and the tensor disagree,
/// and pooling something anyway would answer with a vector that
/// describes neither.
pub fn pool(hidden: &Tensor, kind: Pooling, lengths: Option<&[usize]>) -> CandleResult<Tensor> {
    let (b, t, _d) = hidden.dims3()?;
    if let Some(lengths) = lengths {
        if lengths.len() != b {
            return Err(candle_core::Error::Msg(format!(
                "pooling: {} length(s) for a batch of {b}",
                lengths.len()
            )));
        }
        for (row, len) in lengths.iter().enumerate() {
            if *len == 0 || *len > t {
                return Err(candle_core::Error::Msg(format!(
                    "pooling: row {row} declares {len} real position(s) of {t}"
                )));
            }
        }
    }

    // Row by row, because each row's real length is its own. A mask and
    // a single reduction would be fewer ops on a large batch; the
    // batches here are small, and this way every row's arithmetic is
    // visibly the arithmetic of that row alone.
    let mut rows = Vec::with_capacity(b);
    for row in 0..b {
        let len = lengths.map(|l| l[row]).unwrap_or(t);
        let seq = hidden.narrow(0, row, 1)?.squeeze(0)?.narrow(0, 0, len)?; // [len, dim]
        let pooled = match kind {
            Pooling::Mean => seq.mean(0)?,
            Pooling::Last => seq.narrow(0, len - 1, 1)?.squeeze(0)?,
            Pooling::Max => seq.max(0)?,
        };
        rows.push(pooled.unsqueeze(0)?); // [1, dim]
    }
    Tensor::cat(&rows, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// `[1, 3, 2]` holding `[[1,2],[3,4],[5,6]]`.
    fn hidden() -> Tensor {
        Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn row(t: &Tensor) -> Vec<f32> {
        t.squeeze(0).unwrap().to_vec1().unwrap()
    }

    #[test]
    fn the_three_poolings_disagree_as_they_should() {
        let h = hidden();
        assert_eq!(row(&pool(&h, Pooling::Mean, None).unwrap()), vec![3.0, 4.0]);
        assert_eq!(row(&pool(&h, Pooling::Last, None).unwrap()), vec![5.0, 6.0]);
        assert_eq!(row(&pool(&h, Pooling::Max, None).unwrap()), vec![5.0, 6.0]);
    }

    #[test]
    fn a_length_keeps_the_padding_out_of_the_answer() {
        let h = hidden();
        // Only the first two positions are real, so the mean is of those
        // two and "last" is the second — not the filler behind them.
        assert_eq!(
            row(&pool(&h, Pooling::Mean, Some(&[2])).unwrap()),
            vec![2.0, 3.0]
        );
        assert_eq!(
            row(&pool(&h, Pooling::Last, Some(&[2])).unwrap()),
            vec![3.0, 4.0]
        );
        assert_eq!(
            row(&pool(&h, Pooling::Max, Some(&[2])).unwrap()),
            vec![3.0, 4.0]
        );
    }

    #[test]
    fn each_row_of_a_batch_is_pooled_at_its_own_length() {
        let h = Tensor::from_vec(
            vec![
                1.0f32, 1.0, 3.0, 3.0, 9.0, 9.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0,
            ],
            (2, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let pooled = pool(&h, Pooling::Mean, Some(&[2, 3])).unwrap();
        let values: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
        assert_eq!(values[0], vec![2.0, 2.0], "row 0 averages its first two");
        assert_eq!(values[1], vec![4.0, 4.0], "row 1 averages all three");
    }

    #[test]
    fn a_length_the_tensor_cannot_honour_is_refused() {
        let h = hidden();
        for lengths in [vec![0usize], vec![4]] {
            assert!(
                pool(&h, Pooling::Mean, Some(&lengths)).is_err(),
                "lengths {lengths:?} must be refused"
            );
        }
        assert!(
            pool(&h, Pooling::Mean, Some(&[1, 1])).is_err(),
            "two lengths for a batch of one must be refused"
        );
    }

    #[test]
    fn the_wire_names_round_trip() {
        assert_eq!(Pooling::parse("mean"), Some(Pooling::Mean));
        assert_eq!(Pooling::parse("last"), Some(Pooling::Last));
        assert_eq!(Pooling::parse("max"), Some(Pooling::Max));
        assert_eq!(Pooling::parse("cls"), None);
        for name in Pooling::NAMES {
            assert!(Pooling::parse(name).is_some(), "{name} is advertised");
        }
    }

    #[test]
    fn pooling_keeps_the_dtype_it_was_given() {
        let h = hidden().to_dtype(DType::BF16).unwrap();
        let pooled = pool(&h, Pooling::Mean, None).unwrap();
        assert_eq!(pooled.dtype(), DType::BF16);
        assert_eq!(pooled.dims(), &[1, 2]);
    }
}
