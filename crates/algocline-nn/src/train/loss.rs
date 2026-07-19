//! Loss functions used by the training loop.
//!
//! Introduces a small [`Loss`] trait so the loop stays agnostic to the
//! specific reduction and can accept plain cross-entropy (Full FT), a
//! hard-label distillation variant, or a future KL-soft variant without
//! changing the loop signature.
//!
//! The optional `loss_mask` on `Loss::compute` is what lets a
//! distillation caller zero out prompt-region tokens: for Full FT the
//! caller passes `None`; distillation passes a per-token 0/1 mask so
//! only the response region contributes to the loss.

use candle_core::{Result as CandleResult, Tensor};

/// Reduction strategy applied inside a `Loss::compute` call.
///
/// Kept minimal: the current callers only need `Mean` (average loss
/// across all non-masked positions). A `Sum` variant is easy to add
/// later if a scheduler needs unnormalised loss.
#[derive(Debug, Clone, Copy, Default)]
pub enum Reduction {
    /// Mean across all positions with `loss_mask == 1` (all positions
    /// when the mask is `None`).
    #[default]
    Mean,
}

/// Loss function on `[batch, seq, vocab]` logits.
///
/// `targets` is `[batch, seq]` of `u32` next-token ids. `loss_mask`,
/// when `Some`, is `[batch, seq]` of `f32` (`1.0` = counted, `0.0` =
/// ignored). The trainer loop passes `None` for standard Full FT and a
/// prompt-masked tensor for distillation.
pub trait Loss {
    /// Compute the scalar loss.
    ///
    /// Implementations must not mutate any input tensor.
    fn compute(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        loss_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor>;
}

/// Standard token-level cross-entropy against a hard target.
///
/// Semantics match nanoGPT's loss:
/// `-log_softmax(logits, dim=-1).gather(targets)`, then averaged over
/// masked-in positions. When no mask is given, every `[batch, seq]`
/// position contributes.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropyLoss {
    /// Reduction applied after the per-position cross entropy.
    pub reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Construct with the default (mean) reduction.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Hard-label distillation loss.
///
/// Semantically a thin wrapper around [`CrossEntropyLoss`] — the
/// distillation caller supplies a per-token loss mask (via
/// `TeacherCardDataset`) that zeroes out prompt-region positions,
/// leaving only the response region driving the gradient. A separate
/// named type helps the training loop and Card metadata carry the
/// distinction ("this run was a distillation, not a Full FT") without
/// changing the underlying loss math.
///
/// A future KL-soft variant (needing teacher log-probs, deferred) can
/// live alongside this type without touching the `Loss` trait or the
/// training loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardLabelDistillLoss {
    inner: CrossEntropyLoss,
}

impl HardLabelDistillLoss {
    /// Construct with the default (mean) reduction.
    pub fn new() -> Self {
        Self::default()
    }

    /// Access the underlying cross-entropy loss (mostly useful for
    /// tests that want to compare against a plain CE reference).
    pub fn inner(&self) -> &CrossEntropyLoss {
        &self.inner
    }
}

impl Loss for HardLabelDistillLoss {
    fn compute(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        loss_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        self.inner.compute(logits, targets, loss_mask)
    }
}

impl Loss for CrossEntropyLoss {
    fn compute(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        loss_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Shape checks — surface an early error rather than a candle
        // kernel error later on.
        let (b, t, v) = logits.dims3()?;
        let (tb, tt) = targets.dims2()?;
        if (b, t) != (tb, tt) {
            return Err(candle_core::Error::Msg(format!(
                "cross_entropy: logits shape [{b}, {t}, {v}] does not \
                 match targets shape [{tb}, {tt}]"
            )));
        }
        if let Some(mask) = loss_mask {
            let (mb, mt) = mask.dims2()?;
            if (mb, mt) != (b, t) {
                return Err(candle_core::Error::Msg(format!(
                    "cross_entropy: loss_mask shape [{mb}, {mt}] does \
                     not match [{b}, {t}]"
                )));
            }
        }

        // Flatten batch+seq so we can gather with a 1-D index and stay
        // on candle-nn's supported log_softmax path.
        let logits_flat = logits.reshape((b * t, v))?;
        let targets_flat = targets
            .reshape((b * t,))?
            .to_dtype(candle_core::DType::U32)?;

        // log_softmax over the vocab axis, then negative log-likelihood
        // by gathering the target-column log-prob per row.
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?; // [B*T, V]
        let nll = log_probs
            .gather(&targets_flat.unsqueeze(1)?, 1)?
            .squeeze(1)?
            .neg()?; // [B*T]

        // Apply the mask (if given) then reduce.
        let (numer, denom) = match loss_mask {
            Some(mask) => {
                let mask_flat = mask.reshape((b * t,))?.to_dtype(nll.dtype())?;
                let masked = nll.broadcast_mul(&mask_flat)?;
                let denom = mask_flat.sum_all()?;
                (masked, denom)
            }
            None => {
                let denom = Tensor::new(&[(b * t) as f32], nll.device())?
                    .to_dtype(nll.dtype())?
                    .reshape(())?;
                (nll, denom)
            }
        };

        match self.reduction {
            Reduction::Mean => {
                let numer_sum = numer.sum_all()?;
                // Guard against a fully-masked batch producing NaN.
                let one = Tensor::new(&[1.0f32], numer_sum.device())?
                    .to_dtype(numer_sum.dtype())?
                    .reshape(())?;
                let safe_denom = denom.maximum(&one)?;
                numer_sum.broadcast_div(&safe_denom)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn tiny_logits(dev: &Device) -> Tensor {
        // [1, 2, 3] shape — batch=1, seq=2, vocab=3.
        Tensor::from_slice(
            &[
                0.1f32, 0.2, 0.7, // token 0
                0.9, 0.05, 0.05, // token 1
            ],
            (1, 2, 3),
            dev,
        )
        .unwrap()
    }

    #[test]
    fn cross_entropy_matches_manual_nll_on_tiny_input() {
        let device = Device::Cpu;
        let logits = tiny_logits(&device);
        let targets = Tensor::from_slice(&[2u32, 0], (1, 2), &device).unwrap();

        let loss = CrossEntropyLoss::new();
        let out = loss.compute(&logits, &targets, None).unwrap();
        let val: f32 = out.to_scalar().unwrap();

        // Manual NLL average across the two positions.
        let lp = candle_nn::ops::log_softmax(&logits.reshape((2, 3)).unwrap(), 1).unwrap();
        let lp_vec: Vec<f32> = lp.flatten_all().unwrap().to_vec1().unwrap();
        // targets = [2, 0] pick col 2 of row 0 and col 0 of row 1.
        // Row-major indices spelled out explicitly.
        let expected = -(lp_vec[2] + lp_vec[3]) / 2.0;
        assert!(
            (val - expected).abs() < 1e-5,
            "cross entropy mismatch: got {val}, expected {expected}"
        );
    }

    #[test]
    fn cross_entropy_mask_ignores_zero_positions() {
        let device = Device::Cpu;
        let logits = tiny_logits(&device);
        let targets = Tensor::from_slice(&[2u32, 0], (1, 2), &device).unwrap();
        // Zero out the second position — should equal the loss on the
        // first position only.
        let mask = Tensor::from_slice(&[1.0f32, 0.0], (1, 2), &device).unwrap();

        let loss = CrossEntropyLoss::new();
        let out = loss.compute(&logits, &targets, Some(&mask)).unwrap();
        let val: f32 = out.to_scalar().unwrap();

        let lp = candle_nn::ops::log_softmax(&logits.reshape((2, 3)).unwrap(), 1).unwrap();
        let lp_vec: Vec<f32> = lp.flatten_all().unwrap().to_vec1().unwrap();
        let expected = -lp_vec[2]; // only position 0 (index 2 in flat row) counts
        assert!(
            (val - expected).abs() < 1e-5,
            "masked cross entropy mismatch: got {val}, expected {expected}"
        );
    }

    #[test]
    fn cross_entropy_rejects_shape_mismatch() {
        let device = Device::Cpu;
        let logits = tiny_logits(&device);
        // Wrong seq length — should surface a Msg error, not panic.
        let targets = Tensor::from_slice(&[2u32], (1, 1), &device).unwrap();
        let loss = CrossEntropyLoss::new();
        let err = match loss.compute(&logits, &targets, None) {
            Ok(_) => panic!("expected shape mismatch error"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("does not match"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn cross_entropy_fully_masked_batch_is_finite() {
        let device = Device::Cpu;
        let logits = tiny_logits(&device);
        let targets = Tensor::from_slice(&[2u32, 0], (1, 2), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32, 0.0], (1, 2), &device).unwrap();
        let loss = CrossEntropyLoss::new();
        let out = loss
            .compute(&logits, &targets, Some(&mask))
            .expect("fully-masked batch must not error");
        let val: f32 = out.to_scalar().unwrap();
        // Numerator is zero and the denominator is clamped to 1, so the
        // result is exactly 0 rather than NaN.
        assert_eq!(val, 0.0);
    }

    // Silence the "unused" hint on `DType` for readers of this test
    // module; it is here to make the `to_dtype` implementations
    // discoverable when the module is opened alongside real code.
    #[allow(dead_code)]
    const _: DType = DType::F32;
}
