//! Penalties that read what a generation has already produced.
//!
//! Every sampler in [`super`] reads one logits row and nothing else, so
//! none of them can tell a token the model has emitted six times from
//! one it has never emitted. Left alone, a model that starts repeating
//! keeps repeating — the state that would break the loop is exactly the
//! history the sampler cannot see.
//!
//! Three penalties are in general use and they are not variants of one
//! another:
//!
//! - **Repetition** ([`Penalties::repetition`]) scales the logit of any
//!   token already seen, by a factor rather than a subtraction. From
//!   CTRL ([Keskar et al. 2019](https://arxiv.org/abs/1909.05858) §4.3,
//!   which reports `1.2` as a working value), and it divides positive
//!   logits while multiplying negative ones — the same rule on both
//!   sides would *raise* a negative logit and reward the repeat it was
//!   meant to discourage.
//! - **Frequency** ([`Penalties::frequency`]) subtracts in proportion to
//!   how many times the token appeared, so pressure accumulates with
//!   each repeat.
//! - **Presence** ([`Penalties::presence`]) subtracts a flat amount from
//!   every token that appeared at all, which pushes towards new
//!   vocabulary rather than away from repetition as such.
//!
//! The last two are OpenAI's, additive and independent of scale; the
//! first is multiplicative and interacts with temperature. They compose,
//! and the order is fixed here: repetition scales, then the two
//! subtractions land on the result.
//!
//! # Where the history comes from
//!
//! [`PenalizedSampler`] accumulates the tokens it returns, the same way
//! [`ConstrainedSampler`](super::ConstrainedSampler) accumulates its
//! prefix. A prompt is not part of that by default — see
//! [`PenalizedSampler::with_history`] — because whether the prompt
//! counts is a real choice and not one this crate should make silently:
//! penalising it discourages a summary from reusing the words it was
//! given, which is sometimes exactly wrong and sometimes exactly right.

use candle_core::{Result as CandleResult, Tensor};

use super::Sampler;

/// How hard an already-seen token is pushed down.
///
/// The defaults are all "off", so a `Penalties::default()` wrapper is
/// the inner sampler with an extra allocation.
#[derive(Debug, Clone, Copy)]
pub struct Penalties {
    /// CTRL-style multiplicative penalty on any token in the history.
    ///
    /// `1.0` (default) is off. Above `1.0` discourages repeats; below
    /// `1.0` encourages them, which is occasionally wanted and is
    /// therefore allowed rather than clamped. `0.0` and negatives are
    /// refused — zero erases the distinction between a seen token and
    /// an impossible one, and a negative flips the sign of every logit
    /// it touches.
    pub repetition: f32,
    /// Subtracted once per occurrence: `logit -= frequency · count`.
    /// `0.0` (default) is off.
    pub frequency: f32,
    /// Subtracted once if the token occurs at all:
    /// `logit -= presence`. `0.0` (default) is off.
    pub presence: f32,
    /// Count only the last `n` tokens, or `None` (default) for the
    /// whole history.
    ///
    /// A long generation otherwise accumulates a penalty against every
    /// word it has ever used, which past a few hundred tokens is a
    /// penalty against the language rather than against repetition.
    /// `Some(0)` disables the penalties by leaving nothing to count.
    pub window: Option<usize>,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            repetition: 1.0,
            frequency: 0.0,
            presence: 0.0,
            window: None,
        }
    }
}

impl Penalties {
    /// Whether these would change any logit.
    ///
    /// Used to skip the vocabulary walk entirely on a wrapper that was
    /// built with nothing set.
    pub fn is_noop(&self) -> bool {
        self.repetition == 1.0 && self.frequency == 0.0 && self.presence == 0.0
    }

    /// Refuse a setting that cannot mean what it says.
    fn validate(&self) -> CandleResult<()> {
        if !self.repetition.is_finite() || self.repetition <= 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "penalties: repetition must be a finite positive factor (got {}); \\
                 1.0 is off, and 0 or a negative would flip or erase the logits it touches",
                self.repetition
            )));
        }
        if !self.frequency.is_finite() || !self.presence.is_finite() {
            return Err(candle_core::Error::Msg(format!(
                "penalties: frequency and presence must be finite (got {} and {})",
                self.frequency, self.presence
            )));
        }
        Ok(())
    }
}

/// A sampler that pushes down what the generation has already produced.
///
/// Wraps any [`Sampler`] — including a
/// [`ConstrainedSampler`](super::ConstrainedSampler), which is the usual
/// arrangement: the constraint decides what is allowed and the penalty
/// decides how appealing the allowed repeats are.
///
/// Composition note: wrap the constrained sampler in this one, rather
/// than the other way round. This way the penalty applies to the logits
/// the constraint has already masked, and a token the constraint
/// forbids is never counted or penalised — while the reverse order
/// would spend the penalty on tokens the mask then removes.
#[derive(Debug)]
pub struct PenalizedSampler<S: Sampler> {
    inner: S,
    penalties: Penalties,
    history: Vec<u32>,
}

impl<S: Sampler> PenalizedSampler<S> {
    /// Wrap `inner`, starting from an empty history.
    ///
    /// # Errors
    ///
    /// A `repetition` of zero or below, or a non-finite setting. Checked
    /// here rather than at the first `sample` so a misconfigured
    /// generation fails before it has produced anything.
    pub fn new(inner: S, penalties: Penalties) -> CandleResult<Self> {
        penalties.validate()?;
        Ok(Self {
            inner,
            penalties,
            history: Vec::new(),
        })
    }

    /// Wrap `inner` with the prompt already counted.
    ///
    /// The choice this makes explicit: with the prompt in the history,
    /// the model is discouraged from reusing the words it was given —
    /// right for open continuation, wrong for a task whose answer is
    /// mostly made of the input's own vocabulary.
    pub fn with_history(inner: S, penalties: Penalties, history: &[u32]) -> CandleResult<Self> {
        penalties.validate()?;
        Ok(Self {
            inner,
            penalties,
            history: history.to_vec(),
        })
    }

    /// Tokens counted so far, oldest first.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// The wrapped sampler.
    ///
    /// For a caller that needs something the [`Sampler`] trait does not
    /// carry — a [`ConstrainedSampler`](super::ConstrainedSampler)'s
    /// `is_done`, most of all, which a generation loop polls and which
    /// would otherwise be unreachable through this wrapper.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// The wrapped sampler, mutably — for its `reset`.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Count a token this sampler did not produce.
    ///
    /// For a loop that decides some steps elsewhere — a forced prefix, a
    /// tool call spliced in — and still wants them to weigh on what
    /// follows.
    pub fn observe(&mut self, token: u32) {
        self.history.push(token);
    }

    /// Forget the history so the same sampler can drive another
    /// generation.
    ///
    /// The inner sampler's RNG is deliberately not reset, matching
    /// [`ConstrainedSampler::reset`](super::ConstrainedSampler::reset):
    /// two generations from one reused sampler stay distinct.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// The window of history the penalties read.
    fn counted(&self) -> &[u32] {
        match self.penalties.window {
            Some(n) if n < self.history.len() => &self.history[self.history.len() - n..],
            Some(_) | None => &self.history,
        }
    }
}

impl<S: Sampler> Sampler for PenalizedSampler<S> {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        let counted = self.counted();
        let token = if self.penalties.is_noop() || counted.is_empty() {
            // Nothing to apply, or nothing to apply it to. The caller's
            // tensor goes straight through — no copy, and bit-identical
            // to the unwrapped sampler.
            self.inner.sample(logits)?
        } else {
            let adjusted = apply_penalties(logits, counted, &self.penalties)?;
            self.inner.sample(&adjusted)?
        };
        self.history.push(token);
        Ok(token)
    }
}

/// Apply the penalties to a `[vocab]` logits row against `history`.
///
/// Public so a caller driving its own decode loop can reach the
/// transform without adopting the wrapper — the Lua bridge does exactly
/// that, since a Lua decode loop owns its own history.
///
/// Counts are taken over `history` as given: a caller wanting a window
/// passes the window.
pub fn apply_penalties(
    logits: &Tensor,
    history: &[u32],
    penalties: &Penalties,
) -> CandleResult<Tensor> {
    penalties.validate()?;
    let dims = logits.dims();
    if dims.len() != 1 {
        return Err(candle_core::Error::Msg(format!(
            "penalties: logits must be [vocab] (got {dims:?})"
        )));
    }
    let vocab = dims[0];
    let mut row = logits.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;

    // One pass over the history rather than one over the vocabulary:
    // a generation is shorter than a vocabulary by three or four orders
    // of magnitude, and only the tokens it holds can be penalised.
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for token in history {
        if (*token as usize) < vocab {
            *counts.entry(*token).or_insert(0) += 1;
        }
    }

    for (token, count) in counts {
        let i = token as usize;
        let mut value = row[i];
        if penalties.repetition != 1.0 {
            // CTRL's rule: divide a positive logit, multiply a negative
            // one. Dividing both would move a negative logit *up*.
            value = if value > 0.0 {
                value / penalties.repetition
            } else {
                value * penalties.repetition
            };
        }
        value -= penalties.frequency * count as f32;
        if penalties.presence != 0.0 {
            value -= penalties.presence;
        }
        row[i] = value;
    }

    Tensor::from_vec(row, vocab, logits.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::GreedySampler;
    use candle_core::{DType, Device};

    fn row(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu).unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32).unwrap().to_vec1().unwrap()
    }

    #[test]
    fn repetition_divides_a_positive_logit_and_multiplies_a_negative_one() {
        // The asymmetry is the whole point: one rule for both signs
        // would raise the negative logit and reward the repeat.
        let p = Penalties {
            repetition: 2.0,
            ..Penalties::default()
        };
        let out = values(&apply_penalties(&row(&[4.0, -4.0, 1.0]), &[0, 1], &p).unwrap());
        assert_eq!(out[0], 2.0, "positive logit halved");
        assert_eq!(out[1], -8.0, "negative logit doubled away from zero");
        assert_eq!(out[2], 1.0, "an unseen token is untouched");
    }

    #[test]
    fn frequency_accumulates_and_presence_does_not() {
        let freq = Penalties {
            frequency: 0.5,
            ..Penalties::default()
        };
        let out = values(&apply_penalties(&row(&[1.0, 1.0]), &[0, 0, 0], &freq).unwrap());
        assert_eq!(out[0], 1.0 - 1.5, "three occurrences, three subtractions");
        assert_eq!(out[1], 1.0);

        let pres = Penalties {
            presence: 0.5,
            ..Penalties::default()
        };
        let out = values(&apply_penalties(&row(&[1.0, 1.0]), &[0, 0, 0], &pres).unwrap());
        assert_eq!(out[0], 0.5, "three occurrences, one subtraction");
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn the_three_compose_in_a_fixed_order() {
        let p = Penalties {
            repetition: 2.0,
            frequency: 0.25,
            presence: 1.0,
            window: None,
        };
        let out = values(&apply_penalties(&row(&[4.0]), &[0, 0], &p).unwrap());
        // 4 / 2 = 2, then -0.25·2 = 1.5, then -1 = 0.5.
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn a_window_forgets_what_fell_out_of_it() {
        let inner = GreedySampler;
        let penalties = Penalties {
            presence: 100.0,
            window: Some(2),
            ..Penalties::default()
        };
        let mut sampler =
            PenalizedSampler::with_history(inner, penalties, &[0, 1, 2]).expect("valid");
        // Only ids 1 and 2 are inside the window, so id 0 is not pushed
        // down and wins despite being the older repeat.
        let token = sampler.sample(&row(&[1.0, 0.9, 0.9])).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn the_sampler_counts_what_it_returns() {
        let penalties = Penalties {
            presence: 100.0,
            ..Penalties::default()
        };
        let mut sampler = PenalizedSampler::new(GreedySampler, penalties).expect("valid");
        // First step: nothing seen, the argmax wins.
        assert_eq!(sampler.sample(&row(&[2.0, 1.0, 0.5])).unwrap(), 0);
        // Second step: id 0 now carries the presence penalty, so the
        // runner-up wins.
        assert_eq!(sampler.sample(&row(&[2.0, 1.0, 0.5])).unwrap(), 1);
        assert_eq!(sampler.history(), &[0, 1]);
        sampler.reset();
        assert_eq!(sampler.sample(&row(&[2.0, 1.0, 0.5])).unwrap(), 0);
    }

    #[test]
    fn an_observed_token_weighs_the_same_as_a_sampled_one() {
        let penalties = Penalties {
            presence: 100.0,
            ..Penalties::default()
        };
        let mut sampler = PenalizedSampler::new(GreedySampler, penalties).expect("valid");
        sampler.observe(0);
        assert_eq!(sampler.sample(&row(&[2.0, 1.0])).unwrap(), 1);
    }

    #[test]
    fn a_wrapper_with_nothing_set_returns_what_the_inner_sampler_would() {
        let mut plain = GreedySampler;
        let mut wrapped = PenalizedSampler::new(GreedySampler, Penalties::default()).unwrap();
        let logits = row(&[0.1, 5.0, -2.0]);
        for _ in 0..3 {
            assert_eq!(
                wrapped.sample(&logits).unwrap(),
                plain.sample(&logits).unwrap()
            );
        }
    }

    #[test]
    fn a_repetition_factor_that_cannot_mean_what_it_says_is_refused() {
        for bad in [0.0, -1.0, f32::NAN] {
            let p = Penalties {
                repetition: bad,
                ..Penalties::default()
            };
            assert!(
                PenalizedSampler::new(GreedySampler, p).is_err(),
                "repetition = {bad} must be refused"
            );
        }
    }

    #[test]
    fn a_history_token_outside_the_vocabulary_is_ignored_rather_than_panicking() {
        let p = Penalties {
            presence: 1.0,
            ..Penalties::default()
        };
        let out = values(&apply_penalties(&row(&[1.0, 1.0]), &[5], &p).unwrap());
        assert_eq!(out, vec![1.0, 1.0]);
    }
}
