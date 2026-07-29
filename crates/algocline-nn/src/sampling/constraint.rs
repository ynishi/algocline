//! Layer 2 of the sampler plan: constraints that mask logits before a
//! Layer 1 [`Sampler`] picks a token.
//!
//! A [`Constraint`] answers two questions per generation step, both from
//! the generated-token prefix alone: *which tokens may come next*
//! ([`Constraint::mask`]) and *should generation stop*
//! ([`Constraint::is_terminal`]). [`ConstrainedSampler`] wires a
//! constraint to an arbitrary inner sampler and is itself a [`Sampler`],
//! so constrained decoding composes with every Layer 1 impl and with
//! future Layer 3 schedules without either side knowing about the other.
//!
//! # Sparse masks
//!
//! [`TokenMask`] is deliberately sparse. A dense `Vec<bool>` of vocab
//! length (32k–256k entries) would be rebuilt on every single token even
//! when the constraint has nothing to say. [`TokenMask::AllowAll`] is the
//! common case for prefix-agnostic constraints (stop tokens, most of a
//! grammar's interior) and costs *nothing*: the logits tensor is handed
//! to the inner sampler untouched, with no device round-trip.
//!
//! # Failure is loud
//!
//! A mask that leaves zero candidate tokens is a caller programming
//! error, not a situation to paper over. Softmax over an all-`-inf` row
//! yields NaNs, and silently falling back to argmax on the *unmasked*
//! logits would emit a token the constraint explicitly forbade — the one
//! outcome constrained decoding exists to prevent. Both cases return
//! `Err` instead.

use candle_core::{Result as CandleResult, Tensor};

use super::{validate_logits, Sampler};

/// Sparse per-step token mask produced by a [`Constraint`].
///
/// Variants are mutually exclusive views of the same decision:
///
/// - [`TokenMask::AllowAll`] — no restriction. The logits tensor is
///   passed through untouched (no allocation, no device round-trip).
/// - [`TokenMask::Deny`] — the listed token ids are masked to `-inf`,
///   everything else survives. Use when the forbidden set is small.
/// - [`TokenMask::Allow`] — *only* the listed token ids survive,
///   everything else is masked to `-inf`. Use when the permitted set is
///   small (a grammar mid-production, a JSON key alternation).
///
/// Ids are token indices into the vocab axis of the logits row. Ids at
/// or beyond `vocab` are rejected at mask-application time — a
/// constraint that emits them has a bug the sampler must not absorb.
/// Duplicate ids are harmless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenMask {
    /// Every token is permitted; the logits are not modified at all.
    AllowAll,
    /// The listed token ids are masked to `-inf`.
    Deny(Vec<u32>),
    /// Only the listed token ids survive; all others are masked to
    /// `-inf`. An empty list is rejected (no token could be sampled).
    Allow(Vec<u32>),
}

/// Per-step restriction on the next token.
///
/// Both methods receive the **generated-token prefix only** — the prompt
/// is not included. A constraint that needs prompt context captures it
/// at construction time. The prefix is exactly the sequence of tokens
/// this [`ConstrainedSampler`] has produced since construction or the
/// last [`ConstrainedSampler::reset`].
///
/// Implementations are expected to be pure with respect to the prefix
/// (`&self`, no interior mutation): the same prefix must yield the same
/// mask, which is what makes seeded generation reproducible end to end.
pub trait Constraint {
    /// Tokens permitted at the position immediately after `prefix`.
    ///
    /// Return [`TokenMask::AllowAll`] when the constraint has nothing to
    /// restrict at this position; that path is free.
    fn mask(&self, prefix: &[u32]) -> TokenMask;

    /// Whether generation should stop, given everything produced so far.
    ///
    /// The sampler itself never stops on this signal — it has no loop to
    /// break. The generation loop owns termination and polls
    /// [`ConstrainedSampler::is_done`].
    fn is_terminal(&self, prefix: &[u32]) -> bool;
}

/// A [`Sampler`] that masks logits through a [`Constraint`] before
/// delegating the actual draw to an inner sampler.
///
/// The wrapper owns the generated-token prefix, which is the one piece
/// of state Layer 1 samplers deliberately do not carry. Composition is
/// arbitrary: the inner sampler may itself be a `ConstrainedSampler`,
/// stacking constraints (the innermost mask is applied last).
///
/// # Invariants
///
/// - The returned token id is always permitted by the constraint's mask
///   for the current prefix; a mask that permits nothing errors rather
///   than falling back.
/// - `prefix` grows by exactly one token per successful
///   [`Sampler::sample`] call and is untouched on error, so a caller can
///   retry a failed step without corrupting the history.
#[derive(Debug, Clone)]
pub struct ConstrainedSampler<S: Sampler, C: Constraint> {
    inner: S,
    constraint: C,
    prefix: Vec<u32>,
}

impl<S: Sampler, C: Constraint> ConstrainedSampler<S, C> {
    /// Wrap `inner` with `constraint`, starting from an empty prefix.
    pub fn new(inner: S, constraint: C) -> Self {
        Self {
            inner,
            constraint,
            prefix: Vec::new(),
        }
    }

    /// Whether the constraint considers the current prefix terminal.
    ///
    /// Polled by the generation loop; the sampler never acts on it.
    pub fn is_done(&self) -> bool {
        self.constraint.is_terminal(&self.prefix)
    }

    /// Tokens generated so far, oldest first.
    pub fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    /// Drop the generated prefix so the same sampler can drive another
    /// generation.
    ///
    /// The inner sampler's RNG is intentionally *not* reset: two
    /// generations from one reused sampler stay distinct, while a caller
    /// wanting bit-identical repeats constructs a fresh sampler from the
    /// same seed.
    pub fn reset(&mut self) {
        self.prefix.clear();
    }
}

impl<S: Sampler, C: Constraint> Sampler for ConstrainedSampler<S, C> {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        let mask = self.constraint.mask(&self.prefix);
        let token = match mask {
            // Free path: hand the caller's tensor straight through.
            TokenMask::AllowAll => self.inner.sample(logits)?,
            _ => {
                let masked = apply_mask(logits, &mask)?;
                self.inner.sample(&masked)?
            }
        };
        self.prefix.push(token);
        Ok(token)
    }
}

/// Terminate generation when one of a fixed set of token ids is emitted.
///
/// The stop tokens are **not** masked out: `mask` always returns
/// [`TokenMask::AllowAll`], matching the semantics every mainstream
/// runtime uses — the stop token is a legitimate sample, and it is the
/// generation *loop* that halts once it appears. Masking them instead
/// would make the model unable to ever finish.
///
/// Detection is on the last token only. A stop token deeper in the
/// prefix means the loop ignored an earlier stop signal, which is the
/// caller's decision to make.
#[derive(Debug, Clone, Default)]
pub struct StopTokensConstraint {
    stop_tokens: Vec<u32>,
}

impl StopTokensConstraint {
    /// Build a constraint terminating on any of `stop_tokens`.
    ///
    /// An empty list yields a constraint that never terminates, which is
    /// a valid (if unusual) request — the loop is then bounded by a
    /// max-token budget instead.
    pub fn new(stop_tokens: Vec<u32>) -> Self {
        Self { stop_tokens }
    }

    /// The configured stop token ids.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }
}

impl Constraint for StopTokensConstraint {
    fn mask(&self, _prefix: &[u32]) -> TokenMask {
        TokenMask::AllowAll
    }

    fn is_terminal(&self, prefix: &[u32]) -> bool {
        match prefix.last() {
            Some(last) => self.stop_tokens.contains(last),
            None => false,
        }
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

/// Build a new logits row with the masked-out entries set to `-inf`.
///
/// Round-trips through host memory (`to_vec1` → mutate → `from_vec`)
/// rather than composing tensor ops: the mask is sparse and the vocab
/// row is a single vector, so the scatter is cheaper than materialising
/// a full-width mask tensor. The result lands on the input's device.
///
/// Errors on an out-of-range token id and on a mask that leaves no
/// candidate at all; see the module doc for why neither is recoverable.
fn apply_mask(logits: &Tensor, mask: &TokenMask) -> CandleResult<Tensor> {
    validate_logits(logits)?;
    let mut values = logits.to_vec1::<f32>()?;
    let vocab = values.len();

    // `keep` starts at the variant's default answer and the id list
    // flips the exceptions, so both variants share one scatter loop.
    let (ids, kept_default) = match mask {
        TokenMask::AllowAll => return Ok(logits.clone()),
        TokenMask::Deny(ids) => (ids, true),
        TokenMask::Allow(ids) => {
            if ids.is_empty() {
                return Err(candle_core::Error::Msg(
                    "ConstrainedSampler: TokenMask::Allow with an empty token list leaves no candidate tokens".into(),
                ));
            }
            (ids, false)
        }
    };

    let mut keep = vec![kept_default; vocab];
    for &id in ids {
        let idx = id as usize;
        if idx >= vocab {
            return Err(candle_core::Error::Msg(format!(
                "ConstrainedSampler: mask token id {id} is out of range for vocab {vocab}"
            )));
        }
        keep[idx] = !kept_default;
    }

    if keep.iter().all(|k| !*k) {
        return Err(candle_core::Error::Msg(format!(
            "ConstrainedSampler: mask leaves no candidate tokens (vocab {vocab} fully masked)"
        )));
    }

    for (value, keep) in values.iter_mut().zip(keep) {
        if !keep {
            *value = f32::NEG_INFINITY;
        }
    }

    Tensor::from_vec(values, vocab, logits.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{GreedySampler, TopKTopPSampler};
    use candle_core::Device;

    fn cpu_logits(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(),), &Device::Cpu).unwrap()
    }

    /// Logits used across the mask tests: argmax is index 1.
    fn fixture() -> Tensor {
        cpu_logits(&[0.1, 3.2, 0.5, 2.7, 1.0])
    }

    /// Constraint returning one fixed mask regardless of prefix, so the
    /// mask-application path can be tested independently of any real
    /// grammar.
    struct FixedMask(TokenMask);

    impl Constraint for FixedMask {
        fn mask(&self, _prefix: &[u32]) -> TokenMask {
            self.0.clone()
        }

        fn is_terminal(&self, _prefix: &[u32]) -> bool {
            false
        }
    }

    /// A stop token is sampled like any other token; what changes is
    /// that the constraint flips to terminal once it lands in the
    /// prefix. Guards the "loop stops, token is not masked" semantics.
    #[test]
    fn stop_token_flips_is_done() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![1]));
        assert!(!s.is_done(), "empty prefix must not be terminal");

        let token = s.sample(&fixture()).unwrap();
        assert_eq!(token, 1, "stop token must remain sampleable");
        assert!(s.is_done(), "stop token in prefix must be terminal");
    }

    /// A non-stop token leaves the constraint non-terminal. Without this
    /// the previous test would pass on an `is_terminal` that always
    /// returns true after the first token.
    #[test]
    fn non_stop_token_does_not_terminate() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![4]));
        let token = s.sample(&fixture()).unwrap();
        assert_eq!(token, 1);
        assert!(!s.is_done(), "non-stop token must not terminate");
    }

    /// `Deny` on the argmax hands the runner-up to the inner sampler.
    /// The greedy inner sampler makes the effect unambiguous.
    #[test]
    fn deny_excludes_the_argmax() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Deny(vec![1])));
        assert_eq!(s.sample(&fixture()).unwrap(), 3, "runner-up expected");
        assert_eq!(s.prefix(), &[3], "prefix must record the sampled token");
    }

    /// `Allow` of a single non-argmax token forces that token even
    /// though its logit is the lowest in the row.
    #[test]
    fn allow_forces_the_listed_token() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Allow(vec![0])));
        assert_eq!(s.sample(&fixture()).unwrap(), 0);
    }

    /// An empty `Allow` permits nothing. Erroring is the point: argmax
    /// on the unmasked logits would return a token the constraint
    /// forbade, and softmax over an all-`-inf` row is NaN.
    #[test]
    fn allow_empty_errors() {
        let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(TokenMask::Allow(vec![])));
        assert!(s.sample(&fixture()).is_err(), "empty Allow must error");
        assert!(s.prefix().is_empty(), "failed step must not grow prefix");
    }

    /// `Deny` covering the whole vocab is the same degenerate state as
    /// an empty `Allow` and must fail the same way.
    #[test]
    fn deny_covering_full_vocab_errors() {
        let mut s = ConstrainedSampler::new(
            GreedySampler,
            FixedMask(TokenMask::Deny(vec![0, 1, 2, 3, 4])),
        );
        assert!(s.sample(&fixture()).is_err(), "full Deny must error");
    }

    /// A token id at or beyond `vocab` is a constraint bug. Absorbing it
    /// (clamping, skipping) would hide the bug behind plausible output.
    #[test]
    fn out_of_range_token_id_errors() {
        for mask in [TokenMask::Deny(vec![5]), TokenMask::Allow(vec![0, 9])] {
            let mut s = ConstrainedSampler::new(GreedySampler, FixedMask(mask.clone()));
            assert!(s.sample(&fixture()).is_err(), "{mask:?} must error");
        }
    }

    /// Masking must not disturb the inner sampler's reproducibility: two
    /// constrained samplers built from the same seed and fed the same
    /// logits stream produce the same tokens, and the mask actually
    /// binds (the denied ids never appear).
    #[test]
    fn composition_with_top_k_top_p_stays_reproducible() {
        let logits = cpu_logits(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5]);
        let denied = vec![2, 7];

        let build = || {
            ConstrainedSampler::new(
                TopKTopPSampler::new(Some(4), Some(0.95), 1.0, 24601),
                FixedMask(TokenMask::Deny(denied.clone())),
            )
        };
        let mut a = build();
        let mut b = build();

        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b, "constrained sampler diverged on shared seed");
        assert!(
            seq_a.iter().all(|t| !denied.contains(t)),
            "denied tokens leaked into the stream: {seq_a:?}"
        );
    }

    /// `reset` returns the sampler to the pre-generation state so one
    /// sampler can drive several generations.
    #[test]
    fn reset_clears_the_prefix() {
        let mut s = ConstrainedSampler::new(GreedySampler, StopTokensConstraint::new(vec![1]));
        s.sample(&fixture()).unwrap();
        assert!(s.is_done());

        s.reset();
        assert!(s.prefix().is_empty(), "reset must clear the prefix");
        assert!(!s.is_done(), "reset must clear the terminal state");
    }

    /// `StopTokensConstraint` never masks — it only terminates. Asserted
    /// on the variant itself so a future refactor cannot quietly start
    /// denying the stop tokens (which would make generation unable to
    /// finish).
    #[test]
    fn stop_tokens_constraint_never_masks() {
        let c = StopTokensConstraint::new(vec![1, 2]);
        for prefix in [vec![], vec![9], vec![1], vec![2, 1]] {
            assert_eq!(
                c.mask(&prefix),
                TokenMask::AllowAll,
                "prefix {prefix:?} must stay unmasked"
            );
        }
    }
}
