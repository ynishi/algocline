//! Next-token samplers for the inference path.
//!
//! Consumes a `[vocab]`-shaped logits row (one position, one batch item)
//! and returns the sampled token id. Callers responsible for the outer
//! loop — the sampler holds no cache, no state beyond its own RNG, and
//! no dependency on the specific adapter that produced the logits.
//!
//! # The Sampler 3-layer plan
//!
//! - **Layer 1 (this module)** — [`Sampler`] trait + Rust default
//!   implementations ([`GreedySampler`] / [`TemperatureSampler`] /
//!   [`TopKTopPSampler`]). Every consumer starts here.
//! - **Layer 2 ([`constraint`] / [`json_schema`])** — filters that mask
//!   logits before a Layer 1 sampler picks, wired in through
//!   [`ConstrainedSampler`], itself a `Sampler`. Three [`Constraint`]s
//!   have landed: [`StopTokensConstraint`] (termination only),
//!   [`RegexConstraint`] (anchored full-match over a tokenizer's surface
//!   strings) and [`JsonSchemaConstraint`] (a JSON schema compiled to a
//!   regex and enforced through the previous one). GBNF grammars are a
//!   future addition behind the same trait — and the one that unlocks
//!   recursive schemas, which no regular language can express.
//! - **Layer 3 (engine side: `alc.nn.sampler` / `alc.nn.constraint`)** —
//!   Lua factories that build the types above, compose them, and let a
//!   Lua function *be* a `Sampler`. No scheduling primitive lives here:
//!   the generation loop is already Lua-side, so swapping the active
//!   sampler per position is plain Lua control flow over two handles.
//!   The only thing this layer needed from Layer 1 was the ability to
//!   erase the sampler's concrete type — see the
//!   `impl Sampler for Box<dyn Sampler + Send>` below.
//!
//! Layer 2 / 3 attach as additional `impl Sampler` types (including the
//! Lua-callback bridge on the engine side) without changing the trait.
//! That is the entire point of Layer 1.
//!
//! # Determinism
//!
//! Every stochastic sampler carries its own [`StdRng`]. A caller that
//! needs save/load reproducibility supplies a fixed seed at
//! construction; the RNG state advances by exactly one draw per
//! [`Sampler::sample`] call, so two runs with the same seed and same
//! logits stream reproduce the same tokens.
//!
//! [`Sampler::sample`] takes `&mut self` to allow the RNG state to
//! advance; a caller sharing a sampler across generation loops (unlikely
//! given the state semantics) is expected to serialise access
//! themselves.

pub mod constraint;
pub mod json_schema;

use candle_core::{DType, Result as CandleResult, Tensor};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;

pub use constraint::{
    ConstrainedSampler, Constraint, RegexConstraint, StopTokensConstraint, TokenMask,
};
pub use json_schema::JsonSchemaConstraint;

/// Next-token sampler.
///
/// See the module-level doc for the 3-layer plan this trait is the base
/// of. Contract: given a `[vocab]`-shaped `f32` logits row, return the
/// sampled token id.
///
/// # Invariants (per impl)
///
/// - `logits.dims()` MUST be `[vocab]` and `logits.dtype() == DType::F32`;
///   an ill-shaped tensor is a caller programming error, not something
///   the sampler tries to reinterpret. `[batch, vocab]` inputs must be
///   split by the caller (one `sample` call per batch row).
/// - The returned `u32` MUST be a valid vocab index (`0..vocab`). Every
///   impl here upholds this by construction; a Layer-2 constraint that
///   masks *every* logit to `-inf` would produce NaNs on softmax, so
///   [`ConstrainedSampler`] rejects such a mask before it reaches an
///   inner sampler.
pub trait Sampler {
    /// Sample a single token id from `logits`.
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32>;
}

/// A boxed, type-erased sampler is still a [`Sampler`].
///
/// Layer 3 picks the concrete sampler at runtime (a Lua caller chooses
/// `greedy` / `temperature` / a callback of their own), while
/// [`ConstrainedSampler`] is generic over `S: Sampler` rather than taking
/// a trait object. Without this impl the engine could not name the type
/// `ConstrainedSampler<Box<dyn Sampler + Send>, _>` and would need a
/// parallel erasure enum with one arm per concrete sampler — a list that
/// would have to grow with every impl added here, which is precisely what
/// the trait exists to avoid.
///
/// `Send` sits in the bound rather than in a separate `Box<dyn Sampler>`
/// impl because the only consumer is mlua's `send` feature, which
/// requires the `UserData` holding a sampler to be `Send`. Adding the
/// unbounded impl too would make `Box<dyn Sampler + Send>` ambiguous at
/// no benefit.
impl Sampler for Box<dyn Sampler + Send> {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        (**self).sample(logits)
    }
}

/// Argmax: pick the highest-scoring token.
///
/// Deterministic, no RNG. `temperature == 0` on the two stochastic
/// samplers falls through to this behaviour for numerical stability
/// (softmax with `temperature → 0` collapses to a one-hot at the argmax
/// but the division underflows on the way there).
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        argmax_u32(logits)
    }
}

/// Temperature-scaled multinomial sampler.
///
/// Divides logits by `temperature`, softmaxes, and draws one token from
/// the resulting categorical distribution. `temperature <= 0` degrades
/// to [`GreedySampler`] rather than dividing by zero.
///
/// `temperature > 1` flattens the distribution (more randomness),
/// `temperature < 1` sharpens it (more deterministic). `temperature = 1`
/// samples directly from the model's own distribution.
#[derive(Debug)]
pub struct TemperatureSampler {
    /// Softmax temperature. `> 1.0` flattens the distribution
    /// (more randomness), `< 1.0` sharpens it, `<= 0.0` falls through
    /// to [`GreedySampler`] to avoid division by zero.
    pub temperature: f32,
    /// Seeded RNG. Two `TemperatureSampler`s built from the same seed
    /// and fed the same logits stream produce identical token streams.
    pub rng: StdRng,
}

impl TemperatureSampler {
    /// Build a sampler with a fixed seed for reproducibility.
    pub fn new(temperature: f32, seed: u64) -> Self {
        Self {
            temperature,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for TemperatureSampler {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        if self.temperature <= 0.0 {
            return GreedySampler.sample(logits);
        }
        let probs = softmax_scaled(logits, self.temperature)?;
        multinomial_sample(&probs, &mut self.rng)
    }
}

/// Nucleus (top-p) + top-k truncation, then temperature-scaled
/// multinomial sampling over the surviving tokens.
///
/// Application order matches the de-facto convention: temperature scale
/// → softmax → top-k truncation → top-p (cumulative-mass) truncation →
/// renormalise → multinomial draw. `top_k == None` and `top_p == None`
/// both disable their respective truncation; the sampler then reduces
/// to plain temperature sampling.
///
/// `top_k = Some(1)` and `top_p = Some(0.0)` both degrade to argmax
/// (the truncation retains a single token), matching the behaviour of
/// [`GreedySampler`] up to a single draw from a degenerate distribution.
#[derive(Debug)]
pub struct TopKTopPSampler {
    /// Retain only the `k` highest-probability tokens before drawing.
    /// `None` disables the truncation. `Some(0)` is a caller error
    /// (no tokens survive) and is rejected at [`Sampler::sample`] time.
    pub top_k: Option<usize>,
    /// Retain the smallest set of tokens whose cumulative probability
    /// covers `p`. `None` disables the truncation. Must be in
    /// `[0.0, 1.0]`; out-of-range values are rejected at
    /// [`Sampler::sample`] time.
    pub top_p: Option<f32>,
    /// Softmax temperature applied before truncation. Same degenerate
    /// path as [`TemperatureSampler::temperature`]: `<= 0.0` falls
    /// through to [`GreedySampler`].
    pub temperature: f32,
    /// Seeded RNG. Same reproducibility guarantee as
    /// [`TemperatureSampler::rng`].
    pub rng: StdRng,
}

impl TopKTopPSampler {
    /// Build a sampler with a fixed seed for reproducibility.
    pub fn new(top_k: Option<usize>, top_p: Option<f32>, temperature: f32, seed: u64) -> Self {
        Self {
            top_k,
            top_p,
            temperature,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for TopKTopPSampler {
    fn sample(&mut self, logits: &Tensor) -> CandleResult<u32> {
        if self.temperature <= 0.0 {
            return GreedySampler.sample(logits);
        }
        let temp = self.temperature.max(f32::MIN_POSITIVE);
        let probs = softmax_scaled(logits, temp)?;
        let mut probs_vec = probs.to_vec1::<f32>()?;

        // top-k: keep the k largest entries, zero the rest. `sort_by`
        // over indexed clones keeps the vocab-index mapping intact.
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(candle_core::Error::Msg(
                    "TopKTopPSampler: top_k = Some(0) leaves no valid tokens".into(),
                ));
            }
            if k < probs_vec.len() {
                let mut indexed: Vec<(usize, f32)> =
                    probs_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(idx, _) in &indexed[k..] {
                    probs_vec[idx] = 0.0;
                }
            }
        }

        // top-p: keep the smallest set of entries whose cumulative mass
        // covers `p`. Zero everything below. `p >= 1.0` keeps all,
        // `p == 0.0` keeps only the max (argmax).
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(candle_core::Error::Msg(format!(
                    "TopKTopPSampler: top_p = {p} out of [0.0, 1.0]"
                )));
            }
            let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut cutoff_idx = indexed.len();
            for (i, &(_, v)) in indexed.iter().enumerate() {
                cum += v;
                if cum >= p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            for &(idx, _) in &indexed[cutoff_idx..] {
                probs_vec[idx] = 0.0;
            }
        }

        // Renormalise. If every entry survived at zero (should be
        // impossible after the guards above), fall back to argmax on
        // the original logits so we still return a valid token.
        let sum: f32 = probs_vec.iter().sum();
        if sum <= 0.0 {
            return argmax_u32(logits);
        }
        for v in probs_vec.iter_mut() {
            *v /= sum;
        }

        multinomial_sample_vec(&probs_vec, &mut self.rng)
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

/// `argmax` returning a `u32` token id.
///
/// Extracted so the two stochastic samplers can delegate on the
/// `temperature <= 0` degenerate path without depending on each other.
fn argmax_u32(logits: &Tensor) -> CandleResult<u32> {
    validate_logits(logits)?;
    let idx = logits.argmax(0)?;
    idx.to_scalar::<u32>()
}

/// Divide logits by `temperature` and softmax over the vocab axis. Runs
/// in `f32` regardless of input dtype so downstream `to_vec1::<f32>()`
/// on the resulting probabilities always succeeds. Callers upstream
/// ensure `temperature > 0`.
fn softmax_scaled(logits: &Tensor, temperature: f32) -> CandleResult<Tensor> {
    validate_logits(logits)?;
    let scaled = (logits / (temperature as f64))?;
    candle_nn::ops::softmax(&scaled, 0)
}

/// Draw one token id from a `[vocab]`-shaped probability tensor.
fn multinomial_sample(probs: &Tensor, rng: &mut StdRng) -> CandleResult<u32> {
    let v = probs.to_vec1::<f32>()?;
    multinomial_sample_vec(&v, rng)
}

/// Draw one token id from a plain `[vocab]` probability vector.
///
/// Uses `rand::distr::weighted::WeightedIndex` — the standard "alias
/// method" implementation `rand` ships — for the categorical draw. A
/// zero-sum input degrades to argmax on the vector (returning the
/// largest weight's index) as a defensive path; every caller here
/// guarantees a positive sum, so this branch only fires on a caller
/// programming error.
fn multinomial_sample_vec(probs: &[f32], rng: &mut StdRng) -> CandleResult<u32> {
    let dist = WeightedIndex::new(probs).map_err(|e| {
        candle_core::Error::Msg(format!(
            "multinomial_sample: WeightedIndex construction failed: {e}"
        ))
    })?;
    let idx = dist.sample(rng);
    u32::try_from(idx).map_err(|_| {
        candle_core::Error::Msg(format!(
            "multinomial_sample: sampled index {idx} does not fit in u32"
        ))
    })
}

fn validate_logits(logits: &Tensor) -> CandleResult<()> {
    if logits.dims().len() != 1 {
        return Err(candle_core::Error::Msg(format!(
            "Sampler expects logits shape [vocab], got {:?}",
            logits.dims()
        )));
    }
    if logits.dtype() != DType::F32 {
        return Err(candle_core::Error::Msg(format!(
            "Sampler expects logits dtype f32, got {:?}",
            logits.dtype()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu_logits(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(),), &Device::Cpu).unwrap()
    }

    /// `GreedySampler` picks the highest-scoring token. Baseline.
    #[test]
    fn greedy_picks_the_argmax_token() {
        let mut s = GreedySampler;
        let logits = cpu_logits(&[0.1, 3.2, 0.5, 2.7, 1.0]);
        assert_eq!(s.sample(&logits).unwrap(), 1);
    }

    /// `TemperatureSampler` at `temperature == 0` MUST fall through to
    /// argmax rather than divide by zero and produce NaNs. Same
    /// invariant for `temperature < 0` (caller programming error, but
    /// the sampler picks a sane default rather than error out at
    /// runtime).
    #[test]
    fn temperature_zero_degrades_to_greedy() {
        let logits = cpu_logits(&[0.1, 3.2, 0.5, 2.7, 1.0]);
        for t in [0.0, -1.0] {
            let mut s = TemperatureSampler::new(t, 42);
            assert_eq!(s.sample(&logits).unwrap(), 1, "temperature {t} not greedy");
        }
    }

    /// `TopKTopPSampler` with `top_k = Some(1)` collapses to the argmax
    /// (only one token survives truncation, so the multinomial draw is
    /// degenerate). Same for `top_p = Some(0.0)`.
    #[test]
    fn top_k_one_and_top_p_zero_both_collapse_to_greedy() {
        let logits = cpu_logits(&[0.1, 3.2, 0.5, 2.7, 1.0]);
        for opts in [
            (Some(1), None, 1.0),
            (None, Some(0.0), 1.0),
            (Some(1), Some(0.0), 1.0),
        ] {
            let mut s = TopKTopPSampler::new(opts.0, opts.1, opts.2, 7);
            assert_eq!(
                s.sample(&logits).unwrap(),
                1,
                "opts {opts:?} did not collapse to greedy"
            );
        }
    }

    /// Same seed + same logits stream = same tokens. This is the save /
    /// load reproducibility invariant the Sampler trait docs promise.
    #[test]
    fn stochastic_samplers_are_reproducible_by_seed() {
        // A moderately soft distribution so we actually exercise the
        // stochastic path (temperature 1.0 means no scaling).
        let logits = cpu_logits(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5]);

        let mut a = TemperatureSampler::new(1.0, 12345);
        let mut b = TemperatureSampler::new(1.0, 12345);
        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b, "temperature sampler diverged on shared seed");

        let mut c = TopKTopPSampler::new(Some(3), Some(0.9), 1.0, 6789);
        let mut d = TopKTopPSampler::new(Some(3), Some(0.9), 1.0, 6789);
        let seq_c: Vec<u32> = (0..8).map(|_| c.sample(&logits).unwrap()).collect();
        let seq_d: Vec<u32> = (0..8).map(|_| d.sample(&logits).unwrap()).collect();
        assert_eq!(seq_c, seq_d, "top-k/top-p sampler diverged on shared seed");
    }

    /// Different seeds MUST produce different token streams on a
    /// non-degenerate distribution — the "stochastic" in "stochastic
    /// sampler" would be a lie otherwise. The distribution is soft
    /// enough that at least one of eight draws will disagree with
    /// astronomically high probability; a passing run gates against a
    /// silently seed-ignoring implementation.
    #[test]
    fn different_seeds_produce_different_streams() {
        let logits = cpu_logits(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.5, 2.5]);
        let mut a = TemperatureSampler::new(1.0, 1);
        let mut b = TemperatureSampler::new(1.0, 2);
        let seq_a: Vec<u32> = (0..8).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.sample(&logits).unwrap()).collect();
        assert_ne!(
            seq_a, seq_b,
            "temperature sampler ignored the seed (streams identical across seeds)"
        );
    }

    /// Wrong logits shape / dtype is a caller programming error and
    /// MUST error rather than silently reinterpret. Guards the
    /// invariants stated on the trait.
    #[test]
    fn wrong_shape_or_dtype_errors() {
        let two_d = Tensor::from_slice(&[0.1f32, 0.9], (1, 2), &Device::Cpu).unwrap();
        assert!(GreedySampler.sample(&two_d).is_err(), "2-D input must err");

        let f64_logits = Tensor::from_slice(&[0.1f64, 3.2, 0.5], (3,), &Device::Cpu).unwrap();
        assert!(
            GreedySampler.sample(&f64_logits).is_err(),
            "f64 input must err"
        );
    }

    /// `top_k = Some(0)` leaves no candidates — an unambiguous caller
    /// programming error. Refuse rather than silently returning
    /// something (argmax on the untouched logits would hide the bug).
    #[test]
    fn top_k_zero_errors() {
        let logits = cpu_logits(&[0.1, 3.2, 0.5]);
        let mut s = TopKTopPSampler::new(Some(0), None, 1.0, 42);
        assert!(s.sample(&logits).is_err(), "top_k=0 must error");
    }

    /// `top_p` outside `[0.0, 1.0]` is a caller programming error.
    #[test]
    fn top_p_out_of_range_errors() {
        let logits = cpu_logits(&[0.1, 3.2, 0.5]);
        for bad_p in [-0.1, 1.1, 2.0] {
            let mut s = TopKTopPSampler::new(None, Some(bad_p), 1.0, 42);
            assert!(s.sample(&logits).is_err(), "top_p={bad_p} must error");
        }
    }
}
