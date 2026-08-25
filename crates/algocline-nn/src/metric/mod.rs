//! Distribution distance / entropy primitives, and interval estimates
//! for statistics built out of them.
//!
//! Domain-agnostic scalar metrics computed from probability vectors. All
//! four primitives ([`kl`], [`js`], [`tvd`], [`entropy`]) operate on
//! `&[f32]` slices interpreted as discrete probability distributions.
//!
//! [`bootstrap`] sits one level up: given observations that arrive in
//! correlated groups, it puts a confidence interval around a statistic
//! computed over them (which may well be one of the primitives here,
//! meaned over a sample).
//!
//! # Layer boundary
//!
//! These primitives live in the Rust core so they are cheap, allocation-
//! free callables from either Rust or the Lua bridge (added in a later
//! subtask). Domain-specific composition — collecting an action
//! distribution from a Card, comparing two Cards over a prompt set — is
//! the responsibility of the Lua side, which chains these primitives
//! together.
//!
//! # Input contract
//!
//! Callers are expected to pass **already-normalized** probability
//! vectors (sum ≈ 1.0 within [`NORM_TOL`]). Every primitive re-checks
//! this invariant and refuses loudly via [`MetricError::NotNormalized`]
//! rather than silently returning a meaningless number. Empty inputs,
//! non-finite values, and negative entries are similarly rejected.
//!
//! # Failure is loud
//!
//! A distribution that sums to `0.7` is almost certainly a caller bug
//! (a missing softmax, a forgotten renormalization after masking).
//! Silently normalizing on the caller's behalf would let the bug reach
//! downstream metrics (level-sweep gating, style comparison) that
//! consume these outputs as if they were meaningful. Every validation
//! failure surfaces as a typed [`MetricError`] variant carrying enough
//! context for the caller to locate and fix the source.

use thiserror::Error;

pub mod bootstrap;

/// Absolute tolerance used when checking that a distribution sums to
/// `1.0`.
///
/// `1e-4` is loose enough to accept the rounding drift of an f32
/// softmax over a few thousand elements yet tight enough to catch the
/// common "forgot to renormalize after masking" bug (which usually
/// pushes the sum by 0.05 or more).
pub const NORM_TOL: f32 = 1e-4;

/// Error variants produced by the four metric primitives.
///
/// Each variant carries the offending index / value / sum so a Lua
/// bridge (or a caller inspecting the `Result`) can pinpoint the row
/// that violated the contract instead of receiving a bare boolean
/// "invalid input".
#[derive(Debug, Error, PartialEq)]
pub enum MetricError {
    /// A zero-length distribution was passed. Every primitive rejects
    /// empty inputs because probability over an empty support is
    /// undefined.
    #[error("metric: empty distribution")]
    Empty,

    /// Two distributions handed to a pairwise primitive
    /// ([`kl`] / [`js`] / [`tvd`]) had differing lengths. Both sides of
    /// a KL / JS / TVD computation must share the same support.
    #[error("metric: length mismatch: p={p}, q={q}")]
    LengthMismatch {
        /// Length of the `p` distribution.
        p: usize,
        /// Length of the `q` distribution.
        q: usize,
    },

    /// A distribution contained a non-finite value (`NaN` / `±inf`) at
    /// the given index. `is_finite()` is the sole check — subnormals
    /// pass.
    #[error("metric: non-finite element at index {index}: {value}")]
    NonFinite {
        /// Zero-based index of the offending element.
        index: usize,
        /// The non-finite value that was observed.
        value: f32,
    },

    /// A distribution contained a strictly negative probability at the
    /// given index. Zero is permitted (interpreted per the
    /// `0·log 0 := 0` convention in [`kl`] / [`entropy`]).
    #[error("metric: negative probability at index {index}: {value}")]
    Negative {
        /// Zero-based index of the offending element.
        index: usize,
        /// The negative value that was observed.
        value: f32,
    },

    /// A distribution failed the normalization check: its sum lies
    /// outside `1.0 ± tol`. The observed `sum` and the `tol` in force
    /// are both returned for debugging.
    #[error("metric: distribution sums to {sum}, expected 1.0 ± {tol}")]
    NotNormalized {
        /// Observed sum of the distribution.
        sum: f32,
        /// Absolute tolerance that was applied.
        tol: f32,
    },
}

/// Verify that `dist` is a well-formed probability distribution.
///
/// Checks (in order):
///
/// 1. Non-empty.
/// 2. Every element is finite (rejects `NaN` and `±inf`).
/// 3. Every element is non-negative.
/// 4. The sum lies within `1.0 ± NORM_TOL`.
///
/// The order is chosen so the more specific per-index diagnostics
/// (`NonFinite` / `Negative`) win over the aggregate `NotNormalized`
/// verdict: a distribution with a `NaN` entry would trivially fail both
/// checks, and pointing at the offending index is more useful than
/// reporting the resulting `NaN` sum.
fn validate(dist: &[f32]) -> Result<(), MetricError> {
    if dist.is_empty() {
        return Err(MetricError::Empty);
    }
    let mut sum = 0.0f32;
    for (index, &value) in dist.iter().enumerate() {
        if !value.is_finite() {
            return Err(MetricError::NonFinite { index, value });
        }
        if value < 0.0 {
            return Err(MetricError::Negative { index, value });
        }
        sum += value;
    }
    if (sum - 1.0).abs() > NORM_TOL {
        return Err(MetricError::NotNormalized { sum, tol: NORM_TOL });
    }
    Ok(())
}

/// Verify pairwise inputs share the same length in addition to the
/// per-distribution checks in [`validate`].
///
/// Length check first (cheapest and structurally most damning), then
/// each side is validated on its own so per-index diagnostics still
/// point at the actual offending row.
fn validate_pair(p: &[f32], q: &[f32]) -> Result<(), MetricError> {
    if p.len() != q.len() {
        return Err(MetricError::LengthMismatch {
            p: p.len(),
            q: q.len(),
        });
    }
    validate(p)?;
    validate(q)?;
    Ok(())
}

/// Kullback-Leibler divergence `D_KL(p || q)`, in nats.
///
/// Non-symmetric: `kl(p, q) != kl(q, p)` in general.
///
/// # Conventions
///
/// - `p[i] == 0` contributes `0` to the sum (the `0·log 0 := 0`
///   convention). This is the standard extension by continuity used
///   throughout information theory.
/// - `q[i] == 0 && p[i] > 0` yields `+∞` — the KL definition is
///   genuinely infinite here (`p` places mass where `q` places none),
///   and the caller is expected to treat the infinity as a valid
///   answer rather than a bug.
/// - `p[i] == 0 && q[i] == 0` contributes `0` (both terms vanish).
///
/// # Errors
///
/// Returns [`MetricError::LengthMismatch`] if `p.len() != q.len()`,
/// and any of the [`validate`] failures for either side.
pub fn kl(p: &[f32], q: &[f32]) -> Result<f32, MetricError> {
    validate_pair(p, q)?;
    let mut acc = 0.0f32;
    for (pi, qi) in p.iter().zip(q.iter()) {
        if *pi == 0.0 {
            continue;
        }
        if *qi == 0.0 {
            return Ok(f32::INFINITY);
        }
        acc += pi * (pi / qi).ln();
    }
    Ok(acc)
}

/// Jensen-Shannon divergence, in nats.
///
/// Defined as `0.5 * (KL(p || m) + KL(q || m))` with
/// `m = 0.5 * (p + q)`. Symmetric and bounded on `[0, ln 2]`.
///
/// # Numerical properties
///
/// Because `m[i] >= 0.5 * max(p[i], q[i])`, the mixture is strictly
/// positive wherever either `p[i]` or `q[i]` is, so JS never emits
/// `+∞` for validated inputs — the failure mode that plagues raw KL
/// under sparse `q` disappears.
///
/// # Errors
///
/// Same as [`kl`]: length mismatch and the per-side [`validate`]
/// failures.
pub fn js(p: &[f32], q: &[f32]) -> Result<f32, MetricError> {
    validate_pair(p, q)?;
    let mut acc = 0.0f32;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let mi = 0.5 * (pi + qi);
        if *pi > 0.0 {
            acc += pi * (pi / mi).ln();
        }
        if *qi > 0.0 {
            acc += qi * (qi / mi).ln();
        }
    }
    Ok(0.5 * acc)
}

/// Total Variation Distance = `0.5 * L1(p, q)`.
///
/// Symmetric and bounded on `[0, 1]`, which makes it the most directly
/// display-friendly of the four primitives ("the two distributions
/// disagree on 37% of the probability mass").
///
/// # Errors
///
/// Same as [`kl`]: length mismatch and the per-side [`validate`]
/// failures.
pub fn tvd(p: &[f32], q: &[f32]) -> Result<f32, MetricError> {
    validate_pair(p, q)?;
    let sum: f32 = p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).sum();
    Ok(0.5 * sum)
}

/// Shannon entropy `H(p) = -Σ p[i] * ln(p[i])`, in nats.
///
/// The `p[i] == 0` term is dropped (the `0·log 0 := 0` convention),
/// making entropy well-defined on distributions with hard zeros.
/// Bounded on `[0, ln(len)]`: hits `0` on a one-hot and reaches
/// `ln(n)` on the uniform distribution of length `n`.
///
/// # Errors
///
/// Any of the [`validate`] failures.
pub fn entropy(p: &[f32]) -> Result<f32, MetricError> {
    validate(p)?;
    let mut acc = 0.0f32;
    for &pi in p {
        if pi > 0.0 {
            acc -= pi * pi.ln();
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests;
