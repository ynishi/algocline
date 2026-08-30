//! Interval estimates for statistics measured during a training run.
//!
//! [`bootstrap`] is what is left here: given observations that arrive in
//! correlated groups, it puts a confidence interval around a statistic
//! computed over them. That is the question a checkpoint search asks —
//! two candidates differ by some margin, and the caller needs to know
//! how much of the margin is the draw.
//!
//! # What used to be here
//!
//! Four distribution primitives (`kl`, `js`, `tvd`, `entropy`) and their
//! `MetricError` / `NORM_TOL` validation contract. They were
//! general-purpose mathematics rather than anything this crate owns, and
//! `mlua-mathlib` had grown its own information-theory module, so the
//! two implementations disagreed about the one thing they both had to
//! decide: how much rounding drift a vector may carry and still count as
//! a distribution. This crate's answer was a fixed `1e-4` at every
//! length; mathlib's grows as `32·√n·u`, which is the shape the error
//! actually takes. Keeping the looser fixed bound would have meant a
//! 50257-entry vocabulary could lose 0.3% of its mass unnoticed.
//!
//! Callers read them from `alc.math.{kl_divergence, js_divergence, tvd,
//! entropy}` now. Nothing in this crate consumed them from Rust.
//!
//! # Why the bootstrap stays
//!
//! It is not a distribution primitive. It takes a caller's statistic as
//! a closure and resamples *clusters* of observations, which is a claim
//! about how training-run measurements correlate (games sharing a seed,
//! positions sharing a game) rather than a claim about probability
//! vectors. That belongs next to the trainer.

pub mod bootstrap;
