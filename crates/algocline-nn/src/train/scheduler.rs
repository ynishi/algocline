//! Learning-rate schedules for the training loop.
//!
//! Trainers ask for a scheduler once at construction time and then call
//! [`Scheduler::lr_at`] each step to get the current learning rate.
//! The loop then passes the value through to the optimizer via
//! `AdamW::set_learning_rate`.
//!
//! Two schedules ship today: a plain constant one (useful for tests and
//! sanity checks) and a cosine schedule with linear warmup that matches
//! the nanoGPT / HF Trainer default. Both live behind the [`Schedule`]
//! enum so a config can pick between them via a plain string field.

use std::f64::consts::PI;

/// Which schedule variant the trainer requested.
///
/// Adding a new variant is intentionally cheap — grow the enum and
/// implement [`Scheduler::lr_at`] for it. The loop does not need to be
/// touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleKind {
    /// Constant learning rate for every step.
    Constant,
    /// Linear warmup from 0 to `base_lr` over `warmup` steps, then a
    /// half-cosine decay to `min_lr` across the remaining steps.
    CosineWithWarmup,
}

impl ScheduleKind {
    /// Parse the wire form written by callers (Lua bridge, JSON config).
    ///
    /// Returns `None` on an unknown string so the caller can surface an
    /// actionable error message. Deliberately not `impl FromStr`: the
    /// signature returns `Option` rather than `Result<_, Err>` so the
    /// bridge can turn "unknown schedule" into a bespoke Lua error
    /// message rather than a generic `FromStr::Err`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "constant" | "const" => Some(Self::Constant),
            "cosine" | "cosine_with_warmup" => Some(Self::CosineWithWarmup),
            _ => None,
        }
    }
}

/// Learning-rate schedule state carried across steps.
#[derive(Debug, Clone)]
pub struct Scheduler {
    kind: ScheduleKind,
    base_lr: f64,
    min_lr: f64,
    warmup: usize,
    total_steps: usize,
}

impl Scheduler {
    /// Build a scheduler.
    ///
    /// `warmup` may be 0 for a schedule that starts at `base_lr`
    /// immediately. `total_steps` must be at least `warmup + 1`; the
    /// constructor clamps rather than errors so a caller cannot trip a
    /// runtime divide-by-zero later.
    pub fn new(
        kind: ScheduleKind,
        base_lr: f64,
        min_lr: f64,
        warmup: usize,
        total_steps: usize,
    ) -> Self {
        let total_steps = total_steps.max(warmup + 1);
        Self {
            kind,
            base_lr,
            min_lr,
            warmup,
            total_steps,
        }
    }

    /// Convenience — a cosine schedule with zero minimum LR.
    pub fn cosine(base_lr: f64, warmup: usize, total_steps: usize) -> Self {
        Self::new(
            ScheduleKind::CosineWithWarmup,
            base_lr,
            0.0,
            warmup,
            total_steps,
        )
    }

    /// The base learning rate — the peak of any schedule.
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// The schedule variant this scheduler is running.
    pub fn kind(&self) -> ScheduleKind {
        self.kind
    }

    /// Return the learning rate for `step` (0-indexed).
    ///
    /// Steps beyond `total_steps` return the terminal LR (either
    /// `min_lr` for the cosine schedule or `base_lr` for a constant
    /// schedule). This lets a caller do a "long tail" pass without
    /// coding a special case.
    pub fn lr_at(&self, step: usize) -> f64 {
        match self.kind {
            ScheduleKind::Constant => self.base_lr,
            ScheduleKind::CosineWithWarmup => {
                if step < self.warmup {
                    // Linear warmup — divide by (warmup + 1) so step
                    // 0 gets a positive LR rather than exactly zero.
                    let denom = (self.warmup as f64).max(1.0);
                    self.base_lr * ((step as f64 + 1.0) / (denom + 1.0)).min(1.0)
                } else if step >= self.total_steps {
                    self.min_lr
                } else {
                    // Half-cosine from `base_lr` down to `min_lr`.
                    let progress = (step - self.warmup) as f64
                        / (self.total_steps - self.warmup).max(1) as f64;
                    let progress = progress.clamp(0.0, 1.0);
                    let cos = (PI * progress).cos();
                    self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_recognises_aliases_and_rejects_unknown() {
        assert_eq!(
            ScheduleKind::parse("constant"),
            Some(ScheduleKind::Constant)
        );
        assert_eq!(ScheduleKind::parse("const"), Some(ScheduleKind::Constant));
        assert_eq!(
            ScheduleKind::parse("cosine"),
            Some(ScheduleKind::CosineWithWarmup)
        );
        assert_eq!(
            ScheduleKind::parse("cosine_with_warmup"),
            Some(ScheduleKind::CosineWithWarmup)
        );
        assert!(ScheduleKind::parse("banana").is_none());
    }

    #[test]
    fn constant_schedule_returns_base_lr_at_every_step() {
        let s = Scheduler::new(ScheduleKind::Constant, 3e-4, 0.0, 0, 100);
        assert_eq!(s.lr_at(0), 3e-4);
        assert_eq!(s.lr_at(50), 3e-4);
        assert_eq!(s.lr_at(1_000_000), 3e-4);
    }

    #[test]
    fn cosine_warmup_ramps_from_zero_and_peaks_at_end_of_warmup() {
        // 100 warmup steps, 1_000 total.
        let s = Scheduler::cosine(3e-4, 100, 1000);
        // Step 0 should be small but strictly positive.
        let lr0 = s.lr_at(0);
        assert!(lr0 > 0.0 && lr0 < 3e-4);
        // At the last warmup step the LR reaches (or exceeds) base_lr.
        let lr_peak = s.lr_at(99);
        assert!(lr_peak >= s.lr_at(50));
        assert!(lr_peak > 0.9 * 3e-4);
    }

    #[test]
    fn cosine_decays_to_min_lr_at_total_steps() {
        let s = Scheduler::new(ScheduleKind::CosineWithWarmup, 1.0, 0.05, 10, 100);
        let lr_end = s.lr_at(100);
        assert!(
            (lr_end - 0.05).abs() < 1e-9,
            "expected min_lr at total_steps, got {lr_end}"
        );
        // Beyond total_steps we clamp to min_lr.
        let lr_beyond = s.lr_at(500);
        assert_eq!(lr_beyond, 0.05);
    }

    #[test]
    fn cosine_lr_is_monotonically_non_increasing_after_warmup() {
        let s = Scheduler::cosine(1.0, 5, 50);
        let mut prev = s.lr_at(5);
        for step in 6..50 {
            let cur = s.lr_at(step);
            assert!(
                cur <= prev + 1e-12,
                "cosine tail must not increase (step={step}, prev={prev}, cur={cur})"
            );
            prev = cur;
        }
    }

    #[test]
    fn total_steps_less_than_warmup_is_clamped() {
        // Deliberately misconfigured: warmup > total. Constructor
        // should clamp so subsequent `lr_at` calls do not divide by
        // zero.
        let s = Scheduler::new(ScheduleKind::CosineWithWarmup, 1e-3, 0.0, 100, 10);
        // Every step should return a finite number.
        for step in 0..200 {
            let lr = s.lr_at(step);
            assert!(lr.is_finite(), "step={step}, lr={lr}");
        }
    }
}
