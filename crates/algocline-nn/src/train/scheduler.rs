//! Learning-rate schedules for the training loop.
//!
//! Trainers ask for a scheduler once at construction time and then call
//! [`Scheduler::lr_at`] each step to get the current learning rate.
//! The loop then passes the value through to the optimizer via
//! `AdamW::set_learning_rate`.
//!
//! Four schedules ship: a plain constant one (useful for tests and
//! sanity checks), cosine with linear warmup (the nanoGPT / HF Trainer
//! default), linear with warmup, and warmup-stable-decay. All live
//! behind the [`ScheduleKind`] enum so a config can pick between them
//! via a plain string field.
//!
//! The three warmup-bearing schedules share one warmup ramp and differ
//! only in what they do afterwards, which is why they sit in one enum
//! rather than in separate types: a caller switching between them is
//! changing the tail of a curve, not the kind of thing it is.

use std::f64::consts::PI;

/// Which schedule variant the trainer requested.
///
/// Adding a new variant is intentionally cheap — grow the enum and
/// implement [`Scheduler::lr_at`] for it. The loop does not need to be
/// touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScheduleKind {
    /// Constant learning rate for every step.
    Constant,
    /// Linear warmup from 0 to `base_lr` over `warmup` steps, then a
    /// half-cosine decay to `min_lr` across the remaining steps.
    CosineWithWarmup,
    /// Linear warmup from 0 to `base_lr` over `warmup` steps, then a
    /// straight line down to `min_lr` across the remaining steps.
    ///
    /// The shape HF Transformers calls `linear` and uses as its own
    /// default, where the floor is a hard zero; here it is `min_lr`, so
    /// the zero form is the default `min_lr` rather than a separate
    /// variant.
    Linear,
    /// Linear warmup, then a constant stretch at `base_lr`, then a
    /// half-cosine decay to `min_lr` over the last `decay_steps`.
    ///
    /// Introduced as WSD in MiniCPM (arXiv:2404.06395 Eq. 1), where the
    /// decay function `f` is left open — any decreasing `f` with
    /// `0 < f ≤ 1` is the scheme. Cosine is the choice here, matching
    /// the `decay_type` default of HF's `get_wsd_schedule`.
    ///
    /// What separates it from [`Self::CosineWithWarmup`] is that the
    /// stable stretch does not know where the run ends: a checkpoint
    /// taken during it was trained at the same LR as the one before it,
    /// so the run can be continued rather than only restarted. That is
    /// the property the schedule exists for, and it is lost the moment
    /// the decay begins.
    WarmupStableDecay,
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
            "linear" | "linear_with_warmup" => Some(Self::Linear),
            "wsd" | "warmup_stable_decay" => Some(Self::WarmupStableDecay),
            _ => None,
        }
    }

    /// Every wire name this version accepts, for error messages that
    /// name the alternatives rather than only the rejected value.
    pub const NAMES: [&'static str; 8] = [
        "constant",
        "const",
        "cosine",
        "cosine_with_warmup",
        "linear",
        "linear_with_warmup",
        "wsd",
        "warmup_stable_decay",
    ];
}

/// Learning-rate schedule state carried across steps.
#[derive(Debug, Clone)]
pub struct Scheduler {
    kind: ScheduleKind,
    base_lr: f64,
    min_lr: f64,
    warmup: usize,
    total_steps: usize,
    /// Length of the decay stretch, for the schedules that have one
    /// separate from "everything after warmup". `None` leaves it at the
    /// share [`Self::DEFAULT_DECAY_FRACTION`] names.
    decay_steps: Option<usize>,
}

impl Scheduler {
    /// Build a scheduler.
    ///
    /// `warmup` may be 0 for a schedule that starts at `base_lr`
    /// immediately. `total_steps` must be at least `warmup + 1`; the
    /// constructor clamps rather than errors so a caller cannot trip a
    /// runtime divide-by-zero later.
    ///
    /// `min_lr` floors the **tail** — where a decaying schedule lands
    /// at `total_steps` and stays afterwards. It does not raise the
    /// warmup ramp, which climbs from near zero and therefore passes
    /// below `min_lr` on its way up.
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
            decay_steps: None,
        }
    }

    /// Share of the run the decay stretch takes when the caller does
    /// not name one.
    ///
    /// MiniCPM (arXiv:2404.06395 §3.2) reports 10% of tokens as
    /// sufficient and 2.5% as short of it, so 10% is the default rather
    /// than a round number picked for looking like one.
    pub const DEFAULT_DECAY_FRACTION: f64 = 0.1;

    /// Set the length of the decay stretch, for
    /// [`ScheduleKind::WarmupStableDecay`].
    ///
    /// A builder rather than a sixth constructor argument so callers of
    /// [`Self::new`] keep compiling. Clamped into `1..=(total - warmup)`
    /// at read time: a decay longer than the run would start before
    /// warmup ended, and a decay of zero is a constant schedule wearing
    /// another name.
    pub fn with_decay_steps(mut self, decay_steps: usize) -> Self {
        self.decay_steps = Some(decay_steps);
        self
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

    /// Length of the decay stretch, resolved and clamped.
    fn decay_len(&self) -> usize {
        let after_warmup = self.total_steps.saturating_sub(self.warmup).max(1);
        let requested = self.decay_steps.unwrap_or_else(|| {
            ((self.total_steps as f64) * Self::DEFAULT_DECAY_FRACTION).round() as usize
        });
        requested.clamp(1, after_warmup)
    }

    /// The warmup ramp shared by every schedule that has one.
    ///
    /// Divides by `warmup + 1` so step 0 gets a positive LR rather than
    /// exactly zero — a first step at zero LR is a step that did not
    /// happen, and it is invisible in the loss curve.
    fn warmup_lr(&self, step: usize) -> f64 {
        let denom = (self.warmup as f64).max(1.0);
        self.base_lr * ((step as f64 + 1.0) / (denom + 1.0)).min(1.0)
    }

    /// Half-cosine from `base_lr` down to `min_lr` over `progress`
    /// in `[0, 1]`.
    fn cosine_at(&self, progress: f64) -> f64 {
        let progress = progress.clamp(0.0, 1.0);
        let cos = (PI * progress).cos();
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos)
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
        if !matches!(self.kind, ScheduleKind::Constant) && step < self.warmup {
            return self.warmup_lr(step);
        }
        match self.kind {
            ScheduleKind::Constant => self.base_lr,
            ScheduleKind::CosineWithWarmup => {
                if step >= self.total_steps {
                    self.min_lr
                } else {
                    let progress = (step - self.warmup) as f64
                        / (self.total_steps - self.warmup).max(1) as f64;
                    self.cosine_at(progress)
                }
            }
            ScheduleKind::Linear => {
                if step >= self.total_steps {
                    self.min_lr
                } else {
                    let progress = (step - self.warmup) as f64
                        / (self.total_steps - self.warmup).max(1) as f64;
                    let progress = progress.clamp(0.0, 1.0);
                    self.base_lr + (self.min_lr - self.base_lr) * progress
                }
            }
            ScheduleKind::WarmupStableDecay => {
                if step >= self.total_steps {
                    return self.min_lr;
                }
                let decay_len = self.decay_len();
                // The stable stretch runs to `total - decay_len`, never
                // ending before warmup does: a decay clamped to the
                // whole post-warmup run leaves no stable stretch, which
                // is the cosine schedule and is reported as such rather
                // than as a WSD run that silently had none.
                let decay_start = self.total_steps.saturating_sub(decay_len).max(self.warmup);
                if step < decay_start {
                    self.base_lr
                } else {
                    let progress = (step - decay_start) as f64
                        / (self.total_steps - decay_start).max(1) as f64;
                    self.cosine_at(progress)
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

    /// The shape HF calls `linear`: a straight line, so the midpoint of
    /// the post-warmup stretch sits exactly halfway between the peak
    /// and the floor. That is what separates it from the cosine, which
    /// is above the halfway mark there.
    #[test]
    fn linear_falls_at_a_constant_rate_to_min_lr() {
        let s = Scheduler::new(ScheduleKind::Linear, 1.0, 0.2, 0, 100);
        assert!((s.lr_at(0) - 1.0).abs() < 1e-12, "{}", s.lr_at(0));
        assert!((s.lr_at(50) - 0.6).abs() < 1e-12, "{}", s.lr_at(50));
        assert!((s.lr_at(100) - 0.2).abs() < 1e-12, "{}", s.lr_at(100));

        // Constant slope: every consecutive difference is the same.
        let d1 = s.lr_at(10) - s.lr_at(11);
        let d2 = s.lr_at(70) - s.lr_at(71);
        assert!((d1 - d2).abs() < 1e-12, "d1={d1} d2={d2}");

        let cosine = Scheduler::new(ScheduleKind::CosineWithWarmup, 1.0, 0.2, 0, 100);
        assert!(
            cosine.lr_at(50) > s.lr_at(50),
            "the cosine sits above the straight line at the midpoint"
        );
    }

    /// The warmup ramp is shared, so the three schedules that have one
    /// agree step for step until it ends.
    #[test]
    fn every_warmup_bearing_schedule_ramps_identically() {
        let kinds = [
            ScheduleKind::CosineWithWarmup,
            ScheduleKind::Linear,
            ScheduleKind::WarmupStableDecay,
        ];
        let scheds: Vec<Scheduler> = kinds
            .iter()
            .map(|k| Scheduler::new(*k, 1.0, 0.0, 10, 100))
            .collect();
        for step in 0..10 {
            let first = scheds[0].lr_at(step);
            assert!(first > 0.0, "step 0 must not be a step at zero LR");
            for s in &scheds[1..] {
                assert!(
                    (s.lr_at(step) - first).abs() < 1e-12,
                    "warmup step {step} disagrees: {} vs {first}",
                    s.lr_at(step)
                );
            }
        }
    }

    /// WSD's stable stretch is the point of the schedule: a checkpoint
    /// taken inside it was trained at the same LR as the ones around
    /// it, so the run can be continued rather than only restarted.
    #[test]
    fn wsd_holds_base_lr_until_the_decay_begins() {
        // 100 steps, 10 warmup, decay named explicitly as the last 20.
        let s =
            Scheduler::new(ScheduleKind::WarmupStableDecay, 1.0, 0.1, 10, 100).with_decay_steps(20);

        for step in 10..80 {
            assert!(
                (s.lr_at(step) - 1.0).abs() < 1e-12,
                "stable stretch must hold base_lr (step={step}, lr={})",
                s.lr_at(step)
            );
        }
        assert!(s.lr_at(80) <= 1.0 + 1e-12);
        assert!(s.lr_at(90) < 1.0, "the decay must have started by step 90");
        assert!((s.lr_at(100) - 0.1).abs() < 1e-12, "{}", s.lr_at(100));

        // Monotone once it starts down.
        let mut prev = s.lr_at(80);
        for step in 81..=100 {
            let cur = s.lr_at(step);
            assert!(cur <= prev + 1e-12, "step={step} prev={prev} cur={cur}");
            prev = cur;
        }
    }

    /// Unnamed, the decay takes the share MiniCPM reports as
    /// sufficient. Pinned because the alternative — deriving it from
    /// whatever the caller left unset — is the kind of default that
    /// changes under you.
    #[test]
    fn wsd_decay_defaults_to_a_tenth_of_the_run() {
        let s = Scheduler::new(ScheduleKind::WarmupStableDecay, 1.0, 0.0, 0, 1000);
        assert!(
            (s.lr_at(899) - 1.0).abs() < 1e-12,
            "step 899 is still stable, got {}",
            s.lr_at(899)
        );
        assert!(
            s.lr_at(950) < 1.0,
            "step 950 is inside the last tenth and must be decaying"
        );
    }

    /// A decay longer than the run cannot start before warmup ends, and
    /// a decay of zero is not a way to spell "constant".
    ///
    /// The band is checked from the end of warmup on, not from step 0:
    /// `min_lr` is the floor of the *tail*, and the warmup ramp climbs
    /// from near zero, so it passes below the floor on its way up. That
    /// is what every warmup-bearing schedule here already does, and
    /// making this one start at `min_lr` instead would split the ramp
    /// the sibling schedules share.
    #[test]
    fn wsd_clamps_a_decay_that_does_not_fit() {
        for decay in [0usize, 1, 10_000] {
            let s = Scheduler::new(ScheduleKind::WarmupStableDecay, 1.0, 0.1, 10, 100)
                .with_decay_steps(decay);
            for step in 0..150 {
                let lr = s.lr_at(step);
                assert!(lr.is_finite(), "decay={decay} step={step} lr={lr}");
                if step >= 10 {
                    assert!(
                        (0.1 - 1e-12..=1.0 + 1e-12).contains(&lr),
                        "decay={decay} step={step} lr={lr} left the [min_lr, base_lr] band"
                    );
                }
            }
        }
    }

    /// The floor governs the tail, not the ramp — pinned so the
    /// behaviour is a decision on record rather than something noticed
    /// later in a curve.
    #[test]
    fn warmup_passes_below_min_lr_on_its_way_up() {
        for kind in [
            ScheduleKind::CosineWithWarmup,
            ScheduleKind::Linear,
            ScheduleKind::WarmupStableDecay,
        ] {
            let s = Scheduler::new(kind, 1.0, 0.5, 10, 100);
            assert!(
                s.lr_at(0) < 0.5,
                "{kind:?}: the first warmup step sits below min_lr by construction"
            );
            assert!(
                s.lr_at(99) >= 0.5 - 1e-12,
                "{kind:?}: the tail is floored at min_lr"
            );
        }
    }

    #[test]
    fn the_new_names_parse_and_are_listed() {
        assert_eq!(ScheduleKind::parse("linear"), Some(ScheduleKind::Linear));
        assert_eq!(
            ScheduleKind::parse("linear_with_warmup"),
            Some(ScheduleKind::Linear)
        );
        assert_eq!(
            ScheduleKind::parse("wsd"),
            Some(ScheduleKind::WarmupStableDecay)
        );
        assert_eq!(
            ScheduleKind::parse("warmup_stable_decay"),
            Some(ScheduleKind::WarmupStableDecay)
        );
        assert_eq!(ScheduleKind::parse("triangular"), None);

        // Every listed name parses: the list exists so a refusal can
        // name the alternatives, and a name in it that does not parse
        // would send the caller to a value that is also refused.
        for name in ScheduleKind::NAMES {
            assert!(
                ScheduleKind::parse(name).is_some(),
                "NAMES lists `{name}`, which does not parse"
            );
        }
    }
}
