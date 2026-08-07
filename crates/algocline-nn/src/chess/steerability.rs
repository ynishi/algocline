//! The two pre-registered statistics, assembled from several arms'
//! records over one resample of the games.
//!
//! Steerability is how far behaviour moves when the conditioning input
//! changes. Two questions were fixed before any of it was measured:
//! whether conditioning at every position changes the played move more
//! often than a prefix token does (H14), and whether it also reaches
//! further down the game (H15).
//!
//! # Both are differences, and neither is a difference of two intervals
//!
//! H14's quantity is
//!
//! ```text
//! M = flip(perpos) - flip(prefix)      the two-arm margin
//! G = |flip(perpos) - flip(perpos-b)|  the same-arm gap
//! ```
//!
//! confirmed when `M - G` is entirely above zero and refuted when
//! `M + G` is entirely below it. The two criteria are mirrored on
//! purpose. Under the null that the two arms have the same true rate,
//! `E[M] = 0` while `G` is a strictly positive gap between two real
//! checkpoints, so `M - G` converges to something negative **under the
//! null** — a refute branch keyed on it would fire whenever
//! per-position merely failed to win, including when the two arms were
//! exactly equal. `M + G` asks the intended question: worse by more than
//! the same-arm gap.
//!
//! Both terms are recomputed inside every draw, over the same resampled
//! games. Nothing is frozen as a scalar: the gap is one realisation from
//! one pair of runs, and treating it as a known constant would report a
//! precision that pair does not carry.
//!
//! H15's quantity is the difference of two decay ratios,
//! `JS(deep) / JS(shallow)`, one per arm, on the same resamples.
//!
//! # Why a shared resample rather than two intervals compared
//!
//! An interval on `flip(perpos)` and an interval on `flip(prefix)`
//! would each be correct and would say nothing about the gap between
//! them: whether two intervals overlap is not a test of whether their
//! difference excludes zero. Every draw here is one list of games, and
//! every term in that draw is computed from it
//! ([`crate::metric::bootstrap`]).
//!
//! # The floor is one realisation
//!
//! `G` estimates run-to-run variation from a single pair of runs — the
//! same arm trained twice with a different shuffle seed. Resampling
//! games says how precisely *that pair's* gap was measured on this
//! position set. It says nothing about how large a gap two other runs
//! would show, so a confirmed H14 reads "the margin exceeds one observed
//! same-arm gap" and not "exceeds the floor".

use thiserror::Error;

use crate::chess::records::{AlignError, AlignedArms, GammaRecord};
use crate::chess::window::COND_PREFIX_LEN;
use crate::chess::CondEncoding;
use crate::metric::bootstrap::{cluster_bootstrap, BootstrapError, ClusterTally, Interval};

/// Arm trained with the band conditioning every position.
pub const PERPOS: &str = "perpos";

/// Arm trained with the band as a prefix token, which is what every
/// checkpoint before this experiment did.
pub const PREFIX: &str = "prefix";

/// Second per-position run, differing from [`PERPOS`] only in the
/// shuffle seed. The gap between the two is the same-arm gap `G`.
pub const PERPOS_B: &str = "perpos-b";

/// Bootstrap draws.
///
/// Two thousand, from the plan. Fixed here rather than left to the
/// caller for the same reason the confidence level is: a draw count
/// chosen per invocation is one that can be raised until an interval
/// lands where someone wanted it.
pub const DRAWS: usize = 2000;

/// How far an arm's mean top-1 match may sit from the prefix arm's.
///
/// The second validity gate. Not a threshold this file invented: it is
/// recorded rather than optimised, and an arm outside it is excluded at
/// that gamma rather than adjusted.
pub const TOP1_GATE: f64 = 0.02;

/// How far an arm's legal mass may sit from the prefix arm's, as a
/// fraction.
///
/// The third validity gate, checked at every gamma because over-guidance
/// drains legal mass and that is its documented shape.
pub const LEGAL_MASS_GATE: f64 = 0.20;

/// Plies the shallow depth bucket covers: the opening.
pub const SHALLOW_BUCKET: (usize, usize) = (0, 10);

/// First ply of the deep bucket. Its last ply depends on the context
/// window — see [`deep_bucket`].
pub const DEEP_BUCKET_LOW: usize = 40;

/// The deep bucket at a given context window, as a half-open range.
///
/// It ends at `ctx - COND_PREFIX_LEN` rather than running to the end of
/// the game. A conditioned row is `[BOS, band] + moves`, so at ply
/// `ctx - 1` the row outgrows the window and the tail slice starts
/// dropping tokens off the front. Positions past that boundary measure
/// truncation rather than distance, and mixing the two would put a
/// regime this plan holds out of scope into the evidence for a
/// hypothesis about depth.
///
/// The boundary is one ply short of where truncation actually begins —
/// ply `ctx - 2` makes exactly `ctx` tokens and still fits whole — which
/// is kept rather than corrected, because the published deep figures
/// were computed against this edge and moving it would make them
/// incomparable.
pub fn deep_bucket(ctx: usize) -> (usize, usize) {
    (DEEP_BUCKET_LOW, ctx.saturating_sub(COND_PREFIX_LEN))
}

/// Why a statistic could not be assembled.
#[derive(Debug, Error)]
pub enum StatError {
    /// The arms do not describe one position stream, or one of the
    /// named arms is missing.
    #[error(transparent)]
    Align(#[from] AlignError),

    /// The resampling refused.
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),

    /// A depth bucket caught no positions, so the ratio it is the
    /// denominator or numerator of does not exist.
    #[error(
        "the ply {low}-{high} bucket holds no positions, so the depth ratio has no {role}; \
         a walk that never reached this depth cannot answer a question about it"
    )]
    EmptyBucket {
        /// First ply of the bucket.
        low: usize,
        /// Last ply of the bucket.
        high: usize,
        /// Which half of the ratio it was needed for.
        role: &'static str,
    },

    /// An arm is playing a role its checkpoint was not conditioned for.
    ///
    /// The roles are not interchangeable and the numbers do not say so:
    /// swap the per-position arm with the prefix one and every figure is
    /// well-formed while the margin means the opposite of what it is
    /// read as. `chess_stats` prints the role-to-file mapping, which
    /// helps a reader who is looking; this is what refuses.
    ///
    /// The encoding is on the header because `chess_cond` writes it
    /// there, so the check costs nothing and rests on what the
    /// checkpoint's own sidecar recorded rather than on the file name.
    #[error(
        "arm {arm:?} was scored from a checkpoint conditioned by {found}, and that role calls \
         for {want}; a role swap leaves every number well-formed and reverses what the margin \
         means"
    )]
    WrongEncodingForRole {
        /// Arm whose encoding does not fit its role.
        arm: &'static str,
        /// What the role requires.
        want: CondEncoding,
        /// What the arm's header records.
        found: CondEncoding,
    },
}

/// Check every canonical role at once, before anything is computed.
///
/// The prefix arm's checkpoint must have been conditioned by a prefix
/// token; both per-position arms' must not.
///
/// The per-statistic checks fire from inside [`h14`], [`h15`] and the
/// two validity gates, each refusing its own call. That is too late for
/// a program that prints as it goes: the per-arm figures
/// ([`flip_rate`], [`legal_mass`], [`top1_match`]) take no baseline and
/// so refuse nothing, and under a perpos/prefix swap a full, plausible
/// table of them reaches the reader before the first refusal does. A
/// caller that prints should call this immediately after building the
/// arms.
///
/// The two per-position runs are **not** distinguished, and cannot be:
/// they differ only in a shuffle seed, which nothing in a record
/// records. See [`h14`] for what a swap between those two does and does
/// not change.
///
/// # Errors
///
/// One of the three arms is missing, or its checkpoint's conditioning
/// does not fit its role.
pub fn check_roles(arms: &AlignedArms) -> Result<(), StatError> {
    require_roles(
        arms,
        &[
            (PERPOS, CondEncoding::EveryPosition),
            (PREFIX, CondEncoding::Prefix),
            (PERPOS_B, CondEncoding::EveryPosition),
        ],
    )
}

fn require_roles(
    arms: &AlignedArms,
    roles: &[(&'static str, CondEncoding)],
) -> Result<(), StatError> {
    for (arm, want) in roles {
        let found = arms.walk(arm)?.header.encoding;
        if found != *want {
            return Err(StatError::WrongEncodingForRole {
                arm,
                want: *want,
                found,
            });
        }
    }
    Ok(())
}

/// Tally one per-position quantity, game by game.
///
/// `extract` returns `None` where the quantity is not defined at that
/// position, which drops it from both the sum and the count — a top-1
/// match against a move that is not in the vocabulary has no value, and
/// scoring it as a miss would drag the figure down by however many of
/// them there were.
fn tally(
    arms: &AlignedArms,
    arm: &str,
    gamma_ix: usize,
    extract: impl Fn(&GammaRecord) -> Option<f64>,
) -> Result<ClusterTally, StatError> {
    let walk = arms.walk(arm)?;
    let mut out = ClusterTally::new(arms.games());
    for (record, cluster) in walk.records.iter().zip(arms.clusters()) {
        let Some(at) = record.at.get(gamma_ix) else {
            continue;
        };
        if let Some(value) = extract(at) {
            out.push(*cluster, value)?;
        }
    }
    Ok(out)
}

/// Same, restricted to a half-open ply range.
fn tally_in_bucket(
    arms: &AlignedArms,
    arm: &str,
    gamma_ix: usize,
    bucket: (usize, usize),
    extract: impl Fn(&GammaRecord) -> Option<f64>,
) -> Result<ClusterTally, StatError> {
    let walk = arms.walk(arm)?;
    let mut out = ClusterTally::new(arms.games());
    for (record, cluster) in walk.records.iter().zip(arms.clusters()) {
        if record.ply < bucket.0 || record.ply >= bucket.1 {
            continue;
        }
        let Some(at) = record.at.get(gamma_ix) else {
            continue;
        };
        if let Some(value) = extract(at) {
            out.push(*cluster, value)?;
        }
    }
    Ok(out)
}

fn flipped(at: &GammaRecord) -> Option<f64> {
    Some(if at.flipped { 1.0 } else { 0.0 })
}

fn widest_js(at: &GammaRecord) -> Option<f64> {
    Some(at.widest_js)
}

/// Share of the bands whose top legal move equalled the human's.
///
/// `None` where the played move is not in the vocabulary, so there is
/// nothing to match against — absent rather than zero, since counting an
/// unscoreable position as a miss would drag the figure down by however
/// many of them there were.
fn top1_mean(at: &GammaRecord) -> Option<f64> {
    let per_band = at.top1.as_ref()?;
    if per_band.is_empty() {
        return None;
    }
    let hits = per_band.iter().filter(|m| **m).count() as f64;
    Some(hits / per_band.len() as f64)
}

/// Share of positions where at least one band's top legal move differed
/// from the first band's, with a cluster bootstrap over games.
///
/// # Errors
///
/// The arm or gamma is not among those walked, or the resampling
/// refused.
pub fn flip_rate(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Interval, StatError> {
    let gamma_ix = arms.gamma_index(gamma)?;
    let flips = tally(arms, arm, gamma_ix, flipped)?;
    Ok(cluster_bootstrap(arms.games(), draws, seed, |draw| {
        flips.mean_over(draw)
    })?)
}

/// Softmax mass landing on legal moves, meaned over the bands, with a
/// cluster bootstrap over games.
///
/// One of the validity gates, which is checked at every gamma used in a
/// judgement because over-guidance drains legal mass.
///
/// # Errors
///
/// As [`flip_rate`].
pub fn legal_mass(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Interval, StatError> {
    let gamma_ix = arms.gamma_index(gamma)?;
    let mass = tally(arms, arm, gamma_ix, |at| Some(at.legal_mass))?;
    Ok(cluster_bootstrap(arms.games(), draws, seed, |draw| {
        mass.mean_over(draw)
    })?)
}

/// Share of positions where the top legal move equalled the human's,
/// meaned over the bands, with a cluster bootstrap over games.
///
/// The other validity gate. Positions where the played move is not in
/// the vocabulary carry no value and are left out rather than counted as
/// misses.
///
/// # Errors
///
/// As [`flip_rate`].
pub fn top1_match(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Interval, StatError> {
    let gamma_ix = arms.gamma_index(gamma)?;
    let matches = tally(arms, arm, gamma_ix, top1_mean)?;
    Ok(cluster_bootstrap(arms.games(), draws, seed, |draw| {
        matches.mean_over(draw)
    })?)
}

/// One validity gate, as a difference against the prefix arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    /// Interval on the difference, over one set of game resamples.
    pub interval: Interval,
    /// How far the point estimate may sit from zero and still pass.
    pub tolerance: f64,
}

impl Gate {
    /// Whether the point estimate is inside the tolerance.
    ///
    /// Read off the point estimate rather than the interval, because
    /// the gate is a statement about the arms as measured and is
    /// **recorded rather than optimised** — the interval is reported so
    /// a reader can see how well the difference was pinned down, not so
    /// the gate can be argued into passing.
    pub fn passes(&self) -> bool {
        self.interval.point.abs() <= self.tolerance
    }

    /// `"pass"` or `"outside"`, for a report.
    pub fn verdict(&self) -> &'static str {
        if self.passes() {
            "pass"
        } else {
            "outside"
        }
    }
}

/// Second validity gate: mean top-1 match, as a difference against the
/// prefix arm.
///
/// A **difference on the shared draw**, not two intervals to be eyeballed
/// for overlap. Whether two intervals overlap is not a test of whether
/// their difference excludes zero, and printing the two arms side by
/// side invites exactly that reading — the one this module's header says
/// is not a test.
///
/// Returns a gate whose interval is centred on zero when the two arms
/// agree; the prefix arm compared against itself is identically zero.
///
/// # What is checked, and what cannot be
///
/// The baseline is [`PREFIX`] and nothing else, so this refuses a
/// [`PREFIX`] slot holding a checkpoint that was not conditioned by a
/// prefix token — a wrong-baseline gate table is the failure
/// [`check_roles`] exists for, and a function that depends on the
/// baseline should not be relying on its caller having run that first.
///
/// The subject is not checked here, and not because `arm` carries no
/// role — [`PERPOS`] and [`PERPOS_B`] are role names with declared
/// encodings, and a per-position slot holding a prefix checkpoint would
/// produce a well-formed row attributed to the wrong arm. It is because
/// the subject's role is validated up front by [`check_roles`], which
/// covers all three at once and refuses before anything is printed.
/// This function cannot make that check itself: `arm` is also a
/// legitimate free string ([`PREFIX`] against itself is a real call),
/// so the only place a role-bearing name can be recognised as one is
/// where the whole set is known.
///
/// A caller that reaches this without having run [`check_roles`] is
/// therefore guarded on its baseline and not on its subject.
///
/// # Errors
///
/// The arm or [`PREFIX`] is missing, [`PREFIX`] was not scored from a
/// prefix-conditioned checkpoint, the gamma was not swept, or the
/// resampling refused.
pub fn gate_top1(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    require_roles(arms, &[(PREFIX, CondEncoding::Prefix)])?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, arm, gamma_ix, top1_mean)?;
    let baseline = tally(arms, PREFIX, gamma_ix, top1_mean)?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(subject.mean_over(draw)? - baseline.mean_over(draw)?)
    })?;
    Ok(Gate {
        interval,
        tolerance: TOP1_GATE,
    })
}

/// Third validity gate: legal mass, as a **relative** difference against
/// the prefix arm.
///
/// Relative because the gate is stated as a percentage: an arm passes
/// while its legal mass is within a fifth of the prefix arm's. The
/// quantity bootstrapped is therefore `mass(arm)/mass(prefix) - 1`,
/// which is zero when the two agree and `-1` when the arm has lost all
/// of its legal mass, so the tolerance reads directly as the fraction
/// the gate names.
///
/// The baseline is guarded as in [`gate_top1`]: [`PREFIX`] has to have
/// been scored from a prefix-conditioned checkpoint, since that is the
/// arm this quantity divides by.
///
/// # Errors
///
/// As [`gate_top1`], plus an undefined ratio where the prefix arm's mass
/// is zero — which is reported as an undefined draw rather than as an
/// infinity.
pub fn gate_legal_mass(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    require_roles(arms, &[(PREFIX, CondEncoding::Prefix)])?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, arm, gamma_ix, |at| Some(at.legal_mass))?;
    let baseline = tally(arms, PREFIX, gamma_ix, |at| Some(at.legal_mass))?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        let reference = baseline.mean_over(draw)?;
        if reference <= 0.0 {
            return None;
        }
        Some(subject.mean_over(draw)? / reference - 1.0)
    })?;
    Ok(Gate {
        interval,
        tolerance: LEGAL_MASS_GATE,
    })
}

/// H14: does conditioning at every position change the played move more
/// often than a prefix token does?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H14 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// Flip rate of the per-position arm, on the sample as walked.
    pub flip_perpos: f64,
    /// Flip rate of the prefix arm.
    pub flip_prefix: f64,
    /// Flip rate of the second per-position run.
    pub flip_perpos_b: f64,
    /// `M`, the two-arm margin, on the sample as walked.
    pub margin: f64,
    /// `G`, the same-arm gap, on the sample as walked.
    pub gap: f64,
    /// Interval on `M - G`. Confirmed when it excludes zero from above.
    pub confirm: Interval,
    /// Interval on `M + G`. Refuted when it excludes zero from below.
    pub refute: Interval,
    /// Positions the three arms share.
    pub positions: usize,
    /// Games those positions came from — the clusters resampled.
    pub games: usize,
}

impl H14 {
    /// Whether the confirm interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below.
    pub fn refuted(&self) -> bool {
        self.refute.excludes_zero_from_below()
    }

    /// The verdict on one month, in the three-way form the decision
    /// table partitions outcomes with. A month is only judged when both
    /// months agree, which is the caller's business, not this type's.
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            // The region between the two criteria, which is wide by
            // construction: it is the price of estimating the run-to-run
            // gap from one pair of runs. Both non-confirmed outcomes are
            // treated identically downstream.
            _ => "undetermined",
        }
    }
}

/// Assemble H14 from three arms scored on one position stream.
///
/// The margin and the gap are recomputed inside every draw over the same
/// resampled games. The two intervals come from two calls to
/// [`cluster_bootstrap`], which draw the same games in the same order
/// because they are seeded identically over the same cluster count — its
/// resampling is deterministic, and its being so is tested.
///
/// # What a swap between the two per-position arms does
///
/// [`PERPOS`] and [`PERPOS_B`] are two runs of the same arm, differing
/// only in the shuffle seed, and nothing in a record says which is
/// which. Exchanging them on a command line is therefore accepted, and
/// what it changes is bounded: `G` is symmetric in the pair and does not
/// move at all; `M` is not, so it becomes the other replicate's margin
/// against the prefix arm. Both are valid readings of the same
/// experiment — the plan already treats the gap as one realisation from
/// one pair of runs — so the swap picks between two equally good draws
/// rather than producing a wrong one. That is the whole of it, and it is
/// why no check tries to tell the two apart.
///
/// # Errors
///
/// One of [`PERPOS`], [`PREFIX`], [`PERPOS_B`] is missing from `arms`,
/// the gamma was not swept, or the resampling refused.
pub fn h14(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H14, StatError> {
    require_roles(
        arms,
        &[
            (PERPOS, CondEncoding::EveryPosition),
            (PREFIX, CondEncoding::Prefix),
            (PERPOS_B, CondEncoding::EveryPosition),
        ],
    )?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let perpos = tally(arms, PERPOS, gamma_ix, flipped)?;
    let prefix = tally(arms, PREFIX, gamma_ix, flipped)?;
    let perpos_b = tally(arms, PERPOS_B, gamma_ix, flipped)?;

    // One draw, every term. `margin_and_gap` is the whole statistic: a
    // caller cannot reach a second resample from inside it, so the two
    // terms cannot be estimated against different game lists.
    let margin_and_gap = |draw: &[usize]| -> Option<(f64, f64)> {
        let a = perpos.mean_over(draw)?;
        let b = prefix.mean_over(draw)?;
        let c = perpos_b.mean_over(draw)?;
        Some((a - b, (a - c).abs()))
    };

    let confirm = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_gap(draw).map(|(m, g)| m - g)
    })?;
    let refute = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_gap(draw).map(|(m, g)| m + g)
    })?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    let (margin, gap) = margin_and_gap(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
    Ok(H14 {
        gamma,
        flip_perpos: perpos.total().mean().unwrap_or(f64::NAN),
        flip_prefix: prefix.total().mean().unwrap_or(f64::NAN),
        flip_perpos_b: perpos_b.total().mean().unwrap_or(f64::NAN),
        margin,
        gap,
        confirm,
        refute,
        positions: arms.positions(),
        games: arms.games(),
    })
}

/// One depth bucket, with the counts `§5.2` asks to be reported beside
/// any number drawn from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bucket {
    /// First ply in the bucket.
    pub low: usize,
    /// Last ply in the bucket.
    pub high: usize,
    /// Positions it holds.
    pub positions: usize,
    /// Games those positions came from.
    ///
    /// Reported because it is not the walk's game count: the deep bucket
    /// only draws from games long enough to reach ply 40, a smaller and
    /// self-selected set of clusters, and a reader who assumed otherwise
    /// would credit the deep figure with more independent evidence than
    /// it has.
    pub games: usize,
}

/// H15: does conditioning at every position also reach further down the
/// game?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H15 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// The opening bucket, shared by both arms.
    pub shallow: Bucket,
    /// The deep bucket, shared by both arms.
    pub deep: Bucket,
    /// `JS(deep) / JS(shallow)` for the per-position arm, on the sample
    /// as walked. Higher is a shallower decay.
    pub ratio_perpos: f64,
    /// The same for the prefix arm.
    pub ratio_prefix: f64,
    /// Interval on `ratio(perpos) - ratio(prefix)`. Confirmed when it
    /// excludes zero from above, refuted when from below.
    pub difference: Interval,
}

impl H15 {
    /// Whether the interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.difference.excludes_zero_from_above()
    }

    /// Whether the interval clears zero from below.
    pub fn refuted(&self) -> bool {
        self.difference.excludes_zero_from_below()
    }

    /// As [`H14::verdict`].
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H15 from the two arms whose attachment point differs.
///
/// The depth buckets come from the shared context window, so both arms
/// bucket identically; [`AlignedArms`] refuses a set whose arms were
/// walked at different context sizes.
///
/// # Errors
///
/// [`PERPOS`] or [`PREFIX`] is missing, the gamma was not swept, a
/// bucket caught no positions, or the resampling refused.
pub fn h15(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H15, StatError> {
    require_roles(
        arms,
        &[
            (PERPOS, CondEncoding::EveryPosition),
            (PREFIX, CondEncoding::Prefix),
        ],
    )?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let shallow_range = SHALLOW_BUCKET;
    let deep_range = deep_bucket(arms.ctx());

    let perpos_shallow = tally_in_bucket(arms, PERPOS, gamma_ix, shallow_range, widest_js)?;
    let perpos_deep = tally_in_bucket(arms, PERPOS, gamma_ix, deep_range, widest_js)?;
    let prefix_shallow = tally_in_bucket(arms, PREFIX, gamma_ix, shallow_range, widest_js)?;
    let prefix_deep = tally_in_bucket(arms, PREFIX, gamma_ix, deep_range, widest_js)?;

    if perpos_shallow.total().n == 0 {
        return Err(StatError::EmptyBucket {
            low: shallow_range.0,
            high: shallow_range.1.saturating_sub(1),
            role: "denominator",
        });
    }
    if perpos_deep.total().n == 0 {
        return Err(StatError::EmptyBucket {
            low: deep_range.0,
            high: deep_range.1.saturating_sub(1),
            role: "numerator",
        });
    }

    // A ratio needs a denominator that is not zero as well as one that
    // is not absent. Divergence is non-negative, so a shallow mean of
    // exactly zero means the bands were identical everywhere in the
    // opening — at which point the decay has nothing to decay from.
    let ratio = |deep: &ClusterTally, shallow: &ClusterTally, draw: &[usize]| -> Option<f64> {
        let denominator = shallow.mean_over(draw)?;
        if denominator <= 0.0 {
            return None;
        }
        Some(deep.mean_over(draw)? / denominator)
    };

    let difference = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        let a = ratio(&perpos_deep, &perpos_shallow, draw)?;
        let b = ratio(&prefix_deep, &prefix_shallow, draw)?;
        Some(a - b)
    })?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    Ok(H15 {
        gamma,
        shallow: Bucket {
            low: shallow_range.0,
            high: shallow_range.1.saturating_sub(1),
            positions: perpos_shallow.total().n,
            games: perpos_shallow.clusters_present(),
        },
        deep: Bucket {
            low: deep_range.0,
            high: deep_range.1.saturating_sub(1),
            positions: perpos_deep.total().n,
            games: perpos_deep.clusters_present(),
        },
        ratio_perpos: ratio(&perpos_deep, &perpos_shallow, &whole).unwrap_or(f64::NAN),
        ratio_prefix: ratio(&prefix_deep, &prefix_shallow, &whole).unwrap_or(f64::NAN),
        difference,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::records::{GammaRecord, PositionRecord, Walk, WalkHeader, FORMAT_VERSION};
    use crate::chess::CondEncoding;

    const GAMMAS: [f32; 1] = [1.0];
    const SEED: u64 = 0x0806_2026;

    fn header(positions: usize, games: usize) -> WalkHeader {
        WalkHeader {
            version: FORMAT_VERSION,
            ckpt: "ckpt".into(),
            holdout: "holdout-2026-05.pgn".into(),
            side: "White".into(),
            encoding: CondEncoding::Prefix,
            ctx: 128,
            bands: vec!["<lo>".into(), "<mid>".into(), "<hi>".into()],
            gammas: GAMMAS.to_vec(),
            positions,
            games,
        }
    }

    /// Label a walk as an arm: its own checkpoint, and the conditioning
    /// its role calls for.
    ///
    /// Both are checked now — two arms of one checkpoint would have a
    /// gap of zero in every draw, and a role swap would leave every
    /// number well-formed while reversing what the margin means.
    fn arm(name: &'static str, mut walk: Walk) -> (String, Walk) {
        walk.header.ckpt = format!("/root/ckpt/{name}/run.safetensors");
        walk.header.encoding = match name {
            PREFIX => CondEncoding::Prefix,
            _ => CondEncoding::EveryPosition,
        };
        (name.to_string(), walk)
    }

    /// A walk over `games` games of `per_game` positions each, where
    /// `flip(game, index)` decides the flip and `js` the divergence.
    fn walk(
        games: usize,
        per_game: usize,
        flip: impl Fn(usize, usize) -> bool,
        js: impl Fn(usize) -> f64,
    ) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ix in 0..per_game {
                // Plies 0, 2, 4, ... so both depth buckets are reachable
                // when `per_game` is large enough.
                let ply = ix * 2;
                records.push(PositionRecord {
                    game,
                    ply,
                    at: vec![GammaRecord {
                        flipped: flip(game, ix),
                        widest_js: js(ply),
                        legal_mass: 0.9,
                        top1: Some(vec![true, false, false]),
                    }],
                });
            }
        }
        Walk {
            header: header(records.len(), games),
            records,
        }
    }

    fn three_arms(perpos: Walk, prefix: Walk, perpos_b: Walk) -> AlignedArms {
        AlignedArms::new(vec![
            arm(PERPOS, perpos),
            arm(PREFIX, prefix),
            arm(PERPOS_B, perpos_b),
        ])
        .expect("the fixture arms share a position stream")
    }

    /// A rate of exactly one, from a single game, has nowhere to move
    /// under resampling: every draw picks that game and every draw sees
    /// the same positions.
    #[test]
    fn a_rate_of_one_from_a_single_game_gives_a_degenerate_interval() {
        let arms = AlignedArms::new(vec![arm(PERPOS, walk(1, 40, |_, _| true, |_| 0.01))]).unwrap();
        let interval = flip_rate(&arms, PERPOS, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(interval.point, 1.0);
        assert_eq!(interval.low, 1.0);
        assert_eq!(interval.high, 1.0);
        assert_eq!(interval.clusters, 1);
    }

    /// H14 on a fixture whose answer is fixed by construction: the
    /// per-position arm flips at every position, the prefix arm at none,
    /// and the second per-position run flips everywhere but one.
    ///
    /// So `M = 1` and `G = 1/96` exactly, and the two criteria are
    /// `1 - 1/96` and `1 + 1/96`.
    ///
    /// The second run used to be an exact copy of the first, which made
    /// `G` zero and both intervals degenerate. That fixture is now
    /// refused before it reaches here, and rightly: two arms with
    /// identical records are one checkpoint under two names, and a gap
    /// of zero is the floor having quietly disappeared rather than
    /// having been measured.
    #[test]
    fn h14_assembles_from_three_arms_with_a_known_answer() {
        let arms = three_arms(
            walk(8, 12, |_, _| true, |_| 0.02),
            walk(8, 12, |_, _| false, |_| 0.02),
            // One position of 96 where the replicate disagrees.
            walk(8, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
        );
        let result = h14(&arms, 1.0, DRAWS, SEED).unwrap();

        let expected_gap = 1.0 / 96.0;
        assert_eq!(result.flip_perpos, 1.0);
        assert_eq!(result.flip_prefix, 0.0);
        assert!((result.flip_perpos_b - 95.0 / 96.0).abs() < 1e-12);
        assert_eq!(result.margin, 1.0);
        assert!((result.gap - expected_gap).abs() < 1e-12);
        assert!((result.confirm.point - (1.0 - expected_gap)).abs() < 1e-12);
        assert!((result.refute.point - (1.0 + expected_gap)).abs() < 1e-12);
        assert!(result.confirm.low <= result.confirm.point);
        assert!(result.confirm.point <= result.confirm.high);
        assert!(result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "confirmed");
        assert_eq!(result.positions, 96);
        assert_eq!(result.games, 8);
        assert_eq!(result.confirm.seed, SEED);
    }

    /// The mirrored criteria, on the null the plan describes: the two
    /// arms have the same true rate, so `M` is zero, while `G` is a real
    /// positive gap between two runs.
    ///
    /// `M - G` is then negative — which is why it cannot be the refute
    /// test — and `M + G` is positive, so the refute branch correctly
    /// does not fire.
    #[test]
    fn under_the_null_the_refute_branch_does_not_fire() {
        // The per-position arm and the prefix arm flip on the same
        // *share* of positions — 6 of every 12 — while disagreeing about
        // which ones, so `M` is exactly zero without the two being the
        // same checkpoint. The second per-position run flips on 4 of
        // every 12, so `G` is a real positive gap.
        let arms = three_arms(
            walk(8, 12, |_, ix| ix % 2 == 0, |_| 0.02),
            walk(8, 12, |_, ix| ix % 2 == 1, |_| 0.02),
            walk(8, 12, |_, ix| ix % 3 == 0, |_| 0.02),
        );
        let result = h14(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!((result.margin - 0.0).abs() < 1e-12, "{}", result.margin);
        assert!(
            result.gap > 0.0,
            "the two runs should differ: {}",
            result.gap
        );
        assert!(
            result.confirm.point < 0.0,
            "M - G is negative under the null, which is why it is not the refute test"
        );
        assert!(result.refute.point > 0.0);
        assert!(
            !result.refuted(),
            "the null must not be reported as a refutation"
        );
        assert!(!result.confirmed());
        assert_eq!(result.verdict(), "undetermined");
    }

    /// A genuinely worse per-position arm does fire the refute branch:
    /// the prefix arm flips everywhere, the per-position arm nowhere,
    /// and the two per-position runs are one position apart, so
    /// `M = -1` and `G = 1/96` — worse by far more than the gap.
    #[test]
    fn an_arm_that_is_worse_by_more_than_the_gap_is_refuted() {
        let arms = three_arms(
            walk(8, 12, |_, _| false, |_| 0.02),
            walk(8, 12, |_, _| true, |_| 0.02),
            walk(8, 12, |g, ix| g == 0 && ix == 0, |_| 0.02),
        );
        let result = h14(&arms, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(result.margin, -1.0);
        assert!((result.gap - 1.0 / 96.0).abs() < 1e-12);
        assert!(result.refuted());
        assert!(!result.confirmed());
        assert_eq!(result.verdict(), "refuted");
    }

    /// H15 on a fixture where each arm's decay is fixed: the
    /// per-position arm holds its divergence with depth and the prefix
    /// arm loses nine tenths of it, so the ratio difference is
    /// `1.0 - 0.1`.
    #[test]
    fn h15_reads_the_decay_ratio_off_the_depth_buckets() {
        // 60 positions per game reaches ply 118, so both buckets fill.
        let flat = |_ply: usize| 0.04;
        let decaying = |ply: usize| if ply < 10 { 0.04 } else { 0.004 };
        let arms = AlignedArms::new(vec![
            arm(PERPOS, walk(6, 60, |_, _| true, flat)),
            arm(PREFIX, walk(6, 60, |_, _| true, decaying)),
        ])
        .unwrap();

        let result = h15(&arms, 1.0, DRAWS, SEED).unwrap();
        assert!((result.ratio_perpos - 1.0).abs() < 1e-12);
        assert!((result.ratio_prefix - 0.1).abs() < 1e-12);
        assert!((result.difference.point - 0.9).abs() < 1e-12);
        assert!(result.confirmed());
        assert_eq!(result.shallow.low, 0);
        assert_eq!(result.shallow.high, 9);
        // ctx 128, so the deep bucket runs to ply 125 and excludes the
        // windowed regime.
        assert_eq!(result.deep.low, 40);
        assert_eq!(result.deep.high, 125);
    }

    /// The bucket counts `§5.2` asks for, and the reason it asks: the
    /// deep bucket rests on fewer games than the walk did.
    #[test]
    fn the_deep_bucket_reports_the_games_that_actually_reached_it() {
        let mut short = walk(4, 6, |_, _| true, |_| 0.02);
        // A fifth game that runs long enough to reach ply 40, appended
        // to four that stop at ply 10.
        let long: Vec<PositionRecord> = (0..40)
            .map(|ix| PositionRecord {
                game: 4,
                ply: ix * 2,
                at: vec![GammaRecord {
                    flipped: true,
                    widest_js: 0.02,
                    legal_mass: 0.9,
                    top1: Some(vec![true, false, false]),
                }],
            })
            .collect();
        short.records.extend(long);
        short.header.positions = short.records.len();
        short.header.games = 5;

        // One position of difference, so the two are two arms rather
        // than one checkpoint under two names.
        let mut other = short.clone();
        other.records[0].at[0].flipped = false;

        let arms = AlignedArms::new(vec![arm(PERPOS, short), arm(PREFIX, other)]).unwrap();
        let result = h15(&arms, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(result.shallow.games, 5, "every game reaches the opening");
        assert_eq!(result.deep.games, 1, "only one game reaches ply 40");
        assert_eq!(arms.games(), 5);
    }

    /// A walk that never reached the deep bucket cannot answer a
    /// question about depth, and says so rather than dividing by an
    /// absent mean.
    #[test]
    fn a_walk_that_never_reached_the_deep_bucket_is_refused() {
        let arms = AlignedArms::new(vec![
            arm(PERPOS, walk(4, 4, |_, _| true, |_| 0.02)),
            arm(PREFIX, walk(4, 4, |_, _| false, |_| 0.02)),
        ])
        .unwrap();
        let err = h15(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::EmptyBucket {
                    low: 40,
                    high: 125,
                    role: "numerator"
                }
            ),
            "{err:?}"
        );
    }

    /// A role swap leaves every number well-formed and reverses what the
    /// margin means, so the encoding each checkpoint was actually
    /// trained under is checked against the role it has been given.
    #[test]
    fn an_arm_playing_the_wrong_role_is_refused() {
        // The first two files handed over the other way round, which is
        // what a swap on the command line produces: three distinct
        // checkpoints, positions that still line up, and a margin that
        // now reads backwards.
        let (_, prefix_file) = arm(PREFIX, walk(6, 12, |_, _| false, |_| 0.02));
        let (_, perpos_file) = arm(PERPOS, walk(6, 12, |_, _| true, |_| 0.02));
        let arms = AlignedArms::new(vec![
            (PERPOS.to_string(), prefix_file),
            (PREFIX.to_string(), perpos_file),
            arm(
                PERPOS_B,
                walk(6, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
            ),
        ])
        .unwrap();
        let err = h14(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::WrongEncodingForRole {
                    arm: PERPOS,
                    want: CondEncoding::EveryPosition,
                    found: CondEncoding::Prefix,
                }
            ),
            "{err:?}"
        );

        // And the same swap is available before anything is computed,
        // which is what a program that prints as it goes needs: the
        // validity gate is a difference against the prefix arm, so under
        // a swap it is computed against the wrong baseline and a full,
        // plausible table would reach the reader ahead of the refusal.
        assert!(check_roles(&arms).is_err());
    }

    /// The roles as they should be, so the check above is not passing
    /// for want of ever succeeding.
    #[test]
    fn correctly_assigned_roles_pass_the_up_front_check() {
        let arms = three_arms(
            walk(6, 12, |_, _| true, |_| 0.02),
            walk(6, 12, |_, _| false, |_| 0.02),
            walk(6, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
        );
        check_roles(&arms).expect("the canonical roles");
    }

    /// The validity gates are differences against the prefix arm, so the
    /// prefix arm against itself is identically zero in every draw and
    /// the gate is degenerate at pass.
    #[test]
    fn the_gates_are_differences_and_the_prefix_arm_is_its_own_baseline() {
        let arms = three_arms(
            walk(6, 12, |_, _| true, |_| 0.02),
            walk(6, 12, |_, _| false, |_| 0.02),
            walk(6, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
        );
        let top1 = gate_top1(&arms, PREFIX, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(top1.interval.point, 0.0);
        assert_eq!(top1.interval.low, 0.0);
        assert_eq!(top1.interval.high, 0.0);
        assert!(top1.passes());
        assert_eq!(top1.tolerance, TOP1_GATE);

        let mass = gate_legal_mass(&arms, PREFIX, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(mass.interval.point, 0.0);
        assert!(mass.passes());
        assert_eq!(mass.tolerance, LEGAL_MASS_GATE);
        assert_eq!(mass.verdict(), "pass");
    }

    /// The gates divide by the prefix arm, so they check it themselves
    /// rather than resting on the caller having run [`check_roles`]
    /// first.
    ///
    /// A per-position checkpoint in the [`PREFIX`] slot makes every gate
    /// a difference against the wrong baseline — the failure
    /// [`check_roles`] is documented from — and each figure it produces
    /// is well-formed.
    #[test]
    fn a_gate_refuses_a_baseline_that_is_not_the_prefix_arm() {
        // The two files handed over the other way round.
        let (_, prefix_file) = arm(PREFIX, walk(6, 12, |_, _| false, |_| 0.02));
        let (_, perpos_file) = arm(PERPOS, walk(6, 12, |_, _| true, |_| 0.02));
        let arms = AlignedArms::new(vec![
            (PERPOS.to_string(), prefix_file),
            (PREFIX.to_string(), perpos_file),
        ])
        .unwrap();

        for err in [
            gate_top1(&arms, PERPOS, 1.0, DRAWS, SEED).unwrap_err(),
            gate_legal_mass(&arms, PERPOS, 1.0, DRAWS, SEED).unwrap_err(),
        ] {
            assert!(
                matches!(
                    err,
                    StatError::WrongEncodingForRole {
                        arm: PREFIX,
                        want: CondEncoding::Prefix,
                        found: CondEncoding::EveryPosition,
                    }
                ),
                "{err:?}"
            );
        }
    }

    /// An arm that has drained its legal mass fails the third gate, and
    /// the quantity is relative so the tolerance reads as the fraction
    /// the gate names.
    #[test]
    fn an_arm_that_lost_its_legal_mass_is_outside_the_gate() {
        let mut drained = walk(6, 12, |_, _| true, |_| 0.02);
        for record in drained.records.iter_mut() {
            for at in record.at.iter_mut() {
                // Half the prefix arm's 0.9, so the relative difference
                // is -0.5 against a tolerance of 0.20.
                at.legal_mass = 0.45;
            }
        }
        let arms = three_arms(
            drained,
            walk(6, 12, |_, _| false, |_| 0.02),
            walk(6, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
        );
        let mass = gate_legal_mass(&arms, PERPOS, 1.0, DRAWS, SEED).unwrap();
        assert!((mass.interval.point - -0.5).abs() < 1e-12, "{mass:?}");
        assert!(!mass.passes());
        assert_eq!(mass.verdict(), "outside");
    }

    /// A missing arm is named rather than silently producing a statistic
    /// over whatever was supplied.
    #[test]
    fn h14_without_all_three_arms_is_refused() {
        let arms = AlignedArms::new(vec![
            arm(PERPOS, walk(4, 4, |_, _| true, |_| 0.02)),
            arm(PREFIX, walk(4, 4, |_, _| false, |_| 0.02)),
        ])
        .unwrap();
        let err = h14(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(err, StatError::Align(AlignError::UnknownArm { .. })),
            "{err:?}"
        );
    }

    /// Positions whose played move is not in the vocabulary carry no
    /// top-1 value, and are dropped rather than counted as misses.
    #[test]
    fn unscoreable_positions_do_not_drag_the_top1_figure_down() {
        let mut w = walk(3, 10, |_, _| false, |_| 0.02);
        for (ix, record) in w.records.iter_mut().enumerate() {
            for at in record.at.iter_mut() {
                at.top1 = if ix % 2 == 0 {
                    Some(vec![true, true, true])
                } else {
                    None
                };
            }
        }
        let arms = AlignedArms::new(vec![arm(PERPOS, w)]).unwrap();
        let interval = top1_match(&arms, PERPOS, 1.0, 100, SEED).unwrap();
        assert_eq!(
            interval.point, 1.0,
            "the scoreable half all matched, so the mean is one"
        );
    }

    /// Both intervals of one H14 come from the same draws, in the same
    /// order, so a relation that holds draw by draw survives into the
    /// **interval endpoints**.
    ///
    /// The fixture is built so that the same-arm gap is the same number
    /// in every resample: every game has twelve positions, and the
    /// second per-position run flips on exactly one fewer of them per
    /// game, so `G = 1/12` whichever games a draw picks. The margin does
    /// vary, because the per-position arm's rate differs game to game
    /// while the prefix arm's does not — so the intervals have width and
    /// the equality below is not vacuous.
    ///
    /// With `G` constant, `refute = confirm + 2G` at every draw, and the
    /// percentiles inherit it **only if the two are percentiles of the
    /// same draws**. Bootstrapped separately they would land `2G` apart
    /// give or take resampling noise, not to twelve decimal places.
    ///
    /// The point estimates are deliberately not what is asserted here.
    /// `refute.point - confirm.point == 2G` is arithmetic on one call to
    /// `margin_and_gap` over the whole sample, with no resampling in it
    /// at all; it would hold just as exactly if the two intervals came
    /// from unrelated seeds, so it is evidence for nothing.
    #[test]
    fn the_two_h14_intervals_are_two_readings_of_one_set_of_draws() {
        let arms = three_arms(
            // Three to seven flips of twelve, varying by game.
            walk(8, 12, |g, ix| ix < (g % 5) + 3, |_| 0.02),
            // Three of twelve in every game, so the margin moves only
            // with the per-position arm.
            walk(8, 12, |_, ix| ix < 3, |_| 0.02),
            // One fewer than the first arm, in every game.
            walk(8, 12, |g, ix| ix < (g % 5) + 2, |_| 0.02),
        );
        let result = h14(&arms, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(result.confirm.seed, result.refute.seed);
        assert_eq!(result.confirm.clusters, result.refute.clusters);
        assert_eq!(result.confirm.draws, result.refute.draws);
        assert!(
            (result.gap - 1.0 / 12.0).abs() < 1e-12,
            "the fixture's gap should be one position in twelve: {}",
            result.gap
        );
        assert!(
            result.confirm.low < result.confirm.high,
            "an interval of zero width would make the endpoint equality vacuous: {:?}",
            result.confirm
        );
        for (refute_end, confirm_end, which) in [
            (result.refute.low, result.confirm.low, "low"),
            (result.refute.high, result.confirm.high, "high"),
        ] {
            assert!(
                (refute_end - (confirm_end + 2.0 * result.gap)).abs() < 1e-12,
                "the {which} ends should sit exactly 2G apart, which they only do if the \
                 two intervals rank the same draws: {refute_end} vs {confirm_end}"
            );
        }
    }

    /// The machinery has to work at the scale it will actually be used
    /// at, and the plan needs to know what that costs before it decides
    /// whether the measurement runs locally or on a rented machine.
    ///
    /// So this is the real shape: three arms over 3,000 positions from
    /// 95 games, seven gammas, 2,000 draws — written to disk, read back,
    /// aligned, and put through both hypotheses plus the three per-arm
    /// figures the validity gate is read off. The cluster sizes are
    /// uneven, as games are, so the sum-over-sum weighting is exercised
    /// rather than the equal-size shortcut.
    ///
    /// Timings go to stderr; `cargo test -- plan_scale --nocapture`
    /// shows them.
    #[test]
    fn the_whole_statistic_runs_at_plan_scale() {
        use std::time::Instant;
        use tempfile::TempDir;

        const GAMES: usize = 95;
        const POSITIONS: usize = 3_000;
        const SWEEP: [f32; 7] = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

        // 95 games of 31 or 32 positions each, summing to 3,000. A
        // position is a White ply, so game `g` runs to ply `2n - 2`;
        // 31-32 positions reaches ply 60-62, which fills the shallow
        // bucket from every game and the deep bucket from all of them.
        let sizes: Vec<usize> = (0..GAMES)
            .map(|g| if g < POSITIONS % GAMES { 32 } else { 31 })
            .collect();
        assert_eq!(sizes.iter().sum::<usize>(), POSITIONS);

        let build = |role: &'static str, flip_every: usize, deep_js: f64| -> Walk {
            let mut records = Vec::with_capacity(POSITIONS);
            for (game, size) in sizes.iter().enumerate() {
                for ix in 0..*size {
                    let ply = ix * 2;
                    let js = if ply < 10 { 0.027 } else { deep_js };
                    records.push(PositionRecord {
                        game,
                        ply,
                        at: SWEEP
                            .iter()
                            .map(|_| GammaRecord {
                                flipped: (game + ix).is_multiple_of(flip_every),
                                widest_js: js,
                                legal_mass: 0.83,
                                top1: Some(vec![true, false, false]),
                            })
                            .collect(),
                    })
                }
            }
            let mut header = header(records.len(), GAMES);
            header.gammas = SWEEP.to_vec();
            // Distinct checkpoints and role-correct conditioning: both
            // are checked, and a fixture that skipped either would be
            // testing a set of arms the real thing refuses.
            let (_, walk) = arm(role, Walk { header, records });
            walk
        };

        let tmp = TempDir::new().expect("a temp dir");
        let paths: Vec<(String, std::path::PathBuf)> = [PERPOS, PREFIX, PERPOS_B]
            .iter()
            .map(|role| (role.to_string(), tmp.path().join(format!("{role}.jsonl"))))
            .collect();

        let t_write = Instant::now();
        build(PERPOS, 7, 0.0021)
            .write_jsonl(&paths[0].1)
            .expect("write");
        build(PREFIX, 9, 0.0002)
            .write_jsonl(&paths[1].1)
            .expect("write");
        build(PERPOS_B, 8, 0.0021)
            .write_jsonl(&paths[2].1)
            .expect("write");
        let write_elapsed = t_write.elapsed();

        let t_read = Instant::now();
        let arms = AlignedArms::read(&paths).expect("three arms over one stream");
        let read_elapsed = t_read.elapsed();
        assert_eq!(arms.positions(), POSITIONS);
        assert_eq!(arms.games(), GAMES);

        let t_stats = Instant::now();
        for (role, _) in &paths {
            for interval in [
                flip_rate(&arms, role, 1.0, DRAWS, SEED).expect("a flip rate"),
                legal_mass(&arms, role, 1.0, DRAWS, SEED).expect("a legal mass"),
                top1_match(&arms, role, 1.0, DRAWS, SEED).expect("a top-1 match"),
                gate_top1(&arms, role, 1.0, DRAWS, SEED)
                    .expect("the top-1 gate")
                    .interval,
                gate_legal_mass(&arms, role, 1.0, DRAWS, SEED)
                    .expect("the legal-mass gate")
                    .interval,
            ] {
                assert!(interval.point.is_finite());
                assert_eq!(interval.undefined_draws, 0);
            }
        }
        let h14 = h14(&arms, 1.0, DRAWS, SEED).expect("H14");
        let h15 = h15(&arms, 1.0, DRAWS, SEED).expect("H15");
        let stats_elapsed = t_stats.elapsed();

        assert!(h14.confirm.low.is_finite() && h14.confirm.high.is_finite());
        assert!(h14.confirm.low <= h14.confirm.point && h14.confirm.point <= h14.confirm.high);
        assert_eq!(h14.confirm.clusters, GAMES);
        assert_eq!(h14.confirm.draws, DRAWS);
        assert_eq!(h15.shallow.positions, GAMES * 5);
        assert_eq!(h15.shallow.games, GAMES);
        assert!(h15.deep.positions > 0);
        assert!(h15.difference.low.is_finite());

        let bytes = std::fs::metadata(&paths[0].1).map(|m| m.len()).unwrap_or(0);
        let gammas = SWEEP.len();
        let total = t_write.elapsed();
        eprintln!(
            "plan scale: {POSITIONS} positions from {GAMES} games, {gammas} gammas, {DRAWS} draws\n  \
             write 3 arms   {write_elapsed:.1?}  ({bytes} bytes each)\n  \
             read + align   {read_elapsed:.1?}\n  \
             statistics     {stats_elapsed:.1?}  (15 per-arm intervals + H14 + H15 = 18 bootstraps)\n  \
             total          {total:.1?}"
        );
    }

    /// Reproducibility, end to end: the same seed gives the same
    /// interval, a different one does not.
    #[test]
    fn the_seed_reproduces_the_whole_statistic() {
        let build = || {
            three_arms(
                walk(9, 11, |g, ix| (g + ix) % 3 == 0, |_| 0.02),
                walk(9, 11, |g, ix| (g + ix) % 4 == 0, |_| 0.02),
                walk(9, 11, |g, ix| (g + ix) % 5 == 0, |_| 0.02),
            )
        };
        let a = h14(&build(), 1.0, 500, SEED).unwrap();
        let b = h14(&build(), 1.0, 500, SEED).unwrap();
        let c = h14(&build(), 1.0, 500, SEED + 1).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.confirm, c.confirm);
    }
}
