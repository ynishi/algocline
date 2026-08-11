//! Error bars for a statistic whose observations arrive in correlated
//! groups.
//!
//! # Why the groups matter
//!
//! The measurements this exists for are per-position: a flip rate over
//! 3,000 positions, a divergence meaned over a depth bucket. Those
//! 3,000 positions come from roughly 95 games, and consecutive
//! positions within one game share an opening, a pair of players and
//! most of a board. They are not 3,000 independent draws, and treating
//! them as though they were reports a precision the sample does not
//! have — the naive interval narrows as `1/sqrt(positions)` when the
//! information in it only grows as `1/sqrt(games)`.
//!
//! So the unit resampled here is the **cluster** (a game), not the
//! observation (a position). A draw picks `G` games with replacement
//! out of the `G` that were walked, and every quantity in that draw is
//! recomputed over exactly those games — including the ones that appear
//! twice and excluding the ones that did not come up.
//!
//! # Why every term shares one draw
//!
//! The statistics this feeds are differences: one arm's rate minus
//! another's, one arm's decay ratio minus another's. A difference of two
//! separately bootstrapped quantities has no joint distribution — the
//! two intervals would each be correct about their own quantity and say
//! nothing about the gap between them, which is the thing being judged.
//!
//! [`cluster_bootstrap`] therefore hands the statistic **the draw**, a
//! list of cluster indices, and lets it compute every term it needs from
//! that one list. The property this buys, stated exactly: the function
//! supplies one draw and no second one, so every term written against
//! the argument shares it. That is weaker than "a second resample is
//! unreachable" — an earlier version of this paragraph claimed the
//! stronger thing and it was false, since `cluster_bootstrap` is
//! re-entrant. The resampler is crate-private, which narrows the gap
//! from the other side: nothing outside this crate can construct one.
//! Inside it, [`crate::chess::steerability`] still could — the guard is
//! a boundary, not a proof, and the property above is what actually
//! holds.
//!
//! # Reproducibility
//!
//! The resampler is a SplitMix64 carried in this file rather than a
//! generator from a dependency. `rand`'s `StdRng` is explicitly allowed
//! to change algorithm between releases, so a run reproduced after a
//! `cargo update` would draw a different sample from the same seed and
//! quietly report a different interval. A generator written out here
//! makes reproducibility a property of this code.
//!
//! The seed travels on [`Interval::seed`] so it can be recorded next to
//! the number it produced.

use thiserror::Error;

/// Confidence level every interval here is reported at.
///
/// Fixed rather than taken as a parameter. A level chosen per call is a
/// level that can be widened after the interval has been seen, which is
/// the one thing a pre-registered criterion exists to prevent.
pub const CONFIDENCE: f64 = 0.95;

/// A deterministic uniform source.
///
/// SplitMix64 — the finaliser Java's `SplittableRandom` uses. Chosen for
/// being short enough to carry (a fixed additive step and two multiply-
/// xorshift rounds), which is what makes a run reproducible against this
/// source rather than against whichever algorithm a dependency shipped
/// that month. Statistical quality is ample for picking indices out of a
/// list of games; nothing here is cryptographic.
///
/// Crate-private on purpose, so that [`cluster_bootstrap`] is the only
/// way in from outside this crate and a statistic written there cannot
/// construct a second stream to bound one of its terms against a
/// different set of games. Within the crate it is reachable, so that is
/// a narrowed surface rather than an impossibility — see the module
/// header for the property that does hold unconditionally.
#[derive(Debug, Clone)]
pub(crate) struct Resampler {
    state: u64,
}

impl Resampler {
    /// Start a stream from `seed`.
    ///
    /// The same seed always yields the same stream, on any host and any
    /// build of this crate.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        // Odd increment, so the additive step is a full-period walk over
        // the 64-bit state regardless of where it started.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..n`, or `None` for an empty range.
    ///
    /// Lemire's multiply-shift with rejection: the plain `% n` is biased
    /// towards small indices whenever `n` does not divide `2^64`, and
    /// with 95 games the bias is far below the noise here — but it costs
    /// one comparison to remove and a biased resampler is not something
    /// a reader of the interval can check.
    pub(crate) fn below(&mut self, n: usize) -> Option<usize> {
        let n = u64::try_from(n).ok().filter(|n| *n > 0)?;
        let mut product = u128::from(self.next_u64()) * u128::from(n);
        let mut low = product as u64;
        if low < n {
            // Reject the leftover window at the bottom of the range,
            // which is the part that has one extra representative.
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(n);
                low = product as u64;
            }
        }
        usize::try_from(product >> 64).ok()
    }

    /// Fill `into` with `into.len()` cluster indices drawn from
    /// `0..clusters`, with replacement.
    ///
    /// Returns `false` and leaves `into` untouched when `clusters` is
    /// zero, which is the only way the draw can fail.
    pub(crate) fn draw_into(&mut self, clusters: usize, into: &mut [usize]) -> bool {
        if clusters == 0 {
            return false;
        }
        for slot in into.iter_mut() {
            match self.below(clusters) {
                Some(ix) => *slot = ix,
                // Unreachable while `clusters > 0`: `below` only
                // returns `None` on an empty range or a width that does
                // not fit a `usize`, and `clusters` is already one.
                None => return false,
            }
        }
        true
    }
}

/// The sum of one quantity over some set of observations, and how many
/// there were.
///
/// Carried as a pair rather than as a mean so that sums from several
/// clusters can be added: a mean over a resample is the sum of the drawn
/// clusters' sums over the sum of their counts, which is not the average
/// of their means unless every cluster is the same size — and games are
/// not.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tally {
    /// Total of the values seen.
    pub sum: f64,
    /// How many values contributed.
    pub n: usize,
}

impl Tally {
    /// Fold another tally in.
    pub fn merge(&mut self, other: Tally) {
        self.sum += other.sum;
        self.n += other.n;
    }

    /// Mean of the values, or `None` if there were none.
    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}

/// One quantity, tallied cluster by cluster.
///
/// Built once over the whole sample; a bootstrap draw then reads it by
/// cluster index, so a draw costs `O(clusters)` rather than
/// `O(observations)`. With 2,000 draws over 95 games that is the
/// difference between 190 thousand additions and 6 million.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterTally {
    per_cluster: Vec<Tally>,
}

impl ClusterTally {
    /// An empty tally over `clusters` clusters.
    pub fn new(clusters: usize) -> Self {
        Self {
            per_cluster: vec![Tally::default(); clusters],
        }
    }

    /// How many clusters this was built over.
    pub fn clusters(&self) -> usize {
        self.per_cluster.len()
    }

    /// Add one observation to `cluster`.
    ///
    /// # Errors
    ///
    /// `cluster` is not one of the clusters this was built over. An
    /// error rather than a growing `Vec`, because the cluster count is
    /// fixed by the sample and an index past it means the caller is
    /// tallying against a different sample than it thinks.
    pub fn push(&mut self, cluster: usize, value: f64) -> Result<(), BootstrapError> {
        match self.per_cluster.get_mut(cluster) {
            Some(tally) => {
                tally.sum += value;
                tally.n += 1;
                Ok(())
            }
            None => Err(BootstrapError::ClusterOutOfRange {
                cluster,
                clusters: self.per_cluster.len(),
            }),
        }
    }

    /// Totals over a draw, counting a cluster once per appearance.
    ///
    /// `None` if any index in `draw` is outside `0..clusters()`.
    ///
    /// # Precondition, and why it is not merely skipped
    ///
    /// Every index must name a cluster this was built over.
    /// [`ClusterTally::push`] refuses one that does not, because an
    /// index past the end means the caller is tallying against a
    /// different sample than it thinks — and the same is true here, with
    /// a worse consequence. This is reached on every draw, so dropping
    /// an unknown index would quietly remove observations from the
    /// resample, and fewer observations is a **narrower interval**: the
    /// mistake would make the number look more precise rather than make
    /// anything fail.
    ///
    /// So the whole draw goes undefined instead. A debug build trips the
    /// assertion; a release build reports every draw as undefined, which
    /// [`cluster_bootstrap`] turns into
    /// [`BootstrapError::EveryDrawUndefined`]. Neither can be mistaken
    /// for a measurement.
    pub fn over(&self, draw: &[usize]) -> Option<Tally> {
        let mut total = Tally::default();
        for &cluster in draw {
            let Some(tally) = self.per_cluster.get(cluster) else {
                debug_assert!(
                    false,
                    "cluster {cluster} is outside the {} this tally was built over",
                    self.per_cluster.len()
                );
                return None;
            };
            total.merge(*tally);
        }
        Some(total)
    }

    /// Mean over a draw.
    ///
    /// `None` when the draw caught no observations, or when it names a
    /// cluster this was not built over — see [`ClusterTally::over`].
    pub fn mean_over(&self, draw: &[usize]) -> Option<f64> {
        self.over(draw)?.mean()
    }

    /// Totals over every cluster.
    pub fn total(&self) -> Tally {
        let mut total = Tally::default();
        for tally in &self.per_cluster {
            total.merge(*tally);
        }
        total
    }

    /// How many distinct clusters contributed at least one observation.
    ///
    /// This is the figure `§5.2` asks to be reported beside a number:
    /// the depth ratio's deep bucket only draws from games long enough
    /// to reach ply 40, so its denominator rests on fewer games than the
    /// position count suggests.
    pub fn clusters_present(&self) -> usize {
        self.per_cluster.iter().filter(|t| t.n > 0).count()
    }
}

/// A point estimate with a percentile interval around it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// The statistic on the sample as walked, with no resampling.
    pub point: f64,
    /// Lower end of the [`CONFIDENCE`] percentile interval.
    pub low: f64,
    /// Upper end.
    pub high: f64,
    /// Draws that produced a usable value, and so went into the
    /// percentiles.
    pub draws: usize,
    /// Draws whose statistic was undefined or non-finite, and so were
    /// left out of the percentiles.
    ///
    /// Reported rather than swallowed: dropping draws biases the
    /// interval, and a reader has to be able to see how many were
    /// dropped before believing it.
    pub undefined_draws: usize,
    /// Clusters resampled per draw.
    pub clusters: usize,
    /// Seed the draws came from.
    pub seed: u64,
}

impl Interval {
    /// Whether the whole interval lies above zero.
    ///
    /// This is `§4`'s "confirmed" test, spelled once here so that the
    /// two hypotheses cannot disagree about what it means.
    pub fn excludes_zero_from_above(&self) -> bool {
        self.low > 0.0
    }

    /// Whether the whole interval lies below zero — `§4`'s "refuted".
    pub fn excludes_zero_from_below(&self) -> bool {
        self.high < 0.0
    }
}

impl std::fmt::Display for Interval {
    /// Routed through [`std::fmt::Formatter::pad`] rather than `write!`
    /// so that a caller's width specifier is honoured.
    ///
    /// `write!` ignores `f.width()`, which is not a cosmetic detail in a
    /// table: a header row written with `{:<28}` would line up while
    /// every data row beneath it ran to its natural length, and the
    /// columns a reader compares down would not be columns.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rendered = format!("{:+.6} [{:+.6}, {:+.6}]", self.point, self.low, self.high);
        f.pad(&rendered)
    }
}

/// Why a bootstrap could not produce an interval.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BootstrapError {
    /// Nothing to resample. A bootstrap over zero clusters would return
    /// an interval of width zero around a statistic computed from no
    /// data.
    #[error("a cluster bootstrap needs at least one cluster to resample")]
    NoClusters,

    /// Zero draws were asked for, so there are no percentiles to take.
    #[error("a cluster bootstrap needs at least one draw")]
    NoDraws,

    /// The statistic is not defined on the sample as walked, so there is
    /// no point estimate to put an interval around. A depth ratio whose
    /// denominator bucket is empty lands here.
    #[error("the statistic is undefined on the sample as walked, so there is nothing to bound")]
    UndefinedOnWholeSample,

    /// Every draw was undefined. Distinct from
    /// [`BootstrapError::UndefinedOnWholeSample`]: the statistic exists
    /// on the full sample and fell apart on every resample of it, which
    /// means it rests on so few clusters that resampling destroys it.
    #[error(
        "all {draws} draws were undefined even though the statistic exists on the whole sample; \
         it rests on too few clusters to resample"
    )]
    EveryDrawUndefined {
        /// How many draws were attempted.
        draws: usize,
    },

    /// A tally was handed an observation for a cluster it was not built
    /// over.
    #[error("cluster {cluster} is outside the {clusters} cluster(s) this tally was built over")]
    ClusterOutOfRange {
        /// Index the caller supplied.
        cluster: usize,
        /// How many clusters the tally holds.
        clusters: usize,
    },
}

/// Resample `clusters` clusters `draws` times and bound `stat`.
///
/// `stat` receives one draw — a list of cluster indices, with
/// repetitions — and returns its value on exactly those clusters, or
/// `None` where the draw leaves it undefined. Every term of a difference
/// must be computed from that same list; see the module header for why
/// bootstrapping the terms separately answers a different question.
///
/// The point estimate is `stat` on `0..clusters`, each cluster once,
/// which is the sample as walked.
///
/// # Errors
///
/// - `clusters` or `draws` is zero.
/// - `stat` is undefined on the whole sample, or on every draw of it.
pub fn cluster_bootstrap(
    clusters: usize,
    draws: usize,
    seed: u64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<Interval, BootstrapError> {
    let (point, mut values, undefined_draws) = resample(clusters, draws, seed, stat)?;
    let (low, high) = percentile_interval(&mut values);
    Ok(Interval {
        point,
        low,
        high,
        draws: values.len(),
        undefined_draws,
        clusters,
        seed,
    })
}

/// [`cluster_bootstrap`] over several independent strata, resampled
/// together.
///
/// `strata` holds one cluster count per stratum, and every draw
/// resamples **each stratum from its own clusters** — `stat` receives
/// one index list per stratum, in order. This is what a statistic
/// pooled across separate position streams needs: the streams share no
/// games, so a single pooled cluster space would let a draw omit a
/// stratum entirely and would treat clusters from different
/// populations as exchangeable, and both misstate the sampling the
/// interval claims to describe.
///
/// The point estimate is `stat` on every stratum's whole sample. The
/// percentile machinery, the seed discipline and the undefined-draw
/// accounting are [`cluster_bootstrap`]'s; [`Interval::clusters`]
/// reports the total across strata.
///
/// # Errors
///
/// - `strata` is empty, any stratum has zero clusters, or `draws` is
///   zero.
/// - `stat` is undefined on the whole sample, or on every draw of it.
pub fn stratified_cluster_bootstrap(
    strata: &[usize],
    draws: usize,
    seed: u64,
    stat: impl Fn(&[Vec<usize>]) -> Option<f64>,
) -> Result<Interval, BootstrapError> {
    if strata.is_empty() || strata.contains(&0) {
        return Err(BootstrapError::NoClusters);
    }
    if draws == 0 {
        return Err(BootstrapError::NoDraws);
    }

    let whole: Vec<Vec<usize>> = strata.iter().map(|n| (0..*n).collect()).collect();
    let point = stat(&whole)
        .filter(|v| v.is_finite())
        .ok_or(BootstrapError::UndefinedOnWholeSample)?;

    let mut resampler = Resampler::new(seed);
    let mut draw: Vec<Vec<usize>> = strata.iter().map(|n| vec![0usize; *n]).collect();
    let mut values: Vec<f64> = Vec::with_capacity(draws);
    let mut undefined_draws = 0usize;
    for _ in 0..draws {
        for (stratum, clusters) in draw.iter_mut().zip(strata) {
            if !resampler.draw_into(*clusters, stratum) {
                // Only reachable with zero clusters, refused above.
                return Err(BootstrapError::NoClusters);
            }
        }
        match stat(&draw).filter(|v| v.is_finite()) {
            Some(value) => values.push(value),
            None => undefined_draws += 1,
        }
    }
    if values.is_empty() {
        return Err(BootstrapError::EveryDrawUndefined { draws });
    }
    let (low, high) = percentile_interval(&mut values);
    Ok(Interval {
        point,
        low,
        high,
        draws: values.len(),
        undefined_draws,
        clusters: strata.iter().sum(),
        seed,
    })
}

/// An interval and the one-sided levels the same draws also carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedInterval {
    /// The percentile interval, exactly as [`cluster_bootstrap`] would
    /// have produced it from these draws.
    pub interval: Interval,
    /// Share of the usable draws that landed **at or above** zero: the
    /// one-sided achieved level for the claim that the quantity is
    /// below zero.
    ///
    /// Draws at exactly zero count here, and in
    /// [`Self::p_above_zero`] as well. The two therefore need not sum
    /// to one, and each is the conservative reading of its own claim.
    pub p_below_zero: f64,
    /// Share at or below zero: the level for the claim that the
    /// quantity is above zero.
    pub p_above_zero: f64,
}

/// [`cluster_bootstrap`], with the one-sided levels kept.
///
/// The interval is the same object the plain call returns, from the
/// same draws in the same order. What is added is the pair of levels,
/// which exist for one reason: a multiplicity correction has to tighten
/// the level per hypothesis, and [`CONFIDENCE`] is deliberately not a
/// parameter — a level that could be chosen per call is a level that
/// could be widened after the interval had been seen. Reporting the
/// achieved level instead leaves the fixed one fixed and lets the
/// caller compare it against whatever threshold its correction hands
/// down.
///
/// At the uncorrected level the two agree, up to the rounding of a
/// nearest-rank endpoint: a 95% percentile interval lying wholly below
/// zero is the statement that under 2.5% of the draws reached zero.
///
/// # Errors
///
/// As [`cluster_bootstrap`].
pub fn cluster_bootstrap_signed(
    clusters: usize,
    draws: usize,
    seed: u64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<SignedInterval, BootstrapError> {
    let (point, mut values, undefined_draws) = resample(clusters, draws, seed, stat)?;
    // Counted before the sort, though the count does not depend on the
    // order; taking them here keeps the two readings of one resample
    // visibly the same list.
    let usable = values.len() as f64;
    let at_or_above = values.iter().filter(|v| **v >= 0.0).count() as f64;
    let at_or_below = values.iter().filter(|v| **v <= 0.0).count() as f64;
    let (low, high) = percentile_interval(&mut values);
    Ok(SignedInterval {
        interval: Interval {
            point,
            low,
            high,
            draws: values.len(),
            undefined_draws,
            clusters,
            seed,
        },
        p_below_zero: at_or_above / usable,
        p_above_zero: at_or_below / usable,
    })
}

/// The draw loop both entry points share: the point estimate, the
/// usable draw values, and how many draws fell apart.
///
/// # Errors
///
/// - `clusters` or `draws` is zero.
/// - `stat` is undefined on the whole sample, or on every draw of it.
fn resample(
    clusters: usize,
    draws: usize,
    seed: u64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<(f64, Vec<f64>, usize), BootstrapError> {
    if clusters == 0 {
        return Err(BootstrapError::NoClusters);
    }
    if draws == 0 {
        return Err(BootstrapError::NoDraws);
    }

    let whole: Vec<usize> = (0..clusters).collect();
    let point = stat(&whole)
        .filter(|v| v.is_finite())
        .ok_or(BootstrapError::UndefinedOnWholeSample)?;

    let mut resampler = Resampler::new(seed);
    let mut draw = vec![0usize; clusters];
    let mut values: Vec<f64> = Vec::with_capacity(draws);
    let mut undefined_draws = 0usize;
    for _ in 0..draws {
        if !resampler.draw_into(clusters, &mut draw) {
            // Only reachable with zero clusters, which is refused above.
            return Err(BootstrapError::NoClusters);
        }
        // A non-finite value is undefined in every sense a reader cares
        // about — a ratio over an empty bucket arrives as an infinity
        // rather than as a `None` — so the two are counted together.
        match stat(&draw).filter(|v| v.is_finite()) {
            Some(value) => values.push(value),
            None => undefined_draws += 1,
        }
    }
    if values.is_empty() {
        return Err(BootstrapError::EveryDrawUndefined { draws });
    }
    Ok((point, values, undefined_draws))
}

/// The [`CONFIDENCE`] percentile interval of `values`, which is sorted
/// in place.
///
/// Nearest-rank percentiles: the low end is the value at rank
/// `floor(alpha/2 * B)` and the high end at `ceil((1 - alpha/2) * B) - 1`,
/// which for `B = 2000` at 95% is the 50th and the 1949th — the
/// conventional 2.5 / 97.5 percentile bootstrap.
///
/// Not public: it takes `&mut` and hands back no record of the seed or
/// the draw count, so a caller reaching it directly would produce an
/// interval nothing can reproduce. [`cluster_bootstrap`] is the way in.
fn percentile_interval(values: &mut [f64]) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let last = values.len().saturating_sub(1);
    let alpha = 1.0 - CONFIDENCE;
    let low_rank = (alpha / 2.0 * values.len() as f64).floor() as usize;
    let high_rank = ((1.0 - alpha / 2.0) * values.len() as f64).ceil() as usize;
    let low_ix = low_rank.min(last);
    let high_ix = high_rank.saturating_sub(1).min(last);
    // Both indices are clamped into range, so the gets cannot miss; the
    // fallbacks keep this free of indexing panics regardless.
    let low = values.get(low_ix).copied().unwrap_or(f64::NAN);
    let high = values.get(high_ix).copied().unwrap_or(f64::NAN);
    (low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tally where cluster `g` holds `n` copies of `value(g)`.
    fn blocked(values: &[f64], per_cluster: usize) -> ClusterTally {
        let mut tally = ClusterTally::new(values.len());
        for (cluster, value) in values.iter().enumerate() {
            for _ in 0..per_cluster {
                tally.push(cluster, *value).unwrap();
            }
        }
        tally
    }

    /// The property the whole file rests on: a seed names a sample.
    #[test]
    fn the_same_seed_draws_the_same_clusters() {
        let mut a = Resampler::new(0x0806_2026);
        let mut b = Resampler::new(0x0806_2026);
        let mut left = vec![0usize; 64];
        let mut right = vec![0usize; 64];
        for _ in 0..100 {
            assert!(a.draw_into(95, &mut left));
            assert!(b.draw_into(95, &mut right));
            assert_eq!(left, right);
        }
        // And a different seed does not, which is what makes the
        // agreement above evidence of anything.
        let mut other = Resampler::new(0x0806_2027);
        assert!(other.draw_into(95, &mut right));
        let mut first = vec![0usize; 64];
        let mut fresh = Resampler::new(0x0806_2026);
        assert!(fresh.draw_into(95, &mut first));
        assert_ne!(first, right);
    }

    /// Whole bootstraps agree too, not just the raw stream.
    #[test]
    fn the_same_seed_yields_the_same_interval() {
        let tally = blocked(&[0.0, 1.0, 0.25, 0.75, 0.5], 7);
        let run = |seed| {
            cluster_bootstrap(tally.clusters(), 500, seed, |draw| tally.mean_over(draw)).unwrap()
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
    }

    /// Every draw resamples the same one cluster, every draw sees the
    /// same values, so the interval has nowhere to go.
    #[test]
    fn a_single_cluster_of_ones_gives_a_degenerate_interval() {
        let mut tally = ClusterTally::new(1);
        for _ in 0..500 {
            tally.push(0, 1.0).unwrap();
        }
        let interval =
            cluster_bootstrap(1, 2000, 0x0806_2026, |draw| tally.mean_over(draw)).unwrap();
        assert_eq!(interval.point, 1.0);
        assert_eq!(interval.low, 1.0);
        assert_eq!(interval.high, 1.0);
        assert_eq!(interval.undefined_draws, 0);
        assert!(!interval.excludes_zero_from_below());
        assert!(interval.excludes_zero_from_above());
    }

    /// The reason this module exists.
    ///
    /// The same 500 numbers are bootstrapped two ways: resampling the
    /// 50 clusters they arrived in, and resampling the 500 values as
    /// though they were independent. Within a cluster every value is
    /// identical, so the cluster carries one number's worth of
    /// information and not ten — and the cluster interval has to be the
    /// wider of the two. If it is not, either this implementation or the
    /// argument for it is wrong.
    #[test]
    fn correlated_observations_widen_the_interval() {
        const CLUSTERS: usize = 50;
        const PER_CLUSTER: usize = 10;
        // Spread over [0, 1) with an irrational step, so the cluster
        // means genuinely vary and nothing keys on a particular draw.
        let values: Vec<f64> = (0..CLUSTERS)
            .map(|g| ((g as f64) * 0.618_033_988_749_9) % 1.0)
            .collect();
        let clustered = blocked(&values, PER_CLUSTER);

        // Same numbers, one per cluster: the naive bootstrap that treats
        // every position as its own independent observation.
        let flat: Vec<f64> = values
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, PER_CLUSTER))
            .collect();
        let mut naive = ClusterTally::new(flat.len());
        for (ix, value) in flat.iter().enumerate() {
            naive.push(ix, *value).unwrap();
        }

        let by_cluster =
            cluster_bootstrap(CLUSTERS, 2000, 7, |draw| clustered.mean_over(draw)).unwrap();
        let by_position =
            cluster_bootstrap(flat.len(), 2000, 7, |draw| naive.mean_over(draw)).unwrap();

        let clustered_width = by_cluster.high - by_cluster.low;
        let naive_width = by_position.high - by_position.low;
        assert!(
            clustered_width > naive_width,
            "clustering must not narrow the interval: cluster {clustered_width:.6} \
             vs position {naive_width:.6}"
        );
        // And by roughly the factor the argument predicts: the naive
        // bootstrap sees `PER_CLUSTER` times as many "independent"
        // observations, so its interval is about `sqrt(PER_CLUSTER)`
        // times too narrow. Loose bounds — this is a check that the
        // effect is of the stated size, not a distributional assertion.
        let ratio = clustered_width / naive_width;
        let expected = (PER_CLUSTER as f64).sqrt();
        assert!(
            ratio > expected * 0.7 && ratio < expected * 1.4,
            "expected roughly sqrt({PER_CLUSTER}) = {expected:.2}x wider, got {ratio:.2}x"
        );
    }

    /// The two ends both point estimate the same thing, so the
    /// difference is only ever in the width.
    #[test]
    fn clustering_does_not_move_the_point_estimate() {
        let values = [0.1, 0.9, 0.4, 0.6];
        let clustered = blocked(&values, 5);
        let interval =
            cluster_bootstrap(values.len(), 200, 3, |draw| clustered.mean_over(draw)).unwrap();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((interval.point - mean).abs() < 1e-12);
    }

    /// A draw is `clusters` picks with replacement, so the count of
    /// observations swings between draws — which is why a mean has to be
    /// a sum over a sum rather than an average of cluster means.
    #[test]
    fn uneven_clusters_are_weighted_by_their_size() {
        let mut tally = ClusterTally::new(2);
        // One observation of 0.0, ninety-nine of 1.0.
        tally.push(0, 0.0).unwrap();
        for _ in 0..99 {
            tally.push(1, 1.0).unwrap();
        }
        assert_eq!(tally.mean_over(&[0, 1]), Some(0.99));
        // Not the average of the two cluster means, which would be 0.5.
        assert_eq!(tally.total().n, 100);
        assert_eq!(tally.clusters_present(), 2);
    }

    /// An index the tally was not built over takes the whole draw out
    /// rather than quietly shrinking it. Dropping it would remove
    /// observations from the resample, and fewer observations narrow the
    /// interval — the failure would make a number look *more* precise.
    ///
    /// Release-mode behaviour only; a debug build trips the assertion
    /// inside `over` first, which is why this is `#[cfg(not(debug_assertions))]`.
    ///
    /// **This does not run under `just test`, `just test-nn` or `just ci`** —
    /// all three are dev profile. It runs under `just test-release`, which
    /// exists for it and is deliberately outside `ci`. Read it as "the
    /// release branch is asserted, by a command someone has to type" rather
    /// than as coverage: nothing reaches an out-of-range index here, since
    /// every draw is filled from `below(clusters)` and every `ClusterTally`
    /// in `chess::steerability` is sized from the same game count the
    /// bootstrap resamples.
    #[test]
    #[cfg(not(debug_assertions))]
    fn a_draw_naming_an_unknown_cluster_is_undefined_rather_than_shorter() {
        let tally = blocked(&[1.0, 1.0], 4);
        assert_eq!(tally.mean_over(&[0, 1]), Some(1.0));
        assert_eq!(tally.mean_over(&[0, 9]), None);
        assert_eq!(
            cluster_bootstrap(2, 10, 1, |_| tally.mean_over(&[0, 9])),
            Err(BootstrapError::UndefinedOnWholeSample)
        );
    }

    /// The interval renders into a caller's column width, so a table of
    /// them lines up under its own header.
    #[test]
    fn the_interval_honours_a_width_specifier() {
        let interval = Interval {
            point: 0.5,
            low: 0.25,
            high: 0.75,
            draws: 10,
            undefined_draws: 0,
            clusters: 2,
            seed: 1,
        };
        let bare = format!("{interval}");
        let padded = format!("{interval:<40}");
        assert_eq!(padded.len(), 40, "the width must be honoured: {padded:?}");
        assert!(padded.starts_with(&bare));
        assert_eq!(padded.trim_end(), bare);
    }

    #[test]
    fn a_tally_refuses_a_cluster_it_was_not_built_over() {
        let mut tally = ClusterTally::new(3);
        assert_eq!(
            tally.push(3, 1.0),
            Err(BootstrapError::ClusterOutOfRange {
                cluster: 3,
                clusters: 3
            })
        );
    }

    #[test]
    fn an_empty_sample_is_refused_rather_than_bounded() {
        assert_eq!(
            cluster_bootstrap(0, 10, 1, |_| Some(1.0)),
            Err(BootstrapError::NoClusters)
        );
        assert_eq!(
            cluster_bootstrap(3, 0, 1, |_| Some(1.0)),
            Err(BootstrapError::NoDraws)
        );
        assert_eq!(
            cluster_bootstrap(3, 10, 1, |_| None),
            Err(BootstrapError::UndefinedOnWholeSample)
        );
    }

    /// Defined on the whole sample and nowhere else. Kept separate from
    /// `UndefinedOnWholeSample` because the two say different things
    /// about what went wrong.
    ///
    /// The statistic here needs every cluster present, which the whole
    /// sample satisfies by construction and a draw with replacement over
    /// 20 clusters does not: the chance of one being a permutation is
    /// `20!/20^20`, about 2 in 100 million.
    #[test]
    fn a_statistic_that_survives_only_the_whole_sample_is_refused() {
        let err = cluster_bootstrap(20, 10, 1, |draw| {
            let distinct: std::collections::BTreeSet<usize> = draw.iter().copied().collect();
            (distinct.len() == 20).then_some(1.0)
        });
        assert_eq!(err, Err(BootstrapError::EveryDrawUndefined { draws: 10 }));
    }

    /// Draws that fall apart are counted rather than hidden, so a reader
    /// can see how much of the interval was thrown away.
    #[test]
    fn undefined_draws_are_counted_on_the_interval() {
        let calls = std::cell::Cell::new(0usize);
        let interval = cluster_bootstrap(4, 100, 5, |_| {
            let n = calls.get();
            calls.set(n + 1);
            // The first call is the point estimate; drop every other
            // draw after that.
            (n == 0 || n.is_multiple_of(2)).then_some(1.0)
        })
        .unwrap();
        assert_eq!(interval.undefined_draws, 50);
        assert_eq!(interval.draws, 50);
    }

    /// The signed call is the plain one with two more numbers read off
    /// the same draws, so the interval it carries has to be identical —
    /// not merely close.
    #[test]
    fn the_signed_call_returns_the_same_interval_as_the_plain_one() {
        let tally = blocked(&[0.2, -0.1, 0.4, 0.9, -0.3], 6);
        let plain = cluster_bootstrap(5, 500, 21, |draw| tally.mean_over(draw)).unwrap();
        let signed = cluster_bootstrap_signed(5, 500, 21, |draw| tally.mean_over(draw)).unwrap();
        assert_eq!(signed.interval, plain);
    }

    /// The levels say what the interval says, at the level the interval
    /// is drawn at. A quantity comfortably above zero has essentially no
    /// draws at or below it, and its interval clears zero from above;
    /// negate it and both statements turn over.
    #[test]
    fn the_levels_agree_with_the_interval_they_came_from() {
        let above = blocked(&[0.5, 0.6, 0.4, 0.55, 0.45], 6);
        let signed = cluster_bootstrap_signed(5, 2000, 3, |draw| above.mean_over(draw)).unwrap();
        assert!(signed.interval.excludes_zero_from_above());
        assert_eq!(signed.p_above_zero, 0.0);
        assert_eq!(signed.p_below_zero, 1.0);

        let below = blocked(&[-0.5, -0.6, -0.4, -0.55, -0.45], 6);
        let signed = cluster_bootstrap_signed(5, 2000, 3, |draw| below.mean_over(draw)).unwrap();
        assert!(signed.interval.excludes_zero_from_below());
        assert_eq!(signed.p_below_zero, 0.0);
        assert_eq!(signed.p_above_zero, 1.0);
    }

    /// A quantity straddling zero has a level in between, and neither
    /// end of the interval clears it — which is what makes the two
    /// assertions above evidence of anything.
    #[test]
    fn a_quantity_straddling_zero_has_a_level_in_between() {
        let tally = blocked(&[-1.0, -0.5, 0.0, 0.5, 1.0], 4);
        let signed = cluster_bootstrap_signed(5, 2000, 9, |draw| tally.mean_over(draw)).unwrap();
        assert!(!signed.interval.excludes_zero_from_above());
        assert!(!signed.interval.excludes_zero_from_below());
        assert!(
            signed.p_below_zero > 0.05 && signed.p_below_zero < 0.95,
            "{signed:?}"
        );
        // A draw at exactly zero counts towards both levels, so they
        // sum to at least one rather than to exactly one.
        assert!(
            signed.p_below_zero + signed.p_above_zero >= 1.0,
            "{signed:?}"
        );
    }

    /// Percentile ranks land where the doc comment says they do.
    #[test]
    fn the_percentiles_are_the_2_5_and_97_5_ranks() {
        let mut values: Vec<f64> = (0..2000).map(|i| i as f64).collect();
        let (low, high) = percentile_interval(&mut values);
        assert_eq!(low, 50.0);
        assert_eq!(high, 1949.0);
    }

    /// `below` covers its range and never leaves it.
    #[test]
    fn draws_stay_inside_the_range_and_reach_both_ends() {
        let mut rng = Resampler::new(99);
        let mut seen = [false; 5];
        for _ in 0..10_000 {
            let ix = rng.below(5).expect("a non-empty range");
            assert!(ix < 5);
            seen[ix] = true;
        }
        assert!(seen.iter().all(|s| *s), "every index should come up");
        assert_eq!(rng.below(0), None);
    }
}
