//! Error bars for a statistic whose observations arrive in correlated
//! groups.
//!
//! # Why the groups matter
//!
//! The measurements this exists for are per-observation: a rate over
//! several thousand sampled positions, a divergence meaned over a
//! bucket. Those observations do not arrive one at a time from
//! independent sources — they come from a few hundred *runs* (one
//! generation, one evaluation pass, one recorded session), and
//! consecutive observations inside one run share a prompt, a seed and
//! most of a context. They are not thousands of independent draws, and
//! treating them as though they were reports a precision the sample
//! does not have: the naive interval narrows as `1/sqrt(observations)`
//! when the information in it only grows as `1/sqrt(runs)`.
//!
//! So the unit resampled here is the **cluster** (one run), not the
//! observation. A draw picks `C` clusters with replacement out of the
//! `C` that were walked, and every quantity in that draw is recomputed
//! over exactly those clusters — including the ones that appear twice
//! and excluding the ones that did not come up.
//!
//! # Why every term shares one draw
//!
//! The statistics this feeds are usually differences: one variant's
//! rate minus another's, one arm's ratio minus another's. A difference
//! of two separately bootstrapped quantities has no joint distribution
//! — the two intervals would each be correct about their own quantity
//! and say nothing about the gap between them, which is the thing being
//! judged.
//!
//! [`cluster_bootstrap`] therefore hands the statistic **the draw**, a
//! list of cluster indices, and lets it compute every term it needs
//! from that one list. The property this buys, stated exactly: the
//! function supplies one draw and no second one, so every term written
//! against the argument shares it. That is weaker than "a second
//! resample is unreachable" — [`cluster_bootstrap`] is re-entrant, and
//! a caller determined to bound one term against a different draw can
//! call it again. The resampler being private to this module narrows
//! the gap from the other side; the property above is what actually
//! holds.
//!
//! # Reproducibility
//!
//! The resampler is a SplitMix64 carried in this file rather than a
//! generator pulled from a dependency. A general-purpose RNG is allowed
//! to change algorithm between releases, so a run reproduced after a
//! dependency update would draw a different sample from the same seed
//! and quietly report a different interval. A generator written out
//! here makes reproducibility a property of this code, and the seed is
//! always supplied by the caller — nothing in this module reads a
//! clock or any other source of entropy.
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
/// SplitMix64 — a fixed additive step followed by two multiply-xorshift
/// rounds. Chosen for being short enough to carry in this file, which
/// is what makes a run reproducible against this code rather than
/// against whichever algorithm a dependency shipped that month.
/// Statistical quality is ample for picking indices out of a list;
/// nothing here is cryptographic.
///
/// Private to this module, so [`cluster_bootstrap`] is the only way to
/// obtain a stream. That narrows the gap the module header describes
/// from one side; it does not close it, since [`cluster_bootstrap`] is
/// re-entrant and a caller can call it again for a second draw.
#[derive(Debug, Clone)]
struct Resampler {
    state: u64,
}

impl Resampler {
    /// Start a stream from `seed`.
    ///
    /// The same seed always yields the same stream, on any host and any
    /// build of this crate.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 bits.
    fn next_u64(&mut self) -> u64 {
        // Odd increment, so the additive step is a full-period walk
        // over the 64-bit state regardless of where it started.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..n`, or `None` for an empty range.
    ///
    /// Lemire's multiply-shift with rejection: the plain `% n` is
    /// biased towards small indices whenever `n` does not divide
    /// `2^64`. At the cluster counts this is used with the bias sits
    /// far below the noise it would perturb — but it costs one
    /// comparison to remove, and a biased resampler is not something a
    /// reader of the interval can check.
    fn below(&mut self, n: usize) -> Option<usize> {
        let n = u64::try_from(n).ok().filter(|n| *n > 0)?;
        let mut product = u128::from(self.next_u64()) * u128::from(n);
        let mut low = product as u64;
        if low < n {
            // Reject the leftover window at the bottom of the range,
            // which is the part with one extra representative.
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
    /// Returns `false` and leaves `into` in an unspecified state when
    /// `clusters` is zero, which is the only way the draw can fail.
    fn draw_into(&mut self, clusters: usize, into: &mut [usize]) -> bool {
        if clusters == 0 {
            return false;
        }
        for slot in into.iter_mut() {
            match self.below(clusters) {
                Some(ix) => *slot = ix,
                // Unreachable while `clusters > 0`: `below` only
                // returns `None` on an empty range or on a width that
                // does not fit a `u64`, and `clusters` is already one
                // of neither.
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
/// clusters can be added: a mean over a resample is the sum of the
/// drawn clusters' sums over the sum of their counts, which is not the
/// average of their means unless every cluster holds the same number of
/// observations — and clusters generally do not.
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
    ///
    /// `None` rather than `0.0`: a mean over nothing is undefined, and
    /// a zero returned in its place is indistinguishable from a real
    /// measurement that happened to be zero.
    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}

/// One quantity, tallied cluster by cluster.
///
/// Built once over the whole sample; a bootstrap draw then reads it by
/// cluster index, so a draw costs `O(clusters)` rather than
/// `O(observations)`. With 2,000 draws over 100 clusters that is the
/// difference between 200 thousand additions and however many
/// observations the sample holds, times 2,000.
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
    /// [`BootstrapError::ClusterOutOfRange`] when `cluster` is not one
    /// of the clusters this was built over. An error rather than a
    /// growing `Vec`, because the cluster count is fixed by the sample
    /// and an index past it means the caller is tallying against a
    /// different sample than it thinks.
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
    /// # Precondition, and why an unknown index is not merely skipped
    ///
    /// Every index must name a cluster this was built over.
    /// [`ClusterTally::push`] refuses one that does not, because an
    /// index past the end means the caller is tallying against a
    /// different sample than it thinks — and the same is true here,
    /// with a worse consequence. This is reached on every draw, so
    /// dropping an unknown index would quietly remove observations from
    /// the resample, and fewer observations is a **narrower interval**:
    /// the mistake would make the number look more precise rather than
    /// make anything fail.
    ///
    /// So the whole draw goes undefined instead.
    /// [`cluster_bootstrap`] counts those draws in
    /// [`Interval::undefined_draws`] and, if none survive, refuses with
    /// [`BootstrapError::EveryDrawUndefined`]. Neither outcome can be
    /// mistaken for a measurement.
    pub fn over(&self, draw: &[usize]) -> Option<Tally> {
        let mut total = Tally::default();
        for &cluster in draw {
            total.merge(*self.per_cluster.get(cluster)?);
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

    /// Totals over every cluster, each counted once: the sample as
    /// walked.
    pub fn total(&self) -> Tally {
        let mut total = Tally::default();
        for tally in &self.per_cluster {
            total.merge(*tally);
        }
        total
    }

    /// How many distinct clusters contributed at least one observation.
    ///
    /// Worth reporting beside a number whose observations were filtered:
    /// a bucket that only some clusters reach rests on fewer clusters
    /// than [`ClusterTally::clusters`] suggests, and the interval's
    /// width is the only other place that shows.
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
    /// Spelled once here so that two callers testing the same claim
    /// cannot disagree about what it means.
    pub fn excludes_zero_from_above(&self) -> bool {
        self.low > 0.0
    }

    /// Whether the whole interval lies below zero.
    pub fn excludes_zero_from_below(&self) -> bool {
        self.high < 0.0
    }
}

impl std::fmt::Display for Interval {
    /// Routed through [`std::fmt::Formatter::pad`] rather than `write!`
    /// so that a caller's width specifier is honoured.
    ///
    /// `write!` ignores `f.width()`, which is not a cosmetic detail in
    /// a table: a header row written with `{:<28}` would line up while
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

    /// The statistic is not defined on the sample as walked, so there
    /// is no point estimate to put an interval around. A ratio whose
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
/// `stat` receives one draw — a list of `clusters` cluster indices,
/// with repetitions — and returns its value on exactly those clusters,
/// or `None` where the draw leaves it undefined. Every term of a
/// difference must be computed from that same list; see the module
/// header for why bootstrapping the terms separately answers a
/// different question.
///
/// The point estimate is `stat` on `0..clusters`, each cluster once,
/// which is the sample as walked. `seed` is the caller's: the same seed
/// over the same sample reproduces the interval exactly.
///
/// A draw whose value is non-finite is counted as undefined, because a
/// ratio over an emptied bucket arrives as an infinity rather than as a
/// `None` and the two mean the same thing to a reader.
///
/// # Errors
///
/// - [`BootstrapError::NoClusters`] / [`BootstrapError::NoDraws`] when
///   either count is zero.
/// - [`BootstrapError::UndefinedOnWholeSample`] when `stat` has no
///   value on the sample as walked.
/// - [`BootstrapError::EveryDrawUndefined`] when it has one there and
///   on no draw of it.
pub fn cluster_bootstrap(
    clusters: usize,
    draws: usize,
    seed: u64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<Interval, BootstrapError> {
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
            // Only reachable with zero clusters, which is refused
            // above; kept as a refusal rather than an unwrap so the
            // guard cannot decay into a panic if that changes.
            return Err(BootstrapError::NoClusters);
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
        clusters,
        seed,
    })
}

/// The [`CONFIDENCE`] percentile interval of `values`, which is sorted
/// in place.
///
/// Nearest-rank percentiles: the low end is the value at rank
/// `floor(alpha/2 * B)` and the high end at
/// `ceil((1 - alpha/2) * B) - 1`, which for `B = 2000` at 95% is the
/// 50th and the 1949th — the conventional 2.5 / 97.5 percentile
/// bootstrap.
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
    use std::cell::Cell;

    /// A tally over `clusters` clusters holding `per` copies of
    /// `value` in every cluster.
    fn flat_tally(clusters: usize, per: usize, value: f64) -> ClusterTally {
        let mut tally = ClusterTally::new(clusters);
        for cluster in 0..clusters {
            for _ in 0..per {
                tally.push(cluster, value).expect("cluster is in range");
            }
        }
        tally
    }

    #[test]
    fn tally_merges_sums_and_counts() {
        let mut a = Tally { sum: 1.5, n: 2 };
        a.merge(Tally { sum: 2.5, n: 3 });
        assert_eq!(a, Tally { sum: 4.0, n: 5 });
        assert_eq!(a.mean(), Some(0.8));
    }

    #[test]
    fn tally_mean_is_none_without_observations() {
        assert_eq!(Tally::default().mean(), None);
    }

    #[test]
    fn push_rejects_a_cluster_outside_the_sample() {
        let mut tally = ClusterTally::new(3);
        assert_eq!(
            tally.push(3, 1.0),
            Err(BootstrapError::ClusterOutOfRange {
                cluster: 3,
                clusters: 3
            })
        );
        // The refusal must not have recorded anything.
        assert_eq!(tally.total(), Tally::default());
    }

    #[test]
    fn totals_count_a_cluster_once_per_appearance() {
        let mut tally = ClusterTally::new(3);
        tally.push(0, 1.0).unwrap();
        tally.push(1, 2.0).unwrap();
        tally.push(1, 4.0).unwrap();

        assert_eq!(tally.clusters(), 3);
        assert_eq!(tally.clusters_present(), 2, "cluster 2 saw nothing");
        assert_eq!(tally.total(), Tally { sum: 7.0, n: 3 });

        // Cluster 1 drawn twice contributes twice.
        assert_eq!(tally.over(&[1, 1]), Some(Tally { sum: 12.0, n: 4 }));
        assert_eq!(tally.mean_over(&[1, 1]), Some(3.0));
        // A draw that only caught the empty cluster has no mean.
        assert_eq!(tally.over(&[2]), Some(Tally::default()));
        assert_eq!(tally.mean_over(&[2]), None);
    }

    #[test]
    fn a_draw_naming_an_unknown_cluster_is_undefined_not_shortened() {
        let tally = flat_tally(2, 3, 1.0);
        assert_eq!(tally.over(&[0, 2]), None, "the whole draw goes undefined");
        assert_eq!(tally.mean_over(&[0, 2]), None);
    }

    #[test]
    fn identical_clusters_give_a_zero_width_interval() {
        let tally = flat_tally(6, 4, 2.0);
        let interval = cluster_bootstrap(tally.clusters(), 200, 7, |draw| tally.mean_over(draw))
            .expect("interval");
        assert_eq!(interval.point, 2.0);
        assert_eq!(interval.low, 2.0);
        assert_eq!(interval.high, 2.0);
        assert_eq!(interval.draws, 200);
        assert_eq!(interval.undefined_draws, 0);
        assert_eq!(interval.clusters, 6);
        assert_eq!(interval.seed, 7);
    }

    #[test]
    fn a_single_cluster_is_degenerate() {
        // With one cluster every draw is that cluster, so the interval
        // collapses onto the point estimate: resampling cannot express
        // between-cluster variation that the sample never showed.
        let mut tally = ClusterTally::new(1);
        tally.push(0, 1.0).unwrap();
        tally.push(0, 3.0).unwrap();
        let interval =
            cluster_bootstrap(1, 64, 99, |draw| tally.mean_over(draw)).expect("interval");
        assert_eq!(
            (interval.point, interval.low, interval.high),
            (2.0, 2.0, 2.0)
        );
    }

    #[test]
    fn a_spread_sample_gives_a_positive_width_interval_around_the_point() {
        let mut tally = ClusterTally::new(8);
        for cluster in 0..8 {
            tally.push(cluster, cluster as f64).unwrap();
        }
        let interval =
            cluster_bootstrap(8, 500, 12345, |draw| tally.mean_over(draw)).expect("interval");
        assert_eq!(interval.point, 3.5, "mean of 0..8");
        assert!(
            interval.low < interval.point && interval.point < interval.high,
            "expected the point inside a non-degenerate interval, got {interval}"
        );
        assert!(
            interval.low >= 0.0 && interval.high <= 7.0,
            "got {interval}"
        );
    }

    #[test]
    fn the_same_seed_reproduces_the_interval() {
        let mut tally = ClusterTally::new(10);
        for cluster in 0..10 {
            tally.push(cluster, (cluster * cluster) as f64).unwrap();
        }
        let run = |seed| {
            cluster_bootstrap(10, 300, seed, |draw| tally.mean_over(draw)).expect("interval")
        };
        assert_eq!(run(2026), run(2026), "same seed, same interval");
        assert_ne!(
            run(2026).low,
            run(2027).low,
            "a different seed must draw a different sample"
        );
    }

    #[test]
    fn the_resampler_stream_is_a_function_of_its_seed() {
        let stream = |seed| {
            let mut r = Resampler::new(seed);
            (0..32).map(|_| r.below(10).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(stream(1), stream(1));
        assert_ne!(stream(1), stream(2));
        assert!(stream(1).iter().all(|ix| *ix < 10), "indices stay in range");
    }

    #[test]
    fn a_draw_fills_every_slot() {
        let mut r = Resampler::new(5);
        let mut draw = vec![usize::MAX; 4];
        assert!(r.draw_into(3, &mut draw));
        assert!(draw.iter().all(|ix| *ix < 3), "got {draw:?}");
        assert!(!r.draw_into(0, &mut draw), "zero clusters cannot be drawn");
        assert_eq!(r.below(0), None);
    }

    #[test]
    fn zero_clusters_and_zero_draws_are_refused() {
        assert_eq!(
            cluster_bootstrap(0, 10, 1, |_| Some(1.0)),
            Err(BootstrapError::NoClusters)
        );
        assert_eq!(
            cluster_bootstrap(4, 0, 1, |_| Some(1.0)),
            Err(BootstrapError::NoDraws)
        );
    }

    #[test]
    fn a_statistic_undefined_on_the_whole_sample_is_refused() {
        assert_eq!(
            cluster_bootstrap(4, 10, 1, |_| None),
            Err(BootstrapError::UndefinedOnWholeSample)
        );
        assert_eq!(
            cluster_bootstrap(4, 10, 1, |_| Some(f64::INFINITY)),
            Err(BootstrapError::UndefinedOnWholeSample),
            "a non-finite point estimate is no estimate"
        );
    }

    #[test]
    fn a_statistic_that_survives_only_the_whole_sample_is_refused() {
        // The first call is the point estimate; every draw after it is
        // undefined. That is the shape of a statistic resting on too
        // few clusters to resample, and it must not come back as an
        // interval of width zero.
        let first = Cell::new(true);
        let err = cluster_bootstrap(3, 25, 1, move |_| first.replace(false).then_some(1.0))
            .expect_err("every draw undefined");
        assert_eq!(err, BootstrapError::EveryDrawUndefined { draws: 25 });
    }

    #[test]
    fn undefined_draws_are_counted_not_hidden() {
        let tally = flat_tally(4, 2, 1.0);
        // Draws that picked cluster 0 more than once are declared
        // undefined. The sample as walked holds every cluster exactly
        // once, so the point estimate survives and only resamples fall
        // out — which is the situation this accounting exists for.
        let interval = cluster_bootstrap(4, 200, 31, |draw| {
            if draw.iter().filter(|c| **c == 0).count() > 1 {
                None
            } else {
                tally.mean_over(draw)
            }
        })
        .expect("interval");
        assert!(interval.undefined_draws > 0, "got {interval:?}");
        assert_eq!(
            interval.draws + interval.undefined_draws,
            200,
            "every draw is accounted for"
        );
    }

    #[test]
    fn a_non_finite_draw_counts_as_undefined() {
        let seen = Cell::new(0usize);
        let interval = cluster_bootstrap(3, 20, 4, move |_| {
            let n = seen.replace(seen.get() + 1);
            // The point estimate is finite; every other call is not.
            Some(if n == 0 { 1.0 } else { f64::NAN })
        });
        assert_eq!(
            interval,
            Err(BootstrapError::EveryDrawUndefined { draws: 20 })
        );
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut values: Vec<f64> = (1..=100).map(f64::from).collect();
        // Deliberately unsorted on the way in: the helper sorts.
        values.reverse();
        let (low, high) = percentile_interval(&mut values);
        assert_eq!((low, high), (3.0, 98.0));
    }

    #[test]
    fn percentiles_of_a_single_value_clamp_onto_it() {
        let mut values = vec![4.5];
        assert_eq!(percentile_interval(&mut values), (4.5, 4.5));
    }

    #[test]
    fn zero_exclusion_reads_both_ends() {
        let base = Interval {
            point: 0.0,
            low: 0.0,
            high: 0.0,
            draws: 1,
            undefined_draws: 0,
            clusters: 1,
            seed: 0,
        };
        let above = Interval {
            point: 0.5,
            low: 0.1,
            high: 0.9,
            ..base
        };
        let below = Interval {
            point: -0.5,
            low: -0.9,
            high: -0.1,
            ..base
        };
        let straddling = Interval {
            point: 0.0,
            low: -0.1,
            high: 0.9,
            ..base
        };

        assert!(above.excludes_zero_from_above() && !above.excludes_zero_from_below());
        assert!(below.excludes_zero_from_below() && !below.excludes_zero_from_above());
        assert!(
            !straddling.excludes_zero_from_above() && !straddling.excludes_zero_from_below(),
            "an interval containing zero excludes it from neither side"
        );
        // A boundary exactly at zero is not exclusion.
        assert!(!base.excludes_zero_from_above() && !base.excludes_zero_from_below());
    }

    #[test]
    fn display_honours_a_width_specifier() {
        let interval = Interval {
            point: 0.5,
            low: 0.25,
            high: 0.75,
            draws: 10,
            undefined_draws: 0,
            clusters: 2,
            seed: 0,
        };
        let rendered = format!("{interval}");
        assert_eq!(rendered, "+0.500000 [+0.250000, +0.750000]");
        let padded = format!("{interval:<40}|");
        assert_eq!(padded.len(), 41, "width must be honoured: {padded:?}");
    }
}
