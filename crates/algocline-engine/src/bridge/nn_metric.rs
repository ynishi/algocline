//! `alc.nn.metric.*` — the cluster bootstrap (feature `nn`).
//!
//! # What is here, and what left
//!
//! One function: [`bootstrap_ci_impl`]. It answers "how much of this
//! difference is the draw" for a statistic whose observations arrive in
//! correlated groups, which is the question a checkpoint search asks
//! about two candidates.
//!
//! `kl` / `js` / `tvd` / `entropy` used to sit here too. They are
//! general-purpose mathematics rather than anything the nn layer owns,
//! and `mlua-mathlib` had grown its own information-theory module, so
//! keeping a second implementation on this namespace meant one
//! definition of "is this a distribution" per crate. They now live at
//! `alc.math.{kl_divergence, js_divergence, tvd, entropy}`.
//!
//! # What this module is not
//!
//! Measuring a checkpoint mid-run is the `on_ckpt` hook's job, not this
//! module's. The hook takes a Lua function
//! ([`super::nn_opts::extract_on_ckpt_hook`]) and the caller closes over
//! whatever it wants to measure, so nothing here needs to resolve a
//! metric by name. A `registry` sub-table used to sit on this namespace
//! for exactly that purpose; it turned a function into a string and back
//! again without a boundary in between, and was removed.

use mlua::prelude::*;

use algocline_nn::metric::bootstrap;

/// Register `alc.nn.metric.*` onto the pre-existing `alc.nn` table.
///
/// Called from [`super::register`] after [`super::register_nn`] has
/// populated `alc.nn`. Installs [`bootstrap_ci_impl`].
pub(super) fn register_nn_metric(lua: &Lua, nn_table: &LuaTable) -> LuaResult<()> {
    let metric_ns = lua.create_table()?;

    let bootstrap_ci = lua.create_function(
        |lua, (clusters, opts): (LuaTable, Option<LuaTable>)| -> LuaResult<LuaTable> {
            bootstrap_ci_impl(lua, &clusters, opts.as_ref())
        },
    )?;
    metric_ns.set("bootstrap_ci", bootstrap_ci)?;

    nn_table.set("metric", metric_ns)?;

    Ok(())
}

/// Error prefix for the `alc.nn.metric.bootstrap_ci` surface.
const BOOTSTRAP_CI_ERR_PREFIX: &str = "alc.nn.metric.bootstrap_ci";

/// Default resamples. 2,000 is the count the percentile ranks in
/// [`algocline_nn::metric::bootstrap`] are documented against (the
/// 50th and 1,949th order statistics at 95%).
const BOOTSTRAP_CI_DEFAULT_DRAWS: usize = 2_000;

/// `alc.nn.metric.bootstrap_ci(clusters, opts?) -> table`
///
/// Bound the mean of a sample whose observations arrive in groups that
/// are not independent of each other.
///
/// ```text
/// local ci = alc.nn.metric.bootstrap_ci(
///     { { 1.0, 0.0, 1.0 },   -- cluster 1's observations
///       { 0.0, 0.0 } },      -- cluster 2's
///     { draws = 2000, seed = 42 })
/// -- ci.point / ci.low / ci.high / ci.draws / ci.undefined_draws
/// -- ci.clusters / ci.seed
/// ```
///
/// # Why the clusters have to be spelled out
///
/// The resampling unit is the cluster, not the observation: a
/// bootstrap that drew observations would treat two readings from the
/// same group as two independent facts and return an interval narrower
/// than the sample supports. Which readings belong together is
/// something only the caller knows, so it is stated rather than
/// inferred — a flat list of numbers has no shape that could carry it.
///
/// # `seed` is required
///
/// The interval is a function of the draws, and the draws are a
/// function of the seed. Defaulting it would make the same call on the
/// same sample return different bounds with nothing in the result
/// saying why, so the caller supplies it and the result carries it
/// back.
///
/// # Errors
///
/// A missing or non-integer `seed`, an empty cluster list, a cluster
/// that is not an array of numbers, `draws` of zero, and the two
/// refusals from the Rust entry point: a statistic undefined on the
/// sample as walked (every cluster empty), and one that survives the
/// whole sample but no resample of it.
fn bootstrap_ci_impl(
    lua: &Lua,
    clusters: &LuaTable,
    opts: Option<&LuaTable>,
) -> LuaResult<LuaTable> {
    let count = clusters.raw_len();
    if count == 0 {
        return Err(LuaError::external(format!(
            "{BOOTSTRAP_CI_ERR_PREFIX}: clusters must be a non-empty array of observation \
             arrays; the resampling unit is the cluster, so there is nothing to resample"
        )));
    }

    let mut tally = bootstrap::ClusterTally::new(count);
    for i in 1..=count {
        let observations: Vec<f64> = clusters.get(i).map_err(|e| {
            LuaError::external(format!(
                "{BOOTSTRAP_CI_ERR_PREFIX}: cluster {i} must be an array of numbers: {e}"
            ))
        })?;
        for value in observations {
            if !value.is_finite() {
                return Err(LuaError::external(format!(
                    "{BOOTSTRAP_CI_ERR_PREFIX}: cluster {i} holds a non-finite observation; \
                     one would carry through every draw it appears in"
                )));
            }
            // `i` is 1-based on the Lua side and the tally is 0-based.
            tally
                .push(i - 1, value)
                .map_err(|e| LuaError::external(format!("{BOOTSTRAP_CI_ERR_PREFIX}: {e}")))?;
        }
    }

    let draws = match opts {
        Some(t) => t
            .get::<Option<usize>>("draws")
            .map_err(|e| {
                LuaError::external(format!(
                    "{BOOTSTRAP_CI_ERR_PREFIX}: opts.draws must be a positive integer: {e}"
                ))
            })?
            .unwrap_or(BOOTSTRAP_CI_DEFAULT_DRAWS),
        None => BOOTSTRAP_CI_DEFAULT_DRAWS,
    };
    let seed: u64 = opts
        .map(|t| {
            t.get::<Option<u64>>("seed").map_err(|e| {
                LuaError::external(format!(
                    "{BOOTSTRAP_CI_ERR_PREFIX}: opts.seed must be a non-negative integer: {e}"
                ))
            })
        })
        .transpose()?
        .flatten()
        .ok_or_else(|| {
            LuaError::external(format!(
                "{BOOTSTRAP_CI_ERR_PREFIX}: opts.seed is required; the same seed over the \
                 same sample reproduces the interval exactly, and an interval nothing can \
                 reproduce is not a measurement"
            ))
        })?;

    let interval = bootstrap::cluster_bootstrap(count, draws, seed, |draw| tally.mean_over(draw))
        .map_err(|e| LuaError::external(format!("{BOOTSTRAP_CI_ERR_PREFIX}: {e}")))?;

    let out = lua.create_table_with_capacity(0, 7)?;
    out.set("point", interval.point)?;
    out.set("low", interval.low)?;
    out.set("high", interval.high)?;
    out.set("draws", interval.draws)?;
    out.set("undefined_draws", interval.undefined_draws)?;
    out.set("clusters", interval.clusters)?;
    out.set("seed", interval.seed)?;
    Ok(out)
}
