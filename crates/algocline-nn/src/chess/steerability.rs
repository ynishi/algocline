//! The pre-registered statistics, assembled from several arms' records
//! over one resample of the games.
//!
//! Steerability is how far behaviour moves when the conditioning input
//! changes. Two questions were fixed before any of it was measured:
//! whether conditioning at every position changes the played move more
//! often than a prefix token does (H14), and whether it also reaches
//! further down the game (H15).
//!
//! # The legality experiment
//!
//! A second set of arms asks a different question: what handing the
//! model the rules frees, rather than where the condition attaches.
//! Every arm of it is prefix-conditioned, so the axis they differ on is
//! whether the legal set reached the forward pass — [`h21`] on the cost
//! of the played move, [`h20`] on whether steerability survives it, and
//! [`h19`] as a manipulation check that carries no verdict because its
//! direction was known before the data arrived.
//!
//! Those arms have their own role check ([`check_legality_roles`]) and
//! their own top-1 tolerance ([`gate_top1_legality`]), rather than the
//! H14/H15 ones widened to cover both sets. A check that passes on two
//! arm sets pins neither, and the arms plan 02 was judged under keep
//! the criteria they were judged under.
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
use crate::metric::bootstrap::{
    cluster_bootstrap, cluster_bootstrap_signed, stratified_cluster_bootstrap, BootstrapError,
    ClusterTally, Interval, CONFIDENCE,
};

/// Arm trained with the band conditioning every position.
pub const PERPOS: &str = "perpos";

/// Arm trained with the band as a prefix token, which is what every
/// checkpoint before this experiment did.
pub const PREFIX: &str = "prefix";

/// Second per-position run, differing from [`PERPOS`] only in the
/// shuffle seed. The gap between the two is the same-arm gap `G`.
pub const PERPOS_B: &str = "perpos-b";

/// The legality experiment's control: the arm plan 02 already baked,
/// under its own name.
///
/// Not a second checkpoint. `§3.1` makes every arm of that experiment
/// prefix-conditioned — `Gpt2Custom::validate` refuses a model carrying
/// both a conditioning table and a legality table, so a legality arm
/// cannot also be per-position — and the control is therefore the
/// prefix arm rather than anything new. Spelled as its own constant
/// because a reader of [`h19`] should not have to know that the string
/// is shared to see which role is meant.
pub const CONTROL: &str = PREFIX;

/// Legality experiment, treatment arm `A`: loss over the legal moves,
/// legality **not** supplied as an input.
pub const ARM_A: &str = "A";

/// Second run of [`ARM_A`], differing only in the shuffle seed.
pub const ARM_A_B: &str = "A-b";

/// Legality experiment, treatment arm `AB`: loss over the legal moves
/// **and** the legal ids handed to the forward pass.
///
/// Its distance from [`ARM_A`] is the one comparison in `§4` that is
/// neither circular nor already known in direction: the two train on
/// the same objective and differ only in whether the legal set reaches
/// the model.
pub const ARM_AB: &str = "AB";

/// Second run of [`ARM_AB`], differing only in the shuffle seed.
pub const ARM_AB_B: &str = "AB-b";

/// The legality experiment's five arms, in the order `§3.3` lists them.
pub const LEGALITY_ARMS: [&str; 5] = [CONTROL, ARM_A, ARM_A_B, ARM_AB, ARM_AB_B];

/// What each of those five arms' checkpoints must have been, on both
/// axes.
///
/// Every one of them is prefix-conditioned, so the encoding alone
/// separates nothing: the axis that tells the arms apart is the second,
/// and it is the one a header could not state before format version 4.
/// See [`check_legality_roles`].
pub const LEGALITY_ROLES: [(&str, CondEncoding, bool); 5] = [
    (CONTROL, CondEncoding::Prefix, false),
    (ARM_A, CondEncoding::Prefix, false),
    (ARM_A_B, CondEncoding::Prefix, false),
    (ARM_AB, CondEncoding::Prefix, true),
    (ARM_AB_B, CondEncoding::Prefix, true),
];

/// Survival experiment, arm `P`: per-position conditioning **with** the
/// loss taken over the legal moves.
///
/// The combination plan 02 and plan 03 between them never baked. Plan 02
/// measured per-position against prefix with the loss over the whole
/// vocabulary; plan 03 took the loss over the legal moves but made every
/// arm prefix-conditioned, because [`crate::arch::Gpt2Custom::validate`]
/// refuses a model carrying both the conditioning table and the legality
/// table. **That refusal is about legality as an input and says nothing
/// about the loss**, which carries no table — so this arm was buildable
/// throughout.
pub const ARM_P: &str = "P";

/// Second run of [`ARM_P`], differing only in the shuffle seed.
pub const ARM_P_B: &str = "P-b";

/// The survival experiment's four arms.
///
/// Two are new; [`ARM_A`] and [`ARM_A_B`] are plan 03's, reused as the
/// prefix side because they are the same recipe on the same corpus at
/// the same step count with the same pair of seeds.
pub const SURVIVAL_ARMS: [&str; 4] = [ARM_A, ARM_A_B, ARM_P, ARM_P_B];

/// What each of those four checkpoints must have been, on both axes.
///
/// Here the **encoding** is what separates the arms, which is the mirror
/// of [`LEGALITY_ROLES`]: there every arm was prefix and the legality
/// axis told them apart. The legality axis is `false` throughout —
/// plan 04 holds it constant rather than varying it, so a legality
/// checkpoint reaching any slot is a mixed-up arm set rather than a
/// third condition.
pub const SURVIVAL_ROLES: [(&str, CondEncoding, bool); 4] = [
    (ARM_A, CondEncoding::Prefix, false),
    (ARM_A_B, CondEncoding::Prefix, false),
    (ARM_P, CondEncoding::EveryPosition, false),
    (ARM_P_B, CondEncoding::EveryPosition, false),
];

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

/// How far a legality arm's mean top-1 match may sit from the control's.
///
/// Wider than [`TOP1_GATE`] by design, not by drift. `§5.1` widens it
/// because the same-recipe seed gap on top-1 was measured at 0.0146 and
/// 0.0093, so a 0.02 tolerance can be tripped by the shuffle seed alone
/// — every other criterion in that plan carries a floor and this one
/// did not. Recorded before any arm of the set was baked, and applied
/// rather than tuned.
pub const TOP1_GATE_LEGALITY: f64 = 0.03;

/// How far [`ARM_P`]'s mean top-1 match may sit from [`ARM_A`]'s.
///
/// **The reference is the arm being compared, not a control.** Plan 03
/// set its admission against the control and every treatment arm fell
/// outside it — for being *better*, by +0.062 to +0.077, which is the
/// loss mask's intended effect against a tolerance sized from the
/// shuffle seed. A band around a baseline on a different objective
/// excludes by construction, and no data can rescue it.
///
/// Here both sides carry the same objective, and the treatment's
/// intended effect on top-1 is about zero: plan 02 measured
/// per-position against prefix at +0.006 / +0.004 with intervals
/// spanning zero. Against a tolerance of 0.03 and a same-recipe seed
/// gap measured at 0.0123 / 0.0112 on this very arm pair, both passing
/// and failing are reachable — which is the check plan 03 did not make.
pub const TOP1_GATE_SURVIVAL: f64 = 0.03;

/// Oldest record format that carries `ce` and `top2_margin`.
///
/// `§5.1`'s third admission item asks that every record carry both, and
/// asks it of the **format version** rather than of the fields. Their
/// `None` says the position could not be scored, which is a fact about
/// the position; a walk from a stale binary has no `ce` at all, which
/// is a fact about the run. Probing the fields would collapse the two
/// and read a whole stale walk as a walk of unscoreable positions. See
/// [`crate::chess::records::GammaRecord::ce`], which documents the same
/// split from the other side.
pub const SCOREABLE_VERSION: u32 = 3;

/// Oldest record format whose header can state the legality axis.
///
/// [`crate::chess::records::WalkHeader::legal_input`] arrived at
/// version 4, and before it the field parses as `false` whatever the
/// checkpoint was. Checking the value alone therefore protects only the
/// slots whose role calls for `true`: a version 3 walk of a legality
/// checkpoint dropped into [`ARM_A`], [`ARM_A_B`] or [`CONTROL`] passes
/// on the default, having said nothing.
///
/// Requiring the version closes that from the other side. A walk older
/// than this is unverifiable on the axis in **either** direction rather
/// than merely unstated in one, and is refused as such.
///
/// The residual it removes pointed the safe way: a legality checkpoint
/// standing in for `A` puts two like arms on the two sides of `D`,
/// which pushes the difference toward zero and so toward an
/// undetermined verdict rather than toward a confirmation. That is a
/// direction and not a guarantee, and an undetermined verdict reached
/// that way is indistinguishable from an honest one. The check is one
/// comparison.
pub const LEGALITY_AXIS_VERSION: u32 = 4;

/// Jouseki experiment (plan 06), primary seed arm.
///
/// One recipe, two seeds: the comparison H29 makes is **within** the
/// model — the correct opening-family token against the wrong one —
/// so there is no treatment arm and no control, and the second seed
/// exists only to price the floor.
pub const ARM_J: &str = "J";

/// Second run of [`ARM_J`], differing only in the shuffle seed.
pub const ARM_J_B: &str = "J-b";

/// The jouseki experiment's two arms.
pub const JOUSEKI_ARMS: [&str; 2] = [ARM_J, ARM_J_B];

/// What both checkpoints must have been, on both axes.
///
/// Per-position conditioning with no legality input — plan 04's
/// adopted operating point, held constant rather than varied. Unlike
/// [`SURVIVAL_ROLES`] the two arms here agree on every axis a header
/// states about the **checkpoint**; what separates a jouseki walk from
/// another is which games it walked, and that axis is checked against
/// [`crate::chess::records::WalkHeader::games_of`] by
/// [`check_jouseki_roles`] rather than here.
pub const JOUSEKI_ROLES: [(&str, CondEncoding, bool); 2] = [
    (ARM_J, CondEncoding::EveryPosition, false),
    (ARM_J_B, CondEncoding::EveryPosition, false),
];

/// Oldest record format whose header states which games were walked.
///
/// [`crate::chess::records::WalkHeader::games_of`] arrived at version
/// 5. Before it a walk over one family's games and a walk over
/// another's produced headers that could not be told apart, and H29's
/// cost — read per band against the walk's own family — would have its
/// sign silently reversed by feeding it the other file. The same move
/// [`LEGALITY_AXIS_VERSION`] makes for the legality axis: refuse on
/// the version, where the field's absence and its default cannot be
/// confused.
pub const WALK_FILTER_VERSION: u32 = 5;

/// How far [`ARM_J_B`]'s mean top-1 match may sit from [`ARM_J`]'s.
///
/// The reference is the seed replicate — there is no other arm. The
/// two runs differ in nothing but the shuffle seed, so the only thing
/// this can catch is a run defect (a wrong corpus, a wrong step count,
/// a stale checkpoint in one slot); the treatment axis H29 reads runs
/// **within** each model and no admission band can sit across it. The
/// tolerance is [`TOP1_GATE_SURVIVAL`]'s, carried rather than re-derived:
/// same recipe, same objective, and the same-recipe seed gap it was
/// sized against (0.0123 / 0.0112) is the quantity being bounded here.
pub const TOP1_GATE_JOUSEKI: f64 = 0.03;

/// The shallow stratum H29 prints descriptively: ply 0 through 19.
///
/// Plan 06 §3 registers the main judgment over the whole walk and the
/// ply strata as description only — an opening family is a property of
/// the early plies, so the cost fading with depth is expected rather
/// than disqualifying, and a stratum is not given a verdict.
pub const JOUSEKI_SHALLOW: (usize, usize) = (0, 20);

/// Games the floor `§5.3` measured is quoted at, and the anchor of the
/// curve [`Resolution::at`] reads.
///
/// Not the walk's own target. `§5.3` sizes the walk at 6,200 positions,
/// which lands near 204 games for a floor near 0.05 nats. This pair is
/// where the measurement is stated; the walk's own floor is that
/// measurement carried along the curve, not a second calculation.
pub const PLANNED_GAMES: usize = 180;

/// Minimum detectable effect on `ce_legal`, in nats, at
/// [`PLANNED_GAMES`].
///
/// **Measured rather than derived**, and it moved when it was measured.
/// `§5.3`'s first table solved for the effect from the standard
/// deviation of *per-game* means — equal weight per game — while
/// `ce_legal` aggregates sum-over-sum, so a long game counts for more.
/// Those are two different estimators, and the first put this figure at
/// 0.05: optimistic, in the direction that admits verdicts. The number
/// here is the re-measurement, made by running the estimator itself
/// over bootstrap draws of the 2026-05 records, aggregating exactly as
/// [`ClusterTally::mean_over`] does.
pub const PLANNED_EFFECT: f64 = 0.0533;

/// Strata the flip-rate comparison is cut into — quartiles, so four.
pub const STRATA: usize = 4;

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

    /// An arm is playing a role its checkpoint's legality axis does not
    /// fit.
    ///
    /// [`StatError::WrongEncodingForRole`]'s counterpart, and the one
    /// that matters for the legality experiment: every arm there is
    /// prefix-conditioned, so the encoding separates none of them and
    /// this is the only axis a swap could show up on. Exchange `A` and
    /// `AB` on a command line and `D = ce_legal(A) - ce_legal(AB)`
    /// changes sign while every figure stays well-formed.
    ///
    /// This is a disagreement between what the header **says** and what
    /// the role needs, so it is about a walk whose header could say
    /// something. Every path in this module that produces it reads the
    /// format version first and returns
    /// [`StatError::RecordsPredateLegalityAxis`] for a walk older than
    /// [`LEGALITY_AXIS_VERSION`], so a defaulted `false` does not
    /// arrive here dressed as a recorded one. See
    /// [`crate::chess::records::WalkHeader::legal_input`].
    #[error(
        "arm {arm:?} was scored from a checkpoint whose header records legality-as-input = \
         {found}, and that role calls for {want}; the arms of this set differ on no other axis, \
         so a swap here leaves every number well-formed and reverses what the difference means"
    )]
    WrongLegalInputForRole {
        /// Arm whose legality axis does not fit its role.
        arm: &'static str,
        /// What the role requires.
        want: bool,
        /// What the arm's header records.
        found: bool,
    },

    /// An arm's walk predates the header field that states the legality
    /// axis.
    ///
    /// Separate from [`StatError::WrongLegalInputForRole`] because the
    /// two have different remedies: that one says a file is in the
    /// wrong slot, this one says the file cannot answer the question in
    /// any slot and has to be walked again by a current build.
    ///
    /// It is what closes the default's remaining gap. Before format
    /// version 4 the field parses as `false` whatever the checkpoint
    /// was, so reading the value alone only ever refuses a slot that
    /// requires `true` — the [`ARM_AB`] pair — while the same stale
    /// walk of the same legality checkpoint passes in [`ARM_A`],
    /// [`ARM_A_B`] or [`CONTROL`] on a default it never recorded.
    #[error(
        "arm {arm:?} was walked at record format version {found}, and checking its legality \
         axis needs {needed} or later; before that the header could not state the axis at all \
         and reads false whatever the checkpoint was, so a slot whose role calls for false \
         would admit it on a default rather than on anything the walk recorded"
    )]
    RecordsPredateLegalityAxis {
        /// Arm whose walk is too old to state the axis.
        arm: &'static str,
        /// Version its header declares.
        found: u32,
        /// Oldest version that carries the field.
        needed: u32,
    },

    /// An arm's walk predates the header field that states which games
    /// were walked.
    ///
    /// [`StatError::RecordsPredateLegalityAxis`]'s counterpart for the
    /// walk-filter axis: before [`WALK_FILTER_VERSION`] the header
    /// could not say whose games these are, and H29's per-band cost
    /// read against the wrong family reverses sign with every figure
    /// well-formed. Same remedy as there — walk again with a current
    /// binary — rather than trusting a default.
    #[error(
        "arm {arm:?} was walked at record format version {found}, and checking which games it \
         walked needs {needed} or later; before that the header could not state the filter at \
         all, and a walk of the wrong family's games reverses the sign of the per-band cost \
         with every number well-formed"
    )]
    RecordsPredateWalkFilter {
        /// Arm whose walk is too old to state the filter.
        arm: &'static str,
        /// Version its header declares.
        found: u32,
        /// Oldest version that carries the field.
        needed: u32,
    },

    /// An arm's walk was over different games than its slot calls for.
    ///
    /// The jouseki arms differ from each other on no checkpoint axis —
    /// same recipe, both per-position, both maskless on the input side
    /// — so which **games** were walked is the only axis a swap could
    /// show up on, and this is what refuses it. The header states it
    /// ([`crate::chess::records::WalkHeader::games_of`]) since format
    /// version 5, and [`StatError::RecordsPredateWalkFilter`] keeps an
    /// older walk from arriving here dressed as an unfiltered one.
    #[error(
        "arm {arm:?} was walked over games of {found}, and this slot reads them as games of \
         {want:?}; the per-band cost read against the wrong family reverses sign with every \
         number well-formed"
    )]
    WrongWalkedGamesForRole {
        /// Arm whose walked games do not fit the slot.
        arm: &'static str,
        /// The family token the slot is being read as.
        want: String,
        /// What the walk's header records, rendered for the message.
        found: String,
    },

    /// The family token a slot is being read as is not one of the
    /// walk's bands.
    ///
    /// The cost is an index into the per-band columns, so a token the
    /// band list does not carry has no column — and the near-miss this
    /// refuses is a checkpoint of another band vocabulary reaching the
    /// judge with every file self-consistent.
    #[error(
        "family token {token:?} is not among the walk's bands {bands:?}, so it names no per-band \
         column to read a cost from"
    )]
    FamilyTokenNotABand {
        /// The token the caller asked to read.
        token: String,
        /// The bands the walk carries.
        bands: Vec<String>,
    },

    /// The duo judge needs exactly four two-part combination columns.
    ///
    /// A 2×2 design is what makes "the combination that falsifies one
    /// slot" unique; with any other grid that phrase names zero or
    /// several columns and the judge would be choosing.
    #[error(
        "the duo walk carries {found} band column(s) and the registered design has exactly \
         four two-part combinations — outside a 2×2 grid, `the combination that falsifies \
         one slot` names zero or several columns"
    )]
    DuoNeedsFourCombos {
        /// Band columns the walk carries.
        found: usize,
    },

    /// The cell a duo walk claims matches none of its combination
    /// columns, or has no single-slot falsification among them.
    #[error(
        "the walked cell {games_of:?} does not resolve against the combination columns \
         {bands:?}; either it is not one of them or a single-slot falsification of it is \
         missing"
    )]
    CellNotACombo {
        /// The cell the walk's header claims.
        games_of: Vec<String>,
        /// The combination columns it carries.
        bands: Vec<String>,
    },

    /// Two arm pairs handed to the duo judge claim the same cell.
    ///
    /// The same walk fed twice would count one cell's evidence as two,
    /// with the verdict's "every cell" quantifier quietly weakened.
    #[error("two arm pairs claim the cell {label}; each cell's evidence enters once")]
    DuplicateCell {
        /// The duplicated cell's label.
        label: String,
    },

    /// The jouseki judge needs exactly two bands.
    ///
    /// Plan 06 registers a two-family design: one correct column, one
    /// wrong one. A third band would make "the wrong token" a choice,
    /// and a choice made after the walk is a knob; the plan has none,
    /// so the judge refuses rather than picking.
    #[error(
        "the jouseki walk carries {found} band(s) and the registered design has exactly two — \
         with more, which token counts as the wrong one becomes a post-hoc choice"
    )]
    JousekiNeedsTwoBands {
        /// Bands the walk carries.
        found: usize,
    },

    /// An arm's records predate the fields a statistic reads.
    ///
    /// `§5.1`'s third admission item. A version 1 or 2 walk reaching a
    /// run that needs `ce` means a stale binary wrote it, and every
    /// position in it would read as unscoreable — a `ce_legal` over
    /// nothing rather than a refusal, if the fields were probed instead
    /// of the version.
    #[error(
        "arm {arm:?} was walked at record format version {found}, and reading its cost per band \
         needs {needed} or later; a stale binary wrote it, and every position in it would \
         otherwise read as one the walk could not score"
    )]
    RecordsPredateScoring {
        /// Arm whose walk is too old.
        arm: &'static str,
        /// Version its header declares.
        found: u32,
        /// Oldest version that carries the fields.
        needed: u32,
    },

    /// A margin stratum caught no position for one of the arms, so the
    /// flip rate inside it is a rate of nothing.
    ///
    /// Reachable when the pooled margins are heavily tied — every cut
    /// point equal, so three of the four strata are empty — or when one
    /// arm's margins sit entirely on one side of a cut the pool placed
    /// from the other arm's.
    #[error(
        "stratum {stratum} of the pooled top-two margin holds no position for arm {arm}, so the \
         flip rate inside it is a rate of nothing; the cut points are quantiles of the two \
         compared arms' margins pooled, and heavily tied margins can leave a stratum empty"
    )]
    EmptyStratum {
        /// Which of the [`STRATA`] strata, counting from zero.
        stratum: usize,
        /// Arm that had nothing in it.
        arm: &'static str,
    },

    /// No position in the two compared arms carries a `top2_margin`, so
    /// there is nothing to cut strata on.
    #[error(
        "neither arm {a:?} nor arm {b:?} carries a top-two margin at any position, so the \
         strata have no cut points; the margin arrived at record format version 2"
    )]
    NoMargins {
        /// First arm of the pool.
        a: &'static str,
        /// Second arm of the pool.
        b: &'static str,
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

/// Check every canonical role of the **legality** experiment, on both
/// axes, before anything is computed.
///
/// A separate entry point from [`check_roles`] rather than a widening
/// of it. That one names `perpos` / `prefix` / `perpos-b` and the arms
/// here are `prefix` / `A` / `A-b` / `AB` / `AB-b`; bending it to take
/// either set would leave a check that passes on both and therefore
/// pins neither. The arms plan 02 already walked keep the check they
/// were judged under.
///
/// Both axes are verified for every arm:
///
/// - **encoding** is [`CondEncoding::Prefix`] for all five. `§3.1`
///   makes that a constraint and not a choice — a model carrying both
///   a conditioning table and a legality table is refused at
///   construction — so an arm here that is per-position conditioned is
///   not this experiment's arm at all.
/// - **legality as input** is true for exactly [`ARM_AB`] and
///   [`ARM_AB_B`]. This is the axis the whole set differs on, and until
///   record format version 4 a header could not state it: two walks
///   that differ in the only way that matters produced headers that
///   could not be told apart, and this check was blind to precisely the
///   swap it exists for.
/// - **the walk is new enough to have stated it** —
///   [`LEGALITY_AXIS_VERSION`]. Without this the axis is only half
///   checked, since a stale walk's default `false` is indistinguishable
///   from a recorded one and passes wherever `false` is what the role
///   wants.
///
/// What is still **not** distinguished, and cannot be: which of `A` and
/// `A-b` is which, or of `AB` and `AB-b`. They are two runs of one
/// recipe differing only in a shuffle seed, which nothing in a record
/// records. Exchanging a pair leaves the floor untouched — it is
/// symmetric in the pair — and makes the difference the other
/// replicate's, which is a second reading of one experiment rather than
/// a wrong one. See [`h21`].
///
/// # Errors
///
/// One of the five arms is missing, its conditioning does not fit its
/// role ([`StatError::WrongEncodingForRole`]), its walk predates the
/// header field that states the axis
/// ([`StatError::RecordsPredateLegalityAxis`]), or its legality axis
/// does not fit ([`StatError::WrongLegalInputForRole`]).
pub fn check_legality_roles(arms: &AlignedArms) -> Result<(), StatError> {
    require_kinds(arms, &LEGALITY_ROLES)
}

/// The survival experiment's four arms fit [`SURVIVAL_ROLES`], on both
/// axes.
///
/// The encoding axis is the one that separates the arms here, so a
/// swapped pair is caught by the first check rather than the third —
/// the reverse of [`check_legality_roles`], where every arm is prefix
/// and only the legality axis tells them apart.
///
/// The legality axis is nonetheless checked, and against `false` for all
/// four: plan 04 holds legality-as-input out of the experiment after
/// plan 03 measured it, so a legality checkpoint reaching any slot would
/// be a third condition nobody registered rather than a mislabelled arm.
///
/// # Errors
///
/// As [`check_legality_roles`], over four arms rather than five.
pub fn check_survival_roles(arms: &AlignedArms) -> Result<(), StatError> {
    require_kinds(arms, &SURVIVAL_ROLES)
}

/// Both axes of a checkpoint's kind, per arm.
///
/// Encoding first, so that a set handed over with the wrong experiment
/// entirely is reported as that rather than as a legality mismatch.
/// Then the format version, and only then the legality axis itself: a
/// walk older than [`LEGALITY_AXIS_VERSION`] has no answer on that axis
/// to disagree with, and reporting it as a mismatch would send a reader
/// looking for a file in the wrong slot when the fix is to walk it
/// again. This is also what makes the check symmetric — the value alone
/// refuses only the slots that require `true`.
fn require_kinds(
    arms: &AlignedArms,
    roles: &[(&'static str, CondEncoding, bool)],
) -> Result<(), StatError> {
    for (arm, want_encoding, want_legal) in roles {
        let header = &arms.walk(arm)?.header;
        if header.encoding != *want_encoding {
            return Err(StatError::WrongEncodingForRole {
                arm,
                want: *want_encoding,
                found: header.encoding,
            });
        }
        if header.version < LEGALITY_AXIS_VERSION {
            return Err(StatError::RecordsPredateLegalityAxis {
                arm,
                found: header.version,
                needed: LEGALITY_AXIS_VERSION,
            });
        }
        if header.legal_input != *want_legal {
            return Err(StatError::WrongLegalInputForRole {
                arm,
                want: *want_legal,
                found: header.legal_input,
            });
        }
    }
    Ok(())
}

/// `§5.1` item 3: every named arm's records carry `ce` and
/// `top2_margin`.
///
/// Asked of [`crate::chess::records::WalkHeader::version`] and not of
/// the fields. Their `None` is a statement about a position the walk
/// could not score; a walk from a stale binary has no such field at
/// all, and probing the fields would read that whole walk as a walk of
/// unscoreable positions — a `ce_legal` over nothing, or a stratum
/// boundary drawn from no margins, rather than a refusal.
///
/// # Errors
///
/// A named arm is missing, or its walk predates
/// [`SCOREABLE_VERSION`].
pub fn check_scoreable(arms: &AlignedArms, roles: &[&'static str]) -> Result<(), StatError> {
    for arm in roles {
        let found = arms.walk(arm)?.header.version;
        if found < SCOREABLE_VERSION {
            return Err(StatError::RecordsPredateScoring {
                arm,
                found,
                needed: SCOREABLE_VERSION,
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

/// What the played move cost, in nats, meaned over the bands.
///
/// The mean over bands is pinned by `§4` rather than left to the
/// caller, because the three bands span 0.047 nats on one of plan 02's
/// arm-months — the same order as the seed gap the criteria are floored
/// by — so leaving the choice open would leave a choice on the floor's
/// own scale.
///
/// `None` where the played move is not in the vocabulary, so there was
/// no probability to take a log of. Absent rather than zero, as
/// [`top1_mean`] is: a cost of zero is a certainty, and counting an
/// unscoreable position as one would drag the figure down by however
/// many of them there were.
fn ce_mean(at: &GammaRecord) -> Option<f64> {
    let per_band = at.ce.as_ref()?;
    if per_band.is_empty() {
        return None;
    }
    Some(per_band.iter().sum::<f64>() / per_band.len() as f64)
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

/// Cross-entropy on the played move over the legal moves, meaned over
/// the bands and then over the positions of the resampled games, with a
/// cluster bootstrap.
///
/// The per-arm figure `§4`'s `ce_legal(arm)` names. Reported on its own
/// for a reader; the hypotheses do **not** subtract two of these, since
/// two separately bootstrapped quantities carry no joint distribution —
/// see [`h21`], which recomputes both sides inside one draw.
///
/// # Errors
///
/// The arm or gamma is not among those walked, or the resampling
/// refused. In particular a walk whose records predate
/// [`SCOREABLE_VERSION`] carries no cost at any position and arrives
/// here as an undefined statistic; [`check_scoreable`] is what turns
/// that into a message about the run.
pub fn ce_legal(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Interval, StatError> {
    let gamma_ix = arms.gamma_index(gamma)?;
    let cost = tally(arms, arm, gamma_ix, ce_mean)?;
    Ok(cluster_bootstrap(arms.games(), draws, seed, |draw| {
        cost.mean_over(draw)
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

/// The legality experiment's only computed admission gate: mean top-1
/// match, as a difference against the control.
///
/// `§5.1` item 2, at [`TOP1_GATE_LEGALITY`]. A separate function from
/// [`gate_top1`] rather than a parameter added to it: the tolerance is
/// pre-registered per experiment, and a tolerance a caller passes in is
/// a tolerance that can be raised once the difference has been seen.
/// The arms plan 02 was judged on keep [`TOP1_GATE`].
///
/// # What this set does *not* gate on
///
/// **Competence.** Plan 02's `ce_legal < ce_uniform` is deliberately
/// absent, and its absence is a finding rather than an omission
/// (`§2.1`): it fails on all six of plan 02's arm-months, it tests
/// calibration while being read as competence, and any full-vocabulary
/// objective fails it structurally — so carrying it here would exclude
/// the control. It must not reappear under another name, which is why
/// this doc says so rather than leaving the gate merely missing.
///
/// **Legal mass.** [`gate_legal_mass`] is not part of this set's
/// admission. The reason recorded when this was written — that a model
/// handed the legal set can put all its mass there trivially, so the
/// quantity stops discriminating — **is not what happened**: at γ=1 the
/// arms measured 0.0066 / 0.0064 without the legality input and
/// 0.0068 / 0.0070 with it, a gap smaller than either pair's seed gap
/// [実測: `rec-plan03`, both months]. The conclusion stands for a
/// different reason. Under a loss taken over the legal moves nothing
/// trains the full-vocabulary softmax, so no gradient pushes mass onto
/// the legal set and none pushes it off; the quantity is unconstrained
/// rather than saturated. It therefore also **cannot** serve as evidence
/// that the legality input is or is not being read, which is a use it
/// was put to once and should not be again — an ablation at inference
/// is the measurement that answers that.
///
/// The baseline is guarded as in [`gate_top1`], on both axes: the
/// control has to be prefix-conditioned **and** to have had no legality
/// input, since a legality checkpoint in the control slot would make
/// every row here a difference against a treatment arm.
///
/// # Errors
///
/// As [`gate_top1`], with the baseline's legality axis checked as well
/// — including that its walk was new enough to state one
/// ([`LEGALITY_AXIS_VERSION`]).
pub fn gate_top1_legality(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    require_kinds(arms, &[(CONTROL, CondEncoding::Prefix, false)])?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, arm, gamma_ix, top1_mean)?;
    let baseline = tally(arms, CONTROL, gamma_ix, top1_mean)?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(subject.mean_over(draw)? - baseline.mean_over(draw)?)
    })?;
    Ok(Gate {
        interval,
        tolerance: TOP1_GATE_LEGALITY,
    })
}

/// The survival experiment's admission gate: mean top-1 match, as a
/// difference against [`ARM_A`] — the arm being compared, not a control.
///
/// `plan 04 §3`, at [`TOP1_GATE_SURVIVAL`]. A third function rather than
/// a tolerance parameter, for the reason [`gate_top1_legality`] gives:
/// a tolerance a caller passes in is one that can be raised after the
/// difference is known.
///
/// # Why the reference moved
///
/// Plan 03's gate measured every arm against the control and excluded
/// all four, because the control trains a different objective and the
/// treatment's intended effect on top-1 is larger than the tolerance.
/// Both sides here carry the same objective and differ only in where the
/// band is attached, so the difference this bootstraps is one the
/// treatment is not designed to move.
///
/// The baseline is guarded on both axes, as [`gate_top1_legality`]'s is:
/// [`ARM_A`] has to be prefix-conditioned and to have had no legality
/// input. The subject is not guarded here for the same reason as there —
/// `arm` is a free string, and [`check_survival_roles`] is where the
/// whole set is known.
///
/// # Errors
///
/// As [`gate_top1_legality`], with [`ARM_A`] as the baseline.
pub fn gate_top1_survival(
    arms: &AlignedArms,
    arm: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    require_kinds(arms, &[(ARM_A, CondEncoding::Prefix, false)])?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, arm, gamma_ix, top1_mean)?;
    let baseline = tally(arms, ARM_A, gamma_ix, top1_mean)?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(subject.mean_over(draw)? - baseline.mean_over(draw)?)
    })?;
    Ok(Gate {
        interval,
        tolerance: TOP1_GATE_SURVIVAL,
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

/// How small an effect the walk that produced a verdict could have
/// separated.
///
/// `§5.3` fixes the walk at 6,200 positions for a minimum detectable
/// effect near 0.05 nats **at ~204 games**, and the game count is an
/// outcome rather than a setting: a position enters the sample when the
/// board offers at least two legal moves, games run 7 to 83 positions
/// with a coefficient of variation of 0.54, so 6,200 positions land
/// *near* 204 games rather than on it. A verdict read against the
/// plan's estimate is a verdict read against a number the run did not
/// have.
///
/// So this is carried beside every judged difference. An undetermined
/// result at a floor of 0.09 nats and one at 0.04 are different
/// findings — the first says the instrument was blunt, the second that
/// the effect is small — and the two are indistinguishable if only the
/// verdict is reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution {
    /// Games the interval actually resampled.
    pub games: usize,
    /// Smallest difference in nats this many games can separate.
    pub effect: f64,
    /// Games the curve is anchored at ([`PLANNED_GAMES`]).
    pub planned_games: usize,
    /// The floor `§5.3` measured, at that many games.
    ///
    /// Carried so a report can put the walk's own floor beside the one
    /// the plan was written against. They differ whenever the game
    /// count does, which is most of the time — the walk is sized in
    /// positions.
    pub planned_effect: f64,
}

impl Resolution {
    /// The resolution `games` games buy.
    ///
    /// # The curve, and where its one point came from
    ///
    /// `§5.3` gives a table rather than a formula:
    ///
    /// | games | effect |
    /// |---|---|
    /// | 99 | 0.0718 nats |
    /// | 180 | 0.0533 nats |
    /// | 204 | 0.0500 nats |
    /// | 1,080 | 0.0217 nats |
    ///
    /// One row of it is a measurement and the rest is a shape. The
    /// measured row is the first — the walk's own 99 games, where
    /// running the estimator over bootstrap draws put the floor at
    /// 0.0280 with a half-interval of 0.0438. A minimum detectable
    /// effect scales as `1/sqrt(n)` at a fixed paired standard
    /// deviation, whatever the power and level behind it, since those
    /// enter only as a constant; every other row is that scaling
    /// applied to that point.
    ///
    /// So reproducing the table is **not** two calculations agreeing,
    /// and is not offered as evidence that this floor is right. It says
    /// this reads the same curve the plan does — which is the property
    /// worth holding, because the number a verdict is judged against
    /// has to be the one pre-registered rather than one this file
    /// arrived at. The constant is written at 180 games rather than at
    /// the measured 99 because that is where `§4` quotes it, so the
    /// 99-game row comes back to a tenth of a percent rather than
    /// exactly: the table prints four decimals and the round trip
    /// through them is what that tenth of a percent is.
    ///
    /// It is a curve and not the calculation behind it, and what it
    /// reports is a **lower bound on the floor** rather than the floor.
    /// Two things it does not model — the t-quantile at small `n`, and
    /// any drift in positions per game — push the same way, so a walk's
    /// true resolution is no better than this and may be worse.
    ///
    /// `None` for zero games, where there is no resolution to report.
    /// Unreachable from a built [`AlignedArms`], which refuses a set
    /// with no positions, and total anyway.
    pub fn at(games: usize) -> Option<Self> {
        (games > 0).then(|| Self {
            games,
            effect: PLANNED_EFFECT * (PLANNED_GAMES as f64 / games as f64).sqrt(),
            planned_games: PLANNED_GAMES,
            planned_effect: PLANNED_EFFECT,
        })
    }

    /// Whether an effect of this size is at or above the resolution.
    ///
    /// Compared on magnitude, so it governs a refutation exactly as it
    /// governs a confirmation: an effect too small to separate is too
    /// small in either direction.
    pub fn resolves(&self, effect: f64) -> bool {
        effect.abs() >= self.effect
    }
}

/// H21: does handing the model the legal set add what masking the loss
/// does not?
///
/// `D = ce_legal(A) - ce_legal(AB)`, positive meaning `AB` puts more
/// probability on the move a human played. The two arms train on the
/// same objective and differ only in whether the legal set reaches the
/// forward pass, which is what makes this the one judged comparison in
/// the plan whose direction is not already known (`§4`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H21 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// `ce_legal(A)` on the sample as walked.
    pub ce_a: f64,
    /// `ce_legal(A-b)`, the replicate of `A`.
    pub ce_a_b: f64,
    /// `ce_legal(AB)`.
    pub ce_ab: f64,
    /// `ce_legal(AB-b)`, the replicate of `AB`.
    pub ce_ab_b: f64,
    /// `D`, on the sample as walked.
    pub difference: f64,
    /// `F`, on the sample as walked: the larger of the two same-recipe
    /// seed gaps.
    pub floor: f64,
    /// Interval on `D - F`. Confirmed when it excludes zero from above.
    pub confirm: Interval,
    /// Interval on `D + F`. Refuted when it excludes zero from below.
    pub refute: Interval,
    /// What the walk that produced this could separate.
    pub resolution: Resolution,
    /// Positions the arms share.
    pub positions: usize,
    /// Games those positions came from — the clusters resampled.
    pub games: usize,
}

impl H21 {
    /// Whether the effect is large enough for this walk to have
    /// separated it.
    ///
    /// `§4` fixes a minimum detectable effect and says an effect below
    /// it reads undetermined **however clean the point estimate looks**
    /// — recorded in the hypothesis so that a result document cannot
    /// report a margin under the floor as a near miss. The floor is
    /// read off the games actually resampled rather than off the plan's
    /// estimate of them ([`Resolution`]).
    pub fn resolved(&self) -> bool {
        self.resolution.resolves(self.difference)
    }

    /// Whether the confirm interval clears zero from above, at an
    /// effect this walk could separate.
    pub fn confirmed(&self) -> bool {
        self.resolved() && self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below, at an effect
    /// this walk could separate.
    pub fn refuted(&self) -> bool {
        self.resolved() && self.refute.excludes_zero_from_below()
    }

    /// The verdict on one month, in the three-way form the decision
    /// table partitions outcomes with. A month is only judged when both
    /// months agree, which is the caller's business, not this type's.
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H21 from the four treatment arms.
///
/// # Why the two criteria are mirrored
///
/// `D - F` for the confirmation and `D + F` for the refutation, as
/// [`h14`]'s are. Under the null that the two arms predict equally well
/// `E[D] = 0` while `F` is a strictly positive gap between two real
/// checkpoints, so `D - F` converges to something negative: a refute
/// branch keyed on it would fire whenever `AB` merely failed to win,
/// including when the two arms were exactly equal. `D + F` asks the
/// intended question — worse by more than both observed same-recipe
/// gaps.
///
/// # The floor is matched, not borrowed
///
/// `F = max(|AB - AB-b|, |A - A-b|)`, recomputed inside every draw.
/// Both sides of the comparison are replicated, so both gaps are
/// observed and neither side's floor is inferred from the other's.
/// Nothing is frozen as a scalar: a gap held constant across the draws
/// would report a precision one pair of runs does not carry.
///
/// A max over two remains a max over two. Under exchangeability a third
/// run exceeds it about a third of the time, so a confirmed H21 reads
/// **"the margin exceeds both observed same-recipe gaps"** and never
/// "clears the seed floor".
///
/// # What a swap between a pair's two runs does
///
/// [`ARM_A`] and [`ARM_A_B`] are two runs of one recipe, and nothing in
/// a record says which is which. Exchanging them leaves `F` alone — it
/// is symmetric in the pair — and makes `D` the other replicate's
/// difference against `AB`: two equally good readings of one
/// experiment. The same holds for the `AB` pair. A swap **across** the
/// pairs is a different matter entirely, reverses the sign of `D`, and
/// is what [`check_legality_roles`] refuses.
///
/// # Errors
///
/// One of the four arms is missing, an arm's checkpoint does not fit
/// its role on either axis or its walk is too old to state the second
/// one ([`LEGALITY_AXIS_VERSION`]), the gamma was not swept, or the
/// resampling refused. A walk that predates [`SCOREABLE_VERSION`]
/// carries no cost and arrives as an undefined statistic — call
/// [`check_scoreable`] first for a message that says so.
pub fn h21(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H21, StatError> {
    require_kinds(
        arms,
        &[
            (ARM_A, CondEncoding::Prefix, false),
            (ARM_A_B, CondEncoding::Prefix, false),
            (ARM_AB, CondEncoding::Prefix, true),
            (ARM_AB_B, CondEncoding::Prefix, true),
        ],
    )?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let a = tally(arms, ARM_A, gamma_ix, ce_mean)?;
    let a_b = tally(arms, ARM_A_B, gamma_ix, ce_mean)?;
    let ab = tally(arms, ARM_AB, gamma_ix, ce_mean)?;
    let ab_b = tally(arms, ARM_AB_B, gamma_ix, ce_mean)?;

    // One draw, every term. A caller cannot reach a second resample
    // from inside this, so the difference and the floor cannot be
    // estimated against different game lists.
    let difference_and_floor = |draw: &[usize]| -> Option<(f64, f64)> {
        let plain = a.mean_over(draw)?;
        let plain_b = a_b.mean_over(draw)?;
        let legal = ab.mean_over(draw)?;
        let legal_b = ab_b.mean_over(draw)?;
        Some((
            plain - legal,
            (legal - legal_b).abs().max((plain - plain_b).abs()),
        ))
    };

    let confirm = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        difference_and_floor(draw).map(|(d, f)| d - f)
    })?;
    let refute = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        difference_and_floor(draw).map(|(d, f)| d + f)
    })?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    let (difference, floor) =
        difference_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
    // Unreachable after the bootstraps above, which refuse a zero
    // cluster count outright; total rather than indexed into.
    let resolution = Resolution::at(arms.games()).ok_or(BootstrapError::NoClusters)?;
    Ok(H21 {
        gamma,
        ce_a: a.total().mean().unwrap_or(f64::NAN),
        ce_a_b: a_b.total().mean().unwrap_or(f64::NAN),
        ce_ab: ab.total().mean().unwrap_or(f64::NAN),
        ce_ab_b: ab_b.total().mean().unwrap_or(f64::NAN),
        difference,
        floor,
        confirm,
        refute,
        resolution,
        positions: arms.positions(),
        games: arms.games(),
    })
}

/// Plan 04's only judged hypothesis: does the conditioning advantage
/// survive the loss mask?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H22 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// Flip rate of [`ARM_P`], on the sample as walked.
    pub flip_p: f64,
    /// Flip rate of [`ARM_P_B`], the replicate of `P`.
    pub flip_p_b: f64,
    /// Flip rate of [`ARM_A`].
    pub flip_a: f64,
    /// Flip rate of [`ARM_A_B`], the replicate of `A`.
    pub flip_a_b: f64,
    /// `M = flip(P) - flip(A)`, on the sample as walked.
    pub margin: f64,
    /// `F`, the larger of the two same-recipe seed gaps.
    pub floor: f64,
    /// Interval on `M - F`. Confirmed when it excludes zero from above.
    pub confirm: Interval,
    /// Interval on `M + F`. Refuted when it excludes zero from below.
    pub refute: Interval,
    /// Positions the arms share.
    pub positions: usize,
    /// Games those positions came from — the clusters resampled.
    pub games: usize,
}

impl H22 {
    /// Whether the confirm interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below.
    ///
    /// **Hard to reach by construction, and the plan says so before any
    /// arm was baked.** `F` is a positive gap between two real
    /// checkpoints, so `M + F < 0` needs the per-position arm to flip
    /// *less* than the prefix arm by more than both observed seed gaps
    /// — not the shape the axis dying would take. What dying looks like
    /// here is [`Self::verdict`] returning `"undetermined"`, and plan 04
    /// §3 registers that reading in advance so it cannot be presented
    /// afterwards as a near miss.
    pub fn refuted(&self) -> bool {
        self.refute.excludes_zero_from_below()
    }

    /// The verdict on one month, in the three-way form the decision
    /// table partitions outcomes with.
    ///
    /// `"undetermined"` is the load-bearing outcome here rather than the
    /// leftover one — see [`Self::refuted`].
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H22 from the four survival arms.
///
/// [`h14`] with its floor repaired and its arms re-cast. The margin is
/// the same quantity — per-position's flip rate less prefix's — but
/// measured where the loss was taken over the legal moves, which is
/// where plan 02 never measured it.
///
/// # Why the floor is a max over both pairs
///
/// [`h14`] took `G = |flip(perpos) - flip(perpos-b)|`: the treatment
/// pair's gap, standing in for the prefix side's as well. Plan 03 §3.4
/// calls that a borrowed floor and both sides are replicated here, so
/// `F = max(|flip(P) - flip(P-b)|, |flip(A) - flip(A-b)|)` as [`h21`]'s
/// is. The prefix pair's gap is not small — 0.062 and 0.054 on plan 03's
/// records — so borrowing it would have mattered.
///
/// Recomputed inside every draw, like every other term: a gap frozen as
/// a scalar would report a precision one pair of runs does not carry.
///
/// # What the floor does *not* shrink with
///
/// Positions. `F` is a distance between two checkpoints, so walking more
/// of the holdout pins it down without making it smaller — which is why
/// plan 04 §5 spends nothing on a larger walk and why more seeds would
/// widen it rather than tighten it.
///
/// # What a swap within a pair does
///
/// Nothing that matters, as in [`h14`] and [`h21`]: `F` is symmetric in
/// each pair, and `M` becomes the other replicate's margin — another
/// reading of the same experiment. A swap **across** the pairs reverses
/// the sign of `M`, and that is what [`check_survival_roles`] refuses,
/// on the encoding axis rather than the legality one.
///
/// # Errors
///
/// One of the four arms is missing, an arm's checkpoint does not fit its
/// role on either axis or its walk is too old to state the second one
/// ([`LEGALITY_AXIS_VERSION`]), the gamma was not swept, or the
/// resampling refused.
pub fn h22(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H22, StatError> {
    require_kinds(arms, &SURVIVAL_ROLES)?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let p = tally(arms, ARM_P, gamma_ix, flipped)?;
    let p_b = tally(arms, ARM_P_B, gamma_ix, flipped)?;
    let a = tally(arms, ARM_A, gamma_ix, flipped)?;
    let a_b = tally(arms, ARM_A_B, gamma_ix, flipped)?;

    // One draw, every term, as in `h14` and `h21`.
    let margin_and_floor = |draw: &[usize]| -> Option<(f64, f64)> {
        let per = p.mean_over(draw)?;
        let per_b = p_b.mean_over(draw)?;
        let pre = a.mean_over(draw)?;
        let pre_b = a_b.mean_over(draw)?;
        Some((per - pre, (per - per_b).abs().max((pre - pre_b).abs())))
    };

    let confirm = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(m, f)| m - f)
    })?;
    let refute = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(m, f)| m + f)
    })?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    let (margin, floor) = margin_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
    Ok(H22 {
        gamma,
        flip_p: p.total().mean().unwrap_or(f64::NAN),
        flip_p_b: p_b.total().mean().unwrap_or(f64::NAN),
        flip_a: a.total().mean().unwrap_or(f64::NAN),
        flip_a_b: a_b.total().mean().unwrap_or(f64::NAN),
        margin,
        floor,
        confirm,
        refute,
        positions: arms.positions(),
        games: arms.games(),
    })
}

/// Plan 04's secondary hypothesis: does the advantage still reach the
/// deep positions once the loss is masked?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H23 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// The opening bucket, shared by every arm.
    pub shallow: Bucket,
    /// The deep bucket, shared by every arm.
    pub deep: Bucket,
    /// `JS(deep) / JS(shallow)` for [`ARM_P`]. Higher is a shallower
    /// decay.
    pub ratio_p: f64,
    /// The same for [`ARM_P_B`].
    pub ratio_p_b: f64,
    /// The same for [`ARM_A`].
    pub ratio_a: f64,
    /// The same for [`ARM_A_B`].
    pub ratio_a_b: f64,
    /// `R = ratio(P) - ratio(A)`, on the sample as walked.
    pub margin: f64,
    /// `F_R`, the larger of the two same-recipe seed gaps in the ratio.
    pub floor: f64,
    /// Interval on `R - F_R`. Confirmed when it excludes zero from
    /// above.
    pub confirm: Interval,
    /// Interval on `R + F_R`. Refuted when it excludes zero from below.
    pub refute: Interval,
}

impl H23 {
    /// Whether the confirm interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below.
    pub fn refuted(&self) -> bool {
        self.refute.excludes_zero_from_below()
    }

    /// As [`H22::verdict`].
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H23 from the four survival arms.
///
/// [`h15`]'s quantity — the ratio of deep divergence to shallow — with
/// a floor, which [`h15`] has none of.
///
/// # Why a floor here and not there
///
/// Plan 02 replicated one side only, so there was no second reading of
/// the prefix arm to measure a gap against and the difference had to
/// stand alone. Both sides are replicated here, so the same rule the
/// rest of plan 04 runs on applies (`§3.4`, the floor is matched rather
/// than borrowed).
///
/// The plan's own text said "the same way of drawing the floor as
/// plan 02's H15", which does not name a rule since that hypothesis
/// draws none. The disambiguation is recorded in plan 04 §2, written
/// after H22's verdict and **before this was computed** — H22's numbers
/// are not inputs to this statistic, so fixing the reading between them
/// is not a criterion moved to fit a result. The conservative side was
/// taken: a floor makes confirming harder.
///
/// # Errors
///
/// One of the four arms is missing, an arm's checkpoint does not fit
/// its role, the gamma was not swept, a bucket caught no positions, or
/// the resampling refused.
pub fn h23(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H23, StatError> {
    require_kinds(arms, &SURVIVAL_ROLES)?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let shallow_range = SHALLOW_BUCKET;
    let deep_range = deep_bucket(arms.ctx());

    let bucketed =
        |arm: &str, range: (usize, usize)| tally_in_bucket(arms, arm, gamma_ix, range, widest_js);
    let p_shallow = bucketed(ARM_P, shallow_range)?;
    let p_deep = bucketed(ARM_P, deep_range)?;
    let p_b_shallow = bucketed(ARM_P_B, shallow_range)?;
    let p_b_deep = bucketed(ARM_P_B, deep_range)?;
    let a_shallow = bucketed(ARM_A, shallow_range)?;
    let a_deep = bucketed(ARM_A, deep_range)?;
    let a_b_shallow = bucketed(ARM_A_B, shallow_range)?;
    let a_b_deep = bucketed(ARM_A_B, deep_range)?;

    if p_shallow.total().n == 0 {
        return Err(StatError::EmptyBucket {
            low: shallow_range.0,
            high: shallow_range.1.saturating_sub(1),
            role: "denominator",
        });
    }
    if p_deep.total().n == 0 {
        return Err(StatError::EmptyBucket {
            low: deep_range.0,
            high: deep_range.1.saturating_sub(1),
            role: "numerator",
        });
    }

    // Divergence is non-negative, so a shallow mean of exactly zero
    // means the bands agreed everywhere in the opening — at which point
    // the decay has nothing to decay from, as in `h15`.
    let ratio = |deep: &ClusterTally, shallow: &ClusterTally, draw: &[usize]| -> Option<f64> {
        let denominator = shallow.mean_over(draw)?;
        if denominator <= 0.0 {
            return None;
        }
        Some(deep.mean_over(draw)? / denominator)
    };

    // One draw, every term.
    let margin_and_floor = |draw: &[usize]| -> Option<(f64, f64)> {
        let p = ratio(&p_deep, &p_shallow, draw)?;
        let p_b = ratio(&p_b_deep, &p_b_shallow, draw)?;
        let a = ratio(&a_deep, &a_shallow, draw)?;
        let a_b = ratio(&a_b_deep, &a_b_shallow, draw)?;
        Some((p - a, (p - p_b).abs().max((a - a_b).abs())))
    };

    let confirm = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(r, f)| r - f)
    })?;
    let refute = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(r, f)| r + f)
    })?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    let (margin, floor) = margin_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
    Ok(H23 {
        gamma,
        shallow: Bucket {
            low: shallow_range.0,
            high: shallow_range.1.saturating_sub(1),
            positions: p_shallow.total().n,
            games: p_shallow.clusters_present(),
        },
        deep: Bucket {
            low: deep_range.0,
            high: deep_range.1.saturating_sub(1),
            positions: p_deep.total().n,
            games: p_deep.clusters_present(),
        },
        ratio_p: ratio(&p_deep, &p_shallow, &whole).unwrap_or(f64::NAN),
        ratio_p_b: ratio(&p_b_deep, &p_b_shallow, &whole).unwrap_or(f64::NAN),
        ratio_a: ratio(&a_deep, &a_shallow, &whole).unwrap_or(f64::NAN),
        ratio_a_b: ratio(&a_b_deep, &a_b_shallow, &whole).unwrap_or(f64::NAN),
        margin,
        floor,
        confirm,
        refute,
    })
}

/// The jouseki arms fit [`JOUSEKI_ROLES`] and this pair of walks is the
/// family it claims to be. Returns the (correct, wrong) band columns.
///
/// Three refusals, in the order a reader would want them: the
/// checkpoint axes ([`require_kinds`], as every judge here), the walk
/// filter's *statability* ([`WALK_FILTER_VERSION`] — a pre-5 walk
/// cannot say whose games it walked, in either direction), and the
/// walk filter's *value* — `games_of` must be exactly the one family
/// token this pair of walks is being read as. The band list must carry
/// exactly two tokens ([`StatError::JousekiNeedsTwoBands`]), and the
/// family must be one of them; the other is the wrong column.
///
/// What a swap does: feed the C-games walks where the B-games walks
/// belong and the cost is read against the wrong family — sign
/// reversed, every figure well-formed. That is the swap the `games_of`
/// comparison refuses, and the reason the field exists at all.
pub fn check_jouseki_roles(arms: &AlignedArms, family: &str) -> Result<(usize, usize), StatError> {
    require_kinds(arms, &JOUSEKI_ROLES)?;
    for arm in JOUSEKI_ARMS {
        let header = &arms.walk(arm)?.header;
        if header.version < WALK_FILTER_VERSION {
            return Err(StatError::RecordsPredateWalkFilter {
                arm,
                found: header.version,
                needed: WALK_FILTER_VERSION,
            });
        }
        if header.games_of.as_deref() != Some(std::slice::from_ref(&family.to_string())) {
            return Err(StatError::WrongWalkedGamesForRole {
                arm,
                want: family.to_string(),
                found: match &header.games_of {
                    Some(tokens) => format!("{tokens:?}"),
                    None => "(unstated)".to_string(),
                },
            });
        }
    }
    let bands = &arms.walk(ARM_J)?.header.bands;
    if bands.len() != 2 {
        return Err(StatError::JousekiNeedsTwoBands { found: bands.len() });
    }
    let correct =
        bands
            .iter()
            .position(|b| b == family)
            .ok_or_else(|| StatError::FamilyTokenNotABand {
                token: family.to_string(),
                bands: bands.clone(),
            })?;
    Ok((correct, 1 - correct))
}

/// One family's half of H29: the mis-conditioning cost of the walk's
/// own family, floored by the pair's seed gap.
#[derive(Debug, Clone, PartialEq)]
pub struct H29Family {
    /// The family token the walked games belong to.
    pub family: String,
    /// `cost = top1(correct token) - top1(wrong token)` for [`ARM_J`],
    /// on the sample as walked.
    pub cost_j: f64,
    /// The same quantity for [`ARM_J_B`], the seed replicate.
    pub cost_j_b: f64,
    /// `F = |cost(J) - cost(J-b)|`, the pair's seed gap.
    ///
    /// One pair rather than [`h22`]'s two, and not a borrowed floor:
    /// borrowing is standing one recipe's gap in for another's, and
    /// H29 has one recipe — the comparison runs within each model,
    /// between its own two condition columns.
    pub floor: f64,
    /// Interval on `cost(J) - F`. This family confirms when it
    /// excludes zero from above.
    pub confirm: Interval,
    /// Interval on `cost(J) + F`. This family refutes when it excludes
    /// zero from below.
    pub refute: Interval,
    /// `cost(J)` over ply 0-19 only — description, no verdict. `None`
    /// when the stratum caught no scoreable position.
    pub shallow_cost_j: Option<f64>,
    /// `cost(J)` over ply 20 and deeper — description, no verdict.
    pub deep_cost_j: Option<f64>,
    /// Positions the two arms share.
    pub positions: usize,
    /// Games those positions came from — the clusters resampled.
    pub games: usize,
}

impl H29Family {
    /// Whether the confirm interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below.
    pub fn refuted(&self) -> bool {
        self.refute.excludes_zero_from_below()
    }
}

/// One family's cost, floor and intervals.
///
/// The cost is per position `top1[correct] - top1[wrong]` — `+1` where
/// only the walk's own family token ranks the played move first, `-1`
/// where only the wrong token does, `0` where they agree — meaned over
/// the positions, so the family-level figure is the familiar
/// difference of top-1 rates. `None` positions (a played move outside
/// the vocabulary) drop from both columns at once, since both are read
/// off the same record.
///
/// Floor and margin are recomputed inside every draw, as [`h22`]'s
/// are, and for the same reason: a gap frozen as a scalar would report
/// a precision one pair of runs does not carry.
///
/// # Errors
///
/// Whatever [`check_jouseki_roles`] refuses, a gamma that was not
/// swept, or the resampling refusing.
pub fn h29_family(
    arms: &AlignedArms,
    family: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29Family, StatError> {
    jouseki_family_in(arms, family, None, gamma, draws, seed)
}

/// [`h29_family`]'s statistic restricted to the shallow stratum —
/// H31's per-family half.
///
/// The domain is a parameter of the computation and not of the type:
/// what comes back is still "a cost, its floor, and the intervals on
/// both sides", read over ply 0 through 19 only. The two descriptive
/// stratum fields keep their meaning (the shallow one now restates the
/// judged quantity; the deep one shows what was excluded).
///
/// Registered by plan 07 because plan 06's stratum description showed
/// the cost concentrating where an opening lives, and the all-ply mean
/// dilutes it by roughly the share of plies past 19 — but that
/// observation was made on months 2026-05 and 2026-04, so those months
/// are design input here and the verdict is taken on months the
/// criterion never saw.
pub fn h31_family(
    arms: &AlignedArms,
    family: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29Family, StatError> {
    jouseki_family_in(arms, family, Some(JOUSEKI_SHALLOW), gamma, draws, seed)
}

/// The shared computation: cost, floor and intervals over the whole
/// walk (`bucket: None`) or one ply range.
fn jouseki_family_in(
    arms: &AlignedArms,
    family: &str,
    bucket: Option<(usize, usize)>,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29Family, StatError> {
    let (correct, wrong) = check_jouseki_roles(arms, family)?;
    cost_family_between(
        arms, ARM_J, ARM_J_B, family, correct, wrong, bucket, gamma, draws, seed,
    )
}

/// The cost-with-floor computation both the jouseki and the duo judges
/// share: `top1[correct] - top1[wrong]` for `arm_a`, floored by its gap
/// to `arm_b`, over the whole walk or one ply range.
///
/// A free function over indices rather than a method over roles,
/// because the two judges resolve their indices differently (a family
/// token against two bands; a cell against four combinations) and the
/// arithmetic below is identical once they have.
#[allow(clippy::too_many_arguments)]
fn cost_family_between(
    arms: &AlignedArms,
    arm_a: &'static str,
    arm_b: &'static str,
    label: &str,
    correct: usize,
    wrong: usize,
    bucket: Option<(usize, usize)>,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29Family, StatError> {
    let gamma_ix = arms.gamma_index(gamma)?;
    let cost = move |at: &GammaRecord| -> Option<f64> {
        let per_band = at.top1.as_ref()?;
        let c = *per_band.get(correct)?;
        let w = *per_band.get(wrong)?;
        Some((c as i64 - w as i64) as f64)
    };
    let judged = bucket.unwrap_or((0, usize::MAX));
    let j = tally_in_bucket(arms, arm_a, gamma_ix, judged, cost)?;
    let j_b = tally_in_bucket(arms, arm_b, gamma_ix, judged, cost)?;
    if j.total().n == 0 {
        return Err(StatError::EmptyBucket {
            low: judged.0,
            high: judged.1.saturating_sub(1),
            role: "cost",
        });
    }

    // One draw, every term, as in `h22`.
    let margin_and_floor = |draw: &[usize]| -> Option<(f64, f64)> {
        let a = j.mean_over(draw)?;
        let b = j_b.mean_over(draw)?;
        Some((a, (a - b).abs()))
    };
    let confirm = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(m, f)| m - f)
    })?;
    let refute = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        margin_and_floor(draw).map(|(m, f)| m + f)
    })?;
    let whole: Vec<usize> = (0..arms.games()).collect();
    let (cost_j, floor) = margin_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;

    let deep_range = (JOUSEKI_SHALLOW.1, usize::MAX);
    let shallow = tally_in_bucket(arms, arm_a, gamma_ix, JOUSEKI_SHALLOW, cost)?;
    let deep = tally_in_bucket(arms, arm_a, gamma_ix, deep_range, cost)?;

    Ok(H29Family {
        family: label.to_string(),
        cost_j,
        cost_j_b: j_b.total().mean().unwrap_or(f64::NAN),
        floor,
        confirm,
        refute,
        shallow_cost_j: shallow.total().mean(),
        deep_cost_j: deep.total().mean(),
        positions: j.total().n,
        games: j.clusters_present(),
    })
}

/// Plan 06's primary hypothesis: does a jouseki token steer toward its
/// own family's games?
#[derive(Debug, Clone, PartialEq)]
pub struct H29 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// The two families, each judged on its own walks.
    pub families: [H29Family; 2],
}

impl H29 {
    /// The verdict on one month.
    ///
    /// Confirmed only when **both** families confirm — plan 06 §3's
    /// min-logic, taken at the verdict layer because the two families'
    /// walks are different position streams with different clusters,
    /// so a single interval on the min would need a joint resample the
    /// registered statistic does not define. A result where only one
    /// family's token steers is exactly what the min exists to keep
    /// out of a confirmation. Refuted when either family refutes;
    /// everything else is undetermined.
    pub fn verdict(&self) -> &'static str {
        let all_confirmed = self.families.iter().all(H29Family::confirmed);
        let any_refuted = self.families.iter().any(H29Family::refuted);
        match (all_confirmed, any_refuted) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H29 from the two families' arm pairs.
///
/// Two [`AlignedArms`] rather than one because the families' walks are
/// different games — there is no shared position stream for one
/// alignment to hold, and each family's bootstrap resamples its own
/// clusters.
///
/// # Errors
///
/// As [`h29_family`], for either family; or the two pairs disagree on
/// the band list, which would mean two different checkpoints' walks
/// were mixed — [`StatError::FamilyTokenNotABand`] would catch most
/// such mixes late and confusingly, so the lists are compared first.
pub fn h29(
    arms_by_family: [(&AlignedArms, &str); 2],
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29, StatError> {
    jouseki_in(arms_by_family, None, gamma, draws, seed)
}

/// Plan 07's primary hypothesis: [`h29`]'s statistic over ply 0-19
/// only, with the same admission, floor construction and verdict-layer
/// min-logic. See [`h31_family`] for what moves and what does not.
pub fn h31(
    arms_by_family: [(&AlignedArms, &str); 2],
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29, StatError> {
    jouseki_in(arms_by_family, Some(JOUSEKI_SHALLOW), gamma, draws, seed)
}

fn jouseki_in(
    arms_by_family: [(&AlignedArms, &str); 2],
    bucket: Option<(usize, usize)>,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H29, StatError> {
    let [(arms_a, family_a), (arms_b, family_b)] = arms_by_family;
    let bands_a = &arms_a.walk(ARM_J)?.header.bands;
    let bands_b = &arms_b.walk(ARM_J)?.header.bands;
    if bands_a != bands_b {
        return Err(StatError::FamilyTokenNotABand {
            token: family_b.to_string(),
            bands: bands_a.clone(),
        });
    }
    Ok(H29 {
        gamma,
        families: [
            jouseki_family_in(arms_a, family_a, bucket, gamma, draws, seed)?,
            jouseki_family_in(arms_b, family_b, bucket, gamma, draws, seed)?,
        ],
    })
}

/// The jouseki admission gate: [`ARM_J_B`]'s mean top-1 match as a
/// difference against [`ARM_J`]'s, at [`TOP1_GATE_JOUSEKI`].
///
/// The reference is the seed replicate because there is nothing else:
/// no control, no treatment arm. The two runs differ in the shuffle
/// seed alone, so a difference outside the band is a run defect — a
/// wrong corpus or step count in one slot — rather than an effect,
/// and both passing and failing are reachable (the same-recipe seed
/// gap was measured at 0.0123 / 0.0112 against a 0.03 tolerance).
///
/// # Errors
///
/// As [`check_jouseki_roles`], plus a gamma that was not swept or the
/// resampling refusing.
pub fn gate_top1_jouseki(
    arms: &AlignedArms,
    family: &str,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    check_jouseki_roles(arms, family)?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, ARM_J_B, gamma_ix, top1_mean)?;
    let baseline = tally(arms, ARM_J, gamma_ix, top1_mean)?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(subject.mean_over(draw)? - baseline.mean_over(draw)?)
    })?;
    Ok(Gate {
        interval,
        tolerance: TOP1_GATE_JOUSEKI,
    })
}

/// Duo experiment (plan 08), primary seed arm.
pub const ARM_K: &str = "K";

/// Second run of [`ARM_K`], differing only in the shuffle seed.
pub const ARM_K_B: &str = "K-b";

/// The duo experiment's two arms.
pub const DUO_ARMS: [&str; 2] = [ARM_K, ARM_K_B];

/// What both checkpoints must have been, on both axes — the
/// [`JOUSEKI_ROLES`] stance with the duo's arm names. Which **cell**
/// was walked is the axis a swap could show up on, and that is checked
/// against the records' own `games_of` by [`check_duo_roles`].
pub const DUO_ROLES: [(&str, CondEncoding, bool); 2] = [
    (ARM_K, CondEncoding::EveryPosition, false),
    (ARM_K_B, CondEncoding::EveryPosition, false),
];

/// The cell a pair of duo walks claims, resolved against the four
/// combination columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuoCell {
    /// The correct combination's label — the cell itself.
    pub label: String,
    /// Column of the correct combination.
    pub correct: usize,
    /// Per slot, in the label's order: the label and column of the
    /// combination that falsifies **that slot only**.
    pub wrong: [(String, usize); 2],
}

/// The duo arms fit [`DUO_ROLES`] and this pair of walks is the cell it
/// claims to be. Returns the resolved [`DuoCell`].
///
/// The walk's `bands` are combination labels — one band token per slot
/// joined with `+`, written by the walk in the order its batch ran them
/// — and its `games_of` carries the two tokens the cell's games were
/// narrowed to. The correct column is the label whose parts are exactly
/// those two tokens (compared as a set, since the walk's argument order
/// is the caller's); each wrong column shares one part with it and
/// differs in the other. A 2×2 design makes both unique, and anything
/// else — three combinations, a label of one part, a `games_of` no
/// label matches — is refused by name rather than resolved.
pub fn check_duo_roles(arms: &AlignedArms) -> Result<DuoCell, StatError> {
    require_kinds(arms, &DUO_ROLES)?;
    let header = &arms.walk(ARM_K)?.header;
    for arm in DUO_ARMS {
        let header = &arms.walk(arm)?.header;
        if header.version < WALK_FILTER_VERSION {
            return Err(StatError::RecordsPredateWalkFilter {
                arm,
                found: header.version,
                needed: WALK_FILTER_VERSION,
            });
        }
    }
    let cell_tokens = match &header.games_of {
        Some(tokens) if tokens.len() == 2 => tokens.clone(),
        other => {
            return Err(StatError::WrongWalkedGamesForRole {
                arm: ARM_K,
                want: "two cell tokens".to_string(),
                found: match other {
                    Some(tokens) => format!("{tokens:?}"),
                    None => "(unstated)".to_string(),
                },
            })
        }
    };
    // Both walks must claim the same cell; the alignment does not
    // compare this field, so it is compared here.
    let replicate = &arms.walk(ARM_K_B)?.header.games_of;
    if replicate.as_ref() != Some(&cell_tokens) {
        return Err(StatError::WrongWalkedGamesForRole {
            arm: ARM_K_B,
            want: format!("{cell_tokens:?}"),
            found: format!("{replicate:?}"),
        });
    }
    let bands = &header.bands;
    if bands.len() != 4 {
        return Err(StatError::DuoNeedsFourCombos { found: bands.len() });
    }
    let parts: Vec<Vec<&str>> = bands
        .iter()
        .map(|label| label.split('+').collect::<Vec<_>>())
        .collect();
    if parts.iter().any(|p| p.len() != 2) {
        return Err(StatError::DuoNeedsFourCombos { found: bands.len() });
    }
    let matches_cell = |p: &[&str]| {
        (p[0] == cell_tokens[0] && p[1] == cell_tokens[1])
            || (p[0] == cell_tokens[1] && p[1] == cell_tokens[0])
    };
    let correct =
        parts
            .iter()
            .position(|p| matches_cell(p))
            .ok_or_else(|| StatError::CellNotACombo {
                games_of: cell_tokens.clone(),
                bands: bands.clone(),
            })?;
    let correct_parts = &parts[correct];
    let mut wrong: Vec<(String, usize)> = Vec::with_capacity(2);
    for slot in 0..2 {
        let other = slot ^ 1;
        let found = parts.iter().enumerate().find(|(ix, p)| {
            *ix != correct && p[other] == correct_parts[other] && p[slot] != correct_parts[slot]
        });
        match found {
            Some((ix, _)) => wrong.push((bands[ix].clone(), ix)),
            None => {
                return Err(StatError::CellNotACombo {
                    games_of: cell_tokens.clone(),
                    bands: bands.clone(),
                })
            }
        }
    }
    let wrong: [(String, usize); 2] = wrong.try_into().expect("two slots pushed two entries");
    Ok(DuoCell {
        label: bands[correct].clone(),
        correct,
        wrong,
    })
}

/// One cell's half of H30: the cost of falsifying each slot alone,
/// judged over ply 0-19.
///
/// Returns one [`H29Family`]-shaped result per slot, labelled
/// `"<cell> vs <wrong label>"` so the printed row says which
/// combination stood in for the mistake. The domain is the shallow
/// stratum unconditionally — plan 08 registers no all-ply variant.
///
/// # Errors
///
/// Whatever [`check_duo_roles`] refuses, a gamma that was not swept,
/// an empty judged bucket, or the resampling refusing.
pub fn h30_cell(
    arms: &AlignedArms,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<[H29Family; 2], StatError> {
    let cell = check_duo_roles(arms)?;
    let mut out = Vec::with_capacity(2);
    for (wrong_label, wrong_ix) in &cell.wrong {
        out.push(cost_family_between(
            arms,
            ARM_K,
            ARM_K_B,
            &format!("{} vs {}", cell.label, wrong_label),
            cell.correct,
            *wrong_ix,
            Some(JOUSEKI_SHALLOW),
            gamma,
            draws,
            seed,
        )?);
    }
    Ok(out.try_into().expect("two wrongs produced two results"))
}

/// Plan 08's primary hypothesis: are both slots read, each in its own
/// attribute's direction?
#[derive(Debug, Clone, PartialEq)]
pub struct H30 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// Per cell: its label and the two per-slot judgments.
    pub cells: Vec<(String, [H29Family; 2])>,
}

impl H30 {
    /// The verdict on one month: confirmed only when **every**
    /// (cell, slot) confirms — the min-logic of [`H29::verdict`],
    /// widened to the grid, because a slot that only works in some
    /// cells and a cell where only one slot works are both exactly
    /// what the min exists to keep out of a confirmation. Refuted when
    /// any (cell, slot) refutes; everything else undetermined.
    ///
    /// The verdict is over the cells the caller fed — which cells
    /// those must be is the plan's registration, not this function's;
    /// it refuses none and reports all.
    pub fn verdict(&self) -> &'static str {
        let all = self
            .cells
            .iter()
            .flat_map(|(_, slots)| slots.iter())
            .all(H29Family::confirmed);
        let any_refuted = self
            .cells
            .iter()
            .flat_map(|(_, slots)| slots.iter())
            .any(H29Family::refuted);
        match (all, any_refuted) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H30 from one arm pair per cell.
///
/// Separate [`AlignedArms`] per cell for the reason [`h29`] takes two:
/// the cells' walks are different games and each bootstrap resamples
/// its own clusters. Two cells claiming the same label are refused —
/// the same walk fed twice would count one cell's evidence as two.
///
/// # Errors
///
/// As [`h30_cell`] for any cell, or a duplicated cell label.
pub fn h30(
    arms_by_cell: &[&AlignedArms],
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H30, StatError> {
    let mut cells = Vec::with_capacity(arms_by_cell.len());
    for arms in arms_by_cell {
        let cell = check_duo_roles(arms)?;
        if cells.iter().any(|(label, _)| *label == cell.label) {
            return Err(StatError::DuplicateCell { label: cell.label });
        }
        cells.push((cell.label.clone(), h30_cell(arms, gamma, draws, seed)?));
    }
    Ok(H30 { gamma, cells })
}

/// One slot axis of H32: the pooled mis-conditioning cost of
/// falsifying that slot alone, over every cell at once.
#[derive(Debug, Clone, PartialEq)]
pub struct H32Axis {
    /// Which slot, as the position in the combination labels (slot 0
    /// is the label's first part).
    pub slot: usize,
    /// `D_pool = Σ_cells Σ_positions cost / Σ_cells positions` for
    /// [`ARM_K`], judged over ply 0-19.
    pub d_pool_k: f64,
    /// The same for [`ARM_K_B`].
    pub d_pool_k_b: f64,
    /// `F_pool = |D_pool(K) - D_pool(K-b)|` on the sample as walked.
    pub floor: f64,
    /// Interval on `D_pool(K) - F_pool`, floor recomputed inside every
    /// stratified draw. Confirms when it excludes zero from above.
    pub confirm: Interval,
    /// Interval on `D_pool(K) + F_pool`. Refutes when it excludes zero
    /// from below.
    pub refute: Interval,
    /// Per cell, descriptively: label, `D(K)`, `D(K-b)` — the grid
    /// plan 08 judged, kept visible without carrying a verdict.
    pub per_cell: Vec<(String, f64, f64)>,
    /// Positions pooled in the judged stratum, across cells.
    pub positions: usize,
    /// Games those came from, across cells — the clusters resampled.
    pub games: usize,
}

impl H32Axis {
    /// Whether the confirm interval clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.confirm.excludes_zero_from_above()
    }

    /// Whether the refute interval clears zero from below.
    pub fn refuted(&self) -> bool {
        self.refute.excludes_zero_from_below()
    }
}

/// Plan 09's primary hypothesis: pooled over the cells, is each slot
/// read in its own attribute's direction?
#[derive(Debug, Clone, PartialEq)]
pub struct H32 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// The two slot axes.
    pub axes: [H32Axis; 2],
}

impl H32 {
    /// The verdict on one month: confirmed when **both** axes confirm,
    /// refuted when either refutes, undetermined otherwise. Per-cell
    /// survival carries no verdict here — plan 09 demotes it to the
    /// description [`H32Axis::per_cell`] keeps visible — which is the
    /// deliberate weakening plan 08's floor resolution forced, stated
    /// rather than smuggled.
    pub fn verdict(&self) -> &'static str {
        let all = self.axes.iter().all(H32Axis::confirmed);
        let any_refuted = self.axes.iter().any(H32Axis::refuted);
        match (all, any_refuted) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H32 from one arm pair per cell.
///
/// The pooled mean is position-weighted across cells, and the
/// bootstrap is **stratified**: each cell's games resample from their
/// own cluster space
/// ([`crate::metric::bootstrap::stratified_cluster_bootstrap`]),
/// because the cells share no games and a pooled cluster space would
/// treat them as exchangeable. The floor is recomputed inside every
/// draw, as every judge here does.
///
/// # Errors
///
/// As [`check_duo_roles`] for any cell, a duplicated cell, a gamma
/// that was not swept, an empty judged stratum, or the resampling
/// refusing.
pub fn h32(
    arms_by_cell: &[&AlignedArms],
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<H32, StatError> {
    // Per cell, per arm, per axis: a ClusterTally of the judged
    // stratum's per-position costs.
    let mut labels: Vec<String> = Vec::with_capacity(arms_by_cell.len());
    let mut tallies: Vec<[[ClusterTally; 2]; 2]> = Vec::with_capacity(arms_by_cell.len());
    for arms in arms_by_cell {
        let cell = check_duo_roles(arms)?;
        if labels.contains(&cell.label) {
            return Err(StatError::DuplicateCell { label: cell.label });
        }
        let gamma_ix = arms.gamma_index(gamma)?;
        let mut per_arm: Vec<[ClusterTally; 2]> = Vec::with_capacity(2);
        for arm in DUO_ARMS {
            let mut per_axis: Vec<ClusterTally> = Vec::with_capacity(2);
            for (_, wrong_ix) in &cell.wrong {
                let correct = cell.correct;
                let wrong = *wrong_ix;
                let cost = move |at: &GammaRecord| -> Option<f64> {
                    let per_band = at.top1.as_ref()?;
                    let c = *per_band.get(correct)?;
                    let w = *per_band.get(wrong)?;
                    Some((c as i64 - w as i64) as f64)
                };
                let tally = tally_in_bucket(arms, arm, gamma_ix, JOUSEKI_SHALLOW, cost)?;
                if tally.total().n == 0 {
                    return Err(StatError::EmptyBucket {
                        low: JOUSEKI_SHALLOW.0,
                        high: JOUSEKI_SHALLOW.1.saturating_sub(1),
                        role: "cost",
                    });
                }
                per_axis.push(tally);
            }
            per_arm.push(per_axis.try_into().expect("two axes"));
        }
        labels.push(cell.label);
        tallies.push(per_arm.try_into().expect("two arms"));
    }

    let strata: Vec<usize> = arms_by_cell.iter().map(|arms| arms.games()).collect();
    let pool = |axis: usize, arm: usize, draw: &[Vec<usize>]| -> Option<f64> {
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for (cell, stratum) in tallies.iter().zip(draw) {
            let t = cell[arm][axis].over(stratum)?;
            sum += t.sum;
            n += t.n;
        }
        (n > 0).then(|| sum / n as f64)
    };

    let mut axes: Vec<H32Axis> = Vec::with_capacity(2);
    for axis in 0..2 {
        let margin_and_floor = |draw: &[Vec<usize>]| -> Option<(f64, f64)> {
            let k = pool(axis, 0, draw)?;
            let k_b = pool(axis, 1, draw)?;
            Some((k, (k - k_b).abs()))
        };
        let confirm = stratified_cluster_bootstrap(&strata, draws, seed, |draw| {
            margin_and_floor(draw).map(|(m, f)| m - f)
        })?;
        let refute = stratified_cluster_bootstrap(&strata, draws, seed, |draw| {
            margin_and_floor(draw).map(|(m, f)| m + f)
        })?;
        let whole: Vec<Vec<usize>> = strata.iter().map(|n| (0..*n).collect()).collect();
        let (d_pool_k, floor) =
            margin_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
        let d_pool_k_b = pool(axis, 1, &whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
        let per_cell: Vec<(String, f64, f64)> = labels
            .iter()
            .zip(&tallies)
            .map(|(label, cell)| {
                (
                    label.clone(),
                    cell[0][axis].total().mean().unwrap_or(f64::NAN),
                    cell[1][axis].total().mean().unwrap_or(f64::NAN),
                )
            })
            .collect();
        let positions: usize = tallies.iter().map(|cell| cell[0][axis].total().n).sum();
        let games: usize = tallies
            .iter()
            .map(|cell| cell[0][axis].clusters_present())
            .sum();
        axes.push(H32Axis {
            slot: axis,
            d_pool_k,
            d_pool_k_b,
            floor,
            confirm,
            refute,
            per_cell,
            positions,
            games,
        });
    }
    Ok(H32 {
        gamma,
        axes: axes.try_into().expect("two axes"),
    })
}

/// The duo admission gate: the seed replicate's mean top-1 against the
/// primary's, at [`TOP1_GATE_JOUSEKI`] — the same reference and the
/// same tolerance as [`gate_top1_jouseki`], for the same reason: the
/// pair differ in the shuffle seed alone, so this catches a run defect
/// and nothing else can sit between them.
///
/// # Errors
///
/// As [`check_duo_roles`], plus a gamma that was not swept or the
/// resampling refusing.
pub fn gate_top1_duo(
    arms: &AlignedArms,
    gamma: f32,
    draws: usize,
    seed: u64,
) -> Result<Gate, StatError> {
    check_duo_roles(arms)?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let subject = tally(arms, ARM_K_B, gamma_ix, top1_mean)?;
    let baseline = tally(arms, ARM_K, gamma_ix, top1_mean)?;
    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(subject.mean_over(draw)? - baseline.mean_over(draw)?)
    })?;
    Ok(Gate {
        interval,
        tolerance: TOP1_GATE_JOUSEKI,
    })
}

/// One band of contestability, and what the two arms did inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stratum {
    /// Which of the [`STRATA`] strata, counting from zero.
    pub index: usize,
    /// Lowest margin in the band, or `None` for the open end.
    pub low: Option<f64>,
    /// Cut point the band stops below, or `None` for the open end.
    pub high: Option<f64>,
    /// Flip rate of [`ARM_A`] inside it, on the sample as walked.
    pub flip_a: f64,
    /// Flip rate of [`ARM_AB`] inside it.
    pub flip_ab: f64,
    /// Positions of `A` the band caught.
    pub positions_a: usize,
    /// Positions of `AB` it caught.
    ///
    /// Not the same number as [`Self::positions_a`], and not meant to
    /// be. The cut points are common, so a uniformly sharper arm puts
    /// more of its positions in the high bands — which is the
    /// difference the strata are there to hold still rather than to
    /// remove.
    pub positions_ab: usize,
    /// `F_flip` inside the band, on the sample as walked.
    pub floor: f64,
    /// Interval on `flip(AB) - flip(A) + F_flip`.
    pub interval: Interval,
    /// Share of the draws that reached zero from above — the one-sided
    /// level the Holm correction is applied to.
    pub p_below_zero: f64,
    /// Threshold Holm handed this band, given where its level ranked
    /// among the four.
    pub holm_threshold: f64,
    /// Whether the band refutes, **after** the correction.
    pub refuted: bool,
    /// Whether the band confirms: the interval above zero, uncorrected.
    ///
    /// No correction, and not by oversight. A confirmation needs every
    /// band to clear zero, so the four are combined by intersection and
    /// the joint claim is already at the level each part is tested at —
    /// correcting would make a criterion that fires less often than it
    /// says it does.
    pub confirmed: bool,
}

/// H20: does handing over the legal set cost steerability?
///
/// Flip rate, stratified. The raw rate is confounded across
/// formulations (`§2`): a sharper model flips less under the same
/// conditioning, and a legal mask or a legality input changes how sharp
/// the distribution is, so an unstratified comparison would read
/// sharpness as steerability.
///
/// A **non-inferiority** test rather than a detection one, because
/// `§3.1` puts every arm on the prefix channel and plan 02 measured
/// that channel at 15.2% flip against per-position's 49.0%. The
/// question is whether steerability survives, not whether it grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H20 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// The three quartile cut points, in ascending order.
    pub cuts: [f64; STRATA - 1],
    /// The four bands, in ascending order of margin.
    pub strata: [Stratum; STRATA],
    /// Level each band's Holm threshold was derived from.
    pub alpha: f64,
    /// Positions the arms share.
    pub positions: usize,
    /// Games those positions came from.
    pub games: usize,
}

impl H20 {
    /// Whether every band clears zero from above.
    pub fn confirmed(&self) -> bool {
        self.strata.iter().all(|s| s.confirmed)
    }

    /// Whether any band refutes after the correction.
    ///
    /// **On this month only.** `§4` also requires that the *same* band
    /// refute in both months, which no single month's statistic can
    /// answer — see [`Self::refuting`], which is what a caller
    /// intersects.
    pub fn refuted(&self) -> bool {
        self.strata.iter().any(|s| s.refuted)
    }

    /// Which bands refute, band by band.
    ///
    /// The form the two-month rule needs. `§4` refutes when a band
    /// clears zero from below in **both** months, so a caller holding
    /// two of these takes the elementwise conjunction and asks whether
    /// anything survives it. A summary verdict per month cannot answer
    /// that: two months refuting in different bands would read as two
    /// refutations and agree on nothing.
    pub fn refuting(&self) -> [bool; STRATA] {
        std::array::from_fn(|i| self.strata.get(i).is_some_and(|s| s.refuted))
    }

    /// The verdict on one month, in the three-way form.
    ///
    /// The two branches read the same quantity in opposite tails, at
    /// levels that do not meet: a confirmation needs every band's
    /// interval above zero, which puts every band's `p_below_zero` at
    /// or above `1 - alpha`, and no Holm threshold is above `alpha`.
    /// So a confirmed month has nothing left to refute with.
    pub fn verdict(&self) -> &'static str {
        match (self.confirmed(), self.refuted()) {
            (true, false) => "confirmed",
            (false, true) => "refuted",
            _ => "undetermined",
        }
    }
}

/// Assemble H20 from the four treatment arms.
///
/// # Why the cut points are pooled and common
///
/// The quartiles are taken over `A`'s and `AB`'s `top2_margin` values
/// **pooled**, and the resulting three cut points are then applied to
/// both arms. Not per-arm quantiles: those match on rank rather than on
/// contestability, so a uniformly sharper arm's own Q1 sits at a higher
/// absolute margin and the sharpness difference moves inside every
/// stratum instead of being removed.
///
/// The consequence is that the bands hold **different positions for the
/// two arms**, and that is intended. This is a comparison at matched
/// contestability, not a paired comparison of the same positions;
/// [`Stratum::positions_a`] and [`Stratum::positions_ab`] are reported
/// separately so a reader can see by how much.
///
/// The two replicates are placed against the same cut points, since a
/// floor measured on a differently drawn band would not be this band's
/// floor.
///
/// # The direction the tolerance points
///
/// The quantity is `flip(AB) - flip(A) + F_flip`, so `F_flip` **widens**
/// the tolerance: a noisier replicate pair makes "steerability
/// survives" easier to declare. In [`h21`] the floor points the other
/// way. `§4` records this asymmetry and does not fix it — a confirm
/// here is the weaker of the two claims, and `F_flip` being a max over
/// two pairs biases it further in that direction.
///
/// # Errors
///
/// One of the four arms is missing, an arm's checkpoint does not fit
/// its role or its walk is too old to state the legality axis
/// ([`LEGALITY_AXIS_VERSION`]), the gamma was not swept, no position
/// carries a margin ([`StatError::NoMargins`]), a band caught nothing
/// for one of the arms ([`StatError::EmptyStratum`]), or the resampling
/// refused.
pub fn h20(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H20, StatError> {
    require_kinds(
        arms,
        &[
            (ARM_A, CondEncoding::Prefix, false),
            (ARM_A_B, CondEncoding::Prefix, false),
            (ARM_AB, CondEncoding::Prefix, true),
            (ARM_AB_B, CondEncoding::Prefix, true),
        ],
    )?;
    let gamma_ix = arms.gamma_index(gamma)?;

    let mut pooled = margins(arms, ARM_A, gamma_ix)?;
    pooled.extend(margins(arms, ARM_AB, gamma_ix)?);
    if pooled.is_empty() {
        return Err(StatError::NoMargins {
            a: ARM_A,
            b: ARM_AB,
        });
    }
    let cuts = quartile_cuts(&mut pooled);

    let a = flips_by_stratum(arms, ARM_A, gamma_ix, &cuts)?;
    let a_b = flips_by_stratum(arms, ARM_A_B, gamma_ix, &cuts)?;
    let ab = flips_by_stratum(arms, ARM_AB, gamma_ix, &cuts)?;
    let ab_b = flips_by_stratum(arms, ARM_AB_B, gamma_ix, &cuts)?;

    let whole: Vec<usize> = (0..arms.games()).collect();
    let mut intervals: Vec<Interval> = Vec::with_capacity(STRATA);
    let mut levels = [1.0f64; STRATA];
    let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity(STRATA);
    let mut counts: Vec<(usize, usize)> = Vec::with_capacity(STRATA);

    for stratum in 0..STRATA {
        // Every one of these is in range — the arrays are `[_; STRATA]`
        // and so is the loop — and taken by `get` rather than by index
        // so that this function carries no panic.
        let missing = |arm: &'static str| StatError::EmptyStratum { stratum, arm };
        let in_a = a.get(stratum).ok_or_else(|| missing(ARM_A))?;
        let in_a_b = a_b.get(stratum).ok_or_else(|| missing(ARM_A_B))?;
        let in_ab = ab.get(stratum).ok_or_else(|| missing(ARM_AB))?;
        let in_ab_b = ab_b.get(stratum).ok_or_else(|| missing(ARM_AB_B))?;
        for (arm, tally) in [
            (ARM_A, in_a),
            (ARM_A_B, in_a_b),
            (ARM_AB, in_ab),
            (ARM_AB_B, in_ab_b),
        ] {
            if tally.total().n == 0 {
                return Err(StatError::EmptyStratum { stratum, arm });
            }
        }
        // Every band is read off the same seed, so the four are four
        // readings of one resample of games rather than four
        // independent ones — the same discipline the terms inside a
        // band are held to.
        let margin_and_floor = |draw: &[usize]| -> Option<(f64, f64, f64)> {
            let plain = in_a.mean_over(draw)?;
            let plain_b = in_a_b.mean_over(draw)?;
            let legal = in_ab.mean_over(draw)?;
            let legal_b = in_ab_b.mean_over(draw)?;
            Some((
                legal,
                plain,
                (legal - legal_b).abs().max((plain - plain_b).abs()),
            ))
        };
        let signed = cluster_bootstrap_signed(arms.games(), draws, seed, |draw| {
            margin_and_floor(draw).map(|(ab, a, f)| ab - a + f)
        })?;
        let (flip_ab, flip_a, floor) =
            margin_and_floor(&whole).ok_or(BootstrapError::UndefinedOnWholeSample)?;
        if let Some(slot) = levels.get_mut(stratum) {
            *slot = signed.p_below_zero;
        }
        intervals.push(signed.interval);
        points.push((flip_a, flip_ab, floor));
        counts.push((in_a.total().n, in_ab.total().n));
    }

    // One-sided, because a refutation is a claim about one direction.
    // At this level, uncorrected, it is the statement the 95% interval
    // already makes.
    let alpha = (1.0 - CONFIDENCE) / 2.0;
    let (rejected, thresholds) = holm(levels, alpha);

    let strata = std::array::from_fn(|i| {
        let (flip_a, flip_ab, floor) = points.get(i).copied().unwrap_or((f64::NAN, f64::NAN, 0.0));
        let (positions_a, positions_ab) = counts.get(i).copied().unwrap_or((0, 0));
        let interval = intervals.get(i).copied().unwrap_or(Interval {
            point: f64::NAN,
            low: f64::NAN,
            high: f64::NAN,
            draws: 0,
            undefined_draws: 0,
            clusters: arms.games(),
            seed,
        });
        Stratum {
            index: i,
            low: i.checked_sub(1).and_then(|c| cuts.get(c).copied()),
            high: cuts.get(i).copied(),
            flip_a,
            flip_ab,
            positions_a,
            positions_ab,
            floor,
            interval,
            p_below_zero: levels.get(i).copied().unwrap_or(1.0),
            holm_threshold: thresholds.get(i).copied().unwrap_or(0.0),
            refuted: rejected.get(i).copied().unwrap_or(false),
            confirmed: interval.excludes_zero_from_above(),
        }
    });

    Ok(H20 {
        gamma,
        cuts,
        strata,
        alpha,
        positions: arms.positions(),
        games: arms.games(),
    })
}

/// Every `top2_margin` an arm recorded at one gamma, positions that
/// carry none left out.
fn margins(arms: &AlignedArms, arm: &str, gamma_ix: usize) -> Result<Vec<f64>, StatError> {
    let walk = arms.walk(arm)?;
    Ok(walk
        .records
        .iter()
        .filter_map(|r| r.at.get(gamma_ix).and_then(|at| at.top2_margin))
        .collect())
}

/// The three quartile cut points of `pooled`, which is sorted in place.
///
/// Nearest-rank, as [`crate::metric::bootstrap`]'s percentiles are: the
/// `k`-th cut is the value at rank `floor(k/4 * n)`. Ties are not broken
/// — a pool where more than a quarter of the margins share a value puts
/// two cut points on that value, and the band between them is empty.
/// That is reported as [`StatError::EmptyStratum`] rather than papered
/// over, since a band holding nothing cannot carry a flip rate.
fn quartile_cuts(pooled: &mut [f64]) -> [f64; STRATA - 1] {
    pooled.sort_by(f64::total_cmp);
    let n = pooled.len();
    std::array::from_fn(|k| {
        let rank = ((k + 1) as f64 / STRATA as f64 * n as f64).floor() as usize;
        pooled
            .get(rank.min(n.saturating_sub(1)))
            .copied()
            .unwrap_or(f64::NAN)
    })
}

/// Which band a margin falls in, against common cut points.
///
/// Half-open upwards: a margin equal to a cut point belongs to the band
/// above it, so the bands partition the line and no position is counted
/// twice.
fn stratum_of(margin: f64, cuts: &[f64; STRATA - 1]) -> usize {
    cuts.iter().filter(|cut| margin >= **cut).count()
}

/// One arm's flips, tallied into the bands by that arm's own margins.
fn flips_by_stratum(
    arms: &AlignedArms,
    arm: &str,
    gamma_ix: usize,
    cuts: &[f64; STRATA - 1],
) -> Result<[ClusterTally; STRATA], StatError> {
    let walk = arms.walk(arm)?;
    let mut out: [ClusterTally; STRATA] = std::array::from_fn(|_| ClusterTally::new(arms.games()));
    for (record, cluster) in walk.records.iter().zip(arms.clusters()) {
        let Some(at) = record.at.get(gamma_ix) else {
            continue;
        };
        // A position with no margin cannot be placed at a
        // contestability, so it leaves the comparison rather than
        // landing in an end band by default.
        let Some(margin) = at.top2_margin else {
            continue;
        };
        // At most `cuts.len()` cuts can sit at or below a margin, so
        // the index is inside the array; the `get_mut` keeps this free
        // of indexing panics regardless.
        if let Some(tally) = out.get_mut(stratum_of(margin, cuts)) {
            tally.push(*cluster, if at.flipped { 1.0 } else { 0.0 })?;
        }
    }
    Ok(out)
}

/// Holm-Bonferroni across the bands.
///
/// Step-down: the smallest level is compared against `alpha/m`, the
/// next against `alpha/(m-1)`, and so on, stopping at the first that
/// does not clear its threshold. Valid under arbitrary dependence
/// between the bands, which matters here because they are four readings
/// of one resample of the same games.
///
/// Returns the rejections and the threshold each band was compared
/// against, both in the bands' own order rather than in ranked order.
fn holm(levels: [f64; STRATA], alpha: f64) -> ([bool; STRATA], [f64; STRATA]) {
    let mut order: [usize; STRATA] = std::array::from_fn(|i| i);
    order.sort_by(|a, b| {
        let (a, b) = (levels.get(*a), levels.get(*b));
        match (a, b) {
            (Some(a), Some(b)) => a.total_cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    });
    let mut rejected = [false; STRATA];
    let mut thresholds = [0.0f64; STRATA];
    let mut still_rejecting = true;
    for (rank, band) in order.iter().enumerate() {
        let threshold = alpha / (STRATA - rank) as f64;
        if let Some(slot) = thresholds.get_mut(*band) {
            *slot = threshold;
        }
        let clears = levels.get(*band).is_some_and(|level| *level <= threshold);
        if still_rejecting && clears {
            if let Some(slot) = rejected.get_mut(*band) {
                *slot = true;
            }
        } else {
            still_rejecting = false;
        }
    }
    (rejected, thresholds)
}

/// H19: the manipulation check, which carries no verdict.
///
/// `ce_legal(control) - ce_legal(A)`. `A` optimises `ce_legal` and the
/// control does not, so a difference is expected by construction —
/// `handoff.md:52` already measured it at 0.48 nats. It is here to say
/// the bake did what the flag claims, and a figure far from 0.48 is a
/// signal that something is wrong with the run rather than a finding.
///
/// # Why this type has no `verdict`
///
/// Not an omission and not a convention to be observed. A hypothesis
/// whose direction is known before the data arrives cannot be confirmed
/// by the data arriving in that direction, and revision 2 of the plan
/// made exactly that mistake — it judged this comparison, so confirm
/// carried no information and refute was unreachable. A type that
/// cannot express a verdict here is what stops the mistake from being
/// available: there is no `confirmed`, no `refuted` and no `verdict` to
/// call, so a caller that wanted one would have to write it itself and
/// would be visibly the author of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H19 {
    /// Guidance strength it was read at.
    pub gamma: f32,
    /// `ce_legal(control)` on the sample as walked.
    pub ce_control: f64,
    /// `ce_legal(A)`.
    pub ce_a: f64,
    /// The difference, on the sample as walked.
    pub difference: f64,
    /// Interval on the difference, over one set of game resamples.
    pub interval: Interval,
    /// Positions the arms share.
    pub positions: usize,
    /// Games those positions came from.
    pub games: usize,
}

/// Assemble H19 from the control and `A`.
///
/// # Errors
///
/// [`CONTROL`] or [`ARM_A`] is missing, either checkpoint does not fit
/// its role or was walked too early to state the legality axis
/// ([`LEGALITY_AXIS_VERSION`]), the gamma was not swept, or the
/// resampling refused.
pub fn h19(arms: &AlignedArms, gamma: f32, draws: usize, seed: u64) -> Result<H19, StatError> {
    require_kinds(
        arms,
        &[
            (CONTROL, CondEncoding::Prefix, false),
            (ARM_A, CondEncoding::Prefix, false),
        ],
    )?;
    let gamma_ix = arms.gamma_index(gamma)?;
    let control = tally(arms, CONTROL, gamma_ix, ce_mean)?;
    let a = tally(arms, ARM_A, gamma_ix, ce_mean)?;

    let interval = cluster_bootstrap(arms.games(), draws, seed, |draw| {
        Some(control.mean_over(draw)? - a.mean_over(draw)?)
    })?;
    Ok(H19 {
        gamma,
        ce_control: control.total().mean().unwrap_or(f64::NAN),
        ce_a: a.total().mean().unwrap_or(f64::NAN),
        difference: interval.point,
        interval,
        positions: arms.positions(),
        games: arms.games(),
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
            legal_input: false,
            ctx: 128,
            bands: vec!["<lo>".into(), "<mid>".into(), "<hi>".into()],
            gammas: GAMMAS.to_vec(),
            positions,
            games,
            games_of: Some(vec!["<lo>".into()]),
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
                    // Absent throughout these fixtures, as the margin
                    // and the per-band cost below are: nothing here
                    // reads any of the three, and a number no assertion
                    // depends on would read as though one did.
                    n_legal: None,
                    at: vec![GammaRecord {
                        flipped: flip(game, ix),
                        widest_js: js(ply),
                        legal_mass: 0.9,
                        top1: Some(vec![true, false, false]),
                        ce: None,
                        top2_margin: None,
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
                n_legal: None,
                at: vec![GammaRecord {
                    flipped: true,
                    widest_js: 0.02,
                    legal_mass: 0.9,
                    top1: Some(vec![true, false, false]),
                    ce: None,
                    top2_margin: None,
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
                        n_legal: None,
                        at: SWEEP
                            .iter()
                            .map(|_| GammaRecord {
                                flipped: (game + ix).is_multiple_of(flip_every),
                                widest_js: js,
                                legal_mass: 0.83,
                                top1: Some(vec![true, false, false]),
                                ce: None,
                                top2_margin: None,
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

    // ---- the legality experiment -------------------------------------
    //
    // Its arms are all prefix-conditioned, so nothing above tells them
    // apart and the fixtures below are built on the axis that does.

    /// The four margins the strata fixtures use.
    ///
    /// Dyadic rationals, so every value and every quartile cut point is
    /// exact in `f64` and an assertion about a cut is arithmetic rather
    /// than a tolerance. Spread evenly, so pooling two arms that carry
    /// the same set puts the three cuts at the second, third and fourth
    /// of them.
    const MARGINS: [f64; 4] = [0.125, 0.25, 0.375, 0.5];

    /// A walk carrying what the legality statistics read: a cost per
    /// band, a flip, and a top-two margin.
    ///
    /// The cost is the same in all three bands, so the mean over them
    /// is the number the fixture names and a test's arithmetic is the
    /// test's own rather than the fixture's.
    fn legality_walk(
        games: usize,
        per_game: usize,
        ce: impl Fn(usize, usize) -> f64,
        flip: impl Fn(usize, usize) -> bool,
        margin: impl Fn(usize) -> f64,
    ) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ix in 0..per_game {
                records.push(PositionRecord {
                    game,
                    ply: ix * 2,
                    n_legal: Some(30),
                    at: vec![GammaRecord {
                        flipped: flip(game, ix),
                        widest_js: 0.02,
                        legal_mass: 0.9,
                        top1: Some(vec![true, false, false]),
                        ce: Some(vec![ce(game, ix); 3]),
                        top2_margin: Some(margin(ix)),
                    }],
                });
            }
        }
        Walk {
            header: header(records.len(), games),
            records,
        }
    }

    /// A walk at one constant cost, whose flip and margin are keyed on
    /// the position within the game.
    ///
    /// Sixteen positions a game against four margin values makes each
    /// band catch four positions per game, so a flip rule keyed on `ix`
    /// is a constant rate **inside every band and in every resample** —
    /// which is what makes the answers assertable rather than merely
    /// bounded.
    fn banded_walk(
        games: usize,
        per_game: usize,
        cost: f64,
        flip: impl Fn(usize) -> bool,
        margin: impl Fn(usize) -> Option<f64>,
    ) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ix in 0..per_game {
                records.push(PositionRecord {
                    game,
                    ply: ix * 2,
                    n_legal: Some(30),
                    at: vec![GammaRecord {
                        flipped: flip(ix),
                        widest_js: 0.02,
                        legal_mass: 0.9,
                        top1: Some(vec![true, false, false]),
                        ce: Some(vec![cost; 3]),
                        top2_margin: margin(ix),
                    }],
                });
            }
        }
        Walk {
            header: header(records.len(), games),
            records,
        }
    }

    /// Label a walk as one of the legality arms: its own checkpoint,
    /// prefix conditioning, and the legality axis its role calls for.
    ///
    /// The axis is set from the name rather than passed in, so a
    /// fixture cannot quietly build an arm whose header disagrees with
    /// the slot it is about to go into — the tests that want that
    /// disagreement build it in the open.
    fn legality_arm(name: &'static str, mut walk: Walk) -> (String, Walk) {
        walk.header.ckpt = format!("/root/ckpt/{name}/run.safetensors");
        walk.header.encoding = CondEncoding::Prefix;
        walk.header.legal_input = matches!(name, ARM_AB | ARM_AB_B);
        (name.to_string(), walk)
    }

    fn four_arms(a: Walk, a_b: Walk, ab: Walk, ab_b: Walk) -> AlignedArms {
        AlignedArms::new(vec![
            legality_arm(ARM_A, a),
            legality_arm(ARM_A_B, a_b),
            legality_arm(ARM_AB, ab),
            legality_arm(ARM_AB_B, ab_b),
        ])
        .expect("the fixture arms share a position stream")
    }

    /// Four arms whose per-band cost is a constant each, which fixes
    /// `D` and `F` by arithmetic: every draw sees the same means, so
    /// both intervals are degenerate at their point estimates.
    fn arms_at_costs(a: f64, a_b: f64, ab: f64, ab_b: f64) -> AlignedArms {
        let walk = |cost: f64, phase: usize| {
            legality_walk(
                8,
                12,
                move |_, _| cost,
                // The flip pattern differs per arm so that two arms
                // sharing a cost are still two arms: `AlignedArms`
                // refuses a set whose records agree everywhere.
                move |_, ix| ix % 4 == phase,
                |ix| MARGINS[ix % MARGINS.len()],
            )
        };
        four_arms(walk(a, 0), walk(a_b, 1), walk(ab, 2), walk(ab_b, 3))
    }

    /// `§5.3`'s table, read off the curve this implements.
    ///
    /// Not two calculations agreeing. One row of that table is a
    /// measurement and every other row is `1/sqrt(n)` applied to it, so
    /// what this checks is that the code reads the same curve the plan
    /// does — which is the property that matters, since the floor a
    /// verdict is judged against has to be the pre-registered one
    /// rather than one this file arrived at.
    ///
    /// The anchor is written at 180 games while the measurement was
    /// made at 99, so the 99-game row returns through the table's four
    /// printed decimals rather than exactly. The test says by how much.
    #[test]
    fn the_resolution_reproduces_the_plans_own_table() {
        let planned = Resolution::at(PLANNED_GAMES).expect("the anchor");
        assert_eq!(planned.effect, PLANNED_EFFECT);

        for (games, printed) in [(99usize, 0.0718), (204, 0.0500), (1080, 0.0217)] {
            let row = Resolution::at(games).expect("a row of the table");
            assert!(
                (row.effect / printed - 1.0).abs() < 0.005,
                "{games} games: {} against the table's {printed}, more than half a percent apart",
                row.effect
            );
        }

        // And the values themselves, so that moving the anchor has to
        // be done deliberately rather than absorbed by the tolerance
        // above.
        let at = |games: usize| Resolution::at(games).expect("a row").effect;
        assert!((at(99) - 0.071_870).abs() < 1e-6, "{}", at(99));
        assert!((at(204) - 0.050_067).abs() < 1e-6, "{}", at(204));
        assert!((at(1080) - 0.021_760).abs() < 1e-6, "{}", at(1080));
    }

    /// Fewer games, blunter instrument. The point of carrying it at
    /// all: the walk is sized in positions and the game count is an
    /// outcome, so a run that lands at 150 games has a higher floor
    /// than the plan's estimate and a verdict has to be read against
    /// the one it got.
    #[test]
    fn the_resolution_falls_away_as_the_games_do() {
        let short = Resolution::at(150).expect("a short walk");
        let planned = Resolution::at(PLANNED_GAMES).expect("the planned walk");
        let long = Resolution::at(400).expect("a long walk");
        assert!(short.effect > planned.effect, "{short:?} vs {planned:?}");
        assert!(long.effect < planned.effect, "{long:?} vs {planned:?}");
        assert_eq!(short.planned_games, PLANNED_GAMES);
        assert_eq!(short.planned_effect, PLANNED_EFFECT);
        assert_eq!(Resolution::at(0), None);

        // On magnitude, so it governs a refutation as it governs a
        // confirmation.
        assert!(planned.resolves(PLANNED_EFFECT));
        assert!(planned.resolves(-PLANNED_EFFECT));
        assert!(!planned.resolves(PLANNED_EFFECT - 0.001));
        assert!(!planned.resolves(-(PLANNED_EFFECT - 0.001)));
    }

    /// H21 on a fixture whose answer is arithmetic: `A` costs 1.0 a
    /// position and `AB` costs nothing, so `D = 1`; the two replicates
    /// sit 0.24 and 0.12 from their partners, so `F = 0.24`.
    ///
    /// Every position carries the same cost, so every draw sees the
    /// same means and both intervals are degenerate — which is what
    /// makes the endpoints assertable rather than merely bounded.
    #[test]
    fn h21_assembles_from_four_arms_with_a_known_answer() {
        let arms = arms_at_costs(1.0, 1.24, 0.0, 0.12);
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!((result.ce_a - 1.0).abs() < 1e-12);
        assert!((result.ce_ab - 0.0).abs() < 1e-12);
        assert!((result.difference - 1.0).abs() < 1e-12);
        assert!((result.floor - 0.24).abs() < 1e-12, "{}", result.floor);
        assert!((result.confirm.point - 0.76).abs() < 1e-12);
        assert!((result.refute.point - 1.24).abs() < 1e-12);
        assert_eq!(result.positions, 96);
        assert_eq!(result.games, 8);
        assert_eq!(result.resolution.games, 8);
        assert!(result.resolved(), "{:?}", result.resolution);
        assert!(result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "confirmed");
        assert_eq!(result.confirm.seed, SEED);
    }

    /// The mirrored criteria, on the null `§4` describes: the two arms
    /// predict equally well, so `D` is zero while `F` is a real
    /// positive gap between two runs.
    ///
    /// `D - F` is then negative — which is why it cannot be the refute
    /// test — and `D + F` positive, so the refute branch does not fire.
    /// Asserted on the interval itself and not only on the verdict, so
    /// that the test would still catch a refute keyed on the wrong side
    /// if the resolution gate below it were removed.
    #[test]
    fn under_the_null_h21s_refute_branch_does_not_fire() {
        let arms = arms_at_costs(0.4, 0.64, 0.4, 0.52);
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!(
            (result.difference - 0.0).abs() < 1e-12,
            "{}",
            result.difference
        );
        assert!(
            result.floor > 0.0,
            "the two runs should differ: {}",
            result.floor
        );
        assert!(
            result.confirm.point < 0.0,
            "D - F is negative under the null, which is why it is not the refute test"
        );
        assert!(result.refute.point > 0.0);
        assert!(
            !result.refute.excludes_zero_from_below(),
            "the null must not be reported as a refutation: {:?}",
            result.refute
        );
        assert!(!result.confirm.excludes_zero_from_above());
        assert!(!result.refuted());
        assert!(!result.confirmed());
        assert_eq!(result.verdict(), "undetermined");
    }

    /// An arm that predicts worse by more than both observed
    /// same-recipe gaps is refuted: `AB` costs 1.0 where `A` costs
    /// nothing, so `D = -1` against a floor of 0.24.
    #[test]
    fn an_arm_worse_by_more_than_the_floor_refutes_h21() {
        let arms = arms_at_costs(0.0, 0.24, 1.0, 1.12);
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();
        assert!((result.difference + 1.0).abs() < 1e-12);
        assert!((result.floor - 0.24).abs() < 1e-12);
        assert!((result.refute.point + 0.76).abs() < 1e-12);
        assert!(result.resolved());
        assert!(result.refuted());
        assert!(!result.confirmed());
        assert_eq!(result.verdict(), "refuted");
    }

    /// The case `§4` records the minimum detectable effect for: an
    /// effect that is real, in the right direction, cleanly above its
    /// floor and above zero in every draw — and still too small for the
    /// walk that measured it.
    ///
    /// `D = 0.1` against a floor of 0.02, so the confirm interval
    /// clears zero and would read as a confirmation on the interval
    /// alone. Eight games resolve 0.253 nats, so it reads undetermined.
    /// This is what stops a result document reporting a margin under
    /// the floor as a near miss.
    #[test]
    fn an_effect_below_the_resolution_reads_undetermined_however_clean() {
        let arms = arms_at_costs(0.1, 0.12, 0.0, 0.01);
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!((result.difference - 0.1).abs() < 1e-9);
        assert!((result.floor - 0.02).abs() < 1e-9, "{}", result.floor);
        assert!(
            result.confirm.excludes_zero_from_above(),
            "the interval itself is clean: {:?}",
            result.confirm
        );
        assert!(
            result.resolution.effect > result.difference,
            "the walk must be blunter than the effect for this test to be about anything: \
             {:?} vs {}",
            result.resolution,
            result.difference
        );
        assert!(!result.resolved());
        assert!(!result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "undetermined");
    }

    /// The floor's own case, kept apart from the resolution's: an
    /// effect in the right direction, large enough for this walk to
    /// separate, and smaller than the larger of the two same-recipe
    /// seed gaps.
    ///
    /// `D = 0.3` against `F = 0.5`, at eight games resolving 0.253. So
    /// `resolved` is true and the verdict is still undetermined, which
    /// is what says the floor did the refusing here rather than the
    /// resolution. A confirmed H21 has to exceed **both** observed
    /// gaps, and this exceeds neither.
    #[test]
    fn an_effect_smaller_than_the_seed_floor_confirms_nothing() {
        let arms = arms_at_costs(0.3, 0.8, 0.0, 0.1);
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!((result.difference - 0.3).abs() < 1e-9);
        assert!((result.floor - 0.5).abs() < 1e-9, "{}", result.floor);
        assert!(
            result.resolved(),
            "the walk must resolve this effect for the test to be about the floor: {:?} vs {}",
            result.resolution,
            result.difference
        );
        assert!(
            result.confirm.point < 0.0,
            "D - F is negative below the floor: {:?}",
            result.confirm
        );
        assert!(result.refute.point > 0.0);
        assert!(!result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "undetermined");
    }

    /// Both intervals of one H21 come from the same draws in the same
    /// order, so a relation holding draw by draw survives into the
    /// **endpoints**.
    ///
    /// The fixture fixes `F` at 0.24 in every resample — each replicate
    /// sits a constant distance from its partner at every position — so
    /// `refute = confirm + 2F` at every draw. The margin does vary,
    /// because `A`'s cost differs game to game while `AB`'s does not,
    /// so the intervals have width and the equality is not vacuous.
    ///
    /// The point estimates are deliberately not what is asserted: their
    /// difference is arithmetic on one call over the whole sample, and
    /// would hold just as exactly if the two intervals came from
    /// unrelated seeds.
    #[test]
    fn the_two_h21_intervals_are_two_readings_of_one_set_of_draws() {
        let vary = |game: usize, _: usize| 0.5 + (game % 5) as f64 * 0.125;
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let arms = four_arms(
            legality_walk(8, 12, vary, |_, ix| ix % 4 == 0, margin),
            legality_walk(
                8,
                12,
                move |g, i| vary(g, i) + 0.24,
                |_, ix| ix % 4 == 1,
                margin,
            ),
            legality_walk(8, 12, |_, _| 0.2, |_, ix| ix % 4 == 2, margin),
            legality_walk(8, 12, |_, _| 0.32, |_, ix| ix % 4 == 3, margin),
        );
        let result = h21(&arms, 1.0, DRAWS, SEED).unwrap();

        assert_eq!(result.confirm.seed, result.refute.seed);
        assert_eq!(result.confirm.draws, result.refute.draws);
        assert!(
            (result.floor - 0.24).abs() < 1e-9,
            "the floor should be the larger, constant gap: {}",
            result.floor
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
                (refute_end - (confirm_end + 2.0 * result.floor)).abs() < 1e-9,
                "the {which} ends should sit exactly 2F apart, which they only do if the two \
                 intervals rank the same draws: {refute_end} vs {confirm_end}"
            );
        }
    }

    /// A walk whose flip rate, cost and top-1 share are each a constant
    /// fixed by arithmetic.
    ///
    /// Twelve positions a game, so `flips` of them flipping is a rate of
    /// `flips / 12` inside every resample — which makes `M` and `F`
    /// assertable rather than merely bounded.
    fn survival_walk(flips: usize, cost: f64, top1: usize) -> Walk {
        let mut walk = legality_walk(
            8,
            12,
            move |_, _| cost,
            move |_, ix| ix < flips,
            |ix| MARGINS[ix % MARGINS.len()],
        );
        // `legality_walk` writes one top-1 pattern for every record.
        // Rewrite it per position so an arm's top-1 share is `top1 / 12`
        // over the first band, and a third of that meaned over three.
        for (ix, record) in walk.records.iter_mut().enumerate() {
            let within = ix % 12;
            for at in &mut record.at {
                at.top1 = Some(vec![within < top1, false, false]);
            }
        }
        walk
    }

    /// Label a walk as a survival arm: its own checkpoint, and the
    /// conditioning its role calls for.
    ///
    /// The encoding is what separates these arms, so it is set from the
    /// role rather than left at the header default — the mirror of
    /// [`legality_arm`], where every arm is prefix.
    fn survival_arm(name: &'static str, mut walk: Walk) -> (String, Walk) {
        walk.header.ckpt = format!("/root/ckpt/{name}/run.safetensors");
        walk.header.encoding = match name {
            ARM_P | ARM_P_B => CondEncoding::EveryPosition,
            _ => CondEncoding::Prefix,
        };
        walk.header.legal_input = false;
        (name.to_string(), walk)
    }

    fn survival_arms(a: Walk, a_b: Walk, p: Walk, p_b: Walk) -> AlignedArms {
        AlignedArms::new(vec![
            survival_arm(ARM_A, a),
            survival_arm(ARM_A_B, a_b),
            survival_arm(ARM_P, p),
            survival_arm(ARM_P_B, p_b),
        ])
        .expect("the fixture arms share a position stream")
    }

    /// `F` is the larger of **both** pairs' gaps, not the treatment
    /// pair's standing in for both.
    ///
    /// The whole point of the correction over [`h14`], so the fixture is
    /// built with the prefix pair as the noisier one: `P` flips 9/12 and
    /// `P-b` 8/12 (a gap of 1/12), while `A` flips 3/12 and `A-b` 5/12
    /// (a gap of 2/12). A floor borrowed from the treatment pair would
    /// be 0.0833 and the confirm interval would sit at 0.4167; the
    /// matched floor is 0.1667 and it sits at 0.3333.
    #[test]
    fn h22_takes_the_larger_of_both_seed_gaps() {
        let arms = survival_arms(
            survival_walk(3, 2.7, 3),
            survival_walk(5, 2.75, 3),
            survival_walk(9, 2.8, 3),
            survival_walk(8, 2.85, 3),
        );
        let result = h22(&arms, 1.0, DRAWS, SEED).unwrap();

        assert!((result.flip_p - 0.75).abs() < 1e-12, "{}", result.flip_p);
        assert!((result.flip_a - 0.25).abs() < 1e-12, "{}", result.flip_a);
        assert!((result.margin - 0.5).abs() < 1e-12, "{}", result.margin);
        assert!(
            (result.floor - 2.0 / 12.0).abs() < 1e-12,
            "the prefix pair's gap is the larger one and has to win: {}",
            result.floor
        );
        assert!(
            (result.confirm.point - (0.5 - 2.0 / 12.0)).abs() < 1e-12,
            "{}",
            result.confirm.point
        );
        assert!(
            (result.refute.point - (0.5 + 2.0 / 12.0)).abs() < 1e-12,
            "{}",
            result.refute.point
        );
        assert!(result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "confirmed");
    }

    /// The axis that separates these arms is the encoding, so a prefix
    /// checkpoint in a per-position slot is refused before a figure is
    /// computed.
    #[test]
    fn h22_refuses_a_prefix_checkpoint_in_a_per_position_slot() {
        let mut swapped = survival_arm(ARM_P, survival_walk(9, 2.8, 3));
        swapped.1.header.encoding = CondEncoding::Prefix;
        let arms = AlignedArms::new(vec![
            survival_arm(ARM_A, survival_walk(3, 2.7, 3)),
            survival_arm(ARM_A_B, survival_walk(5, 2.75, 3)),
            swapped,
            survival_arm(ARM_P_B, survival_walk(8, 2.85, 3)),
        ])
        .unwrap();
        let err = h22(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::WrongEncodingForRole {
                    arm: ARM_P,
                    want: CondEncoding::EveryPosition,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Legality-as-input is held out of plan 04 entirely, so a legality
    /// checkpoint anywhere is an unregistered third condition rather
    /// than a mislabelled arm.
    #[test]
    fn h22_refuses_a_legality_checkpoint_in_any_slot() {
        let mut legality = survival_arm(ARM_P, survival_walk(9, 2.8, 3));
        legality.1.header.legal_input = true;
        let arms = AlignedArms::new(vec![
            survival_arm(ARM_A, survival_walk(3, 2.7, 3)),
            survival_arm(ARM_A_B, survival_walk(5, 2.75, 3)),
            legality,
            survival_arm(ARM_P_B, survival_walk(8, 2.85, 3)),
        ])
        .unwrap();
        let err = h22(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::WrongLegalInputForRole {
                    arm: ARM_P,
                    want: false,
                    found: true
                }
            ),
            "{err:?}"
        );
    }

    /// Exchanging the two runs of one recipe leaves the floor alone and
    /// makes the margin the other replicate's — two readings of one
    /// experiment, which is why nothing tries to tell them apart.
    #[test]
    fn swapping_within_a_pair_leaves_h22s_floor_alone() {
        let straight = survival_arms(
            survival_walk(3, 2.7, 3),
            survival_walk(5, 2.75, 3),
            survival_walk(9, 2.8, 3),
            survival_walk(8, 2.85, 3),
        );
        let swapped = survival_arms(
            survival_walk(5, 2.75, 3),
            survival_walk(3, 2.7, 3),
            survival_walk(9, 2.8, 3),
            survival_walk(8, 2.85, 3),
        );
        let a = h22(&straight, 1.0, DRAWS, SEED).unwrap();
        let b = h22(&swapped, 1.0, DRAWS, SEED).unwrap();

        assert!(
            (a.floor - b.floor).abs() < 1e-12,
            "{} vs {}",
            a.floor,
            b.floor
        );
        assert!(
            (b.margin - (0.75 - 5.0 / 12.0)).abs() < 1e-12,
            "the margin becomes the other replicate's: {}",
            b.margin
        );
    }

    /// A walk whose divergence is one constant in the opening and
    /// another deep, so the depth ratio is fixed by arithmetic.
    ///
    /// Twelve positions a game: the first six at ply 0, 2, 4, 6, 8, 10
    /// and the rest at 40, 42, ... 50, which puts five in the shallow
    /// bucket (`ply < 10`), six in the deep one (`40 <= ply <= 125`),
    /// and one in neither.
    fn depth_walk(shallow_js: f64, deep_js: f64) -> Walk {
        let mut records = Vec::new();
        for game in 0..8 {
            for ix in 0..12usize {
                let (ply, js) = if ix < 6 {
                    (ix * 2, shallow_js)
                } else {
                    (40 + (ix - 6) * 2, deep_js)
                };
                records.push(PositionRecord {
                    game,
                    ply,
                    n_legal: Some(30),
                    at: vec![GammaRecord {
                        flipped: ix % 3 == 0,
                        widest_js: js,
                        legal_mass: 0.9,
                        top1: Some(vec![true, false, false]),
                        ce: Some(vec![2.7; 3]),
                        top2_margin: Some(MARGINS[ix % MARGINS.len()]),
                    }],
                });
            }
        }
        Walk {
            header: header(records.len(), 8),
            records,
        }
    }

    /// `F_R` is the larger of both pairs' gaps in the ratio, which is
    /// what `h15` has none of and what the plan's ambiguous wording had
    /// to be resolved into.
    ///
    /// The fixture puts the noisier pair on the prefix side again:
    /// `P` and `P-b` sit at ratios 0.5 and 0.4 (a gap of 0.1) while `A`
    /// and `A-b` sit at 0.2 and 0.35 (a gap of 0.15). Borrowing the
    /// treatment pair's gap would leave the confirm interval at 0.2;
    /// the matched floor puts it at 0.15.
    #[test]
    fn h23_takes_the_larger_of_both_seed_gaps_in_the_ratio() {
        let arms = survival_arms(
            depth_walk(0.1, 0.02),
            depth_walk(0.1, 0.035),
            depth_walk(0.1, 0.05),
            depth_walk(0.1, 0.04),
        );
        let result = h23(&arms, 1.0, DRAWS, SEED).unwrap();

        assert_eq!(result.shallow.positions, 8 * 5);
        assert_eq!(result.deep.positions, 8 * 6);
        assert!((result.ratio_p - 0.5).abs() < 1e-12, "{}", result.ratio_p);
        assert!((result.ratio_a - 0.2).abs() < 1e-12, "{}", result.ratio_a);
        assert!((result.margin - 0.3).abs() < 1e-12, "{}", result.margin);
        assert!(
            (result.floor - 0.15).abs() < 1e-12,
            "the prefix pair's gap is the larger one and has to win: {}",
            result.floor
        );
        assert!(
            (result.confirm.point - 0.15).abs() < 1e-12,
            "{}",
            result.confirm.point
        );
        assert!((result.refute.point - 0.45).abs() < 1e-12);
        assert!(result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "confirmed");
    }

    /// An opening in which the bands never disagree leaves the ratio
    /// undefined rather than infinite, as `h15`'s does.
    #[test]
    fn h23_refuses_a_ratio_with_no_opening_divergence() {
        let arms = survival_arms(
            depth_walk(0.0, 0.02),
            depth_walk(0.0, 0.035),
            depth_walk(0.0, 0.05),
            depth_walk(0.0, 0.04),
        );
        let err = h23(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::Bootstrap(BootstrapError::EveryDrawUndefined { .. })
                    | StatError::Bootstrap(BootstrapError::UndefinedOnWholeSample)
            ),
            "{err:?}"
        );
    }

    /// The admission gate's reference is `A`, the arm being compared —
    /// the correction plan 03's gate needed.
    ///
    /// Two arms that match each other pass however far both sit from
    /// anything else, and an arm that differs from `A` by more than the
    /// tolerance is outside. A gate keyed on some other baseline could
    /// not produce both of these from one fixture.
    #[test]
    fn gate_top1_survival_measures_against_a() {
        let arms = survival_arms(
            survival_walk(3, 2.7, 3),
            survival_walk(5, 2.75, 3),
            survival_walk(9, 2.8, 3),
            survival_walk(8, 2.85, 9),
        );
        // `P` and `A` share a top-1 share of (3/12)/3, so the gate is
        // centred on zero however far that is from the control plan 03
        // used.
        let matched = gate_top1_survival(&arms, ARM_P, 1.0, DRAWS, SEED).unwrap();
        assert!(matched.interval.point.abs() < 1e-12, "{}", matched.interval);
        assert!(matched.passes());
        assert_eq!(matched.verdict(), "pass");

        // `P-b` is at (9/12)/3, which is 0.1667 away — five times the
        // tolerance.
        let apart = gate_top1_survival(&arms, ARM_P_B, 1.0, DRAWS, SEED).unwrap();
        assert!(
            (apart.interval.point - (0.75 - 0.25) / 3.0).abs() < 1e-12,
            "{}",
            apart.interval
        );
        assert!(!apart.passes());
        assert_eq!(apart.verdict(), "outside");
        assert_eq!(apart.tolerance, TOP1_GATE_SURVIVAL);

        // And `A` against itself is identically zero, which is the
        // property that says the baseline is the one named.
        let self_ref = gate_top1_survival(&arms, ARM_A, 1.0, DRAWS, SEED).unwrap();
        assert!(self_ref.interval.point.abs() < 1e-12);
    }

    #[test]
    fn h21_without_all_four_arms_is_refused() {
        let walk = |cost: f64, phase: usize| {
            legality_walk(
                4,
                8,
                move |_, _| cost,
                move |_, ix| ix % 4 == phase,
                |ix| MARGINS[ix % MARGINS.len()],
            )
        };
        let arms = AlignedArms::new(vec![
            legality_arm(ARM_A, walk(1.0, 0)),
            legality_arm(ARM_A_B, walk(1.2, 1)),
            legality_arm(ARM_AB, walk(0.0, 2)),
        ])
        .unwrap();
        let err = h21(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(err, StatError::Align(AlignError::UnknownArm { .. })),
            "{err:?}"
        );
    }

    /// The hole this set is built around. Every arm is
    /// prefix-conditioned, so the encoding check passes on a swap of
    /// `A` and `AB` and every figure downstream stays well-formed while
    /// `D` changes sign. The legality axis is what refuses it — and
    /// could not, before the header recorded it.
    #[test]
    fn an_arm_on_the_wrong_side_of_the_legality_axis_is_refused() {
        let walk = |cost: f64, phase: usize| {
            legality_walk(
                6,
                8,
                move |_, _| cost,
                move |_, ix| ix % 4 == phase,
                |ix| MARGINS[ix % MARGINS.len()],
            )
        };
        // The two treatment files handed over the other way round,
        // which is what a swap on a command line produces. The control
        // is present so that the up-front check below reaches the swap
        // rather than stopping at a missing arm.
        let (_, plain) = legality_arm(ARM_A, walk(1.0, 0));
        let (_, legal) = legality_arm(ARM_AB, walk(0.0, 2));
        let arms = AlignedArms::new(vec![
            legality_arm(CONTROL, walk(1.4, 4)),
            (ARM_A.to_string(), legal),
            (ARM_A_B.to_string(), legality_arm(ARM_A_B, walk(1.2, 1)).1),
            (ARM_AB.to_string(), plain),
            (ARM_AB_B.to_string(), legality_arm(ARM_AB_B, walk(0.1, 3)).1),
        ])
        .unwrap();

        // Both encodings are `Prefix`, so the check that catches a role
        // swap in the other experiment sees nothing wrong here.
        assert!(require_roles(
            &arms,
            &[
                (ARM_A, CondEncoding::Prefix),
                (ARM_AB, CondEncoding::Prefix)
            ]
        )
        .is_ok());

        let err = h21(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::WrongLegalInputForRole {
                    arm: ARM_A,
                    want: false,
                    found: true,
                }
            ),
            "{err:?}"
        );
        // And before anything is computed, which is what a program that
        // prints as it goes needs — on the same variant, so that this
        // half of the test cannot pass on some unrelated refusal.
        let up_front = check_legality_roles(&arms).unwrap_err();
        assert!(
            matches!(
                up_front,
                StatError::WrongLegalInputForRole {
                    arm: ARM_A,
                    want: false,
                    found: true,
                }
            ),
            "{up_front:?}"
        );
    }

    /// A legality arm walked by a build that predates the header field
    /// is refused rather than believed — **in every slot**, which is
    /// what reading the value alone could not do.
    ///
    /// Before version 4 the field parses as `false` whatever the
    /// checkpoint was. Comparing that `false` against the role refuses
    /// a stale walk only where the role calls for `true`: the same
    /// stale walk of the same legality checkpoint, put in `A`, `A-b` or
    /// the control, agrees with what its role wanted and passes having
    /// said nothing. So the refusal is keyed on the version, and both
    /// slots are asserted here — the second is the one that used to
    /// pass.
    #[test]
    fn a_legality_arm_walked_before_the_field_existed_is_refused() {
        let walk =
            |cost: f64, phase: usize| banded_walk(6, 8, cost, move |ix| ix % 5 == phase, cycled);
        // What a version 3 walk parses into: the axis unrecorded, so
        // read back as false whatever the checkpoint was.
        let five = |stale: &'static str| {
            let named: Vec<(String, Walk)> = [
                (CONTROL, 1.5, 0),
                (ARM_A, 1.0, 1),
                (ARM_A_B, 1.1, 2),
                (ARM_AB, 0.8, 3),
                (ARM_AB_B, 0.9, 4),
            ]
            .into_iter()
            .map(|(name, cost, phase)| {
                let (slot, mut w) = legality_arm(name, walk(cost, phase));
                if name == stale {
                    w.header.version = 3;
                    w.header.legal_input = false;
                }
                (slot, w)
            })
            .collect();
            AlignedArms::new(named).expect("the fixture arms share a position stream")
        };

        // The `A` slot first: it is the one the value check admits.
        for slot in [ARM_A, ARM_AB] {
            let err = check_legality_roles(&five(slot)).unwrap_err();
            assert!(
                matches!(
                    err,
                    StatError::RecordsPredateLegalityAxis {
                        arm,
                        found: 3,
                        needed: LEGALITY_AXIS_VERSION,
                    } if arm == slot
                ),
                "{slot}: {err:?}"
            );
        }

        // And why the `A` slot needs the version rather than the value:
        // there the stale default is exactly what the role requires, so
        // the value check has nothing to object to.
        let arms = five(ARM_A);
        assert!(!arms.walk(ARM_A).unwrap().header.legal_input);
        assert_eq!(LEGALITY_ROLES[1], (ARM_A, CondEncoding::Prefix, false));
    }

    /// `§5.1` item 3, asked of the version rather than of the fields.
    #[test]
    fn a_walk_that_predates_the_scored_fields_is_refused() {
        let arms = arms_at_costs(1.0, 1.24, 0.0, 0.12);
        assert!(check_scoreable(&arms, &[ARM_A, ARM_A_B, ARM_AB, ARM_AB_B]).is_ok());

        let mut named: Vec<(String, Walk)> = [ARM_A, ARM_A_B, ARM_AB, ARM_AB_B]
            .iter()
            .map(|role| ((*role).to_string(), arms.walk(role).unwrap().clone()))
            .collect();
        if let Some((_, walk)) = named.first_mut() {
            walk.header.version = 2;
        }
        let stale = AlignedArms::new(named).unwrap();
        let err = check_scoreable(&stale, &[ARM_A, ARM_A_B, ARM_AB, ARM_AB_B]).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::RecordsPredateScoring {
                    arm: ARM_A,
                    found: 2,
                    needed: SCOREABLE_VERSION,
                }
            ),
            "{err:?}"
        );
    }

    /// The two role checks are not interchangeable, which is why there
    /// are two of them: each refuses the other's arm set.
    #[test]
    fn the_two_role_checks_each_refuse_the_others_arm_set() {
        let legality = arms_at_costs(1.0, 1.24, 0.0, 0.12);
        check_legality_roles(&legality).expect_err("the control is missing from this set");
        assert!(check_roles(&legality).is_err());

        let conditioning = three_arms(
            walk(6, 12, |_, _| true, |_| 0.02),
            walk(6, 12, |_, _| false, |_| 0.02),
            walk(6, 12, |g, ix| !(g == 0 && ix == 0), |_| 0.02),
        );
        check_roles(&conditioning).expect("the canonical roles");
        assert!(check_legality_roles(&conditioning).is_err());
    }

    /// The five roles as they should be, so the refusals above are not
    /// passing for want of the check ever succeeding.
    #[test]
    fn the_five_legality_roles_pass_the_up_front_check() {
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let walk = |cost: f64, phase: usize| {
            legality_walk(6, 8, move |_, _| cost, move |_, ix| ix % 5 == phase, margin)
        };
        let arms = AlignedArms::new(vec![
            legality_arm(CONTROL, walk(1.5, 0)),
            legality_arm(ARM_A, walk(1.0, 1)),
            legality_arm(ARM_A_B, walk(1.1, 2)),
            legality_arm(ARM_AB, walk(0.8, 3)),
            legality_arm(ARM_AB_B, walk(0.9, 4)),
        ])
        .unwrap();
        check_legality_roles(&arms).expect("the canonical legality roles");
        check_scoreable(&arms, &LEGALITY_ARMS).expect("records written by this build");

        // The control is the prefix arm under another name, and both
        // spellings reach the same walk.
        assert_eq!(CONTROL, PREFIX);
        assert_eq!(arms.walk(CONTROL).unwrap(), arms.walk(PREFIX).unwrap());
    }

    /// H19 is reported and not judged. Its direction is known before
    /// the data arrives — `A` optimises this quantity and the control
    /// does not — so there is no `confirmed`, no `refuted` and no
    /// `verdict` on the type to call.
    #[test]
    fn h19_reports_a_difference_without_a_verdict() {
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let walk = |cost: f64, phase: usize| {
            legality_walk(6, 8, move |_, _| cost, move |_, ix| ix % 5 == phase, margin)
        };
        let arms = AlignedArms::new(vec![
            legality_arm(CONTROL, walk(1.48, 0)),
            legality_arm(ARM_A, walk(1.0, 1)),
        ])
        .unwrap();
        let result = h19(&arms, 1.0, DRAWS, SEED).unwrap();
        assert!((result.ce_control - 1.48).abs() < 1e-12);
        assert!((result.ce_a - 1.0).abs() < 1e-12);
        assert!((result.difference - 0.48).abs() < 1e-12);
        assert_eq!(result.interval.point, result.difference);
        assert_eq!(result.games, 6);
        assert_eq!(result.positions, 48);
    }

    /// The control slot is guarded on both axes, since a legality
    /// checkpoint there would make H19 a difference against a treatment
    /// arm while every figure stayed well-formed.
    #[test]
    fn h19_refuses_a_legality_checkpoint_in_the_control_slot() {
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let walk = |cost: f64, phase: usize| {
            legality_walk(6, 8, move |_, _| cost, move |_, ix| ix % 5 == phase, margin)
        };
        let (_, mut control) = legality_arm(CONTROL, walk(1.48, 0));
        control.header.legal_input = true;
        let arms = AlignedArms::new(vec![
            (CONTROL.to_string(), control),
            legality_arm(ARM_A, walk(1.0, 1)),
        ])
        .unwrap();
        let err = h19(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::WrongLegalInputForRole {
                    arm: CONTROL,
                    want: false,
                    found: true,
                }
            ),
            "{err:?}"
        );
    }

    /// The margin every strata fixture uses: [`MARGINS`] cycled.
    fn cycled(ix: usize) -> Option<f64> {
        MARGINS.get(ix % MARGINS.len()).copied()
    }

    /// Four arms for the strata fixtures, differing in their flip rules
    /// and — so that two arms sharing a rule are still two arms — in a
    /// per-band cost nothing in H20 reads.
    fn stratified_arms(
        flip_a: impl Fn(usize) -> bool,
        flip_a_b: impl Fn(usize) -> bool,
        flip_ab: impl Fn(usize) -> bool,
        flip_ab_b: impl Fn(usize) -> bool,
    ) -> AlignedArms {
        four_arms(
            banded_walk(8, 16, 1.0, flip_a, cycled),
            banded_walk(8, 16, 1.1, flip_a_b, cycled),
            banded_walk(8, 16, 0.9, flip_ab, cycled),
            banded_walk(8, 16, 0.8, flip_ab_b, cycled),
        )
    }

    /// H20 on a fixture whose every band answers the same way: `AB`
    /// flips everywhere and `A` nowhere, so the margin is `+1` in each
    /// band; each replicate flips on half its band's positions, so
    /// `F_flip` is 0.5 in each.
    ///
    /// The cut points are the second, third and fourth of [`MARGINS`],
    /// because the two compared arms carry the same set and pooling
    /// them leaves the quartiles where a single copy would.
    #[test]
    fn h20_cuts_common_strata_and_reads_a_known_answer() {
        let arms = stratified_arms(|_| false, |ix| ix < 8, |_| true, |ix| ix >= 8);
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();

        assert_eq!(result.cuts, [MARGINS[1], MARGINS[2], MARGINS[3]]);
        assert_eq!(result.positions, 128);
        assert_eq!(result.games, 8);
        for band in &result.strata {
            assert_eq!(band.positions_a, 32, "{band:?}");
            assert_eq!(band.positions_ab, 32, "{band:?}");
            assert!((band.flip_a - 0.0).abs() < 1e-12, "{band:?}");
            assert!((band.flip_ab - 1.0).abs() < 1e-12, "{band:?}");
            assert!((band.floor - 0.5).abs() < 1e-12, "{band:?}");
            assert!((band.interval.point - 1.5).abs() < 1e-12, "{band:?}");
            assert!(band.confirmed, "{band:?}");
            assert!(!band.refuted, "{band:?}");
        }
        // Bands are half-open upwards and cover the line.
        assert_eq!(result.strata[0].low, None);
        assert_eq!(result.strata[0].high, Some(MARGINS[1]));
        assert_eq!(result.strata[3].low, Some(MARGINS[3]));
        assert_eq!(result.strata[3].high, None);

        assert!(result.confirmed());
        assert!(!result.refuted());
        assert_eq!(result.verdict(), "confirmed");
        assert_eq!(result.refuting(), [false; STRATA]);
    }

    /// A cost larger than the floor refutes, in every band, and the
    /// bands are reported one by one so the caller can intersect two
    /// months rather than two summaries.
    ///
    /// `AB` flips on a quarter of each band against `A`'s everywhere,
    /// so the margin is `-0.75` against a floor of 0.5.
    #[test]
    fn a_cost_larger_than_the_floor_refutes_h20() {
        let arms = stratified_arms(|_| true, |ix| ix < 8, |ix| ix < 4, |ix| ix < 4);
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();

        for band in &result.strata {
            assert!((band.flip_a - 1.0).abs() < 1e-12, "{band:?}");
            assert!((band.flip_ab - 0.25).abs() < 1e-12, "{band:?}");
            assert!((band.floor - 0.5).abs() < 1e-12, "{band:?}");
            assert!((band.interval.point + 0.25).abs() < 1e-12, "{band:?}");
            assert!(band.refuted, "{band:?}");
            assert!(!band.confirmed, "{band:?}");
        }
        assert!(result.refuted());
        assert!(!result.confirmed());
        assert_eq!(result.verdict(), "refuted");
        assert_eq!(result.refuting(), [true; STRATA]);

        // The correction is applied rather than the bare level: four
        // bands, so the thresholds step 0.025/4, /3, /2, /1.
        let mut thresholds: Vec<f64> = result.strata.iter().map(|s| s.holm_threshold).collect();
        thresholds.sort_by(f64::total_cmp);
        assert!((thresholds[0] - result.alpha / 4.0).abs() < 1e-12);
        assert!((thresholds[1] - result.alpha / 3.0).abs() < 1e-12);
        assert!((thresholds[2] - result.alpha / 2.0).abs() < 1e-12);
        assert!((thresholds[3] - result.alpha).abs() < 1e-12);
        assert!((result.alpha - (1.0 - CONFIDENCE) / 2.0).abs() < 1e-12);

        // The reported levels and thresholds are the ones the verdicts
        // came from. On this fixture that is the whole of what it says:
        // every band's flip rate is the same in every resample, so
        // every level is exactly zero and corrected and uncorrected
        // cannot disagree. The fixture where they do is
        // `the_correction_and_the_bare_interval_disagree`.
        assert_eq!(result.strata.map(|s| s.p_below_zero), [0.0; STRATA]);
        let levels: [f64; STRATA] = std::array::from_fn(|i| result.strata[i].p_below_zero);
        let (rejected, _) = holm(levels, result.alpha);
        for (band, want) in result.strata.iter().zip(rejected) {
            assert_eq!(band.refuted, want, "{band:?}");
        }
    }

    /// The one fixture where the correction changes an answer: every
    /// band refutes on its own interval, and none refutes once the four
    /// are corrected together.
    ///
    /// Every other H20 fixture here keys its flips on the position
    /// alone, so a band's rate is the same in every resample and its
    /// level is exactly zero or one. Corrected and uncorrected coincide
    /// on all of them, and none of them would notice a `refuted` read
    /// off `Interval::excludes_zero_from_below` instead of off Holm.
    ///
    /// Here one game in eight carries the whole disagreement: `A` flips
    /// on none of that game's positions and seven eighths of every
    /// other game's, `AB` the other way round. A draw's sign therefore
    /// turns on how many times that one game is picked — the quantity
    /// clears zero at four or more, which is 1.1% of draws in
    /// expectation and 1.2% at this seed. That is under the uncorrected
    /// 2.5% and over Holm's tightest 0.625%, so the two answers differ,
    /// and the month reads undetermined where the bare interval would
    /// read refuted.
    ///
    /// The floor is held at exactly 0.125 in every draw — each
    /// replicate flips on one more of its band's eight positions than
    /// its partner does, in every game, so the gap survives resampling
    /// unchanged and the sign above is arithmetic rather than an
    /// accident of which games were drawn.
    #[test]
    fn the_correction_and_the_bare_interval_disagree() {
        /// Flip the first `count(game)` positions of each band.
        ///
        /// Thirty-two positions a game against four margin values puts
        /// eight of each band in every game, and they arrive in order,
        /// so `count` is the band's flip count for that game exactly.
        fn banded(count: impl Fn(usize) -> usize) -> impl Fn(usize, usize) -> bool {
            move |game, ix| ix / MARGINS.len() < count(game)
        }
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let arms = four_arms(
            legality_walk(
                8,
                32,
                |_, _| 1.0,
                banded(|g| if g == 0 { 0 } else { 7 }),
                margin,
            ),
            legality_walk(
                8,
                32,
                |_, _| 1.1,
                banded(|g| if g == 0 { 1 } else { 8 }),
                margin,
            ),
            legality_walk(
                8,
                32,
                |_, _| 0.9,
                banded(|g| if g == 0 { 6 } else { 0 }),
                margin,
            ),
            legality_walk(
                8,
                32,
                |_, _| 0.8,
                banded(|g| if g == 0 { 7 } else { 1 }),
                margin,
            ),
        );
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();
        let tightest = result.alpha / STRATA as f64;

        for band in &result.strata {
            assert_eq!(band.positions_a, 64, "{band:?}");
            assert_eq!(band.positions_ab, 64, "{band:?}");
            assert!((band.floor - 0.125).abs() < 1e-12, "{band:?}");
            // On the sample as walked each game appears once: `A` flips
            // on 49 of its 64, `AB` on 6.
            assert!((band.flip_a - 49.0 / 64.0).abs() < 1e-12, "{band:?}");
            assert!((band.flip_ab - 6.0 / 64.0).abs() < 1e-12, "{band:?}");
            assert!(
                (band.interval.point + 35.0 / 64.0).abs() < 1e-12,
                "{band:?}"
            );

            assert!(
                band.interval.excludes_zero_from_below(),
                "uncorrected, this band refutes: {band:?}"
            );
            assert!(
                band.p_below_zero > tightest,
                "and its level fails Holm's tightest threshold {tightest}: {band:?}"
            );
            assert!(
                band.p_below_zero < result.alpha,
                "while clearing the uncorrected one: {band:?}"
            );
            assert!(!band.refuted, "so the correction refuses it: {band:?}");
            assert!(!band.confirmed, "{band:?}");
        }

        assert!(!result.refuted());
        assert_eq!(result.refuting(), [false; STRATA]);
        assert_eq!(
            result.verdict(),
            "undetermined",
            "a verdict read off the intervals alone would say refuted"
        );

        // The step-down is what refuses the band handed the uncorrected
        // threshold: its own level clears the threshold it was given,
        // and it is rejected anyway only if the procedure never halted.
        let loosest = result
            .strata
            .iter()
            .max_by(|a, b| a.holm_threshold.total_cmp(&b.holm_threshold))
            .expect("four bands");
        assert!((loosest.holm_threshold - result.alpha).abs() < 1e-12);
        assert!(
            loosest.p_below_zero <= loosest.holm_threshold,
            "{loosest:?}"
        );
        assert!(!loosest.refuted, "{loosest:?}");
    }

    /// The asymmetry `§4` records and does not fix: a real cost that is
    /// smaller than the floor reads as "steerability survives".
    ///
    /// `AB` flips on three quarters of each band against `A`'s
    /// everywhere, so it genuinely flips less — a margin of `-0.25`
    /// against a floor of 0.5, which the tolerance absorbs. A confirm
    /// here is the weaker of the plan's two claims, and this is why.
    #[test]
    fn a_cost_smaller_than_the_floor_still_confirms_h20() {
        let arms = stratified_arms(|_| true, |ix| ix < 8, |ix| ix < 12, |ix| ix < 12);
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();

        for band in &result.strata {
            assert!(
                band.flip_ab < band.flip_a,
                "the cost has to be real for this test to be about anything: {band:?}"
            );
            assert!(
                (band.flip_ab - band.flip_a + 0.25).abs() < 1e-12,
                "{band:?}"
            );
            assert!((band.floor - 0.5).abs() < 1e-12, "{band:?}");
            assert!((band.interval.point - 0.25).abs() < 1e-12, "{band:?}");
            assert!(band.confirmed, "{band:?}");
            assert!(!band.refuted, "{band:?}");
        }
        assert_eq!(result.verdict(), "confirmed");
    }

    /// Under the null the two arms flip alike, and the criterion is
    /// `margin + F_flip`, so a confirm fires on a positive floor alone.
    /// That is `§4`'s recorded asymmetry rather than a defect, and it
    /// is asserted here so that removing the floor would be visible as
    /// a change of verdict rather than as a shift in a number.
    ///
    /// What must **not** happen under the null is a refutation.
    #[test]
    fn under_the_null_h20_does_not_refute() {
        let arms = stratified_arms(|ix| ix < 8, |ix| ix < 4, |ix| ix >= 8, |ix| ix >= 12);
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();

        for band in &result.strata {
            assert!((band.flip_a - 0.5).abs() < 1e-12, "{band:?}");
            assert!((band.flip_ab - 0.5).abs() < 1e-12, "{band:?}");
            assert!((band.floor - 0.25).abs() < 1e-12, "{band:?}");
            assert!(!band.refuted, "the null must not refute: {band:?}");
        }
        assert!(!result.refuted());
        assert_eq!(result.refuting(), [false; STRATA]);
    }

    /// The strata hold different positions for the two arms, and that
    /// is the design rather than a defect: the cut points are common,
    /// so a uniformly sharper arm puts more of its positions in the
    /// high bands. Per-arm quantiles would match on rank and move the
    /// sharpness difference inside every band instead of holding it
    /// still.
    #[test]
    fn a_sharper_arm_fills_the_high_bands_rather_than_matching_on_rank() {
        // `AB` is sharper over three quarters of each game: those
        // positions' margins are one step up from `A`'s, and the last
        // four are left alone so that the bottom band still catches
        // something for it.
        let sharper = |ix: usize| {
            if ix < 12 {
                MARGINS
                    .get((ix % MARGINS.len() + 1).min(MARGINS.len() - 1))
                    .copied()
            } else {
                cycled(ix)
            }
        };
        let arms = four_arms(
            banded_walk(8, 16, 1.0, |ix| ix % 4 == 0, cycled),
            banded_walk(8, 16, 1.1, |ix| ix % 4 == 1, cycled),
            banded_walk(8, 16, 0.9, |ix| ix % 4 == 2, sharper),
            banded_walk(8, 16, 0.8, |ix| ix % 4 == 3, sharper),
        );
        let result = h20(&arms, 1.0, DRAWS, SEED).unwrap();

        let lowest = &result.strata[0];
        let highest = &result.strata[STRATA - 1];
        assert!(
            lowest.positions_ab < lowest.positions_a,
            "the sharper arm should be under-represented at the bottom: {lowest:?}"
        );
        assert!(
            highest.positions_ab > highest.positions_a,
            "and over-represented at the top: {highest:?}"
        );
        // The bands still partition each arm's positions: nothing is
        // counted twice and nothing carrying a margin is dropped.
        let total_a: usize = result.strata.iter().map(|s| s.positions_a).sum();
        let total_ab: usize = result.strata.iter().map(|s| s.positions_ab).sum();
        assert_eq!(total_a, result.positions);
        assert_eq!(total_ab, result.positions);
    }

    /// A band that catches nothing for one of the arms says so, rather
    /// than reporting a flip rate over no positions.
    #[test]
    fn an_empty_band_is_refused() {
        // Every margin identical, so all three cut points land on the
        // same value and the three bands below it are empty.
        let flat = |_: usize| Some(0.25);
        let arms = four_arms(
            banded_walk(4, 8, 1.0, |ix| ix % 4 == 0, flat),
            banded_walk(4, 8, 1.1, |ix| ix % 4 == 1, flat),
            banded_walk(4, 8, 0.9, |ix| ix % 4 == 2, flat),
            banded_walk(4, 8, 0.8, |ix| ix % 4 == 3, flat),
        );
        let err = h20(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::EmptyStratum {
                    stratum: 0,
                    arm: ARM_A
                }
            ),
            "{err:?}"
        );
    }

    /// A set carrying no margin at all is reported as that, rather than
    /// as an empty band: the margin arrived at record format version 2,
    /// and a version 1 walk reaching here has nothing to cut on.
    #[test]
    fn a_set_with_no_margins_has_no_cut_points() {
        let none = |_: usize| None;
        let arms = four_arms(
            banded_walk(4, 8, 1.0, |ix| ix % 4 == 0, none),
            banded_walk(4, 8, 1.1, |ix| ix % 4 == 1, none),
            banded_walk(4, 8, 0.9, |ix| ix % 4 == 2, none),
            banded_walk(4, 8, 0.8, |ix| ix % 4 == 3, none),
        );
        let err = h20(&arms, 1.0, DRAWS, SEED).unwrap_err();
        assert!(
            matches!(
                err,
                StatError::NoMargins {
                    a: ARM_A,
                    b: ARM_AB
                }
            ),
            "{err:?}"
        );
    }

    /// What the correction is for: a band whose level would refute on
    /// its own does not refute once it is one of four.
    ///
    /// 0.02 clears the uncorrected 0.025 — which is the statement a 95%
    /// interval lying below zero makes — and fails Holm's first
    /// threshold of 0.00625. Without the correction the plan's "refute
    /// in any stratum" would be a criterion that fires roughly four
    /// times as often as it claims to.
    #[test]
    fn holm_refuses_a_band_that_would_refute_on_its_own() {
        let alpha = (1.0 - CONFIDENCE) / 2.0;
        let (rejected, thresholds) = holm([0.02, 0.5, 0.5, 0.5], alpha);
        assert!(0.02 < alpha, "the band clears the uncorrected level");
        assert_eq!(rejected, [false; STRATA]);
        assert!((thresholds[0] - alpha / 4.0).abs() < 1e-12);
    }

    /// And it does reject, so the test above is not passing for want of
    /// the correction ever letting anything through.
    #[test]
    fn holm_rejects_a_band_that_clears_its_own_threshold() {
        let alpha = (1.0 - CONFIDENCE) / 2.0;
        let (rejected, _) = holm([0.001, 0.5, 0.5, 0.5], alpha);
        assert_eq!(rejected, [true, false, false, false]);

        // All four at once, which is what the fixtures above produce.
        let (all, _) = holm([0.0; STRATA], alpha);
        assert_eq!(all, [true; STRATA]);
    }

    /// Step-down, not step-up: the procedure stops at the first band
    /// that fails its threshold, and rejects nothing after it even when
    /// a later band would clear the threshold it was handed.
    ///
    /// Levels 0.0001 and 0.001 clear 0.00625 and 0.00833. 0.02 then
    /// fails 0.0125 and the procedure halts — so 0.024, which sits
    /// under its own 0.025, is refused. Without the halt this would be
    /// a per-band test wearing a correction's name.
    #[test]
    fn holm_stops_at_the_first_band_that_fails_its_threshold() {
        let alpha = (1.0 - CONFIDENCE) / 2.0;
        let (rejected, thresholds) = holm([0.000_1, 0.02, 0.001, 0.024], alpha);
        assert_eq!(rejected, [true, false, true, false]);
        assert!(
            0.024 <= thresholds[3],
            "the last band clears the threshold it was handed ({}) and is still not rejected",
            thresholds[3]
        );
        assert!((thresholds[3] - alpha).abs() < 1e-12);
        assert!((thresholds[1] - alpha / 2.0).abs() < 1e-12);
    }

    /// The legality set's top-1 tolerance is wider than plan 02's, and
    /// the widening is load-bearing rather than cosmetic: this arm sits
    /// 0.0208 from the control, which the seed alone can produce
    /// (0.0146 / 0.0093 were measured), so plan 02's 0.02 would exclude
    /// it and `§5.1`'s 0.03 admits it.
    #[test]
    fn the_legality_top1_gate_admits_what_the_narrower_one_would_exclude() {
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let mut subject = legality_walk(8, 12, |_, _| 1.0, |_, ix| ix % 4 == 1, margin);
        // Every position scores 1 of 3 bands; six of the 96 score 2,
        // so this arm's mean sits exactly 2/96 above the control's —
        // 0.02083, inside 0.03 and outside 0.02.
        for record in subject.records.iter_mut().take(6) {
            for at in record.at.iter_mut() {
                at.top1 = Some(vec![true, true, false]);
            }
        }
        let control = legality_walk(8, 12, |_, _| 1.2, |_, ix| ix % 4 == 0, margin);
        let arms = AlignedArms::new(vec![
            legality_arm(CONTROL, control),
            legality_arm(ARM_A, subject),
        ])
        .unwrap();

        let wide = gate_top1_legality(&arms, ARM_A, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(wide.tolerance, TOP1_GATE_LEGALITY);
        assert!(
            (wide.interval.point - 2.0 / 96.0).abs() < 1e-12,
            "{:?}",
            wide.interval
        );
        assert!(wide.passes(), "{wide:?}");
        assert_eq!(wide.verdict(), "pass");

        // The same arms under plan 02's tolerance, which is the gate
        // `§5.1` widened and the reason it did.
        let narrow = gate_top1(&arms, ARM_A, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(narrow.tolerance, TOP1_GATE);
        assert_eq!(narrow.interval, wide.interval);
        assert!(!narrow.passes(), "{narrow:?}");
        const { assert!(TOP1_GATE_LEGALITY > TOP1_GATE) };
    }

    /// The gate is a difference on one draw, so the control against
    /// itself is identically zero and the gate is degenerate at pass.
    #[test]
    fn the_legality_gate_is_a_difference_against_the_control() {
        let margin = |ix: usize| MARGINS[ix % MARGINS.len()];
        let arms = AlignedArms::new(vec![
            legality_arm(
                CONTROL,
                legality_walk(6, 8, |_, _| 1.2, |_, ix| ix % 4 == 0, margin),
            ),
            legality_arm(
                ARM_AB,
                legality_walk(6, 8, |_, _| 0.9, |_, ix| ix % 4 == 2, margin),
            ),
        ])
        .unwrap();
        let gate = gate_top1_legality(&arms, CONTROL, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(gate.interval.point, 0.0);
        assert_eq!(gate.interval.low, 0.0);
        assert_eq!(gate.interval.high, 0.0);
        assert!(gate.passes());
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

    const ECO_B: &str = "<eco:B>";
    const ECO_C: &str = "<eco:C>";

    /// A jouseki walk: per-position conditioning, two family bands,
    /// and a header that states whose games these are. `hit(band, game,
    /// ix)` decides each band's top-1 column; `js` marks the arm so
    /// two otherwise-identical fixtures are not refused as one file
    /// read twice (nothing the jouseki statistics read consumes it).
    fn jouseki_walk(
        games: usize,
        per_game: usize,
        family: &str,
        js: f64,
        hit: impl Fn(usize, usize, usize) -> bool,
    ) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ix in 0..per_game {
                records.push(PositionRecord {
                    game,
                    ply: ix * 2,
                    n_legal: None,
                    at: vec![GammaRecord {
                        flipped: false,
                        widest_js: js,
                        legal_mass: 0.9,
                        top1: Some(vec![hit(0, game, ix), hit(1, game, ix)]),
                        ce: None,
                        top2_margin: None,
                    }],
                });
            }
        }
        let mut header = header(records.len(), games);
        header.encoding = CondEncoding::EveryPosition;
        header.bands = vec![ECO_B.into(), ECO_C.into()];
        header.gammas = GAMMAS.to_vec();
        header.games_of = Some(vec![family.into()]);
        Walk { header, records }
    }

    fn jouseki_arms(mut j: Walk, mut j_b: Walk) -> AlignedArms {
        j.header.ckpt = "/root/ckpt/J/run.safetensors".into();
        j_b.header.ckpt = "/root/ckpt/J-b/run.safetensors".into();
        AlignedArms::new(vec![(ARM_J.to_string(), j), (ARM_J_B.to_string(), j_b)])
            .expect("the fixture arms share a position stream")
    }

    /// A walk too old to state whose games it walked is refused on the
    /// version, not read on a default.
    #[test]
    fn a_pre_v5_jouseki_walk_is_refused() {
        let mut j = jouseki_walk(4, 6, ECO_B, 0.0, |band, _, _| band == 0);
        j.header.version = 4;
        j.header.games_of = None;
        let mut j_b = jouseki_walk(4, 6, ECO_B, 0.001, |band, _, _| band == 0);
        j_b.header.version = 4;
        j_b.header.games_of = None;
        let arms = jouseki_arms(j, j_b);
        assert!(matches!(
            check_jouseki_roles(&arms, ECO_B),
            Err(StatError::RecordsPredateWalkFilter { needed: 5, .. })
        ));
    }

    /// The swap the field exists to refuse: the other family's walks in
    /// this family's slot. The cost would reverse sign with every
    /// figure well-formed, so the mismatch has to be an error.
    #[test]
    fn a_walk_of_the_other_familys_games_is_refused() {
        let arms = jouseki_arms(
            jouseki_walk(4, 6, ECO_C, 0.0, |band, _, _| band == 0),
            jouseki_walk(4, 6, ECO_C, 0.001, |band, _, _| band == 0),
        );
        assert!(matches!(
            check_jouseki_roles(&arms, ECO_B),
            Err(StatError::WrongWalkedGamesForRole { .. })
        ));
        // And read as its own family it passes, naming B's column as
        // correct is index 0 only when the family is B.
        let arms = jouseki_arms(
            jouseki_walk(4, 6, ECO_C, 0.0, |band, _, _| band == 0),
            jouseki_walk(4, 6, ECO_C, 0.001, |band, _, _| band == 0),
        );
        assert_eq!(check_jouseki_roles(&arms, ECO_C).unwrap(), (1, 0));
    }

    /// A version 5 walk that states "deliberately unfiltered" is still
    /// not a family's walk — `Some(vec![])` is a statement, and it is
    /// the wrong one for a slot that reads games of one family.
    #[test]
    fn a_deliberately_unfiltered_walk_is_not_a_familys_walk() {
        let mut j = jouseki_walk(4, 6, ECO_B, 0.0, |band, _, _| band == 0);
        j.header.games_of = Some(vec![]);
        let mut j_b = jouseki_walk(4, 6, ECO_B, 0.001, |band, _, _| band == 0);
        j_b.header.games_of = Some(vec![]);
        let arms = jouseki_arms(j, j_b);
        assert!(matches!(
            check_jouseki_roles(&arms, ECO_B),
            Err(StatError::WrongWalkedGamesForRole { .. })
        ));
    }

    /// A third band would make "the wrong token" a post-hoc choice, so
    /// the registered two-family design is enforced.
    #[test]
    fn a_third_band_is_refused() {
        let mut j = jouseki_walk(4, 6, ECO_B, 0.0, |band, _, _| band == 0);
        j.header.bands.push("<eco:A>".into());
        let mut j_b = jouseki_walk(4, 6, ECO_B, 0.001, |band, _, _| band == 0);
        j_b.header.bands.push("<eco:A>".into());
        let arms = jouseki_arms(j, j_b);
        assert!(matches!(
            check_jouseki_roles(&arms, ECO_B),
            Err(StatError::JousekiNeedsTwoBands { found: 3 })
        ));
    }

    /// The cost on a fixture fixed by construction: the correct column
    /// hits everywhere, the wrong one nowhere, both seeds agree — cost
    /// 1, floor 0, confirmed. And the deliberate near-miss: a model
    /// whose columns agree everywhere prices the cost at exactly zero,
    /// which confirms nothing.
    #[test]
    fn h29_family_prices_the_cost_against_the_walks_own_family() {
        let strong = |band: usize, _: usize, _: usize| band == 0;
        let arms = jouseki_arms(
            jouseki_walk(6, 8, ECO_B, 0.0, strong),
            jouseki_walk(6, 8, ECO_B, 0.001, strong),
        );
        let family = h29_family(&arms, ECO_B, 1.0, 200, SEED).unwrap();
        assert_eq!(family.cost_j, 1.0);
        assert_eq!(family.floor, 0.0);
        assert!(family.confirmed());
        assert_eq!(family.shallow_cost_j, Some(1.0));

        let flat = |_: usize, _: usize, _: usize| true;
        let arms = jouseki_arms(
            jouseki_walk(6, 8, ECO_B, 0.0, flat),
            jouseki_walk(6, 8, ECO_B, 0.001, flat),
        );
        let family = h29_family(&arms, ECO_B, 1.0, 200, SEED).unwrap();
        assert_eq!(family.cost_j, 0.0);
        assert!(!family.confirmed());
        assert!(!family.refuted());
    }

    /// Plan 06 §3's min-logic at the verdict layer: one steering family
    /// is not a confirmation of the month.
    #[test]
    fn the_month_confirms_only_when_both_families_do() {
        let steer_b = |band: usize, _: usize, _: usize| band == 0;
        let steer_c = |band: usize, _: usize, _: usize| band == 1;
        let flat = |_: usize, _: usize, _: usize| true;

        let b_arms = jouseki_arms(
            jouseki_walk(6, 8, ECO_B, 0.0, steer_b),
            jouseki_walk(6, 8, ECO_B, 0.001, steer_b),
        );
        let c_flat = jouseki_arms(
            jouseki_walk(6, 8, ECO_C, 0.0, flat),
            jouseki_walk(6, 8, ECO_C, 0.001, flat),
        );
        let one_sided = h29([(&b_arms, ECO_B), (&c_flat, ECO_C)], 1.0, 200, SEED).unwrap();
        assert_eq!(one_sided.verdict(), "undetermined");

        let c_arms = jouseki_arms(
            jouseki_walk(6, 8, ECO_C, 0.0, steer_c),
            jouseki_walk(6, 8, ECO_C, 0.001, steer_c),
        );
        let both = h29([(&b_arms, ECO_B), (&c_arms, ECO_C)], 1.0, 200, SEED).unwrap();
        assert_eq!(both.verdict(), "confirmed");
    }

    /// H31 judges inside ply 0-19 and nothing past it: a fixture that
    /// steers only in the opening prices at exactly 1 there, while the
    /// all-ply reading is diluted by the deep positions.
    #[test]
    fn h31_reads_the_shallow_stratum_and_only_it() {
        let opening_only = |band: usize, _: usize, ix: usize| band == 0 && ix < 10;
        let arms = jouseki_arms(
            jouseki_walk(6, 15, ECO_B, 0.0, opening_only),
            jouseki_walk(6, 15, ECO_B, 0.001, opening_only),
        );
        let shallow = h31_family(&arms, ECO_B, 1.0, 200, SEED).unwrap();
        assert_eq!(shallow.cost_j, 1.0);
        assert_eq!(shallow.positions, 6 * 10);
        assert!(shallow.confirmed());

        let all_ply = h29_family(&arms, ECO_B, 1.0, 200, SEED).unwrap();
        assert!(all_ply.cost_j < 1.0);
        assert_eq!(all_ply.positions, 6 * 15);
    }

    const ELO_LO: &str = "<elo:1100-1299>";
    const ELO_HI: &str = "<elo:1900-2099>";

    /// The four combination labels in the batch's own order — eco
    /// varies slower — as `chess_cond` writes them for a 2×2 model.
    fn combo_bands() -> Vec<String> {
        vec![
            format!("{ECO_B}+{ELO_LO}"),
            format!("{ECO_B}+{ELO_HI}"),
            format!("{ECO_C}+{ELO_LO}"),
            format!("{ECO_C}+{ELO_HI}"),
        ]
    }

    /// A duo walk over one cell's games: four combination columns,
    /// `games_of` naming the two cell tokens, `hit(combo, game, ix)`
    /// deciding each column's top-1.
    fn duo_walk(
        games: usize,
        per_game: usize,
        cell: [&str; 2],
        js: f64,
        hit: impl Fn(usize, usize, usize) -> bool,
    ) -> Walk {
        let mut records = Vec::new();
        for game in 0..games {
            for ix in 0..per_game {
                records.push(PositionRecord {
                    game,
                    ply: ix * 2,
                    n_legal: None,
                    at: vec![GammaRecord {
                        flipped: false,
                        widest_js: js,
                        legal_mass: 0.9,
                        top1: Some((0..4).map(|c| hit(c, game, ix)).collect()),
                        ce: None,
                        top2_margin: None,
                    }],
                });
            }
        }
        let mut header = header(records.len(), games);
        header.encoding = CondEncoding::EveryPosition;
        header.bands = combo_bands();
        header.gammas = GAMMAS.to_vec();
        header.games_of = Some(cell.iter().map(|t| t.to_string()).collect());
        Walk { header, records }
    }

    fn duo_arms(k: Walk, k_b: Walk) -> AlignedArms {
        let mut k = k;
        let mut k_b = k_b;
        k.header.ckpt = "/root/ckpt/K/run.safetensors".into();
        k_b.header.ckpt = "/root/ckpt/K-b/run.safetensors".into();
        AlignedArms::new(vec![(ARM_K.to_string(), k), (ARM_K_B.to_string(), k_b)])
            .expect("the fixture arms share a position stream")
    }

    /// The cell resolves to its own column and each wrong column
    /// falsifies exactly one slot — including when the walk's
    /// `games_of` lists the tokens in the other order.
    #[test]
    fn a_duo_cell_resolves_its_column_and_both_single_slot_wrongs() {
        for cell in [[ECO_B, ELO_HI], [ELO_HI, ECO_B]] {
            let arms = duo_arms(
                duo_walk(4, 6, cell, 0.0, |c, _, _| c == 1),
                duo_walk(4, 6, cell, 0.001, |c, _, _| c == 1),
            );
            let resolved = check_duo_roles(&arms).unwrap();
            assert_eq!(resolved.label, format!("{ECO_B}+{ELO_HI}"));
            assert_eq!(resolved.correct, 1);
            // Wrong slot 0 (the eco part) is C+hi; wrong slot 1 (the
            // elo part) is B+lo.
            assert_eq!(resolved.wrong[0], (format!("{ECO_C}+{ELO_HI}"), 3));
            assert_eq!(resolved.wrong[1], (format!("{ECO_B}+{ELO_LO}"), 0));
        }
    }

    /// A cell no column matches is refused, as is a walk claiming one
    /// token where a cell is two.
    #[test]
    fn a_duo_walk_of_no_known_cell_is_refused() {
        let arms = duo_arms(
            duo_walk(4, 6, [ECO_B, "<elo:other>"], 0.0, |c, _, _| c == 0),
            duo_walk(4, 6, [ECO_B, "<elo:other>"], 0.001, |c, _, _| c == 0),
        );
        assert!(matches!(
            check_duo_roles(&arms),
            Err(StatError::CellNotACombo { .. })
        ));

        let mut k = duo_walk(4, 6, [ECO_B, ELO_LO], 0.0, |c, _, _| c == 0);
        k.header.games_of = Some(vec![ECO_B.into()]);
        let mut k_b = duo_walk(4, 6, [ECO_B, ELO_LO], 0.001, |c, _, _| c == 0);
        k_b.header.games_of = Some(vec![ECO_B.into()]);
        let arms = duo_arms(k, k_b);
        assert!(matches!(
            check_duo_roles(&arms),
            Err(StatError::WrongWalkedGamesForRole { .. })
        ));
    }

    /// The grid min-logic: a model that ignores one slot confirms that
    /// slot's cost at zero, and the month cannot confirm through it.
    #[test]
    fn a_slot_the_model_ignores_keeps_the_month_from_confirming() {
        // Column top-1 depends only on the eco half: 0/1 share fate,
        // 2/3 share fate. Falsifying eco costs 1; falsifying elo
        // costs 0.
        let eco_only = |c: usize, _: usize, _: usize| c < 2;
        let arms = duo_arms(
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.0, eco_only),
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.001, eco_only),
        );
        let [eco_axis, elo_axis] = h30_cell(&arms, 1.0, 200, SEED).unwrap();
        assert_eq!(eco_axis.cost_j, 1.0);
        assert!(eco_axis.confirmed());
        assert_eq!(elo_axis.cost_j, 0.0);
        assert!(!elo_axis.confirmed());

        let month = h30(&[&arms], 1.0, 200, SEED).unwrap();
        assert_eq!(month.verdict(), "undetermined");

        // Both slots read: every falsification costs.
        let both = |c: usize, _: usize, _: usize| c == 0;
        let arms = duo_arms(
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.0, both),
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.001, both),
        );
        let month = h30(&[&arms], 1.0, 200, SEED).unwrap();
        assert_eq!(month.verdict(), "confirmed");

        // And the same pair fed twice is not two cells.
        assert!(matches!(
            h30(&[&arms, &arms], 1.0, 200, SEED),
            Err(StatError::DuplicateCell { .. })
        ));
    }

    /// H32 pools the cells per axis: with one slot ignored everywhere,
    /// the pooled axis prices at zero and the month cannot confirm;
    /// with both slots read in both cells, it does. The per-cell grid
    /// stays visible as description.
    #[test]
    fn h32_pools_the_cells_and_still_requires_both_axes() {
        let eco_only = |c: usize, _: usize, _: usize| c < 2;
        let cell_b = duo_arms(
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.0, eco_only),
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.001, eco_only),
        );
        let cell_c = duo_arms(
            duo_walk(6, 8, [ECO_C, ELO_HI], 0.0, |c, _, _| c >= 2),
            duo_walk(6, 8, [ECO_C, ELO_HI], 0.001, |c, _, _| c >= 2),
        );
        let month = h32(&[&cell_b, &cell_c], 1.0, 200, SEED).unwrap();
        assert_eq!(month.axes[0].d_pool_k, 1.0);
        assert_eq!(month.axes[1].d_pool_k, 0.0);
        assert!(month.axes[0].confirmed());
        assert!(!month.axes[1].confirmed());
        assert_eq!(month.verdict(), "undetermined");
        assert_eq!(month.axes[0].per_cell.len(), 2);

        let both_b = duo_arms(
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.0, |c, _, _| c == 0),
            duo_walk(6, 8, [ECO_B, ELO_LO], 0.001, |c, _, _| c == 0),
        );
        let both_c = duo_arms(
            duo_walk(6, 8, [ECO_C, ELO_HI], 0.0, |c, _, _| c == 3),
            duo_walk(6, 8, [ECO_C, ELO_HI], 0.001, |c, _, _| c == 3),
        );
        let month = h32(&[&both_b, &both_c], 1.0, 200, SEED).unwrap();
        assert_eq!(month.verdict(), "confirmed");
    }

    /// Two identical runs sit at zero distance, and the gate passes.
    #[test]
    fn identical_jouseki_seeds_pass_the_admission_gate() {
        let hit = |band: usize, g: usize, ix: usize| band == 0 && (g + ix).is_multiple_of(2);
        let arms = jouseki_arms(
            jouseki_walk(6, 8, ECO_B, 0.0, hit),
            jouseki_walk(6, 8, ECO_B, 0.001, hit),
        );
        let gate = gate_top1_jouseki(&arms, ECO_B, 1.0, DRAWS, SEED).unwrap();
        assert_eq!(gate.interval.point, 0.0);
        assert!(gate.passes());
    }
}
