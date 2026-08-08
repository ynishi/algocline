//! Measure what a condition token does, and what amplifying it does.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_cond -- <ckpt> <holdout.pgn> [side] [max_positions] [walk_band] [gammas] [records.jsonl]
//! ```
//!
//! `gammas` is a comma-separated guidance sweep, default
//! `1,1.5,2,3,4,6,8`. One is the unguided model, so the sweep always
//! contains the baseline it is read against.
//!
//! # The summary and the records are for different readers
//!
//! What this prints is what a person reads while a run is going. It has
//! moved with the measurement rather than staying put. Two changes are
//! unconditional: the header now names the conditioning convention the
//! checkpoint records, and the walk line separates the games that
//! contributed a position from the games the filter accepted, because
//! those two counts are different numbers and the error bars resample
//! the first. A third is not — passing the seventh argument adds a
//! `records` line naming the file and the counts written to it, so an
//! operator can see the walk landed without opening it. Anyone diffing
//! an old run's output against a new one meets all three. What
//! `records.jsonl` holds is what a
//! *second* run of this program has to be joined against, and a summary
//! cannot serve that purpose: every hypothesis in the plan is a
//! difference between arms scored on the same positions, and by the time
//! a mean has been printed the positions it came from are gone.
//!
//! So passing a seventh argument also writes one line per position —
//! the game it came from, its ply, and per gamma whether the bands
//! flipped, how far apart the widest pair was, and which bands matched
//! the human. `chess_stats` reads several of those files, checks they
//! describe the same positions, and puts error bars on the differences
//! ([`algocline_nn::chess::records`],
//! [`algocline_nn::chess::steerability`]).
//!
//! Without the game index the error bars are not available at all: 3,000
//! positions come from roughly 95 games and positions within one game
//! are correlated, so a bootstrap has to resample games rather than
//! positions and nothing else here retained which game a position was
//! from.
//!
//! # Why not move match
//!
//! Top-1 move match cannot see a condition token. On the two-band
//! checkpoint it read 0.2088 asked as the low band against low-band
//! games and 0.2080 asked as the high band — 0.0008 apart — while the
//! same model in the Open Sicilian after 3...cxd4 put 0.116 on Qxd4 as
//! the low band and 0.018 as the high one. Both bands answer Nxd4
//! first, so top-1 sees nothing and the 6x change sits underneath it.
//!
//! What separates the bands is the shape of the distribution, so that
//! is what gets measured: Jensen-Shannon divergence between the
//! model's legal-move distributions under each token, at the same
//! position, in bits so the bound is `[0, 1]`.
//!
//! # The references it is read against
//!
//! A divergence has no meaning alone. Each band is also measured
//! against a uniform draw over the same legal moves, which is what the
//! model looks like knowing nothing — a band-to-band figure that is a
//! small fraction of that says the token nudges the distribution, one
//! of the same order says it moves it as much as knowing chess does.
//!
//! Divergences are split by whether the bands agree on the top move,
//! since that is precisely the split move match is blind to, and by
//! depth, since a condition carried by one token at the far left of the
//! row may or may not survive to move 40.
//!
//! # The sweep
//!
//! Guidance ([`algocline_nn::chess::guide`]) amplifies whatever the
//! token was doing. Raising gamma should raise the divergence; the
//! question is what it costs, so legal-move mass and fidelity to the
//! human move are reported alongside it at every gamma. Over-guidance
//! is a documented failure mode and here it has an obvious shape —
//! extrapolated logits are under no obligation to stay on moves that
//! exist in the position.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device};
use cozy_chess::Board;

use algocline_nn::arch::Gpt2Model;
use algocline_nn::chess::batch::BandBatch;
use algocline_nn::chess::corpus::ScoredSide;
use algocline_nn::chess::filter::GameFilter;
use algocline_nn::chess::guide::{guide_logits, mean_logits};
use algocline_nn::chess::pgn::{resolve_san, san_tokens, uci_standard, PgnReader};
use algocline_nn::chess::records::{
    ce_over_legal, top2_margin, GammaRecord, PositionRecord, Walk, WalkHeader, FORMAT_VERSION,
};
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::window::{play_row, COND_PREFIX_LEN};
use algocline_nn::chess::{open_reader_shape, ModelShape};
use algocline_nn::metric::{self, MetricError};

/// Jensen-Shannon divergence in bits, via [`algocline_nn::metric::js`].
///
/// The crate's implementation reports nats and carries its own tests;
/// bits are what a reader wants here, since the bound becomes `[0, 1]`
/// — zero when two distributions agree everywhere, one when they put
/// their mass on disjoint moves.
fn js_bits(p: &[f32], q: &[f32]) -> Result<f64, MetricError> {
    Ok(metric::js(p, q)? as f64 / std::f64::consts::LN_2)
}

/// Softmax over a logit row.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    exp.iter().map(|e| e / total).collect()
}

/// Renormalise to sum to one, or fall back to uniform on no mass.
///
/// `metric::js` validates that its inputs are distributions, so this
/// is a precondition rather than a nicety: the softmax restricted to
/// the legal subset sums to whatever mass the model left there.
fn normalise(v: &[f32]) -> Vec<f32> {
    let total: f32 = v.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / v.len() as f32; v.len()];
    }
    v.iter().map(|x| x / total).collect()
}

/// Running mean, kept as a pair so an empty set reads as absent rather
/// than as zero.
#[derive(Default, Clone)]
struct Mean {
    sum: f64,
    n: usize,
}

impl Mean {
    fn push(&mut self, x: f64) {
        self.sum += x;
        self.n += 1;
    }
    fn value(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}

/// Everything accumulated at one guidance strength.
struct GammaStats {
    gamma: f32,
    /// Divergence between the first and last band.
    widest_js: Mean,
    /// Softmax mass landing on legal moves, averaged over bands.
    legal_mass: Mean,
    /// Share of positions where a band's top move equals the human's.
    top1: Vec<Mean>,
    /// Positions where the bands disagree on the top move.
    top1_differs: usize,
    /// How far ahead the leader was, where the bands disagreed.
    ///
    /// A flip counted on a first place that led the second by a
    /// hair says nothing about the condition: two orderings of a
    /// near-tie are one arithmetic step apart. This records the gap
    /// between first and second within each band's legal-normalised
    /// distribution, at exactly the positions the flip count is
    /// drawn from, so the count can be read against the margin that
    /// produced it.
    flip_margin: Mean,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_cond: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Resolve a band argument against the checkpoint's own band list.
fn resolve_band(shape: &ModelShape, arg: &str) -> Result<String, String> {
    let token = if arg.starts_with('<') {
        arg.to_string()
    } else {
        format!("<elo:{arg}>")
    };
    match shape.band(&token) {
        Some(_) => Ok(token),
        None => Err(format!(
            "checkpoint has no band {token}; it carries {:?}",
            shape.band_tokens()
        )),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let ckpt: PathBuf = args
        .next()
        .ok_or(
            "usage: chess_cond <ckpt> <holdout.pgn> [side] [max_positions] [walk_band] \
             [gammas] [records.jsonl]",
        )?
        .into();
    let pgn = args.next().ok_or("missing holdout pgn path")?;
    let side = match args.next().as_deref() {
        None | Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some("both") => ScoredSide::Both,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    // Every position costs one forward per band, and the gamma sweep
    // rides on those same forwards, so the sweep is nearly free.
    let max_positions: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(600);
    let walk_band = args.next().filter(|s| s != "-");
    let gammas: Vec<f32> = args
        .next()
        .unwrap_or_else(|| "1,1.5,2,3,4,6,8".into())
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<Result<_, _>>()?;
    // Absent by default: a walk that only wants the summary should not
    // have to name a file it will never read.
    let records_path: Option<PathBuf> = args.next().filter(|s| s != "-").map(PathBuf::from);

    // Either conditioning convention. This used to be
    // `load_as(.., Prefix)`, which refused a per-position checkpoint —
    // and Phase 5 measures the per-position arm with this program, so
    // that refusal made the plan unable to run its own measurement step.
    //
    // What the two conventions share is the row. Under both, a position
    // is `[BOS, band] + moves`: the per-position arm keeps the band
    // token deliberately, so that the two arms train on the same corpus
    // with the same row lengths. What differs is one extra channel — the
    // band also arrives as an argument to the forward pass, indexing a
    // conditioning table of its own.
    //
    // Not the legality axis, though. A checkpoint trained with the legal
    // moves supplied at every position has to be scored with them
    // supplied, and this walk builds its legal set for the position it
    // is standing on rather than for every position of the row. That
    // refusal is inside the entry point below rather than beside it.
    let shape = open_reader_shape(&ckpt)?;
    if shape.bands.len() < 2 {
        return Err(format!(
            "this checkpoint carries {} band(s); comparing conditions needs at least 2",
            shape.bands.len()
        )
        .into());
    }
    let vocab = MoveVocab::new(&shape.band_tokens())?;
    let band_ids: Vec<u32> = shape
        .bands
        .iter()
        .map(|b| {
            vocab
                .id_of(&b.token)
                .ok_or_else(|| format!("band {} missing", b.token))
        })
        .collect::<Result<_, _>>()?;
    let n_bands = band_ids.len();

    let cfg = shape.config(Device::Cpu, DType::F32);
    let model = Gpt2Model::from_safetensors_file(&cfg, &ckpt)?;

    // Which games the positions come from. Left open by default: the
    // question is how the token changes play, not how well it matches
    // any one population. Setting it makes the fidelity column mean
    // something, since then the humans being matched are the band's.
    let mut filter = GameFilter::accept_all()
        .decided_on_the_board()
        .with_min_base_seconds(180)
        .with_ply_bounds(10, None);
    let walk_token = match &walk_band {
        Some(arg) => {
            let token = resolve_band(&shape, arg)?;
            let band = shape.band(&token).expect("resolved");
            filter = filter.with_rating_band(band.min, band.max);
            Some(token)
        }
        None => None,
    };

    let mut reader = PgnReader::new(BufReader::new(File::open(&pgn)?));
    let t0 = Instant::now();
    let mut positions = 0usize;
    let mut games = 0usize;
    // Games that contributed at least one position. This is the one the
    // error bars resample, and it is not `games` — see the increment.
    let mut scoring_games = 0usize;

    let mut sweep: Vec<GammaStats> = gammas
        .iter()
        .map(|g| GammaStats {
            gamma: *g,
            widest_js: Mean::default(),
            legal_mass: Mean::default(),
            top1: vec![Mean::default(); n_bands],
            top1_differs: 0,
            flip_margin: Mean::default(),
        })
        .collect();

    // Detail at the unguided model: every pair, and the depth profile
    // of the widest one.
    let mut pair_js: Vec<Mean> = vec![Mean::default(); n_bands * n_bands];
    let mut pair_js_agree: Vec<Mean> = vec![Mean::default(); n_bands * n_bands];
    let mut pair_js_differ: Vec<Mean> = vec![Mean::default(); n_bands * n_bands];
    let mut uniform_js: Vec<Mean> = vec![Mean::default(); n_bands];
    // Depth buckets. The last edge is not a round number: it separates
    // the plies whose row reaches the context window from the rest.
    //
    // The edge enters the committed source here. The figures in
    // `style-transfer-2026-08-06.md` were taken with the same edge
    // applied as a local edit that was never committed, so this is the
    // first time the two agree.
    //
    // It is `ctx - 2` — 126 at the shipped ctx of 128 — and that is
    // deliberately one short of where truncation begins. A row is
    // `[BOS, band] + moves`, so ply 126 makes exactly 128 tokens and
    // still fits whole; truncation starts at ply 127. The bucket
    // therefore admits one unwindowed ply, which is why 2026-04 read
    // 0.0004 there rather than zero. Keeping the off-by-one rather
    // than fixing it, because the published deep figures were computed
    // against this boundary and moving it would make them
    // incomparable.
    //
    // What the bucket *means* has changed, though. It used to hold the
    // positions where the condition had been lost: the window was a
    // plain tail slice, it dropped `[BOS, band]` first, every band saw
    // the same row, and the divergence there was zero by construction
    // rather than by measurement. `play_row` keeps the prefix, so
    // those positions now measure distance like every other bucket.
    const DEPTH_EDGES: [usize; 5] = [0, 10, 20, 30, 40];
    let mut ply_edges: Vec<usize> = DEPTH_EDGES.to_vec();
    // Derived from the configured ctx rather than written as 126, so a
    // run with CHESS_CTX set does not silently bucket against a
    // boundary from another shape. Dropped when it would not increase
    // (a ctx small enough to overlap the fixed edges), since the
    // bucket search assumes they ascend.
    let overflow_edge = shape.ctx.saturating_sub(COND_PREFIX_LEN);
    if overflow_edge > DEPTH_EDGES[4] {
        ply_edges.push(overflow_edge);
    }
    ply_edges.push(usize::MAX);
    let mut by_ply: Vec<Mean> = vec![Mean::default(); ply_edges.len() - 1];

    // Where the training loss actually goes. The trainer optimises
    // cross-entropy over the whole vocabulary, so every nat spent
    // keeping mass off illegal moves is a nat not spent on preferring
    // one legal move over another — and decoding throws that work away,
    // because it walks the ranking against the legal set regardless.
    //
    // Splitting it says whether masking illegal moves during training
    // would help. `ce_full` is what the trainer sees. `ce_legal` is the
    // same model renormalised over the legal moves, i.e. what it would
    // score if legality were free. `ce_uniform` is a uniform draw over
    // the same legal moves. If `ce_legal` still loses to `ce_uniform`,
    // the burden is not legality — it is that the preferences are
    // wrong, and masking would move the loss without fixing anything.
    let mut ce_full: Vec<Mean> = vec![Mean::default(); n_bands];
    let mut ce_legal: Vec<Mean> = vec![Mean::default(); n_bands];
    let mut ce_uniform = Mean::default();
    let mut legal_count = Mean::default();

    // Per-position records, kept only when a file was asked for. Held in
    // memory and written once at the end rather than streamed, so that
    // the header's counts are the file's counts and a run that died half
    // way leaves nothing that could be mistaken for a shorter walk.
    let mut records: Vec<PositionRecord> = Vec::new();
    let recording = records_path.is_some();

    while positions < max_positions {
        let Some(game) = reader.next_game()? else {
            break;
        };
        if !filter.accepts_tags(&game) {
            continue;
        }
        let tokens = san_tokens(&game.movetext);
        if tokens.len() < 10 {
            continue;
        }
        // Taken before the increment so it counts accepted games from
        // zero. This is the bootstrap's cluster: positions inside one
        // game share an opening and a pair of players, so an error bar
        // that resampled positions would report a precision the sample
        // does not have.
        let game_index = games;
        games += 1;
        // A game can pass the filter and still contribute nothing — the
        // replay breaks, or the cap is reached mid-game. Those games are
        // in `games` and are not clusters, so both counts are kept and
        // both are printed.
        let mut contributed = false;

        let mut board = Board::default();
        let mut moves_so_far: Vec<u32> = Vec::new();
        for (ply, token) in tokens.iter().enumerate() {
            let Ok(mv) = resolve_san(&board, token) else {
                break;
            };
            let played = uci_standard(&board, mv);

            let scored = match side {
                ScoredSide::Both => true,
                ScoredSide::White => ply.is_multiple_of(2),
                ScoredSide::Black => !ply.is_multiple_of(2),
            };
            if scored && positions < max_positions {
                let legal = legal_moves(&board, &vocab);
                if legal.len() >= 2 {
                    // One forward for all bands: the rows differ only in
                    // the condition token, so they batch cleanly.
                    //
                    // `BandBatch` is what decides whether the band also
                    // reaches the model as a forward argument, from the
                    // checkpoint's own recorded encoding. Shared with
                    // `chess_play` rather than written out here, because
                    // this is the decision that produces ordinary-looking
                    // moves from a model that was never told which band
                    // it is, and one implementation under test beats two
                    // that agree today.
                    //
                    // Guidance is untouched by the choice. Its reference
                    // is the mean over the band rows' logits — the
                    // population the bands were split out of — and that
                    // is the same object however the band reached the
                    // model. So `mean_logits` and `guide_logits` below
                    // read the same way on both arms, and the sweep
                    // still rides on these forwards rather than adding
                    // any.
                    let window = play_row(Some(band_ids[0]), &moves_so_far, shape.ctx)?;
                    let batch = BandBatch::over_bands(&window, &shape, &vocab)?;
                    let band_logits = batch.logits(&model, &cfg.device)?;
                    let reference = mean_logits(&band_logits);

                    let played_at = legal.iter().position(|(uci, _)| *uci == played);
                    let mut gamma_records: Vec<GammaRecord> =
                        Vec::with_capacity(if recording { sweep.len() } else { 0 });

                    for stats in sweep.iter_mut() {
                        let mut dists = Vec::with_capacity(n_bands);
                        // The same guided rows as `dists`, before the
                        // softmax, restricted to the legal moves. Kept
                        // because the cost of the played move is taken
                        // in log space: `dists` is `f32`, and an entry
                        // that underflowed there would read as an
                        // infinite cost. `ce_over_legal` says the rest.
                        let mut legal_logits: Vec<Vec<f32>> = Vec::with_capacity(n_bands);
                        let mut mass_over_bands = 0.0f64;
                        for logits in &band_logits {
                            let guided = guide_logits(logits, &reference, stats.gamma);
                            let probs = softmax(&guided);
                            let on_legal: Vec<f32> =
                                legal.iter().map(|(_, id)| probs[*id as usize]).collect();
                            let mass = on_legal.iter().sum::<f32>() as f64;
                            stats.legal_mass.push(mass);
                            mass_over_bands += mass;
                            legal_logits
                                .push(legal.iter().map(|(_, id)| guided[*id as usize]).collect());
                            dists.push(normalise(&on_legal));
                        }
                        let tops: Vec<usize> = dists.iter().map(|d| argmax(d)).collect();
                        // Hoisted out of the accumulators below because
                        // the per-position record needs the same three
                        // quantities the summary means over, and reading
                        // them back off a `Mean` is not possible.
                        let top1: Option<Vec<bool>> =
                            played_at.map(|want| tops.iter().map(|t| *t == want).collect());
                        // Absent under the same condition as `top1`,
                        // and from the same lookup: with the played
                        // move outside the vocabulary there is nothing
                        // to match a top move against and nothing to
                        // take the log of. `ce_over_legal` returns
                        // `None` only for an index past the end of its
                        // row, and `want` came from searching the list
                        // these rows were built from, so the `collect`
                        // below does not drop a band on its own.
                        //
                        // Computed whether or not a record file was
                        // asked for, because the summary's
                        // "renormalised over legal" line is this same
                        // quantity at gamma 1 and now reads it from
                        // here.
                        let ce: Option<Vec<f64>> = played_at.and_then(|want| {
                            legal_logits
                                .iter()
                                .map(|row| ce_over_legal(row, want))
                                .collect()
                        });
                        let flipped = tops.iter().any(|t| *t != tops[0]);
                        let widest = js_bits(&dists[0], &dists[n_bands - 1])?;

                        if let Some(matched) = &top1 {
                            for (b, hit) in matched.iter().enumerate() {
                                stats.top1[b].push(if *hit { 1.0 } else { 0.0 });
                            }
                        }
                        if flipped {
                            stats.top1_differs += 1;
                            // The arithmetic the record carries, over a
                            // different selection: every band, and only
                            // where a flip happened. Sharing the function
                            // rather than repeating the sort is what keeps
                            // the summary and the records from drifting
                            // into two definitions of one word.
                            //
                            // It replaced a local sort that read
                            // `sorted[0] - sorted.get(1).unwrap_or(0.0)`.
                            // The two agree on every distribution this
                            // loop can be handed and differ on one it
                            // cannot: a single entry, where the old form
                            // returned the entry and this returns 1.0.
                            // The walk enters only under `legal.len() >= 2`
                            // and `dists` entries carry that length, so
                            // the differing case is unreachable from here
                            // — which is a property of the guard above,
                            // not of these two expressions.
                            for d in &dists {
                                stats.flip_margin.push(top2_margin(d));
                            }
                        }
                        stats.widest_js.push(widest);
                        if recording {
                            gamma_records.push(GammaRecord {
                                flipped,
                                widest_js: widest,
                                legal_mass: mass_over_bands / n_bands as f64,
                                top1,
                                ce: ce.clone(),
                                // From `dists[0]` — the same array
                                // `tops[0]` was taken from, and already
                                // renormalised over the legal moves by
                                // `normalise` above. Read off it here
                                // rather than recomputed, so the margin
                                // and the flip it belongs to cannot come
                                // to describe different distributions.
                                top2_margin: Some(top2_margin(&dists[0])),
                            });
                        }

                        // The unguided pass also fills the detail.
                        if stats.gamma == 1.0 {
                            // Loss decomposition, on the move actually
                            // played: what the trainer sees, what it
                            // would see if legality were free, and what
                            // a uniform draw over the legal moves costs.
                            if let Some(want) = played_at {
                                ce_uniform.push((legal.len() as f64).ln());
                                legal_count.push(legal.len() as f64);
                                for (b, logits) in band_logits.iter().enumerate() {
                                    let probs = softmax(logits);
                                    let p_true = probs[legal[want].1 as usize] as f64;
                                    // The legal-mass guard that used to
                                    // stand beside this one is implied
                                    // by it: `p_true` is one of the
                                    // non-negative terms of that sum, so
                                    // a positive `p_true` is a positive
                                    // mass. It was dropped along with
                                    // the sum it guarded.
                                    if p_true > 0.0 {
                                        ce_full[b].push(-p_true.ln());
                                        // The renormalised figure is
                                        // the per-position cost the
                                        // records now carry, at gamma
                                        // 1 where guidance is the
                                        // identity. Read from there
                                        // rather than recomputed as
                                        // `-(p_true / mass).ln()`, so
                                        // that the summary line and the
                                        // record cannot come to mean
                                        // two different things by one
                                        // word — the same reason
                                        // `top2_margin` is a shared
                                        // function.
                                        //
                                        // The same quantity, not the
                                        // same bits. Two reasons, both
                                        // small: it is taken in log
                                        // space rather than as a ratio
                                        // of `f32` sums, and it reads
                                        // the guided row, where
                                        // `r + 1*(c-r)` recovers `c` to
                                        // within a rounding step rather
                                        // than exactly.
                                        if let Some(cost) = ce.as_ref().and_then(|v| v.get(b)) {
                                            ce_legal[b].push(*cost);
                                        }
                                    }
                                }
                            }
                            let uniform = vec![1.0f32 / legal.len() as f32; legal.len()];
                            for (b, d) in dists.iter().enumerate() {
                                uniform_js[b].push(js_bits(d, &uniform)?);
                            }
                            for i in 0..n_bands {
                                for j in (i + 1)..n_bands {
                                    let d = js_bits(&dists[i], &dists[j])?;
                                    pair_js[i * n_bands + j].push(d);
                                    if tops[i] == tops[j] {
                                        pair_js_agree[i * n_bands + j].push(d);
                                    } else {
                                        pair_js_differ[i * n_bands + j].push(d);
                                    }
                                    if i == 0 && j == n_bands - 1 {
                                        let bucket = ply_edges
                                            .windows(2)
                                            .position(|w| ply >= w[0] && ply < w[1])
                                            .unwrap_or(0);
                                        by_ply[bucket].push(d);
                                    }
                                }
                            }
                        }
                    }
                    if recording {
                        records.push(PositionRecord {
                            game: game_index,
                            ply,
                            // The same set every band's row above was
                            // restricted to, so a reader of `ce` can
                            // read it against `ln(n_legal)` — what a
                            // uniform draw over these moves would have
                            // cost.
                            n_legal: Some(legal.len()),
                            at: gamma_records,
                        });
                    }
                    positions += 1;
                    contributed = true;
                }
            }

            board.play_unchecked(mv);
            let Some(id) = vocab.id_of(&played) else {
                break;
            };
            moves_so_far.push(id);
        }
        if contributed {
            scoring_games += 1;
        }
    }

    if positions == 0 {
        return Err("no positions measured".into());
    }
    let n = positions as f64;

    println!("ckpt       {}", ckpt.display());
    println!("encoding   {} conditioning", shape.encoding);
    println!("holdout    {pgn}");
    // Two game counts, named apart. `scoring_games` is the clustering
    // figure — the one every error bar resamples and the one
    // `chess_stats` reports — while `games` counts everything the filter
    // accepted, including games that broke on replay or arrived after
    // the cap. Printing one number under the word "games" left the two
    // tools able to disagree with no indication which was which.
    println!(
        "walk       {positions} positions from {scoring_games} scoring game(s) \
         ({games} accepted by the filter), side {side:?}{}",
        walk_token
            .as_ref()
            .map(|t| format!(", games of {t}"))
            .unwrap_or_default()
    );
    println!("elapsed    {:.1?}", t0.elapsed());

    println!();
    println!(
        "guidance sweep (widest pair {} vs {})",
        shape.bands[0].token,
        shape.bands[n_bands - 1].token
    );
    println!(
        "{:>6} {:>10} {:>11} {:>11} {:>12} top1 match by band",
        "gamma", "JS bits", "legal mass", "top1 flips", "flip margin"
    );
    for stats in &sweep {
        let matches: Vec<String> = stats
            .top1
            .iter()
            .map(|m| match m.value() {
                Some(v) => format!("{v:.4}"),
                None => "  -   ".to_string(),
            })
            .collect();
        println!(
            "{:>6.2} {:>10.4} {:>11.4} {:>10.2}% {:>12.6} {}",
            stats.gamma,
            stats.widest_js.value().unwrap_or(f64::NAN),
            stats.legal_mass.value().unwrap_or(f64::NAN),
            100.0 * stats.top1_differs as f64 / n,
            stats.flip_margin.value().unwrap_or(f64::NAN),
            matches.join("  ")
        );
    }
    println!("bands in order: {:?}", shape.band_tokens());

    println!();
    println!("unguided detail — JS divergence in bits (0 = identical, 1 = disjoint)");
    for i in 0..n_bands {
        for j in (i + 1)..n_bands {
            println!(
                "  {} vs {}   mean {:.4}",
                shape.bands[i].token,
                shape.bands[j].token,
                pair_js[i * n_bands + j].value().unwrap_or(f64::NAN)
            );
            if let Some(v) = pair_js_agree[i * n_bands + j].value() {
                println!(
                    "    where top-1 agrees   {v:.4}  (n={})",
                    pair_js_agree[i * n_bands + j].n
                );
            }
            if let Some(v) = pair_js_differ[i * n_bands + j].value() {
                println!(
                    "    where top-1 differs  {v:.4}  (n={})",
                    pair_js_differ[i * n_bands + j].n
                );
            }
        }
    }
    println!();
    println!("  where the loss goes (nats, on the move actually played):");
    println!(
        "    legal moves per position   {:.1}",
        legal_count.value().unwrap_or(f64::NAN)
    );
    println!(
        "    uniform over legal         {:.4}  <- the bar any model has to beat",
        ce_uniform.value().unwrap_or(f64::NAN)
    );
    for b in 0..n_bands {
        println!(
            "    {}  full vocab {:.4}   renormalised over legal {:.4}",
            shape.bands[b].token,
            ce_full[b].value().unwrap_or(f64::NAN),
            ce_legal[b].value().unwrap_or(f64::NAN)
        );
    }
    println!();
    println!("  each band against a uniform draw over the same legal moves:");
    for (b, m) in uniform_js.iter().enumerate() {
        println!(
            "    {} vs uniform   {:.4}",
            shape.bands[b].token,
            m.value().unwrap_or(f64::NAN)
        );
    }
    println!();
    println!("  widest pair by depth:");
    for (b, m) in by_ply.iter().enumerate() {
        let hi = ply_edges[b + 1];
        let label = if hi == usize::MAX {
            format!("ply {}+", ply_edges[b])
        } else {
            format!("ply {}-{}", ply_edges[b], hi - 1)
        };
        match m.value() {
            Some(v) => println!("    {label:<12} {v:.4}  (n={})", m.n),
            None => println!("    {label:<12} (no positions)"),
        }
    }

    // Written after the summary rather than before it. Both orders
    // return the error, and this one has already put the run's numbers
    // on stdout by the time a failing write is reported — the walk is
    // minutes of work and the summary is the part a person reads.
    if let Some(path) = &records_path {
        let walk = Walk {
            header: WalkHeader {
                version: FORMAT_VERSION,
                ckpt: ckpt.display().to_string(),
                holdout: pgn.clone(),
                side: format!("{side:?}"),
                encoding: shape.encoding,
                ctx: shape.ctx,
                bands: shape.band_tokens(),
                gammas: gammas.clone(),
                positions: records.len(),
                games,
            },
            records,
        };
        walk.write_jsonl(path)?;
        println!();
        println!(
            "records    {} position(s) from {scoring_games} scoring game(s) -> {}",
            walk.records.len(),
            path.display()
        );
    }
    Ok(())
}

/// Index of the largest entry.
fn argmax(d: &[f32]) -> usize {
    d.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Legal moves in a position, with their vocabulary ids.
fn legal_moves(board: &Board, vocab: &MoveVocab) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            let uci = uci_standard(board, mv);
            if let Some(id) = vocab.id_of(&uci) {
                out.push((uci, id));
            }
        }
        false
    });
    out
}
