//! Train one model on a rating-banded chess corpus.
//!
//! Reads PGN, filters to a band, encodes, and runs a full fine-tune
//! from scratch, writing a checkpoint the player example reloads.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example chess_bake -- <path.pgn> <bands> <max_rows> <steps> <side> [ckpt_dir]
//! ```
//!
//! `bands` is one or more rating ranges (`1600-1799`, or
//! `1100-1299,1900-2099` to train one model that can be asked for
//! either) or ECO opening families (`eco:B,eco:C`). Each band gets a
//! condition token prefixed to its games, and
//! a game outside every band is dropped.
//!
//! # Where the condition reaches the model
//!
//! By default the band is a token at the front of the row and the model
//! reads it like any other token, so its influence has to travel the
//! length of the sequence through attention. That influence was
//! measured to decay with depth: band divergence falls to about a tenth
//! between the opening and ply 40-125.
//!
//! `CHESS_COND_EVERY=1` trains the other convention. The band is passed
//! to the forward pass as an argument, one per row of the batch, and
//! the vector it selects out of a conditioning table of its own is
//! added at *every* position — so there is no distance for it to decay
//! over. The row is unchanged: it still begins `[BOS, band]`, so the
//! two arms train on the same corpus, the same row lengths and the same
//! token counts, and "the rows differed" is not among the things an arm
//! difference could be put down to.
//!
//! Which convention a run used is written into its shape file, and
//! every reader refuses a checkpoint of the other one. The two differ
//! by a single small tensor otherwise, so a reader that guessed would
//! get a full set of plausible numbers rather than an error.
//!
//! # Handing the model its legality
//!
//! `CHESS_LEGAL_MASK=1` restricts the loss to the moves that were legal
//! in the position, so the model is scored on choosing among them
//! rather than on knowing which ones exist — measured here, the second
//! question was taking 1.59 of 4.52 nats, and decoding discards all of
//! that work because it walks the ranking against the legal set
//! regardless.
//!
//! `CHESS_LEGAL_INPUT=1` goes further and hands the same sets to the
//! model as *input*: at each position the mean of a table of its own,
//! `legal_wte`, over the ids available there is added to the residual
//! stream. It requires the mask flag, because the sets arrive through
//! the dataset that flag selects. The two arms this serves are the mask
//! alone and the mask with the input.
//!
//! It does not compose with `CHESS_COND_EVERY`: one delivers the band
//! at every position and the other the legal moves, and no forward pass
//! in this build reads both. Asked for together, the run stops before
//! reading the corpus.
//!
//! One run is meant to finish in tens of seconds on a laptop CPU.
//! `steps * batch` is what decides that, and the corpus needs at least
//! that many rows, so raising either raises the wall clock directly.
//! Measure before scaling: the earlier Othello work spent an hour of
//! someone else's machine finding that out.
//!
//! # Resuming
//!
//! `CHESS_INIT_FROM=<path.safetensors>` starts the run from an existing
//! checkpoint rather than from a random initialisation, which is what
//! makes an interrupted run recoverable and a curriculum expressible as
//! two runs.
//!
//! The checkpoint has to describe the same model, and two separate
//! checks say so. The weights themselves are matched name by name and
//! shape by shape, which catches a changed `CHESS_DIM`, `CHESS_LAYERS`
//! or `CHESS_CTX`. That is not enough on its own, because two of the
//! dials move no tensor whatsoever: a changed band list leaves the
//! condition tokens on ids `2..2+n` whatever they are named, with a
//! vocabulary that rounds to 2048 for any band count this program can
//! be given, and a changed `CHESS_HEADS` only reshapes what `c_attn`
//! already produces — that projection is `dim -> 3*dim` at four heads
//! and at eight alike. `CHESS_COND_EVERY` does move a tensor, the
//! `cond_wte` table, but only in one direction does that help: a
//! prefix checkpoint restored into a per-position run is missing
//! `cond_wte.weight` and `restore_into` refuses it, while a
//! per-position checkpoint restored into a prefix run carries one
//! tensor the model does not want, which `restore_into` accepts and
//! reports as unused. In that second direction the shape file is the
//! only thing standing between the operator and a resume from weights
//! trained under another rule.
//!
//! `CHESS_LEGAL_INPUT` is the same shape of problem on its own tensor,
//! `legal_wte`, and it is the one an operator is likelier to meet. The
//! two arms this dial defines — the mask alone and the mask with the
//! input — are meant to be run over the same corpus at the same dials,
//! so reaching for one as the other's starting point is a natural
//! thing to do, and it would leave a checkpoint labelled as the arm it
//! did not start from. A legality checkpoint restored into an ordinary
//! run is accepted, its table listed in `unused_from_file`, and every
//! number afterwards well-formed.
//!
//! So the shape is compared field by field first — every field of it,
//! heads, band list, conditioning and legality included — and a
//! disagreement stops the run before training. Resuming a `1600-1799`
//! model as `1100-1299` is therefore an error, not a run whose
//! embedding row 2 quietly means something else than it did yesterday.
//!
//! Every checkpoint carries its own shape file, written as it lands,
//! so the pairing survives an interrupted run and cannot be rewritten
//! by a later run that happens to share the filename.
//!
//! # Validation
//!
//! Pass a second PGN and the run holds nothing back from it — it is
//! already held back, being a different month. Rows are built from it
//! under the same bands and the same filter, and every periodic
//! checkpoint is scored against them after training, giving a loss
//! curve on data the run never touched.
//!
//! Without that curve, training loss is the only reading available,
//! and training loss cannot say whether a run stopped because it
//! converged or because it ran out of rows. Earlier runs here stopped
//! for the second reason and were reported as though the first had
//! happened.
//!
//! The validation month must be a *different* month rather than a
//! longer prefix of the training one: Lichess archives are a single
//! zstd frame, so reading further into the same file returns games the
//! training slice may already have consumed.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use algocline_nn::arch::{CondIndex, Gpt2Model};
use algocline_nn::chess::corpus::{
    build_rows, ConditionBand, ConditionMatcher, ConditionSpec, CorpusOptions, LegalMaskedDataset,
    ScoredSide, TeacherRow,
};
use algocline_nn::chess::filter::GameFilter;
use algocline_nn::chess::pgn::PgnReader;
use algocline_nn::chess::train::{cond_table, row_conditions};
use algocline_nn::chess::vocab::MoveVocab;
use algocline_nn::chess::{CondEncoding, ModelShape, ShapeError, ShapeKind};
use algocline_nn::train::{
    allowed_logit_mask, legal_input_sets, restore_into, restore_into_partial, run_conditioned_ft,
    run_full_ft, run_legal_ft, CkptControl, CkptHook, CkptInfo, CrossEntropyLoss, Dataset,
    DatasetOpts, FullFtConfig, Loss, ScheduleKind, TeacherCardDataset, TrainingLease,
};

/// Read a `usize` from the environment, falling back to `default`.
///
/// The shape and the training dials are environment-driven so a GPU
/// run can be scaled without editing the source the pod cloned.
fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read an `f64` from the environment, falling back to `default`.
fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Read a `0` / `1` switch from the environment.
///
/// Every arm is spelled out, which is the point of having this rather
/// than a comparison against `"1"`:
///
/// - an unreadable value is an **error**. `CHESS_LEGAL_MASK=true` under
///   the old comparison meant "off", and a run that was asked for one
///   objective and trained under the other reports nothing about it;
/// - `NotUnicode` — a variable that *is* set, to something this program
///   cannot read — is an error too, rather than being folded in with
///   "not set at all";
/// - empty means "not asked for", because that is how a shell blanks a
///   variable it has already exported. Said out loud, since an operator
///   who meant to pass a value should see that none arrived.
fn env_switch(key: &str) -> Result<bool, String> {
    match env::var(key) {
        Ok(v) => match v.as_str() {
            "1" => Ok(true),
            "0" => Ok(false),
            "" => {
                eprintln!("[bake] {key} is set but empty; treating it as 0");
                Ok(false)
            }
            other => Err(format!("unknown {key} {other:?}; pass 0 or 1")),
        },
        Err(env::VarError::NotPresent) => Ok(false),
        Err(e @ env::VarError::NotUnicode(_)) => Err(format!(
            "{key} is set to something this program cannot read ({e}); pass 0 or 1"
        )),
    }
}

/// Deterministic xorshift, used only to order rows.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Fisher-Yates, so consecutive batches are not consecutive games.
///
/// Rows arrive in the order the archive lists them, which is the order
/// the games were played — so an unshuffled batch is a few dozen games
/// that started within the same second. Nothing forces those to be
/// alike, but nothing prevents it either, and the fix is four lines.
fn shuffle<T>(rows: &mut [T], rng: &mut Rng) {
    for i in (1..rows.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        rows.swap(i, j);
    }
}

/// Mean cross-entropy over the legal moves, on held-out rows.
///
/// **Always** restricted to the legal moves, whichever objective the
/// run trained under. Scoring a legal-masked model over the whole
/// vocabulary would charge it for mass on moves it was never asked to
/// suppress, and scoring an unmasked one the same way would credit it
/// for work decoding does for free. The legal-restricted number is
/// what both models are ultimately for, so it is the one that makes
/// the two comparable.
///
/// `conds` carries the band of every row when the run conditions at
/// every position, and is `None` otherwise. It is not optional in the
/// sense of being a nicety: scoring a per-position model through the
/// plain forward would run it in a state it never trained in — with the
/// condition vector absent from every position — and report the result
/// as a validation loss.
///
/// `legal_input` says the same thing about the other channel: a model
/// trained with the legal moves supplied at every position has to be
/// scored with them supplied too. It is a separate argument rather than
/// read off the batch because every batch here carries the sets — that
/// is what makes the legal-restricted number above comparable across
/// runs — and only the run knows whether the model was also handed
/// them.
#[allow(clippy::too_many_arguments)]
fn eval_loss(
    model: &Gpt2Model,
    rows: &[TeacherRow],
    conds: Option<&[CondIndex]>,
    conds_per_row: usize,
    prefix_len: usize,
    legal_input: bool,
    vocab: &MoveVocab,
    ctx: usize,
    batch: usize,
    device: &Device,
) -> Result<f32, Box<dyn std::error::Error>> {
    let loss = CrossEntropyLoss::new();
    let ds = LegalMaskedDataset::new(
        TeacherRow::into_pairs(rows.to_vec()),
        vocab.clone(),
        prefix_len,
        DatasetOpts {
            batch_size: batch,
            ctx_len: ctx,
            shuffle: false,
            pad_id: 0,
            text_field: "text".into(),
        },
    );
    let mut ds = match conds {
        Some(conds) => ds.with_condition_groups(conds.to_vec(), conds_per_row)?,
        None => ds,
    };
    let mut total = 0f64;
    let mut counted = 0usize;
    while let Some(b) = ds.next_batch()? {
        let rows_in = b.input_ids.len();
        let width = b.input_ids[0].len();
        let flat: Vec<u32> = b.input_ids.concat();
        let flat_mask: Vec<f32> = b
            .loss_mask
            .as_ref()
            .expect("teacher rows carry a loss mask")
            .concat();
        let full = Tensor::from_vec(flat, (rows_in, width), device)?;
        let full_mask = Tensor::from_vec(flat_mask, (rows_in, width), device)?;
        // Same shift the training loop applies.
        let inputs = full.narrow(1, 0, width - 1)?.contiguous()?;
        let targets = full.narrow(1, 1, width - 1)?.contiguous()?;
        let m = full_mask.narrow(1, 1, width - 1)?.contiguous()?;
        // The same entry point the run trained through, so the model is
        // scored in the state it was trained in. `legal_input` decides
        // it rather than the batch, because the batch carries its sets
        // under both objectives — they are the loss mask's business on
        // one and the model's input as well on the other.
        let logits = match (legal_input, b.conds.as_deref()) {
            (true, _) => {
                let sets = legal_input_sets(&b, width, device)?
                    .ok_or("a legality-input run needs a dataset that carries the legal sets")?;
                model.forward_legal(&inputs, &sets)?
            }
            (false, Some(conds)) => {
                model.forward_conditioned_groups(&inputs, conds, b.conds_per_row)?
            }
            (false, None) => model.forward(&inputs)?,
        };
        let logits = match allowed_logit_mask(&b, width, logits.dim(2)?, device)? {
            Some(am) => logits.broadcast_add(&am)?,
            None => logits,
        };
        let l = loss
            .compute(&logits, &targets, Some(&m))?
            .to_scalar::<f32>()?;
        total += l as f64 * rows_in as f64;
        counted += rows_in;
    }
    if counted == 0 {
        return Err("validation set is empty".into());
    }
    Ok((total / counted as f64) as f32)
}

/// Parse `1100-1299,1900-2099` (rating ranges) or `eco:B,eco:C`
/// (opening families) into condition bands.
///
/// Several bands in one corpus is the interesting case: the model sees
/// which band it is playing as, so one checkpoint can be asked for
/// either. Baking a model per band answers a different and weaker
/// question, since two separately trained models differ for every
/// reason at once.
/// Parse a `;`-separated list of band groups, each a [`parse_bands`]
/// spec — one group per condition slot.
///
/// Two or more groups make a multi-slot model: the corpus keeps the
/// condition tokens out of its rows, every forward receives one table
/// row per group, and the shape records the partition
/// ([`ModelShape::cond_groups`]). Each group is single-axis on its own
/// (the mixed-axis refusal below is per group); the axes may differ
/// **across** groups, which is the point of having them.
fn parse_band_groups(spec: &str) -> Result<Vec<Vec<ConditionBand>>, String> {
    let groups = spec
        .split(';')
        .map(|group| {
            let group = group.trim();
            if group.is_empty() {
                return Err(format!("band spec {spec:?} has an empty slot"));
            }
            parse_bands(group)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = std::collections::HashSet::new();
    for band in groups.iter().flatten() {
        if !seen.insert(band.token.as_str()) {
            return Err(format!(
                "band {} appears in more than one slot; a token can occupy one \
                 conditioning-table row only",
                band.token
            ));
        }
    }
    Ok(groups)
}

fn parse_bands(spec: &str) -> Result<Vec<ConditionBand>, String> {
    let mut out = Vec::new();
    let (mut any_eco, mut any_rating) = (false, false);
    for part in spec.split(',') {
        let part = part.trim();
        if let Some(prefix) = part.strip_prefix("eco:") {
            // What can match an ECO code, not merely what looks tidy:
            // a capital A-E, then up to two digits. `eco:c` and
            // `eco:B2X` are one keystroke from a band that collects
            // zero games, and the zero-row refusal downstream should
            // be the second line of defence rather than the first.
            let mut chars = prefix.chars();
            let well_formed = matches!(chars.next(), Some('A'..='E'))
                && prefix.len() <= 3
                && chars.all(|c| c.is_ascii_digit());
            if !well_formed {
                return Err(format!(
                    "band {part:?} is not an ECO prefix (a capital letter A-E, then up to two \
                     digits)"
                ));
            }
            any_eco = true;
            out.push(ConditionBand::tag_prefix(
                "ECO",
                prefix,
                format!("<eco:{prefix}>"),
            ));
            continue;
        }
        any_rating = true;
        let (lo, hi) = part
            .split_once('-')
            .ok_or_else(|| format!("band {part:?} is not a min-max range"))?;
        let min: i64 = lo
            .trim()
            .parse()
            .map_err(|_| format!("band {part:?} has an unreadable minimum"))?;
        let max: i64 = hi
            .trim()
            .parse()
            .map_err(|_| format!("band {part:?} has an unreadable maximum"))?;
        if min > max {
            return Err(format!("band {part:?} is inverted"));
        }
        out.push(ConditionBand::rating(
            min,
            max,
            format!("<elo:{min}-{max}>"),
        ));
    }
    if out.is_empty() {
        return Err("no bands given".into());
    }
    if any_eco && any_rating {
        // The two derivations disagree about what the complement of a
        // band is. `band_for` tests in order, so an ECO game inside the
        // rating range is claimed by whichever band comes first and the
        // rating band's rows mean "in the range and not that family" —
        // while `narrow`, walking that band later, re-admits the
        // family's games. One corpus, two readings of the same token.
        return Err(
            "bands mix ECO families and rating ranges; the corpus derivation (first match wins) \
             and the walk-side filter would disagree about what each band's games are. Condition \
             on one axis and narrow the population on the other (CHESS_FILTER_ELO)"
                .into(),
        );
    }
    Ok(out)
}

/// Render one band's matcher for a message or a label.
fn describe_matcher(band: &ConditionBand) -> String {
    match &band.matcher {
        ConditionMatcher::IntRange { min, max } => format!("{min}-{max}"),
        ConditionMatcher::TagPrefix { key, prefix } => format!("{key}:{prefix}"),
    }
}

/// Render a band list for an error message.
fn describe_bands(bands: &[ConditionBand]) -> String {
    bands
        .iter()
        .map(|b| format!("{} as {}", describe_matcher(b), b.token))
        .collect::<Vec<_>>()
        .join(", ")
}

/// How a shape's legality axis reads in a message.
fn describe_legal_input(on: bool) -> &'static str {
    match on {
        true => "supplied at every position",
        false => "absent",
    }
}

/// Every field on which a checkpoint's recorded shape differs from the
/// one this run is configured for.
///
/// Field by field rather than one equality test so the message can name
/// what moved. An operator who resumed with the wrong `CHESS_DIM`
/// wants to read "dim is 128 in the checkpoint and 256 here", not that
/// two structs differed.
///
/// "Every field" is a claim about all nine of [`ModelShape`]'s, and it
/// is held by the destructuring below rather than by this sentence: a
/// tenth field stops this function compiling, which is the moment to
/// decide whether resuming across it is safe. An earlier version of
/// this doc made the same claim over seven of eight — `legal_input` was
/// not compared, and that is the axis where the asymmetry described in
/// the module header bites hardest.
fn shape_disagreements(want: &ModelShape, found: &ModelShape) -> Vec<String> {
    // Named rather than reached through `found.` so that adding a field
    // to `ModelShape` is a compile error here.
    let ModelShape {
        layers,
        heads,
        dim,
        ctx,
        vocab,
        bands,
        encoding,
        legal_input,
        cond_groups,
    } = found;
    let mut out = Vec::new();
    for (label, w, f) in [
        ("layers", want.layers, *layers),
        ("heads", want.heads, *heads),
        ("dim", want.dim, *dim),
        ("ctx", want.ctx, *ctx),
        ("vocab", want.vocab, *vocab),
    ] {
        if w != f {
            out.push(format!("{label} is {f} in the checkpoint and {w} here"));
        }
    }
    if want.bands != *bands {
        out.push(format!(
            "bands are [{}] in the checkpoint and [{}] here",
            describe_bands(bands),
            describe_bands(&want.bands)
        ));
    }
    // Nothing in the weights records this, and the two conventions
    // train different models out of the same tensors, so resuming
    // across it would continue one run's model under the other's rule.
    if want.encoding != *encoding {
        out.push(format!(
            "conditioning is {} in the checkpoint and {} here",
            encoding, want.encoding
        ));
    }
    // The same asymmetry as `encoding`, on the other axis, and the same
    // remedy. A legality checkpoint restored into an ordinary run
    // carries `legal_wte.weight`, which the model does not ask for:
    // `restore_into` accepts the restore, lists the tensor as unused,
    // and the run trains a model that never reads the channel its
    // starting weights were shaped by. The reverse direction — an
    // ordinary checkpoint into a legality run — is caught by the
    // restore, because the table would be missing.
    if want.legal_input != *legal_input {
        out.push(format!(
            "the legality input is {} in the checkpoint and {} here",
            describe_legal_input(*legal_input),
            describe_legal_input(want.legal_input)
        ));
    }
    // Same tensors, different partition: the grouping decides how many
    // rows a forward sums per batch row, so resuming across it would
    // continue a run whose every step conditions differently.
    if want.cond_groups != *cond_groups {
        out.push(format!(
            "the condition grouping is {cond_groups:?} in the checkpoint and {:?} here",
            want.cond_groups
        ));
    }
    out
}

/// The checkpoint a `CHESS_INIT_FROM` resume starts from, or `None` for
/// a random initialisation.
///
/// Resolved next to the argument parsing rather than at the restore
/// itself, which sits below the corpus read: a mistyped path or a
/// checkpoint of another shape should cost seconds, and the corpus read
/// is minutes on a real Lichess slice. This is the same reason the
/// validation PGN is opened before training rather than after.
///
/// The shape file is required, not preferred. Its absence cannot be
/// waved through, because the check it carries is the only one there
/// is for the bands: `restore_into` compares names, shapes and dtypes,
/// and a changed band list leaves all three identical. Proceeding
/// without the sidecar would restore a `1600-1799` model into a run
/// configured as `1100-1299` and report a flawless resume, which is the
/// exact silence this check exists to remove.
/// The checkpoint a `CHESS_INIT_BODY_FROM` run starts its **body**
/// from, or `None`.
///
/// The sibling of [`resume_from`] for the one case that one refuses on
/// purpose: starting several differently-conditioned runs from one
/// shared model. `CHESS_INIT_FROM` requires the band lists to agree,
/// because resuming across them would continue one run's model under
/// another's meaning. Here the band lists are *supposed* to differ —
/// what is shared is everything except the conditioning — so the axes
/// checked are the ones the body actually depends on (layers, heads,
/// dim, ctx, vocab) and the band list is not one of them.
///
/// The restore is [`restore_into_partial`], so the conditioning table
/// this run registers and the base does not carry is left at its
/// random initialisation rather than failing the load. A base that
/// *does* carry one is refused here: two tables of different heights
/// are a shape error inside the restore, and one of the same height
/// would be silently adopted as this run's conditioning, which is the
/// thing a shared base must not decide.
///
/// Why this exists: plan 11 measured two independently-trained
/// conditioned models at cosine 0.02 on their body weights — nearly
/// orthogonal, so their mean is neither of them. A shared base is the
/// structural answer (task arithmetic), and it needs an init that
/// crosses the band list.
fn body_init_from(want: &ModelShape) -> Result<Option<PathBuf>, String> {
    let raw = match env::var_os("CHESS_INIT_BODY_FROM") {
        Some(v) => v,
        None => return Ok(None),
    };
    if raw.is_empty() {
        eprintln!(
            "[bake] CHESS_INIT_BODY_FROM is set but empty; starting from a random initialisation"
        );
        return Ok(None);
    }
    let path = PathBuf::from(raw);
    if !path.is_file() {
        return Err(format!(
            "CHESS_INIT_BODY_FROM={} is not a readable file",
            path.display()
        ));
    }
    let found = ModelShape::load_any(&path).map_err(|e| {
        format!(
            "CHESS_INIT_BODY_FROM={}: cannot read the shape written beside it. {e}",
            path.display()
        )
    })?;
    let axes: [(&str, String, String); 5] = [
        ("layers", want.layers.to_string(), found.layers.to_string()),
        ("heads", want.heads.to_string(), found.heads.to_string()),
        ("dim", want.dim.to_string(), found.dim.to_string()),
        ("ctx", want.ctx.to_string(), found.ctx.to_string()),
        ("vocab", want.vocab.to_string(), found.vocab.to_string()),
    ];
    if let Some((field, w, f)) = axes.into_iter().find(|(_, w, f)| w != f) {
        return Err(format!(
            "CHESS_INIT_BODY_FROM={} has {field} {f} where this run has {w}; the body cannot \
             be carried across that",
            path.display()
        ));
    }
    if found.encoding == CondEncoding::EveryPosition {
        return Err(format!(
            "CHESS_INIT_BODY_FROM={} conditions every position, so it carries a conditioning \
             table of its own. A shared base must not decide this run's conditioning: bake the \
             base with prefix conditioning (leave CHESS_COND_EVERY unset) so it carries a body \
             and nothing else",
            path.display()
        ));
    }
    Ok(Some(path))
}

fn resume_from(want: &ModelShape) -> Result<Option<PathBuf>, String> {
    let raw = match env::var_os("CHESS_INIT_FROM") {
        Some(v) => v,
        None => return Ok(None),
    };
    if raw.is_empty() {
        // `CHESS_INIT_FROM=` is how a shell blanks a variable it has
        // already exported, so it means "no resume" rather than "resume
        // from the file named ''" — which would only surface as an
        // unreadable empty path further down. Said out loud, because an
        // operator who meant to pass a path should see that none
        // arrived.
        eprintln!("[bake] CHESS_INIT_FROM is set but empty; starting from a random initialisation");
        return Ok(None);
    }

    let path = PathBuf::from(raw);
    if !path.is_file() {
        return Err(format!(
            "CHESS_INIT_FROM={} is not a readable file",
            path.display()
        ));
    }

    // Every checkpoint carries its own shape file, the periodic ones
    // included, so this is a plain lookup with no filename arithmetic
    // in it — and the same lookup `chess_play` and the other readers
    // do, rather than a rule only this program knows.
    let found = ModelShape::load_any(&path).map_err(|e| {
        format!(
            "CHESS_INIT_FROM={}: cannot read the shape written beside it, so there is no way \
             to tell which bands it was trained on — the weights carry the band count only as \
             a vocabulary size, and that rounds to the same power of two either way. {e}",
            path.display()
        )
    })?;

    let disagreements = shape_disagreements(want, &found);
    if !disagreements.is_empty() {
        return Err(format!(
            "CHESS_INIT_FROM={} was trained at a different shape: {}. Resuming anyway would \
             train from weights that mean something else than this run assumes",
            path.display(),
            disagreements.join("; ")
        ));
    }
    Ok(Some(path))
}

/// Remove one shape file, and say what happened to it.
///
/// Deliberately silent about which run wrote what was removed: it may
/// be an earlier run's file, or this run's own half-written JSON if the
/// failure came part-way through the write. Either way the point is
/// that no file describing the wrong model is left beside the weights.
/// An empty string means there was nothing there, which is the common
/// case and not worth a line of output.
fn sweep_one(path: &Path) -> String {
    match std::fs::remove_file(path) {
        Ok(()) => format!(
            "; the shape file at {} was removed, so it can no longer be read as describing \
             these weights",
            path.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => format!(
            "; and a shape file at {} could not be removed ({e}) — if it predates this run it \
             now describes weights it was not written for, so delete it by hand before reading \
             this checkpoint",
            path.display()
        ),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chess_bake: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(
        "usage: chess_bake <path.pgn> <bands> [max_rows] [steps] [side] [ckpt_dir]\n\
         bands is comma-separated ranges or ECO families, e.g. 1600-1799 or \
         1100-1299,1900-2099 or eco:B,eco:C; a `;` separates condition slots, \
         e.g. 'eco:B,eco:C;1100-1599,1600-2099' for a two-slot model",
    )?;
    let groups = parse_band_groups(&args.next().unwrap_or_else(|| "1600-1799".into()))?;
    let group_sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
    let multi_slot = groups.len() > 1;
    let bands: Vec<ConditionBand> = groups.iter().flatten().cloned().collect();
    let max_rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4000);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let side = match args.next().as_deref() {
        None | Some("both") => ScoredSide::Both,
        Some("white") => ScoredSide::White,
        Some("black") => ScoredSide::Black,
        Some(other) => return Err(format!("unknown side {other:?}").into()),
    };
    let ckpt_dir: PathBuf = args.next().unwrap_or_else(|| "/tmp".into()).into();
    let val_pgn = args.next();

    let tokens: Vec<String> = bands.iter().map(|b| b.token.clone()).collect();
    let vocab = MoveVocab::new(&tokens)?;

    let mut shape = ModelShape::compact(vocab.model_vocab_size(), bands.clone());
    shape.layers = env_usize("CHESS_LAYERS", shape.layers);
    shape.heads = env_usize("CHESS_HEADS", shape.heads);
    shape.dim = env_usize("CHESS_DIM", shape.dim);
    shape.ctx = env_usize("CHESS_CTX", shape.ctx);
    // Which conditioning convention this run trains under, and whether
    // it hands the model the legal moves. Both go into the shape file,
    // because that is where a reader can act on them before loading
    // anything, and both are read here so a bad value costs seconds
    // rather than a corpus read. `env_switch` has the arms.
    shape.encoding = if env_switch("CHESS_COND_EVERY")? {
        CondEncoding::EveryPosition
    } else {
        CondEncoding::Prefix
    };
    // Restricting the loss to the legal moves is opt-in so the two
    // objectives can be compared on the same corpus and the same seed.
    let legal_mask = env_switch("CHESS_LEGAL_MASK")?;
    // And handing the same sets to the model as *input*, which is a
    // second question about the same list: the mask stops charging for
    // moves that were never available, this stops the model having to
    // work out which ones those were.
    shape.legal_input = env_switch("CHESS_LEGAL_INPUT")?;
    // Whether every row's band is passed to the forward pass. Read once
    // here so the rest of the run asks this rather than the shape, and
    // so the two cannot drift apart within one run.
    let per_position = shape.encoding == CondEncoding::EveryPosition;

    if multi_slot {
        // A multi-slot corpus carries no condition token in its rows,
        // so a prefix-encoded run would have nowhere to read the
        // condition from at all — refused here, before the corpus.
        if !per_position {
            return Err(
                "a multi-slot band spec conditions through forward arguments only (its rows \
                 carry no condition token); pass CHESS_COND_EVERY=1"
                    .into(),
            );
        }
        shape.cond_groups = group_sizes.clone();
    }
    // How many non-move tokens open every row: `BOS` plus, under a
    // single slot, the condition token. A multi-slot row carries none.
    let prefix_len = if multi_slot { 1 } else { 2 };

    if shape.legal_input && !legal_mask {
        // The legal sets reach the trainer through the dataset, and the
        // dataset that produces them is the one `CHESS_LEGAL_MASK`
        // selects. Refused rather than turned on quietly, because
        // switching the dataset would also switch the objective — the
        // shared loop masks the logits of every batch that carries
        // sets — and a run comparing objectives cannot have one of them
        // arrive as a side effect.
        return Err(
            "CHESS_LEGAL_INPUT=1 needs CHESS_LEGAL_MASK=1: the legal sets reach the model \
             through the legal-masked dataset, and turning that on by itself would change \
             the objective as well as the input"
                .into(),
        );
    }
    if shape.legal_input && per_position {
        // `Gpt2Custom::validate` refuses the pair when the model is
        // built; this says so first, with the two dials named, before
        // the corpus is read.
        return Err(
            "CHESS_COND_EVERY=1 and CHESS_LEGAL_INPUT=1 together have no forward pass in this \
             build: one delivers the band at every position and the other the legal moves, and \
             nothing reads both. Pick one"
                .into(),
        );
    }
    let batch = env_usize("CHESS_BATCH", 32);
    let lr = env_f64("CHESS_LR", 3e-3);
    // More than one pass over the rows is what lets a run continue past
    // the point the corpus runs out. Whether that helps or overfits is
    // exactly what the validation curve is for.
    let epochs = env_usize("CHESS_EPOCHS", 1).max(1);
    let seed = env_usize("CHESS_SEED", 20260805) as u64;
    let eval_every = env_usize("CHESS_EVAL_EVERY", 0);
    let val_rows_cap = env_usize("CHESS_VAL_ROWS", 3000);
    // A cosine schedule decays the rate to nearly zero at the end, so a
    // validation curve run under it always flattens — whether or not
    // the model converged. `constant` removes that confound: under a
    // fixed rate, a flat tail is the model, not the schedule.
    //
    // Spelled out arm by arm for the reason `env_switch` is: a variable
    // that is set to something this program cannot read is an error
    // rather than a run that silently took the default, and a run that
    // was asked for one schedule and trained under the other reports
    // nothing about it.
    let schedule = match env::var("CHESS_SCHEDULE") {
        Ok(v) => match v.as_str() {
            "constant" | "const" => ScheduleKind::Constant,
            "cosine" | "cosine_with_warmup" => ScheduleKind::CosineWithWarmup,
            "" => {
                eprintln!("[bake] CHESS_SCHEDULE is set but empty; using the cosine schedule");
                ScheduleKind::CosineWithWarmup
            }
            other => {
                return Err(
                    format!("unknown CHESS_SCHEDULE {other:?}; pass constant or cosine").into(),
                )
            }
        },
        Err(env::VarError::NotPresent) => ScheduleKind::CosineWithWarmup,
        Err(e @ env::VarError::NotUnicode(_)) => {
            return Err(format!(
                "CHESS_SCHEDULE is set to something this program cannot read ({e}); \
                 pass constant or cosine"
            )
            .into())
        }
    };

    // Resolved here, before the corpus read, so a bad path or a
    // checkpoint of another shape stops the run in seconds. The restore
    // itself has to wait for the model to exist and stays below.
    let init_from = resume_from(&shape)?;
    let body_init = body_init_from(&shape)?;
    if init_from.is_some() && body_init.is_some() {
        // Two inits would leave which one wins depending on the order
        // of two restores, and the order is an implementation detail.
        return Err(
            "CHESS_INIT_FROM and CHESS_INIT_BODY_FROM are both set; pick one — a resume \
             continues one model, a body init starts several runs from a shared one"
                .into(),
        );
    }
    if let Some(p) = &init_from {
        eprintln!("[bake] resuming from {}", p.display());
    }

    // One `ConditionSpec` value per slot for the whole run. Every
    // row's ordinals index into *these* lists, and `cond_table` below
    // turns those ordinals into conditioning-table rows against the
    // same values — so the two ends cannot be different lists that
    // happen to be the same lengths.
    let specs: Vec<ConditionSpec> = groups
        .iter()
        .map(|group| ConditionSpec {
            key: "WhiteElo".to_string(),
            bands: group.clone(),
        })
        .collect();

    // The band is selected by the condition rather than by the filter:
    // a game outside every band is rejected when its token is resolved,
    // which is one code path instead of keeping a filter and a band
    // list in agreement.
    //
    // `CHESS_FILTER_ELO=min-max` narrows the population on the axis the
    // bands stopped occupying: when the bands are opening families, the
    // condition no longer says anything about who was playing, and a
    // corpus of "everyone who ever opened 1. e4" spans 400 to 3100. One
    // contiguous range (both players inside it), because the filter's
    // predicates conjoin — it cannot say "one of these three disjoint
    // bands", and nothing in the plan this serves needs it to.
    let mut filter = GameFilter::accept_all()
        .decided_on_the_board()
        .with_min_base_seconds(180)
        .with_ply_bounds(10, None);
    if let Ok(range) = env::var("CHESS_FILTER_ELO") {
        let (lo, hi) = range
            .split_once('-')
            .ok_or_else(|| format!("CHESS_FILTER_ELO {range:?} is not a min-max range"))?;
        let min: i64 = lo
            .trim()
            .parse()
            .map_err(|_| format!("CHESS_FILTER_ELO {range:?} has an unreadable minimum"))?;
        let max: i64 = hi
            .trim()
            .parse()
            .map_err(|_| format!("CHESS_FILTER_ELO {range:?} has an unreadable maximum"))?;
        if min > max {
            return Err(format!("CHESS_FILTER_ELO {range:?} is inverted").into());
        }
        eprintln!("[bake] population narrowed to both players inside {min}-{max}");
        filter = filter.with_rating_band(min, max);
    }
    // `CHESS_FILTER_ECO=B,C` narrows the population to those opening
    // families, which is a different statement from conditioning on
    // them: a condition rejects the games it cannot label, so two
    // models conditioned on different attributes end up looking at
    // different games unless the population is stated here. Plan 10
    // shipped exactly that confound; this is what lets a run hold the
    // corpus fixed while the condition varies.
    if let Ok(families) = env::var("CHESS_FILTER_ECO") {
        let prefixes: Vec<String> = families
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if prefixes.is_empty() {
            return Err(format!("CHESS_FILTER_ECO {families:?} names no family").into());
        }
        eprintln!("[bake] population narrowed to ECO families {prefixes:?}");
        filter = filter.with_eco_prefixes(prefixes);
    }
    let opts = CorpusOptions {
        filter,
        max_rows,
        max_len: Some(shape.ctx),
        conditions: specs.clone(),
        scored_side: side,
        ..Default::default()
    };

    eprintln!("[bake] reading {path}");
    let t0 = Instant::now();
    let mut reader = PgnReader::new(BufReader::new(File::open(&path)?));
    let corpus = build_rows(&mut reader, &vocab, &opts)?;
    eprintln!(
        "[bake] corpus: {} rows from {} games in {:.2?} ({} tokens)",
        corpus.stats.rows,
        corpus.stats.games_read,
        t0.elapsed(),
        corpus.stats.tokens
    );
    // Per band, and a refusal on zero. The aggregate above cannot show
    // one band collecting nothing — a typo'd band (`eco:c` for `eco:C`,
    // `9000-9999`) folds its games into `rejected_by_condition` and the
    // run proceeds to write a sidecar listing a token whose condition
    // row no gradient ever reached. Downstream accepts that token and
    // sweeps an untrained embedding, every figure well-formed.
    if let Some(row_bands) = &corpus.bands {
        let mut counts = vec![0usize; bands.len()];
        // One ordinal per slot per row; the offsets rebuild the flat
        // band list's numbering, group by group.
        let offsets: Vec<usize> = group_sizes
            .iter()
            .scan(0usize, |acc, size| {
                let here = *acc;
                *acc += size;
                Some(here)
            })
            .collect();
        for bs in row_bands {
            for (g, ordinal) in bs.iter().enumerate() {
                if let Some(slot) = counts.get_mut(offsets[g] + ordinal) {
                    *slot += 1;
                }
            }
        }
        for (band, n) in bands.iter().zip(&counts) {
            eprintln!("[bake] band {}: {} row(s)", band.token, n);
        }
        if let Some((empty, _)) = bands.iter().zip(&counts).find(|(_, n)| **n == 0) {
            return Err(format!(
                "band {} collected zero rows, so its condition row would ship untrained while \
                 the sidecar still lists it; check the band against the archive's tags",
                empty.token
            )
            .into());
        }
    }

    // Shuffle, then lay down one shuffled copy per epoch. The dataset
    // walks its rows once in order, so repetition has to be materialised
    // here; re-shuffling per epoch keeps the second pass from replaying
    // the first batch for batch.
    let base = corpus.into_teacher_rows()?;
    let mut rng = Rng::new(seed);
    let mut rows: Vec<TeacherRow> = Vec::with_capacity(base.len() * epochs);
    for _ in 0..epochs {
        let mut pass = base.clone();
        shuffle(&mut pass, &mut rng);
        rows.extend(pass);
    }
    eprintln!(
        "[bake] rows {} ({} unique x {epochs} epoch(s), shuffled, seed {seed})",
        rows.len(),
        base.len()
    );

    // The loop consumes `steps * batch` rows and refuses to run short.
    let needed = steps * batch + batch;
    if rows.len() < needed {
        return Err(format!(
            "have {} rows but {steps} steps at batch {batch} need {needed}; \
             read a longer slice, raise CHESS_EPOCHS, or lower steps",
            rows.len()
        )
        .into());
    }

    let vocab_size = vocab.model_vocab_size();
    let ds_opts = DatasetOpts {
        batch_size: batch,
        ctx_len: shape.ctx,
        shuffle: false,
        pad_id: 0,
        text_field: "text".into(),
    };

    // Which conditioning-table row each band occupies, and from that,
    // which row every corpus row trains under. Both are resolved
    // through the band's token rather than through its position in a
    // list or its id in the vocabulary — the three numberings overlap
    // and none of them is the others (`chess::train`).
    //
    // Built here, from `rows` as they now stand, so the shuffle and the
    // epoch replication above are already accounted for: a row and the
    // band it trains under cannot be one apart.
    let tables = if per_position {
        Some(
            specs
                .iter()
                .map(|spec| cond_table(&shape, spec))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let train_conds = match &tables {
        Some(tables) => Some(row_conditions(&rows, tables)?),
        None => None,
    };

    let mut dataset: Box<dyn Dataset + Send> = if legal_mask {
        eprintln!(
            "[bake] loss restricted to legal moves (measured: 1.59 of 4.52 nats \
             otherwise goes on suppressing illegal ones, which decoding gives free)"
        );
        if shape.legal_input {
            eprintln!(
                "[bake] and the same sets are handed to the model at every position, so it is \
                 told what is available rather than having to infer it"
            );
        }
        let ds = LegalMaskedDataset::new(
            TeacherRow::into_pairs(rows),
            MoveVocab::new(&tokens)?,
            // [BOS, band, moves..] under one slot — two tokens before
            // the first move; a multi-slot row is [BOS, moves..].
            prefix_len,
            ds_opts,
        );
        match train_conds {
            Some(conds) => Box::new(ds.with_condition_groups(conds, specs.len())?),
            None => Box::new(ds),
        }
    } else {
        let ds = TeacherCardDataset::from_rows(TeacherRow::into_pairs(rows), ds_opts)?;
        match train_conds {
            Some(conds) => Box::new(ds.with_condition_groups(conds, specs.len())?),
            None => Box::new(ds),
        }
    };

    // CUDA when the build has it, so the same example serves the pod.
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let cfg = shape.config(device, DType::F32);
    eprintln!(
        "[bake] model layers={} heads={} dim={} ctx={} vocab={} side={side:?} \
         lr={lr} schedule={schedule:?}",
        cfg.layers, cfg.heads, cfg.dim, cfg.ctx, cfg.vocab
    );
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb)?;

    // The resume itself. `resume_from` has already established that the
    // file exists and that its shape file agrees with this run down to
    // the band list; what is left is the weights, and the restore is
    // the strict one so that a name or shape the sidecar did not cover
    // is an error rather than a variable quietly left at its random
    // initialisation. A permissive load would report success and train
    // from noise, with a training loss that looks like a fresh run
    // because it would be one.
    if let Some(init_from) = &init_from {
        let report = restore_into(&vm, init_from)?;
        eprintln!("[bake] init: {}", report.summary());
        if !report.unused_from_file.is_empty() {
            eprintln!(
                "[bake] init: {} tensor(s) in the checkpoint are not part of this model \
                 and were ignored: {}",
                report.unused_from_file.len(),
                report.unused_from_file.join(", ")
            );
        }
    }
    // The shared-base init. Partial, so this run's conditioning table —
    // which the base does not carry, and `body_init_from` has refused a
    // base that does — stays at its random initialisation. The skipped
    // names are printed rather than assumed: "the table and nothing
    // else was skipped" is the claim this init rests on.
    if let Some(base) = &body_init {
        let report = restore_into_partial(&vm, base)?;
        eprintln!(
            "[bake] body init from {}: {}",
            base.display(),
            report.summary()
        );
        if !report.absent_from_file.is_empty() {
            eprintln!(
                "[bake] body init: {} variable(s) left at their initialisation: {}",
                report.absent_from_file.len(),
                report.absent_from_file.join(", ")
            );
        }
    }

    // Validation rows, built from a month the training slice cannot
    // contain. Read before training so a bad path fails in seconds
    // rather than after the run.
    let val_rows: Option<Vec<TeacherRow>> = match &val_pgn {
        Some(p) => {
            let val_opts = CorpusOptions {
                max_rows: val_rows_cap,
                ..opts.clone()
            };
            let mut r = PgnReader::new(BufReader::new(File::open(p)?));
            let c = build_rows(&mut r, &vocab, &val_opts)?;
            eprintln!(
                "[bake] validation: {} rows from {} games of {p}",
                c.stats.rows, c.stats.games_read
            );
            Some(c.into_teacher_rows()?)
        }
        None => {
            eprintln!(
                "[bake] validation: none (pass a holdout PGN as the 7th argument). \
                 Training loss alone cannot tell convergence from running out of rows."
            );
            None
        }
    };
    // Held-out rows are conditioned the same way the training rows
    // were, through the same tables.
    let val_conds = match (&tables, &val_rows) {
        (Some(tables), Some(val)) => Some(row_conditions(val, tables)?),
        _ => None,
    };

    let ft_cfg = FullFtConfig {
        lr,
        batch_size: batch,
        grad_accum: 1,
        steps,
        warmup: steps.min(10),
        schedule,
        weight_decay: 0.0,
        ckpt_every: eval_every,
        // Every periodic checkpoint is scored afterwards, so none of
        // them may be rotated away. The hook below writes each
        // checkpoint's shape file, but it cannot do the scoring
        // itself: `CkptHook` is `'static`, and the model is not.
        ckpt_keep: steps.checked_div(eval_every).map_or(1, |n| n + 2),
    };

    let band_label = bands
        .iter()
        .map(describe_matcher)
        .collect::<Vec<_>>()
        .join("_")
        .replace(':', "");
    let prefix = format!("chess-{band_label}-{side:?}").to_lowercase();

    // The shape rides with every checkpoint, written the moment the
    // checkpoint itself lands. The weights alone do not say how many
    // layers produced them, nor how many heads split them, nor which
    // band each condition id stands for, and a reader that guesses
    // wrong either fails on a tensor name or — for the heads and for
    // the bands — does not fail at all. `c_attn` is `dim -> 3*dim`
    // whichever head count reshapes it, so no name-and-shape check
    // downstream can recover what this file records.
    //
    // One sidecar per checkpoint rather than one per run, because a
    // per-run file describes the most recent run that *started*.
    // `prefix` is built from the bands and the side alone, so two runs
    // that differ only in CHESS_HEADS share every filename: writing
    // the shape up front would leave the second run's heads sitting
    // beside the first run's weights the moment the second was
    // interrupted, and every reader (the resume here, chess_play,
    // chess_eval, chess_cond, chess_match) would believe it. Written
    // from one `ModelShape` value inside one process, a checkpoint and
    // its shape file cannot disagree.
    //
    // Three residual gaps, all far narrower than the one this closes.
    // A crash in the moment between `save_step` returning and this
    // hook running leaves the previous sidecar next to a new
    // checkpoint. Rotation unlinks step checkpoints without their
    // shape files, leaving inert orphans nothing reads. And the shape
    // write itself can fail — the arm below, which is the one that has
    // actually been hit.
    //
    // That arm has to sweep, and what it sweeps depends on how `save`
    // failed — the two failures leave opposite states:
    //
    // - the **write** failed. New weights now sit at a name an earlier
    //   run with these bands may already have written a shape file
    //   for, under either convention, and that stale file would
    //   outlive the aborted run: not a window of microseconds but a
    //   permanent mispairing, and one a later resume would accept
    //   whenever the operator's CHESS_HEADS happens to match what the
    //   old file says. Both names are swept, because a run of the
    //   other convention could have left either, and what is wanted is
    //   a checkpoint with no shape at all — which every reader
    //   refuses;
    // - the write **succeeded** and the sweep inside `save` failed.
    //   The file `save` wrote is correct and current; what is left
    //   over is the other convention's. Removing this run's own file
    //   here would leave that stale one alone beside the new weights,
    //   which is not "a checkpoint with no shape" but a checkpoint
    //   with the *wrong* shape. That state is refused rather than
    //   scored — `load_any` compares the recorded encoding against
    //   whether `cond_wte.weight` is in the file, and a stale sidecar
    //   of the other convention contradicts it by construction — but a
    //   contradiction is something an operator then has to work out,
    //   where a swept name leaves a plain "no shape file". So that
    //   branch removes the stale name, and if it cannot, both files
    //   remain and `load_any` refuses the pair as ambiguous.
    //
    // Every removal's own outcome rides on the error message rather
    // than being dropped, since a sweep that silently failed would
    // leave exactly the state it was supposed to prevent.
    let shape_for_ckpt = shape.clone();
    let on_ckpt: CkptHook = Box::new(move |info: &CkptInfo| {
        match shape_for_ckpt.save(&info.ckpt_path) {
            Ok(_) => Ok(CkptControl::Continue),
            Err(e) => {
                let swept = match &e {
                    ShapeError::SweepFailed { stale, .. } => sweep_one(stale),
                    _ => ShapeKind::ALL
                        .iter()
                        .map(|kind| sweep_one(&ModelShape::path_for_kind(&info.ckpt_path, *kind)))
                        .collect::<Vec<_>>()
                        .join(""),
                };
                // The weights are on disk and cost whatever the run has
                // cost so far. The abort skips the retrieval hint at
                // the end of the program, so this is the only place
                // that can say the checkpoint survived and what it
                // still needs.
                Err(format!(
                    "{e}{swept}. The checkpoint itself was written and is intact at {}; it is \
                     unusable until exactly one correct shape file sits beside it, and the \
                     `[bake] model …` line above carries every field that file needs",
                    info.ckpt_path.display()
                ))
            }
        }
    });

    eprintln!(
        "[bake] training {steps} steps at batch {batch}, conditioning by {}…",
        shape.encoding
    );
    let t0 = Instant::now();
    // The two entry points differ in one thing: whether the band
    // reaches the model. `run_conditioned_ft` takes each batch's own
    // per-row indices and refuses a batch that carries none, so a run
    // that lost its conditions somewhere upstream stops rather than
    // training unconditioned under a checkpoint labelled otherwise.
    let loss_fn = CrossEntropyLoss::new();
    let lease = Arc::new(TrainingLease::new());
    let ckpt = if shape.legal_input {
        // The third entry point, and the same discipline: it refuses a
        // batch that carries no legal sets rather than running the
        // plain forward, which on this model would fail at the forward
        // anyway — the table is there and unread.
        run_legal_ft(
            &model,
            &vm,
            dataset.as_mut(),
            &ft_cfg,
            &loss_fn,
            &ckpt_dir,
            &prefix,
            lease,
            Some(on_ckpt),
        )?
    } else if per_position {
        run_conditioned_ft(
            &model,
            &vm,
            dataset.as_mut(),
            &ft_cfg,
            &loss_fn,
            &ckpt_dir,
            &prefix,
            lease,
            Some(on_ckpt),
        )?
    } else {
        run_full_ft(
            &model,
            &vm,
            dataset.as_mut(),
            &ft_cfg,
            &loss_fn,
            &ckpt_dir,
            &prefix,
            lease,
            Some(on_ckpt),
        )?
    };
    let elapsed = t0.elapsed();

    let min_loss = ckpt
        .metrics
        .get("min_train_loss")
        .copied()
        .unwrap_or(f32::NAN);
    // A model that learned nothing sits at the uniform-draw loss, so
    // that is the floor a run has to beat to have done anything.
    let uniform = (vocab_size as f32).ln();
    eprintln!(
        "[bake] done in {:.2?} ({:.2}s/step): final_loss={:.4} min_loss={:.4} \
         uniform_baseline={uniform:.4}",
        elapsed,
        elapsed.as_secs_f64() / steps.max(1) as f64,
        ckpt.train_loss,
        min_loss
    );

    // The final checkpoint's own sidecar, written once the run has
    // actually produced one. After `run_full_ft` and not before: until
    // it returns, `<prefix>.safetensors` is still whatever an earlier
    // run with these bands left there, and putting this run's shape
    // beside those weights would describe them with a configuration
    // that never produced them.
    let ckpt_path = ckpt_dir.join(format!("{prefix}.safetensors"));
    let shape_path = shape.save(&ckpt_path)?;
    eprintln!("[bake] shape written to {}", shape_path.display());

    // The validation curve. Scored after the fact from the periodic
    // checkpoints, which is what says whether the run stopped because
    // it converged or because it ran out of rows.
    if let Some(val) = &val_rows {
        eprintln!(
            "[bake] scoring {} validation rows against each checkpoint…",
            val.len()
        );
        let val_conds = val_conds.as_deref();
        let final_val = eval_loss(
            &model,
            val,
            val_conds,
            specs.len(),
            prefix_len,
            shape.legal_input,
            &vocab,
            shape.ctx,
            batch,
            &cfg.device,
        )?;
        let mut curve: Vec<(usize, f32)> = Vec::new();
        if eval_every > 0 {
            let mut step = eval_every;
            while step <= steps {
                let p = ckpt_dir.join(format!("{prefix}-step{step}.safetensors"));
                if p.exists() {
                    let m = Gpt2Model::from_safetensors_file(&cfg, &p)?;
                    curve.push((
                        step,
                        eval_loss(
                            &m,
                            val,
                            val_conds,
                            specs.len(),
                            prefix_len,
                            shape.legal_input,
                            &vocab,
                            shape.ctx,
                            batch,
                            &cfg.device,
                        )?,
                    ));
                }
                step += eval_every;
            }
        }
        println!("step,val_loss");
        for (s, v) in &curve {
            println!("{s},{v:.4}");
        }
        println!("{steps},{final_val:.4}");
        let best = curve
            .iter()
            .copied()
            .chain(std::iter::once((steps, final_val)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((s, v)) = best {
            eprintln!(
                "[bake] val_loss final={final_val:.4}  best={v:.4} at step {s}{}",
                if s == steps {
                    " (still improving at the end — undertrained)"
                } else {
                    " (past its best — later steps overfit)"
                }
            );
        }
    }
    print_retrieval_command(&ckpt_dir, &prefix);
    println!("{}", ckpt_path.display());
    Ok(())
}

/// Prints the command that copies this run's output off the machine.
///
/// A rented GPU is deleted at the end of the session and takes its disk
/// with it. Two runs have already been lost that way — the weights
/// existed, nobody pulled them, the pod went away. The run itself is the
/// only place that knows the directory and the prefix, so it is the
/// place to say how to fetch them, at the moment the operator is looking
/// at the output and deciding whether to shut the machine down.
///
/// RunPod exports the SSH endpoint into the pod's environment; when it
/// is absent (a local run, or another host) the placeholders stay in
/// and the line is still a usable template.
///
/// The key is a placeholder rather than a path. Which private key opens
/// a given host is the operator's business and not something to publish
/// with the source: naming one here would put a specific key file of a
/// specific machine into a public repository, which says more about that
/// machine than a template needs to.
fn print_retrieval_command(ckpt_dir: &Path, prefix: &str) {
    let ip = env::var("RUNPOD_PUBLIC_IP").unwrap_or_else(|_| "<public-ip>".into());
    let port = env::var("RUNPOD_TCP_PORT_22").unwrap_or_else(|_| "<ssh-port>".into());
    eprintln!(
        "\n[bake] pull this before deleting the pod, then ls the local copy:\n\
         \x20 scp -i <ssh-key> -P {port} \
         'root@{ip}:{dir}/{prefix}*' <local-dir>/\n\
         \x20 ls -la <local-dir>/{prefix}*\n",
        dir = ckpt_dir.display(),
    );
}
