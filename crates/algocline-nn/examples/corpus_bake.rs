//! Corpus bake driver — trains a small GPT-2 custom model from
//! pre-tokenized corpus files and writes one safetensors bundle.
//!
//! This is the entry point for hosts that have no MCP client available
//! (a rented GPU pod, a CI runner): everything the run needs arrives
//! through the environment, and the only thing it prints on stdout is a
//! one-line JSON summary a caller can parse.
//!
//! # Corpus format
//!
//! The files are read by [`algocline_nn::train::CorpusFile`], whose module
//! documentation is the format's specification — including the opt-in
//! per-row allowed-id sets (`meta.requires: ["per_row_allowed"]`).
//! Several files are merged round-robin by
//! [`algocline_nn::train::corpus::interleave_labelled`]. Nothing about the
//! format is re-implemented here, so a file this driver accepts is a file
//! `alc.nn.data.corpus` accepts, and the other way round.
//!
//! # Inputs (environment)
//!
//! Corpora and output:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `NN_BAKE_CORPUS` | *required* | Comma-separated training corpus paths |
//! | `NN_BAKE_OUT` | *required* | Output `*.safetensors` path |
//! | `NN_BAKE_STEPS` | *required* | Optimizer steps |
//! | `NN_BAKE_COND` | unset | Condition slot per training corpus, same arity as `NN_BAKE_CORPUS` |
//! | `NN_BAKE_COND_SLOTS` | `max(cond) + 1` | Conditioning table size |
//! | `NN_BAKE_VAL_CORPUS` | unset | Comma-separated held-out corpus paths |
//! | `NN_BAKE_VAL_COND` | unset | Condition slot per held-out corpus; required exactly when `NN_BAKE_COND` is set |
//! | `NN_BAKE_MASK_LOGITS` | `0` | Score each target among the ids its position allowed |
//! | `NN_BAKE_ALLOWED_INPUT` | `0` | Hand the allowed ids to the model as an input channel |
//! | `NN_BAKE_PAD_ID` | `0` | Padding token id |
//! | `NN_BAKE_MASK_PAD` | `1` | Leave padded positions out of the loss |
//!
//! Model:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `NN_BAKE_LAYERS` | `2` | Transformer blocks |
//! | `NN_BAKE_HEADS` | `4` | Attention heads (must divide `NN_BAKE_DIM`) |
//! | `NN_BAKE_DIM` | `128` | Hidden size |
//! | `NN_BAKE_SEED` | unset | Seed for the initial weights; unset draws them unseeded |
//!
//! Optimisation:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `NN_BAKE_BATCH` | `32` | Rows per step |
//! | `NN_BAKE_LR` | `3e-3` | Peak learning rate |
//! | `NN_BAKE_SCHEDULE` | `constant` | `constant` / `cosine` / `linear` / `wsd` |
//! | `NN_BAKE_WARMUP` | `0` | Warmup steps |
//! | `NN_BAKE_MIN_LR` | `0` | Floor the decaying schedules end at |
//! | `NN_BAKE_OPTIMIZER` | `adamw` | `adamw` / `lion` |
//! | `NN_BAKE_WEIGHT_DECAY` | `0.1` | Decoupled weight decay |
//! | `NN_BAKE_CLIP_GRAD_NORM` | unset | Cap on the joint gradient L2 norm |
//! | `NN_BAKE_GRAD_CHECKPOINT` | `0` | Recompute activations instead of keeping them |
//!
//! Run control:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `NN_BAKE_EVAL_EVERY` | `0` | Score the held-out corpus every N steps; non-zero exactly when `NN_BAKE_VAL_CORPUS` is set |
//! | `NN_BAKE_EARLY_STOP_PATIENCE` | unset | Evaluations without improvement before stopping |
//! | `NN_BAKE_EARLY_STOP_MIN_DELTA` | unset | Improvement that counts; set together with the patience |
//! | `NN_BAKE_METRICS_EVERY` | `0` | Append a curve point to `<stem>-metrics.jsonl` every N steps |
//! | `NN_BAKE_CKPT_EVERY` | `0` | Rotating checkpoint every N steps |
//! | `NN_BAKE_CKPT_KEEP` | `3` | Rotating checkpoints kept |
//! | `NN_BAKE_SAVE_OPTIMIZER_STATE` | `0` | Write the optimizer state beside each checkpoint |
//! | `NN_BAKE_INIT_FROM` | unset | Checkpoint to start from; a resume when its optimizer state is there |
//!
//! Context length and vocabulary size are not settable: they come from
//! the corpus `meta`, and every corpus of one run — held-out ones
//! included — must agree on them.
//!
//! Every value that is set and cannot be read is refused by name; none
//! falls back to its default. Combinations the trainer would refuse
//! (validation without `NN_BAKE_EVAL_EVERY`, early stop without
//! validation, a warmup the schedule cannot fit) are left to the trainer
//! to refuse, so there is one statement of each rule.
//!
//! # Conditioning
//!
//! Giving `NN_BAKE_COND` switches the run to the conditioned trainer:
//! the model grows a conditioning table of `NN_BAKE_COND_SLOTS` rows and
//! every corpus row is trained under the slot its own corpus was
//! labelled with. A held-out corpus is labelled the same way, through
//! `NN_BAKE_VAL_COND`, because a conditioned model cannot be scored on a
//! row without a condition.
//!
//! # Allowed-id sets
//!
//! When the corpora carry them, at least one of two independent switches
//! has to consume them, or the run is refused — sets that are loaded and
//! then unused make a plain run wearing the label of a constrained one:
//!
//! - `NN_BAKE_MASK_LOGITS` scores each target among the ids its position
//!   allowed rather than among the whole vocabulary.
//! - `NN_BAKE_ALLOWED_INPUT` gives the model the set as an input, so it
//!   is told what is available before it answers.
//!
//! `NN_BAKE_ALLOWED_INPUT` cannot be combined with `NN_BAKE_COND`: the
//! architecture refuses a model carrying both tables, because neither
//! forward pass delivers both channels.
//!
//! # Outputs
//!
//! - `NN_BAKE_OUT` — the final weights, with the identity header every
//!   checkpoint written here carries.
//! - `<stem>-metrics.jsonl` beside it, when `NN_BAKE_METRICS_EVERY` is set.
//! - stdout — one JSON line with the run's shape and result: `steps`,
//!   `final_loss`, `min_loss`, `val_loss`, `min_val_loss`, `stopped_early`,
//!   `elapsed_s`, `out`, `metrics`, `rows`, `corpus_rows`, `val_rows`,
//!   `seed`, `schedule`, `optimizer`, `cond_slots`, `allowed_positions`,
//!   `mask_disallowed_logits`, `allowed_input`, `mask_pad`, `device`,
//!   `ctx_len`, `vocab_size`. A field the run has no value for is `null`.
//! - stderr — progress lines, plus the per-step loss when the caller
//!   sets `RUST_LOG=algocline_nn=info`.
//!
//! # Usage
//!
//! ```bash
//! # CUDA (requires nvcc + the nn-cuda feature at compile time)
//! NN_BAKE_CORPUS=/data/a.json,/data/b.json \
//! NN_BAKE_COND=0,1 \
//! NN_BAKE_VAL_CORPUS=/data/a_val.json,/data/b_val.json \
//! NN_BAKE_VAL_COND=0,1 \
//! NN_BAKE_EVAL_EVERY=50 \
//! NN_BAKE_SEED=7 \
//! NN_BAKE_STEPS=250 \
//! NN_BAKE_OUT=/data/out.safetensors \
//!   cargo run --release --features nn-cuda --example corpus_bake
//!
//! # CPU (small shapes only)
//! NN_BAKE_CORPUS=/data/a.json NN_BAKE_STEPS=2 NN_BAKE_OUT=/tmp/o.safetensors \
//!   cargo run --release --example corpus_bake
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use algocline_nn::arch::{seeded_var_builder, CondIndex, Gpt2Config, Gpt2Custom, Gpt2Model};
use algocline_nn::train::corpus::interleave_labelled;
use algocline_nn::train::{
    run_allowed_ft, run_conditioned_ft, run_full_ft, BundleIdentity, CorpusFile, CrossEntropyLoss,
    Dataset, DatasetOpts, EarlyStop, FullFtConfig, OptimizerKind, ScheduleKind, TokenizedDataset,
    TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

// ─── Environment ────────────────────────────────────────────────

/// Read `key`, treating "unset" and "set to whitespace" alike.
///
/// A value that is not UTF-8 is an error rather than an absence: the
/// caller meant to say something and this driver cannot read it.
fn env_str(key: &str) -> Result<Option<String>, String> {
    match std::env::var(key) {
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{key} is not valid UTF-8")),
    }
}

fn env_required(key: &str, what: &str) -> Result<String, String> {
    env_str(key)?.ok_or_else(|| format!("{key} is required ({what})"))
}

/// Parse an optional value of any `FromStr` type. A value that is
/// present but unparseable is refused rather than replaced by a default
/// — a typo'd step count that silently becomes the default is a run
/// nobody can interpret.
fn env_parse<T>(key: &str, what: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env_str(key)? {
        None => Ok(None),
        Some(v) => v
            .trim()
            .parse::<T>()
            .map(Some)
            .map_err(|e| format!("{key}={v:?} is not {what} ({e})")),
    }
}

fn env_usize(key: &str, default: usize) -> Result<usize, String> {
    Ok(env_parse::<usize>(key, "a non-negative integer")?.unwrap_or(default))
}

fn env_required_usize(key: &str, what: &str) -> Result<usize, String> {
    env_parse::<usize>(key, "a non-negative integer")?
        .ok_or_else(|| format!("{key} is required ({what})"))
}

/// Parse an optional finite number. `NaN` and the infinities parse as
/// `f64` and mean nothing as a learning rate or a clip threshold.
fn env_f64_opt(key: &str) -> Result<Option<f64>, String> {
    match env_parse::<f64>(key, "a number")? {
        Some(v) if !v.is_finite() => Err(format!("{key}={v} is not a finite number")),
        other => Ok(other),
    }
}

fn env_f64(key: &str, default: f64) -> Result<f64, String> {
    Ok(env_f64_opt(key)?.unwrap_or(default))
}

/// Parse an optional switch.
///
/// A value that is present but not one of the accepted spellings is
/// refused rather than read as `false`: a misspelled switch that
/// silently means "off" produces a run answering a different question
/// than the caller asked, with every number well-formed.
fn env_bool(key: &str, default: bool) -> Result<bool, String> {
    match env_str(key)? {
        None => Ok(default),
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!(
                "{key}={v:?} is not a switch (1/0, true/false, yes/no, on/off)"
            )),
        },
    }
}

/// Split a comma-separated list, dropping surrounding whitespace and
/// refusing empty entries (a trailing comma is a list one item shorter
/// than the caller thinks it is).
fn split_list(key: &str, raw: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (idx, part) in raw.split(',').enumerate() {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!(
                "{key}={raw:?} has an empty entry at position {idx} — the list is \
                 comma-separated with no empty members"
            ));
        }
        out.push(part.to_string());
    }
    Ok(out)
}

/// Parse a condition list into one slot id per corpus path.
///
/// The two lists are paired positionally, so a length disagreement
/// means some corpus would be trained under another corpus's condition
/// with every shape still agreeing.
fn parse_conds(
    key: &str,
    raw: &str,
    corpus_key: &str,
    corpus_count: usize,
) -> Result<Vec<u32>, String> {
    let parts = split_list(key, raw)?;
    if parts.len() != corpus_count {
        return Err(format!(
            "{key} names {} condition(s) for the {corpus_count} corpus path(s) in \
             {corpus_key} — the pairing is positional, so it takes exactly one per corpus",
            parts.len()
        ));
    }
    parts
        .iter()
        .map(|part| {
            part.parse::<u32>()
                .map_err(|e| format!("{key} entry {part:?} is not a slot id ({e})"))
        })
        .collect()
}

// ─── Datasets ───────────────────────────────────────────────────

/// A set of corpora merged into one dataset, plus what the summary
/// reports about it.
struct Built {
    dataset: TokenizedDataset,
    /// Rows in the merged corpora, before any cycling.
    corpus_rows: usize,
    /// Rows the dataset holds, after cycling.
    rows: usize,
    ctx_len: usize,
    vocab_size: usize,
    /// Common width of the allowed-id sets, or `None` when the corpora
    /// carry none.
    allowed_width: Option<usize>,
}

/// What [`build_dataset`] is asked to build.
struct BuildSpec<'a> {
    /// The variable the paths came from, for error messages.
    key: &'a str,
    paths: &'a [String],
    /// One slot per corpus, and the table size the slots index.
    conds: Option<(&'a [u32], usize)>,
    /// Rows the dataset has to hold. The dataset is one-pass, so a
    /// training set is cycled — repeating the merged order — until it
    /// covers the run; a held-out set passes `0` and is read once.
    min_rows: usize,
    /// Common width to build the allowed-id sets at, so a held-out set
    /// matches the training set's. `None` takes the corpora's own.
    width: Option<usize>,
    opts: DatasetOpts,
}

/// Load the corpora, merge them round-robin, and build one dataset.
///
/// The corpora leave each row's allowed-id sets at the row's own last
/// listed position, and both readers of the sets take one width for the
/// whole batch. Widening with the empty set says nothing new: a position
/// nobody listed was already unconstrained.
fn build_dataset(spec: BuildSpec<'_>) -> Result<Built, String> {
    let key = spec.key;
    let corpora = spec
        .paths
        .iter()
        .map(|p| CorpusFile::load(Path::new(p)).map_err(|e| format!("{key}: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let sources: Vec<&CorpusFile> = corpora.iter().collect();
    let merged = interleave_labelled(&sources).map_err(|e| format!("{key}: {e}"))?;
    for corpus in &corpora {
        eprintln!(
            "[bake] {key} {:?}: rows={} allowed={}",
            corpus.path(),
            corpus.rows().len(),
            corpus.allowed().is_some()
        );
    }

    // Conditions are resolved before any row is built, so a slot outside
    // the table is refused at the variable that named it.
    let per_source: Option<Vec<CondIndex>> = match spec.conds {
        None => None,
        Some((slots, table)) => Some(
            slots
                .iter()
                .map(|&slot| {
                    CondIndex::new(slot, table).map_err(|e| {
                        format!("{key}: condition slot {slot} is not a row of a {table}-slot table ({e})")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };

    let own_width = merged
        .iter()
        .filter_map(|row| row.allowed.as_ref())
        .map(Vec::len)
        .max();
    let allowed_width = match (own_width, spec.width) {
        (None, _) => None,
        (Some(own), Some(forced)) if own > forced => {
            return Err(format!(
                "{key}: the allowed-id sets reach position {own}, past the {forced} the \
                 training corpora reach — the model's allowed-id input is built at the \
                 training width"
            ))
        }
        (Some(_), Some(forced)) => Some(forced),
        (Some(0), None) => {
            return Err(format!(
                "{key}: the corpora carry \"allowed\" sets that constrain no position of any \
                 row — a producer with nothing to say about the run cannot say it here"
            ))
        }
        (Some(own), None) => Some(own),
    };

    let corpus_rows = merged.len();
    let total = spec.min_rows.max(corpus_rows);
    let mut rows = Vec::with_capacity(total);
    let mut conds = per_source.as_ref().map(|_| Vec::with_capacity(total));
    let mut allowed = allowed_width.map(|_| Vec::with_capacity(total));
    for i in 0..total {
        let row = &merged[i % corpus_rows];
        rows.push(row.ids.clone());
        // Both side channels are pushed at the same index as the row, in
        // the same iteration, so the cycling cannot shift one against
        // the other.
        if let (Some(conds), Some(per_source)) = (conds.as_mut(), per_source.as_ref()) {
            conds.push(per_source[row.source]);
        }
        if let (Some(allowed), Some(width)) = (allowed.as_mut(), allowed_width) {
            let mut sets = row.allowed.clone().unwrap_or_default();
            sets.resize(width, Vec::new());
            allowed.push(sets);
        }
    }

    // Every file agreed on the shape — the merge refused the call
    // otherwise — so the first one speaks for all of them.
    let ctx_len = corpora[0].ctx_len();
    let vocab_size = corpora[0].vocab_size();
    let mut opts = spec.opts;
    opts.ctx_len = ctx_len;
    let mut dataset = TokenizedDataset::new(rows, opts);
    if let Some(conds) = conds {
        dataset = dataset
            .with_conditions(conds)
            .map_err(|e| format!("{key}: attaching the per-row conditions failed: {e}"))?;
    }
    if let Some(allowed) = allowed {
        // This is where a set that excludes the token its own position
        // holds is refused: the loss could only answer it with a number.
        dataset = dataset
            .with_allowed_ids(allowed)
            .map_err(|e| format!("{key}: attaching the per-row allowed-id sets failed: {e}"))?;
    }
    Ok(Built {
        dataset,
        corpus_rows,
        rows: total,
        ctx_len,
        vocab_size,
        allowed_width,
    })
}

// ─── Device ─────────────────────────────────────────────────────

/// CUDA when the build has it and the host offers it, CPU otherwise.
/// Which one was taken lands in the summary, so a run that quietly fell
/// back is visible to whoever reads the output rather than only to
/// whoever watched stderr.
fn resolve_device() -> (Device, &'static str) {
    #[cfg(feature = "nn-cuda")]
    {
        match Device::new_cuda(0) {
            Ok(dev) => {
                eprintln!("[bake] using CUDA device 0");
                return (dev, "cuda");
            }
            Err(e) => {
                eprintln!("[bake] cuda unavailable ({e}); falling back to CPU");
            }
        }
    }
    eprintln!("[bake] using CPU device");
    (Device::Cpu, "cpu")
}

// ─── Driver ─────────────────────────────────────────────────────

fn run() -> Result<(), String> {
    let corpus_paths = split_list(
        "NN_BAKE_CORPUS",
        &env_required("NN_BAKE_CORPUS", "comma-separated corpus JSON paths")?,
    )?;
    let val_paths = env_str("NN_BAKE_VAL_CORPUS")?
        .map(|raw| split_list("NN_BAKE_VAL_CORPUS", &raw))
        .transpose()?;
    let steps = env_required_usize("NN_BAKE_STEPS", "number of optimizer steps")?;
    if steps == 0 {
        return Err("NN_BAKE_STEPS=0 trains nothing".to_string());
    }
    let batch = env_usize("NN_BAKE_BATCH", 32)?;
    if batch == 0 {
        return Err("NN_BAKE_BATCH=0 leaves every step without rows".to_string());
    }
    let layers = env_usize("NN_BAKE_LAYERS", 2)?;
    let heads = env_usize("NN_BAKE_HEADS", 4)?;
    let dim = env_usize("NN_BAKE_DIM", 128)?;
    if layers == 0 || heads == 0 || dim == 0 {
        return Err(format!(
            "NN_BAKE_LAYERS={layers} / NN_BAKE_HEADS={heads} / NN_BAKE_DIM={dim} — \
             each has to be at least 1"
        ));
    }
    if dim % heads != 0 {
        return Err(format!(
            "NN_BAKE_DIM={dim} is not divisible by NN_BAKE_HEADS={heads}"
        ));
    }
    let seed = env_parse::<u64>("NN_BAKE_SEED", "a non-negative integer")?;
    let pad_id = env_parse::<u32>("NN_BAKE_PAD_ID", "a non-negative integer")?.unwrap_or(0);
    let mask_pad = env_bool("NN_BAKE_MASK_PAD", true)?;

    let schedule_raw = env_str("NN_BAKE_SCHEDULE")?.unwrap_or_else(|| "constant".into());
    let schedule = ScheduleKind::parse(schedule_raw.trim()).ok_or_else(|| {
        format!(
            "NN_BAKE_SCHEDULE={schedule_raw:?} is not a schedule (one of {})",
            ScheduleKind::NAMES.join(", ")
        )
    })?;
    let optimizer_raw = env_str("NN_BAKE_OPTIMIZER")?.unwrap_or_else(|| "adamw".into());
    let optimizer = OptimizerKind::parse(optimizer_raw.trim()).ok_or_else(|| {
        format!(
            "NN_BAKE_OPTIMIZER={optimizer_raw:?} is not an optimizer (one of {})",
            OptimizerKind::NAMES.join(", ")
        )
    })?;

    let early_stop = match (
        env_parse::<usize>("NN_BAKE_EARLY_STOP_PATIENCE", "a non-negative integer")?,
        env_parse::<f32>("NN_BAKE_EARLY_STOP_MIN_DELTA", "a number")?,
    ) {
        (None, None) => None,
        (Some(patience), Some(min_delta)) => Some(EarlyStop {
            patience,
            min_delta,
        }),
        // Neither half has a default: `patience` alone stops on noise,
        // `min_delta` alone never stops a run creeping down by nothing.
        (Some(_), None) => {
            return Err("NN_BAKE_EARLY_STOP_PATIENCE was set without \
                 NN_BAKE_EARLY_STOP_MIN_DELTA — the rule needs both"
                .to_string())
        }
        (None, Some(_)) => {
            return Err("NN_BAKE_EARLY_STOP_MIN_DELTA was set without \
                 NN_BAKE_EARLY_STOP_PATIENCE — the rule needs both"
                .to_string())
        }
    };

    // The two independent uses of the allowed-id sets: as a mask on the
    // loss, and as an input to the model. Either, both, or neither.
    let mask_logits = env_bool("NN_BAKE_MASK_LOGITS", false)?;
    let allowed_input = env_bool("NN_BAKE_ALLOWED_INPUT", false)?;

    // Conditions, when the caller named any.
    let cond_raw = env_str("NN_BAKE_COND")?;
    let val_cond_raw = env_str("NN_BAKE_VAL_COND")?;
    if cond_raw.is_none() && env_str("NN_BAKE_COND_SLOTS")?.is_some() {
        return Err(
            "NN_BAKE_COND_SLOTS was set without NN_BAKE_COND — a conditioning table \
             with nothing selecting a row from it trains nothing"
                .to_string(),
        );
    }
    if allowed_input && cond_raw.is_some() {
        return Err(
            "NN_BAKE_ALLOWED_INPUT was set together with NN_BAKE_COND — the architecture \
             rejects `cond_slots` together with `allowed_input`, because neither forward \
             pass delivers both channels; pick one"
                .to_string(),
        );
    }
    let conds: Option<Vec<u32>> = cond_raw
        .as_deref()
        .map(|raw| parse_conds("NN_BAKE_COND", raw, "NN_BAKE_CORPUS", corpus_paths.len()))
        .transpose()?;
    let val_conds: Option<Vec<u32>> = match (&conds, &val_paths, val_cond_raw.as_deref()) {
        (_, None, Some(_)) => {
            return Err("NN_BAKE_VAL_COND was set without NN_BAKE_VAL_CORPUS".to_string())
        }
        (None, Some(_), Some(_)) => {
            return Err(
                "NN_BAKE_VAL_COND was set for an unconditioned run (no NN_BAKE_COND) — the \
                 model has no table to select a row from"
                    .to_string(),
            )
        }
        (Some(_), Some(_), None) => {
            return Err(
                "NN_BAKE_VAL_CORPUS was set for a conditioned run without NN_BAKE_VAL_COND — \
                 a conditioned model cannot be scored on a row that names no condition"
                    .to_string(),
            )
        }
        (Some(_), Some(paths), Some(raw)) => Some(parse_conds(
            "NN_BAKE_VAL_COND",
            raw,
            "NN_BAKE_VAL_CORPUS",
            paths.len(),
        )?),
        _ => None,
    };
    let cond_slots: Option<usize> = match &conds {
        None => None,
        Some(slots) => {
            let implied = slots
                .iter()
                .chain(val_conds.iter().flatten())
                .copied()
                .max()
                .unwrap_or(0) as usize
                + 1;
            match env_usize("NN_BAKE_COND_SLOTS", implied)? {
                n if n < implied => {
                    return Err(format!(
                        "NN_BAKE_COND_SLOTS={n} is smaller than the {implied} slot(s) \
                         NN_BAKE_COND / NN_BAKE_VAL_COND select"
                    ))
                }
                n => Some(n),
            }
        }
    };

    // Output path. The trainer writes `<prefix>.safetensors` into a
    // directory, so a caller naming anything else would find their
    // weights beside the path they asked for rather than at it.
    let out_raw = env_required("NN_BAKE_OUT", "output *.safetensors path")?;
    let out = PathBuf::from(&out_raw);
    if out.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(format!(
            "NN_BAKE_OUT={out_raw:?} does not end in .safetensors — that is the \
             format the bundle is written in"
        ));
    }
    let out_dir = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let out_prefix = out
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("NN_BAKE_OUT={out_raw:?} has no file name"))?
        .to_string();
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("NN_BAKE_OUT={out_raw:?}: could not create {out_dir:?}: {e}"))?;

    let dataset_opts = DatasetOpts {
        batch_size: batch,
        // Off: the rows are in the rotation the merge put them in, and a
        // positional condition list cannot survive a re-order.
        shuffle: false,
        pad_id,
        mask_pad,
        ..DatasetOpts::default()
    };
    // `steps` batches of `batch`, plus one batch of margin.
    let needed = steps.saturating_mul(batch).saturating_add(batch);
    let train = build_dataset(BuildSpec {
        key: "NN_BAKE_CORPUS",
        paths: &corpus_paths,
        conds: conds.as_deref().zip(cond_slots),
        min_rows: needed,
        width: None,
        opts: dataset_opts.clone(),
    })?;
    let (ctx_len, vocab_size) = (train.ctx_len, train.vocab_size);
    let mut val = match &val_paths {
        None => None,
        Some(paths) => {
            let built = build_dataset(BuildSpec {
                key: "NN_BAKE_VAL_CORPUS",
                paths,
                conds: val_conds.as_deref().zip(cond_slots),
                min_rows: 0,
                width: train.allowed_width,
                opts: dataset_opts,
            })?;
            if (built.ctx_len, built.vocab_size) != (ctx_len, vocab_size) {
                return Err(format!(
                    "NN_BAKE_VAL_CORPUS declares ctx_len {} / vocab_size {} and NN_BAKE_CORPUS \
                     declares ctx_len {ctx_len} / vocab_size {vocab_size} — one model cannot be \
                     scored on both",
                    built.ctx_len, built.vocab_size
                ));
            }
            if built.allowed_width.is_some() != train.allowed_width.is_some() {
                return Err(
                    "NN_BAKE_VAL_CORPUS and NN_BAKE_CORPUS disagree about carrying allowed-id \
                     sets — a held-out set scored without the constraint the run trained under \
                     measures a different loss"
                        .to_string(),
                );
            }
            Some(built)
        }
    };

    // The pad id is the one id that reaches training without coming out
    // of a corpus, and it reaches almost every row. Left unchecked, one
    // inside the model's vocabulary but outside the corpus's trains the
    // model to emit an id the corpus declares does not exist.
    if pad_id as usize >= vocab_size {
        return Err(format!(
            "NN_BAKE_PAD_ID={pad_id} is at or past the meta.vocab_size {vocab_size} the \
             corpora declare"
        ));
    }

    let asked_for_sets = match (mask_logits, allowed_input) {
        (true, true) => Some("NN_BAKE_MASK_LOGITS and NN_BAKE_ALLOWED_INPUT"),
        (true, false) => Some("NN_BAKE_MASK_LOGITS"),
        (false, true) => Some("NN_BAKE_ALLOWED_INPUT"),
        (false, false) => None,
    };
    match (train.allowed_width.is_some(), asked_for_sets) {
        (true, None) => {
            return Err(
                "the NN_BAKE_CORPUS entries carry per-row allowed-id sets and neither \
                 NN_BAKE_MASK_LOGITS nor NN_BAKE_ALLOWED_INPUT is set — the sets would be \
                 read and then used for nothing, leaving a plain run under the name of a \
                 constrained one"
                    .to_string(),
            )
        }
        (false, Some(flags)) => {
            return Err(format!(
                "{flags} asks for the per-row allowed-id sets, and no NN_BAKE_CORPUS entry \
                 carries them — a corpus states them as a top-level \"allowed\" array \
                 announced by meta.requires [\"per_row_allowed\"]"
            ))
        }
        _ => {}
    }

    let (device, device_name) = resolve_device();
    let cfg = Gpt2Config {
        layers,
        heads,
        dim,
        ctx: ctx_len,
        vocab: vocab_size,
        dtype: DType::F32,
        device,
        eps: 1e-5,
        moe: None,
        custom: if cond_slots.is_some() || allowed_input {
            Some(Gpt2Custom {
                cond_slots,
                allowed_input,
                ..Gpt2Custom::default()
            })
        } else {
            None
        },
    };
    eprintln!(
        "[bake] model layers={layers} heads={heads} dim={dim} ctx={ctx_len} \
         vocab={vocab_size} cond_slots={cond_slots:?} allowed_input={allowed_input} \
         mask_logits={mask_logits} seed={seed:?} device={device_name}"
    );
    let vm = VarMap::new();
    let vb = match seed {
        Some(seed) => seeded_var_builder(&vm, seed, cfg.dtype, &cfg.device),
        None => VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device),
    };
    let model = Gpt2Model::new(&cfg, vb).map_err(|e| format!("building the model failed: {e}"))?;

    let defaults = FullFtConfig::default();
    let ft_cfg = FullFtConfig {
        lr: env_f64("NN_BAKE_LR", 3e-3)?,
        batch_size: batch,
        steps,
        warmup: env_usize("NN_BAKE_WARMUP", 0)?,
        schedule,
        min_lr: env_f64("NN_BAKE_MIN_LR", defaults.min_lr)?,
        optimizer,
        weight_decay: env_f64("NN_BAKE_WEIGHT_DECAY", defaults.weight_decay)?,
        clip_grad_norm: env_f64_opt("NN_BAKE_CLIP_GRAD_NORM")?,
        grad_checkpoint: env_bool("NN_BAKE_GRAD_CHECKPOINT", false)?,
        eval_every: env_usize("NN_BAKE_EVAL_EVERY", 0)?,
        early_stop,
        metrics_every: env_usize("NN_BAKE_METRICS_EVERY", 0)?,
        ckpt_every: env_usize("NN_BAKE_CKPT_EVERY", 0)?,
        ckpt_keep: env_usize("NN_BAKE_CKPT_KEEP", defaults.ckpt_keep)?,
        save_optimizer_state: env_bool("NN_BAKE_SAVE_OPTIMIZER_STATE", false)?,
        init_from: env_str("NN_BAKE_INIT_FROM")?.map(PathBuf::from),
        mask_disallowed_logits: mask_logits,
        bundle_identity: Some(BundleIdentity {
            architecture: "gpt2-custom".into(),
            vocab: vocab_size,
            ctx: ctx_len,
            dtype: "f32".into(),
            // No Card is written by this driver.
            card_id: None,
        }),
        ..defaults
    };
    let lease = Arc::new(TrainingLease::new());
    let loss = CrossEntropyLoss::new();

    eprintln!(
        "[bake] training steps={steps} batch={batch} lr={} schedule={schedule_raw} \
         optimizer={optimizer_raw} rows={} (corpus rows {}) val_rows={:?}",
        ft_cfg.lr,
        train.rows,
        train.corpus_rows,
        val.as_ref().map(|v| v.rows)
    );
    let val_rows = val.as_ref().map(|v| v.rows);
    let mut dataset = train.dataset;
    let val_ds = val.as_mut().map(|v| &mut v.dataset as &mut dyn Dataset);
    let t0 = Instant::now();
    // Which entry point the run takes is which channel the model was
    // built with: the allowed-id input needs the forward pass that
    // carries it, and conditioning needs the one that carries a slot.
    // `mask_disallowed_logits` is orthogonal to both — it acts on the
    // loss inside whichever loop runs.
    let ckpt = if allowed_input {
        run_allowed_ft(
            &model,
            &vm,
            &mut dataset,
            val_ds,
            &ft_cfg,
            &loss,
            &out_dir,
            &out_prefix,
            lease,
            None,
        )
    } else if cond_slots.is_some() {
        run_conditioned_ft(
            &model,
            &vm,
            &mut dataset,
            val_ds,
            &ft_cfg,
            &loss,
            &out_dir,
            &out_prefix,
            lease,
            None,
        )
    } else {
        run_full_ft(
            &model,
            &vm,
            &mut dataset,
            val_ds,
            &ft_cfg,
            &loss,
            &out_dir,
            &out_prefix,
            lease,
            None,
        )
    }
    .map_err(|e| format!("training failed: {e}"))?;
    let elapsed = t0.elapsed();

    let bundle = out_dir.join(&ckpt.bundle_ref);
    if !bundle.exists() {
        return Err(format!(
            "training reported {bundle:?} but no such file exists"
        ));
    }
    let metrics_path = out_dir.join(format!("{out_prefix}-metrics.jsonl"));

    let summary = serde_json::json!({
        "steps": ckpt.step,
        "final_loss": ckpt.train_loss,
        "min_loss": ckpt.metrics.get("min_train_loss"),
        "val_loss": ckpt.val_loss,
        "min_val_loss": ckpt.metrics.get("min_val_loss"),
        "stopped_early": ckpt.metrics.contains_key("early_stop"),
        "elapsed_s": elapsed.as_secs_f64(),
        "out": bundle.to_string_lossy(),
        "metrics": metrics_path.exists().then(|| metrics_path.to_string_lossy().into_owned()),
        "rows": train.rows,
        "corpus_rows": train.corpus_rows,
        "val_rows": val_rows,
        "seed": seed,
        "schedule": schedule_raw.trim(),
        "optimizer": optimizer_raw.trim(),
        "cond_slots": cond_slots,
        "allowed_positions": train.allowed_width,
        "mask_disallowed_logits": mask_logits,
        "allowed_input": allowed_input,
        "mask_pad": mask_pad,
        "device": device_name,
        "ctx_len": ctx_len,
        "vocab_size": vocab_size,
    });
    println!("{summary}");
    Ok(())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    if let Err(e) = run() {
        eprintln!("corpus_bake: {e}");
        std::process::exit(1);
    }
}
